/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::approx_constant, internal_features)]
#![feature(core_intrinsics)]

//! Unified Atomics Test Example
//!
//! Comprehensive test suite for sound atomic operations:
//!
//! **Phase 1 (DeviceAtomicU32/I32, load/store/fetch_add/CAS):**
//!  1. `atomic_fetch_add_test` -- DeviceAtomicU32 fetch_add (Relaxed)
//!  2. `atomic_load_store_test` -- DeviceAtomicU32 load/store (Acquire/Release)
//!  3. `atomic_cas_test` -- DeviceAtomicU32 compare_exchange (AcqRel)
//!  4. `atomic_fetch_add_acqrel_test` -- fence-splitting workaround (AcqRel)
//!  5. `atomic_fetch_add_seqcst_test` -- fence.sc pattern (SeqCst)
//!  6. `atomic_i32_test` -- DeviceAtomicI32 fetch_add + compare_exchange
//!  7. `atomic_multiblock_test` -- device-scope atomics across CTAs
//!
//! **Phase 2 (new types + new RMW ops):**
//!  8. `atomic_u64_fetch_add_test` -- DeviceAtomicU64 fetch_add (64-bit)
//!  9. `atomic_i64_test` -- DeviceAtomicI64 fetch_add + compare_exchange
//! 10. `atomic_fetch_sub_test` -- DeviceAtomicU32 fetch_sub
//! 11. `atomic_bitwise_test` -- fetch_and, fetch_or, fetch_xor
//! 12. `atomic_swap_test` -- DeviceAtomicU32 swap (exchange)
//! 13. `atomic_minmax_test` -- DeviceAtomicI32 fetch_min / fetch_max (signed)
//! 14. `atomic_f32_fetch_add_test` -- DeviceAtomicF32 fetch_add (float)
//! 15. `atomic_f64_fetch_add_test` -- DeviceAtomicF64 fetch_add (64-bit float)
//! 16. `atomic_f32_swap_test` -- DeviceAtomicF32 swap (float exchange)
//! 17. `atomic_unsigned_minmax_test` -- DeviceAtomicU32 fetch_min/max (UMin/UMax)
//! 18. `atomic_block_scope_test` -- BlockAtomicU32 fetch_add (.cta scope, Relaxed)
//! 19. `atomic_block_scope_acqrel_test` -- BlockAtomicU32 fetch_add (.cta scope, AcqRel)
//! 20. `core_atomic_fetch_add_test` -- core::sync::atomic::AtomicU32 (system scope)
//! 21. `core_atomic_ordering_probe` -- compile-only core intrinsic ordering coverage
//!
//! Build and run with:
//!   cargo oxide run atomics

use core::sync::atomic::Ordering;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::atomic::{
    AtomicOrdering, BlockAtomicU32, DeviceAtomicF32, DeviceAtomicF64, DeviceAtomicI32,
    DeviceAtomicI64, DeviceAtomicU32, DeviceAtomicU64,
};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Test 1: Atomic fetch_add -- every thread atomically increments a counter.
    ///
    /// After N threads run, counter[0] should equal N.
    /// This tests the atomicrmw path with the fence-splitting workaround.
    #[kernel]
    pub fn atomic_fetch_add_test(counter: &[u32], mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();

        // Get an DeviceAtomicU32 reference to counter[0] (shared access via interior mutability)
        let atomic_counter = unsafe { &*(counter.as_ptr() as *const DeviceAtomicU32) };

        // Each thread atomically increments the counter and gets the old value
        let old = atomic_counter.fetch_add(1, AtomicOrdering::Relaxed);

        // Store the old value so we can verify uniqueness on the host
        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = old;
        }
    }

    /// Test 2: Atomic load/store -- thread 0 stores a value, all threads load it.
    ///
    /// Uses Acquire/Release ordering for proper visibility:
    /// - Thread 0: store with Release (makes the write visible)
    /// - Other threads: load with Acquire (sees the Release'd write)
    #[kernel]
    pub fn atomic_load_store_test(flag: &[u32], mut out: DisjointSlice<u32>) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        // Get an DeviceAtomicU32 reference to flag[0] (shared access via interior mutability)
        let atomic_flag = unsafe { &*(flag.as_ptr() as *const DeviceAtomicU32) };

        // Thread 0 stores a sentinel value
        if tid == 0 {
            atomic_flag.store(42, AtomicOrdering::Release);
        }

        // Barrier ensures all threads see the store
        thread::sync_threads();

        // All threads load the value
        let val = atomic_flag.load(AtomicOrdering::Acquire);
        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = val;
        }
    }

    /// Test 3: Atomic compare_exchange -- only one thread wins the CAS race.
    ///
    /// All threads try to CAS 0 -> their_tid. Exactly one succeeds.
    #[kernel]
    pub fn atomic_cas_test(winner: &[u32], mut out: DisjointSlice<u32>) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        // Get an DeviceAtomicU32 reference to winner[0] (shared access via interior mutability)
        let atomic_winner = unsafe { &*(winner.as_ptr() as *const DeviceAtomicU32) };

        // Try to be the first thread to swap 0 -> (tid + 1)
        // We use tid+1 so that thread 0's success value (1) differs from the
        // initial value (0).
        let result = atomic_winner.compare_exchange(
            0,
            tid + 1,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Relaxed,
        );

        if let Some(out_elem) = out.get_mut(gid) {
            match result {
                Ok(_old) => {
                    // This thread won the race
                    *out_elem = 1;
                }
                Err(_old) => {
                    // Another thread already swapped
                    *out_elem = 0;
                }
            }
        }
    }

    /// Test 4: Atomic fetch_add with AcqRel ordering -- exercises fence-splitting workaround.
    ///
    /// The LLVM NVPTX backend drops orderings on atomicrmw (fix in LLVM 23).
    /// We work around this by emitting: fence release + atomicrmw monotonic + fence acquire.
    /// This test verifies that path produces correct results.
    #[kernel]
    pub fn atomic_fetch_add_acqrel_test(counter: &[u32], mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();

        let atomic_counter = unsafe { &*(counter.as_ptr() as *const DeviceAtomicU32) };

        // AcqRel triggers: fence release + atomicrmw monotonic + fence acquire
        let old = atomic_counter.fetch_add(1, AtomicOrdering::AcqRel);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = old;
        }
    }

    /// Test 5: Atomic fetch_add with SeqCst ordering -- exercises fence.sc pattern.
    ///
    /// SeqCst emits: fence seq_cst + atomicrmw monotonic + fence seq_cst.
    #[kernel]
    pub fn atomic_fetch_add_seqcst_test(counter: &[u32], mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();

        let atomic_counter = unsafe { &*(counter.as_ptr() as *const DeviceAtomicU32) };

        // SeqCst triggers: fence seq_cst + atomicrmw monotonic + fence seq_cst
        let old = atomic_counter.fetch_add(1, AtomicOrdering::SeqCst);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = old;
        }
    }

    /// Test 6: DeviceAtomicI32 -- exercises signed atomic type with fetch_add and compare_exchange.
    ///
    /// Verifies that signed atomics work correctly (i32 vs u32 in LLVM IR).
    #[kernel]
    pub fn atomic_i32_test(counter: &[i32], cas_target: &[i32], mut out: DisjointSlice<i32>) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_counter = unsafe { &*(counter.as_ptr() as *const DeviceAtomicI32) };
        let atomic_cas = unsafe { &*(cas_target.as_ptr() as *const DeviceAtomicI32) };

        // All threads increment the counter
        let _old = atomic_counter.fetch_add(1, AtomicOrdering::Relaxed);

        // Thread 0 does a CAS: swap 0 -> -42
        if let Some(out_elem) = out.get_mut(gid) {
            if tid == 0 {
                let result = atomic_cas.compare_exchange(
                    0,
                    -42,
                    AtomicOrdering::AcqRel,
                    AtomicOrdering::Relaxed,
                );
                match result {
                    Ok(_) => *out_elem = 1,
                    Err(_) => *out_elem = 0,
                }
            } else {
                // Other threads just read the CAS target after a barrier
                thread::sync_threads();
                let val = atomic_cas.load(AtomicOrdering::Acquire);
                *out_elem = val;
            }
        }
    }

    /// Test 7: Multi-block fetch_add -- exercises device-scope atomics across CTAs.
    ///
    /// Uses 4 blocks x 64 threads = 256 threads total. The counter must reach 256,
    /// proving that atomics work across different thread blocks (CTAs).
    #[kernel]
    pub fn atomic_multiblock_test(counter: &[u32], mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();

        let atomic_counter = unsafe { &*(counter.as_ptr() as *const DeviceAtomicU32) };

        let old = atomic_counter.fetch_add(1, AtomicOrdering::Relaxed);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = old;
        }
    }

    /// Test 8: DeviceAtomicU64 fetch_add -- 64-bit unsigned atomics.
    ///
    /// Same pattern as test 1 but with u64 to verify 64-bit type plumbing.
    #[kernel]
    pub fn atomic_u64_fetch_add_test(counter: &[u64], mut out: DisjointSlice<u64>) {
        let gid = thread::index_1d();

        let atomic_counter = unsafe { &*(counter.as_ptr() as *const DeviceAtomicU64) };

        let old = atomic_counter.fetch_add(1, AtomicOrdering::Relaxed);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = old;
        }
    }

    /// Test 9: DeviceAtomicI64 fetch_add + compare_exchange -- 64-bit signed atomics.
    ///
    /// Verifies i64 path: fetch_add increments, CAS swaps 0 -> -100.
    #[kernel]
    pub fn atomic_i64_test(counter: &[i64], cas_target: &[i64], mut out: DisjointSlice<i64>) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_counter = unsafe { &*(counter.as_ptr() as *const DeviceAtomicI64) };
        let atomic_cas = unsafe { &*(cas_target.as_ptr() as *const DeviceAtomicI64) };

        // All threads increment the counter
        let _old = atomic_counter.fetch_add(1, AtomicOrdering::Relaxed);

        // Thread 0 does a CAS: swap 0 -> -100
        if let Some(out_elem) = out.get_mut(gid) {
            if tid == 0 {
                let result = atomic_cas.compare_exchange(
                    0,
                    -100,
                    AtomicOrdering::AcqRel,
                    AtomicOrdering::Relaxed,
                );
                match result {
                    Ok(_) => *out_elem = 1,
                    Err(_) => *out_elem = 0,
                }
            } else {
                thread::sync_threads();
                let val = atomic_cas.load(AtomicOrdering::Acquire);
                *out_elem = val;
            }
        }
    }

    /// Test 10: DeviceAtomicU32 fetch_sub -- subtraction RMW op.
    ///
    /// Start counter at N, each thread subtracts 1. Result should be 0.
    #[kernel]
    pub fn atomic_fetch_sub_test(counter: &[u32], mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();

        let atomic_counter = unsafe { &*(counter.as_ptr() as *const DeviceAtomicU32) };

        let old = atomic_counter.fetch_sub(1, AtomicOrdering::Relaxed);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = old;
        }
    }

    /// Test 11: DeviceAtomicU32 bitwise ops -- fetch_and, fetch_or, fetch_xor.
    ///
    /// Three separate counters:
    /// - or_acc: starts at 0, each thread ORs in (1 << (tid % 32))
    /// - and_acc: starts at 0xFFFFFFFF, thread 0 ANDs with 0x0000FFFF
    /// - xor_acc: starts at 0, each thread XORs with 1 (odd/even toggle)
    #[kernel]
    pub fn atomic_bitwise_test(
        or_acc: &[u32],
        and_acc: &[u32],
        xor_acc: &[u32],
        mut out: DisjointSlice<u32>,
    ) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_or = unsafe { &*(or_acc.as_ptr() as *const DeviceAtomicU32) };
        let atomic_and = unsafe { &*(and_acc.as_ptr() as *const DeviceAtomicU32) };
        let atomic_xor = unsafe { &*(xor_acc.as_ptr() as *const DeviceAtomicU32) };

        // Each thread sets its bit in the OR accumulator
        let bit = 1u32 << (tid % 32);
        atomic_or.fetch_or(bit, AtomicOrdering::Relaxed);

        // Thread 0 masks out the upper 16 bits via AND
        if tid == 0 {
            atomic_and.fetch_and(0x0000FFFF, AtomicOrdering::Relaxed);
        }

        // Every thread XORs with 1 (toggles bit 0)
        atomic_xor.fetch_xor(1, AtomicOrdering::Relaxed);

        // Store tid so we can verify the kernel ran
        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = tid;
        }
    }

    /// Test 12: DeviceAtomicU32 swap -- atomic exchange.
    ///
    /// Thread 0 swaps in a sentinel value (0xDEADBEEF), gets back the old value (0).
    #[kernel]
    pub fn atomic_swap_test(target: &[u32], mut out: DisjointSlice<u32>) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_target = unsafe { &*(target.as_ptr() as *const DeviceAtomicU32) };

        if let Some(out_elem) = out.get_mut(gid) {
            if tid == 0 {
                let old = atomic_target.swap(0xDEADBEEF, AtomicOrdering::AcqRel);
                *out_elem = old; // Should be 0 (initial value)
            } else {
                // Other threads wait, then read
                thread::sync_threads();
                let val = atomic_target.load(AtomicOrdering::Acquire);
                *out_elem = val; // Should be 0xDEADBEEF
            }
        }
    }

    /// Test 13: DeviceAtomicI32 fetch_min / fetch_max -- signed min/max RMW.
    ///
    /// All threads atomically update min and max accumulators with their
    /// (signed) thread id offset by -128. After 256 threads:
    /// - min should be -128 (thread 0's value)
    /// - max should be 127 (thread 255's value)
    #[kernel]
    pub fn atomic_minmax_test(min_acc: &[i32], max_acc: &[i32], mut out: DisjointSlice<i32>) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_min = unsafe { &*(min_acc.as_ptr() as *const DeviceAtomicI32) };
        let atomic_max = unsafe { &*(max_acc.as_ptr() as *const DeviceAtomicI32) };

        // Each thread contributes a signed value: tid - 128
        // Range: -128 to +127 for 256 threads
        let val = tid as i32 - 128;

        atomic_min.fetch_min(val, AtomicOrdering::Relaxed);
        atomic_max.fetch_max(val, AtomicOrdering::Relaxed);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = val;
        }
    }

    /// Test 14: DeviceAtomicF32 fetch_add -- floating-point atomic add.
    ///
    /// Each thread adds 1.0 to a counter. After N threads, should equal N.0.
    /// This tests the FAdd RMW kind path (atomicrmw fadd).
    #[kernel]
    pub fn atomic_f32_fetch_add_test(counter: &[f32], mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();

        let atomic_counter = unsafe { &*(counter.as_ptr() as *const DeviceAtomicF32) };

        // Each thread adds 1.0
        let _old = atomic_counter.fetch_add(1.0, AtomicOrdering::Relaxed);

        // Store 1 to indicate this thread ran
        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = 1;
        }
    }

    /// Test 15: DeviceAtomicF64 fetch_add -- 64-bit floating-point atomic add.
    ///
    /// Same pattern as test 14 but with f64. Requires sm_60+ (we target sm_80+).
    #[kernel]
    pub fn atomic_f64_fetch_add_test(counter: &[f64], mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();

        let atomic_counter = unsafe { &*(counter.as_ptr() as *const DeviceAtomicF64) };

        let _old = atomic_counter.fetch_add(1.0, AtomicOrdering::Relaxed);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = 1;
        }
    }

    /// Test 16: DeviceAtomicF32 swap -- float atomic exchange.
    ///
    /// Thread 0 swaps in 3.14, gets back 0.0. Other threads read 3.14 after barrier.
    /// This verifies atomicrmw xchg works on float types.
    #[kernel]
    pub fn atomic_f32_swap_test(target: &[f32], mut out: DisjointSlice<u32>) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_target = unsafe { &*(target.as_ptr() as *const DeviceAtomicF32) };

        if let Some(out_elem) = out.get_mut(gid) {
            if tid == 0 {
                let old = atomic_target.swap(3.14, AtomicOrdering::AcqRel);
                // old should be 0.0 (initial); write 1 to indicate success
                // We check old == 0.0 by testing if it's < 0.01
                if old < 0.01 {
                    *out_elem = 1;
                } else {
                    *out_elem = 0;
                }
            } else {
                thread::sync_threads();
                let val = atomic_target.load(AtomicOrdering::Acquire);
                // All other threads should read ~3.14
                if val > 3.0 {
                    *out_elem = 1;
                } else {
                    *out_elem = 0;
                }
            }
        }
    }

    /// Test 17: DeviceAtomicU32 fetch_min / fetch_max -- unsigned min/max (UMin/UMax).
    ///
    /// All threads contribute their tid. After 256 threads:
    /// - min should be 0 (thread 0's value)
    /// - max should be 255 (thread 255's value)
    ///
    /// This specifically tests the UMin/UMax path (unsigned), whereas test 13
    /// tests the Min/Max path (signed).
    #[kernel]
    pub fn atomic_unsigned_minmax_test(
        min_acc: &[u32],
        max_acc: &[u32],
        mut out: DisjointSlice<u32>,
    ) {
        let tid = thread::threadIdx_x();
        let gid = thread::index_1d();

        let atomic_min = unsafe { &*(min_acc.as_ptr() as *const DeviceAtomicU32) };
        let atomic_max = unsafe { &*(max_acc.as_ptr() as *const DeviceAtomicU32) };

        atomic_min.fetch_min(tid, AtomicOrdering::Relaxed);
        atomic_max.fetch_max(tid, AtomicOrdering::Relaxed);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = tid;
        }
    }

    /// Test 18: BlockAtomicU32 fetch_add -- block-scope (.cta) atomics.
    ///
    /// Uses BlockAtomicU32 which emits syncscope("block") → `.cta` in PTX.
    /// Since we launch a single block, block scope is correct here.
    /// The counter should reach N just like device-scope, but with cheaper
    /// coherence (block scope only guarantees visibility within the CTA).
    #[kernel]
    pub fn atomic_block_scope_test(counter: &[u32], mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();

        let atomic_counter = unsafe { &*(counter.as_ptr() as *const BlockAtomicU32) };

        let old = atomic_counter.fetch_add(1, AtomicOrdering::Relaxed);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = old;
        }
    }

    /// Test 19: BlockAtomicU32 fetch_add with AcqRel -- proves `.cta` scope on fences.
    ///
    /// Same logic as test 18, but uses AcqRel ordering.  Fence-splitting emits:
    ///   fence.acq_rel.cta;  atom.add.u32 ...;  fence.acq_rel.cta;
    /// The `.cta` on the fences confirms the block-scope syncscope is propagated.
    #[kernel]
    pub fn atomic_block_scope_acqrel_test(counter: &[u32], mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();

        let atomic_counter = unsafe { &*(counter.as_ptr() as *const BlockAtomicU32) };

        let old = atomic_counter.fetch_add(1, AtomicOrdering::AcqRel);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = old;
        }
    }

    /// Test 20: core::sync::atomic::AtomicU32 -- standard library atomics.
    ///
    /// Uses `core::sync::atomic::AtomicU32` with full path (no alias) to avoid
    /// collision with `cuda_device::atomic::DeviceAtomicU32`. Verifies that
    /// std::intrinsics::atomic_xadd is intercepted and lowered to NVVM atomics
    /// with system scope.
    #[kernel]
    pub fn core_atomic_fetch_add_test(counter: &[u32], mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();

        let atomic_counter =
            unsafe { &*(counter.as_ptr() as *const core::sync::atomic::AtomicU32) };

        let old = atomic_counter.fetch_add(1, Ordering::Relaxed);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = old;
        }
    }

    /// Compile-only coverage for core intrinsic generic layouts and tuple results.
    #[kernel]
    pub fn core_atomic_ordering_probe(counter: &[u32], mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();
        let pointer = counter.as_ptr() as *mut u32;
        let current = unsafe {
            core::intrinsics::atomic_load::<u32, { core::intrinsics::AtomicOrdering::Acquire }>(
                pointer,
            )
        };
        unsafe {
            core::intrinsics::atomic_store::<u32, { core::intrinsics::AtomicOrdering::Release }>(
                pointer, current,
            )
        };
        let swapped = unsafe {
            core::intrinsics::atomic_xchg::<u32, { core::intrinsics::AtomicOrdering::AcqRel }>(
                pointer, current,
            )
        };
        let (observed, succeeded) = unsafe {
            core::intrinsics::atomic_cxchg::<
                u32,
                { core::intrinsics::AtomicOrdering::Release },
                { core::intrinsics::AtomicOrdering::Acquire },
            >(pointer, swapped, swapped + 1)
        };

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = observed + succeeded as u32;
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    println!("=== Unified Atomics Test ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let module = ctx
        .load_module_from_file("atomics.ptx")
        .expect("Failed to load PTX module");
    let module = kernels::from_module(module).expect("Failed to initialize typed CUDA module");

    const N: usize = 256;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (N as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // =========================================================================
    // Test 1: fetch_add
    // =========================================================================
    println!("--- Test 1: atomic_fetch_add_test ---");
    {
        // Allocate a single u32 counter initialized to 0
        let counter_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.atomic_fetch_add_test((stream).as_ref(), cfg, &counter_dev, &mut out_dev) }
            .expect("Kernel launch failed");

        stream.synchronize().unwrap();

        // The counter should equal N after all threads increment it
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let out_vals = out_dev.to_host_vec(&stream).unwrap();

        if counter_val[0] == N as u32 {
            println!("  Counter final value: {} (expected {})", counter_val[0], N);

            // Verify all old values are unique (each thread got a different value)
            let mut sorted = out_vals.clone();
            sorted.sort();
            sorted.dedup();
            if sorted.len() == N {
                println!("  All {} fetch_add return values are unique", N);
            } else {
                println!(
                    "  WARNING: Only {} unique values (expected {})",
                    sorted.len(),
                    N
                );
            }
        } else {
            println!("  FAIL: Counter = {} (expected {})", counter_val[0], N);
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 2: load/store
    // =========================================================================
    println!("\n--- Test 2: atomic_load_store_test ---");
    {
        let flag_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.atomic_load_store_test((stream).as_ref(), cfg, &flag_dev, &mut out_dev) }
            .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        let all_42 = result.iter().all(|&x| x == 42);
        if all_42 {
            println!("  All {} threads read 42 after atomic store", N);
        } else {
            let mismatches: Vec<_> = result
                .iter()
                .enumerate()
                .filter(|&(_, &x)| x != 42)
                .take(10)
                .collect();
            println!("  FAIL: {} mismatches: {:?}", mismatches.len(), mismatches);
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 3: compare_exchange
    // =========================================================================
    println!("\n--- Test 3: atomic_cas_test ---");
    {
        let winner_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.atomic_cas_test((stream).as_ref(), cfg, &winner_dev, &mut out_dev) }
            .expect("Kernel launch failed");

        stream.synchronize().unwrap();

        let winner_val = winner_dev.to_host_vec(&stream).unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        let num_winners: usize = result.iter().filter(|&&x| x == 1).count();
        let winner_tid = winner_val[0];

        if num_winners == 1 && winner_tid >= 1 && winner_tid <= N as u32 {
            println!("  Exactly 1 winner (tid {})", winner_tid - 1);
            println!("  {} threads lost the CAS race", N - 1);
        } else {
            println!(
                "  FAIL: {} winners, winner_val = {} (expected exactly 1 winner)",
                num_winners, winner_tid
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 4: fetch_add with AcqRel (fence-splitting workaround)
    // =========================================================================
    println!("\n--- Test 4: atomic_fetch_add_acqrel_test ---");
    {
        let counter_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_fetch_add_acqrel_test((stream).as_ref(), cfg, &counter_dev, &mut out_dev)
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let out_vals = out_dev.to_host_vec(&stream).unwrap();

        if counter_val[0] == N as u32 {
            let mut sorted = out_vals.clone();
            sorted.sort();
            sorted.dedup();
            if sorted.len() == N {
                println!(
                    "  Counter = {} with AcqRel ordering, all {} values unique",
                    N, N
                );
            } else {
                println!(
                    "  FAIL: Only {} unique values (expected {})",
                    sorted.len(),
                    N
                );
                std::process::exit(1);
            }
        } else {
            println!("  FAIL: Counter = {} (expected {})", counter_val[0], N);
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 5: fetch_add with SeqCst (fence.sc pattern)
    // =========================================================================
    println!("\n--- Test 5: atomic_fetch_add_seqcst_test ---");
    {
        let counter_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_fetch_add_seqcst_test((stream).as_ref(), cfg, &counter_dev, &mut out_dev)
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let out_vals = out_dev.to_host_vec(&stream).unwrap();

        if counter_val[0] == N as u32 {
            let mut sorted = out_vals.clone();
            sorted.sort();
            sorted.dedup();
            if sorted.len() == N {
                println!(
                    "  Counter = {} with SeqCst ordering, all {} values unique",
                    N, N
                );
            } else {
                println!(
                    "  FAIL: Only {} unique values (expected {})",
                    sorted.len(),
                    N
                );
                std::process::exit(1);
            }
        } else {
            println!("  FAIL: Counter = {} (expected {})", counter_val[0], N);
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 6: DeviceAtomicI32 (signed atomics)
    // =========================================================================
    println!("\n--- Test 6: atomic_i32_test ---");
    {
        let counter_dev = DeviceBuffer::<i32>::zeroed(&stream, 1).unwrap();
        let cas_target_dev = DeviceBuffer::<i32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_i32_test(
                (stream).as_ref(),
                cfg,
                &counter_dev,
                &cas_target_dev,
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let cas_val = cas_target_dev.to_host_vec(&stream).unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        let counter_ok = counter_val[0] == N as i32;
        let cas_ok = cas_val[0] == -42;
        // Thread 0 should have written 1 (CAS succeeded)
        let thread0_ok = result[0] == 1;

        if counter_ok && cas_ok && thread0_ok {
            println!("  i32 counter = {} (expected {})", counter_val[0], N);
            println!("  i32 CAS: 0 -> {} (expected -42)", cas_val[0]);
            println!("  Thread 0 CAS result = {} (1 = success)", result[0]);
        } else {
            println!(
                "  FAIL: counter={} cas={} thread0={}",
                counter_val[0], cas_val[0], result[0]
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 7: Multi-block fetch_add (device-scope across CTAs)
    // =========================================================================
    println!("\n--- Test 7: atomic_multiblock_test ---");
    {
        let multiblock_cfg = LaunchConfig {
            grid_dim: (4, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        let total_threads: usize = 4 * 64;

        let counter_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, total_threads).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_multiblock_test(
                (stream).as_ref(),
                multiblock_cfg,
                &counter_dev,
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let out_vals = out_dev.to_host_vec(&stream).unwrap();

        if counter_val[0] == total_threads as u32 {
            let mut sorted = out_vals.clone();
            sorted.sort();
            sorted.dedup();
            if sorted.len() == total_threads {
                println!(
                    "  Counter = {} across 4 blocks x 64 threads, all {} values unique",
                    total_threads, total_threads
                );
            } else {
                println!(
                    "  FAIL: Only {} unique values (expected {})",
                    sorted.len(),
                    total_threads
                );
                std::process::exit(1);
            }
        } else {
            println!(
                "  FAIL: Counter = {} (expected {})",
                counter_val[0], total_threads
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 8: DeviceAtomicU64 fetch_add (64-bit unsigned)
    // =========================================================================
    println!("\n--- Test 8: atomic_u64_fetch_add_test ---");
    {
        let counter_dev = DeviceBuffer::<u64>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_u64_fetch_add_test((stream).as_ref(), cfg, &counter_dev, &mut out_dev)
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let out_vals = out_dev.to_host_vec(&stream).unwrap();

        if counter_val[0] == N as u64 {
            let mut sorted = out_vals.clone();
            sorted.sort();
            sorted.dedup();
            if sorted.len() == N {
                println!("  u64 counter = {}, all {} values unique", N, N);
            } else {
                println!(
                    "  FAIL: Only {} unique values (expected {})",
                    sorted.len(),
                    N
                );
                std::process::exit(1);
            }
        } else {
            println!("  FAIL: Counter = {} (expected {})", counter_val[0], N);
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 9: DeviceAtomicI64 fetch_add + compare_exchange (64-bit signed)
    // =========================================================================
    println!("\n--- Test 9: atomic_i64_test ---");
    {
        let counter_dev = DeviceBuffer::<i64>::zeroed(&stream, 1).unwrap();
        let cas_target_dev = DeviceBuffer::<i64>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<i64>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_i64_test(
                (stream).as_ref(),
                cfg,
                &counter_dev,
                &cas_target_dev,
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let cas_val = cas_target_dev.to_host_vec(&stream).unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        let counter_ok = counter_val[0] == N as i64;
        let cas_ok = cas_val[0] == -100;
        let thread0_ok = result[0] == 1;

        if counter_ok && cas_ok && thread0_ok {
            println!("  i64 counter = {} (expected {})", counter_val[0], N);
            println!("  i64 CAS: 0 -> {} (expected -100)", cas_val[0]);
            println!("  Thread 0 CAS result = {} (1 = success)", result[0]);
        } else {
            println!(
                "  FAIL: counter={} cas={} thread0={}",
                counter_val[0], cas_val[0], result[0]
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 10: fetch_sub
    // =========================================================================
    println!("\n--- Test 10: atomic_fetch_sub_test ---");
    {
        // Start counter at N, each thread subtracts 1 → should reach 0
        let counter_host: Vec<u32> = vec![N as u32];
        let counter_dev = DeviceBuffer::from_host(&stream, &counter_host).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.atomic_fetch_sub_test((stream).as_ref(), cfg, &counter_dev, &mut out_dev) }
            .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let out_vals = out_dev.to_host_vec(&stream).unwrap();

        if counter_val[0] == 0 {
            let mut sorted = out_vals.clone();
            sorted.sort();
            sorted.dedup();
            if sorted.len() == N {
                println!("  fetch_sub: {} -> 0, all {} old values unique", N, N);
            } else {
                println!(
                    "  FAIL: Only {} unique values (expected {})",
                    sorted.len(),
                    N
                );
                std::process::exit(1);
            }
        } else {
            println!("  FAIL: Counter = {} (expected 0)", counter_val[0]);
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 11: bitwise ops (fetch_and, fetch_or, fetch_xor)
    // =========================================================================
    println!("\n--- Test 11: atomic_bitwise_test ---");
    {
        // OR accumulator: starts at 0
        let or_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        // AND accumulator: starts at 0xFFFFFFFF
        let and_host: Vec<u32> = vec![0xFFFFFFFF];
        let and_dev = DeviceBuffer::from_host(&stream, &and_host).unwrap();
        // XOR accumulator: starts at 0
        let xor_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();

        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_bitwise_test(
                (stream).as_ref(),
                cfg,
                &or_dev,
                &and_dev,
                &xor_dev,
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let or_val = or_dev.to_host_vec(&stream).unwrap()[0];
        let and_val = and_dev.to_host_vec(&stream).unwrap()[0];
        let xor_val = xor_dev.to_host_vec(&stream).unwrap()[0];

        // With N=256 threads and tid % 32, all 32 bits should be set
        let or_ok = or_val == 0xFFFFFFFF;
        // Thread 0 ANDs with 0x0000FFFF, clearing upper 16 bits
        let and_ok = and_val == 0x0000FFFF;
        // 256 threads each XOR with 1: even count → back to 0
        let xor_ok = xor_val == 0;

        if or_ok && and_ok && xor_ok {
            println!("  fetch_or:  0x{:08X} (expected 0xFFFFFFFF)", or_val);
            println!("  fetch_and: 0x{:08X} (expected 0x0000FFFF)", and_val);
            println!("  fetch_xor: 0x{:08X} (expected 0x00000000)", xor_val);
        } else {
            println!(
                "  FAIL: or=0x{:08X} and=0x{:08X} xor=0x{:08X}",
                or_val, and_val, xor_val
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 12: swap
    // =========================================================================
    println!("\n--- Test 12: atomic_swap_test ---");
    {
        let target_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.atomic_swap_test((stream).as_ref(), cfg, &target_dev, &mut out_dev) }
            .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let target_val = target_dev.to_host_vec(&stream).unwrap()[0];
        let result = out_dev.to_host_vec(&stream).unwrap();

        // Thread 0 swapped 0 -> 0xDEADBEEF, got back 0
        let swap_ok = result[0] == 0;
        // Target should now be 0xDEADBEEF
        let target_ok = target_val == 0xDEADBEEF;

        if swap_ok && target_ok {
            println!(
                "  swap: old=0x{:08X} (expected 0), target=0x{:08X}",
                result[0], target_val
            );
        } else {
            println!(
                "  FAIL: old=0x{:08X} target=0x{:08X}",
                result[0], target_val
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 13: fetch_min / fetch_max (signed)
    // =========================================================================
    println!("\n--- Test 13: atomic_minmax_test ---");
    {
        // Initialize min to i32::MAX and max to i32::MIN
        let min_host: Vec<i32> = vec![i32::MAX];
        let max_host: Vec<i32> = vec![i32::MIN];
        let min_dev = DeviceBuffer::from_host(&stream, &min_host).unwrap();
        let max_dev = DeviceBuffer::from_host(&stream, &max_host).unwrap();
        let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_minmax_test((stream).as_ref(), cfg, &min_dev, &max_dev, &mut out_dev)
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let min_val = min_dev.to_host_vec(&stream).unwrap()[0];
        let max_val = max_dev.to_host_vec(&stream).unwrap()[0];

        // With 256 threads, values range from -128 to +127
        let min_ok = min_val == -128;
        let max_ok = max_val == 127;

        if min_ok && max_ok {
            println!("  fetch_min: {} (expected -128)", min_val);
            println!("  fetch_max: {} (expected +127)", max_val);
        } else {
            println!("  FAIL: min={} max={}", min_val, max_val);
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 14: DeviceAtomicF32 fetch_add (float atomic add)
    // =========================================================================
    println!("\n--- Test 14: atomic_f32_fetch_add_test ---");
    {
        let counter_dev = DeviceBuffer::<f32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_f32_fetch_add_test((stream).as_ref(), cfg, &counter_dev, &mut out_dev)
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        let expected = N as f32;
        let diff = (counter_val[0] - expected).abs();
        // f32 atomic adds may have small rounding; allow epsilon
        let counter_ok = diff < 0.01;
        let all_ran = result.iter().all(|&x| x == 1);

        if counter_ok && all_ran {
            println!("  f32 counter = {} (expected {})", counter_val[0], expected);
        } else {
            println!(
                "  FAIL: counter={} (expected ~{}), all_ran={}",
                counter_val[0], expected, all_ran
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 15: DeviceAtomicF64 fetch_add (64-bit float atomic add)
    // =========================================================================
    println!("\n--- Test 15: atomic_f64_fetch_add_test ---");
    {
        let counter_dev = DeviceBuffer::<f64>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_f64_fetch_add_test((stream).as_ref(), cfg, &counter_dev, &mut out_dev)
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        let expected = N as f64;
        let diff = (counter_val[0] - expected).abs();
        let counter_ok = diff < 0.01;
        let all_ran = result.iter().all(|&x| x == 1);

        if counter_ok && all_ran {
            println!("  f64 counter = {} (expected {})", counter_val[0], expected);
        } else {
            println!(
                "  FAIL: counter={} (expected ~{}), all_ran={}",
                counter_val[0], expected, all_ran
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 16: DeviceAtomicF32 swap (float atomic exchange)
    // =========================================================================
    println!("\n--- Test 16: atomic_f32_swap_test ---");
    {
        let target_dev = DeviceBuffer::<f32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.atomic_f32_swap_test((stream).as_ref(), cfg, &target_dev, &mut out_dev) }
            .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let target_val = target_dev.to_host_vec(&stream).unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        // target should now hold 3.14 (swapped by thread 0)
        let swap_ok = (target_val[0] - 3.14).abs() < 0.01;
        // Thread 0 must have succeeded (out[0] == 1)
        let t0_ok = result[0] == 1;

        if swap_ok && t0_ok {
            println!("  target = {} (expected ~3.14), thread 0 ok", target_val[0]);
        } else {
            println!(
                "  FAIL: target={} (expected ~3.14), t0={}",
                target_val[0], result[0]
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 17: DeviceAtomicU32 unsigned fetch_min / fetch_max (UMin/UMax)
    // =========================================================================
    println!("\n--- Test 17: atomic_unsigned_minmax_test ---");
    {
        // Initialize min accumulator to u32::MAX so any tid beats it
        let min_host = vec![u32::MAX; 1];
        let min_dev = DeviceBuffer::from_host(&stream, &min_host).unwrap();
        // Initialize max accumulator to 0 so any tid beats it
        let max_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_unsigned_minmax_test(
                (stream).as_ref(),
                cfg,
                &min_dev,
                &max_dev,
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let min_val = min_dev.to_host_vec(&stream).unwrap();
        let max_val = max_dev.to_host_vec(&stream).unwrap();

        let min_ok = min_val[0] == 0; // thread 0's tid
        let max_ok = max_val[0] == (N as u32 - 1); // thread 255's tid

        if min_ok && max_ok {
            println!(
                "  min = {} (expected 0), max = {} (expected {})",
                min_val[0],
                max_val[0],
                N - 1
            );
        } else {
            println!(
                "  FAIL: min={} (expected 0), max={} (expected {})",
                min_val[0],
                max_val[0],
                N - 1
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 18: BlockAtomicU32 fetch_add (block scope / .cta, Relaxed)
    // =========================================================================
    println!("\n--- Test 18: atomic_block_scope_test ---");
    {
        let counter_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_block_scope_test((stream).as_ref(), cfg, &counter_dev, &mut out_dev)
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        let counter_ok = counter_val[0] == N as u32;
        // Each old value should be unique in [0, N)
        let mut seen = vec![false; N];
        let mut unique_ok = true;
        for &v in &result {
            if (v as usize) < N && !seen[v as usize] {
                seen[v as usize] = true;
            } else {
                unique_ok = false;
                break;
            }
        }

        if counter_ok && unique_ok {
            println!(
                "  counter = {} (expected {}), all old values unique",
                counter_val[0], N
            );
        } else {
            println!(
                "  FAIL: counter={} (expected {}), unique={}",
                counter_val[0], N, unique_ok
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 19: BlockAtomicU32 fetch_add with AcqRel (.cta scope on fences)
    // =========================================================================
    println!("\n--- Test 19: atomic_block_scope_acqrel_test ---");
    {
        let counter_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.atomic_block_scope_acqrel_test(
                (stream).as_ref(),
                cfg,
                &counter_dev,
                &mut out_dev,
            )
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        let counter_ok = counter_val[0] == N as u32;
        let mut seen = vec![false; N];
        let mut unique_ok = true;
        for &v in &result {
            if (v as usize) < N && !seen[v as usize] {
                seen[v as usize] = true;
            } else {
                unique_ok = false;
                break;
            }
        }

        if counter_ok && unique_ok {
            println!(
                "  counter = {} (expected {}), all old values unique",
                counter_val[0], N
            );
        } else {
            println!(
                "  FAIL: counter={} (expected {}), unique={}",
                counter_val[0], N, unique_ok
            );
            std::process::exit(1);
        }
    }

    // =========================================================================
    // Test 20: core::sync::atomic::AtomicU32 fetch_add (system scope)
    // =========================================================================
    println!("\n--- Test 20: core_atomic_fetch_add_test (core::sync::atomic) ---");
    {
        let counter_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe {
            module.core_atomic_fetch_add_test((stream).as_ref(), cfg, &counter_dev, &mut out_dev)
        }
        .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let counter_val = counter_dev.to_host_vec(&stream).unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        let counter_ok = counter_val[0] == N as u32;
        let mut seen = vec![false; N];
        let mut unique_ok = true;
        for &v in &result {
            if (v as usize) < N && !seen[v as usize] {
                seen[v as usize] = true;
            } else {
                unique_ok = false;
                break;
            }
        }

        if counter_ok && unique_ok {
            println!(
                "  counter = {} (expected {}), all old values unique",
                counter_val[0], N
            );
        } else {
            println!(
                "  FAIL: counter={} (expected {}), unique={}",
                counter_val[0], N, unique_ok
            );
            std::process::exit(1);
        }
    }

    println!("\n=== SUCCESS: All 20 atomic tests passed! ===");
}
