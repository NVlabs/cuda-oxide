/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Scalar transmutes of shared-memory pointers (guards issue #874).
//!
//! `core::mem::transmute` on a shared-memory (`addrspace(3)`) pointer used
//! to be rejected wholesale because the pointer's physical width is
//! target-mode dependent (64-bit PTX/legacy, 32-bit modern NVVM `p3:32`).
//! The scalar forms never needed the physical width: they bridge through
//! the CUDA generic address space, where every pointer's Rust-visible form
//! is its 64-bit generic address.
//!
//! The kernel exercises each scalar direction and then USES the
//! reconstructed pointers, so a wrong lowering is a wrong answer rather
//! than a silently odd bit pattern:
//!
//! - pointer -> `usize`: genericize (`addrspacecast` to generic), then
//!   `ptrtoint`. The result must be non-zero even though the first static
//!   shared allocation has shared-local offset zero.
//! - `usize` -> pointer: `inttoptr` to generic, then `addrspacecast` back
//!   into the shared space. Apply + undo must be pointer-value identity,
//!   and every thread writes its own element through the recovered
//!   pointer.
//! - pointer -> pointer across address spaces: a direct `addrspacecast`
//!   each way. The restored shared pointer must equal the original and a
//!   store through it must be visible through the original allocation.
//!
//! Each transmute lives in an `#[inline(never)]` device function so LLVM
//! cannot see both halves of a round trip in one body and fold the pair
//! away; the lowering's own bridge is what executes.
//!
//! Run: `cargo oxide run shared_ptr_transmute`

#![allow(static_mut_refs)]

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, device, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    const THREADS: usize = 32;

    // Clippy would rewrite these transmutes as `as` casts, but the `as`
    // form lowers through a different compiler path (PtrToAddress /
    // pointer casts); `Transmute` lowering is the path under test here.

    /// Shared pointer -> `usize` as a literal transmute (the sysroot
    /// implements `ptr::addr()` the same way). Must genericize first.
    #[allow(clippy::transmutes_expressible_as_ptr_casts)]
    #[inline(never)]
    #[device]
    fn pointer_to_bits(pointer: *mut SharedArray<u32, THREADS>) -> usize {
        unsafe { core::mem::transmute(pointer) }
    }

    /// `usize` -> shared pointer: must re-enter the shared space through
    /// the generic space (inttoptr to generic, then addrspacecast).
    #[allow(clippy::transmutes_expressible_as_ptr_casts)]
    #[inline(never)]
    #[device]
    fn bits_to_pointer(bits: usize) -> *mut SharedArray<u32, THREADS> {
        unsafe { core::mem::transmute(bits) }
    }

    /// Shared pointer -> generic pointer: a direct addrspacecast.
    #[allow(clippy::transmutes_expressible_as_ptr_casts)]
    #[inline(never)]
    #[device]
    fn pointer_to_alias(pointer: *mut SharedArray<u32, THREADS>) -> *mut u32 {
        unsafe { core::mem::transmute(pointer) }
    }

    /// Generic pointer -> shared pointer: the reverse addrspacecast.
    #[allow(clippy::transmutes_expressible_as_ptr_casts)]
    #[inline(never)]
    #[device]
    fn alias_to_pointer(alias: *mut u32) -> *mut SharedArray<u32, THREADS> {
        unsafe { core::mem::transmute(alias) }
    }

    #[kernel]
    pub fn shared_ptr_transmute(mut out: DisjointSlice<u32>) {
        static mut SCRATCH: SharedArray<u32, THREADS> = SharedArray::UNINIT;

        let lane = thread::threadIdx_x() as usize;
        let raw = &raw mut SCRATCH;

        // usize round trip: apply + undo == identity, checked as pointer
        // VALUE equality. Hardware masks the shared-window base out of
        // st.shared addresses, so a wrong pointer value could still store
        // to the right slot; `ptr::eq` catches that.
        let bits = pointer_to_bits(raw);
        let recovered = bits_to_pointer(bits);

        // Every thread writes its own element through the recovered
        // pointer. No thread constructs an `&mut SharedArray` spanning
        // elements owned by other threads.
        let base = unsafe { SharedArray::as_raw_mut_ptr(recovered) };
        if lane < THREADS {
            unsafe { base.add(lane).write(lane as u32 + 1) };
        }
        thread::sync_threads();

        if lane == 0 {
            let mut sum = 0_u32;
            for index in 0..THREADS {
                sum += unsafe { base.add(index).read() };
            }

            // Pointer -> pointer across address spaces, each way.
            let alias = pointer_to_alias(raw);
            let restored = alias_to_pointer(alias);

            unsafe {
                // The first static shared allocation has shared-local
                // offset zero, but its exposed bits are a generic address
                // and must not look null.
                *out.get_unchecked_mut(0) = (core::ptr::eq(recovered, raw) && bits != 0) as u32;
                *out.get_unchecked_mut(1) = sum;
                *out.get_unchecked_mut(2) = core::ptr::eq(restored, raw) as u32;
                // Store through the restored pointer, observe through the
                // original allocation.
                (&mut (*restored))[0] = 0xC0DE;
                *out.get_unchecked_mut(3) = SCRATCH[0];
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== shared_ptr_transmute (issue #874 regression) ===");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, 4)?;

    // SAFETY: one 32-thread block writes 32 distinct shared elements, then
    // thread 0 reads them after a block-wide barrier. Only thread 0 writes
    // the four output elements.
    unsafe { module.shared_ptr_transmute(&stream, cfg, &mut out) }?;

    let result = out.to_host_vec(&stream)?;
    let expected_sum: u32 = (1..=32).sum();

    if result[0] != 1 {
        eprintln!("FAIL shared_ptr_transmute: usize round trip did not recover the pointer");
        std::process::exit(1);
    }
    if result[1] != expected_sum {
        eprintln!(
            "FAIL shared_ptr_transmute: writes through the recovered pointer summed to {}, expected {expected_sum}",
            result[1]
        );
        std::process::exit(1);
    }
    if result[2] != 1 {
        eprintln!(
            "FAIL shared_ptr_transmute: pointer-to-pointer round trip did not recover the pointer"
        );
        std::process::exit(1);
    }
    if result[3] != 0xC0DE {
        eprintln!(
            "FAIL shared_ptr_transmute: store through the restored pointer read back {:#X}, expected 0xC0DE",
            result[3]
        );
        std::process::exit(1);
    }
    println!(
        "PASS shared_ptr_transmute: usize and cross-space pointer transmutes round-trip, sum={}, readback={:#X}",
        result[1], result[3]
    );
    Ok(())
}
