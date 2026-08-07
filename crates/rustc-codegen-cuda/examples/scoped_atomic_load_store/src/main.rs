// SPDX-License-Identifier: Apache-2.0

//! Scoped atomic load/store and fences, exercised end to end.
//!
//! This is a regression example for a path that previously could not be
//! compiled at all. `DeviceAtomic*::load` and `::store` lowered to LLVM
//! `load atomic` / `store atomic`, and any `AcqRel` or `SeqCst` ordering
//! emitted an LLVM `fence`; libNVVM rejects all three:
//!
//! ```text
//! context:   %v48 = load atomic i32, ptr %v47 syncscope("device") monotonic
//!   Atomic loads/stores are not supported
//! context:   fence syncscope("block") release
//!   Illegal instruction: fence
//! ```
//!
//! So every call to these documented APIs was a build failure under
//! `--materialize-cubin`, which is to say in any configuration that produces a
//! real kernel. They now lower to the PTX instructions that have existed for
//! this purpose since sm_70 (`ld.acquire.gpu`, `st.release.gpu`, `fence.acq_rel.gpu`).
//!
//! The example deliberately uses the release/acquire *publication* idiom rather
//! than a synthetic load and store, because that is what the feature is for:
//! one thread writes a payload and publishes a flag with a release store;
//! another observes the flag with an acquire load and is then guaranteed to see
//! the payload.
//!
//! It uses a barrier rather than a spin loop to pair writer and reader. A spin
//! would be the more literal test, but a hung example on a scheduling quirk is
//! a much worse failure mode than a slightly weaker one, and the instruction
//! coverage is identical.
//!
//! Build and run with:
//!   cargo oxide run scoped_atomic_load_store --materialize-cubin

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const N: usize = 256;

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32, DeviceAtomicU64};

    /// Each thread publishes a payload with a release store, then reads its
    /// neighbour's with an acquire load.
    ///
    /// Exercises, in one kernel: `st.release.gpu.b32`, `ld.acquire.gpu.b32`,
    /// `st.relaxed.gpu.b64`, `ld.relaxed.gpu.b64`, and the `fence.acq_rel.gpu`
    /// emitted by the AcqRel compare-exchange. None of these could be compiled
    /// under `--materialize-cubin` before.
    #[kernel]
    pub fn publish_and_observe(
        flags: &[u32],
        wide: &[u64],
        mut observed: DisjointSlice<u32>,
        mut observed_wide: DisjointSlice<u64>,
        mut cas_ok: DisjointSlice<u32>,
    ) {
        let tid = thread::threadIdx_x() as usize;
        if tid >= N {
            return;
        }
        let next = (tid + 1) % N;

        // SAFETY: `tid` and `next` are both < N, the buffers have N elements,
        // and every access to them in this kernel is atomic.
        let mine = unsafe { DeviceAtomicU32::from_ptr((flags.as_ptr() as *mut u32).add(tid)) };
        let mine_wide =
            unsafe { DeviceAtomicU64::from_ptr((wide.as_ptr() as *mut u64).add(tid)) };
        let neighbour =
            unsafe { DeviceAtomicU32::from_ptr((flags.as_ptr() as *mut u32).add(next)) };
        let neighbour_wide =
            unsafe { DeviceAtomicU64::from_ptr((wide.as_ptr() as *mut u64).add(next)) };

        // Publish. The release store is what an acquire reader synchronises with.
        mine.store(tid as u32 + 1, AtomicOrdering::Release);
        mine_wide.store((tid as u64 + 1) << 32, AtomicOrdering::Relaxed);

        // Pair writers with readers without spinning: a hung example is a far
        // worse failure mode than a slightly weaker test, and the instruction
        // coverage is identical either way.
        thread::sync_threads();

        let seen = neighbour.load(AtomicOrdering::Acquire);
        let seen_wide = neighbour_wide.load(AtomicOrdering::Relaxed);

        if let Some(out) = observed.get_mut(thread::index_1d()) {
            *out = seen;
        }
        if let Some(out) = observed_wide.get_mut(thread::index_1d()) {
            *out = seen_wide;
        }

        // AcqRel compare-exchange: emits the fence libNVVM used to reject.
        let ok = mine
            .compare_exchange(
                tid as u32 + 1,
                tid as u32 + 1,
                AtomicOrdering::AcqRel,
                AtomicOrdering::Relaxed,
            )
            .is_ok();
        if let Some(out) = cas_ok.get_mut(thread::index_1d()) {
            *out = u32::from(ok);
        }
    }
}

fn main() {
    println!("=== Scoped atomic load/store and fences ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let flags = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    let wide = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();
    let mut observed = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    let mut observed_wide = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();
    let mut cas_ok = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // SAFETY: one block of N threads, and every buffer has N elements.
    unsafe {
        module.publish_and_observe(
            &stream,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (N as u32, 1, 1),
                shared_mem_bytes: 0,
            },
            &flags,
            &wide,
            &mut observed,
            &mut observed_wide,
            &mut cas_ok,
        )
    }
    .expect("Kernel launch failed");

    let observed = observed.to_host_vec(&stream).unwrap();
    let observed_wide = observed_wide.to_host_vec(&stream).unwrap();
    let cas_ok = cas_ok.to_host_vec(&stream).unwrap();

    let mut errors = 0;
    for i in 0..N {
        let next = (i + 1) % N;
        let want = next as u32 + 1;
        let want_wide = (next as u64 + 1) << 32;
        if observed[i] != want {
            if errors < 5 {
                eprintln!("  acquire load: thread {i} saw {}, expected {want}", observed[i]);
            }
            errors += 1;
        }
        if observed_wide[i] != want_wide {
            if errors < 5 {
                eprintln!(
                    "  relaxed 64-bit load: thread {i} saw {:#x}, expected {want_wide:#x}",
                    observed_wide[i]
                );
            }
            errors += 1;
        }
        if cas_ok[i] != 1 {
            if errors < 5 {
                eprintln!("  AcqRel compare_exchange failed on thread {i}");
            }
            errors += 1;
        }
    }

    println!("  release store / acquire load   {} threads", N);
    println!("  relaxed 64-bit store / load    {} threads", N);
    println!("  AcqRel compare_exchange        {} threads", N);

    if errors == 0 {
        println!("\n=== SUCCESS: scoped atomic load/store and fences all correct ===");
    } else {
        eprintln!("\n=== FAILURE: {errors} mismatches ===");
        std::process::exit(1);
    }
}
