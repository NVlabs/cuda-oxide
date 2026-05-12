//! Reproducer for bug A of limitation #4: helper fn declared at crate root.
//!
//! Observed diagnostic on current `main`:
//!
//!   Verification failed for 'llvm module': Symbol
//!   helper_outside_module__get_thread_idx not found
//!
//! ## What actually triggers it
//!
//! The brief framed this as "crate-root pub fns aren't collected." That's
//! not the full story: a plain-arithmetic crate-root helper
//! (`pub fn double(x: u32) -> u32 { x.wrapping_mul(2) }`) gets collected
//! and emitted as a real device function. The PTX shows a `.visible .func`
//! `helper_outside_module__double` with a body, and the kernel `call.uni`s it.
//!
//! The failure shape requires a crate-root non-inline helper whose body
//! transitively reaches a `cuda_device` intrinsic (here, `thread::index_1d`).
//! In that case the collector emits the call site but never the body —
//! something about the intrinsic-bearing path through the helper loses
//! the body before PTX emission.
//!
//! This matches the original vanity-miner-rs failure: `utilities::get_thread_idx`
//! was a crate-root helper wrapping `cuda_device::thread::index_1d().get()`.
//!
//! ## Workaround
//!
//! Move the helper *inside* the `#[cuda_module]` mod AND mark it `#[inline]`.
//! See `helper_no_inline` for the partial-fix failure mode.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

// === Helpers at crate root (OUTSIDE `#[cuda_module]`) ===

/// SAFE: pure-Rust crate-root helper. The collector picks this up and
/// emits a real `.visible .func helper_outside_module__double` body.
/// Provided as a counterexample so the reader can see the precise shape
/// that fails.
pub fn double(x: u32) -> u32 {
    x.wrapping_mul(2)
}

/// FAILS: crate-root helper whose body calls a `cuda_device` intrinsic.
/// The call site `helper_outside_module__get_thread_idx` ends up in the
/// PTX module but with no body — LLVM verification rejects it.
///
/// (This matches the original vanity-miner-rs `utilities::get_thread_idx`
/// shape that motivated this bug report.)
pub fn get_thread_idx() -> usize {
    thread::index_1d().get()
}

#[cuda_module]
mod kernels {
    use super::*;

    /// PASSES on current `main`. The arithmetic-only crate-root helper
    /// gets walked correctly. Comment out the `use_intrinsic_helper`
    /// kernel below and this example will compile end-to-end.
    #[kernel]
    pub fn use_arith_helper(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            *slot = double(input[i]);
        }
    }

    /// FAILS on current `main`. Calling the intrinsic-wrapping crate-root
    /// helper drops its body before PTX emission.
    #[kernel]
    pub fn use_intrinsic_helper(input: &[u32], mut out: DisjointSlice<u32>) {
        let i = get_thread_idx();
        let idx = thread::index_1d();
        if let Some(slot) = out.get_mut(idx) {
            *slot = input[i].wrapping_mul(2);
        }
    }
}

fn main() {
    println!("=== Limitation #4 bug A: helper at crate root ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 64;
    let host: Vec<u32> = (0..N as u32).collect();
    let dev = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    module
        .use_arith_helper(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &dev,
            &mut out,
        )
        .expect("Kernel launch failed (arith helper)");
    let r = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        assert_eq!(r[i], host[i].wrapping_mul(2));
    }

    module
        .use_intrinsic_helper(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &dev,
            &mut out,
        )
        .expect("Kernel launch failed (intrinsic helper)");
    let r = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        assert_eq!(r[i], host[i].wrapping_mul(2));
    }

    println!("SUCCESS: both crate-root helpers codegen'd to PTX");
}
