//! Regression test for `core::intrinsics::volatile_load` lowering.
//!
//! ## Pre-fix wall
//!
//! ```text
//! Symbol _RINvNtCsbBDxv2Oq2Kj_4core10intrinsics13volatile_loadhE not found
//! ```
//!
//! Surfaced from `~/vanity-miner-rs/` via `core::ptr::read_volatile`,
//! which is a thin wrapper around `intrinsics::volatile_load`. The
//! intrinsic has no MIR body, so the collector skipped it; the
//! translator emitted a regular call to a symbol nothing defined.
//!
//! ## What landed
//!
//! `try_dispatch_intrinsic`'s match block now recognises
//! `core::intrinsics::volatile_load` (and the `std::` re-export) and
//! lowers it to a plain `mir.load`. On GPU there is no meaningful
//! "volatile" semantics — these calls come from defensive library
//! code where a normal load is correct.
//!
//! ## Build with
//!
//!     cargo oxide build volatile_load_intrinsic

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// `#[inline(never)]` keeps `read_volatile` from being inlined away
/// before the codegen sees it. The function is generic over T so the
/// intrinsic gets monomorphized per-call (matches the user wall on
/// `volatile_load::<u8>`).
#[inline(never)]
fn read_via_volatile<T: Copy>(src: *const T) -> T {
    // `read_volatile` is the public wrapper around
    // `core::intrinsics::volatile_load`. The intrinsic body is bare
    // `#[rustc_intrinsic]` — exactly the shape we want to exercise.
    unsafe { core::ptr::read_volatile(src) }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn run(input: &[u8], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i < input.len()
        {
            // Volatile-load a u8 (matches `volatile_loadhE` in the
            // mangled symbol from the user wall).
            let byte: u8 = super::read_via_volatile::<u8>(&input[i] as *const u8);
            *slot = byte as u32;
        }
    }
}

fn main() {
    println!("=== volatile_load_intrinsic ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 16;
    let host: Vec<u8> = (0..N as u8).map(|i| i.wrapping_mul(7).wrapping_add(3)).collect();
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .run(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        assert_eq!(result[i], host[i] as u32, "thread {} mismatch", i);
    }
    println!("SUCCESS: volatile_load intrinsic codegen'd to PTX");
}
