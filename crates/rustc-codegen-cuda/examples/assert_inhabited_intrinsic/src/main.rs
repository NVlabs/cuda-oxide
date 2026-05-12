//! Regression test for `core::intrinsics::assert_inhabited`
//! lowering as a no-op.
//!
//! ## Pre-fix wall
//!
//! ```text
//! Symbol _RINvNtCsbBDxv2Oq2Kj_4core10intrinsics16assert_inhabitedI...E not found
//! ```
//!
//! Surfaced from `~/vanity-miner-rs/` via
//! `MaybeUninit::<GenericArray<u8, U33>>::assume_init()`. The
//! intrinsic is a `#[rustc_intrinsic]` with no MIR body — collector
//! skipped it, translator emitted a regular call to a symbol nothing
//! defined.
//!
//! ## What landed
//!
//! `try_dispatch_intrinsic`'s `cold_path` arm now also matches
//! `core::intrinsics::assert_inhabited` (and `std::intrinsics::...`).
//! Both route through `helpers::emit_unit_noop_intrinsic` — runtime
//! no-op (the check that T isn't `!` / empty enum is purely
//! compile-time).
//!
//! ## Build with
//!
//!     cargo oxide build assert_inhabited_intrinsic

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Newtype wrapping an array, generic over T. Mirrors the
/// `GenericArray<u8, N>` shape that triggers the user's wall —
/// generic + non-trivial type prevents `MaybeUninit::assume_init`
/// from inlining the `assert_inhabited` away.
#[derive(Clone, Copy)]
pub struct Wrapper<T: Copy>(pub [T; 8]);

/// `#[inline(never)]` on a function generic over the wrapper makes
/// the optimizer treat `assume_init` as opaque per monomorphization,
/// preserving the `core::intrinsics::assert_inhabited::<Wrapper<T>>`
/// call.
#[inline(never)]
fn assume_init_wrapper<T: Copy + Default>(input: T, fill: impl Fn(T, usize) -> T) -> Wrapper<T> {
    let mut x = core::mem::MaybeUninit::<Wrapper<T>>::uninit();
    unsafe {
        let p = x.as_mut_ptr() as *mut T;
        let mut k = 0usize;
        while k < 8 {
            *p.add(k) = fill(input, k);
            k += 1;
        }
        x.assume_init()
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn run(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i < input.len()
        {
            let w =
                super::assume_init_wrapper::<u32>(input[i], |v, k| v.wrapping_mul(k as u32 + 1));
            let mut acc: u32 = 0;
            let mut k = 0;
            while k < 8 {
                acc = acc.wrapping_add(w.0[k]);
                k += 1;
            }
            *slot = acc;
        }
    }
}

fn main() {
    println!("=== assert_inhabited_intrinsic ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 8;
    let host: Vec<u32> = (0..N as u32).collect();
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
        let mut expected: u32 = 0;
        for k in 0..8 {
            expected = expected.wrapping_add(host[i].wrapping_mul(k as u32 + 1));
        }
        assert_eq!(result[i], expected, "thread {} mismatch", i);
    }
    println!("SUCCESS: assert_inhabited intrinsic codegen'd to PTX");
}
