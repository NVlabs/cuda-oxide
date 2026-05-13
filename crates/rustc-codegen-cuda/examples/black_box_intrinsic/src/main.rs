//! Regression test for `core::hint::black_box` lowering.
//!
//! ## Pre-fix wall
//!
//! ```text
//! Unsupported construct: unhandled core intrinsic: `std::intrinsics::black_box`
//! (bodyless `#[rustc_intrinsic]`, no MIR body).
//! ```
//!
//! Surfaced from `vanity-miner-rs/logic/src/self_test.rs` — the
//! per-PTX-op bisection slots (slots 31-40) wrap their operands in
//! `core::hint::black_box` so the runtime expression isn't const-
//! folded against the host-evaluated `EXPECTED` baseline. Without
//! `black_box` support, the importer's guardrail rejects the build
//! before any PTX is emitted.
//!
//! ## What landed
//!
//! `try_dispatch_intrinsic` now recognises `core::intrinsics::black_box`
//! (and the `std::` re-export) and lowers to a new `nvvm.black_box`
//! op. `MirToLlvmConversion` on that op emits an empty-template inline
//! `asm sideeffect` with register input/output — the same shape
//! rustc's LLVM backend uses. LLVM treats this as opaque, so the
//! optimizer can't see through `black_box` and const-fold its
//! arguments — which is the whole point of the intrinsic.
//!
//! ## Build with
//!
//!     cargo oxide build black_box_intrinsic
//!
//! ## Verification
//!
//! On real hardware, the kernel writes `wrapping_mul(a, b)` for each
//! thread's `(a, b)` pair. The host asserts the result. A passing run
//! proves both (a) the build succeeded and (b) the lowered inline asm
//! preserves the value through the barrier. The PTX should contain a
//! real `mul.lo.s64` against runtime registers, not a hoisted const.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Each thread `i` writes `black_box(C0).wrapping_mul(black_box(C1) ^ i)`.
    /// The XOR by `i` makes the result thread-dependent so the
    /// compiler can't hoist the whole expression to a kernel-uniform
    /// constant even if `black_box` somehow folded through.
    #[kernel]
    pub fn run(mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            let a: u64 = core::hint::black_box(0xDEADBEEFCAFEBABE_u64);
            let b: u64 = core::hint::black_box(0x123456789ABCDEF0_u64);
            let mixed = b ^ (i as u64);
            *slot = a.wrapping_mul(mixed);
        }
    }
}

fn main() {
    println!("=== black_box_intrinsic ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 16;
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .run(&stream, LaunchConfig::for_num_elems(N as u32), &mut out)
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let a: u64 = 0xDEADBEEFCAFEBABE;
        let b: u64 = 0x123456789ABCDEF0;
        let expected = a.wrapping_mul(b ^ (i as u64));
        assert_eq!(result[i], expected, "thread {} mismatch", i);
    }
    println!("SUCCESS: black_box intrinsic lowered to inline-asm barrier");
}
