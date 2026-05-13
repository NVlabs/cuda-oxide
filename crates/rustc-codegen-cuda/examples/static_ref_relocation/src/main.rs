//! Known-failure reproducer for cross-static pointer relocations
//! not being applied in device-global memory.
//!
//! ## Status
//!
//! `cargo oxide build` succeeds today — the bug is in the emitted
//! PTX, not at codegen time. On real GPU the kernel reads `0x0`
//! from where `OUTER`'s body should hold the address of `INNER`,
//! then dereferences and faults with
//! `CUDA_ERROR_ILLEGAL_ADDRESS` (700).
//!
//! ## Pre-fix wall (compute-sanitizer on real GPU)
//!
//! ```text
//! Invalid __global__ read of size 8 bytes
//!     at $kernel_..._FieldElement51_..._conditional_assign+0x59b0
//!     by thread (0,0,0) in block (0,0,0)
//!     Access to 0x0 is out of bounds
//!     and is 46170898432 bytes before the nearest allocation at 0xac0000000
//! ```
//!
//! Address `0x0`, not a slightly-bad address — clean null deref.
//! The kernel read the 8-byte pointer body of an outer
//! `pub static X: &T = &INNER` static, expecting it to hold
//! `INNER`'s relocated address, and got zero.
//!
//! Surfaced from `~/vanity-miner-rs/`'s self-test slot 1
//! (`kernel_self_test_primitive_ed25519`):
//!   `<Scalar as Mul<EdwardsBasepointTable>>::mul`
//!     -> `<EdwardsBasepointTable as Mul<Scalar>>::mul`
//!       -> `EdwardsBasepointTable::mul_base`
//!         -> `LookupTable<AffineNielsPoint>::select`
//!           -> `AffineNielsPoint::conditional_assign`
//!             -> `FieldElement51::conditional_assign`   <-- faulting frame
//!
//! ## What triggers it
//!
//! curve25519-dalek (and many other crypto crates) lay out
//! their precomputed tables as a `pub static` whose body is a
//! reference to a private inner static holding the data:
//!
//! ```rust,ignore
//! // curve25519-dalek-4.1.3/src/backend/serial/u64/constants.rs:331
//! pub static ED25519_BASEPOINT_TABLE: &EdwardsBasepointTable =
//!     &ED25519_BASEPOINT_TABLE_INNER_DOC_HIDDEN;
//!
//! static ED25519_BASEPOINT_TABLE_INNER_DOC_HIDDEN: EdwardsBasepointTable =
//!     EdwardsBasepointTable([ /* ~30KB of precomputed scalar mult tables */ ]);
//! ```
//!
//! This is a **cross-static relocation**: the outer static's body
//! is a pointer that the linker fills in with the inner static's
//! resolved address at link time. In LLVM IR, this is the shape
//!
//! ```llvm,ignore
//! @INNER = internal global [4 x i64] [i64 1, i64 2, i64 3, i64 4]
//! @OUTER = constant ptr @INNER
//! ```
//!
//! cuda-oxide is currently emitting `@OUTER`'s body as zero (or
//! omitting `@INNER` from the device module entirely), so at
//! runtime the kernel loads `0x0` instead of `@INNER`'s address.
//!
//! ## Verifying the bug in the emitted PTX
//!
//! ```sh
//! cargo oxide build static_ref_relocation
//! grep -E '^\.global|^\.visible \.global|^\.const' \
//!     crates/rustc-codegen-cuda/examples/static_ref_relocation/static_ref_relocation.ptx
//! ```
//!
//! Bug present: the inner static's `.global` symbol is missing,
//! zero-initialized, or the outer ref-static's body holds a
//! null pointer instead of the inner's relocated address.
//!
//! ## What we expect to land
//!
//! Wherever mir-importer / mir-lower / dialect-llvm export
//! handles `pub static X: &T = &INNER`:
//!
//! 1. Ensure `INNER` is emitted as a `.global` (or `.const`)
//!    with its full initializer, even when reachable only
//!    through the outer ref-static.
//! 2. Ensure the outer static's body is emitted as the
//!    relocated address of `INNER`, not as a null pointer
//!    or as a zero-initialized buffer.
//!
//! The same shape underlies many crypto/lookup-table crates:
//! sha2's K constants, k256's affine generator, etc. Fixing
//! this one shape unblocks the family.
//!
//! ## Why a fresh hardware run of this repro may pass
//!
//! Depending on which step is broken in the pipeline, this
//! repro could also build to PTX that "works by luck" (e.g.,
//! a constant-folded compile-time read of a known-value INNER).
//! Confirmation is a PTX-text check, not just a hardware run.
//!
//! ## What this example is NOT
//!
//! - Not about a local stack misalignment (see
//!   `xoshiro_seed_misalign/` for that bug, fixed by the
//!   conservative alloca-align rule).
//! - Not about a missing intrinsic handler (e.g. `raw_eq`,
//!   covered by `array_eq_raw/`).
//! - Not specific to ed25519 / curve25519-dalek — any
//!   `pub static X: &T = &INNER` exhibits the same shape.
//!
//! ## Build with
//!
//!     cargo oxide build static_ref_relocation   # codegen check (passes today)
//!     cargo oxide run   static_ref_relocation   # faults on hardware until fixed

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Inner static: the actual data.
static INNER: [u64; 4] = [1, 2, 3, 4];

/// Outer static: a reference to the inner. Its body must be
/// relocated to `INNER`'s address at link time. cuda-oxide
/// currently emits this body as null, so dereferencing `OUTER`
/// inside a kernel reads `0x0`.
static OUTER: &[u64; 4] = &INNER;

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Each thread reads one slot of `OUTER`. If the cross-static
    /// relocation worked, thread `i` writes `INNER[i]` = `i + 1`.
    /// If not, `OUTER` is null at runtime and the load faults
    /// (or returns 0).
    #[kernel]
    pub fn static_ref_relocation(mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i < 4
        {
            // Two-step to keep the pointer load visible in the IR
            // (rather than something the optimizer might constant-fold
            // through if it can see through the static chain).
            let t: &[u64; 4] = OUTER;
            *slot = t[i];
        }
    }
}

fn main() {
    println!("=== static_ref_relocation ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const THREADS: usize = 4;
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, THREADS).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .static_ref_relocation(
            &stream,
            LaunchConfig::for_num_elems(THREADS as u32),
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    let expected: [u64; 4] = [1, 2, 3, 4];
    for i in 0..THREADS {
        assert_eq!(
            result[i], expected[i],
            "thread {} mismatch (got {}, want {}). \
             cross-static relocation likely produced null OUTER body.",
            i, result[i], expected[i]
        );
    }
    println!("SUCCESS: static-of-static-ref relocation lands in PTX");
}
