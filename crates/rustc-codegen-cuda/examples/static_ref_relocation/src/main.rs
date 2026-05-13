//! Regression test for cross-static pointer relocations being
//! applied (and the inner static actually emitted) in the device
//! module's global memory.
//!
//! ## Pre-fix wall
//!
//! `cargo oxide build` succeeded, but the emitted PTX had only
//! `@__device_global_0 = addrspace(1) global ptr zeroinitializer` —
//! the outer static was a null-bodied pointer slot, and the inner
//! static was missing from the module entirely.
//!
//! On hardware, dereferencing the outer static read `0x0`. compute-
//! sanitizer flagged the resulting null read inside
//! `FieldElement51::conditional_assign`:
//!
//! ```text
//! Invalid __global__ read of size 8 bytes
//!     at $kernel_..._FieldElement51_..._conditional_assign+0x59b0
//!     Access to 0x0 is out of bounds
//! ```
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
//! curve25519-dalek (and many other crypto crates) lay out their
//! precomputed tables as a `pub static` whose body is a reference
//! to a private inner static holding the data:
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
//! resolved address. In LLVM IR, the shape is
//!
//! ```llvm,ignore
//! @INNER = addrspace(1) global [4 x i64] [i64 1, i64 2, i64 3, i64 4]
//! @OUTER = addrspace(1) global ptr addrspacecast (ptr addrspace(1) @INNER to ptr)
//! ```
//!
//! ## What landed
//!
//! Replaced `ensure_zero_initializer` in
//! `crates/mir-importer/src/translator/rvalue.rs` with
//! `compute_static_initializer` + `collect_reachable_statics`:
//!
//! * `compute_static_initializer` returns the raw bytes plus an
//!   offset-keyed list of pointer relocations (extracted from
//!   `alloc.provenance.ptrs`, which the old placeholder discarded).
//! * `collect_reachable_statics` walks the transitive closure of
//!   referenced statics so every static the kernel can reach via
//!   pointer chasing ends up in the device module.
//!
//! Plumbed through new `initializer_bytes` and
//! `initializer_relocations` attributes on `MirGlobalAllocOp` ->
//! the LLVM `GlobalOp` -> the dialect-llvm exporter
//! (`export_global`), which now emits either
//!
//! * pure bytes:
//!   `@INNER = addrspace(1) global [N x i8] c"\01\00..."`, or
//! * a packed struct interleaving byte runs with addrspacecast'd
//!   pointer relocations:
//!   `@OUTER = addrspace(1) global <{ ptr }> <{ ptr addrspacecast (ptr addrspace(1) @INNER to ptr) }>`
//!
//! Relocation target keys are resolved through a
//! `source_key_to_llvm_name` map built by a pre-pass over the
//! module's globals, so OUTER's `ptr @INNER` reference lands on
//! the correct synthetic `__device_global_N` symbol.
//!
//! Secondary MirGlobalAllocOps (the transitively-reachable ones
//! whose values the kernel doesn't directly consume) are inserted
//! at the front of the kernel block to keep the kernel's own
//! terminator the last op in its block.
//!
//! NVPTX renders the addrspacecast as `generic(__device_global_N)`
//! in the final PTX, which the JIT resolves to INNER's runtime
//! address.
//!
//! ## Verifying the fix in the emitted PTX
//!
//! ```sh
//! cargo oxide build static_ref_relocation
//! grep -E '^\.global' \
//!     crates/rustc-codegen-cuda/examples/static_ref_relocation/static_ref_relocation.ptx
//! ```
//!
//! Post-fix output:
//!
//! ```text
//! .visible .global .align 8 .b8 __device_global_0[32] = {1, 0, 0, 0, ...};
//! .visible .global .align 8 .u64 __device_global_1[1] = {generic(__device_global_0)};
//! ```
//!
//! Both halves: INNER's bytes are present, and OUTER's body holds
//! INNER's relocated address.
//!
//! ## Relationship to other repros
//!
//! - `xoshiro_seed_misalign`: local-stack-alloca alignment bug. Same
//!   class (PTX-shape codegen issue), different storage class.
//! - `array_eq_raw`: introduced the `raw_eq` intrinsic handler that
//!   produces wide loads. Independent of this fix.
//! - Not specific to ed25519 / curve25519-dalek — any
//!   `pub static X: &T = &INNER` exhibits this shape, including
//!   sha2's K constants, k256's affine generator, etc.
//!
//! ## Build with
//!
//!     cargo oxide build static_ref_relocation   # codegen check (passes today)
//!     cargo oxide run   static_ref_relocation   # faults on hardware until fixed

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Inner static: the actual data.
static INNER: [u64; 4] = [1, 2, 3, 4];

/// Outer static: a reference to the inner. Its body is the
/// relocated address of `INNER`. Pre-fix this body was null;
/// post-fix the device module contains a packed-struct global
/// whose single field is `addrspacecast (ptr addrspace(1) @INNER to ptr)`.
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
