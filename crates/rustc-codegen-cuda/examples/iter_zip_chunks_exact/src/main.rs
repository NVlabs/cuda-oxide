//! Regression test for translating `Iterator::zip` + `chunks_exact`
//! through the mir-importer's type translator and the mir-lower call
//! lowering.
//!
//! ## Pre-fix walls
//!
//! Three layers had to land for this example to translate end-to-end.
//! Each surfaced after fixing the previous:
//!
//! 1. **`Alias(IntoIterator::IntoIter)` not handled.**
//!    `Iterator::zip<U>` returns `Zip<Self, U::IntoIter>`. After
//!    monomorphization the return type still contains the alias
//!    `<ChunksExact<u8> as IntoIterator>::IntoIter` — `rustc_public`
//!    presents types un-normalized. Added an arm in
//!    `mir-importer/src/translator/types.rs` that recurses on
//!    `self_ty` (the blanket `impl<I: Iterator> IntoIterator for I`
//!    makes `IntoIter = Self`).
//!
//! 2. **`Alias(Iterator::Item)` not handled.** Same shape, different
//!    associated type. `Item` is per-impl, so the new arm hand-rolls
//!    the `core::slice::iter` adapters that actually surface
//!    (`IterMut::Item = &mut T`, `ChunksExact::Item = &[T]`, …) and
//!    synthesizes the ref via `Ty::new_ref`.
//!
//! 3. **Tuple-result calls misclassified as unit.** `(1)` alone made
//!    the type translator succeed; mir-lower's
//!    `convert::ops::call::convert` then panicked at pliron's
//!    "Operation with use(s) being erased". Root cause: `is_unit`
//!    was set for *any* `MirTupleType`, so a real 2-tuple like
//!    `(&[u8], &[u8])` from `split_at_unchecked` (used by
//!    `ChunksExact::next` internally) got Void-return-type'd; the
//!    LLVM call then had no result, and the MIR op got erased with
//!    its tuple result still in use. Fix checked
//!    `tt.get_types().is_empty()`.
//!
//! All three landed in a single commit:
//! `Translate IntoIter / Item aliases and stop misclassifying
//! tuple-result calls as unit`.
//!
//! 4. **Missing `ptr_offset_from_unsigned` handler.** After (1)-(3)
//!    landed, codegen reached llc verification with a dangling
//!    reference to `core::intrinsics::ptr_offset_from_unsigned`
//!    (used by `ChunksExact::next`'s `len_remaining` bookkeeping).
//!    Fixed in `examples/ptr_offset_intrinsic/` + the
//!    `convert_rust_ptr_arith_intrinsic` handler.
//!
//! Originally surfaced from `~/vanity-miner-rs/logic/` after the
//! cross-crate MIR fix
//! (`examples/cross_crate_pubfn/` + commit `Enable cross-crate
//! `pub fn` codegen by forcing MIR encoding`) unblocked dep-crate
//! bodies — base58/bech32 helpers in `logic` use
//! `chunks_exact(...).zip(iter_mut())`.
//!
//! ## Why this is its own example
//!
//! `cross_crate_pubfn/` regression-tests MIR availability cross-crate.
//! This one is the type-translator + call-lowering surface — failures
//! reproduce single-file, no crate boundary needed. Keeping them
//! separate lets each flip independently if a future regression hits
//! only one.
//!
//! ## Long-term escape hatch
//!
//! The Alias arms are now thirteen-plus hand-rolled cases deep. The
//! principled replacement is to drop down to
//! `rustc_middle::ty::TyCtxt::normalize_erasing_regions` and delete
//! the special-cased arms. `rustc_public::ty` doesn't expose
//! normalization, so this requires threading `TyCtxt` through
//! `translate_type` — a bigger refactor, deferred.
//!
//! ## Build with
//!
//!     cargo oxide build iter_zip_chunks_exact
//!
//! Expected: green build with a real `.visible .entry` for
//! `zip_chunks_into_slice` in the emitted PTX.
//!
//! ## What this example is NOT
//!
//! - Not a cross-crate bug. The alias types are built from `core`,
//!   single-file is enough.
//! - Not about `#[device]` / `#[cuda_module]` / `#[inline]`. The
//!   trigger is generic iterator machinery, not user-helper shape.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// FAILS: the `dst.iter_mut().zip(src.chunks_exact(1))` line is
    /// the trigger. `Iterator::zip` returns
    /// `Zip<IterMut<u8>, <ChunksExact<u8> as IntoIterator>::IntoIter>`
    /// post-monomorphization, and the alias type
    /// `<ChunksExact<u8> as IntoIterator>::IntoIter` is what the
    /// translator can't handle.
    #[kernel]
    pub fn zip_chunks_into_slice(src: &[u8], mut dst: DisjointSlice<u8>) {
        let idx = thread::index_1d();
        let i = idx.get();
        // One thread does the iteration so the example stays a
        // minimal trigger; correctness of the output doesn't matter,
        // we just need codegen to attempt translation of the zip.
        if i == 0 && let Some(slot0) = dst.get_mut(idx) {
            let mut local = [0u8; 32];
            // Trigger: `iter_mut().zip(chunks_exact(...))`.
            for (out_byte, in_chunk) in local.iter_mut().zip(src.chunks_exact(1)) {
                *out_byte = in_chunk[0];
            }
            *slot0 = local[0];
        }
    }
}

fn main() {
    println!("=== iter_zip_chunks_exact ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 32;
    let host: Vec<u8> = (0..N as u8).collect();
    let src = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut dst = DeviceBuffer::<u8>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .zip_chunks_into_slice(&stream, LaunchConfig::for_num_elems(N as u32), &src, &mut dst)
        .expect("kernel launch");

    let result = dst.to_host_vec(&stream).unwrap();
    assert_eq!(result[0], host[0]);
    println!("SUCCESS: zip + chunks_exact codegen'd to PTX");
}
