//! Known-failure reproducer for `Iterator::zip` + `chunks_exact`
//! hitting an un-normalized alias type in the mir-importer's type
//! translator.
//!
//! ## Expected failure
//!
//! ```text
//! error: [rustc_codegen_cuda] Device codegen failed: PTX generation
//!        failed: Translation failed: <...>::Iterator::zip::<ChunksExact<u8>>:
//!        Compilation error: invalid input program.
//!        Unsupported construct: Alias type not yet supported:
//!        AliasDef(DefId { ..., name: "std::iter::IntoIterator::IntoIter" })
//! ```
//!
//! ## What triggers it
//!
//! The kernel does:
//!
//! ```rust,ignore
//! for (slot, chunk) in dst.iter_mut().zip(src.chunks_exact(1)) {
//!     *slot = chunk[0];
//! }
//! ```
//!
//! `Iterator::zip` has signature
//! `fn zip<U: IntoIterator>(self, other: U) -> Zip<Self, U::IntoIter>`.
//! Once monomorphized with `U = ChunksExact<u8>` the return type
//! contains `<ChunksExact<u8> as IntoIterator>::IntoIter` — an
//! *associated-type alias*. In normal rustc codegen this is
//! normalized to the concrete `ChunksExact<u8>` (chunks_exact is its
//! own iterator) before the codegen backend ever sees it.
//!
//! cuda-oxide reads types through the `rustc_public::ty` API
//! ("stable mir"), which appears not to normalize aliases on read.
//! The MIR translator hits the `Alias(Projection, …)` arm in
//! `crates/mir-importer/src/translator/types.rs:644` and tries to
//! match `def_name` against a small set of hand-rolled cases —
//! `FnOnce::Output`, `Index::Output` on `SharedArray`, and
//! arithmetic-trait outputs on primitives. `IntoIterator::IntoIter`
//! matches none of them, so the translator bails with `Alias type
//! not yet supported`.
//!
//! Originally surfaced from `~/vanity-miner-rs/logic/` after the
//! cross-crate MIR fix
//! (`examples/cross_crate_pubfn/` + commit `Enable cross-crate
//! `pub fn` codegen by forcing MIR encoding`) unblocked dep-crate
//! bodies — base58/bech32 helpers in `logic` use
//! `chunks_exact(...).zip(iter_mut())` patterns. The fix to that
//! commit didn't introduce the bug; it just unmasked it. Pre-fix
//! the cross-crate body was silently dropped before the type
//! translator ever ran on it.
//!
//! ## Why this is its own example
//!
//! `cross_crate_pubfn/` is a regression test for *MIR availability*
//! cross-crate. This one is a regression test for *type translation*
//! and is independent of crate boundaries — the failure reproduces
//! single-file, all the alias-type plumbing is inside `core`. Keeping
//! them separate lets each example fail and recover independently.
//!
//! ## What a fix needs to do
//!
//! Two plausible directions, both in
//! `crates/mir-importer/src/translator/types.rs`:
//!
//! 1. **Generic normalization.** Before hitting the Alias arm at
//!    L644, resolve `Alias(Projection, …)` via the rustc public-mir
//!    normalization API (whichever flavour `rustc_public::ty`
//!    exposes — there is probably a `normalize` / `try_normalize`
//!    method on `Ty` or a free fn taking a `TypingEnv`). Once
//!    normalized, dispatch the concrete type through the existing
//!    `translate_type` chain. This drops all the hand-rolled cases
//!    (FnOnce::Output, Index::Output, arith::Output) — they all
//!    become consequences of normalization. Right shape if the API
//!    exists.
//!
//! 2. **Targeted handler.** Add an `IntoIterator::IntoIter` arm
//!    next to the others: extract `self_ty` from `alias_ty.args`,
//!    note that for any `Iterator` self type
//!    `<Self as IntoIterator>::IntoIter = Self`, recurse with
//!    `self_ty`. Narrow, doesn't fix the underlying gap, but trivial
//!    to write and unblocks this specific surface. Next alias type
//!    in a different crate hits the same wall.
//!
//! Option 1 is the right call. The hand-rolled arms are themselves
//! evidence that ad-hoc dispatch isn't sustainable.
//!
//! ## Build with
//!
//!     cargo oxide build iter_zip_chunks_exact
//!
//! Expected: build error from the mir-importer:
//! `Alias type not yet supported: AliasDef(... "std::iter::IntoIterator::IntoIter")`.
//!
//! ## What this example is NOT
//!
//! - Not a cross-crate bug. Adding a sibling crate doesn't change
//!   the failure; the alias type is built from `core` either way.
//! - Not about `#[device]` / `#[cuda_module]` / `#[inline]`. The
//!   bug fires on a one-file kernel with no helpers.

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
