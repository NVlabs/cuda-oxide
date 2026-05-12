//! Regression test for `<T as GenericSequence<…>>::Sequence` aliases —
//! the third-party-trait associated type from `generic_array`. The
//! reproducer uses a local mock `GenericSequence` trait (no
//! `generic_array` dep) so the matcher substring
//! `"GenericSequence::Sequence"` catches it exactly as it would the
//! real `generic_array::sequence::GenericSequence::Sequence`.
//!
//! ## Pre-fix wall
//!
//! ```text
//! error: [rustc_codegen_cuda] Device codegen failed: PTX generation
//!        failed: Translation failed:
//!        <kernel-symbol>: Compilation error: invalid input program.
//!        Unsupported construct: Alias type not yet supported:
//!        AliasDef(DefId { id: ...,
//!          name: "generic_array::sequence::GenericSequence::Sequence" })
//! ```
//!
//! ## What triggers it
//!
//! Any trait method whose return type is the trait's associated
//! type, called through dispatch that prevents the optimizer from
//! folding the alias before monomorphization is fully resolved.
//!
//! ```rust,ignore
//! trait GenericSequence<T> {
//!     type Sequence;
//!     fn into_out(self) -> Self::Sequence;     // ← return is Self::Sequence
//! }
//!
//! impl GenericSequence<u32> for Arr {
//!     type Sequence = Arr;
//!     #[inline(never)]
//!     fn into_out(self) -> Self::Sequence { self }
//! }
//!
//! GenericSequence::into_out(arr);  // <Arr as GenericSequence<u32>>::Sequence alias survives
//! ```
//!
//! ## Where the gap lives
//!
//! `crates/mir-importer/src/translator/types.rs`'s alias arm dispatches
//! on `def_name` substrings:
//!
//! * `IntoIterator::IntoIter` → recurse on Self (ADT).
//! * `Iterator::Item` → adapter-specific item type table.
//! * `FnOnce::Output` / `FnMut::Output` / `Fn::Output` → fn-sig of
//!   the closure.
//! * `Index::Output` / `IndexMut::Output` on `SharedArray<T, N>` → `T`.
//! * `Mul::Output` / `Add::Output` / … on primitive or `Self == RHS`
//!   ADT → recurse on Self.
//!
//! Anything else falls through to the catch-all
//! `"Alias type not yet supported: …"` error. Every new third-party
//! trait whose associated type ends up in a device-reachable MIR
//! signature surfaces a new wall — each fix is ~5 lines but the
//! compound bookkeeping is real.
//!
//! ## What landed
//!
//! Added a `"GenericSequence::Sequence"` arm in the alias matcher,
//! recursing on `self_ty` for ADT Self. Same shape as the
//! `IntoIterator::IntoIter` and arith-output handlers above it. The
//! canonical impl is `impl<T, N> GenericSequence<T> for
//! GenericArray<T, N> { type Sequence = Self; … }` and every reachable
//! device impl follows the same `Sequence = Self` pattern.
//!
//! Doesn't generalize — every new third-party trait whose associated
//! type reaches a device-reachable MIR signature still needs its own
//! arm here. Real fix would thread `rustc_middle::ty::TyCtxt::
//! normalize_erasing_regions` through `translate_type`. Out of scope
//! for this loop.
//!
//! ## User-side workaround for vanity-miner-rs
//!
//! Stay off `k256` / `elliptic_curve` for the device path entirely.
//! That whole crate family is heavily `generic_array`-backed — every
//! "get the bytes out" API path goes through `GenericSequence`,
//! `ArrayLength`, and a tower of associated types. Even with the
//! one-line fix here, the next call past `to_encoded_point` will
//! surface another alias wall. Probable practical answer is to
//! hand-roll the secp256k1 point→bytes step on the device (the
//! field-element bytes for an `AffinePoint` are already concrete
//! `[u8; 32]` if you reach into `k256::arithmetic` directly).
//!
//! Originally surfaced from `~/vanity-miner-rs/`:
//! `logic::secp256k1_derive_public_key` calling
//! `public_key.to_encoded_point(true)`.
//!
//! ## Build with
//!
//!     cargo oxide run generic_sequence_alias
//!
//! Expected: kernel runs, each output equals `input[i].wrapping_add(1)`.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Mock of `generic_array::sequence::GenericSequence` — minimal trait
/// with an associated type returned by a method. Single type param so
/// the alias args shape matches `GenericSequence<T>::Sequence`.
pub trait GenericSequence<T> {
    type Sequence;
    fn into_out(self) -> Self::Sequence;
}

/// Mock of `GenericArray<T, N>`. One field so it has non-trivial
/// layout but stays small enough that the optimizer might be tempted
/// to inline — `#[inline(never)]` on the impl method keeps the alias
/// alive.
#[derive(Clone, Copy)]
pub struct Arr {
    pub v: u32,
}

impl GenericSequence<u32> for Arr {
    type Sequence = Arr;
    #[inline(never)]
    fn into_out(self) -> Self::Sequence {
        Arr { v: self.v.wrapping_add(1) }
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Trigger: `GenericSequence::into_out(arr)` returns `<Arr as GenericSequence<u32>>::Sequence`.
    /// With `#[inline(never)]` on the impl, the alias survives into
    /// the call's return-type slot in the kernel's MIR, where the
    /// type translator hits the catch-all error.
    #[kernel]
    pub fn seq_call(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i < input.len()
        {
            let arr = Arr { v: input[i] };
            let next = GenericSequence::into_out(arr);
            *slot = next.v;
        }
    }
}

fn main() {
    println!("=== generic_sequence_alias ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 16;
    let host: Vec<u32> = (0..N as u32).map(|n| n * 17 + 11).collect();
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .seq_call(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let expected = host[i].wrapping_add(1);
        assert_eq!(result[i], expected, "thread {} mismatch", i);
    }
    println!("SUCCESS: <Arr as GenericSequence<u32>>::Sequence codegen'd to PTX");
}
