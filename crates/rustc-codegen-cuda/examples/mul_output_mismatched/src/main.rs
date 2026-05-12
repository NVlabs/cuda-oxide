//! Known-failure reproducer for `impl Mul<RHS> for LHS where type
//! Output != Self`. The follow-on to `examples/mul_output_adt/` —
//! that example's fix recurses on `self_ty` for ADT Self, which is
//! only correct when the impl declares `Output = Self`. This
//! reproducer exercises the *other* shape: `Output` is a different
//! ADT than `Self`.
//!
//! ## Expected failure
//!
//! ```text
//! error: [rustc_codegen_cuda] Device codegen failed: PTX generation
//!        failed: Translation failed:
//!        <kernel-symbol>: Compilation error: invalid input program.
//!        Unsupported construct: Alias type not yet supported:
//!        AliasDef(DefId { id: ..., name: "std::ops::Mul::Output" })
//! ```
//!
//! ## What triggers it
//!
//! Real-world shape: `curve25519_dalek::EdwardsBasepointTable * &Scalar
//! -> EdwardsPoint`. The impl is roughly:
//!
//! ```rust,ignore
//! impl<'a, 'b> Mul<&'b Scalar> for &'a EdwardsBasepointTable {
//!     type Output = EdwardsPoint;     // ← NOT Self
//!     fn mul(self, scalar: &'b Scalar) -> EdwardsPoint { ... }
//! }
//! ```
//!
//! Surfaced from `~/vanity-miner-rs/`:
//!
//! ```rust,ignore
//! pub fn ed25519_derive_public_key(seed: &[u8; 64]) -> [u8; 32] {
//!     let scalar = Scalar::from_bytes_mod_order(clamp_integer(input));
//!     let point = ED25519_BASEPOINT_TABLE * &scalar;   // ← triggers
//!     point.compress().to_bytes()
//! }
//! ```
//!
//! ## Why the `mul_output_adt` fix doesn't catch this
//!
//! The fix matcher (in
//! `crates/mir-importer/src/translator/types.rs:743`):
//!
//! ```rust,ignore
//! if let TyKind::RigidTy(
//!     RigidTy::Int(_) | RigidTy::Uint(_) | … | RigidTy::Adt(_, _),
//! ) = self_ty.kind() {
//!     return translate_type(ctx, self_ty);
//! }
//! ```
//!
//! recurses on `self_ty` — but `self_ty` here is `&Lhs` (i.e.
//! `RigidTy::Ref(_, Lhs, _)`), not `RigidTy::Adt(...)` directly, so
//! the matcher correctly falls through. Even if it caught `Ref`,
//! recursing would yield a pointer-to-`Lhs` type — completely wrong
//! for the actual `Output = Rhs`. The previous fix's docstring
//! flagged this exact case as "mismatched-Output impls are rare but
//! exist" and chose to hard-error rather than silently miscompile.
//!
//! ## What would it take to fix
//!
//! Real trait-impl resolution. Given the alias type
//! `<&Lhs as Mul<&Scalar>>::Output`, look up the impl whose Self
//! type matches `&Lhs` and whose trait `Args` match `&Scalar`, then
//! read the impl's `type Output = X;` declaration. `rustc_public`
//! exposes `ImplDef::associated_items` and `ImplDef::trait_impl`,
//! but no direct "find the impl that monomorphizes this projection"
//! query — the rustc-internal equivalent is
//! `TyCtxt::normalize_erasing_regions`, not surfaced in stable MIR.
//!
//! Building it in-tree would require enumerating impls of `Mul`,
//! matching each impl's Self type against `self_ty` and its trait
//! args against the projection's args, then reading the matched
//! impl's `Output`. Doable but non-trivial — type unification is
//! exactly what's missing from `rustc_public`'s API for cases like
//! this.
//!
//! ## User-side workaround
//!
//! Replace the operator with the explicit method call. Every
//! mismatched-Output `Mul` impl in practice has a named-method
//! sibling that takes concrete types:
//!
//! ```rust,ignore
//! // before:
//! let point = ED25519_BASEPOINT_TABLE * &scalar;
//! // after:
//! let point = EdwardsPoint::mul_base(&scalar);
//! ```
//!
//! The named-method API is the recommended path for device code in
//! curve25519-dalek's docs anyway (the operator desugar drags in
//! more machinery via the trait method's blanket return type than
//! the direct method needs).
//!
//! Same root cause — any `impl Mul<Rhs> for Lhs where type Output =
//! Other` trips this. Common across crypto:
//!
//! * `&EdwardsBasepointTable * &Scalar -> EdwardsPoint`
//!   (curve25519-dalek).
//! * `&ProjectivePoint * &Scalar -> ProjectivePoint`
//!   (k256 / p256 — here Output happens to be Self,
//!   `mul_output_adt`'s fix covers it).
//! * `&Matrix3 * &Vector3 -> Vector3` (nalgebra).
//! * `Duration * u32 -> Duration` (std — primitive RHS).
//!
//! ## Build with
//!
//!     cargo oxide build mul_output_mismatched
//!
//! Expected: build error from the mir-importer's type translator —
//! `Alias type not yet supported: AliasDef(DefId { … name:
//! "std::ops::Mul::Output" })`.
//!
//! ## What this example is NOT
//!
//! - Not the same as `examples/mul_output_adt/` (`Output = Self`,
//!   recoverable by recursion on `self_ty`).
//! - Not about primitive arithmetic.
//! - Not about closures, iterators, or function-pointer aliases.

use core::ops::Mul;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// "Basepoint table" stand-in — corresponds to
/// `curve25519_dalek::EdwardsBasepointTable`.
#[derive(Clone, Copy)]
pub struct Table {
    pub k: u32,
}

/// Scalar — corresponds to `curve25519_dalek::Scalar`.
#[derive(Clone, Copy)]
pub struct Scalar {
    pub n: u32,
}

/// `Mul`'s `Output` — corresponds to `EdwardsPoint`. Crucially, this
/// is *not* `Table`, so recursing-on-Self in the alias arm would
/// produce a wrong type.
#[derive(Clone, Copy)]
pub struct Point {
    pub lo: u32,
    pub hi: u32,
}

impl Mul<Scalar> for Table {
    type Output = Point;
    // `#[inline(never)]` keeps the alias alive past
    // monomorphization. Without it the optimizer folds away the
    // call and the bug doesn't surface.
    #[inline(never)]
    fn mul(self, rhs: Scalar) -> Point {
        Point {
            lo: self.k.wrapping_mul(rhs.n),
            hi: self.k.wrapping_add(rhs.n),
        }
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Trigger: `table * scalar` calls `<Table as Mul<Scalar>>::mul`,
    /// whose return type is the alias `<Table as Mul<Scalar>>::Output
    /// = Point`. The alias arm doesn't know how to resolve it without
    /// real trait-impl normalization.
    #[kernel]
    pub fn table_scalar_mul(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i < input.len()
        {
            let table = Table { k: input[i] };
            let scalar = Scalar { n: input[i].wrapping_add(7) };
            let point = table * scalar;
            *slot = point.lo ^ point.hi;
        }
    }
}

fn main() {
    println!("=== mul_output_mismatched ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 16;
    let host: Vec<u32> = (0..N as u32).map(|n| n * 13 + 5).collect();
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .table_scalar_mul(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let k = host[i];
        let n = k.wrapping_add(7);
        let lo = k.wrapping_mul(n);
        let hi = k.wrapping_add(n);
        let expected = lo ^ hi;
        assert_eq!(result[i], expected, "thread {} mismatch", i);
    }
    println!("SUCCESS: <Table as Mul<Scalar>>::Output codegen'd to PTX");
}
