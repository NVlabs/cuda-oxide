//! Known-failure reproducer for `<T as core::ops::Mul>::Output` (and
//! related arithmetic-trait Output aliases) surviving the type
//! translator when `T` is a user-defined struct.
//!
//! ## Expected failure
//!
//! ```text
//! error: [rustc_codegen_cuda] Device codegen failed: PTX generation
//!        failed: Translation failed: <kernel-symbol>: Compilation
//!        error: invalid input program.
//!        Unsupported construct: Alias type not yet supported:
//!        AliasDef(DefId { id: ..., name: "std::ops::Mul::Output" })
//! ```
//!
//! ## What triggers it
//!
//! Multiplying two values of a custom struct that implements
//! `core::ops::Mul`. The kernel does:
//!
//! ```rust,ignore
//! let z: Fe = x * y;          // ⇒ <Fe as Mul>::mul(x, y)
//! ```
//!
//! `<Fe as Mul>::mul`'s return type is the alias
//! `<Fe as Mul>::Output`. cuda-oxide's alias arm at
//! `crates/mir-importer/src/translator/types.rs` (the
//! `TyKind::Alias(Projection, …)` branch) recognizes
//! `std::ops::Mul::Output` / `core::ops::Mul::Output` only when the
//! Self type is a primitive (`Int` / `Uint` / `Float` / `Bool` /
//! `Char`):
//!
//! ```rust,ignore
//! // crates/mir-importer/src/translator/types.rs:743
//! if is_arith_output {
//!     let args = &alias_ty.args.0;
//!     if let Some(GenericArgKind::Type(self_ty)) = args.first() {
//!         if let TyKind::RigidTy(RigidTy::Int(_) | RigidTy::Uint(_) | ...)
//!         //                                                       ^^^
//!         //                                       no ADT arm here
//!         = self_ty.kind() {
//!             return translate_type(ctx, self_ty);
//!         }
//!     }
//! }
//! ```
//!
//! For `self_ty = Adt(Fe, _)`, the inner branch doesn't match and the
//! alias arm falls through every other recognized case (`IntoIter`,
//! `Item`, `FnOnce::Output`, `Index::Output`, …) before hitting the
//! catch-all `"Alias type not yet supported"` at the bottom of the
//! arm.
//!
//! Same root cause — any of `Add::Output`, `Sub::Output`,
//! `Div::Output`, `Rem::Output`, `BitAnd::Output`, `BitOr::Output`,
//! `BitXor::Output`, `Shl::Output`, `Shr::Output`, `Neg::Output`,
//! `Not::Output` on a user struct hits this. Same for `Mul<RHS, …>`
//! with a different `RHS` if the impl declares `type Output = Self`.
//!
//! ## What it would take to fix
//!
//! Extend the `is_arith_output` arm to recurse on ADT self types in
//! the same shape as the `IntoIterator::IntoIter` handler immediately
//! below it. The blanket pattern for arithmetic traits on user types
//! is `impl Mul for T { type Output = T; … }` — the Output associated
//! type is `Self`. Recursing on the first generic arg covers it:
//!
//! ```rust,ignore
//! if is_arith_output {
//!     let args = &alias_ty.args.0;
//!     if let Some(GenericArgKind::Type(self_ty)) = args.first() {
//!         if matches!(self_ty.kind(),
//!             TyKind::RigidTy(RigidTy::Int(_) | RigidTy::Uint(_) | …
//!                             | RigidTy::Adt(_, _))) {
//!             return translate_type(ctx, self_ty);
//!         }
//!     }
//! }
//! ```
//!
//! Risk: an impl like `impl Mul<Scalar> for Point { type Output =
//! Vec3; … }` declares `Output != Self`. We'd silently substitute
//! Self and produce wrong types. Without normalization access
//! (`rustc_middle::ty::TyCtxt::normalize_erasing_regions`), there's
//! no perfectly safe move. Pragmatic stance: the overwhelming
//! majority of `impl Mul/Add/Sub/Div/Rem for T` declare `type Output
//! = T` (every standard numeric type in `num`, `nalgebra`, every
//! curve-point type in `ed25519` / `curve25519-dalek` / `k256` /
//! `secp256k1`). Mismatched-Output impls are rare. If they surface
//! we hard-error one layer down (return-type vs call-site mismatch)
//! rather than producing wrong PTX.
//!
//! Originally surfaced from `~/vanity-miner-rs/`: the
//! `logic::ed25519_derive_public_key` path multiplies curve-point
//! field elements, which monomorphizes to
//! `<FieldElement2625 as Mul>::mul -> <… as Mul>::Output`.
//!
//! ## Build with
//!
//!     cargo oxide build mul_output_adt
//!
//! Expected: build error from the mir-importer's type translator —
//! `Alias type not yet supported: AliasDef(DefId { … name:
//! "std::ops::Mul::Output" })`.
//!
//! ## What this example is NOT
//!
//! - Not about primitive-typed arithmetic (`u32 * u32` etc.) — that
//!   path is already handled by the primitive arm.
//! - Not about `Mul` with a different RHS than Self — the kernel
//!   uses the simplest shape (`impl Mul for Fe`).
//! - Not about `Iterator::Item` or `IntoIterator::IntoIter`
//!   (`iter_zip_chunks_exact`'s territory), `FnOnce::Output`
//!   (closures' territory), or `Index::Output` (`SharedArray`'s
//!   territory). The trigger here is purely the arithmetic-Output
//!   alias on an ADT.

use core::ops::Mul;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Minimal "field element" stand-in for ed25519-style arithmetic.
/// Two `u32`s so the layout is non-trivial (struct with multiple
/// fields), but the math is trivial (component-wise multiply).
#[derive(Clone, Copy)]
pub struct Fe {
    pub lo: u32,
    pub hi: u32,
}

impl Mul for Fe {
    type Output = Fe;
    // `#[inline(never)]` is load-bearing — without it the optimizer
    // inlines `mul` at the call site and the `<Fe as Mul>::Output`
    // alias is folded into `Fe` before reaching the type translator.
    // vanity-miner-rs's `ed25519_derive_public_key` triggers the bug
    // because its `Mul` impl is large enough (and lives in a
    // separate crate) that inlining doesn't happen.
    #[inline(never)]
    fn mul(self, rhs: Fe) -> Fe {
        // Wrapping component-wise multiply — keeps the body
        // primitive-only so the only un-handled type artifact is the
        // outer `<Fe as Mul>::Output` return alias.
        Fe {
            lo: self.lo.wrapping_mul(rhs.lo),
            hi: self.hi.wrapping_mul(rhs.hi),
        }
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Trigger: `a * b` on `Fe` returns `<Fe as Mul>::Output`. The
    /// alias survives monomorphization in stable MIR and the type
    /// translator has no arm for it when Self is an ADT.
    #[kernel]
    pub fn fe_mul(input_lo: &[u32], input_hi: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i < input_lo.len()
            && i < input_hi.len()
        {
            let a = Fe {
                lo: input_lo[i],
                hi: input_hi[i],
            };
            let b = Fe {
                lo: input_hi[i],
                hi: input_lo[i],
            };
            let c = a * b;
            *slot = c.lo ^ c.hi;
        }
    }
}

fn main() {
    println!("=== mul_output_adt ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 32;
    let lo: Vec<u32> = (0..N as u32).map(|n| n * 3 + 1).collect();
    let hi: Vec<u32> = (0..N as u32).map(|n| n * 5 + 2).collect();

    let input_lo = DeviceBuffer::from_host(&stream, &lo).unwrap();
    let input_hi = DeviceBuffer::from_host(&stream, &hi).unwrap();
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .fe_mul(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input_lo,
            &input_hi,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let a_lo = lo[i];
        let a_hi = hi[i];
        let b_lo = hi[i];
        let b_hi = lo[i];
        let c_lo = a_lo.wrapping_mul(b_lo);
        let c_hi = a_hi.wrapping_mul(b_hi);
        let expected = c_lo ^ c_hi;
        assert_eq!(result[i], expected, "thread {} mismatch", i);
    }
    println!("SUCCESS: <Fe as Mul>::Output codegen'd to PTX");
}
