//! Regression test for `impl Mul<RHS> for LHS where type Output !=
//! Self` — the mismatched-Output trait projection shape. Sibling of
//! `examples/mul_output_adt/`, which covers `Output = Self`.
//!
//! ## Pre-fix wall
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
//! ## Why the hand-rolled matchers can't catch this
//!
//! The arith-output arm (`mul_output_adt/`'s fix) recurses on
//! `self_ty` when Self is an ADT and `args[0] == args[1]`. That
//! correctly handles `impl Mul for T { type Output = T; … }` but
//! deliberately refuses `impl Mul<U> for T { type Output = V; … }`
//! because substituting Self would be a real miscompile. Same shape
//! every other hand-rolled matcher used to bail on:
//! `IntoIterator::IntoIter`, `GenericSequence::Sequence`, etc. — all
//! premised on `Output = Self`. Mismatched-Output projections
//! genuinely need real normalization.
//!
//! ## What landed
//!
//! A universal fallback at the bottom of the alias arm, replacing
//! the open-ended "every new third-party trait needs its own arm
//! here" treadmill. Instead of pattern-matching trait names, we
//! resolve any unhandled projection by going through
//! `Instance::resolve` on the parent trait's method:
//!
//! 1. Walk `all_trait_decls()` and find the trait whose
//!    `associated_items()` contains the alias's `def_id`. Compare
//!    inner `DefId`s directly — `AssocDef` wraps `pub DefId`.
//! 2. Pick any method on that trait
//!    (`AssocKind::Fn { has_self: true }`).
//! 3. Rewrap the method's `AssocDef` as a `FnDef`. The macros that
//!    generate both expose `pub DefId`; rustc's queries are
//!    def-id-driven, so the wrapper type doesn't matter to
//!    `Instance::resolve`.
//! 4. `Instance::resolve(method_fn_def, alias_ty.args)`. rustc looks
//!    up the matching impl and monomorphizes `Self::Output`.
//! 5. Read the resolved instance's signature output type and recurse
//!    into `translate_type`.
//!
//! Why this works without `TyCtxt::normalize_erasing_regions`:
//! `Instance::resolve` already triggers rustc's impl resolution
//! machinery internally. The monomorphized instance carries
//! post-normalization types in its signature. We fish the resolved
//! type out of the signature rather than calling `normalize_*`
//! directly.
//!
//! The pre-existing hand-rolled matchers (FnOnce::Output,
//! Index::Output on SharedArray, arith-output, IntoIterator::IntoIter,
//! Iterator::Item, GenericSequence::Sequence) remain above the
//! fallback because they handle their cases without the
//! `all_trait_decls` walk and without requiring `Instance::resolve`'s
//! monomorphization preconditions — faster path for the common
//! cases.
//!
//! Bonus catch: this same fallback should also resolve any future
//! mismatched-Output trait reaching device codegen without needing a
//! new matcher arm. Common cases:
//!
//! * `&EdwardsBasepointTable * &Scalar -> EdwardsPoint`
//!   (curve25519-dalek, the original trigger).
//! * `&Matrix3 * &Vector3 -> Vector3` (nalgebra).
//! * `Duration * u32 -> Duration` (std — primitive RHS).
//!
//! ## Build with
//!
//!     cargo oxide run mul_output_mismatched
//!
//! Expected: kernel runs, output matches the host-side
//! `(k.wrapping_mul(k+7)) ^ (k.wrapping_add(k+7))` per thread.
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
