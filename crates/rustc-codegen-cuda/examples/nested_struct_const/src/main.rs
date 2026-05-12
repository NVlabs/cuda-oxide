//! Regression test for struct constants whose field types are
//! themselves nested ADTs (struct/array/tuple). Real-world surface:
//! `elliptic_curve::scalar::primitive::ScalarPrimitive<C>`
//! (a struct wrapping `C::Uint` = `crypto_bigint::U256` = struct
//! wrapping `[Limb; 4]`), invoked from `ScalarPrimitive::from_bytes`
//! when reading `C::ORDER` (a `const` of the nested-struct shape).
//!
//! ## Pre-fix wall
//!
//! ```text
//! error: [rustc_codegen_cuda] Device codegen failed: PTX generation
//!        failed: Translation failed:
//!        <ScalarPrimitive<Secp256k1>>::from_bytes: Compilation error:
//!        invalid input program.
//!        Unsupported construct: Struct constant field 0 has
//!        unsupported type. Consider using inline construction
//!        instead of const.
//! ```
//!
//! ## What triggers it
//!
//! A `const` item (or `static`) of a type like:
//!
//! ```rust,ignore
//! struct Outer { inner: Inner }       // outer struct
//! struct Inner { limbs: [u64; 4] }    // nested struct wrapping array
//! const ZERO: Outer = Outer { inner: Inner { limbs: [0; 4] } };
//! ```
//!
//! When a kernel reads `ZERO`, the constant's allocation bytes feed
//! `translate_struct_constant` (the routine that parses raw byte
//! allocations back into typed SSA values). Its field-type
//! classifier currently knows about:
//!
//! * ZSTs (struct + tuple)
//! * Integer
//! * Float (f16, f32)
//! * Pointer
//!
//! Everything else — nested `MirStructType`, `MirArrayType`,
//! `MirTupleType`, `MirEnumType` — falls into the catch-all
//! `Unsupported` arm. The byte size of those types is not constant-
//! sized in the classifier's view, so the parser doesn't know how
//! many bytes to consume for that field, and bailing is the safe
//! move.
//!
//! ## Why a workaround is awkward
//!
//! The error message suggests "inline construction": replace
//! `const X: Outer = Outer { ... };` + `... = X;` with `... = Outer
//! { ... };` directly at the use site. That works when *you* author
//! the const. For `elliptic_curve::ScalarPrimitive::from_bytes`'s
//! reference to `C::ORDER`, the const lives in a dependency crate
//! you don't control.
//!
//! ## What landed
//!
//! Pulled the field-parse loop out of `translate_struct_constant`
//! into a recursive helper
//! `parse_const_value_from_bytes(ctx, ty, &bytes, …) -> (Value, _,
//! byte_size_consumed)`. The helper handles all field type kinds —
//! primitives plus nested `MirStructType`, `MirArrayType`,
//! `MirTupleType` by recursion. Byte sizes for nested types come
//! from a sibling helper `byte_size_of_dialect_mir_type` that
//! walks the dialect-mir type tree; struct sizes prefer the
//! rustc-supplied `total_size` (handles trailing padding), with
//! field-by-field summation as fallback. Field offsets come from
//! the struct's `field_offsets()` metadata when available
//! (handles padding / reordered fields).
//!
//! `translate_struct_constant` now calls the helper once per
//! top-level field instead of inlining the per-type-kind switch.
//! Enum constants intentionally remain on the unsupported path
//! (their discriminant + payload layout needs its own design pass).
//!
//! No `rustc_public` API gaps in the way — everything needed was
//! already in the dialect-mir type metadata.
//!
//! Originally surfaced from `~/vanity-miner-rs/`:
//! `logic::secp256k1_derive_public_key` calling
//! `SecretKey::from_bytes(…)`, which threads through
//! `ScalarPrimitive::<Secp256k1>::from_bytes` and reads
//! `Secp256k1::ORDER` (a `crypto_bigint::U256` const).
//!
//! ## Build with
//!
//!     cargo oxide run nested_struct_const
//!
//! Expected: kernel runs, each output slot equals the first limb
//! of `ORDER` (0xFFFF_FFFF_FFFF_FFFF for the secp256k1 mock).

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Mock of `crypto_bigint::U256` — a struct wrapping a fixed array.
/// The `#[repr(transparent)]` mirrors the real crate's layout choice
/// (though it doesn't matter for the trigger; nested ADT field is
/// what surfaces the bug).
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Wide {
    pub limbs: [u64; 4],
}

/// Mock of `ScalarPrimitive<C>` — a struct wrapping `Wide`. This is
/// the actual nested-struct shape that breaks
/// `translate_struct_constant`: field 0's type is `Wide` (a struct),
/// not a primitive.
#[derive(Clone, Copy)]
pub struct ScalarLike {
    pub inner: Wide,
}

/// The constant. Putting it at module scope (rather than inlining at
/// the use site) is what produces the MIR `ConstantKind::Allocated`
/// shape that goes through `translate_struct_constant`. Inlining
/// would go through `Aggregate(Adt)` instead, which works fine.
pub const ORDER: ScalarLike = ScalarLike {
    inner: Wide {
        limbs: [
            0xFFFF_FFFF_FFFF_FFFFu64,
            0xFFFF_FFFF_FFFF_FFFEu64,
            0xBAAE_DCE6_AF48_A03Bu64,
            0xBFD2_5E8C_D036_4141u64,
        ],
    },
};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Trigger: reading the `ORDER` const forces the constant's
    /// allocation bytes through `translate_struct_constant`, which
    /// hits the `FieldTypeKind::Unsupported` arm on field 0 (type
    /// `Wide` — a struct).
    #[kernel]
    pub fn order_first_limb(mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        if let Some(slot) = out.get_mut(idx) {
            let order = ORDER;
            *slot = order.inner.limbs[0];
        }
    }
}

fn main() {
    println!("=== nested_struct_const ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 8;
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .order_first_limb(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for &v in &result {
        assert_eq!(v, 0xFFFF_FFFF_FFFF_FFFF, "expected first limb of ORDER");
    }
    println!("SUCCESS: nested-struct const codegen'd to PTX");
}
