/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Attributes belonging to the MIR dialect.

use std::hash::{Hash, Hasher};

use pliron::attribute::Attribute;
use pliron::builtin::attr_interfaces::{FloatAttr, TypedAttrInterface};
use pliron::context::Context;
use pliron::derive::{attr_interface_impl, pliron_attr};
use pliron::r#type::{TypeHandle, Typed};
use pliron::utils::apfloat::{self, Float, GetSemantics};

use crate::types::MirFP16Type;

/// MIR cast kind — preserves the semantic intent of the cast from Rust MIR.
///
/// The lowering dispatches on this to pick the correct LLVM instruction,
/// rather than guessing from source/destination types.
#[pliron_attr(name = "mir.cast_kind", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum MirCastKindAttr {
    IntToInt,
    IntToFloat,
    FloatToInt,
    FloatToFloat,
    PtrToPtr,
    FnPtrToPtr,
    PointerExposeAddress,
    PointerWithExposedProvenance,
    Transmute,
    PointerCoercionUnsize,
    PointerCoercionMutToConst,
    PointerCoercionArrayToPointer,
    PointerCoercionReifyFnPointer,
    PointerCoercionUnsafeFnPointer,
    PointerCoercionClosureFnPointer,
    Subtype,
}

/// Boolean attribute for reference mutability.
///
/// Replaces the overloaded `IntegerAttr` pattern with a self-documenting
/// domain-specific attribute.
#[pliron_attr(name = "mir.mutability", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct MutabilityAttr(pub bool);

/// Structural field index for aggregate access ops
/// (`mir.extract_field`, `mir.insert_field`, `mir.field_addr`, `mir.enum_payload`).
#[pliron_attr(name = "mir.field_index", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct FieldIndexAttr(pub u32);

/// Enum variant index for variant-level ops
/// (`mir.construct_enum`, `mir.enum_payload`).
#[pliron_attr(name = "mir.variant_index", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct VariantIndexAttr(pub u32);

/// The unroll factor carried by a [`MirUnrollHintOp`](crate::ops::MirUnrollHintOp).
///
/// `#[unroll]` / `#[unroll(N)]` written on a loop makes the frontend plant a
/// `mir.unroll_hint` op inside that loop's body; this attribute is the factor it
/// carries, and the loop-unroll pass reads it to decide how to unroll that one
/// loop:
///
/// * `0` -- **full unroll**: if the loop's trip count is a compile-time
///   constant, unroll it completely, so the induction variable becomes a literal
///   in each copy (this is what lets index arithmetic such as `i & 3` fold to a
///   constant).
/// * `n >= 2` -- **unroll by `n`**: do `n` copies of the body per trip, leaving
///   a remainder loop when `n` does not divide the trip count.
#[pliron_attr(name = "mir.unroll", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct UnrollAttr(pub u32);

/// Niche encoding for a `Cast(Transmute)` whose destination is a
/// niche-optimised enum.
///
/// rustc stores `Option<NonZeroT>`, `Option<&T>`, `Option<Box<T>>`,
/// `Option<NonNull<T>>`, `Option<bool>`, `Option<char>`, etc. as a single
/// scalar where one forbidden bit pattern of the inner type stands in for
/// the niche variant (typically `None`) and any other bit pattern means
/// the active variant (typically `Some(x)`).
///
/// When mir-lower has to rebuild the un-niched `{ discriminant, payload }`
/// aggregate from such a scalar it needs:
///
/// * `niche_start` -- the bit pattern that signals the niche variant.
/// * `niche_variant_idx` -- the discriminant value for the niche variant.
/// * `untagged_variant_idx` -- the discriminant value for the active variant.
///
/// All three come from `ty.layout().shape().variants` when the tag
/// encoding is `TagEncoding::Niche`. `niche_start` is stored as `u64`
/// rather than the `u128` rustc-public exposes: niched scalars are at
/// most 64 bits wide, so the bit pattern always fits. The importer
/// rejects wider niches up front rather than truncating silently.
#[pliron_attr(name = "mir.niche_encoding", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct NicheEncodingAttr {
    pub niche_start: u64,
    pub niche_variant_idx: u32,
    pub untagged_variant_idx: u32,
}

/// IEEE 754 binary16 floating-point attribute for Rust MIR `f16` constants.
#[pliron_attr(name = "mir.fp16_attr", format = "$0", verifier = "succ")]
#[derive(PartialEq, Clone, Debug)]
pub struct MirFP16Attr(pub apfloat::Half);

impl MirFP16Attr {
    pub fn from_bits(bits: u16) -> Self {
        MirFP16Attr(<apfloat::Half as Float>::from_bits(bits as u128))
    }

    pub fn to_bits(&self) -> u16 {
        self.0.to_bits() as u16
    }
}

impl Hash for MirFP16Attr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.to_bits().hash(state);
    }
}

impl Typed for MirFP16Attr {
    fn get_type(&self, ctx: &Context) -> TypeHandle {
        MirFP16Type::get(ctx).into()
    }
}

#[attr_interface_impl]
impl TypedAttrInterface for MirFP16Attr {
    fn get_type(&self, ctx: &Context) -> TypeHandle {
        MirFP16Type::get(ctx).into()
    }
}

#[attr_interface_impl]
impl FloatAttr for MirFP16Attr {
    fn get_inner(&self) -> &dyn apfloat::DynFloat {
        &self.0
    }

    fn build_from(&self, df: Box<dyn apfloat::DynFloat>) -> Box<dyn FloatAttr> {
        let df = df
            .downcast::<apfloat::Half>()
            .expect("Expected a half precision float");
        Box::new(MirFP16Attr(*df))
    }

    fn get_semantics(&self) -> apfloat::Semantics {
        Self::get_semantics_static()
    }

    fn get_semantics_static() -> apfloat::Semantics
    where
        Self: Sized,
    {
        <apfloat::Half as GetSemantics>::get_semantics()
    }
}

pub fn register(ctx: &mut Context) {
    MirCastKindAttr::register(ctx);
    MutabilityAttr::register(ctx);
    FieldIndexAttr::register(ctx);
    VariantIndexAttr::register(ctx);
    UnrollAttr::register(ctx);
    NicheEncodingAttr::register(ctx);
    MirFP16Attr::register(ctx);
}
