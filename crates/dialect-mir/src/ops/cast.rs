/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR cast operations.
//!
//! This module defines type conversion operations for the MIR dialect.

use crate::attributes::{MirCastKindAttr, NicheEncodingAttr};
use crate::types::MirPtrType;
use pliron::{
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface, OneOpdInterface, OneResultInterface},
        type_interfaces::FloatTypeInterface,
        types::IntegerType,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::{Typed, type_impls},
    verify_err,
};
use pliron_derive::pliron_op;

// ============================================================================
// MirCastOp
// ============================================================================

/// MIR cast operation — type conversion with preserved semantic intent.
///
/// The `cast_kind` attribute records the MIR `CastKind` so the lowering can
/// dispatch correctly (e.g. `Transmute` → `bitcast`/`extractvalue`, not `sitofp`).
///
/// # Operands
///
/// | Name    | Type     |
/// |---------|----------|
/// | `value` | Any type |
///
/// # Results
///
/// | Name  | Type        |
/// |-------|-------------|
/// | `res` | Target type |
///
/// # Attributes
///
/// | Name              | Type                 | Description                                                                                  |
/// |-------------------|----------------------|----------------------------------------------------------------------------------------------|
/// | `cast_kind`       | `MirCastKindAttr`    | Semantic cast kind from MIR.                                                                 |
/// | `niche_encoding`  | `NicheEncodingAttr`  | Optional. Niche layout for a `Transmute` whose destination is a niche-optimised enum.        |
///
/// `niche_encoding` is only meaningful when `cast_kind == Transmute`. The
/// importer attaches it whenever rustc tells us the destination type uses
/// `TagEncoding::Niche`; mir-lower reads it to rebuild the un-niched
/// `{ discriminant, payload }` aggregate explicitly.
///
/// # Verification
///
/// Requires `cast_kind` and checks that operand/result types match the kind:
/// - **IntToInt** / **FloatToFloat**: operand and result are both integer or both float types.
/// - **IntToFloat** / **FloatToInt**: operand and result are the appropriate integer/float pair.
/// - **PointerExposeAddress**: operand is pointer, result is integer.
/// - **PointerWithExposedProvenance**: operand is integer, result is pointer.
/// - **PtrToPtr**, **Transmute**, **PointerCoercion\***, **Subtype**: no extra type check (lowering handles ptr/struct/tuple etc.).
///
/// If `niche_encoding` is present, `cast_kind` must be `Transmute` (rejected otherwise).
#[pliron_op(
    name = "mir.cast",
    format,
    interfaces = [NOpdsInterface<1>, OneOpdInterface, NResultsInterface<1>, OneResultInterface],
    attributes = (cast_kind: MirCastKindAttr, niche_encoding: NicheEncodingAttr)
)]
pub struct MirCastOp;

impl MirCastOp {
    /// Create a new MirCastOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirCastOp { op }
    }
}

impl Verify for MirCastOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let loc = op.loc();

        // Structural: one operand, one result (OneOpdInterface / OneResultInterface guarantee count).
        let opd_val = op.get_operand(0);
        let res_val = op.get_result(0);
        let opd_ty = opd_val.get_type(ctx);
        let res_ty = res_val.get_type(ctx);
        let opd_ty_obj = opd_ty.deref(ctx);
        let res_ty_obj = res_ty.deref(ctx);

        let cast_kind = match self.get_attr_cast_kind(ctx) {
            Some(r) => r.clone(),
            None => return verify_err!(loc, "MirCastOp must have a cast_kind attribute"),
        };

        match &cast_kind {
            MirCastKindAttr::IntToInt => {
                if opd_ty_obj.downcast_ref::<IntegerType>().is_none() {
                    return verify_err!(loc, "IntToInt cast requires integer operand type");
                }
                if res_ty_obj.downcast_ref::<IntegerType>().is_none() {
                    return verify_err!(loc, "IntToInt cast requires integer result type");
                }
            }
            MirCastKindAttr::IntToFloat => {
                if opd_ty_obj.downcast_ref::<IntegerType>().is_none() {
                    return verify_err!(loc, "IntToFloat cast requires integer operand type");
                }
                if !type_impls::<dyn FloatTypeInterface>(&*res_ty_obj) {
                    return verify_err!(loc, "IntToFloat cast requires float result type");
                }
            }
            MirCastKindAttr::FloatToInt => {
                if !type_impls::<dyn FloatTypeInterface>(&*opd_ty_obj) {
                    return verify_err!(loc, "FloatToInt cast requires float operand type");
                }
                if res_ty_obj.downcast_ref::<IntegerType>().is_none() {
                    return verify_err!(loc, "FloatToInt cast requires integer result type");
                }
            }
            MirCastKindAttr::FloatToFloat => {
                if !type_impls::<dyn FloatTypeInterface>(&*opd_ty_obj) {
                    return verify_err!(loc, "FloatToFloat cast requires float operand type");
                }
                if !type_impls::<dyn FloatTypeInterface>(&*res_ty_obj) {
                    return verify_err!(loc, "FloatToFloat cast requires float result type");
                }
            }
            MirCastKindAttr::PointerExposeAddress => {
                if opd_ty_obj.downcast_ref::<MirPtrType>().is_none() {
                    return verify_err!(
                        loc,
                        "PointerExposeAddress cast requires pointer operand type"
                    );
                }
                if res_ty_obj.downcast_ref::<IntegerType>().is_none() {
                    return verify_err!(
                        loc,
                        "PointerExposeAddress cast requires integer result type"
                    );
                }
            }
            MirCastKindAttr::PointerWithExposedProvenance => {
                if opd_ty_obj.downcast_ref::<IntegerType>().is_none() {
                    return verify_err!(
                        loc,
                        "PointerWithExposedProvenance cast requires integer operand type"
                    );
                }
                if res_ty_obj.downcast_ref::<MirPtrType>().is_none() {
                    return verify_err!(
                        loc,
                        "PointerWithExposedProvenance cast requires pointer result type"
                    );
                }
            }
            // PtrToPtr, FnPtrToPtr, Transmute, PointerCoercion*, Subtype: operand/result can be
            // ptr, struct, tuple, etc.; lowering handles the details. No strict type check here.
            MirCastKindAttr::PtrToPtr
            | MirCastKindAttr::FnPtrToPtr
            | MirCastKindAttr::Transmute
            | MirCastKindAttr::PointerCoercionUnsize
            | MirCastKindAttr::PointerCoercionMutToConst
            | MirCastKindAttr::PointerCoercionArrayToPointer
            | MirCastKindAttr::PointerCoercionReifyFnPointer
            | MirCastKindAttr::PointerCoercionUnsafeFnPointer
            | MirCastKindAttr::PointerCoercionClosureFnPointer
            | MirCastKindAttr::Subtype => {}
        }

        // `niche_encoding` only makes sense on a Transmute. Catching this
        // here means downstream lowering can assume that whenever a niche
        // encoding is present the cast really is a Transmute.
        if self.get_attr_niche_encoding(ctx).is_some()
            && !matches!(cast_kind, MirCastKindAttr::Transmute)
        {
            return verify_err!(
                loc,
                "niche_encoding attribute is only valid on a Transmute cast"
            );
        }

        Ok(())
    }
}

/// Register cast operations into the given context.
pub fn register(ctx: &mut Context) {
    MirCastOp::register(ctx);
}
