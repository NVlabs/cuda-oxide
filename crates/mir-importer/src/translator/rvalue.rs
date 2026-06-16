/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rvalue translation: MIR expressions → `dialect-mir` operations.
//!
//! Translates the right-hand side of MIR assignments into `dialect-mir` ops.
//!
//! # Supported Rvalues
//!
//! | MIR Rvalue          | `dialect-mir` Op                                      |
//! |---------------------|-------------------------------------------------------|
//! | `BinaryOp(+,-,*,/)` | `mir.add`, `mir.sub`, `mir.mul`, `mir.div`            |
//! | `BinaryOp(<,<=,>)`  | `mir.lt`, `mir.le`, `mir.gt`, etc.                    |
//! | `CheckedBinaryOp`   | `mir.checked_add`, etc. (returns tuple)               |
//! | `UnaryOp(Not,Neg)`  | `mir.not`, `mir.neg`                                  |
//! | `Cast`              | `mir.cast`                                            |
//! | `Ref`               | Slot pointer for locals; `mir.ref` for SSA values     |
//! | `Use(operand)`      | `mir.load` of the source slot (no op for constants)   |
//! | `Aggregate`         | `mir.construct_tuple/struct/enum/array`               |
//! | `Repeat`            | `mir.construct_array` (array repeat syntax)           |
//!
//! # Key Functions
//!
//! - [`translate_rvalue`]: Main entry point for rvalue translation
//! - [`translate_operand`]: Translates operands (Copy/Move/Constant/RuntimeChecks)
//! - [`translate_place`]: Translates places to their SSA values (handles ghost locals)
//! - `translate_constant`: Translates MIR constants to `dialect-mir`
//! - `create_ghost_enum_default`: Synthesises a placeholder for never-assigned enum locals

use super::types;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::values::ValueMap;
use dialect_mir::attributes::MirCastKindAttr;
use dialect_mir::attributes::MirFP16Attr;
use dialect_mir::ops::{
    MirAddOp, MirBitAndOp, MirBitOrOp, MirBitXorOp, MirCastOp, MirCheckedAddOp, MirCheckedMulOp,
    MirCheckedSubOp, MirCmpOp, MirConstructArrayOp, MirConstructEnumOp, MirConstructStructOp,
    MirDivOp, MirEqOp, MirExtractFieldOp, MirGeOp, MirGlobalAllocOp, MirGtOp, MirLeOp, MirLoadOp,
    MirLtOp, MirMulOp, MirNeOp, MirNegOp, MirNotOp, MirPtrOffsetOp, MirRefOp, MirRemOp, MirShlOp,
    MirShrOp, MirSubOp,
};
use dialect_mir::types::MirFP16Type;
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;
use pliron::r#type::{TypeObj, Typed};
use pliron::utils::apint::APInt;
use pliron::value::Value;
use pliron::{input_err, input_err_noloc, input_error, input_error_noloc};
use rustc_public::CrateDef;
use rustc_public::mir;
use rustc_public::mir::ProjectionElem;
use rustc_public::ty::{AdtKind, ConstantKind};
use rustc_public_bridge::IndexedVal;
use std::num::NonZeroUsize;

/// Cast a value to a target type if address spaces differ.
///
/// When constructing structs/enums, the field type uses generic address space (0)
/// because Rust's type system doesn't carry address space info. But the actual
/// value may have a specific address space (e.g., addrspace:3 for shared memory).
///
/// This function inserts a MirCastOp to convert from the specific address space
/// to the generic address space, following LLVM's model where generic pointers
/// can hold any address space pointer.
///
/// Returns the (possibly casted) value and the last inserted operation.
fn cast_to_generic_addrspace_if_needed(
    ctx: &mut Context,
    value: Value,
    expected_type: Ptr<TypeObj>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> (Value, Option<Ptr<Operation>>) {
    let value_type = value.get_type(ctx);

    // Check if both are pointer types
    let value_ptr_info: Option<(Ptr<TypeObj>, bool, u32)> = {
        let ty_ref = value_type.deref(ctx);
        ty_ref
            .downcast_ref::<dialect_mir::types::MirPtrType>()
            .map(|pt| (pt.pointee, pt.is_mutable, pt.address_space))
    };

    let expected_ptr_info: Option<(Ptr<TypeObj>, bool, u32)> = {
        let ty_ref = expected_type.deref(ctx);
        ty_ref
            .downcast_ref::<dialect_mir::types::MirPtrType>()
            .map(|pt| (pt.pointee, pt.is_mutable, pt.address_space))
    };

    if let (
        Some((val_pointee, val_mut, val_addrspace)),
        Some((exp_pointee, exp_mut, exp_addrspace)),
    ) = (value_ptr_info, expected_ptr_info)
    {
        // Both are pointers - check if address spaces differ
        if val_addrspace != exp_addrspace && val_pointee == exp_pointee && val_mut == exp_mut {
            // Need to insert an address space cast
            // Create the target type (same pointer but with expected address space)
            let target_ptr_ty =
                dialect_mir::types::MirPtrType::get(ctx, exp_pointee, exp_mut, exp_addrspace);

            let cast_op = Operation::new(
                ctx,
                MirCastOp::get_concrete_op_info(),
                vec![target_ptr_ty.into()],
                vec![value],
                vec![],
                0,
            );
            cast_op.deref_mut(ctx).set_loc(loc);
            MirCastOp::new(cast_op).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);

            if let Some(prev) = prev_op {
                cast_op.insert_after(ctx, prev);
            } else {
                cast_op.insert_at_front(block_ptr, ctx);
            }

            let casted_value = cast_op.deref(ctx).get_result(0);
            return (casted_value, Some(cast_op));
        }
    }

    // No cast needed
    (value, prev_op)
}

/// Cast struct field values to match expected field types (address space normalization).
///
/// When constructing a struct, field values may have specific address spaces (e.g., addrspace:3)
/// but the struct type's field definitions use generic address space (addrspace:0).
/// This function casts each field value to match its expected type.
fn cast_struct_fields_to_expected_types(
    ctx: &mut Context,
    field_values: Vec<Value>,
    struct_type: Ptr<TypeObj>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> (Vec<Value>, Option<Ptr<Operation>>) {
    // Get field types from the struct type
    let field_types: Vec<Ptr<TypeObj>> = {
        let ty_ref = struct_type.deref(ctx);
        if let Some(st) = ty_ref.downcast_ref::<dialect_mir::types::MirStructType>() {
            st.field_types.clone()
        } else {
            // Not a struct type, return as-is
            return (field_values, prev_op);
        }
    };

    let mut result_values = Vec::with_capacity(field_values.len());
    let mut current_prev_op = prev_op;

    for (i, value) in field_values.into_iter().enumerate() {
        if let Some(expected_type) = field_types.get(i) {
            let (casted_value, new_prev_op) = cast_to_generic_addrspace_if_needed(
                ctx,
                value,
                *expected_type,
                block_ptr,
                current_prev_op,
                loc.clone(),
            );
            result_values.push(casted_value);
            current_prev_op = new_prev_op;
        } else {
            result_values.push(value);
        }
    }

    (result_values, current_prev_op)
}

/// Cast enum variant field values to match expected field types (address space normalization).
///
/// Similar to cast_struct_fields_to_expected_types, but for enum variants.
/// Gets the field types for the specific variant and casts each field value.
fn cast_enum_fields_to_expected_types(
    ctx: &mut Context,
    field_values: Vec<Value>,
    enum_type: Ptr<TypeObj>,
    variant_idx: usize,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> (Vec<Value>, Option<Ptr<Operation>>) {
    // Get the field types for this variant from the enum type
    let variant_field_types: Vec<Ptr<TypeObj>> = {
        let ty_ref = enum_type.deref(ctx);
        if let Some(et) = ty_ref.downcast_ref::<dialect_mir::types::MirEnumType>() {
            // Calculate the field offset for this variant
            let field_offset: usize = et.variant_field_counts[..variant_idx]
                .iter()
                .map(|&x| x as usize)
                .sum();
            let field_count = et.variant_field_counts[variant_idx] as usize;

            // Get the field types for this variant
            et.all_field_types[field_offset..field_offset + field_count].to_vec()
        } else {
            // Not an enum type, return as-is
            return (field_values, prev_op);
        }
    };

    let mut result_values = Vec::with_capacity(field_values.len());
    let mut current_prev_op = prev_op;

    for (i, value) in field_values.into_iter().enumerate() {
        if let Some(expected_type) = variant_field_types.get(i) {
            let (casted_value, new_prev_op) = cast_to_generic_addrspace_if_needed(
                ctx,
                value,
                *expected_type,
                block_ptr,
                current_prev_op,
                loc.clone(),
            );
            result_values.push(casted_value);
            current_prev_op = new_prev_op;
        } else {
            result_values.push(value);
        }
    }

    (result_values, current_prev_op)
}

/// Translates a MIR rvalue to pliron IR operation(s).
///
/// # Returns
///
/// Tuple of `(Option<op>, result_value, last_inserted)`:
/// - `op`: The main operation (None for `Rvalue::Use`)
/// - `result_value`: The SSA value produced
/// - `last_inserted`: Last inserted helper op (for operation ordering)
///
/// The operation is created but **not inserted** - caller must insert it.
pub fn translate_rvalue(
    ctx: &mut Context,
    body: &mir::Body,
    rvalue: &mir::Rvalue,
    value_map: &mut ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Option<Ptr<Operation>>, Value, Option<Ptr<Operation>>)> {
    match rvalue {
        mir::Rvalue::BinaryOp(bin_op, left, right) => {
            let (left_val, prev_op_after_left) =
                translate_operand(ctx, body, left, value_map, block_ptr, prev_op, loc.clone())?;
            let (right_val, prev_op_after_right) = translate_operand(
                ctx,
                body,
                right,
                value_map,
                block_ptr,
                prev_op_after_left,
                loc.clone(),
            )?;

            // Check if this is a comparison operation that may need type coercion
            let is_comparison = matches!(
                bin_op,
                mir::BinOp::Eq
                    | mir::BinOp::Ne
                    | mir::BinOp::Lt
                    | mir::BinOp::Le
                    | mir::BinOp::Gt
                    | mir::BinOp::Ge
            );

            // For comparison operations, handle type mismatches by casting the right operand
            // to match the left operand's type. This commonly occurs when comparing enum
            // discriminants (u8) against isize constants in Rust's MIR.
            let (final_right_val, final_prev_op) = if is_comparison {
                let left_type = left_val.get_type(ctx);
                let right_type = right_val.get_type(ctx);

                if left_type != right_type {
                    // Insert a cast operation to coerce right to match left's type
                    let cast_op = Operation::new(
                        ctx,
                        MirCastOp::get_concrete_op_info(),
                        vec![left_type],
                        vec![right_val],
                        vec![],
                        0,
                    );
                    cast_op.deref_mut(ctx).set_loc(loc.clone());
                    let coercion_kind = {
                        let l = left_type.deref(ctx);
                        let r = right_type.deref(ctx);
                        if l.downcast_ref::<IntegerType>().is_some()
                            && r.downcast_ref::<IntegerType>().is_some()
                        {
                            MirCastKindAttr::IntToInt
                        } else if l.downcast_ref::<FP32Type>().is_some()
                            || l.downcast_ref::<FP64Type>().is_some()
                        {
                            if r.downcast_ref::<FP32Type>().is_some()
                                || r.downcast_ref::<FP64Type>().is_some()
                            {
                                MirCastKindAttr::FloatToFloat
                            } else {
                                MirCastKindAttr::Transmute
                            }
                        } else if l.downcast_ref::<dialect_mir::types::MirPtrType>().is_some()
                            && r.downcast_ref::<dialect_mir::types::MirPtrType>().is_some()
                        {
                            MirCastKindAttr::PtrToPtr
                        } else {
                            MirCastKindAttr::Transmute
                        }
                    };
                    MirCastOp::new(cast_op).set_attr_cast_kind(ctx, coercion_kind);

                    // Insert the cast op after the right operand was processed
                    if let Some(prev) = prev_op_after_right {
                        cast_op.insert_after(ctx, prev);
                    } else {
                        cast_op.insert_at_front(block_ptr, ctx);
                    }

                    let casted_right = cast_op.deref(ctx).get_result(0);
                    (casted_right, Some(cast_op))
                } else {
                    (right_val, prev_op_after_right)
                }
            } else {
                (right_val, prev_op_after_right)
            };

            // Determine result type and operation
            // Comparison operations return bool (i1), arithmetic ops return operand type
            let (op_id, result_type) = match bin_op {
                // Arithmetic operations - return same type as operands
                // Unchecked variants are identical - overflow check is elided at MIR level
                mir::BinOp::Add | mir::BinOp::AddUnchecked => {
                    (MirAddOp::get_concrete_op_info(), left_val.get_type(ctx))
                }
                mir::BinOp::Sub | mir::BinOp::SubUnchecked => {
                    (MirSubOp::get_concrete_op_info(), left_val.get_type(ctx))
                }
                mir::BinOp::Mul | mir::BinOp::MulUnchecked => {
                    (MirMulOp::get_concrete_op_info(), left_val.get_type(ctx))
                }
                mir::BinOp::Div => (MirDivOp::get_concrete_op_info(), left_val.get_type(ctx)),
                mir::BinOp::Rem => (MirRemOp::get_concrete_op_info(), left_val.get_type(ctx)),

                // Comparison operations - return bool (i1)
                mir::BinOp::Lt => (
                    MirLtOp::get_concrete_op_info(),
                    types::get_bool_type(ctx).to_ptr(),
                ),
                mir::BinOp::Le => (
                    MirLeOp::get_concrete_op_info(),
                    types::get_bool_type(ctx).to_ptr(),
                ),
                mir::BinOp::Gt => (
                    MirGtOp::get_concrete_op_info(),
                    types::get_bool_type(ctx).to_ptr(),
                ),
                mir::BinOp::Ge => (
                    MirGeOp::get_concrete_op_info(),
                    types::get_bool_type(ctx).to_ptr(),
                ),
                mir::BinOp::Eq => (
                    MirEqOp::get_concrete_op_info(),
                    types::get_bool_type(ctx).to_ptr(),
                ),
                mir::BinOp::Ne => (
                    MirNeOp::get_concrete_op_info(),
                    types::get_bool_type(ctx).to_ptr(),
                ),
                // Three-way comparison (`Ord::cmp`) - returns
                // `core::cmp::Ordering`. rustc's `BinOp::ty` knows the
                // result type of every binop (including `Cmp`, for which it
                // returns the `Ordering` enum), so derive it locally from
                // the operand types instead of threading the assignment
                // destination type through every translate_rvalue caller.
                mir::BinOp::Cmp => {
                    let left_ty = left.ty(body.locals()).map_err(|e| {
                        pliron::input_error!(
                            loc.clone(),
                            TranslationErr::unsupported(format!(
                                "Failed to resolve BinOp::Cmp lhs type: {:?}",
                                e
                            ))
                        )
                    })?;
                    let right_ty = right.ty(body.locals()).map_err(|e| {
                        pliron::input_error!(
                            loc.clone(),
                            TranslationErr::unsupported(format!(
                                "Failed to resolve BinOp::Cmp rhs type: {:?}",
                                e
                            ))
                        )
                    })?;
                    let ordering_ty = bin_op.ty(left_ty, right_ty);
                    (
                        MirCmpOp::get_concrete_op_info(),
                        types::translate_type(ctx, &ordering_ty)?,
                    )
                }

                // Pointer offset - ptr.add(n) returns ptr + n * sizeof(element)
                mir::BinOp::Offset => (
                    MirPtrOffsetOp::get_concrete_op_info(),
                    left_val.get_type(ctx), // Result is same pointer type
                ),

                // Shift operations - result is same as left operand type
                // Unchecked variants are identical - overflow check is elided at MIR level
                mir::BinOp::Shr | mir::BinOp::ShrUnchecked => {
                    (MirShrOp::get_concrete_op_info(), left_val.get_type(ctx))
                }
                mir::BinOp::Shl | mir::BinOp::ShlUnchecked => {
                    (MirShlOp::get_concrete_op_info(), left_val.get_type(ctx))
                }

                // Bitwise operations - result is same as operand type
                mir::BinOp::BitAnd => (MirBitAndOp::get_concrete_op_info(), left_val.get_type(ctx)),
                mir::BinOp::BitOr => (MirBitOrOp::get_concrete_op_info(), left_val.get_type(ctx)),
                mir::BinOp::BitXor => (MirBitXorOp::get_concrete_op_info(), left_val.get_type(ctx)),
            };

            let op = Operation::new(
                ctx,
                op_id,
                vec![result_type],
                vec![left_val, final_right_val],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc);

            let result = op.deref(ctx).get_result(0);

            Ok((Some(op), result, final_prev_op))
        }
        mir::Rvalue::UnaryOp(un_op, operand) => {
            match un_op {
                mir::UnOp::PtrMetadata => {
                    // PtrMetadata extracts the length from a slice (fat pointer)
                    // For a slice &[T], this is field 1 (field 0 is the pointer, field 1 is length)
                    let (operand_val, prev_op_after_operand) = translate_operand(
                        ctx,
                        body,
                        operand,
                        value_map,
                        block_ptr,
                        prev_op,
                        loc.clone(),
                    )?;

                    // Result type is usize (the length)
                    let result_type = types::get_usize_type(ctx);

                    // Create an extract field operation to get field 1 (length)
                    let op = Operation::new(
                        ctx,
                        MirExtractFieldOp::get_concrete_op_info(),
                        vec![result_type.to_ptr()],
                        vec![operand_val],
                        vec![],
                        0,
                    );
                    op.deref_mut(ctx).set_loc(loc.clone());

                    let extract_op = MirExtractFieldOp::new(op);
                    extract_op.set_attr_index(ctx, dialect_mir::attributes::FieldIndexAttr(1));

                    let result = extract_op.get_operation().deref(ctx).get_result(0);

                    Ok((
                        Some(extract_op.get_operation()),
                        result,
                        prev_op_after_operand,
                    ))
                }
                mir::UnOp::Not | mir::UnOp::Neg => {
                    let (operand_val, prev_op_after_operand) = translate_operand(
                        ctx,
                        body,
                        operand,
                        value_map,
                        block_ptr,
                        prev_op,
                        loc.clone(),
                    )?;
                    let result_type = operand_val.get_type(ctx);

                    let op_id = match un_op {
                        mir::UnOp::Not => MirNotOp::get_concrete_op_info(),
                        mir::UnOp::Neg => MirNegOp::get_concrete_op_info(),
                        _ => unreachable!(),
                    };

                    let op =
                        Operation::new(ctx, op_id, vec![result_type], vec![operand_val], vec![], 0);
                    op.deref_mut(ctx).set_loc(loc);

                    let result = op.deref(ctx).get_result(0);

                    Ok((Some(op), result, prev_op_after_operand))
                }
            }
        }
        mir::Rvalue::Cast(kind, operand, ty) => {
            // `let f: fn(u32) -> u32 = inc;` compiles to a ReifyFnPointer
            // cast. It is not a value conversion: the fn item `inc` is
            // zero-sized, so there is nothing to convert. What the program
            // needs is some address-like value identifying the function.
            // Real code addresses do not exist on the device (the function
            // may not even be compiled), so we make a stable stand-in: a
            // hash of the function's mangled name, cast int -> ptr. With
            // that, `f == f` is true and two different functions compare
            // unequal (Rust permits, but does not promise, distinct fn
            // addresses, so a hash stand-in is within contract). CALLING
            // through the pointer is still unsupported and fails loudly at
            // the call site. Handled before translate_operand because the
            // zero-sized fn-item operand itself never becomes a value.
            if let mir::CastKind::PointerCoercion(mir::PointerCoercion::ReifyFnPointer(_)) = kind {
                return translate_reify_fn_pointer(ctx, body, operand, ty, block_ptr, prev_op, loc);
            }

            let (operand_val, prev_op_after_operand) = translate_operand(
                ctx,
                body,
                operand,
                value_map,
                block_ptr,
                prev_op,
                loc.clone(),
            )?;

            let result_type = types::translate_type(ctx, ty)?;

            let op = Operation::new(
                ctx,
                MirCastOp::get_concrete_op_info(),
                vec![result_type],
                vec![operand_val],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc);

            let cast_kind_attr = match kind {
                mir::CastKind::IntToInt => MirCastKindAttr::IntToInt,
                mir::CastKind::IntToFloat => MirCastKindAttr::IntToFloat,
                mir::CastKind::FloatToInt => MirCastKindAttr::FloatToInt,
                mir::CastKind::FloatToFloat => MirCastKindAttr::FloatToFloat,
                mir::CastKind::PtrToPtr => MirCastKindAttr::PtrToPtr,
                mir::CastKind::FnPtrToPtr => MirCastKindAttr::FnPtrToPtr,
                mir::CastKind::PointerExposeAddress => MirCastKindAttr::PointerExposeAddress,
                mir::CastKind::PointerWithExposedProvenance => {
                    MirCastKindAttr::PointerWithExposedProvenance
                }
                mir::CastKind::Transmute => MirCastKindAttr::Transmute,
                mir::CastKind::PointerCoercion(coercion) => match coercion {
                    mir::PointerCoercion::Unsize => MirCastKindAttr::PointerCoercionUnsize,
                    mir::PointerCoercion::MutToConstPointer => {
                        MirCastKindAttr::PointerCoercionMutToConst
                    }
                    mir::PointerCoercion::ArrayToPointer => {
                        MirCastKindAttr::PointerCoercionArrayToPointer
                    }
                    mir::PointerCoercion::ReifyFnPointer(_) => {
                        MirCastKindAttr::PointerCoercionReifyFnPointer
                    }
                    mir::PointerCoercion::UnsafeFnPointer => {
                        MirCastKindAttr::PointerCoercionUnsafeFnPointer
                    }
                    mir::PointerCoercion::ClosureFnPointer(_safety) => {
                        MirCastKindAttr::PointerCoercionClosureFnPointer
                    }
                },
                mir::CastKind::Subtype => MirCastKindAttr::Subtype,
            };
            let cast_op = MirCastOp::new(op);
            cast_op.set_attr_cast_kind(ctx, cast_kind_attr);

            // Record rustc's niche encoding on the cast so mir-lower can
            // rebuild our un-niched `MirEnumType` aggregate (issue #21).
            // The attribute is a typed `NicheEncodingAttr` so the contract
            // between importer and lowering is enforced by pliron rather
            // than by a hand-rolled string key.
            if matches!(kind, mir::CastKind::Transmute)
                && let Ok(layout) = ty.layout()
                && let rustc_public::abi::VariantsShape::Multiple {
                    tag_encoding:
                        rustc_public::abi::TagEncoding::Niche {
                            untagged_variant,
                            niche_variants,
                            niche_start,
                        },
                    ..
                } = &layout.shape().variants
            {
                // Niched scalars are at most 64 bits wide. If rustc ever
                // hands us something wider, fail loudly instead of
                // truncating: the wrong bit pattern would silently match a
                // different enum variant at runtime.
                let niche_start_u64 = u64::try_from(*niche_start).map_err(|_| {
                    input_error_noloc!(TranslationErr::unsupported(format!(
                        "Niche start {} exceeds u64; niched-enum Transmute with > 64-bit scalar is not supported",
                        niche_start
                    )))
                })?;
                let niche_variant_idx = niche_variants.start().to_index() as u32;
                let untagged_variant_idx = untagged_variant.to_index() as u32;
                cast_op.set_attr_niche_encoding(
                    ctx,
                    dialect_mir::attributes::NicheEncodingAttr {
                        niche_start: niche_start_u64,
                        niche_variant_idx,
                        untagged_variant_idx,
                    },
                );
            }

            let result = op.deref(ctx).get_result(0);

            Ok((Some(op), result, prev_op_after_operand))
        }
        mir::Rvalue::CheckedBinaryOp(bin_op, left, right) => {
            // CheckedBinaryOp produces a tuple (result, overflow_flag)

            // Handle checked operations (Add, Sub, Mul)
            match bin_op {
                mir::BinOp::Add | mir::BinOp::Sub | mir::BinOp::Mul => {
                    // Get operands from value_map, tracking the last inserted operation
                    let (left_val, prev_op_after_left) = translate_operand(
                        ctx,
                        body,
                        left,
                        value_map,
                        block_ptr,
                        prev_op,
                        loc.clone(),
                    )?;
                    let (right_val, prev_op_after_right) = translate_operand(
                        ctx,
                        body,
                        right,
                        value_map,
                        block_ptr,
                        prev_op_after_left,
                        loc.clone(),
                    )?;

                    // Get the result type: tuple(operand_type, bool)
                    // The first element matches the operand type (could be i32, usize, etc.)
                    let operand_type = left_val.get_type(ctx);
                    let bool_type = types::get_bool_type(ctx).into();
                    let tuple_type = types::MirTupleType::get(ctx, vec![operand_type, bool_type]);
                    let result_type_ptr = tuple_type.to_ptr();

                    // Create a checked operation based on the binary operator
                    let op_id = match bin_op {
                        mir::BinOp::Add => MirCheckedAddOp::get_concrete_op_info(),
                        mir::BinOp::Sub => MirCheckedSubOp::get_concrete_op_info(),
                        mir::BinOp::Mul => MirCheckedMulOp::get_concrete_op_info(),
                        _ => unreachable!(),
                    };
                    let op = Operation::new(
                        ctx,
                        op_id,
                        vec![result_type_ptr],     // Result type (tuple)
                        vec![left_val, right_val], // Operands
                        vec![],                    // No successors
                        0,                         // No regions
                    );
                    op.deref_mut(ctx).set_loc(loc);

                    // Get the result value
                    let result = op.deref(ctx).get_result(0);

                    // Return Some(operation) - caller must insert it after field extractions
                    Ok((Some(op), result, prev_op_after_right))
                }
                _ => input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "CheckedBinaryOp {:?} not yet implemented",
                        bin_op
                    ))
                ),
            }
        }
        mir::Rvalue::Use(operand) => {
            // Use just copies/moves a value - no operation needed, just pass through
            // The operand translation may insert field extraction operations
            let (val, last_inserted) =
                translate_operand(ctx, body, operand, value_map, block_ptr, prev_op, loc)?;

            // Return None for operation - Use doesn't create an operation
            // Any field extractions are already inserted and tracked in last_inserted
            Ok((None, val, last_inserted))
        }
        mir::Rvalue::Ref(_region, borrow_kind, place) => {
            // Ref creates a reference to a place: &place or &mut place.
            //
            // Strategy:
            //
            // 1. `&local` / `&mut local` -- return the local's alloca slot
            //    pointer directly (ZST locals get a synthesised pointer).
            // 2. Any projected place -- compute the real in-memory address
            //    by walking the FULL projection list from the base local's
            //    slot via `translate_place_address`: `&(*ptr)` loads the
            //    pointer, `&(*ptr).field` adds a `mir.field_addr`,
            //    `&x.arr[i]` adds a `mir.array_element_addr`, and arbitrary
            //    combinations compose. Borrows produced this way ALIAS the
            //    original storage, which is what Rust requires: e.g.
            //    `Enumerate::next` takes `&mut (*_1).0` and `Iter::next`
            //    must advance the ORIGINAL Iter in place -- a `mir.ref` of
            //    an extracted field VALUE would mutate a copy and loop
            //    forever.
            // 3. Only when no address can be computed (slot-less computed
            //    value, or a projection the walker cannot lower, e.g.
            //    Downcast) do we fall back to materialising the VALUE and
            //    wrapping it in `mir.ref` (fresh slot + store of a COPY).
            //    That is sound for shared borrows (reads through a copy)
            //    and a silent miscompile for mutable ones (writes land in
            //    the copy), so mutable borrows hard-error instead.

            // Case 1: bare local reference `&local` / `&mut local`.
            //
            // Alloca + load/store model: every non-ZST MIR local is backed by
            // a stack slot emitted at the top of the entry block. Taking the
            // address of the local therefore just returns that slot pointer --
            // no extra allocation is needed. `mem2reg` folds this back into
            // SSA when the borrow doesn't escape.
            //
            // Mutability: slots are always allocated mutable (we may store
            // into them regardless of the Rust mutability of the local).
            // Callers that expect a `*const T` pointer handle the coercion
            // via `MirCastOp::PointerCoercionMutToConst`; most consumers in
            // the dialect (FieldAddr, ArrayElementAddr, Load, Store) are
            // mutability-agnostic at the pliron level.
            let is_mutable = matches!(borrow_kind, mir::BorrowKind::Mut { .. });
            if place.projection.is_empty() {
                if let Some(slot) = value_map.get_slot(place.local) {
                    return Ok((None, slot, prev_op));
                }
                // ZST local (no slot). Synthesise a pointer-to-ZST via
                // MirRefOp as a fallback so callers still get a well-typed
                // pointer value.
                let local_decl = &body.locals()[place.local];
                let ty_ptr = super::types::translate_type(ctx, &local_decl.ty)?;
                let (zst_val, last_inserted) =
                    if ty_ptr.deref(ctx).is::<dialect_mir::types::MirEnumType>() {
                        let op = create_ghost_enum_default(ctx, ty_ptr, loc.clone());
                        match prev_op {
                            Some(p) => op.insert_after(ctx, p),
                            None => op.insert_at_front(block_ptr, ctx),
                        }
                        (op.deref(ctx).get_result(0), Some(op))
                    } else {
                        let op = create_zst_aggregate(ctx, ty_ptr, loc.clone());
                        match prev_op {
                            Some(p) => op.insert_after(ctx, p),
                            None => op.insert_at_front(block_ptr, ctx),
                        }
                        (op.deref(ctx).get_result(0), Some(op))
                    };
                let ptr_ty = dialect_mir::types::MirPtrType::get_generic(ctx, ty_ptr, is_mutable);
                let ref_op = Operation::new(
                    ctx,
                    MirRefOp::get_concrete_op_info(),
                    vec![ptr_ty.into()],
                    vec![zst_val],
                    vec![],
                    0,
                );
                ref_op.deref_mut(ctx).set_loc(loc);
                MirRefOp::new(ref_op).set_mutable(ctx, is_mutable);
                match last_inserted {
                    Some(p) => ref_op.insert_after(ctx, p),
                    None => ref_op.insert_at_front(block_ptr, ctx),
                }
                let result_val = ref_op.deref(ctx).get_result(0);
                return Ok((None, result_val, Some(ref_op)));
            }

            // Case 2: unified address path -- walk the full projection list
            // (`Deref`, `Field`, `Index`, `ConstantIndex`) from the base
            // local's alloca slot. This is the "correct-refs" path: the
            // resulting pointer addresses the ORIGINAL storage, so writes
            // through the borrow mutate the borrowed place.
            if let Some((result_val, last_inserted)) = translate_place_address(
                ctx,
                body,
                value_map,
                place,
                is_mutable,
                block_ptr,
                prev_op,
                loc.clone(),
            )? {
                return Ok((None, result_val, last_inserted));
            }

            // No address could be computed. The only remaining strategy is
            // the value-copy fallback below, which is a silent miscompile
            // for mutable borrows: writes through the borrow would land in
            // the copy and the original place would never change. Refuse
            // loudly instead of emitting wrong code.
            if is_mutable {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Rvalue::Ref: cannot compute an in-memory address for the mutable \
                         borrow of place {:?} (projection {:?}); the value-copy fallback \
                         would silently discard writes through the borrow",
                        place, place.projection
                    ))
                );
            }

            // Case 3: shared-borrow fallback -- reference to a computed
            // value that has no backing slot (e.g. the result of an rvalue
            // expression) or whose projection the address walker cannot
            // lower (e.g. enum Downcast, issues #131/#146). Emit `mir.ref`
            // which allocates a fresh slot, stores a COPY of the value, and
            // returns the pointer. Sound for shared borrows only (reads);
            // mutable borrows were rejected above.
            let (val, last_inserted) =
                translate_place(ctx, body, place, value_map, block_ptr, prev_op, loc.clone())?;

            let val_ty = val.get_type(ctx);
            let ptr_ty = dialect_mir::types::MirPtrType::get_generic(ctx, val_ty, is_mutable);

            let ref_op = Operation::new(
                ctx,
                MirRefOp::get_concrete_op_info(),
                vec![ptr_ty.into()],
                vec![val],
                vec![],
                0,
            );
            ref_op.deref_mut(ctx).set_loc(loc);
            MirRefOp::new(ref_op).set_mutable(ctx, is_mutable);

            let result_val = ref_op.deref(ctx).get_result(0);
            Ok((Some(ref_op), result_val, last_inserted))
        }
        mir::Rvalue::AddressOf(mutability, place) => {
            // AddressOf creates a raw pointer to a place: `&raw const place`
            // / `&raw mut place` (also `core::ptr::addr_of!`). Raw pointers
            // have the same aliasing requirement as references: the pointer
            // must address the ORIGINAL place, so this routes through the
            // same unified address walker as `Rvalue::Ref` (which also gives
            // raw pointers the runtime-Index / ConstantIndex handling).
            let is_mutable = matches!(mutability, mir::RawPtrKind::Mut);

            // Bare local: the alloca slot IS the address.
            if place.projection.is_empty()
                && let Some(slot) = value_map.get_slot(place.local)
            {
                return Ok((None, slot, prev_op));
            }

            // Unified address path: full projection walk from the slot
            // (`&raw (*ptr)` loads the pointer, `&raw (*ptr).field[i]`
            // composes field + element addresses, ...).
            if let Some((result_val, last_inserted)) = translate_place_address(
                ctx,
                body,
                value_map,
                place,
                is_mutable,
                block_ptr,
                prev_op,
                loc.clone(),
            )? {
                return Ok((None, result_val, last_inserted));
            }

            // No address could be computed. The value-copy fallback below
            // returns a pointer to a COPY, so writes through a `&raw mut`
            // would be silently lost -- refuse loudly. Exception: a bare
            // slot-less local is a ZST (no bytes), so a copy cannot lose
            // writes; let it use the fallback for both mutabilities, the
            // same way `Rvalue::Ref` synthesises ZST borrows.
            if is_mutable && !place.projection.is_empty() {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Rvalue::AddressOf: cannot compute an in-memory address for \
                         `&raw mut` of place {:?} (projection {:?}); the value-copy \
                         fallback would silently discard writes through the pointer",
                        place, place.projection
                    ))
                );
            }

            // Shared (or bare-ZST) fallback: translate to a value and
            // materialize an address of a copy.
            let (val, last_inserted) =
                translate_place(ctx, body, place, value_map, block_ptr, prev_op, loc.clone())?;

            let val_ty = val.get_type(ctx);
            let ptr_ty = dialect_mir::types::MirPtrType::get_generic(ctx, val_ty, is_mutable);

            use dialect_mir::ops::MirRefOp;
            let ref_op = Operation::new(
                ctx,
                MirRefOp::get_concrete_op_info(),
                vec![ptr_ty.into()],
                vec![val],
                vec![],
                0,
            );
            ref_op.deref_mut(ctx).set_loc(loc);

            let mir_ref_op = MirRefOp::new(ref_op);
            mir_ref_op.set_mutable(ctx, is_mutable);

            let result_val = ref_op.deref(ctx).get_result(0);

            Ok((Some(ref_op), result_val, last_inserted))
        }
        mir::Rvalue::Aggregate(aggregate_kind, operands) => {
            // Aggregate constructs a compound type from individual values.
            // This is used for:
            // - Tuple construction: (a, b, c)
            // - Struct construction: MyStruct { field1: a, field2: b }
            // - Array construction: [a, b, c]

            match aggregate_kind {
                mir::AggregateKind::Adt(adt_def, variant_idx, substs, _, _) => {
                    let adt_kind = adt_def.kind();

                    // Get the type using adt_def.ty_with_args()
                    let adt_ty_rust = adt_def.ty_with_args(substs);
                    let adt_ty = types::translate_type(ctx, &adt_ty_rust)?;
                    let (field_values, current_prev_op) = translate_adt_aggregate_field_values(
                        ctx,
                        body,
                        *adt_def,
                        *variant_idx,
                        substs,
                        operands,
                        value_map,
                        block_ptr,
                        prev_op,
                        loc.clone(),
                    )?;

                    match adt_kind {
                        AdtKind::Struct => {
                            // Check if the translated type is a struct type.
                            // Scalar-lowered newtypes like ThreadIndex are translated to
                            // their single runtime field type. They may still have ZST
                            // marker fields in MIR, so select the one field whose
                            // translated value matches the scalar result type.
                            let is_struct_type = {
                                let ty_obj = adt_ty.deref(ctx);
                                ty_obj.is::<dialect_mir::types::MirStructType>()
                                    || ty_obj.is::<dialect_mir::types::MirTupleType>()
                            };

                            if !is_struct_type {
                                // Scalar-lowered ADT: layout collapsed to a single runtime
                                // value. The MIR Aggregate may still list ZST fields
                                // (PhantomData, etc.) -- those translate to types other
                                // than `adt_ty`, so filtering by "type matches the
                                // collapsed scalar" reliably picks the one runtime field.
                                //
                                // This works for shapes like
                                //     ThreadIndex { raw: usize, _kernel: PhantomData<...>, ... }
                                // where exactly one field shares the scalar type. If a
                                // future scalar-lowered ADT has two runtime fields with
                                // the same type, the filter returns >1 match and we bail
                                // -- the assumption is wrong and the translator needs an
                                // explicit story for that shape.
                                let runtime_fields: Vec<Value> = field_values
                                    .iter()
                                    .copied()
                                    .filter(|value| value.get_type(ctx) == adt_ty)
                                    .collect();

                                if runtime_fields.len() == 1 {
                                    Ok((None, runtime_fields[0], current_prev_op))
                                } else {
                                    input_err!(
                                        loc,
                                        TranslationErr::unsupported(format!(
                                            "Scalar-lowered ADT expected exactly one runtime field, found {}",
                                            runtime_fields.len()
                                        ))
                                    )
                                }
                            } else {
                                // Cast field values to expected types (address space normalization)
                                // This handles cases where field values have specific address spaces
                                // (e.g., addrspace:3 for shared memory) but the struct type expects
                                // generic address space (addrspace:0)
                                let (casted_field_values, prev_after_casts) =
                                    cast_struct_fields_to_expected_types(
                                        ctx,
                                        field_values,
                                        adt_ty,
                                        block_ptr,
                                        current_prev_op,
                                        loc.clone(),
                                    );

                                // Create the construct_struct operation
                                let op = Operation::new(
                                    ctx,
                                    MirConstructStructOp::get_concrete_op_info(),
                                    vec![adt_ty],
                                    casted_field_values,
                                    vec![],
                                    0,
                                );
                                op.deref_mut(ctx).set_loc(loc);

                                let result = op.deref(ctx).get_result(0);

                                Ok((Some(op), result, prev_after_casts))
                            }
                        }
                        AdtKind::Enum => {
                            // Get the variant index for the enum
                            // NOTE: variant_idx IS the index (0, 1, 2, ...), NOT the discriminant!
                            // discriminant_for_variant returns the discriminant VALUE which may differ
                            // (e.g., enum Foo { A = 0, B = 2, C = 6 } has indices 0,1,2 but discriminants 0,2,6)
                            let variant_index_val: usize = variant_idx.to_index();

                            // Cast field values to expected types (address space normalization)
                            // This handles cases where field values have specific address spaces
                            // (e.g., addrspace:3 for shared memory) but the enum type expects
                            // generic address space (addrspace:0)
                            let (casted_field_values, prev_after_casts) =
                                cast_enum_fields_to_expected_types(
                                    ctx,
                                    field_values,
                                    adt_ty,
                                    variant_index_val,
                                    block_ptr,
                                    current_prev_op,
                                    loc.clone(),
                                );

                            // Create the construct_enum operation with variant_index attribute
                            let op = Operation::new(
                                ctx,
                                MirConstructEnumOp::get_concrete_op_info(),
                                vec![adt_ty],
                                casted_field_values,
                                vec![],
                                0,
                            );
                            op.deref_mut(ctx).set_loc(loc.clone());

                            let enum_op = MirConstructEnumOp::new(op);
                            enum_op.set_attr_construct_enum_variant_index(
                                ctx,
                                dialect_mir::attributes::VariantIndexAttr(variant_index_val as u32),
                            );

                            let result = op.deref(ctx).get_result(0);

                            Ok((Some(op), result, prev_after_casts))
                        }
                        AdtKind::Union => {
                            input_err!(
                                loc,
                                TranslationErr::unsupported(format!(
                                    "Union aggregate not yet supported: {}",
                                    adt_def.trimmed_name()
                                ))
                            )
                        }
                    }
                }
                mir::AggregateKind::Tuple => {
                    // Tuple construction: (a, b, c)
                    // Similar to struct construction but with positional fields

                    // Translate all element operands
                    let mut element_values = Vec::with_capacity(operands.len());
                    let mut element_types = Vec::with_capacity(operands.len());
                    let mut current_prev_op = prev_op;

                    for operand in operands {
                        let (val, new_prev_op) = translate_operand(
                            ctx,
                            body,
                            operand,
                            value_map,
                            block_ptr,
                            current_prev_op,
                            loc.clone(),
                        )?;
                        element_values.push(val);
                        element_types.push(val.get_type(ctx));
                        current_prev_op = new_prev_op;
                    }

                    // Create the tuple type
                    let tuple_ty = dialect_mir::types::MirTupleType::get(ctx, element_types);

                    // Create mir.construct_tuple operation
                    use dialect_mir::ops::MirConstructTupleOp;

                    let op = Operation::new(
                        ctx,
                        MirConstructTupleOp::get_concrete_op_info(),
                        vec![tuple_ty.into()],
                        element_values,
                        vec![],
                        0,
                    );
                    op.deref_mut(ctx).set_loc(loc);

                    let result = op.deref(ctx).get_result(0);

                    Ok((Some(op), result, current_prev_op))
                }
                mir::AggregateKind::Array(elem_ty) => {
                    // Array construction: [e0, e1, e2, ...] -> mir.construct_array
                    // Translate the element type
                    let element_type = types::translate_type(ctx, elem_ty)?;
                    let array_size = operands.len() as u64;

                    // Translate all element operands
                    let mut element_values = Vec::with_capacity(operands.len());
                    let mut current_prev_op = prev_op;

                    for operand in operands {
                        let (val, new_prev_op) = translate_operand(
                            ctx,
                            body,
                            operand,
                            value_map,
                            block_ptr,
                            current_prev_op,
                            loc.clone(),
                        )?;
                        let (val, new_prev_op) = cast_to_generic_addrspace_if_needed(
                            ctx,
                            val,
                            element_type,
                            block_ptr,
                            new_prev_op,
                            loc.clone(),
                        );
                        element_values.push(val);
                        current_prev_op = new_prev_op;
                    }

                    // Create the array type
                    let array_ty =
                        dialect_mir::types::MirArrayType::get(ctx, element_type, array_size);

                    // Create mir.construct_array operation
                    let op = Operation::new(
                        ctx,
                        MirConstructArrayOp::get_concrete_op_info(),
                        vec![array_ty.into()],
                        element_values,
                        vec![],
                        0,
                    );
                    op.deref_mut(ctx).set_loc(loc);

                    let result = op.deref(ctx).get_result(0);

                    Ok((Some(op), result, current_prev_op))
                }
                mir::AggregateKind::Closure(closure_def, substs) => {
                    // Closure construction with captures
                    // The operands are the captured values that form the closure environment
                    //
                    // MIR: _N = Aggregate(Closure(...), [captured_val1, captured_val2, ...])
                    // We construct a struct with the captured values as fields

                    // Translate all captured operands
                    let mut capture_values = Vec::with_capacity(operands.len());
                    let mut current_prev_op = prev_op;

                    for operand in operands {
                        let (val, new_prev_op) = translate_operand(
                            ctx,
                            body,
                            operand,
                            value_map,
                            block_ptr,
                            current_prev_op,
                            loc.clone(),
                        )?;
                        capture_values.push(val);
                        current_prev_op = new_prev_op;
                    }

                    // Get the closure type
                    let closure_ty_rust =
                        rustc_public::ty::Ty::new_closure(*closure_def, substs.clone());
                    let closure_ty = types::translate_type(ctx, &closure_ty_rust)?;

                    if capture_values.is_empty() {
                        // ZST closure (no captures) - create empty struct
                        let op = Operation::new(
                            ctx,
                            MirConstructStructOp::get_concrete_op_info(),
                            vec![closure_ty],
                            vec![],
                            vec![],
                            0,
                        );
                        op.deref_mut(ctx).set_loc(loc);
                        let result = op.deref(ctx).get_result(0);
                        Ok((Some(op), result, current_prev_op))
                    } else {
                        // Closure with captures - create struct with captured values
                        // Cast captured values to expected types (address space normalization)
                        let (casted_capture_values, prev_after_casts) =
                            cast_struct_fields_to_expected_types(
                                ctx,
                                capture_values,
                                closure_ty,
                                block_ptr,
                                current_prev_op,
                                loc.clone(),
                            );

                        let op = Operation::new(
                            ctx,
                            MirConstructStructOp::get_concrete_op_info(),
                            vec![closure_ty],
                            casted_capture_values,
                            vec![],
                            0,
                        );
                        op.deref_mut(ctx).set_loc(loc);
                        let result = op.deref(ctx).get_result(0);
                        Ok((Some(op), result, prev_after_casts))
                    }
                }
                mir::AggregateKind::RawPtr(pointee_ty, mutability) => {
                    // Raw pointer construction from parts: rustc lowers the
                    // `aggregate_raw_ptr` intrinsic to this aggregate kind.
                    // It is reached by re-slicing (`&bytes[2..]` goes through
                    // `slice::index::get_offset_len_noubcheck`) and by
                    // `ptr::slice_from_raw_parts` / `ptr::from_raw_parts`.
                    // The two operands are (data_pointer, metadata).
                    use rustc_public::mir::Mutability;
                    use rustc_public::ty::{RigidTy, TyKind};

                    if operands.len() != 2 {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(format!(
                                "RawPtr aggregate expected 2 operands (data, metadata), found {}",
                                operands.len()
                            ))
                        );
                    }

                    let is_mutable = matches!(mutability, Mutability::Mut);

                    match pointee_ty.kind() {
                        TyKind::RigidTy(RigidTy::Slice(elem_ty)) => {
                            // `*const [T]` / `*mut [T]`: the metadata operand is
                            // the element count. `*const [T]` translates to
                            // `MirSliceType` (same runtime layout as `&[T]`), so
                            // build the fat pointer with `mir.construct_slice`.
                            let element_type = types::translate_type(ctx, &elem_ty)?;

                            let (data_val, prev_after_data) = translate_operand(
                                ctx,
                                body,
                                &operands[0],
                                value_map,
                                block_ptr,
                                prev_op,
                                loc.clone(),
                            )?;
                            let (len_val, prev_after_len) = translate_operand(
                                ctx,
                                body,
                                &operands[1],
                                value_map,
                                block_ptr,
                                prev_after_data,
                                loc.clone(),
                            )?;

                            // The fat pointer's data slot is a generic-addrspace
                            // pointer. Values coming from shared memory carry
                            // addrspace(3); normalize them like the struct/array
                            // arms do.
                            let expected_ptr_ty: Ptr<TypeObj> =
                                dialect_mir::types::MirPtrType::get_generic(
                                    ctx,
                                    element_type,
                                    is_mutable,
                                )
                                .into();
                            let (data_val, current_prev_op) = cast_to_generic_addrspace_if_needed(
                                ctx,
                                data_val,
                                expected_ptr_ty,
                                block_ptr,
                                prev_after_len,
                                loc.clone(),
                            );

                            let slice_ty = dialect_mir::types::MirSliceType::get(ctx, element_type);

                            use dialect_mir::ops::MirConstructSliceOp;
                            let op = Operation::new(
                                ctx,
                                MirConstructSliceOp::get_concrete_op_info(),
                                vec![slice_ty.into()],
                                vec![data_val, len_val],
                                vec![],
                                0,
                            );
                            op.deref_mut(ctx).set_loc(loc);

                            let result = op.deref(ctx).get_result(0);

                            Ok((Some(op), result, current_prev_op))
                        }
                        TyKind::RigidTy(RigidTy::Str) => {
                            // Blocked on `str` having a device-side type
                            // translation (issue #76).
                            input_err!(
                                loc,
                                TranslationErr::unsupported(
                                    "RawPtr aggregate with `str` pointee not yet supported \
                                     (no `str` type translation on device)"
                                        .to_string()
                                )
                            )
                        }
                        TyKind::RigidTy(RigidTy::Dynamic(..)) => {
                            // Trait objects need a vtable, which has no
                            // device-side story.
                            input_err!(
                                loc,
                                TranslationErr::unsupported(
                                    "RawPtr aggregate with `dyn Trait` pointee not supported \
                                     (no vtable support on device)"
                                        .to_string()
                                )
                            )
                        }
                        _ => {
                            // `Sized` pointee: the metadata operand is `()`, so
                            // the aggregate is just the data pointer re-typed as
                            // `*const P` / `*mut P`. Confirm the metadata really
                            // is unit before dropping it; an unsized-tail struct
                            // pointee would carry real metadata here.
                            let metadata_ty = operands[1].ty(body.locals()).map_err(|e| {
                                input_error!(
                                    loc.clone(),
                                    TranslationErr::unsupported(format!(
                                        "Cannot get RawPtr aggregate metadata type: {e}"
                                    ))
                                )
                            })?;
                            let metadata_is_unit = matches!(
                                metadata_ty.kind(),
                                TyKind::RigidTy(RigidTy::Tuple(tys)) if tys.is_empty()
                            );
                            if !metadata_is_unit {
                                return input_err!(
                                    loc,
                                    TranslationErr::unsupported(format!(
                                        "RawPtr aggregate with non-unit metadata of type {:?} \
                                         not yet supported",
                                        metadata_ty
                                    ))
                                );
                            }

                            // Translate the target pointer type through the same
                            // path as the destination local, so the two agree
                            // (including SharedArray/Barrier special cases).
                            let raw_ptr_ty_rust =
                                rustc_public::ty::Ty::new_ptr(*pointee_ty, *mutability);
                            let target_ty = types::translate_type(ctx, &raw_ptr_ty_rust)?;

                            let (data_val, current_prev_op) = translate_operand(
                                ctx,
                                body,
                                &operands[0],
                                value_map,
                                block_ptr,
                                prev_op,
                                loc.clone(),
                            )?;

                            if data_val.get_type(ctx) == target_ty {
                                // Already the right pointer type: pass through.
                                Ok((None, data_val, current_prev_op))
                            } else {
                                // Pointer re-typing, e.g. `*const ()` data
                                // pointer becoming `*const P`.
                                let cast_op = Operation::new(
                                    ctx,
                                    MirCastOp::get_concrete_op_info(),
                                    vec![target_ty],
                                    vec![data_val],
                                    vec![],
                                    0,
                                );
                                cast_op.deref_mut(ctx).set_loc(loc);
                                MirCastOp::new(cast_op)
                                    .set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);

                                let result = cast_op.deref(ctx).get_result(0);

                                Ok((Some(cast_op), result, current_prev_op))
                            }
                        }
                    }
                }
                _ => {
                    input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "Aggregate kind {:?} not yet supported",
                            aggregate_kind
                        ))
                    )
                }
            }
        }
        mir::Rvalue::Discriminant(place) => {
            // Get the discriminant (tag) from an enum value.
            //
            // Two discriminant types are in play:
            //   - `native_tag_ty`: the physical tag type stored in memory,
            //     tracked by `MirEnumType::discriminant_type()` (e.g. `u8`
            //     for a niche-optimized `Option<*mut T>`).
            //   - `mir_discr_ty`: the type stable-MIR declares for the
            //     `Rvalue::Discriminant(place)` value itself, via
            //     `Ty::kind().discriminant_ty()`. This is what rustc uses
            //     to type the destination local (often `i64`).
            //
            // `MirGetDiscriminantOp` returns the native tag. When the two
            // types disagree we widen via `mir.cast IntToInt` so the rvalue
            // matches what stable-MIR promised. Without this, storing the
            // result into its destination slot would fail verification.
            use dialect_mir::ops::MirGetDiscriminantOp;
            use dialect_mir::types::MirEnumType;
            use pliron::builtin::types::IntegerType;
            use pliron::printable::Printable;

            let (enum_val, prev_op_after) =
                translate_place(ctx, body, place, value_map, block_ptr, prev_op, loc.clone())?;

            let enum_ty = enum_val.get_type(ctx);
            let native_tag_ty = {
                let enum_ty_obj = enum_ty.deref(ctx);
                if let Some(enum_type) = enum_ty_obj.downcast_ref::<MirEnumType>() {
                    enum_type.discriminant_type()
                } else {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "Discriminant on non-enum type: {}",
                            enum_ty.disp(ctx)
                        ))
                    );
                }
            };

            let get_disc_op = Operation::new(
                ctx,
                MirGetDiscriminantOp::get_concrete_op_info(),
                vec![native_tag_ty],
                vec![enum_val],
                vec![],
                0,
            );
            get_disc_op.deref_mut(ctx).set_loc(loc.clone());
            let native_result = get_disc_op.deref(ctx).get_result(0);

            // Ask stable-MIR what the declared discriminant type of this
            // place is. For well-formed MIR on an enum place this should
            // always succeed; if we can't compute it, fall back to the
            // native tag (no cast). In the fallback path we preserve the
            // original contract: the caller inserts `get_disc_op`.
            let place_ty = place.ty(body.locals()).map_err(|e| {
                input_error!(
                    loc.clone(),
                    TranslationErr::unsupported(format!(
                        "Failed to resolve place type for Discriminant: {:?}",
                        e
                    ))
                )
            })?;
            let declared_discr_ty = place_ty.kind().discriminant_ty();

            let mir_discr_ty = match declared_discr_ty {
                Some(ty) => super::types::translate_type(ctx, &ty)?,
                None => {
                    return Ok((Some(get_disc_op), native_result, prev_op_after));
                }
            };

            // Only widen when both sides are integers and differ. Anything
            // else is either already matched or a dialect-level mismatch
            // that deserves its own verifier error upstream.
            let needs_cast = mir_discr_ty != native_tag_ty && {
                let src = native_tag_ty.deref(ctx);
                let dst = mir_discr_ty.deref(ctx);
                src.is::<IntegerType>() && dst.is::<IntegerType>()
            };

            if !needs_cast {
                return Ok((Some(get_disc_op), native_result, prev_op_after));
            }

            // Widening path: we emit two ops (get_discriminant + cast) and
            // the caller only inserts a single "main" op. Insert both here
            // and return `None` as the main op so the caller does not try
            // to re-insert.
            if let Some(prev) = prev_op_after {
                get_disc_op.insert_after(ctx, prev);
            } else {
                get_disc_op.insert_at_front(block_ptr, ctx);
            }

            let cast_op = Operation::new(
                ctx,
                MirCastOp::get_concrete_op_info(),
                vec![mir_discr_ty],
                vec![native_result],
                vec![],
                0,
            );
            cast_op.deref_mut(ctx).set_loc(loc);
            MirCastOp::new(cast_op).set_attr_cast_kind(ctx, MirCastKindAttr::IntToInt);
            cast_op.insert_after(ctx, get_disc_op);

            let result = cast_op.deref(ctx).get_result(0);
            Ok((None, result, Some(cast_op)))
        }
        mir::Rvalue::Repeat(operand, count) => {
            // Array repeat: [value; N] -> mir.construct_array with N copies of value
            //
            // MIR: _1 = Repeat(Constant(0.0f32), 16)
            // Produces: [0.0, 0.0, 0.0, ...] (16 elements)

            // Extract the count from TyConst
            let array_size = count.eval_target_usize().map_err(|e| {
                input_error!(
                    loc.clone(),
                    TranslationErr::unsupported(format!(
                        "Failed to evaluate Repeat count: {:?}",
                        e
                    ))
                )
            })?;

            // Translate the operand to get the element value
            let (element_val, prev_op_after_operand) = translate_operand(
                ctx,
                body,
                operand,
                value_map,
                block_ptr,
                prev_op,
                loc.clone(),
            )?;

            // Get the element type from the value
            let element_type = element_val.get_type(ctx);

            // Create element values by repeating the single value
            let element_values: Vec<Value> =
                std::iter::repeat_n(element_val, array_size as usize).collect();

            // Create the array type
            let array_ty = dialect_mir::types::MirArrayType::get(ctx, element_type, array_size);

            // Create mir.construct_array operation
            let op = Operation::new(
                ctx,
                MirConstructArrayOp::get_concrete_op_info(),
                vec![array_ty.into()],
                element_values,
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc);

            let result = op.deref(ctx).get_result(0);

            Ok((Some(op), result, prev_op_after_operand))
        }
        _ => {
            // TODO (npasham): Handle other Rvalue variants
            input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "Rvalue variant {:?} not yet implemented",
                    rvalue
                ))
            )
        }
    }
}

/// Translate a MIR Operand to a pliron IR [`Value`].
/// Returns the value and the last inserted operation (for proper ordering).
///
/// Handles Copy, Move (via translate_place) and Constant operands.
pub fn translate_operand(
    ctx: &mut Context,
    body: &mir::Body,
    operand: &mir::Operand,
    value_map: &mut ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    match operand {
        mir::Operand::Copy(place) | mir::Operand::Move(place) => {
            // Get the value from the place
            translate_place(ctx, body, place, value_map, block_ptr, prev_op, loc)
        }
        mir::Operand::Constant(constant) => {
            // Get the Rust type of this constant
            let rust_ty = constant.ty();

            // Check if this is a pointer to SharedArray (static shared memory)
            if is_shared_array_pointer(&rust_ty) {
                // Extract element type, size, and alignment from SharedArray<T, N, ALIGN>
                let (elem_ty, array_size, alignment) = extract_shared_array_info(ctx, &rust_ty)?;

                // Create a shared memory pointer type
                let ptr_ty = dialect_mir::types::MirPtrType::get_shared(ctx, elem_ty, true).into();

                // Create a MirSharedAllocOp to represent the shared memory allocation
                // This will be lowered to an LLVM global with addrspace(3)
                //
                // NOTE: We include the alloc key in the operation so the LLVM lowering
                // phase can deduplicate multiple references to the same static.
                use dialect_mir::ops::MirSharedAllocOp;
                let op = Operation::new(
                    ctx,
                    MirSharedAllocOp::get_concrete_op_info(),
                    vec![ptr_ty],
                    vec![],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc);

                let shared_alloc = MirSharedAllocOp::new(op);

                // Set the element type, size, and alloc key attributes
                use pliron::builtin::attributes::{IntegerAttr, StringAttr, TypeAttr};
                shared_alloc.set_attr_elem_type(ctx, TypeAttr::new(elem_ty));
                let size_attr = IntegerAttr::new(
                    pliron::builtin::types::IntegerType::get(
                        ctx,
                        64,
                        pliron::builtin::types::Signedness::Signless,
                    ),
                    pliron::utils::apint::APInt::from_u64(
                        array_size as u64,
                        std::num::NonZeroUsize::new(64).unwrap(),
                    ),
                );
                shared_alloc.set_attr_size(ctx, size_attr);

                // Store the alloc key so lowering can deduplicate
                let alloc_key = format!("{:?}", constant.const_);
                shared_alloc.set_attr_alloc_key(ctx, StringAttr::new(alloc_key));

                // Set alignment if specified (non-zero)
                if alignment > 0 {
                    shared_alloc.set_alignment_value(ctx, alignment as u64);
                }

                if let Some(prev) = prev_op {
                    shared_alloc.get_operation().insert_after(ctx, prev);
                } else {
                    shared_alloc.get_operation().insert_at_front(block_ptr, ctx);
                }

                let val = shared_alloc.get_operation().deref(ctx).get_result(0);

                return Ok((val, Some(shared_alloc.get_operation())));
            }

            // Check if this is a pointer to Barrier (static barrier in shared memory)
            if is_barrier_pointer(&rust_ty) {
                // Barrier is a single 64-bit value in shared memory (mbarrier state)
                let elem_ty = pliron::builtin::types::IntegerType::get(
                    ctx,
                    64,
                    pliron::builtin::types::Signedness::Unsigned,
                )
                .into();

                // Create a shared memory pointer type (addrspace 3)
                let ptr_ty = dialect_mir::types::MirPtrType::get_shared(ctx, elem_ty, true).into();

                // Create a MirSharedAllocOp for the barrier
                use dialect_mir::ops::MirSharedAllocOp;
                let op = Operation::new(
                    ctx,
                    MirSharedAllocOp::get_concrete_op_info(),
                    vec![ptr_ty],
                    vec![],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc);

                let shared_alloc = MirSharedAllocOp::new(op);

                // Set attributes: element type (i64), size (1 element)
                use pliron::builtin::attributes::{IntegerAttr, StringAttr, TypeAttr};
                shared_alloc.set_attr_elem_type(ctx, TypeAttr::new(elem_ty));
                let size_attr = IntegerAttr::new(
                    pliron::builtin::types::IntegerType::get(
                        ctx,
                        64,
                        pliron::builtin::types::Signedness::Signless,
                    ),
                    pliron::utils::apint::APInt::from_u64(
                        1, // Single barrier element
                        std::num::NonZeroUsize::new(64).unwrap(),
                    ),
                );
                shared_alloc.set_attr_size(ctx, size_attr);

                // Store the alloc key so lowering can deduplicate
                let alloc_key = format!("{:?}", constant.const_);
                shared_alloc.set_attr_alloc_key(ctx, StringAttr::new(alloc_key));

                if let Some(prev) = prev_op {
                    shared_alloc.get_operation().insert_after(ctx, prev);
                } else {
                    shared_alloc.get_operation().insert_at_front(block_ptr, ctx);
                }

                let val = shared_alloc.get_operation().deref(ctx).get_result(0);

                return Ok((val, Some(shared_alloc.get_operation())));
            }

            // Ordinary Rust `static` / `static mut` values in device code live in
            // CUDA global memory (addrspace 1) by default. SharedArray/Barrier
            // statics have already been intercepted above and remain addrspace 3.
            // Statics tagged `#[constant]` (detected by the mangled symbol
            // prefix) instead lower into constant memory (addrspace 4).
            if let Some(static_def) = static_def_from_constant(constant)?
                && let Some((pointee_ty, is_mutable)) = get_static_pointer_info(&rust_ty)
            {
                // All device-side statics — `#[constant]` and ordinary — must
                // currently be zero-initialized. Lowering honored initializers
                // into PTX `.const` (or `.global`) bytes is on the roadmap;
                // for now use `ConstantMemory::UNINIT` and populate from host.
                ensure_zero_initializer(&static_def, loc.clone())?;
                let is_constant = is_constant_wrapper_type(&pointee_ty);

                // Constants need the linker-visible mangled name (honors
                // `#[export_name]`) so mir-lower can emit a matching LLVM
                // symbol that the host resolves via `cuModuleGetGlobal`.
                // Ordinary statics only need a unique key for in-pass
                // deduplication, so we take the cheaper definition-path name.
                let global_key: String = if is_constant {
                    rustc_public::mir::mono::Instance::from(static_def)
                        .mangled_name()
                        .to_string()
                } else {
                    static_def.name()
                };

                let global_ty = types::translate_type(ctx, &pointee_ty)?;
                let ptr_ty = if is_constant {
                    dialect_mir::types::MirPtrType::get_constant(ctx, global_ty, is_mutable).into()
                } else {
                    dialect_mir::types::MirPtrType::get_global(ctx, global_ty, is_mutable).into()
                };

                let op = Operation::new(
                    ctx,
                    MirGlobalAllocOp::get_concrete_op_info(),
                    vec![ptr_ty],
                    vec![],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc);

                let global_alloc = MirGlobalAllocOp::new(op);

                use pliron::builtin::attributes::{StringAttr, TypeAttr};
                global_alloc.set_attr_global_type(ctx, TypeAttr::new(global_ty));
                global_alloc.set_attr_global_key(ctx, StringAttr::new(global_key));

                if let Some(alignment) = static_alignment(&static_def)? {
                    global_alloc.set_alignment_value(ctx, alignment);
                }

                if let Some(prev) = prev_op {
                    global_alloc.get_operation().insert_after(ctx, prev);
                } else {
                    global_alloc.get_operation().insert_at_front(block_ptr, ctx);
                }

                let val = global_alloc.get_operation().deref(ctx).get_result(0);

                return Ok((val, Some(global_alloc.get_operation())));
            }

            let const_ty_ptr = types::translate_type(ctx, &rust_ty)?;

            // Check if this is a ZST (Zero-Sized Type) like PhantomData<T>
            // ZSTs have no runtime representation, so we create a value of the appropriate type.
            // This is critical for iterator support (Iter contains PhantomData).
            if types::is_zst_type(ctx, const_ty_ptr) {
                // Determine if this is a struct ZST (like PhantomData) or tuple ZST
                let is_struct_zst = const_ty_ptr
                    .deref(ctx)
                    .is::<dialect_mir::types::MirStructType>();

                let op = if is_struct_zst {
                    // Create empty struct constructor for struct ZSTs (e.g., PhantomData<T>)
                    Operation::new(
                        ctx,
                        MirConstructStructOp::get_concrete_op_info(),
                        vec![const_ty_ptr], // Use the actual struct type
                        vec![],             // No operands for ZST
                        vec![],
                        0,
                    )
                } else {
                    // Create empty tuple constructor for tuple ZSTs
                    use dialect_mir::ops::MirConstructTupleOp;
                    let empty_tuple_ty = dialect_mir::types::MirTupleType::get(ctx, vec![]).into();
                    Operation::new(
                        ctx,
                        MirConstructTupleOp::get_concrete_op_info(),
                        vec![empty_tuple_ty],
                        vec![], // No operands for ZST
                        vec![],
                        0,
                    )
                };
                op.deref_mut(ctx).set_loc(loc);

                if let Some(prev) = prev_op {
                    op.insert_after(ctx, prev);
                } else {
                    op.insert_at_front(block_ptr, ctx);
                }

                let val = op.deref(ctx).get_result(0);
                return Ok((val, Some(op)));
            }

            // Check if this is a struct type (non-ZST)
            // For struct constants, we need to construct the struct from its field values.
            let is_struct = const_ty_ptr
                .deref(ctx)
                .is::<dialect_mir::types::MirStructType>();
            let is_tuple = const_ty_ptr
                .deref(ctx)
                .is::<dialect_mir::types::MirTupleType>();

            // Check if this is a float type (f16, f32, or f64)
            let is_float_16 = const_ty_ptr.deref(ctx).is::<MirFP16Type>();
            let is_float_32 = const_ty_ptr.deref(ctx).is::<FP32Type>();
            let is_float_64 = const_ty_ptr.deref(ctx).is::<FP64Type>();
            let is_float = is_float_16 || is_float_32 || is_float_64;

            // Check if this is an enum type
            let is_enum = const_ty_ptr
                .deref(ctx)
                .is::<dialect_mir::types::MirEnumType>();

            // Check if this is a pointer to an array (byte strings, or typed arrays like [f64; 3])
            let is_ptr_to_array = const_ty_ptr
                .deref(ctx)
                .downcast_ref::<dialect_mir::types::MirPtrType>()
                .map(|ptr_ty| {
                    ptr_ty
                        .pointee
                        .deref(ctx)
                        .is::<dialect_mir::types::MirArrayType>()
                })
                .unwrap_or(false);

            // Parse constant value from debug string (HACK for prototype)
            let const_str = format!("{:?}", constant.const_);

            // Handle pointer-to-array constants (byte strings, typed arrays like [f64; 3], etc.)
            if is_ptr_to_array {
                return translate_ptr_to_array_constant(
                    ctx,
                    constant,
                    const_ty_ptr,
                    block_ptr,
                    prev_op,
                    loc,
                );
            }

            if is_struct {
                // Non-ZST struct constant - extract field values and construct the struct
                translate_struct_constant(
                    ctx,
                    constant,
                    &rust_ty,
                    const_ty_ptr,
                    block_ptr,
                    prev_op,
                    loc,
                )
            } else if is_tuple {
                translate_tuple_constant(
                    ctx,
                    constant,
                    &rust_ty,
                    const_ty_ptr,
                    block_ptr,
                    prev_op,
                    loc,
                )
            } else if is_enum {
                translate_enum_constant(
                    ctx,
                    constant,
                    &rust_ty,
                    const_ty_ptr,
                    block_ptr,
                    prev_op,
                    loc,
                )
            } else if is_float {
                // Parse bytes for float (f16, f32, or f64)
                use dialect_mir::ops::MirFloatConstantOp;

                if is_float_16 {
                    let bytes = constant_bytes(constant, "f16", loc.clone())?;
                    if bytes.len() < 2 {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(format!(
                                "f16 constant needs 2 bytes, found {}",
                                bytes.len()
                            ))
                        );
                    }
                    let bits = read_uint_from_bytes(&bytes[..2]) as u16;
                    let float_attr = MirFP16Attr::from_bits(bits);

                    let op = Operation::new(
                        ctx,
                        MirFloatConstantOp::get_concrete_op_info(),
                        vec![const_ty_ptr],
                        vec![],
                        vec![],
                        0,
                    );
                    op.deref_mut(ctx).set_loc(loc);

                    let float_op = MirFloatConstantOp::new(op);
                    float_op.set_attr_float_value_f16(ctx, float_attr);

                    if let Some(prev) = prev_op {
                        float_op.get_operation().insert_after(ctx, prev);
                    } else {
                        float_op.get_operation().insert_at_front(block_ptr, ctx);
                    }

                    let val = float_op.get_operation().deref(ctx).get_result(0);

                    Ok((val, Some(float_op.get_operation())))
                } else if is_float_64 {
                    // Handle f64 (8 bytes)
                    let float_val = if const_str.contains("bytes: [") {
                        if let Some(bytes_part) = const_str.split("bytes: [").nth(1) {
                            let bytes_end = bytes_part.split(']').next().unwrap_or("");
                            let mut bytes = [0u8; 8];
                            for (i, byte_str) in bytes_end.split(',').enumerate() {
                                if i >= 8 {
                                    break;
                                }
                                let b_str = byte_str.trim();
                                if let Some(num_str) = b_str
                                    .strip_prefix("Some(")
                                    .and_then(|s| s.strip_suffix(')'))
                                    && let Ok(byte) = num_str.parse::<u8>()
                                {
                                    bytes[i] = byte;
                                }
                            }
                            f64::from_le_bytes(bytes)
                        } else {
                            0.0f64
                        }
                    } else {
                        // Try to parse as literal float
                        const_str
                            .split(':')
                            .next()
                            .unwrap_or("0.0")
                            .trim()
                            .replace('_', "")
                            .parse()
                            .unwrap_or(0.0f64)
                    };

                    let float_attr = pliron::builtin::attributes::FPDoubleAttr::from(float_val);

                    let op = Operation::new(
                        ctx,
                        MirFloatConstantOp::get_concrete_op_info(),
                        vec![const_ty_ptr],
                        vec![],
                        vec![],
                        0,
                    );
                    op.deref_mut(ctx).set_loc(loc.clone());

                    let float_op = MirFloatConstantOp::new(op);
                    float_op.set_attr_float_value_f64(ctx, float_attr);

                    if let Some(prev) = prev_op {
                        float_op.get_operation().insert_after(ctx, prev);
                    } else {
                        float_op.get_operation().insert_at_front(block_ptr, ctx);
                    }

                    let val = float_op.get_operation().deref(ctx).get_result(0);

                    Ok((val, Some(float_op.get_operation())))
                } else {
                    // Handle f32 (4 bytes)
                    let float_val = if const_str.contains("bytes: [") {
                        if let Some(bytes_part) = const_str.split("bytes: [").nth(1) {
                            let bytes_end = bytes_part.split(']').next().unwrap_or("");
                            let mut bytes = [0u8; 4];
                            for (i, byte_str) in bytes_end.split(',').enumerate() {
                                if i >= 4 {
                                    break;
                                }
                                let b_str = byte_str.trim();
                                if let Some(num_str) = b_str
                                    .strip_prefix("Some(")
                                    .and_then(|s| s.strip_suffix(')'))
                                    && let Ok(byte) = num_str.parse::<u8>()
                                {
                                    bytes[i] = byte;
                                }
                            }
                            f32::from_le_bytes(bytes)
                        } else {
                            0.0f32
                        }
                    } else {
                        // Try to parse as literal float
                        const_str
                            .split(':')
                            .next()
                            .unwrap_or("0.0")
                            .trim()
                            .replace('_', "")
                            .parse()
                            .unwrap_or(0.0f32)
                    };

                    let float_attr = pliron::builtin::attributes::FPSingleAttr::from(float_val);

                    let op = Operation::new(
                        ctx,
                        MirFloatConstantOp::get_concrete_op_info(),
                        vec![const_ty_ptr],
                        vec![],
                        vec![],
                        0,
                    );
                    op.deref_mut(ctx).set_loc(loc);

                    let float_op = MirFloatConstantOp::new(op);
                    float_op.set_attr_float_value(ctx, float_attr);

                    if let Some(prev) = prev_op {
                        float_op.get_operation().insert_after(ctx, prev);
                    } else {
                        float_op.get_operation().insert_at_front(block_ptr, ctx);
                    }

                    let val = float_op.get_operation().deref(ctx).get_result(0);

                    Ok((val, Some(float_op.get_operation())))
                }
            } else if const_ty_ptr
                .deref(ctx)
                .is::<dialect_mir::types::MirPtrType>()
            {
                // Pointer type constant - could be:
                // 1. A raw pointer constant (like core::ptr::null()) - just bytes,
                //    no provenance
                // 2. A reference to a constant struct (like &(8..16)) - need
                //    struct + mir.ref
                // 3. A reference to any other promoted constant (like the `&77`
                //    that -O const-folds out of `Option<&u32>::unwrap_or(&77)`,
                //    issue #132) - follow the allocation provenance, materialize
                //    the pointee constant, then mir.ref
                //
                // Only constants WITHOUT provenance may take the raw-pointer
                // path; a provenance entry always names a real allocation, and
                // ignoring it would lower the reference to `inttoptr 0` (a null
                // pointer).

                // Extract pointer type info before further borrows
                let (pointee_ty, is_mutable, pointee_is_struct) = {
                    let ty_ref = const_ty_ptr.deref(ctx);
                    let ptr_ty = ty_ref
                        .downcast_ref::<dialect_mir::types::MirPtrType>()
                        .unwrap();
                    let pointee = ptr_ty.pointee;
                    let mutable = ptr_ty.is_mutable;
                    let is_struct = pointee.deref(ctx).is::<dialect_mir::types::MirStructType>();
                    (pointee, mutable, is_struct)
                };

                // Check if the constant has actual struct data (not all zeros)
                // Handle both Allocated constants and promoted constants (Ty variant)
                //
                // Debug: print constant info for reference-to-struct types
                if pointee_is_struct && std::env::var("CUDA_OXIDE_DEBUG_CONST").is_ok() {
                    eprintln!(
                        "[DEBUG] Ptr-to-struct constant: kind={:?}, str={:?}",
                        constant.const_.kind(),
                        const_str
                    );
                }

                let has_struct_data = if pointee_is_struct {
                    match constant.const_.kind() {
                        ConstantKind::Allocated(alloc) => {
                            // For promoted constants like &(8..16), the bytes are zeros
                            // (pointer placeholder) but provenance indicates a real allocation.
                            // Check for provenance OR non-zero bytes.
                            let has_provenance = !alloc.provenance.ptrs.is_empty();
                            let has_nonzero_bytes = alloc
                                .raw_bytes()
                                .ok()
                                .map(|bytes| bytes.iter().any(|&b| b != 0))
                                .unwrap_or(false);
                            has_provenance || has_nonzero_bytes
                        }
                        ConstantKind::Ty(_) => {
                            // Promoted constants (like &(8..16)) are Ty variants
                            // These contain the actual struct data
                            true
                        }
                        _ => false,
                    }
                } else {
                    false
                };

                if has_struct_data {
                    // This is a reference to a constant struct (like &(8..16))

                    // Create the struct constant, then use mir.ref to get a pointer
                    let (struct_val, last_op) = translate_struct_constant(
                        ctx,
                        constant,
                        &rust_ty,
                        pointee_ty,
                        block_ptr,
                        prev_op,
                        loc.clone(),
                    )?;

                    // Now create mir.ref to get a pointer to the struct
                    use dialect_mir::ops::MirRefOp;
                    let ref_op = Operation::new(
                        ctx,
                        MirRefOp::get_concrete_op_info(),
                        vec![const_ty_ptr], // Result is pointer to struct
                        vec![struct_val],   // Operand is the struct value
                        vec![],
                        0,
                    );
                    ref_op.deref_mut(ctx).set_loc(loc);

                    let mir_ref = MirRefOp::new(ref_op);

                    mir_ref
                        .set_attr_mutable(ctx, dialect_mir::attributes::MutabilityAttr(is_mutable));

                    if let Some(prev) = last_op {
                        mir_ref.get_operation().insert_after(ctx, prev);
                    } else {
                        mir_ref.get_operation().insert_at_front(block_ptr, ctx);
                    }

                    let ptr_val = mir_ref.get_operation().deref(ctx).get_result(0);
                    return Ok((ptr_val, Some(mir_ref.get_operation())));
                }

                // Reference to a non-struct promoted constant (issue #132).
                //
                // Under -O, MIR const-folds e.g. the `None` arm of
                // `Option<&u32>::unwrap_or(&77)` into a constant of type `&u32`
                // whose data bytes are a pointer placeholder and whose
                // provenance entry names the allocation holding the literal
                // `77`. Struct pointees were already handled above; follow the
                // provenance for every other pointee type too, materialize the
                // pointee through the shared constant-from-bytes path, and take
                // its address with mir.ref (mem2reg/lowering turn that into an
                // alloca + store + address; sound because promoted constants
                // are immutable).
                let backing_alloc: Option<&rustc_public::ty::Allocation> =
                    match constant.const_.kind() {
                        ConstantKind::Allocated(alloc) => Some(alloc),
                        ConstantKind::Ty(ty_const) => match ty_const.kind() {
                            rustc_public::ty::TyConstKind::Value(_, alloc) => Some(alloc),
                            _ => None,
                        },
                        _ => None,
                    };

                if let Some(alloc) = backing_alloc
                    && let Some(&(prov_pos, prov)) = alloc.provenance.ptrs.first()
                {
                    use rustc_public::mir::alloc::GlobalAlloc;
                    let alloc_id = prov.0;

                    // The pointer's own data bytes encode the byte offset into
                    // the target allocation (zero for plain promoted literals
                    // like `&77`). The struct/array provenance branches assume
                    // offset zero; here the slice below honors a non-zero
                    // offset, and an unreadable offset is a hard error rather
                    // than a silently wrong address.
                    let ptr_width =
                        rustc_public::target::MachineInfo::target_pointer_width().bytes();
                    let target_offset = alloc
                        .read_partial_uint(prov_pos..prov_pos + ptr_width)
                        .map_err(|e| {
                            input_error_noloc!(TranslationErr::unsupported(format!(
                                "Failed to read pointer constant provenance offset: {:?}",
                                e
                            )))
                        })? as usize;

                    let target_bytes: Vec<u8> = match GlobalAlloc::from(alloc_id) {
                        GlobalAlloc::Memory(target_alloc) => {
                            target_alloc.raw_bytes().ok().unwrap_or_else(|| {
                                target_alloc
                                    .bytes
                                    .iter()
                                    .map(|opt: &Option<u8>| opt.unwrap_or(0))
                                    .collect::<Vec<u8>>()
                            })
                        }
                        GlobalAlloc::Static(static_def) => {
                            let target_alloc = static_def.eval_initializer().map_err(|e| {
                                input_error_noloc!(TranslationErr::unsupported(format!(
                                    "Failed to evaluate static initializer for pointer constant: {:?}",
                                    e
                                )))
                            })?;
                            target_alloc.raw_bytes().ok().unwrap_or_else(|| {
                                target_alloc
                                    .bytes
                                    .iter()
                                    .map(|opt: &Option<u8>| opt.unwrap_or(0))
                                    .collect::<Vec<u8>>()
                            })
                        }
                        other => {
                            return input_err!(
                                loc,
                                TranslationErr::unsupported(format!(
                                    "Pointer constant provenance points to non-memory allocation: {:?}",
                                    other
                                ))
                            );
                        }
                    };

                    if target_offset > target_bytes.len() {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(format!(
                                "Pointer constant provenance offset {} exceeds target allocation size {}",
                                target_offset,
                                target_bytes.len()
                            ))
                        );
                    }

                    // The shared materializer needs the pointee's Rust type for
                    // enum-layout queries and ZST detection.
                    let Some((pointee_rust_ty, _)) = get_static_pointer_info(&rust_ty) else {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(format!(
                                "Pointer constant with provenance has unsupported Rust type: {:?}",
                                rust_ty
                            ))
                        );
                    };

                    let (pointee_val, last_op) = translate_constant_value_from_bytes(
                        ctx,
                        &pointee_rust_ty,
                        pointee_ty,
                        &target_bytes[target_offset..],
                        block_ptr,
                        prev_op,
                        loc.clone(),
                    )?;

                    // Take the address of the materialized value, exactly like
                    // the struct branch above.
                    let ref_op = Operation::new(
                        ctx,
                        MirRefOp::get_concrete_op_info(),
                        vec![const_ty_ptr], // Result is pointer to the pointee
                        vec![pointee_val],  // Operand is the materialized value
                        vec![],
                        0,
                    );
                    ref_op.deref_mut(ctx).set_loc(loc);

                    let mir_ref = MirRefOp::new(ref_op);
                    mir_ref
                        .set_attr_mutable(ctx, dialect_mir::attributes::MutabilityAttr(is_mutable));

                    if let Some(prev) = last_op {
                        mir_ref.get_operation().insert_after(ctx, prev);
                    } else {
                        mir_ref.get_operation().insert_at_front(block_ptr, ctx);
                    }

                    let ptr_val = mir_ref.get_operation().deref(ctx).get_result(0);
                    return Ok((ptr_val, Some(mir_ref.get_operation())));
                }

                // Raw pointer constant (like core::ptr::null()).
                //
                // Only reachable for constants WITHOUT provenance (true null or
                // int-to-ptr values); provenance-carrying constants returned
                // above. Create an integer constant with the pointer value
                // (0 for null), then convert it to a pointer type using
                // MirCastOp
                use dialect_mir::ops::MirCastOp;

                // Parse the pointer value from the constant bytes (typically all zeros for null)
                let ptr_val = if const_str.contains("bytes: [") {
                    if let Some(bytes_part) = const_str.split("bytes: [").nth(1) {
                        let bytes_end = bytes_part.split(']').next().unwrap_or("");
                        let mut bytes = Vec::new();
                        for byte_str in bytes_end.split(',') {
                            if bytes.len() >= 8 {
                                break;
                            }
                            let b_str = byte_str.trim();
                            if let Some(num_str) = b_str
                                .strip_prefix("Some(")
                                .and_then(|s| s.strip_suffix(')'))
                                && let Ok(byte) = num_str.parse::<u8>()
                            {
                                bytes.push(byte);
                            }
                        }
                        let mut res: u64 = 0;
                        for (i, byte) in bytes.iter().enumerate() {
                            res |= (*byte as u64) << (i * 8);
                        }
                        res
                    } else {
                        0
                    }
                } else {
                    0 // Default to null pointer
                };

                // Create integer constant (i64) for the pointer value
                let i64_ty = pliron::builtin::types::IntegerType::get(
                    ctx,
                    64,
                    pliron::builtin::types::Signedness::Signless,
                );
                let apint = APInt::from_u64(ptr_val, NonZeroUsize::new(64).unwrap());
                let int_attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, apint);

                use dialect_mir::ops::MirConstantOp;
                let int_op = Operation::new(
                    ctx,
                    MirConstantOp::get_concrete_op_info(),
                    vec![i64_ty.into()],
                    vec![],
                    vec![],
                    0,
                );
                int_op.deref_mut(ctx).set_loc(loc.clone());

                let const_op = MirConstantOp::new(int_op);
                const_op.set_attr_value(ctx, int_attr);

                if let Some(prev) = prev_op {
                    const_op.get_operation().insert_after(ctx, prev);
                } else {
                    const_op.get_operation().insert_at_front(block_ptr, ctx);
                }

                let int_val = const_op.get_operation().deref(ctx).get_result(0);

                // Cast integer to pointer type using MirCastOp
                let cast_op = Operation::new(
                    ctx,
                    MirCastOp::get_concrete_op_info(),
                    vec![const_ty_ptr], // Result is the pointer type
                    vec![int_val],      // Operand is the integer value
                    vec![],
                    0,
                );
                cast_op.deref_mut(ctx).set_loc(loc);
                MirCastOp::new(cast_op)
                    .set_attr_cast_kind(ctx, MirCastKindAttr::PointerWithExposedProvenance);

                cast_op.insert_after(ctx, const_op.get_operation());

                let ptr_val_result = cast_op.deref(ctx).get_result(0);

                Ok((ptr_val_result, Some(cast_op)))
            } else if const_ty_ptr.deref(ctx).is::<IntegerType>() {
                // Integer constant
                let (width_val, signedness) = {
                    let const_ty_obj = const_ty_ptr.deref(ctx);
                    let int_ty = const_ty_obj
                        .downcast_ref::<IntegerType>()
                        .expect("already checked is::<IntegerType>()");
                    (int_ty.width(), int_ty.signedness())
                };

                let byte_size = (width_val as usize).div_ceil(8);
                let int_val = constant_bytes(constant, "integer", loc.clone())
                    .ok()
                    .and_then(|bytes| {
                        (bytes.len() >= byte_size)
                            .then(|| read_uint_from_bytes(&bytes[..byte_size]))
                    })
                    .unwrap_or_else(|| {
                        let val_str_base = const_str.split(':').next().unwrap_or("0").trim();
                        let val_str = val_str_base.split('_').next().unwrap_or("0").trim();
                        let val_clean: String = val_str
                            .chars()
                            .filter(|c| c.is_ascii_digit() || *c == '-')
                            .collect();
                        val_clean.parse::<i128>().unwrap_or(0) as u128
                    });

                let width = NonZeroUsize::new(width_val as usize).unwrap();
                let apint = APInt::from_u128(int_val, width);

                let int_attr = pliron::builtin::attributes::IntegerAttr::new(
                    pliron::builtin::types::IntegerType::get(ctx, width_val, signedness),
                    apint,
                );

                use dialect_mir::ops::MirConstantOp;
                let op = Operation::new(
                    ctx,
                    MirConstantOp::get_concrete_op_info(),
                    vec![const_ty_ptr],
                    vec![],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc);

                let const_op = MirConstantOp::new(op);
                const_op.set_attr_value(ctx, int_attr);

                if let Some(prev) = prev_op {
                    const_op.get_operation().insert_after(ctx, prev);
                } else {
                    const_op.get_operation().insert_at_front(block_ptr, ctx);
                }

                let val = const_op.get_operation().deref(ctx).get_result(0);

                Ok((val, Some(const_op.get_operation())))
            } else {
                // No matching type handler — report what we got so it's clear what needs support.
                let pliron_ty_dbg = format!("{:?}", const_ty_ptr.deref(ctx));
                Err(input_error_noloc!(TranslationErr::unsupported(format!(
                    "Unsupported constant type in translate_constant.\n\
                     \n  Rust type : {:?}\
                     \n  pliron type: {}\
                     \n  const repr : {}\
                     \n\
                     \nThe type dispatch (ZST -> ptr_to_array -> struct -> enum -> float -> pointer -> integer)\n\
                     did not match this constant. A new handler may need to be added.",
                    rust_ty, pliron_ty_dbg, const_str
                ))))
            }
        }
        mir::Operand::RuntimeChecks(_) => {
            // RuntimeChecks variants (UbChecks, ContractChecks, OverflowChecks)
            // evaluate to `false` on GPU -- runtime safety checks are disabled.
            //
            // Emits a `mir.constant false : i1` and inserts it into the current
            // block. The op *must* be linked before returning; callers use the
            // returned `last_op` as the insertion point for subsequent ops.
            use dialect_mir::ops::MirConstantOp;
            use pliron::builtin::attributes::IntegerAttr;
            use pliron::builtin::types::{IntegerType, Signedness};
            use pliron::utils::apint::APInt;

            let bool_ty: Ptr<TypeObj> = IntegerType::get(ctx, 1, Signedness::Signless).into();
            let false_val = APInt::from_u64(0, std::num::NonZeroUsize::new(1).unwrap());
            let const_attr =
                IntegerAttr::new(IntegerType::get(ctx, 1, Signedness::Signless), false_val);

            let op = Operation::new(
                ctx,
                MirConstantOp::get_concrete_op_info(),
                vec![bool_ty],
                vec![],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc);

            let const_op = MirConstantOp::new(op);
            const_op.set_attr_value(ctx, const_attr);

            match prev_op {
                Some(p) => op.insert_after(ctx, p),
                None => op.insert_at_front(block_ptr, ctx),
            }

            let val = const_op.get_operation().deref(ctx).get_result(0);

            Ok((val, Some(const_op.get_operation())))
        }
    }
}

/// Translate a MIR [`Place`](mir::Place) to its corresponding pliron IR SSA [`Value`].
///
/// For a simple local with no projections this is a lookup in `value_map`.
/// For projections (`field`, `index`, `deref`, `downcast`) the function
/// creates the necessary pliron IR operations and inserts them after `prev_op`.
///
/// # Ghost locals
///
/// A local may have no backing slot in `value_map` if rustc optimised away its
/// assignment, or if the local is ZST and has no runtime footprint.
///
/// When such a local is still *used* within a block (e.g. `discriminant(_6)`)
/// and happens to be an enum, we synthesise a variant-0 default via
/// `create_ghost_enum_default`. Non-enum ghost locals currently produce an
/// error -- extend this match if new patterns appear in future toolchains.
///
/// This is the SSA equivalent of rustc's codegen reading an uninitialized
/// alloca, which produces LLVM `undef`.
///
/// # Returns
///
/// `(value, last_inserted_op)` -- the pliron IR value for the place and the last
/// operation inserted into the block (for op-ordering bookkeeping).
pub fn translate_place(
    ctx: &mut Context,
    body: &mir::Body,
    place: &mir::Place,
    value_map: &ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    if place.projection.is_empty() {
        let local = place.local;
        // Alloca + load/store model: emit `mir.load slot`. Every non-ZST local
        // has a slot allocated in the entry block, so the loaded value is the
        // local's current contents. `mem2reg` promotes these loads back into
        // SSA form when the slot's address doesn't escape.
        if let Some((load_op, val)) = value_map.load_local(ctx, local, block_ptr, prev_op) {
            return Ok((val, Some(load_op)));
        }
        // ZST or unsupported local -- synthesise a value for it so callers
        // can uniformly consume a `Value`. An enum gets its variant-0 default
        // (ghost-enum), a struct/tuple ZST gets an empty aggregate. Loads of
        // these are otherwise meaningless.
        let local_decl = &body.locals()[local];
        let ty_ptr = types::translate_type(ctx, &local_decl.ty)?;
        if ty_ptr.deref(ctx).is::<dialect_mir::types::MirEnumType>() {
            let op = create_ghost_enum_default(ctx, ty_ptr, loc.clone());
            match prev_op {
                Some(p) => op.insert_after(ctx, p),
                None => op.insert_at_front(block_ptr, ctx),
            }
            let val = op.deref(ctx).get_result(0);
            return Ok((val, Some(op)));
        }
        if types::is_zst_type(ctx, ty_ptr) {
            let op = create_zst_aggregate(ctx, ty_ptr, loc.clone());
            match prev_op {
                Some(p) => op.insert_after(ctx, p),
                None => op.insert_at_front(block_ptr, ctx),
            }
            let val = op.deref(ctx).get_result(0);
            return Ok((val, Some(op)));
        }
        input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Local {} has no alloca slot and is not a ZST",
                Into::<usize>::into(local)
            ))
        )
    } else {
        // Handle projections (place.field, place[index], etc.)
        // For now, handle tuple field projections (_3.0, _3.1, etc.)
        if place.projection.len() == 1 {
            // Check if this is a tuple field projection
            match &place.projection[0] {
                ProjectionElem::Deref => {
                    // Dereference: *ptr
                    // The base value must be a pointer
                    let base_place = mir::Place {
                        local: place.local,
                        projection: vec![],
                    };
                    let (base_value, prev_op_after_base) = translate_place(
                        ctx,
                        body,
                        &base_place,
                        value_map,
                        block_ptr,
                        prev_op,
                        loc.clone(),
                    )?;

                    // Get the result type from the pointer's element type
                    let base_ty = base_value.get_type(ctx);

                    // Extract pointee info while holding the borrow, then release before fallback
                    let pointee_info: Option<(Ptr<pliron::r#type::TypeObj>, bool)> = {
                        let base_ty_ref = base_ty.deref(ctx);
                        base_ty_ref
                            .downcast_ref::<dialect_mir::types::MirPtrType>()
                            .map(|ptr_ty| {
                                let pointee = ptr_ty.pointee;
                                let pointee_ref = pointee.deref(ctx);

                                // Check if pointee is a ZST (empty tuple) - this happens for SharedArray
                                // which is a zero-sized type. For ZSTs, dereferencing just returns the
                                // same pointer (there's nothing to load).
                                let is_empty_tuple = pointee_ref
                                    .downcast_ref::<dialect_mir::types::MirTupleType>()
                                    .is_some_and(|tt| tt.get_types().is_empty());

                                (pointee, is_empty_tuple)
                            })
                    };

                    let (res_ty, is_zst) = pointee_info.unwrap_or_else(|| {
                        // Fallback: assume i32 if we can't determine the type
                        (types::get_i32_type(ctx).to_ptr(), false)
                    });

                    // For ZST pointees (like SharedArray), don't create a load op.
                    // Instead, just return the pointer itself - dereferencing a pointer
                    // to a ZST and taking a reference back gives the same pointer.
                    // NOTE: We still load from shared memory pointers (addrspace:3) -
                    // the ZST check only applies to SharedArray itself, not to data
                    // stored in shared memory.
                    if is_zst {
                        return Ok((base_value, prev_op_after_base));
                    }

                    let op = Operation::new(
                        ctx,
                        MirLoadOp::get_concrete_op_info(),
                        vec![res_ty],
                        vec![base_value],
                        vec![],
                        0,
                    );
                    op.deref_mut(ctx).set_loc(loc);

                    let load_op = MirLoadOp::new(op);

                    if let Some(prev) = prev_op_after_base {
                        load_op.get_operation().insert_after(ctx, prev);
                    } else {
                        load_op.get_operation().insert_at_front(block_ptr, ctx);
                    }

                    let loaded_val = load_op.get_operation().deref(ctx).get_result(0);

                    Ok((loaded_val, Some(load_op.get_operation())))
                }
                ProjectionElem::Field(field_idx, ty) => {
                    // Get the base value (the tuple/struct).
                    //
                    // In the alloca model the recursive call may emit a
                    // `mir.load <slot>` into the block to materialise the
                    // aggregate value; we must anchor our `mir.extract_field`
                    // **after** that load, otherwise the extract ends up
                    // before the load (and subsequent ops keep pushing the
                    // load past the block's terminator).
                    let base_place = mir::Place {
                        local: place.local,
                        projection: vec![],
                    };
                    let (base_value, prev_op_after_base) = translate_place(
                        ctx,
                        body,
                        &base_place,
                        value_map,
                        block_ptr,
                        prev_op,
                        loc.clone(),
                    )?;
                    let anchor = prev_op_after_base.or(prev_op);

                    let field_type = types::translate_type(ctx, ty)?;

                    let op = Operation::new(
                        ctx,
                        MirExtractFieldOp::get_concrete_op_info(),
                        vec![field_type],
                        vec![base_value],
                        vec![],
                        0,
                    );
                    op.deref_mut(ctx).set_loc(loc);

                    let extract_op = MirExtractFieldOp::new(op);
                    extract_op.set_attr_index(
                        ctx,
                        dialect_mir::attributes::FieldIndexAttr(*field_idx as u32),
                    );

                    if let Some(prev) = anchor {
                        extract_op.get_operation().insert_after(ctx, prev);
                    } else {
                        extract_op.get_operation().insert_at_front(block_ptr, ctx);
                    }

                    let field_value = extract_op.get_operation().deref(ctx).get_result(0);
                    Ok((field_value, Some(extract_op.get_operation())))
                }
                ProjectionElem::Downcast(_variant_idx) => {
                    // Downcast by itself is a no-op - it just narrows the type.
                    // The actual field extraction happens with the following Field projection.
                    // For now, just return the base value unchanged.
                    let base_place = mir::Place {
                        local: place.local,
                        projection: vec![],
                    };
                    translate_place(ctx, body, &base_place, value_map, block_ptr, prev_op, loc)
                }
                ProjectionElem::Index(index_local) => {
                    // Array indexing with a runtime index: array[index]
                    //
                    // Alloca model: `array` is backed by a stack slot whose
                    // pointee is `MirArrayType`, so we compute the element
                    // address from that slot directly (no MirRefOp needed)
                    // and load the element.

                    let mut current_prev = prev_op;

                    let Some(arr_ptr) = value_map.get_slot(place.local) else {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(format!(
                                "Array local {} has no alloca slot; cannot index",
                                Into::<usize>::into(place.local)
                            ))
                        );
                    };

                    // Get the index value
                    let index_place = mir::Place {
                        local: *index_local,
                        projection: vec![],
                    };
                    let (index_value, prev_op_after_index) = translate_place(
                        ctx,
                        body,
                        &index_place,
                        value_map,
                        block_ptr,
                        current_prev,
                        loc.clone(),
                    )?;
                    current_prev = prev_op_after_index;

                    // Get element type from pointer type
                    let arr_ptr_ty = arr_ptr.get_type(ctx);
                    let element_ty = {
                        let arr_ptr_ty_ref = arr_ptr_ty.deref(ctx);
                        let mir_ptr_ty = arr_ptr_ty_ref
                            .downcast_ref::<dialect_mir::types::MirPtrType>()
                            .expect("Memory array pointer should be MirPtrType");
                        let array_ty = mir_ptr_ty.pointee;
                        let array_ty_ref = array_ty.deref(ctx);
                        array_ty_ref
                            .downcast_ref::<dialect_mir::types::MirArrayType>()
                            .expect("Pointee should be MirArrayType")
                            .element_type()
                    };

                    // Get address space from array pointer
                    let address_space = {
                        let arr_ptr_ty_ref = arr_ptr_ty.deref(ctx);
                        arr_ptr_ty_ref
                            .downcast_ref::<dialect_mir::types::MirPtrType>()
                            .expect("Should be MirPtrType")
                            .address_space
                    };

                    // Create element pointer type
                    let elem_ptr_ty =
                        dialect_mir::types::MirPtrType::get(ctx, element_ty, false, address_space)
                            .into();

                    // Create MirArrayElementAddrOp to get element pointer
                    use dialect_mir::ops::MirArrayElementAddrOp;
                    let addr_op = Operation::new(
                        ctx,
                        MirArrayElementAddrOp::get_concrete_op_info(),
                        vec![elem_ptr_ty],
                        vec![arr_ptr, index_value],
                        vec![],
                        0,
                    );
                    addr_op.deref_mut(ctx).set_loc(loc.clone());

                    if let Some(prev) = current_prev {
                        addr_op.insert_after(ctx, prev);
                    } else {
                        addr_op.insert_at_front(block_ptr, ctx);
                    }
                    current_prev = Some(addr_op);

                    let elem_ptr = addr_op.deref(ctx).get_result(0);

                    // Load the element value
                    use dialect_mir::ops::MirLoadOp;
                    let load_op = Operation::new(
                        ctx,
                        MirLoadOp::get_concrete_op_info(),
                        vec![element_ty],
                        vec![elem_ptr],
                        vec![],
                        0,
                    );
                    load_op.deref_mut(ctx).set_loc(loc);

                    if let Some(prev) = current_prev {
                        load_op.insert_after(ctx, prev);
                    } else {
                        load_op.insert_at_front(block_ptr, ctx);
                    }

                    let result = load_op.deref(ctx).get_result(0);
                    Ok((result, Some(load_op)))
                }
                ProjectionElem::ConstantIndex {
                    offset,
                    min_length: _,
                    from_end,
                } => {
                    // Array indexing with a compile-time constant index.
                    //
                    // Alloca model: the array local already has a `*mut [T; N]`
                    // slot, so compute the element address via
                    // `MirConstantOp` + `MirArrayElementAddrOp` and load.
                    // `mem2reg` collapses the resulting load-after-store pairs
                    // back into SSA extracts for promotable arrays.

                    let index = if *from_end {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(
                                "ConstantIndex with from_end=true not yet supported"
                            )
                        );
                    } else {
                        *offset as usize
                    };

                    // Load the current array value if we don't have a slot (ZST/edge case)
                    // so that we fall back to the SSA extract-field behaviour.
                    let Some(arr_ptr) = value_map.get_slot(place.local) else {
                        // ZST / no-slot fallback: materialise the whole
                        // aggregate and extract. Anchor the extract after
                        // whatever the base-place materialiser inserted.
                        let base_place = mir::Place {
                            local: place.local,
                            projection: vec![],
                        };
                        let (array_value, prev_op_after_base) = translate_place(
                            ctx,
                            body,
                            &base_place,
                            value_map,
                            block_ptr,
                            prev_op,
                            loc.clone(),
                        )?;
                        let anchor = prev_op_after_base.or(prev_op);

                        let array_ty = array_value.get_type(ctx);
                        let element_ty = {
                            let array_ty_ref = array_ty.deref(ctx);
                            if let Some(arr_ty) =
                                array_ty_ref.downcast_ref::<dialect_mir::types::MirArrayType>()
                            {
                                arr_ty.element_type()
                            } else {
                                return input_err!(
                                    loc,
                                    TranslationErr::unsupported(format!(
                                        "ConstantIndex projection on non-array type: {}",
                                        array_ty.disp(ctx)
                                    ))
                                );
                            }
                        };

                        let op = Operation::new(
                            ctx,
                            MirExtractFieldOp::get_concrete_op_info(),
                            vec![element_ty],
                            vec![array_value],
                            vec![],
                            0,
                        );
                        op.deref_mut(ctx).set_loc(loc);

                        let extract_op = MirExtractFieldOp::new(op);
                        extract_op.set_attr_index(
                            ctx,
                            dialect_mir::attributes::FieldIndexAttr(index as u32),
                        );

                        if let Some(prev) = anchor {
                            extract_op.get_operation().insert_after(ctx, prev);
                        } else {
                            extract_op.get_operation().insert_at_front(block_ptr, ctx);
                        }

                        let result = extract_op.get_operation().deref(ctx).get_result(0);
                        return Ok((result, Some(extract_op.get_operation())));
                    };

                    // Slot-backed path: GEP + load from the slot.
                    let mut current_prev = prev_op;

                    let (element_ty, address_space) = {
                        let arr_ptr_ty = arr_ptr.get_type(ctx);
                        let arr_ptr_ty_ref = arr_ptr_ty.deref(ctx);
                        let mir_ptr_ty = arr_ptr_ty_ref
                            .downcast_ref::<dialect_mir::types::MirPtrType>()
                            .ok_or_else(|| {
                                input_error!(
                                    loc.clone(),
                                    TranslationErr::unsupported(format!(
                                        "ConstantIndex base slot is not a pointer: {}",
                                        arr_ptr_ty.disp(ctx)
                                    ))
                                )
                            })?;
                        let array_ty_ref = mir_ptr_ty.pointee.deref(ctx);
                        let elem_ty = array_ty_ref
                            .downcast_ref::<dialect_mir::types::MirArrayType>()
                            .ok_or_else(|| {
                                input_error_noloc!(TranslationErr::unsupported(
                                    "ConstantIndex base slot pointee is not MirArrayType"
                                ))
                            })?
                            .element_type();
                        (elem_ty, mir_ptr_ty.address_space)
                    };

                    use dialect_mir::ops::MirConstantOp;
                    use pliron::builtin::attributes::IntegerAttr;

                    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signed);
                    let index_apint = APInt::from_i64(index as i64, NonZeroUsize::new(64).unwrap());
                    let index_attr = IntegerAttr::new(i64_ty, index_apint);

                    let const_op_ptr = Operation::new(
                        ctx,
                        MirConstantOp::get_concrete_op_info(),
                        vec![i64_ty.into()],
                        vec![],
                        vec![],
                        0,
                    );
                    const_op_ptr.deref_mut(ctx).set_loc(loc.clone());
                    MirConstantOp::new(const_op_ptr).set_attr_value(ctx, index_attr);
                    if let Some(prev) = current_prev {
                        const_op_ptr.insert_after(ctx, prev);
                    } else {
                        const_op_ptr.insert_at_front(block_ptr, ctx);
                    }
                    current_prev = Some(const_op_ptr);
                    let index_value = const_op_ptr.deref(ctx).get_result(0);

                    let elem_ptr_ty =
                        dialect_mir::types::MirPtrType::get(ctx, element_ty, false, address_space)
                            .into();

                    use dialect_mir::ops::MirArrayElementAddrOp;
                    let addr_op = Operation::new(
                        ctx,
                        MirArrayElementAddrOp::get_concrete_op_info(),
                        vec![elem_ptr_ty],
                        vec![arr_ptr, index_value],
                        vec![],
                        0,
                    );
                    addr_op.deref_mut(ctx).set_loc(loc.clone());
                    if let Some(prev) = current_prev {
                        addr_op.insert_after(ctx, prev);
                    } else {
                        addr_op.insert_at_front(block_ptr, ctx);
                    }
                    current_prev = Some(addr_op);
                    let elem_ptr = addr_op.deref(ctx).get_result(0);

                    let load_op = Operation::new(
                        ctx,
                        MirLoadOp::get_concrete_op_info(),
                        vec![element_ty],
                        vec![elem_ptr],
                        vec![],
                        0,
                    );
                    load_op.deref_mut(ctx).set_loc(loc);
                    if let Some(prev) = current_prev {
                        load_op.insert_after(ctx, prev);
                    } else {
                        load_op.insert_at_front(block_ptr, ctx);
                    }

                    let result = load_op.deref(ctx).get_result(0);
                    Ok((result, Some(load_op)))
                }
                _ => input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Projection element {:?} not yet implemented",
                        place.projection[0]
                    ))
                ),
            }
        } else {
            // Multi-level projections (2+): use iterative processing.
            // The iterative path handles Deref on slices (extracts data pointer),
            // Index/ConstantIndex on both arrays and pointers, Field, Downcast, etc.
            translate_place_iterative(ctx, body, place, value_map, block_ptr, prev_op, loc)
        }
    }
}

// ============================================================================
// Iterative Projection Helpers
// ============================================================================
// These functions support the iterative processing of MIR projections.
// Each projection element is handled independently, allowing arbitrary depth.

/// Apply a Deref projection: load from pointer.
fn apply_deref_projection(
    ctx: &mut Context,
    ptr_value: Value,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let ptr_ty = ptr_value.get_type(ctx);

    enum DerefKind {
        Ptr {
            pointee: Ptr<pliron::r#type::TypeObj>,
            is_zst: bool,
        },
        Slice {
            element_ty: Ptr<pliron::r#type::TypeObj>,
        },
    }

    let deref_kind = {
        let ptr_ty_ref = ptr_ty.deref(ctx);
        if let Some(mir_ptr_ty) = ptr_ty_ref.downcast_ref::<dialect_mir::types::MirPtrType>() {
            let pointee = mir_ptr_ty.pointee;
            let is_zst = pointee
                .deref(ctx)
                .downcast_ref::<dialect_mir::types::MirTupleType>()
                .is_some_and(|tt| tt.get_types().is_empty());
            Some(DerefKind::Ptr { pointee, is_zst })
        } else {
            ptr_ty_ref
                .downcast_ref::<dialect_mir::types::MirSliceType>()
                .map(|slice_ty| DerefKind::Slice {
                    element_ty: slice_ty.element_type(),
                })
        }
    };

    let deref_kind = deref_kind.ok_or_else(|| {
        let ty_dbg = format!("{:?}", ptr_ty.deref(ctx));
        input_error_noloc!(TranslationErr::unsupported(format!(
            "Deref projection on unsupported type in apply_deref_projection.\n\
             \n  pliron type: {}\n\
             \n  display    : {}\n\
             \n\
             \nDeref currently handles MirPtrType (thin pointer load) and MirSliceType\n\
             (fat pointer → extract data pointer). The type above matched neither.\n\
             A new handler may need to be added.",
            ty_dbg,
            ptr_ty.disp(ctx)
        )))
    })?;

    match deref_kind {
        DerefKind::Ptr { pointee, is_zst } => {
            if is_zst {
                return Ok((ptr_value, prev_op));
            }

            let op = Operation::new(
                ctx,
                MirLoadOp::get_concrete_op_info(),
                vec![pointee],
                vec![ptr_value],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc);
            let load_op = MirLoadOp::new(op);

            if let Some(prev) = prev_op {
                load_op.get_operation().insert_after(ctx, prev);
            } else {
                load_op.get_operation().insert_at_front(block_ptr, ctx);
            }

            Ok((
                load_op.get_operation().deref(ctx).get_result(0),
                Some(load_op.get_operation()),
            ))
        }

        DerefKind::Slice { element_ty } => {
            // Slices are unsized — we can't load `[T]` into an SSA value.
            // Extract the data pointer (field 0 of the fat pointer {ptr, len}).
            // Subsequent Index/ConstantIndex projections will do ptr arithmetic + load.
            let ptr_ty = dialect_mir::types::MirPtrType::get_generic(ctx, element_ty, false).into();

            let extract_op = Operation::new(
                ctx,
                MirExtractFieldOp::get_concrete_op_info(),
                vec![ptr_ty],
                vec![ptr_value],
                vec![],
                0,
            );
            extract_op.deref_mut(ctx).set_loc(loc);

            let extract = MirExtractFieldOp::new(extract_op);
            extract.set_attr_index(ctx, dialect_mir::attributes::FieldIndexAttr(0));

            if let Some(prev) = prev_op {
                extract.get_operation().insert_after(ctx, prev);
            } else {
                extract.get_operation().insert_at_front(block_ptr, ctx);
            }

            Ok((
                extract.get_operation().deref(ctx).get_result(0),
                Some(extract.get_operation()),
            ))
        }
    }
}

/// Apply a Field projection against a POINTER to the aggregate: compute the
/// field's address with `mir.field_addr` and load the field value.
///
/// Used when the projection walk holds an address rather than an aggregate
/// value, which happens after dereferencing a fat pointer (the unsized
/// pointee cannot be loaded whole, so the deref hands back the data
/// pointer; see `apply_deref_projection`).
fn apply_field_addr_and_load(
    ctx: &mut Context,
    aggregate_ptr: Value,
    field_idx: mir::FieldIdx,
    field_ty: &rustc_public::ty::Ty,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    use dialect_mir::ops::MirFieldAddrOp;

    let field_type = types::translate_type(ctx, field_ty)?;
    let field_ptr_ty: Ptr<TypeObj> =
        dialect_mir::types::MirPtrType::get_generic(ctx, field_type, false).into();

    let addr_op = Operation::new(
        ctx,
        MirFieldAddrOp::get_concrete_op_info(),
        vec![field_ptr_ty],
        vec![aggregate_ptr],
        vec![],
        0,
    );
    addr_op.deref_mut(ctx).set_loc(loc.clone());
    MirFieldAddrOp::new(addr_op).set_attr_field_index(
        ctx,
        dialect_mir::attributes::FieldIndexAttr(field_idx as u32),
    );
    match prev_op {
        Some(p) => addr_op.insert_after(ctx, p),
        None => addr_op.insert_at_front(block_ptr, ctx),
    }
    let field_ptr = addr_op.deref(ctx).get_result(0);

    let load_op = Operation::new(
        ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![field_type],
        vec![field_ptr],
        vec![],
        0,
    );
    load_op.deref_mut(ctx).set_loc(loc);
    load_op.insert_after(ctx, addr_op);

    Ok((load_op.deref(ctx).get_result(0), Some(load_op)))
}

/// Apply a Field projection: extract field from struct/tuple.
fn apply_field_projection(
    ctx: &mut Context,
    aggregate_value: Value,
    field_idx: mir::FieldIdx,
    field_ty: &rustc_public::ty::Ty,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let field_type = types::translate_type(ctx, field_ty)?;

    let op = Operation::new(
        ctx,
        MirExtractFieldOp::get_concrete_op_info(),
        vec![field_type],
        vec![aggregate_value],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc);

    let extract_op = MirExtractFieldOp::new(op);
    extract_op.set_attr_index(
        ctx,
        dialect_mir::attributes::FieldIndexAttr(field_idx as u32),
    );

    if let Some(prev) = prev_op {
        extract_op.get_operation().insert_after(ctx, prev);
    } else {
        extract_op.get_operation().insert_at_front(block_ptr, ctx);
    }

    let field_value = extract_op.get_operation().deref(ctx).get_result(0);

    Ok((field_value, Some(extract_op.get_operation())))
}

/// Apply a Field projection on an enum variant (after Downcast).
fn apply_enum_field_projection(
    ctx: &mut Context,
    enum_value: Value,
    enum_rust_ty: &rustc_public::ty::Ty,
    variant_idx: rustc_public::ty::VariantIdx,
    field_idx: mir::FieldIdx,
    field_ty: &rustc_public::ty::Ty,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    use dialect_mir::ops::MirEnumPayloadOp;

    let field_type = types::translate_type(ctx, field_ty)?;

    let op = Operation::new(
        ctx,
        MirEnumPayloadOp::get_concrete_op_info(),
        vec![field_type],
        vec![enum_value],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    let payload_op = MirEnumPayloadOp::new(op);

    // Get the variant index
    // NOTE: variant_idx IS the index (0, 1, 2, ...), NOT the discriminant!
    // We just need to validate it's an ADT type, then use the index directly.
    let variant_idx_val: usize = match enum_rust_ty.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Adt(_adt_def, _)) => {
            variant_idx.to_index()
        }
        _ => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "Downcast on non-ADT type: {:?}",
                    enum_rust_ty
                ))
            );
        }
    };

    payload_op.set_attr_payload_variant_index(
        ctx,
        dialect_mir::attributes::VariantIndexAttr(variant_idx_val as u32),
    );
    payload_op.set_attr_payload_field_index(
        ctx,
        dialect_mir::attributes::FieldIndexAttr(field_idx as u32),
    );

    if let Some(prev) = prev_op {
        payload_op.get_operation().insert_after(ctx, prev);
    } else {
        payload_op.get_operation().insert_at_front(block_ptr, ctx);
    }

    let payload_value = payload_op.get_operation().deref(ctx).get_result(0);

    Ok((payload_value, Some(payload_op.get_operation())))
}

/// Compute the in-memory address of `place` by walking its FULL projection
/// list starting from `place.local`'s alloca slot.
///
/// Single entry point for `Rvalue::Ref` / `Rvalue::AddressOf` address
/// materialisation: `&(*ptr)` loads the pointer, `&(*ptr).field` adds a
/// field address, `&x.arr[i]` adds an element address, and arbitrary
/// combinations compose.
///
/// Returns `Ok(None)` when the local has no slot (ZST / ghost locals) or
/// when the projection chain contains an element
/// [`translate_place_addr_from_slot`] cannot lower. The caller decides
/// whether a value-copy fallback is sound (shared borrows: reads through a
/// copy are fine) or the construct must be rejected (mutable borrows / raw
/// mut pointers: writes through a copy are silently lost).
///
/// Also used by statement translation to compute the destination address
/// of projected assignments (indexed `(*ptr)[i]` writes and 3+ element
/// projection chains), where the same "walk the chain, then act through
/// the address" logic applies with a store instead of a borrow.
#[allow(clippy::too_many_arguments)]
pub(crate) fn translate_place_address(
    ctx: &mut Context,
    body: &mir::Body,
    value_map: &ValueMap,
    place: &mir::Place,
    is_mutable: bool,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<Option<(Value, Option<Ptr<Operation>>)>> {
    let Some(slot) = value_map.get_slot(place.local) else {
        return Ok(None);
    };
    translate_place_addr_from_slot(
        ctx,
        body,
        value_map,
        slot,
        &place.projection,
        is_mutable,
        block_ptr,
        prev_op,
        loc,
    )
}

/// Compute the in-memory address of `place` starting from its alloca `slot`.
///
/// Walks the projection chain and emits the correct pliron ops for each
/// element:
///
/// - `Field(idx, _)`   → [`MirFieldAddrOp`]
/// - `ConstantIndex {offset, from_end: false, ..}` → `MirConstantOp` + [`MirArrayElementAddrOp`]
///   (array pointee) or `MirConstantOp` + [`MirPtrOffsetOp`] (slice data pointer)
/// - `Index(local)`    → `load_local(local)` + [`MirArrayElementAddrOp`]
///   (array pointee) or `load_local(local)` + [`MirPtrOffsetOp`] (slice data pointer)
/// - `Deref`           → `MirLoadOp` of the pointer (the loaded pointer IS
///   the pointee's address); subsequent projections apply to the pointee.
///   ZST pointees skip the load (SharedArray exception). Fat (slice-shaped)
///   pointees scalarize to a (data ptr, len) pair: a mid-chain fat deref
///   loads the whole fat value and extracts the thin data pointer (field 0)
///   so the walk continues against the ORIGINAL elements, while a trailing
///   fat deref (`&*s` reborrow) is just a load of the fat value.
///
/// `Downcast` (enum payload addressing; issues #131/#146), `Subslice` and
/// from-end `ConstantIndex` are NOT handled; the walker punts on them
/// (returns `Ok(None)`).
///
/// Returns `Ok(Some((addr, last_op)))` on success, `Ok(None)` if the
/// projection chain contains an element this helper doesn't know how to
/// turn into an address (the caller decides whether a value fallback is
/// sound or the construct must be rejected), or `Err` if something
/// structurally invalid happens (wrong pointee kind, unsupported type).
///
/// `is_mutable` governs the mutability of intermediate pointer types; the
/// final result pointer also carries this mutability.
fn translate_place_addr_from_slot(
    ctx: &mut Context,
    body: &mir::Body,
    value_map: &ValueMap,
    slot: Value,
    projection: &[mir::ProjectionElem],
    is_mutable: bool,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<Option<(Value, Option<Ptr<Operation>>)>> {
    use dialect_mir::ops::{MirConstantOp, MirFieldAddrOp};

    let mut current = slot;
    let mut current_prev_op = prev_op;
    let mut current_is_slice_data = false;

    for (proj_idx, elem) in projection.iter().enumerate() {
        // The slice-data provenance bit only describes the pointer produced by
        // the immediately-preceding `Deref` of a fat slice (index it by
        // element, not as a pointer to one array object). Capture it for this
        // iteration and clear it up front, so the invariant stays local: it is
        // true only when the previous step handed us a slice DATA pointer, and
        // no later projection arm can accidentally leak it forward.
        let entered_as_slice_data = current_is_slice_data;
        current_is_slice_data = false;

        match elem {
            // `*place` -- the place walked so far holds a pointer; the
            // address of the dereferenced place is that pointer VALUE, so a
            // single `mir.load` of `current` yields it. Subsequent
            // projections then apply to the pointee.
            mir::ProjectionElem::Deref => {
                // Type of the place being dereferenced (= pointee of the
                // `current` address).
                let place_ty = {
                    let ty = current.get_type(ctx);
                    let ty_ref = ty.deref(ctx);
                    match ty_ref.downcast_ref::<dialect_mir::types::MirPtrType>() {
                        Some(pt) => pt.pointee,
                        // `current` is not a pointer-typed address; punt to
                        // the caller.
                        None => return Ok(None),
                    }
                };
                let (pointee_is_zst_tuple, pointee_is_thin_ptr, fat_elem_ty) = {
                    let p_ref = place_ty.deref(ctx);
                    let is_zst_tuple = p_ref
                        .downcast_ref::<dialect_mir::types::MirTupleType>()
                        .is_some_and(|tt| tt.get_types().is_empty());
                    let is_thin_ptr = p_ref.is::<dialect_mir::types::MirPtrType>();
                    // Slice-shaped (fat) pointees carry their element type.
                    let fat_elem_ty = p_ref
                        .downcast_ref::<dialect_mir::types::MirSliceType>()
                        .map(|st| st.element_type())
                        .or_else(|| {
                            p_ref
                                .downcast_ref::<dialect_mir::types::MirDisjointSliceType>()
                                .map(|st| st.element_type())
                        });
                    (is_zst_tuple, is_thin_ptr, fat_elem_ty)
                };

                if pointee_is_zst_tuple {
                    // ZST-pointee no-load exception (mirrors the Deref
                    // handling in `translate_place`, where it covers
                    // SharedArray): a pointer to a ZST *is* the runtime
                    // representation of the ZST place, so the deref adds no
                    // indirection. Keep `current` unchanged instead of
                    // emitting a meaningless load.
                    continue;
                }

                let is_last = proj_idx + 1 == projection.len();
                if let Some(elem_ty) = fat_elem_ty {
                    // Fat values (`&[T]`, `DisjointSlice<T>`, fat references
                    // to slice-tailed structs) are a (data pointer, element
                    // count) pair; dereferencing THROUGH them with a single
                    // `mir.load` would treat the pair as a thin address, a
                    // silent miscompile, so we never do that. What we CAN do:
                    //
                    // - Trailing `&*s` reborrow (the deref is the last
                    //   projection): the borrow result IS the fat value,
                    //   which lives whole in the slot, so the plain load
                    //   below is exactly right.
                    //
                    // - When the next projection is one we understand,
                    //   continue the walk by hand: load the fat PAIR,
                    //   extract its data pointer (field 0), and process the
                    //   following projection against that pointer. The data
                    //   pointer addresses the ORIGINAL elements, so both
                    //   shared and mutable borrows stay sound. This covers
                    //   field access through a fat reference to a
                    //   slice-tailed struct (the `(*iter).alive.start`
                    //   accesses inside `core::array::IntoIter::next`,
                    //   issue #138) and element access through a slice
                    //   reference (`(*slice)[i]`, including the inlined
                    //   body of `slice::get_mut`, issue #58).
                    //
                    // - Anything else keeps the loud failure (mutable) or
                    //   the value-copy fallback (shared).
                    if is_last {
                        // Fall through to the load below.
                    } else {
                        // Load the fat (ptr, len) pair from the slot.
                        let fat_load = Operation::new(
                            ctx,
                            MirLoadOp::get_concrete_op_info(),
                            vec![place_ty],
                            vec![current],
                            vec![],
                            0,
                        );
                        fat_load.deref_mut(ctx).set_loc(loc.clone());
                        match current_prev_op {
                            Some(p) => fat_load.insert_after(ctx, p),
                            None => fat_load.insert_at_front(block_ptr, ctx),
                        }
                        let fat_val = fat_load.deref(ctx).get_result(0);

                        // Extract the data pointer (field 0 of the pair).
                        // Its pointee is the slice's element type: the
                        // struct itself for a fat struct reference, or the
                        // element for an ordinary `&[T]` / `DisjointSlice`.
                        let data_ptr_ty: Ptr<TypeObj> =
                            dialect_mir::types::MirPtrType::get_generic(ctx, elem_ty, is_mutable)
                                .into();
                        let extract_ptr = Operation::new(
                            ctx,
                            MirExtractFieldOp::get_concrete_op_info(),
                            vec![data_ptr_ty],
                            vec![fat_val],
                            vec![],
                            0,
                        );
                        extract_ptr.deref_mut(ctx).set_loc(loc.clone());
                        MirExtractFieldOp::new(extract_ptr)
                            .set_attr_index(ctx, dialect_mir::attributes::FieldIndexAttr(0));
                        extract_ptr.insert_after(ctx, fat_load);
                        let data_ptr = extract_ptr.deref(ctx).get_result(0);
                        current_prev_op = Some(extract_ptr);

                        // Borrow of the struct's unsized slice tail, e.g.
                        // `&(*iter).data`. No thin pointer can represent
                        // that place: the result must itself be a fat
                        // (tail pointer, len) pair, with the len carried
                        // over from the fat reference we walked through.
                        // Only valid as the FINAL projection.
                        if let mir::ProjectionElem::Field(field_idx, field_rust_ty) =
                            &projection[proj_idx + 1]
                            && let rustc_public::ty::TyKind::RigidTy(
                                rustc_public::ty::RigidTy::Slice(tail_elem_rust_ty),
                            ) = field_rust_ty.kind()
                        {
                            if proj_idx + 2 != projection.len() {
                                // Projections continuing past an unsized
                                // tail borrow are not a shape rustc emits;
                                // punt rather than guess.
                                return Ok(None);
                            }
                            let tail_elem_ty = types::translate_type(ctx, &tail_elem_rust_ty)?;

                            // Address of the first tail element. The struct
                            // model stores the tail field with the ELEMENT
                            // type (see `translate_type`'s ADT arm), so the
                            // field-addr result is a pointer to the element
                            // and the dialect verifier agrees.
                            let tail_ptr_ty: Ptr<TypeObj> =
                                dialect_mir::types::MirPtrType::get_generic(
                                    ctx,
                                    tail_elem_ty,
                                    is_mutable,
                                )
                                .into();
                            let tail_addr = Operation::new(
                                ctx,
                                MirFieldAddrOp::get_concrete_op_info(),
                                vec![tail_ptr_ty],
                                vec![data_ptr],
                                vec![],
                                0,
                            );
                            tail_addr.deref_mut(ctx).set_loc(loc.clone());
                            MirFieldAddrOp::new(tail_addr).set_attr_field_index(
                                ctx,
                                dialect_mir::attributes::FieldIndexAttr(*field_idx as u32),
                            );
                            tail_addr.insert_after(ctx, extract_ptr);
                            let tail_ptr = tail_addr.deref(ctx).get_result(0);

                            // The element count (field 1 of the fat pair).
                            let usize_ty = types::get_usize_type(ctx);
                            let extract_len = Operation::new(
                                ctx,
                                MirExtractFieldOp::get_concrete_op_info(),
                                vec![usize_ty.to_ptr()],
                                vec![fat_val],
                                vec![],
                                0,
                            );
                            extract_len.deref_mut(ctx).set_loc(loc.clone());
                            MirExtractFieldOp::new(extract_len)
                                .set_attr_index(ctx, dialect_mir::attributes::FieldIndexAttr(1));
                            extract_len.insert_after(ctx, tail_addr);
                            let len_val = extract_len.deref(ctx).get_result(0);

                            let slice_ty = dialect_mir::types::MirSliceType::get(ctx, tail_elem_ty);
                            use dialect_mir::ops::MirConstructSliceOp;
                            let construct = Operation::new(
                                ctx,
                                MirConstructSliceOp::get_concrete_op_info(),
                                vec![slice_ty.into()],
                                vec![tail_ptr, len_val],
                                vec![],
                                0,
                            );
                            construct.deref_mut(ctx).set_loc(loc.clone());
                            construct.insert_after(ctx, extract_len);
                            return Ok(Some((construct.deref(ctx).get_result(0), Some(construct))));
                        }

                        match &projection[proj_idx + 1] {
                            // Sized field access: hand the data pointer to
                            // the field arm below.
                            mir::ProjectionElem::Field(..) => {
                                current = data_ptr;
                                continue;
                            }
                            // Element access through a slice data pointer is
                            // pointer arithmetic over the slice element type.
                            // That remains true when the element type is
                            // itself an array (`&[[T; N]][i]`), where a
                            // type-only check would otherwise mistake the
                            // data pointer for a pointer to one array object
                            // and index inside row 0.
                            mir::ProjectionElem::Index(_)
                            | mir::ProjectionElem::ConstantIndex {
                                from_end: false, ..
                            } => {
                                current = data_ptr;
                                current_is_slice_data = true;
                                continue;
                            }
                            // Unknown continuation: keep the conservative
                            // behaviour (loud failure for mutable borrows,
                            // value-copy fallback for shared ones).
                            _ => {
                                if is_mutable {
                                    return input_err!(
                                        loc,
                                        TranslationErr::unsupported(format!(
                                            "cannot compute a mutable in-memory address through \
                                             fat-pointer deref (projection {:?})",
                                            projection
                                        ))
                                    );
                                }
                                return Ok(None);
                            }
                        }
                    }
                } else if !pointee_is_thin_ptr {
                    // Deref of a non-pointer-typed place (a type the
                    // importer models by value); punt to the caller.
                    return Ok(None);
                }

                let load_op = Operation::new(
                    ctx,
                    MirLoadOp::get_concrete_op_info(),
                    vec![place_ty],
                    vec![current],
                    vec![],
                    0,
                );
                load_op.deref_mut(ctx).set_loc(loc.clone());
                match current_prev_op {
                    Some(p) => load_op.insert_after(ctx, p),
                    None => load_op.insert_at_front(block_ptr, ctx),
                }
                current = load_op.deref(ctx).get_result(0);
                current_prev_op = Some(load_op);
            }

            mir::ProjectionElem::Field(field_idx, field_ty) => {
                let field_type = types::translate_type(ctx, field_ty)?;
                let result_ptr_ty =
                    dialect_mir::types::MirPtrType::get_generic(ctx, field_type, is_mutable);
                let op = Operation::new(
                    ctx,
                    MirFieldAddrOp::get_concrete_op_info(),
                    vec![result_ptr_ty.into()],
                    vec![current],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc.clone());
                MirFieldAddrOp::new(op).set_attr_field_index(
                    ctx,
                    dialect_mir::attributes::FieldIndexAttr(*field_idx as u32),
                );
                match current_prev_op {
                    Some(p) => op.insert_after(ctx, p),
                    None => op.insert_at_front(block_ptr, ctx),
                }
                current = op.deref(ctx).get_result(0);
                current_prev_op = Some(op);
            }

            mir::ProjectionElem::ConstantIndex {
                offset,
                min_length: _,
                from_end,
            } => {
                if *from_end {
                    return Ok(None);
                }
                let (mut pointee_kind, addr_space) = match pointer_pointee_kind(ctx, current) {
                    Some(kind) => kind,
                    None => return Ok(None),
                };
                if entered_as_slice_data {
                    pointee_kind = PointeeKind::Direct;
                }

                let i64_ty = IntegerType::get(ctx, 64, Signedness::Signed);
                let index_apint = APInt::from_i64(*offset as i64, NonZeroUsize::new(64).unwrap());
                let const_attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, index_apint);
                let const_op_ptr = Operation::new(
                    ctx,
                    MirConstantOp::get_concrete_op_info(),
                    vec![i64_ty.into()],
                    vec![],
                    vec![],
                    0,
                );
                const_op_ptr.deref_mut(ctx).set_loc(loc.clone());
                MirConstantOp::new(const_op_ptr).set_attr_value(ctx, const_attr);
                match current_prev_op {
                    Some(p) => const_op_ptr.insert_after(ctx, p),
                    None => const_op_ptr.insert_at_front(block_ptr, ctx),
                }
                current_prev_op = Some(const_op_ptr);
                let index_val = const_op_ptr.deref(ctx).get_result(0);

                let (addr_op, next_current) = emit_indexed_element_addr(
                    ctx,
                    current,
                    index_val,
                    pointee_kind,
                    addr_space,
                    is_mutable,
                    block_ptr,
                    current_prev_op,
                    loc.clone(),
                );
                current = next_current;
                current_prev_op = Some(addr_op);
            }

            // Runtime `arr[i]` indexing. Without this arm, a place like
            // `&(*ptr).field[i]` would silently drop the `Index` projection
            // and return a pointer to the array's first slot, miscompiling
            // every load through the reference into a load of element 0.
            mir::ProjectionElem::Index(index_local) => {
                let (mut pointee_kind, addr_space) = match pointer_pointee_kind(ctx, current) {
                    Some(kind) => kind,
                    None => return Ok(None),
                };
                if entered_as_slice_data {
                    pointee_kind = PointeeKind::Direct;
                }

                let index_place = mir::Place {
                    local: *index_local,
                    projection: vec![],
                };
                let (index_val, next_prev_op) = translate_place(
                    ctx,
                    body,
                    &index_place,
                    value_map,
                    block_ptr,
                    current_prev_op,
                    loc.clone(),
                )?;
                current_prev_op = next_prev_op;

                let (addr_op, next_current) = emit_indexed_element_addr(
                    ctx,
                    current,
                    index_val,
                    pointee_kind,
                    addr_space,
                    is_mutable,
                    block_ptr,
                    current_prev_op,
                    loc.clone(),
                );
                current = next_current;
                current_prev_op = Some(addr_op);
            }

            // Enum-variant downcast (`(x as Variant).field`). Addressing an
            // enum payload in memory needs variant/niche layout machinery
            // (per-variant payload offsets, tag placement) that the importer
            // currently models only in VALUE space via
            // `MirExtractEnumPayloadOp`. This arm is the designed extension
            // point for the enum-layout work tracked in issues #131/#146;
            // until that lands, punt so shared borrows can fall back to a
            // value copy and mutable borrows fail loudly at the caller.
            mir::ProjectionElem::Downcast(_) => return Ok(None),

            // Remaining projection kinds (Subslice, from-end ConstantIndex,
            // ...) aren't lowered to addresses here yet. Punt to the caller,
            // which decides between a value fallback (shared borrows) and a
            // hard error (mutable borrows).
            _ => return Ok(None),
        }
    }

    Ok(Some((current, current_prev_op)))
}

/// Describes what a pointer points to (array vs. anything else) for
/// address-computation dispatch.
enum PointeeKind {
    /// Pointee is `[T; N]` (carries `T`). Element addressing GEPs through
    /// the array type via `mir.array_element_addr`.
    Array(Ptr<TypeObj>),
    /// Pointee is any other type. When an `Index` / `ConstantIndex`
    /// projection meets such a pointer, MIR typing guarantees the indexed
    /// place is a slice whose data pointer (produced by the fat-pointer
    /// `Deref` arm) points directly at the elements, so element addressing
    /// is a plain `mir.ptr_offset` keeping the pointer's own type.
    Direct,
}

/// Emit the address of element `index_val` behind `current`, which is either
/// a pointer to a whole array (`&arr[i]`: `mir.array_element_addr`) or a
/// pointer to a single ELEMENT, i.e. the data pointer of a fat slice value
/// extracted by the Deref arm above (`(*slice)[i]`: element-size pointer
/// arithmetic via `mir.ptr_offset`). Returns the emitted op and the element
/// address it produces.
#[allow(clippy::too_many_arguments)]
fn emit_indexed_element_addr(
    ctx: &mut Context,
    current: Value,
    index_val: Value,
    pointee_kind: PointeeKind,
    addr_space: u32,
    is_mutable: bool,
    block_ptr: Ptr<BasicBlock>,
    current_prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> (Ptr<Operation>, Value) {
    use dialect_mir::ops::MirArrayElementAddrOp;

    let addr_op = match pointee_kind {
        PointeeKind::Array(element_ty) => {
            let elem_ptr_ty =
                dialect_mir::types::MirPtrType::get(ctx, element_ty, is_mutable, addr_space).into();
            Operation::new(
                ctx,
                MirArrayElementAddrOp::get_concrete_op_info(),
                vec![elem_ptr_ty],
                vec![current, index_val],
                vec![],
                0,
            )
        }
        PointeeKind::Direct => {
            // The pointee IS the element type, so indexing is plain
            // element-size pointer arithmetic and the result keeps the
            // pointer's own type.
            let ptr_ty = current.get_type(ctx);
            Operation::new(
                ctx,
                MirPtrOffsetOp::get_concrete_op_info(),
                vec![ptr_ty],
                vec![current, index_val],
                vec![],
                0,
            )
        }
    };
    addr_op.deref_mut(ctx).set_loc(loc);
    match current_prev_op {
        Some(p) => addr_op.insert_after(ctx, p),
        None => addr_op.insert_at_front(block_ptr, ctx),
    }
    let result = addr_op.deref(ctx).get_result(0);
    (addr_op, result)
}

/// Inspect a pointer value and return its pointee kind + address space, or
/// `None` if the value's type isn't a `MirPtrType`.
fn pointer_pointee_kind(ctx: &Context, ptr_value: Value) -> Option<(PointeeKind, u32)> {
    let ty = ptr_value.get_type(ctx);
    let ty_ref = ty.deref(ctx);
    let mir_ptr_ty = ty_ref.downcast_ref::<dialect_mir::types::MirPtrType>()?;
    let pointee = mir_ptr_ty.pointee;
    let addr_space = mir_ptr_ty.address_space;
    let pointee_ref = pointee.deref(ctx);
    let kind = if let Some(arr_ty) = pointee_ref.downcast_ref::<dialect_mir::types::MirArrayType>()
    {
        PointeeKind::Array(arr_ty.element_type())
    } else {
        PointeeKind::Direct
    };
    Some((kind, addr_space))
}

/// Translate a MIR Place using iterative projection processing.
/// This handles arbitrary depth projections by processing each element in sequence.
pub fn translate_place_iterative(
    ctx: &mut Context,
    body: &mir::Body,
    place: &mir::Place,
    value_map: &ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    // Start with the base local's current value. In the alloca model every
    // non-ZST local has a stack slot, so we emit `mir.load` once here and
    // then layer projections on top of the loaded SSA value; `mem2reg` folds
    // the load back into a direct SSA use when the slot is promotable. ZST /
    // unsupported locals fall back to the same ghost-enum / empty-aggregate
    // synthesis as [`translate_place`].
    let local = place.local;
    let (mut current_value, mut current_prev_op) =
        match value_map.load_local(ctx, local, block_ptr, prev_op) {
            Some((load_op, val)) => (val, Some(load_op)),
            None => {
                let local_decl = &body.locals()[local];
                let ty_ptr = types::translate_type(ctx, &local_decl.ty)?;
                let synth_op = if ty_ptr.deref(ctx).is::<dialect_mir::types::MirEnumType>() {
                    create_ghost_enum_default(ctx, ty_ptr, loc.clone())
                } else if types::is_zst_type(ctx, ty_ptr) {
                    create_zst_aggregate(ctx, ty_ptr, loc.clone())
                } else {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "Local {} has no alloca slot and is not a ZST",
                            Into::<usize>::into(local)
                        ))
                    );
                };
                match prev_op {
                    Some(p) => synth_op.insert_after(ctx, p),
                    None => synth_op.insert_at_front(block_ptr, ctx),
                }
                let val = synth_op.deref(ctx).get_result(0);
                (val, Some(synth_op))
            }
        };

    // Track the Rust type of `current_value` alongside the pliron value.
    // Each iteration below advances it through rustc_public's own projection
    // typing (`ProjectionElem::ty`) AFTER the arm has processed the element,
    // so every arm observes the type *before* its own projection applies and
    // the next iteration sees the narrowed type. `Downcast` deliberately
    // leaves the type unchanged (still the enum ADT), which is exactly what
    // `apply_enum_field_projection` expects when the following `Field` fires.
    //
    // This single fold is the only place `current_rust_ty` is updated;
    // individual arms must not update it themselves. Per-arm updates were
    // the cause of issue #131: only `Field` advanced the type, so chains
    // like `[Index, Downcast, Field]` (from `match xs[i]` over an array of
    // enums) handed the stale Array type to the Downcast/Field handler,
    // which bailed with "Downcast on non-ADT type: Array". The same
    // staleness affected `Deref` and `ConstantIndex`.
    let mut current_rust_ty = body.locals()[local].ty;

    // Track pending downcast (Downcast is a no-op, but we need variant info for Field on enums)
    // Type inferred from ProjectionElem::Downcast pattern
    let mut pending_downcast = None;

    // Process each projection element iteratively
    for projection in &place.projection {
        match projection {
            ProjectionElem::Deref => {
                (current_value, current_prev_op) = apply_deref_projection(
                    ctx,
                    current_value,
                    block_ptr,
                    current_prev_op,
                    loc.clone(),
                )?;
                pending_downcast = None;
            }

            ProjectionElem::Field(field_idx, field_ty) => {
                // Check if this is a field access on an enum (preceded by Downcast)
                if let Some(variant_idx) = pending_downcast.take() {
                    // Enum variant field access - use MirEnumPayloadOp
                    (current_value, current_prev_op) = apply_enum_field_projection(
                        ctx,
                        current_value,
                        &current_rust_ty,
                        variant_idx,
                        *field_idx,
                        field_ty,
                        block_ptr,
                        current_prev_op,
                        loc.clone(),
                    )?;
                } else {
                    let current_is_ptr = current_value
                        .get_type(ctx)
                        .deref(ctx)
                        .is::<dialect_mir::types::MirPtrType>();
                    if current_is_ptr {
                        // `current_value` is an ADDRESS, not an aggregate
                        // value. This happens after dereferencing a fat
                        // pointer: `apply_deref_projection` cannot load an
                        // unsized pointee, so it hands back the data
                        // pointer instead (e.g. reading
                        // `(*iter).alive.start` through the fat
                        // `&mut PolymorphicIter<[MaybeUninit<T>]>` inside
                        // `core::array::IntoIter::next`, issue #138).
                        // Compute the field's address and load the field.
                        (current_value, current_prev_op) = apply_field_addr_and_load(
                            ctx,
                            current_value,
                            *field_idx,
                            field_ty,
                            block_ptr,
                            current_prev_op,
                            loc.clone(),
                        )?;
                    } else {
                        // Regular struct/tuple field access
                        (current_value, current_prev_op) = apply_field_projection(
                            ctx,
                            current_value,
                            *field_idx,
                            field_ty,
                            block_ptr,
                            current_prev_op,
                            loc.clone(),
                        )?;
                    }
                }
            }

            ProjectionElem::Downcast(variant_idx) => {
                // Downcast is a no-op - it just narrows the type for the next Field access
                // Store the variant index for use by the next Field projection
                pending_downcast = Some(*variant_idx);
                // Don't change current_value
            }

            ProjectionElem::Index(index_local) => {
                let index_place = mir::Place {
                    local: *index_local,
                    projection: vec![],
                };
                let (index_value, next_prev_op) = translate_place(
                    ctx,
                    body,
                    &index_place,
                    value_map,
                    block_ptr,
                    current_prev_op,
                    loc.clone(),
                )?;
                current_prev_op = next_prev_op;

                // Determine indexable kind upfront so we drop the immutable borrow
                // before creating operations (which need &mut ctx).
                enum IndexableKind {
                    Array {
                        element_ty: Ptr<TypeObj>,
                    },
                    Ptr {
                        element_ty: Ptr<TypeObj>,
                        ptr_ty: Ptr<TypeObj>,
                    },
                }

                let cur_ty = current_value.get_type(ctx);
                let kind = {
                    let cur_ty_ref = cur_ty.deref(ctx);
                    if let Some(arr_ty) =
                        cur_ty_ref.downcast_ref::<dialect_mir::types::MirArrayType>()
                    {
                        Ok(IndexableKind::Array {
                            element_ty: arr_ty.element_type(),
                        })
                    } else if let Some(ptr_ty) =
                        cur_ty_ref.downcast_ref::<dialect_mir::types::MirPtrType>()
                    {
                        Ok(IndexableKind::Ptr {
                            element_ty: ptr_ty.pointee,
                            ptr_ty: cur_ty,
                        })
                    } else {
                        let ty_dbg = format!("{:?}", cur_ty_ref);
                        Err(ty_dbg)
                    }
                };

                match kind {
                    Ok(IndexableKind::Array { element_ty }) => {
                        use dialect_mir::ops::MirExtractArrayElementOp;
                        let op = Operation::new(
                            ctx,
                            MirExtractArrayElementOp::get_concrete_op_info(),
                            vec![element_ty],
                            vec![current_value, index_value],
                            vec![],
                            0,
                        );
                        op.deref_mut(ctx).set_loc(loc.clone());

                        if let Some(prev) = current_prev_op {
                            op.insert_after(ctx, prev);
                        } else {
                            op.insert_at_front(block_ptr, ctx);
                        }

                        current_value = op.deref(ctx).get_result(0);
                        current_prev_op = Some(op);
                    }
                    Ok(IndexableKind::Ptr { element_ty, ptr_ty }) => {
                        let offset_op = Operation::new(
                            ctx,
                            MirPtrOffsetOp::get_concrete_op_info(),
                            vec![ptr_ty],
                            vec![current_value, index_value],
                            vec![],
                            0,
                        );
                        offset_op.deref_mut(ctx).set_loc(loc.clone());
                        if let Some(prev) = current_prev_op {
                            offset_op.insert_after(ctx, prev);
                        } else {
                            offset_op.insert_at_front(block_ptr, ctx);
                        }
                        current_prev_op = Some(offset_op);
                        let offset_ptr = offset_op.deref(ctx).get_result(0);

                        let load_op = Operation::new(
                            ctx,
                            MirLoadOp::get_concrete_op_info(),
                            vec![element_ty],
                            vec![offset_ptr],
                            vec![],
                            0,
                        );
                        load_op.deref_mut(ctx).set_loc(loc.clone());
                        let load = MirLoadOp::new(load_op);
                        if let Some(prev) = current_prev_op {
                            load.get_operation().insert_after(ctx, prev);
                        } else {
                            load.get_operation().insert_at_front(block_ptr, ctx);
                        }

                        current_value = load.get_operation().deref(ctx).get_result(0);
                        current_prev_op = Some(load.get_operation());
                    }
                    Err(ty_dbg) => {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(format!(
                                "Index projection on unsupported type.\n\
                                 \n  pliron type: {}\n\
                                 \n  display    : {}\n\
                                 \n\
                                 \nIndex handles MirArrayType (extract_array_element) and MirPtrType\n\
                                 (pointer offset + load, e.g. after Deref on a slice). The type above\n\
                                 matched neither. A new handler may need to be added.",
                                ty_dbg,
                                cur_ty.disp(ctx)
                            ))
                        );
                    }
                }
                pending_downcast = None;
            }

            ProjectionElem::ConstantIndex {
                offset,
                min_length: _,
                from_end,
            } => {
                if *from_end {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(
                            "ConstantIndex with from_end=true not yet supported"
                        )
                    );
                }
                let index = *offset as usize;

                // Determine indexable kind upfront so we drop the immutable borrow
                // before creating operations (which need &mut ctx).
                enum ConstIndexKind {
                    Array {
                        element_ty: Ptr<TypeObj>,
                    },
                    Ptr {
                        element_ty: Ptr<TypeObj>,
                        ptr_ty: Ptr<TypeObj>,
                    },
                }

                let cur_ty = current_value.get_type(ctx);
                let kind = {
                    let cur_ty_ref = cur_ty.deref(ctx);
                    if let Some(arr_ty) =
                        cur_ty_ref.downcast_ref::<dialect_mir::types::MirArrayType>()
                    {
                        Ok(ConstIndexKind::Array {
                            element_ty: arr_ty.element_type(),
                        })
                    } else if let Some(ptr_ty) =
                        cur_ty_ref.downcast_ref::<dialect_mir::types::MirPtrType>()
                    {
                        Ok(ConstIndexKind::Ptr {
                            element_ty: ptr_ty.pointee,
                            ptr_ty: cur_ty,
                        })
                    } else {
                        let ty_dbg = format!("{:?}", cur_ty_ref);
                        Err(ty_dbg)
                    }
                };

                match kind {
                    Ok(ConstIndexKind::Array { element_ty }) => {
                        let op = Operation::new(
                            ctx,
                            MirExtractFieldOp::get_concrete_op_info(),
                            vec![element_ty],
                            vec![current_value],
                            vec![],
                            0,
                        );
                        op.deref_mut(ctx).set_loc(loc.clone());
                        let extract_op = MirExtractFieldOp::new(op);
                        extract_op.set_attr_index(
                            ctx,
                            dialect_mir::attributes::FieldIndexAttr(index as u32),
                        );

                        if let Some(prev) = current_prev_op {
                            extract_op.get_operation().insert_after(ctx, prev);
                        } else {
                            extract_op.get_operation().insert_at_front(block_ptr, ctx);
                        }

                        current_value = extract_op.get_operation().deref(ctx).get_result(0);
                        current_prev_op = Some(extract_op.get_operation());
                    }
                    Ok(ConstIndexKind::Ptr { element_ty, ptr_ty }) => {
                        // Create constant index value
                        let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
                        let apint = APInt::from_u32(index as u32, NonZeroUsize::new(32).unwrap());
                        let index_attr =
                            pliron::builtin::attributes::IntegerAttr::new(i32_ty, apint);
                        use dialect_mir::ops::MirConstantOp;
                        let const_op = Operation::new(
                            ctx,
                            MirConstantOp::get_concrete_op_info(),
                            vec![i32_ty.into()],
                            vec![],
                            vec![],
                            0,
                        );
                        const_op.deref_mut(ctx).set_loc(loc.clone());
                        let const_mir = MirConstantOp::new(const_op);
                        const_mir.set_attr_value(ctx, index_attr);
                        if let Some(prev) = current_prev_op {
                            const_mir.get_operation().insert_after(ctx, prev);
                        } else {
                            const_mir.get_operation().insert_at_front(block_ptr, ctx);
                        }
                        current_prev_op = Some(const_mir.get_operation());
                        let index_value = const_mir.get_operation().deref(ctx).get_result(0);

                        // Pointer offset
                        let offset_op = Operation::new(
                            ctx,
                            MirPtrOffsetOp::get_concrete_op_info(),
                            vec![ptr_ty],
                            vec![current_value, index_value],
                            vec![],
                            0,
                        );
                        offset_op.deref_mut(ctx).set_loc(loc.clone());
                        if let Some(prev) = current_prev_op {
                            offset_op.insert_after(ctx, prev);
                        } else {
                            offset_op.insert_at_front(block_ptr, ctx);
                        }
                        current_prev_op = Some(offset_op);
                        let offset_ptr = offset_op.deref(ctx).get_result(0);

                        // Load element
                        let load_op = Operation::new(
                            ctx,
                            MirLoadOp::get_concrete_op_info(),
                            vec![element_ty],
                            vec![offset_ptr],
                            vec![],
                            0,
                        );
                        load_op.deref_mut(ctx).set_loc(loc.clone());
                        let load = MirLoadOp::new(load_op);
                        if let Some(prev) = current_prev_op {
                            load.get_operation().insert_after(ctx, prev);
                        } else {
                            load.get_operation().insert_at_front(block_ptr, ctx);
                        }

                        current_value = load.get_operation().deref(ctx).get_result(0);
                        current_prev_op = Some(load.get_operation());
                    }
                    Err(ty_dbg) => {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(format!(
                                "ConstantIndex projection on unsupported type.\n\
                                 \n  pliron type: {}\n\
                                 \n  display    : {}\n\
                                 \n  index      : {}\n\
                                 \n\
                                 \nConstantIndex handles MirArrayType (extractvalue) and MirPtrType\n\
                                 (pointer offset + load, e.g. after Deref on a slice). The type above\n\
                                 matched neither. A new handler may need to be added.",
                                ty_dbg,
                                cur_ty.disp(ctx),
                                index
                            ))
                        );
                    }
                }
                pending_downcast = None;
            }

            // Unsupported projection types
            other => {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Projection element {:?} not yet implemented in iterative mode",
                        other
                    ))
                );
            }
        }

        // Advance the running Rust type with rustc_public's own projection
        // typing (single source of truth; see the comment on
        // `current_rust_ty` above). For well-formed MIR this never fails;
        // if it does, surface the projection element and the type it was
        // applied to so the bail-out is actionable.
        current_rust_ty = projection.ty(current_rust_ty).map_err(|e| {
            input_error!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "Failed to type projection {:?} applied to {:?}: {:?}",
                    projection, current_rust_ty, e
                ))
            )
        })?;
    }

    Ok((current_value, current_prev_op))
}

/// Translate a pointer-to-array constant to MIR operations.
///
/// Handles both byte string literals (`&[u8; N]`, e.g. `b"hello\0"`) and typed
/// array constants (`&[f64; 3]`, `&[u32; 4]`, etc.). The function:
/// 1. Extracts raw bytes from the constant's allocation
/// 2. Groups bytes into element-sized chunks based on the array element type
/// 3. Creates typed constants for each element (u8, u32, f32, f64, etc.)
/// 4. Returns a pointer to the constructed array
fn translate_ptr_to_array_constant(
    ctx: &mut Context,
    constant: &mir::ConstOperand,
    const_ty_ptr: Ptr<TypeObj>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    use pliron::builtin::types::{FP32Type, FP64Type, IntegerType};
    use rustc_public::ty::ConstantKind;

    // Extract array type and element type info from the pointer type
    let (array_ty, element_ty_ptr, element_count) = {
        let ty_obj = const_ty_ptr.deref(ctx);
        let ptr_ty = ty_obj
            .downcast_ref::<dialect_mir::types::MirPtrType>()
            .ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(
                    "translate_ptr_to_array_constant: expected pointer type"
                ))
            })?;

        let arr_ty_obj = ptr_ty.pointee.deref(ctx);
        let arr_ty = arr_ty_obj
            .downcast_ref::<dialect_mir::types::MirArrayType>()
            .ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(
                    "translate_ptr_to_array_constant: expected array pointee"
                ))
            })?;

        (ptr_ty.pointee, arr_ty.element_type(), arr_ty.size())
    };

    // Determine element size in bytes from the pliron element type
    let element_byte_size: usize = {
        let elem_obj = element_ty_ptr.deref(ctx);
        if let Some(int_ty) = elem_obj.downcast_ref::<IntegerType>() {
            (int_ty.width() as usize).div_ceil(8)
        } else if elem_obj.is::<MirFP16Type>() {
            2
        } else if elem_obj.is::<FP32Type>() {
            4
        } else if elem_obj.is::<FP64Type>() {
            8
        } else {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "translate_ptr_to_array_constant: unsupported element type: {:?}",
                    elem_obj
                ))
            );
        }
    };

    // Extract raw bytes from the constant's allocation.
    // For promoted constants, the allocation contains a pointer (with provenance)
    // to another allocation with the actual data.
    let bytes = match constant.const_.kind() {
        ConstantKind::Allocated(alloc) => {
            if let Some((_, prov)) = alloc.provenance.ptrs.first() {
                use rustc_public::mir::alloc::GlobalAlloc;
                let alloc_id = prov.0;
                match GlobalAlloc::from(alloc_id) {
                    GlobalAlloc::Memory(target_alloc) => {
                        target_alloc.raw_bytes().ok().unwrap_or_else(|| {
                            target_alloc
                                .bytes
                                .iter()
                                .map(|opt: &Option<u8>| opt.unwrap_or(0))
                                .collect::<Vec<u8>>()
                        })
                    }
                    GlobalAlloc::Static(static_def) => {
                        let target_alloc = static_def.eval_initializer().map_err(|e| {
                            input_error_noloc!(TranslationErr::unsupported(format!(
                                "Failed to evaluate static initializer for array constant: {:?}",
                                e
                            )))
                        })?;
                        target_alloc.raw_bytes().ok().unwrap_or_else(|| {
                            target_alloc
                                .bytes
                                .iter()
                                .map(|opt: &Option<u8>| opt.unwrap_or(0))
                                .collect::<Vec<u8>>()
                        })
                    }
                    other => {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(format!(
                                "Array constant provenance points to non-memory allocation: {:?}",
                                other
                            ))
                        );
                    }
                }
            } else {
                alloc.raw_bytes().ok().unwrap_or_else(|| {
                    alloc
                        .bytes
                        .iter()
                        .map(|opt: &Option<u8>| opt.unwrap_or(0))
                        .collect::<Vec<u8>>()
                })
            }
        }
        _ => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "Array constant must be Allocated, got: {:?}",
                    constant.const_.kind()
                ))
            );
        }
    };

    // Validate: bytes should be element_count * element_byte_size
    let expected_bytes = element_count as usize * element_byte_size;
    if bytes.len() < expected_bytes {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Array constant has {} bytes but expected {} ({} elements x {} bytes each)",
                bytes.len(),
                expected_bytes,
                element_count,
                element_byte_size
            ))
        );
    }

    // Determine element kind once (drops the borrow before we mutate ctx in the loop)
    #[derive(Clone, Copy)]
    enum ElemKind {
        F64,
        F32,
        F16,
        Int { width: u32, signedness: Signedness },
    }
    let elem_kind = {
        let elem_obj = element_ty_ptr.deref(ctx);
        if elem_obj.is::<FP64Type>() {
            ElemKind::F64
        } else if elem_obj.is::<FP32Type>() {
            ElemKind::F32
        } else if elem_obj.is::<MirFP16Type>() {
            ElemKind::F16
        } else if let Some(int_ty) = elem_obj.downcast_ref::<IntegerType>() {
            ElemKind::Int {
                width: int_ty.width(),
                signedness: int_ty.signedness(),
            }
        } else {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "translate_ptr_to_array_constant: unsupported element type: {:?}",
                    elem_obj
                ))
            );
        }
    };

    // Create typed element constants by grouping bytes
    let mut element_values = Vec::with_capacity(element_count as usize);
    let mut last_op = prev_op;

    for i in 0..element_count as usize {
        let chunk = &bytes[i * element_byte_size..(i + 1) * element_byte_size];

        let (elem_val, elem_last_op) = match elem_kind {
            ElemKind::F64 => {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(chunk);
                let float_val = f64::from_le_bytes(buf);
                let float_attr = pliron::builtin::attributes::FPDoubleAttr::from(float_val);

                use dialect_mir::ops::MirFloatConstantOp;
                let op = Operation::new(
                    ctx,
                    MirFloatConstantOp::get_concrete_op_info(),
                    vec![element_ty_ptr],
                    vec![],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc.clone());
                let float_op = MirFloatConstantOp::new(op);
                float_op.set_attr_float_value_f64(ctx, float_attr);

                if let Some(prev) = last_op {
                    float_op.get_operation().insert_after(ctx, prev);
                } else {
                    float_op.get_operation().insert_at_front(block_ptr, ctx);
                }
                (
                    float_op.get_operation().deref(ctx).get_result(0),
                    Some(float_op.get_operation()),
                )
            }
            ElemKind::F32 => {
                let mut buf = [0u8; 4];
                buf.copy_from_slice(chunk);
                let float_val = f32::from_le_bytes(buf);
                let float_attr = pliron::builtin::attributes::FPSingleAttr::from(float_val);

                use dialect_mir::ops::MirFloatConstantOp;
                let op = Operation::new(
                    ctx,
                    MirFloatConstantOp::get_concrete_op_info(),
                    vec![element_ty_ptr],
                    vec![],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc.clone());
                let float_op = MirFloatConstantOp::new(op);
                float_op.set_attr_float_value(ctx, float_attr);

                if let Some(prev) = last_op {
                    float_op.get_operation().insert_after(ctx, prev);
                } else {
                    float_op.get_operation().insert_at_front(block_ptr, ctx);
                }
                (
                    float_op.get_operation().deref(ctx).get_result(0),
                    Some(float_op.get_operation()),
                )
            }
            ElemKind::F16 => {
                let bits = read_uint_from_bytes(chunk) as u16;
                let float_attr = MirFP16Attr::from_bits(bits);

                use dialect_mir::ops::MirFloatConstantOp;
                let op = Operation::new(
                    ctx,
                    MirFloatConstantOp::get_concrete_op_info(),
                    vec![element_ty_ptr],
                    vec![],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc.clone());
                let float_op = MirFloatConstantOp::new(op);
                float_op.set_attr_float_value_f16(ctx, float_attr);

                if let Some(prev) = last_op {
                    float_op.get_operation().insert_after(ctx, prev);
                } else {
                    float_op.get_operation().insert_at_front(block_ptr, ctx);
                }
                (
                    float_op.get_operation().deref(ctx).get_result(0),
                    Some(float_op.get_operation()),
                )
            }
            ElemKind::Int { width, signedness } => {
                let val = read_uint_from_bytes(chunk);
                let apint = APInt::from_u128(val, NonZeroUsize::new(width as usize).unwrap());
                let int_attr = pliron::builtin::attributes::IntegerAttr::new(
                    IntegerType::get(ctx, width, signedness),
                    apint,
                );

                use dialect_mir::ops::MirConstantOp;
                let op = Operation::new(
                    ctx,
                    MirConstantOp::get_concrete_op_info(),
                    vec![element_ty_ptr],
                    vec![],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc.clone());
                let const_op = MirConstantOp::new(op);
                const_op.set_attr_value(ctx, int_attr);

                if let Some(prev) = last_op {
                    const_op.get_operation().insert_after(ctx, prev);
                } else {
                    const_op.get_operation().insert_at_front(block_ptr, ctx);
                }
                (
                    const_op.get_operation().deref(ctx).get_result(0),
                    Some(const_op.get_operation()),
                )
            }
        };

        element_values.push(elem_val);
        last_op = elem_last_op;
    }

    // Create the array construction operation with typed element values
    use dialect_mir::ops::MirConstructArrayOp;
    let construct_op = Operation::new(
        ctx,
        MirConstructArrayOp::get_concrete_op_info(),
        vec![array_ty],
        element_values,
        vec![],
        0,
    );
    construct_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        construct_op.insert_after(ctx, prev);
    } else {
        construct_op.insert_at_front(block_ptr, ctx);
    }
    last_op = Some(construct_op);

    let array_val = construct_op.deref(ctx).get_result(0);

    // Create reference operation to get pointer to the array
    use dialect_mir::ops::MirRefOp;
    use dialect_mir::types::MirPtrType;

    let generic_ptr_ty = MirPtrType::get_generic(ctx, array_ty, false);

    let ref_op = Operation::new(
        ctx,
        MirRefOp::get_concrete_op_info(),
        vec![generic_ptr_ty.into()],
        vec![array_val],
        vec![],
        0,
    );
    ref_op.deref_mut(ctx).set_loc(loc);

    let ref_op_wrapper = MirRefOp::new(ref_op);
    ref_op_wrapper.set_mutable(ctx, false);

    if let Some(prev) = last_op {
        ref_op.insert_after(ctx, prev);
    } else {
        ref_op.insert_at_front(block_ptr, ctx);
    }

    let ptr_val = ref_op.deref(ctx).get_result(0);

    Ok((ptr_val, Some(ref_op)))
}

/// ## How it works
///
/// 1. Get the struct's field types from the MIR type
/// 2. Extract bytes from the constant's allocation
/// 3. Parse bytes for each field (handling ZST fields specially)
/// 4. Create constant operations for each field
/// 5. Create MirConstructStructOp with those operands
///
/// ## Limitations
///
/// - Assumes simple layout without complex padding (works for most structs)
/// - Nested structs with complex layouts may need refinement
fn translate_struct_constant(
    ctx: &mut Context,
    constant: &mir::ConstOperand,
    _rust_ty: &rustc_public::ty::Ty, // Reserved for future layout computation
    const_ty_ptr: Ptr<TypeObj>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};

    // Get the struct type to access field information
    // Clone field types to avoid borrow conflicts when we need to mutate ctx later
    let field_types: Vec<Ptr<TypeObj>> = {
        let ty_obj = const_ty_ptr.deref(ctx);
        let struct_ty = ty_obj
            .downcast_ref::<dialect_mir::types::MirStructType>()
            .ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(
                    "translate_struct_constant called on non-struct type"
                ))
            })?;
        struct_ty.field_types().to_vec()
    };

    // Get the bytes from the constant's allocation.
    // For promoted constants like &(8..16), the allocation contains a pointer
    // (8 zero bytes with provenance) pointing to another allocation with the actual struct data.
    // We need to follow the provenance to get the real struct bytes.
    let bytes = match constant.const_.kind() {
        ConstantKind::Allocated(alloc) => {
            // Check if this allocation has provenance (i.e., it's a pointer to another allocation)
            if let Some((_, prov)) = alloc.provenance.ptrs.first() {
                // Follow the provenance to get the actual struct allocation
                use rustc_public::mir::alloc::GlobalAlloc;
                let alloc_id = prov.0;
                match GlobalAlloc::from(alloc_id) {
                    GlobalAlloc::Memory(target_alloc) => {
                        target_alloc.raw_bytes().ok().unwrap_or_else(|| {
                            target_alloc
                                .bytes
                                .iter()
                                .map(|opt: &Option<u8>| opt.unwrap_or(0))
                                .collect::<Vec<u8>>()
                        })
                    }
                    GlobalAlloc::Static(static_def) => {
                        let target_alloc = static_def.eval_initializer().map_err(|e| {
                            input_error_noloc!(TranslationErr::unsupported(format!(
                                "Failed to evaluate static initializer for struct constant: {:?}",
                                e
                            )))
                        })?;
                        target_alloc.raw_bytes().ok().unwrap_or_else(|| {
                            target_alloc
                                .bytes
                                .iter()
                                .map(|opt: &Option<u8>| opt.unwrap_or(0))
                                .collect::<Vec<u8>>()
                        })
                    }
                    other => {
                        return input_err!(
                            loc,
                            TranslationErr::unsupported(format!(
                                "Struct constant provenance points to non-memory allocation: {:?}",
                                other
                            ))
                        );
                    }
                }
            } else {
                // No provenance - use bytes directly (inline struct constant)
                alloc.raw_bytes().ok().unwrap_or_else(|| {
                    alloc
                        .bytes
                        .iter()
                        .map(|opt| opt.unwrap_or(0))
                        .collect::<Vec<u8>>()
                })
            }
        }
        ConstantKind::ZeroSized => {
            // ZeroSized structs have no bytes - this shouldn't happen for non-ZST structs
            // but handle gracefully
            vec![]
        }
        _ => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "Struct constant must be Allocated, got: {:?}. \
                     Consider using inline construction: `let s = MyStruct {{ field: value }};`",
                    constant.const_.kind()
                ))
            );
        }
    };

    // Parse field values from the bytes
    let mut field_values = Vec::with_capacity(field_types.len());
    let mut current_prev_op = prev_op;
    let mut byte_offset = 0usize;

    for (field_idx, field_ty_ptr) in field_types.iter().copied().enumerate() {
        // First, gather type information we need while holding immutable borrow
        enum FieldTypeKind {
            ZstStruct, // Struct ZST like PhantomData<T>
            ZstTuple,  // Tuple ZST like ()
            Integer { width: u32, signedness: Signedness },
            Float16,
            Float32,
            Pointer,
            Unsupported,
        }

        let field_kind = {
            let field_ty = field_ty_ptr.deref(ctx);

            // Check for ZST
            if types::is_zst_type(ctx, field_ty_ptr) {
                // Distinguish between struct ZSTs and tuple ZSTs
                if field_ty.is::<dialect_mir::types::MirStructType>() {
                    FieldTypeKind::ZstStruct
                } else {
                    FieldTypeKind::ZstTuple
                }
            } else if let Some(int_ty) = field_ty.downcast_ref::<IntegerType>() {
                FieldTypeKind::Integer {
                    width: int_ty.width(),
                    signedness: int_ty.signedness(),
                }
            } else if field_ty.is::<MirFP16Type>() {
                FieldTypeKind::Float16
            } else if field_ty.is::<FP32Type>() {
                FieldTypeKind::Float32
            } else if field_ty.is::<dialect_mir::types::MirPtrType>() {
                FieldTypeKind::Pointer
            } else {
                FieldTypeKind::Unsupported
            }
        };

        // Now handle each field type kind with mutable operations
        match field_kind {
            FieldTypeKind::ZstStruct => {
                // Struct ZST fields (like PhantomData<T>) produce empty struct values
                let op = Operation::new(
                    ctx,
                    MirConstructStructOp::get_concrete_op_info(),
                    vec![field_ty_ptr], // Use the actual struct type
                    vec![],             // No operands for ZST
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc.clone());

                if let Some(prev) = current_prev_op {
                    op.insert_after(ctx, prev);
                } else {
                    op.insert_at_front(block_ptr, ctx);
                }

                current_prev_op = Some(op);
                field_values.push(op.deref(ctx).get_result(0));
                // ZST takes no bytes
            }

            FieldTypeKind::ZstTuple => {
                // Tuple ZST fields produce empty tuple values
                let empty_tuple_ty = dialect_mir::types::MirTupleType::get(ctx, vec![]).into();

                use dialect_mir::ops::MirConstructTupleOp;
                let op = Operation::new(
                    ctx,
                    MirConstructTupleOp::get_concrete_op_info(),
                    vec![empty_tuple_ty],
                    vec![],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc.clone());

                if let Some(prev) = current_prev_op {
                    op.insert_after(ctx, prev);
                } else {
                    op.insert_at_front(block_ptr, ctx);
                }

                current_prev_op = Some(op);
                field_values.push(op.deref(ctx).get_result(0));
                // ZST takes no bytes
            }

            FieldTypeKind::Integer { width, signedness } => {
                let byte_size = (width as usize).div_ceil(8);

                // Extract bytes for this field
                let field_bytes = if byte_offset + byte_size <= bytes.len() {
                    &bytes[byte_offset..byte_offset + byte_size]
                } else {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "Struct constant has insufficient bytes for field {} (need {} bytes at offset {}, have {})",
                            field_idx,
                            byte_size,
                            byte_offset,
                            bytes.len()
                        ))
                    );
                };

                let int_val = read_uint_from_bytes(field_bytes);

                // Create the constant operation
                let width_nz = NonZeroUsize::new(width as usize).unwrap();
                let apint = APInt::from_u128(int_val, width_nz);
                let int_attr = pliron::builtin::attributes::IntegerAttr::new(
                    IntegerType::get(ctx, width, signedness),
                    apint,
                );

                use dialect_mir::ops::MirConstantOp;
                let op = Operation::new(
                    ctx,
                    MirConstantOp::get_concrete_op_info(),
                    vec![field_ty_ptr],
                    vec![],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc.clone());

                let const_op = MirConstantOp::new(op);
                const_op.set_attr_value(ctx, int_attr);

                if let Some(prev) = current_prev_op {
                    const_op.get_operation().insert_after(ctx, prev);
                } else {
                    const_op.get_operation().insert_at_front(block_ptr, ctx);
                }

                current_prev_op = Some(const_op.get_operation());
                field_values.push(const_op.get_operation().deref(ctx).get_result(0));

                byte_offset += byte_size;
            }

            FieldTypeKind::Float16 => {
                let byte_size = 2;

                let field_bytes = if byte_offset + byte_size <= bytes.len() {
                    &bytes[byte_offset..byte_offset + byte_size]
                } else {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "Struct constant has insufficient bytes for f16 field {}",
                            field_idx
                        ))
                    );
                };

                let bits = read_uint_from_bytes(field_bytes) as u16;
                let float_attr = MirFP16Attr::from_bits(bits);

                use dialect_mir::ops::MirFloatConstantOp;
                let op = Operation::new(
                    ctx,
                    MirFloatConstantOp::get_concrete_op_info(),
                    vec![field_ty_ptr],
                    vec![],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc.clone());

                let float_op = MirFloatConstantOp::new(op);
                float_op.set_attr_float_value_f16(ctx, float_attr);

                if let Some(prev) = current_prev_op {
                    float_op.get_operation().insert_after(ctx, prev);
                } else {
                    float_op.get_operation().insert_at_front(block_ptr, ctx);
                }

                current_prev_op = Some(float_op.get_operation());
                field_values.push(float_op.get_operation().deref(ctx).get_result(0));

                byte_offset += byte_size;
            }

            FieldTypeKind::Float32 => {
                let byte_size = 4;

                let field_bytes = if byte_offset + byte_size <= bytes.len() {
                    &bytes[byte_offset..byte_offset + byte_size]
                } else {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "Struct constant has insufficient bytes for f32 field {} (need {} bytes at offset {}, have {})",
                            field_idx,
                            byte_size,
                            byte_offset,
                            bytes.len()
                        ))
                    );
                };

                let float_val = f32::from_le_bytes([
                    field_bytes[0],
                    field_bytes[1],
                    field_bytes[2],
                    field_bytes[3],
                ]);

                let float_attr = pliron::builtin::attributes::FPSingleAttr::from(float_val);

                use dialect_mir::ops::MirFloatConstantOp;
                let op = Operation::new(
                    ctx,
                    MirFloatConstantOp::get_concrete_op_info(),
                    vec![field_ty_ptr],
                    vec![],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc.clone());

                let float_op = MirFloatConstantOp::new(op);
                float_op.set_attr_float_value(ctx, float_attr);

                if let Some(prev) = current_prev_op {
                    float_op.get_operation().insert_after(ctx, prev);
                } else {
                    float_op.get_operation().insert_at_front(block_ptr, ctx);
                }

                current_prev_op = Some(float_op.get_operation());
                field_values.push(float_op.get_operation().deref(ctx).get_result(0));

                byte_offset += byte_size;
            }

            FieldTypeKind::Pointer => {
                let byte_size = 8; // 64-bit pointers

                let field_bytes = if byte_offset + byte_size <= bytes.len() {
                    &bytes[byte_offset..byte_offset + byte_size]
                } else {
                    return input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "Struct constant has insufficient bytes for pointer field {} (need {} bytes at offset {}, have {})",
                            field_idx,
                            byte_size,
                            byte_offset,
                            bytes.len()
                        ))
                    );
                };

                let mut ptr_val: u64 = 0;
                for (i, &byte) in field_bytes.iter().enumerate() {
                    ptr_val |= (byte as u64) << (i * 8);
                }

                // Create integer constant then cast to pointer
                let i64_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
                let apint = APInt::from_u64(ptr_val, NonZeroUsize::new(64).unwrap());
                let int_attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, apint);

                use dialect_mir::ops::MirConstantOp;
                let op = Operation::new(
                    ctx,
                    MirConstantOp::get_concrete_op_info(),
                    vec![i64_ty.into()],
                    vec![],
                    vec![],
                    0,
                );
                op.deref_mut(ctx).set_loc(loc.clone());

                let const_op = MirConstantOp::new(op);
                const_op.set_attr_value(ctx, int_attr);

                if let Some(prev) = current_prev_op {
                    const_op.get_operation().insert_after(ctx, prev);
                } else {
                    const_op.get_operation().insert_at_front(block_ptr, ctx);
                }

                // Cast to pointer type
                use dialect_mir::ops::MirCastOp;
                let const_value = const_op.get_operation().deref(ctx).get_result(0);
                let cast_op = Operation::new(
                    ctx,
                    MirCastOp::get_concrete_op_info(),
                    vec![field_ty_ptr],
                    vec![const_value],
                    vec![],
                    0,
                );
                cast_op.deref_mut(ctx).set_loc(loc.clone());
                MirCastOp::new(cast_op)
                    .set_attr_cast_kind(ctx, MirCastKindAttr::PointerWithExposedProvenance);
                cast_op.insert_after(ctx, const_op.get_operation());

                current_prev_op = Some(cast_op);
                field_values.push(cast_op.deref(ctx).get_result(0));

                byte_offset += byte_size;
            }

            FieldTypeKind::Unsupported => {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Struct constant field {} has unsupported type. \
                         Consider using inline construction instead of const.",
                        field_idx
                    ))
                );
            }
        }
    }

    // Cast field values to expected types (address space normalization)
    let (casted_field_values, prev_after_casts) = cast_struct_fields_to_expected_types(
        ctx,
        field_values,
        const_ty_ptr,
        block_ptr,
        current_prev_op,
        loc.clone(),
    );

    // Create the MirConstructStructOp with all field values
    let op = Operation::new(
        ctx,
        MirConstructStructOp::get_concrete_op_info(),
        vec![const_ty_ptr],
        casted_field_values,
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc);

    if let Some(prev) = prev_after_casts {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }

    let val = op.deref(ctx).get_result(0);
    Ok((val, Some(op)))
}

/// Translate a non-empty tuple constant from its raw allocation bytes.
fn translate_tuple_constant(
    ctx: &mut Context,
    constant: &mir::ConstOperand,
    rust_ty: &rustc_public::ty::Ty,
    const_ty_ptr: Ptr<TypeObj>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let field_types = {
        let ty_ref = const_ty_ptr.deref(ctx);
        ty_ref
            .downcast_ref::<dialect_mir::types::MirTupleType>()
            .ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(
                    "translate_tuple_constant called on non-tuple type"
                ))
            })?
            .get_types()
            .to_vec()
    };

    let rust_field_types = match rust_ty.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Tuple(fields)) => {
            fields.to_vec()
        }
        other => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "Tuple constant expected Rust tuple type, got {:?}",
                    other
                ))
            );
        }
    };

    if field_types.len() != rust_field_types.len() {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Tuple constant type mismatch: MIR has {} fields, Rust type has {}",
                field_types.len(),
                rust_field_types.len()
            ))
        );
    }

    let bytes = constant_bytes(constant, "tuple", loc.clone())?;
    let mut values = Vec::with_capacity(field_types.len());
    let mut byte_offset = 0usize;
    let mut current_prev_op = prev_op;

    for (field_idx, (field_ty, rust_field_ty)) in field_types
        .iter()
        .copied()
        .zip(rust_field_types.iter())
        .enumerate()
    {
        let byte_size = constant_storage_size(ctx, field_ty).ok_or_else(|| {
            input_error!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "Tuple constant field {} has unsupported type {:?}",
                    field_idx,
                    field_ty.deref(ctx)
                ))
            )
        })?;

        let field_bytes = if byte_size == 0 {
            &[][..]
        } else if byte_offset + byte_size <= bytes.len() {
            &bytes[byte_offset..byte_offset + byte_size]
        } else {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "Tuple constant has insufficient bytes for field {} (need {} bytes at offset {}, have {})",
                    field_idx,
                    byte_size,
                    byte_offset,
                    bytes.len()
                ))
            );
        };

        let (value, new_prev_op) = translate_constant_value_from_bytes(
            ctx,
            rust_field_ty,
            field_ty,
            field_bytes,
            block_ptr,
            current_prev_op,
            loc.clone(),
        )?;
        values.push(value);
        current_prev_op = new_prev_op;
        byte_offset += byte_size;
    }

    use dialect_mir::ops::MirConstructTupleOp;
    let op = Operation::new(
        ctx,
        MirConstructTupleOp::get_concrete_op_info(),
        vec![const_ty_ptr],
        values,
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc);

    if let Some(prev) = current_prev_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }

    Ok((op.deref(ctx).get_result(0), Some(op)))
}

fn constant_storage_size(ctx: &Context, ty_ptr: Ptr<TypeObj>) -> Option<usize> {
    let ty_ref = ty_ptr.deref(ctx);
    if types::is_zst_type(ctx, ty_ptr) {
        Some(0)
    } else if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        Some((int_ty.width() as usize).div_ceil(8))
    } else if ty_ref.is::<MirFP16Type>() {
        Some(2)
    } else if ty_ref.is::<FP32Type>() {
        Some(4)
    } else if ty_ref.is::<FP64Type>() {
        Some(8)
    } else if ty_ref.is::<dialect_mir::types::MirPtrType>() {
        Some(rustc_public::target::MachineInfo::target_pointer_width().bytes())
    } else {
        None
    }
}

/// Translate an enum constant by reconstructing both its active variant and any
/// payload operands from the constant's layout bytes.
fn translate_enum_constant(
    ctx: &mut Context,
    constant: &mir::ConstOperand,
    rust_ty: &rustc_public::ty::Ty,
    const_ty_ptr: Ptr<TypeObj>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let enum_bytes = constant_bytes(constant, "enum", loc.clone())?;
    translate_enum_constant_from_bytes(
        ctx,
        rust_ty,
        const_ty_ptr,
        &enum_bytes,
        block_ptr,
        prev_op,
        loc,
    )
}

/// Translate an enum value from raw bytes plus the Rust type/layout metadata.
fn translate_enum_constant_from_bytes(
    ctx: &mut Context,
    rust_ty: &rustc_public::ty::Ty,
    const_ty_ptr: Ptr<TypeObj>,
    enum_bytes: &[u8],
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let enum_variant = {
        let ty_obj = const_ty_ptr.deref(ctx);
        let enum_ty = ty_obj
            .downcast_ref::<dialect_mir::types::MirEnumType>()
            .ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(
                    "translate_enum_constant_from_bytes called on non-enum type"
                ))
            })?;

        let variant_index = enum_variant_index_from_bytes(rust_ty, enum_bytes, loc.clone())?;
        let variant = enum_ty.get_variant(variant_index).ok_or_else(|| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "Enum constant resolved to variant index {} outside translated MIR enum '{}'",
                variant_index,
                enum_ty.name()
            )))
        })?;
        (variant_index, variant)
    };
    let variant_index = enum_variant.0;
    let variant = enum_variant.1;

    let mut field_values = Vec::with_capacity(variant.field_types.len());
    let mut current_prev_op = prev_op;

    if !variant.field_types.is_empty() {
        use rustc_public::ty::{RigidTy, TyKind};

        let layout = rust_ty.layout().map_err(|e| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "Failed to query layout for enum constant: {:?}",
                e
            )))
        })?;
        let field_offsets =
            enum_variant_field_offsets(&layout.shape(), variant_index, loc.clone())?;

        let (adt_def, substs) = match rust_ty.kind() {
            TyKind::RigidTy(RigidTy::Adt(adt_def, substs)) => (adt_def, substs),
            other => {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Expected ADT Rust type for enum constant, got {:?}",
                        other
                    ))
                );
            }
        };
        let rust_variant = &adt_def.variants()[variant_index];

        for (field_idx, field_ty_ptr) in variant.field_types.iter().copied().enumerate() {
            let rust_field_ty = rust_variant.fields()[field_idx].ty_with_args(&substs);
            let field_layout = rust_field_ty.layout().map_err(|e| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "Failed to query layout for enum field {} of variant '{}': {:?}",
                    field_idx,
                    rust_variant.name(),
                    e
                )))
            })?;
            let field_offset = *field_offsets.get(field_idx).ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "Missing layout offset for enum field {} of variant '{}'",
                    field_idx,
                    rust_variant.name()
                )))
            })?;
            let field_size = field_layout.shape().size.bytes() as usize;
            let field_end = field_offset.checked_add(field_size).ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "Enum field {} of variant '{}' overflowed offset computation",
                    field_idx,
                    rust_variant.name()
                )))
            })?;

            if field_end > enum_bytes.len() {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Enum constant for variant '{}' has {} bytes, but field {} needs [{}..{})",
                        rust_variant.name(),
                        enum_bytes.len(),
                        field_idx,
                        field_offset,
                        field_end
                    ))
                );
            }

            let field_bytes = &enum_bytes[field_offset..field_end];
            let (field_val, new_prev_op) = translate_constant_value_from_bytes(
                ctx,
                &rust_field_ty,
                field_ty_ptr,
                field_bytes,
                block_ptr,
                current_prev_op,
                loc.clone(),
            )?;
            field_values.push(field_val);
            current_prev_op = new_prev_op;
        }

        let (casted_field_values, prev_after_casts) = cast_enum_fields_to_expected_types(
            ctx,
            field_values,
            const_ty_ptr,
            variant_index,
            block_ptr,
            current_prev_op,
            loc.clone(),
        );
        field_values = casted_field_values;
        current_prev_op = prev_after_casts;
    }

    let op = Operation::new(
        ctx,
        MirConstructEnumOp::get_concrete_op_info(),
        vec![const_ty_ptr],
        field_values,
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    let enum_op = MirConstructEnumOp::new(op);
    enum_op.set_attr_construct_enum_variant_index(
        ctx,
        dialect_mir::attributes::VariantIndexAttr(variant_index as u32),
    );

    if let Some(prev) = current_prev_op {
        enum_op.get_operation().insert_after(ctx, prev);
    } else {
        enum_op.get_operation().insert_at_front(block_ptr, ctx);
    }

    let val = enum_op.get_operation().deref(ctx).get_result(0);

    Ok((val, Some(enum_op.get_operation())))
}

/// Translate one field-sized byte slice into a constant value.
fn translate_constant_value_from_bytes(
    ctx: &mut Context,
    rust_ty: &rustc_public::ty::Ty,
    ty_ptr: Ptr<TypeObj>,
    bytes: &[u8],
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    let is_enum = {
        let ty_ref = ty_ptr.deref(ctx);
        ty_ref.is::<dialect_mir::types::MirEnumType>()
    };
    if is_enum {
        return translate_enum_constant_from_bytes(
            ctx, rust_ty, ty_ptr, bytes, block_ptr, prev_op, loc,
        );
    }

    let is_zst = rust_ty
        .layout()
        .map(|layout| layout.shape().is_1zst())
        .unwrap_or(false);
    if is_zst || types::is_zst_type(ctx, ty_ptr) {
        return translate_zero_sized_constant_value(ctx, ty_ptr, block_ptr, prev_op, loc);
    }

    enum ValueKind {
        Integer { width: u32, signedness: Signedness },
        Float16,
        Float32,
        Float64,
        Pointer,
        Unsupported(String),
    }

    let value_kind = {
        let ty_ref = ty_ptr.deref(ctx);
        if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
            ValueKind::Integer {
                width: int_ty.width(),
                signedness: int_ty.signedness(),
            }
        } else if ty_ref.is::<MirFP16Type>() {
            ValueKind::Float16
        } else if ty_ref.is::<FP32Type>() {
            ValueKind::Float32
        } else if ty_ref.is::<FP64Type>() {
            ValueKind::Float64
        } else if ty_ref.is::<dialect_mir::types::MirPtrType>() {
            ValueKind::Pointer
        } else {
            ValueKind::Unsupported(format!("{:?}", ty_ref))
        }
    };

    match value_kind {
        ValueKind::Integer { width, signedness } => {
            let byte_size = (width as usize).div_ceil(8);
            if bytes.len() < byte_size {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Integer constant needs {} bytes, found {}",
                        byte_size,
                        bytes.len()
                    ))
                );
            }

            let int_val = read_uint_from_bytes(&bytes[..byte_size]);
            let width_nz = NonZeroUsize::new(width as usize).unwrap();
            let apint = APInt::from_u128(int_val, width_nz);
            let int_attr = pliron::builtin::attributes::IntegerAttr::new(
                IntegerType::get(ctx, width, signedness),
                apint,
            );

            use dialect_mir::ops::MirConstantOp;
            let op = Operation::new(
                ctx,
                MirConstantOp::get_concrete_op_info(),
                vec![ty_ptr],
                vec![],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc.clone());
            let const_op = MirConstantOp::new(op);
            const_op.set_attr_value(ctx, int_attr);

            if let Some(prev) = prev_op {
                const_op.get_operation().insert_after(ctx, prev);
            } else {
                const_op.get_operation().insert_at_front(block_ptr, ctx);
            }

            Ok((
                const_op.get_operation().deref(ctx).get_result(0),
                Some(const_op.get_operation()),
            ))
        }
        ValueKind::Float16 => {
            if bytes.len() < 2 {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "f16 constant needs 2 bytes, found {}",
                        bytes.len()
                    ))
                );
            }

            let bits = read_uint_from_bytes(&bytes[..2]) as u16;
            let float_attr = MirFP16Attr::from_bits(bits);

            use dialect_mir::ops::MirFloatConstantOp;
            let op = Operation::new(
                ctx,
                MirFloatConstantOp::get_concrete_op_info(),
                vec![ty_ptr],
                vec![],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc.clone());
            let float_op = MirFloatConstantOp::new(op);
            float_op.set_attr_float_value_f16(ctx, float_attr);

            if let Some(prev) = prev_op {
                float_op.get_operation().insert_after(ctx, prev);
            } else {
                float_op.get_operation().insert_at_front(block_ptr, ctx);
            }

            Ok((
                float_op.get_operation().deref(ctx).get_result(0),
                Some(float_op.get_operation()),
            ))
        }
        ValueKind::Float32 => {
            if bytes.len() < 4 {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "f32 constant needs 4 bytes, found {}",
                        bytes.len()
                    ))
                );
            }

            let mut field_bytes = [0u8; 4];
            field_bytes.copy_from_slice(&bytes[..4]);
            let float_val = match rustc_public::target::MachineInfo::target_endianness() {
                rustc_public::target::Endian::Little => f32::from_le_bytes(field_bytes),
                rustc_public::target::Endian::Big => f32::from_be_bytes(field_bytes),
            };
            let float_attr = pliron::builtin::attributes::FPSingleAttr::from(float_val);

            use dialect_mir::ops::MirFloatConstantOp;
            let op = Operation::new(
                ctx,
                MirFloatConstantOp::get_concrete_op_info(),
                vec![ty_ptr],
                vec![],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc.clone());
            let float_op = MirFloatConstantOp::new(op);
            float_op.set_attr_float_value(ctx, float_attr);

            if let Some(prev) = prev_op {
                float_op.get_operation().insert_after(ctx, prev);
            } else {
                float_op.get_operation().insert_at_front(block_ptr, ctx);
            }

            Ok((
                float_op.get_operation().deref(ctx).get_result(0),
                Some(float_op.get_operation()),
            ))
        }
        ValueKind::Float64 => {
            if bytes.len() < 8 {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "f64 constant needs 8 bytes, found {}",
                        bytes.len()
                    ))
                );
            }

            let mut field_bytes = [0u8; 8];
            field_bytes.copy_from_slice(&bytes[..8]);
            let float_val = match rustc_public::target::MachineInfo::target_endianness() {
                rustc_public::target::Endian::Little => f64::from_le_bytes(field_bytes),
                rustc_public::target::Endian::Big => f64::from_be_bytes(field_bytes),
            };
            let float_attr = pliron::builtin::attributes::FPDoubleAttr::from(float_val);

            use dialect_mir::ops::MirFloatConstantOp;
            let op = Operation::new(
                ctx,
                MirFloatConstantOp::get_concrete_op_info(),
                vec![ty_ptr],
                vec![],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc.clone());
            let float_op = MirFloatConstantOp::new(op);
            float_op.set_attr_float_value_f64(ctx, float_attr);

            if let Some(prev) = prev_op {
                float_op.get_operation().insert_after(ctx, prev);
            } else {
                float_op.get_operation().insert_at_front(block_ptr, ctx);
            }

            Ok((
                float_op.get_operation().deref(ctx).get_result(0),
                Some(float_op.get_operation()),
            ))
        }
        ValueKind::Pointer => {
            let pointer_bytes = rustc_public::target::MachineInfo::target_pointer_width().bytes();
            if bytes.len() < pointer_bytes {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Pointer constant needs {} bytes, found {}",
                        pointer_bytes,
                        bytes.len()
                    ))
                );
            }

            let ptr_val = read_uint_from_bytes(&bytes[..pointer_bytes]) as u64;
            let i64_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
            let apint = APInt::from_u64(ptr_val, NonZeroUsize::new(64).unwrap());
            let int_attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, apint);

            use dialect_mir::ops::MirConstantOp;
            let int_op = Operation::new(
                ctx,
                MirConstantOp::get_concrete_op_info(),
                vec![i64_ty.into()],
                vec![],
                vec![],
                0,
            );
            int_op.deref_mut(ctx).set_loc(loc.clone());
            let const_op = MirConstantOp::new(int_op);
            const_op.set_attr_value(ctx, int_attr);

            if let Some(prev) = prev_op {
                const_op.get_operation().insert_after(ctx, prev);
            } else {
                const_op.get_operation().insert_at_front(block_ptr, ctx);
            }

            let const_value = const_op.get_operation().deref(ctx).get_result(0);
            let cast_op = Operation::new(
                ctx,
                MirCastOp::get_concrete_op_info(),
                vec![ty_ptr],
                vec![const_value],
                vec![],
                0,
            );
            cast_op.deref_mut(ctx).set_loc(loc.clone());
            MirCastOp::new(cast_op)
                .set_attr_cast_kind(ctx, MirCastKindAttr::PointerWithExposedProvenance);
            cast_op.insert_after(ctx, const_op.get_operation());

            Ok((cast_op.deref(ctx).get_result(0), Some(cast_op)))
        }
        ValueKind::Unsupported(ty_name) => input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Enum payload constant field type is not yet supported: {}",
                ty_name
            ))
        ),
    }
}

/// Build a zero-sized struct or tuple value.
fn translate_zero_sized_constant_value(
    ctx: &mut Context,
    ty_ptr: Ptr<TypeObj>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Option<Ptr<Operation>>)> {
    enum ZeroSizedKind {
        Struct,
        EmptyTuple,
        Unsupported(String),
    }

    let zero_sized_kind = {
        let ty_ref = ty_ptr.deref(ctx);
        if ty_ref.is::<dialect_mir::types::MirStructType>() {
            ZeroSizedKind::Struct
        } else if let Some(tuple_ty) = ty_ref.downcast_ref::<dialect_mir::types::MirTupleType>() {
            if tuple_ty.get_types().is_empty() {
                ZeroSizedKind::EmptyTuple
            } else {
                ZeroSizedKind::Unsupported(
                    "Only empty tuple constants can be synthesized as zero-sized values"
                        .to_string(),
                )
            }
        } else {
            ZeroSizedKind::Unsupported(format!(
                "Zero-sized constant synthesis does not support type {:?}",
                ty_ref
            ))
        }
    };

    let op = match zero_sized_kind {
        ZeroSizedKind::Struct => Operation::new(
            ctx,
            MirConstructStructOp::get_concrete_op_info(),
            vec![ty_ptr],
            vec![],
            vec![],
            0,
        ),
        ZeroSizedKind::EmptyTuple => {
            use dialect_mir::ops::MirConstructTupleOp;
            Operation::new(
                ctx,
                MirConstructTupleOp::get_concrete_op_info(),
                vec![ty_ptr],
                vec![],
                vec![],
                0,
            )
        }
        ZeroSizedKind::Unsupported(message) => {
            return input_err!(loc, TranslationErr::unsupported(message));
        }
    };
    op.deref_mut(ctx).set_loc(loc);

    if let Some(prev) = prev_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }

    Ok((op.deref(ctx).get_result(0), Some(op)))
}

/// Translate ADT aggregate operands, synthesizing omitted runtime-ZST fields when
/// MIR carries only the non-ZST runtime operands.
fn translate_adt_aggregate_field_values(
    ctx: &mut Context,
    body: &mir::Body,
    adt_def: rustc_public::ty::AdtDef,
    variant_idx: rustc_public::ty::VariantIdx,
    substs: &rustc_public::ty::GenericArgs,
    operands: &[mir::Operand],
    value_map: &mut ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Vec<Value>, Option<Ptr<Operation>>)> {
    let variant_index = variant_idx.to_index();
    let variant = &adt_def.variants()[variant_index];

    let mut field_infos = Vec::with_capacity(variant.fields().len());
    for field in variant.fields() {
        let field_rust_ty = field.ty_with_args(substs);
        let translated_ty = types::translate_type(ctx, &field_rust_ty)?;
        let is_runtime_zst = field_rust_ty
            .layout()
            .map(|layout| layout.shape().is_1zst())
            .unwrap_or(false);
        field_infos.push((field_rust_ty, translated_ty, is_runtime_zst));
    }

    let total_field_count = field_infos.len();
    let non_zst_count = field_infos
        .iter()
        .filter(|(_, _, is_runtime_zst)| !*is_runtime_zst)
        .count();

    let synthesize_runtime_zsts = if operands.len() == total_field_count {
        false
    } else if operands.len() == non_zst_count {
        true
    } else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "ADT aggregate '{}' variant '{}' has {} translated fields, {} non-ZST runtime fields, but MIR provided {} operands",
                adt_def.trimmed_name(),
                variant.name(),
                total_field_count,
                non_zst_count,
                operands.len()
            ))
        );
    };

    let mut field_values = Vec::with_capacity(total_field_count);
    let mut current_prev_op = prev_op;
    let mut operand_iter = operands.iter();

    for (field_rust_ty, translated_ty, is_runtime_zst) in field_infos {
        if synthesize_runtime_zsts && is_runtime_zst {
            let (value, new_prev_op) = translate_constant_value_from_bytes(
                ctx,
                &field_rust_ty,
                translated_ty,
                &[],
                block_ptr,
                current_prev_op,
                loc.clone(),
            )?;
            field_values.push(value);
            current_prev_op = new_prev_op;
            continue;
        }

        let operand = operand_iter.next().ok_or_else(|| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "ADT aggregate '{}' variant '{}' ran out of MIR operands while translating fields",
                adt_def.trimmed_name(),
                variant.name()
            )))
        })?;
        let (value, new_prev_op) = translate_operand(
            ctx,
            body,
            operand,
            value_map,
            block_ptr,
            current_prev_op,
            loc.clone(),
        )?;
        field_values.push(value);
        current_prev_op = new_prev_op;
    }

    if operand_iter.next().is_some() {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "ADT aggregate '{}' variant '{}' left unused MIR operands after field translation",
                adt_def.trimmed_name(),
                variant.name()
            ))
        );
    }

    Ok((field_values, current_prev_op))
}

/// Fetch the raw bytes backing a constant, following provenance for promoted
/// aggregate constants when necessary.
pub(crate) fn constant_bytes(
    constant: &mir::ConstOperand,
    kind_name: &str,
    loc: Location,
) -> TranslationResult<Vec<u8>> {
    use rustc_public::ty::TyConstKind;

    fn allocation_bytes(
        alloc: &rustc_public::ty::Allocation,
        kind_name: &str,
        loc: Location,
    ) -> TranslationResult<Vec<u8>> {
        use rustc_public::mir::alloc::GlobalAlloc;

        if let Some((_, prov)) = alloc.provenance.ptrs.first() {
            let alloc_id = prov.0;
            match GlobalAlloc::from(alloc_id) {
                GlobalAlloc::Memory(target_alloc) => {
                    Ok(target_alloc.raw_bytes().ok().unwrap_or_else(|| {
                        target_alloc
                            .bytes
                            .iter()
                            .map(|opt: &Option<u8>| opt.unwrap_or(0))
                            .collect::<Vec<u8>>()
                    }))
                }
                GlobalAlloc::Static(static_def) => {
                    let target_alloc = static_def.eval_initializer().map_err(|e| {
                        input_error_noloc!(TranslationErr::unsupported(format!(
                            "Failed to evaluate static initializer for {} constant: {:?}",
                            kind_name, e
                        )))
                    })?;
                    Ok(target_alloc.raw_bytes().ok().unwrap_or_else(|| {
                        target_alloc
                            .bytes
                            .iter()
                            .map(|opt: &Option<u8>| opt.unwrap_or(0))
                            .collect::<Vec<u8>>()
                    }))
                }
                other => input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "{} constant provenance points to non-memory allocation: {:?}",
                        kind_name, other
                    ))
                ),
            }
        } else {
            Ok(alloc.raw_bytes().ok().unwrap_or_else(|| {
                alloc
                    .bytes
                    .iter()
                    .map(|opt| opt.unwrap_or(0))
                    .collect::<Vec<u8>>()
            }))
        }
    }

    match constant.const_.kind() {
        ConstantKind::Allocated(alloc) => allocation_bytes(alloc, kind_name, loc),
        ConstantKind::Ty(ty_const) => match ty_const.kind() {
            TyConstKind::Value(_, alloc) => allocation_bytes(alloc, kind_name, loc),
            TyConstKind::ZSTValue(_) => Ok(vec![]),
            other => input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "{} constant must be backed by bytes, found TyConstKind::{:?}",
                    kind_name, other
                ))
            ),
        },
        ConstantKind::ZeroSized => Ok(vec![]),
        other => input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "{} constant must be Allocated or Ty::Value, got {:?}",
                kind_name, other
            ))
        ),
    }
}

/// Determine the active enum variant from layout metadata plus raw bytes.
fn enum_variant_index_from_bytes(
    rust_ty: &rustc_public::ty::Ty,
    enum_bytes: &[u8],
    loc: Location,
) -> TranslationResult<usize> {
    let layout = rust_ty.layout().map_err(|e| {
        input_error_noloc!(TranslationErr::unsupported(format!(
            "Failed to query enum layout: {:?}",
            e
        )))
    })?;
    let shape = layout.shape();

    match &shape.variants {
        rustc_public::abi::VariantsShape::Single { index } => Ok(index.to_index()),
        rustc_public::abi::VariantsShape::Empty => input_err!(
            loc,
            TranslationErr::unsupported("Cannot materialize a constant for an uninhabited enum")
        ),
        rustc_public::abi::VariantsShape::Multiple {
            tag,
            tag_encoding,
            tag_field,
            ..
        } => {
            let tag_value =
                read_enum_tag_value(enum_bytes, &shape.fields, *tag_field, *tag, loc.clone())?;

            match tag_encoding {
                rustc_public::abi::TagEncoding::Direct => {
                    // The tag bytes hold a declared discriminant VALUE
                    // truncated to the PHYSICAL tag width; the caller wants
                    // a variant INDEX. `discriminant_for_variant().val` is
                    // at the declared discriminant type's width (isize for
                    // default-repr enums), so the comparison must mask both
                    // sides to the tag width (`Neg::N = -5` is byte 0xFB in
                    // an i8 tag but 0xFFFF_FFFF_FFFF_FFFB as isize). A tag
                    // that matches no declared discriminant means we
                    // misread the constant; falling back to
                    // "value == index" would silently conflate the two
                    // semantics (the issue #146 bug class).
                    let primitive = match tag {
                        rustc_public::abi::Scalar::Initialized { value, .. }
                        | rustc_public::abi::Scalar::Union { value } => *value,
                    };
                    let scalar_size = primitive.size(&rustc_public::target::MachineInfo::target());
                    let mask = scalar_size.unsigned_int_max().ok_or_else(|| {
                        input_error_noloc!(TranslationErr::unsupported(format!(
                            "Enum tag width {} exceeds 128 bits",
                            scalar_size.bits()
                        )))
                    })?;

                    discriminant_to_variant_index(rust_ty, tag_value, mask).ok_or_else(|| {
                        input_error!(
                            loc.clone(),
                            TranslationErr::unsupported(format!(
                                "Enum constant tag value {} matches no declared discriminant",
                                tag_value
                            ))
                        )
                    })
                }
                rustc_public::abi::TagEncoding::Niche {
                    untagged_variant,
                    niche_variants,
                    niche_start,
                } => {
                    let primitive = match tag {
                        rustc_public::abi::Scalar::Initialized { value, .. }
                        | rustc_public::abi::Scalar::Union { value } => *value,
                    };
                    let scalar_size = primitive.size(&rustc_public::target::MachineInfo::target());
                    let mask = scalar_size.unsigned_int_max().ok_or_else(|| {
                        input_error_noloc!(TranslationErr::unsupported(format!(
                            "Enum niche tag width {} exceeds 128 bits",
                            scalar_size.bits()
                        )))
                    })?;

                    let niche_start_idx = niche_variants.start().to_index();
                    let niche_end_idx = niche_variants.end().to_index();
                    let relative = tag_value.wrapping_sub(*niche_start) & mask;
                    let candidate = niche_start_idx.saturating_add(relative as usize);

                    if candidate >= niche_start_idx && candidate <= niche_end_idx {
                        Ok(candidate)
                    } else {
                        Ok(untagged_variant.to_index())
                    }
                }
            }
        }
    }
}

/// Lower a `fn item -> fn pointer` coercion (`ReifyFnPointer`).
///
/// Emits a stable per-function token (hash of the function's mangled
/// name, never 0 so it cannot look like a null pointer) and casts it
/// int -> ptr. See the comment at the `Rvalue::Cast` arm for why a token
/// stands in for a code address on the device.
fn translate_reify_fn_pointer(
    ctx: &mut Context,
    body: &mir::Body,
    operand: &mir::Operand,
    dest_ty: &rustc_public::ty::Ty,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Option<Ptr<Operation>>, Value, Option<Ptr<Operation>>)> {
    use dialect_mir::ops::MirConstantOp;
    use rustc_public::mir::mono::Instance;
    use std::hash::{Hash, Hasher};

    // The operand's type names the function being reified.
    let operand_ty = operand.ty(body.locals()).map_err(|e| {
        input_error_noloc!(TranslationErr::unsupported(format!(
            "ReifyFnPointer: cannot read operand type: {e:?}"
        )))
    })?;
    let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnDef(fn_def, substs)) =
        operand_ty.kind()
    else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "ReifyFnPointer on a non-fn-item operand of type {operand_ty:?}"
            ))
        );
    };
    let mangled = Instance::resolve(fn_def, &substs)
        .map_err(|e| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "ReifyFnPointer: cannot resolve fn item: {e:?}"
            )))
        })?
        .mangled_name();
    let token = {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        mangled.hash(&mut h);
        h.finish() | 1
    };

    // Materialize the token and cast it to the fn-pointer type, the same
    // two-op shape used for provenance-carrying pointer constants.
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let apint = APInt::from_u64(token, NonZeroUsize::new(64).unwrap());
    let int_attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, apint);
    let int_op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![],
        vec![],
        0,
    );
    int_op.deref_mut(ctx).set_loc(loc.clone());
    MirConstantOp::new(int_op).set_attr_value(ctx, int_attr);
    match prev_op {
        Some(prev) => int_op.insert_after(ctx, prev),
        None => int_op.insert_at_front(block_ptr, ctx),
    }
    let int_val = int_op.deref(ctx).get_result(0);

    let result_type = types::translate_type(ctx, dest_ty)?;
    let cast_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![result_type],
        vec![int_val],
        vec![],
        0,
    );
    cast_op.deref_mut(ctx).set_loc(loc);
    MirCastOp::new(cast_op).set_attr_cast_kind(ctx, MirCastKindAttr::PointerWithExposedProvenance);

    let result = cast_op.deref(ctx).get_result(0);
    Ok((Some(cast_op), result, Some(int_op)))
}

// Byte-offset lookups over rustc enum layout live in the shared
// `translator::layout` module so type import and constant decoding cannot
// drift on how an offset is derived.
use crate::translator::layout::{enum_tag_offset, enum_variant_field_offsets};

/// Read an enum tag scalar from raw bytes using the stable layout metadata.
fn read_enum_tag_value(
    enum_bytes: &[u8],
    fields: &rustc_public::abi::FieldsShape,
    tag_field: usize,
    tag: rustc_public::abi::Scalar,
    loc: Location,
) -> TranslationResult<u128> {
    let primitive = match tag {
        rustc_public::abi::Scalar::Initialized { value, .. }
        | rustc_public::abi::Scalar::Union { value } => value,
    };
    let byte_size = primitive
        .size(&rustc_public::target::MachineInfo::target())
        .bytes();

    let offset = enum_tag_offset(fields, tag_field, loc.clone())?;

    let end = offset.checked_add(byte_size).ok_or_else(|| {
        input_error_noloc!(TranslationErr::unsupported(format!(
            "Enum tag overflowed offset computation: offset={}, size={}",
            offset, byte_size
        )))
    })?;
    if end > enum_bytes.len() {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Enum tag needs bytes [{}..{}), but constant only has {} bytes",
                offset,
                end,
                enum_bytes.len()
            ))
        );
    }

    Ok(read_uint_from_bytes(&enum_bytes[offset..end]))
}

/// Decode an integer from raw bytes using the current target endianness.
fn read_uint_from_bytes(bytes: &[u8]) -> u128 {
    match rustc_public::target::MachineInfo::target_endianness() {
        rustc_public::target::Endian::Little => {
            bytes.iter().enumerate().fold(0u128, |acc, (idx, byte)| {
                acc | ((*byte as u128) << (idx * 8))
            })
        }
        rustc_public::target::Endian::Big => bytes
            .iter()
            .fold(0u128, |acc, byte| (acc << 8) | (*byte as u128)),
    }
}

/// Convert a discriminant value to a variant index.
///
/// For enums with explicit discriminants (e.g., `enum { A = 0, B = 2, C = 6 }`),
/// the discriminant value differs from the variant index:
/// - Variant index: position in the enum (0, 1, 2, ...)
/// - Discriminant: the explicit or implicit value assigned to each variant
///
/// `tag_value` is the raw tag read from memory, i.e. the discriminant
/// truncated to the PHYSICAL tag width, while `discriminant_for_variant`
/// reports values at the declared discriminant type's width (isize for
/// default-repr enums). `mask` is the tag width's unsigned max; both
/// sides are masked to it so negative discriminants compare correctly
/// (`-5` is `0xFB` in an i8 tag but `0xFFFF_FFFF_FFFF_FFFB` as isize).
///
/// This function iterates through variants to find which one has the given discriminant.
fn discriminant_to_variant_index(
    rust_ty: &rustc_public::ty::Ty,
    tag_value: u128,
    mask: u128,
) -> Option<usize> {
    use rustc_public::ty::{RigidTy, TyKind};

    match rust_ty.kind() {
        TyKind::RigidTy(RigidTy::Adt(adt_def, _)) => {
            for (idx, _variant_def) in adt_def.variants().iter().enumerate() {
                let variant_idx = rustc_public::ty::VariantIdx::to_val(idx);
                let discr = adt_def.discriminant_for_variant(variant_idx);
                if discr.val & mask == tag_value & mask {
                    return Some(idx);
                }
            }
            None
        }
        _ => None,
    }
}

/// Extract enum discriminant from a MirConst using proper rustc_public API.
///
/// This function properly extracts the discriminant value from the constant's
/// allocated bytes, avoiding fragile debug string parsing.
///
/// ## How it works
///
/// For enum constants, rustc stores the discriminant in `ConstantKind::Allocated(Allocation)`.
/// The `Allocation.bytes` field contains the raw bytes of the discriminant value.
/// We use `read_uint()` to properly interpret these bytes.
///
/// ## Fallback behavior
///
/// If the proper API extraction fails (e.g., for ZeroSized constants), we fall back
/// to debug string parsing as a last resort, but this should be rare.
pub(crate) fn extract_enum_discriminant(
    mir_const: &rustc_public::ty::MirConst,
    const_str: &str,
) -> usize {
    // Try to extract using proper API first
    match mir_const.kind() {
        ConstantKind::Allocated(alloc) => {
            // Use read_uint() to properly parse the bytes
            if let Ok(val) = alloc.read_uint() {
                return val as usize;
            }
            // If read_uint fails, try raw_bytes
            if let Ok(bytes) = alloc.raw_bytes()
                && !bytes.is_empty()
            {
                // Convert bytes to usize (little-endian)
                let mut value: usize = 0;
                for (i, &byte) in bytes.iter().take(8).enumerate() {
                    value |= (byte as usize) << (i * 8);
                }
                return value;
            }
            // Last resort: bytes field directly
            if !alloc.bytes.is_empty() {
                let mut value: usize = 0;
                for (i, opt_byte) in alloc.bytes.iter().take(8).enumerate() {
                    if let Some(byte) = opt_byte {
                        value |= (*byte as usize) << (i * 8);
                    }
                }
                return value;
            }
            0
        }
        ConstantKind::ZeroSized => {
            // ZeroSized typically means discriminant 0 (e.g., None)
            0
        }
        ConstantKind::Ty(_ty_const) => {
            // TyConst - try to evaluate
            if let Ok(val) = mir_const.eval_target_usize() {
                return val as usize;
            }
            // Fall back to parsing for TyConst
            parse_discriminant_from_debug_string(const_str)
        }
        ConstantKind::Unevaluated(_) | ConstantKind::Param(_) => {
            // These are rare for enum discriminants; fall back to string parsing
            parse_discriminant_from_debug_string(const_str)
        }
    }
}

/// Fallback: parse discriminant from debug string representation.
/// This is a last resort when the proper API doesn't work.
fn parse_discriminant_from_debug_string(const_str: &str) -> usize {
    // Try to extract discriminant from bytes: [Some(N)] format
    if let Some(bytes_start) = const_str.find("bytes: [Some(") {
        let after_prefix = &const_str[bytes_start + 13..]; // skip "bytes: [Some("
        if let Some(end) = after_prefix.find(')') {
            let discr_str = &after_prefix[..end];
            if let Ok(discr) = discr_str.parse::<usize>() {
                return discr;
            }
        }
    }
    // Try variant name patterns
    if const_str.contains("::None") || const_str.ends_with("None") {
        return 0;
    }
    if const_str.contains("::Some") {
        return 1;
    }
    // Default to 0
    0
}

/// Check if a type is a pointer to SharedArray.
fn is_shared_array_pointer(ty: &rustc_public::ty::Ty) -> bool {
    use rustc_public::ty::{RigidTy, TyKind};

    match ty.kind() {
        TyKind::RigidTy(RigidTy::RawPtr(pointee_ty, _)) => {
            // Check if the pointee is SharedArray
            match pointee_ty.kind() {
                TyKind::RigidTy(RigidTy::Adt(adt_def, _)) => {
                    adt_def.trimmed_name() == "SharedArray"
                }
                _ => false,
            }
        }
        _ => false,
    }
}

/// Check if a type is a pointer to Barrier (mbarrier state in shared memory).
fn is_barrier_pointer(ty: &rustc_public::ty::Ty) -> bool {
    use rustc_public::ty::{RigidTy, TyKind};

    match ty.kind() {
        TyKind::RigidTy(RigidTy::RawPtr(pointee_ty, _)) => {
            // Check if the pointee is Barrier
            match pointee_ty.kind() {
                TyKind::RigidTy(RigidTy::Adt(adt_def, _)) => adt_def.trimmed_name() == "Barrier",
                _ => false,
            }
        }
        _ => false,
    }
}

/// Resolve a constant pointer/reference to the Rust static it points at, if any.
///
/// Null pointers and pointers to anonymous memory allocations deliberately return
/// `None`; they should continue through normal constant handling.
fn static_def_from_constant(
    constant: &mir::ConstOperand,
) -> TranslationResult<Option<rustc_public::mir::mono::StaticDef>> {
    use rustc_public::mir::alloc::GlobalAlloc;

    let ConstantKind::Allocated(alloc) = constant.const_.kind() else {
        return Ok(None);
    };
    if alloc.is_null().unwrap_or(false) {
        return Ok(None);
    }

    let Some((_, prov)) = alloc.provenance.ptrs.first() else {
        return Ok(None);
    };

    match GlobalAlloc::from(prov.0) {
        GlobalAlloc::Static(static_def) => Ok(Some(static_def)),
        _ => Ok(None),
    }
}

/// Ordinary device globals are currently emitted as `zeroinitializer`.
/// Reject non-zero initializers until constant-data export is implemented.
fn ensure_zero_initializer(
    static_def: &rustc_public::mir::mono::StaticDef,
    loc: Location,
) -> TranslationResult<()> {
    let alloc = static_def.eval_initializer().map_err(|e| {
        input_error_noloc!(TranslationErr::unsupported(format!(
            "Failed to evaluate initializer for device static {}: {:?}",
            static_def.name(),
            e
        )))
    })?;
    let bytes = alloc.raw_bytes().map_err(|e| {
        input_error_noloc!(TranslationErr::unsupported(format!(
            "Device static {} has unsupported uninitialized bytes: {:?}",
            static_def.name(),
            e
        )))
    })?;

    if bytes.iter().all(|byte| *byte == 0) {
        Ok(())
    } else {
        input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Device static {} has a non-zero initializer; cuda-oxide currently supports zero-initialized device statics",
                static_def.name()
            ))
        )
    }
}

fn static_alignment(
    static_def: &rustc_public::mir::mono::StaticDef,
) -> TranslationResult<Option<u64>> {
    let alloc = static_def.eval_initializer().map_err(|e| {
        input_error_noloc!(TranslationErr::unsupported(format!(
            "Failed to evaluate initializer for device static {}: {:?}",
            static_def.name(),
            e
        )))
    })?;
    Ok((alloc.align > 0).then_some(alloc.align))
}

/// Check if a type is a pointer/reference to a static allocation.
/// Returns `(pointee_ty, is_mutable)` when the type can carry a static address.
use super::values::is_constant_wrapper_type;

fn get_static_pointer_info(ty: &rustc_public::ty::Ty) -> Option<(rustc_public::ty::Ty, bool)> {
    use rustc_public::mir::Mutability;
    use rustc_public::ty::{RigidTy, TyKind};

    match ty.kind() {
        TyKind::RigidTy(RigidTy::RawPtr(pointee_ty, mutability)) => {
            Some((pointee_ty, mutability == Mutability::Mut))
        }
        TyKind::RigidTy(RigidTy::Ref(_, pointee_ty, mutability)) => {
            Some((pointee_ty, mutability == Mutability::Mut))
        }
        _ => None,
    }
}

/// Extract element type, size, and alignment from a pointer to SharedArray<T, N, ALIGN>.
/// Returns (element_type, size, alignment) where alignment is 0 for natural alignment.
fn extract_shared_array_info(
    ctx: &mut Context,
    ty: &rustc_public::ty::Ty,
) -> TranslationResult<(Ptr<pliron::r#type::TypeObj>, usize, usize)> {
    use rustc_public::ty::{GenericArgKind, RigidTy, TyKind};

    /// Parse a const generic value from debug string
    fn parse_const_value(c: &rustc_public::ty::TyConst) -> Option<usize> {
        let const_str = format!("{:?}", c);
        // Parse the bytes from the debug string
        if let Some(bytes_part) = const_str.split("bytes: [").nth(1)
            && let Some(bytes_end) = bytes_part.split(']').next()
        {
            let mut bytes = Vec::new();
            for byte_str in bytes_end.split(',') {
                if bytes.len() >= 8 {
                    break;
                }
                let b_str = byte_str.trim();
                if let Some(num_str) = b_str
                    .strip_prefix("Some(")
                    .and_then(|s| s.strip_suffix(')'))
                    && let Ok(byte) = num_str.parse::<u8>()
                {
                    bytes.push(byte);
                }
            }
            // Convert bytes to usize (little-endian)
            let mut value: usize = 0;
            for (i, byte) in bytes.iter().enumerate() {
                value |= (*byte as usize) << (i * 8);
            }
            return Some(value);
        }
        None
    }

    match ty.kind() {
        TyKind::RigidTy(RigidTy::RawPtr(pointee_ty, _)) => {
            match pointee_ty.kind() {
                TyKind::RigidTy(RigidTy::Adt(adt_def, substs)) => {
                    if adt_def.trimmed_name() != "SharedArray" {
                        return input_err_noloc!(TranslationErr::unsupported(format!(
                            "Expected SharedArray, got {}",
                            adt_def.trimmed_name()
                        )));
                    }

                    let generic_args = &substs.0;

                    // Find the element type (first Type arg)
                    let elem_ty = generic_args
                        .iter()
                        .find_map(|arg| match arg {
                            GenericArgKind::Type(t) => Some(t),
                            _ => None,
                        })
                        .ok_or_else(|| {
                            input_error_noloc!(TranslationErr::unsupported(
                                "SharedArray missing element type"
                            ))
                        })?;

                    // Collect all const generic arguments (N, then ALIGN)
                    let const_values: Vec<usize> = generic_args
                        .iter()
                        .filter_map(|arg| match arg {
                            GenericArgKind::Const(c) => parse_const_value(c),
                            _ => None,
                        })
                        .collect();

                    // First const is N (size), required
                    let size = *const_values.first().ok_or_else(|| {
                        input_error_noloc!(TranslationErr::unsupported(
                            "SharedArray missing size const"
                        ))
                    })?;

                    // Second const is ALIGN (alignment), optional, defaults to 0
                    let alignment = const_values.get(1).copied().unwrap_or(0);

                    let translated_elem_ty = types::translate_type(ctx, elem_ty)?;
                    Ok((translated_elem_ty, size, alignment))
                }
                _ => input_err_noloc!(TranslationErr::unsupported(
                    "Expected ADT type for SharedArray"
                )),
            }
        }
        _ => input_err_noloc!(TranslationErr::unsupported("Expected raw pointer type")),
    }
}

/// Create a placeholder ZST aggregate (struct / tuple) value.
///
/// Used for locals whose Rust type is zero-sized: these get no alloca slot
/// (the alloca model skips ZST locals), yet they may still flow through the
/// translator as SSA values (e.g. unit-type temporaries, closure-capture
/// ZSTs). We synthesise an empty aggregate on demand so that every read of
/// a ZST local produces a usable `Value`.
///
/// The caller is responsible for inserting the returned op into a block.
fn create_zst_aggregate(
    ctx: &mut Context,
    ty_ptr: Ptr<pliron::r#type::TypeObj>,
    loc: Location,
) -> Ptr<Operation> {
    use dialect_mir::ops::{MirConstructStructOp, MirConstructTupleOp};
    use dialect_mir::types::MirStructType;

    let op = if ty_ptr.deref(ctx).is::<MirStructType>() {
        Operation::new(
            ctx,
            MirConstructStructOp::get_concrete_op_info(),
            vec![ty_ptr],
            vec![],
            vec![],
            0,
        )
    } else {
        Operation::new(
            ctx,
            MirConstructTupleOp::get_concrete_op_info(),
            vec![ty_ptr],
            vec![],
            vec![],
            0,
        )
    };
    op.deref_mut(ctx).set_loc(loc);
    op
}

/// Create a placeholder `MirConstructEnumOp` for a ghost local.
///
/// Ghost locals are MIR locals that are referenced but never assigned — e.g.
/// rustc optimised away their definition. When translation encounters one we
/// synthesise a variant-0 enum value with no fields -- the moral equivalent
/// of LLVM `undef` for an enum.
///
/// Typical trigger: `Option<Infallible>` which is always `None` (variant 0,
/// no payload) after MIR optimisations.
///
/// The returned operation is **not** inserted into any block; the caller must
/// link it via `insert_after` / `insert_at_front`.
fn create_ghost_enum_default(
    ctx: &mut Context,
    ty_ptr: Ptr<pliron::r#type::TypeObj>,
    loc: Location,
) -> Ptr<Operation> {
    use dialect_mir::ops::MirConstructEnumOp;
    let op = Operation::new(
        ctx,
        MirConstructEnumOp::get_concrete_op_info(),
        vec![ty_ptr],
        vec![],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc);
    MirConstructEnumOp::new(op)
        .set_attr_construct_enum_variant_index(ctx, dialect_mir::attributes::VariantIndexAttr(0));
    op
}

// (The hand-rolled niche-attribute writer that lived here was replaced
// by `MirCastOp::set_attr_niche_encoding(...)`, generated from the typed
// `NicheEncodingAttr` slot declared on the op.)
