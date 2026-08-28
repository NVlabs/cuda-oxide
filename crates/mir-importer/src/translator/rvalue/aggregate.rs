/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Aggregate construction and field-value helpers.

use super::coerce::cast_to_expected_pointer_type_if_needed;
use super::const_bytes::translate_constant_value_from_bytes;
use super::operand::translate_operand;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::facts;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_mir::ops::{MirInsertFieldOp, MirUndefOp};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use pliron::{input_err, input_error_noloc};
use rustc_public::CrateDef;
use rustc_public::mir;
use rustc_public_bridge::IndexedVal;

/// Build a `DisjointSlice` value from the fields of its MIR aggregate.
///
/// The literal lists `ptr`, `len`, the index space's runtime layout, and the
/// marker fields; the markers are zero-sized and carry nothing, so dropping
/// them leaves the operands `mir.construct_disjoint_slice` takes, in the same
/// order. An index space with no runtime layout (`Index1D`, `Index2D<S>`)
/// stores `()` there, which drops with the markers and leaves the two-word
/// slice.
///
/// Field selection is positional, which the op's verifier then checks against
/// the result type: the data pointer must point to the element type, the
/// length must be an integer, and each remaining operand must match the index
/// space's layout types in order. A reordered or retyped field therefore
/// fails at verification rather than silently writing a row width into the
/// length slot.
pub(super) fn construct_disjoint_slice_aggregate(
    ctx: &mut Context,
    adt_ty: TypeHandle,
    field_values: &[Value],
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Option<Ptr<Operation>>, Value, Option<Ptr<Operation>>)> {
    let (element_type, space_tys) = {
        let ty_obj = adt_ty.deref(ctx);
        let slice_ty = ty_obj
            .downcast_ref::<dialect_mir::types::MirDisjointSliceType>()
            .expect("caller checked the disjoint slice type");
        (slice_ty.element_type(), slice_ty.space_types().to_vec())
    };

    let runtime_fields: Vec<Value> = field_values
        .iter()
        .copied()
        .filter(|value| !types::is_zst_type(ctx, value.get_type(ctx)))
        .collect();

    let expected = 2 + space_tys.len();
    if runtime_fields.len() != expected {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "DisjointSlice aggregate expected {} runtime fields for {}, found {}",
                expected,
                adt_ty.disp(ctx),
                runtime_fields.len()
            ))
        );
    }

    // The data pointer reaches the slice through the generic address space,
    // as the fat-pointer arm does for `*mut [T]`: a value coming from shared
    // memory carries addrspace(3) and would not match the element pointer the
    // verifier expects.
    let expected_ptr_ty: TypeHandle =
        facts::mint_generic_ptr_type(ctx, element_type, facts::abi_disjoint_slice_data_ptr())
            .into();
    let (data_val, current_prev_op) = cast_to_expected_pointer_type_if_needed(
        ctx,
        runtime_fields[0],
        expected_ptr_ty,
        block_ptr,
        prev_op,
        loc.clone(),
    );

    let mut operands = vec![data_val];
    operands.extend_from_slice(&runtime_fields[1..]);

    let op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructDisjointSliceOp::get_concrete_op_info(),
        vec![adt_ty],
        operands,
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc);

    let result = op.deref(ctx).get_result(0);
    Ok((Some(op), result, current_prev_op))
}

/// Translate ADT aggregate operands, synthesizing omitted runtime-ZST fields when
/// MIR carries only the non-ZST runtime operands.
pub(super) fn translate_adt_aggregate_field_values(
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

/// Construct a union by writing the one active field into shared storage.
///
/// MIR supplies exactly one operand plus the declaration index of its active
/// field. Start with undefined union storage and use `mir.insert_field` to
/// write that typed view at byte zero. The union-specific lowering preserves
/// every other byte as undefined; it never invents one independent slot per
/// field.
#[allow(clippy::too_many_arguments)]
pub(super) fn translate_union_aggregate(
    ctx: &mut Context,
    body: &mir::Body,
    adt_def: rustc_public::ty::AdtDef,
    union_ty: TypeHandle,
    active_field_idx: Option<usize>,
    operands: &[mir::Operand],
    value_map: &mut ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Option<Ptr<Operation>>, Value, Option<Ptr<Operation>>)> {
    let active_field_idx = active_field_idx.ok_or_else(|| {
        input_error_noloc!(TranslationErr::unsupported(format!(
            "Union aggregate '{}' did not identify an active field",
            adt_def.trimmed_name()
        )))
    })?;

    if operands.len() != 1 {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Union aggregate '{}' expected exactly one operand for active field {}, found {}",
                adt_def.trimmed_name(),
                active_field_idx,
                operands.len()
            ))
        );
    }

    let (field_count, expected_field_ty) = {
        let ty_ref = union_ty.deref(ctx);
        let union = ty_ref
            .downcast_ref::<dialect_mir::types::MirUnionType>()
            .ok_or_else(|| {
                input_error_noloc!(TranslationErr::unsupported(format!(
                    "Union aggregate '{}' did not translate to MirUnionType",
                    adt_def.trimmed_name()
                )))
            })?;
        (union.field_count(), union.get_field_type(active_field_idx))
    };
    if active_field_idx >= field_count {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "Union aggregate '{}' active field {} is out of bounds for {} fields",
                adt_def.trimmed_name(),
                active_field_idx,
                field_count
            ))
        );
    }
    let expected_field_ty = expected_field_ty.expect("active union field was bounds-checked");

    let (active_value, current_prev_op) = translate_operand(
        ctx,
        body,
        &operands[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let (active_value, current_prev_op) = cast_to_expected_pointer_type_if_needed(
        ctx,
        active_value,
        expected_field_ty,
        block_ptr,
        current_prev_op,
        loc.clone(),
    );

    let undef_op = MirUndefOp::new(ctx, union_ty).get_operation();
    undef_op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(prev) = current_prev_op {
        undef_op.insert_after(ctx, prev);
    } else {
        undef_op.insert_at_front(block_ptr, ctx);
    }
    let undef_value = undef_op.deref(ctx).get_result(0);

    let insert_op = Operation::new(
        ctx,
        MirInsertFieldOp::get_concrete_op_info(),
        vec![union_ty],
        vec![undef_value, active_value],
        vec![],
        0,
    );
    insert_op.deref_mut(ctx).set_loc(loc);
    MirInsertFieldOp::new(insert_op).set_attr_insert_index(
        ctx,
        dialect_mir::attributes::FieldIndexAttr(active_field_idx as u32),
    );
    let result = insert_op.deref(ctx).get_result(0);

    Ok((Some(insert_op), result, Some(undef_op)))
}
