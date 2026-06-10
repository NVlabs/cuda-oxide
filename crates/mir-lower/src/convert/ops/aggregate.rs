/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Aggregate operation conversion: `dialect-mir` → LLVM dialect.
//!
//! Converts `dialect-mir` aggregate operations (structs, tuples, enums) to
//! their LLVM dialect equivalents.
//!
//! # Operations
//!
//! | MIR Operation            | LLVM Operation(s)                    | Description            |
//! |--------------------------|--------------------------------------|------------------------|
//! | `mir.extract_field`      | `llvm.extractvalue`                  | Get struct/tuple field |
//! | `mir.insert_field`       | `llvm.insertvalue`                   | Set struct/tuple field |
//! | `mir.construct_struct`   | `llvm.undef` + `llvm.insertvalue`    | Build struct           |
//! | `mir.construct_tuple`    | `llvm.undef` + `llvm.insertvalue`    | Build tuple            |
//! | `mir.construct_enum`     | `llvm.undef` + `llvm.insertvalue`    | Build enum             |
//! | `mir.get_discriminant`   | `llvm.extractvalue`                  | Get enum tag           |
//! | `mir.enum_payload`       | `llvm.extractvalue`                  | Get enum payload       |
//!
//! # Enum Representation
//!
//! Enums are represented as `{ discriminant, field0, field1, ... }` structs where
//! fields from all variants are flattened into a single struct.
//!
//! The discriminant slot (field 0) holds the variant's DECLARED
//! discriminant value, not its variant index. For `core::cmp::Ordering`
//! that means `Less` stores -1 (i8 bit pattern 255), `Equal` 0,
//! `Greater` 1; a variant-index tag would make `Less` match the `Equal`
//! arm (issue #146). The value-to-store comes from
//! `MirEnumType::variant_discriminants`, which the importer fills from
//! `rustc`'s `discriminant_for_variant`.

use crate::convert::types::{
    StructLayoutInfo, StructSlotMap, build_struct_slot_map, convert_type, make_slice_struct,
};
use dialect_mir::ops::{
    MirConstructEnumOp, MirEnumPayloadOp, MirExtractFieldOp, MirFieldAddrOp, MirInsertFieldOp,
};
use dialect_mir::types::{
    MirArrayType, MirDisjointSliceType, MirEnumType, MirPtrType, MirSliceType, MirStructType,
    MirTupleType,
};
use llvm_export::ops as llvm;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;
use pliron::result::Result;
use pliron::r#type::Typed;
use pliron::utils::apint::APInt;
use pliron::value::Value;
use std::num::NonZeroUsize;

fn anyhow_to_pliron(e: anyhow::Error) -> pliron::result::Error {
    pliron::input_error_noloc!("{e}")
}

/// How the MIR-level field indices of an aggregate operand map onto the
/// lowered LLVM aggregate.
enum AggregateSlots {
    /// Lowered from a `MirStructType`/`MirTupleType`: use the slot map the
    /// type converter built (accounts for reordering, `[N x i8]` padding
    /// slots and stripped ZST fields).
    Mapped(StructSlotMap),
    /// The MIR index is already the final LLVM index. Sound only for
    /// aggregates whose lowered layout is index-preserving by construction:
    /// arrays and slice fat pointers (`{ ptr, i64 }`).
    Identity,
}

/// Resolve how field indices of `aggregate` map onto its lowered type.
///
/// Recover-or-error (issue #128): when the operand has no recorded
/// `MirStructType`/`MirTupleType` conversion history, identity indexing is
/// only sound for aggregates the converter lowers without reordering,
/// padding, or ZST stripping: arrays and slice fat pointers. Anything
/// else is a lowering bug; guessing identity there silently reads or
/// writes the wrong field, so we error out loudly instead.
fn resolve_aggregate_slots(
    ctx: &mut Context,
    operands_info: &OperandsInfo,
    aggregate: Value,
) -> Result<AggregateSlots> {
    let layout = operands_info
        .lookup_most_recent_of_type::<MirStructType>(ctx, aggregate)
        .map(|struct_ref| StructLayoutInfo::of_struct(&struct_ref))
        .or_else(|| {
            operands_info
                .lookup_most_recent_of_type::<MirTupleType>(ctx, aggregate)
                .map(|tuple_ref| StructLayoutInfo::of_tuple(&tuple_ref))
        });

    if let Some(layout) = layout {
        let map = build_struct_slot_map(ctx, &layout).map_err(anyhow_to_pliron)?;
        return Ok(AggregateSlots::Mapped(map));
    }

    // Arrays keep their element indices: `[N x T]` has no reorder, no
    // padding, no ZST stripping.
    let is_array_history = operands_info
        .lookup_most_recent_of_type::<MirArrayType>(ctx, aggregate)
        .is_some();
    // Slices lower to the `{ ptr, i64 }` fat pointer, where index 0 = ptr
    // and index 1 = len by construction.
    let is_slice_history = operands_info
        .lookup_most_recent_of_type::<MirSliceType>(ctx, aggregate)
        .is_some()
        || operands_info
            .lookup_most_recent_of_type::<MirDisjointSliceType>(ctx, aggregate)
            .is_some();
    if is_array_history || is_slice_history {
        return Ok(AggregateSlots::Identity);
    }

    // No conversion history at all (e.g. a slice reconstructed in the entry
    // prologue, which is born as an LLVM struct). Identity is still fine if
    // the current type is the fat-pointer struct or an LLVM array.
    let aggregate_ty = aggregate.get_type(ctx);
    let slice_struct_ty = make_slice_struct(ctx);
    let is_llvm_array = aggregate_ty
        .deref(ctx)
        .is::<llvm_export::types::ArrayType>();
    if aggregate_ty == slice_struct_ty || is_llvm_array {
        return Ok(AggregateSlots::Identity);
    }

    let ty_disp = aggregate_ty.deref(ctx).disp(ctx).to_string();
    pliron::input_err_noloc!(
        "Cannot map field indices for aggregate of type {ty_disp}: no struct/tuple \
         conversion history was recorded for this operand, and identity indexing is \
         only sound for arrays and slice fat pointers. Refusing to guess a field \
         mapping (issue #128)."
    )
}

/// Convert `mir.extract_field` to `llvm.extractvalue`.
///
/// Handles scalar-lowered newtype case: if the operand is a scalar (e.g., `ThreadIndex`),
/// no extraction is needed.
///
/// The declaration-order field index is mapped to the LLVM slot via
/// [`resolve_aggregate_slots`], which shares the type converter's view of
/// the struct (reorder, `[N x i8]` padding slots, stripped ZSTs). If
/// extracting a ZST field, we return undef of its (empty) type.
pub(crate) fn convert_extract_field(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let aggregate = op.deref(ctx).get_operand(0);

    let is_scalar = aggregate
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some();

    if is_scalar {
        rewriter.replace_operation_with_values(ctx, op, vec![aggregate]);
        return Ok(());
    }

    let extract_op = MirExtractFieldOp::new(op);
    let decl_index = match extract_op.get_attr_index(ctx) {
        Some(attr) => attr.0 as usize,
        None => return pliron::input_err_noloc!("Missing index attribute on extract_field"),
    };

    let llvm_index = match resolve_aggregate_slots(ctx, operands_info, aggregate)? {
        AggregateSlots::Mapped(map) => match map.decl_to_llvm.get(decl_index) {
            Some(Some(slot)) => *slot,
            Some(None) => {
                // ZST field: stripped from the LLVM struct, so there is
                // nothing to extract. Materialize undef of its empty type.
                let zst_ty = map.field_llvm_types[decl_index];
                let undef_op = llvm::UndefOp::new(ctx, zst_ty);
                rewriter.insert_operation(ctx, undef_op.get_operation());
                rewriter.replace_operation(ctx, op, undef_op.get_operation());
                return Ok(());
            }
            None => {
                return pliron::input_err_noloc!(
                    "extract_field index {} out of bounds for aggregate with {} fields",
                    decl_index,
                    map.decl_to_llvm.len()
                );
            }
        },
        AggregateSlots::Identity => decl_index as u32,
    };

    let llvm_extract = llvm::ExtractValueOp::new(ctx, aggregate, vec![llvm_index])?;
    rewriter.insert_operation(ctx, llvm_extract.get_operation());
    rewriter.replace_operation(ctx, op, llvm_extract.get_operation());

    Ok(())
}

/// Convert `mir.insert_field` to `llvm.insertvalue`.
///
/// Operands: `[aggregate, new_value]`
/// Returns a new aggregate with the field at `insert_index` replaced.
///
/// The declaration-order field index is mapped to the LLVM slot via
/// [`resolve_aggregate_slots`] (arrays keep their element index). If
/// inserting a ZST field, we return the original aggregate unchanged.
pub(crate) fn convert_insert_field(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let aggregate = op.deref(ctx).get_operand(0);
    let new_value = op.deref(ctx).get_operand(1);

    let insert_op = MirInsertFieldOp::new(op);
    let decl_index = match insert_op.get_attr_insert_index(ctx) {
        Some(attr) => attr.0 as usize,
        None => return pliron::input_err_noloc!("Missing insert_index attribute on insert_field"),
    };

    let llvm_index = match resolve_aggregate_slots(ctx, operands_info, aggregate)? {
        AggregateSlots::Mapped(map) => match map.decl_to_llvm.get(decl_index) {
            Some(Some(slot)) => *slot,
            Some(None) => {
                // ZST field: stripped from the LLVM struct, so inserting
                // into it is a no-op. Forward the aggregate unchanged.
                rewriter.replace_operation_with_values(ctx, op, vec![aggregate]);
                return Ok(());
            }
            None => {
                return pliron::input_err_noloc!(
                    "insert_field index {} out of bounds for aggregate with {} fields",
                    decl_index,
                    map.decl_to_llvm.len()
                );
            }
        },
        AggregateSlots::Identity => decl_index as u32,
    };

    let llvm_insert = llvm::InsertValueOp::new(ctx, aggregate, new_value, vec![llvm_index]);
    rewriter.insert_operation(ctx, llvm_insert.get_operation());
    rewriter.replace_operation(ctx, op, llvm_insert.get_operation());

    Ok(())
}

/// Convert `mir.construct_struct` to a chain of `llvm.insertvalue` operations.
///
/// Builds a struct by:
/// 1. Creating an `undef` value of the lowered struct type
/// 2. Inserting each operand at the LLVM slot its field landed in
///
/// Operand order matches field order in the struct type (declaration order).
/// The LLVM struct type and the slot of each field both come from
/// [`build_struct_slot_map`], so the insert indices skip `[N x i8]` padding
/// slots exactly the way the type converter laid them out. ZST fields
/// (e.g. PhantomData) have no slot and are skipped.
pub(crate) fn convert_construct_struct(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let (result_ty, operands) = {
        let mir_op = op.deref(ctx);
        let result_ty = mir_op.get_result(0).get_type(ctx);
        let operands: Vec<_> = mir_op.operands().collect();
        (result_ty, operands)
    };

    let layout = {
        let ty_ref = result_ty.deref(ctx);
        match ty_ref.downcast_ref::<MirStructType>() {
            Some(s) => StructLayoutInfo::of_struct(s),
            None => {
                return pliron::input_err_noloc!(
                    "MirConstructStructOp result type must be MirStructType"
                );
            }
        }
    };

    if operands.len() != layout.field_types.len() {
        return pliron::input_err_noloc!(
            "construct_struct has {} operands for a struct with {} fields",
            operands.len(),
            layout.field_types.len()
        );
    }

    let map = build_struct_slot_map(ctx, &layout).map_err(anyhow_to_pliron)?;

    let undef_op = llvm::UndefOp::new(ctx, map.llvm_struct_ty);
    rewriter.insert_operation(ctx, undef_op.get_operation());
    let mut current_struct = undef_op.get_operation().deref(ctx).get_result(0);

    let mut last_insert: Option<Ptr<Operation>> = None;
    // Walk in memory order so the insertvalue chain ascends slot indices.
    for &decl_idx in &layout.mem_to_decl {
        let Some(slot) = map.decl_to_llvm[decl_idx] else {
            continue; // ZST field: no slot in the LLVM struct.
        };

        let insert_op =
            llvm::InsertValueOp::new(ctx, current_struct, operands[decl_idx], vec![slot]);
        rewriter.insert_operation(ctx, insert_op.get_operation());
        current_struct = insert_op.get_operation().deref(ctx).get_result(0);
        last_insert = Some(insert_op.get_operation());
    }

    match last_insert {
        Some(last_op) => rewriter.replace_operation(ctx, op, last_op),
        None => rewriter.replace_operation(ctx, op, undef_op.get_operation()),
    }

    Ok(())
}

/// Convert `mir.construct_tuple` to a chain of `llvm.insertvalue` operations.
///
/// Tuples are represented as LLVM structs. Same construction pattern as
/// structs, and like structs the element slots come from
/// [`build_struct_slot_map`] (identity order, no padding; ZST elements are
/// stripped and skipped).
pub(crate) fn convert_construct_tuple(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let (result_ty, operands) = {
        let mir_op = op.deref(ctx);
        let result_ty = mir_op.get_result(0).get_type(ctx);
        let operands: Vec<_> = mir_op.operands().collect();
        (result_ty, operands)
    };

    let layout = {
        let ty_ref = result_ty.deref(ctx);
        match ty_ref.downcast_ref::<MirTupleType>() {
            Some(t) => StructLayoutInfo::of_tuple(t),
            None => {
                return pliron::input_err_noloc!(
                    "MirConstructTupleOp result type must be MirTupleType"
                );
            }
        }
    };

    if operands.len() != layout.field_types.len() {
        return pliron::input_err_noloc!(
            "construct_tuple has {} operands for a tuple with {} elements",
            operands.len(),
            layout.field_types.len()
        );
    }

    let map = build_struct_slot_map(ctx, &layout).map_err(anyhow_to_pliron)?;

    let undef_op = llvm::UndefOp::new(ctx, map.llvm_struct_ty);
    rewriter.insert_operation(ctx, undef_op.get_operation());
    let mut current_tuple = undef_op.get_operation().deref(ctx).get_result(0);

    let mut last_insert: Option<Ptr<Operation>> = None;
    for (mir_idx, operand) in operands.iter().enumerate() {
        let Some(slot) = map.decl_to_llvm[mir_idx] else {
            continue; // ZST element: no slot in the LLVM struct.
        };

        let insert_op = llvm::InsertValueOp::new(ctx, current_tuple, *operand, vec![slot]);
        rewriter.insert_operation(ctx, insert_op.get_operation());
        current_tuple = insert_op.get_operation().deref(ctx).get_result(0);
        last_insert = Some(insert_op.get_operation());
    }

    match last_insert {
        Some(last_op) => rewriter.replace_operation(ctx, op, last_op),
        None => rewriter.replace_operation(ctx, op, undef_op.get_operation()),
    }

    Ok(())
}

/// Convert `mir.construct_array` to a chain of `llvm.insertvalue` operations.
///
/// Arrays are represented as LLVM arrays. Same construction pattern as structs:
/// 1. Create `undef` of the array type
/// 2. Insert each element at its index
pub(crate) fn convert_construct_array(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let (result_ty, operands) = {
        let mir_op = op.deref(ctx);
        let result_ty = mir_op.get_result(0).get_type(ctx);
        let operands: Vec<_> = mir_op.operands().collect();
        (result_ty, operands)
    };

    let (element_ty, array_size) = {
        let ty_ref = result_ty.deref(ctx);
        match ty_ref.downcast_ref::<MirArrayType>() {
            Some(a) => (a.element_type(), a.size()),
            None => {
                return pliron::input_err_noloc!(
                    "MirConstructArrayOp result type must be MirArrayType"
                );
            }
        }
    };

    let llvm_element_ty = convert_type(ctx, element_ty).map_err(anyhow_to_pliron)?;
    let llvm_array_ty = llvm_export::types::ArrayType::get(ctx, llvm_element_ty, array_size);

    let undef_op = llvm::UndefOp::new(ctx, llvm_array_ty.into());
    rewriter.insert_operation(ctx, undef_op.get_operation());
    let mut current_array = undef_op.get_operation().deref(ctx).get_result(0);

    let mut last_insert: Option<Ptr<Operation>> = None;
    for (i, operand) in operands.iter().enumerate() {
        let insert_op = llvm::InsertValueOp::new(ctx, current_array, *operand, vec![i as u32]);
        rewriter.insert_operation(ctx, insert_op.get_operation());
        current_array = insert_op.get_operation().deref(ctx).get_result(0);
        last_insert = Some(insert_op.get_operation());
    }

    match last_insert {
        Some(last_op) => rewriter.replace_operation(ctx, op, last_op),
        None => rewriter.replace_operation(ctx, op, undef_op.get_operation()),
    }

    Ok(())
}

/// Convert `mir.extract_array_element` to LLVM alloca+store+GEP+load sequence.
///
/// Since LLVM's `extractvalue` only supports constant indices, we need to:
/// 1. Allocate stack space for the array
/// 2. Store the array value to the stack
/// 3. GEP to compute the element address
/// 4. Load the element
pub(crate) fn convert_extract_array_element(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let array_val = op.deref(ctx).get_operand(0);
    let index_val = op.deref(ctx).get_operand(1);

    let (element_ty, array_size) = {
        match operands_info.lookup_most_recent_of_type::<MirArrayType>(ctx, array_val) {
            Some(r) => (r.element_type(), r.size()),
            None => return pliron::input_err_noloc!("Expected MirArrayType"),
        }
    };

    let llvm_element_ty = convert_type(ctx, element_ty).map_err(anyhow_to_pliron)?;
    let llvm_array_ty = llvm_export::types::ArrayType::get(ctx, llvm_element_ty, array_size);

    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let one_val = {
        let one_apint = APInt::from_i64(1, NonZeroUsize::new(64).unwrap());
        let one_attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, one_apint);
        let const_op = llvm::ConstantOp::new(ctx, one_attr.into());
        rewriter.insert_operation(ctx, const_op.get_operation());
        const_op.get_operation().deref(ctx).get_result(0)
    };

    let alloca_op = llvm::AllocaOp::new(ctx, llvm_array_ty.into(), one_val);
    rewriter.insert_operation(ctx, alloca_op.get_operation());
    let array_ptr = alloca_op.get_operation().deref(ctx).get_result(0);

    let store_op = llvm::StoreOp::new(ctx, array_val, array_ptr);
    rewriter.insert_operation(ctx, store_op.get_operation());

    use llvm_export::ops::GepIndex;
    let gep_indices = vec![GepIndex::Constant(0), GepIndex::Value(index_val)];
    let gep_op = llvm::GetElementPtrOp::new(ctx, array_ptr, gep_indices, llvm_array_ty.into());
    rewriter.insert_operation(ctx, gep_op.get_operation());
    let element_ptr = gep_op.get_operation().deref(ctx).get_result(0);

    let load_op = llvm::LoadOp::new(ctx, element_ptr, llvm_element_ty);
    rewriter.insert_operation(ctx, load_op.get_operation());
    rewriter.replace_operation(ctx, op, load_op.get_operation());

    Ok(())
}

/// Convert `mir.construct_enum` to LLVM struct operations.
///
/// Enums are `{discriminant, payload...}` structs.
pub(crate) fn convert_construct_enum(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let (result_ty, operands, variant_index) = {
        let mir_op = op.deref(ctx);
        let result_ty = mir_op.get_result(0).get_type(ctx);
        let operands: Vec<_> = mir_op.operands().collect();

        let enum_op = MirConstructEnumOp::new(op);
        let variant_index = enum_op
            .get_attr_construct_enum_variant_index(ctx)
            .map(|attr| attr.0 as usize)
            .unwrap_or(0);

        (result_ty, operands, variant_index)
    };

    let (variant_discriminants, variant_field_counts): (Vec<u64>, Vec<u32>) = {
        let ty_ref = result_ty.deref(ctx);
        match ty_ref.downcast_ref::<MirEnumType>() {
            Some(e) => (
                e.variant_discriminants.clone(),
                e.variant_field_counts.clone(),
            ),
            None => {
                return pliron::input_err_noloc!(
                    "MirConstructEnumOp result type must be MirEnumType"
                );
            }
        }
    };

    // Build the value with the CONVERTED enum type (the same one the type
    // converter produces for block args, loads, allocas, ...), so constructed
    // values and the converted type cannot diverge. The converter returns
    // `{tag, fields..., [pad]?}`; any trailing pad field is simply never
    // written. Field 0 is always the discriminant.
    let llvm_struct_ty = convert_type(ctx, result_ty).map_err(anyhow_to_pliron)?;
    let llvm_discriminant_ty = {
        let ty_ref = llvm_struct_ty.deref(ctx);
        let struct_ty = ty_ref
            .downcast_ref::<llvm_export::types::StructType>()
            .ok_or_else(|| {
                pliron::input_error_noloc!("converted MirEnumType must be an LLVM struct")
            })?;
        struct_ty.fields().next().ok_or_else(|| {
            pliron::input_error_noloc!("converted MirEnumType struct must have a tag field")
        })?
    };

    let undef_op = llvm::UndefOp::new(ctx, llvm_struct_ty);
    rewriter.insert_operation(ctx, undef_op.get_operation());
    let mut current_struct = undef_op.get_operation().deref(ctx).get_result(0);

    // The tag width comes from the enum's discriminant type; assuming a
    // width (the old `unwrap_or(8)`) would silently store a wrong-sized
    // tag for any enum whose discriminant is not an integer type.
    let discr_width = match llvm_discriminant_ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|t| t.width())
    {
        Some(w) => w,
        None => {
            return pliron::input_err_noloc!(
                "MirConstructEnumOp discriminant type must be an integer type"
            );
        }
    };
    // The stored tag is the variant's declared discriminant VALUE (not the
    // variant index). A variant index without a discriminant entry means
    // the MirEnumType is malformed; falling back to the index would
    // silently resurrect the issue #146 miscompile.
    let discr_value = match variant_discriminants.get(variant_index).copied() {
        Some(v) => v,
        None => {
            return pliron::input_err_noloc!(
                "MirConstructEnumOp variant index {} has no discriminant ({} discriminants recorded)",
                variant_index,
                variant_discriminants.len()
            );
        }
    };
    let discr_apint = APInt::from_u64(
        discr_value,
        NonZeroUsize::new(discr_width as usize).unwrap(),
    );
    let llvm_discr_ty = IntegerType::get(ctx, discr_width, Signedness::Signless);
    let discr_attr = pliron::builtin::attributes::IntegerAttr::new(llvm_discr_ty, discr_apint);
    let discr_const = llvm::ConstantOp::new(ctx, discr_attr.into());
    rewriter.insert_operation(ctx, discr_const.get_operation());
    let discr_val = discr_const.get_operation().deref(ctx).get_result(0);

    let insert_discr = llvm::InsertValueOp::new(ctx, current_struct, discr_val, vec![0]);
    rewriter.insert_operation(ctx, insert_discr.get_operation());
    current_struct = insert_discr.get_operation().deref(ctx).get_result(0);

    let field_offset: usize = variant_field_counts
        .iter()
        .take(variant_index)
        .map(|&c| c as usize)
        .sum();

    let mut last_op = insert_discr.get_operation();
    for (i, operand) in operands.into_iter().enumerate() {
        let llvm_idx = 1 + field_offset + i;
        let insert_op =
            llvm::InsertValueOp::new(ctx, current_struct, operand, vec![llvm_idx as u32]);
        rewriter.insert_operation(ctx, insert_op.get_operation());
        current_struct = insert_op.get_operation().deref(ctx).get_result(0);
        last_op = insert_op.get_operation();
    }

    rewriter.replace_operation(ctx, op, last_op);

    Ok(())
}

/// Convert `mir.get_discriminant` to `llvm.extractvalue`.
///
/// Extracts the discriminant (tag) from an enum value. The discriminant
/// is always at index 0 in the LLVM struct representation, and it holds
/// the variant's DECLARED discriminant value (what `construct_enum`
/// stored), so downstream `SwitchInt` comparisons match against
/// discriminant values, never variant indices.
pub(crate) fn convert_get_discriminant(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let enum_val = match op.deref(ctx).operands().next() {
        Some(v) => v,
        None => return pliron::input_err_noloc!("MirGetDiscriminantOp requires an operand"),
    };

    let extract_op = llvm::ExtractValueOp::new(ctx, enum_val, vec![0])?;
    rewriter.insert_operation(ctx, extract_op.get_operation());
    rewriter.replace_operation(ctx, op, extract_op.get_operation());

    Ok(())
}

/// Convert `mir.enum_payload` to `llvm.extractvalue`.
///
/// Extracts a field from an enum variant's payload. The LLVM struct index
/// is computed as: `1 + sum(field_counts[0..variant]) + field_index`
/// where 1 accounts for the discriminant at index 0.
pub(crate) fn convert_enum_payload(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let enum_val = match op.deref(ctx).operands().next() {
        Some(v) => v,
        None => return pliron::input_err_noloc!("MirEnumPayloadOp requires an operand"),
    };

    let payload_op = MirEnumPayloadOp::new(op);
    let variant_index = payload_op
        .get_attr_payload_variant_index(ctx)
        .map(|attr| attr.0 as usize)
        .unwrap_or(0);
    let field_index = payload_op
        .get_attr_payload_field_index(ctx)
        .map(|attr| attr.0 as usize)
        .unwrap_or(0);

    let variant_field_counts = {
        match operands_info.lookup_most_recent_of_type::<MirEnumType>(ctx, enum_val) {
            Some(r) => r.variant_field_counts.clone(),
            None => {
                return pliron::input_err_noloc!(
                    "Expected MirEnumType for enum payload extraction"
                );
            }
        }
    };

    let field_offset: usize = variant_field_counts
        .iter()
        .take(variant_index)
        .map(|&c| c as usize)
        .sum();

    let llvm_idx = 1 + field_offset + field_index;

    let extract_op = llvm::ExtractValueOp::new(ctx, enum_val, vec![llvm_idx as u32])?;
    rewriter.insert_operation(ctx, extract_op.get_operation());
    rewriter.replace_operation(ctx, op, extract_op.get_operation());

    Ok(())
}

// ============================================================================
// MirFieldAddrOp Conversion
// ============================================================================

/// Convert `mir.field_addr` to `llvm.getelementptr`.
///
/// Computes the address of a struct field using GEP. This is needed when
/// Rust code takes `&mut self.field` — we need the ADDRESS of the field,
/// not a COPY of its value.
///
/// The GEP field index and the struct type it indexes into both come from
/// [`build_struct_slot_map`], so the index accounts for reordering,
/// `[N x i8]` padding slots and stripped ZSTs (ZST-ness is decided on the
/// converted LLVM field type, like the value-level sites). Taking the
/// address of a ZST field forwards the struct pointer itself.
pub(crate) fn convert_field_addr(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let ptr_operand = op.deref(ctx).get_operand(0);

    let field_addr_op = MirFieldAddrOp::new(op);
    let field_index = match field_addr_op.get_attr_field_index(ctx) {
        Some(attr) => attr.0 as usize,
        None => return pliron::input_err_noloc!("MirFieldAddrOp missing field_index attribute"),
    };

    let layout = {
        let mir_ptr_pointee =
            match operands_info.lookup_most_recent_of_type::<MirPtrType>(ctx, ptr_operand) {
                Some(r) => r.pointee,
                None => {
                    return pliron::input_err_noloc!("MirFieldAddrOp operand must be pointer type");
                }
            };

        let pointee_ref = mir_ptr_pointee.deref(ctx);
        match pointee_ref.downcast_ref::<MirStructType>() {
            Some(struct_ty) => StructLayoutInfo::of_struct(struct_ty),
            None => {
                return pliron::input_err_noloc!(
                    "MirFieldAddrOp pointer must point to struct type"
                );
            }
        }
    };

    let map = build_struct_slot_map(ctx, &layout).map_err(anyhow_to_pliron)?;

    let slot = match map.decl_to_llvm.get(field_index) {
        Some(Some(slot)) => *slot,
        Some(None) => {
            // ZST field: it has no storage; the struct address stands in
            // for the field address.
            rewriter.replace_operation_with_values(ctx, op, vec![ptr_operand]);
            return Ok(());
        }
        None => {
            return pliron::input_err_noloc!(
                "field_addr index {} out of bounds for struct with {} fields",
                field_index,
                map.decl_to_llvm.len()
            );
        }
    };

    use llvm_export::ops::GepIndex;
    let gep_indices = vec![GepIndex::Constant(0), GepIndex::Constant(slot)];

    let gep_op = llvm::GetElementPtrOp::new(ctx, ptr_operand, gep_indices, map.llvm_struct_ty);
    rewriter.insert_operation(ctx, gep_op.get_operation());
    rewriter.replace_operation(ctx, op, gep_op.get_operation());

    Ok(())
}

// ============================================================================
// MirArrayElementAddrOp Conversion
// ============================================================================

/// Convert `mir.array_element_addr` to `llvm.getelementptr`.
///
/// This computes the address of an array element using a runtime index.
/// The operation is: `&arr[i]` → `getelementptr [N x T], ptr %arr_ptr, i64 0, i64 %i`
pub(crate) fn convert_array_element_addr(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let arr_ptr = op.deref(ctx).get_operand(0);
    let index = op.deref(ctx).get_operand(1);

    let pointee_ty = {
        let mir_ptr_pointee =
            match operands_info.lookup_most_recent_of_type::<MirPtrType>(ctx, arr_ptr) {
                Some(r) => r.pointee,
                None => {
                    return pliron::input_err_noloc!(
                        "MirArrayElementAddrOp operand must be pointer type"
                    );
                }
            };

        let pointee_ref = mir_ptr_pointee.deref(ctx);
        if pointee_ref.downcast_ref::<MirArrayType>().is_none() {
            return pliron::input_err_noloc!(
                "MirArrayElementAddrOp pointer must point to array type"
            );
        }
        mir_ptr_pointee
    };

    let llvm_array_ty = convert_type(ctx, pointee_ty).map_err(anyhow_to_pliron)?;

    use llvm_export::ops::GepIndex;
    let gep_indices = vec![GepIndex::Constant(0), GepIndex::Value(index)];

    let gep_op = llvm::GetElementPtrOp::new(ctx, arr_ptr, gep_indices, llvm_array_ty);
    rewriter.insert_operation(ctx, gep_op.get_operation());
    rewriter.replace_operation(ctx, op, gep_op.get_operation());

    Ok(())
}

#[cfg(test)]
mod tests {
    // TODO: Add unit tests for aggregate conversion
}
