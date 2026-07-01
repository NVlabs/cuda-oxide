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
//! | `mir.construct_slice`    | `llvm.undef` + `llvm.insertvalue`    | Build slice fat ptr    |
//! | `mir.construct_enum`     | `llvm.undef` + `llvm.insertvalue`    | Build enum             |
//! | `mir.get_discriminant`   | `llvm.extractvalue`                  | Get enum tag           |
//! | `mir.set_discriminant`   | `llvm.getelementptr` + `llvm.store`  | Write enum tag         |
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
    EnumSlotMap, StructLayoutInfo, StructSlotMap, build_enum_slot_map, build_struct_slot_map,
    build_union_storage_type, convert_type, is_zero_sized_type, make_slice_struct,
};
use dialect_mir::ops::{
    MirConstructEnumOp, MirEnumPayloadOp, MirExtractFieldOp, MirFieldAddrOp, MirInsertFieldOp,
};
use dialect_mir::types::{
    MirArrayType, MirDisjointSliceType, MirEnumType, MirPtrType, MirSliceType, MirStructType,
    MirTupleType, MirUnionType,
};
use llvm_export::ops as llvm;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};
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

    let extract_op = MirExtractFieldOp::new(op);
    let decl_index = match extract_op.get_attr_index(ctx) {
        Some(attr) => attr.0 as usize,
        None => return pliron::input_err_noloc!("Missing index attribute on extract_field"),
    };

    if operands_info
        .lookup_most_recent_of_type::<MirUnionType>(ctx, aggregate)
        .is_some()
    {
        return convert_extract_union_field(
            ctx,
            rewriter,
            op,
            aggregate,
            decl_index,
            operands_info,
        );
    }

    let is_scalar = aggregate
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some();

    if is_scalar {
        rewriter.replace_operation_with_values(ctx, op, vec![aggregate]);
        return Ok(());
    }

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

    if operands_info
        .lookup_most_recent_of_type::<MirUnionType>(ctx, aggregate)
        .is_some()
    {
        return convert_insert_union_field(
            ctx,
            rewriter,
            op,
            aggregate,
            new_value,
            decl_index,
            operands_info,
        );
    }

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

fn union_type_of_operand(
    ctx: &Context,
    operands_info: &OperandsInfo,
    value: Value,
) -> Result<MirUnionType> {
    operands_info
        .lookup_most_recent_of_type::<MirUnionType>(ctx, value)
        .map(|union_ty| union_ty.clone())
        .ok_or_else(|| {
            pliron::create_error!(
                pliron::location::Location::Unknown,
                pliron::result::ErrorKind::VerificationFailed,
                pliron::result::StringError(
                    "Expected MirUnionType conversion history for union value".to_string()
                )
            )
        })
}

/// Read one typed view of a union's shared bytes.
fn convert_extract_union_field(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    union_value: Value,
    field_index: usize,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let union_ty = union_type_of_operand(ctx, operands_info, union_value)?;
    let Some(field_mir_ty) = union_ty.get_field_type(field_index) else {
        return pliron::input_err_noloc!(
            "union field index {} is out of bounds for `{}`",
            field_index,
            union_ty.name()
        );
    };
    let field_llvm_ty = convert_type(ctx, field_mir_ty).map_err(anyhow_to_pliron)?;
    if is_zero_sized_type(ctx, field_llvm_ty) {
        let undef = llvm::UndefOp::new(ctx, field_llvm_ty);
        rewriter.insert_operation(ctx, undef.get_operation());
        rewriter.replace_operation(ctx, op, undef.get_operation());
        return Ok(());
    }

    let storage_ty = build_union_storage_type(ctx, &union_ty).map_err(anyhow_to_pliron)?;
    let ptr = spill_enum_value(ctx, rewriter, union_value, storage_ty, union_ty.abi_align());
    let load = llvm::LoadOp::new(ctx, ptr, field_llvm_ty);
    llvm_export::ops::set_op_alignment(ctx, load.get_operation(), union_ty.abi_align() as u32);
    rewriter.insert_operation(ctx, load.get_operation());
    rewriter.replace_operation(ctx, op, load.get_operation());
    Ok(())
}

/// Write one typed view at byte zero while preserving the rest of the union.
fn convert_insert_union_field(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    union_value: Value,
    new_value: Value,
    field_index: usize,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let union_ty = union_type_of_operand(ctx, operands_info, union_value)?;
    let Some(field_mir_ty) = union_ty.get_field_type(field_index) else {
        return pliron::input_err_noloc!(
            "union field index {} is out of bounds for `{}`",
            field_index,
            union_ty.name()
        );
    };
    let field_llvm_ty = convert_type(ctx, field_mir_ty).map_err(anyhow_to_pliron)?;
    if is_zero_sized_type(ctx, field_llvm_ty) {
        rewriter.replace_operation_with_values(ctx, op, vec![union_value]);
        return Ok(());
    }

    let storage_ty = build_union_storage_type(ctx, &union_ty).map_err(anyhow_to_pliron)?;
    let ptr = spill_enum_value(ctx, rewriter, union_value, storage_ty, union_ty.abi_align());
    let store = llvm::StoreOp::new(ctx, new_value, ptr);
    llvm_export::ops::set_op_alignment(ctx, store.get_operation(), union_ty.abi_align() as u32);
    rewriter.insert_operation(ctx, store.get_operation());

    let load = llvm::LoadOp::new(ctx, ptr, storage_ty);
    llvm_export::ops::set_op_alignment(ctx, load.get_operation(), union_ty.abi_align() as u32);
    rewriter.insert_operation(ctx, load.get_operation());
    rewriter.replace_operation(ctx, op, load.get_operation());
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

/// Convert `mir.construct_slice` to `llvm.undef` + two `llvm.insertvalue`s.
///
/// `MirSliceType` lowers to the `{ ptr, i64 }` fat-pointer struct, where
/// field 0 is the data pointer and field 1 is the element count by
/// construction (the same layout the entry prologue's `reconstruct_slice`
/// and the Unsize cast path build).
pub(crate) fn convert_construct_slice(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let (result_ty, data_val, len_val) = {
        let mir_op = op.deref(ctx);
        (
            mir_op.get_result(0).get_type(ctx),
            mir_op.get_operand(0),
            mir_op.get_operand(1),
        )
    };

    if !result_ty.deref(ctx).is::<MirSliceType>() {
        return pliron::input_err_noloc!("MirConstructSliceOp result type must be MirSliceType");
    }

    let slice_struct_ty = make_slice_struct(ctx);

    let undef_op = llvm::UndefOp::new(ctx, slice_struct_ty);
    rewriter.insert_operation(ctx, undef_op.get_operation());
    let undef_val = undef_op.get_operation().deref(ctx).get_result(0);

    let insert_ptr = llvm::InsertValueOp::new(ctx, undef_val, data_val, vec![0]);
    rewriter.insert_operation(ctx, insert_ptr.get_operation());
    let with_ptr = insert_ptr.get_operation().deref(ctx).get_result(0);

    let insert_len = llvm::InsertValueOp::new(ctx, with_ptr, len_val, vec![1]);
    rewriter.insert_operation(ctx, insert_len.get_operation());

    rewriter.replace_operation(ctx, op, insert_len.get_operation());

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

/// Copy an enum value into a fresh stack slot and return the pointer.
///
/// This is how we reach a payload field that has no struct slot of its
/// own (its bytes are shared with a different-typed field of another
/// variant): once the value sits in memory, a byte-precise pointer can
/// read or write any part of it, no struct field needed.
///
/// The slot is marked with the enum's real (rustc) alignment. The struct
/// type alone can look under-aligned: `{ i8, [7 x i8] }` says "align 1"
/// to LLVM, while Rust may require 8.
///
/// The alloca lands at the use site, same as
/// [`convert_extract_array_element`]; the standard `opt -O2` run (SROA)
/// removes it again. Hoisting these into the function's entry block is a
/// known follow-up for the unoptimized (`CUDA_OXIDE_NO_OPT=1`) path.
fn spill_enum_value(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    enum_val: Value,
    llvm_struct_ty: TypeHandle,
    abi_align: u64,
) -> Value {
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let one_apint = APInt::from_i64(1, NonZeroUsize::new(64).unwrap());
    let one_attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, one_apint);
    let one_const = llvm::ConstantOp::new(ctx, one_attr.into());
    rewriter.insert_operation(ctx, one_const.get_operation());
    let one_val = one_const.get_operation().deref(ctx).get_result(0);

    let alloca_op = llvm::AllocaOp::new(ctx, llvm_struct_ty, one_val);
    rewriter.insert_operation(ctx, alloca_op.get_operation());
    if abi_align > 0 {
        llvm_export::ops::set_op_alignment(ctx, alloca_op.get_operation(), abi_align as u32);
    }
    let slot_ptr = alloca_op.get_operation().deref(ctx).get_result(0);

    let store_op = llvm::StoreOp::new(ctx, enum_val, slot_ptr);
    rewriter.insert_operation(ctx, store_op.get_operation());
    if abi_align > 0 {
        llvm_export::ops::set_op_alignment(ctx, store_op.get_operation(), abi_align as u32);
    }
    slot_ptr
}

/// Pointer to `base + offset` bytes, for reaching a payload field inside
/// a spilled enum (`getelementptr i8, ptr base, offset`).
fn enum_byte_gep(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    base: Value,
    offset: u64,
) -> Value {
    use llvm_export::ops::GepIndex;
    let i8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signless).into();
    let gep_op =
        llvm::GetElementPtrOp::new(ctx, base, vec![GepIndex::Constant(offset as u32)], i8_ty);
    rewriter.insert_operation(ctx, gep_op.get_operation());
    gep_op.get_operation().deref(ctx).get_result(0)
}

/// Convert `mir.construct_enum` (e.g. `E::A(x)`) to LLVM operations.
///
/// Builds the enum value slot by slot, taking every index from
/// [`build_enum_slot_map`] (indexes are never computed by hand here):
///
/// 1. Put the variant's declared discriminant VALUE into the tag slot.
/// 2. `insertvalue` each payload field that owns a struct slot.
/// 3. If some field has no slot (its bytes are shared with a
///    different-typed field of another variant), finish through memory:
///    copy the value to a stack slot, store that field at its byte
///    position, and load the completed enum back.
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

    let (variant_discriminants, variant_field_counts, mir_discr_ty, abi_align): (
        Vec<u64>,
        Vec<u32>,
        TypeHandle,
        u64,
    ) = {
        let ty_ref = result_ty.deref(ctx);
        match ty_ref.downcast_ref::<MirEnumType>() {
            Some(e) => (
                e.variant_discriminants.clone(),
                e.variant_field_counts.clone(),
                e.discriminant_ty,
                e.abi_align(),
            ),
            None => {
                return pliron::input_err_noloc!(
                    "MirConstructEnumOp result type must be MirEnumType"
                );
            }
        }
    };

    // Build the value as the SAME struct type the type converter
    // produces everywhere else (block args, loads, allocas, ...). Taking
    // both the type and the indices from one slot map is what keeps them
    // in agreement. Filler slots are simply never written.
    let slot_map = build_enum_slot_map(ctx, result_ty).map_err(anyhow_to_pliron)?;
    let llvm_struct_ty = slot_map.llvm_struct_ty;
    let llvm_discriminant_ty = convert_type(ctx, mir_discr_ty).map_err(anyhow_to_pliron)?;

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

    let insert_discr =
        llvm::InsertValueOp::new(ctx, current_struct, discr_val, vec![slot_map.tag_slot]);
    rewriter.insert_operation(ctx, insert_discr.get_operation());
    current_struct = insert_discr.get_operation().deref(ctx).get_result(0);

    let field_base: usize = variant_field_counts
        .iter()
        .take(variant_index)
        .map(|&c| c as usize)
        .sum();

    // Insert every payload field that owns a struct slot; remember the
    // slotless ones for the memory pass below.
    let mut deferred: Vec<(usize, Value)> = Vec::new();
    let mut last_op = insert_discr.get_operation();
    for (i, operand) in operands.into_iter().enumerate() {
        let flat = field_base + i;
        let Some(slot) = slot_map.field_slots.get(flat) else {
            return pliron::input_err_noloc!(
                "MirConstructEnumOp field {} of variant {} is out of range for the enum's {} fields",
                i,
                variant_index,
                slot_map.field_slots.len()
            );
        };
        match slot {
            Some(slot) => {
                let insert_op = llvm::InsertValueOp::new(ctx, current_struct, operand, vec![*slot]);
                rewriter.insert_operation(ctx, insert_op.get_operation());
                current_struct = insert_op.get_operation().deref(ctx).get_result(0);
                last_op = insert_op.get_operation();
            }
            None => {
                // Zero-sized fields own no bytes; nothing to write.
                if is_zero_sized_type(ctx, slot_map.field_llvm_types[flat]) {
                    continue;
                }
                deferred.push((flat, operand));
            }
        }
    }

    if deferred.is_empty() {
        rewriter.replace_operation(ctx, op, last_op);
        return Ok(());
    }

    // Slotless fields: copy the half-built value to the stack, write
    // each remaining payload at its byte position, and load the finished
    // enum back as the result.
    let slot_ptr = spill_enum_value(ctx, rewriter, current_struct, llvm_struct_ty, abi_align);
    for (flat, operand) in deferred {
        let field_ptr = enum_byte_gep(ctx, rewriter, slot_ptr, slot_map.field_offsets[flat]);
        let store_op = llvm::StoreOp::new(ctx, operand, field_ptr);
        rewriter.insert_operation(ctx, store_op.get_operation());
    }
    let load_op = llvm::LoadOp::new(ctx, slot_ptr, llvm_struct_ty);
    rewriter.insert_operation(ctx, load_op.get_operation());
    if abi_align > 0 {
        llvm_export::ops::set_op_alignment(ctx, load_op.get_operation(), abi_align as u32);
    }
    rewriter.replace_operation(ctx, op, load_op.get_operation());

    Ok(())
}

/// Get the slot map for an enum operand.
///
/// By the time an op is converted, its operand's type has already been
/// rewritten to the LLVM struct, so we look up the ORIGINAL `MirEnumType`
/// the framework recorded for it and rebuild the map from that. Also
/// returns the enum's rustc alignment, which spill slots need.
fn enum_slot_map_of_operand(
    ctx: &mut Context,
    operands_info: &OperandsInfo,
    enum_val: Value,
) -> Result<(EnumSlotMap, u64)> {
    // Clone the type data out so the `Ref` borrow of `ctx` ends before
    // re-interning (types are hash-consed: registering an equal instance
    // returns the existing pointer).
    let enum_ty: MirEnumType = {
        match operands_info.lookup_most_recent_of_type::<MirEnumType>(ctx, enum_val) {
            Some(r) => r.clone(),
            None => {
                return pliron::input_err_noloc!("Expected MirEnumType for enum value access");
            }
        }
    };
    let abi_align = enum_ty.abi_align();
    let mir_ty: TypeHandle = pliron::r#type::Type::register_instance(enum_ty, ctx).into();
    let map = build_enum_slot_map(ctx, mir_ty).map_err(anyhow_to_pliron)?;
    Ok((map, abi_align))
}

/// Convert `mir.set_discriminant` (writing an enum's tag) to
/// `llvm.getelementptr` + `llvm.store`.
///
/// The tag is written into the slot map's `tag_slot`. The discriminant
/// value operand is already the variant's DECLARED discriminant, so no
/// index-to-value translation is needed here.
pub(crate) fn convert_set_discriminant(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let enum_ptr = op.deref(ctx).get_operand(0);
    let discr_val = op.deref(ctx).get_operand(1);

    let enum_ty: MirEnumType = {
        let mir_ptr_pointee =
            match operands_info.lookup_most_recent_of_type::<MirPtrType>(ctx, enum_ptr) {
                Some(r) => r.pointee,
                None => {
                    return pliron::input_err_noloc!(
                        "MirSetDiscriminantOp operand must be pointer type"
                    );
                }
            };
        match mir_ptr_pointee.deref(ctx).downcast_ref::<MirEnumType>() {
            Some(et) => et.clone(),
            None => {
                return pliron::input_err_noloc!(
                    "MirSetDiscriminantOp pointer must point to enum type"
                );
            }
        }
    };

    // Only direct-tag enums have a physical tag address. The importer handles
    // a no-tag Single layout as a no-op and rejects niche/empty layouts before
    // creating this operation. Keep the same invariant here so malformed or
    // hand-built dialect IR cannot store into the synthetic fallback tag used
    // by other enum operations.
    if enum_ty.total_size() == 0 {
        return pliron::input_err_noloc!(
            "MirSetDiscriminantOp requires a direct-tag enum with recorded layout"
        );
    }

    let mir_ty: TypeHandle = pliron::r#type::Type::register_instance(enum_ty, ctx).into();
    let slot_map = build_enum_slot_map(ctx, mir_ty).map_err(anyhow_to_pliron)?;
    let llvm_struct_ty = slot_map.llvm_struct_ty;

    use llvm_export::ops::GepIndex;
    let gep_indices = vec![GepIndex::Constant(0), GepIndex::Constant(slot_map.tag_slot)];
    let tag_ptr_op = llvm::GetElementPtrOp::new(ctx, enum_ptr, gep_indices, llvm_struct_ty);
    rewriter.insert_operation(ctx, tag_ptr_op.get_operation());

    let tag_ptr = tag_ptr_op.get_operation().deref(ctx).get_result(0);
    let store_op = llvm::StoreOp::new(ctx, discr_val, tag_ptr);
    rewriter.insert_operation(ctx, store_op.get_operation());

    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert `mir.get_discriminant` (reading which variant is alive) to
/// `llvm.extractvalue`.
///
/// The tag is read from the slot map's `tag_slot`. That is usually slot
/// 0, but rustc may put the tag after a payload, so the slot is never
/// assumed. The value read is the variant's DECLARED discriminant (what
/// `construct_enum` stored), so the `match` that follows compares
/// declared values, never variant positions.
pub(crate) fn convert_get_discriminant(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let enum_val = match op.deref(ctx).operands().next() {
        Some(v) => v,
        None => return pliron::input_err_noloc!("MirGetDiscriminantOp requires an operand"),
    };

    let (slot_map, _abi_align) = enum_slot_map_of_operand(ctx, operands_info, enum_val)?;

    let extract_op = llvm::ExtractValueOp::new(ctx, enum_val, vec![slot_map.tag_slot])?;
    rewriter.insert_operation(ctx, extract_op.get_operation());
    rewriter.replace_operation(ctx, op, extract_op.get_operation());

    Ok(())
}

/// Convert `mir.enum_payload` (reading a variant's field, e.g. the `x`
/// in `E::A(x) => x`) to a payload-field read.
///
/// Three cases, decided by the [`EnumSlotMap`]:
///
/// - The field owns a struct slot: a plain `llvm.extractvalue`.
/// - The field has no slot (its bytes are shared with a different-typed
///   field of another variant): go through memory. Copy the enum to a
///   stack slot, point at the field's byte position, and load it with
///   its own type. Same trick as [`convert_extract_array_element`], and
///   it avoids LLVM `bitcast` entirely.
/// - The field is zero-sized: there is nothing to read; produce `undef`.
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
    let (slot_map, abi_align) = enum_slot_map_of_operand(ctx, operands_info, enum_val)?;

    let field_base: usize = variant_field_counts
        .iter()
        .take(variant_index)
        .map(|&c| c as usize)
        .sum();
    let flat = field_base + field_index;
    let Some(slot) = slot_map.field_slots.get(flat).copied() else {
        return pliron::input_err_noloc!(
            "MirEnumPayloadOp field {} of variant {} is out of range for the enum's {} fields",
            field_index,
            variant_index,
            slot_map.field_slots.len()
        );
    };

    match slot {
        Some(slot) => {
            let extract_op = llvm::ExtractValueOp::new(ctx, enum_val, vec![slot])?;
            rewriter.insert_operation(ctx, extract_op.get_operation());
            rewriter.replace_operation(ctx, op, extract_op.get_operation());
        }
        None if is_zero_sized_type(ctx, slot_map.field_llvm_types[flat]) => {
            let undef_op = llvm::UndefOp::new(ctx, slot_map.field_llvm_types[flat]);
            rewriter.insert_operation(ctx, undef_op.get_operation());
            rewriter.replace_operation(ctx, op, undef_op.get_operation());
        }
        None => {
            let slot_ptr =
                spill_enum_value(ctx, rewriter, enum_val, slot_map.llvm_struct_ty, abi_align);
            let field_ptr = enum_byte_gep(ctx, rewriter, slot_ptr, slot_map.field_offsets[flat]);
            let load_op = llvm::LoadOp::new(ctx, field_ptr, slot_map.field_llvm_types[flat]);
            rewriter.insert_operation(ctx, load_op.get_operation());
            rewriter.replace_operation(ctx, op, load_op.get_operation());
        }
    }

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

    let mir_ptr_pointee =
        match operands_info.lookup_most_recent_of_type::<MirPtrType>(ctx, ptr_operand) {
            Some(r) => r.pointee,
            None => {
                return pliron::input_err_noloc!("MirFieldAddrOp operand must be pointer type");
            }
        };

    let union_field_count = mir_ptr_pointee
        .deref(ctx)
        .downcast_ref::<MirUnionType>()
        .map(MirUnionType::field_count);
    if let Some(field_count) = union_field_count {
        if field_index >= field_count {
            return pliron::input_err_noloc!(
                "field_addr index {} out of bounds for union with {} fields",
                field_index,
                field_count
            );
        }
        // Every union field begins at byte zero. Emit an explicit zero-offset
        // GEP instead of forwarding the base SSA value directly: the distinct
        // result keeps dialect conversion's pointer-type history unambiguous
        // for repeated field accesses and for `union.struct_field.inner`.
        use llvm_export::ops::GepIndex;
        let i8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signless).into();
        let gep = llvm::GetElementPtrOp::new(ctx, ptr_operand, vec![GepIndex::Constant(0)], i8_ty);
        rewriter.insert_operation(ctx, gep.get_operation());
        rewriter.replace_operation(ctx, op, gep.get_operation());
        return Ok(());
    }

    let layout = {
        let pointee_ref = mir_ptr_pointee.deref(ctx);
        match pointee_ref.downcast_ref::<MirStructType>() {
            Some(struct_ty) => StructLayoutInfo::of_struct(struct_ty),
            None => {
                return pliron::input_err_noloc!(
                    "MirFieldAddrOp pointer must point to struct or union type, got {}",
                    mir_ptr_pointee.deref(ctx).disp(ctx)
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
    use super::*;
    use crate::convert::ops::test_util::*;
    use dialect_mir::attributes::{FieldIndexAttr, VariantIndexAttr};
    use dialect_mir::ops as mir;
    use dialect_mir::types::{EnumVariant, MirPtrType, MirSliceType, MirStructType};
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::common_traits::Verify;

    fn insert_indices(ctx: &Context, inserts: &[llvm::InsertValueOp]) -> Vec<Vec<u32>> {
        inserts.iter().map(|op| op.indices(ctx)).collect()
    }

    fn empty_struct_ty(ctx: &mut Context, name: &str) -> TypeHandle {
        MirStructType::get(ctx, name.to_string(), vec![], vec![]).into()
    }

    fn padded_struct_with_zst_ty(ctx: &mut Context) -> (TypeHandle, TypeHandle) {
        let i8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signless).into();
        let i64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Signless).into();
        let zst_ty = empty_struct_ty(ctx, "Marker");

        let struct_ty = MirStructType::get_with_full_layout(
            ctx,
            "Padded".to_string(),
            vec!["a".to_string(), "marker".to_string(), "b".to_string()],
            vec![i8_ty, zst_ty, i64_ty],
            vec![0, 1, 2],
            vec![0, 1, 8],
            16,
            8,
        );

        (struct_ty.into(), zst_ty)
    }

    fn append_empty_struct_value(
        ctx: &mut Context,
        block: Ptr<pliron::basic_block::BasicBlock>,
        zst_ty: TypeHandle,
    ) -> Value {
        let op = Operation::new(
            ctx,
            mir::MirConstructStructOp::get_concrete_op_info(),
            vec![zst_ty],
            vec![],
            vec![],
            0,
        );
        op.insert_at_back(block, ctx);
        op.deref(ctx).get_result(0)
    }

    /// `mir.construct_slice` lowers to the canonical fat-pointer value:
    /// `undef { ptr, i64 }`, then insert data pointer at slot 0 and length at slot 1.
    #[test]
    fn construct_slice_lowers_to_ptr_len_insert_values() {
        let mut ctx = make_ctx();

        let i8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let usize_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, i8_ty, false).into();
        let slice_ty: TypeHandle = MirSliceType::get(&mut ctx, i8_ty).into();

        let (module_ptr, block) = build_kernel(&mut ctx, vec![ptr_ty, usize_ty], vec![]);
        let data_ptr = block.deref(&ctx).get_argument(0);
        let len = block.deref(&ctx).get_argument(1);

        let op = Operation::new(
            &mut ctx,
            mir::MirConstructSliceOp::get_concrete_op_info(),
            vec![slice_ty],
            vec![data_ptr, len],
            vec![],
            0,
        );
        op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let inserts = find_all::<llvm::InsertValueOp>(&ctx, &body);

        assert_eq!(
            insert_indices(&ctx, &inserts),
            vec![vec![0], vec![1]],
            "slice construction must insert data pointer at slot 0 and length at slot 1"
        );
        let first_insert = inserts[0].get_operation();
        let second_insert = inserts[1].get_operation();
        assert_eq!(
            first_insert.deref(&ctx).get_operand(1),
            data_ptr,
            "slice slot 0 must receive the original data pointer"
        );
        assert_eq!(
            second_insert.deref(&ctx).get_operand(0),
            first_insert.deref(&ctx).get_result(0),
            "the length insertion must consume the aggregate produced by the pointer insertion"
        );
        assert_eq!(
            second_insert.deref(&ctx).get_operand(1),
            len,
            "slice slot 1 must receive the original length"
        );
        assert!(
            inserts.iter().all(|insert| insert.verify(&ctx).is_ok()),
            "both slice insertions must satisfy LLVM dialect verification"
        );
        assert_eq!(
            count_ops::<llvm::UndefOp>(&ctx, &body),
            1,
            "slice construction should start from one undef aggregate"
        );
    }

    /// Explicit rustc layout must be respected: field `b` is at byte offset 8,
    /// so the lowered LLVM struct has a padding slot between `a` and `b`.
    /// The ZST marker field is stripped and receives no insert_value.
    #[test]
    fn construct_struct_uses_layout_slots_and_skips_zst() {
        let mut ctx = make_ctx();

        let i8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Signless).into();
        let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signless).into();
        let (struct_ty, zst_ty) = padded_struct_with_zst_ty(&mut ctx);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![i8_ty, i64_ty], vec![]);
        let a = block.deref(&ctx).get_argument(0);
        let b = block.deref(&ctx).get_argument(1);
        let marker = append_empty_struct_value(&mut ctx, block, zst_ty);

        let op = Operation::new(
            &mut ctx,
            mir::MirConstructStructOp::get_concrete_op_info(),
            vec![struct_ty],
            vec![a, marker, b],
            vec![],
            0,
        );
        op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let inserts = find_all::<llvm::InsertValueOp>(&ctx, &body);

        assert_eq!(
            insert_indices(&ctx, &inserts),
            vec![vec![0], vec![2]],
            "non-ZST fields must be inserted at their layout slots, skipping padding and ZSTs"
        );
        let first_insert = inserts[0].get_operation();
        let second_insert = inserts[1].get_operation();
        assert_eq!(
            first_insert.deref(&ctx).get_operand(1),
            a,
            "struct slot 0 must receive field `a`"
        );
        assert_eq!(
            second_insert.deref(&ctx).get_operand(0),
            first_insert.deref(&ctx).get_result(0),
            "field `b` must be inserted into the aggregate containing field `a`"
        );
        assert_eq!(
            second_insert.deref(&ctx).get_operand(1),
            b,
            "struct slot 2 must receive field `b`"
        );
        assert!(
            inserts.iter().all(|insert| insert.verify(&ctx).is_ok()),
            "both struct insertions must satisfy LLVM dialect verification"
        );
    }

    /// Extracting a ZST field must not emit `extract_value`: the field has no
    /// storage in the lowered LLVM struct, so lowering materializes an undef
    /// zero-sized value instead.
    #[test]
    fn extract_zst_field_lowers_to_undef_without_extract_value() {
        let mut ctx = make_ctx();

        let i8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Signless).into();
        let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signless).into();
        let (struct_ty, zst_ty) = padded_struct_with_zst_ty(&mut ctx);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![i8_ty, i64_ty], vec![]);
        let a = block.deref(&ctx).get_argument(0);
        let b = block.deref(&ctx).get_argument(1);
        let marker = append_empty_struct_value(&mut ctx, block, zst_ty);

        let construct = Operation::new(
            &mut ctx,
            mir::MirConstructStructOp::get_concrete_op_info(),
            vec![struct_ty],
            vec![a, marker, b],
            vec![],
            0,
        );
        construct.insert_at_back(block, &ctx);
        let aggregate = construct.deref(&ctx).get_result(0);

        let extract = Operation::new(
            &mut ctx,
            MirExtractFieldOp::get_concrete_op_info(),
            vec![zst_ty],
            vec![aggregate],
            vec![],
            0,
        );
        MirExtractFieldOp::new(extract).set_attr_index(&ctx, FieldIndexAttr(1));
        extract.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);

        assert_eq!(
            count_ops::<llvm::ExtractValueOp>(&ctx, &body),
            0,
            "extracting a stripped ZST field must not emit llvm.extractvalue"
        );

        let zst_un_defs = find_all::<llvm::UndefOp>(&ctx, &body)
            .into_iter()
            .filter(|op| {
                let result_ty = op.get_operation().deref(&ctx).get_result(0).get_type(&ctx);
                is_zero_sized_type(&ctx, result_ty)
            })
            .count();

        assert_eq!(
            zst_un_defs, 2,
            "one undef should build the ZST value and one should materialize the extracted ZST"
        );
    }

    /// Enum construction must store the declared discriminant value, not the
    /// variant index. This locks the `Ordering::Less = -1` style case as the
    /// i8 bit-pattern `255`.
    #[test]
    fn construct_enum_uses_declared_discriminant_not_variant_index() {
        let mut ctx = make_ctx();

        let discr_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Signed).into();
        let enum_ty: TypeHandle = MirEnumType::get(
            &mut ctx,
            "OrderingLike".to_string(),
            discr_ty,
            vec![255, 0, 1],
            vec![
                EnumVariant::unit("Less".to_string()),
                EnumVariant::unit("Equal".to_string()),
                EnumVariant::unit("Greater".to_string()),
            ],
        )
        .into();

        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);

        let op = Operation::new(
            &mut ctx,
            MirConstructEnumOp::get_concrete_op_info(),
            vec![enum_ty],
            vec![],
            vec![],
            0,
        );
        MirConstructEnumOp::new(op)
            .set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(0));
        op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let inserts = find_all::<llvm::InsertValueOp>(&ctx, &body);

        assert_eq!(
            insert_indices(&ctx, &inserts),
            vec![vec![0]],
            "unit enum construction should insert only the discriminant tag"
        );
        let tag_insert = &inserts[0];
        assert!(
            tag_insert.verify(&ctx).is_ok(),
            "the enum tag insertion must satisfy LLVM dialect verification"
        );
        let tag = tag_insert.get_operation().deref(&ctx).get_operand(1);
        let tag_def = tag
            .defining_op()
            .expect("the inserted enum tag must have a defining operation");
        let tag_constant = Operation::get_op::<llvm::ConstantOp>(tag_def, &ctx)
            .expect("the inserted enum tag must be defined by llvm.constant");
        let tag_attr = tag_constant.get_value(&ctx);
        let tag_integer = tag_attr
            .downcast_ref::<IntegerAttr>()
            .expect("the inserted enum tag must be an integer constant");
        assert_eq!(tag_integer.value().bw(), 8, "the enum tag must be 8-bit");
        assert_eq!(
            tag_integer.value().to_u64(),
            255,
            "Less must lower to its declared i8 bit-pattern 255, not variant index 0"
        );
    }

    /// SetDiscriminant must use the slot map instead of assuming that the tag
    /// is field zero, and its GEP must retain the source pointer's GPU address
    /// space. This shape puts an i64 payload first and an i8 tag at byte 8.
    #[test]
    fn set_discriminant_uses_tag_slot_and_preserves_shared_address_space() {
        use llvm_export::ops::GepIndex;
        use llvm_export::types::{PointerType, address_space};

        let mut ctx = make_ctx();
        let discr_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let payload_a: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let payload_b: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "TagAfterPayload".to_string(),
            discr_ty,
            vec![3, 7],
            vec![
                EnumVariant::new_with_offsets("A".to_string(), vec![payload_a], vec![0]),
                EnumVariant::new_with_offsets("B".to_string(), vec![payload_b], vec![0]),
            ],
            8,
            16,
            8,
        )
        .into();
        let ptr_ty: TypeHandle = MirPtrType::get_shared(&mut ctx, enum_ty, true).into();

        let (module_ptr, block) = build_kernel(&mut ctx, vec![ptr_ty, discr_ty], vec![]);
        let enum_ptr = block.deref(&ctx).get_argument(0);
        let discr = block.deref(&ctx).get_argument(1);
        let set = Operation::new(
            &mut ctx,
            mir::MirSetDiscriminantOp::get_concrete_op_info(),
            vec![],
            vec![enum_ptr, discr],
            vec![],
            0,
        );
        set.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        assert_eq!(count_ops::<mir::MirSetDiscriminantOp>(&ctx, &body), 0);
        assert_eq!(count_ops::<llvm::GetElementPtrOp>(&ctx, &body), 1);
        assert_eq!(count_ops::<llvm::StoreOp>(&ctx, &body), 1);

        let gep = find_first::<llvm::GetElementPtrOp>(&ctx, &body).unwrap();
        let indices = gep.indices(&ctx);
        assert!(
            matches!(
                indices.as_slice(),
                [GepIndex::Constant(0), GepIndex::Constant(1)]
            ),
            "the tag at byte 8 must be LLVM struct slot 1"
        );
        let gep_result_ty = gep.get_operation().deref(&ctx).get_result(0).get_type(&ctx);
        assert_eq!(
            gep_result_ty
                .deref(&ctx)
                .downcast_ref::<PointerType>()
                .expect("GEP result must be a pointer")
                .address_space(),
            address_space::SHARED,
            "tag GEP must preserve shared address space"
        );

        let store = find_first::<llvm::StoreOp>(&ctx, &body).unwrap();
        let stored_ty = store.get_operand_value(&ctx).get_type(&ctx);
        assert_eq!(
            stored_ty
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .expect("stored tag must be an integer")
                .width(),
            8
        );
        assert_eq!(
            store.get_operand_address(&ctx),
            gep.get_operation().deref(&ctx).get_result(0),
            "the store must use the tag GEP result"
        );
    }

    /// A synthetic tag used for cuda-oxide's device-private niche model is
    /// not a physical rustc tag and must never be written by SetDiscriminant.
    #[test]
    fn set_discriminant_rejects_enum_without_recorded_direct_layout() {
        let mut ctx = make_ctx();
        let discr_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let enum_ty: TypeHandle = MirEnumType::get(
            &mut ctx,
            "NoPhysicalTag".to_string(),
            discr_ty,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".to_string()),
                EnumVariant::unit("Some".to_string()),
            ],
        )
        .into();
        let ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, enum_ty, true).into();

        let (module_ptr, block) = build_kernel(&mut ctx, vec![ptr_ty, discr_ty], vec![]);
        let enum_ptr = block.deref(&ctx).get_argument(0);
        let discr = block.deref(&ctx).get_argument(1);
        let set = Operation::new(
            &mut ctx,
            mir::MirSetDiscriminantOp::get_concrete_op_info(),
            vec![],
            vec![enum_ptr, discr],
            vec![],
            0,
        );
        set.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        let error = crate::lower_mir_to_llvm(&mut ctx, module_ptr)
            .expect_err("a no-tag enum must not lower to a synthetic tag store");
        assert!(
            error
                .to_string()
                .contains("requires a direct-tag enum with recorded layout"),
            "unexpected lowering error: {error}"
        );
    }
}
