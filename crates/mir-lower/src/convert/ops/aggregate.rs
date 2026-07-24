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
//! Enums use rustc's physical layout, not a cuda-oxide-only tagged struct.
//! `build_enum_slot_map` places the direct tag or niche carrier and every
//! payload at rustc's byte offsets, reuses identical overlapping storage, and
//! routes differently typed overlaps through byte-addressed memory. `Single`
//! and `Empty` layouts have no carrier at all.
//!
//! A direct tag holds the variant's DECLARED discriminant value, not its
//! position. For `core::cmp::Ordering`, `Less` therefore stores -1 (the i8 bit
//! pattern 255), `Equal` stores 0, and `Greater` stores 1. A niche layout
//! instead uses rustc's wrapping `niche_start + variant_offset` encoding and
//! introduces no extra tag.

use crate::convert::types::{
    EnumSlotMap, StructLayoutInfo, StructSlotMap, build_enum_slot_map, build_struct_slot_map,
    build_union_storage_type, convert_type, is_zero_sized_type, llvm_byte_faithful_twin,
    llvm_type_contains_i1, make_slice_struct, mir_type_abi_align,
};
use dialect_mir::ops::{
    MirConstructEnumOp, MirEnumPayloadOp, MirExtractFieldOp, MirFieldAddrOp, MirInsertFieldOp,
    MirSetDiscriminantOp,
};
use dialect_mir::types::{
    EnumCarrierKind, EnumLayoutKind, MirArrayType, MirDisjointSliceType, MirEnumType, MirPtrType,
    MirSliceType, MirStructType, MirTupleType, MirUnionType,
};
use llvm_export::attributes::{ICmpPredicateAttr, IntegerOverflowFlagsAttr};
use llvm_export::op_interfaces::{
    CastOpInterface, CastOpWithNNegInterface, IntBinArithOpWithOverflowFlag,
};
use llvm_export::ops as llvm;
use llvm_export::types as llvm_types;
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
    let abi_align = mir_type_abi_align(ctx, element_ty);

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
    if let Some(align) = abi_align {
        llvm_export::ops::set_op_alignment(ctx, alloca_op.get_operation(), align as u32);
    }
    let array_ptr = alloca_op.get_operation().deref(ctx).get_result(0);

    let store_op = llvm::StoreOp::new(ctx, array_val, array_ptr);
    rewriter.insert_operation(ctx, store_op.get_operation());
    if let Some(align) = abi_align {
        llvm_export::ops::set_op_alignment(ctx, store_op.get_operation(), align as u32);
    }

    use llvm_export::ops::GepIndex;
    let gep_indices = vec![GepIndex::Constant(0), GepIndex::Value(index_val)];
    let gep_op = llvm::GetElementPtrOp::new(ctx, array_ptr, gep_indices, llvm_array_ty.into());
    rewriter.insert_operation(ctx, gep_op.get_operation());
    let element_ptr = gep_op.get_operation().deref(ctx).get_result(0);

    let load_op = llvm::LoadOp::new(ctx, element_ptr, llvm_element_ty);
    rewriter.insert_operation(ctx, load_op.get_operation());
    if let Some(align) = abi_align {
        llvm_export::ops::set_op_alignment(ctx, load_op.get_operation(), align as u32);
    }
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
    let offset = emit_integer_constant(ctx, rewriter, 64, u128::from(offset));
    let gep_op = llvm::GetElementPtrOp::new(ctx, base, vec![GepIndex::Value(offset)], i8_ty);
    rewriter.insert_operation(ctx, gep_op.get_operation());
    gep_op.get_operation().deref(ctx).get_result(0)
}

/// The physical carrier write required to select one source variant.
/// `None` is a real semantic result for the untagged niche variant and for
/// rustc's single inhabited variant; it must never be replaced by a guessed
/// memory write.
fn enum_carrier_bits_for_variant(
    enum_ty: &MirEnumType,
    variant: usize,
) -> std::result::Result<Option<u128>, String> {
    if variant >= enum_ty.variant_count() || enum_ty.variant_is_inhabited(variant) != Some(true) {
        return Err(format!(
            "cannot select uninhabited or missing variant {} of '{}'",
            variant,
            enum_ty.name()
        ));
    }

    match enum_ty.layout_kind {
        EnumLayoutKind::Direct => enum_ty
            .variant_discriminants
            .get(variant)
            .copied()
            .map(|bits| Some(u128::from(bits)))
            .ok_or_else(|| format!("variant {} has no declared discriminant", variant)),
        EnumLayoutKind::Niche => {
            // This check deliberately comes before the encoded range check:
            // rustc permits the untagged variant index to lie inside that
            // range. Its range position is a dead niche value; selecting the
            // actual untagged variant remains a no-op.
            if variant == enum_ty.untagged_variant as usize {
                return Ok(None);
            }
            if !(enum_ty.niche_variant_start as usize..=enum_ty.niche_variant_end as usize)
                .contains(&variant)
            {
                return Err(format!(
                    "inhabited variant {} is not representable by niche layout of '{}'",
                    variant,
                    enum_ty.name()
                ));
            }
            let offset = (variant as u128) - u128::from(enum_ty.niche_variant_start);
            let mut bits = enum_ty.niche_start().wrapping_add(offset);
            if enum_ty.carrier_width < 128 {
                bits &= (1u128 << enum_ty.carrier_width) - 1;
            }
            Ok(Some(bits))
        }
        EnumLayoutKind::Single if variant == enum_ty.single_variant as usize => Ok(None),
        EnumLayoutKind::Single => Err(format!(
            "variant {} is not the single inhabited variant of '{}'",
            variant,
            enum_ty.name()
        )),
        EnumLayoutKind::Empty => Err(format!("enum '{}' is uninhabited", enum_ty.name())),
        EnumLayoutKind::Unknown => Err(format!(
            "enum '{}' has unknown physical layout",
            enum_ty.name()
        )),
    }
}

fn emit_integer_constant(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    width: u32,
    bits: u128,
) -> Value {
    let ty = IntegerType::get(ctx, width, Signedness::Signless);
    let attr = pliron::builtin::attributes::IntegerAttr::new(
        ty,
        APInt::from_u128(bits, NonZeroUsize::new(width as usize).unwrap()),
    );
    let op = llvm::ConstantOp::new(ctx, attr.into());
    rewriter.insert_operation(ctx, op.get_operation());
    op.get_operation().deref(ctx).get_result(0)
}

fn emit_carrier_constant(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    enum_ty: &MirEnumType,
    carrier_ty: TypeHandle,
    bits: u128,
) -> Result<Value> {
    let integer = emit_integer_constant(ctx, rewriter, enum_ty.carrier_width, bits);
    match enum_ty.carrier_kind {
        EnumCarrierKind::Integer => Ok(integer),
        EnumCarrierKind::Pointer => {
            let cast = llvm::IntToPtrOp::new(ctx, integer, carrier_ty);
            rewriter.insert_operation(ctx, cast.get_operation());
            Ok(cast.get_operation().deref(ctx).get_result(0))
        }
        _ => pliron::input_err_noloc!("enum carrier constant requested without a carrier"),
    }
}

/// Convert a value to its byte-faithful storage twin: every `i1` leaf is
/// zero-extended to its canonical `i8` memory byte, recursively through
/// structs and arrays. Values without `i1` storage pass through unchanged.
///
/// This is the value-level half of [`llvm_byte_faithful_twin`]: the enum
/// slot map claims twin-typed storage for bool-bearing payloads, and this
/// produces the twin-typed value the store into that storage needs.
fn canonicalize_bool_value_bytes(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: Value,
) -> Result<Value> {
    let ty = value.get_type(ctx);
    if !llvm_type_contains_i1(ctx, ty) {
        return Ok(value);
    }
    let is_scalar_i1 = ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| integer.width() == 1);
    if is_scalar_i1 {
        let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signless).into();
        let zext = llvm::ZExtOp::new_with_nneg(ctx, value, byte_ty, false);
        rewriter.insert_operation(ctx, zext.get_operation());
        return Ok(zext.get_operation().deref(ctx).get_result(0));
    }
    let Some(twin) = llvm_byte_faithful_twin(ctx, ty) else {
        return pliron::input_err_noloc!(
            "enum construction: bool storage in this value's shape cannot be canonicalized"
        );
    };
    let element_count = {
        let ty_ref = ty.deref(ctx);
        if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
            struct_ty.fields().count() as u64
        } else if let Some(array_ty) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
            array_ty.size()
        } else {
            return pliron::input_err_noloc!(
                "enum construction: unexpected container for bool storage canonicalization"
            );
        }
    };
    let undef_op = llvm::UndefOp::new(ctx, twin);
    rewriter.insert_operation(ctx, undef_op.get_operation());
    let mut current = undef_op.get_operation().deref(ctx).get_result(0);
    for index in 0..element_count {
        let extract_op = llvm::ExtractValueOp::new(ctx, value, vec![index as u32])?;
        rewriter.insert_operation(ctx, extract_op.get_operation());
        let element = extract_op.get_operation().deref(ctx).get_result(0);
        let converted = canonicalize_bool_value_bytes(ctx, rewriter, element)?;
        let insert_op = llvm::InsertValueOp::new(ctx, current, converted, vec![index as u32]);
        rewriter.insert_operation(ctx, insert_op.get_operation());
        current = insert_op.get_operation().deref(ctx).get_result(0);
    }
    Ok(current)
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

    let enum_ty: MirEnumType = {
        let ty_ref = result_ty.deref(ctx);
        match ty_ref.downcast_ref::<MirEnumType>() {
            Some(e) => e.clone(),
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

    let undef_op = llvm::UndefOp::new(ctx, llvm_struct_ty);
    rewriter.insert_operation(ctx, undef_op.get_operation());
    let mut current_struct = undef_op.get_operation().deref(ctx).get_result(0);
    let mut last_op = undef_op.get_operation();

    if let Some(bits) = enum_carrier_bits_for_variant(&enum_ty, variant_index)
        .map_err(|error| pliron::input_error_noloc!("MirConstructEnumOp: {error}"))?
    {
        let carrier_slot = slot_map.carrier_slot.ok_or_else(|| {
            pliron::input_error_noloc!("MirConstructEnumOp requires a physical carrier slot")
        })?;
        let carrier_ty = slot_map.carrier_llvm_ty.ok_or_else(|| {
            pliron::input_error_noloc!("MirConstructEnumOp requires a physical carrier type")
        })?;
        let carrier = emit_carrier_constant(ctx, rewriter, &enum_ty, carrier_ty, bits)?;
        let insert = llvm::InsertValueOp::new(ctx, current_struct, carrier, vec![carrier_slot]);
        rewriter.insert_operation(ctx, insert.get_operation());
        current_struct = insert.get_operation().deref(ctx).get_result(0);
        last_op = insert.get_operation();
    }

    let field_base: usize = enum_ty
        .variant_field_counts
        .iter()
        .take(variant_index)
        .map(|&c| c as usize)
        .sum();

    // Insert every payload field that owns a struct slot; remember the
    // slotless ones for the memory pass below.
    let mut deferred: Vec<(usize, Value)> = Vec::new();
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
    let abi_align = enum_ty.abi_align();
    let slot_ptr = spill_enum_value(ctx, rewriter, current_struct, llvm_struct_ty, abi_align);
    for (flat, operand) in deferred {
        let field_ptr = enum_byte_gep(ctx, rewriter, slot_ptr, slot_map.field_offsets[flat]);
        // `bool` is an LLVM i1 as a value but occupies one full byte in
        // Rust memory. Enum storage claims the byte-faithful twin of every
        // bool-bearing payload (scalar i8 byte, or an aggregate with each
        // i1 leaf widened to i8), so canonicalize the stored value to that
        // twin: every physical bool byte becomes an unambiguous 0 or 1.
        let stored_operand = canonicalize_bool_value_bytes(ctx, rewriter, operand)?;
        let store_op = llvm::StoreOp::new(ctx, stored_operand, field_ptr);
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

/// Convert `mir.set_discriminant` to the one physical carrier write rustc's
/// layout requires. Direct layouts write the declared discriminant; Niche
/// layouts write wrapping `niche_start + range_offset`. Selecting the
/// untagged Niche variant or an inhabited Single variant is a no-op.
pub(crate) fn convert_set_discriminant(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let enum_ptr = op.deref(ctx).get_operand(0);
    let target = MirSetDiscriminantOp::new(op)
        .get_attr_set_discriminant_variant_index(ctx)
        .map(|attr| attr.0 as usize)
        .ok_or_else(|| {
            pliron::input_error_noloc!(
                "MirSetDiscriminantOp missing set_discriminant_variant_index"
            )
        })?;

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

    let Some(bits) = enum_carrier_bits_for_variant(&enum_ty, target)
        .map_err(|error| pliron::input_error_noloc!("MirSetDiscriminantOp: {error}"))?
    else {
        rewriter.erase_operation(ctx, op);
        return Ok(());
    };

    let tag_offset = enum_ty.tag_offset();
    let mir_ty: TypeHandle = pliron::r#type::Type::register_instance(enum_ty.clone(), ctx).into();
    let slot_map = build_enum_slot_map(ctx, mir_ty).map_err(anyhow_to_pliron)?;
    let carrier_ty = slot_map.carrier_llvm_ty.ok_or_else(|| {
        pliron::input_error_noloc!("MirSetDiscriminantOp physical write has no carrier type")
    })?;
    let carrier = emit_carrier_constant(ctx, rewriter, &enum_ty, carrier_ty, bits)?;
    let carrier_ptr = enum_byte_gep(ctx, rewriter, enum_ptr, tag_offset);
    let store_op = llvm::StoreOp::new(ctx, carrier, carrier_ptr);
    rewriter.insert_operation(ctx, store_op.get_operation());

    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert `mir.get_discriminant` (reading which variant is alive) to
/// `llvm.extractvalue`.
///
/// Direct layouts read the tag from the slot map's carrier slot. Niche
/// layouts decode rustc's wrapping carrier range; Single layouts materialize
/// their one logical discriminant as a constant. No slot number is assumed.
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

    let enum_ty: MirEnumType = operands_info
        .lookup_most_recent_of_type::<MirEnumType>(ctx, enum_val)
        .map(|ty| ty.clone())
        .ok_or_else(|| pliron::input_error_noloc!("Expected MirEnumType for discriminant read"))?;
    let mir_ty: TypeHandle = pliron::r#type::Type::register_instance(enum_ty.clone(), ctx).into();
    let slot_map = build_enum_slot_map(ctx, mir_ty).map_err(anyhow_to_pliron)?;
    let logical_ty = convert_type(ctx, enum_ty.discriminant_ty).map_err(anyhow_to_pliron)?;
    let logical_width = logical_ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(IntegerType::width)
        .ok_or_else(|| {
            pliron::input_error_noloc!("MirGetDiscriminantOp logical result must be integer")
        })?;

    let result = match enum_ty.layout_kind {
        EnumLayoutKind::Direct => {
            let slot = slot_map.carrier_slot.ok_or_else(|| {
                pliron::input_error_noloc!("Direct enum has no physical carrier slot")
            })?;
            let extract = llvm::ExtractValueOp::new(ctx, enum_val, vec![slot])?;
            rewriter.insert_operation(ctx, extract.get_operation());
            extract.get_operation().deref(ctx).get_result(0)
        }
        EnumLayoutKind::Single => {
            if enum_ty.variant_is_inhabited(enum_ty.single_variant as usize) != Some(true) {
                return pliron::input_err_noloc!(
                    "Cannot read discriminant of an uninhabited single-variant enum"
                );
            }
            let value = *enum_ty
                .variant_discriminants
                .get(enum_ty.single_variant as usize)
                .ok_or_else(|| {
                    pliron::input_error_noloc!("Single enum has no declared discriminant")
                })?;
            emit_integer_constant(ctx, rewriter, logical_width, u128::from(value))
        }
        EnumLayoutKind::Niche => {
            let slot = slot_map.carrier_slot.ok_or_else(|| {
                pliron::input_error_noloc!("Niche enum has no physical carrier slot")
            })?;
            let extract = llvm::ExtractValueOp::new(ctx, enum_val, vec![slot])?;
            rewriter.insert_operation(ctx, extract.get_operation());
            let carrier = extract.get_operation().deref(ctx).get_result(0);
            let carrier_int_ty: TypeHandle =
                IntegerType::get(ctx, enum_ty.carrier_width, Signedness::Signless).into();
            let carrier_int = if enum_ty.carrier_kind == EnumCarrierKind::Pointer {
                let cast = llvm::PtrToIntOp::new(ctx, carrier, carrier_int_ty);
                rewriter.insert_operation(ctx, cast.get_operation());
                cast.get_operation().deref(ctx).get_result(0)
            } else {
                carrier
            };

            let niche_start =
                emit_integer_constant(ctx, rewriter, enum_ty.carrier_width, enum_ty.niche_start());
            let relative = llvm::SubOp::new_with_overflow_flag(
                ctx,
                carrier_int,
                niche_start,
                IntegerOverflowFlagsAttr::default(),
            )
            .get_operation();
            rewriter.insert_operation(ctx, relative);
            let relative_val = relative.deref(ctx).get_result(0);
            let max = emit_integer_constant(
                ctx,
                rewriter,
                enum_ty.carrier_width,
                u128::from(enum_ty.niche_variant_end - enum_ty.niche_variant_start),
            );
            let in_range =
                llvm::ICmpOp::new(ctx, ICmpPredicateAttr::ULE, relative_val, max).get_operation();
            rewriter.insert_operation(ctx, in_range);
            // The range test is carrier-width wrapping arithmetic, exactly
            // like rustc. Variant indices themselves live at the logical
            // discriminant width: the start index may be larger than the
            // carrier can represent (e.g. variants 298..=299 in an i8 niche).
            let logical_relative = match enum_ty.carrier_width.cmp(&logical_width) {
                std::cmp::Ordering::Equal => relative_val,
                std::cmp::Ordering::Greater => {
                    let cast = llvm::TruncOp::new(ctx, relative_val, logical_ty).get_operation();
                    rewriter.insert_operation(ctx, cast);
                    cast.deref(ctx).get_result(0)
                }
                std::cmp::Ordering::Less => {
                    // LLVM's zext op requires its explicit `nneg` flag even
                    // when the flag is false. Niche decoding is ordinary
                    // unsigned extension, so it must never claim nneg.
                    let cast = llvm::ZExtOp::new_with_nneg(ctx, relative_val, logical_ty, false)
                        .get_operation();
                    rewriter.insert_operation(ctx, cast);
                    cast.deref(ctx).get_result(0)
                }
            };
            let niche_base = emit_integer_constant(
                ctx,
                rewriter,
                logical_width,
                u128::from(enum_ty.niche_variant_start),
            );
            let niche_variant = llvm::AddOp::new_with_overflow_flag(
                ctx,
                logical_relative,
                niche_base,
                IntegerOverflowFlagsAttr::default(),
            )
            .get_operation();
            rewriter.insert_operation(ctx, niche_variant);
            let untagged = emit_integer_constant(
                ctx,
                rewriter,
                logical_width,
                u128::from(enum_ty.untagged_variant),
            );
            let in_range_value = in_range.deref(ctx).get_result(0);
            let niche_variant_value = niche_variant.deref(ctx).get_result(0);
            let select = llvm::SelectOp::new(ctx, in_range_value, niche_variant_value, untagged)
                .get_operation();
            rewriter.insert_operation(ctx, select);
            select.deref(ctx).get_result(0)
        }
        EnumLayoutKind::Empty => {
            return pliron::input_err_noloc!("Cannot read discriminant of uninhabited enum");
        }
        _ => {
            return pliron::input_err_noloc!(
                "Cannot read discriminant of enum with unknown physical layout"
            );
        }
    };

    rewriter.replace_operation_with_values(ctx, op, vec![result]);

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
    use dialect_mir::attributes::{FieldIndexAttr, MirCastKindAttr, VariantIndexAttr};
    use dialect_mir::ops as mir;
    use dialect_mir::types::{
        EnumEncoding, EnumVariant, MirPtrType, MirSliceType, MirStructType, MirTupleType,
    };
    use llvm_export::types as llvm_types;
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
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "OrderingLike".to_string(),
            discr_ty,
            vec![255, 0, 1],
            vec![
                EnumVariant::unit("Less".to_string()),
                EnumVariant::unit("Equal".to_string()),
                EnumVariant::unit("Greater".to_string()),
            ],
            0,
            1,
            1,
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

    #[test]
    fn nested_pointer_niche_tuple_construct_extract_and_discriminant_lower() {
        let mut ctx = make_ctx();
        let logical: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signed).into();
        let index: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, pointee, false).into();
        let tuple_ty: TypeHandle = MirTupleType::get(&mut ctx, vec![index, pointer]).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "Option".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![tuple_ty], vec![0], vec![16]),
            ],
            EnumEncoding {
                tag_offset: 8,
                total_size: 16,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Pointer,
                carrier_width: 64,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();
        let slot_map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
        assert_eq!(slot_map.carrier_slot, Some(1));
        assert_eq!(slot_map.field_slots, vec![None]);
        let lowered_tuple = convert_type(&mut ctx, tuple_ty).unwrap();

        let (module, block) = build_kernel(&mut ctx, vec![index, pointer], vec![]);
        let index_value = block.deref(&ctx).get_argument(0);
        let pointer_value = block.deref(&ctx).get_argument(1);
        let tuple = Operation::new(
            &mut ctx,
            mir::MirConstructTupleOp::get_concrete_op_info(),
            vec![tuple_ty],
            vec![index_value, pointer_value],
            vec![],
            0,
        );
        tuple.insert_at_back(block, &ctx);
        let tuple_value = tuple.deref(&ctx).get_result(0);

        let construct = Operation::new(
            &mut ctx,
            mir::MirConstructEnumOp::get_concrete_op_info(),
            vec![enum_ty],
            vec![tuple_value],
            vec![],
            0,
        );
        mir::MirConstructEnumOp::new(construct)
            .set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(1));
        construct.insert_at_back(block, &ctx);
        let enum_value = construct.deref(&ctx).get_result(0);

        let payload = Operation::new(
            &mut ctx,
            mir::MirEnumPayloadOp::get_concrete_op_info(),
            vec![tuple_ty],
            vec![enum_value],
            vec![],
            0,
        );
        mir::MirEnumPayloadOp::new(payload)
            .set_attr_payload_variant_index(&ctx, VariantIndexAttr(1));
        mir::MirEnumPayloadOp::new(payload).set_attr_payload_field_index(&ctx, FieldIndexAttr(0));
        payload.insert_at_back(block, &ctx);

        let discriminant = Operation::new(
            &mut ctx,
            mir::MirGetDiscriminantOp::get_concrete_op_info(),
            vec![logical],
            vec![enum_value],
            vec![],
            0,
        );
        discriminant.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
        let body = kernel_blocks(&ctx, module);
        assert_eq!(
            count_ops::<llvm::IntToPtrOp>(&ctx, &body),
            0,
            "constructing the untagged Some payload must not recreate its pointer from bits"
        );
        assert_eq!(
            count_ops::<llvm::PtrToIntOp>(&ctx, &body),
            1,
            "reading the pointer niche should inspect the carrier exactly once"
        );
        assert_eq!(
            count_ops::<llvm::StoreOp>(&ctx, &body),
            3,
            "construction and extraction should each spill the enum, plus one tuple payload store"
        );
        assert_eq!(
            count_ops::<llvm::LoadOp>(&ctx, &body),
            2,
            "construction should reload the enum and extraction should load the tuple payload"
        );
        assert_eq!(
            find_all::<llvm::StoreOp>(&ctx, &body)
                .iter()
                .filter(|store| store.get_operand_value(&ctx).get_type(&ctx) == lowered_tuple)
                .count(),
            1,
            "the complete {{i64, ptr}} payload must be written into the enum storage"
        );
        assert_eq!(
            find_all::<llvm::LoadOp>(&ctx, &body)
                .iter()
                .filter(|load| {
                    load.get_operation()
                        .deref(&ctx)
                        .get_result(0)
                        .get_type(&ctx)
                        == lowered_tuple
                })
                .count(),
            1,
            "payload extraction must read the complete {{i64, ptr}} tuple back"
        );
    }

    /// SetDiscriminant must use the slot map instead of assuming that the tag
    /// is field zero, and its GEP must retain the source pointer's GPU address
    /// space. This shape puts an i64 payload first and an i8 tag above the
    /// u32 range, proving the byte GEP does not truncate offsets to u32.
    #[test]
    fn set_discriminant_uses_tag_slot_and_preserves_shared_address_space() {
        use llvm_export::ops::GepIndex;
        use llvm_export::types::{PointerType, address_space};

        let mut ctx = make_ctx();
        let discr_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let payload_a: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let payload_b: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let tag_offset = u64::from(u32::MAX) + 1;
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "TagAfterPayload".to_string(),
            discr_ty,
            vec![3, 7],
            vec![
                EnumVariant::new_with_layout("A".to_string(), vec![payload_a], vec![0], vec![8]),
                EnumVariant::new_with_layout("B".to_string(), vec![payload_b], vec![0], vec![8]),
            ],
            tag_offset,
            tag_offset + 8,
            8,
        )
        .into();
        let ptr_ty: TypeHandle = MirPtrType::get_shared(&mut ctx, enum_ty, true).into();

        let (module_ptr, block) = build_kernel(&mut ctx, vec![ptr_ty], vec![]);
        let enum_ptr = block.deref(&ctx).get_argument(0);
        let set = Operation::new(
            &mut ctx,
            mir::MirSetDiscriminantOp::get_concrete_op_info(),
            vec![],
            vec![enum_ptr],
            vec![],
            0,
        );
        mir::MirSetDiscriminantOp::new(set)
            .set_attr_set_discriminant_variant_index(&ctx, VariantIndexAttr(1));
        set.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        assert_eq!(count_ops::<mir::MirSetDiscriminantOp>(&ctx, &body), 0);
        assert_eq!(count_ops::<llvm::GetElementPtrOp>(&ctx, &body), 1);
        assert_eq!(count_ops::<llvm::StoreOp>(&ctx, &body), 1);

        let gep = find_first::<llvm::GetElementPtrOp>(&ctx, &body).unwrap();
        let indices = gep.indices(&ctx);
        assert!(matches!(indices.as_slice(), [GepIndex::Value(_)]));
        let GepIndex::Value(offset) = indices[0] else {
            unreachable!()
        };
        let offset_def = offset.defining_op().expect("byte offset must be constant");
        let offset_constant = Operation::get_op::<llvm::ConstantOp>(offset_def, &ctx)
            .expect("byte offset must be an LLVM constant");
        let offset_attr = offset_constant.get_value(&ctx);
        assert_eq!(
            offset_attr
                .downcast_ref::<IntegerAttr>()
                .expect("byte offset must be integer")
                .value()
                .to_u64(),
            tag_offset,
            "the write must use rustc's absolute carrier byte offset"
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

    fn unit_niche_enum(
        ctx: &mut Context,
        carrier: (EnumCarrierKind, u32, u32),
        niche_start: u128,
        niche_range: std::ops::RangeInclusive<u32>,
        untagged_variant: u32,
        inhabited: Vec<u8>,
    ) -> TypeHandle {
        let (carrier_kind, carrier_width, carrier_address_space) = carrier;
        let logical_width = if inhabited.len() > u8::MAX as usize {
            16
        } else {
            8
        };
        let logical_ty: TypeHandle =
            IntegerType::get(ctx, logical_width, Signedness::Unsigned).into();
        let variants = (0..inhabited.len())
            .map(|index| EnumVariant::unit(format!("V{index}")))
            .collect::<Vec<_>>();
        let discriminants = (0..inhabited.len() as u64).collect();
        let carrier_size = u64::from(carrier_width).div_ceil(8);
        let carrier_align = carrier_size.next_power_of_two().min(16);
        MirEnumType::get_with_encoding(
            ctx,
            "UnitNiche".into(),
            logical_ty,
            discriminants,
            variants,
            EnumEncoding {
                tag_offset: 0,
                total_size: carrier_size,
                abi_align: carrier_align,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind,
                carrier_width,
                carrier_address_space,
                niche_start,
                niche_variant_start: *niche_range.start(),
                niche_variant_end: *niche_range.end(),
                untagged_variant,
                variant_inhabited: inhabited,
                ..EnumEncoding::default()
            },
        )
        .into()
    }

    fn over_aligned_tuple_ty(ctx: &mut Context) -> TypeHandle {
        let byte: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        let marker: TypeHandle = MirStructType::get_with_full_layout(
            ctx,
            "Align32".into(),
            vec![],
            vec![],
            vec![],
            vec![],
            0,
            32,
        )
        .into();
        MirTupleType::get_with_layout(ctx, vec![marker, byte], vec![0, 1], vec![0, 0], 32, 32)
            .into()
    }

    #[test]
    fn dynamic_array_extract_preserves_recursive_element_alignment() {
        let mut ctx = make_ctx();
        let tuple_ty = over_aligned_tuple_ty(&mut ctx);
        let inner: TypeHandle = MirArrayType::get(&mut ctx, tuple_ty, 2).into();
        let outer: TypeHandle = MirArrayType::get(&mut ctx, inner, 3).into();
        let index_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let (module_ptr, block) = build_kernel(&mut ctx, vec![index_ty], vec![]);
        let index = block.deref(&ctx).get_argument(0);

        let undef = mir::MirUndefOp::new(&mut ctx, outer);
        undef.get_operation().insert_at_back(block, &ctx);
        let array = undef.get_operation().deref(&ctx).get_result(0);
        let extract = Operation::new(
            &mut ctx,
            mir::MirExtractArrayElementOp::get_concrete_op_info(),
            vec![inner],
            vec![array, index],
            vec![],
            0,
        );
        extract.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let alloca = find_first::<llvm::AllocaOp>(&ctx, &body).expect("expected llvm.alloca");
        let store = find_first::<llvm::StoreOp>(&ctx, &body).expect("expected llvm.store");
        let load = find_first::<llvm::LoadOp>(&ctx, &body).expect("expected llvm.load");
        for memory_op in [
            alloca.get_operation(),
            store.get_operation(),
            load.get_operation(),
        ] {
            assert_eq!(llvm_export::ops::op_alignment(&ctx, memory_op), Some(32));
        }
    }

    #[test]
    fn niche_encoding_handles_untagged_inside_range_and_u128_wrap() {
        let mut ctx = make_ctx();
        let inside = unit_niche_enum(
            &mut ctx,
            (EnumCarrierKind::Integer, 8, 0),
            42,
            0..=1,
            1,
            vec![1, 1],
        );
        let inside_ref = inside.deref(&ctx);
        let inside_enum = inside_ref.downcast_ref::<MirEnumType>().unwrap();
        assert_eq!(enum_carrier_bits_for_variant(inside_enum, 0), Ok(Some(42)));
        assert_eq!(
            enum_carrier_bits_for_variant(inside_enum, 1),
            Ok(None),
            "the untagged variant is a no-op even when its index is in the niche range"
        );
        drop(inside_ref);

        let wrapping = unit_niche_enum(
            &mut ctx,
            (EnumCarrierKind::Integer, 128, 0),
            u128::MAX,
            0..=1,
            0,
            vec![1, 1],
        );
        let wrapping_ref = wrapping.deref(&ctx);
        let wrapping_enum = wrapping_ref.downcast_ref::<MirEnumType>().unwrap();
        assert_eq!(
            enum_carrier_bits_for_variant(wrapping_enum, 1),
            Ok(Some(0)),
            "niche arithmetic must wrap across the full u128 carrier"
        );
    }

    #[test]
    fn set_discriminant_niche_writes_only_tagged_variant_carrier() {
        for (target, expected_stores) in [(0, 1), (1, 0)] {
            let mut ctx = make_ctx();
            let enum_ty = unit_niche_enum(
                &mut ctx,
                (EnumCarrierKind::Integer, 8, 0),
                0,
                0..=0,
                1,
                vec![1, 1],
            );
            let ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, enum_ty, true).into();
            let (module, block) = build_kernel(&mut ctx, vec![ptr_ty], vec![]);
            let ptr = block.deref(&ctx).get_argument(0);
            let set = Operation::new(
                &mut ctx,
                mir::MirSetDiscriminantOp::get_concrete_op_info(),
                vec![],
                vec![ptr],
                vec![],
                0,
            );
            mir::MirSetDiscriminantOp::new(set)
                .set_attr_set_discriminant_variant_index(&ctx, VariantIndexAttr(target));
            set.insert_at_back(block, &ctx);
            append_mir_return(&mut ctx, block, vec![]);

            crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
            let body = kernel_blocks(&ctx, module);
            assert_eq!(count_ops::<llvm::StoreOp>(&ctx, &body), expected_stores);
            assert_eq!(count_ops::<mir::MirSetDiscriminantOp>(&ctx, &body), 0);
        }
    }

    #[test]
    fn shared_pointer_niche_carrier_rejects_target_dependent_width() {
        let mut ctx = make_ctx();
        let enum_ty = unit_niche_enum(
            &mut ctx,
            (EnumCarrierKind::Pointer, 64, 3),
            0,
            0..=0,
            1,
            vec![1, 1],
        );
        let error = build_enum_slot_map(&mut ctx, enum_ty)
            .err()
            .expect("shared pointer carrier must reject");
        assert!(
            error.to_string().contains("target-mode dependent"),
            "{error}"
        );
    }

    #[test]
    fn option_bool_uses_i8_carrier_and_spills_i1_payload() {
        let mut ctx = make_ctx();
        let logical: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let bool_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "OptionBool".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![bool_ty], vec![0], vec![1]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 1,
                abi_align: 1,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Integer,
                carrier_width: 8,
                niche_start: 2,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();
        let slot_map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
        assert_eq!(
            slot_map
                .carrier_llvm_ty
                .unwrap()
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .unwrap()
                .width(),
            8,
            "Option<bool>'s memory carrier is i8, not bool's semantic i1"
        );
        assert_eq!(slot_map.field_slots, vec![None]);

        let (module, block) = build_kernel(&mut ctx, vec![bool_ty], vec![enum_ty]);
        let payload = block.deref(&ctx).get_argument(0);
        let construct = Operation::new(
            &mut ctx,
            mir::MirConstructEnumOp::get_concrete_op_info(),
            vec![enum_ty],
            vec![payload],
            vec![],
            0,
        );
        mir::MirConstructEnumOp::new(construct)
            .set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(1));
        construct.insert_at_back(block, &ctx);
        let result = construct.deref(&ctx).get_result(0);
        append_mir_return(&mut ctx, block, vec![result]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
        let body = kernel_blocks(&ctx, module);
        assert_eq!(count_ops::<llvm::InsertValueOp>(&ctx, &body), 0);
        assert!(find_all::<llvm::ZExtOp>(&ctx, &body).iter().any(|zext| {
            zext.get_operation()
                .deref(&ctx)
                .get_operand(0)
                .get_type(&ctx)
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .is_some_and(|integer| integer.width() == 1)
                && zext
                    .get_operation()
                    .deref(&ctx)
                    .get_result(0)
                    .get_type(&ctx)
                    .deref(&ctx)
                    .downcast_ref::<IntegerType>()
                    .is_some_and(|integer| integer.width() == 8)
        }));
        assert!(find_all::<llvm::StoreOp>(&ctx, &body).iter().any(|store| {
            store
                .get_operand_value(&ctx)
                .get_type(&ctx)
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .is_some_and(|integer| integer.width() == 8)
        }));
    }

    #[test]
    fn direct_bool_uses_i8_storage_and_spills_i1_payload() {
        let mut ctx = make_ctx();
        let tag: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let bool_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "DirectBool".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::new_with_layout("A".into(), vec![bool_ty], vec![4], vec![1]),
                EnumVariant::unit("B".into()),
            ],
            0,
            8,
            4,
        )
        .into();
        let slot_map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
        assert_eq!(slot_map.field_slots, vec![None]);
        let storage_fields = slot_map
            .llvm_struct_ty
            .deref(&ctx)
            .downcast_ref::<llvm_types::StructType>()
            .expect("enum storage must be an LLVM struct")
            .fields()
            .collect::<Vec<_>>();
        assert_eq!(
            storage_fields[1]
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .map(IntegerType::width),
            Some(8),
            "the standalone Rust bool byte must use physical i8 storage"
        );

        let (module, block) = build_kernel(&mut ctx, vec![bool_ty], vec![enum_ty]);
        let payload = block.deref(&ctx).get_argument(0);
        let construct = Operation::new(
            &mut ctx,
            mir::MirConstructEnumOp::get_concrete_op_info(),
            vec![enum_ty],
            vec![payload],
            vec![],
            0,
        );
        mir::MirConstructEnumOp::new(construct)
            .set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(0));
        construct.insert_at_back(block, &ctx);
        let result = construct.deref(&ctx).get_result(0);
        append_mir_return(&mut ctx, block, vec![result]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
        let body = kernel_blocks(&ctx, module);
        assert_eq!(
            count_ops::<llvm::InsertValueOp>(&ctx, &body),
            1,
            "only the direct tag should be inserted as an SSA struct field"
        );
        assert!(find_all::<llvm::ZExtOp>(&ctx, &body).iter().any(|zext| {
            zext.get_operation()
                .deref(&ctx)
                .get_operand(0)
                .get_type(&ctx)
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .is_some_and(|integer| integer.width() == 1)
                && zext
                    .get_operation()
                    .deref(&ctx)
                    .get_result(0)
                    .get_type(&ctx)
                    .deref(&ctx)
                    .downcast_ref::<IntegerType>()
                    .is_some_and(|integer| integer.width() == 8)
        }));
        assert!(find_all::<llvm::StoreOp>(&ctx, &body).iter().any(|store| {
            store
                .get_operand_value(&ctx)
                .get_type(&ctx)
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .is_some_and(|integer| integer.width() == 8)
        }));
    }

    #[test]
    fn later_field_niche_set_writes_exact_carrier_offset() {
        use llvm_export::ops::GepIndex;

        let mut ctx = make_ctx();
        let logical: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let wrapper: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Wrapper".into(),
            vec!["pad".into(), "nz".into()],
            vec![u32_ty, u32_ty],
            vec![0, 1],
            vec![0, 4],
            8,
            4,
        )
        .into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "MaybeWrapper".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![wrapper], vec![0], vec![8]),
            ],
            EnumEncoding {
                tag_offset: 4,
                total_size: 8,
                abi_align: 4,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Integer,
                carrier_width: 32,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();
        let ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, enum_ty, true).into();
        let (module, block) = build_kernel(&mut ctx, vec![ptr_ty], vec![]);
        let ptr = block.deref(&ctx).get_argument(0);
        let set = Operation::new(
            &mut ctx,
            mir::MirSetDiscriminantOp::get_concrete_op_info(),
            vec![],
            vec![ptr],
            vec![],
            0,
        );
        mir::MirSetDiscriminantOp::new(set)
            .set_attr_set_discriminant_variant_index(&ctx, VariantIndexAttr(0));
        set.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
        let body = kernel_blocks(&ctx, module);
        let gep = find_first::<llvm::GetElementPtrOp>(&ctx, &body).unwrap();
        let indices = gep.indices(&ctx);
        let [GepIndex::Value(offset)] = indices.as_slice() else {
            panic!("carrier access must use a byte-offset SSA value");
        };
        let constant =
            Operation::get_op::<llvm::ConstantOp>(offset.defining_op().unwrap(), &ctx).unwrap();
        assert_eq!(
            constant
                .get_value(&ctx)
                .downcast_ref::<IntegerAttr>()
                .unwrap()
                .value()
                .to_u64(),
            4
        );
    }

    #[test]
    fn get_niche_discriminant_adds_large_range_start_at_logical_width() {
        let mut ctx = make_ctx();
        let enum_ty = unit_niche_enum(
            &mut ctx,
            (EnumCarrierKind::Integer, 8, 0),
            0,
            298..=299,
            0,
            {
                let mut inhabited = vec![0; 300];
                inhabited[0] = 1;
                inhabited[298] = 1;
                inhabited[299] = 1;
                inhabited
            },
        );
        let logical_ty: TypeHandle = IntegerType::get(&ctx, 16, Signedness::Unsigned).into();
        let (module, block) = build_kernel(&mut ctx, vec![enum_ty], vec![logical_ty]);
        let value = block.deref(&ctx).get_argument(0);
        let get = Operation::new(
            &mut ctx,
            mir::MirGetDiscriminantOp::get_concrete_op_info(),
            vec![logical_ty],
            vec![value],
            vec![],
            0,
        );
        get.insert_at_back(block, &ctx);
        let result = get.deref(&ctx).get_result(0);
        append_mir_return(&mut ctx, block, vec![result]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
        let body = kernel_blocks(&ctx, module);
        assert_eq!(count_ops::<llvm::ZExtOp>(&ctx, &body), 1);
        let add = find_first::<llvm::AddOp>(&ctx, &body).expect("logical variant add");
        for operand in add.get_operation().deref(&ctx).operands() {
            assert_eq!(
                operand
                    .get_type(&ctx)
                    .deref(&ctx)
                    .downcast_ref::<IntegerType>()
                    .unwrap()
                    .width(),
                16,
                "variant-index arithmetic must occur at logical width"
            );
        }
    }

    #[test]
    fn direct_negative_discriminant_read_remains_signed_for_widening() {
        let mut ctx = make_ctx();
        let i8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Signed).into();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signed).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "Negative".into(),
            i8_ty,
            vec![255, 0],
            vec![EnumVariant::unit("N".into()), EnumVariant::unit("Z".into())],
            0,
            1,
            1,
        )
        .into();
        let (module, block) = build_kernel(&mut ctx, vec![], vec![i32_ty]);
        let construct = Operation::new(
            &mut ctx,
            mir::MirConstructEnumOp::get_concrete_op_info(),
            vec![enum_ty],
            vec![],
            vec![],
            0,
        );
        mir::MirConstructEnumOp::new(construct)
            .set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(0));
        construct.insert_at_back(block, &ctx);
        let enum_value = construct.deref(&ctx).get_result(0);
        let get = Operation::new(
            &mut ctx,
            mir::MirGetDiscriminantOp::get_concrete_op_info(),
            vec![i8_ty],
            vec![enum_value],
            vec![],
            0,
        );
        get.insert_at_back(block, &ctx);
        let discriminant = get.deref(&ctx).get_result(0);
        let cast = Operation::new(
            &mut ctx,
            mir::MirCastOp::get_concrete_op_info(),
            vec![i32_ty],
            vec![discriminant],
            vec![],
            0,
        );
        mir::MirCastOp::new(cast).set_attr_cast_kind(&ctx, MirCastKindAttr::IntToInt);
        cast.insert_at_back(block, &ctx);
        let widened = cast.deref(&ctx).get_result(0);
        append_mir_return(&mut ctx, block, vec![widened]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
        let body = kernel_blocks(&ctx, module);
        assert_eq!(count_ops::<llvm::SExtOp>(&ctx, &body), 1);
        assert_eq!(count_ops::<llvm::ZExtOp>(&ctx, &body), 0);
    }

    #[test]
    fn single_layout_preserves_large_and_negative_declared_discriminants() {
        for (width, signedness, bits, widened_width, expects_sext) in [
            (16, Signedness::Unsigned, 1_000, 16, false),
            (8, Signedness::Signed, 251, 32, true),
        ] {
            let mut ctx = make_ctx();
            let logical: TypeHandle = IntegerType::get(&ctx, width, signedness).into();
            let destination: TypeHandle = IntegerType::get(&ctx, widened_width, signedness).into();
            let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
                &mut ctx,
                "Single".into(),
                logical,
                vec![bits],
                vec![EnumVariant::unit("Only".into())],
                EnumEncoding {
                    tag_offset: 0,
                    total_size: 0,
                    abi_align: 1,
                    layout_kind: EnumLayoutKind::Single,
                    variant_inhabited: vec![1],
                    ..EnumEncoding::default()
                },
            )
            .into();
            let (module, block) = build_kernel(&mut ctx, vec![], vec![destination]);
            let construct = Operation::new(
                &mut ctx,
                mir::MirConstructEnumOp::get_concrete_op_info(),
                vec![enum_ty],
                vec![],
                vec![],
                0,
            );
            mir::MirConstructEnumOp::new(construct)
                .set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(0));
            construct.insert_at_back(block, &ctx);
            let enum_value = construct.deref(&ctx).get_result(0);
            let get = Operation::new(
                &mut ctx,
                mir::MirGetDiscriminantOp::get_concrete_op_info(),
                vec![logical],
                vec![enum_value],
                vec![],
                0,
            );
            get.insert_at_back(block, &ctx);
            let discr = get.deref(&ctx).get_result(0);
            let returned = if widened_width == width {
                discr
            } else {
                let cast = Operation::new(
                    &mut ctx,
                    mir::MirCastOp::get_concrete_op_info(),
                    vec![destination],
                    vec![discr],
                    vec![],
                    0,
                );
                mir::MirCastOp::new(cast).set_attr_cast_kind(&ctx, MirCastKindAttr::IntToInt);
                cast.insert_at_back(block, &ctx);
                cast.deref(&ctx).get_result(0)
            };
            append_mir_return(&mut ctx, block, vec![returned]);

            crate::lower_mir_to_llvm(&mut ctx, module).expect("lowering failed");
            let body = kernel_blocks(&ctx, module);
            let found_bits = find_all::<llvm::ConstantOp>(&ctx, &body)
                .iter()
                .any(|constant| {
                    constant
                        .get_value(&ctx)
                        .downcast_ref::<IntegerAttr>()
                        .is_some_and(|value| value.value().to_u64() == bits)
                });
            assert!(
                found_bits,
                "single discriminant {bits} must not be truncated"
            );
            assert_eq!(count_ops::<llvm::SExtOp>(&ctx, &body) == 1, expects_sext);
        }
    }
}
