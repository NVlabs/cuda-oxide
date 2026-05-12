/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Aggregate operation conversion: `dialect-mir` → `dialect-llvm`.
//!
//! Converts `dialect-mir` aggregate operations (structs, tuples, enums) to
//! their `dialect-llvm` equivalents.
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

use crate::convert::types::{convert_type, is_zero_sized_type};
use dialect_llvm::ops as llvm;
use dialect_mir::ops::{
    MirConstructEnumOp, MirEnumPayloadOp, MirExtractFieldOp, MirFieldAddrOp, MirInsertFieldOp,
};
use dialect_mir::types::{MirArrayType, MirEnumType, MirPtrType, MirStructType, MirTupleType};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeObj, Typed};
use pliron::utils::apint::APInt;
use std::num::NonZeroUsize;

fn anyhow_to_pliron(e: anyhow::Error) -> pliron::result::Error {
    pliron::input_error_noloc!("{e}")
}

/// Convert `mir.extract_field` to `llvm.extractvalue`.
///
/// Handles scalar-lowered newtype case: if the operand is a scalar (e.g., `ThreadIndex`),
/// no extraction is needed.
///
/// Note: Zero-sized fields are stripped from LLVM structs, so we need to remap
/// MIR field indices to LLVM indices. If extracting a ZST field, we return undef.
///
/// For structs with reordered fields, declaration index is mapped to memory index first.
pub(crate) fn convert_extract_field(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let aggregate = op.deref(ctx).get_operand(0);
    let result = op.deref(ctx).get_result(0);

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

    let (field_types, mem_to_decl) = {
        if let Some(struct_ref) =
            operands_info.lookup_most_recent_of_type::<MirStructType>(ctx, aggregate)
        {
            (struct_ref.field_types.clone(), struct_ref.memory_order())
        } else if let Some(tuple_ref) =
            operands_info.lookup_most_recent_of_type::<MirTupleType>(ctx, aggregate)
        {
            let types = tuple_ref.get_types().to_vec();
            let identity: Vec<usize> = (0..types.len()).collect();
            (types, identity)
        } else {
            (vec![], vec![])
        }
    };

    let target_field_llvm_ty = if decl_index < field_types.len() {
        convert_type(ctx, field_types[decl_index]).map_err(anyhow_to_pliron)?
    } else {
        let result_ty = result.get_type(ctx);
        convert_type(ctx, result_ty).map_err(anyhow_to_pliron)?
    };

    if is_zero_sized_type(ctx, target_field_llvm_ty) {
        let undef_op = llvm::UndefOp::new(ctx, target_field_llvm_ty);
        rewriter.insert_operation(ctx, undef_op.get_operation());
        rewriter.replace_operation(ctx, op, undef_op.get_operation());
    } else {
        let mem_index = mem_to_decl
            .iter()
            .position(|&d| d == decl_index)
            .unwrap_or(decl_index);

        let llvm_index = if !field_types.is_empty() {
            let mut idx = 0u32;
            for i in 0..mem_index {
                let decl_idx = mem_to_decl[i];
                let llvm_ty = convert_type(ctx, field_types[decl_idx]).map_err(anyhow_to_pliron)?;
                if !is_zero_sized_type(ctx, llvm_ty) {
                    idx += 1;
                }
            }
            idx
        } else {
            mem_index as u32
        };

        let llvm_extract = llvm::ExtractValueOp::new(ctx, aggregate, vec![llvm_index])?;
        rewriter.insert_operation(ctx, llvm_extract.get_operation());
        rewriter.replace_operation(ctx, op, llvm_extract.get_operation());
    }

    Ok(())
}

/// Convert `mir.insert_field` to `llvm.insertvalue`.
///
/// Operands: `[aggregate, new_value]`
/// Returns a new aggregate with the field at `insert_index` replaced.
///
/// Note: Zero-sized fields are stripped from LLVM structs, so we need to remap
/// MIR field indices to LLVM indices. If inserting a ZST field, we return the
/// original aggregate unchanged.
///
/// For structs with reordered fields, declaration index is mapped to memory index first.
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

    enum AggregateKind {
        Array,
        Struct {
            field_types: Vec<Ptr<TypeObj>>,
            mem_to_decl: Vec<usize>,
        },
        Tuple {
            field_types: Vec<Ptr<TypeObj>>,
            mem_to_decl: Vec<usize>,
        },
        Other,
    }

    let aggregate_kind = {
        if let Some(struct_ref) =
            operands_info.lookup_most_recent_of_type::<MirStructType>(ctx, aggregate)
        {
            AggregateKind::Struct {
                field_types: struct_ref.field_types.clone(),
                mem_to_decl: struct_ref.memory_order(),
            }
        } else if let Some(tuple_ref) =
            operands_info.lookup_most_recent_of_type::<MirTupleType>(ctx, aggregate)
        {
            let types = tuple_ref.get_types().to_vec();
            let identity: Vec<usize> = (0..types.len()).collect();
            AggregateKind::Tuple {
                field_types: types,
                mem_to_decl: identity,
            }
        } else if operands_info
            .lookup_most_recent_of_type::<MirArrayType>(ctx, aggregate)
            .is_some()
        {
            AggregateKind::Array
        } else {
            AggregateKind::Other
        }
    };

    if matches!(aggregate_kind, AggregateKind::Array) {
        let llvm_insert =
            llvm::InsertValueOp::new(ctx, aggregate, new_value, vec![decl_index as u32]);
        rewriter.insert_operation(ctx, llvm_insert.get_operation());
        rewriter.replace_operation(ctx, op, llvm_insert.get_operation());
        return Ok(());
    }

    let (field_types, mem_to_decl): (Vec<Ptr<TypeObj>>, Vec<usize>) = match aggregate_kind {
        AggregateKind::Struct {
            field_types,
            mem_to_decl,
        } => (field_types, mem_to_decl),
        AggregateKind::Tuple {
            field_types,
            mem_to_decl,
        } => (field_types, mem_to_decl),
        _ => (vec![], vec![]),
    };

    let target_field_is_zst = if decl_index < field_types.len() {
        let llvm_ty = convert_type(ctx, field_types[decl_index]).map_err(anyhow_to_pliron)?;
        is_zero_sized_type(ctx, llvm_ty)
    } else {
        false
    };

    if target_field_is_zst {
        rewriter.replace_operation_with_values(ctx, op, vec![aggregate]);
    } else {
        let mem_index = mem_to_decl
            .iter()
            .position(|&d| d == decl_index)
            .unwrap_or(decl_index);

        let llvm_index = if !field_types.is_empty() {
            let mut idx = 0u32;
            for i in 0..mem_index {
                let decl_idx = mem_to_decl[i];
                let llvm_ty = convert_type(ctx, field_types[decl_idx]).map_err(anyhow_to_pliron)?;
                if !is_zero_sized_type(ctx, llvm_ty) {
                    idx += 1;
                }
            }
            idx
        } else {
            mem_index as u32
        };

        let llvm_insert = llvm::InsertValueOp::new(ctx, aggregate, new_value, vec![llvm_index]);
        rewriter.insert_operation(ctx, llvm_insert.get_operation());
        rewriter.replace_operation(ctx, op, llvm_insert.get_operation());
    }

    Ok(())
}

/// Convert `mir.construct_struct` to a chain of `llvm.insertvalue` operations.
///
/// Builds a struct by:
/// 1. Creating an `undef` value of the struct type
/// 2. Inserting each operand at its corresponding field index
///
/// Operand order matches field order in the struct type (declaration order).
/// LLVM struct is built in memory order (from MirStructType::memory_order()).
///
/// Note: Zero-sized types (ZST) like PhantomData are skipped, as they are
/// stripped from the LLVM struct type. We track the LLVM field index separately.
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

    let (field_types, mem_to_decl, has_explicit_layout) = {
        let ty_ref = result_ty.deref(ctx);
        let mir_struct_ty = match ty_ref.downcast_ref::<MirStructType>() {
            Some(s) => s,
            None => {
                return pliron::input_err_noloc!(
                    "MirConstructStructOp result type must be MirStructType"
                );
            }
        };
        (
            mir_struct_ty.field_types.clone(),
            mir_struct_ty.memory_order(),
            mir_struct_ty.has_explicit_layout(),
        )
    };

    let mut is_zst_by_decl = vec![false; field_types.len()];
    for (decl_idx, field_ty) in field_types.iter().enumerate() {
        let llvm_ty = convert_type(ctx, *field_ty).map_err(anyhow_to_pliron)?;
        is_zst_by_decl[decl_idx] = is_zero_sized_type(ctx, llvm_ty);
    }

    let llvm_struct_ty: Ptr<TypeObj> = if has_explicit_layout {
        convert_type(ctx, result_ty).map_err(anyhow_to_pliron)?
    } else {
        let mut llvm_field_types = Vec::with_capacity(field_types.len());
        for mem_idx in 0..field_types.len() {
            let decl_idx = mem_to_decl[mem_idx];
            if !is_zst_by_decl[decl_idx] {
                let llvm_ty = convert_type(ctx, field_types[decl_idx]).map_err(anyhow_to_pliron)?;
                llvm_field_types.push(llvm_ty);
            }
        }
        dialect_llvm::types::StructType::get_unnamed(ctx, llvm_field_types).into()
    };

    let undef_op = llvm::UndefOp::new(ctx, llvm_struct_ty);
    rewriter.insert_operation(ctx, undef_op.get_operation());
    let mut current_struct = undef_op.get_operation().deref(ctx).get_result(0);

    let mut llvm_idx = 0u32;
    let mut last_insert: Option<Ptr<Operation>> = None;
    for mem_idx in 0..field_types.len() {
        let decl_idx = mem_to_decl[mem_idx];
        if is_zst_by_decl[decl_idx] {
            continue;
        }

        let field_val = operands[decl_idx];

        let insert_op = llvm::InsertValueOp::new(ctx, current_struct, field_val, vec![llvm_idx]);
        rewriter.insert_operation(ctx, insert_op.get_operation());
        current_struct = insert_op.get_operation().deref(ctx).get_result(0);
        last_insert = Some(insert_op.get_operation());
        llvm_idx += 1;
    }

    match last_insert {
        Some(last_op) => rewriter.replace_operation(ctx, op, last_op),
        None => rewriter.replace_operation(ctx, op, undef_op.get_operation()),
    }

    Ok(())
}

/// Convert `mir.construct_tuple` to a chain of `llvm.insertvalue` operations.
///
/// Tuples are represented as LLVM structs. Same construction pattern as structs:
/// 1. Create `undef` of the tuple/struct type
/// 2. Insert each element at its index
///
/// Note: Zero-sized types (ZST) like PhantomData are skipped, as they are
/// stripped from the LLVM struct type. We track the LLVM element index separately.
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

    let element_types = {
        let ty_ref = result_ty.deref(ctx);
        match ty_ref.downcast_ref::<MirTupleType>() {
            Some(t) => t.get_types().to_vec(),
            None => {
                return pliron::input_err_noloc!(
                    "MirConstructTupleOp result type must be MirTupleType"
                );
            }
        }
    };

    let mut llvm_element_types = Vec::with_capacity(element_types.len());
    let mut is_zst = Vec::with_capacity(element_types.len());
    for elem_ty in &element_types {
        let llvm_ty = convert_type(ctx, *elem_ty).map_err(anyhow_to_pliron)?;
        let zst = is_zero_sized_type(ctx, llvm_ty);
        is_zst.push(zst);
        if !zst {
            llvm_element_types.push(llvm_ty);
        }
    }

    let llvm_struct_ty = dialect_llvm::types::StructType::get_unnamed(ctx, llvm_element_types);

    let undef_op = llvm::UndefOp::new(ctx, llvm_struct_ty.into());
    rewriter.insert_operation(ctx, undef_op.get_operation());
    let mut current_tuple = undef_op.get_operation().deref(ctx).get_result(0);

    let mut llvm_idx = 0u32;
    let mut last_insert: Option<Ptr<Operation>> = None;
    for (mir_idx, operand) in operands.iter().enumerate() {
        if is_zst[mir_idx] {
            continue;
        }

        let insert_op = llvm::InsertValueOp::new(ctx, current_tuple, *operand, vec![llvm_idx]);
        rewriter.insert_operation(ctx, insert_op.get_operation());
        current_tuple = insert_op.get_operation().deref(ctx).get_result(0);
        last_insert = Some(insert_op.get_operation());
        llvm_idx += 1;
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
    let llvm_array_ty = dialect_llvm::types::ArrayType::get(ctx, llvm_element_ty, array_size);

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
    let llvm_array_ty = dialect_llvm::types::ArrayType::get(ctx, llvm_element_ty, array_size);

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

    use dialect_llvm::ops::GepIndex;
    let gep_indices = vec![GepIndex::Constant(0), GepIndex::Value(index_val)];
    let gep_op = llvm::GetElementPtrOp::new(ctx, array_ptr, gep_indices, llvm_array_ty.into())?;
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

    let (discriminant_ty, variant_field_counts, all_field_types): (
        Ptr<TypeObj>,
        Vec<u32>,
        Vec<Ptr<TypeObj>>,
    ) = {
        let ty_ref = result_ty.deref(ctx);
        match ty_ref.downcast_ref::<MirEnumType>() {
            Some(e) => (
                e.discriminant_ty,
                e.variant_field_counts.clone(),
                e.all_field_types.clone(),
            ),
            None => {
                return pliron::input_err_noloc!(
                    "MirConstructEnumOp result type must be MirEnumType"
                );
            }
        }
    };

    let llvm_discriminant_ty = convert_type(ctx, discriminant_ty).map_err(anyhow_to_pliron)?;
    let mut llvm_payload_types = Vec::new();
    for field_ty in &all_field_types {
        llvm_payload_types.push(convert_type(ctx, *field_ty).map_err(anyhow_to_pliron)?);
    }

    let mut llvm_field_types = vec![llvm_discriminant_ty];
    llvm_field_types.extend(llvm_payload_types);
    let llvm_struct_ty = dialect_llvm::types::StructType::get_unnamed(ctx, llvm_field_types);

    let undef_op = llvm::UndefOp::new(ctx, llvm_struct_ty.into());
    rewriter.insert_operation(ctx, undef_op.get_operation());
    let mut current_struct = undef_op.get_operation().deref(ctx).get_result(0);

    let discr_width = llvm_discriminant_ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|t| t.width())
        .unwrap_or(8);
    let discr_apint = APInt::from_u64(
        variant_index as u64,
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
/// is always at index 0 in the LLVM struct representation.
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

    let (field_types, mem_to_decl, pointee_ty) = {
        let mir_ptr_pointee =
            match operands_info.lookup_most_recent_of_type::<MirPtrType>(ctx, ptr_operand) {
                Some(r) => r.pointee,
                None => {
                    return pliron::input_err_noloc!("MirFieldAddrOp operand must be pointer type");
                }
            };

        // Both struct and tuple pointees are supported — they're both
        // positionally-indexed aggregates. Tuples have no `mem_to_decl`
        // reordering (declaration order == memory order), so we
        // synthesize the identity mapping for them.
        let pointee_ref = mir_ptr_pointee.deref(ctx);
        if let Some(struct_ty) = pointee_ref.downcast_ref::<MirStructType>() {
            let ft = struct_ty.field_types.clone();
            let mtd = struct_ty.memory_order();
            (ft, mtd, mir_ptr_pointee)
        } else if let Some(tuple_ty) = pointee_ref.downcast_ref::<dialect_mir::types::MirTupleType>()
        {
            let ft: Vec<_> = tuple_ty.get_types().to_vec();
            let mtd: Vec<usize> = (0..ft.len()).collect();
            (ft, mtd, mir_ptr_pointee)
        } else {
            return pliron::input_err_noloc!(
                "MirFieldAddrOp pointer must point to struct or tuple type"
            );
        }
    };

    let mem_index = match mem_to_decl
        .iter()
        .position(|&decl_idx| decl_idx == field_index)
    {
        Some(idx) => idx,
        None => {
            return pliron::input_err_noloc!(
                "Field index {} not found in memory order mapping",
                field_index
            );
        }
    };

    // ZST predecessors get *dropped* from the LLVM struct (see
    // `MirStructType` lowering and `build_struct_with_explicit_padding`),
    // so the LLVM GEP index must skip them. `is_zero_sized_type` only
    // recognizes LLVM struct types, so we have to `convert_type` the
    // MIR field type first — otherwise EVERY predecessor counts and
    // the GEP index drifts past the LLVM struct's actual field count.
    let mut llvm_field_idx = 0u32;
    for i in 0..mem_index {
        let decl_idx = mem_to_decl[i];
        let llvm_field_ty =
            convert_type(ctx, field_types[decl_idx]).map_err(anyhow_to_pliron)?;
        if !is_zero_sized_type(ctx, llvm_field_ty) {
            llvm_field_idx += 1;
        }
    }

    let target_llvm_ty =
        convert_type(ctx, field_types[field_index]).map_err(anyhow_to_pliron)?;
    let target_is_zst = is_zero_sized_type(ctx, target_llvm_ty);
    if target_is_zst {
        rewriter.replace_operation_with_values(ctx, op, vec![ptr_operand]);
        return Ok(());
    }

    let llvm_struct_ty = convert_type(ctx, pointee_ty).map_err(anyhow_to_pliron)?;

    use dialect_llvm::ops::GepIndex;
    let gep_indices = vec![GepIndex::Constant(0), GepIndex::Constant(llvm_field_idx)];

    let gep_op = llvm::GetElementPtrOp::new(ctx, ptr_operand, gep_indices, llvm_struct_ty)?;
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

    use dialect_llvm::ops::GepIndex;
    let gep_indices = vec![GepIndex::Constant(0), GepIndex::Value(index)];

    let gep_op = llvm::GetElementPtrOp::new(ctx, arr_ptr, gep_indices, llvm_array_ty)?;
    rewriter.insert_operation(ctx, gep_op.get_operation());
    rewriter.replace_operation(ctx, op, gep_op.get_operation());

    Ok(())
}

#[cfg(test)]
mod tests {
    // TODO: Add unit tests for aggregate conversion
}
