/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thread and block indexing intrinsics.
//!
//! Handles translation of position-related intrinsics that query thread/block
//! identity and compute global indices.
//!
//! # Intrinsic Table
//!
//! | Intrinsic                  | NVVM Op                 | Description                                          |
//! |----------------------------|-------------------------|------------------------------------------------------|
//! | `threadIdx_x/y/z`          | `ReadPtxSregTidX/Y/Z`   | Thread ID within block                               |
//! | `blockIdx_x/y/z`           | `ReadPtxSregCtaidX/Y/Z` | Block ID within grid                                 |
//! | `blockDim_x/y/z`           | `ReadPtxSregNtidX/Y/Z`  | Block dimensions                                     |
//! | `index_1d()`               | Normal function call    | Global 1D thread index                               |
//! | `index_2d_row/col()`       | Normal function call    | 2D row/column indices                                |
//! | `index_2d::<S>()`          | Normal function call    | Const-stride 2D index (returns `Option<ThreadIndex>`)|
//! | `index_2d_runtime(s)`      | Normal function call    | Runtime-stride 2D index (caller-asserted)            |
//! | `get_thread_local()`       | `MirPtrOffsetOp`        | DisjointSlice element ptr                            |
//! | `len()`                    | `MirExtractFieldOp`     | Slice length extraction                              |
//!
//! # Index Formulas
//!
//! - `index_1d() = blockIdx.x * blockDim.x + threadIdx.x`
//! - `index_2d_row() = blockIdx.y * blockDim.y + threadIdx.y`
//! - `index_2d_col() = blockIdx.x * blockDim.x + threadIdx.x`
//! - `index_2d::<S>() = if col < S { Some(row * S + col) } else { None }`
//! - `index_2d_runtime(s) = if col < s { Some(row * s + col) } else { None }`

use super::super::helpers::{emit_store_result_and_goto, set_generated_intrinsic_marker};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_mir::attributes::MirCastKindAttr;
use dialect_mir::ops::{MirAddOp, MirCastOp, MirMulOp};
use dialect_nvvm::ops::{
    ReadPtxSregCtaidXOp, ReadPtxSregCtaidYOp, ReadPtxSregNtidXOp, ReadPtxSregNtidYOp,
    ReadPtxSregTidXOp, ReadPtxSregTidYOp,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::Typed;
use pliron::value::Value;
use rustc_public::mir;

fn generated_sreg_op(
    ctx: &mut Context,
    opid: (fn(Ptr<Operation>) -> pliron::op::OpObj, std::any::TypeId),
    result_type: pliron::r#type::TypeHandle,
    loc: Location,
) -> Ptr<Operation> {
    let op = Operation::new(ctx, opid, vec![result_type], vec![], vec![], 0);
    op.deref_mut(ctx).set_loc(loc);
    let op_name = Operation::get_opid(op, ctx).to_string();
    let marker = cuda_oxide_codegen::__private::generated_intrinsic_marker_by_op_name(&op_name)
        .unwrap_or_else(|| panic!("generated sreg op `{op_name}` has no generated target record"));
    set_generated_intrinsic_marker(ctx, op, marker);
    op
}

/// Emits `row * stride + col` for `index_2d::<S>()` and `index_2d_runtime(s)`.
///
/// Where `row = index_2d_row()` and `col = index_2d_col()`. The `stride`
/// is the const generic for `index_2d::<S>` and the runtime arg for
/// `index_2d_runtime`.
#[allow(clippy::too_many_arguments)]
pub fn emit_index_2d(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let u32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);
    let usize_type = types::get_usize_type(ctx);

    // Get the stride argument
    let (stride_val, mut last_op) = match &args[0] {
        mir::Operand::Copy(place) | mir::Operand::Move(place) => {
            rvalue::translate_place(ctx, body, place, value_map, block_ptr, prev_op, loc.clone())?
        }
        _ => {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(
                    "Constant stride in index_2d not yet supported".to_string()
                )
            );
        }
    };

    // Emit row = blockIdx.y * blockDim.y + threadIdx.y
    let tid_y_op = generated_sreg_op(
        ctx,
        ReadPtxSregTidYOp::get_concrete_op_info(),
        u32_type.to_handle(),
        loc.clone(),
    );
    match last_op {
        Some(prev) => tid_y_op.insert_after(ctx, prev),
        None => tid_y_op.insert_at_front(block_ptr, ctx),
    }
    let tid_y_val = tid_y_op.deref(ctx).get_result(0);
    last_op = Some(tid_y_op);

    let bid_y_op = generated_sreg_op(
        ctx,
        ReadPtxSregCtaidYOp::get_concrete_op_info(),
        u32_type.to_handle(),
        loc.clone(),
    );
    bid_y_op.insert_after(ctx, last_op.unwrap());
    let bid_y_val = bid_y_op.deref(ctx).get_result(0);
    last_op = Some(bid_y_op);

    let bdim_y_op = generated_sreg_op(
        ctx,
        ReadPtxSregNtidYOp::get_concrete_op_info(),
        u32_type.to_handle(),
        loc.clone(),
    );
    bdim_y_op.insert_after(ctx, last_op.unwrap());
    let bdim_y_val = bdim_y_op.deref(ctx).get_result(0);
    last_op = Some(bdim_y_op);

    let mul_y_op = Operation::new(
        ctx,
        MirMulOp::get_concrete_op_info(),
        vec![u32_type.to_handle()],
        vec![bid_y_val, bdim_y_val],
        vec![],
        0,
    );
    mul_y_op.deref_mut(ctx).set_loc(loc.clone());
    mul_y_op.insert_after(ctx, last_op.unwrap());
    let mul_y_val = mul_y_op.deref(ctx).get_result(0);
    last_op = Some(mul_y_op);

    let row_u32_op = Operation::new(
        ctx,
        MirAddOp::get_concrete_op_info(),
        vec![u32_type.to_handle()],
        vec![mul_y_val, tid_y_val],
        vec![],
        0,
    );
    row_u32_op.deref_mut(ctx).set_loc(loc.clone());
    row_u32_op.insert_after(ctx, last_op.unwrap());
    let row_u32_val = row_u32_op.deref(ctx).get_result(0);
    last_op = Some(row_u32_op);

    // Cast row to usize
    let row_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![usize_type.to_handle()],
        vec![row_u32_val],
        vec![],
        0,
    );
    row_op.deref_mut(ctx).set_loc(loc.clone());
    MirCastOp::new(row_op).set_attr_cast_kind(ctx, MirCastKindAttr::IntToInt);
    row_op.insert_after(ctx, last_op.unwrap());
    let row_val = row_op.deref(ctx).get_result(0);
    last_op = Some(row_op);

    // Emit col = blockIdx.x * blockDim.x + threadIdx.x
    let tid_x_op = generated_sreg_op(
        ctx,
        ReadPtxSregTidXOp::get_concrete_op_info(),
        u32_type.to_handle(),
        loc.clone(),
    );
    tid_x_op.insert_after(ctx, last_op.unwrap());
    let tid_x_val = tid_x_op.deref(ctx).get_result(0);
    last_op = Some(tid_x_op);

    let bid_x_op = generated_sreg_op(
        ctx,
        ReadPtxSregCtaidXOp::get_concrete_op_info(),
        u32_type.to_handle(),
        loc.clone(),
    );
    bid_x_op.insert_after(ctx, last_op.unwrap());
    let bid_x_val = bid_x_op.deref(ctx).get_result(0);
    last_op = Some(bid_x_op);

    let bdim_x_op = generated_sreg_op(
        ctx,
        ReadPtxSregNtidXOp::get_concrete_op_info(),
        u32_type.to_handle(),
        loc.clone(),
    );
    bdim_x_op.insert_after(ctx, last_op.unwrap());
    let bdim_x_val = bdim_x_op.deref(ctx).get_result(0);
    last_op = Some(bdim_x_op);

    let mul_x_op = Operation::new(
        ctx,
        MirMulOp::get_concrete_op_info(),
        vec![u32_type.to_handle()],
        vec![bid_x_val, bdim_x_val],
        vec![],
        0,
    );
    mul_x_op.deref_mut(ctx).set_loc(loc.clone());
    mul_x_op.insert_after(ctx, last_op.unwrap());
    let mul_x_val = mul_x_op.deref(ctx).get_result(0);
    last_op = Some(mul_x_op);

    let col_u32_op = Operation::new(
        ctx,
        MirAddOp::get_concrete_op_info(),
        vec![u32_type.to_handle()],
        vec![mul_x_val, tid_x_val],
        vec![],
        0,
    );
    col_u32_op.deref_mut(ctx).set_loc(loc.clone());
    col_u32_op.insert_after(ctx, last_op.unwrap());
    let col_u32_val = col_u32_op.deref(ctx).get_result(0);
    last_op = Some(col_u32_op);

    // Cast col to usize
    let col_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![usize_type.to_handle()],
        vec![col_u32_val],
        vec![],
        0,
    );
    col_op.deref_mut(ctx).set_loc(loc.clone());
    MirCastOp::new(col_op).set_attr_cast_kind(ctx, MirCastKindAttr::IntToInt);
    col_op.insert_after(ctx, last_op.unwrap());
    let col_val = col_op.deref(ctx).get_result(0);
    last_op = Some(col_op);

    // Compute row * stride
    let row_stride_op = Operation::new(
        ctx,
        MirMulOp::get_concrete_op_info(),
        vec![usize_type.to_handle()],
        vec![row_val, stride_val],
        vec![],
        0,
    );
    row_stride_op.deref_mut(ctx).set_loc(loc.clone());
    row_stride_op.insert_after(ctx, last_op.unwrap());
    let row_stride_val = row_stride_op.deref(ctx).get_result(0);
    last_op = Some(row_stride_op);

    // Compute (row * stride) + col
    let result_op = Operation::new(
        ctx,
        MirAddOp::get_concrete_op_info(),
        vec![usize_type.to_handle()],
        vec![row_stride_val, col_val],
        vec![],
        0,
    );
    result_op.deref_mut(ctx).set_loc(loc.clone());
    result_op.insert_after(ctx, last_op.unwrap());
    let result_val = result_op.deref(ctx).get_result(0);

    emit_store_result_and_goto(
        ctx,
        destination,
        result_val,
        target,
        block_ptr,
        result_op,
        value_map,
        block_map,
        loc,
        "Call terminator without target not supported",
    )
}

/// Load the `DisjointSlice` value behind a method receiver.
///
/// `DisjointSlice::len` has an `&self` receiver, so fully monomorphized MIR
/// passes a `mir.ptr<mir.disjoint_slice<T>>`. Keep that source-level contract
/// explicit: accept exactly one pointer layer whose pointee is the compiler's
/// `MirDisjointSliceType`, and reject every other shape instead of guessing.
fn load_disjoint_slice_receiver(
    ctx: &mut Context,
    receiver: Value,
    block_ptr: Ptr<BasicBlock>,
    last_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<(Value, Ptr<Operation>)> {
    let receiver_ty = receiver.get_type(ctx);
    let pointee = {
        let receiver_ty_obj = receiver_ty.deref(ctx);
        let pointee = receiver_ty_obj
            .downcast_ref::<dialect_mir::types::MirPtrType>()
            .map(|ptr_ty| ptr_ty.pointee);
        match pointee {
            Some(pointee)
                if pointee
                    .deref(ctx)
                    .downcast_ref::<dialect_mir::types::MirDisjointSliceType>()
                    .is_some() =>
            {
                pointee
            }
            _ => {
                return input_err!(
                    loc,
                    TranslationErr::type_error(
                        "DisjointSlice::len receiver must be a pointer to mir.disjoint_slice"
                            .to_string(),
                    )
                );
            }
        }
    };

    let load_op = Operation::new(
        ctx,
        dialect_mir::ops::MirLoadOp::get_concrete_op_info(),
        vec![pointee],
        vec![receiver],
        vec![],
        0,
    );
    load_op.deref_mut(ctx).set_loc(loc);
    match last_op {
        Some(prev) => load_op.insert_after(ctx, prev),
        None => load_op.insert_at_front(block_ptr, ctx),
    }

    let loaded_val = load_op.deref(ctx).get_result(0);
    Ok((loaded_val, load_op))
}

/// Emits `DisjointSlice::get_thread_local(&self, idx) -> &mut T`.
///
/// Computes a pointer to the element at `idx` within the slice. The DisjointSlice
/// type provides safe per-thread indexing into global memory.
///
/// # DisjointSlice Layout
///
/// ```text
/// struct DisjointSlice<T> {
///     ptr: *mut T,        // field 0 - base pointer
///     len: usize,         // field 1 - element count
///     _marker: PhantomData // field 2 - type marker (ZST)
/// }
/// ```
///
/// # Implementation
///
/// 1. Extract `ptr` field (index 0) from the slice
/// 2. Compute `ptr + idx` using `MirPtrOffsetOp`
/// 3. Return the offset pointer
///
/// # Arguments
///
/// - `args[0]`: `&mut DisjointSlice<T>` or `*mut DisjointSlice<T>`
/// - `args[1]`: `usize` - Index into the slice
///
/// # Returns
///
/// `*mut T` - Pointer to the element (generic address space)
#[allow(clippy::too_many_arguments)]
pub fn emit_get_thread_local(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    use dialect_mir::ops::MirPtrOffsetOp;

    // Args should be: [&mut DisjointSlice, usize]
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "get_thread_local expects 2 arguments, got {}",
                args.len()
            ))
        );
    }

    // Get the DisjointSlice value (arg 0)
    let (disjoint_slice_val, mut last_op) = match &args[0] {
        mir::Operand::Copy(place) | mir::Operand::Move(place) => {
            rvalue::translate_place(ctx, body, place, value_map, block_ptr, prev_op, loc.clone())?
        }
        _ => {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported("Constant DisjointSlice not supported".to_string())
            );
        }
    };

    // Get the index value (arg 1)
    let (index_val, last_op_after_index) = match &args[1] {
        mir::Operand::Copy(place) | mir::Operand::Move(place) => {
            rvalue::translate_place(ctx, body, place, value_map, block_ptr, last_op, loc.clone())?
        }
        _ => {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(
                    "Constant index in get_thread_local not yet supported".to_string()
                )
            );
        }
    };
    last_op = last_op_after_index;

    // Extract ptr field (field 0) from DisjointSlice
    // DisjointSlice layout: { ptr: *mut T, len: usize, _marker: PhantomData }
    let slice_ty = disjoint_slice_val.get_type(ctx);

    // Determine if we have a DisjointSlice value or a pointer to one
    enum SliceKind {
        Direct {
            element_ty: pliron::r#type::TypeHandle,
        },
        Pointer {
            pointee: pliron::r#type::TypeHandle,
            element_ty: pliron::r#type::TypeHandle,
        },
    }

    let slice_kind = {
        let slice_ty_obj = slice_ty.deref(ctx);
        if let Some(dst) = slice_ty_obj.downcast_ref::<dialect_mir::types::MirDisjointSliceType>() {
            SliceKind::Direct {
                element_ty: dst.element_type(),
            }
        } else if let Some(ptr_ty) = slice_ty_obj.downcast_ref::<dialect_mir::types::MirPtrType>() {
            let pointee = ptr_ty.pointee;
            let element_ty = pointee
                .deref(ctx)
                .downcast_ref::<dialect_mir::types::MirDisjointSliceType>()
                .map(|dst| dst.element_type())
                .unwrap_or_else(|| panic!("Expected pointer to DisjointSliceType"));
            SliceKind::Pointer {
                pointee,
                element_ty,
            }
        } else {
            panic!("Expected DisjointSliceType or pointer to it");
        }
    };

    // If we have a pointer to DisjointSlice, we need to load it first
    let (actual_slice_val, element_ty) = match slice_kind {
        SliceKind::Direct { element_ty } => (disjoint_slice_val, element_ty),
        SliceKind::Pointer {
            pointee,
            element_ty,
        } => {
            let load_op = Operation::new(
                ctx,
                dialect_mir::ops::MirLoadOp::get_concrete_op_info(),
                vec![pointee],
                vec![disjoint_slice_val],
                vec![],
                0,
            );
            load_op.deref_mut(ctx).set_loc(loc.clone());

            match last_op {
                Some(prev) => load_op.insert_after(ctx, prev),
                None => load_op.insert_at_front(block_ptr, ctx),
            }
            last_op = Some(load_op);

            let loaded_val = load_op.deref(ctx).get_result(0);
            (loaded_val, element_ty)
        }
    };

    // Use generic address space for DisjointSlice (global memory with per-thread indexing)
    let ptr_ty = dialect_mir::types::MirPtrType::get_generic(ctx, element_ty, true).into();

    let extract_ptr_op = Operation::new(
        ctx,
        dialect_mir::ops::MirExtractFieldOp::get_concrete_op_info(),
        vec![ptr_ty],
        vec![actual_slice_val],
        vec![],
        0,
    );
    extract_ptr_op.deref_mut(ctx).set_loc(loc.clone());

    let extract_ptr = dialect_mir::ops::MirExtractFieldOp::new(extract_ptr_op);
    extract_ptr.set_attr_index(ctx, dialect_mir::attributes::FieldIndexAttr(0));

    match last_op {
        Some(prev) => extract_ptr.get_operation().insert_after(ctx, prev),
        None => extract_ptr.get_operation().insert_at_front(block_ptr, ctx),
    }
    last_op = Some(extract_ptr.get_operation());

    let ptr_val = extract_ptr.get_operation().deref(ctx).get_result(0);

    // Compute ptr + idx using MirPtrOffsetOp
    let offset_op = Operation::new(
        ctx,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![ptr_ty],
        vec![ptr_val, index_val],
        vec![],
        0,
    );
    offset_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        offset_op.insert_after(ctx, prev);
    } else {
        offset_op.insert_at_front(block_ptr, ctx);
    }
    last_op = Some(offset_op);

    let result_ptr = offset_op.deref(ctx).get_result(0);

    let prev = last_op.expect("should have at least offset_op");
    emit_store_result_and_goto(
        ctx,
        destination,
        result_ptr,
        target,
        block_ptr,
        prev,
        value_map,
        block_map,
        loc,
        "get_thread_local call without target block",
    )
}

/// Emits `DisjointSlice::len()`: Extract the length field from a DisjointSlice.
///
/// # DisjointSlice Layout
///
/// ```text
/// struct DisjointSlice<T> {
///     ptr: *mut T,        // field 0
///     len: usize,         // field 1 ← extracted
///     _marker: PhantomData // field 2
/// }
/// ```
///
/// # Arguments
///
/// - `args[0]`: `&DisjointSlice<T>` - Reference to the slice
///
/// # Returns
///
/// `usize` - Number of elements in the slice
#[allow(clippy::too_many_arguments)]
pub fn emit_len(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    // Args should be: [&DisjointSlice]
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!("len expects 1 argument, got {}", args.len()))
        );
    }

    // Get the DisjointSlice value (arg 0)
    let (disjoint_slice_val, last_op) = match &args[0] {
        mir::Operand::Copy(place) | mir::Operand::Move(place) => {
            rvalue::translate_place(ctx, body, place, value_map, block_ptr, prev_op, loc.clone())?
        }
        _ => {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported("Constant DisjointSlice not supported".to_string(),)
            );
        }
    };

    let (disjoint_slice_val, load_op) =
        load_disjoint_slice_receiver(ctx, disjoint_slice_val, block_ptr, last_op, loc.clone())?;
    let mut last_op = Some(load_op);

    // Extract len field (field 1) from DisjointSlice
    // DisjointSlice layout: { ptr: *mut T, len: usize, _marker: PhantomData }
    // We need the result type (usize). In MIR lowering we map usize to i64 usually.
    let usize_ty = types::get_usize_type(ctx);

    let extract_len_op = Operation::new(
        ctx,
        dialect_mir::ops::MirExtractFieldOp::get_concrete_op_info(),
        vec![usize_ty.into()],
        vec![disjoint_slice_val],
        vec![],
        0,
    );
    extract_len_op.deref_mut(ctx).set_loc(loc.clone());

    let extract_len = dialect_mir::ops::MirExtractFieldOp::new(extract_len_op);
    extract_len.set_attr_index(ctx, dialect_mir::attributes::FieldIndexAttr(1));

    if let Some(prev) = last_op {
        extract_len.get_operation().insert_after(ctx, prev);
    } else {
        extract_len.get_operation().insert_at_front(block_ptr, ctx);
    }
    last_op = Some(extract_len.get_operation());

    let len_val = extract_len.get_operation().deref(ctx).get_result(0);

    let prev = last_op.expect("should have at least extract_len op");
    emit_store_result_and_goto(
        ctx,
        destination,
        len_val,
        target,
        block_ptr,
        prev,
        value_map,
        block_map,
        loc,
        "len call without target block",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::ops::MirLoadOp;
    use dialect_mir::types::{MirDisjointSliceType, MirPtrType, MirSliceType};
    use pliron::builtin::attributes::StringAttr;
    use pliron::common_traits::Verify;
    use pliron::identifier::Identifier;
    use pliron::linked_list::ContainsLinkedList;

    #[test]
    fn disjoint_slice_len_receiver_loads_exactly_one_typed_pointer_layer() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let element_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned).to_handle();
        let disjoint_ty: pliron::r#type::TypeHandle =
            MirDisjointSliceType::get(&mut ctx, element_ty).into();
        let receiver_ty = MirPtrType::get_generic(&mut ctx, disjoint_ty, false);
        let block = BasicBlock::new(&mut ctx, None, vec![receiver_ty.into()]);
        let receiver = block.deref(&ctx).get_argument(0);

        let (loaded, load_op) =
            load_disjoint_slice_receiver(&mut ctx, receiver, block, None, Location::Unknown)
                .expect("a pointer to MirDisjointSliceType is the len receiver shape");

        assert_eq!(loaded.get_type(&ctx), disjoint_ty);
        assert_eq!(block.deref(&ctx).iter(&ctx).count(), 1);
        let load = MirLoadOp::new(load_op);
        assert_eq!(load.address_opd(&ctx), receiver);
        assert!(load.verify(&ctx).is_ok());
    }

    #[test]
    fn disjoint_slice_len_receiver_rejects_near_miss_shapes() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let element_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned).to_handle();
        let disjoint_ty: pliron::r#type::TypeHandle =
            MirDisjointSliceType::get(&mut ctx, element_ty).into();
        let receiver_ty: pliron::r#type::TypeHandle =
            MirPtrType::get_generic(&mut ctx, disjoint_ty, false).into();

        // `len(&self)` always supplies one pointer layer. A direct fat value
        // or another pointer layer indicates a broken caller/translation and
        // must not be accepted by recursively guessing at the representation.
        for near_miss_ty in [disjoint_ty, {
            MirPtrType::get_generic(&mut ctx, receiver_ty, false).into()
        }] {
            let block = BasicBlock::new(&mut ctx, None, vec![near_miss_ty]);
            let receiver = block.deref(&ctx).get_argument(0);
            assert!(
                load_disjoint_slice_receiver(&mut ctx, receiver, block, None, Location::Unknown,)
                    .is_err()
            );
            assert_eq!(block.deref(&ctx).iter(&ctx).count(), 0);
        }

        // An ordinary Rust slice is also a `(ptr, len)` carrier, but it is not
        // a DisjointSlice receiver and must not pass a shape-only check.
        let ordinary_slice_ty: pliron::r#type::TypeHandle =
            MirSliceType::get(&mut ctx, element_ty).into();
        let ordinary_receiver_ty = MirPtrType::get_generic(&mut ctx, ordinary_slice_ty, false);
        let block = BasicBlock::new(&mut ctx, None, vec![ordinary_receiver_ty.into()]);
        let receiver = block.deref(&ctx).get_argument(0);
        assert!(
            load_disjoint_slice_receiver(&mut ctx, receiver, block, None, Location::Unknown,)
                .is_err()
        );
        assert_eq!(block.deref(&ctx).iter(&ctx).count(), 0);
    }

    #[test]
    fn index_1d_sreg_ops_carry_their_exact_generated_markers() {
        let mut ctx = Context::new();
        dialect_nvvm::register(&mut ctx);
        let result_type = IntegerType::get(&ctx, 32, Signedness::Unsigned).to_handle();
        let ops = [
            (
                generated_sreg_op(
                    &mut ctx,
                    ReadPtxSregTidXOp::get_concrete_op_info(),
                    result_type,
                    Location::Unknown,
                ),
                "v1:i0001",
            ),
            (
                generated_sreg_op(
                    &mut ctx,
                    ReadPtxSregCtaidXOp::get_concrete_op_info(),
                    result_type,
                    Location::Unknown,
                ),
                "v1:i0002",
            ),
            (
                generated_sreg_op(
                    &mut ctx,
                    ReadPtxSregNtidXOp::get_concrete_op_info(),
                    result_type,
                    Location::Unknown,
                ),
                "v1:i0003",
            ),
        ];
        let key =
            Identifier::try_from(cuda_oxide_codegen::__private::GENERATED_INTRINSIC_MARKER_ATTR)
                .unwrap();

        for (op, expected) in ops {
            let op_ref = op.deref(&ctx);
            let marker: &StringAttr = op_ref.attributes.get(&key).unwrap();
            assert_eq!(String::from(marker.clone()), expected);
        }
    }
}
