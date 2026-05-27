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
//! | `index_1d()`               | Arithmetic expansion    | Global 1D thread index                               |
//! | `index_2d_row/col()`       | Arithmetic expansion    | 2D row/column indices                                |
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

use super::super::helpers::emit_store_result_and_goto;
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
use rustc_public::mir;
/// Emits the expansion of `index_1d()`: `(blockIdx.x * blockDim.x + threadIdx.x) as usize`
///
/// `index_1d()` is marked `#[inline(always)]` but rustc doesn't always inline it,
/// so we manually expand it here into the three NVVM intrinsics and arithmetic operations.
#[allow(clippy::too_many_arguments)]
pub fn emit_index_1d_expansion(
    ctx: &mut Context,
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

    // Emit threadIdx.x
    let tid_op = Operation::new(
        ctx,
        ReadPtxSregTidXOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    tid_op.deref_mut(ctx).set_loc(loc.clone());
    let tid_op = match prev_op {
        Some(prev) => {
            tid_op.insert_after(ctx, prev);
            tid_op
        }
        None => {
            tid_op.insert_at_front(block_ptr, ctx);
            tid_op
        }
    };
    let tid_val = tid_op.deref(ctx).get_result(0);

    // Emit blockIdx.x
    let bid_op = Operation::new(
        ctx,
        ReadPtxSregCtaidXOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    bid_op.deref_mut(ctx).set_loc(loc.clone());
    bid_op.insert_after(ctx, tid_op);
    let bid_val = bid_op.deref(ctx).get_result(0);

    // Emit blockDim.x
    let bdim_op = Operation::new(
        ctx,
        ReadPtxSregNtidXOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    bdim_op.deref_mut(ctx).set_loc(loc.clone());
    bdim_op.insert_after(ctx, bid_op);
    let bdim_val = bdim_op.deref(ctx).get_result(0);

    // Emit bid * bdim
    let mul_op = Operation::new(
        ctx,
        MirMulOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![bid_val, bdim_val],
        vec![],
        0,
    );
    mul_op.deref_mut(ctx).set_loc(loc.clone());
    mul_op.insert_after(ctx, bdim_op);
    let mul_val = mul_op.deref(ctx).get_result(0);

    // Emit (bid * bdim) + tid
    let add_op = Operation::new(
        ctx,
        MirAddOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![mul_val, tid_val],
        vec![],
        0,
    );
    add_op.deref_mut(ctx).set_loc(loc.clone());
    add_op.insert_after(ctx, mul_op);
    let add_val = add_op.deref(ctx).get_result(0);

    // Cast u32 to usize
    let cast_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![usize_type.to_ptr()],
        vec![add_val],
        vec![],
        0,
    );
    cast_op.deref_mut(ctx).set_loc(loc.clone());
    MirCastOp::new(cast_op).set_attr_cast_kind(ctx, MirCastKindAttr::IntToInt);
    cast_op.insert_after(ctx, add_op);
    let result_val = cast_op.deref(ctx).get_result(0);

    emit_store_result_and_goto(
        ctx,
        destination,
        result_val,
        target,
        block_ptr,
        cast_op,
        value_map,
        block_map,
        loc,
        "Call terminator without target not supported",
    )
}

/// Emits `index_2d_row() = blockIdx.y * blockDim.y + threadIdx.y`
#[allow(clippy::too_many_arguments)]
pub fn emit_index_2d_row(
    ctx: &mut Context,
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

    // Emit threadIdx.y
    let tid_op = Operation::new(
        ctx,
        ReadPtxSregTidYOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    tid_op.deref_mut(ctx).set_loc(loc.clone());
    let tid_op = match prev_op {
        Some(prev) => {
            tid_op.insert_after(ctx, prev);
            tid_op
        }
        None => {
            tid_op.insert_at_front(block_ptr, ctx);
            tid_op
        }
    };
    let tid_val = tid_op.deref(ctx).get_result(0);

    // Emit blockIdx.y
    let bid_op = Operation::new(
        ctx,
        ReadPtxSregCtaidYOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    bid_op.deref_mut(ctx).set_loc(loc.clone());
    bid_op.insert_after(ctx, tid_op);
    let bid_val = bid_op.deref(ctx).get_result(0);

    // Emit blockDim.y
    let bdim_op = Operation::new(
        ctx,
        ReadPtxSregNtidYOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    bdim_op.deref_mut(ctx).set_loc(loc.clone());
    bdim_op.insert_after(ctx, bid_op);
    let bdim_val = bdim_op.deref(ctx).get_result(0);

    // Emit bid * bdim
    let mul_op = Operation::new(
        ctx,
        MirMulOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![bid_val, bdim_val],
        vec![],
        0,
    );
    mul_op.deref_mut(ctx).set_loc(loc.clone());
    mul_op.insert_after(ctx, bdim_op);
    let mul_val = mul_op.deref(ctx).get_result(0);

    // Emit (bid * bdim) + tid
    let add_op = Operation::new(
        ctx,
        MirAddOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![mul_val, tid_val],
        vec![],
        0,
    );
    add_op.deref_mut(ctx).set_loc(loc.clone());
    add_op.insert_after(ctx, mul_op);
    let add_val = add_op.deref(ctx).get_result(0);

    // Cast u32 to usize
    let cast_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![usize_type.to_ptr()],
        vec![add_val],
        vec![],
        0,
    );
    cast_op.deref_mut(ctx).set_loc(loc.clone());
    MirCastOp::new(cast_op).set_attr_cast_kind(ctx, MirCastKindAttr::IntToInt);
    cast_op.insert_after(ctx, add_op);
    let result_val = cast_op.deref(ctx).get_result(0);

    emit_store_result_and_goto(
        ctx,
        destination,
        result_val,
        target,
        block_ptr,
        cast_op,
        value_map,
        block_map,
        loc,
        "Call terminator without target not supported",
    )
}

/// Emits `index_2d_col() = blockIdx.x * blockDim.x + threadIdx.x`
///
/// This is the same computation as `index_1d`.
#[allow(clippy::too_many_arguments)]
pub fn emit_index_2d_col(
    ctx: &mut Context,
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    emit_index_1d_expansion(
        ctx,
        destination,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
    )
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
    let tid_y_op = Operation::new(
        ctx,
        ReadPtxSregTidYOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    tid_y_op.deref_mut(ctx).set_loc(loc.clone());
    match last_op {
        Some(prev) => tid_y_op.insert_after(ctx, prev),
        None => tid_y_op.insert_at_front(block_ptr, ctx),
    }
    let tid_y_val = tid_y_op.deref(ctx).get_result(0);
    last_op = Some(tid_y_op);

    let bid_y_op = Operation::new(
        ctx,
        ReadPtxSregCtaidYOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    bid_y_op.deref_mut(ctx).set_loc(loc.clone());
    bid_y_op.insert_after(ctx, last_op.unwrap());
    let bid_y_val = bid_y_op.deref(ctx).get_result(0);
    last_op = Some(bid_y_op);

    let bdim_y_op = Operation::new(
        ctx,
        ReadPtxSregNtidYOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    bdim_y_op.deref_mut(ctx).set_loc(loc.clone());
    bdim_y_op.insert_after(ctx, last_op.unwrap());
    let bdim_y_val = bdim_y_op.deref(ctx).get_result(0);
    last_op = Some(bdim_y_op);

    let mul_y_op = Operation::new(
        ctx,
        MirMulOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
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
        vec![u32_type.to_ptr()],
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
        vec![usize_type.to_ptr()],
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
    let tid_x_op = Operation::new(
        ctx,
        ReadPtxSregTidXOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    tid_x_op.deref_mut(ctx).set_loc(loc.clone());
    tid_x_op.insert_after(ctx, last_op.unwrap());
    let tid_x_val = tid_x_op.deref(ctx).get_result(0);
    last_op = Some(tid_x_op);

    let bid_x_op = Operation::new(
        ctx,
        ReadPtxSregCtaidXOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    bid_x_op.deref_mut(ctx).set_loc(loc.clone());
    bid_x_op.insert_after(ctx, last_op.unwrap());
    let bid_x_val = bid_x_op.deref(ctx).get_result(0);
    last_op = Some(bid_x_op);

    let bdim_x_op = Operation::new(
        ctx,
        ReadPtxSregNtidXOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    bdim_x_op.deref_mut(ctx).set_loc(loc.clone());
    bdim_x_op.insert_after(ctx, last_op.unwrap());
    let bdim_x_val = bdim_x_op.deref(ctx).get_result(0);
    last_op = Some(bdim_x_op);

    let mul_x_op = Operation::new(
        ctx,
        MirMulOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
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
        vec![u32_type.to_ptr()],
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
        vec![usize_type.to_ptr()],
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
        vec![usize_type.to_ptr()],
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
        vec![usize_type.to_ptr()],
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
            element_ty: Ptr<pliron::r#type::TypeObj>,
        },
        Pointer {
            pointee: Ptr<pliron::r#type::TypeObj>,
            element_ty: Ptr<pliron::r#type::TypeObj>,
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
    let (disjoint_slice_val, mut last_op) = match &args[0] {
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
