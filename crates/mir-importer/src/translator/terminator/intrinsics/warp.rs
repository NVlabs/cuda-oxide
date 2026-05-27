/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level primitives.
//!
//! Handles translation of warp shuffle and vote intrinsics.

use super::super::helpers::{emit_goto, emit_store_result_and_goto};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{ActiveMaskOp, BarWarpSyncOp, ReadPtxSregLaneIdOp};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;
/// Emits `lane_id()`: Get the lane index within the warp.
///
/// Returns the thread's position within its 32-thread warp (0-31).
///
/// # Generated Operation
///
/// `nvvm.read.ptx.sreg.laneid` - Maps to PTX `mov.u32 %r, %laneid`
///
/// # Returns
///
/// `u32` - Lane index (0-31)
pub fn emit_lane_id(
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

    // Create lane_id operation
    let lane_id_op = Operation::new(
        ctx,
        ReadPtxSregLaneIdOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    lane_id_op.deref_mut(ctx).set_loc(loc.clone());

    let lane_id_op = if let Some(prev) = prev_op {
        lane_id_op.insert_after(ctx, prev);
        lane_id_op
    } else {
        lane_id_op.insert_at_front(block_ptr, ctx);
        lane_id_op
    };

    let result_value = lane_id_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        lane_id_op,
        value_map,
        block_map,
        loc,
        "lane_id call without target block",
    )
}

/// Emits `active_mask()`: a 32-bit mask of currently-converged lanes.
///
/// Generates an `nvvm.activemask` op that lowers to PTX `activemask.b32`
/// / `@llvm.nvvm.activemask`. Useful for building the mask passed to
/// `*_sync` intrinsics from inside divergent control flow, and for
/// constructing the `CoalescedThreads` cooperative group.
pub fn emit_active_mask(
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

    let active_mask_op = Operation::new(
        ctx,
        ActiveMaskOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    active_mask_op.deref_mut(ctx).set_loc(loc.clone());

    let active_mask_op = if let Some(prev) = prev_op {
        active_mask_op.insert_after(ctx, prev);
        active_mask_op
    } else {
        active_mask_op.insert_at_front(block_ptr, ctx);
        active_mask_op
    };

    let result_value = active_mask_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        active_mask_op,
        value_map,
        block_map,
        loc,
        "active_mask call without target block",
    )
}

/// Emits `warp::sync_mask(mask)`: barrier across a subset of warp lanes.
///
/// Lowers to `nvvm.bar_warp_sync` (PTX `bar.warp.sync`). All lanes named
/// in `mask` must reach the call with the same mask. Returns no value.
pub fn emit_warp_sync_mask(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "warp::sync_mask expects 1 argument [mask], got {}",
                args.len()
            ))
        );
    }

    let (mask, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let sync_op = Operation::new(
        ctx,
        BarWarpSyncOp::get_concrete_op_info(),
        vec![],
        vec![mask],
        vec![],
        0,
    );
    sync_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        sync_op.insert_after(ctx, prev);
    } else {
        sync_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, sync_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("warp::sync_mask call without target block".to_string())
        )
    }
}

/// Emit a warp shuffle operation for i32.
///
/// # Parameters
/// - `shuffle_opid`: The NVVM opid for the specific shuffle variant
/// - `args`: `[mask, value, lane/lane_mask/delta]`
pub fn emit_warp_shuffle_i32(
    ctx: &mut Context,
    body: &mir::Body,
    shuffle_opid: (
        fn(pliron::context::Ptr<pliron::operation::Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "warp shuffle expects 3 arguments [mask, value, lane], got {}",
                args.len()
            ))
        );
    }

    let u32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let (mask, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let (val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (lane_or_delta, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let shuffle_op = Operation::new(
        ctx,
        shuffle_opid,
        vec![u32_type.to_ptr()],
        vec![mask, val, lane_or_delta],
        vec![],
        0,
    );
    shuffle_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        shuffle_op.insert_after(ctx, prev);
    } else {
        shuffle_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = shuffle_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        shuffle_op,
        value_map,
        block_map,
        loc,
        "warp shuffle call without target block",
    )
}

/// Emit a warp shuffle operation for f32.
///
/// # Parameters
/// - `shuffle_opid`: The NVVM opid for the specific shuffle variant
/// - `args`: `[mask, value, lane/lane_mask/delta]`
pub fn emit_warp_shuffle_f32(
    ctx: &mut Context,
    body: &mir::Body,
    shuffle_opid: (
        fn(pliron::context::Ptr<pliron::operation::Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    use pliron::builtin::types::FP32Type;

    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "warp shuffle f32 expects 3 arguments [mask, value, lane], got {}",
                args.len()
            ))
        );
    }

    let f32_type = FP32Type::get(ctx);

    let (mask, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let (val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (lane_or_delta, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let shuffle_op = Operation::new(
        ctx,
        shuffle_opid,
        vec![f32_type.into()],
        vec![mask, val, lane_or_delta],
        vec![],
        0,
    );
    shuffle_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        shuffle_op.insert_after(ctx, prev);
    } else {
        shuffle_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = shuffle_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        shuffle_op,
        value_map,
        block_map,
        loc,
        "warp shuffle f32 call without target block",
    )
}

/// Emit a warp match operation (`match.any.sync` or `match.all.sync`).
///
/// Both ops have the same shape from MIR's perspective: 2 operands
/// `[mask, value]` and 1 result (a u32 bitmask). The op identifier picks the
/// 32-bit vs 64-bit and any vs all variant.
///
/// # Parameters
/// - `match_opid`: The NVVM opid for the specific match variant
/// - `value_is_i64`: true if the value operand is u64 (picks the i64 intrinsic)
/// - `args`: `[mask, value]`
pub fn emit_warp_match(
    ctx: &mut Context,
    body: &mir::Body,
    match_opid: (
        fn(pliron::context::Ptr<pliron::operation::Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
    value_is_i64: bool,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "warp match expects 2 arguments [mask, value], got {}",
                args.len()
            ))
        );
    }

    let _ = value_is_i64;

    let result_ty = IntegerType::get(ctx, 32, Signedness::Unsigned).to_ptr();

    let (mask, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let (value, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let match_op = Operation::new(
        ctx,
        match_opid,
        vec![result_ty],
        vec![mask, value],
        vec![],
        0,
    );
    match_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        match_op.insert_after(ctx, prev);
    } else {
        match_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = match_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        match_op,
        value_map,
        block_map,
        loc,
        "warp match call without target block",
    )
}

/// Emit a warp vote operation (all, any, ballot).
///
/// # Parameters
/// - `vote_opid`: The NVVM opid for the specific vote variant
/// - `result_is_i32`: true for `ballot` (returns i32 bitmask), false for `all`/`any` (returns i1)
/// - `args`: `[mask, predicate]`
pub fn emit_warp_vote(
    ctx: &mut Context,
    body: &mir::Body,
    vote_opid: (
        fn(pliron::context::Ptr<pliron::operation::Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
    result_is_i32: bool,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "warp vote expects 2 arguments [mask, predicate], got {}",
                args.len()
            ))
        );
    }

    let (mask, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let (predicate, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let result_type = if result_is_i32 {
        IntegerType::get(ctx, 32, Signedness::Unsigned).to_ptr()
    } else {
        types::get_bool_type(ctx).to_ptr()
    };

    let vote_op = Operation::new(
        ctx,
        vote_opid,
        vec![result_type],
        vec![mask, predicate],
        vec![],
        0,
    );
    vote_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        vote_op.insert_after(ctx, prev);
    } else {
        vote_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = vote_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        vote_op,
        value_map,
        block_map,
        loc,
        "warp vote call without target block",
    )
}
