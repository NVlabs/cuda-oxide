/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cluster Launch Control (CLC) intrinsics for Blackwell+ (SM 100+).
//!
//! Handles `clc_try_cancel`, `clc_try_cancel_multicast`, and `clc_query_*` intrinsics.

use super::super::helpers::{emit_goto, emit_store_result_and_goto};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{
    ClcQueryGetFirstCtaidXOp, ClcQueryGetFirstCtaidYOp, ClcQueryGetFirstCtaidZOp,
    ClcQueryIsCanceledOp, ClcTryCancelMulticastOp, ClcTryCancelOp,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;
/// Emit clc_try_cancel: async request to steal a pending CTA's work.
///
/// Args:
/// - args[0]: *mut u8 (response - 16-byte aligned shared memory)
/// - args[1]: *mut Barrier (mbar - initialized mbarrier)
///
/// Returns: void
pub fn emit_clc_try_cancel(
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
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "clc_try_cancel expects 2 arguments (response, mbar), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    let (response, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (mbar, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let op = Operation::new(
        ctx,
        ClcTryCancelOp::get_concrete_op_info(),
        vec![],
        vec![response, mbar],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("clc_try_cancel call without target block".to_string())
        )
    }
}

/// Emit clc_try_cancel_multicast: multicast variant of try_cancel.
///
/// Args: same as clc_try_cancel
/// Returns: void
pub fn emit_clc_try_cancel_multicast(
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
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "clc_try_cancel_multicast expects 2 arguments (response, mbar), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    let (response, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (mbar, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let op = Operation::new(
        ctx,
        ClcTryCancelMulticastOp::get_concrete_op_info(),
        vec![],
        vec![response, mbar],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "clc_try_cancel_multicast call without target block".to_string()
            )
        )
    }
}

enum ClcQueryKind {
    IsCanceled,
    GetFirstCtaidX,
    GetFirstCtaidY,
    GetFirstCtaidZ,
}

/// Helper for query_cancel intrinsics that return a u32.
///
/// All query_cancel variants share the same signature:
/// Args: (resp_lo: u64, resp_hi: u64) -> u32
fn emit_clc_query(
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
    kind: ClcQueryKind,
) -> TranslationResult<Ptr<Operation>> {
    let op_name = match kind {
        ClcQueryKind::IsCanceled => "clc_query_is_canceled",
        ClcQueryKind::GetFirstCtaidX => "clc_query_get_first_ctaid_x",
        ClcQueryKind::GetFirstCtaidY => "clc_query_get_first_ctaid_y",
        ClcQueryKind::GetFirstCtaidZ => "clc_query_get_first_ctaid_z",
    };

    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{} expects 2 arguments (resp_lo, resp_hi), got {}",
                op_name,
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    let (resp_lo, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (resp_hi, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let i32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let concrete_op_info = match kind {
        ClcQueryKind::IsCanceled => ClcQueryIsCanceledOp::get_concrete_op_info(),
        ClcQueryKind::GetFirstCtaidX => ClcQueryGetFirstCtaidXOp::get_concrete_op_info(),
        ClcQueryKind::GetFirstCtaidY => ClcQueryGetFirstCtaidYOp::get_concrete_op_info(),
        ClcQueryKind::GetFirstCtaidZ => ClcQueryGetFirstCtaidZOp::get_concrete_op_info(),
    };

    let query_op = Operation::new(
        ctx,
        concrete_op_info,
        vec![i32_type.to_ptr()],
        vec![resp_lo, resp_hi],
        vec![],
        0,
    );
    query_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        query_op.insert_after(ctx, prev);
    } else {
        query_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = query_op.deref(ctx).get_result(0);
    let no_target_msg = format!("{} call without target block", op_name);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        query_op,
        value_map,
        block_map,
        loc,
        &no_target_msg,
    )
}

/// Emit clc_query_is_canceled: check if try_cancel was canceled (no work).
pub fn emit_clc_query_is_canceled(
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
    emit_clc_query(
        ctx,
        body,
        args,
        destination,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        ClcQueryKind::IsCanceled,
    )
}

/// Emit clc_query_get_first_ctaid_x: get X coordinate of stolen CTA.
pub fn emit_clc_query_get_first_ctaid_x(
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
    emit_clc_query(
        ctx,
        body,
        args,
        destination,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        ClcQueryKind::GetFirstCtaidX,
    )
}

/// Emit clc_query_get_first_ctaid_y: get Y coordinate of stolen CTA.
pub fn emit_clc_query_get_first_ctaid_y(
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
    emit_clc_query(
        ctx,
        body,
        args,
        destination,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        ClcQueryKind::GetFirstCtaidY,
    )
}

/// Emit clc_query_get_first_ctaid_z: get Z coordinate of stolen CTA.
pub fn emit_clc_query_get_first_ctaid_z(
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
    emit_clc_query(
        ctx,
        body,
        args,
        destination,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        ClcQueryKind::GetFirstCtaidZ,
    )
}
