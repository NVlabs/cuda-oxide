/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Asynchronous copy (`cp.async`) intrinsic emission.
//!
//! | Function              | PTX                                              |
//! |-----------------------|--------------------------------------------------|
//! | `emit_cp_async_ca_4`  | `cp.async.ca.shared.global [smem], [gmem], 4;`   |
//! | `emit_cp_async_ca_8`  | `cp.async.ca.shared.global [smem], [gmem], 8;`   |

use super::super::helpers::emit_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{CpAsyncCa4Op, CpAsyncCa8Op};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

/// Translate two operands (shared_dst, global_src) from MIR arguments.
fn translate_cp_async_operands(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    loc: Location,
    intrinsic_name: &str,
) -> TranslationResult<(Vec<pliron::value::Value>, Option<Ptr<Operation>>)> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{intrinsic_name} expects 2 arguments (shared_dst, global_src), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(2);

    for arg in args.iter().take(2) {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }

    Ok((operands, last_op))
}

/// Insert the cp.async op and emit the goto to the target block.
fn insert_and_goto(
    ctx: &mut Context,
    cp_op: Ptr<Operation>,
    last_op: Option<Ptr<Operation>>,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    intrinsic_name: &str,
) -> TranslationResult<Ptr<Operation>> {
    if let Some(prev) = last_op {
        cp_op.insert_after(ctx, prev);
    } else {
        cp_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, cp_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!("{intrinsic_name} call without target block"))
        )
    }
}

/// Emits `cp.async.ca.shared.global [...], [...], 4;`
///
/// # Arguments
///
/// - `args[0]`: `*mut u32` - Destination pointer in shared memory
/// - `args[1]`: `*const u32` - Source pointer in global memory
pub fn emit_cp_async_ca_4(
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
    let (operands, last_op) = translate_cp_async_operands(
        ctx,
        body,
        args,
        block_ptr,
        prev_op,
        value_map,
        loc.clone(),
        "cp_async_ca_4",
    )?;

    let cp_op = Operation::new(
        ctx,
        CpAsyncCa4Op::get_concrete_op_info(),
        vec![],
        operands,
        vec![],
        0,
    );
    cp_op.deref_mut(ctx).set_loc(loc.clone());

    insert_and_goto(
        ctx,
        cp_op,
        last_op,
        target,
        block_ptr,
        block_map,
        loc,
        "cp_async_ca_4",
    )
}

/// Emits `cp.async.ca.shared.global [...], [...], 8;`
///
/// # Arguments
///
/// - `args[0]`: `*mut u32` - Destination pointer in shared memory
/// - `args[1]`: `*const u32` - Source pointer in global memory
pub fn emit_cp_async_ca_8(
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
    let (operands, last_op) = translate_cp_async_operands(
        ctx,
        body,
        args,
        block_ptr,
        prev_op,
        value_map,
        loc.clone(),
        "cp_async_ca_8",
    )?;

    let cp_op = Operation::new(
        ctx,
        CpAsyncCa8Op::get_concrete_op_info(),
        vec![],
        operands,
        vec![],
        0,
    );
    cp_op.deref_mut(ctx).set_loc(loc.clone());

    insert_and_goto(
        ctx,
        cp_op,
        last_op,
        target,
        block_ptr,
        block_map,
        loc,
        "cp_async_ca_8",
    )
}
