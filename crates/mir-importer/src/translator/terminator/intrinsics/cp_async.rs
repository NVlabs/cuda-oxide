/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Asynchronous copy (`cp.async`) intrinsic emission.
//!
//! | Function                    | PTX                                                   |
//! |-----------------------------|-------------------------------------------------------|
//! | `emit_cp_async_ca_4`        | `cp.async.ca.shared.global [smem], [gmem], 4;`        |
//! | `emit_cp_async_ca_8`        | `cp.async.ca.shared.global [smem], [gmem], 8;`        |
//! | `emit_cp_async_ca_zfill_4`  | `cp.async.ca.shared.global [smem], [gmem], 4, src;`   |
//! | `emit_cp_async_ca_zfill_8`  | `cp.async.ca.shared.global [smem], [gmem], 8, src;`   |
//! | `emit_cp_async_ca_zfill_16` | `cp.async.ca.shared.global [smem], [gmem], 16, src;`  |

use super::super::helpers::emit_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{
    CpAsyncCa4Op, CpAsyncCa8Op, CpAsyncCaZfill4Op, CpAsyncCaZfill8Op, CpAsyncCaZfill16Op,
};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

/// Translate `n` operands from MIR arguments.
fn translate_operands(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    n: usize,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    loc: Location,
    intrinsic_name: &str,
) -> TranslationResult<(Vec<pliron::value::Value>, Option<Ptr<Operation>>)> {
    if args.len() != n {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{intrinsic_name} expects {n} arguments, got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(n);

    for arg in args.iter().take(n) {
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

/// Shared emitter for all cp.async variants.
fn emit_cp_async<T: Op>(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    n_operands: usize,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    name: &str,
) -> TranslationResult<Ptr<Operation>> {
    let (operands, last_op) = translate_operands(
        ctx,
        body,
        args,
        n_operands,
        block_ptr,
        prev_op,
        value_map,
        loc.clone(),
        name,
    )?;

    let cp_op = Operation::new(ctx, T::get_concrete_op_info(), vec![], operands, vec![], 0);
    cp_op.deref_mut(ctx).set_loc(loc.clone());

    insert_and_goto(ctx, cp_op, last_op, target, block_ptr, block_map, loc, name)
}

// =============================================================================
// 2-operand cp.async (no zero-fill)
// =============================================================================

/// Emits `cp.async.ca.shared.global [...], [...], 4;`
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
    emit_cp_async::<CpAsyncCa4Op>(
        ctx,
        body,
        args,
        2,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        "cp_async_ca_4",
    )
}

/// Emits `cp.async.ca.shared.global [...], [...], 8;`
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
    emit_cp_async::<CpAsyncCa8Op>(
        ctx,
        body,
        args,
        2,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        "cp_async_ca_8",
    )
}

// =============================================================================
// 3-operand cp.async with zero-fill (dst, src, src_size)
// =============================================================================

/// Emits `cp.async.ca.shared.global [...], [...], 4, src_size;`
pub fn emit_cp_async_ca_zfill_4(
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
    emit_cp_async::<CpAsyncCaZfill4Op>(
        ctx,
        body,
        args,
        3,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        "cp_async_ca_zfill_4",
    )
}

/// Emits `cp.async.ca.shared.global [...], [...], 8, src_size;`
pub fn emit_cp_async_ca_zfill_8(
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
    emit_cp_async::<CpAsyncCaZfill8Op>(
        ctx,
        body,
        args,
        3,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        "cp_async_ca_zfill_8",
    )
}

/// Emits `cp.async.ca.shared.global [...], [...], 16, src_size;`
pub fn emit_cp_async_ca_zfill_16(
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
    emit_cp_async::<CpAsyncCaZfill16Op>(
        ctx,
        body,
        args,
        3,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        "cp_async_ca_zfill_16",
    )
}
