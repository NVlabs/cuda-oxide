/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! WMMA (Warp Matrix Multiply-Accumulate) binary mma.sync intrinsic translation for SM 80+.

use super::super::helpers::emit_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{MmaM16N8K128S32B1Op, MmaM16N8K256S32B1Op};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

/// Emit `mma_m16n8k128_s32_b1`: Warp MMA with s32 accumulator and b1 inputs (xor.popc).
///
/// Args:
/// - `args[0]`: `&mut [i32; 4]` (accumulator pointer, read-modify-write)
/// - `args[1]`: `&[u32; 2]` (A fragment pointer, packed b1)
/// - `args[2]`: `&u32` (B fragment pointer, packed b1)
///
/// Returns: void (accumulator updated in-place)
pub fn emit_mma_m16n8k128_s32_b1(
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
    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "mma_m16n8k128_s32_b1 expects 3 arguments (acc, a, b), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    let (acc_ptr, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (a_ptr, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (b_ptr, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let mma_op = Operation::new(
        ctx,
        MmaM16N8K128S32B1Op::get_concrete_op_info(),
        vec![],
        vec![acc_ptr, a_ptr, b_ptr],
        vec![],
        0,
    );
    mma_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        mma_op.insert_after(ctx, prev);
    } else {
        mma_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, mma_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "mma_m16n8k128_s32_b1 call without target block".to_string(),
            )
        )
    }
}

/// Emit `mma_m16n8k256_s32_b1`: Warp MMA with s32 accumulator and b1 inputs (xor.popc).
///
/// Args:
/// - `args[0]`: `&mut [i32; 4]` (accumulator pointer, read-modify-write)
/// - `args[1]`: `&[u32; 4]` (A fragment pointer, packed b1)
/// - `args[2]`: `&[u32; 2]` (B fragment pointer, packed b1)
///
/// Returns: void (accumulator updated in-place)
pub fn emit_mma_m16n8k256_s32_b1(
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
    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "mma_m16n8k256_s32_b1 expects 3 arguments (acc, a, b), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    let (acc_ptr, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (a_ptr, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (b_ptr, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let mma_op = Operation::new(
        ctx,
        MmaM16N8K256S32B1Op::get_concrete_op_info(),
        vec![],
        vec![acc_ptr, a_ptr, b_ptr],
        vec![],
        0,
    );
    mma_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        mma_op.insert_after(ctx, prev);
    } else {
        mma_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, mma_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "mma_m16n8k256_s32_b1 call without target block".to_string(),
            )
        )
    }
}
