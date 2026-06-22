/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! WMMA (Warp Matrix Multiply-Accumulate) intrinsic emission.
//!
//! Translates `cuda_device::wmma::*` intrinsic calls into `dialect-nvvm` WMMA operations.

use super::super::helpers::emit_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{
    MmaM16N8K32S32S4Op, MmaM16N8K32S32U4Op, MmaM16N8K64S32S4Op, MmaM16N8K64S32U4Op,
};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

/// Emit `mma_m16n8k32_s32_s4`: signed int4 MMA, k=32.
///
/// Args: `(acc: &mut [i32; 4], a: &[u32; 2], b: &u32)`
/// Returns: void (result written through `acc` pointer)
pub fn emit_mma_m16n8k32_s32_s4(
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
                "mma_m16n8k32_s32_s4 expects 3 arguments (acc, a, b), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(3);
    for arg in args.iter().take(3) {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }

    let mma_op = Operation::new(
        ctx,
        MmaM16N8K32S32S4Op::get_concrete_op_info(),
        vec![],
        operands,
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
                "mma_m16n8k32_s32_s4 call without target block".to_string(),
            )
        )
    }
}

/// Emit `mma_m16n8k32_s32_u4`: unsigned int4 MMA, k=32.
///
/// Args: `(acc: &mut [i32; 4], a: &[u32; 2], b: &u32)`
/// Returns: void (result written through `acc` pointer)
pub fn emit_mma_m16n8k32_s32_u4(
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
                "mma_m16n8k32_s32_u4 expects 3 arguments (acc, a, b), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(3);
    for arg in args.iter().take(3) {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }

    let mma_op = Operation::new(
        ctx,
        MmaM16N8K32S32U4Op::get_concrete_op_info(),
        vec![],
        operands,
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
                "mma_m16n8k32_s32_u4 call without target block".to_string(),
            )
        )
    }
}

/// Emit `mma_m16n8k64_s32_s4`: signed int4 MMA, k=64.
///
/// Args: `(acc: &mut [i32; 4], a: &[u32; 4], b: &[u32; 2])`
/// Returns: void (result written through `acc` pointer)
pub fn emit_mma_m16n8k64_s32_s4(
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
                "mma_m16n8k64_s32_s4 expects 3 arguments (acc, a, b), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(3);
    for arg in args.iter().take(3) {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }

    let mma_op = Operation::new(
        ctx,
        MmaM16N8K64S32S4Op::get_concrete_op_info(),
        vec![],
        operands,
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
                "mma_m16n8k64_s32_s4 call without target block".to_string(),
            )
        )
    }
}

/// Emit `mma_m16n8k64_s32_u4`: unsigned int4 MMA, k=64.
///
/// Args: `(acc: &mut [i32; 4], a: &[u32; 4], b: &[u32; 2])`
/// Returns: void (result written through `acc` pointer)
pub fn emit_mma_m16n8k64_s32_u4(
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
                "mma_m16n8k64_s32_u4 expects 3 arguments (acc, a, b), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(3);
    for arg in args.iter().take(3) {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }

    let mma_op = Operation::new(
        ctx,
        MmaM16N8K64S32U4Op::get_concrete_op_info(),
        vec![],
        operands,
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
                "mma_m16n8k64_s32_u4 call without target block".to_string(),
            )
        )
    }
}
