/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix intrinsics (`movmatrix`, `mma.sync`).

use super::super::helpers::{emit_goto, emit_store_result_and_goto};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{MmaM16N8K16F32Bf16Op, MovmatrixTransB16Op};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

/// Emit movmatrix_trans_b16: in-register 8×8 matrix transpose.
///
/// Takes one u32 operand and returns one u32.
#[allow(clippy::too_many_arguments)]
pub fn emit_movmatrix_trans_b16(
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
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "movmatrix_trans_b16 expects 1 argument, got {}",
                args.len()
            ))
        );
    }

    let (a_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let mov_op = Operation::new(
        ctx,
        MovmatrixTransB16Op::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![a_val],
        vec![],
        0,
    );
    mov_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        mov_op.insert_after(ctx, prev);
    } else {
        mov_op.insert_at_front(block_ptr, ctx);
    }

    let result = mov_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        mov_op,
        value_map,
        block_map,
        loc,
        "movmatrix_trans_b16 call without target block",
    )
}

/// Emit `mma_m16n8k16_f32_bf16`: Warp MMA with f32 accumulator and bf16 inputs.
///
/// Args:
/// - `args[0]`: `&mut [f32; 4]` (accumulator pointer, read-modify-write)
/// - `args[1]`: `&[u32; 4]` (A fragment pointer)
/// - `args[2]`: `&[u32; 2]` (B fragment pointer)
///
/// Returns: void (accumulator updated in-place)
pub fn emit_mma_m16n8k16_f32_bf16(
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
                "mma_m16n8k16_f32_bf16 expects 3 arguments (acc, a, b), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    // arg[0]: acc_ptr (&mut [f32; 4])
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

    // arg[1]: a_ptr (&[u32; 4])
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

    // arg[2]: b_ptr (&[u32; 2])
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

    // Create the MMA operation
    let mma_op = Operation::new(
        ctx,
        MmaM16N8K16F32Bf16Op::get_concrete_op_info(),
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
                "mma_m16n8k16_f32_bf16 call without target block".to_string()
            )
        )
    }
}
