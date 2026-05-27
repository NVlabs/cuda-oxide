/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Hopper WGMMA (Warpgroup Matrix Multiply-Accumulate) intrinsics.
//!
//! Handles SM90 (Hopper) asynchronous warpgroup matrix operations.

use super::super::helpers::{emit_goto, emit_store_result_and_goto};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_mir::ops::MirConstantOp;
use dialect_nvvm::ops::{
    WgmmaCommitGroupSyncAlignedOp, WgmmaFenceSyncAlignedOp, WgmmaMakeSmemDescOp,
    WgmmaMmaM64N64K16F32Bf16Op, WgmmaWaitGroupSyncAlignedOp,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;
/// Emits `wgmma_fence()`: WGMMA input fence.
///
/// Establishes ordering between shared memory writes and subsequent WGMMA
/// operations. Must be called before WGMMA to ensure input data is visible.
///
/// # Generated Operation
///
/// `nvvm.wgmma.fence.sync.aligned` - Maps to PTX `wgmma.fence.sync.aligned`
///
/// # Hopper+ Only
///
/// This instruction is only available on SM90 (Hopper) and later.
pub fn emit_wgmma_fence(
    ctx: &mut Context,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if !args.is_empty() {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "wgmma_fence expects 0 arguments, got {}",
                args.len()
            ))
        );
    }

    // Create the fence operation (void return, no operands)
    let fence_op = Operation::new(
        ctx,
        WgmmaFenceSyncAlignedOp::get_concrete_op_info(),
        vec![], // No results
        vec![], // No operands
        vec![],
        0,
    );
    fence_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = prev_op {
        fence_op.insert_after(ctx, prev);
    } else {
        fence_op.insert_at_front(block_ptr, ctx);
    }

    // Emit goto to target block
    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, fence_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("wgmma_fence call without target block".to_string())
        )
    }
}

/// Emit wgmma_commit_group: Commit pending WGMMA operations to a group.
///
/// Args: none
/// Returns: void
pub fn emit_wgmma_commit_group(
    ctx: &mut Context,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if !args.is_empty() {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "wgmma_commit_group expects 0 arguments, got {}",
                args.len()
            ))
        );
    }

    // Create the commit operation (void return, no operands)
    let commit_op = Operation::new(
        ctx,
        WgmmaCommitGroupSyncAlignedOp::get_concrete_op_info(),
        vec![], // No results
        vec![], // No operands
        vec![],
        0,
    );
    commit_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = prev_op {
        commit_op.insert_after(ctx, prev);
    } else {
        commit_op.insert_at_front(block_ptr, ctx);
    }

    // Emit goto to target block
    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, commit_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("wgmma_commit_group call without target block".to_string())
        )
    }
}

/// Emit wgmma_wait_group: Wait for N groups to complete.
///
/// This is a const generic function, so N is extracted from the function path.
///
/// Args: none (N comes from const generic)
/// Returns: void
pub fn emit_wgmma_wait_group(
    ctx: &mut Context,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    n: u64,
) -> TranslationResult<Ptr<Operation>> {
    if !args.is_empty() {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "wgmma_wait_group expects 0 arguments, got {}",
                args.len()
            ))
        );
    }

    // Create constant for N
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let apint = pliron::utils::apint::APInt::from_u64(n, std::num::NonZeroUsize::new(64).unwrap());
    let int_attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, apint);
    let const_raw_op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![],
        vec![],
        0,
    );
    const_raw_op.deref_mut(ctx).set_loc(loc.clone());
    let n_const = MirConstantOp::new(const_raw_op);
    n_const.set_attr_value(ctx, int_attr);

    let last_op = if let Some(prev) = prev_op {
        n_const.get_operation().insert_after(ctx, prev);
        n_const.get_operation()
    } else {
        n_const.get_operation().insert_at_front(block_ptr, ctx);
        n_const.get_operation()
    };

    let n_value = n_const.get_operation().deref(ctx).get_result(0);

    // Create the wait operation (void return, 1 operand for N)
    let wait_op = Operation::new(
        ctx,
        WgmmaWaitGroupSyncAlignedOp::get_concrete_op_info(),
        vec![],        // No results
        vec![n_value], // N operand
        vec![],
        0,
    );
    wait_op.deref_mut(ctx).set_loc(loc.clone());
    wait_op.insert_after(ctx, last_op);

    // Emit goto to target block
    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, wait_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("wgmma_wait_group call without target block".to_string())
        )
    }
}

/// Emit make_smem_desc: Create SMEM descriptor for WGMMA.
///
/// Args:
/// - args[0]: *const u8 (pointer to shared memory)
///
/// Returns: u64 (64-bit descriptor)
pub fn emit_wgmma_make_smem_desc(
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
                "make_smem_desc expects 1 argument, got {}",
                args.len()
            ))
        );
    }

    // Translate the pointer argument
    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Create the make_smem_desc operation (returns u64)
    // Use Unsigned signedness to match Rust's u64 type
    let u64_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
    let desc_op = Operation::new(
        ctx,
        WgmmaMakeSmemDescOp::get_concrete_op_info(),
        vec![u64_ty.into()], // Result: u64
        vec![ptr_val],       // Operand: ptr
        vec![],
        0,
    );
    desc_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        desc_op.insert_after(ctx, prev);
    } else {
        desc_op.insert_at_front(block_ptr, ctx);
    }

    // Map the result
    let result_value = desc_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        desc_op,
        value_map,
        block_map,
        loc,
        "make_smem_desc call without target block",
    )
}

/// Emit wgmma_mma_m64n64k16_f32_bf16: WGMMA matrix multiply-accumulate.
///
/// Performs D = A × B + D where:
/// - A: 64×16 (from SMEM descriptor)
/// - B: 16×64 (from SMEM descriptor)
/// - D: 64×64 accumulator (32 f32 values per thread, passed by pointer)
///
/// Args:
/// - args[0]: &mut [[f32; 8]; 4] (accumulator pointer, read-modify-write)
/// - args[1]: u64 (desc_a - SMEM descriptor for A)
/// - args[2]: u64 (desc_b - SMEM descriptor for B)
///
/// Returns: void (accumulator updated in-place)
pub fn emit_wgmma_mma_m64n64k16_f32_bf16(
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
                "wgmma_mma_m64n64k16_f32_bf16 expects 3 arguments (acc_ptr, desc_a, desc_b), got {}",
                args.len()
            ))
        );
    }

    // Translate arguments
    let mut last_op = prev_op;

    // arg[0]: acc_ptr (pointer to accumulator array)
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

    // arg[1]: desc_a (u64 descriptor)
    let (desc_a, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // arg[2]: desc_b (u64 descriptor)
    let (desc_b, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // Create the WGMMA MMA operation
    let mma_op = Operation::new(
        ctx,
        WgmmaMmaM64N64K16F32Bf16Op::get_concrete_op_info(),
        vec![],                        // No results (void)
        vec![acc_ptr, desc_a, desc_b], // Operands
        vec![],
        0,
    );
    mma_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        mma_op.insert_after(ctx, prev);
    } else {
        mma_op.insert_at_front(block_ptr, ctx);
    }

    // Emit goto to target block
    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, mma_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "wgmma_mma_m64n64k16_f32_bf16 call without target block".to_string()
            )
        )
    }
}
