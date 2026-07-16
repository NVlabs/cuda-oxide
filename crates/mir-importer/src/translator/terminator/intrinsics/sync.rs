/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Synchronization and barrier intrinsics.
//!
//! Handles handwritten synchronization primitives including:
//! - `mbarrier_*` - Asynchronous barrier operations
//! - `fence_proxy_async_shared_cta()` - Async proxy fence

use super::super::helpers::{emit_goto, emit_store_result_and_goto};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{
    FenceMbarrierInitReleaseClusterOp, FenceProxyAsyncGenericAcquireSharedClusterClusterOp,
    FenceProxyAsyncGenericReleaseSharedCtaClusterOp, FenceProxyAsyncSharedCtaOp,
    MbarrierArriveClusterOp, MbarrierArriveExpectTxClusterOp, MbarrierArriveExpectTxSharedOp,
    MbarrierTryWaitParityClusterOp, MbarrierTryWaitParitySharedOp, MbarrierTryWaitSharedOp,
    NanosleepOp,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;
/// Emit mbarrier_arrive_expect_tx: arrive at barrier with expected transaction bytes.
///
/// This is required for TMA's complete_tx::bytes mode. The barrier must be told
/// how many bytes to expect from the TMA transaction before the TMA is initiated.
///
/// Args:
/// - `args[0]`: *const Barrier (pointer to barrier in shared memory)
/// - `args[1]`: u32 (tx_count - unused, kept for API compatibility)
/// - `args[2]`: u32 (bytes - expected transaction byte count)
///
/// Returns: u64 (phase token)
pub fn emit_mbarrier_arrive_expect_tx(
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
    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "mbarrier_arrive_expect_tx expects 3 arguments (bar, tx_count, bytes), got {}",
                args.len()
            ))
        );
    }

    // Get the barrier pointer (arg 0)
    let (bar_ptr, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Skip tx_count (arg 1) - not used in the PTX instruction
    // Get the expected bytes (arg 2)
    let (bytes, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // Result type: i64 (phase token)
    let i64_type = IntegerType::get(ctx, 64, Signedness::Unsigned);

    // Create the mbarrier_arrive_expect_tx_shared operation
    let arrive_op = Operation::new(
        ctx,
        MbarrierArriveExpectTxSharedOp::get_concrete_op_info(),
        vec![i64_type.to_handle()], // Result: i64 token
        vec![bar_ptr, bytes],       // Operands: ptr, bytes
        vec![],
        0,
    );
    arrive_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        arrive_op.insert_after(ctx, prev);
    } else {
        arrive_op.insert_at_front(block_ptr, ctx);
    }

    // Store the result (token) in the destination and branch to the success target.
    let result_value = arrive_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        arrive_op,
        value_map,
        block_map,
        loc,
        "mbarrier_arrive_expect_tx call without target block",
    )
}

/// Emit cluster-scope mbarrier_arrive_expect_tx.
///
/// Args:
/// - `args[0]`: *const Barrier (pointer to barrier in CTA shared memory)
/// - `args[1]`: u32 (tx_count - unused, kept for API compatibility)
/// - `args[2]`: u32 (bytes - expected transaction byte count)
///
/// Returns: u64 (phase token)
pub fn emit_mbarrier_arrive_expect_tx_cluster(
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
    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "mbarrier_arrive_expect_tx_cluster expects 3 arguments (bar, tx_count, bytes), got {}",
                args.len()
            ))
        );
    }

    let (bar_ptr, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // tx_count (arg 1) is retained by the device API but is not a PTX operand.
    let (bytes, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let i64_type = IntegerType::get(ctx, 64, Signedness::Unsigned);
    let arrive_op = Operation::new(
        ctx,
        MbarrierArriveExpectTxClusterOp::get_concrete_op_info(),
        vec![i64_type.to_handle()],
        vec![bar_ptr, bytes],
        vec![],
        0,
    );
    arrive_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        arrive_op.insert_after(ctx, prev);
    } else {
        arrive_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = arrive_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        arrive_op,
        value_map,
        block_map,
        loc,
        "mbarrier_arrive_expect_tx_cluster call without target block",
    )
}

/// Emit mbarrier_arrive_cluster: arrive at a barrier in another CTA's shared memory.
///
/// Takes a raw u64 address (from map_shared_rank cast to integer) to avoid
/// LLVM IR address-space conflicts in loop phi nodes.
///
/// Args:
/// - `args[0]`: u64 (cluster-scope barrier address from mapa)
///
/// Returns: void
pub fn emit_mbarrier_arrive_cluster(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    _destination: &mir::Place,
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
                "mbarrier_arrive_cluster expects 1 argument (addr: u64), got {}",
                args.len()
            ))
        );
    }

    let (addr, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let arrive_op = Operation::new(
        ctx,
        MbarrierArriveClusterOp::get_concrete_op_info(),
        vec![],     // No results
        vec![addr], // Operand: u64 address
        vec![],
        0,
    );
    arrive_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        arrive_op.insert_after(ctx, prev);
    } else {
        arrive_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, arrive_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "mbarrier_arrive_cluster call without target block".to_string(),
            )
        )
    }
}

/// Emit nanosleep: suspend thread for approximately N nanoseconds.
///
/// Args:
/// - `args[0]`: u32 (nanoseconds)
///
/// Returns: void
pub fn emit_nanosleep(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    _destination: &mir::Place,
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
                "nanosleep expects 1 argument (ns: u32), got {}",
                args.len()
            ))
        );
    }

    let (ns_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let sleep_op = Operation::new(
        ctx,
        NanosleepOp::get_concrete_op_info(),
        vec![],
        vec![ns_val],
        vec![],
        0,
    );
    sleep_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        sleep_op.insert_after(ctx, prev);
    } else {
        sleep_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, sleep_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("nanosleep call without target block".to_string(),)
        )
    }
}

/// Emit mbarrier_try_wait: try to wait for barrier phase (with scheduling hints).
///
/// Similar to mbarrier_test_wait but uses try_wait which provides better scheduling
/// hints to the hardware. This is the preferred instruction for TMA synchronization.
///
/// Args:
/// - `args[0]`: *const Barrier (pointer to barrier in shared memory)
/// - `args[1]`: u64 (phase token)
///
/// Returns: bool
pub fn emit_mbarrier_try_wait(
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
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "mbarrier_try_wait expects 2 arguments, got {}",
                args.len()
            ))
        );
    }

    // Get the barrier pointer (arg 0)
    let (bar_ptr, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Get the token (arg 1)
    let (token, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // Result type: i1 (bool), signless to match Rust `bool`.
    let i1_type = types::get_bool_type(ctx);

    // Create the mbarrier_try_wait_shared operation
    let try_wait_op = Operation::new(
        ctx,
        MbarrierTryWaitSharedOp::get_concrete_op_info(),
        vec![i1_type.to_handle()], // Result: i1 (bool)
        vec![bar_ptr, token],      // Operands: ptr, token
        vec![],
        0,
    );
    try_wait_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        try_wait_op.insert_after(ctx, prev);
    } else {
        try_wait_op.insert_at_front(block_ptr, ctx);
    }

    // Store the result in the destination slot and branch to the success target.
    let result_value = try_wait_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        try_wait_op,
        value_map,
        block_map,
        loc,
        "mbarrier_try_wait call without target block",
    )
}

/// Emit mbarrier_try_wait_parity: parity-based wait for barrier phase.
///
/// Args:
/// - `args[0]`: *const Barrier (ptr to barrier in shared)
/// - `args[1]`: u32 parity
///
/// Returns: bool
pub fn emit_mbarrier_try_wait_parity(
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
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "mbarrier_try_wait_parity expects 2 arguments, got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let (bar_ptr, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;
    let (parity, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // Result type: i1
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let op_ptr = Operation::new(
        ctx,
        MbarrierTryWaitParitySharedOp::get_concrete_op_info(),
        vec![i1_ty.into()],
        vec![bar_ptr, parity],
        vec![],
        0,
    );
    op_ptr.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op_ptr.insert_after(ctx, prev);
    } else {
        op_ptr.insert_at_front(block_ptr, ctx);
    }

    // Store the result in the destination slot and branch to the success target.
    let result_value = op_ptr.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        op_ptr,
        value_map,
        block_map,
        loc,
        "mbarrier_try_wait_parity call without target block",
    )
}

/// Emit cluster-scope parity wait for a CTA-shared barrier.
///
/// Args:
/// - `args[0]`: *const Barrier (ptr to barrier in CTA shared memory)
/// - `args[1]`: u32 parity
///
/// Returns: bool
pub fn emit_mbarrier_try_wait_parity_cluster(
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
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "mbarrier_try_wait_parity_cluster expects 2 arguments, got {}",
                args.len()
            ))
        );
    }

    let (bar_ptr, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let (parity, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let op_ptr = Operation::new(
        ctx,
        MbarrierTryWaitParityClusterOp::get_concrete_op_info(),
        vec![i1_ty.into()],
        vec![bar_ptr, parity],
        vec![],
        0,
    );
    op_ptr.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op_ptr.insert_after(ctx, prev);
    } else {
        op_ptr.insert_at_front(block_ptr, ctx);
    }

    let result_value = op_ptr.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        op_ptr,
        value_map,
        block_map,
        loc,
        "mbarrier_try_wait_parity_cluster call without target block",
    )
}

/// Emit fence_proxy_async_shared_cta: fence to sync generic proxy with async proxy.
///
/// This fence ensures memory operations through the generic proxy (like mbarrier.init)
/// are visible to the async proxy (TMA hardware). Critical for TMA operations!
///
/// Args: none
/// Returns: void
pub fn emit_fence_proxy_async_shared_cta(
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
                "fence_proxy_async_shared_cta expects 0 arguments, got {}",
                args.len()
            ))
        );
    }

    // Create the fence operation (void return, no operands)
    let fence_op = Operation::new(
        ctx,
        FenceProxyAsyncSharedCtaOp::get_concrete_op_info(),
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
            TranslationErr::unsupported(
                "fence_proxy_async_shared_cta call without target block".to_string(),
            )
        )
    }
}

/// Emit `fence.mbarrier_init.release.cluster`.
///
/// Args: none
/// Returns: void
pub fn emit_fence_mbarrier_init_release_cluster(
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
                "fence_mbarrier_init_release_cluster expects 0 arguments, got {}",
                args.len()
            ))
        );
    }

    let fence_op = Operation::new(
        ctx,
        FenceMbarrierInitReleaseClusterOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    fence_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = prev_op {
        fence_op.insert_after(ctx, prev);
    } else {
        fence_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, fence_op, block_map, loc))
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "fence_mbarrier_init_release_cluster call without target block".to_string(),
            )
        )
    }
}

/// Emit the cluster-scope generic-to-async proxy release fence.
///
/// Args: none
/// Returns: void
pub fn emit_fence_proxy_async_generic_release_shared_cta_cluster(
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
                "fence_proxy_async_generic_release_shared_cta_cluster expects 0 arguments, got {}",
                args.len()
            ))
        );
    }

    let fence_op = Operation::new(
        ctx,
        FenceProxyAsyncGenericReleaseSharedCtaClusterOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    fence_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = prev_op {
        fence_op.insert_after(ctx, prev);
    } else {
        fence_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, fence_op, block_map, loc))
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "fence_proxy_async_generic_release_shared_cta_cluster call without target block"
                    .to_string(),
            )
        )
    }
}

/// Emit the cluster-scope async-to-generic proxy acquire fence.
///
/// Args: none
/// Returns: void
pub fn emit_fence_proxy_async_generic_acquire_shared_cluster_cluster(
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
                "fence_proxy_async_generic_acquire_shared_cluster_cluster expects 0 arguments, got {}",
                args.len()
            ))
        );
    }

    let fence_op = Operation::new(
        ctx,
        FenceProxyAsyncGenericAcquireSharedClusterClusterOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    fence_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = prev_op {
        fence_op.insert_after(ctx, prev);
    } else {
        fence_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, fence_op, block_map, loc))
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "fence_proxy_async_generic_acquire_shared_cluster_cluster call without target block"
                    .to_string(),
            )
        )
    }
}
