/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Synchronization and barrier intrinsics.
//!
//! Handles translation of synchronization primitives including:
//! - `sync_threads()` - CTA-wide barrier
//! - `mbarrier_*` - Asynchronous barrier operations
//! - `fence_proxy_async_shared_cta()` - Async proxy fence

use super::super::helpers::{emit_goto, emit_store_result_and_goto};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{
    FenceProxyAsyncSharedCtaOp, MbarrierArriveClusterOp, MbarrierArriveExpectTxSharedOp,
    MbarrierArriveSharedOp, MbarrierInitSharedOp, MbarrierInvalSharedOp, MbarrierTestWaitSharedOp,
    MbarrierTryWaitParitySharedOp, MbarrierTryWaitSharedOp, NanosleepOp, ThreadfenceBlockOp,
    ThreadfenceOp, ThreadfenceSystemOp,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;
/// Emits `sync_threads()`: CTA-wide barrier synchronization.
///
/// All threads in the CTA (Cooperative Thread Array) must reach the barrier
/// before any can proceed. This is the fundamental synchronization primitive.
///
/// # Generated Operation
///
/// `nvvm.barrier0` - Maps to PTX `bar.sync 0`
///
/// # Returns
///
/// void (no result value)
pub fn emit_sync_threads(
    ctx: &mut Context,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    use dialect_nvvm::ops::Barrier0Op;

    // Create the barrier operation
    let barrier_op = Operation::new(
        ctx,
        Barrier0Op::get_concrete_op_info(),
        vec![], // No results
        vec![], // No operands
        vec![], // No successors
        0,      // No regions
    );
    barrier_op.deref_mut(ctx).set_loc(loc.clone());

    // Insert the barrier operation
    let last_op = if let Some(prev) = prev_op {
        barrier_op.insert_after(ctx, prev);
        barrier_op
    } else {
        barrier_op.insert_at_front(block_ptr, ctx);
        barrier_op
    };

    // Branch to the call's success successor (barrier-like intrinsics always
    // have exactly one MIR successor).
    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, last_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("sync_threads call without target block".to_string(),)
        )
    }
}

/// Emit a zero-operand, zero-result synchronization op and branch to the
/// target block.
fn emit_zero_arg_void_sync_op<F>(
    ctx: &mut Context,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    build_op: F,
    op_name: &str,
) -> TranslationResult<Ptr<Operation>>
where
    F: FnOnce(&mut Context) -> Ptr<Operation>,
{
    let sync_op = build_op(ctx);
    sync_op.deref_mut(ctx).set_loc(loc.clone());

    let last_op = if let Some(prev) = prev_op {
        sync_op.insert_after(ctx, prev);
        sync_op
    } else {
        sync_op.insert_at_front(block_ptr, ctx);
        sync_op
    };

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, last_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!("{op_name} call without target block"))
        )
    }
}

/// Emits `threadfence_block()`: CTA-scoped memory fence.
pub fn emit_threadfence_block(
    ctx: &mut Context,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    emit_zero_arg_void_sync_op(
        ctx,
        target,
        block_ptr,
        prev_op,
        block_map,
        loc,
        |ctx| {
            Operation::new(
                ctx,
                ThreadfenceBlockOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                0,
            )
        },
        "threadfence_block",
    )
}

/// Emits `threadfence()`: device-scoped memory fence.
pub fn emit_threadfence(
    ctx: &mut Context,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    emit_zero_arg_void_sync_op(
        ctx,
        target,
        block_ptr,
        prev_op,
        block_map,
        loc,
        |ctx| {
            Operation::new(
                ctx,
                ThreadfenceOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                0,
            )
        },
        "threadfence",
    )
}

/// Emits `threadfence_system()`: system-scoped memory fence.
pub fn emit_threadfence_system(
    ctx: &mut Context,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    emit_zero_arg_void_sync_op(
        ctx,
        target,
        block_ptr,
        prev_op,
        block_map,
        loc,
        |ctx| {
            Operation::new(
                ctx,
                ThreadfenceSystemOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                0,
            )
        },
        "threadfence_system",
    )
}
/// Emits `mbarrier_init(bar_ptr, count)`: Initialize an mbarrier.
///
/// Sets up an asynchronous barrier in shared memory with the expected
/// arrival count. Must be called before any arrive/wait operations.
///
/// # Arguments
///
/// - `args[0]`: `*mut u64` - Pointer to barrier in shared memory
/// - `args[1]`: `u32` - Expected arrival count
///
/// # Generated Operation
///
/// `nvvm.mbarrier.init.shared` - Maps to PTX `mbarrier.init.shared.b64`
pub fn emit_mbarrier_init(
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
                "mbarrier_init expects 2 arguments, got {}",
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

    // Get the expected count (arg 1)
    let (count, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // Create the mbarrier_init_shared operation (void return)
    let init_op = Operation::new(
        ctx,
        MbarrierInitSharedOp::get_concrete_op_info(),
        vec![],               // No results
        vec![bar_ptr, count], // Operands: ptr, count
        vec![],
        0,
    );
    init_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        init_op.insert_after(ctx, prev);
    } else {
        init_op.insert_at_front(block_ptr, ctx);
    }

    // Emit goto to target block
    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, init_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("mbarrier_init call without target block".to_string(),)
        )
    }
}

/// Emit mbarrier_arrive: arrive at barrier, returns token.
///
/// Args:
/// - args[0]: *const Barrier (pointer to barrier in shared memory)
///
/// Returns: u64 (phase token)
pub fn emit_mbarrier_arrive(
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
                "mbarrier_arrive expects 1 argument, got {}",
                args.len()
            ))
        );
    }

    // Get the barrier pointer (arg 0)
    let (bar_ptr, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Result type: i64 (phase token)
    let i64_type = IntegerType::get(ctx, 64, Signedness::Unsigned);

    // Create the mbarrier_arrive_shared operation
    let arrive_op = Operation::new(
        ctx,
        MbarrierArriveSharedOp::get_concrete_op_info(),
        vec![i64_type.to_ptr()], // Result: i64 token
        vec![bar_ptr],           // Operand: ptr
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
        "mbarrier_arrive call without target block",
    )
}

/// Emit mbarrier_arrive_expect_tx: arrive at barrier with expected transaction bytes.
///
/// This is required for TMA's complete_tx::bytes mode. The barrier must be told
/// how many bytes to expect from the TMA transaction before the TMA is initiated.
///
/// Args:
/// - args[0]: *const Barrier (pointer to barrier in shared memory)
/// - args[1]: u32 (tx_count - unused, kept for API compatibility)
/// - args[2]: u32 (bytes - expected transaction byte count)
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
        vec![i64_type.to_ptr()], // Result: i64 token
        vec![bar_ptr, bytes],    // Operands: ptr, bytes
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

/// Emit mbarrier_arrive_cluster: arrive at a barrier in another CTA's shared memory.
///
/// Takes a raw u64 address (from map_shared_rank cast to integer) to avoid
/// LLVM IR address-space conflicts in loop phi nodes.
///
/// Args:
/// - args[0]: u64 (cluster-scope barrier address from mapa)
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
/// - args[0]: u32 (nanoseconds)
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

/// Emit mbarrier_test_wait: test if barrier phase is complete (non-blocking).
///
/// Args:
/// - args[0]: *const Barrier (pointer to barrier in shared memory)
/// - args[1]: u64 (phase token)
///
/// Returns: bool
pub fn emit_mbarrier_test_wait(
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
                "mbarrier_test_wait expects 2 arguments, got {}",
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

    // Create the mbarrier_test_wait_shared operation
    let test_wait_op = Operation::new(
        ctx,
        MbarrierTestWaitSharedOp::get_concrete_op_info(),
        vec![i1_type.to_ptr()], // Result: i1 (bool)
        vec![bar_ptr, token],   // Operands: ptr, token
        vec![],
        0,
    );
    test_wait_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        test_wait_op.insert_after(ctx, prev);
    } else {
        test_wait_op.insert_at_front(block_ptr, ctx);
    }

    // Store the result in the destination slot and branch to the success target.
    let result_value = test_wait_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        test_wait_op,
        value_map,
        block_map,
        loc,
        "mbarrier_test_wait call without target block",
    )
}

/// Emit mbarrier_try_wait: try to wait for barrier phase (with scheduling hints).
///
/// Similar to mbarrier_test_wait but uses try_wait which provides better scheduling
/// hints to the hardware. This is the preferred instruction for TMA synchronization.
///
/// Args:
/// - args[0]: *const Barrier (pointer to barrier in shared memory)
/// - args[1]: u64 (phase token)
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
        vec![i1_type.to_ptr()], // Result: i1 (bool)
        vec![bar_ptr, token],   // Operands: ptr, token
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
/// - args[0]: *const Barrier (ptr to barrier in shared)
/// - args[1]: u32 parity
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

/// Emit mbarrier_inval: invalidate a barrier.
///
/// Args:
/// - args[0]: *mut Barrier (pointer to barrier in shared memory)
///
/// Returns: void
pub fn emit_mbarrier_inval(
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
                "mbarrier_inval expects 1 argument, got {}",
                args.len()
            ))
        );
    }

    // Get the barrier pointer (arg 0)
    let (bar_ptr, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Create the mbarrier_inval_shared operation (void return)
    let inval_op = Operation::new(
        ctx,
        MbarrierInvalSharedOp::get_concrete_op_info(),
        vec![],        // No results
        vec![bar_ptr], // Operand: ptr
        vec![],
        0,
    );
    inval_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        inval_op.insert_after(ctx, prev);
    } else {
        inval_op.insert_at_front(block_ptr, ctx);
    }

    // Emit goto to target block
    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, inval_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("mbarrier_inval call without target block".to_string(),)
        )
    }
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
