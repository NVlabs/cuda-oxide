/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Debug and profiling intrinsics.
//!
//! Handles translation of debug/profiling primitives including:
//! - `clock()` - 32-bit GPU clock counter
//! - `clock64()` - 64-bit GPU clock counter
//! - `globaltimer()` - 64-bit GPU global timer
//! - `trap()` - Abort kernel execution
//! - `breakpoint()` - cuda-gdb breakpoint
//! - `prof_trigger()` - Profiler event trigger
//! - `__gpu_vprintf()` - Formatted output to host console

use super::super::helpers::{emit_goto, emit_store_result_and_goto};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{
    BreakpointOp, PmEventOp, ReadPtxSregClock64Op, ReadPtxSregClockOp, ReadPtxSregGlobaltimerOp,
    TrapOp, VprintfOp,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;
/// Emits `clock()`: Read 32-bit GPU clock counter.
///
/// # Generated Operation
///
/// `nvvm.read_ptx_sreg_clock` - Maps to PTX `mov.u32 %r, %clock;`
///
/// # Returns
///
/// u32 clock cycle count
pub fn emit_clock(
    ctx: &mut Context,
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    // Result type: i32
    let i32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);

    // Create the clock operation
    let clock_op = Operation::new(
        ctx,
        ReadPtxSregClockOp::get_concrete_op_info(),
        vec![i32_type.to_ptr()], // Result: i32
        vec![],                  // No operands
        vec![],
        0,
    );
    clock_op.deref_mut(ctx).set_loc(loc.clone());

    // Insert the operation
    if let Some(prev) = prev_op {
        clock_op.insert_after(ctx, prev);
    } else {
        clock_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = clock_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        clock_op,
        value_map,
        block_map,
        loc,
        "clock call without target block",
    )
}

/// Emits `clock64()`: Read 64-bit GPU clock counter.
///
/// # Generated Operation
///
/// `nvvm.read_ptx_sreg_clock64` - Maps to PTX `mov.u64 %rd, %clock64;`
///
/// # Returns
///
/// u64 clock cycle count
pub fn emit_clock64(
    ctx: &mut Context,
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    // Result type: i64
    let i64_type = IntegerType::get(ctx, 64, Signedness::Unsigned);

    // Create the clock64 operation
    let clock_op = Operation::new(
        ctx,
        ReadPtxSregClock64Op::get_concrete_op_info(),
        vec![i64_type.to_ptr()], // Result: i64
        vec![],                  // No operands
        vec![],
        0,
    );
    clock_op.deref_mut(ctx).set_loc(loc.clone());

    // Insert the operation
    if let Some(prev) = prev_op {
        clock_op.insert_after(ctx, prev);
    } else {
        clock_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = clock_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        clock_op,
        value_map,
        block_map,
        loc,
        "clock64 call without target block",
    )
}

/// Emits `globaltimer()`: Read 64-bit GPU global timer.
///
/// # Generated Operation
///
/// `nvvm.read_ptx_sreg_globaltimer` - Maps to PTX `mov.u64 %rd, %globaltimer;`
///
/// # Returns
///
/// u64 global timer tick count
pub fn emit_globaltimer(
    ctx: &mut Context,
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let i64_type = IntegerType::get(ctx, 64, Signedness::Unsigned);

    let timer_op = Operation::new(
        ctx,
        ReadPtxSregGlobaltimerOp::get_concrete_op_info(),
        vec![i64_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    timer_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = prev_op {
        timer_op.insert_after(ctx, prev);
    } else {
        timer_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = timer_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        timer_op,
        value_map,
        block_map,
        loc,
        "globaltimer call without target block",
    )
}

/// Emits `trap()`: Abort kernel execution.
///
/// # Generated Operation
///
/// `nvvm.trap` followed by `mir.unreachable` - Maps to PTX `trap;`
///
/// # Returns
///
/// Never returns (divergent) - terminates the block with unreachable
pub fn emit_trap(
    ctx: &mut Context,
    _target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    _block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    // Create the trap operation (void, no return)
    let trap_op = Operation::new(
        ctx,
        TrapOp::get_concrete_op_info(),
        vec![], // No results
        vec![], // No operands
        vec![],
        0,
    );
    trap_op.deref_mut(ctx).set_loc(loc.clone());

    // Insert the trap operation
    if let Some(prev) = prev_op {
        trap_op.insert_after(ctx, prev);
    } else {
        trap_op.insert_at_front(block_ptr, ctx);
    }

    // trap() never returns, so we terminate the block with unreachable
    // (no goto needed since control flow ends here)
    let unreachable_op = Operation::new(
        ctx,
        dialect_mir::ops::MirUnreachableOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    unreachable_op.deref_mut(ctx).set_loc(loc);
    unreachable_op.insert_after(ctx, trap_op);

    Ok(unreachable_op)
}

/// Emits `breakpoint()`: Insert cuda-gdb breakpoint.
///
/// # Generated Operation
///
/// `nvvm.brkpt` - Maps to PTX `brkpt;`
///
/// # Returns
///
/// void
pub fn emit_breakpoint(
    ctx: &mut Context,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    // Create the breakpoint operation (void)
    let brkpt_op = Operation::new(
        ctx,
        BreakpointOp::get_concrete_op_info(),
        vec![], // No results
        vec![], // No operands
        vec![],
        0,
    );
    brkpt_op.deref_mut(ctx).set_loc(loc.clone());

    // Insert the operation
    if let Some(prev) = prev_op {
        brkpt_op.insert_after(ctx, prev);
    } else {
        brkpt_op.insert_at_front(block_ptr, ctx);
    }

    // Emit goto to target block
    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, brkpt_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("breakpoint call without target block".to_string(),)
        )
    }
}

/// Emits `prof_trigger::<N>()`: Signal profiler event.
///
/// # Generated Operation
///
/// `nvvm.pmevent` - Maps to PTX `pmevent N;`
///
/// # Arguments
///
/// * `event_id` - The profiler event ID (extracted from const generic)
///
/// # Returns
///
/// void
#[allow(clippy::too_many_arguments)]
pub fn emit_prof_trigger(
    ctx: &mut Context,
    event_id: u32,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    // Create the pmevent operation with event_id as an attribute
    let pmevent_op = PmEventOp::new_with_event_id(ctx, event_id);
    pmevent_op.deref_mut(ctx).set_loc(loc.clone());

    // Insert the operation
    if let Some(prev) = prev_op {
        pmevent_op.insert_after(ctx, prev);
    } else {
        pmevent_op.insert_at_front(block_ptr, ctx);
    }

    // Emit goto to target block
    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, pmevent_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("prof_trigger call without target block".to_string(),)
        )
    }
}

/// Emits `__gpu_vprintf()`: Formatted output to host console.
///
/// # Generated Operation
///
/// `nvvm.vprintf` - Maps to CUDA `vprintf(format, args)`
///
/// # Arguments
///
/// * `args[0]` - Pointer to null-terminated format string (*const u8)
/// * `args[1]` - Pointer to packed argument buffer (*const u8)
///
/// # Returns
///
/// i32 - Number of arguments on success (negative on error)
#[allow(clippy::too_many_arguments)]
pub fn emit_vprintf(
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
    use crate::translator::rvalue;

    // Validate we have exactly 2 arguments: format_ptr and args_ptr
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "__gpu_vprintf expects 2 arguments, got {}",
                args.len()
            ))
        );
    }

    // Translate the format pointer operand
    let (format_ptr, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Translate the args pointer operand
    let (args_ptr, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    // Create the vprintf operation
    let vprintf_op = VprintfOp::build(ctx, format_ptr, args_ptr);
    vprintf_op.deref_mut(ctx).set_loc(loc.clone());

    // Insert the operation
    if let Some(prev) = last_op {
        vprintf_op.insert_after(ctx, prev);
    } else {
        vprintf_op.insert_at_front(block_ptr, ctx);
    }

    // Store the result (i32) in the destination
    let result_value = vprintf_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        vprintf_op,
        value_map,
        block_map,
        loc,
        "__gpu_vprintf call without target block",
    )
}
