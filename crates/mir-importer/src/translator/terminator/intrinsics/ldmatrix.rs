/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ldmatrix intrinsic emission for warp-cooperative matrix loads.
//!
//! # Return Patterns
//!
//! - **x1 variants**: Scalar u32 return via `emit_store_result_and_goto`.
//! - **x2/x4 variants**: Alloca-slot pattern -- the IR op is VOID and the
//!   lowering writes directly to the destination slot via inline PTX stores.

use super::super::helpers::{emit_goto, emit_store_result_and_goto};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{
    LdmatrixX1Op, LdmatrixX1TransOp, LdmatrixX2Op, LdmatrixX2TransOp, LdmatrixX4Op,
    LdmatrixX4TransOp,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

// =============================================================================
// x1 variants (scalar return: 1 operand, 1 result)
// =============================================================================

/// Emits `ldmatrix.x1`: load one 8×8 tile from shared memory.
///
/// # Arguments
///
/// - `args[0]`: `*const u32` - Source pointer in shared memory
///
/// # Returns
///
/// `u32` - single loaded register
#[allow(clippy::too_many_arguments)]
pub fn emit_ldmatrix_x1(
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
    emit_ldmatrix_x1_impl(
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
        LdmatrixX1Op::get_concrete_op_info(),
        "ldmatrix_x1",
    )
}

/// Emits `ldmatrix.x1.trans`: load one 8×8 tile with transpose.
#[allow(clippy::too_many_arguments)]
pub fn emit_ldmatrix_x1_trans(
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
    emit_ldmatrix_x1_impl(
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
        LdmatrixX1TransOp::get_concrete_op_info(),
        "ldmatrix_x1_trans",
    )
}

/// Shared implementation for x1 variants (scalar return).
#[allow(clippy::too_many_arguments)]
fn emit_ldmatrix_x1_impl(
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
    op_info: (fn(Ptr<Operation>) -> pliron::op::OpObj, std::any::TypeId),
    name: &str,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{name} expects 1 argument (smem_ptr), got {}",
                args.len()
            ))
        );
    }

    let (smem_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let u32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let ld_op = Operation::new(
        ctx,
        op_info,
        vec![u32_type.to_handle()],
        vec![smem_val],
        vec![],
        0,
    );
    ld_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        ld_op.insert_after(ctx, prev);
    } else {
        ld_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = ld_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        ld_op,
        value_map,
        block_map,
        loc,
        &format!("{name} call without target block"),
    )
}

// =============================================================================
// x2/x4 variants (alloca-slot: 2 operands [smem_ptr, dest_ptr], 0 results)
// =============================================================================

/// Emits `ldmatrix.x2`: load two 8×8 tiles from shared memory.
#[allow(clippy::too_many_arguments)]
pub fn emit_ldmatrix_x2(
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
    emit_ldmatrix_array_impl(
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
        LdmatrixX2Op::get_concrete_op_info(),
        "ldmatrix_x2",
    )
}

/// Emits `ldmatrix.x2.trans`: load two 8×8 tiles with transpose.
#[allow(clippy::too_many_arguments)]
pub fn emit_ldmatrix_x2_trans(
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
    emit_ldmatrix_array_impl(
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
        LdmatrixX2TransOp::get_concrete_op_info(),
        "ldmatrix_x2_trans",
    )
}

/// Emits `ldmatrix.x4`: load four 8×8 tiles from shared memory.
#[allow(clippy::too_many_arguments)]
pub fn emit_ldmatrix_x4(
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
    emit_ldmatrix_array_impl(
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
        LdmatrixX4Op::get_concrete_op_info(),
        "ldmatrix_x4",
    )
}

/// Emits `ldmatrix.x4.trans`: load four 8×8 tiles with transpose.
#[allow(clippy::too_many_arguments)]
pub fn emit_ldmatrix_x4_trans(
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
    emit_ldmatrix_array_impl(
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
        LdmatrixX4TransOp::get_concrete_op_info(),
        "ldmatrix_x4_trans",
    )
}

/// Shared implementation for x2/x4 variants (alloca-slot pattern).
///
/// Creates a void IR op with 2 operands `[smem_ptr, dest_ptr]`. The lowering
/// emits inline PTX that loads from shared memory and stores results directly
/// into the destination alloca slot.
#[allow(clippy::too_many_arguments)]
fn emit_ldmatrix_array_impl(
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
    op_info: (fn(Ptr<Operation>) -> pliron::op::OpObj, std::any::TypeId),
    name: &str,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{name} expects 1 argument (smem_ptr), got {}",
                args.len()
            ))
        );
    }

    let (smem_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Get the destination alloca slot pointer for writing the array result.
    let dest_ptr = match value_map.get_slot(destination.local) {
        Some(slot) => slot,
        None => {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "{name}: destination local has no backing alloca slot"
                ))
            );
        }
    };

    // Create void op: 0 results, 2 operands [smem_ptr, dest_ptr]
    let ld_op = Operation::new(ctx, op_info, vec![], vec![smem_val, dest_ptr], vec![], 0);
    ld_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        ld_op.insert_after(ctx, prev);
    } else {
        ld_op.insert_at_front(block_ptr, ctx);
    }

    // No result to store -- the inline asm writes directly to dest_ptr.
    // Just emit the goto to the successor block.
    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, ld_op, block_map, loc))
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!("{name} call without target block"))
        )
    }
}
