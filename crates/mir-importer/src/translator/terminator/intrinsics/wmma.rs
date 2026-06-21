/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ampere+ ldmatrix intrinsics.
//!
//! Handles SM75+ warp-level shared memory matrix load operations.

use super::super::helpers::emit_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{
    LdmatrixX2Op, LdmatrixX2TransOp, LdmatrixX4Op, LdmatrixX4TransOp,
};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

/// Helper: get the alloca slot pointer for a destination place.
///
/// For ldmatrix intrinsics, the destination is a `[u32; N]` local.
/// We need its alloca pointer so the lowered inline PTX can store directly into it.
fn get_dest_slot(
    value_map: &ValueMap,
    destination: &mir::Place,
    loc: &Location,
    intrinsic_name: &str,
) -> TranslationResult<pliron::value::Value> {
    match value_map.get_slot(destination.local) {
        Some(slot) => Ok(slot),
        None => input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{intrinsic_name}: destination local has no backing slot"
            ))
        ),
    }
}

/// Shared implementation for all ldmatrix variants.
///
/// All variants share identical logic: validate 1 argument, translate
/// the shared memory pointer operand, get the destination slot, create the op,
/// and emit a goto. They differ only in the dialect op type `T`.
fn emit_ldmatrix_impl<T: Op>(
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
    name: &str,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{name} expects 1 argument, got {}",
                args.len()
            ))
        );
    }

    let (smem_ptr, last_op) = rvalue::translate_operand(
        ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),
    )?;

    let dest_ptr = get_dest_slot(value_map, destination, &loc, name)?;

    let op = Operation::new(
        ctx,
        T::get_concrete_op_info(),
        vec![],
        vec![smem_ptr, dest_ptr],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(prev) = last_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, op, block_map, loc))
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!("{name} call without target block"))
        )
    }
}

/// Emit ldmatrix_x4: Load 4 × 8×8 matrices from shared memory.
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
    emit_ldmatrix_impl::<LdmatrixX4Op>(
        ctx, body, args, destination, target, block_ptr, prev_op, value_map, block_map, loc,
        "ldmatrix_x4",
    )
}

/// Emit ldmatrix_x2: Load 2 × 8×8 matrices from shared memory.
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
    emit_ldmatrix_impl::<LdmatrixX2Op>(
        ctx, body, args, destination, target, block_ptr, prev_op, value_map, block_map, loc,
        "ldmatrix_x2",
    )
}

/// Emit ldmatrix_x4_trans: Load 4 × 8×8 matrices with transpose.
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
    emit_ldmatrix_impl::<LdmatrixX4TransOp>(
        ctx, body, args, destination, target, block_ptr, prev_op, value_map, block_map, loc,
        "ldmatrix_x4_trans",
    )
}

/// Emit ldmatrix_x2_trans: Load 2 × 8×8 matrices with transpose.
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
    emit_ldmatrix_impl::<LdmatrixX2TransOp>(
        ctx, body, args, destination, target, block_ptr, prev_op, value_map, block_map, loc,
        "ldmatrix_x2_trans",
    )
}
