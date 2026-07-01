/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Import ldmatrix calls as register-producing NVVM-dialect operations.
//!
//! The PTX instruction returns register fragments, so x2 and x4 stay as SSA
//! values here. They are bundled into the Rust array result only after the
//! dialect operation, avoiding a temporary stack slot and inline-assembly
//! stores.

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_mir::{ops::MirConstructArrayOp, types::MirArrayType};
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

type OpInfo = (fn(Ptr<Operation>) -> pliron::op::OpObj, std::any::TypeId);

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
    emit_ldmatrix(
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
        1,
    )
}

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
    emit_ldmatrix(
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
        1,
    )
}

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
    emit_ldmatrix(
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
        2,
    )
}

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
    emit_ldmatrix(
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
        2,
    )
}

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
    emit_ldmatrix(
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
        4,
    )
}

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
    emit_ldmatrix(
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
        4,
    )
}

#[allow(clippy::too_many_arguments)]
fn emit_ldmatrix(
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
    op_info: OpInfo,
    name: &str,
    register_count: usize,
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

    let (smem_ptr, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
    let result_types = (0..register_count).map(|_| u32_ty.into()).collect();
    let load = Operation::new(ctx, op_info, result_types, vec![smem_ptr], vec![], 0);
    load.deref_mut(ctx).set_loc(loc.clone());
    if let Some(previous) = last_op {
        load.insert_after(ctx, previous);
    } else {
        load.insert_at_front(block_ptr, ctx);
    }

    let (value, last_op) = if register_count == 1 {
        (load.deref(ctx).get_result(0), load)
    } else {
        let registers = (0..register_count)
            .map(|index| load.deref(ctx).get_result(index))
            .collect();
        let array_ty = MirArrayType::get(ctx, u32_ty.into(), register_count as u64);
        let array = Operation::new(
            ctx,
            MirConstructArrayOp::get_concrete_op_info(),
            vec![array_ty.into()],
            registers,
            vec![],
            0,
        );
        array.deref_mut(ctx).set_loc(loc.clone());
        array.insert_after(ctx, load);
        (array.deref(ctx).get_result(0), array)
    };

    emit_store_result_and_goto(
        ctx,
        destination,
        value,
        target,
        block_ptr,
        last_op,
        value_map,
        block_map,
        loc,
        &format!("{name} call without target block"),
    )
}
