// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integer dot product intrinsics (`dp4a`, `dp2a`).
//!
//! Translates `cuda_device::dotprod::*` calls into `dialect-nvvm` dot product
//! operations.

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{Dp2aS32Op, Dp2aU32Op, Dp4aS32Op, Dp4aU32Op};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

/// Emit `dp4a_s32`: signed 4-element byte dot product + accumulate.
///
/// Args: `(a: u32, b: u32, c: i32)`. Returns: `i32`.
pub fn emit_dp4a_s32(
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
                "dp4a_s32 expects 3 arguments (a: u32, b: u32, c: i32), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    let (a_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (b_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (c_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signed);

    let dp_op = Operation::new(
        ctx,
        Dp4aS32Op::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![a_val, b_val, c_val],
        vec![],
        0,
    );
    dp_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        dp_op.insert_after(ctx, prev);
    } else {
        dp_op.insert_at_front(block_ptr, ctx);
    }

    let result = dp_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        dp_op,
        value_map,
        block_map,
        loc,
        "dp4a_s32 call without target block",
    )
}

/// Emit `dp4a_u32`: unsigned 4-element byte dot product + accumulate.
///
/// Args: `(a: u32, b: u32, c: u32)`. Returns: `u32`.
pub fn emit_dp4a_u32(
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
                "dp4a_u32 expects 3 arguments (a: u32, b: u32, c: u32), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    let (a_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (b_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (c_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let dp_op = Operation::new(
        ctx,
        Dp4aU32Op::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![a_val, b_val, c_val],
        vec![],
        0,
    );
    dp_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        dp_op.insert_after(ctx, prev);
    } else {
        dp_op.insert_at_front(block_ptr, ctx);
    }

    let result = dp_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        dp_op,
        value_map,
        block_map,
        loc,
        "dp4a_u32 call without target block",
    )
}

/// Emit `dp2a_s32`: signed 2-element half-word × byte dot product + accumulate.
///
/// Args: `(a: u32, b: u32, c: i32)`. Returns: `i32`.
pub fn emit_dp2a_s32(
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
                "dp2a_s32 expects 3 arguments (a: u32, b: u32, c: i32), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    let (a_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (b_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (c_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signed);

    let dp_op = Operation::new(
        ctx,
        Dp2aS32Op::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![a_val, b_val, c_val],
        vec![],
        0,
    );
    dp_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        dp_op.insert_after(ctx, prev);
    } else {
        dp_op.insert_at_front(block_ptr, ctx);
    }

    let result = dp_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        dp_op,
        value_map,
        block_map,
        loc,
        "dp2a_s32 call without target block",
    )
}

/// Emit `dp2a_u32`: unsigned 2-element half-word × byte dot product + accumulate.
///
/// Args: `(a: u32, b: u32, c: u32)`. Returns: `u32`.
pub fn emit_dp2a_u32(
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
                "dp2a_u32 expects 3 arguments (a: u32, b: u32, c: u32), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    let (a_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (b_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (c_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let dp_op = Operation::new(
        ctx,
        Dp2aU32Op::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![a_val, b_val, c_val],
        vec![],
        0,
    );
    dp_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        dp_op.insert_after(ctx, prev);
    } else {
        dp_op.insert_at_front(block_ptr, ctx);
    }

    let result = dp_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        dp_op,
        value_map,
        block_map,
        loc,
        "dp2a_u32 call without target block",
    )
}
