/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Type conversion intrinsics.
//!
//! Translates `cuda_device::convert::*` intrinsic calls into `dialect-nvvm`
//! conversion operations.

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{
    CvtF16x2F32Op, CvtRnReluBf16x2F32Op, CvtRnReluF16x2F32Op, CvtRzBf16x2F32Op, CvtRzF16x2F32Op,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

/// Emit cvt_f16x2_f32: convert two f32 values to packed f16x2 (u32).
///
/// Args:
/// - `args[0]`: f32 (lo: value for bits `[15:0]`)
/// - `args[1]`: f32 (hi: value for bits `[31:16]`)
///
/// Returns: u32 (packed f16x2)
pub fn emit_cvt_f16x2_f32(
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
                "cvt_f16x2_f32 expects 2 arguments, got {}",
                args.len()
            ))
        );
    }

    let (lo_val, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let (hi_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let i32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let cvt_op = Operation::new(
        ctx,
        CvtF16x2F32Op::get_concrete_op_info(),
        vec![i32_type.to_handle()],
        vec![lo_val, hi_val],
        vec![],
        0,
    );
    cvt_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        cvt_op.insert_after(ctx, prev);
    } else {
        cvt_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = cvt_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        cvt_op,
        value_map,
        block_map,
        loc,
        "cvt_f16x2_f32 call without target block",
    )
}

/// Emit cvt_rz_f16x2_f32: convert two f32 values to packed f16x2 (u32)
/// with truncation rounding.
pub fn emit_cvt_rz_f16x2_f32(
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
                "cvt_rz_f16x2_f32 expects 2 arguments, got {}",
                args.len()
            ))
        );
    }

    let (lo_val, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let (hi_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let i32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);
    let cvt_op = Operation::new(
        ctx,
        CvtRzF16x2F32Op::get_concrete_op_info(),
        vec![i32_type.to_handle()],
        vec![lo_val, hi_val],
        vec![],
        0,
    );
    cvt_op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(prev) = last_op {
        cvt_op.insert_after(ctx, prev);
    } else {
        cvt_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = cvt_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        cvt_op,
        value_map,
        block_map,
        loc,
        "cvt_rz_f16x2_f32 call without target block",
    )
}

/// Emit cvt_rn_relu_f16x2_f32: convert two f32 values to packed f16x2 (u32)
/// with fused ReLU.
pub fn emit_cvt_rn_relu_f16x2_f32(
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
                "cvt_rn_relu_f16x2_f32 expects 2 arguments, got {}",
                args.len()
            ))
        );
    }

    let (lo_val, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let (hi_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let i32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);
    let cvt_op = Operation::new(
        ctx,
        CvtRnReluF16x2F32Op::get_concrete_op_info(),
        vec![i32_type.to_handle()],
        vec![lo_val, hi_val],
        vec![],
        0,
    );
    cvt_op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(prev) = last_op {
        cvt_op.insert_after(ctx, prev);
    } else {
        cvt_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = cvt_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        cvt_op,
        value_map,
        block_map,
        loc,
        "cvt_rn_relu_f16x2_f32 call without target block",
    )
}

/// Emit cvt_rn_relu_bf16x2_f32: convert two f32 values to packed bf16x2 (u32)
/// with fused ReLU.
pub fn emit_cvt_rn_relu_bf16x2_f32(
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
                "cvt_rn_relu_bf16x2_f32 expects 2 arguments, got {}",
                args.len()
            ))
        );
    }

    let (lo_val, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let (hi_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let i32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);
    let cvt_op = Operation::new(
        ctx,
        CvtRnReluBf16x2F32Op::get_concrete_op_info(),
        vec![i32_type.to_handle()],
        vec![lo_val, hi_val],
        vec![],
        0,
    );
    cvt_op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(prev) = last_op {
        cvt_op.insert_after(ctx, prev);
    } else {
        cvt_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = cvt_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        cvt_op,
        value_map,
        block_map,
        loc,
        "cvt_rn_relu_bf16x2_f32 call without target block",
    )
}

/// Emit cvt_rz_bf16x2_f32: convert two f32 values to packed bf16x2 (u32)
/// with truncation rounding.
pub fn emit_cvt_rz_bf16x2_f32(
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
                "cvt_rz_bf16x2_f32 expects 2 arguments, got {}",
                args.len()
            ))
        );
    }

    let (lo_val, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let (hi_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let i32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);
    let cvt_op = Operation::new(
        ctx,
        CvtRzBf16x2F32Op::get_concrete_op_info(),
        vec![i32_type.to_handle()],
        vec![lo_val, hi_val],
        vec![],
        0,
    );
    cvt_op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(prev) = last_op {
        cvt_op.insert_after(ctx, prev);
    } else {
        cvt_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = cvt_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        cvt_op,
        value_map,
        block_map,
        loc,
        "cvt_rz_bf16x2_f32 call without target block",
    )
}
