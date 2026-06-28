// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Packed bf16x2 ALU intrinsics.
//!
//! FMA, min, max, negation, and absolute value require `sm_80+`. Add,
//! subtract, and multiply require `sm_90+`.

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{
    AbsBf16x2Op, AddBf16x2Op, FmaBf16x2Op, FmaReluBf16x2Op, MaxBf16x2Op, MinBf16x2Op, MulBf16x2Op,
    NegBf16x2Op, SubBf16x2Op,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::{Op, OpObj};
use pliron::operation::Operation;
use rustc_public::mir;

// ---------------------------------------------------------------------------
// Helper: emit a ternary bf16x2 op (3 u32 inputs, 1 u32 output)
// ---------------------------------------------------------------------------

/// Emit a ternary packed bf16x2 operation.
fn emit_ternary_bf16x2(
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
    op_info: (fn(Ptr<Operation>) -> OpObj, std::any::TypeId),
    name: &str,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{name} expects 3 arguments (a: u32, b: u32, c: u32), got {}",
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

    let tern_op = Operation::new(
        ctx,
        op_info,
        vec![u32_ty.into()],
        vec![a_val, b_val, c_val],
        vec![],
        0,
    );
    tern_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        tern_op.insert_after(ctx, prev);
    } else {
        tern_op.insert_at_front(block_ptr, ctx);
    }

    let result = tern_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        tern_op,
        value_map,
        block_map,
        loc,
        &format!("{name} call without target block"),
    )
}

// ---------------------------------------------------------------------------
// Helper: emit a binary bf16x2 op (2 u32 inputs, 1 u32 output)
// ---------------------------------------------------------------------------

/// Emit a binary packed bf16x2 operation.
fn emit_binary_bf16x2(
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
    op_info: (fn(Ptr<Operation>) -> OpObj, std::any::TypeId),
    name: &str,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{name} expects 2 arguments (a: u32, b: u32), got {}",
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

    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let bin_op = Operation::new(
        ctx,
        op_info,
        vec![u32_ty.into()],
        vec![a_val, b_val],
        vec![],
        0,
    );
    bin_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        bin_op.insert_after(ctx, prev);
    } else {
        bin_op.insert_at_front(block_ptr, ctx);
    }

    let result = bin_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        bin_op,
        value_map,
        block_map,
        loc,
        &format!("{name} call without target block"),
    )
}

// ---------------------------------------------------------------------------
// Helper: emit a unary bf16x2 op (1 u32 input, 1 u32 output)
// ---------------------------------------------------------------------------

/// Emit a unary packed bf16x2 operation.
fn emit_unary_bf16x2(
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
    op_info: (fn(Ptr<Operation>) -> OpObj, std::any::TypeId),
    name: &str,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{name} expects 1 argument (a: u32), got {}",
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

    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let un_op = Operation::new(ctx, op_info, vec![u32_ty.into()], vec![a_val], vec![], 0);
    un_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        un_op.insert_after(ctx, prev);
    } else {
        un_op.insert_at_front(block_ptr, ctx);
    }

    let result = un_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        un_op,
        value_map,
        block_map,
        loc,
        &format!("{name} call without target block"),
    )
}

// ---------------------------------------------------------------------------
// Public emit functions
// ---------------------------------------------------------------------------

/// Emit `fma_bf16x2`: packed bf16x2 fused multiply-add.
///
/// Args: `(a: u32, b: u32, c: u32)`, each carrying two packed bf16 lanes.
/// Returns: `u32`, packed bf16x2 of `a * b + c`.
pub fn emit_fma_bf16x2(
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
    emit_ternary_bf16x2(
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
        FmaBf16x2Op::get_concrete_op_info(),
        "fma_bf16x2",
    )
}

/// Emit `fma_relu_bf16x2`: packed bf16x2 fused multiply-add with ReLU.
///
/// Args: `(a: u32, b: u32, c: u32)`, each carrying two packed bf16 lanes.
/// Returns: `u32`, packed bf16x2 of `max(0, a * b + c)`.
pub fn emit_fma_relu_bf16x2(
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
    emit_ternary_bf16x2(
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
        FmaReluBf16x2Op::get_concrete_op_info(),
        "fma_relu_bf16x2",
    )
}

/// Emit `add_bf16x2`: packed bf16x2 addition.
pub fn emit_add_bf16x2(
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
    emit_binary_bf16x2(
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
        AddBf16x2Op::get_concrete_op_info(),
        "add_bf16x2",
    )
}

/// Emit `sub_bf16x2`: packed bf16x2 subtraction.
pub fn emit_sub_bf16x2(
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
    emit_binary_bf16x2(
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
        SubBf16x2Op::get_concrete_op_info(),
        "sub_bf16x2",
    )
}

/// Emit `mul_bf16x2`: packed bf16x2 multiplication.
pub fn emit_mul_bf16x2(
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
    emit_binary_bf16x2(
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
        MulBf16x2Op::get_concrete_op_info(),
        "mul_bf16x2",
    )
}

/// Emit `min_bf16x2`: packed bf16x2 minimum.
pub fn emit_min_bf16x2(
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
    emit_binary_bf16x2(
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
        MinBf16x2Op::get_concrete_op_info(),
        "min_bf16x2",
    )
}

/// Emit `max_bf16x2`: packed bf16x2 maximum.
pub fn emit_max_bf16x2(
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
    emit_binary_bf16x2(
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
        MaxBf16x2Op::get_concrete_op_info(),
        "max_bf16x2",
    )
}

/// Emit `neg_bf16x2`: packed bf16x2 negation.
pub fn emit_neg_bf16x2(
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
    emit_unary_bf16x2(
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
        NegBf16x2Op::get_concrete_op_info(),
        "neg_bf16x2",
    )
}

/// Emit `abs_bf16x2`: packed bf16x2 absolute value.
pub fn emit_abs_bf16x2(
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
    emit_unary_bf16x2(
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
        AbsBf16x2Op::get_concrete_op_info(),
        "abs_bf16x2",
    )
}
