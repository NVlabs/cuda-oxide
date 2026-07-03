// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Packed f16x2 ALU intrinsic conversions.
//!
//! Add, subtract, multiply, FMA, negation, and absolute value require
//! `sm_53+`. Min, max, and fused multiply-add with ReLU require `sm_80+`.

use llvm_export::ops::{self as llvm, AsmKind, InlineAsmOpExt};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

// ---------------------------------------------------------------------------
// Helper: convert a ternary f16x2 op to inline PTX
// ---------------------------------------------------------------------------

/// Convert a ternary packed f16x2 op (`$0 = op $1, $2, $3`) to inline PTX.
fn convert_ternary_f16x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    ptx_mnemonic: &str,
    name: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 3 {
        return pliron::input_err_noloc!("{} requires 3 operands", name);
    }

    let a_val = operands[0];
    let b_val = operands[1];
    let c_val = operands[2];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let asm_str = format!("{ptx_mnemonic} $0, $1, $2, $3;");
    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        i32_ty.into(),
        vec![a_val, b_val, c_val],
        &asm_str,
        "=r,r,r,r",
        AsmKind::Pure,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: convert a binary f16x2 op to inline PTX
// ---------------------------------------------------------------------------

/// Convert a binary packed f16x2 op (`$0 = op $1, $2`) to inline PTX.
fn convert_binary_f16x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    ptx_mnemonic: &str,
    name: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 2 {
        return pliron::input_err_noloc!("{} requires 2 operands", name);
    }

    let a_val = operands[0];
    let b_val = operands[1];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let asm_str = format!("{ptx_mnemonic} $0, $1, $2;");
    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        i32_ty.into(),
        vec![a_val, b_val],
        &asm_str,
        "=r,r,r",
        AsmKind::Pure,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: convert a unary f16x2 op to inline PTX
// ---------------------------------------------------------------------------

/// Convert a unary packed f16x2 op (`$0 = op $1`) to inline PTX.
fn convert_unary_f16x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    ptx_mnemonic: &str,
    name: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.is_empty() {
        return pliron::input_err_noloc!("{} requires 1 operand", name);
    }

    let a_val = operands[0];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let asm_str = format!("{ptx_mnemonic} $0, $1;");
    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        i32_ty.into(),
        vec![a_val],
        &asm_str,
        "=r,r",
        AsmKind::Pure,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

// ---------------------------------------------------------------------------
// Public conversion functions
// ---------------------------------------------------------------------------

/// Convert `nvvm.add_f16x2` to inline PTX: `add.rn.f16x2 $0, $1, $2;`
pub(crate) fn convert_add_f16x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_binary_f16x2(ctx, rewriter, op, "add.rn.f16x2", "add_f16x2")
}

/// Convert `nvvm.sub_f16x2` to inline PTX: `sub.rn.f16x2 $0, $1, $2;`
pub(crate) fn convert_sub_f16x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_binary_f16x2(ctx, rewriter, op, "sub.rn.f16x2", "sub_f16x2")
}

/// Convert `nvvm.mul_f16x2` to inline PTX: `mul.rn.f16x2 $0, $1, $2;`
pub(crate) fn convert_mul_f16x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_binary_f16x2(ctx, rewriter, op, "mul.rn.f16x2", "mul_f16x2")
}

/// Convert `nvvm.fma_f16x2` to inline PTX: `fma.rn.f16x2 $0, $1, $2, $3;`
pub(crate) fn convert_fma_f16x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ternary_f16x2(ctx, rewriter, op, "fma.rn.f16x2", "fma_f16x2")
}

/// Convert `nvvm.neg_f16x2` to inline PTX: `neg.f16x2 $0, $1;`
pub(crate) fn convert_neg_f16x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_unary_f16x2(ctx, rewriter, op, "neg.f16x2", "neg_f16x2")
}

/// Convert `nvvm.abs_f16x2` to inline PTX: `abs.f16x2 $0, $1;`
pub(crate) fn convert_abs_f16x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_unary_f16x2(ctx, rewriter, op, "abs.f16x2", "abs_f16x2")
}

/// Convert `nvvm.min_f16x2` to inline PTX: `min.f16x2 $0, $1, $2;`
pub(crate) fn convert_min_f16x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_binary_f16x2(ctx, rewriter, op, "min.f16x2", "min_f16x2")
}

/// Convert `nvvm.max_f16x2` to inline PTX: `max.f16x2 $0, $1, $2;`
pub(crate) fn convert_max_f16x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_binary_f16x2(ctx, rewriter, op, "max.f16x2", "max_f16x2")
}

/// Convert `nvvm.fma_relu_f16x2` to inline PTX: `fma.rn.relu.f16x2 $0, $1, $2, $3;`
pub(crate) fn convert_fma_relu_f16x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ternary_f16x2(ctx, rewriter, op, "fma.rn.relu.f16x2", "fma_relu_f16x2")
}
