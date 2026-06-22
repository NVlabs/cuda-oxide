/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Type conversion intrinsic lowering.
//!
//! | Operation              | PTX                                    |
//! |------------------------|----------------------------------------|
//! | `CvtF16x2F32`         | `cvt.rn.f16x2.f32 d, hi, lo;`         |
//! | `CvtRzF16x2F32`       | `cvt.rz.f16x2.f32 d, hi, lo;`         |
//! | `CvtRnReluF16x2F32`   | `cvt.rn.relu.f16x2.f32 d, hi, lo;`    |
//! | `CvtRnReluBf16x2F32`  | `cvt.rn.relu.bf16x2.f32 d, hi, lo;`   |
//! | `CvtRzBf16x2F32`      | `cvt.rz.bf16x2.f32 d, hi, lo;`        |

use llvm_export::ops::{self as llvm, InlineAsmOpExt};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Convert `cvt.rn.f16x2.f32`: pack two f32 values into f16x2 (u32).
///
/// Operands:
/// - `$1`: f32 lo value (bits `[15:0]`)
/// - `$2`: f32 hi value (bits `[31:16]`)
///
/// Result:
/// - `$0`: u32 packed f16x2
pub(crate) fn convert_cvt_f16x2_f32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 2 {
        return pliron::input_err_noloc!("cvt_f16x2_f32 requires 2 operands");
    }

    let lo_val = operands[0];
    let hi_val = operands[1];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    // Pure inline asm (data conversion, not a collective op)
    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        i32_ty.into(),
        vec![lo_val, hi_val],
        "cvt.rn.f16x2.f32 $0, $2, $1;",
        "=r,f,f",
        llvm_export::ops::AsmKind::Pure,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Convert `cvt.rz.f16x2.f32`: pack two f32 values into f16x2 (u32)
/// with truncation rounding.
pub(crate) fn convert_cvt_rz_f16x2_f32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 2 {
        return pliron::input_err_noloc!("cvt_rz_f16x2_f32 requires 2 operands");
    }

    let lo_val = operands[0];
    let hi_val = operands[1];
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        i32_ty.into(),
        vec![lo_val, hi_val],
        "cvt.rz.f16x2.f32 $0, $2, $1;",
        "=r,f,f",
        llvm_export::ops::AsmKind::Pure,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Convert `cvt.rn.relu.f16x2.f32`: pack two f32 values into f16x2 (u32)
/// with fused ReLU.
pub(crate) fn convert_cvt_rn_relu_f16x2_f32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 2 {
        return pliron::input_err_noloc!("cvt_rn_relu_f16x2_f32 requires 2 operands");
    }

    let lo_val = operands[0];
    let hi_val = operands[1];
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        i32_ty.into(),
        vec![lo_val, hi_val],
        "cvt.rn.relu.f16x2.f32 $0, $2, $1;",
        "=r,f,f",
        llvm_export::ops::AsmKind::Pure,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Convert `cvt.rn.relu.bf16x2.f32`: pack two f32 values into bf16x2 (u32)
/// with fused ReLU.
pub(crate) fn convert_cvt_rn_relu_bf16x2_f32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 2 {
        return pliron::input_err_noloc!("cvt_rn_relu_bf16x2_f32 requires 2 operands");
    }

    let lo_val = operands[0];
    let hi_val = operands[1];
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        i32_ty.into(),
        vec![lo_val, hi_val],
        "cvt.rn.relu.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        llvm_export::ops::AsmKind::Pure,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Convert `cvt.rz.bf16x2.f32`: pack two f32 values into bf16x2 (u32)
/// with truncation rounding.
pub(crate) fn convert_cvt_rz_bf16x2_f32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 2 {
        return pliron::input_err_noloc!("cvt_rz_bf16x2_f32 requires 2 operands");
    }

    let lo_val = operands[0];
    let hi_val = operands[1];
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        i32_ty.into(),
        vec![lo_val, hi_val],
        "cvt.rz.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        llvm_export::ops::AsmKind::Pure,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}
