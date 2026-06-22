// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integer dot product intrinsic conversions (`dp4a`, `dp2a`).
//!
//! Each op is lowered to inline PTX assembly (non-convergent, pure).

use llvm_export::ops::{self as llvm, AsmKind, InlineAsmOpExt};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Convert `nvvm.dp4a_s32` to inline PTX.
///
/// `dp4a.s32.s32 %d, %a, %b, %c;` (per-thread arithmetic, non-convergent).
pub(crate) fn convert_dp4a_s32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 3 {
        return pliron::input_err_noloc!("dp4a_s32 requires 3 operands");
    }

    let a_val = operands[0];
    let b_val = operands[1];
    let c_val = operands[2];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        i32_ty.into(),
        vec![a_val, b_val, c_val],
        "dp4a.s32.s32 $0, $1, $2, $3;",
        "=r,r,r,r",
        AsmKind::Pure,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Convert `nvvm.dp4a_u32` to inline PTX.
///
/// `dp4a.u32.u32 %d, %a, %b, %c;` (per-thread arithmetic, non-convergent).
pub(crate) fn convert_dp4a_u32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 3 {
        return pliron::input_err_noloc!("dp4a_u32 requires 3 operands");
    }

    let a_val = operands[0];
    let b_val = operands[1];
    let c_val = operands[2];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        i32_ty.into(),
        vec![a_val, b_val, c_val],
        "dp4a.u32.u32 $0, $1, $2, $3;",
        "=r,r,r,r",
        AsmKind::Pure,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Convert `nvvm.dp2a_s32` to inline PTX.
///
/// `dp2a.lo.s32.s32 %d, %a, %b, %c;` (per-thread arithmetic, non-convergent).
pub(crate) fn convert_dp2a_s32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 3 {
        return pliron::input_err_noloc!("dp2a_s32 requires 3 operands");
    }

    let a_val = operands[0];
    let b_val = operands[1];
    let c_val = operands[2];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        i32_ty.into(),
        vec![a_val, b_val, c_val],
        "dp2a.lo.s32.s32 $0, $1, $2, $3;",
        "=r,r,r,r",
        AsmKind::Pure,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Convert `nvvm.dp2a_u32` to inline PTX.
///
/// `dp2a.lo.u32.u32 %d, %a, %b, %c;` (per-thread arithmetic, non-convergent).
pub(crate) fn convert_dp2a_u32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 3 {
        return pliron::input_err_noloc!("dp2a_u32 requires 3 operands");
    }

    let a_val = operands[0];
    let b_val = operands[1];
    let c_val = operands[2];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        i32_ty.into(),
        vec![a_val, b_val, c_val],
        "dp2a.lo.u32.u32 $0, $1, $2, $3;",
        "=r,r,r,r",
        AsmKind::Pure,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}
