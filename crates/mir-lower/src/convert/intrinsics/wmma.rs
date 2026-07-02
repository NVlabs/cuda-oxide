/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix intrinsic lowering (`movmatrix`, `mma.sync`).

use crate::convert::intrinsics::common::*;
use llvm_export::ops::{self as llvm, AsmKind, InlineAsmOpExt};
use llvm_export::types as llvm_types;
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Convert `nvvm.movmatrix_trans_b16` to inline PTX.
///
/// `movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;`
///
/// Warp-synchronous, uses convergent inline assembly.
pub(crate) fn convert_movmatrix_trans_b16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!(
            "movmatrix_trans_b16 requires 1 operand, got {}",
            operands.len()
        );
    }

    let a_val = operands[0];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        i32_ty.into(),
        vec![a_val],
        "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
        "=r,r",
        AsmKind::Convergent,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Convert `mma_m16n8k16_f32_bf16` to one register-only inline PTX operation.
///
/// Operand order is C[0..4], A[0..4], B[0..2]. The four D registers are
/// returned as an LLVM struct and then split back into the dialect op's four
/// SSA results. There are no hidden pointer, stack, load, or store operands.
pub(crate) fn convert_mma_m16n8k16_f32_bf16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 10 {
        return pliron::input_err_noloc!(
            "mma_m16n8k16_f32_bf16 requires 10 register operands, got {}",
            operands.len()
        );
    }

    let f32_ty = FP32Type::get(ctx);
    let result_ty = llvm_types::StructType::get_unnamed(ctx, vec![f32_ty.into(); 4]);
    let template = concat!(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ",
        "{$0, $1, $2, $3}, ",
        "{$8, $9, $10, $11}, ",
        "{$12, $13}, ",
        "{$4, $5, $6, $7};"
    );
    let constraints = "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r";
    let inline_asm = inline_asm_convergent(
        ctx,
        rewriter,
        result_ty.into(),
        operands,
        template,
        constraints,
    );

    let aggregate = inline_asm.deref(ctx).get_result(0);
    let mut results = Vec::with_capacity(4);
    for index in 0..4 {
        let extract = llvm::ExtractValueOp::new(ctx, aggregate, vec![index as u32])
            .map_err(|error| pliron::input_error_noloc!("{}", error))?;
        rewriter.insert_operation(ctx, extract.get_operation());
        results.push(extract.get_operation().deref(ctx).get_result(0));
    }
    rewriter.replace_operation_with_values(ctx, op, results);
    Ok(())
}
