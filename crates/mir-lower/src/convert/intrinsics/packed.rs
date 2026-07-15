// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared lowering helpers for generated packed arithmetic and conversions.

use llvm_export::ops::{self as llvm, AsmKind, InlineAsmOpExt};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::DialectConversionRewriter;
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Lower one generated packed ALU operation to its reviewed PTX instruction.
pub(crate) fn convert_generated_packed_alu(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    ptx_mnemonic: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let constraints = match operands.len() {
        1 => "=r,r",
        2 => "=r,r,r",
        3 => "=r,r,r,r",
        count => {
            return pliron::input_err_noloc!(
                "generated packed ALU operation requires 1 to 3 operands, got {count}"
            );
        }
    };
    let operand_list = (0..=operands.len())
        .map(|index| format!("${index}"))
        .collect::<Vec<_>>()
        .join(", ");
    let result_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        result_ty.into(),
        operands,
        &format!("{ptx_mnemonic} {operand_list};"),
        constraints,
        AsmKind::Pure,
    );
    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Pack two `f32` values, keeping the first argument in the low half.
pub(crate) fn convert_generated_packed_f32x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    ptx_type: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!(
            "generated packed f32x2 conversion requires 2 operands, got {}",
            operands.len()
        );
    }
    let result_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        result_ty.into(),
        operands,
        &format!("cvt.rn.{ptx_type}.f32 $0, $2, $1;"),
        "=r,f,f",
        AsmKind::Pure,
    );
    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}
