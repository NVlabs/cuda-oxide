/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lowering helper for generated extended min/max operations.

use llvm_export::ops::{self as llvm, AsmKind, InlineAsmOpExt};
use pliron::{
    builtin::types::{FP32Type, IntegerType, Signedness},
    context::{Context, Ptr},
    irbuild::{
        dialect_conversion::DialectConversionRewriter, inserter::Inserter, rewriter::Rewriter,
    },
    op::Op,
    operation::Operation,
    result::Result,
};

pub(crate) fn convert_generated_extended_minmax(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    ptx_mnemonic: &str,
    packed: bool,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 || op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err_noloc!(
            "generated extended min/max requires two operands and one result"
        );
    }
    let (result_ty, constraints) = if packed {
        (
            IntegerType::get(ctx, 32, Signedness::Signless).into(),
            "=r,r,r",
        )
    } else {
        (FP32Type::get(ctx).into(), "=f,f,f")
    };
    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        result_ty,
        operands,
        &format!("{ptx_mnemonic} $0, $1, $2;"),
        constraints,
        AsmKind::Pure,
    );
    let inline_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, inline_op);
    rewriter.replace_operation(ctx, op, inline_op);
    Ok(())
}
