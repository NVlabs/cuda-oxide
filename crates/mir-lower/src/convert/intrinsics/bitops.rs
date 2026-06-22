/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bit-manipulation intrinsic conversion.
//!
//! | Operation  | Implementation | Description                      |
//! |------------|----------------|----------------------------------|
//! | `PrmtB32`  | Inline PTX     | Byte permute on two 32-bit words |

use llvm_export::ops::{self as llvm, InlineAsmOpExt};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// prmt.b32: (a, b, c) -> result (inline PTX, pure)
pub(crate) fn convert_prmt_b32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 3 {
        return pliron::input_err_noloc!("prmt_b32 requires 3 operands");
    }

    let a = operands[0];
    let b = operands[1];
    let c = operands[2];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    // Pure inline asm (per-thread data permutation, not a collective op)
    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        i32_ty.into(),
        vec![a, b, c],
        "prmt.b32 $0, $1, $2, $3;",
        "=r,r,r,r",
        llvm_export::ops::AsmKind::Pure,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}
