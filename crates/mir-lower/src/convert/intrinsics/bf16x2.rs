// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Packed bf16x2 ALU intrinsic conversions.
//!
//! `add.bf16x2` / `mul.bf16x2` require `sm_90+`. `fma.rn.bf16x2` is supported
//! from `sm_80`, so the FMA op is what we expose; on Ampere, packed add can
//! be expressed by the caller as `fma(a, ONE_BF16X2, b)`.

use llvm_export::ops as llvm;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Convert `nvvm.fma_bf16x2` to inline PTX.
///
/// `fma.rn.bf16x2 %d, %a, %b, %c;`  — per-thread arithmetic, non-convergent.
pub(crate) fn convert_fma_bf16x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 3 {
        return pliron::input_err_noloc!("fma_bf16x2 requires 3 operands");
    }

    let a_val = operands[0];
    let b_val = operands[1];
    let c_val = operands[2];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let inline_asm = llvm::InlineAsmOp::new(
        ctx,
        i32_ty.into(),
        vec![a_val, b_val, c_val],
        "fma.rn.bf16x2 $0, $1, $2, $3;",
        "=r,r,r,r",
        false,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}
