/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix intrinsic lowering (`movmatrix`).

use llvm_export::ops::{self as llvm, AsmKind, InlineAsmOpExt};
use pliron::builtin::types::{IntegerType, Signedness};
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

/// Convert `mma_m8n8k4_f64` to inline PTX assembly.
///
/// The inline asm block:
/// 1. Loads 2 f64 accumulators from `acc_ptr`
/// 2. Loads 1 f64 A-fragment value from `a_ptr`
/// 3. Loads 1 f64 B-fragment value from `b_ptr`
/// 4. Executes `mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64`
/// 5. Stores 2 f64 results back to `acc_ptr`
pub(crate) fn convert_mma_m8n8k4_f64(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 3 {
        return pliron::input_err_noloc!(
            "mma_m8n8k4_f64 requires 3 operands (acc_ptr, a_ptr, b_ptr)"
        );
    }

    // $0 = acc_ptr, $1 = a_ptr, $2 = b_ptr
    let asm = "\
        .reg .f64 c<2>; \
        .reg .f64 d<2>; \
        .reg .f64 a0; \
        .reg .f64 b0; \
        ld.f64 c0, [$0]; \
        ld.f64 c1, [$0+8]; \
        ld.f64 a0, [$1]; \
        ld.f64 b0, [$2]; \
        mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 \
            {d0, d1}, \
            {a0}, \
            {b0}, \
            {c0, c1}; \
        st.f64 [$0], d0; \
        st.f64 [$0+8], d1;";

    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        operands,
        asm,
        "l,l,l,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}