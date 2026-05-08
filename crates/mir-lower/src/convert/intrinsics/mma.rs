/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level mma.sync intrinsic conversion (SM80+).
//!
//! Lowers `nvvm.mma_m16n8k16_bf16_f32` to inline PTX:
//! `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {d0..d3}, {a0..a3}, {b0,b1}, {c0..c3};`

use dialect_llvm::ops as llvm;
use dialect_llvm::types as llvm_types;
use pliron::builtin::types::FP32Type;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::TypeObj;

/// Convert nvvm.mma_m16n8k16_bf16_f32 to inline PTX.
///
/// Inputs (10): a0..a3 (i32), b0..b1 (i32), c0..c3 (f32).
/// Outputs (4): d0..d3 (f32).
pub(crate) fn convert_mma_m16n8k16_bf16_f32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 10 {
        return pliron::input_err_noloc!(
            "mma_m16n8k16_bf16_f32 requires 10 operands, got {}",
            operands.len()
        );
    }

    let f32_ty = FP32Type::get(ctx);
    let field_types: Vec<Ptr<TypeObj>> = (0..4).map(|_| f32_ty.into()).collect();
    let struct_ty = llvm_types::StructType::get_unnamed(ctx, field_types);

    let inline_asm = llvm::InlineAsmOp::new_convergent(
        ctx,
        struct_ty.into(),
        operands,
        concat!(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ",
            "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};"
        ),
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);

    let struct_result = asm_op.deref(ctx).get_result(0);
    let mut extracted_values = Vec::with_capacity(4);
    for i in 0..4u32 {
        let extract_op = llvm::ExtractValueOp::new(ctx, struct_result, vec![i])
            .map_err(|e| pliron::input_error_noloc!("{}", e))?;
        rewriter.insert_operation(ctx, extract_op.get_operation());
        let field_val = extract_op.get_operation().deref(ctx).get_result(0);
        extracted_values.push(field_val);
    }
    rewriter.replace_operation_with_values(ctx, op, extracted_values);

    Ok(())
}
