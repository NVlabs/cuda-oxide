/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-cooperative `ldmatrix.sync.aligned.m8n8.x4` lowering (SM75+).
//!
//! Lowers `nvvm.ldmatrix_x4_b16` / `nvvm.ldmatrix_x4_trans_b16` to a single
//! inline-PTX block that converts the generic shared pointer with
//! `cvta.to.shared.u32` and then emits the ldmatrix instruction.

use dialect_llvm::ops as llvm;
use dialect_llvm::types as llvm_types;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::TypeObj;

fn convert_inner(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    asm_template: &'static str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!(
            "ldmatrix_x4 requires 1 operand (smem_ptr), got {}",
            operands.len()
        );
    }

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let field_types: Vec<Ptr<TypeObj>> = (0..4).map(|_| i32_ty.into()).collect();
    let struct_ty = llvm_types::StructType::get_unnamed(ctx, field_types);

    let inline_asm = llvm::InlineAsmOp::new_convergent(
        ctx,
        struct_ty.into(),
        operands,
        asm_template,
        "=r,=r,=r,=r,l",
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

/// Convert nvvm.ldmatrix_x4_b16 to inline PTX.
pub(crate) fn convert_ldmatrix_x4_b16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_inner(
        ctx,
        rewriter,
        op,
        concat!(
            "{ ",
            ".reg .u64 %ptr64; ",
            ".reg .u32 %saddr; ",
            "cvta.to.shared.u64 %ptr64, $4; ",
            "cvt.u32.u64 %saddr, %ptr64; ",
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [%saddr]; ",
            "}"
        ),
    )
}

/// Convert nvvm.ldmatrix_x4_trans_b16 to inline PTX.
pub(crate) fn convert_ldmatrix_x4_trans_b16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_inner(
        ctx,
        rewriter,
        op,
        concat!(
            "{ ",
            ".reg .u64 %ptr64; ",
            ".reg .u32 %saddr; ",
            "cvta.to.shared.u64 %ptr64, $4; ",
            "cvt.u32.u64 %saddr, %ptr64; ",
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [%saddr]; ",
            "}"
        ),
    )
}
