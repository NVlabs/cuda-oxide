// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared lowering helpers for generated packed arithmetic and conversions.

use crate::convert::intrinsics::common::call_intrinsic;
use crate::{IntrinsicBackend, context};
use llvm_export::ops::{self as llvm, AsmKind, InlineAsmOpExt};
use llvm_export::types as llvm_types;
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
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

/// Pack two `f32` values, keeping the first argument in the low lane.
pub(crate) fn convert_generated_packed_f32x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    typed_intrinsic_name: Option<&str>,
    ptx_mnemonic: &str,
    result_width: u32,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 || op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err_noloc!(
            "generated packed f32x2 conversion requires two operands and one result"
        );
    }
    let constraint = match result_width {
        16 => "=h,f,f",
        32 => "=r,f,f",
        width => {
            return pliron::input_err_noloc!(
                "generated packed f32x2 conversion requires a 16- or 32-bit result, got {width}"
            );
        }
    };
    let result_ty = IntegerType::get(ctx, result_width, Signedness::Signless);
    match (
        context::lowering_options(ctx).intrinsic_backend,
        typed_intrinsic_name,
    ) {
        (IntrinsicBackend::LlvmNvptx, Some(intrinsic_name)) => {
            let f32_ty = FP32Type::get(ctx);
            let function_ty = llvm_types::FuncType::get(
                ctx,
                result_ty.into(),
                vec![f32_ty.into(), f32_ty.into()],
                false,
            );
            let call = call_intrinsic(
                ctx,
                rewriter,
                op,
                intrinsic_name,
                function_ty,
                vec![operands[1], operands[0]],
            )?;
            rewriter.replace_operation(ctx, op, call);
            Ok(())
        }
        _ => {
            let inline_asm = llvm::InlineAsmOp::build(
                ctx,
                result_ty.into(),
                operands,
                &format!("{ptx_mnemonic} $0, $2, $1;"),
                constraint,
                AsmKind::Pure,
            );
            let asm_op = inline_asm.get_operation();
            rewriter.insert_operation(ctx, asm_op);
            rewriter.replace_operation(ctx, op, asm_op);
            Ok(())
        }
    }
}
