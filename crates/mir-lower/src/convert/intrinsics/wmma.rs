/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix intrinsic lowering (`movmatrix`, `mma.sync`).

use crate::convert::intrinsics::common::*;
use llvm_export::ops::{self as llvm, AsmKind, InlineAsmOpExt};
use llvm_export::types as llvm_types;
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
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

/// Convert `mma_m16n8k16_f32_f16` to one register-only inline PTX operation.
///
/// Operand order is C[0..4], A[0..4], B[0..2]. The four D registers are
/// returned as an LLVM struct and then split back into the dialect op's four
/// SSA results. There are no hidden pointer, stack, load, or store operands.
pub(crate) fn convert_mma_m16n8k16_f32_f16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 10 {
        return pliron::input_err_noloc!(
            "mma_m16n8k16_f32_f16 requires 10 register operands, got {}",
            operands.len()
        );
    }

    let f32_ty = FP32Type::get(ctx);
    let result_ty = llvm_types::StructType::get_unnamed(ctx, vec![f32_ty.into(); 4]);
    let template = concat!(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ",
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

/// Convert `mma_m16n8k8_f32_tf32` to one register-only inline PTX operation.
///
/// Operand order is C[0..4], A[0..4], B[0..2]. The four D registers are
/// returned as an LLVM struct and then split back into the dialect op's four
/// SSA results. There are no hidden pointer, stack, load, or store operands.
pub(crate) fn convert_mma_m16n8k8_f32_tf32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 10 {
        return pliron::input_err_noloc!(
            "mma_m16n8k8_f32_tf32 requires 10 register operands, got {}",
            operands.len()
        );
    }

    let f32_ty = FP32Type::get(ctx);
    let result_ty = llvm_types::StructType::get_unnamed(ctx, vec![f32_ty.into(); 4]);
    let template = concat!(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 ",
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

/// Convert `mma_m16n8k32_s32_s8` to one register-only inline PTX operation.
///
/// Operand order is C[0..4], A[0..4], B[0..2]. The four D registers are
/// returned as an LLVM struct and then split back into the dialect op's four
/// SSA results. All operands and results use the `r` (integer) constraint.
pub(crate) fn convert_mma_m16n8k32_s32_s8(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 10 {
        return pliron::input_err_noloc!(
            "mma_m16n8k32_s32_s8 requires 10 register operands, got {}",
            operands.len()
        );
    }

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let result_ty = llvm_types::StructType::get_unnamed(ctx, vec![i32_ty.into(); 4]);
    let template = concat!(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ",
        "{$0, $1, $2, $3}, ",
        "{$8, $9, $10, $11}, ",
        "{$12, $13}, ",
        "{$4, $5, $6, $7};"
    );
    let constraints = "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r";
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

/// Convert `mma_m8n8k4_f64` to inline PTX assembly.
///
/// The operation consumes the two C registers plus A and B directly, and
/// returns both D fragment registers. No pointer or memory operand is involved.
pub(crate) fn convert_mma_m8n8k4_f64(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 4 {
        return pliron::input_err_noloc!(
            "mma_m8n8k4_f64 requires 4 f64 operands (c0, c1, a, b), got {}",
            operands.len()
        );
    }

    let f64_ty = FP64Type::get(ctx);
    let result_ty = llvm_types::StructType::get_unnamed(ctx, vec![f64_ty.into(), f64_ty.into()]);
    let inline_asm = inline_asm_convergent(
        ctx,
        rewriter,
        result_ty.into(),
        operands,
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 \
         {$0, $1}, {$4}, {$5}, {$2, $3};",
        "=d,=d,d,d,d,d",
    );

    let aggregate = inline_asm.deref(ctx).get_result(0);
    let mut results = Vec::with_capacity(2);
    for index in 0..2 {
        let extract = llvm::ExtractValueOp::new(ctx, aggregate, vec![index])?;
        rewriter.insert_operation(ctx, extract.get_operation());
        results.push(extract.get_operation().deref(ctx).get_result(0));
    }
    rewriter.replace_operation_with_values(ctx, op, results);
    Ok(())
}

/// Convert `mma_m16n8k32_f8f6f4` to inline PTX assembly.
///
/// Reads `mma_atype` and `mma_btype` string attributes from the op to
/// generate the correct PTX mnemonic. The element type is part of the PTX
/// instruction type qualifiers (unlike tcgen05 where it's in the idesc).
///
/// PTX: `mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.{atype}.{btype}.f32`
///
/// Operand order is C[0..4], A[0..4], B[0..2].  The four D registers are
/// returned as an LLVM struct and then split back into the dialect op's four
/// SSA results.
pub(crate) fn convert_mma_m16n8k32_f8f6f4(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    use dialect_nvvm::ops::MmaM16N8K32F8F6F4Op;

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 10 {
        return pliron::input_err_noloc!(
            "mma_m16n8k32_f8f6f4 requires 10 register operands, got {}",
            operands.len()
        );
    }

    // Read atype/btype from the op's string attributes.
    let typed = MmaM16N8K32F8F6F4Op::new(op);
    let atype = {
        let attr = typed.get_attr_mma_atype(ctx).ok_or_else(|| {
            pliron::input_error_noloc!("mma_m16n8k32_f8f6f4 requires mma_atype attribute")
        })?;
        String::from((*attr).clone())
    };
    let btype = {
        let attr = typed.get_attr_mma_btype(ctx).ok_or_else(|| {
            pliron::input_error_noloc!("mma_m16n8k32_f8f6f4 requires mma_btype attribute")
        })?;
        String::from((*attr).clone())
    };

    let f32_ty = FP32Type::get(ctx);
    let result_ty = llvm_types::StructType::get_unnamed(ctx, vec![f32_ty.into(); 4]);
    let template = format!(
        "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.{atype}.{btype}.f32 \
         {{$0, $1, $2, $3}}, {{$8, $9, $10, $11}}, {{$12, $13}}, {{$4, $5, $6, $7}};"
    );
    let constraints = "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r";
    let inline_asm = inline_asm_convergent(
        ctx,
        rewriter,
        result_ty.into(),
        operands,
        &template,
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

/// Convert `mma_m16n8k64_mxf4` to inline PTX assembly.
///
/// Reads the scale-factor selector attributes (byte_id_a, thread_id_a,
/// byte_id_b, thread_id_b) and embeds them as literal values in the PTX
/// template. The scale-data registers are passed as inline-asm operands.
///
/// PTX: `mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0`
///
/// Operand order: A[0..4], B[0..2], C[0..4], scale_a, scale_b (12 operands).
/// Results: D[0..4] (4 f32).
pub(crate) fn convert_mma_m16n8k64_mxf4(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    use dialect_nvvm::ops::MmaM16N8K64Mxf4Op;

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 12 {
        return pliron::input_err_noloc!(
            "mma_m16n8k64_mxf4 requires 12 operands, got {}",
            operands.len()
        );
    }

    // Read selector attributes — these are compile-time constants
    // embedded in the PTX template, not register operands.
    let typed = MmaM16N8K64Mxf4Op::new(op);
    let byte_id_a = {
        let attr = typed.get_attr_mma_byte_id_a(ctx).ok_or_else(|| {
            pliron::input_error_noloc!("mma_m16n8k64_mxf4 requires mma_byte_id_a attribute")
        })?;
        String::from((*attr).clone())
    };
    let thread_id_a = {
        let attr = typed.get_attr_mma_thread_id_a(ctx).ok_or_else(|| {
            pliron::input_error_noloc!("mma_m16n8k64_mxf4 requires mma_thread_id_a attribute")
        })?;
        String::from((*attr).clone())
    };
    let byte_id_b = {
        let attr = typed.get_attr_mma_byte_id_b(ctx).ok_or_else(|| {
            pliron::input_error_noloc!("mma_m16n8k64_mxf4 requires mma_byte_id_b attribute")
        })?;
        String::from((*attr).clone())
    };
    let thread_id_b = {
        let attr = typed.get_attr_mma_thread_id_b(ctx).ok_or_else(|| {
            pliron::input_error_noloc!("mma_m16n8k64_mxf4 requires mma_thread_id_b attribute")
        })?;
        String::from((*attr).clone())
    };

    let f32_ty = FP32Type::get(ctx);
    let result_ty = llvm_types::StructType::get_unnamed(ctx, vec![f32_ty.into(); 4]);
    // PTX template: selectors {byte_id_a, thread_id_a} etc. are literal
    // values embedded in the string, not $N placeholders.
    let template = format!(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0 \
         {{$0, $1, $2, $3}}, {{$4, $5, $6, $7}}, {{$8, $9}}, {{$10, $11, $12, $13}}, \
         $14, {{{byte_id_a}, {thread_id_a}}}, $15, {{{byte_id_b}, {thread_id_b}}};"
    );
    // Constraints: 4 D results (f32), 4 A (r), 2 B (r), 4 C (f32), 1 scale_a (r), 1 scale_b (r)
    let constraints = "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,r";
    let inline_asm = inline_asm_convergent(
        ctx,
        rewriter,
        result_ty.into(),
        operands,
        &template,
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
