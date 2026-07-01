/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lower ldmatrix operations to convergent inline PTX.
//!
//! All variants return their packed b16 fragments directly in registers.
//! The memory clobber is required because ldmatrix reads shared memory even
//! though that read is expressed inside inline assembly.

use crate::convert::intrinsics::common::inline_asm_convergent;
use llvm_export::{ops as llvm, types as llvm_types};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::TypeHandle;

pub(crate) fn convert_ldmatrix_x1(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix(ctx, rewriter, op, 1, false, "ldmatrix_x1")
}

pub(crate) fn convert_ldmatrix_x1_trans(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix(ctx, rewriter, op, 1, true, "ldmatrix_x1_trans")
}

pub(crate) fn convert_ldmatrix_x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix(ctx, rewriter, op, 2, false, "ldmatrix_x2")
}

pub(crate) fn convert_ldmatrix_x2_trans(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix(ctx, rewriter, op, 2, true, "ldmatrix_x2_trans")
}

pub(crate) fn convert_ldmatrix_x4(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix(ctx, rewriter, op, 4, false, "ldmatrix_x4")
}

pub(crate) fn convert_ldmatrix_x4_trans(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix(ctx, rewriter, op, 4, true, "ldmatrix_x4_trans")
}

fn convert_ldmatrix(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    register_count: usize,
    transposed: bool,
    name: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!("{} requires one shared-memory pointer", name);
    }

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let result_ty: TypeHandle = if register_count == 1 {
        i32_ty.into()
    } else {
        let fields: Vec<TypeHandle> = (0..register_count).map(|_| i32_ty.into()).collect();
        llvm_types::StructType::get_unnamed(ctx, fields).into()
    };

    let outputs = (0..register_count)
        .map(|index| format!("${index}"))
        .collect::<Vec<_>>()
        .join(", ");
    let pointer_operand = register_count;
    let trans = if transposed { ".trans" } else { "" };
    let template = format!(
        "{{ .reg .u64 %ptr64; .reg .u32 %ptr32; \
         cvta.to.shared.u64 %ptr64, ${pointer_operand}; \
         cvt.u32.u64 %ptr32, %ptr64; \
         ldmatrix.sync.aligned.m8n8.x{register_count}{trans}.shared.b16 \
         {{{outputs}}}, [%ptr32]; }}"
    );
    let constraints = (0..register_count)
        .map(|_| "=r")
        .chain(["l", "~{memory}"])
        .collect::<Vec<_>>()
        .join(",");

    let inline_asm =
        inline_asm_convergent(ctx, rewriter, result_ty, operands, &template, &constraints);

    if register_count == 1 {
        rewriter.replace_operation(ctx, op, inline_asm);
        return Ok(());
    }

    let aggregate = inline_asm.deref(ctx).get_result(0);
    let mut registers = Vec::with_capacity(register_count);
    for index in 0..register_count {
        let extract = llvm::ExtractValueOp::new(ctx, aggregate, vec![index as u32])
            .map_err(|error| pliron::input_error_noloc!("{}", error))?;
        rewriter.insert_operation(ctx, extract.get_operation());
        registers.push(extract.get_operation().deref(ctx).get_result(0));
    }
    rewriter.replace_operation_with_values(ctx, op, registers);
    Ok(())
}
