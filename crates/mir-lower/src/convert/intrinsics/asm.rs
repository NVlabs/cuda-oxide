/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! User-authored inline PTX lowering.

use crate::convert::types::convert_type;
use dialect_nvvm::ops::InlinePtxOp;
use llvm_export::ops as llvm;
use llvm_export::types as llvm_types;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::Typed;

pub(crate) fn convert_inline_ptx(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let loc = op.deref(ctx).loc();
    let inline_ptx = InlinePtxOp::new(op);
    let template = inline_ptx
        .get_attr_ptx_template(ctx)
        .map(|attr| String::from((*attr).clone()))
        .ok_or_else(|| pliron::input_error!(loc.clone(), "nvvm.inline_ptx missing ptx_template"))?;
    let constraints = inline_ptx
        .get_attr_ptx_constraints(ctx)
        .map(|attr| String::from((*attr).clone()))
        .ok_or_else(|| {
            pliron::input_error!(loc.clone(), "nvvm.inline_ptx missing ptx_constraints")
        })?;
    let sideeffect = inline_ptx
        .get_attr_ptx_sideeffect(ctx)
        .map(|attr| bool::from((*attr).clone()))
        .ok_or_else(|| {
            pliron::input_error!(loc.clone(), "nvvm.inline_ptx missing ptx_sideeffect")
        })?;
    let convergent = inline_ptx
        .get_attr_ptx_convergent(ctx)
        .map(|attr| bool::from((*attr).clone()))
        .ok_or_else(|| {
            pliron::input_error!(loc.clone(), "nvvm.inline_ptx missing ptx_convergent")
        })?;

    let num_results = op.deref(ctx).get_num_results();
    let result_ty = match num_results {
        0 => llvm_types::VoidType::get(ctx).into(),
        1 => {
            let mir_ty = {
                let op_ref = op.deref(ctx);
                op_ref.get_result(0).get_type(ctx)
            };
            convert_type(ctx, mir_ty).map_err(|err| pliron::input_error!(loc.clone(), "{err}"))?
        }
        n => {
            return pliron::input_err!(loc, "nvvm.inline_ptx supports at most one result, got {n}");
        }
    };

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let inline_asm = llvm::InlineAsmOp::new(
        ctx,
        result_ty,
        operands,
        &template,
        &constraints,
        convergent,
    );
    let asm_op = inline_asm.get_operation();
    llvm::set_inline_asm_sideeffect(ctx, asm_op, sideeffect);
    rewriter.insert_operation(ctx, asm_op);

    if num_results == 1 {
        rewriter.replace_operation(ctx, op, asm_op);
    } else {
        rewriter.erase_operation(ctx, op);
    }
    Ok(())
}
