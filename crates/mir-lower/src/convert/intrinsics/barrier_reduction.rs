/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lowering for CTA barrier reductions.

use crate::convert::intrinsics::common::{call_intrinsic, inline_asm_convergent, trunc_to_i1};
use crate::{IntrinsicBackend, context};
use llvm_export::types as llvm_types;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::rewriter::Rewriter;
use pliron::operation::Operation;
use pliron::result::Result;

pub(crate) fn convert_barrier_reduction(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
    ptx_instruction: &str,
    predicate_result: bool,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if !matches!(operands.len(), 2 | 3) || op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err_noloc!(
            "CTA barrier reduction requires two or three operands and one result"
        );
    }
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    if context::lowering_options(ctx).intrinsic_backend == IntrinsicBackend::LlvmNvptx {
        let result_ty = if predicate_result { i1_ty } else { i32_ty };
        let mut argument_types = vec![i32_ty.into()];
        if operands.len() == 3 {
            argument_types.push(i32_ty.into());
        }
        argument_types.push(i1_ty.into());
        let function_ty = llvm_types::FuncType::get(ctx, result_ty.into(), argument_types, false);
        let call = call_intrinsic(ctx, rewriter, op, intrinsic_name, function_ty, operands)?;
        rewriter.replace_operation(ctx, op, call);
        return Ok(());
    }

    let (template, constraints) =
        barrier_reduction_inline_recipe(ptx_instruction, operands.len(), predicate_result);
    if predicate_result {
        let asm = inline_asm_convergent(
            ctx,
            rewriter,
            op,
            i32_ty.into(),
            operands,
            &template,
            constraints,
        );
        let materialized = asm.deref(ctx).get_result(0);
        let result = trunc_to_i1(ctx, rewriter, materialized);
        rewriter.replace_operation_with_values(ctx, op, vec![result]);
    } else {
        let asm = inline_asm_convergent(
            ctx,
            rewriter,
            op,
            i32_ty.into(),
            operands,
            &template,
            constraints,
        );
        rewriter.replace_operation(ctx, op, asm);
    }
    Ok(())
}

fn barrier_reduction_inline_recipe(
    ptx_instruction: &str,
    operand_count: usize,
    predicate_result: bool,
) -> (String, &'static str) {
    debug_assert!(matches!(operand_count, 2 | 3));
    let input_refs = (1..=operand_count)
        .map(|index| format!("${index}"))
        .collect::<Vec<_>>()
        .join(", ");
    let constraints = if operand_count == 3 {
        "=r,r,r,b,~{memory}"
    } else {
        "=r,r,b,~{memory}"
    };
    if predicate_result {
        (
            format!("{{ .reg .pred p; {ptx_instruction} p, {input_refs}; selp.b32 $0, 1, 0, p; }}"),
            constraints,
        )
    } else {
        (format!("{ptx_instruction} $0, {input_refs};"), constraints)
    }
}

#[cfg(test)]
mod tests {
    use super::barrier_reduction_inline_recipe;

    #[test]
    fn predicate_recipe_materializes_predicate_through_selp() {
        let (template, constraints) =
            barrier_reduction_inline_recipe("barrier.red.and.pred", 3, true);
        assert_eq!(
            template,
            "{ .reg .pred p; barrier.red.and.pred p, $1, $2, $3; selp.b32 $0, 1, 0, p; }"
        );
        assert_eq!(constraints, "=r,r,r,b,~{memory}");
    }

    #[test]
    fn popc_recipe_returns_the_direct_u32_result() {
        let (template, constraints) = barrier_reduction_inline_recipe("bar.red.popc.u32", 2, false);
        assert_eq!(template, "bar.red.popc.u32 $0, $1, $2;");
        assert_eq!(constraints, "=r,r,b,~{memory}");
    }
}
