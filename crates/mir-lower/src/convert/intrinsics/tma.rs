/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! TMA conversion for Hopper and newer GPUs.

use crate::convert::intrinsics::common::*;
use crate::helpers;
use crate::{IntrinsicBackend, context};
use llvm_export::ops as llvm;
use llvm_export::types as llvm_types;
use pliron::builtin::op_interfaces::CallOpCallable;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Convert TMA G2S (global to shared) operations using LLVM intrinsics.
pub(crate) fn convert_g2s(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    dims: usize,
    multicast: bool,
) -> Result<()> {
    convert_g2s_impl(ctx, rewriter, op, dims, multicast, 0)
}

fn g2s_inline_asm(dims: usize, multicast: bool, cta_group: i32) -> (String, String) {
    let coordinates = (0..dims)
        .map(|index| format!("${}", 3 + index))
        .collect::<Vec<_>>()
        .join(", ");
    let multicast_modifier = if multicast { ".multicast::cluster" } else { "" };
    let cta_group_modifier = if cta_group == 2 { ".cta_group::2" } else { "" };
    let mask = if multicast {
        format!(", ${}", 3 + dims)
    } else {
        String::new()
    };
    let template = format!(
        "cp.async.bulk.tensor.{dims}d.shared::cluster.global.tile.mbarrier::complete_tx::bytes{multicast_modifier}{cta_group_modifier} [$0], [$2, {{{coordinates}}}], [$1]{mask};"
    );
    let mut constraints = vec!["l"; 3];
    constraints.extend(std::iter::repeat_n("r", dims));
    if multicast {
        constraints.push("h");
    }
    constraints.push("~{memory}");
    (template, constraints.join(","))
}

fn convert_g2s_impl(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    dims: usize,
    multicast: bool,
    cta_group: i32,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i16_ty = IntegerType::get(ctx, 16, Signedness::Signless);
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let void_ty = llvm_types::VoidType::get(ctx);
    let shared_cluster_ptr_ty = llvm_types::PointerType::get(ctx, 7);
    let smem_ptr_ty = llvm_types::PointerType::get(ctx, 3);
    let generic_ptr_ty = llvm_types::PointerType::get(ctx, 0);

    let operands: Vec<_> = op.deref(ctx).operands().collect();

    let expected_operands = 3 + dims + 2;
    if operands.len() != expected_operands {
        return pliron::input_err_noloc!(
            "TMA G2S {}D requires {} operands, got {}",
            dims,
            expected_operands,
            operands.len()
        );
    }

    let dst_casted = cast_to_cluster_shared_addrspace(ctx, rewriter, operands[0]);
    let barrier_casted = cast_to_shared_addrspace(ctx, rewriter, operands[1]);

    if context::lowering_options(ctx).intrinsic_backend == IntrinsicBackend::LibNvvm {
        let mut inputs = vec![dst_casted, barrier_casted, operands[2]];
        inputs.extend(operands[3..3 + dims].iter().copied());
        if multicast {
            inputs.push(operands[3 + dims]);
        }

        let (template, constraints) = g2s_inline_asm(dims, multicast, cta_group);

        inline_asm_convergent(
            ctx,
            rewriter,
            void_ty.into(),
            inputs,
            &template,
            &constraints,
        );
        rewriter.erase_operation(ctx, op);
        return Ok(());
    }

    let mut arg_types: Vec<pliron::r#type::TypeHandle> = vec![
        shared_cluster_ptr_ty.into(),
        smem_ptr_ty.into(),
        generic_ptr_ty.into(),
    ];
    for _ in 0..dims {
        arg_types.push(i32_ty.into());
    }
    arg_types.push(i16_ty.into()); // cta_mask
    arg_types.push(i64_ty.into()); // cache_hint
    arg_types.push(i1_ty.into()); // use_cta_mask
    arg_types.push(i1_ty.into()); // use_cache_hint
    arg_types.push(i32_ty.into()); // cta_group

    let intrinsic_name = format!("llvm_nvvm_cp_async_bulk_tensor_g2s_tile_{}d", dims);
    let func_ty = llvm_types::FuncType::get(ctx, void_ty.into(), arg_types, false);

    let parent_block = op.deref(ctx).get_parent_block().unwrap();
    helpers::ensure_intrinsic_declared(ctx, parent_block, &intrinsic_name, func_ty)
        .map_err(|e| pliron::input_error_noloc!("{}", e))?;

    let mut call_args = vec![dst_casted, barrier_casted];
    call_args.extend(operands[2..].iter().copied());

    let use_cta_mask = create_i1_const(ctx, rewriter, multicast);
    let use_cache_hint = create_i1_const(ctx, rewriter, false);
    let cta_group_val = create_i32_const(ctx, rewriter, cta_group);
    call_args.push(use_cta_mask);
    call_args.push(use_cache_hint);
    call_args.push(cta_group_val);

    let sym_name: pliron::identifier::Identifier = intrinsic_name.as_str().try_into().unwrap();
    let callee = CallOpCallable::Direct(sym_name);
    let llvm_call = llvm::CallOp::new(ctx, callee, func_ty, call_args);
    rewriter.insert_operation(ctx, llvm_call.get_operation());
    rewriter.erase_operation(ctx, op);

    Ok(())
}

/// Convert TMA G2S 2D multicast with cta_group::2 via LLVM intrinsic.
pub(crate) fn convert_g2s_multicast_cg2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_g2s_impl(ctx, rewriter, op, 2, true, 2)
}

fn s2g_inline_asm(dims: usize) -> (String, String) {
    let coordinates = (0..dims)
        .map(|index| format!("${}", 2 + index))
        .collect::<Vec<_>>()
        .join(", ");
    let template = format!(
        "cp.async.bulk.tensor.{dims}d.global.shared::cta.tile.bulk_group [$1, {{{coordinates}}}], [$0];"
    );
    let mut constraints = vec!["l"; 2];
    constraints.extend(std::iter::repeat_n("r", dims));
    constraints.push("~{memory}");
    (template, constraints.join(","))
}

/// Convert TMA S2G (shared to global) operations using LLVM intrinsics.
pub(crate) fn convert_s2g(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    dims: usize,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let void_ty = llvm_types::VoidType::get(ctx);
    let smem_ptr_ty = llvm_types::PointerType::get(ctx, 3);
    let generic_ptr_ty = llvm_types::PointerType::get(ctx, 0);

    let operands: Vec<_> = op.deref(ctx).operands().collect();

    let expected_operands = 2 + dims;
    if operands.len() != expected_operands {
        return pliron::input_err_noloc!(
            "TMA S2G {}D requires {} operands, got {}",
            dims,
            expected_operands,
            operands.len()
        );
    }

    let src_casted = cast_to_shared_addrspace(ctx, rewriter, operands[0]);

    if context::lowering_options(ctx).intrinsic_backend == IntrinsicBackend::LibNvvm {
        let mut inputs = vec![src_casted, operands[1]];
        inputs.extend(operands[2..].iter().copied());
        let (template, constraints) = s2g_inline_asm(dims);
        inline_asm_convergent(
            ctx,
            rewriter,
            void_ty.into(),
            inputs,
            &template,
            &constraints,
        );
        rewriter.erase_operation(ctx, op);
        return Ok(());
    }

    let mut arg_types: Vec<pliron::r#type::TypeHandle> =
        vec![smem_ptr_ty.into(), generic_ptr_ty.into()];
    for _ in 0..dims {
        arg_types.push(i32_ty.into());
    }
    arg_types.push(i64_ty.into()); // cache_hint
    arg_types.push(i1_ty.into()); // use_cache_hint

    let intrinsic_name = format!("llvm_nvvm_cp_async_bulk_tensor_s2g_tile_{}d", dims);
    let func_ty = llvm_types::FuncType::get(ctx, void_ty.into(), arg_types, false);

    let parent_block = op.deref(ctx).get_parent_block().unwrap();
    helpers::ensure_intrinsic_declared(ctx, parent_block, &intrinsic_name, func_ty)
        .map_err(|e| pliron::input_error_noloc!("{}", e))?;

    let mut call_args = vec![src_casted];
    call_args.extend(operands[1..].iter().copied());
    call_args.push(create_i64_const(ctx, rewriter, 0));
    call_args.push(create_i1_const(ctx, rewriter, false));

    let sym_name: pliron::identifier::Identifier = intrinsic_name.as_str().try_into().unwrap();
    let callee = CallOpCallable::Direct(sym_name);
    let llvm_call = llvm::CallOp::new(ctx, callee, func_ty, call_args);
    rewriter.insert_operation(ctx, llvm_call.get_operation());
    rewriter.erase_operation(ctx, op);

    Ok(())
}

/// Convert one TMA group-control operation through the selected backend.
pub(crate) fn convert_control(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    operation: &str,
    intrinsic_name: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let expected_operands = match operation {
        "commit_group" => 0,
        "wait_group" | "wait_group_read" => 1,
        _ => return pliron::input_err_noloc!("unsupported TMA control `{operation}`"),
    };
    if operands.len() != expected_operands || op.deref(ctx).get_num_results() != 0 {
        return pliron::input_err_noloc!(
            "TMA {operation} requires {expected_operands} operand(s) and no results"
        );
    }

    let void_ty = llvm_types::VoidType::get(ctx);
    match context::lowering_options(ctx).intrinsic_backend {
        IntrinsicBackend::LlvmNvptx => {
            let argument_types = if operands.is_empty() {
                vec![]
            } else {
                vec![IntegerType::get(ctx, 32, Signedness::Signless).into()]
            };
            let function_ty = llvm_types::FuncType::get(ctx, void_ty.into(), argument_types, false);
            call_intrinsic(ctx, rewriter, op, intrinsic_name, function_ty, operands)?;
        }
        IntrinsicBackend::LibNvvm => {
            let (template, constraints) = match operation {
                "commit_group" => ("cp.async.bulk.commit_group;", "~{memory}"),
                "wait_group" => ("cp.async.bulk.wait_group $0;", "n,~{memory}"),
                "wait_group_read" => ("cp.async.bulk.wait_group.read $0;", "n,~{memory}"),
                _ => unreachable!("operation was validated"),
            };
            inline_asm_sideeffect(
                ctx,
                rewriter,
                void_ty.into(),
                operands,
                template,
                constraints,
            );
        }
    }
    rewriter.erase_operation(ctx, op);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{g2s_inline_asm, s2g_inline_asm};

    #[test]
    fn inline_tma_templates_keep_exact_ptx_shapes() {
        assert_eq!(
            g2s_inline_asm(1, false, 0),
            (
                "cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [$0], [$2, {$3}], [$1];".into(),
                "l,l,l,r,~{memory}".into(),
            )
        );
        assert_eq!(
            g2s_inline_asm(2, true, 2),
            (
                "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2 [$0], [$2, {$3, $4}], [$1], $5;".into(),
                "l,l,l,r,r,h,~{memory}".into(),
            )
        );
        assert_eq!(
            s2g_inline_asm(5),
            (
                "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group [$1, {$2, $3, $4, $5, $6}], [$0];".into(),
                "l,l,r,r,r,r,r,~{memory}".into(),
            )
        );
    }
}
