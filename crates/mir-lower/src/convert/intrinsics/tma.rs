/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! TMA (Tensor Memory Accelerator) intrinsic conversion for Hopper+ GPUs.
//!
//! # Operations
//!
//! | Category           | Operations                 | Target          |
//! |--------------------|----------------------------|-----------------|
//! | G2S                | `G2sTile1d..5d`            | LLVM intrinsics |
//! | G2S + multicast    | `G2sTile2dMulticast`       | LLVM intrinsics |
//! | G2S + multicast cg2| `G2sTile2dMulticastCg2`    | LLVM intrinsics |
//! | S2G                | `S2gTile1d..5d`            | LLVM intrinsics |
//! | Sync               | `CommitGroup`, `WaitGroup` | Inline PTX      |

use crate::convert::intrinsics::common::*;
use crate::helpers;
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

/// Convert TMA commit_group to inline PTX.
pub(crate) fn convert_commit_group(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![],
        "cp.async.bulk.commit_group;",
        "~{memory}",
    );
    Ok(())
}

/// Convert TMA wait_group to inline PTX.
pub(crate) fn convert_wait_group(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    is_read: bool,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let n = operands
        .first()
        .copied()
        .unwrap_or_else(|| create_i32_const(ctx, rewriter, 0));

    let asm = if is_read {
        "cp.async.bulk.wait_group.read $0;"
    } else {
        "cp.async.bulk.wait_group $0;"
    };
    inline_asm_convergent(ctx, rewriter, void_ty.into(), vec![n], asm, "n,~{memory}");
    rewriter.erase_operation(ctx, op);
    Ok(())
}

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

    let mut arg_types: Vec<Ptr<pliron::r#type::TypeObj>> = vec![
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

/// Convert TMA S2G (shared to global) operations using LLVM intrinsics.
pub(crate) fn convert_s2g(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    dims: usize,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
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

    let mut arg_types: Vec<Ptr<pliron::r#type::TypeObj>> =
        vec![smem_ptr_ty.into(), generic_ptr_ty.into()];
    for _ in 0..dims {
        arg_types.push(i32_ty.into());
    }

    let intrinsic_name = format!("llvm_nvvm_cp_async_bulk_tensor_s2g_tile_{}d", dims);
    let func_ty = llvm_types::FuncType::get(ctx, void_ty.into(), arg_types, false);

    let parent_block = op.deref(ctx).get_parent_block().unwrap();
    helpers::ensure_intrinsic_declared(ctx, parent_block, &intrinsic_name, func_ty)
        .map_err(|e| pliron::input_error_noloc!("{}", e))?;

    let mut call_args = vec![src_casted];
    call_args.extend(operands[1..].iter().copied());

    let sym_name: pliron::identifier::Identifier = intrinsic_name.as_str().try_into().unwrap();
    let callee = CallOpCallable::Direct(sym_name);
    let llvm_call = llvm::CallOp::new(ctx, callee, func_ty, call_args);
    rewriter.insert_operation(ctx, llvm_call.get_operation());
    rewriter.erase_operation(ctx, op);

    Ok(())
}
