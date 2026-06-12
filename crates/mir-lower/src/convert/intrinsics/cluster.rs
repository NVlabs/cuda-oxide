/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thread Block Cluster intrinsic conversion (SM 90+ Hopper).
//!
//! This module converts cluster operations to inline PTX assembly.
//! All cluster operations require sm_90 or later.

use crate::convert::intrinsics::common::*;
use llvm_export::op_interfaces::CastOpInterface;
use llvm_export::ops as llvm_ops;
use llvm_export::types as llvm_types;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Convert a cluster special register read to inline PTX.
pub(crate) fn convert_cluster_sreg(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    sreg_name: &str,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let asm_template = format!("mov.u32 $0, {};", sreg_name);

    let asm_op = inline_asm_convergent(ctx, rewriter, i32_ty.into(), vec![], &asm_template, "=r");
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Convert cluster_sync to inline PTX.
pub(crate) fn convert_cluster_sync(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![],
        "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;",
        "~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert `mapa.shared::cluster` to inline PTX.
pub(crate) fn convert_mapa_shared_cluster(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("mapa_shared_cluster requires 2 operands (ptr, rank)");
    }

    let llvm_ptr = operands[0];
    let llvm_rank = operands[1];

    let shared_ptr = cast_to_shared_addrspace(ctx, rewriter, llvm_ptr);
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);

    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        i64_ty.into(),
        vec![shared_ptr, llvm_rank],
        "mapa.shared::cluster.u64 $0, $1, $2;",
        "=l,l,r",
    );

    let i64_result = asm_op.deref(ctx).get_result(0);
    let llvm_ptr_ty = llvm_types::PointerType::get(ctx, 3);
    let inttoptr_op = llvm_ops::IntToPtrOp::new(ctx, i64_result, llvm_ptr_ty.into());
    rewriter.insert_operation(ctx, inttoptr_op.get_operation());
    rewriter.replace_operation(ctx, op, inttoptr_op.get_operation());

    Ok(())
}

/// Convert `dsmem_read_u32` to combined mapa + ld.shared::cluster inline PTX.
pub(crate) fn convert_dsmem_read_u32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("dsmem_read_u32 requires 2 operands (ptr, rank)");
    }

    let llvm_ptr = operands[0];
    let llvm_rank = operands[1];

    let shared_ptr = cast_to_shared_addrspace(ctx, rewriter, llvm_ptr);
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        i32_ty.into(),
        vec![shared_ptr, llvm_rank],
        "{ .reg .u64 %mapped; mapa.shared::cluster.u64 %mapped, $1, $2; ld.shared::cluster.u32 $0, [%mapped]; }",
        "=r,l,r,~{memory}",
    );
    rewriter.replace_operation(ctx, op, asm_op);

    Ok(())
}
