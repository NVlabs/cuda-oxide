/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Basic NVVM intrinsic conversion: thread IDs, block IDs, barrier.
//!
//! | Operation    | LLVM Intrinsic                    |
//! |--------------|-----------------------------------|
//! | `ReadTidX`   | `llvm_nvvm_read_ptx_sreg_tid_x`   |
//! | `ReadCtaidX` | `llvm_nvvm_read_ptx_sreg_ctaid_x` |
//! | `ReadNtidX`  | `llvm_nvvm_read_ptx_sreg_ntid_x`  |
//! | `Barrier0`   | `llvm_nvvm_barrier0`              |
//! | `ThreadfenceBlock` | inline PTX `membar.cta`      |
//! | `Threadfence` | inline PTX `membar.gl`           |
//! | `ThreadfenceSystem` | inline PTX `membar.sys`     |

use crate::convert::intrinsics::common::*;
use llvm_export::types as llvm_types;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::rewriter::Rewriter;
use pliron::operation::Operation;
use pliron::result::Result;

pub(crate) fn convert_sreg_read_i32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let func_ty = llvm_types::FuncType::get(ctx, i32_ty.into(), vec![], false);
    let call_op = call_intrinsic(ctx, rewriter, op, intrinsic_name, func_ty, vec![])?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// Convert `mir.barrier0` to `llvm.nvvm.barrier0` intrinsic call.
pub(crate) fn convert_barrier0(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let func_ty = llvm_types::FuncType::get(ctx, void_ty.into(), vec![], false);
    call_intrinsic(ctx, rewriter, op, "llvm_nvvm_barrier0", func_ty, vec![])?;
    rewriter.erase_operation(ctx, op);
    Ok(())
}

fn convert_membar(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    asm_template: &str,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![],
        asm_template,
        "~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert a block-scoped memory fence to inline PTX `membar.cta`.
pub(crate) fn convert_threadfence_block(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_membar(ctx, rewriter, op, "membar.cta;")
}

/// Convert a device-scoped memory fence to inline PTX `membar.gl`.
pub(crate) fn convert_threadfence(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_membar(ctx, rewriter, op, "membar.gl;")
}

/// Convert a system-scoped memory fence to inline PTX `membar.sys`.
pub(crate) fn convert_threadfence_system(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_membar(ctx, rewriter, op, "membar.sys;")
}
