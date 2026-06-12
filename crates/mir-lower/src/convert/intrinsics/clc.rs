/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cluster Launch Control (CLC) intrinsic conversion for Blackwell+ GPUs.
//!
//! # LLVM Backend Limitations
//!
//! Both LLVM 21 and LLVM 22 define CLC intrinsics in `IntrinsicsNVVM.td` but
//! the NVPTX backend does NOT lower them to PTX. The backend emits unresolved
//! `call.uni` stubs that `ptxas` rejects. Therefore, we use inline PTX assembly
//! for all CLC operations, matching the pattern used by NVIDIA's own CCCL headers.
//!
//! # Operations
//!
//! | Operation              | PTX Instruction                               | Inline PTX |
//! |------------------------|-----------------------------------------------|------------|
//! | `TryCancel`            | `clusterlaunchcontrol.try_cancel.async...`    | convergent |
//! | `TryCancelMulticast`   | `...multicast::cluster::all...`               | convergent |
//! | `QueryIsCanceled`      | `...query_cancel.is_canceled.pred.b128`       | convergent |
//! | `QueryGetFirstCtaidX`  | `...query_cancel.get_first_ctaid::x.b32.b128` | convergent |
//! | `QueryGetFirstCtaidY`  | `...query_cancel.get_first_ctaid::y.b32.b128` | convergent |
//! | `QueryGetFirstCtaidZ`  | `...query_cancel.get_first_ctaid::z.b32.b128` | convergent |

use crate::convert::intrinsics::common::*;
use llvm_export::ops as llvm;
use llvm_export::ops::InlineAsmOpExt;
use llvm_export::types as llvm_types;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

// ============================================================================
// try_cancel operations
// ============================================================================

/// Convert nvvm.clc_try_cancel / nvvm.clc_try_cancel_multicast to inline PTX.
///
/// Both variants take two pointer operands (response, mbar) in shared memory.
/// The pointers arrive as generic 64-bit addresses and must be converted to
/// 32-bit shared addresses for the PTX instruction.
///
/// PTX (unicast):
///   clusterlaunchcontrol.try_cancel.async.shared::cta
///     .mbarrier::complete_tx::bytes.b128 [resp], [mbar];
///
/// PTX (multicast):
///   clusterlaunchcontrol.try_cancel.async.shared::cta
///     .mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [resp], [mbar];
pub(crate) fn convert_try_cancel(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    multicast: bool,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 2 {
        return pliron::input_err_noloc!("clc_try_cancel requires 2 operands (response, mbar)");
    }
    let response = operands[0];
    let mbar = operands[1];

    let void_ty = llvm_types::VoidType::get(ctx);

    let ptx = if multicast {
        concat!(
            "{ ",
            ".reg .u64 %resp_shared64; .reg .u32 %resp_shared32; ",
            "cvta.to.shared.u64 %resp_shared64, $0; cvt.u32.u64 %resp_shared32, %resp_shared64; ",
            ".reg .u64 %mbar_shared64; .reg .u32 %mbar_shared32; ",
            "cvta.to.shared.u64 %mbar_shared64, $1; cvt.u32.u64 %mbar_shared32, %mbar_shared64; ",
            "clusterlaunchcontrol.try_cancel.async.shared::cta",
            ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128 ",
            "[%resp_shared32], [%mbar_shared32]; ",
            "}"
        )
    } else {
        concat!(
            "{ ",
            ".reg .u64 %resp_shared64; .reg .u32 %resp_shared32; ",
            "cvta.to.shared.u64 %resp_shared64, $0; cvt.u32.u64 %resp_shared32, %resp_shared64; ",
            ".reg .u64 %mbar_shared64; .reg .u32 %mbar_shared32; ",
            "cvta.to.shared.u64 %mbar_shared64, $1; cvt.u32.u64 %mbar_shared32, %mbar_shared64; ",
            "clusterlaunchcontrol.try_cancel.async.shared::cta",
            ".mbarrier::complete_tx::bytes.b128 ",
            "[%resp_shared32], [%mbar_shared32]; ",
            "}"
        )
    };

    let inline_asm = llvm::InlineAsmOp::new_convergent(
        ctx,
        void_ty.into(),
        vec![response, mbar],
        ptx,
        "l,l,~{memory}",
    );
    rewriter.insert_operation(ctx, inline_asm.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}

// ============================================================================
// query_cancel operations
// ============================================================================

/// Convert nvvm.clc_query_is_canceled to inline PTX.
///
/// Takes two u64 inputs (resp_lo, resp_hi), packs them into a .b128,
/// and extracts a predicate indicating whether the request was canceled.
///
/// PTX:
///   .reg .b128 %resp; mov.b128 %resp, {lo, hi};
///   .reg .pred %p;
///   clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 %p, %resp;
///   selp.b32 result, 1, 0, %p;
pub(crate) fn convert_query_is_canceled(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 2 {
        return pliron::input_err_noloc!(
            "clc_query_is_canceled requires 2 operands (resp_lo, resp_hi)"
        );
    }
    let resp_lo = operands[0];
    let resp_hi = operands[1];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        i32_ty.into(),
        vec![resp_lo, resp_hi],
        concat!(
            "{ ",
            ".reg .b128 %resp; mov.b128 %resp, {$1, $2}; ",
            ".reg .pred %p; ",
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 %p, %resp; ",
            "selp.b32 $0, 1, 0, %p; ",
            "}"
        ),
        "=r,l,l",
    );
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Convert nvvm.clc_query_get_first_ctaid_{x,y,z} to inline PTX.
///
/// Takes two u64 inputs (resp_lo, resp_hi), packs them into a .b128,
/// and extracts the CTA coordinate for the specified dimension.
///
/// PTX:
///   .reg .b128 %resp; mov.b128 %resp, {lo, hi};
///   clusterlaunchcontrol.query_cancel.get_first_ctaid::{dim}.b32.b128 result, %resp;
pub(crate) fn convert_query_get_first_ctaid(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    dim: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 2 {
        return pliron::input_err_noloc!(
            "clc_query_get_first_ctaid_{} requires 2 operands (resp_lo, resp_hi)",
            dim
        );
    }
    let resp_lo = operands[0];
    let resp_hi = operands[1];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let asm_template = format!(
        "{{ .reg .b128 %resp; mov.b128 %resp, {{$1, $2}}; \
         clusterlaunchcontrol.query_cancel.get_first_ctaid::{}.b32.b128 $0, %resp; }}",
        dim
    );

    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        i32_ty.into(),
        vec![resp_lo, resp_hi],
        &asm_template,
        "=r,l,l",
    );
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}
