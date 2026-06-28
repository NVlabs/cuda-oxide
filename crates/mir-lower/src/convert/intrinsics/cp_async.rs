/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Asynchronous copy (`cp.async`) intrinsic conversion.
//!
//! | Operation    | PTX                                              |
//! |--------------|--------------------------------------------------|
//! | `CpAsyncCa4` | `cp.async.ca.shared.global [smem], [gmem], 4;`  |
//! | `CpAsyncCa8` | `cp.async.ca.shared.global [smem], [gmem], 8;`  |

use crate::convert::intrinsics::common::*;
use llvm_export::types as llvm_types;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::rewriter::Rewriter;
use pliron::operation::Operation;
use pliron::result::Result;

/// Shared lowering for all `cp.async` variants.
///
/// Emits inline PTX:
/// ```ptx
/// cvta.to.shared.u64 %smem64, $0;
/// cvt.u32.u64 %smem32, %smem64;
/// cvta.to.global.u64 %gmem64, $1;
/// cp.async.{cache_policy}.shared.global [%smem32], [%gmem64], {copy_size};
/// ```
fn convert_cp_async_impl(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    cache_policy: &str,
    copy_size: u32,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 2 {
        return pliron::input_err_noloc!(
            "cp.async.{}.{}B requires 2 operands",
            cache_policy,
            copy_size
        );
    }
    inline_asm_sideeffect(
        ctx,
        rewriter,
        void_ty.into(),
        operands,
        &format!(
            "{{ \
            .reg .u64 %smem64; \
            .reg .u32 %smem32; \
            .reg .u64 %gmem64; \
            cvta.to.shared.u64 %smem64, $0; \
            cvt.u32.u64 %smem32, %smem64; \
            cvta.to.global.u64 %gmem64, $1; \
            cp.async.{cache_policy}.shared.global [%smem32], [%gmem64], {copy_size}; \
            }}"
        ),
        "l,l,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

pub(crate) fn convert_cp_async_ca_4(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_cp_async_impl(ctx, rewriter, op, "ca", 4)
}

pub(crate) fn convert_cp_async_ca_8(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_cp_async_impl(ctx, rewriter, op, "ca", 8)
}
