/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ampere async copy (`cp.async`) intrinsic lowering.
//!
//! # Operations
//!
//! | Operation            | PTX                                                |
//! |----------------------|----------------------------------------------------|
//! | `CpAsyncCg16`        | `cp.async.cg.shared.global [smem], [gmem], 16;`    |
//! | `CpAsyncCa16`        | `cp.async.ca.shared.global [smem], [gmem], 16;`    |

use crate::convert::intrinsics::common::*;
use dialect_llvm::types as llvm_types;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::rewriter::Rewriter;
use pliron::operation::Operation;
use pliron::result::Result;

/// Shared implementation for cp.async 16-byte copy lowering.
///
/// Both `.cg` (L2-only) and `.ca` (L1+L2) variants use identical PTX except
/// for the cache policy qualifier.
///
/// The PTX needs the shared pointer in `.shared` address space (32-bit),
/// so we use `cvta.to.shared.u64` + `cvt.u32.u64` like stmatrix/ldmatrix.
fn convert_cp_async_16_impl(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    cache_policy: &str,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 2 {
        return pliron::input_err_noloc!("cp.async.{}.16 requires 2 operands", cache_policy);
    }
    let asm = format!(
        "{{ \
         .reg .u64 %smem64; \
         .reg .u32 %smem32; \
         cvta.to.shared.u64 %smem64, $0; \
         cvt.u32.u64 %smem32, %smem64; \
         cp.async.{cache_policy}.shared.global [%smem32], [$1], 16; \
         }}"
    );
    inline_asm_convergent(ctx, rewriter, void_ty.into(), operands, &asm, "l,l,~{memory}");
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert `cp.async.cg.shared.global` — 16-byte async copy, L2-only cache policy.
pub(crate) fn convert_cp_async_cg_16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_cp_async_16_impl(ctx, rewriter, op, "cg")
}

/// Convert `cp.async.ca.shared.global` — 16-byte async copy, L1+L2 cache policy.
pub(crate) fn convert_cp_async_ca_16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_cp_async_16_impl(ctx, rewriter, op, "ca")
}
