/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! cp.async zero-fill intrinsic conversion.
//!
//! # Operations
//!
//! | Operation     | PTX                                                    | Description       |
//! |---------------|--------------------------------------------------------|-------------------|
//! | `CaZfill4`    | `cp.async.ca.shared.global [dst], [src], 4, src_size`  | 4-byte zero-fill  |
//! | `CaZfill8`    | `cp.async.ca.shared.global [dst], [src], 8, src_size`  | 8-byte zero-fill  |
//! | `CaZfill16`   | `cp.async.ca.shared.global [dst], [src], 16, src_size` | 16-byte zero-fill |

use crate::convert::intrinsics::common::*;
use llvm_export::types as llvm_types;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::rewriter::Rewriter;
use pliron::operation::Operation;
use pliron::result::Result;

pub(crate) fn convert_ca_zfill_4(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 3 {
        return pliron::input_err_noloc!("cp.async.ca.zfill.4 requires 3 operands");
    }
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        operands,
        concat!(
            "{ ",
            ".reg .u64 %dptr64; ",
            ".reg .u32 %dptr32; ",
            "cvta.to.shared.u64 %dptr64, $0; ",
            "cvt.u32.u64 %dptr32, %dptr64; ",
            "cp.async.ca.shared.global [%dptr32], [$1], 4, $2; ",
            "}"
        ),
        "l,l,r,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

pub(crate) fn convert_ca_zfill_8(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 3 {
        return pliron::input_err_noloc!("cp.async.ca.zfill.8 requires 3 operands");
    }
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        operands,
        concat!(
            "{ ",
            ".reg .u64 %dptr64; ",
            ".reg .u32 %dptr32; ",
            "cvta.to.shared.u64 %dptr64, $0; ",
            "cvt.u32.u64 %dptr32, %dptr64; ",
            "cp.async.ca.shared.global [%dptr32], [$1], 8, $2; ",
            "}"
        ),
        "l,l,r,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

pub(crate) fn convert_ca_zfill_16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 3 {
        return pliron::input_err_noloc!("cp.async.ca.zfill.16 requires 3 operands");
    }
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        operands,
        concat!(
            "{ ",
            ".reg .u64 %dptr64; ",
            ".reg .u32 %dptr32; ",
            "cvta.to.shared.u64 %dptr64, $0; ",
            "cvt.u32.u64 %dptr32, %dptr64; ",
            "cp.async.ca.shared.global [%dptr32], [$1], 16, $2; ",
            "}"
        ),
        "l,l,r,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}
