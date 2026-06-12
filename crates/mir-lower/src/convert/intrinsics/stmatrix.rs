/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stmatrix intrinsic conversion for matrix store operations.
//!
//! # Operations
//!
//! | Operation     | PTX                                             | Description         |
//! |---------------|-------------------------------------------------|---------------------|
//! | `M8n8X4`      | `stmatrix.sync.aligned.m8n8.x4.shared.b16`      | Store 4 8x8 matrices|
//! | `M8n8X4Trans` | `stmatrix.sync.aligned.m8n8.x4.trans.shared.b16`| Store 4 transposed  |
//! | `M8n8X2`      | `stmatrix.sync.aligned.m8n8.x2.shared.b16`      | Store 2 8x8 matrices|
//! | `M8n8X2Trans` | `stmatrix.sync.aligned.m8n8.x2.trans.shared.b16`| Store 2 transposed  |

use crate::convert::intrinsics::common::*;
use llvm_export::types as llvm_types;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::rewriter::Rewriter;
use pliron::operation::Operation;
use pliron::result::Result;

pub(crate) fn convert_m8n8_x4(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 5 {
        return pliron::input_err_noloc!("stmatrix.m8n8.x4 requires 5 operands");
    }
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        operands,
        concat!(
            "{ ",
            ".reg .u64 %ptr64; ",
            ".reg .u32 %ptr32; ",
            "cvta.to.shared.u64 %ptr64, $0; ",
            "cvt.u32.u64 %ptr32, %ptr64; ",
            "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%ptr32], {$1, $2, $3, $4}; ",
            "}"
        ),
        "l,r,r,r,r,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

pub(crate) fn convert_m8n8_x4_trans(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 5 {
        return pliron::input_err_noloc!("stmatrix.m8n8.x4.trans requires 5 operands");
    }
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        operands,
        concat!(
            "{ ",
            ".reg .u64 %ptr64; ",
            ".reg .u32 %ptr32; ",
            "cvta.to.shared.u64 %ptr64, $0; ",
            "cvt.u32.u64 %ptr32, %ptr64; ",
            "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%ptr32], {$1, $2, $3, $4}; ",
            "}"
        ),
        "l,r,r,r,r,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

pub(crate) fn convert_m8n8_x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 3 {
        return pliron::input_err_noloc!("stmatrix.m8n8.x2 requires 3 operands");
    }
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        operands,
        concat!(
            "{ ",
            ".reg .u64 %ptr64; ",
            ".reg .u32 %ptr32; ",
            "cvta.to.shared.u64 %ptr64, $0; ",
            "cvt.u32.u64 %ptr32, %ptr64; ",
            "stmatrix.sync.aligned.m8n8.x2.shared.b16 [%ptr32], {$1, $2}; ",
            "}"
        ),
        "l,r,r,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

pub(crate) fn convert_m8n8_x2_trans(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 3 {
        return pliron::input_err_noloc!("stmatrix.m8n8.x2.trans requires 3 operands");
    }
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        operands,
        concat!(
            "{ ",
            ".reg .u64 %ptr64; ",
            ".reg .u32 %ptr32; ",
            "cvta.to.shared.u64 %ptr64, $0; ",
            "cvt.u32.u64 %ptr32, %ptr64; ",
            "stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%ptr32], {$1, $2}; ",
            "}"
        ),
        "l,r,r,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}
