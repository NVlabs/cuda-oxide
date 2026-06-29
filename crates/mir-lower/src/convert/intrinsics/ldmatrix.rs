/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ldmatrix intrinsic conversion for warp-cooperative matrix load operations.
//!
//! # Operations
//!
//! | Operation      | PTX                                              | Description          |
//! |----------------|--------------------------------------------------|----------------------|
//! | `X1`           | `ldmatrix.sync.aligned.m8n8.x1.shared.b16`       | Load 1 8x8 matrix    |
//! | `X1Trans`      | `ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16` | Load 1 transposed    |
//! | `X2`           | `ldmatrix.sync.aligned.m8n8.x2.shared.b16`       | Load 2 8x8 matrices  |
//! | `X2Trans`      | `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16` | Load 2 transposed    |
//! | `X4`           | `ldmatrix.sync.aligned.m8n8.x4.shared.b16`       | Load 4 8x8 matrices  |
//! | `X4Trans`      | `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` | Load 4 transposed    |

use crate::convert::intrinsics::common::*;
use llvm_export::types as llvm_types;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::rewriter::Rewriter;
use pliron::operation::Operation;
use pliron::result::Result;

// =============================================================================
// x1 variants (scalar return: 1 operand, 1 result)
// =============================================================================

pub(crate) fn convert_ldmatrix_x1(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix_x1_impl(ctx, rewriter, op, false)
}

pub(crate) fn convert_ldmatrix_x1_trans(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix_x1_impl(ctx, rewriter, op, true)
}

fn convert_ldmatrix_x1_impl(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    trans: bool,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.is_empty() {
        return pliron::input_err_noloc!("ldmatrix_x1 requires 1 operand");
    }

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let trans_mod = if trans { ".trans" } else { "" };

    let asm_template = format!(
        "{{ .reg .u64 %ptr64; .reg .u32 %ptr32; \
         cvta.to.shared.u64 %ptr64, $1; \
         cvt.u32.u64 %ptr32, %ptr64; \
         ldmatrix.sync.aligned.m8n8.x1{trans_mod}.shared.b16 {{$0}}, [%ptr32]; }}"
    );

    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        i32_ty.into(),
        operands,
        &asm_template,
        "=r,l",
    );
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

// =============================================================================
// x2/x4 variants (void, stores to dest_ptr)
// =============================================================================

pub(crate) fn convert_ldmatrix_x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix_array_impl(ctx, rewriter, op, 2, false, "ldmatrix_x2")
}

pub(crate) fn convert_ldmatrix_x2_trans(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix_array_impl(ctx, rewriter, op, 2, true, "ldmatrix_x2_trans")
}

pub(crate) fn convert_ldmatrix_x4(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix_array_impl(ctx, rewriter, op, 4, false, "ldmatrix_x4")
}

pub(crate) fn convert_ldmatrix_x4_trans(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix_array_impl(ctx, rewriter, op, 4, true, "ldmatrix_x4_trans")
}

fn convert_ldmatrix_array_impl(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    num_regs: usize,
    trans: bool,
    name: &str,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 2 {
        return pliron::input_err_noloc!("{} requires 2 operands (smem_ptr, dest_ptr)", name);
    }
    let smem_ptr = operands[0];
    let dest_ptr = operands[1];

    let reg_list: String = (0..num_regs)
        .map(|i| format!("r{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let trans_suffix = if trans { ".trans" } else { "" };

    let stores: String = (0..num_regs)
        .map(|i| {
            if i == 0 {
                "st.b32 [$0], r0; ".to_string()
            } else {
                format!("st.b32 [$0+{}], r{i}; ", i * 4)
            }
        })
        .collect::<String>();

    let asm = format!(
        "{{ .reg .b32 r<{num_regs}>; \
         .reg .u64 smem64; \
         .reg .u32 smem32; \
         cvta.to.shared.u64 smem64, $1; \
         cvt.u32.u64 smem32, smem64; \
         ldmatrix.sync.aligned.m8n8.x{num_regs}{trans_suffix}.shared.b16 {{{reg_list}}}, [smem32]; \
         {stores}}}"
    );

    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![dest_ptr, smem_ptr],
        &asm,
        "l,l,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}
