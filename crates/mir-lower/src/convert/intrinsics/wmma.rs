/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! WMMA (Warp Matrix Multiply-Accumulate) binary mma.sync intrinsic lowering for SM 80+.

use crate::convert::intrinsics::common::*;
use llvm_export::types as llvm_types;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::rewriter::Rewriter;
use pliron::operation::Operation;
use pliron::result::Result;

/// Convert `mma_m16n8k128_s32_b1` to inline PTX assembly.
///
/// The inline asm block:
/// 1. Loads 4 s32 accumulators from `acc_ptr`
/// 2. Loads 2 u32 A-fragment values from `a_ptr` (packed b1)
/// 3. Loads 1 u32 B-fragment value from `b_ptr` (packed b1)
/// 4. Executes `mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc`
/// 5. Stores 4 s32 results back to `acc_ptr`
pub(crate) fn convert_mma_m16n8k128_s32_b1(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 3 {
        return pliron::input_err_noloc!(
            "mma_m16n8k128_s32_b1 requires 3 operands (acc_ptr, a_ptr, b_ptr)"
        );
    }

    // $0 = acc_ptr, $1 = a_ptr, $2 = b_ptr
    let asm = "\
        .reg .b32 c<4>; \
        .reg .b32 d<4>; \
        .reg .b32 a<2>; \
        .reg .b32 b0; \
        ld.b32 c0, [$0]; \
        ld.b32 c1, [$0+4]; \
        ld.b32 c2, [$0+8]; \
        ld.b32 c3, [$0+12]; \
        ld.b32 a0, [$1]; \
        ld.b32 a1, [$1+4]; \
        ld.b32 b0, [$2]; \
        mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc \
            {d0, d1, d2, d3}, \
            {a0, a1}, \
            {b0}, \
            {c0, c1, c2, c3}; \
        st.b32 [$0], d0; \
        st.b32 [$0+4], d1; \
        st.b32 [$0+8], d2; \
        st.b32 [$0+12], d3;";

    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        operands,
        asm,
        "l,l,l,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert `mma_m16n8k256_s32_b1` to inline PTX assembly.
///
/// The inline asm block:
/// 1. Loads 4 s32 accumulators from `acc_ptr`
/// 2. Loads 4 u32 A-fragment values from `a_ptr` (packed b1)
/// 3. Loads 2 u32 B-fragment values from `b_ptr` (packed b1)
/// 4. Executes `mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc`
/// 5. Stores 4 s32 results back to `acc_ptr`
pub(crate) fn convert_mma_m16n8k256_s32_b1(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 3 {
        return pliron::input_err_noloc!(
            "mma_m16n8k256_s32_b1 requires 3 operands (acc_ptr, a_ptr, b_ptr)"
        );
    }

    // $0 = acc_ptr, $1 = a_ptr, $2 = b_ptr
    let asm = "\
        .reg .b32 c<4>; \
        .reg .b32 d<4>; \
        .reg .b32 a<4>; \
        .reg .b32 b<2>; \
        ld.b32 c0, [$0]; \
        ld.b32 c1, [$0+4]; \
        ld.b32 c2, [$0+8]; \
        ld.b32 c3, [$0+12]; \
        ld.b32 a0, [$1]; \
        ld.b32 a1, [$1+4]; \
        ld.b32 a2, [$1+8]; \
        ld.b32 a3, [$1+12]; \
        ld.b32 b0, [$2]; \
        ld.b32 b1, [$2+4]; \
        mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc \
            {d0, d1, d2, d3}, \
            {a0, a1, a2, a3}, \
            {b0, b1}, \
            {c0, c1, c2, c3}; \
        st.b32 [$0], d0; \
        st.b32 [$0+4], d1; \
        st.b32 [$0+8], d2; \
        st.b32 [$0+12], d3;";

    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        operands,
        asm,
        "l,l,l,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}
