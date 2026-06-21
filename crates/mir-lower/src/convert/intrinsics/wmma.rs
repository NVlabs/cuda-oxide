/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! WMMA (mma.sync) intrinsic conversion for Ampere+ GPUs.
//!
//! Converts dialect-nvvm WMMA operations into inline PTX assembly.
//!
//! # Operations
//!
//! | Operation            | PTX                                              |
//! |----------------------|--------------------------------------------------|
//! | `LdmatrixX4`         | `ldmatrix.sync.aligned.m8n8.x4.shared.b16`      |
//! | `LdmatrixX2`         | `ldmatrix.sync.aligned.m8n8.x2.shared.b16`      |
//! | `LdmatrixX4Trans`    | `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` |
//! | `LdmatrixX2Trans`    | `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16` |
//! | `MmaM16N8K16F32F16`  | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` |

use crate::convert::intrinsics::common::*;
use llvm_export::types as llvm_types;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Shared implementation for all ldmatrix lowering variants.
///
/// Builds inline PTX for `ldmatrix.sync.aligned.m8n8.xN[.trans].shared.b16`
/// that loads `num_regs` × u32 from shared memory and stores to `dest_ptr`.
///
/// Note: `smem_ptr` is a generic-space pointer. The PTX uses `cvta.to.shared`
/// to convert it (same pattern as stmatrix.rs). Do NOT use
/// `cast_to_shared_addrspace` — that would double-convert.
fn convert_ldmatrix_impl(
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

    // Build register list: {r0} or {r0, r1} or {r0, r1, r2, r3}
    let reg_list: String = (0..num_regs)
        .map(|i| format!("r{i}"))
        .collect::<Vec<_>>()
        .join(", ");
    let trans_suffix = if trans { ".trans" } else { "" };

    // Build store sequence: st.b32 [$0+offset], rN;
    let stores: String = (0..num_regs)
        .map(|i| {
            if i == 0 {
                format!("st.b32 [$0], r0; ")
            } else {
                format!("st.b32 [$0+{}], r{i}; ", i * 4)
            }
        })
        .collect::<String>();

    let asm = format!(
        "{{ \
         .reg .b32 r<{num_regs}>; \
         .reg .u64 smem64; \
         .reg .u32 smem32; \
         cvta.to.shared.u64 smem64, $1; \
         cvt.u32.u64 smem32, smem64; \
         ldmatrix.sync.aligned.m8n8.x{num_regs}{trans_suffix}.shared.b16 {{{reg_list}}}, [smem32]; \
         {stores}\
         }}"
    );

    inline_asm_convergent(ctx, rewriter, void_ty.into(), vec![dest_ptr, smem_ptr], &asm, "l,l");
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert `ldmatrix.sync.aligned.m8n8.x4.shared.b16` — load 4 × u32 from shared.
pub(crate) fn convert_ldmatrix_x4(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix_impl(ctx, rewriter, op, 4, false, "ldmatrix_x4")
}

/// Convert `ldmatrix.sync.aligned.m8n8.x2.shared.b16` — load 2 × u32 from shared.
pub(crate) fn convert_ldmatrix_x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix_impl(ctx, rewriter, op, 2, false, "ldmatrix_x2")
}

/// Convert `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` — load 4 × u32 transposed.
pub(crate) fn convert_ldmatrix_x4_trans(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix_impl(ctx, rewriter, op, 4, true, "ldmatrix_x4_trans")
}

/// Convert `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16` — load 2 × u32 transposed.
pub(crate) fn convert_ldmatrix_x2_trans(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_ldmatrix_impl(ctx, rewriter, op, 2, true, "ldmatrix_x2_trans")
}

/// Convert mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
///
/// Operands: [acc_ptr, a_ptr, b_ptr]
/// - acc_ptr: pointer to [f32; 4] (read-modify-write)
/// - a_ptr:   pointer to [u32; 4] (A fragment)
/// - b_ptr:   pointer to [u32; 2] (B fragment)
///
/// The lowering loads the fragments from pointers into PTX registers,
/// executes the mma.sync instruction, and stores results back.
/// Uses generic ld/st since the pointers are in generic address space.
pub(crate) fn convert_mma_m16n8k16_f32_f16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 3 {
        return pliron::input_err_noloc!(
            "mma_m16n8k16_f32_f16 requires 3 operands (acc_ptr, a_ptr, b_ptr)"
        );
    }
    let acc_ptr = operands[0];
    let a_ptr = operands[1];
    let b_ptr = operands[2];

    let asm = concat!(
        "{ ",
        ".reg .f32 d<4>; ",
        ".reg .b32 a<4>; ",
        ".reg .b32 b<2>; ",
        "ld.f32 d0, [$0]; ",
        "ld.f32 d1, [$0+4]; ",
        "ld.f32 d2, [$0+8]; ",
        "ld.f32 d3, [$0+12]; ",
        "ld.b32 a0, [$1]; ",
        "ld.b32 a1, [$1+4]; ",
        "ld.b32 a2, [$1+8]; ",
        "ld.b32 a3, [$1+12]; ",
        "ld.b32 b0, [$2]; ",
        "ld.b32 b1, [$2+4]; ",
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ",
        "{d0, d1, d2, d3}, ",
        "{a0, a1, a2, a3}, ",
        "{b0, b1}, ",
        "{d0, d1, d2, d3}; ",
        "st.f32 [$0], d0; ",
        "st.f32 [$0+4], d1; ",
        "st.f32 [$0+8], d2; ",
        "st.f32 [$0+12], d3; ",
        "}"
    );

    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![acc_ptr, a_ptr, b_ptr],
        asm,
        "l,l,l,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}
