/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tcgen05 (Tensor Core Gen 5) intrinsic conversion for Blackwell+ GPUs.
//!
//! # IMPORTANT: LLVM Backend Limitations
//!
//! LLVM 20/21 have tcgen05 intrinsics DECLARED but NOT LOWERED properly.
//! The backend emits `.extern .func` instead of actual PTX instructions.
//! Therefore, we use inline PTX assembly for all tcgen05 operations.
//!
//! # Key Differences from WGMMA
//!
//! - MMA is SINGLE-THREAD (not 128 threads like WGMMA)
//! - TMEM allocation uses 32-bit addresses (opaque handles)
//! - Synchronization via mbarrier (not separate wait instructions)
//!
//! # Operations
//!
//! | Category        | Operations                                                      |
//! |-----------------|-----------------------------------------------------------------|
//! | Allocation      | `Alloc`, `Dealloc`, `RelinquishAllocPermit`                     |
//! | Synchronization | `FenceBeforeThreadSync`, `FenceAfterThreadSync`, `Commit`       |
//! | Memory          | `CpSmemToTmem`, `StTmemToSmem`, `StTmemToSmemOffset`             |
//! | MMA             | `MmaWsF16`, `MmaWsBf16`, `MmaWsTf32`, `MmaF16`                   |
//! | Load            | `Ld16x256bX4`, `Ld16x256bX8`, `Ld16x256bX16`, `Ld16x256bX32`... |

use crate::convert::intrinsics::common::*;
use llvm_export::ops as llvm;
use llvm_export::ops::InlineAsmOpExt;
use llvm_export::types as llvm_types;
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::TypeObj;

// ============================================================================
// Allocation operations
// ============================================================================

/// Convert nvvm.tcgen05_alloc to inline PTX.
///
/// Allocates Tensor Memory (TMEM) columns. This is WARP-SYNCHRONOUS.
/// The TMEM address is written to shared memory at dst_smem.
///
/// PTX: `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [dst], n_cols;`
///
/// IMPORTANT: LLVM may pass a generic address (64-bit), but tcgen05.alloc requires
/// a shared memory address (32-bit). We convert: generic(64) -> shared(64) -> truncate(32).
pub(crate) fn convert_alloc(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 2 {
        return pliron::input_err_noloc!("tcgen05_alloc requires 2 operands");
    }
    let dst_smem = operands[0];
    let n_cols = operands[1];

    let void_ty = llvm_types::VoidType::get(ctx);

    let inline_asm = llvm::InlineAsmOp::new_convergent(
        ctx,
        void_ty.into(),
        vec![dst_smem, n_cols],
        concat!(
            "{ ",
            ".reg .u64 %shared64; ",
            ".reg .u32 %shared32; ",
            "cvta.to.shared.u64 %shared64, $0; ",
            "cvt.u32.u64 %shared32, %shared64; ",
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%shared32], $1; ",
            "}"
        ),
        "l,r,~{memory}",
    );
    rewriter.insert_operation(ctx, inline_asm.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert nvvm.tcgen05_dealloc to inline PTX.
///
/// Deallocates Tensor Memory. This is WARP-SYNCHRONOUS.
/// MUST be called for all allocations before kernel exits!
///
/// PTX: `tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, n_cols;`
pub(crate) fn convert_dealloc(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 2 {
        return pliron::input_err_noloc!("tcgen05_dealloc requires 2 operands");
    }
    let tmem_addr = operands[0];
    let n_cols = operands[1];

    let void_ty = llvm_types::VoidType::get(ctx);
    let inline_asm = llvm::InlineAsmOp::new_convergent(
        ctx,
        void_ty.into(),
        vec![tmem_addr, n_cols],
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, $1;",
        "r,r,~{memory}",
    );
    rewriter.insert_operation(ctx, inline_asm.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert nvvm.tcgen05_relinquish_alloc_permit to inline PTX.
///
/// Relinquishes the right to allocate TMEM. Optional optimization.
///
/// PTX: `tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;`
pub(crate) fn convert_relinquish_alloc_permit(
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
        "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;",
        "~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

// ============================================================================
// Synchronization operations
// ============================================================================

/// Convert nvvm.tcgen05_fence_before_thread_sync to inline PTX.
///
/// Fence for ordering BEFORE thread synchronization.
/// Use before signaling other threads via relaxed memory operations.
///
/// PTX: `tcgen05.fence::before_thread_sync;`
pub(crate) fn convert_fence_before_thread_sync(
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
        "tcgen05.fence::before_thread_sync;",
        "~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert nvvm.tcgen05_fence_after_thread_sync to inline PTX.
///
/// Fence for ordering AFTER thread synchronization.
/// Use after receiving signal from other threads via relaxed memory operations.
///
/// PTX: `tcgen05.fence::after_thread_sync;`
pub(crate) fn convert_fence_after_thread_sync(
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
        "tcgen05.fence::after_thread_sync;",
        "~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert nvvm.tcgen05_commit to inline PTX.
///
/// Commits pending tcgen05 operations to an mbarrier.
/// The mbarrier will signal when all prior tcgen05 ops complete.
///
/// PTX: `tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mbar];`
pub(crate) fn convert_commit(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.is_empty() {
        return pliron::input_err_noloc!("tcgen05_commit requires operand");
    }
    let mbar = operands[0];

    let void_ty = llvm_types::VoidType::get(ctx);
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![mbar],
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [$0];",
        "r,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert nvvm.tcgen05_commit_shared_cluster to inline PTX.
///
/// PTX: `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [mbar];`
pub(crate) fn convert_commit_shared_cluster(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.is_empty() {
        return pliron::input_err_noloc!("tcgen05_commit_shared_cluster requires operand");
    }
    let mbar = operands[0];

    let void_ty = llvm_types::VoidType::get(ctx);
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![mbar],
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];",
        "r,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

// ============================================================================
// MMA operations
// ============================================================================

/// Convert nvvm.tcgen05_mma_ws_* to inline PTX.
///
/// Matrix multiply-accumulate: D = A × B + D (or D = A × B if enable_d is false)
///
/// **SINGLE-THREAD SEMANTICS**: Unlike WGMMA (128 threads), only ONE thread issues this!
///
/// PTX: `tcgen05.mma.ws.cta_group::1.kind::<kind> [d_tmem], [a_tmem], a_desc, b_desc, idesc, enable_d;`
///
/// Parameters:
/// - kind: f16, bf16, or tf32
pub(crate) fn convert_mma_ws(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    kind: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 6 {
        return pliron::input_err_noloc!("tcgen05_mma_ws requires 6 operands");
    }

    let d_tmem = operands[0];
    let a_tmem = operands[1];
    let a_desc = operands[2];
    let b_desc = operands[3];
    let idesc = operands[4];
    let enable_d = operands[5];

    let asm_template = format!(
        concat!(
            "{{ ",
            ".reg .pred %enable_pred; ",
            "setp.ne.s32 %enable_pred, $5, 0; ",
            "tcgen05.mma.ws.cta_group::1.kind::{} [$0], [$1], $3, $4, %enable_pred; ",
            "}}"
        ),
        kind
    );

    let void_ty = llvm_types::VoidType::get(ctx);
    let inline_asm = llvm::InlineAsmOp::new_convergent(
        ctx,
        void_ty.into(),
        vec![d_tmem, a_tmem, a_desc, b_desc, idesc, enable_d],
        &asm_template,
        "r,r,l,l,r,r,~{memory}",
    );
    rewriter.insert_operation(ctx, inline_asm.get_operation());
    rewriter.erase_operation(ctx, op);

    Ok(())
}

/// Convert nvvm.tcgen05_mma_f16 (non-ws) to inline PTX.
///
/// PTX (as emitted by reference implementations):
/// `tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc, {0,0,0,0}, enable_d;`
pub(crate) fn convert_mma_f16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 5 {
        return pliron::input_err_noloc!("tcgen05_mma_f16 requires 5 operands");
    }

    let d_tmem = operands[0];
    let a_desc = operands[1];
    let b_desc = operands[2];
    let idesc = operands[3];
    let enable_d = operands[4];

    let asm_template = concat!(
        "{ ",
        ".reg .pred %enable_pred; ",
        "setp.ne.s32 %enable_pred, $4, 0; ",
        ".reg .u32 %z; ",
        "mov.u32 %z, 0; ",
        "tcgen05.mma.cta_group::1.kind::f16 [$0], $1, $2, $3, {%z, %z, %z, %z}, %enable_pred; ",
        "}"
    );

    let void_ty = llvm_types::VoidType::get(ctx);
    let inline_asm = llvm::InlineAsmOp::new_convergent(
        ctx,
        void_ty.into(),
        vec![d_tmem, a_desc, b_desc, idesc, enable_d],
        asm_template,
        "r,l,l,r,r,~{memory}",
    );
    rewriter.insert_operation(ctx, inline_asm.get_operation());
    rewriter.erase_operation(ctx, op);

    Ok(())
}

// ============================================================================
// Memory copy operations
// ============================================================================

/// Convert nvvm.tcgen05_cp_smem_to_tmem to inline PTX.
///
/// Copies a tile of data from shared memory to tensor memory.
/// This is used to load matrix A into TMEM before MMA operations.
///
/// PTX: `tcgen05.cp.cta_group::1.128x256b [tmem_addr], smem_desc;`
pub(crate) fn convert_cp_smem_to_tmem(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 2 {
        return pliron::input_err_noloc!("tcgen05_cp_smem_to_tmem requires 2 operands");
    }

    let tmem_addr = operands[0];
    let smem_desc = operands[1];

    let void_ty = llvm_types::VoidType::get(ctx);
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![tmem_addr, smem_desc],
        "tcgen05.cp.cta_group::1.128x256b [$0], $1;",
        "r,l,~{memory}",
    );
    rewriter.erase_operation(ctx, op);

    Ok(())
}

// ============================================================================
// Pure load operations (return values in registers)
// ============================================================================

/// Convert nvvm.tcgen05_ld_16x256b_x8_pure to inline PTX.
///
/// This is the PURE load variant - it returns 32 f32 values in registers,
/// NOT storing to shared memory. This is the correct way to do the epilog:
/// tcgen05.ld → regs → convert → stmatrix.
///
/// PTX: tcgen05.ld.sync.aligned.16x256b.x8.b32 {r0..r31}, [tmem_addr];
///
/// Returns 32 f32 results (one per thread register).
pub(crate) fn convert_ld_16x256b_x8_pure(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let tmem_addr = op.deref(ctx).get_operand(0);

    let f32_ty = FP32Type::get(ctx);
    let field_types: Vec<Ptr<TypeObj>> = (0..32).map(|_| f32_ty.into()).collect();
    let struct_ty = llvm_types::StructType::get_unnamed(ctx, field_types);

    let inline_asm = llvm::InlineAsmOp::new_convergent(
        ctx,
        struct_ty.into(),
        vec![tmem_addr],
        concat!(
            "tcgen05.ld.sync.aligned.16x256b.x8.b32 ",
            "{$0,$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,",
            "$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31}, [$32];"
        ),
        "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,r",
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);

    let struct_result = asm_op.deref(ctx).get_result(0);
    let mut extracted_values = Vec::with_capacity(32);
    for i in 0..32u32 {
        let extract_op = llvm::ExtractValueOp::new(ctx, struct_result, vec![i])
            .map_err(|e| pliron::input_error_noloc!("{}", e))?;
        rewriter.insert_operation(ctx, extract_op.get_operation());
        let field_val = extract_op.get_operation().deref(ctx).get_result(0);
        extracted_values.push(field_val);
    }
    rewriter.replace_operation_with_values(ctx, op, extracted_values);

    Ok(())
}

/// Convert nvvm.tcgen05_ld_16x256b_pure to inline PTX.
///
/// This is the base LDTM load (x1 multiplier) - returns 4 f32 values per thread.
/// This is what cuBLAS uses in combination with stmatrix.m8n8.x2.
///
/// PTX: tcgen05.ld.sync.aligned.16x256b.x1.b32 {r0, r1, r2, r3}, [tmem_addr];
///
/// Returns 4 f32 results (one per thread register).
pub(crate) fn convert_ld_16x256b_pure(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let tmem_addr = op.deref(ctx).get_operand(0);

    let f32_ty = FP32Type::get(ctx);
    let field_types: Vec<Ptr<TypeObj>> = (0..4).map(|_| f32_ty.into()).collect();
    let struct_ty = llvm_types::StructType::get_unnamed(ctx, field_types);

    let inline_asm = llvm::InlineAsmOp::new_convergent(
        ctx,
        struct_ty.into(),
        vec![tmem_addr],
        "tcgen05.ld.sync.aligned.16x256b.x1.b32 {$0,$1,$2,$3}, [$4];",
        "=f,=f,=f,=f,r",
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);

    let struct_result = asm_op.deref(ctx).get_result(0);
    let mut extracted_values = Vec::with_capacity(4);
    for i in 0..4u32 {
        let extract_op = llvm::ExtractValueOp::new(ctx, struct_result, vec![i])
            .map_err(|e| pliron::input_error_noloc!("{}", e))?;
        rewriter.insert_operation(ctx, extract_op.get_operation());
        let field_val = extract_op.get_operation().deref(ctx).get_result(0);
        extracted_values.push(field_val);
    }
    rewriter.replace_operation_with_values(ctx, op, extracted_values);

    Ok(())
}

// ============================================================================
// Conversion and synchronization operations
// ============================================================================

/// Convert nvvm.cvt_f32x2_bf16x2 to inline PTX.
///
/// Converts two f32 values to packed bf16x2 using PTX cvt instruction.
/// PTX: cvt.rn.bf16x2.f32 %result, %b, %a;
///
/// The result is a u32 with two bf16 values: (bf16(b) << 16) | bf16(a)
pub(crate) fn convert_cvt_f32x2_bf16x2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 2 {
        return pliron::input_err_noloc!("cvt_f32x2_bf16x2 requires 2 operands");
    }

    let a_val = operands[0];
    let b_val = operands[1];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    // Non-convergent inline asm (this is a pure data conversion, not a collective op)
    let inline_asm = llvm::InlineAsmOp::new(
        ctx,
        i32_ty.into(),
        vec![a_val, b_val],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        false,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// Convert nvvm.tcgen05_load_wait to inline PTX.
///
/// This is a critical synchronization barrier for tcgen05.ld operations.
/// PTX: tcgen05.wait::ld.sync.aligned;
pub(crate) fn convert_load_wait(
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
        "tcgen05.wait::ld.sync.aligned;",
        "~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert nvvm.tcgen05_store_wait to inline PTX.
///
/// This is a synchronization barrier for tcgen05.st operations.
/// PTX: tcgen05.wait::st.sync.aligned;
pub(crate) fn convert_store_wait(
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
        "tcgen05.wait::st.sync.aligned;",
        "~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

// ============================================================================
// CTA Pair (cta_group::2) operations
// ============================================================================

/// PTX: `tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [dst], n_cols;`
pub(crate) fn convert_alloc_cg2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 2 {
        return pliron::input_err_noloc!("tcgen05_alloc_cg2 requires 2 operands");
    }
    let dst_smem = operands[0];
    let n_cols = operands[1];

    let void_ty = llvm_types::VoidType::get(ctx);
    let inline_asm = llvm::InlineAsmOp::new_convergent(
        ctx,
        void_ty.into(),
        vec![dst_smem, n_cols],
        concat!(
            "{ ",
            ".reg .u64 %shared64; ",
            ".reg .u32 %shared32; ",
            "cvta.to.shared.u64 %shared64, $0; ",
            "cvt.u32.u64 %shared32, %shared64; ",
            "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%shared32], $1; ",
            "}"
        ),
        "l,r,~{memory}",
    );
    rewriter.insert_operation(ctx, inline_asm.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// PTX: `tcgen05.dealloc.cta_group::2.sync.aligned.b32 tmem_addr, n_cols;`
pub(crate) fn convert_dealloc_cg2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 2 {
        return pliron::input_err_noloc!("tcgen05_dealloc_cg2 requires 2 operands");
    }
    let tmem_addr = operands[0];
    let n_cols = operands[1];

    let void_ty = llvm_types::VoidType::get(ctx);
    let inline_asm = llvm::InlineAsmOp::new_convergent(
        ctx,
        void_ty.into(),
        vec![tmem_addr, n_cols],
        "tcgen05.dealloc.cta_group::2.sync.aligned.b32 $0, $1;",
        "r,r,~{memory}",
    );
    rewriter.insert_operation(ctx, inline_asm.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// PTX: `tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;`
pub(crate) fn convert_relinquish_alloc_permit_cg2(
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
        "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;",
        "~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// PTX: `tcgen05.mma.cta_group::2.kind::f16 [d], a_desc, b_desc, idesc, {0,...,0}, enable_d;`
///
/// cta_group::2 requires an 8-element disable-output-lane vector (vs 4 for cta_group::1)
/// because the CTA pair spans twice as many TMEM lanes.
pub(crate) fn convert_mma_f16_cg2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 5 {
        return pliron::input_err_noloc!("tcgen05_mma_f16_cg2 requires 5 operands");
    }

    let d_tmem = operands[0];
    let a_desc = operands[1];
    let b_desc = operands[2];
    let idesc = operands[3];
    let enable_d = operands[4];

    let asm_template = concat!(
        "{ ",
        ".reg .pred %enable_pred; ",
        "setp.ne.s32 %enable_pred, $4, 0; ",
        ".reg .u32 %z; ",
        "mov.u32 %z, 0; ",
        "tcgen05.mma.cta_group::2.kind::f16 [$0], $1, $2, $3, {%z, %z, %z, %z, %z, %z, %z, %z}, %enable_pred; ",
        "}"
    );

    let void_ty = llvm_types::VoidType::get(ctx);
    let inline_asm = llvm::InlineAsmOp::new_convergent(
        ctx,
        void_ty.into(),
        vec![d_tmem, a_desc, b_desc, idesc, enable_d],
        asm_template,
        "r,l,l,r,r,~{memory}",
    );
    rewriter.insert_operation(ctx, inline_asm.get_operation());
    rewriter.erase_operation(ctx, op);

    Ok(())
}

/// PTX: `tcgen05.commit.cta_group::2.mbarrier::arrive::one.b64 [mbar];`
pub(crate) fn convert_commit_cg2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.is_empty() {
        return pliron::input_err_noloc!("tcgen05_commit_cg2 requires operand");
    }
    let mbar = operands[0];

    let void_ty = llvm_types::VoidType::get(ctx);
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![mbar],
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.b64 [$0];",
        "r,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// PTX: `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [mbar];`
pub(crate) fn convert_commit_shared_cluster_cg2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.is_empty() {
        return pliron::input_err_noloc!("tcgen05_commit_shared_cluster_cg2 requires operand");
    }
    let mbar = operands[0];

    let void_ty = llvm_types::VoidType::get(ctx);
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![mbar],
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [$0];",
        "r,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Multicast commit for CTA pairs: signals mbarrier in every CTA whose bit
/// is set in `cta_mask`. Used after cooperative MMA to signal both partner
/// CTAs' barriers in one instruction.
///
/// PTX: `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [mbar], ctaMask;`
pub(crate) fn convert_commit_multicast_cg2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 2 {
        return pliron::input_err_noloc!(
            "tcgen05_commit_multicast_cg2 requires 2 operands (mbar, cta_mask)"
        );
    }
    let mbar = operands[0];
    let cta_mask = operands[1];

    let void_ty = llvm_types::VoidType::get(ctx);
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![mbar, cta_mask],
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [$0], $1;",
        "r,h,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// PTX: `tcgen05.cp.cta_group::2.128x256b [tmem_addr], smem_desc;`
pub(crate) fn convert_cp_smem_to_tmem_cg2(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 2 {
        return pliron::input_err_noloc!("tcgen05_cp_smem_to_tmem_cg2 requires 2 operands");
    }

    let tmem_addr = operands[0];
    let smem_desc = operands[1];

    let void_ty = llvm_types::VoidType::get(ctx);
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![tmem_addr, smem_desc],
        "tcgen05.cp.cta_group::2.128x256b [$0], $1;",
        "r,l,~{memory}",
    );
    rewriter.erase_operation(ctx, op);

    Ok(())
}
