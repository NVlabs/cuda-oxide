/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Blackwell tcgen05 tensor core intrinsics.
//!
//! Handles SM100 (Blackwell) 5th generation tensor core operations.

use super::super::helpers::{emit_goto, emit_store_result_and_goto};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{
    CvtF32x2Bf16x2Op, StmatrixM8n8X2Op, StmatrixM8n8X2TransOp, StmatrixM8n8X4Op,
    StmatrixM8n8X4TransOp, Tcgen05AllocCg2Op, Tcgen05AllocOp, Tcgen05CommitCg2Op,
    Tcgen05CommitMulticastCg2Op, Tcgen05CommitOp, Tcgen05CommitSharedClusterCg2Op,
    Tcgen05CommitSharedClusterOp, Tcgen05CpSmemToTmemCg2Op, Tcgen05CpSmemToTmemOp,
    Tcgen05DeallocCg2Op, Tcgen05DeallocOp, Tcgen05FenceAfterThreadSyncOp,
    Tcgen05FenceBeforeThreadSyncOp, Tcgen05Ld16x256bPureOp, Tcgen05Ld16x256bX8PureOp,
    Tcgen05LoadWaitOp, Tcgen05MmaF16Cg2Op, Tcgen05MmaF16Op, Tcgen05MmaWsBf16Op, Tcgen05MmaWsF16Op,
    Tcgen05MmaWsTf32Op, Tcgen05RelinquishAllocPermitCg2Op, Tcgen05RelinquishAllocPermitOp,
    Tcgen05StoreWaitOp,
};
// NOTE: Removed imports for deprecated ops (now in cuda-core as builders):
// Tcgen05MakeSmemDescOp, Tcgen05MakeSmemDescStridedOp, Tcgen05StTmemToSmemOp,
// Tcgen05StTmemToSmemOffsetOp, Tcgen05Ld16x256bX4Op, Tcgen05Ld16x256bX8Op,
// Tcgen05Ld16x256bX16Op, Tcgen05Ld16x256bX32Op, Tcgen05Ld32x32bX64Op
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::TypeObj;
use pliron::value::Value;
use rustc_public::mir;

/// Translate the destination place's Rust-declared type (typically a
/// `CuSimd<T, N>` alias) to its full-layout MIR struct type.
///
/// Intrinsic handlers that produce struct-typed results must use this type
/// so the constructed value matches the alloca slot emitted for the
/// destination local. Manufacturing a layout-less `MirStructType::get(...)`
/// struct would diverge from the ADT-translated slot type and trip the
/// `MirStoreOp` verifier.
fn destination_struct_type(
    ctx: &mut Context,
    body: &mir::Body,
    destination: &mir::Place,
    loc: Location,
) -> TranslationResult<Ptr<TypeObj>> {
    let dest_rust_ty = match destination.ty(body.locals()) {
        Ok(t) => t,
        Err(e) => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "failed to resolve destination type for intrinsic result: {e:?}"
                ))
            );
        }
    };
    types::translate_type(ctx, &dest_rust_ty)
}

/// Emits `tcgen05_alloc(dst_smem, n_cols)`: Allocate Tensor Memory (TMEM).
///
/// Allocates tensor memory for the 5th generation tensor core operations.
/// The allocated TMEM address is written to shared memory at `dst_smem`.
///
/// # Warp-Synchronous
///
/// This instruction is WARP-SYNCHRONOUS: all 32 threads in a warp must
/// execute together with identical arguments.
///
/// # Arguments
///
/// - `args[0]`: `*mut u32` - Destination in shared memory for TMEM address
/// - `args[1]`: `u32` - Number of columns to allocate
///
/// # Blackwell+ Only
///
/// This instruction is only available on SM100 (Blackwell) and later.
pub fn emit_tcgen05_alloc(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_alloc expects 2 arguments (dst_smem, n_cols), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    // arg[0]: dst_smem (pointer to shared memory)
    let (dst_smem, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // arg[1]: n_cols (u32)
    let (n_cols, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let alloc_op = Operation::new(
        ctx,
        Tcgen05AllocOp::get_concrete_op_info(),
        vec![],                 // No results (void)
        vec![dst_smem, n_cols], // Operands
        vec![],
        0,
    );
    alloc_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        alloc_op.insert_after(ctx, prev);
    } else {
        alloc_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, alloc_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("tcgen05_alloc call without target block".to_string())
        )
    }
}

/// Emit tcgen05_dealloc: Deallocate Tensor Memory (TMEM).
///
/// This is WARP-SYNCHRONOUS: all 32 threads in a warp must execute together.
/// MUST be called for all allocations before kernel exits!
///
/// Args:
/// - `args[0]`: u32 (tmem_addr - the TMEM address from tcgen05_alloc)
/// - `args[1]`: u32 (n_cols - must match the allocation)
///
/// Returns: void
pub fn emit_tcgen05_dealloc(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_dealloc expects 2 arguments (tmem_addr, n_cols), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    // arg[0]: tmem_addr (u32)
    let (tmem_addr, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // arg[1]: n_cols (u32)
    let (n_cols, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let dealloc_op = Operation::new(
        ctx,
        Tcgen05DeallocOp::get_concrete_op_info(),
        vec![],                  // No results (void)
        vec![tmem_addr, n_cols], // Operands
        vec![],
        0,
    );
    dealloc_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        dealloc_op.insert_after(ctx, prev);
    } else {
        dealloc_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, dealloc_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("tcgen05_dealloc call without target block".to_string())
        )
    }
}

/// Emit tcgen05_relinquish_alloc_permit: Relinquish the right to allocate TMEM.
///
/// After calling this, no more tcgen05_alloc calls are allowed in the CTA.
/// This is an optional optimization hint.
///
/// Args: none
/// Returns: void
pub fn emit_tcgen05_relinquish_alloc_permit(
    ctx: &mut Context,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if !args.is_empty() {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_relinquish_alloc_permit expects 0 arguments, got {}",
                args.len()
            ))
        );
    }

    let relinquish_op = Operation::new(
        ctx,
        Tcgen05RelinquishAllocPermitOp::get_concrete_op_info(),
        vec![], // No results
        vec![], // No operands
        vec![],
        0,
    );
    relinquish_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = prev_op {
        relinquish_op.insert_after(ctx, prev);
    } else {
        relinquish_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, relinquish_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "tcgen05_relinquish_alloc_permit call without target block".to_string()
            )
        )
    }
}

/// Emit tcgen05_fence_before_thread_sync: Fence for ordering BEFORE thread sync.
///
/// Use this before signaling other threads (e.g., via relaxed store to a flag).
/// Ensures prior tcgen05 ops complete before the signal.
///
/// Args: none
/// Returns: void
pub fn emit_tcgen05_fence_before_thread_sync(
    ctx: &mut Context,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if !args.is_empty() {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_fence_before_thread_sync expects 0 arguments, got {}",
                args.len()
            ))
        );
    }

    let fence_op = Operation::new(
        ctx,
        Tcgen05FenceBeforeThreadSyncOp::get_concrete_op_info(),
        vec![], // No results
        vec![], // No operands
        vec![],
        0,
    );
    fence_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = prev_op {
        fence_op.insert_after(ctx, prev);
    } else {
        fence_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, fence_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "tcgen05_fence_before_thread_sync call without target block".to_string()
            )
        )
    }
}

/// Emit tcgen05_fence_after_thread_sync: Fence for ordering AFTER thread sync.
///
/// Use this after receiving a signal from another thread (e.g., via relaxed load).
/// Ensures TMEM access is safe after receiving the signal.
///
/// Args: none
/// Returns: void
pub fn emit_tcgen05_fence_after_thread_sync(
    ctx: &mut Context,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if !args.is_empty() {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_fence_after_thread_sync expects 0 arguments, got {}",
                args.len()
            ))
        );
    }

    let fence_op = Operation::new(
        ctx,
        Tcgen05FenceAfterThreadSyncOp::get_concrete_op_info(),
        vec![], // No results
        vec![], // No operands
        vec![],
        0,
    );
    fence_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = prev_op {
        fence_op.insert_after(ctx, prev);
    } else {
        fence_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, fence_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "tcgen05_fence_after_thread_sync call without target block".to_string()
            )
        )
    }
}

/// Emit tcgen05_commit: Commit pending tcgen05 operations to an mbarrier.
///
/// The mbarrier will signal when all prior tcgen05 ops complete.
/// Use mbarrier_try_wait to wait for completion.
///
/// Args:
/// - `args[0]`: *mut u64 (mbar - pointer to mbarrier in shared memory)
///
/// Returns: void
pub fn emit_tcgen05_commit(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_commit expects 1 argument (mbar), got {}",
                args.len()
            ))
        );
    }

    let (mbar, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let commit_op = Operation::new(
        ctx,
        Tcgen05CommitOp::get_concrete_op_info(),
        vec![],     // No results (void)
        vec![mbar], // Operands
        vec![],
        0,
    );
    commit_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        commit_op.insert_after(ctx, prev);
    } else {
        commit_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, commit_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("tcgen05_commit call without target block".to_string())
        )
    }
}

/// Emit tcgen05_commit_shared_cluster: Commit pending tcgen05 operations to an mbarrier
/// using the `.shared::cluster` address space variant.
///
/// Args:
/// - `args[0]`: *mut u64 (mbar - pointer to mbarrier in shared memory)
///
/// Returns: void
pub fn emit_tcgen05_commit_shared_cluster(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_commit_shared_cluster expects 1 argument (mbar), got {}",
                args.len()
            ))
        );
    }

    let (mbar, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let commit_op = Operation::new(
        ctx,
        Tcgen05CommitSharedClusterOp::get_concrete_op_info(),
        vec![],     // No results (void)
        vec![mbar], // Operands
        vec![],
        0,
    );
    commit_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        commit_op.insert_after(ctx, prev);
    } else {
        commit_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, commit_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "tcgen05_commit_shared_cluster call without target block".to_string()
            )
        )
    }
}

// NOTE: emit_tcgen05_make_smem_desc and emit_tcgen05_make_smem_desc_strided removed.
// Use Tcgen05SmemDescriptor::builder() in cuda-core instead.

/// Emit tcgen05_mma_ws_f16: Matrix multiply-accumulate with f16 inputs.
///
/// D = A × B + D (or D = A × B if enable_d is false)
///
/// **SINGLE-THREAD SEMANTICS**: Unlike WGMMA, only ONE thread issues this instruction!
///
/// Args:
/// - `args[0]`: u32 (d_tmem - TMEM address for D matrix)
/// - `args[1]`: u32 (a_tmem - TMEM address for A matrix)
/// - `args[2]`: u64 (a_desc - SMEM descriptor for A)
/// - `args[3]`: u64 (b_desc - SMEM descriptor for B)
/// - `args[4]`: u32 (idesc - instruction descriptor)
/// - `args[5]`: bool (enable_d - true to accumulate, false to overwrite)
///
/// Returns: void
pub fn emit_tcgen05_mma_ws_f16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 6 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_mma_ws_f16 expects 6 arguments, got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    // Translate all 6 arguments
    let (d_tmem, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (a_tmem, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (a_desc, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (b_desc, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[3],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (idesc, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[4],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (enable_d, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[5],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let mma_op = Operation::new(
        ctx,
        Tcgen05MmaWsF16Op::get_concrete_op_info(),
        vec![], // No results (void)
        vec![d_tmem, a_tmem, a_desc, b_desc, idesc, enable_d],
        vec![],
        0,
    );
    mma_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        mma_op.insert_after(ctx, prev);
    } else {
        mma_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, mma_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("tcgen05_mma_ws_f16 call without target block".to_string())
        )
    }
}

/// Emit nvvm.tcgen05_mma_f16 (non-ws MMA: A/B from SMEM descriptors).
///
/// Args: (d_tmem: u32, a_desc: u64, b_desc: u64, idesc: u32, enable_d: bool)
/// Returns: void
pub fn emit_tcgen05_mma_f16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 5 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_mma_f16 expects 5 arguments, got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    let (d_tmem, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (a_desc, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (b_desc, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (idesc, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[3],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (enable_d, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[4],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let mma_op = Operation::new(
        ctx,
        Tcgen05MmaF16Op::get_concrete_op_info(),
        vec![], // No results (void)
        vec![d_tmem, a_desc, b_desc, idesc, enable_d],
        vec![],
        0,
    );
    mma_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        mma_op.insert_after(ctx, prev);
    } else {
        mma_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, mma_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("tcgen05_mma_bf16 call without target block".to_string())
        )
    }
}

/// Emit tcgen05_mma_ws_bf16: Matrix multiply-accumulate with bf16 inputs.
///
/// Same semantics as tcgen05_mma_ws_f16 but with bfloat16 inputs.
pub fn emit_tcgen05_mma_ws_bf16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 6 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_mma_ws_bf16 expects 6 arguments, got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    let (d_tmem, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (a_tmem, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (a_desc, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (b_desc, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[3],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (idesc, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[4],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (enable_d, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[5],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let mma_op = Operation::new(
        ctx,
        Tcgen05MmaWsBf16Op::get_concrete_op_info(),
        vec![],
        vec![d_tmem, a_tmem, a_desc, b_desc, idesc, enable_d],
        vec![],
        0,
    );
    mma_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        mma_op.insert_after(ctx, prev);
    } else {
        mma_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, mma_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "tcgen05_mma_ws_bf16 call without target block".to_string()
            )
        )
    }
}

/// Emit tcgen05_mma_ws_tf32: Matrix multiply-accumulate with tf32 inputs.
///
/// TensorFloat-32 provides better precision than f16/bf16 while maintaining
/// high tensor core throughput.
pub fn emit_tcgen05_mma_ws_tf32(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 6 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_mma_ws_tf32 expects 6 arguments, got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    let (d_tmem, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (a_tmem, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (a_desc, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (b_desc, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[3],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (idesc, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[4],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (enable_d, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[5],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let mma_op = Operation::new(
        ctx,
        Tcgen05MmaWsTf32Op::get_concrete_op_info(),
        vec![],
        vec![d_tmem, a_tmem, a_desc, b_desc, idesc, enable_d],
        vec![],
        0,
    );
    mma_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        mma_op.insert_after(ctx, prev);
    } else {
        mma_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, mma_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "tcgen05_mma_ws_tf32 call without target block".to_string()
            )
        )
    }
}

/// Emit tcgen05_cp_smem_to_tmem: Copy data from shared memory to tensor memory.
///
/// This is used to load matrix A into TMEM before MMA operations.
///
/// Args:
/// - `args[0]`: u32 (tmem_addr - destination address in tensor memory)
/// - `args[1]`: u64 (smem_desc - source shared memory descriptor)
///
/// Returns: void
pub fn emit_tcgen05_cp_smem_to_tmem(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_cp_smem_to_tmem expects 2 arguments (tmem_addr, smem_desc), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    // arg[0]: tmem_addr (u32 - destination in TMEM)
    let (tmem_addr, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // arg[1]: smem_desc (u64 - source SMEM descriptor)
    let (smem_desc, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let cp_op = Operation::new(
        ctx,
        Tcgen05CpSmemToTmemOp::get_concrete_op_info(),
        vec![],                     // No results (void)
        vec![tmem_addr, smem_desc], // Operands
        vec![],
        0,
    );
    cp_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        cp_op.insert_after(ctx, prev);
    } else {
        cp_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, cp_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "tcgen05_cp_smem_to_tmem call without target block".to_string()
            )
        )
    }
}

// NOTE: emit_tcgen05_st_tmem_to_smem removed (incorrect approach)

// NOTE: emit_tcgen05_st_tmem_to_smem_offset removed (incorrect approach)

// NOTE: emit_tcgen05_ld_16x256b_x4 removed (wrong design - stores to SMEM)

// NOTE: emit_tcgen05_ld_16x256b_x8 removed (wrong design - stores to SMEM)
// Use emit_tcgen05_ld_16x256b_x8_pure instead

// NOTE: emit_tcgen05_ld_16x256b_x16 removed (wrong design - stores to SMEM)

// NOTE: emit_tcgen05_ld_16x256b_x32 removed (wrong design - stores to SMEM)

// NOTE: emit_tcgen05_ld_32x32b_x64 removed (wrong design - stores to SMEM)

/// Emit stmatrix_m8n8_x4: Warp-cooperative matrix store.
///
/// Args: (smem_ptr: *mut u8, r0: f32, r1: f32, r2: f32, r3: f32)
/// Returns: void
pub fn emit_stmatrix_m8n8_x4(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 5 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "stmatrix_m8n8_x4 expects 5 arguments (smem_ptr, r0, r1, r2, r3), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(5);

    for arg in args.iter().take(5) {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }

    let st_op = Operation::new(
        ctx,
        StmatrixM8n8X4Op::get_concrete_op_info(),
        vec![],
        operands,
        vec![],
        0,
    );
    st_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        st_op.insert_after(ctx, prev);
    } else {
        st_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, st_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("stmatrix_m8n8_x4 call without target block".to_string())
        )
    }
}

/// Emit stmatrix_m8n8_x4_trans: Warp-cooperative matrix store with transpose.
///
/// This version uses the `.trans` modifier to transform data from fragment
/// layout to row-major layout during the store operation.
///
/// Args: (smem_ptr: *mut u8, r0: u32, r1: u32, r2: u32, r3: u32)
///       where each u32 contains 2 packed bf16 values
/// Returns: void
pub fn emit_stmatrix_m8n8_x4_trans(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 5 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "stmatrix_m8n8_x4_trans expects 5 arguments (smem_ptr, r0, r1, r2, r3), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(5);

    for arg in args.iter().take(5) {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }

    let st_op = Operation::new(
        ctx,
        StmatrixM8n8X4TransOp::get_concrete_op_info(),
        vec![],
        operands,
        vec![],
        0,
    );
    st_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        st_op.insert_after(ctx, prev);
    } else {
        st_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, st_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "stmatrix_m8n8_x4_trans call without target block".to_string()
            )
        )
    }
}

/// Emit tcgen05_ld_16x256b_x8_pure: Pure TMEM load returning 32 f32 values.
///
/// Unlike emit_tcgen05_ld_16x256b_x8, this returns values in registers (no SMEM store).
/// The result is a `CuSimd<f32, 32>` that can be used for subsequent operations.
///
/// Args: (tmem_addr: u32)
/// Returns: CuSimd<f32, 32> (TmemF32x32 type alias)
pub fn emit_tcgen05_ld_16x256b_x8_pure(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_ld_16x256b_x8_pure expects 1 argument (tmem_addr), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    // arg[0]: tmem_addr (u32)
    let (tmem_addr, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // Create 32 f32 result types
    let f32_ty = FP32Type::get(ctx);
    let result_types = (0..32).map(|_| f32_ty.into()).collect();

    let ld_op = Operation::new(
        ctx,
        Tcgen05Ld16x256bX8PureOp::get_concrete_op_info(),
        result_types,
        vec![tmem_addr],
        vec![],
        0,
    );
    ld_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        ld_op.insert_after(ctx, prev);
    } else {
        ld_op.insert_at_front(block_ptr, ctx);
    }

    // Get all 32 f32 results from the load operation
    let results: Vec<Value> = (0..32).map(|i| ld_op.deref(ctx).get_result(i)).collect();

    // Create array type [f32; 32] for CuSimd's data field
    let array_ty = dialect_mir::types::MirArrayType::get(ctx, f32_ty.into(), 32);

    // Create a construct_array operation to bundle the 32 results into an array
    let array_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructArrayOp::get_concrete_op_info(),
        vec![array_ty.into()],
        results,
        vec![],
        0,
    );
    array_op.deref_mut(ctx).set_loc(loc.clone());
    array_op.insert_after(ctx, ld_op);

    // Use the destination local's Rust-declared type so the constructed
    // `CuSimd` struct matches the alloca slot (same `get_with_full_layout`
    // layout info that ADT translation produces). Without this, the struct
    // produced here would lack layout info and fail the MirStoreOp verifier.
    let struct_ty = destination_struct_type(ctx, body, destination, loc.clone())?;

    // Create a construct_struct operation to wrap the array in CuSimd
    let array_result = array_op.deref(ctx).get_result(0);
    let struct_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructStructOp::get_concrete_op_info(),
        vec![struct_ty],
        vec![array_result],
        vec![],
        0,
    );
    struct_op.deref_mut(ctx).set_loc(loc.clone());
    struct_op.insert_after(ctx, array_op);

    let struct_result = struct_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        struct_result,
        target,
        block_ptr,
        struct_op,
        value_map,
        block_map,
        loc,
        "tcgen05_ld_16x256b_x8_pure call without target block",
    )
}

/// Emit tcgen05_ld_16x256b_pure: Base TMEM load returning 4 f32 values.
///
/// This is the base LDTM.16dp256bit instruction (no .x8 multiplier).
/// Returns 4 f32 values per thread for use with stmatrix.m8n8.x2.
///
/// Args: (tmem_addr: u32)
/// Returns: CuSimd<f32, 4> (TmemF32x4 type alias)
pub fn emit_tcgen05_ld_16x256b_pure(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_ld_16x256b_pure expects 1 argument (tmem_addr), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    // arg[0]: tmem_addr (u32)
    let (tmem_addr, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // Create 4 f32 result types
    let f32_ty = FP32Type::get(ctx);
    let result_types = (0..4).map(|_| f32_ty.into()).collect();

    let ld_op = Operation::new(
        ctx,
        Tcgen05Ld16x256bPureOp::get_concrete_op_info(),
        result_types,
        vec![tmem_addr],
        vec![],
        0,
    );
    ld_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        ld_op.insert_after(ctx, prev);
    } else {
        ld_op.insert_at_front(block_ptr, ctx);
    }

    // Get all 4 f32 results from the load operation
    let results: Vec<Value> = (0..4).map(|i| ld_op.deref(ctx).get_result(i)).collect();

    // Create array type [f32; 4] for CuSimd's data field
    let array_ty = dialect_mir::types::MirArrayType::get(ctx, f32_ty.into(), 4);

    // Create a construct_array operation to bundle the 4 results into an array
    let array_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructArrayOp::get_concrete_op_info(),
        vec![array_ty.into()],
        results,
        vec![],
        0,
    );
    array_op.deref_mut(ctx).set_loc(loc.clone());
    array_op.insert_after(ctx, ld_op);

    // Use the destination local's Rust-declared type so the constructed
    // `CuSimd` struct matches the alloca slot (same `get_with_full_layout`
    // layout info that ADT translation produces). Without this, the struct
    // produced here would lack layout info and fail the MirStoreOp verifier.
    let struct_ty = destination_struct_type(ctx, body, destination, loc.clone())?;

    // Create a construct_struct operation to wrap the array in CuSimd
    let array_result = array_op.deref(ctx).get_result(0);
    let struct_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructStructOp::get_concrete_op_info(),
        vec![struct_ty],
        vec![array_result],
        vec![],
        0,
    );
    struct_op.deref_mut(ctx).set_loc(loc.clone());
    struct_op.insert_after(ctx, array_op);

    let struct_result = struct_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        struct_result,
        target,
        block_ptr,
        struct_op,
        value_map,
        block_map,
        loc,
        "tcgen05_ld_16x256b_pure call without target block",
    )
}

/// Emit stmatrix_m8n8_x2: Warp-cooperative matrix store (NON-trans, x2).
///
/// This stores 2 matrix tiles (16 columns) WITHOUT transpose.
/// SASS encoding: STSM.16.MT88.2
///
/// Args: (smem_ptr: *mut u8, r0: u32, r1: u32)
///       where each u32 contains 2 packed bf16 values
/// Returns: void
pub fn emit_stmatrix_m8n8_x2(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "stmatrix_m8n8_x2 expects 3 arguments (smem_ptr, r0, r1), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(3);

    for arg in args.iter().take(3) {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }

    let st_op = Operation::new(
        ctx,
        StmatrixM8n8X2Op::get_concrete_op_info(),
        vec![],
        operands,
        vec![],
        0,
    );
    st_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        st_op.insert_after(ctx, prev);
    } else {
        st_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, st_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("stmatrix_m8n8_x2 call without target block".to_string())
        )
    }
}

/// Emit stmatrix.m8n8.x2.trans - TRANSPOSE version matching cuBLAS STSM.16.MT88.2.
///
/// Args: (smem_ptr: *mut u8, r0: u32, r1: u32)
///       where each u32 contains 2 packed bf16 values
/// Returns: void
pub fn emit_stmatrix_m8n8_x2_trans(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "stmatrix_m8n8_x2_trans expects 3 arguments (smem_ptr, r0, r1), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(3);

    for arg in args.iter().take(3) {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }

    let st_op = Operation::new(
        ctx,
        StmatrixM8n8X2TransOp::get_concrete_op_info(),
        vec![],
        operands,
        vec![],
        0,
    );
    st_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        st_op.insert_after(ctx, prev);
    } else {
        st_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, st_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "stmatrix_m8n8_x2_trans call without target block".to_string()
            )
        )
    }
}

/// Emit cvt_f32x2_bf16x2: Convert two f32 to packed bf16x2.
///
/// Args: (a: f32, b: f32)
/// Returns: u32 (packed bf16x2)
pub fn emit_cvt_f32x2_bf16x2(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "cvt_f32x2_bf16x2 expects 2 arguments (a: f32, b: f32), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    // arg[0]: a (f32)
    let (a_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // arg[1]: b (f32)
    let (b_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // Result is u32 (packed bf16x2); Rust-side signature is `u32` and the
    // destination local is unsigned, so match that here to avoid the
    // MirStoreOp verifier flagging a signless-vs-unsigned mismatch.
    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let cvt_op = Operation::new(
        ctx,
        CvtF32x2Bf16x2Op::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![a_val, b_val],
        vec![],
        0,
    );
    cvt_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        cvt_op.insert_after(ctx, prev);
    } else {
        cvt_op.insert_at_front(block_ptr, ctx);
    }

    let result = cvt_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        cvt_op,
        value_map,
        block_map,
        loc,
        "cvt_f32x2_bf16x2 call without target block",
    )
}

/// Emit tcgen05_load_wait: Wait for tcgen05.ld operations to complete.
///
/// Args: none
/// Returns: void
pub fn emit_tcgen05_load_wait(
    ctx: &mut Context,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if !args.is_empty() {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_load_wait expects 0 arguments, got {}",
                args.len()
            ))
        );
    }

    let wait_op = Operation::new(
        ctx,
        Tcgen05LoadWaitOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    wait_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = prev_op {
        wait_op.insert_after(ctx, prev);
    } else {
        wait_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, wait_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("tcgen05_load_wait call without target block".to_string())
        )
    }
}

/// Emit tcgen05_store_wait: Wait for tcgen05.st operations to complete.
///
/// Args: none
/// Returns: void
pub fn emit_tcgen05_store_wait(
    ctx: &mut Context,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if !args.is_empty() {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_store_wait expects 0 arguments, got {}",
                args.len()
            ))
        );
    }

    let wait_op = Operation::new(
        ctx,
        Tcgen05StoreWaitOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    wait_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = prev_op {
        wait_op.insert_after(ctx, prev);
    } else {
        wait_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, wait_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("tcgen05_store_wait call without target block".to_string())
        )
    }
}

// =============================================================================
// CTA Pair (cta_group::2) Variants
// =============================================================================

pub fn emit_tcgen05_alloc_cg2(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_alloc_cg2 expects 2 arguments, got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    let (dst_smem, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (n_cols, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let alloc_op = Operation::new(
        ctx,
        Tcgen05AllocCg2Op::get_concrete_op_info(),
        vec![],
        vec![dst_smem, n_cols],
        vec![],
        0,
    );
    alloc_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        alloc_op.insert_after(ctx, prev);
    } else {
        alloc_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, alloc_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("tcgen05_alloc_cg2 call without target block".to_string())
        )
    }
}

pub fn emit_tcgen05_dealloc_cg2(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_dealloc_cg2 expects 2 arguments, got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    let (tmem_addr, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (n_cols, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let dealloc_op = Operation::new(
        ctx,
        Tcgen05DeallocCg2Op::get_concrete_op_info(),
        vec![],
        vec![tmem_addr, n_cols],
        vec![],
        0,
    );
    dealloc_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        dealloc_op.insert_after(ctx, prev);
    } else {
        dealloc_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, dealloc_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "tcgen05_dealloc_cg2 call without target block".to_string()
            )
        )
    }
}

pub fn emit_tcgen05_relinquish_alloc_permit_cg2(
    ctx: &mut Context,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if !args.is_empty() {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_relinquish_alloc_permit_cg2 expects 0 arguments, got {}",
                args.len()
            ))
        );
    }

    let relinquish_op = Operation::new(
        ctx,
        Tcgen05RelinquishAllocPermitCg2Op::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    relinquish_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = prev_op {
        relinquish_op.insert_after(ctx, prev);
    } else {
        relinquish_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, relinquish_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "tcgen05_relinquish_alloc_permit_cg2 call without target block".to_string()
            )
        )
    }
}

pub fn emit_tcgen05_mma_f16_cg2(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 5 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_mma_f16_cg2 expects 5 arguments, got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    let (d_tmem, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (a_desc, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (b_desc, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (idesc, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[3],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (enable_d, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[4],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let mma_op = Operation::new(
        ctx,
        Tcgen05MmaF16Cg2Op::get_concrete_op_info(),
        vec![],
        vec![d_tmem, a_desc, b_desc, idesc, enable_d],
        vec![],
        0,
    );
    mma_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        mma_op.insert_after(ctx, prev);
    } else {
        mma_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, mma_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "tcgen05_mma_f16_cg2 call without target block".to_string()
            )
        )
    }
}

pub fn emit_tcgen05_commit_cg2(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_commit_cg2 expects 1 argument, got {}",
                args.len()
            ))
        );
    }

    let (mbar, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let commit_op = Operation::new(
        ctx,
        Tcgen05CommitCg2Op::get_concrete_op_info(),
        vec![],
        vec![mbar],
        vec![],
        0,
    );
    commit_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        commit_op.insert_after(ctx, prev);
    } else {
        commit_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, commit_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("tcgen05_commit_cg2 call without target block".to_string())
        )
    }
}

pub fn emit_tcgen05_commit_shared_cluster_cg2(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_commit_shared_cluster_cg2 expects 1 argument, got {}",
                args.len()
            ))
        );
    }

    let (mbar, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let commit_op = Operation::new(
        ctx,
        Tcgen05CommitSharedClusterCg2Op::get_concrete_op_info(),
        vec![],
        vec![mbar],
        vec![],
        0,
    );
    commit_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        commit_op.insert_after(ctx, prev);
    } else {
        commit_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, commit_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "tcgen05_commit_shared_cluster_cg2 call without target block".to_string()
            )
        )
    }
}

pub fn emit_tcgen05_commit_multicast_cg2(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_commit_multicast_cg2 expects 2 arguments (mbar, cta_mask), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    let (mbar, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (cta_mask, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let commit_op = Operation::new(
        ctx,
        Tcgen05CommitMulticastCg2Op::get_concrete_op_info(),
        vec![],
        vec![mbar, cta_mask],
        vec![],
        0,
    );
    commit_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        commit_op.insert_after(ctx, prev);
    } else {
        commit_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, commit_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "tcgen05_commit_multicast_cg2 call without target block".to_string()
            )
        )
    }
}

pub fn emit_tcgen05_cp_smem_to_tmem_cg2(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "tcgen05_cp_smem_to_tmem_cg2 expects 2 arguments, got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    let (tmem_addr, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (smem_desc, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let cp_op = Operation::new(
        ctx,
        Tcgen05CpSmemToTmemCg2Op::get_concrete_op_info(),
        vec![],
        vec![tmem_addr, smem_desc],
        vec![],
        0,
    );
    cp_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        cp_op.insert_after(ctx, prev);
    } else {
        cp_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, cp_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "tcgen05_cp_smem_to_tmem_cg2 call without target block".to_string()
            )
        )
    }
}
