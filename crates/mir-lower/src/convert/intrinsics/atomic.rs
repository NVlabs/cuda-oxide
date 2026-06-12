/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Atomic operation conversion: NVVM atomic dialect → LLVM atomic instructions.
//!
//! Converts NVVM atomic ops to standard LLVM atomic instructions with
//! proper ordering and syncscope.
//!
//! # Lowering Strategy
//!
//! Unlike most GPU intrinsics that lower to LLVM NVVM intrinsic calls or
//! inline PTX, atomic operations lower to **standard LLVM IR instructions**:
//!
//! | NVVM Op                 | LLVM IR                                  |
//! |-------------------------|------------------------------------------|
//! | `NvvmAtomicLoadOp`      | `load atomic ... syncscope("device")`    |
//! | `NvvmAtomicStoreOp`     | `store atomic ... syncscope("device")`   |
//! | `NvvmAtomicRmwOp`       | `atomicrmw ... syncscope("device")` `[*]`  |
//! | `NvvmAtomicCmpxchgOp`   | `cmpxchg ... syncscope("device")`        |
//!
//! `[*]` atomicrmw uses fence splitting workaround -- see below.
//!
//! # atomicrmw Fence Splitting Workaround
//!
//! LLVM's NVPTX backend silently drops orderings on `atomicrmw`
//! (fix is in LLVM 23 via PR #176015). Until then, we emit:
//!
//! ```text
//! Relaxed:  atomicrmw ... monotonic
//! Acquire:  atomicrmw ... monotonic  +  fence acquire
//! Release:  fence release  +  atomicrmw ... monotonic
//! AcqRel:   fence release  +  atomicrmw ... monotonic  +  fence acquire
//! SeqCst:   fence seq_cst  +  atomicrmw ... monotonic  +  fence seq_cst
//! ```
//!
//! All fences carry the same syncscope as the atomic op.
//!
//! # Scope → Syncscope Mapping
//!
//! | NVVM Scope | LLVM syncscope     | PTX scope |
//! |------------|--------------------|-----------|
//! | Device     | `"device"`         | `.gpu`    |
//! | Block      | `"block"`          | `.cta`    |
//! | System     | (default)          | `.sys`    |

use crate::convert::types::convert_type;

use dialect_nvvm::ops::atomic::{
    AtomicOrdering as NvvmOrdering, AtomicRmwKind as NvvmRmwKind, AtomicScope as NvvmScope,
    NvvmAtomicCmpxchgOp, NvvmAtomicLoadOp, NvvmAtomicOpInterface, NvvmAtomicRmwOp,
    NvvmAtomicStoreOp,
};
use llvm_export::attributes::{LlvmAtomicOrdering, LlvmAtomicRmwKind, LlvmSyncScope};
use llvm_export::ops as llvm;

use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::Typed;

// =============================================================================
// Scope / Ordering Mapping
// =============================================================================

fn map_scope(scope: &NvvmScope) -> LlvmSyncScope {
    match scope {
        NvvmScope::Device => LlvmSyncScope::Device,
        NvvmScope::Block => LlvmSyncScope::Block,
        NvvmScope::System => LlvmSyncScope::System,
    }
}

fn map_ordering(ord: &NvvmOrdering) -> LlvmAtomicOrdering {
    match ord {
        NvvmOrdering::Relaxed => LlvmAtomicOrdering::Monotonic,
        NvvmOrdering::Acquire => LlvmAtomicOrdering::Acquire,
        NvvmOrdering::Release => LlvmAtomicOrdering::Release,
        NvvmOrdering::AcqRel => LlvmAtomicOrdering::AcqRel,
        NvvmOrdering::SeqCst => LlvmAtomicOrdering::SeqCst,
    }
}

fn map_rmw_kind(kind: &NvvmRmwKind) -> LlvmAtomicRmwKind {
    match kind {
        NvvmRmwKind::Add => LlvmAtomicRmwKind::Add,
        NvvmRmwKind::Sub => LlvmAtomicRmwKind::Sub,
        NvvmRmwKind::And => LlvmAtomicRmwKind::And,
        NvvmRmwKind::Or => LlvmAtomicRmwKind::Or,
        NvvmRmwKind::Xor => LlvmAtomicRmwKind::Xor,
        NvvmRmwKind::Xchg => LlvmAtomicRmwKind::Xchg,
        NvvmRmwKind::Min => LlvmAtomicRmwKind::Min,
        NvvmRmwKind::Max => LlvmAtomicRmwKind::Max,
        NvvmRmwKind::UMin => LlvmAtomicRmwKind::UMin,
        NvvmRmwKind::UMax => LlvmAtomicRmwKind::UMax,
        NvvmRmwKind::FAdd => LlvmAtomicRmwKind::FAdd,
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn emit_fence(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    ordering: LlvmAtomicOrdering,
    syncscope: LlvmSyncScope,
) {
    let fence = llvm::FenceOp::new(ctx, ordering, syncscope.to_pliron());
    rewriter.insert_operation(ctx, fence.get_operation());
}

// =============================================================================
// Load
// =============================================================================

pub(crate) fn convert_atomic_load(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let nvvm_op = NvvmAtomicLoadOp::new(op);
    let ordering = map_ordering(&nvvm_op.ordering(ctx));
    let syncscope = map_scope(&nvvm_op.scope(ctx));

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let ptr = operands[0];
    let mir_result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let result_ty =
        convert_type(ctx, mir_result_ty).map_err(|e| pliron::input_error_noloc!("{}", e))?;

    let llvm_load = llvm::AtomicLoadOp::new(ctx, ptr, result_ty, ordering, syncscope.to_pliron());
    rewriter.insert_operation(ctx, llvm_load.get_operation());
    rewriter.replace_operation(ctx, op, llvm_load.get_operation());

    Ok(())
}

// =============================================================================
// Store
// =============================================================================

pub(crate) fn convert_atomic_store(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let nvvm_op = NvvmAtomicStoreOp::new(op);
    let ordering = map_ordering(&nvvm_op.ordering(ctx));
    let syncscope = map_scope(&nvvm_op.scope(ctx));

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let val = operands[0];
    let ptr = operands[1];

    let llvm_store = llvm::AtomicStoreOp::new(ctx, val, ptr, ordering, syncscope.to_pliron());
    rewriter.insert_operation(ctx, llvm_store.get_operation());
    rewriter.erase_operation(ctx, op);

    Ok(())
}

// =============================================================================
// Read-Modify-Write (with fence splitting workaround)
// =============================================================================

pub(crate) fn convert_atomic_rmw(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let nvvm_op = NvvmAtomicRmwOp::new(op);
    let nvvm_ordering = nvvm_op.ordering(ctx);
    let syncscope = map_scope(&nvvm_op.scope(ctx));
    let rmw_kind = map_rmw_kind(&nvvm_op.rmw_kind(ctx));

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let ptr = operands[0];
    let val = operands[1];

    // Fence splitting workaround for LLVM NVPTX atomicrmw ordering bug.
    // We emit: [optional pre-fence] + atomicrmw monotonic + [optional post-fence]
    // The actual atomicrmw always uses Monotonic because LLVM drops the
    // ordering anyway. The fences provide the correct ordering semantics.

    // Pre-fence (if needed)
    match nvvm_ordering {
        NvvmOrdering::Release | NvvmOrdering::AcqRel => {
            emit_fence(ctx, rewriter, LlvmAtomicOrdering::Release, syncscope);
        }
        NvvmOrdering::SeqCst => {
            emit_fence(ctx, rewriter, LlvmAtomicOrdering::SeqCst, syncscope);
        }
        NvvmOrdering::Relaxed | NvvmOrdering::Acquire => {}
    }

    // The atomicrmw itself -- always Monotonic
    let llvm_rmw = llvm::AtomicRmwOp::new(
        ctx,
        ptr,
        val,
        rmw_kind,
        LlvmAtomicOrdering::Monotonic,
        syncscope.to_pliron(),
    );
    rewriter.insert_operation(ctx, llvm_rmw.get_operation());

    // Post-fence (if needed)
    match nvvm_ordering {
        NvvmOrdering::Acquire | NvvmOrdering::AcqRel => {
            emit_fence(ctx, rewriter, LlvmAtomicOrdering::Acquire, syncscope);
        }
        NvvmOrdering::SeqCst => {
            emit_fence(ctx, rewriter, LlvmAtomicOrdering::SeqCst, syncscope);
        }
        NvvmOrdering::Relaxed | NvvmOrdering::Release => {}
    }

    rewriter.replace_operation(ctx, op, llvm_rmw.get_operation());

    Ok(())
}

// =============================================================================
// Compare-and-Exchange
// =============================================================================

pub(crate) fn convert_atomic_cmpxchg(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let nvvm_op = NvvmAtomicCmpxchgOp::new(op);
    let success_ord = map_ordering(&nvvm_op.success_ordering(ctx));
    let failure_ord = map_ordering(&nvvm_op.failure_ordering(ctx));
    let syncscope = map_scope(&nvvm_op.scope(ctx));

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let ptr = operands[0];
    let cmp = operands[1];
    let new_val = operands[2];
    let llvm_cmpxchg = llvm::AtomicCmpxchgOp::new(
        ctx,
        ptr,
        cmp,
        new_val,
        success_ord,
        failure_ord,
        syncscope.to_pliron(),
    );
    rewriter.insert_operation(ctx, llvm_cmpxchg.get_operation());

    // Upstream `cmpxchg` returns `{ T, i1 }`, but the NVVM op models only the
    // loaded value `T`. Extract element 0 and replace the NVVM op with it; this
    // emits the same `cmpxchg` + `extractvalue` LLVM as the pre-migration path.
    let cmpxchg_res = llvm_cmpxchg.get_operation().deref(ctx).get_result(0);
    let extract = llvm::ExtractValueOp::new(ctx, cmpxchg_res, vec![0])
        .map_err(|e| pliron::input_error_noloc!("{}", e))?;
    rewriter.insert_operation(ctx, extract.get_operation());
    rewriter.replace_operation(ctx, op, extract.get_operation());

    Ok(())
}
