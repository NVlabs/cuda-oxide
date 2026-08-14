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
//! RMW and compare-exchange lower to standard LLVM IR instructions. Loads
//! and stores lower to inline PTX, because libNVVM rejects `load atomic` /
//! `store atomic` outright ("Atomic loads/stores are not supported"):
//!
//! | NVVM Op                 | Lowered form                             |
//! |-------------------------|------------------------------------------|
//! | `NvvmAtomicLoadOp`      | inline PTX `ld.{sem}.{scope}.{ty}`       |
//! | `NvvmAtomicStoreOp`     | inline PTX `st.{sem}.{scope}.{ty}`       |
//! | `NvvmAtomicFenceOp`     | inline PTX or `llvm.nvvm.membar.*`       |
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

use crate::convert::intrinsics::common;
use crate::convert::types::convert_type;

use dialect_nvvm::ops::atomic::{
    AtomicOrdering as NvvmOrdering, AtomicRmwKind as NvvmRmwKind, AtomicScope as NvvmScope,
    NvvmAtomicCmpxchgOp, NvvmAtomicFenceOp, NvvmAtomicLoadOp, NvvmAtomicOpInterface,
    NvvmAtomicRmwOp, NvvmAtomicStoreOp,
};
use llvm_export::attributes::{LlvmAtomicOrdering, LlvmAtomicRmwKind, LlvmSyncScope};
use llvm_export::op_interfaces::CastOpInterface;
use llvm_export::ops as llvm;
use llvm_export::ops::{AsmKind, InlineAsmOpExt};
use llvm_export::types as llvm_types;

use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
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

/// PTX type suffix, register constraint class, and (for floats) the integer
/// type the value is staged through.
///
/// The asm always uses an integer register class (`h`/`r`/`l`). A float
/// operand or result is bitcast to and from the same-width integer type at
/// the LLVM dialect level, so the asm operand type always agrees with its
/// constraint; handing llc a float value under an integer constraint is a
/// constraint mismatch.
fn ptx_type_and_reg(
    ctx: &Context,
    ty: pliron::r#type::TypeHandle,
) -> Option<(
    &'static str,
    &'static str,
    Option<pliron::r#type::TypeHandle>,
)> {
    let staging = |width: u32| -> pliron::r#type::TypeHandle {
        IntegerType::get(ctx, width, Signedness::Signless).into()
    };
    let ty_ref = ty.deref(ctx);
    if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        return match int_ty.width() {
            16 => Some(("b16", "h", None)),
            32 => Some(("b32", "r", None)),
            64 => Some(("b64", "l", None)),
            _ => None,
        };
    }
    if ty_ref.is::<llvm_types::HalfType>() {
        return Some(("b16", "h", Some(staging(16))));
    }
    if ty_ref.is::<FP32Type>() {
        return Some(("b32", "r", Some(staging(32))));
    }
    if ty_ref.is::<FP64Type>() {
        return Some(("b64", "l", Some(staging(64))));
    }
    None
}

/// PTX scope qualifier.
fn ptx_scope(scope: &NvvmScope) -> &'static str {
    match scope {
        NvvmScope::Device => "gpu",
        NvvmScope::Block => "cta",
        NvvmScope::System => "sys",
    }
}

/// Inline-PTX template for an atomic load.
///
/// PTX has no sequentially consistent load instruction. libcu++ maps a SeqCst
/// load to `fence.sc.{scope}` followed by an acquire load at the same scope;
/// the same mapping is emitted here, fused into a single asm template so the
/// fence can never be separated from the access.
fn ptx_load_template(ord: &NvvmOrdering, scope: &str, ptx_ty: &str) -> Result<String> {
    match ord {
        NvvmOrdering::Relaxed => Ok(format!("ld.relaxed.{scope}.{ptx_ty} $0, [$1];")),
        NvvmOrdering::Acquire => Ok(format!("ld.acquire.{scope}.{ptx_ty} $0, [$1];")),
        NvvmOrdering::SeqCst => Ok(format!(
            "fence.sc.{scope}; ld.acquire.{scope}.{ptx_ty} $0, [$1];"
        )),
        other => pliron::input_err_noloc!(
            "atomic load cannot have {:?} ordering; use Relaxed, Acquire or SeqCst",
            other
        ),
    }
}

/// Inline-PTX template for an atomic store.
///
/// SeqCst mirrors the load mapping (libcu++'s): `fence.sc.{scope}` followed
/// by a release store at the same scope, fused into one template.
fn ptx_store_template(ord: &NvvmOrdering, scope: &str, ptx_ty: &str) -> Result<String> {
    match ord {
        NvvmOrdering::Relaxed => Ok(format!("st.relaxed.{scope}.{ptx_ty} [$0], $1;")),
        NvvmOrdering::Release => Ok(format!("st.release.{scope}.{ptx_ty} [$0], $1;")),
        NvvmOrdering::SeqCst => Ok(format!(
            "fence.sc.{scope}; st.release.{scope}.{ptx_ty} [$0], $1;"
        )),
        other => pliron::input_err_noloc!(
            "atomic store cannot have {:?} ordering; use Relaxed, Release or SeqCst",
            other
        ),
    }
}

/// Emit a memory fence.
///
/// libNVVM rejects the LLVM `fence` instruction outright:
///
/// ```text
/// context:   fence syncscope("block") release
///   Illegal instruction: fence
/// ```
///
/// so every AcqRel or SeqCst atomic was unbuildable under `--materialize-cubin`,
/// including the ones in the shipped `atomics` example.
///
/// Two routes, chosen by what the ordering actually needs.
///
/// **SeqCst goes through the typed NVVM intrinsic.** PTX defines `membar.level`
/// as a synonym for `fence.sc.level`, so `llvm.nvvm.membar.{cta,gl,sys}` is an
/// exact match, and it is the route the rest of this crate already uses for
/// fences: `cuda_device::fence::threadfence` is documented as lowering to
/// `llvm.nvvm.membar.gl`. Going through the intrinsic keeps the fence
/// something LLVM can reason about rather than opaque assembly.
///
/// **Acquire, Release and AcqRel go through inline PTX**, because no intrinsic
/// for them exists. The catalog carries only the three `membar` scopes plus the
/// special-purpose `fence.proxy` and `fence.mbarrier_init` forms. Emitting
/// `membar` for an AcqRel fence would be correct but would silently upgrade the
/// caller's request to sequential consistency, so `fence.acq_rel.{scope}` is
/// emitted directly instead. PTX has no separate acquire or release fence;
/// `fence.acq_rel` is the primitive both lower to.
fn emit_fence(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    ordering: LlvmAtomicOrdering,
    syncscope: LlvmSyncScope,
) -> Result<()> {
    let scope = match syncscope {
        LlvmSyncScope::Device => "gl",
        LlvmSyncScope::Block => "cta",
        LlvmSyncScope::System => "sys",
    };

    if matches!(ordering, LlvmAtomicOrdering::SeqCst) {
        let void_ty = llvm_types::VoidType::get(ctx);
        let func_ty = llvm_types::FuncType::get(ctx, void_ty.into(), vec![], false);
        common::call_intrinsic(
            ctx,
            rewriter,
            op,
            &format!("llvm_nvvm_membar_{scope}"),
            func_ty,
            vec![],
        )?;
        return Ok(());
    }

    // Acquire, Release and AcqRel all lower to the same PTX fence.
    let ptx_scope = match syncscope {
        LlvmSyncScope::Device => "gpu",
        LlvmSyncScope::Block => "cta",
        LlvmSyncScope::System => "sys",
    };
    let void_ty = llvm_types::VoidType::get(ctx);
    let asm = llvm::InlineAsmOp::build(
        ctx,
        void_ty.into(),
        vec![],
        &format!("fence.acq_rel.{ptx_scope};"),
        "~{memory}",
        AsmKind::SideEffect,
    );
    rewriter.insert_operation(ctx, asm.get_operation());
    Ok(())
}

// =============================================================================
// Fence
// =============================================================================

pub(crate) fn convert_atomic_fence(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let fence = NvvmAtomicFenceOp::new(op);
    let ordering = fence.ordering(ctx);
    if matches!(ordering, NvvmOrdering::Relaxed) {
        return pliron::input_err_noloc!("atomic fence cannot use Relaxed ordering");
    }

    emit_fence(
        ctx,
        rewriter,
        op,
        map_ordering(&ordering),
        map_scope(&fence.scope(ctx)),
    )?;
    rewriter.erase_operation(ctx, op);
    Ok(())
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
    let ordering = nvvm_op.ordering(ctx);
    let scope = ptx_scope(&nvvm_op.scope(ctx));

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let ptr = operands[0];
    let mir_result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let result_ty =
        convert_type(ctx, mir_result_ty).map_err(|e| pliron::input_error_noloc!("{}", e))?;

    let (ptx_ty, reg, staging_int_ty) = ptx_type_and_reg(ctx, result_ty)
        .ok_or_else(|| pliron::input_error_noloc!("atomic load of unsupported operand type"))?;
    let template = ptx_load_template(&ordering, scope, ptx_ty)?;

    // SideEffect plus an unconditional `~{memory}` clobber, including for
    // Relaxed. Without the clobber LLVM may move plain loads and stores of
    // the same address across the asm, breaking the single-thread coherence
    // Rust still guarantees for Relaxed atomics; libcu++ keeps the clobber
    // on its relaxed accesses for the same reason.
    //
    // Floats travel through the integer register class: the asm produces the
    // same-width integer and the value is bitcast back below.
    let asm_result_ty = staging_int_ty.unwrap_or(result_ty);
    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        asm_result_ty,
        vec![ptr],
        &template,
        &format!("={reg},l,~{{memory}}"),
        AsmKind::SideEffect,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);

    if staging_int_ty.is_some() {
        let asm_result = asm_op.deref(ctx).get_result(0);
        let bitcast = llvm::BitcastOp::new(ctx, asm_result, result_ty);
        rewriter.insert_operation(ctx, bitcast.get_operation());
        rewriter.replace_operation(ctx, op, bitcast.get_operation());
    } else {
        rewriter.replace_operation(ctx, op, asm_op);
    }

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
    let ordering = nvvm_op.ordering(ctx);
    let scope = ptx_scope(&nvvm_op.scope(ctx));

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let val = operands[0];
    let ptr = operands[1];

    let val_ty = val.get_type(ctx);
    let (ptx_ty, reg, staging_int_ty) = ptx_type_and_reg(ctx, val_ty)
        .ok_or_else(|| pliron::input_error_noloc!("atomic store of unsupported operand type"))?;
    let template = ptx_store_template(&ordering, scope, ptx_ty)?;

    // A float value is bitcast to the same-width integer first, so the asm
    // operand type matches the integer register class in the constraint.
    let val = match staging_int_ty {
        Some(int_ty) => {
            let bitcast = llvm::BitcastOp::new(ctx, val, int_ty);
            let bitcast_op = bitcast.get_operation();
            rewriter.insert_operation(ctx, bitcast_op);
            bitcast_op.deref(ctx).get_result(0)
        }
        None => val,
    };

    // No result: a store produces nothing. Operand order matches the template,
    // address first then value, which is the reverse of the NVVM op's operands.
    // The `~{memory}` clobber is unconditional, Relaxed included, for the same
    // single-thread coherence reason as the load above.
    let void_ty = llvm_types::VoidType::get(ctx);
    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        void_ty.into(),
        vec![ptr, val],
        &template,
        &format!("l,{reg},~{{memory}}"),
        AsmKind::SideEffect,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
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
            emit_fence(ctx, rewriter, op, LlvmAtomicOrdering::Release, syncscope)?;
        }
        NvvmOrdering::SeqCst => {
            emit_fence(ctx, rewriter, op, LlvmAtomicOrdering::SeqCst, syncscope)?;
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
            emit_fence(ctx, rewriter, op, LlvmAtomicOrdering::Acquire, syncscope)?;
        }
        NvvmOrdering::SeqCst => {
            emit_fence(ctx, rewriter, op, LlvmAtomicOrdering::SeqCst, syncscope)?;
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

// =============================================================================
// Packed Atomic Add (f16x2, bf16x2) -- inline PTX
// =============================================================================

/// Convert a packed atomic add op to inline PTX.
///
/// Constraints: `=r,l,r,~{memory}` -- output register, address pointer, input
/// register, memory clobber.
///
/// Uses `SideEffect` (not convergent): atomics are per-thread, not
/// warp-synchronous.
pub(crate) fn convert_packed_atom_add(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    ptx_type: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!(
            "packed atomic add requires 2 operands (address, addend), got {}",
            operands.len()
        );
    }
    let addr = operands[0];
    let val = operands[1];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        i32_ty.into(),
        vec![addr, val],
        &format!("atom.global.add.noftz.{ptx_type} $0, [$1], $2;"),
        "=r,l,r,~{memory}",
        AsmKind::SideEffect,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}
