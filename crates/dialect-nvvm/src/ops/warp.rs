/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level operations: shuffle, vote, and lane identification.
//!
//! A warp is a group of 32 threads that execute in lockstep. These operations
//! enable efficient intra-warp communication without shared memory.
//!
//! # Shuffle Operations
//!
//! Shuffle operations allow threads to exchange register values directly:
//!
//! ```text
//! ┌──────┬──────────────────────┬───────────────────────────────────┐
//! │ Mode │ PTX                  │ Description                       │
//! ├──────┼──────────────────────┼───────────────────────────────────┤
//! │ idx  │ shfl.sync.idx.b32    │ Read from specific lane           │
//! │ bfly │ shfl.sync.bfly.b32   │ XOR lane ID with mask (butterfly) │
//! │ down │ shfl.sync.down.b32   │ Read from lane + delta            │
//! │ up   │ shfl.sync.up.b32     │ Read from lane - delta            │
//! └──────┴──────────────────────┴───────────────────────────────────┘
//! ```
//!
//! # Vote Operations
//!
//! Vote operations perform warp-wide predicate evaluation:
//!
//! ```text
//! ┌─────────────┬──────────────────────────────────────────────────────┐
//! │ Operation   │ Returns                                              │
//! ├─────────────┼──────────────────────────────────────────────────────┤
//! │ vote.all    │ true if ALL active threads have predicate true       │
//! │ vote.any    │ true if ANY active thread has predicate true         │
//! │ vote.ballot │ 32-bit mask where bit[i] = thread i's predicate      │
//! └─────────────┴──────────────────────────────────────────────────────┘
//! ```
//!
//! # Operand convention — `mask` is always operand 0
//!
//! Every shuffle and vote op in this module takes the warp participation
//! mask (i32) as operand 0. The mask names the lanes that are guaranteed
//! to converge at the call site — bit `k` set means lane `k` participates.
//!
//! For full-warp ops, callers pass `0xFFFFFFFF` (`-1` as i32). For sub-warp
//! tiles or coalesced groups, the mask is computed at runtime or baked in
//! by a typed wrapper (`WarpTile<N>`, `CoalescedThreads`).

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    builtin::types::IntegerType,
    common_traits::Verify,
    context::Context,
    context::Ptr,
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    verify_err,
};
use pliron_derive::pliron_op;

/// Verify a special-register operation has one 32-bit integer result.
fn verify_lanemask_result(ctx: &Context, op: Ptr<Operation>, op_name: &str) -> Result<(), Error> {
    let op = &*op.deref(ctx);
    let res = op.get_result(0);
    let ty = res.get_type(ctx);

    let ty_obj = ty.deref(ctx);
    let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
        Some(ty) => ty,
        None => {
            return verify_err!(op.loc(), "{} result must be integer", op_name);
        }
    };

    if int_ty.width() != 32 {
        return verify_err!(op.loc(), "{} result must be 32-bit integer", op_name);
    }
    Ok(())
}

// =============================================================================
// Leader Election (sm_90+)
// =============================================================================

/// Warp leader election: choose the lowest participating lane as leader.
///
/// PTX `elect.sync d|p, membermask`. Requires sm_90+ (Hopper). Lowered to
/// convergent inline PTX (the `@llvm.nvvm.elect.sync` intrinsic has no NVPTX
/// instruction-selection pattern in current LLVM). The instruction yields a
/// lane id and a predicate; this op exposes both directly as two results, so
/// no field is discarded.
///
/// # Operands
///
/// - `mask` (i32): warp lane participation mask (`-1` = full warp)
///
/// # Results
///
/// - `leader` (i32): lane id of the elected leader (lowest lane in `mask`).
///   PTX only defines this value on the elected lane; it is unspecified on
///   non-elected lanes
/// - `is_elected` (i1): true only on the calling lane if it is the leader
#[pliron_op(
    name = "nvvm.elect_sync",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<2>],
)]
pub struct ElectSyncOp;

impl ElectSyncOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ElectSyncOp { op }
    }
}

// =============================================================================
// Hardware Warp Identification
// =============================================================================

/// Read the hardware warp ID within the current SM.
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.warpid` / PTX `%warpid`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_warpid",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregWarpIdOp;

impl ReadPtxSregWarpIdOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregWarpIdOp { op }
    }
}

impl Verify for ReadPtxSregWarpIdOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_lanemask_result(ctx, self.get_operation(), "nvvm.read_ptx_sreg_warpid")
    }
}

/// Read the maximum number of hardware warp slots per SM (max warp ID + 1).
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.nwarpid` / PTX `%nwarpid`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_nwarpid",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregNwarpIdOp;

impl ReadPtxSregNwarpIdOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregNwarpIdOp { op }
    }
}

impl Verify for ReadPtxSregNwarpIdOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_lanemask_result(ctx, self.get_operation(), "nvvm.read_ptx_sreg_nwarpid")
    }
}

/// Register warp operations with the context.
pub(super) fn register(ctx: &mut Context) {
    // Leader election (sm_90+)
    ElectSyncOp::register(ctx);
    // Hardware warp identification
    ReadPtxSregWarpIdOp::register(ctx);
    ReadPtxSregNwarpIdOp::register(ctx);
}
