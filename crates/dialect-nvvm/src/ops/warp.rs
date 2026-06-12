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

// =============================================================================
// Lane Identification
// =============================================================================

/// Read the lane ID within the warp (0-31).
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.laneid` / PTX `%laneid`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_laneid",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregLaneIdOp;

impl ReadPtxSregLaneIdOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregLaneIdOp { op }
    }
}

impl Verify for ReadPtxSregLaneIdOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(op.loc(), "nvvm.read_ptx_sreg_laneid result must be integer");
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_laneid result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

// =============================================================================
// Warp Shuffle - Integer (i32)
// =============================================================================

/// Warp shuffle: read from a specific lane (idx mode) for i32.
///
/// Corresponds to `llvm.nvvm.shfl.sync.idx.i32`.
///
/// # Operands
///
/// - `mask` (i32): warp lane participation mask (`-1` = full warp)
/// - `value` (i32): the value to share
/// - `src_lane` (i32): the lane index to read from (0-31)
///
/// # Results
///
/// - `result` (i32): the value from the source lane
#[pliron_op(
    name = "nvvm.shfl_sync_idx_i32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct ShflSyncIdxI32Op;

impl ShflSyncIdxI32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ShflSyncIdxI32Op { op }
    }
}

/// Warp shuffle: butterfly (XOR) pattern for i32.
///
/// Reads from lane `(lane_id XOR lane_mask)`. This pattern is commonly used
/// for parallel reductions (e.g., XOR with 16, 8, 4, 2, 1 for warp-wide sum).
///
/// Corresponds to `llvm.nvvm.shfl.sync.bfly.i32`.
///
/// # Operands
///
/// - `mask` (i32): warp lane participation mask (`-1` = full warp)
/// - `value` (i32): the value to exchange
/// - `lane_mask` (i32): XOR mask for lane calculation
///
/// # Results
///
/// - `result` (i32): the value from lane `(self XOR mask)`
#[pliron_op(
    name = "nvvm.shfl_sync_bfly_i32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct ShflSyncBflyI32Op;

impl ShflSyncBflyI32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ShflSyncBflyI32Op { op }
    }
}

/// Warp shuffle: read from higher lane (down mode) for i32.
///
/// Reads from lane `(lane_id + delta)`. Values from out-of-range lanes are undefined.
///
/// Corresponds to `llvm.nvvm.shfl.sync.down.i32`.
///
/// # Operands
///
/// - `mask` (i32): warp lane participation mask (`-1` = full warp)
/// - `value` (i32): the value to share
/// - `delta` (i32): offset to add to lane ID
///
/// # Results
///
/// - `result` (i32): the value from lane `(self + delta)`
#[pliron_op(
    name = "nvvm.shfl_sync_down_i32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct ShflSyncDownI32Op;

impl ShflSyncDownI32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ShflSyncDownI32Op { op }
    }
}

/// Warp shuffle: read from lower lane (up mode) for i32.
///
/// Reads from lane `(lane_id - delta)`. Values from negative lanes are undefined.
///
/// Corresponds to `llvm.nvvm.shfl.sync.up.i32`.
///
/// # Operands
///
/// - `mask` (i32): warp lane participation mask (`-1` = full warp)
/// - `value` (i32): the value to share
/// - `delta` (i32): offset to subtract from lane ID
///
/// # Results
///
/// - `result` (i32): the value from lane `(self - delta)`
#[pliron_op(
    name = "nvvm.shfl_sync_up_i32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct ShflSyncUpI32Op;

impl ShflSyncUpI32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ShflSyncUpI32Op { op }
    }
}

// =============================================================================
// Warp Shuffle - Float (f32)
// =============================================================================

/// Warp shuffle: read from a specific lane (idx mode) for f32.
///
/// Corresponds to `llvm.nvvm.shfl.sync.idx.f32`.
///
/// # Operands
///
/// - `mask` (i32): warp lane participation mask (`-1` = full warp)
/// - `value` (f32): the value to share
/// - `src_lane` (i32): the lane index to read from (0-31)
///
/// # Results
///
/// - `result` (f32): the value from the source lane
#[pliron_op(
    name = "nvvm.shfl_sync_idx_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct ShflSyncIdxF32Op;

impl ShflSyncIdxF32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ShflSyncIdxF32Op { op }
    }
}

/// Warp shuffle: butterfly (XOR) pattern for f32.
///
/// Corresponds to `llvm.nvvm.shfl.sync.bfly.f32`.
///
/// # Operands
///
/// - `mask` (i32): warp lane participation mask (`-1` = full warp)
/// - `value` (f32): the value to exchange
/// - `lane_mask` (i32): XOR mask for lane calculation
///
/// # Results
///
/// - `result` (f32): the value from lane `(self XOR mask)`
#[pliron_op(
    name = "nvvm.shfl_sync_bfly_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct ShflSyncBflyF32Op;

impl ShflSyncBflyF32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ShflSyncBflyF32Op { op }
    }
}

/// Warp shuffle: read from higher lane (down mode) for f32.
///
/// Corresponds to `llvm.nvvm.shfl.sync.down.f32`.
///
/// # Operands
///
/// - `mask` (i32): warp lane participation mask (`-1` = full warp)
/// - `value` (f32): the value to share
/// - `delta` (i32): offset to add to lane ID
///
/// # Results
///
/// - `result` (f32): the value from lane `(self + delta)`
#[pliron_op(
    name = "nvvm.shfl_sync_down_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct ShflSyncDownF32Op;

impl ShflSyncDownF32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ShflSyncDownF32Op { op }
    }
}

/// Warp shuffle: read from lower lane (up mode) for f32.
///
/// Corresponds to `llvm.nvvm.shfl.sync.up.f32`.
///
/// # Operands
///
/// - `mask` (i32): warp lane participation mask (`-1` = full warp)
/// - `value` (f32): the value to share
/// - `delta` (i32): offset to subtract from lane ID
///
/// # Results
///
/// - `result` (f32): the value from lane `(self - delta)`
#[pliron_op(
    name = "nvvm.shfl_sync_up_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct ShflSyncUpF32Op;

impl ShflSyncUpF32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ShflSyncUpF32Op { op }
    }
}

// =============================================================================
// Warp Vote Operations
// =============================================================================

/// Warp vote: returns true if ALL active threads have predicate true.
///
/// Corresponds to `llvm.nvvm.vote.sync.all`.
///
/// # Operands
///
/// - `mask` (i32): warp lane participation mask (`-1` = full warp)
/// - `predicate` (i1): the condition to check
///
/// # Results
///
/// - `result` (i1): true if all active threads have predicate true
#[pliron_op(
    name = "nvvm.vote_sync_all",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct VoteSyncAllOp;

impl VoteSyncAllOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        VoteSyncAllOp { op }
    }
}

/// Warp vote: returns true if ANY active thread has predicate true.
///
/// Corresponds to `llvm.nvvm.vote.sync.any`.
///
/// # Operands
///
/// - `mask` (i32): warp lane participation mask (`-1` = full warp)
/// - `predicate` (i1): the condition to check
///
/// # Results
///
/// - `result` (i1): true if any active thread has predicate true
#[pliron_op(
    name = "nvvm.vote_sync_any",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct VoteSyncAnyOp;

impl VoteSyncAnyOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        VoteSyncAnyOp { op }
    }
}

// =============================================================================
// Active Mask
// =============================================================================

/// Read the active-thread mask of the current warp.
///
/// Corresponds to `llvm.nvvm.activemask` / PTX `activemask.b32` (PTX 6.2+).
/// The result is a 32-bit bitmask: bit `k` is set iff lane `k` is currently
/// converged with this thread (i.e. participating in this dynamic execution
/// region). For full-warp execution this is `0xFFFFFFFF`; in divergent code
/// it shrinks to whatever subset of lanes reached this point together.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.activemask",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ActiveMaskOp;

impl ActiveMaskOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ActiveMaskOp { op }
    }
}

impl Verify for ActiveMaskOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(op.loc(), "nvvm.activemask result must be integer");
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(op.loc(), "nvvm.activemask result must be 32-bit integer");
        }
        Ok(())
    }
}

// =============================================================================
// Warp-scoped barrier (sub-warp synchronization)
// =============================================================================

/// Synchronize a subset of warp lanes given by `mask`.
///
/// Corresponds to `llvm.nvvm.bar.warp.sync` / PTX `bar.warp.sync`. Acts as
/// a convergence point for every lane bit set in `mask`: each such lane
/// must reach this op with the same `mask` value before any of them
/// proceeds. Lanes whose bit is clear are not affected and need not
/// reach the call.
///
/// This is the primitive that backs `CoalescedThreads::sync()` and the
/// `WarpTile<N>::sync()` method on sub-warp tiles. Callers who already
/// know the lanes are converged in lockstep (e.g. straight-line warp-
/// uniform code) do not need this — but its presence forces the SIMT
/// reconvergence model on Volta+ targets and is required after a
/// divergent branch before any other `*.sync` collective.
///
/// # Operands
///
/// - `mask` (i32): warp lane participation mask (`-1` = full warp)
///
/// # Results
///
/// - none
#[pliron_op(
    name = "nvvm.bar_warp_sync",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<0>],
)]
pub struct BarWarpSyncOp;

impl BarWarpSyncOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        BarWarpSyncOp { op }
    }
}

// =============================================================================
// Warp Match Operations (sm_70+)
// =============================================================================

/// Warp match-any (32-bit value): bitmask of lanes whose value equals mine.
///
/// Corresponds to `llvm.nvvm.match.any.sync.i32` / PTX `match.any.sync.b32`.
/// Requires sm_70+.
///
/// # Operands
///
/// - `mask` (i32): warp lane participation mask (`-1` = full warp)
/// - `value` (i32): value broadcast-and-compared across the mask
///
/// # Results
///
/// - `result` (i32): bitmask of lanes (within `mask`) whose `value` equals this lane's
#[pliron_op(
    name = "nvvm.match_any_sync_i32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct MatchAnySyncI32Op;

impl MatchAnySyncI32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MatchAnySyncI32Op { op }
    }
}

/// Warp match-any (64-bit value): 64-bit variant of [`MatchAnySyncI32Op`].
///
/// Corresponds to `llvm.nvvm.match.any.sync.i64` / PTX `match.any.sync.b64`.
///
/// # Operands
///
/// - `mask` (i32): warp lane participation mask (`-1` = full warp)
/// - `value` (i64): value broadcast-and-compared across the mask
///
/// # Results
///
/// - `result` (i32): bitmask of lanes (within `mask`) whose `value` equals this lane's
#[pliron_op(
    name = "nvvm.match_any_sync_i64",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct MatchAnySyncI64Op;

impl MatchAnySyncI64Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MatchAnySyncI64Op { op }
    }
}

/// Warp match-all (32-bit value): full participating mask if every lane agrees, else 0.
///
/// Corresponds to `llvm.nvvm.match.all.sync.i32p` / PTX `match.all.sync.b32`.
/// The LLVM intrinsic returns `{i32, i1}`; the lowering extracts field 0
/// (the matching mask). Callers can recover the predicate as `result != 0`.
/// Requires sm_70+.
///
/// # Operands
///
/// - `mask` (i32): warp lane participation mask (`-1` = full warp)
/// - `value` (i32): value broadcast-and-compared across the mask
///
/// # Results
///
/// - `result` (i32): `mask` if every participating lane has the same `value`, else 0
#[pliron_op(
    name = "nvvm.match_all_sync_i32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct MatchAllSyncI32Op;

impl MatchAllSyncI32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MatchAllSyncI32Op { op }
    }
}

/// Warp match-all (64-bit value): 64-bit variant of [`MatchAllSyncI32Op`].
///
/// Corresponds to `llvm.nvvm.match.all.sync.i64p` / PTX `match.all.sync.b64`.
///
/// # Operands
///
/// - `mask` (i32): warp lane participation mask (`-1` = full warp)
/// - `value` (i64): value broadcast-and-compared across the mask
///
/// # Results
///
/// - `result` (i32): `mask` if every participating lane has the same `value`, else 0
#[pliron_op(
    name = "nvvm.match_all_sync_i64",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct MatchAllSyncI64Op;

impl MatchAllSyncI64Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MatchAllSyncI64Op { op }
    }
}

/// Warp ballot: returns a 32-bit mask where `bit[i]` indicates thread i's predicate.
///
/// Corresponds to `llvm.nvvm.vote.sync.ballot`.
///
/// # Operands
///
/// - `mask` (i32): warp lane participation mask (`-1` = full warp)
/// - `predicate` (i1): the condition to check
///
/// # Results
///
/// - `result` (i32): bitmask where bit `i` is set if thread `i` has predicate true
#[pliron_op(
    name = "nvvm.vote_sync_ballot",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct VoteSyncBallotOp;

impl VoteSyncBallotOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        VoteSyncBallotOp { op }
    }
}

/// Register warp operations with the context.
pub(super) fn register(ctx: &mut Context) {
    // Lane identification
    ReadPtxSregLaneIdOp::register(ctx);
    // Shuffle - i32
    ShflSyncIdxI32Op::register(ctx);
    ShflSyncBflyI32Op::register(ctx);
    ShflSyncDownI32Op::register(ctx);
    ShflSyncUpI32Op::register(ctx);
    // Shuffle - f32
    ShflSyncIdxF32Op::register(ctx);
    ShflSyncBflyF32Op::register(ctx);
    ShflSyncDownF32Op::register(ctx);
    ShflSyncUpF32Op::register(ctx);
    // Vote
    VoteSyncAllOp::register(ctx);
    VoteSyncAnyOp::register(ctx);
    VoteSyncBallotOp::register(ctx);
    // Match
    MatchAnySyncI32Op::register(ctx);
    MatchAnySyncI64Op::register(ctx);
    MatchAllSyncI32Op::register(ctx);
    MatchAllSyncI64Op::register(ctx);
    // Active mask
    ActiveMaskOp::register(ctx);
    // Warp-scoped barrier
    BarWarpSyncOp::register(ctx);
}
