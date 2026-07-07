/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tensor Core Gen 5 (tcgen05) operations for Blackwell+ GPUs.
//!
//! Tcgen05 introduces Tensor Memory (TMEM), a dedicated on-chip memory for tensor
//! core operations that provides higher bandwidth than shared memory.
//!
//! # Key Concepts
//!
//! ## Tensor Memory (TMEM)
//!
//! - Dedicated memory for tensor core operands and results
//! - Must be explicitly allocated/deallocated
//! - Addressed via 32-bit handles (not pointers)
//! - Format: `(row_offset << 16) | column_address`
//!
//! ## Execution Model
//!
//! - **Warp-synchronous**: Allocation/deallocation (all 32 threads)
//! - **Single-thread**: MMA and fence operations (thread 0 typically)
//!
//! # Operation Categories
//!
//! ```text
//! ┌─────────────────┬─────────────────────────────────────────────────────────┐
//! │ Category        │ Operations                                              │
//! ├─────────────────┼─────────────────────────────────────────────────────────┤
//! │ Allocation      │ Tcgen05Alloc, Tcgen05Dealloc, RelinquishAllocPermit     │
//! │ Fencing         │ Tcgen05FenceBefore/AfterThreadSync                      │
//! │ Commit/Wait     │ Tcgen05Commit, Tcgen05CommitSharedCluster               │
//! │ Descriptors     │ Tcgen05MakeSmemDesc, Tcgen05MakeSmemDescStrided         │
//! │ MMA             │ Tcgen05MmaWsF16, Tcgen05MmaF16, Tcgen05MmaWsBf16, ...   │
//! │ Data Movement   │ Tcgen05CpSmemToTmem, Tcgen05StTmemToSmem                │
//! │ TMEM Load       │ Tcgen05Ld* variants (16x256b, 32x32b layouts)           │
//! │ Synchronization │ Tcgen05LoadWait, Tcgen05StoreWait                       │
//! └─────────────────┴─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Requirements
//!
//! - **PTX ISA**: 8.6+
//! - **Architecture**: sm_100+ (Blackwell)
//! - **Critical**: All TMEM must be deallocated before kernel exits!

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

// =============================================================================
// TMEM Allocation
// =============================================================================

/// Allocate Tensor Memory (TMEM) for Blackwell tensor cores.
///
/// **WARP-SYNCHRONOUS**: All 32 threads in the warp must execute this instruction.
///
/// PTX: `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [dst], n_cols;`
///
/// # Operands
///
/// - `dst_smem` (ptr addrspace(3)): shared memory location to store TMEM address
/// - `n_cols` (i32): number of columns (32, 64, 128, 256, or 512)
///
/// # Results
///
/// - None (TMEM address written to dst_smem)
#[pliron_op(
    name = "nvvm.tcgen05_alloc",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct Tcgen05AllocOp;

impl Tcgen05AllocOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05AllocOp { op }
    }
}

/// Deallocate Tensor Memory (TMEM).
///
/// **WARP-SYNCHRONOUS**: All 32 threads in the warp must execute this instruction.
///
/// **CRITICAL**: All TMEM must be deallocated before kernel exits!
///
/// PTX: `tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, n_cols;`
///
/// # Operands
///
/// - `tmem_addr` (i32): TMEM address from `tcgen05_alloc`
/// - `n_cols` (i32): number of columns (must match allocation)
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.tcgen05_dealloc",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct Tcgen05DeallocOp;

impl Tcgen05DeallocOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05DeallocOp { op }
    }
}

/// Relinquish TMEM allocation permit.
///
/// **WARP-SYNCHRONOUS**. After this call, no more TMEM allocations are allowed
/// from this CTA.
///
/// PTX: `tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;`
///
/// # Operands
///
/// - None
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.tcgen05_relinquish_alloc_permit",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct Tcgen05RelinquishAllocPermitOp;

impl Tcgen05RelinquishAllocPermitOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05RelinquishAllocPermitOp { op }
    }
}

// =============================================================================
// Fence Operations
// =============================================================================

/// Fence for ordering BEFORE thread synchronization.
///
/// **SINGLE-THREAD** semantics.
///
/// PTX: `tcgen05.fence::before_thread_sync;`
///
/// # Operands
///
/// - None
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.tcgen05_fence_before_thread_sync",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct Tcgen05FenceBeforeThreadSyncOp;

impl Tcgen05FenceBeforeThreadSyncOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05FenceBeforeThreadSyncOp { op }
    }
}

/// Fence for ordering AFTER thread synchronization.
///
/// **SINGLE-THREAD** semantics.
///
/// PTX: `tcgen05.fence::after_thread_sync;`
///
/// # Operands
///
/// - None
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.tcgen05_fence_after_thread_sync",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct Tcgen05FenceAfterThreadSyncOp;

impl Tcgen05FenceAfterThreadSyncOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05FenceAfterThreadSyncOp { op }
    }
}

// =============================================================================
// Commit Operations
// =============================================================================

/// Commit pending tcgen05 operations to mbarrier.
///
/// **SINGLE-THREAD** semantics.
///
/// PTX: `tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mbar];`
///
/// # Operands
///
/// - `mbar` (ptr addrspace(3)): pointer to mbarrier in shared memory
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.tcgen05_commit",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<0>],
)]
pub struct Tcgen05CommitOp;

impl Tcgen05CommitOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05CommitOp { op }
    }
}

/// Commit pending tcgen05 operations (shared::cluster variant).
///
/// PTX: `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [mbar];`
///
/// # Operands
///
/// - `mbar` (ptr addrspace(3)): pointer to mbarrier in shared memory
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.tcgen05_commit_shared_cluster",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<0>],
)]
pub struct Tcgen05CommitSharedClusterOp;

impl Tcgen05CommitSharedClusterOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05CommitSharedClusterOp { op }
    }
}

// NOTE: Tcgen05MakeSmemDescOp and Tcgen05MakeSmemDescStridedOp were removed.
// Use Tcgen05SmemDescriptor::builder() in cuda-device instead.

// =============================================================================
// MMA Operations
// =============================================================================

/// tcgen05 MMA with f16 inputs (warp-specialized form).
///
/// **SINGLE-THREAD** semantics: ONE thread issues the entire MMA!
///
/// PTX: `tcgen05.mma.ws.cta_group::1.kind::f16 [d-tmem], [a-tmem], a-desc, b-desc, idesc, enable-d;`
///
/// # Operands
///
/// - `d_tmem` (i32): TMEM address for D matrix (output)
/// - `a_tmem` (i32): TMEM address for A matrix (input)
/// - `a_desc` (i64): SMEM descriptor for A
/// - `b_desc` (i64): SMEM descriptor for B
/// - `idesc` (i32): instruction descriptor
/// - `enable_d` (i1): whether to accumulate (true) or overwrite (false)
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.tcgen05_mma_ws_f16",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<6>, NResultsInterface<0>],
)]
pub struct Tcgen05MmaWsF16Op;

impl Tcgen05MmaWsF16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05MmaWsF16Op { op }
    }
}

/// tcgen05 MMA with f16 inputs (non-warp-specialized form).
///
/// **SINGLE-THREAD** semantics: ONE thread issues the entire MMA.
///
/// PTX: `tcgen05.mma.cta_group::1.kind::f16 [d-tmem], a-desc, b-desc, idesc, {0,0,0,0}, enable-d;`
///
/// # Operands
///
/// - `d_tmem` (i32): TMEM address for D matrix
/// - `a_desc` (i64): SMEM descriptor for A
/// - `b_desc` (i64): SMEM descriptor for B
/// - `idesc` (i32): instruction descriptor
/// - `enable_d` (i1): whether to accumulate
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.tcgen05_mma_f16",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<5>, NResultsInterface<0>],
)]
pub struct Tcgen05MmaF16Op;

impl Tcgen05MmaF16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05MmaF16Op { op }
    }
}

/// tcgen05 MMA with bf16 inputs.
#[pliron_op(
    name = "nvvm.tcgen05_mma_ws_bf16",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<6>, NResultsInterface<0>],
)]
pub struct Tcgen05MmaWsBf16Op;

impl Tcgen05MmaWsBf16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05MmaWsBf16Op { op }
    }
}

/// tcgen05 MMA with tf32 inputs.
#[pliron_op(
    name = "nvvm.tcgen05_mma_ws_tf32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<6>, NResultsInterface<0>],
)]
pub struct Tcgen05MmaWsTf32Op;

impl Tcgen05MmaWsTf32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05MmaWsTf32Op { op }
    }
}

/// tcgen05 MMA with FP8 E4M3 inputs.
#[pliron_op(
    name = "nvvm.tcgen05_mma_ws_e4m3",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<6>, NResultsInterface<0>],
)]
pub struct Tcgen05MmaWsE4M3Op;

impl Tcgen05MmaWsE4M3Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05MmaWsE4M3Op { op }
    }
}

/// tcgen05 MMA with FP8 E5M2 inputs.
#[pliron_op(
    name = "nvvm.tcgen05_mma_ws_e5m2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<6>, NResultsInterface<0>],
)]
pub struct Tcgen05MmaWsE5M2Op;

impl Tcgen05MmaWsE5M2Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05MmaWsE5M2Op { op }
    }
}

/// tcgen05 MMA with FP6 E2M3 inputs.
#[pliron_op(
    name = "nvvm.tcgen05_mma_ws_e2m3",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<6>, NResultsInterface<0>],
)]
pub struct Tcgen05MmaWsE2M3Op;

impl Tcgen05MmaWsE2M3Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05MmaWsE2M3Op { op }
    }
}

/// tcgen05 MMA with FP6 E3M2 inputs.
#[pliron_op(
    name = "nvvm.tcgen05_mma_ws_e3m2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<6>, NResultsInterface<0>],
)]
pub struct Tcgen05MmaWsE3M2Op;

impl Tcgen05MmaWsE3M2Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05MmaWsE3M2Op { op }
    }
}

/// tcgen05 MMA with FP4 E2M1 inputs.
#[pliron_op(
    name = "nvvm.tcgen05_mma_ws_e2m1",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<6>, NResultsInterface<0>],
)]
pub struct Tcgen05MmaWsE2M1Op;

impl Tcgen05MmaWsE2M1Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05MmaWsE2M1Op { op }
    }
}

// =============================================================================
// Data Movement Operations
// =============================================================================

/// Copy from shared memory to tensor memory.
///
/// Loads a tile of data from SMEM into TMEM for MMA operations.
///
/// PTX: `tcgen05.cp.cta_group::1.128x256b [tmem_addr], smem_desc;`
///
/// # Operands
///
/// - `tmem_addr` (i32): destination address in tensor memory
/// - `smem_desc` (i64): source shared memory descriptor
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.tcgen05_cp_smem_to_tmem",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct Tcgen05CpSmemToTmemOp;

impl Tcgen05CpSmemToTmemOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05CpSmemToTmemOp { op }
    }
}

// NOTE: The following deprecated ops were removed:
// - Tcgen05StTmemToSmemOp (wrong - no direct TMEM→SMEM PTX instruction)
// - Tcgen05StTmemToSmemOffsetOp (wrong)
// - Tcgen05Ld16x256bX4Op/X8Op/X16Op/X32Op (wrong - stored to SMEM instead of returning registers)
// - Tcgen05Ld32x32bX64Op (wrong)
//
// Use the Pure variants below instead.

// =============================================================================
// Pure TMEM Load Operations (return registers, no SMEM store)
// =============================================================================

/// Pure TMEM load: returns 32 f32 values in registers (no SMEM store).
///
/// The **correct** way to load from TMEM for epilog processing.
/// Unlike the pointer variants, this returns values as results for subsequent
/// in-register operations (convert, pack, stmatrix).
///
/// PTX: `tcgen05.ld.sync.aligned.16x256b.x8.b32 {r0..r31}, [tmem_addr];`
///
/// # Operands
///
/// - `tmem_addr` (i32): TMEM address
///
/// # Results
///
/// - 32 × f32 values (r0, r1, ..., r31)
#[pliron_op(
    name = "nvvm.tcgen05_ld_16x256b_x8_pure",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<32>],
)]
pub struct Tcgen05Ld16x256bX8PureOp;

impl Tcgen05Ld16x256bX8PureOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05Ld16x256bX8PureOp { op }
    }
}

/// Pure TMEM load: returns 4 f32 values in registers.
///
/// Base LDTM.16dp256bit instruction without .x8 multiplier.
/// Returns 4 f32 values per thread for use with stmatrix.m8n8.x2.
///
/// PTX: `tcgen05.ld.sync.aligned.16x256b.b32 {r0, r1, r2, r3}, [tmem_addr];`
///
/// # Operands
///
/// - `tmem_addr` (i32): TMEM address
///
/// # Results
///
/// - 4 × f32 values (r0, r1, r2, r3)
#[pliron_op(
    name = "nvvm.tcgen05_ld_16x256b_pure",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<4>],
)]
pub struct Tcgen05Ld16x256bPureOp;

impl Tcgen05Ld16x256bPureOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05Ld16x256bPureOp { op }
    }
}

// =============================================================================
// Synchronization Operations
// =============================================================================

/// Wait for tcgen05 tensor memory loads to complete.
///
/// **Critical**: Must be called after tcgen05.ld instructions before accessing data.
///
/// PTX: `tcgen05.wait::ld.sync.aligned;`
///
/// # Operands
///
/// - None
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.tcgen05_load_wait",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct Tcgen05LoadWaitOp;

impl Tcgen05LoadWaitOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05LoadWaitOp { op }
    }
}

/// Wait for tcgen05 tensor memory stores to complete.
///
/// PTX: `tcgen05.wait::st.sync.aligned;`
///
/// # Operands
///
/// - None
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.tcgen05_store_wait",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct Tcgen05StoreWaitOp;

impl Tcgen05StoreWaitOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05StoreWaitOp { op }
    }
}

// =============================================================================
// CTA Pair (cta_group::2) Variants
// =============================================================================

/// Allocate TMEM for a CTA pair (`cta_group::2`).
///
/// PTX: `tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [dst], n_cols;`
///
/// # Operands
///
/// - `dst_smem` (ptr addrspace(3))
/// - `n_cols` (i32)
#[pliron_op(
    name = "nvvm.tcgen05_alloc_cg2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct Tcgen05AllocCg2Op;

impl Tcgen05AllocCg2Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05AllocCg2Op { op }
    }
}

/// Deallocate TMEM for a CTA pair (`cta_group::2`).
///
/// PTX: `tcgen05.dealloc.cta_group::2.sync.aligned.b32 tmem_addr, n_cols;`
///
/// # Operands
///
/// - `tmem_addr` (i32)
/// - `n_cols` (i32)
#[pliron_op(
    name = "nvvm.tcgen05_dealloc_cg2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct Tcgen05DeallocCg2Op;

impl Tcgen05DeallocCg2Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05DeallocCg2Op { op }
    }
}

/// Relinquish TMEM allocation permit for a CTA pair (`cta_group::2`).
///
/// PTX: `tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;`
#[pliron_op(
    name = "nvvm.tcgen05_relinquish_alloc_permit_cg2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct Tcgen05RelinquishAllocPermitCg2Op;

impl Tcgen05RelinquishAllocPermitCg2Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05RelinquishAllocPermitCg2Op { op }
    }
}

/// tcgen05 MMA with f16 inputs for a CTA pair (`cta_group::2`).
///
/// PTX: `tcgen05.mma.cta_group::2.kind::f16 [d], a_desc, b_desc, idesc, {0,0,0,0}, enable_d;`
///
/// # Operands
///
/// - `d_tmem` (i32), `a_desc` (i64), `b_desc` (i64), `idesc` (i32), `enable_d` (i1)
#[pliron_op(
    name = "nvvm.tcgen05_mma_f16_cg2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<5>, NResultsInterface<0>],
)]
pub struct Tcgen05MmaF16Cg2Op;

impl Tcgen05MmaF16Cg2Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05MmaF16Cg2Op { op }
    }
}

/// Commit CTA-pair tcgen05 operations to mbarrier (`cta_group::2`).
///
/// PTX: `tcgen05.commit.cta_group::2.mbarrier::arrive::one.b64 [mbar];`
///
/// # Operands
///
/// - `mbar` (ptr addrspace(3))
#[pliron_op(
    name = "nvvm.tcgen05_commit_cg2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<0>],
)]
pub struct Tcgen05CommitCg2Op;

impl Tcgen05CommitCg2Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05CommitCg2Op { op }
    }
}

/// Commit CTA-pair tcgen05 operations via `.shared::cluster` (`cta_group::2`).
///
/// PTX: `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [mbar];`
///
/// # Operands
///
/// - `mbar` (ptr addrspace(3))
#[pliron_op(
    name = "nvvm.tcgen05_commit_shared_cluster_cg2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<0>],
)]
pub struct Tcgen05CommitSharedClusterCg2Op;

impl Tcgen05CommitSharedClusterCg2Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05CommitSharedClusterCg2Op { op }
    }
}

/// Commit CTA-pair tcgen05 operations with multicast (`cta_group::2`).
///
/// Signals the mbarrier in every CTA whose bit is set in `cta_mask`.
/// Used after cooperative MMA to signal both CTAs' barriers in one instruction.
///
/// PTX: `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [mbar], ctaMask;`
///
/// # Operands
///
/// - `mbar` (ptr addrspace(3))
/// - `cta_mask` (i16)
#[pliron_op(
    name = "nvvm.tcgen05_commit_multicast_cg2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct Tcgen05CommitMulticastCg2Op;

impl Tcgen05CommitMulticastCg2Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05CommitMulticastCg2Op { op }
    }
}

/// Copy SMEM to TMEM for a CTA pair (`cta_group::2`).
///
/// PTX: `tcgen05.cp.cta_group::2.128x256b [tmem_addr], smem_desc;`
///
/// # Operands
///
/// - `tmem_addr` (i32)
/// - `smem_desc` (i64)
#[pliron_op(
    name = "nvvm.tcgen05_cp_smem_to_tmem_cg2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct Tcgen05CpSmemToTmemCg2Op;

impl Tcgen05CpSmemToTmemCg2Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        Tcgen05CpSmemToTmemCg2Op { op }
    }
}

/// Register tcgen05 operations with the context.
pub(super) fn register(ctx: &mut Context) {
    // Allocation
    Tcgen05AllocOp::register(ctx);
    Tcgen05DeallocOp::register(ctx);
    Tcgen05RelinquishAllocPermitOp::register(ctx);
    // Fencing
    Tcgen05FenceBeforeThreadSyncOp::register(ctx);
    Tcgen05FenceAfterThreadSyncOp::register(ctx);
    // Commit
    Tcgen05CommitOp::register(ctx);
    Tcgen05CommitSharedClusterOp::register(ctx);
    // NOTE: Tcgen05MakeSmemDescOp and Tcgen05MakeSmemDescStridedOp removed
    // Use Tcgen05SmemDescriptor::builder() in cuda-device instead
    // MMA
    Tcgen05MmaWsF16Op::register(ctx);
    Tcgen05MmaF16Op::register(ctx);
    Tcgen05MmaWsBf16Op::register(ctx);
    Tcgen05MmaWsTf32Op::register(ctx);
    Tcgen05MmaWsE4M3Op::register(ctx);
    Tcgen05MmaWsE5M2Op::register(ctx);
    Tcgen05MmaWsE2M3Op::register(ctx);
    Tcgen05MmaWsE3M2Op::register(ctx);
    Tcgen05MmaWsE2M1Op::register(ctx);
    // Data movement (SMEM→TMEM only; TMEM→SMEM removed - use Pure load + stmatrix)
    Tcgen05CpSmemToTmemOp::register(ctx);
    // NOTE: StTmemToSmem ops removed - incorrect approach
    // NOTE: Ld16x256bX4/X8/X16/X32Op and Ld32x32bX64Op removed - wrong design
    // Pure TMEM load (correct approach - returns registers)
    Tcgen05Ld16x256bX8PureOp::register(ctx);
    Tcgen05Ld16x256bPureOp::register(ctx);
    // Synchronization
    Tcgen05LoadWaitOp::register(ctx);
    Tcgen05StoreWaitOp::register(ctx);
    // CTA pair (cta_group::2) variants
    Tcgen05AllocCg2Op::register(ctx);
    Tcgen05DeallocCg2Op::register(ctx);
    Tcgen05RelinquishAllocPermitCg2Op::register(ctx);
    Tcgen05MmaF16Cg2Op::register(ctx);
    Tcgen05CommitCg2Op::register(ctx);
    Tcgen05CommitSharedClusterCg2Op::register(ctx);
    Tcgen05CommitMulticastCg2Op::register(ctx);
    Tcgen05CpSmemToTmemCg2Op::register(ctx);
}
