/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Extended mbarrier and async-proxy operations for Hopper+ GPUs.
//!
//! Basic Ampere operations are generated in `ops::generated`.
//!
//! Memory barriers (mbarriers) are hardware-accelerated synchronization primitives
//! that enable efficient coordination between threads, particularly for async
//! copy operations like TMA.
//!
//! # Barrier Lifecycle
//!
//! ```text
//! ┌───────────────────────┬────────────────────────────────────────────────────┐
//! │ 1. mbarrier.init      │ Initialize with expected arrival count             │
//! │ 2. fence.proxy.async  │ Make init visible to async proxy (TMA)             │
//! │ 3. mbarrier.arrive    │ Signal arrival (with optional tx bytes)            │
//! │ 4. mbarrier.try_wait  │ Poll for phase completion                          │
//! │ 5. mbarrier.inval     │ Invalidate when done                               │
//! └───────────────────────┴────────────────────────────────────────────────────┘
//! ```
//!
//! # TMA Integration
//!
//! When used with Tensor Memory Accelerator (TMA) operations:
//! - `arrive_expect_tx` declares expected transaction bytes
//! - TMA hardware automatically signals completion when bytes transfer
//! - `try_wait` polls for completion without blocking
//!
//! # Requirements
//!
//! - **PTX ISA**: 8.0+
//! - **Architecture**: sm_90+ (Hopper and newer)
//! - **Memory**: Barrier must reside in shared memory (addrspace 3)

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

// =============================================================================
// Arrival Operations
// =============================================================================

/// Arrive at an mbarrier with expected transaction byte count.
///
/// Used with TMA's `complete_tx::bytes` mode to track asynchronous data transfer
/// completion. The barrier will complete when the expected bytes have been transferred.
///
/// **Note**: This instruction does NOT have an LLVM intrinsic and requires inline PTX.
///
/// PTX: `mbarrier.arrive.expect_tx.shared.b64`
///
/// # Operands
///
/// - `bar_ptr` (ptr addrspace(3)): pointer to barrier in shared memory
/// - `bytes` (i32): expected transaction byte count
///
/// # Results
///
/// - `token` (i64): phase token for wait operations
#[pliron_op(
    name = "nvvm.mbarrier_arrive_expect_tx_shared",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct MbarrierArriveExpectTxSharedOp;

impl MbarrierArriveExpectTxSharedOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MbarrierArriveExpectTxSharedOp { op }
    }
}

/// Arrive at a CTA-shared mbarrier with cluster-scope expected transaction bytes.
///
/// PTX: `mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64`
///
/// # Operands
///
/// - `bar_ptr` (ptr addrspace(3)): pointer to barrier in CTA shared memory
/// - `bytes` (i32): expected transaction byte count
///
/// # Results
///
/// - `token` (i64): phase token for wait operations
#[pliron_op(
    name = "nvvm.mbarrier_arrive_expect_tx_cluster",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct MbarrierArriveExpectTxClusterOp;

impl MbarrierArriveExpectTxClusterOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MbarrierArriveExpectTxClusterOp { op }
    }
}

/// Arrive at a barrier in another CTA's shared memory via cluster addressing.
///
/// Used for cross-CTA synchronization in TMA multicast patterns. Each CTA's
/// MMA warp arrives at rank 0's consumer barrier to signal buffer consumption.
///
/// Takes a raw u64 address (from mapa) to avoid address-space issues in phi nodes.
///
/// PTX: `mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$addr];`
///
/// # Operands
///
/// - `addr` (i64): cluster-scope shared memory address from mapa
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.mbarrier_arrive_cluster",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<0>],
)]
pub struct MbarrierArriveClusterOp;

impl MbarrierArriveClusterOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MbarrierArriveClusterOp { op }
    }
}

// =============================================================================
// Wait Operations
// =============================================================================

/// Try to wait for an mbarrier phase to complete (with scheduling hints).
///
/// Similar to `test_wait` but provides better scheduling hints to the hardware.
/// This is the **preferred instruction for TMA synchronization** as it matches
/// nvcc's generated code and allows the GPU to efficiently schedule other work.
///
/// **Note**: This instruction does NOT have an LLVM intrinsic and requires inline PTX.
///
/// PTX: `mbarrier.try_wait.shared.b64`
///
/// # Operands
///
/// - `bar_ptr` (ptr addrspace(3)): pointer to barrier in shared memory
/// - `token` (i64): phase token from arrive operation
///
/// # Results
///
/// - `complete` (i1): true if the phase is complete
#[pliron_op(
    name = "nvvm.mbarrier_try_wait_shared",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct MbarrierTryWaitSharedOp;

impl MbarrierTryWaitSharedOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MbarrierTryWaitSharedOp { op }
    }
}

/// Try wait with parity check (shared::cta variant).
///
/// Waits based on phase parity (0 or 1) rather than a specific token.
/// Useful for double-buffering patterns.
///
/// PTX: `mbarrier.try_wait.parity.shared::cta.b64 pred, [addr], parity;`
///
/// # Operands
///
/// - `bar_ptr` (ptr addrspace(3)): pointer to barrier in shared memory
/// - `parity` (i32): expected parity (0 or 1)
///
/// # Results
///
/// - `complete` (i1): true if the phase with given parity is complete
#[pliron_op(
    name = "nvvm.mbarrier_try_wait_parity_shared",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct MbarrierTryWaitParitySharedOp;

impl MbarrierTryWaitParitySharedOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MbarrierTryWaitParitySharedOp { op }
    }
}

/// Try wait with parity and cluster-scope acquire semantics.
///
/// PTX: `mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64`
///
/// # Operands
///
/// - `bar_ptr` (ptr addrspace(3)): pointer to barrier in CTA shared memory
/// - `parity` (i32): expected parity (0 or 1)
///
/// # Results
///
/// - `complete` (i1): true if the phase with given parity is complete
#[pliron_op(
    name = "nvvm.mbarrier_try_wait_parity_cluster",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct MbarrierTryWaitParityClusterOp;

impl MbarrierTryWaitParityClusterOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MbarrierTryWaitParityClusterOp { op }
    }
}

// =============================================================================
// Memory Fence Operations
// =============================================================================

/// Fence to synchronize generic proxy with async proxy in shared memory.
///
/// This fence ensures that memory operations performed through the generic
/// proxy (normal thread operations like `mbarrier.init`) are visible to the
/// async proxy (hardware async operations like TMA `cp.async.bulk`).
///
/// **Critical for TMA**: Must be called after `mbarrier_init` and before
/// issuing TMA operations.
///
/// # PTX
///
/// ```ptx
/// fence.proxy.async.shared::cta;
/// ```
///
/// # Requirements
///
/// - PTX ISA 8.0+
/// - sm_90+ (Hopper and newer)
///
/// # Operands
///
/// - None
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.fence_proxy_async_shared_cta",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct FenceProxyAsyncSharedCtaOp;

impl FenceProxyAsyncSharedCtaOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        FenceProxyAsyncSharedCtaOp { op }
    }
}

/// Release prior mbarrier initialization at cluster scope.
///
/// PTX: `fence.mbarrier_init.release.cluster;`
#[pliron_op(
    name = "nvvm.fence_mbarrier_init_release_cluster",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct FenceMbarrierInitReleaseClusterOp;

impl FenceMbarrierInitReleaseClusterOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        FenceMbarrierInitReleaseClusterOp { op }
    }
}

/// Release generic-proxy writes to the async proxy for CTA shared memory at cluster scope.
///
/// PTX: `fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;`
#[pliron_op(
    name = "nvvm.fence_proxy_async_generic_release_shared_cta_cluster",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct FenceProxyAsyncGenericReleaseSharedCtaClusterOp;

impl FenceProxyAsyncGenericReleaseSharedCtaClusterOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        FenceProxyAsyncGenericReleaseSharedCtaClusterOp { op }
    }
}

/// Acquire async-proxy writes through the generic proxy for cluster shared memory.
///
/// PTX: `fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;`
#[pliron_op(
    name = "nvvm.fence_proxy_async_generic_acquire_shared_cluster_cluster",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct FenceProxyAsyncGenericAcquireSharedClusterClusterOp;

impl FenceProxyAsyncGenericAcquireSharedClusterClusterOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        FenceProxyAsyncGenericAcquireSharedClusterClusterOp { op }
    }
}

// =============================================================================
// Thread Scheduling Hints
// =============================================================================

/// Suspend thread for a specified number of nanoseconds.
///
/// Used in spin-wait loops to reduce interconnect contention and allow
/// pending cluster-scope operations to complete.
///
/// PTX: `nanosleep.u32 N;`
///
/// # Operands
///
/// - `ns` (i32): approximate nanoseconds to sleep
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.nanosleep",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<0>],
)]
pub struct NanosleepOp;

impl NanosleepOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        NanosleepOp { op }
    }
}

/// Register mbarrier operations with the context.
pub(super) fn register(ctx: &mut Context) {
    MbarrierArriveExpectTxSharedOp::register(ctx);
    MbarrierArriveExpectTxClusterOp::register(ctx);
    MbarrierArriveClusterOp::register(ctx);
    MbarrierTryWaitSharedOp::register(ctx);
    MbarrierTryWaitParitySharedOp::register(ctx);
    MbarrierTryWaitParityClusterOp::register(ctx);
    FenceProxyAsyncSharedCtaOp::register(ctx);
    FenceMbarrierInitReleaseClusterOp::register(ctx);
    FenceProxyAsyncGenericReleaseSharedCtaClusterOp::register(ctx);
    FenceProxyAsyncGenericAcquireSharedClusterClusterOp::register(ctx);
    NanosleepOp::register(ctx);
}
