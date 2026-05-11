/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tensor Memory Accelerator (TMA) operations for Hopper+ GPUs.
//!
//! TMA provides hardware-accelerated bulk data movement between global and shared
//! memory with support for multi-dimensional tensor addressing.
//!
//! # Operation Types
//!
//! ```text
//! ┌─────────────────┬──────────────────────────────────────────────────────────┐
//! │ Direction       │ Operations                                               │
//! ├─────────────────┼──────────────────────────────────────────────────────────┤
//! │ Global→Shared   │ CpAsyncBulkTensorG2sTile{1d,2d,3d,4d,5d}Op               │
//! │ G2S + multicast │ CpAsyncBulkTensorG2sTile2dMulticastOp                     │
//! │ Shared→Global   │ CpAsyncBulkTensorS2gTile{1d,2d,3d,4d,5d}Op               │
//! │ Synchronization │ CpAsyncBulkCommitGroupOp, CpAsyncBulk{Wait,WaitRead}Op   │
//! └─────────────────┴──────────────────────────────────────────────────────────┘
//! ```
//!
//! # TMA Workflow
//!
//! ```text
//! 1. Host: Create TMA descriptor (cuTensorMapCreate)
//! 2. Device: Initialize mbarrier (mbarrier.init)
//! 3. Device: Issue fence (fence.proxy.async)
//! 4. Device: Arrive with expected bytes (mbarrier.arrive.expect_tx)
//! 5. Device: Issue TMA load (cp.async.bulk.tensor.g2s)
//! 6. Device: Commit group (cp.async.bulk.commit_group)
//! 7. Device: Wait for completion (mbarrier.try_wait)
//! ```
//!
//! # Requirements
//!
//! - **PTX ISA**: 8.0+
//! - **Architecture**: sm_90+ (Hopper and newer)

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

// =============================================================================
// Global-to-Shared (G2S) Operations
// =============================================================================

/// Async 1D tensor copy from global to shared memory via TMA.
///
/// Corresponds to `llvm.nvvm.cp.async.bulk.tensor.g2s.tile.1d`.
///
/// # Operands
///
/// - `dst` (ptr addrspace(3)): destination in shared memory
/// - `barrier` (ptr addrspace(3)): mbarrier for completion tracking
/// - `tensor_map` (ptr): TMA descriptor created by host
/// - `coord0` (i32): coordinate along dimension 0
/// - `cta_mask` (i16): CTA mask (typically 0)
/// - `cache_hint` (i64): cache hint (typically 0)
///
/// # Results
///
/// - None (completion signaled via mbarrier)
#[pliron_op(
    name = "nvvm.cp_async_bulk_tensor_g2s_tile_1d",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<6>, NResultsInterface<0>],
)]
pub struct CpAsyncBulkTensorG2sTile1dOp;

impl CpAsyncBulkTensorG2sTile1dOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Async 2D tensor copy from global to shared memory via TMA.
///
/// Corresponds to `llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d`.
///
/// # Operands
///
/// - `dst` (ptr addrspace(3)): destination in shared memory
/// - `barrier` (ptr addrspace(3)): mbarrier for completion tracking
/// - `tensor_map` (ptr): TMA descriptor
/// - `coord0` (i32): coordinate along dimension 0
/// - `coord1` (i32): coordinate along dimension 1
/// - `cta_mask` (i16): CTA mask
/// - `cache_hint` (i64): cache hint
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_bulk_tensor_g2s_tile_2d",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<7>, NResultsInterface<0>],
)]
pub struct CpAsyncBulkTensorG2sTile2dOp;

impl CpAsyncBulkTensorG2sTile2dOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Async 2D tensor copy from global to shared memory via TMA **with multicast**.
///
/// Same LLVM intrinsic as [`CpAsyncBulkTensorG2sTile2dOp`] but the lowering
/// sets `use_cta_mask = true` so the NVPTX backend emits the
/// `multicast::cluster` qualifier.
///
/// # Operands
///
/// - `dst` (ptr addrspace(3)): destination in shared memory
/// - `barrier` (ptr addrspace(3)): mbarrier for completion tracking
/// - `tensor_map` (ptr): TMA descriptor
/// - `coord0` (i32): coordinate along dimension 0
/// - `coord1` (i32): coordinate along dimension 1
/// - `cta_mask` (i16): bitmask of destination CTAs in the cluster
/// - `cache_hint` (i64): cache hint (typically 0)
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_bulk_tensor_g2s_tile_2d_multicast",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<7>, NResultsInterface<0>],
)]
pub struct CpAsyncBulkTensorG2sTile2dMulticastOp;

impl CpAsyncBulkTensorG2sTile2dMulticastOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Async 2D tensor copy from global to shared memory via TMA **with multicast
/// and cta_group::2** (TPC pair).
///
/// Lowered to inline PTX (no LLVM intrinsic exists for cta_group::2):
/// ```ptx
/// cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global
///     .mbarrier::complete_tx::bytes.multicast::cluster
///     [dst], [tensorMap, {coord0, coord1}], [mbar], ctaMask;
/// ```
///
/// # Operands
///
/// - `dst` (ptr addrspace(3)): destination in shared memory
/// - `barrier` (ptr addrspace(3)): mbarrier for completion tracking
/// - `tensor_map` (ptr): TMA descriptor
/// - `coord0` (i32): coordinate along dimension 0
/// - `coord1` (i32): coordinate along dimension 1
/// - `cta_mask` (i16): bitmask of destination CTAs in the cluster
/// - `cache_hint` (i64): cache hint (unused, for consistency)
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_bulk_tensor_g2s_tile_2d_multicast_cg2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<7>, NResultsInterface<0>],
)]
pub struct CpAsyncBulkTensorG2sTile2dMulticastCg2Op;

impl CpAsyncBulkTensorG2sTile2dMulticastCg2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Async 3D tensor copy from global to shared memory via TMA.
///
/// Corresponds to `llvm.nvvm.cp.async.bulk.tensor.g2s.tile.3d`.
///
/// # Operands
///
/// - `dst` (ptr addrspace(3)): destination in shared memory
/// - `barrier` (ptr addrspace(3)): mbarrier for completion tracking
/// - `tensor_map` (ptr): TMA descriptor
/// - `coord0` (i32): coordinate along dimension 0
/// - `coord1` (i32): coordinate along dimension 1
/// - `coord2` (i32): coordinate along dimension 2
/// - `cta_mask` (i16): CTA mask
/// - `cache_hint` (i64): cache hint
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_bulk_tensor_g2s_tile_3d",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<8>, NResultsInterface<0>],
)]
pub struct CpAsyncBulkTensorG2sTile3dOp;

impl CpAsyncBulkTensorG2sTile3dOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Async 4D tensor copy from global to shared memory via TMA.
///
/// Corresponds to `llvm.nvvm.cp.async.bulk.tensor.g2s.tile.4d`.
#[pliron_op(
    name = "nvvm.cp_async_bulk_tensor_g2s_tile_4d",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<9>, NResultsInterface<0>],
)]
pub struct CpAsyncBulkTensorG2sTile4dOp;

impl CpAsyncBulkTensorG2sTile4dOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Async 5D tensor copy from global to shared memory via TMA.
///
/// Corresponds to `llvm.nvvm.cp.async.bulk.tensor.g2s.tile.5d`.
#[pliron_op(
    name = "nvvm.cp_async_bulk_tensor_g2s_tile_5d",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<10>, NResultsInterface<0>],
)]
pub struct CpAsyncBulkTensorG2sTile5dOp;

impl CpAsyncBulkTensorG2sTile5dOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

// =============================================================================
// Shared-to-Global (S2G) Operations
// =============================================================================

/// Async 1D tensor copy from shared to global memory via TMA.
///
/// Corresponds to `llvm.nvvm.cp.async.bulk.tensor.s2g.tile.1d`.
///
/// # Operands
///
/// - `src` (ptr addrspace(3)): source in shared memory
/// - `tensor_map` (ptr): TMA descriptor
/// - `coord0` (i32): coordinate along dimension 0
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_bulk_tensor_s2g_tile_1d",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct CpAsyncBulkTensorS2gTile1dOp;

impl CpAsyncBulkTensorS2gTile1dOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Async 2D tensor copy from shared to global memory via TMA.
///
/// Corresponds to `llvm.nvvm.cp.async.bulk.tensor.s2g.tile.2d`.
#[pliron_op(
    name = "nvvm.cp_async_bulk_tensor_s2g_tile_2d",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<4>, NResultsInterface<0>],
)]
pub struct CpAsyncBulkTensorS2gTile2dOp;

impl CpAsyncBulkTensorS2gTile2dOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Async 3D tensor copy from shared to global memory via TMA.
///
/// Corresponds to `llvm.nvvm.cp.async.bulk.tensor.s2g.tile.3d`.
#[pliron_op(
    name = "nvvm.cp_async_bulk_tensor_s2g_tile_3d",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<5>, NResultsInterface<0>],
)]
pub struct CpAsyncBulkTensorS2gTile3dOp;

impl CpAsyncBulkTensorS2gTile3dOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Async 4D tensor copy from shared to global memory via TMA.
///
/// Corresponds to `llvm.nvvm.cp.async.bulk.tensor.s2g.tile.4d`.
#[pliron_op(
    name = "nvvm.cp_async_bulk_tensor_s2g_tile_4d",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<6>, NResultsInterface<0>],
)]
pub struct CpAsyncBulkTensorS2gTile4dOp;

impl CpAsyncBulkTensorS2gTile4dOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Async 5D tensor copy from shared to global memory via TMA.
///
/// Corresponds to `llvm.nvvm.cp.async.bulk.tensor.s2g.tile.5d`.
#[pliron_op(
    name = "nvvm.cp_async_bulk_tensor_s2g_tile_5d",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<7>, NResultsInterface<0>],
)]
pub struct CpAsyncBulkTensorS2gTile5dOp;

impl CpAsyncBulkTensorS2gTile5dOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

// =============================================================================
// Async Group Operations
// =============================================================================

/// Commit pending async bulk operations to a group.
///
/// Groups pending TMA operations together for collective waiting.
/// Must be called after issuing TMA loads/stores.
///
/// Corresponds to `llvm.nvvm.cp.async.bulk.commit.group`.
///
/// # Operands
///
/// - None
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_bulk_commit_group",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct CpAsyncBulkCommitGroupOp;

impl CpAsyncBulkCommitGroupOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Wait for async bulk operation groups to complete.
///
/// Blocks until at most `n` groups are still pending. Use `n=0` to wait for all.
///
/// Corresponds to `llvm.nvvm.cp.async.bulk.wait.group`.
///
/// # Operands
///
/// - `n` (i32): maximum number of pending groups (0 = wait for all)
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_bulk_wait_group",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<0>],
)]
pub struct CpAsyncBulkWaitGroupOp;

impl CpAsyncBulkWaitGroupOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Wait for async bulk operation groups with read completion.
///
/// Similar to `wait_group` but ensures data is readable after the wait.
/// Use this before accessing the loaded data.
///
/// Corresponds to `llvm.nvvm.cp.async.bulk.wait.group.read`.
///
/// # Operands
///
/// - `n` (i32): maximum number of pending groups (0 = wait for all)
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_bulk_wait_group_read",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<0>],
)]
pub struct CpAsyncBulkWaitGroupReadOp;

impl CpAsyncBulkWaitGroupReadOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Register TMA operations with the context.
pub(super) fn register(ctx: &mut Context) {
    // G2S (Global to Shared)
    CpAsyncBulkTensorG2sTile1dOp::register(ctx);
    CpAsyncBulkTensorG2sTile2dOp::register(ctx);
    CpAsyncBulkTensorG2sTile2dMulticastOp::register(ctx);
    CpAsyncBulkTensorG2sTile2dMulticastCg2Op::register(ctx);
    CpAsyncBulkTensorG2sTile3dOp::register(ctx);
    CpAsyncBulkTensorG2sTile4dOp::register(ctx);
    CpAsyncBulkTensorG2sTile5dOp::register(ctx);
    // S2G (Shared to Global)
    CpAsyncBulkTensorS2gTile1dOp::register(ctx);
    CpAsyncBulkTensorS2gTile2dOp::register(ctx);
    CpAsyncBulkTensorS2gTile3dOp::register(ctx);
    CpAsyncBulkTensorS2gTile4dOp::register(ctx);
    CpAsyncBulkTensorS2gTile5dOp::register(ctx);
    // Group operations
    CpAsyncBulkCommitGroupOp::register(ctx);
    CpAsyncBulkWaitGroupOp::register(ctx);
    CpAsyncBulkWaitGroupReadOp::register(ctx);
}
