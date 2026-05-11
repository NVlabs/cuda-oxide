/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warpgroup Matrix Multiply-Accumulate (WGMMA) operations for Hopper+ GPUs.
//!
//! WGMMA provides tensor core operations that operate at the warpgroup level
//! (4 warps = 128 threads) for high-throughput matrix multiplication.
//!
//! # WGMMA Workflow
//!
//! ```text
//! 1. wgmma.fence       │ Ensure shared memory is visible
//! 2. wgmma.mma         │ Issue matrix multiply (may issue multiple)
//! 3. wgmma.commit      │ Commit pending operations to group
//! 4. wgmma.wait        │ Wait for group completion
//! ```
//!
//! # Matrix Dimensions
//!
//! The `m64n64k16` variant computes:
//! - D = A × B + C where A is 64×16, B is 16×64, D/C is 64×64
//! - Each thread holds 32 f32 accumulator elements
//!
//! # Shared Memory Descriptors
//!
//! WGMMA uses 64-bit descriptors that encode:
//! - Base address (in shared memory address space)
//! - Stride information
//! - Swizzle mode for bank conflict avoidance
//!
//! # Requirements
//!
//! - **PTX ISA**: 8.0+
//! - **Architecture**: sm_90+ (Hopper)
//! - **Execution**: Warpgroup-synchronous (128 threads must execute together)

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

// =============================================================================
// Synchronization Operations
// =============================================================================

/// WGMMA fence for memory synchronization before MMA operations.
///
/// Ensures that shared memory writes are visible to the tensor core unit
/// before issuing WGMMA operations.
///
/// Corresponds to `llvm.nvvm.wgmma.fence.sync.aligned`.
///
/// PTX: `wgmma.fence.sync.aligned;`
///
/// # Operands
///
/// - None
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.wgmma_fence_sync_aligned",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct WgmmaFenceSyncAlignedOp;

impl WgmmaFenceSyncAlignedOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// WGMMA commit group - commits pending WGMMA operations.
///
/// Groups pending WGMMA operations for collective waiting.
///
/// Corresponds to `llvm.nvvm.wgmma.commit_group.sync.aligned`.
///
/// PTX: `wgmma.commit_group.sync.aligned;`
///
/// # Operands
///
/// - None
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.wgmma_commit_group_sync_aligned",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct WgmmaCommitGroupSyncAlignedOp;

impl WgmmaCommitGroupSyncAlignedOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// WGMMA wait group - waits for N groups to complete.
///
/// Blocks until at most N WGMMA groups are still pending.
///
/// Corresponds to `llvm.nvvm.wgmma.wait_group.sync.aligned`.
///
/// PTX: `wgmma.wait_group.sync.aligned N;`
///
/// # Operands
///
/// - `N` (i64): maximum pending groups (0-7)
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.wgmma_wait_group_sync_aligned",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<0>],
)]
pub struct WgmmaWaitGroupSyncAlignedOp;

impl WgmmaWaitGroupSyncAlignedOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

// =============================================================================
// Descriptor Operations
// =============================================================================

/// Create a shared memory descriptor for WGMMA.
///
/// Converts a generic pointer to shared memory into a 64-bit WGMMA descriptor
/// that encodes base address, stride information, and swizzle mode.
///
/// Uses inline PTX: `cvta.shared.u32` + bit manipulation
///
/// # Operands
///
/// - `ptr` (ptr): generic pointer to shared memory
///
/// # Results
///
/// - `desc` (i64): 64-bit WGMMA descriptor
#[pliron_op(
    name = "nvvm.wgmma_make_smem_desc",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct WgmmaMakeSmemDescOp;

impl WgmmaMakeSmemDescOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

// =============================================================================
// Matrix Multiply-Accumulate Operations
// =============================================================================

/// WGMMA Matrix Multiply-Accumulate: m64n64k16 with f32 accumulator and bf16 inputs.
///
/// Performs warpgroup-level matrix multiplication: D = A × B + D
/// - A: 64×16 (bf16)
/// - B: 16×64 (bf16)
/// - D: 64×64 (f32, 32 elements per thread, updated in-place)
///
/// PTX: `wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16`
///
/// # Operands
///
/// - `acc_ptr` (ptr): pointer to accumulator array (32 f32 values, read-modify-write)
/// - `desc_a` (i64): SMEM descriptor for matrix A
/// - `desc_b` (i64): SMEM descriptor for matrix B
///
/// # Results
///
/// - None (accumulator is updated in-place via pointer)
#[pliron_op(
    name = "nvvm.wgmma_mma_m64n64k16_f32_bf16",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct WgmmaMmaM64N64K16F32Bf16Op;

impl WgmmaMmaM64N64K16F32Bf16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Register WGMMA operations with the context.
pub(super) fn register(ctx: &mut Context) {
    WgmmaFenceSyncAlignedOp::register(ctx);
    WgmmaCommitGroupSyncAlignedOp::register(ctx);
    WgmmaWaitGroupSyncAlignedOp::register(ctx);
    WgmmaMakeSmemDescOp::register(ctx);
    WgmmaMmaM64N64K16F32Bf16Op::register(ctx);
}
