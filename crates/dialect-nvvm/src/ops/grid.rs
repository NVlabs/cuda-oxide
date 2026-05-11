/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Grid-scoped NVVM operations.
//!
//! The grid is the union of every block (and cluster) in a single kernel
//! launch. There is no PTX instruction that synchronises the grid directly:
//! `bar.sync 0` is block-scoped and `barrier.cluster.{arrive,wait}` is
//! cluster-scoped. Grid-wide barriers are implemented in software (see the
//! lowering for [`GridSyncOp`]) and require the kernel to be launched
//! cooperatively (`cuLaunchKernelEx` + `CU_LAUNCH_ATTRIBUTE_COOPERATIVE`).

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

// =============================================================================
// Grid Sync
// =============================================================================

/// Grid-wide barrier.
///
/// Lowered to an inline-PTX sequence that uses an internal-linkage 32-bit
/// counter `__cuda_oxide_grid_sync_counter` in global memory:
///
/// ```text
/// bar.sync 0;                                       // intra-block barrier
/// if (linear_tid == 0) {                            // block leader path
///     nb = nctaid.x * nctaid.y * nctaid.z;
///     my_idx = atom.add.release.gpu [counter], 1;
///     target = my_idx - (my_idx % nb) + nb;
///     while (ld.acquire.gpu [counter] < target) {}
/// }
/// bar.sync 0;                                       // release block
/// ```
///
/// The host's cooperative launcher is responsible for resetting the counter
/// to zero before each launch so the modular target arithmetic remains valid
/// across grids of different sizes.
///
/// # Cooperative launch required
///
/// `cuLaunchKernelEx` must be called with `CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 1`.
/// CUDA enforces that every block of a cooperative kernel is co-resident on
/// the GPU, which is the necessary condition for the busy-wait spin to make
/// progress.
#[pliron_op(
    name = "nvvm.grid_sync",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct GridSyncOp;

impl GridSyncOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Register all grid-scoped operations.
pub fn register(ctx: &mut Context) {
    GridSyncOp::register(ctx);
}
