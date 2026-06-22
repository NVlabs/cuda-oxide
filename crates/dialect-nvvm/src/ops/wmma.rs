/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! WMMA (Warp Matrix Multiply-Accumulate) binary mma.sync operations for SM 80+ GPUs.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Warp MMA: m16n8k128 with s32 accumulator and b1 inputs (xor.popc).
///
/// # Operands
///
/// - `acc_ptr` (ptr): pointer to `[i32; 4]` accumulator (read-modify-write)
/// - `a_ptr` (ptr): pointer to `[u32; 2]` A fragment (packed b1)
/// - `b_ptr` (ptr): pointer to `u32` B fragment (packed b1)
///
/// # Results
///
/// - None (accumulator updated in-place via pointer)
#[pliron_op(
    name = "nvvm.mma_m16n8k128_s32_b1",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct MmaM16N8K128S32B1Op;

impl MmaM16N8K128S32B1Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K128S32B1Op { op }
    }
}

/// Warp MMA: m16n8k256 with s32 accumulator and b1 inputs (xor.popc).
///
/// # Operands
///
/// - `acc_ptr` (ptr): pointer to `[i32; 4]` accumulator (read-modify-write)
/// - `a_ptr` (ptr): pointer to `[u32; 4]` A fragment (packed b1)
/// - `b_ptr` (ptr): pointer to `[u32; 2]` B fragment (packed b1)
///
/// # Results
///
/// - None (accumulator updated in-place via pointer)
#[pliron_op(
    name = "nvvm.mma_m16n8k256_s32_b1",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct MmaM16N8K256S32B1Op;

impl MmaM16N8K256S32B1Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K256S32B1Op { op }
    }
}

/// Register WMMA operations with the context.
pub(super) fn register(ctx: &mut Context) {
    MmaM16N8K128S32B1Op::register(ctx);
    MmaM16N8K256S32B1Op::register(ctx);
}
