/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! WMMA (Warp Matrix Multiply-Accumulate) mma.sync operations for SM 80+ GPUs.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Warp MMA: m16n8k16 with f32 accumulator and bf16 inputs.
///
/// # Operands
///
/// - `acc_ptr` (ptr): pointer to `[f32; 4]` accumulator (read-modify-write)
/// - `a_ptr` (ptr): pointer to `[u32; 4]` A fragment (packed bf16)
/// - `b_ptr` (ptr): pointer to `[u32; 2]` B fragment (packed bf16)
///
/// # Results
///
/// - None (accumulator updated in-place via pointer)
#[pliron_op(
    name = "nvvm.mma_m16n8k16_f32_bf16",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct MmaM16N8K16F32Bf16Op;

impl MmaM16N8K16F32Bf16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K16F32Bf16Op { op }
    }
}

/// Register WMMA operations with the context.
pub(super) fn register(ctx: &mut Context) {
    MmaM16N8K16F32Bf16Op::register(ctx);
}
