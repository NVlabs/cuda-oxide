/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level mma.sync operations (SM80+).

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
/// PTX: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {d0..d3},{a0..a3},{b0,b1},{c0..c3};`
///
/// # Operands
///
/// - `a0..a3` (4 x i32): packed bf16x2 register pairs for matrix A
/// - `b0..b1` (2 x i32): packed bf16x2 register pairs for matrix B
/// - `c0..c3` (4 x f32): accumulator inputs
///
/// # Results
///
/// - `d0..d3` (4 x f32): accumulator outputs (D = A * B + C)
#[pliron_op(
    name = "nvvm.mma_m16n8k16_bf16_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<10>, NResultsInterface<4>],
)]
pub struct MmaM16n8k16Bf16F32Op;

impl MmaM16n8k16Bf16F32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16n8k16Bf16F32Op { op }
    }
}

/// Register MMA operations with the context.
pub(super) fn register(ctx: &mut Context) {
    MmaM16n8k16Bf16F32Op::register(ctx);
}
