// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Packed `bf16x2` arithmetic operations.
//!
//! Single-thread, non-convergent packed bf16 ALU ops lowered to inline PTX.
//! Currently only FMA is exposed because `add.bf16x2` / `mul.bf16x2` require
//! `sm_90+`, while `fma.rn.bf16x2` is supported from `sm_80`.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Fused multiply-add on packed bf16x2 values: `d = a * b + c`.
///
/// Each operand is a `u32` carrying two bf16 lanes (low 16 / high 16). The
/// result is the packed pairwise FMA.
///
/// PTX: `fma.rn.bf16x2 $0, $1, $2, $3;`  (requires `sm_80+`)
///
/// # Operands
///
/// - `a` (u32): packed bf16x2 multiplicand
/// - `b` (u32): packed bf16x2 multiplier
/// - `c` (u32): packed bf16x2 addend
///
/// # Results
///
/// - `d` (u32): packed bf16x2 result
#[pliron_op(
    name = "nvvm.fma_bf16x2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct FmaBf16x2Op;

impl FmaBf16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        FmaBf16x2Op { op }
    }
}

/// Register bf16x2 operations with the context.
pub(super) fn register(ctx: &mut Context) {
    FmaBf16x2Op::register(ctx);
}
