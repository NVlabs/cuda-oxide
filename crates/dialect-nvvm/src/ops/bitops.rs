/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bit-manipulation PTX operations.
//!
//! | Operation  | PTX Instruction | Description                      |
//! |------------|-----------------|----------------------------------|
//! | `PrmtB32`  | `prmt.b32`      | Byte permute on two 32-bit words |
//!
//! # Requirements
//!
//! - **PTX ISA**: 2.0+
//! - **Architecture**: sm_20+ (all modern GPUs)

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Byte permute: rearrange bytes from two 32-bit words.
///
/// Selects four bytes from the concatenation of `a` (high) and `b` (low)
/// according to the control word `c`, producing a new 32-bit value.
///
/// This is a pure per-thread operation (not convergent).
///
/// PTX: `prmt.b32 $0, $1, $2, $3;`
///
/// # Operands
///
/// - `a` (i32): upper source word
/// - `b` (i32): lower source word
/// - `c` (i32): control word (byte selectors)
///
/// # Results
///
/// - `result` (i32): permuted output
#[pliron_op(
    name = "nvvm.prmt_b32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct PrmtB32Op;

impl PrmtB32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        PrmtB32Op { op }
    }
}

/// Register bitops operations with the context.
pub(super) fn register(ctx: &mut Context) {
    PrmtB32Op::register(ctx);
}
