/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix operations (movmatrix).
//!
//! Provides in-register matrix transpose for warp-distributed 8×8 tiles
//! of 16-bit elements.
//!
//! ```text
//! ┌───────────────────────┬──────────┬─────────┬───────────────────────────────────────┐
//! │ Operation             │ Operands │ Results │ PTX                                   │
//! ├───────────────────────┼──────────┼─────────┼───────────────────────────────────────┤
//! │ MovmatrixTransB16Op   │ 1 (u32)  │ 1 (u32) │ movmatrix.sync.aligned.m8n8.trans.b16 │
//! └───────────────────────┴──────────┴─────────┴───────────────────────────────────────┘
//! ```
//!
//! # Requirements
//!
//! - **Execution**: Warp-synchronous (all 32 threads must participate)
//! - **Architecture**: sm_90+ (Hopper and later)

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// In-register transpose of an 8×8 matrix of b16 elements across the warp.
///
/// Each lane provides a single `u32` register containing two packed b16 values.
/// The instruction collectively transposes the 8×8 tile distributed across the
/// warp and writes the transposed pair into each lane's result register.
///
/// PTX: `movmatrix.sync.aligned.m8n8.trans.b16 %d, %a;`
///
/// # Operands
///
/// - `a` (i32): source register containing 2 packed b16 values
///
/// # Results
///
/// - `d` (i32): destination register with the transposed packed b16 pair
#[pliron_op(
    name = "nvvm.movmatrix_trans_b16",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct MovmatrixTransB16Op;

impl MovmatrixTransB16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MovmatrixTransB16Op { op }
    }
}

/// Register wmma operations with the context.
pub(super) fn register(ctx: &mut Context) {
    MovmatrixTransB16Op::register(ctx);
}
