/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level shared memory matrix load (`ldmatrix`) operations for Ampere+ GPUs.
//!
//! Provides warp-level `ldmatrix.sync` PTX instructions for loading packed 8×8
//! matrices from shared memory into registers, available on SM_75+.
//!
//! # Operations
//!
//! | Operation            | PTX                                    | Description                     |
//! |----------------------|----------------------------------------|---------------------------------|
//! | `LdmatrixX4`         | `ldmatrix.sync.aligned.m8n8.x4`       | Load 4×8×8 from SMEM            |
//! | `LdmatrixX2`         | `ldmatrix.sync.aligned.m8n8.x2`       | Load 2×8×8 from SMEM            |
//! | `LdmatrixX4Trans`    | `ldmatrix.sync.aligned.m8n8.x4.trans`  | Load 4×8×8 transposed           |
//! | `LdmatrixX2Trans`    | `ldmatrix.sync.aligned.m8n8.x2.trans`  | Load 2×8×8 transposed           |

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

// =============================================================================
// ldmatrix Operations
// =============================================================================

/// Load 4 packed 8×8 matrices from shared memory.
///
/// PTX: `ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0,r1,r2,r3}, [addr];`
///
/// # Operands
/// - `smem_ptr` (ptr): pointer to shared memory
///
/// # Results
/// - `result` (ptr): pointer to output array [u32; 4]
#[pliron_op(
    name = "nvvm.ldmatrix_x4",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct LdmatrixX4Op;

impl LdmatrixX4Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        LdmatrixX4Op { op }
    }
}

/// Load 2 packed 8×8 matrices from shared memory.
///
/// PTX: `ldmatrix.sync.aligned.m8n8.x2.shared.b16 {r0,r1}, [addr];`
///
/// # Operands
/// - `smem_ptr` (ptr): pointer to shared memory
///
/// # Results
/// - `result` (ptr): pointer to output array [u32; 2]
#[pliron_op(
    name = "nvvm.ldmatrix_x2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct LdmatrixX2Op;

impl LdmatrixX2Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        LdmatrixX2Op { op }
    }
}

/// Load 4 packed 8×8 matrices from shared memory with transpose.
///
/// PTX: `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {r0,r1,r2,r3}, [addr];`
#[pliron_op(
    name = "nvvm.ldmatrix_x4_trans",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct LdmatrixX4TransOp;

impl LdmatrixX4TransOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        LdmatrixX4TransOp { op }
    }
}

/// Load 2 packed 8×8 matrices from shared memory with transpose.
///
/// PTX: `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {r0,r1}, [addr];`
#[pliron_op(
    name = "nvvm.ldmatrix_x2_trans",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct LdmatrixX2TransOp;

impl LdmatrixX2TransOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        LdmatrixX2TransOp { op }
    }
}

/// Register ldmatrix operations with the context.
pub(super) fn register(ctx: &mut Context) {
    LdmatrixX4Op::register(ctx);
    LdmatrixX2Op::register(ctx);
    LdmatrixX4TransOp::register(ctx);
    LdmatrixX2TransOp::register(ctx);
}
