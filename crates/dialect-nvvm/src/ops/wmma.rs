/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp Matrix Multiply-Accumulate (WMMA/mma.sync) operations for Ampere+ GPUs.
//!
//! Provides warp-level tensor core operations using `mma.sync` PTX instructions,
//! available on SM_80+ (Ampere, Ada Lovelace, Hopper, Blackwell consumer).
//!
//! # Operations
//!
//! | Operation            | PTX                                    | Description                     |
//! |----------------------|----------------------------------------|---------------------------------|
//! | `LdmatrixX4`         | `ldmatrix.sync.aligned.m8n8.x4`       | Load 4×8×8 from SMEM            |
//! | `LdmatrixX2`         | `ldmatrix.sync.aligned.m8n8.x2`       | Load 2×8×8 from SMEM            |
//! | `LdmatrixX4Trans`    | `ldmatrix.sync.aligned.m8n8.x4.trans`  | Load 4×8×8 transposed           |
//! | `LdmatrixX2Trans`    | `ldmatrix.sync.aligned.m8n8.x2.trans`  | Load 2×8×8 transposed           |
//! | `MmaM16N8K16F32F16`  | `mma.sync.aligned.m16n8k16.f32.f16`   | Matrix multiply-accumulate       |

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

// =============================================================================
// mma.sync Operations
// =============================================================================

/// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
///
/// Warp-synchronous matrix multiply-accumulate: D = A × B + C
/// - A: 16×16 (f16, row-major), 4 × u32 per thread
/// - B: 16×8 (f16, col-major), 2 × u32 per thread
/// - D/C: 16×8 (f32), 4 × f32 per thread
///
/// # Operands
///
/// - `acc_ptr` (ptr): pointer to [f32; 4] accumulator (read-modify-write)
/// - `a_ptr` (ptr): pointer to [u32; 4] A fragment
/// - `b_ptr` (ptr): pointer to [u32; 2] B fragment
///
/// # Results
///
/// - None (accumulator is updated in-place via pointer)
#[pliron_op(
    name = "nvvm.mma_m16n8k16_f32_f16",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct MmaM16N8K16F32F16Op;

impl MmaM16N8K16F32F16Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K16F32F16Op { op }
    }
}

/// Register WMMA operations with the context.
pub(super) fn register(ctx: &mut Context) {
    LdmatrixX4Op::register(ctx);
    LdmatrixX2Op::register(ctx);
    LdmatrixX4TransOp::register(ctx);
    LdmatrixX2TransOp::register(ctx);
    MmaM16N8K16F32F16Op::register(ctx);
}
