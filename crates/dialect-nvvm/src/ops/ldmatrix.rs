/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-cooperative `ldmatrix.sync.aligned.m8n8.x4` operations (SM75+).

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Warp ldmatrix: load 4 x 8x8 b16 matrices from shared memory.
///
/// PTX: `ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0, r1, r2, r3}, [smem_ptr];`
///
/// # Operands
///
/// - `smem_ptr` (ptr): generic pointer into shared memory (cvta'd inside the asm)
///
/// # Results
///
/// - `r0..r3` (4 x i32): per-lane register fragments (each holds 2 packed b16)
#[pliron_op(
    name = "nvvm.ldmatrix_x4_b16",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<4>],
)]
pub struct LdmatrixX4B16Op;

impl LdmatrixX4B16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        LdmatrixX4B16Op { op }
    }
}

/// Warp ldmatrix with transpose: load 4 x 8x8 b16 matrices, transposed.
///
/// PTX: `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {r0, r1, r2, r3}, [smem_ptr];`
///
/// # Operands / Results
///
/// Same shape as [`LdmatrixX4B16Op`].
#[pliron_op(
    name = "nvvm.ldmatrix_x4_trans_b16",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<4>],
)]
pub struct LdmatrixX4TransB16Op;

impl LdmatrixX4TransB16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        LdmatrixX4TransB16Op { op }
    }
}

/// Register ldmatrix operations with the context.
pub(super) fn register(ctx: &mut Context) {
    LdmatrixX4B16Op::register(ctx);
    LdmatrixX4TransB16Op::register(ctx);
}
