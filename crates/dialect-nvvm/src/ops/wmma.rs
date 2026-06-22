/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! WMMA (Warp Matrix Multiply-Accumulate) operations for integer sub-byte types.
//!
//! These operations map to PTX `mma.sync.aligned` instructions for int4 types.
//!
//! # Operations
//!
//! | Operation              | PTX instruction                                          | A frags | B frags |
//! |------------------------|----------------------------------------------------------|---------|---------|
//! | `MmaM16N8K32S32S4Op`  | `mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32`       | 2×u32   | 1×u32   |
//! | `MmaM16N8K32S32U4Op`  | `mma.sync.aligned.m16n8k32.row.col.s32.u4.u4.s32`       | 2×u32   | 1×u32   |
//! | `MmaM16N8K64S32S4Op`  | `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32`       | 4×u32   | 2×u32   |
//! | `MmaM16N8K64S32U4Op`  | `mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32`       | 4×u32   | 2×u32   |
//!
//! All operations take 3 pointer operands: `(acc_ptr, a_ptr, b_ptr)` and produce no results.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

// =============================================================================
// Signed int4 operations
// =============================================================================

/// MMA m16n8k32 with signed int4 (s4) operands, s32 accumulator.
///
/// PTX: `mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32`
///
/// # Operands
/// - `acc_ptr` (ptr): pointer to 4×i32 accumulator (read-modify-write)
/// - `a_ptr` (ptr): pointer to 2×u32 A-fragment
/// - `b_ptr` (ptr): pointer to 1×u32 B-fragment
#[pliron_op(
    name = "nvvm.mma_m16n8k32_s32_s4",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct MmaM16N8K32S32S4Op;

impl MmaM16N8K32S32S4Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K32S32S4Op { op }
    }
}

/// MMA m16n8k64 with signed int4 (s4) operands, s32 accumulator.
///
/// PTX: `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32`
///
/// # Operands
/// - `acc_ptr` (ptr): pointer to 4×i32 accumulator (read-modify-write)
/// - `a_ptr` (ptr): pointer to 4×u32 A-fragment
/// - `b_ptr` (ptr): pointer to 2×u32 B-fragment
#[pliron_op(
    name = "nvvm.mma_m16n8k64_s32_s4",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct MmaM16N8K64S32S4Op;

impl MmaM16N8K64S32S4Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K64S32S4Op { op }
    }
}

// =============================================================================
// Unsigned int4 operations
// =============================================================================

/// MMA m16n8k32 with unsigned int4 (u4) operands, s32 accumulator.
///
/// PTX: `mma.sync.aligned.m16n8k32.row.col.s32.u4.u4.s32`
///
/// # Operands
/// - `acc_ptr` (ptr): pointer to 4×i32 accumulator (read-modify-write)
/// - `a_ptr` (ptr): pointer to 2×u32 A-fragment
/// - `b_ptr` (ptr): pointer to 1×u32 B-fragment
#[pliron_op(
    name = "nvvm.mma_m16n8k32_s32_u4",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct MmaM16N8K32S32U4Op;

impl MmaM16N8K32S32U4Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K32S32U4Op { op }
    }
}

/// MMA m16n8k64 with unsigned int4 (u4) operands, s32 accumulator.
///
/// PTX: `mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32`
///
/// # Operands
/// - `acc_ptr` (ptr): pointer to 4×i32 accumulator (read-modify-write)
/// - `a_ptr` (ptr): pointer to 4×u32 A-fragment
/// - `b_ptr` (ptr): pointer to 2×u32 B-fragment
#[pliron_op(
    name = "nvvm.mma_m16n8k64_s32_u4",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct MmaM16N8K64S32U4Op;

impl MmaM16N8K64S32U4Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K64S32U4Op { op }
    }
}

/// Register all WMMA operations with the context.
pub(super) fn register(ctx: &mut Context) {
    MmaM16N8K32S32S4Op::register(ctx);
    MmaM16N8K32S32U4Op::register(ctx);
    MmaM16N8K64S32S4Op::register(ctx);
    MmaM16N8K64S32U4Op::register(ctx);
}
