/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Asynchronous copy (`cp.async`) operations.
//!
//! ```text
//! +---------------------+-------+--------+----------------------------------------------------+
//! | Operation           | Bytes | Cache  | PTX                                                |
//! +---------------------+-------+--------+----------------------------------------------------+
//! | CpAsyncCa4Op        | 4     | .ca    | cp.async.ca.shared.global [smem], [gmem], 4;       |
//! | CpAsyncCa8Op        | 8     | .ca    | cp.async.ca.shared.global [smem], [gmem], 8;       |
//! | CpAsyncCaZfill4Op   | 4     | .ca    | cp.async.ca.shared.global [smem], [gmem], 4, src;  |
//! | CpAsyncCaZfill8Op   | 8     | .ca    | cp.async.ca.shared.global [smem], [gmem], 8, src;  |
//! | CpAsyncCaZfill16Op  | 16    | .ca    | cp.async.ca.shared.global [smem], [gmem],16, src;  |
//! +---------------------+-------+--------+----------------------------------------------------+
//! ```
//!
//! The `.cg` cache policy is only supported for 16-byte copies.
//! The zero-fill variants (Zfill) copy `src_size` bytes and zero-fill the rest.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Asynchronous 4-byte copy from global to shared memory (`.ca` cache policy).
///
/// PTX: `cp.async.ca.shared.global [%smem32], [$1], 4;`
///
/// # Operands
///
/// - `shared_dst` (ptr): destination pointer in shared memory
/// - `global_src` (ptr): source pointer in global memory
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_ca_4",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct CpAsyncCa4Op;

impl CpAsyncCa4Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        CpAsyncCa4Op { op }
    }
}

/// Asynchronous 8-byte copy from global to shared memory (`.ca` cache policy).
///
/// PTX: `cp.async.ca.shared.global [%smem32], [$1], 8;`
///
/// # Operands
///
/// - `shared_dst` (ptr): destination pointer in shared memory
/// - `global_src` (ptr): source pointer in global memory
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_ca_8",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct CpAsyncCa8Op;

impl CpAsyncCa8Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        CpAsyncCa8Op { op }
    }
}

// =============================================================================
// cp.async with zero-fill (3 operands: dst, src, src_size)
// =============================================================================

/// Async copy 4 bytes from global to shared with zero-fill.
///
/// PTX: `cp.async.ca.shared.global [smem], [gmem], 4, src_size;`
///
/// # Operands
///
/// - `shared_dst` (ptr): destination pointer in shared memory
/// - `global_src` (ptr): source pointer in global memory
/// - `src_size` (i32): number of valid source bytes (0..=4)
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_ca_zfill_4",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct CpAsyncCaZfill4Op;

impl CpAsyncCaZfill4Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        CpAsyncCaZfill4Op { op }
    }
}

/// Async copy 8 bytes from global to shared with zero-fill.
///
/// PTX: `cp.async.ca.shared.global [smem], [gmem], 8, src_size;`
///
/// # Operands
///
/// - `shared_dst` (ptr): destination pointer in shared memory
/// - `global_src` (ptr): source pointer in global memory
/// - `src_size` (i32): number of valid source bytes (0..=8)
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_ca_zfill_8",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct CpAsyncCaZfill8Op;

impl CpAsyncCaZfill8Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        CpAsyncCaZfill8Op { op }
    }
}

/// Async copy 16 bytes from global to shared with zero-fill.
///
/// PTX: `cp.async.ca.shared.global [smem], [gmem], 16, src_size;`
///
/// # Operands
///
/// - `shared_dst` (ptr): destination pointer in shared memory
/// - `global_src` (ptr): source pointer in global memory
/// - `src_size` (i32): number of valid source bytes (0..=16)
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_ca_zfill_16",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct CpAsyncCaZfill16Op;

impl CpAsyncCaZfill16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        CpAsyncCaZfill16Op { op }
    }
}

/// Register cp.async operations with the context.
pub(super) fn register(ctx: &mut Context) {
    CpAsyncCa4Op::register(ctx);
    CpAsyncCa8Op::register(ctx);
    CpAsyncCaZfill4Op::register(ctx);
    CpAsyncCaZfill8Op::register(ctx);
    CpAsyncCaZfill16Op::register(ctx);
}
