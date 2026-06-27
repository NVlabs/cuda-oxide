/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Asynchronous copy (`cp.async`) operations.
//!
//! ```text
//! +----------------+-------+--------+--------------------------------------------------+
//! | Operation      | Bytes | Cache  | PTX                                              |
//! +----------------+-------+--------+--------------------------------------------------+
//! | CpAsyncCa4Op   | 4     | .ca    | cp.async.ca.shared.global [smem], [gmem], 4;     |
//! | CpAsyncCa8Op   | 8     | .ca    | cp.async.ca.shared.global [smem], [gmem], 8;     |
//! +----------------+-------+--------+--------------------------------------------------+
//! ```
//!
//! The `.cg` cache policy is only supported for 16-byte copies.

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

/// Register cp.async operations with the context.
pub(super) fn register(ctx: &mut Context) {
    CpAsyncCa4Op::register(ctx);
    CpAsyncCa8Op::register(ctx);
}
