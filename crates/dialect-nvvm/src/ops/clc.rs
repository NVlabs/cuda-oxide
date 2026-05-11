/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cluster Launch Control (CLC) operations for Blackwell+ (SM 100+).
//!
//! CLC provides hardware-managed work-stealing for persistent kernels,
//! replacing software atomic counters with `try_cancel` / `query_cancel`.
//!
//! # Operations
//!
//! | Operation                  | Operands                | Results | PTX ISA |
//! |----------------------------|-------------------------|---------|---------|
//! | `ClcTryCancel`             | response_ptr, mbar_ptr  | 0       | 8.6+    |
//! | `ClcTryCancelMulticast`    | response_ptr, mbar_ptr  | 0       | 8.6+    |
//! | `ClcQueryIsCanceled`       | resp_lo, resp_hi        | 1 (u32) | 8.6+    |
//! | `ClcQueryGetFirstCtaidX`   | resp_lo, resp_hi        | 1 (u32) | 8.6+    |
//! | `ClcQueryGetFirstCtaidY`   | resp_lo, resp_hi        | 1 (u32) | 8.6+    |
//! | `ClcQueryGetFirstCtaidZ`   | resp_lo, resp_hi        | 1 (u32) | 8.6+    |

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

// =============================================================================
// try_cancel operations
// =============================================================================

/// Async try_cancel: request to steal a pending CTA's work.
///
/// Writes a 16-byte response to shared memory and signals the mbarrier.
///
/// PTX: `clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128
///        [response], [mbar];`
///
/// # Operands
///
/// - `response` (ptr): shared memory pointer for 16-byte response
/// - `mbar` (ptr): shared memory pointer to mbarrier
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.clc_try_cancel",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct ClcTryCancelOp;

impl ClcTryCancelOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Multicast try_cancel: broadcasts response to all CTAs in the cluster.
///
/// PTX: `clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes
///        .multicast::cluster::all.b128 [response], [mbar];`
///
/// # Operands
///
/// - `response` (ptr): shared memory pointer for 16-byte response
/// - `mbar` (ptr): shared memory pointer to mbarrier
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.clc_try_cancel_multicast",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct ClcTryCancelMulticastOp;

impl ClcTryCancelMulticastOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

// =============================================================================
// query_cancel operations
// =============================================================================

/// Query whether the try_cancel was canceled (no more work).
///
/// Returns 1 if canceled, 0 if a CTA was stolen (work available).
///
/// PTX: `clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 pred, response;`
///
/// # Operands
///
/// - `resp_lo` (i64): low 64 bits of the 16-byte response
/// - `resp_hi` (i64): high 64 bits of the 16-byte response
///
/// # Results
///
/// - `result` (i32): 1 if canceled, 0 if work available
#[pliron_op(
    name = "nvvm.clc_query_is_canceled",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct ClcQueryIsCanceledOp;

impl ClcQueryIsCanceledOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Get the X coordinate of the canceled CTA.
///
/// PTX: `clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 ret, response;`
///
/// # Operands
///
/// - `resp_lo` (i64): low 64 bits of the 16-byte response
/// - `resp_hi` (i64): high 64 bits of the 16-byte response
///
/// # Results
///
/// - `ctaid_x` (i32): X coordinate
#[pliron_op(
    name = "nvvm.clc_query_get_first_ctaid_x",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct ClcQueryGetFirstCtaidXOp;

impl ClcQueryGetFirstCtaidXOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Get the Y coordinate of the canceled CTA.
///
/// PTX: `clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 ret, response;`
#[pliron_op(
    name = "nvvm.clc_query_get_first_ctaid_y",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct ClcQueryGetFirstCtaidYOp;

impl ClcQueryGetFirstCtaidYOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Get the Z coordinate of the canceled CTA.
///
/// PTX: `clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 ret, response;`
#[pliron_op(
    name = "nvvm.clc_query_get_first_ctaid_z",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct ClcQueryGetFirstCtaidZOp;

impl ClcQueryGetFirstCtaidZOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Register CLC operations with the context.
pub(super) fn register(ctx: &mut Context) {
    ClcTryCancelOp::register(ctx);
    ClcTryCancelMulticastOp::register(ctx);
    ClcQueryIsCanceledOp::register(ctx);
    ClcQueryGetFirstCtaidXOp::register(ctx);
    ClcQueryGetFirstCtaidYOp::register(ctx);
    ClcQueryGetFirstCtaidZOp::register(ctx);
}
