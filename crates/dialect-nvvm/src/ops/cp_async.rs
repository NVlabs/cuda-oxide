/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ampere async copy (`cp.async`) operations for SM 80+ GPUs.
//!
//! These operations provide asynchronous global→shared memory copies that
//! bypass the register file, allowing warps to overlap compute with memory
//! transfers.
//!
//! # Operations
//!
//! ```text
//! ┌──────────────────────┬─────────────────────────────────────────────────┐
//! │ Operation            │ PTX                                             │
//! ├──────────────────────┼─────────────────────────────────────────────────┤
//! │ CpAsyncCg16Op        │ cp.async.cg.shared.global [smem], [gmem], 16;  │
//! │ CpAsyncCa16Op        │ cp.async.ca.shared.global [smem], [gmem], 16;  │
//! └──────────────────────┴─────────────────────────────────────────────────┘
//! ```

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Async 16-byte copy from global to shared memory with `.cg` cache policy.
///
/// # Operands
///
/// - `shared_dst` (ptr): destination in shared memory (16-byte aligned)
/// - `global_src` (ptr): source in global memory (16-byte aligned)
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_cg_16",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct CpAsyncCg16Op;

impl CpAsyncCg16Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        CpAsyncCg16Op { op }
    }
}

/// Async 16-byte copy from global to shared memory with `.ca` cache policy.
///
/// `.ca` caches at ALL levels (L1 + L2), unlike `.cg` which only caches in L2.
/// Use for data that benefits from L1 caching (e.g., small reused activation tiles).
///
/// # Operands
///
/// - `shared_dst` (ptr): destination in shared memory (16-byte aligned)
/// - `global_src` (ptr): source in global memory (16-byte aligned)
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_ca_16",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct CpAsyncCa16Op;

impl CpAsyncCa16Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        CpAsyncCa16Op { op }
    }
}

/// Register all cp.async operations with the context.
pub(super) fn register(ctx: &mut Context) {
    CpAsyncCg16Op::register(ctx);
    CpAsyncCa16Op::register(ctx);
}
