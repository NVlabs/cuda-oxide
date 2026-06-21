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
//! │ CpAsyncCommitGroupOp │ cp.async.commit_group;                          │
//! │ CpAsyncWaitGroupOp   │ cp.async.wait_group N;                          │
//! │ CpAsyncWaitAllOp     │ cp.async.wait_all;                              │
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

/// Commit all prior `cp.async` operations into a completion group.
///
/// # Operands
///
/// - None
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_commit_group",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct CpAsyncCommitGroupOp;

impl CpAsyncCommitGroupOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        CpAsyncCommitGroupOp { op }
    }
}

/// Wait until at most N completion groups remain in-flight.
///
/// The N value is stored as an attribute on the operation.
///
/// # Attributes
///
/// * `wait_n` - Number of groups that may remain in-flight (u32)
///
/// # Operands
///
/// - None
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_wait_group",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct CpAsyncWaitGroupOp;

impl CpAsyncWaitGroupOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        CpAsyncWaitGroupOp { op }
    }

    /// Create a new wait_group operation with the given N value.
    pub fn new_with_n(ctx: &mut Context, n: u32) -> Ptr<Operation> {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);

        use pliron::builtin::attributes::IntegerAttr;
        use pliron::builtin::types::{IntegerType, Signedness};
        use pliron::identifier::Identifier;
        use pliron::utils::apint::APInt;
        use std::num::NonZeroUsize;

        let i32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
        let apint = APInt::from_u64(n as u64, NonZeroUsize::new(32).unwrap());
        let attr = IntegerAttr::new(i32_ty, apint);
        let key = Identifier::try_from("wait_n").unwrap();
        op.deref_mut(ctx).attributes.set(key, attr);

        op
    }

    /// Get the N value from the operation's attributes.
    pub fn get_wait_n(&self, ctx: &Context) -> Option<u32> {
        use pliron::builtin::attributes::IntegerAttr;
        use pliron::identifier::Identifier;

        let key = Identifier::try_from("wait_n").unwrap();
        let op_ref = self.get_operation().deref(ctx);
        let int_attr: &IntegerAttr = op_ref.attributes.get(&key)?;
        Some(int_attr.value().to_u64() as u32)
    }
}

/// Wait for all outstanding `cp.async` groups to complete.
///
/// # Operands
///
/// - None
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_wait_all",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct CpAsyncWaitAllOp;

impl CpAsyncWaitAllOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        CpAsyncWaitAllOp { op }
    }
}

/// Register all cp.async operations with the context.
pub(super) fn register(ctx: &mut Context) {
    CpAsyncCg16Op::register(ctx);
    CpAsyncCa16Op::register(ctx);
    CpAsyncCommitGroupOp::register(ctx);
    CpAsyncWaitGroupOp::register(ctx);
    CpAsyncWaitAllOp::register(ctx);
}
