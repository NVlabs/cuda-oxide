/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared memory size query operations.
//!
//! Read-only special registers that report the amount of shared memory
//! available to the current kernel:
//!
//! ```text
//! ┌──────────────────────────────────┬───────────────────────┬──────────────────────────────────┐
//! │ Operation                        │ PTX Register          │ Description                      │
//! ├──────────────────────────────────┼───────────────────────┼──────────────────────────────────┤
//! │ ReadPtxSregDynamicSmemSizeOp     │ %dynamic_smem_size    │ Dynamic shared memory (bytes)     │
//! │ ReadPtxSregTotalSmemSizeOp       │ %total_smem_size      │ Total shared memory (bytes)       │
//! └──────────────────────────────────┴───────────────────────┴──────────────────────────────────┘
//! ```

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    builtin::types::IntegerType,
    common_traits::Verify,
    context::Context,
    context::Ptr,
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    verify_err,
};
use pliron_derive::pliron_op;

/// Read the size (in bytes) of dynamic shared memory for this kernel.
///
/// Corresponds to PTX `%dynamic_smem_size`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_dynamic_smem_size",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregDynamicSmemSizeOp;

impl ReadPtxSregDynamicSmemSizeOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregDynamicSmemSizeOp { op }
    }
}

impl Verify for ReadPtxSregDynamicSmemSizeOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_smem_size_result(ctx, self.get_operation(), "dynamic_smem_size")
    }
}

/// Read the total size (in bytes) of shared memory for this kernel.
///
/// Corresponds to PTX `%total_smem_size`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_total_smem_size",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregTotalSmemSizeOp;

impl ReadPtxSregTotalSmemSizeOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregTotalSmemSizeOp { op }
    }
}

impl Verify for ReadPtxSregTotalSmemSizeOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_smem_size_result(ctx, self.get_operation(), "total_smem_size")
    }
}

/// Shared verifier for shared-memory size ops: a single 32-bit integer result.
fn verify_smem_size_result(ctx: &Context, op: Ptr<Operation>, op_name: &str) -> Result<(), Error> {
    let op = &*op.deref(ctx);
    let res = op.get_result(0);
    let ty = res.get_type(ctx);

    let ty_obj = ty.deref(ctx);
    let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
        Some(ty) => ty,
        None => {
            return verify_err!(op.loc(), "{} result must be integer", op_name);
        }
    };

    if int_ty.width() != 32 {
        return verify_err!(op.loc(), "{} result must be 32-bit integer", op_name);
    }
    Ok(())
}

/// Register shared memory size query operations with the context.
pub(super) fn register(ctx: &mut Context) {
    ReadPtxSregDynamicSmemSizeOp::register(ctx);
    ReadPtxSregTotalSmemSizeOp::register(ctx);
}
