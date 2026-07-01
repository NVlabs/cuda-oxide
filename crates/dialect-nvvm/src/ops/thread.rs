/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thread, block, and grid indexing operations.
//!
//! This module provides operations for reading GPU thread hierarchy registers:
//!
//! ```text
//! ┌──────────────────────┬──────────────┬────────────────────────────┐
//! │ Operation            │ PTX Register │ Description                │
//! ├──────────────────────┼──────────────┼────────────────────────────┤
//! │ ReadPtxSregTidXOp    │ %tid.x       │ Thread ID within block (X) │
//! │ ReadPtxSregTidYOp    │ %tid.y       │ Thread ID within block (Y) │
//! │ ReadPtxSregTidZOp    │ %tid.z       │ Thread ID within block (Z) │
//! │ ReadPtxSregCtaidXOp  │ %ctaid.x     │ Block ID within grid (X)   │
//! │ ReadPtxSregCtaidYOp  │ %ctaid.y     │ Block ID within grid (Y)   │
//! │ ReadPtxSregCtaidZOp  │ %ctaid.z     │ Block ID within grid (Z)   │
//! │ ReadPtxSregNtidXOp   │ %ntid.x      │ Block dimension (X)        │
//! │ ReadPtxSregNtidYOp   │ %ntid.y      │ Block dimension (Y)        │
//! │ ReadPtxSregNtidZOp   │ %ntid.z      │ Block dimension (Z)        │
//! │ ReadPtxSregNctaidXOp │ %nctaid.x    │ Grid dimension (X)         │
//! │ ReadPtxSregNctaidYOp │ %nctaid.y    │ Grid dimension (Y)         │
//! │ ReadPtxSregNctaidZOp │ %nctaid.z    │ Grid dimension (Z)         │
//! │ ReadPtxSregEnvReg1Op │ %envreg1     │ Driver ABI envreg 1        │
//! │ ReadPtxSregEnvReg2Op │ %envreg2     │ Driver ABI envreg 2        │
//! │ Barrier0Op           │ bar.sync 0   │ Block-wide barrier         │
//! │ ThreadfenceBlockOp   │ membar.cta   │ Block-scoped memory fence  │
//! │ ThreadfenceOp        │ membar.gl    │ Device-scoped memory fence │
//! │ ThreadfenceSystemOp  │ membar.sys   │ System-scoped memory fence │
//! └──────────────────────┴──────────────┴────────────────────────────┘
//! ```
//!
//! # Thread Hierarchy
//!
//! ```text
//! Grid (gridDim.x × gridDim.y blocks)
//! └── Block (blockDim.x × blockDim.y threads)
//!     └── Thread (identified by threadIdx)
//! ```
//!
//! Each operation returns a 32-bit integer representing the index or dimension.

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

// =============================================================================
// X-Dimension Indexing
// =============================================================================

/// Read the X component of the thread ID within the block.
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.tid.x` / PTX `%tid.x`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_tid_x",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregTidXOp;

impl ReadPtxSregTidXOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregTidXOp { op }
    }
}

impl Verify for ReadPtxSregTidXOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(op.loc(), "nvvm.read_ptx_sreg_tid_x result must be integer");
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_tid_x result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Read the X component of the block ID within the grid.
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.ctaid.x` / PTX `%ctaid.x`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_ctaid_x",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregCtaidXOp;

impl ReadPtxSregCtaidXOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregCtaidXOp { op }
    }
}

impl Verify for ReadPtxSregCtaidXOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(
                    op.loc(),
                    "nvvm.read_ptx_sreg_ctaid_x result must be integer"
                );
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_ctaid_x result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Read the X component of the block dimension (threads per block).
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.ntid.x` / PTX `%ntid.x`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_ntid_x",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregNtidXOp;

impl ReadPtxSregNtidXOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregNtidXOp { op }
    }
}

impl Verify for ReadPtxSregNtidXOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(op.loc(), "nvvm.read_ptx_sreg_ntid_x result must be integer");
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_ntid_x result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

// =============================================================================
// Y-Dimension Indexing
// =============================================================================

/// Read the Y component of the thread ID within the block.
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.tid.y` / PTX `%tid.y`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_tid_y",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregTidYOp;

impl ReadPtxSregTidYOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregTidYOp { op }
    }
}

impl Verify for ReadPtxSregTidYOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(op.loc(), "nvvm.read_ptx_sreg_tid_y result must be integer");
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_tid_y result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Read the Y component of the block ID within the grid.
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.ctaid.y` / PTX `%ctaid.y`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_ctaid_y",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregCtaidYOp;

impl ReadPtxSregCtaidYOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregCtaidYOp { op }
    }
}

impl Verify for ReadPtxSregCtaidYOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(
                    op.loc(),
                    "nvvm.read_ptx_sreg_ctaid_y result must be integer"
                );
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_ctaid_y result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Read the Y component of the block dimension (threads per block).
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.ntid.y` / PTX `%ntid.y`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_ntid_y",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregNtidYOp;

impl ReadPtxSregNtidYOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregNtidYOp { op }
    }
}

impl Verify for ReadPtxSregNtidYOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(op.loc(), "nvvm.read_ptx_sreg_ntid_y result must be integer");
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_ntid_y result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

// =============================================================================
// Z-Dimension Indexing
// =============================================================================

/// Read the Z component of the thread ID within the block.
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.tid.z` / PTX `%tid.z`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_tid_z",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregTidZOp;

impl ReadPtxSregTidZOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregTidZOp { op }
    }
}

impl Verify for ReadPtxSregTidZOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(op.loc(), "nvvm.read_ptx_sreg_tid_z result must be integer");
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_tid_z result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Read the Z component of the block ID within the grid.
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.ctaid.z` / PTX `%ctaid.z`.
#[pliron_op(
    name = "nvvm.read_ptx_sreg_ctaid_z",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregCtaidZOp;

impl ReadPtxSregCtaidZOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregCtaidZOp { op }
    }
}

impl Verify for ReadPtxSregCtaidZOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(
                    op.loc(),
                    "nvvm.read_ptx_sreg_ctaid_z result must be integer"
                );
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_ctaid_z result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Read the Z component of the block dimension (threads per block).
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.ntid.z` / PTX `%ntid.z`.
#[pliron_op(
    name = "nvvm.read_ptx_sreg_ntid_z",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregNtidZOp;

impl ReadPtxSregNtidZOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregNtidZOp { op }
    }
}

impl Verify for ReadPtxSregNtidZOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(op.loc(), "nvvm.read_ptx_sreg_ntid_z result must be integer");
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_ntid_z result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

// =============================================================================
// Grid Dimension Indexing (nctaid)
// =============================================================================

/// Read the X component of the grid dimension (number of blocks per grid).
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.nctaid.x` / PTX `%nctaid.x`.
#[pliron_op(
    name = "nvvm.read_ptx_sreg_nctaid_x",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregNctaidXOp;

impl ReadPtxSregNctaidXOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregNctaidXOp { op }
    }
}

impl Verify for ReadPtxSregNctaidXOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(
                    op.loc(),
                    "nvvm.read_ptx_sreg_nctaid_x result must be integer"
                );
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_nctaid_x result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Read the Y component of the grid dimension.
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.nctaid.y` / PTX `%nctaid.y`.
#[pliron_op(
    name = "nvvm.read_ptx_sreg_nctaid_y",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregNctaidYOp;

impl ReadPtxSregNctaidYOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregNctaidYOp { op }
    }
}

impl Verify for ReadPtxSregNctaidYOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(
                    op.loc(),
                    "nvvm.read_ptx_sreg_nctaid_y result must be integer"
                );
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_nctaid_y result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Read the Z component of the grid dimension.
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.nctaid.z` / PTX `%nctaid.z`.
#[pliron_op(
    name = "nvvm.read_ptx_sreg_nctaid_z",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregNctaidZOp;

impl ReadPtxSregNctaidZOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregNctaidZOp { op }
    }
}

impl Verify for ReadPtxSregNctaidZOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(
                    op.loc(),
                    "nvvm.read_ptx_sreg_nctaid_z result must be integer"
                );
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_nctaid_z result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

// =============================================================================
// Driver-ABI Environment Registers (envreg)
// =============================================================================
//
// PTX exposes 32 environment registers (%envreg0..%envreg31) that the CUDA
// driver populates before kernel entry. For cooperative launches the driver
// writes the address of a per-launch "grid workspace" struct into envreg1
// (low 32 bits) and envreg2 (high 32 bits). The grid workspace contains the
// barrier counter used by `cooperative_groups::grid::sync()`.

/// Read PTX environment register 1.
///
/// For cooperative launches: low 32 bits of the grid workspace pointer.
#[pliron_op(
    name = "nvvm.read_ptx_sreg_envreg1",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregEnvReg1Op;

impl ReadPtxSregEnvReg1Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregEnvReg1Op { op }
    }
}

impl Verify for ReadPtxSregEnvReg1Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(
                    op.loc(),
                    "nvvm.read_ptx_sreg_envreg1 result must be integer"
                );
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_envreg1 result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Read PTX environment register 2.
///
/// For cooperative launches: high 32 bits of the grid workspace pointer.
#[pliron_op(
    name = "nvvm.read_ptx_sreg_envreg2",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregEnvReg2Op;

impl ReadPtxSregEnvReg2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregEnvReg2Op { op }
    }
}

impl Verify for ReadPtxSregEnvReg2Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(
                    op.loc(),
                    "nvvm.read_ptx_sreg_envreg2 result must be integer"
                );
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_envreg2 result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

// =============================================================================
// Block Synchronization
// =============================================================================

/// Block-wide barrier synchronization.
///
/// All threads in the block must reach this barrier before any can proceed.
/// Corresponds to `llvm.nvvm.barrier0` / PTX `bar.sync 0`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 0 results
#[pliron_op(
    name = "nvvm.barrier0",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct Barrier0Op;

impl Barrier0Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Barrier0Op { op }
    }
}

/// Block-scoped memory fence.
///
/// Orders the calling thread's prior memory operations before later memory
/// operations as observed by threads in the same CTA. Corresponds to PTX
/// `membar.cta`.
#[pliron_op(
    name = "nvvm.threadfence_block",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct ThreadfenceBlockOp;

impl ThreadfenceBlockOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ThreadfenceBlockOp { op }
    }
}

/// Device-scoped memory fence.
///
/// Orders the calling thread's prior global-memory operations before later
/// memory operations as observed by threads on the same GPU. Corresponds to
/// PTX `membar.gl`.
#[pliron_op(
    name = "nvvm.threadfence",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct ThreadfenceOp;

impl ThreadfenceOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ThreadfenceOp { op }
    }
}

/// System-scoped memory fence.
///
/// Orders the calling thread's prior global-memory operations before later
/// memory operations as observed by other GPUs or the CPU. Corresponds to PTX
/// `membar.sys`.
#[pliron_op(
    name = "nvvm.threadfence_system",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct ThreadfenceSystemOp;

impl ThreadfenceSystemOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ThreadfenceSystemOp { op }
    }
}

// =============================================================================
// SM and Grid Identification
// =============================================================================

/// Read the SM (streaming multiprocessor) processor ID.
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.smid` / PTX `%smid`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_smid",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregSmIdOp;

impl ReadPtxSregSmIdOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregSmIdOp { op }
    }
}

impl Verify for ReadPtxSregSmIdOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_sreg_integer_result(ctx, self.get_operation(), "smid", 32)
    }
}

/// Read the maximum SM ID + 1 (number of SM slots).
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.nsmid` / PTX `%nsmid`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_nsmid",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregNsmIdOp;

impl ReadPtxSregNsmIdOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregNsmIdOp { op }
    }
}

impl Verify for ReadPtxSregNsmIdOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_sreg_integer_result(ctx, self.get_operation(), "nsmid", 32)
    }
}

/// Read the grid launch identifier.
///
/// Corresponds to the modern 64-bit PTX `%gridid` register.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i64`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_gridid",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregGridIdOp;

impl ReadPtxSregGridIdOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregGridIdOp { op }
    }
}

impl Verify for ReadPtxSregGridIdOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_sreg_integer_result(ctx, self.get_operation(), "gridid", 64)
    }
}

/// Shared verifier for SM/grid identification special-register results.
fn verify_sreg_integer_result(
    ctx: &Context,
    op: Ptr<Operation>,
    register: &str,
    width: u32,
) -> Result<(), Error> {
    let op = &*op.deref(ctx);
    let res = op.get_result(0);
    let ty = res.get_type(ctx);

    let ty_obj = ty.deref(ctx);
    let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
        Some(ty) => ty,
        None => {
            return verify_err!(op.loc(), "%{} result must be integer", register);
        }
    };

    if int_ty.width() != width {
        return verify_err!(
            op.loc(),
            "%{} result must be {}-bit integer",
            register,
            width
        );
    }
    Ok(())
}

/// Register thread indexing operations with the context.
pub(super) fn register(ctx: &mut Context) {
    // X-dimension
    ReadPtxSregTidXOp::register(ctx);
    ReadPtxSregCtaidXOp::register(ctx);
    ReadPtxSregNtidXOp::register(ctx);
    // Y-dimension
    ReadPtxSregTidYOp::register(ctx);
    ReadPtxSregCtaidYOp::register(ctx);
    ReadPtxSregNtidYOp::register(ctx);
    // Z-dimension
    ReadPtxSregTidZOp::register(ctx);
    ReadPtxSregCtaidZOp::register(ctx);
    ReadPtxSregNtidZOp::register(ctx);
    // Grid dimensions
    ReadPtxSregNctaidXOp::register(ctx);
    ReadPtxSregNctaidYOp::register(ctx);
    ReadPtxSregNctaidZOp::register(ctx);
    // Driver-ABI environment registers
    ReadPtxSregEnvReg1Op::register(ctx);
    ReadPtxSregEnvReg2Op::register(ctx);
    // SM and grid identification
    ReadPtxSregSmIdOp::register(ctx);
    ReadPtxSregNsmIdOp::register(ctx);
    ReadPtxSregGridIdOp::register(ctx);
    // Synchronization
    Barrier0Op::register(ctx);
    ThreadfenceBlockOp::register(ctx);
    ThreadfenceOp::register(ctx);
    ThreadfenceSystemOp::register(ctx);
}
