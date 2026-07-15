/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared memory matrix store (stmatrix) operations.
//!
//! Stmatrix provides warp-cooperative matrix store operations that properly
//! handle tensor core fragment layouts when writing to shared memory.
//!
//! # Layout Transformation
//!
//! Tensor core operations produce register fragments optimized for computation.
//! `stmatrix` stores those fragments in row-major order by default; the
//! optional `.trans` qualifier selects column-major storage.
//!
//! ```text
//! ┌─────────────────────┬───────┬──────────┬───────────┬────────────────────┐
//! │ Operation           │ Tiles │ Elements │ Transpose │ PTX                │
//! ├─────────────────────┼───────┼──────────┼───────────┼────────────────────┤
//! │ StmatrixM8n8X4Op    │ 4     │ 256      │ No        │ stmatrix...m8n8.x4 │
//! │ StmatrixM8n8X4Trans │ 4     │ 256      │ Yes       │ stmatrix...x4.trans│
//! │ StmatrixM8n8X2Op    │ 2     │ 128      │ No        │ stmatrix...m8n8.x2 │
//! │ StmatrixM8n8X2Trans │ 2     │ 128      │ Yes       │ stmatrix...x2.trans│
//! └─────────────────────┴───────┴──────────┴───────────┴────────────────────┘
//! ```
//!
//! # Requirements
//!
//! - **Execution**: Warp-synchronous (all 32 threads must participate)
//! - **Memory**: Destination must be in shared memory
//! - **Alignment**: Each four-lane group writes a naturally aligned 16-byte row
//! - **Ordering**: This is a weak memory operation; callers must use a suitable
//!   barrier or fence before a dependent memory access
//! - **Architecture**: sm_90+ and PTX 7.8+

use dialect_mir::types::MirPtrType;
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

fn verify_stmatrix_operands(
    ctx: &Context,
    op: Ptr<Operation>,
    op_name: &str,
    register_count: usize,
) -> Result<(), Error> {
    let op = &*op.deref(ctx);
    let operands: Vec<_> = op.operands().collect();
    if operands.len() != register_count + 1 {
        return verify_err!(
            op.loc(),
            "{} requires one pointer and {} register operands",
            op_name,
            register_count
        );
    }

    let pointer_ty = operands[0].get_type(ctx);
    if pointer_ty.deref(ctx).downcast_ref::<MirPtrType>().is_none() {
        return verify_err!(op.loc(), "{} operand 0 must be a MIR pointer", op_name);
    }

    for (index, register) in operands.iter().enumerate().skip(1) {
        let ty = register.get_type(ctx);
        let ty = ty.deref(ctx);
        let Some(int_ty) = ty.downcast_ref::<IntegerType>() else {
            return verify_err!(
                op.loc(),
                "{} register operand {} must be an integer",
                op_name,
                index - 1
            );
        };
        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "{} register operand {} must be 32 bits",
                op_name,
                index - 1
            );
        }
    }

    Ok(())
}

// =============================================================================
// 4-Tile Store Operations
// =============================================================================

/// Store four 8×8 matrix tiles to shared memory.
///
/// Warp-cooperative matrix store without transpose.
///
/// PTX: `stmatrix.sync.aligned.m8n8.x4.shared.b16 [ptr], {r0, r1, r2, r3};`
///
/// # Operands
///
/// - `smem_ptr` (ptr): destination pointer in shared memory
/// - `r0` (i32): first register containing two packed b16 values
/// - `r1` (i32): second packed register
/// - `r2` (i32): third packed register
/// - `r3` (i32): fourth packed register
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.stmatrix_m8n8_x4",
    format,
    interfaces = [NOpdsInterface<5>, NResultsInterface<0>],
)]
pub struct StmatrixM8n8X4Op;

impl StmatrixM8n8X4Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        StmatrixM8n8X4Op { op }
    }
}

impl Verify for StmatrixM8n8X4Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_stmatrix_operands(ctx, self.get_operation(), "nvvm.stmatrix_m8n8_x4", 4)
    }
}

/// Store four 8×8 matrix tiles with transpose.
///
/// Warp-cooperative matrix store with the `.trans` modifier, which selects
/// column-major storage.
///
/// PTX: `stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [ptr], {r0, r1, r2, r3};`
///
/// # Operands
///
/// - `smem_ptr` (ptr): destination pointer in shared memory
/// - `r0` (u32): first register (2 packed bf16 values)
/// - `r1` (u32): second register
/// - `r2` (u32): third register
/// - `r3` (u32): fourth register
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.stmatrix_m8n8_x4_trans",
    format,
    interfaces = [NOpdsInterface<5>, NResultsInterface<0>],
)]
pub struct StmatrixM8n8X4TransOp;

impl StmatrixM8n8X4TransOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        StmatrixM8n8X4TransOp { op }
    }
}

impl Verify for StmatrixM8n8X4TransOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_stmatrix_operands(ctx, self.get_operation(), "nvvm.stmatrix_m8n8_x4_trans", 4)
    }
}

// =============================================================================
// 2-Tile Store Operations
// =============================================================================

/// Store two 8×8 matrix tiles to shared memory.
///
/// Warp-cooperative matrix store without transpose. Stores 16 columns
/// (2 × 8×8 tiles) per call.
///
/// SASS encoding: `STSM.16.MT88.2`
///
/// PTX: `stmatrix.sync.aligned.m8n8.x2.shared.b16 [ptr], {r0, r1};`
///
/// # Operands
///
/// - `smem_ptr` (ptr): destination pointer in shared memory
/// - `r0` (i32): first register (2 packed bf16 values)
/// - `r1` (i32): second register
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.stmatrix_m8n8_x2",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct StmatrixM8n8X2Op;

impl StmatrixM8n8X2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        StmatrixM8n8X2Op { op }
    }
}

impl Verify for StmatrixM8n8X2Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_stmatrix_operands(ctx, self.get_operation(), "nvvm.stmatrix_m8n8_x2", 2)
    }
}

/// Store two 8×8 bf16 matrices to shared memory with transpose.
///
/// The transpose form stores the matrix in column-major order.
///
/// PTX: `stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [ptr], {r0, r1};`
///
/// # Operands
///
/// - `smem_ptr` (ptr): destination pointer in shared memory
/// - `r0` (i32): first register (2 packed bf16 values)
/// - `r1` (i32): second register
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.stmatrix_m8n8_x2_trans",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct StmatrixM8n8X2TransOp;

impl StmatrixM8n8X2TransOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        StmatrixM8n8X2TransOp { op }
    }
}

impl Verify for StmatrixM8n8X2TransOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_stmatrix_operands(ctx, self.get_operation(), "nvvm.stmatrix_m8n8_x2_trans", 2)
    }
}

/// Register stmatrix operations with the context.
pub(super) fn register(ctx: &mut Context) {
    // 4-tile store
    StmatrixM8n8X4Op::register(ctx);
    StmatrixM8n8X4TransOp::register(ctx);
    // 2-tile store
    StmatrixM8n8X2Op::register(ctx);
    StmatrixM8n8X2TransOp::register(ctx);
}
