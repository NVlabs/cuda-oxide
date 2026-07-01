/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix dialect operations.

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

/// In-register 8×8 matrix transpose (movmatrix.sync.aligned.m8n8.trans.b16).
#[pliron_op(
    name = "nvvm.movmatrix_trans_b16",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct MovmatrixTransB16Op;

impl MovmatrixTransB16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for MovmatrixTransB16Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);

        if op.operands().count() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "nvvm.movmatrix_trans_b16 requires one operand and one result"
            );
        }

        for (name, ty) in [
            ("operand", op.get_operand(0).get_type(ctx)),
            ("result", op.get_result(0).get_type(ctx)),
        ] {
            let ty_ref = ty.deref(ctx);
            let Some(integer) = ty_ref.downcast_ref::<IntegerType>() else {
                return verify_err!(
                    op.loc(),
                    "nvvm.movmatrix_trans_b16 {} must be a 32-bit integer",
                    name
                );
            };
            if integer.width() != 32 {
                return verify_err!(
                    op.loc(),
                    "nvvm.movmatrix_trans_b16 {} must be a 32-bit integer",
                    name
                );
            }
        }

        Ok(())
    }
}

/// Warp MMA: m8n8k4 with f64 accumulator and f64 inputs.
///
/// # Operands
///
/// - `acc_ptr` (ptr): pointer to `[f64; 2]` accumulator (read-modify-write)
/// - `a_ptr` (ptr): pointer to `f64` A fragment
/// - `b_ptr` (ptr): pointer to `f64` B fragment
///
/// # Results
///
/// - None (accumulator updated in-place via pointer)
#[pliron_op(
    name = "nvvm.mma_m8n8k4_f64",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct MmaM8N8K4F64Op;

impl Verify for MmaM8N8K4F64Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let operands: Vec<_> = op.operands().collect();

        if operands.len() != 3 {
            return verify_err!(
                op.loc(),
                "nvvm.mma_m8n8k4_f64 requires 3 pointer operands, got {}",
                operands.len()
            );
        }

        for (i, operand) in operands.iter().enumerate() {
            let ty = operand.get_type(ctx);
            if ty.deref(ctx).downcast_ref::<MirPtrType>().is_none() {
                return verify_err!(
                    op.loc(),
                    "nvvm.mma_m8n8k4_f64 operand {} must be a MIR pointer",
                    i
                );
            }
        }

        if op.get_num_results() != 0 {
            return verify_err!(op.loc(), "nvvm.mma_m8n8k4_f64 must have 0 results");
        }

        Ok(())
    }
}

impl MmaM8N8K4F64Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM8N8K4F64Op { op }
    }
}

pub(super) fn register(ctx: &mut Context) {
    MovmatrixTransB16Op::register(ctx);
    MmaM8N8K4F64Op::register(ctx);
}
