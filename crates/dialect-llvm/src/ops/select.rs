/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Select operation - conditional value selection.
//!
//! This module contains the LLVM dialect select operation:
//!
//! ```text
//! ┌───────────┬─────────────┬─────────────────────────────────────────────┐
//! │ Operation │ LLVM Opcode │ Description                                 │
//! ├───────────┼─────────────┼─────────────────────────────────────────────┤
//! │ SelectOp  │ select      │ Choose between two values based on condition│
//! └───────────┴─────────────┴─────────────────────────────────────────────┘
//! ```

use pliron::{
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface, OneResultInterface},
        types::IntegerType,
    },
    common_traits::Verify,
    context::Context,
    derive::{def_op, derive_op_interface_impl},
    op::Op,
    operation::Operation,
    result::Result,
    r#type::Typed,
    value::Value,
    verify_err,
};

// ============================================================================
// Select Operation
// ============================================================================

/// Verification errors for [`SelectOp`].
#[derive(thiserror::Error, Debug)]
pub enum SelectOpVerifyErr {
    #[error("Result must be the same as the true and false destination types")]
    ResultTypeErr,
    #[error("Condition must be an i1")]
    ConditionTypeErr,
}

/// Conditional value selection.
///
/// Returns `true_val` if `condition` is true, otherwise returns `false_val`.
///
/// Equivalent to LLVM's `select` instruction.
///
/// ### Operands
///
/// ```text
/// | operand      | description                   |
/// |--------------|-------------------------------|
/// | `condition`  | i1                            |
/// | `true_val`   | any type                      |
/// | `false_val`  | any type (same as true_val)   |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description                         |
/// |--------|-------------------------------------|
/// | `res`  | any type (same as true_val/false_val)|
/// ```
#[def_op("llvm.select")]
#[derive_op_interface_impl(NResultsInterface<1>, OneResultInterface, NOpdsInterface<3>)]
#[pliron::derive::format_op("$0 ` ? ` $1 ` : ` $2 ` : ` type($0)")]
pub struct SelectOp;

impl SelectOp {
    /// Create a new [`SelectOp`].
    pub fn new(ctx: &mut Context, cond: Value, true_val: Value, false_val: Value) -> Self {
        let result_type = true_val.get_type(ctx);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type],
            vec![cond, true_val, false_val],
            vec![],
            0,
        );
        Self { op }
    }
}

impl Verify for SelectOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let op = &*self.op.deref(ctx);
        let ty = op.get_type(0);
        let cond_ty = op.get_operand(0).get_type(ctx);
        let true_ty = op.get_operand(1).get_type(ctx);
        let false_ty = op.get_operand(2).get_type(ctx);
        if ty != true_ty || ty != false_ty {
            return verify_err!(loc, SelectOpVerifyErr::ResultTypeErr);
        }

        let cond_ty = cond_ty.deref(ctx);
        let cond_ty = cond_ty.downcast_ref::<IntegerType>();
        if cond_ty.is_none_or(|ty| ty.width() != 1) {
            return verify_err!(loc, SelectOpVerifyErr::ConditionTypeErr);
        }
        Ok(())
    }
}

// ============================================================================
// Registration
// ============================================================================

/// Register all select operations.
pub fn register(ctx: &mut Context) {
    SelectOp::register(ctx);
}
