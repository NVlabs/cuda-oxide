/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Variadic argument operations.
//!
//! This module contains LLVM dialect operations for variadic function support:
//!
//! ```text
//! ┌───────────┬─────────────┬─────────────────────────────────────────┐
//! │ Operation │ LLVM Opcode │ Description                             │
//! ├───────────┼─────────────┼─────────────────────────────────────────┤
//! │ VAArgOp   │ va_arg      │ Extract next argument from va_list      │
//! └───────────┴─────────────┴─────────────────────────────────────────┘
//! ```

use pliron::{
    builtin::op_interfaces::{
        NOpdsInterface, NResultsInterface, OneOpdInterface, OneResultInterface,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    derive::{def_op, derive_op_interface_impl},
    op::Op,
    operation::Operation,
    result::Result,
    r#type::TypeObj,
    value::Value,
    verify_err,
};

use crate::types::PointerType;

// ============================================================================
// VAArg Operation
// ============================================================================

/// Verification errors for [`VAArgOp`].
#[derive(thiserror::Error, Debug)]
pub enum VAArgOpVerifyErr {
    #[error("Operand must be a pointer type")]
    OperandNotPointer,
}

/// Extract next argument from a variadic argument list.
///
/// Equivalent to LLVM's `va_arg` instruction.
///
/// ### Operands
///
/// ```text
/// | operand | description          |
/// |---------|----------------------|
/// | `list`  | Pointer to va_list   |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description                       |
/// |--------|-----------------------------------|
/// | `res`  | Any type (the extracted argument) |
/// ```
#[def_op("llvm.va_arg")]
#[derive_op_interface_impl(NResultsInterface<1>, OneResultInterface, NOpdsInterface<1>, OneOpdInterface)]
#[pliron::derive::format_op("$0 ` : ` type($0)")]
pub struct VAArgOp;

impl Verify for VAArgOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);

        // Check that the argument is a pointer.
        let opd_ty = self.operand_type(ctx).deref(ctx);
        if !opd_ty.is::<PointerType>() {
            return verify_err!(loc, VAArgOpVerifyErr::OperandNotPointer);
        }

        Ok(())
    }
}

impl VAArgOp {
    /// Create a new [`VAArgOp`].
    ///
    /// # Arguments
    /// * `ctx` - The Pliron IR context
    /// * `list` - Pointer to the va_list
    /// * `ty` - Type of the argument to extract
    pub fn new(ctx: &mut Context, list: Value, ty: Ptr<TypeObj>) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![ty],
            vec![list],
            vec![],
            0,
        );
        Self { op }
    }
}

// ============================================================================
// Registration
// ============================================================================

/// Register all variadic argument operations.
pub fn register(ctx: &mut Context) {
    VAArgOp::register(ctx);
}
