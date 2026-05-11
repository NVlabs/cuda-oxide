/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Comparison operations for integers and floating-point values.
//!
//! This module contains LLVM dialect operations for comparing values:
//!
//! ```text
//! ┌───────────┬─────────────┬──────────────────────────────────────────┐
//! │ Operation │ LLVM Opcode │ Description                              │
//! ├───────────┼─────────────┼──────────────────────────────────────────┤
//! │ ICmpOp    │ icmp        │ Integer/pointer comparison (returns i1)  │
//! │ FCmpOp    │ fcmp        │ Floating-point comparison (returns i1)   │
//! └───────────┴─────────────┴──────────────────────────────────────────┘
//! ```
//!
//! Both operations return an `i1` (1-bit integer) result representing boolean true/false.

use pliron::{
    builtin::{
        op_interfaces::{
            AtLeastNOpdsInterface, NOpdsInterface, NResultsInterface, OneResultInterface,
            SameOperandsType,
        },
        type_interfaces::FloatTypeInterface,
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::Context,
    derive::pliron_op,
    location::Located,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::{TypePtr, type_impls},
    value::Value,
    verify_err,
};

use crate::{
    attributes::{FCmpPredicateAttr, FastmathFlagsAttr, ICmpPredicateAttr},
    op_interfaces::FastMathFlags,
    types::PointerType,
};

// ============================================================================
// Integer Comparison
// ============================================================================

/// Verification errors for [`ICmpOp`].
#[derive(thiserror::Error, Debug)]
pub enum ICmpOpVerifyErr {
    #[error("Result must be 1-bit integer (bool)")]
    ResultNotBool,
    #[error("Operand must be integer or pointer types")]
    IncorrectOperandsType,
    #[error("Missing or incorrect predicate attribute")]
    PredAttrErr,
}

/// Integer comparison operation.
///
/// Compares two integer or pointer values and returns an `i1` result.
/// The comparison predicate determines the type of comparison (eq, ne, slt, sgt, etc.).
///
/// Equivalent to LLVM's `icmp` instruction.
///
/// ### Operand(s):
///
/// ```text
/// | operand | description                 |
/// |---------|-----------------------------|
/// | `lhs`   | Signless integer or pointer |
/// | `rhs`   | Signless integer or pointer |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description            |
/// |--------|------------------------|
/// | `res`  | 1-bit signless integer |
/// ```
#[pliron_op(
    name = "llvm.icmp",
    format = "$0 ` <` attr($icmp_predicate, $ICmpPredicateAttr) `> ` $1 ` : ` type($0)",
    interfaces = [SameOperandsType, AtLeastNOpdsInterface<1>, NResultsInterface<1>, OneResultInterface, NOpdsInterface<2>],
    attributes = (icmp_predicate: ICmpPredicateAttr)
)]
pub struct ICmpOp;

impl ICmpOp {
    /// Create a new [`ICmpOp`].
    pub fn new(ctx: &mut Context, pred: ICmpPredicateAttr, lhs: Value, rhs: Value) -> Self {
        let bool_ty = IntegerType::get(ctx, 1, Signedness::Signless);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![bool_ty.into()],
            vec![lhs, rhs],
            vec![],
            0,
        );
        let op = Self { op };
        op.set_attr_icmp_predicate(ctx, pred);
        op
    }

    /// Get the comparison predicate.
    #[must_use]
    pub fn predicate(&self, ctx: &Context) -> ICmpPredicateAttr {
        self.get_attr_icmp_predicate(ctx)
            .expect("ICmpOp missing or incorrect predicate attribute type")
            .clone()
    }
}

impl Verify for ICmpOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);

        if self.get_attr_icmp_predicate(ctx).is_none() {
            verify_err!(loc.clone(), ICmpOpVerifyErr::PredAttrErr)?;
        }

        let res_ty: TypePtr<IntegerType> =
            TypePtr::from_ptr(self.result_type(ctx), ctx).map_err(|mut err| {
                err.set_loc(loc.clone());
                err
            })?;

        if res_ty.deref(ctx).width() != 1 {
            return verify_err!(loc, ICmpOpVerifyErr::ResultNotBool);
        }

        let opd_ty = self.operand_type(ctx).deref(ctx);
        if !(opd_ty.is::<IntegerType>() || opd_ty.is::<PointerType>()) {
            return verify_err!(loc, ICmpOpVerifyErr::IncorrectOperandsType);
        }

        Ok(())
    }
}

// ============================================================================
// Floating-Point Comparison
// ============================================================================

/// Verification errors for [`FCmpOp`].
#[derive(thiserror::Error, Debug)]
pub enum FCmpOpVerifyErr {
    #[error("Result must be 1-bit integer (bool)")]
    ResultNotBool,
    #[error("Operand must be floating point type")]
    IncorrectOperandsType,
    #[error("Missing or incorrect predicate attribute")]
    PredAttrErr,
}

/// Floating-point comparison operation.
///
/// Compares two floating-point values and returns an `i1` result.
/// The comparison predicate determines the type of comparison (oeq, ogt, olt, etc.).
/// Supports fast-math flags for optimization hints.
///
/// Equivalent to LLVM's `fcmp` instruction.
///
/// ### Operand(s):
///
/// ```text
/// | operand | description |
/// |---------|-------------|
/// | `lhs`   | float       |
/// | `rhs`   | float       |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description            |
/// |--------|------------------------|
/// | `res`  | 1-bit signless integer |
/// ```
#[pliron_op(
    name = "llvm.fcmp",
    format = "attr($llvm_fast_math_flags, $FastmathFlagsAttr) ` ` $0 ` <` attr($fcmp_predicate, $FCmpPredicateAttr) `> ` $1 ` : ` type($0)",
    interfaces = [NResultsInterface<1>, OneResultInterface, SameOperandsType, AtLeastNOpdsInterface<1>, NOpdsInterface<2>, FastMathFlags],
    attributes = (fcmp_predicate: FCmpPredicateAttr)
)]
pub struct FCmpOp;

impl FCmpOp {
    /// Create a new [`FCmpOp`].
    pub fn new(ctx: &mut Context, pred: FCmpPredicateAttr, lhs: Value, rhs: Value) -> Self {
        let bool_ty = IntegerType::get(ctx, 1, Signedness::Signless);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![bool_ty.into()],
            vec![lhs, rhs],
            vec![],
            0,
        );
        let op = Self { op };
        op.set_attr_fcmp_predicate(ctx, pred);
        // Set default (empty) fast math flags - required by format_op
        op.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
        op
    }

    /// Get the comparison predicate.
    #[must_use]
    pub fn predicate(&self, ctx: &Context) -> FCmpPredicateAttr {
        self.get_attr_fcmp_predicate(ctx)
            .expect("FCmpOp missing or incorrect predicate attribute type")
            .clone()
    }
}

impl Verify for FCmpOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);

        if self.get_attr_fcmp_predicate(ctx).is_none() {
            verify_err!(loc.clone(), FCmpOpVerifyErr::PredAttrErr)?;
        }

        let res_ty: TypePtr<IntegerType> =
            TypePtr::from_ptr(self.result_type(ctx), ctx).map_err(|mut err| {
                err.set_loc(loc.clone());
                err
            })?;

        if res_ty.deref(ctx).width() != 1 {
            return verify_err!(loc, FCmpOpVerifyErr::ResultNotBool);
        }

        let opd_ty = self.operand_type(ctx).deref(ctx);
        if !(type_impls::<dyn FloatTypeInterface>(&**opd_ty)) {
            return verify_err!(loc, FCmpOpVerifyErr::IncorrectOperandsType);
        }

        Ok(())
    }
}

// ============================================================================
// Registration
// ============================================================================

/// Register all comparison operations.
pub fn register(ctx: &mut Context) {
    ICmpOp::register(ctx);
    FCmpOp::register(ctx);
}
