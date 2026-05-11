/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Constant value operations.
//!
//! This module contains LLVM dialect operations for creating constant values:
//!
//! ```text
//! ┌────────────┬──────────────────────────────────────────────────────────┐
//! │ Operation  │ Description                                              │
//! ├────────────┼──────────────────────────────────────────────────────────┤
//! │ ConstantOp │ Creates a numeric constant (integer or float)            │
//! │ ZeroOp     │ Creates a zero-initialized value of any type             │
//! │ UndefOp    │ Creates an undefined value (for optimization purposes)   │
//! └────────────┴──────────────────────────────────────────────────────────┘
//! ```

use pliron::{
    attribute::{AttrObj, attr_cast, attr_impls},
    builtin::{
        attr_interfaces::TypedAttrInterface,
        attributes::IntegerAttr,
        op_interfaces::{NOpdsInterface, NResultsInterface, OneResultInterface},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    derive::{def_op, derive_attr_get_set, derive_op_interface_impl},
    op::Op,
    operation::Operation,
    result::Result,
    r#type::TypeObj,
    verify_err,
};
use pliron_derive::verify_succ;

// ============================================================================
// Constant Operation
// ============================================================================

/// Verification errors for [`ConstantOp`].
#[derive(thiserror::Error, Debug)]
#[error("{}: Unexpected type", ConstantOp::get_opid_static())]
pub enum ConstantOpVerifyErr {
    #[error("ConstantOp must have either an integer or a float value")]
    InvalidValue,
}

/// Numeric (integer or floating-point) constant.
///
/// See upstream MLIR's [llvm.mlir.constant](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlirconstant-llvmconstantop).
///
/// ### Results:
///
/// ```text
/// | result   | description |
/// |----------|-------------|
/// | `result` | any type    |
/// ```
#[def_op("llvm.constant")]
#[derive_op_interface_impl(NOpdsInterface<0>, NResultsInterface<1>, OneResultInterface)]
#[pliron::derive::format_op("`<` $constant_value `>` ` : ` type($0)")]
#[derive_attr_get_set(constant_value)]
pub struct ConstantOp;

impl ConstantOp {
    /// Get the constant value that this Op defines.
    #[must_use]
    pub fn get_value(&self, ctx: &Context) -> AttrObj {
        self.get_attr_constant_value(ctx).unwrap().clone()
    }

    /// Create a new [`ConstantOp`].
    pub fn new(ctx: &mut Context, value: AttrObj) -> Self {
        let result_type = attr_cast::<dyn TypedAttrInterface>(&*value)
            .expect("ConstantOp const value must provide TypedAttrInterface")
            .get_type(ctx);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type],
            vec![],
            vec![],
            0,
        );
        let op = Self { op };
        op.set_attr_constant_value(ctx, value);
        op
    }
}

impl Verify for ConstantOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        use pliron::builtin::attr_interfaces::FloatAttr;

        let loc = self.loc(ctx);
        let value = self.get_value(ctx);
        if !(value.is::<IntegerAttr>() || attr_impls::<dyn FloatAttr>(&*value)) {
            return verify_err!(loc, ConstantOpVerifyErr::InvalidValue)?;
        }
        Ok(())
    }
}

// ============================================================================
// Zero Operation
// ============================================================================

/// Creates a zero-initialized value of the specified LLVM IR dialect type.
///
/// Same as upstream MLIR's LLVM dialect [ZeroOp](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlirzero-llvmzeroop).
///
/// ### Results:
///
/// ```text
/// | result   | description |
/// |----------|-------------|
/// | `result` | any type    |
/// ```
#[def_op("llvm.zero")]
#[derive_op_interface_impl(NOpdsInterface<0>, NResultsInterface<1>, OneResultInterface)]
#[pliron::derive::format_op("`: ` type($0)")]
#[verify_succ]
pub struct ZeroOp;

impl ZeroOp {
    /// Create a new [`ZeroOp`].
    pub fn new(ctx: &mut Context, result_ty: Ptr<TypeObj>) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty],
            vec![],
            vec![],
            0,
        );
        Self { op }
    }
}

// ============================================================================
// Undef Operation
// ============================================================================

/// Undefined value of a type.
///
/// Represents an undefined value, which LLVM can use for optimization.
/// Reading an undef value yields an arbitrary bit pattern.
///
/// See upstream MLIR's [llvm.mlir.undef](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlirundef-llvmundefop).
///
/// ### Results:
///
/// ```text
/// | result   | description |
/// |----------|-------------|
/// | `result` | any type    |
/// ```
#[def_op("llvm.undef")]
#[derive_op_interface_impl(NResultsInterface<1>, OneResultInterface, NOpdsInterface<0>)]
#[pliron::derive::format_op("`: ` type($0)")]
#[verify_succ]
pub struct UndefOp;

impl UndefOp {
    /// Create a new [`UndefOp`].
    pub fn new(ctx: &mut Context, result_ty: Ptr<TypeObj>) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty],
            vec![],
            vec![],
            0,
        );
        Self { op }
    }
}

// ============================================================================
// Registration
// ============================================================================

/// Register all constant operations.
pub fn register(ctx: &mut Context) {
    ConstantOp::register(ctx);
    ZeroOp::register(ctx);
    UndefOp::register(ctx);
}
