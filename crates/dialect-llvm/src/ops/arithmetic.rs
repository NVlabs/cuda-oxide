/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integer and floating-point arithmetic operations.
//!
//! This module contains LLVM dialect operations for arithmetic computations:
//!
//! # Integer Arithmetic
//!
//! ```text
//! ┌───────────┬─────────────┬──────────────────────────┬────────────────┐
//! │ Operation │ LLVM Opcode │ Description              │ Overflow Flags │
//! ├───────────┼─────────────┼──────────────────────────┼────────────────┤
//! │ AddOp     │ add         │ Integer addition         │ Yes (nsw/nuw)  │
//! │ SubOp     │ sub         │ Integer subtraction      │ Yes (nsw/nuw)  │
//! │ MulOp     │ mul         │ Integer multiplication   │ Yes (nsw/nuw)  │
//! │ ShlOp     │ shl         │ Shift left               │ Yes (nsw/nuw)  │
//! │ UDivOp    │ udiv        │ Unsigned division        │ No             │
//! │ SDivOp    │ sdiv        │ Signed division          │ No             │
//! │ URemOp    │ urem        │ Unsigned remainder       │ No             │
//! │ SRemOp    │ srem        │ Signed remainder         │ No             │
//! │ LShrOp    │ lshr        │ Logical shift right      │ No             │
//! │ AShrOp    │ ashr        │ Arithmetic shift right   │ No             │
//! └───────────┴─────────────┴──────────────────────────┴────────────────┘
//! ```
//!
//! # Bitwise Operations
//!
//! ```text
//! ┌───────────┬─────────────┬─────────────┐
//! │ Operation │ LLVM Opcode │ Description │
//! ├───────────┼─────────────┼─────────────┤
//! │ AndOp     │ and         │ Bitwise AND │
//! │ OrOp      │ or          │ Bitwise OR  │
//! │ XorOp     │ xor         │ Bitwise XOR │
//! └───────────┴─────────────┴─────────────┘
//! ```
//!
//! # Floating-Point Arithmetic
//!
//! All floating-point operations support fast-math flags for optimization hints.
//!
//! ```text
//! ┌───────────┬─────────────┬─────────────────────────────────┐
//! │ Operation │ LLVM Opcode │ Description                     │
//! ├───────────┼─────────────┼─────────────────────────────────┤
//! │ FAddOp    │ fadd        │ Floating-point addition         │
//! │ FSubOp    │ fsub        │ Floating-point subtraction      │
//! │ FMulOp    │ fmul        │ Floating-point multiplication   │
//! │ FDivOp    │ fdiv        │ Floating-point division         │
//! │ FRemOp    │ frem        │ Floating-point remainder        │
//! │ FNegOp    │ fneg        │ Floating-point negation (unary) │
//! └───────────┴─────────────┴─────────────────────────────────┘
//! ```

use pliron::{
    builtin::{
        op_interfaces::{
            AtLeastNOpdsInterface, AtLeastNResultsInterface, NOpdsInterface, NResultsInterface,
            OneResultInterface, SameOperandsAndResultType, SameOperandsType, SameResultsType,
        },
        type_interfaces::FloatTypeInterface,
        types::IntegerType,
    },
    common_traits::Verify,
    context::Context,
    derive::pliron_op,
    op::Op,
    result::Result,
    r#type::{Typed as TypedTrait, type_impls},
    verify_err,
};

use crate::{
    attributes::FastmathFlagsAttr,
    op_interfaces::{
        BinArithOp, FastMathFlags, FloatBinArithOp, FloatBinArithOpWithFastMathFlags,
        IntBinArithOp, IntBinArithOpWithOverflowFlag,
    },
};

// ============================================================================
// Integer Binary Operations - Macros
// ============================================================================

macro_rules! new_int_bin_op_base {
    (   $(#[$outer:meta])*
        $op_name:ident, $op_id:literal, $fmt:literal
    ) => {
        $(#[$outer])*
        /// ### Operands:
        ///
        /// ```text
        /// | operand | description      |
        /// |---------|------------------|
        /// | `lhs`   | Signless integer |
        /// | `rhs`   | Signless integer |
        /// ```
        ///
        /// ### Result(s):
        ///
        /// ```text
        /// | result | description      |
        /// |--------|------------------|
        /// | `res`  | Signless integer |
        /// ```
        #[pliron::derive::pliron_op(
            name = $op_id,
            format = $fmt,
            interfaces = [
                NResultsInterface<1>, OneResultInterface, NOpdsInterface<2>,
                AtLeastNOpdsInterface<1>, AtLeastNResultsInterface<1>,
                SameOperandsType, SameResultsType,
                SameOperandsAndResultType, BinArithOp, IntBinArithOp
            ]
        )]
        pub struct $op_name;

        impl Verify for $op_name {
            fn verify(&self, ctx: &Context) -> Result<()> {
                use pliron::op::Op;
                let loc = self.loc(ctx);
                let op = self.get_operation().deref(ctx);
                let lhs = op.get_operand(0);
                let rhs = op.get_operand(1);
                let lhs_ty = TypedTrait::get_type(&lhs, ctx);
                let rhs_ty = TypedTrait::get_type(&rhs, ctx);
                let res_ty = OneResultInterface::result_type(self, ctx);

                if lhs_ty != rhs_ty {
                    return verify_err!(loc, "Operand types must match");
                }
                if lhs_ty != res_ty {
                    return verify_err!(loc, "Result type must match operands");
                }
                if !lhs_ty.deref(ctx).is::<IntegerType>() {
                     return verify_err!(loc, "Operands must be integer type");
                }
                Ok(())
            }
        }
    }
}

macro_rules! new_int_bin_op {
    (   $(#[$outer:meta])*
        $op_name:ident, $op_id:literal
    ) => {
        new_int_bin_op_base!(
            $(#[$outer])*
            $op_name,
            $op_id,
            "$0 `, ` $1 ` : ` type($0)"
        );
    }
}

macro_rules! new_int_bin_op_with_overflow {
    (   $(#[$outer:meta])*
        $op_name:ident, $op_id:literal
    ) => {
        new_int_bin_op_base!(
            $(#[$outer])*
            /// ### Attributes:
            ///
            /// ```text
            /// | key                                                                                         | value                                                                      | via Interface                     |
            /// |---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|-----------------------------------|
            /// | [ATTR_KEY_INTEGER_OVERFLOW_FLAGS](crate::op_interfaces::ATTR_KEY_INTEGER_OVERFLOW_FLAGS)    | [IntegerOverflowFlagsAttr](crate::attributes::IntegerOverflowFlagsAttr)    | [IntBinArithOpWithOverflowFlag]   |
            /// ```
            $op_name,
            $op_id,
            "$0 `, ` $1 ` <` attr($llvm_integer_overflow_flags, `super::super::attributes::IntegerOverflowFlagsAttr`) `>` `: ` type($0)"
        );
        #[pliron::derive::op_interface_impl]
        impl IntBinArithOpWithOverflowFlag for $op_name {}
    }
}

// ============================================================================
// Integer Binary Operations with Overflow Flags
// ============================================================================

new_int_bin_op_with_overflow!(
    /// Integer addition with optional overflow flags (nsw/nuw).
    ///
    /// Equivalent to LLVM's `add` instruction.
    AddOp,
    "llvm.add"
);

new_int_bin_op_with_overflow!(
    /// Integer subtraction with optional overflow flags (nsw/nuw).
    ///
    /// Equivalent to LLVM's `sub` instruction.
    SubOp,
    "llvm.sub"
);

new_int_bin_op_with_overflow!(
    /// Integer multiplication with optional overflow flags (nsw/nuw).
    ///
    /// Equivalent to LLVM's `mul` instruction.
    MulOp,
    "llvm.mul"
);

new_int_bin_op_with_overflow!(
    /// Left shift with optional overflow flags (nsw/nuw).
    ///
    /// Equivalent to LLVM's `shl` instruction.
    ShlOp,
    "llvm.shl"
);

// ============================================================================
// Integer Binary Operations without Overflow Flags
// ============================================================================

new_int_bin_op!(
    /// Unsigned integer division.
    ///
    /// Equivalent to LLVM's `udiv` instruction.
    UDivOp,
    "llvm.udiv"
);

new_int_bin_op!(
    /// Signed integer division.
    ///
    /// Equivalent to LLVM's `sdiv` instruction.
    SDivOp,
    "llvm.sdiv"
);

new_int_bin_op!(
    /// Unsigned integer remainder.
    ///
    /// Equivalent to LLVM's `urem` instruction.
    URemOp,
    "llvm.urem"
);

new_int_bin_op!(
    /// Signed integer remainder.
    ///
    /// Equivalent to LLVM's `srem` instruction.
    SRemOp,
    "llvm.srem"
);

new_int_bin_op!(
    /// Bitwise AND operation.
    ///
    /// Equivalent to LLVM's `and` instruction.
    AndOp,
    "llvm.and"
);

new_int_bin_op!(
    /// Bitwise OR operation.
    ///
    /// Equivalent to LLVM's `or` instruction.
    OrOp,
    "llvm.or"
);

new_int_bin_op!(
    /// Bitwise XOR operation.
    ///
    /// Equivalent to LLVM's `xor` instruction.
    XorOp,
    "llvm.xor"
);

new_int_bin_op!(
    /// Logical (unsigned) right shift.
    ///
    /// Equivalent to LLVM's `lshr` instruction.
    LShrOp,
    "llvm.lshr"
);

new_int_bin_op!(
    /// Arithmetic (signed) right shift.
    ///
    /// Equivalent to LLVM's `ashr` instruction.
    AShrOp,
    "llvm.ashr"
);

// ============================================================================
// Floating-Point Binary Operations - Macro
// ============================================================================

macro_rules! new_float_bin_op {
    (   $(#[$outer:meta])*
        $op_name:ident, $op_id:literal
    ) => {
        $(#[$outer])*
        /// ### Operands:
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
        /// | result | description |
        /// |--------|-------------|
        /// | `res`  | float       |
        /// ```
        #[pliron::derive::pliron_op(
            name = $op_id,
            format = "attr($llvm_fast_math_flags, $FastmathFlagsAttr) ` ` $0 `, ` $1 ` : ` type($0)",
            interfaces = [
                NResultsInterface<1>, OneResultInterface, NOpdsInterface<2>,
                AtLeastNOpdsInterface<1>, AtLeastNResultsInterface<1>,
                SameOperandsType, SameResultsType,
                SameOperandsAndResultType, BinArithOp, FloatBinArithOp,
                FloatBinArithOpWithFastMathFlags, FastMathFlags
            ]
        )]
        pub struct $op_name;

        impl Verify for $op_name {
            fn verify(&self, ctx: &Context) -> Result<()> {
                use pliron::op::Op;
                let loc = self.loc(ctx);
                let op = self.get_operation().deref(ctx);
                let lhs = op.get_operand(0);
                let rhs = op.get_operand(1);
                let lhs_ty = TypedTrait::get_type(&lhs, ctx);
                let rhs_ty = TypedTrait::get_type(&rhs, ctx);
                let res_ty = OneResultInterface::result_type(self, ctx);

                if lhs_ty != rhs_ty {
                    return verify_err!(loc, "Operand types must match");
                }
                if lhs_ty != res_ty {
                    return verify_err!(loc, "Result type must match operands");
                }
                if !type_impls::<dyn FloatTypeInterface>(&**lhs_ty.deref(ctx)) {
                    return verify_err!(loc, "Operands must be float type");
                }
                Ok(())
            }
        }
    }
}

// ============================================================================
// Floating-Point Binary Operations
// ============================================================================

new_float_bin_op! {
    /// Floating-point addition with optional fast-math flags.
    ///
    /// Equivalent to LLVM's `fadd` instruction.
    FAddOp,
    "llvm.fadd"
}

new_float_bin_op! {
    /// Floating-point subtraction with optional fast-math flags.
    ///
    /// Equivalent to LLVM's `fsub` instruction.
    FSubOp,
    "llvm.fsub"
}

new_float_bin_op! {
    /// Floating-point multiplication with optional fast-math flags.
    ///
    /// Equivalent to LLVM's `fmul` instruction.
    FMulOp,
    "llvm.fmul"
}

new_float_bin_op! {
    /// Floating-point division with optional fast-math flags.
    ///
    /// Equivalent to LLVM's `fdiv` instruction.
    FDivOp,
    "llvm.fdiv"
}

new_float_bin_op! {
    /// Floating-point remainder with optional fast-math flags.
    ///
    /// Equivalent to LLVM's `frem` instruction.
    FRemOp,
    "llvm.frem"
}

// ============================================================================
// Floating-Point Unary Operations
// ============================================================================

/// Floating-point negation with optional fast-math flags.
///
/// Equivalent to LLVM's `fneg` instruction.
///
/// ### Operands
///
/// ```text
/// | operand | description |
/// |---------|-------------|
/// | `arg`   | float       |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description |
/// |--------|-------------|
/// | `res`  | float       |
/// ```
#[pliron_op(
    name = "llvm.fneg",
    format = "attr($llvm_fast_math_flags, $FastmathFlagsAttr) $0 ` : ` type($0)",
    interfaces = [
        NResultsInterface<1>,
        OneResultInterface,
        pliron::builtin::op_interfaces::NOpdsInterface<1>,
        pliron::builtin::op_interfaces::OneOpdInterface,
        AtLeastNOpdsInterface<1>,
        AtLeastNResultsInterface<1>,
        SameResultsType,
        SameOperandsType,
        SameOperandsAndResultType,
        FastMathFlags
    ]
)]
pub struct FNegOp;

impl Verify for FNegOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        use pliron::op::Op;

        let loc = self.loc(ctx);
        let op = &*self.get_operation().deref(ctx);
        let arg_ty = op.get_operand(0).get_type(ctx);
        if !type_impls::<dyn FloatTypeInterface>(&**arg_ty.deref(ctx)) {
            return verify_err!(loc, FNegOpVerifyErr::ArgumentMustBeFloat);
        }
        Ok(())
    }
}

impl FNegOp {
    /// Create a new [`FNegOp`] with fast-math flags.
    pub fn new_with_fast_math_flags(
        ctx: &mut Context,
        arg: pliron::value::Value,
        fast_math_flags: FastmathFlagsAttr,
    ) -> Self {
        use pliron::operation::Operation;

        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![arg.get_type(ctx)],
            vec![arg],
            vec![],
            0,
        );
        let op = Self { op };
        op.set_fast_math_flags(ctx, fast_math_flags);
        op
    }
}

/// Verification errors for [`FNegOp`].
#[derive(thiserror::Error, Debug)]
pub enum FNegOpVerifyErr {
    #[error("Argument must be a float")]
    ArgumentMustBeFloat,
    #[error("Fast math flags must be set")]
    FastMathFlagsMustBeSet,
}

// ============================================================================
// Registration
// ============================================================================

/// Register all arithmetic operations.
pub fn register(ctx: &mut Context) {
    // Integer ops with overflow flags
    AddOp::register(ctx);
    SubOp::register(ctx);
    MulOp::register(ctx);
    ShlOp::register(ctx);

    // Integer ops without overflow flags
    UDivOp::register(ctx);
    SDivOp::register(ctx);
    URemOp::register(ctx);
    SRemOp::register(ctx);
    AndOp::register(ctx);
    OrOp::register(ctx);
    XorOp::register(ctx);
    LShrOp::register(ctx);
    AShrOp::register(ctx);

    // Float ops
    FAddOp::register(ctx);
    FSubOp::register(ctx);
    FMulOp::register(ctx);
    FDivOp::register(ctx);
    FRemOp::register(ctx);
    FNegOp::register(ctx);
}
