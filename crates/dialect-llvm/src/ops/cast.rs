/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Type conversion (cast) operations.
//!
//! This module contains LLVM dialect operations for converting between types:
//!
//! # Integer Casts
//!
//! ```text
//! ┌───────────┬─────────────┬─────────────────────────────────────────┐
//! │ Operation │ LLVM Opcode │ Description                             │
//! ├───────────┼─────────────┼─────────────────────────────────────────┤
//! │ SExtOp    │ sext        │ Sign-extend integer (smaller → larger)  │
//! │ ZExtOp    │ zext        │ Zero-extend integer (smaller → larger)  │
//! │ TruncOp   │ trunc       │ Truncate integer (larger → smaller)     │
//! └───────────┴─────────────┴─────────────────────────────────────────┘
//! ```
//!
//! # Floating-Point Casts
//!
//! ```text
//! ┌───────────┬─────────────┬───────────────────────────────────────────┐
//! │ Operation │ LLVM Opcode │ Description                               │
//! ├───────────┼─────────────┼───────────────────────────────────────────┤
//! │ FPExtOp   │ fpext       │ Extend float precision (smaller → larger) │
//! │ FPTruncOp │ fptrunc     │ Truncate float precision (larger → smaller)│
//! └───────────┴─────────────┴───────────────────────────────────────────┘
//! ```
//!
//! # Integer ↔ Float Conversions
//!
//! ```text
//! ┌───────────┬─────────────┬────────────────────────────┐
//! │ Operation │ LLVM Opcode │ Description                │
//! ├───────────┼─────────────┼────────────────────────────┤
//! │ FPToSIOp  │ fptosi      │ Float → signed integer     │
//! │ FPToUIOp  │ fptoui      │ Float → unsigned integer   │
//! │ SIToFPOp  │ sitofp      │ Signed integer → float     │
//! │ UIToFPOp  │ uitofp      │ Unsigned integer → float   │
//! └───────────┴─────────────┴────────────────────────────┘
//! ```
//!
//! # Pointer/Bitwise Casts
//!
//! ```text
//! ┌────────────────┬───────────────┬────────────────────────────────────┐
//! │ Operation      │ LLVM Opcode   │ Description                        │
//! ├────────────────┼───────────────┼────────────────────────────────────┤
//! │ BitcastOp      │ bitcast       │ Reinterpret bits (same size types) │
//! │ IntToPtrOp     │ inttoptr      │ Integer → pointer                  │
//! │ PtrToIntOp     │ ptrtoint      │ Pointer → integer                  │
//! │ AddrSpaceCast  │ addrspacecast │ Pointer address space conversion   │
//! └────────────────┴───────────────┴────────────────────────────────────┘
//! ```

use pliron::{
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface, OneOpdInterface, OneResultInterface},
        type_interfaces::FloatTypeInterface,
        types::IntegerType,
    },
    common_traits::Verify,
    context::Context,
    derive::pliron_op,
    location::Located,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::{type_cast, type_impls},
    value::Value,
    verify_err,
};

use crate::{
    attributes::ICmpPredicateAttr,
    op_interfaces::{CastOpInterface, CastOpWithNNegInterface, FastMathFlags, NNegFlag},
    types::PointerType,
};

// ============================================================================
// Pointer/Bitwise Casts
// ============================================================================

/// Bitcast operation - reinterpret bits without changing them.
///
/// Equivalent to LLVM's `bitcast` instruction.
///
/// ### Operands
///
/// ```text
/// | operand | description             |
/// |---------|-------------------------|
/// | `arg`   | non-aggregate LLVM type |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description             |
/// |--------|-------------------------|
/// | `res`  | non-aggregate LLVM type |
/// ```
#[pliron_op(
    name = "llvm.bitcast",
    format = "$0 ` to ` type($0)",
    interfaces = [NResultsInterface<1>, OneResultInterface, NOpdsInterface<1>, OneOpdInterface, CastOpInterface],
    verifier = "succ"
)]
pub struct BitcastOp;

/// Verification errors for [`IntToPtrOp`].
#[derive(thiserror::Error, Debug)]
pub enum IntToPtrOpErr {
    #[error("Operand must be a signless integer")]
    OperandTypeErr,
    #[error("Result must be a pointer type")]
    ResultTypeErr,
}

/// Integer to pointer conversion.
///
/// Equivalent to LLVM's `inttoptr` instruction.
///
/// ### Operands
///
/// ```text
/// | operand | description      |
/// |---------|------------------|
/// | `arg`   | Signless integer |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description   |
/// |--------|---------------|
/// | `res`  | [PointerType] |
/// ```
#[pliron_op(
    name = "llvm.inttoptr",
    format = "$0 ` to ` type($0)",
    interfaces = [NResultsInterface<1>, OneResultInterface, NOpdsInterface<1>, OneOpdInterface, CastOpInterface]
)]
pub struct IntToPtrOp;

impl Verify for IntToPtrOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        if !self.operand_type(ctx).deref(ctx).is::<IntegerType>() {
            return verify_err!(loc, IntToPtrOpErr::OperandTypeErr);
        }
        if !self.result_type(ctx).deref(ctx).is::<PointerType>() {
            return verify_err!(loc, IntToPtrOpErr::ResultTypeErr);
        }
        Ok(())
    }
}

/// Verification errors for [`PtrToIntOp`].
#[derive(thiserror::Error, Debug)]
pub enum PtrToIntOpErr {
    #[error("Operand must be a pointer type")]
    OperandTypeErr,
    #[error("Result must be a signless integer type")]
    ResultTypeErr,
}

/// Pointer to integer conversion.
///
/// Equivalent to LLVM's `ptrtoint` instruction.
///
/// ### Operands
///
/// ```text
/// | operand | description     |
/// |---------|-----------------|
/// | `arg`   | [`PointerType`] |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description      |
/// |--------|------------------|
/// | `res`  | Signless integer |
/// ```
#[pliron_op(
    name = "llvm.ptrtoint",
    format = "$0 ` to ` type($0)",
    interfaces = [NResultsInterface<1>, OneResultInterface, NOpdsInterface<1>, OneOpdInterface, CastOpInterface]
)]
pub struct PtrToIntOp;

impl Verify for PtrToIntOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        if !self.operand_type(ctx).deref(ctx).is::<PointerType>() {
            return verify_err!(loc, PtrToIntOpErr::OperandTypeErr);
        }
        if !self.result_type(ctx).deref(ctx).is::<IntegerType>() {
            return verify_err!(loc, PtrToIntOpErr::ResultTypeErr);
        }
        Ok(())
    }
}

/// Verification errors for [`AddrSpaceCastOp`].
#[derive(thiserror::Error, Debug)]
pub enum AddrSpaceCastOpErr {
    #[error("Operand must be a pointer type")]
    OperandTypeErr,
    #[error("Result must be a pointer type")]
    ResultTypeErr,
}

/// Address space cast - convert pointer between address spaces.
///
/// Casts a pointer from one address space to another without changing
/// the bit representation. Used for casting between generic (0) and
/// specific address spaces (shared=3, global=1, etc).
///
/// Equivalent to LLVM's `addrspacecast` instruction.
///
/// ### Operands
///
/// ```text
/// | operand | description                           |
/// |---------|---------------------------------------|
/// | `arg`   | [PointerType] in source address space |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description                           |
/// |--------|---------------------------------------|
/// | `res`  | [PointerType] in target address space |
/// ```
#[pliron_op(
    name = "llvm.addrspacecast",
    format = "$0 ` to ` type($0)",
    interfaces = [NResultsInterface<1>, OneResultInterface, NOpdsInterface<1>, OneOpdInterface, CastOpInterface]
)]
pub struct AddrSpaceCastOp;

impl AddrSpaceCastOp {
    /// Create an address space cast from a pointer to a different address space.
    pub fn new(ctx: &mut Context, operand: Value, target_addrspace: u32) -> Self {
        let result_ty = PointerType::get(ctx, target_addrspace);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty.into()],
            vec![operand],
            vec![],
            0,
        );
        Self { op }
    }
}

impl Verify for AddrSpaceCastOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        if !self.operand_type(ctx).deref(ctx).is::<PointerType>() {
            return verify_err!(loc, AddrSpaceCastOpErr::OperandTypeErr);
        }
        if !self.result_type(ctx).deref(ctx).is::<PointerType>() {
            return verify_err!(loc, AddrSpaceCastOpErr::ResultTypeErr);
        }
        Ok(())
    }
}

// ============================================================================
// Integer Casts
// ============================================================================

/// Verification errors for integer casts.
#[derive(thiserror::Error, Debug)]
enum IntCastVerifyErr {
    #[error("Result must be an integer")]
    ResultTypeErr,
    #[error("Operand must be an integer")]
    OperandTypeErr,
    #[error("Result type must be larger than operand type")]
    ResultTypeSmallerThanOperand,
    #[error("Result type must be smaller than operand type")]
    ResultTypeLargerThanOperand,
    #[error("Result type must be equal to operand type")]
    ResultTypeEqualToOperand,
}

/// Ensure that the integer cast operation is valid.
///
/// Checks that both operand and result are integers, and that their widths
/// satisfy the relationship specified by `cmp`.
fn integer_cast_verify(
    op: &pliron::operation::Operation,
    ctx: &Context,
    cmp: ICmpPredicateAttr,
) -> Result<()> {
    use pliron::r#type::Typed;

    let loc = op.loc();
    let res_ty = op.get_type(0).deref(ctx);
    let opd_ty = op.get_operand(0).get_type(ctx).deref(ctx);
    let Some(res_ty) = res_ty.downcast_ref::<IntegerType>() else {
        return verify_err!(loc, IntCastVerifyErr::ResultTypeErr);
    };
    let Some(opd_ty) = opd_ty.downcast_ref::<IntegerType>() else {
        return verify_err!(loc, IntCastVerifyErr::OperandTypeErr);
    };

    match cmp {
        ICmpPredicateAttr::SLT | ICmpPredicateAttr::ULT => {
            if res_ty.width() >= opd_ty.width() {
                return verify_err!(loc, IntCastVerifyErr::ResultTypeLargerThanOperand);
            }
        }
        ICmpPredicateAttr::SGT | ICmpPredicateAttr::UGT => {
            if res_ty.width() <= opd_ty.width() {
                return verify_err!(loc, IntCastVerifyErr::ResultTypeSmallerThanOperand);
            }
        }
        ICmpPredicateAttr::SLE | ICmpPredicateAttr::ULE => {
            if res_ty.width() > opd_ty.width() {
                return verify_err!(loc, IntCastVerifyErr::ResultTypeLargerThanOperand);
            }
        }
        ICmpPredicateAttr::SGE | ICmpPredicateAttr::UGE => {
            if res_ty.width() < opd_ty.width() {
                return verify_err!(loc, IntCastVerifyErr::ResultTypeSmallerThanOperand);
            }
        }
        ICmpPredicateAttr::EQ | ICmpPredicateAttr::NE => {
            if res_ty.width() != opd_ty.width() {
                return verify_err!(loc, IntCastVerifyErr::ResultTypeEqualToOperand);
            }
        }
    }
    Ok(())
}

/// Sign-extend integer to larger width.
///
/// Equivalent to LLVM's `sext` instruction.
///
/// ### Operands
///
/// ```text
/// | operand | description      |
/// |---------|------------------|
/// | `arg`   | Signless integer |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description                     |
/// |--------|---------------------------------|
/// | `res`  | Signless integer (larger width) |
/// ```
#[pliron_op(
    name = "llvm.sext",
    format = "$0 ` to ` type($0)",
    interfaces = [CastOpInterface, NResultsInterface<1>, OneResultInterface, NOpdsInterface<1>, OneOpdInterface]
)]
pub struct SExtOp;

impl Verify for SExtOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        integer_cast_verify(
            &self.get_operation().deref(ctx),
            ctx,
            ICmpPredicateAttr::SGT,
        )
    }
}

/// Zero-extend integer to larger width.
///
/// Equivalent to LLVM's `zext` instruction. Supports the `nneg` flag
/// indicating the source is known to be non-negative.
///
/// ### Operands
///
/// ```text
/// | operand | description      |
/// |---------|------------------|
/// | `arg`   | Signless integer |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description                     |
/// |--------|---------------------------------|
/// | `res`  | Signless integer (larger width) |
/// ```
#[pliron_op(
    name = "llvm.zext",
    format = "`<nneg=` attr($llvm_nneg_flag, `pliron::builtin::attributes::BoolAttr`) `> ` $0 ` to ` type($0)",
    interfaces = [
        CastOpInterface,
        NResultsInterface<1>,
        OneResultInterface,
        NOpdsInterface<1>,
        OneOpdInterface,
        NNegFlag,
        CastOpWithNNegInterface
    ]
)]
pub struct ZExtOp;

impl Verify for ZExtOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        integer_cast_verify(
            &self.get_operation().deref(ctx),
            ctx,
            ICmpPredicateAttr::UGT,
        )
    }
}

/// Truncate integer to smaller width.
///
/// Equivalent to LLVM's `trunc` instruction.
///
/// ### Operands
///
/// ```text
/// | operand | description      |
/// |---------|------------------|
/// | `arg`   | Signless integer |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description                      |
/// |--------|----------------------------------|
/// | `res`  | Signless integer (smaller width) |
/// ```
#[pliron_op(
    name = "llvm.trunc",
    format = "$0 ` to ` type($0)",
    interfaces = [CastOpInterface, NResultsInterface<1>, OneResultInterface, NOpdsInterface<1>, OneOpdInterface]
)]
pub struct TruncOp;

impl Verify for TruncOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        integer_cast_verify(
            &self.get_operation().deref(ctx),
            ctx,
            ICmpPredicateAttr::ULT,
        )
    }
}

// ============================================================================
// Floating-Point Casts
// ============================================================================

/// Verification errors for floating-point casts.
#[derive(thiserror::Error, Debug)]
pub enum FloatCastVerifyErr {
    #[error("Incorrect operand type")]
    OperandTypeErr,
    #[error("Incorrect result type")]
    ResultTypeErr,
    #[error("Result type must be bigger than the operand type")]
    ResultTypeSmallerThanOperand,
    #[error("Operand type must be bigger than the result type")]
    OperandTypeSmallerThanResult,
}

/// Extend floating-point precision.
///
/// Equivalent to LLVM's `fpext` instruction.
///
/// ### Operands
///
/// ```text
/// | operand | description           |
/// |---------|-----------------------|
/// | `arg`   | Floating-point number |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description                             |
/// |--------|-----------------------------------------|
/// | `res`  | Floating-point number (larger precision)|
/// ```
#[pliron_op(
    name = "llvm.fpext",
    format = "attr($llvm_fast_math_flags, `super::super::attributes::FastmathFlagsAttr`) ` ` $0 ` to ` type($0)",
    interfaces = [CastOpInterface, NResultsInterface<1>, OneResultInterface, NOpdsInterface<1>, OneOpdInterface, FastMathFlags]
)]
pub struct FPExtOp;

impl Verify for FPExtOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let opd_ty = OneOpdInterface::operand_type(self, ctx).deref(ctx);
        let Some(opd_float_ty) = type_cast::<dyn FloatTypeInterface>(&**opd_ty) else {
            return verify_err!(self.loc(ctx), FloatCastVerifyErr::OperandTypeErr);
        };
        let res_ty = OneResultInterface::result_type(self, ctx).deref(ctx);
        let Some(res_float_ty) = type_cast::<dyn FloatTypeInterface>(&**res_ty) else {
            return verify_err!(self.loc(ctx), FloatCastVerifyErr::ResultTypeErr);
        };

        let opd_size = opd_float_ty.get_semantics().bits;
        let res_size = res_float_ty.get_semantics().bits;
        if res_size <= opd_size {
            return verify_err!(
                self.loc(ctx),
                FloatCastVerifyErr::ResultTypeSmallerThanOperand
            );
        }
        Ok(())
    }
}

/// Truncate floating-point precision.
///
/// Equivalent to LLVM's `fptrunc` instruction.
///
/// ### Operands
///
/// ```text
/// | operand | description           |
/// |---------|-----------------------|
/// | `arg`   | Floating-point number |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description                               |
/// |--------|-------------------------------------------|
/// | `res`  | Floating-point number (smaller precision) |
/// ```
#[pliron_op(
    name = "llvm.fptrunc",
    format = "attr($llvm_fast_math_flags, `super::super::attributes::FastmathFlagsAttr`) ` ` $0 ` to ` type($0)",
    interfaces = [CastOpInterface, NResultsInterface<1>, OneResultInterface, NOpdsInterface<1>, OneOpdInterface, FastMathFlags]
)]
pub struct FPTruncOp;

impl Verify for FPTruncOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let opd_ty = OneOpdInterface::operand_type(self, ctx).deref(ctx);
        let Some(opd_float_ty) = type_cast::<dyn FloatTypeInterface>(&**opd_ty) else {
            return verify_err!(self.loc(ctx), FloatCastVerifyErr::OperandTypeErr);
        };
        let res_ty = OneResultInterface::result_type(self, ctx).deref(ctx);
        let Some(res_float_ty) = type_cast::<dyn FloatTypeInterface>(&**res_ty) else {
            return verify_err!(self.loc(ctx), FloatCastVerifyErr::ResultTypeErr);
        };

        let opd_size = opd_float_ty.get_semantics().bits;
        let res_size = res_float_ty.get_semantics().bits;
        if opd_size <= res_size {
            return verify_err!(
                self.loc(ctx),
                FloatCastVerifyErr::OperandTypeSmallerThanResult
            );
        }
        Ok(())
    }
}

// ============================================================================
// Integer ↔ Float Conversions
// ============================================================================

/// Convert floating-point to signed integer.
///
/// Equivalent to LLVM's `fptosi` instruction.
///
/// ### Operands
///
/// ```text
/// | operand | description           |
/// |---------|-----------------------|
/// | `arg`   | Floating-point number |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description    |
/// |--------|----------------|
/// | `res`  | Signed integer |
/// ```
#[pliron_op(
    name = "llvm.fptosi",
    format = "$0 ` to ` type($0)",
    interfaces = [CastOpInterface, NResultsInterface<1>, OneResultInterface, NOpdsInterface<1>, OneOpdInterface]
)]
pub struct FPToSIOp;

impl Verify for FPToSIOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let opd_ty = OneOpdInterface::operand_type(self, ctx).deref(ctx);
        if !type_impls::<dyn FloatTypeInterface>(&**opd_ty) {
            return verify_err!(self.loc(ctx), FloatCastVerifyErr::OperandTypeErr);
        }
        let res_ty = OneResultInterface::result_type(self, ctx).deref(ctx);
        let Some(res_int_ty) = res_ty.downcast_ref::<IntegerType>() else {
            return verify_err!(self.loc(ctx), FloatCastVerifyErr::ResultTypeErr);
        };
        if !res_int_ty.is_signless() {
            return verify_err!(self.loc(ctx), FloatCastVerifyErr::ResultTypeErr);
        }
        Ok(())
    }
}

/// Convert floating-point to unsigned integer.
///
/// Equivalent to LLVM's `fptoui` instruction.
///
/// ### Operands
///
/// ```text
/// | operand | description           |
/// |---------|-----------------------|
/// | `arg`   | Floating-point number |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description      |
/// |--------|------------------|
/// | `res`  | Unsigned integer |
/// ```
#[pliron_op(
    name = "llvm.fptoui",
    format = "$0 ` to ` type($0)",
    interfaces = [CastOpInterface, NResultsInterface<1>, OneResultInterface, NOpdsInterface<1>, OneOpdInterface]
)]
pub struct FPToUIOp;

impl Verify for FPToUIOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let opd_ty = OneOpdInterface::operand_type(self, ctx).deref(ctx);
        if !type_impls::<dyn FloatTypeInterface>(&**opd_ty) {
            return verify_err!(self.loc(ctx), FloatCastVerifyErr::OperandTypeErr);
        }
        let res_ty = OneResultInterface::result_type(self, ctx).deref(ctx);
        let Some(res_int_ty) = res_ty.downcast_ref::<IntegerType>() else {
            return verify_err!(self.loc(ctx), FloatCastVerifyErr::ResultTypeErr);
        };
        if !res_int_ty.is_signless() {
            return verify_err!(self.loc(ctx), FloatCastVerifyErr::ResultTypeErr);
        }
        Ok(())
    }
}

/// Convert signed integer to floating-point.
///
/// Equivalent to LLVM's `sitofp` instruction.
///
/// ### Operands
///
/// ```text
/// | operand | description    |
/// |---------|----------------|
/// | `arg`   | Signed integer |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description           |
/// |--------|-----------------------|
/// | `res`  | Floating-point number |
/// ```
#[pliron_op(
    name = "llvm.sitofp",
    format = "$0 ` to ` type($0)",
    interfaces = [CastOpInterface, NResultsInterface<1>, OneResultInterface, NOpdsInterface<1>, OneOpdInterface]
)]
pub struct SIToFPOp;

impl Verify for SIToFPOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let opd_ty = OneOpdInterface::operand_type(self, ctx).deref(ctx);
        let Some(opd_ty_int) = opd_ty.downcast_ref::<IntegerType>() else {
            return verify_err!(self.loc(ctx), FloatCastVerifyErr::OperandTypeErr);
        };
        if !opd_ty_int.is_signless() {
            return verify_err!(self.loc(ctx), FloatCastVerifyErr::OperandTypeErr);
        }
        let res_ty = OneResultInterface::result_type(self, ctx).deref(ctx);
        if !type_impls::<dyn FloatTypeInterface>(&**res_ty) {
            return verify_err!(self.loc(ctx), FloatCastVerifyErr::ResultTypeErr);
        }
        Ok(())
    }
}

/// Convert unsigned integer to floating-point.
///
/// Equivalent to LLVM's `uitofp` instruction. Supports the `nneg` flag
/// indicating the source is known to be non-negative.
///
/// ### Operands
///
/// ```text
/// | operand | description      |
/// |---------|------------------|
/// | `arg`   | Unsigned integer |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description           |
/// |--------|-----------------------|
/// | `res`  | Floating-point number |
/// ```
#[pliron_op(
    name = "llvm.uitofp",
    format = "`<nneg=` attr($llvm_nneg_flag, `pliron::builtin::attributes::BoolAttr`) `> `$0 ` to ` type($0)",
    interfaces = [
        CastOpInterface,
        NResultsInterface<1>,
        OneResultInterface,
        NOpdsInterface<1>,
        OneOpdInterface,
        CastOpWithNNegInterface,
        NNegFlag
    ]
)]
pub struct UIToFPOp;

impl Verify for UIToFPOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let opd_ty = OneOpdInterface::operand_type(self, ctx).deref(ctx);
        let Some(opd_ty_int) = opd_ty.downcast_ref::<IntegerType>() else {
            return verify_err!(self.loc(ctx), FloatCastVerifyErr::OperandTypeErr);
        };
        if !opd_ty_int.is_signless() {
            return verify_err!(self.loc(ctx), FloatCastVerifyErr::OperandTypeErr);
        }
        let res_ty = OneResultInterface::result_type(self, ctx).deref(ctx);
        if !type_impls::<dyn FloatTypeInterface>(&**res_ty) {
            return verify_err!(self.loc(ctx), FloatCastVerifyErr::ResultTypeErr);
        }
        Ok(())
    }
}

// ============================================================================
// Registration
// ============================================================================

/// Register all cast operations.
pub fn register(ctx: &mut Context) {
    // Pointer/bitwise
    BitcastOp::register(ctx);
    PtrToIntOp::register(ctx);
    IntToPtrOp::register(ctx);
    AddrSpaceCastOp::register(ctx);

    // Integer casts
    SExtOp::register(ctx);
    ZExtOp::register(ctx);
    TruncOp::register(ctx);

    // Float casts
    FPTruncOp::register(ctx);
    FPExtOp::register(ctx);

    // Int ↔ Float
    FPToSIOp::register(ctx);
    FPToUIOp::register(ctx);
    SIToFPOp::register(ctx);
    UIToFPOp::register(ctx);
}
