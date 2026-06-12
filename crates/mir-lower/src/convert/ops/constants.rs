/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Constant operation conversion: `dialect-mir` → LLVM dialect.
//!
//! Converts constant definitions from `dialect-mir` to the LLVM dialect.
//!
//! # Supported Operations
//!
//! | `dialect-mir` Op     | LLVM dialect Op   | Description       |
//! |----------------------|-------------------|-------------------|
//! | `mir.constant`       | `llvm.constant`   | Integer constants |
//! | `mir.float_constant` | `llvm.constant`   | Float constants   |
//!
//! # Type Handling
//!
//! `dialect-mir` uses signed/unsigned integer types (`ui64`, `si64`), while
//! the LLVM dialect uses signless integers (`i64`). The conversion preserves
//! bit-width but changes to the signless representation.
//!
//! Float constants (f32, f64) pass through unchanged.

use dialect_mir::attributes::MirFP16Attr;
use dialect_mir::ops::{MirConstantOp, MirFloatConstantOp, MirUndefOp};
use llvm_export::ops as llvm;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::Typed;

use crate::convert::types::convert_type;

/// Convert `mir.constant` (integer) to `llvm.constant`.
///
/// MIR integer types are signed/unsigned (`ui64`, `si64`), but LLVM uses
/// signless integers. This conversion preserves the bit pattern and width
/// while changing to signless representation.
pub(crate) fn convert_integer(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    use pliron::builtin::attributes::IntegerAttr;

    let (apint_value, width) = {
        let mir_const = MirConstantOp::new(op);
        let int_attr = mir_const.get_attr_value(ctx).ok_or_else(|| {
            pliron::input_error!(
                op.deref(ctx).loc(),
                "Missing value attribute on mir.constant"
            )
        })?;

        let apint = int_attr.value().clone();
        let mir_int_ty = int_attr.get_type();
        let w = mir_int_ty.deref(ctx).width();
        (apint, w)
    };

    // Create signless LLVM integer type (MIR uses signed/unsigned, LLVM uses signless)
    let llvm_int_ty = IntegerType::get(ctx, width, Signedness::Signless);
    let llvm_int_attr = IntegerAttr::new(llvm_int_ty, apint_value);

    let llvm_const = llvm::ConstantOp::new(ctx, llvm_int_attr.into());
    rewriter.insert_operation(ctx, llvm_const.get_operation());
    rewriter.replace_operation(ctx, op, llvm_const.get_operation());

    Ok(())
}

/// Convert `mir.float_constant` to `llvm.constant`.
///
/// Float constants pass through with their type preserved (f32, f64).
pub(crate) fn convert_float(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    enum FloatAttr {
        F16(MirFP16Attr),
        F32(pliron::builtin::attributes::FPSingleAttr),
        F64(pliron::builtin::attributes::FPDoubleAttr),
    }

    let float_attr = {
        let mir_const = MirFloatConstantOp::new(op);
        if let Some(attr) = mir_const.get_attr_float_value_f16(ctx) {
            FloatAttr::F16(attr.clone())
        } else if let Some(attr) = mir_const.get_attr_float_value(ctx) {
            FloatAttr::F32(attr.clone())
        } else if let Some(attr) = mir_const.get_attr_float_value_f64(ctx) {
            FloatAttr::F64(attr.clone())
        } else {
            return pliron::input_err!(
                op.deref(ctx).loc(),
                "Missing float_value or float_value_f64 attribute on mir.float_constant"
            );
        }
    };

    let llvm_const = match float_attr {
        FloatAttr::F16(attr) => {
            llvm::ConstantOp::new(ctx, llvm_export::fp16_attr_from_bits(attr.to_bits()).into())
        }
        FloatAttr::F32(attr) => llvm::ConstantOp::new(ctx, attr.into()),
        FloatAttr::F64(attr) => llvm::ConstantOp::new(ctx, attr.into()),
    };

    rewriter.insert_operation(ctx, llvm_const.get_operation());
    rewriter.replace_operation(ctx, op, llvm_const.get_operation());

    Ok(())
}

/// Convert `mir.undef` to `llvm.undef`.
///
/// Passes the converted result type through to `llvm::UndefOp::new`.
pub(crate) fn convert_undef(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let _mir_undef = Operation::get_op::<MirUndefOp>(op, ctx).expect("expected MirUndefOp");
    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let llvm_result_ty = convert_type(ctx, result_ty).map_err(|e| {
        pliron::create_error!(
            op.deref(ctx).loc(),
            pliron::result::ErrorKind::VerificationFailed,
            "{e}"
        )
    })?;

    let llvm_undef = llvm::UndefOp::new(ctx, llvm_result_ty);
    rewriter.insert_operation(ctx, llvm_undef.get_operation());
    rewriter.replace_operation(ctx, op, llvm_undef.get_operation());
    Ok(())
}

#[cfg(test)]
mod tests {
    // TODO: Add unit tests for constant conversion
}
