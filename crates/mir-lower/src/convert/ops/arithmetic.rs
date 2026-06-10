/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Arithmetic operation conversion: `dialect-mir` → LLVM dialect.
//!
//! Converts `dialect-mir` arithmetic, bitwise, and comparison operations to
//! their LLVM dialect equivalents.
//!
//! # Operations
//!
//! | Category      | MIR Operations                    | LLVM Operations                              |
//! |---------------|-----------------------------------|----------------------------------------------|
//! | Integer Arith | `add`, `sub`, `mul`, `div`, `rem` | `add`, `sub`, `mul`, `sdiv`/`udiv`, `srem`/`urem` |
//! | Float Arith   | `add`, `sub`, `mul`, `div`, `rem` | `fadd`, `fsub`, `fmul`, `fdiv`, `frem`       |
//! | Unary         | `neg`, `not`                      | `fneg` / `sub 0, x`, `xor`                   |
//! | Bitwise       | `and`, `or`, `xor`, `not`         | `and`, `or`, `xor`                           |
//! | Shifts        | `shl`, `shr`                      | `shl`, `lshr`/`ashr`                         |
//! | Comparison    | `lt`, `le`, `gt`, `ge`, `eq`, `ne`| `icmp` (signed/unsigned predicates), `fcmp`   |
//! | Checked       | `checked_add`                     | `add` + overflow tuple                       |
//!
//! # Type Handling
//!
//! - Integer operations use signless LLVM types
//! - Float operations automatically use `fadd`, `fmul`, etc. with fastmath flags
//! - Shift amounts are cast and masked to match Rust's unchecked shift semantics
//! - Checked operations return `(result, overflow_flag)` tuples

use crate::convert::types::convert_type;
use llvm_export::attributes::{
    FCmpPredicateAttr, FastmathFlagsAttr, ICmpPredicateAttr, IntegerOverflowFlagsAttr,
};
use llvm_export::op_interfaces::{BinArithOp, CastOpInterface, IntBinArithOpWithOverflowFlag};
use llvm_export::ops as llvm;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::Typed;
use pliron::value::Value;

// ============================================================================
// Helper functions for binary operations
// ============================================================================

/// Extract binary operands from the (already-converted) operation.
fn get_binary_operands(op: Ptr<Operation>, ctx: &Context) -> Result<(Value, Value)> {
    let loc = op.deref(ctx).loc();
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    match operands.as_slice() {
        [lhs, rhs] => Ok((*lhs, *rhs)),
        _ => pliron::input_err!(loc, "Binary operation requires exactly 2 operands"),
    }
}

/// Check if a value has floating-point type.
fn is_float_type(ctx: &Context, val: Value) -> bool {
    let ty = val.get_type(ctx);
    ty.deref(ctx).is::<llvm_export::types::HalfType>()
        || ty.deref(ctx).is::<FP32Type>()
        || ty.deref(ctx).is::<FP64Type>()
}

/// Check if a binary operation's integer operands were signed before type conversion.
///
/// Uses `operands_info` to read the *pre-conversion* MIR type (which preserves
/// Rust's signedness). After DialectConversion, the live operand type is already
/// signless. Pointer types are treated as unsigned.
fn is_signed_int_op(
    ctx: &Context,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<bool> {
    let operand = op.deref(ctx).get_operand(0);
    if let Some(int_ty) = operands_info.lookup_most_recent_of_type::<IntegerType>(ctx, operand) {
        Ok(int_ty.signedness() == Signedness::Signed)
    } else if operands_info
        .lookup_most_recent_of_type::<dialect_mir::types::MirPtrType>(ctx, operand)
        .is_some()
    {
        Ok(false)
    } else {
        pliron::input_err!(
            op.deref(ctx).loc(),
            "expected IntegerType or MirPtrType operand in arithmetic op"
        )
    }
}

/// Add fastmath flags attribute to a floating-point operation.
fn add_fastmath_flags(ctx: &mut Context, op: Ptr<Operation>) {
    let flags = FastmathFlagsAttr::default();
    let key: pliron::identifier::Identifier = "llvm_fast_math_flags".try_into().unwrap();
    op.deref_mut(ctx).attributes.0.insert(key, flags.into());
}

// ============================================================================
// Arithmetic operations
// ============================================================================

/// Convert `mir.add` to `llvm.add` (integer) or `llvm.fadd` (float).
///
/// Integer additions use default overflow flags (no wrapping behavior).
/// Float additions include fastmath flags for potential optimizations.
pub(crate) fn convert_add(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let (lhs, rhs) = get_binary_operands(op, ctx)?;

    let llvm_op = if is_float_type(ctx, lhs) {
        let fadd = llvm::FAddOp::new(ctx, lhs, rhs);
        add_fastmath_flags(ctx, fadd.get_operation());
        fadd.get_operation()
    } else {
        let flags = IntegerOverflowFlagsAttr::default();
        llvm::AddOp::new_with_overflow_flag(ctx, lhs, rhs, flags).get_operation()
    };

    rewriter.insert_operation(ctx, llvm_op);
    rewriter.replace_operation(ctx, op, llvm_op);
    Ok(())
}

/// Convert `mir.sub` to `llvm.sub` (integer) or `llvm.fsub` (float).
///
/// Integer subtractions use default overflow flags.
/// Float subtractions include fastmath flags.
pub(crate) fn convert_sub(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let (lhs, rhs) = get_binary_operands(op, ctx)?;

    let llvm_op = if is_float_type(ctx, lhs) {
        let fsub = llvm::FSubOp::new(ctx, lhs, rhs);
        add_fastmath_flags(ctx, fsub.get_operation());
        fsub.get_operation()
    } else {
        let flags = IntegerOverflowFlagsAttr::default();
        llvm::SubOp::new_with_overflow_flag(ctx, lhs, rhs, flags).get_operation()
    };

    rewriter.insert_operation(ctx, llvm_op);
    rewriter.replace_operation(ctx, op, llvm_op);
    Ok(())
}

/// Convert `mir.mul` to `llvm.mul` (integer) or `llvm.fmul` (float).
///
/// Integer multiplications use default overflow flags.
/// Float multiplications include fastmath flags.
pub(crate) fn convert_mul(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let (lhs, rhs) = get_binary_operands(op, ctx)?;

    let llvm_op = if is_float_type(ctx, lhs) {
        let fmul = llvm::FMulOp::new(ctx, lhs, rhs);
        add_fastmath_flags(ctx, fmul.get_operation());
        fmul.get_operation()
    } else {
        let flags = IntegerOverflowFlagsAttr::default();
        llvm::MulOp::new_with_overflow_flag(ctx, lhs, rhs, flags).get_operation()
    };

    rewriter.insert_operation(ctx, llvm_op);
    rewriter.replace_operation(ctx, op, llvm_op);
    Ok(())
}

/// Convert `mir.div` to `llvm.sdiv` (signed), `llvm.udiv` (unsigned), or `llvm.fdiv` (float).
///
/// Uses pre-conversion MIR operand type signedness to select between signed
/// and unsigned integer division.
pub(crate) fn convert_div(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let (lhs, rhs) = get_binary_operands(op, ctx)?;

    let llvm_op = if is_float_type(ctx, lhs) {
        let fdiv = llvm::FDivOp::new(ctx, lhs, rhs);
        add_fastmath_flags(ctx, fdiv.get_operation());
        fdiv.get_operation()
    } else if is_signed_int_op(ctx, op, operands_info)? {
        llvm::SDivOp::new(ctx, lhs, rhs).get_operation()
    } else {
        llvm::UDivOp::new(ctx, lhs, rhs).get_operation()
    };

    rewriter.insert_operation(ctx, llvm_op);
    rewriter.replace_operation(ctx, op, llvm_op);
    Ok(())
}

/// Convert `mir.rem` to `llvm.srem` (signed), `llvm.urem` (unsigned), or `llvm.frem` (float).
///
/// Uses pre-conversion MIR operand type signedness to select between signed
/// and unsigned integer remainder.
pub(crate) fn convert_rem(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let (lhs, rhs) = get_binary_operands(op, ctx)?;

    let llvm_op = if is_float_type(ctx, lhs) {
        let frem = llvm::FRemOp::new(ctx, lhs, rhs);
        add_fastmath_flags(ctx, frem.get_operation());
        frem.get_operation()
    } else if is_signed_int_op(ctx, op, operands_info)? {
        llvm::SRemOp::new(ctx, lhs, rhs).get_operation()
    } else {
        llvm::URemOp::new(ctx, lhs, rhs).get_operation()
    };

    rewriter.insert_operation(ctx, llvm_op);
    rewriter.replace_operation(ctx, op, llvm_op);
    Ok(())
}

// ============================================================================
// Checked operations (GPU: no overflow checking, just return (result, false))
// ============================================================================

/// Convert `mir.checked_add` to regular addition returning `(result, false)`.
///
/// GPU kernels don't perform overflow checking for performance. The overflow
/// flag is always `false`. Returns a struct `{ result: T, overflow: i1 }`.
pub(crate) fn convert_checked_add(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let (lhs, rhs) = get_binary_operands(op, ctx)?;
    convert_checked_binop(ctx, rewriter, op, lhs, rhs, |ctx, l, r| {
        let flags = IntegerOverflowFlagsAttr::default();
        llvm::AddOp::new_with_overflow_flag(ctx, l, r, flags).get_operation()
    })
}

/// Convert `mir.checked_mul` to regular multiplication returning `(result, false)`.
///
/// GPU kernels don't perform overflow checking for performance. The overflow
/// flag is always `false`. Returns a struct `{ result: T, overflow: i1 }`.
pub(crate) fn convert_checked_mul(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let (lhs, rhs) = get_binary_operands(op, ctx)?;
    convert_checked_binop(ctx, rewriter, op, lhs, rhs, |ctx, l, r| {
        let flags = IntegerOverflowFlagsAttr::default();
        llvm::MulOp::new_with_overflow_flag(ctx, l, r, flags).get_operation()
    })
}

/// Convert `mir.checked_sub` to regular subtraction returning `(result, false)`.
///
/// GPU kernels don't perform overflow checking for performance. The overflow
/// flag is always `false`. Returns a struct `{ result: T, overflow: i1 }`.
pub(crate) fn convert_checked_sub(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let (lhs, rhs) = get_binary_operands(op, ctx)?;
    convert_checked_binop(ctx, rewriter, op, lhs, rhs, |ctx, l, r| {
        let flags = IntegerOverflowFlagsAttr::default();
        llvm::SubOp::new_with_overflow_flag(ctx, l, r, flags).get_operation()
    })
}

/// Shared implementation for checked binary ops: compute result, pack with `false` overflow flag.
fn convert_checked_binop<F>(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    lhs: Value,
    rhs: Value,
    build_arith: F,
) -> Result<()>
where
    F: FnOnce(&mut Context, Value, Value) -> Ptr<Operation>,
{
    let arith_op = build_arith(ctx, lhs, rhs);
    rewriter.insert_operation(ctx, arith_op);
    let result_value = arith_op.deref(ctx).get_result(0);

    // Create false constant for overflow flag (GPU doesn't check overflow)
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let false_attr = pliron::builtin::attributes::IntegerAttr::new(
        i1_ty,
        pliron::utils::apint::APInt::from_u32(0, std::num::NonZeroUsize::new(1).unwrap()),
    );
    let false_const = llvm::ConstantOp::new(ctx, false_attr.into());
    rewriter.insert_operation(ctx, false_const.get_operation());
    let overflow_flag = false_const.get_operation().deref(ctx).get_result(0);

    // Get result type and convert
    let mir_result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let loc = op.deref(ctx).loc();
    let llvm_result_ty =
        convert_type(ctx, mir_result_ty).map_err(|e| pliron::input_error!(loc, "{e}"))?;

    // Create tuple struct: {result, overflow_flag}
    let undef = llvm::UndefOp::new(ctx, llvm_result_ty);
    rewriter.insert_operation(ctx, undef.get_operation());
    let struct_val = undef.get_operation().deref(ctx).get_result(0);

    let insert0 = llvm::InsertValueOp::new(ctx, struct_val, result_value, vec![0]);
    rewriter.insert_operation(ctx, insert0.get_operation());
    let struct_with_result = insert0.get_operation().deref(ctx).get_result(0);

    let insert1 = llvm::InsertValueOp::new(ctx, struct_with_result, overflow_flag, vec![1]);
    rewriter.insert_operation(ctx, insert1.get_operation());

    rewriter.replace_operation(ctx, op, insert1.get_operation());
    Ok(())
}

// ============================================================================
// Shift operations
// ============================================================================

/// Convert `mir.shr` to `llvm.ashr` (signed, arithmetic) or `llvm.lshr` (unsigned, logical).
///
/// Signed types use arithmetic shift right (sign-extending), unsigned types
/// use logical shift right (zero-filling). The shift count is cast and masked
/// before lowering because LLVM shifts are poison when the count is too large.
pub(crate) fn convert_shr(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let signed = is_signed_int_op(ctx, op, operands_info)?;
    convert_shift(ctx, rewriter, op, |ctx, lhs, rhs| {
        if signed {
            llvm::AShrOp::new(ctx, lhs, rhs).get_operation()
        } else {
            llvm::LShrOp::new(ctx, lhs, rhs).get_operation()
        }
    })
}

/// Convert `mir.shl` to `llvm.shl` (shift left).
///
/// Includes default overflow flags. The shift count is cast and masked before
/// lowering because LLVM shifts are poison when the count is too large.
pub(crate) fn convert_shl(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_shift(ctx, rewriter, op, |ctx, lhs, rhs| {
        let shl_op = llvm::ShlOp::new(ctx, lhs, rhs);
        let flags = IntegerOverflowFlagsAttr::default();
        shl_op.get_operation().deref_mut(ctx).attributes.set(
            llvm_export::op_interfaces::ATTR_KEY_INTEGER_OVERFLOW_FLAGS.clone(),
            flags,
        );
        shl_op.get_operation()
    })
}

/// Common shift operation converter with Rust-compatible count handling.
///
/// LLVM requires the shift amount to have the same type as the value being
/// shifted. This function handles automatic widening (zext) or narrowing
/// (trunc) of the shift amount to match, then masks it with `bit_width - 1`.
/// That matches Rust's unchecked/release shift behavior and avoids LLVM poison
/// for oversized counts.
fn convert_shift<F>(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    builder: F,
) -> Result<()>
where
    F: FnOnce(&mut Context, Value, Value) -> Ptr<Operation>,
{
    let (lhs, rhs) = get_binary_operands(op, ctx)?;

    let lhs_ty = lhs.get_type(ctx);
    let rhs_ty = rhs.get_type(ctx);
    let lhs_width = lhs_ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .ok_or_else(|| {
            pliron::input_error!(op.deref(ctx).loc(), "Shift value must be integer type")
        })?
        .width();

    let rhs_casted = if lhs_ty != rhs_ty {
        let rhs_width = rhs_ty
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .ok_or_else(|| {
                pliron::input_error!(op.deref(ctx).loc(), "Shift amount must be integer type")
            })?
            .width();

        let cast_op = if lhs_width > rhs_width {
            let zext = llvm::ZExtOp::new(ctx, rhs, lhs_ty);
            let nneg_key: pliron::identifier::Identifier = "llvm_nneg_flag".try_into().unwrap();
            zext.get_operation().deref_mut(ctx).attributes.0.insert(
                nneg_key,
                pliron::builtin::attributes::BoolAttr::new(false).into(),
            );
            zext.get_operation()
        } else {
            llvm::TruncOp::new(ctx, rhs, lhs_ty).get_operation()
        };
        rewriter.insert_operation(ctx, cast_op);
        cast_op.deref(ctx).get_result(0)
    } else {
        rhs
    };

    let rhs_masked = mask_shift_amount(ctx, rewriter, rhs_casted, lhs_ty, lhs_width);
    let llvm_op = builder(ctx, lhs, rhs_masked);
    rewriter.insert_operation(ctx, llvm_op);
    rewriter.replace_operation(ctx, op, llvm_op);
    Ok(())
}

fn mask_shift_amount(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    rhs: Value,
    lhs_ty: Ptr<pliron::r#type::TypeObj>,
    lhs_width: u32,
) -> Value {
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let mask_ty = IntegerType::get(ctx, lhs_width, Signedness::Signless);
    let mask_attr = pliron::builtin::attributes::IntegerAttr::new(
        mask_ty,
        APInt::from_u128(
            u128::from(lhs_width - 1),
            NonZeroUsize::new(lhs_width as usize).unwrap(),
        ),
    );
    let mask_op = llvm::ConstantOp::new(ctx, mask_attr.into());
    rewriter.insert_operation(ctx, mask_op.get_operation());
    let mask_value = mask_op.get_operation().deref(ctx).get_result(0);

    let and_op = llvm::AndOp::new(ctx, rhs, mask_value).get_operation();
    rewriter.insert_operation(ctx, and_op);

    debug_assert_eq!(and_op.deref(ctx).get_result(0).get_type(ctx), lhs_ty);
    and_op.deref(ctx).get_result(0)
}

// ============================================================================
// Bitwise operations
// ============================================================================

/// Convert `mir.bitand` to `llvm.and`.
pub(crate) fn convert_bitand(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let (lhs, rhs) = get_binary_operands(op, ctx)?;
    let llvm_op = llvm::AndOp::new(ctx, lhs, rhs).get_operation();
    rewriter.insert_operation(ctx, llvm_op);
    rewriter.replace_operation(ctx, op, llvm_op);
    Ok(())
}

/// Convert `mir.bitor` to `llvm.or`.
pub(crate) fn convert_bitor(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let (lhs, rhs) = get_binary_operands(op, ctx)?;
    let llvm_op = llvm::OrOp::new(ctx, lhs, rhs).get_operation();
    rewriter.insert_operation(ctx, llvm_op);
    rewriter.replace_operation(ctx, op, llvm_op);
    Ok(())
}

/// Convert `mir.bitxor` to `llvm.xor`.
pub(crate) fn convert_bitxor(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let (lhs, rhs) = get_binary_operands(op, ctx)?;
    let llvm_op = llvm::XorOp::new(ctx, lhs, rhs).get_operation();
    rewriter.insert_operation(ctx, llvm_op);
    rewriter.replace_operation(ctx, op, llvm_op);
    Ok(())
}

/// Convert `mir.neg` to `llvm.fneg` for floats or `0 - x` for integers.
///
/// LLVM has a dedicated floating-point negation op. Integer negation is a
/// subtraction from zero, which also matches how LLVM represents integer `neg`.
pub(crate) fn convert_neg(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let operand = op.deref(ctx).get_operand(0);
    let operand_ty = operand.get_type(ctx);

    let llvm_op = if is_float_type(ctx, operand) {
        llvm::FNegOp::new_with_fast_math_flags(ctx, operand, FastmathFlagsAttr::default())
            .get_operation()
    } else {
        let width = operand_ty
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .ok_or_else(|| {
                pliron::input_error!(
                    op.deref(ctx).loc(),
                    "NEG only supports integer or float types"
                )
            })?
            .width();

        let zero_ty = IntegerType::get(ctx, width, Signedness::Signless);
        let zero_attr = IntegerAttr::new(
            zero_ty,
            APInt::from_u128(0, NonZeroUsize::new(width as usize).unwrap()),
        );
        let zero_op = llvm::ConstantOp::new(ctx, zero_attr.into()).get_operation();
        rewriter.insert_operation(ctx, zero_op);
        let zero = zero_op.deref(ctx).get_result(0);

        let flags = IntegerOverflowFlagsAttr::default();
        llvm::SubOp::new_with_overflow_flag(ctx, zero, operand, flags).get_operation()
    };

    rewriter.insert_operation(ctx, llvm_op);
    rewriter.replace_operation(ctx, op, llvm_op);
    Ok(())
}

/// Convert `mir.not` to `llvm.xor` with all-ones constant.
///
/// LLVM has no direct NOT instruction. Bitwise NOT is implemented as
/// XOR with -1 (all bits set). The constant is created with the same
/// bit width as the operand.
pub(crate) fn convert_not(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let operand = op.deref(ctx).get_operand(0);

    let ty = operand.get_type(ctx);
    let width = ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .ok_or_else(|| {
            pliron::input_error!(op.deref(ctx).loc(), "NOT only supports integer types")
        })?
        .width();

    // Create all-ones constant (-1)
    let llvm_ty = IntegerType::get(ctx, width, Signedness::Signless);
    let apint = APInt::from_i64(-1, NonZeroUsize::new(width as usize).unwrap());
    let attr = pliron::builtin::attributes::IntegerAttr::new(llvm_ty, apint);
    let ones_const = llvm::ConstantOp::new(ctx, attr.into()).get_operation();
    rewriter.insert_operation(ctx, ones_const);
    let ones_val = ones_const.deref(ctx).get_result(0);

    let xor_op = llvm::XorOp::new(ctx, operand, ones_val).get_operation();
    rewriter.insert_operation(ctx, xor_op);
    rewriter.replace_operation(ctx, op, xor_op);
    Ok(())
}

// ============================================================================
// Comparison operations
// ============================================================================

/// Convert MIR comparison to `llvm.icmp` (integer) or `llvm.fcmp` (float).
///
/// Integer comparisons use signed or unsigned predicates based on
/// pre-conversion MIR operand type signedness.
///
/// Float predicates mirror rustc_codegen_ssa's `bin_op_to_fcmp_predicate`:
/// `Eq -> oeq`, `Lt -> olt`, `Le -> ole`, `Gt -> ogt`, `Ge -> oge` (ordered,
/// false if either operand is NaN), and `Ne -> une` (UNordered, true if
/// either operand is NaN) so that `a != b` equals `!(a == b)` per Rust
/// `PartialEq` semantics. An ordered `one` here folds the canonical NaN
/// check `x != x` to `false` (issue #123).
pub(crate) fn convert_cmp(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
    signed_pred: ICmpPredicateAttr,
    unsigned_pred: ICmpPredicateAttr,
    float_pred: FCmpPredicateAttr,
) -> Result<()> {
    let (lhs, rhs) = get_binary_operands(op, ctx)?;

    let llvm_op = if is_float_type(ctx, lhs) {
        // Upstream FCmpOp carries the FastMathFlags interface, whose verifier
        // requires the `llvm_fast_math_flags` attribute to be present (it is
        // not set by `FCmpOp::new`). Attach default flags, as the float
        // arithmetic ops do.
        let fcmp = llvm::FCmpOp::new(ctx, float_pred, lhs, rhs).get_operation();
        add_fastmath_flags(ctx, fcmp);
        fcmp
    } else {
        let pred = if is_signed_int_op(ctx, op, operands_info)? {
            signed_pred
        } else {
            unsigned_pred
        };
        llvm::ICmpOp::new(ctx, pred, lhs, rhs).get_operation()
    };

    rewriter.insert_operation(ctx, llvm_op);
    rewriter.replace_operation(ctx, op, llvm_op);
    Ok(())
}

pub(crate) fn convert_three_way_cmp(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let (lhs, rhs) = get_binary_operands(op, ctx)?;
    if is_float_type(ctx, lhs) {
        // rustc never emits BinOp::Cmp for floats: f32/f64 are not Ord,
        // and f32::total_cmp lowers to integer bit-twiddling, not Cmp.
        // An OLT/OGT select chain would also miscompile: NaN compares
        // false on both predicates and would silently yield Equal.
        let loc = op.deref(ctx).loc();
        return pliron::input_err!(
            loc,
            "BinOp::Cmp on floats is never emitted by rustc; total-order lowering unimplemented"
        );
    }
    let is_signed = is_signed_int_op(ctx, op, operands_info)?;

    let is_lt = emit_cmp_value(ctx, rewriter, op, lhs, rhs, is_signed, true);
    let is_gt = emit_cmp_value(ctx, rewriter, op, lhs, rhs, is_signed, false);

    let mir_result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let loc = op.deref(ctx).loc();
    let (discr_ty, variant_discriminants) = {
        let mir_result_ty_obj = mir_result_ty.deref(ctx);
        let enum_ty = mir_result_ty_obj
            .downcast_ref::<dialect_mir::types::MirEnumType>()
            .ok_or_else(|| pliron::input_error!(loc.clone(), "mir.cmp result must be an enum"))?;
        (
            enum_ty.discriminant_ty,
            enum_ty.variant_discriminants.clone(),
        )
    };
    let llvm_result_ty =
        convert_type(ctx, mir_result_ty).map_err(|e| pliron::input_error!(loc.clone(), "{e}"))?;
    let llvm_discr_ty =
        convert_type(ctx, discr_ty).map_err(|e| pliron::input_error!(loc.clone(), "{e}"))?;

    if variant_discriminants.len() != 3 {
        return pliron::input_err_noloc!("mir.cmp result enum must have three discriminants");
    }

    let less = emit_discriminant_const(ctx, rewriter, llvm_discr_ty, variant_discriminants[0])?;
    let equal = emit_discriminant_const(ctx, rewriter, llvm_discr_ty, variant_discriminants[1])?;
    let greater = emit_discriminant_const(ctx, rewriter, llvm_discr_ty, variant_discriminants[2])?;

    let gt_or_equal = llvm::SelectOp::new(ctx, is_gt, greater, equal).get_operation();
    rewriter.insert_operation(ctx, gt_or_equal);
    let gt_or_equal_val = gt_or_equal.deref(ctx).get_result(0);

    let selected = llvm::SelectOp::new(ctx, is_lt, less, gt_or_equal_val).get_operation();
    rewriter.insert_operation(ctx, selected);
    let selected_discr = selected.deref(ctx).get_result(0);

    let undef = llvm::UndefOp::new(ctx, llvm_result_ty);
    rewriter.insert_operation(ctx, undef.get_operation());
    let enum_value = undef.get_operation().deref(ctx).get_result(0);

    let insert_discr = llvm::InsertValueOp::new(ctx, enum_value, selected_discr, vec![0]);
    rewriter.insert_operation(ctx, insert_discr.get_operation());
    rewriter.replace_operation(ctx, op, insert_discr.get_operation());
    Ok(())
}

/// Emit one integer comparison leg of the three-way compare.
///
/// Float operands are rejected by [`convert_three_way_cmp`] before this
/// runs, so only integer predicates are needed.
fn emit_cmp_value(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    lhs: Value,
    rhs: Value,
    is_signed: bool,
    less: bool,
) -> Value {
    let pred = match (less, is_signed) {
        (true, true) => ICmpPredicateAttr::SLT,
        (true, false) => ICmpPredicateAttr::ULT,
        (false, true) => ICmpPredicateAttr::SGT,
        (false, false) => ICmpPredicateAttr::UGT,
    };
    let cmp_op = llvm::ICmpOp::new(ctx, pred, lhs, rhs).get_operation();
    cmp_op.deref_mut(ctx).set_loc(op.deref(ctx).loc());
    rewriter.insert_operation(ctx, cmp_op);
    cmp_op.deref(ctx).get_result(0)
}

fn emit_discriminant_const(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    discr_ty: Ptr<pliron::r#type::TypeObj>,
    value: u64,
) -> Result<Value> {
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let width = discr_ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .ok_or_else(|| pliron::input_error_noloc!("Ordering discriminant must be integer"))?
        .width();
    let signless_ty = IntegerType::get(ctx, width, Signedness::Signless);
    let attr = IntegerAttr::new(
        signless_ty,
        APInt::from_u64(value, NonZeroUsize::new(width as usize).unwrap()),
    );
    let const_op = llvm::ConstantOp::new(ctx, attr.into()).get_operation();
    rewriter.insert_operation(ctx, const_op);
    Ok(const_op.deref(ctx).get_result(0))
}

// Conversion coverage for this module lives in the crate's integration
// tests: `tests/lowering_test.rs::test_cmp_predicate_lowering` locks the
// comparison predicate table (and empty fastmath flags) end-to-end through
// the DialectConversion framework.
