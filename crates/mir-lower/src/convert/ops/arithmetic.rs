/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Arithmetic operation conversion: `dialect-mir` → `dialect-llvm`.
//!
//! Converts `dialect-mir` arithmetic, bitwise, and comparison operations to
//! their `dialect-llvm` equivalents.
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
use dialect_llvm::attributes::{
    FCmpPredicateAttr, FastmathFlagsAttr, ICmpPredicateAttr, IntegerOverflowFlagsAttr,
};
use dialect_llvm::op_interfaces::{BinArithOp, CastOpInterface, IntBinArithOpWithOverflowFlag};
use dialect_llvm::ops as llvm;
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
    ty.deref(ctx).is::<dialect_llvm::types::HalfType>()
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

/// Convert `mir.checked_add` to `(lhs + rhs, overflow)` where the
/// overflow flag is computed from the actual operation, not hardcoded.
///
/// Rust's `u64::overflowing_add` / `i64::overflowing_add` return the
/// real carry-out; multi-limb carry chains (dalek 5×u52, k256 4×u64,
/// base58_encode's byte loop) depend on it. An earlier version pinned
/// the flag to `i1 0` which silently dropped the carry between limbs.
///
/// For unsigned: `carry = (sum < lhs)` — wraparound implies sum
/// underflows below either operand.
/// For signed: overflow happens when `lhs` and `rhs` share a sign and
/// `sum`'s sign differs — `((sum ^ lhs) & (sum ^ rhs)) >> (width - 1)`.
pub(crate) fn convert_checked_add(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let (lhs, rhs) = get_binary_operands(op, ctx)?;
    let signed = is_signed_int_op(ctx, op, operands_info)?;
    convert_checked_binop(ctx, rewriter, op, lhs, rhs, CheckedKind::Add, signed)
}

/// Convert `mir.checked_sub` to `(lhs - rhs, borrow)` with real borrow.
///
/// For unsigned: `borrow = (lhs < rhs)` — underflow when subtrahend
/// exceeds minuend.
/// For signed: overflow when `lhs` and `rhs` have different signs and
/// `diff`'s sign differs from `lhs` —
/// `((lhs ^ rhs) & (lhs ^ diff)) >> (width - 1)`.
pub(crate) fn convert_checked_sub(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let (lhs, rhs) = get_binary_operands(op, ctx)?;
    let signed = is_signed_int_op(ctx, op, operands_info)?;
    convert_checked_binop(ctx, rewriter, op, lhs, rhs, CheckedKind::Sub, signed)
}

/// Convert `mir.checked_mul` to `(lhs * rhs, false)`.
///
/// The overflow flag is still hardcoded to `false` here — a correct
/// detection would need `umul.with.overflow` / `smul.with.overflow`
/// intrinsics or a widening-multiply followed by a high-bits check.
/// The downstream carry-chain consumers don't use `checked_mul`, so
/// leaving this as-is until a repro motivates the fuller fix.
pub(crate) fn convert_checked_mul(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let (lhs, rhs) = get_binary_operands(op, ctx)?;
    let signed = is_signed_int_op(ctx, op, operands_info)?;
    convert_checked_binop(ctx, rewriter, op, lhs, rhs, CheckedKind::Mul, signed)
}

#[derive(Clone, Copy)]
enum CheckedKind {
    Add,
    Sub,
    Mul,
}

/// Shared implementation: compute the result and the real overflow flag.
fn convert_checked_binop(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    lhs: Value,
    rhs: Value,
    kind: CheckedKind,
    signed: bool,
) -> Result<()> {
    let flags = IntegerOverflowFlagsAttr::default();
    let arith_op = match kind {
        CheckedKind::Add => {
            llvm::AddOp::new_with_overflow_flag(ctx, lhs, rhs, flags).get_operation()
        }
        CheckedKind::Sub => {
            llvm::SubOp::new_with_overflow_flag(ctx, lhs, rhs, flags).get_operation()
        }
        CheckedKind::Mul => {
            llvm::MulOp::new_with_overflow_flag(ctx, lhs, rhs, flags).get_operation()
        }
    };
    rewriter.insert_operation(ctx, arith_op);
    let result_value = arith_op.deref(ctx).get_result(0);

    let overflow_flag = match kind {
        CheckedKind::Add | CheckedKind::Sub if !signed => {
            // Unsigned add: carry = (sum < lhs).
            // Unsigned sub: borrow = (lhs < rhs).
            let (cmp_lhs, cmp_rhs) = match kind {
                CheckedKind::Add => (result_value, lhs),
                CheckedKind::Sub => (lhs, rhs),
                CheckedKind::Mul => unreachable!(),
            };
            let icmp = llvm::ICmpOp::new(ctx, ICmpPredicateAttr::ULT, cmp_lhs, cmp_rhs);
            rewriter.insert_operation(ctx, icmp.get_operation());
            icmp.get_operation().deref(ctx).get_result(0)
        }
        CheckedKind::Add | CheckedKind::Sub => {
            // Signed overflow detection via sign-bit XOR pattern.
            //
            // Add: overflow when lhs and rhs share a sign but sum's
            //      sign differs → `((sum ^ lhs) & (sum ^ rhs)) < 0`.
            // Sub: overflow when lhs and rhs have different signs and
            //      diff's sign differs from lhs →
            //      `((lhs ^ rhs) & (lhs ^ diff)) < 0`.
            let lhs_ty = lhs.get_type(ctx);
            let int_width = {
                let ty_obj = lhs_ty.deref(ctx);
                ty_obj
                    .downcast_ref::<IntegerType>()
                    .ok_or_else(|| {
                        pliron::input_error_noloc!(
                            "checked op signed-overflow detection: lhs is not an integer"
                        )
                    })?
                    .width()
            };

            let (a, b, c) = match kind {
                CheckedKind::Add => (result_value, lhs, rhs),
                CheckedKind::Sub => (lhs, rhs, result_value),
                CheckedKind::Mul => unreachable!(),
            };
            let xor1 = llvm::XorOp::new(ctx, a, b).get_operation();
            rewriter.insert_operation(ctx, xor1);
            let xor1_val = xor1.deref(ctx).get_result(0);

            let xor2 = llvm::XorOp::new(ctx, a, c).get_operation();
            rewriter.insert_operation(ctx, xor2);
            let xor2_val = xor2.deref(ctx).get_result(0);

            let and_op = llvm::AndOp::new(ctx, xor1_val, xor2_val).get_operation();
            rewriter.insert_operation(ctx, and_op);
            let and_val = and_op.deref(ctx).get_result(0);

            // Compare the AND result against zero — if its sign bit is
            // set the operation overflowed. icmp slt … 0 == true iff
            // top bit is 1.
            let zero_attr = pliron::builtin::attributes::IntegerAttr::new(
                IntegerType::get(ctx, int_width, Signedness::Signless),
                pliron::utils::apint::APInt::from_u128(
                    0,
                    std::num::NonZeroUsize::new(int_width as usize).unwrap(),
                ),
            );
            let zero_const = llvm::ConstantOp::new(ctx, zero_attr.into());
            rewriter.insert_operation(ctx, zero_const.get_operation());
            let zero_val = zero_const.get_operation().deref(ctx).get_result(0);

            let icmp = llvm::ICmpOp::new(ctx, ICmpPredicateAttr::SLT, and_val, zero_val);
            rewriter.insert_operation(ctx, icmp.get_operation());
            icmp.get_operation().deref(ctx).get_result(0)
        }
        CheckedKind::Mul => {
            // checked_mul overflow detection not yet implemented; stay
            // with the hardcoded `false` until a downstream consumer
            // demands the real flag.
            let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
            let false_attr = pliron::builtin::attributes::IntegerAttr::new(
                i1_ty,
                pliron::utils::apint::APInt::from_u32(0, std::num::NonZeroUsize::new(1).unwrap()),
            );
            let false_const = llvm::ConstantOp::new(ctx, false_attr.into());
            rewriter.insert_operation(ctx, false_const.get_operation());
            false_const.get_operation().deref(ctx).get_result(0)
        }
    };

    let mir_result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let loc = op.deref(ctx).loc();
    let llvm_result_ty =
        convert_type(ctx, mir_result_ty).map_err(|e| pliron::input_error!(loc, "{e}"))?;

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
            dialect_llvm::op_interfaces::ATTR_KEY_INTEGER_OVERFLOW_FLAGS.clone(),
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
/// Integer comparisons use signed or unsigned predicates based on pre-conversion
/// MIR operand type signedness. Float comparisons use ordered predicates
/// (olt, ole, etc.) which return false if either operand is NaN.
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
        llvm::FCmpOp::new(ctx, float_pred, lhs, rhs).get_operation()
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

#[cfg(test)]
mod tests {
    // TODO: Add unit tests for arithmetic conversion
}
