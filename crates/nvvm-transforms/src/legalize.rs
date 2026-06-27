/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Converts modern LLVM operations to the LLVM 7 forms accepted by
//! pre-Blackwell libNVVM.
//!
//! The pass runs after MIR-to-LLVM lowering. It preserves operation semantics
//! and reports an error when no equivalent legacy form is available.

use std::num::NonZeroUsize;

use llvm_export::{
    attributes::{
        FCmpPredicateAttr, FastmathFlagsAttr, ICmpPredicateAttr, IntegerOverflowFlagsAttr,
    },
    op_interfaces::{
        BinArithOp, CastOpInterface, CastOpWithNNegInterface, FastMathFlags,
        IntBinArithOpWithOverflowFlag,
    },
    ops as llvm, types as llvm_types,
};
use pliron::{
    builtin::{
        attributes::{BoolAttr, FPDoubleAttr, FPSingleAttr, IntegerAttr, TypeAttr},
        op_interfaces::{CallOpCallable, CallOpInterface, SymbolOpInterface},
        type_interfaces::FunctionTypeInterface,
        types::{FP16Type, FP32Type, FP64Type, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    identifier::Identifier,
    linked_list::ContainsLinkedList,
    location::Located,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::{TypeHandle, Typed},
    utils::apint::APInt,
    value::Value,
};

use crate::helpers;

const NNEG_ATTR: &str = "llvm_nneg_flag";

/// Rewrite a lowered LLVM module to the LLVM 7 subset used by legacy NVVM IR.
///
/// Atomic and fence operations that cuda-oxide has not yet legalized return an
/// error instead of being emitted with unverified semantics.
pub(crate) fn legalize_for_legacy_nvvm(ctx: &mut Context, module: Ptr<Operation>) -> Result<()> {
    let mut ops = Vec::new();
    collect_ops(ctx, module, &mut ops);

    // Validate the complete module before changing it. Some unsupported
    // ordering and scope settings are otherwise ignored by libNVVM.
    for &op in &ops {
        reject_nonportable_f16_types(ctx, op)?;
        reject_unsupported_op(ctx, op)?;
        validate_rewrite_candidate(ctx, op)?;
    }

    let mut obsolete_declarations = Vec::new();
    for op in ops {
        remove_nneg(ctx, op);

        if Operation::get_op::<llvm::FNegOp>(op, ctx).is_some() {
            rewrite_fneg(ctx, op)?;
            continue;
        }

        let Some(call) = Operation::get_op::<llvm::CallOp>(op, ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(ctx) else {
            continue;
        };
        let name = callee.to_string();

        if parse_integer_sat_name(&name).is_some() {
            rewrite_integer_sat(ctx, op, &name)?;
            obsolete_declarations.push(name);
        } else if parse_float_to_int_sat_name(&name).is_some() {
            rewrite_float_to_int_sat(ctx, op, &name)?;
            obsolete_declarations.push(name);
        } else if legacy_bit_rewrite(&name).is_some() {
            rewrite_legacy_bit_intrinsic(ctx, op, &name)?;
            obsolete_declarations.push(name);
        } else if shuffle_mode(&name).is_some() {
            rewrite_shuffle(ctx, op, &name)?;
            obsolete_declarations.push(name);
        } else if vote_mode(&name).is_some() {
            rewrite_vote(ctx, op, &name)?;
            obsolete_declarations.push(name);
        } else if let Some(legacy_name) = legacy_memory_intrinsic_name(&name) {
            rewrite_renamed_call(ctx, op, &legacy_name)?;
            obsolete_declarations.push(name);
        }
    }

    remove_obsolete_declarations(ctx, module, &obsolete_declarations);
    verify_legacy_subset(ctx, module)
}

/// Rewrite bit-manipulation intrinsics whose integer widths NVVM rejects in
/// both textual dialects.
///
/// Unlike [`legalize_for_legacy_nvvm`], this leaves modern operations, flags,
/// f16, atomics, and intrinsic signatures unchanged. Blackwell and newer
/// targets accept those forms, but NVVM bit intrinsics still stop at i64.
pub(crate) fn legalize_nvvm_bit_intrinsics(
    ctx: &mut Context,
    module: Ptr<Operation>,
) -> Result<()> {
    let mut ops = Vec::new();
    collect_ops(ctx, module, &mut ops);

    // Validate every candidate before changing the module.
    for &op in &ops {
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(ctx) else {
            continue;
        };
        let name = callee.to_string();
        if legacy_bit_rewrite(&name).is_some() {
            validate_legacy_bit_call(ctx, op, &name)?;
        }
    }

    let mut obsolete_declarations = Vec::new();
    for op in ops {
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(ctx) else {
            continue;
        };
        let name = callee.to_string();
        if legacy_bit_rewrite(&name).is_some() {
            rewrite_legacy_bit_intrinsic(ctx, op, &name)?;
            obsolete_declarations.push(name);
        }
    }

    remove_obsolete_declarations(ctx, module, &obsolete_declarations);
    let mut rewritten_ops = Vec::new();
    collect_ops(ctx, module, &mut rewritten_ops);
    for op in rewritten_ops {
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(ctx) else {
            continue;
        };
        if legacy_bit_rewrite(&callee.to_string()).is_some() {
            return pliron::input_err!(
                op.deref(ctx).loc(),
                "NVVM bit-intrinsic legalization left unsupported call @{callee} behind"
            );
        }
    }
    Ok(())
}

/// CUDA 12's LLVM 7 NVVM dialect does not support the scalar `half` type.
///
/// This checks operands, results, function and global signatures, block
/// arguments, aggregate elements, and operation type attributes. Named
/// recursive structs are guarded by the `seen` set.
fn reject_nonportable_f16_types(ctx: &Context, op: Ptr<Operation>) -> Result<()> {
    let has_half = op
        .deref(ctx)
        .operands()
        .any(|value| type_contains_f16(ctx, value.get_type(ctx), &mut Vec::new()))
        || op
            .deref(ctx)
            .results()
            .any(|value| type_contains_f16(ctx, value.get_type(ctx), &mut Vec::new()))
        || op.deref(ctx).attributes.0.values().any(|attr| {
            attr.downcast_ref::<TypeAttr>()
                .is_some_and(|ty| type_contains_f16(ctx, ty.get_type(ctx), &mut Vec::new()))
        })
        || op.deref(ctx).regions().any(|region| {
            region.deref(ctx).iter(ctx).any(|block| {
                block
                    .deref(ctx)
                    .arguments()
                    .any(|argument| type_contains_f16(ctx, argument.get_type(ctx), &mut Vec::new()))
            })
        })
        || Operation::get_op::<llvm::FuncOp>(op, ctx)
            .is_some_and(|func| type_contains_f16(ctx, func.get_type(ctx).into(), &mut Vec::new()))
        || Operation::get_op::<llvm::GlobalOp>(op, ctx)
            .is_some_and(|global| type_contains_f16(ctx, global.get_type(ctx), &mut Vec::new()));

    if has_half {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "legacy NVVM IR cannot contain scalar f16 because cuda-oxide supports the CUDA 12 LLVM 7 dialect; use f32/f64 or the modern NVVM/PTX path"
        );
    }
    Ok(())
}

fn type_contains_f16(ctx: &Context, ty: TypeHandle, seen: &mut Vec<TypeHandle>) -> bool {
    if ty.deref(ctx).is::<FP16Type>() {
        return true;
    }
    if seen.contains(&ty) {
        return false;
    }
    seen.push(ty);

    let ty_ref = ty.deref(ctx);
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
        !struct_ty.is_opaque()
            && struct_ty
                .fields()
                .any(|field| type_contains_f16(ctx, field, seen))
    } else if let Some(array_ty) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
        type_contains_f16(ctx, array_ty.elem_type(), seen)
    } else if let Some(vector_ty) = ty_ref.downcast_ref::<llvm_types::VectorType>() {
        type_contains_f16(ctx, vector_ty.elem_type(), seen)
    } else if let Some(func_ty) = ty_ref.downcast_ref::<llvm_types::FuncType>() {
        type_contains_f16(ctx, func_ty.result_type(), seen)
            || func_ty
                .arg_types()
                .into_iter()
                .any(|arg| type_contains_f16(ctx, arg, seen))
    } else {
        false
    }
}

fn collect_ops(ctx: &Context, root: Ptr<Operation>, out: &mut Vec<Ptr<Operation>>) {
    out.push(root);
    let regions: Vec<_> = root.deref(ctx).regions().collect();
    for region in regions {
        let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
        for block in blocks {
            let children: Vec<_> = block.deref(ctx).iter(ctx).collect();
            for child in children {
                collect_ops(ctx, child, out);
            }
        }
    }
}

fn reject_unsupported_op(ctx: &Context, op: Ptr<Operation>) -> Result<()> {
    let reason = if Operation::get_op::<llvm::AtomicLoadOp>(op, ctx).is_some() {
        Some("atomic loads")
    } else if Operation::get_op::<llvm::AtomicStoreOp>(op, ctx).is_some() {
        Some("atomic stores")
    } else if Operation::get_op::<llvm::AtomicRmwOp>(op, ctx).is_some() {
        Some("atomic read-modify-write operations")
    } else if Operation::get_op::<llvm::AtomicCmpxchgOp>(op, ctx).is_some() {
        Some("atomic compare-exchange operations")
    } else if Operation::get_op::<llvm::FenceOp>(op, ctx).is_some() {
        Some("LLVM fences")
    } else if Operation::get_op::<llvm::DebugValueOp>(op, ctx).is_some() {
        Some("LLVM debug-value records")
    } else {
        None
    };

    if let Some(reason) = reason {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "cuda-oxide has not yet legalized {reason} for legacy NVVM IR; use ordinary PTX output or a Blackwell NVVM target"
        );
    }
    Ok(())
}

fn validate_rewrite_candidate(ctx: &Context, op: Ptr<Operation>) -> Result<()> {
    if Operation::get_op::<llvm::FNegOp>(op, ctx).is_some() {
        let ty = op.deref(ctx).get_operand(0).get_type(ctx);
        if !matches!(float_width(ctx, ty), Some(32) | Some(64)) {
            return pliron::input_err!(
                op.deref(ctx).loc(),
                "legacy NVVM fneg legalization supports only scalar f32 and f64; scalar f16 is not portable to the supported CUDA 12 legacy dialect"
            );
        }
    }

    let Some(call) = Operation::get_op::<llvm::CallOp>(op, ctx) else {
        return Ok(());
    };
    let CallOpCallable::Direct(callee) = call.callee(ctx) else {
        return Ok(());
    };
    let name = callee.to_string();
    if parse_integer_sat_name(&name).is_some() {
        validate_integer_sat_call(ctx, op, &name)?;
    } else if parse_float_to_int_sat_name(&name).is_some() {
        validate_float_to_int_sat_call(ctx, op, &name)?;
    } else if legacy_bit_rewrite(&name).is_some() {
        validate_legacy_bit_call(ctx, op, &name)?;
    } else if shuffle_mode(&name).is_some() {
        validate_shuffle_call(ctx, op, &name)?;
    } else if vote_mode(&name).is_some() {
        validate_vote_call(ctx, op, &name)?;
    }
    Ok(())
}

fn remove_nneg(ctx: &mut Context, op: Ptr<Operation>) {
    let key: Identifier = NNEG_ATTR
        .try_into()
        .expect("static attribute name is valid");
    // `ZExtOp` and `UIToFPOp` implement pliron-LLVM's NNegFlag interface,
    // whose verifier requires the attribute to exist even when it is false.
    // LLVM 7 cannot parse the textual `nneg` keyword, so clear the semantic
    // bit while retaining the dialect invariant; the exporter omits false.
    if op.deref(ctx).attributes.0.contains_key(&key) {
        op.deref_mut(ctx).attributes.set(key, BoolAttr::new(false));
    }
}

fn insert_before(ctx: &mut Context, anchor: Ptr<Operation>, op: Ptr<Operation>) -> Value {
    op.insert_before(ctx, anchor);
    op.deref(ctx).get_result(0)
}

fn replace_one_result(ctx: &mut Context, old_op: Ptr<Operation>, replacement: Value) -> Result<()> {
    if old_op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(
            old_op.deref(ctx).loc(),
            "legacy legalization expected exactly one result"
        );
    }
    let old = old_op.deref(ctx).get_result(0);
    if old.get_type(ctx) != replacement.get_type(ctx) {
        return pliron::input_err!(
            old_op.deref(ctx).loc(),
            "legacy legalization produced a replacement with a different type"
        );
    }
    old.replace_all_uses_with(ctx, &replacement);
    old_op.unlink(ctx);
    Ok(())
}

fn integer_const(ctx: &mut Context, anchor: Ptr<Operation>, width: u32, bits: u128) -> Value {
    let ty = IntegerType::get(ctx, width, Signedness::Signless);
    let attr = IntegerAttr::new(
        ty,
        APInt::from_u128(
            bits,
            NonZeroUsize::new(width as usize).expect("integer width is non-zero"),
        ),
    );
    let op = llvm::ConstantOp::new(ctx, attr.into()).get_operation();
    insert_before(ctx, anchor, op)
}

fn bool_binop<T: Op + BinArithOp>(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    lhs: Value,
    rhs: Value,
) -> Value {
    let op = T::new(ctx, lhs, rhs).get_operation();
    insert_before(ctx, anchor, op)
}

fn shl(ctx: &mut Context, anchor: Ptr<Operation>, lhs: Value, rhs: Value) -> Value {
    let op = llvm::ShlOp::new(ctx, lhs, rhs).get_operation();
    op.deref_mut(ctx).attributes.set(
        llvm_export::op_interfaces::ATTR_KEY_INTEGER_OVERFLOW_FLAGS.clone(),
        IntegerOverflowFlagsAttr::default(),
    );
    insert_before(ctx, anchor, op)
}

fn icmp(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    predicate: ICmpPredicateAttr,
    lhs: Value,
    rhs: Value,
) -> Value {
    let op = llvm::ICmpOp::new(ctx, predicate, lhs, rhs).get_operation();
    insert_before(ctx, anchor, op)
}

fn select(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    condition: Value,
    if_true: Value,
    if_false: Value,
) -> Value {
    let op = llvm::SelectOp::new(ctx, condition, if_true, if_false).get_operation();
    insert_before(ctx, anchor, op)
}

fn rewrite_fneg(ctx: &mut Context, op: Ptr<Operation>) -> Result<()> {
    let value = op.deref(ctx).get_operand(0);
    let float_ty = value.get_type(ctx);
    let width = float_width(ctx, float_ty).ok_or_else(|| {
        pliron::input_error!(
            op.deref(ctx).loc(),
            "legacy fneg legalization supports only scalar f32 and f64"
        )
    })?;
    let int_ty: TypeHandle = IntegerType::get(ctx, width, Signedness::Signless).into();

    // IEEE negation is exactly a sign-bit toggle, including for NaNs and
    // signed zero.  This is stronger than `0.0 - x`, which changes NaNs and
    // mishandles +0.0.
    let as_int_op = llvm::BitcastOp::new(ctx, value, int_ty).get_operation();
    let as_int = insert_before(ctx, op, as_int_op);
    let sign_mask = integer_const(ctx, op, width, 1_u128 << (width - 1));
    let toggled_op = llvm::XorOp::new(ctx, as_int, sign_mask).get_operation();
    let toggled = insert_before(ctx, op, toggled_op);
    let result_op = llvm::BitcastOp::new(ctx, toggled, float_ty).get_operation();
    let result = insert_before(ctx, op, result_op);
    replace_one_result(ctx, op, result)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IntegerSatKind {
    SignedAdd,
    SignedSub,
    UnsignedAdd,
    UnsignedSub,
}

fn parse_integer_sat_name(name: &str) -> Option<(IntegerSatKind, u32)> {
    let (kind, width) = if let Some(rest) = name.strip_prefix("llvm_sadd_sat_i") {
        (IntegerSatKind::SignedAdd, rest)
    } else if let Some(rest) = name.strip_prefix("llvm_ssub_sat_i") {
        (IntegerSatKind::SignedSub, rest)
    } else if let Some(rest) = name.strip_prefix("llvm_uadd_sat_i") {
        (IntegerSatKind::UnsignedAdd, rest)
    } else if let Some(rest) = name.strip_prefix("llvm_usub_sat_i") {
        (IntegerSatKind::UnsignedSub, rest)
    } else {
        return None;
    };
    let width = width.parse().ok()?;
    (width > 0 && width <= 128).then_some((kind, width))
}

fn validate_integer_sat_call(ctx: &Context, op: Ptr<Operation>, name: &str) -> Result<()> {
    let (_, expected_width) = parse_integer_sat_name(name).expect("validated intrinsic name");
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 || op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} must have two operands and one result"
        );
    }
    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let width = scalar_integer_width(ctx, result_ty);
    if width != Some(expected_width) || operands.iter().any(|v| v.get_type(ctx) != result_ty) {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} has a signature inconsistent with its overload suffix"
        );
    }
    Ok(())
}

fn rewrite_integer_sat(ctx: &mut Context, op: Ptr<Operation>, name: &str) -> Result<()> {
    let (kind, width) = parse_integer_sat_name(name).expect("validated intrinsic name");
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let (lhs, rhs) = (operands[0], operands[1]);
    let flags = IntegerOverflowFlagsAttr::default();
    let arithmetic = match kind {
        IntegerSatKind::SignedAdd | IntegerSatKind::UnsignedAdd => {
            llvm::AddOp::new_with_overflow_flag(ctx, lhs, rhs, flags).get_operation()
        }
        IntegerSatKind::SignedSub | IntegerSatKind::UnsignedSub => {
            llvm::SubOp::new_with_overflow_flag(ctx, lhs, rhs, flags).get_operation()
        }
    };
    let value = insert_before(ctx, op, arithmetic);

    let result = match kind {
        IntegerSatKind::UnsignedAdd => {
            let overflow = icmp(ctx, op, ICmpPredicateAttr::ULT, value, lhs);
            let max = integer_const(ctx, op, width, mask_for_width(width));
            select(ctx, op, overflow, max, value)
        }
        IntegerSatKind::UnsignedSub => {
            let underflow = icmp(ctx, op, ICmpPredicateAttr::ULT, lhs, rhs);
            let zero = integer_const(ctx, op, width, 0);
            select(ctx, op, underflow, zero, value)
        }
        IntegerSatKind::SignedAdd | IntegerSatKind::SignedSub => {
            let zero = integer_const(ctx, op, width, 0);
            let lhs_negative = icmp(ctx, op, ICmpPredicateAttr::SLT, lhs, zero);
            let rhs_negative = icmp(ctx, op, ICmpPredicateAttr::SLT, rhs, zero);
            let result_negative = icmp(ctx, op, ICmpPredicateAttr::SLT, value, zero);
            let lhs_rhs_relation = icmp(
                ctx,
                op,
                if kind == IntegerSatKind::SignedAdd {
                    ICmpPredicateAttr::EQ
                } else {
                    ICmpPredicateAttr::NE
                },
                lhs_negative,
                rhs_negative,
            );
            let sign_changed = icmp(
                ctx,
                op,
                ICmpPredicateAttr::NE,
                lhs_negative,
                result_negative,
            );
            let overflow = bool_binop::<llvm::AndOp>(ctx, op, lhs_rhs_relation, sign_changed);
            let min = integer_const(ctx, op, width, 1_u128 << (width - 1));
            let max = integer_const(ctx, op, width, (1_u128 << (width - 1)) - 1);
            let clamp = select(ctx, op, lhs_negative, min, max);
            select(ctx, op, overflow, clamp, value)
        }
    };
    replace_one_result(ctx, op, result)
}

fn mask_for_width(width: u32) -> u128 {
    if width == 128 {
        u128::MAX
    } else {
        (1_u128 << width) - 1
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LegacyBitRewrite {
    InvalidBswapI8,
    FunnelLeftI128,
    FunnelRightI128,
    BswapI128,
    BitreverseI128,
    CtpopI128,
    CtlzI128,
    CttzI128,
}

/// NVVM only accepts bit-manipulation intrinsics through i64. Rust exposes the
/// same operations for i128, so split them into exact i64 operations before
/// handing the module to libNVVM. `llvm.bswap.i8` is not valid LLVM IR at all;
/// recognize it only so malformed input fails during validation rather than in
/// the textual LLVM verifier. Rust's `u8::swap_bytes` is lowered to an identity
/// before this pass.
fn legacy_bit_rewrite(name: &str) -> Option<LegacyBitRewrite> {
    match name {
        "llvm_bswap_i8" => Some(LegacyBitRewrite::InvalidBswapI8),
        "llvm_fshl_i128" => Some(LegacyBitRewrite::FunnelLeftI128),
        "llvm_fshr_i128" => Some(LegacyBitRewrite::FunnelRightI128),
        "llvm_bswap_i128" => Some(LegacyBitRewrite::BswapI128),
        "llvm_bitreverse_i128" => Some(LegacyBitRewrite::BitreverseI128),
        "llvm_ctpop_i128" => Some(LegacyBitRewrite::CtpopI128),
        "llvm_ctlz_i128" => Some(LegacyBitRewrite::CtlzI128),
        "llvm_cttz_i128" => Some(LegacyBitRewrite::CttzI128),
        _ => None,
    }
}

fn validate_legacy_bit_call(ctx: &Context, op: Ptr<Operation>, name: &str) -> Result<()> {
    let kind = legacy_bit_rewrite(name).expect("validated legacy bit intrinsic name");
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} must produce exactly one result"
        );
    }
    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let i1_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
    let expected_width = if kind == LegacyBitRewrite::InvalidBswapI8 {
        8
    } else {
        128
    };
    if scalar_integer_width(ctx, result_ty) != Some(expected_width) {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} result type does not match its overload suffix"
        );
    }

    let valid = match kind {
        LegacyBitRewrite::FunnelLeftI128 | LegacyBitRewrite::FunnelRightI128 => {
            operands.len() == 3
                && operands
                    .iter()
                    .all(|value| value.get_type(ctx) == result_ty)
        }
        LegacyBitRewrite::CtlzI128 | LegacyBitRewrite::CttzI128 => {
            operands.len() == 2
                && operands[0].get_type(ctx) == result_ty
                && operands[1].get_type(ctx) == i1_ty
        }
        _ => operands.len() == 1 && operands[0].get_type(ctx) == result_ty,
    };
    if !valid {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} has a signature inconsistent with its overload suffix"
        );
    }

    if kind == LegacyBitRewrite::InvalidBswapI8 {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} is not a valid LLVM intrinsic: byte-swap widths must be a multiple of 16 bits"
        );
    }

    if matches!(
        kind,
        LegacyBitRewrite::CtlzI128 | LegacyBitRewrite::CttzI128
    ) && literal_i1_constant(ctx, operands[1]).is_none()
    {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} is_zero_poison flag must be an immediate i1 constant (0 or 1)"
        );
    }
    Ok(())
}

fn literal_i1_constant(ctx: &Context, value: Value) -> Option<bool> {
    let defining_op = value.defining_op()?;
    let constant = Operation::get_op::<llvm::ConstantOp>(defining_op, ctx)?;
    let attr = constant.get_value(ctx);
    let integer = attr.downcast_ref::<IntegerAttr>()?;
    match integer.value().to_u64() {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    }
}

fn split_i128(ctx: &mut Context, anchor: Ptr<Operation>, value: Value) -> (Value, Value) {
    let i64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Signless).into();
    let low_op = llvm::TruncOp::new(ctx, value, i64_ty).get_operation();
    let low = insert_before(ctx, anchor, low_op);
    let shift = integer_const(ctx, anchor, 128, 64);
    let shifted_op = llvm::LShrOp::new(ctx, value, shift).get_operation();
    let shifted = insert_before(ctx, anchor, shifted_op);
    let high_op = llvm::TruncOp::new(ctx, shifted, i64_ty).get_operation();
    let high = insert_before(ctx, anchor, high_op);
    (low, high)
}

fn zext_to_i128(ctx: &mut Context, anchor: Ptr<Operation>, value: Value) -> Value {
    let i128_ty: TypeHandle = IntegerType::get(ctx, 128, Signedness::Signless).into();
    let extend = llvm::ZExtOp::new_with_nneg(ctx, value, i128_ty, false).get_operation();
    insert_before(ctx, anchor, extend)
}

fn call_i64_intrinsic(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    name: &str,
    args: Vec<Value>,
) -> Result<Value> {
    let i64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Signless).into();
    let param_types = args.iter().map(|value| value.get_type(ctx)).collect();
    let func_ty = llvm_types::FuncType::get(ctx, i64_ty, param_types, false);
    let parent_block = anchor.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(
            anchor.deref(ctx).loc(),
            "intrinsic call has no parent block"
        )
    })?;
    helpers::ensure_intrinsic_declared(ctx, parent_block, name, func_ty)
        .map_err(|error| pliron::input_error!(anchor.deref(ctx).loc(), "{error}"))?;
    let call = llvm::CallOp::new(
        ctx,
        CallOpCallable::Direct(name.try_into().map_err(|error| {
            pliron::input_error!(anchor.deref(ctx).loc(), "invalid intrinsic name: {error}")
        })?),
        func_ty,
        args,
    );
    Ok(insert_before(ctx, anchor, call.get_operation()))
}

fn combine_reversed_i64_halves(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    low: Value,
    high: Value,
) -> Value {
    // Reversing bytes or bits across the complete i128 swaps the transformed
    // halves: transformed(low) becomes the high half of the result.
    let low = zext_to_i128(ctx, anchor, low);
    let high = zext_to_i128(ctx, anchor, high);
    let shift = integer_const(ctx, anchor, 128, 64);
    let low_as_high = shl(ctx, anchor, low, shift);
    let combined = llvm::OrOp::new(ctx, low_as_high, high).get_operation();
    insert_before(ctx, anchor, combined)
}

fn rewrite_legacy_bit_intrinsic(ctx: &mut Context, op: Ptr<Operation>, name: &str) -> Result<()> {
    let kind = legacy_bit_rewrite(name).expect("validated legacy bit intrinsic name");
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let result = match kind {
        LegacyBitRewrite::InvalidBswapI8 => {
            return pliron::input_err!(
                op.deref(ctx).loc(),
                "{name} reached rewriting despite being rejected during validation"
            );
        }
        LegacyBitRewrite::FunnelLeftI128 | LegacyBitRewrite::FunnelRightI128 => {
            let mask = integer_const(ctx, op, 128, 127);
            let shift = bool_binop::<llvm::AndOp>(ctx, op, operands[2], mask);
            let zero = integer_const(ctx, op, 128, 0);
            let negative_shift_op = llvm::SubOp::new_with_overflow_flag(
                ctx,
                zero,
                shift,
                IntegerOverflowFlagsAttr::default(),
            )
            .get_operation();
            let negative_shift = insert_before(ctx, op, negative_shift_op);
            let reverse_shift = bool_binop::<llvm::AndOp>(ctx, op, negative_shift, mask);
            let is_zero = icmp(ctx, op, ICmpPredicateAttr::EQ, shift, zero);
            let (left, right, zero_result) = if kind == LegacyBitRewrite::FunnelLeftI128 {
                (operands[0], operands[1], operands[0])
            } else {
                (operands[0], operands[1], operands[1])
            };
            let left_shift = if kind == LegacyBitRewrite::FunnelLeftI128 {
                shift
            } else {
                reverse_shift
            };
            let right_shift = if kind == LegacyBitRewrite::FunnelLeftI128 {
                reverse_shift
            } else {
                shift
            };
            let left = shl(ctx, op, left, left_shift);
            let right_op = llvm::LShrOp::new(ctx, right, right_shift).get_operation();
            let right = insert_before(ctx, op, right_op);
            let combined_op = llvm::OrOp::new(ctx, left, right).get_operation();
            let combined = insert_before(ctx, op, combined_op);
            select(ctx, op, is_zero, zero_result, combined)
        }
        LegacyBitRewrite::BswapI128 | LegacyBitRewrite::BitreverseI128 => {
            let (low, high) = split_i128(ctx, op, operands[0]);
            let intrinsic = if kind == LegacyBitRewrite::BswapI128 {
                "llvm_bswap_i64"
            } else {
                "llvm_bitreverse_i64"
            };
            let low = call_i64_intrinsic(ctx, op, intrinsic, vec![low])?;
            let high = call_i64_intrinsic(ctx, op, intrinsic, vec![high])?;
            combine_reversed_i64_halves(ctx, op, low, high)
        }
        LegacyBitRewrite::CtpopI128 => {
            let (low, high) = split_i128(ctx, op, operands[0]);
            let low = call_i64_intrinsic(ctx, op, "llvm_ctpop_i64", vec![low])?;
            let high = call_i64_intrinsic(ctx, op, "llvm_ctpop_i64", vec![high])?;
            let low = zext_to_i128(ctx, op, low);
            let high = zext_to_i128(ctx, op, high);
            let sum = llvm::AddOp::new_with_overflow_flag(
                ctx,
                low,
                high,
                IntegerOverflowFlagsAttr::default(),
            )
            .get_operation();
            insert_before(ctx, op, sum)
        }
        LegacyBitRewrite::CtlzI128 | LegacyBitRewrite::CttzI128 => {
            let (low, high) = split_i128(ctx, op, operands[0]);
            let flag = operands[1];
            let (primary, secondary, intrinsic) = if kind == LegacyBitRewrite::CtlzI128 {
                (high, low, "llvm_ctlz_i64")
            } else {
                (low, high, "llvm_cttz_i64")
            };
            let primary_count = call_i64_intrinsic(ctx, op, intrinsic, vec![primary, flag])?;
            let secondary_count = call_i64_intrinsic(ctx, op, intrinsic, vec![secondary, flag])?;
            let zero64 = integer_const(ctx, op, 64, 0);
            let primary_is_zero = icmp(ctx, op, ICmpPredicateAttr::EQ, primary, zero64);
            let sixty_four = integer_const(ctx, op, 64, 64);
            let extended_count = llvm::AddOp::new_with_overflow_flag(
                ctx,
                sixty_four,
                secondary_count,
                IntegerOverflowFlagsAttr::default(),
            )
            .get_operation();
            let secondary_count = insert_before(ctx, op, extended_count);
            let count = select(ctx, op, primary_is_zero, secondary_count, primary_count);
            zext_to_i128(ctx, op, count)
        }
    };
    replace_one_result(ctx, op, result)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FloatToIntSatKind {
    Signed,
    Unsigned,
}

fn parse_float_to_int_sat_name(name: &str) -> Option<(FloatToIntSatKind, u32, u32)> {
    let (kind, rest) = if let Some(rest) = name.strip_prefix("llvm_fptosi_sat_i") {
        (FloatToIntSatKind::Signed, rest)
    } else if let Some(rest) = name.strip_prefix("llvm_fptoui_sat_i") {
        (FloatToIntSatKind::Unsigned, rest)
    } else {
        return None;
    };
    let (int_width, float_width) = rest.split_once("_f")?;
    let int_width: u32 = int_width.parse().ok()?;
    let float_width: u32 = float_width.parse().ok()?;
    (int_width > 0 && int_width <= 128 && matches!(float_width, 16 | 32 | 64)).then_some((
        kind,
        int_width,
        float_width,
    ))
}

fn validate_float_to_int_sat_call(ctx: &Context, op: Ptr<Operation>, name: &str) -> Result<()> {
    let (_, expected_int_width, expected_float_width) =
        parse_float_to_int_sat_name(name).expect("validated intrinsic name");
    if expected_float_width == 16 {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "legacy NVVM saturating conversion from f16 is not portable to the supported CUDA 12 legacy dialect"
        );
    }
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 || op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} must have one operand and one result"
        );
    }
    let result_width = scalar_integer_width(ctx, op.deref(ctx).get_result(0).get_type(ctx));
    let source_width = float_width(ctx, operands[0].get_type(ctx));
    if result_width != Some(expected_int_width) || source_width != Some(expected_float_width) {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} has a signature inconsistent with its overload suffix"
        );
    }
    Ok(())
}

fn rewrite_float_to_int_sat(ctx: &mut Context, op: Ptr<Operation>, name: &str) -> Result<()> {
    let (kind, width, source_width) =
        parse_float_to_int_sat_name(name).expect("validated intrinsic name");
    let mut source = op.deref(ctx).get_operand(0);

    // If f16 support is added later, extending to f32 preserves every f16
    // value exactly and allows the same clamp algorithm to be reused.
    let compare_width = if source_width == 16 {
        let f32_ty: TypeHandle = FP32Type::get(ctx).into();
        let extend = llvm::FPExtOp::new(ctx, source, f32_ty);
        extend.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
        source = insert_before(ctx, op, extend.get_operation());
        32
    } else {
        source_width
    };

    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let raw = match kind {
        FloatToIntSatKind::Signed => llvm::FPToSIOp::new(ctx, source, result_ty).get_operation(),
        FloatToIntSatKind::Unsigned => llvm::FPToUIOp::new(ctx, source, result_ty).get_operation(),
    };
    let raw = insert_before(ctx, op, raw);

    let (low_float, high_float) = match kind {
        FloatToIntSatKind::Signed => (
            -(2.0_f64).powi(width as i32 - 1),
            (2.0_f64).powi(width as i32 - 1),
        ),
        FloatToIntSatKind::Unsigned => (0.0, (2.0_f64).powi(width as i32)),
    };
    let low = float_const(ctx, op, compare_width, low_float)?;
    let high = float_const(ctx, op, compare_width, high_float)?;
    let is_low = fcmp(ctx, op, FCmpPredicateAttr::OLE, source, low);
    let is_high = fcmp(ctx, op, FCmpPredicateAttr::OGE, source, high);
    let is_nan = fcmp(ctx, op, FCmpPredicateAttr::UNO, source, source);

    let zero = integer_const(ctx, op, width, 0);
    let (low_value, high_value) = match kind {
        FloatToIntSatKind::Signed => (
            integer_const(ctx, op, width, 1_u128 << (width - 1)),
            integer_const(ctx, op, width, (1_u128 << (width - 1)) - 1),
        ),
        FloatToIntSatKind::Unsigned => (zero, integer_const(ctx, op, width, mask_for_width(width))),
    };
    let high_or_raw = select(ctx, op, is_high, high_value, raw);
    let low_or_bounded = select(ctx, op, is_low, low_value, high_or_raw);
    let result = select(ctx, op, is_nan, zero, low_or_bounded);
    replace_one_result(ctx, op, result)
}

fn float_const(ctx: &mut Context, anchor: Ptr<Operation>, width: u32, value: f64) -> Result<Value> {
    let attr = match width {
        32 => FPSingleAttr::from(value as f32).into(),
        64 => FPDoubleAttr::from(value).into(),
        _ => {
            return pliron::input_err!(
                anchor.deref(ctx).loc(),
                "legacy float clamp constants support only f32 and f64"
            );
        }
    };
    let const_op = llvm::ConstantOp::new(ctx, attr).get_operation();
    Ok(insert_before(ctx, anchor, const_op))
}

fn fcmp(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    predicate: FCmpPredicateAttr,
    lhs: Value,
    rhs: Value,
) -> Value {
    let compare = llvm::FCmpOp::new(ctx, predicate, lhs, rhs);
    compare.set_fast_math_flags(ctx, FastmathFlagsAttr::default());
    insert_before(ctx, anchor, compare.get_operation())
}

fn float_width(ctx: &Context, ty: TypeHandle) -> Option<u32> {
    let ty = ty.deref(ctx);
    if ty.is::<FP16Type>() {
        Some(16)
    } else if ty.is::<FP32Type>() {
        Some(32)
    } else if ty.is::<FP64Type>() {
        Some(64)
    } else {
        None
    }
}

fn scalar_integer_width(ctx: &Context, ty: TypeHandle) -> Option<u32> {
    ty.deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|t| t.width())
}

fn shuffle_mode(name: &str) -> Option<u32> {
    let stem = name
        .strip_suffix("_i32")
        .or_else(|| name.strip_suffix("_f32"))?;
    match stem {
        "llvm_nvvm_shfl_sync_idx" => Some(0),
        "llvm_nvvm_shfl_sync_up" => Some(1),
        "llvm_nvvm_shfl_sync_down" => Some(2),
        "llvm_nvvm_shfl_sync_bfly" => Some(3),
        _ => None,
    }
}

fn validate_shuffle_call(ctx: &Context, op: Ptr<Operation>, name: &str) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 4 || op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} must have four operands and one result"
        );
    }
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let value_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let expected_value_ty = if name.ends_with("_f32") {
        FP32Type::get(ctx).into()
    } else {
        i32_ty
    };
    if value_ty != expected_value_ty
        || operands[1].get_type(ctx) != expected_value_ty
        || operands[0].get_type(ctx) != i32_ty
        || operands[2].get_type(ctx) != i32_ty
        || operands[3].get_type(ctx) != i32_ty
    {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} has an invalid legacy-shuffle signature"
        );
    }
    Ok(())
}

fn rewrite_shuffle(ctx: &mut Context, op: Ptr<Operation>, name: &str) -> Result<()> {
    let mode = shuffle_mode(name).expect("validated shuffle name");
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let i1_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
    let struct_ty = llvm_types::StructType::get_unnamed(ctx, vec![i32_ty, i1_ty]);
    let func_ty = llvm_types::FuncType::get(
        ctx,
        struct_ty.into(),
        vec![i32_ty, i32_ty, i32_ty, i32_ty, i32_ty],
        false,
    );
    let parent_block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(op.deref(ctx).loc(), "shuffle call has no parent block")
    })?;
    helpers::ensure_intrinsic_declared(ctx, parent_block, "llvm_nvvm_shfl_sync_i32", func_ty)
        .map_err(|e| pliron::input_error!(op.deref(ctx).loc(), "{e}"))?;

    let mut value = operands[1];
    if name.ends_with("_f32") {
        let cast_op = llvm::BitcastOp::new(ctx, value, i32_ty).get_operation();
        value = insert_before(ctx, op, cast_op);
    }
    let mode = integer_const(ctx, op, 32, mode as u128);
    let callee = CallOpCallable::Direct(
        "llvm_nvvm_shfl_sync_i32"
            .try_into()
            .expect("static intrinsic name is valid"),
    );
    let call = llvm::CallOp::new(
        ctx,
        callee,
        func_ty,
        vec![operands[0], mode, value, operands[2], operands[3]],
    );
    let aggregate = insert_before(ctx, op, call.get_operation());
    let extracted = llvm::ExtractValueOp::new(ctx, aggregate, vec![0])
        .map_err(|e| pliron::input_error!(op.deref(ctx).loc(), "{e}"))?;
    let mut result = insert_before(ctx, op, extracted.get_operation());
    if name.ends_with("_f32") {
        let f32_ty: TypeHandle = FP32Type::get(ctx).into();
        let cast_op = llvm::BitcastOp::new(ctx, result, f32_ty).get_operation();
        result = insert_before(ctx, op, cast_op);
    }
    replace_one_result(ctx, op, result)
}

fn vote_mode(name: &str) -> Option<(u32, u32)> {
    match name {
        "llvm_nvvm_vote_all_sync" => Some((0, 1)),
        "llvm_nvvm_vote_any_sync" => Some((1, 1)),
        "llvm_nvvm_vote_ballot_sync" => Some((3, 0)),
        _ => None,
    }
}

fn validate_vote_call(ctx: &Context, op: Ptr<Operation>, name: &str) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 || op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} must have two operands and one result"
        );
    }
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let i1_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
    let expected_result = if name.ends_with("ballot_sync") {
        i32_ty
    } else {
        i1_ty
    };
    if operands[0].get_type(ctx) != i32_ty
        || operands[1].get_type(ctx) != i1_ty
        || op.deref(ctx).get_result(0).get_type(ctx) != expected_result
    {
        return pliron::input_err!(
            op.deref(ctx).loc(),
            "{name} has an invalid legacy-vote signature"
        );
    }
    Ok(())
}

fn rewrite_vote(ctx: &mut Context, op: Ptr<Operation>, name: &str) -> Result<()> {
    let (mode, field) = vote_mode(name).expect("validated vote name");
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let i1_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
    let struct_ty = llvm_types::StructType::get_unnamed(ctx, vec![i32_ty, i1_ty]);
    let func_ty =
        llvm_types::FuncType::get(ctx, struct_ty.into(), vec![i32_ty, i32_ty, i1_ty], false);
    let parent_block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(op.deref(ctx).loc(), "vote call has no parent block")
    })?;
    helpers::ensure_intrinsic_declared(ctx, parent_block, "llvm_nvvm_vote_sync", func_ty)
        .map_err(|e| pliron::input_error!(op.deref(ctx).loc(), "{e}"))?;
    let mode = integer_const(ctx, op, 32, mode as u128);
    let call = llvm::CallOp::new(
        ctx,
        CallOpCallable::Direct(
            "llvm_nvvm_vote_sync"
                .try_into()
                .expect("static intrinsic name is valid"),
        ),
        func_ty,
        vec![operands[0], mode, operands[1]],
    );
    let aggregate = insert_before(ctx, op, call.get_operation());
    let extract = llvm::ExtractValueOp::new(ctx, aggregate, vec![field])
        .map_err(|e| pliron::input_error!(op.deref(ctx).loc(), "{e}"))?;
    let result = insert_before(ctx, op, extract.get_operation());
    replace_one_result(ctx, op, result)
}

fn legacy_memory_intrinsic_name(name: &str) -> Option<String> {
    for stem in ["memcpy", "memmove"] {
        if let Some(rest) = name.strip_prefix(&format!("llvm_{stem}_p")) {
            // Modern opaque-pointer spelling:
            // llvm.{memcpy,memmove}.p<dst>.p<src>.i<bits>
            let (dst, rest) = rest.split_once("_p")?;
            let (src, bits) = rest.split_once("_i")?;
            dst.parse::<u32>().ok()?;
            src.parse::<u32>().ok()?;
            bits.parse::<u32>().ok()?;
            return Some(format!("llvm_{stem}_p{dst}i8_p{src}i8_i{bits}"));
        }
    }

    if let Some(rest) = name.strip_prefix("llvm_memset_p") {
        // Modern opaque-pointer spelling: llvm.memset.p<dst>.i<bits>
        let (dst, bits) = rest.split_once("_i")?;
        dst.parse::<u32>().ok()?;
        bits.parse::<u32>().ok()?;
        return Some(format!("llvm_memset_p{dst}i8_i{bits}"));
    }

    None
}

fn rewrite_renamed_call(
    ctx: &mut Context,
    op: Ptr<Operation>,
    replacement_name: &str,
) -> Result<()> {
    let call = Operation::get_op::<llvm::CallOp>(op, ctx).expect("validated call op");
    let args = call.args(ctx);
    let func_ty = call.callee_type(ctx);
    let func_ty = pliron::r#type::TypedHandle::from_handle(func_ty, ctx).map_err(|e| {
        pliron::input_error!(
            op.deref(ctx).loc(),
            "memory intrinsic call has invalid function type: {e}"
        )
    })?;
    let parent_block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(
            op.deref(ctx).loc(),
            "memory intrinsic call has no parent block"
        )
    })?;
    helpers::ensure_intrinsic_declared(ctx, parent_block, replacement_name, func_ty)
        .map_err(|e| pliron::input_error!(op.deref(ctx).loc(), "{e}"))?;
    let replacement = llvm::CallOp::new(
        ctx,
        CallOpCallable::Direct(replacement_name.try_into().map_err(|e| {
            pliron::input_error!(
                op.deref(ctx).loc(),
                "invalid legacy memory intrinsic name: {e}"
            )
        })?),
        func_ty,
        args,
    );
    let result = insert_before(ctx, op, replacement.get_operation());
    replace_one_result(ctx, op, result)
}

fn remove_obsolete_declarations(
    ctx: &mut Context,
    module: Ptr<Operation>,
    obsolete_names: &[String],
) {
    if obsolete_names.is_empty() {
        return;
    }
    let Some(region) = module.deref(ctx).regions().next() else {
        return;
    };
    let Some(block) = region.deref(ctx).iter(ctx).next() else {
        return;
    };
    let top_level: Vec<_> = block.deref(ctx).iter(ctx).collect();
    for op in top_level {
        let Some(func) = Operation::get_op::<llvm::FuncOp>(op, ctx) else {
            continue;
        };
        let name = func.get_symbol_name(ctx).to_string();
        if obsolete_names.iter().any(|old| old == &name) {
            op.unlink(ctx);
        }
    }
}

fn verify_legacy_subset(ctx: &Context, module: Ptr<Operation>) -> Result<()> {
    let mut ops = Vec::new();
    collect_ops(ctx, module, &mut ops);
    for op in ops {
        reject_unsupported_op(ctx, op)?;
        if Operation::get_op::<llvm::FNegOp>(op, ctx).is_some() {
            return pliron::input_err!(
                op.deref(ctx).loc(),
                "legacy NVVM legalization left an llvm.fneg behind"
            );
        }
        let nneg: Identifier = NNEG_ATTR
            .try_into()
            .expect("static attribute name is valid");
        let has_true_nneg = op
            .deref(ctx)
            .attributes
            .get::<BoolAttr>(&nneg)
            .is_some_and(|flag| bool::from(flag.clone()));
        if has_true_nneg {
            return pliron::input_err!(
                op.deref(ctx).loc(),
                "legacy NVVM legalization left a true LLVM nneg flag behind"
            );
        }
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(ctx) else {
            continue;
        };
        let name = callee.to_string();
        if parse_integer_sat_name(&name).is_some()
            || parse_float_to_int_sat_name(&name).is_some()
            || legacy_bit_rewrite(&name).is_some()
            || shuffle_mode(&name).is_some()
            || vote_mode(&name).is_some()
            || legacy_memory_intrinsic_name(&name).is_some()
        {
            return pliron::input_err!(
                op.deref(ctx).loc(),
                "legacy NVVM legalization left unsupported call @{name} behind"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use llvm_export::{
        op_interfaces::{CastOpWithNNegInterface, NNegFlag},
        types::{FuncType, PointerType, VoidType},
    };
    use pliron::{
        basic_block::BasicBlock,
        builtin::{op_interfaces::SymbolOpInterface, ops::ModuleOp},
        common_traits::Verify,
        printable::Printable,
    };

    fn module_block(ctx: &Context, module: &ModuleOp) -> Ptr<BasicBlock> {
        module
            .get_operation()
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .iter(ctx)
            .next()
            .expect("module has an entry block")
    }

    fn function(
        ctx: &mut Context,
        module: &ModuleOp,
        name: &str,
        result: TypeHandle,
        params: Vec<TypeHandle>,
    ) -> (llvm::FuncOp, Ptr<BasicBlock>) {
        let ty = FuncType::get(ctx, result, params, false);
        let func = llvm::FuncOp::new(ctx, name.try_into().unwrap(), ty);
        let entry = func.get_or_create_entry_block(ctx);
        func.get_operation()
            .insert_at_back(module_block(ctx, module), ctx);
        (func, entry)
    }

    fn operation_ids(ctx: &Context, root: Ptr<Operation>) -> Vec<String> {
        let mut ops = Vec::new();
        collect_ops(ctx, root, &mut ops);
        ops.into_iter()
            .map(|op| Operation::get_opid(op, ctx).to_string())
            .collect()
    }

    #[test]
    fn fneg_is_an_exact_sign_bit_toggle_and_module_still_verifies() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "fneg".try_into().unwrap());
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let (_func, entry) = function(&mut ctx, &module, "neg", f32_ty, vec![f32_ty]);
        let input = entry.deref(&ctx).get_argument(0);
        let neg =
            llvm::FNegOp::new_with_fast_math_flags(&mut ctx, input, FastmathFlagsAttr::default());
        neg.get_operation().insert_at_back(entry, &ctx);
        let neg_result = neg.get_operation().deref(&ctx).get_result(0);
        llvm::ReturnOp::new(&mut ctx, Some(neg_result))
            .get_operation()
            .insert_at_back(entry, &ctx);

        crate::legalize_for_nvvm(
            &mut ctx,
            module.get_operation(),
            llvm_export::export::NvvmIrDialect::LegacyLlvm7,
        )
        .unwrap();

        let ids = operation_ids(&ctx, module.get_operation());
        assert!(!ids.iter().any(|id| id == "llvm.fneg"), "{ids:?}");
        assert_eq!(ids.iter().filter(|id| *id == "llvm.bitcast").count(), 2);
        assert_eq!(ids.iter().filter(|id| *id == "llvm.xor").count(), 1);
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();
    }

    #[test]
    fn nneg_is_cleared_but_required_dialect_attribute_is_retained() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "nneg".try_into().unwrap());
        let i8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Signless).into();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let (_func, entry) = function(&mut ctx, &module, "extend", i32_ty, vec![i8_ty]);
        let input = entry.deref(&ctx).get_argument(0);
        let zext = llvm::ZExtOp::new_with_nneg(&mut ctx, input, i32_ty, true);
        zext.get_operation().insert_at_back(entry, &ctx);
        let result = zext.get_operation().deref(&ctx).get_result(0);
        llvm::ReturnOp::new(&mut ctx, Some(result))
            .get_operation()
            .insert_at_back(entry, &ctx);

        legalize_for_legacy_nvvm(&mut ctx, module.get_operation()).unwrap();

        assert!(!zext.nneg(&ctx));
        let key: Identifier = NNEG_ATTR.try_into().unwrap();
        assert!(
            zext.get_operation()
                .deref(&ctx)
                .attributes
                .get::<BoolAttr>(&key)
                .is_some(),
            "pliron-LLVM requires the false nneg attribute to remain"
        );
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();
    }

    fn direct_intrinsic_call(
        ctx: &mut Context,
        module: &ModuleOp,
        name: &str,
        result_ty: TypeHandle,
        param_tys: Vec<TypeHandle>,
    ) -> (llvm::CallOp, Ptr<BasicBlock>) {
        let void_ty: TypeHandle = VoidType::get(ctx).into();
        let (_func, entry) = function(ctx, module, "caller", void_ty, param_tys.clone());
        let func_ty = FuncType::get(ctx, result_ty, param_tys, false);
        helpers::ensure_intrinsic_declared(ctx, entry, name, func_ty).unwrap();
        let args = entry.deref(ctx).arguments().collect();
        let call = llvm::CallOp::new(
            ctx,
            CallOpCallable::Direct(name.try_into().unwrap()),
            func_ty,
            args,
        );
        call.get_operation().insert_at_back(entry, ctx);
        llvm::ReturnOp::new(ctx, None)
            .get_operation()
            .insert_at_back(entry, ctx);
        (call, entry)
    }

    #[test]
    fn integer_saturation_intrinsics_are_fully_expanded() {
        for name in [
            "llvm_sadd_sat_i32",
            "llvm_ssub_sat_i32",
            "llvm_uadd_sat_i32",
            "llvm_usub_sat_i32",
        ] {
            let mut ctx = Context::new();
            let module = ModuleOp::new(&mut ctx, name.try_into().unwrap());
            let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
            direct_intrinsic_call(&mut ctx, &module, name, i32_ty, vec![i32_ty, i32_ty]);

            legalize_for_legacy_nvvm(&mut ctx, module.get_operation()).unwrap();

            let ids = operation_ids(&ctx, module.get_operation());
            assert!(ids.iter().any(|id| id == "llvm.select"), "{name}: {ids:?}");
            assert!(ids.iter().any(|id| id == "llvm.icmp"), "{name}: {ids:?}");
            let old_decl_remains = module_block(&ctx, &module)
                .deref(&ctx)
                .iter(&ctx)
                .any(|op| {
                    Operation::get_op::<llvm::FuncOp>(op, &ctx)
                        .is_some_and(|f| f.get_symbol_name(&ctx).to_string() == name)
                });
            assert!(!old_decl_remains, "obsolete declaration @{name} remains");
            module.get_operation().deref(&ctx).verify(&ctx).unwrap();
        }
    }

    #[test]
    fn unsupported_i128_bit_intrinsics_expand_to_legacy_i64_operations() {
        for (name, parameter_count, helper) in [
            ("llvm_fshl_i128", 3, None),
            ("llvm_fshr_i128", 3, None),
            ("llvm_bswap_i128", 1, Some("llvm_bswap_i64")),
            ("llvm_bitreverse_i128", 1, Some("llvm_bitreverse_i64")),
            ("llvm_ctpop_i128", 1, Some("llvm_ctpop_i64")),
        ] {
            let mut ctx = Context::new();
            let module = ModuleOp::new(&mut ctx, name.try_into().unwrap());
            let i128_ty: TypeHandle = IntegerType::get(&ctx, 128, Signedness::Signless).into();
            direct_intrinsic_call(
                &mut ctx,
                &module,
                name,
                i128_ty,
                vec![i128_ty; parameter_count],
            );

            legalize_for_legacy_nvvm(&mut ctx, module.get_operation()).unwrap();

            let mut ops = Vec::new();
            collect_ops(&ctx, module.get_operation(), &mut ops);
            let calls: Vec<_> = ops
                .into_iter()
                .filter_map(|op| Operation::get_op::<llvm::CallOp>(op, &ctx))
                .filter_map(|call| match call.callee(&ctx) {
                    CallOpCallable::Direct(name) => Some(name.to_string()),
                    CallOpCallable::Indirect(_) => None,
                })
                .collect();
            assert!(!calls.iter().any(|callee| callee == name), "{calls:?}");
            if let Some(helper) = helper {
                assert!(calls.iter().any(|callee| callee == helper), "{calls:?}");
            }
            module.get_operation().deref(&ctx).verify(&ctx).unwrap();
        }

        for (name, helper) in [
            ("llvm_ctlz_i128", "llvm_ctlz_i64"),
            ("llvm_cttz_i128", "llvm_cttz_i64"),
        ] {
            for zero_poison in [false, true] {
                let mut ctx = Context::new();
                let module = ModuleOp::new(&mut ctx, name.try_into().unwrap());
                let i128_ty: TypeHandle = IntegerType::get(&ctx, 128, Signedness::Signless).into();
                let i1_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
                let void_ty: TypeHandle = VoidType::get(&ctx).into();
                let (_func, entry) = function(&mut ctx, &module, "caller", void_ty, vec![i128_ty]);
                let input = entry.deref(&ctx).get_argument(0);
                let flag = helpers::create_i1_constant(&mut ctx, entry, zero_poison).unwrap();
                let func_ty = FuncType::get(&ctx, i128_ty, vec![i128_ty, i1_ty], false);
                helpers::ensure_intrinsic_declared(&mut ctx, entry, name, func_ty).unwrap();
                let call = llvm::CallOp::new(
                    &mut ctx,
                    CallOpCallable::Direct(name.try_into().unwrap()),
                    func_ty,
                    vec![input, flag],
                );
                call.get_operation().insert_at_back(entry, &ctx);
                llvm::ReturnOp::new(&mut ctx, None)
                    .get_operation()
                    .insert_at_back(entry, &ctx);

                legalize_for_legacy_nvvm(&mut ctx, module.get_operation()).unwrap();

                let mut ops = Vec::new();
                collect_ops(&ctx, module.get_operation(), &mut ops);
                let calls: Vec<_> = ops
                    .into_iter()
                    .filter_map(|op| Operation::get_op::<llvm::CallOp>(op, &ctx))
                    .filter_map(|call| match call.callee(&ctx) {
                        CallOpCallable::Direct(name) => Some(name.to_string()),
                        CallOpCallable::Indirect(_) => None,
                    })
                    .collect();
                assert!(!calls.iter().any(|callee| callee == name), "{calls:?}");
                assert!(calls.iter().any(|callee| callee == helper), "{calls:?}");
                module.get_operation().deref(&ctx).verify(&ctx).unwrap();
            }
        }
    }

    #[test]
    fn invalid_bswap_i8_is_rejected_without_mutating_the_module() {
        for common_only in [false, true] {
            let mut ctx = Context::new();
            let module = ModuleOp::new(&mut ctx, "bswap_i8".try_into().unwrap());
            let i8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Signless).into();
            direct_intrinsic_call(&mut ctx, &module, "llvm_bswap_i8", i8_ty, vec![i8_ty]);

            let before = module.get_operation().deref(&ctx).disp(&ctx).to_string();
            let error = if common_only {
                legalize_nvvm_bit_intrinsics(&mut ctx, module.get_operation()).unwrap_err()
            } else {
                legalize_for_legacy_nvvm(&mut ctx, module.get_operation()).unwrap_err()
            };
            let after = module.get_operation().deref(&ctx).disp(&ctx).to_string();

            assert!(
                error
                    .disp(&ctx)
                    .to_string()
                    .contains("not a valid LLVM intrinsic")
            );
            assert_eq!(before, after);
        }
    }

    #[test]
    fn dynamic_count_zero_poison_flag_is_rejected_without_mutating_the_module() {
        for name in ["llvm_ctlz_i128", "llvm_cttz_i128"] {
            for common_only in [false, true] {
                let mut ctx = Context::new();
                let module = ModuleOp::new(&mut ctx, name.try_into().unwrap());
                let i128_ty: TypeHandle = IntegerType::get(&ctx, 128, Signedness::Signless).into();
                let i1_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
                direct_intrinsic_call(&mut ctx, &module, name, i128_ty, vec![i128_ty, i1_ty]);

                let before = module.get_operation().deref(&ctx).disp(&ctx).to_string();
                let error = if common_only {
                    legalize_nvvm_bit_intrinsics(&mut ctx, module.get_operation()).unwrap_err()
                } else {
                    legalize_for_legacy_nvvm(&mut ctx, module.get_operation()).unwrap_err()
                };
                let after = module.get_operation().deref(&ctx).disp(&ctx).to_string();

                assert!(
                    error
                        .disp(&ctx)
                        .to_string()
                        .contains("immediate i1 constant"),
                    "{}",
                    error.disp(&ctx)
                );
                assert_eq!(before, after, "{name} changed before validation failed");
            }
        }
    }

    #[test]
    fn common_nvvm_bit_legalizer_leaves_modern_operations_untouched() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "modern_bits".try_into().unwrap());
        let i128_ty: TypeHandle = IntegerType::get(&ctx, 128, Signedness::Signless).into();
        direct_intrinsic_call(&mut ctx, &module, "llvm_bswap_i128", i128_ty, vec![i128_ty]);

        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let (_func, entry) = function(&mut ctx, &module, "modern_fneg", f32_ty, vec![f32_ty]);
        let input = entry.deref(&ctx).get_argument(0);
        let fneg =
            llvm::FNegOp::new_with_fast_math_flags(&mut ctx, input, FastmathFlagsAttr::default());
        fneg.get_operation().insert_at_back(entry, &ctx);
        let result = fneg.get_operation().deref(&ctx).get_result(0);
        llvm::ReturnOp::new(&mut ctx, Some(result))
            .get_operation()
            .insert_at_back(entry, &ctx);

        crate::legalize_for_nvvm(
            &mut ctx,
            module.get_operation(),
            llvm_export::export::NvvmIrDialect::Modern,
        )
        .unwrap();

        let ids = operation_ids(&ctx, module.get_operation());
        assert!(ids.iter().any(|id| id == "llvm.fneg"), "{ids:?}");
        let mut calls = Vec::new();
        let mut ops = Vec::new();
        collect_ops(&ctx, module.get_operation(), &mut ops);
        for op in ops {
            if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
                && let CallOpCallable::Direct(callee) = call.callee(&ctx)
            {
                calls.push(callee.to_string());
            }
        }
        assert!(!calls.iter().any(|name| name == "llvm_bswap_i128"));
        assert!(calls.iter().any(|name| name == "llvm_bswap_i64"));
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();
    }

    #[test]
    fn float_to_int_saturation_handles_bounds_and_nan_without_modern_intrinsic() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "fptosi_sat".try_into().unwrap());
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        direct_intrinsic_call(
            &mut ctx,
            &module,
            "llvm_fptosi_sat_i32_f32",
            i32_ty,
            vec![f32_ty],
        );

        legalize_for_legacy_nvvm(&mut ctx, module.get_operation()).unwrap();

        let ids = operation_ids(&ctx, module.get_operation());
        assert_eq!(ids.iter().filter(|id| *id == "llvm.fcmp").count(), 3);
        assert_eq!(ids.iter().filter(|id| *id == "llvm.select").count(), 3);
        assert_eq!(ids.iter().filter(|id| *id == "llvm.fptosi").count(), 1);
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();
    }

    #[test]
    fn modern_shuffle_and_vote_calls_use_legacy_aggregate_intrinsics() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "warp".try_into().unwrap());
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        direct_intrinsic_call(
            &mut ctx,
            &module,
            "llvm_nvvm_shfl_sync_bfly_i32",
            i32_ty,
            vec![i32_ty, i32_ty, i32_ty, i32_ty],
        );

        // Use a distinct caller name so this second helper can coexist in the
        // same module without violating symbol uniqueness.
        let i1_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let vote_ty = FuncType::get(&ctx, i1_ty, vec![i32_ty, i1_ty], false);
        let void_ty: TypeHandle = VoidType::get(&ctx).into();
        let (_vote_func, vote_entry) = function(
            &mut ctx,
            &module,
            "vote_caller",
            void_ty,
            vec![i32_ty, i1_ty],
        );
        helpers::ensure_intrinsic_declared(
            &mut ctx,
            vote_entry,
            "llvm_nvvm_vote_any_sync",
            vote_ty,
        )
        .unwrap();
        let vote_args = vote_entry.deref(&ctx).arguments().collect();
        llvm::CallOp::new(
            &mut ctx,
            CallOpCallable::Direct("llvm_nvvm_vote_any_sync".try_into().unwrap()),
            vote_ty,
            vote_args,
        )
        .get_operation()
        .insert_at_back(vote_entry, &ctx);
        llvm::ReturnOp::new(&mut ctx, None)
            .get_operation()
            .insert_at_back(vote_entry, &ctx);

        legalize_for_legacy_nvvm(&mut ctx, module.get_operation()).unwrap();

        let mut calls = Vec::new();
        let mut ops = Vec::new();
        collect_ops(&ctx, module.get_operation(), &mut ops);
        for op in ops {
            if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
                && let CallOpCallable::Direct(callee) = call.callee(&ctx)
            {
                calls.push(callee.to_string());
            }
        }
        assert!(calls.iter().any(|n| n == "llvm_nvvm_shfl_sync_i32"));
        assert!(calls.iter().any(|n| n == "llvm_nvvm_vote_sync"));
        assert!(!calls.iter().any(|n| shuffle_mode(n).is_some()));
        assert!(!calls.iter().any(|n| vote_mode(n).is_some()));
        let ids = operation_ids(&ctx, module.get_operation());
        assert_eq!(
            ids.iter().filter(|id| *id == "llvm.extract_value").count(),
            2
        );
        module.get_operation().deref(&ctx).verify(&ctx).unwrap();
    }

    #[test]
    fn f16_function_signature_is_rejected_without_needing_an_fneg() {
        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "half".try_into().unwrap());
        let f16_ty: TypeHandle = FP16Type::get(&ctx).into();
        let (_func, entry) = function(&mut ctx, &module, "identity_half", f16_ty, vec![f16_ty]);
        let input = entry.deref(&ctx).get_argument(0);
        llvm::ReturnOp::new(&mut ctx, Some(input))
            .get_operation()
            .insert_at_back(entry, &ctx);

        let error = legalize_for_legacy_nvvm(&mut ctx, module.get_operation()).unwrap_err();
        let text = error.disp(&ctx).to_string();
        assert!(text.contains("f16"), "{text}");
        assert!(text.contains("CUDA 12"), "{text}");
    }

    #[test]
    fn opaque_memory_intrinsic_overloads_gain_legacy_i8_pointees() {
        assert_eq!(
            legacy_memory_intrinsic_name("llvm_memcpy_p1_p3_i64").as_deref(),
            Some("llvm_memcpy_p1i8_p3i8_i64")
        );
        assert_eq!(
            legacy_memory_intrinsic_name("llvm_memmove_p0_p1_i32").as_deref(),
            Some("llvm_memmove_p0i8_p1i8_i32")
        );
        assert_eq!(
            legacy_memory_intrinsic_name("llvm_memset_p3_i64").as_deref(),
            Some("llvm_memset_p3i8_i64")
        );
        assert_eq!(
            legacy_memory_intrinsic_name("llvm_memcpy_p0i8_p0i8_i64"),
            None,
            "an already-legacy overload must not be rewritten again"
        );
    }

    #[test]
    fn unsupported_atomic_load_fails_before_mutating_other_ops() {
        use llvm_export::attributes::AtomicOrderingAttr;

        let mut ctx = Context::new();
        let module = ModuleOp::new(&mut ctx, "atomic".try_into().unwrap());
        let ptr_ty: TypeHandle = PointerType::get(&ctx, 1).into();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let void_ty: TypeHandle = VoidType::get(&ctx).into();
        let (_func, entry) = function(&mut ctx, &module, "atomic_load", void_ty, vec![ptr_ty]);
        let ptr = entry.deref(&ctx).get_argument(0);
        let load =
            llvm::AtomicLoadOp::new(&mut ctx, ptr, i32_ty, AtomicOrderingAttr::Monotonic, None);
        load.get_operation().insert_at_back(entry, &ctx);
        llvm::ReturnOp::new(&mut ctx, None)
            .get_operation()
            .insert_at_back(entry, &ctx);

        let error = legalize_for_legacy_nvvm(&mut ctx, module.get_operation()).unwrap_err();
        assert!(error.disp(&ctx).to_string().contains("atomic loads"));
        assert!(Operation::get_op::<llvm::AtomicLoadOp>(load.get_operation(), &ctx).is_some());
    }
}
