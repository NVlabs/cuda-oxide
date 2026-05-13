/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Debug and profiling intrinsic conversion.
//!
//! | Operation     | Lowering                            | PTX Output         |
//! |---------------|-------------------------------------|--------------------|
//! | `Clock`       | `llvm_nvvm_read_ptx_sreg_clock`     | `mov %r, %clock`   |
//! | `Clock64`     | `llvm_nvvm_read_ptx_sreg_clock64`   | `mov %rd, %clock64`|
//! | `Trap`        | inline PTX `trap;`                  | `trap;`            |
//! | `Breakpoint`  | inline PTX `brkpt;`                 | `brkpt;`           |
//! | `PmEvent`     | inline PTX `pmevent N;`             | `pmevent N;`       |
//! | `Vprintf`     | `call @vprintf`                     | `call vprintf`     |
//! | `BlackBox`    | empty `asm sideeffect` barrier      | (no instructions)  |

use crate::convert::intrinsics::common::*;
use crate::helpers;
use dialect_llvm::op_interfaces::{BinArithOp, CastOpInterface};
use dialect_llvm::ops as llvm;
use dialect_llvm::types as llvm_types;
use pliron::builtin::op_interfaces::CallOpCallable;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::Typed;

pub(crate) fn convert_clock(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let func_ty = llvm_types::FuncType::get(ctx, i32_ty.into(), vec![], false);

    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        "llvm_nvvm_read_ptx_sreg_clock",
        func_ty,
        vec![],
    )?;
    rewriter.replace_operation(ctx, op, call_op);

    Ok(())
}

pub(crate) fn convert_clock64(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let func_ty = llvm_types::FuncType::get(ctx, i64_ty.into(), vec![], false);

    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        "llvm_nvvm_read_ptx_sreg_clock64",
        func_ty,
        vec![],
    )?;
    rewriter.replace_operation(ctx, op, call_op);

    Ok(())
}

pub(crate) fn convert_trap(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    inline_asm_convergent(ctx, rewriter, void_ty.into(), vec![], "trap;", "");
    rewriter.erase_operation(ctx, op);
    Ok(())
}

pub(crate) fn convert_breakpoint(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    inline_asm_convergent(ctx, rewriter, void_ty.into(), vec![], "brkpt;", "");
    rewriter.erase_operation(ctx, op);
    Ok(())
}

pub(crate) fn convert_pm_event(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    use dialect_nvvm::ops::PmEventOp;

    let pmevent_op = PmEventOp::new(op);
    let event_id = pmevent_op.get_event_id(ctx).unwrap_or(0);

    let void_ty = llvm_types::VoidType::get(ctx);

    let asm_str = format!("pmevent {};", event_id);
    inline_asm_convergent(ctx, rewriter, void_ty.into(), vec![], &asm_str, "");
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Lower `nvvm.black_box` to an empty inline `asm sideeffect` with
/// register input/output — the same shape rustc's LLVM backend emits
/// for `core::hint::black_box`. LLVM treats this as opaque, so the
/// optimizer can't see through it and const-fold the operand back into
/// a constant.
///
/// Constraint letter is picked from the integer bit-width using the
/// NVPTX inline-asm register classes:
/// * 8 / 16 / 32-bit → `r` (32-bit register, NVPTX promotes narrower)
/// * 64-bit → `l` (64-bit register)
///
/// i128 is split into hi/lo i64 halves, each run through its own
/// barrier, and recombined via shl/or. NVPTX has no native i128
/// register class for inline asm, but splitting matches what the
/// backend would do for ordinary i128 arithmetic anyway.
pub(crate) fn convert_black_box(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!(
            "nvvm.black_box requires 1 operand, got {}",
            operands.len()
        );
    }
    let input_val = operands[0];
    let value_ty = input_val.get_type(ctx);

    let int_width = {
        let ty_obj = value_ty.deref(ctx);
        match ty_obj.downcast_ref::<IntegerType>() {
            Some(int_ty) => int_ty.width(),
            None => {
                return pliron::input_err_noloc!(
                    "nvvm.black_box of non-integer type not yet supported"
                );
            }
        }
    };

    match int_width {
        1 | 8 | 16 | 32 | 64 => {
            let constraint_letter = if int_width == 64 { "l" } else { "r" };
            let constraints = format!("={c},{c}", c = constraint_letter);
            let asm_op =
                inline_asm_convergent(ctx, rewriter, value_ty, vec![input_val], "", &constraints);
            rewriter.replace_operation(ctx, op, asm_op);
            Ok(())
        }
        128 => convert_black_box_i128(ctx, rewriter, op, input_val),
        w => pliron::input_err_noloc!(
            "nvvm.black_box of i{w} not yet supported \
             (split into 32/64/128-bit halves or extend convert_black_box)"
        ),
    }
}

/// Split an i128 black_box into two i64 barriers and recombine.
///
/// Emits:
/// ```text
/// %lo      = trunc i128 %x to i64
/// %hi_raw  = lshr i128 %x, 64
/// %hi      = trunc i128 %hi_raw to i64
/// %lo_bb   = call i64 asm sideeffect "", "=l,l"(i64 %lo)
/// %hi_bb   = call i64 asm sideeffect "", "=l,l"(i64 %hi)
/// %lo_z    = zext i64 %lo_bb to i128
/// %hi_z    = zext i64 %hi_bb to i128
/// %hi_shl  = shl  i128 %hi_z, 64
/// %result  = or   i128 %hi_shl, %lo_z
/// ```
fn convert_black_box_i128(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    input_val: pliron::value::Value,
) -> Result<()> {
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let i128_ty = IntegerType::get(ctx, 128, Signedness::Signless);

    // %lo = trunc i128 %x to i64
    let lo_trunc = llvm::TruncOp::new(ctx, input_val, i64_ty.into()).get_operation();
    rewriter.insert_operation(ctx, lo_trunc);
    let lo_val = lo_trunc.deref(ctx).get_result(0);

    // %shift64 = i128 64
    let shift64_attr = IntegerAttr::new(
        i128_ty,
        APInt::from_u128(64, NonZeroUsize::new(128).unwrap()),
    );
    let shift64_const = llvm::ConstantOp::new(ctx, shift64_attr.into()).get_operation();
    rewriter.insert_operation(ctx, shift64_const);
    let shift64_val = shift64_const.deref(ctx).get_result(0);

    // %hi_raw = lshr i128 %x, 64
    let hi_lshr = llvm::LShrOp::new(ctx, input_val, shift64_val).get_operation();
    rewriter.insert_operation(ctx, hi_lshr);
    let hi_raw_val = hi_lshr.deref(ctx).get_result(0);

    // %hi = trunc i128 %hi_raw to i64
    let hi_trunc = llvm::TruncOp::new(ctx, hi_raw_val, i64_ty.into()).get_operation();
    rewriter.insert_operation(ctx, hi_trunc);
    let hi_val = hi_trunc.deref(ctx).get_result(0);

    // %lo_bb = call i64 asm sideeffect "", "=l,l"(i64 %lo)
    let lo_bb_op =
        inline_asm_convergent(ctx, rewriter, i64_ty.into(), vec![lo_val], "", "=l,l");
    let lo_bb_val = lo_bb_op.deref(ctx).get_result(0);

    // %hi_bb = call i64 asm sideeffect "", "=l,l"(i64 %hi)
    let hi_bb_op =
        inline_asm_convergent(ctx, rewriter, i64_ty.into(), vec![hi_val], "", "=l,l");
    let hi_bb_val = hi_bb_op.deref(ctx).get_result(0);

    let nneg_key: pliron::identifier::Identifier = "llvm_nneg_flag".try_into().unwrap();
    let nneg_false = || pliron::builtin::attributes::BoolAttr::new(false).into();

    // %lo_z = zext i64 %lo_bb to i128
    let lo_zext_struct = llvm::ZExtOp::new(ctx, lo_bb_val, i128_ty.into());
    lo_zext_struct
        .get_operation()
        .deref_mut(ctx)
        .attributes
        .0
        .insert(nneg_key.clone(), nneg_false());
    let lo_zext = lo_zext_struct.get_operation();
    rewriter.insert_operation(ctx, lo_zext);
    let lo_zext_val = lo_zext.deref(ctx).get_result(0);

    // %hi_z = zext i64 %hi_bb to i128
    let hi_zext_struct = llvm::ZExtOp::new(ctx, hi_bb_val, i128_ty.into());
    hi_zext_struct
        .get_operation()
        .deref_mut(ctx)
        .attributes
        .0
        .insert(nneg_key, nneg_false());
    let hi_zext = hi_zext_struct.get_operation();
    rewriter.insert_operation(ctx, hi_zext);
    let hi_zext_val = hi_zext.deref(ctx).get_result(0);

    // %hi_shl = shl i128 %hi_z, 64
    let hi_shl_struct = llvm::ShlOp::new(ctx, hi_zext_val, shift64_val);
    let iof_flags = dialect_llvm::attributes::IntegerOverflowFlagsAttr::default();
    hi_shl_struct
        .get_operation()
        .deref_mut(ctx)
        .attributes
        .set(
            dialect_llvm::op_interfaces::ATTR_KEY_INTEGER_OVERFLOW_FLAGS.clone(),
            iof_flags,
        );
    let hi_shl = hi_shl_struct.get_operation();
    rewriter.insert_operation(ctx, hi_shl);
    let hi_shl_val = hi_shl.deref(ctx).get_result(0);

    // %result = or i128 %hi_shl, %lo_z
    let or_op = llvm::OrOp::new(ctx, hi_shl_val, lo_zext_val).get_operation();
    rewriter.insert_operation(ctx, or_op);

    rewriter.replace_operation(ctx, op, or_op);
    Ok(())
}

pub(crate) fn convert_vprintf(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("vprintf requires 2 operands, got {}", operands.len());
    }

    let format_ptr = operands[0];
    let args_ptr = operands[1];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i8_ptr_ty = llvm_types::PointerType::get(ctx, 0);

    let func_ty = llvm_types::FuncType::get(
        ctx,
        i32_ty.into(),
        vec![i8_ptr_ty.into(), i8_ptr_ty.into()],
        false,
    );

    let parent_block = op.deref(ctx).get_parent_block().unwrap();
    helpers::ensure_intrinsic_declared(ctx, parent_block, "vprintf", func_ty)
        .map_err(|e| pliron::input_error_noloc!("{}", e))?;

    let sym_name: pliron::identifier::Identifier = "vprintf".try_into().unwrap();
    let callee = CallOpCallable::Direct(sym_name);
    let call_op = llvm::CallOp::new(ctx, callee, func_ty, vec![format_ptr, args_ptr]);
    rewriter.insert_operation(ctx, call_op.get_operation());
    rewriter.replace_operation(ctx, op, call_op.get_operation());

    Ok(())
}
