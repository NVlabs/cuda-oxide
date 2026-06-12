/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level intrinsic conversion: shuffle and vote operations.
//!
//! # Shuffle Operations
//!
//! | Operation          | LLVM Intrinsic                 | Description       |
//! |--------------------|--------------------------------|-------------------|
//! | `ShflSyncIdxI32`   | `llvm.nvvm.shfl.sync.idx.i32`  | Indexed shuffle   |
//! | `ShflSyncBflyI32`  | `llvm.nvvm.shfl.sync.bfly.i32` | Butterfly shuffle |
//! | `ShflSyncDownI32`  | `llvm.nvvm.shfl.sync.down.i32` | Down shuffle      |
//! | `ShflSyncUpI32`    | `llvm.nvvm.shfl.sync.up.i32`   | Up shuffle        |
//!
//! # Vote Operations
//!
//! | Operation        | LLVM Intrinsic               | Description           |
//! |------------------|------------------------------|-----------------------|
//! | `VoteSyncAll`    | `llvm.nvvm.vote.all.sync`    | All lanes true        |
//! | `VoteSyncAny`    | `llvm.nvvm.vote.any.sync`    | Any lane true         |
//! | `VoteSyncBallot` | `llvm.nvvm.vote.ballot.sync` | Bitmask of predicates |
//!
//! # Match Operations (sm_70+)
//!
//! | Operation         | LLVM Intrinsic                    | Description                  |
//! |-------------------|-----------------------------------|------------------------------|
//! | `MatchAnySyncI32` | `llvm.nvvm.match.any.sync.i32`    | Mask of equal-value lanes    |
//! | `MatchAnySyncI64` | `llvm.nvvm.match.any.sync.i64`    | 64-bit variant               |
//! | `MatchAllSyncI32` | `llvm.nvvm.match.all.sync.i32p`   | Full mask iff all agree      |
//! | `MatchAllSyncI64` | `llvm.nvvm.match.all.sync.i64p`   | 64-bit variant               |

use crate::convert::intrinsics::common::*;
use llvm_export::types as llvm_types;
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Convert i32 shuffle operation to LLVM intrinsic call.
///
/// Operand layout: `[mask, value, lane_or_delta]`. The mask reaches us
/// already type-converted by the framework (any `u32`/`i32` carrier
/// works); we forward it straight to the intrinsic. For full-warp ops
/// the mask is just `0xFFFFFFFF` baked in by the caller.
pub(crate) fn convert_shuffle_i32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
    clamp: i32,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 3 {
        return pliron::input_err_noloc!(
            "Warp shuffle i32 requires 3 operands [mask, value, lane_or_delta]"
        );
    }
    let (mask, val, lane_or_delta) = (operands[0], operands[1], operands[2]);

    let clamp_val = create_i32_const(ctx, rewriter, clamp);

    let func_ty = llvm_types::FuncType::get(
        ctx,
        i32_ty.into(),
        vec![i32_ty.into(), i32_ty.into(), i32_ty.into(), i32_ty.into()],
        false,
    );

    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![mask, val, lane_or_delta, clamp_val],
    )?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// Convert f32 shuffle operation to LLVM intrinsic call.
///
/// Operand layout: `[mask, value, lane_or_delta]`. See `convert_shuffle_i32`
/// for the mask forwarding rationale.
pub(crate) fn convert_shuffle_f32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
    clamp: i32,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let f32_ty = FP32Type::get(ctx);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 3 {
        return pliron::input_err_noloc!(
            "Warp shuffle f32 requires 3 operands [mask, value, lane_or_delta]"
        );
    }
    let (mask, val, lane_or_delta) = (operands[0], operands[1], operands[2]);

    let clamp_val = create_i32_const(ctx, rewriter, clamp);

    let func_ty = llvm_types::FuncType::get(
        ctx,
        f32_ty.into(),
        vec![i32_ty.into(), f32_ty.into(), i32_ty.into(), i32_ty.into()],
        false,
    );

    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![mask, val, lane_or_delta, clamp_val],
    )?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// Convert vote operation to LLVM intrinsic call.
///
/// Operand layout: `[mask, predicate]`. See `convert_shuffle_i32` for
/// the mask forwarding rationale.
pub(crate) fn convert_vote(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("Warp vote requires 2 operands [mask, predicate]");
    }
    let (mask, predicate) = (operands[0], operands[1]);

    let result_ty: Ptr<pliron::r#type::TypeObj> = if intrinsic_name.contains("ballot") {
        i32_ty.into()
    } else {
        i1_ty.into()
    };

    let func_ty =
        llvm_types::FuncType::get(ctx, result_ty, vec![i32_ty.into(), i1_ty.into()], false);
    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![mask, predicate],
    )?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// Convert a `match.any.sync` op to its LLVM intrinsic call.
///
/// Operand layout: `[mask, value]`. Result is i32 (bitmask of equal-value lanes).
/// The `value_ty` is i32 or i64 to pick `@llvm.nvvm.match.any.sync.{i32,i64}`.
pub(crate) fn convert_match_any(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
    value_ty: Ptr<pliron::r#type::TypeObj>,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("match.any.sync requires 2 operands [mask, value]");
    }
    let (mask, value) = (operands[0], operands[1]);

    let func_ty =
        llvm_types::FuncType::get(ctx, i32_ty.into(), vec![i32_ty.into(), value_ty], false);

    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![mask, value],
    )?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// Convert an `activemask` op to its LLVM intrinsic call.
///
/// Lowers to `call i32 @llvm.nvvm.activemask()`. The op has no operands.
pub(crate) fn convert_active_mask(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let func_ty = llvm_types::FuncType::get(ctx, i32_ty.into(), vec![], false);

    let call_op = call_intrinsic(ctx, rewriter, op, "llvm_nvvm_activemask", func_ty, vec![])?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// Convert a `bar.warp.sync` op to its LLVM intrinsic call.
///
/// Lowers to `call void @llvm.nvvm.bar.warp.sync(i32 mask)`. The op has one
/// operand (the participation mask) and no result.
pub(crate) fn convert_bar_warp_sync(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let void_ty = llvm_types::VoidType::get(ctx);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!("bar.warp.sync requires 1 operand [mask]");
    }
    let mask = operands[0];

    let func_ty = llvm_types::FuncType::get(ctx, void_ty.into(), vec![i32_ty.into()], false);
    call_intrinsic(
        ctx,
        rewriter,
        op,
        "llvm_nvvm_bar_warp_sync",
        func_ty,
        vec![mask],
    )?;
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert a `match.all.sync` op to its LLVM intrinsic call.
///
/// The LLVM intrinsic signature is `{i32, i1} @llvm.nvvm.match.all.sync.*p(i32 mask, T value)`:
/// field 0 is the matching mask, field 1 is the all-match predicate. We expose
/// only the mask (callers can recover the predicate as `result != 0`); the
/// extracted i1 is dead and gets removed by LLVM DCE.
pub(crate) fn convert_match_all(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    intrinsic_name: &str,
    value_ty: Ptr<pliron::r#type::TypeObj>,
) -> Result<()> {
    use llvm_export::ops::ExtractValueOp;

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("match.all.sync requires 2 operands [mask, value]");
    }
    let (mask, value) = (operands[0], operands[1]);

    let struct_ty = llvm_types::StructType::get_unnamed(ctx, vec![i32_ty.into(), i1_ty.into()]);
    let func_ty =
        llvm_types::FuncType::get(ctx, struct_ty.into(), vec![i32_ty.into(), value_ty], false);

    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        intrinsic_name,
        func_ty,
        vec![mask, value],
    )?;
    let struct_result = call_op.deref(ctx).get_result(0);

    let extract_op = ExtractValueOp::new(ctx, struct_result, vec![0])
        .map_err(|e| pliron::input_error_noloc!("match.all.sync extractvalue: {}", e))?;
    rewriter.insert_operation(ctx, extract_op.get_operation());
    let mask_result = extract_op.get_operation().deref(ctx).get_result(0);

    rewriter.replace_operation_with_values(ctx, op, vec![mask_result]);
    Ok(())
}
