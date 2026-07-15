/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Manual translation helpers for warp reduction and election.

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::ElectSyncOp;
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

/// Emit a warp reduction operation (`redux.sync.{add,min,max,and,or,xor}`).
///
/// Takes 2 operands `[mask, value]` and returns one result. This helper is
/// shared by the whole integer reduction family.
///
/// # Parameters
/// - `redux_opid`: The NVVM opid for the specific reduction variant
/// - `signed`: result signedness — `true` for the signed `min.s32`/`max.s32`
///   variants (result type must match an `i32` destination slot), `false` for
///   `add`, the unsigned `min.u32`/`max.u32`, and the bitwise `and`/`or`/`xor`
///   variants (all `u32`).
/// - `args`: `[mask, value]`
pub fn emit_warp_redux(
    ctx: &mut Context,
    body: &mir::Body,
    redux_opid: (
        fn(pliron::context::Ptr<pliron::operation::Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
    signed: bool,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "warp redux expects 2 arguments [mask, value], got {}",
                args.len()
            ))
        );
    }

    // Result signedness must match the destination local's slot type so the
    // store typechecks: `i32` locals are `Signed`, `u32` locals `Unsigned`.
    let signedness = if signed {
        Signedness::Signed
    } else {
        Signedness::Unsigned
    };
    let result_ty = IntegerType::get(ctx, 32, signedness).to_handle();

    let (mask, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let (value, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let redux_op = Operation::new(
        ctx,
        redux_opid,
        vec![result_ty],
        vec![mask, value],
        vec![],
        0,
    );
    redux_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        redux_op.insert_after(ctx, prev);
    } else {
        redux_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = redux_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        redux_op,
        value_map,
        block_map,
        loc,
        "warp redux call without target block",
    )
}

/// Emit `elect.sync`: elect the lowest participating lane as leader (sm_90+).
///
/// The device fn returns `(u32, bool)` = `(leader_lane, is_elected)`. The LLVM
/// intrinsic produces both halves in one `{i32, i1}` struct, so we build a
/// 2-result `nvvm.elect_sync` op and pack its results into the destination
/// tuple here; the lowering does the struct field extraction. `args` is
/// `[mask]`.
pub fn emit_elect_sync(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    use dialect_mir::ops::MirConstructTupleOp;
    use dialect_mir::types::MirTupleType;

    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "warp::elect_sync expects 1 argument [mask], got {}",
                args.len()
            ))
        );
    }

    // The destination local is the `(u32, bool)` tuple. Derive the leader and
    // predicate element types from it so the op results and the packed tuple
    // typecheck against the slot exactly.
    let tuple_ty = types::translate_type(ctx, &body.locals()[destination.local].ty)?;
    let (leader_ty, elected_ty) = {
        let t = tuple_ty.deref(ctx);
        match t.downcast_ref::<MirTupleType>() {
            Some(tup) if tup.get_types().len() == 2 => (tup.get_types()[0], tup.get_types()[1]),
            _ => {
                return input_err!(
                    loc.clone(),
                    TranslationErr::unsupported(
                        "warp::elect_sync destination must be a (u32, bool) tuple".to_string()
                    )
                );
            }
        }
    };

    let (mask, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // One op, two results: [leader (i32), is_elected (i1)].
    let elect_op = Operation::new(
        ctx,
        ElectSyncOp::get_concrete_op_info(),
        vec![leader_ty, elected_ty],
        vec![mask],
        vec![],
        0,
    );
    elect_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        elect_op.insert_after(ctx, prev);
    } else {
        elect_op.insert_at_front(block_ptr, ctx);
    }

    let leader_val = elect_op.deref(ctx).get_result(0);
    let elected_val = elect_op.deref(ctx).get_result(1);

    // Pack (leader, is_elected) into the destination tuple.
    let tuple_op = Operation::new(
        ctx,
        MirConstructTupleOp::get_concrete_op_info(),
        vec![tuple_ty],
        vec![leader_val, elected_val],
        vec![],
        0,
    );
    tuple_op.deref_mut(ctx).set_loc(loc.clone());
    tuple_op.insert_after(ctx, elect_op);
    let tuple_val = tuple_op.deref(ctx).get_result(0);

    emit_store_result_and_goto(
        ctx,
        destination,
        tuple_val,
        target,
        block_ptr,
        tuple_op,
        value_map,
        block_map,
        loc,
        "warp::elect_sync call without target block",
    )
}
