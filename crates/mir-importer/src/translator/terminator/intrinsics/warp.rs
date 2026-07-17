/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Manual translation helper for warp reduction.

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
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
