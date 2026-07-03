// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte permute intrinsic (`prmt.b32`).
//!
//! Translates `cuda_device::prmt::prmt` calls into the `dialect-nvvm` prmt
//! operation.

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::PrmtOp;
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

/// Emit `prmt`: byte permute from concatenation of `a` and `b`.
///
/// Args: `(a: u32, b: u32, control: u32)`. Returns: `u32`.
pub fn emit_prmt(
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
    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "prmt expects 3 arguments (a: u32, b: u32, control: u32), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    let (a_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (b_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let (control_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let prmt_op = Operation::new(
        ctx,
        PrmtOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![a_val, b_val, control_val],
        vec![],
        0,
    );
    prmt_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        prmt_op.insert_after(ctx, prev);
    } else {
        prmt_op.insert_at_front(block_ptr, ctx);
    }

    let result = prmt_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        prmt_op,
        value_map,
        block_map,
        loc,
        "prmt call without target block",
    )
}
