/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Handwritten warp-matrix intrinsics not yet owned by the generator.

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::MovmatrixTransB16Op;
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

/// Emit an in-register 8x8 matrix transpose.
#[allow(clippy::too_many_arguments)]
pub fn emit_movmatrix_trans_b16(
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
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "movmatrix_trans_b16 expects 1 argument, got {}",
                args.len()
            ))
        );
    }

    let (a_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
    let mov_op = Operation::new(
        ctx,
        MovmatrixTransB16Op::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![a_val],
        vec![],
        0,
    );
    mov_op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(prev) = last_op {
        mov_op.insert_after(ctx, prev);
    } else {
        mov_op.insert_at_front(block_ptr, ctx);
    }

    let result = mov_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        mov_op,
        value_map,
        block_map,
        loc,
        "movmatrix_trans_b16 call without target block",
    )
}
