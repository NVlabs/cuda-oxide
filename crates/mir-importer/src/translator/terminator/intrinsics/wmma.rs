/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix operation intrinsics (movmatrix).
//!
//! Handles translation of `cuda_device::wmma::movmatrix_trans_b16` into
//! the `nvvm.movmatrix_trans_b16` dialect operation.

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

/// Emits `wmma::movmatrix_trans_b16(a)`: in-register 8×8 b16 matrix transpose.
///
/// # Generated Operation
///
/// `nvvm.movmatrix_trans_b16` — one i32 operand, one i32 result.
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
                "wmma::movmatrix_trans_b16 expects 1 argument [a], got {}",
                args.len()
            ))
        );
    }

    let u32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let (a_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let movmatrix_op = Operation::new(
        ctx,
        MovmatrixTransB16Op::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![a_val],
        vec![],
        0,
    );
    movmatrix_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        movmatrix_op.insert_after(ctx, prev);
    } else {
        movmatrix_op.insert_at_front(block_ptr, ctx);
    }

    let result_value = movmatrix_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        movmatrix_op,
        value_map,
        block_map,
        loc,
        "movmatrix_trans_b16 call without target block",
    )
}
