// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Packed bf16x2 ALU intrinsics.
//!
//! Currently exposes only `fma_bf16x2`, since `add.bf16x2` / `mul.bf16x2`
//! require `sm_90+`. See [`crate::translator::terminator::intrinsics::bf16x2`]
//! and the `cuda-device` `bf16x2` module for the host-side declaration.

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::FmaBf16x2Op;
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

/// Emit `fma_bf16x2`: packed bf16x2 fused multiply-add.
///
/// Args: `(a: u32, b: u32, c: u32)` — each carrying two packed bf16 lanes.
/// Returns: `u32` — packed bf16x2 of `a * b + c`.
pub fn emit_fma_bf16x2(
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
                "fma_bf16x2 expects 3 arguments (a: u32, b: u32, c: u32), got {}",
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

    let (c_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // The Rust-side signature is `u32`; match the destination local's
    // unsigned-ness to avoid the MirStoreOp verifier flagging a
    // signless-vs-unsigned mismatch.
    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let fma_op = Operation::new(
        ctx,
        FmaBf16x2Op::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![a_val, b_val, c_val],
        vec![],
        0,
    );
    fma_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        fma_op.insert_after(ctx, prev);
    } else {
        fma_op.insert_at_front(block_ptr, ctx);
    }

    let result = fma_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        fma_op,
        value_map,
        block_map,
        loc,
        "fma_bf16x2 call without target block",
    )
}
