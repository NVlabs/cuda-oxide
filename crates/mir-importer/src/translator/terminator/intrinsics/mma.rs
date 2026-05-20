/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level mma.sync intrinsics (SM80+).

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::MmaM16n8k16Bf16F32Op;
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::FP32Type;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::TypeObj;
use pliron::value::Value;
use rustc_public::mir;

fn destination_struct_type(
    ctx: &mut Context,
    body: &mir::Body,
    destination: &mir::Place,
    loc: Location,
) -> TranslationResult<Ptr<TypeObj>> {
    let dest_rust_ty = match destination.ty(body.locals()) {
        Ok(t) => t,
        Err(e) => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "failed to resolve destination type for intrinsic result: {e:?}"
                ))
            );
        }
    };
    types::translate_type(ctx, &dest_rust_ty)
}

/// Emit `mma_m16n8k16_bf16_f32(a0,a1,a2,a3,b0,b1,c0,c1,c2,c3) -> CuSimd<f32, 4>`.
///
/// Maps to PTX `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` via the
/// `nvvm.mma_m16n8k16_bf16_f32` op (10 operands, 4 results).
pub fn emit_mma_m16n8k16_bf16_f32(
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
    if args.len() != 10 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "mma_m16n8k16_bf16_f32 expects 10 arguments (4xA, 2xB, 4xC), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(10);
    for arg in args.iter().take(10) {
        let (val, last_op_after) = rvalue::translate_operand(
            ctx,
            body,
            arg,
            value_map,
            block_ptr,
            last_op,
            loc.clone(),
        )?;
        last_op = last_op_after;
        operands.push(val);
    }

    // Create the MMA op with 4 f32 results.
    let f32_ty = FP32Type::get(ctx);
    let result_types = (0..4).map(|_| f32_ty.into()).collect();
    let mma_op = Operation::new(
        ctx,
        MmaM16n8k16Bf16F32Op::get_concrete_op_info(),
        result_types,
        operands,
        vec![],
        0,
    );
    mma_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        mma_op.insert_after(ctx, prev);
    } else {
        mma_op.insert_at_front(block_ptr, ctx);
    }

    // Bundle the 4 f32 results into a CuSimd<f32, 4> struct (matching destination layout).
    let results: Vec<Value> = (0..4).map(|i| mma_op.deref(ctx).get_result(i)).collect();

    let array_ty = dialect_mir::types::MirArrayType::get(ctx, f32_ty.into(), 4);
    let array_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructArrayOp::get_concrete_op_info(),
        vec![array_ty.into()],
        results,
        vec![],
        0,
    );
    array_op.deref_mut(ctx).set_loc(loc.clone());
    array_op.insert_after(ctx, mma_op);

    let struct_ty = destination_struct_type(ctx, body, destination, loc.clone())?;
    let array_result = array_op.deref(ctx).get_result(0);
    let struct_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructStructOp::get_concrete_op_info(),
        vec![struct_ty],
        vec![array_result],
        vec![],
        0,
    );
    struct_op.deref_mut(ctx).set_loc(loc.clone());
    struct_op.insert_after(ctx, array_op);

    let struct_result = struct_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        struct_result,
        target,
        block_ptr,
        struct_op,
        value_map,
        block_map,
        loc,
        "mma_m16n8k16_bf16_f32 call without target block",
    )
}
