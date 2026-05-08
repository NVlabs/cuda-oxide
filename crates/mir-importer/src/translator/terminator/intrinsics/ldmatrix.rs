/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-cooperative ldmatrix.sync intrinsics (SM75+).

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{LdmatrixX4B16Op, LdmatrixX4TransB16Op};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
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

/// Variant tag controlling which dialect-nvvm op is emitted.
#[derive(Copy, Clone)]
enum LdmatrixVariant {
    Plain,
    Trans,
}

fn emit_ldmatrix_x4(
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
    variant: LdmatrixVariant,
) -> TranslationResult<Ptr<Operation>> {
    let intrinsic_name = match variant {
        LdmatrixVariant::Plain => "ldmatrix_x4_b16",
        LdmatrixVariant::Trans => "ldmatrix_x4_trans_b16",
    };
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{intrinsic_name} expects 1 argument (smem_ptr), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let (smem_ptr, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // 4 x u32 results (matches CuSimd<u32, 4> destination layout).
    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
    let result_types: Vec<Ptr<TypeObj>> = (0..4).map(|_| u32_ty.into()).collect();
    let op_info = match variant {
        LdmatrixVariant::Plain => LdmatrixX4B16Op::get_concrete_op_info(),
        LdmatrixVariant::Trans => LdmatrixX4TransB16Op::get_concrete_op_info(),
    };
    let ld_op = Operation::new(ctx, op_info, result_types, vec![smem_ptr], vec![], 0);
    ld_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        ld_op.insert_after(ctx, prev);
    } else {
        ld_op.insert_at_front(block_ptr, ctx);
    }

    // Bundle the 4 i32 results into CuSimd<u32, 4> matching the destination layout.
    let results: Vec<Value> = (0..4).map(|i| ld_op.deref(ctx).get_result(i)).collect();

    let array_ty = dialect_mir::types::MirArrayType::get(ctx, u32_ty.into(), 4);
    let array_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructArrayOp::get_concrete_op_info(),
        vec![array_ty.into()],
        results,
        vec![],
        0,
    );
    array_op.deref_mut(ctx).set_loc(loc.clone());
    array_op.insert_after(ctx, ld_op);

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
        "ldmatrix call without target block",
    )
}

/// Emit `ldmatrix_x4_b16(smem_ptr) -> CuSimd<u32, 4>`.
pub fn emit_ldmatrix_x4_b16(
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
    emit_ldmatrix_x4(
        ctx,
        body,
        args,
        destination,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        LdmatrixVariant::Plain,
    )
}

/// Emit `ldmatrix_x4_trans_b16(smem_ptr) -> CuSimd<u32, 4>`.
pub fn emit_ldmatrix_x4_trans_b16(
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
    emit_ldmatrix_x4(
        ctx,
        body,
        args,
        destination,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        LdmatrixVariant::Trans,
    )
}
