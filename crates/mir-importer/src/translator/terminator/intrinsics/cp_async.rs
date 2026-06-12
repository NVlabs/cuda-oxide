/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ampere async copy (`cp.async`) intrinsics for SM 80+.
//!
//! Translates `cuda_device::async_copy::*` intrinsic calls into `dialect-nvvm`
//! cp.async operations.

use super::super::helpers::emit_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{CpAsyncCa16Op, CpAsyncCg16Op};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

/// Shared implementation for cp.async 16-byte copy variants.
///
/// Both `cg` (L2-only) and `ca` (L1+L2) variants take the same operands
/// and follow the same emit pattern, differing only in the dialect op type `T`.
fn emit_cp_async_16_impl<T: Op>(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    name: &str,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{name} expects 2 arguments, got {}",
                args.len()
            ))
        );
    }

    let (shared_dst, mut last_op) = rvalue::translate_operand(
        ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),
    )?;

    let (global_src, last_op_after) = rvalue::translate_operand(
        ctx, body, &args[1], value_map, block_ptr, last_op, loc.clone(),
    )?;
    last_op = last_op_after;

    let cp_op = Operation::new(
        ctx,
        T::get_concrete_op_info(),
        vec![],
        vec![shared_dst, global_src],
        vec![],
        0,
    );
    cp_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        cp_op.insert_after(ctx, prev);
    } else {
        cp_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, cp_op, block_map, loc))
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!("{name} call without target block"))
        )
    }
}

/// Emit cp_async_cg_16: async 16-byte copy from global to shared memory.
///
/// Uses `.cg` cache policy (L2-only, bypasses L1).
///
/// Args:
/// - `args[0]`: *mut u8 (shared memory destination, 16-byte aligned)
/// - `args[1]`: *const u8 (global memory source, 16-byte aligned)
pub fn emit_cp_async_cg_16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    emit_cp_async_16_impl::<CpAsyncCg16Op>(
        ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
        "cp_async_cg_16",
    )
}

/// Emit cp_async_ca_16: async 16-byte copy with `.ca` cache policy (L1+L2).
///
/// Args:
/// - `args[0]`: *mut u8 (shared memory destination, 16-byte aligned)
/// - `args[1]`: *const u8 (global memory source, 16-byte aligned)
pub fn emit_cp_async_ca_16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    emit_cp_async_16_impl::<CpAsyncCa16Op>(
        ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
        "cp_async_ca_16",
    )
}
