/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Debug and profiling intrinsics.
//!
//! Handles translation of debug/profiling primitives including:
//! - `clock()` - 32-bit GPU clock counter
//! - `clock64()` - 64-bit GPU clock counter
//! - `globaltimer()` - 64-bit GPU global timer
//! - `__gpu_vprintf()` - Formatted output to host console

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::VprintfOp;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::operation::Operation;
use rustc_public::mir;

/// Emits `__gpu_vprintf()`: Formatted output to host console.
///
/// # Generated Operation
///
/// `nvvm.vprintf` - Maps to CUDA `vprintf(format, args)`
///
/// # Arguments
///
/// * `args[0]` - Pointer to null-terminated format string (*const u8)
/// * `args[1]` - Pointer to packed argument buffer (*const u8)
///
/// # Returns
///
/// i32 - Number of arguments on success (negative on error)
#[allow(clippy::too_many_arguments)]
pub fn emit_vprintf(
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
    use crate::translator::rvalue;

    // Validate we have exactly 2 arguments: format_ptr and args_ptr
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "__gpu_vprintf expects 2 arguments, got {}",
                args.len()
            ))
        );
    }

    // Translate the format pointer operand
    let (format_ptr, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Translate the args pointer operand
    let (args_ptr, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    // Create the vprintf operation
    let vprintf_op = VprintfOp::build(ctx, format_ptr, args_ptr);
    vprintf_op.deref_mut(ctx).set_loc(loc.clone());

    // Insert the operation
    if let Some(prev) = last_op {
        vprintf_op.insert_after(ctx, prev);
    } else {
        vprintf_op.insert_at_front(block_ptr, ctx);
    }

    // Store the result (i32) in the destination
    let result_value = vprintf_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        vprintf_op,
        value_map,
        block_map,
        loc,
        "__gpu_vprintf call without target block",
    )
}
