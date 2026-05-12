/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `core::intrinsics::raw_eq` placeholder bridge.
//!
//! `raw_eq::<T>(a: &T, b: &T) -> bool` is a compiler intrinsic with no MIR
//! body. It surfaces from any `[T; N] == [T; N]` (and other `T: BytewiseEq`
//! comparisons) via the memcmp-style fast path in
//! `core/src/array/equality.rs`. The collector skips bodyless intrinsics, so
//! without this hop the call site survives into LLVM as a `call.uni` to a
//! symbol nothing defines. See the lowerer side at
//! `crates/mir-lower/src/convert/ops/call.rs::convert_rust_raw_eq`.
//!
//! Surfaced from `examples/array_eq_raw/`.
//!
//! Note: kept in its own module rather than appended to `ptr_arith.rs`
//! because `raw_eq` is a memory comparison, not pointer arithmetic — the
//! shared mechanism (recovering pointee size from the operand's
//! most-recent `MirPtrType`) is incidental.

use super::super::helpers;
use crate::error::TranslationResult;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_mir::rust_intrinsics;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::location::Location;
use pliron::operation::Operation;
use rustc_public::mir;

/// Recognize the libcore `raw_eq` intrinsic path that survived into MIR.
pub fn is_raw_eq(name: &str) -> bool {
    matches!(
        name,
        "core::intrinsics::raw_eq" | "std::intrinsics::raw_eq"
    )
}

/// Emit a placeholder `mir.call` for `core::intrinsics::raw_eq`.
#[allow(clippy::too_many_arguments)]
pub fn emit_rust_raw_eq(
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
    let return_type = types::translate_type(ctx, &body.locals()[destination.local].ty)?;
    helpers::emit_function_call(
        ctx,
        body,
        rust_intrinsics::CALLEE_RAW_EQ,
        args,
        destination,
        return_type,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
    )
}
