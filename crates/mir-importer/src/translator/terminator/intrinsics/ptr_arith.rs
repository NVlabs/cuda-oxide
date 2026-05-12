/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rust compiler pointer-arithmetic intrinsics.
//!
//! `core::intrinsics::ptr_offset_from_unsigned` is what
//! `<*const T>::offset_from_unsigned(origin)` lowers to. The intrinsic has no
//! MIR body — rustc's own codegen backend lowers it directly to
//! ptrtoint/sub/udiv. The cuda-oxide collector skips bodyless intrinsics,
//! so without this placeholder hop the call site survives into LLVM IR as a
//! `call.uni` to a symbol nothing defines and `llc` rejects the module.
//!
//! Surfaced from `examples/iter_zip_chunks_exact/` (transitively, via
//! `core::slice::iter::ChunksExact`'s `len_remaining` bookkeeping). See the
//! lowerer side at `crates/mir-lower/src/convert/ops/call.rs`.

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

/// Pointer-arithmetic intrinsic from libcore that lowers to plain LLVM ops
/// (not an `llvm.*` intrinsic).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RustPtrArithIntrinsic {
    /// `core::intrinsics::ptr_offset_from_unsigned`. Returns `usize` — the
    /// number of `T`-sized elements between `self` and `origin` (callee
    /// guarantees `self >= origin`).
    OffsetFromUnsigned,
}

impl RustPtrArithIntrinsic {
    /// Recognize the libcore intrinsic path that survived into MIR.
    pub fn from_core_path(name: &str) -> Option<Self> {
        match name {
            "core::intrinsics::ptr_offset_from_unsigned"
            | "std::intrinsics::ptr_offset_from_unsigned" => Some(Self::OffsetFromUnsigned),
            _ => None,
        }
    }

    /// Return the internal placeholder name used until MIR-to-LLVM lowering.
    pub fn placeholder_callee(self) -> &'static str {
        match self {
            Self::OffsetFromUnsigned => rust_intrinsics::CALLEE_PTR_OFFSET_FROM_UNSIGNED,
        }
    }
}

/// Emit a placeholder `mir.call` for a rustc pointer-arithmetic intrinsic.
#[allow(clippy::too_many_arguments)]
pub fn emit_rust_ptr_arith_intrinsic(
    ctx: &mut Context,
    body: &mir::Body,
    intrinsic: RustPtrArithIntrinsic,
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
        intrinsic.placeholder_callee(),
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
