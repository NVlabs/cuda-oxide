/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rust MIR to `dialect-mir` translator.
//!
//! Converts Rust's MIR (from rustc) into [`dialect-mir`][dialect_mir] ops.
//! This is the core of cuda-oxide's ability to compile Rust to GPU code.
//!
//! # Module Structure
//!
//! | Module         | Purpose                                           |
//! |----------------|---------------------------------------------------|
//! | [`body`]       | Function-level translation, alloca setup          |
//! | [`block`]      | Basic block translation coordinator               |
//! | [`statement`]  | Statement translation (assignments, storage)      |
//! | [`terminator`] | Terminator translation (goto, call, return)       |
//! | [`rvalue`]     | Expression translation (binops, casts, etc.)      |
//! | [`types`]      | Rust type → `dialect-mir` type conversion         |
//! | [`values`]     | MIR local → alloca slot mapping                   |
//!
//! # Translation Flow
//!
//! ```text
//! translate_function()
//!   └─▶ body::translate_body()
//!         ├─▶ emit_entry_allocas()        // one alloca per non-ZST local
//!         └─▶ For each reachable block:
//!               └─▶ block::translate_block()
//!                     ├─▶ statement::translate_statement()
//!                     │     └─▶ rvalue::translate_rvalue()
//!                     └─▶ terminator::translate_terminator()
//! ```
//!
//! # Alloca + load/store model
//!
//! Every non-ZST MIR local is backed by a single `mir.alloca` emitted at the
//! top of the function's entry block. Defs lower to `mir.store`, uses lower
//! to `mir.load`. Cross-block data flow happens via these slots — no block
//! arguments other than the entry block's function parameters.
//!
//! The `mem2reg` pass in [`crate::pipeline`] promotes the scalar slots back
//! into SSA before the `dialect-mir` → LLVM dialect lowering runs.

pub mod block;
pub mod body;
pub(crate) mod layout;
pub(crate) mod location;
pub mod rvalue;
pub mod statement;
pub mod terminator;
pub mod types;
pub mod values;

use crate::error::{TranslationErr, TranslationResult};
use llvm_export::export::DebugKind;
use pliron::context::{Context, Ptr};
use pliron::identifier::Legaliser;
use pliron::input_error_noloc;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;
use rustc_public::mir::mono;

/// Public `SharedArray` methods whose compiler expansion returns a generic
/// pointer to the underlying shared allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SharedArrayPointerMethod {
    /// `SharedArray::as_ptr(&self)`.
    BorrowedConst,
    /// `SharedArray::as_mut_ptr(&mut self)`.
    BorrowedMut,
    /// `SharedArray::as_raw_mut_ptr(*mut Self)`.
    RawMut,
}

/// Recognize exactly the three public `SharedArray` pointer conversions.
///
/// Intrinsic dispatch and destination address-space classification both use
/// this helper so adding a method cannot update one compiler path without the
/// other.
pub(crate) fn shared_array_pointer_method(path: &str) -> Option<SharedArrayPointerMethod> {
    if !path.starts_with("cuda_device::")
        || !path.split("::").any(|component| component == "SharedArray")
    {
        return None;
    }

    match path.rsplit("::").next() {
        Some("as_ptr") => Some(SharedArrayPointerMethod::BorrowedConst),
        Some("as_mut_ptr") => Some(SharedArrayPointerMethod::BorrowedMut),
        Some("as_raw_mut_ptr") => Some(SharedArrayPointerMethod::RawMut),
        _ => None,
    }
}

/// Registers all dialects needed for translation.
///
/// Registers `dialect-mir` (our MIR modelling dialect), `dialect-nvvm`
/// (GPU intrinsics), and the `builtin` dialect (`ModuleOp`, `FunctionType`).
/// Note: Each dialect's `register()` function uses `entry().or_insert()`,
/// so it's safe to call even if already registered.
pub fn register_dialects(ctx: &mut Context) {
    dialect_mir::register(ctx);

    // dialect-nvvm is required for thread / block / warp intrinsics.
    dialect_nvvm::register(ctx);

    // The builtin dialect (ModuleOp etc.) is auto-registered by pliron 0.14.
}

/// Translates a Rust MIR function to a pliron module in `dialect-mir`.
///
/// Creates a `builtin.module` containing a single `mir.func` with the
/// translated function body. Registers required dialects automatically.
///
/// # Returns
///
/// The `builtin.module` operation pointer containing the translated function.
pub fn translate_function(
    ctx: &mut Context,
    body: &mir::Body,
    instance: &mono::Instance,
    is_kernel: bool,
    legaliser: &mut Legaliser,
) -> TranslationResult<Ptr<Operation>> {
    register_dialects(ctx);

    // Translate the function body. This helper is for tests/utilities that
    // don't have access to rustc's CodegenFnAttrs, so `is_inline_always` is
    // always false here. The real pipeline call (in `pipeline.rs`) threads
    // the flag through from `rustc-codegen-cuda`.
    // This utility does not participate in rustc-codegen-cuda's collector,
    // so it deliberately translates every block. The production pipeline
    // passes rustc's exact per-instance reachability instead.
    let all_successors: Vec<Vec<usize>> = body
        .blocks
        .iter()
        .map(|block| block.terminator.successors())
        .collect();
    let func_op = body::translate_body(
        ctx,
        body,
        instance,
        body.blocks.len(),
        &all_successors,
        is_kernel,
        /* is_inline_always */ false,
        None,
        legaliser,
        DebugKind::Off,
        None,
    )?;

    // Create a builtin.module operation using ModuleOp::new
    let module_name = instance.name();
    let module_name_ident: pliron::identifier::Identifier =
        module_name.clone().try_into().map_err(|_| {
            input_error_noloc!(TranslationErr::unsupported(format!(
                "Invalid module name: {}",
                module_name
            )))
        })?;

    let module = pliron::builtin::ops::ModuleOp::new(ctx, module_name_ident);

    // Append the function operation to the module's region 0
    // Get the module's operation and append to its region 0
    let module_op = module.get_operation();
    let module_region = module_op.deref(ctx).get_region(0);

    // Get or create the first block in the module region
    use pliron::basic_block::BasicBlock;
    let module_block = {
        let region_ref = module_region.deref(ctx);
        if let Some(first_block) = region_ref.iter(ctx).next() {
            first_block
        } else {
            drop(region_ref); // Release the immutable borrow
            let new_block = BasicBlock::new(ctx, None, vec![]);
            new_block.insert_at_front(module_region, ctx);
            new_block
        }
    };

    // Insert the function operation into the module block
    func_op.insert_at_front(module_block, ctx);

    Ok(module_op)
}

#[cfg(test)]
mod tests {
    use super::{SharedArrayPointerMethod, shared_array_pointer_method};

    #[test]
    fn shared_array_pointer_recognition_is_exact_and_centralized() {
        for (path, expected) in [
            (
                "cuda_device::shared::SharedArray::as_ptr",
                SharedArrayPointerMethod::BorrowedConst,
            ),
            (
                "cuda_device::shared::SharedArray::as_mut_ptr",
                SharedArrayPointerMethod::BorrowedMut,
            ),
            (
                "cuda_device::shared::SharedArray::as_raw_mut_ptr",
                SharedArrayPointerMethod::RawMut,
            ),
        ] {
            assert_eq!(shared_array_pointer_method(path), Some(expected), "{path}");
        }

        for near_match in [
            "cuda_device::shared::DynamicSharedArray::as_ptr",
            "cuda_device::shared::SharedArrayHelper::as_ptr",
            "cuda_device::shared::SharedArray::as_raw_mut_ptr_extra",
            "other_crate::SharedArray::as_raw_mut_ptr",
        ] {
            assert_eq!(
                shared_array_pointer_method(near_match),
                None,
                "{near_match}"
            );
        }
    }
}
