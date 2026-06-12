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
pub mod rvalue;
pub mod statement;
pub mod terminator;
pub mod types;
pub mod values;

use crate::error::{TranslationErr, TranslationResult};
use pliron::context::{Context, Ptr};
use pliron::identifier::Legaliser;
use pliron::input_error_noloc;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;
use rustc_public::mir::mono;

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

    // Translate the function body
    let func_op = body::translate_body(ctx, body, instance, is_kernel, None, legaliser)?;

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
