/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// MIR translation functions often have many parameters to pass context
#![allow(clippy::too_many_arguments)]
// Complex types are unavoidable when working with rustc internals
#![allow(clippy::type_complexity)]

//! Rust MIR to `dialect-mir` translator and compilation pipeline for cuda-oxide.
//!
//! This crate translates Rust's Mid-level Intermediate Representation (MIR)
//! into [`dialect-mir`][dialect_mir] — a pliron dialect (MLIR-like) that
//! preserves Rust semantics — then drives the rest of the compilation pipeline
//! down to PTX.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────── mir-importer ──────────────────────────────────┐
//! │                                                                       │
//! │  ┌──────────────┐   ┌─────────────────────────────────────────────┐   │
//! │  │  translator  │──▶│                  pipeline                   │   │
//! │  │              │   │                                             │   │
//! │  │     MIR      │   │  dialect-mir (alloca)                       │   │
//! │  │      ──▶     │   │    ──▶ mem2reg                              │   │
//! │  │  dialect-mir │   │    ──▶ dialect-mir (SSA)                    │   │
//! │  │   (alloca)   │   │    ──▶ LLVM dialect  (via mir-lower)        │   │
//! │  │              │   │    ──▶ LLVM IR ──▶ PTX  (via llc)           │   │
//! │  └──────────────┘   └─────────────────────────────────────────────┘   │
//! │                                                                       │
//! └───────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Modules
//!
//! | Module         | Purpose                                                     |
//! |----------------|-------------------------------------------------------------|
//! | [`translator`] | MIR → `dialect-mir` (alloca + load/store)                   |
//! | [`pipeline`]   | `mem2reg`, lower to LLVM dialect, export LLVM IR, run llc   |
//! | [`error`]      | Error types integrated with pliron's error system           |
//!
//! Note: Function collection is handled by `rustc-codegen-cuda/src/collector.rs`
//! which uses rustc internals for efficient traversal.
//!
//! # Example
//!
//! ```rust,ignore
//! use pliron::context::Context;
//! use rustc_public::mir::mono::Instance;
//!
//! // Inside rustc callback:
//! let body = instance.body().unwrap();
//! let mut ctx = Context::new();
//!
//! let module_op = mir_importer::translator::translate_function(
//!     &mut ctx, &body, &instance, /* is_kernel */ true
//! )?;
//! ```
//!
//! # Alloca + load/store model
//!
//! Every non-ZST MIR local is materialised as a single `mir.alloca` emitted
//! at the top of the function's entry block. Defs lower to `mir.store`, uses
//! lower to `mir.load`. Cross-block data flow happens through the slots, so
//! blocks (other than the entry) take no arguments. Pliron's `mem2reg` pass
//! promotes the slots back into SSA before the `dialect-mir` → LLVM dialect
//! lowering runs.

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_public;
extern crate rustc_public_bridge;
extern crate rustc_span;

pub mod error;
mod llvm_tools;
pub mod pipeline;
pub mod translator;

pub use error::{TranslationErr, TranslationResult};
pub use pipeline::{
    CollectedFunction, CompilationArtifactKind, CompilationResult, DeviceExternAttrs,
    DeviceExternDecl, PipelineConfig, PipelineError, run_pipeline,
};

// ── Array-length const evaluation bridge ─────────────────────────────────────
// The type translator must turn `[T; N]` array lengths into concrete values,
// but stable_mir cannot *evaluate* an `Unevaluated` length const (e.g. a `const
// SUB_LEN: usize = 2` reference) at the type-translation layer. The codegen
// backend (which holds `TyCtxt`) installs a lifetime-erased eval callback here
// for the duration of one `rustc_internal::run` scope, while tcx is alive.
type ArrayLenEval = Box<dyn Fn(&rustc_public::ty::TyConst) -> Option<u64>>;
thread_local! {
    static ARRAY_LEN_EVAL: std::cell::RefCell<Option<ArrayLenEval>> =
        const { std::cell::RefCell::new(None) };
}
/// Install the array-length eval callback. SAFETY CONTRACT: `f` borrows the
/// compiler `TyCtxt`; the caller MUST `clear_array_len_eval()` before that tcx
/// is released (i.e. within the same `rustc_internal::run` scope).
pub fn install_array_len_eval(f: ArrayLenEval) {
    ARRAY_LEN_EVAL.with(|c| *c.borrow_mut() = Some(f));
}
/// Remove the array-length eval callback.
pub fn clear_array_len_eval() {
    ARRAY_LEN_EVAL.with(|c| *c.borrow_mut() = None);
}
/// Evaluate an array-length const via the installed callback (if any).
pub(crate) fn eval_array_len(tc: &rustc_public::ty::TyConst) -> Option<u64> {
    ARRAY_LEN_EVAL.with(|c| c.borrow().as_ref().and_then(|f| f(tc)))
}
