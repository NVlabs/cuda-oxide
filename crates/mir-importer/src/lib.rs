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
//! │  │   (alloca)   │   │    ──▶ annotated loop unroll                │   │
//! │  │              │   │    ──▶ LLVM dialect  (via mir-lower)        │   │
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
//! | [`pipeline`]   | `mem2reg`, unroll, lower, export LLVM IR, run llc           |
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
