/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! rustc-independent cuda-oxide PTX backend.
//!
//! Takes a `dialect-mir` module assembled by any frontend and produces PTX
//! via verify, lowering to the LLVM dialect, `.ll` rendering, and `opt`/`llc`.
//! This crate has no rustc linkage: consumers need neither `rustc_private`
//! nor a matching nightly's `rustc_driver`.

use pliron::context::Context;

pub mod error;
pub mod export;
pub mod llvm_tools;
pub mod lower;
pub mod options;
pub mod ptx;
pub mod target;
pub mod verify;

pub use error::{PipelineError, PtxError};
pub use export::{DeviceExternAttrs, DeviceExternDecl};
pub use llvm_export::export::DeviceExternType;
pub use options::PtxConfig;
pub use ptx::compile_to_ptx;

/// Registers the dialects a frontend needs to build modules for this backend.
///
/// Forwards to `dialect_mir::register` and `dialect_nvvm::register`; both are
/// idempotent. The pliron builtin dialect self-registers on `Context::new`.
pub fn register_backend_dialects(ctx: &mut Context) {
    dialect_mir::register(ctx);
    dialect_nvvm::register(ctx);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_backend_dialects_is_idempotent() {
        let mut ctx = Context::new();
        register_backend_dialects(&mut ctx);
        register_backend_dialects(&mut ctx);
    }
}
