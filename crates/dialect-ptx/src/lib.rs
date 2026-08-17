/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Structured PTX operations, construction, emission, and lossless-source
//! projection.
//!
//! [`ptx_parse::Document`] remains authoritative for lossless text and edits.
//! This crate owns a canonical, independently constructible PTX operation tree;
//! [`Projection`] records optional lineage when that tree came from source.

pub mod attributes;
pub mod builder;
pub mod emitter;
pub mod ops;
mod projection;

pub use builder::{PtxBodyBuilder, PtxBuilder};
pub use emitter::{EmitError, emit_module, write_module};
pub use projection::{ProjectedBlock, ProjectedNode, Projection, SourceNode};

use pliron::context::Context;
use pliron::dialect::{Dialect, DialectName};

pub const PTX_DIALECT_NAME: &str = "ptx";

pub fn register(ctx: &mut Context) {
    Dialect::register(
        ctx,
        &DialectName::try_new(PTX_DIALECT_NAME).expect("valid dialect name"),
    );
    attributes::register(ctx);
    ops::register(ctx);
}
