/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Asynchronous copy with zero-fill (`cp.async`) operations.
//!
//! These operations correspond to the PTX `cp.async.ca.shared.global` instruction
//! with the optional `src_size` parameter for zero-fill semantics. When
//! `src_size < cp_size`, the remaining bytes in the shared-memory destination
//! are zero-filled by hardware, which is useful for boundary tiles in tiled
//! algorithms.
//!
//! ```text
//! ┌──────────────────────┬─────────┬──────────────────────────────────────────────┐
//! │ Operation            │ cp_size │ PTX                                          │
//! ├──────────────────────┼─────────┼──────────────────────────────────────────────┤
//! │ CpAsyncCaZfill4Op    │ 4       │ cp.async.ca.shared.global [dst], [src], 4, s │
//! │ CpAsyncCaZfill8Op    │ 8       │ cp.async.ca.shared.global [dst], [src], 8, s │
//! │ CpAsyncCaZfill16Op   │ 16      │ cp.async.ca.shared.global [dst], [src],16, s │
//! └──────────────────────┴─────────┴──────────────────────────────────────────────┘
//! ```
//!
//! # Requirements
//!
//! - **Architecture**: sm_80+ (Ampere and later)
//! - **Memory**: `dst` must be in shared memory, `src` in global memory
//! - **Alignment**: `dst` must be aligned to `cp_size` bytes

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Async copy 4 bytes from global to shared with zero-fill.
///
/// PTX: `cp.async.ca.shared.global [dst], [src], 4, src_size;`
///
/// # Operands
///
/// - `dst` (ptr): destination pointer in shared memory
/// - `src` (ptr): source pointer in global memory
/// - `src_size` (i32): number of valid source bytes (0..=4)
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_ca_zfill_4",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct CpAsyncCaZfill4Op;

impl CpAsyncCaZfill4Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        CpAsyncCaZfill4Op { op }
    }
}

/// Async copy 8 bytes from global to shared with zero-fill.
///
/// PTX: `cp.async.ca.shared.global [dst], [src], 8, src_size;`
///
/// # Operands
///
/// - `dst` (ptr): destination pointer in shared memory
/// - `src` (ptr): source pointer in global memory
/// - `src_size` (i32): number of valid source bytes (0..=8)
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_ca_zfill_8",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct CpAsyncCaZfill8Op;

impl CpAsyncCaZfill8Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        CpAsyncCaZfill8Op { op }
    }
}

/// Async copy 16 bytes from global to shared with zero-fill.
///
/// PTX: `cp.async.ca.shared.global [dst], [src], 16, src_size;`
///
/// # Operands
///
/// - `dst` (ptr): destination pointer in shared memory
/// - `src` (ptr): source pointer in global memory
/// - `src_size` (i32): number of valid source bytes (0..=16)
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.cp_async_ca_zfill_16",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct CpAsyncCaZfill16Op;

impl CpAsyncCaZfill16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        CpAsyncCaZfill16Op { op }
    }
}

/// Register cp.async zero-fill operations with the context.
pub(super) fn register(ctx: &mut Context) {
    CpAsyncCaZfill4Op::register(ctx);
    CpAsyncCaZfill8Op::register(ctx);
    CpAsyncCaZfill16Op::register(ctx);
}
