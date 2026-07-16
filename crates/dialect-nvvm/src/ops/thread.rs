/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thread, block, and grid indexing operations.
//!
//! This module provides operations for reading GPU thread hierarchy registers:
//!
//! ```text
//! ┌──────────────────────┬──────────────┬────────────────────────────┐
//! │ Operation            │ PTX Register │ Description                │
//! ├──────────────────────┼──────────────┼────────────────────────────┤
//! │ ReadPtxSregTidXOp    │ %tid.x       │ Thread ID within block (X) │
//! │ ReadPtxSregTidYOp    │ %tid.y       │ Thread ID within block (Y) │
//! │ ReadPtxSregTidZOp    │ %tid.z       │ Thread ID within block (Z) │
//! │ ReadPtxSregCtaidXOp  │ %ctaid.x     │ Block ID within grid (X)   │
//! │ ReadPtxSregCtaidYOp  │ %ctaid.y     │ Block ID within grid (Y)   │
//! │ ReadPtxSregCtaidZOp  │ %ctaid.z     │ Block ID within grid (Z)   │
//! │ ReadPtxSregNtidXOp   │ %ntid.x      │ Block dimension (X)        │
//! │ ReadPtxSregNtidYOp   │ %ntid.y      │ Block dimension (Y)        │
//! │ ReadPtxSregNtidZOp   │ %ntid.z      │ Block dimension (Z)        │
//! │ ReadPtxSregNctaidXOp │ %nctaid.x    │ Grid dimension (X)         │
//! │ ReadPtxSregNctaidYOp │ %nctaid.y    │ Grid dimension (Y)         │
//! │ ReadPtxSregNctaidZOp │ %nctaid.z    │ Grid dimension (Z)         │
//! │ ReadPtxSregEnvReg1Op │ %envreg1     │ Driver ABI envreg 1        │
//! │ ReadPtxSregEnvReg2Op │ %envreg2     │ Driver ABI envreg 2        │
//! │ ThreadfenceBlockOp   │ membar.cta   │ Block-scoped memory fence  │
//! │ ThreadfenceOp        │ membar.gl    │ Device-scoped memory fence │
//! │ ThreadfenceSystemOp  │ membar.sys   │ System-scoped memory fence │
//! └──────────────────────┴──────────────┴────────────────────────────┘
//! ```
//!
//! # Thread Hierarchy
//!
//! ```text
//! Grid (gridDim.x × gridDim.y blocks)
//! └── Block (blockDim.x × blockDim.y threads)
//!     └── Thread (identified by threadIdx)
//! ```
//!
//! Each operation returns a 32-bit integer representing the index or dimension.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Block-scoped memory fence.
///
/// Orders the calling thread's prior memory operations before later memory
/// operations as observed by threads in the same CTA. Corresponds to PTX
/// `membar.cta`.
#[pliron_op(
    name = "nvvm.threadfence_block",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct ThreadfenceBlockOp;

impl ThreadfenceBlockOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ThreadfenceBlockOp { op }
    }
}

/// Device-scoped memory fence.
///
/// Orders the calling thread's prior global-memory operations before later
/// memory operations as observed by threads on the same GPU. Corresponds to
/// PTX `membar.gl`.
#[pliron_op(
    name = "nvvm.threadfence",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct ThreadfenceOp;

impl ThreadfenceOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ThreadfenceOp { op }
    }
}

/// System-scoped memory fence.
///
/// Orders the calling thread's prior global-memory operations before later
/// memory operations as observed by other GPUs or the CPU. Corresponds to PTX
/// `membar.sys`.
#[pliron_op(
    name = "nvvm.threadfence_system",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct ThreadfenceSystemOp;

impl ThreadfenceSystemOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ThreadfenceSystemOp { op }
    }
}

/// Register thread indexing operations with the context.
pub(super) fn register(ctx: &mut Context) {
    // Memory fences
    ThreadfenceBlockOp::register(ctx);
    ThreadfenceOp::register(ctx);
    ThreadfenceSystemOp::register(ctx);
}
