/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared memory matrix load (ldmatrix) operations.
//!
//! Ldmatrix provides warp-cooperative matrix load operations that read 8×8
//! matrix tiles from shared memory into registers in the fragment layout
//! expected by tensor core `mma.sync` instructions.
//!
//! # Layout
//!
//! ```text
//! ┌───────────────────────┬───────┬──────────┬───────────┬─────────────────────┐
//! │ Operation             │ Tiles │ Regs     │ Transpose │ PTX                 │
//! ├───────────────────────┼───────┼──────────┼───────────┼─────────────────────┤
//! │ LdmatrixX1Op          │ 1     │ 1 × u32  │ No        │ ldmatrix...m8n8.x1  │
//! │ LdmatrixX1TransOp     │ 1     │ 1 × u32  │ Yes       │ ldmatrix...x1.trans │
//! │ LdmatrixX2Op          │ 2     │ 2 × u32  │ No        │ ldmatrix...m8n8.x2  │
//! │ LdmatrixX2TransOp     │ 2     │ 2 × u32  │ Yes       │ ldmatrix...x2.trans │
//! │ LdmatrixX4Op          │ 4     │ 4 × u32  │ No        │ ldmatrix...m8n8.x4  │
//! │ LdmatrixX4TransOp     │ 4     │ 4 × u32  │ Yes       │ ldmatrix...x4.trans │
//! └───────────────────────┴───────┴──────────┴───────────┴─────────────────────┘
//! ```
//!
//! # Return Patterns
//!
//! - **x1 variants**: 1 operand (smem_ptr), 1 result (u32) -- scalar return
//! - **x2/x4 variants**: 2 operands (smem_ptr, dest_ptr), 0 results -- alloca-slot pattern
//!
//! # Requirements
//!
//! - **Execution**: Warp-synchronous (all 32 threads must participate)
//! - **Memory**: Source must be in shared memory
//! - **Alignment**: Pointer must be 16-byte aligned
//! - **Architecture**: sm_75+ (Turing and later)

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

// =============================================================================
// 1-Tile Load Operations (scalar return: 1 operand, 1 result)
// =============================================================================

/// Load one 8×8 matrix tile from shared memory.
///
/// PTX: `ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%r0}, [addr];`
///
/// # Operands
///
/// - `smem_ptr` (ptr): source pointer in shared memory
///
/// # Results
///
/// - `r0` (u32): loaded register (2 packed b16 values)
#[pliron_op(
    name = "nvvm.ldmatrix_x1",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct LdmatrixX1Op;

impl LdmatrixX1Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        LdmatrixX1Op { op }
    }
}

/// Load one 8×8 matrix tile from shared memory with transpose.
///
/// PTX: `ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%r0}, [addr];`
///
/// # Operands
///
/// - `smem_ptr` (ptr): source pointer in shared memory
///
/// # Results
///
/// - `r0` (u32): loaded register (2 packed b16 values)
#[pliron_op(
    name = "nvvm.ldmatrix_x1_trans",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct LdmatrixX1TransOp;

impl LdmatrixX1TransOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        LdmatrixX1TransOp { op }
    }
}

// =============================================================================
// 2-Tile Load Operations (alloca-slot: 2 operands, 0 results)
// =============================================================================

/// Load two 8×8 matrix tiles from shared memory.
///
/// PTX: `ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%r0, %r1}, [addr];`
///
/// Uses the alloca-slot pattern: the inline asm loads from shared memory and
/// stores the results directly into the destination slot pointer.
///
/// # Operands
///
/// - `smem_ptr` (ptr): source pointer in shared memory
/// - `dest_ptr` (ptr): destination alloca slot pointer
///
/// # Results
///
/// - None (void op; results written via dest_ptr)
#[pliron_op(
    name = "nvvm.ldmatrix_x2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct LdmatrixX2Op;

impl LdmatrixX2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        LdmatrixX2Op { op }
    }
}

/// Load two 8×8 matrix tiles from shared memory with transpose.
///
/// PTX: `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r0, %r1}, [addr];`
///
/// # Operands
///
/// - `smem_ptr` (ptr): source pointer in shared memory
/// - `dest_ptr` (ptr): destination alloca slot pointer
///
/// # Results
///
/// - None (void op; results written via dest_ptr)
#[pliron_op(
    name = "nvvm.ldmatrix_x2_trans",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct LdmatrixX2TransOp;

impl LdmatrixX2TransOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        LdmatrixX2TransOp { op }
    }
}

// =============================================================================
// 4-Tile Load Operations (alloca-slot: 2 operands, 0 results)
// =============================================================================

/// Load four 8×8 matrix tiles from shared memory.
///
/// PTX: `ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r0, %r1, %r2, %r3}, [addr];`
///
/// Uses the alloca-slot pattern: the inline asm loads from shared memory and
/// stores the results directly into the destination slot pointer.
///
/// # Operands
///
/// - `smem_ptr` (ptr): source pointer in shared memory
/// - `dest_ptr` (ptr): destination alloca slot pointer
///
/// # Results
///
/// - None (void op; results written via dest_ptr)
#[pliron_op(
    name = "nvvm.ldmatrix_x4",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct LdmatrixX4Op;

impl LdmatrixX4Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        LdmatrixX4Op { op }
    }
}

/// Load four 8×8 matrix tiles from shared memory with transpose.
///
/// PTX: `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r0, %r1, %r2, %r3}, [addr];`
///
/// # Operands
///
/// - `smem_ptr` (ptr): source pointer in shared memory
/// - `dest_ptr` (ptr): destination alloca slot pointer
///
/// # Results
///
/// - None (void op; results written via dest_ptr)
#[pliron_op(
    name = "nvvm.ldmatrix_x4_trans",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
)]
pub struct LdmatrixX4TransOp;

impl LdmatrixX4TransOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        LdmatrixX4TransOp { op }
    }
}

/// Register ldmatrix operations with the context.
pub(super) fn register(ctx: &mut Context) {
    // 1-tile load (scalar return)
    LdmatrixX1Op::register(ctx);
    LdmatrixX1TransOp::register(ctx);
    // 2-tile load (alloca-slot)
    LdmatrixX2Op::register(ctx);
    LdmatrixX2TransOp::register(ctx);
    // 4-tile load (alloca-slot)
    LdmatrixX4Op::register(ctx);
    LdmatrixX4TransOp::register(ctx);
}
