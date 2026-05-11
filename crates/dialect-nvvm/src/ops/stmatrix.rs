/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared memory matrix store (stmatrix) operations.
//!
//! Stmatrix provides warp-cooperative matrix store operations that properly
//! handle tensor core fragment layouts when writing to shared memory.
//!
//! # Layout Transformation
//!
//! Tensor core operations produce data in "fragment" layout (optimized for
//! computation). Stmatrix transforms this to row-major layout for memory access.
//!
//! ```text
//! ┌─────────────────────┬───────┬──────────┬───────────┬────────────────────┐
//! │ Operation           │ Tiles │ Elements │ Transpose │ PTX                │
//! ├─────────────────────┼───────┼──────────┼───────────┼────────────────────┤
//! │ StmatrixM8n8X4Op    │ 4     │ 256      │ No        │ stmatrix...m8n8.x4 │
//! │ StmatrixM8n8X4Trans │ 4     │ 256      │ Yes       │ stmatrix...x4.trans│
//! │ StmatrixM8n8X2Op    │ 2     │ 128      │ No        │ stmatrix...m8n8.x2 │
//! │ StmatrixM8n8X2Trans │ 2     │ 128      │ Yes       │ stmatrix...x2.trans│
//! └─────────────────────┴───────┴──────────┴───────────┴────────────────────┘
//! ```
//!
//! # Type Conversion
//!
//! - `CvtF32x2Bf16x2Op`: Convert two f32 values to packed bf16x2 (round-to-nearest-even)
//!
//! # Requirements
//!
//! - **Execution**: Warp-synchronous (all 32 threads must participate)
//! - **Memory**: Destination must be in shared memory
//! - **Alignment**: Pointer must be aligned to tile size

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

// =============================================================================
// 4-Tile Store Operations
// =============================================================================

/// Store four 8×8 matrix tiles to shared memory.
///
/// Warp-cooperative matrix store without transpose.
///
/// PTX: `stmatrix.sync.aligned.m8n8.x4.shared.b16 [ptr], {r0, r1, r2, r3};`
///
/// # Operands
///
/// - `smem_ptr` (ptr): destination pointer in shared memory
/// - `r0` (f32): first register value
/// - `r1` (f32): second register value
/// - `r2` (f32): third register value
/// - `r3` (f32): fourth register value
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.stmatrix_m8n8_x4",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<5>, NResultsInterface<0>],
)]
pub struct StmatrixM8n8X4Op;

impl StmatrixM8n8X4Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Store four 8×8 matrix tiles with transpose.
///
/// Warp-cooperative matrix store with the `.trans` modifier that transforms
/// data from fragment layout to row-major layout.
///
/// PTX: `stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [ptr], {r0, r1, r2, r3};`
///
/// # Operands
///
/// - `smem_ptr` (ptr): destination pointer in shared memory
/// - `r0` (u32): first register (2 packed bf16 values)
/// - `r1` (u32): second register
/// - `r2` (u32): third register
/// - `r3` (u32): fourth register
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.stmatrix_m8n8_x4_trans",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<5>, NResultsInterface<0>],
)]
pub struct StmatrixM8n8X4TransOp;

impl StmatrixM8n8X4TransOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

// =============================================================================
// 2-Tile Store Operations
// =============================================================================

/// Store two 8×8 matrix tiles to shared memory.
///
/// Warp-cooperative matrix store without transpose. Stores 16 columns
/// (2 × 8×8 tiles) per call.
///
/// SASS encoding: `STSM.16.MT88.2`
///
/// PTX: `stmatrix.sync.aligned.m8n8.x2.shared.b16 [ptr], {r0, r1};`
///
/// # Operands
///
/// - `smem_ptr` (ptr): destination pointer in shared memory
/// - `r0` (i32): first register (2 packed bf16 values)
/// - `r1` (i32): second register
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.stmatrix_m8n8_x2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct StmatrixM8n8X2Op;

impl StmatrixM8n8X2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Store two 8×8 bf16 matrices to shared memory with transpose.
///
/// The TRANSPOSE version matching cuBLAS `STSM.16.MT88.2`.
/// Converts from fragment (column-major) to row-major layout during store.
///
/// PTX: `stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [ptr], {r0, r1};`
///
/// # Operands
///
/// - `smem_ptr` (ptr): destination pointer in shared memory
/// - `r0` (i32): first register (2 packed bf16 values)
/// - `r1` (i32): second register
///
/// # Results
///
/// - None
#[pliron_op(
    name = "nvvm.stmatrix_m8n8_x2_trans",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct StmatrixM8n8X2TransOp;

impl StmatrixM8n8X2TransOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

// =============================================================================
// Type Conversion Operations
// =============================================================================

/// Convert two f32 values to packed bf16x2 using round-to-nearest-even.
///
/// Uses PTX `cvt.rn.bf16x2.f32` instruction for proper IEEE rounding.
///
/// PTX: `cvt.rn.bf16x2.f32 %result, %b, %a;`
///
/// # Operands
///
/// - `a` (f32): first value (goes to low 16 bits of result)
/// - `b` (f32): second value (goes to high 16 bits of result)
///
/// # Results
///
/// - `packed` (i32): packed bf16x2 as `(bf16(b) << 16) | bf16(a)`
#[pliron_op(
    name = "nvvm.cvt_f32x2_bf16x2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct CvtF32x2Bf16x2Op;

impl CvtF32x2Bf16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Register stmatrix operations with the context.
pub(super) fn register(ctx: &mut Context) {
    // 4-tile store
    StmatrixM8n8X4Op::register(ctx);
    StmatrixM8n8X4TransOp::register(ctx);
    // 2-tile store
    StmatrixM8n8X2Op::register(ctx);
    StmatrixM8n8X2TransOp::register(ctx);
    // Type conversion
    CvtF32x2Bf16x2Op::register(ctx);
}
