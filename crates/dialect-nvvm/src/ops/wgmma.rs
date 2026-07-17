/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warpgroup Matrix Multiply-Accumulate (WGMMA) operations for Hopper `sm_90a`.
//!
//! WGMMA provides tensor core operations that operate at the warpgroup level
//! (4 warps = 128 threads) for high-throughput matrix multiplication.
//!
//! # WGMMA Workflow
//!
//! ```text
//! 1. wgmma.fence       │ Ensure shared memory is visible
//! 2. wgmma.mma         │ Issue matrix multiply (may issue multiple)
//! 3. wgmma.commit      │ Commit pending operations to group
//! 4. wgmma.wait        │ Wait for group completion
//! ```
//!
//! # Matrix Dimensions
//!
//! The `m64n64k16` variant computes:
//! - D = A × B + C where A is 64×16, B is 16×64, D/C is 64×64
//! - Each thread holds 32 f32 accumulator elements
//!
//! # Shared Memory Descriptors
//!
//! WGMMA uses 64-bit descriptors that encode:
//! - Base address (in shared memory address space)
//! - Stride information
//! - Swizzle mode for bank conflict avoidance
//!
//! # Requirements
//!
//! - **PTX ISA**: 8.0+
//! - **Architecture**: `sm_90a` (Hopper)
//! - **Execution**: Warpgroup-synchronous (128 threads must execute together)

use dialect_mir::types::{MirPtrType, address_space};
use pliron::{
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface},
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::Context,
    context::Ptr,
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    verify_err,
};
use pliron_derive::pliron_op;

// =============================================================================
// Descriptor Operations
// =============================================================================

/// Create a shared memory descriptor for WGMMA.
///
/// Converts a generic pointer into the fixed-layout descriptor used by the
/// current WGMMA lowering. It converts with `cvta.to.shared.u64`, shifts the
/// address right by 4, masks it with `0x3fff`, and ORs
/// `0xC000000800080000`.
///
/// # Operands
///
/// - `ptr` (ptr): generic pointer to shared memory
///
/// # Results
///
/// - `desc` (u64): 64-bit WGMMA descriptor
#[pliron_op(
    name = "nvvm.wgmma_make_smem_desc",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct WgmmaMakeSmemDescOp;

impl WgmmaMakeSmemDescOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        WgmmaMakeSmemDescOp { op }
    }
}

fn is_u64(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {
    ty.deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| {
            integer.width() == 64 && integer.signedness() == Signedness::Unsigned
        })
}

impl Verify for WgmmaMakeSmemDescOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_make_smem_desc requires one operand and one result"
            );
        }
        let pointer_ty = op.get_operand(0).get_type(ctx);
        let pointer_ty_obj = pointer_ty.deref(ctx);
        let Some(pointer_ty) = pointer_ty_obj.downcast_ref::<MirPtrType>() else {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_make_smem_desc operand must be a MIR pointer"
            );
        };
        if !matches!(
            pointer_ty.address_space,
            address_space::GENERIC | address_space::SHARED
        ) {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_make_smem_desc operand must point to generic or shared memory"
            );
        }
        if !is_u64(ctx, op.get_result(0).get_type(ctx)) {
            return verify_err!(op.loc(), "nvvm.wgmma_make_smem_desc result must be u64");
        }
        Ok(())
    }
}

// =============================================================================
// Matrix Multiply-Accumulate Operations
// =============================================================================

/// WGMMA Matrix Multiply-Accumulate: m64n64k16 with f32 accumulator and bf16 inputs.
///
/// Performs warpgroup-level matrix multiplication: D = A × B + D
/// - A: 64×16 (bf16)
/// - B: 16×64 (bf16)
/// - D: 64×64 (f32, 32 elements per thread, updated in-place)
///
/// PTX: `wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16`
///
/// # Operands
///
/// - `acc_ptr` (ptr): pointer to accumulator array (32 f32 values, read-modify-write)
/// - `desc_a` (u64): SMEM descriptor for matrix A
/// - `desc_b` (u64): SMEM descriptor for matrix B
///
/// # Results
///
/// - None (accumulator is updated in-place via pointer)
#[pliron_op(
    name = "nvvm.wgmma_mma_m64n64k16_f32_bf16",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<0>],
)]
pub struct WgmmaMmaM64N64K16F32Bf16Op;

impl WgmmaMmaM64N64K16F32Bf16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        WgmmaMmaM64N64K16F32Bf16Op { op }
    }
}

impl Verify for WgmmaMmaM64N64K16F32Bf16Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 3 || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_m64n64k16_f32_bf16 requires three operands and no results"
            );
        }
        let accumulator_ty = op.get_operand(0).get_type(ctx);
        if accumulator_ty
            .deref(ctx)
            .downcast_ref::<MirPtrType>()
            .is_none()
        {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_m64n64k16_f32_bf16 accumulator must be a MIR pointer"
            );
        }
        if !is_u64(ctx, op.get_operand(1).get_type(ctx))
            || !is_u64(ctx, op.get_operand(2).get_type(ctx))
        {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_m64n64k16_f32_bf16 descriptors must be u64"
            );
        }
        Ok(())
    }
}

/// Register WGMMA operations with the context.
pub(super) fn register(ctx: &mut Context) {
    WgmmaMakeSmemDescOp::register(ctx);
    WgmmaMmaM64N64K16F32Bf16Op::register(ctx);
}
