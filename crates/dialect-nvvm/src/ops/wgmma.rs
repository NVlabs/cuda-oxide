/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warpgroup Matrix Multiply-Accumulate (WGMMA) operations for Hopper `sm_90a`.
//!
//! WGMMA provides tensor core operations that operate at the warpgroup level
//! (4 warps = 128 threads) for high-throughput matrix multiplication.
//!
//! The public importer first creates a pointer-form MMA operation. Before LLVM
//! lowering, `mir-lower` recognizes complete straight-line regions and a narrow
//! canonical counted K-loop shape. Straight-line pointer-form sequences may use
//! the deferred group, which keeps all 32 per-thread accumulator values in
//! one inline-PTX scope until `wait_group<0>` completes.
//!
//! The internal value-form group represents those same 32 accumulator values
//! explicitly as SSA operands/results. A counted-loop variant additionally owns
//! the descriptor recurrences and loop control so the complete asynchronous
//! lifetime remains inside one convergent inline-PTX region. The pointer-form
//! group remains available as the deferred fallback.

use dialect_mir::types::{MirPtrType, address_space};
use pliron::{
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface},
        types::{FP32Type, IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    value::Value,
    verify_err,
};
use pliron_derive::pliron_op;

const WGMMA_M64N64_F32_ACCUMULATOR_COUNT: usize = 32;
const WGMMA_COUNTED_LOOP_CONTROL_COUNT: usize = 5;

// =============================================================================
// Descriptor Operations
// =============================================================================

/// Create a shared memory descriptor for WGMMA.
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

fn is_f32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {
    ty.deref(ctx).downcast_ref::<FP32Type>().is_some()
}

fn is_supported_wgmma_accumulator(ctx: &Context, value: Value) -> bool {
    let value_type = value.get_type(ctx);
    let value_type_ref = value_type.deref(ctx);
    let Some(pointer_type) = value_type_ref.downcast_ref::<MirPtrType>() else {
        return false;
    };

    pointer_type.is_mutable() && pointer_type.address_space() == address_space::GENERIC
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

/// Pointer-form BF16 WGMMA operation emitted by `mir-importer`.
///
/// This operation is not legal at final lowering. It must be consumed by the
/// deferred-accumulator fusion pass together with its fence, commit, and
/// `wait_group<0>` operations.
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
        if !is_supported_wgmma_accumulator(ctx, op.get_operand(0)) {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_m64n64k16_f32_bf16 accumulator must be a mutable generic MIR pointer"
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

/// Deferred BF16 WGMMA group with one accumulator and one or more descriptor pairs.
///
/// Operand layout:
///
/// ```text
/// [acc_ptr, desc_a_0, desc_b_0, ..., desc_a_n, desc_b_n]
/// ```
///
/// The operation represents a complete sequence containing an implicit fence,
/// all MMA instructions, one commit, and `wait_group<0>`. It has no results
/// because the accumulator is written back through `acc_ptr` after the wait.
#[pliron_op(name = "nvvm.wgmma_mma_group_m64n64k16_f32_bf16", format)]
pub struct WgmmaMmaGroupM64N64K16F32Bf16Op;

impl WgmmaMmaGroupM64N64K16F32Bf16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Build a deferred group from one accumulator and descriptor pairs.
    pub fn build(ctx: &mut Context, accumulator: Value, descriptors: Vec<Value>) -> Ptr<Operation> {
        let mut operands = Vec::with_capacity(1 + descriptors.len());
        operands.push(accumulator);
        operands.extend(descriptors);
        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            operands,
            vec![],
            0,
        )
    }
}

impl Verify for WgmmaMmaGroupM64N64K16F32Bf16Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let operand_count = op.get_num_operands();
        if operand_count < 3 || operand_count.is_multiple_of(2) || op.get_num_results() != 0 {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_group_m64n64k16_f32_bf16 requires one accumulator, one or more descriptor pairs, and no results"
            );
        }
        if !is_supported_wgmma_accumulator(ctx, op.get_operand(0)) {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_group_m64n64k16_f32_bf16 accumulator must be a mutable generic MIR pointer"
            );
        }
        for descriptor_index in 1..operand_count {
            if !is_u64(ctx, op.get_operand(descriptor_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_group_m64n64k16_f32_bf16 descriptors must be u64"
                );
            }
        }
        Ok(())
    }
}

/// Value-form BF16 WGMMA group with 32 SSA accumulator values.
///
/// Operand layout:
///
/// ```text
/// [acc_0, ..., acc_31, desc_a_0, desc_b_0, ..., desc_a_n, desc_b_n]
/// ```
///
/// Result layout:
///
/// ```text
/// [acc_0', ..., acc_31']
/// ```
///
/// The operation represents one complete register lifetime scope containing an
/// implicit fence, one or more MMA instructions, one commit, and
/// `wait_group<0>`. Unlike the deferred pointer-form group, it does not
/// materialize the accumulator through memory at either boundary.
#[pliron_op(name = "nvvm.wgmma_mma_group_values_m64n64k16_f32_bf16", format)]
pub struct WgmmaMmaGroupValuesM64N64K16F32Bf16Op;

impl WgmmaMmaGroupValuesM64N64K16F32Bf16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Build a value-form group from 32 accumulator values and descriptor pairs.
    pub fn build(
        ctx: &mut Context,
        accumulators: Vec<Value>,
        descriptors: Vec<Value>,
    ) -> Ptr<Operation> {
        let f32_ty = FP32Type::get(ctx);
        let mut operands = Vec::with_capacity(accumulators.len() + descriptors.len());
        operands.extend(accumulators);
        operands.extend(descriptors);

        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![f32_ty.into(); WGMMA_M64N64_F32_ACCUMULATOR_COUNT],
            operands,
            vec![],
            0,
        )
    }
}

impl Verify for WgmmaMmaGroupValuesM64N64K16F32Bf16Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let operand_count = op.get_num_operands();
        let result_count = op.get_num_results();

        if result_count != WGMMA_M64N64_F32_ACCUMULATOR_COUNT {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_group_values_m64n64k16_f32_bf16 requires exactly 32 f32 results"
            );
        }

        if operand_count < WGMMA_M64N64_F32_ACCUMULATOR_COUNT + 2 {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_group_values_m64n64k16_f32_bf16 requires 32 f32 accumulators and one or more descriptor pairs"
            );
        }

        let descriptor_count = operand_count - WGMMA_M64N64_F32_ACCUMULATOR_COUNT;
        if !descriptor_count.is_multiple_of(2) {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_group_values_m64n64k16_f32_bf16 descriptors must form pairs"
            );
        }

        for accumulator_index in 0..WGMMA_M64N64_F32_ACCUMULATOR_COUNT {
            if !is_f32(ctx, op.get_operand(accumulator_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_group_values_m64n64k16_f32_bf16 accumulator operands must be f32"
                );
            }

            if !is_f32(ctx, op.get_result(accumulator_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_group_values_m64n64k16_f32_bf16 results must be f32"
                );
            }
        }

        for descriptor_index in WGMMA_M64N64_F32_ACCUMULATOR_COUNT..operand_count {
            if !is_u64(ctx, op.get_operand(descriptor_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_group_values_m64n64k16_f32_bf16 descriptors must be u64"
                );
            }
        }

        Ok(())
    }
}

/// Value-form BF16 WGMMA counted loop with 32 SSA accumulator values.
///
/// Operand layout:
///
/// ```text
/// [
///   acc_0, ..., acc_31,
///   desc_a_base, desc_b_base,
///   desc_a_step, desc_b_step,
///   trip_count,
/// ]
/// ```
///
/// Result layout:
///
/// ```text
/// [acc_0', ..., acc_31']
/// ```
///
/// The operation owns one complete asynchronous WGMMA lifetime. It fences the
/// accumulator registers, executes one MMA per counted-loop iteration while
/// advancing both descriptors by their supplied descriptor deltas,
/// commits the resulting group, and performs a final `wait_group<0>` before the
/// 32 accumulator values become visible to LLVM again.
#[pliron_op(name = "nvvm.wgmma_mma_loop_values_m64n64k16_f32_bf16", format)]
pub struct WgmmaMmaLoopValuesM64N64K16F32Bf16Op;

impl WgmmaMmaLoopValuesM64N64K16F32Bf16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    /// Build a counted-loop group from 32 accumulators and loop-control values.
    pub fn build(
        ctx: &mut Context,
        accumulators: Vec<Value>,
        desc_a_base: Value,
        desc_b_base: Value,
        desc_a_step: Value,
        desc_b_step: Value,
        trip_count: Value,
    ) -> Ptr<Operation> {
        let f32_ty = FP32Type::get(ctx);
        let mut operands =
            Vec::with_capacity(accumulators.len() + WGMMA_COUNTED_LOOP_CONTROL_COUNT);
        operands.extend(accumulators);
        operands.extend([
            desc_a_base,
            desc_b_base,
            desc_a_step,
            desc_b_step,
            trip_count,
        ]);

        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![f32_ty.into(); WGMMA_M64N64_F32_ACCUMULATOR_COUNT],
            operands,
            vec![],
            0,
        )
    }
}

impl Verify for WgmmaMmaLoopValuesM64N64K16F32Bf16Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let expected_operands =
            WGMMA_M64N64_F32_ACCUMULATOR_COUNT + WGMMA_COUNTED_LOOP_CONTROL_COUNT;

        if op.get_num_operands() != expected_operands {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_loop_values_m64n64k16_f32_bf16 requires 32 f32 accumulators and exactly five u64 loop-control operands"
            );
        }

        if op.get_num_results() != WGMMA_M64N64_F32_ACCUMULATOR_COUNT {
            return verify_err!(
                op.loc(),
                "nvvm.wgmma_mma_loop_values_m64n64k16_f32_bf16 requires exactly 32 f32 results"
            );
        }

        for accumulator_index in 0..WGMMA_M64N64_F32_ACCUMULATOR_COUNT {
            if !is_f32(ctx, op.get_operand(accumulator_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_loop_values_m64n64k16_f32_bf16 accumulator operands must be f32"
                );
            }
            if !is_f32(ctx, op.get_result(accumulator_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_loop_values_m64n64k16_f32_bf16 results must be f32"
                );
            }
        }

        for control_index in WGMMA_M64N64_F32_ACCUMULATOR_COUNT..expected_operands {
            if !is_u64(ctx, op.get_operand(control_index).get_type(ctx)) {
                return verify_err!(
                    op.loc(),
                    "nvvm.wgmma_mma_loop_values_m64n64k16_f32_bf16 descriptor bases, descriptor steps, and trip count must be u64"
                );
            }
        }

        Ok(())
    }
}

/// Register WGMMA operations with the context.
pub(super) fn register(ctx: &mut Context) {
    WgmmaMakeSmemDescOp::register(ctx);
    WgmmaMmaM64N64K16F32Bf16Op::register(ctx);
    WgmmaMmaGroupM64N64K16F32Bf16Op::register(ctx);
    WgmmaMmaGroupValuesM64N64K16F32Bf16Op::register(ctx);
    WgmmaMmaLoopValuesM64N64K16F32Bf16Op::register(ctx);
}
