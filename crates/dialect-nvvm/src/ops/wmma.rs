/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix dialect operations.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    builtin::types::{FP32Type, FP64Type, IntegerType},
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

/// In-register 8×8 matrix transpose (movmatrix.sync.aligned.m8n8.trans.b16).
#[pliron_op(
    name = "nvvm.movmatrix_trans_b16",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct MovmatrixTransB16Op;

impl MovmatrixTransB16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for MovmatrixTransB16Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);

        if op.operands().count() != 1 || op.get_num_results() != 1 {
            return verify_err!(
                op.loc(),
                "nvvm.movmatrix_trans_b16 requires one operand and one result"
            );
        }

        for (name, ty) in [
            ("operand", op.get_operand(0).get_type(ctx)),
            ("result", op.get_result(0).get_type(ctx)),
        ] {
            let ty_ref = ty.deref(ctx);
            let Some(integer) = ty_ref.downcast_ref::<IntegerType>() else {
                return verify_err!(
                    op.loc(),
                    "nvvm.movmatrix_trans_b16 {} must be a 32-bit integer",
                    name
                );
            };
            if integer.width() != 32 {
                return verify_err!(
                    op.loc(),
                    "nvvm.movmatrix_trans_b16 {} must be a 32-bit integer",
                    name
                );
            }
        }

        Ok(())
    }
}

/// Register-only warp MMA: m16n8k16 with f32 accumulator and bf16 inputs.
///
/// # Operands
///
/// - operands 0-3: four f32 C accumulator registers
/// - operands 4-7: four i32 A fragment registers, each packing two BF16 values
/// - operands 8-9: two i32 B fragment registers, each packing two BF16 values
///
/// # Results
///
/// - results 0-3: four f32 D accumulator registers
#[pliron_op(
    name = "nvvm.mma_m16n8k16_f32_bf16",
    format,
    interfaces = [NOpdsInterface<10>, NResultsInterface<4>],
)]
pub struct MmaM16N8K16F32Bf16Op;

impl Verify for MmaM16N8K16F32Bf16Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let operands: Vec<_> = op.operands().collect();

        if operands.len() != 10 {
            return verify_err!(
                op.loc(),
                "nvvm.mma_m16n8k16_f32_bf16 requires 10 register operands, got {}",
                operands.len()
            );
        }

        for (index, operand) in operands.iter().take(4).enumerate() {
            let ty = operand.get_type(ctx);
            if ty.deref(ctx).downcast_ref::<FP32Type>().is_none() {
                return verify_err!(
                    op.loc(),
                    "nvvm.mma_m16n8k16_f32_bf16 C operand {} must be f32",
                    index
                );
            }
        }

        for (index, operand) in operands.iter().enumerate().skip(4) {
            let ty = operand.get_type(ctx);
            let ty = ty.deref(ctx);
            let Some(integer) = ty.downcast_ref::<IntegerType>() else {
                return verify_err!(
                    op.loc(),
                    "nvvm.mma_m16n8k16_f32_bf16 packed operand {} must be i32",
                    index
                );
            };
            if integer.width() != 32 {
                return verify_err!(
                    op.loc(),
                    "nvvm.mma_m16n8k16_f32_bf16 packed operand {} must be i32",
                    index
                );
            }
        }

        if op.get_num_results() != 4 {
            return verify_err!(
                op.loc(),
                "nvvm.mma_m16n8k16_f32_bf16 requires 4 f32 results"
            );
        }

        for index in 0..4 {
            let ty = op.get_result(index).get_type(ctx);
            if ty.deref(ctx).downcast_ref::<FP32Type>().is_none() {
                return verify_err!(
                    op.loc(),
                    "nvvm.mma_m16n8k16_f32_bf16 result {} must be f32",
                    index
                );
            }
        }

        Ok(())
    }
}

impl MmaM16N8K16F32Bf16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K16F32Bf16Op { op }
    }
}

/// Register-only warp MMA: m16n8k16 with f32 accumulator and f16 inputs.
///
/// # Operands
///
/// - operands 0-3: four f32 C accumulator registers
/// - operands 4-7: four i32 A fragment registers, each packing two F16 values
/// - operands 8-9: two i32 B fragment registers, each packing two F16 values
///
/// # Results
///
/// - results 0-3: four f32 D accumulator registers
#[pliron_op(
    name = "nvvm.mma_m16n8k16_f32_f16",
    format,
    interfaces = [NOpdsInterface<10>, NResultsInterface<4>],
)]
pub struct MmaM16N8K16F32F16Op;

impl Verify for MmaM16N8K16F32F16Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let operands: Vec<_> = op.operands().collect();

        if operands.len() != 10 {
            return verify_err!(
                op.loc(),
                "nvvm.mma_m16n8k16_f32_f16 requires 10 register operands, got {}",
                operands.len()
            );
        }

        for (index, operand) in operands.iter().take(4).enumerate() {
            let ty = operand.get_type(ctx);
            if ty.deref(ctx).downcast_ref::<FP32Type>().is_none() {
                return verify_err!(
                    op.loc(),
                    "nvvm.mma_m16n8k16_f32_f16 C operand {} must be f32",
                    index
                );
            }
        }

        for (index, operand) in operands.iter().enumerate().skip(4) {
            let ty = operand.get_type(ctx);
            let ty = ty.deref(ctx);
            let Some(integer) = ty.downcast_ref::<IntegerType>() else {
                return verify_err!(
                    op.loc(),
                    "nvvm.mma_m16n8k16_f32_f16 packed operand {} must be i32",
                    index
                );
            };
            if integer.width() != 32 {
                return verify_err!(
                    op.loc(),
                    "nvvm.mma_m16n8k16_f32_f16 packed operand {} must be i32",
                    index
                );
            }
        }

        if op.get_num_results() != 4 {
            return verify_err!(op.loc(), "nvvm.mma_m16n8k16_f32_f16 requires 4 f32 results");
        }

        for index in 0..4 {
            let ty = op.get_result(index).get_type(ctx);
            if ty.deref(ctx).downcast_ref::<FP32Type>().is_none() {
                return verify_err!(
                    op.loc(),
                    "nvvm.mma_m16n8k16_f32_f16 result {} must be f32",
                    index
                );
            }
        }

        Ok(())
    }
}

impl MmaM16N8K16F32F16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K16F32F16Op { op }
    }
}

/// Register-only warp MMA: m16n8k8 with f32 accumulator and tf32 inputs.
///
/// # Operands
///
/// - operands 0-3: four f32 C accumulator registers
/// - operands 4-7: four i32 A fragment registers, each holding one TF32 value
/// - operands 8-9: two i32 B fragment registers, each holding one TF32 value
///
/// # Results
///
/// - results 0-3: four f32 D accumulator registers
#[pliron_op(
    name = "nvvm.mma_m16n8k8_f32_tf32",
    format,
    interfaces = [NOpdsInterface<10>, NResultsInterface<4>],
)]
pub struct MmaM16N8K8F32Tf32Op;

impl Verify for MmaM16N8K8F32Tf32Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let operands: Vec<_> = op.operands().collect();

        if operands.len() != 10 {
            return verify_err!(
                op.loc(),
                "nvvm.mma_m16n8k8_f32_tf32 requires 10 register operands, got {}",
                operands.len()
            );
        }

        for (index, operand) in operands.iter().take(4).enumerate() {
            let ty = operand.get_type(ctx);
            if ty.deref(ctx).downcast_ref::<FP32Type>().is_none() {
                return verify_err!(
                    op.loc(),
                    "nvvm.mma_m16n8k8_f32_tf32 C operand {} must be f32",
                    index
                );
            }
        }

        for (index, operand) in operands.iter().enumerate().skip(4) {
            let ty = operand.get_type(ctx);
            let ty = ty.deref(ctx);
            let Some(integer) = ty.downcast_ref::<IntegerType>() else {
                return verify_err!(
                    op.loc(),
                    "nvvm.mma_m16n8k8_f32_tf32 packed operand {} must be i32",
                    index
                );
            };
            if integer.width() != 32 {
                return verify_err!(
                    op.loc(),
                    "nvvm.mma_m16n8k8_f32_tf32 packed operand {} must be i32",
                    index
                );
            }
        }

        if op.get_num_results() != 4 {
            return verify_err!(op.loc(), "nvvm.mma_m16n8k8_f32_tf32 requires 4 f32 results");
        }

        for index in 0..4 {
            let ty = op.get_result(index).get_type(ctx);
            if ty.deref(ctx).downcast_ref::<FP32Type>().is_none() {
                return verify_err!(
                    op.loc(),
                    "nvvm.mma_m16n8k8_f32_tf32 result {} must be f32",
                    index
                );
            }
        }

        Ok(())
    }
}

impl MmaM16N8K8F32Tf32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM16N8K8F32Tf32Op { op }
    }
}

/// Warp MMA: m8n8k4 with f64 accumulator and f64 inputs.
///
/// # Operands
///
/// - `c0`, `c1` (f64): lane-local accumulator fragment
/// - `a` (f64): lane-local A fragment
/// - `b` (f64): lane-local B fragment
///
/// # Results
///
/// - `d0`, `d1` (f64): lane-local result fragment
#[pliron_op(
    name = "nvvm.mma_m8n8k4_f64",
    format,
    interfaces = [NOpdsInterface<4>, NResultsInterface<2>],
)]
pub struct MmaM8N8K4F64Op;

impl Verify for MmaM8N8K4F64Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let operands: Vec<_> = op.operands().collect();

        if operands.len() != 4 {
            return verify_err!(
                op.loc(),
                "nvvm.mma_m8n8k4_f64 requires 4 f64 operands (c0, c1, a, b), got {}",
                operands.len()
            );
        }

        for (i, operand) in operands.iter().enumerate() {
            let ty = operand.get_type(ctx);
            if ty.deref(ctx).downcast_ref::<FP64Type>().is_none() {
                return verify_err!(op.loc(), "nvvm.mma_m8n8k4_f64 operand {} must be f64", i);
            }
        }

        if op.get_num_results() != 2 {
            return verify_err!(
                op.loc(),
                "nvvm.mma_m8n8k4_f64 requires 2 f64 results (d0, d1), got {}",
                op.get_num_results()
            );
        }
        for i in 0..2 {
            let ty = op.get_result(i).get_type(ctx);
            if ty.deref(ctx).downcast_ref::<FP64Type>().is_none() {
                return verify_err!(op.loc(), "nvvm.mma_m8n8k4_f64 result {} must be f64", i);
            }
        }

        Ok(())
    }
}

impl MmaM8N8K4F64Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaM8N8K4F64Op { op }
    }
}

/// Register WMMA operations with the context.
pub(super) fn register(ctx: &mut Context) {
    MovmatrixTransB16Op::register(ctx);
    MmaM16N8K16F32Bf16Op::register(ctx);
    MmaM16N8K16F32F16Op::register(ctx);
    MmaM16N8K8F32Tf32Op::register(ctx);
    MmaM8N8K4F64Op::register(ctx);
}
