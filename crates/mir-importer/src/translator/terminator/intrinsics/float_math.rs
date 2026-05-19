/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Rust compiler floating-point math intrinsics.

use super::super::helpers;
use crate::error::TranslationResult;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_mir::rust_intrinsics;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::location::Location;
use pliron::operation::Operation;
use rustc_public::mir;

/// Floating-point math intrinsic from libcore.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RustFloatMathIntrinsic {
    /// `core::intrinsics::sqrtf32`.
    SqrtF32,
    /// `core::intrinsics::sqrtf64`.
    SqrtF64,
    /// `cuda_device::math::rsqrt_f32` (cuda-oxide-specific; no rustc analogue).
    RsqrtF32,
    /// `cuda_device::math::rsqrt_f64` (cuda-oxide-specific; no rustc analogue).
    RsqrtF64,
    /// `core::intrinsics::powif32`.
    PowiF32,
    /// `core::intrinsics::powif64`.
    PowiF64,
    /// `core::intrinsics::sinf32`.
    SinF32,
    /// `core::intrinsics::sinf64`.
    SinF64,
    /// `core::intrinsics::cosf32`.
    CosF32,
    /// `core::intrinsics::cosf64`.
    CosF64,
    /// `core::intrinsics::tanf32`.
    TanF32,
    /// `core::intrinsics::tanf64`.
    TanF64,
    /// `core::intrinsics::powf32`.
    PowfF32,
    /// `core::intrinsics::powf64`.
    PowfF64,
    /// `core::intrinsics::expf32`.
    ExpF32,
    /// `core::intrinsics::expf64`.
    ExpF64,
    /// `core::intrinsics::exp2f32`.
    Exp2F32,
    /// `core::intrinsics::exp2f64`.
    Exp2F64,
    /// `core::intrinsics::logf32`.
    LogF32,
    /// `core::intrinsics::logf64`.
    LogF64,
    /// `core::intrinsics::log2f32`.
    Log2F32,
    /// `core::intrinsics::log2f64`.
    Log2F64,
    /// `core::intrinsics::log10f32`.
    Log10F32,
    /// `core::intrinsics::log10f64`.
    Log10F64,
    /// `core::intrinsics::fmaf32`.
    FmaF32,
    /// `core::intrinsics::fmaf64`.
    FmaF64,
    /// `core::intrinsics::fmuladdf32`.
    FmuladdF32,
    /// `core::intrinsics::fmuladdf64`.
    FmuladdF64,
    /// `core::intrinsics::floorf32`.
    FloorF32,
    /// `core::intrinsics::floorf64`.
    FloorF64,
    /// `core::intrinsics::ceilf32`.
    CeilF32,
    /// `core::intrinsics::ceilf64`.
    CeilF64,
    /// `core::intrinsics::truncf32`.
    TruncF32,
    /// `core::intrinsics::truncf64`.
    TruncF64,
    /// `core::intrinsics::roundf32`.
    RoundF32,
    /// `core::intrinsics::roundf64`.
    RoundF64,
    /// `core::intrinsics::round_ties_even_f32`.
    RoundevenF32,
    /// `core::intrinsics::round_ties_even_f64`.
    RoundevenF64,
    /// Generic `core::intrinsics::fabs`.
    Fabs,
    /// `core::intrinsics::copysignf32`.
    CopysignF32,
    /// `core::intrinsics::copysignf64`.
    CopysignF64,
}

impl RustFloatMathIntrinsic {
    /// Recognize the libcore intrinsic path that survived into MIR.
    pub fn from_core_path(name: &str) -> Option<Self> {
        match name {
            "core::intrinsics::sqrtf32" | "std::intrinsics::sqrtf32" => Some(Self::SqrtF32),
            "core::intrinsics::sqrtf64" | "std::intrinsics::sqrtf64" => Some(Self::SqrtF64),
            "core::intrinsics::powif32" | "std::intrinsics::powif32" => Some(Self::PowiF32),
            "core::intrinsics::powif64" | "std::intrinsics::powif64" => Some(Self::PowiF64),
            "core::intrinsics::sinf32" | "std::intrinsics::sinf32" => Some(Self::SinF32),
            "core::intrinsics::sinf64" | "std::intrinsics::sinf64" => Some(Self::SinF64),
            "core::intrinsics::cosf32" | "std::intrinsics::cosf32" => Some(Self::CosF32),
            "core::intrinsics::cosf64" | "std::intrinsics::cosf64" => Some(Self::CosF64),
            "core::intrinsics::tanf32" | "std::intrinsics::tanf32" => Some(Self::TanF32),
            "core::intrinsics::tanf64" | "std::intrinsics::tanf64" => Some(Self::TanF64),
            "core::intrinsics::powf32" | "std::intrinsics::powf32" => Some(Self::PowfF32),
            "core::intrinsics::powf64" | "std::intrinsics::powf64" => Some(Self::PowfF64),
            "core::intrinsics::expf32" | "std::intrinsics::expf32" => Some(Self::ExpF32),
            "core::intrinsics::expf64" | "std::intrinsics::expf64" => Some(Self::ExpF64),
            "core::intrinsics::exp2f32" | "std::intrinsics::exp2f32" => Some(Self::Exp2F32),
            "core::intrinsics::exp2f64" | "std::intrinsics::exp2f64" => Some(Self::Exp2F64),
            "core::intrinsics::logf32" | "std::intrinsics::logf32" => Some(Self::LogF32),
            "core::intrinsics::logf64" | "std::intrinsics::logf64" => Some(Self::LogF64),
            "core::intrinsics::log2f32" | "std::intrinsics::log2f32" => Some(Self::Log2F32),
            "core::intrinsics::log2f64" | "std::intrinsics::log2f64" => Some(Self::Log2F64),
            "core::intrinsics::log10f32" | "std::intrinsics::log10f32" => Some(Self::Log10F32),
            "core::intrinsics::log10f64" | "std::intrinsics::log10f64" => Some(Self::Log10F64),
            "core::intrinsics::fmaf32" | "std::intrinsics::fmaf32" => Some(Self::FmaF32),
            "core::intrinsics::fmaf64" | "std::intrinsics::fmaf64" => Some(Self::FmaF64),
            "core::intrinsics::fmuladdf32" | "std::intrinsics::fmuladdf32" => {
                Some(Self::FmuladdF32)
            }
            "core::intrinsics::fmuladdf64" | "std::intrinsics::fmuladdf64" => {
                Some(Self::FmuladdF64)
            }
            "core::intrinsics::floorf32" | "std::intrinsics::floorf32" => Some(Self::FloorF32),
            "core::intrinsics::floorf64" | "std::intrinsics::floorf64" => Some(Self::FloorF64),
            "core::intrinsics::ceilf32" | "std::intrinsics::ceilf32" => Some(Self::CeilF32),
            "core::intrinsics::ceilf64" | "std::intrinsics::ceilf64" => Some(Self::CeilF64),
            "core::intrinsics::truncf32" | "std::intrinsics::truncf32" => Some(Self::TruncF32),
            "core::intrinsics::truncf64" | "std::intrinsics::truncf64" => Some(Self::TruncF64),
            "core::intrinsics::roundf32" | "std::intrinsics::roundf32" => Some(Self::RoundF32),
            "core::intrinsics::roundf64" | "std::intrinsics::roundf64" => Some(Self::RoundF64),
            "core::intrinsics::round_ties_even_f32" | "std::intrinsics::round_ties_even_f32" => {
                Some(Self::RoundevenF32)
            }
            "core::intrinsics::round_ties_even_f64" | "std::intrinsics::round_ties_even_f64" => {
                Some(Self::RoundevenF64)
            }
            "core::intrinsics::fabs" | "std::intrinsics::fabs" => Some(Self::Fabs),
            "core::intrinsics::copysignf32" | "std::intrinsics::copysignf32" => {
                Some(Self::CopysignF32)
            }
            "core::intrinsics::copysignf64" | "std::intrinsics::copysignf64" => {
                Some(Self::CopysignF64)
            }
            _ => None,
        }
    }

    /// Recognize cuda-oxide-specific math paths that don't have a Rust core analogue.
    ///
    /// Currently covers `cuda_device::math::rsqrt_f32` and `rsqrt_f64`, which lower
    /// to libdevice `__nv_rsqrtf` / `__nv_rsqrt`.
    pub fn from_cuda_device_math_path(name: &str) -> Option<Self> {
        match name {
            "cuda_device::math::rsqrt_f32" => Some(Self::RsqrtF32),
            "cuda_device::math::rsqrt_f64" => Some(Self::RsqrtF64),
            _ => None,
        }
    }

    /// Return the internal placeholder name used until MIR-to-LLVM lowering.
    pub fn placeholder_callee(self) -> &'static str {
        match self {
            Self::SqrtF32 => rust_intrinsics::CALLEE_SQRT_F32,
            Self::SqrtF64 => rust_intrinsics::CALLEE_SQRT_F64,
            Self::RsqrtF32 => rust_intrinsics::CALLEE_RSQRT_F32,
            Self::RsqrtF64 => rust_intrinsics::CALLEE_RSQRT_F64,
            Self::PowiF32 => rust_intrinsics::CALLEE_POWI_F32,
            Self::PowiF64 => rust_intrinsics::CALLEE_POWI_F64,
            Self::SinF32 => rust_intrinsics::CALLEE_SIN_F32,
            Self::SinF64 => rust_intrinsics::CALLEE_SIN_F64,
            Self::CosF32 => rust_intrinsics::CALLEE_COS_F32,
            Self::CosF64 => rust_intrinsics::CALLEE_COS_F64,
            Self::TanF32 => rust_intrinsics::CALLEE_TAN_F32,
            Self::TanF64 => rust_intrinsics::CALLEE_TAN_F64,
            Self::PowfF32 => rust_intrinsics::CALLEE_POWF_F32,
            Self::PowfF64 => rust_intrinsics::CALLEE_POWF_F64,
            Self::ExpF32 => rust_intrinsics::CALLEE_EXP_F32,
            Self::ExpF64 => rust_intrinsics::CALLEE_EXP_F64,
            Self::Exp2F32 => rust_intrinsics::CALLEE_EXP2_F32,
            Self::Exp2F64 => rust_intrinsics::CALLEE_EXP2_F64,
            Self::LogF32 => rust_intrinsics::CALLEE_LOG_F32,
            Self::LogF64 => rust_intrinsics::CALLEE_LOG_F64,
            Self::Log2F32 => rust_intrinsics::CALLEE_LOG2_F32,
            Self::Log2F64 => rust_intrinsics::CALLEE_LOG2_F64,
            Self::Log10F32 => rust_intrinsics::CALLEE_LOG10_F32,
            Self::Log10F64 => rust_intrinsics::CALLEE_LOG10_F64,
            Self::FmaF32 => rust_intrinsics::CALLEE_FMA_F32,
            Self::FmaF64 => rust_intrinsics::CALLEE_FMA_F64,
            Self::FmuladdF32 => rust_intrinsics::CALLEE_FMULADD_F32,
            Self::FmuladdF64 => rust_intrinsics::CALLEE_FMULADD_F64,
            Self::FloorF32 => rust_intrinsics::CALLEE_FLOOR_F32,
            Self::FloorF64 => rust_intrinsics::CALLEE_FLOOR_F64,
            Self::CeilF32 => rust_intrinsics::CALLEE_CEIL_F32,
            Self::CeilF64 => rust_intrinsics::CALLEE_CEIL_F64,
            Self::TruncF32 => rust_intrinsics::CALLEE_TRUNC_F32,
            Self::TruncF64 => rust_intrinsics::CALLEE_TRUNC_F64,
            Self::RoundF32 => rust_intrinsics::CALLEE_ROUND_F32,
            Self::RoundF64 => rust_intrinsics::CALLEE_ROUND_F64,
            Self::RoundevenF32 => rust_intrinsics::CALLEE_ROUNDEVEN_F32,
            Self::RoundevenF64 => rust_intrinsics::CALLEE_ROUNDEVEN_F64,
            Self::Fabs => rust_intrinsics::CALLEE_FABS,
            Self::CopysignF32 => rust_intrinsics::CALLEE_COPYSIGN_F32,
            Self::CopysignF64 => rust_intrinsics::CALLEE_COPYSIGN_F64,
        }
    }
}

/// Emit a placeholder `mir.call` for a rustc float math intrinsic.
#[allow(clippy::too_many_arguments)]
pub fn emit_rust_float_math_intrinsic(
    ctx: &mut Context,
    body: &mir::Body,
    intrinsic: RustFloatMathIntrinsic,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let return_type = types::translate_type(ctx, &body.locals()[destination.local].ty)?;
    helpers::emit_function_call(
        ctx,
        body,
        intrinsic.placeholder_callee(),
        args,
        destination,
        return_type,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
    )
}
