/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Type conversion operations.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Convert two f32 values to packed f16x2 (u32).
///
/// Maps to PTX: `cvt.rn.f16x2.f32 d, hi, lo;`
///
/// # Operands
///
/// - `lo` (f32): value for bits `[15:0]`
/// - `hi` (f32): value for bits `[31:16]`
///
/// # Results
///
/// - packed f16x2 as u32
#[pliron_op(
    name = "nvvm.cvt_f16x2_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct CvtF16x2F32Op;

impl CvtF16x2F32Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        CvtF16x2F32Op { op }
    }
}

/// Convert two f32 values to packed f16x2 (u32) with truncation rounding.
///
/// Maps to PTX: `cvt.rz.f16x2.f32 d, hi, lo;`
///
/// # Operands
///
/// - `lo` (f32): value for bits `[15:0]`
/// - `hi` (f32): value for bits `[31:16]`
///
/// # Results
///
/// - packed f16x2 as u32
#[pliron_op(
    name = "nvvm.cvt_rz_f16x2_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct CvtRzF16x2F32Op;

impl CvtRzF16x2F32Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        CvtRzF16x2F32Op { op }
    }
}

/// Convert two f32 values to packed f16x2 (u32) with fused ReLU.
///
/// Maps to PTX: `cvt.rn.relu.f16x2.f32 d, hi, lo;`
///
/// # Operands
///
/// - `lo` (f32): value for bits `[15:0]`
/// - `hi` (f32): value for bits `[31:16]`
///
/// # Results
///
/// - packed f16x2 as u32 (ReLU applied)
#[pliron_op(
    name = "nvvm.cvt_rn_relu_f16x2_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct CvtRnReluF16x2F32Op;

impl CvtRnReluF16x2F32Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        CvtRnReluF16x2F32Op { op }
    }
}

/// Convert two f32 values to packed bf16x2 (u32) with fused ReLU.
///
/// Maps to PTX: `cvt.rn.relu.bf16x2.f32 d, hi, lo;`
///
/// # Operands
///
/// - `lo` (f32): value for bits `[15:0]`
/// - `hi` (f32): value for bits `[31:16]`
///
/// # Results
///
/// - packed bf16x2 as u32 (ReLU applied)
#[pliron_op(
    name = "nvvm.cvt_rn_relu_bf16x2_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct CvtRnReluBf16x2F32Op;

impl CvtRnReluBf16x2F32Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        CvtRnReluBf16x2F32Op { op }
    }
}

/// Convert two f32 values to packed bf16x2 (u32) with truncation rounding.
///
/// Maps to PTX: `cvt.rz.bf16x2.f32 d, hi, lo;`
///
/// # Operands
///
/// - `lo` (f32): value for bits `[15:0]`
/// - `hi` (f32): value for bits `[31:16]`
///
/// # Results
///
/// - packed bf16x2 as u32
#[pliron_op(
    name = "nvvm.cvt_rz_bf16x2_f32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct CvtRzBf16x2F32Op;

impl CvtRzBf16x2F32Op {
    pub fn new(op: Ptr<Operation>) -> Self {
        CvtRzBf16x2F32Op { op }
    }
}

/// Register convert operations with the context.
pub(super) fn register(ctx: &mut Context) {
    CvtF16x2F32Op::register(ctx);
    CvtRzF16x2F32Op::register(ctx);
    CvtRnReluF16x2F32Op::register(ctx);
    CvtRnReluBf16x2F32Op::register(ctx);
    CvtRzBf16x2F32Op::register(ctx);
}
