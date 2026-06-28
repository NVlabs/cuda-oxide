// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Packed `bf16x2` arithmetic operations.
//!
//! Single-thread, non-convergent packed bf16 ALU ops lowered to inline PTX.
//! FMA, min, max, negation, and absolute value require `sm_80+`. Add,
//! subtract, and multiply require `sm_90+`.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Fused multiply-add on packed bf16x2 values: `d = a * b + c`.
///
/// PTX: `fma.rn.bf16x2 $0, $1, $2, $3;`  (requires `sm_80+`)
#[pliron_op(
    name = "nvvm.fma_bf16x2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct FmaBf16x2Op;

impl FmaBf16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        FmaBf16x2Op { op }
    }
}

/// Fused multiply-add with ReLU on packed bf16x2 values: `d = max(0, a * b + c)`.
///
/// PTX: `fma.rn.relu.bf16x2 $0, $1, $2, $3;`  (requires `sm_80+`)
#[pliron_op(
    name = "nvvm.fma_relu_bf16x2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct FmaReluBf16x2Op;

impl FmaReluBf16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        FmaReluBf16x2Op { op }
    }
}

/// Packed bf16x2 addition: `d = a + b`.
///
/// PTX: `add.rn.bf16x2 $0, $1, $2;`  (requires `sm_90+`)
#[pliron_op(
    name = "nvvm.add_bf16x2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct AddBf16x2Op;

impl AddBf16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        AddBf16x2Op { op }
    }
}

/// Packed bf16x2 subtraction: `d = a - b`.
///
/// PTX: `sub.rn.bf16x2 $0, $1, $2;`  (requires `sm_90+`)
#[pliron_op(
    name = "nvvm.sub_bf16x2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct SubBf16x2Op;

impl SubBf16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        SubBf16x2Op { op }
    }
}

/// Packed bf16x2 multiplication: `d = a * b`.
///
/// PTX: `mul.rn.bf16x2 $0, $1, $2;`  (requires `sm_90+`)
#[pliron_op(
    name = "nvvm.mul_bf16x2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct MulBf16x2Op;

impl MulBf16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MulBf16x2Op { op }
    }
}

/// Packed bf16x2 minimum: `d = min(a, b)`.
///
/// PTX: `min.bf16x2 $0, $1, $2;`  (requires `sm_80+`)
#[pliron_op(
    name = "nvvm.min_bf16x2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct MinBf16x2Op;

impl MinBf16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MinBf16x2Op { op }
    }
}

/// Packed bf16x2 maximum: `d = max(a, b)`.
///
/// PTX: `max.bf16x2 $0, $1, $2;`  (requires `sm_80+`)
#[pliron_op(
    name = "nvvm.max_bf16x2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct MaxBf16x2Op;

impl MaxBf16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MaxBf16x2Op { op }
    }
}

/// Packed bf16x2 negation: `d = -a`.
///
/// PTX: `neg.bf16x2 $0, $1;`  (requires `sm_80+`)
#[pliron_op(
    name = "nvvm.neg_bf16x2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct NegBf16x2Op;

impl NegBf16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        NegBf16x2Op { op }
    }
}

/// Packed bf16x2 absolute value: `d = |a|`.
///
/// PTX: `abs.bf16x2 $0, $1;`  (requires `sm_80+`)
#[pliron_op(
    name = "nvvm.abs_bf16x2",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct AbsBf16x2Op;

impl AbsBf16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        AbsBf16x2Op { op }
    }
}

/// Register bf16x2 operations with the context.
pub(super) fn register(ctx: &mut Context) {
    FmaBf16x2Op::register(ctx);
    FmaReluBf16x2Op::register(ctx);
    AddBf16x2Op::register(ctx);
    SubBf16x2Op::register(ctx);
    MulBf16x2Op::register(ctx);
    MinBf16x2Op::register(ctx);
    MaxBf16x2Op::register(ctx);
    NegBf16x2Op::register(ctx);
    AbsBf16x2Op::register(ctx);
}
