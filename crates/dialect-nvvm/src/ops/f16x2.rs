// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Packed `f16x2` arithmetic operations.
//!
//! Single-thread, non-convergent packed f16 ALU ops lowered to inline PTX.
//! Add, subtract, multiply, FMA, negation, and absolute value require
//! `sm_53+`. Min, max, and fused multiply-add with ReLU require `sm_80+`.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    builtin::types::IntegerType,
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

/// Verify that all operands and the single result of an f16x2 operation are
/// 32-bit integers (the packed representation of two f16 values).
fn verify_f16x2_op(
    ctx: &Context,
    op: &Operation,
    op_name: &str,
    expected_operands: usize,
) -> Result<(), Error> {
    let operands: Vec<_> = op.operands().collect();
    if operands.len() != expected_operands {
        return verify_err!(
            op.loc(),
            "{} requires {} operand(s), got {}",
            op_name,
            expected_operands,
            operands.len()
        );
    }

    for (i, operand) in operands.iter().enumerate() {
        let ty = operand.get_type(ctx);
        let ty_ref = ty.deref(ctx);
        let Some(integer) = ty_ref.downcast_ref::<IntegerType>() else {
            return verify_err!(
                op.loc(),
                "{} operand {} must be a 32-bit integer (packed f16x2)",
                op_name,
                i
            );
        };
        if integer.width() != 32 {
            return verify_err!(
                op.loc(),
                "{} operand {} must be a 32-bit integer (packed f16x2)",
                op_name,
                i
            );
        }
    }

    if op.get_num_results() != 1 {
        return verify_err!(
            op.loc(),
            "{} requires 1 result, got {}",
            op_name,
            op.get_num_results()
        );
    }

    let res_ty = op.get_result(0).get_type(ctx);
    let res_ref = res_ty.deref(ctx);
    let Some(integer) = res_ref.downcast_ref::<IntegerType>() else {
        return verify_err!(
            op.loc(),
            "{} result must be a 32-bit integer (packed f16x2)",
            op_name
        );
    };
    if integer.width() != 32 {
        return verify_err!(
            op.loc(),
            "{} result must be a 32-bit integer (packed f16x2)",
            op_name
        );
    }

    Ok(())
}

/// Packed f16x2 addition: `d = a + b`.
///
/// PTX: `add.rn.f16x2 $0, $1, $2;`  (requires `sm_53+`)
#[pliron_op(
    name = "nvvm.add_f16x2",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct AddF16x2Op;

impl AddF16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        AddF16x2Op { op }
    }
}

impl Verify for AddF16x2Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        verify_f16x2_op(ctx, &op, "nvvm.add_f16x2", 2)
    }
}

/// Packed f16x2 subtraction: `d = a - b`.
///
/// PTX: `sub.rn.f16x2 $0, $1, $2;`  (requires `sm_53+`)
#[pliron_op(
    name = "nvvm.sub_f16x2",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct SubF16x2Op;

impl SubF16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        SubF16x2Op { op }
    }
}

impl Verify for SubF16x2Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        verify_f16x2_op(ctx, &op, "nvvm.sub_f16x2", 2)
    }
}

/// Packed f16x2 multiplication: `d = a * b`.
///
/// PTX: `mul.rn.f16x2 $0, $1, $2;`  (requires `sm_53+`)
#[pliron_op(
    name = "nvvm.mul_f16x2",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct MulF16x2Op;

impl MulF16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MulF16x2Op { op }
    }
}

impl Verify for MulF16x2Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        verify_f16x2_op(ctx, &op, "nvvm.mul_f16x2", 2)
    }
}

/// Fused multiply-add on packed f16x2 values: `d = a * b + c`.
///
/// PTX: `fma.rn.f16x2 $0, $1, $2, $3;`  (requires `sm_53+`)
#[pliron_op(
    name = "nvvm.fma_f16x2",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct FmaF16x2Op;

impl FmaF16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        FmaF16x2Op { op }
    }
}

impl Verify for FmaF16x2Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        verify_f16x2_op(ctx, &op, "nvvm.fma_f16x2", 3)
    }
}

/// Packed f16x2 negation: `d = -a`.
///
/// PTX: `neg.f16x2 $0, $1;`  (requires `sm_53+`)
#[pliron_op(
    name = "nvvm.neg_f16x2",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct NegF16x2Op;

impl NegF16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        NegF16x2Op { op }
    }
}

impl Verify for NegF16x2Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        verify_f16x2_op(ctx, &op, "nvvm.neg_f16x2", 1)
    }
}

/// Packed f16x2 absolute value: `d = |a|`.
///
/// PTX: `abs.f16x2 $0, $1;`  (requires `sm_53+`)
#[pliron_op(
    name = "nvvm.abs_f16x2",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct AbsF16x2Op;

impl AbsF16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        AbsF16x2Op { op }
    }
}

impl Verify for AbsF16x2Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        verify_f16x2_op(ctx, &op, "nvvm.abs_f16x2", 1)
    }
}

/// Packed f16x2 minimum: `d = min(a, b)`.
///
/// PTX: `min.f16x2 $0, $1, $2;`  (requires `sm_80+`)
#[pliron_op(
    name = "nvvm.min_f16x2",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct MinF16x2Op;

impl MinF16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MinF16x2Op { op }
    }
}

impl Verify for MinF16x2Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        verify_f16x2_op(ctx, &op, "nvvm.min_f16x2", 2)
    }
}

/// Packed f16x2 maximum: `d = max(a, b)`.
///
/// PTX: `max.f16x2 $0, $1, $2;`  (requires `sm_80+`)
#[pliron_op(
    name = "nvvm.max_f16x2",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct MaxF16x2Op;

impl MaxF16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MaxF16x2Op { op }
    }
}

impl Verify for MaxF16x2Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        verify_f16x2_op(ctx, &op, "nvvm.max_f16x2", 2)
    }
}

/// Fused multiply-add with ReLU on packed f16x2 values: `d = max(0, a * b + c)`.
///
/// PTX: `fma.rn.relu.f16x2 $0, $1, $2, $3;`  (requires `sm_80+`)
#[pliron_op(
    name = "nvvm.fma_relu_f16x2",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct FmaReluF16x2Op;

impl FmaReluF16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        FmaReluF16x2Op { op }
    }
}

impl Verify for FmaReluF16x2Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        verify_f16x2_op(ctx, &op, "nvvm.fma_relu_f16x2", 3)
    }
}

/// Register f16x2 operations with the context.
pub(super) fn register(ctx: &mut Context) {
    AddF16x2Op::register(ctx);
    SubF16x2Op::register(ctx);
    MulF16x2Op::register(ctx);
    FmaF16x2Op::register(ctx);
    NegF16x2Op::register(ctx);
    AbsF16x2Op::register(ctx);
    MinF16x2Op::register(ctx);
    MaxF16x2Op::register(ctx);
    FmaReluF16x2Op::register(ctx);
}
