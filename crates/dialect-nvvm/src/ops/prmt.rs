// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte permute operation (`prmt.b32`).
//!
//! A single-thread, non-convergent byte permute instruction lowered to
//! inline PTX. Available on all architectures (SM 1.0+, PTX ISA 1.0+).

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

/// Byte permute: select 4 bytes from the concatenation of `a` and `b`.
///
/// Each nibble of `control` (bits 3:0, 7:4, 11:8, 15:12) selects one byte
/// from the 8-byte value `{b, a}` (byte 0 = LSB of `a`, byte 7 = MSB of `b`).
///
/// PTX: `prmt.b32 $0, $1, $2, $3;` (available on all architectures)
///
/// # Operands
///
/// - `a` (u32): first source word
/// - `b` (u32): second source word
/// - `control` (u32): byte selector
///
/// # Results
///
/// - `d` (u32): permuted result
#[pliron_op(
    name = "nvvm.prmt",
    format,
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct PrmtOp;

impl PrmtOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        PrmtOp { op }
    }
}

impl Verify for PrmtOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);

        if op.operands().count() != 3 || op.get_num_results() != 1 {
            return verify_err!(op.loc(), "nvvm.prmt requires three operands and one result");
        }

        for (name, ty) in [
            ("operand 0", op.get_operand(0).get_type(ctx)),
            ("operand 1", op.get_operand(1).get_type(ctx)),
            ("operand 2", op.get_operand(2).get_type(ctx)),
            ("result", op.get_result(0).get_type(ctx)),
        ] {
            let ty_ref = ty.deref(ctx);
            let Some(integer) = ty_ref.downcast_ref::<IntegerType>() else {
                return verify_err!(op.loc(), "nvvm.prmt {} must be a 32-bit integer", name);
            };
            if integer.width() != 32 {
                return verify_err!(op.loc(), "nvvm.prmt {} must be a 32-bit integer", name);
            }
        }

        Ok(())
    }
}

/// Register the prmt operation with the context.
pub(super) fn register(ctx: &mut Context) {
    PrmtOp::register(ctx);
}
