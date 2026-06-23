/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! User-authored inline PTX operations.

use pliron::{
    builtin::attributes::{BoolAttr, StringAttr},
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::TypeHandle,
    value::Value,
    verify_err,
};
use pliron_derive::pliron_op;

/// User-authored inline PTX.
///
/// This operation is produced by the MIR importer for `cuda_device::ptx_asm!`
/// marker calls and lowered to LLVM inline assembly.
#[pliron_op(
    name = "nvvm.inline_ptx",
    format,
    attributes = (
        ptx_template: StringAttr,
        ptx_constraints: StringAttr,
        ptx_sideeffect: BoolAttr,
        ptx_convergent: BoolAttr
    )
)]
pub struct InlinePtxOp;

impl InlinePtxOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        InlinePtxOp { op }
    }

    /// Build an inline PTX operation with zero or one result.
    pub fn build(
        ctx: &mut Context,
        result_tys: Vec<TypeHandle>,
        inputs: Vec<Value>,
        template: &str,
        constraints: &str,
        sideeffect: bool,
        convergent: bool,
    ) -> Ptr<Operation> {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            result_tys,
            inputs,
            vec![],
            0,
        );
        let wrapped = InlinePtxOp { op };
        wrapped.set_attr_ptx_template(ctx, StringAttr::new(template.to_string()));
        wrapped.set_attr_ptx_constraints(ctx, StringAttr::new(constraints.to_string()));
        wrapped.set_attr_ptx_sideeffect(ctx, BoolAttr::new(sideeffect));
        wrapped.set_attr_ptx_convergent(ctx, BoolAttr::new(convergent));
        wrapped.get_operation()
    }
}

impl Verify for InlinePtxOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if self.get_attr_ptx_template(ctx).is_none() {
            return verify_err!(op.loc(), "nvvm.inline_ptx requires ptx_template attribute");
        }
        if self.get_attr_ptx_constraints(ctx).is_none() {
            return verify_err!(
                op.loc(),
                "nvvm.inline_ptx requires ptx_constraints attribute"
            );
        }
        if self.get_attr_ptx_sideeffect(ctx).is_none() {
            return verify_err!(
                op.loc(),
                "nvvm.inline_ptx requires ptx_sideeffect attribute"
            );
        }
        if self.get_attr_ptx_convergent(ctx).is_none() {
            return verify_err!(
                op.loc(),
                "nvvm.inline_ptx requires ptx_convergent attribute"
            );
        }
        if op.get_num_results() > 1 {
            return verify_err!(op.loc(), "nvvm.inline_ptx supports at most one result");
        }
        Ok(())
    }
}

/// Register inline PTX operations with the context.
pub(super) fn register(ctx: &mut Context) {
    InlinePtxOp::register(ctx);
}
