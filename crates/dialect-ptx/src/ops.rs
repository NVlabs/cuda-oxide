/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Structured PTX operations.
//!
//! These operations can be created either by projecting a lossless syntax
//! document or directly by a producer. Source locations intentionally live in
//! [`crate::Projection`]'s lineage table rather than in the operations, so a
//! freshly-built module does not need to invent source spans.

use crate::attributes::CallableKindAttr;
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{BoolAttr, StringAttr, VecAttr},
        op_interfaces::{
            NOpdsInterface, NRegionsInterface, NResultsInterface, NoTerminatorInterface,
            OneRegionInterface, SingleBlockRegionInterface,
        },
    },
    common_traits::Verify,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    location::Located,
    op::Op,
    operation::Operation,
    region::Region,
    result::Error,
    verify_err,
};
use pliron_derive::pliron_op;

/// Root of one structured PTX module.
#[pliron_op(
    name = "ptx.module",
    format,
    interfaces = [
        NRegionsInterface<1>,
        OneRegionInterface,
        SingleBlockRegionInterface,
        NoTerminatorInterface,
        NOpdsInterface<0>,
        NResultsInterface<0>
    ]
)]
pub struct PtxModuleOp;

impl PtxModuleOp {
    pub fn build(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 1);
        let region = op.deref(ctx).get_region(0);
        BasicBlock::new(ctx, None, vec![]).insert_at_back(region, ctx);
        Self { op }
    }

    pub fn body(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_operation()
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_head()
            .expect("ptx.module always has a body block")
    }
}

impl Verify for PtxModuleOp {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        Ok(())
    }
}

/// One PTX directive at module, callable, or lexical-scope level.
#[pliron_op(
    name = "ptx.directive",
    format,
    interfaces = [NRegionsInterface<0>, NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (
        directive_name: StringAttr,
        directive_arguments: StringAttr
    )
)]
pub struct PtxDirectiveOp;

impl PtxDirectiveOp {
    pub fn build(ctx: &mut Context, name: &str, arguments: &str) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let wrapped = Self { op };
        wrapped.set_attr_directive_name(ctx, StringAttr::new(name.to_string()));
        wrapped.set_attr_directive_arguments(ctx, StringAttr::new(arguments.to_string()));
        wrapped
    }

    pub fn name(&self, ctx: &Context) -> String {
        self.get_attr_directive_name(ctx)
            .expect("verified ptx.directive has a name")
            .as_str()
            .to_string()
    }

    pub fn arguments(&self, ctx: &Context) -> String {
        self.get_attr_directive_arguments(ctx)
            .expect("verified ptx.directive has arguments")
            .as_str()
            .to_string()
    }
}

impl Verify for PtxDirectiveOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let operation = self.get_operation().deref(ctx);
        let Some(name) = self.get_attr_directive_name(ctx) else {
            return verify_err!(operation.loc(), "ptx.directive requires a name");
        };
        if !name.as_str().starts_with('.') {
            return verify_err!(operation.loc(), "PTX directive name must start with '.'");
        }
        if self.get_attr_directive_arguments(ctx).is_none() {
            return verify_err!(operation.loc(), "ptx.directive requires arguments");
        }
        Ok(())
    }
}

/// A declaration or definition of a PTX `.entry` or `.func`.
///
/// Declarations have no regions. Definitions own exactly one region containing
/// one or more PTX basic blocks. `header` is the complete spelling before the
/// declaration semicolon or definition opening brace; consumers can gradually
/// replace its generic pieces with typed attributes without losing syntax.
#[pliron_op(
    name = "ptx.callable",
    format,
    interfaces = [
        SingleBlockRegionInterface,
        NoTerminatorInterface,
        NOpdsInterface<0>,
        NResultsInterface<0>
    ],
    attributes = (
        callable_name: StringAttr,
        callable_kind: CallableKindAttr,
        callable_external: BoolAttr,
        callable_header: StringAttr
    )
)]
pub struct PtxCallableOp;

impl PtxCallableOp {
    pub fn build_declaration(
        ctx: &mut Context,
        name: &str,
        kind: CallableKindAttr,
        is_extern: bool,
        header: &str,
    ) -> Self {
        Self::build(ctx, name, kind, is_extern, header, false)
    }

    pub fn build_definition(
        ctx: &mut Context,
        name: &str,
        kind: CallableKindAttr,
        is_extern: bool,
        header: &str,
    ) -> Self {
        Self::build(ctx, name, kind, is_extern, header, true)
    }

    fn build(
        ctx: &mut Context,
        name: &str,
        kind: CallableKindAttr,
        is_extern: bool,
        header: &str,
        has_body: bool,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            usize::from(has_body),
        );
        let wrapped = Self { op };
        wrapped.set_attr_callable_name(ctx, StringAttr::new(name.to_string()));
        wrapped.set_attr_callable_kind(ctx, kind);
        wrapped.set_attr_callable_external(ctx, BoolAttr::new(is_extern));
        wrapped.set_attr_callable_header(ctx, StringAttr::new(header.to_string()));
        if has_body {
            let region = op.deref(ctx).get_region(0);
            BasicBlock::new(ctx, None, vec![]).insert_at_back(region, ctx);
        }
        wrapped
    }

    pub fn name(&self, ctx: &Context) -> String {
        self.get_attr_callable_name(ctx)
            .expect("verified ptx.callable has a name")
            .as_str()
            .to_string()
    }

    pub fn kind(&self, ctx: &Context) -> CallableKindAttr {
        *self
            .get_attr_callable_kind(ctx)
            .expect("verified ptx.callable has a kind")
    }

    pub fn is_external(&self, ctx: &Context) -> bool {
        bool::from(
            self.get_attr_callable_external(ctx)
                .expect("verified ptx.callable has an external flag")
                .clone(),
        )
    }

    pub fn header(&self, ctx: &Context) -> String {
        self.get_attr_callable_header(ctx)
            .expect("verified ptx.callable has a header")
            .as_str()
            .to_string()
    }

    pub fn region(&self, ctx: &Context) -> Option<Ptr<Region>> {
        (self.get_operation().deref(ctx).num_regions() == 1)
            .then(|| self.get_operation().deref(ctx).get_region(0))
    }

    pub fn entry_block(&self, ctx: &Context) -> Option<Ptr<BasicBlock>> {
        self.region(ctx)
            .and_then(|region| region.deref(ctx).get_entry_block())
    }

    pub fn is_definition(&self, ctx: &Context) -> bool {
        self.region(ctx).is_some()
    }
}

impl Verify for PtxCallableOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let operation = self.get_operation().deref(ctx);
        if self.get_attr_callable_name(ctx).is_none()
            || self.get_attr_callable_kind(ctx).is_none()
            || self.get_attr_callable_external(ctx).is_none()
            || self.get_attr_callable_header(ctx).is_none()
        {
            return verify_err!(
                operation.loc(),
                "ptx.callable requires name, kind, external flag, and header"
            );
        }
        if operation.num_regions() > 1 {
            return verify_err!(
                operation.loc(),
                "ptx.callable supports at most one body region"
            );
        }
        if let Some(region) = self.region(ctx)
            && region.deref(ctx).get_entry_block().is_none()
        {
            return verify_err!(
                operation.loc(),
                "PTX callable definition requires a body block"
            );
        }
        Ok(())
    }
}

/// An anonymous or header-owned lexical PTX scope.
#[pliron_op(
    name = "ptx.scope",
    format,
    interfaces = [
        NRegionsInterface<1>,
        OneRegionInterface,
        SingleBlockRegionInterface,
        NoTerminatorInterface,
        NOpdsInterface<0>,
        NResultsInterface<0>
    ],
    attributes = (scope_header: StringAttr)
)]
pub struct PtxScopeOp;

impl PtxScopeOp {
    pub fn build(ctx: &mut Context, header: &str) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 1);
        let wrapped = Self { op };
        wrapped.set_attr_scope_header(ctx, StringAttr::new(header.to_string()));
        let region = op.deref(ctx).get_region(0);
        BasicBlock::new(ctx, None, vec![]).insert_at_back(region, ctx);
        wrapped
    }

    pub fn header(&self, ctx: &Context) -> String {
        self.get_attr_scope_header(ctx)
            .expect("verified ptx.scope has a header")
            .as_str()
            .to_string()
    }

    pub fn body(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_operation()
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_entry_block()
            .expect("ptx.scope always has a body block")
    }
}

impl Verify for PtxScopeOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let operation = self.get_operation().deref(ctx);
        if self.get_attr_scope_header(ctx).is_none() {
            return verify_err!(operation.loc(), "ptx.scope requires a header attribute");
        }
        Ok(())
    }
}

/// One structurally discovered or directly constructed PTX instruction.
#[pliron_op(
    name = "ptx.instruction",
    format,
    interfaces = [NRegionsInterface<0>, NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (
        instruction_prefix: StringAttr,
        instruction_head: StringAttr,
        instruction_operands: VecAttr
    )
)]
pub struct PtxInstructionOp;

impl PtxInstructionOp {
    pub fn build<'operand>(
        ctx: &mut Context,
        prefix: &str,
        head: &str,
        operands: impl IntoIterator<Item = &'operand str>,
    ) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let wrapped = Self { op };
        wrapped.set_attr_instruction_prefix(ctx, StringAttr::new(prefix.to_string()));
        wrapped.set_attr_instruction_head(ctx, StringAttr::new(head.to_string()));
        wrapped.set_attr_instruction_operands(
            ctx,
            VecAttr::new(
                operands
                    .into_iter()
                    .map(|operand| StringAttr::new(operand.to_string()).into())
                    .collect(),
            ),
        );
        wrapped
    }

    pub fn prefix(&self, ctx: &Context) -> String {
        self.get_attr_instruction_prefix(ctx)
            .expect("verified ptx.instruction has a prefix")
            .as_str()
            .to_string()
    }

    pub fn head(&self, ctx: &Context) -> String {
        self.get_attr_instruction_head(ctx)
            .expect("verified ptx.instruction has a head")
            .as_str()
            .to_string()
    }

    pub fn operands(&self, ctx: &Context) -> Vec<String> {
        self.get_attr_instruction_operands(ctx)
            .expect("verified ptx.instruction has operands")
            .0
            .iter()
            .map(|operand| {
                operand
                    .downcast_ref::<StringAttr>()
                    .expect("verified PTX operands are strings")
                    .as_str()
                    .to_string()
            })
            .collect()
    }
}

impl Verify for PtxInstructionOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let operation = self.get_operation().deref(ctx);
        if self.get_attr_instruction_prefix(ctx).is_none() {
            return verify_err!(operation.loc(), "ptx.instruction requires a prefix");
        }
        let Some(head) = self.get_attr_instruction_head(ctx) else {
            return verify_err!(operation.loc(), "ptx.instruction requires a head");
        };
        if head.as_str().is_empty() {
            return verify_err!(operation.loc(), "PTX instruction head must not be empty");
        }
        let Some(operands) = self.get_attr_instruction_operands(ctx) else {
            return verify_err!(operation.loc(), "ptx.instruction requires operands");
        };
        if operands
            .0
            .iter()
            .any(|operand| operand.downcast_ref::<StringAttr>().is_none())
        {
            return verify_err!(operation.loc(), "PTX instruction operands must be strings");
        }
        Ok(())
    }
}

/// A structurally retained statement for syntax not yet modeled by this dialect.
#[pliron_op(
    name = "ptx.raw",
    format,
    interfaces = [NRegionsInterface<0>, NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (raw_text: StringAttr)
)]
pub struct PtxRawOp;

impl PtxRawOp {
    pub fn build(ctx: &mut Context, text: &str) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let wrapped = Self { op };
        wrapped.set_attr_raw_text(ctx, StringAttr::new(text.to_string()));
        wrapped
    }

    pub fn text(&self, ctx: &Context) -> String {
        self.get_attr_raw_text(ctx)
            .expect("verified ptx.raw has text")
            .as_str()
            .to_string()
    }
}

impl Verify for PtxRawOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let operation = self.get_operation().deref(ctx);
        if self.get_attr_raw_text(ctx).is_none() {
            return verify_err!(operation.loc(), "ptx.raw requires text");
        }
        Ok(())
    }
}

pub fn register(ctx: &mut Context) {
    PtxModuleOp::register(ctx);
    PtxDirectiveOp::register(ctx);
    PtxCallableOp::register(ctx);
    PtxScopeOp::register(ctx);
    PtxInstructionOp::register(ctx);
    PtxRawOp::register(ctx);
}
