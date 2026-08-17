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

use crate::attributes::{CallableKindAttr, PredicateAttr};
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

/// One PTX statement label. Labels remain explicit operations until control
/// flow recovery resolves them to Pliron basic blocks.
#[pliron_op(
    name = "ptx.label",
    format,
    interfaces = [NRegionsInterface<0>, NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (label_name: StringAttr)
)]
pub struct PtxLabelOp;

impl PtxLabelOp {
    pub fn build(ctx: &mut Context, name: &str) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let wrapped = Self { op };
        wrapped.set_attr_label_name(ctx, StringAttr::new(name.to_string()));
        wrapped
    }

    pub fn name(&self, ctx: &Context) -> String {
        self.get_attr_label_name(ctx)
            .expect("verified ptx.label has a name")
            .as_str()
            .to_string()
    }
}

impl Verify for PtxLabelOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let operation = self.get_operation().deref(ctx);
        let Some(name) = self.get_attr_label_name(ctx) else {
            return verify_err!(operation.loc(), "ptx.label requires a name");
        };
        if name.as_str().is_empty() {
            return verify_err!(operation.loc(), "PTX label name must not be empty");
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
///
/// # Header/attribute contract
///
/// The typed attributes (`callable_name`, `callable_kind`,
/// `callable_external`) are the queryable truth; `callable_header` is the
/// print form and still carries syntax the dialect does not model yet
/// (parameter lists, performance directives such as `.maxntid`). The emitter
/// prints only the header, so [`Verify`] re-parses the header with
/// [`ptx_parse`] and rejects the operation whenever the header disagrees with
/// the typed attributes. Mutating one side without the other can therefore
/// never silently desync: it is a verification error, and emission verifies
/// first.
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
        // The header is the print carrier; the typed attributes are the
        // queryable truth. Re-parse the header as a declaration and require
        // agreement so neither side can silently desync from the other.
        let header = self.header(ctx);
        let declaration = format!("{};\n", header.trim());
        let document = match ptx_parse::Document::parse(&declaration) {
            Ok(document) => document,
            Err(error) => {
                return verify_err!(
                    operation.loc(),
                    "ptx.callable header {header:?} does not parse as PTX: {error}"
                );
            }
        };
        let [parsed] = document.callables() else {
            return verify_err!(
                operation.loc(),
                "ptx.callable header {header:?} does not spell exactly one PTX callable"
            );
        };
        if parsed.name() != self.name(ctx) {
            return verify_err!(
                operation.loc(),
                "ptx.callable header names {:?} but callable_name is {:?}",
                parsed.name(),
                self.name(ctx)
            );
        }
        if CallableKindAttr::from(parsed.kind()) != self.kind(ctx) {
            return verify_err!(
                operation.loc(),
                "ptx.callable header spells {:?} but callable_kind is {:?}",
                CallableKindAttr::from(parsed.kind()),
                self.kind(ctx)
            );
        }
        if parsed.is_extern() != self.is_external(ctx) {
            return verify_err!(
                operation.loc(),
                "ptx.callable header spells external = {} but callable_external is {}",
                parsed.is_extern(),
                self.is_external(ctx)
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
///
/// # Header/attribute contract
///
/// `scope_header` is the print form emitted before the opening brace, and is
/// empty for anonymous scopes. PTX callables are the one brace-headed
/// construct this dialect models with typed attributes, so [`Verify`]
/// re-parses a non-empty header with [`ptx_parse`] and rejects callable
/// headers here: routing a callable through `ptx.scope` would bypass
/// `ptx.callable`'s queryable name/kind/external attributes.
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
        let header = self.header(ctx);
        let header = header.trim();
        if header.is_empty() {
            return Ok(());
        }
        if header.ends_with(';') || header.ends_with('{') {
            return verify_err!(
                operation.loc(),
                "ptx.scope header {header:?} must not carry its own terminator; \
                 the emitter prints the brace"
            );
        }
        // A callable header smuggled into a scope would print as a valid
        // definition while bypassing ptx.callable's typed attributes.
        let declaration = format!("{header};\n");
        if let Ok(document) = ptx_parse::Document::parse(&declaration)
            && let [parsed] = document.callables()
        {
            return verify_err!(
                operation.loc(),
                "ptx.scope header spells the PTX callable {:?}; use ptx.callable so the \
                 name, kind, and external flag stay queryable",
                parsed.name()
            );
        }
        Ok(())
    }
}

/// One structurally discovered or directly constructed PTX instruction.
///
/// Predication is the only instruction prefix PTX defines, so the guard is a
/// typed, optional [`PredicateAttr`] rather than free-form prefix text. The
/// emitter derives the `@%p` / `@!%p` spelling from it.
#[pliron_op(
    name = "ptx.instruction",
    format,
    interfaces = [NRegionsInterface<0>, NOpdsInterface<0>, NResultsInterface<0>],
    attributes = (
        instruction_predicate: PredicateAttr,
        instruction_head: StringAttr,
        instruction_operands: VecAttr
    )
)]
pub struct PtxInstructionOp;

impl PtxInstructionOp {
    pub fn build<'operand>(
        ctx: &mut Context,
        predicate: Option<PredicateAttr>,
        head: &str,
        operands: impl IntoIterator<Item = &'operand str>,
    ) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let wrapped = Self { op };
        if let Some(predicate) = predicate {
            wrapped.set_attr_instruction_predicate(ctx, predicate);
        }
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

    pub fn predicate(&self, ctx: &Context) -> Option<PredicateAttr> {
        self.get_attr_instruction_predicate(ctx)
            .map(|predicate| predicate.clone())
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
    PtxLabelOp::register(ctx);
    PtxCallableOp::register(ctx);
    PtxScopeOp::register(ctx);
    PtxInstructionOp::register(ctx);
    PtxRawOp::register(ctx);
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::builtin::attributes::StringAttr;

    fn test_context() -> Context {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        ctx
    }

    #[test]
    fn callable_with_agreeing_header_and_attributes_verifies() {
        let mut ctx = test_context();
        let callable = PtxCallableOp::build_definition(
            &mut ctx,
            "kernel",
            CallableKindAttr::Entry,
            false,
            ".visible .entry kernel(.param .u64 p0)",
        );
        callable.verify(&ctx).unwrap();
    }

    #[test]
    fn callable_name_mutated_without_header_fails_verification() {
        let mut ctx = test_context();
        let callable = PtxCallableOp::build_definition(
            &mut ctx,
            "kernel",
            CallableKindAttr::Entry,
            false,
            ".visible .entry kernel()",
        );
        callable.set_attr_callable_name(&ctx, StringAttr::new("renamed".to_string()));
        let error = callable.verify(&ctx).unwrap_err().to_string();
        assert!(
            error.contains("header names \"kernel\" but callable_name is \"renamed\""),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn callable_kind_desync_fails_verification() {
        let mut ctx = test_context();
        let callable = PtxCallableOp::build_declaration(
            &mut ctx,
            "helper",
            CallableKindAttr::Entry,
            true,
            ".extern .func helper(.param .b32 x)",
        );
        let error = callable.verify(&ctx).unwrap_err().to_string();
        assert!(
            error.contains("header spells Function but callable_kind is Entry"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn callable_external_desync_fails_verification() {
        let mut ctx = test_context();
        let callable = PtxCallableOp::build_declaration(
            &mut ctx,
            "helper",
            CallableKindAttr::Function,
            false,
            ".extern .func helper(.param .b32 x)",
        );
        let error = callable.verify(&ctx).unwrap_err().to_string();
        assert!(
            error.contains("header spells external = true but callable_external is false"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn callable_header_that_is_not_a_callable_fails_verification() {
        let mut ctx = test_context();
        let callable = PtxCallableOp::build_declaration(
            &mut ctx,
            "kernel",
            CallableKindAttr::Entry,
            false,
            ".pragma \"not a callable\"",
        );
        let error = callable.verify(&ctx).unwrap_err().to_string();
        assert!(
            error.contains("does not spell exactly one PTX callable"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn anonymous_and_plain_scope_headers_verify() {
        let mut ctx = test_context();
        PtxScopeOp::build(&mut ctx, "").verify(&ctx).unwrap();
    }

    #[test]
    fn scope_smuggling_a_callable_header_fails_verification() {
        let mut ctx = test_context();
        let scope = PtxScopeOp::build(&mut ctx, ".visible .entry kernel()");
        let error = scope.verify(&ctx).unwrap_err().to_string();
        assert!(
            error.contains("spells the PTX callable \"kernel\""),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn unpredicated_instruction_verifies_without_a_predicate_attribute() {
        let mut ctx = test_context();
        let instruction = PtxInstructionOp::build(&mut ctx, None, "ret", []);
        instruction.verify(&ctx).unwrap();
        assert_eq!(instruction.predicate(&ctx), None);
    }

    #[test]
    fn predicate_register_must_be_percent_prefixed() {
        let mut ctx = test_context();
        let instruction = PtxInstructionOp::build(
            &mut ctx,
            Some(PredicateAttr::new("p1", false)),
            "bra",
            ["L0"],
        );
        let error = instruction
            .get_operation()
            .deref(&ctx)
            .verify(&ctx)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("must be a %-prefixed register name"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn scope_header_carrying_its_own_brace_fails_verification() {
        let mut ctx = test_context();
        let scope = PtxScopeOp::build(&mut ctx, ".pragma \"x\" {");
        let error = scope.verify(&ctx).unwrap_err().to_string();
        assert!(
            error.contains("must not carry its own terminator"),
            "unexpected error: {error}"
        );
    }
}
