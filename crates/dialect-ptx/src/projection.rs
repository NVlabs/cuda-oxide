/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::attributes::CallableKindAttr;
use crate::ops::{
    PtxCallableOp, PtxDirectiveOp, PtxInstructionOp, PtxLabelOp, PtxModuleOp, PtxRawOp, PtxScopeOp,
};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::op::Op;
use pliron::operation::Operation;
use ptx_parse::{
    Callable, CallableKind, Directive, Document, Instruction, Label, LabelId, ParseError, ScopeId,
    StatementId, StatementKind,
};
use std::collections::HashMap;
use std::ops::Range;

/// The authoritative syntax node from which a projected entity was built.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SourceNode {
    Label { label: LabelId },
    Statement { statement: StatementId },
    Scope { scope: ScopeId },
}

impl SourceNode {
    pub fn statement(self) -> Option<StatementId> {
        match self {
            Self::Label { .. } => None,
            Self::Statement { statement } => Some(statement),
            Self::Scope { .. } => None,
        }
    }

    pub fn scope(self) -> Option<ScopeId> {
        match self {
            Self::Label { .. } => None,
            Self::Statement { .. } => None,
            Self::Scope { scope } => Some(scope),
        }
    }

    pub fn label(self) -> Option<LabelId> {
        match self {
            Self::Label { label } => Some(label),
            Self::Statement { .. } | Self::Scope { .. } => None,
        }
    }
}

/// One Pliron operation and its immutable source lineage.
#[derive(Clone, Debug)]
pub struct ProjectedNode {
    operation: Ptr<Operation>,
    source_node: SourceNode,
    source_span: Range<usize>,
}

impl ProjectedNode {
    pub fn operation(&self) -> Ptr<Operation> {
        self.operation
    }

    pub fn source_node(&self) -> SourceNode {
        self.source_node
    }

    pub fn source_span(&self) -> Range<usize> {
        self.source_span.clone()
    }
}

/// One Pliron basic block and the lexical scope it represents.
#[derive(Clone, Debug)]
pub struct ProjectedBlock {
    block: Ptr<BasicBlock>,
    source_scope: ScopeId,
    source_span: Range<usize>,
}

impl ProjectedBlock {
    pub fn block(&self) -> Ptr<BasicBlock> {
        self.block
    }

    pub fn source_scope(&self) -> ScopeId {
        self.source_scope
    }

    pub fn source_span(&self) -> Range<usize> {
        self.source_span.clone()
    }
}

/// A lossless syntax document paired with a structured, independently
/// emittable Pliron PTX module.
pub struct Projection<'source> {
    document: Document<'source>,
    module: PtxModuleOp,
    nodes: Vec<ProjectedNode>,
    nodes_by_operation: HashMap<Ptr<Operation>, usize>,
    blocks: Vec<ProjectedBlock>,
    blocks_by_pointer: HashMap<Ptr<BasicBlock>, usize>,
}

impl<'source> Projection<'source> {
    /// Parse PTX and project its structural statements and lexical scopes.
    ///
    /// The caller must register this dialect in `ctx` before parsing, matching
    /// the lifecycle of CUDA Oxide's other Pliron dialects.
    pub fn parse(ctx: &mut Context, source: &'source str) -> Result<Self, ParseError> {
        let document = Document::parse(source)?;
        Ok(Self::from_document(ctx, document))
    }

    /// Project an already-parsed document. Source text remains authoritative
    /// for lossless edits; the produced operation tree is authoritative for
    /// structured analysis, construction, and canonical emission.
    pub fn from_document(ctx: &mut Context, document: Document<'source>) -> Self {
        let module = PtxModuleOp::build(ctx);
        let root_block = module.body(ctx);
        let (nodes, blocks) = {
            let mut projector = Projector::new(ctx, &document);
            projector.record_block(root_block, ScopeId::ROOT);
            projector.project_scope(ScopeId::ROOT, root_block);
            (projector.nodes, projector.blocks)
        };
        let nodes_by_operation = nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.operation, index))
            .collect();
        let blocks_by_pointer = blocks
            .iter()
            .enumerate()
            .map(|(index, block)| (block.block, index))
            .collect();

        Self {
            document,
            module,
            nodes,
            nodes_by_operation,
            blocks,
            blocks_by_pointer,
        }
    }

    pub fn document(&self) -> &Document<'source> {
        &self.document
    }

    pub fn module(&self) -> PtxModuleOp {
        self.module
    }

    pub fn nodes(&self) -> &[ProjectedNode] {
        &self.nodes
    }

    pub fn blocks(&self) -> &[ProjectedBlock] {
        &self.blocks
    }

    pub fn source_node(&self, operation: Ptr<Operation>) -> Option<SourceNode> {
        self.nodes_by_operation
            .get(&operation)
            .map(|index| self.nodes[*index].source_node)
    }

    pub fn source_scope(&self, block: Ptr<BasicBlock>) -> Option<ScopeId> {
        self.blocks_by_pointer
            .get(&block)
            .map(|index| self.blocks[*index].source_scope)
    }
}

struct Projector<'ctx, 'document, 'source> {
    ctx: &'ctx mut Context,
    document: &'document Document<'source>,
    statements_by_scope: HashMap<ScopeId, Vec<StatementId>>,
    scopes_by_parent: HashMap<ScopeId, Vec<ScopeId>>,
    directives: HashMap<StatementId, &'document Directive<'source>>,
    callables: HashMap<StatementId, &'document Callable<'source>>,
    instructions: HashMap<StatementId, &'document Instruction<'source>>,
    labels: HashMap<StatementId, Vec<&'document Label<'source>>>,
    nodes: Vec<ProjectedNode>,
    blocks: Vec<ProjectedBlock>,
}

impl<'ctx, 'document, 'source> Projector<'ctx, 'document, 'source> {
    fn new(ctx: &'ctx mut Context, document: &'document Document<'source>) -> Self {
        let mut statements_by_scope: HashMap<ScopeId, Vec<StatementId>> = HashMap::new();
        for statement in document.statements() {
            statements_by_scope
                .entry(statement.scope())
                .or_default()
                .push(statement.id());
        }
        let mut scopes_by_parent: HashMap<ScopeId, Vec<ScopeId>> = HashMap::new();
        for scope in document.scopes().iter().skip(1) {
            if let Some(parent) = scope.parent() {
                scopes_by_parent.entry(parent).or_default().push(scope.id());
            }
        }
        let directives = document
            .directives()
            .iter()
            .map(|directive| (directive.statement(), directive))
            .collect();
        let callables = document
            .callables()
            .iter()
            .map(|callable| (callable.statement(), callable))
            .collect();
        let instructions = document
            .instructions()
            .iter()
            .map(|instruction| (instruction.statement(), instruction))
            .collect();
        let mut labels: HashMap<StatementId, Vec<&Label<'source>>> = HashMap::new();
        for label in document.labels() {
            labels.entry(label.statement()).or_default().push(label);
        }
        Self {
            ctx,
            document,
            statements_by_scope,
            scopes_by_parent,
            directives,
            callables,
            instructions,
            labels,
            nodes: Vec::new(),
            blocks: Vec::new(),
        }
    }

    fn record_block(&mut self, block: Ptr<BasicBlock>, scope: ScopeId) {
        let source_span = self
            .document
            .scope(scope)
            .expect("projected scope belongs to the document")
            .body_span();
        self.blocks.push(ProjectedBlock {
            block,
            source_scope: scope,
            source_span,
        });
    }

    fn record_operation(
        &mut self,
        operation: Ptr<Operation>,
        source_node: SourceNode,
        source_span: Range<usize>,
        destination: Ptr<BasicBlock>,
    ) {
        operation.insert_at_back(destination, self.ctx);
        self.nodes.push(ProjectedNode {
            operation,
            source_node,
            source_span,
        });
    }

    fn project_scope(&mut self, scope: ScopeId, destination: Ptr<BasicBlock>) {
        #[derive(Clone, Copy)]
        enum Event {
            Statement(StatementId),
            AnonymousScope(ScopeId),
        }

        let child_scopes = self
            .scopes_by_parent
            .get(&scope)
            .cloned()
            .unwrap_or_default();
        let scopes_by_header: HashMap<StatementId, ScopeId> = child_scopes
            .iter()
            .filter_map(|scope| {
                self.document
                    .scope(*scope)
                    .and_then(|scope_node| scope_node.header().map(|header| (header, *scope)))
            })
            .collect();
        let mut events: Vec<(usize, Event)> = self
            .statements_by_scope
            .get(&scope)
            .into_iter()
            .flatten()
            .copied()
            .map(|statement| {
                let start = self
                    .document
                    .statement(statement)
                    .expect("indexed statement belongs to the document")
                    .span()
                    .start;
                (start, Event::Statement(statement))
            })
            .collect();
        events.extend(child_scopes.into_iter().filter_map(|child| {
            let child = self.document.scope(child)?;
            if child.header().is_some() {
                return None;
            }
            Some((child.open_span()?.start, Event::AnonymousScope(child.id())))
        }));
        events.sort_by_key(|(start, _)| *start);

        for (_, event) in events {
            match event {
                Event::Statement(statement) => {
                    if let Some(child_scope) = scopes_by_header.get(&statement).copied() {
                        self.project_header_scope(statement, child_scope, destination);
                    } else {
                        self.project_statement(statement, destination);
                    }
                }
                Event::AnonymousScope(child_scope) => {
                    self.project_lexical_scope(child_scope, "", destination);
                }
            }
        }
    }

    fn project_header_scope(
        &mut self,
        statement: StatementId,
        scope: ScopeId,
        destination: Ptr<BasicBlock>,
    ) {
        self.project_labels(statement, destination);
        if let Some(callable) = self.callables.get(&statement).copied() {
            let kind = match callable.kind() {
                CallableKind::Entry => CallableKindAttr::Entry,
                CallableKind::Function => CallableKindAttr::Function,
            };
            let statement_node = self
                .document
                .statement(statement)
                .expect("callable statement belongs to the document");
            let header = trim_header(statement_node.text(self.document.source()));
            let operation = PtxCallableOp::build_definition(
                self.ctx,
                callable.name(),
                kind,
                callable.is_extern(),
                header,
            );
            let body = operation
                .entry_block(self.ctx)
                .expect("a definition has an entry block");
            self.record_operation(
                operation.get_operation(),
                SourceNode::Statement { statement },
                statement_node.span(),
                destination,
            );
            self.record_block(body, scope);
            self.project_scope(scope, body);
            return;
        }

        let header = self
            .document
            .statement(statement)
            .expect("scope header belongs to the document")
            .text(self.document.source());
        self.project_lexical_scope(scope, trim_header(header), destination);
    }

    fn project_lexical_scope(
        &mut self,
        scope: ScopeId,
        header: &str,
        destination: Ptr<BasicBlock>,
    ) {
        let source_span = self
            .document
            .scope(scope)
            .expect("projected scope belongs to the document")
            .span();
        let operation = PtxScopeOp::build(self.ctx, header);
        let body = operation.body(self.ctx);
        self.record_operation(
            operation.get_operation(),
            SourceNode::Scope { scope },
            source_span,
            destination,
        );
        self.record_block(body, scope);
        self.project_scope(scope, body);
    }

    fn project_statement(&mut self, statement: StatementId, destination: Ptr<BasicBlock>) {
        let projected_labels = self.project_labels(statement, destination);
        let statement_node = self
            .document
            .statement(statement)
            .expect("indexed statement belongs to the document");
        let source_span = statement_node.span();
        let operation = match statement_node.kind() {
            StatementKind::Directive => self.directives.get(&statement).map(|directive| {
                PtxDirectiveOp::build(self.ctx, directive.name(), directive.arguments())
                    .get_operation()
            }),
            StatementKind::Instruction => self.instructions.get(&statement).map(|instruction| {
                let prefix = instruction
                    .predicate()
                    .map_or_else(String::new, |predicate| {
                        format!(
                            "@{}{}",
                            if predicate.is_negated() { "!" } else { "" },
                            predicate.register()
                        )
                    });
                PtxInstructionOp::build(
                    self.ctx,
                    &prefix,
                    instruction.head(),
                    instruction.operands(),
                )
                .get_operation()
            }),
            StatementKind::CallableHeader => self.callables.get(&statement).map(|callable| {
                let kind = match callable.kind() {
                    CallableKind::Entry => CallableKindAttr::Entry,
                    CallableKind::Function => CallableKindAttr::Function,
                };
                PtxCallableOp::build_declaration(
                    self.ctx,
                    callable.name(),
                    kind,
                    callable.is_extern(),
                    trim_header(statement_node.text(self.document.source())),
                )
                .get_operation()
            }),
            StatementKind::Label if projected_labels => return,
            StatementKind::Label | StatementKind::Preprocessor | StatementKind::Unknown => None,
        }
        .unwrap_or_else(|| {
            PtxRawOp::build(self.ctx, statement_node.text(self.document.source())).get_operation()
        });
        self.record_operation(
            operation,
            SourceNode::Statement { statement },
            source_span,
            destination,
        );
    }

    fn project_labels(&mut self, statement: StatementId, destination: Ptr<BasicBlock>) -> bool {
        let labels = self.labels.get(&statement).cloned().unwrap_or_default();
        for label in &labels {
            let operation = PtxLabelOp::build(self.ctx, label.name()).get_operation();
            self.record_operation(
                operation,
                SourceNode::Label { label: label.id() },
                label.span(),
                destination,
            );
        }
        !labels.is_empty()
    }
}

fn trim_header(text: &str) -> &str {
    text.trim().trim_end_matches([';', '{']).trim_end()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{PtxCallableOp, PtxDirectiveOp, PtxInstructionOp, PtxLabelOp, PtxScopeOp};
    use pliron::common_traits::Verify;
    use pliron::context::Context;
    use pliron::linked_list::ContainsLinkedList;
    use pliron::op::Op;

    #[test]
    fn projects_module_and_callable_structure_without_source_attributes() {
        let source = "\
.version 8.9
.target sm_120a
.visible .entry kernel() {
    .reg .pred %p<2>;
L0:
    @%p0 future.op.u32 {%r1, %r2}, [%rd3];
    {
        mov.u32 %r1, 7;
    }
    ret;
}
";
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let projection = Projection::parse(&mut ctx, source).unwrap();
        assert_eq!(projection.document().source(), source);

        let module_ops: Vec<_> = projection
            .module()
            .body(&ctx)
            .deref(&ctx)
            .iter(&ctx)
            .collect();
        assert_eq!(module_ops.len(), 3);
        assert!(Operation::is_op::<PtxDirectiveOp>(module_ops[0], &ctx));
        assert!(Operation::is_op::<PtxDirectiveOp>(module_ops[1], &ctx));
        let callable = Operation::get_op::<PtxCallableOp>(module_ops[2], &ctx).unwrap();
        assert!(callable.is_definition(&ctx));

        let callable_ops: Vec<_> = callable
            .entry_block(&ctx)
            .unwrap()
            .deref(&ctx)
            .iter(&ctx)
            .collect();
        assert!(Operation::is_op::<PtxDirectiveOp>(callable_ops[0], &ctx));
        assert!(Operation::is_op::<PtxLabelOp>(callable_ops[1], &ctx));
        assert!(Operation::is_op::<PtxInstructionOp>(callable_ops[2], &ctx));
        assert!(Operation::is_op::<PtxScopeOp>(callable_ops[3], &ctx));
        assert!(Operation::is_op::<PtxInstructionOp>(callable_ops[4], &ctx));

        assert_eq!(projection.blocks().len(), 3);
        for node in projection.nodes() {
            assert_eq!(
                projection.source_node(node.operation()),
                Some(node.source_node())
            );
            if let SourceNode::Label { label } = node.source_node() {
                assert_eq!(
                    projection.document().label(label).unwrap().span(),
                    node.source_span()
                );
            }
        }
        for block in projection.blocks() {
            assert_eq!(
                projection.source_scope(block.block()),
                Some(block.source_scope())
            );
        }
        projection
            .module()
            .get_operation()
            .deref(&ctx)
            .verify(&ctx)
            .unwrap();
    }

    #[test]
    fn projects_declarations_without_inventing_body_regions() {
        let source = ".extern .func helper(.param .b32 x);\n";
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let projection = Projection::parse(&mut ctx, source).unwrap();
        let operation = projection
            .module()
            .body(&ctx)
            .deref(&ctx)
            .iter(&ctx)
            .next()
            .unwrap();
        let callable = Operation::get_op::<PtxCallableOp>(operation, &ctx).unwrap();
        assert!(!callable.is_definition(&ctx));
        assert!(callable.is_external(&ctx));
    }
}
