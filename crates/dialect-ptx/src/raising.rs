/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Transactional raising from lossless surface PTX to native Pliron CFG.

use crate::attributes::{CallableKindAttr, TerminatorKindAttr};
use crate::cfg::{BlockId, CfgError, ControlFlow, EdgeKind, ExitKind};
use crate::ops::{
    PtxCallableOp, PtxCfgCallableOp, PtxDirectiveOp, PtxInstructionOp, PtxLabelOp, PtxModuleOp,
    PtxRawOp, PtxTerminatorOp,
};
use crate::projection::SourceNode;
use crate::scopes::{ScopeFlattenError, ScopeFlattenPlan};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::op::Op;
use pliron::operation::Operation;
use ptx_syntax::{
    Callable, CallableKind, Document, EditError, LabelId, ParseError, ScopeId, StatementId,
    StatementKind,
};
use std::collections::HashMap;
use std::fmt;
use std::ops::Range;

/// A fully checked, context-independent native CFG construction plan.
///
/// Analysis performs all fallible parsing, normalization, CFG recovery, and
/// statement placement. [`Self::materialize`] only allocates the already
/// proven operation graph, so a failed analysis cannot partially mutate a
/// caller's Pliron IR.
pub struct NativeCfgPlan {
    normalized_source: String,
    items: Vec<ModuleItemPlan>,
    block_count: usize,
}

impl NativeCfgPlan {
    pub fn analyze(source: &str) -> Result<Self, RaiseError> {
        let surface = Document::parse(source).map_err(RaiseError::Parse)?;
        let flatten = ScopeFlattenPlan::analyze(&surface).map_err(RaiseError::ScopeFlatten)?;
        let normalized_source = flatten.apply(source).map_err(RaiseError::Edit)?;
        let document = Document::parse(&normalized_source).map_err(RaiseError::Parse)?;
        let control_flow = ControlFlow::analyze(&document).map_err(RaiseError::ControlFlow)?;
        validate_root_scopes(&document)?;

        let directives = document
            .directives()
            .iter()
            .map(|directive| (directive.statement(), directive))
            .collect::<HashMap<_, _>>();
        let callables = document
            .callables()
            .iter()
            .map(|callable| (callable.statement(), callable))
            .collect::<HashMap<_, _>>();
        let labels = labels_by_statement(&document);
        let mut items = Vec::new();
        let mut block_count = 0;
        for statement in document
            .statements()
            .iter()
            .filter(|statement| statement.scope() == ScopeId::ROOT)
        {
            let item = match statement.kind() {
                StatementKind::Directive => {
                    if labels.contains_key(&statement.id()) {
                        ModuleItemPlan::Raw {
                            statement: statement.id(),
                            span: statement.span(),
                            text: statement.text(document.source()).to_string(),
                        }
                    } else {
                        let directive = directives.get(&statement.id()).ok_or(
                            RaiseError::UnsupportedStatement {
                                statement: statement.id(),
                                kind: statement.kind(),
                            },
                        )?;
                        ModuleItemPlan::Directive {
                            statement: statement.id(),
                            span: statement.span(),
                            name: directive.name().to_string(),
                            arguments: directive.arguments().to_string(),
                        }
                    }
                }
                StatementKind::CallableHeader => {
                    let callable =
                        callables
                            .get(&statement.id())
                            .ok_or(RaiseError::UnsupportedStatement {
                                statement: statement.id(),
                                kind: statement.kind(),
                            })?;
                    if callable.definition_scope().is_some() {
                        let recovered = control_flow.for_callable(statement.id()).ok_or(
                            RaiseError::MissingCallableControlFlow {
                                statement: statement.id(),
                            },
                        )?;
                        let plan =
                            plan_callable(&document, callable, recovered, &directives, &labels)?;
                        block_count += plan.blocks.len();
                        ModuleItemPlan::Definition(plan)
                    } else {
                        ModuleItemPlan::Declaration(CallableHeaderPlan::new(
                            callable,
                            trim_header(statement.text(document.source())),
                            statement.span(),
                        ))
                    }
                }
                StatementKind::Preprocessor => ModuleItemPlan::Raw {
                    statement: statement.id(),
                    span: statement.span(),
                    text: statement.text(document.source()).to_string(),
                },
                kind => {
                    return Err(RaiseError::UnsupportedStatement {
                        statement: statement.id(),
                        kind,
                    });
                }
            };
            items.push(item);
        }
        Ok(Self {
            normalized_source,
            items,
            block_count,
        })
    }

    pub fn normalized_source(&self) -> &str {
        &self.normalized_source
    }

    pub fn callable_count(&self) -> usize {
        self.items
            .iter()
            .filter(|item| matches!(item, ModuleItemPlan::Definition(_)))
            .count()
    }

    pub fn block_count(&self) -> usize {
        self.block_count
    }

    pub fn materialize(self, ctx: &mut Context) -> NativeCfgProjection {
        let module = PtxModuleOp::build(ctx);
        let destination = module.body(ctx);
        let mut nodes = Vec::new();
        let mut blocks = Vec::with_capacity(self.block_count);
        for item in self.items {
            match item {
                ModuleItemPlan::Directive {
                    statement,
                    span,
                    name,
                    arguments,
                } => {
                    let operation = PtxDirectiveOp::build(ctx, &name, &arguments).get_operation();
                    insert_node(
                        ctx,
                        operation,
                        destination,
                        Some(SourceNode::Statement { statement }),
                        Some(span),
                        &mut nodes,
                    );
                }
                ModuleItemPlan::Raw {
                    statement,
                    span,
                    text,
                } => {
                    let operation = PtxRawOp::build(ctx, &text).get_operation();
                    insert_node(
                        ctx,
                        operation,
                        destination,
                        Some(SourceNode::Statement { statement }),
                        Some(span),
                        &mut nodes,
                    );
                }
                ModuleItemPlan::Declaration(header) => {
                    let operation = PtxCallableOp::build_declaration(
                        ctx,
                        &header.name,
                        header.kind,
                        header.is_extern,
                        &header.header,
                    )
                    .get_operation();
                    insert_node(
                        ctx,
                        operation,
                        destination,
                        Some(SourceNode::Statement {
                            statement: header.statement,
                        }),
                        Some(header.span),
                        &mut nodes,
                    );
                }
                ModuleItemPlan::Definition(callable) => {
                    materialize_callable(ctx, callable, destination, &mut nodes, &mut blocks)
                }
            }
        }
        let nodes_by_operation = nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.operation, index))
            .collect();
        let nodes_by_source = nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| Some((node.source_node?, index)))
            .collect();
        NativeCfgProjection {
            normalized_source: self.normalized_source,
            module,
            nodes,
            nodes_by_operation,
            nodes_by_source,
            blocks,
        }
    }
}

pub struct NativeCfgProjection {
    normalized_source: String,
    module: PtxModuleOp,
    nodes: Vec<RaisedNode>,
    nodes_by_operation: HashMap<Ptr<Operation>, usize>,
    nodes_by_source: HashMap<SourceNode, usize>,
    blocks: Vec<RaisedBlock>,
}

impl NativeCfgProjection {
    pub fn normalized_source(&self) -> &str {
        &self.normalized_source
    }

    pub fn module(&self) -> PtxModuleOp {
        self.module
    }

    pub fn nodes(&self) -> &[RaisedNode] {
        &self.nodes
    }

    pub fn blocks(&self) -> &[RaisedBlock] {
        &self.blocks
    }

    pub fn source_node(&self, operation: Ptr<Operation>) -> Option<SourceNode> {
        self.nodes_by_operation
            .get(&operation)
            .and_then(|index| self.nodes[*index].source_node)
    }

    pub fn operation_for_source(&self, source: SourceNode) -> Option<Ptr<Operation>> {
        self.nodes_by_source
            .get(&source)
            .map(|index| self.nodes[*index].operation)
    }
}

#[derive(Clone, Debug)]
pub struct RaisedNode {
    operation: Ptr<Operation>,
    source_node: Option<SourceNode>,
    source_span: Option<Range<usize>>,
}

impl RaisedNode {
    pub fn operation(&self) -> Ptr<Operation> {
        self.operation
    }

    pub fn source_node(&self) -> Option<SourceNode> {
        self.source_node
    }

    pub fn source_span(&self) -> Option<Range<usize>> {
        self.source_span.clone()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RaisedBlock {
    block: Ptr<BasicBlock>,
    callable: StatementId,
    source_block: BlockId,
}

impl RaisedBlock {
    pub fn block(self) -> Ptr<BasicBlock> {
        self.block
    }

    pub fn callable(self) -> StatementId {
        self.callable
    }

    pub fn source_block(self) -> BlockId {
        self.source_block
    }
}

#[derive(Debug)]
pub enum RaiseError {
    Parse(ParseError),
    ScopeFlatten(ScopeFlattenError),
    Edit(EditError),
    ControlFlow(CfgError),
    UnsupportedRootScope {
        scope: ScopeId,
        header: Option<StatementId>,
    },
    UnsupportedStatement {
        statement: StatementId,
        kind: StatementKind,
    },
    MissingCallableControlFlow {
        statement: StatementId,
    },
    MissingInstructionBlock {
        callable: StatementId,
        statement: StatementId,
    },
    TrailingStatement {
        callable: StatementId,
        statement: StatementId,
    },
}

impl fmt::Display for RaiseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(error) => error.fmt(formatter),
            Self::ScopeFlatten(error) => error.fmt(formatter),
            Self::Edit(error) => error.fmt(formatter),
            Self::ControlFlow(error) => error.fmt(formatter),
            Self::UnsupportedRootScope { scope, header } => write!(
                formatter,
                "PTX native CFG raising does not support root scope {} with header {header:?}",
                scope.index()
            ),
            Self::UnsupportedStatement { statement, kind } => write!(
                formatter,
                "PTX native CFG raising does not support {kind:?} statement {}",
                statement.index()
            ),
            Self::MissingCallableControlFlow { statement } => write!(
                formatter,
                "PTX callable statement {} has no recovered control flow",
                statement.index()
            ),
            Self::MissingInstructionBlock {
                callable,
                statement,
            } => write!(
                formatter,
                "PTX callable statement {} instruction statement {} has no recovered block",
                callable.index(),
                statement.index()
            ),
            Self::TrailingStatement {
                callable,
                statement,
            } => write!(
                formatter,
                "PTX callable statement {} has statement {} after its final instruction",
                callable.index(),
                statement.index()
            ),
        }
    }
}

impl std::error::Error for RaiseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(error) => Some(error),
            Self::ScopeFlatten(error) => Some(error),
            Self::Edit(error) => Some(error),
            Self::ControlFlow(error) => Some(error),
            Self::UnsupportedRootScope { .. }
            | Self::UnsupportedStatement { .. }
            | Self::MissingCallableControlFlow { .. }
            | Self::MissingInstructionBlock { .. }
            | Self::TrailingStatement { .. } => None,
        }
    }
}

enum ModuleItemPlan {
    Directive {
        statement: StatementId,
        span: Range<usize>,
        name: String,
        arguments: String,
    },
    Raw {
        statement: StatementId,
        span: Range<usize>,
        text: String,
    },
    Declaration(CallableHeaderPlan),
    Definition(CallablePlan),
}

struct CallableHeaderPlan {
    statement: StatementId,
    span: Range<usize>,
    name: String,
    kind: CallableKindAttr,
    is_extern: bool,
    header: String,
}

impl CallableHeaderPlan {
    fn new(callable: &Callable<'_>, header: &str, span: Range<usize>) -> Self {
        Self {
            statement: callable.statement(),
            span,
            name: callable.name().to_string(),
            kind: callable_kind(callable.kind()),
            is_extern: callable.is_extern(),
            header: header.to_string(),
        }
    }
}

struct CallablePlan {
    header: CallableHeaderPlan,
    blocks: Vec<BlockPlan>,
}

struct BlockPlan {
    id: BlockId,
    nodes: Vec<NodePlan>,
    terminator: TerminatorPlan,
}

enum NodePlan {
    Label {
        label: LabelId,
        span: Range<usize>,
        name: String,
    },
    Directive {
        statement: StatementId,
        span: Range<usize>,
        name: String,
        arguments: String,
    },
    Instruction {
        statement: StatementId,
        span: Range<usize>,
        prefix: String,
        head: String,
        operands: Vec<String>,
    },
    Raw {
        statement: StatementId,
        span: Range<usize>,
        text: String,
    },
}

struct TerminatorPlan {
    source: Option<(StatementId, Range<usize>)>,
    kind: TerminatorKindAttr,
    prefix: String,
    head: String,
    operands: Vec<String>,
    has_fallthrough: bool,
    successors: Vec<usize>,
}

fn validate_root_scopes(document: &Document<'_>) -> Result<(), RaiseError> {
    let callable_scopes = document
        .callables()
        .iter()
        .filter_map(|callable| callable.definition_scope())
        .collect::<std::collections::HashSet<_>>();
    for scope in document
        .scopes()
        .iter()
        .filter(|scope| scope.parent() == Some(ScopeId::ROOT))
    {
        if !callable_scopes.contains(&scope.id()) {
            return Err(RaiseError::UnsupportedRootScope {
                scope: scope.id(),
                header: scope.header(),
            });
        }
    }
    Ok(())
}

fn plan_callable(
    document: &Document<'_>,
    callable: &Callable<'_>,
    recovered: &crate::cfg::CallableControlFlow,
    directives: &HashMap<StatementId, &ptx_syntax::Directive<'_>>,
    labels: &HashMap<StatementId, Vec<&ptx_syntax::Label<'_>>>,
) -> Result<CallablePlan, RaiseError> {
    let scope = callable
        .definition_scope()
        .expect("a recovered callable is a definition");
    let mut instruction_blocks = HashMap::new();
    for block in recovered.blocks() {
        for statement in block.instructions() {
            instruction_blocks.insert(*statement, block.id().index());
        }
    }
    let mut statements_by_block = vec![Vec::new(); recovered.blocks().len()];
    let mut pending = Vec::new();
    for statement in document
        .statements()
        .iter()
        .filter(|statement| statement.scope() == scope)
    {
        if statement.kind() == StatementKind::Instruction {
            let block = instruction_blocks.get(&statement.id()).copied().ok_or(
                RaiseError::MissingInstructionBlock {
                    callable: callable.statement(),
                    statement: statement.id(),
                },
            )?;
            statements_by_block[block].append(&mut pending);
            statements_by_block[block].push(statement.id());
        } else {
            pending.push(statement.id());
        }
    }
    if let Some(statement) = pending.first().copied() {
        return Err(RaiseError::TrailingStatement {
            callable: callable.statement(),
            statement,
        });
    }

    let instructions = document
        .instructions()
        .iter()
        .map(|instruction| (instruction.statement(), instruction))
        .collect::<HashMap<_, _>>();
    let mut blocks = Vec::with_capacity(recovered.blocks().len());
    for block in recovered.blocks() {
        let actual_kind = actual_terminator_kind(block);
        let final_instruction = block.instructions().last().copied();
        let mut nodes = Vec::new();
        let mut terminator = None;
        for statement_id in &statements_by_block[block.id().index()] {
            let statement = document
                .statement(*statement_id)
                .expect("planned statement belongs to the document");
            match statement.kind() {
                StatementKind::Label => plan_labels(labels, *statement_id, &mut nodes),
                StatementKind::Directive => {
                    if labels.contains_key(statement_id) {
                        nodes.push(NodePlan::Raw {
                            statement: *statement_id,
                            span: statement.span(),
                            text: statement.text(document.source()).to_string(),
                        });
                    } else {
                        let directive = directives.get(statement_id).ok_or(
                            RaiseError::UnsupportedStatement {
                                statement: *statement_id,
                                kind: statement.kind(),
                            },
                        )?;
                        nodes.push(NodePlan::Directive {
                            statement: *statement_id,
                            span: statement.span(),
                            name: directive.name().to_string(),
                            arguments: directive.arguments().to_string(),
                        });
                    }
                }
                StatementKind::Instruction => {
                    plan_labels(labels, *statement_id, &mut nodes);
                    let instruction =
                        instructions
                            .get(statement_id)
                            .ok_or(RaiseError::UnsupportedStatement {
                                statement: *statement_id,
                                kind: statement.kind(),
                            })?;
                    let prefix = instruction
                        .predicate()
                        .map_or_else(String::new, |predicate| {
                            format!(
                                "@{}{}",
                                if predicate.is_negated() { "!" } else { "" },
                                predicate.register()
                            )
                        });
                    if Some(*statement_id) == final_instruction
                        && let Some(kind) = actual_kind
                    {
                        terminator = Some(TerminatorPlan {
                            source: Some((*statement_id, statement.span())),
                            kind,
                            prefix,
                            head: instruction.head().to_string(),
                            operands: instruction.operands().map(str::to_string).collect(),
                            has_fallthrough: block
                                .successors()
                                .iter()
                                .any(|edge| edge.kind() == EdgeKind::Fallthrough),
                            successors: ordered_successors(block),
                        });
                    } else {
                        nodes.push(NodePlan::Instruction {
                            statement: *statement_id,
                            span: statement.span(),
                            prefix,
                            head: instruction.head().to_string(),
                            operands: instruction.operands().map(str::to_string).collect(),
                        });
                    }
                }
                kind => {
                    return Err(RaiseError::UnsupportedStatement {
                        statement: *statement_id,
                        kind,
                    });
                }
            }
        }
        let terminator = terminator.unwrap_or_else(|| TerminatorPlan {
            source: None,
            kind: TerminatorKindAttr::Fallthrough,
            prefix: String::new(),
            head: String::new(),
            operands: Vec::new(),
            has_fallthrough: false,
            successors: ordered_successors(block),
        });
        blocks.push(BlockPlan {
            id: block.id(),
            nodes,
            terminator,
        });
    }
    let header = CallableHeaderPlan::new(
        callable,
        callable
            .definition_header_text()
            .expect("a recovered callable has a closed header")
            .trim(),
        document
            .statement(callable.statement())
            .expect("callable statement belongs to the document")
            .span(),
    );
    Ok(CallablePlan { header, blocks })
}

fn actual_terminator_kind(block: &crate::cfg::BasicBlock) -> Option<TerminatorKindAttr> {
    if let Some(exit) = block.exit() {
        return Some(match exit {
            ExitKind::Return => TerminatorKindAttr::Return,
            ExitKind::Thread => TerminatorKindAttr::ThreadExit,
            ExitKind::Trap => TerminatorKindAttr::Trap,
        });
    }
    if block
        .successors()
        .iter()
        .any(|edge| edge.kind() == EdgeKind::IndexedBranch)
    {
        return Some(TerminatorKindAttr::IndexedBranch);
    }
    block
        .successors()
        .iter()
        .any(|edge| edge.kind() == EdgeKind::Branch)
        .then_some(TerminatorKindAttr::Branch)
}

fn ordered_successors(block: &crate::cfg::BasicBlock) -> Vec<usize> {
    let mut edges = block.successors().to_vec();
    edges.sort_by_key(|edge| {
        (
            usize::from(edge.kind() != EdgeKind::Fallthrough),
            edge.block().index(),
        )
    });
    edges.into_iter().map(|edge| edge.block().index()).collect()
}

fn plan_labels(
    labels: &HashMap<StatementId, Vec<&ptx_syntax::Label<'_>>>,
    statement: StatementId,
    nodes: &mut Vec<NodePlan>,
) {
    for label in labels.get(&statement).into_iter().flatten() {
        nodes.push(NodePlan::Label {
            label: label.id(),
            span: label.span(),
            name: label.name().to_string(),
        });
    }
}

fn labels_by_statement<'document, 'source>(
    document: &'document Document<'source>,
) -> HashMap<StatementId, Vec<&'document ptx_syntax::Label<'source>>> {
    let mut labels = HashMap::new();
    for label in document.labels() {
        labels
            .entry(label.statement())
            .or_insert_with(Vec::new)
            .push(label);
    }
    labels
}

fn materialize_callable(
    ctx: &mut Context,
    callable: CallablePlan,
    destination: Ptr<BasicBlock>,
    nodes: &mut Vec<RaisedNode>,
    raised_blocks: &mut Vec<RaisedBlock>,
) {
    let operation = PtxCfgCallableOp::build(
        ctx,
        &callable.header.name,
        callable.header.kind,
        callable.header.is_extern,
        &callable.header.header,
    );
    insert_node(
        ctx,
        operation.get_operation(),
        destination,
        Some(SourceNode::Statement {
            statement: callable.header.statement,
        }),
        Some(callable.header.span),
        nodes,
    );
    let block_ptrs = callable
        .blocks
        .iter()
        .map(|_| operation.append_block(ctx))
        .collect::<Vec<_>>();
    for (block_plan, block) in callable.blocks.into_iter().zip(block_ptrs.iter().copied()) {
        raised_blocks.push(RaisedBlock {
            block,
            callable: callable.header.statement,
            source_block: block_plan.id,
        });
        for node in block_plan.nodes {
            materialize_node(ctx, node, block, nodes);
        }
        let successors = block_plan
            .terminator
            .successors
            .iter()
            .map(|index| block_ptrs[*index])
            .collect::<Vec<_>>();
        let terminator = PtxTerminatorOp::build(
            ctx,
            block_plan.terminator.kind,
            &block_plan.terminator.prefix,
            &block_plan.terminator.head,
            block_plan.terminator.operands.iter().map(String::as_str),
            block_plan.terminator.has_fallthrough,
            successors,
        )
        .get_operation();
        let (source_node, source_span) = block_plan
            .terminator
            .source
            .map_or((None, None), |(statement, span)| {
                (Some(SourceNode::Statement { statement }), Some(span))
            });
        insert_node(ctx, terminator, block, source_node, source_span, nodes);
    }
}

fn materialize_node(
    ctx: &mut Context,
    node: NodePlan,
    destination: Ptr<BasicBlock>,
    nodes: &mut Vec<RaisedNode>,
) {
    let (operation, source, span) = match node {
        NodePlan::Label { label, span, name } => (
            PtxLabelOp::build(ctx, &name).get_operation(),
            SourceNode::Label { label },
            span,
        ),
        NodePlan::Directive {
            statement,
            span,
            name,
            arguments,
        } => (
            PtxDirectiveOp::build(ctx, &name, &arguments).get_operation(),
            SourceNode::Statement { statement },
            span,
        ),
        NodePlan::Instruction {
            statement,
            span,
            prefix,
            head,
            operands,
        } => (
            PtxInstructionOp::build(ctx, &prefix, &head, operands.iter().map(String::as_str))
                .get_operation(),
            SourceNode::Statement { statement },
            span,
        ),
        NodePlan::Raw {
            statement,
            span,
            text,
        } => (
            PtxRawOp::build(ctx, &text).get_operation(),
            SourceNode::Statement { statement },
            span,
        ),
    };
    insert_node(ctx, operation, destination, Some(source), Some(span), nodes);
}

fn insert_node(
    ctx: &Context,
    operation: Ptr<Operation>,
    destination: Ptr<BasicBlock>,
    source_node: Option<SourceNode>,
    source_span: Option<Range<usize>>,
    nodes: &mut Vec<RaisedNode>,
) {
    operation.insert_at_back(destination, ctx);
    nodes.push(RaisedNode {
        operation,
        source_node,
        source_span,
    });
}

fn callable_kind(kind: CallableKind) -> CallableKindAttr {
    match kind {
        CallableKind::Entry => CallableKindAttr::Entry,
        CallableKind::Function => CallableKindAttr::Function,
    }
}

fn trim_header(text: &str) -> &str {
    text.trim().trim_end_matches([';', '{']).trim_end()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{emit_module, register};
    use pliron::common_traits::Verify;

    #[test]
    fn plans_before_materializing_loops_predicates_and_indexed_targets() {
        let source = "\
.version 9.3
.target sm_120a
.visible .entry kernel() {
    .reg .pred %p;
    .reg .b32 %r<2>;
    {
        .reg .b32 %r0;
        mov.u32 %r0, 0;
    }
targets: .branchtargets L0, Done;
L0:
    @%p bra Done;
    @%p brx.idx %r0, targets;
    bra L0;
Done:
    ret;
}
";
        let plan = NativeCfgPlan::analyze(source).unwrap();
        assert_eq!(plan.callable_count(), 1);
        assert_eq!(plan.block_count(), 5);
        assert!(!plan.normalized_source().contains(".reg .b32 %r0;"));

        let mut ctx = Context::new();
        register(&mut ctx);
        let raised = plan.materialize(&mut ctx);
        raised
            .module()
            .get_operation()
            .deref(&ctx)
            .verify(&ctx)
            .unwrap();
        assert_eq!(raised.blocks().len(), 5);
        let emitted = emit_module(&ctx, &raised.module()).unwrap();
        let reparsed = Document::parse(&emitted).unwrap();
        let cfg = ControlFlow::analyze(&reparsed).unwrap();
        assert_eq!(cfg.callables()[0].blocks().len(), 5);
    }

    #[test]
    fn analysis_failure_does_not_require_or_mutate_a_context() {
        let source = ".version 9.3\n.entry kernel() { { .pragma \"nounroll\"; ret; } }";
        assert!(matches!(
            NativeCfgPlan::analyze(source),
            Err(RaiseError::ScopeFlatten(
                ScopeFlattenError::UnsupportedDirective { .. }
            ))
        ));
    }
}
