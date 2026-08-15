/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lossless structural views over PTX source text.
//!
//! This crate deliberately does not type-check the PTX ISA. Instructions are
//! discovered structurally, so an opcode introduced by a newer PTX version is
//! retained with the same source spans as a known opcode. Consumers which need
//! ISA semantics can layer that policy over [`Instruction::head`].

mod edit;
mod lexer;
mod syntax;

pub use edit::{EditError, EditScript};
pub use lexer::{Token, TokenKind};
pub use syntax::{
    Coverage, Diagnostic, DiagnosticKind, Scope, ScopeId, Statement, StatementId, StatementKind,
};

use std::fmt;
use std::ops::Range;

/// A parsed PTX document which borrows its source and owns only structural
/// indices into it.
#[derive(Clone, Debug)]
pub struct Document<'source> {
    source: &'source str,
    tokens: Vec<Token>,
    statements: Vec<Statement>,
    scopes: Vec<Scope>,
    diagnostics: Vec<Diagnostic>,
    coverage: Coverage,
    labels: Vec<Label<'source>>,
    directives: Vec<Directive<'source>>,
    callables: Vec<Callable<'source>>,
    instructions: Vec<Instruction<'source>>,
}

/// Stable index of a projected label in [`Document::labels`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LabelId(usize);

impl LabelId {
    pub fn index(self) -> usize {
        self.0
    }
}

/// One PTX statement label, including a label prefixed to another statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Label<'source> {
    source: &'source str,
    id: LabelId,
    statement: StatementId,
    scope: ScopeId,
    span: Range<usize>,
    name_span: Range<usize>,
}

/// A typed view of one PTX directive statement.
///
/// The view retains the original spelling and is projected from a
/// [`StatementKind::Directive`] node; it does not independently rescan source
/// lines.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Directive<'source> {
    source: &'source str,
    statement: StatementId,
    scope: ScopeId,
    span: Range<usize>,
    line_span: Range<usize>,
    name_span: Range<usize>,
    arguments_span: Range<usize>,
    label_name_spans: Vec<Range<usize>>,
}

/// The two callable forms defined by PTX.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallableKind {
    Entry,
    Function,
}

/// A typed view of one PTX callable declaration or definition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Callable<'source> {
    source: &'source str,
    statement: StatementId,
    scope: ScopeId,
    definition_scope: Option<ScopeId>,
    kind: CallableKind,
    span: Range<usize>,
    header_span: Range<usize>,
    body_span: Option<Range<usize>>,
    name_span: Range<usize>,
    is_extern: bool,
}

/// One semicolon-terminated PTX instruction.
///
/// The source text is not normalized. Predicates and same-line labels are
/// exposed through [`Self::prefix`], while [`Self::head`] retains the exact
/// opcode and modifier spelling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instruction<'source> {
    source: &'source str,
    statement: StatementId,
    scope: ScopeId,
    span: Range<usize>,
    prefix_span: Range<usize>,
    head_span: Range<usize>,
    operand_spans: Vec<Range<usize>>,
    label_name_spans: Vec<Range<usize>>,
    predicate: Option<Predicate<'source>>,
}

/// A guard predicate recovered from an instruction statement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Predicate<'source> {
    register: &'source str,
    negated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LabelSpans {
    span: Range<usize>,
    name_span: Range<usize>,
}

/// A lexical error which prevents reliable source-span recovery.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseError {
    SourceTooLarge { bytes: usize },
    UnterminatedBlockComment { offset: usize },
    UnterminatedQuotedString { offset: usize },
}

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceTooLarge { bytes } => {
                write!(formatter, "PTX source is {bytes} bytes; maximum is 4 GiB")
            }
            Self::UnterminatedBlockComment { offset } => {
                write!(formatter, "unterminated PTX block comment at byte {offset}")
            }
            Self::UnterminatedQuotedString { offset } => {
                write!(formatter, "unterminated PTX quoted string at byte {offset}")
            }
        }
    }
}

impl std::error::Error for ParseError {}

impl<'source> Document<'source> {
    /// Parse a lossless structural view of `source`.
    ///
    /// Tokens losslessly partition the original source. Unknown instruction
    /// heads are accepted, while unterminated comments and strings fail the
    /// document because every following source span would be ambiguous.
    pub fn parse(source: &'source str) -> Result<Self, ParseError> {
        let tokens = lexer::lex(source)?;
        let masked = mask_non_code(source, &tokens);
        let mut parsed = syntax::parse(source, &tokens);
        let labels = discover_labels(source, &tokens, &parsed.statements);
        let (directives, directive_diagnostics) =
            discover_directives(source, &tokens, &parsed.statements);
        let (callables, callable_diagnostics) =
            discover_callables(source, &tokens, &parsed.statements, &parsed.scopes);
        let (instructions, instruction_diagnostics) = discover_instructions(
            source,
            &masked,
            &tokens,
            &parsed.statements,
            &parsed.diagnostics,
        );
        parsed.coverage.add_diagnostics(
            directive_diagnostics.len()
                + callable_diagnostics.len()
                + instruction_diagnostics.len(),
        );
        parsed.diagnostics.extend(directive_diagnostics);
        parsed.diagnostics.extend(callable_diagnostics);
        parsed.diagnostics.extend(instruction_diagnostics);
        Ok(Self {
            source,
            tokens,
            statements: parsed.statements,
            scopes: parsed.scopes,
            diagnostics: parsed.diagnostics,
            coverage: parsed.coverage,
            labels,
            directives,
            callables,
            instructions,
        })
    }

    pub fn source(&self) -> &'source str {
        self.source
    }

    /// Lossless lexical tokens in source order.
    ///
    /// Their spans are contiguous and cover exactly [`Self::source`].
    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }

    /// Structural statements in source order. Every non-trivia token is owned
    /// by one statement or a lexical scope delimiter. Unrecognized input is
    /// retained as [`StatementKind::Unknown`].
    pub fn statements(&self) -> &[Statement] {
        &self.statements
    }

    /// Lexical scopes in source order, beginning with [`ScopeId::ROOT`].
    pub fn scopes(&self) -> &[Scope] {
        &self.scopes
    }

    pub fn statement(&self, id: StatementId) -> Option<&Statement> {
        self.statements.get(id.index())
    }

    pub fn scope(&self, id: ScopeId) -> Option<&Scope> {
        self.scopes.get(id.index())
    }

    /// Recoverable structural problems found while retaining later nodes.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    pub fn coverage(&self) -> Coverage {
        self.coverage
    }

    pub fn labels(&self) -> &[Label<'source>] {
        &self.labels
    }

    pub fn label(&self, id: LabelId) -> Option<&Label<'source>> {
        self.labels.get(id.index())
    }

    pub fn directives(&self) -> &[Directive<'source>] {
        &self.directives
    }

    pub fn callables(&self) -> &[Callable<'source>] {
        &self.callables
    }

    pub fn instructions(&self) -> &[Instruction<'source>] {
        &self.instructions
    }
}

impl<'source> Label<'source> {
    pub fn id(&self) -> LabelId {
        self.id
    }

    pub fn statement(&self) -> StatementId {
        self.statement
    }

    pub fn scope(&self) -> ScopeId {
        self.scope
    }

    /// Byte range covering the label name and terminal colon.
    pub fn span(&self) -> Range<usize> {
        self.span.clone()
    }

    pub fn name(&self) -> &'source str {
        &self.source[self.name_span.clone()]
    }
}

impl<'source> Directive<'source> {
    pub fn statement(&self) -> StatementId {
        self.statement
    }

    pub fn scope(&self) -> ScopeId {
        self.scope
    }

    /// Byte range covering optional labels and the directive without leading
    /// indentation, trailing comment, or newline.
    pub fn span(&self) -> Range<usize> {
        self.span.clone()
    }

    /// Byte range covering the physical source line where the directive
    /// begins, including its newline when present.
    pub fn line_span(&self) -> Range<usize> {
        self.line_span.clone()
    }

    pub fn name(&self) -> &'source str {
        &self.source[self.name_span.clone()]
    }

    pub fn labels(&self) -> impl ExactSizeIterator<Item = &'source str> + '_ {
        self.label_name_spans
            .iter()
            .map(|span| &self.source[span.clone()])
    }

    pub fn arguments(&self) -> &'source str {
        &self.source[self.arguments_span.clone()]
    }

    pub fn arguments_span(&self) -> Range<usize> {
        self.arguments_span.clone()
    }

    pub fn text(&self) -> &'source str {
        &self.source[self.span.clone()]
    }
}

impl<'source> Callable<'source> {
    pub fn statement(&self) -> StatementId {
        self.statement
    }

    pub fn scope(&self) -> ScopeId {
        self.scope
    }

    /// Scope containing the callable body, or `None` for a declaration.
    pub fn definition_scope(&self) -> Option<ScopeId> {
        self.definition_scope
    }

    pub fn kind(&self) -> CallableKind {
        self.kind
    }

    /// Byte range covering the complete declaration or definition.
    pub fn span(&self) -> Range<usize> {
        self.span.clone()
    }

    /// Byte range from the first linkage directive through the callable name.
    pub fn header_span(&self) -> Range<usize> {
        self.header_span.clone()
    }

    /// Source range inside a definition's outer braces.
    pub fn body_span(&self) -> Option<Range<usize>> {
        self.body_span.clone()
    }

    pub fn name(&self) -> &'source str {
        &self.source[self.name_span.clone()]
    }

    pub fn is_extern(&self) -> bool {
        self.is_extern
    }
}

impl<'source> Instruction<'source> {
    pub fn statement(&self) -> StatementId {
        self.statement
    }

    pub fn scope(&self) -> ScopeId {
        self.scope
    }

    /// Byte range covering the optional predicate/labels through the terminal
    /// semicolon. Leading indentation and trailing comments are excluded.
    pub fn span(&self) -> Range<usize> {
        self.span.clone()
    }

    /// Byte offset at which the opcode begins.
    pub fn head_offset(&self) -> usize {
        self.head_span.start
    }

    /// Byte offset immediately after the terminal semicolon.
    pub fn end_offset(&self) -> usize {
        self.span.end
    }

    /// Optional predicate and same-line labels preceding the opcode.
    pub fn prefix(&self) -> &'source str {
        &self.source[self.prefix_span.clone()]
    }

    pub fn labels(&self) -> impl ExactSizeIterator<Item = &'source str> + '_ {
        self.label_name_spans
            .iter()
            .map(|span| &self.source[span.clone()])
    }

    pub fn predicate(&self) -> Option<Predicate<'source>> {
        self.predicate
    }

    /// Exact opcode and ordered modifier spelling.
    pub fn head(&self) -> &'source str {
        &self.source[self.head_span.clone()]
    }

    /// Top-level operands in source order.
    pub fn operands(&self) -> impl ExactSizeIterator<Item = &'source str> + '_ {
        self.operand_spans
            .iter()
            .map(|span| &self.source[span.clone()])
    }

    pub fn text(&self) -> &'source str {
        &self.source[self.span.clone()]
    }
}

impl<'source> Predicate<'source> {
    pub fn register(self) -> &'source str {
        self.register
    }

    pub fn is_negated(self) -> bool {
        self.negated
    }
}

/// Split a comma-separated PTX operand list without splitting nested register
/// lists, addresses, or parameter tuples.
pub fn split_top_level(source: &str) -> Option<Vec<&str>> {
    split_top_level_spans(source, 0)
        .map(|spans| spans.into_iter().map(|span| &source[span]).collect())
}

fn discover_labels<'source>(
    source: &'source str,
    tokens: &[Token],
    statements: &[Statement],
) -> Vec<Label<'source>> {
    let mut labels = Vec::new();
    for statement in statements {
        let significant = significant_token_indices(tokens, statement);
        let (_, spans) = leading_label_spans(source, tokens, &significant);
        for spans in spans {
            labels.push(Label {
                source,
                id: LabelId(labels.len()),
                statement: statement.id(),
                scope: statement.scope(),
                span: spans.span,
                name_span: spans.name_span,
            });
        }
    }
    labels
}

fn discover_directives<'source>(
    source: &'source str,
    tokens: &[Token],
    statements: &[Statement],
) -> (Vec<Directive<'source>>, Vec<Diagnostic>) {
    let mut directives = Vec::new();
    let mut diagnostics = Vec::new();
    for statement in statements
        .iter()
        .filter(|statement| statement.kind() == StatementKind::Directive)
    {
        let significant = significant_token_indices(tokens, statement);
        let (cursor, label_spans) = leading_label_spans(source, tokens, &significant);
        let Some(name) = significant
            .get(cursor)
            .map(|index| &tokens[*index])
            .filter(|token| token.kind() == TokenKind::Word && token.text(source).starts_with('.'))
        else {
            diagnostics.push(Diagnostic::new(
                DiagnosticKind::MalformedDirective,
                statement.span(),
            ));
            continue;
        };
        let span = statement.span();
        directives.push(Directive {
            source,
            statement: statement.id(),
            scope: statement.scope(),
            line_span: physical_line_span(source, span.start),
            arguments_span: trim_span(source, name.span().end..span.end),
            name_span: name.span(),
            label_name_spans: label_spans
                .into_iter()
                .map(|spans| spans.name_span)
                .collect(),
            span,
        });
    }
    (directives, diagnostics)
}

fn discover_callables<'source>(
    source: &'source str,
    tokens: &[Token],
    statements: &[Statement],
    scopes: &[Scope],
) -> (Vec<Callable<'source>>, Vec<Diagnostic>) {
    let mut callables = Vec::new();
    let mut diagnostics = Vec::new();
    for statement in statements
        .iter()
        .filter(|statement| statement.kind() == StatementKind::CallableHeader)
    {
        let significant = significant_token_indices(tokens, statement);
        let Some((keyword_cursor, kind)) =
            significant.iter().enumerate().find_map(|(cursor, index)| {
                match tokens[*index].text(source) {
                    ".entry" => Some((cursor, CallableKind::Entry)),
                    ".func" => Some((cursor, CallableKind::Function)),
                    _ => None,
                }
            })
        else {
            diagnostics.push(Diagnostic::new(
                DiagnosticKind::MalformedCallable,
                statement.span(),
            ));
            continue;
        };

        let mut name_cursor = keyword_cursor + 1;
        if kind == CallableKind::Function
            && significant
                .get(name_cursor)
                .is_some_and(|index| tokens[*index].text(source) == "(")
        {
            let Some(after_parameters) =
                skip_balanced_tokens(source, tokens, &significant, name_cursor, "(", ")")
            else {
                diagnostics.push(Diagnostic::new(
                    DiagnosticKind::MalformedCallable,
                    statement.span(),
                ));
                continue;
            };
            name_cursor = after_parameters;
        }
        let Some(name) = significant
            .get(name_cursor)
            .map(|index| &tokens[*index])
            .filter(|token| token.kind() == TokenKind::Word)
        else {
            diagnostics.push(Diagnostic::new(
                DiagnosticKind::MalformedCallable,
                statement.span(),
            ));
            continue;
        };
        let definition_scope = scopes
            .iter()
            .find(|scope| scope.header() == Some(statement.id()))
            .map(Scope::id);
        let definition = definition_scope.and_then(|scope| scopes.get(scope.index()));
        let closed_definition = definition.filter(|scope| scope.close_span().is_some());
        let span = closed_definition.map_or_else(
            || statement.span(),
            |scope| statement.span().start..scope.span().end,
        );
        let body_span = closed_definition.map(Scope::body_span);
        callables.push(Callable {
            source,
            statement: statement.id(),
            scope: statement.scope(),
            definition_scope,
            kind,
            span,
            header_span: statement.span().start..name.span().end,
            body_span,
            name_span: name.span(),
            is_extern: significant[..keyword_cursor]
                .iter()
                .any(|index| tokens[*index].text(source) == ".extern"),
        });
    }
    (callables, diagnostics)
}

fn significant_token_indices(tokens: &[Token], statement: &Statement) -> Vec<usize> {
    statement
        .token_range()
        .filter(|index| !tokens[*index].kind().is_trivia())
        .collect()
}

fn leading_label_spans(
    source: &str,
    tokens: &[Token],
    significant: &[usize],
) -> (usize, Vec<LabelSpans>) {
    let mut cursor = 0usize;
    let mut spans = Vec::new();
    while cursor + 1 < significant.len()
        && tokens[significant[cursor]].kind() == TokenKind::Word
        && tokens[significant[cursor + 1]].text(source) == ":"
        && (cursor + 2 == significant.len() || tokens[significant[cursor + 2]].text(source) != ":")
    {
        let name_span = tokens[significant[cursor]].span();
        let span = name_span.start..tokens[significant[cursor + 1]].span().end;
        spans.push(LabelSpans { span, name_span });
        cursor += 2;
    }
    (cursor, spans)
}

fn skip_balanced_tokens(
    source: &str,
    tokens: &[Token],
    significant: &[usize],
    start: usize,
    open: &str,
    close: &str,
) -> Option<usize> {
    let mut depth = 0usize;
    for (cursor, index) in significant.iter().enumerate().skip(start) {
        match tokens[*index].text(source) {
            token if token == open => depth += 1,
            token if token == close => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(cursor + 1);
                }
            }
            _ => {}
        }
    }
    None
}

fn physical_line_span(source: &str, offset: usize) -> Range<usize> {
    let start = source[..offset]
        .rfind('\n')
        .map_or(0, |newline| newline + 1);
    let end = source[offset..]
        .find('\n')
        .map_or(source.len(), |newline| offset + newline + 1);
    start..end
}

fn discover_instructions<'source>(
    source: &'source str,
    masked: &str,
    tokens: &[Token],
    statements: &[Statement],
    diagnostics: &[Diagnostic],
) -> (Vec<Instruction<'source>>, Vec<Diagnostic>) {
    let mut instructions = Vec::new();
    let mut projection_diagnostics = Vec::new();
    for statement in statements
        .iter()
        .filter(|statement| statement.kind() == StatementKind::Instruction)
    {
        if let Some(instruction) = instruction_from_statement(source, masked, tokens, statement) {
            instructions.push(instruction);
        } else if !diagnostics
            .iter()
            .any(|diagnostic| ranges_overlap(diagnostic.span(), statement.span()))
        {
            projection_diagnostics.push(Diagnostic::new(
                DiagnosticKind::MalformedInstruction,
                statement.span(),
            ));
        }
    }
    (instructions, projection_diagnostics)
}

fn ranges_overlap(left: Range<usize>, right: Range<usize>) -> bool {
    left.start < right.end && right.start < left.end
}

fn instruction_from_statement<'source>(
    source: &'source str,
    masked: &str,
    tokens: &[Token],
    statement: &Statement,
) -> Option<Instruction<'source>> {
    let significant = significant_token_indices(tokens, statement);
    let (mut cursor, label_spans) = leading_label_spans(source, tokens, &significant);
    let mut predicate = None;
    if significant
        .get(cursor)
        .is_some_and(|index| tokens[*index].text(source) == "@")
    {
        cursor += 1;
        if significant
            .get(cursor)
            .is_some_and(|index| tokens[*index].text(source) == "!")
        {
            cursor += 1;
        }
        let register = tokens.get(*significant.get(cursor)?)?;
        if register.kind() != TokenKind::Word {
            return None;
        }
        predicate = Some(Predicate {
            register: register.text(source),
            negated: significant
                .get(cursor.wrapping_sub(1))
                .is_some_and(|index| tokens[*index].text(source) == "!"),
        });
        cursor += 1;
    }
    let head_start = tokens.get(*significant.get(cursor)?)?;
    let mut head_end = head_start.span().end;
    while cursor + 2 < significant.len()
        && tokens[significant[cursor + 1]].text(source) == "::"
        && tokens[significant[cursor + 2]].kind() == TokenKind::Word
    {
        cursor += 2;
        head_end = tokens[significant[cursor]].span().end;
    }
    let semicolon = tokens.get(*significant.last()?)?;
    if head_start.kind() != TokenKind::Word || semicolon.text(source) != ";" {
        return None;
    }
    let head_span = head_start.span().start..head_end;
    let prefix_span = trim_span(masked, statement.span().start..head_span.start);
    let operands = trim_span(masked, head_span.end..semicolon.span().start);
    let operand_spans = if operands.is_empty() {
        Vec::new()
    } else {
        split_top_level_spans(&masked[operands.clone()], operands.start)?
    };
    Some(Instruction {
        source,
        statement: statement.id(),
        scope: statement.scope(),
        span: statement.span(),
        prefix_span,
        head_span,
        operand_spans,
        label_name_spans: label_spans
            .into_iter()
            .map(|spans| spans.name_span)
            .collect(),
        predicate,
    })
}

fn split_top_level_spans(source: &str, base: usize) -> Option<Vec<Range<usize>>> {
    let leading = source.len() - source.trim_start().len();
    let source = source.trim();
    let base = base + leading;
    if source.is_empty() {
        return Some(Vec::new());
    }

    let mut operands = Vec::new();
    let mut delimiters = Vec::new();
    let mut operand_start = 0usize;
    for (index, byte) in source.bytes().enumerate() {
        match byte {
            b'{' => delimiters.push(b'}'),
            b'[' => delimiters.push(b']'),
            b'(' => delimiters.push(b')'),
            b'}' | b']' | b')' if delimiters.pop() != Some(byte) => return None,
            b'}' | b']' | b')' => {}
            b',' if delimiters.is_empty() => {
                let span = trim_span(source, operand_start..index);
                if span.is_empty() {
                    return None;
                }
                operands.push(base + span.start..base + span.end);
                operand_start = index + 1;
            }
            _ => {}
        }
    }
    if !delimiters.is_empty() {
        return None;
    }
    let span = trim_span(source, operand_start..source.len());
    if span.is_empty() {
        return None;
    }
    operands.push(base + span.start..base + span.end);
    Some(operands)
}

fn trim_span(source: &str, span: Range<usize>) -> Range<usize> {
    let text = &source[span.clone()];
    if text.trim().is_empty() {
        return span.end..span.end;
    }
    let leading = text.len() - text.trim_start().len();
    let trailing = text.trim_end().len();
    span.start + leading..span.start + trailing
}

fn mask_non_code(source: &str, tokens: &[Token]) -> String {
    let mut masked = source.as_bytes().to_vec();
    for token in tokens.iter().filter(|token| {
        matches!(
            token.kind(),
            TokenKind::LineComment
                | TokenKind::BlockComment
                | TokenKind::QuotedString
                | TokenKind::Preprocessor
        )
    }) {
        for byte in &mut masked[token.span()] {
            if *byte != b'\n' && *byte != b'\r' {
                *byte = b' ';
            }
        }
    }
    String::from_utf8(masked).expect("masking tokens preserves UTF-8 boundaries")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_unknown_predicated_and_multiline_instructions() {
        let source = ".visible .entry kernel() {\n\
                      L0: @!%p1 future.op.u32\n\
                          {%r1, %r2}, [%rd3, {%r4, %r5}]; // tail\n\
                      ret;\n\
                      }\n";
        let document = Document::parse(source).unwrap();
        assert_eq!(document.instructions().len(), 2);
        let future = &document.instructions()[0];
        assert_eq!(future.prefix(), "L0: @!%p1");
        assert_eq!(future.labels().collect::<Vec<_>>(), ["L0"]);
        assert_eq!(future.predicate().unwrap().register(), "%p1");
        assert!(future.predicate().unwrap().is_negated());
        assert_eq!(future.head(), "future.op.u32");
        assert_eq!(
            future.operands().collect::<Vec<_>>(),
            ["{%r1, %r2}", "[%rd3, {%r4, %r5}]"]
        );
        assert_eq!(
            future.text(),
            "L0: @!%p1 future.op.u32\n{%r1, %r2}, [%rd3, {%r4, %r5}];"
        );
        assert_eq!(document.instructions()[1].head(), "ret");
    }

    #[test]
    fn ignores_comments_strings_directives_and_operand_symbols() {
        let source = "// fake.u32 %r1;\n\
                      .file 1 \"quoted.u32 %r2;\"\n\
                      .target sm_90\n\
                      .visible .entry kernel() {\n\
                      call.uni (%r1), helper, (%r2);\n\
                      /* fake2.u32 %r3; */ ret;\n\
                      }";
        let document = Document::parse(source).unwrap();
        assert_eq!(
            document
                .instructions()
                .iter()
                .map(Instruction::head)
                .collect::<Vec<_>>(),
            ["call.uni", "ret"]
        );
    }

    #[test]
    fn projects_directives_from_statement_nodes() {
        let source = "  .version 8.9\n.target sm_120a, debug // generated\n";
        let document = Document::parse(source).unwrap();
        assert_eq!(document.directives().len(), 2);
        let target = &document.directives()[1];
        assert_eq!(target.name(), ".target");
        assert_eq!(target.arguments(), "sm_120a, debug");
        assert_eq!(target.text(), ".target sm_120a, debug");
        assert_eq!(
            &source[target.line_span()],
            ".target sm_120a, debug // generated\n"
        );
        assert_eq!(
            document.statement(target.statement()).unwrap().kind(),
            StatementKind::Directive
        );
        assert_eq!(target.scope(), ScopeId::ROOT);
    }

    #[test]
    fn projects_prefixed_and_standalone_labels_with_lineage() {
        let source = "L0:\nL1: L2: @%p0 bra L0;\nts: .branchtargets L0, L1;\n";
        let document = Document::parse(source).unwrap();
        assert_eq!(
            document
                .labels()
                .iter()
                .map(Label::name)
                .collect::<Vec<_>>(),
            ["L0", "L1", "L2", "ts"]
        );
        assert_eq!(
            document.instructions()[0].labels().collect::<Vec<_>>(),
            ["L1", "L2"]
        );
        assert_eq!(
            document.directives()[0].labels().collect::<Vec<_>>(),
            ["ts"]
        );
        assert_eq!(document.directives()[0].name(), ".branchtargets");
        assert_eq!(document.directives()[0].arguments(), "L0, L1;");
        for label in document.labels() {
            assert_eq!(document.label(label.id()), Some(label));
            assert_eq!(
                document.statement(label.statement()).unwrap().scope(),
                label.scope()
            );
        }
    }

    #[test]
    fn projects_multiline_callable_headers_and_definition_scopes() {
        let source = "\
.visible
.entry kernel() { ret; }
.extern .func (.param .b32 result)
    __nv_helper(.param .b32 input);
.weak .func local_helper() { ret; }
";
        let document = Document::parse(source).unwrap();
        assert_eq!(document.callables().len(), 3);
        assert_eq!(document.callables()[0].name(), "kernel");
        assert_eq!(document.callables()[0].kind(), CallableKind::Entry);
        assert!(!document.callables()[0].is_extern());
        assert!(document.callables()[0].definition_scope().is_some());
        assert!(document.callables()[0].body_span().is_some());
        assert_eq!(
            &source[document.callables()[0].span()],
            ".visible\n.entry kernel() { ret; }"
        );
        assert_eq!(document.callables()[1].name(), "__nv_helper");
        assert_eq!(document.callables()[1].kind(), CallableKind::Function);
        assert!(document.callables()[1].is_extern());
        assert!(document.callables()[1].definition_scope().is_none());
        assert_eq!(
            document.callables()[1].span(),
            document
                .statement(document.callables()[1].statement())
                .unwrap()
                .span()
        );
        assert_eq!(document.callables()[2].name(), "local_helper");
        assert!(!document.callables()[2].is_extern());
        assert!(document.callables()[2].definition_scope().is_some());
        assert!(document.callables().iter().all(|callable| {
            document
                .statement(callable.statement())
                .is_some_and(|statement| statement.kind() == StatementKind::CallableHeader)
        }));
    }

    #[test]
    fn does_not_claim_an_unclosed_callable_body() {
        let document = Document::parse(".entry incomplete() {\nret;\n").unwrap();
        let callable = &document.callables()[0];
        assert!(callable.definition_scope().is_some());
        assert!(callable.body_span().is_none());
        assert_eq!(
            callable.span(),
            callable.header_span().start
                ..document.statement(callable.statement()).unwrap().span().end
        );
        assert!(
            document
                .diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.kind() == DiagnosticKind::UnterminatedDelimiter)
        );
    }

    #[test]
    fn retains_quoted_directive_arguments() {
        let document = Document::parse(".file 1 \"kernel.cu\"\n").unwrap();
        assert_eq!(document.directives()[0].arguments(), "1 \"kernel.cu\"");
    }

    #[test]
    fn preserves_utf8_byte_offsets_while_masking_non_code() {
        let source = "// λλ\nmov.u32 %r1, %tid.x;";
        let document = Document::parse(source).unwrap();
        let instruction = &document.instructions()[0];
        assert_eq!(&source[instruction.span()], instruction.text());
        assert_eq!(instruction.head_offset(), source.find("mov.u32").unwrap());
    }

    #[test]
    fn supports_multiple_instructions_on_one_line() {
        let document = Document::parse("mov.u32 %r1, 1; add.u32 %r2, %r1, 2;").unwrap();
        assert_eq!(
            document
                .instructions()
                .iter()
                .map(Instruction::head)
                .collect::<Vec<_>>(),
            ["mov.u32", "add.u32"]
        );
    }

    #[test]
    fn keeps_valid_instructions_after_a_recoverable_statement_error() {
        let document = Document::parse("mov.u32 %r1, [oops;\nret;").unwrap();
        assert_eq!(
            document
                .instructions()
                .iter()
                .map(Instruction::head)
                .collect::<Vec<_>>(),
            ["ret"]
        );
        assert_eq!(
            document.diagnostics()[0].kind(),
            DiagnosticKind::UnterminatedDelimiter
        );
    }

    #[test]
    fn retains_double_colon_instruction_modifiers_in_the_head() {
        let document = Document::parse(
            "L0: tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%r1];",
        )
        .unwrap();
        let instruction = &document.instructions()[0];
        assert_eq!(
            instruction.head(),
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
        );
        assert_eq!(instruction.prefix(), "L0:");
        assert_eq!(instruction.operands().collect::<Vec<_>>(), ["[%r1]"]);
    }

    #[test]
    fn rejects_unterminated_non_code_regions() {
        assert_eq!(
            Document::parse("/* no end").unwrap_err(),
            ParseError::UnterminatedBlockComment { offset: 0 }
        );
        assert_eq!(
            Document::parse(".file 1 \"no end").unwrap_err(),
            ParseError::UnterminatedQuotedString { offset: 8 }
        );
    }

    #[test]
    fn splits_nested_top_level_operands() {
        assert_eq!(
            split_top_level("{%r1, %r2}, [%rd1, {%r3, %r4}], (%r5, %r6)").unwrap(),
            ["{%r1, %r2}", "[%rd1, {%r3, %r4}]", "(%r5, %r6)"]
        );
        assert!(split_top_level("%r1, [%r2").is_none());
    }

    #[test]
    fn arbitrary_ascii_never_panics_or_loses_successfully_lexed_input() {
        const ALPHABET: &[u8] = b" abcXYZ019_.$%@!,:;{}[]()/*\\\"#\n\t+-|=";
        let mut state = 0x9e37_79b9_u32;
        for length in 0..512 {
            let mut source = String::with_capacity(length);
            for _ in 0..length {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                source.push(ALPHABET[(state as usize) % ALPHABET.len()] as char);
            }
            let Ok(document) = Document::parse(&source) else {
                continue;
            };
            assert_eq!(
                document
                    .tokens()
                    .iter()
                    .map(|token| token.text(&source))
                    .collect::<String>(),
                source
            );
            assert!(document.coverage().is_lossless());
            assert!(
                document
                    .statements()
                    .iter()
                    .all(|statement| document.scope(statement.scope()).is_some())
            );
            assert!(document.instructions().iter().all(|instruction| {
                document
                    .statement(instruction.statement())
                    .is_some_and(|statement| statement.kind() == StatementKind::Instruction)
            }));
        }
    }
}
