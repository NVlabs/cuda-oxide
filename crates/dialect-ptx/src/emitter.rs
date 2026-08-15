/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Deterministic PTX assembly-syntax emission from structured operations.

use crate::ops::{
    PtxCallableOp, PtxDirectiveOp, PtxInstructionOp, PtxModuleOp, PtxRawOp, PtxScopeOp,
};
use pliron::{
    basic_block::BasicBlock,
    common_traits::Verify,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
};
use std::fmt::{self, Write};

#[derive(Debug)]
pub enum EmitError {
    Verification(String),
    UnsupportedOperation(String),
    Format(fmt::Error),
}

impl fmt::Display for EmitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Verification(error) => write!(formatter, "invalid structured PTX: {error}"),
            Self::UnsupportedOperation(operation) => {
                write!(formatter, "cannot emit non-PTX operation {operation}")
            }
            Self::Format(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for EmitError {}

impl From<fmt::Error> for EmitError {
    fn from(error: fmt::Error) -> Self {
        Self::Format(error)
    }
}

/// Emit one structured PTX module to a newly allocated string.
pub fn emit_module(ctx: &Context, module: &PtxModuleOp) -> Result<String, EmitError> {
    let mut output = String::new();
    write_module(ctx, module, &mut output)?;
    Ok(output)
}

/// Emit one structured PTX module into a caller-owned formatting sink.
pub fn write_module(
    ctx: &Context,
    module: &PtxModuleOp,
    output: &mut impl Write,
) -> Result<(), EmitError> {
    module
        .get_operation()
        .deref(ctx)
        .verify(ctx)
        .map_err(|error| EmitError::Verification(error.to_string()))?;
    emit_block(ctx, module.body(ctx), 0, output)
}

fn emit_block(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    indent: usize,
    output: &mut impl Write,
) -> Result<(), EmitError> {
    for operation in block.deref(ctx).iter(ctx) {
        emit_operation(ctx, operation, indent, output)?;
    }
    Ok(())
}

fn emit_operation(
    ctx: &Context,
    operation: Ptr<Operation>,
    indent: usize,
    output: &mut impl Write,
) -> Result<(), EmitError> {
    if let Some(directive) = Operation::get_op::<PtxDirectiveOp>(operation, ctx) {
        write_indent(output, indent)?;
        output.write_str(&directive.name(ctx))?;
        let arguments = directive.arguments(ctx);
        if !arguments.is_empty() {
            output.write_char(' ')?;
            output.write_str(&arguments)?;
        }
        output.write_char('\n')?;
        return Ok(());
    }
    if let Some(callable) = Operation::get_op::<PtxCallableOp>(operation, ctx) {
        write_indent(output, indent)?;
        output.write_str(callable.header(ctx).trim())?;
        if let Some(region) = callable.region(ctx) {
            output.write_char('\n')?;
            write_indent(output, indent)?;
            output.write_str("{\n")?;
            for block in region.deref(ctx).iter(ctx) {
                emit_block(ctx, block, indent + 1, output)?;
            }
            write_indent(output, indent)?;
            output.write_str("}\n")?;
        } else {
            output.write_str(";\n")?;
        }
        return Ok(());
    }
    if let Some(scope) = Operation::get_op::<PtxScopeOp>(operation, ctx) {
        let header = scope.header(ctx);
        if !header.is_empty() {
            write_indent(output, indent)?;
            output.write_str(header.trim())?;
            output.write_char('\n')?;
        }
        write_indent(output, indent)?;
        output.write_str("{\n")?;
        emit_block(ctx, scope.body(ctx), indent + 1, output)?;
        write_indent(output, indent)?;
        output.write_str("}\n")?;
        return Ok(());
    }
    if let Some(instruction) = Operation::get_op::<PtxInstructionOp>(operation, ctx) {
        write_indent(output, indent)?;
        let prefix = instruction.prefix(ctx);
        if !prefix.is_empty() {
            output.write_str(prefix.trim())?;
            output.write_char(' ')?;
        }
        output.write_str(&instruction.head(ctx))?;
        let operands = instruction.operands(ctx);
        if !operands.is_empty() {
            output.write_char(' ')?;
            for (index, operand) in operands.iter().enumerate() {
                if index != 0 {
                    output.write_str(", ")?;
                }
                output.write_str(operand)?;
            }
        }
        output.write_str(";\n")?;
        return Ok(());
    }
    if let Some(raw) = Operation::get_op::<PtxRawOp>(operation, ctx) {
        let text = raw.text(ctx);
        for line in text.trim().lines() {
            write_indent(output, indent)?;
            output.write_str(line.trim())?;
            output.write_char('\n')?;
        }
        return Ok(());
    }
    Err(EmitError::UnsupportedOperation(
        Operation::get_opid(operation, ctx).to_string(),
    ))
}

fn write_indent(output: &mut impl Write, indent: usize) -> fmt::Result {
    for _ in 0..indent {
        output.write_str("    ")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Projection;

    #[test]
    fn projection_emits_canonical_nested_ptx() {
        let source = "\
.version 8.9
.target sm_120a
.address_size 64
.visible .entry kernel() {
    .reg .b32 %r<2>;
    {
      mov.u32 %r0, 7;
    }
    ret;
}
";
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        let projection = Projection::parse(&mut ctx, source).unwrap();
        let emitted = emit_module(&ctx, &projection.module()).unwrap();
        assert_eq!(
            emitted,
            "\
.version 8.9
.target sm_120a
.address_size 64
.visible .entry kernel()
{
    .reg .b32 %r<2>;
    {
        mov.u32 %r0, 7;
    }
    ret;
}
"
        );
        let reparsed = ptx_syntax::Document::parse(&emitted).unwrap();
        assert!(reparsed.coverage().is_complete());
    }
}
