/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Inline PTX marker-call translation.

use super::super::helpers::{emit_goto, emit_store_result_and_goto};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::values::ValueMap;
use crate::translator::{rvalue, types};
use dialect_nvvm::ops::InlinePtxOp;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::operation::Operation;
use rustc_public::mir;

const OUT_PREFIX: &str = "cuda_device::ptx::__ptx_asm_out_";
const VOID_PREFIX: &str = "cuda_device::ptx::__ptx_asm_void_";
const REGISTER_ONLY_OPTION: &str = "register_only";

#[derive(Copy, Clone)]
struct InlinePtxOptions {
    sideeffect: bool,
    convergent: bool,
}

#[derive(Copy, Clone)]
pub enum InlinePtxCallKind {
    Output { inputs: usize },
    Void { inputs: usize },
}

impl InlinePtxCallKind {
    pub fn from_path(path: &str) -> Option<Self> {
        if let Some(rest) = path.strip_prefix(OUT_PREFIX) {
            return rest
                .parse::<usize>()
                .ok()
                .map(|inputs| InlinePtxCallKind::Output { inputs });
        }
        if let Some(rest) = path.strip_prefix(VOID_PREFIX) {
            return rest
                .parse::<usize>()
                .ok()
                .map(|inputs| InlinePtxCallKind::Void { inputs });
        }
        None
    }

    fn has_output(self) -> bool {
        matches!(self, InlinePtxCallKind::Output { .. })
    }

    fn inputs(self) -> usize {
        match self {
            InlinePtxCallKind::Output { inputs } | InlinePtxCallKind::Void { inputs } => inputs,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn emit_inline_ptx(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    kind: InlinePtxCallKind,
) -> TranslationResult<Ptr<Operation>> {
    let expected_args = 3 + kind.inputs();
    if args.len() != expected_args {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "ptx_asm marker expected {expected_args} arguments, got {}",
                args.len()
            ))
        );
    }

    let template = literal_operand_string(&args[0], "ptx_asm template", loc.clone())?;
    let constraints = literal_operand_string(&args[1], "ptx_asm constraints", loc.clone())?;
    let options_marker = literal_operand_string(&args[2], "ptx_asm options", loc.clone())?;
    let options = parse_options(&options_marker, loc.clone())?;

    let mut input_values = Vec::with_capacity(kind.inputs());
    let mut last_op = prev_op;
    for arg in &args[3..] {
        let (value, arg_last_op) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        input_values.push(value);
        last_op = arg_last_op;
    }

    let result_tys = if kind.has_output() {
        vec![types::translate_destination_type(
            ctx,
            body,
            destination,
            &loc,
        )?]
    } else {
        vec![]
    };

    let inline_ptx = InlinePtxOp::build(
        ctx,
        result_tys,
        input_values,
        &template,
        &constraints,
        options.sideeffect,
        options.convergent,
    );
    inline_ptx.deref_mut(ctx).set_loc(loc.clone());

    let inline_ptx = if let Some(prev) = last_op {
        inline_ptx.insert_after(ctx, prev);
        inline_ptx
    } else {
        inline_ptx.insert_at_front(block_ptr, ctx);
        inline_ptx
    };

    if kind.has_output() {
        let result_value = inline_ptx.deref(ctx).get_result(0);
        emit_store_result_and_goto(
            ctx,
            destination,
            result_value,
            target,
            block_ptr,
            inline_ptx,
            value_map,
            block_map,
            loc,
            "ptx_asm output call without target block",
        )
    } else if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, inline_ptx, block_map, loc))
    } else {
        input_err!(
            loc,
            TranslationErr::unsupported("ptx_asm void call without target block".to_string())
        )
    }
}

fn literal_operand_string(
    operand: &mir::Operand,
    kind_name: &str,
    loc: Location,
) -> TranslationResult<String> {
    let bytes = match operand {
        mir::Operand::Constant(constant) => {
            rvalue::constant_bytes(constant, kind_name, loc.clone())?
        }
        other => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "{kind_name} must be a byte string literal, got MIR operand {other:?}"
                ))
            );
        }
    };

    String::from_utf8(bytes).map_err(|err| {
        pliron::input_error_noloc!(TranslationErr::unsupported(format!(
            "{kind_name} must be valid UTF-8: {err}"
        )))
    })
}

fn parse_options(marker: &str, loc: Location) -> TranslationResult<InlinePtxOptions> {
    match marker {
        "" => Ok(InlinePtxOptions {
            sideeffect: true,
            convergent: true,
        }),
        REGISTER_ONLY_OPTION => Ok(InlinePtxOptions {
            sideeffect: false,
            convergent: false,
        }),
        other => input_err!(
            loc,
            TranslationErr::unsupported(format!("unsupported ptx_asm options marker `{other}`"))
        ),
    }
}
