/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Inline PTX marker-call translation.

use super::super::helpers::{emit_goto, emit_store_result_and_goto};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::values::ValueMap;
use crate::translator::{rvalue, types};
use dialect_mir::ops::MirConstructTupleOp;
use dialect_mir::types::MirTupleType;
use dialect_nvvm::ops::InlinePtxOp;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

const OUT_PREFIX: &str = "cuda_device::ptx::__ptx_asm_out_";
const VOID_PREFIX: &str = "cuda_device::ptx::__ptx_asm_void_";
const REGISTER_ONLY_OPTION: &str = "register_only";
const REGISTER_ONLY_MAY_DIVERGE_OPTIONS: &str = "register_only,may_diverge";

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

/// Count the number of output constraints in a constraint string.
///
/// Output constraints are prefixed with `=` (e.g. `=r`, `=f`).
fn count_output_constraints(constraints: &str) -> usize {
    constraints
        .split(',')
        .filter(|c| c.starts_with('='))
        .count()
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

    if !kind.has_output() {
        // Void call: no results.
        let inline_ptx = InlinePtxOp::build(
            ctx,
            vec![],
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

        if let Some(target_idx) = target {
            return Ok(emit_goto(ctx, *target_idx, inline_ptx, block_map, loc));
        } else {
            return input_err!(
                loc,
                TranslationErr::unsupported("ptx_asm void call without target block".to_string())
            );
        }
    }

    // Output call: determine how many output constraints we have.
    let num_outputs = count_output_constraints(&constraints);

    if num_outputs <= 1 {
        // Single-output (backward-compatible path): the destination type is the result type.
        let result_tys = vec![types::translate_destination_type(
            ctx,
            body,
            destination,
            &loc,
        )?];

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
    } else {
        // Multi-output: the destination type is a tuple. Decompose it to get
        // per-output result types, then pack the results back into a tuple.
        let dest_ty = types::translate_destination_type(ctx, body, destination, &loc)?;
        let element_types = {
            let t = dest_ty.deref(ctx);
            match t.downcast_ref::<MirTupleType>() {
                Some(tup) if tup.get_types().len() == num_outputs => tup.get_types().to_vec(),
                _ => {
                    return input_err!(
                        loc.clone(),
                        TranslationErr::unsupported(format!(
                            "ptx_asm multi-output destination must be a {num_outputs}-element tuple"
                        ))
                    );
                }
            }
        };

        let inline_ptx = InlinePtxOp::build(
            ctx,
            element_types,
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

        // Collect results from the multi-output op.
        let result_values: Vec<_> = (0..num_outputs)
            .map(|i| inline_ptx.deref(ctx).get_result(i))
            .collect();

        // Pack results into a tuple to match the destination type.
        let tuple_op = Operation::new(
            ctx,
            MirConstructTupleOp::get_concrete_op_info(),
            vec![dest_ty],
            result_values,
            vec![],
            0,
        );
        tuple_op.deref_mut(ctx).set_loc(loc.clone());
        tuple_op.insert_after(ctx, inline_ptx);
        let tuple_val = tuple_op.deref(ctx).get_result(0);

        emit_store_result_and_goto(
            ctx,
            destination,
            tuple_val,
            target,
            block_ptr,
            tuple_op,
            value_map,
            block_map,
            loc,
            "ptx_asm multi-output call without target block",
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
            convergent: true,
        }),
        REGISTER_ONLY_MAY_DIVERGE_OPTIONS => Ok(InlinePtxOptions {
            sideeffect: false,
            convergent: false,
        }),
        other => input_err!(
            loc,
            TranslationErr::unsupported(format!("unsupported ptx_asm options marker `{other}`"))
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_only_keeps_inline_ptx_convergent() {
        let options = parse_options(REGISTER_ONLY_OPTION, Location::Unknown).unwrap();

        assert!(!options.sideeffect);
        assert!(options.convergent);
    }

    #[test]
    fn may_diverge_opt_in_drops_convergent() {
        let options = parse_options(REGISTER_ONLY_MAY_DIVERGE_OPTIONS, Location::Unknown).unwrap();

        assert!(!options.sideeffect);
        assert!(!options.convergent);
    }

    #[test]
    fn count_output_constraints_single() {
        assert_eq!(count_output_constraints("=r,r,r"), 1);
    }

    #[test]
    fn count_output_constraints_multi() {
        assert_eq!(count_output_constraints("=r,=r,=f,=d,r,l"), 4);
    }

    #[test]
    fn count_output_constraints_none() {
        assert_eq!(count_output_constraints("r,l,~{memory}"), 0);
    }

    #[test]
    fn count_output_constraints_empty() {
        assert_eq!(count_output_constraints(""), 0);
    }
}
