/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Annotation-driven loop unrolling over `dialect-mir`.
//!
//! Driven by a `mir.unroll` attribute (`UnrollAttr`) on a function op:
//! `0` = full unroll of a constant-trip-count loop, `n` = unroll by `n` with a
//! remainder loop (Step 4, in progress). Built on the reusable
//! [`LoopInfo`](crate::loop_info) + [`induction`](crate::induction) analyses and
//! pliron's IR cloning (`pliron::irbuild::cloning`).

use dialect_mir::attributes::UnrollAttr;
use dialect_mir::ops::constants::MirConstantOp;
use dialect_mir::ops::function::MirFuncOp;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::types::IntegerType;
use pliron::common_traits::Named;
use pliron::context::{Context, Ptr};
use pliron::graph::dominance::DomInfo;
use pliron::identifier::Identifier;
use pliron::irbuild::cloning::{IrMapping, clone_operation};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::opts::simplify_cfg::simplify_cfg;
use pliron::pass_manager::AnalysisManager;
use pliron::printable::Printable;
use pliron::r#type::{TypeHandle, Typed, TypedHandle};
use pliron::region::Region;
use pliron::result::Result;
use pliron::utils::apint::APInt;
use pliron::value::Value;
use rustc_hash::FxHashSet;
use std::num::NonZero;

use crate::induction::{self, ArgKind};
use crate::loop_info::LoopInfo;

/// Attribute-dict key under which `#[unroll]` plumbing stores the `UnrollAttr`.
const UNROLL_ATTR_KEY: &str = "unroll";

fn verbose() -> bool {
    std::env::var("CUDA_OXIDE_VERBOSE").is_ok()
}

/// Run annotation-driven loop unrolling over every function in `module` that
/// carries a `mir.unroll` attribute. A no-op for functions without it.
pub fn unroll_annotated_loops(
    module: Ptr<Operation>,
    ctx: &mut Context,
    analyses: &mut AnalysisManager,
) -> Result<()> {
    let key: Identifier = UNROLL_ATTR_KEY.try_into().expect("valid identifier");
    let annotated = collect_annotated_functions(module, ctx, &key);
    if annotated.is_empty() {
        return Ok(());
    }

    let mut changed = false;
    for (func_op, factor) in annotated {
        let region = func_op.deref(ctx).get_region(0);
        let info = {
            let mut dom_info = analyses.get_analysis_mut::<DomInfo>(module, ctx)?;
            let dom = dom_info.get_dom_tree(ctx, region);
            LoopInfo::compute(ctx, region, dom)
        };

        // Collect each loop's preheader + IV analysis up front.
        let mut targets = Vec::new();
        for id in 0..info.loops().len() {
            if let Some(ph) = info.preheader(ctx, region, id) {
                let rec = induction::analyze(ctx, &info, id, ph);
                targets.push((id, ph, rec));
            }
        }

        if verbose() {
            log_loops(ctx, region, &info, func_op, factor, &targets);
        }

        for (id, ph, rec) in &targets {
            if factor == 0 {
                // Full unroll: needs a constant trip count.
                if full_unroll(ctx, &info, region, *id, *ph, rec)? {
                    changed = true;
                }
            }
            // factor > 0 (partial unroll + remainder) is Step 4.
        }
    }

    if changed {
        // Delete the now-unreachable original loop blocks and fold any
        // guards left constant. Reusable pliron CFG cleanup.
        simplify_cfg(module, ctx)?;
    }
    Ok(())
}

/// Fully unroll a constant-trip-count counted loop.
///
/// Handles the canonical post-mem2reg shape: a header carrying the loop's
/// block-arguments and an exit guard, plus a single latch block holding the
/// body and the back-edge. Bails (returns `Ok(false)`, leaving IR untouched)
/// for anything outside that shape.
///
/// For trip count `T` it lays `T` straight-line copies of the body into the
/// preheader, substituting the induction variable with the literal value of
/// that iteration (`init + k*step`) and threading the other carried values copy
/// to copy. The preheader's back-edge is then redirected to the loop exit and
/// the original header/latch become unreachable (removed by `simplify_cfg`).
fn full_unroll(
    ctx: &mut Context,
    info: &LoopInfo,
    region: Ptr<Region>,
    id: usize,
    preheader: Ptr<BasicBlock>,
    rec: &induction::LoopRecurrences,
) -> Result<bool> {
    let l = &info.loops()[id];
    // Canonical shape only: a flat (non-nested) loop of exactly {header, latch}.
    if l.parent.is_some() || !l.children.is_empty() {
        return Ok(false);
    }
    if l.latches.len() != 1 || l.blocks.len() != 2 {
        return Ok(false);
    }
    let header = l.header;
    let latch = l.latches[0];
    let loop_blocks: FxHashSet<Ptr<BasicBlock>> = l.blocks.clone();

    let exits = info.exit_blocks(ctx, region, id);
    if exits.len() != 1 {
        return Ok(false);
    }
    let exit = exits[0];

    let (iv_idx, trip) = match (rec.primary_iv, rec.trip_count) {
        (Some(i), Some(t)) => (i, t),
        _ => return Ok(false),
    };
    let (iv_init, iv_step) = match &rec.args[iv_idx] {
        ArgKind::BasicIv { init, step } => (*init, *step),
        _ => return Ok(false),
    };

    let nargs = header.deref(ctx).get_num_arguments();
    let header_args: Vec<Value> = (0..nargs).map(|i| header.deref(ctx).get_argument(i)).collect();
    let iv_type = header_args[iv_idx].get_type(ctx);

    let init_ops = match induction::edge_operands(ctx, preheader, header) {
        Some(v) if v.len() == nargs => v,
        _ => return Ok(false),
    };
    let recur_ops = match induction::edge_operands(ctx, latch, header) {
        Some(v) if v.len() == nargs => v,
        _ => return Ok(false),
    };
    let exit_ops_raw = induction::edge_operands(ctx, header, exit).unwrap_or_default();

    let p_term = match preheader.deref(ctx).get_terminator(ctx) {
        Some(t) => t,
        None => return Ok(false),
    };
    let latch_term = match latch.deref(ctx).get_terminator(ctx) {
        Some(t) => t,
        None => return Ok(false),
    };
    // The loop body to replicate: the latch's ops minus its back-edge terminator.
    let body_ops: Vec<Ptr<Operation>> = latch
        .deref(ctx)
        .iter(ctx)
        .filter(|&op| op != latch_term)
        .collect();

    // Running carried value per header arg; starts at the preheader's init.
    let mut running: Vec<Value> = init_ops;

    for k in 0..trip {
        let mut mapper = IrMapping::new();
        // The induction variable is the literal value at iteration k.
        let iv_val = make_const(ctx, iv_type, iv_init + (k as i128) * iv_step, p_term);
        for a in 0..nargs {
            let mapped = if a == iv_idx { iv_val } else { running[a] };
            mapper.map_value(header_args[a], mapped);
        }
        for &op in &body_ops {
            let cloned = clone_operation(op, ctx, &mut mapper);
            cloned.insert_before(ctx, p_term);
        }
        // Next iteration's carried values are this copy's recurrence results.
        for a in 0..nargs {
            if a != iv_idx {
                running[a] = mapper.lookup_value_or_default(recur_ops[a]);
            }
        }
    }

    // If the exit consumes the final IV value, materialise it (init + T*step).
    if exit_ops_raw.iter().any(|&v| v == header_args[iv_idx]) {
        running[iv_idx] = make_const(ctx, iv_type, iv_init + (trip as i128) * iv_step, p_term);
    }

    // Redirect uses of the header args that live OUTSIDE the loop (e.g. the exit
    // block's use of the final accumulator) to the unrolled running values. Uses
    // inside the loop are left alone; those blocks become dead and are removed.
    for a in 0..nargs {
        let replacement = running[a];
        header_args[a].replace_some_uses_with(
            ctx,
            |ctx, u| match u.user_op().deref(ctx).get_parent_block() {
                Some(b) => !loop_blocks.contains(&b),
                None => true,
            },
            &replacement,
        );
    }

    // Redirect the preheader's back-edge: `goto header(init...)` -> `goto exit`.
    let exit_ops: Vec<Value> = exit_ops_raw
        .iter()
        .map(|&v| match header_args.iter().position(|&h| h == v) {
            Some(a) => running[a],
            None => v,
        })
        .collect();
    Operation::replace_successor(p_term, ctx, 0, exit);
    let n_existing = p_term.deref(ctx).get_num_operands();
    for _ in 0..n_existing {
        Operation::remove_operand(p_term, ctx, 0);
    }
    for v in exit_ops {
        Operation::push_operand(p_term, ctx, v);
    }

    Ok(true)
}

/// Create a `mir.constant` of integer type `ty` holding `value`, inserted just
/// before `before`, and return its result value.
fn make_const(ctx: &mut Context, ty: TypeHandle, value: i128, before: Ptr<Operation>) -> Value {
    let typed = TypedHandle::<IntegerType>::from_handle(ty, ctx).expect("IV is an integer type");
    let width = typed.deref(ctx).width() as usize;
    let apint = APInt::from_i64(value as i64, NonZero::new(width).expect("non-zero width"));
    let attr = IntegerAttr::new(typed, apint);
    let op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![ty],
        vec![],
        vec![],
        0,
    );
    MirConstantOp::new(op).set_attr_value(ctx, attr);
    op.insert_before(ctx, before);
    op.deref(ctx).get_result(0)
}

/// Find function ops carrying an `UnrollAttr`, returning each with its factor.
fn collect_annotated_functions(
    module: Ptr<Operation>,
    ctx: &Context,
    key: &Identifier,
) -> Vec<(Ptr<Operation>, u32)> {
    let mut out = Vec::new();
    let module_region = module.deref(ctx).get_region(0);
    let blocks: Vec<Ptr<BasicBlock>> = module_region.deref(ctx).iter(ctx).collect();
    for block in blocks {
        let ops: Vec<Ptr<Operation>> = block.deref(ctx).iter(ctx).collect();
        for op in ops {
            if Operation::get_op::<MirFuncOp>(op, ctx).is_none() {
                continue;
            }
            if let Some(factor) = op.deref(ctx).attributes.get::<UnrollAttr>(key).map(|a| a.0) {
                out.push((op, factor));
            }
        }
    }
    out
}

/// Diagnostic dump of detected loops + IV analysis (gated on `CUDA_OXIDE_VERBOSE`).
fn log_loops(
    ctx: &Context,
    region: Ptr<Region>,
    info: &LoopInfo,
    func_op: Ptr<Operation>,
    factor: u32,
    targets: &[(usize, Ptr<BasicBlock>, induction::LoopRecurrences)],
) {
    let kind = if factor == 0 {
        "full".to_string()
    } else {
        format!("by {factor}")
    };
    eprintln!(
        "\n=== loop-unroll: {} loop(s) in a #[unroll({kind})] function, op {} ===",
        info.loops().len(),
        Operation::get_opid(func_op, ctx).disp(ctx),
    );
    for (id, _ph, rec) in targets {
        let l = &info.loops()[*id];
        eprintln!(
            "  loop#{id}: header={} latch={:?} exit={:?} trip_count={:?} primary_iv={:?}",
            l.header.unique_name(ctx),
            l.latches.iter().map(|b| b.unique_name(ctx)).collect::<Vec<_>>(),
            info.exit_blocks(ctx, region, *id)
                .iter()
                .map(|b| b.unique_name(ctx))
                .collect::<Vec<_>>(),
            rec.trip_count,
            rec.primary_iv,
        );
    }
}
