/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Annotation-driven loop unrolling over `dialect-mir`.
//!
//! Driven by a `mir.unroll` attribute (`UnrollAttr`) on a function op:
//! `0` = full unroll of a constant-trip-count loop, `n` = unroll by `n` with a
//! remainder loop. Built on the reusable [`LoopInfo`](crate::loop_info) analysis
//! and pliron's IR cloning (`pliron::irbuild::cloning`).

use dialect_mir::attributes::UnrollAttr;
use dialect_mir::ops::function::MirFuncOp;
use pliron::basic_block::BasicBlock;
use pliron::common_traits::Named;
use pliron::context::{Context, Ptr};
use pliron::graph::dominance::DomInfo;
use pliron::identifier::Identifier;
use pliron::linked_list::ContainsLinkedList;
use pliron::operation::Operation;
use pliron::pass_manager::AnalysisManager;
use pliron::printable::Printable;
use pliron::result::Result;

use crate::loop_info::LoopInfo;

/// Attribute-dict key under which `#[unroll]` plumbing stores the `UnrollAttr`
/// on a function op. (The attribute is named `mir.unroll`; pliron keys the dict
/// by the bare attribute name.)
const UNROLL_ATTR_KEY: &str = "unroll";

fn verbose() -> bool {
    std::env::var("CUDA_OXIDE_VERBOSE").is_ok()
}

/// Run annotation-driven loop unrolling over every function in `module` that
/// carries a `mir.unroll` attribute.
///
/// A no-op for functions without the attribute, so emitted IR is identical to
/// today unless `#[unroll]` / `#[unroll(N)]` is present.
pub fn unroll_annotated_loops(
    module: Ptr<Operation>,
    ctx: &mut Context,
    analyses: &mut AnalysisManager,
) -> Result<()> {
    let key: Identifier = UNROLL_ATTR_KEY.try_into().expect("valid identifier");

    // Collect (function op, unroll factor) for annotated functions up front,
    // so we are not holding an IR borrow while computing analyses.
    let annotated = collect_annotated_functions(module, ctx, &key);
    if annotated.is_empty() {
        return Ok(());
    }

    let mut dom_info = analyses.get_analysis_mut::<DomInfo>(module, ctx)?;
    for (func_op, factor) in annotated {
        let region = func_op.deref(ctx).get_region(0);
        let dom = dom_info.get_dom_tree(ctx, region);
        let info = LoopInfo::compute(ctx, region, dom);

        if verbose() {
            log_loops(ctx, region, &info, func_op, factor);
        }

        // Step 3+ (the transform) plugs in here: for each annotated loop, run
        // full unroll (factor 0) or partial-by-N + remainder, using the loop's
        // header/latch/body from `info` and the induction-variable analysis.
    }

    Ok(())
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
            let factor = op
                .deref(ctx)
                .attributes
                .get::<UnrollAttr>(key)
                .map(|a| a.0);
            if let Some(factor) = factor {
                out.push((op, factor));
            }
        }
    }
    out
}

/// Diagnostic dump of the detected loops (gated on `CUDA_OXIDE_VERBOSE`).
fn log_loops(
    ctx: &Context,
    region: Ptr<pliron::region::Region>,
    info: &LoopInfo,
    func_op: Ptr<Operation>,
    factor: u32,
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
    for (i, l) in info.loops().iter().enumerate() {
        let body: Vec<String> = l
            .blocks
            .iter()
            .map(|b| b.unique_name(ctx).to_string())
            .collect();
        let latches: Vec<String> = l
            .latches
            .iter()
            .map(|b| b.unique_name(ctx).to_string())
            .collect();
        let preheader = info
            .preheader(ctx, region, i)
            .map(|b| b.unique_name(ctx).to_string())
            .unwrap_or_else(|| "<none/ambiguous>".to_string());
        let exits: Vec<String> = info
            .exit_blocks(ctx, region, i)
            .iter()
            .map(|b| b.unique_name(ctx).to_string())
            .collect();
        eprintln!(
            "  loop#{i}: header={} latches={:?} preheader={preheader} exits={:?} body={:?} parent={:?}",
            l.header.unique_name(ctx),
            latches,
            exits,
            body,
            l.parent,
        );
        if let Some(ph) = info.preheader(ctx, region, i) {
            let rec = crate::induction::analyze(ctx, info, i, ph);
            eprintln!(
                "    iv: primary_arg={:?} bound={:?} continue_pred={:?} trip_count={:?}",
                rec.primary_iv, rec.bound, rec.continue_pred, rec.trip_count,
            );
            for (ai, kind) in rec.args.iter().enumerate() {
                eprintln!("      header arg#{ai}: {kind:?}");
            }
        }
    }
}
