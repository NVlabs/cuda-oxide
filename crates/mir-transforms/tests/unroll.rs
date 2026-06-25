/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end tests for the unroll pass itself. Rather than poke at the
//! internal shape analysis, these build a counted loop, plant an `#[unroll]`
//! marker (`mir.unroll_hint`) in its body, run `unroll_annotated_loops`, and
//! check the result through the public `LoopInfo` analysis.
//!
//! The key observable: fully unrolling the only loop in a function leaves a
//! function with no loops at all.

mod common;

use common::{counted_loop, mir_ctx};
use dialect_mir::ops::MirUnrollHintOp;
use mir_transforms::unroll::unroll_annotated_loops;
use pliron::graph::dominance::DomInfo;
use pliron::op::Op;
use pliron::pass_manager::AnalysisManager;

use mir_transforms::analyses::loop_info::LoopInfo;

/// How many natural loops are left in `lp`'s region.
fn loop_count(
    ctx: &pliron::context::Context,
    region: pliron::context::Ptr<pliron::region::Region>,
) -> usize {
    let mut dom = DomInfo::default();
    let dt = dom.get_dom_tree(ctx, region);
    LoopInfo::compute(ctx, region, dt).loops().len()
}

/// A full-unroll hint on a constant-trip loop deletes the loop entirely:
/// `while i < 4 { .. }` becomes four straight-line copies with no back-edge.
#[test]
fn full_unroll_removes_the_loop() {
    let mut ctx = mir_ctx();
    let lp = counted_loop(&mut ctx, 4); // while i < 4 -> trip count 4

    assert_eq!(loop_count(&ctx, lp.region), 1, "starts with one loop");

    // Plant a full-unroll marker (factor 0 = full) in the loop body.
    let hint = MirUnrollHintOp::new(&mut ctx, 0);
    hint.get_operation().insert_at_front(lp.latch, &ctx);

    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses).expect("unroll pass succeeds");

    assert_eq!(
        loop_count(&ctx, lp.region),
        0,
        "fully unrolling the only loop should leave no loop"
    );
}

/// With no `#[unroll]` marker the pass is a no-op: the loop is left intact.
#[test]
fn no_hint_leaves_the_loop_intact() {
    let mut ctx = mir_ctx();
    let lp = counted_loop(&mut ctx, 4);

    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses).expect("unroll pass succeeds");

    assert_eq!(
        loop_count(&ctx, lp.region),
        1,
        "no hint => the loop is untouched"
    );
}
