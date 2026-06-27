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

use common::{
    counted_loop, counted_loop_from_step, early_exit_counted_loop, early_exit_with_direct_liveout,
    mir_ctx, multi_latch_counted_loop, multiple_exit_counted_loop, nested_counted_loop,
};
use dialect_mir::ops::{
    MirBitAndOp, MirCallOp, MirCondBranchOp, MirConstantOp, MirGeOp, MirReturnOp, MirUnrollHintOp,
};
use mir_transforms::unroll::unroll_annotated_loops;
use pliron::builtin::attributes::{IntegerAttr, StringAttr};
use pliron::builtin::ops::ConstantOp;
use pliron::context::{Context, Ptr};
use pliron::graph::{ControlFlowGraph, dominance::DomInfo};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::pass_manager::AnalysisManager;
use pliron::region::Region;

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

fn operations(ctx: &Context, region: Ptr<Region>) -> Vec<Ptr<Operation>> {
    region
        .deref(ctx)
        .iter(ctx)
        .flat_map(|block| block.deref(ctx).iter(ctx))
        .collect()
}

fn hint_count(ctx: &Context, region: Ptr<Region>) -> usize {
    operations(ctx, region)
        .into_iter()
        .filter(|&op| Operation::get_op::<MirUnrollHintOp>(op, ctx).is_some())
        .count()
}

fn cond_branch_count(ctx: &Context, region: Ptr<Region>) -> usize {
    operations(ctx, region)
        .into_iter()
        .filter(|&op| Operation::get_op::<MirCondBranchOp>(op, ctx).is_some())
        .count()
}

fn return_count(ctx: &Context, region: Ptr<Region>) -> usize {
    operations(ctx, region)
        .into_iter()
        .filter(|&op| Operation::get_op::<MirReturnOp>(op, ctx).is_some())
        .count()
}

fn constant_i128(ctx: &Context, value: pliron::value::Value) -> Option<i128> {
    let def = value.defining_op()?;
    if let Some(c) = Operation::get_op::<MirConstantOp>(def, ctx) {
        return c.get_attr_value(ctx).map(|a| a.value().to_i128());
    }
    Operation::get_op::<ConstantOp>(def, ctx)?
        .get_value(ctx)
        .downcast_ref::<IntegerAttr>()
        .map(|a| a.value().to_i128())
}

fn sole_return_constant(ctx: &Context, region: Ptr<Region>) -> Option<i128> {
    let returns: Vec<_> = operations(ctx, region)
        .into_iter()
        .filter(|&op| Operation::get_op::<MirReturnOp>(op, ctx).is_some())
        .collect();
    if returns.len() != 1 || returns[0].deref(ctx).get_num_operands() != 1 {
        return None;
    }
    constant_i128(ctx, returns[0].deref(ctx).get_operand(0))
}

fn loop_info(ctx: &Context, region: Ptr<Region>) -> LoopInfo {
    let mut dom = DomInfo::default();
    let dt = dom.get_dom_tree(ctx, region);
    LoopInfo::compute(ctx, region, dt)
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

/// Fully unrolling an OUTER loop that *contains* an inner loop clones the inner
/// loop wholesale once per outer iteration: the outer loop disappears, the inner
/// loop stays a loop, and there is one inner-loop copy per outer iteration. This
/// is the capability the `!children` bail used to forbid.
#[test]
fn nested_outer_full_unroll_clones_the_inner_loop() {
    let mut ctx = mir_ctx();
    let lp = nested_counted_loop(&mut ctx, 3, 2); // outer trip 3, inner trip 2

    assert_eq!(
        loop_count(&ctx, lp.region),
        2,
        "starts with outer + inner loop"
    );

    // Full-unroll marker on the OUTER loop (its body block dominating the inner).
    let hint = MirUnrollHintOp::new(&mut ctx, 0);
    hint.get_operation().insert_at_front(lp.outer_body, &ctx);

    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses).expect("nested outer unroll");

    // The transform must leave valid IR (it clones inner-loop blocks + back-edges
    // and recomputes loop structure; a bug here would corrupt the CFG).
    pliron::operation::verify_operation(lp.module, &ctx).expect("valid IR after nested unroll");

    let info = {
        let mut dom = DomInfo::default();
        let dt = dom.get_dom_tree(&ctx, lp.region);
        LoopInfo::compute(&ctx, lp.region, dt)
    };
    assert!(
        !info.loops().iter().any(|l| l.header == lp.outer_header),
        "the outer loop is gone (fully unrolled)"
    );
    assert_eq!(
        info.loops().len(),
        3,
        "the inner loop is cloned once per outer iteration (3 inner loops, 0 outer)"
    );
}

/// Fully unrolling only the inner loop removes that loop and leaves its outer
/// container intact.
#[test]
fn nested_inner_full_unroll_keeps_outer_loop() {
    let mut ctx = mir_ctx();
    let lp = nested_counted_loop(&mut ctx, 3, 2);

    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.inner_body, &ctx);

    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses).expect("nested inner unroll");

    pliron::operation::verify_operation(lp.module, &ctx)
        .expect("valid IR after unrolling the inner loop");
    let info = {
        let mut dom = DomInfo::default();
        let dt = dom.get_dom_tree(&ctx, lp.region);
        LoopInfo::compute(&ctx, lp.region, dt)
    };
    assert_eq!(info.loops().len(), 1, "only the outer loop should remain");
    assert_eq!(info.loops()[0].header, lp.outer_header);
}

/// Partially unrolling an outer loop that contains an inner loop is valid: the
/// outer loop becomes a main loop (stepping by N) plus a remainder, and each
/// copy carries its own clone of the inner loop. We assert the IR verifies and
/// that loops survive the transform (the gemm K-loop shape, minus the fold which
/// `unroll_smoke` checks numerically).
#[test]
fn nested_outer_partial_unroll_is_valid() {
    let mut ctx = mir_ctx();
    let lp = nested_counted_loop(&mut ctx, 8, 2);

    assert_eq!(loop_count(&ctx, lp.region), 2);

    let hint = MirUnrollHintOp::new(&mut ctx, 4); // partial unroll by 4
    hint.get_operation().insert_at_front(lp.outer_body, &ctx);

    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("nested outer partial unroll");

    pliron::operation::verify_operation(lp.module, &ctx)
        .expect("valid IR after nested partial unroll");
    let info = {
        let mut dom = DomInfo::default();
        let dt = dom.get_dom_tree(&ctx, lp.region);
        LoopInfo::compute(&ctx, lp.region, dt)
    };
    assert_eq!(
        info.loops().len(),
        7,
        "main outer + remainder outer + original inner + four cloned inner loops"
    );
    assert_eq!(info.top_level().len(), 2, "main loop + remainder loop");
    let mut child_counts: Vec<usize> = info
        .top_level()
        .iter()
        .map(|&id| info.loops()[id].children.len())
        .collect();
    child_counts.sort_unstable();
    assert_eq!(child_counts, [1, 4], "one and four nested inner loops");
}

/// When both loops are annotated, the driver must consume the inner hint first.
/// Otherwise cloning the outer body would duplicate the inner hint and either
/// unroll it repeatedly or leave marker operations behind.
#[test]
fn nested_inner_and_outer_full_unroll_innermost_first() {
    let mut ctx = mir_ctx();
    let lp = nested_counted_loop(&mut ctx, 3, 2);

    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.outer_body, &ctx);
    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.inner_body, &ctx);

    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("nested inner + outer unroll");

    pliron::operation::verify_operation(lp.module, &ctx)
        .expect("valid IR after unrolling both nested loops");
    assert_eq!(
        loop_count(&ctx, lp.region),
        0,
        "both fully unrolled loops should be gone"
    );
}

#[test]
fn full_unroll_handles_multi_latch_continue() {
    let mut ctx = mir_ctx();
    let lp = multi_latch_counted_loop(&mut ctx, 4, 1, 1);
    pliron::operation::verify_operation(lp.module, &ctx).expect("valid input IR");

    let before = loop_info(&ctx, lp.region);
    let id = before.innermost_loop(lp.header).unwrap();
    assert_eq!(before.loops()[id].latches.len(), 2);

    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.choose, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("full unroll of multi-latch loop");

    pliron::operation::verify_operation(lp.module, &ctx)
        .expect("valid IR after multi-latch full unroll");
    assert_eq!(loop_count(&ctx, lp.region), 0);
    assert_eq!(sole_return_constant(&ctx, lp.region), Some(5));
}

#[test]
fn partial_unroll_handles_multi_latch_continue() {
    let mut ctx = mir_ctx();
    let lp = multi_latch_counted_loop(&mut ctx, 5, 1, 1);

    MirUnrollHintOp::new(&mut ctx, 2)
        .get_operation()
        .insert_at_front(lp.choose, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("partial unroll of multi-latch loop");

    pliron::operation::verify_operation(lp.module, &ctx)
        .expect("valid IR after multi-latch partial unroll");
    assert_eq!(
        loop_count(&ctx, lp.region),
        2,
        "partial unroll should build a main loop and keep a remainder"
    );
}

#[test]
fn inconsistent_multi_latch_steps_are_skipped() {
    let mut ctx = mir_ctx();
    let lp = multi_latch_counted_loop(&mut ctx, 4, 1, 2);

    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.choose, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("inconsistent recurrence is a warning + skip");

    pliron::operation::verify_operation(lp.module, &ctx).expect("skipped loop remains valid");
    assert_eq!(loop_count(&ctx, lp.region), 1);
    assert_eq!(hint_count(&ctx, lp.region), 0, "the request was consumed");
}

#[test]
fn full_unroll_preserves_an_early_break() {
    let mut ctx = mir_ctx();
    let lp = early_exit_counted_loop(&mut ctx, 4, 2);
    pliron::operation::verify_operation(lp.module, &ctx).expect("valid input IR");

    let before = loop_info(&ctx, lp.region);
    let id = before.innermost_loop(lp.header).unwrap();
    assert_eq!(before.exiting_blocks(&ctx, lp.region, id).len(), 2);
    assert_eq!(before.exit_blocks(&ctx, lp.region, id), vec![lp.exit]);

    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.body, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("full unroll with early break");

    pliron::operation::verify_operation(lp.module, &ctx)
        .expect("valid IR after early-break full unroll");
    assert_eq!(loop_count(&ctx, lp.region), 0);
    assert_eq!(
        sole_return_constant(&ctx, lp.region),
        Some(3),
        "0 + 1 + 2, then the early break bypasses the fourth copy"
    );
}

#[test]
fn full_unroll_preserves_multiple_exit_targets() {
    let mut ctx = mir_ctx();
    let lp = multiple_exit_counted_loop(&mut ctx, 4);
    pliron::operation::verify_operation(lp.module, &ctx).expect("valid input IR");

    let before = loop_info(&ctx, lp.region);
    let id = before.innermost_loop(lp.header).unwrap();
    assert_eq!(before.exiting_blocks(&ctx, lp.region, id).len(), 3);
    assert_eq!(before.exit_blocks(&ctx, lp.region, id).len(), 3);

    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.check_a, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("full unroll with multiple exits");

    pliron::operation::verify_operation(lp.module, &ctx)
        .expect("valid IR after multi-exit full unroll");
    assert_eq!(loop_count(&ctx, lp.region), 0);
    assert_eq!(
        cond_branch_count(&ctx, lp.region),
        8,
        "two runtime break checks survive in each of four copies"
    );
    assert_eq!(
        lp.region.predecessors(&ctx, &lp.exit_a).len(),
        4,
        "each copy keeps its first early-exit edge"
    );
    assert_eq!(
        lp.region.predecessors(&ctx, &lp.exit_b).len(),
        4,
        "each copy keeps its second early-exit edge"
    );
    assert_eq!(return_count(&ctx, lp.region), 3);
}

#[test]
fn partial_unroll_with_extra_exits_is_skipped() {
    let mut ctx = mir_ctx();
    let lp = multiple_exit_counted_loop(&mut ctx, 4);

    MirUnrollHintOp::new(&mut ctx, 2)
        .get_operation()
        .insert_at_front(lp.check_a, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("partial multi-exit is a warning + skip");

    pliron::operation::verify_operation(lp.module, &ctx).expect("skipped loop remains valid");
    let after = loop_info(&ctx, lp.region);
    assert_eq!(after.loops().len(), 1, "no main loop was introduced");
    let id = after.innermost_loop(lp.header).unwrap();
    assert_eq!(after.loops()[id].latches, vec![lp.latch]);
    assert_eq!(after.exiting_blocks(&ctx, lp.region, id).len(), 3);
    assert_eq!(after.exit_blocks(&ctx, lp.region, id).len(), 3);
    assert_eq!(cond_branch_count(&ctx, lp.region), 3);
    assert_eq!(hint_count(&ctx, lp.region), 0, "the request was consumed");
}

#[test]
fn full_unroll_routes_a_direct_header_liveout_through_exit_arguments() {
    let mut ctx = mir_ctx();
    let lp = early_exit_with_direct_liveout(&mut ctx, 4);
    pliron::operation::verify_operation(lp.module, &ctx).expect("valid input IR");

    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.body, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("full unroll canonicalizes direct header live-outs");

    pliron::operation::verify_operation(lp.module, &ctx)
        .expect("valid IR after live-out canonicalization and unroll");
    assert_eq!(loop_count(&ctx, lp.region), 0);
    assert_eq!(hint_count(&ctx, lp.region), 0, "the request was consumed");
}

#[test]
fn full_unroll_skips_a_counter_that_wraps_before_exit() {
    let mut ctx = mir_ctx();
    let lp = counted_loop_from_step(&mut ctx, i64::from(u32::MAX) - 1, i64::from(u32::MAX), 2);

    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.latch, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("wrapping full-unroll request is a warning + skip");

    pliron::operation::verify_operation(lp.module, &ctx).expect("skipped loop remains valid");
    assert_eq!(loop_count(&ctx, lp.region), 1);
    assert_eq!(hint_count(&ctx, lp.region), 0);
}

#[test]
fn partial_unroll_guards_the_last_copy_against_wraparound() {
    let mut ctx = mir_ctx();
    let lp = counted_loop_from_step(&mut ctx, i64::from(u32::MAX) - 4, i64::from(u32::MAX), 1);

    MirUnrollHintOp::new(&mut ctx, 4)
        .get_operation()
        .insert_at_front(lp.latch, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("partial unroll near the unsigned boundary");

    pliron::operation::verify_operation(lp.module, &ctx)
        .expect("valid IR with a no-wrap group guard");
    assert_eq!(loop_count(&ctx, lp.region), 2);
    assert!(
        operations(&ctx, lp.region)
            .iter()
            .any(|&op| Operation::get_op::<MirGeOp>(op, &ctx).is_some()),
        "main-loop guard must compare last_iv >= current_iv"
    );
    assert!(
        operations(&ctx, lp.region)
            .iter()
            .any(|&op| Operation::get_op::<MirBitAndOp>(op, &ctx).is_some()),
        "main-loop guard must require both in-bounds and no-wrap"
    );
}

#[test]
fn side_effecting_loop_header_is_skipped() {
    let mut ctx = mir_ctx();
    let lp = counted_loop(&mut ctx, 4);

    let header_term = lp
        .header
        .deref(&ctx)
        .get_terminator(&ctx)
        .expect("loop header terminator");
    let side_effect = Operation::new(
        &mut ctx,
        MirCallOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    MirCallOp::new(side_effect).set_attr_callee(&ctx, StringAttr::new("header_effect".into()));
    side_effect.insert_before(&ctx, header_term);

    MirUnrollHintOp::new(&mut ctx, 0)
        .get_operation()
        .insert_at_front(lp.latch, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("a side-effecting header is a warning + skip");

    pliron::operation::verify_operation(lp.module, &ctx).expect("skipped loop remains valid");
    assert_eq!(loop_count(&ctx, lp.region), 1, "the source loop remains");
    assert_eq!(hint_count(&ctx, lp.region), 0, "the request was consumed");
}

#[test]
fn huge_partial_unroll_factor_is_skipped_before_cloning() {
    let mut ctx = mir_ctx();
    let lp = counted_loop(&mut ctx, 4);

    MirUnrollHintOp::new(&mut ctx, u32::MAX)
        .get_operation()
        .insert_at_front(lp.latch, &ctx);
    let mut analyses = AnalysisManager::default();
    unroll_annotated_loops(lp.module, &mut ctx, &mut analyses)
        .expect("an oversized partial-unroll request is a warning + skip");

    pliron::operation::verify_operation(lp.module, &ctx).expect("skipped loop remains valid");
    assert_eq!(loop_count(&ctx, lp.region), 1, "the source loop remains");
    assert_eq!(hint_count(&ctx, lp.region), 0, "the request was consumed");
}
