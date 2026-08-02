/*
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end tests for bounded full unrolling.
//!
//! The bounded iterator transform must preserve every ordinary exit cloned from
//! the source loop. The only edge it synthesizes is the impossible backedge of
//! the final guard copy, which becomes `mir.unreachable`.

mod common;

use common::{early_exit_counted_loop, mir_ctx, multiple_exit_counted_loop};
use dialect_mir::ops::MirReturnOp;
use dialect_mir::ops::control_flow::MirBoundedUnrollHintOp;
use mir_transforms::analyses::loop_info::LoopInfo;
use mir_transforms::bounded_unroll::unroll_bounded_loops;
use pliron::context::{Context, Ptr};
use pliron::graph::dominance::DomInfo;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::pass::AnalysisManager;
use pliron::region::Region;

fn operations(ctx: &Context, region: Ptr<Region>) -> Vec<Ptr<Operation>> {
    region
        .deref(ctx)
        .iter(ctx)
        .flat_map(|block| block.deref(ctx).iter(ctx))
        .collect()
}

fn loop_count(ctx: &Context, region: Ptr<Region>) -> usize {
    let mut dom = DomInfo::default();
    let tree = dom.get_dom_tree(ctx, region);
    LoopInfo::compute(ctx, region, tree).loops().len()
}

fn hint_count(ctx: &Context, region: Ptr<Region>) -> usize {
    operations(ctx, region)
        .into_iter()
        .filter(|&operation| Operation::get_op::<MirBoundedUnrollHintOp>(operation, ctx).is_some())
        .count()
}

fn return_count(ctx: &Context, region: Ptr<Region>) -> usize {
    operations(ctx, region)
        .into_iter()
        .filter(|&operation| Operation::get_op::<MirReturnOp>(operation, ctx).is_some())
        .count()
}

#[test]
fn bounded_unroll_preserves_multiple_edges_to_a_shared_exit() {
    let mut ctx = mir_ctx();
    let loop_ir = early_exit_counted_loop(&mut ctx, 4, 2);

    assert_eq!(loop_count(&ctx, loop_ir.region), 1);

    MirBoundedUnrollHintOp::new(&mut ctx, 4)
        .get_operation()
        .insert_at_front(loop_ir.body, &ctx);

    let mut analyses = AnalysisManager::default();
    unroll_bounded_loops(loop_ir.module, &mut ctx, &mut analyses)
        .expect("bounded unroll with a shared exit succeeds");

    pliron::operation::verify_operation(loop_ir.module, &ctx)
        .expect("valid IR after bounded unroll with a shared exit");
    assert_eq!(loop_count(&ctx, loop_ir.region), 0);
    assert_eq!(hint_count(&ctx, loop_ir.region), 0);
    assert_eq!(return_count(&ctx, loop_ir.region), 1);
}

#[test]
fn bounded_unroll_preserves_multiple_distinct_exit_targets() {
    let mut ctx = mir_ctx();
    let loop_ir = multiple_exit_counted_loop(&mut ctx, 4);

    assert_eq!(loop_count(&ctx, loop_ir.region), 1);

    MirBoundedUnrollHintOp::new(&mut ctx, 4)
        .get_operation()
        .insert_at_front(loop_ir.check_a, &ctx);

    let mut analyses = AnalysisManager::default();
    unroll_bounded_loops(loop_ir.module, &mut ctx, &mut analyses)
        .expect("bounded unroll with distinct exits succeeds");

    pliron::operation::verify_operation(loop_ir.module, &ctx)
        .expect("valid IR after bounded unroll with distinct exits");
    assert_eq!(loop_count(&ctx, loop_ir.region), 0);
    assert_eq!(hint_count(&ctx, loop_ir.region), 0);
    assert_eq!(return_count(&ctx, loop_ir.region), 3);
}
