/*
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bounded full unrolling for loops whose maximum number of successful
//! iterations is known, even when their ordinary trip count is runtime-dependent.
//!
//! This is used for small fixed arrays consumed through iterator adapters.
//! The importer plants `mir.bounded_unroll_hint` in the block containing
//! `Iterator::next`. For a bound `N`, this pass clones the complete loop
//! iteration `N + 1` times. The extra copy executes the terminal `next()` that
//! returns `None`, preserving iterator state and ordinary loop-exit semantics.
//!
//! The transformation is deliberately conservative:
//!
//! - one preheader
//! - one canonical latch
//! - one or more ordinary loop exits
//! - loop-closed live-outs
//! - bounded code growth
//!
//! Unsupported shapes are left unchanged with a warning.

use crate::analyses::loop_info::LoopInfo;
use crate::canonicalize::{CanonicalizeOutcome, close_header_liveouts, merge_backedges};
use dialect_mir::ops::control_flow::{MirBoundedUnrollHintOp, MirGotoOp, MirUnreachableOp};
use dialect_mir::ops::function::MirFuncOp;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::graph::ControlFlowGraph;
use pliron::graph::dominance::DomInfo;
use pliron::irbuild::{
    cloning::{IrMapping, clone_blocks_into},
    listener::DummyListener,
    rewriter::IRRewriter,
};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::opts::constants::sccp::sccp;
use pliron::opts::dce::dce;
use pliron::opts::simplify_cfg::simplify_cfg;
use pliron::pass::AnalysisManager;
use pliron::region::Region;
use pliron::result::Result;
use pliron::value::Value;
use rustc_hash::FxHashSet;

const MAX_BOUNDED_COPIES: u64 = 17;
const MAX_CLONED_BLOCKS: u64 = 8_192;
const MAX_CLONED_OPS: u64 = 65_536;

pub fn unroll_bounded_loops(
    module: Ptr<Operation>,
    ctx: &mut Context,
    _analyses: &mut AnalysisManager,
) -> Result<()> {
    let has_hints = collect_functions(module, ctx).iter().any(|&function| {
        let region = function.deref(ctx).get_region(0);
        !collect_hints(ctx, region).is_empty()
    });
    if !has_hints {
        return Ok(());
    }

    let mut module_changed = false;

    for function in collect_functions(module, ctx) {
        let region = function.deref(ctx).get_region(0);
        if collect_hints(ctx, region).is_empty() {
            continue;
        }

        let mut function_changed = false;
        let mut skip_simplify_once = false;

        loop {
            if skip_simplify_once {
                skip_simplify_once = false;
            } else {
                simplify_cfg(function, ctx)?;
            }

            let hints = collect_hints(ctx, region);
            if hints.is_empty() {
                break;
            }

            let info = {
                let mut analyses = AnalysisManager::default();
                let mut dom_info = analyses.get_analysis_mut::<DomInfo>(module, ctx)?;
                let dom = dom_info.get_dom_tree(ctx, region);
                LoopInfo::compute(ctx, region, dom)
            };

            let mut selected: Option<(usize, usize, u32)> = None;
            for (_, block, max_iterations) in &hints {
                if let Some(loop_id) = info.innermost_loop(*block) {
                    let size = info.loops()[loop_id].blocks.len();
                    match selected {
                        None => selected = Some((loop_id, size, *max_iterations)),
                        Some((selected_id, selected_size, selected_max))
                            if loop_id == selected_id =>
                        {
                            selected = Some((
                                selected_id,
                                selected_size,
                                selected_max.max(*max_iterations),
                            ));
                        }
                        Some((_, selected_size, _)) if size < selected_size => {
                            selected = Some((loop_id, size, *max_iterations));
                        }
                        _ => {}
                    }
                }
            }

            let Some((loop_id, _, max_iterations)) = selected else {
                for (hint, _, _) in hints {
                    hint.unlink(ctx);
                }
                eprintln!(
                    "warning: bounded iterator unrolling requested, but the marker is not inside a recognizable loop"
                );
                break;
            };

            let Some(preheader) = info.preheader(ctx, region, loop_id) else {
                remove_loop_hints(ctx, &info, loop_id, &hints);
                eprintln!(
                    "warning: bounded iterator loop was not unrolled: it has no single preheader"
                );
                continue;
            };

            match close_header_liveouts(ctx, &info, loop_id) {
                CanonicalizeOutcome::Unchanged => {}
                CanonicalizeOutcome::Changed => {
                    function_changed = true;
                    module_changed = true;
                    skip_simplify_once = true;
                    continue;
                }
                CanonicalizeOutcome::Unsupported(reason) => {
                    remove_loop_hints(ctx, &info, loop_id, &hints);
                    eprintln!(
                        "warning: bounded iterator loop was not unrolled: could not close live-outs: {reason}"
                    );
                    continue;
                }
            }

            match merge_backedges(ctx, &info, loop_id) {
                CanonicalizeOutcome::Unchanged => {}
                CanonicalizeOutcome::Changed => {
                    function_changed = true;
                    module_changed = true;
                    skip_simplify_once = true;
                    continue;
                }
                CanonicalizeOutcome::Unsupported(reason) => {
                    remove_loop_hints(ctx, &info, loop_id, &hints);
                    eprintln!(
                        "warning: bounded iterator loop was not unrolled: could not canonicalize back-edges: {reason}"
                    );
                    continue;
                }
            }

            remove_loop_hints(ctx, &info, loop_id, &hints);

            match bounded_unroll(ctx, &info, region, loop_id, preheader, max_iterations)? {
                BoundedUnrollOutcome::Unrolled => {
                    function_changed = true;
                    module_changed = true;
                }
                BoundedUnrollOutcome::Skipped(reason) => {
                    eprintln!("warning: bounded iterator loop was not unrolled: {reason}");
                }
            }
        }

        if function_changed {
            sccp(function, ctx)?;
            simplify_cfg(function, ctx)?;
            dce(function, ctx)?;
        }
    }

    if module_changed {
        pliron::operation::verify_operation(module, ctx)?;
    }

    Ok(())
}

enum BoundedUnrollOutcome {
    Unrolled,
    Skipped(String),
}

struct BoundedLoopShape {
    header: Ptr<BasicBlock>,
    latch: Ptr<BasicBlock>,
    loop_blocks: FxHashSet<Ptr<BasicBlock>>,
    blocks_ordered: Vec<Ptr<BasicBlock>>,
    init_operands: Vec<Value>,
    recur_operands: Vec<Value>,
    preheader_term: Ptr<Operation>,
}

struct LoopCopy {
    header: Ptr<BasicBlock>,
    latch_term: Ptr<Operation>,
    next_running: Vec<Value>,
}

fn bounded_unroll(
    ctx: &mut Context,
    info: &LoopInfo,
    region: Ptr<Region>,
    loop_id: usize,
    preheader: Ptr<BasicBlock>,
    max_iterations: u32,
) -> Result<BoundedUnrollOutcome> {
    let shape = match analyze_shape(ctx, info, region, loop_id, preheader) {
        Ok(shape) => shape,
        Err(reason) => return Ok(BoundedUnrollOutcome::Skipped(reason)),
    };

    let copies = u64::from(max_iterations) + 1;

    if let Err(reason) = check_clone_budget(ctx, copies, &shape.blocks_ordered) {
        return Ok(BoundedUnrollOutcome::Skipped(reason));
    }

    if !liveouts_are_safe(ctx, &shape) {
        return Ok(BoundedUnrollOutcome::Skipped(
            "values defined in the loop are used directly outside it".to_string(),
        ));
    }

    let mut running = shape.init_operands.clone();
    let mut previous_tail = shape.preheader_term;

    for copy_index in 0..copies {
        let copy = clone_loop(ctx, region, &shape);
        rewire_goto(ctx, previous_tail, copy.header, &running);
        previous_tail = copy.latch_term;

        if copy_index + 1 < copies {
            running = copy.next_running;
        }
    }

    // The extra copy exists only to execute the terminal iterator check.
    // Reaching its latch would mean the loop exceeded the compiler-proven bound.
    replace_with_unreachable(ctx, previous_tail);

    Ok(BoundedUnrollOutcome::Unrolled)
}

fn analyze_shape(
    ctx: &Context,
    info: &LoopInfo,
    region: Ptr<Region>,
    loop_id: usize,
    preheader: Ptr<BasicBlock>,
) -> std::result::Result<BoundedLoopShape, String> {
    let loop_info = &info.loops()[loop_id];

    if loop_info.latches.len() != 1 {
        return Err(format!(
            "expected one canonical latch, found {}",
            loop_info.latches.len()
        ));
    }

    let header = loop_info.header;
    let latch = loop_info.latches[0];

    let preheader_term = preheader
        .deref(ctx)
        .get_terminator(ctx)
        .ok_or("the preheader has no terminator")?;
    if Operation::get_op::<MirGotoOp>(preheader_term, ctx).is_none()
        || preheader_term.deref(ctx).get_num_successors() != 1
        || preheader_term.deref(ctx).get_successor(0) != header
    {
        return Err("the preheader must be a direct mir.goto to the loop header".to_string());
    }

    let latch_term = latch
        .deref(ctx)
        .get_terminator(ctx)
        .ok_or("the latch has no terminator")?;
    if Operation::get_op::<MirGotoOp>(latch_term, ctx).is_none()
        || latch_term.deref(ctx).get_num_successors() != 1
        || latch_term.deref(ctx).get_successor(0) != header
    {
        return Err("the canonical latch must be a direct mir.goto to the loop header".to_string());
    }

    let header_args = header.deref(ctx).arguments().collect::<Vec<_>>();
    let init_operands = preheader_term.deref(ctx).operands().collect::<Vec<_>>();
    let recur_operands = latch_term.deref(ctx).operands().collect::<Vec<_>>();

    if init_operands.len() != header_args.len() || recur_operands.len() != header_args.len() {
        return Err("loop-carried operand count does not match the header arguments".to_string());
    }

    let mut exit_edge_count = 0usize;
    for &block in &loop_info.blocks {
        let Some(term) = block.deref(ctx).get_terminator(ctx) else {
            return Err("a loop block has no terminator".to_string());
        };
        exit_edge_count += term
            .deref(ctx)
            .successors()
            .filter(|successor| !loop_info.blocks.contains(successor))
            .count();
    }

    if exit_edge_count == 0 {
        return Err("the loop has no ordinary exit edge".to_string());
    }

    let blocks_ordered = reachable_loop_blocks(ctx, region, header, &loop_info.blocks);
    if blocks_ordered.len() != loop_info.blocks.len() {
        return Err("the loop body is irreducible or has multiple entries".to_string());
    }

    Ok(BoundedLoopShape {
        header,
        latch,
        loop_blocks: loop_info.blocks.clone(),
        blocks_ordered,
        init_operands,
        recur_operands,
        preheader_term,
    })
}

fn clone_loop(ctx: &mut Context, region: Ptr<Region>, shape: &BoundedLoopShape) -> LoopCopy {
    let mut mapping = IrMapping::new();
    let mut rewriter = IRRewriter::<DummyListener>::default();

    clone_blocks_into(
        &shape.blocks_ordered,
        region,
        ctx,
        &mut rewriter,
        &mut mapping,
    );

    let header = mapping.lookup_block_or_default(shape.header);
    let latch = mapping.lookup_block_or_default(shape.latch);
    let latch_term = latch
        .deref(ctx)
        .get_terminator(ctx)
        .expect("a cloned latch has a terminator");
    let next_running = shape
        .recur_operands
        .iter()
        .map(|&value| mapping.lookup_value_or_default(value))
        .collect();

    LoopCopy {
        header,
        latch_term,
        next_running,
    }
}

fn reachable_loop_blocks(
    ctx: &Context,
    region: Ptr<Region>,
    entry: Ptr<BasicBlock>,
    loop_blocks: &FxHashSet<Ptr<BasicBlock>>,
) -> Vec<Ptr<BasicBlock>> {
    let mut visited = FxHashSet::default();
    let mut ordered = Vec::new();
    let mut stack = vec![entry];

    while let Some(block) = stack.pop() {
        if !visited.insert(block) {
            continue;
        }
        ordered.push(block);

        for successor in region.successors(ctx, &block) {
            if loop_blocks.contains(&successor) && !visited.contains(&successor) {
                stack.push(successor);
            }
        }
    }

    ordered
}

fn liveouts_are_safe(ctx: &Context, shape: &BoundedLoopShape) -> bool {
    for &block in &shape.loop_blocks {
        for value in block.deref(ctx).arguments() {
            if has_direct_outside_use(ctx, value, &shape.loop_blocks) {
                return false;
            }
        }
        for operation in block.deref(ctx).iter(ctx).collect::<Vec<_>>() {
            for value in operation.deref(ctx).results() {
                if has_direct_outside_use(ctx, value, &shape.loop_blocks) {
                    return false;
                }
            }
        }
    }
    true
}

fn has_direct_outside_use(
    ctx: &Context,
    value: Value,
    loop_blocks: &FxHashSet<Ptr<BasicBlock>>,
) -> bool {
    value.uses(ctx).iter().any(|r#use| {
        r#use
            .user_op()
            .deref(ctx)
            .get_parent_block()
            .is_none_or(|block| !loop_blocks.contains(&block))
    })
}

fn check_clone_budget(
    ctx: &Context,
    copies: u64,
    blocks: &[Ptr<BasicBlock>],
) -> std::result::Result<(), String> {
    if copies > MAX_BOUNDED_COPIES {
        return Err(format!(
            "bounded unrolling would create {copies} copies; the safety limit is {MAX_BOUNDED_COPIES}"
        ));
    }

    let blocks_per_copy = u64::try_from(blocks.len())
        .map_err(|_| "the loop body is too large to budget safely".to_string())?;
    let total_blocks = copies
        .checked_mul(blocks_per_copy)
        .ok_or_else(|| "the cloned block count overflowed".to_string())?;
    if total_blocks > MAX_CLONED_BLOCKS {
        return Err(format!(
            "bounded unrolling would clone {total_blocks} blocks; the safety limit is {MAX_CLONED_BLOCKS}"
        ));
    }

    let mut operations_per_copy = 0u64;
    for &block in blocks {
        let operations = u64::try_from(block.deref(ctx).iter(ctx).count())
            .map_err(|_| "the loop has too many operations to budget safely".to_string())?;
        operations_per_copy = operations_per_copy
            .checked_add(operations)
            .ok_or_else(|| "the operation count overflowed".to_string())?;
    }

    let total_operations = copies
        .checked_mul(operations_per_copy)
        .ok_or_else(|| "the cloned operation count overflowed".to_string())?;
    if total_operations > MAX_CLONED_OPS {
        return Err(format!(
            "bounded unrolling would clone {total_operations} operations; the safety limit is {MAX_CLONED_OPS}"
        ));
    }

    Ok(())
}

fn replace_with_unreachable(ctx: &mut Context, terminator: Ptr<Operation>) {
    let block = terminator
        .deref(ctx)
        .get_parent_block()
        .expect("a loop latch terminator belongs to a block");

    // Erase the old branch through Pliron's ownership-aware API. This removes
    // both SSA operand uses and CFG successor uses before deallocating the op.
    Operation::erase(terminator, ctx);

    MirUnreachableOp::new(ctx)
        .get_operation()
        .insert_at_back(block, ctx);
}

fn rewire_goto(
    ctx: &mut Context,
    terminator: Ptr<Operation>,
    successor: Ptr<BasicBlock>,
    operands: &[Value],
) {
    Operation::replace_successor(terminator, ctx, 0, successor);

    let operand_count = terminator.deref(ctx).get_num_operands();
    for _ in 0..operand_count {
        Operation::remove_operand(terminator, ctx, 0);
    }
    for &operand in operands {
        Operation::push_operand(terminator, ctx, operand);
    }
}

fn remove_loop_hints(
    ctx: &mut Context,
    info: &LoopInfo,
    loop_id: usize,
    hints: &[(Ptr<Operation>, Ptr<BasicBlock>, u32)],
) {
    for (hint, block, _) in hints {
        if info.innermost_loop(*block) == Some(loop_id) {
            hint.unlink(ctx);
        }
    }
}

fn collect_functions(module: Ptr<Operation>, ctx: &Context) -> Vec<Ptr<Operation>> {
    let mut functions = Vec::new();
    let region = module.deref(ctx).get_region(0);
    let blocks = region.deref(ctx).iter(ctx).collect::<Vec<_>>();

    for block in blocks {
        for operation in block.deref(ctx).iter(ctx).collect::<Vec<_>>() {
            if Operation::get_op::<MirFuncOp>(operation, ctx).is_some() {
                functions.push(operation);
            }
        }
    }

    functions
}

fn collect_hints(
    ctx: &Context,
    region: Ptr<Region>,
) -> Vec<(Ptr<Operation>, Ptr<BasicBlock>, u32)> {
    let mut hints = Vec::new();
    let blocks = region.deref(ctx).iter(ctx).collect::<Vec<_>>();

    for block in blocks {
        for operation in block.deref(ctx).iter(ctx).collect::<Vec<_>>() {
            if let Some(hint) = Operation::get_op::<MirBoundedUnrollHintOp>(operation, ctx) {
                hints.push((operation, block, hint.max_iterations(ctx)));
            }
        }
    }

    hints
}
