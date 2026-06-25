/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Loop unrolling, switched on by a `#[unroll]` annotation.
//!
//! Unrolling means making copies of a loop body so the loop runs fewer times
//! (or not at all), trading bigger code for less per-iteration overhead and more
//! chances to optimise. For example:
//!
//! ```text
//!   for i in 0..4 { f(i) }      unrolled fully becomes:   f(0); f(1); f(2); f(3);
//! ```
//!
//! The user asks for it with `#[unroll]` or `#[unroll(N)]`, which the frontend
//! records as a `mir.unroll` attribute (`UnrollAttr`) on the function. A factor
//! of `0` means **full unroll**: replace a loop whose iteration count is known
//! at compile time with that many straight-line copies of the body, no loop
//! left. A factor of `N` (>= 2) means **partial unroll**: do `N` body copies per
//! trip and add a small leftover ("remainder") loop for the iterations that
//! don't divide evenly.
//!
//! This pass is the reference example for writing an optimisation pass in oxide.
//! It builds on two reusable analyses, [`LoopInfo`](crate::analyses::loop_info)
//! (finds the loops) and [`induction`] (finds the counters and how many times
//! each loop runs), and on pliron's IR cloning (`pliron::irbuild::cloning`) to
//! duplicate the body.

use dialect_mir::ops::arithmetic::{MirAddOp, MirBitAndOp, MirRemOp, MirSubOp};
use dialect_mir::ops::comparison::{MirGeOp, MirGtOp, MirLeOp, MirLtOp};
use dialect_mir::ops::constants::MirConstantOp;
use dialect_mir::ops::control_flow::{MirCondBranchOp, MirUnrollHintOp};
use dialect_mir::ops::function::MirFuncOp;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::op_interfaces::OperandSegmentInterface;
use pliron::builtin::types::{IntegerType, Signedness};
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
use pliron::pass_manager::AnalysisManager;
use pliron::region::Region;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed, TypedHandle};
use pliron::utils::apint::APInt;
use pliron::value::Value;
use rustc_hash::{FxHashMap, FxHashSet};
use std::num::NonZero;

use crate::analyses::induction::{self, ArgKind, CmpPred};
use crate::analyses::loop_info::LoopInfo;

fn verbose() -> bool {
    std::env::var("CUDA_OXIDE_VERBOSE").is_ok()
}

/// The pass entry point. Unrolls each loop that carries an in-body
/// `mir.unroll_hint` (planted by `#[unroll]` / `#[unroll(N)]` written on that
/// loop), and leaves every other loop and function untouched.
///
/// Per function: find the loops (`LoopInfo`), map each hint to the loop whose
/// body it sits in, remove the hint, analyse that loop's counter, and full- or
/// partial-unroll by the hint's factor. Afterwards two stock pliron passes clean
/// up: `dce` drops what unrolling makes unused (the `& mask` ops we folded to
/// constants, full unroll's now-dead counter increments), and `simplify_cfg`
/// deletes the original loop blocks once they are unreachable.
pub fn unroll_annotated_loops(
    module: Ptr<Operation>,
    ctx: &mut Context,
    // The caller threads in the manager mem2reg used. We deliberately do not use
    // it: the CFG-normalization step below changes block structure, which would
    // stale that manager's cached dominator trees. We build a fresh manager
    // afterwards instead. (Kept in the signature to match pliron's pass shape.)
    _analyses: &mut AnalysisManager,
) -> Result<()> {
    // Nothing annotated => leave every function byte-for-byte untouched.
    let has_hints = collect_functions(module, ctx)
        .iter()
        .any(|&f| !collect_hints(ctx, f.deref(ctx).get_region(0)).is_empty());
    if !has_hints {
        return Ok(());
    }

    // Normalize the CFG first. `#[unroll]` is carried into the IR as a marker
    // call planted at the top of the loop body; Rust MIR turns that call into
    // its own basic block, so a plain counted loop arrives here looking like
    // three blocks (header, a one-line marker block, body) instead of the two
    // the transform handles. `simplify_cfg` merges that marker block back into
    // the body (a block whose only successor has it as the only predecessor),
    // restoring the simple shape. The `mir.unroll_hint` op rides along into the
    // merged block; we remove it below, before cloning the body.
    simplify_cfg(module, ctx)?;

    // `simplify_cfg` rewrote the CFG, so dominator info cached by an earlier
    // pass is no longer valid. Compute the loop analyses from a clean manager.
    let mut analyses = AnalysisManager::default();

    let mut changed = false;
    for func_op in collect_functions(module, ctx) {
        let region = func_op.deref(ctx).get_region(0);

        // What loops did the author annotate? (op, containing block, factor)
        let hints = collect_hints(ctx, region);
        if hints.is_empty() {
            continue;
        }

        let info = {
            let mut dom_info = analyses.get_analysis_mut::<DomInfo>(module, ctx)?;
            let dom = dom_info.get_dom_tree(ctx, region);
            LoopInfo::compute(ctx, region, dom)
        };

        // Map each hint to the loop whose body it sits in, then remove the hint
        // so it is not copied when we clone the body and never reaches lowering.
        let mut loop_factor: FxHashMap<usize, u32> = FxHashMap::default();
        for (hint_op, block, factor) in &hints {
            if let Some(loop_id) = info.innermost_loop(*block) {
                loop_factor.entry(loop_id).or_insert(*factor);
            }
            hint_op.unlink(ctx);
        }
        // Process loops in a stable order (by id) so diagnostics are deterministic.
        let mut loop_factor: Vec<(usize, u32)> = loop_factor.into_iter().collect();
        loop_factor.sort_by_key(|&(loop_id, _)| loop_id);

        // We computed `info` once, above, and reuse it while unrolling each loop
        // below even though each unroll mutates the CFG. That is sound here: a
        // full/partial unroll only rewires its own loop's preheader and clones
        // its own (disjoint) blocks onto the end of the region; the original
        // blocks are not deleted until the `dce` + `simplify_cfg` after this
        // whole loop. So another hinted loop's `Loop` snapshot stays valid, and
        // the per-loop queries (`preheader`, `exiting_blocks`, `exit_blocks`)
        // re-read the live CFG. A loop that contains another is never unrolled
        // (the `!children.is_empty()` bail in `analyze_shape`), so an outer and
        // its inner are never both transformed against the same stale `info`.
        for (loop_id, factor) in loop_factor {
            // How the user spelled the request, for diagnostics.
            let kind = if factor == 0 {
                "#[unroll]".to_string()
            } else {
                format!("#[unroll({factor})]")
            };
            let Some(ph) = info.preheader(ctx, region, loop_id) else {
                // The author asked for unrolling; never silently do nothing.
                eprintln!(
                    "warning: {kind} requested but the loop was not unrolled: it has no single preheader (it is entered from more than one place)"
                );
                continue;
            };
            let rec = induction::analyze(ctx, &info, loop_id, ph);
            if verbose() {
                eprintln!(
                    "loop-unroll: loop#{loop_id} factor={factor} trip={:?} primary_iv={:?}",
                    rec.trip_count, rec.primary_iv,
                );
            }
            let outcome = if factor == 0 {
                // Full unroll: only works when the trip count is a constant.
                full_unroll(ctx, &info, region, loop_id, ph, &rec)?
            } else {
                // Partial unroll by `factor`, with a remainder loop for the tail.
                partial_unroll(ctx, &info, region, loop_id, ph, &rec, factor)?
            };
            match outcome {
                UnrollOutcome::Unrolled => changed = true,
                // Requested but unsupported shape: report exactly why, loudly, so
                // it is never a silent no-op.
                UnrollOutcome::Skipped(reason) => {
                    eprintln!("warning: {kind} requested but the loop was not unrolled: {reason}");
                }
            }
        }
    }

    if changed {
        // Constant-fold in our own middle-end so the unrolled index math and the
        // match-on-a-constant-index collapse are guaranteed regardless of the
        // backend optimiser (`opt -O2` today, NVVM later) rather than left to it.
        // `sccp` folds the now-constant index arithmetic (a full-unrolled counter
        // is a literal) and infers constant branch conditions; `simplify_cfg`
        // collapses those branches into unconditional `mir.goto`; `dce` removes
        // the ops the fold left dead. This is scoped to unrolled functions (gated
        // on `changed`), so non-unrolled kernels are untouched. Inferred constants
        // are materialised as `builtin.constant`, which lowering passes through
        // unchanged and the textual exporter emits.
        sccp(module, ctx)?;
        simplify_cfg(module, ctx)?;
        dce(module, ctx)?;
        // Self-check: the input was already verified before this pass, so any
        // verification failure here is a bug in the unroller, not the user's
        // code. Surface it loudly rather than letting malformed IR slip into
        // lowering. (Runs after `simplify_cfg`, which deletes the now-unreachable
        // original loop blocks, so the dominator-based verifier sees a clean CFG.)
        pliron::operation::verify_operation(module, ctx)?;
    }
    Ok(())
}

/// What happened when we tried to unroll one loop.
enum UnrollOutcome {
    /// The loop was unrolled; the IR changed.
    Unrolled,
    /// The loop was left unchanged, with a plain-English reason. The caller turns
    /// this into a loud warning (the author asked for unrolling, so we never just
    /// silently do nothing).
    Skipped(String),
}

/// The facts the unroller needs about a loop, gathered once and shared by full
/// and partial unroll, plus the checks that the loop is in the shape we support.
///
/// v1 supports a loop with **arbitrary internal control flow** (the body may be
/// many blocks: `if`/`else`, `match`, `&&`/`||`), as long as it is otherwise
/// simple: one back-edge (latch), the only way out is the header's exit test
/// (no early `break`), one block after the loop (exit), a recognized counter, a
/// dedicated preheader, and no nested loops. `analyze_shape` returns `Err(reason)`
/// for anything else, so a requested-but-impossible unroll reports exactly why.
///
/// We deliberately do **not** clone the header. The header holds the loop's
/// carried values as block arguments (the counter and any accumulators); the
/// body reads those by dominance. Cloning only the body and substituting those
/// arguments per copy keeps the counter a literal in each copy (the property that
/// lets `i & 3` fold). A header that *computes* a value the body reads would
/// break that, so we reject it (`Err`) and leave header-cloning to a follow-up.
struct LoopShape {
    header: Ptr<BasicBlock>,
    latch: Ptr<BasicBlock>,
    exit: Ptr<BasicBlock>,
    /// The header's single in-loop successor: the first block of the body.
    body_entry: Ptr<BasicBlock>,
    /// Every body block (the loop minus the header) forward-reachable from
    /// `body_entry`, in a deterministic visit order. `clone_blocks_into` is
    /// order-independent, so the order is not load-bearing; we keep a stable
    /// order only for readable IR dumps. Its length (vs the body-block count) is
    /// what detects unreachable / irreducible body blocks (see
    /// [`reachable_body_blocks`]).
    body_blocks_ordered: Vec<Ptr<BasicBlock>>,
    /// The header's block arguments (the loop-carried values, counter included).
    header_args: Vec<Value>,
    nargs: usize,
    /// preheader -> header operands (the loop's initial carried values).
    init_ops: Vec<Value>,
    /// latch -> header operands (the updated carried values each iteration).
    recur_ops: Vec<Value>,
    /// header -> body_entry operands (args the header passes into the body).
    entry_ops: Vec<Value>,
    /// header -> exit operands (the loop's live-out values).
    exit_ops: Vec<Value>,
    iv_idx: usize,
    iv_init: i128,
    iv_step: i128,
    iv_type: TypeHandle,
    /// The boolean type of the header's exit test (for any new comparison).
    i1_type: TypeHandle,
    /// The preheader's terminator (a plain branch to the header).
    preheader_term: Ptr<Operation>,
    /// Every block of the loop (header included). Used to tell loop-internal uses
    /// of the carried values from out-of-loop (live-out) uses.
    loop_blocks: FxHashSet<Ptr<BasicBlock>>,
}

/// True if `v` is the result of an operation located in one of `set`'s blocks
/// (a block argument, having no defining op, is not "defined in" the set this
/// way). Used to tell loop-variant values from loop-invariant ones.
fn defined_in_loop(ctx: &Context, v: Value, set: &FxHashSet<Ptr<BasicBlock>>) -> bool {
    v.defining_op()
        .and_then(|d| d.deref(ctx).get_parent_block())
        .map(|b| set.contains(&b))
        .unwrap_or(false)
}

/// The blocks of `set` forward-reachable from `entry`, following only edges that
/// stay inside `set`, in a deterministic visit order.
///
/// `clone_blocks_into` is order-independent (it records every clone block, block
/// argument, and op result before wiring any operand or successor), so the order
/// here is not required for cloning; a plain reachability walk is enough. We keep
/// the result ordered only so IR dumps are stable. Its main job is the soundness
/// check at the call site: if it is shorter than `set`, some body block reaches
/// the latch but is not reachable from the single entry, i.e. irreducible /
/// multi-entry control flow the v1 shape does not support.
fn reachable_body_blocks(
    ctx: &Context,
    region: Ptr<Region>,
    entry: Ptr<BasicBlock>,
    set: &FxHashSet<Ptr<BasicBlock>>,
) -> Vec<Ptr<BasicBlock>> {
    let mut visited: FxHashSet<Ptr<BasicBlock>> = FxHashSet::default();
    let mut order: Vec<Ptr<BasicBlock>> = Vec::new();
    let mut stack: Vec<Ptr<BasicBlock>> = vec![entry];
    while let Some(b) = stack.pop() {
        if !visited.insert(b) {
            continue;
        }
        order.push(b);
        for s in region.successors(ctx, &b) {
            if set.contains(&s) && !visited.contains(&s) {
                stack.push(s);
            }
        }
    }
    order
}

/// Gather the loop facts and confirm the v1 shape. See [`LoopShape`].
fn analyze_shape(
    ctx: &Context,
    info: &LoopInfo,
    region: Ptr<Region>,
    id: usize,
    preheader: Ptr<BasicBlock>,
    rec: &induction::LoopRecurrences,
) -> std::result::Result<LoopShape, String> {
    let l = &info.loops()[id];
    // An inner loop (one with a parent) is fine to unroll: its body reads the
    // outer loop's values by dominance, and cloning leaves those untouched, so
    // the copies still see them. But unrolling a loop that itself *contains*
    // another loop would have to duplicate that inner loop too; leave that case
    // to a follow-up and tell the author to annotate the inner loop instead.
    // (This `children` test is load-bearing for nested correctness: it relies on
    // `LoopInfo` recording a true container as a parent, which holds because in a
    // reducible CFG a parent loop's block set is a strict superset of each child's.)
    if !l.children.is_empty() {
        return Err("this loop contains an inner loop; unrolling a loop that contains loops is a follow-up (annotate the inner loop instead)".into());
    }
    if l.latches.len() != 1 {
        return Err(format!(
            "the loop has {} back-edges; only single-latch loops are supported for now",
            l.latches.len()
        ));
    }
    let header = l.header;
    let latch = l.latches[0];

    // The only way out must be the header's exit test: no early `break`.
    let exiting = info.exiting_blocks(ctx, region, id);
    if exiting.len() != 1 || exiting[0] != header {
        return Err("the body can leave the loop early (a `break`/extra exit); only loops whose sole exit is the header test are supported for now".into());
    }
    let exits = info.exit_blocks(ctx, region, id);
    if exits.len() != 1 {
        return Err(format!(
            "the loop has {} exit targets; only single-exit loops are supported for now",
            exits.len()
        ));
    }
    let exit = exits[0];

    let iv_idx = rec
        .primary_iv
        .ok_or("no recognized induction variable (loop counter)")?;
    let (iv_init, iv_step) = match &rec.args[iv_idx] {
        ArgKind::BasicIv { init, step } => (*init, *step),
        _ => return Err("the loop counter is not a simple induction variable".into()),
    };

    // The body entry is the header's one successor that stays inside the loop.
    let header_term = header
        .deref(ctx)
        .get_terminator(ctx)
        .ok_or("the header has no terminator")?;
    let in_loop: Vec<Ptr<BasicBlock>> = header_term
        .deref(ctx)
        .successors()
        .filter(|s| l.blocks.contains(s))
        .collect();
    if in_loop.len() != 1 {
        return Err("could not identify a single loop-body entry from the header".into());
    }
    let body_entry = in_loop[0];

    // The preheader must end in a plain branch to the header (dedicated preheader).
    let preheader_term = preheader
        .deref(ctx)
        .get_terminator(ctx)
        .ok_or("the preheader has no terminator")?;
    let p_succs: Vec<Ptr<BasicBlock>> = preheader_term.deref(ctx).successors().collect();
    if p_succs != [header] {
        return Err("the loop has no dedicated preheader (its entry edge is not a plain branch to the header)".into());
    }

    // The latch must end in a plain back-edge to the header.
    let latch_term = latch
        .deref(ctx)
        .get_terminator(ctx)
        .ok_or("the latch has no terminator")?;
    let l_succs: Vec<Ptr<BasicBlock>> = latch_term.deref(ctx).successors().collect();
    if l_succs != [header] {
        return Err("the loop latch does not end in a single back-edge to the header".into());
    }

    let nargs = header.deref(ctx).get_num_arguments();
    let header_args: Vec<Value> = (0..nargs)
        .map(|i| header.deref(ctx).get_argument(i))
        .collect();
    let iv_type = header_args[iv_idx].get_type(ctx);
    let i1_type = header_term.deref(ctx).get_operand(0).get_type(ctx);

    let init_ops = induction::edge_operands(ctx, preheader, header)
        .filter(|v| v.len() == nargs)
        .ok_or("preheader carried-value arity mismatch")?;
    let recur_ops = induction::edge_operands(ctx, latch, header)
        .filter(|v| v.len() == nargs)
        .ok_or("latch carried-value arity mismatch")?;
    let entry_ops = induction::edge_operands(ctx, header, body_entry).unwrap_or_default();
    let exit_ops = induction::edge_operands(ctx, header, exit).unwrap_or_default();

    // Clean-header check (see [`LoopShape`]): the body must not read any value the
    // header *computes*. Reading the header's block arguments is fine (we
    // substitute those per copy); reading a header op result is not.
    let body_blocks: FxHashSet<Ptr<BasicBlock>> =
        l.blocks.iter().copied().filter(|&b| b != header).collect();
    let defined_in_header = |ctx: &Context, v: Value| -> bool {
        v.defining_op()
            .map(|d| d.deref(ctx).get_parent_block() == Some(header))
            .unwrap_or(false)
    };
    for &b in &body_blocks {
        for op in b.deref(ctx).iter(ctx).collect::<Vec<_>>() {
            let nops = op.deref(ctx).get_num_operands();
            for o in 0..nops {
                if defined_in_header(ctx, op.deref(ctx).get_operand(o)) {
                    return Err("the header computes a value the body reads; this shape needs header cloning (a follow-up)".into());
                }
            }
        }
    }
    for &v in &entry_ops {
        if defined_in_header(ctx, v) {
            return Err("the header passes a computed value into the body; this shape needs header cloning (a follow-up)".into());
        }
    }
    // Same for live-outs: a header *block argument* the exit reads is handled (we
    // substitute the final value), but a header op *result* on the exit edge
    // would dangle once full unroll deletes the header. Reject it loudly.
    for &v in &exit_ops {
        if defined_in_header(ctx, v) {
            return Err("the header computes a live-out value the exit reads; this shape needs header cloning (a follow-up)".into());
        }
    }

    let body_blocks_ordered = reachable_body_blocks(ctx, region, body_entry, &body_blocks);
    if body_blocks_ordered.len() != body_blocks.len() {
        return Err("the loop body has blocks unreachable from its entry (irreducible control flow); not supported".into());
    }

    Ok(LoopShape {
        header,
        latch,
        exit,
        body_entry,
        body_blocks_ordered,
        header_args,
        nargs,
        init_ops,
        recur_ops,
        entry_ops,
        exit_ops,
        iv_idx,
        iv_init,
        iv_step,
        iv_type,
        i1_type,
        preheader_term,
        loop_blocks: l.blocks.clone(),
    })
}

/// One cloned copy of the loop body.
struct CopyResult {
    /// The clone of `body_entry`: where control enters this copy.
    entry: Ptr<BasicBlock>,
    /// The clone of the latch's terminator (its back-edge `goto header`), which
    /// the caller repoints to the next copy (or the exit).
    latch_term: Ptr<Operation>,
    /// The carried values this copy produces, to feed the next copy (the latch's
    /// back-edge operands, mapped through this copy's substitution).
    next_running: Vec<Value>,
    /// The operands to pass when branching into `entry` (the header -> body_entry
    /// operands, mapped through this copy's substitution).
    entry_args: Vec<Value>,
    /// This copy's cloned blocks, in the same order as
    /// `LoopShape::body_blocks_ordered`.
    blocks: Vec<Ptr<BasicBlock>>,
}

/// Clone one copy of the loop body, substituting `subst[a]` for header argument
/// `a` (so `subst` carries this copy's counter value and accumulators). The
/// clone's internal branches and block-argument passes are remapped
/// automatically by [`clone_blocks_into`]; the caller wires the boundary edges.
fn clone_one_copy(
    ctx: &mut Context,
    region: Ptr<Region>,
    s: &LoopShape,
    subst: &[Value],
) -> CopyResult {
    let mut mapper = IrMapping::new();
    for (a, &hv) in s.header_args.iter().enumerate() {
        mapper.map_value(hv, subst[a]);
    }
    // Operands for the edge that enters this copy: the header -> body_entry
    // operands with this copy's carried values substituted in. Computed before
    // cloning (they reference only header args / outer values, never body values).
    let entry_args: Vec<Value> = s
        .entry_ops
        .iter()
        .map(|&v| mapper.lookup_value_or_default(v))
        .collect();
    let mut rewriter = IRRewriter::<DummyListener>::default();
    clone_blocks_into(
        &s.body_blocks_ordered,
        region,
        ctx,
        &mut rewriter,
        &mut mapper,
    );
    let entry = mapper.lookup_block_or_default(s.body_entry);
    let latch = mapper.lookup_block_or_default(s.latch);
    let latch_term = latch
        .deref(ctx)
        .get_terminator(ctx)
        .expect("a cloned latch has a terminator");
    let next_running: Vec<Value> = s
        .recur_ops
        .iter()
        .map(|&v| mapper.lookup_value_or_default(v))
        .collect();
    let blocks: Vec<Ptr<BasicBlock>> = s
        .body_blocks_ordered
        .iter()
        .map(|&b| mapper.lookup_block_or_default(b))
        .collect();
    CopyResult {
        entry,
        latch_term,
        next_running,
        entry_args,
        blocks,
    }
}

/// Repoint a single-successor branch (`goto`) at `new_succ`, replacing all its
/// edge operands with `operands`.
fn rewire_goto(
    ctx: &mut Context,
    term: Ptr<Operation>,
    new_succ: Ptr<BasicBlock>,
    operands: &[Value],
) {
    Operation::replace_successor(term, ctx, 0, new_succ);
    let n = term.deref(ctx).get_num_operands();
    for _ in 0..n {
        Operation::remove_operand(term, ctx, 0);
    }
    for &v in operands {
        Operation::push_operand(term, ctx, v);
    }
}

/// Fully unroll a loop whose iteration count is known at compile time, so no
/// loop is left at all. Works for any body shape the [`LoopShape`] checks allow
/// (single latch, single header-test exit, no nested loops), including bodies
/// with internal `if`/`else`/`match`.
///
/// For a trip count `T`, it lays down `T` copies of the body, chained one into
/// the next: copy 0 entered from the preheader, copy `k`'s latch flowing into
/// copy `k+1`, and the last copy's latch flowing to the loop's exit. In copy `k`
/// the counter is the literal `init + k*step`, and the other carried values are
/// threaded from each copy to the next. The original loop blocks become
/// unreachable and `simplify_cfg` deletes them.
fn full_unroll(
    ctx: &mut Context,
    info: &LoopInfo,
    region: Ptr<Region>,
    id: usize,
    preheader: Ptr<BasicBlock>,
    rec: &induction::LoopRecurrences,
) -> Result<UnrollOutcome> {
    let s = match analyze_shape(ctx, info, region, id, preheader, rec) {
        Ok(s) => s,
        Err(why) => return Ok(UnrollOutcome::Skipped(why)),
    };
    let trip = match rec.trip_count {
        Some(t) => t as i128,
        None => {
            return Ok(UnrollOutcome::Skipped(
                "full #[unroll] needs a compile-time-constant trip count; this loop's count is only known at runtime (use #[unroll(N)] for partial unrolling)".into(),
            ));
        }
    };

    // Carried values flowing into the next copy; start at the loop's initial
    // values. `prev_tail` is the branch we must point at the next copy (the
    // preheader first, then each copy's latch).
    let mut running: Vec<Value> = s.init_ops.clone();
    let mut prev_tail = s.preheader_term;

    for k in 0..trip {
        // The counter in copy k is the literal init + k*step. Materialize it just
        // before the branch into this copy, so it dominates the copy.
        let iv_val = make_const(ctx, s.iv_type, s.iv_init + k * s.iv_step, prev_tail);
        let mut subst = running.clone();
        subst[s.iv_idx] = iv_val;
        let c = clone_one_copy(ctx, region, &s, &subst);
        rewire_goto(ctx, prev_tail, c.entry, &c.entry_args);
        prev_tail = c.latch_term;
        running = c.next_running;
    }

    // The loop has finished: the counter's final value is the literal
    // init + T*step. Use it for any live-out reads of the counter.
    let final_iv = make_const(ctx, s.iv_type, s.iv_init + trip * s.iv_step, prev_tail);
    running[s.iv_idx] = final_iv;

    // Branch the last copy to the exit, feeding the exit's own block arguments (if
    // any) the final carried values.
    let exit_args: Vec<Value> = s
        .exit_ops
        .iter()
        .map(|&v| match s.header_args.iter().position(|&h| h == v) {
            Some(a) => running[a],
            None => v,
        })
        .collect();
    rewire_goto(ctx, prev_tail, s.exit, &exit_args);

    // The exit (and code after it) may also read the carried values *directly* by
    // dominance (this IR is not loop-closed SSA, so a header block argument can be
    // used outside the loop). The original header is now dead, so those reads must
    // be repointed to the final unrolled values. Uses inside the loop are left
    // alone; their blocks are unreachable and get deleted by `simplify_cfg`.
    for a in 0..s.nargs {
        let replacement = running[a];
        s.header_args[a].replace_some_uses_with(
            ctx,
            |ctx, u| match u.user_op().deref(ctx).get_parent_block() {
                Some(b) => !s.loop_blocks.contains(&b),
                None => true,
            },
            &replacement,
        );
    }

    Ok(UnrollOutcome::Unrolled)
}

/// Build an integer constant op (`mir.constant`) of type `ty` holding `value`,
/// place it just before the op `before`, and hand back the value it produces.
fn make_const(ctx: &mut Context, ty: TypeHandle, value: i128, before: Ptr<Operation>) -> Value {
    let typed = TypedHandle::<IntegerType>::from_handle(ty, ctx).expect("IV is an integer type");
    let width = typed.deref(ctx).width() as usize;
    let apint = APInt::from_i128(value, NonZero::new(width).expect("non-zero width"));
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

/// Partially unroll a loop: do `factor` iterations' worth of work per trip, and
/// keep a small "remainder" loop for the iterations left over when the total
/// doesn't divide evenly by `factor`. Works for any body shape the [`LoopShape`]
/// checks allow (multi-block bodies included).
///
/// Unlike full unroll, this works even when the iteration count is only known at
/// runtime. The original loop is reused as the **remainder loop**, running the
/// last `trip % factor` iterations one at a time. In front of it we build a new
/// **main loop** whose body is `factor` copies of the loop body chained
/// together, advancing the counter by `factor*step` each trip. The main loop
/// keeps going only while a whole group of `factor` more iterations still fits;
/// once fewer than that remain, control falls into the remainder loop.
///
/// Only counting-up loops (test `<` or `<=`, positive step) are handled for now.
///
/// What it produces (the loop was entered as `preheader -> header`):
/// ```text
///   preheader -> main_h(init...)
///   main_h(acc, i):                       (i = counter, acc = carried values)
///       if (i + (factor-1)*step) <pred> bound  -> copy0   (a full group fits)
///       else                                   -> header  (run the remainder)
///   copy0 .. copy(factor-1): the body, factor times, chained; the last copy's
///       latch loops back to main_h with (acc', i + factor*step)
///   header/.../latch: the original loop, now just the leftover tail
/// ```
fn partial_unroll(
    ctx: &mut Context,
    info: &LoopInfo,
    region: Ptr<Region>,
    id: usize,
    preheader: Ptr<BasicBlock>,
    rec: &induction::LoopRecurrences,
    factor: u32,
) -> Result<UnrollOutcome> {
    let n = factor as i128;
    if n < 2 {
        return Ok(UnrollOutcome::Skipped(format!(
            "unroll factor {factor} is too small (need 2 or more)"
        )));
    }
    let s = match analyze_shape(ctx, info, region, id, preheader, rec) {
        Ok(s) => s,
        Err(why) => return Ok(UnrollOutcome::Skipped(why)),
    };
    // Partial unroll needs an up-counting loop with a known bound value.
    if s.iv_step <= 0 || !matches!(rec.continue_pred, Some(CmpPred::Lt) | Some(CmpPred::Le)) {
        return Ok(UnrollOutcome::Skipped(
            "partial #[unroll(N)] supports only up-counting loops (test < or <=, positive step) for now".into(),
        ));
    }
    let pred = rec.continue_pred.unwrap();
    let bound = match rec.bound_value {
        Some(b) => b,
        None => {
            return Ok(UnrollOutcome::Skipped(
                "partial #[unroll(N)] needs the loop bound as a value".into(),
            ));
        }
    };
    // The guard we build below, `i + (N-1)*step <pred> bound`, is only correct if
    // `bound` is the same on every iteration. A constant bound is always fine (we
    // re-materialize it in the main header below, so where its op sits does not
    // matter). A non-constant bound that is a loop-carried header argument or is
    // computed inside the loop can change within a group of N iterations, which
    // would make the guard admit too many iterations (a miscompile), and would
    // not dominate the new main header either. Bail loudly on those.
    let bound_is_const = induction::const_i128(ctx, bound).is_some();
    if !bound_is_const
        && (s.header_args.contains(&bound) || defined_in_loop(ctx, bound, &s.loop_blocks))
    {
        return Ok(UnrollOutcome::Skipped(
            "partial #[unroll(N)] needs a loop-invariant bound (the loop's limit must be the same on every iteration); this loop's limit changes inside the loop".into(),
        ));
    }

    let arg_types: Vec<TypeHandle> = s.header_args.iter().map(|a| a.get_type(ctx)).collect();
    // The new main-loop header, taking the same carried values as the original.
    let main_h = BasicBlock::new(ctx, None, arg_types);
    main_h.insert_before(ctx, s.header);
    let mh_args: Vec<Value> = (0..s.nargs)
        .map(|i| main_h.deref(ctx).get_argument(i))
        .collect();
    let mh_iv = mh_args[s.iv_idx];

    // Lay down `factor` copies of the body, threading carried values copy to
    // copy. Copy 0 uses the main header's args (its counter is mh_iv); each later
    // copy uses the previous copy's latch results, so its counter is the previous
    // counter + step (a runtime value, not a literal: partial keeps it runtime).
    let mut running: Vec<Value> = mh_args.clone();
    let mut copies: Vec<CopyResult> = Vec::with_capacity(factor as usize);
    for _ in 0..factor {
        let subst = running.clone();
        let c = clone_one_copy(ctx, region, &s, &subst);
        running = c.next_running.clone();
        copies.push(c);
    }
    // Chain copy j's latch into copy j+1's entry; the last copy's latch loops
    // back to main_h carrying `running` (counter now mh_iv + factor*step).
    for j in 0..(factor as usize - 1) {
        let next_entry = copies[j + 1].entry;
        let next_args = copies[j + 1].entry_args.clone();
        rewire_goto(ctx, copies[j].latch_term, next_entry, &next_args);
    }
    let last_latch = copies[factor as usize - 1].latch_term;
    rewire_goto(ctx, last_latch, main_h, &running);

    // The main-loop counter is mh_iv = init + (factor*step)*t, so it is always
    // init plus a multiple of factor*step. That lets us replace counter-derived
    // index ops -- `(counter +/- const) & mask` and `(counter +/- const) % 2^k`
    // -- in the copies with literals (`fold_constant_index_in_copies`), the main
    // payoff of unrolling. Scan every cloned copy block.
    let copy_blocks: Vec<Ptr<BasicBlock>> = copies
        .iter()
        .flat_map(|c| c.blocks.iter().copied())
        .collect();
    fold_constant_index_in_copies(ctx, &copy_blocks, mh_iv, s.iv_init, n * s.iv_step);

    // main_h guard: stay in the main loop only while a whole group of `factor`
    // iterations still fits. The last copy in a group uses counter
    // mh_iv + (factor-1)*step, so the group fits exactly when that still passes
    // the test. True -> copy 0 (enter the main body); False -> the original
    // header (the remainder loop), passing the current carried values.
    // A constant bound may be defined inside the (original) loop, which does not
    // dominate `main_h`; re-materialize it here. A non-constant bound was already
    // checked to be defined outside the loop, so it dominates `main_h` as-is.
    let guard_bound = match induction::const_i128(ctx, bound) {
        Some(c) => append_const(ctx, bound.get_type(ctx), c, main_h),
        None => bound,
    };
    let last_off = append_const(ctx, s.iv_type, (n - 1) * s.iv_step, main_h);
    let last_iv = append_add(ctx, s.iv_type, mh_iv, last_off, main_h);
    let cont = append_cmp(ctx, pred, last_iv, guard_bound, s.i1_type, main_h);
    let entry0 = copies[0].entry;
    let entry0_args = copies[0].entry_args.clone();
    let (flat, segs) =
        MirCondBranchOp::compute_segment_sizes(vec![vec![cont], entry0_args, mh_args.clone()]);
    let cbr = Operation::new(
        ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        flat,
        vec![entry0, s.header],
        0,
    );
    Operation::get_op::<MirCondBranchOp>(cbr, ctx)
        .expect("MirCondBranchOp")
        .set_operand_segment_sizes(ctx, segs);
    cbr.insert_at_back(main_h, ctx);

    // Finally, make the preheader branch into the new main loop instead of the
    // original header, reusing the same initial values it already passed. The
    // original loop stays in place and becomes the remainder.
    rewire_goto(ctx, s.preheader_term, main_h, &s.init_ops);
    Ok(UnrollOutcome::Unrolled)
}

/// If `v` is the counter `iv` plus or minus some constants (e.g. `iv`, `iv + 1`,
/// `iv + 4 - 2`), return that net constant offset; `None` otherwise. So `iv + 1`
/// gives `Some(1)` and `iv` gives `Some(0)`. The caller uses this to spot values
/// that track the counter with a known fixed offset.
fn affine_offset(ctx: &Context, v: Value, iv: Value) -> Option<i128> {
    if v == iv {
        return Some(0);
    }
    let def = v.defining_op()?;
    if Operation::get_op::<MirAddOp>(def, ctx).is_some() {
        let a = def.deref(ctx).get_operand(0);
        let b = def.deref(ctx).get_operand(1);
        if let (Some(o), Some(c)) = (affine_offset(ctx, a, iv), induction::const_i128(ctx, b)) {
            return Some(o + c);
        }
        if let (Some(o), Some(c)) = (affine_offset(ctx, b, iv), induction::const_i128(ctx, a)) {
            return Some(o + c);
        }
    } else if Operation::get_op::<MirSubOp>(def, ctx).is_some() {
        let a = def.deref(ctx).get_operand(0);
        let b = def.deref(ctx).get_operand(1);
        if let (Some(o), Some(c)) = (affine_offset(ctx, a, iv), induction::const_i128(ctx, b)) {
            return Some(o - c);
        }
    }
    None
}

/// Peephole: in each partial-unroll copy, replace a counter-derived index with
/// the constant it always equals.
///
/// After unrolling by N, the main counter `iv` only takes values N apart:
/// `init, init+N, init+2N, ...`. Copy `j` of the body uses `iv + j`. Two index
/// shapes are then the same constant on every iteration, so we replace each with
/// a literal (note `x & (2^k - 1)` and `x % 2^k` are the same operation):
///
/// ```text
///   iv & MASK   (MASK = 2^k - 1)      keep the low k bits
///   iv % M      (M a power of two)    same thing: x % 2^k == x & (2^k - 1)
/// ```
///
/// Both read only the low k bits, and a multiple of N has *fixed* low k bits
/// exactly when the window `2^k` divides the unroll step `N*step`. Example,
/// N = 4, `(iv + j) & 3` (the gemm pipeline-stage index):
///
/// ```text
///   iv = 0, 4, 8, 12, ...   all end in ...00, so:
///     (iv + 0) & 3 = 0       (iv + 2) & 3 = 2
///     (iv + 1) & 3 = 1       (iv + 3) & 3 = 3
///   => each copy's stage index is a compile-time constant.
/// ```
///
/// Fires only when ALL of these hold (otherwise the value genuinely changes each
/// iteration, so there is nothing to fold and we skip):
///
/// - the op is `(iv +/- consts) & MASK` or `(iv +/- consts) % M`;
/// - the window (`MASK + 1`, or `M`) is a power of two -- so it reads only low
///   bits and is therefore immune to the type's wraparound;
/// - that window divides the unroll step `N*step` -- so the low bits never move;
/// - for `%`, the operand type is unsigned (signed `%` follows the dividend's
///   sign, which breaks the equality with the masked low bits);
/// - `M > 0` -- never fold `% 0` (rem-by-zero is a Rust panic).
///
/// Deliberately NOT handled: non-power-of-two `% M` (e.g. `% 3`). The congruence
/// still holds on paper, but `%` by a non-power-of-two is not wraparound-safe:
/// near the type's max the counter wraps by `2^width`, which is not a multiple
/// of `M`, shifting the result. Documented gap, left for later.
///
/// Full unroll never needs this: there `iv` is a literal per copy, so ordinary
/// constant folding already turns `i & 3` / `i % 3` into a number. The leftover
/// dead `& MASK` / `% M` ops are removed by the unroll pass's later `dce`.
///
/// Parameters: `iv` is the counter; `init` its start; `step_jump` the unroll step
/// (`N * original_step`), so `iv = init + step_jump * t` for iteration `t`.
/// `blocks` are all the cloned copy blocks (one copy may span several blocks).
fn fold_constant_index_in_copies(
    ctx: &mut Context,
    blocks: &[Ptr<BasicBlock>],
    iv: Value,
    init: i128,
    step_jump: i128,
) {
    if step_jump <= 0 {
        return;
    }
    for &block in blocks {
        let ops: Vec<Ptr<Operation>> = block.deref(ctx).iter(ctx).collect();
        for op in ops {
            let Some((offset, window)) = counter_index_window(ctx, op, iv) else {
                continue;
            };
            // Foldable only when the window is a power of two that divides the
            // unroll step: then the counter's low bits never move, so the index
            // is the same every iteration (and power-of-two makes it wrap-safe).
            // The power-of-two check also rejects a non-low-bit `& C` (e.g.
            // `x & 5` is not `x % 6`).
            if window <= 0 || (window & (window - 1)) != 0 || step_jump % window != 0 {
                continue;
            }
            let folded = (init + offset).rem_euclid(window);
            let result = op.deref(ctx).get_result(0);
            let ty = result.get_type(ctx);
            let lit = make_const(ctx, ty, folded, op);
            result.replace_all_uses_with(ctx, &lit);
        }
    }
}

/// If `op` is a counter-derived index we might fold -- `(iv +/- consts) & MASK`
/// or `(iv +/- consts) % M` -- return `(offset, window)`, where `offset` is the
/// counter's constant offset and `window` is `MASK + 1` (for `&`) or `M` (for
/// `%`). The caller still checks that `window` is a power of two dividing the
/// unroll step. Returns `None` for any other op.
fn counter_index_window(ctx: &Context, op: Ptr<Operation>, iv: Value) -> Option<(i128, i128)> {
    if Operation::get_op::<MirBitAndOp>(op, ctx).is_some() {
        // `&` is commutative: either operand may be the counter.
        let a = op.deref(ctx).get_operand(0);
        let b = op.deref(ctx).get_operand(1);
        let (offset, mask) = if let (Some(o), Some(m)) =
            (affine_offset(ctx, a, iv), induction::const_i128(ctx, b))
        {
            (o, m)
        } else if let (Some(o), Some(m)) =
            (affine_offset(ctx, b, iv), induction::const_i128(ctx, a))
        {
            (o, m)
        } else {
            return None;
        };
        if mask < 0 {
            return None;
        }
        // window = MASK + 1.
        Some((offset, mask + 1))
    } else if Operation::get_op::<MirRemOp>(op, ctx).is_some() {
        // `%` is NOT commutative: only the dividend (operand 0) may be the
        // counter, and only for unsigned types.
        let dividend = op.deref(ctx).get_operand(0);
        let divisor = op.deref(ctx).get_operand(1);
        if !is_unsigned_int(ctx, dividend) {
            return None;
        }
        let (Some(offset), Some(m)) = (
            affine_offset(ctx, dividend, iv),
            induction::const_i128(ctx, divisor),
        ) else {
            return None;
        };
        if m <= 0 {
            return None;
        }
        // window = M.
        Some((offset, m))
    } else {
        None
    }
}

/// True if `v` has an unsigned (or signless) integer type. Keeps the `%` fold off
/// signed remainders, whose result follows the dividend's sign rather than the
/// masked low bits.
fn is_unsigned_int(ctx: &Context, v: Value) -> bool {
    let ty = v.get_type(ctx);
    match TypedHandle::<IntegerType>::from_handle(ty, ctx) {
        Ok(t) => t.deref(ctx).signedness() != Signedness::Signed,
        Err(_) => false,
    }
}

/// Build an integer constant `value` of type `ty`, add it as the last op of
/// `block`, and return the value it produces. (Same as [`make_const`] but
/// appends to a block instead of inserting before a given op.)
fn append_const(ctx: &mut Context, ty: TypeHandle, value: i128, block: Ptr<BasicBlock>) -> Value {
    let typed = TypedHandle::<IntegerType>::from_handle(ty, ctx).expect("integer type");
    let width = typed.deref(ctx).width() as usize;
    let apint = APInt::from_i128(value, NonZero::new(width).expect("non-zero width"));
    let op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![ty],
        vec![],
        vec![],
        0,
    );
    MirConstantOp::new(op).set_attr_value(ctx, IntegerAttr::new(typed, apint));
    op.insert_at_back(block, ctx);
    op.deref(ctx).get_result(0)
}

/// Build `a + b` (an integer `mir.add` of type `ty`), add it as the last op of
/// `block`, and return its result value.
fn append_add(
    ctx: &mut Context,
    ty: TypeHandle,
    a: Value,
    b: Value,
    block: Ptr<BasicBlock>,
) -> Value {
    let op = Operation::new(
        ctx,
        MirAddOp::get_concrete_op_info(),
        vec![ty],
        vec![a, b],
        vec![],
        0,
    );
    op.insert_at_back(block, ctx);
    op.deref(ctx).get_result(0)
}

/// Build the comparison `a <pred> b` (a boolean, of type `i1_type`), add it as
/// the last op of `block`, and return its result value.
fn append_cmp(
    ctx: &mut Context,
    pred: CmpPred,
    a: Value,
    b: Value,
    i1_type: TypeHandle,
    block: Ptr<BasicBlock>,
) -> Value {
    let info = match pred {
        CmpPred::Lt => MirLtOp::get_concrete_op_info(),
        CmpPred::Le => MirLeOp::get_concrete_op_info(),
        CmpPred::Gt => MirGtOp::get_concrete_op_info(),
        CmpPred::Ge => MirGeOp::get_concrete_op_info(),
    };
    let op = Operation::new(ctx, info, vec![i1_type], vec![a, b], vec![], 0);
    op.insert_at_back(block, ctx);
    op.deref(ctx).get_result(0)
}

/// All function ops in the module (each `mir.func`).
fn collect_functions(module: Ptr<Operation>, ctx: &Context) -> Vec<Ptr<Operation>> {
    let mut out = Vec::new();
    let module_region = module.deref(ctx).get_region(0);
    let blocks: Vec<Ptr<BasicBlock>> = module_region.deref(ctx).iter(ctx).collect();
    for block in blocks {
        for op in block.deref(ctx).iter(ctx).collect::<Vec<_>>() {
            if Operation::get_op::<MirFuncOp>(op, ctx).is_some() {
                out.push(op);
            }
        }
    }
    out
}

/// Find the `mir.unroll_hint` ops in `region`, each with the block it sits in
/// (used to locate the enclosing loop) and its requested factor (0 = full).
fn collect_hints(
    ctx: &Context,
    region: Ptr<Region>,
) -> Vec<(Ptr<Operation>, Ptr<BasicBlock>, u32)> {
    let mut out = Vec::new();
    let blocks: Vec<Ptr<BasicBlock>> = region.deref(ctx).iter(ctx).collect();
    for block in blocks {
        for op in block.deref(ctx).iter(ctx).collect::<Vec<_>>() {
            if let Some(hint) = Operation::get_op::<MirUnrollHintOp>(op, ctx) {
                out.push((op, block, hint.factor(ctx)));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{counted_loop, mir_ctx};
    use pliron::graph::dominance::DomInfo;

    /// `analyze_shape` accepts the canonical counted loop and reports the right
    /// facts (the supported shape: single latch, single header-test exit,
    /// recognized counter, single-block body).
    #[test]
    fn analyze_shape_accepts_canonical_counted_loop() {
        let mut ctx = mir_ctx();
        let lp = counted_loop(&mut ctx, 8);

        let mut dom = DomInfo::default();
        let info = {
            let dt = dom.get_dom_tree(&ctx, lp.region);
            LoopInfo::compute(&ctx, lp.region, dt)
        };
        let id = info.innermost_loop(lp.header).unwrap();
        let ph = info.preheader(&ctx, lp.region, id).unwrap();
        let rec = induction::analyze(&ctx, &info, id, ph);

        let shape = analyze_shape(&ctx, &info, lp.region, id, ph, &rec)
            .expect("canonical counted loop is a supported shape");

        assert_eq!(shape.header, lp.header);
        assert_eq!(shape.latch, lp.latch);
        assert_eq!(shape.exit, lp.exit);
        assert_eq!(shape.nargs, 2); // (acc, i)
        assert_eq!(shape.iv_idx, 1); // i
        assert_eq!(shape.iv_init, 0);
        assert_eq!(shape.iv_step, 1);
        // The body is the single latch block; its entry is the latch itself.
        assert_eq!(shape.body_entry, lp.latch);
        assert_eq!(shape.body_blocks_ordered, vec![lp.latch]);
    }
}
