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

use dialect_mir::ops::arithmetic::{MirAddOp, MirBitAndOp, MirSubOp};
use dialect_mir::ops::comparison::{MirGeOp, MirGtOp, MirLeOp, MirLtOp};
use dialect_mir::ops::constants::MirConstantOp;
use dialect_mir::ops::control_flow::{MirCondBranchOp, MirGotoOp, MirUnrollHintOp};
use dialect_mir::ops::function::MirFuncOp;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::op_interfaces::OperandSegmentInterface;
use pliron::builtin::types::IntegerType;
use pliron::context::{Context, Ptr};
use pliron::graph::dominance::DomInfo;
use pliron::irbuild::cloning::{IrMapping, clone_operation};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::opts::dce::dce;
use pliron::opts::simplify_cfg::simplify_cfg;
use pliron::pass_manager::AnalysisManager;
use pliron::r#type::{TypeHandle, Typed, TypedHandle};
use pliron::region::Region;
use pliron::result::Result;
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

        for (loop_id, factor) in loop_factor {
            let Some(ph) = info.preheader(ctx, region, loop_id) else {
                continue;
            };
            let rec = induction::analyze(ctx, &info, loop_id, ph);
            if verbose() {
                eprintln!(
                    "loop-unroll: loop#{loop_id} factor={factor} trip={:?} primary_iv={:?}",
                    rec.trip_count, rec.primary_iv,
                );
            }
            let unrolled = if factor == 0 {
                // Full unroll: only works when the trip count is a constant.
                full_unroll(ctx, &info, region, loop_id, ph, &rec)?
            } else {
                // Partial unroll by `factor`, with a remainder loop for the tail.
                partial_unroll(ctx, &info, region, loop_id, ph, &rec, factor)?
            };
            changed |= unrolled;
        }
    }

    if changed {
        dce(module, ctx)?;
        simplify_cfg(module, ctx)?;
    }
    Ok(())
}

/// Fully unroll a loop whose iteration count is known at compile time, so no
/// loop is left at all.
///
/// It handles only the simple two-block shape mem2reg leaves a counted loop in:
/// a **header** block (carries the loop's per-iteration values as block
/// arguments and holds the `i < n` exit test) and one **latch** block (holds the
/// body and the branch back to the header). Anything more complicated returns
/// `Ok(false)` and leaves the IR alone.
///
/// For a trip count `T`, it writes `T` back-to-back copies of the body into the
/// preheader. In copy `k` the counter is replaced by its actual value that
/// iteration, `init + k*step` (a compile-time constant), and the other carried
/// values are passed from one copy to the next. Finally the preheader's branch
/// is pointed straight at the loop's exit, so the original header and latch
/// become unreachable and get deleted later by `simplify_cfg`.
fn full_unroll(
    ctx: &mut Context,
    info: &LoopInfo,
    region: Ptr<Region>,
    id: usize,
    preheader: Ptr<BasicBlock>,
    rec: &induction::LoopRecurrences,
) -> Result<bool> {
    let l = &info.loops()[id];
    // Only the simple shape: a loop that is neither inside another loop nor has
    // one inside it, and is made of exactly two blocks (header and latch).
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

    // We need both a counter and a known iteration count to unroll fully.
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
    // The body we copy is everything in the latch except its final branch (the
    // branch back to the header); we don't want to copy that loop-back itself.
    let body_ops: Vec<Ptr<Operation>> = latch
        .deref(ctx)
        .iter(ctx)
        .filter(|&op| op != latch_term)
        .collect();

    // Current value of each carried header argument; starts at the values the
    // preheader passes in (the loop's initial values).
    let mut running: Vec<Value> = init_ops;

    for k in 0..trip {
        // `mapper` records, for this body copy, what to substitute for each
        // value the original body referred to.
        let mut mapper = IrMapping::new();
        // The counter's value in copy k is the constant init + k*step.
        let iv_val = make_const(ctx, iv_type, iv_init + (k as i128) * iv_step, p_term);
        for a in 0..nargs {
            let mapped = if a == iv_idx { iv_val } else { running[a] };
            mapper.map_value(header_args[a], mapped);
        }
        for &op in &body_ops {
            let cloned = clone_operation(op, ctx, &mut mapper);
            cloned.insert_before(ctx, p_term);
        }
        // What this copy produced for each carried value becomes the input to
        // the next copy (these are the values the latch would have looped back).
        for a in 0..nargs {
            if a != iv_idx {
                running[a] = mapper.lookup_value_or_default(recur_ops[a]);
            }
        }
    }

    // If code after the loop reads the counter's final value, build it as the
    // constant init + T*step (its value once the loop has finished).
    if exit_ops_raw.iter().any(|&v| v == header_args[iv_idx]) {
        running[iv_idx] = make_const(ctx, iv_type, iv_init + (trip as i128) * iv_step, p_term);
    }

    // Anything outside the loop that referred to a header argument (e.g. the
    // exit block reading the final accumulator) should now use our final
    // unrolled value instead. Uses inside the loop are left as they are: those
    // blocks become dead and get deleted.
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

    // Repoint the preheader's branch from the (now dead) header to the loop's
    // exit: `goto header(init...)` becomes `goto exit(...)`, carrying whatever
    // values the exit expects.
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

/// Build an integer constant op (`mir.constant`) of type `ty` holding `value`,
/// place it just before the op `before`, and hand back the value it produces.
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

/// Partially unroll a loop: do `factor` iterations' worth of work per trip, and
/// keep a small "remainder" loop for the iterations left over when the total
/// doesn't divide evenly by `factor`.
///
/// Unlike full unroll, this works even when the iteration count is only known at
/// runtime. The original loop (header + latch) is reused as the **remainder
/// loop**, running the last `trip % factor` iterations one at a time. In front
/// of it we build a new **main loop** that does `factor` copies of the body per
/// trip and advances the counter by `factor*step` each time. The main loop keeps
/// going only while a whole group of `factor` more iterations still fits; once
/// fewer than that remain, control falls through into the remainder loop.
///
/// Only counting-up loops (test `<` or `<=`, positive step) are handled for now;
/// anything else returns `Ok(false)` and changes nothing.
///
/// What it produces (the loop was entered as `preheader -> header`):
/// ```text
///   preheader -> main_h(init...)
///   main_h(acc, i):                       (i = counter, acc = carried values)
///       if (i + (factor-1)*step) <pred> bound  -> main_l   (a full group fits)
///       else                                   -> header   (run the remainder)
///   main_l: factor body copies at i, i+step, ...; then
///           goto main_h(acc', i + factor*step)
///   header/latch: the original loop, now just the leftover tail
/// ```
fn partial_unroll(
    ctx: &mut Context,
    info: &LoopInfo,
    region: Ptr<Region>,
    id: usize,
    _preheader: Ptr<BasicBlock>,
    rec: &induction::LoopRecurrences,
    factor: u32,
) -> Result<bool> {
    let n = factor as i128;
    if n < 2 {
        return Ok(false);
    }
    let l = &info.loops()[id];
    if l.parent.is_some() || !l.children.is_empty() {
        return Ok(false);
    }
    if l.latches.len() != 1 || l.blocks.len() != 2 {
        return Ok(false);
    }
    let header = l.header;
    let latch = l.latches[0];
    if info.exit_blocks(ctx, region, id).len() != 1 {
        return Ok(false);
    }

    let iv_idx = match rec.primary_iv {
        Some(i) => i,
        None => return Ok(false),
    };
    let (iv_init, iv_step) = match &rec.args[iv_idx] {
        ArgKind::BasicIv { init, step } => (*init, *step),
        _ => return Ok(false),
    };
    // Only loops that count upward (positive step, test `<` or `<=`) for now.
    if iv_step <= 0 || !matches!(rec.continue_pred, Some(CmpPred::Lt) | Some(CmpPred::Le)) {
        return Ok(false);
    }
    let pred = rec.continue_pred.unwrap();
    let bound = match rec.bound_value {
        Some(b) => b,
        None => return Ok(false),
    };

    let nargs = header.deref(ctx).get_num_arguments();
    let header_args: Vec<Value> = (0..nargs).map(|i| header.deref(ctx).get_argument(i)).collect();
    let iv_type = header_args[iv_idx].get_type(ctx);
    let arg_types: Vec<TypeHandle> = header_args.iter().map(|a| a.get_type(ctx)).collect();

    // Grab the boolean (i1) type from the original test's condition, so the new
    // comparison we build below has the right result type.
    let h_term = match header.deref(ctx).get_terminator(ctx) {
        Some(t) => t,
        None => return Ok(false),
    };
    let i1_type = h_term.deref(ctx).get_operand(0).get_type(ctx);

    let recur_ops = match induction::edge_operands(ctx, latch, header) {
        Some(v) if v.len() == nargs => v,
        _ => return Ok(false),
    };
    let latch_term = match latch.deref(ctx).get_terminator(ctx) {
        Some(t) => t,
        None => return Ok(false),
    };
    let body_ops: Vec<Ptr<Operation>> = latch
        .deref(ctx)
        .iter(ctx)
        .filter(|&op| op != latch_term)
        .collect();
    let p_term = match _preheader.deref(ctx).get_terminator(ctx) {
        Some(t) => t,
        None => return Ok(false),
    };

    // Two new blocks: the main loop's header (main_h) and its body (main_l).
    // main_h takes the same arguments the original header did.
    let main_h = BasicBlock::new(ctx, None, arg_types);
    let main_l = BasicBlock::new(ctx, None, vec![]);
    main_h.insert_before(ctx, header);
    main_l.insert_before(ctx, header);
    let mh_args: Vec<Value> = (0..nargs).map(|i| main_h.deref(ctx).get_argument(i)).collect();
    let mh_iv = mh_args[iv_idx];

    // --- main_l: lay down `factor` body copies, then advance the counter by
    //     factor*step for the next trip ---
    let mut running: Vec<Value> = mh_args.clone();
    for j in 0..factor {
        let mut mapper = IrMapping::new();
        // Copy j uses counter value `i + j*step` (copy 0 is just `i`).
        let iv_j = if j == 0 {
            mh_iv
        } else {
            let c = append_const(ctx, iv_type, (j as i128) * iv_step, main_l);
            append_add(ctx, iv_type, mh_iv, c, main_l)
        };
        for a in 0..nargs {
            let mapped = if a == iv_idx { iv_j } else { running[a] };
            mapper.map_value(header_args[a], mapped);
        }
        for &op in &body_ops {
            let cloned = clone_operation(op, ctx, &mut mapper);
            cloned.insert_at_back(main_l, ctx);
        }
        for a in 0..nargs {
            if a != iv_idx {
                running[a] = mapper.lookup_value_or_default(recur_ops[a]);
            }
        }
    }
    let step_c = append_const(ctx, iv_type, n * iv_step, main_l);
    running[iv_idx] = append_add(ctx, iv_type, mh_iv, step_c, main_l);
    let back = Operation::new(
        ctx,
        MirGotoOp::get_concrete_op_info(),
        vec![],
        running,
        vec![main_h],
        0,
    );
    back.insert_at_back(main_l, ctx);

    // The main-loop counter starts at iv_init and grows by factor*step each
    // trip, so it is always iv_init plus a multiple of factor*step. That lets us
    // replace `(counter +/- const) & mask` ops inside the body with constants
    // (see `fold_iv_congruences`), which is the main payoff of unrolling.
    fold_iv_congruences(ctx, main_l, mh_iv, iv_init, n * iv_step);

    // --- main_h: the guard. Keep going in the main loop only while a whole group
    //     of `factor` iterations still fits; otherwise hand off to the remainder.
    //     The last copy in a group uses counter `i + (factor-1)*step`, so the
    //     whole group fits exactly when that value still satisfies the test. ---
    let last_off = append_const(ctx, iv_type, (n - 1) * iv_step, main_h);
    let last = append_add(ctx, iv_type, mh_iv, last_off, main_h);
    let cont = append_cmp(ctx, pred, last, bound, i1_type, main_h);
    // True (group fits): go to main_l (it needs no arguments). False: go to the
    // original header (the remainder loop), passing the current carried values.
    let (flat, segs) =
        MirCondBranchOp::compute_segment_sizes(vec![vec![cont], vec![], mh_args.clone()]);
    let cbr = Operation::new(
        ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        flat,
        vec![main_l, header],
        0,
    );
    Operation::get_op::<MirCondBranchOp>(cbr, ctx)
        .expect("MirCondBranchOp")
        .set_operand_segment_sizes(ctx, segs);
    cbr.insert_at_back(main_h, ctx);

    // Finally, make the preheader branch into the new main loop instead of the
    // original header, reusing the same initial values it already passed.
    Operation::replace_successor(p_term, ctx, 0, main_h);
    Ok(true)
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

/// Replace `x & MASK` with a constant inside the unrolled main loop, when the
/// loop counter guarantees the result is the same on every iteration.
///
/// Why it's possible: after unrolling by N, the main loop's counter `iv` runs
/// `0, N, 2N, ...`, so it is always a multiple of N. A multiple of N has its
/// lowest bits all zero (with N = 4: `iv` is `0, 4, 8, 12, ...`, which in binary
/// always ends in `00`). Copy `j` of the body uses `iv + j`, and `(iv + j) & 3`
/// keeps only those low 2 bits, so it equals `j` -- a compile-time constant:
///
/// ```text
///   copy 0:  (iv + 0) & 3 = 0
///   copy 1:  (iv + 1) & 3 = 1
///   copy 2:  (iv + 2) & 3 = 2
///   copy 3:  (iv + 3) & 3 = 3
/// ```
///
/// That `& 3` is exactly the "which pipeline stage am I on" index in the gemm
/// kernel, and turning it into a constant is the main payoff of unrolling.
///
/// What this does: scan `block` for `x & MASK` where `x` is the counter plus a
/// constant offset (so `x == iv + offset`) and `MASK` is a low-bit mask of the
/// form `2^k - 1` (e.g. 1, 3, 7) small enough that the counter contributes only
/// zeros to it. For each, replace every use of the result with the constant
/// `(init + offset) & MASK`. The leftover `x & MASK` is now unused and removed
/// later by dead-code elimination.
///
/// Parameters: `iv` is the counter; `init` its starting value; `modulus` the
/// unrolled step (`N * original_step`). Together: `iv = init + modulus * t` for
/// iteration `t`, which is why `iv`'s low bits match `init`'s.
fn fold_iv_congruences(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    iv: Value,
    init: i128,
    modulus: i128,
) {
    if modulus <= 0 {
        return;
    }
    let ops: Vec<Ptr<Operation>> = block.deref(ctx).iter(ctx).collect();
    for op in ops {
        if Operation::get_op::<MirBitAndOp>(op, ctx).is_none() {
            continue;
        }
        let a = op.deref(ctx).get_operand(0);
        let b = op.deref(ctx).get_operand(1);
        // One operand affine in the IV, the other a constant mask.
        let (offset, mask) =
            if let (Some(o), Some(m)) = (affine_offset(ctx, a, iv), induction::const_i128(ctx, b)) {
                (o, m)
            } else if let (Some(o), Some(m)) =
                (affine_offset(ctx, b, iv), induction::const_i128(ctx, a))
            {
                (o, m)
            } else {
                continue;
            };
        // `mask` must be a low-bit mask (2^k - 1) whose `mask+1` divides `modulus`.
        let m1 = mask + 1;
        if mask < 0 || m1 == 0 || (m1 & mask) != 0 || modulus % m1 != 0 {
            continue;
        }
        let folded = (init + offset) & mask;
        let result = op.deref(ctx).get_result(0);
        let ty = result.get_type(ctx);
        let lit = make_const(ctx, ty, folded, op);
        result.replace_all_uses_with(ctx, &lit);
    }
}

/// Build an integer constant `value` of type `ty`, add it as the last op of
/// `block`, and return the value it produces. (Same as [`make_const`] but
/// appends to a block instead of inserting before a given op.)
fn append_const(ctx: &mut Context, ty: TypeHandle, value: i128, block: Ptr<BasicBlock>) -> Value {
    let typed = TypedHandle::<IntegerType>::from_handle(ty, ctx).expect("integer type");
    let width = typed.deref(ctx).width() as usize;
    let apint = APInt::from_i64(value as i64, NonZero::new(width).expect("non-zero width"));
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
fn append_add(ctx: &mut Context, ty: TypeHandle, a: Value, b: Value, block: Ptr<BasicBlock>) -> Value {
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
fn collect_hints(ctx: &Context, region: Ptr<Region>) -> Vec<(Ptr<Operation>, Ptr<BasicBlock>, u32)> {
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
