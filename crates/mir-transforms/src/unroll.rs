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
use dialect_mir::ops::arithmetic::{MirAddOp, MirBitAndOp, MirSubOp};
use dialect_mir::ops::comparison::{MirGeOp, MirGtOp, MirLeOp, MirLtOp};
use dialect_mir::ops::constants::MirConstantOp;
use dialect_mir::ops::control_flow::{MirCondBranchOp, MirGotoOp};
use dialect_mir::ops::function::MirFuncOp;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::op_interfaces::OperandSegmentInterface;
use pliron::builtin::types::IntegerType;
use pliron::common_traits::Named;
use pliron::context::{Context, Ptr};
use pliron::graph::dominance::DomInfo;
use pliron::identifier::Identifier;
use pliron::irbuild::cloning::{IrMapping, clone_operation};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::opts::dce::dce;
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

use crate::analyses::induction::{self, ArgKind, CmpPred};
use crate::analyses::loop_info::LoopInfo;

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
            let unrolled = if factor == 0 {
                // Full unroll: needs a constant trip count.
                full_unroll(ctx, &info, region, *id, *ph, rec)?
            } else {
                // Partial unroll by `factor`, with a remainder loop for the tail.
                partial_unroll(ctx, &info, region, *id, *ph, rec, factor)?
            };
            changed |= unrolled;
        }
    }

    if changed {
        // Clean up: DCE removes the now-dead index `bitand`s (replaced by the
        // congruence fold) and full-unroll's dead IV increments; simplify_cfg
        // deletes the unreachable original loop blocks. Both reusable pliron passes.
        dce(module, ctx)?;
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

/// Partially unroll a counted loop by `factor`, leaving a remainder loop.
///
/// Keeps the original loop (`header`/`latch`) as the **remainder** that runs the
/// `trip % factor` tail one iteration at a time, and prepends a **main loop**
/// that does `factor` body copies per trip and steps the counter by
/// `factor*step`. The main loop runs only while a full group of `factor`
/// iterations remains; when fewer than that remain it falls into the original
/// loop. Counting-up loops (`<`/`<=`, positive step) only; bails otherwise.
///
/// Shape produced (entry was `preheader -> header`):
/// ```text
///   preheader -> main_h(init...)
///   main_h(acc,i): if (i + (factor-1)*step) <pred> bound  -> main_l  else -> header
///   main_l: <factor body copies, i+0,i+step,...>; goto main_h(acc', i + factor*step)
///   header/latch: the original loop, now the remainder for the tail
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
    // Counting-up loops only for now.
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

    // i1 type from the original guard's condition operand.
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

    // Two new blocks: the main (unrolled) header and body.
    let main_h = BasicBlock::new(ctx, None, arg_types);
    let main_l = BasicBlock::new(ctx, None, vec![]);
    main_h.insert_before(ctx, header);
    main_l.insert_before(ctx, header);
    let mh_args: Vec<Value> = (0..nargs).map(|i| main_h.deref(ctx).get_argument(i)).collect();
    let mh_iv = mh_args[iv_idx];

    // --- main_l: `factor` body copies, then step the counter by factor*step ---
    let mut running: Vec<Value> = mh_args.clone();
    for j in 0..factor {
        let mut mapper = IrMapping::new();
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

    // Step 5: the main-loop counter is a multiple of `factor*step` (it starts at
    // `iv_init` and steps by `factor*step`), so `mh_iv ≡ iv_init (mod factor*step)`.
    // Fold `(affine-in-IV) & mask` to the literal that congruence guarantees.
    fold_iv_congruences(ctx, main_l, mh_iv, iv_init, n * iv_step);

    // --- main_h: guard "a full group of `factor` fits", else fall to remainder ---
    let last_off = append_const(ctx, iv_type, (n - 1) * iv_step, main_h);
    let last = append_add(ctx, iv_type, mh_iv, last_off, main_h);
    let cont = append_cmp(ctx, pred, last, bound, i1_type, main_h);
    // cond true (group fits) -> main_l (no args); false -> header (remainder), carrying mh_args.
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

    // Enter the main loop instead of the original header (same init operands).
    Operation::replace_successor(p_term, ctx, 0, main_h);
    Ok(true)
}

/// The constant offset of `v` relative to `iv`, when `v` is `iv` plus/minus a
/// chain of integer constants (`v == iv + offset`). `None` otherwise.
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

/// Fold `mir.bitand(v, mask)` in `block` to a literal where `v` is affine in the
/// main-loop IV `iv` (which satisfies `iv ≡ init (mod modulus)`) and `mask + 1`
/// is a power of two dividing `modulus`. Then the masked low bits of `v` are
/// fully determined by the congruence, so `(iv + j) & (N-1)` folds to `j`. This
/// is the guaranteed partial-unroll index fold; the now-dead `bitand` is left
/// for DCE.
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

/// Append a `mir.constant` of integer type `ty` to the end of `block`.
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

/// Append a `mir.add` of integer type `ty` to the end of `block`.
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

/// Append a comparison `a <pred> b` (result `i1_type`) to the end of `block`.
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
