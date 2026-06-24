/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Induction-variable analysis for a single natural loop.
//!
//! After mem2reg a loop's carried values are the header's block arguments,
//! threaded by the preheader edge (initial values) and the latch edge
//! (next-iteration values). This analysis classifies each header argument and
//! derives the loop's trip count:
//!
//!   * **Basic induction variable** — the latch feeds back `arg + c` for a
//!     constant `c`; with a constant initial value it has a recurrence
//!     `{init, step}` and the congruence `arg ≡ init (mod step)`.
//!   * **Reduction** — carried and updated by a non-constant recurrence
//!     (e.g. `acc = acc + (i & 3)`); threaded through unrolled copies, not an IV.
//!   * **Invariant** — fed back unchanged.
//!
//! The **trip count** comes from the header's exit guard `IV <pred> bound` with
//! a constant `bound`. This is the reusable scalar-evolution-lite that the
//! unroller (and later LICM / strength reduction) consume. It is intentionally
//! conservative: anything not matching the recognised counted-loop shape yields
//! `Unknown` / `None` rather than a guess.

use dialect_mir::ops::arithmetic::{MirAddOp, MirNotOp, MirSubOp};
use dialect_mir::ops::comparison::{MirGeOp, MirGtOp, MirLeOp, MirLtOp};
use dialect_mir::ops::constants::MirConstantOp;
use pliron::basic_block::BasicBlock;
use pliron::builtin::op_interfaces::BranchOpInterface;
use pliron::context::{Context, Ptr};
use pliron::op::op_cast;
use pliron::operation::Operation;
use pliron::value::Value;

use crate::analyses::loop_info::{LoopId, LoopInfo};

/// A relational predicate as written `lhs <pred> rhs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpPred {
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpPred {
    /// Logical negation: the predicate that holds exactly when this one doesn't.
    fn negate(self) -> CmpPred {
        match self {
            CmpPred::Lt => CmpPred::Ge,
            CmpPred::Le => CmpPred::Gt,
            CmpPred::Gt => CmpPred::Le,
            CmpPred::Ge => CmpPred::Lt,
        }
    }
    /// The predicate with operands swapped: `a <pred> b` == `b <swapped> a`.
    fn swap(self) -> CmpPred {
        match self {
            CmpPred::Lt => CmpPred::Gt,
            CmpPred::Gt => CmpPred::Lt,
            CmpPred::Le => CmpPred::Ge,
            CmpPred::Ge => CmpPred::Le,
        }
    }
}

/// Classification of one header block argument.
#[derive(Debug, Clone)]
pub enum ArgKind {
    /// Basic induction variable with recurrence `arg = init + step*iteration`.
    BasicIv { init: i128, step: i128 },
    /// Carried value updated by a non-constant recurrence (e.g. an accumulator).
    Reduction,
    /// Fed back unchanged across iterations.
    Invariant,
    /// Not recognised.
    Unknown,
}

/// IV analysis result for one loop.
#[derive(Debug, Clone)]
pub struct LoopRecurrences {
    /// Per header-argument classification (index == header arg index).
    pub args: Vec<ArgKind>,
    /// Header-arg index of the IV used in the exit guard, if found.
    pub primary_iv: Option<usize>,
    /// Constant bound from the guard `IV <pred> bound`.
    pub bound: Option<i128>,
    /// The loop-continue predicate: the body runs while `IV <pred> bound`.
    pub continue_pred: Option<CmpPred>,
    /// Constant trip count, when init/step/bound/pred are all known.
    pub trip_count: Option<u64>,
}

/// Read the constant integer a value is defined by, if it is a `mir.constant`.
fn const_i128(ctx: &Context, v: Value) -> Option<i128> {
    let def = v.defining_op()?;
    let c = Operation::get_op::<MirConstantOp>(def, ctx)?;
    let attr = c.get_attr_value(ctx)?;
    Some(attr.value().to_i128())
}

/// The operands a predecessor's terminator passes to `header`'s block args.
pub(crate) fn edge_operands(
    ctx: &Context,
    pred: Ptr<BasicBlock>,
    header: Ptr<BasicBlock>,
) -> Option<Vec<Value>> {
    let term = pred.deref(ctx).get_terminator(ctx)?;
    let succs: Vec<Ptr<BasicBlock>> = term.deref(ctx).successors().collect();
    let idx = succs.iter().position(|&s| s == header)?;
    let opobj = Operation::get_op_dyn(term, ctx);
    let br = op_cast::<dyn BranchOpInterface>(opobj.as_ref())?;
    Some(br.successor_operands(ctx, idx))
}

/// Strip any chain of `mir.not` from a boolean value, returning the underlying
/// value and whether an odd number of negations was removed.
fn unwrap_not(ctx: &Context, mut v: Value) -> (Value, bool) {
    let mut negated = false;
    while let Some(def) = v.defining_op() {
        if Operation::get_op::<MirNotOp>(def, ctx).is_some() {
            v = def.deref(ctx).get_operand(0);
            negated = !negated;
        } else {
            break;
        }
    }
    (v, negated)
}

/// Match a comparison op, returning its predicate (as written) and operands.
fn match_cmp(ctx: &Context, op: Ptr<Operation>) -> Option<(CmpPred, Value, Value)> {
    let pred = if Operation::get_op::<MirLtOp>(op, ctx).is_some() {
        CmpPred::Lt
    } else if Operation::get_op::<MirLeOp>(op, ctx).is_some() {
        CmpPred::Le
    } else if Operation::get_op::<MirGtOp>(op, ctx).is_some() {
        CmpPred::Gt
    } else if Operation::get_op::<MirGeOp>(op, ctx).is_some() {
        CmpPred::Ge
    } else {
        return None;
    };
    let o = op.deref(ctx);
    Some((pred, o.get_operand(0), o.get_operand(1)))
}

/// Analyse the induction variables and trip count of loop `id`.
pub fn analyze(
    ctx: &Context,
    info: &LoopInfo,
    id: LoopId,
    preheader: Ptr<BasicBlock>,
) -> LoopRecurrences {
    let l = &info.loops()[id];
    let header = l.header;
    let nargs = header.deref(ctx).get_num_arguments();
    let header_args: Vec<Value> = (0..nargs).map(|i| header.deref(ctx).get_argument(i)).collect();

    let pre_ops = edge_operands(ctx, preheader, header);
    // Canonical counted loops have a single latch; bail (Unknown) otherwise.
    let latch_ops = l
        .latches
        .first()
        .copied()
        .and_then(|latch| edge_operands(ctx, latch, header));

    // Classify each header argument.
    let mut args = Vec::with_capacity(nargs);
    for i in 0..nargs {
        args.push(classify_arg(
            ctx,
            header_args[i],
            i,
            pre_ops.as_deref(),
            latch_ops.as_deref(),
        ));
    }

    // Exit guard -> primary IV, bound, continue predicate.
    let (primary_iv, bound, continue_pred) =
        analyze_guard(ctx, info, id, &header_args, &args);

    let trip_count = match (primary_iv, bound, continue_pred) {
        (Some(iv), Some(b), Some(p)) => match &args[iv] {
            ArgKind::BasicIv { init, step } => trip_count(*init, *step, b, p),
            _ => None,
        },
        _ => None,
    };

    LoopRecurrences {
        args,
        primary_iv,
        bound,
        continue_pred,
        trip_count,
    }
}

fn classify_arg(
    ctx: &Context,
    arg: Value,
    i: usize,
    pre_ops: Option<&[Value]>,
    latch_ops: Option<&[Value]>,
) -> ArgKind {
    let latch_val = match latch_ops.and_then(|o| o.get(i).copied()) {
        Some(v) => v,
        None => return ArgKind::Unknown,
    };
    if latch_val == arg {
        return ArgKind::Invariant;
    }
    // arg + c / c + arg / arg - c ?
    let step = step_of(ctx, latch_val, arg);
    if let Some(step) = step {
        if let Some(init) = pre_ops.and_then(|o| o.get(i).copied()).and_then(|v| const_i128(ctx, v))
        {
            return ArgKind::BasicIv { init, step };
        }
        // IV-shaped but non-constant init: treat as carried.
        return ArgKind::Reduction;
    }
    ArgKind::Reduction
}

/// If `v` is `arg + c`, `c + arg`, or `arg - c` for a constant `c`, return the step.
fn step_of(ctx: &Context, v: Value, arg: Value) -> Option<i128> {
    let def = v.defining_op()?;
    if Operation::get_op::<MirAddOp>(def, ctx).is_some() {
        let a = def.deref(ctx).get_operand(0);
        let b = def.deref(ctx).get_operand(1);
        if a == arg {
            return const_i128(ctx, b);
        }
        if b == arg {
            return const_i128(ctx, a);
        }
    } else if Operation::get_op::<MirSubOp>(def, ctx).is_some() {
        let a = def.deref(ctx).get_operand(0);
        let b = def.deref(ctx).get_operand(1);
        if a == arg {
            return const_i128(ctx, b).map(|c| -c);
        }
    }
    None
}

/// From the header's `cond_br`, find the IV header-arg, the constant bound, and
/// the predicate under which the body executes (`IV <pred> bound`).
fn analyze_guard(
    ctx: &Context,
    info: &LoopInfo,
    id: LoopId,
    header_args: &[Value],
    args: &[ArgKind],
) -> (Option<usize>, Option<i128>, Option<CmpPred>) {
    let l = &info.loops()[id];
    let term = match l.header.deref(ctx).get_terminator(ctx) {
        Some(t) => t,
        None => return (None, None, None),
    };
    let succs: Vec<Ptr<BasicBlock>> = term.deref(ctx).successors().collect();
    if succs.len() != 2 {
        return (None, None, None);
    }
    // Which successor stays in the loop (the body)?
    let body_idx = if l.blocks.contains(&succs[0]) {
        0
    } else if l.blocks.contains(&succs[1]) {
        1
    } else {
        return (None, None, None);
    };
    // cond_br operand 0 is the condition; the body is taken when cond == (body_idx == 0).
    let cond = term.deref(ctx).get_operand(0);
    let (cmp_val, negated) = unwrap_not(ctx, cond);
    let body_when_cmp_true = (body_idx == 0) ^ negated;

    let def = match cmp_val.defining_op() {
        Some(d) => d,
        None => return (None, None, None),
    };
    let (pred_written, lhs, rhs) = match match_cmp(ctx, def) {
        Some(t) => t,
        None => return (None, None, None),
    };
    // Continue predicate (as written) for "body runs": negate if body runs when cmp is false.
    let mut pred = if body_when_cmp_true {
        pred_written
    } else {
        pred_written.negate()
    };

    // Orient so the IV is on the left and the bound on the right.
    let iv_is_lhs = header_args.iter().position(|&a| a == lhs);
    let iv_is_rhs = header_args.iter().position(|&a| a == rhs);
    let (iv_index, bound_val) = match (iv_is_lhs, iv_is_rhs) {
        (Some(idx), _) => (idx, rhs),
        (None, Some(idx)) => {
            pred = pred.swap();
            (idx, lhs)
        }
        _ => return (None, None, None),
    };
    if !matches!(args[iv_index], ArgKind::BasicIv { .. }) {
        return (None, None, None);
    }
    (Some(iv_index), const_i128(ctx, bound_val), Some(pred))
}

/// Trip count for a loop whose body runs while `IV <pred> bound`, given the
/// IV's `init`/`step`. `None` when the direction/step don't form a finite count.
fn trip_count(init: i128, step: i128, bound: i128, pred: CmpPred) -> Option<u64> {
    let count = match pred {
        // Counting up.
        CmpPred::Lt if step > 0 => div_ceil(bound - init, step),
        CmpPred::Le if step > 0 => div_ceil(bound - init + 1, step),
        // Counting down.
        CmpPred::Gt if step < 0 => div_ceil(init - bound, -step),
        CmpPred::Ge if step < 0 => div_ceil(init - bound + 1, -step),
        _ => return None,
    };
    Some(count.max(0) as u64)
}

/// Ceiling division for non-negative results; clamps negatives to 0.
fn div_ceil(num: i128, den: i128) -> i128 {
    if num <= 0 {
        0
    } else {
        (num + den - 1) / den
    }
}
