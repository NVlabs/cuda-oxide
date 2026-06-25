/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Works out how a loop's counters and accumulators change each iteration.
//!
//! An **induction variable** (IV) is a value that changes by a fixed amount on
//! every trip through the loop: the `i` in `for i in 0..n` (it goes `0, 1, 2,
//! ...`), or one stepping by 4 to give `0, 4, 8, ...`. This analysis finds the
//! IVs of one loop and, when it can, how many times the loop runs.
//!
//! How the values flow. After mem2reg, a loop's per-iteration values live as the
//! header block's **block arguments** (mem2reg is the pass that turns memory
//! slots into SSA values; "block arguments" are pliron's way of passing values
//! into a block, the same role LLVM gives to phi nodes). Each predecessor branch
//! supplies the values for those arguments. So a header argument gets its value
//! from two edges: the **preheader edge** (the branch that enters the loop)
//! gives the starting value, and the **latch edge** (the branch that loops back)
//! gives the value to use next time round.
//!
//! This analysis looks at those two values for each header argument and labels
//! it one of:
//!
//!   * **Basic induction variable.** The latch feeds back `arg + c` for a
//!     constant `c` (the per-iteration step). With a constant starting value we
//!     describe it by a **recurrence** `{init, step}`, meaning value =
//!     `init + step * iteration_number`. For example `{0, 4}` is the sequence
//!     `0, 4, 8, 12, ...`. Such an IV is always a multiple of `step` away from
//!     `init`; that fact ("`arg` is **congruent** to `init` modulo `step`",
//!     i.e. `arg` and `init` leave the same remainder when divided by `step`) is
//!     what the unroller exploits when it folds `& mask` operations to
//!     constants.
//!   * **Reduction** (a loop-carried accumulator). A value carried across
//!     iterations and updated by something other than a constant step, e.g.
//!     `acc = acc + (i & 3)`. It is not a counter, so we cannot replace it with
//!     a formula; the unroller just threads it from one unrolled body copy to
//!     the next.
//!   * **Invariant.** Fed back unchanged, so it has the same value every
//!     iteration.
//!
//! The **trip count** is how many times the loop body runs. We read it off the
//! header's exit test `IV <pred> bound` (e.g. `i < 16`) when `init`, `step`, and
//! a constant `bound` are all known. For `i = 0; i < 16; i += 4` the trip count
//! is 4.
//!
//! This is a small, reusable stand-in for full scalar evolution that the
//! unroller (and later loop passes) build on. It is deliberately cautious:
//! anything that does not match the simple counted-loop shape it recognises is
//! reported as `Unknown` / `None` rather than guessed at.

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

/// A comparison operator, the `<pred>` in a test written `lhs <pred> rhs`:
/// less-than, less-or-equal, greater-than, greater-or-equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpPred {
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpPred {
    /// The opposite test: the one that is true exactly when this one is false.
    /// (`<` becomes `>=`, and so on.)
    fn negate(self) -> CmpPred {
        match self {
            CmpPred::Lt => CmpPred::Ge,
            CmpPred::Le => CmpPred::Gt,
            CmpPred::Gt => CmpPred::Le,
            CmpPred::Ge => CmpPred::Lt,
        }
    }
    /// The test you get by swapping the two sides while keeping the same
    /// meaning: `a < b` is the same fact as `b > a`, so `swap(<)` is `>`.
    fn swap(self) -> CmpPred {
        match self {
            CmpPred::Lt => CmpPred::Gt,
            CmpPred::Gt => CmpPred::Lt,
            CmpPred::Le => CmpPred::Ge,
            CmpPred::Ge => CmpPred::Le,
        }
    }
}

/// What one header block argument turned out to be: a counter, an accumulator,
/// a constant-across-iterations value, or something we don't recognise.
#[derive(Debug, Clone)]
pub enum ArgKind {
    /// A counter: its value is `init + step * iteration`, so it forms the
    /// sequence `init, init+step, init+2*step, ...`.
    BasicIv { init: i128, step: i128 },
    /// A value carried across iterations and updated by something other than a
    /// fixed step, e.g. an accumulator `acc = acc + (i & 3)`.
    Reduction,
    /// The same value on every iteration (fed back unchanged).
    Invariant,
    /// Did not match any of the patterns above.
    Unknown,
}

/// Everything this analysis learned about one loop.
#[derive(Debug, Clone)]
pub struct LoopRecurrences {
    /// What each header argument is, in header-argument order: `args[i]`
    /// describes header argument `i`.
    pub args: Vec<ArgKind>,
    /// Which header argument is the counter the loop tests against to decide
    /// whether to keep going (its index in `args`), if we found one.
    pub primary_iv: Option<usize>,
    /// The loop's limit as a plain number, from a test `IV <pred> bound`, when
    /// `bound` is a compile-time constant.
    pub bound: Option<i128>,
    /// The same limit as an IR value rather than a number. The limit can be a
    /// value only known at runtime (e.g. an array length), which is fine for
    /// partial unrolling, so we keep the value here even when `bound` is `None`.
    pub bound_value: Option<Value>,
    /// The test that keeps the loop going: the body runs while
    /// `IV <continue_pred> bound` holds (e.g. `<` for `while i < n`).
    pub continue_pred: Option<CmpPred>,
    /// How many times the body runs, when `init`, `step`, `bound`, and the
    /// predicate are all known constants; `None` otherwise.
    pub trip_count: Option<u64>,
}

/// If `v` is a compile-time integer constant (a `mir.constant` op), return its
/// value; `None` if `v` is computed at runtime.
pub(crate) fn const_i128(ctx: &Context, v: Value) -> Option<i128> {
    let def = v.defining_op()?;
    let c = Operation::get_op::<MirConstantOp>(def, ctx)?;
    let attr = c.get_attr_value(ctx)?;
    Some(attr.value().to_i128())
}

/// The values that the branch from `pred` to `header` supplies for `header`'s
/// block arguments. (`pred`'s last instruction is its branch; we find the slot
/// that targets `header` and read off the values passed along that edge.)
/// Returns `None` if `pred` does not actually branch to `header`.
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

/// Peel off any run of `mir.not` (boolean negation) wrapping `v`. Returns the
/// value underneath and whether the number peeled off was odd (so `!!x` reports
/// "not negated" and `!x` reports "negated"). Used so a guard written `!(i < n)`
/// is understood the same as `i >= n`.
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

/// If `op` is a comparison (`<`, `<=`, `>`, `>=`), return which one it is and
/// its two sides (left, right); `None` for any other op.
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

/// Run the full analysis on loop `id`: classify every header argument, find the
/// counter the loop tests, and compute the trip count when possible. `preheader`
/// is the block that enters the loop (see [`LoopInfo::preheader`]); we read each
/// counter's starting value off the branch from it.
///
/// [`LoopInfo::preheader`]: crate::analyses::loop_info::LoopInfo::preheader
pub fn analyze(
    ctx: &Context,
    info: &LoopInfo,
    id: LoopId,
    preheader: Ptr<BasicBlock>,
) -> LoopRecurrences {
    let l = &info.loops()[id];
    let header = l.header;
    let nargs = header.deref(ctx).get_num_arguments();
    let header_args: Vec<Value> = (0..nargs)
        .map(|i| header.deref(ctx).get_argument(i))
        .collect();

    // Starting values come in on the preheader edge; next-iteration values come
    // in on the latch edge. A simple counted loop has exactly one latch; if it
    // has more we leave `latch_ops` as None and classify everything as Unknown.
    let pre_ops = edge_operands(ctx, preheader, header);
    let latch_ops = l
        .latches
        .first()
        .copied()
        .and_then(|latch| edge_operands(ctx, latch, header));

    // Classify each header argument.
    let mut args = Vec::with_capacity(nargs);
    for (i, &arg) in header_args.iter().enumerate() {
        args.push(classify_arg(
            ctx,
            arg,
            i,
            pre_ops.as_deref(),
            latch_ops.as_deref(),
        ));
    }

    // Read the header's exit test to find the counter it checks, the limit, and
    // the keep-going predicate.
    let (primary_iv, bound, bound_value, continue_pred) =
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
        bound_value,
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
    // Fed back unchanged -> same value every iteration.
    if latch_val == arg {
        return ArgKind::Invariant;
    }
    // Is the fed-back value `arg + c`, `c + arg`, or `arg - c` for a constant c?
    // If so, c is the per-iteration step and this is a counter.
    let step = step_of(ctx, latch_val, arg);
    if let Some(step) = step {
        if let Some(init) = pre_ops
            .and_then(|o| o.get(i).copied())
            .and_then(|v| const_i128(ctx, v))
        {
            return ArgKind::BasicIv { init, step };
        }
        // Steps like a counter, but its starting value isn't a constant, so we
        // can't give it a numeric formula; treat it as a carried value instead.
        return ArgKind::Reduction;
    }
    // Changes each iteration but not by a fixed step: an accumulator.
    ArgKind::Reduction
}

/// If `v` is `arg + c`, `c + arg`, or `arg - c` for a constant `c`, return the
/// per-iteration step (`c`, or `-c` for the subtraction). `None` otherwise.
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

/// Read the header's conditional branch (its `i < n`-style exit test) and pull
/// out three things: which header argument is the counter being tested, the
/// limit it is compared against (as a constant when possible, always as a
/// value), and the predicate under which the body keeps running (so the loop
/// continues while `IV <pred> bound`). Returns all-`None` if the header doesn't
/// have a recognisable counted-loop test.
fn analyze_guard(
    ctx: &Context,
    info: &LoopInfo,
    id: LoopId,
    header_args: &[Value],
    args: &[ArgKind],
) -> (Option<usize>, Option<i128>, Option<Value>, Option<CmpPred>) {
    let l = &info.loops()[id];
    let term = match l.header.deref(ctx).get_terminator(ctx) {
        Some(t) => t,
        None => return (None, None, None, None),
    };
    let succs: Vec<Ptr<BasicBlock>> = term.deref(ctx).successors().collect();
    if succs.len() != 2 {
        return (None, None, None, None);
    }
    // The header branches two ways: into the body, or out of the loop. Find
    // which of the two successors is the body (the one still inside the loop).
    let body_idx = if l.blocks.contains(&succs[0]) {
        0
    } else if l.blocks.contains(&succs[1]) {
        1
    } else {
        return (None, None, None, None);
    };
    // The branch's first operand is the boolean condition; the body is the
    // successor taken when that condition is true (successor 0 is the true side).
    // Peel off any `!` so we compare against the real underlying test, and track
    // whether peeling flipped the sense.
    let cond = term.deref(ctx).get_operand(0);
    let (cmp_val, negated) = unwrap_not(ctx, cond);
    let body_when_cmp_true = (body_idx == 0) ^ negated;

    let def = match cmp_val.defining_op() {
        Some(d) => d,
        None => return (None, None, None, None),
    };
    let (pred_written, lhs, rhs) = match match_cmp(ctx, def) {
        Some(t) => t,
        None => return (None, None, None, None),
    };
    // We want the predicate that is true when the body runs. If the body runs
    // when the comparison is true, that's the comparison itself; if it runs when
    // the comparison is false, flip the comparison to its opposite.
    let mut pred = if body_when_cmp_true {
        pred_written
    } else {
        pred_written.negate()
    };

    // The test could be written `i < n` or `n > i`. Figure out which side is the
    // counter (a header argument) and which is the limit, and if the counter is
    // on the right, swap the predicate so we always end up with `IV <pred> bound`.
    let iv_is_lhs = header_args.iter().position(|&a| a == lhs);
    let iv_is_rhs = header_args.iter().position(|&a| a == rhs);
    let (iv_index, bound_val) = match (iv_is_lhs, iv_is_rhs) {
        (Some(idx), _) => (idx, rhs),
        (None, Some(idx)) => {
            pred = pred.swap();
            (idx, lhs)
        }
        _ => return (None, None, None, None),
    };
    // The thing being tested must actually be a counter for this to be a
    // counted loop.
    if !matches!(args[iv_index], ArgKind::BasicIv { .. }) {
        return (None, None, None, None);
    }
    (
        Some(iv_index),
        const_i128(ctx, bound_val),
        Some(bound_val),
        Some(pred),
    )
}

/// How many times the body runs for a loop that continues while
/// `IV <pred> bound`, given the counter's `init` and `step`. For example
/// `i = 0; i < 16; i += 4` gives 4. Returns `None` when the step points the
/// wrong way for the test (e.g. `i < n` while `i` decreases), which would be an
/// infinite or zero loop we don't count.
fn trip_count(init: i128, step: i128, bound: i128, pred: CmpPred) -> Option<u64> {
    let count = match pred {
        // Counting up toward an upper limit.
        CmpPred::Lt if step > 0 => div_ceil(bound - init, step),
        CmpPred::Le if step > 0 => div_ceil(bound - init + 1, step),
        // Counting down toward a lower limit.
        CmpPred::Gt if step < 0 => div_ceil(init - bound, -step),
        CmpPred::Ge if step < 0 => div_ceil(init - bound + 1, -step),
        _ => return None,
    };
    Some(count.max(0) as u64)
}

/// Divide and round up (so 5/4 is 2). Returns 0 when the numerator is zero or
/// negative, which corresponds to a loop that never runs.
fn div_ceil(num: i128, den: i128) -> i128 {
    if num <= 0 { 0 } else { (num + den - 1) / den }
}
