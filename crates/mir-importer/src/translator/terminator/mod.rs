/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Terminator translation: MIR terminators → `dialect-mir` control flow.
//!
//! This module translates MIR terminators (return, goto, call, switch, etc.)
//! into `dialect-mir` operations. GPU intrinsics from `cuda_device` are
//! expanded inline to `dialect-nvvm` operations.
//!
//! All non-entry blocks are argument-less: cross-block data flow travels
//! through the per-local alloca slots owned by [`ValueMap`], so every branch
//! terminator emitted here is zero-operand. For example, a MIR `goto` whose
//! successor reads a local set by the predecessor translates as:
//!
//! ```text
//! // Rust MIR
//! bb0: { _1 = 42_i32; goto -> bb1 }
//! bb1: { _0 = _1;     return }
//!
//! // dialect-mir (pre-mem2reg)
//! ^bb0:
//!   %s1 = mir.alloca          : !mir.ptr<i32>
//!   %c  = mir.constant 42_i32 : i32
//!   mir.store %c, %s1
//!   mir.goto ^bb1                    // zero-operand; _1 flows via %s1
//! ^bb1:                              // no block arguments
//!   %r = mir.load %s1 : i32
//!   mir.return %r : i32
//! ```
//!
//! The `mem2reg` pass (run later in [`crate::pipeline`]) folds these slot
//! round-trips into SSA, so the above collapses to a direct `mir.return %c`.
//!
//! # Function Name Resolution
//!
//! `extract_func_info` uses `CrateDef::name()` which returns fully qualified
//! names (FQDNs, e.g. `helper_fn::cuda_oxide_device_<hash>_vecadd`). This FQDN is
//! used as both `pattern_name` (for intrinsic matching against paths like
//! `cuda_device::thread::threadIdx_x`) and `call_name` (for non-generic calls).
//! The collector produces matching FQDNs, and the lowering layer converts
//! `::` to `__` on both sides.
//!
//! # Module Structure
//!
//! - [`helpers`]: Common utilities (`emit_goto`, `emit_function_call`)
//! - [`intrinsics`]: GPU intrinsic handlers organized by category:
//!   - `indexing`: Thread/block IDs, `index_1d`, `index_2d::<S>`, `index_2d_runtime`
//!   - `sync`: Barriers, mbarrier operations
//!   - `warp`: Shuffle, vote primitives
//!   - `wgmma`: Hopper matrix operations
//!   - `tcgen05`: Blackwell tensor core operations
//!   - `tma`: Tensor memory access
//!   - `memory`: SharedArray indexing, stmatrix

mod drop_glue;
pub mod helpers;
pub mod intrinsics;

use super::types;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::location::span_to_location;
use crate::translator::rvalue;
use crate::translator::values::{ValueMap, maybe_ptr_coerce};
use dialect_mir::ops::{
    MirAssertOp, MirCondBranchOp, MirConstantOp, MirEqOp, MirGotoOp, MirNotOp, MirReturnOp,
    MirUnrollHintOp,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::op_interfaces::OperandSegmentInterface;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::identifier::Legaliser;
use pliron::linked_list::ContainsLinkedList;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::Typed;
use pliron::{input_err, input_error};
use rustc_public::CrateDef;
use rustc_public::mir;
use rustc_public::ty::ConstantKind;
/// Translates a MIR terminator to Pliron IR control flow operation(s).
///
/// Handles all MIR terminator kinds:
/// - `Return`: Function return
/// - `Goto`: Unconditional branch
/// - `SwitchInt`: Multi-way branch (for enums, match)
/// - `Assert`: Runtime assertions with panic on failure
/// - `Call`: Function/intrinsic calls
/// - `Drop`: Destructor calls (no-op for Copy types)
/// - `Unreachable`: Marks unreachable code
///
/// # GPU Intrinsics
///
/// Calls to `cuda_device` functions are expanded inline to `dialect-nvvm` operations.
/// This includes thread indexing, synchronization, warp primitives, and
/// tensor core operations.
#[allow(clippy::too_many_arguments)]
pub fn translate_terminator(
    ctx: &mut Context,
    body: &mir::Body,
    term: &mir::Terminator,
    value_map: &mut ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    legaliser: &mut Legaliser,
) -> TranslationResult<Ptr<Operation>> {
    let loc = span_to_location(ctx, term.span);

    match &term.kind {
        mir::TerminatorKind::Return => {
            translate_return(ctx, body, value_map, block_ptr, prev_op, loc)
        }

        mir::TerminatorKind::Goto { target } => {
            translate_goto(ctx, *target, block_ptr, prev_op, block_map, loc)
        }

        mir::TerminatorKind::Assert {
            cond,
            expected,
            msg: _,
            target,
            unwind,
        } => translate_assert(
            ctx, body, cond, *expected, *target, unwind, block_ptr, prev_op, value_map, block_map,
            loc,
        ),

        mir::TerminatorKind::Call {
            func,
            args,
            destination,
            target,
            unwind,
        } => translate_call(
            ctx,
            body,
            func,
            args,
            destination,
            target,
            unwind,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            legaliser,
        ),

        mir::TerminatorKind::SwitchInt { discr, targets } => translate_switch(
            ctx, body, discr, targets, block_ptr, prev_op, value_map, block_map, loc,
        ),

        mir::TerminatorKind::Drop {
            place,
            target,
            unwind,
        } => translate_drop(
            ctx, body, place, *target, unwind, block_ptr, prev_op, block_map, loc,
        ),

        mir::TerminatorKind::Unreachable => {
            // Create an unreachable operation
            let op = Operation::new(
                ctx,
                dialect_mir::ops::MirUnreachableOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc);
            if let Some(prev) = prev_op {
                op.insert_after(ctx, prev);
            } else {
                op.insert_at_front(block_ptr, ctx);
            }
            Ok(op)
        }

        _ => input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "Terminator kind {:?} not yet implemented",
                term.kind
            ))
        ),
    }
}

// ============================================================================
// Core Terminator Handlers
// ============================================================================

/// Translates a MIR `Return` terminator to a `mir.return` operation.
///
/// Handles the return value (`_0`) from the function:
/// - For non-unit returns: passes the return value as an operand
/// - For unit returns (empty tuple): emits return with no operands
///
/// # MIR Semantics
///
/// In MIR, local 0 (`_0`) holds the return value. The return terminator
/// transfers control back to the caller with this value.
fn translate_return(
    ctx: &mut Context,
    body: &mir::Body,
    value_map: &mut ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    // MIR local `_0` holds the return value. In the alloca + load/store model
    // we emit a `mir.load` from its slot to materialise the SSA value, then
    // pass it as the `mir.return` operand. ZSTs (including `()` kernel
    // returns) have no slot, so we simply emit a bare `return`.
    let return_local = mir::Local::from(0usize);
    let return_decl = &body.locals()[return_local];
    let return_type = types::translate_type(ctx, &return_decl.ty)?;
    let is_unit_return = {
        use dialect_mir::types::MirTupleType;
        let return_type_obj = return_type.deref(ctx);
        if let Some(tuple_ty) = return_type_obj.downcast_ref::<MirTupleType>() {
            tuple_ty.get_types().is_empty()
        } else {
            false
        }
    };
    let loaded = value_map.load_local(ctx, return_local, block_ptr, prev_op);

    let (operands, terminator_prev_op) = match loaded {
        Some((load_op, val)) => {
            if is_unit_return {
                // Unit return: the load we just emitted is dead, but harmless;
                // leave it as prev_op so the return chains after it.
                (vec![], Some(load_op))
            } else {
                let (val, prev_op) =
                    maybe_ptr_coerce(ctx, val, return_type, block_ptr, Some(load_op));
                (vec![val], prev_op)
            }
        }
        None => {
            let return_ty = types::translate_type(ctx, &body.locals()[return_local].ty)?;
            let is_unit = return_ty
                .deref(ctx)
                .downcast_ref::<dialect_mir::types::MirTupleType>()
                .is_some_and(|tuple| tuple.get_types().is_empty());
            if types::is_zst_type(ctx, return_ty) && !is_unit {
                // Non-unit ZST returns still need a MIR-level value so the
                // `mir.return` verifier agrees with the function signature.
                // LLVM lowering erases the value and emits a void return.
                let undef = dialect_mir::ops::MirUndefOp::new(ctx, return_ty).get_operation();
                undef.deref_mut(ctx).set_loc(loc.clone());
                if let Some(prev) = prev_op {
                    undef.insert_after(ctx, prev);
                } else {
                    undef.insert_at_front(block_ptr, ctx);
                }
                (vec![undef.deref(ctx).get_result(0)], Some(undef))
            } else {
                (vec![], prev_op)
            }
        }
    };

    let op = Operation::new(
        ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![], // No results
        operands,
        vec![], // No successors
        0,      // No regions
    );
    op.deref_mut(ctx).set_loc(loc);

    if let Some(prev) = terminator_prev_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }

    Ok(op)
}

/// Translates a MIR `Goto` terminator to a zero-operand `mir.goto` operation.
///
/// Non-entry blocks carry no arguments; cross-block data flow travels through
/// per-local alloca slots instead.
#[allow(clippy::too_many_arguments)]
fn translate_goto(
    ctx: &mut Context,
    target: mir::BasicBlockIdx,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let target_idx: usize = target;
    let target_block = block_map[target_idx];

    let op = Operation::new(
        ctx,
        MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![target_block],
        0,
    );
    op.deref_mut(ctx).set_loc(loc);

    if let Some(prev) = prev_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }

    Ok(op)
}

/// Translates a MIR `Assert` terminator to a `mir.assert` operation.
///
/// Asserts that a condition matches the expected value, trapping on failure.
/// On success, branches to the target block.
///
/// # GPU Constraints
///
/// The CUDA toolchain does not support unwinding today; we treat all
/// unwind edges as unreachable.
///
/// # Condition Handling
///
/// If `expected == false`, the condition is negated before the assert:
/// - `assert!(cond, expected=true)` → assert condition is true
/// - `assert!(cond, expected=false)` → assert condition is false (negated)
#[allow(clippy::too_many_arguments)]
fn translate_assert(
    ctx: &mut Context,
    body: &mir::Body,
    cond: &mir::Operand,
    expected: bool,
    target: mir::BasicBlockIdx,
    unwind: &mir::UnwindAction,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    // The CUDA toolchain doesn't support stack unwinding today (the hardware
    // could, but nvcc/ptxas don't wire it up). We ignore the unwind action
    // and only generate code for the success path. External crates (like core)
    // may carry unwind edges in their MIR; those are dead code on GPU -- if a
    // panic occurs, the GPU thread traps.
    let _ = unwind;

    // Translate the condition operand
    let (cond_value, mut last_inserted) = match cond {
        mir::Operand::Copy(place) | mir::Operand::Move(place) => {
            rvalue::translate_place(ctx, body, place, value_map, block_ptr, prev_op, loc.clone())?
        }
        _ => {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(
                    "Constant conditions in assert not yet implemented".to_string(),
                )
            );
        }
    };

    // Apply negation if expected == false
    let final_cond = if !expected {
        let bool_type = types::get_bool_type(ctx);
        let not_op = Operation::new(
            ctx,
            MirNotOp::get_concrete_op_info(),
            vec![bool_type.to_handle()],
            vec![cond_value],
            vec![],
            0,
        );
        not_op.deref_mut(ctx).set_loc(loc.clone());

        if let Some(prev) = last_inserted {
            not_op.insert_after(ctx, prev);
        } else if let Some(prev) = prev_op {
            not_op.insert_after(ctx, prev);
        } else {
            not_op.insert_at_front(block_ptr, ctx);
        }

        last_inserted = Some(not_op);
        not_op.deref(ctx).get_result(0)
    } else {
        cond_value
    };

    // Alloca + load/store model: successor block has no arguments; assert
    // carries only its condition operand.
    let target_idx: usize = target;
    let target_block = block_map[target_idx];

    let (flat_operands, segment_sizes) =
        MirAssertOp::compute_segment_sizes(vec![vec![final_cond], vec![]]);

    let op = Operation::new(
        ctx,
        MirAssertOp::get_concrete_op_info(),
        vec![],
        flat_operands,
        vec![target_block],
        0,
    );
    Operation::get_op::<MirAssertOp>(op, ctx)
        .expect("MirAssertOp")
        .set_operand_segment_sizes(ctx, segment_sizes);
    op.deref_mut(ctx).set_loc(loc);

    if let Some(prev) = last_inserted {
        op.insert_after(ctx, prev);
    } else if let Some(prev) = prev_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }

    Ok(op)
}

/// Translates a MIR `SwitchInt` terminator to conditional branches.
///
/// Handles multi-way branches used for `match` expressions and enum dispatch:
///
/// # Boolean Switch (1 branch)
///
/// Uses `mir.cond_branch`:
/// - `switchInt(bool) → [0: bb_false, otherwise: bb_true]`
/// - Creates comparison or negation as needed
///
/// # Multi-way Switch (N branches)
///
/// Creates a chain of conditional branches:
/// ```text
/// current:     cmp0 = (discr == v0); cond_br cmp0, t0, intermediate_1
/// intermediate_1: cmp1 = (discr == v1); cond_br cmp1, t1, intermediate_2
/// ...
/// intermediate_N: cmpN = (discr == vN); cond_br cmpN, tN, otherwise
/// ```
#[allow(clippy::too_many_arguments)]
fn translate_switch(
    ctx: &mut Context,
    body: &mir::Body,
    discr: &mir::Operand,
    targets: &mir::SwitchTargets,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    // Translate discriminant
    let (discr_value, last_op) = match discr {
        mir::Operand::Copy(place) | mir::Operand::Move(place) => {
            rvalue::translate_place(ctx, body, place, value_map, block_ptr, prev_op, loc.clone())?
        }
        _ => {
            rvalue::translate_operand(ctx, body, discr, value_map, block_ptr, prev_op, loc.clone())?
        }
    };

    let branches: Vec<_> = targets.branches().collect();
    let otherwise_idx: usize = targets.otherwise();

    // For bool switches (2 branches), use MirCondBranchOp
    if branches.len() == 1 {
        let (val, target_bb) = branches[0];
        let target_idx: usize = target_bb;

        // For MirCondBranchOp, we need an i1 (boolean) condition.
        // If discr is already i1, we can use it directly (with appropriate target ordering).
        // Otherwise, we need to create a comparison: (discr == val).
        use pliron::r#type::Typed;
        let discr_ty = discr_value.get_type(ctx);
        let bool_ty = types::get_bool_type(ctx);

        let (condition, last_inserted_op) = if discr_ty == bool_ty.to_handle() {
            // discr is already i1
            // For boolean switch: val=0 means "if false", val=1 means "if true"
            // switchInt(bool) -> [0: bb_false, otherwise: bb_true]
            // Since val == 0 means "go to target when discr == 0", we need condition = !discr
            if val == 0 {
                // Create NOT operation: condition = !discr
                let not_op = Operation::new(
                    ctx,
                    MirNotOp::get_concrete_op_info(),
                    vec![bool_ty.to_handle()],
                    vec![discr_value],
                    vec![],
                    0,
                );
                not_op.deref_mut(ctx).set_loc(loc.clone());
                if let Some(prev) = last_op {
                    not_op.insert_after(ctx, prev);
                } else {
                    not_op.insert_at_front(block_ptr, ctx);
                }
                let cond = not_op.deref(ctx).get_result(0);
                (cond, Some(not_op))
            } else {
                // val == 1: condition is discr itself
                (discr_value, last_op)
            }
        } else {
            // discr is not i1 (e.g., u32 from lane_id(), or enum discriminant)
            // Create comparison: condition = (discr == val)
            let (width, signedness) =
                if let Some(int_ty) = discr_ty.deref(ctx).downcast_ref::<IntegerType>() {
                    (int_ty.width() as usize, int_ty.signedness())
                } else {
                    (64, Signedness::Unsigned) // Default to 64-bit unsigned if we can't determine
                };

            // Create constant for val with SAME type as discriminant.
            // SwitchInt values are u128 bit patterns at the discriminant's
            // width; the dialect stores tags as u64 (same limit as
            // MirEnumType::variant_discriminants in types.rs), so values
            // that need more than 64 bits must fail loudly instead of
            // silently truncating.
            let switch_val = u64::try_from(val).map_err(|_| {
                input_error!(
                    loc.clone(),
                    TranslationErr::unsupported(format!(
                        "SwitchInt value {} does not fit in 64 bits",
                        val
                    ))
                )
            })?;
            let width_nz = NonZeroUsize::new(width).unwrap();
            let apint = APInt::from_u64(switch_val, width_nz);
            let int_attr = pliron::builtin::attributes::IntegerAttr::new(
                IntegerType::get(ctx, width as u32, signedness),
                apint,
            );

            let const_op = Operation::new(
                ctx,
                MirConstantOp::get_concrete_op_info(),
                vec![discr_ty],
                vec![],
                vec![],
                0,
            );
            const_op.deref_mut(ctx).set_loc(loc.clone());
            let const_op_wrapped = MirConstantOp::new(const_op);
            const_op_wrapped.set_attr_value(ctx, int_attr);

            if let Some(prev) = last_op {
                const_op_wrapped.get_operation().insert_after(ctx, prev);
            } else {
                const_op_wrapped
                    .get_operation()
                    .insert_at_front(block_ptr, ctx);
            }
            let const_val = const_op_wrapped.get_operation().deref(ctx).get_result(0);

            // Create comparison operation: discr == val
            let eq_op = Operation::new(
                ctx,
                MirEqOp::get_concrete_op_info(),
                vec![bool_ty.to_handle()],
                vec![discr_value, const_val],
                vec![],
                0,
            );
            eq_op.deref_mut(ctx).set_loc(loc.clone());
            eq_op.insert_after(ctx, const_op_wrapped.get_operation());

            let cond = eq_op.deref(ctx).get_result(0);
            (cond, Some(eq_op))
        };

        // With condition = (discr == val) [or !discr for boolean val==0 case]:
        // true_target = target (go here when condition is true, i.e., discr == val)
        // false_target = otherwise (go here when condition is false)
        let true_idx = target_idx;
        let false_idx = otherwise_idx;

        let true_block = block_map[true_idx];
        let false_block = block_map[false_idx];

        // Alloca + load/store model: both branch successors are argument-less;
        // the cond_br carries only its boolean condition.
        let (flat_operands, segment_sizes) =
            MirCondBranchOp::compute_segment_sizes(vec![vec![condition], vec![], vec![]]);

        let op = Operation::new(
            ctx,
            MirCondBranchOp::get_concrete_op_info(),
            vec![],
            flat_operands,
            vec![true_block, false_block],
            0,
        );
        Operation::get_op::<MirCondBranchOp>(op, ctx)
            .expect("MirCondBranchOp")
            .set_operand_segment_sizes(ctx, segment_sizes);
        op.deref_mut(ctx).set_loc(loc);

        // Use last_inserted_op (which accounts for NOT/EQ ops created above)
        if let Some(prev) = last_inserted_op {
            op.insert_after(ctx, prev);
        } else if let Some(prev) = last_op {
            op.insert_after(ctx, prev);
        } else {
            op.insert_at_front(block_ptr, ctx);
        }

        return Ok(op);
    }

    // For multi-way switches, create a chain of conditional branches
    // switchInt(discr) -> [v0: t0, v1: t1, ..., otherwise: default]
    // Becomes:
    //   current_block: cmp0 = discr == v0; cond_br cmp0, t0, intermediate_1
    //   intermediate_1: cmp1 = discr == v1; cond_br cmp1, t1, intermediate_2
    //   ...
    //   intermediate_N-1: cmpN = discr == v(N-1); cond_br cmpN, t(N-1), default
    use pliron::r#type::Typed;

    let n = branches.len();
    let discr_ty = discr_value.get_type(ctx);
    let bool_ty = types::get_bool_type(ctx);
    let (width, signedness) =
        if let Some(int_ty) = discr_ty.deref(ctx).downcast_ref::<IntegerType>() {
            (int_ty.width() as usize, int_ty.signedness())
        } else {
            (64, Signedness::Unsigned) // Default to 64-bit unsigned
        };

    // Create N-1 intermediate blocks for the comparison chain
    let mut intermediate_blocks: Vec<Ptr<BasicBlock>> = Vec::new();
    let mut prev_block = block_ptr;
    for _ in 0..(n - 1) {
        let intermediate = BasicBlock::new(ctx, None, vec![]);
        intermediate.insert_after(ctx, prev_block);
        intermediate_blocks.push(intermediate);
        prev_block = intermediate;
    }

    // Process each branch in the chain
    let mut current_block_ptr = block_ptr;
    let mut current_prev_op = last_op;

    for (i, (val, target_bb)) in branches.iter().enumerate() {
        let target_idx: usize = *target_bb;
        let target_block = block_map[target_idx];

        // Determine the "else" block (next in chain or otherwise).
        let else_block: Ptr<BasicBlock> = if i < n - 1 {
            intermediate_blocks[i]
        } else {
            block_map[otherwise_idx]
        };

        // Create constant for comparison with SAME type as discriminant.
        // Same checked u128 -> u64 narrowing as the single-branch path
        // above: a silently truncated switch value would compare against
        // the wrong arm.
        let switch_val = u64::try_from(*val).map_err(|_| {
            input_error!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "SwitchInt value {} does not fit in 64 bits",
                    val
                ))
            )
        })?;
        let width_nz = NonZeroUsize::new(width).unwrap();
        let apint = APInt::from_u64(switch_val, width_nz);
        let int_attr = pliron::builtin::attributes::IntegerAttr::new(
            IntegerType::get(ctx, width as u32, signedness),
            apint,
        );

        let const_op = Operation::new(
            ctx,
            MirConstantOp::get_concrete_op_info(),
            vec![discr_ty],
            vec![],
            vec![],
            0,
        );
        const_op.deref_mut(ctx).set_loc(loc.clone());
        let const_op_wrapped = MirConstantOp::new(const_op);
        const_op_wrapped.set_attr_value(ctx, int_attr);

        if let Some(prev) = current_prev_op {
            const_op_wrapped.get_operation().insert_after(ctx, prev);
        } else {
            const_op_wrapped
                .get_operation()
                .insert_at_front(current_block_ptr, ctx);
        }

        let const_val = const_op_wrapped.get_operation().deref(ctx).get_result(0);

        // Create comparison: discr == val
        let cmp_op = Operation::new(
            ctx,
            MirEqOp::get_concrete_op_info(),
            vec![bool_ty.to_handle()],
            vec![discr_value, const_val],
            vec![],
            0,
        );
        cmp_op.deref_mut(ctx).set_loc(loc.clone());
        cmp_op.insert_after(ctx, const_op_wrapped.get_operation());

        let condition = cmp_op.deref(ctx).get_result(0);

        // Alloca + load/store model: argument-less successor blocks; the
        // cond_br carries only its boolean condition.
        let (flat_operands, segment_sizes) =
            MirCondBranchOp::compute_segment_sizes(vec![vec![condition], vec![], vec![]]);

        let branch_op = Operation::new(
            ctx,
            MirCondBranchOp::get_concrete_op_info(),
            vec![],
            flat_operands,
            vec![target_block, else_block],
            0,
        );
        Operation::get_op::<MirCondBranchOp>(branch_op, ctx)
            .expect("MirCondBranchOp")
            .set_operand_segment_sizes(ctx, segment_sizes);
        branch_op.deref_mut(ctx).set_loc(loc.clone());
        branch_op.insert_after(ctx, cmp_op);

        // Move to next intermediate block for next iteration
        if i < n - 1 {
            current_block_ptr = intermediate_blocks[i];
            current_prev_op = None;
        }
    }

    // Return the terminator from the original block
    let first_branch = block_ptr
        .deref(ctx)
        .iter(ctx)
        .last()
        .expect("Block should have terminator after multi-way switch translation");

    Ok(first_branch)
}

/// Translates a MIR `Drop` terminator.
///
/// rustc emits `TerminatorKind::Drop` only for places whose type has drop
/// glue. cuda-oxide does not yet emit device-side `drop_in_place` calls,
/// so a destructor that actually does something cannot run on the device.
///
/// Two cases:
///
/// 1. **Provably no-op glue**: when the monomorphized drop glue does
///    nothing observable (checked by [`drop_glue::drop_glue_is_noop`]),
///    the terminator lowers to a plain branch to its target block.
///    The common source pattern is `for x in arr` over a by-value
///    array: the loop's `core::array::IntoIter<T, N>` has an
///    `impl Drop`, but for element types without drop glue that
///    destructor folds to nothing.
///
/// 2. **Genuinely effectful glue**: we surface a hard error with the
///    dropped place's type so the user can diagnose and restructure
///    the kernel. Lowering to a goto here would silently skip the
///    destructor and miscompile.
///
/// Suppressing drop glue on a Copy-shaped value (e.g. wrapping in
/// `core::mem::ManuallyDrop`) prevents the Drop terminator from being
/// emitted in the first place and lets the kernel compile.
///
/// The unwind action is ignored: device code is panic=abort, and for the
/// no-op case there is nothing that could unwind anyway.
#[allow(clippy::too_many_arguments)]
fn translate_drop(
    ctx: &mut Context,
    body: &mir::Body,
    place: &mir::Place,
    target: mir::BasicBlockIdx,
    _unwind: &mir::UnwindAction,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let dropped_ty = place.ty(body.locals()).map_err(|e| {
        input_error!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "drop terminator: failed to compute place type: {e:?}"
            ))
        )
    })?;
    if drop_glue::drop_glue_is_noop(dropped_ty) {
        return translate_goto(ctx, target, block_ptr, prev_op, block_map, loc);
    }
    input_err!(
        loc,
        TranslationErr::unsupported(format!(
            "drop of `{:?}` is not supported on the device; its destructor \
             does observable work and cuda-oxide does not yet emit \
             device-side `drop_in_place` calls. Restructure the kernel to \
             use only `Copy` types, or wrap the value in \
             `core::mem::ManuallyDrop` to suppress drop glue.",
            dropped_ty.kind()
        ))
    )
}

// ============================================================================
// Call Translation (includes intrinsic dispatch)
// ============================================================================

/// Translates a MIR `Call` terminator to Pliron IR operations.
///
/// This is the main entry point for function call translation. It handles:
///
/// 1. **Intrinsic dispatch**: Calls to `cuda_device::*` are expanded inline
///    to `dialect-nvvm` operations (thread IDs, barriers, warp ops, etc.)
///
/// 2. **Closure calls**: `FnOnce::call_once`, `FnMut::call_mut`, `Fn::call`
///    require unpacking tuple arguments before calling the closure body
///
/// 3. **Regular calls**: Other functions are emitted as `mir.call` operations
///
/// # GPU Constraints
///
/// Unwind edges are treated as unreachable (CUDA toolchain limitation, not HW).
///
/// # Flow
///
/// ```text
/// Call → extract_func_info → try_dispatch_intrinsic
///                         ↓ (if intrinsic)
///                         intrinsics::* handlers
///                         ↓ (if closure)
///                         translate_closure_call
///                         ↓ (otherwise)
///                         helpers::emit_function_call
/// ```
#[allow(clippy::too_many_arguments)]
fn translate_call(
    ctx: &mut Context,
    body: &mir::Body,
    func: &mir::Operand,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<mir::BasicBlockIdx>,
    unwind: &mir::UnwindAction,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    legaliser: &mut Legaliser,
) -> TranslationResult<Ptr<Operation>> {
    // See comment in translate_assert for rationale.
    let _ = unwind;

    // Convert target to Option<usize>
    let target_usize = target.map(|t| t);

    // Extract function info
    let (pattern_name, call_name, substs_str) = extract_func_info(func, &loc)?;

    // Helper to check if substitutions contain a type
    let substs_contains =
        |pattern: &str| -> bool { substs_str.as_ref().is_some_and(|s| s.contains(pattern)) };

    // Skip precondition_check calls - these are UB check assertions that are
    // dead code because we return false for RuntimeChecks(UbChecks).
    // The MIR still contains these calls, but they're in dead branches.
    if let Some(ref name) = pattern_name
        && name.contains("precondition_check")
    {
        // Just emit a goto to the target block, skipping the call entirely
        if let Some(target_idx) = target_usize {
            let actual_prev_op = if let Some(p) = prev_op {
                p
            } else {
                // Create a dummy i1 constant (false) as a placeholder operation
                use pliron::builtin::attributes::IntegerAttr;
                use pliron::utils::apint::APInt;
                use std::num::NonZeroUsize;

                let bool_ty = IntegerType::get(ctx, 1, Signedness::Signless);
                let dummy = Operation::new(
                    ctx,
                    MirConstantOp::get_concrete_op_info(),
                    vec![bool_ty.into()],
                    vec![],
                    vec![],
                    0,
                );
                dummy.deref_mut(ctx).set_loc(loc.clone());
                let const_op = MirConstantOp::new(dummy);
                let false_val = APInt::from_u64(0, NonZeroUsize::new(1).unwrap());
                const_op.set_attr_value(ctx, IntegerAttr::new(bool_ty, false_val));
                let dummy = const_op.get_operation();
                dummy.insert_at_front(block_ptr, ctx);
                dummy
            };
            return Ok(helpers::emit_goto(
                ctx,
                target_idx,
                actual_prev_op,
                block_map,
                loc,
            ));
        }
    }

    // Identify the actual core callable-trait methods. Matching text in an
    // arbitrary function name is not sufficient: a user function named, for
    // example, `call_once_helper` must remain an ordinary call.
    if let Some(call_info) = callable_trait_call_info(func)
        && !args.is_empty()
    {
        if receiver_is_closure(&args[0], body) {
            return translate_closure_call(
                ctx,
                body,
                &call_name,
                call_info.resolved_is_shim,
                args,
                destination,
                &target_usize,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
                legaliser,
            );
        }
        if let Some(function_item) = extract_function_item_target(&args[0], body, &loc)? {
            return translate_function_item_call(
                ctx,
                body,
                &function_item,
                args,
                destination,
                &target_usize,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
                legaliser,
            );
        }
    }

    // Per-loop unroll marker from `#[unroll]` / `#[unroll(N)]`. The enclosing
    // `#[kernel]` or `#[device]` macro injects this call at the start of the loop
    // body, so we plant a `mir.unroll_hint` op right here, inside that loop
    // body. The loop-unroll pass later maps the hint back to its enclosing loop
    // (via LoopInfo) and consumes it, so it never reaches lowering. The factor
    // is the call's const generic (`0` = full unroll). Then branch to the
    // target as usual.
    // Match both the full path (`cuda_device::thread::__unroll_config`) and the
    // re-exported short path (`cuda_device::__unroll_config`), mirroring the
    // robust suffix match in `body::detect_unroll_config`.
    if is_cuda_device_const_marker(func, "__unroll_config") {
        let Some(factor) = extract_unroll_factor(func) else {
            return input_err!(
                loc,
                TranslationErr::invalid_op(
                    "could not read the const-generic factor from an unroll marker",
                )
            );
        };
        if factor == 1 || factor > 1024 {
            return input_err!(
                loc,
                TranslationErr::invalid_op(format!(
                    "partial unroll factor must be in 2..=1024, or 0 for full unrolling; got {factor}"
                ))
            );
        }
        let Some(target) = target_usize else {
            return input_err!(
                loc,
                TranslationErr::invalid_op("an unroll marker call has no return target")
            );
        };
        let hint = MirUnrollHintOp::new(ctx, factor).get_operation();
        hint.deref_mut(ctx).set_loc(loc.clone());
        match prev_op {
            Some(prev) => hint.insert_after(ctx, prev),
            None => hint.insert_at_front(block_ptr, ctx),
        }
        return Ok(helpers::emit_goto(ctx, target, hint, block_map, loc));
    }

    // Handle DynamicSharedArray specially to extract the ALIGN const generic
    if let Some(ref name) = pattern_name
        && name.contains("DynamicSharedArray")
        && (name.contains("::get") || name.contains("::offset"))
    {
        // Extract the ALIGN const generic from the function type
        // DynamicSharedArray<T, ALIGN> has T as first generic, ALIGN as second
        let alignment = if let mir::Operand::Constant(const_op) = func {
            if let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnDef(_, substs)) =
                const_op.const_.ty().kind()
            {
                // The ALIGN const generic is the second generic argument (index 1)
                // First is T (type), second is ALIGN (const)
                if let Some(rustc_public::ty::GenericArgKind::Const(c)) = substs.0.get(1) {
                    use rustc_public::ty::TyConstKind;
                    match c.kind() {
                        TyConstKind::Value(_, alloc) => alloc.read_uint().unwrap_or(16) as u64,
                        _ => c.eval_target_usize().unwrap_or(16),
                    }
                } else {
                    16 // Default alignment (matches nvcc)
                }
            } else {
                16
            }
        } else {
            16
        };

        if name.contains("::get") {
            // Both get() and get_raw() use the same handler with offset 0
            return intrinsics::memory::emit_dynamic_shared_get(
                ctx,
                body,
                args,
                destination,
                &target_usize,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
                0,         // byte_offset = 0 for get() and get_raw()
                alignment, // User-specified or default alignment
            );
        } else if name.contains("::offset") {
            // DynamicSharedArray::offset(byte_offset) - get pointer at byte offset
            return intrinsics::memory::emit_dynamic_shared_offset(
                ctx,
                body,
                args,
                destination,
                &target_usize,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
                alignment, // User-specified or default alignment
            );
        }
    }

    // Try to dispatch core::sync::atomic intrinsics (std::intrinsics::atomic_*)
    // These use const generics for ordering, so we intercept them here before
    // the regular intrinsic dispatch and extract generics from the func operand.
    if let Some(ref name) = pattern_name
        && intrinsics::atomic::is_core_atomic_intrinsic(name)
    {
        return intrinsics::atomic::dispatch_core_intrinsic(
            ctx,
            body,
            func,
            args,
            destination,
            &target_usize,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            name,
        );
    }

    // `assert_inhabited::<T>()` is a compile-time validity check that rustc
    // plants in `MaybeUninit::assume_init_read`, which the `for x in arr`
    // loop machinery calls for every yielded element (issue #138). The
    // intrinsic panics only when `T` has no possible values at all (an
    // "uninhabited" type such as `core::convert::Infallible`); for any
    // ordinary type it compiles to nothing. We decide which case applies
    // from the monomorphized type's layout: uninhabited types are exactly
    // those whose layout has `VariantsShape::Empty`. Inhabited types lower
    // to a unit no-op; uninhabited ones lower to `unreachable`, matching
    // how device code models panics (they cannot execute on the GPU).
    // If the generic argument or its layout cannot be read, fall through
    // to the loud "not yet supported" rejection below.
    if let Some(ref name) = pattern_name
        && (name == "core::intrinsics::assert_inhabited"
            || name == "std::intrinsics::assert_inhabited")
        && let mir::Operand::Constant(const_op) = func
        && let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnDef(_, substs)) =
            const_op.const_.ty().kind()
        && let Some(rustc_public::ty::GenericArgKind::Type(checked_ty)) = substs.0.first()
        && let Ok(layout) = checked_ty.layout()
    {
        let uninhabited = matches!(
            layout.shape().variants,
            rustc_public::abi::VariantsShape::Empty
        );
        if uninhabited {
            let op = Operation::new(
                ctx,
                dialect_mir::ops::MirUnreachableOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc);
            if let Some(prev) = prev_op {
                op.insert_after(ctx, prev);
            } else {
                op.insert_at_front(block_ptr, ctx);
            }
            return Ok(op);
        }
        return helpers::emit_unit_noop_intrinsic(
            ctx,
            destination,
            &target_usize,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            name,
        );
    }

    // Try to dispatch as intrinsic
    if let Some(ref name) = pattern_name
        && let Some(result) = try_dispatch_intrinsic(
            ctx,
            body,
            func,
            name,
            args,
            destination,
            &target_usize,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc.clone(),
            &substs_contains,
        )?
    {
        return Ok(result);
    }

    // Handle diverging calls (calls that never return, like unwrap_failed, panic, etc.)
    // These have no target block because the function never returns.
    // In GPU code, we emit an unreachable terminator since panics can't actually execute.
    if target_usize.is_none() {
        // This is a diverging call (returns !) - emit unreachable
        // Examples: unwrap_failed(), panic!(), abort()
        return Ok(emit_unreachable_after(ctx, block_ptr, prev_op, loc));
    }

    // A call to a rustc intrinsic that no dispatch arm above recognized can
    // never be emitted as a regular function call: rustc resolves intrinsics
    // to `InstanceKind::Intrinsic`, the collector skips those by design, so
    // no definition for the symbol will ever exist in the module. Emitting
    // the call anyway would only fail much later, as a confusing
    // "Symbol ... not found" verifier error on the LLVM dialect module.
    // Fail here instead, with the intrinsic's name and source location, so
    // each gap surfaces as an actionable per-site diagnostic (issue #137).
    if let Some(ref name) = pattern_name
        && (name.starts_with("core::intrinsics::") || name.starts_with("std::intrinsics::"))
    {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "rustc intrinsic `{name}` is not yet supported on the device"
            ))
        );
    }

    // The collector skips the entire `libm` crate: every libm call must be
    // intercepted by the float-math dispatch above and rerouted to a
    // libdevice intrinsic, so no definition for a libm symbol ever exists in
    // the module. A libm function the dispatch does not recognize would only
    // fail much later, as a bare "Symbol libm__cbrtf not found" verifier
    // error on the LLVM dialect module. Fail here instead, with the
    // function's name and source location.
    if let Some(ref name) = pattern_name
        && intrinsics::float_math::is_libm_path(name)
    {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "libm function `{name}` is not yet mapped to a libdevice intrinsic; \
                 add it to `from_libm_path` in the mir-importer float-math dispatch"
            ))
        );
    }

    // Not an intrinsic - emit regular function call
    let raw_name = call_name.unwrap_or_else(|| "unknown_function".to_string());
    let legal_name = legaliser.legalise(&raw_name);

    // Type the call result from the caller's destination place, not from the
    // callee's declared signature. The declared signature of a trait method
    // is written against the trait, so its return type can be an unresolved
    // associated-type projection such as `<&Foo as Mul>::Output` (issue #133),
    // which the type translator cannot turn into a concrete layout. The
    // destination local in the caller's monomorphized MIR already has that
    // projection resolved (`Foo`), and it is by construction the exact type
    // the call result is stored into, so the `mir.call` result type and the
    // destination slot always agree. The callee `mir.func` return type is
    // independently derived from the callee body's return place, which is
    // normalized the same way, so caller and callee stay consistent.
    let return_type = types::translate_destination_type(ctx, body, destination, &loc)?;

    helpers::emit_function_call(
        ctx,
        body,
        &legal_name,
        args,
        destination,
        return_type,
        &target_usize,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
    )
}

/// Handle `FnOnce::call_once`, `FnMut::call_mut`, or `Fn::call` when the
/// receiver is a function item.
///
/// Rust-call ABI passes `(self, tuple_args)` to the trait shim, but the
/// function item's real body expects only the tuple elements as ordinary
/// arguments. Lowering the shim as an ordinary call leaves a dangling
/// `<fn item as FnOnce>::call_once` symbol because no MIR body is collected
/// for that shim.
#[allow(clippy::too_many_arguments)]
fn translate_function_item_call(
    ctx: &mut Context,
    body: &mir::Body,
    function_item: &FunctionItemTarget,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    legaliser: &mut Legaliser,
) -> TranslationResult<Ptr<Operation>> {
    use dialect_mir::ops::{MirCallOp, MirExtractFieldOp};
    use pliron::builtin::attributes::StringAttr;
    use pliron::identifier::Identifier;

    if function_item.requires_direct_dispatch {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "calling `{}` through Fn/FnMut/FnOnce is not yet supported because that target requires intrinsic or external dispatch; wrap it in a local `#[device]` function and pass the wrapper instead",
                function_item.name
            ))
        );
    }

    let return_type = types::translate_destination_type(ctx, body, destination, &loc)?;
    let callee = legaliser.legalise(&function_item.name).to_string();

    let mut unpacked_args = Vec::new();
    let mut last_op = prev_op;

    if let Some(tuple_arg) = args.get(1) {
        let (tuple_value, tuple_last_op) = rvalue::translate_operand(
            ctx,
            body,
            tuple_arg,
            value_map,
            block_ptr,
            last_op,
            loc.clone(),
        )?;
        last_op = tuple_last_op;

        let tuple_ty = tuple_value.get_type(ctx);
        let element_types: Option<Vec<_>> = {
            let tuple_ty_obj = tuple_ty.deref(ctx);
            tuple_ty_obj
                .downcast_ref::<dialect_mir::types::MirTupleType>()
                .map(|mir_tuple_ty| mir_tuple_ty.get_types().to_vec())
        };

        if let Some(element_types) = element_types {
            for (i, elem_ty) in element_types.iter().enumerate() {
                let extract_op = Operation::new(
                    ctx,
                    MirExtractFieldOp::get_concrete_op_info(),
                    vec![*elem_ty],
                    vec![tuple_value],
                    vec![],
                    0,
                );
                extract_op.deref_mut(ctx).set_loc(loc.clone());

                let mir_extract = MirExtractFieldOp::new(extract_op);
                mir_extract.set_attr_index(ctx, dialect_mir::attributes::FieldIndexAttr(i as u32));

                if let Some(prev) = last_op {
                    extract_op.insert_after(ctx, prev);
                } else {
                    extract_op.insert_at_front(block_ptr, ctx);
                }
                last_op = Some(extract_op);
                unpacked_args.push(extract_op.deref(ctx).get_result(0));
            }
        } else {
            unpacked_args.push(tuple_value);
        }
    }

    let call_op = Operation::new(
        ctx,
        MirCallOp::get_concrete_op_info(),
        vec![return_type],
        unpacked_args,
        vec![],
        0,
    );
    call_op.deref_mut(ctx).set_loc(loc.clone());

    call_op.deref_mut(ctx).attributes.set(
        Identifier::try_from("callee").unwrap(),
        StringAttr::new(callee),
    );

    if let Some(prev) = last_op {
        call_op.insert_after(ctx, prev);
    } else {
        call_op.insert_at_front(block_ptr, ctx);
    }

    if target.is_none() {
        return Ok(emit_unreachable_after(ctx, block_ptr, Some(call_op), loc));
    }

    let result_value = call_op.deref(ctx).get_result(0);
    let last_inserted = value_map
        .store_local(
            ctx,
            destination.local,
            result_value,
            block_ptr,
            Some(call_op),
        )
        .unwrap_or(call_op);

    if let Some(target_idx) = target {
        Ok(helpers::emit_goto(
            ctx,
            *target_idx,
            last_inserted,
            block_map,
            loc,
        ))
    } else {
        Ok(call_op)
    }
}

/// Handle closure trait method calls (FnOnce::call_once, FnMut::call_mut, Fn::call).
///
/// These calls pass arguments as a tuple, but the closure body expects unpacked args:
/// - MIR: `<{closure} as FnMut<(u32,)>>::call_mut(self_ref, tuple_args)`
/// - Closure body expects: `fn(self_ref, unpacked_arg1, unpacked_arg2, ...)`
///
/// We detect these calls and unpack the tuple argument before calling the closure.
///
/// ## Important: Closure Body Resolution
///
/// When an `Fn` or `FnMut` closure is requested through `FnOnce`,
/// `Instance::resolve` returns a `ClosureOnce` adapter shim rather than the
/// closure body. Other closure calls may resolve directly to the body. In
/// either case, extract the closure DefId from `args[0]` so the emitted device
/// call targets the body; `resolved_is_shim` separately controls receiver
/// adaptation.
///
/// See `device_closures/README.md` for detailed documentation.
#[allow(clippy::too_many_arguments)]
fn translate_closure_call(
    ctx: &mut Context,
    body: &mir::Body,
    call_name: &Option<String>,
    resolved_is_shim: bool,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    legaliser: &mut Legaliser,
) -> TranslationResult<Ptr<Operation>> {
    use dialect_mir::ops::{MirCallOp, MirExtractFieldOp};
    use pliron::builtin::attributes::{IntegerAttr, StringAttr};
    use pliron::identifier::Identifier;
    use pliron::r#type::Typed;
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    // Same reasoning as the regular-call path: the trait-level signature of
    // `FnOnce::call_once` types its result as the projection
    // `<{closure} as FnOnce<Args>>::Output`. The caller's destination local
    // carries the already-resolved concrete type, so use that.
    let return_type = types::translate_destination_type(ctx, body, destination, &loc)?;

    // Extract the closure body's name from the closure type in args[0]. This
    // avoids targeting a ClosureOnce adapter when instance resolution selected
    // one, while remaining correct for calls that resolve directly to the body.
    let closure_body_name = extract_closure_body_name(&args[0], body);

    let raw_callee = closure_body_name
        .or_else(|| call_name.as_ref().map(|s| s.to_string()))
        .unwrap_or_else(|| "unknown_closure".to_string());
    let callee = legaliser.legalise(&raw_callee).to_string();

    // Translate self argument (args[0])
    let (self_value, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Translate tuple argument (args[1])
    let (tuple_value, tuple_last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = tuple_last_op;

    // Determine whether bypassing a resolved adapter shim requires us to
    // reproduce its receiver borrow. A ClosureOnce shim for an `Fn`/`FnMut`
    // closure receives the closure by value but calls a body that expects a
    // reference. A genuine by-value `FnOnce` closure resolves directly to its
    // body and must stay by value. A receiver that MIR already passes by
    // reference needs no extra borrow.
    let receiver_needs_borrow = resolved_is_shim
        && operand_type(&args[0], body).is_some_and(|ty| {
            !matches!(
                ty.kind(),
                rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Ref(_, _, _))
            )
        });

    let self_arg = if receiver_needs_borrow {
        // Reproduce the adapter shim's borrow before calling the body directly.
        let self_ty = self_value.get_type(ctx);
        let ptr_ty = dialect_mir::types::MirPtrType::get(ctx, self_ty, true, 0);

        let ref_op = Operation::new(
            ctx,
            dialect_mir::ops::MirRefOp::get_concrete_op_info(),
            vec![ptr_ty.into()],
            vec![self_value],
            vec![],
            0,
        );
        ref_op.deref_mut(ctx).set_loc(loc.clone());

        // Set mutable attribute (true for &mut)
        let bool_type = IntegerType::get(ctx, 1, Signedness::Unsigned);
        let mutable_attr =
            IntegerAttr::new(bool_type, APInt::from_i64(1, NonZeroUsize::new(1).unwrap()));
        ref_op
            .deref_mut(ctx)
            .attributes
            .set(Identifier::try_from("mutable").unwrap(), mutable_attr);

        // Insert after previous op
        if let Some(prev) = last_op {
            ref_op.insert_after(ctx, prev);
        } else {
            ref_op.insert_at_front(block_ptr, ctx);
        }
        last_op = Some(ref_op);

        // Use the reference as self arg
        ref_op.deref(ctx).get_result(0)
    } else {
        // For call_mut/call: self is already a reference, use as-is
        self_value
    };

    // Build unpacked arguments, starting with the original or adapted receiver.
    let mut unpacked_args = vec![self_arg];

    // Unpack the tuple - extract each field
    let tuple_ty = tuple_value.get_type(ctx);
    let element_types: Option<Vec<_>> = {
        let tuple_ty_obj = tuple_ty.deref(ctx);
        tuple_ty_obj
            .downcast_ref::<dialect_mir::types::MirTupleType>()
            .map(|mir_tuple_ty| mir_tuple_ty.get_types().to_vec())
    };

    if let Some(element_types) = element_types {
        for (i, elem_ty) in element_types.iter().enumerate() {
            // Create extract_field operation
            let extract_op = Operation::new(
                ctx,
                MirExtractFieldOp::get_concrete_op_info(),
                vec![*elem_ty],
                vec![tuple_value],
                vec![],
                0,
            );
            extract_op.deref_mut(ctx).set_loc(loc.clone());

            let mir_extract = MirExtractFieldOp::new(extract_op);
            mir_extract.set_attr_index(ctx, dialect_mir::attributes::FieldIndexAttr(i as u32));

            // Insert after previous op
            if let Some(prev) = last_op {
                extract_op.insert_after(ctx, prev);
            } else {
                extract_op.insert_at_front(block_ptr, ctx);
            }
            last_op = Some(extract_op);

            // Get the extracted value
            let elem_value = extract_op.deref(ctx).get_result(0);
            unpacked_args.push(elem_value);
        }
    } else {
        // Not a tuple type - just pass as is (single arg case)
        unpacked_args.push(tuple_value);
    }

    // Now emit the call with unpacked arguments
    let call_op = Operation::new(
        ctx,
        MirCallOp::get_concrete_op_info(),
        vec![return_type],
        unpacked_args,
        vec![],
        0,
    );
    call_op.deref_mut(ctx).set_loc(loc.clone());

    // Set callee attribute
    let callee_attr = StringAttr::new(callee);
    call_op
        .deref_mut(ctx)
        .attributes
        .set(Identifier::try_from("callee").unwrap(), callee_attr);

    // Insert the call
    let call_op = if let Some(prev) = last_op {
        call_op.insert_after(ctx, prev);
        call_op
    } else {
        call_op.insert_at_front(block_ptr, ctx);
        call_op
    };

    if target.is_none() {
        return Ok(emit_unreachable_after(ctx, block_ptr, Some(call_op), loc));
    }

    // Store the call result into the destination local's slot.
    let result_value = call_op.deref(ctx).get_result(0);
    let last_inserted = value_map
        .store_local(
            ctx,
            destination.local,
            result_value,
            block_ptr,
            Some(call_op),
        )
        .unwrap_or(call_op);

    // Emit goto to target
    if let Some(target_idx) = target {
        Ok(helpers::emit_goto(
            ctx,
            *target_idx,
            last_inserted,
            block_map,
            loc,
        ))
    } else {
        Ok(call_op)
    }
}

fn emit_unreachable_after(
    ctx: &mut Context,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> Ptr<Operation> {
    let op = Operation::new(
        ctx,
        dialect_mir::ops::MirUnreachableOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc);
    if let Some(prev) = prev_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }
    op
}

/// True only when the rust-call receiver is itself a closure.
fn receiver_is_closure(receiver: &mir::Operand, body: &mir::Body) -> bool {
    let Some(ty) = operand_type(receiver, body) else {
        return false;
    };

    let inner = match ty.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Ref(_, inner, _)) => inner,
        _ => ty,
    };

    matches!(
        inner.kind(),
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Closure(_, _))
    )
}

struct FunctionItemTarget {
    name: String,
    requires_direct_dispatch: bool,
}

fn extract_function_item_target(
    receiver: &mir::Operand,
    body: &mir::Body,
    loc: &Location,
) -> TranslationResult<Option<FunctionItemTarget>> {
    let Some(ty) = operand_type(receiver, body) else {
        return Ok(None);
    };
    let inner = match ty.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Ref(_, inner, _)) => inner,
        _ => ty,
    };

    let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnDef(fn_def, substs)) =
        inner.kind()
    else {
        return Ok(None);
    };

    let Some(instance) = rustc_public::mir::mono::Instance::resolve(fn_def, &substs).ok() else {
        return Ok(None);
    };
    let crate_name = fn_def.krate().name;
    let generated_direct_call_only =
        intrinsics::generated::require_supported_raw_intrinsic(fn_def, loc)?.is_some();
    Ok(Some(FunctionItemTarget {
        name: function_item_call_name(instance),
        requires_direct_dispatch: fn_def.is_intrinsic()
            || instance.is_foreign_item()
            || !instance.has_body()
            || matches!(crate_name.as_str(), "cuda_device" | "cuda-device" | "libm")
            || generated_direct_call_only,
    }))
}

fn function_item_call_name(instance: rustc_public::mir::mono::Instance) -> String {
    if instance.is_foreign_item() || !instance.args().0.is_empty() {
        instance.mangled_name()
    } else {
        instance.name().to_string()
    }
}

fn operand_type(receiver: &mir::Operand, body: &mir::Body) -> Option<rustc_public::ty::Ty> {
    receiver.ty(body.locals()).ok()
}

struct CallableTraitCallInfo {
    resolved_is_shim: bool,
}

/// Recognize the three callable traits by compiler identity available through
/// `rustc_public`.
///
/// These methods are defined in `core`, use the rust-call ABI, and have an
/// exact parent/method pair (`Fn::call`, `FnMut::call_mut`, or
/// `FnOnce::call_once`). The combined check cannot be spoofed by a user item
/// whose name merely contains one of those strings. The resolved instance may
/// be either a shim or the closure body itself, so instance kind is returned as
/// lowering information rather than used as the recognition predicate.
fn callable_trait_call_info(func: &mir::Operand) -> Option<CallableTraitCallInfo> {
    use rustc_public::mir::mono::{Instance, InstanceKind};
    use rustc_public::ty::{Abi, RigidTy, TyKind};

    let mir::Operand::Constant(const_op) = func else {
        return None;
    };
    let TyKind::RigidTy(RigidTy::FnDef(fn_def, substs)) = const_op.const_.ty().kind() else {
        return None;
    };
    if fn_def.fn_sig().skip_binder().abi != Abi::RustCall || fn_def.krate().name.as_str() != "core"
    {
        return None;
    }

    let method_name = fn_def.def_id().name();
    let method = method_name.as_str().rsplit("::").next()?;
    let parent_name = fn_def.def_id().parent()?.name();
    let parent = parent_name.as_str().rsplit("::").next()?;
    let is_callable_method = matches!(
        (parent, method),
        ("Fn", "call") | ("FnMut", "call_mut") | ("FnOnce", "call_once")
    );
    if !is_callable_method {
        return None;
    }

    let instance = Instance::resolve(fn_def, &substs).ok()?;
    Some(CallableTraitCallInfo {
        resolved_is_shim: instance.kind == InstanceKind::Shim,
    })
}

/// Extracts the closure body's mangled name from a closure operand.
///
/// When instance resolution selects an adapter shim for a call like:
///   `<{closure} as FnOnce<(u32,)>>::call_once(closure_ref, args_tuple)`
///
/// the resolved call name identifies the shim, not the closure body. Extract
/// the closure's DefId from `args[0]` and resolve the body independently.
///
/// The closure argument can be:
/// - A direct closure value (type is `Closure(def, substs)`)
/// - A reference to a closure (type is `Ref(_, Closure(def, substs), _)`)
/// - A mutable reference (same pattern)
fn extract_closure_body_name(closure_arg: &mir::Operand, body: &mir::Body) -> Option<String> {
    let closure_ty = operand_type(closure_arg, body)?;

    // Unwrap references to get the actual closure type
    let inner_ty = match closure_ty.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Ref(_, inner, _)) => inner,
        _ => closure_ty,
    };

    // Extract the closure DefId and substs
    let (closure_def, substs) = match inner_ty.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Closure(def, substs)) => {
            (def, substs.clone())
        }
        _ => return None,
    };

    // Get the closure body instance directly. Resolving the callable-trait
    // method can legitimately select a ClosureOnce adapter instead.
    //
    // The closure_def.def_id() gives us the DefId of the closure body.
    // We construct the mangled name by creating an FnDef and resolving it.
    use rustc_public::mir::mono::Instance;
    use rustc_public::ty::FnDef;

    // Create an FnDef from the closure's DefId
    let fn_def = FnDef(closure_def.def_id());

    if let Ok(instance) = Instance::resolve(fn_def, &substs) {
        return Some(instance.mangled_name());
    }

    // Fallback: try the old resolve_closure method
    Instance::resolve_closure(closure_def, &substs, rustc_public::ty::ClosureKind::FnOnce)
        .ok()
        .map(|instance| instance.mangled_name())
}

/// Read the const-generic `FACTOR` from a `__unroll_config::<FACTOR>()` callee
/// (`0` = full unroll).
///
/// Returns `None` when the callee is malformed instead of silently turning the
/// request into a full unroll.
fn extract_unroll_factor(func: &mir::Operand) -> Option<u32> {
    use rustc_public::ty::{RigidTy, TyConstKind, TyKind};
    let mir::Operand::Constant(constant) = func else {
        return None;
    };
    let TyKind::RigidTy(RigidTy::FnDef(definition, args)) = constant.const_.ty().kind() else {
        return None;
    };
    let definition_name = definition.name();
    if definition.krate().name.as_str() != "cuda_device"
        || (definition_name != "__unroll_config" && !definition_name.ends_with("::__unroll_config"))
        || args.0.len() != 1
    {
        return None;
    }
    if let Some(arg) = args.0.first()
        && let rustc_public::ty::GenericArgKind::Const(c) = arg
    {
        return match c.kind() {
            TyConstKind::Value(_, alloc) => {
                alloc.read_uint().ok().and_then(|v| u32::try_from(v).ok())
            }
            _ => c
                .eval_target_usize()
                .ok()
                .and_then(|v| u32::try_from(v).ok()),
        };
    }
    None
}

/// Match a compiler marker by its resolved definition, not by a debug string.
///
/// A user function with the same spelling must remain an ordinary call. The
/// `FnDef` identity also guarantees that generic arguments were type-checked by
/// rustc before the importer reads them.
fn is_cuda_device_const_marker(func: &mir::Operand, expected_name: &str) -> bool {
    use rustc_public::CrateDef;
    use rustc_public::ty::{RigidTy, TyKind};

    let mir::Operand::Constant(constant) = func else {
        return false;
    };
    let TyKind::RigidTy(RigidTy::FnDef(definition, _)) = constant.const_.ty().kind() else {
        return false;
    };
    if definition.krate().name.as_str() != "cuda_device" {
        return false;
    }
    let definition_name = definition.name();
    definition_name == expected_name || definition_name.ends_with(&format!("::{expected_name}"))
}

/// Extracts function metadata from a MIR function operand.
///
/// Returns a tuple of:
/// - `pattern_name`: The function's simple name (e.g., `"cuda_device::index_1d"`)
/// - `call_name`: The name used for the call target in generated code
/// - `substs_str`: Debug string of generic substitutions (for pattern matching)
///
/// Deliberately NOT returned: the callee's declared return type. The
/// declared `fn_sig` of a trait method is written against the trait, so its
/// output can be an unresolved associated-type projection such as
/// `<&Foo as Mul>::Output` (issue #133). Call results are instead typed from
/// the caller's destination place, which rustc has already monomorphized and
/// normalized. If a callee-signature type is ever genuinely needed here,
/// resolve the instance first (`Instance::resolve`) and query the signature
/// on the resolved instance so associated types arrive normalized.
///
/// This information is used to:
/// 1. Match intrinsic patterns by `pattern_name` (full FQDN, e.g. `cuda_device::thread::threadIdx_x`)
/// 2. Check for closure types via `substs_str.contains("Closure")`
/// 3. Generate the correct call target name (FQDN for non-generic, mangled for generic)
///
/// # Naming strategy
///
/// `CrateDef::name()` returns the fully qualified name (FQDN) in the
/// `rustc_public` API (e.g. `helper_fn::cuda_oxide_device_<hash>_vecadd_device`).
/// We use the raw `FnDef` name as `pattern_name` for intrinsic matching, then
/// use the resolved `Instance::name()` as `call_name` for monomorphic calls.
/// Raw `FnDef` substitutions are not authoritative for this decision: concrete
/// trait impl calls can carry a trait self type in the operand while resolving
/// to a monomorphic instance, and the resolved impl FQDN can differ from the
/// trait item FQDN.
///
/// For resolved instances that still carry generic args, `Instance::mangled_name`
/// is used instead, which the collector also matches via `compute_export_name`.
/// Non-generic FQDNs with characters such as `<`, `>`, and `::` are passed raw to
/// the same pliron `Legaliser` used for definition symbols.
///
/// Foreign items (`extern "C"` block declarations) are the exception: they
/// have no MIR body, so the collector never exports a definition under the
/// FQDN and the device linker (libdevice, external LTOIR) only knows the
/// link symbol. `call_name` for those is `Instance::mangled_name`, which is
/// the link symbol (it honours `#[link_name]`).
fn extract_func_info(
    func: &mir::Operand,
    loc: &Location,
) -> TranslationResult<(Option<String>, Option<String>, Option<String>)> {
    Ok(match func {
        mir::Operand::Constant(const_op) => match const_op.const_.kind() {
            ConstantKind::ZeroSized => {
                let ty_kind = const_op.const_.ty().kind();
                match &ty_kind {
                    rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnDef(
                        fn_def,
                        substs,
                    )) => {
                        use rustc_public::mir::mono::Instance;

                        let pattern_name =
                            intrinsics::generated::require_supported_raw_intrinsic(*fn_def, loc)?
                                .unwrap_or_else(|| fn_def.name().as_str().to_string());

                        let resolved = Instance::resolve(*fn_def, substs).ok();
                        let call_name = if let Some(instance) = resolved {
                            if instance.is_foreign_item() {
                                // Foreign items (`extern "C"` blocks) have no MIR
                                // body, so no definition is ever exported under
                                // the FQDN. Emit the call under the link symbol
                                // (e.g. `__nv_asinf`), which is what libdevice or
                                // externally linked LTOIR actually provides.
                                instance.mangled_name()
                            } else if !instance.args().0.is_empty() {
                                instance.mangled_name()
                            } else {
                                instance.name().to_string()
                            }
                        } else {
                            pattern_name.clone()
                        };

                        let substs_debug = format!("{:?}", substs);
                        (Some(pattern_name), Some(call_name), Some(substs_debug))
                    }
                    _ => (None, None, None),
                }
            }
            _ => (None, None, None),
        },
        _ => (None, None, None),
    })
}

/// Lower `core::intrinsics::typed_swap_nonoverlapping::<T>(x, y)`, the
/// primitive behind `core::mem::swap`/`mem::replace`, as load/load/store/store.
/// The two pointers are guaranteed non-overlapping, so the temp-free crossover
/// `t0 = *x; t1 = *y; *x = t1; *y = t0` is valid (the loaded SSA values are
/// captured before either store runs). Returns a unit result + goto target.
#[allow(clippy::too_many_arguments)]
fn emit_typed_swap(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    use dialect_mir::ops::{MirConstructTupleOp, MirLoadOp, MirStoreOp};
    use dialect_mir::types::{MirPtrType, MirTupleType};

    if args.len() != 2 {
        return input_err!(
            loc,
            TranslationErr::unsupported(
                "typed_swap_nonoverlapping requires two pointer operands".to_string()
            )
        );
    }

    let (ptr_x, last) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let (ptr_y, last) =
        rvalue::translate_operand(ctx, body, &args[1], value_map, block_ptr, last, loc.clone())?;

    let elem_ty = {
        let t = ptr_x.get_type(ctx);
        let r = t.deref(ctx);
        match r.downcast_ref::<MirPtrType>() {
            Some(p) => p.pointee,
            None => {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(
                        "typed_swap_nonoverlapping operand is not a pointer".to_string()
                    )
                );
            }
        }
    };

    // t0 = *x
    let load_x = Operation::new(
        ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![elem_ty],
        vec![ptr_x],
        vec![],
        0,
    );
    load_x.deref_mut(ctx).set_loc(loc.clone());
    match last {
        Some(p) => load_x.insert_after(ctx, p),
        None => load_x.insert_at_front(block_ptr, ctx),
    }
    let vx = load_x.deref(ctx).get_result(0);

    // t1 = *y
    let load_y = Operation::new(
        ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![elem_ty],
        vec![ptr_y],
        vec![],
        0,
    );
    load_y.deref_mut(ctx).set_loc(loc.clone());
    load_y.insert_after(ctx, load_x);
    let vy = load_y.deref(ctx).get_result(0);

    // *x = t1
    let store_x = Operation::new(
        ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![ptr_x, vy],
        vec![],
        0,
    );
    store_x.deref_mut(ctx).set_loc(loc.clone());
    store_x.insert_after(ctx, load_y);

    // *y = t0
    let store_y = Operation::new(
        ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![ptr_y, vx],
        vec![],
        0,
    );
    store_y.deref_mut(ctx).set_loc(loc.clone());
    store_y.insert_after(ctx, store_x);

    // unit result
    let unit_ty = MirTupleType::get(ctx, vec![]);
    let unit_op = Operation::new(
        ctx,
        MirConstructTupleOp::get_concrete_op_info(),
        vec![unit_ty.into()],
        vec![],
        vec![],
        0,
    );
    unit_op.deref_mut(ctx).set_loc(loc.clone());
    unit_op.insert_after(ctx, store_y);
    let unit_val = unit_op.deref(ctx).get_result(0);

    let goto_prev = value_map
        .store_local(ctx, destination.local, unit_val, block_ptr, Some(unit_op))
        .unwrap_or(unit_op);

    if let Some(target_idx) = target {
        Ok(helpers::emit_goto(
            ctx,
            *target_idx,
            goto_prev,
            block_map,
            loc,
        ))
    } else {
        input_err!(
            loc,
            TranslationErr::unsupported(
                "typed_swap_nonoverlapping call without target not supported".to_string()
            )
        )
    }
}

/// Dispatches `cuda_device` intrinsic calls to their respective handlers.
///
/// Returns `Ok(Some(op))` if the call was an intrinsic, `Ok(None)` otherwise.
///
/// # Intrinsic Categories
///
/// | Category          | Examples                                          |
/// |-------------------|---------------------------------------------------|
/// | Thread Position   | `threadIdx_x`, `blockIdx_y`, `blockDim_x`         |
/// | Index Helpers     | `index_1d`, `index_2d::<S>`, `index_2d_runtime`, `index_2d_row`, `index_2d_col` |
/// | Synchronization   | `sync_threads`, `mbarrier_*`, `fence_*`           |
/// | Warp Primitives   | `shuffle_*`, `vote_*`, `lane_id`                  |
/// | WGMMA (Hopper `sm_90a`) | `wgmma_fence`, `wgmma_mma_*`, `make_smem_desc` |
/// | TMA               | `cp_async_bulk_tensor_*_g2s/s2g`, `wait_group`    |
/// | Tcgen05 (Blackwell)| `tcgen05_alloc`, `tcgen05_mma_*`, `tcgen05_ld_*` |
/// | Memory            | `SharedArray::index`, `stmatrix_*`, `cvt_*`       |
/// | DisjointSlice     | `get_thread_local`, `len`                         |
#[allow(clippy::too_many_arguments)]
fn try_dispatch_intrinsic(
    ctx: &mut Context,
    body: &mir::Body,
    func: &mir::Operand,
    name: &str,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    substs_contains: &impl Fn(&str) -> bool,
) -> TranslationResult<Option<Ptr<Operation>>> {
    intrinsics::wgmma::reject_unsupported(name, loc.clone())?;

    if let Some(operation) = intrinsics::generated::try_dispatch_generated_intrinsic(
        ctx,
        body,
        name,
        args,
        destination,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc.clone(),
    )? {
        return Ok(Some(operation));
    }

    if let Some(kind) = intrinsics::asm::InlinePtxCallKind::from_path(name) {
        return Ok(Some(intrinsics::asm::emit_inline_ptx(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            kind,
        )?));
    }

    if name == "core::intrinsics::typed_swap_nonoverlapping"
        || name == "std::intrinsics::typed_swap_nonoverlapping"
    {
        return Ok(Some(emit_typed_swap(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?));
    }

    if let Some(intrinsic) = intrinsics::bitops::RustBitIntrinsic::from_core_path(name) {
        return Ok(Some(intrinsics::bitops::emit_rust_bit_intrinsic(
            ctx,
            body,
            intrinsic,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?));
    }

    if let Some(intrinsic) = intrinsics::saturating::RustSaturatingIntrinsic::from_core_path(name) {
        return Ok(Some(
            intrinsics::saturating::emit_rust_saturating_intrinsic(
                ctx,
                body,
                intrinsic,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?,
        ));
    }

    if let Some(intrinsic) = intrinsics::bigint::RustBigIntIntrinsic::from_core_path(name) {
        return Ok(Some(intrinsics::bigint::emit_rust_bigint_intrinsic(
            ctx,
            body,
            intrinsic,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?));
    }

    if let Some(is_f64) = intrinsics::float_math::libm_sincos_is_f64(name) {
        return Ok(Some(intrinsics::float_math::emit_sincos(
            ctx,
            body,
            is_f64,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?));
    }

    if let Some(intrinsic) = intrinsics::float_math::RustFloatMathIntrinsic::from_core_path(name) {
        return Ok(Some(
            intrinsics::float_math::emit_rust_float_math_intrinsic(
                ctx,
                body,
                intrinsic,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?,
        ));
    }

    match name {
        // =================================================================
        // Compiler Hints
        // These intrinsics only guide optimization and do not affect semantics.
        // =================================================================
        "core::intrinsics::cold_path" | "std::intrinsics::cold_path" => {
            Ok(Some(helpers::emit_unit_noop_intrinsic(
                ctx,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
                name,
            )?))
        }
        "core::intrinsics::volatile_load" | "std::intrinsics::volatile_load" => {
            Ok(Some(intrinsics::memory::emit_volatile_load(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "core::intrinsics::volatile_store" | "std::intrinsics::volatile_store" => {
            Ok(Some(intrinsics::memory::emit_volatile_store(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }

        "core::intrinsics::ptr_offset_from" | "std::intrinsics::ptr_offset_from" => {
            Ok(Some(intrinsics::memory::emit_ptr_offset_from(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }

        "core::intrinsics::ptr_offset_from_unsigned"
        | "std::intrinsics::ptr_offset_from_unsigned" => {
            Ok(Some(intrinsics::memory::emit_ptr_offset_from_unsigned(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }

        // =================================================================
        // Thread Index Helpers (from intrinsics::indexing)
        //
        // The index helpers are normal Rust functions over generated leaf
        // intrinsics. Their bodies use the normal function-call path.
        // =================================================================
        "cuda_device::thread::__internal::index_1d"
        | "cuda_device::index_1d"
        | "cuda_device::thread::index_1d"
        | "cuda_device::index_2d_row"
        | "cuda_device::thread::index_2d_row"
        | "cuda_device::index_2d_col"
        | "cuda_device::thread::index_2d_col"
        | "cuda_device::thread::__internal::index_2d"
        | "cuda_device::thread::__internal::index_2d_runtime"
        | "cuda_device::index_2d"
        | "cuda_device::thread::index_2d"
        | "cuda_device::index_2d_runtime"
        | "cuda_device::thread::index_2d_runtime" => Ok(None),

        // =================================================================
        // Debug & Profiling (from intrinsics::debug)
        // =================================================================
        "cuda_device::debug::__gpu_vprintf" => Ok(Some(intrinsics::debug::emit_vprintf(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),

        // =================================================================
        // Compile-time cluster configuration
        // =================================================================
        "cuda_device::cluster::__cluster_config" => {
            // Compile-time cluster configuration marker from #[cluster(x,y,z)] attribute.
            // The cluster dimensions are extracted in body.rs during MIR scanning.
            // This call generates no runtime code - just emit a goto to the target block.
            //
            // We need a prev_op to insert after. If none exists, create a dummy constant.
            let actual_prev_op = match prev_op {
                Some(op) => op,
                None => {
                    // Create a dummy i1 constant to use as insertion point
                    let bool_ty = IntegerType::get(ctx, 1, Signedness::Signless);
                    let dummy = Operation::new(
                        ctx,
                        MirConstantOp::get_concrete_op_info(),
                        vec![bool_ty.into()],
                        vec![],
                        vec![],
                        0,
                    );
                    dummy.deref_mut(ctx).set_loc(loc.clone());
                    let const_op = MirConstantOp::new(dummy);
                    use pliron::builtin::attributes::IntegerAttr;
                    use pliron::utils::apint::APInt;
                    use std::num::NonZeroUsize;
                    let false_val = APInt::from_u64(0, NonZeroUsize::new(1).unwrap());
                    const_op.set_attr_value(ctx, IntegerAttr::new(bool_ty, false_val));
                    let dummy = const_op.get_operation();
                    dummy.insert_at_front(block_ptr, ctx);
                    dummy
                }
            };
            Ok(Some(helpers::emit_goto(
                ctx,
                target.expect("__cluster_config must have target"),
                actual_prev_op,
                block_map,
                loc,
            )))
        }
        "cuda_device::__launch_bounds_config"
        | "cuda_device::thread::__launch_bounds_config"
        | "cuda_device::__launch_contract_config"
        | "cuda_device::thread::__launch_contract_config" => {
            let expected_marker = match name {
                "cuda_device::__launch_bounds_config"
                | "cuda_device::thread::__launch_bounds_config" => "__launch_bounds_config",
                "cuda_device::__launch_contract_config"
                | "cuda_device::thread::__launch_contract_config" => "__launch_contract_config",
                _ => unreachable!("launch metadata arm matched an unknown marker"),
            };
            if !is_cuda_device_const_marker(func, expected_marker) {
                return Ok(None);
            }
            // Compile-time launch metadata marker. Launch bounds are extracted
            // in body.rs; the contract marker selects the kernel's typed launch
            // context during macro expansion. Neither marker emits runtime code.
            // Emit only the control-flow edge to the call's target.
            //
            // We need a prev_op to insert after. If none exists, create a dummy constant.
            let actual_prev_op = match prev_op {
                Some(op) => op,
                None => {
                    // Create a dummy i1 constant to use as insertion point
                    let bool_ty = IntegerType::get(ctx, 1, Signedness::Signless);
                    let dummy = Operation::new(
                        ctx,
                        MirConstantOp::get_concrete_op_info(),
                        vec![bool_ty.into()],
                        vec![],
                        vec![],
                        0,
                    );
                    dummy.deref_mut(ctx).set_loc(loc.clone());
                    let const_op = MirConstantOp::new(dummy);
                    use pliron::builtin::attributes::IntegerAttr;
                    use pliron::utils::apint::APInt;
                    use std::num::NonZeroUsize;
                    let false_val = APInt::from_u64(0, NonZeroUsize::new(1).unwrap());
                    const_op.set_attr_value(ctx, IntegerAttr::new(bool_ty, false_val));
                    let dummy = const_op.get_operation();
                    dummy.insert_at_front(block_ptr, ctx);
                    dummy
                }
            };
            Ok(Some(helpers::emit_goto(
                ctx,
                target.expect("launch metadata marker must have target"),
                actual_prev_op,
                block_map,
                loc,
            )))
        }
        "cuda_device::shared::__dynamic_shared_alignment" => {
            // Zero-cost marker injected by #[launch_contract]. body.rs records
            // the const alignment on the kernel; no runtime call survives.
            let actual_prev_op = match prev_op {
                Some(op) => op,
                None => {
                    let bool_ty = IntegerType::get(ctx, 1, Signedness::Signless);
                    let dummy = Operation::new(
                        ctx,
                        MirConstantOp::get_concrete_op_info(),
                        vec![bool_ty.into()],
                        vec![],
                        vec![],
                        0,
                    );
                    dummy.deref_mut(ctx).set_loc(loc.clone());
                    let const_op = MirConstantOp::new(dummy);
                    use pliron::builtin::attributes::IntegerAttr;
                    use pliron::utils::apint::APInt;
                    use std::num::NonZeroUsize;
                    let false_val = APInt::from_u64(0, NonZeroUsize::new(1).unwrap());
                    const_op.set_attr_value(ctx, IntegerAttr::new(bool_ty, false_val));
                    let dummy = const_op.get_operation();
                    dummy.insert_at_front(block_ptr, ctx);
                    dummy
                }
            };
            Ok(Some(helpers::emit_goto(
                ctx,
                target.expect("__dynamic_shared_alignment must have target"),
                actual_prev_op,
                block_map,
                loc,
            )))
        }
        // =================================================================
        // WGMMA (from intrinsics::wgmma)
        // =================================================================
        "cuda_device::wgmma::make_smem_desc" => {
            Ok(Some(intrinsics::wgmma::emit_wgmma_make_smem_desc(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::wgmma::wgmma_mma_m64n64k16_f32_bf16" => {
            Ok(Some(intrinsics::wgmma::emit_wgmma_mma_m64n64k16_f32_bf16(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }

        // =================================================================
        // DisjointSlice and SharedArray operations
        // =================================================================
        "cuda_device::DisjointSlice::get_thread_local" => {
            Ok(Some(intrinsics::indexing::emit_get_thread_local(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::DisjointSlice::len" => Ok(Some(intrinsics::indexing::emit_len(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),

        // Trait method - check substs for SharedArray
        // Note: Index/IndexMut can appear as either std::ops or core::ops
        "std::ops::IndexMut::index_mut" | "core::ops::IndexMut::index_mut"
            if substs_contains("SharedArray") =>
        {
            Ok(Some(intrinsics::memory::emit_shared_array_index(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
                true,
            )?))
        }
        "std::ops::Index::index" | "core::ops::Index::index" if substs_contains("SharedArray") => {
            Ok(Some(intrinsics::memory::emit_shared_array_index(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
                false,
            )?))
        }

        // DisjointSlice methods using prefix matching (covers generic
        // instantiations whose full path includes type parameters).
        // Note: `get_mut` and `get_unchecked_mut` are `#[inline]` in
        // cuda-device and are always inlined by rustc before MIR reaches the
        // translator. Routing them here would produce a type mismatch
        // (`emit_get_thread_local` returns `*mut T` but `get_mut` returns
        // `Option<&mut T>`). They are intentionally absent from this match.
        path if path.starts_with("cuda_device::DisjointSlice::") => {
            if let Some(method) = path.rsplit("::").next() {
                match method {
                    "get_thread_local" => Ok(Some(intrinsics::indexing::emit_get_thread_local(
                        ctx,
                        body,
                        args,
                        destination,
                        target,
                        block_ptr,
                        prev_op,
                        value_map,
                        block_map,
                        loc,
                    )?)),
                    "len" => Ok(Some(intrinsics::indexing::emit_len(
                        ctx,
                        body,
                        args,
                        destination,
                        target,
                        block_ptr,
                        prev_op,
                        value_map,
                        block_map,
                        loc,
                    )?)),
                    _ => Ok(None),
                }
            } else {
                Ok(None)
            }
        }

        // SharedArray::as_ptr and as_mut_ptr - convert shared memory pointer to generic
        path if path.contains("SharedArray") && path.contains("as_ptr") => {
            Ok(Some(intrinsics::memory::emit_shared_array_as_ptr(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        path if path.contains("SharedArray") && path.contains("as_mut_ptr") => {
            Ok(Some(intrinsics::memory::emit_shared_array_as_ptr(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }

        // Note: DynamicSharedArray operations are handled specially before this function
        // to extract the ALIGN const generic parameter. See the handling above
        // try_dispatch_intrinsic() call.

        // =================================================================
        // Atomic Operations (all cuda_device::atomic::* types and scopes)
        // =================================================================
        path if intrinsics::atomic::is_atomic_path(path) => intrinsics::atomic::dispatch(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            path,
        ),

        // Not an intrinsic
        _ => Ok(None),
    }
}
