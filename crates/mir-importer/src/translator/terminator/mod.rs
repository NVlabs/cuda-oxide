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
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_mir::ops::{
    MirAssertOp, MirCondBranchOp, MirConstantOp, MirEqOp, MirGotoOp, MirNotOp, MirReturnOp,
};
use dialect_nvvm::ops::{
    ReadPtxSregCtaidXOp, ReadPtxSregCtaidYOp, ReadPtxSregNtidXOp, ReadPtxSregNtidYOp,
    ReadPtxSregTidXOp, ReadPtxSregTidYOp,
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
    // Use Debug representation of the span as location
    let loc = Location::Named {
        name: format!("{:?}", term.span),
        child_loc: Box::new(Location::Unknown),
    };

    match &term.kind {
        mir::TerminatorKind::Return => translate_return(ctx, value_map, block_ptr, prev_op, loc),

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
    let loaded = value_map.load_local(ctx, return_local, block_ptr, prev_op);

    let (operands, terminator_prev_op) = match loaded {
        Some((load_op, val)) => {
            use dialect_mir::types::MirTupleType;
            let val_type = val.get_type(ctx);
            let val_type_obj = val_type.deref(ctx);
            if let Some(tuple_ty) = val_type_obj.downcast_ref::<MirTupleType>() {
                if tuple_ty.get_types().is_empty() {
                    // Unit return: the load we just emitted is dead, but
                    // harmless; leave it as prev_op so the return chains
                    // after it.
                    (vec![], Some(load_op))
                } else {
                    (vec![val], Some(load_op))
                }
            } else {
                (vec![val], Some(load_op))
            }
        }
        None => (vec![], prev_op),
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
            vec![bool_type.to_ptr()],
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

        let (condition, last_inserted_op) = if discr_ty == bool_ty.to_ptr() {
            // discr is already i1
            // For boolean switch: val=0 means "if false", val=1 means "if true"
            // switchInt(bool) -> [0: bb_false, otherwise: bb_true]
            // Since val == 0 means "go to target when discr == 0", we need condition = !discr
            if val == 0 {
                // Create NOT operation: condition = !discr
                let not_op = Operation::new(
                    ctx,
                    MirNotOp::get_concrete_op_info(),
                    vec![bool_ty.to_ptr()],
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
                vec![bool_ty.to_ptr()],
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
            vec![bool_ty.to_ptr()],
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
    let (pattern_name, call_name, substs_str) = extract_func_info(func);

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

    // Handle closure trait method calls (FnOnce::call_once, FnMut::call_mut, Fn::call)
    // These calls pass arguments as a tuple, but the closure body expects unpacked args.
    // We need to unpack the tuple before calling the closure.
    //
    // MIR shows: <{closure} as FnMut<(u32,)>>::call_mut(self_ref, tuple_args)
    // But the closure body expects: fn(self_ref, unpacked_arg1, unpacked_arg2, ...)
    if let Some(ref name) = pattern_name
        && (name.contains("call_once") || name.contains("call_mut") || name.ends_with("::call"))
        && substs_contains("Closure")
    {
        return translate_closure_call(
            ctx,
            body,
            &call_name,
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

    // Handle prof_trigger specially to extract const generic N
    if let Some(ref name) = pattern_name
        && name == "cuda_device::debug::prof_trigger"
    {
        // Extract the const generic N from the function type
        if let mir::Operand::Constant(const_op) = func
            && let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnDef(_, substs)) =
                const_op.const_.ty().kind()
        {
            // The const generic N is the first generic argument
            if let Some(rustc_public::ty::GenericArgKind::Const(c)) = substs.0.first() {
                use rustc_public::ty::TyConstKind;

                let event_id = match c.kind() {
                    TyConstKind::Value(_, alloc) => {
                        // Read the allocation bytes (little-endian u32)
                        alloc.read_uint().unwrap_or(0) as u32
                    }
                    _ => {
                        // Try eval_target_usize as fallback
                        c.eval_target_usize().unwrap_or(0) as u32
                    }
                };
                return intrinsics::debug::emit_prof_trigger(
                    ctx,
                    event_id,
                    &target_usize,
                    block_ptr,
                    prev_op,
                    block_map,
                    loc,
                );
            }
        }
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
/// In unified compilation with `std`, `Instance::resolve` for `<Closure as FnOnce>::call_once`
/// returns a trait method **shim**, not the closure body directly. We must extract the closure's
/// DefId from `args[0]`'s type and resolve that to get the actual closure body's mangled name.
///
/// See `device_closures/README.md` for detailed documentation.
#[allow(clippy::too_many_arguments)]
fn translate_closure_call(
    ctx: &mut Context,
    body: &mir::Body,
    call_name: &Option<String>,
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

    // Extract the closure body's name from the closure type in args[0].
    // This is critical for unified compilation where Instance::resolve returns
    // a trait shim instead of the closure body directly.
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

    // Determine if this is call_once (by value) vs call_mut/call (by reference).
    //
    // In `std` mode, rustc generates `FnOnce::call_once(self, args)` which passes
    // the closure BY VALUE. But the closure body expects `&self` (a reference).
    //
    // In `no_std` mode, rustc generates `FnMut::call_mut(&mut self, args)` which
    // already passes a reference.
    //
    // When we have call_once, we need to create a reference to the closure value
    // before calling the closure body.
    //
    // We check the ORIGINAL call_name (the trait method), not the resolved callee
    // (which is the closure body name).
    let is_call_once = call_name
        .as_ref()
        .map(|n| n.contains("call_once"))
        .unwrap_or(false);

    let self_arg = if is_call_once {
        // For call_once: self is passed by value, but closure body expects reference.
        // Create a MirRefOp to take a reference to the closure value.
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
        ref_op.deref_mut(ctx).attributes.0.insert(
            Identifier::try_from("mutable").unwrap(),
            mutable_attr.into(),
        );

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

    // Build unpacked arguments: start with self (or ref to self for call_once)
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
        .0
        .insert(Identifier::try_from("callee").unwrap(), callee_attr.into());

    // Insert the call
    let call_op = if let Some(prev) = last_op {
        call_op.insert_after(ctx, prev);
        call_op
    } else {
        call_op.insert_at_front(block_ptr, ctx);
        call_op
    };

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

/// Extracts the closure body's mangled name from a closure operand.
///
/// In unified compilation with `std`, when we see a call like:
///   `<{closure} as FnOnce<(u32,)>>::call_once(closure_ref, args_tuple)`
///
/// The `call_name` from `Instance::resolve` gives us the trait method shim's name,
/// not the closure body. We need to extract the closure's DefId from `args[0]`'s type
/// and resolve that to get the actual closure body's mangled name.
///
/// The closure argument can be:
/// - A direct closure value (type is `Closure(def, substs)`)
/// - A reference to a closure (type is `Ref(_, Closure(def, substs), _)`)
/// - A mutable reference (same pattern)
fn extract_closure_body_name(closure_arg: &mir::Operand, body: &mir::Body) -> Option<String> {
    // Get the type of the closure argument
    let closure_ty = match closure_arg {
        mir::Operand::Copy(place) | mir::Operand::Move(place) => {
            // Get the type from the place's local
            let local: usize = place.local;
            let local_decls: Vec<_> = body.local_decls().collect();
            local_decls.get(local).map(|(_, decl)| decl.ty)
        }
        mir::Operand::Constant(const_op) => Some(const_op.const_.ty()),
        _ => None,
    }?;

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

    // Get the closure body instance directly, NOT through resolve_closure
    // (which returns a call_once shim in unified/std compilation).
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
/// `CrateDef::name()` returns the fully qualified name (FQDN) in the `rustc_public`
/// API (e.g. `helper_fn::cuda_oxide_device_<hash>_vecadd_device`). We use this directly as
/// both `pattern_name` and `call_name` (for non-generic calls). The collector
/// produces matching FQDNs, and the lowering layer (`mir-lower`) converts `::` to
/// `__` on both sides to produce valid LLVM/PTX identifiers.
///
/// For generic calls, `Instance::resolve` + `mangled_name` is used instead, which
/// the collector also matches via `compute_export_name`.
///
fn extract_func_info(func: &mir::Operand) -> (Option<String>, Option<String>, Option<String>) {
    match func {
        mir::Operand::Constant(const_op) => match const_op.const_.kind() {
            ConstantKind::ZeroSized => {
                let ty_kind = const_op.const_.ty().kind();
                match &ty_kind {
                    rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnDef(
                        fn_def,
                        substs,
                    )) => {
                        let pattern_name = fn_def.name().as_str().to_string();

                        let has_generic_args = !substs.0.is_empty();
                        let call_name = if has_generic_args {
                            use rustc_public::mir::mono::Instance;
                            if let Ok(instance) = Instance::resolve(*fn_def, substs) {
                                instance.mangled_name()
                            } else {
                                pattern_name.clone()
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
/// | WGMMA (Hopper)    | `wgmma_fence`, `wgmma_mma_*`, `make_smem_desc`    |
/// | TMA               | `cp_async_bulk_tensor_*_g2s/s2g`, `wait_group`    |
/// | Tcgen05 (Blackwell)| `tcgen05_alloc`, `tcgen05_mma_*`, `tcgen05_ld_*` |
/// | Memory            | `SharedArray::index`, `stmatrix_*`, `cvt_*`       |
/// | DisjointSlice     | `get_thread_local`, `len`                         |
#[allow(clippy::too_many_arguments)]
fn try_dispatch_intrinsic(
    ctx: &mut Context,
    body: &mir::Body,
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

        // =================================================================
        // Thread/Block Position Intrinsics
        // Support both re-exported (cuda_device::) and full paths (cuda_device::thread::)
        // =================================================================
        "cuda_device::threadIdx_x" | "cuda_device::thread::threadIdx_x" => {
            Ok(Some(helpers::emit_nvvm_intrinsic(
                ctx,
                ReadPtxSregTidXOp::get_concrete_op_info(),
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::threadIdx_y" | "cuda_device::thread::threadIdx_y" => {
            Ok(Some(helpers::emit_nvvm_intrinsic(
                ctx,
                ReadPtxSregTidYOp::get_concrete_op_info(),
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::blockIdx_x" | "cuda_device::thread::blockIdx_x" => {
            Ok(Some(helpers::emit_nvvm_intrinsic(
                ctx,
                ReadPtxSregCtaidXOp::get_concrete_op_info(),
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::blockIdx_y" | "cuda_device::thread::blockIdx_y" => {
            Ok(Some(helpers::emit_nvvm_intrinsic(
                ctx,
                ReadPtxSregCtaidYOp::get_concrete_op_info(),
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::blockDim_x" | "cuda_device::thread::blockDim_x" => {
            Ok(Some(helpers::emit_nvvm_intrinsic(
                ctx,
                ReadPtxSregNtidXOp::get_concrete_op_info(),
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::blockDim_y" | "cuda_device::thread::blockDim_y" => {
            Ok(Some(helpers::emit_nvvm_intrinsic(
                ctx,
                ReadPtxSregNtidYOp::get_concrete_op_info(),
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::threadIdx_z" | "cuda_device::thread::threadIdx_z" => {
            Ok(Some(helpers::emit_nvvm_intrinsic(
                ctx,
                dialect_nvvm::ops::ReadPtxSregTidZOp::get_concrete_op_info(),
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::blockIdx_z" | "cuda_device::thread::blockIdx_z" => {
            Ok(Some(helpers::emit_nvvm_intrinsic(
                ctx,
                dialect_nvvm::ops::ReadPtxSregCtaidZOp::get_concrete_op_info(),
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::blockDim_z" | "cuda_device::thread::blockDim_z" => {
            Ok(Some(helpers::emit_nvvm_intrinsic(
                ctx,
                dialect_nvvm::ops::ReadPtxSregNtidZOp::get_concrete_op_info(),
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::gridDim_x" | "cuda_device::thread::gridDim_x" => {
            Ok(Some(helpers::emit_nvvm_intrinsic(
                ctx,
                dialect_nvvm::ops::ReadPtxSregNctaidXOp::get_concrete_op_info(),
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::gridDim_y" | "cuda_device::thread::gridDim_y" => {
            Ok(Some(helpers::emit_nvvm_intrinsic(
                ctx,
                dialect_nvvm::ops::ReadPtxSregNctaidYOp::get_concrete_op_info(),
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::gridDim_z" | "cuda_device::thread::gridDim_z" => {
            Ok(Some(helpers::emit_nvvm_intrinsic(
                ctx,
                dialect_nvvm::ops::ReadPtxSregNctaidZOp::get_concrete_op_info(),
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::grid::envreg1" => Ok(Some(helpers::emit_nvvm_intrinsic(
            ctx,
            dialect_nvvm::ops::ReadPtxSregEnvReg1Op::get_concrete_op_info(),
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),
        "cuda_device::grid::envreg2" => Ok(Some(helpers::emit_nvvm_intrinsic(
            ctx,
            dialect_nvvm::ops::ReadPtxSregEnvReg2Op::get_concrete_op_info(),
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),

        // =================================================================
        // Thread Index Helpers (from intrinsics::indexing)
        // Support both re-exported (cuda_device::) and full paths (cuda_device::thread::)
        //
        // TODO: These three handlers (index_1d, index_2d_row, index_2d_col) are not
        // strictly necessary. Their Rust bodies are real code that calls true intrinsics
        // (threadIdx_x, blockIdx_y, etc.), so they compile correctly through the normal
        // function path. They exist as a reliability workaround because #[inline(always)]
        // is not always honored by rustc — without these handlers, a non-inlined call
        // would require compiling the function body separately. The arithmetic expansion
        // is trivial (3 ops + cast), so we keep them for now. If rustc inlining becomes
        // reliable, these can be removed along with emit_index_1d_expansion,
        // emit_index_2d_row, and emit_index_2d_col in intrinsics/indexing.rs.
        // =================================================================
        "cuda_device::thread::__internal::index_1d"
        | "cuda_device::index_1d"
        | "cuda_device::thread::index_1d" => {
            Ok(Some(intrinsics::indexing::emit_index_1d_expansion(
                ctx,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::index_2d_row" | "cuda_device::thread::index_2d_row" => {
            Ok(Some(intrinsics::indexing::emit_index_2d_row(
                ctx,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::index_2d_col" | "cuda_device::thread::index_2d_col" => {
            Ok(Some(intrinsics::indexing::emit_index_2d_col(
                ctx,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        // index_2d returns Option<ThreadIndex> with an internal col < stride check.
        // It is compiled as a normal function (not expanded inline) so that the
        // Option construction and branch are handled by the standard MIR translator.
        "cuda_device::thread::__internal::index_2d"
        | "cuda_device::thread::__internal::index_2d_runtime"
        | "cuda_device::index_2d"
        | "cuda_device::thread::index_2d"
        | "cuda_device::index_2d_runtime"
        | "cuda_device::thread::index_2d_runtime" => Ok(None),

        // =================================================================
        // Synchronization (from intrinsics::sync)
        // =================================================================
        "cuda_device::sync_threads" => Ok(Some(intrinsics::sync::emit_sync_threads(
            ctx, target, block_ptr, prev_op, block_map, loc,
        )?)),
        "cuda_device::threadfence_block" | "cuda_device::fence::threadfence_block" => {
            Ok(Some(intrinsics::sync::emit_threadfence_block(
                ctx, target, block_ptr, prev_op, block_map, loc,
            )?))
        }
        "cuda_device::threadfence" | "cuda_device::fence::threadfence" => Ok(Some(
            intrinsics::sync::emit_threadfence(ctx, target, block_ptr, prev_op, block_map, loc)?,
        )),
        "cuda_device::threadfence_system" | "cuda_device::fence::threadfence_system" => {
            Ok(Some(intrinsics::sync::emit_threadfence_system(
                ctx, target, block_ptr, prev_op, block_map, loc,
            )?))
        }
        "cuda_device::barrier::mbarrier_init" => Ok(Some(intrinsics::sync::emit_mbarrier_init(
            ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
        )?)),
        "cuda_device::barrier::mbarrier_arrive" => {
            Ok(Some(intrinsics::sync::emit_mbarrier_arrive(
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
        "cuda_device::barrier::mbarrier_arrive_expect_tx" => {
            Ok(Some(intrinsics::sync::emit_mbarrier_arrive_expect_tx(
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
        "cuda_device::barrier::mbarrier_arrive_cluster" => {
            Ok(Some(intrinsics::sync::emit_mbarrier_arrive_cluster(
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
        "cuda_device::barrier::nanosleep" => Ok(Some(intrinsics::sync::emit_nanosleep(
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
        "cuda_device::barrier::mbarrier_test_wait" => {
            Ok(Some(intrinsics::sync::emit_mbarrier_test_wait(
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
        "cuda_device::barrier::mbarrier_try_wait" => {
            Ok(Some(intrinsics::sync::emit_mbarrier_try_wait(
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
        "cuda_device::barrier::mbarrier_try_wait_parity" => {
            Ok(Some(intrinsics::sync::emit_mbarrier_try_wait_parity(
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
        "cuda_device::barrier::mbarrier_inval" => Ok(Some(intrinsics::sync::emit_mbarrier_inval(
            ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
        )?)),
        "cuda_device::barrier::fence_proxy_async_shared_cta" => {
            Ok(Some(intrinsics::sync::emit_fence_proxy_async_shared_cta(
                ctx, args, target, block_ptr, prev_op, block_map, loc,
            )?))
        }

        // =================================================================
        // Debug & Profiling (from intrinsics::debug)
        // =================================================================
        "cuda_device::debug::clock" => Ok(Some(intrinsics::debug::emit_clock(
            ctx,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),
        "cuda_device::debug::clock64" => Ok(Some(intrinsics::debug::emit_clock64(
            ctx,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),
        "cuda_device::debug::globaltimer" => Ok(Some(intrinsics::debug::emit_globaltimer(
            ctx,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),
        "cuda_device::debug::trap" => Ok(Some(intrinsics::debug::emit_trap(
            ctx, target, block_ptr, prev_op, block_map, loc,
        )?)),
        "cuda_device::debug::breakpoint" => Ok(Some(intrinsics::debug::emit_breakpoint(
            ctx, target, block_ptr, prev_op, block_map, loc,
        )?)),
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
        // Thread Block Clusters (from intrinsics::cluster) - sm_90+
        // =================================================================
        "cuda_device::cluster::cluster_ctaidX" => {
            Ok(Some(intrinsics::cluster::emit_cluster_ctaid_x(
                ctx,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::cluster::cluster_ctaidY" => {
            Ok(Some(intrinsics::cluster::emit_cluster_ctaid_y(
                ctx,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::cluster::cluster_ctaidZ" => {
            Ok(Some(intrinsics::cluster::emit_cluster_ctaid_z(
                ctx,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::cluster::cluster_nctaidX" => {
            Ok(Some(intrinsics::cluster::emit_cluster_nctaid_x(
                ctx,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::cluster::cluster_nctaidY" => {
            Ok(Some(intrinsics::cluster::emit_cluster_nctaid_y(
                ctx,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::cluster::cluster_nctaidZ" => {
            Ok(Some(intrinsics::cluster::emit_cluster_nctaid_z(
                ctx,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
            )?))
        }
        "cuda_device::cluster::cluster_idx" => Ok(Some(intrinsics::cluster::emit_cluster_idx(
            ctx,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),
        "cuda_device::cluster::num_clusters" => Ok(Some(intrinsics::cluster::emit_num_clusters(
            ctx,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),
        "cuda_device::cluster::cluster_sync" => Ok(Some(intrinsics::cluster::emit_cluster_sync(
            ctx, target, block_ptr, prev_op, block_map, loc,
        )?)),
        "cuda_device::cluster::map_shared_rank" => {
            Ok(Some(intrinsics::cluster::emit_map_shared_rank(
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
        "cuda_device::cluster::map_shared_rank_mut" => {
            // Same implementation as map_shared_rank, just different pointer mutability
            Ok(Some(intrinsics::cluster::emit_map_shared_rank(
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
        "cuda_device::cluster::dsmem_read_u32" => {
            Ok(Some(intrinsics::cluster::emit_dsmem_read_u32(
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
        "cuda_device::thread::__launch_bounds_config" => {
            // Compile-time launch bounds marker from #[launch_bounds(max, min)] attribute.
            // The launch bounds are extracted in body.rs during MIR scanning.
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
                target.expect("__launch_bounds_config must have target"),
                actual_prev_op,
                block_map,
                loc,
            )))
        }

        // =================================================================
        // Warp Primitives (from intrinsics::warp)
        // =================================================================
        "cuda_device::warp::lane_id" => Ok(Some(intrinsics::warp::emit_lane_id(
            ctx,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),
        "cuda_device::warp::active_mask" => Ok(Some(intrinsics::warp::emit_active_mask(
            ctx,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),
        "cuda_device::warp::sync_mask" => Ok(Some(intrinsics::warp::emit_warp_sync_mask(
            ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
        )?)),
        "cuda_device::warp::shuffle_sync" => Ok(Some(intrinsics::warp::emit_warp_shuffle_i32(
            ctx,
            body,
            dialect_nvvm::ops::ShflSyncIdxI32Op::get_concrete_op_info(),
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),
        "cuda_device::warp::shuffle_up_sync" => Ok(Some(intrinsics::warp::emit_warp_shuffle_i32(
            ctx,
            body,
            dialect_nvvm::ops::ShflSyncUpI32Op::get_concrete_op_info(),
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),
        "cuda_device::warp::shuffle_down_sync" => {
            Ok(Some(intrinsics::warp::emit_warp_shuffle_i32(
                ctx,
                body,
                dialect_nvvm::ops::ShflSyncDownI32Op::get_concrete_op_info(),
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
        "cuda_device::warp::shuffle_xor_sync" => Ok(Some(intrinsics::warp::emit_warp_shuffle_i32(
            ctx,
            body,
            dialect_nvvm::ops::ShflSyncBflyI32Op::get_concrete_op_info(),
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),
        "cuda_device::warp::shuffle_f32_sync" => Ok(Some(intrinsics::warp::emit_warp_shuffle_f32(
            ctx,
            body,
            dialect_nvvm::ops::ShflSyncIdxF32Op::get_concrete_op_info(),
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),
        "cuda_device::warp::shuffle_up_f32_sync" => {
            Ok(Some(intrinsics::warp::emit_warp_shuffle_f32(
                ctx,
                body,
                dialect_nvvm::ops::ShflSyncUpF32Op::get_concrete_op_info(),
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
        "cuda_device::warp::shuffle_down_f32_sync" => {
            Ok(Some(intrinsics::warp::emit_warp_shuffle_f32(
                ctx,
                body,
                dialect_nvvm::ops::ShflSyncDownF32Op::get_concrete_op_info(),
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
        "cuda_device::warp::shuffle_xor_f32_sync" => {
            Ok(Some(intrinsics::warp::emit_warp_shuffle_f32(
                ctx,
                body,
                dialect_nvvm::ops::ShflSyncBflyF32Op::get_concrete_op_info(),
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
        "cuda_device::warp::all_sync" => Ok(Some(intrinsics::warp::emit_warp_vote(
            ctx,
            body,
            dialect_nvvm::ops::VoteSyncAllOp::get_concrete_op_info(),
            false,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),
        "cuda_device::warp::any_sync" => Ok(Some(intrinsics::warp::emit_warp_vote(
            ctx,
            body,
            dialect_nvvm::ops::VoteSyncAnyOp::get_concrete_op_info(),
            false,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),
        "cuda_device::warp::ballot_sync" => Ok(Some(intrinsics::warp::emit_warp_vote(
            ctx,
            body,
            dialect_nvvm::ops::VoteSyncBallotOp::get_concrete_op_info(),
            true,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),
        "cuda_device::warp::match_any_sync" => Ok(Some(intrinsics::warp::emit_warp_match(
            ctx,
            body,
            dialect_nvvm::ops::MatchAnySyncI32Op::get_concrete_op_info(),
            false,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),
        "cuda_device::warp::match_any_i64_sync" => Ok(Some(intrinsics::warp::emit_warp_match(
            ctx,
            body,
            dialect_nvvm::ops::MatchAnySyncI64Op::get_concrete_op_info(),
            true,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),
        "cuda_device::warp::match_all_sync" => Ok(Some(intrinsics::warp::emit_warp_match(
            ctx,
            body,
            dialect_nvvm::ops::MatchAllSyncI32Op::get_concrete_op_info(),
            false,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
        )?)),
        "cuda_device::warp::match_all_i64_sync" => Ok(Some(intrinsics::warp::emit_warp_match(
            ctx,
            body,
            dialect_nvvm::ops::MatchAllSyncI64Op::get_concrete_op_info(),
            true,
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
        // WGMMA (from intrinsics::wgmma)
        // =================================================================
        "cuda_device::wgmma::wgmma_fence" => Ok(Some(intrinsics::wgmma::emit_wgmma_fence(
            ctx, args, target, block_ptr, prev_op, block_map, loc,
        )?)),
        "cuda_device::wgmma::wgmma_commit_group" => {
            Ok(Some(intrinsics::wgmma::emit_wgmma_commit_group(
                ctx, args, target, block_ptr, prev_op, block_map, loc,
            )?))
        }
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
        // TMA (from intrinsics::tma)
        // =================================================================
        "cuda_device::tma::cp_async_bulk_tensor_1d_g2s" => Ok(Some(intrinsics::tma::emit_tma_g2s(
            ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc, 1,
        )?)),
        "cuda_device::tma::cp_async_bulk_tensor_2d_g2s" => Ok(Some(intrinsics::tma::emit_tma_g2s(
            ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc, 2,
        )?)),
        "cuda_device::tma::cp_async_bulk_tensor_2d_g2s_multicast" => {
            Ok(Some(intrinsics::tma::emit_tma_g2s_multicast(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }
        "cuda_device::tma::cp_async_bulk_tensor_2d_g2s_multicast_cg2" => {
            Ok(Some(intrinsics::tma::emit_tma_g2s_multicast_cg2(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }
        "cuda_device::tma::cp_async_bulk_tensor_3d_g2s" => Ok(Some(intrinsics::tma::emit_tma_g2s(
            ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc, 3,
        )?)),
        "cuda_device::tma::cp_async_bulk_tensor_4d_g2s" => Ok(Some(intrinsics::tma::emit_tma_g2s(
            ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc, 4,
        )?)),
        "cuda_device::tma::cp_async_bulk_tensor_5d_g2s" => Ok(Some(intrinsics::tma::emit_tma_g2s(
            ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc, 5,
        )?)),
        "cuda_device::tma::cp_async_bulk_tensor_1d_s2g" => Ok(Some(intrinsics::tma::emit_tma_s2g(
            ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc, 1,
        )?)),
        "cuda_device::tma::cp_async_bulk_tensor_2d_s2g" => Ok(Some(intrinsics::tma::emit_tma_s2g(
            ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc, 2,
        )?)),
        "cuda_device::tma::cp_async_bulk_tensor_3d_s2g" => Ok(Some(intrinsics::tma::emit_tma_s2g(
            ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc, 3,
        )?)),
        "cuda_device::tma::cp_async_bulk_tensor_4d_s2g" => Ok(Some(intrinsics::tma::emit_tma_s2g(
            ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc, 4,
        )?)),
        "cuda_device::tma::cp_async_bulk_tensor_5d_s2g" => Ok(Some(intrinsics::tma::emit_tma_s2g(
            ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc, 5,
        )?)),
        "cuda_device::tma::cp_async_bulk_commit_group" => {
            Ok(Some(intrinsics::tma::emit_tma_commit_group(
                ctx, args, target, block_ptr, prev_op, block_map, loc,
            )?))
        }
        "cuda_device::tma::cp_async_bulk_wait_group" => {
            Ok(Some(intrinsics::tma::emit_tma_wait_group(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc, false,
            )?))
        }
        "cuda_device::tma::cp_async_bulk_wait_group_read" => {
            Ok(Some(intrinsics::tma::emit_tma_wait_group(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc, true,
            )?))
        }

        // =================================================================
        // Tcgen05 (from intrinsics::tcgen05)
        // =================================================================
        "cuda_device::tcgen05::tcgen05_alloc" => Ok(Some(intrinsics::tcgen05::emit_tcgen05_alloc(
            ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
        )?)),
        "cuda_device::tcgen05::tcgen05_dealloc" => {
            Ok(Some(intrinsics::tcgen05::emit_tcgen05_dealloc(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }
        "cuda_device::tcgen05::tcgen05_relinquish_alloc_permit" => Ok(Some(
            intrinsics::tcgen05::emit_tcgen05_relinquish_alloc_permit(
                ctx, args, target, block_ptr, prev_op, block_map, loc,
            )?,
        )),
        "cuda_device::tcgen05::tcgen05_fence_before_thread_sync" => Ok(Some(
            intrinsics::tcgen05::emit_tcgen05_fence_before_thread_sync(
                ctx, args, target, block_ptr, prev_op, block_map, loc,
            )?,
        )),
        "cuda_device::tcgen05::tcgen05_fence_after_thread_sync" => Ok(Some(
            intrinsics::tcgen05::emit_tcgen05_fence_after_thread_sync(
                ctx, args, target, block_ptr, prev_op, block_map, loc,
            )?,
        )),
        "cuda_device::tcgen05::tcgen05_commit" => {
            Ok(Some(intrinsics::tcgen05::emit_tcgen05_commit(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }
        "cuda_device::tcgen05::tcgen05_commit_shared_cluster" => Ok(Some(
            intrinsics::tcgen05::emit_tcgen05_commit_shared_cluster(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?,
        )),
        // NOTE: tcgen05_make_smem_desc and tcgen05_make_smem_desc_strided removed
        // Use Tcgen05SmemDescriptor::builder() in cuda-core instead
        "cuda_device::tcgen05::tcgen05_mma_ws_f16" => {
            Ok(Some(intrinsics::tcgen05::emit_tcgen05_mma_ws_f16(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }
        "cuda_device::tcgen05::tcgen05_mma_f16" => {
            Ok(Some(intrinsics::tcgen05::emit_tcgen05_mma_f16(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }
        "cuda_device::tcgen05::tcgen05_mma_ws_bf16" => {
            Ok(Some(intrinsics::tcgen05::emit_tcgen05_mma_ws_bf16(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }
        "cuda_device::tcgen05::tcgen05_mma_ws_tf32" => {
            Ok(Some(intrinsics::tcgen05::emit_tcgen05_mma_ws_tf32(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }
        "cuda_device::tcgen05::tcgen05_cp_smem_to_tmem" => {
            Ok(Some(intrinsics::tcgen05::emit_tcgen05_cp_smem_to_tmem(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }
        // NOTE: tcgen05_st_tmem_to_smem* and tcgen05_ld_* (non-pure) removed
        // Use tcgen05_ld_16x256b_pure or tcgen05_ld_16x256b_x8_pure instead
        "cuda_device::tcgen05::tcgen05_ld_16x256b_x8_pure" => {
            Ok(Some(intrinsics::tcgen05::emit_tcgen05_ld_16x256b_x8_pure(
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
        "cuda_device::tcgen05::tcgen05_ld_16x256b_pure" => {
            Ok(Some(intrinsics::tcgen05::emit_tcgen05_ld_16x256b_pure(
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
        "cuda_device::tcgen05::tcgen05_load_wait" => {
            Ok(Some(intrinsics::tcgen05::emit_tcgen05_load_wait(
                ctx, args, target, block_ptr, prev_op, block_map, loc,
            )?))
        }
        "cuda_device::tcgen05::tcgen05_store_wait" => {
            Ok(Some(intrinsics::tcgen05::emit_tcgen05_store_wait(
                ctx, args, target, block_ptr, prev_op, block_map, loc,
            )?))
        }

        // CTA pair (cta_group::2) variants
        "cuda_device::tcgen05::tcgen05_alloc_cg2" => {
            Ok(Some(intrinsics::tcgen05::emit_tcgen05_alloc_cg2(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }
        "cuda_device::tcgen05::tcgen05_dealloc_cg2" => {
            Ok(Some(intrinsics::tcgen05::emit_tcgen05_dealloc_cg2(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }
        "cuda_device::tcgen05::tcgen05_relinquish_alloc_permit_cg2" => Ok(Some(
            intrinsics::tcgen05::emit_tcgen05_relinquish_alloc_permit_cg2(
                ctx, args, target, block_ptr, prev_op, block_map, loc,
            )?,
        )),
        "cuda_device::tcgen05::tcgen05_mma_f16_cg2" => {
            Ok(Some(intrinsics::tcgen05::emit_tcgen05_mma_f16_cg2(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }
        "cuda_device::tcgen05::tcgen05_commit_cg2" => {
            Ok(Some(intrinsics::tcgen05::emit_tcgen05_commit_cg2(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }
        "cuda_device::tcgen05::tcgen05_commit_shared_cluster_cg2" => Ok(Some(
            intrinsics::tcgen05::emit_tcgen05_commit_shared_cluster_cg2(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?,
        )),
        "cuda_device::tcgen05::tcgen05_commit_multicast_cg2" => Ok(Some(
            intrinsics::tcgen05::emit_tcgen05_commit_multicast_cg2(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?,
        )),
        "cuda_device::tcgen05::tcgen05_cp_smem_to_tmem_cg2" => {
            Ok(Some(intrinsics::tcgen05::emit_tcgen05_cp_smem_to_tmem_cg2(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }

        // =================================================================
        // Memory Operations (from intrinsics::memory)
        // Note: stmatrix and cvt are under cuda_device::tcgen05::
        // =================================================================
        "cuda_device::tcgen05::stmatrix_m8n8_x4" => {
            Ok(Some(intrinsics::memory::emit_stmatrix_m8n8_x4(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }
        "cuda_device::tcgen05::stmatrix_m8n8_x4_trans" => {
            Ok(Some(intrinsics::memory::emit_stmatrix_m8n8_x4_trans(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }
        "cuda_device::tcgen05::stmatrix_m8n8_x2" => {
            Ok(Some(intrinsics::memory::emit_stmatrix_m8n8_x2(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }
        "cuda_device::tcgen05::stmatrix_m8n8_x2_trans" => {
            Ok(Some(intrinsics::memory::emit_stmatrix_m8n8_x2_trans(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }
        "cuda_device::tcgen05::cvt_f32x2_bf16x2" => {
            Ok(Some(intrinsics::memory::emit_cvt_f32x2_bf16x2(
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
        // CLC - Cluster Launch Control (from intrinsics::clc)
        // =================================================================
        "cuda_device::clc::clc_try_cancel" => Ok(Some(intrinsics::clc::emit_clc_try_cancel(
            ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
        )?)),
        "cuda_device::clc::clc_try_cancel_multicast" => {
            Ok(Some(intrinsics::clc::emit_clc_try_cancel_multicast(
                ctx, body, args, target, block_ptr, prev_op, value_map, block_map, loc,
            )?))
        }
        "cuda_device::clc::clc_query_is_canceled" => {
            Ok(Some(intrinsics::clc::emit_clc_query_is_canceled(
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
        "cuda_device::clc::clc_query_get_first_ctaid_x" => {
            Ok(Some(intrinsics::clc::emit_clc_query_get_first_ctaid_x(
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
        "cuda_device::clc::clc_query_get_first_ctaid_y" => {
            Ok(Some(intrinsics::clc::emit_clc_query_get_first_ctaid_y(
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
        "cuda_device::clc::clc_query_get_first_ctaid_z" => {
            Ok(Some(intrinsics::clc::emit_clc_query_get_first_ctaid_z(
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

        // =================================================================
        // Prefix-based matches (for const generics like wgmma_wait_group::<N>)
        // =================================================================
        path if path.starts_with("cuda_device::wgmma::wgmma_wait_group") => {
            // Extract N from path like "cuda_device::wgmma::wgmma_wait_group::<0>"
            let n = path
                .split("::<")
                .nth(1)
                .and_then(|s| s.strip_suffix('>'))
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0);
            Ok(Some(intrinsics::wgmma::emit_wgmma_wait_group(
                ctx, args, target, block_ptr, prev_op, block_map, loc, n,
            )?))
        }

        // DisjointSlice methods using prefix matching
        path if path.starts_with("cuda_device::DisjointSlice::") => {
            if let Some(method) = path.rsplit("::").next() {
                match method {
                    "get_mut" | "get_unchecked_mut" | "get_thread_local" => {
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
