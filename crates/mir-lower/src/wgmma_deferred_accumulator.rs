/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Fuse sound BF16 WGMMA sequences before MIR-to-LLVM conversion.
//!
//! The public MMA operation exposes its accumulator through a pointer, but PTX
//! requires all 32 accumulator registers to remain inaccessible until the
//! corresponding `wgmma.wait_group` completes. This pass recognizes both closed
//! straight-line regions and one deliberately narrow counted K-loop shape. The
//! canonical `[[f32; 8]; 4]` accumulator is adapted through 32 scalar SSA values;
//! unsupported accumulator shapes retain the existing deferred pointer fallback.
//!
//! Straight-line regions keep the existing shape:
//!
//! ```text
//! wgmma.fence
//! one or more m64n64k16.f32.bf16.bf16 MMA operations on one accumulator
//! wgmma.commit_group
//! wgmma.wait_group<0>
//! ```
//!
//! A counted K-loop may place the fence in the loop preheader, one pointer-form
//! MMA in the unique latch, and the commit/final `wait_group<0>` in the unique
//! exit. The loop must have a compile-time trip count and two `u64` descriptor
//! block arguments whose back-edge values are either unchanged or `arg + const`.
//! The complete asynchronous lifetime is then represented by one value-form loop
//! operation so LLVM never sees an in-flight accumulator between iterations.
//!
//! Straight-line partial-wait pipelines are recognized separately. A static
//! `wait_group<N>` with `N > 0` requires `N + 1` canonical accumulator slots.
//! Groups are committed and the slots are reused round-robin only after the
//! corresponding partial wait has made the oldest slot safe. Every accepted
//! pipeline ends with `wait_group<0>` before any accumulator value escapes.

use dialect_mir::{
    ops::{
        MirAddOp, MirArrayElementAddrOp, MirCondBranchOp, MirConstantOp, MirFuncOp, MirGeOp,
        MirGotoOp, MirGtOp, MirLeOp, MirLoadOp, MirLtOp, MirNotOp, MirStorageDeadOp,
        MirStorageLiveOp, MirStoreOp, MirSubOp,
    },
    types::{MirArrayType, MirPtrType, address_space},
};
use dialect_nvvm::ops::{
    WgmmaCommitGroupSyncAlignedOp, WgmmaFenceSyncAlignedOp, WgmmaMmaGroupM64N64K16F32Bf16Op,
    WgmmaMmaGroupValuesM64N64K16F32Bf16Op, WgmmaMmaLoopValuesM64N64K16F32Bf16Op,
    WgmmaMmaM64N64K16F32Bf16Op, WgmmaMmaPipelineValuesM64N64K16F32Bf16Op,
    WgmmaWaitGroupSyncAlignedOp,
};
use mir_transforms::analyses::{induction, loop_info::LoopInfo};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::IntegerAttr,
        op_interfaces::BranchOpInterface,
        ops::ConstantOp,
        types::{FP32Type, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    graph::dominance::DomInfo,
    irbuild::{
        listener::Recorder,
        rewriter::{IRRewriter, Rewriter},
    },
    linked_list::ContainsLinkedList,
    location::Located,
    op::{Op, op_cast},
    operation::Operation,
    opts::simplify_cfg::remove_blocks_inside_op,
    region::Region,
    result::Result,
    r#type::{TypeHandle, Typed},
    utils::apint::APInt,
    value::Value,
};
use rustc_hash::FxHashSet;
use std::num::NonZeroUsize;

const ACCUMULATOR_ROWS: usize = 4;
const ACCUMULATOR_COLUMNS: usize = 8;
const ACCUMULATOR_LEN: usize = ACCUMULATOR_ROWS * ACCUMULATOR_COLUMNS;

struct FusionPlan {
    fence: Ptr<Operation>,
    mmas: Vec<Ptr<Operation>>,
    commit: Ptr<Operation>,
    wait: Ptr<Operation>,
    accumulator: Value,
    descriptors: Vec<Value>,
}

struct PipelinePlan {
    fence: Ptr<Operation>,
    mmas: Vec<Ptr<Operation>>,
    commits: Vec<Ptr<Operation>>,
    waits: Vec<Ptr<Operation>>,
    accumulators: Vec<Value>,
    descriptors: Vec<Value>,
    max_pending_groups: u8,
}

#[derive(Clone, Copy)]
enum PipelineEvent {
    Mma {
        operation: Ptr<Operation>,
        accumulator: Value,
        desc_a: Value,
        desc_b: Value,
    },
    Commit(Ptr<Operation>),
    Wait {
        operation: Ptr<Operation>,
        max_pending: u64,
    },
}

struct CountedLoopPlan {
    fence: Ptr<Operation>,
    preheader: Ptr<BasicBlock>,
    preheader_terminator: Ptr<Operation>,
    exit: Ptr<BasicBlock>,
    commit: Ptr<Operation>,
    wait: Ptr<Operation>,
    accumulator: Value,
    desc_a_base: Value,
    desc_b_base: Value,
    desc_a_step: u64,
    desc_b_step: u64,
    trip_count: u64,
    row_type: TypeHandle,
    element_type: TypeHandle,
}

fn collect_blocks(ctx: &Context, root: Ptr<Operation>) -> Vec<Ptr<BasicBlock>> {
    fn visit(ctx: &Context, op: Ptr<Operation>, blocks: &mut Vec<Ptr<BasicBlock>>) {
        let regions: Vec<_> = op.deref(ctx).regions().collect();
        for region in regions {
            let region_blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
            for block in region_blocks {
                blocks.push(block);
                let children: Vec<_> = block.deref(ctx).iter(ctx).collect();
                for child in children {
                    visit(ctx, child, blocks);
                }
            }
        }
    }

    let mut blocks = Vec::new();
    visit(ctx, root, &mut blocks);
    blocks
}

fn collect_functions(ctx: &Context, root: Ptr<Operation>) -> Vec<Ptr<Operation>> {
    fn visit(ctx: &Context, op: Ptr<Operation>, functions: &mut Vec<Ptr<Operation>>) {
        if Operation::get_op::<MirFuncOp>(op, ctx).is_some() {
            functions.push(op);
            return;
        }

        let regions: Vec<_> = op.deref(ctx).regions().collect();
        for region in regions {
            let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
            for block in blocks {
                let children: Vec<_> = block.deref(ctx).iter(ctx).collect();
                for child in children {
                    visit(ctx, child, functions);
                }
            }
        }
    }

    let mut functions = Vec::new();
    visit(ctx, root, &mut functions);
    functions
}

fn is_u64_value(ctx: &Context, value: Value) -> bool {
    let ty = value.get_type(ctx);
    let ty = ty.deref(ctx);
    ty.downcast_ref::<IntegerType>().is_some_and(|integer| {
        integer.width() == 64 && integer.signedness() == Signedness::Unsigned
    })
}

fn edge_operands(
    ctx: &Context,
    pred: Ptr<BasicBlock>,
    succ: Ptr<BasicBlock>,
) -> Option<Vec<Value>> {
    let terminator = pred.deref(ctx).get_terminator(ctx)?;
    let successors: Vec<_> = terminator.deref(ctx).successors().collect();
    let successor_index = successors.iter().position(|candidate| *candidate == succ)?;
    let operation = Operation::get_op_dyn(terminator, ctx);
    let branch = op_cast::<dyn BranchOpInterface>(operation.as_ref())?;
    Some(branch.successor_operands(ctx, successor_index))
}

fn value_has_outside_loop_use(
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

fn loop_values_escape(ctx: &Context, loop_blocks: &FxHashSet<Ptr<BasicBlock>>) -> bool {
    for &block in loop_blocks {
        for argument in block.deref(ctx).arguments() {
            if value_has_outside_loop_use(ctx, argument, loop_blocks) {
                return true;
            }
        }
        for operation in block.deref(ctx).iter(ctx).collect::<Vec<_>>() {
            for result in operation.deref(ctx).results() {
                if value_has_outside_loop_use(ctx, result, loop_blocks) {
                    return true;
                }
            }
        }
    }
    false
}

fn counted_loop_operation_is_supported(ctx: &Context, operation: Ptr<Operation>) -> bool {
    Operation::get_op::<MirConstantOp>(operation, ctx).is_some()
        || Operation::get_op::<ConstantOp>(operation, ctx).is_some()
        || Operation::get_op::<MirStorageLiveOp>(operation, ctx).is_some()
        || Operation::get_op::<MirStorageDeadOp>(operation, ctx).is_some()
        || Operation::get_op::<MirAddOp>(operation, ctx).is_some()
        || Operation::get_op::<MirSubOp>(operation, ctx).is_some()
        || Operation::get_op::<MirNotOp>(operation, ctx).is_some()
        || Operation::get_op::<MirLtOp>(operation, ctx).is_some()
        || Operation::get_op::<MirLeOp>(operation, ctx).is_some()
        || Operation::get_op::<MirGtOp>(operation, ctx).is_some()
        || Operation::get_op::<MirGeOp>(operation, ctx).is_some()
        || Operation::get_op::<MirCondBranchOp>(operation, ctx).is_some()
        || Operation::get_op::<MirGotoOp>(operation, ctx).is_some()
        || Operation::get_op::<WgmmaMmaM64N64K16F32Bf16Op>(operation, ctx).is_some()
}

fn find_preheader_fence(
    ctx: &Context,
    preheader: Ptr<BasicBlock>,
) -> Result<Option<Ptr<Operation>>> {
    let operations: Vec<_> = preheader.deref(ctx).iter(ctx).collect();
    let fences = operations
        .iter()
        .copied()
        .filter(|operation| Operation::get_op::<WgmmaFenceSyncAlignedOp>(*operation, ctx).is_some())
        .collect::<Vec<_>>();
    let [fence] = fences.as_slice() else {
        return Ok(None);
    };
    require_nullary_control_op(ctx, *fence, "WGMMA fence")?;

    let fence_index = operations
        .iter()
        .position(|operation| operation == fence)
        .expect("fence came from preheader operation list");
    for operation in operations.iter().copied().skip(fence_index + 1) {
        if preheader.deref(ctx).get_terminator(ctx) == Some(operation)
            || is_ignorable(ctx, operation)
        {
            continue;
        }
        return Ok(None);
    }

    Ok(Some(*fence))
}

fn find_exit_commit_wait(
    ctx: &Context,
    exit: Ptr<BasicBlock>,
) -> Result<Option<(Ptr<Operation>, Ptr<Operation>)>> {
    let mut commit = None;
    for operation in exit.deref(ctx).iter(ctx).collect::<Vec<_>>() {
        if is_ignorable(ctx, operation) {
            continue;
        }

        if commit.is_none() {
            if Operation::get_op::<WgmmaCommitGroupSyncAlignedOp>(operation, ctx).is_none() {
                return Ok(None);
            }
            require_nullary_control_op(ctx, operation, "WGMMA commit_group")?;
            commit = Some(operation);
            continue;
        }

        if Operation::get_op::<WgmmaWaitGroupSyncAlignedOp>(operation, ctx).is_some() {
            require_wait_shape(ctx, operation)?;
            let wait_operand = operation.deref(ctx).get_operand(0);
            if integer_constant_u64(ctx, wait_operand) != Some(0) {
                return pliron::input_err_noloc!(
                    "counted WGMMA loop requires a final wait_group<0>"
                );
            }
            return Ok(Some((commit.expect("commit is set"), operation)));
        }

        return pliron::input_err_noloc!(
            "unsupported operation between WGMMA commit_group and final wait_group<0>"
        );
    }

    Ok(None)
}

fn affine_u64_step(ctx: &Context, argument: Value, next: Value) -> Option<u64> {
    if !is_u64_value(ctx, argument) || !is_u64_value(ctx, next) {
        return None;
    }
    if next == argument {
        return Some(0);
    }

    let defining_op = next.defining_op()?;
    Operation::get_op::<MirAddOp>(defining_op, ctx)?;

    let operation = defining_op.deref(ctx);
    if operation.get_num_operands() != 2 || operation.get_num_results() != 1 {
        return None;
    }
    let lhs = operation.get_operand(0);
    let rhs = operation.get_operand(1);
    if lhs == argument {
        integer_constant_u64(ctx, rhs)
    } else if rhs == argument {
        integer_constant_u64(ctx, lhs)
    } else {
        None
    }
}

fn match_counted_loop(
    ctx: &Context,
    info: &LoopInfo,
    region: Ptr<Region>,
    loop_id: usize,
) -> Result<Option<CountedLoopPlan>> {
    let r#loop = &info.loops()[loop_id];
    if !r#loop.children.is_empty() || r#loop.latches.len() != 1 {
        return Ok(None);
    }

    let Some(preheader) = info.preheader(ctx, region, loop_id) else {
        return Ok(None);
    };
    let Some(fence) = find_preheader_fence(ctx, preheader)? else {
        return Ok(None);
    };

    let exiting_blocks = info.exiting_blocks(ctx, region, loop_id);
    if exiting_blocks.len() != 1 || exiting_blocks[0] != r#loop.header {
        return Ok(None);
    }
    let exit_blocks = info.exit_blocks(ctx, region, loop_id);
    if exit_blocks.len() != 1 {
        return Ok(None);
    }
    let exit = exit_blocks[0];
    if exit.deref(ctx).get_num_arguments() != 0 {
        return Ok(None);
    }

    let Some((commit, wait)) = find_exit_commit_wait(ctx, exit)? else {
        return Ok(None);
    };

    let preheader_terminator = preheader
        .deref(ctx)
        .get_terminator(ctx)
        .expect("counted loop preheader must have a terminator");
    if Operation::get_op::<MirGotoOp>(preheader_terminator, ctx).is_none() {
        return Ok(None);
    }

    let latch = r#loop.latches[0];
    let latch_terminator = latch
        .deref(ctx)
        .get_terminator(ctx)
        .expect("counted loop latch must have a terminator");
    if Operation::get_op::<MirGotoOp>(latch_terminator, ctx).is_none() {
        return Ok(None);
    }

    let header = r#loop.header;
    let header_args = header.deref(ctx).arguments().collect::<Vec<_>>();
    let Some(init_operands) = edge_operands(ctx, preheader, header) else {
        return Ok(None);
    };
    let Some(recurrence_operands) = edge_operands(ctx, latch, header) else {
        return Ok(None);
    };
    if init_operands.len() != header_args.len() || recurrence_operands.len() != header_args.len() {
        return Ok(None);
    }

    let recurrences = induction::analyze(ctx, info, loop_id, preheader);
    let Some(primary_iv) = recurrences.primary_iv else {
        return Ok(None);
    };
    let Some(trip_count) = recurrences.trip_count else {
        return Ok(None);
    };
    if trip_count == 0 || primary_iv >= header_args.len() {
        return Ok(None);
    }

    let mut mmas = Vec::new();
    for &block in &r#loop.blocks {
        for operation in block.deref(ctx).iter(ctx).collect::<Vec<_>>() {
            if !counted_loop_operation_is_supported(ctx, operation) {
                return Ok(None);
            }
            if Operation::get_op::<WgmmaMmaM64N64K16F32Bf16Op>(operation, ctx).is_some() {
                mmas.push(operation);
            }
        }
    }
    let [mma] = mmas.as_slice() else {
        return Ok(None);
    };
    if mma.deref(ctx).get_parent_block() != Some(latch) {
        return Ok(None);
    }
    require_pointer_mma_shape(ctx, *mma)?;

    let mma_ref = mma.deref(ctx);
    let accumulator = mma_ref.get_operand(0);
    require_supported_accumulator(ctx, accumulator)?;
    let Some((row_type, element_type)) = value_accumulator_shape(ctx, accumulator) else {
        return Ok(None);
    };
    if accumulator
        .defining_block()
        .is_some_and(|block| r#loop.blocks.contains(&block))
        || accumulator
            .defining_op()
            .and_then(|operation| operation.deref(ctx).get_parent_block())
            .is_some_and(|block| r#loop.blocks.contains(&block))
    {
        return Ok(None);
    }

    let desc_a = mma_ref.get_operand(1);
    let desc_b = mma_ref.get_operand(2);
    if !is_u64_value(ctx, desc_a) || !is_u64_value(ctx, desc_b) {
        return Ok(None);
    }

    let Some(desc_a_index) = header_args.iter().position(|value| *value == desc_a) else {
        return Ok(None);
    };
    let Some(desc_b_index) = header_args.iter().position(|value| *value == desc_b) else {
        return Ok(None);
    };
    if desc_a_index == desc_b_index || desc_a_index == primary_iv || desc_b_index == primary_iv {
        return Ok(None);
    }

    let desc_a_base = init_operands[desc_a_index];
    let desc_b_base = init_operands[desc_b_index];
    if !is_u64_value(ctx, desc_a_base) || !is_u64_value(ctx, desc_b_base) {
        return Ok(None);
    }

    let Some(desc_a_step) = affine_u64_step(
        ctx,
        header_args[desc_a_index],
        recurrence_operands[desc_a_index],
    ) else {
        return Ok(None);
    };
    let Some(desc_b_step) = affine_u64_step(
        ctx,
        header_args[desc_b_index],
        recurrence_operands[desc_b_index],
    ) else {
        return Ok(None);
    };

    if loop_values_escape(ctx, &r#loop.blocks) {
        return Ok(None);
    }

    Ok(Some(CountedLoopPlan {
        fence,
        preheader,
        preheader_terminator,
        exit,
        commit,
        wait,
        accumulator,
        desc_a_base,
        desc_b_base,
        desc_a_step,
        desc_b_step,
        trip_count,
        row_type,
        element_type,
    }))
}

fn integer_constant_u64(ctx: &Context, value: Value) -> Option<u64> {
    let value_type = value.get_type(ctx);
    let value_type_ref = value_type.deref(ctx);
    let integer_type = value_type_ref.downcast_ref::<IntegerType>()?;
    if integer_type.width() != 64 || integer_type.signedness() != Signedness::Unsigned {
        return None;
    }

    let defining_op = value.defining_op()?;

    if let Some(constant) = Operation::get_op::<MirConstantOp>(defining_op, ctx) {
        return constant
            .get_attr_value(ctx)
            .map(|attribute| attribute.value().to_u64());
    }

    let constant = Operation::get_op::<ConstantOp>(defining_op, ctx)?;
    let attribute = constant.get_value(ctx);
    attribute
        .downcast_ref::<IntegerAttr>()
        .map(|integer| integer.value().to_u64())
}

fn require_nullary_control_op(
    ctx: &Context,
    operation: Ptr<Operation>,
    operation_name: &str,
) -> Result<()> {
    let operation_ref = operation.deref(ctx);
    if operation_ref.get_num_operands() != 0 || operation_ref.get_num_results() != 0 {
        return pliron::input_err_noloc!("{operation_name} requires no operands and no results");
    }
    Ok(())
}

fn require_pointer_mma_shape(ctx: &Context, operation: Ptr<Operation>) -> Result<()> {
    let operation_ref = operation.deref(ctx);
    if operation_ref.get_num_operands() != 3 || operation_ref.get_num_results() != 0 {
        return pliron::input_err_noloc!(
            "WGMMA pointer-form MMA requires three operands and no results"
        );
    }

    for operand_index in [1, 2] {
        let descriptor_type = operation_ref.get_operand(operand_index).get_type(ctx);
        let descriptor_type_ref = descriptor_type.deref(ctx);
        let Some(integer_type) = descriptor_type_ref.downcast_ref::<IntegerType>() else {
            return pliron::input_err_noloc!("WGMMA pointer-form MMA descriptors must be u64");
        };

        if integer_type.width() != 64 || integer_type.signedness() != Signedness::Unsigned {
            return pliron::input_err_noloc!("WGMMA pointer-form MMA descriptors must be u64");
        }
    }

    Ok(())
}

fn require_wait_shape(ctx: &Context, operation: Ptr<Operation>) -> Result<()> {
    let operation_ref = operation.deref(ctx);
    if operation_ref.get_num_operands() != 1 || operation_ref.get_num_results() != 0 {
        return pliron::input_err_noloc!("WGMMA wait_group requires one operand and no results");
    }
    Ok(())
}

fn require_supported_accumulator(ctx: &Context, accumulator: Value) -> Result<()> {
    let accumulator_type = accumulator.get_type(ctx);
    let accumulator_type_ref = accumulator_type.deref(ctx);
    let Some(pointer_type) = accumulator_type_ref.downcast_ref::<MirPtrType>() else {
        return pliron::input_err_noloc!("WGMMA deferred accumulator must be a MIR pointer");
    };

    if !pointer_type.is_mutable() {
        return pliron::input_err_noloc!("WGMMA deferred accumulator must be mutable");
    }
    if pointer_type.address_space() != address_space::GENERIC {
        return pliron::input_err_noloc!(
            "WGMMA deferred accumulator must use the generic address space"
        );
    }
    Ok(())
}

fn value_accumulator_shape(ctx: &Context, accumulator: Value) -> Option<(TypeHandle, TypeHandle)> {
    let accumulator_type = accumulator.get_type(ctx);
    let accumulator_type_ref = accumulator_type.deref(ctx);
    let pointer_type = accumulator_type_ref.downcast_ref::<MirPtrType>()?;
    if !pointer_type.is_mutable() || pointer_type.address_space() != address_space::GENERIC {
        return None;
    }

    let outer_type = pointer_type.pointee;
    let outer_type_ref = outer_type.deref(ctx);
    let outer_array = outer_type_ref.downcast_ref::<MirArrayType>()?;
    if outer_array.size() != ACCUMULATOR_ROWS as u64 {
        return None;
    }

    let row_type = outer_array.element_type();
    let row_type_ref = row_type.deref(ctx);
    let row_array = row_type_ref.downcast_ref::<MirArrayType>()?;
    if row_array.size() != ACCUMULATOR_COLUMNS as u64 {
        return None;
    }

    let element_type = row_array.element_type();
    element_type.deref(ctx).downcast_ref::<FP32Type>()?;

    Some((row_type, element_type))
}

fn insert_u64_constant_before(ctx: &mut Context, value: u64, before: Ptr<Operation>) -> Value {
    let u64_type = IntegerType::get(ctx, 64, Signedness::Unsigned);
    let constant = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![u64_type.into()],
        vec![],
        vec![],
        0,
    );
    MirConstantOp::new(constant).set_attr_value(
        ctx,
        IntegerAttr::new(
            u64_type,
            APInt::from_u64(value, NonZeroUsize::new(64).unwrap()),
        ),
    );
    constant.insert_before(ctx, before);
    constant.deref(ctx).get_result(0)
}

fn erase_original_sequence(
    ctx: &mut Context,
    fence: Ptr<Operation>,
    mmas: Vec<Ptr<Operation>>,
    commit: Ptr<Operation>,
    wait: Ptr<Operation>,
) {
    let mut rewriter = IRRewriter::<Recorder>::default();
    rewriter.erase_operation(ctx, fence);
    for mma in mmas {
        rewriter.erase_operation(ctx, mma);
    }
    rewriter.erase_operation(ctx, commit);
    rewriter.erase_operation(ctx, wait);
}

fn load_accumulator_values_before(
    ctx: &mut Context,
    before: Ptr<Operation>,
    accumulator: Value,
    row_type: TypeHandle,
    element_type: TypeHandle,
    loc: pliron::location::Location,
) -> (Vec<Value>, Vec<Value>) {
    let row_pointer_type: TypeHandle = MirPtrType::get_generic(ctx, row_type, true).into();
    let element_pointer_type: TypeHandle = MirPtrType::get_generic(ctx, element_type, true).into();

    let row_indices = (0..ACCUMULATOR_ROWS)
        .map(|index| insert_u64_constant_before(ctx, index as u64, before))
        .collect::<Vec<_>>();
    let column_indices = (0..ACCUMULATOR_COLUMNS)
        .map(|index| insert_u64_constant_before(ctx, index as u64, before))
        .collect::<Vec<_>>();

    let mut element_pointers = Vec::with_capacity(ACCUMULATOR_LEN);
    let mut accumulator_values = Vec::with_capacity(ACCUMULATOR_LEN);

    for row in 0..ACCUMULATOR_ROWS {
        let row_address = Operation::new(
            ctx,
            MirArrayElementAddrOp::get_concrete_op_info(),
            vec![row_pointer_type],
            vec![accumulator, row_indices[row]],
            vec![],
            0,
        );
        row_address.deref_mut(ctx).set_loc(loc.clone());
        row_address.insert_before(ctx, before);
        let row_pointer = row_address.deref(ctx).get_result(0);

        for column in 0..ACCUMULATOR_COLUMNS {
            let element_address = Operation::new(
                ctx,
                MirArrayElementAddrOp::get_concrete_op_info(),
                vec![element_pointer_type],
                vec![row_pointer, column_indices[column]],
                vec![],
                0,
            );
            element_address.deref_mut(ctx).set_loc(loc.clone());
            element_address.insert_before(ctx, before);
            let element_pointer = element_address.deref(ctx).get_result(0);

            let load = Operation::new(
                ctx,
                MirLoadOp::get_concrete_op_info(),
                vec![element_type],
                vec![element_pointer],
                vec![],
                0,
            );
            load.deref_mut(ctx).set_loc(loc.clone());
            load.insert_before(ctx, before);

            element_pointers.push(element_pointer);
            accumulator_values.push(load.deref(ctx).get_result(0));
        }
    }

    (element_pointers, accumulator_values)
}

fn store_accumulator_values_before(
    ctx: &mut Context,
    before: Ptr<Operation>,
    element_pointers: Vec<Value>,
    accumulator_results: Vec<Value>,
    loc: pliron::location::Location,
) {
    for (element_pointer, result) in element_pointers.into_iter().zip(accumulator_results) {
        let store = Operation::new(
            ctx,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![element_pointer, result],
            vec![],
            0,
        );
        store.deref_mut(ctx).set_loc(loc.clone());
        store.insert_before(ctx, before);
    }
}

fn store_canonical_accumulator_values_before(
    ctx: &mut Context,
    before: Ptr<Operation>,
    accumulator: Value,
    row_type: TypeHandle,
    element_type: TypeHandle,
    accumulator_results: &[Value],
    loc: pliron::location::Location,
) {
    debug_assert_eq!(accumulator_results.len(), ACCUMULATOR_LEN);

    let row_pointer_type: TypeHandle = MirPtrType::get_generic(ctx, row_type, true).into();
    let element_pointer_type: TypeHandle = MirPtrType::get_generic(ctx, element_type, true).into();
    let row_indices = (0..ACCUMULATOR_ROWS)
        .map(|index| insert_u64_constant_before(ctx, index as u64, before))
        .collect::<Vec<_>>();
    let column_indices = (0..ACCUMULATOR_COLUMNS)
        .map(|index| insert_u64_constant_before(ctx, index as u64, before))
        .collect::<Vec<_>>();

    for row in 0..ACCUMULATOR_ROWS {
        let row_address = Operation::new(
            ctx,
            MirArrayElementAddrOp::get_concrete_op_info(),
            vec![row_pointer_type],
            vec![accumulator, row_indices[row]],
            vec![],
            0,
        );
        row_address.deref_mut(ctx).set_loc(loc.clone());
        row_address.insert_before(ctx, before);
        let row_pointer = row_address.deref(ctx).get_result(0);

        for column in 0..ACCUMULATOR_COLUMNS {
            let element_address = Operation::new(
                ctx,
                MirArrayElementAddrOp::get_concrete_op_info(),
                vec![element_pointer_type],
                vec![row_pointer, column_indices[column]],
                vec![],
                0,
            );
            element_address.deref_mut(ctx).set_loc(loc.clone());
            element_address.insert_before(ctx, before);
            let element_pointer = element_address.deref(ctx).get_result(0);
            let result = accumulator_results[row * ACCUMULATOR_COLUMNS + column];

            let store = Operation::new(
                ctx,
                MirStoreOp::get_concrete_op_info(),
                vec![],
                vec![element_pointer, result],
                vec![],
                0,
            );
            store.deref_mut(ctx).set_loc(loc.clone());
            store.insert_before(ctx, before);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_value_plan(
    ctx: &mut Context,
    fence: Ptr<Operation>,
    mmas: Vec<Ptr<Operation>>,
    commit: Ptr<Operation>,
    wait: Ptr<Operation>,
    accumulator: Value,
    descriptors: Vec<Value>,
    row_type: TypeHandle,
    element_type: TypeHandle,
) {
    let loc = fence.deref(ctx).loc();
    let (element_pointers, accumulator_values) =
        load_accumulator_values_before(ctx, wait, accumulator, row_type, element_type, loc.clone());

    let group = WgmmaMmaGroupValuesM64N64K16F32Bf16Op::build(ctx, accumulator_values, descriptors);
    group.deref_mut(ctx).set_loc(loc.clone());
    let accumulator_results = (0..ACCUMULATOR_LEN)
        .map(|index| group.deref(ctx).get_result(index))
        .collect::<Vec<_>>();
    group.insert_before(ctx, wait);

    store_accumulator_values_before(ctx, wait, element_pointers, accumulator_results, loc);

    erase_original_sequence(ctx, fence, mmas, commit, wait);
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

fn apply_counted_loop_plan(ctx: &mut Context, plan: CountedLoopPlan) {
    let CountedLoopPlan {
        fence,
        preheader,
        preheader_terminator,
        exit,
        commit,
        wait,
        accumulator,
        desc_a_base,
        desc_b_base,
        desc_a_step,
        desc_b_step,
        trip_count,
        row_type,
        element_type,
    } = plan;

    debug_assert_eq!(
        preheader.deref(ctx).get_terminator(ctx),
        Some(preheader_terminator)
    );
    debug_assert_eq!(exit.deref(ctx).get_num_arguments(), 0);

    let loc = fence.deref(ctx).loc();
    let (element_pointers, accumulator_values) = load_accumulator_values_before(
        ctx,
        preheader_terminator,
        accumulator,
        row_type,
        element_type,
        loc.clone(),
    );

    let desc_a_step = insert_u64_constant_before(ctx, desc_a_step, preheader_terminator);
    let desc_b_step = insert_u64_constant_before(ctx, desc_b_step, preheader_terminator);
    let trip_count = insert_u64_constant_before(ctx, trip_count, preheader_terminator);

    let group = WgmmaMmaLoopValuesM64N64K16F32Bf16Op::build(
        ctx,
        accumulator_values,
        desc_a_base,
        desc_b_base,
        desc_a_step,
        desc_b_step,
        trip_count,
    );
    group.deref_mut(ctx).set_loc(loc.clone());
    let accumulator_results = (0..ACCUMULATOR_LEN)
        .map(|index| group.deref(ctx).get_result(index))
        .collect::<Vec<_>>();
    group.insert_before(ctx, preheader_terminator);

    store_accumulator_values_before(
        ctx,
        preheader_terminator,
        element_pointers,
        accumulator_results,
        loc,
    );

    let mut rewriter = IRRewriter::<Recorder>::default();
    rewriter.erase_operation(ctx, fence);
    rewriter.erase_operation(ctx, commit);
    rewriter.erase_operation(ctx, wait);

    rewire_goto(ctx, preheader_terminator, exit, &[]);
}

fn apply_pointer_fallback(
    ctx: &mut Context,
    fence: Ptr<Operation>,
    mmas: Vec<Ptr<Operation>>,
    commit: Ptr<Operation>,
    wait: Ptr<Operation>,
    accumulator: Value,
    descriptors: Vec<Value>,
) {
    let group = WgmmaMmaGroupM64N64K16F32Bf16Op::build(ctx, accumulator, descriptors);
    group.deref_mut(ctx).set_loc(fence.deref(ctx).loc());
    group.insert_before(ctx, wait);
    erase_original_sequence(ctx, fence, mmas, commit, wait);
}

fn is_ignorable(ctx: &Context, op: Ptr<Operation>) -> bool {
    Operation::get_op::<MirConstantOp>(op, ctx).is_some()
        || Operation::get_op::<ConstantOp>(op, ctx).is_some()
        || Operation::get_op::<MirStorageLiveOp>(op, ctx).is_some()
        || Operation::get_op::<MirStorageDeadOp>(op, ctx).is_some()
        || Operation::get_op::<MirGotoOp>(op, ctx).is_some()
}

fn next_linear_block(
    ctx: &Context,
    block: Ptr<BasicBlock>,
    sequence_started: bool,
) -> Result<Option<Ptr<BasicBlock>>> {
    let Some(terminator) = block.deref(ctx).get_terminator(ctx) else {
        return Ok(None);
    };
    if Operation::get_op::<MirGotoOp>(terminator, ctx).is_none() {
        if !sequence_started {
            return Ok(None);
        }
        return pliron::input_err_noloc!(
            "WGMMA deferred accumulator region crosses non-linear control flow"
        );
    }
    let successors: Vec<_> = terminator.deref(ctx).successors().collect();
    if successors.len() != 1 {
        if !sequence_started {
            return Ok(None);
        }
        return pliron::input_err_noloc!(
            "WGMMA deferred accumulator region requires exactly one successor"
        );
    }
    let successor = successors[0];
    if successor.preds(ctx).len() != 1 {
        if !sequence_started {
            return Ok(None);
        }
        return pliron::input_err_noloc!(
            "WGMMA deferred accumulator region cannot cross a control-flow join"
        );
    }
    Ok(Some(successor))
}

fn collect_pipeline_events(
    ctx: &Context,
    fence: Ptr<Operation>,
) -> Result<Option<Vec<PipelineEvent>>> {
    require_nullary_control_op(ctx, fence, "WGMMA fence")?;

    let mut block = fence
        .deref(ctx)
        .get_parent_block()
        .expect("WGMMA fence must be inside a basic block");
    let mut start_index = block
        .deref(ctx)
        .iter(ctx)
        .position(|operation| operation == fence)
        .expect("WGMMA fence must occur in its parent block")
        + 1;

    let mut events = Vec::new();
    let mut saw_partial_wait = false;

    loop {
        let operations: Vec<_> = block.deref(ctx).iter(ctx).collect();
        for operation in operations.iter().copied().skip(start_index) {
            if is_ignorable(ctx, operation) {
                continue;
            }

            if Operation::get_op::<WgmmaFenceSyncAlignedOp>(operation, ctx).is_some() {
                require_nullary_control_op(ctx, operation, "WGMMA fence")?;
                if !saw_partial_wait {
                    return Ok(None);
                }
                return pliron::input_err_noloc!(
                    "nested WGMMA fences are not supported in one pipelined accumulator region"
                );
            }

            if Operation::get_op::<WgmmaMmaM64N64K16F32Bf16Op>(operation, ctx).is_some() {
                require_pointer_mma_shape(ctx, operation)?;
                let operation_ref = operation.deref(ctx);
                let accumulator = operation_ref.get_operand(0);
                require_supported_accumulator(ctx, accumulator)?;
                events.push(PipelineEvent::Mma {
                    operation,
                    accumulator,
                    desc_a: operation_ref.get_operand(1),
                    desc_b: operation_ref.get_operand(2),
                });
                continue;
            }

            if Operation::get_op::<WgmmaCommitGroupSyncAlignedOp>(operation, ctx).is_some() {
                require_nullary_control_op(ctx, operation, "WGMMA commit_group")?;
                events.push(PipelineEvent::Commit(operation));
                continue;
            }

            if Operation::get_op::<WgmmaWaitGroupSyncAlignedOp>(operation, ctx).is_some() {
                require_wait_shape(ctx, operation)?;
                let wait_operand = operation.deref(ctx).get_operand(0);
                let Some(max_pending) = integer_constant_u64(ctx, wait_operand) else {
                    return pliron::input_err_noloc!(
                        "WGMMA pipelined lowering requires a statically known wait_group<N> immediate"
                    );
                };
                if max_pending > 7 {
                    return pliron::input_err_noloc!(
                        "WGMMA wait_group<N> immediate must be in 0..=7"
                    );
                }

                events.push(PipelineEvent::Wait {
                    operation,
                    max_pending,
                });
                if max_pending == 0 {
                    return if saw_partial_wait {
                        Ok(Some(events))
                    } else {
                        Ok(None)
                    };
                }
                saw_partial_wait = true;
                continue;
            }

            if !saw_partial_wait {
                return Ok(None);
            }
            if block.deref(ctx).get_terminator(ctx) == Some(operation) {
                return pliron::input_err_noloc!(
                    "WGMMA partial-wait pipeline requires a final wait_group<0>"
                );
            }
            return pliron::input_err_noloc!(
                "unsupported operation inside WGMMA pipelined accumulator region: {}",
                Operation::get_opid(operation, ctx)
            );
        }

        let Some(successor) = next_linear_block(ctx, block, saw_partial_wait)? else {
            if saw_partial_wait {
                return pliron::input_err_noloc!(
                    "WGMMA partial-wait pipeline requires a final wait_group<0>"
                );
            }
            return Ok(None);
        };
        block = successor;
        start_index = 0;
    }
}

fn validate_pipeline_events(
    ctx: &Context,
    fence: Ptr<Operation>,
    events: Vec<PipelineEvent>,
) -> Result<PipelinePlan> {
    let mut group_accumulators = Vec::new();
    let mut descriptors = Vec::new();
    let mut mmas = Vec::new();
    let mut commits = Vec::new();
    let mut waits = Vec::new();
    let mut partial_wait_positions = Vec::new();
    let mut partial_wait_value = None;
    let mut index = 0usize;
    let mut saw_final_wait = false;

    while index < events.len() {
        let PipelineEvent::Mma {
            operation: mma,
            accumulator,
            desc_a,
            desc_b,
        } = events[index]
        else {
            return pliron::input_err_noloc!(
                "WGMMA pipelined region requires exactly one MMA before each commit_group"
            );
        };
        index += 1;

        let Some(PipelineEvent::Commit(commit)) = events.get(index).copied() else {
            return pliron::input_err_noloc!(
                "WGMMA pipelined region requires commit_group immediately after each MMA"
            );
        };
        index += 1;

        group_accumulators.push(accumulator);
        descriptors.extend([desc_a, desc_b]);
        mmas.push(mma);
        commits.push(commit);

        if let Some(PipelineEvent::Wait {
            operation,
            max_pending,
        }) = events.get(index).copied()
        {
            if max_pending == 0 {
                waits.push(operation);
                index += 1;
                saw_final_wait = true;
                if index != events.len() {
                    return pliron::input_err_noloc!(
                        "no WGMMA operation may follow the final wait_group<0>"
                    );
                }
                break;
            }

            waits.push(operation);
            index += 1;
            match partial_wait_value {
                Some(expected) if expected != max_pending => {
                    return pliron::input_err_noloc!(
                        "one WGMMA pipeline must use a single statically known wait_group<N> depth"
                    );
                }
                None => partial_wait_value = Some(max_pending),
                _ => {}
            }
            partial_wait_positions.push(group_accumulators.len());

            // The final full drain is allowed immediately after the partial
            // wait for the last committed group. It belongs to the same closed
            // async lifetime, not to a new MMA/commit pair.
            if let Some(PipelineEvent::Wait {
                operation,
                max_pending: 0,
            }) = events.get(index).copied()
            {
                waits.push(operation);
                index += 1;
                saw_final_wait = true;
                if index != events.len() {
                    return pliron::input_err_noloc!(
                        "no WGMMA operation may follow the final wait_group<0>"
                    );
                }
                break;
            }
        }
    }

    if !saw_final_wait {
        return pliron::input_err_noloc!(
            "WGMMA partial-wait pipeline requires a final wait_group<0>"
        );
    }
    let max_pending_groups = partial_wait_value.expect("pipeline collector saw a partial wait");
    let slot_count =
        usize::try_from(max_pending_groups + 1).expect("wait_group depth in 1..=7 must fit usize");
    if group_accumulators.len() < slot_count {
        return pliron::input_err_noloc!(
            "WGMMA wait_group<{}> pipeline requires at least {} committed groups",
            max_pending_groups,
            slot_count
        );
    }

    let expected_wait_positions = (slot_count..=group_accumulators.len()).collect::<Vec<_>>();
    if partial_wait_positions != expected_wait_positions {
        return pliron::input_err_noloc!(
            "WGMMA partial waits must begin after max_pending_groups + 1 commits and occur after every later commit before accumulator-slot reuse"
        );
    }

    let accumulators = group_accumulators[..slot_count].to_vec();
    for left in 0..accumulators.len() {
        if value_accumulator_shape(ctx, accumulators[left]).is_none() {
            return pliron::input_err_noloc!(
                "WGMMA partial-wait pipeline requires canonical [[f32; 8]; 4] accumulator slots"
            );
        }
        for right in (left + 1)..accumulators.len() {
            if accumulators[left] == accumulators[right] {
                return pliron::input_err_noloc!(
                    "WGMMA partial-wait pipeline requires max_pending_groups + 1 distinct accumulator slots"
                );
            }
        }
    }

    for (group_index, accumulator) in group_accumulators.iter().copied().enumerate() {
        let expected = accumulators[group_index % slot_count];
        if accumulator != expected {
            return pliron::input_err_noloc!(
                "WGMMA partial-wait pipeline must reuse accumulator slots in round-robin order only after wait_group<N> makes the slot safe"
            );
        }
    }

    Ok(PipelinePlan {
        fence,
        mmas,
        commits,
        waits,
        accumulators,
        descriptors,
        max_pending_groups: u8::try_from(max_pending_groups)
            .expect("wait_group depth in 1..=7 must fit u8"),
    })
}

fn match_pipeline_sequence(ctx: &Context, fence: Ptr<Operation>) -> Result<Option<PipelinePlan>> {
    let Some(events) = collect_pipeline_events(ctx, fence)? else {
        return Ok(None);
    };
    validate_pipeline_events(ctx, fence, events).map(Some)
}

fn match_sequence(ctx: &Context, fence: Ptr<Operation>) -> Result<Option<FusionPlan>> {
    require_nullary_control_op(ctx, fence, "WGMMA fence")?;

    let mut block = fence
        .deref(ctx)
        .get_parent_block()
        .expect("WGMMA fence must be inside a basic block");
    let mut start_index = block
        .deref(ctx)
        .iter(ctx)
        .position(|operation| operation == fence)
        .expect("WGMMA fence must occur in its parent block")
        + 1;

    let mut mmas = Vec::new();
    let mut commit = None;
    let mut accumulator = None;
    let mut descriptors = Vec::new();

    loop {
        let operations: Vec<_> = block.deref(ctx).iter(ctx).collect();
        for operation in operations.iter().copied().skip(start_index) {
            if is_ignorable(ctx, operation) {
                continue;
            }

            if Operation::get_op::<WgmmaFenceSyncAlignedOp>(operation, ctx).is_some() {
                require_nullary_control_op(ctx, operation, "WGMMA fence")?;
                if mmas.is_empty() {
                    return Ok(None);
                }
                return pliron::input_err_noloc!(
                    "nested WGMMA fences are not supported in one deferred accumulator region"
                );
            }

            if Operation::get_op::<WgmmaMmaM64N64K16F32Bf16Op>(operation, ctx).is_some() {
                require_pointer_mma_shape(ctx, operation)?;
                if commit.is_some() {
                    return pliron::input_err_noloc!(
                        "WGMMA MMA cannot appear after commit_group in a deferred accumulator region"
                    );
                }
                let operation_ref = operation.deref(ctx);
                let current_accumulator = operation_ref.get_operand(0);
                require_supported_accumulator(ctx, current_accumulator)?;
                match accumulator {
                    Some(expected) if expected != current_accumulator => {
                        return pliron::input_err_noloc!(
                            "WGMMA deferred accumulator region uses more than one accumulator"
                        );
                    }
                    None => accumulator = Some(current_accumulator),
                    _ => {}
                }
                descriptors.push(operation_ref.get_operand(1));
                descriptors.push(operation_ref.get_operand(2));
                mmas.push(operation);
                continue;
            }

            if Operation::get_op::<WgmmaCommitGroupSyncAlignedOp>(operation, ctx).is_some() {
                require_nullary_control_op(ctx, operation, "WGMMA commit_group")?;
                if mmas.is_empty() {
                    return Ok(None);
                }
                if commit.replace(operation).is_some() {
                    return pliron::input_err_noloc!(
                        "WGMMA deferred accumulator region supports exactly one commit_group"
                    );
                }
                continue;
            }

            if Operation::get_op::<WgmmaWaitGroupSyncAlignedOp>(operation, ctx).is_some() {
                require_wait_shape(ctx, operation)?;
                if mmas.is_empty() {
                    return Ok(None);
                }
                let Some(commit) = commit else {
                    return pliron::input_err_noloc!(
                        "WGMMA wait_group requires a preceding commit_group"
                    );
                };
                let wait_operand = operation.deref(ctx).get_operand(0);
                if integer_constant_u64(ctx, wait_operand) != Some(0) {
                    return pliron::input_err_noloc!(
                        "WGMMA deferred accumulator lowering requires wait_group<0>"
                    );
                }
                return Ok(Some(FusionPlan {
                    fence,
                    mmas,
                    commit,
                    wait: operation,
                    accumulator: accumulator.expect("MMA list is non-empty"),
                    descriptors,
                }));
            }

            if mmas.is_empty() {
                return Ok(None);
            }
            return pliron::input_err_noloc!(
                "unsupported operation inside WGMMA deferred accumulator region: {}",
                Operation::get_opid(operation, ctx)
            );
        }

        let Some(successor) = next_linear_block(ctx, block, !mmas.is_empty())? else {
            if mmas.is_empty() {
                return Ok(None);
            }
            return pliron::input_err_noloc!(
                "WGMMA deferred accumulator region ended before wait_group<0>"
            );
        };
        block = successor;
        start_index = 0;
    }
}

fn apply_pipeline_plan(ctx: &mut Context, plan: PipelinePlan) {
    let PipelinePlan {
        fence,
        mmas,
        commits,
        waits,
        accumulators,
        descriptors,
        max_pending_groups,
    } = plan;

    let final_wait = *waits
        .last()
        .expect("pipeline plan has a final wait_group<0>");
    let loc = fence.deref(ctx).loc();
    let mut slot_types = Vec::with_capacity(accumulators.len());
    let mut all_accumulator_values = Vec::with_capacity(accumulators.len() * ACCUMULATOR_LEN);

    for accumulator in accumulators.iter().copied() {
        let (row_type, element_type) = value_accumulator_shape(ctx, accumulator)
            .expect("pipeline validation requires canonical accumulator slots");
        let (_element_pointers, accumulator_values) = load_accumulator_values_before(
            ctx,
            final_wait,
            accumulator,
            row_type,
            element_type,
            loc.clone(),
        );
        slot_types.push((row_type, element_type));
        all_accumulator_values.extend(accumulator_values);
    }

    let pipeline = WgmmaMmaPipelineValuesM64N64K16F32Bf16Op::build(
        ctx,
        all_accumulator_values,
        descriptors,
        max_pending_groups,
    );
    pipeline.deref_mut(ctx).set_loc(loc.clone());
    let result_count = accumulators.len() * ACCUMULATOR_LEN;
    let accumulator_results = (0..result_count)
        .map(|index| pipeline.deref(ctx).get_result(index))
        .collect::<Vec<_>>();
    pipeline.insert_before(ctx, final_wait);

    for (slot, accumulator) in accumulators.iter().copied().enumerate() {
        let begin = slot * ACCUMULATOR_LEN;
        let end = begin + ACCUMULATOR_LEN;
        let (row_type, element_type) = slot_types[slot];
        store_canonical_accumulator_values_before(
            ctx,
            final_wait,
            accumulator,
            row_type,
            element_type,
            &accumulator_results[begin..end],
            loc.clone(),
        );
    }

    let mut rewriter = IRRewriter::<Recorder>::default();
    rewriter.erase_operation(ctx, fence);
    for mma in mmas {
        rewriter.erase_operation(ctx, mma);
    }
    for commit in commits {
        rewriter.erase_operation(ctx, commit);
    }
    for wait in waits {
        rewriter.erase_operation(ctx, wait);
    }
}

fn apply_plan(ctx: &mut Context, plan: FusionPlan) {
    let FusionPlan {
        fence,
        mmas,
        commit,
        wait,
        accumulator,
        descriptors,
    } = plan;

    if let Some((row_type, element_type)) = value_accumulator_shape(ctx, accumulator) {
        apply_value_plan(
            ctx,
            fence,
            mmas,
            commit,
            wait,
            accumulator,
            descriptors,
            row_type,
            element_type,
        );
    } else {
        apply_pointer_fallback(ctx, fence, mmas, commit, wait, accumulator, descriptors);
    }
}

/// Return whether every block in `region` is reachable from its entry.
///
/// Pliron's dominator tree intentionally contains only reachable blocks.  Some
/// malformed/negative-test MIR regions can still contain detached blocks, and
/// asking `LoopInfo` to inspect such a region would query dominance for blocks
/// absent from that tree.  Counted-loop fusion is an optimization, so fail
/// closed here and leave those regions to the existing straight-line WGMMA
/// validation path.
fn region_is_fully_reachable(ctx: &Context, region: Ptr<Region>) -> bool {
    let Some(entry) = region.deref(ctx).get_head() else {
        return true;
    };

    let all_blocks: FxHashSet<_> = region.deref(ctx).iter(ctx).collect();
    let mut reachable = FxHashSet::default();
    let mut worklist = vec![entry];

    while let Some(block) = worklist.pop() {
        if !reachable.insert(block) {
            continue;
        }

        let Some(terminator) = block.deref(ctx).get_terminator(ctx) else {
            continue;
        };

        for successor in terminator.deref(ctx).successors() {
            if all_blocks.contains(&successor) && !reachable.contains(&successor) {
                worklist.push(successor);
            }
        }
    }

    reachable.len() == all_blocks.len()
}

/// Adapt every supported pointer-form BF16 WGMMA sequence in `module_op`.
///
/// Canonical counted loops are handled first because their fence and final wait
/// live in different CFG blocks. Each successful loop rewrite bypasses the old
/// loop and immediately removes the now-unreachable loop blocks so no stale
/// pointer-form MMA reaches final lowering. Remaining straight-line regions
/// first try the statically scheduled partial-wait pipeline and then fall back
/// to the existing single-group linear adapter.
pub(crate) fn fuse_deferred_accumulators(
    ctx: &mut Context,
    module_op: Ptr<Operation>,
) -> Result<()> {
    loop {
        let mut counted_plan = None;

        'functions: for function in collect_functions(ctx, module_op) {
            let region = function.deref(ctx).get_region(0);

            // `DomInfo` does not assign dominance nodes to unreachable blocks.
            // Skip loop analysis for such regions so malformed MIR continues to
            // receive the established straight-line WGMMA diagnostics instead
            // of panicking inside dominance lookup.
            if !region_is_fully_reachable(ctx, region) {
                continue;
            }

            let mut dom_info = DomInfo::default();
            let info = {
                let dom_tree = dom_info.get_dom_tree(ctx, region);
                LoopInfo::compute(ctx, region, dom_tree)
            };

            for loop_id in 0..info.loops().len() {
                if let Some(plan) = match_counted_loop(ctx, &info, region, loop_id)? {
                    counted_plan = Some((function, plan));
                    break 'functions;
                }
            }
        }

        let Some((function, plan)) = counted_plan else {
            break;
        };

        apply_counted_loop_plan(ctx, plan);
        let mut rewriter = IRRewriter::<Recorder>::default();
        remove_blocks_inside_op(function, ctx, &mut rewriter);
    }

    let fences: Vec<_> = collect_blocks(ctx, module_op)
        .into_iter()
        .flat_map(|block| block.deref(ctx).iter(ctx).collect::<Vec<_>>())
        .filter(|operation| Operation::get_op::<WgmmaFenceSyncAlignedOp>(*operation, ctx).is_some())
        .collect();

    for fence in fences {
        if let Some(plan) = match_pipeline_sequence(ctx, fence)? {
            apply_pipeline_plan(ctx, plan);
            continue;
        }
        if let Some(plan) = match_sequence(ctx, fence)? {
            apply_plan(ctx, plan);
        }
    }
    Ok(())
}
