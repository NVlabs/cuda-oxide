/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Fuse sound BF16 WGMMA sequences before MIR-to-LLVM conversion.
//!
//! The public MMA operation exposes its accumulator through a pointer, but PTX
//! requires all 32 accumulator registers to remain inaccessible until the
//! corresponding `wgmma.wait_group` completes. This pass recognizes a closed,
//! straight-line sequence and replaces it with one deferred group operation.
//!
//! The accepted initial form is deliberately narrow:
//!
//! ```text
//! wgmma.fence
//! one or more m64n64k16.f32.bf16.bf16 MMA operations on one accumulator
//! wgmma.commit_group
//! wgmma.wait_group<0>
//! ```
//!
//! Compiler-generated integer constants, storage markers, and unconditional
//! gotos may separate those operations. All other operations, branches, joins,
//! partial waits, and accumulator changes are rejected. A sequence that
//! crosses a loop back-edge is rejected through the nested-fence or
//! control-flow-join checks; a complete sequence inside a loop body fuses.

use dialect_mir::{
    ops::{MirConstantOp, MirGotoOp, MirStorageDeadOp, MirStorageLiveOp},
    types::{MirPtrType, address_space},
};
use dialect_nvvm::ops::{
    WgmmaCommitGroupSyncAlignedOp, WgmmaFenceSyncAlignedOp, WgmmaMmaGroupM64N64K16F32Bf16Op,
    WgmmaMmaM64N64K16F32Bf16Op, WgmmaWaitGroupSyncAlignedOp,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::IntegerAttr,
        ops::ConstantOp,
        types::{IntegerType, Signedness},
    },
    context::{Context, Ptr},
    irbuild::{
        listener::Recorder,
        rewriter::{IRRewriter, Rewriter},
    },
    linked_list::ContainsLinkedList,
    location::Located,
    operation::Operation,
    result::Result,
    r#type::Typed,
    value::Value,
};

struct FusionPlan {
    fence: Ptr<Operation>,
    mmas: Vec<Ptr<Operation>>,
    commit: Ptr<Operation>,
    wait: Ptr<Operation>,
    accumulator: Value,
    descriptors: Vec<Value>,
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

fn apply_plan(ctx: &mut Context, plan: FusionPlan) {
    let group = WgmmaMmaGroupM64N64K16F32Bf16Op::build(ctx, plan.accumulator, plan.descriptors);
    group.deref_mut(ctx).set_loc(plan.fence.deref(ctx).loc());
    group.insert_before(ctx, plan.wait);

    let mut rewriter = IRRewriter::<Recorder>::default();
    rewriter.erase_operation(ctx, plan.fence);
    for mma in plan.mmas {
        rewriter.erase_operation(ctx, mma);
    }
    rewriter.erase_operation(ctx, plan.commit);
    rewriter.erase_operation(ctx, plan.wait);
}

/// Fuse every supported pointer-form BF16 WGMMA sequence in `module_op`.
pub(crate) fn fuse_deferred_accumulators(
    ctx: &mut Context,
    module_op: Ptr<Operation>,
) -> Result<()> {
    let fences: Vec<_> = collect_blocks(ctx, module_op)
        .into_iter()
        .flat_map(|block| block.deref(ctx).iter(ctx).collect::<Vec<_>>())
        .filter(|operation| Operation::get_op::<WgmmaFenceSyncAlignedOp>(*operation, ctx).is_some())
        .collect();

    for fence in fences {
        if let Some(plan) = match_sequence(ctx, fence)? {
            apply_plan(ctx, plan);
        }
    }
    Ok(())
}
