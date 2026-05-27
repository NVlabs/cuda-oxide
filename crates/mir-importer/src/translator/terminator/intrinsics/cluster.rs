/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thread Block Cluster intrinsics (sm_90+ Hopper).
//!
//! Handles translation of cluster operations:
//! - `cluster_ctaidX/Y/Z()` - Block position within cluster
//! - `cluster_nctaidX/Y/Z()` - Cluster dimensions
//! - `cluster_idx()` - Cluster's linear index in grid
//! - `num_clusters()` - Total clusters in grid
//! - `cluster_sync()` - Cluster-wide barrier
//! - `map_shared_rank()` - Distributed shared memory address mapping

use super::super::helpers::{emit_goto, emit_store_result_and_goto};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{
    ClusterSyncOp, DsmemReadU32Op, MapaSharedClusterOp, ReadPtxSregClusterCtaidXOp,
    ReadPtxSregClusterCtaidYOp, ReadPtxSregClusterCtaidZOp, ReadPtxSregClusterIdxOp,
    ReadPtxSregClusterNctaidXOp, ReadPtxSregClusterNctaidYOp, ReadPtxSregClusterNctaidZOp,
    ReadPtxSregNclusterIdOp,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::Typed;
use rustc_public::mir;
// =============================================================================
// Block Position Within Cluster
// =============================================================================

/// Emit `cluster_ctaidX()`: Get block's X position within cluster.
///
/// Returns u32 in range `[0, cluster_nctaidX)`.
pub fn emit_cluster_ctaid_x(
    ctx: &mut Context,
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let u32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let op = Operation::new(
        ctx,
        ReadPtxSregClusterCtaidXOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    let op = if let Some(prev) = prev_op {
        op.insert_after(ctx, prev);
        op
    } else {
        op.insert_at_front(block_ptr, ctx);
        op
    };

    let result_value = op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        op,
        value_map,
        block_map,
        loc,
        "cluster_ctaidX call without target block",
    )
}

/// Emit `cluster_ctaidY()`: Get block's Y position within cluster.
pub fn emit_cluster_ctaid_y(
    ctx: &mut Context,
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let u32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let op = Operation::new(
        ctx,
        ReadPtxSregClusterCtaidYOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    let op = if let Some(prev) = prev_op {
        op.insert_after(ctx, prev);
        op
    } else {
        op.insert_at_front(block_ptr, ctx);
        op
    };

    let result_value = op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        op,
        value_map,
        block_map,
        loc,
        "cluster_ctaidY call without target block",
    )
}

/// Emit `cluster_ctaidZ()`: Get block's Z position within cluster.
pub fn emit_cluster_ctaid_z(
    ctx: &mut Context,
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let u32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let op = Operation::new(
        ctx,
        ReadPtxSregClusterCtaidZOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    let op = if let Some(prev) = prev_op {
        op.insert_after(ctx, prev);
        op
    } else {
        op.insert_at_front(block_ptr, ctx);
        op
    };

    let result_value = op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        op,
        value_map,
        block_map,
        loc,
        "cluster_ctaidZ call without target block",
    )
}

// =============================================================================
// Cluster Dimensions
// =============================================================================

/// Emit `cluster_nctaidX()`: Get cluster X dimension.
pub fn emit_cluster_nctaid_x(
    ctx: &mut Context,
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let u32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let op = Operation::new(
        ctx,
        ReadPtxSregClusterNctaidXOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    let op = if let Some(prev) = prev_op {
        op.insert_after(ctx, prev);
        op
    } else {
        op.insert_at_front(block_ptr, ctx);
        op
    };

    let result_value = op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        op,
        value_map,
        block_map,
        loc,
        "cluster_nctaidX call without target block",
    )
}

/// Emit `cluster_nctaidY()`: Get cluster Y dimension.
pub fn emit_cluster_nctaid_y(
    ctx: &mut Context,
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let u32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let op = Operation::new(
        ctx,
        ReadPtxSregClusterNctaidYOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    let op = if let Some(prev) = prev_op {
        op.insert_after(ctx, prev);
        op
    } else {
        op.insert_at_front(block_ptr, ctx);
        op
    };

    let result_value = op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        op,
        value_map,
        block_map,
        loc,
        "cluster_nctaidY call without target block",
    )
}

/// Emit `cluster_nctaidZ()`: Get cluster Z dimension.
pub fn emit_cluster_nctaid_z(
    ctx: &mut Context,
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let u32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let op = Operation::new(
        ctx,
        ReadPtxSregClusterNctaidZOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    let op = if let Some(prev) = prev_op {
        op.insert_after(ctx, prev);
        op
    } else {
        op.insert_at_front(block_ptr, ctx);
        op
    };

    let result_value = op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        op,
        value_map,
        block_map,
        loc,
        "cluster_nctaidZ call without target block",
    )
}

// =============================================================================
// Cluster Grid Position
// =============================================================================

/// Emit `cluster_idx()`: Get cluster's linear index within grid.
pub fn emit_cluster_idx(
    ctx: &mut Context,
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let u32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let op = Operation::new(
        ctx,
        ReadPtxSregClusterIdxOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    let op = if let Some(prev) = prev_op {
        op.insert_after(ctx, prev);
        op
    } else {
        op.insert_at_front(block_ptr, ctx);
        op
    };

    let result_value = op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        op,
        value_map,
        block_map,
        loc,
        "cluster_idx call without target block",
    )
}

/// Emit `num_clusters()`: Get total number of clusters in grid.
pub fn emit_num_clusters(
    ctx: &mut Context,
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let u32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let op = Operation::new(
        ctx,
        ReadPtxSregNclusterIdOp::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    let op = if let Some(prev) = prev_op {
        op.insert_after(ctx, prev);
        op
    } else {
        op.insert_at_front(block_ptr, ctx);
        op
    };

    let result_value = op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        op,
        value_map,
        block_map,
        loc,
        "num_clusters call without target block",
    )
}

// =============================================================================
// Cluster Synchronization
// =============================================================================

/// Emit `cluster_sync()`: Cluster-wide barrier synchronization.
///
/// All threads in all blocks of the cluster must reach this barrier.
pub fn emit_cluster_sync(
    ctx: &mut Context,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let op = Operation::new(
        ctx,
        ClusterSyncOp::get_concrete_op_info(),
        vec![], // No results
        vec![], // No operands
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    let op = if let Some(prev) = prev_op {
        op.insert_after(ctx, prev);
        op
    } else {
        op.insert_at_front(block_ptr, ctx);
        op
    };

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("cluster_sync call without target block".to_string())
        )
    }
}

// =============================================================================
// Distributed Shared Memory
// =============================================================================

/// Emit `map_shared_rank(ptr, rank)`: Map shared memory to another block's address space.
///
/// Args:
/// - args[0]: *const T - Local shared memory pointer
/// - args[1]: u32 - Target block's rank within cluster
///
/// Returns: *const T - Pointer to same offset in target block's shared memory
pub fn emit_map_shared_rank(
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
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "map_shared_rank expects 2 arguments (ptr, rank), got {}",
                args.len()
            ))
        );
    }

    // Get the source pointer (arg 0)
    let (src_ptr, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Get the target rank (arg 1)
    let (rank, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // Result type: same pointer type as input
    let result_type = src_ptr.get_type(ctx);

    let op = Operation::new(
        ctx,
        MapaSharedClusterOp::get_concrete_op_info(),
        vec![result_type],   // Result: mapped pointer
        vec![src_ptr, rank], // Operands: ptr, rank
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }

    let result_value = op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        op,
        value_map,
        block_map,
        loc,
        "map_shared_rank call without target block",
    )
}

/// Emit `dsmem_read_u32(ptr, rank)`: Read u32 from another block's shared memory.
///
/// Combines mapa.shared::cluster + ld.shared::cluster.u32 into one operation.
///
/// Args:
/// - args[0]: *const u32 - Local shared memory pointer
/// - args[1]: u32 - Target block's rank within cluster
///
/// Returns: u32 - Value read from the target block's shared memory
pub fn emit_dsmem_read_u32(
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
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "dsmem_read_u32 expects 2 arguments (ptr, rank), got {}",
                args.len()
            ))
        );
    }

    let (src_ptr, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let (rank, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    let u32_type = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let op = Operation::new(
        ctx,
        DsmemReadU32Op::get_concrete_op_info(),
        vec![u32_type.to_ptr()],
        vec![src_ptr, rank],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }

    let result_value = op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        op,
        value_map,
        block_map,
        loc,
        "dsmem_read_u32 call without target block",
    )
}
