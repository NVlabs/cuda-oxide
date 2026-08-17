/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Expansion of dialect-cute ops into dialect-mir / LLVM-dialect-typed ops:
//! the analogue of the internal CuTe IR's `cute-expand-ops`.
//!
//! Runs in the device pipeline right after MIR preparation (mem2reg/unroll)
//! and BEFORE backend selection scans the module for generated-intrinsic
//! requirements. After this pass no cute op remains and the existing
//! `lower_to_llvm` path proceeds unchanged.
//!
//! ## `cute.copy` expansion
//!
//! A contiguous identity-map tile of `n` elements (layout `(n):(1)`) becomes
//! one vectorized transfer through `<n x elem>`-typed pointers:
//!
//! ```text
//! cute.copy(%src, %dst) {layout = "(4):(1)", elem = i32}
//!   ==>
//! %sv = mir.cast<PtrToPtr> %src : mir.ptr<vector<4 x i32>>
//! %v  = mir.load %sv           : vector<4 x i32>     // align 16 (natural)
//! %dv = mir.cast<PtrToPtr> %dst : mir.ptr<vector<4 x i32>>
//! mir.store %dv, %v
//! ```
//!
//! The LLVM exporter prints the load/store with the vector type's natural
//! (power-of-2 total-width) alignment, and llc's LoadStoreVectorizer turns
//! the aligned access into a single 128-bit PTX transaction
//! (`ld.global.v2.b64` spelling on llc-21/22). The alignment is sound by the
//! `cute.copy` contract: tiles are naturally aligned, like CUDA's `float4`.
//!
//! Everything here evaluates the layout via `cute-layout` (`crate::layout`);
//! no layout math is re-derived from scalar IR.

use std::num::NonZeroUsize;

use cute_layout::{ComposedLayout, IntTuple, Swizzle, validate_cooperative_copy_plan};
use dialect_mir::attributes::{FieldIndexAttr, MirCastKindAttr};
use dialect_mir::ops::{
    MirAddOp, MirBitAndOp, MirBitOrOp, MirBitXorOp, MirCallOp, MirCastOp, MirCondBranchOp,
    MirConstantOp, MirConstructArrayOp, MirConstructStructOp, MirConstructTupleOp, MirDivOp,
    MirEqOp, MirExtractFieldOp, MirGeOp, MirGotoOp, MirLoadOp, MirLtOp, MirMulOp, MirNeOp,
    MirPtrOffsetOp, MirRemOp, MirShlOp, MirShrOp, MirStoreOp, MirSubOp,
};
use dialect_mir::types::{
    MirArrayType, MirFP16Type, MirPtrType, MirStructType, MirTupleType, address_space,
};
use dialect_nvvm::ops::{
    Barrier0Op, BarrierCtaSyncAlignedCountOp, CpAsyncBulkCommitGroupOp, CpAsyncBulkWaitGroupReadOp,
    CpAsyncCaZfill4Op, CpAsyncCaZfill8Op, CpAsyncCaZfill16Op, CvtF16x2F32Op, CvtRnBf16x2Ue8m0x2Op,
    CvtRnF16x2E2m1x2Op, FenceMbarrierInitReleaseClusterOp, FenceProxyAsyncSharedCtaOp,
    MbarrierArriveExpectTxSharedOp, MbarrierArriveSharedOp, MbarrierInitSharedOp,
    MbarrierTryWaitParitySharedOp, ReadPtxSregCtaidXOp, ReadPtxSregLaneIdOp, ReadPtxSregNctaidXOp,
    ReadPtxSregTidXOp, StmatrixM8n8X2Op,
};
use llvm_export::types as llvm_types;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::{IntegerAttr, StringAttr};
use pliron::builtin::op_interfaces::OperandSegmentInterface;
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::common_traits::Verify;
use pliron::context::{Context, Ptr};
use pliron::irbuild::listener::Recorder;
use pliron::irbuild::rewriter::{IRRewriter, Rewriter};
use pliron::linked_list::{ContainsLinkedList, LinkedList};
use pliron::location::{Located, Location};
use pliron::op::{Op, OpId};
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, Typed};
use pliron::utils::apint::APInt;
use pliron::value::Value;

use crate::attributes::{
    CuteEpilogueSyncPhaseAttr, CutePipelineRoleAttr, CutePipelineStateAttr, CuteTensorRoleAttr,
    CuteTileGridAttr,
};
use crate::epilogue_ops::{
    CuteEpilogueHalfOp, CuteEpilogueSmemOverlayOp, CuteEpilogueStoreFragmentOp, CuteEpilogueSyncOp,
    CuteEpilogueWarpSliceOp, CuteTmaStore2dSemanticOp, CuteTmaStoreAcquireOp, CuteTmaStoreCommitOp,
    CuteTmaStoreTailOp,
};
use crate::gemm_tma_ops::{CuteTmaCopy2dOp, CuteTmaGmemViewOp, CuteTmaSmemViewOp};
use crate::gemv_ops::{
    CuteDotOp, CuteScaledViewKTileOp, CuteScaledViewLoadOp, CuteScaledViewMakeOp,
    CuteScaledViewRowOp, CuteTensorMake2DOp,
};
use crate::ops::{
    CuteAssumeDivOp, CuteCopyG2SOp, CuteCopyOp, CuteLdmatrixOp, CuteTmaLoad2dOp, CuteTmaStore2dOp,
};
use crate::pipeline_ops::{
    CutePipelineConsumerReleaseOp, CutePipelineConsumerWaitOp, CutePipelineProducerAcquireOp,
    CutePipelineProducerExpectTxOp, CutePipelineProducerTailOp, CutePipelineStateAdvanceOp,
    CutePipelineStateNewOp, CutePipelineStateSlotOp, CuteTmaLoadPipelineInitOp,
    CuteTmaLoadPipelineMakeOp,
};
use crate::scheduler_ops::{
    CuteSchedulerAdvanceOp, CuteSchedulerCurrentOp, CuteSchedulerHasWorkOp, CuteSchedulerNew1dOp,
    CuteWorkTileCoordinatesOp,
};
use crate::smem_mma_ops::{
    CuteFragmentFillOp, CuteFragmentSliceKOp, CuteMmaLoadAOp, CuteMmaLoadScalesOp,
    CuteMmaPartitionBOp, CuteSmemTensorOverlayOp, CuteTiledGemmOp, CuteTiledMmaSliceOp,
};
use crate::tensor_ops::{
    CuteTensorBaseOp, CuteTensorIsFullOp, CuteTensorLoadIntoOp, CuteTensorMakeOp,
    CuteTensorSliceOp, CuteTensorStoreElementAbsOp, CuteTensorStoreFromOp,
    CuteTensorZippedDivideOp,
};
use crate::types::{CuteSmemTensorType, CuteTensorViewType, CuteTmaViewType};
use crate::verify::{VerifyError, verify_cute_semantics};

/// Temporary tag for a generated NVVM operation emitted by this pass.
///
/// dialect-cute cannot depend on cuda-oxide-codegen's generated catalog
/// without creating a dependency cycle. Instead it places this unit tag:
///
/// ```text
/// cute expansion ── tag ──> codegen catalog resolves exact ABI marker
/// ```
///
/// Codegen removes the tag before its required-marker scan. A missing,
/// ambiguous, or structurally incompatible catalog entry is a hard error.
pub const GENERATED_INTRINSIC_REQUEST_ATTR: &str = "cute_generated_intrinsic_request";

/// Mark an NVVM operation whose exact generated ABI marker must be resolved
/// by the codegen catalog immediately after this expansion pass.
pub fn request_generated_intrinsic_marker(ctx: &mut Context, op: Ptr<Operation>) {
    use pliron::builtin::attributes::UnitAttr;
    use pliron::identifier::Identifier;

    op.deref_mut(ctx).attributes.set(
        Identifier::try_from(GENERATED_INTRINSIC_REQUEST_ATTR)
            .expect("cute generated-marker request key must be a valid identifier"),
        UnitAttr,
    );
}

#[derive(Debug, thiserror::Error)]
pub enum ExpandError {
    #[error(transparent)]
    Semantic(#[from] VerifyError),
    #[error("cute op expansion is not implemented yet: {0}")]
    NotImplemented(String),
    #[error("invalid cute op: {0}")]
    Invalid(String),
}

/// The runtime facts hidden behind one ghost tensor-view value.
///
/// `zipped_divide` changes only `tile_size`; `slice` adds `tile_index`.
/// Keeping this as analysis data means no descriptor struct reaches LLVM.
#[derive(Clone, Copy)]
struct TensorViewState {
    data: Value,
    len: Value,
    storage: TypeHandle,
    tile_size: Option<u64>,
    tile_index: Option<Value>,
}

/// Read the view's producer chain without guessing from pointer arithmetic.
fn resolve_tensor_view(
    ctx: &Context,
    value: Value,
    depth: usize,
) -> Result<TensorViewState, ExpandError> {
    if depth > 16 {
        return Err(ExpandError::Invalid(
            "cute tensor-view producer chain is unexpectedly deep".into(),
        ));
    }
    let Some(defining_op) = value.defining_op() else {
        return Err(ExpandError::Invalid(
            "cute tensor view reached a block argument; v0 needs a direct make/zipped/slice chain"
                .into(),
        ));
    };
    let opid = Operation::get_opid(defining_op, ctx);
    if opid == CuteTensorMakeOp::get_opid_static() {
        let op = defining_op.deref(ctx);
        let result_ty = value.get_type(ctx);
        let result_ty_ref = result_ty.deref(ctx);
        let view = result_ty_ref
            .downcast_ref::<CuteTensorViewType>()
            .ok_or_else(|| {
                ExpandError::Invalid("cute.tensor_make result is not a tensor view".into())
            })?;
        return Ok(TensorViewState {
            data: op.get_operand(0),
            len: op.get_operand(1),
            storage: view.storage,
            tile_size: None,
            tile_index: None,
        });
    }
    if opid == CuteTensorZippedDivideOp::get_opid_static() {
        let op = defining_op.deref(ctx);
        let mut state = resolve_tensor_view(ctx, op.get_operand(0), depth + 1)?;
        let result_ty = value.get_type(ctx);
        let result_ty_ref = result_ty.deref(ctx);
        let view = result_ty_ref
            .downcast_ref::<CuteTensorViewType>()
            .ok_or_else(|| {
                ExpandError::Invalid("cute.tensor_zipped_divide result is not a tensor view".into())
            })?;
        state.tile_size = view.layout.tile_size();
        return Ok(state);
    }
    if opid == CuteTensorSliceOp::get_opid_static() {
        let op = defining_op.deref(ctx);
        let mut state = resolve_tensor_view(ctx, op.get_operand(0), depth + 1)?;
        let result_ty = value.get_type(ctx);
        let result_ty_ref = result_ty.deref(ctx);
        let view = result_ty_ref
            .downcast_ref::<CuteTensorViewType>()
            .ok_or_else(|| {
                ExpandError::Invalid("cute.tensor_slice result is not a tensor view".into())
            })?;
        state.tile_size = view.selected_tile_size();
        state.tile_index = Some(op.get_operand(1));
        return Ok(state);
    }
    Err(ExpandError::Invalid(format!(
        "tensor view is produced by unsupported operation `{opid}`"
    )))
}

fn tensor_tile_state(ctx: &Context, value: Value) -> Result<TensorViewState, ExpandError> {
    let state = resolve_tensor_view(ctx, value, 0)?;
    if state.tile_size.is_none() || state.tile_index.is_none() {
        return Err(ExpandError::Invalid(
            "tensor operation needs a selected tile view".into(),
        ));
    }
    Ok(state)
}

/// Emit Rust's `tile_index.saturating_mul(tile_size)` before a consumer.
///
/// The overflow test is written as `limit < tile_index`, where
/// `limit = u64::MAX / tile_size`. Turning that boolean into either an all-zero
/// or all-one mask gives a branch-free choice:
///
/// ```text
/// no overflow: wrapping_product | 0        = wrapping_product
/// overflow:    wrapping_product | u64::MAX = u64::MAX
/// ```
fn emit_tensor_tile_base(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    state: TensorViewState,
) -> Value {
    let tile_size = state
        .tile_size
        .expect("tensor preflight must resolve a selected tile width");
    let tile_index = state
        .tile_index
        .expect("tensor preflight must resolve a selected tile index");
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let bool_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
    let width = emit_u64_const_bits(ctx, anchor, loc, tile_size);
    let product = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        tile_index,
        width,
    );
    let limit = emit_u64_const_bits(ctx, anchor, loc, u64::MAX / tile_size);
    let overflow = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirLtOp::get_concrete_op_info(),
        bool_ty,
        limit,
        tile_index,
    );
    let overflow_u64_op = emit_op_before(
        ctx,
        anchor,
        loc,
        MirCastOp::get_concrete_op_info(),
        vec![u64_ty],
        vec![overflow],
    );
    MirCastOp::new(overflow_u64_op).set_attr_cast_kind(ctx, MirCastKindAttr::IntToInt);
    let overflow_u64 = overflow_u64_op.deref(ctx).get_result(0);
    let zero = emit_u64_const_bits(ctx, anchor, loc, 0);
    let overflow_mask = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirSubOp::get_concrete_op_info(),
        u64_ty,
        zero,
        overflow_u64,
    );
    emit_bin_op(
        ctx,
        anchor,
        loc,
        MirBitOrOp::get_concrete_op_info(),
        u64_ty,
        product,
        overflow_mask,
    )
}

fn emit_tensor_element_pointer(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    state: TensorViewState,
    index: Value,
) -> Value {
    let pointer_ty = state.data.get_type(ctx);
    let offset = emit_op_before(
        ctx,
        anchor,
        loc,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![pointer_ty],
        vec![state.data, index],
    );
    offset.deref(ctx).get_result(0)
}

#[derive(Clone, Copy)]
struct SchedulerHasWorkPlan {
    op: Ptr<Operation>,
    current: Value,
    total_tiles: u64,
}

#[derive(Clone, Copy)]
struct SchedulerCoordinatesPlan {
    op: Ptr<Operation>,
    current: Value,
    grid: CuteTileGridAttr,
}

#[derive(Clone, Copy)]
struct SchedulerAdvancePlan {
    op: Ptr<Operation>,
    current: Value,
    stride: Value,
}

fn pipeline_semantic_ids() -> [OpId; 10] {
    [
        CuteTmaLoadPipelineMakeOp::get_opid_static(),
        CuteTmaLoadPipelineInitOp::get_opid_static(),
        CutePipelineStateNewOp::get_opid_static(),
        CutePipelineStateSlotOp::get_opid_static(),
        CutePipelineStateAdvanceOp::get_opid_static(),
        CutePipelineProducerAcquireOp::get_opid_static(),
        CutePipelineProducerExpectTxOp::get_opid_static(),
        CutePipelineConsumerWaitOp::get_opid_static(),
        CutePipelineConsumerReleaseOp::get_opid_static(),
        CutePipelineProducerTailOp::get_opid_static(),
    ]
}

fn smem_mma_semantic_ids() -> [OpId; 8] {
    [
        CuteSmemTensorOverlayOp::get_opid_static(),
        CuteTiledMmaSliceOp::get_opid_static(),
        CuteFragmentFillOp::get_opid_static(),
        CuteMmaLoadScalesOp::get_opid_static(),
        CuteFragmentSliceKOp::get_opid_static(),
        CuteMmaLoadAOp::get_opid_static(),
        CuteMmaPartitionBOp::get_opid_static(),
        CuteTiledGemmOp::get_opid_static(),
    ]
}

fn epilogue_semantic_ids() -> [OpId; 9] {
    [
        CuteEpilogueSmemOverlayOp::get_opid_static(),
        CuteEpilogueWarpSliceOp::get_opid_static(),
        CuteEpilogueStoreFragmentOp::get_opid_static(),
        CuteEpilogueSyncOp::get_opid_static(),
        CuteEpilogueHalfOp::get_opid_static(),
        CuteTmaStoreAcquireOp::get_opid_static(),
        CuteTmaStoreCommitOp::get_opid_static(),
        CuteTmaStoreTailOp::get_opid_static(),
        CuteTmaStore2dSemanticOp::get_opid_static(),
    ]
}

fn preflight_scheduler_coordinates(
    ctx: &Context,
    op: Ptr<Operation>,
) -> Result<SchedulerCoordinatesPlan, ExpandError> {
    let coordinates = CuteWorkTileCoordinatesOp::wrap(op);
    let producer = coordinates
        .work_tile(ctx)
        .defining_op()
        .expect("shared verifier requires a direct scheduler-current producer");
    debug_assert!(Operation::get_opid(producer, ctx) == CuteSchedulerCurrentOp::get_opid_static());
    let current = CuteSchedulerCurrentOp::wrap(producer);
    let grid = current
        .tile_grid(ctx)
        .expect("shared verifier requires a scheduler tile grid");
    Ok(SchedulerCoordinatesPlan {
        op,
        current: current.current(ctx),
        grid,
    })
}

fn emit_scheduler_special_u64(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    info: OpInfoPair,
) -> Value {
    let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let read = emit_op_before(ctx, anchor, loc, info, vec![u32_ty], vec![]);
    request_generated_intrinsic_marker(ctx, read);
    let value = read.deref(ctx).get_result(0);
    emit_cast_value(ctx, anchor, loc, value, u64_ty, MirCastKindAttr::IntToInt)
}

fn emit_scheduler_coordinates(ctx: &mut Context, plan: SchedulerCoordinatesPlan) -> [Value; 4] {
    let loc = plan.op.deref(ctx).loc().clone();
    let ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let m_width = emit_u64_const_bits(ctx, plan.op, &loc, plan.grid.m_tiles);
    let n_batch = emit_bin_op(
        ctx,
        plan.op,
        &loc,
        MirDivOp::get_concrete_op_info(),
        ty,
        plan.current,
        m_width,
    );
    let m_width = emit_u64_const_bits(ctx, plan.op, &loc, plan.grid.m_tiles);
    let m_base = emit_bin_op(
        ctx,
        plan.op,
        &loc,
        MirMulOp::get_concrete_op_info(),
        ty,
        n_batch,
        m_width,
    );
    let m_tile = emit_bin_op(
        ctx,
        plan.op,
        &loc,
        MirSubOp::get_concrete_op_info(),
        ty,
        plan.current,
        m_base,
    );
    let n_width = emit_u64_const_bits(ctx, plan.op, &loc, plan.grid.n_tiles);
    let batch = emit_bin_op(
        ctx,
        plan.op,
        &loc,
        MirDivOp::get_concrete_op_info(),
        ty,
        n_batch,
        n_width,
    );
    let n_width = emit_u64_const_bits(ctx, plan.op, &loc, plan.grid.n_tiles);
    let n_base = emit_bin_op(
        ctx,
        plan.op,
        &loc,
        MirMulOp::get_concrete_op_info(),
        ty,
        batch,
        n_width,
    );
    let n_tile = emit_bin_op(
        ctx,
        plan.op,
        &loc,
        MirSubOp::get_concrete_op_info(),
        ty,
        n_batch,
        n_base,
    );
    [plan.current, m_tile, n_tile, batch]
}

fn emit_scheduler_saturating_add(ctx: &mut Context, plan: SchedulerAdvancePlan) -> Value {
    let loc = plan.op.deref(ctx).loc().clone();
    let ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let call = emit_op_before(
        ctx,
        plan.op,
        &loc,
        MirCallOp::get_concrete_op_info(),
        vec![ty],
        vec![plan.current, plan.stride],
    );
    MirCallOp::new(call).set_attr_callee(
        ctx,
        StringAttr::new(dialect_mir::rust_intrinsics::CALLEE_SATURATING_ADD.to_string()),
    );
    call.deref(ctx).get_result(0)
}

/// Turn the visible scheduler story back into its small runtime values.
///
/// ```text
/// new_1d                 blockIdx.x, gridDim.x
/// has_work(current)      current < total tiles
/// current -> coordinates two div + two mul/sub pairs
/// advance                u64 saturating_add(current, stride)
/// ```
///
/// A work tile is only an explanation for the coordinate math. It must flow
/// directly from `scheduler_current` to `work_tile_coordinates`; no runtime
/// `WorkTile` struct or pointer is allowed to survive this pass.
///
/// Backend A v0 also uses one logical grid for the whole module. Two
/// independent producer/consumer scheduler chains are fine when they repeat
/// that same grid:
///
/// ```text
/// producer scheduler ── grid<M,N,B>
/// consumer scheduler ── grid<M,N,B>
/// ```
///
/// Mixing two grids is rejected before either chain is changed.
fn lower_scheduler_to_mir(ctx: &mut Context, module: Ptr<Operation>) -> Result<(), ExpandError> {
    let mut all_ops = Vec::new();
    collect_ops(ctx, module, &mut all_ops);

    let new_id = CuteSchedulerNew1dOp::get_opid_static();
    let has_work_id = CuteSchedulerHasWorkOp::get_opid_static();
    let current_id = CuteSchedulerCurrentOp::get_opid_static();
    let coordinates_id = CuteWorkTileCoordinatesOp::get_opid_static();
    let advance_id = CuteSchedulerAdvanceOp::get_opid_static();
    let mut starts = Vec::new();
    let mut has_work = Vec::new();
    let mut currents = Vec::new();
    let mut coordinates = Vec::new();
    let mut advances = Vec::new();
    for op in &all_ops {
        let opid = Operation::get_opid(*op, ctx);
        if opid == new_id {
            starts.push(*op);
        } else if opid == has_work_id {
            let semantic = CuteSchedulerHasWorkOp::wrap(*op);
            let grid = semantic.tile_grid(ctx).ok_or_else(|| {
                ExpandError::Invalid("cute.scheduler_has_work is missing its tile grid".into())
            })?;
            let total_tiles = grid.total_tiles().ok_or_else(|| {
                ExpandError::Invalid("cute.scheduler_has_work tile count overflows u64".into())
            })?;
            has_work.push(SchedulerHasWorkPlan {
                op: *op,
                current: semantic.current(ctx),
                total_tiles,
            });
        } else if opid == current_id {
            currents.push(*op);
        } else if opid == coordinates_id {
            coordinates.push(preflight_scheduler_coordinates(ctx, *op)?);
        } else if opid == advance_id {
            let semantic = CuteSchedulerAdvanceOp::wrap(*op);
            advances.push(SchedulerAdvancePlan {
                op: *op,
                current: semantic.current(ctx),
                stride: semantic.stride(ctx),
            });
        }
    }
    if starts.is_empty()
        && has_work.is_empty()
        && currents.is_empty()
        && coordinates.is_empty()
        && advances.is_empty()
    {
        return Ok(());
    }
    // Native plan extraction completes before this first insertion.
    let mut rewriter = IRRewriter::<Recorder>::default();
    for plan in coordinates {
        let results = emit_scheduler_coordinates(ctx, plan);
        rewriter.replace_operation_with_values(ctx, plan.op, results.to_vec());
    }
    for plan in has_work {
        let loc = plan.op.deref(ctx).loc().clone();
        let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
        let bool_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
        let total = emit_u64_const_bits(ctx, plan.op, &loc, plan.total_tiles);
        let predicate = emit_bin_op(
            ctx,
            plan.op,
            &loc,
            MirLtOp::get_concrete_op_info(),
            bool_ty,
            plan.current,
            total,
        );
        debug_assert_eq!(plan.current.get_type(ctx), u64_ty);
        rewriter.replace_operation_with_values(ctx, plan.op, vec![predicate]);
    }
    for plan in advances {
        let next = emit_scheduler_saturating_add(ctx, plan);
        rewriter.replace_operation_with_values(ctx, plan.op, vec![next]);
    }
    for current in &currents {
        assert!(
            !CuteSchedulerCurrentOp::wrap(*current)
                .work_tile(ctx)
                .is_used(ctx),
            "scheduler preflight missed a live work-tile use"
        );
        rewriter.erase_operation(ctx, *current);
    }
    // Lower starts last. Arithmetic emitted above may still use a start
    // result; replacing it now updates every one of those new uses too.
    for start in starts {
        let loc = start.deref(ctx).loc().clone();
        let current = emit_scheduler_special_u64(
            ctx,
            start,
            &loc,
            ReadPtxSregCtaidXOp::get_concrete_op_info(),
        );
        let stride = emit_scheduler_special_u64(
            ctx,
            start,
            &loc,
            ReadPtxSregNctaidXOp::get_concrete_op_info(),
        );
        rewriter.replace_operation_with_values(ctx, start, vec![current, stride]);
    }
    Ok(())
}

/// Runtime facts hidden by one load-pipeline handle.
///
/// The handle itself is compiler-only. Backend A carries just the shared
/// barrier base plus the three static numbers needed by the hardware leaves.
#[derive(Clone, Copy)]
struct LoadPipelinePlan {
    base: Value,
    stages: u64,
    consumer_warps: u32,
    transaction_bytes: u32,
}

#[derive(Clone, Copy)]
enum PipelineSemanticPlan {
    Make(Ptr<Operation>),
    Init {
        op: Ptr<Operation>,
        pipeline: LoadPipelinePlan,
        init_thread: Value,
    },
    StateNew {
        op: Ptr<Operation>,
        state: CutePipelineStateAttr,
    },
    StateSlot {
        op: Ptr<Operation>,
        slot: Value,
    },
    StateAdvance {
        op: Ptr<Operation>,
        slot: Value,
        phase: Value,
        state: CutePipelineStateAttr,
    },
    ProducerAcquire {
        op: Ptr<Operation>,
        pipeline: LoadPipelinePlan,
        slot: Value,
        phase: Value,
    },
    ProducerExpectTx {
        op: Ptr<Operation>,
        pipeline: LoadPipelinePlan,
        slot: Value,
    },
    ConsumerWait {
        op: Ptr<Operation>,
        pipeline: LoadPipelinePlan,
        slot: Value,
        phase: Value,
    },
    ConsumerRelease {
        op: Ptr<Operation>,
        pipeline: LoadPipelinePlan,
        slot: Value,
    },
    ProducerTail {
        op: Ptr<Operation>,
        pipeline: LoadPipelinePlan,
        slot: Value,
        phase: Value,
    },
}

fn resolve_load_pipeline(ctx: &Context, value: Value) -> LoadPipelinePlan {
    let make = value
        .defining_op()
        .expect("shared verifier requires a direct load-pipeline make producer");
    debug_assert!(Operation::get_opid(make, ctx) == CuteTmaLoadPipelineMakeOp::get_opid_static());
    let make_op = CuteTmaLoadPipelineMakeOp::wrap(make);
    debug_assert_eq!(make_op.pipeline(ctx), value);
    let pipeline = make_op
        .pipeline_type(ctx)
        .expect("shared verifier requires a load-pipeline type");
    LoadPipelinePlan {
        base: make_op.base(ctx),
        stages: pipeline.stages,
        consumer_warps: pipeline.consumer_warps,
        transaction_bytes: pipeline.transaction_bytes,
    }
}

fn preflight_pipeline_plan(
    ctx: &Context,
    op: Ptr<Operation>,
    opid: &OpId,
) -> Result<PipelineSemanticPlan, ExpandError> {
    if *opid == CuteTmaLoadPipelineMakeOp::get_opid_static() {
        return Ok(PipelineSemanticPlan::Make(op));
    }
    if *opid == CuteTmaLoadPipelineInitOp::get_opid_static() {
        let semantic = CuteTmaLoadPipelineInitOp::wrap(op);
        return Ok(PipelineSemanticPlan::Init {
            op,
            pipeline: resolve_load_pipeline(ctx, semantic.pipeline(ctx)),
            init_thread: semantic.init_thread(ctx),
        });
    }
    if *opid == CutePipelineStateNewOp::get_opid_static() {
        let semantic = CutePipelineStateNewOp::wrap(op);
        return Ok(PipelineSemanticPlan::StateNew {
            op,
            state: semantic
                .state(ctx)
                .expect("shared verifier requires pipeline state metadata"),
        });
    }
    if *opid == CutePipelineStateSlotOp::get_opid_static() {
        let semantic = CutePipelineStateSlotOp::wrap(op);
        return Ok(PipelineSemanticPlan::StateSlot {
            op,
            slot: semantic.slot(ctx),
        });
    }
    if *opid == CutePipelineStateAdvanceOp::get_opid_static() {
        let semantic = CutePipelineStateAdvanceOp::wrap(op);
        return Ok(PipelineSemanticPlan::StateAdvance {
            op,
            slot: semantic.slot(ctx),
            phase: semantic.phase(ctx),
            state: semantic
                .state(ctx)
                .expect("shared verifier requires pipeline state metadata"),
        });
    }
    if *opid == CutePipelineProducerAcquireOp::get_opid_static() {
        let semantic = CutePipelineProducerAcquireOp::wrap(op);
        return Ok(PipelineSemanticPlan::ProducerAcquire {
            op,
            pipeline: resolve_load_pipeline(ctx, semantic.pipeline(ctx)),
            slot: semantic.slot(ctx),
            phase: semantic.phase(ctx),
        });
    }
    if *opid == CutePipelineProducerExpectTxOp::get_opid_static() {
        let semantic = CutePipelineProducerExpectTxOp::wrap(op);
        let pipeline = resolve_load_pipeline(ctx, semantic.pipeline(ctx));
        return Ok(PipelineSemanticPlan::ProducerExpectTx {
            op,
            pipeline,
            slot: semantic.slot(ctx),
        });
    }
    if *opid == CutePipelineConsumerWaitOp::get_opid_static() {
        let semantic = CutePipelineConsumerWaitOp::wrap(op);
        return Ok(PipelineSemanticPlan::ConsumerWait {
            op,
            pipeline: resolve_load_pipeline(ctx, semantic.pipeline(ctx)),
            slot: semantic.slot(ctx),
            phase: semantic.phase(ctx),
        });
    }
    if *opid == CutePipelineConsumerReleaseOp::get_opid_static() {
        let semantic = CutePipelineConsumerReleaseOp::wrap(op);
        return Ok(PipelineSemanticPlan::ConsumerRelease {
            op,
            pipeline: resolve_load_pipeline(ctx, semantic.pipeline(ctx)),
            slot: semantic.slot(ctx),
        });
    }
    if *opid == CutePipelineProducerTailOp::get_opid_static() {
        let semantic = CutePipelineProducerTailOp::wrap(op);
        return Ok(PipelineSemanticPlan::ProducerTail {
            op,
            pipeline: resolve_load_pipeline(ctx, semantic.pipeline(ctx)),
            slot: semantic.slot(ctx),
            phase: semantic.phase(ctx),
        });
    }
    Err(ExpandError::Invalid(format!(
        "unknown load-pipeline operation `{opid}`"
    )))
}

fn append_pipeline_op(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    loc: &Location,
    info: OpInfoPair,
    results: Vec<TypeHandle>,
    operands: Vec<Value>,
) -> Ptr<Operation> {
    let op = Operation::new(ctx, info, results, operands, vec![], 0);
    op.deref_mut(ctx).set_loc(loc.clone());
    op.insert_at_back(block, ctx);
    op
}

fn append_pipeline_u32(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    loc: &Location,
    value: u32,
) -> Value {
    let ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
    let op = append_pipeline_op(
        ctx,
        block,
        loc,
        MirConstantOp::get_concrete_op_info(),
        vec![ty.into()],
        vec![],
    );
    MirConstantOp::new(op).set_attr_value(
        ctx,
        IntegerAttr::new(
            ty,
            APInt::from_u64(u64::from(value), NonZeroUsize::new(32).unwrap()),
        ),
    );
    op.deref(ctx).get_result(0)
}

fn append_pipeline_u64(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    loc: &Location,
    value: u64,
) -> Value {
    let ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
    let op = append_pipeline_op(
        ctx,
        block,
        loc,
        MirConstantOp::get_concrete_op_info(),
        vec![ty.into()],
        vec![],
    );
    MirConstantOp::new(op).set_attr_value(
        ctx,
        IntegerAttr::new(ty, APInt::from_u64(value, NonZeroUsize::new(64).unwrap())),
    );
    op.deref(ctx).get_result(0)
}

fn append_pipeline_cast_u32_to_u64(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    loc: &Location,
    value: Value,
) -> Value {
    let ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let cast = append_pipeline_op(
        ctx,
        block,
        loc,
        MirCastOp::get_concrete_op_info(),
        vec![ty],
        vec![value],
    );
    MirCastOp::new(cast).set_attr_cast_kind(ctx, MirCastKindAttr::IntToInt);
    cast.deref(ctx).get_result(0)
}

fn append_pipeline_barrier_pointer(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    loc: &Location,
    pipeline: LoadPipelinePlan,
    slot: Value,
    empty_ring: bool,
) -> Value {
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let mut index = append_pipeline_cast_u32_to_u64(ctx, block, loc, slot);
    if empty_ring {
        let stages = append_pipeline_u64(ctx, block, loc, pipeline.stages);
        index = append_pipeline_op(
            ctx,
            block,
            loc,
            MirAddOp::get_concrete_op_info(),
            vec![u64_ty],
            vec![index, stages],
        )
        .deref(ctx)
        .get_result(0);
    }
    append_pipeline_op(
        ctx,
        block,
        loc,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![pipeline.base.get_type(ctx)],
        vec![pipeline.base, index],
    )
    .deref(ctx)
    .get_result(0)
}

fn emit_pipeline_barrier_pointer_before(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    pipeline: LoadPipelinePlan,
    slot: Value,
    empty_ring: bool,
) -> Value {
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let mut index = emit_cast_value(ctx, anchor, loc, slot, u64_ty, MirCastKindAttr::IntToInt);
    if empty_ring {
        let stages = emit_u64_const_bits(ctx, anchor, loc, pipeline.stages);
        index = emit_bin_op(
            ctx,
            anchor,
            loc,
            MirAddOp::get_concrete_op_info(),
            u64_ty,
            index,
            stages,
        );
    }
    emit_op_before(
        ctx,
        anchor,
        loc,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![pipeline.base.get_type(ctx)],
        vec![pipeline.base, index],
    )
    .deref(ctx)
    .get_result(0)
}

fn append_pipeline_state_advance(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    loc: &Location,
    slot: Value,
    phase: Value,
    stages: u64,
) -> [Value; 2] {
    let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    let bool_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
    let one = append_pipeline_u32(ctx, block, loc, 1);
    let incremented = append_pipeline_op(
        ctx,
        block,
        loc,
        MirAddOp::get_concrete_op_info(),
        vec![u32_ty],
        vec![slot, one],
    )
    .deref(ctx)
    .get_result(0);
    let width = append_pipeline_u32(ctx, block, loc, stages as u32);
    let wraps = append_pipeline_op(
        ctx,
        block,
        loc,
        MirEqOp::get_concrete_op_info(),
        vec![bool_ty],
        vec![incremented, width],
    )
    .deref(ctx)
    .get_result(0);
    let wraps_u32 = append_pipeline_op(
        ctx,
        block,
        loc,
        MirCastOp::get_concrete_op_info(),
        vec![u32_ty],
        vec![wraps],
    );
    MirCastOp::new(wraps_u32).set_attr_cast_kind(ctx, MirCastKindAttr::IntToInt);
    let wraps_u32 = wraps_u32.deref(ctx).get_result(0);
    // `wraps - 1` is all ones when the slot stays in range and zero when it
    // wraps. This is a MIR-only select, so signedness cannot drift while MIR
    // operations are converted to LLVM one by one.
    let one = append_pipeline_u32(ctx, block, loc, 1);
    let keep_mask = append_pipeline_op(
        ctx,
        block,
        loc,
        MirSubOp::get_concrete_op_info(),
        vec![u32_ty],
        vec![wraps_u32, one],
    )
    .deref(ctx)
    .get_result(0);
    let next_slot = append_pipeline_op(
        ctx,
        block,
        loc,
        MirBitAndOp::get_concrete_op_info(),
        vec![u32_ty],
        vec![incremented, keep_mask],
    )
    .deref(ctx)
    .get_result(0);
    let next_phase = append_pipeline_op(
        ctx,
        block,
        loc,
        MirBitXorOp::get_concrete_op_info(),
        vec![u32_ty],
        vec![phase, wraps_u32],
    )
    .deref(ctx)
    .get_result(0);
    [next_slot, next_phase]
}

fn emit_pipeline_state_advance_before(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    slot: Value,
    phase: Value,
    stages: u64,
) -> [Value; 2] {
    let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    let bool_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
    let one = emit_u32_const(ctx, anchor, loc, 1);
    let incremented = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirAddOp::get_concrete_op_info(),
        u32_ty,
        slot,
        one,
    );
    let width = emit_u32_const(ctx, anchor, loc, stages as u32);
    let wraps = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirEqOp::get_concrete_op_info(),
        bool_ty,
        incremented,
        width,
    );
    let wraps_u32 = emit_op_before(
        ctx,
        anchor,
        loc,
        MirCastOp::get_concrete_op_info(),
        vec![u32_ty],
        vec![wraps],
    );
    MirCastOp::new(wraps_u32).set_attr_cast_kind(ctx, MirCastKindAttr::IntToInt);
    let wraps_u32 = wraps_u32.deref(ctx).get_result(0);
    let one = emit_u32_const(ctx, anchor, loc, 1);
    let keep_mask = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirSubOp::get_concrete_op_info(),
        u32_ty,
        wraps_u32,
        one,
    );
    let next_slot = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirBitAndOp::get_concrete_op_info(),
        u32_ty,
        incremented,
        keep_mask,
    );
    let next_phase = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirBitXorOp::get_concrete_op_info(),
        u32_ty,
        phase,
        wraps_u32,
    );
    [next_slot, next_phase]
}

fn append_pipeline_goto(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    loc: &Location,
    target: Ptr<BasicBlock>,
) -> Ptr<Operation> {
    let op = Operation::new(
        ctx,
        MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![target],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());
    op.insert_at_back(block, ctx);
    op
}

fn append_pipeline_cond_branch(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    loc: &Location,
    condition: Value,
    if_true: Ptr<BasicBlock>,
    if_false: Ptr<BasicBlock>,
) -> Ptr<Operation> {
    let (operands, segment_sizes) =
        MirCondBranchOp::compute_segment_sizes(vec![vec![condition], vec![], vec![]]);
    let op = Operation::new(
        ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        operands,
        vec![if_true, if_false],
        0,
    );
    MirCondBranchOp::new(op).set_operand_segment_sizes(ctx, segment_sizes);
    op.deref_mut(ctx).set_loc(loc.clone());
    op.insert_at_back(block, ctx);
    op
}

/// Move everything after `anchor` into a fresh continuation block.
///
/// Preflight proves the anchor is linked before any rewrite starts. The old
/// terminator moves with the trailing operations, leaving the source block
/// ready for one explicit branch.
fn split_pipeline_anchor(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
) -> (Ptr<BasicBlock>, Ptr<BasicBlock>) {
    let source = anchor
        .deref(ctx)
        .get_parent_block()
        .expect("pipeline preflight must reject an unlinked semantic op");
    let continuation = BasicBlock::new(ctx, None, vec![]);
    continuation.insert_after(ctx, source);

    let mut next = anchor.deref(ctx).get_next();
    while let Some(operation) = next {
        next = operation.deref(ctx).get_next();
        operation.unlink(ctx);
        operation.insert_at_back(continuation, ctx);
    }
    (source, continuation)
}

fn append_marked_pipeline_intrinsic(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    loc: &Location,
    op: Ptr<Operation>,
) {
    op.deref_mut(ctx).set_loc(loc.clone());
    op.insert_at_back(block, ctx);
    request_generated_intrinsic_marker(ctx, op);
}

fn emit_pipeline_init(
    ctx: &mut Context,
    rewriter: &mut IRRewriter<Recorder>,
    op: Ptr<Operation>,
    pipeline: LoadPipelinePlan,
    init_thread: Value,
) {
    let loc = op.deref(ctx).loc().clone();
    let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    let bool_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
    let tid = emit_op_before(
        ctx,
        op,
        &loc,
        ReadPtxSregTidXOp::get_concrete_op_info(),
        vec![u32_ty],
        vec![],
    );
    request_generated_intrinsic_marker(ctx, tid);
    let tid = tid.deref(ctx).get_result(0);
    let is_initializer = emit_bin_op(
        ctx,
        op,
        &loc,
        MirEqOp::get_concrete_op_info(),
        bool_ty,
        tid,
        init_thread,
    );

    let (source, continuation) = split_pipeline_anchor(ctx, op);
    let init = BasicBlock::new(ctx, None, vec![]);
    init.insert_after(ctx, source);
    let sync = BasicBlock::new(ctx, None, vec![]);
    sync.insert_after(ctx, init);
    rewriter.erase_operation(ctx, op);
    append_pipeline_cond_branch(ctx, source, &loc, is_initializer, init, sync);

    for stage in 0..pipeline.stages {
        let slot = append_pipeline_u64(ctx, init, &loc, stage);
        let full = append_pipeline_op(
            ctx,
            init,
            &loc,
            MirPtrOffsetOp::get_concrete_op_info(),
            vec![pipeline.base.get_type(ctx)],
            vec![pipeline.base, slot],
        )
        .deref(ctx)
        .get_result(0);
        let one = append_pipeline_u32(ctx, init, &loc, 1);
        let initialize = MbarrierInitSharedOp::build(ctx, full, one);
        append_marked_pipeline_intrinsic(ctx, init, &loc, initialize);

        let slot = append_pipeline_u64(ctx, init, &loc, pipeline.stages + stage);
        let empty = append_pipeline_op(
            ctx,
            init,
            &loc,
            MirPtrOffsetOp::get_concrete_op_info(),
            vec![pipeline.base.get_type(ctx)],
            vec![pipeline.base, slot],
        )
        .deref(ctx)
        .get_result(0);
        let warps = append_pipeline_u32(ctx, init, &loc, pipeline.consumer_warps);
        let initialize = MbarrierInitSharedOp::build(ctx, empty, warps);
        append_marked_pipeline_intrinsic(ctx, init, &loc, initialize);
    }
    let fence = FenceMbarrierInitReleaseClusterOp::build(ctx);
    append_marked_pipeline_intrinsic(ctx, init, &loc, fence);
    append_pipeline_goto(ctx, init, &loc, sync);

    let barrier = Operation::new(
        ctx,
        Barrier0Op::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    append_marked_pipeline_intrinsic(ctx, sync, &loc, barrier);
    append_pipeline_goto(ctx, sync, &loc, continuation);
}

fn emit_pipeline_poll(
    ctx: &mut Context,
    rewriter: &mut IRRewriter<Recorder>,
    op: Ptr<Operation>,
    pipeline: LoadPipelinePlan,
    slot: Value,
    phase: Value,
    empty_ring: bool,
) {
    let loc = op.deref(ctx).loc().clone();
    let barrier = emit_pipeline_barrier_pointer_before(ctx, op, &loc, pipeline, slot, empty_ring);
    let (source, continuation) = split_pipeline_anchor(ctx, op);
    let poll = BasicBlock::new(ctx, None, vec![]);
    poll.insert_after(ctx, source);
    rewriter.erase_operation(ctx, op);
    append_pipeline_goto(ctx, source, &loc, poll);
    let wait = MbarrierTryWaitParitySharedOp::build(ctx, barrier, phase);
    let ready = wait.deref(ctx).get_result(0);
    append_marked_pipeline_intrinsic(ctx, poll, &loc, wait);
    append_pipeline_cond_branch(ctx, poll, &loc, ready, continuation, poll);
}

fn emit_pipeline_expect_tx(
    ctx: &mut Context,
    rewriter: &mut IRRewriter<Recorder>,
    op: Ptr<Operation>,
    pipeline: LoadPipelinePlan,
    slot: Value,
) {
    let loc = op.deref(ctx).loc().clone();
    let barrier = emit_pipeline_barrier_pointer_before(ctx, op, &loc, pipeline, slot, false);
    let bytes = emit_u32_const(ctx, op, &loc, pipeline.transaction_bytes);
    let expect = MbarrierArriveExpectTxSharedOp::build(ctx, barrier, bytes);
    expect.deref_mut(ctx).set_loc(loc);
    expect.insert_before(ctx, op);
    request_generated_intrinsic_marker(ctx, expect);
    // The hardware token is deliberately ignored. TMA copies need the exact
    // full-barrier pointer that was passed to the arrival.
    rewriter.replace_operation_with_values(ctx, op, vec![barrier]);
}

fn emit_pipeline_release(
    ctx: &mut Context,
    rewriter: &mut IRRewriter<Recorder>,
    op: Ptr<Operation>,
    pipeline: LoadPipelinePlan,
    slot: Value,
) {
    let loc = op.deref(ctx).loc().clone();
    let barrier = emit_pipeline_barrier_pointer_before(ctx, op, &loc, pipeline, slot, true);
    let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    let bool_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
    let lane = emit_op_before(
        ctx,
        op,
        &loc,
        ReadPtxSregLaneIdOp::get_concrete_op_info(),
        vec![u32_ty],
        vec![],
    );
    request_generated_intrinsic_marker(ctx, lane);
    let lane = lane.deref(ctx).get_result(0);
    let zero = emit_u32_const(ctx, op, &loc, 0);
    let lane_zero = emit_bin_op(
        ctx,
        op,
        &loc,
        MirEqOp::get_concrete_op_info(),
        bool_ty,
        lane,
        zero,
    );
    let (source, continuation) = split_pipeline_anchor(ctx, op);
    let arrive = BasicBlock::new(ctx, None, vec![]);
    arrive.insert_after(ctx, source);
    rewriter.erase_operation(ctx, op);
    append_pipeline_cond_branch(ctx, source, &loc, lane_zero, arrive, continuation);
    let arrival = MbarrierArriveSharedOp::build(ctx, barrier);
    append_marked_pipeline_intrinsic(ctx, arrive, &loc, arrival);
    append_pipeline_goto(ctx, arrive, &loc, continuation);
}

fn emit_pipeline_tail(
    ctx: &mut Context,
    rewriter: &mut IRRewriter<Recorder>,
    op: Ptr<Operation>,
    pipeline: LoadPipelinePlan,
    mut slot: Value,
    mut phase: Value,
) {
    let loc = op.deref(ctx).loc().clone();
    let (source, continuation) = split_pipeline_anchor(ctx, op);
    rewriter.erase_operation(ctx, op);

    let mut entry = source;
    let mut last_inserted = source;
    for stage in 0..pipeline.stages {
        let barrier = append_pipeline_barrier_pointer(ctx, entry, &loc, pipeline, slot, true);
        let poll = BasicBlock::new(ctx, None, vec![]);
        poll.insert_after(ctx, last_inserted);
        last_inserted = poll;
        append_pipeline_goto(ctx, entry, &loc, poll);
        let wait = MbarrierTryWaitParitySharedOp::build(ctx, barrier, phase);
        let ready = wait.deref(ctx).get_result(0);
        append_marked_pipeline_intrinsic(ctx, poll, &loc, wait);

        if stage + 1 == pipeline.stages {
            append_pipeline_cond_branch(ctx, poll, &loc, ready, continuation, poll);
        } else {
            let advance = BasicBlock::new(ctx, None, vec![]);
            advance.insert_after(ctx, last_inserted);
            last_inserted = advance;
            append_pipeline_cond_branch(ctx, poll, &loc, ready, advance, poll);
            [slot, phase] =
                append_pipeline_state_advance(ctx, advance, &loc, slot, phase, pipeline.stages);
            entry = advance;
        }
    }
}

/// Expand the visible TMA load-pipeline story to the proven mbarrier recipe.
///
/// ```text
/// make<3>                 no runtime object
/// init                    6 mbarrier.init + fence + CTA barrier
/// acquire / wait          retry one parity probe until ready
/// expect bytes            arrive-expect-tx; return the same full pointer
/// release                 lane 0 arrives on the empty barrier
/// tail<3>                 three statically visible empty-barrier waits
/// ```
///
/// `expect bytes` must feed at least one direct TMA copy. Backend A adds the
/// static byte size of every attached tile and requires that total to equal
/// the pipeline's `transaction_bytes`; it does not assume a copy count. The
/// runtime story must still execute each attached copy exactly once. Static
/// SSA uses alone cannot describe branch frequency.
///
/// Every verifier, producer lookup, pointer-use check, and block-attachment
/// check finishes before the first insertion. The CFG rewrite therefore
/// cannot leave half of a pipeline expanded when a later operation is bad.
fn lower_pipeline_to_mir(ctx: &mut Context, module: Ptr<Operation>) -> Result<(), ExpandError> {
    let mut all_ops = Vec::new();
    collect_ops(ctx, module, &mut all_ops);
    let pipeline_ids = pipeline_semantic_ids();

    let mut plans = Vec::new();
    let mut makes = Vec::new();
    for op in &all_ops {
        let opid = Operation::get_opid(*op, ctx);
        if !pipeline_ids.contains(&opid) {
            continue;
        }
        let plan = preflight_pipeline_plan(ctx, *op, &opid)?;
        if let PipelineSemanticPlan::Make(make) = plan {
            makes.push(make);
        }
        plans.push(plan);
    }
    if plans.is_empty() {
        return Ok(());
    }

    // No mutation occurs above this line.
    let mut rewriter = IRRewriter::<Recorder>::default();
    // Rewrite consumers before producers. A plan stores the original SSA
    // value, so lowering a state producer first would erase that value before
    // a later plan can attach its new use. Replacing the producer last updates
    // every leaf/CFG use emitted from the still-live original value.
    for plan in plans.into_iter().rev() {
        match plan {
            PipelineSemanticPlan::Make(_) => {}
            PipelineSemanticPlan::Init {
                op,
                pipeline,
                init_thread,
            } => emit_pipeline_init(ctx, &mut rewriter, op, pipeline, init_thread),
            PipelineSemanticPlan::StateNew { op, state } => {
                let loc = op.deref(ctx).loc().clone();
                let slot = emit_u32_const(ctx, op, &loc, 0);
                let phase = emit_u32_const(
                    ctx,
                    op,
                    &loc,
                    u32::from(state.role == CutePipelineRoleAttr::Producer),
                );
                rewriter.replace_operation_with_values(ctx, op, vec![slot, phase]);
            }
            PipelineSemanticPlan::StateSlot { op, slot } => {
                let loc = op.deref(ctx).loc().clone();
                let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
                let index = emit_cast_value(ctx, op, &loc, slot, u64_ty, MirCastKindAttr::IntToInt);
                rewriter.replace_operation_with_values(ctx, op, vec![index]);
            }
            PipelineSemanticPlan::StateAdvance {
                op,
                slot,
                phase,
                state,
            } => {
                let loc = op.deref(ctx).loc().clone();
                let next =
                    emit_pipeline_state_advance_before(ctx, op, &loc, slot, phase, state.stages);
                rewriter.replace_operation_with_values(ctx, op, next.to_vec());
            }
            PipelineSemanticPlan::ProducerAcquire {
                op,
                pipeline,
                slot,
                phase,
            } => emit_pipeline_poll(ctx, &mut rewriter, op, pipeline, slot, phase, true),
            PipelineSemanticPlan::ProducerExpectTx { op, pipeline, slot } => {
                emit_pipeline_expect_tx(ctx, &mut rewriter, op, pipeline, slot)
            }
            PipelineSemanticPlan::ConsumerWait {
                op,
                pipeline,
                slot,
                phase,
            } => emit_pipeline_poll(ctx, &mut rewriter, op, pipeline, slot, phase, false),
            PipelineSemanticPlan::ConsumerRelease { op, pipeline, slot } => {
                emit_pipeline_release(ctx, &mut rewriter, op, pipeline, slot)
            }
            PipelineSemanticPlan::ProducerTail {
                op,
                pipeline,
                slot,
                phase,
            } => emit_pipeline_tail(ctx, &mut rewriter, op, pipeline, slot, phase),
        }
    }

    for make in makes {
        let handle = CuteTmaLoadPipelineMakeOp::wrap(make).pipeline(ctx);
        if handle.is_used(ctx) {
            return Err(ExpandError::Invalid(
                "pipeline preflight missed a live handle use after expansion".into(),
            ));
        }
        rewriter.erase_operation(ctx, make);
    }
    Ok(())
}

struct EpilogueStorePlan {
    semantic_store: Ptr<Operation>,
    source_view: Ptr<Operation>,
    destination_view: Ptr<Operation>,
    leaf_operands: [Value; 4],
    leaf_layout: ComposedLayout,
    leaf_element: TypeHandle,
}

struct EpilogueLoweringPlan {
    overlay: Ptr<Operation>,
    warp_slice: Ptr<Operation>,
    store_fragment: Ptr<Operation>,
    accumulator: MmaCarrierLayout,
    reusable_sync: Ptr<Operation>,
    ready_sync: Ptr<Operation>,
    halves: [Ptr<Operation>; 2],
    acquire: Ptr<Operation>,
    commit: Ptr<Operation>,
    tail: Ptr<Operation>,
    stores: [EpilogueStorePlan; 2],
}

fn preflight_epilogue_store(
    ctx: &Context,
    semantic_store: Ptr<Operation>,
) -> Result<EpilogueStorePlan, ExpandError> {
    let store = CuteTmaStore2dSemanticOp::wrap(semantic_store);
    let source_view = direct_tma_view_producer(
        ctx,
        store.source(ctx),
        &CuteTmaSmemViewOp::get_opid_static(),
        "epilogue store source",
    );
    let destination_view = direct_tma_view_producer(
        ctx,
        store.destination(ctx),
        &CuteTmaGmemViewOp::get_opid_static(),
        "epilogue store destination",
    );
    let source = CuteTmaSmemViewOp::wrap(source_view);
    let destination = CuteTmaGmemViewOp::wrap(destination_view);
    let (element, layout) = {
        let source_ty = store.source(ctx).get_type(ctx);
        let source_ty_ref = source_ty.deref(ctx);
        let view = source_ty_ref
            .downcast_ref::<CuteTmaViewType>()
            .ok_or_else(|| {
                ExpandError::Invalid("epilogue store source has no TMA view type".into())
            })?;
        let element = view.element(ctx).ok_or_else(|| {
            ExpandError::Invalid("epilogue store source has no physical carrier type".into())
        })?;
        (element, view.smem_layout.0.clone())
    };
    Ok(EpilogueStorePlan {
        semantic_store,
        source_view,
        destination_view,
        leaf_operands: [
            source.base(ctx),
            destination.descriptor(ctx),
            store.tile_row(ctx),
            store.tile_column(ctx),
        ],
        leaf_layout: layout,
        leaf_element: element,
    })
}

fn materialize_epilogue_store_leaves(
    ctx: &mut Context,
    stores: &[EpilogueStorePlan; 2],
) -> Result<[Ptr<Operation>; 2], ExpandError> {
    let mut leaves = Vec::with_capacity(stores.len());
    for store in stores {
        let leaf = CuteTmaStore2dOp::new(
            ctx,
            store.leaf_operands,
            store.leaf_layout.clone(),
            store.leaf_element,
        );
        let leaf_op = leaf.get_operation();
        leaf_op
            .deref_mut(ctx)
            .set_loc(store.semantic_store.deref(ctx).loc().clone());
        if let Err(error) = leaf.verify(ctx) {
            Operation::erase(leaf_op, ctx);
            for pending in leaves.drain(..) {
                Operation::erase(pending, ctx);
            }
            return Err(ExpandError::Invalid(format!(
                "semantic `cute.tma_store_2d` cannot become the existing TMA-store leaf: {error}"
            )));
        }
        leaves.push(leaf_op);
    }
    Ok(leaves
        .try_into()
        .expect("the epilogue preflight always describes exactly two stores"))
}

fn unique_operation_push(operations: &mut Vec<Ptr<Operation>>, operation: Ptr<Operation>) {
    if !operations.contains(&operation) {
        operations.push(operation);
    }
}

fn preflight_epilogue(
    ctx: &mut Context,
    all_ops: &[Ptr<Operation>],
) -> Result<Option<EpilogueLoweringPlan>, ExpandError> {
    let ids = epilogue_semantic_ids();
    let mut overlays = Vec::new();
    let mut slices = Vec::new();
    let mut fragments = Vec::new();
    let mut syncs = Vec::new();
    let mut halves = Vec::new();
    let mut acquires = Vec::new();
    let mut commits = Vec::new();
    let mut tails = Vec::new();
    let mut stores = Vec::new();
    for operation in all_ops {
        let opid = Operation::get_opid(*operation, ctx);
        if !ids.contains(&opid) {
            continue;
        }
        if opid == ids[0] {
            overlays.push(*operation);
        } else if opid == ids[1] {
            slices.push(*operation);
        } else if opid == ids[2] {
            fragments.push(*operation);
        } else if opid == ids[3] {
            syncs.push(*operation);
        } else if opid == ids[4] {
            halves.push(*operation);
        } else if opid == ids[5] {
            acquires.push(*operation);
        } else if opid == ids[6] {
            commits.push(*operation);
        } else if opid == ids[7] {
            tails.push(*operation);
        } else if opid == ids[8] {
            stores.push(*operation);
        }
    }

    let found = overlays.len()
        + slices.len()
        + fragments.len()
        + syncs.len()
        + halves.len()
        + acquires.len()
        + commits.len()
        + tails.len()
        + stores.len();
    if found == 0 {
        return Ok(None);
    }
    if overlays.len() != 1
        || slices.len() != 1
        || fragments.len() != 1
        || syncs.len() != 2
        || halves.len() != 2
        || acquires.len() != 1
        || commits.len() != 1
        || tails.len() != 1
        || stores.len() != 2
    {
        return Err(ExpandError::Invalid(format!(
            "Backend A epilogue v0 needs exactly 1 overlay, 1 warp slice, 1 fragment store, 2 syncs, 2 halves, 1 acquire, 2 TMA stores, 1 commit, and 1 tail; found {}, {}, {}, {}, {}, {}, {}, {}, {}",
            overlays.len(),
            slices.len(),
            fragments.len(),
            syncs.len(),
            halves.len(),
            acquires.len(),
            stores.len(),
            commits.len(),
            tails.len()
        )));
    }

    let overlay = overlays[0];
    let warp_slice = slices[0];
    let store_fragment = fragments[0];
    let acquire = acquires[0];
    let commit = commits[0];
    let tail = tails[0];
    let fragment = CuteEpilogueStoreFragmentOp::wrap(store_fragment);
    let accumulator = preflight_mma_carrier(
        ctx,
        fragment.accumulator(ctx),
        MmaRegisterKind::F32,
        64,
        "cute.epilogue_store_fragment accumulator",
    )?;

    let reusable_sync = syncs
        .iter()
        .copied()
        .find(|operation| {
            CuteEpilogueSyncOp::wrap(*operation).phase(ctx)
                == Some(CuteEpilogueSyncPhaseAttr::Reusable)
        })
        .expect("shared verifier guarantees the Reusable sync");
    let ready_sync = syncs
        .iter()
        .copied()
        .find(|operation| {
            CuteEpilogueSyncOp::wrap(*operation).phase(ctx)
                == Some(CuteEpilogueSyncPhaseAttr::ReadyForTma)
        })
        .expect("shared verifier guarantees the ReadyForTma sync");

    let half_zero = halves
        .iter()
        .copied()
        .find(|operation| {
            CuteEpilogueHalfOp::wrap(*operation)
                .half(ctx)
                .is_some_and(|h| h.0 == 0)
        })
        .expect("shared verifier guarantees epilogue half 0");
    let half_one = halves
        .iter()
        .copied()
        .find(|operation| {
            CuteEpilogueHalfOp::wrap(*operation)
                .half(ctx)
                .is_some_and(|h| h.0 == 1)
        })
        .expect("shared verifier guarantees epilogue half 1");

    let store_zero = preflight_epilogue_store(ctx, stores[0])?;
    let store_one = preflight_epilogue_store(ctx, stores[1])?;

    Ok(Some(EpilogueLoweringPlan {
        overlay,
        warp_slice,
        store_fragment,
        accumulator,
        reusable_sync,
        ready_sync,
        halves: [half_zero, half_one],
        acquire,
        commit,
        tail,
        stores: [store_zero, store_one],
    }))
}

fn emit_epilogue_store_fragment(ctx: &mut Context, plan: &EpilogueLoweringPlan) {
    let semantic = CuteEpilogueStoreFragmentOp::wrap(plan.store_fragment);
    let anchor = plan.store_fragment;
    let loc = anchor.deref(ctx).loc().clone();
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let u8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let u8_ptr: TypeHandle = MirPtrType::get(ctx, u8_ty, true, address_space::SHARED).into();
    let warp = semantic.warp_id(ctx);
    let lane = semantic.lane(ctx);
    let base = semantic.base(ctx);

    // The safety contract gives warp in 0..8 and lane in 0..32. Keep the
    // address algebra small and visual instead of rebuilding the generic
    // layout evaluator sixteen times:
    //
    // row*64 = (warp%4)*1024 + (lane&15)*64 + m_band*4096
    // col    = (warp/4)*16 + local N offset
    // swizzle S<3,3,3> toggles (lane&7)*8 in each 128x64 half.
    let three = emit_u64_const_bits(ctx, anchor, &loc, 3);
    let warp_m = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirBitAndOp::get_concrete_op_info(),
        u64_ty,
        warp,
        three,
    );
    let shift_two = emit_i32_const(ctx, anchor, &loc, 2);
    let warp_n = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirShrOp::get_concrete_op_info(),
        u64_ty,
        warp,
        shift_two,
    );
    let lane64 = emit_cast_before(ctx, anchor, &loc, lane, u64_ty, MirCastKindAttr::IntToInt);
    let fifteen = emit_u64_const_bits(ctx, anchor, &loc, 15);
    let lane15 = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirBitAndOp::get_concrete_op_info(),
        u64_ty,
        lane64,
        fifteen,
    );
    let rows_per_warp_m = emit_u64_const_bits(ctx, anchor, &loc, 1024);
    let warp_rows = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        warp_m,
        rows_per_warp_m,
    );
    let row_stride = emit_u64_const_bits(ctx, anchor, &loc, 64);
    let lane_rows = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        lane15,
        row_stride,
    );
    let row_zero = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirAddOp::get_concrete_op_info(),
        u64_ty,
        warp_rows,
        lane_rows,
    );
    let m_band_stride = emit_u64_const_bits(ctx, anchor, &loc, 4096);
    let row_one = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirAddOp::get_concrete_op_info(),
        u64_ty,
        row_zero,
        m_band_stride,
    );
    let columns_per_warp_n = emit_u64_const_bits(ctx, anchor, &loc, 16);
    let column_base = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        warp_n,
        columns_per_warp_n,
    );
    let seven = emit_u64_const_bits(ctx, anchor, &loc, 7);
    let lane_seven = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirBitAndOp::get_concrete_op_info(),
        u64_ty,
        lane64,
        seven,
    );
    let swizzle_stride = emit_u64_const_bits(ctx, anchor, &loc, 8);
    let swizzle = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        lane_seven,
        swizzle_stride,
    );
    let half_stride = emit_u64_const_bits(ctx, anchor, &loc, 8192);

    let mut accumulator = Vec::with_capacity(64);
    emit_carrier_extracts(
        ctx,
        anchor,
        &loc,
        semantic.accumulator(ctx),
        &plan.accumulator,
        &mut accumulator,
    );
    debug_assert_eq!(accumulator.len(), 64);
    const N_OFFSETS: [u64; 8] = [0, 8, 32, 40, 64, 72, 96, 104];
    for m_band in 0..2 {
        for (n_slot, n_offset) in N_OFFSETS.iter().copied().enumerate() {
            let local_offset = n_offset % 64;
            let local_offset = emit_u64_const_bits(ctx, anchor, &loc, local_offset);
            let column = emit_bin_op(
                ctx,
                anchor,
                &loc,
                MirAddOp::get_concrete_op_info(),
                u64_ty,
                column_base,
                local_offset,
            );
            let logical = emit_bin_op(
                ctx,
                anchor,
                &loc,
                MirAddOp::get_concrete_op_info(),
                u64_ty,
                if m_band == 0 { row_zero } else { row_one },
                column,
            );
            let swizzled = emit_bin_op(
                ctx,
                anchor,
                &loc,
                MirBitXorOp::get_concrete_op_info(),
                u64_ty,
                logical,
                swizzle,
            );
            let physical = if n_offset < 64 {
                swizzled
            } else {
                emit_bin_op(
                    ctx,
                    anchor,
                    &loc,
                    MirAddOp::get_concrete_op_info(),
                    u64_ty,
                    swizzled,
                    half_stride,
                )
            };
            let pointer = emit_op_before(
                ctx,
                anchor,
                &loc,
                MirPtrOffsetOp::get_concrete_op_info(),
                vec![base.get_type(ctx)],
                vec![base, physical],
            )
            .deref(ctx)
            .get_result(0);
            let byte_pointer = emit_cast_before(
                ctx,
                anchor,
                &loc,
                pointer,
                u8_ptr,
                MirCastKindAttr::PtrToPtr,
            );
            let cell = (m_band * 8 + n_slot) * 4;
            let top = CvtF16x2F32Op::build(ctx, accumulator[cell], accumulator[cell + 1]);
            top.deref_mut(ctx).set_loc(loc.clone());
            top.insert_before(ctx, anchor);
            request_generated_intrinsic_marker(ctx, top);
            let bottom = CvtF16x2F32Op::build(ctx, accumulator[cell + 2], accumulator[cell + 3]);
            bottom.deref_mut(ctx).set_loc(loc.clone());
            bottom.insert_before(ctx, anchor);
            request_generated_intrinsic_marker(ctx, bottom);
            let top_result = top.deref(ctx).get_result(0);
            let bottom_result = bottom.deref(ctx).get_result(0);
            let store = Operation::new(
                ctx,
                StmatrixM8n8X2Op::get_concrete_op_info(),
                vec![],
                vec![byte_pointer, top_result, bottom_result],
                vec![],
                0,
            );
            store.deref_mut(ctx).set_loc(loc.clone());
            store.insert_before(ctx, anchor);
            request_generated_intrinsic_marker(ctx, store);
        }
    }
}

fn emit_epilogue_sync(ctx: &mut Context, op: Ptr<Operation>) {
    let semantic = CuteEpilogueSyncOp::wrap(op);
    let phase = semantic
        .phase(ctx)
        .expect("verified epilogue sync has a phase");
    let barrier = semantic
        .barrier(ctx)
        .expect("verified epilogue sync has a barrier");
    let loc = op.deref(ctx).loc().clone();
    let id = emit_u32_const(ctx, op, &loc, barrier.barrier_id);
    let participants = emit_u32_const(
        ctx,
        op,
        &loc,
        barrier
            .participant_threads()
            .expect("verified epilogue barrier has a thread count"),
    );
    if phase == CuteEpilogueSyncPhaseAttr::ReadyForTma {
        let fence = FenceProxyAsyncSharedCtaOp::build(ctx);
        fence.deref_mut(ctx).set_loc(loc.clone());
        fence.insert_before(ctx, op);
        request_generated_intrinsic_marker(ctx, fence);
    }
    let leaf = Operation::new(
        ctx,
        BarrierCtaSyncAlignedCountOp::get_concrete_op_info(),
        vec![],
        vec![id, participants],
        vec![],
        0,
    );
    leaf.deref_mut(ctx).set_loc(loc);
    leaf.insert_before(ctx, op);
    request_generated_intrinsic_marker(ctx, leaf);
}

fn emit_epilogue_store_wait(ctx: &mut Context, op: Ptr<Operation>, pending: u32) {
    let loc = op.deref(ctx).loc().clone();
    let pending = emit_u32_const(ctx, op, &loc, pending);
    let leaf = Operation::new(
        ctx,
        CpAsyncBulkWaitGroupReadOp::get_concrete_op_info(),
        vec![],
        vec![pending],
        vec![],
        0,
    );
    leaf.deref_mut(ctx).set_loc(loc);
    leaf.insert_before(ctx, op);
    request_generated_intrinsic_marker(ctx, leaf);
}

fn emit_epilogue_store_commit(ctx: &mut Context, op: Ptr<Operation>) {
    let leaf = Operation::new(
        ctx,
        CpAsyncBulkCommitGroupOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    leaf.deref_mut(ctx).set_loc(op.deref(ctx).loc().clone());
    leaf.insert_before(ctx, op);
    request_generated_intrinsic_marker(ctx, leaf);
}

/// Expand the complete v0 shared-memory epilogue back to the proven leaves.
///
/// ```text
/// 64xf32 -> 32 packed conversions -> 16 stmatrix
///              -> proxy fence -> counted sync
/// two shared halves -> two existing TMA-store leaves -> one commit
/// ```
///
/// The static checks prove two straight-line pieces:
///
/// ```text
/// same shared tile -> writers -> fence -> Ready
/// half 0 -> store(column) -> half 1 -> store(column + 1) -> commit
/// ```
///
/// They also prove one descriptor, one row, and no hidden memory effect in
/// either piece. Runtime control still owns the links between those pieces:
/// `warp < 8`, `lane < 32`, the `tid == 0` issuer branch, acquire/tail being
/// reached in the same loop story, counted-barrier participation across
/// threads, and both output columns being inside the TMA descriptor (so the
/// recognized unsigned `column + 1` cannot wrap).
fn lower_epilogue_to_legacy_cute(
    ctx: &mut Context,
    module: Ptr<Operation>,
) -> Result<(), ExpandError> {
    let mut all_ops = Vec::new();
    collect_ops(ctx, module, &mut all_ops);
    let Some(plan) = preflight_epilogue(ctx, &all_ops)? else {
        return Ok(());
    };
    let store_leaves = materialize_epilogue_store_leaves(ctx, &plan.stores)?;

    // No module mutation occurs above this line.
    emit_epilogue_store_fragment(ctx, &plan);
    emit_epilogue_sync(ctx, plan.reusable_sync);
    emit_epilogue_sync(ctx, plan.ready_sync);
    let acquire_pending = CuteTmaStoreAcquireOp::wrap(plan.acquire)
        .pipeline(ctx)
        .and_then(|pipeline| pipeline.max_pending())
        .expect("verified v0 store pipeline has a wait count");
    emit_epilogue_store_wait(ctx, plan.acquire, acquire_pending);
    emit_epilogue_store_commit(ctx, plan.commit);
    emit_epilogue_store_wait(ctx, plan.tail, 0);

    let mut rewriter = IRRewriter::<Recorder>::default();
    for (store, leaf) in plan.stores.iter().zip(store_leaves) {
        leaf.insert_before(ctx, store.semantic_store);
        rewriter.erase_operation(ctx, store.semantic_store);
    }
    for half in plan.halves {
        let semantic = CuteEpilogueHalfOp::wrap(half);
        let loc = half.deref(ctx).loc().clone();
        let index = semantic
            .half(ctx)
            .expect("verified epilogue half has an index")
            .0;
        let base = if index == 0 {
            semantic.full_base(ctx)
        } else {
            let half_offset = emit_u64_const_bits(ctx, half, &loc, 8192);
            emit_op_before(
                ctx,
                half,
                &loc,
                MirPtrOffsetOp::get_concrete_op_info(),
                vec![semantic.full_base(ctx).get_type(ctx)],
                vec![semantic.full_base(ctx), half_offset],
            )
            .deref(ctx)
            .get_result(0)
        };
        let capacity = emit_u64_const_bits(ctx, half, &loc, 8192);
        rewriter.replace_operation_with_values(ctx, half, vec![base, capacity]);
    }

    rewriter.erase_operation(ctx, plan.store_fragment);
    rewriter.erase_operation(ctx, plan.reusable_sync);
    rewriter.erase_operation(ctx, plan.ready_sync);
    rewriter.erase_operation(ctx, plan.acquire);
    rewriter.erase_operation(ctx, plan.commit);
    rewriter.erase_operation(ctx, plan.tail);
    let warp_slice_inputs = (0..3)
        .map(|index| plan.warp_slice.deref(ctx).get_operand(index))
        .collect();
    rewriter.replace_operation_with_values(ctx, plan.warp_slice, warp_slice_inputs);
    let overlay_input = plan.overlay.deref(ctx).get_operand(0);
    rewriter.replace_operation_with_values(ctx, plan.overlay, vec![overlay_input]);

    let mut view_producers = Vec::new();
    for store in &plan.stores {
        unique_operation_push(&mut view_producers, store.source_view);
        unique_operation_push(&mut view_producers, store.destination_view);
    }
    for producer in view_producers {
        if producer.deref(ctx).get_result(0).is_used(ctx) {
            return Err(ExpandError::Invalid(
                "epilogue TMA view still has a user after its two stores were lowered".into(),
            ));
        }
        rewriter.erase_operation(ctx, producer);
    }
    Ok(())
}

/// One fully checked semantic TMA copy and the data needed to build its leaf.
///
/// Keeping preflight allocation-free matters to the whole-module validator.
/// The later materializer either erases every pending leaf on error or
/// attaches all of them to the disposable clone before returning.
struct TmaSemanticCopyPlan {
    semantic_copy: Ptr<Operation>,
    leaf_operands: [Value; 5],
    leaf_layout: ComposedLayout,
    leaf_element: TypeHandle,
}

fn direct_tma_view_producer(
    ctx: &Context,
    value: Value,
    expected: &OpId,
    what: &str,
) -> Ptr<Operation> {
    let producer = value.defining_op().unwrap_or_else(|| {
        panic!("shared verifier requires a direct semantic TMA {what} producer")
    });
    let actual = Operation::get_opid(producer, ctx);
    assert!(
        actual == *expected,
        "shared verifier requires the expected semantic TMA {what} producer"
    );
    producer
}

fn preflight_semantic_tma_copy(
    ctx: &Context,
    semantic_copy: Ptr<Operation>,
) -> Result<TmaSemanticCopyPlan, ExpandError> {
    let copy = CuteTmaCopy2dOp::wrap(semantic_copy);
    let source_producer = direct_tma_view_producer(
        ctx,
        copy.source(ctx),
        &CuteTmaGmemViewOp::get_opid_static(),
        "source",
    );
    let destination_producer = direct_tma_view_producer(
        ctx,
        copy.destination(ctx),
        &CuteTmaSmemViewOp::get_opid_static(),
        "destination",
    );
    let source = CuteTmaGmemViewOp::wrap(source_producer);
    let destination = CuteTmaSmemViewOp::wrap(destination_producer);

    let (element, layout) = {
        let source_type = copy.source(ctx).get_type(ctx);
        let source_type_ref = source_type.deref(ctx);
        let source_view = source_type_ref
            .downcast_ref::<CuteTmaViewType>()
            .ok_or_else(|| {
                ExpandError::Invalid("semantic TMA source has no TMA view type".into())
            })?;
        let element = source_view.element(ctx).ok_or_else(|| {
            ExpandError::Invalid("semantic TMA source has no physical carrier type".into())
        })?;
        (element, source_view.smem_layout.0.clone())
    };
    let operands = [
        destination.base(ctx),
        copy.completion_barrier(ctx),
        source.descriptor(ctx),
        copy.tile_row(ctx),
        copy.tile_column(ctx),
    ];
    Ok(TmaSemanticCopyPlan {
        semantic_copy,
        leaf_operands: operands,
        leaf_layout: layout,
        leaf_element: element,
    })
}

fn materialize_semantic_tma_leaves(
    ctx: &mut Context,
    plans: &[TmaSemanticCopyPlan],
) -> Result<Vec<Ptr<Operation>>, ExpandError> {
    let mut leaves = Vec::with_capacity(plans.len());
    for plan in plans {
        let leaf = CuteTmaLoad2dOp::new(
            ctx,
            plan.leaf_operands,
            plan.leaf_layout.clone(),
            plan.leaf_element,
        );
        let leaf_op = leaf.get_operation();
        leaf_op
            .deref_mut(ctx)
            .set_loc(plan.semantic_copy.deref(ctx).loc().clone());
        if let Err(error) = leaf.verify(ctx) {
            Operation::erase(leaf_op, ctx);
            for pending in leaves.drain(..) {
                Operation::erase(pending, ctx);
            }
            return Err(ExpandError::Invalid(format!(
                "semantic `cute.tma_copy_2d` cannot become the existing TMA leaf: {error}"
            )));
        }
        leaves.push(leaf_op);
    }
    Ok(leaves)
}

/// Turn descriptor/shared transport views into the existing TMA leaf.
///
/// ```text
/// gmem_view(descriptor) ─┐
///                        ├─ tma_copy_2d(row, col, barrier)
/// smem_view(base, cap) ──┘
///               │
///               ▼
/// copy_tma_2d(base, barrier, descriptor, row, col)
/// ```
///
/// `capacity` is a runtime safety promise on the shared view. The legacy leaf
/// has no capacity operand, so this lowering preserves the existing contract:
/// the selected allocation must contain the complete typed tile.
fn lower_tma_views_to_legacy_cute(
    ctx: &mut Context,
    module: Ptr<Operation>,
) -> Result<(), ExpandError> {
    let mut all_ops = Vec::new();
    collect_ops(ctx, module, &mut all_ops);
    let gmem_id = CuteTmaGmemViewOp::get_opid_static();
    let smem_id = CuteTmaSmemViewOp::get_opid_static();
    let copy_id = CuteTmaCopy2dOp::get_opid_static();

    let mut gmem_views = Vec::new();
    let mut smem_views = Vec::new();
    let mut copies = Vec::new();
    for op in &all_ops {
        let opid = Operation::get_opid(*op, ctx);
        if opid == gmem_id {
            gmem_views.push(*op);
        } else if opid == smem_id {
            smem_views.push(*op);
        } else if opid == copy_id {
            copies.push(*op);
        }
    }
    if gmem_views.is_empty() && smem_views.is_empty() && copies.is_empty() {
        return Ok(());
    }
    // Resolve every prospective leaf without allocating an operation. This
    // keeps a failed validation clone self-contained and fully erasable.
    let mut plans = Vec::with_capacity(copies.len());
    for copy in &copies {
        plans.push(preflight_semantic_tma_copy(ctx, *copy)?);
    }
    let leaves = materialize_semantic_tma_leaves(ctx, &plans)?;

    let mut rewriter = IRRewriter::<Recorder>::default();
    for (plan, leaf) in plans.into_iter().zip(leaves) {
        leaf.insert_before(ctx, plan.semantic_copy);
        rewriter.erase_operation(ctx, plan.semantic_copy);
    }
    for layer in [&smem_views, &gmem_views] {
        for producer in layer {
            assert!(
                !producer.deref(ctx).get_result(0).is_used(ctx),
                "semantic TMA preflight missed a live view use"
            );
            rewriter.erase_operation(ctx, *producer);
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MmaRegisterKind {
    U32,
    F32,
}

/// The exact Rust aggregate shape around one ordinary register carrier.
///
/// The semantic ops never introduce a fragment ABI. They keep arrays,
/// tuples, structs, and empty marker structs exactly as MIR imported them.
/// This tree is built during preflight, then reused to extract and rebuild
/// those same carriers without any late type discovery.
#[derive(Clone)]
struct MmaCarrierLayout {
    ty: TypeHandle,
    kind: MmaCarrierLayoutKind,
}

#[derive(Clone)]
enum MmaCarrierLayoutKind {
    Register(MmaRegisterKind),
    Array(Vec<MmaCarrierLayout>),
    Tuple(Vec<MmaCarrierLayout>),
    Struct(Vec<MmaCarrierLayout>),
}

impl MmaCarrierLayout {
    fn register_count(&self, kind: MmaRegisterKind) -> u64 {
        match &self.kind {
            MmaCarrierLayoutKind::Register(found) => u64::from(*found == kind),
            MmaCarrierLayoutKind::Array(children)
            | MmaCarrierLayoutKind::Tuple(children)
            | MmaCarrierLayoutKind::Struct(children) => children
                .iter()
                .map(|child| child.register_count(kind))
                .sum(),
        }
    }
}

fn mma_carrier_layout(
    ctx: &Context,
    ty: TypeHandle,
    depth: usize,
) -> Result<MmaCarrierLayout, ExpandError> {
    if depth > 64 {
        return Err(ExpandError::Invalid(
            "MMA carrier aggregate is unexpectedly deep".into(),
        ));
    }
    let ty_ref = ty.deref(ctx);
    let kind = if let Some(integer) = ty_ref.downcast_ref::<IntegerType>() {
        if integer.width() != 32 || !integer.is_unsigned() {
            return Err(ExpandError::Invalid(
                "MMA carriers may contain only unsigned u32 or f32 registers".into(),
            ));
        }
        MmaCarrierLayoutKind::Register(MmaRegisterKind::U32)
    } else if ty_ref.downcast_ref::<FP32Type>().is_some() {
        MmaCarrierLayoutKind::Register(MmaRegisterKind::F32)
    } else if let Some(array) = ty_ref.downcast_ref::<MirArrayType>() {
        let child = mma_carrier_layout(ctx, array.element_ty, depth + 1)?;
        let count = usize::try_from(array.size)
            .map_err(|_| ExpandError::Invalid("MMA carrier array is too large to expand".into()))?;
        MmaCarrierLayoutKind::Array(vec![child; count])
    } else if let Some(tuple) = ty_ref.downcast_ref::<MirTupleType>() {
        MmaCarrierLayoutKind::Tuple(
            tuple
                .types
                .iter()
                .map(|child| mma_carrier_layout(ctx, *child, depth + 1))
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else if let Some(structure) = ty_ref.downcast_ref::<MirStructType>() {
        MmaCarrierLayoutKind::Struct(
            structure
                .field_types
                .iter()
                .map(|child| mma_carrier_layout(ctx, *child, depth + 1))
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else {
        return Err(ExpandError::Invalid(
            "MMA carrier must be an ordinary u32/f32 array, tuple, or struct".into(),
        ));
    };
    Ok(MmaCarrierLayout { ty, kind })
}

fn preflight_mma_carrier(
    ctx: &Context,
    value: Value,
    kind: MmaRegisterKind,
    count: u64,
    label: &str,
) -> Result<MmaCarrierLayout, ExpandError> {
    let layout = mma_carrier_layout(ctx, value.get_type(ctx), 0)?;
    let other = match kind {
        MmaRegisterKind::U32 => MmaRegisterKind::F32,
        MmaRegisterKind::F32 => MmaRegisterKind::U32,
    };
    if layout.register_count(kind) != count || layout.register_count(other) != 0 {
        return Err(ExpandError::Invalid(format!(
            "{label} must contain exactly {count} {kind:?} registers"
        )));
    }
    Ok(layout)
}

#[derive(Clone)]
struct MmaSmemLayoutPlan {
    rows: i64,
    leaves: Vec<ModeLeaf>,
    byte_offset: i64,
    swizzle: Swizzle,
    mutable: bool,
    address_space: u32,
}

fn preflight_mma_smem_layout(
    ctx: &Context,
    base: Value,
    view: &CuteSmemTensorType,
    label: &str,
) -> Result<MmaSmemLayoutPlan, ExpandError> {
    let tensor = view
        .tensor_view(ctx)
        .ok_or_else(|| ExpandError::Invalid(format!("{label} has no wrapped tensor view")))?;
    let storage_bytes = tensor
        .storage_bytes(ctx)
        .ok_or_else(|| ExpandError::Invalid(format!("{label} storage width is not known")))?;
    let storage_bytes = i64::try_from(storage_bytes)
        .map_err(|_| ExpandError::Invalid(format!("{label} storage width is too large")))?;
    let byte_layout = view
        .placement
        .0
        .to_byte_offsets(storage_bytes)
        .map_err(|error| ExpandError::Invalid(format!("{label} has no byte layout: {error}")))?;
    let modes = byte_layout.inner().modes();
    if modes.len() != 2 {
        return Err(ExpandError::Invalid(format!(
            "{label} shared layout needs two modes"
        )));
    }
    let rows = modes[0]
        .checked_size()
        .ok_or_else(|| ExpandError::Invalid(format!("{label} row extent is invalid")))?;
    let columns = modes[1]
        .checked_size()
        .ok_or_else(|| ExpandError::Invalid(format!("{label} column extent is invalid")))?;
    cute_layout::validate_ldmatrix_source(&byte_layout, rows, columns).map_err(|error| {
        ExpandError::Invalid(format!("{label} is not loadable by ldmatrix: {error}"))
    })?;
    let (_, mutable, address_space) = pointer_fields(ctx, base, label)?;
    Ok(MmaSmemLayoutPlan {
        rows,
        leaves: mode_leaves(byte_layout.inner())?,
        byte_offset: byte_layout.offset(),
        swizzle: byte_layout.outer(),
        mutable,
        address_space,
    })
}

#[derive(Clone)]
enum SmemMmaSemanticPlan {
    Overlay {
        op: Ptr<Operation>,
    },
    Slice {
        op: Ptr<Operation>,
    },
    Fill {
        op: Ptr<Operation>,
        result: MmaCarrierLayout,
    },
    LoadScales {
        op: Ptr<Operation>,
        result: MmaCarrierLayout,
    },
    SliceK {
        op: Ptr<Operation>,
        input: MmaCarrierLayout,
        result: MmaCarrierLayout,
    },
    LoadA {
        op: Ptr<Operation>,
        smem: MmaSmemLayoutPlan,
        result: MmaCarrierLayout,
    },
    PartitionB {
        op: Ptr<Operation>,
    },
    Gemm {
        op: Ptr<Operation>,
        smem_b: MmaSmemLayoutPlan,
        a: MmaCarrierLayout,
        scales: MmaCarrierLayout,
        accumulator: MmaCarrierLayout,
        result: MmaCarrierLayout,
    },
}

impl SmemMmaSemanticPlan {
    fn op(&self) -> Ptr<Operation> {
        match self {
            Self::Overlay { op }
            | Self::Slice { op }
            | Self::Fill { op, .. }
            | Self::LoadScales { op, .. }
            | Self::SliceK { op, .. }
            | Self::LoadA { op, .. }
            | Self::PartitionB { op }
            | Self::Gemm { op, .. } => *op,
        }
    }
}

fn preflight_smem_mma_op(
    ctx: &Context,
    op: Ptr<Operation>,
    opid: &OpId,
) -> Result<SmemMmaSemanticPlan, ExpandError> {
    let operation = op.deref(ctx);
    let plan = if *opid == CuteSmemTensorOverlayOp::get_opid_static() {
        SmemMmaSemanticPlan::Overlay { op }
    } else if *opid == CuteTiledMmaSliceOp::get_opid_static() {
        SmemMmaSemanticPlan::Slice { op }
    } else if *opid == CuteFragmentFillOp::get_opid_static() {
        SmemMmaSemanticPlan::Fill {
            op,
            result: preflight_mma_carrier(
                ctx,
                operation.get_result(0),
                MmaRegisterKind::F32,
                64,
                "cute.fragment_fill result",
            )?,
        }
    } else if *opid == CuteMmaLoadScalesOp::get_opid_static() {
        SmemMmaSemanticPlan::LoadScales {
            op,
            result: preflight_mma_carrier(
                ctx,
                operation.get_result(0),
                MmaRegisterKind::U32,
                10,
                "cute.mma_load_scales result",
            )?,
        }
    } else if *opid == CuteFragmentSliceKOp::get_opid_static() {
        SmemMmaSemanticPlan::SliceK {
            op,
            input: preflight_mma_carrier(
                ctx,
                operation.get_operand(0),
                MmaRegisterKind::U32,
                10,
                "cute.fragment_slice_k input",
            )?,
            result: preflight_mma_carrier(
                ctx,
                operation.get_result(0),
                MmaRegisterKind::U32,
                10,
                "cute.fragment_slice_k result",
            )?,
        }
    } else if *opid == CuteMmaLoadAOp::get_opid_static() {
        let semantic = CuteMmaLoadAOp::wrap(op);
        let view = semantic.view(ctx).ok_or_else(|| {
            ExpandError::Invalid("cute.mma_load_a is missing its typed shared view".into())
        })?;
        SmemMmaSemanticPlan::LoadA {
            op,
            smem: preflight_mma_smem_layout(
                ctx,
                operation.get_operand(1),
                &view,
                "cute.mma_load_a source",
            )?,
            result: preflight_mma_carrier(
                ctx,
                operation.get_result(0),
                MmaRegisterKind::U32,
                8,
                "cute.mma_load_a result",
            )?,
        }
    } else if *opid == CuteMmaPartitionBOp::get_opid_static() {
        SmemMmaSemanticPlan::PartitionB { op }
    } else {
        let semantic = CuteTiledGemmOp::wrap(op);
        let view = semantic.b_view(ctx).ok_or_else(|| {
            ExpandError::Invalid("cute.tiled_gemm is missing its typed B view".into())
        })?;
        SmemMmaSemanticPlan::Gemm {
            op,
            smem_b: preflight_mma_smem_layout(
                ctx,
                operation.get_operand(2),
                &view,
                "cute.tiled_gemm B source",
            )?,
            a: preflight_mma_carrier(
                ctx,
                operation.get_operand(1),
                MmaRegisterKind::U32,
                8,
                "cute.tiled_gemm A input",
            )?,
            scales: preflight_mma_carrier(
                ctx,
                operation.get_operand(6),
                MmaRegisterKind::U32,
                10,
                "cute.tiled_gemm scale input",
            )?,
            accumulator: preflight_mma_carrier(
                ctx,
                operation.get_operand(7),
                MmaRegisterKind::F32,
                64,
                "cute.tiled_gemm accumulator input",
            )?,
            result: preflight_mma_carrier(
                ctx,
                operation.get_result(0),
                MmaRegisterKind::F32,
                64,
                "cute.tiled_gemm result",
            )?,
        }
    };
    if operation.get_parent_block().is_none() {
        return Err(ExpandError::Invalid(format!(
            "shared/MMA operation `{opid}` is not attached to a block"
        )));
    }
    Ok(plan)
}

fn emit_carrier_extracts(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    value: Value,
    layout: &MmaCarrierLayout,
    output: &mut Vec<Value>,
) {
    match &layout.kind {
        MmaCarrierLayoutKind::Register(_) => output.push(value),
        MmaCarrierLayoutKind::Array(children)
        | MmaCarrierLayoutKind::Tuple(children)
        | MmaCarrierLayoutKind::Struct(children) => {
            for (index, child) in children.iter().enumerate() {
                let extract = emit_op_before(
                    ctx,
                    anchor,
                    loc,
                    MirExtractFieldOp::get_concrete_op_info(),
                    vec![child.ty],
                    vec![value],
                );
                MirExtractFieldOp::new(extract).set_attr_index(ctx, FieldIndexAttr(index as u32));
                let extracted = extract.deref(ctx).get_result(0);
                emit_carrier_extracts(ctx, anchor, loc, extracted, child, output);
            }
        }
    }
}

fn emit_carrier_construct(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    layout: &MmaCarrierLayout,
    leaves: &[Value],
    cursor: &mut usize,
) -> Value {
    if matches!(layout.kind, MmaCarrierLayoutKind::Register(_)) {
        let value = leaves[*cursor];
        *cursor += 1;
        debug_assert_eq!(value.get_type(ctx), layout.ty);
        return value;
    }
    let children = match &layout.kind {
        MmaCarrierLayoutKind::Array(children)
        | MmaCarrierLayoutKind::Tuple(children)
        | MmaCarrierLayoutKind::Struct(children) => children,
        MmaCarrierLayoutKind::Register(_) => unreachable!(),
    };
    let operands = children
        .iter()
        .map(|child| emit_carrier_construct(ctx, anchor, loc, child, leaves, cursor))
        .collect();
    let info = match layout.kind {
        MmaCarrierLayoutKind::Array(_) => MirConstructArrayOp::get_concrete_op_info(),
        MmaCarrierLayoutKind::Tuple(_) => MirConstructTupleOp::get_concrete_op_info(),
        MmaCarrierLayoutKind::Struct(_) => MirConstructStructOp::get_concrete_op_info(),
        MmaCarrierLayoutKind::Register(_) => unreachable!(),
    };
    emit_op_before(ctx, anchor, loc, info, vec![layout.ty], operands)
        .deref(ctx)
        .get_result(0)
}

fn rebuild_mma_carrier(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    layout: &MmaCarrierLayout,
    leaves: &[Value],
) -> Value {
    let mut cursor = 0;
    let result = emit_carrier_construct(ctx, anchor, loc, layout, leaves, &mut cursor);
    assert_eq!(
        cursor,
        leaves.len(),
        "shared/MMA preflight produced a mismatched carrier leaf plan"
    );
    result
}

fn emit_scale_word_offset(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    row: Value,
) -> Value {
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let mask_31 = emit_u64_const_bits(ctx, anchor, loc, 31);
    let row_low = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirBitAndOp::get_concrete_op_info(),
        u64_ty,
        row,
        mask_31,
    );
    let four = emit_u64_const_bits(ctx, anchor, loc, 4);
    let low_term = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        row_low,
        four,
    );
    let shift_5 = emit_i32_const(ctx, anchor, loc, 5);
    let quadrant = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirShrOp::get_concrete_op_info(),
        u64_ty,
        row,
        shift_5,
    );
    let mask_3 = emit_u64_const_bits(ctx, anchor, loc, 3);
    let quadrant = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirBitAndOp::get_concrete_op_info(),
        u64_ty,
        quadrant,
        mask_3,
    );
    emit_bin_op(
        ctx,
        anchor,
        loc,
        MirAddOp::get_concrete_op_info(),
        u64_ty,
        low_term,
        quadrant,
    )
}

fn emit_shared_u32_load(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    base: Value,
    row: Value,
) -> Value {
    let offset = emit_scale_word_offset(ctx, anchor, loc, row);
    let pointer = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirPtrOffsetOp::get_concrete_op_info(),
        base.get_type(ctx),
        base,
        offset,
    );
    let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    let load = emit_op_before(
        ctx,
        anchor,
        loc,
        MirLoadOp::get_concrete_op_info(),
        vec![u32_ty],
        vec![pointer],
    );
    MirLoadOp::new(load).set_volatile(ctx, false);
    load.deref(ctx).get_result(0)
}

fn emit_mma_load_scales(ctx: &mut Context, op: Ptr<Operation>, result: &MmaCarrierLayout) -> Value {
    let loc = op.deref(ctx).loc().clone();
    let operation = op.deref(ctx);
    let lane = operation.get_operand(0);
    let scale_a = operation.get_operand(1);
    let scale_b = operation.get_operand(3);
    let warp_m = operation.get_operand(5);
    let warp_n = operation.get_operand(6);
    drop(operation);

    let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let shift_2 = emit_i32_const(ctx, op, &loc, 2);
    let q = emit_bin_op(
        ctx,
        op,
        &loc,
        MirShrOp::get_concrete_op_info(),
        u32_ty,
        lane,
        shift_2,
    );
    let mask_3 = emit_u32_const(ctx, op, &loc, 3);
    let r = emit_bin_op(
        ctx,
        op,
        &loc,
        MirBitAndOp::get_concrete_op_info(),
        u32_ty,
        lane,
        mask_3,
    );
    let mask_1 = emit_u32_const(ctx, op, &loc, 1);
    let parity = emit_bin_op(
        ctx,
        op,
        &loc,
        MirBitAndOp::get_concrete_op_info(),
        u32_ty,
        r,
        mask_1,
    );
    let eight = emit_u32_const(ctx, op, &loc, 8);
    let provider_high = emit_bin_op(
        ctx,
        op,
        &loc,
        MirMulOp::get_concrete_op_info(),
        u32_ty,
        parity,
        eight,
    );
    let a_provider = emit_bin_op(
        ctx,
        op,
        &loc,
        MirAddOp::get_concrete_op_info(),
        u32_ty,
        q,
        provider_high,
    );
    let a_provider = emit_cast_value(ctx, op, &loc, a_provider, u64_ty, MirCastKindAttr::IntToInt);
    let q = emit_cast_value(ctx, op, &loc, q, u64_ty, MirCastKindAttr::IntToInt);

    let sixteen = emit_u64_const_bits(ctx, op, &loc, 16);
    let a_band = emit_bin_op(
        ctx,
        op,
        &loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        warp_m,
        sixteen,
    );
    let a_row0 = emit_bin_op(
        ctx,
        op,
        &loc,
        MirAddOp::get_concrete_op_info(),
        u64_ty,
        a_band,
        a_provider,
    );
    let sixty_four = emit_u64_const_bits(ctx, op, &loc, 64);
    let a_row1 = emit_bin_op(
        ctx,
        op,
        &loc,
        MirAddOp::get_concrete_op_info(),
        u64_ty,
        a_row0,
        sixty_four,
    );
    let sixteen = emit_u64_const_bits(ctx, op, &loc, 16);
    let b_band = emit_bin_op(
        ctx,
        op,
        &loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        warp_n,
        sixteen,
    );
    let b_row0 = emit_bin_op(
        ctx,
        op,
        &loc,
        MirAddOp::get_concrete_op_info(),
        u64_ty,
        b_band,
        q,
    );

    let mut rows = vec![a_row0, a_row1, b_row0];
    for delta in [8_u64, 32, 40, 64, 72, 96, 104] {
        let delta = emit_u64_const_bits(ctx, op, &loc, delta);
        rows.push(emit_bin_op(
            ctx,
            op,
            &loc,
            MirAddOp::get_concrete_op_info(),
            u64_ty,
            b_row0,
            delta,
        ));
    }
    let words: Vec<_> = rows
        .into_iter()
        .enumerate()
        .map(|(index, row)| {
            emit_shared_u32_load(
                ctx,
                op,
                &loc,
                if index < 2 { scale_a } else { scale_b },
                row,
            )
        })
        .collect();
    rebuild_mma_carrier(ctx, op, &loc, result, &words)
}

fn emit_mma_slice_k(
    ctx: &mut Context,
    op: Ptr<Operation>,
    input: &MmaCarrierLayout,
    result: &MmaCarrierLayout,
) -> Value {
    let loc = op.deref(ctx).loc().clone();
    let stage = op.deref(ctx).get_operand(0);
    let k_half = op.deref(ctx).get_operand(1);
    let mut words = Vec::with_capacity(10);
    emit_carrier_extracts(ctx, op, &loc, stage, input, &mut words);
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signed).into();
    let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    let sixteen = emit_u64_const_bits(ctx, op, &loc, 16);
    let shift = emit_bin_op(
        ctx,
        op,
        &loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        k_half,
        sixteen,
    );
    let shift = emit_cast_value(ctx, op, &loc, shift, i32_ty, MirCastKindAttr::IntToInt);
    let mask = emit_u32_const(ctx, op, &loc, 0xffff);
    let selected: Vec<_> = words
        .into_iter()
        .map(|word| {
            let shifted = emit_bin_op(
                ctx,
                op,
                &loc,
                MirShrOp::get_concrete_op_info(),
                u32_ty,
                word,
                shift,
            );
            emit_bin_op(
                ctx,
                op,
                &loc,
                MirBitAndOp::get_concrete_op_info(),
                u32_ty,
                shifted,
                mask,
            )
        })
        .collect();
    rebuild_mma_carrier(ctx, op, &loc, result, &selected)
}

#[allow(clippy::too_many_arguments)]
fn emit_mma_ldmatrix_x4(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    smem: &MmaSmemLayoutPlan,
    base: Value,
    warp_tile_row: Value,
    warp_tile_column: Value,
    lane: Value,
) -> [Value; 4] {
    use dialect_nvvm::ops::{
        LdmatrixElementAttr, LdmatrixLayoutAttr, LdmatrixMultiplicityAttr, LdmatrixOp,
        LdmatrixShapeAttr, LdmatrixStateSpaceAttr,
    };

    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let lane64 = emit_cast_value(ctx, anchor, loc, lane, u64_ty, MirCastKindAttr::IntToInt);
    let eight = emit_u64_const_bits(ctx, anchor, loc, 8);
    let submatrix = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirDivOp::get_concrete_op_info(),
        u64_ty,
        lane64,
        eight,
    );
    let eight = emit_u64_const_bits(ctx, anchor, loc, 8);
    let row_in = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirRemOp::get_concrete_op_info(),
        u64_ty,
        lane64,
        eight,
    );
    let two = emit_u64_const_bits(ctx, anchor, loc, 2);
    let sub_parity = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirRemOp::get_concrete_op_info(),
        u64_ty,
        submatrix,
        two,
    );
    let eight = emit_u64_const_bits(ctx, anchor, loc, 8);
    let row_offset = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        sub_parity,
        eight,
    );
    let sixteen = emit_u64_const_bits(ctx, anchor, loc, 16);
    let row_base = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        warp_tile_row,
        sixteen,
    );
    let row = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirAddOp::get_concrete_op_info(),
        u64_ty,
        row_base,
        row_offset,
    );
    let row = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirAddOp::get_concrete_op_info(),
        u64_ty,
        row,
        row_in,
    );

    let sixteen = emit_u64_const_bits(ctx, anchor, loc, 16);
    let column_base = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        warp_tile_column,
        sixteen,
    );
    let two = emit_u64_const_bits(ctx, anchor, loc, 2);
    let sub_half = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirDivOp::get_concrete_op_info(),
        u64_ty,
        submatrix,
        two,
    );
    let eight = emit_u64_const_bits(ctx, anchor, loc, 8);
    let column_offset = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        sub_half,
        eight,
    );
    let column = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirAddOp::get_concrete_op_info(),
        u64_ty,
        column_base,
        column_offset,
    );
    let rows = emit_u64_const(ctx, anchor, loc, smem.rows);
    let column_term = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        column,
        rows,
    );
    let cell = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirAddOp::get_concrete_op_info(),
        u64_ty,
        row,
        column_term,
    );
    let byte_offset = emit_folded_smem_byte_offset(
        ctx,
        anchor,
        loc,
        u64_ty,
        &smem.leaves,
        smem.byte_offset,
        &smem.swizzle,
        cell,
    );
    let byte_pointer_ty: TypeHandle =
        MirPtrType::get(ctx, byte_ty, smem.mutable, smem.address_space).into();
    let byte_base = emit_cast_value(
        ctx,
        anchor,
        loc,
        base,
        byte_pointer_ty,
        MirCastKindAttr::PtrToPtr,
    );
    let byte_address = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirPtrOffsetOp::get_concrete_op_info(),
        byte_pointer_ty,
        byte_base,
        byte_offset,
    );
    let address_ty: TypeHandle =
        MirPtrType::get(ctx, u32_ty, smem.mutable, smem.address_space).into();
    let address = emit_cast_value(
        ctx,
        anchor,
        loc,
        byte_address,
        address_ty,
        MirCastKindAttr::PtrToPtr,
    );
    let transaction = LdmatrixOp::build(
        ctx,
        address,
        LdmatrixShapeAttr::M8n8,
        LdmatrixMultiplicityAttr::X4,
        LdmatrixLayoutAttr::Normal,
        LdmatrixElementAttr::B16,
        LdmatrixStateSpaceAttr::Shared,
    );
    transaction.deref_mut(ctx).set_loc(loc.clone());
    transaction.insert_before(ctx, anchor);
    request_generated_intrinsic_marker(ctx, transaction);
    std::array::from_fn(|index| transaction.deref(ctx).get_result(index))
}

fn emit_mma_load_a(
    ctx: &mut Context,
    op: Ptr<Operation>,
    smem: &MmaSmemLayoutPlan,
    result: &MmaCarrierLayout,
) -> Value {
    let loc = op.deref(ctx).loc().clone();
    let operation = op.deref(ctx);
    let lane = operation.get_operand(0);
    let base = operation.get_operand(1);
    let warp_m = operation.get_operand(3);
    let k_half = operation.get_operand(4);
    drop(operation);
    let four = emit_u64_const_bits(ctx, op, &loc, 4);
    let second_row = emit_bin_op(
        ctx,
        op,
        &loc,
        MirAddOp::get_concrete_op_info(),
        IntegerType::get(ctx, 64, Signedness::Unsigned).into(),
        warp_m,
        four,
    );
    let first = emit_mma_ldmatrix_x4(ctx, op, &loc, smem, base, warp_m, k_half, lane);
    let second = emit_mma_ldmatrix_x4(ctx, op, &loc, smem, base, second_row, k_half, lane);
    let mut registers = first.to_vec();
    registers.extend(second);
    rebuild_mma_carrier(ctx, op, &loc, result, &registers)
}

fn emit_mxf4_mma(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    accumulator: &[Value],
    a: &[Value],
    b: &[Value],
    scales: [Value; 2],
) -> [Value; 4] {
    use dialect_nvvm::ops::{
        RegisterMmaAccumulatorAttr, RegisterMmaElementAttr, RegisterMmaKindAttr,
        RegisterMmaLayoutAttr, RegisterMmaOp, RegisterMmaOperationAttr, RegisterMmaOverflowAttr,
        RegisterMmaShapeAttr,
    };

    debug_assert_eq!(accumulator.len(), 4);
    debug_assert_eq!(a.len(), 4);
    debug_assert_eq!(b.len(), 2);
    let zero0 = emit_u16_const(ctx, anchor, loc, 0);
    let zero1 = emit_u16_const(ctx, anchor, loc, 0);
    let zero2 = emit_u16_const(ctx, anchor, loc, 0);
    let zero3 = emit_u16_const(ctx, anchor, loc, 0);
    let mut operands = Vec::with_capacity(16);
    operands.extend_from_slice(accumulator);
    operands.extend_from_slice(a);
    operands.extend_from_slice(b);
    operands.extend([scales[0], zero0, zero1, scales[1], zero2, zero3]);
    let result_ty: TypeHandle = FP32Type::get(ctx).into();
    let operation = Operation::new(
        ctx,
        RegisterMmaOp::get_concrete_op_info(),
        vec![result_ty; 4],
        operands,
        vec![],
        0,
    );
    operation.deref_mut(ctx).set_loc(loc.clone());
    let mma = RegisterMmaOp::new(operation);
    mma.set_attr_nvvm_register_mma_shape(ctx, RegisterMmaShapeAttr::M16n8k64);
    mma.set_attr_nvvm_register_mma_operation(ctx, RegisterMmaOperationAttr::Multiply);
    mma.set_attr_nvvm_register_mma_kind(ctx, RegisterMmaKindAttr::Mxf4);
    mma.set_attr_nvvm_register_mma_accumulator(ctx, RegisterMmaAccumulatorAttr::F32);
    mma.set_attr_nvvm_register_mma_a_element(ctx, RegisterMmaElementAttr::E2m1);
    mma.set_attr_nvvm_register_mma_b_element(ctx, RegisterMmaElementAttr::E2m1);
    mma.set_attr_nvvm_register_mma_a_layout(ctx, RegisterMmaLayoutAttr::Row);
    mma.set_attr_nvvm_register_mma_b_layout(ctx, RegisterMmaLayoutAttr::Col);
    mma.set_attr_nvvm_register_mma_overflow(ctx, RegisterMmaOverflowAttr::NotApplicable);
    operation.insert_before(ctx, anchor);
    request_generated_intrinsic_marker(ctx, operation);
    std::array::from_fn(|index| operation.deref(ctx).get_result(index))
}

fn emit_tiled_gemm(
    ctx: &mut Context,
    op: Ptr<Operation>,
    smem_b: &MmaSmemLayoutPlan,
    a_layout: &MmaCarrierLayout,
    scales_layout: &MmaCarrierLayout,
    accumulator_layout: &MmaCarrierLayout,
    result_layout: &MmaCarrierLayout,
) -> Value {
    let loc = op.deref(ctx).loc().clone();
    let operation = op.deref(ctx);
    let lane = operation.get_operand(0);
    let a_carrier = operation.get_operand(1);
    let b_base = operation.get_operand(2);
    let warp_n = operation.get_operand(4);
    let k_half = operation.get_operand(5);
    let scale_carrier = operation.get_operand(6);
    let accumulator_carrier = operation.get_operand(7);
    drop(operation);

    let mut a = Vec::with_capacity(8);
    emit_carrier_extracts(ctx, op, &loc, a_carrier, a_layout, &mut a);
    let mut scales = Vec::with_capacity(10);
    emit_carrier_extracts(ctx, op, &loc, scale_carrier, scales_layout, &mut scales);
    let mut accumulator = Vec::with_capacity(64);
    emit_carrier_extracts(
        ctx,
        op,
        &loc,
        accumulator_carrier,
        accumulator_layout,
        &mut accumulator,
    );

    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    for pair in 0..4_usize {
        let pair_stride = emit_u64_const_bits(ctx, op, &loc, (pair * 2) as u64);
        let b_row = emit_bin_op(
            ctx,
            op,
            &loc,
            MirAddOp::get_concrete_op_info(),
            u64_ty,
            warp_n,
            pair_stride,
        );
        let loaded = emit_mma_ldmatrix_x4(ctx, op, &loc, smem_b, b_base, b_row, k_half, lane);
        let b_fragments = [[loaded[0], loaded[2]], [loaded[1], loaded[3]]];
        for m in 0..2_usize {
            for (n_in_pair, b_fragment) in b_fragments.iter().enumerate() {
                let n = pair * 2 + n_in_pair;
                let cell = m * 8 + n;
                let updated = emit_mxf4_mma(
                    ctx,
                    op,
                    &loc,
                    &accumulator[cell * 4..cell * 4 + 4],
                    &a[m * 4..m * 4 + 4],
                    b_fragment,
                    [scales[m], scales[2 + n]],
                );
                accumulator[cell * 4..cell * 4 + 4].copy_from_slice(&updated);
            }
        }
    }
    rebuild_mma_carrier(ctx, op, &loc, result_layout, &accumulator)
}

/// Expand the visible shared-tensor/MMA spine back to the proven SM120
/// register schedule.
///
/// ```text
/// overlay / tiled slice / B partition       ordinary identity values
/// scale stage                               10 shared u32 loads
/// K-half scale slice                        10 packed halfword selects
/// A fragment                                2 normal ldmatrix.x4
/// tiled GEMM                                4 x (B ldmatrix.x4 + 4 MMA)
/// ```
///
/// Every verifier, aggregate walk, and shared-layout check completes before
/// the first rewrite. Register carriers remain their original Rust-shaped
/// MIR aggregates; only the explanation around them disappears.
fn lower_smem_mma_to_mir(ctx: &mut Context, module: Ptr<Operation>) -> Result<(), ExpandError> {
    let mut all_ops = Vec::new();
    collect_ops(ctx, module, &mut all_ops);
    let semantic_ids = smem_mma_semantic_ids();
    let mut plans = Vec::new();
    for op in &all_ops {
        let opid = Operation::get_opid(*op, ctx);
        if !semantic_ids.contains(&opid) {
            continue;
        }
        plans.push(preflight_smem_mma_op(ctx, *op, &opid)?);
    }
    if plans.is_empty() {
        return Ok(());
    }

    // No module mutation occurs above this line.
    let mut rewriter = IRRewriter::<Recorder>::default();
    for plan in plans {
        let op = plan.op();
        match plan {
            SmemMmaSemanticPlan::Overlay { .. } => {
                let operation = op.deref(ctx);
                let values = vec![operation.get_operand(0), operation.get_operand(1)];
                drop(operation);
                rewriter.replace_operation_with_values(ctx, op, values);
            }
            SmemMmaSemanticPlan::Slice { .. } => {
                let lane = op.deref(ctx).get_operand(0);
                rewriter.replace_operation_with_values(ctx, op, vec![lane]);
            }
            SmemMmaSemanticPlan::Fill { result, .. } => {
                let loc = op.deref(ctx).loc().clone();
                let fill = op.deref(ctx).get_operand(0);
                let leaves = vec![fill; 64];
                let carrier = rebuild_mma_carrier(ctx, op, &loc, &result, &leaves);
                rewriter.replace_operation_with_values(ctx, op, vec![carrier]);
            }
            SmemMmaSemanticPlan::LoadScales { result, .. } => {
                let carrier = emit_mma_load_scales(ctx, op, &result);
                rewriter.replace_operation_with_values(ctx, op, vec![carrier]);
            }
            SmemMmaSemanticPlan::SliceK { input, result, .. } => {
                let carrier = emit_mma_slice_k(ctx, op, &input, &result);
                rewriter.replace_operation_with_values(ctx, op, vec![carrier]);
            }
            SmemMmaSemanticPlan::LoadA { smem, result, .. } => {
                let carrier = emit_mma_load_a(ctx, op, &smem, &result);
                rewriter.replace_operation_with_values(ctx, op, vec![carrier]);
            }
            SmemMmaSemanticPlan::PartitionB { .. } => {
                let operation = op.deref(ctx);
                let values = (0..4).map(|index| operation.get_operand(index)).collect();
                drop(operation);
                rewriter.replace_operation_with_values(ctx, op, values);
            }
            SmemMmaSemanticPlan::Gemm {
                smem_b,
                a,
                scales,
                accumulator,
                result,
                ..
            } => {
                let carrier = emit_tiled_gemm(ctx, op, &smem_b, &a, &scales, &accumulator, &result);
                rewriter.replace_operation_with_values(ctx, op, vec![carrier]);
            }
        }
    }

    Ok(())
}

/// Runtime pieces carried by one `cute.tensor_make_2d` result.
///
/// The type owns format/layout/alignment facts. These values are the changing
/// coordinates needed to reconstruct the current native address calculation.
#[derive(Clone, Copy)]
struct GemvTensor2DState {
    data: Value,
    #[allow(dead_code)]
    len: Value,
    rows: Value,
    k: Value,
    role: CuteTensorRoleAttr,
}

/// One direct block-scaled selection chain ending at a K=64 tile.
#[derive(Clone, Copy)]
struct GemvTileState {
    values: GemvTensor2DState,
    scales: GemvTensor2DState,
    batch: Value,
    row: Value,
    tile_index: Value,
    row_anchor: Ptr<Operation>,
    tile_anchor: Ptr<Operation>,
}

#[derive(Clone, Copy)]
struct GemvScaledState {
    values: GemvTensor2DState,
    scales: GemvTensor2DState,
    batch: Option<Value>,
    row: Option<Value>,
    tile_index: Option<Value>,
    row_anchor: Option<Ptr<Operation>>,
    tile_anchor: Option<Ptr<Operation>>,
}

#[derive(Clone, Copy)]
struct GemvFragmentPlan {
    load: Ptr<Operation>,
    tile: GemvTileState,
    value_alignment_bytes: u64,
    scale_alignment_bytes: u64,
}

#[derive(Clone, Copy)]
struct GemvDotPlan {
    op: Ptr<Operation>,
    matrix: GemvFragmentPlan,
    vector: GemvFragmentPlan,
    acc: Value,
}

fn resolve_gemv_tensor_2d(
    ctx: &Context,
    value: Value,
    depth: usize,
) -> Result<GemvTensor2DState, ExpandError> {
    if depth > 16 {
        return Err(ExpandError::Invalid(
            "GEMV tensor producer chain is unexpectedly deep".into(),
        ));
    }
    let Some(defining_op) = value.defining_op() else {
        return Err(ExpandError::Invalid(
            "GEMV tensor reached a block argument; Backend A v0 needs direct semantic SSA".into(),
        ));
    };
    let opid = Operation::get_opid(defining_op, ctx);
    if opid != CuteTensorMake2DOp::get_opid_static() {
        return Err(ExpandError::Invalid(format!(
            "GEMV tensor is produced by unsupported operation `{opid}`"
        )));
    }
    let op = defining_op.deref(ctx);
    let result_ty = value.get_type(ctx);
    let result_ty_ref = result_ty.deref(ctx);
    let view = result_ty_ref
        .downcast_ref::<CuteTensorViewType>()
        .ok_or_else(|| {
            ExpandError::Invalid("cute.tensor_make_2d result is not a tensor view".into())
        })?;
    Ok(GemvTensor2DState {
        data: op.get_operand(0),
        len: op.get_operand(1),
        rows: op.get_operand(2),
        k: op.get_operand(3),
        role: view.role,
    })
}

fn resolve_gemv_scaled_view(
    ctx: &Context,
    value: Value,
    depth: usize,
) -> Result<GemvScaledState, ExpandError> {
    if depth > 16 {
        return Err(ExpandError::Invalid(
            "GEMV scaled-view producer chain is unexpectedly deep".into(),
        ));
    }
    let Some(defining_op) = value.defining_op() else {
        return Err(ExpandError::Invalid(
            "GEMV scaled view reached a block argument; Backend A v0 needs direct semantic SSA"
                .into(),
        ));
    };
    let opid = Operation::get_opid(defining_op, ctx);
    let op = defining_op.deref(ctx);
    if opid == CuteScaledViewMakeOp::get_opid_static() {
        let values = resolve_gemv_tensor_2d(ctx, op.get_operand(0), depth + 1)?;
        let scales = resolve_gemv_tensor_2d(ctx, op.get_operand(1), depth + 1)?;
        if values.rows != scales.rows || values.k != scales.k {
            return Err(ExpandError::Invalid(
                "cute.scaled_view_make values and scales must use the same runtime rows and K operands"
                    .into(),
            ));
        }
        return Ok(GemvScaledState {
            values,
            scales,
            batch: None,
            row: None,
            tile_index: None,
            row_anchor: None,
            tile_anchor: None,
        });
    }
    if opid == CuteScaledViewRowOp::get_opid_static() {
        let mut state = resolve_gemv_scaled_view(ctx, op.get_operand(0), depth + 1)?;
        if state.batch.is_some() || state.row.is_some() {
            return Err(ExpandError::Invalid(
                "cute.scaled_view_row selected a row more than once".into(),
            ));
        }
        state.batch = Some(op.get_operand(1));
        state.row = Some(op.get_operand(2));
        state.row_anchor = Some(defining_op);
        return Ok(state);
    }
    if opid == CuteScaledViewKTileOp::get_opid_static() {
        let mut state = resolve_gemv_scaled_view(ctx, op.get_operand(0), depth + 1)?;
        if state.tile_index.is_some() {
            return Err(ExpandError::Invalid(
                "cute.scaled_view_k_tile selected a K tile more than once".into(),
            ));
        }
        state.tile_index = Some(op.get_operand(1));
        state.tile_anchor = Some(defining_op);
        return Ok(state);
    }
    Err(ExpandError::Invalid(format!(
        "scaled view is produced by unsupported operation `{opid}`"
    )))
}

fn resolve_gemv_tile(ctx: &Context, value: Value) -> Result<GemvTileState, ExpandError> {
    let state = resolve_gemv_scaled_view(ctx, value, 0)?;
    let (Some(batch), Some(row), Some(tile_index), Some(row_anchor), Some(tile_anchor)) = (
        state.batch,
        state.row,
        state.tile_index,
        state.row_anchor,
        state.tile_anchor,
    ) else {
        return Err(ExpandError::Invalid(
            "GEMV load needs a selected row and K tile".into(),
        ));
    };
    if state.values.role != state.scales.role {
        return Err(ExpandError::Invalid(
            "GEMV values and scales must keep the same role".into(),
        ));
    }
    Ok(GemvTileState {
        values: state.values,
        scales: state.scales,
        batch,
        row,
        tile_index,
        row_anchor,
        tile_anchor,
    })
}

fn resolve_gemv_fragment(ctx: &Context, value: Value) -> Result<GemvFragmentPlan, ExpandError> {
    let Some(load) = value.defining_op() else {
        return Err(ExpandError::Invalid(
            "GEMV fragment reached a block argument; Backend A v0 needs direct load -> dot SSA"
                .into(),
        ));
    };
    let opid = Operation::get_opid(load, ctx);
    if opid != CuteScaledViewLoadOp::get_opid_static() {
        return Err(ExpandError::Invalid(format!(
            "GEMV fragment is produced by unsupported operation `{opid}`"
        )));
    }
    let typed_load = CuteScaledViewLoadOp::wrap(load);
    let value_alignment_bytes = typed_load.promised_value_alignment(ctx).ok_or_else(|| {
        ExpandError::Invalid("cute.scaled_view_load is missing its value alignment promise".into())
    })?;
    let scale_alignment_bytes = typed_load.promised_scale_alignment(ctx).ok_or_else(|| {
        ExpandError::Invalid("cute.scaled_view_load is missing its scale alignment promise".into())
    })?;
    if value_alignment_bytes < 16 || scale_alignment_bytes < 4 {
        return Err(ExpandError::Invalid(
            "GEMV K=64 load needs value alignment >= 16 and scale alignment >= 4".into(),
        ));
    }
    let tile = resolve_gemv_tile(ctx, load.deref(ctx).get_operand(0))?;
    Ok(GemvFragmentPlan {
        load,
        tile,
        value_alignment_bytes,
        scale_alignment_bytes,
    })
}

/// Expand the preserved GEMV story into the same copies, packed conversions,
/// and ordered f32 arithmetic produced by the former inlined Rust bodies.
///
/// The first implementation deliberately accepts only a direct chain:
///
/// ```text
/// make_2d -> scaled_make -> row -> KTile<64> -> load -> dot
/// ```
///
/// MIR preparation must remove temporary ghost slots before this function.
/// The direct value/scale tensors must share the same `rows` and `K` SSA
/// values. Their lengths do not match because their storage formats differ.
///
/// The unsafe load still owns two runtime preconditions: its selected row and
/// K tile must be in bounds, and the backing slices must contain the 32 value
/// bytes plus four scale bytes read for that tile. Its alignment attributes
/// promise at least 16-byte value and four-byte scale addresses; this lowering
/// checks those promises before turning them into `cute.assume_div` facts.
fn lower_gemv_views_to_legacy_cute(
    ctx: &mut Context,
    module: Ptr<Operation>,
) -> Result<(), ExpandError> {
    let mut all_ops = Vec::new();
    collect_ops(ctx, module, &mut all_ops);

    let make_id = CuteTensorMake2DOp::get_opid_static();
    let scaled_make_id = CuteScaledViewMakeOp::get_opid_static();
    let row_id = CuteScaledViewRowOp::get_opid_static();
    let k_tile_id = CuteScaledViewKTileOp::get_opid_static();
    let load_id = CuteScaledViewLoadOp::get_opid_static();
    let dot_id = CuteDotOp::get_opid_static();

    let mut makes = Vec::new();
    let mut scaled_makes = Vec::new();
    let mut rows = Vec::new();
    let mut k_tiles = Vec::new();
    let mut loads = Vec::new();
    let mut dots = Vec::new();
    for op in &all_ops {
        let opid = Operation::get_opid(*op, ctx);
        if opid == make_id {
            makes.push(*op);
        } else if opid == scaled_make_id {
            scaled_makes.push(*op);
        } else if opid == row_id {
            rows.push(*op);
        } else if opid == k_tile_id {
            k_tiles.push(*op);
        } else if opid == load_id {
            loads.push(*op);
        } else if opid == dot_id {
            dots.push(*op);
        }
    }

    if makes.is_empty()
        && scaled_makes.is_empty()
        && rows.is_empty()
        && k_tiles.is_empty()
        && loads.is_empty()
        && dots.is_empty()
    {
        return Ok(());
    }

    let mut plans = Vec::with_capacity(dots.len());
    for dot in &dots {
        let op = dot.deref(ctx);
        plans.push(GemvDotPlan {
            op: *dot,
            matrix: resolve_gemv_fragment(ctx, op.get_operand(0))?,
            vector: resolve_gemv_fragment(ctx, op.get_operand(1))?,
            acc: op.get_operand(2),
        });
    }

    let mut rewriter = IRRewriter::<Recorder>::default();
    for plan in plans {
        let loc = plan.op.deref(ctx).loc().clone();
        let matrix = emit_gemv_fragment(ctx, plan.matrix);
        let vector = emit_gemv_fragment(ctx, plan.vector);
        let acc = emit_gemv_dot(ctx, plan.op, &loc, matrix, vector, plan.acc);
        rewriter.replace_operation_with_values(ctx, plan.op, vec![acc]);
    }

    // Dot operands are gone. Remove semantic producers from leaves to roots.
    for layer in [&loads, &k_tiles, &rows, &scaled_makes, &makes] {
        for producer in layer {
            assert!(
                !producer.deref(ctx).get_result(0).is_used(ctx),
                "preflight missed a live use of GEMV producer `{}`",
                Operation::get_opid(*producer, ctx)
            );
            rewriter.erase_operation(ctx, *producer);
        }
    }

    Ok(())
}

#[derive(Clone, Copy)]
struct GemvRowBases {
    value_row_base: Value,
    scale_row_base: Value,
}

#[derive(Clone, Copy)]
struct GemvTileBases {
    value_base: Value,
    scale_base: Value,
}

#[derive(Clone)]
struct EmittedGemvFragment {
    values_lo: Value,
    values_hi: Value,
    scales: [Value; 4],
}

fn emit_cast_value(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    value: Value,
    result_ty: TypeHandle,
    kind: MirCastKindAttr,
) -> Value {
    let op = emit_op_before(
        ctx,
        anchor,
        loc,
        MirCastOp::get_concrete_op_info(),
        vec![result_ty],
        vec![value],
    );
    MirCastOp::new(op).set_attr_cast_kind(ctx, kind);
    op.deref(ctx).get_result(0)
}

fn emit_ceil_div_u64(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    value: Value,
    divisor: u64,
) -> Value {
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let bool_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
    let divisor = emit_u64_const_bits(ctx, anchor, loc, divisor);
    let quotient = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirDivOp::get_concrete_op_info(),
        u64_ty,
        value,
        divisor,
    );
    let remainder = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirRemOp::get_concrete_op_info(),
        u64_ty,
        value,
        divisor,
    );
    let zero = emit_u64_const_bits(ctx, anchor, loc, 0);
    let has_remainder = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirNeOp::get_concrete_op_info(),
        bool_ty,
        remainder,
        zero,
    );
    let round_up = emit_cast_value(
        ctx,
        anchor,
        loc,
        has_remainder,
        u64_ty,
        MirCastKindAttr::IntToInt,
    );
    emit_bin_op(
        ctx,
        anchor,
        loc,
        MirAddOp::get_concrete_op_info(),
        u64_ty,
        quotient,
        round_up,
    )
}

/// Recreate `thread_row`: its arithmetic stays beside the semantic row op, so
/// loop-invariant row bases remain outside the K loop just as in the Rust body.
fn emit_gemv_row_bases(ctx: &mut Context, tile: GemvTileState) -> GemvRowBases {
    let anchor = tile.row_anchor;
    let loc = anchor.deref(ctx).loc().clone();
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();

    let two = emit_u64_const_bits(ctx, anchor, &loc, 2);
    let packed_k = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirDivOp::get_concrete_op_info(),
        u64_ty,
        tile.values.k,
        two,
    );
    let batch_rows = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        tile.batch,
        tile.values.rows,
    );
    let linear_row = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirAddOp::get_concrete_op_info(),
        u64_ty,
        batch_rows,
        tile.row,
    );
    let value_row_base = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        linear_row,
        packed_k,
    );

    // Blackwell's canonical [128 rows x 4 K groups] scale block.
    let rest_m = emit_ceil_div_u64(ctx, anchor, &loc, tile.scales.rows, 128);
    let rest_k = emit_ceil_div_u64(ctx, anchor, &loc, tile.scales.k, 64);
    let block_bytes = emit_u64_const_bits(ctx, anchor, &loc, 512);

    let batch_blocks = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        tile.batch,
        rest_m,
    );
    let batch_blocks = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        batch_blocks,
        rest_k,
    );
    let batch_bytes = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        batch_blocks,
        block_bytes,
    );

    let shift_7 = emit_i32_const(ctx, anchor, &loc, 7);
    let row_block = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirShrOp::get_concrete_op_info(),
        u64_ty,
        tile.row,
        shift_7,
    );
    let row_blocks = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        row_block,
        rest_k,
    );
    let row_block_bytes = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        row_blocks,
        block_bytes,
    );

    let mask_31 = emit_u64_const_bits(ctx, anchor, &loc, 31);
    let row_in_quadrant = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirBitAndOp::get_concrete_op_info(),
        u64_ty,
        tile.row,
        mask_31,
    );
    let shift_4 = emit_i32_const(ctx, anchor, &loc, 4);
    let row_in_quadrant_bytes = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirShlOp::get_concrete_op_info(),
        u64_ty,
        row_in_quadrant,
        shift_4,
    );

    let shift_5 = emit_i32_const(ctx, anchor, &loc, 5);
    let quadrant = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirShrOp::get_concrete_op_info(),
        u64_ty,
        tile.row,
        shift_5,
    );
    let mask_3 = emit_u64_const_bits(ctx, anchor, &loc, 3);
    let quadrant = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirBitAndOp::get_concrete_op_info(),
        u64_ty,
        quadrant,
        mask_3,
    );
    let shift_2 = emit_i32_const(ctx, anchor, &loc, 2);
    let quadrant_bytes = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirShlOp::get_concrete_op_info(),
        u64_ty,
        quadrant,
        shift_2,
    );

    let scale_row_base = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirAddOp::get_concrete_op_info(),
        u64_ty,
        batch_bytes,
        row_block_bytes,
    );
    let scale_row_base = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirAddOp::get_concrete_op_info(),
        u64_ty,
        scale_row_base,
        row_in_quadrant_bytes,
    );
    let scale_row_base = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirAddOp::get_concrete_op_info(),
        u64_ty,
        scale_row_base,
        quadrant_bytes,
    );

    GemvRowBases {
        value_row_base,
        scale_row_base,
    }
}

fn emit_gemv_tile_bases(
    ctx: &mut Context,
    tile: GemvTileState,
    row: GemvRowBases,
) -> GemvTileBases {
    let anchor = tile.tile_anchor;
    let loc = anchor.deref(ctx).loc().clone();
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let value_stride = emit_u64_const_bits(ctx, anchor, &loc, 32);
    let value_offset = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        tile.tile_index,
        value_stride,
    );
    let value_base = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirAddOp::get_concrete_op_info(),
        u64_ty,
        row.value_row_base,
        value_offset,
    );
    let scale_stride = emit_u64_const_bits(ctx, anchor, &loc, 512);
    let scale_offset = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirMulOp::get_concrete_op_info(),
        u64_ty,
        tile.tile_index,
        scale_stride,
    );
    let scale_base = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirAddOp::get_concrete_op_info(),
        u64_ty,
        row.scale_row_base,
        scale_offset,
    );
    GemvTileBases {
        value_base,
        scale_base,
    }
}

fn emit_assume_div(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    value: Value,
    divisor: u64,
) -> Value {
    let assume = CuteAssumeDivOp::new(ctx, value, divisor);
    let op = assume.get_operation();
    op.deref_mut(ctx).set_loc(loc.clone());
    op.insert_before(ctx, anchor);
    op.deref(ctx).get_result(0)
}

fn emit_gemv_vector_load(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    data: Value,
    base: Value,
    element_type: TypeHandle,
    width: u64,
) -> Value {
    let source = emit_op_before(
        ctx,
        anchor,
        loc,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![data.get_type(ctx)],
        vec![data, base],
    )
    .deref(ctx)
    .get_result(0);

    let width = u32::try_from(width).expect("GEMV fixed vector width fits u32");
    let vector_type: TypeHandle =
        llvm_types::VectorType::get(ctx, element_type, width, llvm_types::VectorTypeKind::Fixed)
            .into();
    let (source_mutable, source_address_space) = {
        let source_type = source.get_type(ctx);
        let source_type_ref = source_type.deref(ctx);
        let source_pointer = source_type_ref
            .downcast_ref::<MirPtrType>()
            .expect("GEMV tensor data must be a MIR pointer");
        (source_pointer.is_mutable, source_pointer.address_space)
    };
    let vector_pointer_type: TypeHandle =
        MirPtrType::get(ctx, vector_type, source_mutable, source_address_space).into();
    let vector_pointer = emit_cast_value(
        ctx,
        anchor,
        loc,
        source,
        vector_pointer_type,
        MirCastKindAttr::PtrToPtr,
    );
    let load = emit_op_before(
        ctx,
        anchor,
        loc,
        MirLoadOp::get_concrete_op_info(),
        vec![vector_type],
        vec![vector_pointer],
    );
    MirLoadOp::new(load).set_volatile(ctx, false);
    load.deref(ctx).get_result(0)
}

fn emit_extract_vector_element(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    vector: Value,
    index: u64,
) -> Value {
    let index = emit_u64_const_bits(ctx, anchor, loc, index);
    let extract = llvm_export::ops::ExtractElementOp::new(ctx, vector, index).get_operation();
    extract.deref_mut(ctx).set_loc(loc.clone());
    extract.insert_before(ctx, anchor);
    extract.deref(ctx).get_result(0)
}

fn emit_ue8m0_pair_to_f32(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    low: Value,
    high: Value,
) -> [Value; 2] {
    let u16_ty: TypeHandle = IntegerType::get(ctx, 16, Signedness::Unsigned).into();
    let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    let f32_ty: TypeHandle = FP32Type::get(ctx).into();
    let low = emit_cast_value(ctx, anchor, loc, low, u16_ty, MirCastKindAttr::IntToInt);
    let high = emit_cast_value(ctx, anchor, loc, high, u16_ty, MirCastKindAttr::IntToInt);
    let shift_8 = emit_i32_const(ctx, anchor, loc, 8);
    let high = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirShlOp::get_concrete_op_info(),
        u16_ty,
        high,
        shift_8,
    );
    let packed = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirBitOrOp::get_concrete_op_info(),
        u16_ty,
        low,
        high,
    );
    let converted = CvtRnBf16x2Ue8m0x2Op::build(ctx, packed);
    converted.deref_mut(ctx).set_loc(loc.clone());
    converted.insert_before(ctx, anchor);
    request_generated_intrinsic_marker(ctx, converted);
    let packed_bf16 = converted.deref(ctx).get_result(0);

    let shift_16 = emit_i32_const(ctx, anchor, loc, 16);
    let low_bits = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirShlOp::get_concrete_op_info(),
        u32_ty,
        packed_bf16,
        shift_16,
    );
    let low = emit_cast_value(
        ctx,
        anchor,
        loc,
        low_bits,
        f32_ty,
        MirCastKindAttr::Transmute,
    );
    let high_mask = emit_u32_const(ctx, anchor, loc, 0xffff_0000);
    let high_bits = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirBitAndOp::get_concrete_op_info(),
        u32_ty,
        packed_bf16,
        high_mask,
    );
    let high = emit_cast_value(
        ctx,
        anchor,
        loc,
        high_bits,
        f32_ty,
        MirCastKindAttr::Transmute,
    );
    [low, high]
}

fn emit_gemv_fragment(ctx: &mut Context, plan: GemvFragmentPlan) -> EmittedGemvFragment {
    let row_bases = emit_gemv_row_bases(ctx, plan.tile);
    let bases = emit_gemv_tile_bases(ctx, plan.tile, row_bases);
    let anchor = plan.load;
    let loc = anchor.deref(ctx).loc().clone();
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    // LLVM vectors use signless integer lanes. Packedness, not signedness,
    // gives these bytes their meaning; the conversion helpers widen them to
    // the unsigned MIR carriers they require.
    let i8_lane_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signless).into();

    // Match BlockScaledTile64::load: state both alignment facts first, then
    // values low/high, then the four scale bytes.
    let value_base = emit_assume_div(
        ctx,
        anchor,
        &loc,
        bases.value_base,
        plan.value_alignment_bytes,
    );
    let scale_base = emit_assume_div(
        ctx,
        anchor,
        &loc,
        bases.scale_base,
        plan.scale_alignment_bytes,
    );
    let values_lo = emit_gemv_vector_load(
        ctx,
        anchor,
        &loc,
        plan.tile.values.data,
        value_base,
        i8_lane_ty,
        16,
    );
    let sixteen = emit_u64_const_bits(ctx, anchor, &loc, 16);
    let value_high_base = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirAddOp::get_concrete_op_info(),
        u64_ty,
        value_base,
        sixteen,
    );
    let values_hi = emit_gemv_vector_load(
        ctx,
        anchor,
        &loc,
        plan.tile.values.data,
        value_high_base,
        i8_lane_ty,
        16,
    );
    let scale_bytes = emit_gemv_vector_load(
        ctx,
        anchor,
        &loc,
        plan.tile.scales.data,
        scale_base,
        i8_lane_ty,
        4,
    );

    let scale_0 = emit_extract_vector_element(ctx, anchor, &loc, scale_bytes, 0);
    let scale_1 = emit_extract_vector_element(ctx, anchor, &loc, scale_bytes, 1);
    let pair_01 = emit_ue8m0_pair_to_f32(ctx, anchor, &loc, scale_0, scale_1);
    let scale_2 = emit_extract_vector_element(ctx, anchor, &loc, scale_bytes, 2);
    let scale_3 = emit_extract_vector_element(ctx, anchor, &loc, scale_bytes, 3);
    let pair_23 = emit_ue8m0_pair_to_f32(ctx, anchor, &loc, scale_2, scale_3);

    EmittedGemvFragment {
        values_lo,
        values_hi,
        scales: [pair_01[0], pair_01[1], pair_23[0], pair_23[1]],
    }
}

fn emit_e2m1_pair_to_f32(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    packed_byte: Value,
) -> [Value; 2] {
    let u16_ty: TypeHandle = IntegerType::get(ctx, 16, Signedness::Unsigned).into();
    let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    let f16_ty: TypeHandle = MirFP16Type::get(ctx).into();
    let f32_ty: TypeHandle = FP32Type::get(ctx).into();
    let packed = emit_cast_value(
        ctx,
        anchor,
        loc,
        packed_byte,
        u16_ty,
        MirCastKindAttr::IntToInt,
    );
    let converted = CvtRnF16x2E2m1x2Op::build(ctx, packed);
    converted.deref_mut(ctx).set_loc(loc.clone());
    converted.insert_before(ctx, anchor);
    request_generated_intrinsic_marker(ctx, converted);
    let packed_f16 = converted.deref(ctx).get_result(0);

    // Keep the exact helper order: form both f16 lanes first, then widen low
    // and high to f32.
    let low_bits = emit_cast_value(
        ctx,
        anchor,
        loc,
        packed_f16,
        u16_ty,
        MirCastKindAttr::IntToInt,
    );
    let low_f16 = emit_cast_value(
        ctx,
        anchor,
        loc,
        low_bits,
        f16_ty,
        MirCastKindAttr::Transmute,
    );
    let shift_16 = emit_i32_const(ctx, anchor, loc, 16);
    let high_bits = emit_bin_op(
        ctx,
        anchor,
        loc,
        MirShrOp::get_concrete_op_info(),
        u32_ty,
        packed_f16,
        shift_16,
    );
    let high_bits = emit_cast_value(
        ctx,
        anchor,
        loc,
        high_bits,
        u16_ty,
        MirCastKindAttr::IntToInt,
    );
    let high_f16 = emit_cast_value(
        ctx,
        anchor,
        loc,
        high_bits,
        f16_ty,
        MirCastKindAttr::Transmute,
    );
    let low = emit_cast_value(
        ctx,
        anchor,
        loc,
        low_f16,
        f32_ty,
        MirCastKindAttr::FloatToFloat,
    );
    let high = emit_cast_value(
        ctx,
        anchor,
        loc,
        high_f16,
        f32_ty,
        MirCastKindAttr::FloatToFloat,
    );
    [low, high]
}

fn emit_gemv_value_pair(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    fragment: &EmittedGemvFragment,
    pair: usize,
) -> [Value; 2] {
    let (bytes, byte_index) = if pair < 16 {
        (fragment.values_lo, pair)
    } else {
        (fragment.values_hi, pair - 16)
    };
    let packed = emit_extract_vector_element(ctx, anchor, loc, bytes, byte_index as u64);
    emit_e2m1_pair_to_f32(ctx, anchor, loc, packed)
}

fn emit_gemv_dot(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    matrix: EmittedGemvFragment,
    vector: EmittedGemvFragment,
    mut acc: Value,
) -> Value {
    let f32_ty: TypeHandle = FP32Type::get(ctx).into();
    for group in 0..4 {
        let scale_a = matrix.scales[group];
        let scale_b = vector.scales[group];
        for pair in (group * 8)..(group * 8 + 8) {
            // The Rust helper evaluates and converts A before B.
            let a = emit_gemv_value_pair(ctx, anchor, loc, &matrix, pair);
            let b = emit_gemv_value_pair(ctx, anchor, loc, &vector, pair);
            for lane in 0..2 {
                let scaled_a = emit_bin_op(
                    ctx,
                    anchor,
                    loc,
                    MirMulOp::get_concrete_op_info(),
                    f32_ty,
                    a[lane],
                    scale_a,
                );
                let product = emit_bin_op(
                    ctx,
                    anchor,
                    loc,
                    MirMulOp::get_concrete_op_info(),
                    f32_ty,
                    scaled_a,
                    b[lane],
                );
                let scaled_b = emit_bin_op(
                    ctx,
                    anchor,
                    loc,
                    MirMulOp::get_concrete_op_info(),
                    f32_ty,
                    product,
                    scale_b,
                );
                acc = emit_bin_op(
                    ctx,
                    anchor,
                    loc,
                    MirAddOp::get_concrete_op_info(),
                    f32_ty,
                    acc,
                    scaled_b,
                );
            }
        }
    }
    acc
}

#[derive(Clone, Copy)]
enum TensorConsumerPlan {
    IsFull {
        op: Ptr<Operation>,
        state: TensorViewState,
    },
    Base {
        op: Ptr<Operation>,
        state: TensorViewState,
    },
    Load {
        op: Ptr<Operation>,
        state: TensorViewState,
        destination: Value,
        element_count: i64,
    },
    Store {
        op: Ptr<Operation>,
        state: TensorViewState,
        source: Value,
        element_count: i64,
    },
    StoreElement {
        op: Ptr<Operation>,
        state: TensorViewState,
        index: Value,
        value: Value,
    },
}

impl TensorConsumerPlan {
    fn op(self) -> Ptr<Operation> {
        match self {
            Self::IsFull { op, .. }
            | Self::Base { op, .. }
            | Self::Load { op, .. }
            | Self::Store { op, .. }
            | Self::StoreElement { op, .. } => op,
        }
    }
}

/// Lower the semantic tensor-view layer into the already proven `cute.copy`
/// leaf plus ordinary MIR arithmetic.
///
/// All tensor chains are verified and resolved into immutable plans before
/// the first rewrite. Dynamic safety facts remain caller contracts:
/// full-tile copies require `tensor_is_full == true` and the promised runtime
/// alignment; a scalar tail index must be inside the selected tile and tensor.
fn lower_tensor_views_to_legacy_cute(
    ctx: &mut Context,
    module: Ptr<Operation>,
) -> Result<(), ExpandError> {
    let mut all_ops = Vec::new();
    collect_ops(ctx, module, &mut all_ops);

    let make_id = CuteTensorMakeOp::get_opid_static();
    let zipped_id = CuteTensorZippedDivideOp::get_opid_static();
    let slice_id = CuteTensorSliceOp::get_opid_static();
    let is_full_id = CuteTensorIsFullOp::get_opid_static();
    let base_id = CuteTensorBaseOp::get_opid_static();
    let load_id = CuteTensorLoadIntoOp::get_opid_static();
    let store_id = CuteTensorStoreFromOp::get_opid_static();
    let store_element_id = CuteTensorStoreElementAbsOp::get_opid_static();
    let mut makes = Vec::new();
    let mut zipped_divides = Vec::new();
    let mut slices = Vec::new();
    let mut consumer_ops = Vec::new();
    for op in &all_ops {
        let opid = Operation::get_opid(*op, ctx);
        if opid == make_id {
            makes.push(*op);
        } else if opid == zipped_id {
            zipped_divides.push(*op);
        } else if opid == slice_id {
            slices.push(*op);
        } else if opid == is_full_id
            || opid == base_id
            || opid == load_id
            || opid == store_id
            || opid == store_element_id
        {
            consumer_ops.push(*op);
        }
    }

    let producers: Vec<_> = makes
        .iter()
        .chain(&zipped_divides)
        .chain(&slices)
        .copied()
        .collect();

    // Resolve every chain and every static conversion before emitting any
    // arithmetic or erasing an operation. An invalid second consumer cannot
    // leave a successfully rewritten first consumer behind.
    let mut consumers = Vec::with_capacity(consumer_ops.len());
    for consumer in consumer_ops {
        let opid = Operation::get_opid(consumer, ctx);
        let op = consumer.deref(ctx);
        if opid == is_full_id {
            consumers.push(TensorConsumerPlan::IsFull {
                op: consumer,
                state: tensor_tile_state(ctx, op.get_operand(0))?,
            });
        } else if opid == base_id {
            consumers.push(TensorConsumerPlan::Base {
                op: consumer,
                state: tensor_tile_state(ctx, op.get_operand(0))?,
            });
        } else if opid == load_id {
            let state = tensor_tile_state(ctx, op.get_operand(0))?;
            let element_count = i64::try_from(state.tile_size.expect("tile state checked"))
                .map_err(|_| ExpandError::Invalid("tensor load tile is too wide".into()))?;
            consumers.push(TensorConsumerPlan::Load {
                op: consumer,
                state,
                destination: op.get_operand(1),
                element_count,
            });
        } else if opid == store_id {
            let state = tensor_tile_state(ctx, op.get_operand(1))?;
            let element_count = i64::try_from(state.tile_size.expect("tile state checked"))
                .map_err(|_| ExpandError::Invalid("tensor store tile is too wide".into()))?;
            consumers.push(TensorConsumerPlan::Store {
                op: consumer,
                state,
                source: op.get_operand(0),
                element_count,
            });
        } else {
            consumers.push(TensorConsumerPlan::StoreElement {
                op: consumer,
                state: tensor_tile_state(ctx, op.get_operand(0))?,
                index: op.get_operand(1),
                value: op.get_operand(2),
            });
        }
    }

    if producers.is_empty() && consumers.is_empty() {
        return Ok(());
    }

    let mut rewriter = IRRewriter::<Recorder>::default();
    for plan in consumers {
        let consumer = plan.op();
        let loc = consumer.deref(ctx).loc().clone();
        if let TensorConsumerPlan::IsFull { state, .. } = plan {
            let base = emit_tensor_tile_base(ctx, consumer, &loc, state);
            let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
            let bool_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
            let width = emit_u64_const_bits(
                ctx,
                consumer,
                &loc,
                state.tile_size.expect("tile state checked"),
            );
            let in_range = emit_bin_op(
                ctx,
                consumer,
                &loc,
                MirLtOp::get_concrete_op_info(),
                bool_ty,
                base,
                state.len,
            );
            // Unsigned subtraction wraps when `base >= len`; `in_range` then
            // forces the final predicate false. No `base + width` overflow is
            // possible in this formulation.
            let remaining = emit_bin_op(
                ctx,
                consumer,
                &loc,
                MirSubOp::get_concrete_op_info(),
                u64_ty,
                state.len,
                base,
            );
            let enough = emit_bin_op(
                ctx,
                consumer,
                &loc,
                MirGeOp::get_concrete_op_info(),
                bool_ty,
                remaining,
                width,
            );
            let full = emit_bin_op(
                ctx,
                consumer,
                &loc,
                MirBitAndOp::get_concrete_op_info(),
                bool_ty,
                in_range,
                enough,
            );
            rewriter.replace_operation_with_values(ctx, consumer, vec![full]);
        } else if let TensorConsumerPlan::Base { state, .. } = plan {
            let base = emit_tensor_tile_base(ctx, consumer, &loc, state);
            rewriter.replace_operation_with_values(ctx, consumer, vec![base]);
        } else if let TensorConsumerPlan::Load {
            state,
            destination,
            element_count,
            ..
        } = plan
        {
            let base = emit_tensor_tile_base(ctx, consumer, &loc, state);
            let source = emit_tensor_element_pointer(ctx, consumer, &loc, state, base);
            let copy = CuteCopyOp::new(
                ctx,
                source,
                destination,
                crate::layout::Layout::contiguous(element_count),
                state.storage,
            );
            let copy_op = copy.get_operation();
            copy_op.deref_mut(ctx).set_loc(loc);
            copy_op.insert_before(ctx, consumer);
            rewriter.erase_operation(ctx, consumer);
        } else if let TensorConsumerPlan::Store {
            state,
            source,
            element_count,
            ..
        } = plan
        {
            let base = emit_tensor_tile_base(ctx, consumer, &loc, state);
            let destination = emit_tensor_element_pointer(ctx, consumer, &loc, state, base);
            let copy = CuteCopyOp::new(
                ctx,
                source,
                destination,
                crate::layout::Layout::contiguous(element_count),
                state.storage,
            );
            let copy_op = copy.get_operation();
            copy_op.deref_mut(ctx).set_loc(loc);
            copy_op.insert_before(ctx, consumer);
            rewriter.erase_operation(ctx, consumer);
        } else if let TensorConsumerPlan::StoreElement {
            state,
            index,
            value,
            ..
        } = plan
        {
            let destination = emit_tensor_element_pointer(ctx, consumer, &loc, state, index);
            let store = emit_op_before(
                ctx,
                consumer,
                &loc,
                MirStoreOp::get_concrete_op_info(),
                vec![],
                vec![destination, value],
            );
            MirStoreOp::new(store).set_volatile(ctx, false);
            rewriter.erase_operation(ctx, consumer);
        }
    }

    // Consumers are gone, so erase each semantic layer from leaves to roots.
    // Preflight proved that no other use exists; a live use here is an
    // internal invariant failure, not a recoverable half-rewrite error.
    for layer in [&slices, &zipped_divides, &makes] {
        for producer in layer {
            assert!(
                !producer.deref(ctx).get_result(0).is_used(ctx),
                "preflight missed a live use of tensor producer `{}`",
                Operation::get_opid(*producer, ctx)
            );
            rewriter.erase_operation(ctx, *producer);
        }
    }

    Ok(())
}

/// Everything needed to emit one copy after module-wide validation succeeds.
/// Keeping a plan separate from mutation prevents this failure mode:
///
/// ```text
/// copy 0 rewritten ──> copy 1 invalid ──> half-expanded module
/// ```
struct CopyPlan {
    op: Ptr<Operation>,
    src: Value,
    dst: Value,
    loc: Location,
    elem: TypeHandle,
    element_count: i64,
    src_mutable: bool,
    src_address_space: u32,
    dst_mutable: bool,
    dst_address_space: u32,
}

/// One `(extent, stride)` pair of a flattened layout mode, in the colex
/// order `crd2idx` walks (leftmost leaf changes fastest).
type ModeLeaf = (i64, i64);

/// Everything needed to emit one cooperative copy, computed and checked
/// before phase 2 touches any IR.
///
/// The static half of every address is folded here at compile time; only the
/// terms that depend on runtime values (`tidx`, `ld`, the tile coordinate,
/// the matrix extents) survive as emitted arithmetic:
///
/// ```text
/// cell(t, atom) = f(t) + atom_cell_offsets[atom]
///                 │
///                 └── divmod chain over thread_leaves, all constants static
/// ```
struct CopyG2sPlan {
    op: Ptr<Operation>,
    loc: Location,
    operands: Vec<Value>,
    atom_bytes: u32,
    elem_bytes: i64,
    /// Tile extents `[rows, columns]` from the validated plan.
    tile_rows: i64,
    tile_cols: i64,
    values_per_atom: i64,
    /// TV thread mode, flattened to `(extent, stride)` leaves.
    thread_leaves: Vec<ModeLeaf>,
    /// `tv(0, atom * values_per_atom)` per atom: the static cell base.
    atom_cell_offsets: Vec<i64>,
    /// Shared-memory inner byte layout, flattened to `(extent, stride)`.
    smem_leaves: Vec<ModeLeaf>,
    /// Composed byte offset added before the swizzle.
    smem_byte_offset: i64,
    /// Byte-unit swizzle applied last (bits == 0 means none).
    smem_swizzle: Swizzle,
    dst_mutable: bool,
    dst_address_space: u32,
}

/// An `assume_div` is erased only after future direct SSA consumers have had
/// a chance to inspect the complete fact list. Cooperative pitches use a
/// call-visible static divisor instead because their value lives in a struct.
struct AssumePlan {
    op: Ptr<Operation>,
    #[allow(dead_code)]
    result: Value,
    #[allow(dead_code)]
    divisor: u64,
}

/// Validate the backend-neutral semantic story, then lower it to the native
/// MIR/NVVM implementation. The shared verifier performs no cloning or native
/// expansion, so semantic failures leave the caller's module untouched.
pub fn expand_cute_ops(ctx: &mut Context, module: Ptr<Operation>) -> Result<(), ExpandError> {
    verify_cute_semantics(ctx, module)?;
    expand_cute_ops_impl(ctx, module)
}

fn expand_cute_ops_impl(ctx: &mut Context, module: Ptr<Operation>) -> Result<(), ExpandError> {
    // Keep the two layers separate:
    //
    // high-level tensor story ──► existing cute.copy leaves ──► MIR/NVVM
    //
    // The shared verifier has accepted the complete semantic story. The native
    // implementation now turns the visible scheduler into ordinary loop values, expands the
    // load-pipeline story, expands the output epilogue, exposes TMA leaves,
    // then removes the GEMV and elementwise view layers.
    // The proven leaf expansion below then runs unchanged.
    lower_scheduler_to_mir(ctx, module)?;
    lower_pipeline_to_mir(ctx, module)?;
    lower_epilogue_to_legacy_cute(ctx, module)?;
    lower_tma_views_to_legacy_cute(ctx, module)?;
    lower_smem_mma_to_mir(ctx, module)?;
    lower_gemv_views_to_legacy_cute(ctx, module)?;
    lower_tensor_views_to_legacy_cute(ctx, module)?;

    let mut ops = Vec::new();
    collect_ops(ctx, module, &mut ops);
    let copy_opid = CuteCopyOp::get_opid_static();
    let g2s_opid = CuteCopyG2SOp::get_opid_static();
    let ldmatrix_opid = CuteLdmatrixOp::get_opid_static();
    let tma_opid = CuteTmaLoad2dOp::get_opid_static();
    let tma_store_opid = CuteTmaStore2dOp::get_opid_static();
    let assume_opid = CuteAssumeDivOp::get_opid_static();

    // Phase 1: inspect every cute op and build immutable plans. Every static
    // and runtime-plan check finishes before phase 2 changes any IR. An
    // unknown cute op is a compiler gap, never something this pass may skip.
    let mut copies = Vec::new();
    let mut cooperative_copies = Vec::new();
    let mut matrix_loads = Vec::new();
    let mut tma_loads = Vec::new();
    let mut tma_stores = Vec::new();
    let mut assumptions = Vec::new();
    for op in ops {
        let opid = Operation::get_opid(op, ctx);
        if opid == copy_opid {
            copies.push(preflight_copy(ctx, op)?);
        } else if opid == g2s_opid {
            cooperative_copies.push(preflight_copy_g2s(ctx, op)?);
        } else if opid == ldmatrix_opid {
            matrix_loads.push(preflight_ldmatrix(ctx, op)?);
        } else if opid == tma_opid {
            tma_loads.push(preflight_tma_load(ctx, op)?);
        } else if opid == tma_store_opid {
            tma_stores.push(preflight_tma_store(ctx, op)?);
        } else if opid == assume_opid {
            assumptions.push(preflight_assume(ctx, op)?);
        } else if opid.dialect.to_string() == crate::CUTE_DIALECT_NAME {
            return Err(ExpandError::NotImplemented(format!(
                "unhandled operation `{opid}`"
            )));
        }
    }

    // Phase 2: every plan is valid, so rewriting cannot discover a late
    // semantic error and leave a partly-expanded module.
    let mut rewriter = IRRewriter::<Recorder>::default();
    for copy in copies {
        expand_copy(ctx, &mut rewriter, copy);
    }
    for cooperative in cooperative_copies {
        expand_copy_g2s(ctx, &mut rewriter, cooperative);
    }
    for load in matrix_loads {
        expand_ldmatrix(ctx, &mut rewriter, load);
    }
    for load in tma_loads {
        expand_tma_load(ctx, &mut rewriter, load);
    }
    for store in tma_stores {
        expand_tma_store(ctx, &mut rewriter, store);
    }
    for assumption in assumptions {
        // Read the live operand now, after earlier identity facts may have
        // rewritten it:
        //
        // %a = assume %x       erase %a -> %b now uses %x
        // %b = assume %a       erase %b -> read that updated %x
        //
        // Caching `%a` during preflight would leave a dangling SSA value.
        let input = assumption.op.deref(ctx).get_operand(0);
        rewriter.replace_operation_with_values(ctx, assumption.op, vec![input]);
    }

    let mut leftovers = Vec::new();
    collect_ops(ctx, module, &mut leftovers);
    if let Some(op) = leftovers
        .into_iter()
        .find(|op| Operation::get_opid(*op, ctx).dialect.to_string() == crate::CUTE_DIALECT_NAME)
    {
        return Err(ExpandError::Invalid(format!(
            "cute expansion left `{}` in the module",
            Operation::get_opid(op, ctx)
        )));
    }
    Ok(())
}

fn collect_ops(ctx: &Context, root: Ptr<Operation>, output: &mut Vec<Ptr<Operation>>) {
    output.push(root);
    let regions: Vec<_> = root.deref(ctx).regions().collect();
    for region in regions {
        let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
        for block in blocks {
            let children: Vec<_> = block.deref(ctx).iter(ctx).collect();
            for child in children {
                collect_ops(ctx, child, output);
            }
        }
    }
}

/// Byte size of a tile element type. v0 supports the scalar types the
/// vectorizer can handle; anything else is a loud gap, not a silent skip.
fn elem_byte_size(ctx: &Context, elem: TypeHandle) -> Option<i64> {
    let ty = elem.deref(ctx);
    if let Some(int_ty) = ty.downcast_ref::<pliron::builtin::types::IntegerType>() {
        let width = int_ty.width() as i64;
        return (width % 8 == 0).then_some(width / 8);
    }
    if ty
        .downcast_ref::<pliron::builtin::types::FP32Type>()
        .is_some()
    {
        return Some(4);
    }
    if ty
        .downcast_ref::<pliron::builtin::types::FP64Type>()
        .is_some()
    {
        return Some(8);
    }
    if ty
        .downcast_ref::<dialect_mir::types::MirFP16Type>()
        .is_some()
    {
        return Some(2);
    }
    None
}

fn checked_layout_size(layout: &crate::layout::Layout) -> Result<i64, ExpandError> {
    layout
        .shape
        .leaves()
        .into_iter()
        .try_fold(1_i64, |size, extent| {
            if extent <= 0 {
                return Err(ExpandError::Invalid(format!(
                    "layout shape extents must be positive, got {extent}"
                )));
            }
            size.checked_mul(extent).ok_or_else(|| {
                ExpandError::Invalid("layout element count overflows i64".to_string())
            })
        })
}

fn pointer_fields(
    ctx: &Context,
    value: Value,
    role: &str,
) -> Result<(TypeHandle, bool, u32), ExpandError> {
    let ty = value.get_type(ctx);
    let ty_ref = ty.deref(ctx);
    let ptr = ty_ref
        .downcast_ref::<MirPtrType>()
        .ok_or_else(|| ExpandError::Invalid(format!("cute.copy {role} is not a mir.ptr")))?;
    Ok((ptr.pointee, ptr.is_mutable, ptr.address_space))
}

fn preflight_copy(ctx: &Context, copy_ptr: Ptr<Operation>) -> Result<CopyPlan, ExpandError> {
    {
        let op = copy_ptr.deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 0 {
            return Err(ExpandError::Invalid(
                "cute.copy must have two operands and zero results".into(),
            ));
        }
    }
    let copy = CuteCopyOp::wrap(copy_ptr);

    let layout = copy
        .layout(ctx)
        .ok_or_else(|| ExpandError::Invalid("cute.copy without layout attribute".into()))?;
    let elem = copy
        .elem(ctx)
        .ok_or_else(|| ExpandError::Invalid("cute.copy without elem attribute".into()))?;
    let (src, dst, loc) = {
        let op = copy_ptr.deref(ctx);
        (op.get_operand(0), op.get_operand(1), op.loc())
    };

    let n = checked_layout_size(&layout)?;
    let esize = elem_byte_size(ctx, elem).ok_or_else(|| {
        ExpandError::NotImplemented(format!(
            "unsupported cute.copy element type {:?}",
            elem.deref(ctx)
        ))
    })?;
    let total = n.checked_mul(esize).ok_or_else(|| {
        ExpandError::Invalid(format!(
            "cute.copy byte width overflows: {n} elements x {esize} bytes"
        ))
    })?;
    // 4..=16 matches the documented device contract in cute-rs (tile.rs
    // module docs): the float4-style natural-alignment model starts at one
    // 32-bit word, and nothing narrower is tested or promised.
    if !(4..=16).contains(&total) || (total & (total - 1)) != 0 || n < 2 {
        return Err(ExpandError::NotImplemented(format!(
            "cute.copy tile of {n} x {esize}-byte elements ({total} bytes) is not a \
             supported vector transfer width (need power-of-2 total in 4..=16 and n >= 2)"
        )));
    }
    // Only small, width-eligible tiles reach this exact mapping check. This
    // avoids walking an attacker-sized textual layout merely to reject its
    // byte width afterward.
    if !layout.is_identity_map() {
        return Err(ExpandError::NotImplemented(format!(
            "non-contiguous tile layout {layout} in cute.copy"
        )));
    }

    let (src_pointee, src_mutable, src_address_space) = pointer_fields(ctx, src, "source")?;
    let (dst_pointee, dst_mutable, dst_address_space) = pointer_fields(ctx, dst, "destination")?;
    if src_pointee != elem || dst_pointee != elem {
        return Err(ExpandError::Invalid(
            "cute.copy pointer pointees must match its elem attribute".into(),
        ));
    }
    if !dst_mutable {
        return Err(ExpandError::Invalid(
            "cute.copy destination pointer is not mutable".into(),
        ));
    }

    Ok(CopyPlan {
        op: copy_ptr,
        src,
        dst,
        loc,
        elem,
        element_count: n,
        src_mutable,
        src_address_space,
        dst_mutable,
        dst_address_space,
    })
}

fn preflight_assume(ctx: &Context, op_ptr: Ptr<Operation>) -> Result<AssumePlan, ExpandError> {
    let op = op_ptr.deref(ctx);
    if op.get_num_operands() != 1 || op.get_num_results() != 1 {
        return Err(ExpandError::Invalid(
            "cute.assume_div must have one operand and one result".into(),
        ));
    }
    let assume = CuteAssumeDivOp::wrap(op_ptr);
    let divisor = assume
        .divisor(ctx)
        .ok_or_else(|| ExpandError::Invalid("cute.assume_div has no divisor".into()))?;
    if divisor == 0 {
        return Err(ExpandError::Invalid(
            "cute.assume_div divisor must be greater than zero".into(),
        ));
    }
    let input = op.get_operand(0);
    let result = op.get_result(0);
    if input.get_type(ctx) != result.get_type(ctx)
        || input
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<pliron::builtin::types::IntegerType>()
            .is_none()
    {
        return Err(ExpandError::Invalid(
            "cute.assume_div must be an integer identity".into(),
        ));
    }
    Ok(AssumePlan {
        op: op_ptr,
        result,
        divisor,
    })
}

fn expand_copy(ctx: &mut Context, rewriter: &mut IRRewriter<Recorder>, plan: CopyPlan) {
    let CopyPlan {
        op: copy_ptr,
        src,
        dst,
        loc,
        elem,
        element_count: n,
        src_mutable: src_mut,
        src_address_space: src_as,
        dst_mutable: dst_mut,
        dst_address_space: dst_as,
    } = plan;

    let vec_ty: TypeHandle =
        llvm_types::VectorType::get(ctx, elem, n as u32, llvm_types::VectorTypeKind::Fixed).into();

    let src_vec_ptr_ty: TypeHandle = MirPtrType::get(ctx, vec_ty, src_mut, src_as).into();
    let dst_vec_ptr_ty: TypeHandle = MirPtrType::get(ctx, vec_ty, dst_mut, dst_as).into();

    type OpInfo = (fn(Ptr<Operation>) -> pliron::op::OpObj, std::any::TypeId);
    let emit_before = |ctx: &mut Context,
                       info: OpInfo,
                       results: Vec<TypeHandle>,
                       operands: Vec<pliron::value::Value>|
     -> Ptr<Operation> {
        let op = Operation::new(ctx, info, results, operands, vec![], 0);
        op.deref_mut(ctx).set_loc(loc.clone());
        op.insert_before(ctx, copy_ptr);
        op
    };

    let src_cast = emit_before(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![src_vec_ptr_ty],
        vec![src],
    );
    MirCastOp::new(src_cast).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    let src_vec_ptr = src_cast.deref(ctx).get_result(0);

    let load = emit_before(
        ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![vec_ty],
        vec![src_vec_ptr],
    );
    MirLoadOp::new(load).set_volatile(ctx, false);
    let loaded = load.deref(ctx).get_result(0);

    let dst_cast = emit_before(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![dst_vec_ptr_ty],
        vec![dst],
    );
    MirCastOp::new(dst_cast).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    let dst_vec_ptr = dst_cast.deref(ctx).get_result(0);

    let store = emit_before(
        ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![dst_vec_ptr, loaded],
    );
    MirStoreOp::new(store).set_volatile(ctx, false);

    rewriter.erase_operation(ctx, copy_ptr);
}

/// `(constructor, type id)` pair identifying one concrete op kind.
type OpInfoPair = (fn(Ptr<Operation>) -> pliron::op::OpObj, std::any::TypeId);

/// Create one op with `loc`, inserted before `anchor`.
fn emit_op_before(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    info: OpInfoPair,
    results: Vec<TypeHandle>,
    operands: Vec<Value>,
) -> Ptr<Operation> {
    let op = Operation::new(ctx, info, results, operands, vec![], 0);
    op.deref_mut(ctx).set_loc(loc.clone());
    op.insert_before(ctx, anchor);
    op
}

fn emit_cast_before(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    value: Value,
    result: TypeHandle,
    kind: MirCastKindAttr,
) -> Value {
    let op = emit_op_before(
        ctx,
        anchor,
        loc,
        MirCastOp::get_concrete_op_info(),
        vec![result],
        vec![value],
    );
    MirCastOp::new(op).set_attr_cast_kind(ctx, kind);
    op.deref(ctx).get_result(0)
}

/// Emit one `u64` constant before `anchor` without narrowing through `i64`.
fn emit_u64_const_bits(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    value: u64,
) -> Value {
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let op = emit_op_before(
        ctx,
        anchor,
        loc,
        MirConstantOp::get_concrete_op_info(),
        vec![u64_ty],
        vec![],
    );
    let attr = IntegerAttr::new(
        IntegerType::get(ctx, 64, Signedness::Unsigned),
        APInt::from_u64(value, NonZeroUsize::new(64).unwrap()),
    );
    MirConstantOp::new(op).set_attr_value(ctx, attr);
    op.deref(ctx).get_result(0)
}

/// Emit one small `u64` constant before `anchor`.
fn emit_u64_const(ctx: &mut Context, anchor: Ptr<Operation>, loc: &Location, value: i64) -> Value {
    emit_u64_const_bits(ctx, anchor, loc, value as u64)
}

fn emit_u32_const(ctx: &mut Context, anchor: Ptr<Operation>, loc: &Location, value: u32) -> Value {
    let ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
    let op = emit_op_before(
        ctx,
        anchor,
        loc,
        MirConstantOp::get_concrete_op_info(),
        vec![ty.into()],
        vec![],
    );
    MirConstantOp::new(op).set_attr_value(
        ctx,
        IntegerAttr::new(
            ty,
            APInt::from_u64(u64::from(value), NonZeroUsize::new(32).unwrap()),
        ),
    );
    op.deref(ctx).get_result(0)
}

fn emit_u16_const(ctx: &mut Context, anchor: Ptr<Operation>, loc: &Location, value: u16) -> Value {
    let ty = IntegerType::get(ctx, 16, Signedness::Unsigned);
    let op = emit_op_before(
        ctx,
        anchor,
        loc,
        MirConstantOp::get_concrete_op_info(),
        vec![ty.into()],
        vec![],
    );
    MirConstantOp::new(op).set_attr_value(
        ctx,
        IntegerAttr::new(
            ty,
            APInt::from_u64(u64::from(value), NonZeroUsize::new(16).unwrap()),
        ),
    );
    op.deref(ctx).get_result(0)
}

fn emit_i32_const(ctx: &mut Context, anchor: Ptr<Operation>, loc: &Location, value: i32) -> Value {
    let ty = IntegerType::get(ctx, 32, Signedness::Signed);
    let op = emit_op_before(
        ctx,
        anchor,
        loc,
        MirConstantOp::get_concrete_op_info(),
        vec![ty.into()],
        vec![],
    );
    MirConstantOp::new(op).set_attr_value(
        ctx,
        IntegerAttr::new(
            ty,
            APInt::from_i64(i64::from(value), NonZeroUsize::new(32).unwrap()),
        ),
    );
    op.deref(ctx).get_result(0)
}

/// Emit one two-operand op before `anchor` and return its result.
fn emit_bin_op(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    info: OpInfoPair,
    ty: TypeHandle,
    a: Value,
    b: Value,
) -> Value {
    let op = emit_op_before(ctx, anchor, loc, info, vec![ty], vec![a, b]);
    op.deref(ctx).get_result(0)
}

/// Emit the folded shared-memory BYTE offset for one symbolic flat cell:
/// decompose `cell` over the byte layout's `(extent, stride)` leaves exactly
/// as `crd2idx` would (mod on every leaf but the last), add the composed
/// offset, then apply the swizzle's and/shift/xor when one is present.
#[allow(clippy::too_many_arguments)]
fn emit_folded_smem_byte_offset(
    ctx: &mut Context,
    anchor: Ptr<Operation>,
    loc: &Location,
    u64_ty: TypeHandle,
    leaves: &[ModeLeaf],
    base_offset: i64,
    swizzle: &Swizzle,
    cell: Value,
) -> Value {
    let add_info = MirAddOp::get_concrete_op_info();
    let mul_info = MirMulOp::get_concrete_op_info();
    let div_info = MirDivOp::get_concrete_op_info();
    let rem_info = MirRemOp::get_concrete_op_info();

    let mut remaining = cell;
    let mut offset = emit_u64_const(ctx, anchor, loc, base_offset);
    let leaf_count = leaves.len();
    for (index, &(extent, stride)) in leaves.iter().enumerate() {
        let is_last = index + 1 == leaf_count;
        let coordinate = if is_last {
            remaining
        } else {
            let extent_c = emit_u64_const(ctx, anchor, loc, extent);
            let coord = emit_bin_op(ctx, anchor, loc, rem_info, u64_ty, remaining, extent_c);
            let extent_c2 = emit_u64_const(ctx, anchor, loc, extent);
            remaining = emit_bin_op(ctx, anchor, loc, div_info, u64_ty, remaining, extent_c2);
            coord
        };
        if stride == 0 {
            continue;
        }
        let stride_c = emit_u64_const(ctx, anchor, loc, stride);
        let term = emit_bin_op(ctx, anchor, loc, mul_info, u64_ty, coordinate, stride_c);
        offset = emit_bin_op(ctx, anchor, loc, add_info, u64_ty, offset, term);
    }
    if swizzle.bits > 0 {
        let mask_c = emit_u64_const(ctx, anchor, loc, swizzle.y_mask());
        let masked = emit_bin_op(
            ctx,
            anchor,
            loc,
            MirBitAndOp::get_concrete_op_info(),
            u64_ty,
            offset,
            mask_c,
        );
        let distance = i64::from(swizzle.shift.unsigned_abs());
        let distance_c = emit_u64_const(ctx, anchor, loc, distance);
        let moved = if swizzle.shift >= 0 {
            emit_bin_op(
                ctx,
                anchor,
                loc,
                MirShrOp::get_concrete_op_info(),
                u64_ty,
                masked,
                distance_c,
            )
        } else {
            emit_bin_op(
                ctx,
                anchor,
                loc,
                MirShlOp::get_concrete_op_info(),
                u64_ty,
                masked,
                distance_c,
            )
        };
        offset = emit_bin_op(
            ctx,
            anchor,
            loc,
            MirBitXorOp::get_concrete_op_info(),
            u64_ty,
            offset,
            moved,
        );
    }
    offset
}

/// Flatten one layout mode into `(extent, stride)` leaves in the colex order
/// `crd2idx` decomposes an integer coordinate (leftmost leaf fastest).
fn mode_leaves(mode: &crate::layout::Layout) -> Result<Vec<ModeLeaf>, ExpandError> {
    let extents = mode.shape.leaves();
    let strides = mode.stride.leaves();
    if extents.len() != strides.len() {
        return Err(ExpandError::Invalid(
            "layout mode shape and stride leaf counts differ".into(),
        ));
    }
    if extents.iter().any(|&extent| extent <= 0) {
        return Err(ExpandError::Invalid(
            "layout mode extents must be positive".into(),
        ));
    }
    Ok(extents.into_iter().zip(strides).collect())
}

fn preflight_copy_g2s(ctx: &Context, op_ptr: Ptr<Operation>) -> Result<CopyG2sPlan, ExpandError> {
    {
        let op = op_ptr.deref(ctx);
        if op.get_num_operands() != 9 || op.get_num_results() != 0 {
            return Err(ExpandError::Invalid(
                "cute.copy_g2s must have nine operands and zero results".into(),
            ));
        }
    }
    let g2s = CuteCopyG2SOp::wrap(op_ptr);
    let missing = |what: &str| ExpandError::Invalid(format!("cute.copy_g2s without {what}"));
    let atom_bytes = g2s
        .get_attr_atom_bytes(ctx)
        .map(|attr| attr.0)
        .ok_or_else(|| missing("atom_bytes"))?;
    let thread_layout = g2s
        .get_attr_thread_layout(ctx)
        .map(|attr| attr.0.clone())
        .ok_or_else(|| missing("a thread layout"))?;
    let value_layout = g2s
        .get_attr_value_layout(ctx)
        .map(|attr| attr.0.clone())
        .ok_or_else(|| missing("a value layout"))?;
    let tile_layout = g2s
        .get_attr_tile_layout(ctx)
        .map(|attr| attr.0.clone())
        .ok_or_else(|| missing("a tile layout"))?;
    let smem_layout = g2s
        .get_attr_smem_layout(ctx)
        .map(|attr| attr.0.clone())
        .ok_or_else(|| missing("a shared-memory layout"))?;
    let elem: TypeHandle = g2s
        .get_attr_copy_g2s_elem(ctx)
        .map(|attr| attr.clone().into())
        .ok_or_else(|| missing("an element type"))?;
    let elem_bytes = elem_byte_size(ctx, elem).ok_or_else(|| {
        ExpandError::NotImplemented(format!(
            "unsupported cute.copy_g2s element type {:?}",
            elem.deref(ctx)
        ))
    })?;

    // The shared validator already ran at decode time and inside Verify;
    // this pass consumes its checked maps rather than re-deriving any fact.
    let plan = validate_cooperative_copy_plan(
        atom_bytes,
        &thread_layout,
        &value_layout,
        &tile_layout,
        &smem_layout,
        elem_bytes,
    )
    .map_err(|error| ExpandError::Invalid(format!("invalid cooperative copy plan: {error}")))?;

    // Every atom's global address is `base + esize*(g_row*ld + g_col)`. The
    // ld promise makes row starts atom-aligned; in-tile offsets are validated
    // atom-aligned. The remaining term is the tile-column origin, so the tile
    // row width in bytes must keep whole-atom alignment for every tile_c.
    let tile_row_bytes = plan
        .tiler_shape
        .get(1)
        .copied()
        .unwrap_or(0)
        .checked_mul(elem_bytes)
        .ok_or_else(|| ExpandError::Invalid("tile row byte width overflows i64".into()))?;
    if tile_row_bytes % i64::from(atom_bytes) != 0 {
        return Err(ExpandError::Invalid(format!(
            "cute.copy_g2s tile row is {tile_row_bytes} bytes, which does not preserve \
             {atom_bytes}-byte atom alignment across tile columns"
        )));
    }

    let values_per_atom = i64::from(atom_bytes) / elem_bytes;
    let atoms_per_thread = plan.values_per_thread / values_per_atom;
    let tv_modes = plan.tv_layout.modes();
    if tv_modes.len() != 2 {
        return Err(ExpandError::Invalid(
            "TV layout must have thread and value modes".into(),
        ));
    }
    let thread_leaves = mode_leaves(&tv_modes[0])?;
    let mut atom_cell_offsets = Vec::with_capacity(atoms_per_thread as usize);
    for atom in 0..atoms_per_thread {
        let coordinate = IntTuple::Tuple(vec![
            IntTuple::Leaf(0),
            IntTuple::Leaf(atom * values_per_atom),
        ]);
        let cell = plan.tv_layout.checked_call(&coordinate).ok_or_else(|| {
            ExpandError::Invalid(format!("TV layout rejects atom {atom} base coordinate"))
        })?;
        atom_cell_offsets.push(cell);
    }

    let smem_leaves = mode_leaves(plan.smem_byte_layout.inner())?;
    let dst_operand = {
        let op = op_ptr.deref(ctx);
        op.get_operand(6)
    };
    let (dst_pointee, dst_mutable, dst_address_space) =
        pointer_fields(ctx, dst_operand, "shared destination")?;
    if dst_pointee != elem || !dst_mutable {
        return Err(ExpandError::Invalid(
            "cute.copy_g2s shared destination must be a mutable pointer to elem".into(),
        ));
    }

    let (operands, loc) = {
        let op = op_ptr.deref(ctx);
        (op.operands().collect::<Vec<_>>(), op.loc())
    };

    Ok(CopyG2sPlan {
        op: op_ptr,
        loc,
        operands,
        atom_bytes,
        elem_bytes,
        tile_rows: plan.tiler_shape[0],
        tile_cols: plan.tiler_shape[1],
        values_per_atom,
        thread_leaves,
        atom_cell_offsets,
        smem_leaves,
        smem_byte_offset: plan.smem_byte_layout.offset(),
        smem_swizzle: plan.smem_byte_layout.outer(),
        dst_mutable,
        dst_address_space,
    })
}

/// Expand one `cute.copy_g2s` into per-atom address arithmetic plus one
/// predicated `cp.async` zfill transaction per atom.
///
/// Everything static folded at compile time; the emitted scalar ops carry
/// only the runtime-dependent terms:
///
/// ```text
/// cell  = f(tidx) + static_atom_base          (divmod chain, static consts)
/// row   = cell % TR         col = cell / TR
/// g_row = tile_r*TR + row   g_col = tile_c*TC + col
/// src   = gmem + (g_row*ld + g_col) * in_bounds        (base+0 fallback)
/// bytes = clamp(cols - g_col, 0..=vpa) * esize * in_bounds
/// dst   = smem + swizzle(smem_bytes(cell))             (always in-tile)
/// cp.async.ca zfill: reads `bytes`, zero-fills the rest of the atom
/// ```
///
/// Uniform predication: every thread issues every atom; no branch exists for
/// a barrier to deadlock against, and edge atoms zero-fill their cells.
fn expand_copy_g2s(ctx: &mut Context, rewriter: &mut IRRewriter<Recorder>, plan: CopyG2sPlan) {
    let anchor = plan.op;
    let loc = plan.loc.clone();
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
    let bool_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
    let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();

    type OpInfo = (fn(Ptr<Operation>) -> pliron::op::OpObj, std::any::TypeId);
    let emit = |ctx: &mut Context,
                info: OpInfo,
                results: Vec<TypeHandle>,
                operands: Vec<Value>|
     -> Ptr<Operation> {
        let op = Operation::new(ctx, info, results, operands, vec![], 0);
        op.deref_mut(ctx).set_loc(loc.clone());
        op.insert_before(ctx, anchor);
        op
    };
    let value_of = |ctx: &Context, op: Ptr<Operation>| op.deref(ctx).get_result(0);
    let c64 = |ctx: &mut Context, value: i64| -> Value {
        let op = emit(
            ctx,
            MirConstantOp::get_concrete_op_info(),
            vec![u64_ty],
            vec![],
        );
        let attr = IntegerAttr::new(
            IntegerType::get(ctx, 64, Signedness::Unsigned),
            APInt::from_u64(value as u64, NonZeroUsize::new(64).unwrap()),
        );
        MirConstantOp::new(op).set_attr_value(ctx, attr);
        value_of(ctx, op)
    };
    let bin = |ctx: &mut Context, info: OpInfo, ty: TypeHandle, a: Value, b: Value| -> Value {
        let op = emit(ctx, info, vec![ty], vec![a, b]);
        value_of(ctx, op)
    };
    let zext64 = |ctx: &mut Context, narrow: Value| -> Value {
        let op = emit(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![u64_ty],
            vec![narrow],
        );
        MirCastOp::new(op).set_attr_cast_kind(ctx, MirCastKindAttr::IntToInt);
        value_of(ctx, op)
    };

    let gmem_base = plan.operands[0];
    let rows = plan.operands[1];
    let cols = plan.operands[2];
    let leading_dim = plan.operands[3];
    let tile_row = plan.operands[4];
    let tile_col = plan.operands[5];
    let smem_base = plan.operands[6];
    let tidx = plan.operands[8];

    let add_info = MirAddOp::get_concrete_op_info();
    let sub_info = MirSubOp::get_concrete_op_info();
    let mul_info = MirMulOp::get_concrete_op_info();
    let div_info = MirDivOp::get_concrete_op_info();
    let rem_info = MirRemOp::get_concrete_op_info();

    // Shared destination as a byte pointer: the swizzled smem map is byte
    // addressed, and one cast serves every atom.
    let smem_byte_ptr_ty: TypeHandle =
        MirPtrType::get(ctx, byte_ty, plan.dst_mutable, plan.dst_address_space).into();
    let smem_cast = emit(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![smem_byte_ptr_ty],
        vec![smem_base],
    );
    MirCastOp::new(smem_cast).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    let smem_bytes_base = value_of(ctx, smem_cast);

    // f(tidx): decompose the thread index over the TV thread mode, exactly
    // as crd2idx would (mod on every leaf but the last), and sum the static
    // stride terms. Extent-1 and stride-0 leaves fold away at compile time.
    let tidx64 = zext64(ctx, tidx);
    let mut remaining = tidx64;
    let mut thread_term: Option<Value> = None;
    let leaf_count = plan.thread_leaves.len();
    for (index, &(extent, stride)) in plan.thread_leaves.iter().enumerate() {
        let is_last = index + 1 == leaf_count;
        let coordinate = if is_last {
            remaining
        } else {
            let extent_c = c64(ctx, extent);
            let coord = bin(ctx, rem_info, u64_ty, remaining, extent_c);
            let extent_c2 = c64(ctx, extent);
            remaining = bin(ctx, div_info, u64_ty, remaining, extent_c2);
            coord
        };
        if stride == 0 {
            continue;
        }
        let stride_c = c64(ctx, stride);
        let term = bin(ctx, mul_info, u64_ty, coordinate, stride_c);
        thread_term = Some(match thread_term {
            Some(acc) => bin(ctx, add_info, u64_ty, acc, term),
            None => term,
        });
    }

    let tile_rows_c = c64(ctx, plan.tile_rows);
    let tile_cols_c = c64(ctx, plan.tile_cols);
    let values_per_atom_c = c64(ctx, plan.values_per_atom);
    let elem_bytes_c = c64(ctx, plan.elem_bytes);
    let one_c = c64(ctx, 1);
    let tile_origin_row = bin(ctx, mul_info, u64_ty, tile_row, tile_rows_c);
    let tile_origin_col = bin(ctx, mul_info, u64_ty, tile_col, tile_cols_c);

    for &cell_base in &plan.atom_cell_offsets {
        // cell = f(tidx) + static atom base
        let cell = match thread_term {
            Some(f_t) => {
                let base_c = c64(ctx, cell_base);
                bin(ctx, add_info, u64_ty, f_t, base_c)
            }
            None => c64(ctx, cell_base),
        };
        // Tile-cell decode, the validator's convention: row = cell % TR,
        // col = cell / TR (colex, rows fastest).
        let row = bin(ctx, rem_info, u64_ty, cell, tile_rows_c);
        let col = bin(ctx, div_info, u64_ty, cell, tile_rows_c);
        let g_row = bin(ctx, add_info, u64_ty, tile_origin_row, row);
        let g_col = bin(ctx, add_info, u64_ty, tile_origin_col, col);

        // Uniform predication, all arithmetic (no branch, no select op):
        // in_bounds is a 0/1 mask; the masked offset falls back to base+0.
        let row_ok = bin(ctx, MirLtOp::get_concrete_op_info(), bool_ty, g_row, rows);
        let col_ok = bin(ctx, MirLtOp::get_concrete_op_info(), bool_ty, g_col, cols);
        let ok = bin(
            ctx,
            MirBitAndOp::get_concrete_op_info(),
            bool_ty,
            row_ok,
            col_ok,
        );
        let ok64 = zext64(ctx, ok);

        let row_offset = bin(ctx, mul_info, u64_ty, g_row, leading_dim);
        let element_offset = bin(ctx, add_info, u64_ty, row_offset, g_col);
        let masked_offset = bin(ctx, mul_info, u64_ty, element_offset, ok64);
        let src_ptr = bin(
            ctx,
            MirPtrOffsetOp::get_concrete_op_info(),
            gmem_base.get_type(ctx),
            gmem_base,
            masked_offset,
        );

        // available = cols - g_col (wrapping garbage when out of bounds, but
        // the whole term is multiplied by the 0 mask below).
        let available = bin(ctx, sub_info, u64_ty, cols, g_col);
        let full = bin(
            ctx,
            MirGeOp::get_concrete_op_info(),
            bool_ty,
            available,
            values_per_atom_c,
        );
        let full64 = zext64(ctx, full);
        let not_full64 = bin(ctx, sub_info, u64_ty, one_c, full64);
        let full_part = bin(ctx, mul_info, u64_ty, values_per_atom_c, full64);
        let partial_part = bin(ctx, mul_info, u64_ty, available, not_full64);
        let taken_elements = bin(ctx, add_info, u64_ty, full_part, partial_part);
        let live_elements = bin(ctx, mul_info, u64_ty, taken_elements, ok64);
        let live_bytes = bin(ctx, mul_info, u64_ty, live_elements, elem_bytes_c);
        let source_size_op = emit(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![u32_ty],
            vec![live_bytes],
        );
        MirCastOp::new(source_size_op).set_attr_cast_kind(ctx, MirCastKindAttr::IntToInt);
        let source_size = value_of(ctx, source_size_op);

        // Shared-memory byte offset: decompose the cell over the inner byte
        // layout's leaves, add the composed offset, then swizzle.
        let smem_offset = emit_folded_smem_byte_offset(
            ctx,
            anchor,
            &loc,
            u64_ty,
            &plan.smem_leaves,
            plan.smem_byte_offset,
            &plan.smem_swizzle,
            cell,
        );
        let dst_ptr = bin(
            ctx,
            MirPtrOffsetOp::get_concrete_op_info(),
            smem_bytes_base.get_type(ctx),
            smem_bytes_base,
            smem_offset,
        );

        let transaction = match plan.atom_bytes {
            4 => CpAsyncCaZfill4Op::build(ctx, dst_ptr, src_ptr, source_size),
            8 => CpAsyncCaZfill8Op::build(ctx, dst_ptr, src_ptr, source_size),
            _ => CpAsyncCaZfill16Op::build(ctx, dst_ptr, src_ptr, source_size),
        };
        transaction.deref_mut(ctx).set_loc(loc.clone());
        transaction.insert_before(ctx, anchor);
        request_generated_intrinsic_marker(ctx, transaction);
    }

    rewriter.erase_operation(ctx, anchor);
}

/// Everything needed to emit one warp-cooperative `ldmatrix` fragment load.
struct LdmatrixPlan {
    op: Ptr<Operation>,
    loc: Location,
    operands: Vec<Value>,
    role: crate::attributes::CuteMatrixRoleAttr,
    /// Shared tile extents in elements: `[rows, columns]`.
    smem_rows: i64,
    /// Byte-layout decomposition of the shared tile plus its swizzle.
    smem_leaves: Vec<ModeLeaf>,
    smem_byte_offset: i64,
    smem_swizzle: Swizzle,
    smem_mutable: bool,
    smem_address_space: u32,
    slot_mutable: bool,
    slot_address_space: u32,
}

fn preflight_ldmatrix(ctx: &Context, op_ptr: Ptr<Operation>) -> Result<LdmatrixPlan, ExpandError> {
    {
        let op = op_ptr.deref(ctx);
        if op.get_num_operands() != 5 || op.get_num_results() != 0 {
            return Err(ExpandError::Invalid(
                "cute.ldmatrix must have five operands and zero results".into(),
            ));
        }
    }
    let load = CuteLdmatrixOp::wrap(op_ptr);
    let missing = |what: &str| ExpandError::Invalid(format!("cute.ldmatrix without {what}"));
    let role = load
        .get_attr_matrix_role(ctx)
        .map(|attr| *attr)
        .ok_or_else(|| missing("a matrix role"))?;
    let smem_layout = load
        .get_attr_ldmatrix_smem_layout(ctx)
        .map(|attr| attr.0.clone())
        .ok_or_else(|| missing("a shared-memory layout"))?;

    let modes = smem_layout.inner().modes();
    if modes.len() != 2 {
        return Err(ExpandError::Invalid(
            "cute.ldmatrix shared layout needs two modes".into(),
        ));
    }
    let (rows, columns) = (
        modes[0]
            .checked_size()
            .ok_or_else(|| ExpandError::Invalid("shared layout row extent is invalid".into()))?,
        modes[1]
            .checked_size()
            .ok_or_else(|| ExpandError::Invalid("shared layout column extent is invalid".into()))?,
    );
    let byte_layout = smem_layout.to_byte_offsets(2).map_err(|error| {
        ExpandError::Invalid(format!("shared layout has no byte form: {error}"))
    })?;
    cute_layout::validate_ldmatrix_source(&byte_layout, rows, columns).map_err(|error| {
        ExpandError::Invalid(format!("ldmatrix source is not loadable: {error}"))
    })?;

    let (smem_operand, slot_operand, operands, loc) = {
        let op = op_ptr.deref(ctx);
        (
            op.get_operand(0),
            op.get_operand(4),
            op.operands().collect::<Vec<_>>(),
            op.loc(),
        )
    };
    let (_, smem_mutable, smem_address_space) =
        pointer_fields(ctx, smem_operand, "ldmatrix source")?;
    let (_, slot_mutable, slot_address_space) = pointer_fields(ctx, slot_operand, "fragment slot")?;
    if !slot_mutable {
        return Err(ExpandError::Invalid(
            "cute.ldmatrix fragment slot must be mutable".into(),
        ));
    }

    let smem_leaves = mode_leaves(byte_layout.inner())?;
    Ok(LdmatrixPlan {
        op: op_ptr,
        loc,
        operands,
        role,
        smem_rows: rows,
        smem_leaves,
        smem_byte_offset: byte_layout.offset(),
        smem_swizzle: byte_layout.outer(),
        smem_mutable,
        smem_address_space,
        slot_mutable,
        slot_address_space,
    })
}

/// Expand one `cute.ldmatrix` into per-lane address arithmetic plus one
/// `nvvm.ldmatrix`, storing the returned registers into the fragment slot.
///
/// Each lane supplies the shared address of ONE row of one 8x8 `f16` matrix;
/// the hardware distributes the fragments. Role A loads a 16x16 window as
/// four matrices (`.x4`), role B a 16x8 window as two (`.x2`, transposed):
///
/// ```text
/// submatrix = lane / 8        row_in = lane % 8
/// A: row_off = (submatrix % 2) * 8    col_off = (submatrix / 2) * 8
/// B: row_off = (submatrix % 2) * 8    col_off = 0
/// row  = 16*warp_tile_r + row_off + row_in
/// col  = W*warp_tile_c + col_off               (W = 16 for A, 8 for B)
/// cell = row + ROWS * col
/// addr = smem + swizzle(byte_layout(cell))
/// ```
fn expand_ldmatrix(ctx: &mut Context, rewriter: &mut IRRewriter<Recorder>, plan: LdmatrixPlan) {
    use crate::attributes::CuteMatrixRoleAttr;
    use dialect_nvvm::ops::{
        LdmatrixElementAttr, LdmatrixLayoutAttr, LdmatrixMultiplicityAttr, LdmatrixOp,
        LdmatrixShapeAttr, LdmatrixStateSpaceAttr,
    };

    let anchor = plan.op;
    let loc = plan.loc.clone();
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();

    let add_info = MirAddOp::get_concrete_op_info();
    let mul_info = MirMulOp::get_concrete_op_info();
    let div_info = MirDivOp::get_concrete_op_info();
    let rem_info = MirRemOp::get_concrete_op_info();

    let smem_base = plan.operands[0];
    let warp_tile_r = plan.operands[1];
    let warp_tile_c = plan.operands[2];
    let lane = plan.operands[3];
    let slot = plan.operands[4];

    let (window_cols, multiplicity, matrix_layout, register_count) = match plan.role {
        CuteMatrixRoleAttr::A => (
            16,
            LdmatrixMultiplicityAttr::X4,
            LdmatrixLayoutAttr::Normal,
            4,
        ),
        CuteMatrixRoleAttr::B => (
            8,
            LdmatrixMultiplicityAttr::X2,
            LdmatrixLayoutAttr::Transposed,
            2,
        ),
    };

    // lane decomposition (divisors are powers of two; llc emits shifts).
    let lane64_op = emit_op_before(
        ctx,
        anchor,
        &loc,
        MirCastOp::get_concrete_op_info(),
        vec![u64_ty],
        vec![lane],
    );
    MirCastOp::new(lane64_op).set_attr_cast_kind(ctx, MirCastKindAttr::IntToInt);
    let lane64 = lane64_op.deref(ctx).get_result(0);
    let eight = emit_u64_const(ctx, anchor, &loc, 8);
    let submatrix = emit_bin_op(ctx, anchor, &loc, div_info, u64_ty, lane64, eight);
    let eight2 = emit_u64_const(ctx, anchor, &loc, 8);
    let row_in = emit_bin_op(ctx, anchor, &loc, rem_info, u64_ty, lane64, eight2);

    let two = emit_u64_const(ctx, anchor, &loc, 2);
    let sub_parity = emit_bin_op(ctx, anchor, &loc, rem_info, u64_ty, submatrix, two);
    let eight3 = emit_u64_const(ctx, anchor, &loc, 8);
    let row_off = emit_bin_op(ctx, anchor, &loc, mul_info, u64_ty, sub_parity, eight3);

    // row = 16*warp_tile_r + row_off + row_in
    let sixteen = emit_u64_const(ctx, anchor, &loc, 16);
    let window_row = emit_bin_op(ctx, anchor, &loc, mul_info, u64_ty, warp_tile_r, sixteen);
    let row_partial = emit_bin_op(ctx, anchor, &loc, add_info, u64_ty, window_row, row_off);
    let row = emit_bin_op(ctx, anchor, &loc, add_info, u64_ty, row_partial, row_in);

    // col = W*warp_tile_c (+ (submatrix / 2) * 8 for role A)
    let window_cols_c = emit_u64_const(ctx, anchor, &loc, window_cols);
    let mut col = emit_bin_op(
        ctx,
        anchor,
        &loc,
        mul_info,
        u64_ty,
        warp_tile_c,
        window_cols_c,
    );
    if matches!(plan.role, CuteMatrixRoleAttr::A) {
        let two2 = emit_u64_const(ctx, anchor, &loc, 2);
        let sub_half = emit_bin_op(ctx, anchor, &loc, div_info, u64_ty, submatrix, two2);
        let eight4 = emit_u64_const(ctx, anchor, &loc, 8);
        let col_off = emit_bin_op(ctx, anchor, &loc, mul_info, u64_ty, sub_half, eight4);
        col = emit_bin_op(ctx, anchor, &loc, add_info, u64_ty, col, col_off);
    }

    // cell = row + ROWS * col, then through the byte map + swizzle.
    let rows_c = emit_u64_const(ctx, anchor, &loc, plan.smem_rows);
    let col_term = emit_bin_op(ctx, anchor, &loc, mul_info, u64_ty, col, rows_c);
    let cell = emit_bin_op(ctx, anchor, &loc, add_info, u64_ty, row, col_term);
    let byte_offset = emit_folded_smem_byte_offset(
        ctx,
        anchor,
        &loc,
        u64_ty,
        &plan.smem_leaves,
        plan.smem_byte_offset,
        &plan.smem_swizzle,
        cell,
    );

    let smem_byte_ptr_ty: TypeHandle =
        MirPtrType::get(ctx, byte_ty, plan.smem_mutable, plan.smem_address_space).into();
    let smem_cast = emit_op_before(
        ctx,
        anchor,
        &loc,
        MirCastOp::get_concrete_op_info(),
        vec![smem_byte_ptr_ty],
        vec![smem_base],
    );
    MirCastOp::new(smem_cast).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    let smem_bytes = smem_cast.deref(ctx).get_result(0);
    let byte_address = emit_bin_op(
        ctx,
        anchor,
        &loc,
        MirPtrOffsetOp::get_concrete_op_info(),
        smem_bytes.get_type(ctx),
        smem_bytes,
        byte_offset,
    );
    // nvvm.ldmatrix's B16 ABI expects a u32-pointee address (each lane's
    // segment is read as four 32-bit words).
    let address_ptr_ty: TypeHandle =
        MirPtrType::get(ctx, u32_ty, plan.smem_mutable, plan.smem_address_space).into();
    let address_cast = emit_op_before(
        ctx,
        anchor,
        &loc,
        MirCastOp::get_concrete_op_info(),
        vec![address_ptr_ty],
        vec![byte_address],
    );
    MirCastOp::new(address_cast).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    let address = address_cast.deref(ctx).get_result(0);

    let transaction = LdmatrixOp::build(
        ctx,
        address,
        LdmatrixShapeAttr::M8n8,
        multiplicity,
        matrix_layout,
        LdmatrixElementAttr::B16,
        LdmatrixStateSpaceAttr::Shared,
    );
    transaction.deref_mut(ctx).set_loc(loc.clone());
    transaction.insert_before(ctx, anchor);
    request_generated_intrinsic_marker(ctx, transaction);

    // Store the returned registers into the fragment slot ([u32; N]).
    let slot_u32_ty: TypeHandle =
        MirPtrType::get(ctx, u32_ty, plan.slot_mutable, plan.slot_address_space).into();
    let slot_cast = emit_op_before(
        ctx,
        anchor,
        &loc,
        MirCastOp::get_concrete_op_info(),
        vec![slot_u32_ty],
        vec![slot],
    );
    MirCastOp::new(slot_cast).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    let slot_u32 = slot_cast.deref(ctx).get_result(0);
    for register in 0..register_count {
        let value = transaction.deref(ctx).get_result(register);
        let index = emit_u64_const(ctx, anchor, &loc, register as i64);
        let target = emit_bin_op(
            ctx,
            anchor,
            &loc,
            MirPtrOffsetOp::get_concrete_op_info(),
            slot_u32.get_type(ctx),
            slot_u32,
            index,
        );
        let store = emit_op_before(
            ctx,
            anchor,
            &loc,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![target, value],
        );
        MirStoreOp::new(store).set_volatile(ctx, false);
    }

    rewriter.erase_operation(ctx, anchor);
}

/// Everything needed to emit one hardware tile copy.
struct TmaLoadPlan {
    op: Ptr<Operation>,
    loc: Location,
    operands: Vec<Value>,
    tile_rows: i64,
    tile_cols: i64,
    smem_mutable: bool,
    smem_address_space: u32,
}

fn preflight_tma_load(ctx: &Context, op_ptr: Ptr<Operation>) -> Result<TmaLoadPlan, ExpandError> {
    {
        let op = op_ptr.deref(ctx);
        if op.get_num_operands() != 5 || op.get_num_results() != 0 {
            return Err(ExpandError::Invalid(
                "cute.copy_tma_2d must have five operands and zero results".into(),
            ));
        }
    }
    let load = CuteTmaLoad2dOp::wrap(op_ptr);
    let smem_layout = load
        .get_attr_tma_smem_layout(ctx)
        .map(|attr| attr.0.clone())
        .ok_or_else(|| ExpandError::Invalid("cute.copy_tma_2d without a layout".into()))?;
    let elem: TypeHandle = load
        .get_attr_tma_elem(ctx)
        .map(|attr| attr.clone().into())
        .ok_or_else(|| ExpandError::Invalid("cute.copy_tma_2d without an element type".into()))?;
    let elem_bytes = elem_byte_size(ctx, elem).ok_or_else(|| {
        ExpandError::NotImplemented("unsupported cute.copy_tma_2d element type".into())
    })?;
    cute_layout::validate_tma_encodable(&smem_layout, elem_bytes)
        .map_err(|error| ExpandError::Invalid(format!("layout is not TMA-encodable: {error}")))?;
    let modes = smem_layout.inner().modes();
    let (tile_rows, tile_cols) = (
        modes[0]
            .checked_size()
            .ok_or_else(|| ExpandError::Invalid("TMA tile row extent is invalid".into()))?,
        modes[1]
            .checked_size()
            .ok_or_else(|| ExpandError::Invalid("TMA tile column extent is invalid".into()))?,
    );
    let (smem_operand, operands, loc) = {
        let op = op_ptr.deref(ctx);
        (
            op.get_operand(0),
            op.operands().collect::<Vec<_>>(),
            op.loc(),
        )
    };
    let (_, smem_mutable, smem_address_space) =
        pointer_fields(ctx, smem_operand, "TMA destination")?;

    Ok(TmaLoadPlan {
        op: op_ptr,
        loc,
        operands,
        tile_rows,
        tile_cols,
        smem_mutable,
        smem_address_space,
    })
}

/// Expand one `cute.copy_tma_2d` into the marker-requested bulk tensor copy.
///
/// TMA owns the layout (it is inside the descriptor), so no address math is
/// emitted: the expansion converts the tile index to element origins, adds
/// the fixed cta-mask/cache-hint operands the NVVM op expects, and hands the
/// operation to the generated-intrinsic catalog:
///
/// ```text
/// cute.copy_tma_2d (smem, bar, map, r, c)
///   ==>
/// nvvm.cp_async_bulk_tensor_g2s_tile_2d
///     (smem_u8, bar, map, i32(c*COLS), i32(r*ROWS), i16 0, i64 0)
/// ```
fn expand_tma_load(ctx: &mut Context, rewriter: &mut IRRewriter<Recorder>, plan: TmaLoadPlan) {
    use dialect_nvvm::ops::CpAsyncBulkTensorG2sTile2dOp;

    let anchor = plan.op;
    let loc = plan.loc.clone();
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signed).into();
    let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let mul_info = MirMulOp::get_concrete_op_info();

    let smem_base = plan.operands[0];
    let barrier = plan.operands[1];
    let tensor_map = plan.operands[2];
    let tile_r = plan.operands[3];
    let tile_c = plan.operands[4];

    let smem_byte_ptr_ty: TypeHandle =
        MirPtrType::get(ctx, byte_ty, plan.smem_mutable, plan.smem_address_space).into();
    let smem_cast = emit_op_before(
        ctx,
        anchor,
        &loc,
        MirCastOp::get_concrete_op_info(),
        vec![smem_byte_ptr_ty],
        vec![smem_base],
    );
    MirCastOp::new(smem_cast).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    let smem_u8 = smem_cast.deref(ctx).get_result(0);

    // Tile index -> element origin; TMA coordinates run innermost first
    // (column, then row), as 32-bit signed integers.
    let coord32 = |ctx: &mut Context, index: Value, extent: i64| -> Value {
        let extent_c = emit_u64_const(ctx, anchor, &loc, extent);
        let elements = emit_bin_op(ctx, anchor, &loc, mul_info, u64_ty, index, extent_c);
        let cast = emit_op_before(
            ctx,
            anchor,
            &loc,
            MirCastOp::get_concrete_op_info(),
            vec![i32_ty],
            vec![elements],
        );
        MirCastOp::new(cast).set_attr_cast_kind(ctx, MirCastKindAttr::IntToInt);
        cast.deref(ctx).get_result(0)
    };
    let column_origin = coord32(ctx, tile_c, plan.tile_cols);
    let row_origin = coord32(ctx, tile_r, plan.tile_rows);

    let cta_mask = {
        let op = emit_op_before(
            ctx,
            anchor,
            &loc,
            MirConstantOp::get_concrete_op_info(),
            vec![IntegerType::get(ctx, 16, Signedness::Signless).into()],
            vec![],
        );
        let attr = IntegerAttr::new(
            IntegerType::get(ctx, 16, Signedness::Signless),
            APInt::from_u64(0, NonZeroUsize::new(16).unwrap()),
        );
        MirConstantOp::new(op).set_attr_value(ctx, attr);
        op.deref(ctx).get_result(0)
    };
    let cache_hint = {
        let op = emit_op_before(
            ctx,
            anchor,
            &loc,
            MirConstantOp::get_concrete_op_info(),
            vec![IntegerType::get(ctx, 64, Signedness::Signless).into()],
            vec![],
        );
        let attr = IntegerAttr::new(
            IntegerType::get(ctx, 64, Signedness::Signless),
            APInt::from_u64(0, NonZeroUsize::new(64).unwrap()),
        );
        MirConstantOp::new(op).set_attr_value(ctx, attr);
        op.deref(ctx).get_result(0)
    };

    let transaction = Operation::new(
        ctx,
        CpAsyncBulkTensorG2sTile2dOp::get_concrete_op_info(),
        vec![],
        vec![
            smem_u8,
            barrier,
            tensor_map,
            column_origin,
            row_origin,
            cta_mask,
            cache_hint,
        ],
        vec![],
        0,
    );
    transaction.deref_mut(ctx).set_loc(loc.clone());
    transaction.insert_before(ctx, anchor);
    request_generated_intrinsic_marker(ctx, transaction);

    rewriter.erase_operation(ctx, anchor);
}

/// Everything needed to emit one shared-to-global hardware tile copy.
struct TmaStorePlan {
    op: Ptr<Operation>,
    loc: Location,
    operands: Vec<Value>,
    tile_rows: i64,
    tile_cols: i64,
    smem_mutable: bool,
    smem_address_space: u32,
}

fn preflight_tma_store(ctx: &Context, op_ptr: Ptr<Operation>) -> Result<TmaStorePlan, ExpandError> {
    {
        let op = op_ptr.deref(ctx);
        if op.get_num_operands() != 4 || op.get_num_results() != 0 {
            return Err(ExpandError::Invalid(
                "cute.copy_tma_s2g_2d must have four operands and zero results".into(),
            ));
        }
    }
    let store = CuteTmaStore2dOp::wrap(op_ptr);
    let smem_layout = store
        .get_attr_tma_store_smem_layout(ctx)
        .map(|attr| attr.0.clone())
        .ok_or_else(|| ExpandError::Invalid("cute.copy_tma_s2g_2d without a layout".into()))?;
    let elem: TypeHandle = store
        .get_attr_tma_store_elem(ctx)
        .map(|attr| attr.clone().into())
        .ok_or_else(|| {
            ExpandError::Invalid("cute.copy_tma_s2g_2d without an element type".into())
        })?;
    let elem_bytes = elem_byte_size(ctx, elem).ok_or_else(|| {
        ExpandError::NotImplemented("unsupported cute.copy_tma_s2g_2d element type".into())
    })?;
    cute_layout::validate_tma_encodable(&smem_layout, elem_bytes)
        .map_err(|error| ExpandError::Invalid(format!("layout is not TMA-encodable: {error}")))?;
    let modes = smem_layout.inner().modes();
    let (tile_rows, tile_cols) = (
        modes[0]
            .checked_size()
            .ok_or_else(|| ExpandError::Invalid("TMA store tile row extent is invalid".into()))?,
        modes[1].checked_size().ok_or_else(|| {
            ExpandError::Invalid("TMA store tile column extent is invalid".into())
        })?,
    );
    let (smem_operand, operands, loc) = {
        let op = op_ptr.deref(ctx);
        (
            op.get_operand(0),
            op.operands().collect::<Vec<_>>(),
            op.loc(),
        )
    };
    let (_, smem_mutable, smem_address_space) =
        pointer_fields(ctx, smem_operand, "TMA store source")?;

    Ok(TmaStorePlan {
        op: op_ptr,
        loc,
        operands,
        tile_rows,
        tile_cols,
        smem_mutable,
        smem_address_space,
    })
}

/// Expand one `cute.copy_tma_s2g_2d` into one bulk-group tensor store.
///
/// ```text
/// cute.copy_tma_s2g_2d (smem, map, r, c)
///   ==>
/// nvvm.cp_async_bulk_tensor_s2g_tile_2d
///     (smem_u8, map, i32(c*COLS), i32(r*ROWS))
/// ```
fn expand_tma_store(ctx: &mut Context, rewriter: &mut IRRewriter<Recorder>, plan: TmaStorePlan) {
    use dialect_nvvm::ops::CpAsyncBulkTensorS2gTile2dOp;

    let anchor = plan.op;
    let loc = plan.loc.clone();
    let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signed).into();
    let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
    let mul_info = MirMulOp::get_concrete_op_info();

    let smem_base = plan.operands[0];
    let tensor_map = plan.operands[1];
    let tile_r = plan.operands[2];
    let tile_c = plan.operands[3];

    let smem_byte_ptr_ty: TypeHandle =
        MirPtrType::get(ctx, byte_ty, plan.smem_mutable, plan.smem_address_space).into();
    let smem_cast = emit_op_before(
        ctx,
        anchor,
        &loc,
        MirCastOp::get_concrete_op_info(),
        vec![smem_byte_ptr_ty],
        vec![smem_base],
    );
    MirCastOp::new(smem_cast).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    let smem_u8 = smem_cast.deref(ctx).get_result(0);

    let coord32 = |ctx: &mut Context, index: Value, extent: i64| -> Value {
        let extent_c = emit_u64_const(ctx, anchor, &loc, extent);
        let elements = emit_bin_op(ctx, anchor, &loc, mul_info, u64_ty, index, extent_c);
        let cast = emit_op_before(
            ctx,
            anchor,
            &loc,
            MirCastOp::get_concrete_op_info(),
            vec![i32_ty],
            vec![elements],
        );
        MirCastOp::new(cast).set_attr_cast_kind(ctx, MirCastKindAttr::IntToInt);
        cast.deref(ctx).get_result(0)
    };
    let column_origin = coord32(ctx, tile_c, plan.tile_cols);
    let row_origin = coord32(ctx, tile_r, plan.tile_rows);

    let transaction = Operation::new(
        ctx,
        CpAsyncBulkTensorS2gTile2dOp::get_concrete_op_info(),
        vec![],
        vec![smem_u8, tensor_map, column_origin, row_origin],
        vec![],
        0,
    );
    transaction.deref_mut(ctx).set_loc(loc.clone());
    transaction.insert_before(ctx, anchor);
    request_generated_intrinsic_marker(ctx, transaction);

    rewriter.erase_operation(ctx, anchor);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{CuteAssumeDivOp, CuteCopyG2SOp, CuteCopyOp};
    use crate::types::CuteEpilogueTileType;
    use cute_layout::{ComposedLayout, IntTuple, Layout, OffsetUnit, Swizzle};
    use dialect_mir::ops::MirUndefOp;
    use dialect_mir::types::{MirArrayType, MirPtrType, address_space};
    use pliron::builtin::ops::ModuleOp;
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};

    fn module_top(ctx: &mut Context) -> (Ptr<Operation>, Ptr<pliron::basic_block::BasicBlock>) {
        dialect_mir::register(ctx);
        dialect_nvvm::register(ctx);
        crate::register(ctx);
        let module = ModuleOp::new(ctx, "cute_expand_test".try_into().unwrap());
        let module_op = module.get_operation();
        let region = module_op.deref(ctx).get_region(0);
        let block = region.deref(ctx).iter(ctx).next().unwrap();
        (module_op, block)
    }

    fn undef(
        ctx: &mut Context,
        block: Ptr<pliron::basic_block::BasicBlock>,
        ty: TypeHandle,
    ) -> Value {
        let op = MirUndefOp::new(ctx, ty).get_operation();
        op.insert_at_back(block, ctx);
        op.deref(ctx).get_result(0)
    }

    fn append_copy(
        ctx: &mut Context,
        block: Ptr<pliron::basic_block::BasicBlock>,
        layout: Layout,
    ) -> Ptr<Operation> {
        let elem: TypeHandle = FP32Type::get(ctx).into();
        let src_ty: TypeHandle = MirPtrType::get_generic(ctx, elem, false).into();
        let dst_ty: TypeHandle = MirPtrType::get_generic(ctx, elem, true).into();
        let src = undef(ctx, block, src_ty);
        let dst = undef(ctx, block, dst_ty);
        let copy = CuteCopyOp::new(ctx, src, dst, layout, elem).get_operation();
        copy.insert_at_back(block, ctx);
        copy
    }

    fn append_copy_g2s(
        ctx: &mut Context,
        block: Ptr<pliron::basic_block::BasicBlock>,
        smem: ComposedLayout,
    ) -> Ptr<Operation> {
        let elem: TypeHandle = FP32Type::get(ctx).into();
        let src_ty: TypeHandle = MirPtrType::get(ctx, elem, false, address_space::GLOBAL).into();
        let dst_ty: TypeHandle = MirPtrType::get(ctx, elem, true, address_space::SHARED).into();
        let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let operands = [
            undef(ctx, block, src_ty),
            undef(ctx, block, u64_ty),
            undef(ctx, block, u64_ty),
            undef(ctx, block, u64_ty),
            undef(ctx, block, u64_ty),
            undef(ctx, block, u64_ty),
            undef(ctx, block, dst_ty),
            undef(ctx, block, u64_ty),
            undef(ctx, block, u32_ty),
        ];
        let copy = CuteCopyG2SOp::new(
            ctx,
            operands,
            16,
            "(6,1):(1,0)".parse().unwrap(),
            "(1,4):(0,1)".parse().unwrap(),
            "(6,4):(4,1)".parse().unwrap(),
            smem,
            4,
            elem,
        )
        .get_operation();
        copy.insert_at_back(block, ctx);
        copy
    }

    fn identity_smem() -> ComposedLayout {
        ComposedLayout::from_layout("(6,4):(4,1)".parse().unwrap(), OffsetUnit::Elements)
    }

    fn swizzled_smem() -> ComposedLayout {
        ComposedLayout::new(
            cute_layout::Swizzle::new(3, 4, 3),
            8,
            "(6,4):(4,1)".parse().unwrap(),
            OffsetUnit::Elements,
        )
        .unwrap()
    }

    fn gemm_value_tma_layout() -> ComposedLayout {
        ComposedLayout::new(
            Swizzle::new(2, 4, 3),
            0,
            "(128,64):(64,1)".parse().unwrap(),
            OffsetUnit::Elements,
        )
        .unwrap()
    }

    fn gemm_scale_tma_layout() -> ComposedLayout {
        ComposedLayout::new(
            Swizzle::IDENTITY,
            0,
            "(1,256):(256,1)".parse().unwrap(),
            OffsetUnit::Elements,
        )
        .unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn append_semantic_tma_copy(
        ctx: &mut Context,
        block: Ptr<pliron::basic_block::BasicBlock>,
        descriptor: Value,
        shared_base: Value,
        capacity: Value,
        barrier: Value,
        row: Value,
        column: Value,
        element: TypeHandle,
        source_layout: ComposedLayout,
        destination_layout: ComposedLayout,
        alignment_bytes: u64,
    ) -> Ptr<Operation> {
        let source = CuteTmaGmemViewOp::new(ctx, descriptor, element, source_layout);
        let source_op = source.get_operation();
        source_op.insert_at_back(block, ctx);
        let source = source_op.deref(ctx).get_result(0);
        let destination = CuteTmaSmemViewOp::new(
            ctx,
            shared_base,
            capacity,
            element,
            destination_layout,
            alignment_bytes,
        );
        let destination_op = destination.get_operation();
        destination_op.insert_at_back(block, ctx);
        let destination = destination_op.deref(ctx).get_result(0);
        let copy = CuteTmaCopy2dOp::new(ctx, source, destination, row, column, barrier);
        let copy = copy.get_operation();
        copy.insert_at_back(block, ctx);
        copy
    }

    fn append_scheduler_story(
        ctx: &mut Context,
        block: Ptr<pliron::basic_block::BasicBlock>,
        grid: CuteTileGridAttr,
        include_coordinates: bool,
    ) {
        let start = CuteSchedulerNew1dOp::new(ctx, grid).get_operation();
        start.insert_at_back(block, ctx);
        let current = CuteSchedulerNew1dOp::wrap(start).current(ctx);
        let stride = CuteSchedulerNew1dOp::wrap(start).stride(ctx);

        for op in [
            CuteSchedulerHasWorkOp::new(ctx, current, grid).get_operation(),
            CuteSchedulerAdvanceOp::new(ctx, current, stride, grid).get_operation(),
        ] {
            op.insert_at_back(block, ctx);
        }
        let selected = CuteSchedulerCurrentOp::new(ctx, current, grid).get_operation();
        selected.insert_at_back(block, ctx);
        if include_coordinates {
            let tile = CuteSchedulerCurrentOp::wrap(selected).work_tile(ctx);
            CuteWorkTileCoordinatesOp::new(ctx, tile)
                .get_operation()
                .insert_at_back(block, ctx);
        }
    }

    fn append_single_copy_pipeline_expect(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        transaction_bytes: u32,
        attach_copy: bool,
    ) -> (Ptr<Operation>, Ptr<Operation>, Option<Ptr<Operation>>) {
        let u8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
        let barrier_ty: TypeHandle =
            MirPtrType::get(ctx, u64_ty, true, address_space::SHARED).into();
        let barrier_base = undef(ctx, block, barrier_ty);
        let make = CuteTmaLoadPipelineMakeOp::new(ctx, barrier_base, 3, 8, transaction_bytes, 8)
            .get_operation();
        make.insert_at_back(block, ctx);
        let pipeline = CuteTmaLoadPipelineMakeOp::wrap(make).pipeline(ctx);
        let slot = u32_constant(ctx, block, 0);
        let expect = CutePipelineProducerExpectTxOp::new(
            ctx,
            pipeline,
            slot,
            CutePipelineStateAttr::producer(3),
        )
        .get_operation();
        expect.insert_at_back(block, ctx);

        let copy = attach_copy.then(|| {
            let descriptor_ty: TypeHandle = MirPtrType::get_generic(ctx, u8_ty, false).into();
            let shared_u8: TypeHandle =
                MirPtrType::get(ctx, u8_ty, true, address_space::SHARED).into();
            let descriptor = undef(ctx, block, descriptor_ty);
            let shared = undef(ctx, block, shared_u8);
            let capacity = undef(ctx, block, u64_ty);
            let row = undef(ctx, block, u64_ty);
            let column = undef(ctx, block, u64_ty);
            let completion = CutePipelineProducerExpectTxOp::wrap(expect).completion_barrier(ctx);
            append_semantic_tma_copy(
                ctx,
                block,
                descriptor,
                shared,
                capacity,
                completion,
                row,
                column,
                u8_ty,
                gemm_value_tma_layout(),
                gemm_value_tma_layout(),
                1024,
            )
        });
        (make, expect, copy)
    }

    /// Pure mirror of the emitted divmod chain: mod on every leaf but the
    /// last, then sum the static stride terms.
    fn fold_leaves(leaves: &[(i64, i64)], x: i64) -> i64 {
        let mut remaining = x;
        let mut total = 0;
        let count = leaves.len();
        for (index, &(extent, stride)) in leaves.iter().enumerate() {
            let coordinate = if index + 1 == count {
                remaining
            } else {
                let c = remaining % extent;
                remaining /= extent;
                c
            };
            total += coordinate * stride;
        }
        total
    }

    #[test]
    fn preflight_failure_does_not_half_expand_the_module() {
        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        let first = append_copy(&mut ctx, block, Layout::contiguous(4));
        let second = append_copy(
            &mut ctx,
            block,
            Layout::new(IntTuple::Leaf(2), IntTuple::Leaf(2)),
        );

        let error = expand_cute_ops(&mut ctx, module).unwrap_err().to_string();
        assert!(error.contains("non-contiguous"), "{error}");
        let children: Vec<_> = block.deref(&ctx).iter(&ctx).collect();
        assert!(children.contains(&first));
        assert!(children.contains(&second));
    }

    #[test]
    fn copy_g2s_expands_to_marked_cp_async_atoms() {
        use pliron::identifier::Identifier;

        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        append_copy_g2s(&mut ctx, block, identity_smem());

        expand_cute_ops(&mut ctx, module).unwrap();

        let children: Vec<_> = block.deref(&ctx).iter(&ctx).collect();
        assert!(children.iter().all(|op| {
            Operation::get_opid(*op, &ctx).dialect.to_string() != crate::CUTE_DIALECT_NAME
        }));
        // 4 values per thread / 4 values per 16-byte atom = one transaction.
        let zfill_opid = CpAsyncCaZfill16Op::get_opid_static();
        let transactions: Vec<_> = children
            .iter()
            .filter(|op| Operation::get_opid(**op, &ctx) == zfill_opid)
            .collect();
        assert_eq!(transactions.len(), 1);
        let request_key = Identifier::try_from(GENERATED_INTRINSIC_REQUEST_ATTR).unwrap();
        for transaction in transactions {
            assert!(
                transaction
                    .deref(&ctx)
                    .attributes
                    .0
                    .contains_key(&request_key),
                "cp.async transaction is missing its catalog request tag"
            );
        }
        // The identity smem map needs no swizzle arithmetic.
        let xor_opid = dialect_mir::ops::MirBitXorOp::get_opid_static();
        assert!(
            children
                .iter()
                .all(|op| Operation::get_opid(*op, &ctx) != xor_opid)
        );
    }

    #[test]
    fn swizzled_copy_g2s_emits_the_xor_scramble() {
        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        append_copy_g2s(&mut ctx, block, swizzled_smem());

        expand_cute_ops(&mut ctx, module).unwrap();

        let children: Vec<_> = block.deref(&ctx).iter(&ctx).collect();
        let xor_opid = dialect_mir::ops::MirBitXorOp::get_opid_static();
        assert!(
            children
                .iter()
                .any(|op| Operation::get_opid(*op, &ctx) == xor_opid),
            "swizzled smem layout must XOR the byte offset"
        );
    }

    fn append_ldmatrix(
        ctx: &mut Context,
        block: Ptr<pliron::basic_block::BasicBlock>,
        role: crate::attributes::CuteMatrixRoleAttr,
        smem: ComposedLayout,
    ) -> Ptr<Operation> {
        let elem: TypeHandle = dialect_mir::types::MirFP16Type::get(ctx).into();
        let u32_scalar: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let src_ty: TypeHandle = MirPtrType::get(ctx, elem, true, address_space::SHARED).into();
        let slot_ty: TypeHandle =
            MirPtrType::get(ctx, u32_scalar, true, address_space::GENERIC).into();
        let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let operands = [
            undef(ctx, block, src_ty),
            undef(ctx, block, u64_ty),
            undef(ctx, block, u64_ty),
            undef(ctx, block, u32_ty),
            undef(ctx, block, slot_ty),
        ];
        let load = crate::ops::CuteLdmatrixOp::new(ctx, operands, role, smem, elem).get_operation();
        load.insert_at_back(block, ctx);
        load
    }

    fn smem_32x32_row_major() -> ComposedLayout {
        ComposedLayout::from_layout("(32,32):(32,1)".parse().unwrap(), OffsetUnit::Elements)
    }

    fn smem_32x32_swizzled() -> ComposedLayout {
        // Element-unit S<2,3,3>: in bytes S<2,4,3>, protecting 16-byte rows.
        ComposedLayout::new(
            cute_layout::Swizzle::new(2, 3, 3),
            0,
            "(32,32):(32,1)".parse().unwrap(),
            OffsetUnit::Elements,
        )
        .unwrap()
    }

    #[test]
    fn ldmatrix_expands_to_one_marked_transaction_per_load() {
        use crate::attributes::CuteMatrixRoleAttr;
        use pliron::identifier::Identifier;

        for (role, expected_stores) in [(CuteMatrixRoleAttr::A, 4), (CuteMatrixRoleAttr::B, 2)] {
            let mut ctx = Context::new();
            let (module, block) = module_top(&mut ctx);
            append_ldmatrix(&mut ctx, block, role, smem_32x32_row_major());

            expand_cute_ops(&mut ctx, module).unwrap();

            let children: Vec<_> = block.deref(&ctx).iter(&ctx).collect();
            assert!(children.iter().all(|op| {
                Operation::get_opid(*op, &ctx).dialect.to_string() != crate::CUTE_DIALECT_NAME
            }));
            let ldmatrix_opid = dialect_nvvm::ops::LdmatrixOp::get_opid_static();
            let loads: Vec<_> = children
                .iter()
                .filter(|op| Operation::get_opid(**op, &ctx) == ldmatrix_opid)
                .collect();
            assert_eq!(loads.len(), 1);
            let request_key = Identifier::try_from(GENERATED_INTRINSIC_REQUEST_ATTR).unwrap();
            assert!(loads[0].deref(&ctx).attributes.0.contains_key(&request_key));
            let store_opid = MirStoreOp::get_opid_static();
            let stores = children
                .iter()
                .filter(|op| Operation::get_opid(**op, &ctx) == store_opid)
                .count();
            assert_eq!(stores, expected_stores);
        }
    }

    /// The per-lane ldmatrix address oracle: replay the emitted arithmetic in
    /// plain integers for EVERY lane, window position, and role, and compare
    /// with direct composed-layout evaluation. Also require the hardware's
    /// 16-byte segment alignment.
    #[test]
    fn ldmatrix_lane_addresses_match_direct_layout_evaluation() {
        use crate::attributes::CuteMatrixRoleAttr;

        for smem in [smem_32x32_row_major(), smem_32x32_swizzled()] {
            let byte_layout = smem.to_byte_offsets(2).unwrap();
            for role in [CuteMatrixRoleAttr::A, CuteMatrixRoleAttr::B] {
                let mut ctx = Context::new();
                let (_module, block) = module_top(&mut ctx);
                let op = append_ldmatrix(&mut ctx, block, role, smem.clone());
                let plan = preflight_ldmatrix(&ctx, op).unwrap();

                let window_cols = match role {
                    CuteMatrixRoleAttr::A => 16,
                    CuteMatrixRoleAttr::B => 8,
                };
                for wtr in 0..(32 / 16) {
                    for wtc in 0..(32 / window_cols) {
                        for lane in 0..32i64 {
                            let submatrix = lane / 8;
                            let row_in = lane % 8;
                            let row_off = (submatrix % 2) * 8;
                            let col_off = match role {
                                CuteMatrixRoleAttr::A => (submatrix / 2) * 8,
                                CuteMatrixRoleAttr::B => 0,
                            };
                            let row = 16 * wtr + row_off + row_in;
                            let col = window_cols * wtc + col_off;
                            let cell = row + plan.smem_rows * col;
                            let folded =
                                fold_leaves(&plan.smem_leaves, cell) + plan.smem_byte_offset;
                            let formula = plan.smem_swizzle.apply(folded);
                            let direct = byte_layout.checked_call(&IntTuple::Leaf(cell)).unwrap();
                            assert_eq!(formula, direct, "lane {lane} wt ({wtr},{wtc})");
                            assert_eq!(
                                formula % 16,
                                0,
                                "segment for lane {lane} is not 16-byte aligned"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn tma_load_expands_to_one_marked_bulk_tensor_copy() {
        use pliron::identifier::Identifier;

        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        let elem: TypeHandle = dialect_mir::types::MirFP16Type::get(&ctx).into();
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let smem_ty: TypeHandle =
            MirPtrType::get(&mut ctx, elem, true, address_space::SHARED).into();
        let ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, elem, false).into();
        let operands = [
            undef(&mut ctx, block, smem_ty),
            undef(&mut ctx, block, ptr_ty),
            undef(&mut ctx, block, ptr_ty),
            undef(&mut ctx, block, u64_ty),
            undef(&mut ctx, block, u64_ty),
        ];
        // 64x64 f16 tile through the 128-byte swizzle: TMA-encodable.
        let smem = ComposedLayout::new(
            cute_layout::Swizzle::new(3, 3, 3),
            0,
            "(64,64):(64,1)".parse().unwrap(),
            OffsetUnit::Elements,
        )
        .unwrap();
        let load = crate::ops::CuteTmaLoad2dOp::new(&mut ctx, operands, smem, elem).get_operation();
        load.insert_at_back(block, &ctx);

        expand_cute_ops(&mut ctx, module).unwrap();

        let children: Vec<_> = block.deref(&ctx).iter(&ctx).collect();
        assert!(children.iter().all(|op| {
            Operation::get_opid(*op, &ctx).dialect.to_string() != crate::CUTE_DIALECT_NAME
        }));
        let bulk_opid = dialect_nvvm::ops::CpAsyncBulkTensorG2sTile2dOp::get_opid_static();
        let copies: Vec<_> = children
            .iter()
            .filter(|op| Operation::get_opid(**op, &ctx) == bulk_opid)
            .collect();
        assert_eq!(copies.len(), 1);
        assert_eq!(copies[0].deref(&ctx).get_num_operands(), 7);
        let request_key = Identifier::try_from(GENERATED_INTRINSIC_REQUEST_ATTR).unwrap();
        assert!(
            copies[0]
                .deref(&ctx)
                .attributes
                .0
                .contains_key(&request_key)
        );
    }

    #[test]
    fn tma_store_expands_to_one_marked_bulk_group_copy() {
        use pliron::identifier::Identifier;

        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        let elem: TypeHandle = dialect_mir::types::MirFP16Type::get(&ctx).into();
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let smem_ty: TypeHandle =
            MirPtrType::get(&mut ctx, elem, true, address_space::SHARED).into();
        let ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, elem, false).into();
        let operands = [
            undef(&mut ctx, block, smem_ty),
            undef(&mut ctx, block, ptr_ty),
            undef(&mut ctx, block, u64_ty),
            undef(&mut ctx, block, u64_ty),
        ];
        // One half of a 128x128 f16 epilogue. B128 requires the 64-element
        // contiguous mode; two adjacent halves cover the full output tile.
        let smem = ComposedLayout::new(
            cute_layout::Swizzle::new(3, 3, 3),
            0,
            "(128,64):(64,1)".parse().unwrap(),
            OffsetUnit::Elements,
        )
        .unwrap();
        let store =
            crate::ops::CuteTmaStore2dOp::new(&mut ctx, operands, smem, elem).get_operation();
        store.insert_at_back(block, &ctx);

        expand_cute_ops(&mut ctx, module).unwrap();

        let children: Vec<_> = block.deref(&ctx).iter(&ctx).collect();
        assert!(children.iter().all(|op| {
            Operation::get_opid(*op, &ctx).dialect.to_string() != crate::CUTE_DIALECT_NAME
        }));
        let bulk_opid = dialect_nvvm::ops::CpAsyncBulkTensorS2gTile2dOp::get_opid_static();
        let copies: Vec<_> = children
            .iter()
            .filter(|op| Operation::get_opid(**op, &ctx) == bulk_opid)
            .collect();
        assert_eq!(copies.len(), 1);
        assert_eq!(copies[0].deref(&ctx).get_num_operands(), 4);
        let request_key = Identifier::try_from(GENERATED_INTRINSIC_REQUEST_ATTR).unwrap();
        assert!(
            copies[0]
                .deref(&ctx)
                .attributes
                .0
                .contains_key(&request_key)
        );
    }

    /// The in-process address oracle: the static plan the expansion folds
    /// (divmod chain + per-atom cell bases + smem byte fold + swizzle) must
    /// agree with direct cute-layout evaluation for EVERY thread and atom,
    /// in both the identity and the swizzled configuration.
    #[test]
    fn emitted_address_formula_matches_direct_layout_evaluation() {
        for smem in [identity_smem(), swizzled_smem()] {
            let mut ctx = Context::new();
            let (_module, block) = module_top(&mut ctx);
            let op = append_copy_g2s(&mut ctx, block, smem.clone());
            let plan = preflight_copy_g2s(&ctx, op).unwrap();

            let reference = validate_cooperative_copy_plan(
                16,
                &"(6,1):(1,0)".parse().unwrap(),
                &"(1,4):(0,1)".parse().unwrap(),
                &"(6,4):(4,1)".parse().unwrap(),
                &smem,
                4,
            )
            .unwrap();

            for thread in 0..reference.thread_count {
                for (atom, &cell_base) in plan.atom_cell_offsets.iter().enumerate() {
                    let value = atom as i64 * plan.values_per_atom;
                    let formula_cell = fold_leaves(&plan.thread_leaves, thread) + cell_base;
                    let direct_cell = reference
                        .tv_layout
                        .checked_call(&IntTuple::Tuple(vec![
                            IntTuple::Leaf(thread),
                            IntTuple::Leaf(value),
                        ]))
                        .unwrap();
                    assert_eq!(formula_cell, direct_cell, "cell (t={thread}, atom={atom})");

                    let folded =
                        fold_leaves(&plan.smem_leaves, formula_cell) + plan.smem_byte_offset;
                    let formula_smem = plan.smem_swizzle.apply(folded);
                    let direct_smem = reference
                        .smem_byte_layout
                        .checked_call(&IntTuple::Leaf(formula_cell))
                        .unwrap();
                    assert_eq!(
                        formula_smem, direct_smem,
                        "smem byte (t={thread}, atom={atom})"
                    );
                }
            }
        }
    }

    #[test]
    fn assume_div_is_an_identity_fact_and_leaves_no_cute_op() {
        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        let int_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let input = undef(&mut ctx, block, int_ty);
        let assume = CuteAssumeDivOp::new(&mut ctx, input, 16).get_operation();
        assume.insert_at_back(block, &ctx);

        expand_cute_ops(&mut ctx, module).unwrap();

        let children: Vec<_> = block.deref(&ctx).iter(&ctx).collect();
        assert!(children.iter().all(|op| {
            Operation::get_opid(*op, &ctx).dialect.to_string() != crate::CUTE_DIALECT_NAME
        }));
    }

    #[test]
    fn chained_assume_div_facts_rewrite_to_the_live_input() {
        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        let int_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let input = undef(&mut ctx, block, int_ty);

        let first = CuteAssumeDivOp::new(&mut ctx, input, 16).get_operation();
        first.insert_at_back(block, &ctx);
        let first_result = first.deref(&ctx).get_result(0);
        let second = CuteAssumeDivOp::new(&mut ctx, first_result, 8).get_operation();
        second.insert_at_back(block, &ctx);
        let second_result = second.deref(&ctx).get_result(0);

        // Keep a non-cute user alive so the test observes the final SSA edge.
        let consumer = Operation::new(
            &mut ctx,
            MirCastOp::get_concrete_op_info(),
            vec![int_ty],
            vec![second_result],
            vec![],
            0,
        );
        MirCastOp::new(consumer).set_attr_cast_kind(&ctx, MirCastKindAttr::IntToInt);
        consumer.insert_at_back(block, &ctx);

        expand_cute_ops(&mut ctx, module).unwrap();

        assert_eq!(consumer.deref(&ctx).get_operand(0), input);
    }

    #[test]
    fn tensor_story_lowers_to_existing_vector_copy_leaves() {
        use crate::attributes::CuteTensorAccessAttr;

        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        let elem: TypeHandle = FP32Type::get(&ctx).into();
        let global_ro: TypeHandle =
            MirPtrType::get(&mut ctx, elem, false, address_space::GLOBAL).into();
        let global_rw: TypeHandle =
            MirPtrType::get(&mut ctx, elem, true, address_space::GLOBAL).into();
        let local_rw: TypeHandle =
            MirPtrType::get(&mut ctx, elem, true, address_space::LOCAL).into();
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();

        let a_ptr = undef(&mut ctx, block, global_ro);
        let c_ptr = undef(&mut ctx, block, global_rw);
        let carrier = undef(&mut ctx, block, local_rw);
        let len = undef(&mut ctx, block, u64_ty);
        let tid = undef(&mut ctx, block, u64_ty);
        let scalar = undef(&mut ctx, block, elem);

        let append_result = |ctx: &Context, block, op: Ptr<Operation>| {
            op.insert_at_back(block, ctx);
            op.deref(ctx).get_result(0)
        };

        let a = CuteTensorMakeOp::new(
            &mut ctx,
            a_ptr,
            len,
            elem,
            elem,
            CuteTensorAccessAttr::ReadOnly,
            4,
        )
        .get_operation();
        let a = append_result(&ctx, block, a);
        let a = CuteTensorZippedDivideOp::new(&mut ctx, a, 4).get_operation();
        let a = append_result(&ctx, block, a);
        let a = CuteTensorSliceOp::new(&mut ctx, a, tid).get_operation();
        let a = append_result(&ctx, block, a);

        let c = CuteTensorMakeOp::new(
            &mut ctx,
            c_ptr,
            len,
            elem,
            elem,
            CuteTensorAccessAttr::ReadWrite,
            4,
        )
        .get_operation();
        let c = append_result(&ctx, block, c);
        let c = CuteTensorZippedDivideOp::new(&mut ctx, c, 4).get_operation();
        let c = append_result(&ctx, block, c);
        let c = CuteTensorSliceOp::new(&mut ctx, c, tid).get_operation();
        let c = append_result(&ctx, block, c);

        for op in [
            CuteTensorIsFullOp::new(&mut ctx, a).get_operation(),
            CuteTensorBaseOp::new(&mut ctx, a).get_operation(),
            CuteTensorLoadIntoOp::new(&mut ctx, a, carrier, 16).get_operation(),
            CuteTensorStoreFromOp::new(&mut ctx, carrier, c, 16).get_operation(),
            CuteTensorStoreElementAbsOp::new(&mut ctx, c, tid, scalar).get_operation(),
        ] {
            op.insert_at_back(block, &ctx);
        }

        expand_cute_ops(&mut ctx, module).unwrap();

        let mut ops = Vec::new();
        collect_ops(&ctx, module, &mut ops);
        assert!(ops.iter().all(|op| {
            Operation::get_opid(*op, &ctx).dialect.to_string() != crate::CUTE_DIALECT_NAME
        }));
        let loads = ops
            .iter()
            .filter(|op| Operation::get_opid(**op, &ctx) == MirLoadOp::get_opid_static())
            .count();
        let stores = ops
            .iter()
            .filter(|op| Operation::get_opid(**op, &ctx) == MirStoreOp::get_opid_static())
            .count();
        assert_eq!(loads, 2, "global-to-register and register-to-global copies");
        assert_eq!(
            stores, 3,
            "register carrier, full-tile output, and scalar tail stores"
        );
    }

    #[test]
    fn two_scheduler_stories_lower_to_the_native_runtime_recipe() {
        use pliron::identifier::Identifier;

        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        let grid = CuteTileGridAttr::new(3, 2, 2);
        append_scheduler_story(&mut ctx, block, grid, true);
        append_scheduler_story(&mut ctx, block, grid, true);

        lower_scheduler_to_mir(&mut ctx, module).unwrap();

        let mut operations = Vec::new();
        collect_ops(&ctx, module, &mut operations);
        let scheduler_ids = [
            CuteSchedulerNew1dOp::get_opid_static(),
            CuteSchedulerHasWorkOp::get_opid_static(),
            CuteSchedulerCurrentOp::get_opid_static(),
            CuteWorkTileCoordinatesOp::get_opid_static(),
            CuteSchedulerAdvanceOp::get_opid_static(),
        ];
        assert!(
            operations
                .iter()
                .all(|op| { !scheduler_ids.contains(&Operation::get_opid(*op, &ctx)) })
        );

        let count = |opid: OpId| {
            operations
                .iter()
                .filter(|op| Operation::get_opid(**op, &ctx) == opid)
                .count()
        };
        assert_eq!(count(ReadPtxSregCtaidXOp::get_opid_static()), 2);
        assert_eq!(count(ReadPtxSregNctaidXOp::get_opid_static()), 2);
        assert_eq!(count(MirLtOp::get_opid_static()), 2);
        assert_eq!(count(MirDivOp::get_opid_static()), 4);
        assert_eq!(count(MirMulOp::get_opid_static()), 4);
        assert_eq!(count(MirSubOp::get_opid_static()), 4);

        let request_key = Identifier::try_from(GENERATED_INTRINSIC_REQUEST_ATTR).unwrap();
        for op in operations.iter().filter(|op| {
            matches!(
                Operation::get_opid(**op, &ctx),
                id if id == ReadPtxSregCtaidXOp::get_opid_static()
                    || id == ReadPtxSregNctaidXOp::get_opid_static()
            )
        }) {
            assert!(op.deref(&ctx).attributes.0.contains_key(&request_key));
        }

        let saturating_calls: Vec<_> = operations
            .iter()
            .copied()
            .filter(|op| Operation::get_opid(*op, &ctx) == MirCallOp::get_opid_static())
            .filter(|op| {
                MirCallOp::new(*op)
                    .get_attr_callee(&ctx)
                    .is_some_and(|callee| {
                        String::from(callee.clone())
                            == dialect_mir::rust_intrinsics::CALLEE_SATURATING_ADD
                    })
            })
            .collect();
        assert_eq!(saturating_calls.len(), 2);
        assert!(saturating_calls.iter().all(|call| {
            call.deref(&ctx).get_result(0).get_type(&ctx)
                == IntegerType::get(&ctx, 64, Signedness::Unsigned).into()
        }));
    }

    #[test]
    fn pipeline_state_scalars_lower_without_a_runtime_state_object() {
        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        let producer = CutePipelineStateAttr::producer(3);
        let consumer = CutePipelineStateAttr::consumer(3);

        let producer_new = CutePipelineStateNewOp::new(&mut ctx, producer).get_operation();
        producer_new.insert_at_back(block, &ctx);
        let producer_slot = CutePipelineStateNewOp::wrap(producer_new).slot(&ctx);
        let producer_phase = CutePipelineStateNewOp::wrap(producer_new).phase(&ctx);
        for op in [
            CutePipelineStateSlotOp::new(&mut ctx, producer_slot, producer).get_operation(),
            CutePipelineStateAdvanceOp::new(&mut ctx, producer_slot, producer_phase, producer)
                .get_operation(),
        ] {
            op.insert_at_back(block, &ctx);
        }

        let consumer_new = CutePipelineStateNewOp::new(&mut ctx, consumer).get_operation();
        consumer_new.insert_at_back(block, &ctx);
        let consumer_slot = CutePipelineStateNewOp::wrap(consumer_new).slot(&ctx);
        let consumer_phase = CutePipelineStateNewOp::wrap(consumer_new).phase(&ctx);
        CutePipelineStateAdvanceOp::new(&mut ctx, consumer_slot, consumer_phase, consumer)
            .get_operation()
            .insert_at_back(block, &ctx);

        lower_pipeline_to_mir(&mut ctx, module).unwrap();

        let mut operations = Vec::new();
        collect_ops(&ctx, module, &mut operations);
        assert!(
            operations
                .iter()
                .all(|op| { !pipeline_semantic_ids().contains(&Operation::get_opid(*op, &ctx)) })
        );
        let count = |opid: OpId| {
            operations
                .iter()
                .filter(|op| Operation::get_opid(**op, &ctx) == opid)
                .count()
        };
        assert_eq!(count(MirAddOp::get_opid_static()), 2);
        assert_eq!(count(MirEqOp::get_opid_static()), 2);
        assert_eq!(count(MirBitXorOp::get_opid_static()), 2);
        assert_eq!(count(MirSubOp::get_opid_static()), 2);
        assert_eq!(count(MirBitAndOp::get_opid_static()), 2);
        assert_eq!(count(MirCastOp::get_opid_static()), 3);
    }

    #[test]
    fn load_pipeline_story_expands_to_the_exact_barrier_protocol() {
        use pliron::identifier::Identifier;

        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u16_ty: TypeHandle = IntegerType::get(&ctx, 16, Signedness::Unsigned).into();
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let barrier_ty: TypeHandle =
            MirPtrType::get(&mut ctx, u64_ty, true, address_space::SHARED).into();
        let base = undef(&mut ctx, block, barrier_ty);
        let make = CuteTmaLoadPipelineMakeOp::new(&mut ctx, base, 3, 8, 17_408, 8).get_operation();
        make.insert_at_back(block, &ctx);
        let pipeline = CuteTmaLoadPipelineMakeOp::wrap(make).pipeline(&ctx);
        let init_thread = u32_constant(&mut ctx, block, 0);
        CuteTmaLoadPipelineInitOp::new(&mut ctx, pipeline, init_thread)
            .get_operation()
            .insert_at_back(block, &ctx);

        let producer = CutePipelineStateAttr::producer(3);
        let producer_new = CutePipelineStateNewOp::new(&mut ctx, producer).get_operation();
        producer_new.insert_at_back(block, &ctx);
        let producer_slot = CutePipelineStateNewOp::wrap(producer_new).slot(&ctx);
        let producer_phase = CutePipelineStateNewOp::wrap(producer_new).phase(&ctx);
        CutePipelineProducerAcquireOp::new(
            &mut ctx,
            pipeline,
            producer_slot,
            producer_phase,
            producer,
        )
        .get_operation()
        .insert_at_back(block, &ctx);
        let expect =
            CutePipelineProducerExpectTxOp::new(&mut ctx, pipeline, producer_slot, producer)
                .get_operation();
        expect.insert_at_back(block, &ctx);
        let completion = CutePipelineProducerExpectTxOp::wrap(expect).completion_barrier(&ctx);

        let descriptor_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u8_ty, false).into();
        let shared_u8: TypeHandle =
            MirPtrType::get(&mut ctx, u8_ty, true, address_space::SHARED).into();
        let shared_u16: TypeHandle =
            MirPtrType::get(&mut ctx, u16_ty, true, address_space::SHARED).into();
        let descriptors = [
            undef(&mut ctx, block, descriptor_ty),
            undef(&mut ctx, block, descriptor_ty),
            undef(&mut ctx, block, descriptor_ty),
            undef(&mut ctx, block, descriptor_ty),
        ];
        let bases = [
            undef(&mut ctx, block, shared_u8),
            undef(&mut ctx, block, shared_u8),
            undef(&mut ctx, block, shared_u16),
            undef(&mut ctx, block, shared_u16),
        ];
        let capacity = undef(&mut ctx, block, u64_ty);
        let row = undef(&mut ctx, block, u64_ty);
        let column = undef(&mut ctx, block, u64_ty);
        let elements = [u8_ty, u8_ty, u16_ty, u16_ty];
        let layouts = [
            gemm_value_tma_layout(),
            gemm_value_tma_layout(),
            gemm_scale_tma_layout(),
            gemm_scale_tma_layout(),
        ];
        for index in 0..4 {
            append_semantic_tma_copy(
                &mut ctx,
                block,
                descriptors[index],
                bases[index],
                capacity,
                completion,
                row,
                column,
                elements[index],
                layouts[index].clone(),
                layouts[index].clone(),
                if index < 2 { 1024 } else { 512 },
            );
        }

        let consumer = CutePipelineStateAttr::consumer(3);
        let consumer_new = CutePipelineStateNewOp::new(&mut ctx, consumer).get_operation();
        consumer_new.insert_at_back(block, &ctx);
        let consumer_slot = CutePipelineStateNewOp::wrap(consumer_new).slot(&ctx);
        let consumer_phase = CutePipelineStateNewOp::wrap(consumer_new).phase(&ctx);
        CutePipelineConsumerWaitOp::new(
            &mut ctx,
            pipeline,
            consumer_slot,
            consumer_phase,
            consumer,
        )
        .get_operation()
        .insert_at_back(block, &ctx);
        CutePipelineConsumerReleaseOp::new(&mut ctx, pipeline, consumer_slot, consumer)
            .get_operation()
            .insert_at_back(block, &ctx);
        CutePipelineProducerTailOp::new(
            &mut ctx,
            pipeline,
            producer_slot,
            producer_phase,
            producer,
        )
        .get_operation()
        .insert_at_back(block, &ctx);

        lower_pipeline_to_mir(&mut ctx, module).unwrap();

        let mut operations = Vec::new();
        collect_ops(&ctx, module, &mut operations);
        assert!(
            operations
                .iter()
                .all(|op| { !pipeline_semantic_ids().contains(&Operation::get_opid(*op, &ctx)) })
        );
        let count = |opid: OpId| {
            operations
                .iter()
                .filter(|op| Operation::get_opid(**op, &ctx) == opid)
                .count()
        };
        assert_eq!(count(MbarrierInitSharedOp::get_opid_static()), 6);
        assert_eq!(count(MbarrierTryWaitParitySharedOp::get_opid_static()), 5);
        assert_eq!(count(MbarrierArriveExpectTxSharedOp::get_opid_static()), 1);
        assert_eq!(count(MbarrierArriveSharedOp::get_opid_static()), 1);
        assert_eq!(
            count(FenceMbarrierInitReleaseClusterOp::get_opid_static()),
            1
        );
        assert_eq!(count(Barrier0Op::get_opid_static()), 1);
        assert_eq!(count(ReadPtxSregTidXOp::get_opid_static()), 1);
        assert_eq!(count(ReadPtxSregLaneIdOp::get_opid_static()), 1);

        let arrival = operations
            .iter()
            .copied()
            .find(|op| {
                Operation::get_opid(*op, &ctx) == MbarrierArriveExpectTxSharedOp::get_opid_static()
            })
            .unwrap();
        let full_barrier = arrival.deref(&ctx).get_operand(0);
        let copies: Vec<_> = operations
            .iter()
            .copied()
            .filter(|op| Operation::get_opid(*op, &ctx) == CuteTmaCopy2dOp::get_opid_static())
            .collect();
        assert_eq!(copies.len(), 4);
        assert!(
            copies.iter().all(|copy| {
                CuteTmaCopy2dOp::wrap(*copy).completion_barrier(&ctx) == full_barrier
            })
        );

        for wait in operations.iter().copied().filter(|op| {
            Operation::get_opid(*op, &ctx) == MbarrierTryWaitParitySharedOp::get_opid_static()
        }) {
            let poll = wait.deref(&ctx).get_parent_block().unwrap();
            let terminator = poll.deref(&ctx).get_tail().unwrap();
            assert!(Operation::get_opid(terminator, &ctx) == MirCondBranchOp::get_opid_static());
            let successors: Vec<_> = terminator.deref(&ctx).successors().collect();
            assert_eq!(
                successors[1], poll,
                "failed probe must retry the same block"
            );
            assert_ne!(successors[0], poll, "ready probe must leave the poll block");
        }

        let request_key = Identifier::try_from(GENERATED_INTRINSIC_REQUEST_ATTR).unwrap();
        for op in operations.iter().filter(|op| {
            matches!(
                Operation::get_opid(**op, &ctx),
                id if id == MbarrierInitSharedOp::get_opid_static()
                    || id == MbarrierTryWaitParitySharedOp::get_opid_static()
                    || id == MbarrierArriveExpectTxSharedOp::get_opid_static()
                    || id == MbarrierArriveSharedOp::get_opid_static()
                    || id == FenceMbarrierInitReleaseClusterOp::get_opid_static()
                    || id == Barrier0Op::get_opid_static()
                    || id == ReadPtxSregTidXOp::get_opid_static()
                    || id == ReadPtxSregLaneIdOp::get_opid_static()
            )
        }) {
            assert!(op.deref(&ctx).attributes.0.contains_key(&request_key));
        }
    }

    #[test]
    fn pipeline_expect_accepts_one_copy_when_its_tile_bytes_match() {
        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        let (_make, _expect, copy) =
            append_single_copy_pipeline_expect(&mut ctx, block, 8_192, true);
        let copy = copy.unwrap();

        lower_pipeline_to_mir(&mut ctx, module).unwrap();

        assert!(copy.deref(&ctx).get_parent_block().is_some());
        let mut operations = Vec::new();
        collect_ops(&ctx, module, &mut operations);
        assert!(operations.iter().all(|operation| {
            !pipeline_semantic_ids().contains(&Operation::get_opid(*operation, &ctx))
        }));
        assert_eq!(
            operations
                .iter()
                .filter(|operation| {
                    Operation::get_opid(**operation, &ctx)
                        == MbarrierArriveExpectTxSharedOp::get_opid_static()
                })
                .count(),
            1
        );
        let completion = CuteTmaCopy2dOp::wrap(copy).completion_barrier(&ctx);
        let completion = completion.defining_op().unwrap();
        assert!(Operation::get_opid(completion, &ctx) == MirPtrOffsetOp::get_opid_static());
    }

    #[test]
    fn semantic_tma_story_becomes_four_unchanged_leaves() {
        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u16_ty: TypeHandle = IntegerType::get(&ctx, 16, Signedness::Unsigned).into();
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let descriptor_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u8_ty, false).into();
        let shared_u8: TypeHandle =
            MirPtrType::get(&mut ctx, u8_ty, true, address_space::SHARED).into();
        let shared_u16: TypeHandle =
            MirPtrType::get(&mut ctx, u16_ty, true, address_space::SHARED).into();
        let barrier_ty: TypeHandle =
            MirPtrType::get(&mut ctx, u64_ty, true, address_space::SHARED).into();
        let descriptors = [
            undef(&mut ctx, block, descriptor_ty),
            undef(&mut ctx, block, descriptor_ty),
            undef(&mut ctx, block, descriptor_ty),
            undef(&mut ctx, block, descriptor_ty),
        ];
        let bases = [
            undef(&mut ctx, block, shared_u8),
            undef(&mut ctx, block, shared_u8),
            undef(&mut ctx, block, shared_u16),
            undef(&mut ctx, block, shared_u16),
        ];
        let capacity = undef(&mut ctx, block, u64_ty);
        let barrier = undef(&mut ctx, block, barrier_ty);
        let row = undef(&mut ctx, block, u64_ty);
        let column = undef(&mut ctx, block, u64_ty);
        let elements = [u8_ty, u8_ty, u16_ty, u16_ty];
        let layouts = [
            gemm_value_tma_layout(),
            gemm_value_tma_layout(),
            gemm_scale_tma_layout(),
            gemm_scale_tma_layout(),
        ];

        for index in 0..4 {
            append_semantic_tma_copy(
                &mut ctx,
                block,
                descriptors[index],
                bases[index],
                capacity,
                barrier,
                row,
                column,
                elements[index],
                layouts[index].clone(),
                layouts[index].clone(),
                if index < 2 { 1024 } else { 512 },
            );
        }

        lower_tma_views_to_legacy_cute(&mut ctx, module).unwrap();

        let mut operations = Vec::new();
        collect_ops(&ctx, module, &mut operations);
        for semantic in [
            CuteTmaGmemViewOp::get_opid_static(),
            CuteTmaSmemViewOp::get_opid_static(),
            CuteTmaCopy2dOp::get_opid_static(),
        ] {
            assert_eq!(
                operations
                    .iter()
                    .filter(|op| Operation::get_opid(**op, &ctx) == semantic)
                    .count(),
                0
            );
        }
        let leaves: Vec<_> = operations
            .iter()
            .copied()
            .filter(|op| Operation::get_opid(*op, &ctx) == CuteTmaLoad2dOp::get_opid_static())
            .collect();
        assert_eq!(leaves.len(), 4, "one unchanged leaf per semantic copy");
        for (index, leaf_op) in leaves.into_iter().enumerate() {
            let operation = leaf_op.deref(&ctx);
            assert_eq!(
                operation.operands().collect::<Vec<_>>(),
                vec![bases[index], barrier, descriptors[index], row, column]
            );
            let leaf = CuteTmaLoad2dOp::wrap(leaf_op);
            assert!(leaf.verify(&ctx).is_ok());
            assert_eq!(
                leaf.get_attr_tma_smem_layout(&ctx).unwrap().0,
                layouts[index]
            );
            let element: TypeHandle = leaf.get_attr_tma_elem(&ctx).unwrap().clone().into();
            assert_eq!(element, elements[index]);
        }
    }

    #[test]
    fn gemv_story_expands_to_the_native_copy_conversion_and_dot_recipe() {
        use crate::gemv_ops::{
            CuteDotOp, CuteScaledViewKTileOp, CuteScaledViewLoadOp, CuteScaledViewMakeOp,
            CuteScaledViewRowOp, CuteTensorMake2DOp,
        };

        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let global_u8: TypeHandle =
            MirPtrType::get(&mut ctx, u8_ty, false, address_space::GLOBAL).into();
        let a_values = undef(&mut ctx, block, global_u8);
        let a_scales = undef(&mut ctx, block, global_u8);
        let b_values = undef(&mut ctx, block, global_u8);
        let b_scales = undef(&mut ctx, block, global_u8);
        let len = undef(&mut ctx, block, u64_ty);
        let matrix_rows = undef(&mut ctx, block, u64_ty);
        let vector_rows = undef(&mut ctx, block, u64_ty);
        let k = undef(&mut ctx, block, u64_ty);
        let batch = undef(&mut ctx, block, u64_ty);
        let row_index = undef(&mut ctx, block, u64_ty);
        let tile_index = undef(&mut ctx, block, u64_ty);
        let acc = undef(&mut ctx, block, f32_ty);

        let append_result = |ctx: &Context, block, op: Ptr<Operation>| {
            op.insert_at_back(block, ctx);
            op.deref(ctx).get_result(0)
        };
        let make_operand = |ctx: &mut Context, values, scales, rows, role: CuteTensorRoleAttr| {
            let values =
                CuteTensorMake2DOp::new_e2m1(ctx, values, len, rows, k, role, 1).get_operation();
            let values = append_result(ctx, block, values);
            let scales = CuteTensorMake2DOp::new_ue8m0(ctx, scales, len, rows, k, role, 1, 16)
                .get_operation();
            let scales = append_result(ctx, block, scales);
            let scaled = CuteScaledViewMakeOp::new(ctx, values, scales).get_operation();
            append_result(ctx, block, scaled)
        };
        let a = make_operand(
            &mut ctx,
            a_values,
            a_scales,
            matrix_rows,
            CuteTensorRoleAttr::Mkl,
        );
        let b = make_operand(
            &mut ctx,
            b_values,
            b_scales,
            vector_rows,
            CuteTensorRoleAttr::Nkl,
        );
        let make_fragment = |ctx: &mut Context, scaled, selected_row| {
            let row = CuteScaledViewRowOp::new(ctx, scaled, batch, selected_row).get_operation();
            let row = append_result(ctx, block, row);
            let tile = CuteScaledViewKTileOp::new(ctx, row, tile_index).get_operation();
            let tile = append_result(ctx, block, tile);
            let load = CuteScaledViewLoadOp::new(ctx, tile, 16, 4).get_operation();
            append_result(ctx, block, load)
        };
        let a = make_fragment(&mut ctx, a, row_index);
        let b = make_fragment(&mut ctx, b, vector_rows);
        let dot = CuteDotOp::new(&mut ctx, a, b, acc).get_operation();
        dot.insert_at_back(block, &ctx);

        expand_cute_ops(&mut ctx, module).unwrap();

        let mut ops = Vec::new();
        collect_ops(&ctx, module, &mut ops);
        assert!(ops.iter().all(|op| {
            Operation::get_opid(*op, &ctx).dialect.to_string() != crate::CUTE_DIALECT_NAME
        }));

        let count = |opid: OpId| {
            ops.iter()
                .filter(|op| Operation::get_opid(**op, &ctx) == opid)
                .count()
        };
        assert_eq!(count(CvtRnBf16x2Ue8m0x2Op::get_opid_static()), 4);
        assert_eq!(count(CvtRnF16x2E2m1x2Op::get_opid_static()), 64);
        assert_eq!(
            count(dialect_mir::ops::MirAllocaOp::get_opid_static()),
            0,
            "register fragments stay in vector SSA and never need local storage"
        );

        let f32_muls = ops
            .iter()
            .filter(|op| Operation::get_opid(**op, &ctx) == MirMulOp::get_opid_static())
            .filter(|op| op.deref(&ctx).get_result(0).get_type(&ctx) == f32_ty)
            .count();
        let f32_adds = ops
            .iter()
            .filter(|op| Operation::get_opid(**op, &ctx) == MirAddOp::get_opid_static())
            .filter(|op| op.deref(&ctx).get_result(0).get_type(&ctx) == f32_ty)
            .count();
        assert_eq!(
            f32_muls, 192,
            "three ordered multiplies per logical K value"
        );
        assert_eq!(f32_adds, 64, "one ordered accumulation per logical K value");

        let vector_loads = ops
            .iter()
            .filter(|op| Operation::get_opid(**op, &ctx) == MirLoadOp::get_opid_static())
            .filter(|op| {
                op.deref(&ctx)
                    .get_result(0)
                    .get_type(&ctx)
                    .deref(&ctx)
                    .downcast_ref::<llvm_types::VectorType>()
                    .is_some()
            })
            .count();
        assert_eq!(
            vector_loads, 6,
            "two value loads and one scale load per fragment"
        );
    }

    fn u64_constant(
        ctx: &mut Context,
        block: Ptr<pliron::basic_block::BasicBlock>,
        value: u64,
    ) -> Value {
        let ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
        let op = Operation::new(
            ctx,
            MirConstantOp::get_concrete_op_info(),
            vec![ty.into()],
            vec![],
            vec![],
            0,
        );
        MirConstantOp::new(op).set_attr_value(
            ctx,
            IntegerAttr::new(ty, APInt::from_u64(value, NonZeroUsize::new(64).unwrap())),
        );
        op.insert_at_back(block, ctx);
        op.deref(ctx).get_result(0)
    }

    fn u32_constant(
        ctx: &mut Context,
        block: Ptr<pliron::basic_block::BasicBlock>,
        value: u32,
    ) -> Value {
        let ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
        let op = Operation::new(
            ctx,
            MirConstantOp::get_concrete_op_info(),
            vec![ty.into()],
            vec![],
            vec![],
            0,
        );
        MirConstantOp::new(op).set_attr_value(
            ctx,
            IntegerAttr::new(
                ty,
                APInt::from_u64(u64::from(value), NonZeroUsize::new(32).unwrap()),
            ),
        );
        op.insert_at_back(block, ctx);
        op.deref(ctx).get_result(0)
    }

    /// Evaluate the small integer graph emitted for a tensor tile base.
    fn eval_base_graph(ctx: &Context, value: Value) -> u64 {
        let op = value
            .defining_op()
            .expect("base graph value must have a defining operation");
        if let Some(constant) = Operation::get_op::<MirConstantOp>(op, ctx) {
            return constant
                .get_attr_value(ctx)
                .expect("constant must have a value")
                .value()
                .to_u64();
        }
        let operation = op.deref(ctx);
        let opid = Operation::get_opid(op, ctx);
        if opid == MirCastOp::get_opid_static() {
            return eval_base_graph(ctx, operation.get_operand(0));
        }
        let left = eval_base_graph(ctx, operation.get_operand(0));
        let right = eval_base_graph(ctx, operation.get_operand(1));
        if opid == MirMulOp::get_opid_static() {
            left.wrapping_mul(right)
        } else if opid == MirLtOp::get_opid_static() {
            u64::from(left < right)
        } else if opid == MirSubOp::get_opid_static() {
            left.wrapping_sub(right)
        } else if opid == MirBitOrOp::get_opid_static() {
            left | right
        } else {
            panic!("unexpected operation in tensor base graph: {opid}")
        }
    }

    #[test]
    fn tensor_base_keeps_rust_saturating_mul_semantics() {
        use crate::attributes::CuteTensorAccessAttr;

        for (tile_index, expected) in [(3, 12), (1_u64 << 62, u64::MAX)] {
            let mut ctx = Context::new();
            let (module, block) = module_top(&mut ctx);
            let elem: TypeHandle = FP32Type::get(&ctx).into();
            let ptr_ty: TypeHandle =
                MirPtrType::get(&mut ctx, elem, false, address_space::GLOBAL).into();
            let data = undef(&mut ctx, block, ptr_ty);
            let len = u64_constant(&mut ctx, block, u64::MAX);
            let tile_index = u64_constant(&mut ctx, block, tile_index);

            let make = CuteTensorMakeOp::new(
                &mut ctx,
                data,
                len,
                elem,
                elem,
                CuteTensorAccessAttr::ReadOnly,
                4,
            )
            .get_operation();
            make.insert_at_back(block, &ctx);
            let make_result = make.deref(&ctx).get_result(0);
            let divide = CuteTensorZippedDivideOp::new(&mut ctx, make_result, 4).get_operation();
            divide.insert_at_back(block, &ctx);
            let divide_result = divide.deref(&ctx).get_result(0);
            let slice = CuteTensorSliceOp::new(&mut ctx, divide_result, tile_index).get_operation();
            slice.insert_at_back(block, &ctx);
            let slice_result = slice.deref(&ctx).get_result(0);
            let base = CuteTensorBaseOp::new(&mut ctx, slice_result).get_operation();
            base.insert_at_back(block, &ctx);

            expand_cute_ops(&mut ctx, module).unwrap();

            let mut ops = Vec::new();
            collect_ops(&ctx, module, &mut ops);
            let base_roots: Vec<_> = ops
                .into_iter()
                .filter(|op| Operation::get_opid(*op, &ctx) == MirBitOrOp::get_opid_static())
                .collect();
            assert_eq!(base_roots.len(), 1);
            let value = base_roots[0].deref(&ctx).get_result(0);
            assert_eq!(eval_base_graph(&ctx, value), expected);
        }
    }

    fn mma_data_placement() -> ComposedLayout {
        ComposedLayout::new(
            Swizzle::new(2, 3, 3),
            0,
            "(128,32):(32,1)".parse().unwrap(),
            OffsetUnit::Elements,
        )
        .unwrap()
    }

    fn mma_scale_placement() -> ComposedLayout {
        ComposedLayout::new(
            Swizzle::IDENTITY,
            0,
            "(32,4):(4,1)".parse().unwrap(),
            OffsetUnit::Elements,
        )
        .unwrap()
    }

    fn epilogue_half_placement() -> ComposedLayout {
        ComposedLayout::new(
            Swizzle::new(3, 3, 3),
            0,
            "(128,64):(64,1)".parse().unwrap(),
            OffsetUnit::Elements,
        )
        .unwrap()
    }

    struct EpilogueStory {
        semantic_operations: Vec<Ptr<Operation>>,
        writer_store_block: Option<Ptr<BasicBlock>>,
        ready_block: Ptr<BasicBlock>,
    }

    /// Build the complete v0 story in source order. Runtime values are plain
    /// MIR carriers; only the operations' TypeAttrs explain the output tile.
    fn append_epilogue_story(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        split_writer_calls: bool,
    ) -> EpilogueStory {
        use crate::attributes::{
            CuteCountedCtaBarrierAttr, CuteEpilogueHalfAttr, CuteEpilogueSyncPhaseAttr,
            CuteTiledMmaPlanAttr, CuteTmaStorePipelineAttr,
        };

        let u8_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let u64_ty: TypeHandle = IntegerType::get(ctx, 64, Signedness::Unsigned).into();
        let f16_ty: TypeHandle = MirFP16Type::get(ctx).into();
        let f32_ty: TypeHandle = FP32Type::get(ctx).into();
        let shared_f16: TypeHandle =
            MirPtrType::get(ctx, f16_ty, true, address_space::SHARED).into();
        let descriptor_ty: TypeHandle = MirPtrType::get_generic(ctx, u8_ty, false).into();
        let accumulator_ty: TypeHandle = MirArrayType::get(ctx, f32_ty, 64).into();
        let tiled_mma = CuteTiledMmaPlanAttr::mxf4_128x128x128(mma_data_placement());
        let epilogue_plan = crate::attributes::CuteEpiloguePlanAttr::sm120_mxf4_128x128(tiled_mma);
        let tile: TypeHandle = CuteEpilogueTileType::get(ctx, f16_ty, epilogue_plan).into();
        let barrier = CuteCountedCtaBarrierAttr::new(2, 0, 8, 9, 32);
        let pipeline = CuteTmaStorePipelineAttr::new(1);

        let base = undef(ctx, block, shared_f16);
        let warp = undef(ctx, block, u64_ty);
        let lane = undef(ctx, block, u32_ty);
        let accumulator = undef(ctx, block, accumulator_ty);
        let descriptor = undef(ctx, block, descriptor_ty);
        let row = undef(ctx, block, u64_ty);
        let left_column = undef(ctx, block, u64_ty);
        let one = u64_constant(ctx, block, 1);
        let right_column = Operation::new(
            ctx,
            MirAddOp::get_concrete_op_info(),
            vec![u64_ty],
            vec![left_column, one],
            vec![],
            0,
        );
        right_column.insert_at_back(block, ctx);
        let right_column = right_column.deref(ctx).get_result(0);

        let mut semantic_operations = Vec::new();
        let overlay = CuteEpilogueSmemOverlayOp::new(ctx, base, tile).get_operation();
        overlay.insert_at_back(block, ctx);
        semantic_operations.push(overlay);
        let overlay_base = overlay.deref(ctx).get_result(0);

        let slice =
            CuteEpilogueWarpSliceOp::new(ctx, overlay_base, warp, lane, tile).get_operation();
        slice.insert_at_back(block, ctx);
        semantic_operations.push(slice);
        let slice_base = slice.deref(ctx).get_result(0);
        let slice_warp = slice.deref(ctx).get_result(1);
        let slice_lane = slice.deref(ctx).get_result(2);

        let acquire = CuteTmaStoreAcquireOp::new(ctx, pipeline).get_operation();
        acquire.insert_at_back(block, ctx);
        semantic_operations.push(acquire);
        let reusable = CuteEpilogueSyncOp::new(
            ctx,
            overlay_base,
            tile,
            barrier,
            CuteEpilogueSyncPhaseAttr::Reusable,
        )
        .get_operation();
        reusable.insert_at_back(block, ctx);
        semantic_operations.push(reusable);
        let (store_block, ready_block) = if split_writer_calls {
            let store_block = BasicBlock::new(ctx, None, vec![]);
            store_block.insert_after(ctx, block);
            let ready_block = BasicBlock::new(ctx, None, vec![]);
            ready_block.insert_after(ctx, store_block);
            append_pipeline_goto(ctx, block, &Location::Unknown, store_block);
            (store_block, ready_block)
        } else {
            (block, block)
        };
        let store_fragment = CuteEpilogueStoreFragmentOp::new(
            ctx,
            slice_base,
            slice_warp,
            slice_lane,
            accumulator,
            tile,
        )
        .get_operation();
        store_fragment.insert_at_back(store_block, ctx);
        semantic_operations.push(store_fragment);
        if split_writer_calls {
            append_pipeline_goto(ctx, store_block, &Location::Unknown, ready_block);
        }
        let ready = CuteEpilogueSyncOp::new(
            ctx,
            overlay_base,
            tile,
            barrier,
            CuteEpilogueSyncPhaseAttr::ReadyForTma,
        )
        .get_operation();
        ready.insert_at_back(ready_block, ctx);
        semantic_operations.push(ready);

        for (half_index, column) in [(0, left_column), (1, right_column)] {
            let half =
                CuteEpilogueHalfOp::new(ctx, overlay_base, tile, CuteEpilogueHalfAttr(half_index))
                    .get_operation();
            half.insert_at_back(ready_block, ctx);
            semantic_operations.push(half);
            let half_base = half.deref(ctx).get_result(0);
            let half_capacity = half.deref(ctx).get_result(1);
            let source = CuteTmaSmemViewOp::new(
                ctx,
                half_base,
                half_capacity,
                f16_ty,
                epilogue_half_placement(),
                1024,
            )
            .get_operation();
            source.insert_at_back(ready_block, ctx);
            let source_view = source.deref(ctx).get_result(0);
            let destination = CuteTmaGmemViewOp::new_destination(
                ctx,
                descriptor,
                f16_ty,
                epilogue_half_placement(),
            )
            .get_operation();
            destination.insert_at_back(ready_block, ctx);
            let destination_view = destination.deref(ctx).get_result(0);
            let store =
                CuteTmaStore2dSemanticOp::new(ctx, source_view, destination_view, row, column)
                    .get_operation();
            store.insert_at_back(ready_block, ctx);
            semantic_operations.push(store);
        }
        let commit = CuteTmaStoreCommitOp::new(ctx, pipeline).get_operation();
        commit.insert_at_back(ready_block, ctx);
        semantic_operations.push(commit);
        let tail = CuteTmaStoreTailOp::new(ctx, pipeline).get_operation();
        tail.insert_at_back(ready_block, ctx);
        semantic_operations.push(tail);

        EpilogueStory {
            semantic_operations,
            writer_store_block: split_writer_calls.then_some(store_block),
            ready_block,
        }
    }

    fn mma_smem_view(
        ctx: &mut Context,
        storage: TypeHandle,
        alignment: u64,
        format: crate::attributes::CuteTensorFormatAttr,
        role: CuteTensorRoleAttr,
        layout: crate::attributes::CuteTensorLayoutAttr,
        placement: ComposedLayout,
    ) -> TypeHandle {
        use crate::attributes::{CuteTensorAccessAttr, CuteTensorAddressSpaceAttr};

        let logical: TypeHandle = FP32Type::get(ctx).into();
        let tensor: TypeHandle = CuteTensorViewType::get_with_facts(
            ctx,
            logical,
            storage,
            CuteTensorAddressSpaceAttr::Smem,
            CuteTensorAccessAttr::ReadOnly,
            alignment,
            format,
            role,
            layout,
        )
        .into();
        CuteSmemTensorType::get(ctx, tensor, placement).into()
    }

    #[test]
    fn shared_mma_story_expands_to_ten_scales_six_matrix_loads_and_sixteen_mmas() {
        use crate::attributes::{
            CuteMmaCarrierKindAttr, CuteTensorFormatAttr, CuteTensorLayoutAttr,
        };
        use dialect_nvvm::ops::{LdmatrixOp, RegisterMmaOp};
        use pliron::identifier::Identifier;

        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let u64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let f16_ty: TypeHandle = MirFP16Type::get(&ctx).into();
        let data_ptr: TypeHandle =
            MirPtrType::get(&mut ctx, f16_ty, true, address_space::SHARED).into();
        let scale_ptr: TypeHandle =
            MirPtrType::get(&mut ctx, u32_ty, true, address_space::SHARED).into();
        let a_view = mma_smem_view(
            &mut ctx,
            f16_ty,
            16,
            CuteTensorFormatAttr::E2M1,
            CuteTensorRoleAttr::Mkl,
            CuteTensorLayoutAttr::KMajor,
            mma_data_placement(),
        );
        let b_view = mma_smem_view(
            &mut ctx,
            f16_ty,
            16,
            CuteTensorFormatAttr::E2M1,
            CuteTensorRoleAttr::Nkl,
            CuteTensorLayoutAttr::KMajor,
            mma_data_placement(),
        );
        let scale_a_view = mma_smem_view(
            &mut ctx,
            u32_ty,
            4,
            CuteTensorFormatAttr::UE8M0,
            CuteTensorRoleAttr::Mkl,
            CuteTensorLayoutAttr::BlockScaleKMajor(32),
            mma_scale_placement(),
        );
        let scale_b_view = mma_smem_view(
            &mut ctx,
            u32_ty,
            4,
            CuteTensorFormatAttr::UE8M0,
            CuteTensorRoleAttr::Nkl,
            CuteTensorLayoutAttr::BlockScaleKMajor(32),
            mma_scale_placement(),
        );
        let plan = crate::attributes::CuteTiledMmaPlanAttr::mxf4_128x128x128(mma_data_placement());
        let a_carrier: TypeHandle = MirArrayType::get(&mut ctx, u32_ty, 8).into();
        let scale_carrier: TypeHandle = MirArrayType::get(&mut ctx, u32_ty, 10).into();
        let accumulator_carrier: TypeHandle = MirArrayType::get(&mut ctx, f32_ty, 64).into();
        let capacity = undef(&mut ctx, block, u64_ty);

        let overlay = |ctx: &mut Context,
                       block: Ptr<BasicBlock>,
                       pointer_ty: TypeHandle,
                       view: TypeHandle,
                       capacity: Value| {
            let base = undef(ctx, block, pointer_ty);
            let operation = CuteSmemTensorOverlayOp::new(ctx, base, capacity, view).get_operation();
            operation.insert_at_back(block, ctx);
            (
                operation.deref(ctx).get_result(0),
                operation.deref(ctx).get_result(1),
            )
        };
        let (a_base, a_capacity) = overlay(&mut ctx, block, data_ptr, a_view, capacity);
        let (b_base, b_capacity) = overlay(&mut ctx, block, data_ptr, b_view, capacity);
        let (scale_a_base, scale_a_capacity) =
            overlay(&mut ctx, block, scale_ptr, scale_a_view, capacity);
        let (scale_b_base, scale_b_capacity) =
            overlay(&mut ctx, block, scale_ptr, scale_b_view, capacity);

        let lane = undef(&mut ctx, block, u32_ty);
        let warp_m = undef(&mut ctx, block, u64_ty);
        let warp_n = undef(&mut ctx, block, u64_ty);
        let k_half = undef(&mut ctx, block, u64_ty);
        let slice = CuteTiledMmaSliceOp::new(&mut ctx, lane, plan.clone()).get_operation();
        slice.insert_at_back(block, &ctx);
        let lane = slice.deref(&ctx).get_result(0);
        let zero = undef(&mut ctx, block, f32_ty);
        let fill = CuteFragmentFillOp::new(
            &mut ctx,
            zero,
            accumulator_carrier,
            plan.clone(),
            CuteMmaCarrierKindAttr::Accumulator,
        )
        .get_operation();
        fill.insert_at_back(block, &ctx);

        let scales = CuteMmaLoadScalesOp::new(
            &mut ctx,
            lane,
            scale_a_base,
            scale_a_capacity,
            scale_b_base,
            scale_b_capacity,
            warp_m,
            warp_n,
            scale_a_view,
            scale_b_view,
            scale_carrier,
            plan.clone(),
        )
        .get_operation();
        scales.insert_at_back(block, &ctx);
        let stage_scales = scales.deref(&ctx).get_result(0);
        let selected_scales =
            CuteFragmentSliceKOp::new(&mut ctx, stage_scales, k_half, scale_carrier, plan.clone())
                .get_operation();
        selected_scales.insert_at_back(block, &ctx);
        let a = CuteMmaLoadAOp::new(
            &mut ctx,
            lane,
            a_base,
            a_capacity,
            warp_m,
            k_half,
            a_view,
            a_carrier,
            plan.clone(),
        )
        .get_operation();
        a.insert_at_back(block, &ctx);
        let b = CuteMmaPartitionBOp::new(
            &mut ctx,
            b_base,
            b_capacity,
            warp_n,
            k_half,
            b_view,
            plan.clone(),
        )
        .get_operation();
        b.insert_at_back(block, &ctx);
        let a_fragment = a.deref(&ctx).get_result(0);
        let b_base = b.deref(&ctx).get_result(0);
        let b_capacity = b.deref(&ctx).get_result(1);
        let b_warp_n = b.deref(&ctx).get_result(2);
        let b_k_half = b.deref(&ctx).get_result(3);
        let selected_scales = selected_scales.deref(&ctx).get_result(0);
        let accumulator = fill.deref(&ctx).get_result(0);
        let gemm = CuteTiledGemmOp::new(
            &mut ctx,
            lane,
            a_fragment,
            b_base,
            b_capacity,
            b_warp_n,
            b_k_half,
            selected_scales,
            accumulator,
            b_view,
            plan,
        )
        .get_operation();
        gemm.insert_at_back(block, &ctx);

        lower_smem_mma_to_mir(&mut ctx, module).unwrap();
        let mut operations = Vec::new();
        collect_ops(&ctx, module, &mut operations);
        for semantic in smem_mma_semantic_ids() {
            assert_eq!(
                operations
                    .iter()
                    .filter(|op| Operation::get_opid(**op, &ctx) == semantic)
                    .count(),
                0
            );
        }
        assert_eq!(
            operations
                .iter()
                .filter(|op| Operation::get_opid(**op, &ctx) == MirLoadOp::get_opid_static())
                .count(),
            10
        );
        assert_eq!(
            operations
                .iter()
                .filter(|op| Operation::get_opid(**op, &ctx) == LdmatrixOp::get_opid_static())
                .count(),
            6
        );
        assert_eq!(
            operations
                .iter()
                .filter(|op| Operation::get_opid(**op, &ctx) == RegisterMmaOp::get_opid_static())
                .count(),
            16
        );
        let hot_order: Vec<_> = block
            .deref(&ctx)
            .iter(&ctx)
            .filter_map(|operation| {
                let id = Operation::get_opid(operation, &ctx);
                if id == LdmatrixOp::get_opid_static() {
                    Some('L')
                } else if id == RegisterMmaOp::get_opid_static() {
                    Some('M')
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            hot_order,
            vec![
                'L', 'L', // A0, A1
                'L', 'M', 'M', 'M', 'M', // B pair 0, then its four cells
                'L', 'M', 'M', 'M', 'M', // B pair 1
                'L', 'M', 'M', 'M', 'M', // B pair 2
                'L', 'M', 'M', 'M', 'M', // B pair 3
            ]
        );
        for operation in &operations {
            let id = Operation::get_opid(*operation, &ctx);
            if id == LdmatrixOp::get_opid_static() {
                assert!(LdmatrixOp::new(*operation).verify(&ctx).is_ok());
            } else if id == RegisterMmaOp::get_opid_static() {
                assert!(RegisterMmaOp::new(*operation).verify(&ctx).is_ok());
            }
        }
        let request = Identifier::try_from(GENERATED_INTRINSIC_REQUEST_ATTR).unwrap();
        assert!(
            operations
                .iter()
                .filter(|op| {
                    let id = Operation::get_opid(**op, &ctx);
                    id == LdmatrixOp::get_opid_static() || id == RegisterMmaOp::get_opid_static()
                })
                .all(|op| op.deref(&ctx).attributes.0.contains_key(&request))
        );
    }

    #[test]
    fn shared_mma_preflight_failure_does_not_expand_an_earlier_fragment() {
        use crate::attributes::CuteMmaCarrierKindAttr;

        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        let f32_ty: TypeHandle = FP32Type::get(&ctx).into();
        let good_ty: TypeHandle = MirArrayType::get(&mut ctx, f32_ty, 64).into();
        let bad_ty: TypeHandle = MirArrayType::get(&mut ctx, f32_ty, 63).into();
        let fill = undef(&mut ctx, block, f32_ty);
        let plan = crate::attributes::CuteTiledMmaPlanAttr::mxf4_128x128x128(mma_data_placement());
        let good = CuteFragmentFillOp::new(
            &mut ctx,
            fill,
            good_ty,
            plan.clone(),
            CuteMmaCarrierKindAttr::Accumulator,
        )
        .get_operation();
        good.insert_at_back(block, &ctx);
        let bad = CuteFragmentFillOp::new(
            &mut ctx,
            fill,
            bad_ty,
            plan,
            CuteMmaCarrierKindAttr::Accumulator,
        )
        .get_operation();
        bad.insert_at_back(block, &ctx);

        let error = lower_smem_mma_to_mir(&mut ctx, module)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("must contain exactly 64 F32 registers"),
            "{error}"
        );
        assert!(good.deref(&ctx).get_parent_block().is_some());
        assert!(bad.deref(&ctx).get_parent_block().is_some());
        let mut operations = Vec::new();
        collect_ops(&ctx, module, &mut operations);
        assert_eq!(
            operations
                .iter()
                .filter(|op| {
                    Operation::get_opid(**op, &ctx) == MirConstructArrayOp::get_opid_static()
                })
                .count(),
            0
        );
    }

    #[test]
    fn epilogue_story_expands_to_the_exact_store_protocol() {
        use pliron::identifier::Identifier;

        let mut ctx = Context::new();
        let (module, block) = module_top(&mut ctx);
        let story = append_epilogue_story(&mut ctx, block, true);
        let writer_store_block = story
            .writer_store_block
            .expect("split writer story has a store block");
        let ready_block = story.ready_block;

        lower_epilogue_to_legacy_cute(&mut ctx, module).unwrap();

        let mut operations = Vec::new();
        collect_ops(&ctx, module, &mut operations);
        for semantic in epilogue_semantic_ids() {
            assert_eq!(
                operations
                    .iter()
                    .filter(|op| Operation::get_opid(**op, &ctx) == semantic)
                    .count(),
                0,
                "semantic epilogue operation `{semantic}` survived Backend A"
            );
        }
        for operation in story.semantic_operations {
            assert!(!operations.contains(&operation));
        }

        let count = |id: OpId| {
            operations
                .iter()
                .filter(|operation| Operation::get_opid(**operation, &ctx) == id)
                .count()
        };
        assert_eq!(count(CvtF16x2F32Op::get_opid_static()), 32);
        assert_eq!(count(StmatrixM8n8X2Op::get_opid_static()), 16);
        assert_eq!(count(BarrierCtaSyncAlignedCountOp::get_opid_static()), 2);
        assert_eq!(count(CpAsyncBulkWaitGroupReadOp::get_opid_static()), 2);
        assert_eq!(count(CpAsyncBulkCommitGroupOp::get_opid_static()), 1);
        assert_eq!(count(FenceProxyAsyncSharedCtaOp::get_opid_static()), 1);
        assert_eq!(count(CuteTmaStore2dOp::get_opid_static()), 2);
        assert_eq!(count(CuteTmaGmemViewOp::get_opid_static()), 0);
        assert_eq!(count(CuteTmaSmemViewOp::get_opid_static()), 0);

        let ready_sync_order: Vec<_> = ready_block
            .deref(&ctx)
            .iter(&ctx)
            .filter_map(|operation| {
                let id = Operation::get_opid(operation, &ctx);
                if id == FenceProxyAsyncSharedCtaOp::get_opid_static() {
                    Some('F')
                } else if id == BarrierCtaSyncAlignedCountOp::get_opid_static() {
                    Some('B')
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(ready_sync_order, ['F', 'B']);

        // Each accumulator atom is converted top/bottom immediately before
        // its one matrix store. Keeping this visual order avoids widening
        // all 32 packed values' live ranges at once.
        let packed_store_order: Vec<_> = writer_store_block
            .deref(&ctx)
            .iter(&ctx)
            .filter_map(|operation| {
                let id = Operation::get_opid(operation, &ctx);
                if id == CvtF16x2F32Op::get_opid_static() {
                    Some('C')
                } else if id == StmatrixM8n8X2Op::get_opid_static() {
                    Some('S')
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(packed_store_order, ['C', 'C', 'S'].repeat(16));

        let request = Identifier::try_from(GENERATED_INTRINSIC_REQUEST_ATTR).unwrap();
        assert!(
            operations
                .iter()
                .filter(|operation| {
                    let id = Operation::get_opid(**operation, &ctx);
                    id == CvtF16x2F32Op::get_opid_static()
                        || id == StmatrixM8n8X2Op::get_opid_static()
                        || id == BarrierCtaSyncAlignedCountOp::get_opid_static()
                        || id == FenceProxyAsyncSharedCtaOp::get_opid_static()
                        || id == CpAsyncBulkWaitGroupReadOp::get_opid_static()
                        || id == CpAsyncBulkCommitGroupOp::get_opid_static()
                })
                .all(|operation| operation.deref(&ctx).attributes.0.contains_key(&request))
        );
    }
}
