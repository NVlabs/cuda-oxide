/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `MirToLlvmConversion` implementations for all MIR and NVVM ops.
//!
//! Each impl delegates directly to a per-op conversion function,
//! bypassing the old category sub-enum dispatch.

use pliron::{
    builtin::ops::ConstantOp,
    context::Context,
    derive::op_interface_impl,
    irbuild::{
        dialect_conversion::{DialectConversionRewriter, OperandsInfo},
        rewriter::Rewriter,
    },
    op::Op,
    result::Result,
};

use llvm_export::attributes::{FCmpPredicateAttr, ICmpPredicateAttr};

use crate::conversion_interface::MirToLlvmConversion;

use dialect_mir::ops::{
    MirAddOp, MirAllocaOp, MirArrayElementAddrOp, MirAssertOp, MirBitAndOp, MirBitOrOp,
    MirBitXorOp, MirCallOp, MirCastOp, MirCheckedAddOp, MirCheckedMulOp, MirCheckedSubOp, MirCmpOp,
    MirCondBranchOp, MirConstantOp, MirConstructArrayOp, MirConstructEnumOp, MirConstructSliceOp,
    MirConstructStructOp, MirConstructTupleOp, MirDbgValueOp, MirDivOp, MirEnumPayloadOp, MirEqOp,
    MirExtractArrayElementOp, MirExtractFieldOp, MirFieldAddrOp, MirFloatConstantOp, MirGeOp,
    MirGetDiscriminantOp, MirGotoOp, MirGtOp, MirInsertFieldOp, MirLeOp, MirLoadOp, MirLtOp,
    MirMemcpyOp, MirMulOp, MirNeOp, MirNegOp, MirNotOp, MirPtrOffsetOp, MirRefOp, MirRemOp,
    MirReturnOp, MirShlOp, MirShrOp, MirStorageDeadOp, MirStorageLiveOp, MirStoreOp, MirSubOp,
    MirUndefOp, MirUnreachableOp, MirUnrollHintOp,
};
use dialect_nvvm::ops::{
    AbsBf16x2Op, ActiveMaskOp, AddBf16x2Op, BarWarpSyncOp, Barrier0Op, BreakpointOp,
    ClcQueryGetFirstCtaidXOp, ClcQueryGetFirstCtaidYOp, ClcQueryGetFirstCtaidZOp,
    ClcQueryIsCanceledOp, ClcTryCancelMulticastOp, ClcTryCancelOp, ClusterSyncOp,
    CpAsyncBulkCommitGroupOp, CpAsyncBulkTensorG2sTile1dOp,
    CpAsyncBulkTensorG2sTile2dMulticastCg2Op, CpAsyncBulkTensorG2sTile2dMulticastOp,
    CpAsyncBulkTensorG2sTile2dOp, CpAsyncBulkTensorG2sTile3dOp, CpAsyncBulkTensorG2sTile4dOp,
    CpAsyncBulkTensorG2sTile5dOp, CpAsyncBulkTensorS2gTile1dOp, CpAsyncBulkTensorS2gTile2dOp,
    CpAsyncBulkTensorS2gTile3dOp, CpAsyncBulkTensorS2gTile4dOp, CpAsyncBulkTensorS2gTile5dOp,
    CpAsyncBulkWaitGroupOp, CpAsyncBulkWaitGroupReadOp, CpAsyncCa4Op, CpAsyncCa8Op,
    CpAsyncCaZfill4Op, CpAsyncCaZfill8Op, CpAsyncCaZfill16Op, CvtF16x2F32Op, CvtF32x2Bf16x2Op,
    CvtRnReluBf16x2F32Op, CvtRnReluF16x2F32Op, CvtRzBf16x2F32Op, CvtRzF16x2F32Op, Dp2aS32Op,
    Dp2aU32Op, Dp4aS32Op, Dp4aU32Op, DsmemReadU32Op, ElectSyncOp, FenceProxyAsyncSharedCtaOp,
    FmaBf16x2Op, FmaReluBf16x2Op, InlinePtxOp, MapaSharedClusterOp, MatchAllSyncI32Op,
    MatchAllSyncI64Op, MatchAnySyncI32Op, MatchAnySyncI64Op, MaxBf16x2Op, MbarrierArriveClusterOp,
    MbarrierArriveExpectTxSharedOp, MbarrierArriveSharedOp, MbarrierInitSharedOp,
    MbarrierInvalSharedOp, MbarrierTestWaitSharedOp, MbarrierTryWaitParitySharedOp,
    MbarrierTryWaitSharedOp, MinBf16x2Op, MulBf16x2Op, NanosleepOp, NegBf16x2Op,
    NvvmAtomicCmpxchgOp, NvvmAtomicLoadOp, NvvmAtomicRmwOp, NvvmAtomicStoreOp, PmEventOp,
    ReadPtxSregClock64Op, ReadPtxSregClockOp, ReadPtxSregClusterCtaidXOp,
    ReadPtxSregClusterCtaidYOp, ReadPtxSregClusterCtaidZOp, ReadPtxSregClusterIdxOp,
    ReadPtxSregClusterNctaidXOp, ReadPtxSregClusterNctaidYOp, ReadPtxSregClusterNctaidZOp,
    ReadPtxSregCtaidXOp, ReadPtxSregCtaidYOp, ReadPtxSregCtaidZOp, ReadPtxSregEnvReg1Op,
    ReadPtxSregEnvReg2Op, ReadPtxSregGlobaltimerOp, ReadPtxSregLaneIdOp, ReadPtxSregLanemaskEqOp,
    ReadPtxSregLanemaskGeOp, ReadPtxSregLanemaskGtOp, ReadPtxSregLanemaskLeOp,
    ReadPtxSregLanemaskLtOp, ReadPtxSregNclusterIdOp, ReadPtxSregNctaidXOp, ReadPtxSregNctaidYOp,
    ReadPtxSregNctaidZOp, ReadPtxSregNtidXOp, ReadPtxSregNtidYOp, ReadPtxSregNtidZOp,
    ReadPtxSregTidXOp, ReadPtxSregTidYOp, ReadPtxSregTidZOp, ReduxSyncAddOp, ReduxSyncAndOp,
    ReduxSyncMaxOp, ReduxSyncMinOp, ReduxSyncOrOp, ReduxSyncUmaxOp, ReduxSyncUminOp,
    ReduxSyncXorOp, ShflSyncBflyF32Op, ShflSyncBflyI32Op, ShflSyncBflyI64Op, ShflSyncDownF32Op,
    ShflSyncDownI32Op, ShflSyncDownI64Op, ShflSyncIdxF32Op, ShflSyncIdxI32Op, ShflSyncIdxI64Op,
    ShflSyncUpF32Op, ShflSyncUpI32Op, ShflSyncUpI64Op, StmatrixM8n8X2Op, StmatrixM8n8X2TransOp,
    StmatrixM8n8X4Op, StmatrixM8n8X4TransOp, SubBf16x2Op, Tcgen05AllocCg2Op, Tcgen05AllocOp,
    Tcgen05CommitCg2Op, Tcgen05CommitMulticastCg2Op, Tcgen05CommitOp,
    Tcgen05CommitSharedClusterCg2Op, Tcgen05CommitSharedClusterOp, Tcgen05CpSmemToTmemCg2Op,
    Tcgen05CpSmemToTmemOp, Tcgen05DeallocCg2Op, Tcgen05DeallocOp, Tcgen05FenceAfterThreadSyncOp,
    Tcgen05FenceBeforeThreadSyncOp, Tcgen05Ld16x256bPureOp, Tcgen05Ld16x256bX8PureOp,
    Tcgen05LoadWaitOp, Tcgen05MmaF16Cg2Op, Tcgen05MmaF16Op, Tcgen05MmaWsBf16Op, Tcgen05MmaWsF16Op,
    Tcgen05MmaWsTf32Op, Tcgen05RelinquishAllocPermitCg2Op, Tcgen05RelinquishAllocPermitOp,
    Tcgen05StoreWaitOp, ThreadfenceBlockOp, ThreadfenceOp, ThreadfenceSystemOp, TrapOp,
    VoteSyncAllOp, VoteSyncAnyOp, VoteSyncBallotOp, VprintfOp, WgmmaCommitGroupSyncAlignedOp,
    WgmmaFenceSyncAlignedOp, WgmmaMakeSmemDescOp, WgmmaMmaM64N64K16F32Bf16Op,
    WgmmaWaitGroupSyncAlignedOp,
};

// ---- Arithmetic ops --------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for MirAddOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_add(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirCheckedAddOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_checked_add(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirSubOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_sub(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirCheckedSubOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_checked_sub(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirMulOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_mul(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirCheckedMulOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_checked_mul(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirDivOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_div(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirRemOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_rem(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirShrOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_shr(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirShlOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_shl(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirBitAndOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_bitand(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirBitOrOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_bitor(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirBitXorOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_bitxor(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirNotOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_not(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirNegOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_neg(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirLtOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_cmp(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            ICmpPredicateAttr::SLT,
            ICmpPredicateAttr::ULT,
            FCmpPredicateAttr::OLT,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirLeOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_cmp(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            ICmpPredicateAttr::SLE,
            ICmpPredicateAttr::ULE,
            FCmpPredicateAttr::OLE,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirGtOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_cmp(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            ICmpPredicateAttr::SGT,
            ICmpPredicateAttr::UGT,
            FCmpPredicateAttr::OGT,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirGeOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_cmp(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            ICmpPredicateAttr::SGE,
            ICmpPredicateAttr::UGE,
            FCmpPredicateAttr::OGE,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirCmpOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_three_way_cmp(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirEqOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_cmp(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            ICmpPredicateAttr::EQ,
            ICmpPredicateAttr::EQ,
            FCmpPredicateAttr::OEQ,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirNeOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::arithmetic::convert_cmp(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            ICmpPredicateAttr::NE,
            ICmpPredicateAttr::NE,
            FCmpPredicateAttr::UNE,
        )
    }
}

// ---- Memory ops ------------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for MirAllocaOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::memory::convert_alloca(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirStoreOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::memory::convert_store(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirMemcpyOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::memory::convert_memcpy(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirLoadOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::memory::convert_load(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirDbgValueOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::memory::convert_dbg_value(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirRefOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::memory::convert_ref(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirPtrOffsetOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::memory::convert_ptr_offset(ctx, rewriter, self.get_operation(), operands_info)
    }
}

// ---- Constant ops ----------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for MirConstantOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::constants::convert_integer(ctx, rewriter, self.get_operation(), operands_info)
    }
}

/// A `builtin.constant` that `sccp` materialised carries a signed/unsigned MIR
/// integer type; normalise it to a signless constant, exactly like `mir.constant`.
/// Only a non-signless `builtin.constant` reaches here (see `can_convert_op`), so
/// the emitted signless constant is final and the conversion converges.
#[op_interface_impl]
impl MirToLlvmConversion for ConstantOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let value = self.get_value(ctx);
        super::ops::constants::convert_builtin_constant(ctx, rewriter, self.get_operation(), value)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirFloatConstantOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::constants::convert_float(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirUndefOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::constants::convert_undef(ctx, rewriter, self.get_operation(), operands_info)
    }
}

// ---- Cast op ---------------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for MirCastOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::cast::convert(ctx, rewriter, self.get_operation(), operands_info)
    }
}

// ---- Aggregate ops ---------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for MirExtractFieldOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_extract_field(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirInsertFieldOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_insert_field(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirConstructStructOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_construct_struct(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirConstructTupleOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_construct_tuple(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirConstructSliceOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_construct_slice(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirConstructArrayOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_construct_array(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirExtractArrayElementOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_extract_array_element(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirConstructEnumOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_construct_enum(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirGetDiscriminantOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_get_discriminant(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirEnumPayloadOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_enum_payload(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirFieldAddrOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_field_addr(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirArrayElementAddrOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::aggregate::convert_array_element_addr(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

// ---- Control flow ops ------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for MirReturnOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::control_flow::convert_return(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirGotoOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::control_flow::convert_goto(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirCondBranchOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::control_flow::convert_cond_branch(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirAssertOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::control_flow::convert_assert(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirUnreachableOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::control_flow::convert_unreachable(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

// ---- Call op ---------------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for MirCallOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::ops::call::convert(ctx, rewriter, self.get_operation(), operands_info)
    }
}

// ---- No-op markers (StorageLive / StorageDead) -----------------------------

#[op_interface_impl]
impl MirToLlvmConversion for MirStorageLiveOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MirStorageDeadOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}

// Safety net: the loop-unroll pass consumes every `mir.unroll_hint` before
// lowering, so one should never reach here. But if unrolling is skipped (e.g.
// a debug build that bypasses the pass), drop the hint rather than fail the
// conversion: it carries no runtime semantics, only a request to unroll.
#[op_interface_impl]
impl MirToLlvmConversion for MirUnrollHintOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        rewriter.erase_operation(ctx, self.get_operation());
        Ok(())
    }
}

// ---- NVVM Basic ops --------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for InlinePtxOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::asm::convert_inline_ptx(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregTidXOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_sreg_read_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_read_ptx_sreg_tid_x",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregTidYOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_sreg_read_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_read_ptx_sreg_tid_y",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregCtaidXOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_sreg_read_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_read_ptx_sreg_ctaid_x",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregCtaidYOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_sreg_read_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_read_ptx_sreg_ctaid_y",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregNtidXOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_sreg_read_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_read_ptx_sreg_ntid_x",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregNtidYOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_sreg_read_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_read_ptx_sreg_ntid_y",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregTidZOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_sreg_read_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_read_ptx_sreg_tid_z",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregCtaidZOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_sreg_read_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_read_ptx_sreg_ctaid_z",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregNtidZOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_sreg_read_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_read_ptx_sreg_ntid_z",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregNctaidXOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_sreg_read_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_read_ptx_sreg_nctaid_x",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregNctaidYOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_sreg_read_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_read_ptx_sreg_nctaid_y",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregNctaidZOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_sreg_read_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_read_ptx_sreg_nctaid_z",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregEnvReg1Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_sreg_read_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_read_ptx_sreg_envreg1",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregEnvReg2Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_sreg_read_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_read_ptx_sreg_envreg2",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregLaneIdOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_sreg_read_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_read_ptx_sreg_laneid",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregLanemaskLtOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_sreg_read_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_read_ptx_sreg_lanemask_lt",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregLanemaskLeOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_sreg_read_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_read_ptx_sreg_lanemask_le",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregLanemaskEqOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_sreg_read_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_read_ptx_sreg_lanemask_eq",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregLanemaskGeOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_sreg_read_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_read_ptx_sreg_lanemask_ge",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregLanemaskGtOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_sreg_read_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_read_ptx_sreg_lanemask_gt",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Barrier0Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_barrier0(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ThreadfenceBlockOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_threadfence_block(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ThreadfenceOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_threadfence(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ThreadfenceSystemOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::basic::convert_threadfence_system(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

// ---- NVVM Debug ops --------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregClockOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::debug::convert_clock(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregClock64Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::debug::convert_clock64(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregGlobaltimerOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::debug::convert_globaltimer(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for TrapOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::debug::convert_trap(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for BreakpointOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::debug::convert_breakpoint(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for PmEventOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::debug::convert_pm_event(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for VprintfOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::debug::convert_vprintf(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

// ---- NVVM Cluster ops ------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregClusterCtaidXOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::cluster::convert_cluster_sreg(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "%cluster_ctaid.x",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregClusterCtaidYOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::cluster::convert_cluster_sreg(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "%cluster_ctaid.y",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregClusterCtaidZOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::cluster::convert_cluster_sreg(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "%cluster_ctaid.z",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregClusterNctaidXOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::cluster::convert_cluster_sreg(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "%cluster_nctaid.x",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregClusterNctaidYOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::cluster::convert_cluster_sreg(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "%cluster_nctaid.y",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregClusterNctaidZOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::cluster::convert_cluster_sreg(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "%cluster_nctaid.z",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregClusterIdxOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::cluster::convert_cluster_sreg(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "%cluster_idx",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReadPtxSregNclusterIdOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::cluster::convert_cluster_sreg(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "%nclusterid",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ClusterSyncOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::cluster::convert_cluster_sync(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MapaSharedClusterOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::cluster::convert_mapa_shared_cluster(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for DsmemReadU32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::cluster::convert_dsmem_read_u32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

// ---- NVVM Warp ops ---------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for ShflSyncIdxI32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_shuffle_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_shfl_sync_idx_i32",
            31,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ShflSyncBflyI32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_shuffle_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_shfl_sync_bfly_i32",
            31,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ShflSyncDownI32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_shuffle_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_shfl_sync_down_i32",
            31,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ShflSyncUpI32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_shuffle_i32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_shfl_sync_up_i32",
            0,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ShflSyncIdxF32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_shuffle_f32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_shfl_sync_idx_f32",
            31,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ShflSyncBflyF32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_shuffle_f32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_shfl_sync_bfly_f32",
            31,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ShflSyncDownF32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_shuffle_f32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_shfl_sync_down_f32",
            31,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ShflSyncUpF32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_shuffle_f32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_shfl_sync_up_f32",
            0,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ShflSyncIdxI64Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_shuffle_i64(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "idx",
            31,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ShflSyncBflyI64Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_shuffle_i64(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "bfly",
            31,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ShflSyncDownI64Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_shuffle_i64(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "down",
            31,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ShflSyncUpI64Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_shuffle_i64(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "up",
            0,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for VoteSyncAllOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_vote(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_vote_all_sync",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for VoteSyncAnyOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_vote(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_vote_any_sync",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for VoteSyncBallotOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_vote(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_vote_ballot_sync",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MatchAnySyncI32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let i32_ty = pliron::builtin::types::IntegerType::get(
            ctx,
            32,
            pliron::builtin::types::Signedness::Signless,
        );
        super::intrinsics::warp::convert_match_any(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_match_any_sync_i32",
            i32_ty.into(),
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MatchAnySyncI64Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let i64_ty = pliron::builtin::types::IntegerType::get(
            ctx,
            64,
            pliron::builtin::types::Signedness::Signless,
        );
        super::intrinsics::warp::convert_match_any(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_match_any_sync_i64",
            i64_ty.into(),
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MatchAllSyncI32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let i32_ty = pliron::builtin::types::IntegerType::get(
            ctx,
            32,
            pliron::builtin::types::Signedness::Signless,
        );
        super::intrinsics::warp::convert_match_all(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_match_all_sync_i32p",
            i32_ty.into(),
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ActiveMaskOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_active_mask(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for BarWarpSyncOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_bar_warp_sync(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MatchAllSyncI64Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let i64_ty = pliron::builtin::types::IntegerType::get(
            ctx,
            64,
            pliron::builtin::types::Signedness::Signless,
        );
        super::intrinsics::warp::convert_match_all(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_match_all_sync_i64p",
            i64_ty.into(),
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ElectSyncOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_elect_sync(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReduxSyncAddOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_redux(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_redux_sync_add",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReduxSyncUminOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_redux(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_redux_sync_umin",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReduxSyncMinOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_redux(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_redux_sync_min",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReduxSyncUmaxOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_redux(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_redux_sync_umax",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReduxSyncMaxOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_redux(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_redux_sync_max",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReduxSyncAndOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_redux(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_redux_sync_and",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReduxSyncOrOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_redux(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_redux_sync_or",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ReduxSyncXorOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::warp::convert_redux(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "llvm_nvvm_redux_sync_xor",
        )
    }
}

// ---- NVVM Mbarrier ops -----------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for MbarrierInitSharedOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::mbarrier::convert_init(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MbarrierArriveSharedOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::mbarrier::convert_arrive(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MbarrierArriveExpectTxSharedOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::mbarrier::convert_arrive_expect_tx(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MbarrierTestWaitSharedOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::mbarrier::convert_test_wait(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MbarrierTryWaitSharedOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::mbarrier::convert_try_wait(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MbarrierTryWaitParitySharedOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::mbarrier::convert_try_wait_parity(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MbarrierInvalSharedOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::mbarrier::convert_inval(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for FenceProxyAsyncSharedCtaOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::mbarrier::convert_fence_proxy_async(ctx, rewriter, operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MbarrierArriveClusterOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::mbarrier::convert_arrive_cluster(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for NanosleepOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::mbarrier::convert_nanosleep(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

// ---- NVVM WGMMA ops --------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for WgmmaFenceSyncAlignedOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::wgmma::convert_fence(ctx, rewriter, self.get_operation(), operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for WgmmaCommitGroupSyncAlignedOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::wgmma::convert_commit_group(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for WgmmaWaitGroupSyncAlignedOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::wgmma::convert_wait_group(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for WgmmaMakeSmemDescOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::wgmma::convert_make_smem_desc(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for WgmmaMmaM64N64K16F32Bf16Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::wgmma::convert_mma(ctx, rewriter, self.get_operation(), operands_info)
    }
}

// ---- NVVM Convert ops ------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for CvtF16x2F32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::convert::convert_cvt_f16x2_f32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CvtRzF16x2F32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::convert::convert_cvt_rz_f16x2_f32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CvtRnReluF16x2F32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::convert::convert_cvt_rn_relu_f16x2_f32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CvtRnReluBf16x2F32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::convert::convert_cvt_rn_relu_bf16x2_f32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CvtRzBf16x2F32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::convert::convert_cvt_rz_bf16x2_f32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

// ---- NVVM Tcgen05 ops ------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05AllocOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_alloc(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05DeallocOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_dealloc(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05RelinquishAllocPermitOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_relinquish_alloc_permit(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05FenceBeforeThreadSyncOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_fence_before_thread_sync(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05FenceAfterThreadSyncOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_fence_after_thread_sync(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05CommitOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_commit(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05CommitSharedClusterOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_commit_shared_cluster(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05MmaWsF16Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_mma_ws(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "f16",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05MmaWsBf16Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_mma_ws(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "bf16",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05MmaWsTf32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_mma_ws(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "tf32",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05MmaF16Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_mma_f16(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05CpSmemToTmemOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_cp_smem_to_tmem(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05Ld16x256bX8PureOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_ld_16x256b_x8_pure(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05Ld16x256bPureOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_ld_16x256b_pure(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CvtF32x2Bf16x2Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_cvt_f32x2_bf16x2(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

// ---- NVVM bf16x2 arithmetic ops -------------------------------------------
#[op_interface_impl]
impl MirToLlvmConversion for FmaBf16x2Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::bf16x2::convert_fma_bf16x2(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for FmaReluBf16x2Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::bf16x2::convert_fma_relu_bf16x2(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for AddBf16x2Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::bf16x2::convert_add_bf16x2(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for SubBf16x2Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::bf16x2::convert_sub_bf16x2(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MulBf16x2Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::bf16x2::convert_mul_bf16x2(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MinBf16x2Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::bf16x2::convert_min_bf16x2(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for MaxBf16x2Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::bf16x2::convert_max_bf16x2(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for NegBf16x2Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::bf16x2::convert_neg_bf16x2(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for AbsBf16x2Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::bf16x2::convert_abs_bf16x2(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

// ---- NVVM integer dot product ops -----------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for Dp4aS32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::dotprod::convert_dp4a_s32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Dp4aU32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::dotprod::convert_dp4a_u32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Dp2aS32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::dotprod::convert_dp2a_s32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Dp2aU32Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::dotprod::convert_dp2a_u32(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05LoadWaitOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_load_wait(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05StoreWaitOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_store_wait(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05AllocCg2Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_alloc_cg2(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05DeallocCg2Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_dealloc_cg2(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05RelinquishAllocPermitCg2Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_relinquish_alloc_permit_cg2(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05MmaF16Cg2Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_mma_f16_cg2(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05CommitCg2Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_commit_cg2(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05CommitSharedClusterCg2Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_commit_shared_cluster_cg2(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05CommitMulticastCg2Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_commit_multicast_cg2(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05CpSmemToTmemCg2Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tcgen05::convert_cp_smem_to_tmem_cg2(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

// ---- NVVM TMA ops ----------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for CpAsyncBulkTensorG2sTile1dOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tma::convert_g2s(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            1,
            false,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CpAsyncBulkTensorG2sTile2dOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tma::convert_g2s(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            2,
            false,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CpAsyncBulkTensorG2sTile2dMulticastOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tma::convert_g2s(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            2,
            true,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CpAsyncBulkTensorG2sTile2dMulticastCg2Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tma::convert_g2s_multicast_cg2(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CpAsyncBulkTensorG2sTile3dOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tma::convert_g2s(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            3,
            false,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CpAsyncBulkTensorG2sTile4dOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tma::convert_g2s(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            4,
            false,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CpAsyncBulkTensorG2sTile5dOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tma::convert_g2s(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            5,
            false,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CpAsyncBulkTensorS2gTile1dOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tma::convert_s2g(ctx, rewriter, self.get_operation(), operands_info, 1)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CpAsyncBulkTensorS2gTile2dOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tma::convert_s2g(ctx, rewriter, self.get_operation(), operands_info, 2)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CpAsyncBulkTensorS2gTile3dOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tma::convert_s2g(ctx, rewriter, self.get_operation(), operands_info, 3)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CpAsyncBulkTensorS2gTile4dOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tma::convert_s2g(ctx, rewriter, self.get_operation(), operands_info, 4)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CpAsyncBulkTensorS2gTile5dOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tma::convert_s2g(ctx, rewriter, self.get_operation(), operands_info, 5)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CpAsyncBulkCommitGroupOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tma::convert_commit_group(ctx, rewriter, operands_info)
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CpAsyncBulkWaitGroupOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tma::convert_wait_group(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            false,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CpAsyncBulkWaitGroupReadOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::tma::convert_wait_group(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            true,
        )
    }
}

// ---- NVVM cp.async ops -----------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for CpAsyncCa4Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::cp_async::convert_cp_async_ca_4(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CpAsyncCa8Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::cp_async::convert_cp_async_ca_8(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

// ---- NVVM cp.async zero-fill ops -------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for CpAsyncCaZfill4Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::cp_async::convert_cp_async_ca_zfill_4(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CpAsyncCaZfill8Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::cp_async::convert_cp_async_ca_zfill_8(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for CpAsyncCaZfill16Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::cp_async::convert_cp_async_ca_zfill_16(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

// ---- NVVM Stmatrix ops -----------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for StmatrixM8n8X4Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::stmatrix::convert_m8n8_x4(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for StmatrixM8n8X4TransOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::stmatrix::convert_m8n8_x4_trans(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for StmatrixM8n8X2Op {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::stmatrix::convert_m8n8_x2(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for StmatrixM8n8X2TransOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::stmatrix::convert_m8n8_x2_trans(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

// ---- NVVM Atomic ops -------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for NvvmAtomicLoadOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::atomic::convert_atomic_load(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for NvvmAtomicStoreOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::atomic::convert_atomic_store(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for NvvmAtomicRmwOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::atomic::convert_atomic_rmw(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for NvvmAtomicCmpxchgOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::atomic::convert_atomic_cmpxchg(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

// ---- NVVM CLC ops ----------------------------------------------------------

#[op_interface_impl]
impl MirToLlvmConversion for ClcTryCancelOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::clc::convert_try_cancel(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            false,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ClcTryCancelMulticastOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::clc::convert_try_cancel(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            true,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ClcQueryIsCanceledOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::clc::convert_query_is_canceled(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ClcQueryGetFirstCtaidXOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::clc::convert_query_get_first_ctaid(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "x",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ClcQueryGetFirstCtaidYOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::clc::convert_query_get_first_ctaid(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "y",
        )
    }
}

#[op_interface_impl]
impl MirToLlvmConversion for ClcQueryGetFirstCtaidZOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        super::intrinsics::clc::convert_query_get_first_ctaid(
            ctx,
            rewriter,
            self.get_operation(),
            operands_info,
            "z",
        )
    }
}
