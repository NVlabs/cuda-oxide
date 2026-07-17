/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::types::{MirPtrType, address_space};
use dialect_nvvm::ops::{
    ActiveMaskOp, AtomicOrdering, AtomicRmwKind, AtomicScope, BarWarpSyncOp, Barrier0Op,
    ClusterBarrierModeAttr, ClusterBarrierOp, CpAsyncCa4Op, CpAsyncCaZfill4Op,
    CpAsyncMbarrierArriveNoIncOp, CpAsyncMbarrierArriveNoIncSharedOp, CpAsyncMbarrierArriveOp,
    CpAsyncMbarrierArriveSharedOp, CpAsyncWaitGroupOp, Dp2aS32Op, Dp2aU32Op, Dp4aS32Op, Dp4aU32Op,
    ElectSyncOp, FmaBf16x2Op, LdmatrixElementAttr, LdmatrixLayoutAttr, LdmatrixMultiplicityAttr,
    LdmatrixOp, LdmatrixShapeAttr, LdmatrixStateSpaceAttr, LdmatrixX1Op, LdmatrixX1TransOp,
    LdmatrixX2Op, LdmatrixX2TransOp, LdmatrixX4Op, LdmatrixX4TransOp, MatchAllSyncI32Op,
    MatchAllSyncI64Op, MatchAnySyncI32Op, MatchAnySyncI64Op, MbarrierArriveSharedOp,
    MbarrierInitSharedOp, MbarrierInvalSharedOp, MbarrierTestWaitSharedOp, MmaM8N8K4F64Op,
    MmaM16N8K8F32Tf32Op, MmaM16N8K16F32Bf16Op, MmaM16N8K16F32F16Op, MmaM16N8K32S32S8Op,
    MovmatrixTransB16Op, NvvmAtomAddBf16x2Op, NvvmAtomAddF16x2Op, NvvmAtomicCmpxchgOp,
    NvvmAtomicLoadOp, NvvmAtomicRmwOp, NvvmAtomicStoreOp, PackedAtomicAddOp,
    PackedAtomicAtomicityAttr, PackedAtomicFormatAttr, PackedAtomicOrderingAttr,
    PackedAtomicRoundingAttr, PackedAtomicScopeAttr, PackedAtomicStateSpaceAttr,
    PackedAtomicSubnormalAttr, ReadPtxSregClusterIdxOp, ReadPtxSregDynamicSmemSizeOp,
    ReadPtxSregGridIdOp, ReadPtxSregLaneIdOp, ReadPtxSregLanemaskEqOp, ReadPtxSregLanemaskGeOp,
    ReadPtxSregLanemaskGtOp, ReadPtxSregLanemaskLeOp, ReadPtxSregLanemaskLtOp,
    ReadPtxSregNclusterIdOp, ReadPtxSregNsmIdOp, ReadPtxSregNwarpIdOp, ReadPtxSregSmIdOp,
    ReadPtxSregTidXOp, ReadPtxSregTotalSmemSizeOp, ReadPtxSregWarpIdOp, ReduxSyncAddOp,
    ReduxSyncAndOp, ReduxSyncMaxOp, ReduxSyncMinOp, ReduxSyncOrOp, ReduxSyncUmaxOp,
    ReduxSyncUminOp, ReduxSyncXorOp, RegisterMmaAccumulatorAttr, RegisterMmaElementAttr,
    RegisterMmaLayoutAttr, RegisterMmaOp, RegisterMmaOperationAttr, RegisterMmaOverflowAttr,
    RegisterMmaShapeAttr, ScalarArithmeticFormatAttr, ScalarArithmeticOp,
    ScalarArithmeticOperationAttr, ScalarArithmeticRoundingAttr, ScalarArithmeticSaturationAttr,
    ScalarArithmeticSubnormalAttr, ScalarConversionOp, ScalarConversionRoundingAttr,
    ScalarConversionSaturationAttr, ShflSyncBflyI64Op, ShflSyncDownI64Op, ShflSyncIdxI64Op,
    ShflSyncUpI64Op, SparseMmaAccumulatorAttr, SparseMmaElementAttr, SparseMmaLayoutAttr,
    SparseMmaMetadataAttr, SparseMmaOp, SparseMmaOverflowAttr, SparseMmaSelectorAttr,
    SparseMmaShapeAttr, StmatrixM8n8X4Op, Tcgen05AllocOp, Tcgen05CommitMulticastCg2Op,
    Tcgen05Ld16x32bx2X1RawOp, Tcgen05Ld16x256bPureOp, Tcgen05MmaF16Op, ThreadfenceBlockOp,
    ThreadfenceOp, ThreadfenceSystemOp, VoteSyncAllOp, VoteSyncAnyOp, VoteSyncBallotOp,
    VoteSyncUniOp, VprintfOp, WgmmaMakeSmemDescOp, WgmmaMmaM64N64K16F32Bf16Op,
};

#[test]
fn handwritten_ops_match_reviewed_allowlist() {
    use std::{fs, path::Path};

    let ops_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/ops");
    let mut found = Vec::new();

    for entry in fs::read_dir(&ops_dir).expect("read top-level NVVM ops directory") {
        let path = entry.expect("read NVVM ops entry").path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
            continue;
        }

        let source = fs::read_to_string(&path).expect("read handwritten NVVM ops source");
        let file = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("NVVM ops file name");

        for (marker, _) in source.match_indices("#[pliron_op(") {
            let declaration = source[marker..]
                .lines()
                .find_map(|line| line.trim().strip_prefix("pub struct "))
                .expect("pliron_op must be followed by a public struct");
            let name = declaration
                .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
                .next()
                .expect("NVVM op struct name");
            found.push((file.to_owned(), name.to_owned()));
        }
    }

    let mut expected = [
        ("asm.rs", "InlinePtxOp"),
        ("atomic.rs", "NvvmAtomicLoadOp"),
        ("atomic.rs", "NvvmAtomicStoreOp"),
        ("atomic.rs", "NvvmAtomicRmwOp"),
        ("atomic.rs", "NvvmAtomicCmpxchgOp"),
        ("cluster.rs", "ReadPtxSregClusterIdxOp"),
        ("cluster.rs", "ReadPtxSregNclusterIdOp"),
        ("debug.rs", "VprintfOp"),
        ("grid.rs", "GridSyncOp"),
        ("wgmma.rs", "WgmmaMakeSmemDescOp"),
        ("wgmma.rs", "WgmmaMmaM64N64K16F32Bf16Op"),
    ];
    expected.sort_unstable();
    found.sort_unstable();

    assert_eq!(
        found,
        expected.map(|(file, op)| (file.to_owned(), op.to_owned())),
        "top-level handwritten NVVM ops changed; generate leaf ops or review this allowlist"
    );
}

#[test]
fn cluster_grid_compatibility_ops_keep_names_and_i32_shape() {
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::common_traits::Verify;
    use pliron::context::Context;
    use pliron::op::Op;
    use pliron::operation::Operation;

    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);
    assert_eq!(
        ReadPtxSregClusterIdxOp::get_opid_static().to_string(),
        "nvvm.read_ptx_sreg_cluster_idx"
    );
    assert_eq!(
        ReadPtxSregNclusterIdOp::get_opid_static().to_string(),
        "nvvm.read_ptx_sreg_nclusterid"
    );

    let i32_type = IntegerType::get(&ctx, 32, Signedness::Signless);
    for op_info in [
        ReadPtxSregClusterIdxOp::get_concrete_op_info(),
        ReadPtxSregNclusterIdOp::get_concrete_op_info(),
    ] {
        let op = Operation::new(&mut ctx, op_info, vec![i32_type.into()], vec![], vec![], 0);
        assert!(op.deref(&ctx).verify(&ctx).is_ok());
    }
}

#[test]
fn generated_cluster_barrier_requires_one_closed_mode_and_no_values() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    for mode in [
        ClusterBarrierModeAttr::Arrive,
        ClusterBarrierModeAttr::ArriveAligned,
        ClusterBarrierModeAttr::ArriveRelaxed,
        ClusterBarrierModeAttr::ArriveRelaxedAligned,
        ClusterBarrierModeAttr::Wait,
        ClusterBarrierModeAttr::WaitAligned,
    ] {
        let op = ClusterBarrierOp::build(&mut ctx, mode);
        assert!(verify_op(&ClusterBarrierOp::new(op), &ctx).is_ok());
    }

    let missing_mode = Operation::new(
        &mut ctx,
        ClusterBarrierOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    assert!(verify_op(&ClusterBarrierOp::new(missing_mode), &ctx).is_err());

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let wrong_shape = Operation::new(
        &mut ctx,
        ClusterBarrierOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    ClusterBarrierOp::new(wrong_shape)
        .set_attr_nvvm_cluster_barrier_mode(&ctx, ClusterBarrierModeAttr::Wait);
    assert!(verify_op(&ClusterBarrierOp::new(wrong_shape), &ctx).is_err());
}

#[test]
fn generated_scalar_conversion_accepts_only_reviewed_f32_to_i32_variants() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let f32_ty = FP32Type::get(&ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let block = BasicBlock::new(&mut ctx, None, vec![f32_ty.into(), i32_ty.into()]);
    let f32_value = block.deref(&ctx).get_argument(0);
    let i32_value = block.deref(&ctx).get_argument(1);

    for (rounding, saturation) in [
        (
            ScalarConversionRoundingAttr::NearestAway,
            ScalarConversionSaturationAttr::None,
        ),
        (
            ScalarConversionRoundingAttr::NearestAway,
            ScalarConversionSaturationAttr::Satfinite,
        ),
        (
            ScalarConversionRoundingAttr::NearestEven,
            ScalarConversionSaturationAttr::None,
        ),
        (
            ScalarConversionRoundingAttr::NearestEven,
            ScalarConversionSaturationAttr::Relu,
        ),
        (
            ScalarConversionRoundingAttr::NearestEven,
            ScalarConversionSaturationAttr::Satfinite,
        ),
        (
            ScalarConversionRoundingAttr::NearestEven,
            ScalarConversionSaturationAttr::ReluSatfinite,
        ),
        (
            ScalarConversionRoundingAttr::TowardZero,
            ScalarConversionSaturationAttr::None,
        ),
        (
            ScalarConversionRoundingAttr::TowardZero,
            ScalarConversionSaturationAttr::Relu,
        ),
        (
            ScalarConversionRoundingAttr::TowardZero,
            ScalarConversionSaturationAttr::Satfinite,
        ),
        (
            ScalarConversionRoundingAttr::TowardZero,
            ScalarConversionSaturationAttr::ReluSatfinite,
        ),
    ] {
        let op = ScalarConversionOp::build(&mut ctx, f32_value, rounding, saturation);
        assert!(verify_op(&ScalarConversionOp::new(op), &ctx).is_ok());
    }

    let invalid_variant = ScalarConversionOp::build(
        &mut ctx,
        f32_value,
        ScalarConversionRoundingAttr::NearestAway,
        ScalarConversionSaturationAttr::Relu,
    );
    assert!(verify_op(&ScalarConversionOp::new(invalid_variant), &ctx).is_err());

    let wrong_operand = Operation::new(
        &mut ctx,
        ScalarConversionOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![i32_value],
        vec![],
        0,
    );
    let wrong_operand = ScalarConversionOp::new(wrong_operand);
    wrong_operand
        .set_attr_nvvm_scalar_conversion_rounding(&ctx, ScalarConversionRoundingAttr::NearestEven);
    wrong_operand
        .set_attr_nvvm_scalar_conversion_saturation(&ctx, ScalarConversionSaturationAttr::None);
    assert!(verify_op(&wrong_operand, &ctx).is_err());

    let wrong_result = Operation::new(
        &mut ctx,
        ScalarConversionOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![f32_value],
        vec![],
        0,
    );
    let wrong_result = ScalarConversionOp::new(wrong_result);
    wrong_result
        .set_attr_nvvm_scalar_conversion_rounding(&ctx, ScalarConversionRoundingAttr::TowardZero);
    wrong_result
        .set_attr_nvvm_scalar_conversion_saturation(&ctx, ScalarConversionSaturationAttr::None);
    assert!(verify_op(&wrong_result, &ctx).is_err());
}

#[test]
fn generated_scalar_arithmetic_accepts_only_admitted_shapes_and_types() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let f32_ty = FP32Type::get(&ctx);
    let f64_ty = FP64Type::get(&ctx);
    let block = BasicBlock::new(&mut ctx, None, vec![f32_ty.into(), f64_ty.into()]);
    let f32_value = block.deref(&ctx).get_argument(0);
    let f64_value = block.deref(&ctx).get_argument(1);

    let valid_f32 = ScalarArithmeticOp::build(
        &mut ctx,
        vec![f32_value, f32_value],
        ScalarArithmeticFormatAttr::F32,
        ScalarArithmeticOperationAttr::Mul,
        ScalarArithmeticRoundingAttr::Rn,
        ScalarArithmeticSubnormalAttr::Preserve,
        ScalarArithmeticSaturationAttr::None,
    );
    assert!(verify_op(&ScalarArithmeticOp::new(valid_f32), &ctx).is_ok());

    let valid_f64 = ScalarArithmeticOp::build(
        &mut ctx,
        vec![f64_value, f64_value, f64_value],
        ScalarArithmeticFormatAttr::F64,
        ScalarArithmeticOperationAttr::Fma,
        ScalarArithmeticRoundingAttr::Rz,
        ScalarArithmeticSubnormalAttr::Preserve,
        ScalarArithmeticSaturationAttr::None,
    );
    assert!(verify_op(&ScalarArithmeticOp::new(valid_f64), &ctx).is_ok());

    let valid_add = ScalarArithmeticOp::build(
        &mut ctx,
        vec![f32_value, f32_value],
        ScalarArithmeticFormatAttr::F32,
        ScalarArithmeticOperationAttr::Add,
        ScalarArithmeticRoundingAttr::Rp,
        ScalarArithmeticSubnormalAttr::Ftz,
        ScalarArithmeticSaturationAttr::Sat,
    );
    assert!(verify_op(&ScalarArithmeticOp::new(valid_add), &ctx).is_ok());

    let f64_ftz = ScalarArithmeticOp::build(
        &mut ctx,
        vec![f64_value, f64_value],
        ScalarArithmeticFormatAttr::F64,
        ScalarArithmeticOperationAttr::Mul,
        ScalarArithmeticRoundingAttr::Rn,
        ScalarArithmeticSubnormalAttr::Ftz,
        ScalarArithmeticSaturationAttr::None,
    );
    assert!(verify_op(&ScalarArithmeticOp::new(f64_ftz), &ctx).is_err());

    let wrong_arity = ScalarArithmeticOp::build(
        &mut ctx,
        vec![f32_value, f32_value],
        ScalarArithmeticFormatAttr::F32,
        ScalarArithmeticOperationAttr::Fma,
        ScalarArithmeticRoundingAttr::Rn,
        ScalarArithmeticSubnormalAttr::Preserve,
        ScalarArithmeticSaturationAttr::None,
    );
    assert!(verify_op(&ScalarArithmeticOp::new(wrong_arity), &ctx).is_err());

    let wrong_type = ScalarArithmeticOp::build(
        &mut ctx,
        vec![f32_value, f64_value],
        ScalarArithmeticFormatAttr::F32,
        ScalarArithmeticOperationAttr::Mul,
        ScalarArithmeticRoundingAttr::Rn,
        ScalarArithmeticSubnormalAttr::Preserve,
        ScalarArithmeticSaturationAttr::None,
    );
    assert!(verify_op(&ScalarArithmeticOp::new(wrong_type), &ctx).is_err());

    let missing_attrs = Operation::new(
        &mut ctx,
        ScalarArithmeticOp::get_concrete_op_info(),
        vec![f32_ty.into()],
        vec![f32_value, f32_value],
        vec![],
        0,
    );
    assert!(verify_op(&ScalarArithmeticOp::new(missing_attrs), &ctx).is_err());
}

#[test]
fn test_generated_cp_async_accepts_pointer_shapes_and_both_constant_kinds() {
    use dialect_mir::ops::MirConstantOp;
    use pliron::builtin::{attributes::IntegerAttr, ops::ConstantOp};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let dst_ty = MirPtrType::get_generic(&mut ctx, u8_ty.into(), false);
    let src_ty = MirPtrType::get(&mut ctx, u8_ty.into(), true, address_space::GLOBAL);
    let wrong_dst_ty = MirPtrType::get(&mut ctx, u8_ty.into(), true, address_space::GLOBAL);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            dst_ty.into(),
            src_ty.into(),
            wrong_dst_ty.into(),
            u32_ty.into(),
        ],
    );
    let dst = block.deref(&ctx).get_argument(0);
    let src = block.deref(&ctx).get_argument(1);
    let wrong_dst = block.deref(&ctx).get_argument(2);
    let dynamic = block.deref(&ctx).get_argument(3);

    let copy = CpAsyncCa4Op::build(&mut ctx, dst, src);
    assert!(verify_op(&CpAsyncCa4Op::new(copy), &ctx).is_ok());
    let zfill = CpAsyncCaZfill4Op::build(&mut ctx, dst, src, dynamic);
    assert!(verify_op(&CpAsyncCaZfill4Op::new(zfill), &ctx).is_ok());
    let wrong_space = CpAsyncCa4Op::build(&mut ctx, wrong_dst, src);
    assert!(verify_op(&CpAsyncCa4Op::new(wrong_space), &ctx).is_err());

    let value = IntegerAttr::new(u32_ty, APInt::from_u32(0, NonZeroUsize::new(32).unwrap()));
    let builtin = ConstantOp::new(&mut ctx, value.clone().into());
    let builtin_value = builtin.get_operation().deref(&ctx).get_result(0);
    let builtin_wait = CpAsyncWaitGroupOp::build(&mut ctx, builtin_value);
    assert!(verify_op(&CpAsyncWaitGroupOp::new(builtin_wait), &ctx).is_ok());

    let mir_constant = Operation::new(
        &mut ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![],
        vec![],
        0,
    );
    MirConstantOp::new(mir_constant).set_attr_value(&ctx, value);
    let mir_value = mir_constant.deref(&ctx).get_result(0);
    let mir_wait = CpAsyncWaitGroupOp::build(&mut ctx, mir_value);
    assert!(verify_op(&CpAsyncWaitGroupOp::new(mir_wait), &ctx).is_ok());

    let dynamic_wait = CpAsyncWaitGroupOp::build(&mut ctx, dynamic);
    assert!(verify_op(&CpAsyncWaitGroupOp::new(dynamic_wait), &ctx).is_err());
}

#[test]
fn generated_tcgen05_verifies_carriers_and_half_split_constants() {
    use dialect_mir::ops::MirConstantOp;
    use pliron::builtin::{attributes::IntegerAttr, ops::ConstantOp};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let i16_ty = IntegerType::get(&ctx, 16, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let f32_ty = FP32Type::get(&ctx);
    let pointer_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            pointer_ty.into(),
            i1_ty.into(),
            i16_ty.into(),
            i32_ty.into(),
            i64_ty.into(),
            f32_ty.into(),
        ],
    );
    let pointer = block.deref(&ctx).get_argument(0);
    let predicate = block.deref(&ctx).get_argument(1);
    let mask = block.deref(&ctx).get_argument(2);
    let address = block.deref(&ctx).get_argument(3);
    let dynamic_offset = block.deref(&ctx).get_argument(4);

    let offset_attr = IntegerAttr::new(i64_ty, APInt::from_i64(16, NonZeroUsize::new(64).unwrap()));
    let builtin_offset = ConstantOp::new(&mut ctx, offset_attr.clone().into())
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let mir_constant = Operation::new(
        &mut ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![],
        vec![],
        0,
    );
    MirConstantOp::new(mir_constant).set_attr_value(&ctx, offset_attr);
    let mir_offset = mir_constant.deref(&ctx).get_result(0);

    for offset in [builtin_offset, mir_offset] {
        let load = Operation::new(
            &mut ctx,
            Tcgen05Ld16x32bx2X1RawOp::get_concrete_op_info(),
            vec![i32_ty.into()],
            vec![address, offset],
            vec![],
            0,
        );
        assert!(verify_op(&Tcgen05Ld16x32bx2X1RawOp::new(load), &ctx).is_ok());
    }

    let dynamic = Operation::new(
        &mut ctx,
        Tcgen05Ld16x32bx2X1RawOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![address, dynamic_offset],
        vec![],
        0,
    );
    assert!(verify_op(&Tcgen05Ld16x32bx2X1RawOp::new(dynamic), &ctx).is_err());

    let wrong_offset_type = Operation::new(
        &mut ctx,
        Tcgen05Ld16x32bx2X1RawOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![address, address],
        vec![],
        0,
    );
    assert!(verify_op(&Tcgen05Ld16x32bx2X1RawOp::new(wrong_offset_type), &ctx).is_err());

    let wrong_result_type = Operation::new(
        &mut ctx,
        Tcgen05Ld16x32bx2X1RawOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![address, builtin_offset],
        vec![],
        0,
    );
    assert!(verify_op(&Tcgen05Ld16x32bx2X1RawOp::new(wrong_result_type), &ctx).is_err());

    let alloc = Operation::new(
        &mut ctx,
        Tcgen05AllocOp::get_concrete_op_info(),
        vec![],
        vec![pointer, address],
        vec![],
        0,
    );
    assert!(verify_op(&Tcgen05AllocOp::new(alloc), &ctx).is_ok());

    let multicast = Operation::new(
        &mut ctx,
        Tcgen05CommitMulticastCg2Op::get_concrete_op_info(),
        vec![],
        vec![pointer, mask],
        vec![],
        0,
    );
    assert!(verify_op(&Tcgen05CommitMulticastCg2Op::new(multicast), &ctx).is_ok());

    let mma = Operation::new(
        &mut ctx,
        Tcgen05MmaF16Op::get_concrete_op_info(),
        vec![],
        vec![address, dynamic_offset, dynamic_offset, address, predicate],
        vec![],
        0,
    );
    assert!(verify_op(&Tcgen05MmaF16Op::new(mma), &ctx).is_ok());

    let pure_load = Operation::new(
        &mut ctx,
        Tcgen05Ld16x256bPureOp::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        vec![address],
        vec![],
        0,
    );
    assert!(verify_op(&Tcgen05Ld16x256bPureOp::new(pure_load), &ctx).is_ok());
}

#[test]
fn generated_cp_async_mbarrier_requires_mutable_generic_or_shared_u64() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let generic_u64 = MirPtrType::get_generic(&mut ctx, u64_ty.into(), true);
    let shared_u64 = MirPtrType::get_shared(&mut ctx, u64_ty.into(), true);
    let global_u64 = MirPtrType::get_global(&mut ctx, u64_ty.into(), true);
    let immutable_u64 = MirPtrType::get_generic(&mut ctx, u64_ty.into(), false);
    let generic_u32 = MirPtrType::get_generic(&mut ctx, u32_ty.into(), true);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            generic_u64.into(),
            shared_u64.into(),
            global_u64.into(),
            immutable_u64.into(),
            generic_u32.into(),
            u64_ty.into(),
        ],
    );
    let generic = block.deref(&ctx).get_argument(0);
    let shared = block.deref(&ctx).get_argument(1);
    let global = block.deref(&ctx).get_argument(2);
    let immutable = block.deref(&ctx).get_argument(3);
    let wrong_pointee = block.deref(&ctx).get_argument(4);
    let scalar = block.deref(&ctx).get_argument(5);

    macro_rules! check_bridge {
        ($op:ty) => {{
            for barrier in [generic, shared] {
                let valid = <$op>::build(&mut ctx, barrier);
                assert!(verify_op(&<$op>::new(valid), &ctx).is_ok());
            }
            for barrier in [global, immutable, wrong_pointee, scalar] {
                let invalid = <$op>::build(&mut ctx, barrier);
                assert!(verify_op(&<$op>::new(invalid), &ctx).is_err());
            }
            let wrong_shape = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![u64_ty.into()],
                vec![generic],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(wrong_shape), &ctx).is_err());
        }};
    }

    check_bridge!(CpAsyncMbarrierArriveOp);
    check_bridge!(CpAsyncMbarrierArriveSharedOp);
    check_bridge!(CpAsyncMbarrierArriveNoIncOp);
    check_bridge!(CpAsyncMbarrierArriveNoIncSharedOp);
}

#[test]
fn generated_mbarrier_builders_and_verifiers_are_closed_over_their_shapes() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let u1_ty = IntegerType::get(&ctx, 1, Signedness::Unsigned);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let generic_ptr = MirPtrType::get_generic(&mut ctx, u64_ty.into(), false);
    let shared_ptr = MirPtrType::get_shared(&mut ctx, u64_ty.into(), false);
    let global_ptr = MirPtrType::get_global(&mut ctx, u64_ty.into(), false);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            generic_ptr.into(),
            shared_ptr.into(),
            global_ptr.into(),
            i32_ty.into(),
            u32_ty.into(),
            u64_ty.into(),
        ],
    );
    let generic = block.deref(&ctx).get_argument(0);
    let shared = block.deref(&ctx).get_argument(1);
    let global = block.deref(&ctx).get_argument(2);
    let signless_i32 = block.deref(&ctx).get_argument(3);
    let count = block.deref(&ctx).get_argument(4);
    let token = block.deref(&ctx).get_argument(5);

    for barrier in [generic, shared] {
        let init = MbarrierInitSharedOp::build(&mut ctx, barrier, count);
        assert!(MbarrierInitSharedOp::new(init).verify(&ctx).is_ok());
        let arrive = MbarrierArriveSharedOp::build(&mut ctx, barrier);
        assert!(MbarrierArriveSharedOp::new(arrive).verify(&ctx).is_ok());
        let test_wait = MbarrierTestWaitSharedOp::build(&mut ctx, barrier, token);
        assert!(
            MbarrierTestWaitSharedOp::new(test_wait)
                .verify(&ctx)
                .is_ok()
        );
        let inval = MbarrierInvalSharedOp::build(&mut ctx, barrier);
        assert!(MbarrierInvalSharedOp::new(inval).verify(&ctx).is_ok());
    }

    for barrier in [token, global] {
        let inval = MbarrierInvalSharedOp::build(&mut ctx, barrier);
        assert!(MbarrierInvalSharedOp::new(inval).verify(&ctx).is_err());
    }

    let bad_count = MbarrierInitSharedOp::build(&mut ctx, shared, signless_i32);
    assert!(MbarrierInitSharedOp::new(bad_count).verify(&ctx).is_err());
    let bad_arrive_result = Operation::new(
        &mut ctx,
        MbarrierArriveSharedOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![shared],
        vec![],
        0,
    );
    assert!(
        MbarrierArriveSharedOp::new(bad_arrive_result)
            .verify(&ctx)
            .is_err()
    );
    let bad_token = MbarrierTestWaitSharedOp::build(&mut ctx, shared, count);
    assert!(
        MbarrierTestWaitSharedOp::new(bad_token)
            .verify(&ctx)
            .is_err()
    );
    let bad_predicate = Operation::new(
        &mut ctx,
        MbarrierTestWaitSharedOp::get_concrete_op_info(),
        vec![u1_ty.into()],
        vec![shared, token],
        vec![],
        0,
    );
    assert!(
        MbarrierTestWaitSharedOp::new(bad_predicate)
            .verify(&ctx)
            .is_err()
    );

    let missing_operands = Operation::new(
        &mut ctx,
        MbarrierInitSharedOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    assert!(
        MbarrierInitSharedOp::new(missing_operands)
            .verify(&ctx)
            .is_err()
    );
}
use pliron::{
    basic_block::BasicBlock,
    builtin::types::{FP32Type, FP64Type, IntegerType, Signedness},
    common_traits::Verify,
    context::Context,
    op::{Op, verify_op},
    operation::Operation,
    r#type::Typed,
};

#[test]
fn test_mma_m8n8k4_f64_requires_four_f64_operands_and_two_f64_results() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let f64_ty = FP64Type::get(&ctx);
    let f32_ty = FP32Type::get(&ctx);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            f64_ty.into(),
            f64_ty.into(),
            f64_ty.into(),
            f64_ty.into(),
            f32_ty.into(),
        ],
    );
    let f64_operands = (0..4)
        .map(|index| block.deref(&ctx).get_argument(index))
        .collect::<Vec<_>>();
    let f32_value = block.deref(&ctx).get_argument(4);

    let valid = Operation::new(
        &mut ctx,
        MmaM8N8K4F64Op::get_concrete_op_info(),
        vec![f64_ty.into(), f64_ty.into()],
        f64_operands.clone(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM8N8K4F64Op::new(valid), &ctx).is_ok());

    let mut bad_operands = f64_operands.clone();
    bad_operands[2] = f32_value;
    let invalid_operand = Operation::new(
        &mut ctx,
        MmaM8N8K4F64Op::get_concrete_op_info(),
        vec![f64_ty.into(), f64_ty.into()],
        bad_operands,
        vec![],
        0,
    );
    assert!(verify_op(&MmaM8N8K4F64Op::new(invalid_operand), &ctx).is_err());

    let invalid_result = Operation::new(
        &mut ctx,
        MmaM8N8K4F64Op::get_concrete_op_info(),
        vec![f64_ty.into(), f32_ty.into()],
        f64_operands,
        vec![],
        0,
    );
    assert!(verify_op(&MmaM8N8K4F64Op::new(invalid_result), &ctx).is_err());
}

#[test]
fn test_movmatrix_requires_one_i32_operand_and_result() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let f32_ty = FP32Type::get(&ctx);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![i32_ty.into(), i64_ty.into(), f32_ty.into()],
    );
    let i32_value = block.deref(&ctx).get_argument(0);
    let i64_value = block.deref(&ctx).get_argument(1);
    let f32_value = block.deref(&ctx).get_argument(2);

    let valid = Operation::new(
        &mut ctx,
        MovmatrixTransB16Op::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![i32_value],
        vec![],
        0,
    );
    assert!(verify_op(&MovmatrixTransB16Op::new(valid), &ctx).is_ok());

    for (operand, result_type) in [
        (i64_value, i32_ty.into()),
        (f32_value, i32_ty.into()),
        (i32_value, i64_ty.into()),
        (i32_value, f32_ty.into()),
    ] {
        let invalid = Operation::new(
            &mut ctx,
            MovmatrixTransB16Op::get_concrete_op_info(),
            vec![result_type],
            vec![operand],
            vec![],
            0,
        );
        assert!(
            verify_op(&MovmatrixTransB16Op::new(invalid), &ctx).is_err(),
            "movmatrix must reject non-i32 carriers"
        );
    }
}

fn make_ldmatrix_x2(
    ctx: &mut Context,
    pointer: pliron::value::Value,
    result_types: Vec<pliron::r#type::TypeHandle>,
) -> LdmatrixOp {
    let operation = Operation::new(
        ctx,
        LdmatrixOp::get_concrete_op_info(),
        result_types,
        vec![pointer],
        vec![],
        0,
    );
    let ldmatrix = LdmatrixOp::new(operation);
    ldmatrix.set_attr_nvvm_ldmatrix_shape(ctx, LdmatrixShapeAttr::M8n8);
    ldmatrix.set_attr_nvvm_ldmatrix_multiplicity(ctx, LdmatrixMultiplicityAttr::X2);
    ldmatrix.set_attr_nvvm_ldmatrix_layout(ctx, LdmatrixLayoutAttr::Normal);
    ldmatrix.set_attr_nvvm_ldmatrix_element(ctx, LdmatrixElementAttr::B16);
    ldmatrix.set_attr_nvvm_ldmatrix_state_space(ctx, LdmatrixStateSpaceAttr::Shared);
    ldmatrix
}

#[test]
fn test_ldmatrix_accepts_only_generic_or_shared_u32_pointers() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let u16_ty = IntegerType::get(&ctx, 16, Signedness::Unsigned);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let pointer_types = [
        MirPtrType::get_generic(&mut ctx, u32_ty.into(), false),
        MirPtrType::get_shared(&mut ctx, u32_ty.into(), false),
        MirPtrType::get_global(&mut ctx, u32_ty.into(), false),
        MirPtrType::get_constant(&mut ctx, u32_ty.into(), false),
        MirPtrType::get(&mut ctx, u32_ty.into(), false, address_space::LOCAL),
        MirPtrType::get_generic(&mut ctx, u16_ty.into(), false),
    ];
    let block = BasicBlock::new(
        &mut ctx,
        None,
        pointer_types
            .iter()
            .map(|pointer| (*pointer).into())
            .collect(),
    );

    for index in 0..pointer_types.len() {
        let pointer = block.deref(&ctx).get_argument(index);
        let operation = make_ldmatrix_x2(&mut ctx, pointer, vec![u32_ty.into(), u32_ty.into()]);
        let verified = verify_op(&operation, &ctx);
        if index < 2 {
            assert!(verified.is_ok(), "pointer case {index} should be accepted");
        } else {
            assert!(verified.is_err(), "pointer case {index} should be rejected");
        }
    }
}

#[test]
fn generated_ldmatrix_verifier_rejects_zero_or_two_operands_without_panicking() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let pointer_ty = MirPtrType::get_shared(&mut ctx, u32_ty.into(), false);
    let block = BasicBlock::new(&mut ctx, None, vec![pointer_ty.into(), pointer_ty.into()]);
    let pointer0 = block.deref(&ctx).get_argument(0);
    let pointer1 = block.deref(&ctx).get_argument(1);

    let zero = Operation::new(
        &mut ctx,
        LdmatrixOp::get_concrete_op_info(),
        vec![u32_ty.into(); 4],
        vec![],
        vec![],
        0,
    );
    assert!(LdmatrixOp::new(zero).verify(&ctx).is_err());

    let two = Operation::new(
        &mut ctx,
        LdmatrixOp::get_concrete_op_info(),
        vec![u32_ty.into(); 4],
        vec![pointer0, pointer1],
        vec![],
        0,
    );
    assert!(LdmatrixOp::new(two).verify(&ctx).is_err());

    let valid = LdmatrixOp::build(
        &mut ctx,
        pointer0,
        LdmatrixShapeAttr::M8n8,
        LdmatrixMultiplicityAttr::X4,
        LdmatrixLayoutAttr::Normal,
        LdmatrixElementAttr::B16,
        LdmatrixStateSpaceAttr::Shared,
    );
    assert!(LdmatrixOp::new(valid).verify(&ctx).is_ok());
}

#[test]
fn blackwell_ldmatrix_verifier_accepts_only_reviewed_shapes() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);
    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let shared_u8 = MirPtrType::get_shared(&mut ctx, u8_ty.into(), false);
    let generic_u8 = MirPtrType::get_generic(&mut ctx, u8_ty.into(), false);
    let global_u8 = MirPtrType::get_global(&mut ctx, u8_ty.into(), false);
    let shared_u32 = MirPtrType::get_shared(&mut ctx, u32_ty.into(), false);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            shared_u8.into(),
            generic_u8.into(),
            global_u8.into(),
            shared_u32.into(),
        ],
    );
    let shared_pointer = block.deref(&ctx).get_argument(0);

    for pointer_index in [0, 1] {
        let pointer = block.deref(&ctx).get_argument(pointer_index);
        for (shape, multiplicity, layout, element, result_count) in [
            (
                LdmatrixShapeAttr::M16n16,
                LdmatrixMultiplicityAttr::X1,
                LdmatrixLayoutAttr::Transposed,
                LdmatrixElementAttr::B8,
                2,
            ),
            (
                LdmatrixShapeAttr::M16n16,
                LdmatrixMultiplicityAttr::X1,
                LdmatrixLayoutAttr::Transposed,
                LdmatrixElementAttr::B8x16B4x16P64,
                2,
            ),
            (
                LdmatrixShapeAttr::M16n16,
                LdmatrixMultiplicityAttr::X1,
                LdmatrixLayoutAttr::Transposed,
                LdmatrixElementAttr::B8x16B6x16P32,
                2,
            ),
            (
                LdmatrixShapeAttr::M16n16,
                LdmatrixMultiplicityAttr::X2,
                LdmatrixLayoutAttr::Transposed,
                LdmatrixElementAttr::B8,
                4,
            ),
            (
                LdmatrixShapeAttr::M16n16,
                LdmatrixMultiplicityAttr::X2,
                LdmatrixLayoutAttr::Transposed,
                LdmatrixElementAttr::B8x16B4x16P64,
                4,
            ),
            (
                LdmatrixShapeAttr::M16n16,
                LdmatrixMultiplicityAttr::X2,
                LdmatrixLayoutAttr::Transposed,
                LdmatrixElementAttr::B8x16B6x16P32,
                4,
            ),
            (
                LdmatrixShapeAttr::M8n16,
                LdmatrixMultiplicityAttr::X1,
                LdmatrixLayoutAttr::Normal,
                LdmatrixElementAttr::B8x16B4x16P64,
                1,
            ),
            (
                LdmatrixShapeAttr::M8n16,
                LdmatrixMultiplicityAttr::X1,
                LdmatrixLayoutAttr::Normal,
                LdmatrixElementAttr::B8x16B6x16P32,
                1,
            ),
            (
                LdmatrixShapeAttr::M8n16,
                LdmatrixMultiplicityAttr::X2,
                LdmatrixLayoutAttr::Normal,
                LdmatrixElementAttr::B8x16B4x16P64,
                2,
            ),
            (
                LdmatrixShapeAttr::M8n16,
                LdmatrixMultiplicityAttr::X2,
                LdmatrixLayoutAttr::Normal,
                LdmatrixElementAttr::B8x16B6x16P32,
                2,
            ),
            (
                LdmatrixShapeAttr::M8n16,
                LdmatrixMultiplicityAttr::X4,
                LdmatrixLayoutAttr::Normal,
                LdmatrixElementAttr::B8x16B4x16P64,
                4,
            ),
            (
                LdmatrixShapeAttr::M8n16,
                LdmatrixMultiplicityAttr::X4,
                LdmatrixLayoutAttr::Normal,
                LdmatrixElementAttr::B8x16B6x16P32,
                4,
            ),
        ] {
            let op = LdmatrixOp::build(
                &mut ctx,
                pointer,
                shape,
                multiplicity,
                layout,
                element,
                LdmatrixStateSpaceAttr::Shared,
            );
            assert_eq!(op.deref(&ctx).get_num_results(), result_count);
            assert!(LdmatrixOp::new(op).verify(&ctx).is_ok());
        }
    }

    for (shape, multiplicity, layout, element) in [
        (
            LdmatrixShapeAttr::M16n16,
            LdmatrixMultiplicityAttr::X4,
            LdmatrixLayoutAttr::Transposed,
            LdmatrixElementAttr::B8,
        ),
        (
            LdmatrixShapeAttr::M16n16,
            LdmatrixMultiplicityAttr::X1,
            LdmatrixLayoutAttr::Normal,
            LdmatrixElementAttr::B8,
        ),
        (
            LdmatrixShapeAttr::M8n16,
            LdmatrixMultiplicityAttr::X1,
            LdmatrixLayoutAttr::Transposed,
            LdmatrixElementAttr::B8x16B4x16P64,
        ),
        (
            LdmatrixShapeAttr::M8n16,
            LdmatrixMultiplicityAttr::X1,
            LdmatrixLayoutAttr::Normal,
            LdmatrixElementAttr::B8,
        ),
    ] {
        let op = LdmatrixOp::build(
            &mut ctx,
            shared_pointer,
            shape,
            multiplicity,
            layout,
            element,
            LdmatrixStateSpaceAttr::Shared,
        );
        assert!(LdmatrixOp::new(op).verify(&ctx).is_err());
    }

    for pointer_index in [2, 3] {
        let pointer = block.deref(&ctx).get_argument(pointer_index);
        let op = LdmatrixOp::build(
            &mut ctx,
            pointer,
            LdmatrixShapeAttr::M16n16,
            LdmatrixMultiplicityAttr::X1,
            LdmatrixLayoutAttr::Transposed,
            LdmatrixElementAttr::B8,
            LdmatrixStateSpaceAttr::Shared,
        );
        assert!(LdmatrixOp::new(op).verify(&ctx).is_err());
    }
}

#[test]
fn classic_ldmatrix_compatibility_ops_keep_names_and_register_shapes() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let pointer_ty = MirPtrType::get_shared(&mut ctx, u32_ty.into(), false);
    let block = BasicBlock::new(&mut ctx, None, vec![pointer_ty.into()]);
    let pointer = block.deref(&ctx).get_argument(0);

    macro_rules! check_compat {
        ($op:ty, $name:literal, $results:literal) => {{
            assert_eq!(<$op>::get_opid_static().to_string(), $name);
            let valid = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![u32_ty.into(); $results],
                vec![pointer],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(valid), &ctx).is_ok());

            let wrong_shape = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![u32_ty.into(); $results + 1],
                vec![pointer],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(wrong_shape), &ctx).is_err());
        }};
    }

    check_compat!(LdmatrixX1Op, "nvvm.ldmatrix_x1", 1);
    check_compat!(LdmatrixX1TransOp, "nvvm.ldmatrix_x1_trans", 1);
    check_compat!(LdmatrixX2Op, "nvvm.ldmatrix_x2", 2);
    check_compat!(LdmatrixX2TransOp, "nvvm.ldmatrix_x2_trans", 2);
    check_compat!(LdmatrixX4Op, "nvvm.ldmatrix_x4", 4);
    check_compat!(LdmatrixX4TransOp, "nvvm.ldmatrix_x4_trans", 4);
}

#[test]
fn test_mma_m16n8k16_bf16_verifies_exact_register_signature() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let f32_ty = FP32Type::get(&ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![f32_ty.into(), i32_ty.into(), i64_ty.into()],
    );
    let f32_value = block.deref(&ctx).get_argument(0);
    let i32_value = block.deref(&ctx).get_argument(1);
    let i64_value = block.deref(&ctx).get_argument(2);

    let valid_operands = (0..4)
        .map(|_| f32_value)
        .chain((0..6).map(|_| i32_value))
        .collect();
    let valid = Operation::new(
        &mut ctx,
        MmaM16N8K16F32Bf16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        valid_operands,
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32Bf16Op::new(valid), &ctx).is_ok());

    let bad_c_operands = (0..4)
        .map(|index| if index == 0 { i32_value } else { f32_value })
        .chain((0..6).map(|_| i32_value))
        .collect();
    let bad_c = Operation::new(
        &mut ctx,
        MmaM16N8K16F32Bf16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        bad_c_operands,
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32Bf16Op::new(bad_c), &ctx).is_err());

    let bad_a_operands = (0..4)
        .map(|_| f32_value)
        .chain((0..6).map(|index| if index == 0 { i64_value } else { i32_value }))
        .collect();
    let bad_a = Operation::new(
        &mut ctx,
        MmaM16N8K16F32Bf16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        bad_a_operands,
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32Bf16Op::new(bad_a), &ctx).is_err());

    let bad_result = Operation::new(
        &mut ctx,
        MmaM16N8K16F32Bf16Op::get_concrete_op_info(),
        vec![f32_ty.into(), f32_ty.into(), f32_ty.into(), i32_ty.into()],
        (0..4)
            .map(|_| f32_value)
            .chain((0..6).map(|_| i32_value))
            .collect(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32Bf16Op::new(bad_result), &ctx).is_err());

    let bad_arity = Operation::new(
        &mut ctx,
        MmaM16N8K16F32Bf16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        (0..4)
            .map(|_| f32_value)
            .chain((0..5).map(|_| i32_value))
            .collect(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32Bf16Op::new(bad_arity), &ctx).is_err());
}

#[test]
fn test_mma_m16n8k16_f16_verifies_exact_register_signature() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let f32_ty = FP32Type::get(&ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![f32_ty.into(), i32_ty.into(), i64_ty.into()],
    );
    let f32_value = block.deref(&ctx).get_argument(0);
    let i32_value = block.deref(&ctx).get_argument(1);
    let i64_value = block.deref(&ctx).get_argument(2);

    let operands = || {
        (0..4)
            .map(|_| f32_value)
            .chain((0..6).map(|_| i32_value))
            .collect()
    };
    let valid = Operation::new(
        &mut ctx,
        MmaM16N8K16F32F16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        operands(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32F16Op::new(valid), &ctx).is_ok());

    let bad_c_operands = (0..4)
        .map(|index| if index == 0 { i32_value } else { f32_value })
        .chain((0..6).map(|_| i32_value))
        .collect();
    let bad_c = Operation::new(
        &mut ctx,
        MmaM16N8K16F32F16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        bad_c_operands,
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32F16Op::new(bad_c), &ctx).is_err());

    let bad_packed_operands = (0..4)
        .map(|_| f32_value)
        .chain((0..6).map(|index| if index == 0 { i64_value } else { i32_value }))
        .collect();
    let bad_packed = Operation::new(
        &mut ctx,
        MmaM16N8K16F32F16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        bad_packed_operands,
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32F16Op::new(bad_packed), &ctx).is_err());

    let bad_result_type = Operation::new(
        &mut ctx,
        MmaM16N8K16F32F16Op::get_concrete_op_info(),
        vec![f32_ty.into(), f32_ty.into(), f32_ty.into(), i32_ty.into()],
        operands(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32F16Op::new(bad_result_type), &ctx).is_err());

    let bad_operand_arity = Operation::new(
        &mut ctx,
        MmaM16N8K16F32F16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        (0..4)
            .map(|_| f32_value)
            .chain((0..5).map(|_| i32_value))
            .collect(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32F16Op::new(bad_operand_arity), &ctx).is_err());

    let bad_result_arity = Operation::new(
        &mut ctx,
        MmaM16N8K16F32F16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 3],
        operands(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K16F32F16Op::new(bad_result_arity), &ctx).is_err());
}

#[test]
fn test_mma_m16n8k8_tf32_verifies_exact_register_signature() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let f32_ty = FP32Type::get(&ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![f32_ty.into(), i32_ty.into(), i64_ty.into()],
    );
    let f32_value = block.deref(&ctx).get_argument(0);
    let i32_value = block.deref(&ctx).get_argument(1);
    let i64_value = block.deref(&ctx).get_argument(2);

    let operands = || {
        (0..4)
            .map(|_| f32_value)
            .chain((0..6).map(|_| i32_value))
            .collect()
    };
    let valid = Operation::new(
        &mut ctx,
        MmaM16N8K8F32Tf32Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        operands(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K8F32Tf32Op::new(valid), &ctx).is_ok());

    let bad_c_operands = (0..4)
        .map(|index| if index == 0 { i32_value } else { f32_value })
        .chain((0..6).map(|_| i32_value))
        .collect();
    let bad_c = Operation::new(
        &mut ctx,
        MmaM16N8K8F32Tf32Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        bad_c_operands,
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K8F32Tf32Op::new(bad_c), &ctx).is_err());

    let bad_packed_operands = (0..4)
        .map(|_| f32_value)
        .chain((0..6).map(|index| if index == 0 { i64_value } else { i32_value }))
        .collect();
    let bad_packed = Operation::new(
        &mut ctx,
        MmaM16N8K8F32Tf32Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        bad_packed_operands,
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K8F32Tf32Op::new(bad_packed), &ctx).is_err());

    let bad_result_type = Operation::new(
        &mut ctx,
        MmaM16N8K8F32Tf32Op::get_concrete_op_info(),
        vec![f32_ty.into(), f32_ty.into(), f32_ty.into(), i32_ty.into()],
        operands(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K8F32Tf32Op::new(bad_result_type), &ctx).is_err());

    let bad_operand_arity = Operation::new(
        &mut ctx,
        MmaM16N8K8F32Tf32Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        (0..4)
            .map(|_| f32_value)
            .chain((0..5).map(|_| i32_value))
            .collect(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K8F32Tf32Op::new(bad_operand_arity), &ctx).is_err());

    let bad_result_arity = Operation::new(
        &mut ctx,
        MmaM16N8K8F32Tf32Op::get_concrete_op_info(),
        vec![f32_ty.into(); 3],
        operands(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K8F32Tf32Op::new(bad_result_arity), &ctx).is_err());
}

#[test]
fn test_mma_m16n8k32_s8_verifies_exact_register_signature() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let f32_ty = FP32Type::get(&ctx);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![i32_ty.into(), i64_ty.into(), f32_ty.into()],
    );
    let i32_value = block.deref(&ctx).get_argument(0);
    let i64_value = block.deref(&ctx).get_argument(1);
    let f32_value = block.deref(&ctx).get_argument(2);
    let valid_operands = vec![i32_value; 10];

    let valid = Operation::new(
        &mut ctx,
        MmaM16N8K32S32S8Op::get_concrete_op_info(),
        vec![i32_ty.into(); 4],
        valid_operands.clone(),
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K32S32S8Op::new(valid), &ctx).is_ok());

    for bad_value in [i64_value, f32_value] {
        let mut bad_operands = valid_operands.clone();
        bad_operands[4] = bad_value;
        let invalid = Operation::new(
            &mut ctx,
            MmaM16N8K32S32S8Op::get_concrete_op_info(),
            vec![i32_ty.into(); 4],
            bad_operands,
            vec![],
            0,
        );
        assert!(
            verify_op(&MmaM16N8K32S32S8Op::new(invalid), &ctx).is_err(),
            "MMA must reject non-i32 register operands"
        );
    }

    for bad_results in [
        vec![i32_ty.into(), i32_ty.into(), i32_ty.into(), i64_ty.into()],
        vec![i32_ty.into(), i32_ty.into(), i32_ty.into(), f32_ty.into()],
        vec![i32_ty.into(); 3],
    ] {
        let invalid = Operation::new(
            &mut ctx,
            MmaM16N8K32S32S8Op::get_concrete_op_info(),
            bad_results,
            valid_operands.clone(),
            vec![],
            0,
        );
        assert!(
            verify_op(&MmaM16N8K32S32S8Op::new(invalid), &ctx).is_err(),
            "MMA must reject the wrong result register signature"
        );
    }

    let invalid_arity = Operation::new(
        &mut ctx,
        MmaM16N8K32S32S8Op::get_concrete_op_info(),
        vec![i32_ty.into(); 4],
        vec![i32_value; 9],
        vec![],
        0,
    );
    assert!(verify_op(&MmaM16N8K32S32S8Op::new(invalid_arity), &ctx).is_err());
}

#[test]
fn generated_register_mma_verifier_rejects_crossed_variants_and_carriers() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    macro_rules! set_variant {
        ($op:expr, $shape:expr, $acc:expr, $a:expr, $b:expr, $overflow:expr) => {{
            $op.set_attr_nvvm_register_mma_shape(&ctx, $shape);
            $op.set_attr_nvvm_register_mma_accumulator(&ctx, $acc);
            $op.set_attr_nvvm_register_mma_a_element(&ctx, $a);
            $op.set_attr_nvvm_register_mma_b_element(&ctx, $b);
            $op.set_attr_nvvm_register_mma_a_layout(&ctx, RegisterMmaLayoutAttr::Row);
            $op.set_attr_nvvm_register_mma_b_layout(&ctx, RegisterMmaLayoutAttr::Col);
            $op.set_attr_nvvm_register_mma_overflow(&ctx, $overflow);
        }};
    }

    let f32_ty = FP32Type::get(&ctx);
    let f64_ty = FP64Type::get(&ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let signless_i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![f32_ty.into(), f64_ty.into(), i32_ty.into(), u32_ty.into()],
    );
    let f32_value = block.deref(&ctx).get_argument(0);
    let f64_value = block.deref(&ctx).get_argument(1);
    let i32_value = block.deref(&ctx).get_argument(2);
    let u32_value = block.deref(&ctx).get_argument(3);

    let bf16_operation = Operation::new(
        &mut ctx,
        RegisterMmaOp::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        [vec![f32_value; 4], vec![u32_value; 6]].concat(),
        vec![],
        0,
    );
    let bf16 = RegisterMmaOp::new(bf16_operation);
    set_variant!(
        bf16,
        RegisterMmaShapeAttr::M16n8k16,
        RegisterMmaAccumulatorAttr::F32,
        RegisterMmaElementAttr::Bf16,
        RegisterMmaElementAttr::Bf16,
        RegisterMmaOverflowAttr::NotApplicable
    );
    assert!(bf16.get_attr_nvvm_register_mma_operation(&ctx).is_none());
    assert!(verify_op(&bf16, &ctx).is_ok());
    bf16.set_attr_nvvm_register_mma_b_element(&ctx, RegisterMmaElementAttr::F16);
    assert!(verify_op(&bf16, &ctx).is_err());

    let f64_operation = Operation::new(
        &mut ctx,
        RegisterMmaOp::get_concrete_op_info(),
        vec![f64_ty.into(); 2],
        vec![f64_value; 4],
        vec![],
        0,
    );
    let f64_mma = RegisterMmaOp::new(f64_operation);
    set_variant!(
        f64_mma,
        RegisterMmaShapeAttr::M8n8k4,
        RegisterMmaAccumulatorAttr::F64,
        RegisterMmaElementAttr::F64,
        RegisterMmaElementAttr::F64,
        RegisterMmaOverflowAttr::NotApplicable
    );
    assert!(verify_op(&f64_mma, &ctx).is_ok());

    let int_operands = [vec![i32_value; 4], vec![u32_value; 6]].concat();
    let int_operation = Operation::new(
        &mut ctx,
        RegisterMmaOp::get_concrete_op_info(),
        vec![i32_ty.into(); 4],
        int_operands.clone(),
        vec![],
        0,
    );
    let int_mma = RegisterMmaOp::new(int_operation);
    set_variant!(
        int_mma,
        RegisterMmaShapeAttr::M16n8k32,
        RegisterMmaAccumulatorAttr::S32,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::S8,
        RegisterMmaOverflowAttr::Wrapping
    );
    assert!(verify_op(&int_mma, &ctx).is_ok());

    let wrong_signedness = Operation::new(
        &mut ctx,
        RegisterMmaOp::get_concrete_op_info(),
        vec![signless_i32_ty.into(); 4],
        int_operands,
        vec![],
        0,
    );
    let wrong_signedness = RegisterMmaOp::new(wrong_signedness);
    set_variant!(
        wrong_signedness,
        RegisterMmaShapeAttr::M16n8k32,
        RegisterMmaAccumulatorAttr::S32,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::S8,
        RegisterMmaOverflowAttr::Wrapping
    );
    assert!(verify_op(&wrong_signedness, &ctx).is_err());

    let missing_attributes = Operation::new(
        &mut ctx,
        RegisterMmaOp::get_concrete_op_info(),
        vec![f64_ty.into(); 2],
        vec![f64_value; 4],
        vec![],
        0,
    );
    assert!(verify_op(&RegisterMmaOp::new(missing_attributes), &ctx).is_err());
}

#[test]
fn generated_register_mma_verifies_dense_integer_families() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let signless_i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![i32_ty.into(), u32_ty.into(), signless_i32_ty.into()],
    );
    let i32_value = block.deref(&ctx).get_argument(0);
    let u32_value = block.deref(&ctx).get_argument(1);

    macro_rules! int_mma {
        ($shape:expr, $a:expr, $b:expr, $overflow:expr, $operands:expr, $results:expr) => {{
            let operation = Operation::new(
                &mut ctx,
                RegisterMmaOp::get_concrete_op_info(),
                $results,
                $operands,
                vec![],
                0,
            );
            let mma = RegisterMmaOp::new(operation);
            mma.set_attr_nvvm_register_mma_shape(&ctx, $shape);
            mma.set_attr_nvvm_register_mma_operation(&ctx, RegisterMmaOperationAttr::Multiply);
            mma.set_attr_nvvm_register_mma_accumulator(&ctx, RegisterMmaAccumulatorAttr::S32);
            mma.set_attr_nvvm_register_mma_a_element(&ctx, $a);
            mma.set_attr_nvvm_register_mma_b_element(&ctx, $b);
            mma.set_attr_nvvm_register_mma_a_layout(&ctx, RegisterMmaLayoutAttr::Row);
            mma.set_attr_nvvm_register_mma_b_layout(&ctx, RegisterMmaLayoutAttr::Col);
            mma.set_attr_nvvm_register_mma_overflow(&ctx, $overflow);
            mma
        }};
    }

    let mut accepted = 0;
    for (shape, accumulator_count, operand_count, result_count) in [
        (RegisterMmaShapeAttr::M8n8k16, 2, 4, 2),
        (RegisterMmaShapeAttr::M16n8k16, 4, 7, 4),
        (RegisterMmaShapeAttr::M16n8k32, 4, 10, 4),
    ] {
        for (a_element, b_element) in [
            (RegisterMmaElementAttr::S8, RegisterMmaElementAttr::S8),
            (RegisterMmaElementAttr::S8, RegisterMmaElementAttr::U8),
            (RegisterMmaElementAttr::U8, RegisterMmaElementAttr::S8),
            (RegisterMmaElementAttr::U8, RegisterMmaElementAttr::U8),
        ] {
            for overflow in [
                RegisterMmaOverflowAttr::Wrapping,
                RegisterMmaOverflowAttr::Satfinite,
            ] {
                let operands = [
                    vec![i32_value; accumulator_count],
                    vec![u32_value; operand_count - accumulator_count],
                ]
                .concat();
                let mma = int_mma!(
                    shape.clone(),
                    a_element.clone(),
                    b_element.clone(),
                    overflow.clone(),
                    operands,
                    vec![i32_ty.into(); result_count]
                );
                assert_eq!(
                    mma.get_operation().deref(&ctx).get_num_operands(),
                    operand_count
                );
                assert_eq!(
                    mma.get_operation().deref(&ctx).get_num_results(),
                    result_count
                );
                assert!(
                    verify_op(&mma, &ctx).is_ok(),
                    "rejected {shape:?} {a_element:?}x{b_element:?} {overflow:?}"
                );
                accepted += 1;
            }
        }
    }
    assert_eq!(accepted, 24);

    let mut int4_accepted = 0;
    for (shape, accumulator_count, a_count, b_count, result_count) in [
        (RegisterMmaShapeAttr::M8n8k32, 2, 1, 1, 2),
        (RegisterMmaShapeAttr::M16n8k32, 4, 2, 1, 4),
        (RegisterMmaShapeAttr::M16n8k64, 4, 4, 2, 4),
    ] {
        for (a_element, b_element) in [
            (RegisterMmaElementAttr::S4, RegisterMmaElementAttr::S4),
            (RegisterMmaElementAttr::S4, RegisterMmaElementAttr::U4),
            (RegisterMmaElementAttr::U4, RegisterMmaElementAttr::S4),
            (RegisterMmaElementAttr::U4, RegisterMmaElementAttr::U4),
        ] {
            for overflow in [
                RegisterMmaOverflowAttr::Wrapping,
                RegisterMmaOverflowAttr::Satfinite,
            ] {
                let operands = [
                    vec![i32_value; accumulator_count],
                    vec![u32_value; a_count],
                    vec![u32_value; b_count],
                ]
                .concat();
                let expected_operand_types = [
                    vec![i32_ty.into(); accumulator_count],
                    vec![u32_ty.into(); a_count],
                    vec![u32_ty.into(); b_count],
                ]
                .concat();
                let mma = int_mma!(
                    shape.clone(),
                    a_element.clone(),
                    b_element.clone(),
                    overflow.clone(),
                    operands,
                    vec![i32_ty.into(); result_count]
                );
                let operation = mma.get_operation().deref(&ctx);
                assert_eq!(
                    operation.get_num_operands(),
                    accumulator_count + a_count + b_count
                );
                assert_eq!(operation.get_num_results(), result_count);
                assert_eq!(
                    operation
                        .operands()
                        .map(|operand| operand.get_type(&ctx))
                        .collect::<Vec<_>>(),
                    expected_operand_types
                );
                assert_eq!(
                    (0..operation.get_num_results())
                        .map(|index| operation.get_result(index).get_type(&ctx))
                        .collect::<Vec<_>>(),
                    vec![i32_ty.into(); result_count]
                );
                assert_eq!(
                    mma.get_attr_nvvm_register_mma_shape(&ctx).as_deref(),
                    Some(&shape)
                );
                assert_eq!(
                    mma.get_attr_nvvm_register_mma_accumulator(&ctx).as_deref(),
                    Some(&RegisterMmaAccumulatorAttr::S32)
                );
                assert_eq!(
                    mma.get_attr_nvvm_register_mma_a_element(&ctx).as_deref(),
                    Some(&a_element)
                );
                assert_eq!(
                    mma.get_attr_nvvm_register_mma_b_element(&ctx).as_deref(),
                    Some(&b_element)
                );
                assert_eq!(
                    mma.get_attr_nvvm_register_mma_a_layout(&ctx).as_deref(),
                    Some(&RegisterMmaLayoutAttr::Row)
                );
                assert_eq!(
                    mma.get_attr_nvvm_register_mma_b_layout(&ctx).as_deref(),
                    Some(&RegisterMmaLayoutAttr::Col)
                );
                assert_eq!(
                    mma.get_attr_nvvm_register_mma_overflow(&ctx).as_deref(),
                    Some(&overflow)
                );
                assert!(
                    verify_op(&mma, &ctx).is_ok(),
                    "rejected {shape:?} {a_element:?}x{b_element:?} {overflow:?}"
                );
                int4_accepted += 1;
            }
        }
    }
    assert_eq!(int4_accepted, 24);

    for (shape, accumulator_count, operand_count, result_count) in [
        (RegisterMmaShapeAttr::M8n8k32, 2, 4, 2),
        (RegisterMmaShapeAttr::M16n8k32, 4, 7, 4),
        (RegisterMmaShapeAttr::M16n8k64, 4, 10, 4),
    ] {
        for wrong_operand_count in [operand_count - 1, operand_count + 1] {
            let mma = int_mma!(
                shape.clone(),
                RegisterMmaElementAttr::S4,
                RegisterMmaElementAttr::U4,
                RegisterMmaOverflowAttr::Wrapping,
                [
                    vec![i32_value; accumulator_count],
                    vec![u32_value; wrong_operand_count - accumulator_count],
                ]
                .concat(),
                vec![i32_ty.into(); result_count]
            );
            assert!(verify_op(&mma, &ctx).is_err());
        }

        for wrong_result_count in [result_count - 1, result_count + 1] {
            let mma = int_mma!(
                shape.clone(),
                RegisterMmaElementAttr::U4,
                RegisterMmaElementAttr::S4,
                RegisterMmaOverflowAttr::Satfinite,
                [
                    vec![i32_value; accumulator_count],
                    vec![u32_value; operand_count - accumulator_count],
                ]
                .concat(),
                vec![i32_ty.into(); wrong_result_count]
            );
            assert!(verify_op(&mma, &ctx).is_err());
        }
    }

    let int4_on_int8_shape = int_mma!(
        RegisterMmaShapeAttr::M8n8k16,
        RegisterMmaElementAttr::S4,
        RegisterMmaElementAttr::U4,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 2], vec![u32_value; 2]].concat(),
        vec![i32_ty.into(); 2]
    );
    assert!(verify_op(&int4_on_int8_shape, &ctx).is_err());

    let int8_on_int4_shape = int_mma!(
        RegisterMmaShapeAttr::M8n8k32,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::U8,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 2], vec![u32_value; 2]].concat(),
        vec![i32_ty.into(); 2]
    );
    assert!(verify_op(&int8_on_int4_shape, &ctx).is_err());

    let crossed_integer_width = int_mma!(
        RegisterMmaShapeAttr::M16n8k32,
        RegisterMmaElementAttr::S4,
        RegisterMmaElementAttr::U8,
        RegisterMmaOverflowAttr::Satfinite,
        [vec![i32_value; 4], vec![u32_value; 3]].concat(),
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&crossed_integer_width, &ctx).is_err());

    let m16k32_int4_with_int8_carriers = int_mma!(
        RegisterMmaShapeAttr::M16n8k32,
        RegisterMmaElementAttr::U4,
        RegisterMmaElementAttr::S4,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 4], vec![u32_value; 6]].concat(),
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&m16k32_int4_with_int8_carriers, &ctx).is_err());

    let m16k32_int8_with_int4_carriers = int_mma!(
        RegisterMmaShapeAttr::M16n8k32,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::U8,
        RegisterMmaOverflowAttr::Satfinite,
        [vec![i32_value; 4], vec![u32_value; 3]].concat(),
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&m16k32_int8_with_int4_carriers, &ctx).is_err());

    let m16k64_int4_with_k32_carriers = int_mma!(
        RegisterMmaShapeAttr::M16n8k64,
        RegisterMmaElementAttr::S4,
        RegisterMmaElementAttr::U4,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 4], vec![u32_value; 3]].concat(),
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&m16k64_int4_with_k32_carriers, &ctx).is_err());

    let m16k32_int4_with_k64_carriers = int_mma!(
        RegisterMmaShapeAttr::M16n8k32,
        RegisterMmaElementAttr::U4,
        RegisterMmaElementAttr::S4,
        RegisterMmaOverflowAttr::Satfinite,
        [vec![i32_value; 4], vec![u32_value; 6]].concat(),
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&m16k32_int4_with_k64_carriers, &ctx).is_err());

    let int8_on_m16k64_shape = int_mma!(
        RegisterMmaShapeAttr::M16n8k64,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::U8,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 4], vec![u32_value; 6]].concat(),
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&int8_on_m16k64_shape, &ctx).is_err());

    for (shape, accumulator_count, operand_count, result_count) in [
        (RegisterMmaShapeAttr::M8n8k16, 2, 4, 2),
        (RegisterMmaShapeAttr::M16n8k16, 4, 7, 4),
        (RegisterMmaShapeAttr::M16n8k32, 4, 10, 4),
    ] {
        for wrong_count in [operand_count - 1, operand_count + 1] {
            let operands = [
                vec![i32_value; accumulator_count],
                vec![u32_value; wrong_count - accumulator_count],
            ]
            .concat();
            let mma = int_mma!(
                shape.clone(),
                RegisterMmaElementAttr::S8,
                RegisterMmaElementAttr::U8,
                RegisterMmaOverflowAttr::Wrapping,
                operands,
                vec![i32_ty.into(); result_count]
            );
            assert!(verify_op(&mma, &ctx).is_err());
        }
    }

    let m8_wrong_result_signedness = int_mma!(
        RegisterMmaShapeAttr::M8n8k16,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::U8,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 2], vec![u32_value; 2]].concat(),
        vec![signless_i32_ty.into(); 2]
    );
    assert!(verify_op(&m8_wrong_result_signedness, &ctx).is_err());

    let m8_wrong_accumulator_signedness = int_mma!(
        RegisterMmaShapeAttr::M8n8k16,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::U8,
        RegisterMmaOverflowAttr::Wrapping,
        vec![u32_value; 4],
        vec![i32_ty.into(); 2]
    );
    assert!(verify_op(&m8_wrong_accumulator_signedness, &ctx).is_err());

    let m8_wrong_fragment_signedness = int_mma!(
        RegisterMmaShapeAttr::M8n8k16,
        RegisterMmaElementAttr::U8,
        RegisterMmaElementAttr::S8,
        RegisterMmaOverflowAttr::Satfinite,
        vec![i32_value; 4],
        vec![i32_ty.into(); 2]
    );
    assert!(verify_op(&m8_wrong_fragment_signedness, &ctx).is_err());

    let m8_crossed_element = int_mma!(
        RegisterMmaShapeAttr::M8n8k16,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::F16,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 2], vec![u32_value; 2]].concat(),
        vec![i32_ty.into(); 2]
    );
    assert!(verify_op(&m8_crossed_element, &ctx).is_err());

    let m8_crossed_overflow = int_mma!(
        RegisterMmaShapeAttr::M8n8k16,
        RegisterMmaElementAttr::U8,
        RegisterMmaElementAttr::U8,
        RegisterMmaOverflowAttr::NotApplicable,
        [vec![i32_value; 2], vec![u32_value; 2]].concat(),
        vec![i32_ty.into(); 2]
    );
    assert!(verify_op(&m8_crossed_overflow, &ctx).is_err());

    let m8_crossed_carrier_shape = int_mma!(
        RegisterMmaShapeAttr::M8n8k16,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::S8,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 4], vec![u32_value; 3]].concat(),
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&m8_crossed_carrier_shape, &ctx).is_err());

    let m8_crossed_shape = int_mma!(
        RegisterMmaShapeAttr::M8n8k4,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::S8,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 2], vec![u32_value; 2]].concat(),
        vec![i32_ty.into(); 2]
    );
    assert!(verify_op(&m8_crossed_shape, &ctx).is_err());

    let wrong_result_signedness = int_mma!(
        RegisterMmaShapeAttr::M16n8k16,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::U8,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 4], vec![u32_value; 3]].concat(),
        vec![signless_i32_ty.into(); 4]
    );
    assert!(verify_op(&wrong_result_signedness, &ctx).is_err());

    let wrong_accumulator_signedness = int_mma!(
        RegisterMmaShapeAttr::M16n8k16,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::U8,
        RegisterMmaOverflowAttr::Wrapping,
        vec![u32_value; 7],
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&wrong_accumulator_signedness, &ctx).is_err());

    let wrong_fragment_signedness = int_mma!(
        RegisterMmaShapeAttr::M16n8k32,
        RegisterMmaElementAttr::U8,
        RegisterMmaElementAttr::S8,
        RegisterMmaOverflowAttr::Satfinite,
        vec![i32_value; 10],
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&wrong_fragment_signedness, &ctx).is_err());

    let crossed_element = int_mma!(
        RegisterMmaShapeAttr::M16n8k16,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::F16,
        RegisterMmaOverflowAttr::Wrapping,
        [vec![i32_value; 4], vec![u32_value; 3]].concat(),
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&crossed_element, &ctx).is_err());

    let crossed_overflow = int_mma!(
        RegisterMmaShapeAttr::M16n8k32,
        RegisterMmaElementAttr::U8,
        RegisterMmaElementAttr::U8,
        RegisterMmaOverflowAttr::NotApplicable,
        [vec![i32_value; 4], vec![u32_value; 6]].concat(),
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&crossed_overflow, &ctx).is_err());

    let crossed_shape = int_mma!(
        RegisterMmaShapeAttr::M16n8k8,
        RegisterMmaElementAttr::S8,
        RegisterMmaElementAttr::S8,
        RegisterMmaOverflowAttr::Satfinite,
        [vec![i32_value; 4], vec![u32_value; 3]].concat(),
        vec![i32_ty.into(); 4]
    );
    assert!(verify_op(&crossed_shape, &ctx).is_err());
}

#[test]
fn generated_register_mma_verifies_dense_b1_families() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    fn b1_mma(
        ctx: &mut Context,
        shape: RegisterMmaShapeAttr,
        operation: Option<RegisterMmaOperationAttr>,
    ) -> RegisterMmaOp {
        let (accumulator_count, a_count, b_count, result_count) = match shape {
            RegisterMmaShapeAttr::M8n8k128 => (2, 1, 1, 2),
            RegisterMmaShapeAttr::M16n8k128 => (4, 2, 1, 4),
            RegisterMmaShapeAttr::M16n8k256 => (4, 4, 2, 4),
            _ => panic!("unsupported B1 MMA shape"),
        };
        let i32_ty = IntegerType::get(ctx, 32, Signedness::Signed);
        let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
        let argument_types = (0..accumulator_count)
            .map(|_| i32_ty.into())
            .chain((0..a_count + b_count).map(|_| u32_ty.into()))
            .collect();
        let block = BasicBlock::new(ctx, None, argument_types);
        let operands = (0..accumulator_count + a_count + b_count)
            .map(|index| block.deref(ctx).get_argument(index))
            .collect();
        let op = Operation::new(
            ctx,
            RegisterMmaOp::get_concrete_op_info(),
            vec![i32_ty.into(); result_count],
            operands,
            vec![],
            0,
        );
        let mma = RegisterMmaOp::new(op);
        mma.set_attr_nvvm_register_mma_shape(ctx, shape);
        if let Some(operation) = operation {
            mma.set_attr_nvvm_register_mma_operation(ctx, operation);
        }
        mma.set_attr_nvvm_register_mma_accumulator(ctx, RegisterMmaAccumulatorAttr::S32);
        mma.set_attr_nvvm_register_mma_a_element(ctx, RegisterMmaElementAttr::B1);
        mma.set_attr_nvvm_register_mma_b_element(ctx, RegisterMmaElementAttr::B1);
        mma.set_attr_nvvm_register_mma_a_layout(ctx, RegisterMmaLayoutAttr::Row);
        mma.set_attr_nvvm_register_mma_b_layout(ctx, RegisterMmaLayoutAttr::Col);
        mma.set_attr_nvvm_register_mma_overflow(ctx, RegisterMmaOverflowAttr::Wrapping);
        mma
    }

    let mut accepted = 0;
    for shape in [
        RegisterMmaShapeAttr::M8n8k128,
        RegisterMmaShapeAttr::M16n8k128,
        RegisterMmaShapeAttr::M16n8k256,
    ] {
        for operation in [
            RegisterMmaOperationAttr::XorPopc,
            RegisterMmaOperationAttr::AndPopc,
        ] {
            let mma = b1_mma(&mut ctx, shape.clone(), Some(operation));
            assert!(verify_op(&mma, &ctx).is_ok(), "rejected {shape:?}");
            accepted += 1;
        }
    }
    assert_eq!(accepted, 6);

    let multiply = b1_mma(
        &mut ctx,
        RegisterMmaShapeAttr::M8n8k128,
        Some(RegisterMmaOperationAttr::Multiply),
    );
    assert!(verify_op(&multiply, &ctx).is_err());

    let wrong_shape = b1_mma(
        &mut ctx,
        RegisterMmaShapeAttr::M16n8k128,
        Some(RegisterMmaOperationAttr::XorPopc),
    );
    wrong_shape.set_attr_nvvm_register_mma_shape(&ctx, RegisterMmaShapeAttr::M16n8k64);
    assert!(verify_op(&wrong_shape, &ctx).is_err());

    let missing_operation = b1_mma(&mut ctx, RegisterMmaShapeAttr::M8n8k128, None);
    assert!(verify_op(&missing_operation, &ctx).is_err());
}

#[test]
fn generated_sparse_mma_verifies_all_int8_variants_and_metadata_modes() {
    use dialect_mir::ops::MirConstantOp;
    use pliron::builtin::{attributes::IntegerAttr, ops::ConstantOp};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let signless_i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![i32_ty.into(), u32_ty.into(), signless_i32_ty.into()],
    );
    let i32_value = block.deref(&ctx).get_argument(0);
    let u32_value = block.deref(&ctx).get_argument(1);
    let signless_i32_value = block.deref(&ctx).get_argument(2);

    let integer = |value| {
        IntegerAttr::new(
            u32_ty,
            APInt::from_u32(value, NonZeroUsize::new(32).unwrap()),
        )
    };
    let builtin_zero = ConstantOp::new(&mut ctx, integer(0).into());
    let builtin_zero = builtin_zero.get_operation().deref(&ctx).get_result(0);
    let builtin_two = ConstantOp::new(&mut ctx, integer(2).into());
    let builtin_two = builtin_two.get_operation().deref(&ctx).get_result(0);
    let mir_one = Operation::new(
        &mut ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![],
        vec![],
        0,
    );
    MirConstantOp::new(mir_one).set_attr_value(&ctx, integer(1));
    let mir_one = mir_one.deref(&ctx).get_result(0);

    macro_rules! sparse_mma {
        ($operands:expr, $results:expr, $a:expr, $b:expr, $overflow:expr, $metadata:expr) => {{
            let operation = Operation::new(
                &mut ctx,
                SparseMmaOp::get_concrete_op_info(),
                $results,
                $operands,
                vec![],
                0,
            );
            let mma = SparseMmaOp::new(operation);
            mma.set_attr_nvvm_sparse_mma_shape(&ctx, SparseMmaShapeAttr::M16n8k32);
            mma.set_attr_nvvm_sparse_mma_accumulator(&ctx, SparseMmaAccumulatorAttr::S32);
            mma.set_attr_nvvm_sparse_mma_a_element(&ctx, $a);
            mma.set_attr_nvvm_sparse_mma_b_element(&ctx, $b);
            mma.set_attr_nvvm_sparse_mma_a_layout(&ctx, SparseMmaLayoutAttr::Row);
            mma.set_attr_nvvm_sparse_mma_b_layout(&ctx, SparseMmaLayoutAttr::Col);
            mma.set_attr_nvvm_sparse_mma_overflow(&ctx, $overflow);
            mma.set_attr_nvvm_sparse_mma_metadata(&ctx, $metadata);
            mma.set_attr_nvvm_sparse_mma_selector(&ctx, SparseMmaSelectorAttr::ImmediateZeroOrOne);
            mma
        }};
    }

    let variants = [
        (
            SparseMmaElementAttr::S8,
            SparseMmaElementAttr::S8,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::S8,
            SparseMmaElementAttr::U8,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::U8,
            SparseMmaElementAttr::U8,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::U8,
            SparseMmaElementAttr::S8,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::S8,
            SparseMmaElementAttr::S8,
            SparseMmaOverflowAttr::Satfinite,
        ),
        (
            SparseMmaElementAttr::S8,
            SparseMmaElementAttr::U8,
            SparseMmaOverflowAttr::Satfinite,
        ),
        (
            SparseMmaElementAttr::U8,
            SparseMmaElementAttr::U8,
            SparseMmaOverflowAttr::Satfinite,
        ),
        (
            SparseMmaElementAttr::U8,
            SparseMmaElementAttr::S8,
            SparseMmaOverflowAttr::Satfinite,
        ),
    ];
    for metadata in [
        SparseMmaMetadataAttr::Standard,
        SparseMmaMetadataAttr::Ordered,
    ] {
        for (index, (a_element, b_element, overflow)) in variants.iter().enumerate() {
            let selector = if index % 2 == 0 {
                builtin_zero
            } else {
                mir_one
            };
            let operands = [vec![i32_value; 4], vec![u32_value; 5], vec![selector]].concat();
            let mma = sparse_mma!(
                operands,
                vec![i32_ty.into(); 4],
                a_element.clone(),
                b_element.clone(),
                overflow.clone(),
                metadata.clone()
            );
            assert_eq!(
                mma.get_attr_nvvm_sparse_mma_a_element(&ctx).as_deref(),
                Some(a_element)
            );
            assert_eq!(
                mma.get_attr_nvvm_sparse_mma_b_element(&ctx).as_deref(),
                Some(b_element)
            );
            assert_eq!(
                mma.get_attr_nvvm_sparse_mma_overflow(&ctx).as_deref(),
                Some(overflow)
            );
            assert_eq!(
                mma.get_attr_nvvm_sparse_mma_metadata(&ctx).as_deref(),
                Some(&metadata)
            );
            assert!(
                verify_op(&mma, &ctx).is_ok(),
                "rejected sparse {metadata:?} {a_element:?}x{b_element:?} {overflow:?}"
            );
        }
    }

    for metadata in [
        SparseMmaMetadataAttr::Standard,
        SparseMmaMetadataAttr::Ordered,
    ] {
        for selector in [u32_value, builtin_two] {
            let invalid = sparse_mma!(
                [vec![i32_value; 4], vec![u32_value; 5], vec![selector],].concat(),
                vec![i32_ty.into(); 4],
                SparseMmaElementAttr::S8,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Wrapping,
                metadata.clone()
            );
            assert!(verify_op(&invalid, &ctx).is_err());
        }
    }

    let wrong_accumulator_type = sparse_mma!(
        [
            vec![signless_i32_value; 4],
            vec![u32_value; 5],
            vec![builtin_zero],
        ]
        .concat(),
        vec![i32_ty.into(); 4],
        SparseMmaElementAttr::U8,
        SparseMmaElementAttr::S8,
        SparseMmaOverflowAttr::Satfinite,
        SparseMmaMetadataAttr::Standard
    );
    assert!(verify_op(&wrong_accumulator_type, &ctx).is_err());

    let wrong_count = sparse_mma!(
        [vec![i32_value; 4], vec![u32_value; 4], vec![builtin_zero],].concat(),
        vec![i32_ty.into(); 4],
        SparseMmaElementAttr::S8,
        SparseMmaElementAttr::S8,
        SparseMmaOverflowAttr::Wrapping,
        SparseMmaMetadataAttr::Standard
    );
    assert!(verify_op(&wrong_count, &ctx).is_err());

    let wrong_results = sparse_mma!(
        [vec![i32_value; 4], vec![u32_value; 5], vec![mir_one],].concat(),
        vec![signless_i32_ty.into(); 4],
        SparseMmaElementAttr::U8,
        SparseMmaElementAttr::U8,
        SparseMmaOverflowAttr::Wrapping,
        SparseMmaMetadataAttr::Standard
    );
    assert!(verify_op(&wrong_results, &ctx).is_err());

    let wrong_layout = sparse_mma!(
        [vec![i32_value; 4], vec![u32_value; 5], vec![builtin_zero],].concat(),
        vec![i32_ty.into(); 4],
        SparseMmaElementAttr::S8,
        SparseMmaElementAttr::U8,
        SparseMmaOverflowAttr::Satfinite,
        SparseMmaMetadataAttr::Standard
    );
    wrong_layout.set_attr_nvvm_sparse_mma_b_layout(&ctx, SparseMmaLayoutAttr::Row);
    assert!(verify_op(&wrong_layout, &ctx).is_err());
}

#[test]
fn generated_sparse_mma_m16n8k64_verifies_selector_and_carriers() {
    use pliron::builtin::{attributes::IntegerAttr, ops::ConstantOp};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let signless_i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![i32_ty.into(), u32_ty.into(), signless_i32_ty.into()],
    );
    let i32_value = block.deref(&ctx).get_argument(0);
    let u32_value = block.deref(&ctx).get_argument(1);
    let signless_i32_value = block.deref(&ctx).get_argument(2);
    let integer = |value| {
        IntegerAttr::new(
            u32_ty,
            APInt::from_u32(value, NonZeroUsize::new(32).unwrap()),
        )
    };
    let zero = ConstantOp::new(&mut ctx, integer(0).into())
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let one = ConstantOp::new(&mut ctx, integer(1).into())
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let two = ConstantOp::new(&mut ctx, integer(2).into())
        .get_operation()
        .deref(&ctx)
        .get_result(0);

    macro_rules! k64_mma {
        ($operands:expr, $metadata:expr) => {{
            let operation = Operation::new(
                &mut ctx,
                SparseMmaOp::get_concrete_op_info(),
                vec![i32_ty.into(); 4],
                $operands,
                vec![],
                0,
            );
            let mma = SparseMmaOp::new(operation);
            mma.set_attr_nvvm_sparse_mma_shape(&ctx, SparseMmaShapeAttr::M16n8k64);
            mma.set_attr_nvvm_sparse_mma_accumulator(&ctx, SparseMmaAccumulatorAttr::S32);
            mma.set_attr_nvvm_sparse_mma_a_element(&ctx, SparseMmaElementAttr::S8);
            mma.set_attr_nvvm_sparse_mma_b_element(&ctx, SparseMmaElementAttr::U8);
            mma.set_attr_nvvm_sparse_mma_a_layout(&ctx, SparseMmaLayoutAttr::Row);
            mma.set_attr_nvvm_sparse_mma_b_layout(&ctx, SparseMmaLayoutAttr::Col);
            mma.set_attr_nvvm_sparse_mma_overflow(&ctx, SparseMmaOverflowAttr::Wrapping);
            mma.set_attr_nvvm_sparse_mma_metadata(&ctx, $metadata);
            mma.set_attr_nvvm_sparse_mma_selector(&ctx, SparseMmaSelectorAttr::ImmediateZero);
            mma
        }};
    }

    let operands = |selector| [vec![i32_value; 4], vec![u32_value; 9], vec![selector]].concat();
    assert!(
        verify_op(
            &k64_mma!(operands(zero), SparseMmaMetadataAttr::Standard),
            &ctx
        )
        .is_ok()
    );
    assert!(
        verify_op(
            &k64_mma!(operands(zero), SparseMmaMetadataAttr::Ordered),
            &ctx
        )
        .is_ok()
    );
    assert!(
        verify_op(
            &k64_mma!(operands(one), SparseMmaMetadataAttr::Standard),
            &ctx
        )
        .is_err()
    );
    assert!(
        verify_op(
            &k64_mma!(operands(one), SparseMmaMetadataAttr::Ordered),
            &ctx
        )
        .is_err()
    );
    assert!(
        verify_op(
            &k64_mma!(operands(u32_value), SparseMmaMetadataAttr::Standard),
            &ctx
        )
        .is_err()
    );
    assert!(
        verify_op(
            &k64_mma!(operands(u32_value), SparseMmaMetadataAttr::Ordered),
            &ctx
        )
        .is_err()
    );

    let wrong_count = [vec![i32_value; 4], vec![u32_value; 8], vec![zero]].concat();
    assert!(
        verify_op(
            &k64_mma!(wrong_count, SparseMmaMetadataAttr::Standard),
            &ctx
        )
        .is_err()
    );

    let mut wrong_type = operands(zero);
    wrong_type[4] = signless_i32_value;
    assert!(verify_op(&k64_mma!(wrong_type, SparseMmaMetadataAttr::Standard), &ctx).is_err());

    macro_rules! int4_mma {
        ($operands:expr, $a:expr, $b:expr, $overflow:expr, $metadata:expr) => {{
            let operation = Operation::new(
                &mut ctx,
                SparseMmaOp::get_concrete_op_info(),
                vec![i32_ty.into(); 4],
                $operands,
                vec![],
                0,
            );
            let mma = SparseMmaOp::new(operation);
            mma.set_attr_nvvm_sparse_mma_shape(&ctx, SparseMmaShapeAttr::M16n8k64);
            mma.set_attr_nvvm_sparse_mma_accumulator(&ctx, SparseMmaAccumulatorAttr::S32);
            mma.set_attr_nvvm_sparse_mma_a_element(&ctx, $a);
            mma.set_attr_nvvm_sparse_mma_b_element(&ctx, $b);
            mma.set_attr_nvvm_sparse_mma_a_layout(&ctx, SparseMmaLayoutAttr::Row);
            mma.set_attr_nvvm_sparse_mma_b_layout(&ctx, SparseMmaLayoutAttr::Col);
            mma.set_attr_nvvm_sparse_mma_overflow(&ctx, $overflow);
            mma.set_attr_nvvm_sparse_mma_metadata(&ctx, $metadata);
            mma.set_attr_nvvm_sparse_mma_selector(&ctx, SparseMmaSelectorAttr::ImmediateZeroOrOne);
            mma
        }};
    }

    let int4_variants = [
        (
            SparseMmaElementAttr::S4,
            SparseMmaElementAttr::S4,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::S4,
            SparseMmaElementAttr::U4,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::U4,
            SparseMmaElementAttr::U4,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::U4,
            SparseMmaElementAttr::S4,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::S4,
            SparseMmaElementAttr::S4,
            SparseMmaOverflowAttr::Satfinite,
        ),
        (
            SparseMmaElementAttr::S4,
            SparseMmaElementAttr::U4,
            SparseMmaOverflowAttr::Satfinite,
        ),
        (
            SparseMmaElementAttr::U4,
            SparseMmaElementAttr::U4,
            SparseMmaOverflowAttr::Satfinite,
        ),
        (
            SparseMmaElementAttr::U4,
            SparseMmaElementAttr::S4,
            SparseMmaOverflowAttr::Satfinite,
        ),
    ];
    let int4_operands =
        |selector| [vec![i32_value; 4], vec![u32_value; 5], vec![selector]].concat();
    for metadata in [
        SparseMmaMetadataAttr::Standard,
        SparseMmaMetadataAttr::Ordered,
    ] {
        for (index, (a, b, overflow)) in int4_variants.iter().enumerate() {
            let selector = if index % 2 == 0 { zero } else { one };
            assert!(
                verify_op(
                    &int4_mma!(
                        int4_operands(selector),
                        a.clone(),
                        b.clone(),
                        overflow.clone(),
                        metadata.clone()
                    ),
                    &ctx,
                )
                .is_ok()
            );
        }
    }

    assert!(
        verify_op(
            &int4_mma!(
                int4_operands(zero),
                SparseMmaElementAttr::S4,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Standard
            ),
            &ctx,
        )
        .is_err()
    );
    for metadata in [
        SparseMmaMetadataAttr::Standard,
        SparseMmaMetadataAttr::Ordered,
    ] {
        for selector in [two, u32_value] {
            assert!(
                verify_op(
                    &int4_mma!(
                        int4_operands(selector),
                        SparseMmaElementAttr::S4,
                        SparseMmaElementAttr::U4,
                        SparseMmaOverflowAttr::Wrapping,
                        metadata.clone()
                    ),
                    &ctx,
                )
                .is_err()
            );
        }
    }
}

#[test]
fn generated_sparse_mma_m16n8k128_int4_verifies_metadata_selector_and_widths() {
    use pliron::builtin::{attributes::IntegerAttr, ops::ConstantOp};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into(), u32_ty.into()]);
    let i32_value = block.deref(&ctx).get_argument(0);
    let u32_value = block.deref(&ctx).get_argument(1);
    let integer = |value| {
        IntegerAttr::new(
            u32_ty,
            APInt::from_u32(value, NonZeroUsize::new(32).unwrap()),
        )
    };
    let zero = ConstantOp::new(&mut ctx, integer(0).into())
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let one = ConstantOp::new(&mut ctx, integer(1).into())
        .get_operation()
        .deref(&ctx)
        .get_result(0);

    macro_rules! k128_mma {
        ($operands:expr, $a:expr, $b:expr, $overflow:expr, $metadata:expr) => {{
            let operation = Operation::new(
                &mut ctx,
                SparseMmaOp::get_concrete_op_info(),
                vec![i32_ty.into(); 4],
                $operands,
                vec![],
                0,
            );
            let mma = SparseMmaOp::new(operation);
            mma.set_attr_nvvm_sparse_mma_shape(&ctx, SparseMmaShapeAttr::M16n8k128);
            mma.set_attr_nvvm_sparse_mma_accumulator(&ctx, SparseMmaAccumulatorAttr::S32);
            mma.set_attr_nvvm_sparse_mma_a_element(&ctx, $a);
            mma.set_attr_nvvm_sparse_mma_b_element(&ctx, $b);
            mma.set_attr_nvvm_sparse_mma_a_layout(&ctx, SparseMmaLayoutAttr::Row);
            mma.set_attr_nvvm_sparse_mma_b_layout(&ctx, SparseMmaLayoutAttr::Col);
            mma.set_attr_nvvm_sparse_mma_overflow(&ctx, $overflow);
            mma.set_attr_nvvm_sparse_mma_metadata(&ctx, $metadata);
            mma.set_attr_nvvm_sparse_mma_selector(&ctx, SparseMmaSelectorAttr::ImmediateZero);
            mma
        }};
    }

    let variants = [
        (
            SparseMmaElementAttr::S4,
            SparseMmaElementAttr::S4,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::S4,
            SparseMmaElementAttr::U4,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::U4,
            SparseMmaElementAttr::U4,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::U4,
            SparseMmaElementAttr::S4,
            SparseMmaOverflowAttr::Wrapping,
        ),
        (
            SparseMmaElementAttr::S4,
            SparseMmaElementAttr::S4,
            SparseMmaOverflowAttr::Satfinite,
        ),
        (
            SparseMmaElementAttr::S4,
            SparseMmaElementAttr::U4,
            SparseMmaOverflowAttr::Satfinite,
        ),
        (
            SparseMmaElementAttr::U4,
            SparseMmaElementAttr::U4,
            SparseMmaOverflowAttr::Satfinite,
        ),
        (
            SparseMmaElementAttr::U4,
            SparseMmaElementAttr::S4,
            SparseMmaOverflowAttr::Satfinite,
        ),
    ];
    let operands = |selector| [vec![i32_value; 4], vec![u32_value; 9], vec![selector]].concat();
    for metadata in [
        SparseMmaMetadataAttr::Standard,
        SparseMmaMetadataAttr::Ordered,
    ] {
        for (a, b, overflow) in &variants {
            assert!(
                verify_op(
                    &k128_mma!(
                        operands(zero),
                        a.clone(),
                        b.clone(),
                        overflow.clone(),
                        metadata.clone()
                    ),
                    &ctx,
                )
                .is_ok()
            );
        }
    }

    for metadata in [
        SparseMmaMetadataAttr::Standard,
        SparseMmaMetadataAttr::Ordered,
    ] {
        for selector in [one, u32_value] {
            assert!(
                verify_op(
                    &k128_mma!(
                        operands(selector),
                        SparseMmaElementAttr::S4,
                        SparseMmaElementAttr::U4,
                        SparseMmaOverflowAttr::Wrapping,
                        metadata.clone()
                    ),
                    &ctx,
                )
                .is_err()
            );
        }
    }
    assert!(
        verify_op(
            &k128_mma!(
                operands(zero),
                SparseMmaElementAttr::S4,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Standard
            ),
            &ctx,
        )
        .is_err()
    );
    assert!(
        verify_op(
            &k128_mma!(
                [vec![i32_value; 4], vec![u32_value; 8], vec![zero]].concat(),
                SparseMmaElementAttr::S4,
                SparseMmaElementAttr::S4,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Standard
            ),
            &ctx,
        )
        .is_err()
    );
}

#[test]
fn generated_sparse_mma_f8f6f4_verifies_all_formats_and_closed_shape() {
    use pliron::builtin::{attributes::IntegerAttr, ops::ConstantOp};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let f32_ty = FP32Type::get(&ctx);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let block = BasicBlock::new(&mut ctx, None, vec![f32_ty.into(), u32_ty.into()]);
    let f32_value = block.deref(&ctx).get_argument(0);
    let u32_value = block.deref(&ctx).get_argument(1);
    let integer = |value| {
        IntegerAttr::new(
            u32_ty,
            APInt::from_u32(value, NonZeroUsize::new(32).unwrap()),
        )
    };
    let zero = ConstantOp::new(&mut ctx, integer(0).into())
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let one = ConstantOp::new(&mut ctx, integer(1).into())
        .get_operation()
        .deref(&ctx)
        .get_result(0);

    macro_rules! f8f6f4_mma {
        ($operands:expr, $results:expr, $a:expr, $b:expr, $overflow:expr, $metadata:expr) => {{
            let operation = Operation::new(
                &mut ctx,
                SparseMmaOp::get_concrete_op_info(),
                $results,
                $operands,
                vec![],
                0,
            );
            let mma = SparseMmaOp::new(operation);
            mma.set_attr_nvvm_sparse_mma_shape(&ctx, SparseMmaShapeAttr::M16n8k64);
            mma.set_attr_nvvm_sparse_mma_accumulator(&ctx, SparseMmaAccumulatorAttr::F32);
            mma.set_attr_nvvm_sparse_mma_a_element(&ctx, $a);
            mma.set_attr_nvvm_sparse_mma_b_element(&ctx, $b);
            mma.set_attr_nvvm_sparse_mma_a_layout(&ctx, SparseMmaLayoutAttr::Row);
            mma.set_attr_nvvm_sparse_mma_b_layout(&ctx, SparseMmaLayoutAttr::Col);
            mma.set_attr_nvvm_sparse_mma_overflow(&ctx, $overflow);
            mma.set_attr_nvvm_sparse_mma_metadata(&ctx, $metadata);
            mma.set_attr_nvvm_sparse_mma_selector(&ctx, SparseMmaSelectorAttr::ImmediateZero);
            mma
        }};
    }

    let elements = [
        SparseMmaElementAttr::E2m1,
        SparseMmaElementAttr::E2m3,
        SparseMmaElementAttr::E3m2,
        SparseMmaElementAttr::E4m3,
        SparseMmaElementAttr::E5m2,
    ];
    let operands = |selector| [vec![f32_value; 4], vec![u32_value; 9], vec![selector]].concat();
    for a in &elements {
        for b in &elements {
            let mma = f8f6f4_mma!(
                operands(zero),
                vec![f32_ty.into(); 4],
                a.clone(),
                b.clone(),
                SparseMmaOverflowAttr::NotApplicable,
                SparseMmaMetadataAttr::Ordered
            );
            assert!(verify_op(&mma, &ctx).is_ok(), "rejected {a:?}x{b:?}");
        }
    }

    for invalid in [
        f8f6f4_mma!(
            operands(one),
            vec![f32_ty.into(); 4],
            SparseMmaElementAttr::E2m1,
            SparseMmaElementAttr::E2m1,
            SparseMmaOverflowAttr::NotApplicable,
            SparseMmaMetadataAttr::Ordered
        ),
        f8f6f4_mma!(
            operands(zero),
            vec![f32_ty.into(); 4],
            SparseMmaElementAttr::E2m1,
            SparseMmaElementAttr::E2m1,
            SparseMmaOverflowAttr::Wrapping,
            SparseMmaMetadataAttr::Ordered
        ),
        f8f6f4_mma!(
            operands(zero),
            vec![f32_ty.into(); 4],
            SparseMmaElementAttr::E2m1,
            SparseMmaElementAttr::E2m1,
            SparseMmaOverflowAttr::NotApplicable,
            SparseMmaMetadataAttr::Standard
        ),
        f8f6f4_mma!(
            operands(zero),
            vec![u32_ty.into(); 4],
            SparseMmaElementAttr::E2m1,
            SparseMmaElementAttr::E2m1,
            SparseMmaOverflowAttr::NotApplicable,
            SparseMmaMetadataAttr::Ordered
        ),
    ] {
        assert!(verify_op(&invalid, &ctx).is_err());
    }
}

#[test]
fn test_matrix_memory_ops_verify_pointer_and_packed_register_types() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let f32_ty = FP32Type::get(&ctx);
    let ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let load_ptr_ty = MirPtrType::get_generic(&mut ctx, u32_ty.into(), true);

    let load_block = BasicBlock::new(&mut ctx, None, vec![load_ptr_ty.into()]);
    let load_pointer = load_block.deref(&ctx).get_argument(0);
    let load = make_ldmatrix_x2(&mut ctx, load_pointer, vec![u32_ty.into(), u32_ty.into()]);
    assert!(load.verify(&ctx).is_ok());

    let bad_load_pointer_block = BasicBlock::new(&mut ctx, None, vec![i64_ty.into()]);
    let bad_pointer = bad_load_pointer_block.deref(&ctx).get_argument(0);
    let bad_load_pointer =
        make_ldmatrix_x2(&mut ctx, bad_pointer, vec![u32_ty.into(), u32_ty.into()]);
    assert!(bad_load_pointer.verify(&ctx).is_err());

    let bad_load_result =
        make_ldmatrix_x2(&mut ctx, load_pointer, vec![u32_ty.into(), f32_ty.into()]);
    assert!(bad_load_result.verify(&ctx).is_err());

    let store_block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            ptr_ty.into(),
            i32_ty.into(),
            i32_ty.into(),
            i32_ty.into(),
            i32_ty.into(),
        ],
    );
    let store_operands = (0..5)
        .map(|index| store_block.deref(&ctx).get_argument(index))
        .collect();
    let store = Operation::new(
        &mut ctx,
        StmatrixM8n8X4Op::get_concrete_op_info(),
        vec![],
        store_operands,
        vec![],
        0,
    );
    assert!(StmatrixM8n8X4Op::new(store).verify(&ctx).is_ok());

    let bad_store_block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            ptr_ty.into(),
            f32_ty.into(),
            i32_ty.into(),
            i32_ty.into(),
            i32_ty.into(),
        ],
    );
    let bad_store_operands = (0..5)
        .map(|index| bad_store_block.deref(&ctx).get_argument(index))
        .collect();
    let bad_store = Operation::new(
        &mut ctx,
        StmatrixM8n8X4Op::get_concrete_op_info(),
        vec![],
        bad_store_operands,
        vec![],
        0,
    );
    assert!(StmatrixM8n8X4Op::new(bad_store).verify(&ctx).is_err());
}

#[test]
fn test_packed_atomic_add_verifies_exact_raw_u32_shape() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let signless_i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let generic_u32_ptr = MirPtrType::get_generic(&mut ctx, u32_ty.into(), true);
    let global_u32_ptr = MirPtrType::get_global(&mut ctx, u32_ty.into(), true);
    let immutable_u32_ptr = MirPtrType::get_generic(&mut ctx, u32_ty.into(), false);
    let shared_u32_ptr = MirPtrType::get_shared(&mut ctx, u32_ty.into(), true);
    let generic_u64_ptr = MirPtrType::get_generic(&mut ctx, u64_ty.into(), true);

    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            generic_u32_ptr.into(),
            global_u32_ptr.into(),
            immutable_u32_ptr.into(),
            shared_u32_ptr.into(),
            generic_u64_ptr.into(),
            u32_ty.into(),
            signless_i32_ty.into(),
        ],
    );
    let generic_ptr = block.deref(&ctx).get_argument(0);
    let global_ptr = block.deref(&ctx).get_argument(1);
    let immutable_ptr = block.deref(&ctx).get_argument(2);
    let shared_ptr = block.deref(&ctx).get_argument(3);
    let wrong_pointee_ptr = block.deref(&ctx).get_argument(4);
    let addend = block.deref(&ctx).get_argument(5);
    let signless_addend = block.deref(&ctx).get_argument(6);

    macro_rules! check_variant {
        ($op:ty) => {{
            for pointer in [generic_ptr, global_ptr] {
                let valid = Operation::new(
                    &mut ctx,
                    <$op>::get_concrete_op_info(),
                    vec![u32_ty.into()],
                    vec![pointer, addend],
                    vec![],
                    0,
                );
                assert!(verify_op(&<$op>::new(valid), &ctx).is_ok());
            }

            for pointer in [immutable_ptr, shared_ptr, wrong_pointee_ptr] {
                let invalid = Operation::new(
                    &mut ctx,
                    <$op>::get_concrete_op_info(),
                    vec![u32_ty.into()],
                    vec![pointer, addend],
                    vec![],
                    0,
                );
                assert!(verify_op(&<$op>::new(invalid), &ctx).is_err());
            }

            let bad_address = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![u32_ty.into()],
                vec![addend, addend],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(bad_address), &ctx).is_err());

            let bad_addend = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![u32_ty.into()],
                vec![generic_ptr, signless_addend],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(bad_addend), &ctx).is_err());

            let bad_result = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![signless_i32_ty.into()],
                vec![generic_ptr, addend],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(bad_result), &ctx).is_err());

            let bad_counts = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![],
                vec![generic_ptr],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(bad_counts), &ctx).is_err());
        }};
    }

    check_variant!(NvvmAtomAddF16x2Op);
    check_variant!(NvvmAtomAddBf16x2Op);
}

#[test]
fn test_generated_packed_atomic_add_requires_closed_attributes_and_raw_u32_shape() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let signless_i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let signed_i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let global_u32_ptr = MirPtrType::get_global(&mut ctx, u32_ty.into(), true);
    let immutable_u32_ptr = MirPtrType::get_global(&mut ctx, u32_ty.into(), false);
    let shared_u32_ptr = MirPtrType::get_shared(&mut ctx, u32_ty.into(), true);
    let local_u32_ptr = MirPtrType::get(&mut ctx, u32_ty.into(), true, address_space::LOCAL);
    let constant_u32_ptr = MirPtrType::get_constant(&mut ctx, u32_ty.into(), true);
    let global_u64_ptr = MirPtrType::get_global(&mut ctx, u64_ty.into(), true);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            global_u32_ptr.into(),
            immutable_u32_ptr.into(),
            shared_u32_ptr.into(),
            local_u32_ptr.into(),
            constant_u32_ptr.into(),
            global_u64_ptr.into(),
            u32_ty.into(),
            signless_i32_ty.into(),
        ],
    );
    let global = block.deref(&ctx).get_argument(0);
    let immutable = block.deref(&ctx).get_argument(1);
    let shared = block.deref(&ctx).get_argument(2);
    let local = block.deref(&ctx).get_argument(3);
    let constant = block.deref(&ctx).get_argument(4);
    let wrong_pointee = block.deref(&ctx).get_argument(5);
    let addend = block.deref(&ctx).get_argument(6);
    let signless = block.deref(&ctx).get_argument(7);

    for format in [
        PackedAtomicFormatAttr::F16x2,
        PackedAtomicFormatAttr::Bf16x2,
    ] {
        let valid = PackedAtomicAddOp::build(&mut ctx, global, addend, format);
        assert!(verify_op(&PackedAtomicAddOp::new(valid), &ctx).is_ok());
    }

    for pointer in [immutable, shared, local, constant, wrong_pointee, addend] {
        let invalid =
            PackedAtomicAddOp::build(&mut ctx, pointer, addend, PackedAtomicFormatAttr::F16x2);
        assert!(verify_op(&PackedAtomicAddOp::new(invalid), &ctx).is_err());
    }
    let bad_addend = Operation::new(
        &mut ctx,
        PackedAtomicAddOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![global, signless],
        vec![],
        0,
    );
    assert!(verify_op(&PackedAtomicAddOp::new(bad_addend), &ctx).is_err());

    fn set_closed_attributes(ctx: &Context, op: pliron::context::Ptr<Operation>) {
        let packed = PackedAtomicAddOp::new(op);
        packed.set_attr_nvvm_packed_atomic_format(ctx, PackedAtomicFormatAttr::F16x2);
        packed.set_attr_nvvm_packed_atomic_state_space(ctx, PackedAtomicStateSpaceAttr::Global);
        packed.set_attr_nvvm_packed_atomic_ordering(ctx, PackedAtomicOrderingAttr::Relaxed);
        packed.set_attr_nvvm_packed_atomic_scope(ctx, PackedAtomicScopeAttr::Gpu);
        packed.set_attr_nvvm_packed_atomic_rounding(ctx, PackedAtomicRoundingAttr::Rn);
        packed.set_attr_nvvm_packed_atomic_subnormal(ctx, PackedAtomicSubnormalAttr::NoFtz);
        packed.set_attr_nvvm_packed_atomic_atomicity(ctx, PackedAtomicAtomicityAttr::PerElement);
    }

    for result_ty in [signless_i32_ty.into(), signed_i32_ty.into(), u64_ty.into()] {
        let bad_result = Operation::new(
            &mut ctx,
            PackedAtomicAddOp::get_concrete_op_info(),
            vec![result_ty],
            vec![global, addend],
            vec![],
            0,
        );
        set_closed_attributes(&ctx, bad_result);
        assert!(verify_op(&PackedAtomicAddOp::new(bad_result), &ctx).is_err());
    }

    for (results, operands) in [
        (vec![], vec![global, addend]),
        (vec![u32_ty.into(), u32_ty.into()], vec![global, addend]),
        (vec![u32_ty.into()], vec![global]),
        (vec![u32_ty.into()], vec![global, addend, addend]),
    ] {
        let bad_counts = Operation::new(
            &mut ctx,
            PackedAtomicAddOp::get_concrete_op_info(),
            results,
            operands,
            vec![],
            0,
        );
        set_closed_attributes(&ctx, bad_counts);
        assert!(verify_op(&PackedAtomicAddOp::new(bad_counts), &ctx).is_err());
    }

    // A structurally correct operation without the closed semantic attributes
    // must fail instead of inheriting implicit defaults.
    let missing_attributes = Operation::new(
        &mut ctx,
        PackedAtomicAddOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![global, addend],
        vec![],
        0,
    );
    assert!(verify_op(&PackedAtomicAddOp::new(missing_attributes), &ctx).is_err());
}

#[test]
fn test_thread_register_ops_verify_i32_results() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);

    let tid_x = Operation::new(
        &mut ctx,
        ReadPtxSregTidXOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(ReadPtxSregTidXOp::new(tid_x).verify(&ctx).is_ok());

    let lane_id = Operation::new(
        &mut ctx,
        ReadPtxSregLaneIdOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(ReadPtxSregLaneIdOp::new(lane_id).verify(&ctx).is_ok());
}

#[test]
fn test_thread_register_ops_reject_non_i32_results() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let op = Operation::new(
        &mut ctx,
        ReadPtxSregTidXOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![],
        vec![],
        0,
    );

    assert!(ReadPtxSregTidXOp::new(op).verify(&ctx).is_err());
}

#[test]
fn test_lanemask_ops_verify_i32_results() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);

    // Each lane-position mask is a zero-operand, single-i32-result sreg read.
    let lt = Operation::new(
        &mut ctx,
        ReadPtxSregLanemaskLtOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(ReadPtxSregLanemaskLtOp::new(lt).verify(&ctx).is_ok());

    let le = Operation::new(
        &mut ctx,
        ReadPtxSregLanemaskLeOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(ReadPtxSregLanemaskLeOp::new(le).verify(&ctx).is_ok());

    let eq = Operation::new(
        &mut ctx,
        ReadPtxSregLanemaskEqOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(ReadPtxSregLanemaskEqOp::new(eq).verify(&ctx).is_ok());

    let ge = Operation::new(
        &mut ctx,
        ReadPtxSregLanemaskGeOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(ReadPtxSregLanemaskGeOp::new(ge).verify(&ctx).is_ok());

    let gt = Operation::new(
        &mut ctx,
        ReadPtxSregLanemaskGtOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(ReadPtxSregLanemaskGtOp::new(gt).verify(&ctx).is_ok());
}

#[test]
fn test_lanemask_op_rejects_non_i32_result() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    // A 64-bit result must fail the shared lane-position mask verifier.
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let op = Operation::new(
        &mut ctx,
        ReadPtxSregLanemaskLtOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(ReadPtxSregLanemaskLtOp::new(op).verify(&ctx).is_err());
}

#[test]
fn test_generated_vote_sync_family_requires_exact_mask_predicate_and_result_types() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![i32_ty.into(), i1_ty.into(), i64_ty.into()],
    );
    let mask = block.deref(&ctx).get_argument(0);
    let predicate = block.deref(&ctx).get_argument(1);
    let wide_mask = block.deref(&ctx).get_argument(2);

    macro_rules! check_vote {
        ($op:ty, $result_ty:expr, $wrong_result_ty:expr) => {{
            let valid = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![$result_ty.into()],
                vec![mask, predicate],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(valid), &ctx).is_ok());

            for operands in [vec![], vec![mask], vec![mask, predicate, predicate]] {
                let wrong_arity = Operation::new(
                    &mut ctx,
                    <$op>::get_concrete_op_info(),
                    vec![$result_ty.into()],
                    operands,
                    vec![],
                    0,
                );
                assert!(verify_op(&<$op>::new(wrong_arity), &ctx).is_err());
            }

            for results in [vec![], vec![$result_ty.into(), $result_ty.into()]] {
                let wrong_arity = Operation::new(
                    &mut ctx,
                    <$op>::get_concrete_op_info(),
                    results,
                    vec![mask, predicate],
                    vec![],
                    0,
                );
                assert!(verify_op(&<$op>::new(wrong_arity), &ctx).is_err());
            }

            let wrong_mask = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![$result_ty.into()],
                vec![wide_mask, predicate],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(wrong_mask), &ctx).is_err());

            let wrong_predicate = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![$result_ty.into()],
                vec![mask, mask],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(wrong_predicate), &ctx).is_err());

            let wrong_result = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![$wrong_result_ty.into()],
                vec![mask, predicate],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(wrong_result), &ctx).is_err());
        }};
    }

    check_vote!(VoteSyncAllOp, i1_ty, u32_ty);
    check_vote!(VoteSyncAnyOp, i1_ty, u32_ty);
    check_vote!(VoteSyncBallotOp, u32_ty, i1_ty);
    check_vote!(VoteSyncUniOp, i1_ty, u32_ty);
}

#[test]
fn test_generated_active_mask_requires_no_operands_and_i32_result() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let unexpected_operand = block.deref(&ctx).get_argument(0);

    let valid = Operation::new(
        &mut ctx,
        ActiveMaskOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(verify_op(&ActiveMaskOp::new(valid), &ctx).is_ok());

    let wrong_operand_count = Operation::new(
        &mut ctx,
        ActiveMaskOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![unexpected_operand],
        vec![],
        0,
    );
    assert!(verify_op(&ActiveMaskOp::new(wrong_operand_count), &ctx).is_err());

    for results in [vec![], vec![i32_ty.into(), i32_ty.into()]] {
        let wrong_result_count = Operation::new(
            &mut ctx,
            ActiveMaskOp::get_concrete_op_info(),
            results,
            vec![],
            vec![],
            0,
        );
        assert!(verify_op(&ActiveMaskOp::new(wrong_result_count), &ctx).is_err());
    }

    let wrong_result_width = Operation::new(
        &mut ctx,
        ActiveMaskOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(verify_op(&ActiveMaskOp::new(wrong_result_width), &ctx).is_err());
}

#[test]
fn test_generated_match_family_requires_exact_mask_value_and_result_widths() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![i32_ty.into(), i32_ty.into(), i64_ty.into()],
    );
    let mask = block.deref(&ctx).get_argument(0);
    let value32 = block.deref(&ctx).get_argument(1);
    let value64 = block.deref(&ctx).get_argument(2);

    macro_rules! check_match {
        ($op:ty, $value:expr, $wrong_value:expr) => {{
            let valid = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![i32_ty.into()],
                vec![mask, $value],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(valid), &ctx).is_ok());

            for operands in [vec![], vec![mask], vec![mask, $value, $value]] {
                let wrong_operand_count = Operation::new(
                    &mut ctx,
                    <$op>::get_concrete_op_info(),
                    vec![i32_ty.into()],
                    operands,
                    vec![],
                    0,
                );
                assert!(verify_op(&<$op>::new(wrong_operand_count), &ctx).is_err());
            }

            for results in [vec![], vec![i32_ty.into(), i32_ty.into()]] {
                let wrong_result_count = Operation::new(
                    &mut ctx,
                    <$op>::get_concrete_op_info(),
                    results,
                    vec![mask, $value],
                    vec![],
                    0,
                );
                assert!(verify_op(&<$op>::new(wrong_result_count), &ctx).is_err());
            }

            let wrong_mask_width = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![i32_ty.into()],
                vec![value64, $value],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(wrong_mask_width), &ctx).is_err());

            let wrong_value_width = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![i32_ty.into()],
                vec![mask, $wrong_value],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(wrong_value_width), &ctx).is_err());

            let wrong_result_width = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![i64_ty.into()],
                vec![mask, $value],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(wrong_result_width), &ctx).is_err());
        }};
    }

    check_match!(MatchAnySyncI32Op, value32, value64);
    check_match!(MatchAllSyncI32Op, value32, value64);
    check_match!(MatchAnySyncI64Op, value64, value32);
    check_match!(MatchAllSyncI64Op, value64, value32);
}

#[test]
fn test_special_register_ops_verify_authoritative_widths() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);

    macro_rules! check_width {
        ($op:ty, $good:expr, $bad:expr) => {{
            let good = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![$good.into()],
                vec![],
                vec![],
                0,
            );
            assert!(
                verify_op(&<$op>::new(good), &ctx).is_ok(),
                "{} must accept its PTX register width",
                stringify!($op)
            );

            let bad = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![$bad.into()],
                vec![],
                vec![],
                0,
            );
            assert!(
                verify_op(&<$op>::new(bad), &ctx).is_err(),
                "{} must reject the other integer width",
                stringify!($op)
            );
        }};
    }

    check_width!(ReadPtxSregWarpIdOp, i32_ty, i64_ty);
    check_width!(ReadPtxSregNwarpIdOp, i32_ty, i64_ty);
    check_width!(ReadPtxSregSmIdOp, i32_ty, i64_ty);
    check_width!(ReadPtxSregNsmIdOp, i32_ty, i64_ty);
    check_width!(ReadPtxSregDynamicSmemSizeOp, i32_ty, i64_ty);
    check_width!(ReadPtxSregTotalSmemSizeOp, i32_ty, i64_ty);
    check_width!(ReadPtxSregGridIdOp, i64_ty, i32_ty);
}

#[test]
fn test_sync_ops_construct_and_verify() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let barrier = Operation::new(
        &mut ctx,
        Barrier0Op::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    assert!(Barrier0Op::new(barrier).verify(&ctx).is_ok());

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let unexpected_operand = block.deref(&ctx).get_argument(0);
    let bad_operand = Operation::new(
        &mut ctx,
        Barrier0Op::get_concrete_op_info(),
        vec![],
        vec![unexpected_operand],
        vec![],
        0,
    );
    assert!(verify_op(&Barrier0Op::new(bad_operand), &ctx).is_err());

    let bad_result = Operation::new(
        &mut ctx,
        Barrier0Op::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(verify_op(&Barrier0Op::new(bad_result), &ctx).is_err());

    let block_fence = Operation::new(
        &mut ctx,
        ThreadfenceBlockOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    assert!(ThreadfenceBlockOp::new(block_fence).verify(&ctx).is_ok());

    let device_fence = Operation::new(
        &mut ctx,
        ThreadfenceOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    assert!(ThreadfenceOp::new(device_fence).verify(&ctx).is_ok());

    let system_fence = Operation::new(
        &mut ctx,
        ThreadfenceSystemOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    assert!(ThreadfenceSystemOp::new(system_fence).verify(&ctx).is_ok());
}

#[test]
fn test_bf16x2_fma_constructs_and_verifies_three_operands() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);

    let a = Operation::new(
        &mut ctx,
        ReadPtxSregTidXOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    let b = Operation::new(
        &mut ctx,
        ReadPtxSregTidXOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );
    let c = Operation::new(
        &mut ctx,
        ReadPtxSregTidXOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![],
        vec![],
        0,
    );

    let operands = vec![
        a.deref(&ctx).get_result(0),
        b.deref(&ctx).get_result(0),
        c.deref(&ctx).get_result(0),
    ];

    let fma = Operation::new(
        &mut ctx,
        FmaBf16x2Op::get_concrete_op_info(),
        vec![u32_ty.into()],
        operands,
        vec![],
        0,
    );

    assert!(FmaBf16x2Op::new(fma).verify(&ctx).is_ok());
}

#[test]
fn test_generated_dot_products_require_three_i32_operands_and_one_i32_result() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![i32_ty.into(), i32_ty.into(), i32_ty.into(), i64_ty.into()],
    );
    let a = block.deref(&ctx).get_argument(0);
    let b = block.deref(&ctx).get_argument(1);
    let c = block.deref(&ctx).get_argument(2);
    let wide = block.deref(&ctx).get_argument(3);

    macro_rules! check_variant {
        ($op:ty) => {{
            let valid = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![i32_ty.into()],
                vec![a, b, c],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(valid), &ctx).is_ok());

            let wrong_width = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![i32_ty.into()],
                vec![a, b, wide],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(wrong_width), &ctx).is_err());

            for (results, operands) in [
                (vec![], vec![a, b, c]),
                (vec![i32_ty.into(), i32_ty.into()], vec![a, b, c]),
                (vec![i32_ty.into()], vec![a, b]),
                (vec![i32_ty.into()], vec![a, b, c, c]),
            ] {
                let wrong_count = Operation::new(
                    &mut ctx,
                    <$op>::get_concrete_op_info(),
                    results,
                    operands,
                    vec![],
                    0,
                );
                assert!(verify_op(&<$op>::new(wrong_count), &ctx).is_err());
            }
        }};
    }

    check_variant!(Dp4aS32Op);
    check_variant!(Dp4aU32Op);
    check_variant!(Dp2aS32Op);
    check_variant!(Dp2aU32Op);
}

#[test]
fn test_redux_sync_add_construct_and_verify() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);

    // A block supplies the two operands [mask, value].
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into(), i32_ty.into()]);
    let mask = block.deref(&ctx).get_argument(0);
    let value = block.deref(&ctx).get_argument(1);

    // Valid: 2 operands, 1 result (matches NOpdsInterface<2>/NResultsInterface<1>).
    let op = Operation::new(
        &mut ctx,
        ReduxSyncAddOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![mask, value],
        vec![],
        0,
    );
    assert!(verify_op(&ReduxSyncAddOp::new(op), &ctx).is_ok());

    // Invalid: wrong operand count (1 instead of 2) must fail verification.
    let bad_opnds = Operation::new(
        &mut ctx,
        ReduxSyncAddOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![mask],
        vec![],
        0,
    );
    assert!(verify_op(&ReduxSyncAddOp::new(bad_opnds), &ctx).is_err());

    // Invalid: wrong result count (0 instead of 1) must fail verification.
    let bad_results = Operation::new(
        &mut ctx,
        ReduxSyncAddOp::get_concrete_op_info(),
        vec![],
        vec![mask, value],
        vec![],
        0,
    );
    assert!(verify_op(&ReduxSyncAddOp::new(bad_results), &ctx).is_err());
}

#[test]
fn test_redux_sync_integer_family_construct_and_verify() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into(), i32_ty.into()]);
    let mask = block.deref(&ctx).get_argument(0);
    let value = block.deref(&ctx).get_argument(1);

    // Every integer-family variant has the same 2-operand/1-result shape. A
    // valid build of each must verify; a wrong operand count must not. The
    // `new` wrapper is invoked so each concrete op type is exercised.
    macro_rules! check_variant {
        ($op:ty) => {{
            let good = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![i32_ty.into()],
                vec![mask, value],
                vec![],
                0,
            );
            assert!(
                verify_op(&<$op>::new(good), &ctx).is_ok(),
                "{} should verify with [mask, value] -> i32",
                stringify!($op)
            );

            let bad = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![i32_ty.into()],
                vec![mask],
                vec![],
                0,
            );
            assert!(
                verify_op(&<$op>::new(bad), &ctx).is_err(),
                "{} must reject a single operand",
                stringify!($op)
            );
        }};
    }

    check_variant!(ReduxSyncUminOp);
    check_variant!(ReduxSyncMinOp);
    check_variant!(ReduxSyncUmaxOp);
    check_variant!(ReduxSyncMaxOp);
    check_variant!(ReduxSyncAndOp);
    check_variant!(ReduxSyncOrOp);
    check_variant!(ReduxSyncXorOp);
}

#[test]
fn test_bar_warp_sync_construct_and_verify() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into(), i64_ty.into()]);
    let mask = block.deref(&ctx).get_argument(0);
    let wrong_mask = block.deref(&ctx).get_argument(1);

    let valid = BarWarpSyncOp::build(&mut ctx, mask);
    assert!(verify_op(&BarWarpSyncOp::new(valid), &ctx).is_ok());

    let wrong_type = BarWarpSyncOp::build(&mut ctx, wrong_mask);
    assert!(verify_op(&BarWarpSyncOp::new(wrong_type), &ctx).is_err());

    let wrong_arity = Operation::new(
        &mut ctx,
        BarWarpSyncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    assert!(verify_op(&BarWarpSyncOp::new(wrong_arity), &ctx).is_err());
}

#[test]
fn test_elect_sync_construct_and_verify() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);

    // A block supplies the single `mask` operand.
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let mask = block.deref(&ctx).get_argument(0);

    // Valid: 1 operand [mask], 2 results [leader (i32), is_elected (i1)]
    // (matches NOpdsInterface<1>/NResultsInterface<2>).
    let op = Operation::new(
        &mut ctx,
        ElectSyncOp::get_concrete_op_info(),
        vec![i32_ty.into(), i1_ty.into()],
        vec![mask],
        vec![],
        0,
    );
    assert!(verify_op(&ElectSyncOp::new(op), &ctx).is_ok());

    // Invalid: wrong operand count (0 instead of 1) must fail verification.
    let bad_opnds = Operation::new(
        &mut ctx,
        ElectSyncOp::get_concrete_op_info(),
        vec![i32_ty.into(), i1_ty.into()],
        vec![],
        vec![],
        0,
    );
    assert!(verify_op(&ElectSyncOp::new(bad_opnds), &ctx).is_err());

    // Invalid: wrong result count (1 instead of 2) must fail verification.
    let bad_results = Operation::new(
        &mut ctx,
        ElectSyncOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![mask],
        vec![],
        0,
    );
    assert!(verify_op(&ElectSyncOp::new(bad_results), &ctx).is_err());
}

#[test]
fn test_shfl_sync_i64_construct_and_verify() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);

    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![i32_ty.into(), i64_ty.into(), i32_ty.into()],
    );
    let mask = block.deref(&ctx).get_argument(0);
    let value = block.deref(&ctx).get_argument(1);
    let lane = block.deref(&ctx).get_argument(2);

    macro_rules! check_mode {
        ($op:ty) => {{
            let valid = <$op>::build(&mut ctx, mask, value, lane);
            assert!(verify_op(&<$op>::new(valid), &ctx).is_ok());

            for (operands, result_ty) in [
                (vec![mask, value], i64_ty.into()),
                (vec![value, value, lane], i64_ty.into()),
                (vec![mask, mask, lane], i64_ty.into()),
                (vec![mask, value, value], i64_ty.into()),
                (vec![mask, value, lane], i32_ty.into()),
            ] {
                let invalid = Operation::new(
                    &mut ctx,
                    <$op>::get_concrete_op_info(),
                    vec![result_ty],
                    operands,
                    vec![],
                    0,
                );
                assert!(verify_op(&<$op>::new(invalid), &ctx).is_err());
            }

            let no_result = Operation::new(
                &mut ctx,
                <$op>::get_concrete_op_info(),
                vec![],
                vec![mask, value, lane],
                vec![],
                0,
            );
            assert!(verify_op(&<$op>::new(no_result), &ctx).is_err());
        }};
    }

    check_mode!(ShflSyncIdxI64Op);
    check_mode!(ShflSyncBflyI64Op);
    check_mode!(ShflSyncDownI64Op);
    check_mode!(ShflSyncUpI64Op);
}

#[test]
fn handwritten_atomic_carriers_reject_malformed_ir() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let f32_ty = FP32Type::get(&ctx);
    let generic_ptr = MirPtrType::get_generic(&mut ctx, u32_ty.into(), false);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            generic_ptr.into(),
            u32_ty.into(),
            i32_ty.into(),
            f32_ty.into(),
        ],
    );
    let pointer = block.deref(&ctx).get_argument(0);
    let u32_value = block.deref(&ctx).get_argument(1);
    let i32_value = block.deref(&ctx).get_argument(2);
    let f32_value = block.deref(&ctx).get_argument(3);

    assert!(
        NvvmAtomicLoadOp::build(
            &mut ctx,
            pointer,
            u32_ty.into(),
            AtomicOrdering::Acquire,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_ok()
    );
    assert!(
        NvvmAtomicStoreOp::build(
            &mut ctx,
            u32_value,
            pointer,
            AtomicOrdering::Release,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_ok()
    );
    assert!(
        NvvmAtomicRmwOp::build(
            &mut ctx,
            pointer,
            u32_value,
            u32_ty.into(),
            AtomicRmwKind::Add,
            AtomicOrdering::AcqRel,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_ok()
    );
    assert!(
        NvvmAtomicRmwOp::build(
            &mut ctx,
            pointer,
            f32_value,
            f32_ty.into(),
            AtomicRmwKind::FAdd,
            AtomicOrdering::Relaxed,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_ok()
    );
    assert!(
        NvvmAtomicCmpxchgOp::build(
            &mut ctx,
            pointer,
            u32_value,
            u32_value,
            u32_ty.into(),
            AtomicOrdering::Relaxed,
            AtomicOrdering::Acquire,
            AtomicScope::System,
        )
        .verify(&ctx)
        .is_ok()
    );

    for invalid in [
        NvvmAtomicLoadOp::build(
            &mut ctx,
            pointer,
            u32_ty.into(),
            AtomicOrdering::Release,
            AtomicScope::Device,
        ),
        NvvmAtomicLoadOp::build(
            &mut ctx,
            u32_value,
            u32_ty.into(),
            AtomicOrdering::Relaxed,
            AtomicScope::Device,
        ),
    ] {
        assert!(invalid.verify(&ctx).is_err());
    }
    assert!(
        NvvmAtomicStoreOp::build(
            &mut ctx,
            u32_value,
            pointer,
            AtomicOrdering::Acquire,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_err()
    );
    assert!(
        NvvmAtomicRmwOp::build(
            &mut ctx,
            pointer,
            u32_value,
            u64_ty.into(),
            AtomicRmwKind::Add,
            AtomicOrdering::Relaxed,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_err()
    );
    assert!(
        NvvmAtomicRmwOp::build(
            &mut ctx,
            pointer,
            u32_value,
            u32_ty.into(),
            AtomicRmwKind::FAdd,
            AtomicOrdering::Relaxed,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_err()
    );
    assert!(
        NvvmAtomicRmwOp::build(
            &mut ctx,
            pointer,
            i32_value,
            i32_ty.into(),
            AtomicRmwKind::UMin,
            AtomicOrdering::Relaxed,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_err()
    );
    assert!(
        NvvmAtomicCmpxchgOp::build(
            &mut ctx,
            pointer,
            f32_value,
            f32_value,
            f32_ty.into(),
            AtomicOrdering::SeqCst,
            AtomicOrdering::Relaxed,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_err()
    );
    assert!(
        NvvmAtomicCmpxchgOp::build(
            &mut ctx,
            pointer,
            u32_value,
            i32_value,
            u32_ty.into(),
            AtomicOrdering::SeqCst,
            AtomicOrdering::Relaxed,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_err()
    );
    assert!(
        NvvmAtomicCmpxchgOp::build(
            &mut ctx,
            pointer,
            u32_value,
            u32_value,
            u32_ty.into(),
            AtomicOrdering::SeqCst,
            AtomicOrdering::Release,
            AtomicScope::Device,
        )
        .verify(&ctx)
        .is_err()
    );

    let missing_attributes = Operation::new(
        &mut ctx,
        NvvmAtomicRmwOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![pointer, u32_value],
        vec![],
        0,
    );
    assert!(
        NvvmAtomicRmwOp::new(missing_attributes)
            .verify(&ctx)
            .is_err()
    );
    let bad_count = Operation::new(
        &mut ctx,
        NvvmAtomicCmpxchgOp::get_concrete_op_info(),
        vec![],
        vec![pointer, u32_value],
        vec![],
        0,
    );
    assert!(NvvmAtomicCmpxchgOp::new(bad_count).verify(&ctx).is_err());
}

#[test]
fn atomic_cmpxchg_accepts_exactly_llvm_ordering_pairs() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let pointer_ty = MirPtrType::get_generic(&mut ctx, u32_ty.into(), false);
    let block = BasicBlock::new(&mut ctx, None, vec![pointer_ty.into(), u32_ty.into()]);
    let pointer = block.deref(&ctx).get_argument(0);
    let value = block.deref(&ctx).get_argument(1);

    let orderings = [
        AtomicOrdering::Relaxed,
        AtomicOrdering::Acquire,
        AtomicOrdering::Release,
        AtomicOrdering::AcqRel,
        AtomicOrdering::SeqCst,
    ];
    for success in &orderings {
        for failure in &orderings {
            let expected = matches!(
                failure,
                AtomicOrdering::Relaxed | AtomicOrdering::Acquire | AtomicOrdering::SeqCst
            );
            let actual = NvvmAtomicCmpxchgOp::build(
                &mut ctx,
                pointer,
                value,
                value,
                u32_ty.into(),
                success.clone(),
                failure.clone(),
                AtomicScope::Device,
            )
            .verify(&ctx)
            .is_ok();
            assert_eq!(
                actual, expected,
                "unexpected cmpxchg ordering result for success={success:?}, failure={failure:?}"
            );
        }
    }
}

#[test]
fn handwritten_ffi_and_wgmma_carriers_verify_exact_shapes() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let pointer_ty = MirPtrType::get_generic(&mut ctx, u8_ty.into(), false);
    let global_pointer_ty = MirPtrType::get_global(&mut ctx, u8_ty.into(), false);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            pointer_ty.into(),
            global_pointer_ty.into(),
            u32_ty.into(),
            u64_ty.into(),
        ],
    );
    let pointer = block.deref(&ctx).get_argument(0);
    let global_pointer = block.deref(&ctx).get_argument(1);
    let u32_value = block.deref(&ctx).get_argument(2);
    let u64_value = block.deref(&ctx).get_argument(3);

    let vprintf = VprintfOp::build(&mut ctx, pointer, pointer);
    assert!(VprintfOp::new(vprintf).verify(&ctx).is_ok());
    let bad_vprintf = Operation::new(
        &mut ctx,
        VprintfOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![pointer, u32_value],
        vec![],
        0,
    );
    assert!(VprintfOp::new(bad_vprintf).verify(&ctx).is_err());

    let descriptor = Operation::new(
        &mut ctx,
        WgmmaMakeSmemDescOp::get_concrete_op_info(),
        vec![u64_ty.into()],
        vec![pointer],
        vec![],
        0,
    );
    assert!(WgmmaMakeSmemDescOp::new(descriptor).verify(&ctx).is_ok());
    for (operands, results) in [
        (vec![u32_value], vec![u64_ty.into()]),
        (vec![global_pointer], vec![u64_ty.into()]),
        (vec![pointer], vec![u32_ty.into()]),
        (vec![], vec![u64_ty.into()]),
    ] {
        let invalid = Operation::new(
            &mut ctx,
            WgmmaMakeSmemDescOp::get_concrete_op_info(),
            results,
            operands,
            vec![],
            0,
        );
        assert!(WgmmaMakeSmemDescOp::new(invalid).verify(&ctx).is_err());
    }

    let mma = Operation::new(
        &mut ctx,
        WgmmaMmaM64N64K16F32Bf16Op::get_concrete_op_info(),
        vec![],
        vec![pointer, u64_value, u64_value],
        vec![],
        0,
    );
    assert!(WgmmaMmaM64N64K16F32Bf16Op::new(mma).verify(&ctx).is_ok());
    for (operands, results) in [
        (vec![u32_value, u64_value, u64_value], vec![]),
        (vec![pointer, u32_value, u64_value], vec![]),
        (vec![pointer, u64_value], vec![]),
        (vec![pointer, u64_value, u64_value], vec![u32_ty.into()]),
    ] {
        let invalid = Operation::new(
            &mut ctx,
            WgmmaMmaM64N64K16F32Bf16Op::get_concrete_op_info(),
            results,
            operands,
            vec![],
            0,
        );
        assert!(
            WgmmaMmaM64N64K16F32Bf16Op::new(invalid)
                .verify(&ctx)
                .is_err()
        );
    }
}
