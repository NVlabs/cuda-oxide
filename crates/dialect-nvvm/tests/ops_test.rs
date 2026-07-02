/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::types::MirPtrType;
use dialect_nvvm::ops::{
    Barrier0Op, ElectSyncOp, FmaBf16x2Op, LdmatrixX2Op, MmaM16N8K16F32Bf16Op, MovmatrixTransB16Op,
    ReadPtxSregDynamicSmemSizeOp, ReadPtxSregGridIdOp, ReadPtxSregLaneIdOp,
    ReadPtxSregLanemaskEqOp, ReadPtxSregLanemaskGeOp, ReadPtxSregLanemaskGtOp,
    ReadPtxSregLanemaskLeOp, ReadPtxSregLanemaskLtOp, ReadPtxSregNsmIdOp, ReadPtxSregNwarpIdOp,
    ReadPtxSregSmIdOp, ReadPtxSregTidXOp, ReadPtxSregTotalSmemSizeOp, ReadPtxSregWarpIdOp,
    ReduxSyncAddOp, ReduxSyncAndOp, ReduxSyncMaxOp, ReduxSyncMinOp, ReduxSyncOrOp, ReduxSyncUmaxOp,
    ReduxSyncUminOp, ReduxSyncXorOp, ShflSyncBflyI64Op, ShflSyncDownI64Op, ShflSyncIdxI64Op,
    ShflSyncUpI64Op, StmatrixM8n8X4Op, ThreadfenceBlockOp, ThreadfenceOp, ThreadfenceSystemOp,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::types::{FP32Type, IntegerType, Signedness},
    common_traits::Verify,
    context::Context,
    op::{Op, verify_op},
    operation::Operation,
};

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

/// The `(constructor, TypeId)` pair returned by `get_concrete_op_info()`.
type OpInfo = (
    fn(pliron::context::Ptr<Operation>) -> pliron::op::OpObj,
    std::any::TypeId,
);

#[test]
fn test_matrix_memory_ops_verify_pointer_and_packed_register_types() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);

    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let f32_ty = FP32Type::get(&ctx);
    let ptr_ty = MirPtrType::get_generic(&mut ctx, i8_ty.into(), true);

    let load_block = BasicBlock::new(&mut ctx, None, vec![ptr_ty.into()]);
    let load_pointer = load_block.deref(&ctx).get_argument(0);
    let load = Operation::new(
        &mut ctx,
        LdmatrixX2Op::get_concrete_op_info(),
        vec![i32_ty.into(), i32_ty.into()],
        vec![load_pointer],
        vec![],
        0,
    );
    assert!(LdmatrixX2Op::new(load).verify(&ctx).is_ok());

    let bad_load_pointer_block = BasicBlock::new(&mut ctx, None, vec![i64_ty.into()]);
    let bad_pointer = bad_load_pointer_block.deref(&ctx).get_argument(0);
    let bad_load_pointer = Operation::new(
        &mut ctx,
        LdmatrixX2Op::get_concrete_op_info(),
        vec![i32_ty.into(), i32_ty.into()],
        vec![bad_pointer],
        vec![],
        0,
    );
    assert!(LdmatrixX2Op::new(bad_load_pointer).verify(&ctx).is_err());

    let bad_load_result = Operation::new(
        &mut ctx,
        LdmatrixX2Op::get_concrete_op_info(),
        vec![i32_ty.into(), f32_ty.into()],
        vec![load_pointer],
        vec![],
        0,
    );
    assert!(LdmatrixX2Op::new(bad_load_result).verify(&ctx).is_err());

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

    // A block supplies [mask (i32), value (i64), lane/delta (i32)].
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![i32_ty.into(), i64_ty.into(), i32_ty.into()],
    );
    let mask = block.deref(&ctx).get_argument(0);
    let value = block.deref(&ctx).get_argument(1);
    let lane = block.deref(&ctx).get_argument(2);

    // All four modes share the same shape: 3 operands [mask, value, lane], 1
    // i64 result (NOpdsInterface<3>/NResultsInterface<1>).
    let modes: [OpInfo; 4] = [
        ShflSyncIdxI64Op::get_concrete_op_info(),
        ShflSyncBflyI64Op::get_concrete_op_info(),
        ShflSyncDownI64Op::get_concrete_op_info(),
        ShflSyncUpI64Op::get_concrete_op_info(),
    ];

    for opid in modes {
        // Valid.
        let op = Operation::new(
            &mut ctx,
            opid,
            vec![i64_ty.into()],
            vec![mask, value, lane],
            vec![],
            0,
        );
        assert!(verify_op(&ShflSyncIdxI64Op::new(op), &ctx).is_ok());

        // Invalid: wrong operand count (2 instead of 3) must fail verification.
        let bad = Operation::new(
            &mut ctx,
            opid,
            vec![i64_ty.into()],
            vec![mask, value],
            vec![],
            0,
        );
        assert!(verify_op(&ShflSyncIdxI64Op::new(bad), &ctx).is_err());
    }
}
