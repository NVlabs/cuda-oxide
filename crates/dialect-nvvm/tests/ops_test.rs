/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_nvvm::ops::{
    Barrier0Op, FmaBf16x2Op, ReadPtxSregLaneIdOp, ReadPtxSregTidXOp, ReduxSyncAddOp,
    ReduxSyncAndOp, ReduxSyncMaxOp, ReduxSyncMinOp, ReduxSyncOrOp, ReduxSyncUmaxOp,
    ReduxSyncUminOp, ReduxSyncXorOp, ThreadfenceBlockOp, ThreadfenceOp, ThreadfenceSystemOp,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::types::{IntegerType, Signedness},
    common_traits::Verify,
    context::Context,
    op::{Op, verify_op},
    operation::Operation,
};

#[test]
fn test_thread_register_ops_verify_i32_results() {
    let mut ctx = Context::new();
    dialect_nvvm::register(&mut ctx);

    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);

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

    let i64_ty = IntegerType::get(&mut ctx, 64, Signedness::Signless);
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

    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let u32_ty = IntegerType::get(&mut ctx, 32, Signedness::Unsigned);

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

    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);

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

    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
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
