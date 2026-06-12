/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::{
    attributes::MirCastKindAttr,
    ops::{
        MirAddOp, MirAssertOp, MirAssignOp, MirCallOp, MirCastOp, MirCheckedAddOp, MirCmpOp,
        MirCondBranchOp, MirConstantOp, MirConstructSliceOp, MirDivOp, MirEqOp, MirExtractFieldOp,
        MirFuncOp, MirGeOp, MirGlobalAllocOp, MirGotoOp, MirGtOp, MirLeOp, MirLoadOp, MirLtOp,
        MirMulOp, MirNeOp, MirNegOp, MirNotOp, MirPtrOffsetOp, MirRemOp, MirReturnOp, MirStoreOp,
        MirSubOp,
    },
    types::{EnumVariant, MirEnumType, MirPtrType, MirSliceType, MirTupleType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, StringAttr, TypeAttr},
        op_interfaces::OperandSegmentInterface,
        types::{FP32Type, FunctionType, IntegerType, Signedness},
    },
    common_traits::Verify,
    context::Context,
    op::Op,
    operation::Operation,
    utils::apint::APInt,
};
use std::num::NonZeroUsize;

#[test]
fn test_mir_control_flow_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signed);
    let i1_ty = IntegerType::get(&mut ctx, 1, Signedness::Signless);

    // 1. MirGotoOp
    let target_block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let src_block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let arg_val = src_block.deref(&ctx).get_argument(0);

    let op = Operation::new(
        &mut ctx,
        MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![arg_val],
        vec![target_block],
        0,
    );
    let goto_op = MirGotoOp::new(op);
    assert!(goto_op.verify(&ctx).is_ok(), "Valid Goto");

    let op_bad = Operation::new(
        &mut ctx,
        MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![target_block],
        0,
    );
    assert!(
        MirGotoOp::new(op_bad).verify(&ctx).is_err(),
        "Goto missing operand"
    );

    // 2. MirCondBranchOp
    let true_block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let false_block = BasicBlock::new(&mut ctx, None, vec![]);
    let cond_block = BasicBlock::new(&mut ctx, None, vec![i1_ty.into(), i32_ty.into()]);
    let cond_val = cond_block.deref(&ctx).get_argument(0);
    let val = cond_block.deref(&ctx).get_argument(1);

    let (cond_flat, cond_sizes) =
        MirCondBranchOp::compute_segment_sizes(vec![vec![cond_val], vec![val], vec![]]);
    let op_cond = Operation::new(
        &mut ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        cond_flat,
        vec![true_block, false_block],
        0,
    );
    MirCondBranchOp::new(op_cond).set_operand_segment_sizes(&ctx, cond_sizes);
    let cond_br = MirCondBranchOp::new(op_cond);
    assert!(cond_br.verify(&ctx).is_ok(), "Valid CondBranch");

    let (cond_bad_flat, cond_bad_sizes) =
        MirCondBranchOp::compute_segment_sizes(vec![vec![cond_val], vec![], vec![]]);
    let op_cond_bad = Operation::new(
        &mut ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        cond_bad_flat,
        vec![true_block, false_block],
        0,
    );
    MirCondBranchOp::new(op_cond_bad).set_operand_segment_sizes(&ctx, cond_bad_sizes);
    assert!(
        MirCondBranchOp::new(op_cond_bad).verify(&ctx).is_err(),
        "CondBranch missing operand"
    );

    // 3. MirReturnOp
    let func_ty = FunctionType::get(&mut ctx, vec![], vec![i32_ty.into()]);
    let func_ty_attr = TypeAttr::new(func_ty.into());

    let func_op_ptr = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let mir_func = MirFuncOp::new(&mut ctx, func_op_ptr, func_ty_attr);
    let entry_block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let ret_val = entry_block.deref(&ctx).get_argument(0);

    let region = mir_func.get_operation().deref(&ctx).get_region(0);
    entry_block.insert_at_front(region, &ctx);

    let ret_op = Operation::new(
        &mut ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![ret_val],
        vec![],
        0,
    );
    ret_op.insert_at_back(entry_block, &ctx);

    let mir_ret = MirReturnOp::new(ret_op);
    assert!(mir_ret.verify(&ctx).is_ok(), "Valid Return");

    let f32_ty = FP32Type::get(&ctx);
    let f32_block = BasicBlock::new(&mut ctx, None, vec![f32_ty.into()]);
    let f32_val = f32_block.deref(&ctx).get_argument(0);

    let ret_op_bad = Operation::new(
        &mut ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![f32_val],
        vec![],
        0,
    );
    ret_op_bad.insert_at_back(entry_block, &ctx);

    let mir_ret_bad = MirReturnOp::new(ret_op_bad);
    assert!(mir_ret_bad.verify(&ctx).is_err(), "Return type mismatch");

    // 4. MirAssertOp
    let assert_succ = BasicBlock::new(&mut ctx, None, vec![]);

    let (assert_flat, assert_sizes) =
        MirAssertOp::compute_segment_sizes(vec![vec![cond_val], vec![]]);
    let op_assert = Operation::new(
        &mut ctx,
        MirAssertOp::get_concrete_op_info(),
        vec![],
        assert_flat,
        vec![assert_succ],
        0,
    );
    MirAssertOp::new(op_assert).set_operand_segment_sizes(&ctx, assert_sizes);
    let assert_op = MirAssertOp::new(op_assert);
    assert!(assert_op.verify(&ctx).is_ok(), "Valid Assert");

    let (assert_bad_flat, assert_bad_sizes) =
        MirAssertOp::compute_segment_sizes(vec![vec![val], vec![]]);
    let op_assert_bad = Operation::new(
        &mut ctx,
        MirAssertOp::get_concrete_op_info(),
        vec![],
        assert_bad_flat,
        vec![assert_succ],
        0,
    );
    MirAssertOp::new(op_assert_bad).set_operand_segment_sizes(&ctx, assert_bad_sizes);
    assert!(
        MirAssertOp::new(op_assert_bad).verify(&ctx).is_err(),
        "Assert cond type mismatch"
    );
}

#[test]
fn test_mir_load_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signed);
    let ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);

    let block = BasicBlock::new(&mut ctx, None, vec![ptr_ty.into()]);
    let ptr_val = block.deref(&ctx).get_argument(0);

    let op = Operation::new(
        &mut ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![ptr_val],
        vec![],
        0,
    );
    let mir_load = MirLoadOp::new(op);
    assert!(mir_load.verify(&ctx).is_ok(), "Valid MirLoadOp");

    let block_i32 = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let i32_val = block_i32.deref(&ctx).get_argument(0);

    let op_fail_operand = Operation::new(
        &mut ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![i32_val],
        vec![],
        0,
    );
    let mir_load_fail_operand = MirLoadOp::new(op_fail_operand);
    assert!(
        mir_load_fail_operand.verify(&ctx).is_err(),
        "MirLoadOp non-ptr operand"
    );

    let f32_ty = FP32Type::get(&ctx);
    let op_fail_res = Operation::new(
        &mut ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![f32_ty.into()],
        vec![ptr_val],
        vec![],
        0,
    );
    let mir_load_fail_res = MirLoadOp::new(op_fail_res);
    assert!(
        mir_load_fail_res.verify(&ctx).is_err(),
        "MirLoadOp result mismatch"
    );
}

#[test]
fn test_mir_ptr_offset_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signed);
    let ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let usize_ty = IntegerType::get(&mut ctx, 64, Signedness::Signless);

    let block = BasicBlock::new(&mut ctx, None, vec![ptr_ty.into(), usize_ty.into()]);
    let ptr_val = block.deref(&ctx).get_argument(0);
    let idx_val = block.deref(&ctx).get_argument(1);

    let op = Operation::new(
        &mut ctx,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![ptr_ty.into()],
        vec![ptr_val, idx_val],
        vec![],
        0,
    );
    let offset_op = MirPtrOffsetOp::new(op);
    assert!(offset_op.verify(&ctx).is_ok(), "Valid MirPtrOffsetOp");

    let block2 = BasicBlock::new(&mut ctx, None, vec![i32_ty.into(), usize_ty.into()]);
    let i32_val = block2.deref(&ctx).get_argument(0);
    let idx_val2 = block2.deref(&ctx).get_argument(1);

    let op_bad_base = Operation::new(
        &mut ctx,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![ptr_ty.into()],
        vec![i32_val, idx_val2],
        vec![],
        0,
    );
    assert!(MirPtrOffsetOp::new(op_bad_base).verify(&ctx).is_err());

    let f32_ty = FP32Type::get(&ctx);
    let ptr_f32_ty = MirPtrType::get_generic(&mut ctx, f32_ty.into(), false);
    let op_bad_res = Operation::new(
        &mut ctx,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![ptr_f32_ty.into()],
        vec![ptr_val, idx_val],
        vec![],
        0,
    );
    assert!(MirPtrOffsetOp::new(op_bad_res).verify(&ctx).is_err());
}

#[test]
fn test_mir_extract_field_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signed);
    let tuple_ty = MirTupleType::get(&mut ctx, vec![i32_ty.into(), i32_ty.into()]);

    let block = BasicBlock::new(&mut ctx, None, vec![tuple_ty.into()]);
    let tuple_val = block.deref(&ctx).get_argument(0);

    let op = Operation::new(
        &mut ctx,
        MirExtractFieldOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![tuple_val],
        vec![],
        0,
    );
    let extract_op = MirExtractFieldOp::new(op);
    extract_op.set_attr_index(&ctx, dialect_mir::attributes::FieldIndexAttr(0));
    assert!(extract_op.verify(&ctx).is_ok(), "Valid Tuple Extract");

    let op_oob = Operation::new(
        &mut ctx,
        MirExtractFieldOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![tuple_val],
        vec![],
        0,
    );
    let extract_op_oob = MirExtractFieldOp::new(op_oob);
    extract_op_oob.set_attr_index(&ctx, dialect_mir::attributes::FieldIndexAttr(2));
    assert!(extract_op_oob.verify(&ctx).is_err(), "OOB Index");
}

#[test]
fn test_mir_construct_slice_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let u8_ty = IntegerType::get(&mut ctx, 8, Signedness::Unsigned);
    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signed);
    let usize_ty = IntegerType::get(&mut ctx, 64, Signedness::Unsigned);
    let u8_ptr_ty = MirPtrType::get_generic(&mut ctx, u8_ty.into(), false);
    let u8_slice_ty = MirSliceType::get(&mut ctx, u8_ty.into());
    let i32_slice_ty = MirSliceType::get(&mut ctx, i32_ty.into());

    let block = BasicBlock::new(&mut ctx, None, vec![u8_ptr_ty.into(), usize_ty.into()]);
    let ptr_val = block.deref(&ctx).get_argument(0);
    let len_val = block.deref(&ctx).get_argument(1);

    // Valid: (ptr to u8, usize len) -> slice of u8
    let op = Operation::new(
        &mut ctx,
        MirConstructSliceOp::get_concrete_op_info(),
        vec![u8_slice_ty.into()],
        vec![ptr_val, len_val],
        vec![],
        0,
    );
    assert!(
        MirConstructSliceOp::new(op).verify(&ctx).is_ok(),
        "Valid slice construction"
    );

    // Invalid: data pointer pointee does not match slice element type
    let op_bad_elem = Operation::new(
        &mut ctx,
        MirConstructSliceOp::get_concrete_op_info(),
        vec![i32_slice_ty.into()],
        vec![ptr_val, len_val],
        vec![],
        0,
    );
    assert!(
        MirConstructSliceOp::new(op_bad_elem).verify(&ctx).is_err(),
        "Pointee/element mismatch"
    );

    // Invalid: operands swapped (length where the pointer should be)
    let op_swapped = Operation::new(
        &mut ctx,
        MirConstructSliceOp::get_concrete_op_info(),
        vec![u8_slice_ty.into()],
        vec![len_val, ptr_val],
        vec![],
        0,
    );
    assert!(
        MirConstructSliceOp::new(op_swapped).verify(&ctx).is_err(),
        "Swapped operands"
    );

    // Invalid: result is not a slice type
    let op_bad_res = Operation::new(
        &mut ctx,
        MirConstructSliceOp::get_concrete_op_info(),
        vec![u8_ptr_ty.into()],
        vec![ptr_val, len_val],
        vec![],
        0,
    );
    assert!(
        MirConstructSliceOp::new(op_bad_res).verify(&ctx).is_err(),
        "Non-slice result type"
    );
}

#[test]
fn test_mir_arithmetic_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signed);
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into(), i32_ty.into()]);
    let lhs = block.deref(&ctx).get_argument(0);

    let check_bin_op = |opid: (
        fn(pliron::context::Ptr<pliron::operation::Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
                        name: &str| {
        let mut context = Context::new();
        dialect_mir::register(&mut context);
        let ty = IntegerType::get(&mut context, 32, Signedness::Signed);
        let blk = BasicBlock::new(&mut context, None, vec![ty.into(), ty.into()]);
        let l = blk.deref(&context).get_argument(0);
        let r = blk.deref(&context).get_argument(1);

        let op = Operation::new(&mut context, opid, vec![ty.into()], vec![l, r], vec![], 0);
        assert!(op.verify(&context).is_ok(), "Valid {}", name);

        let f32_t = FP32Type::get(&context);
        let blk2 = BasicBlock::new(&mut context, None, vec![f32_t.into()]);
        let f32_val = blk2.deref(&context).get_argument(0);

        let op_bad = Operation::new(
            &mut context,
            opid,
            vec![ty.into()],
            vec![l, f32_val],
            vec![],
            0,
        );
        assert!(op_bad.verify(&context).is_err(), "Type mismatch {}", name);
    };

    check_bin_op(MirAddOp::get_concrete_op_info(), "Add");
    check_bin_op(MirSubOp::get_concrete_op_info(), "Sub");
    check_bin_op(MirMulOp::get_concrete_op_info(), "Mul");
    check_bin_op(MirDivOp::get_concrete_op_info(), "Div");
    check_bin_op(MirRemOp::get_concrete_op_info(), "Rem");

    let op_neg = Operation::new(
        &mut ctx,
        MirNegOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![lhs],
        vec![],
        0,
    );
    assert!(op_neg.verify(&ctx).is_ok(), "Valid Neg");

    let f32_ty = FP32Type::get(&ctx);
    let op_neg_bad = Operation::new(
        &mut ctx,
        MirNegOp::get_concrete_op_info(),
        vec![f32_ty.into()],
        vec![lhs],
        vec![],
        0,
    );
    assert!(op_neg_bad.verify(&ctx).is_err(), "Neg type mismatch");

    let op_not = Operation::new(
        &mut ctx,
        MirNotOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![lhs],
        vec![],
        0,
    );
    assert!(op_not.verify(&ctx).is_ok(), "Valid Not");
}

#[test]
fn test_mir_misc_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signed);
    let i64_ty = IntegerType::get(&mut ctx, 64, Signedness::Signed);
    let i1_ty = IntegerType::get(&mut ctx, 1, Signedness::Signless);

    // 1. MirConstantOp
    let i32_signless = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let width = NonZeroUsize::new(32).unwrap();
    let apint = APInt::from_u32(42, width);
    let int_attr = IntegerAttr::new(i32_signless, apint);

    let const_op_ptr = Operation::new(
        &mut ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![i32_signless.into()],
        vec![],
        vec![],
        0,
    );
    let const_op = MirConstantOp::new(const_op_ptr);
    const_op.set_attr_value(&ctx, int_attr);
    assert!(const_op.verify(&ctx).is_ok(), "Valid Constant");

    // Mismatch type
    let i64_signless = IntegerType::get(&mut ctx, 64, Signedness::Signless);
    let i64_width = NonZeroUsize::new(64).unwrap();
    let i64_attr = IntegerAttr::new(i64_signless, APInt::from_u64(42, i64_width));
    const_op.set_attr_value(&ctx, i64_attr);
    assert!(const_op.verify(&ctx).is_err(), "Constant type mismatch");

    // 2. MirCastOp
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let arg = block.deref(&ctx).get_argument(0);

    let cast_op = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![arg],
        vec![],
        0,
    );
    MirCastOp::new(cast_op).set_attr_cast_kind(&ctx, MirCastKindAttr::IntToInt);
    assert!(MirCastOp::new(cast_op).verify(&ctx).is_ok(), "Valid Cast");

    // 3. MirCheckedAddOp
    let tuple_ty = MirTupleType::get(&mut ctx, vec![i32_ty.into(), i1_ty.into()]);
    let block2 = BasicBlock::new(&mut ctx, None, vec![i32_ty.into(), i32_ty.into()]);
    let lhs = block2.deref(&ctx).get_argument(0);
    let rhs = block2.deref(&ctx).get_argument(1);

    let checked_add = Operation::new(
        &mut ctx,
        MirCheckedAddOp::get_concrete_op_info(),
        vec![tuple_ty.into()],
        vec![lhs, rhs],
        vec![],
        0,
    );
    assert!(
        MirCheckedAddOp::new(checked_add).verify(&ctx).is_ok(),
        "Valid CheckedAdd"
    );

    // Invalid result type (not tuple)
    let checked_add_bad = Operation::new(
        &mut ctx,
        MirCheckedAddOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![lhs, rhs],
        vec![],
        0,
    );
    assert!(
        MirCheckedAddOp::new(checked_add_bad).verify(&ctx).is_err(),
        "CheckedAdd bad result"
    );
}

#[test]
fn test_mir_comparison_verify() {
    let check_cmp = |opid: (
        fn(pliron::context::Ptr<pliron::operation::Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
                     name: &str| {
        let mut context = Context::new();
        dialect_mir::register(&mut context);
        let ty = IntegerType::get(&mut context, 32, Signedness::Signed);
        let res_ty = IntegerType::get(&mut context, 1, Signedness::Signless);
        let blk = BasicBlock::new(&mut context, None, vec![ty.into(), ty.into()]);
        let l = blk.deref(&context).get_argument(0);
        let r = blk.deref(&context).get_argument(1);

        let op = Operation::new(
            &mut context,
            opid,
            vec![res_ty.into()],
            vec![l, r],
            vec![],
            0,
        );
        assert!(op.verify(&context).is_ok(), "Valid {}", name);

        // Invalid operand types
        let f32_ty = FP32Type::get(&context);
        let blk2 = BasicBlock::new(&mut context, None, vec![f32_ty.into()]);
        let f32_val = blk2.deref(&context).get_argument(0);
        let op_bad = Operation::new(
            &mut context,
            opid,
            vec![res_ty.into()],
            vec![l, f32_val],
            vec![],
            0,
        );
        assert!(op_bad.verify(&context).is_err(), "Type mismatch {}", name);

        // Invalid result type
        let op_bad_res = Operation::new(
            &mut context,
            opid,
            vec![ty.into()], // i32 result instead of i1
            vec![l, r],
            vec![],
            0,
        );
        assert!(
            op_bad_res.verify(&context).is_err(),
            "Result type mismatch {}",
            name
        );
    };

    check_cmp(MirEqOp::get_concrete_op_info(), "Eq");
    check_cmp(MirNeOp::get_concrete_op_info(), "Ne");
    check_cmp(MirLtOp::get_concrete_op_info(), "Lt");
    check_cmp(MirLeOp::get_concrete_op_info(), "Le");
    check_cmp(MirGtOp::get_concrete_op_info(), "Gt");
    check_cmp(MirGeOp::get_concrete_op_info(), "Ge");

    let mut context = Context::new();
    dialect_mir::register(&mut context);
    let i8_ty = IntegerType::get(&mut context, 8, Signedness::Signed);
    let i32_ty = IntegerType::get(&mut context, 32, Signedness::Signed);
    let unit = |name: &str| EnumVariant::unit(name.to_string());
    let ordering_ty = MirEnumType::get(
        &mut context,
        "Ordering".to_string(),
        i8_ty.into(),
        vec![255, 0, 1],
        vec![unit("Less"), unit("Equal"), unit("Greater")],
    );
    let blk = BasicBlock::new(&mut context, None, vec![i32_ty.into(), i32_ty.into()]);
    let lhs = blk.deref(&context).get_argument(0);
    let rhs = blk.deref(&context).get_argument(1);
    let two_variant_ty = MirEnumType::get(
        &mut context,
        "Two".to_string(),
        i8_ty.into(),
        vec![0, 1],
        vec![unit("A"), unit("B")],
    );
    // Payload variants disqualify the Ordering shape.
    let payload_ty = MirEnumType::get(
        &mut context,
        "ThreeWithPayload".to_string(),
        i8_ty.into(),
        vec![0, 1, 2],
        vec![
            unit("A"),
            EnumVariant::new("B".to_string(), vec![i32_ty.into()]),
            unit("C"),
        ],
    );
    let mut check_cmp_result = |result_ty, valid| {
        let op = Operation::new(
            &mut context,
            MirCmpOp::get_concrete_op_info(),
            vec![result_ty],
            vec![lhs, rhs],
            vec![],
            0,
        );
        assert_eq!(op.verify(&context).is_ok(), valid);
    };
    check_cmp_result(ordering_ty.into(), true);
    check_cmp_result(i32_ty.into(), false);
    check_cmp_result(two_variant_ty.into(), false);
    check_cmp_result(payload_ty.into(), false);

    // Float operands are rejected: rustc never emits BinOp::Cmp on floats.
    let f32_ty = FP32Type::get(&context);
    let fblk = BasicBlock::new(&mut context, None, vec![f32_ty.into(), f32_ty.into()]);
    let flhs = fblk.deref(&context).get_argument(0);
    let frhs = fblk.deref(&context).get_argument(1);
    let float_cmp = Operation::new(
        &mut context,
        MirCmpOp::get_concrete_op_info(),
        vec![ordering_ty.into()],
        vec![flhs, frhs],
        vec![],
        0,
    );
    assert!(
        float_cmp.verify(&context).is_err(),
        "float mir.cmp must be rejected"
    );
}

#[test]
fn test_mir_func_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signed);
    let func_ty = FunctionType::get(&mut ctx, vec![i32_ty.into()], vec![]);
    let func_ty_attr = TypeAttr::new(func_ty.into());

    // Valid Function
    let op_ptr = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let mir_func = MirFuncOp::new(&mut ctx, op_ptr, func_ty_attr.clone());

    // Add entry block with correct argument
    let region = mir_func.get_operation().deref(&ctx).get_region(0);
    let entry_block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    entry_block.insert_at_front(region, &ctx);

    assert!(mir_func.verify(&ctx).is_ok(), "Valid MirFuncOp");

    // Invalid: Argument count mismatch
    let op_ptr2 = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let mir_func2 = MirFuncOp::new(&mut ctx, op_ptr2, func_ty_attr.clone());
    let region2 = mir_func2.get_operation().deref(&ctx).get_region(0);
    // Block with 0 args
    let entry_block2 = BasicBlock::new(&mut ctx, None, vec![]);
    entry_block2.insert_at_front(region2, &ctx);

    assert!(
        mir_func2.verify(&ctx).is_err(),
        "MirFuncOp argument count mismatch"
    );

    // Invalid: Argument type mismatch
    let op_ptr3 = Operation::new(
        &mut ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let mir_func3 = MirFuncOp::new(&mut ctx, op_ptr3, func_ty_attr);
    let region3 = mir_func3.get_operation().deref(&ctx).get_region(0);
    let f32_ty = FP32Type::get(&ctx);
    let entry_block3 = BasicBlock::new(&mut ctx, None, vec![f32_ty.into()]);
    entry_block3.insert_at_front(region3, &ctx);

    assert!(
        mir_func3.verify(&ctx).is_err(),
        "MirFuncOp argument type mismatch"
    );
}

#[test]
fn test_mir_assign_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signed);
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let val = block.deref(&ctx).get_argument(0);

    let op = Operation::new(
        &mut ctx,
        MirAssignOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![val],
        vec![],
        0,
    );
    assert!(
        MirAssignOp::new(op).verify(&ctx).is_ok(),
        "Valid MirAssignOp"
    );

    let f32_ty = FP32Type::get(&ctx);
    let op_bad = Operation::new(
        &mut ctx,
        MirAssignOp::get_concrete_op_info(),
        vec![f32_ty.into()],
        vec![val],
        vec![],
        0,
    );
    assert!(
        MirAssignOp::new(op_bad).verify(&ctx).is_err(),
        "MirAssignOp type mismatch"
    );
}

#[test]
fn test_mir_call_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let op = Operation::new(
        &mut ctx,
        MirCallOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    let call_op = MirCallOp::new(op);

    // Missing attribute
    assert!(call_op.verify(&ctx).is_err(), "MirCallOp missing attribute");

    // With attribute
    let name = StringAttr::new("my_func".to_string());
    call_op.set_attr_callee(&ctx, name);
    assert!(call_op.verify(&ctx).is_ok(), "Valid MirCallOp");
}

#[test]
fn test_mir_store_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signed);
    let ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let block = BasicBlock::new(&mut ctx, None, vec![ptr_ty.into(), i32_ty.into()]);
    let ptr_val = block.deref(&ctx).get_argument(0);
    let val = block.deref(&ctx).get_argument(1);

    let op = Operation::new(
        &mut ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![ptr_val, val],
        vec![],
        0,
    );
    assert!(MirStoreOp::new(op).verify(&ctx).is_ok(), "Valid MirStoreOp");

    // Invalid: store to non-ptr
    let op_bad_ptr = Operation::new(
        &mut ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![val, val],
        vec![],
        0,
    );
    assert!(
        MirStoreOp::new(op_bad_ptr).verify(&ctx).is_err(),
        "MirStoreOp non-ptr dest"
    );

    // Invalid: type mismatch
    let f32_ty = FP32Type::get(&ctx);
    let block2 = BasicBlock::new(&mut ctx, None, vec![f32_ty.into()]);
    let f32_val = block2.deref(&ctx).get_argument(0);
    let op_bad_type = Operation::new(
        &mut ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![ptr_val, f32_val],
        vec![],
        0,
    );
    assert!(
        MirStoreOp::new(op_bad_type).verify(&ctx).is_err(),
        "MirStoreOp type mismatch"
    );
}

#[test]
fn test_mir_global_alloc_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let f32_ty = FP32Type::get(&ctx);

    // Helper: build a MirGlobalAllocOp whose result pointer is in `ptr_ty`
    // address space, with valid attributes.
    let build = |ctx: &mut Context, ptr_ty: pliron::r#type::TypePtr<MirPtrType>| {
        let op = Operation::new(
            ctx,
            MirGlobalAllocOp::get_concrete_op_info(),
            vec![ptr_ty.into()],
            vec![],
            vec![],
            0,
        );
        let alloc = MirGlobalAllocOp::new(op);
        alloc.set_attr_global_type(ctx, TypeAttr::new(f32_ty.into()));
        alloc.set_attr_global_key(ctx, StringAttr::new("k".to_string()));
        alloc
    };

    // Global memory (addrspace 1) — the original allowed space.
    let ptr_global = MirPtrType::get_global(&mut ctx, f32_ty.into(), true);
    assert!(
        build(&mut ctx, ptr_global).verify(&ctx).is_ok(),
        "global addrspace accepted"
    );

    // Constant memory (addrspace 4) — added for `#[constant]` support.
    let ptr_const = MirPtrType::get_constant(&mut ctx, f32_ty.into(), true);
    assert!(
        build(&mut ctx, ptr_const).verify(&ctx).is_ok(),
        "constant addrspace accepted"
    );

    // Shared memory (addrspace 3) — must be rejected.
    let ptr_shared = MirPtrType::get_shared(&mut ctx, f32_ty.into(), true);
    assert!(
        build(&mut ctx, ptr_shared).verify(&ctx).is_err(),
        "shared addrspace rejected"
    );

    // Missing required attributes.
    let no_attrs = Operation::new(
        &mut ctx,
        MirGlobalAllocOp::get_concrete_op_info(),
        vec![ptr_global.into()],
        vec![],
        vec![],
        0,
    );
    assert!(
        MirGlobalAllocOp::new(no_attrs).verify(&ctx).is_err(),
        "missing attributes rejected"
    );
}
