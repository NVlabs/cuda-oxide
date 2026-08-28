/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::{
    attributes::{FieldIndexAttr, MirCastKindAttr, MirPointerKindAuthorityAttr, VariantIndexAttr},
    ops::{
        MirAddOp, MirAllocaOp, MirArrayElementAddrOp, MirAssertOp, MirAssignOp, MirCallOp,
        MirCastOp, MirCheckedAddOp, MirCmpOp, MirCondBranchOp, MirConstantOp,
        MirConstructDisjointSliceOp, MirConstructEnumOp, MirConstructSliceOp, MirDivOp,
        MirEnumPayloadOp, MirEqOp, MirExtractFieldOp, MirFieldAddrOp, MirFuncOp, MirGeOp,
        MirGetDiscriminantOp, MirGlobalAllocOp, MirGotoOp, MirGtOp, MirLeOp, MirLoadOp, MirLtOp,
        MirMulOp, MirNeOp, MirNegOp, MirNotOp, MirPtrOffsetOp, MirRefOp, MirRemOp, MirReturnOp,
        MirSetDiscriminantOp, MirSharedAllocOp, MirStoreOp, MirSubOp,
    },
    types::{
        EnumVariant, MirArrayType, MirDisjointSliceType, MirEnumType, MirPointerKind, MirPtrType,
        MirSliceType, MirStructType, MirTupleType, MirUnionType,
    },
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
    opts::mem2reg::{AllocInfo, PromotableOpInterface, PromotableOpKind},
    r#type::TypeHandle,
    utils::apint::APInt,
};
use std::num::NonZeroUsize;

#[test]
fn test_mir_control_flow_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);

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
    let func_ty = FunctionType::get(&ctx, vec![], vec![i32_ty.into()]);
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
fn test_mir_pointer_kind_distinguishes_references_from_raw_pointers() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let pointee: pliron::r#type::TypeHandle = i32_ty.into();

    let shared_ref =
        MirPtrType::get_generic_with_kind(&mut ctx, pointee, false, MirPointerKind::SharedRef);
    let raw_const =
        MirPtrType::get_generic_with_kind(&mut ctx, pointee, false, MirPointerKind::RawConst);
    let unique_ref =
        MirPtrType::get_generic_with_kind(&mut ctx, pointee, true, MirPointerKind::UniqueRef);
    let raw_mut =
        MirPtrType::get_generic_with_kind(&mut ctx, pointee, true, MirPointerKind::RawMut);

    assert_ne!(
        shared_ref, raw_const,
        "&T must remain distinct from *const T"
    );
    assert_ne!(
        unique_ref, raw_mut,
        "&mut T must remain distinct from *mut T"
    );
    assert_ne!(
        shared_ref, unique_ref,
        "&T must remain distinct from &mut T"
    );
    assert_ne!(
        raw_const, raw_mut,
        "*const T must remain distinct from *mut T"
    );

    assert_eq!(
        shared_ref.deref(&ctx).pointer_kind(),
        MirPointerKind::SharedRef
    );
    assert_eq!(
        unique_ref.deref(&ctx).pointer_kind(),
        MirPointerKind::UniqueRef
    );
    assert_eq!(
        raw_const.deref(&ctx).pointer_kind(),
        MirPointerKind::RawConst
    );
    assert_eq!(raw_mut.deref(&ctx).pointer_kind(), MirPointerKind::RawMut);
}

#[test]
fn test_mir_slice_preserves_pointer_kind() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let element: pliron::r#type::TypeHandle = u8_ty.into();

    let shared = MirSliceType::get_with_kind(&mut ctx, element, MirPointerKind::SharedRef);
    let raw = MirSliceType::get_with_kind(&mut ctx, element, MirPointerKind::RawConst);

    assert_ne!(shared, raw, "&[T] must remain distinct from *const [T]");
    assert_eq!(shared.deref(&ctx).pointer_kind(), MirPointerKind::SharedRef);
    assert_eq!(raw.deref(&ctx).pointer_kind(), MirPointerKind::RawConst);
}

#[test]
fn test_mir_pointer_kind_mutability_consistency_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let invalid_shared = MirPtrType {
        pointee: i32_ty.into(),
        is_mutable: true,
        address_space: 0,
        kind: MirPointerKind::SharedRef,
    };
    let invalid_raw_mut = MirPtrType {
        pointee: i32_ty.into(),
        is_mutable: false,
        address_space: 0,
        kind: MirPointerKind::RawMut,
    };

    assert!(invalid_shared.verify(&ctx).is_err());
    assert!(invalid_raw_mut.verify(&ctx).is_err());

    let invalid_shared_slice = MirSliceType {
        element_ty: i32_ty.into(),
        is_mutable: true,
        kind: MirPointerKind::SharedRef,
    };
    let valid_erased_mut_slice = MirSliceType {
        element_ty: i32_ty.into(),
        is_mutable: true,
        kind: MirPointerKind::Erased,
    };
    assert!(invalid_shared_slice.verify(&ctx).is_err());
    assert!(valid_erased_mut_slice.verify(&ctx).is_ok());
}

#[test]
fn test_alloca_cannot_claim_a_rust_pointer_kind() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let erased = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let erased_alloca = Operation::new(
        &mut ctx,
        MirAllocaOp::get_concrete_op_info(),
        vec![erased.into()],
        vec![],
        vec![],
        0,
    );
    assert!(MirAllocaOp::new(erased_alloca).verify(&ctx).is_ok());

    let immutable_erased = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let immutable_alloca = Operation::new(
        &mut ctx,
        MirAllocaOp::get_concrete_op_info(),
        vec![immutable_erased.into()],
        vec![],
        vec![],
        0,
    );
    assert!(
        MirAllocaOp::new(immutable_alloca).verify(&ctx).is_err(),
        "an alloca cannot masquerade as the immutable canonical function-pointer carrier"
    );

    let shared_erased = MirPtrType::get_shared(&mut ctx, i32_ty.into(), true);
    let shared_alloca = Operation::new(
        &mut ctx,
        MirAllocaOp::get_concrete_op_info(),
        vec![shared_erased.into()],
        vec![],
        vec![],
        0,
    );
    assert!(
        MirAllocaOp::new(shared_alloca).verify(&ctx).is_err(),
        "a stack allocation must remain in generic address space"
    );

    let unique =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::UniqueRef);
    let unique_alloca = Operation::new(
        &mut ctx,
        MirAllocaOp::get_concrete_op_info(),
        vec![unique.into()],
        vec![],
        vec![],
        0,
    );
    assert!(
        MirAllocaOp::new(unique_alloca).verify(&ctx).is_err(),
        "compiler storage must not manufacture UniqueRef"
    );
}

#[test]
fn test_shared_alloc_result_pointee_must_match_element_type() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Signless).into();
    let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signless).into();
    let usize_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let mismatched_result = MirPtrType::get_shared(&mut ctx, i64_ty, true);
    let op = Operation::new(
        &mut ctx,
        MirSharedAllocOp::get_concrete_op_info(),
        vec![mismatched_result.into()],
        vec![],
        vec![],
        0,
    );
    let alloc = MirSharedAllocOp::new(op);
    alloc.set_attr_elem_type(&ctx, TypeAttr::new(i8_ty));
    alloc.set_attr_size(
        &ctx,
        IntegerAttr::new(usize_ty, APInt::from_u64(1, NonZeroUsize::new(64).unwrap())),
    );
    alloc.set_attr_alloc_key(&ctx, StringAttr::new("mismatched-shared".to_string()));

    let error = alloc
        .verify(&ctx)
        .expect_err("shared storage cannot claim an unrelated result pointee type");
    assert!(
        error
            .to_string()
            .contains("pointee type must match elem_type"),
        "{error}"
    );
}

#[test]
fn test_mir_load_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
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
fn test_mir_load_volatile_is_not_promotable() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
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
    let alloc_info = AllocInfo {
        ptr: ptr_val,
        ty: i32_ty.into(),
    };

    assert!(!mir_load.is_volatile(&ctx));
    assert!(matches!(
        mir_load.promotion_kind(&ctx, &alloc_info),
        PromotableOpKind::Load
    ));

    mir_load.set_volatile(&mut ctx, true);

    assert!(mir_load.is_volatile(&ctx));
    assert!(matches!(
        mir_load.promotion_kind(&ctx, &alloc_info),
        PromotableOpKind::NonPromotableUse
    ));
}

#[test]
fn test_mir_ptr_offset_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let usize_ty = IntegerType::get(&ctx, 64, Signedness::Signless);

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
    assert!(
        offset_op.is_inbounds(&ctx),
        "ordinary pointer offsets default to inbounds"
    );
    offset_op.set_inbounds(&mut ctx, false);
    assert!(
        !offset_op.is_inbounds(&ctx),
        "wrapping pointer offsets retain their explicit semantics"
    );

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
fn test_pointer_kind_laundering_requires_explicit_authority() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let usize_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let raw_mut_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::RawMut);
    let erased_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let erased_read_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let unique_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::UniqueRef);
    let shared_ty = MirPtrType::get_generic_with_kind(
        &mut ctx,
        i32_ty.into(),
        false,
        MirPointerKind::SharedRef,
    );
    let raw_const_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), false, MirPointerKind::RawConst);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            raw_mut_ty.into(),
            erased_ty.into(),
            usize_ty.into(),
            shared_ty.into(),
            raw_const_ty.into(),
            erased_read_ty.into(),
        ],
    );
    let raw_mut = block.deref(&ctx).get_argument(0);
    let erased = block.deref(&ctx).get_argument(1);
    let offset = block.deref(&ctx).get_argument(2);
    let shared = block.deref(&ctx).get_argument(3);
    let raw_const = block.deref(&ctx).get_argument(4);
    let erased_read = block.deref(&ctx).get_argument(5);

    // The concrete laundering example: pointer arithmetic is not a Rust
    // reborrow and cannot manufacture `&mut T` from `*mut T`.
    let raw_offset_to_unique = Operation::new(
        &mut ctx,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![raw_mut, offset],
        vec![],
        0,
    );
    assert!(
        MirPtrOffsetOp::new(raw_offset_to_unique)
            .verify(&ctx)
            .is_err(),
        "ptr_offset must not invent UniqueRef from RawMut"
    );

    let erased_offset_to_unique = Operation::new(
        &mut ctx,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![erased, offset],
        vec![],
        0,
    );
    assert!(
        MirPtrOffsetOp::new(erased_offset_to_unique)
            .verify(&ctx)
            .is_err(),
        "ptr_offset must not recover UniqueRef from Erased"
    );

    let preserving_offset = Operation::new(
        &mut ctx,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![raw_mut_ty.into()],
        vec![raw_mut, offset],
        vec![],
        0,
    );
    assert!(MirPtrOffsetOp::new(preserving_offset).verify(&ctx).is_ok());

    let erasing_offset = Operation::new(
        &mut ctx,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![erased_ty.into()],
        vec![raw_mut, offset],
        vec![],
        0,
    );
    assert!(MirPtrOffsetOp::new(erasing_offset).verify(&ctx).is_ok());

    let mutability_launder = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![erased_ty.into()],
        vec![erased_read],
        vec![],
        0,
    );
    let mutability_launder_cast = MirCastOp::new(mutability_launder);
    mutability_launder_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    assert!(
        mutability_launder_cast.verify(&ctx).is_err(),
        "an unmarked cast cannot manufacture writable Erased evidence"
    );

    let reborrow = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![raw_mut],
        vec![],
        0,
    );
    let reborrow_cast = MirCastOp::new(reborrow);
    reborrow_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    assert!(
        reborrow_cast.verify(&ctx).is_err(),
        "an unmarked cast must not manufacture UniqueRef"
    );
    reborrow_cast.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    assert!(
        reborrow_cast.verify(&ctx).is_ok(),
        "a rustc-declared reborrow is the explicit authority"
    );

    for (target, authority) in [
        (unique_ty.into(), MirPointerKindAuthorityAttr::Reborrow),
        (raw_mut_ty.into(), MirPointerKindAuthorityAttr::RawAddress),
    ] {
        let mutable_storage_boundary = Operation::new(
            &mut ctx,
            MirCastOp::get_concrete_op_info(),
            vec![target],
            vec![erased],
            vec![],
            0,
        );
        let mutable_storage_boundary_cast = MirCastOp::new(mutable_storage_boundary);
        mutable_storage_boundary_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
        mutable_storage_boundary_cast.set_pointer_kind_authority(&mut ctx, authority);
        assert!(
            mutable_storage_boundary_cast.verify(&ctx).is_ok(),
            "mutable compiler storage is a valid source for a mutable Rust boundary"
        );
    }

    let raw_to_shared_static = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![shared_ty.into()],
        vec![raw_const],
        vec![],
        0,
    );
    let raw_to_shared_static_cast = MirCastOp::new(raw_to_shared_static);
    raw_to_shared_static_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    raw_to_shared_static_cast
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);
    assert!(
        raw_to_shared_static_cast.verify(&ctx).is_err(),
        "StaticAddress may establish a typed static value only from Erased storage, not relabel an arbitrary raw pointer as SharedRef"
    );

    let raw_to_erased = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![erased_read_ty.into()],
        vec![raw_const],
        vec![],
        0,
    );
    let raw_to_erased_cast = MirCastOp::new(raw_to_erased);
    raw_to_erased_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    assert!(raw_to_erased_cast.verify(&ctx).is_ok());
    let erased_from_raw = raw_to_erased.deref(&ctx).get_result(0);
    let laundered_static = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![shared_ty.into()],
        vec![erased_from_raw],
        vec![],
        0,
    );
    let laundered_static_cast = MirCastOp::new(laundered_static);
    laundered_static_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    laundered_static_cast
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);
    assert!(
        laundered_static_cast.verify(&ctx).is_err(),
        "erasing RawConst must not make it valid StaticAddress storage"
    );

    let raw_mut_to_erased = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![erased_ty.into()],
        vec![raw_mut],
        vec![],
        0,
    );
    let raw_mut_to_erased_cast = MirCastOp::new(raw_mut_to_erased);
    raw_mut_to_erased_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    assert!(raw_mut_to_erased_cast.verify(&ctx).is_ok());
    let erased_from_raw_mut = raw_mut_to_erased.deref(&ctx).get_result(0);
    let laundered_abi = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![erased_from_raw_mut],
        vec![],
        0,
    );
    let laundered_abi_cast = MirCastOp::new(laundered_abi);
    laundered_abi_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    laundered_abi_cast
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::AbiBoundary);
    assert!(
        laundered_abi_cast.verify(&ctx).is_err(),
        "erasing RawMut must not let AbiBoundary manufacture UniqueRef"
    );

    let global_erased_ty = MirPtrType::get_with_kind(
        &mut ctx,
        i32_ty.into(),
        false,
        dialect_mir::types::address_space::GLOBAL,
        MirPointerKind::Erased,
    );
    let global_op = Operation::new(
        &mut ctx,
        MirGlobalAllocOp::get_concrete_op_info(),
        vec![global_erased_ty.into()],
        vec![],
        vec![],
        0,
    );
    let global = MirGlobalAllocOp::new(global_op);
    global.set_attr_global_type(&ctx, TypeAttr::new(i32_ty.into()));
    global.set_attr_global_key(&ctx, StringAttr::new("lineage-global".to_string()));
    assert!(global.verify(&ctx).is_ok());
    let global_storage = global_op.deref(&ctx).get_result(0);
    let typed_global = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![shared_ty.into()],
        vec![global_storage],
        vec![],
        0,
    );
    let typed_global_cast = MirCastOp::new(typed_global);
    typed_global_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    typed_global_cast
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);
    assert!(
        typed_global_cast.verify(&ctx).is_ok(),
        "StaticAddress accepts a verified global-allocation root"
    );

    let shared_erased_ty = MirPtrType::get_shared(&mut ctx, i32_ty.into(), true);
    let shared_op = Operation::new(
        &mut ctx,
        MirSharedAllocOp::get_concrete_op_info(),
        vec![shared_erased_ty.into()],
        vec![],
        vec![],
        0,
    );
    let shared_alloc = MirSharedAllocOp::new(shared_op);
    shared_alloc.set_attr_elem_type(&ctx, TypeAttr::new(i32_ty.into()));
    shared_alloc.set_attr_size(
        &ctx,
        IntegerAttr::new(usize_ty, APInt::from_u64(1, NonZeroUsize::new(64).unwrap())),
    );
    shared_alloc.set_attr_alloc_key(&ctx, StringAttr::new("lineage-shared".to_string()));
    assert!(shared_alloc.verify(&ctx).is_ok());
    let shared_storage = shared_op.deref(&ctx).get_result(0);
    let typed_shared = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![raw_mut_ty.into()],
        vec![shared_storage],
        vec![],
        0,
    );
    let typed_shared_cast = MirCastOp::new(typed_shared);
    typed_shared_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    typed_shared_cast
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);
    assert!(
        typed_shared_cast.verify(&ctx).is_ok(),
        "StaticAddress accepts a verified shared-allocation root"
    );

    let alloca_op = Operation::new(
        &mut ctx,
        MirAllocaOp::get_concrete_op_info(),
        vec![erased_ty.into()],
        vec![],
        vec![],
        0,
    );
    let alloca = MirAllocaOp::new(alloca_op);
    assert!(alloca.verify(&ctx).is_ok());
    let compiler_storage = alloca_op.deref(&ctx).get_result(0);
    let typed_abi = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![raw_mut_ty.into()],
        vec![compiler_storage],
        vec![],
        0,
    );
    let typed_abi_cast = MirCastOp::new(typed_abi);
    typed_abi_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    typed_abi_cast.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::AbiBoundary);
    assert!(
        typed_abi_cast.verify(&ctx).is_ok(),
        "AbiBoundary accepts verified compiler-owned alloca storage"
    );

    for (target, authority) in [
        (unique_ty.into(), MirPointerKindAuthorityAttr::Reborrow),
        (raw_mut_ty.into(), MirPointerKindAuthorityAttr::RawAddress),
        (
            raw_mut_ty.into(),
            MirPointerKindAuthorityAttr::StaticAddress,
        ),
        (raw_mut_ty.into(), MirPointerKindAuthorityAttr::AbiBoundary),
    ] {
        let immutable_storage_boundary = Operation::new(
            &mut ctx,
            MirCastOp::get_concrete_op_info(),
            vec![target],
            vec![erased_read],
            vec![],
            0,
        );
        let immutable_storage_boundary_cast = MirCastOp::new(immutable_storage_boundary);
        immutable_storage_boundary_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
        immutable_storage_boundary_cast.set_pointer_kind_authority(&mut ctx, authority);
        assert!(
            immutable_storage_boundary_cast.verify(&ctx).is_err(),
            "an immutable Erased thin pointer cannot establish a mutable Rust pointer kind"
        );
    }

    let wrong_authority = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![erased],
        vec![],
        0,
    );
    let wrong_authority_cast = MirCastOp::new(wrong_authority);
    wrong_authority_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    wrong_authority_cast
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::RawAddress);
    assert!(
        wrong_authority_cast.verify(&ctx).is_err(),
        "RawAddress authority cannot establish a reference kind"
    );

    let inline_asm_authority = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![erased],
        vec![],
        0,
    );
    let inline_asm_authority_cast = MirCastOp::new(inline_asm_authority);
    inline_asm_authority_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    inline_asm_authority_cast
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::InlineAsm);
    assert!(
        inline_asm_authority_cast.verify(&ctx).is_err(),
        "InlineAsm is a producer authority and must never authorize a cast"
    );

    let integer_reborrow = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![offset],
        vec![],
        0,
    );
    let integer_reborrow_cast = MirCastOp::new(integer_reborrow);
    integer_reborrow_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    integer_reborrow_cast
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    assert!(
        integer_reborrow_cast.verify(&ctx).is_err(),
        "Reborrow authority cannot reinterpret an integer as a Rust reference"
    );
    integer_reborrow_cast
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::RustCast);
    assert!(
        integer_reborrow_cast.verify(&ctx).is_err(),
        "RustCast authority cannot make an integer a PtrToPtr operand"
    );
    integer_reborrow_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::FnPtrToPtr);
    assert!(
        integer_reborrow_cast.verify(&ctx).is_err(),
        "FnPtrToPtr also requires a real pointer carrier"
    );

    let f32_ty = FP32Type::get(&ctx);
    let wrong_pointee_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, f32_ty.into(), true, MirPointerKind::RawMut);
    let wrong_pointee_block = BasicBlock::new(&mut ctx, None, vec![wrong_pointee_ty.into()]);
    let wrong_pointee = wrong_pointee_block.deref(&ctx).get_argument(0);
    let wrong_pointee_reborrow = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![wrong_pointee],
        vec![],
        0,
    );
    let wrong_pointee_reborrow_cast = MirCastOp::new(wrong_pointee_reborrow);
    wrong_pointee_reborrow_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    wrong_pointee_reborrow_cast
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    assert!(
        wrong_pointee_reborrow_cast.verify(&ctx).is_err(),
        "Reborrow authority must retain the pointee type"
    );

    for immutable_source in [shared, raw_const] {
        let invalid_unique = Operation::new(
            &mut ctx,
            MirCastOp::get_concrete_op_info(),
            vec![unique_ty.into()],
            vec![immutable_source],
            vec![],
            0,
        );
        let invalid_unique_cast = MirCastOp::new(invalid_unique);
        invalid_unique_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
        invalid_unique_cast
            .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
        assert!(
            invalid_unique_cast.verify(&ctx).is_err(),
            "an immutable source cannot be relabelled as UniqueRef by Reborrow authority"
        );

        let invalid_raw_mut = Operation::new(
            &mut ctx,
            MirCastOp::get_concrete_op_info(),
            vec![raw_mut_ty.into()],
            vec![immutable_source],
            vec![],
            0,
        );
        let invalid_raw_mut_cast = MirCastOp::new(invalid_raw_mut);
        invalid_raw_mut_cast.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
        invalid_raw_mut_cast
            .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::RawAddress);
        assert!(
            invalid_raw_mut_cast.verify(&ctx).is_err(),
            "an immutable source cannot be relabelled as RawMut by RawAddress authority"
        );
    }
}

#[test]
fn test_promoted_empty_mutable_reference_is_a_narrow_static_exception() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let empty_array_ty = MirArrayType::get(&mut ctx, i32_ty.into(), 0);
    let nonempty_array_ty = MirArrayType::get(&mut ctx, i32_ty.into(), 1);

    let empty_storage_ty = MirPtrType::get_with_kind(
        &mut ctx,
        empty_array_ty.into(),
        false,
        dialect_mir::types::address_space::GLOBAL,
        MirPointerKind::Erased,
    );
    let empty_unique_ty = MirPtrType::get_generic_with_kind(
        &mut ctx,
        empty_array_ty.into(),
        true,
        MirPointerKind::UniqueRef,
    );
    let empty_global_op = Operation::new(
        &mut ctx,
        MirGlobalAllocOp::get_concrete_op_info(),
        vec![empty_storage_ty.into()],
        vec![],
        vec![],
        0,
    );
    let empty_global = MirGlobalAllocOp::new(empty_global_op);
    empty_global.set_attr_global_type(&ctx, TypeAttr::new(empty_array_ty.into()));
    empty_global.set_attr_global_key(
        &ctx,
        StringAttr::new("promoted-empty-mutable-reference".to_string()),
    );
    empty_global.set_alignment_value(&mut ctx, 4);
    empty_global_op.deref_mut(&ctx).attributes.set(
        "global_initializer_hex".try_into().unwrap(),
        StringAttr::new(String::new()),
    );
    empty_global.mark_immutable(&mut ctx);
    assert!(empty_global.verify(&ctx).is_ok());

    let empty_global_storage = empty_global_op.deref(&ctx).get_result(0);
    let empty_static_borrow_op = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![empty_unique_ty.into()],
        vec![empty_global_storage],
        vec![],
        0,
    );
    let empty_static_borrow = MirCastOp::new(empty_static_borrow_op);
    empty_static_borrow.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    empty_static_borrow
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);

    empty_global.set_alignment_value(&mut ctx, 1);
    assert!(
        empty_static_borrow.verify(&ctx).is_err(),
        "even [i32; 0] must retain i32's natural four-byte alignment"
    );
    empty_global.set_alignment_value(&mut ctx, 4);
    assert!(
        empty_static_borrow.verify(&ctx).is_ok(),
        "an immutable promoted [T; 0] global may back rustc's vacuous &mut []"
    );

    empty_global_op.deref_mut(&ctx).attributes.set(
        "global_initializer_hex".try_into().unwrap(),
        StringAttr::new("00".to_string()),
    );
    assert!(
        empty_static_borrow.verify(&ctx).is_err(),
        "the exception requires an actually empty initializer, not merely the attribute"
    );
    empty_global_op.deref_mut(&ctx).attributes.set(
        "global_initializer_hex".try_into().unwrap(),
        StringAttr::new(String::new()),
    );
    empty_global_op.deref_mut(&ctx).attributes.set(
        "global_initializer_relocations".try_into().unwrap(),
        StringAttr::new("unexpected-relocation".to_string()),
    );
    assert!(
        empty_static_borrow.verify(&ctx).is_err(),
        "zero-byte promoted storage cannot carry a relocation"
    );

    let mutable_empty_global_op = Operation::new(
        &mut ctx,
        MirGlobalAllocOp::get_concrete_op_info(),
        vec![empty_storage_ty.into()],
        vec![],
        vec![],
        0,
    );
    let mutable_empty_global = MirGlobalAllocOp::new(mutable_empty_global_op);
    mutable_empty_global.set_attr_global_type(&ctx, TypeAttr::new(empty_array_ty.into()));
    mutable_empty_global.set_attr_global_key(
        &ctx,
        StringAttr::new("non-promoted-empty-global".to_string()),
    );
    assert!(mutable_empty_global.verify(&ctx).is_ok());
    let mutable_empty_storage = mutable_empty_global_op.deref(&ctx).get_result(0);
    let mutable_empty_borrow_op = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![empty_unique_ty.into()],
        vec![mutable_empty_storage],
        vec![],
        0,
    );
    let mutable_empty_borrow = MirCastOp::new(mutable_empty_borrow_op);
    mutable_empty_borrow.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    mutable_empty_borrow
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);
    assert!(
        mutable_empty_borrow.verify(&ctx).is_err(),
        "the [T; 0] exception requires compiler-promoted immutable storage"
    );

    let aligned_element_ty: TypeHandle = MirStructType::get_with_full_layout(
        &mut ctx,
        "Align16".to_string(),
        vec!["value".to_string()],
        vec![i32_ty.into()],
        vec![],
        vec![0],
        16,
        16,
    )
    .into();
    let aligned_empty_array_ty = MirArrayType::get(&mut ctx, aligned_element_ty, 0);
    let underaligned_storage_ty = MirPtrType::get_with_kind(
        &mut ctx,
        aligned_empty_array_ty.into(),
        false,
        dialect_mir::types::address_space::GLOBAL,
        MirPointerKind::Erased,
    );
    let aligned_empty_unique_ty = MirPtrType::get_generic_with_kind(
        &mut ctx,
        aligned_empty_array_ty.into(),
        true,
        MirPointerKind::UniqueRef,
    );
    let underaligned_global_op = Operation::new(
        &mut ctx,
        MirGlobalAllocOp::get_concrete_op_info(),
        vec![underaligned_storage_ty.into()],
        vec![],
        vec![],
        0,
    );
    let underaligned_global = MirGlobalAllocOp::new(underaligned_global_op);
    underaligned_global.set_attr_global_type(&ctx, TypeAttr::new(aligned_empty_array_ty.into()));
    underaligned_global.set_attr_global_key(
        &ctx,
        StringAttr::new("underaligned-empty-reference".to_string()),
    );
    underaligned_global.set_alignment_value(&mut ctx, 1);
    underaligned_global_op.deref_mut(&ctx).attributes.set(
        "global_initializer_hex".try_into().unwrap(),
        StringAttr::new(String::new()),
    );
    underaligned_global.mark_immutable(&mut ctx);
    assert!(underaligned_global.verify(&ctx).is_ok());
    let underaligned_storage = underaligned_global_op.deref(&ctx).get_result(0);
    let underaligned_borrow_op = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![aligned_empty_unique_ty.into()],
        vec![underaligned_storage],
        vec![],
        0,
    );
    let underaligned_borrow = MirCastOp::new(underaligned_borrow_op);
    underaligned_borrow.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    underaligned_borrow
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);
    assert!(
        underaligned_borrow.verify(&ctx).is_err(),
        "promoted &mut [Align16; 0] requires at least 16-byte global alignment"
    );

    let nonempty_storage_ty = MirPtrType::get_with_kind(
        &mut ctx,
        nonempty_array_ty.into(),
        false,
        dialect_mir::types::address_space::GLOBAL,
        MirPointerKind::Erased,
    );
    let nonempty_unique_ty = MirPtrType::get_generic_with_kind(
        &mut ctx,
        nonempty_array_ty.into(),
        true,
        MirPointerKind::UniqueRef,
    );
    let nonempty_global_op = Operation::new(
        &mut ctx,
        MirGlobalAllocOp::get_concrete_op_info(),
        vec![nonempty_storage_ty.into()],
        vec![],
        vec![],
        0,
    );
    let nonempty_global = MirGlobalAllocOp::new(nonempty_global_op);
    nonempty_global.set_attr_global_type(&ctx, TypeAttr::new(nonempty_array_ty.into()));
    nonempty_global.set_attr_global_key(
        &ctx,
        StringAttr::new("promoted-nonempty-mutable-reference".to_string()),
    );
    nonempty_global.mark_immutable(&mut ctx);
    assert!(nonempty_global.verify(&ctx).is_ok());
    let nonempty_global_storage = nonempty_global_op.deref(&ctx).get_result(0);
    let nonempty_static_borrow_op = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![nonempty_unique_ty.into()],
        vec![nonempty_global_storage],
        vec![],
        0,
    );
    let nonempty_static_borrow = MirCastOp::new(nonempty_static_borrow_op);
    nonempty_static_borrow.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    nonempty_static_borrow
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);
    assert!(
        nonempty_static_borrow.verify(&ctx).is_err(),
        "StaticAddress must never manufacture UniqueRef for non-empty promoted storage"
    );

    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![empty_storage_ty.into(), empty_unique_ty.into()],
    );
    let erased_block_argument = block.deref(&ctx).get_argument(0);
    let block_argument_borrow_op = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![empty_unique_ty.into()],
        vec![erased_block_argument],
        vec![],
        0,
    );
    let block_argument_borrow = MirCastOp::new(block_argument_borrow_op);
    block_argument_borrow.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    block_argument_borrow
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);
    assert!(
        block_argument_borrow.verify(&ctx).is_err(),
        "an Erased [T; 0] block argument has no proven promoted-global lineage"
    );

    let raw_empty_ty = MirPtrType::get_generic_with_kind(
        &mut ctx,
        empty_array_ty.into(),
        false,
        MirPointerKind::RawConst,
    );
    let raw_block = BasicBlock::new(&mut ctx, None, vec![raw_empty_ty.into()]);
    let raw_empty = raw_block.deref(&ctx).get_argument(0);
    let erase_raw_op = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![empty_storage_ty.into()],
        vec![raw_empty],
        vec![],
        0,
    );
    let erase_raw = MirCastOp::new(erase_raw_op);
    erase_raw.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    assert!(erase_raw.verify(&ctx).is_ok());
    let erased_from_raw = erase_raw_op.deref(&ctx).get_result(0);
    let laundered_borrow_op = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![empty_unique_ty.into()],
        vec![erased_from_raw],
        vec![],
        0,
    );
    let laundered_borrow = MirCastOp::new(laundered_borrow_op);
    laundered_borrow.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    laundered_borrow
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);
    assert!(
        laundered_borrow.verify(&ctx).is_err(),
        "RawConst -> Erased must not launder a zero-length pointer into UniqueRef"
    );

    let byte_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let byte_storage_ty = MirPtrType::get_with_kind(
        &mut ctx,
        byte_ty.into(),
        false,
        dialect_mir::types::address_space::GLOBAL,
        MirPointerKind::Erased,
    );
    let byte_global_op = Operation::new(
        &mut ctx,
        MirGlobalAllocOp::get_concrete_op_info(),
        vec![byte_storage_ty.into()],
        vec![],
        vec![],
        0,
    );
    let byte_global = MirGlobalAllocOp::new(byte_global_op);
    byte_global.set_attr_global_type(&ctx, TypeAttr::new(byte_ty.into()));
    byte_global.set_attr_global_key(
        &ctx,
        StringAttr::new("misaligned-empty-reference-root".to_string()),
    );
    byte_global.mark_immutable(&mut ctx);
    assert!(byte_global.verify(&ctx).is_ok());
    let byte_storage = byte_global_op.deref(&ctx).get_result(0);
    let retype_root_op = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![empty_storage_ty.into()],
        vec![byte_storage],
        vec![],
        0,
    );
    let retype_root = MirCastOp::new(retype_root_op);
    retype_root.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    assert!(retype_root.verify(&ctx).is_ok());
    let retyped_storage = retype_root_op.deref(&ctx).get_result(0);
    let misaligned_borrow_op = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![empty_unique_ty.into()],
        vec![retyped_storage],
        vec![],
        0,
    );
    let misaligned_borrow = MirCastOp::new(misaligned_borrow_op);
    misaligned_borrow.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    misaligned_borrow
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::StaticAddress);
    assert!(
        misaligned_borrow.verify(&ctx).is_err(),
        "an immutable byte global cannot be retyped into an aligned &mut [i32; 0] capability"
    );
}

fn promoted_empty_unique_ref_verifies(
    ctx: &mut Context,
    element_ty: TypeHandle,
    alignment: u64,
) -> bool {
    let empty_array_ty: TypeHandle = MirArrayType::get(ctx, element_ty, 0).into();
    let storage_ty = MirPtrType::get_with_kind(
        ctx,
        empty_array_ty,
        false,
        dialect_mir::types::address_space::GLOBAL,
        MirPointerKind::Erased,
    );
    let unique_ty =
        MirPtrType::get_generic_with_kind(ctx, empty_array_ty, true, MirPointerKind::UniqueRef);
    let global_op = Operation::new(
        ctx,
        MirGlobalAllocOp::get_concrete_op_info(),
        vec![storage_ty.into()],
        vec![],
        vec![],
        0,
    );
    let global = MirGlobalAllocOp::new(global_op);
    global.set_attr_global_type(ctx, TypeAttr::new(empty_array_ty));
    global.set_attr_global_key(ctx, StringAttr::new("promoted-empty-shape".to_string()));
    global.set_alignment_value(ctx, alignment);
    global_op.deref_mut(ctx).attributes.set(
        "global_initializer_hex".try_into().unwrap(),
        StringAttr::new(String::new()),
    );
    global.mark_immutable(ctx);

    let storage = global_op.deref(ctx).get_result(0);
    let cast_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![storage],
        vec![],
        0,
    );
    let cast = MirCastOp::new(cast_op);
    cast.set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    cast.set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::StaticAddress);
    cast.verify(ctx).is_ok()
}

#[test]
fn test_promoted_empty_alignment_covers_supported_fat_and_unit_elements() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let byte: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
    let word: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
    let unit: TypeHandle = MirTupleType::get(&mut ctx, vec![]).into();
    let slice: TypeHandle =
        MirSliceType::get_with_kind(&mut ctx, byte, MirPointerKind::SharedRef).into();
    let disjoint: TypeHandle = MirDisjointSliceType::get(&mut ctx, byte).into();

    assert!(promoted_empty_unique_ref_verifies(&mut ctx, unit, 1));
    assert!(promoted_empty_unique_ref_verifies(&mut ctx, slice, 8));
    assert!(promoted_empty_unique_ref_verifies(&mut ctx, disjoint, 8));
    assert!(promoted_empty_unique_ref_verifies(&mut ctx, word, 8));
    assert!(
        !promoted_empty_unique_ref_verifies(&mut ctx, word, 12),
        "a non-power-of-two numeric value is not a valid alignment guarantee"
    );
    assert!(
        !promoted_empty_unique_ref_verifies(&mut ctx, slice, 4),
        "a stored slice value still requires pointer-word alignment"
    );
}

#[test]
fn test_mir_ref_requires_exact_pointer_kind_authority() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let shared_ty = MirPtrType::get_generic_with_kind(
        &mut ctx,
        i32_ty.into(),
        false,
        MirPointerKind::SharedRef,
    );
    let unique_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::UniqueRef);
    let raw_const_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), false, MirPointerKind::RawConst);
    let raw_mut_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::RawMut);
    let erased_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let global_shared_ty = MirPtrType::get_with_kind(
        &mut ctx,
        i32_ty.into(),
        false,
        dialect_mir::types::address_space::GLOBAL,
        MirPointerKind::SharedRef,
    );
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let value = block.deref(&ctx).get_argument(0);

    let build =
        |ctx: &mut Context, result_ty, mutable, authority: Option<MirPointerKindAuthorityAttr>| {
            let op = Operation::new(
                ctx,
                MirRefOp::get_concrete_op_info(),
                vec![result_ty],
                vec![value],
                vec![],
                0,
            );
            let reference = MirRefOp::new(op);
            reference.set_mutable(ctx, mutable);
            if let Some(authority) = authority {
                reference.set_pointer_kind_authority(ctx, authority);
            }
            op
        };

    for (result_ty, mutable, authority) in [
        (
            shared_ty.into(),
            false,
            MirPointerKindAuthorityAttr::Reborrow,
        ),
        (
            unique_ty.into(),
            true,
            MirPointerKindAuthorityAttr::Reborrow,
        ),
        (
            raw_const_ty.into(),
            false,
            MirPointerKindAuthorityAttr::RawAddress,
        ),
        (
            raw_mut_ty.into(),
            true,
            MirPointerKindAuthorityAttr::RawAddress,
        ),
        (
            shared_ty.into(),
            false,
            MirPointerKindAuthorityAttr::StaticAddress,
        ),
    ] {
        assert!(
            MirRefOp::new(build(&mut ctx, result_ty, mutable, Some(authority)))
                .verify(&ctx)
                .is_ok()
        );
    }

    assert!(
        MirRefOp::new(build(&mut ctx, shared_ty.into(), false, None))
            .verify(&ctx)
            .is_err(),
        "mir.ref must visibly identify its Rust semantic origin"
    );
    assert!(
        MirRefOp::new(build(&mut ctx, erased_ty.into(), false, None))
            .verify(&ctx)
            .is_err(),
        "mir.ref is an explicit pointer-creation boundary, not generic Erased storage"
    );
    assert!(
        MirRefOp::new(build(
            &mut ctx,
            shared_ty.into(),
            false,
            Some(MirPointerKindAuthorityAttr::RawAddress),
        ))
        .verify(&ctx)
        .is_err(),
        "RawAddress cannot manufacture SharedRef"
    );
    assert!(
        MirRefOp::new(build(
            &mut ctx,
            unique_ty.into(),
            true,
            Some(MirPointerKindAuthorityAttr::StaticAddress),
        ))
        .verify(&ctx)
        .is_err(),
        "constant/static materialization cannot manufacture uniqueness"
    );
    assert!(
        MirRefOp::new(build(
            &mut ctx,
            unique_ty.into(),
            false,
            Some(MirPointerKindAuthorityAttr::Reborrow),
        ))
        .verify(&ctx)
        .is_err(),
        "reference kind must agree with the operation's mutability"
    );
    assert!(
        MirRefOp::new(build(
            &mut ctx,
            global_shared_ty.into(),
            false,
            Some(MirPointerKindAuthorityAttr::Reborrow),
        ))
        .verify(&ctx)
        .is_err(),
        "mir.ref materializes generic stack storage"
    );
}

#[test]
fn test_aggregate_cast_cannot_hide_pointer_kind_laundering() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let raw_mut_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::RawMut);
    let unique_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::UniqueRef);
    let raw_tuple_ty = MirTupleType::get(&mut ctx, vec![raw_mut_ty.into()]);
    let unique_tuple_ty = MirTupleType::get(&mut ctx, vec![unique_ty.into()]);
    let block = BasicBlock::new(&mut ctx, None, vec![raw_tuple_ty.into()]);
    let raw_tuple = block.deref(&ctx).get_argument(0);

    let build = |ctx: &mut Context, authority: Option<MirPointerKindAuthorityAttr>| {
        let op = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![unique_tuple_ty.into()],
            vec![raw_tuple],
            vec![],
            0,
        );
        let cast = MirCastOp::new(op);
        cast.set_attr_cast_kind(ctx, MirCastKindAttr::Transmute);
        if let Some(authority) = authority {
            cast.set_pointer_kind_authority(ctx, authority);
        }
        op
    };

    assert!(
        MirCastOp::new(build(&mut ctx, None)).verify(&ctx).is_err(),
        "nested RawMut -> UniqueRef laundering must be rejected"
    );
    assert!(
        MirCastOp::new(build(&mut ctx, Some(MirPointerKindAuthorityAttr::Reborrow),))
            .verify(&ctx)
            .is_err(),
        "Rvalue::Ref cannot authorize a nested aggregate transition"
    );
    assert!(
        MirCastOp::new(build(&mut ctx, Some(MirPointerKindAuthorityAttr::RustCast),))
            .verify(&ctx)
            .is_ok(),
        "an explicit rustc aggregate transmute is visible and authorized"
    );
}

#[test]
fn test_aggregate_reinterpretation_requires_rust_cast_authority() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let shared_ty = MirPtrType::get_generic_with_kind(
        &mut ctx,
        i64_ty.into(),
        false,
        MirPointerKind::SharedRef,
    );
    // Both tuple declarations list the same pointer at field 0, but its byte
    // offset changes from 0 to 8. Pairing by declaration index alone would
    // therefore mistake an integer's old bytes for a preserved reference.
    let source_ty = MirTupleType::get_with_layout(
        &mut ctx,
        vec![shared_ty.into(), i64_ty.into()],
        vec![0, 1],
        vec![0, 8],
        16,
        8,
    );
    let target_ty = MirTupleType::get_with_layout(
        &mut ctx,
        vec![shared_ty.into(), i64_ty.into()],
        vec![1, 0],
        vec![8, 0],
        16,
        8,
    );
    let block = BasicBlock::new(&mut ctx, None, vec![source_ty.into()]);
    let source = block.deref(&ctx).get_argument(0);

    let build = |ctx: &mut Context, authority: Option<MirPointerKindAuthorityAttr>| {
        let op = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![target_ty.into()],
            vec![source],
            vec![],
            0,
        );
        let cast = MirCastOp::new(op);
        cast.set_attr_cast_kind(ctx, MirCastKindAttr::Transmute);
        if let Some(authority) = authority {
            cast.set_pointer_kind_authority(ctx, authority);
        }
        op
    };

    assert!(
        MirCastOp::new(build(&mut ctx, None)).verify(&ctx).is_err(),
        "layout-changing aggregate casts must not infer pointer preservation by field index"
    );
    assert!(
        MirCastOp::new(build(&mut ctx, Some(MirPointerKindAuthorityAttr::RustCast),))
            .verify(&ctx)
            .is_ok(),
        "an explicit rustc transmute makes the representation reinterpretation auditable"
    );
}

#[test]
fn test_generic_aggregate_cast_cannot_invent_erased_pointer_carriers() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let word_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let erased_mut = MirPtrType::get_generic(&mut ctx, word_ty.into(), true);
    let source_ty = MirTupleType::get_with_layout(
        &mut ctx,
        vec![erased_mut.into(), word_ty.into()],
        vec![0, 1],
        vec![0, 8],
        16,
        8,
    );
    let extra_pointer_ty = MirTupleType::get_with_layout(
        &mut ctx,
        vec![erased_mut.into(), erased_mut.into()],
        vec![0, 1],
        vec![0, 8],
        16,
        8,
    );
    let moved_pointer_ty = MirTupleType::get_with_layout(
        &mut ctx,
        vec![erased_mut.into(), word_ty.into()],
        vec![1, 0],
        vec![8, 0],
        16,
        8,
    );
    let one_pointer = MirArrayType::get(&mut ctx, erased_mut.into(), 1);
    let two_pointers = MirArrayType::get(&mut ctx, erased_mut.into(), 2);

    let source_block = BasicBlock::new(&mut ctx, None, vec![source_ty.into()]);
    let source = source_block.deref(&ctx).get_argument(0);
    let array_block = BasicBlock::new(&mut ctx, None, vec![one_pointer.into()]);
    let array_source = array_block.deref(&ctx).get_argument(0);
    let build = |ctx: &mut Context, source, target| {
        let op = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![target],
            vec![source],
            vec![],
            0,
        );
        MirCastOp::new(op).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
        op
    };

    assert!(
        MirCastOp::new(build(&mut ctx, source, extra_pointer_ty.into()))
            .verify(&ctx)
            .is_err(),
        "PtrToPtr cannot reinterpret an integer field as writable Erased pointer evidence"
    );
    assert!(
        MirCastOp::new(build(&mut ctx, source, moved_pointer_ty.into()))
            .verify(&ctx)
            .is_err(),
        "field-index equality cannot hide an Erased pointer moving onto integer bytes"
    );
    assert!(
        MirCastOp::new(build(&mut ctx, array_source, two_pointers.into()))
            .verify(&ctx)
            .is_err(),
        "homogeneous array traversal must retain pointer-carrier cardinality"
    );
}

#[test]
fn test_rust_cast_authority_obeys_cast_kind_semantics() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let usize_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let array_ty = MirArrayType::get(&mut ctx, u8_ty.into(), 4);
    let raw_mut =
        MirPtrType::get_generic_with_kind(&mut ctx, u8_ty.into(), true, MirPointerKind::RawMut);
    let raw_const =
        MirPtrType::get_generic_with_kind(&mut ctx, u8_ty.into(), false, MirPointerKind::RawConst);
    let unique =
        MirPtrType::get_generic_with_kind(&mut ctx, u8_ty.into(), true, MirPointerKind::UniqueRef);
    let fn_target = MirStructType::get_with_full_layout(
        &mut ctx,
        "FnPtrTarget".into(),
        vec![],
        vec![],
        vec![],
        vec![],
        0,
        0,
    );
    let fn_target_ty: pliron::r#type::TypeHandle = fn_target.into();
    let fn_carrier = MirPtrType::get_generic(&mut ctx, fn_target_ty, false);
    let raw_mut_array =
        MirPtrType::get_generic_with_kind(&mut ctx, array_ty.into(), true, MirPointerKind::RawMut);
    let raw_const_array = MirPtrType::get_generic_with_kind(
        &mut ctx,
        array_ty.into(),
        false,
        MirPointerKind::RawConst,
    );
    let erased_array = MirPtrType::get_generic(&mut ctx, array_ty.into(), false);
    let erased_slice = MirSliceType::get(&mut ctx, u8_ty.into());
    let erased_mut_slice = MirSliceType::get_with_mutability(&mut ctx, u8_ty.into(), true);
    let raw_const_slice =
        MirSliceType::get_with_kind(&mut ctx, u8_ty.into(), MirPointerKind::RawConst);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            raw_mut.into(),
            raw_const.into(),
            fn_carrier.into(),
            raw_mut_array.into(),
            raw_const_array.into(),
            erased_array.into(),
            usize_ty.into(),
        ],
    );
    let raw_mut_value = block.deref(&ctx).get_argument(0);
    let raw_const_value = block.deref(&ctx).get_argument(1);
    let fn_value = block.deref(&ctx).get_argument(2);
    let raw_mut_array_value = block.deref(&ctx).get_argument(3);
    let raw_const_array_value = block.deref(&ctx).get_argument(4);
    let erased_array_value = block.deref(&ctx).get_argument(5);
    let integer_value = block.deref(&ctx).get_argument(6);

    let build = |ctx: &mut Context,
                 source,
                 target,
                 kind,
                 authority: Option<MirPointerKindAuthorityAttr>| {
        let op = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![target],
            vec![source],
            vec![],
            0,
        );
        let cast = MirCastOp::new(op);
        cast.set_attr_cast_kind(ctx, kind);
        if let Some(authority) = authority {
            cast.set_pointer_kind_authority(ctx, authority);
        }
        op
    };
    let rust_cast = Some(MirPointerKindAuthorityAttr::RustCast);

    assert!(
        MirCastOp::new(build(
            &mut ctx,
            raw_mut_value,
            unique.into(),
            MirCastKindAttr::PtrToPtr,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_err(),
        "PtrToPtr cannot use RustCast to invent a reference"
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            raw_mut_value,
            raw_const.into(),
            MirCastKindAttr::PtrToPtr,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_ok(),
        "PtrToPtr may perform an explicit raw-to-raw cast"
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            fn_value,
            raw_const.into(),
            MirCastKindAttr::FnPtrToPtr,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_ok(),
        "FnPtrToPtr may expose an opaque function pointer as a raw pointer"
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            erased_array_value,
            raw_const.into(),
            MirCastKindAttr::FnPtrToPtr,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_err(),
        "FnPtrToPtr cannot relabel arbitrary Erased storage as a function pointer"
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            fn_value,
            unique.into(),
            MirCastKindAttr::FnPtrToPtr,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_err()
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            raw_mut_value,
            raw_const.into(),
            MirCastKindAttr::PointerCoercionMutToConst,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_ok()
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            raw_const_value,
            raw_mut.into(),
            MirCastKindAttr::PointerCoercionMutToConst,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_err()
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            raw_mut_value,
            raw_mut.into(),
            MirCastKindAttr::PointerCoercionMutToConst,
            None,
        ))
        .verify(&ctx)
        .is_err(),
        "MutToConst cannot masquerade as a same-category pointer cast"
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            raw_const_array_value,
            raw_const_slice.into(),
            MirCastKindAttr::PointerCoercionUnsize,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_ok(),
        "Unsize may change thin/fat shape while preserving its carrier"
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            erased_array_value,
            erased_slice.into(),
            MirCastKindAttr::PointerCoercionUnsize,
            None,
        ))
        .verify(&ctx)
        .is_ok(),
        "an all-Erased unsize still preserves read-only carrier state"
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            erased_array_value,
            erased_mut_slice.into(),
            MirCastKindAttr::PointerCoercionUnsize,
            None,
        ))
        .verify(&ctx)
        .is_err(),
        "Unsize cannot turn an immutable Erased thin pointer into writable fat evidence"
    );

    let prefix_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let sized_tail = MirStructType::get_with_full_layout(
        &mut ctx,
        "TailCarrier".into(),
        vec!["prefix".into(), "tail".into()],
        vec![prefix_ty.into(), array_ty.into()],
        vec![0, 1],
        vec![0, 8],
        16,
        8,
    );
    let shifted_unsized_tail = MirStructType::get_with_full_layout(
        &mut ctx,
        "TailCarrier".into(),
        vec!["prefix".into(), "tail".into()],
        vec![prefix_ty.into(), u8_ty.into()],
        vec![0, 1],
        vec![0, 16],
        24,
        8,
    );
    let sized_tail_ptr = MirPtrType::get_generic_with_kind(
        &mut ctx,
        sized_tail.into(),
        false,
        MirPointerKind::RawConst,
    );
    let shifted_tail_slice = MirSliceType::get_with_kind(
        &mut ctx,
        shifted_unsized_tail.into(),
        MirPointerKind::RawConst,
    );
    let tail_block = BasicBlock::new(&mut ctx, None, vec![sized_tail_ptr.into()]);
    let tail_value = tail_block.deref(&ctx).get_argument(0);
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            tail_value,
            shifted_tail_slice.into(),
            MirCastKindAttr::PointerCoercionUnsize,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_err(),
        "Unsize cannot move the trailing data behind a changed struct-field offset"
    );

    assert!(
        MirCastOp::new(build(
            &mut ctx,
            raw_const_array_value,
            raw_mut.into(),
            MirCastKindAttr::PointerCoercionArrayToPointer,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_err(),
        "ArrayToPointer cannot strengthen const raw storage to mutable"
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            raw_mut_value,
            raw_const.into(),
            MirCastKindAttr::PointerCoercionArrayToPointer,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_err(),
        "ArrayToPointer requires a raw pointer to an array, not an arbitrary pointer"
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            raw_mut_value,
            raw_mut.into(),
            MirCastKindAttr::PointerCoercionArrayToPointer,
            None,
        ))
        .verify(&ctx)
        .is_err(),
        "an unmarked same-kind ArrayToPointer still requires an array source"
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            raw_mut_array_value,
            raw_const.into(),
            MirCastKindAttr::PointerCoercionArrayToPointer,
            rust_cast.clone(),
        ))
        .verify(&ctx)
        .is_ok()
    );
    assert!(
        MirCastOp::new(build(
            &mut ctx,
            integer_value,
            unique.into(),
            MirCastKindAttr::Transmute,
            rust_cast,
        ))
        .verify(&ctx)
        .is_ok(),
        "only an explicit Transmute may establish an arbitrary pointer category"
    );

    let word_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let source_wrapper = MirStructType::get_with_full_layout(
        &mut ctx,
        "CarrierWrapper".into(),
        vec!["pointer".into(), "word".into()],
        vec![unique.into(), word_ty.into()],
        vec![0, 1],
        vec![0, 8],
        16,
        8,
    );
    let moved_wrapper = MirStructType::get_with_full_layout(
        &mut ctx,
        "CarrierWrapper".into(),
        vec!["word".into(), "pointer".into()],
        vec![word_ty.into(), unique.into()],
        vec![0, 1],
        vec![0, 8],
        16,
        8,
    );
    let wrapper_block = BasicBlock::new(&mut ctx, None, vec![source_wrapper.into()]);
    let wrapper_value = wrapper_block.deref(&ctx).get_argument(0);
    for kind in [
        MirCastKindAttr::PointerCoercionUnsize,
        MirCastKindAttr::Subtype,
    ] {
        assert!(
            MirCastOp::new(build(
                &mut ctx,
                wrapper_value,
                moved_wrapper.into(),
                kind,
                Some(MirPointerKindAuthorityAttr::RustCast),
            ))
            .verify(&ctx)
            .is_err(),
            "Unsize/Subtype cannot bless a pointer carrier moved onto integer bytes"
        );
    }
}

#[test]
fn test_pointer_projections_preserve_or_erase_kind_only() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let usize_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let tuple_ty = MirTupleType::get(&mut ctx, vec![i32_ty.into()]);
    let array_ty = MirArrayType::get(&mut ctx, i32_ty.into(), 4);
    let raw_tuple_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, tuple_ty.into(), true, MirPointerKind::RawMut);
    let raw_array_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, array_ty.into(), true, MirPointerKind::RawMut);
    let raw_field_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::RawMut);
    let erased_field_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let erased_read_field_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let unique_field_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::UniqueRef);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![raw_tuple_ty.into(), raw_array_ty.into(), usize_ty.into()],
    );
    let tuple = block.deref(&ctx).get_argument(0);
    let array = block.deref(&ctx).get_argument(1);
    let index = block.deref(&ctx).get_argument(2);

    let field = |ctx: &mut Context, result_ty| {
        let op = Operation::new(
            ctx,
            MirFieldAddrOp::get_concrete_op_info(),
            vec![result_ty],
            vec![tuple],
            vec![],
            0,
        );
        MirFieldAddrOp::new(op).set_attr_field_index(ctx, FieldIndexAttr(0));
        MirFieldAddrOp::new(op).set_attr_aggregate_ty(ctx, TypeAttr::new(tuple_ty.into()));
        op
    };
    assert!(
        MirFieldAddrOp::new(field(&mut ctx, raw_field_ty.into()))
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        MirFieldAddrOp::new(field(&mut ctx, erased_field_ty.into()))
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        MirFieldAddrOp::new(field(&mut ctx, unique_field_ty.into()))
            .verify(&ctx)
            .is_err()
    );
    assert!(
        MirFieldAddrOp::new(field(&mut ctx, erased_read_field_ty.into()))
            .verify(&ctx)
            .is_err(),
        "field projection cannot flip an Erased address from writable to read-only"
    );

    let array_launder = Operation::new(
        &mut ctx,
        MirArrayElementAddrOp::get_concrete_op_info(),
        vec![unique_field_ty.into()],
        vec![array, index],
        vec![],
        0,
    );
    assert!(
        MirArrayElementAddrOp::new(array_launder)
            .verify(&ctx)
            .is_err()
    );
    let array_erase = Operation::new(
        &mut ctx,
        MirArrayElementAddrOp::get_concrete_op_info(),
        vec![erased_field_ty.into()],
        vec![array, index],
        vec![],
        0,
    );
    assert!(
        MirArrayElementAddrOp::new(array_erase).verify(&ctx).is_ok(),
        "projection may erase kind while preserving machine mutability"
    );
    let array_mutability_flip = Operation::new(
        &mut ctx,
        MirArrayElementAddrOp::get_concrete_op_info(),
        vec![erased_read_field_ty.into()],
        vec![array, index],
        vec![],
        0,
    );
    assert!(
        MirArrayElementAddrOp::new(array_mutability_flip)
            .verify(&ctx)
            .is_err(),
        "array projection cannot change machine mutability"
    );
}

#[test]
fn test_pointer_with_exposed_provenance_only_creates_raw_or_erased_pointer() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let usize_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let fn_target = MirStructType::get_with_full_layout(
        &mut ctx,
        "FnPtrTarget".into(),
        vec![],
        vec![],
        vec![],
        vec![],
        0,
        0,
    );
    let fn_target_ty: pliron::r#type::TypeHandle = fn_target.into();
    let raw_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::RawMut);
    let opaque_fn_ty = MirPtrType::get_generic(&mut ctx, fn_target_ty, false);
    let arbitrary_erased_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let writable_erased_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let unique_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, i32_ty.into(), true, MirPointerKind::UniqueRef);
    let block = BasicBlock::new(&mut ctx, None, vec![usize_ty.into()]);
    let address = block.deref(&ctx).get_argument(0);

    let build = |ctx: &mut Context, result_ty| {
        let op = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![result_ty],
            vec![address],
            vec![],
            0,
        );
        let cast = MirCastOp::new(op);
        cast.set_attr_cast_kind(ctx, MirCastKindAttr::PointerWithExposedProvenance);
        cast.set_pointer_kind_authority(ctx, MirPointerKindAuthorityAttr::RustCast);
        op
    };

    assert!(
        MirCastOp::new(build(&mut ctx, raw_ty.into()))
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        MirCastOp::new(build(&mut ctx, unique_ty.into()))
            .verify(&ctx)
            .is_err(),
        "integer provenance cannot directly materialize a Rust reference"
    );

    let build_unmarked = |ctx: &mut Context, result_ty| {
        let op = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![result_ty],
            vec![address],
            vec![],
            0,
        );
        MirCastOp::new(op).set_attr_cast_kind(ctx, MirCastKindAttr::PointerWithExposedProvenance);
        op
    };
    assert!(
        MirCastOp::new(build_unmarked(&mut ctx, opaque_fn_ty.into()))
            .verify(&ctx)
            .is_ok(),
        "function-pointer tokens may materialize only the canonical immutable Erased carrier"
    );
    assert!(
        MirCastOp::new(build_unmarked(&mut ctx, arbitrary_erased_ty.into()))
            .verify(&ctx)
            .is_err(),
        "an integer cannot manufacture arbitrary Erased pointer evidence"
    );
    assert!(
        MirCastOp::new(build_unmarked(&mut ctx, writable_erased_ty.into()))
            .verify(&ctx)
            .is_err(),
        "an integer cannot manufacture writable Erased evidence and then reborrow it as UniqueRef"
    );

    let fn_block = BasicBlock::new(&mut ctx, None, vec![opaque_fn_ty.into()]);
    let opaque_fn = fn_block.deref(&ctx).get_argument(0);
    let disguised_data_pointer = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![arbitrary_erased_ty.into()],
        vec![opaque_fn],
        vec![],
        0,
    );
    MirCastOp::new(disguised_data_pointer)
        .set_attr_cast_kind(&ctx, MirCastKindAttr::PointerCoercionArrayToPointer);
    assert!(
        MirCastOp::new(disguised_data_pointer).verify(&ctx).is_err(),
        "an unrelated coercion cannot turn the opaque function token into Erased data storage"
    );

    let shared_fn_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, fn_target_ty, false, MirPointerKind::SharedRef);
    let fake_reborrow = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![shared_fn_ty.into()],
        vec![opaque_fn],
        vec![],
        0,
    );
    let fake_reborrow = MirCastOp::new(fake_reborrow);
    fake_reborrow.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    fake_reborrow.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    assert!(
        fake_reborrow.verify(&ctx).is_err(),
        "an opaque function-pointer value is not compiler storage that may be reborrowed"
    );

    // Keeping only kind+mutability while recursively pairing aggregate fields
    // is insufficient: the canonical function token is also identified by
    // its pointee. Otherwise a tuple cast can disguise data storage as a
    // function pointer (or vice versa), after which individually legal casts
    // can manufacture a reference.
    let data_tuple_ty = MirTupleType::get(&mut ctx, vec![arbitrary_erased_ty.into()]);
    let fn_tuple_ty = MirTupleType::get(&mut ctx, vec![opaque_fn_ty.into()]);
    let nested_block = BasicBlock::new(
        &mut ctx,
        None,
        vec![data_tuple_ty.into(), fn_tuple_ty.into()],
    );
    let data_tuple = nested_block.deref(&ctx).get_argument(0);
    let fn_tuple = nested_block.deref(&ctx).get_argument(1);
    let nested_cast = |ctx: &mut Context, source, target| {
        let op = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![target],
            vec![source],
            vec![],
            0,
        );
        MirCastOp::new(op).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
        op
    };
    let data_to_fn = nested_cast(&mut ctx, data_tuple, fn_tuple_ty.into());
    assert!(
        MirCastOp::new(data_to_fn).verify(&ctx).is_err(),
        "an aggregate cast cannot manufacture a nested canonical function token"
    );
    assert!(
        MirCastOp::new(nested_cast(&mut ctx, fn_tuple, data_tuple_ty.into(),))
            .verify(&ctx)
            .is_err(),
        "an aggregate cast cannot disguise a nested function token as Erased data"
    );

    let forged_fn_tuple = data_to_fn.deref(&ctx).get_result(0);
    let extract_forged_fn = Operation::new(
        &mut ctx,
        MirExtractFieldOp::get_concrete_op_info(),
        vec![opaque_fn_ty.into()],
        vec![forged_fn_tuple],
        vec![],
        0,
    );
    let extract_forged_fn = MirExtractFieldOp::new(extract_forged_fn);
    extract_forged_fn.set_attr_index(&ctx, FieldIndexAttr(0));
    assert!(extract_forged_fn.verify(&ctx).is_ok());

    let forged_fn = extract_forged_fn.get_operation().deref(&ctx).get_result(0);
    let expose_forged_fn = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![raw_ty.into()],
        vec![forged_fn],
        vec![],
        0,
    );
    let expose_forged_fn = MirCastOp::new(expose_forged_fn);
    expose_forged_fn.set_attr_cast_kind(&ctx, MirCastKindAttr::FnPtrToPtr);
    expose_forged_fn.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::RustCast);
    assert!(expose_forged_fn.verify(&ctx).is_ok());

    let forged_raw = expose_forged_fn.get_operation().deref(&ctx).get_result(0);
    let reborrow_forged_fn = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![forged_raw],
        vec![],
        0,
    );
    let reborrow_forged_fn = MirCastOp::new(reborrow_forged_fn);
    reborrow_forged_fn.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    reborrow_forged_fn.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    assert!(
        reborrow_forged_fn.verify(&ctx).is_ok(),
        "the laundering chain must be rejected at its aggregate reinterpretation"
    );

    let shared_marker_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, fn_target_ty, false, MirPointerKind::SharedRef);
    let marker_tuple_ty = MirTupleType::get(&mut ctx, vec![fn_target_ty]);
    let shared_marker_tuple_ty = MirPtrType::get_generic_with_kind(
        &mut ctx,
        marker_tuple_ty.into(),
        false,
        MirPointerKind::SharedRef,
    );
    let marker_array_ty = MirArrayType::get(&mut ctx, fn_target_ty, 1);
    let shared_marker_array_ty = MirPtrType::get_generic_with_kind(
        &mut ctx,
        marker_array_ty.into(),
        false,
        MirPointerKind::SharedRef,
    );
    let shared_marker_slice_ty =
        MirSliceType::get_with_kind(&mut ctx, fn_target_ty, MirPointerKind::SharedRef);
    let projection_block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            shared_marker_ty.into(),
            shared_marker_tuple_ty.into(),
            shared_marker_array_ty.into(),
            shared_marker_slice_ty.into(),
            usize_ty.into(),
        ],
    );
    let marker_address = projection_block.deref(&ctx).get_argument(0);
    let marker_tuple_address = projection_block.deref(&ctx).get_argument(1);
    let marker_array_address = projection_block.deref(&ctx).get_argument(2);
    let marker_slice = projection_block.deref(&ctx).get_argument(3);
    let zero = projection_block.deref(&ctx).get_argument(4);

    let offset_to_token = Operation::new(
        &mut ctx,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![opaque_fn_ty.into()],
        vec![marker_address, zero],
        vec![],
        0,
    );
    assert!(
        MirPtrOffsetOp::new(offset_to_token).verify(&ctx).is_err(),
        "pointer arithmetic produces a data address, never a function token"
    );

    let field_address_to_token = Operation::new(
        &mut ctx,
        MirFieldAddrOp::get_concrete_op_info(),
        vec![opaque_fn_ty.into()],
        vec![marker_tuple_address],
        vec![],
        0,
    );
    let field_address_to_token = MirFieldAddrOp::new(field_address_to_token);
    field_address_to_token.set_attr_field_index(&ctx, FieldIndexAttr(0));
    assert!(
        field_address_to_token.verify(&ctx).is_err(),
        "a field address cannot masquerade as a function token"
    );

    let array_address_to_token = Operation::new(
        &mut ctx,
        MirArrayElementAddrOp::get_concrete_op_info(),
        vec![opaque_fn_ty.into()],
        vec![marker_array_address, zero],
        vec![],
        0,
    );
    assert!(
        MirArrayElementAddrOp::new(array_address_to_token)
            .verify(&ctx)
            .is_err(),
        "an array element address cannot masquerade as a function token"
    );

    let slice_data_to_token = Operation::new(
        &mut ctx,
        MirExtractFieldOp::get_concrete_op_info(),
        vec![opaque_fn_ty.into()],
        vec![marker_slice],
        vec![],
        0,
    );
    let slice_data_to_token = MirExtractFieldOp::new(slice_data_to_token);
    slice_data_to_token.set_attr_index(&ctx, FieldIndexAttr(0));
    assert!(
        slice_data_to_token.verify(&ctx).is_err(),
        "a slice data address cannot masquerade as a function token"
    );

    // A ClosureFnPointer cast must not extract captured reference bits as a
    // function pointer, expose them as RawMut, and then reborrow them as
    // UniqueRef. Only a genuinely non-capturing, zero-sized closure may enter
    // the opaque function-pointer path.
    let captured_ref = MirPtrType::get_generic_with_kind(
        &mut ctx,
        i32_ty.into(),
        false,
        MirPointerKind::SharedRef,
    );
    let captured_closure = MirStructType::get_with_full_layout(
        &mut ctx,
        "CapturedClosure".into(),
        vec!["capture_0".into()],
        vec![captured_ref.into()],
        vec![0],
        vec![0],
        8,
        8,
    );
    let empty_closure = MirStructType::get_with_full_layout(
        &mut ctx,
        "NonCapturingClosure".into(),
        vec![],
        vec![],
        vec![],
        vec![],
        0,
        1,
    );
    let closure_block = BasicBlock::new(
        &mut ctx,
        None,
        vec![captured_closure.into(), empty_closure.into()],
    );
    let captured_value = closure_block.deref(&ctx).get_argument(0);
    let empty_value = closure_block.deref(&ctx).get_argument(1);
    let closure_cast = |ctx: &mut Context, source| {
        let op = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![opaque_fn_ty.into()],
            vec![source],
            vec![],
            0,
        );
        MirCastOp::new(op)
            .set_attr_cast_kind(ctx, MirCastKindAttr::PointerCoercionClosureFnPointer);
        op
    };
    let captured_to_fn = closure_cast(&mut ctx, captured_value);
    assert!(
        MirCastOp::new(captured_to_fn).verify(&ctx).is_err(),
        "a closure carrying SharedRef bytes cannot become an opaque function pointer"
    );
    assert!(
        MirCastOp::new(closure_cast(&mut ctx, empty_value))
            .verify(&ctx)
            .is_err(),
        "the importer materializes a closure function token directly; the legacy cast is not lowerable"
    );

    let captured_fn_value = captured_to_fn.deref(&ctx).get_result(0);
    let exposed_capture = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![raw_ty.into()],
        vec![captured_fn_value],
        vec![],
        0,
    );
    let exposed_capture = MirCastOp::new(exposed_capture);
    exposed_capture.set_attr_cast_kind(&ctx, MirCastKindAttr::FnPtrToPtr);
    exposed_capture.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::RustCast);
    assert!(exposed_capture.verify(&ctx).is_ok());

    let exposed_capture_value = exposed_capture.get_operation().deref(&ctx).get_result(0);
    let unique_capture = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_ty.into()],
        vec![exposed_capture_value],
        vec![],
        0,
    );
    let unique_capture = MirCastOp::new(unique_capture);
    unique_capture.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    unique_capture.set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    assert!(
        unique_capture.verify(&ctx).is_ok(),
        "the chain must be stopped at ClosureFnPointer before later individually legal boundaries"
    );
}

#[test]
fn test_mir_extract_field_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
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

    let union_ty = MirUnionType::get(
        &mut ctx,
        "Bits".into(),
        vec!["word".into(), "alias".into()],
        vec![i32_ty.into(), i32_ty.into()],
        4,
        4,
    );
    let union_block = BasicBlock::new(&mut ctx, None, vec![union_ty.into()]);
    let union_val = union_block.deref(&ctx).get_argument(0);
    let union_extract = Operation::new(
        &mut ctx,
        MirExtractFieldOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![union_val],
        vec![],
        0,
    );
    let union_extract = MirExtractFieldOp::new(union_extract);
    union_extract.set_attr_index(&ctx, dialect_mir::attributes::FieldIndexAttr(1));
    assert!(union_extract.verify(&ctx).is_ok(), "Valid union extract");
}

#[test]
fn test_mir_construct_disjoint_slice_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let f32_ty = FP32Type::get(&ctx);
    let usize_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let width_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let f32_ptr_ty =
        MirPtrType::get_generic_with_kind(&mut ctx, f32_ty.into(), true, MirPointerKind::RawMut);
    let plain_ty = MirDisjointSliceType::get(&mut ctx, f32_ty.into());
    let width_ty_handle: pliron::r#type::TypeHandle = width_ty.into();
    let row_width_ty =
        MirDisjointSliceType::get_with_space(&mut ctx, f32_ty.into(), vec![width_ty_handle]);

    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![f32_ptr_ty.into(), usize_ty.into(), width_ty.into()],
    );
    let ptr_val = block.deref(&ctx).get_argument(0);
    let len_val = block.deref(&ctx).get_argument(1);
    let width_val = block.deref(&ctx).get_argument(2);

    // Valid: an index space with no runtime layout takes two operands.
    let op = Operation::new(
        &mut ctx,
        MirConstructDisjointSliceOp::get_concrete_op_info(),
        vec![plain_ty.into()],
        vec![ptr_val, len_val],
        vec![],
        0,
    );
    assert!(
        MirConstructDisjointSliceOp::new(op).verify(&ctx).is_ok(),
        "Valid space-free disjoint slice construction"
    );

    // Valid: a runtime row width takes a third operand.
    let op_width = Operation::new(
        &mut ctx,
        MirConstructDisjointSliceOp::get_concrete_op_info(),
        vec![row_width_ty.into()],
        vec![ptr_val, len_val, width_val],
        vec![],
        0,
    );
    assert!(
        MirConstructDisjointSliceOp::new(op_width)
            .verify(&ctx)
            .is_ok(),
        "Valid row-width disjoint slice construction"
    );

    let erased_ptr_ty = MirPtrType::get_generic(&mut ctx, f32_ty.into(), true);
    let erased_block = BasicBlock::new(&mut ctx, None, vec![erased_ptr_ty.into()]);
    let erased_ptr = erased_block.deref(&ctx).get_argument(0);
    let op_erased_data = Operation::new(
        &mut ctx,
        MirConstructDisjointSliceOp::get_concrete_op_info(),
        vec![plain_ty.into()],
        vec![erased_ptr, len_val],
        vec![],
        0,
    );
    assert!(
        MirConstructDisjointSliceOp::new(op_erased_data)
            .verify(&ctx)
            .is_err(),
        "DisjointSlice's fixed RawMut field cannot be reconstructed from Erased"
    );

    // Invalid: the row width is missing, so the slice would carry whatever
    // slot 2 held.
    let op_missing_width = Operation::new(
        &mut ctx,
        MirConstructDisjointSliceOp::get_concrete_op_info(),
        vec![row_width_ty.into()],
        vec![ptr_val, len_val],
        vec![],
        0,
    );
    assert!(
        MirConstructDisjointSliceOp::new(op_missing_width)
            .verify(&ctx)
            .is_err(),
        "Missing index-space operand"
    );

    // Invalid: a space-free slice given a third operand.
    let op_extra = Operation::new(
        &mut ctx,
        MirConstructDisjointSliceOp::get_concrete_op_info(),
        vec![plain_ty.into()],
        vec![ptr_val, len_val, width_val],
        vec![],
        0,
    );
    assert!(
        MirConstructDisjointSliceOp::new(op_extra)
            .verify(&ctx)
            .is_err(),
        "Index-space operand for a space-free slice"
    );

    // Invalid: the row width operand has the wrong width, which would write a
    // 64-bit value into the 32-bit row width slot.
    let op_wrong_width_ty = Operation::new(
        &mut ctx,
        MirConstructDisjointSliceOp::get_concrete_op_info(),
        vec![row_width_ty.into()],
        vec![ptr_val, len_val, len_val],
        vec![],
        0,
    );
    assert!(
        MirConstructDisjointSliceOp::new(op_wrong_width_ty)
            .verify(&ctx)
            .is_err(),
        "Index-space operand type mismatch"
    );

    // Invalid: result is a plain slice, which `mir.construct_slice` owns.
    let plain_slice_ty = MirSliceType::get(&mut ctx, f32_ty.into());
    let op_bad_res = Operation::new(
        &mut ctx,
        MirConstructDisjointSliceOp::get_concrete_op_info(),
        vec![plain_slice_ty.into()],
        vec![ptr_val, len_val],
        vec![],
        0,
    );
    assert!(
        MirConstructDisjointSliceOp::new(op_bad_res)
            .verify(&ctx)
            .is_err(),
        "Result must be a disjoint slice type"
    );
}

#[test]
fn test_mir_construct_slice_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let usize_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
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
fn test_slice_carrier_cannot_launder_pointer_kind() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let usize_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let raw_mut_ptr =
        MirPtrType::get_generic_with_kind(&mut ctx, u8_ty.into(), true, MirPointerKind::RawMut);
    let erased_ptr = MirPtrType::get_generic(&mut ctx, u8_ty.into(), true);
    let erased_const_ptr = MirPtrType::get_generic(&mut ctx, u8_ty.into(), false);
    let global_raw_mut_ptr = MirPtrType::get_with_kind(
        &mut ctx,
        u8_ty.into(),
        true,
        dialect_mir::types::address_space::GLOBAL,
        MirPointerKind::RawMut,
    );
    let unique_ptr =
        MirPtrType::get_generic_with_kind(&mut ctx, u8_ty.into(), true, MirPointerKind::UniqueRef);
    let raw_mut_slice = MirSliceType::get_with_kind(&mut ctx, u8_ty.into(), MirPointerKind::RawMut);
    let unique_slice =
        MirSliceType::get_with_kind(&mut ctx, u8_ty.into(), MirPointerKind::UniqueRef);
    let erased_slice = MirSliceType::get(&mut ctx, u8_ty.into());
    let erased_mut_slice = MirSliceType::get_with_mutability(&mut ctx, u8_ty.into(), true);
    let block = BasicBlock::new(
        &mut ctx,
        None,
        vec![
            raw_mut_ptr.into(),
            erased_ptr.into(),
            erased_const_ptr.into(),
            global_raw_mut_ptr.into(),
            usize_ty.into(),
            raw_mut_slice.into(),
            erased_slice.into(),
            erased_mut_slice.into(),
        ],
    );
    let raw_mut = block.deref(&ctx).get_argument(0);
    let erased = block.deref(&ctx).get_argument(1);
    let erased_const = block.deref(&ctx).get_argument(2);
    let global_raw_mut = block.deref(&ctx).get_argument(3);
    let len = block.deref(&ctx).get_argument(4);
    let raw_slice = block.deref(&ctx).get_argument(5);
    let erased_slice_value = block.deref(&ctx).get_argument(6);
    let erased_mut_slice_value = block.deref(&ctx).get_argument(7);

    let construct = |ctx: &mut Context, data, result_ty| {
        Operation::new(
            ctx,
            MirConstructSliceOp::get_concrete_op_info(),
            vec![result_ty],
            vec![data, len],
            vec![],
            0,
        )
    };
    assert!(
        MirConstructSliceOp::new(construct(&mut ctx, raw_mut, raw_mut_slice.into()))
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        MirConstructSliceOp::new(construct(&mut ctx, raw_mut, unique_slice.into()))
            .verify(&ctx)
            .is_err(),
        "construct_slice must not turn RawMut into UniqueRef"
    );
    assert!(
        MirConstructSliceOp::new(construct(&mut ctx, erased, raw_mut_slice.into()))
            .verify(&ctx)
            .is_err(),
        "construct_slice must not recover RawMut from Erased"
    );
    assert!(
        MirConstructSliceOp::new(construct(&mut ctx, global_raw_mut, raw_mut_slice.into()))
            .verify(&ctx)
            .is_err(),
        "ordinary slice carriers always use generic address space"
    );
    assert!(
        MirConstructSliceOp::new(construct(&mut ctx, erased_const, erased_slice.into()))
            .verify(&ctx)
            .is_ok(),
        "an immutable Erased data pointer constructs an immutable Erased slice"
    );

    let extract = |ctx: &mut Context, result_ty| {
        let op = Operation::new(
            ctx,
            MirExtractFieldOp::get_concrete_op_info(),
            vec![result_ty],
            vec![raw_slice],
            vec![],
            0,
        );
        MirExtractFieldOp::new(op).set_attr_index(ctx, FieldIndexAttr(0));
        op
    };
    assert!(
        MirExtractFieldOp::new(extract(&mut ctx, raw_mut_ptr.into()))
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        MirExtractFieldOp::new(extract(&mut ctx, erased_ptr.into()))
            .verify(&ctx)
            .is_ok(),
        "slice extraction may deliberately erase a concrete kind"
    );
    assert!(
        MirExtractFieldOp::new(extract(&mut ctx, unique_ptr.into()))
            .verify(&ctx)
            .is_err(),
        "slice extraction must not change RawMut into UniqueRef"
    );
    assert!(
        MirExtractFieldOp::new(extract(&mut ctx, global_raw_mut_ptr.into()))
            .verify(&ctx)
            .is_err(),
        "ordinary slice extraction cannot invent a non-generic address space"
    );
    assert!(
        MirExtractFieldOp::new(extract(&mut ctx, erased_const_ptr.into()))
            .verify(&ctx)
            .is_err(),
        "erasing a concrete slice kind must preserve machine mutability"
    );

    let extract_erased = |ctx: &mut Context, result_ty| {
        let op = Operation::new(
            ctx,
            MirExtractFieldOp::get_concrete_op_info(),
            vec![result_ty],
            vec![erased_slice_value],
            vec![],
            0,
        );
        MirExtractFieldOp::new(op).set_attr_index(ctx, FieldIndexAttr(0));
        op
    };
    assert!(
        MirExtractFieldOp::new(extract_erased(&mut ctx, erased_const_ptr.into()))
            .verify(&ctx)
            .is_ok(),
        "an immutable Erased slice extracts an immutable Erased data pointer"
    );
    assert!(
        MirExtractFieldOp::new(extract_erased(&mut ctx, erased_ptr.into()))
            .verify(&ctx)
            .is_err(),
        "an immutable Erased slice cannot manufacture writable data-pointer evidence"
    );

    let extract_erased_mut = |ctx: &mut Context, result_ty| {
        let op = Operation::new(
            ctx,
            MirExtractFieldOp::get_concrete_op_info(),
            vec![result_ty],
            vec![erased_mut_slice_value],
            vec![],
            0,
        );
        MirExtractFieldOp::new(op).set_attr_index(ctx, FieldIndexAttr(0));
        op
    };
    assert!(
        MirExtractFieldOp::new(extract_erased_mut(&mut ctx, erased_ptr.into()))
            .verify(&ctx)
            .is_ok(),
        "a mutable Erased slice retains writable data-pointer evidence"
    );

    let immutable_slice_reborrow = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_slice.into()],
        vec![erased_slice_value],
        vec![],
        0,
    );
    let immutable_slice_reborrow = MirCastOp::new(immutable_slice_reborrow);
    immutable_slice_reborrow.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    immutable_slice_reborrow
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    assert!(
        immutable_slice_reborrow.verify(&ctx).is_err(),
        "an immutable Erased slice cannot be reborrowed as UniqueRef"
    );

    let mutable_slice_reborrow = Operation::new(
        &mut ctx,
        MirCastOp::get_concrete_op_info(),
        vec![unique_slice.into()],
        vec![erased_mut_slice_value],
        vec![],
        0,
    );
    let mutable_slice_reborrow = MirCastOp::new(mutable_slice_reborrow);
    mutable_slice_reborrow.set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
    mutable_slice_reborrow
        .set_pointer_kind_authority(&mut ctx, MirPointerKindAuthorityAttr::Reborrow);
    assert!(
        mutable_slice_reborrow.verify(&ctx).is_ok(),
        "a mutable Erased slice may establish UniqueRef at a real reborrow boundary"
    );
}

#[test]
fn test_mir_arithmetic_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into(), i32_ty.into()]);
    let lhs = block.deref(&ctx).get_argument(0);

    let check_bin_op = |opid: (
        fn(pliron::context::Ptr<pliron::operation::Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
                        name: &str| {
        let mut context = Context::new();
        dialect_mir::register(&mut context);
        let ty = IntegerType::get(&context, 32, Signedness::Signed);
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

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signed);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);

    // 1. MirConstantOp
    let i32_signless = IntegerType::get(&ctx, 32, Signedness::Signless);
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
    let i64_signless = IntegerType::get(&ctx, 64, Signedness::Signless);
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
        let ty = IntegerType::get(&context, 32, Signedness::Signed);
        let res_ty = IntegerType::get(&context, 1, Signedness::Signless);
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
    let i8_ty = IntegerType::get(&context, 8, Signedness::Signed);
    let i32_ty = IntegerType::get(&context, 32, Signedness::Signed);
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

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let func_ty = FunctionType::get(&ctx, vec![i32_ty.into()], vec![]);
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

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
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

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
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
fn test_mir_store_volatile_is_not_promotable() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
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
    let mir_store = MirStoreOp::new(op);
    let alloc_info = AllocInfo {
        ptr: ptr_val,
        ty: i32_ty.into(),
    };

    assert!(!mir_store.is_volatile(&ctx));
    match mir_store.promotion_kind(&ctx, &alloc_info) {
        PromotableOpKind::Store(stored) => assert!(stored == val),
        _ => panic!("non-volatile store should be promotable"),
    }

    mir_store.set_volatile(&mut ctx, true);

    assert!(mir_store.is_volatile(&ctx));
    assert!(matches!(
        mir_store.promotion_kind(&ctx, &alloc_info),
        PromotableOpKind::NonPromotableUse
    ));
}

#[test]
fn test_mir_global_alloc_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let f32_ty = FP32Type::get(&ctx);

    // Helper: build a MirGlobalAllocOp whose result pointer is in `ptr_ty`
    // address space, with valid attributes.
    let build = |ctx: &mut Context, ptr_ty: pliron::r#type::TypedHandle<MirPtrType>| {
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

    let ptr_unique_global = MirPtrType::get_with_kind(
        &mut ctx,
        f32_ty.into(),
        true,
        dialect_mir::types::address_space::GLOBAL,
        MirPointerKind::UniqueRef,
    );
    assert!(
        build(&mut ctx, ptr_unique_global).verify(&ctx).is_err(),
        "a storage allocation cannot directly claim UniqueRef"
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

#[test]
fn test_mir_set_discriminant_verify() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signed);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
    let unit = |name: &str| EnumVariant::unit(name.to_string());

    let enum_ty = MirEnumType::get(
        &mut ctx,
        "DeviceState".to_string(),
        i8_ty.into(),
        vec![0, 1],
        vec![
            unit("Empty"),
            EnumVariant::new("Full".to_string(), vec![i32_ty.into()]),
        ],
    );

    let enum_ptr_ty = MirPtrType::get_generic(&mut ctx, enum_ty.into(), true);
    let blk = BasicBlock::new(&mut ctx, None, vec![enum_ptr_ty.into(), i8_ty.into()]);
    let enum_ptr = blk.deref(&ctx).get_argument(0);
    let discr_val = blk.deref(&ctx).get_argument(1);

    // Malformed IR must be diagnosed without panicking, regardless of whether
    // the generated operand-count interface happens to run first.
    let op_no_operands = Operation::new(
        &mut ctx,
        MirSetDiscriminantOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    let no_operands = MirSetDiscriminantOp::new(op_no_operands);
    no_operands.set_attr_set_discriminant_variant_index(&ctx, VariantIndexAttr(0));
    assert!(no_operands.verify(&ctx).is_err(), "zero operands rejected");

    let op_extra_operand = Operation::new(
        &mut ctx,
        MirSetDiscriminantOp::get_concrete_op_info(),
        vec![],
        vec![enum_ptr, discr_val],
        vec![],
        0,
    );
    let extra_operand = MirSetDiscriminantOp::new(op_extra_operand);
    extra_operand.set_attr_set_discriminant_variant_index(&ctx, VariantIndexAttr(0));
    assert!(
        extra_operand.verify(&ctx).is_err(),
        "extra operand rejected"
    );

    // Valid: pointer to enum plus the semantic target variant attribute.
    let op_valid = Operation::new(
        &mut ctx,
        MirSetDiscriminantOp::get_concrete_op_info(),
        vec![],
        vec![enum_ptr],
        vec![],
        0,
    );
    let valid = MirSetDiscriminantOp::new(op_valid);
    valid.set_attr_set_discriminant_variant_index(&ctx, VariantIndexAttr(1));
    valid.set_attr_set_discriminant_enum_ty(&ctx, TypeAttr::new(enum_ty.into()));
    assert!(valid.verify(&ctx).is_ok(), "Valid set_discriminant");

    // Invalid: first operand is not a pointer.
    let op_bad_ptr = Operation::new(
        &mut ctx,
        MirSetDiscriminantOp::get_concrete_op_info(),
        vec![],
        vec![discr_val],
        vec![],
        0,
    );
    assert!(
        MirSetDiscriminantOp::new(op_bad_ptr).verify(&ctx).is_err(),
        "Non-pointer enum operand rejected"
    );

    // Invalid: SetDiscriminant writes memory and therefore cannot accept an
    // immutable pointer even when the pointee is the right enum.
    let immutable_ptr_ty = MirPtrType::get_generic(&mut ctx, enum_ty.into(), false);
    let immutable_block = BasicBlock::new(&mut ctx, None, vec![immutable_ptr_ty.into()]);
    let immutable_ptr = immutable_block.deref(&ctx).get_argument(0);
    let op_immutable = Operation::new(
        &mut ctx,
        MirSetDiscriminantOp::get_concrete_op_info(),
        vec![],
        vec![immutable_ptr],
        vec![],
        0,
    );
    let immutable = MirSetDiscriminantOp::new(op_immutable);
    immutable.set_attr_set_discriminant_variant_index(&ctx, VariantIndexAttr(0));
    assert!(
        immutable.verify(&ctx).is_err(),
        "immutable pointer rejected"
    );

    // Invalid: pointer does not point to an enum.
    let i32_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let blk_i32 = BasicBlock::new(&mut ctx, None, vec![i32_ptr_ty.into(), i8_ty.into()]);
    let i32_ptr = blk_i32.deref(&ctx).get_argument(0);
    let op_bad_pointee = Operation::new(
        &mut ctx,
        MirSetDiscriminantOp::get_concrete_op_info(),
        vec![],
        vec![i32_ptr],
        vec![],
        0,
    );
    let bad_pointee = MirSetDiscriminantOp::new(op_bad_pointee);
    bad_pointee.set_attr_set_discriminant_variant_index(&ctx, VariantIndexAttr(0));
    assert!(
        bad_pointee.verify(&ctx).is_err(),
        "Non-enum pointee rejected"
    );

    // Invalid: target attribute is required.
    let op_missing_target = Operation::new(
        &mut ctx,
        MirSetDiscriminantOp::get_concrete_op_info(),
        vec![],
        vec![enum_ptr],
        vec![],
        0,
    );
    assert!(
        MirSetDiscriminantOp::new(op_missing_target)
            .verify(&ctx)
            .is_err(),
        "Missing target rejected"
    );

    // Invalid: target index is out of bounds.
    let op_oob = Operation::new(
        &mut ctx,
        MirSetDiscriminantOp::get_concrete_op_info(),
        vec![],
        vec![enum_ptr],
        vec![],
        0,
    );
    let oob = MirSetDiscriminantOp::new(op_oob);
    oob.set_attr_set_discriminant_variant_index(&ctx, VariantIndexAttr(2));
    assert!(oob.verify(&ctx).is_err(), "Out-of-bounds target rejected");
}

#[test]
fn test_mir_enum_ops_malformed_arity_is_diagnostic() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let unit_enum = MirEnumType::get(
        &mut ctx,
        "Unit".into(),
        i8_ty.into(),
        vec![0],
        vec![EnumVariant::unit("Only".into())],
    );
    let payload_enum = MirEnumType::get(
        &mut ctx,
        "Payload".into(),
        i8_ty.into(),
        vec![0],
        vec![EnumVariant::new("Only".into(), vec![i32_ty.into()])],
    );
    let block = BasicBlock::new(&mut ctx, None, vec![unit_enum.into(), payload_enum.into()]);
    let unit_value = block.deref(&ctx).get_argument(0);
    let payload_value = block.deref(&ctx).get_argument(1);

    let construct_no_result = Operation::new(
        &mut ctx,
        MirConstructEnumOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    let construct_no_result = MirConstructEnumOp::new(construct_no_result);
    construct_no_result.set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(0));
    assert!(construct_no_result.verify(&ctx).is_err());

    let construct_extra_result = Operation::new(
        &mut ctx,
        MirConstructEnumOp::get_concrete_op_info(),
        vec![unit_enum.into(), unit_enum.into()],
        vec![],
        vec![],
        0,
    );
    let construct_extra_result = MirConstructEnumOp::new(construct_extra_result);
    construct_extra_result.set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(0));
    assert!(construct_extra_result.verify(&ctx).is_err());

    let get_empty = Operation::new(
        &mut ctx,
        MirGetDiscriminantOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    assert!(MirGetDiscriminantOp::new(get_empty).verify(&ctx).is_err());
    let get_extra = Operation::new(
        &mut ctx,
        MirGetDiscriminantOp::get_concrete_op_info(),
        vec![i8_ty.into(), i8_ty.into()],
        vec![unit_value, unit_value],
        vec![],
        0,
    );
    assert!(MirGetDiscriminantOp::new(get_extra).verify(&ctx).is_err());

    let payload_empty = Operation::new(
        &mut ctx,
        MirEnumPayloadOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    let payload_empty = MirEnumPayloadOp::new(payload_empty);
    payload_empty.set_attr_payload_variant_index(&ctx, VariantIndexAttr(0));
    payload_empty.set_attr_payload_field_index(&ctx, FieldIndexAttr(0));
    assert!(payload_empty.verify(&ctx).is_err());

    let payload_extra = Operation::new(
        &mut ctx,
        MirEnumPayloadOp::get_concrete_op_info(),
        vec![i32_ty.into(), i32_ty.into()],
        vec![payload_value, payload_value],
        vec![],
        0,
    );
    let payload_extra = MirEnumPayloadOp::new(payload_extra);
    payload_extra.set_attr_payload_variant_index(&ctx, VariantIndexAttr(0));
    payload_extra.set_attr_payload_field_index(&ctx, FieldIndexAttr(0));
    assert!(payload_extra.verify(&ctx).is_err());
}

#[test]
fn test_mir_enum_payload_rejects_uninhabited_variant() {
    use dialect_mir::types::{EnumCarrierKind, EnumEncoding, EnumLayoutKind};

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let enum_ty = MirEnumType::get_with_encoding(
        &mut ctx,
        "HasImpossibleField".into(),
        i8_ty.into(),
        vec![0, 1],
        vec![
            EnumVariant::unit("Live".into()),
            // An uninhabited variant's unused physical offsets need not fit
            // the object. The verifier must stop them reaching lowering.
            EnumVariant::new_with_layout(
                "Impossible".into(),
                vec![i32_ty.into()],
                vec![64],
                vec![4],
            ),
        ],
        EnumEncoding {
            tag_offset: 0,
            total_size: 1,
            abi_align: 1,
            layout_kind: EnumLayoutKind::Direct,
            carrier_kind: EnumCarrierKind::Integer,
            carrier_width: 8,
            variant_inhabited: vec![1, 0],
            ..EnumEncoding::default()
        },
    );
    assert!(enum_ty.verify(&ctx).is_ok());

    let block = BasicBlock::new(&mut ctx, None, vec![enum_ty.into()]);
    let value = block.deref(&ctx).get_argument(0);
    let op = Operation::new(
        &mut ctx,
        MirEnumPayloadOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![value],
        vec![],
        0,
    );
    let payload = MirEnumPayloadOp::new(op);
    payload.set_attr_payload_variant_index(&ctx, VariantIndexAttr(1));
    payload.set_attr_payload_field_index(&ctx, FieldIndexAttr(0));
    assert!(payload.verify(&ctx).is_err());
}

/// Reachable-but-dead MIR (e.g. the residual arms of `array::try_from_fn`)
/// can name uninhabited variants. The importer must lower such reads and
/// constructions to typed undefs; if it ever emits the real ops again, these
/// verifiers are the loud stop.
#[test]
fn test_uninhabited_enum_construct_and_discriminant_fail_verification() {
    use dialect_mir::types::{EnumCarrierKind, EnumEncoding, EnumLayoutKind};

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);

    // Constructing an uninhabited variant must fail verification.
    let partial = MirEnumType::get_with_encoding(
        &mut ctx,
        "HasImpossibleVariant".into(),
        i8_ty.into(),
        vec![0, 1],
        vec![
            EnumVariant::unit("Live".into()),
            EnumVariant::new_with_layout(
                "Impossible".into(),
                vec![i32_ty.into()],
                vec![0],
                vec![4],
            ),
        ],
        EnumEncoding {
            tag_offset: 0,
            total_size: 8,
            abi_align: 4,
            layout_kind: EnumLayoutKind::Direct,
            carrier_kind: EnumCarrierKind::Integer,
            carrier_width: 8,
            variant_inhabited: vec![1, 0],
            ..EnumEncoding::default()
        },
    );
    let block = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    let field = block.deref(&ctx).get_argument(0);
    let construct = Operation::new(
        &mut ctx,
        MirConstructEnumOp::get_concrete_op_info(),
        vec![partial.into()],
        vec![field],
        vec![],
        0,
    );
    let construct = MirConstructEnumOp::new(construct);
    construct.set_attr_construct_enum_variant_index(&ctx, VariantIndexAttr(1));
    assert!(construct.verify(&ctx).is_err());

    // Reading the discriminant of a fully uninhabited enum must fail
    // verification.
    let never = MirEnumType::get_with_encoding(
        &mut ctx,
        "Never".into(),
        i8_ty.into(),
        vec![],
        vec![],
        EnumEncoding {
            tag_offset: 0,
            total_size: 0,
            abi_align: 1,
            layout_kind: EnumLayoutKind::Empty,
            carrier_kind: EnumCarrierKind::None,
            carrier_width: 0,
            variant_inhabited: vec![],
            ..EnumEncoding::default()
        },
    );
    assert!(never.verify(&ctx).is_ok());
    let never_block = BasicBlock::new(&mut ctx, None, vec![never.into()]);
    let never_value = never_block.deref(&ctx).get_argument(0);
    let get_discriminant = Operation::new(
        &mut ctx,
        MirGetDiscriminantOp::get_concrete_op_info(),
        vec![i8_ty.into()],
        vec![never_value],
        vec![],
        0,
    );
    let get_discriminant = MirGetDiscriminantOp::new(get_discriminant);
    assert!(get_discriminant.verify(&ctx).is_err());
}

#[test]
fn test_mir_field_addr_tuple_pointee_verify() {
    // `(u8, u32)` laid out the way rustc actually places it: the u32 field
    // first in memory for alignment, so declaration index 0 (`u8`) lands at
    // byte offset 4 and declaration index 1 (`u32`) lands at byte offset 0.
    // `field_addr`'s `field_index` attribute is a DECLARATION index (it names
    // `.0`/`.1` as written), so this test only passes if the op resolves the
    // field's type through `MirTupleType::get_types()` (declaration order)
    // rather than assuming identity with memory order.
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);

    let tuple_ty = MirTupleType::get_with_layout(
        &mut ctx,
        vec![u8_ty.into(), u32_ty.into()],
        vec![1, 0],
        vec![4, 0],
        8,
        4,
    );

    let tuple_ptr_ty = MirPtrType::get_generic(&mut ctx, tuple_ty.into(), false);
    let blk = BasicBlock::new(&mut ctx, None, vec![tuple_ptr_ty.into()]);
    let tuple_ptr = blk.deref(&ctx).get_argument(0);

    let u8_ptr_ty = MirPtrType::get_generic(&mut ctx, u8_ty.into(), false);
    let op_field0 = Operation::new(
        &mut ctx,
        MirFieldAddrOp::get_concrete_op_info(),
        vec![u8_ptr_ty.into()],
        vec![tuple_ptr],
        vec![],
        0,
    );
    let field0 = MirFieldAddrOp::new(op_field0);
    field0.set_attr_field_index(&ctx, FieldIndexAttr(0));
    field0.set_attr_aggregate_ty(&ctx, TypeAttr::new(tuple_ty.into()));
    assert!(
        field0.verify(&ctx).is_ok(),
        "tuple field 0 (u8) address accepted"
    );

    let u32_ptr_ty = MirPtrType::get_generic(&mut ctx, u32_ty.into(), false);
    let op_field1 = Operation::new(
        &mut ctx,
        MirFieldAddrOp::get_concrete_op_info(),
        vec![u32_ptr_ty.into()],
        vec![tuple_ptr],
        vec![],
        0,
    );
    let field1 = MirFieldAddrOp::new(op_field1);
    field1.set_attr_field_index(&ctx, FieldIndexAttr(1));
    field1.set_attr_aggregate_ty(&ctx, TypeAttr::new(tuple_ty.into()));
    assert!(
        field1.verify(&ctx).is_ok(),
        "tuple field 1 (u32) address accepted"
    );

    // Result pointee type must match the DECLARED field type, not whatever
    // sits at that byte offset: pointing field 0's result at u32 (field 1's
    // type) must be rejected even though both are in-bounds indices.
    let op_wrong_result_ty = Operation::new(
        &mut ctx,
        MirFieldAddrOp::get_concrete_op_info(),
        vec![u32_ptr_ty.into()],
        vec![tuple_ptr],
        vec![],
        0,
    );
    let wrong_result_ty = MirFieldAddrOp::new(op_wrong_result_ty);
    wrong_result_ty.set_attr_field_index(&ctx, FieldIndexAttr(0));
    wrong_result_ty.set_attr_aggregate_ty(&ctx, TypeAttr::new(tuple_ty.into()));
    assert!(
        wrong_result_ty.verify(&ctx).is_err(),
        "result pointee type mismatch rejected"
    );

    let op_out_of_bounds = Operation::new(
        &mut ctx,
        MirFieldAddrOp::get_concrete_op_info(),
        vec![u8_ptr_ty.into()],
        vec![tuple_ptr],
        vec![],
        0,
    );
    let out_of_bounds = MirFieldAddrOp::new(op_out_of_bounds);
    out_of_bounds.set_attr_field_index(&ctx, FieldIndexAttr(2));
    out_of_bounds.set_attr_aggregate_ty(&ctx, TypeAttr::new(tuple_ty.into()));
    assert!(
        out_of_bounds.verify(&ctx).is_err(),
        "out-of-bounds tuple field index rejected"
    );
}

#[test]
fn test_mir_field_addr_tuple_pointee_store_verify() {
    // The WRITE side of the tuple-pointee unlock: `t.1 = x` / `arr[i].1 = x`
    // lower to `mir.field_addr` + `mir.store` through the field's address, so
    // a tuple-pointee field address used as a store destination must pass
    // verification too. Same reordered `(u8, u32)` layout as above (the u32
    // field first in memory), so the store type-checks against the DECLARED
    // field type, not whatever occupies that memory slot.
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);

    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);

    let tuple_ty = MirTupleType::get_with_layout(
        &mut ctx,
        vec![u8_ty.into(), u32_ty.into()],
        vec![1, 0],
        vec![4, 0],
        8,
        4,
    );

    let tuple_ptr_ty = MirPtrType::get_generic(&mut ctx, tuple_ty.into(), false);
    let blk = BasicBlock::new(
        &mut ctx,
        None,
        vec![tuple_ptr_ty.into(), u32_ty.into(), u8_ty.into()],
    );
    let tuple_ptr = blk.deref(&ctx).get_argument(0);
    let u32_val = blk.deref(&ctx).get_argument(1);
    let u8_val = blk.deref(&ctx).get_argument(2);

    // `.1 = x`: address declaration field 1 (u32, memory slot 0) and store a
    // u32 through it.
    let u32_ptr_ty = MirPtrType::get_generic(&mut ctx, u32_ty.into(), false);
    let op_field1 = Operation::new(
        &mut ctx,
        MirFieldAddrOp::get_concrete_op_info(),
        vec![u32_ptr_ty.into()],
        vec![tuple_ptr],
        vec![],
        0,
    );
    let field1 = MirFieldAddrOp::new(op_field1);
    field1.set_attr_field_index(&ctx, FieldIndexAttr(1));
    field1.set_attr_aggregate_ty(&ctx, TypeAttr::new(tuple_ty.into()));
    assert!(
        field1.verify(&ctx).is_ok(),
        "tuple field 1 (u32) address accepted as a store destination"
    );
    let field1_ptr = op_field1.deref(&ctx).get_result(0);

    let op_store = Operation::new(
        &mut ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![field1_ptr, u32_val],
        vec![],
        0,
    );
    assert!(
        MirStoreOp::new(op_store).verify(&ctx).is_ok(),
        "store through a tuple field address verifies"
    );

    // The stored value must match the DECLARED field type (`u32` for `.1`),
    // not the type of the field sharing the tuple: a u8 store through the
    // `.1` pointer is a type mismatch.
    let op_store_wrong_ty = Operation::new(
        &mut ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![field1_ptr, u8_val],
        vec![],
        0,
    );
    assert!(
        MirStoreOp::new(op_store_wrong_ty).verify(&ctx).is_err(),
        "store of a mismatched value type through a tuple field address rejected"
    );
}
