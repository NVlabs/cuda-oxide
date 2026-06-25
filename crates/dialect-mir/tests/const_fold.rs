/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Unit tests for the `dialect-mir` constant-folding interfaces (see
//! `src/const_fold.rs`).
//!
//! `ConstFoldInterface::check_fold` takes the operands' known constant values
//! as a parameter and returns the folded result value(s), so we can exercise
//! each op's fold rule directly: build the op, hand `check_fold` two integer
//! attributes, and check the result. `BranchOpFoldInterface::check_fold` is
//! tested the same way (constant condition -> which successor stays feasible).
//!
//! These assert the *fold rules* in isolation; `unroll_smoke` and the
//! mir-transforms unroll-pass test cover the end-to-end `sccp` path.

use std::num::NonZero;

use dialect_mir::ops::{
    MirAddOp, MirBitAndOp, MirBitOrOp, MirBitXorOp, MirCondBranchOp, MirConstantOp, MirDivOp,
    MirEqOp, MirFuncOp, MirGeOp, MirGtOp, MirLeOp, MirLtOp, MirMulOp, MirNeOp, MirRemOp, MirShlOp,
    MirShrOp, MirSubOp,
};
use pliron::attribute::AttrObj;
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::{IntegerAttr, TypeAttr};
use pliron::builtin::op_interfaces::OperandSegmentInterface;
use pliron::builtin::types::{FunctionType, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::op::{Op, op_cast};
use pliron::operation::Operation;
use pliron::opts::constants::{BranchOpFoldInterface, ConstFoldInterface};
use pliron::region::Region;
use pliron::r#type::TypedHandle;
use pliron::value::Value;

/// A fresh context with the `mir` dialect registered.
fn ctx() -> Context {
    let mut c = Context::new();
    dialect_mir::register(&mut c);
    c
}

/// `fn foo()` with one entry block; returns `(region, entry)`.
fn func_with_entry(ctx: &mut Context) -> (Ptr<Region>, Ptr<BasicBlock>) {
    let func_ty = FunctionType::get(ctx, vec![], vec![]);
    let op = Operation::new(
        ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func = MirFuncOp::new(ctx, op, TypeAttr::new(func_ty.into()));
    let region = func.get_operation().deref(ctx).get_region(0);
    let entry = BasicBlock::new(ctx, None, vec![]);
    entry.insert_at_front(region, ctx);
    (region, entry)
}

fn int_ty(ctx: &mut Context, width: u32, sign: Signedness) -> TypedHandle<IntegerType> {
    IntegerType::get(ctx, width, sign)
}

/// An `IntegerAttr` of `ty` holding `v`, as an `AttrObj` (what `check_fold` takes).
fn iattr(ctx: &Context, ty: TypedHandle<IntegerType>, v: i64) -> AttrObj {
    let width = ty.deref(ctx).width() as usize;
    IntegerAttr::new(
        ty,
        pliron::utils::apint::APInt::from_i64(v, NonZero::new(width).unwrap()),
    )
    .into()
}

/// Append an `i32` constant (its value is irrelevant: the fold reads the attrs
/// we pass to `check_fold`, not the operand IR) and return its SSA value, to use
/// as a placeholder operand when building the op under test.
fn placeholder(ctx: &mut Context, blk: Ptr<BasicBlock>) -> Value {
    let ty = int_ty(ctx, 32, Signedness::Signless);
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
            pliron::utils::apint::APInt::from_i64(0, NonZero::new(32).unwrap()),
        ),
    );
    op.insert_at_back(blk, ctx);
    op.deref(ctx).get_result(0)
}

/// Build a two-operand op of `$opty` (result type `$res_ty`), then fold it with
/// operand attrs `$a`,`$b`. Returns the folded result as `Option<i128>` (`None`
/// if it refused to fold). The attrs are bound first so their immutable borrow
/// of `ctx` is released before the op-building mutable borrow.
macro_rules! fold_bin {
    ($ctx:expr, $blk:expr, $opty:ty, $res_ty:expr, $a:expr, $b:expr) => {{
        let a_attr = $a;
        let b_attr = $b;
        let res_ty: pliron::r#type::TypeHandle = $res_ty.into();
        let lv = placeholder($ctx, $blk);
        let rv = placeholder($ctx, $blk);
        let op = Operation::new(
            $ctx,
            <$opty>::get_concrete_op_info(),
            vec![res_ty],
            vec![lv, rv],
            vec![],
            0,
        );
        op.insert_at_back($blk, $ctx);
        let op_dyn = Operation::get_op_dyn(op, $ctx);
        let fold = op_cast::<dyn ConstFoldInterface>(op_dyn.as_ref())
            .expect("op implements ConstFoldInterface");
        let out = fold.check_fold($ctx, &[Some(a_attr), Some(b_attr)]);
        assert_eq!(out.len(), 1, "a binary op has exactly one result");
        match &out[0] {
            Some(attr) => Some(
                attr.downcast_ref::<IntegerAttr>()
                    .unwrap()
                    .value()
                    .to_i128(),
            ),
            None => None,
        }
    }};
}

/// Like [`fold_bin`], but reads the `i1` result of a comparison as a `bool`
/// (the raw bit, not a sign-extended `to_i128`: a 1-bit `1` is `-1` signed).
macro_rules! fold_cmp {
    ($ctx:expr, $blk:expr, $opty:ty, $a:expr, $b:expr) => {{
        let a_attr = $a;
        let b_attr = $b;
        let i1t = int_ty($ctx, 1, Signedness::Signless);
        let lv = placeholder($ctx, $blk);
        let rv = placeholder($ctx, $blk);
        let op = Operation::new(
            $ctx,
            <$opty>::get_concrete_op_info(),
            vec![i1t.into()],
            vec![lv, rv],
            vec![],
            0,
        );
        op.insert_at_back($blk, $ctx);
        let op_dyn = Operation::get_op_dyn(op, $ctx);
        let fold = op_cast::<dyn ConstFoldInterface>(op_dyn.as_ref())
            .expect("op implements ConstFoldInterface");
        let out = fold.check_fold($ctx, &[Some(a_attr), Some(b_attr)]);
        assert_eq!(out.len(), 1, "a comparison has exactly one result");
        out[0].as_ref().map(|attr| {
            !attr
                .downcast_ref::<IntegerAttr>()
                .unwrap()
                .value()
                .is_zero()
        })
    }};
}

#[test]
fn arithmetic_and_bitwise_fold() {
    let mut ctx = ctx();
    let (_r, b) = func_with_entry(&mut ctx);
    let i32t = int_ty(&mut ctx, 32, Signedness::Signless);
    let (a, c) = (12, 4);

    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirAddOp,
            i32t,
            iattr(&ctx, i32t, a),
            iattr(&ctx, i32t, c)
        ),
        Some(16)
    );
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirSubOp,
            i32t,
            iattr(&ctx, i32t, a),
            iattr(&ctx, i32t, c)
        ),
        Some(8)
    );
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirMulOp,
            i32t,
            iattr(&ctx, i32t, a),
            iattr(&ctx, i32t, c)
        ),
        Some(48)
    );
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirBitAndOp,
            i32t,
            iattr(&ctx, i32t, a),
            iattr(&ctx, i32t, c)
        ),
        Some(4)
    );
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirBitOrOp,
            i32t,
            iattr(&ctx, i32t, a),
            iattr(&ctx, i32t, c)
        ),
        Some(12)
    );
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirBitXorOp,
            i32t,
            iattr(&ctx, i32t, a),
            iattr(&ctx, i32t, c)
        ),
        Some(8)
    );
}

#[test]
fn shifts_fold_and_respect_signedness() {
    let mut ctx = ctx();
    let (_r, b) = func_with_entry(&mut ctx);
    let signless = int_ty(&mut ctx, 32, Signedness::Signless);
    let unsigned = int_ty(&mut ctx, 32, Signedness::Unsigned);
    let signed = int_ty(&mut ctx, 32, Signedness::Signed);

    // 1 << 4 == 16
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirShlOp,
            signless,
            iattr(&ctx, signless, 1),
            iattr(&ctx, signless, 4)
        ),
        Some(16)
    );

    // logical shift: 16u32 >> 2 == 4
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirShrOp,
            unsigned,
            iattr(&ctx, unsigned, 16),
            iattr(&ctx, unsigned, 2)
        ),
        Some(4)
    );

    // arithmetic shift: (-16i32) >> 2 == -4 (sign bit copied in)
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirShrOp,
            signed,
            iattr(&ctx, signed, -16),
            iattr(&ctx, signed, 2)
        ),
        Some(-4)
    );
}

#[test]
fn div_rem_fold_and_refuse_div_by_zero() {
    let mut ctx = ctx();
    let (_r, b) = func_with_entry(&mut ctx);
    let unsigned = int_ty(&mut ctx, 32, Signedness::Unsigned);
    let signed = int_ty(&mut ctx, 32, Signedness::Signed);

    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirDivOp,
            unsigned,
            iattr(&ctx, unsigned, 17),
            iattr(&ctx, unsigned, 5)
        ),
        Some(3)
    );
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirRemOp,
            unsigned,
            iattr(&ctx, unsigned, 17),
            iattr(&ctx, unsigned, 5)
        ),
        Some(2)
    );

    // signed division truncates toward zero: -17 / 5 == -3, -17 % 5 == -2
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirDivOp,
            signed,
            iattr(&ctx, signed, -17),
            iattr(&ctx, signed, 5)
        ),
        Some(-3)
    );
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirRemOp,
            signed,
            iattr(&ctx, signed, -17),
            iattr(&ctx, signed, 5)
        ),
        Some(-2)
    );

    // division by zero is a Rust panic, never folded.
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirDivOp,
            unsigned,
            iattr(&ctx, unsigned, 5),
            iattr(&ctx, unsigned, 0)
        ),
        None
    );
    assert_eq!(
        fold_bin!(
            &mut ctx,
            b,
            MirRemOp,
            unsigned,
            iattr(&ctx, unsigned, 5),
            iattr(&ctx, unsigned, 0)
        ),
        None
    );
}

#[test]
fn comparisons_fold_to_i1_and_respect_signedness() {
    let mut ctx = ctx();
    let (_r, b) = func_with_entry(&mut ctx);
    let signed = int_ty(&mut ctx, 32, Signedness::Signed);
    let unsigned = int_ty(&mut ctx, 32, Signedness::Unsigned);

    // equality / inequality / ordering
    assert_eq!(
        fold_cmp!(
            &mut ctx,
            b,
            MirEqOp,
            iattr(&ctx, signed, 5),
            iattr(&ctx, signed, 5)
        ),
        Some(true)
    );
    assert_eq!(
        fold_cmp!(
            &mut ctx,
            b,
            MirNeOp,
            iattr(&ctx, signed, 5),
            iattr(&ctx, signed, 6)
        ),
        Some(true)
    );
    assert_eq!(
        fold_cmp!(
            &mut ctx,
            b,
            MirLeOp,
            iattr(&ctx, signed, 5),
            iattr(&ctx, signed, 5)
        ),
        Some(true)
    );
    assert_eq!(
        fold_cmp!(
            &mut ctx,
            b,
            MirGeOp,
            iattr(&ctx, signed, 5),
            iattr(&ctx, signed, 6)
        ),
        Some(false)
    );
    assert_eq!(
        fold_cmp!(
            &mut ctx,
            b,
            MirGtOp,
            iattr(&ctx, signed, 7),
            iattr(&ctx, signed, 6)
        ),
        Some(true)
    );

    // `-1 < 1` is true signed, but false unsigned (where -1 is the max value).
    assert_eq!(
        fold_cmp!(
            &mut ctx,
            b,
            MirLtOp,
            iattr(&ctx, signed, -1),
            iattr(&ctx, signed, 1)
        ),
        Some(true)
    );
    assert_eq!(
        fold_cmp!(
            &mut ctx,
            b,
            MirLtOp,
            iattr(&ctx, unsigned, -1),
            iattr(&ctx, unsigned, 1)
        ),
        Some(false)
    );
}

#[test]
fn cond_br_folds_to_the_taken_successor() {
    let mut ctx = ctx();
    let (region, entry) = func_with_entry(&mut ctx);
    let i1t = int_ty(&mut ctx, 1, Signedness::Signless);

    let t = BasicBlock::new(&mut ctx, None, vec![]);
    let f = BasicBlock::new(&mut ctx, None, vec![]);
    t.insert_at_back(region, &ctx);
    f.insert_at_back(region, &ctx);

    let cond = placeholder(&mut ctx, entry);
    let (flat, segs) = MirCondBranchOp::compute_segment_sizes(vec![vec![cond], vec![], vec![]]);
    let op = Operation::new(
        &mut ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        flat,
        vec![t, f],
        0,
    );
    Operation::get_op::<MirCondBranchOp>(op, &ctx)
        .unwrap()
        .set_operand_segment_sizes(&ctx, segs);
    op.insert_at_back(entry, &ctx);

    let op_dyn = Operation::get_op_dyn(op, &ctx);
    let branch = op_cast::<dyn BranchOpFoldInterface>(op_dyn.as_ref())
        .expect("cond_br implements BranchOpFoldInterface");

    // condition true -> only the true successor (index 0) stays feasible.
    let true_succs = branch.check_fold(&ctx, &[Some(iattr(&ctx, i1t, 1))]);
    assert_eq!(true_succs, vec![t]);

    // condition false -> only the false successor (index 1).
    let false_succs = branch.check_fold(&ctx, &[Some(iattr(&ctx, i1t, 0))]);
    assert_eq!(false_succs, vec![f]);
}
