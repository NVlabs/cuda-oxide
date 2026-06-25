/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Temporary repro harness: run sccp / simplify_cfg / dce (each individually)
//! on a tiny hand-built MIR function, to find which pass infinite-loops once
//! dialect-mir implements ConstFoldInterface / BranchOpFoldInterface. Run with
//! `--nocapture --test-threads=1` under a `timeout`; the last printed marker
//! without its matching "done" is the culprit.

use dialect_mir::ops::{
    MirAddOp, MirCondBranchOp, MirConstantOp, MirFuncOp, MirGotoOp, MirReturnOp,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, TypeAttr},
        op_interfaces::OperandSegmentInterface,
        types::{FunctionType, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    utils::apint::APInt,
    value::Value,
};
use std::num::NonZero;

fn empty_func(ctx: &mut Context) -> (MirFuncOp, Ptr<pliron::basic_block::BasicBlock>) {
    let func_ty = FunctionType::get(ctx, vec![], vec![]);
    let func_ty_attr = TypeAttr::new(func_ty.into());
    let op = Operation::new(
        ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func = MirFuncOp::new(ctx, op, func_ty_attr);
    let region = func.get_operation().deref(ctx).get_region(0);
    let entry = BasicBlock::new(ctx, None, vec![]);
    entry.insert_at_front(region, ctx);
    (func, entry)
}

fn int_const(
    ctx: &mut Context,
    ty: pliron::r#type::TypedHandle<IntegerType>,
    v: i64,
    width: usize,
) -> (Ptr<Operation>, Value) {
    let op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![ty.into()],
        vec![],
        vec![],
        0,
    );
    let apint = APInt::from_i64(v, NonZero::new(width).unwrap());
    MirConstantOp::new(op).set_attr_value(ctx, IntegerAttr::new(ty, apint));
    let val = op.deref(ctx).get_result(0);
    (op, val)
}

/// `s = add 2, 3` in a function; run sccp only.
#[test]
fn repro_sccp() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signed);

    let (func, entry) = empty_func(&mut ctx);
    let (c2, c2v) = int_const(&mut ctx, i32_ty, 2, 32);
    let (c3, c3v) = int_const(&mut ctx, i32_ty, 3, 32);
    c2.insert_at_back(entry, &ctx);
    c3.insert_at_back(entry, &ctx);
    let add = Operation::new(
        &mut ctx,
        MirAddOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![c2v, c3v],
        vec![],
        0,
    );
    add.insert_at_back(entry, &ctx);
    let ret = Operation::new(
        &mut ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret.insert_at_back(entry, &ctx);

    eprintln!(">>> START sccp");
    pliron::opts::constants::sccp::sccp(func.get_operation(), &mut ctx).unwrap();
    eprintln!(">>> DONE sccp");
}

/// `cond_br <const 1>, t, f` collapsing; run simplify_cfg only.
#[test]
fn repro_simplify_cfg() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let i1_ty = IntegerType::get(&mut ctx, 1, Signedness::Signless);

    let (func, entry) = empty_func(&mut ctx);
    let region = func.get_operation().deref(&ctx).get_region(0);

    let (ctrue, ctrue_v) = int_const(&mut ctx, i1_ty, 1, 1);
    ctrue.insert_at_back(entry, &ctx);

    let tblk = BasicBlock::new(&mut ctx, None, vec![]);
    let fblk = BasicBlock::new(&mut ctx, None, vec![]);
    let merge = BasicBlock::new(&mut ctx, None, vec![]);
    tblk.insert_at_back(region, &ctx);
    fblk.insert_at_back(region, &ctx);
    merge.insert_at_back(region, &ctx);

    let (flat, sizes) = MirCondBranchOp::compute_segment_sizes(vec![vec![ctrue_v], vec![], vec![]]);
    let cbr = Operation::new(
        &mut ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        flat,
        vec![tblk, fblk],
        0,
    );
    MirCondBranchOp::new(cbr).set_operand_segment_sizes(&ctx, sizes);
    cbr.insert_at_back(entry, &ctx);

    for b in [tblk, fblk] {
        let g = Operation::new(
            &mut ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![merge],
            0,
        );
        g.insert_at_back(b, &ctx);
    }
    let ret = Operation::new(
        &mut ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret.insert_at_back(merge, &ctx);

    eprintln!(">>> START simplify_cfg");
    pliron::opts::simplify_cfg::simplify_cfg(func.get_operation(), &mut ctx).unwrap();
    eprintln!(">>> DONE simplify_cfg");
}

/// Full sequence (sccp -> simplify_cfg -> dce) on the match pattern:
/// `s = add 2,3; eq = (s == 5); cond_br eq, t, f`. sccp folds s and eq and
/// materialises the constant condition; simplify_cfg collapses the branch.
#[test]
fn repro_full_match() {
    use dialect_mir::ops::MirEqOp;
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signed);
    let i1_ty = IntegerType::get(&mut ctx, 1, Signedness::Signless);

    let (func, entry) = empty_func(&mut ctx);
    let region = func.get_operation().deref(&ctx).get_region(0);

    let (c2, c2v) = int_const(&mut ctx, i32_ty, 2, 32);
    let (c3, c3v) = int_const(&mut ctx, i32_ty, 3, 32);
    let (c5, c5v) = int_const(&mut ctx, i32_ty, 5, 32);
    for o in [c2, c3, c5] {
        o.insert_at_back(entry, &ctx);
    }
    let add = Operation::new(
        &mut ctx,
        MirAddOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![c2v, c3v],
        vec![],
        0,
    );
    add.insert_at_back(entry, &ctx);
    let sv = add.deref(&ctx).get_result(0);
    let eq = Operation::new(
        &mut ctx,
        MirEqOp::get_concrete_op_info(),
        vec![i1_ty.into()],
        vec![sv, c5v],
        vec![],
        0,
    );
    eq.insert_at_back(entry, &ctx);
    let eqv = eq.deref(&ctx).get_result(0);

    let tblk = BasicBlock::new(&mut ctx, None, vec![]);
    let fblk = BasicBlock::new(&mut ctx, None, vec![]);
    let merge = BasicBlock::new(&mut ctx, None, vec![]);
    for b in [tblk, fblk, merge] {
        b.insert_at_back(region, &ctx);
    }
    let (flat, sizes) = MirCondBranchOp::compute_segment_sizes(vec![vec![eqv], vec![], vec![]]);
    let cbr = Operation::new(
        &mut ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        flat,
        vec![tblk, fblk],
        0,
    );
    MirCondBranchOp::new(cbr).set_operand_segment_sizes(&ctx, sizes);
    cbr.insert_at_back(entry, &ctx);
    for b in [tblk, fblk] {
        let g = Operation::new(
            &mut ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![merge],
            0,
        );
        g.insert_at_back(b, &ctx);
    }
    let ret = Operation::new(
        &mut ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret.insert_at_back(merge, &ctx);

    let m = func.get_operation();
    eprintln!(">>> START full: sccp");
    pliron::opts::constants::sccp::sccp(m, &mut ctx).unwrap();
    eprintln!(">>> full: simplify_cfg");
    pliron::opts::simplify_cfg::simplify_cfg(m, &mut ctx).unwrap();
    eprintln!(">>> full: dce");
    pliron::opts::dce::dce(m, &mut ctx).unwrap();
    eprintln!(">>> DONE full");
}

/// Unused `add 2, 3` (dead, side-effect-free); run dce only.
#[test]
fn repro_dce() {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signed);

    let (func, entry) = empty_func(&mut ctx);
    let (c2, c2v) = int_const(&mut ctx, i32_ty, 2, 32);
    let (c3, c3v) = int_const(&mut ctx, i32_ty, 3, 32);
    c2.insert_at_back(entry, &ctx);
    c3.insert_at_back(entry, &ctx);
    let add = Operation::new(
        &mut ctx,
        MirAddOp::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![c2v, c3v],
        vec![],
        0,
    );
    add.insert_at_back(entry, &ctx);
    let ret = Operation::new(
        &mut ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret.insert_at_back(entry, &ctx);

    eprintln!(">>> START dce");
    pliron::opts::dce::dce(func.get_operation(), &mut ctx).unwrap();
    eprintln!(">>> DONE dce");
}
