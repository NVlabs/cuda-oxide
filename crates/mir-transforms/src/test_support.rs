/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Helpers for building small `dialect-mir` CFGs in unit tests, so the analyses
//! (`LoopInfo`, `induction`) and the unroller's shape checks can be exercised
//! without going through the whole rustc -> MIR pipeline.

use core::num::NonZero;

use dialect_mir::ops::{
    MirAddOp, MirCondBranchOp, MirConstantOp, MirFuncOp, MirGotoOp, MirLtOp, MirNotOp, MirReturnOp,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::{IntegerAttr, TypeAttr};
use pliron::builtin::op_interfaces::{
    OperandSegmentInterface, SingleBlockRegionInterface, SymbolOpInterface,
};
use pliron::builtin::ops::ModuleOp;
use pliron::builtin::types::{FunctionType, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::region::Region;
use pliron::r#type::{TypeHandle, TypedHandle};
use pliron::utils::apint::APInt;
use pliron::value::Value;

/// A fresh context with the `mir` dialect registered (builtin is registered by
/// `Context::new`).
pub(crate) fn mir_ctx() -> Context {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    ctx
}

/// A signless `i1` type.
pub(crate) fn i1(ctx: &mut Context) -> TypedHandle<IntegerType> {
    IntegerType::get(ctx, 1, Signedness::Signless)
}

/// An unsigned 32-bit type (the IV type our kernels use).
pub(crate) fn u32t(ctx: &mut Context) -> TypedHandle<IntegerType> {
    IntegerType::get(ctx, 32, Signedness::Unsigned)
}

/// Create an empty `fn foo()` inside a module and return `(module_op, region)`.
/// Blocks are appended to `region` by the caller; the first block is the entry.
pub(crate) fn empty_func(ctx: &mut Context) -> (Ptr<Operation>, Ptr<Region>) {
    let module = ModuleOp::new(ctx, "test".try_into().unwrap());
    let func_ty = FunctionType::get(ctx, vec![], vec![]);
    let func_op = Operation::new(
        ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func = MirFuncOp::new(ctx, func_op, TypeAttr::new(func_ty.into()));
    func.set_symbol_name(ctx, "foo".try_into().unwrap());
    module.append_operation(ctx, func_op, 0);
    let region = func_op.deref(ctx).get_region(0);
    (module.get_operation(), region)
}

/// Append an empty block (with the given argument types) to `region`.
pub(crate) fn block(
    ctx: &mut Context,
    region: Ptr<Region>,
    args: Vec<TypeHandle>,
) -> Ptr<BasicBlock> {
    let b = BasicBlock::new(ctx, None, args);
    b.insert_at_back(region, ctx);
    b
}

/// Append an integer constant to `b` and return its result value.
pub(crate) fn iconst(
    ctx: &mut Context,
    b: Ptr<BasicBlock>,
    ty: TypedHandle<IntegerType>,
    val: i64,
) -> Value {
    let width = ty.deref(ctx).width() as usize;
    let apint = APInt::from_i64(val, NonZero::new(width).unwrap());
    let op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![ty.into()],
        vec![],
        vec![],
        0,
    );
    MirConstantOp::new(op).set_attr_value(ctx, IntegerAttr::new(ty, apint));
    op.insert_at_back(b, ctx);
    op.deref(ctx).get_result(0)
}

/// Append an unconditional `goto target(operands)` to `b`.
pub(crate) fn goto(
    ctx: &mut Context,
    b: Ptr<BasicBlock>,
    target: Ptr<BasicBlock>,
    operands: Vec<Value>,
) {
    let op = Operation::new(
        ctx,
        MirGotoOp::get_concrete_op_info(),
        vec![],
        operands,
        vec![target],
        0,
    );
    op.insert_at_back(b, ctx);
}

/// Append `cond_br cond [true_succ, false_succ]` (no successor operands) to `b`.
pub(crate) fn cond_br(
    ctx: &mut Context,
    b: Ptr<BasicBlock>,
    cond: Value,
    true_succ: Ptr<BasicBlock>,
    false_succ: Ptr<BasicBlock>,
) {
    let (flat, segs) = MirCondBranchOp::compute_segment_sizes(vec![vec![cond], vec![], vec![]]);
    let op = Operation::new(
        ctx,
        MirCondBranchOp::get_concrete_op_info(),
        vec![],
        flat,
        vec![true_succ, false_succ],
        0,
    );
    Operation::get_op::<MirCondBranchOp>(op, ctx)
        .unwrap()
        .set_operand_segment_sizes(ctx, segs);
    op.insert_at_back(b, ctx);
}

/// Append a `return` (no value) to `b`.
pub(crate) fn ret(ctx: &mut Context, b: Ptr<BasicBlock>) {
    let op = Operation::new(
        ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    op.insert_at_back(b, ctx);
}

/// Append a two-operand op (built from `info`) of `result_ty` to `b`, returning
/// its result. A macro rather than a fn because `ConcreteOpInfo` is crate-private
/// in pliron, so it cannot be named in a parameter type here.
macro_rules! op2 {
    ($ctx:expr, $b:expr, $info:expr, $ty:expr, $lhs:expr, $rhs:expr) => {{
        let op = Operation::new($ctx, $info, vec![$ty], vec![$lhs, $rhs], vec![], 0);
        op.insert_at_back($b, $ctx);
        op.deref($ctx).get_result(0)
    }};
}

/// A built counted loop and the blocks worth asserting on.
pub(crate) struct CountedLoop {
    pub region: Ptr<Region>,
    pub preheader: Ptr<BasicBlock>,
    pub header: Ptr<BasicBlock>,
    pub latch: Ptr<BasicBlock>,
    pub exit: Ptr<BasicBlock>,
}

/// Build the canonical counted loop `while i < n { acc += i; i += 1 }`, in the
/// shape mem2reg leaves it (carried values are header block arguments, the exit
/// test is `not(i < n)`):
///
/// ```text
///   preheader:        acc0=0; i0=0;            goto header(acc0, i0)
///   header(acc, i):   nlt = not(i < n);        cond_br nlt [exit, latch]
///   latch:            acc1=acc+i; i1=i+1;      goto header(acc1, i1)
///   exit:             return
/// ```
pub(crate) fn counted_loop(ctx: &mut Context, n: i64) -> CountedLoop {
    let (_module, region) = empty_func(ctx);
    let u32 = u32t(ctx);
    let i1 = i1(ctx);

    let preheader = block(ctx, region, vec![]);
    let header = block(ctx, region, vec![u32.into(), u32.into()]); // (acc, i)
    let latch = block(ctx, region, vec![]);
    let exit = block(ctx, region, vec![]);

    // preheader: acc0 = 0; i0 = 0; goto header(acc0, i0)
    let acc0 = iconst(ctx, preheader, u32, 0);
    let i0 = iconst(ctx, preheader, u32, 0);
    goto(ctx, preheader, header, vec![acc0, i0]);

    // header(acc, i): nlt = not(i < n); cond_br nlt [exit, latch]
    let acc = header.deref(ctx).get_argument(0);
    let i = header.deref(ctx).get_argument(1);
    let nconst = iconst(ctx, header, u32, n);
    let lt = op2!(
        ctx,
        header,
        MirLtOp::get_concrete_op_info(),
        i1.into(),
        i,
        nconst
    );
    let nlt = {
        let op = Operation::new(
            ctx,
            MirNotOp::get_concrete_op_info(),
            vec![i1.into()],
            vec![lt],
            vec![],
            0,
        );
        op.insert_at_back(header, ctx);
        op.deref(ctx).get_result(0)
    };
    cond_br(ctx, header, nlt, exit, latch);

    // latch: acc1 = acc + i; i1 = i + 1; goto header(acc1, i1)
    let acc1 = op2!(
        ctx,
        latch,
        MirAddOp::get_concrete_op_info(),
        u32.into(),
        acc,
        i
    );
    let one = iconst(ctx, latch, u32, 1);
    let inext = op2!(
        ctx,
        latch,
        MirAddOp::get_concrete_op_info(),
        u32.into(),
        i,
        one
    );
    goto(ctx, latch, header, vec![acc1, inext]);

    // exit: return
    ret(ctx, exit);

    CountedLoop {
        region,
        preheader,
        header,
        latch,
        exit,
    }
}
