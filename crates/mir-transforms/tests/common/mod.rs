/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Helpers for building small `dialect-mir` CFGs in integration tests, so the
//! analyses (`LoopInfo`, `induction`) and the unroll pass can be exercised
//! without going through the whole rustc -> MIR pipeline.
//!
//! Each test binary (`loop_info`, `induction`, `unroll`) pulls this in via
//! `mod common;` and uses only the builders it needs, hence the crate-wide
//! `dead_code` allow.

#![allow(dead_code)]

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
pub fn mir_ctx() -> Context {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    ctx
}

/// A signless `i1` type.
pub fn i1(ctx: &mut Context) -> TypedHandle<IntegerType> {
    IntegerType::get(ctx, 1, Signedness::Signless)
}

/// An unsigned 32-bit type (the IV type our kernels use).
pub fn u32t(ctx: &mut Context) -> TypedHandle<IntegerType> {
    IntegerType::get(ctx, 32, Signedness::Unsigned)
}

/// Create an empty `fn foo()` inside a module and return `(module_op, region)`.
/// Blocks are appended to `region` by the caller; the first block is the entry.
pub fn empty_func(ctx: &mut Context) -> (Ptr<Operation>, Ptr<Region>) {
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
pub fn block(ctx: &mut Context, region: Ptr<Region>, args: Vec<TypeHandle>) -> Ptr<BasicBlock> {
    let b = BasicBlock::new(ctx, None, args);
    b.insert_at_back(region, ctx);
    b
}

/// Append an integer constant to `b` and return its result value.
pub fn iconst(
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
pub fn goto(ctx: &mut Context, b: Ptr<BasicBlock>, target: Ptr<BasicBlock>, operands: Vec<Value>) {
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
pub fn cond_br(
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
pub fn ret(ctx: &mut Context, b: Ptr<BasicBlock>) {
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
pub struct CountedLoop {
    pub module: Ptr<Operation>,
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
pub fn counted_loop(ctx: &mut Context, n: i64) -> CountedLoop {
    let (module, region) = empty_func(ctx);
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
        module,
        region,
        preheader,
        header,
        latch,
        exit,
    }
}

/// A built nested counted loop (outer `while i < n` containing inner
/// `while j < m`) and the blocks worth asserting on.
pub struct NestedLoop {
    pub module: Ptr<Operation>,
    pub region: Ptr<Region>,
    pub preheader: Ptr<BasicBlock>,
    pub outer_header: Ptr<BasicBlock>,
    pub outer_body: Ptr<BasicBlock>,
    pub inner_header: Ptr<BasicBlock>,
    pub inner_body: Ptr<BasicBlock>,
    pub outer_latch: Ptr<BasicBlock>,
    pub exit: Ptr<BasicBlock>,
}

/// Build `while i < n { while j < m { j += 1 } i += 1 }` in the shape mem2reg
/// leaves it (carried values are header block arguments, exit tests are
/// `not(_ < _)`):
///
/// ```text
///   preheader:        i0=0;             goto outer_header(i0)
///   outer_header(i):  nlt = not(i < n); cond_br nlt [exit, outer_body]
///   outer_body:       j0=0;             goto inner_header(j0)   // inner preheader
///   inner_header(j):  mlt = not(j < m); cond_br mlt [outer_latch, inner_body]
///   inner_body:       j1 = j+1;         goto inner_header(j1)
///   outer_latch:      i1 = i+1;         goto outer_header(i1)
///   exit:             return
/// ```
///
/// The outer loop *contains* the inner loop, so this is the shape the
/// nested-unroll path must handle: unrolling the outer clones the inner loop
/// wholesale (it stays a loop in each copy), it is never flattened.
pub fn nested_counted_loop(ctx: &mut Context, n: i64, m: i64) -> NestedLoop {
    let (module, region) = empty_func(ctx);
    let u32 = u32t(ctx);
    let i1 = i1(ctx);

    let preheader = block(ctx, region, vec![]);
    let outer_header = block(ctx, region, vec![u32.into()]); // (i)
    let outer_body = block(ctx, region, vec![]);
    let inner_header = block(ctx, region, vec![u32.into()]); // (j)
    let inner_body = block(ctx, region, vec![]);
    let outer_latch = block(ctx, region, vec![]);
    let exit = block(ctx, region, vec![]);

    let not = |ctx: &mut Context, b: Ptr<BasicBlock>, v: Value| -> Value {
        let op = Operation::new(
            ctx,
            MirNotOp::get_concrete_op_info(),
            vec![i1.into()],
            vec![v],
            vec![],
            0,
        );
        op.insert_at_back(b, ctx);
        op.deref(ctx).get_result(0)
    };

    // preheader: i0 = 0; goto outer_header(i0)
    let i0 = iconst(ctx, preheader, u32, 0);
    goto(ctx, preheader, outer_header, vec![i0]);

    // outer_header(i): nlt = not(i < n); cond_br nlt [exit, outer_body]
    let i = outer_header.deref(ctx).get_argument(0);
    let nconst = iconst(ctx, outer_header, u32, n);
    let lt = op2!(
        ctx,
        outer_header,
        MirLtOp::get_concrete_op_info(),
        i1.into(),
        i,
        nconst
    );
    let nlt = not(ctx, outer_header, lt);
    cond_br(ctx, outer_header, nlt, exit, outer_body);

    // outer_body: j0 = 0; goto inner_header(j0)
    let j0 = iconst(ctx, outer_body, u32, 0);
    goto(ctx, outer_body, inner_header, vec![j0]);

    // inner_header(j): mlt = not(j < m); cond_br mlt [outer_latch, inner_body]
    let j = inner_header.deref(ctx).get_argument(0);
    let mconst = iconst(ctx, inner_header, u32, m);
    let jlt = op2!(
        ctx,
        inner_header,
        MirLtOp::get_concrete_op_info(),
        i1.into(),
        j,
        mconst
    );
    let jnlt = not(ctx, inner_header, jlt);
    cond_br(ctx, inner_header, jnlt, outer_latch, inner_body);

    // inner_body: j1 = j + 1; goto inner_header(j1)
    let one_j = iconst(ctx, inner_body, u32, 1);
    let j1 = op2!(
        ctx,
        inner_body,
        MirAddOp::get_concrete_op_info(),
        u32.into(),
        j,
        one_j
    );
    goto(ctx, inner_body, inner_header, vec![j1]);

    // outer_latch: i1 = i + 1; goto outer_header(i1)
    let one_i = iconst(ctx, outer_latch, u32, 1);
    let inext = op2!(
        ctx,
        outer_latch,
        MirAddOp::get_concrete_op_info(),
        u32.into(),
        i,
        one_i
    );
    goto(ctx, outer_latch, outer_header, vec![inext]);

    // exit: return
    ret(ctx, exit);

    NestedLoop {
        module,
        region,
        preheader,
        outer_header,
        outer_body,
        inner_header,
        inner_body,
        outer_latch,
        exit,
    }
}
