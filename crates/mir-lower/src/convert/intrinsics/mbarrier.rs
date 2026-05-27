/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Mbarrier (async barrier) intrinsic conversion for Hopper+ GPUs.
//!
//! # Operations
//!
//! | Operation          | Implementation | Description                         |
//! |--------------------|----------------|-------------------------------------|
//! | `Init`             | LLVM intrinsic | Initialize barrier with thread count|
//! | `Arrive`           | LLVM intrinsic | Signal arrival                      |
//! | `ArriveExpectTx`   | Inline PTX     | Signal arrival with expected bytes  |
//! | `TestWait`         | Inline PTX     | Non-blocking wait check             |
//! | `TryWait`          | Inline PTX     | Blocking wait with hint             |
//! | `TryWaitParity`    | Inline PTX     | Parity-based wait                   |
//! | `Inval`            | LLVM intrinsic | Invalidate barrier                  |
//! | `FenceProxyAsync`  | Inline PTX     | Memory fence                        |

use crate::convert::intrinsics::common::*;
use dialect_llvm::types as llvm_types;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::rewriter::Rewriter;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::value::DefiningEntity;

/// mbarrier.init.shared: (ptr, count) -> void
pub(crate) fn convert_init(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let ptr_ty = llvm_types::PointerType::get(ctx, 3);
    let void_ty = llvm_types::VoidType::get(ctx);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("mbarrier_init requires 2 operands");
    }
    let (bar_ptr, count) = (operands[0], operands[1]);
    let bar_ptr = cast_to_shared_addrspace(ctx, rewriter, bar_ptr);

    let func_ty = llvm_types::FuncType::get(
        ctx,
        void_ty.into(),
        vec![ptr_ty.into(), i32_ty.into()],
        false,
    );
    call_intrinsic(
        ctx,
        rewriter,
        op,
        "llvm_nvvm_mbarrier_init_shared",
        func_ty,
        vec![bar_ptr, count],
    )?;
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// mbarrier.arrive.shared: (ptr) -> i64
pub(crate) fn convert_arrive(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let ptr_ty = llvm_types::PointerType::get(ctx, 3);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!("mbarrier_arrive requires 1 operand");
    }
    let bar_ptr = cast_to_shared_addrspace(ctx, rewriter, operands[0]);

    let func_ty = llvm_types::FuncType::get(ctx, i64_ty.into(), vec![ptr_ty.into()], false);
    let call_op = call_intrinsic(
        ctx,
        rewriter,
        op,
        "llvm_nvvm_mbarrier_arrive_shared",
        func_ty,
        vec![bar_ptr],
    )?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}

/// mbarrier.arrive.expect_tx: (ptr, bytes) -> i64 (inline PTX)
pub(crate) fn convert_arrive_expect_tx(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("mbarrier_arrive_expect_tx requires 2 operands");
    }
    let bar_ptr = cast_to_shared_addrspace(ctx, rewriter, operands[0]);
    let bytes = operands[1];

    let asm_template = "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;";
    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        i64_ty.into(),
        vec![bar_ptr, bytes],
        asm_template,
        "=l,l,r,~{memory}",
    );
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

/// mbarrier.arrive.release.cluster.shared::cluster.b64: (addr: u64) -> void
pub(crate) fn convert_arrive_cluster(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!("mbarrier_arrive_cluster requires 1 operand (addr: u64)");
    }
    let addr = operands[0];

    let asm_template = "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$0];";
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![addr],
        asm_template,
        "l,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// mbarrier.test_wait: (ptr, token) -> i1 (inline PTX)
pub(crate) fn convert_test_wait(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("mbarrier_test_wait requires 2 operands");
    }
    let bar_ptr = cast_to_shared_addrspace(ctx, rewriter, operands[0]);
    let token = operands[1];

    let asm_template =
        "{ .reg .pred p; mbarrier.test_wait.shared.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }";
    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        i32_ty.into(),
        vec![bar_ptr, token],
        asm_template,
        "=r,l,l,~{memory}",
    );
    let i32_result = asm_op.deref(ctx).get_result(0);
    let trunc_op = trunc_to_i1(ctx, rewriter, i32_result);
    // trunc_to_i1 returns a Value; we need the operation that defined it
    let trunc_def_op = match trunc_op.defining_entity() {
        DefiningEntity::Op(def_op) => def_op,
        _ => unreachable!(),
    };
    rewriter.replace_operation(ctx, op, trunc_def_op);
    Ok(())
}

/// mbarrier.try_wait: (ptr, token) -> i1 (inline PTX)
pub(crate) fn convert_try_wait(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("mbarrier_try_wait requires 2 operands");
    }
    let bar_ptr = cast_to_shared_addrspace(ctx, rewriter, operands[0]);
    let token = operands[1];

    let asm_template =
        "{ .reg .pred p; mbarrier.try_wait.shared.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }";
    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        i32_ty.into(),
        vec![bar_ptr, token],
        asm_template,
        "=r,l,l,~{memory}",
    );
    let i32_result = asm_op.deref(ctx).get_result(0);
    let trunc_val = trunc_to_i1(ctx, rewriter, i32_result);
    let trunc_def_op = match trunc_val.defining_entity() {
        DefiningEntity::Op(def_op) => def_op,
        _ => unreachable!(),
    };
    rewriter.replace_operation(ctx, op, trunc_def_op);
    Ok(())
}

/// mbarrier.try_wait.parity: (ptr, parity) -> i1 (inline PTX)
pub(crate) fn convert_try_wait_parity(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("mbarrier_try_wait_parity requires 2 operands");
    }
    let bar_ptr = cast_to_shared_addrspace(ctx, rewriter, operands[0]);
    let parity = operands[1];

    let asm_template = "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }";
    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        i32_ty.into(),
        vec![bar_ptr, parity],
        asm_template,
        "=r,l,r,~{memory}",
    );
    let i32_result = asm_op.deref(ctx).get_result(0);
    let trunc_val = trunc_to_i1(ctx, rewriter, i32_result);
    let trunc_def_op = match trunc_val.defining_entity() {
        DefiningEntity::Op(def_op) => def_op,
        _ => unreachable!(),
    };
    rewriter.replace_operation(ctx, op, trunc_def_op);
    Ok(())
}

/// mbarrier.inval: (ptr) -> void
pub(crate) fn convert_inval(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    let ptr_ty = llvm_types::PointerType::get(ctx, 3);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!("mbarrier_inval requires 1 operand");
    }
    let bar_ptr = cast_to_shared_addrspace(ctx, rewriter, operands[0]);

    let func_ty = llvm_types::FuncType::get(ctx, void_ty.into(), vec![ptr_ty.into()], false);
    call_intrinsic(
        ctx,
        rewriter,
        op,
        "llvm_nvvm_mbarrier_inval_shared",
        func_ty,
        vec![bar_ptr],
    )?;
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// fence.proxy.async.shared::cta
pub(crate) fn convert_fence_proxy_async(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![],
        "fence.proxy.async.shared::cta;",
        "~{memory}",
    );
    // NOTE: The caller (interface_impls.rs) is responsible for erasing the original op
    // since this function does not receive it.
    Ok(())
}

/// nanosleep.u32: (ns: u32) -> void (inline PTX)
pub(crate) fn convert_nanosleep(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!("nanosleep requires 1 operand (ns: u32)");
    }
    let ns = operands[0];

    let asm_template = "nanosleep.u32 $0;";
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![ns],
        asm_template,
        "r,~{memory}",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}
