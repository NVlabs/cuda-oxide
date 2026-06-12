/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Common helpers for GPU intrinsic conversion.
//!
//! This module provides shared utility functions used across all GPU intrinsic
//! converters. These helpers handle common patterns like:
//!
//! - Creating LLVM constants (i1, i32, i64)
//! - Address space pointer casting (generic → shared)
//! - Declaring and calling LLVM intrinsics
//! - Creating inline PTX assembly with convergent attribute
//! - Type conversions for intrinsic results

use crate::helpers;
use llvm_export::op_interfaces::CastOpInterface;
use llvm_export::ops as llvm;
use llvm_export::ops::InlineAsmOpExt;
use llvm_export::types as llvm_types;
use pliron::builtin::op_interfaces::CallOpCallable;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::DialectConversionRewriter;
use pliron::irbuild::inserter::Inserter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::Typed;
use pliron::utils::apint::APInt;
use pliron::value::Value;
use std::num::NonZeroUsize;

/// Create an i1 (boolean) constant with the given value.
pub fn create_i1_const(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: bool,
) -> Value {
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let const_value = if value { 1i64 } else { 0i64 };
    let apint = APInt::from_i64(const_value, NonZeroUsize::new(1).unwrap());
    let attr = pliron::builtin::attributes::IntegerAttr::new(i1_ty, apint);
    let const_op = llvm::ConstantOp::new(ctx, attr.into());
    rewriter.insert_operation(ctx, const_op.get_operation());
    const_op.get_operation().deref(ctx).get_result(0)
}

/// Create an i32 constant with the given value.
pub fn create_i32_const(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: i32,
) -> Value {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let apint = APInt::from_i64(value as i64, NonZeroUsize::new(32).unwrap());
    let attr = pliron::builtin::attributes::IntegerAttr::new(i32_ty, apint);
    let const_op = llvm::ConstantOp::new(ctx, attr.into());
    rewriter.insert_operation(ctx, const_op.get_operation());
    const_op.get_operation().deref(ctx).get_result(0)
}

/// Create an i64 constant with the given value.
pub fn create_i64_const(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: i64,
) -> Value {
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let apint = APInt::from_i64(value, NonZeroUsize::new(64).unwrap());
    let attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, apint);
    let const_op = llvm::ConstantOp::new(ctx, attr.into());
    rewriter.insert_operation(ctx, const_op.get_operation());
    const_op.get_operation().deref(ctx).get_result(0)
}

/// Cast a pointer value to address space 3 (shared memory) if needed.
pub fn cast_to_shared_addrspace(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    ptr: Value,
) -> Value {
    let ptr_ty = ptr.get_type(ctx);
    let current_addrspace = ptr_ty
        .deref(ctx)
        .downcast_ref::<llvm_types::PointerType>()
        .map(|pt| pt.address_space())
        .unwrap_or(0);

    if current_addrspace != 3 {
        let cast_ty = llvm_types::PointerType::get(ctx, 3).into();
        let cast_op = llvm::AddrSpaceCastOp::new(ctx, ptr, cast_ty);
        rewriter.insert_operation(ctx, cast_op.get_operation());
        cast_op.get_operation().deref(ctx).get_result(0)
    } else {
        ptr
    }
}

/// Cast a pointer to the cluster shared address space (`addrspace(7)`).
pub fn cast_to_cluster_shared_addrspace(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    ptr: Value,
) -> Value {
    let ptr_ty = ptr.get_type(ctx);
    let current_addrspace = ptr_ty
        .deref(ctx)
        .downcast_ref::<llvm_types::PointerType>()
        .map(|pt| pt.address_space())
        .unwrap_or(0);

    if current_addrspace != 7 {
        let cast_ty = llvm_types::PointerType::get(ctx, 7).into();
        let cast_op = llvm::AddrSpaceCastOp::new(ctx, ptr, cast_ty);
        rewriter.insert_operation(ctx, cast_op.get_operation());
        cast_op.get_operation().deref(ctx).get_result(0)
    } else {
        ptr
    }
}

/// Create an LLVM function call to an intrinsic.
///
/// Ensures the intrinsic is declared in the module, then creates a call.
/// `current_op` is the MIR op being converted (used to find the parent module).
pub fn call_intrinsic(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    current_op: Ptr<Operation>,
    intrinsic_name: &str,
    func_ty: pliron::r#type::TypePtr<llvm_types::FuncType>,
    args: Vec<Value>,
) -> Result<Ptr<Operation>> {
    let parent_block = current_op.deref(ctx).get_parent_block().unwrap();
    helpers::ensure_intrinsic_declared(ctx, parent_block, intrinsic_name, func_ty)
        .map_err(|e| pliron::input_error_noloc!("{}", e))?;

    let sym_name: pliron::identifier::Identifier = intrinsic_name.try_into().unwrap();
    let callee = CallOpCallable::Direct(sym_name);
    let llvm_call = llvm::CallOp::new(ctx, callee, func_ty, args);
    rewriter.insert_operation(ctx, llvm_call.get_operation());

    Ok(llvm_call.get_operation())
}

/// Create an inline assembly operation with the convergent attribute.
pub fn inline_asm_convergent(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    result_ty: Ptr<pliron::r#type::TypeObj>,
    inputs: Vec<Value>,
    asm_template: &str,
    constraints: &str,
) -> Ptr<Operation> {
    let inline_asm =
        llvm::InlineAsmOp::new_convergent(ctx, result_ty, inputs, asm_template, constraints);
    rewriter.insert_operation(ctx, inline_asm.get_operation());
    inline_asm.get_operation()
}

/// Truncate an i32 result to i1 (for predicate results).
pub fn trunc_to_i1(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    i32_val: Value,
) -> Value {
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let trunc_op = llvm::TruncOp::new(ctx, i32_val, i1_ty.into());
    rewriter.insert_operation(ctx, trunc_op.get_operation());
    trunc_op.get_operation().deref(ctx).get_result(0)
}

#[cfg(test)]
mod tests {
    // TODO: Add tests for common intrinsic helpers
}
