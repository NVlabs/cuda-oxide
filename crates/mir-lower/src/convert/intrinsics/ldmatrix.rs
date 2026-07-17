/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lower `ldmatrix` operations through the selected intrinsic backend.
//!
//! Both backends first turn a generic pointer into an LLVM shared-memory
//! pointer. The LLVM-NVPTX route then calls the exact pointer-specialized
//! intrinsic. The libNVVM route turns that shared pointer into the 32-bit
//! address consumed by exact `.shared` inline PTX.

use crate::convert::intrinsics::common::{call_intrinsic, inline_asm_convergent};
use crate::{IntrinsicBackend, context};
use llvm_export::op_interfaces::CastOpInterface;
use llvm_export::{ops as llvm, types as llvm_types};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::DialectConversionRewriter;
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;

pub(crate) trait LdmatrixInstructionHead {
    fn resolve(self, register_count: usize) -> String;
}

impl LdmatrixInstructionHead for &str {
    fn resolve(self, _register_count: usize) -> String {
        self.to_owned()
    }
}

// Keep previously generated classic callers source-compatible.
impl LdmatrixInstructionHead for bool {
    fn resolve(self, register_count: usize) -> String {
        let transposed = if self { ".trans" } else { "" };
        format!("ldmatrix.sync.aligned.m8n8.x{register_count}{transposed}.shared.b16")
    }
}

/// Lower one generated `ldmatrix` variant.
pub(crate) fn convert_generated_ldmatrix<I: LdmatrixInstructionHead>(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    register_count: usize,
    instruction_head: I,
    typed_intrinsic_name: &str,
) -> Result<()> {
    let name = "ldmatrix";
    if !matches!(register_count, 1 | 2 | 4) {
        return pliron::input_err_noloc!(
            "{} requires an x1, x2, or x4 register shape, got x{}",
            name,
            register_count
        );
    }
    let instruction_head = instruction_head.resolve(register_count);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!("{} requires exactly one pointer", name);
    }
    if op.deref(ctx).get_num_results() != register_count {
        return pliron::input_err_noloc!(
            "{} requires {} i32 result register(s), got {}",
            name,
            register_count,
            op.deref(ctx).get_num_results()
        );
    }

    let shared_pointer = normalize_shared_pointer(ctx, rewriter, operands[0], name)?;
    let result_ty = register_result_type(ctx, register_count);
    let producer = match context::lowering_options(ctx).intrinsic_backend {
        IntrinsicBackend::LlvmNvptx => lower_with_llvm_intrinsic(
            ctx,
            rewriter,
            op,
            shared_pointer,
            result_ty,
            typed_intrinsic_name,
        )?,
        IntrinsicBackend::LibNvvm => lower_with_inline_ptx(
            ctx,
            rewriter,
            shared_pointer,
            result_ty,
            register_count,
            &instruction_head,
        ),
    };

    replace_with_register_results(ctx, rewriter, op, producer, register_count)
}

fn normalize_shared_pointer(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    pointer: Value,
    name: &str,
) -> Result<Value> {
    let pointer_ty = pointer.get_type(ctx);
    let address_space = {
        let pointer_ty = pointer_ty.deref(ctx);
        let Some(pointer_ty) = pointer_ty.downcast_ref::<llvm_types::PointerType>() else {
            return pliron::input_err_noloc!("{} requires an LLVM pointer operand", name);
        };
        pointer_ty.address_space()
    };

    match address_space {
        llvm_types::address_space::SHARED => Ok(pointer),
        llvm_types::address_space::GENERIC => {
            let shared_ty = llvm_types::PointerType::get(ctx, llvm_types::address_space::SHARED);
            let cast = llvm::AddrSpaceCastOp::new(ctx, pointer, shared_ty.into());
            rewriter.insert_operation(ctx, cast.get_operation());
            Ok(cast.get_operation().deref(ctx).get_result(0))
        }
        address_space => pliron::input_err_noloc!(
            "{} requires a generic (address space 0) or shared (address space 3) pointer, got address space {}",
            name,
            address_space
        ),
    }
}

fn register_result_type(ctx: &mut Context, register_count: usize) -> TypeHandle {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    if register_count == 1 {
        i32_ty.into()
    } else {
        llvm_types::StructType::get_unnamed(
            ctx,
            (0..register_count).map(|_| i32_ty.into()).collect(),
        )
        .into()
    }
}

fn lower_with_llvm_intrinsic(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    shared_pointer: Value,
    result_ty: TypeHandle,
    typed_intrinsic_name: &str,
) -> Result<Ptr<Operation>> {
    let shared_ty = llvm_types::PointerType::get(ctx, llvm_types::address_space::SHARED);
    let function_ty = llvm_types::FuncType::get(ctx, result_ty, vec![shared_ty.into()], false);
    call_intrinsic(
        ctx,
        rewriter,
        op,
        typed_intrinsic_name,
        function_ty,
        vec![shared_pointer],
    )
}

fn lower_with_inline_ptx(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    shared_pointer: Value,
    result_ty: TypeHandle,
    register_count: usize,
    instruction_head: &str,
) -> Ptr<Operation> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let pointer_address = llvm::PtrToIntOp::new(ctx, shared_pointer, i32_ty.into());
    rewriter.insert_operation(ctx, pointer_address.get_operation());
    let pointer_address = pointer_address.get_operation().deref(ctx).get_result(0);

    let outputs = (0..register_count)
        .map(|index| format!("${index}"))
        .collect::<Vec<_>>()
        .join(", ");
    let pointer_operand = register_count;
    let template = format!("{instruction_head} {{{outputs}}}, [${pointer_operand}];");
    let constraints = (0..register_count)
        .map(|_| "=r")
        .chain(["r", "~{memory}"])
        .collect::<Vec<_>>()
        .join(",");

    inline_asm_convergent(
        ctx,
        rewriter,
        result_ty,
        vec![pointer_address],
        &template,
        &constraints,
    )
}

fn replace_with_register_results(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    producer: Ptr<Operation>,
    register_count: usize,
) -> Result<()> {
    if register_count == 1 {
        rewriter.replace_operation(ctx, op, producer);
        return Ok(());
    }

    let aggregate = producer.deref(ctx).get_result(0);
    let mut registers = Vec::with_capacity(register_count);
    for index in 0..register_count {
        let extract = llvm::ExtractValueOp::new(ctx, aggregate, vec![index as u32])
            .map_err(|error| pliron::input_error_noloc!("{}", error))?;
        rewriter.insert_operation(ctx, extract.get_operation());
        registers.push(extract.get_operation().deref(ctx).get_result(0));
    }
    rewriter.replace_operation_with_values(ctx, op, registers);
    Ok(())
}
