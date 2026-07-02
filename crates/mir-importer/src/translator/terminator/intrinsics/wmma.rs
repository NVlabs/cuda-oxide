/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix intrinsics (`movmatrix`, `mma.sync`).

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_mir::{
    attributes::FieldIndexAttr,
    ops::{MirConstructArrayOp, MirExtractFieldOp},
    types::MirArrayType,
};
use dialect_nvvm::ops::{MmaM16N8K16F32Bf16Op, MovmatrixTransB16Op};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use rustc_public::mir;

/// Emit movmatrix_trans_b16: in-register 8×8 matrix transpose.
///
/// Takes one u32 operand and returns one u32.
#[allow(clippy::too_many_arguments)]
pub fn emit_movmatrix_trans_b16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "movmatrix_trans_b16 expects 1 argument, got {}",
                args.len()
            ))
        );
    }

    let (a_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let mov_op = Operation::new(
        ctx,
        MovmatrixTransB16Op::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![a_val],
        vec![],
        0,
    );
    mov_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        mov_op.insert_after(ctx, prev);
    } else {
        mov_op.insert_at_front(block_ptr, ctx);
    }

    let result = mov_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        mov_op,
        value_map,
        block_map,
        loc,
        "movmatrix_trans_b16 call without target block",
    )
}

/// Extract a fixed-size Rust array into scalar SSA register values.
///
/// Constant-field extraction lowers to LLVM `extractvalue`, so no temporary
/// stack slot is introduced for the MMA fragments.
fn extract_array_registers(
    ctx: &mut Context,
    array: Value,
    expected_element_ty: TypeHandle,
    expected_len: usize,
    block_ptr: Ptr<BasicBlock>,
    mut last_op: Option<Ptr<Operation>>,
    loc: Location,
    fragment_name: &str,
) -> TranslationResult<(Vec<Value>, Ptr<Operation>)> {
    let array_ty = array.get_type(ctx);
    let valid_array = {
        let array_ty = array_ty.deref(ctx);
        array_ty
            .downcast_ref::<MirArrayType>()
            .is_some_and(|array_ty| {
                array_ty.size() == expected_len as u64
                    && array_ty.element_type() == expected_element_ty
            })
    };
    if !valid_array {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "mma_m16n8k16_f32_bf16 {fragment_name} fragment must be an array of {expected_len} scalar registers"
            ))
        );
    }

    let mut registers = Vec::with_capacity(expected_len);
    for index in 0..expected_len {
        let extract = Operation::new(
            ctx,
            MirExtractFieldOp::get_concrete_op_info(),
            vec![expected_element_ty],
            vec![array],
            vec![],
            0,
        );
        extract.deref_mut(ctx).set_loc(loc.clone());
        let extract = MirExtractFieldOp::new(extract);
        extract.set_attr_index(ctx, FieldIndexAttr(index as u32));
        if let Some(previous) = last_op {
            extract.get_operation().insert_after(ctx, previous);
        } else {
            extract.get_operation().insert_at_front(block_ptr, ctx);
        }
        last_op = Some(extract.get_operation());
        registers.push(extract.get_operation().deref(ctx).get_result(0));
    }

    Ok((registers, last_op.expect("non-empty MMA fragments")))
}

/// Emit `mma_m16n8k16_f32_bf16` as a register-producing dialect operation.
///
/// Args:
/// - `args[0]`: `[f32; 4]` C accumulator registers
/// - `args[1]`: `[u32; 4]` packed A fragment registers
/// - `args[2]`: `[u32; 2]` packed B fragment registers
///
/// Returns: `[f32; 4]` D accumulator registers.
pub fn emit_mma_m16n8k16_f32_bf16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "mma_m16n8k16_f32_bf16 expects 3 arguments (acc, a, b), got {}",
                args.len()
            ))
        );
    }

    let f32_ty = FP32Type::get(ctx);
    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let (c_array, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let (c_registers, last_op) = extract_array_registers(
        ctx,
        c_array,
        f32_ty.into(),
        4,
        block_ptr,
        last_op,
        loc.clone(),
        "C",
    )?;

    let (a_array, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        Some(last_op),
        loc.clone(),
    )?;
    let (a_registers, last_op) = extract_array_registers(
        ctx,
        a_array,
        u32_ty.into(),
        4,
        block_ptr,
        last_op_after,
        loc.clone(),
        "A",
    )?;

    let (b_array, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        Some(last_op),
        loc.clone(),
    )?;
    let (b_registers, last_op) = extract_array_registers(
        ctx,
        b_array,
        u32_ty.into(),
        2,
        block_ptr,
        last_op_after,
        loc.clone(),
        "B",
    )?;

    let mut operands = c_registers;
    operands.extend(a_registers);
    operands.extend(b_registers);

    let mma_op = Operation::new(
        ctx,
        MmaM16N8K16F32Bf16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        operands,
        vec![],
        0,
    );
    mma_op.deref_mut(ctx).set_loc(loc.clone());
    mma_op.insert_after(ctx, last_op);

    let d_registers = (0..4)
        .map(|index| mma_op.deref(ctx).get_result(index))
        .collect();
    let array_ty = MirArrayType::get(ctx, f32_ty.into(), 4);
    let d_array = Operation::new(
        ctx,
        MirConstructArrayOp::get_concrete_op_info(),
        vec![array_ty.into()],
        d_registers,
        vec![],
        0,
    );
    d_array.deref_mut(ctx).set_loc(loc.clone());
    d_array.insert_after(ctx, mma_op);
    let result = d_array.deref(ctx).get_result(0);

    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        d_array,
        value_map,
        block_map,
        loc,
        "mma_m16n8k16_f32_bf16 call without target block",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::linked_list::ContainsLinkedList;

    #[test]
    fn mma_fragments_are_extracted_as_constant_index_ssa_values() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);

        let f32_ty = FP32Type::get(&ctx);
        let array_ty = MirArrayType::get(&mut ctx, f32_ty.into(), 4);
        let block = BasicBlock::new(&mut ctx, None, vec![array_ty.into()]);
        let array = block.deref(&ctx).get_argument(0);

        let (registers, last_op) = extract_array_registers(
            &mut ctx,
            array,
            f32_ty.into(),
            4,
            block,
            None,
            Location::Unknown,
            "C",
        )
        .expect("valid C fragment must extract");

        assert_eq!(registers.len(), 4);
        assert!(
            registers
                .iter()
                .all(|register| register.get_type(&ctx) == f32_ty.into())
        );

        let operations: Vec<_> = block.deref(&ctx).iter(&ctx).collect();
        assert_eq!(operations.len(), 4);
        assert_eq!(operations.last().copied(), Some(last_op));
        for (index, operation) in operations.into_iter().enumerate() {
            let extract = Operation::get_op::<MirExtractFieldOp>(operation, &ctx)
                .expect("fragment extraction must use constant-index extract_field");
            assert_eq!(
                extract.get_attr_index(&ctx).map(|attr| attr.0),
                Some(index as u32)
            );
        }

        assert!(
            extract_array_registers(
                &mut ctx,
                array,
                f32_ty.into(),
                2,
                block,
                Some(last_op),
                Location::Unknown,
                "B",
            )
            .is_err()
        );
    }
}
