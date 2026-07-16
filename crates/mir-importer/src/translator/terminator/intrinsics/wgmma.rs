/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Hopper WGMMA (Warpgroup Matrix Multiply-Accumulate) intrinsics.
//!
//! Handles Hopper `sm_90a` asynchronous warpgroup matrix operations.

use super::super::helpers::{emit_goto, emit_store_result_and_goto};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{WgmmaMakeSmemDescOp, WgmmaMmaM64N64K16F32Bf16Op};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

const CUSTOM_DESCRIPTOR_UNSUPPORTED: &str = "custom WGMMA descriptor encoding is not yet supported";
const MMA_UNSUPPORTED: &str = "WGMMA MMA is not yet supported: lowering must preserve delayed \
32-register accumulator state across commit_group and wait_group";

fn unsupported_diagnostic(path: &str) -> Option<&'static str> {
    match path {
        "cuda_device::wgmma::make_smem_desc_custom" => Some(CUSTOM_DESCRIPTOR_UNSUPPORTED),
        "cuda_device::wgmma::wgmma_mma_m64n64k16_f32_bf16"
        | "cuda_device::wgmma::wgmma_mma_m64n64k16_f32_f16"
        | "cuda_device::wgmma::wgmma_mma_m64n64k16_f32_tf32" => Some(MMA_UNSUPPORTED),
        _ => None,
    }
}

/// Reject public WGMMA entries that do not have a sound lowering yet.
pub(crate) fn reject_unsupported(path: &str, loc: Location) -> TranslationResult<()> {
    let Some(diagnostic) = unsupported_diagnostic(path) else {
        return Ok(());
    };
    input_err!(loc, TranslationErr::unsupported(diagnostic))
}

/// Emit make_smem_desc: Create SMEM descriptor for WGMMA.
///
/// Args:
/// - `args[0]`: *const u8 (pointer to shared memory)
///
/// Returns: u64 (64-bit descriptor)
pub fn emit_wgmma_make_smem_desc(
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
                "make_smem_desc expects 1 argument, got {}",
                args.len()
            ))
        );
    }

    // Translate the pointer argument
    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Create the make_smem_desc operation (returns u64)
    // Use Unsigned signedness to match Rust's u64 type
    let u64_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
    let desc_op = Operation::new(
        ctx,
        WgmmaMakeSmemDescOp::get_concrete_op_info(),
        vec![u64_ty.into()], // Result: u64
        vec![ptr_val],       // Operand: ptr
        vec![],
        0,
    );
    desc_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        desc_op.insert_after(ctx, prev);
    } else {
        desc_op.insert_at_front(block_ptr, ctx);
    }

    // Map the result
    let result_value = desc_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        desc_op,
        value_map,
        block_map,
        loc,
        "make_smem_desc call without target block",
    )
}

/// Emit wgmma_mma_m64n64k16_f32_bf16: WGMMA matrix multiply-accumulate.
///
/// Performs D = A × B + D where:
/// - A: 64×16 (from SMEM descriptor)
/// - B: 16×64 (from SMEM descriptor)
/// - D: 64×64 accumulator (32 f32 values per thread, passed by pointer)
///
/// Args:
/// - `args[0]`: &mut [[f32; 8]; 4] (accumulator pointer, read-modify-write)
/// - `args[1]`: u64 (desc_a - SMEM descriptor for A)
/// - `args[2]`: u64 (desc_b - SMEM descriptor for B)
///
/// Returns: void (accumulator updated in-place)
pub fn emit_wgmma_mma_m64n64k16_f32_bf16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
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
                "wgmma_mma_m64n64k16_f32_bf16 expects 3 arguments (acc_ptr, desc_a, desc_b), got {}",
                args.len()
            ))
        );
    }

    // Translate arguments
    let mut last_op = prev_op;

    // arg[0]: acc_ptr (pointer to accumulator array)
    let (acc_ptr, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // arg[1]: desc_a (u64 descriptor)
    let (desc_a, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // arg[2]: desc_b (u64 descriptor)
    let (desc_b, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // Create the WGMMA MMA operation
    let mma_op = Operation::new(
        ctx,
        WgmmaMmaM64N64K16F32Bf16Op::get_concrete_op_info(),
        vec![],                        // No results (void)
        vec![acc_ptr, desc_a, desc_b], // Operands
        vec![],
        0,
    );
    mma_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        mma_op.insert_after(ctx, prev);
    } else {
        mma_op.insert_at_front(block_ptr, ctx);
    }

    // Emit goto to target block
    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, mma_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "wgmma_mma_m64n64k16_f32_bf16 call without target block".to_string()
            )
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{CUSTOM_DESCRIPTOR_UNSUPPORTED, MMA_UNSUPPORTED, unsupported_diagnostic};

    #[test]
    fn unsupported_wgmma_paths_are_exact() {
        assert_eq!(
            unsupported_diagnostic("cuda_device::wgmma::make_smem_desc_custom"),
            Some(CUSTOM_DESCRIPTOR_UNSUPPORTED)
        );
        for path in [
            "cuda_device::wgmma::wgmma_mma_m64n64k16_f32_bf16",
            "cuda_device::wgmma::wgmma_mma_m64n64k16_f32_f16",
            "cuda_device::wgmma::wgmma_mma_m64n64k16_f32_tf32",
        ] {
            assert_eq!(unsupported_diagnostic(path), Some(MMA_UNSUPPORTED));
        }

        for path in [
            "cuda_device::wgmma::make_smem_desc",
            "cuda_device::wgmma::wgmma_fence",
            "cuda_device::wgmma::wgmma_mma_m64n64k16_f32_bf16_extra",
            "other_crate::wgmma::wgmma_mma_m64n64k16_f32_bf16",
        ] {
            assert_eq!(unsupported_diagnostic(path), None);
        }
    }
}
