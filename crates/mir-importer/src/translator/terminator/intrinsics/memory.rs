/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Memory access and conversion intrinsics.
//!
//! Handles shared memory indexing, matrix stores, and type conversions.

use super::super::helpers::{emit_goto, emit_store_result_and_goto};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_mir::attributes::MirCastKindAttr;
use dialect_mir::ops::MirCastOp;
use dialect_nvvm::ops::{
    CvtF32x2Bf16x2Op, StmatrixM8n8X2Op, StmatrixM8n8X2TransOp, StmatrixM8n8X4Op,
    StmatrixM8n8X4TransOp,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::Typed;
use rustc_public::mir;
/// Emits `stmatrix.m8n8.x4`: Warp-cooperative matrix store (4 tiles).
///
/// Stores 4 matrix tiles (32 columns) to shared memory using the warp-cooperative
/// stmatrix instruction. Each thread contributes its fragment data.
///
/// # Arguments
///
/// - `args[0]`: `*mut u8` - Destination pointer in shared memory
/// - `args[1-4]`: `u32` - Register values (r0, r1, r2, r3)
///
/// # PTX Instruction
///
/// `stmatrix.sync.aligned.m8n8.x4.shared.b16`
pub fn emit_stmatrix_m8n8_x4(
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
    if args.len() != 5 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "stmatrix_m8n8_x4 expects 5 arguments (smem_ptr, r0, r1, r2, r3), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(5);

    for arg in args.iter().take(5) {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }

    let st_op = Operation::new(
        ctx,
        StmatrixM8n8X4Op::get_concrete_op_info(),
        vec![],
        operands,
        vec![],
        0,
    );
    st_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        st_op.insert_after(ctx, prev);
    } else {
        st_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, st_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("stmatrix_m8n8_x4 call without target block".to_string())
        )
    }
}

/// Emit stmatrix_m8n8_x4_trans: Warp-cooperative matrix store with transpose.
///
/// This version uses the `.trans` modifier to transform data from fragment
/// layout to row-major layout during the store operation.
///
/// Args: (smem_ptr: *mut u8, r0: u32, r1: u32, r2: u32, r3: u32)
///       where each u32 contains 2 packed bf16 values
/// Returns: void
pub fn emit_stmatrix_m8n8_x4_trans(
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
    if args.len() != 5 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "stmatrix_m8n8_x4_trans expects 5 arguments (smem_ptr, r0, r1, r2, r3), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(5);

    for arg in args.iter().take(5) {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }

    let st_op = Operation::new(
        ctx,
        StmatrixM8n8X4TransOp::get_concrete_op_info(),
        vec![],
        operands,
        vec![],
        0,
    );
    st_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        st_op.insert_after(ctx, prev);
    } else {
        st_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, st_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "stmatrix_m8n8_x4_trans call without target block".to_string()
            )
        )
    }
}

/// Emit tcgen05_ld_16x256b_x8_pure: Pure TMEM load returning 32 f32 values.
///
/// Unlike emit_tcgen05_ld_16x256b_x8, this returns values in registers (no SMEM store).
/// The result is a struct with 32 f32 values that can be used for subsequent operations.
///
/// Args: (tmem_addr: u32)
pub fn emit_stmatrix_m8n8_x2(
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
                "stmatrix_m8n8_x2 expects 3 arguments (smem_ptr, r0, r1), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(3);

    for arg in args.iter().take(3) {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }

    let st_op = Operation::new(
        ctx,
        StmatrixM8n8X2Op::get_concrete_op_info(),
        vec![],
        operands,
        vec![],
        0,
    );
    st_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        st_op.insert_after(ctx, prev);
    } else {
        st_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, st_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("stmatrix_m8n8_x2 call without target block".to_string())
        )
    }
}

/// Emit stmatrix.m8n8.x2.trans - TRANSPOSE version matching cuBLAS STSM.16.MT88.2.
///
/// Args: (smem_ptr: *mut u8, r0: u32, r1: u32)
///       where each u32 contains 2 packed bf16 values
/// Returns: void
pub fn emit_stmatrix_m8n8_x2_trans(
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
                "stmatrix_m8n8_x2_trans expects 3 arguments (smem_ptr, r0, r1), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(3);

    for arg in args.iter().take(3) {
        let (val, last_op_after) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = last_op_after;
        operands.push(val);
    }

    let st_op = Operation::new(
        ctx,
        StmatrixM8n8X2TransOp::get_concrete_op_info(),
        vec![],
        operands,
        vec![],
        0,
    );
    st_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        st_op.insert_after(ctx, prev);
    } else {
        st_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        let goto_op = emit_goto(ctx, *target_idx, st_op, block_map, loc);
        Ok(goto_op)
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "stmatrix_m8n8_x2_trans call without target block".to_string()
            )
        )
    }
}

/// Emit cvt_f32x2_bf16x2: Convert two f32 to packed bf16x2.
///
/// Args: (a: f32, b: f32)
pub fn emit_cvt_f32x2_bf16x2(
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
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "cvt_f32x2_bf16x2 expects 2 arguments (a: f32, b: f32), got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;

    // arg[0]: a (f32)
    let (a_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // arg[1]: b (f32)
    let (b_val, last_op_after) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = last_op_after;

    // Result is u32 (packed bf16x2); Rust-side signature is `u32` and the
    // destination local is unsigned, so match that here to avoid the
    // MirStoreOp verifier flagging a signless-vs-unsigned mismatch.
    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);

    let cvt_op = Operation::new(
        ctx,
        CvtF32x2Bf16x2Op::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![a_val, b_val],
        vec![],
        0,
    );
    cvt_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        cvt_op.insert_after(ctx, prev);
    } else {
        cvt_op.insert_at_front(block_ptr, ctx);
    }

    let result = cvt_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        cvt_op,
        value_map,
        block_map,
        loc,
        "cvt_f32x2_bf16x2 call without target block",
    )
}

/// Emits `SharedArray::index()`: Compute pointer to element in shared memory.
///
/// Computes `base_ptr + index` to get a pointer to the indexed element.
/// The result is a shared memory pointer (address space 3).
///
/// # Arguments
///
/// - `args[0]`: `&mut SharedArray<T, N>` - Reference to the shared array
/// - `args[1]`: `usize` - Index into the array
///
/// # Returns
///
/// `*mut T` (addrspace=3) - Pointer to the element in shared memory
pub fn emit_shared_array_index(
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
    _is_mut: bool,
) -> TranslationResult<Ptr<Operation>> {
    use dialect_mir::ops::MirPtrOffsetOp;

    // Args should be: [&mut SharedArray<T, N>, usize]
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "SharedArray::index expects 2 arguments, got {}",
                args.len()
            ))
        );
    }

    // Translate both arguments through the uniform operand helper. It handles
    // `Copy`/`Move`/`Constant` for us, so a literal `smem[0]` (where the index
    // is `Operand::Constant`) and a direct `&raw mut SMEM` reference (where
    // arg 0 is a constant pointer to a `SharedArray<T, N>` static) both work.
    // Earlier this function had two manual `Copy | Move => ...; _ => bail`
    // matches that rejected constant operands. See
    // `.cursor/rules/compiler-gaps-are-bugs.mdc` for why we add the missing
    // arm via the framework helper rather than asking callers to introduce a
    // `let tmp = 0; smem[tmp]` shim.
    let (shared_array_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let (index_val, last_op_after_index) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    let mut last_op = last_op_after_index;

    // The shared_array_val is a pointer to the shared memory array.
    // We need to compute ptr + index to get a pointer to the element.
    // The result should be a shared memory pointer (addrspace 3).
    let ptr_ty = shared_array_val.get_type(ctx);

    // Create ptr offset operation
    let offset_op = Operation::new(
        ctx,
        MirPtrOffsetOp::get_concrete_op_info(),
        vec![ptr_ty], // Result type is same pointer type
        vec![shared_array_val, index_val],
        vec![],
        0,
    );
    offset_op.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        offset_op.insert_after(ctx, prev);
    } else {
        offset_op.insert_at_front(block_ptr, ctx);
    }
    last_op = Some(offset_op);

    let result_ptr = offset_op.deref(ctx).get_result(0);

    let prev = last_op.expect("should have at least offset_op");
    emit_store_result_and_goto(
        ctx,
        destination,
        result_ptr,
        target,
        block_ptr,
        prev,
        value_map,
        block_map,
        loc,
        "SharedArray::index call without target block",
    )
}

/// Emits `SharedArray::as_ptr` or `as_mut_ptr` - returns pointer to shared memory.
///
/// This converts the shared memory address (addrspace 3) to a generic pointer (addrspace 0)
/// following LLVM's opaque pointer model where generic pointers can hold any address space.
///
/// # Arguments
///
/// - `args[0]`: `&SharedArray<T, N>` - Reference to the shared memory array
///
/// # Returns
///
/// `*const T` or `*mut T` - Generic pointer to the shared memory
#[allow(clippy::too_many_arguments)]
pub fn emit_shared_array_as_ptr(
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
    use dialect_mir::types::MirPtrType;

    if args.is_empty() {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "SharedArray::as_ptr expects 1 argument (self), got 0".to_string(),
            )
        );
    }

    // Translate the self argument (shared memory pointer)
    let (shared_ptr, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Get the element type from the shared pointer type
    let elem_ty = {
        let shared_ptr_ty = shared_ptr.get_type(ctx);
        let shared_ptr_obj = shared_ptr_ty.deref(ctx);

        if let Some(mir_ptr) = shared_ptr_obj.downcast_ref::<MirPtrType>() {
            mir_ptr.pointee
        } else {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "SharedArray::as_ptr: expected MirPtrType, got {:?}",
                    shared_ptr_obj
                ))
            );
        }
    }; // shared_ptr_obj borrow ends here

    // Create generic pointer type (addrspace 0) with same element type
    // For simplicity, we use immutable here - mutability is just a Rust concept
    let generic_ptr_ty = MirPtrType::get(ctx, elem_ty, false, 0);

    // Emit cast: shared (3) -> generic (0)
    // This is an addrspace cast but we use MirCastOp which is generic enough
    let cast_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![generic_ptr_ty.into()],
        vec![shared_ptr],
        vec![],
        0,
    );
    cast_op.deref_mut(ctx).set_loc(loc.clone());
    MirCastOp::new(cast_op).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);

    if let Some(prev) = last_op {
        cast_op.insert_after(ctx, prev);
    } else {
        cast_op.insert_at_front(block_ptr, ctx);
    }

    let result_ptr = cast_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_ptr,
        target,
        block_ptr,
        cast_op,
        value_map,
        block_map,
        loc,
        "SharedArray::as_ptr call without target block",
    )
}

// ============================================================================
// DynamicSharedArray (extern shared memory) operations
// ============================================================================

/// Emits `DynamicSharedArray::<T, ALIGN>::get()` or `DynamicSharedArray::<T, ALIGN>::get_raw()`.
///
/// Creates a reference to the extern shared memory global at byte offset 0.
/// The alignment is specified by the ALIGN const generic parameter.
///
/// # PTX Output
///
/// ```ptx
/// .extern .shared .align ALIGN .b8 __dynamic_smem[];
/// // Returns pointer to __dynamic_smem
/// ```
#[allow(clippy::too_many_arguments)]
pub fn emit_dynamic_shared_get(
    ctx: &mut Context,
    body: &mir::Body,
    _args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    byte_offset: u64,
    alignment: u64,
) -> TranslationResult<Ptr<Operation>> {
    use dialect_mir::ops::MirExternSharedOp;
    use dialect_mir::types::MirPtrType;

    // Get the destination type to determine the pointer element type
    // DynamicSharedArray::get() returns *mut T, so the destination is a raw pointer type
    // We need to get the pointee type from it
    let dest_ty = body.locals()[destination.local].ty;

    // Get pointee type from the raw pointer return type
    let pointee_ty = match dest_ty.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::RawPtr(pointee, _)) => pointee,
        _ => {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "DynamicSharedArray::get expected pointer return type, got {:?}",
                    dest_ty
                ))
            );
        }
    };

    let elem_ty = crate::translator::types::translate_type(ctx, &pointee_ty)?;

    // Create a shared memory pointer type (addrspace 3)
    // We use generic pointer type since MirExternSharedOp result will be cast
    let ptr_ty = MirPtrType::get_shared(ctx, elem_ty, true).into();

    // Create MirExternSharedOp
    let op = Operation::new(
        ctx,
        MirExternSharedOp::get_concrete_op_info(),
        vec![ptr_ty],
        vec![],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    let extern_shared = MirExternSharedOp::new(op);

    // Set byte offset (0 for get/get_raw)
    extern_shared.set_byte_offset_value(ctx, byte_offset);

    // Set alignment from the ALIGN const generic (default 16, matches nvcc)
    extern_shared.set_alignment_value(ctx, alignment);

    if let Some(prev) = prev_op {
        extern_shared.get_operation().insert_after(ctx, prev);
    } else {
        extern_shared
            .get_operation()
            .insert_at_front(block_ptr, ctx);
    }

    let result_ptr = extern_shared.get_operation().deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_ptr,
        target,
        block_ptr,
        extern_shared.get_operation(),
        value_map,
        block_map,
        loc,
        "DynamicSharedArray::get call without target block",
    )
}

/// Emits `DynamicSharedArray::<T, ALIGN>::offset(byte_offset)`.
///
/// Creates a reference to the extern shared memory global at the specified byte offset.
/// The alignment is specified by the ALIGN const generic parameter.
///
/// # Arguments
///
/// - `args[0]`: `byte_offset: usize` - Byte offset into dynamic shared memory
/// - `alignment`: Base alignment from the ALIGN const generic
#[allow(clippy::too_many_arguments)]
pub fn emit_dynamic_shared_offset(
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
    alignment: u64,
) -> TranslationResult<Ptr<Operation>> {
    use dialect_mir::ops::MirExternSharedOp;
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    // Get the destination type to determine the pointer element type
    // DynamicSharedArray::offset() returns *mut T, so the destination is a raw pointer type
    let dest_ty = body.locals()[destination.local].ty;

    // Get pointee type from the raw pointer return type
    let pointee_ty = match dest_ty.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::RawPtr(pointee, _)) => pointee,
        _ => {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "DynamicSharedArray::offset expected pointer return type, got {:?}",
                    dest_ty
                ))
            );
        }
    };

    let elem_ty = crate::translator::types::translate_type(ctx, &pointee_ty)?;

    // Create a shared memory pointer type (addrspace 3)
    let ptr_ty = MirPtrType::get_shared(ctx, elem_ty, true).into();

    // Create MirExternSharedOp - we'll handle offset in two ways:
    // 1. If offset is a constant, store it as an attribute
    // 2. If offset is dynamic, we need to emit a GEP after the base pointer

    // First create the base extern shared op
    let op = Operation::new(
        ctx,
        MirExternSharedOp::get_concrete_op_info(),
        vec![ptr_ty],
        vec![],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());

    let extern_shared = MirExternSharedOp::new(op);
    // Set alignment from the ALIGN const generic (default 16, matches nvcc)
    extern_shared.set_alignment_value(ctx, alignment);

    if let Some(prev) = prev_op {
        extern_shared.get_operation().insert_after(ctx, prev);
    } else {
        extern_shared
            .get_operation()
            .insert_at_front(block_ptr, ctx);
    }

    let base_ptr = extern_shared.get_operation().deref(ctx).get_result(0);

    // Now handle the offset
    // If we have an argument, translate it and emit a ptr_offset op
    let (final_ptr, last_op) = if !args.is_empty() {
        // Translate the byte_offset argument
        let (offset_val, offset_last_op) = rvalue::translate_operand(
            ctx,
            body,
            &args[0],
            value_map,
            block_ptr,
            Some(extern_shared.get_operation()),
            loc.clone(),
        )?;

        // Create a byte pointer type for GEP
        let i8_ty = IntegerType::get(ctx, 8, Signedness::Unsigned);
        let byte_ptr_ty = MirPtrType::get_shared(ctx, i8_ty.into(), true);

        // First cast to byte pointer
        let cast_to_byte = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![byte_ptr_ty.into()],
            vec![base_ptr],
            vec![],
            0,
        );
        cast_to_byte.deref_mut(ctx).set_loc(loc.clone());
        MirCastOp::new(cast_to_byte).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
        if let Some(prev) = offset_last_op {
            cast_to_byte.insert_after(ctx, prev);
        } else {
            cast_to_byte.insert_after(ctx, extern_shared.get_operation());
        }

        let byte_ptr = cast_to_byte.deref(ctx).get_result(0);

        // Emit ptr_offset with byte offset
        let offset_op = Operation::new(
            ctx,
            dialect_mir::ops::MirPtrOffsetOp::get_concrete_op_info(),
            vec![byte_ptr_ty.into()],
            vec![byte_ptr, offset_val],
            vec![],
            0,
        );
        offset_op.deref_mut(ctx).set_loc(loc.clone());
        offset_op.insert_after(ctx, cast_to_byte);

        let offset_ptr = offset_op.deref(ctx).get_result(0);

        // Cast back to target element type
        let cast_to_elem = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![ptr_ty],
            vec![offset_ptr],
            vec![],
            0,
        );
        cast_to_elem.deref_mut(ctx).set_loc(loc.clone());
        MirCastOp::new(cast_to_elem).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
        cast_to_elem.insert_after(ctx, offset_op);

        let final_ptr = cast_to_elem.deref(ctx).get_result(0);
        (final_ptr, cast_to_elem)
    } else {
        // No offset argument - use base pointer directly
        (base_ptr, extern_shared.get_operation())
    };

    emit_store_result_and_goto(
        ctx,
        destination,
        final_ptr,
        target,
        block_ptr,
        last_op,
        value_map,
        block_map,
        loc,
        "DynamicSharedArray::offset call without target block",
    )
}
