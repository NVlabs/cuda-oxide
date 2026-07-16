/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Memory access and conversion intrinsics.

use super::super::helpers::{emit_goto, emit_store_result_and_goto};
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::values::ValueMap;
use crate::translator::{rvalue, types};
use dialect_mir::attributes::MirCastKindAttr;
use dialect_mir::ops::{MirCastOp, MirConstantOp, MirDivOp, MirSubOp};
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::types::IntegerType;
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::Typed;
use pliron::utils::apint::APInt;
use pliron::value::Value;
use rustc_public::mir;
use std::num::NonZeroUsize;
/// Emits `core::intrinsics::volatile_load::<T>(ptr)`, which backs
/// `core::ptr::read_volatile`.
#[allow(clippy::too_many_arguments)]
pub fn emit_volatile_load(
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
    use dialect_mir::ops::MirLoadOp;
    use dialect_mir::types::MirPtrType;

    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "volatile_load expects 1 argument (ptr), got {}",
                args.len()
            ))
        );
    }

    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let elem_ty = {
        let ptr_ty = ptr_val.get_type(ctx);
        let ptr_ty_obj = ptr_ty.deref(ctx);
        match ptr_ty_obj.downcast_ref::<MirPtrType>() {
            Some(mir_ptr) => mir_ptr.pointee,
            None => {
                return input_err!(
                    loc.clone(),
                    TranslationErr::unsupported(format!(
                        "volatile_load: expected pointer operand, got {:?}",
                        ptr_ty_obj
                    ))
                );
            }
        }
    };

    let load_op = Operation::new(
        ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![elem_ty],
        vec![ptr_val],
        vec![],
        0,
    );
    load_op.deref_mut(ctx).set_loc(loc.clone());
    MirLoadOp::new(load_op).set_volatile(ctx, true);

    if let Some(prev) = last_op {
        load_op.insert_after(ctx, prev);
    } else {
        load_op.insert_at_front(block_ptr, ctx);
    }

    let result = load_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        load_op,
        value_map,
        block_map,
        loc,
        "volatile_load call without target block",
    )
}

/// Emits `core::intrinsics::volatile_store::<T>(ptr, value)`, which backs
/// `core::ptr::write_volatile`.
#[allow(clippy::too_many_arguments)]
pub fn emit_volatile_store(
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
    use dialect_mir::ops::MirStoreOp;
    use dialect_mir::types::MirPtrType;

    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "volatile_store expects 2 arguments (ptr, value), got {}",
                args.len()
            ))
        );
    }

    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    {
        let ptr_ty = ptr_val.get_type(ctx);
        let ptr_ty_obj = ptr_ty.deref(ctx);
        if ptr_ty_obj.downcast_ref::<MirPtrType>().is_none() {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "volatile_store: expected pointer operand, got {:?}",
                    ptr_ty_obj
                ))
            );
        }
    }

    let (value, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let store_op = Operation::new(
        ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![ptr_val, value],
        vec![],
        0,
    );
    store_op.deref_mut(ctx).set_loc(loc.clone());
    MirStoreOp::new(store_op).set_volatile(ctx, true);

    if let Some(prev) = last_op {
        store_op.insert_after(ctx, prev);
    } else {
        store_op.insert_at_front(block_ptr, ctx);
    }

    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, store_op, block_map, loc))
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("volatile_store call without target block".to_string())
        )
    }
}

#[derive(Clone, Copy)]
enum PtrOffsetFromResult {
    Signed,
    Unsigned,
}

impl PtrOffsetFromResult {
    fn result_type(
        self,
        ctx: &mut Context,
    ) -> pliron::r#type::TypedHandle<pliron::builtin::types::IntegerType> {
        match self {
            Self::Signed => types::get_isize_type(ctx),
            Self::Unsigned => types::get_usize_type(ctx),
        }
    }

    fn intrinsic_name(self) -> &'static str {
        match self {
            Self::Signed => "ptr_offset_from",
            Self::Unsigned => "ptr_offset_from_unsigned",
        }
    }

    fn missing_target_message(self) -> &'static str {
        match self {
            Self::Signed => "ptr_offset_from call without target block",
            Self::Unsigned => "ptr_offset_from_unsigned call without target block",
        }
    }
}

/// Emits `core::intrinsics::ptr_offset_from::<T>(this, other) -> isize`.
///
/// Computes `(this.addr() - other.addr()) / size_of::<T>()`.
#[allow(clippy::too_many_arguments)]
pub fn emit_ptr_offset_from(
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
    emit_ptr_offset_from_with_result(
        ctx,
        body,
        args,
        destination,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        PtrOffsetFromResult::Signed,
    )
}

/// Emits `core::intrinsics::ptr_offset_from_unsigned::<T>(this, other) -> usize`.
///
/// Computes `(this.addr() - other.addr()) / size_of::<T>()`. The intrinsic
/// contract guarantees `this >= other` and an exact multiple.
#[allow(clippy::too_many_arguments)]
pub fn emit_ptr_offset_from_unsigned(
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
    emit_ptr_offset_from_with_result(
        ctx,
        body,
        args,
        destination,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
        PtrOffsetFromResult::Unsigned,
    )
}

#[allow(clippy::too_many_arguments)]
fn emit_ptr_offset_from_with_result(
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
    result_kind: PtrOffsetFromResult,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{} expects 2 arguments (this, other), got {}",
                result_kind.intrinsic_name(),
                args.len()
            ))
        );
    }

    let elem_size = pointee_size_bytes(body, &args[0], result_kind.intrinsic_name(), loc.clone())?;
    let result_ty = result_kind.result_type(ctx);
    let result_type = result_ty.to_handle();

    let (this_ptr, op_after_this) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let this_addr = emit_pointer_expose_address(
        ctx,
        this_ptr,
        result_type,
        op_after_this,
        block_ptr,
        loc.clone(),
    );

    let (other_ptr, op_after_other) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        Some(this_addr),
        loc.clone(),
    )?;
    let other_addr = emit_pointer_expose_address(
        ctx,
        other_ptr,
        result_type,
        op_after_other,
        block_ptr,
        loc.clone(),
    );

    let this_addr_val = this_addr.deref(ctx).get_result(0);
    let other_addr_val = other_addr.deref(ctx).get_result(0);
    let sub_op = Operation::new(
        ctx,
        MirSubOp::get_concrete_op_info(),
        vec![result_type],
        vec![this_addr_val, other_addr_val],
        vec![],
        0,
    );
    sub_op.deref_mut(ctx).set_loc(loc.clone());
    sub_op.insert_after(ctx, other_addr);
    let byte_diff = sub_op.deref(ctx).get_result(0);

    let size_const = emit_integer_constant(ctx, result_ty, elem_size, sub_op, loc.clone())?;
    let size_val = size_const.deref(ctx).get_result(0);

    let div_op = Operation::new(
        ctx,
        MirDivOp::get_concrete_op_info(),
        vec![result_type],
        vec![byte_diff, size_val],
        vec![],
        0,
    );
    div_op.deref_mut(ctx).set_loc(loc.clone());
    div_op.insert_after(ctx, size_const);
    let result = div_op.deref(ctx).get_result(0);

    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        div_op,
        value_map,
        block_map,
        loc,
        result_kind.missing_target_message(),
    )
}

fn pointee_size_bytes(
    body: &mir::Body,
    operand: &mir::Operand,
    intrinsic_name: &str,
    loc: Location,
) -> TranslationResult<u64> {
    use rustc_public::ty::{RigidTy, TyKind};

    let operand_ty = match operand {
        mir::Operand::Copy(place) | mir::Operand::Move(place) => place.ty(body.locals()).ok(),
        mir::Operand::Constant(constant) => Some(constant.const_.ty()),
        mir::Operand::RuntimeChecks(_) => None,
    };
    let Some(operand_ty) = operand_ty else {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{intrinsic_name}: cannot determine pointer operand type"
            ))
        );
    };

    let pointee = match operand_ty.kind() {
        TyKind::RigidTy(RigidTy::RawPtr(pointee, _))
        | TyKind::RigidTy(RigidTy::Ref(_, pointee, _)) => pointee,
        other => {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "{intrinsic_name}: expected raw pointer or reference operand, got {other:?}"
                ))
            );
        }
    };

    let layout = match pointee.layout() {
        Ok(layout) => layout,
        Err(err) => {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "{intrinsic_name}: failed to query pointee layout: {err:?}"
                ))
            );
        }
    };
    let size = layout.shape().size.bytes() as u64;
    if size == 0 {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "{intrinsic_name}: zero-sized pointee type has no element distance"
            ))
        );
    }
    Ok(size)
}

fn emit_pointer_expose_address(
    ctx: &mut Context,
    ptr: Value,
    result_type: pliron::r#type::TypeHandle,
    insert_after: Option<Ptr<Operation>>,
    block_ptr: Ptr<BasicBlock>,
    loc: Location,
) -> Ptr<Operation> {
    let cast_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![result_type],
        vec![ptr],
        vec![],
        0,
    );
    cast_op.deref_mut(ctx).set_loc(loc);
    MirCastOp::new(cast_op).set_attr_cast_kind(ctx, MirCastKindAttr::PointerExposeAddress);
    if let Some(prev) = insert_after {
        cast_op.insert_after(ctx, prev);
    } else {
        cast_op.insert_at_front(block_ptr, ctx);
    }
    cast_op
}

fn emit_integer_constant(
    ctx: &mut Context,
    ty: pliron::r#type::TypedHandle<IntegerType>,
    value: u64,
    insert_after: Ptr<Operation>,
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let value_i64 = i64::try_from(value).map_err(|_| {
        pliron::input_error!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "ptr_offset_from pointee size {value} does not fit in isize"
            ))
        )
    })?;
    let bits = ty.deref(ctx).width() as usize;
    let apint = APInt::from_i64(value_i64, NonZeroUsize::new(bits).unwrap());
    let size_attr = IntegerAttr::new(ty, apint);
    let const_op = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![ty.to_handle()],
        vec![],
        vec![],
        0,
    );
    const_op.deref_mut(ctx).set_loc(loc);
    MirConstantOp::new(const_op).set_attr_value(ctx, size_attr);
    const_op.insert_after(ctx, insert_after);
    Ok(const_op)
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
