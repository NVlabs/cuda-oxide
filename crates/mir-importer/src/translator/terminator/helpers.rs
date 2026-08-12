/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Common helper functions for terminator translation.
//!
//! This module contains utility functions shared across terminator handlers:
//!
//! - [`emit_goto`]: Unconditional zero-operand branch to a target block.
//! - [`emit_store_result_and_goto`]: Write an intrinsic result to the
//!   destination local's slot, then branch to the success target.
//! - [`emit_function_call`]: General function call emission.
//! - [`emit_generated_nvvm_intrinsic`]: Zero-operand NVVM intrinsic emission
//!   for a catalog intrinsic, carrying its generated ABI marker.
//! - [`emit_unit_noop_intrinsic`]: Compiler-hint intrinsics with no codegen effect.
//! - [`insert_op`]: Common operation insertion pattern.

use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_mir::{
    ops::{MirCallOp, MirConstructArrayOp, MirGotoOp},
    types::MirArrayType,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::value::Value;
use rustc_public::mir;

/// Emits a zero-operand `mir.goto` to the target block.
///
/// Non-entry blocks carry no arguments; cross-block data flow travels
/// through the per-local alloca slots instead.
pub fn emit_goto(
    ctx: &mut Context,
    target_idx: usize,
    prev_op: Ptr<Operation>,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> Ptr<Operation> {
    let target_block = block_map[target_idx];
    let goto_op = Operation::new(
        ctx,
        MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![target_block],
        0,
    );
    goto_op.deref_mut(ctx).set_loc(loc);
    goto_op.insert_after(ctx, prev_op);
    goto_op
}

/// Writes `value` into `destination`, honouring a projection on it.
///
/// A bare local goes to its slot, as before. A projected destination needs the
/// address of the place rather than of the local, because storing to the local
/// would overwrite the whole aggregate (or the pointer itself) instead of the
/// part the call names.
///
/// Only the single-element projections a call destination is observed to carry
/// are modelled. Anything else is refused rather than written to the wrong
/// place, since a store aimed at the wrong address is a miscompile and an
/// unsupported-construct error is not.
///
/// Known fidelity gap: rustc evaluates the destination address *before* the
/// call, but this path materializes it *after* the call op. The difference is
/// observable only from custom MIR where the callee mutates the destination's
/// base local through a `&mut` argument.
pub fn store_result_to_place(
    ctx: &mut Context,
    body: &mir::Body,
    destination: &mir::Place,
    value: Value,
    value_map: &mut ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Ptr<Operation>,
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    use crate::translator::statement::{
        emit_array_element_store, pointer_address_space, pointer_is_mutable,
        reject_raw_field_index_on_enum_pointee, slot_array_element_ty,
    };
    use dialect_mir::ops::{MirFieldAddrOp, MirStoreOp};

    match destination.projection.as_slice() {
        [] => Ok(value_map
            .store_local(ctx, destination.local, value, block_ptr, Some(prev_op))
            .unwrap_or(prev_op)),

        [mir::ProjectionElem::Deref] => {
            let base_place = mir::Place {
                local: destination.local,
                projection: vec![],
            };
            let (ptr_val, after_ptr) = rvalue::translate_place(
                ctx,
                body,
                &base_place,
                value_map,
                block_ptr,
                Some(prev_op),
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
            store_op.deref_mut(ctx).set_loc(loc);
            store_op.insert_after(ctx, after_ptr.unwrap_or(prev_op));
            Ok(store_op)
        }

        [mir::ProjectionElem::Field(field_idx, field_ty)] => {
            let Some(slot) = value_map.get_slot(destination.local) else {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Local {:?} has no alloca slot for a field destination",
                        destination.local
                    ))
                );
            };
            reject_raw_field_index_on_enum_pointee(ctx, slot, &destination.projection, &loc)?;

            let field_type = crate::translator::types::translate_type(ctx, field_ty)?;
            let field_ptr_ty = dialect_mir::types::MirPtrType::get(
                ctx,
                field_type,
                pointer_is_mutable(ctx, slot),
                pointer_address_space(ctx, slot),
            )
            .into();
            let field_addr_op = Operation::new(
                ctx,
                MirFieldAddrOp::get_concrete_op_info(),
                vec![field_ptr_ty],
                vec![slot],
                vec![],
                0,
            );
            field_addr_op.deref_mut(ctx).set_loc(loc.clone());
            MirFieldAddrOp::new(field_addr_op).set_attr_field_index(
                ctx,
                dialect_mir::attributes::FieldIndexAttr(*field_idx as u32),
            );
            field_addr_op.insert_after(ctx, prev_op);
            let field_ptr = field_addr_op.deref(ctx).get_result(0);

            let store_op = Operation::new(
                ctx,
                MirStoreOp::get_concrete_op_info(),
                vec![],
                vec![field_ptr, value],
                vec![],
                0,
            );
            store_op.deref_mut(ctx).set_loc(loc);
            store_op.insert_after(ctx, field_addr_op);
            Ok(store_op)
        }

        [mir::ProjectionElem::Index(index_local)] => {
            let Some(arr_ptr) = value_map.get_slot(destination.local) else {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Local {:?} has no alloca slot for a runtime index destination",
                        destination.local
                    ))
                );
            };
            let index_place = mir::Place {
                local: *index_local,
                projection: vec![],
            };
            let (index_value, after_index) = rvalue::translate_place(
                ctx,
                body,
                &index_place,
                value_map,
                block_ptr,
                Some(prev_op),
                loc.clone(),
            )?;
            let (element_ty, address_space) = slot_array_element_ty(ctx, arr_ptr, &loc)?;
            Ok(emit_array_element_store(
                ctx,
                arr_ptr,
                index_value,
                value,
                element_ty,
                address_space,
                block_ptr,
                after_index,
                loc,
            ))
        }

        [
            mir::ProjectionElem::ConstantIndex {
                offset,
                min_length: _,
                from_end,
            },
        ] => {
            if *from_end {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(
                        "ConstantIndex with from_end=true is not supported for a call destination"
                            .to_string()
                    )
                );
            }
            let Some(arr_ptr) = value_map.get_slot(destination.local) else {
                return input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Local {:?} has no alloca slot for an array element destination",
                        destination.local
                    ))
                );
            };
            let (element_ty, address_space) = slot_array_element_ty(ctx, arr_ptr, &loc)?;

            let i64_ty = IntegerType::get(ctx, 64, Signedness::Signed);
            let index_attr = pliron::builtin::attributes::IntegerAttr::new(
                i64_ty,
                pliron::utils::apint::APInt::from_i64(
                    *offset as i64,
                    std::num::NonZeroUsize::new(64).unwrap(),
                ),
            );
            let const_op = Operation::new(
                ctx,
                dialect_mir::ops::MirConstantOp::get_concrete_op_info(),
                vec![i64_ty.into()],
                vec![],
                vec![],
                0,
            );
            const_op.deref_mut(ctx).set_loc(loc.clone());
            dialect_mir::ops::MirConstantOp::new(const_op).set_attr_value(ctx, index_attr);
            const_op.insert_after(ctx, prev_op);
            let index_value = const_op.deref(ctx).get_result(0);

            Ok(emit_array_element_store(
                ctx,
                arr_ptr,
                index_value,
                value,
                element_ty,
                address_space,
                block_ptr,
                Some(const_op),
                loc,
            ))
        }

        projection => input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "call destination with projection {:?} is not supported",
                projection
            ))
        ),
    }
}

/// Stores `result_value` into `destination`'s slot and emits a branch to
/// `target`.
///
/// Shared "write result + branch to success block" epilogue for intrinsic
/// handlers. The store is emitted after `prev_op`; the goto chains after the
/// store (or after `prev_op` directly if the destination is a ZST with no
/// backing slot). Returns the goto operation.
#[allow(clippy::too_many_arguments)]
pub fn emit_store_result_and_goto(
    ctx: &mut Context,
    destination: &mir::Place,
    result_value: Value,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Ptr<Operation>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    no_target_msg: &str,
) -> TranslationResult<Ptr<Operation>> {
    // This epilogue has no `body`, so it cannot compute the address of a
    // projected place the way [`store_result_to_place`] does. Refusing is the
    // alternative to storing to `destination.local`, which for `x.0 = f()` or
    // `(*p) = f()` writes the result over the whole aggregate or over the
    // pointer. Most callers are generated, so the signature stays as it is.
    if !destination.projection.is_empty() {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "intrinsic result written to a projected destination {:?} is not supported",
                destination.projection
            ))
        );
    }

    let goto_prev = value_map
        .store_local(
            ctx,
            destination.local,
            result_value,
            block_ptr,
            Some(prev_op),
        )
        .unwrap_or(prev_op);

    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, goto_prev, block_map, loc))
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(no_target_msg.to_string())
        )
    }
}

/// Inserts an operation after the previous one, or at the front of the block.
///
/// This is a common pattern used throughout terminator translation.
#[inline]
pub fn insert_op(
    ctx: &mut Context,
    op: Ptr<Operation>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
) {
    match prev_op {
        Some(prev) => op.insert_after(ctx, prev),
        None => op.insert_at_front(block_ptr, ctx),
    }
}

/// Attach the exact generated-intrinsic ABI marker to a typed dialect op.
pub fn set_generated_intrinsic_marker(ctx: &mut Context, op: Ptr<Operation>, marker: &str) {
    use pliron::builtin::attributes::StringAttr;
    use pliron::identifier::Identifier;

    op.deref_mut(ctx).attributes.set(
        Identifier::try_from(cuda_oxide_codegen::__private::GENERATED_INTRINSIC_MARKER_ATTR)
            .expect("generated intrinsic marker attribute key must be a valid identifier"),
        StringAttr::new(marker.to_owned()),
    );
}

/// Bundle a generated operation's independent `u32` results into the Rust
/// array value expected by its raw ABI.
///
/// Keeping this adapter here lets later multi-result families reuse the same
/// SSA-to-array boundary without introducing a stack temporary.
pub fn bundle_generated_u32_results_as_array(
    ctx: &mut Context,
    producer: Ptr<Operation>,
    result_count: usize,
    loc: Location,
) -> (Value, Ptr<Operation>) {
    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
    let values = (0..result_count)
        .map(|index| producer.deref(ctx).get_result(index))
        .collect();
    let array_ty = MirArrayType::get(ctx, u32_ty.into(), result_count as u64);
    let array = Operation::new(
        ctx,
        MirConstructArrayOp::get_concrete_op_info(),
        vec![array_ty.into()],
        values,
        vec![],
        0,
    );
    array.deref_mut(ctx).set_loc(loc);
    array.insert_after(ctx, producer);
    (array.deref(ctx).get_result(0), array)
}

/// Emits a regular (non-intrinsic) function call.
///
/// # Process
///
/// 1. Translate all MIR arguments to Pliron IR values
/// 2. Create a `mir.call` operation carrying the callee's name attribute
/// 3. Store the result into the destination local's slot
/// 4. Emit a zero-operand goto to the call's success target
///
/// Reference arguments (`&mut local`) are handed the local's alloca slot
/// pointer directly, so callee writes through the reference are observed by
/// subsequent loads in the caller without any explicit reload plumbing.
#[allow(clippy::too_many_arguments)]
pub fn emit_function_call(
    ctx: &mut Context,
    body: &mir::Body,
    callee_name: &str,
    args: &[mir::Operand],
    destination: &mir::Place,
    return_type: pliron::r#type::TypeHandle,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let mut arg_values = Vec::new();
    let mut last_op = prev_op;

    for arg in args {
        let (arg_value, arg_last_op) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        arg_values.push(arg_value);
        last_op = arg_last_op;
    }

    use pliron::builtin::attributes::StringAttr;

    let call_op = Operation::new(
        ctx,
        MirCallOp::get_concrete_op_info(),
        vec![return_type],
        arg_values,
        vec![],
        0,
    );
    call_op.deref_mut(ctx).set_loc(loc.clone());

    let callee_attr = StringAttr::new(callee_name.into());
    call_op.deref_mut(ctx).attributes.set(
        pliron::identifier::Identifier::try_from("callee").unwrap(),
        callee_attr,
    );

    let call_op = if let Some(prev) = last_op {
        call_op.insert_after(ctx, prev);
        call_op
    } else {
        call_op.insert_at_front(block_ptr, ctx);
        call_op
    };

    let result_value = call_op.deref(ctx).get_result(0);

    let goto_prev = store_result_to_place(
        ctx,
        body,
        destination,
        result_value,
        value_map,
        block_ptr,
        call_op,
        loc.clone(),
    )?;

    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, goto_prev, block_map, loc))
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("Call terminator without target not supported".to_string(),)
        )
    }
}

/// Emits a generated zero-operand NVVM operation returning `u32` and attaches
/// its exact generated-intrinsic ABI marker.
#[allow(clippy::too_many_arguments)]
pub fn emit_generated_nvvm_intrinsic(
    ctx: &mut Context,
    opid: (
        fn(pliron::context::Ptr<pliron::operation::Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
    marker: &str,
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    emit_nvvm_integer_intrinsic(
        ctx,
        opid,
        32,
        Some(marker),
        destination,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
    )
}

/// Emits a generated zero-operand NVVM operation returning `u64` and attaches
/// its exact generated-intrinsic ABI marker.
#[allow(clippy::too_many_arguments)]
pub fn emit_generated_nvvm_intrinsic_u64(
    ctx: &mut Context,
    opid: (
        fn(pliron::context::Ptr<pliron::operation::Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
    marker: &str,
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    emit_nvvm_integer_intrinsic(
        ctx,
        opid,
        64,
        Some(marker),
        destination,
        target,
        block_ptr,
        prev_op,
        value_map,
        block_map,
        loc,
    )
}

#[allow(clippy::too_many_arguments)]
fn emit_nvvm_integer_intrinsic(
    ctx: &mut Context,
    opid: (
        fn(pliron::context::Ptr<pliron::operation::Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
    result_width: u32,
    generated_marker: Option<&str>,
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    let result_type = IntegerType::get(ctx, result_width, Signedness::Unsigned);

    let nvvm_op = Operation::new(ctx, opid, vec![result_type.to_handle()], vec![], vec![], 0);
    nvvm_op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(marker) = generated_marker {
        set_generated_intrinsic_marker(ctx, nvvm_op, marker);
    }

    let last_op = if let Some(prev) = prev_op {
        nvvm_op.insert_after(ctx, prev);
        nvvm_op
    } else {
        nvvm_op.insert_at_front(block_ptr, ctx);
        nvvm_op
    };

    let result_value = nvvm_op.deref(ctx).get_result(0);

    let goto_prev = value_map
        .store_local(
            ctx,
            destination.local,
            result_value,
            block_ptr,
            Some(last_op),
        )
        .unwrap_or(last_op);

    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, goto_prev, block_map, loc))
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported("Call terminator without target not supported".to_string(),)
        )
    }
}

/// Emits a unit-returning intrinsic that has no codegen effect on GPU.
///
/// Used for compiler-hint intrinsics like `core::intrinsics::cold_path` whose
/// semantics are purely advisory. We materialize a unit value for the MIR
/// destination and continue to the target block without emitting a real call.
#[allow(clippy::too_many_arguments)]
pub fn emit_unit_noop_intrinsic(
    ctx: &mut Context,
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    intrinsic_name: &str,
) -> TranslationResult<Ptr<Operation>> {
    let unit_ty = dialect_mir::types::MirTupleType::get(ctx, vec![]);
    let unit_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructTupleOp::get_concrete_op_info(),
        vec![unit_ty.into()],
        vec![],
        vec![],
        0,
    );
    unit_op.deref_mut(ctx).set_loc(loc.clone());
    insert_op(ctx, unit_op, block_ptr, prev_op);

    let unit_val = unit_op.deref(ctx).get_result(0);
    let goto_prev = value_map
        .store_local(ctx, destination.local, unit_val, block_ptr, Some(unit_op))
        .unwrap_or(unit_op);

    if let Some(target_idx) = target {
        Ok(emit_goto(ctx, *target_idx, goto_prev, block_map, loc))
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "{} call without target not supported",
                intrinsic_name
            ))
        )
    }
}
