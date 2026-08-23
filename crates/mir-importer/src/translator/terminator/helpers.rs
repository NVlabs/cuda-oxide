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
//!   destination place, then branch to the success target.
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

/// Writes `value` into `destination`, honouring its full projection chain.
///
/// A bare local goes to its slot, as before. A projected destination is lowered
/// through the same place-address walker used by references and projected
/// assignments, then written with one `mir.store`. This keeps call-result
/// storage aligned with the normal MIR place semantics instead of maintaining
/// a second projection dispatcher here.
///
/// If the shared walker cannot materialize a writable address, fail closed.
/// Falling back to the base local would overwrite the wrong storage and is a
/// miscompile.
///
/// Known fidelity gap: rustc evaluates the destination address *before* the
/// call, but this path materializes it *after* the call op. The difference is
/// observable only from custom MIR where the callee mutates the destination's
/// base local through a `&mut` argument.
pub fn store_result_to_place(
    ctx: &mut Context,
    destination: &mir::Place,
    value: Value,
    value_map: &mut ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Ptr<Operation>,
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    use dialect_mir::ops::MirStoreOp;

    if destination.projection.is_empty() {
        return Ok(value_map
            .store_local(ctx, destination.local, value, block_ptr, Some(prev_op))
            .unwrap_or(prev_op));
    }

    let Some((destination_address, address_prev)) = rvalue::translate_place_address(
        ctx,
        value_map,
        destination,
        /* is_mutable */ true,
        block_ptr,
        Some(prev_op),
        loc.clone(),
    )?
    else {
        return input_err!(
            loc,
            TranslationErr::unsupported(format!(
                "cannot compute writable address for call destination projection {:?}",
                destination.projection
            ))
        );
    };

    let store_op = Operation::new(
        ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![destination_address, value],
        vec![],
        0,
    );
    store_op.deref_mut(ctx).set_loc(loc);
    store_op.insert_after(ctx, address_prev.unwrap_or(prev_op));
    Ok(store_op)
}

/// Stores `result_value` through `destination` and emits a branch to
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
    // Keep every intrinsic-result path on the same place-address grammar.
    let goto_prev = store_result_to_place(
        ctx,
        destination,
        result_value,
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

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

/// Mark an aggregate as the compiler-created Rust ABI bundle for one
/// multi-result device operation.
pub fn set_compiler_result_bundle_marker(ctx: &mut Context, op: Ptr<Operation>) {
    use dialect_mir::attributes::{COMPILER_RESULT_BUNDLE_ATTR_KEY, CompilerResultBundleAttr};
    use pliron::identifier::Identifier;

    op.deref_mut(ctx).attributes.set(
        Identifier::try_from(COMPILER_RESULT_BUNDLE_ATTR_KEY)
            .expect("compiler result bundle attribute key must be a valid identifier"),
        CompilerResultBundleAttr(true),
    );
}

/// Bundle a generated operation's independent `u32` results into the Rust
/// array value expected by its raw ABI and mark the compiler-owned adapter
/// for result forwarding.
///
/// This helper is only for compiler-generated multi-result carriers. Ordinary
/// Rust arrays must never receive the forwarding marker.
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
    set_compiler_result_bundle_marker(ctx, array);
    array.insert_after(ctx, producer);
    (array.deref(ctx).get_result(0), array)
}

/// Emits a regular (non-intrinsic) function call.
///
/// # Process
///
/// 1. Translate all MIR arguments to Pliron IR values
/// 2. Create a `mir.call` operation carrying the callee's name attribute
/// 3. Store the result through the destination place
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

    let goto_prev = store_result_to_place(
        ctx,
        destination,
        result_value,
        value_map,
        block_ptr,
        last_op,
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
    let goto_prev = store_result_to_place(
        ctx,
        destination,
        unit_val,
        value_map,
        block_ptr,
        unit_op,
        loc.clone(),
    )?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::{
        attributes::{COMPILER_RESULT_BUNDLE_ATTR_KEY, CompilerResultBundleAttr},
        ops::MirFuncOp,
    };
    use pliron::{
        builtin::{
            attributes::{StringAttr, TypeAttr},
            op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
            ops::ModuleOp,
            types::FunctionType,
        },
        identifier::Identifier,
        region::Region,
        r#type::TypeHandle,
    };

    #[test]
    fn generated_u32_result_array_is_marked_for_forwarding() {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);

        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let module = ModuleOp::new(&mut ctx, "test".try_into().unwrap());
        let function_type = FunctionType::get(&ctx, vec![], vec![]);
        let function = Operation::new(
            &mut ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function_op = MirFuncOp::new(&mut ctx, function, TypeAttr::new(function_type.into()));
        function_op.set_symbol_name(&mut ctx, "kernel".try_into().unwrap());
        module.append_operation(&mut ctx, function, 0);

        let region: Ptr<Region> = function.deref(&ctx).get_region(0);
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        block.insert_at_back(region, &ctx);

        let producer = Operation::new(
            &mut ctx,
            MirCallOp::get_concrete_op_info(),
            vec![u32_ty; 2],
            vec![],
            vec![],
            0,
        );
        MirCallOp::new(producer)
            .set_attr_callee(&ctx, StringAttr::new("register_pair".to_string()));
        producer.insert_at_back(block, &ctx);

        let loc = producer.deref(&ctx).loc().clone();
        let (_, bundle) = bundle_generated_u32_results_as_array(&mut ctx, producer, 2, loc);

        let key = Identifier::try_from(COMPILER_RESULT_BUNDLE_ATTR_KEY).unwrap();
        let is_marked = {
            let bundle_op = bundle.deref(&ctx);
            bundle_op
                .attributes
                .get::<CompilerResultBundleAttr>(&key)
                .is_some_and(|marker| marker.0)
        };

        assert!(
            is_marked,
            "generated result bundle must carry the forwarding marker"
        );
        assert_eq!(
            bundle.deref(&ctx).get_operand(0),
            producer.deref(&ctx).get_result(0)
        );
        assert_eq!(
            bundle.deref(&ctx).get_operand(1),
            producer.deref(&ctx).get_result(1)
        );
    }

    fn projected_result_fixture(
        ctx: &mut Context,
    ) -> (
        Ptr<BasicBlock>,
        Ptr<BasicBlock>,
        ValueMap,
        Ptr<Operation>,
        mir::Place,
    ) {
        dialect_mir::register(ctx);

        let module = ModuleOp::new(ctx, "test".try_into().unwrap());
        let function_type = FunctionType::get(ctx, vec![], vec![]);
        let function = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function_op = MirFuncOp::new(ctx, function, TypeAttr::new(function_type.into()));
        function_op.set_symbol_name(ctx, "kernel".try_into().unwrap());
        module.append_operation(ctx, function, 0);

        let region: Ptr<Region> = function.deref(ctx).get_region(0);
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(region, ctx);
        let target = BasicBlock::new(ctx, None, vec![]);
        target.insert_at_back(region, ctx);

        let u32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let array_ty: TypeHandle = MirArrayType::get(ctx, u32_ty, 2).into();
        let mut value_map = ValueMap::new(1);
        let (alloca, slot) = ValueMap::emit_alloca(ctx, array_ty, block, None);
        value_map.set_slot(0, slot);

        let destination = mir::Place {
            local: 0,
            projection: vec![mir::ProjectionElem::ConstantIndex {
                offset: 1,
                min_length: 2,
                from_end: false,
            }],
        };

        (block, target, value_map, alloca, destination)
    }

    #[test]
    fn store_result_epilogue_accepts_projected_destination() {
        use dialect_mir::ops::MirUndefOp;

        let mut ctx = Context::new();
        let (block, target, mut value_map, alloca, destination) =
            projected_result_fixture(&mut ctx);

        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let producer = MirUndefOp::new(&mut ctx, u32_ty).get_operation();
        producer.insert_after(&ctx, alloca);
        let result_value = producer.deref(&ctx).get_result(0);

        let result = emit_store_result_and_goto(
            &mut ctx,
            &destination,
            result_value,
            &Some(1),
            block,
            producer,
            &mut value_map,
            &[block, target],
            Location::Unknown,
            "test intrinsic without target",
        );

        assert!(
            result.is_ok(),
            "shared intrinsic epilogue must store through projected destinations"
        );
    }

    #[test]
    fn generated_nvvm_helper_accepts_projected_destination() {
        use dialect_mir::ops::MirUndefOp;

        let mut ctx = Context::new();
        let (block, target, mut value_map, alloca, destination) =
            projected_result_fixture(&mut ctx);

        let result = emit_generated_nvvm_intrinsic(
            &mut ctx,
            MirUndefOp::get_concrete_op_info(),
            "test.generated.projected",
            &destination,
            &Some(1),
            block,
            Some(alloca),
            &mut value_map,
            &[block, target],
            Location::Unknown,
        );

        assert!(
            result.is_ok(),
            "generated zero-operand intrinsic must store through projected destinations"
        );
    }
}
