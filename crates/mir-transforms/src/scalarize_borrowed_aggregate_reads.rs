/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Canonicalize bounded read-only aggregate projections before LLVM lowering.
//!
//! rustc MIR parameters are imported through entry-block slots:
//!
//! ```text
//! %slot = mir.alloca
//! mir.store %slot, %argument
//! ...
//! %field = mir.field_addr %slot, field
//! %elem = mir.array_element_addr %field, %index
//! %value = mir.load %elem
//! ```
//!
//! `mir.field_addr` is intentionally non-promotable, so the ordinary mem2reg
//! pass cannot recover the already-available SSA argument. For a compiler-owned
//! entry slot initialized exactly once from an entry-block argument, this pass
//! validates the complete pointer-use graph and rewrites read-only array loads
//! to value operations:
//!
//! ```text
//! %array = mir.extract_field %argument, field
//! %value = mir.extract_array_element %array, %index
//! ```
//!
//! This pass canonicalizes pointer-based read-only access independently of the
//! runtime index shape. The later `mir.extract_array_element` lowering owns the
//! profitability decision: a bounded `urem value, constant` becomes fixed
//! `extractvalue` candidates plus a select chain, while unsupported indices keep
//! the ordinary memory fallback.
//!
//! The pre-mem2reg phase fails closed on pointer provenance and mutation. It
//! rejects additional stores, volatile loads, mutable derived pointers, calls,
//! pointer casts, pointer PHIs/selects, unknown users, non-array fields, and
//! projections in the entry block before the initializer can be proven to
//! dominate them.
//!
//! A second, post-mem2reg phase handles immutable aggregate pointer arguments
//! such as an `&self` device helper. It accepts only an exact single-use chain:
//!
//! ```text
//! %field = mir.field_addr %aggregate_ptr, field
//! %elem = mir.array_element_addr %field, %index
//! %value = mir.load %elem
//! ```
//!
//! The rewrite retains the constant field projection, loads only that array
//! field as a value, and replaces the dynamic scalar address with
//! `mir.extract_array_element`.
//!
//! The index must either be a bounded unsigned remainder or be guarded by the
//! unique predecessor `mir.assert(mir.lt(index, constant))`. Guarded indices are
//! canonicalized to an equivalent remainder in the assertion-success block so
//! the existing typed `mir.extract_array_element` lowering can scalarize them.
//!
//! The second phase widens one dynamic element load into a load of the whole
//! array field, which is only legal when the pointed-to aggregate lives in
//! caller-private memory. Borrowed pointers that may reference global, shared,
//! or otherwise external memory must keep exactly one dynamic memory access
//! (issue #400, following the #398 precedent). The phase therefore proves
//! caller provenance before rewriting a helper: every `mir.call` of the helper
//! in the module must pass a pointer traceable, through pointer-identity
//! casts, to a caller-local `mir.alloca` of exactly the helper's aggregate
//! type. Kernel pointer parameters, phi/select merges, pointers forwarded
//! from another helper's parameter, shared/global allocations, externally
//! callable device exports, and helpers without a single visible call site
//! all fail closed and keep the original dynamic load.

use std::{collections::HashMap, num::NonZeroUsize};

use dialect_mir::{
    attributes::{FieldIndexAttr, MirCastKindAttr},
    ops::{
        MAX_SCALARIZED_CANDIDATES, MirAllocaOp, MirArrayElementAddrOp, MirAssertOp, MirAssignOp,
        MirCallOp, MirCastOp, MirCondBranchOp, MirConstantOp, MirConstructArrayOp,
        MirConstructEnumOp, MirConstructSliceOp, MirConstructStructOp, MirConstructTupleOp,
        MirEnumPayloadOp, MirEqOp, MirExtractArrayElementOp, MirExtractFieldOp, MirFieldAddrOp,
        MirFuncOp, MirInsertFieldOp, MirLoadOp, MirLtOp, MirNeOp, MirNotOp, MirPtrOffsetOp,
        MirRemOp, MirStoreOp, MirSubOp,
    },
    types::{MirArrayType, MirPtrType, MirStructType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::IntegerAttr,
        op_interfaces::{BranchOpInterface, SymbolOpInterface},
        types::{IntegerType, Signedness},
    },
    context::{Context, Ptr},
    graph::ControlFlowGraph,
    irbuild::{
        listener::Recorder,
        rewriter::{IRRewriter, Rewriter},
    },
    linked_list::ContainsLinkedList,
    location::Located,
    op::{Op, op_cast},
    operation::Operation,
    r#type::{TypeHandle, Typed, TypedHandle},
    utils::apint::APInt,
    value::Value,
};

#[derive(Clone)]
struct LoadRewrite {
    load: Ptr<Operation>,
    field_index: u32,
    index: Value,
    array_type: TypeHandle,
    result_type: TypeHandle,
}

struct AllocaPlan {
    aggregate_value: Value,
    field_addrs: Vec<Ptr<Operation>>,
    array_addrs: Vec<Ptr<Operation>>,
    loads: Vec<LoadRewrite>,
}

/// Rewrite read-only indexed aggregate argument loads before mem2reg.
///
/// Only entry-block allocas initialized from an argument of the same block are
/// considered. Every pointer use must belong to the exact read-only projection
/// graph accepted by `analyze_alloca`.
///
/// `verbose` is threaded from the pipeline's backend options; the pass itself
/// never reads the environment.
pub fn canonicalize_read_only_aggregate_arguments(
    module: Ptr<Operation>,
    ctx: &mut Context,
    verbose: bool,
) {
    let mut ops = Vec::new();
    collect_ops(ctx, module, &mut ops);

    let allocas: Vec<_> = ops
        .into_iter()
        .filter(|op| Operation::get_op::<MirAllocaOp>(*op, ctx).is_some())
        .collect();

    let mut rewritten_loads = 0usize;
    for alloca in allocas {
        let Some(plan) = analyze_alloca(ctx, alloca) else {
            continue;
        };
        rewritten_loads += rewrite_plan(ctx, plan);
    }

    if rewritten_loads > 0 && verbose {
        eprintln!("borrowed-aggregate scalarization: rewrote {rewritten_loads} dynamic load(s)");
    }
}

fn collect_ops(ctx: &Context, root: Ptr<Operation>, output: &mut Vec<Ptr<Operation>>) {
    output.push(root);
    let regions: Vec<_> = root.deref(ctx).regions().collect();
    for region in regions {
        let blocks: Vec<_> = region.deref(ctx).iter(ctx).collect();
        for block in blocks {
            let children: Vec<_> = block.deref(ctx).iter(ctx).collect();
            for child in children {
                collect_ops(ctx, child, output);
            }
        }
    }
}

/// Validate one entry-block aggregate slot without mutating the IR.
fn analyze_alloca(ctx: &Context, alloca: Ptr<Operation>) -> Option<AllocaPlan> {
    let alloca_op = Operation::get_op::<MirAllocaOp>(alloca, ctx)?;
    let pointee = alloca_op.pointee_type(ctx);
    pointee.deref(ctx).downcast_ref::<MirStructType>()?;

    let alloca_block = alloca.deref(ctx).get_parent_block()?;
    let root = alloca.deref(ctx).get_result(0);
    let block_arguments: Vec<_> = alloca_block.deref(ctx).arguments().collect();

    let mut aggregate_value = None;
    let mut field_addrs = Vec::new();
    let mut array_addrs = Vec::new();
    let mut loads = Vec::new();

    for root_use in root.uses(ctx) {
        let user = root_use.user_op();
        let operand_index = root_use.find_index(ctx);

        if let Some(store) = Operation::get_op::<MirStoreOp>(user, ctx) {
            if operand_index != 0
                || store.is_volatile(ctx)
                || user.deref(ctx).get_parent_block() != Some(alloca_block)
                || aggregate_value.is_some()
            {
                return None;
            }

            let stored_value = store.value_opd(ctx);
            if !block_arguments.contains(&stored_value) {
                return None;
            }
            aggregate_value = Some(stored_value);
            continue;
        }

        let field = Operation::get_op::<MirFieldAddrOp>(user, ctx)?;
        if operand_index != 0 || user.deref(ctx).get_parent_block() == Some(alloca_block) {
            return None;
        }

        analyze_field_path(ctx, field, &mut field_addrs, &mut array_addrs, &mut loads)?;
    }

    Some(AllocaPlan {
        aggregate_value: aggregate_value?,
        field_addrs,
        array_addrs,
        loads: (!loads.is_empty()).then_some(loads)?,
    })
}

fn analyze_field_path(
    ctx: &Context,
    field: MirFieldAddrOp,
    field_addrs: &mut Vec<Ptr<Operation>>,
    array_addrs: &mut Vec<Ptr<Operation>>,
    loads: &mut Vec<LoadRewrite>,
) -> Option<()> {
    let field_op = field.get_operation();
    let field_index = field.get_attr_field_index(ctx)?.0;
    let field_pointer = field_op.deref(ctx).get_result(0);
    let field_pointer_type = field_pointer.get_type(ctx);
    let field_pointer_type_ref = field_pointer_type.deref(ctx);
    let field_pointer_type = field_pointer_type_ref.downcast_ref::<MirPtrType>()?;
    if field_pointer_type.is_mutable {
        return None;
    }

    let array_type = field_pointer_type.pointee;
    let array_type_ref = array_type.deref(ctx);
    let array_type_info = array_type_ref.downcast_ref::<MirArrayType>()?;
    if array_type_info.size() == 0 {
        return None;
    }

    let mut local_array_addrs = Vec::new();
    let mut local_loads = Vec::new();

    for field_use in field_pointer.uses(ctx) {
        let array_op = field_use.user_op();
        if field_use.find_index(ctx) != 0 {
            return None;
        }

        Operation::get_op::<MirArrayElementAddrOp>(array_op, ctx)?;
        let array_pointer = array_op.deref(ctx).get_result(0);
        let array_pointer_type = array_pointer.get_type(ctx);
        let array_pointer_type_ref = array_pointer_type.deref(ctx);
        let array_pointer_type = array_pointer_type_ref.downcast_ref::<MirPtrType>()?;
        if array_pointer_type.is_mutable {
            return None;
        }

        let index = array_op.deref(ctx).get_operand(1);
        let mut found_load = false;
        for array_use in array_pointer.uses(ctx) {
            let load_op = array_use.user_op();
            if array_use.find_index(ctx) != 0 {
                return None;
            }
            let load = Operation::get_op::<MirLoadOp>(load_op, ctx)?;
            if load.is_volatile(ctx) {
                return None;
            }

            local_loads.push(LoadRewrite {
                load: load_op,
                field_index,
                index,
                array_type,
                result_type: load_op.deref(ctx).get_result(0).get_type(ctx),
            });
            found_load = true;
        }

        if !found_load {
            return None;
        }
        local_array_addrs.push(array_op);
    }

    if local_array_addrs.is_empty() || local_loads.is_empty() {
        return None;
    }

    field_addrs.push(field_op);
    array_addrs.extend(local_array_addrs);
    loads.extend(local_loads);
    Some(())
}

fn rewrite_plan(ctx: &mut Context, plan: AllocaPlan) -> usize {
    let load_count = plan.loads.len();
    let mut rewriter = IRRewriter::<Recorder>::default();

    for rewrite in plan.loads {
        let location = rewrite.load.deref(ctx).loc().clone();

        let extract_field = Operation::new(
            ctx,
            MirExtractFieldOp::get_concrete_op_info(),
            vec![rewrite.array_type],
            vec![plan.aggregate_value],
            vec![],
            0,
        );
        extract_field.deref_mut(ctx).set_loc(location.clone());
        MirExtractFieldOp::new(extract_field)
            .set_attr_index(ctx, FieldIndexAttr(rewrite.field_index));
        extract_field.insert_before(ctx, rewrite.load);
        let array_value = extract_field.deref(ctx).get_result(0);

        let extract_element = Operation::new(
            ctx,
            MirExtractArrayElementOp::get_concrete_op_info(),
            vec![rewrite.result_type],
            vec![array_value, rewrite.index],
            vec![],
            0,
        );
        extract_element.deref_mut(ctx).set_loc(location);
        extract_element.insert_before(ctx, rewrite.load);
        let replacement = extract_element.deref(ctx).get_result(0);

        let old_result = rewrite.load.deref(ctx).get_result(0);
        old_result.replace_all_uses_with(ctx, &replacement);
        rewriter.erase_operation(ctx, rewrite.load);
    }

    // Loads are gone, so the exact validated pointer chain is dead. Erase it
    // from leaves to root through the rewriter so linked-list bookkeeping and
    // use-list updates remain valid for the immediately following mem2reg pass.
    for array_addr in plan.array_addrs.into_iter().rev() {
        rewriter.erase_operation(ctx, array_addr);
    }
    for field_addr in plan.field_addrs.into_iter().rev() {
        rewriter.erase_operation(ctx, field_addr);
    }

    load_count
}

#[derive(Clone, Copy)]
enum BoundedPointerIndex {
    /// The original index is already `mir.rem(value, constant)`.
    Direct(Value),
    /// The assertion-success block proves `index < bound`. Re-materialize the
    /// equivalent remainder so the typed LLVM lowering sees the bounded shape.
    Asserted { index: Value, bound: Value },
}

struct BorrowedPointerPlan {
    field_pointer: Value,
    array_addr: Ptr<Operation>,
    load: Ptr<Operation>,
    array_type: TypeHandle,
    index: BoundedPointerIndex,
    result_type: TypeHandle,
}

/// Rewrite bounded read-only array loads through immutable aggregate pointer
/// arguments after mem2reg.
///
/// This phase is intentionally narrow. The aggregate pointer must be an entry
/// argument of an `alwaysinline` function, every derived pointer must be
/// immutable, and both pointer results must have exactly one use. The index
/// must be bounded either by an unsigned remainder or by the unique predecessor
/// assertion `assert(index < constant)`.
///
/// On top of the helper-local shape, every call site of the helper must prove
/// that the aggregate pointer targets caller-private memory (see
/// `all_call_sites_pass_owned_aggregate`); any unproven call site keeps the
/// helper untouched.
///
/// `verbose` is threaded from the pipeline's backend options; the pass itself
/// never reads the environment.
pub fn canonicalize_bounded_borrowed_pointer_arguments(
    module: Ptr<Operation>,
    ctx: &mut Context,
    verbose: bool,
) {
    let mut operations = Vec::new();
    collect_ops(ctx, module, &mut operations);

    let mut calls_by_callee: HashMap<String, Vec<Ptr<Operation>>> = HashMap::new();
    let mut array_addrs = Vec::new();
    for operation in operations {
        if let Some(call) = Operation::get_op::<MirCallOp>(operation, ctx) {
            let callee = call
                .get_attr_callee(ctx)
                .map(|attribute| String::from((*attribute).clone()));
            if let Some(callee) = callee {
                calls_by_callee.entry(callee).or_default().push(operation);
            }
            continue;
        }
        if Operation::get_op::<MirArrayElementAddrOp>(operation, ctx).is_some() {
            array_addrs.push(operation);
        }
    }

    let mut provenance_cache: HashMap<(Ptr<Operation>, usize), bool> = HashMap::new();
    let mut rewritten_loads = 0usize;
    for array_addr in array_addrs {
        let Some(plan) =
            analyze_borrowed_pointer_read(ctx, array_addr, &calls_by_callee, &mut provenance_cache)
        else {
            continue;
        };
        rewrite_borrowed_pointer_read(ctx, plan);
        rewritten_loads += 1;
    }

    if rewritten_loads > 0 && verbose {
        eprintln!(
            "borrowed-pointer aggregate scalarization: rewrote \
             {rewritten_loads} dynamic load(s)"
        );
    }
}

fn analyze_borrowed_pointer_read(
    ctx: &Context,
    array_addr: Ptr<Operation>,
    calls_by_callee: &HashMap<String, Vec<Ptr<Operation>>>,
    provenance_cache: &mut HashMap<(Ptr<Operation>, usize), bool>,
) -> Option<BorrowedPointerPlan> {
    Operation::get_op::<MirArrayElementAddrOp>(array_addr, ctx)?;
    let load_block = array_addr.deref(ctx).get_parent_block()?;

    let field_pointer = array_addr.deref(ctx).get_operand(0);
    let field_addr = field_pointer.defining_op()?;
    let field = Operation::get_op::<MirFieldAddrOp>(field_addr, ctx)?;
    if field_addr.deref(ctx).get_parent_block() != Some(load_block)
        || field_pointer.num_uses(ctx) != 1
    {
        return None;
    }
    let field_use = field_pointer.uses(ctx).into_iter().next()?;
    if field_use.user_op() != array_addr || field_use.find_index(ctx) != 0 {
        return None;
    }

    let field_pointer_type = field_pointer.get_type(ctx);
    let field_pointer_type_ref = field_pointer_type.deref(ctx);
    let field_pointer_type = field_pointer_type_ref.downcast_ref::<MirPtrType>()?;
    if field_pointer_type.is_mutable {
        return None;
    }
    let array_type = field_pointer_type.pointee;
    let array_type_ref = array_type.deref(ctx);
    let array_size = array_type_ref.downcast_ref::<MirArrayType>()?.size();
    if array_size == 0 {
        return None;
    }

    let element_pointer = array_addr.deref(ctx).get_result(0);
    let element_pointer_type = element_pointer.get_type(ctx);
    let element_pointer_type_ref = element_pointer_type.deref(ctx);
    let element_pointer_type = element_pointer_type_ref.downcast_ref::<MirPtrType>()?;
    if element_pointer_type.is_mutable || element_pointer.num_uses(ctx) != 1 {
        return None;
    }

    let element_use = element_pointer.uses(ctx).into_iter().next()?;
    if element_use.find_index(ctx) != 0 {
        return None;
    }
    let load = element_use.user_op();
    let load_op = Operation::get_op::<MirLoadOp>(load, ctx)?;
    if load_op.is_volatile(ctx) || load.deref(ctx).get_parent_block() != Some(load_block) {
        return None;
    }

    let aggregate_pointer = field_addr.deref(ctx).get_operand(0);
    let entry_block = aggregate_pointer.defining_block()?;
    let region = entry_block.deref(ctx).get_parent_region()?;
    if region.deref(ctx).iter(ctx).next() != Some(entry_block) {
        return None;
    }

    let function = entry_block.deref(ctx).get_parent_op(ctx)?;
    Operation::get_op::<MirFuncOp>(function, ctx)?;
    let alwaysinline_key: pliron::identifier::Identifier = "alwaysinline".try_into().ok()?;
    function
        .deref(ctx)
        .attributes
        .get::<pliron::builtin::attributes::StringAttr>(&alwaysinline_key)?;

    let aggregate_pointer_type = aggregate_pointer.get_type(ctx);
    let aggregate_pointer_type_ref = aggregate_pointer_type.deref(ctx);
    let aggregate_pointer_type = aggregate_pointer_type_ref.downcast_ref::<MirPtrType>()?;
    if aggregate_pointer_type.is_mutable {
        return None;
    }
    let aggregate_type = aggregate_pointer_type.pointee;
    aggregate_type.deref(ctx).downcast_ref::<MirStructType>()?;

    let argument_index = entry_block
        .deref(ctx)
        .arguments()
        .position(|argument| argument == aggregate_pointer)?;
    let provenance_key = (function, argument_index);
    let caller_owned = match provenance_cache.get(&provenance_key) {
        Some(&caller_owned) => caller_owned,
        None => {
            let caller_owned = all_call_sites_pass_owned_aggregate(
                ctx,
                calls_by_callee,
                function,
                argument_index,
                aggregate_type,
            );
            provenance_cache.insert(provenance_key, caller_owned);
            caller_owned
        }
    };
    if !caller_owned {
        return None;
    }

    let index_value = array_addr.deref(ctx).get_operand(1);
    let index = bounded_pointer_index(ctx, index_value, load_block, array_size)?;

    // Keep the field projection itself. Loading the bounded array field is
    // narrower than loading the complete aggregate and gives LLVM a constant
    // field address to forward after the helper is inlined.
    field.get_attr_field_index(ctx)?;

    Some(BorrowedPointerPlan {
        field_pointer,
        array_addr,
        load,
        array_type,
        index,
        result_type: load.deref(ctx).get_result(0).get_type(ctx),
    })
}

fn bounded_pointer_index(
    ctx: &Context,
    index: Value,
    load_block: Ptr<pliron::basic_block::BasicBlock>,
    array_size: u64,
) -> Option<BoundedPointerIndex> {
    let index_type = index.get_type(ctx);
    let index_type_ref = index_type.deref(ctx);
    let integer_type = index_type_ref.downcast_ref::<IntegerType>()?;
    if integer_type.signedness() != Signedness::Unsigned {
        return None;
    }

    if let Some(defining_op) = index.defining_op()
        && Operation::get_op::<MirRemOp>(defining_op, ctx).is_some()
    {
        let divisor = defining_op.deref(ctx).get_operand(1);
        let candidate_count = integer_constant_u64(ctx, divisor)?;
        validate_candidate_count(candidate_count, array_size)?;
        return Some(BoundedPointerIndex::Direct(index));
    }

    let region = load_block.deref(ctx).get_parent_region()?;
    let predecessors = region.predecessors(ctx, &load_block);
    let [assert_block] = predecessors.as_slice() else {
        return None;
    };
    let terminator = assert_block.deref(ctx).get_terminator(ctx)?;
    Operation::get_op::<MirAssertOp>(terminator, ctx)?;
    if terminator.deref(ctx).get_num_successors() != 1
        || terminator.deref(ctx).get_successor(0) != load_block
    {
        return None;
    }

    let condition = terminator.deref(ctx).get_operand(0);
    let comparison = condition.defining_op()?;
    Operation::get_op::<MirLtOp>(comparison, ctx)?;
    if comparison.deref(ctx).get_parent_block() != Some(*assert_block)
        || comparison.deref(ctx).get_operand(0) != index
    {
        return None;
    }

    let bound = comparison.deref(ctx).get_operand(1);
    if bound.get_type(ctx) != index_type {
        return None;
    }
    let candidate_count = integer_constant_u64(ctx, bound)?;
    validate_candidate_count(candidate_count, array_size)?;
    Some(BoundedPointerIndex::Asserted { index, bound })
}

/// Decide whether every visible call site of `function` passes the pointer
/// argument at `argument_index` into caller-private memory.
///
/// Issue #400 fail-closed rule (the #398 precedent): a borrowed pointer that
/// may reference global, shared, or otherwise external memory must keep
/// exactly one dynamic memory access, so the widened array-field load is only
/// legal when every caller passes the address of a compiler-owned local slot
/// holding exactly the helper's aggregate type. Externally callable device
/// exports have call sites this module cannot see, and a helper without any
/// visible call site proves nothing; both disqualify the helper outright.
fn all_call_sites_pass_owned_aggregate(
    ctx: &Context,
    calls_by_callee: &HashMap<String, Vec<Ptr<Operation>>>,
    function: Ptr<Operation>,
    argument_index: usize,
    aggregate_type: TypeHandle,
) -> bool {
    let Some(function_op) = Operation::get_op::<MirFuncOp>(function, ctx) else {
        return false;
    };
    let symbol = String::from(function_op.get_symbol_name(ctx));
    if reserved_oxide_symbols::is_device_symbol(&symbol) {
        return false;
    }
    let Some(calls) = calls_by_callee.get(&symbol) else {
        return false;
    };
    !calls.is_empty()
        && calls.iter().all(|call| {
            let call_ref = call.deref(ctx);
            if argument_index >= call_ref.get_num_operands() {
                return false;
            }
            let pointer = call_ref.get_operand(argument_index);
            drop(call_ref);
            pointer_is_owned_aggregate_slot(ctx, pointer, aggregate_type)
        })
}

/// Trace `pointer` back to its allocation through pointer-identity casts
/// (an `&mut slot -> &slot` reborrow imports as `mir.cast PtrToPtr`).
///
/// Accept only a function-local `mir.alloca` whose pointee is exactly the
/// helper's aggregate type: reading a whole array field stays inside the
/// allocation only when the slot and the callee agree on the layout. Block
/// arguments (kernel pointer parameters, phi merges, pointers forwarded from
/// another helper's parameter) and every other producer (shared or global
/// allocations, selects, offsets) fail closed.
fn pointer_is_owned_aggregate_slot(
    ctx: &Context,
    mut pointer: Value,
    aggregate_type: TypeHandle,
) -> bool {
    loop {
        let Some(defining_op) = pointer.defining_op() else {
            return false;
        };
        if let Some(cast) = Operation::get_op::<MirCastOp>(defining_op, ctx) {
            let is_pointer_identity_cast = cast
                .get_attr_cast_kind(ctx)
                .is_some_and(|kind| matches!(*kind, MirCastKindAttr::PtrToPtr));
            if !is_pointer_identity_cast {
                return false;
            }
            pointer = defining_op.deref(ctx).get_operand(0);
            continue;
        }
        let Some(alloca) = Operation::get_op::<MirAllocaOp>(defining_op, ctx) else {
            return false;
        };
        return alloca.pointee_type(ctx) == aggregate_type;
    }
}

fn integer_constant_u64(ctx: &Context, value: Value) -> Option<u64> {
    let defining_op = value.defining_op()?;
    if let Some(constant) = Operation::get_op::<MirConstantOp>(defining_op, ctx) {
        let attribute = constant.get_attr_value(ctx)?;
        let constant_value = attribute.value();
        // `APInt::to_u64` truncates wider values, so a >64-bit constant could be
        // misread as a small in-range bound. Fail closed on such widths.
        return (constant_value.bw() <= 64).then(|| constant_value.to_u64());
    }
    if Operation::get_op::<MirAssignOp>(defining_op, ctx).is_some() {
        return integer_constant_u64(ctx, defining_op.deref(ctx).get_operand(0));
    }
    if let Some(extract) = Operation::get_op::<MirExtractFieldOp>(defining_op, ctx)
        && extract
            .get_attr_index(ctx)
            .is_some_and(|index| index.0 == 1)
    {
        let aggregate = defining_op.deref(ctx).get_operand(0);
        if let Some(aggregate_op) = aggregate.defining_op()
            && Operation::get_op::<MirConstructSliceOp>(aggregate_op, ctx).is_some()
        {
            return integer_constant_u64(ctx, aggregate_op.deref(ctx).get_operand(1));
        }
    }
    None
}

fn validate_candidate_count(candidate_count: u64, array_size: u64) -> Option<()> {
    (candidate_count > 0
        && candidate_count <= array_size
        && candidate_count <= MAX_SCALARIZED_CANDIDATES)
        .then_some(())
}

fn rewrite_borrowed_pointer_read(ctx: &mut Context, plan: BorrowedPointerPlan) {
    let location = plan.load.deref(ctx).loc().clone();
    let bounded_index = match plan.index {
        BoundedPointerIndex::Direct(index) => index,
        BoundedPointerIndex::Asserted { index, bound } => {
            let remainder = Operation::new(
                ctx,
                MirRemOp::get_concrete_op_info(),
                vec![index.get_type(ctx)],
                vec![index, bound],
                vec![],
                0,
            );
            remainder.deref_mut(ctx).set_loc(location.clone());
            remainder.insert_before(ctx, plan.load);
            remainder.deref(ctx).get_result(0)
        }
    };

    // Load only the addressed array field at the original access point. The
    // source pointer is immutable and the helper is alwaysinline, so LLVM can
    // forward the caller's by-value aggregate after inlining. Keeping the
    // constant field projection avoids widening the access to the whole struct.
    let array_load = Operation::new(
        ctx,
        MirLoadOp::get_concrete_op_info(),
        vec![plan.array_type],
        vec![plan.field_pointer],
        vec![],
        0,
    );
    array_load.deref_mut(ctx).set_loc(location.clone());
    array_load.insert_before(ctx, plan.load);
    let array_value = array_load.deref(ctx).get_result(0);

    let extract_element = Operation::new(
        ctx,
        MirExtractArrayElementOp::get_concrete_op_info(),
        vec![plan.result_type],
        vec![array_value, bounded_index],
        vec![],
        0,
    );
    extract_element.deref_mut(ctx).set_loc(location);
    extract_element.insert_before(ctx, plan.load);
    let replacement = extract_element.deref(ctx).get_result(0);

    let old_result = plan.load.deref(ctx).get_result(0);
    old_result.replace_all_uses_with(ctx, &replacement);

    let mut rewriter = IRRewriter::<Recorder>::default();
    rewriter.erase_operation(ctx, plan.load);
    rewriter.erase_operation(ctx, plan.array_addr);
}

#[derive(Clone, Copy)]
struct ArrayAddressOrigin {
    alloca: Ptr<Operation>,
    element_type: TypeHandle,
    array_size: u64,
    offset: u64,
}

struct ElementwiseArrayInitialization {
    array_type: TypeHandle,
    element_type: TypeHandle,
    array_size: u64,
    values: Vec<Value>,
    stores: Vec<Ptr<Operation>>,
    element_addrs: Vec<Ptr<Operation>>,
    block: Ptr<BasicBlock>,
}

struct MemoryResidentIteratorPlan {
    iterator_initializer_store: Ptr<Operation>,
    array: ElementwiseArrayInitialization,
    initial_count: Value,
    current_count: Value,
    guard_compare: Ptr<Operation>,
    guard_is_eq: bool,
    element_load: Ptr<Operation>,
    index_type: TypeHandle,
}

/// Recover the post-mem2reg shape emitted by rustc for
/// `array.iter().copied().take(n)`.
///
/// rustc keeps the `Take<Copied<slice::Iter<T>>>` state in a local slot even
/// after ordinary mem2reg. The current pointer, end pointer, and `Take::n` are
/// therefore loaded and stored through nested `mir.field_addr` chains rather
/// than represented as loop block arguments. The important invariant is still
/// explicit in typed MIR:
///
/// - the iterator starts at element zero of one small local array,
/// - the end pointer is exactly `base + N`,
/// - the current pointer advances by exactly one element on the continue path,
/// - `Take::n` is checked for zero and decremented exactly once before that
///   pointer step.
///
/// For a successful element, the exact zero-based iteration index is therefore
/// `initial_take_n - current_take_n_before_decrement`. Rewriting the pointer
/// end check to the equivalent integer `index == N` and the payload load to
/// `mir.extract_array_element` makes the existing bounded-array lowering emit
/// the scalar select chain. The old iterator pointer machinery then becomes
/// dead and is removed by the normal optimization pipeline.
///
/// This recognizer is deliberately structural and fail-closed. It requires the
/// complete memory-resident adapter shape and never guesses from source names.
fn canonicalize_memory_resident_small_array_iterators(
    module: Ptr<Operation>,
    ctx: &mut Context,
) -> usize {
    let mut operations = Vec::new();
    collect_ops(ctx, module, &mut operations);
    let allocas: Vec<_> = operations
        .iter()
        .copied()
        .filter(|operation| Operation::get_op::<MirAllocaOp>(*operation, ctx).is_some())
        .collect();

    let mut rewritten = 0usize;
    for iterator_alloca in allocas {
        if iterator_alloca.deref(ctx).get_parent_block().is_none() {
            continue;
        }
        let Some(function) = iterator_alloca.deref(ctx).get_parent_op(ctx) else {
            continue;
        };
        if Operation::get_op::<MirFuncOp>(function, ctx).is_none() {
            continue;
        }
        let Some(plan) =
            analyze_memory_resident_small_array_iterator(ctx, function, iterator_alloca)
        else {
            continue;
        };
        rewrite_memory_resident_small_array_iterator(ctx, plan);
        rewritten += 1;
    }
    rewritten
}

fn struct_field_type(ctx: &Context, ty: TypeHandle, index: usize) -> Option<TypeHandle> {
    let ty_ref = ty.deref(ctx);
    let struct_ty = ty_ref.downcast_ref::<MirStructType>()?;
    struct_ty.get_field_type(index)
}

fn struct_field_count(ctx: &Context, ty: TypeHandle) -> Option<usize> {
    let ty_ref = ty.deref(ctx);
    Some(ty_ref.downcast_ref::<MirStructType>()?.field_count())
}

/// Validate the nested two-field / one-field / iterator state used by
/// `Take<Copied<slice::Iter<T>>>` without depending on unstable Rust type
/// names. The terminal iterator has `{ current_wrapper, end, ... }`, and the
/// current wrapper must contain exactly one pointer with the same pointee as
/// the end pointer.
fn memory_resident_iterator_layout(ctx: &Context, iterator_type: TypeHandle) -> Option<TypeHandle> {
    if struct_field_count(ctx, iterator_type)? != 2 {
        return None;
    }

    let copied_type = struct_field_type(ctx, iterator_type, 0)?;
    let count_type = struct_field_type(ctx, iterator_type, 1)?;
    let count_ref = count_type.deref(ctx);
    let count_integer = count_ref.downcast_ref::<IntegerType>()?;
    if count_integer.signedness() != Signedness::Unsigned {
        return None;
    }
    drop(count_ref);

    if struct_field_count(ctx, copied_type)? != 1 {
        return None;
    }
    let iter_type = struct_field_type(ctx, copied_type, 0)?;
    if struct_field_count(ctx, iter_type)? < 2 {
        return None;
    }

    let current_wrapper = struct_field_type(ctx, iter_type, 0)?;
    let end_pointer = struct_field_type(ctx, iter_type, 1)?;
    if struct_field_count(ctx, current_wrapper)? != 1 {
        return None;
    }
    let current_pointer = struct_field_type(ctx, current_wrapper, 0)?;

    let current_ref = current_pointer.deref(ctx);
    let current_ptr = current_ref.downcast_ref::<MirPtrType>()?;
    let current_pointee = current_ptr.pointee;
    drop(current_ref);

    let end_ref = end_pointer.deref(ctx);
    let end_ptr = end_ref.downcast_ref::<MirPtrType>()?;
    if end_ptr.pointee != current_pointee {
        return None;
    }

    Some(count_type)
}

fn direct_alloca_initializer_store(
    ctx: &Context,
    alloca: Ptr<Operation>,
) -> Option<(Ptr<Operation>, Value)> {
    let root = alloca.deref(ctx).get_result(0);
    let mut initializer = None;

    for root_use in root.uses(ctx) {
        if root_use.find_index(ctx) != 0 {
            continue;
        }
        let user = root_use.user_op();
        let Some(store) = Operation::get_op::<MirStoreOp>(user, ctx) else {
            continue;
        };
        if store.is_volatile(ctx) || initializer.is_some() {
            return None;
        }
        initializer = Some((user, store.value_opd(ctx)));
    }

    initializer
}

fn trace_array_address_origin(
    ctx: &Context,
    value: Value,
    depth: usize,
) -> Option<ArrayAddressOrigin> {
    if depth > 32 {
        return None;
    }
    let defining_op = value.defining_op()?;

    if let Some(alloca) = Operation::get_op::<MirAllocaOp>(defining_op, ctx) {
        let array_type = alloca.pointee_type(ctx);
        let array_ref = array_type.deref(ctx);
        let array = array_ref.downcast_ref::<MirArrayType>()?;
        let array_size = array.size();
        validate_candidate_count(array_size, array_size)?;
        return Some(ArrayAddressOrigin {
            alloca: defining_op,
            element_type: array.element_type(),
            array_size,
            offset: 0,
        });
    }

    if Operation::get_op::<MirAssignOp>(defining_op, ctx).is_some() {
        return trace_array_address_origin(ctx, defining_op.deref(ctx).get_operand(0), depth + 1);
    }

    if let Some(cast) = Operation::get_op::<MirCastOp>(defining_op, ctx) {
        let kind = cast.get_attr_cast_kind(ctx)?;
        if !matches!(
            *kind,
            MirCastKindAttr::Transmute
                | MirCastKindAttr::PtrToPtr
                | MirCastKindAttr::PointerCoercionUnsize
                | MirCastKindAttr::PointerCoercionArrayToPointer
                | MirCastKindAttr::PointerCoercionMutToConst
        ) {
            return None;
        }
        return trace_array_address_origin(ctx, defining_op.deref(ctx).get_operand(0), depth + 1);
    }

    if Operation::get_op::<MirPtrOffsetOp>(defining_op, ctx).is_some() {
        let increment = integer_constant_u64(ctx, defining_op.deref(ctx).get_operand(1))?;
        let mut origin =
            trace_array_address_origin(ctx, defining_op.deref(ctx).get_operand(0), depth + 1)?;
        origin.offset = origin.offset.checked_add(increment)?;
        return Some(origin);
    }

    if Operation::get_op::<MirArrayElementAddrOp>(defining_op, ctx).is_some() {
        let increment = integer_constant_u64(ctx, defining_op.deref(ctx).get_operand(1))?;
        let mut origin =
            trace_array_address_origin(ctx, defining_op.deref(ctx).get_operand(0), depth + 1)?;
        origin.offset = origin.offset.checked_add(increment)?;
        return Some(origin);
    }

    None
}

fn field_addr_root_path(ctx: &Context, value: Value) -> Option<(Ptr<Operation>, Vec<u32>)> {
    let mut current = value;
    let mut reversed_path = Vec::new();

    loop {
        let defining_op = current.defining_op()?;
        let Some(field) = Operation::get_op::<MirFieldAddrOp>(defining_op, ctx) else {
            Operation::get_op::<MirAllocaOp>(defining_op, ctx)?;
            reversed_path.reverse();
            return Some((defining_op, reversed_path));
        };
        reversed_path.push(field.get_attr_field_index(ctx)?.0);
        current = defining_op.deref(ctx).get_operand(0);
    }
}

fn collect_elementwise_array_initialization(
    ctx: &Context,
    function: Ptr<Operation>,
    alloca: Ptr<Operation>,
) -> Option<ElementwiseArrayInitialization> {
    let alloca_op = Operation::get_op::<MirAllocaOp>(alloca, ctx)?;
    let array_type = alloca_op.pointee_type(ctx);
    let array_ref = array_type.deref(ctx);
    let array = array_ref.downcast_ref::<MirArrayType>()?;
    let array_size = array.size();
    let element_type = array.element_type();
    validate_candidate_count(array_size, array_size)?;
    drop(array_ref);

    let root = alloca.deref(ctx).get_result(0);
    let mut values = vec![None; array_size as usize];
    let mut stores = Vec::with_capacity(array_size as usize);
    let mut element_addrs = Vec::with_capacity(array_size as usize);
    let mut initializer_block = None;

    for root_use in root.uses(ctx) {
        if root_use.find_index(ctx) != 0 {
            continue;
        }
        let user = root_use.user_op();
        if Operation::get_op::<MirArrayElementAddrOp>(user, ctx).is_none() {
            continue;
        }
        let index = integer_constant_u64(ctx, user.deref(ctx).get_operand(1))?;
        if index >= array_size {
            return None;
        }

        let address = user.deref(ctx).get_result(0);
        if address.num_uses(ctx) != 1 {
            return None;
        }
        let address_use = address.uses(ctx).into_iter().next()?;
        if address_use.find_index(ctx) != 0 {
            return None;
        }
        let store_op = address_use.user_op();
        let store = Operation::get_op::<MirStoreOp>(store_op, ctx)?;
        if store.is_volatile(ctx) {
            return None;
        }
        let value = store.value_opd(ctx);
        if value.get_type(ctx) != element_type || values[index as usize].replace(value).is_some() {
            return None;
        }

        let block = store_op.deref(ctx).get_parent_block()?;
        if initializer_block
            .replace(block)
            .is_some_and(|existing| existing != block)
        {
            return None;
        }
        stores.push(store_op);
        element_addrs.push(user);
    }

    if values.iter().any(Option::is_none) || stores.len() != array_size as usize {
        return None;
    }

    let mut operations = Vec::new();
    collect_ops(ctx, function, &mut operations);
    for operation in operations {
        if let Some(store) = Operation::get_op::<MirStoreOp>(operation, ctx) {
            if stores.contains(&operation) {
                continue;
            }
            if trace_array_address_origin(ctx, store.address_opd(ctx), 0)
                .is_some_and(|origin| origin.alloca == alloca)
            {
                return None;
            }
        }
        if Operation::get_op::<MirCallOp>(operation, ctx).is_some() {
            for operand in operation.deref(ctx).operands() {
                if trace_array_address_origin(ctx, operand, 0)
                    .is_some_and(|origin| origin.alloca == alloca)
                {
                    return None;
                }
            }
        }
    }

    Some(ElementwiseArrayInitialization {
        array_type,
        element_type,
        array_size,
        values: values.into_iter().map(Option::unwrap).collect(),
        stores,
        element_addrs,
        block: initializer_block?,
    })
}

fn initialization_dominates_iterator_store(
    ctx: &Context,
    initialization: &ElementwiseArrayInitialization,
    iterator_store: Ptr<Operation>,
) -> bool {
    let Some(iterator_block) = iterator_store.deref(ctx).get_parent_block() else {
        return false;
    };
    if iterator_block == initialization.block {
        let operations: Vec<_> = iterator_block.deref(ctx).iter(ctx).collect();
        let Some(anchor_index) = operations
            .iter()
            .position(|operation| *operation == iterator_store)
        else {
            return false;
        };
        return initialization.stores.iter().all(|store| {
            operations
                .iter()
                .position(|operation| operation == store)
                .is_some_and(|index| index < anchor_index)
        });
    }

    let Some(region) = iterator_block.deref(ctx).get_parent_region() else {
        return false;
    };
    let predecessors = region.predecessors(ctx, &iterator_block);
    predecessors.as_slice() == [initialization.block]
}

fn strip_iterator_identity_casts(ctx: &Context, mut value: Value) -> Option<Value> {
    for _ in 0..16 {
        let Some(defining_op) = value.defining_op() else {
            return Some(value);
        };
        if Operation::get_op::<MirAssignOp>(defining_op, ctx).is_some() {
            value = defining_op.deref(ctx).get_operand(0);
            continue;
        }
        if let Some(cast) = Operation::get_op::<MirCastOp>(defining_op, ctx) {
            if !cast.get_attr_cast_kind(ctx).is_some_and(|kind| {
                matches!(
                    *kind,
                    MirCastKindAttr::Transmute | MirCastKindAttr::PtrToPtr
                )
            }) {
                return Some(value);
            }
            value = defining_op.deref(ctx).get_operand(0);
            continue;
        }
        return Some(value);
    }
    None
}

fn unit_pointer_step_from(ctx: &Context, updated: Value, current: Value) -> bool {
    let Some(updated) = strip_iterator_identity_casts(ctx, updated) else {
        return false;
    };
    let Some(offset) = updated.defining_op() else {
        return false;
    };
    if Operation::get_op::<MirPtrOffsetOp>(offset, ctx).is_none()
        || integer_constant_u64(ctx, offset.deref(ctx).get_operand(1)) != Some(1)
    {
        return false;
    }
    strip_iterator_identity_casts(ctx, offset.deref(ctx).get_operand(0)) == Some(current)
}

fn condition_polarity(ctx: &Context, condition: Value, comparison_result: Value) -> Option<bool> {
    if condition == comparison_result {
        return Some(true);
    }
    let defining_op = condition.defining_op()?;
    if Operation::get_op::<MirNotOp>(defining_op, ctx).is_some()
        && defining_op.deref(ctx).get_operand(0) == comparison_result
    {
        return Some(false);
    }
    None
}

fn find_pointer_guard(
    ctx: &Context,
    guard_block: Ptr<BasicBlock>,
    current: Value,
    end: Value,
    update_block: Ptr<BasicBlock>,
) -> Option<(Ptr<Operation>, bool)> {
    let mut match_result = None;
    for operation in guard_block.deref(ctx).iter(ctx) {
        let is_eq = Operation::get_op::<MirEqOp>(operation, ctx).is_some();
        let is_ne = Operation::get_op::<MirNeOp>(operation, ctx).is_some();
        if !is_eq && !is_ne {
            continue;
        }
        let lhs = strip_iterator_identity_casts(ctx, operation.deref(ctx).get_operand(0));
        let rhs = strip_iterator_identity_casts(ctx, operation.deref(ctx).get_operand(1));
        if !((lhs == Some(current) && rhs == Some(end))
            || (lhs == Some(end) && rhs == Some(current)))
        {
            continue;
        }
        if match_result.replace((operation, is_eq)).is_some() {
            return None;
        }
    }

    let (comparison, is_eq) = match_result?;
    let terminator = guard_block.deref(ctx).get_terminator(ctx)?;
    Operation::get_op::<MirCondBranchOp>(terminator, ctx)?;
    let comparison_result = comparison.deref(ctx).get_result(0);
    let polarity =
        condition_polarity(ctx, terminator.deref(ctx).get_operand(0), comparison_result)?;

    let compare_true_is_continue = !is_eq;
    let condition_true_is_continue = if polarity {
        compare_true_is_continue
    } else {
        !compare_true_is_continue
    };
    let continue_index = if condition_true_is_continue { 0 } else { 1 };
    if terminator.deref(ctx).get_successor(continue_index) != update_block {
        return None;
    }

    Some((comparison, is_eq))
}

fn find_nonzero_count_guard(
    ctx: &Context,
    count_loads: &[Ptr<Operation>],
    decrement_load: Ptr<Operation>,
    decrement_block: Ptr<BasicBlock>,
) -> bool {
    for load in count_loads {
        if *load == decrement_load {
            continue;
        }
        let Some(block) = load.deref(ctx).get_parent_block() else {
            continue;
        };
        let count = load.deref(ctx).get_result(0);

        for operation in block.deref(ctx).iter(ctx) {
            let is_eq = Operation::get_op::<MirEqOp>(operation, ctx).is_some();
            let is_ne = Operation::get_op::<MirNeOp>(operation, ctx).is_some();
            if !is_eq && !is_ne {
                continue;
            }
            let lhs = operation.deref(ctx).get_operand(0);
            let rhs = operation.deref(ctx).get_operand(1);
            let compares_zero = (lhs == count && integer_constant_u64(ctx, rhs) == Some(0))
                || (rhs == count && integer_constant_u64(ctx, lhs) == Some(0));
            if !compares_zero {
                continue;
            }

            let Some(terminator) = block.deref(ctx).get_terminator(ctx) else {
                continue;
            };
            if Operation::get_op::<MirCondBranchOp>(terminator, ctx).is_none() {
                continue;
            }
            let comparison_result = operation.deref(ctx).get_result(0);
            let Some(polarity) =
                condition_polarity(ctx, terminator.deref(ctx).get_operand(0), comparison_result)
            else {
                continue;
            };

            let compare_true_is_nonzero = is_ne;
            let condition_true_is_nonzero = if polarity {
                compare_true_is_nonzero
            } else {
                !compare_true_is_nonzero
            };
            let nonzero_index = if condition_true_is_nonzero { 0 } else { 1 };
            if terminator.deref(ctx).get_successor(nonzero_index) == decrement_block {
                return true;
            }
        }
    }

    false
}

fn collect_enum_constructs_from_current(
    ctx: &Context,
    value: Value,
    update_block: Ptr<BasicBlock>,
    depth: usize,
    output: &mut Vec<Ptr<Operation>>,
) {
    if depth > 12 {
        return;
    }

    for value_use in value.uses(ctx) {
        if value_use.find_index(ctx) != 0 {
            continue;
        }
        let user = value_use.user_op();
        if user.deref(ctx).get_parent_block() != Some(update_block) {
            continue;
        }

        if let Some(cast) = Operation::get_op::<MirCastOp>(user, ctx)
            && cast.get_attr_cast_kind(ctx).is_some_and(|kind| {
                matches!(
                    *kind,
                    MirCastKindAttr::Transmute | MirCastKindAttr::PtrToPtr
                )
            })
        {
            collect_enum_constructs_from_current(
                ctx,
                user.deref(ctx).get_result(0),
                update_block,
                depth + 1,
                output,
            );
            continue;
        }

        if Operation::get_op::<MirAssignOp>(user, ctx).is_some() {
            collect_enum_constructs_from_current(
                ctx,
                user.deref(ctx).get_result(0),
                update_block,
                depth + 1,
                output,
            );
            continue;
        }

        if Operation::get_op::<MirConstructEnumOp>(user, ctx).is_some()
            && user.deref(ctx).get_num_operands() == 1
        {
            output.push(user);
        }
    }
}

fn forwarded_block_arguments(ctx: &Context, value: Value) -> Vec<Value> {
    let mut arguments = Vec::new();

    for value_use in value.uses(ctx) {
        let user = value_use.user_op();
        let operation = Operation::get_op_dyn(user, ctx);
        let Some(branch) = op_cast::<dyn BranchOpInterface>(operation.as_ref()) else {
            continue;
        };
        let successors: Vec<_> = user.deref(ctx).successors().collect();
        for (successor_index, successor) in successors.into_iter().enumerate() {
            let operands = branch.successor_operands(ctx, successor_index);
            for (argument_index, operand) in operands.iter().enumerate() {
                if *operand == value && argument_index < successor.deref(ctx).get_num_arguments() {
                    arguments.push(successor.deref(ctx).get_argument(argument_index));
                }
            }
        }
    }

    arguments
}

fn find_element_load_from_current_pointer(
    ctx: &Context,
    current: Value,
    update_block: Ptr<BasicBlock>,
    element_type: TypeHandle,
) -> Option<Ptr<Operation>> {
    let mut constructors = Vec::new();
    collect_enum_constructs_from_current(ctx, current, update_block, 0, &mut constructors);

    let mut found = None;
    for constructor in constructors {
        let construct = Operation::get_op::<MirConstructEnumOp>(constructor, ctx)?;
        let variant = construct.get_attr_construct_enum_variant_index(ctx)?.0;
        let enum_value = constructor.deref(ctx).get_result(0);

        for argument in forwarded_block_arguments(ctx, enum_value) {
            for argument_use in argument.uses(ctx) {
                if argument_use.find_index(ctx) != 0 {
                    continue;
                }
                let payload_op = argument_use.user_op();
                let Some(payload) = Operation::get_op::<MirEnumPayloadOp>(payload_op, ctx) else {
                    continue;
                };
                if payload.get_attr_payload_variant_index(ctx)?.0 != variant
                    || payload.get_attr_payload_field_index(ctx)?.0 != 0
                {
                    continue;
                }

                let payload_value = payload_op.deref(ctx).get_result(0);
                if payload_value.num_uses(ctx) != 1 {
                    continue;
                }
                let payload_use = payload_value.uses(ctx).into_iter().next()?;
                if payload_use.find_index(ctx) != 0 {
                    continue;
                }
                let load_op = payload_use.user_op();
                let Some(load) = Operation::get_op::<MirLoadOp>(load_op, ctx) else {
                    continue;
                };
                if load.is_volatile(ctx)
                    || load_op.deref(ctx).get_result(0).get_type(ctx) != element_type
                {
                    continue;
                }
                if found.replace(load_op).is_some() {
                    return None;
                }
            }
        }
    }

    found
}

fn analyze_memory_resident_small_array_iterator(
    ctx: &Context,
    function: Ptr<Operation>,
    iterator_alloca: Ptr<Operation>,
) -> Option<MemoryResidentIteratorPlan> {
    let iterator = Operation::get_op::<MirAllocaOp>(iterator_alloca, ctx)?;
    let iterator_type = iterator.pointee_type(ctx);
    let index_type = memory_resident_iterator_layout(ctx, iterator_type)?;

    let (iterator_initializer_store, initializer_value) =
        direct_alloca_initializer_store(ctx, iterator_alloca)?;
    let initializer_op = initializer_value.defining_op()?;
    Operation::get_op::<MirConstructStructOp>(initializer_op, ctx)?;

    let initial_current = project_constructed_value(ctx, initializer_value, &[0, 0, 0])?;
    let initial_end = project_constructed_value(ctx, initializer_value, &[0, 0, 1])?;
    let initial_count = project_constructed_value(ctx, initializer_value, &[1])?;
    if initial_count.get_type(ctx) != index_type {
        return None;
    }

    let current_origin = trace_array_address_origin(ctx, initial_current, 0)?;
    let end_origin = trace_array_address_origin(ctx, initial_end, 0)?;
    if current_origin.alloca != end_origin.alloca
        || current_origin.offset != 0
        || end_origin.offset != current_origin.array_size
        || current_origin.element_type != end_origin.element_type
    {
        return None;
    }

    let array = collect_elementwise_array_initialization(ctx, function, current_origin.alloca)?;
    if array.array_size != current_origin.array_size
        || array.element_type != current_origin.element_type
        || !initialization_dominates_iterator_store(ctx, &array, iterator_initializer_store)
    {
        return None;
    }

    let mut operations = Vec::new();
    collect_ops(ctx, function, &mut operations);

    let mut current_loads = Vec::new();
    let mut end_loads = Vec::new();
    let mut count_loads = Vec::new();
    let mut current_stores = Vec::new();
    let mut count_stores = Vec::new();

    for operation in operations {
        if let Some(load) = Operation::get_op::<MirLoadOp>(operation, ctx) {
            let Some((root_alloca, path)) = field_addr_root_path(ctx, load.address_opd(ctx)) else {
                continue;
            };
            if root_alloca != iterator_alloca {
                continue;
            }
            match path.as_slice() {
                [0, 0, 0] => current_loads.push(operation),
                [0, 0, 1] => end_loads.push(operation),
                [1] => count_loads.push(operation),
                _ => {}
            }
            continue;
        }

        if let Some(store) = Operation::get_op::<MirStoreOp>(operation, ctx) {
            if operation == iterator_initializer_store {
                continue;
            }
            let Some((root_alloca, path)) = field_addr_root_path(ctx, store.address_opd(ctx))
            else {
                continue;
            };
            if root_alloca != iterator_alloca {
                continue;
            }
            match path.as_slice() {
                [0, 0, 0] => current_stores.push(operation),
                [1] => count_stores.push(operation),
                _ => return None,
            }
        }
    }

    let [current_load] = current_loads.as_slice() else {
        return None;
    };
    let [end_load] = end_loads.as_slice() else {
        return None;
    };
    let [current_store] = current_stores.as_slice() else {
        return None;
    };
    let [count_store] = count_stores.as_slice() else {
        return None;
    };

    let guard_block = current_load.deref(ctx).get_parent_block()?;
    if end_load.deref(ctx).get_parent_block() != Some(guard_block)
        || current_store.deref(ctx).get_parent_block().is_none()
        || count_store.deref(ctx).get_parent_block() != Some(guard_block)
    {
        return None;
    }
    let update_block = current_store.deref(ctx).get_parent_block()?;

    let current = current_load.deref(ctx).get_result(0);
    let end = end_load.deref(ctx).get_result(0);
    if !unit_pointer_step_from(
        ctx,
        Operation::get_op::<MirStoreOp>(*current_store, ctx)?.value_opd(ctx),
        current,
    ) {
        return None;
    }

    let decrement_loads: Vec<_> = count_loads
        .iter()
        .copied()
        .filter(|load| load.deref(ctx).get_parent_block() == Some(guard_block))
        .collect();
    let [decrement_load] = decrement_loads.as_slice() else {
        return None;
    };
    let current_count = decrement_load.deref(ctx).get_result(0);
    if current_count.get_type(ctx) != index_type {
        return None;
    }

    let count_store_value = Operation::get_op::<MirStoreOp>(*count_store, ctx)?.value_opd(ctx);
    let count_sub = count_store_value.defining_op()?;
    if Operation::get_op::<MirSubOp>(count_sub, ctx).is_none()
        || count_sub.deref(ctx).get_operand(0) != current_count
        || integer_constant_u64(ctx, count_sub.deref(ctx).get_operand(1)) != Some(1)
    {
        return None;
    }

    if !find_nonzero_count_guard(ctx, &count_loads, *decrement_load, guard_block) {
        return None;
    }

    let (guard_compare, guard_is_eq) =
        find_pointer_guard(ctx, guard_block, current, end, update_block)?;

    let element_load =
        find_element_load_from_current_pointer(ctx, current, update_block, array.element_type)?;

    Some(MemoryResidentIteratorPlan {
        iterator_initializer_store,
        array,
        initial_count,
        current_count,
        guard_compare,
        guard_is_eq,
        element_load,
        index_type,
    })
}

fn rewrite_memory_resident_small_array_iterator(
    ctx: &mut Context,
    plan: MemoryResidentIteratorPlan,
) {
    let array_construct = Operation::new(
        ctx,
        MirConstructArrayOp::get_concrete_op_info(),
        vec![plan.array.array_type],
        plan.array.values.clone(),
        vec![],
        0,
    );
    let initializer_location = plan.iterator_initializer_store.deref(ctx).loc().clone();
    array_construct.deref_mut(ctx).set_loc(initializer_location);
    array_construct.insert_before(ctx, plan.iterator_initializer_store);
    let array_value = array_construct.deref(ctx).get_result(0);

    let iteration_index = Operation::new(
        ctx,
        MirSubOp::get_concrete_op_info(),
        vec![plan.index_type],
        vec![plan.initial_count, plan.current_count],
        vec![],
        0,
    );
    let guard_location = plan.guard_compare.deref(ctx).loc().clone();
    iteration_index
        .deref_mut(ctx)
        .set_loc(guard_location.clone());
    iteration_index.insert_before(ctx, plan.guard_compare);
    let index = iteration_index.deref(ctx).get_result(0);

    let bound = insert_integer_constant_before(
        ctx,
        plan.guard_compare,
        plan.index_type,
        plan.array.array_size,
    );
    let guard_result_type = plan.guard_compare.deref(ctx).get_result(0).get_type(ctx);
    let integer_guard = Operation::new(
        ctx,
        if plan.guard_is_eq {
            MirEqOp::get_concrete_op_info()
        } else {
            MirNeOp::get_concrete_op_info()
        },
        vec![guard_result_type],
        vec![index, bound],
        vec![],
        0,
    );
    integer_guard.deref_mut(ctx).set_loc(guard_location);
    integer_guard.insert_before(ctx, plan.guard_compare);
    let integer_guard_result = integer_guard.deref(ctx).get_result(0);
    let old_guard_result = plan.guard_compare.deref(ctx).get_result(0);
    old_guard_result.replace_all_uses_with(ctx, &integer_guard_result);

    let load_location = plan.element_load.deref(ctx).loc().clone();
    let bounded_index = Operation::new(
        ctx,
        MirRemOp::get_concrete_op_info(),
        vec![plan.index_type],
        vec![index, bound],
        vec![],
        0,
    );
    bounded_index.deref_mut(ctx).set_loc(load_location.clone());
    bounded_index.insert_before(ctx, plan.element_load);
    let bounded_index_value = bounded_index.deref(ctx).get_result(0);

    let extraction = Operation::new(
        ctx,
        MirExtractArrayElementOp::get_concrete_op_info(),
        vec![plan.array.element_type],
        vec![array_value, bounded_index_value],
        vec![],
        0,
    );
    extraction.deref_mut(ctx).set_loc(load_location);
    extraction.insert_before(ctx, plan.element_load);
    let replacement = extraction.deref(ctx).get_result(0);
    let old_load_result = plan.element_load.deref(ctx).get_result(0);
    old_load_result.replace_all_uses_with(ctx, &replacement);

    let mut rewriter = IRRewriter::<Recorder>::default();
    rewriter.erase_operation(ctx, plan.element_load);
    if plan.guard_compare.deref(ctx).get_result(0).num_uses(ctx) == 0 {
        rewriter.erase_operation(ctx, plan.guard_compare);
    }
    for store in plan.array.stores {
        rewriter.erase_operation(ctx, store);
    }
    for address in plan.array.element_addrs {
        if address.deref(ctx).get_result(0).num_uses(ctx) == 0 {
            rewriter.erase_operation(ctx, address);
        }
    }
}

fn project_constructed_value(ctx: &Context, value: Value, path: &[u32]) -> Option<Value> {
    if path.is_empty() {
        return Some(value);
    }

    let defining_op = value.defining_op()?;
    let field = path[0] as usize;
    let rest = &path[1..];

    if Operation::get_op::<MirConstructStructOp>(defining_op, ctx).is_some()
        || Operation::get_op::<MirConstructTupleOp>(defining_op, ctx).is_some()
        || Operation::get_op::<MirConstructSliceOp>(defining_op, ctx).is_some()
    {
        if field >= defining_op.deref(ctx).get_num_operands() {
            return None;
        }
        return project_constructed_value(ctx, defining_op.deref(ctx).get_operand(field), rest);
    }

    if let Some(insert) = Operation::get_op::<MirInsertFieldOp>(defining_op, ctx) {
        let inserted_field = insert.get_attr_insert_index(ctx)?.0 as usize;
        if inserted_field == field {
            return project_constructed_value(ctx, defining_op.deref(ctx).get_operand(1), rest);
        }
        return project_constructed_value(ctx, defining_op.deref(ctx).get_operand(0), path);
    }

    if Operation::get_op::<MirAssignOp>(defining_op, ctx).is_some() {
        return project_constructed_value(ctx, defining_op.deref(ctx).get_operand(0), path);
    }

    None
}

/// Scalarize the memory-resident pointer walk emitted by rustc for a bounded
/// iterator over one small local array after mem2reg.
///
/// The accepted shape is the typed MIR representation of
/// `array.iter().copied().take(n)`: the adapter state remains in a local slot,
/// with nested fields for the current pointer, one-past-end pointer, and the
/// remaining `Take::n` count. The recognizer proves local-array provenance,
/// exact elementwise initialization, `base + N` as the end pointer, a unit
/// pointer step, and the zero/decrement protocol for `Take::n`.
///
/// For each successful element, `initial_take_n - current_take_n` is the exact
/// zero-based array index. Rewriting the pointer guard to an equivalent integer
/// guard and the payload load to `mir.extract_array_element(array, index % N)`
/// exposes the bounded SSA form already scalarized by typed MIR lowering.
/// Unsupported layouts, pointer arithmetic, writes, escapes, or guards fail
/// closed.
pub fn canonicalize_small_local_array_pointer_walks(
    module: Ptr<Operation>,
    ctx: &mut Context,
    verbose: bool,
) {
    let rewritten = canonicalize_memory_resident_small_array_iterators(module, ctx);

    if rewritten > 0 && verbose {
        eprintln!("small-array pointer-walk scalarization: rewrote {rewritten} loop(s)");
    }
}

fn insert_integer_constant_before(
    ctx: &mut Context,
    before: Ptr<Operation>,
    integer_type: TypeHandle,
    value: u64,
) -> Value {
    let typed = TypedHandle::<IntegerType>::from_handle(integer_type, ctx)
        .expect("validated integer induction type");
    let width = typed.deref(ctx).width() as usize;
    let attribute = IntegerAttr::new(
        typed,
        APInt::from_u64(
            value,
            NonZeroUsize::new(width).expect("integer width is non-zero"),
        ),
    );
    let constant = Operation::new(
        ctx,
        MirConstantOp::get_concrete_op_info(),
        vec![integer_type],
        vec![],
        vec![],
        0,
    );
    MirConstantOp::new(constant).set_attr_value(ctx, attribute);
    let location = before.deref(ctx).loc().clone();
    constant.deref_mut(ctx).set_loc(location);
    constant.insert_before(ctx, before);
    constant.deref(ctx).get_result(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::{
        attributes::VariantIndexAttr,
        ops::{MirGotoOp, MirReturnOp},
        types::{EnumVariant, MirArrayType, MirEnumType},
    };
    use pliron::builtin::{
        attributes::TypeAttr,
        op_interfaces::{OperandSegmentInterface, SingleBlockRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::FunctionType,
    };
    use pliron::region::Region;

    struct Fixture {
        module: Ptr<Operation>,
        alloca: Ptr<Operation>,
    }

    fn build_fixture(
        ctx: &mut Context,
        array_size: u64,
        divisor: Option<u64>,
        additional_store: bool,
        volatile_load: bool,
    ) -> Fixture {
        dialect_mir::register(ctx);

        let element_type: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let index_type = IntegerType::get(ctx, 64, Signedness::Unsigned);
        let index_handle: TypeHandle = index_type.into();
        let array_type: TypeHandle = MirArrayType::get(ctx, element_type, array_size).into();
        let aggregate_type: TypeHandle = MirStructType::get_with_full_layout(
            ctx,
            "BorrowedAggregate".into(),
            vec!["values".into()],
            vec![array_type],
            vec![0],
            vec![0],
            array_size * 4,
            4,
        )
        .into();

        let module = ModuleOp::new(ctx, "test".try_into().unwrap());
        let function_type = FunctionType::get(ctx, vec![aggregate_type, index_handle], vec![]);
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
        let entry = BasicBlock::new(ctx, None, vec![aggregate_type, index_handle]);
        entry.insert_at_back(region, ctx);
        let body = BasicBlock::new(ctx, None, vec![]);
        body.insert_at_back(region, ctx);

        let aggregate_argument = entry.deref(ctx).get_argument(0);
        let raw_index = entry.deref(ctx).get_argument(1);

        let aggregate_pointer: TypeHandle =
            MirPtrType::get_generic(ctx, aggregate_type, true).into();
        let alloca = Operation::new(
            ctx,
            MirAllocaOp::get_concrete_op_info(),
            vec![aggregate_pointer],
            vec![],
            vec![],
            0,
        );
        alloca.insert_at_back(entry, ctx);
        let slot = alloca.deref(ctx).get_result(0);

        let store = Operation::new(
            ctx,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![slot, aggregate_argument],
            vec![],
            0,
        );
        store.insert_at_back(entry, ctx);

        if additional_store {
            let second_store = Operation::new(
                ctx,
                MirStoreOp::get_concrete_op_info(),
                vec![],
                vec![slot, aggregate_argument],
                vec![],
                0,
            );
            second_store.insert_at_back(entry, ctx);
        }

        let goto = Operation::new(
            ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![body],
            0,
        );
        goto.insert_at_back(entry, ctx);

        let index = if let Some(divisor) = divisor {
            let divisor_attribute = IntegerAttr::new(
                index_type,
                APInt::from_u64(divisor, NonZeroUsize::new(64).unwrap()),
            );
            let constant = Operation::new(
                ctx,
                MirConstantOp::get_concrete_op_info(),
                vec![index_handle],
                vec![],
                vec![],
                0,
            );
            MirConstantOp::new(constant).set_attr_value(ctx, divisor_attribute);
            constant.insert_at_back(body, ctx);
            let divisor_value = constant.deref(ctx).get_result(0);

            let rem = Operation::new(
                ctx,
                MirRemOp::get_concrete_op_info(),
                vec![index_handle],
                vec![raw_index, divisor_value],
                vec![],
                0,
            );
            rem.insert_at_back(body, ctx);
            rem.deref(ctx).get_result(0)
        } else {
            raw_index
        };

        let field_pointer: TypeHandle = MirPtrType::get_generic(ctx, array_type, false).into();
        let field = Operation::new(
            ctx,
            MirFieldAddrOp::get_concrete_op_info(),
            vec![field_pointer],
            vec![slot],
            vec![],
            0,
        );
        MirFieldAddrOp::new(field).set_attr_field_index(ctx, FieldIndexAttr(0));
        field.insert_at_back(body, ctx);
        let field_value = field.deref(ctx).get_result(0);

        let element_pointer: TypeHandle = MirPtrType::get_generic(ctx, element_type, false).into();
        let element_address = Operation::new(
            ctx,
            MirArrayElementAddrOp::get_concrete_op_info(),
            vec![element_pointer],
            vec![field_value, index],
            vec![],
            0,
        );
        element_address.insert_at_back(body, ctx);
        let element_pointer_value = element_address.deref(ctx).get_result(0);

        let load = Operation::new(
            ctx,
            MirLoadOp::get_concrete_op_info(),
            vec![element_type],
            vec![element_pointer_value],
            vec![],
            0,
        );
        if volatile_load {
            MirLoadOp::new(load).set_volatile(ctx, true);
        }
        load.insert_at_back(body, ctx);

        let return_op = Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        return_op.insert_at_back(body, ctx);

        Fixture {
            module: module.get_operation(),
            alloca,
        }
    }

    fn count<T: Op>(ctx: &Context, root: Ptr<Operation>) -> usize {
        let mut operations = Vec::new();
        collect_ops(ctx, root, &mut operations);
        operations
            .into_iter()
            .filter(|operation| Operation::get_op::<T>(*operation, ctx).is_some())
            .count()
    }

    #[test]
    fn bounded_rem_rewrites_large_array_with_small_candidate_set() {
        let mut ctx = Context::new();
        let fixture = build_fixture(&mut ctx, 64, Some(3), false, false);

        canonicalize_read_only_aggregate_arguments(fixture.module, &mut ctx, false);

        assert_eq!(count::<MirExtractFieldOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirFieldAddrOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirArrayElementAddrOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 0);
        assert!(
            Operation::get_op::<MirAllocaOp>(fixture.alloca, &ctx).is_some(),
            "mem2reg, not this pass, owns erasing the entry slot"
        );
    }

    #[test]
    fn unbounded_index_is_canonicalized_for_lowering_fallback() {
        let mut ctx = Context::new();
        let fixture = build_fixture(&mut ctx, 3, None, false, false);

        canonicalize_read_only_aggregate_arguments(fixture.module, &mut ctx, false);

        assert_eq!(count::<MirExtractFieldOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirFieldAddrOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirArrayElementAddrOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 0);
    }

    #[test]
    fn oversized_candidate_set_is_canonicalized_for_lowering_fallback() {
        let mut ctx = Context::new();
        let fixture = build_fixture(&mut ctx, 64, Some(17), false, false);

        canonicalize_read_only_aggregate_arguments(fixture.module, &mut ctx, false);

        assert_eq!(count::<MirExtractFieldOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirFieldAddrOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirArrayElementAddrOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 0);
    }

    #[test]
    fn additional_store_rejects_the_entire_slot() {
        let mut ctx = Context::new();
        let fixture = build_fixture(&mut ctx, 3, Some(3), true, false);

        canonicalize_read_only_aggregate_arguments(fixture.module, &mut ctx, false);

        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn volatile_load_rejects_the_entire_slot() {
        let mut ctx = Context::new();
        let fixture = build_fixture(&mut ctx, 3, Some(3), false, true);

        canonicalize_read_only_aggregate_arguments(fixture.module, &mut ctx, false);

        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
    }

    struct BorrowedPointerFixture {
        module: Ptr<Operation>,
    }

    /// Who calls the borrowed-pointer helper, and what backs the pointer.
    #[derive(Clone, Copy)]
    enum CallerShape {
        /// Every call site passes the address of a caller-local slot.
        OwnedSlot,
        /// The single call site forwards the caller's own pointer parameter.
        PointerParameter,
        /// One owned-slot call site plus one forwarded-parameter call site.
        Mixed,
        /// The helper has no call site in the module.
        None,
    }

    fn add_helper_call(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        helper_symbol: &str,
        pointer: Value,
        index: Value,
        element_type: TypeHandle,
    ) {
        let call = Operation::new(
            ctx,
            MirCallOp::get_concrete_op_info(),
            vec![element_type],
            vec![pointer, index],
            vec![],
            0,
        );
        MirCallOp::new(call).set_attr_callee(
            ctx,
            pliron::builtin::attributes::StringAttr::new(helper_symbol.to_string()),
        );
        call.insert_at_back(block, ctx);

        let return_op = Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        return_op.insert_at_back(block, ctx);
    }

    fn add_caller_function(
        ctx: &mut Context,
        module: &ModuleOp,
        name: &str,
        argument_types: Vec<TypeHandle>,
    ) -> Ptr<BasicBlock> {
        let function_type = FunctionType::get(ctx, argument_types.clone(), vec![]);
        let function = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function_op = MirFuncOp::new(ctx, function, TypeAttr::new(function_type.into()));
        function_op.set_symbol_name(ctx, name.try_into().unwrap());
        module.append_operation(ctx, function, 0);

        let region: Ptr<Region> = function.deref(ctx).get_region(0);
        let entry = BasicBlock::new(ctx, None, argument_types);
        entry.insert_at_back(region, ctx);
        entry
    }

    /// The type handles a fixture caller needs to call the helper.
    #[derive(Clone, Copy)]
    struct CallerTypes {
        aggregate_type: TypeHandle,
        aggregate_pointer: TypeHandle,
        index_handle: TypeHandle,
        element_type: TypeHandle,
    }

    /// A caller holding the aggregate by value in a local slot, calling the
    /// helper with a `&mut slot -> &slot` reborrow of that slot's address.
    fn add_owned_slot_caller(
        ctx: &mut Context,
        module: &ModuleOp,
        name: &str,
        helper_symbol: &str,
        types: CallerTypes,
    ) {
        let CallerTypes {
            aggregate_type,
            aggregate_pointer,
            index_handle,
            element_type,
        } = types;
        let entry = add_caller_function(ctx, module, name, vec![aggregate_type, index_handle]);
        let aggregate_argument = entry.deref(ctx).get_argument(0);
        let index = entry.deref(ctx).get_argument(1);

        let slot_pointer: TypeHandle = MirPtrType::get_generic(ctx, aggregate_type, true).into();
        let slot = Operation::new(
            ctx,
            MirAllocaOp::get_concrete_op_info(),
            vec![slot_pointer],
            vec![],
            vec![],
            0,
        );
        slot.insert_at_back(entry, ctx);
        let slot_value = slot.deref(ctx).get_result(0);

        let store = Operation::new(
            ctx,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![slot_value, aggregate_argument],
            vec![],
            0,
        );
        store.insert_at_back(entry, ctx);

        let reborrow = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![aggregate_pointer],
            vec![slot_value],
            vec![],
            0,
        );
        MirCastOp::new(reborrow).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
        reborrow.insert_at_back(entry, ctx);
        let reborrow_value = reborrow.deref(ctx).get_result(0);

        add_helper_call(
            ctx,
            entry,
            helper_symbol,
            reborrow_value,
            index,
            element_type,
        );
    }

    /// A caller forwarding its own aggregate pointer parameter, i.e. memory
    /// this module cannot prove to be caller-private.
    fn add_pointer_parameter_caller(
        ctx: &mut Context,
        module: &ModuleOp,
        name: &str,
        helper_symbol: &str,
        types: CallerTypes,
    ) {
        let entry = add_caller_function(
            ctx,
            module,
            name,
            vec![types.aggregate_pointer, types.index_handle],
        );
        let forwarded_pointer = entry.deref(ctx).get_argument(0);
        let index = entry.deref(ctx).get_argument(1);
        add_helper_call(
            ctx,
            entry,
            helper_symbol,
            forwarded_pointer,
            index,
            types.element_type,
        );
    }

    fn build_borrowed_pointer_fixture(
        ctx: &mut Context,
        asserted_bound: Option<u64>,
        alwaysinline: bool,
        volatile_load: bool,
        caller_shape: CallerShape,
        helper_symbol: &str,
    ) -> BorrowedPointerFixture {
        dialect_mir::register(ctx);

        let element_type: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let index_type = IntegerType::get(ctx, 64, Signedness::Unsigned);
        let index_handle: TypeHandle = index_type.into();
        let array_type: TypeHandle = MirArrayType::get(ctx, element_type, 3).into();
        let aggregate_type: TypeHandle = MirStructType::get_with_full_layout(
            ctx,
            "BorrowedAggregate".into(),
            vec!["values".into()],
            vec![array_type],
            vec![0],
            vec![0],
            12,
            4,
        )
        .into();
        let aggregate_pointer: TypeHandle =
            MirPtrType::get_generic(ctx, aggregate_type, false).into();

        let module = ModuleOp::new(ctx, "test".try_into().unwrap());
        let function_type = FunctionType::get(
            ctx,
            vec![aggregate_pointer, index_handle],
            vec![element_type],
        );
        let function = Operation::new(
            ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let function_op = MirFuncOp::new(ctx, function, TypeAttr::new(function_type.into()));
        function_op.set_symbol_name(ctx, helper_symbol.try_into().unwrap());
        if alwaysinline {
            function.deref_mut(ctx).attributes.set(
                "alwaysinline".try_into().unwrap(),
                pliron::builtin::attributes::StringAttr::new("true".to_string()),
            );
        }
        module.append_operation(ctx, function, 0);

        let region: Ptr<Region> = function.deref(ctx).get_region(0);
        let entry = BasicBlock::new(ctx, None, vec![aggregate_pointer, index_handle]);
        entry.insert_at_back(region, ctx);
        let body = BasicBlock::new(ctx, None, vec![]);
        body.insert_at_back(region, ctx);

        let aggregate_argument = entry.deref(ctx).get_argument(0);
        let index = entry.deref(ctx).get_argument(1);

        if let Some(bound) = asserted_bound {
            let bound_attribute = IntegerAttr::new(
                index_type,
                APInt::from_u64(bound, NonZeroUsize::new(64).unwrap()),
            );
            let constant = Operation::new(
                ctx,
                MirConstantOp::get_concrete_op_info(),
                vec![index_handle],
                vec![],
                vec![],
                0,
            );
            MirConstantOp::new(constant).set_attr_value(ctx, bound_attribute);
            constant.insert_at_back(entry, ctx);
            let bound_value = constant.deref(ctx).get_result(0);

            let i1_type: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
            let comparison = Operation::new(
                ctx,
                MirLtOp::get_concrete_op_info(),
                vec![i1_type],
                vec![index, bound_value],
                vec![],
                0,
            );
            comparison.insert_at_back(entry, ctx);
            let condition = comparison.deref(ctx).get_result(0);

            let assert = Operation::new(
                ctx,
                MirAssertOp::get_concrete_op_info(),
                vec![],
                vec![condition],
                vec![body],
                0,
            );
            assert.insert_at_back(entry, ctx);
        } else {
            let goto = Operation::new(
                ctx,
                MirGotoOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![body],
                0,
            );
            goto.insert_at_back(entry, ctx);
        }

        let field_pointer: TypeHandle = MirPtrType::get_generic(ctx, array_type, false).into();
        let field = Operation::new(
            ctx,
            MirFieldAddrOp::get_concrete_op_info(),
            vec![field_pointer],
            vec![aggregate_argument],
            vec![],
            0,
        );
        MirFieldAddrOp::new(field).set_attr_field_index(ctx, FieldIndexAttr(0));
        field.insert_at_back(body, ctx);
        let field_value = field.deref(ctx).get_result(0);

        let element_pointer: TypeHandle = MirPtrType::get_generic(ctx, element_type, false).into();
        let element_address = Operation::new(
            ctx,
            MirArrayElementAddrOp::get_concrete_op_info(),
            vec![element_pointer],
            vec![field_value, index],
            vec![],
            0,
        );
        element_address.insert_at_back(body, ctx);
        let element_pointer_value = element_address.deref(ctx).get_result(0);

        let load = Operation::new(
            ctx,
            MirLoadOp::get_concrete_op_info(),
            vec![element_type],
            vec![element_pointer_value],
            vec![],
            0,
        );
        if volatile_load {
            MirLoadOp::new(load).set_volatile(ctx, true);
        }
        load.insert_at_back(body, ctx);
        let result = load.deref(ctx).get_result(0);

        let return_op = Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![result],
            vec![],
            0,
        );
        return_op.insert_at_back(body, ctx);

        let caller_types = CallerTypes {
            aggregate_type,
            aggregate_pointer,
            index_handle,
            element_type,
        };
        match caller_shape {
            CallerShape::OwnedSlot => {
                add_owned_slot_caller(ctx, &module, "caller_owned", helper_symbol, caller_types);
            }
            CallerShape::PointerParameter => {
                add_pointer_parameter_caller(
                    ctx,
                    &module,
                    "caller_external",
                    helper_symbol,
                    caller_types,
                );
            }
            CallerShape::Mixed => {
                add_owned_slot_caller(ctx, &module, "caller_owned", helper_symbol, caller_types);
                add_pointer_parameter_caller(
                    ctx,
                    &module,
                    "caller_external",
                    helper_symbol,
                    caller_types,
                );
            }
            CallerShape::None => {}
        }

        BorrowedPointerFixture {
            module: module.get_operation(),
        }
    }

    #[test]
    fn asserted_immutable_pointer_read_is_canonicalized_after_mem2reg() {
        let mut ctx = Context::new();
        let fixture = build_borrowed_pointer_fixture(
            &mut ctx,
            Some(3),
            true,
            false,
            CallerShape::OwnedSlot,
            "helper",
        );

        canonicalize_bounded_borrowed_pointer_arguments(fixture.module, &mut ctx, false);

        assert_eq!(count::<MirFieldAddrOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirArrayElementAddrOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirExtractFieldOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirRemOp>(&ctx, fixture.module), 1);
        assert_eq!(
            count::<MirLoadOp>(&ctx, fixture.module),
            1,
            "only the bounded array-field load introduced at the original access point remains"
        );
    }

    #[test]
    fn pointer_read_without_exact_assert_is_left_unchanged() {
        let mut ctx = Context::new();
        let fixture = build_borrowed_pointer_fixture(
            &mut ctx,
            None,
            true,
            false,
            CallerShape::OwnedSlot,
            "helper",
        );

        canonicalize_bounded_borrowed_pointer_arguments(fixture.module, &mut ctx, false);

        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirFieldAddrOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirArrayElementAddrOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn non_alwaysinline_pointer_helper_is_left_unchanged() {
        let mut ctx = Context::new();
        let fixture = build_borrowed_pointer_fixture(
            &mut ctx,
            Some(3),
            false,
            false,
            CallerShape::OwnedSlot,
            "helper",
        );

        canonicalize_bounded_borrowed_pointer_arguments(fixture.module, &mut ctx, false);

        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn volatile_pointer_read_is_left_unchanged() {
        let mut ctx = Context::new();
        let fixture = build_borrowed_pointer_fixture(
            &mut ctx,
            Some(3),
            true,
            true,
            CallerShape::OwnedSlot,
            "helper",
        );

        canonicalize_bounded_borrowed_pointer_arguments(fixture.module, &mut ctx, false);

        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
    }

    /// Asserts the helper kept its single dynamic memory access: the pointer
    /// chain survives, no value-level extraction or widened array-field load
    /// was introduced, and no bounded remainder was materialized.
    fn assert_single_dynamic_load_survives(ctx: &Context, module: Ptr<Operation>) {
        assert_eq!(count::<MirExtractArrayElementOp>(ctx, module), 0);
        assert_eq!(count::<MirExtractFieldOp>(ctx, module), 0);
        assert_eq!(count::<MirFieldAddrOp>(ctx, module), 1);
        assert_eq!(count::<MirArrayElementAddrOp>(ctx, module), 1);
        assert_eq!(count::<MirRemOp>(ctx, module), 0);
        assert_eq!(
            count::<MirLoadOp>(ctx, module),
            1,
            "the original dynamic element load must survive unwidened"
        );
    }

    #[test]
    fn pointer_parameter_call_site_is_left_unchanged() {
        let mut ctx = Context::new();
        let fixture = build_borrowed_pointer_fixture(
            &mut ctx,
            Some(3),
            true,
            false,
            CallerShape::PointerParameter,
            "helper",
        );

        canonicalize_bounded_borrowed_pointer_arguments(fixture.module, &mut ctx, false);

        assert_single_dynamic_load_survives(&ctx, fixture.module);
    }

    #[test]
    fn mixed_call_sites_are_left_unchanged() {
        let mut ctx = Context::new();
        let fixture = build_borrowed_pointer_fixture(
            &mut ctx,
            Some(3),
            true,
            false,
            CallerShape::Mixed,
            "helper",
        );

        canonicalize_bounded_borrowed_pointer_arguments(fixture.module, &mut ctx, false);

        assert_single_dynamic_load_survives(&ctx, fixture.module);
    }

    #[test]
    fn helper_without_visible_call_site_is_left_unchanged() {
        let mut ctx = Context::new();
        let fixture = build_borrowed_pointer_fixture(
            &mut ctx,
            Some(3),
            true,
            false,
            CallerShape::None,
            "helper",
        );

        canonicalize_bounded_borrowed_pointer_arguments(fixture.module, &mut ctx, false);

        assert_single_dynamic_load_survives(&ctx, fixture.module);
    }

    #[test]
    fn device_export_helper_is_left_unchanged() {
        // An exported `#[device]` function is externally callable, so the
        // module-level call scan cannot see every call site. Even an owned
        // in-module call site must not enable the rewrite.
        let mut ctx = Context::new();
        let exported_symbol = reserved_oxide_symbols::device_symbol("helper");
        let fixture = build_borrowed_pointer_fixture(
            &mut ctx,
            Some(3),
            true,
            false,
            CallerShape::OwnedSlot,
            &exported_symbol,
        );

        canonicalize_bounded_borrowed_pointer_arguments(fixture.module, &mut ctx, false);

        assert_single_dynamic_load_survives(&ctx, fixture.module);
    }
    struct MemoryResidentIteratorFixture {
        module: Ptr<Operation>,
    }

    fn append_u64_constant(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        ty: TypedHandle<IntegerType>,
        value: u64,
    ) -> Value {
        let width = ty.deref(ctx).width() as usize;
        let constant = Operation::new(
            ctx,
            MirConstantOp::get_concrete_op_info(),
            vec![ty.into()],
            vec![],
            vec![],
            0,
        );
        MirConstantOp::new(constant).set_attr_value(
            ctx,
            IntegerAttr::new(
                ty,
                APInt::from_u64(value, NonZeroUsize::new(width).unwrap()),
            ),
        );
        constant.insert_at_back(block, ctx);
        constant.deref(ctx).get_result(0)
    }

    fn append_cond_branch(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        condition: Value,
        true_successor: Ptr<BasicBlock>,
        false_successor: Ptr<BasicBlock>,
    ) {
        let (operands, segments) =
            MirCondBranchOp::compute_segment_sizes(vec![vec![condition], Vec::new(), Vec::new()]);
        let branch = Operation::new(
            ctx,
            MirCondBranchOp::get_concrete_op_info(),
            vec![],
            operands,
            vec![true_successor, false_successor],
            0,
        );
        Operation::get_op::<MirCondBranchOp>(branch, ctx)
            .unwrap()
            .set_operand_segment_sizes(ctx, segments);
        branch.insert_at_back(block, ctx);
    }

    fn append_struct(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        result_type: TypeHandle,
        fields: Vec<Value>,
    ) -> Value {
        let operation = Operation::new(
            ctx,
            MirConstructStructOp::get_concrete_op_info(),
            vec![result_type],
            fields,
            vec![],
            0,
        );
        operation.insert_at_back(block, ctx);
        operation.deref(ctx).get_result(0)
    }

    fn append_field_addr_path(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        root: Value,
        root_type: TypeHandle,
        path: &[u32],
    ) -> (Value, TypeHandle) {
        let mut address = root;
        let mut ty = root_type;
        for &field_index in path {
            let field_type = struct_field_type(ctx, ty, field_index as usize).unwrap();
            let field_pointer: TypeHandle = MirPtrType::get_generic(ctx, field_type, true).into();
            let field_addr = Operation::new(
                ctx,
                MirFieldAddrOp::get_concrete_op_info(),
                vec![field_pointer],
                vec![address],
                vec![],
                0,
            );
            MirFieldAddrOp::new(field_addr).set_attr_field_index(ctx, FieldIndexAttr(field_index));
            field_addr.insert_at_back(block, ctx);
            address = field_addr.deref(ctx).get_result(0);
            ty = field_type;
        }
        (address, ty)
    }

    fn append_load_from_path(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        root: Value,
        root_type: TypeHandle,
        path: &[u32],
    ) -> Value {
        let (address, field_type) = append_field_addr_path(ctx, block, root, root_type, path);
        let load = Operation::new(
            ctx,
            MirLoadOp::get_concrete_op_info(),
            vec![field_type],
            vec![address],
            vec![],
            0,
        );
        load.insert_at_back(block, ctx);
        load.deref(ctx).get_result(0)
    }

    fn build_memory_resident_iterator_fixture(
        ctx: &mut Context,
        end_offset: u64,
        step: u64,
    ) -> MemoryResidentIteratorFixture {
        dialect_mir::register(ctx);

        let element_type = IntegerType::get(ctx, 32, Signedness::Unsigned);
        let element_handle: TypeHandle = element_type.into();
        let index_type = IntegerType::get(ctx, 64, Signedness::Unsigned);
        let index_handle: TypeHandle = index_type.into();
        let i1_type: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();
        let tag_type = IntegerType::get(ctx, 8, Signedness::Unsigned);

        let array_type: TypeHandle = MirArrayType::get(ctx, element_handle, 4).into();
        let array_pointer: TypeHandle = MirPtrType::get_generic(ctx, array_type, true).into();
        let element_pointer: TypeHandle = MirPtrType::get_generic(ctx, element_handle, true).into();

        let current_wrapper: TypeHandle = MirStructType::get_with_full_layout(
            ctx,
            "CurrentPointer".into(),
            vec!["ptr".into()],
            vec![element_pointer],
            vec![0],
            vec![0],
            8,
            8,
        )
        .into();
        let iterator_type: TypeHandle = MirStructType::get_with_full_layout(
            ctx,
            "SliceIter".into(),
            vec!["current".into(), "end".into()],
            vec![current_wrapper, element_pointer],
            vec![0, 1],
            vec![0, 8],
            16,
            8,
        )
        .into();
        let copied_type: TypeHandle = MirStructType::get_with_full_layout(
            ctx,
            "Copied".into(),
            vec!["iter".into()],
            vec![iterator_type],
            vec![0],
            vec![0],
            16,
            8,
        )
        .into();
        let take_type: TypeHandle = MirStructType::get_with_full_layout(
            ctx,
            "Take".into(),
            vec!["iter".into(), "n".into()],
            vec![copied_type, index_handle],
            vec![0, 1],
            vec![0, 16],
            24,
            8,
        )
        .into();
        let take_pointer: TypeHandle = MirPtrType::get_generic(ctx, take_type, true).into();

        let option_pointer: TypeHandle = MirEnumType::get(
            ctx,
            "PointerOption".into(),
            tag_type.into(),
            vec![0],
            vec![EnumVariant::new("Some".into(), vec![element_pointer])],
        )
        .into();

        let module = ModuleOp::new(ctx, "memory_resident_iterator".try_into().unwrap());
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
        let entry = BasicBlock::new(ctx, None, vec![]);
        entry.insert_at_back(region, ctx);
        let count_check = BasicBlock::new(ctx, None, vec![]);
        count_check.insert_at_back(region, ctx);
        let guard = BasicBlock::new(ctx, None, vec![]);
        guard.insert_at_back(region, ctx);
        let update = BasicBlock::new(ctx, None, vec![]);
        update.insert_at_back(region, ctx);
        let payload = BasicBlock::new(ctx, None, vec![option_pointer]);
        payload.insert_at_back(region, ctx);
        let exit = BasicBlock::new(ctx, None, vec![]);
        exit.insert_at_back(region, ctx);

        let array_alloca = Operation::new(
            ctx,
            MirAllocaOp::get_concrete_op_info(),
            vec![array_pointer],
            vec![],
            vec![],
            0,
        );
        array_alloca.insert_at_back(entry, ctx);
        let array_slot = array_alloca.deref(ctx).get_result(0);

        for index in 0..4_u64 {
            let index_value = append_u64_constant(ctx, entry, index_type, index);
            let element_addr = Operation::new(
                ctx,
                MirArrayElementAddrOp::get_concrete_op_info(),
                vec![element_pointer],
                vec![array_slot, index_value],
                vec![],
                0,
            );
            element_addr.insert_at_back(entry, ctx);
            let value = append_u64_constant(ctx, entry, element_type, index + 1);
            let element_pointer_value = element_addr.deref(ctx).get_result(0);
            let store = Operation::new(
                ctx,
                MirStoreOp::get_concrete_op_info(),
                vec![],
                vec![element_pointer_value, value],
                vec![],
                0,
            );
            store.insert_at_back(entry, ctx);
        }

        let base = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![element_pointer],
            vec![array_slot],
            vec![],
            0,
        );
        MirCastOp::new(base).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
        base.insert_at_back(entry, ctx);
        let base_pointer = base.deref(ctx).get_result(0);

        let current = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![current_wrapper],
            vec![base_pointer],
            vec![],
            0,
        );
        MirCastOp::new(current).set_attr_cast_kind(ctx, MirCastKindAttr::Transmute);
        current.insert_at_back(entry, ctx);
        let initial_current = current.deref(ctx).get_result(0);

        let end_count = append_u64_constant(ctx, entry, index_type, end_offset);
        let end = Operation::new(
            ctx,
            MirPtrOffsetOp::get_concrete_op_info(),
            vec![element_pointer],
            vec![base_pointer, end_count],
            vec![],
            0,
        );
        end.insert_at_back(entry, ctx);
        let end_pointer = end.deref(ctx).get_result(0);

        let iterator_value = append_struct(
            ctx,
            entry,
            iterator_type,
            vec![initial_current, end_pointer],
        );
        let copied_value = append_struct(ctx, entry, copied_type, vec![iterator_value]);
        let initial_count = append_u64_constant(ctx, entry, index_type, 4);
        let take_value = append_struct(ctx, entry, take_type, vec![copied_value, initial_count]);

        let iterator_alloca = Operation::new(
            ctx,
            MirAllocaOp::get_concrete_op_info(),
            vec![take_pointer],
            vec![],
            vec![],
            0,
        );
        iterator_alloca.insert_at_back(entry, ctx);
        let iterator_slot = iterator_alloca.deref(ctx).get_result(0);
        let initializer_store = Operation::new(
            ctx,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![iterator_slot, take_value],
            vec![],
            0,
        );
        initializer_store.insert_at_back(entry, ctx);

        let to_count_check = Operation::new(
            ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![count_check],
            0,
        );
        to_count_check.insert_at_back(entry, ctx);

        let remaining = append_load_from_path(ctx, count_check, iterator_slot, take_type, &[1]);
        let zero = append_u64_constant(ctx, count_check, index_type, 0);
        let empty = Operation::new(
            ctx,
            MirEqOp::get_concrete_op_info(),
            vec![i1_type],
            vec![remaining, zero],
            vec![],
            0,
        );
        empty.insert_at_back(count_check, ctx);
        let is_empty = empty.deref(ctx).get_result(0);
        append_cond_branch(ctx, count_check, is_empty, exit, guard);

        let current_count = append_load_from_path(ctx, guard, iterator_slot, take_type, &[1]);
        let one = append_u64_constant(ctx, guard, index_type, 1);
        let decremented = Operation::new(
            ctx,
            MirSubOp::get_concrete_op_info(),
            vec![index_handle],
            vec![current_count, one],
            vec![],
            0,
        );
        decremented.insert_at_back(guard, ctx);
        let (count_addr, _) = append_field_addr_path(ctx, guard, iterator_slot, take_type, &[1]);
        let decremented_count = decremented.deref(ctx).get_result(0);
        let count_store = Operation::new(
            ctx,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![count_addr, decremented_count],
            vec![],
            0,
        );
        count_store.insert_at_back(guard, ctx);

        let current_value = append_load_from_path(ctx, guard, iterator_slot, take_type, &[0, 0, 0]);
        let end_value = append_load_from_path(ctx, guard, iterator_slot, take_type, &[0, 0, 1]);
        let current_pointer_cast = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![element_pointer],
            vec![current_value],
            vec![],
            0,
        );
        MirCastOp::new(current_pointer_cast).set_attr_cast_kind(ctx, MirCastKindAttr::Transmute);
        current_pointer_cast.insert_at_back(guard, ctx);
        let raw_current = current_pointer_cast.deref(ctx).get_result(0);
        let at_end = Operation::new(
            ctx,
            MirEqOp::get_concrete_op_info(),
            vec![i1_type],
            vec![raw_current, end_value],
            vec![],
            0,
        );
        at_end.insert_at_back(guard, ctx);
        let reached_end = at_end.deref(ctx).get_result(0);
        append_cond_branch(ctx, guard, reached_end, exit, update);

        // rustc materializes the current pointer again in the continue block
        // before advancing it and building the iterator payload. Keep that
        // block-local projection in the fixture so the recognizer exercises
        // the same SSA use graph as the real `slice::Iter` lowering.
        let update_current_cast = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![element_pointer],
            vec![current_value],
            vec![],
            0,
        );
        MirCastOp::new(update_current_cast).set_attr_cast_kind(ctx, MirCastKindAttr::Transmute);
        update_current_cast.insert_at_back(update, ctx);
        let update_raw_current = update_current_cast.deref(ctx).get_result(0);

        let step_value = append_u64_constant(ctx, update, index_type, step);
        let next_pointer = Operation::new(
            ctx,
            MirPtrOffsetOp::get_concrete_op_info(),
            vec![element_pointer],
            vec![update_raw_current, step_value],
            vec![],
            0,
        );
        next_pointer.insert_at_back(update, ctx);
        let advanced_pointer = next_pointer.deref(ctx).get_result(0);
        let next_current = Operation::new(
            ctx,
            MirCastOp::get_concrete_op_info(),
            vec![current_wrapper],
            vec![advanced_pointer],
            vec![],
            0,
        );
        MirCastOp::new(next_current).set_attr_cast_kind(ctx, MirCastKindAttr::Transmute);
        next_current.insert_at_back(update, ctx);
        let (current_addr, _) =
            append_field_addr_path(ctx, update, iterator_slot, take_type, &[0, 0, 0]);
        let next_current_value = next_current.deref(ctx).get_result(0);
        let current_store = Operation::new(
            ctx,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![current_addr, next_current_value],
            vec![],
            0,
        );
        current_store.insert_at_back(update, ctx);

        let some_pointer = Operation::new(
            ctx,
            MirConstructEnumOp::get_concrete_op_info(),
            vec![option_pointer],
            vec![update_raw_current],
            vec![],
            0,
        );
        MirConstructEnumOp::new(some_pointer)
            .set_attr_construct_enum_variant_index(ctx, VariantIndexAttr(0));
        some_pointer.insert_at_back(update, ctx);
        let some_pointer_value = some_pointer.deref(ctx).get_result(0);
        let to_payload = Operation::new(
            ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![some_pointer_value],
            vec![payload],
            0,
        );
        to_payload.insert_at_back(update, ctx);

        let payload_value = payload.deref(ctx).get_argument(0);
        let payload_pointer = Operation::new(
            ctx,
            MirEnumPayloadOp::get_concrete_op_info(),
            vec![element_pointer],
            vec![payload_value],
            vec![],
            0,
        );
        let payload_op = MirEnumPayloadOp::new(payload_pointer);
        payload_op.set_attr_payload_variant_index(ctx, VariantIndexAttr(0));
        payload_op.set_attr_payload_field_index(ctx, FieldIndexAttr(0));
        payload_pointer.insert_at_back(payload, ctx);
        let payload_element_pointer = payload_pointer.deref(ctx).get_result(0);
        let element_load = Operation::new(
            ctx,
            MirLoadOp::get_concrete_op_info(),
            vec![element_handle],
            vec![payload_element_pointer],
            vec![],
            0,
        );
        element_load.insert_at_back(payload, ctx);
        let loop_back = Operation::new(
            ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![count_check],
            0,
        );
        loop_back.insert_at_back(payload, ctx);

        let return_op = Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        return_op.insert_at_back(exit, ctx);

        MemoryResidentIteratorFixture {
            module: module.get_operation(),
        }
    }

    #[test]
    fn memory_resident_small_array_iterator_becomes_bounded_extract() {
        let mut ctx = Context::new();
        let fixture = build_memory_resident_iterator_fixture(&mut ctx, 4, 1);

        let rewritten =
            canonicalize_memory_resident_small_array_iterators(fixture.module, &mut ctx);
        pliron::operation::verify_operation(fixture.module, &ctx).unwrap();

        assert_eq!(rewritten, 1);
        assert_eq!(count::<MirConstructArrayOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirRemOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirArrayElementAddrOp>(&ctx, fixture.module), 0);
    }

    #[test]
    fn memory_resident_iterator_with_non_unit_step_is_left_unchanged() {
        let mut ctx = Context::new();
        let fixture = build_memory_resident_iterator_fixture(&mut ctx, 4, 2);

        let rewritten =
            canonicalize_memory_resident_small_array_iterators(fixture.module, &mut ctx);

        assert_eq!(rewritten, 0);
        assert_eq!(count::<MirConstructArrayOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirArrayElementAddrOp>(&ctx, fixture.module), 4);
    }

    #[test]
    fn memory_resident_iterator_with_wrong_end_pointer_is_left_unchanged() {
        let mut ctx = Context::new();
        let fixture = build_memory_resident_iterator_fixture(&mut ctx, 3, 1);

        let rewritten =
            canonicalize_memory_resident_small_array_iterators(fixture.module, &mut ctx);

        assert_eq!(rewritten, 0);
        assert_eq!(count::<MirConstructArrayOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirArrayElementAddrOp>(&ctx, fixture.module), 4);
    }
}
