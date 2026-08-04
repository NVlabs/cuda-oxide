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

use dialect_mir::{
    attributes::FieldIndexAttr,
    ops::{
        MirAllocaOp, MirArrayElementAddrOp, MirAssertOp, MirConstantOp, MirExtractArrayElementOp,
        MirExtractFieldOp, MirFieldAddrOp, MirFuncOp, MirLoadOp, MirLtOp, MirRemOp, MirStoreOp,
    },
    types::{MirArrayType, MirPtrType, MirStructType},
};
use pliron::{
    builtin::types::{IntegerType, Signedness},
    context::{Context, Ptr},
    graph::ControlFlowGraph,
    irbuild::{
        listener::Recorder,
        rewriter::{IRRewriter, Rewriter},
    },
    linked_list::ContainsLinkedList,
    location::Located,
    op::Op,
    operation::Operation,
    r#type::{TypeHandle, Typed},
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
/// graph accepted by [`analyze_alloca`].
pub fn canonicalize_read_only_aggregate_arguments(module: Ptr<Operation>, ctx: &mut Context) {
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

    if rewritten_loads > 0 && std::env::var_os("CUDA_OXIDE_VERBOSE").is_some() {
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

const MAX_SCALARIZED_CANDIDATES: u64 = 16;

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
pub fn canonicalize_bounded_borrowed_pointer_arguments(module: Ptr<Operation>, ctx: &mut Context) {
    let mut operations = Vec::new();
    collect_ops(ctx, module, &mut operations);

    let array_addrs: Vec<_> = operations
        .into_iter()
        .filter(|operation| Operation::get_op::<MirArrayElementAddrOp>(*operation, ctx).is_some())
        .collect();

    let mut rewritten_loads = 0usize;
    for array_addr in array_addrs {
        let Some(plan) = analyze_borrowed_pointer_read(ctx, array_addr) else {
            continue;
        };
        rewrite_borrowed_pointer_read(ctx, plan);
        rewritten_loads += 1;
    }

    if rewritten_loads > 0 && std::env::var_os("CUDA_OXIDE_VERBOSE").is_some() {
        eprintln!(
            "borrowed-pointer aggregate scalarization: rewrote \
             {rewritten_loads} dynamic load(s)"
        );
    }
}

fn analyze_borrowed_pointer_read(
    ctx: &Context,
    array_addr: Ptr<Operation>,
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

fn integer_constant_u64(ctx: &Context, value: Value) -> Option<u64> {
    let defining_op = value.defining_op()?;
    let constant = Operation::get_op::<MirConstantOp>(defining_op, ctx)?;
    constant
        .get_attr_value(ctx)
        .map(|attribute| attribute.value().to_u64())
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

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::{
        ops::{MirGotoOp, MirReturnOp},
        types::MirArrayType,
    };
    use pliron::{
        basic_block::BasicBlock,
        builtin::{
            attributes::{IntegerAttr, TypeAttr},
            op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
            ops::ModuleOp,
            types::FunctionType,
        },
        region::Region,
        utils::apint::APInt,
    };
    use std::num::NonZeroUsize;

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

        canonicalize_read_only_aggregate_arguments(fixture.module, &mut ctx);

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

        canonicalize_read_only_aggregate_arguments(fixture.module, &mut ctx);

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

        canonicalize_read_only_aggregate_arguments(fixture.module, &mut ctx);

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

        canonicalize_read_only_aggregate_arguments(fixture.module, &mut ctx);

        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn volatile_load_rejects_the_entire_slot() {
        let mut ctx = Context::new();
        let fixture = build_fixture(&mut ctx, 3, Some(3), false, true);

        canonicalize_read_only_aggregate_arguments(fixture.module, &mut ctx);

        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
    }

    struct BorrowedPointerFixture {
        module: Ptr<Operation>,
    }

    fn build_borrowed_pointer_fixture(
        ctx: &mut Context,
        asserted_bound: Option<u64>,
        alwaysinline: bool,
        volatile_load: bool,
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
        function_op.set_symbol_name(ctx, "helper".try_into().unwrap());
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

        BorrowedPointerFixture {
            module: module.get_operation(),
        }
    }

    #[test]
    fn asserted_immutable_pointer_read_is_canonicalized_after_mem2reg() {
        let mut ctx = Context::new();
        let fixture = build_borrowed_pointer_fixture(&mut ctx, Some(3), true, false);

        canonicalize_bounded_borrowed_pointer_arguments(fixture.module, &mut ctx);

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
        let fixture = build_borrowed_pointer_fixture(&mut ctx, None, true, false);

        canonicalize_bounded_borrowed_pointer_arguments(fixture.module, &mut ctx);

        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirFieldAddrOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirArrayElementAddrOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn non_alwaysinline_pointer_helper_is_left_unchanged() {
        let mut ctx = Context::new();
        let fixture = build_borrowed_pointer_fixture(&mut ctx, Some(3), false, false);

        canonicalize_bounded_borrowed_pointer_arguments(fixture.module, &mut ctx);

        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn volatile_pointer_read_is_left_unchanged() {
        let mut ctx = Context::new();
        let fixture = build_borrowed_pointer_fixture(&mut ctx, Some(3), true, true);

        canonicalize_bounded_borrowed_pointer_arguments(fixture.module, &mut ctx);

        assert_eq!(count::<MirExtractArrayElementOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
    }
}
