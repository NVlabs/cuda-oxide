/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Forward compiler-owned multi-result operations across their Rust aggregate
//! return-ABI adapter.
//!
//! Some device operations produce many independent registers, while their Rust
//! API returns an array, tuple, or one-field SIMD wrapper.  The MIR importer has
//! to construct that aggregate and store it to the destination local.  Address
//! projections in the successor then prevent ordinary mem2reg from recovering
//! the original SSA results.  For wide operations this can become a large
//! insertvalue/store/GEP/load web and materially change register allocation.
//!
//! This pass is deliberately narrower than aggregate SROA.  It considers only
//! importer-marked `mir.compiler_result_bundle` constructors and rewrites only
//! after proving the complete boundary:
//!
//! * the construct tree contains every result of exactly one producer, in order;
//! * the outer bundle has one non-volatile store to one local alloca;
//! * the alloca has no other stores, copies, escapes, or unknown pointer users;
//! * every read is a non-volatile load through constant field/array projections;
//! * the producer/store dominate every rewritten load.
//!
//! Any failed proof leaves the IR untouched.  Ordinary Rust aggregates are
//! never candidates because only the importer can attach the marker.

use dialect_mir::{
    attributes::{COMPILER_RESULT_BUNDLE_ATTR_KEY, CompilerResultBundleAttr},
    ops::{
        MirAllocaOp, MirArrayElementAddrOp, MirConstantOp, MirConstructArrayOp,
        MirConstructStructOp, MirConstructTupleOp, MirFieldAddrOp, MirLoadOp, MirStoreOp,
    },
};
use pliron::{
    basic_block::BasicBlock,
    context::{Context, Ptr},
    graph::dominance::DomInfo,
    irbuild::{
        listener::Recorder,
        rewriter::{IRRewriter, Rewriter},
    },
    linked_list::ContainsLinkedList,
    operation::Operation,
    pass::AnalysisManager,
    result::Result,
    r#type::Typed,
    value::Value,
};
use rustc_hash::FxHashSet;

#[derive(Clone, Copy)]
struct LoadForwarding {
    load: Ptr<Operation>,
    replacement: Value,
}
struct ForwardingPlan {
    store: Ptr<Operation>,
    bundle_nodes: Vec<Ptr<Operation>>,
    projection_nodes: Vec<Ptr<Operation>>,
    loads: Vec<LoadForwarding>,
    producer_block: Ptr<BasicBlock>,
    store_block: Ptr<BasicBlock>,
}

/// Remove proven compiler-created aggregate ABI boundaries before mem2reg.
pub fn forward_compiler_result_bundles(
    module: Ptr<Operation>,
    ctx: &mut Context,
    analyses: &mut AnalysisManager,
    verbose: bool,
) -> Result<usize> {
    let mut operations = Vec::new();
    collect_ops(ctx, module, &mut operations);

    let mut plans = Vec::new();
    for operation in operations {
        let Some(plan) = analyze_marked_bundle(ctx, operation) else {
            continue;
        };
        if domination_is_proven(module, ctx, analyses, &plan)? {
            plans.push(plan);
        }
    }

    let mut forwarded_loads = 0;
    for plan in plans {
        forwarded_loads += rewrite_plan(ctx, plan);
    }

    if forwarded_loads > 0 && verbose {
        eprintln!(
            "compiler-result forwarding: replaced {forwarded_loads} aggregate projection load(s)"
        );
    }
    Ok(forwarded_loads)
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

fn analyze_marked_bundle(ctx: &Context, outer: Ptr<Operation>) -> Option<ForwardingPlan> {
    let key = COMPILER_RESULT_BUNDLE_ATTR_KEY.try_into().ok()?;
    let outer_ref = outer.deref(ctx);
    let marker = outer_ref.attributes.get::<CompilerResultBundleAttr>(&key)?;
    if !marker.0 || !is_construct_op(ctx, outer) {
        return None;
    }

    let outer_value = outer.deref(ctx).get_result(0);
    if outer_value.num_uses(ctx) != 1 {
        return None;
    }
    let outer_use = outer_value.uses(ctx).into_iter().next()?;
    let store_op = outer_use.user_op();
    let store = Operation::get_op::<MirStoreOp>(store_op, ctx)?;
    if outer_use.find_index(ctx) != 1 || store.is_volatile(ctx) {
        return None;
    }

    let root_pointer = store.address_opd(ctx);
    let alloca_op = root_pointer.defining_op()?;
    Operation::get_op::<MirAllocaOp>(alloca_op, ctx)?;

    let mut bundle_nodes = Vec::new();
    let mut leaves = Vec::new();
    flatten_construct_tree(ctx, outer_value, &mut bundle_nodes, &mut leaves)?;
    let first_leaf = *leaves.first()?;
    let producer = first_leaf.defining_op()?;
    if producer.deref(ctx).get_num_results() != leaves.len() || leaves.len() <= 1 {
        return None;
    }
    for (index, leaf) in leaves.iter().enumerate() {
        if *leaf != producer.deref(ctx).get_result(index)
            || leaf.defining_op() != Some(producer)
            || leaf.num_uses(ctx) != 1
        {
            return None;
        }
    }

    let producer_block = producer.deref(ctx).get_parent_block()?;
    let store_block = store_op.deref(ctx).get_parent_block()?;
    if producer_block != store_block || !operation_precedes(ctx, producer, store_op) {
        return None;
    }
    if bundle_nodes.iter().any(|node| {
        node.deref(ctx).get_parent_block() != Some(store_block)
            || !operation_precedes(ctx, *node, store_op)
    }) {
        return None;
    }

    let mut projection_nodes = Vec::new();
    let mut seen_projections = FxHashSet::default();
    let mut loads = Vec::new();
    for root_use in root_pointer.uses(ctx) {
        let user = root_use.user_op();
        let operand_index = root_use.find_index(ctx);
        if user == store_op && operand_index == 0 {
            continue;
        }
        if operand_index != 0
            || (!Operation::get_op::<MirFieldAddrOp>(user, ctx).is_some()
                && !Operation::get_op::<MirArrayElementAddrOp>(user, ctx).is_some())
        {
            return None;
        }
        analyze_projection(
            ctx,
            user,
            outer_value,
            &bundle_nodes,
            Vec::new(),
            &mut seen_projections,
            &mut projection_nodes,
            &mut loads,
        )?;
    }
    if loads.is_empty() {
        return None;
    }

    Some(ForwardingPlan {
        store: store_op,
        bundle_nodes,
        projection_nodes,
        loads,
        producer_block,
        store_block,
    })
}

fn flatten_construct_tree(
    ctx: &Context,
    value: Value,
    nodes: &mut Vec<Ptr<Operation>>,
    leaves: &mut Vec<Value>,
) -> Option<()> {
    let Some(operation) = value.defining_op() else {
        leaves.push(value);
        return Some(());
    };
    if !is_construct_op(ctx, operation) {
        leaves.push(value);
        return Some(());
    }
    if operation.deref(ctx).get_result(0) != value || value.num_uses(ctx) != 1 {
        return None;
    }
    nodes.push(operation);
    for index in 0..operation.deref(ctx).get_num_operands() {
        flatten_construct_tree(ctx, operation.deref(ctx).get_operand(index), nodes, leaves)?;
    }
    Some(())
}

fn is_construct_op(ctx: &Context, operation: Ptr<Operation>) -> bool {
    Operation::get_op::<MirConstructArrayOp>(operation, ctx).is_some()
        || Operation::get_op::<MirConstructTupleOp>(operation, ctx).is_some()
        || Operation::get_op::<MirConstructStructOp>(operation, ctx).is_some()
}

#[allow(clippy::too_many_arguments)]
fn analyze_projection(
    ctx: &Context,
    projection: Ptr<Operation>,
    bundle: Value,
    bundle_nodes: &[Ptr<Operation>],
    mut path: Vec<usize>,
    seen: &mut FxHashSet<Ptr<Operation>>,
    projection_nodes: &mut Vec<Ptr<Operation>>,
    loads: &mut Vec<LoadForwarding>,
) -> Option<()> {
    if !seen.insert(projection) {
        return None;
    }

    if let Some(field) = Operation::get_op::<MirFieldAddrOp>(projection, ctx) {
        path.push(field.get_attr_field_index(ctx)?.0 as usize);
    } else if Operation::get_op::<MirArrayElementAddrOp>(projection, ctx).is_some() {
        path.push(integer_constant_usize(
            ctx,
            projection.deref(ctx).get_operand(1),
        )?);
    } else {
        return None;
    }

    projection_nodes.push(projection);
    let pointer = projection.deref(ctx).get_result(0);
    if pointer.num_uses(ctx) == 0 {
        return None;
    }
    for pointer_use in pointer.uses(ctx) {
        if pointer_use.find_index(ctx) != 0 {
            return None;
        }
        let user = pointer_use.user_op();
        if let Some(load) = Operation::get_op::<MirLoadOp>(user, ctx) {
            if load.is_volatile(ctx) {
                return None;
            }
            let replacement = resolve_construct_path(ctx, bundle, &path)?;
            // A whole-subaggregate read (e.g. loading the array field of a
            // struct-wrapped bundle in one piece) resolves to one of the
            // bundle's own construct results.  Forwarding it would leave live
            // uses of an operation this pass erases, so fail closed and keep
            // the memory path.
            if replacement
                .defining_op()
                .is_some_and(|definer| bundle_nodes.contains(&definer))
            {
                return None;
            }
            if replacement.get_type(ctx) != user.deref(ctx).get_result(0).get_type(ctx) {
                return None;
            }
            loads.push(LoadForwarding {
                load: user,
                replacement,
            });
            continue;
        }
        if Operation::get_op::<MirFieldAddrOp>(user, ctx).is_none()
            && Operation::get_op::<MirArrayElementAddrOp>(user, ctx).is_none()
        {
            return None;
        }
        analyze_projection(
            ctx,
            user,
            bundle,
            bundle_nodes,
            path.clone(),
            seen,
            projection_nodes,
            loads,
        )?;
    }
    Some(())
}

fn integer_constant_usize(ctx: &Context, value: Value) -> Option<usize> {
    let operation = value.defining_op()?;
    let constant = Operation::get_op::<MirConstantOp>(operation, ctx)?;
    let attribute = constant.get_attr_value(ctx)?;
    let integer = attribute.value();
    if integer.bw() > usize::BITS as usize {
        return None;
    }
    usize::try_from(integer.to_u64()).ok()
}

fn resolve_construct_path(ctx: &Context, mut value: Value, path: &[usize]) -> Option<Value> {
    for &index in path {
        let operation = value.defining_op()?;
        if !is_construct_op(ctx, operation) || operation.deref(ctx).get_result(0) != value {
            return None;
        }
        if index >= operation.deref(ctx).get_num_operands() {
            return None;
        }
        value = operation.deref(ctx).get_operand(index);
    }
    Some(value)
}

fn domination_is_proven(
    module: Ptr<Operation>,
    ctx: &mut Context,
    analyses: &mut AnalysisManager,
    plan: &ForwardingPlan,
) -> Result<bool> {
    let Some(region) = plan.store_block.deref(ctx).get_parent_region() else {
        return Ok(false);
    };
    if plan.producer_block.deref(ctx).get_parent_region() != Some(region) {
        return Ok(false);
    }

    let mut dom_info = analyses.get_analysis_mut::<DomInfo>(module, ctx)?;
    let dom = dom_info.get_dom_tree(ctx, region);
    for load in &plan.loads {
        let Some(load_block) = load.load.deref(ctx).get_parent_block() else {
            return Ok(false);
        };
        if load_block.deref(ctx).get_parent_region() != Some(region) {
            return Ok(false);
        }
        if load_block == plan.store_block {
            if !operation_precedes(ctx, plan.store, load.load) {
                return Ok(false);
            }
        } else if !dom.dominates(&plan.store_block, &load_block) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn operation_precedes(ctx: &Context, first: Ptr<Operation>, second: Ptr<Operation>) -> bool {
    let Some(block) = first.deref(ctx).get_parent_block() else {
        return false;
    };
    if second.deref(ctx).get_parent_block() != Some(block) {
        return false;
    }
    for operation in block.deref(ctx).iter(ctx) {
        if operation == first {
            return true;
        }
        if operation == second {
            return false;
        }
    }
    false
}

fn rewrite_plan(ctx: &mut Context, plan: ForwardingPlan) -> usize {
    let count = plan.loads.len();
    let mut rewriter = IRRewriter::<Recorder>::default();

    for forwarding in plan.loads {
        let old_result = forwarding.load.deref(ctx).get_result(0);
        old_result.replace_all_uses_with(ctx, &forwarding.replacement);
        rewriter.erase_operation(ctx, forwarding.load);
    }
    for projection in plan.projection_nodes.into_iter().rev() {
        rewriter.erase_operation(ctx, projection);
    }
    rewriter.erase_operation(ctx, plan.store);
    for constructor in plan.bundle_nodes {
        rewriter.erase_operation(ctx, constructor);
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use dialect_mir::{
        attributes::{CompilerResultBundleAttr, FieldIndexAttr},
        ops::{MirCallOp, MirFuncOp, MirGotoOp, MirReturnOp},
        types::{MirArrayType, MirPtrType, MirStructType},
    };
    use pliron::{
        builtin::{
            attributes::{IntegerAttr, StringAttr, TypeAttr},
            op_interfaces::{SingleBlockRegionInterface, SymbolOpInterface},
            ops::ModuleOp,
            types::{FunctionType, IntegerType, Signedness},
        },
        identifier::Identifier,
        op::Op,
        region::Region,
        r#type::TypeHandle,
        utils::apint::APInt,
    };
    use std::num::NonZeroUsize;

    struct Fixture {
        module: Ptr<Operation>,
        producer: Ptr<Operation>,
        return_op: Ptr<Operation>,
    }

    fn build_fixture(ctx: &mut Context, marked: bool, extra_store: bool, width: usize) -> Fixture {
        dialect_mir::register(ctx);

        let element_type: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let array_type: TypeHandle = MirArrayType::get(ctx, element_type, width as u64).into();
        let pointer_type: TypeHandle = MirPtrType::get_generic(ctx, array_type, true).into();

        let module = ModuleOp::new(ctx, "test".try_into().unwrap());
        let function_type = FunctionType::get(ctx, vec![], vec![element_type]);
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
        let body = BasicBlock::new(ctx, None, vec![]);
        body.insert_at_back(region, ctx);

        let alloca = Operation::new(
            ctx,
            MirAllocaOp::get_concrete_op_info(),
            vec![pointer_type],
            vec![],
            vec![],
            0,
        );
        alloca.insert_at_back(entry, ctx);
        let slot = alloca.deref(ctx).get_result(0);

        // A call is sufficient as a typed, independent two-result producer for
        // this pass test.  The forwarding proof does not depend on its opcode.
        let producer = Operation::new(
            ctx,
            MirCallOp::get_concrete_op_info(),
            vec![element_type; width],
            vec![],
            vec![],
            0,
        );
        MirCallOp::new(producer).set_attr_callee(ctx, StringAttr::new("register_pack".to_string()));
        producer.insert_at_back(entry, ctx);

        let producer_results: Vec<_> = (0..width)
            .map(|index| producer.deref(ctx).get_result(index))
            .collect();
        let bundle = Operation::new(
            ctx,
            MirConstructArrayOp::get_concrete_op_info(),
            vec![array_type],
            producer_results,
            vec![],
            0,
        );
        if marked {
            bundle.deref_mut(ctx).attributes.set(
                Identifier::try_from(COMPILER_RESULT_BUNDLE_ATTR_KEY).unwrap(),
                CompilerResultBundleAttr(true),
            );
        }
        bundle.insert_at_back(entry, ctx);
        let bundle_value = bundle.deref(ctx).get_result(0);

        let store = Operation::new(
            ctx,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![slot, bundle_value],
            vec![],
            0,
        );
        store.insert_at_back(entry, ctx);
        if extra_store {
            let store = Operation::new(
                ctx,
                MirStoreOp::get_concrete_op_info(),
                vec![],
                vec![slot, bundle_value],
                vec![],
                0,
            );
            store.insert_at_back(entry, ctx);
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

        let index_type = IntegerType::get(ctx, 64, Signedness::Unsigned);
        let index = Operation::new(
            ctx,
            MirConstantOp::get_concrete_op_info(),
            vec![index_type.into()],
            vec![],
            vec![],
            0,
        );
        MirConstantOp::new(index).set_attr_value(
            ctx,
            IntegerAttr::new(
                index_type,
                APInt::from_u64((width - 1) as u64, NonZeroUsize::new(64).unwrap()),
            ),
        );
        index.insert_at_back(body, ctx);

        let element_pointer: TypeHandle = MirPtrType::get_generic(ctx, element_type, false).into();
        let index_value = index.deref(ctx).get_result(0);
        let address = Operation::new(
            ctx,
            MirArrayElementAddrOp::get_concrete_op_info(),
            vec![element_pointer],
            vec![slot, index_value],
            vec![],
            0,
        );
        address.insert_at_back(body, ctx);
        let address_value = address.deref(ctx).get_result(0);
        let load = Operation::new(
            ctx,
            MirLoadOp::get_concrete_op_info(),
            vec![element_type],
            vec![address_value],
            vec![],
            0,
        );
        load.insert_at_back(body, ctx);

        let load_value = load.deref(ctx).get_result(0);
        let return_op = Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![load_value],
            vec![],
            0,
        );
        return_op.insert_at_back(body, ctx);

        Fixture {
            module: module.get_operation(),
            producer,
            return_op,
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

    /// Models the importer output for a struct-wrapped bundle such as
    /// `CuSimd([r0, r1])` that is read back in one piece (`CuSimd::to_array`):
    /// a `mir.field_addr` projection followed by a whole-array load.
    fn build_struct_wrapped_fixture(ctx: &mut Context) -> Fixture {
        dialect_mir::register(ctx);

        let element_type: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let array_type: TypeHandle = MirArrayType::get(ctx, element_type, 2).into();
        let struct_type: TypeHandle = MirStructType::get(
            ctx,
            "CuSimd".to_string(),
            vec!["inner".to_string()],
            vec![array_type],
        )
        .into();
        let pointer_type: TypeHandle = MirPtrType::get_generic(ctx, struct_type, true).into();

        let module = ModuleOp::new(ctx, "test".try_into().unwrap());
        let function_type = FunctionType::get(ctx, vec![], vec![array_type]);
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
        let body = BasicBlock::new(ctx, None, vec![]);
        body.insert_at_back(region, ctx);

        let alloca = Operation::new(
            ctx,
            MirAllocaOp::get_concrete_op_info(),
            vec![pointer_type],
            vec![],
            vec![],
            0,
        );
        alloca.insert_at_back(entry, ctx);
        let slot = alloca.deref(ctx).get_result(0);

        let producer = Operation::new(
            ctx,
            MirCallOp::get_concrete_op_info(),
            vec![element_type, element_type],
            vec![],
            vec![],
            0,
        );
        MirCallOp::new(producer).set_attr_callee(ctx, StringAttr::new("register_pair".to_string()));
        producer.insert_at_back(entry, ctx);

        let producer_results = vec![
            producer.deref(ctx).get_result(0),
            producer.deref(ctx).get_result(1),
        ];
        let inner = Operation::new(
            ctx,
            MirConstructArrayOp::get_concrete_op_info(),
            vec![array_type],
            producer_results,
            vec![],
            0,
        );
        inner.insert_at_back(entry, ctx);
        let inner_value = inner.deref(ctx).get_result(0);

        let outer = Operation::new(
            ctx,
            MirConstructStructOp::get_concrete_op_info(),
            vec![struct_type],
            vec![inner_value],
            vec![],
            0,
        );
        outer.deref_mut(ctx).attributes.set(
            Identifier::try_from(COMPILER_RESULT_BUNDLE_ATTR_KEY).unwrap(),
            CompilerResultBundleAttr(true),
        );
        outer.insert_at_back(entry, ctx);
        let outer_value = outer.deref(ctx).get_result(0);

        let store = Operation::new(
            ctx,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![slot, outer_value],
            vec![],
            0,
        );
        store.insert_at_back(entry, ctx);

        let goto = Operation::new(
            ctx,
            MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![body],
            0,
        );
        goto.insert_at_back(entry, ctx);

        let array_pointer: TypeHandle = MirPtrType::get_generic(ctx, array_type, false).into();
        let field_address = Operation::new(
            ctx,
            MirFieldAddrOp::get_concrete_op_info(),
            vec![array_pointer],
            vec![slot],
            vec![],
            0,
        );
        MirFieldAddrOp::new(field_address).set_attr_field_index(ctx, FieldIndexAttr(0));
        field_address.insert_at_back(body, ctx);
        let field_address_value = field_address.deref(ctx).get_result(0);

        let load = Operation::new(
            ctx,
            MirLoadOp::get_concrete_op_info(),
            vec![array_type],
            vec![field_address_value],
            vec![],
            0,
        );
        load.insert_at_back(body, ctx);

        let load_value = load.deref(ctx).get_result(0);
        let return_op = Operation::new(
            ctx,
            MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![load_value],
            vec![],
            0,
        );
        return_op.insert_at_back(body, ctx);

        Fixture {
            module: module.get_operation(),
            producer,
            return_op,
        }
    }

    #[test]
    fn marked_exact_bundle_forwards_to_producer_results() {
        let mut ctx = Context::new();
        let fixture = build_fixture(&mut ctx, true, false, 2);
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 1);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirArrayElementAddrOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirConstructArrayOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirStoreOp>(&ctx, fixture.module), 0);
        assert_eq!(
            fixture.return_op.deref(&ctx).get_operand(0),
            fixture.producer.deref(&ctx).get_result(1)
        );
    }

    /// A 64-result producer models the widest `ptx_asm!` output pack the
    /// macro now accepts (Blackwell tcgen05 tensor-memory loads).  The pass
    /// must forward the full-width bundle just like the two-wide case.
    #[test]
    fn sixty_four_result_pack_forwards_to_producer_results() {
        let mut ctx = Context::new();
        let fixture = build_fixture(&mut ctx, true, false, 64);
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 1);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirConstructArrayOp>(&ctx, fixture.module), 0);
        assert_eq!(count::<MirStoreOp>(&ctx, fixture.module), 0);
        assert_eq!(
            fixture.return_op.deref(&ctx).get_operand(0),
            fixture.producer.deref(&ctx).get_result(63)
        );
    }

    /// Regression test: a whole-array field load out of a struct-wrapped
    /// bundle resolves to the bundle's own inner construct op.  Forwarding it
    /// would leave live uses of an erased operation (an ICE), so the pass must
    /// fail closed and leave the memory path intact.
    #[test]
    fn struct_wrapped_whole_array_load_fails_closed() {
        let mut ctx = Context::new();
        let fixture = build_struct_wrapped_fixture(&mut ctx);
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirFieldAddrOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirConstructStructOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirConstructArrayOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirStoreOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn ordinary_unmarked_aggregate_is_untouched() {
        let mut ctx = Context::new();
        let fixture = build_fixture(&mut ctx, false, false, 2);
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirConstructArrayOp>(&ctx, fixture.module), 1);
    }

    #[test]
    fn additional_bundle_store_fails_closed() {
        let mut ctx = Context::new();
        let fixture = build_fixture(&mut ctx, true, true, 2);
        let mut analyses = AnalysisManager::default();

        let forwarded =
            forward_compiler_result_bundles(fixture.module, &mut ctx, &mut analyses, false)
                .unwrap();

        assert_eq!(forwarded, 0);
        assert_eq!(count::<MirLoadOp>(&ctx, fixture.module), 1);
        assert_eq!(count::<MirStoreOp>(&ctx, fixture.module), 2);
    }
}
