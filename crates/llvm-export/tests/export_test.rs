/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use llvm_export::{
    export::export_module_to_string,
    ops::{
        AddressOfOp, BrOp, FuncOp, GepIndex, GetElementPtrOp, GlobalOp, LoadOp, ReturnOp, StoreOp,
    },
    types::{FuncType, PointerType, VoidType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        ops::ModuleOp,
        types::{IntegerType, Signedness},
    },
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
};

fn module_top_block(ctx: &mut Context, module: &ModuleOp) -> Ptr<BasicBlock> {
    let module_region = module.get_operation().deref(ctx).get_region(0);
    {
        let region = module_region.deref(ctx);
        if let Some(block) = region.iter(ctx).next() {
            return block;
        }
    }

    let block = BasicBlock::new(ctx, None, vec![]);
    block.insert_at_back(module_region, ctx);
    block
}

#[test]
fn export_volatile_load_prints_keyword() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let ptr_ty = PointerType::get(&mut ctx, 0);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&mut ctx, void_ty.to_ptr(), vec![ptr_ty.to_ptr()], false);
    let func = FuncOp::new(&mut ctx, "volatile_load_test".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let ptr = entry.deref(&ctx).get_argument(0);

    let load = LoadOp::new(&mut ctx, ptr, i32_ty.to_ptr());
    llvm_export::ops::set_op_volatile(&mut ctx, load.get_operation(), true);
    load.get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string(&ctx, &module).expect("export succeeds");
    let line = ir
        .lines()
        .find(|line| line.contains("load volatile"))
        .expect("volatile load line");

    assert!(
        line.trim_start().contains(" = load volatile i32, ptr "),
        "volatile load keyword must appear immediately after load:\n{ir}"
    );
}

#[test]
fn export_volatile_store_prints_keyword() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let ptr_ty = PointerType::get(&mut ctx, 0);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(
        &mut ctx,
        void_ty.to_ptr(),
        vec![ptr_ty.to_ptr(), i32_ty.to_ptr()],
        false,
    );
    let func = FuncOp::new(&mut ctx, "volatile_store_test".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let ptr = entry.deref(&ctx).get_argument(0);
    let val = entry.deref(&ctx).get_argument(1);

    let store = StoreOp::new(&mut ctx, val, ptr);
    llvm_export::ops::set_op_volatile(&mut ctx, store.get_operation(), true);
    store.get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string(&ctx, &module).expect("export succeeds");
    let line = ir
        .lines()
        .find(|line| line.contains("store volatile"))
        .expect("volatile store line");

    assert!(
        line.trim_start().starts_with("store volatile i32 "),
        "volatile store keyword must appear immediately after store:\n{ir}"
    );
}

#[test]
fn export_addressof_uses_symbol_when_definition_block_prints_later() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let existing = {
            let region = module_region.deref(&ctx);
            region.iter(&ctx).next()
        };
        if let Some(block) = existing {
            block
        } else {
            let block = BasicBlock::new(&mut ctx, None, vec![]);
            block.insert_at_back(module_region, &ctx);
            block
        }
    };

    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let global = GlobalOp::new(
        &mut ctx,
        "__shared_mem_20".try_into().unwrap(),
        i32_ty.to_ptr(),
    );
    global.set_address_space(&mut ctx, 3);
    global.get_operation().insert_at_back(module_block, &ctx);

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&mut ctx, void_ty.to_ptr(), vec![], false);
    let func = FuncOp::new(&mut ctx, "uses_late_addressof".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let func_region = func.get_operation().deref(&ctx).get_region(0);
    let use_block = BasicBlock::new(&mut ctx, None, vec![]);
    use_block.insert_at_back(func_region, &ctx);
    let address_block = BasicBlock::new(&mut ctx, None, vec![]);
    address_block.insert_at_back(func_region, &ctx);

    BrOp::new(&mut ctx, address_block, vec![])
        .get_operation()
        .insert_at_back(entry, &ctx);

    let address = AddressOfOp::new(&mut ctx, "__shared_mem_20".try_into().unwrap(), 3);
    let address_value = address.get_operation().deref(&ctx).get_result(0);
    address.get_operation().insert_at_back(address_block, &ctx);
    BrOp::new(&mut ctx, use_block, vec![])
        .get_operation()
        .insert_at_back(address_block, &ctx);

    let gep = GetElementPtrOp::new(
        &mut ctx,
        address_value,
        vec![GepIndex::Constant(0)],
        i32_ty.to_ptr(),
    );
    gep.get_operation().insert_at_back(use_block, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(use_block, &ctx);

    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string(&ctx, &module).expect("export succeeds");

    // The shared global must be declared at module scope.
    assert!(
        ir.contains("@__shared_mem_20 = addrspace(3) global"),
        "module must declare the shared global:\n{ir}"
    );

    // The GEP base operand must be the global symbol, not a stale `%vN`.
    let gep_line = ir
        .lines()
        .find(|line| line.contains("getelementptr inbounds"))
        .expect("exported GEP line");
    assert!(
        gep_line.contains("@__shared_mem_20"),
        "GEP must use the global symbol, not a stale temporary:\n{ir}"
    );

    // Bug class from issue #54: every `%vN` reference in the IR must have a
    // matching `%vN = ...` definition. With the bug present the addressof
    // result was named `%v1` but never defined; this catches that and any
    // future regression that re-introduces a dangling SSA reference.
    assert_no_undefined_temporaries(&ir);
}

/// Scans the textual LLVM IR and asserts that every `%vN` token appearing in
/// an operand position has a corresponding `%vN = ...` definition somewhere
/// in the module. Operates on `%v` temporaries only because that's the
/// exporter's naming scheme; named values like `%entry` (block labels) are
/// ignored by construction.
fn assert_no_undefined_temporaries(ir: &str) {
    use std::collections::HashSet;

    let mut defined: HashSet<String> = HashSet::new();
    for line in ir.lines() {
        let trimmed = line.trim_start();
        if !trimmed.starts_with("%v") {
            continue;
        }
        let Some((lhs, _)) = trimmed.split_once('=') else {
            continue;
        };
        defined.insert(lhs.trim().to_string());
    }

    let mut referenced: HashSet<String> = HashSet::new();
    for line in ir.lines() {
        let trimmed = line.trim_start();
        // Skip the lhs of a definition; only operand positions can be stale.
        let body = if trimmed.starts_with("%v")
            && let Some(eq) = trimmed.find('=')
        {
            &trimmed[eq + 1..]
        } else {
            trimmed
        };
        for tok in body.split(|c: char| !c.is_alphanumeric() && c != '%' && c != '_') {
            if let Some(num) = tok.strip_prefix("%v")
                && !num.is_empty()
                && num.chars().all(|c| c.is_ascii_digit())
            {
                referenced.insert(format!("%v{num}"));
            }
        }
    }

    let mut undefined: Vec<&String> = referenced.difference(&defined).collect();
    undefined.sort();
    assert!(
        undefined.is_empty(),
        "IR references undefined SSA temporaries: {undefined:?}\nIR:\n{ir}"
    );
}
