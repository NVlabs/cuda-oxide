/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_llvm::{
    export::{NvvmExportConfig, export_module_to_string, export_module_with_externs},
    op_interfaces::CastOpInterface,
    ops::{
        AddressOfOp, BitcastOp, BrOp, CallOp, FuncOp, GepIndex, GetElementPtrOp, GlobalOp, LoadOp,
        ReturnOp,
    },
    types::{FuncType, PointerType, VoidType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::StringAttr,
        op_interfaces::CallOpCallable,
        ops::ModuleOp,
        types::{IntegerType, Signedness},
    },
    context::Context,
    linked_list::ContainsLinkedList,
    op::Op,
};

#[test]
fn typed_nvvm_export_uses_typed_function_metadata_refs() {
    let mut ctx = Context::new();
    dialect_llvm::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();

    let void_ty = VoidType::get(&mut ctx);
    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let func_ty = FuncType::get(&mut ctx, void_ty.to_ptr(), vec![i32_ty.into()], false);
    let func = FuncOp::new(&mut ctx, "typed_kernel".try_into().unwrap(), func_ty);
    let kernel_key = "gpu_kernel".try_into().unwrap();
    func.get_operation()
        .deref_mut(&mut ctx)
        .attributes
        .0
        .insert(kernel_key, StringAttr::new("true".to_string()).into());
    let entry = func.get_or_create_entry_block(&mut ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = NvvmExportConfig::for_target(Some("sm_86"));
    let ir = export_module_with_externs::<dialect_llvm::export::DeviceExternDecl>(
        &ctx,
        &module,
        &[],
        &config,
    )
    .expect("export succeeds");

    assert!(
        ir.contains("@llvm.used = appending global [1 x i8*] [i8* bitcast (void (i32)* @typed_kernel to i8*)]"),
        "typed NVVM @llvm.used must use i8* bitcast refs:\n{ir}"
    );
    assert!(
        ir.contains("!0 = !{void (i32)* @typed_kernel, !\"kernel\", i32 1}"),
        "typed NVVM annotations must use typed function refs:\n{ir}"
    );
}

#[test]
fn typed_nvvm_export_prints_erased_pointer_types_as_i8_pointers() {
    let mut ctx = Context::new();
    dialect_llvm::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();

    let void_ty = VoidType::get(&mut ctx);
    let ptr_ty = PointerType::get_generic(&mut ctx);
    let func_ty = FuncType::get(&mut ctx, void_ty.to_ptr(), vec![ptr_ty.into()], false);
    let func = FuncOp::new(&mut ctx, "takes_ptr".try_into().unwrap(), func_ty);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = NvvmExportConfig::for_target(Some("sm_86"));
    let ir = export_module_with_externs::<dialect_llvm::export::DeviceExternDecl>(
        &ctx,
        &module,
        &[],
        &config,
    )
    .expect("export succeeds");

    assert!(
        ir.contains("declare void @takes_ptr(i8*)"),
        "typed NVVM declarations must not contain opaque `ptr` parameters:\n{ir}"
    );
}

#[test]
fn typed_nvvm_export_lowers_saturating_intrinsics_to_supported_integer_ops() {
    let mut ctx = Context::new();
    dialect_llvm::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();

    let void_ty = VoidType::get(&mut ctx);
    let i8_ty = IntegerType::get(&mut ctx, 8, Signedness::Signless);
    let func_ty = FuncType::get(
        &mut ctx,
        void_ty.to_ptr(),
        vec![i8_ty.into(), i8_ty.into()],
        false,
    );
    let func = FuncOp::new(&mut ctx, "sat_add".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let args: Vec<_> = entry.deref(&ctx).arguments().collect();
    let intrinsic_ty = FuncType::get(
        &mut ctx,
        i8_ty.into(),
        vec![i8_ty.into(), i8_ty.into()],
        false,
    );
    CallOp::new(
        &mut ctx,
        CallOpCallable::Direct("llvm_sadd_sat_i8".try_into().unwrap()),
        intrinsic_ty,
        vec![args[0], args[1]],
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = NvvmExportConfig::for_target(Some("sm_86"));
    let ir = export_module_with_externs::<dialect_llvm::export::DeviceExternDecl>(
        &ctx,
        &module,
        &[],
        &config,
    )
    .expect("export succeeds");

    assert!(
        !ir.contains("call i8 @llvm.sadd.sat.i8"),
        "typed NVVM export should not emit unsupported saturation intrinsic calls:\n{ir}"
    );
    assert!(
        ir.contains(" = add i8 %v0, %v1"),
        "saturating add lowering should start with a normal add:\n{ir}"
    );
    assert!(
        ir.contains(" = select i1") && ir.contains("i8 -128") && ir.contains("i8 127"),
        "signed saturation lowering must clamp to signed i8 bounds:\n{ir}"
    );
}

#[test]
fn typed_nvvm_export_casts_erased_pointer_before_typed_load() {
    let mut ctx = Context::new();
    dialect_llvm::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();

    let void_ty = VoidType::get(&mut ctx);
    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let ptr_ty = PointerType::get_generic(&mut ctx);
    let func_ty = FuncType::get(&mut ctx, void_ty.to_ptr(), vec![ptr_ty.into()], false);
    let func = FuncOp::new(&mut ctx, "load_i32".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let ptr_arg = entry.deref(&ctx).arguments().next().unwrap();
    LoadOp::new(&mut ctx, ptr_arg, i32_ty.into())
        .get_operation()
        .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = NvvmExportConfig::for_target(Some("sm_86"));
    let ir = export_module_with_externs::<dialect_llvm::export::DeviceExternDecl>(
        &ctx,
        &module,
        &[],
        &config,
    )
    .expect("export succeeds");

    assert!(
        ir.contains("%ptrcast0 = bitcast i8* %v0 to i32*"),
        "typed NVVM loads from erased pointer params need a repair bitcast:\n{ir}"
    );
    assert!(
        ir.contains("load i32, i32* %ptrcast0"),
        "typed NVVM load must use the repaired typed pointer:\n{ir}"
    );
}

#[test]
fn typed_nvvm_export_pretypes_late_gep_pointer_uses() {
    let mut ctx = Context::new();
    dialect_llvm::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();

    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let global = GlobalOp::new_in_address_space(
        &mut ctx,
        "shared_forward".try_into().unwrap(),
        i32_ty.to_ptr(),
        3,
    );
    global.get_operation().insert_at_back(module_block, &ctx);

    let void_ty = VoidType::get(&mut ctx);
    let func_ty = FuncType::get(&mut ctx, void_ty.to_ptr(), vec![], false);
    let func = FuncOp::new(&mut ctx, "late_gep_use".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let func_region = func.get_operation().deref(&ctx).get_region(0);
    let use_block = BasicBlock::new(&mut ctx, None, vec![]);
    use_block.insert_at_back(func_region, &ctx);
    let address_block = BasicBlock::new(&mut ctx, None, vec![]);
    address_block.insert_at_back(func_region, &ctx);

    BrOp::new(&mut ctx, address_block, vec![])
        .get_operation()
        .insert_at_back(entry, &ctx);

    let address = AddressOfOp::new(&mut ctx, "shared_forward".try_into().unwrap(), 3);
    let address_value = address.get_operation().deref(&ctx).get_result(0);
    address.get_operation().insert_at_back(address_block, &ctx);
    let erased_shared_ptr_ty = PointerType::get(&mut ctx, 3);
    let erase_address = BitcastOp::new(&mut ctx, address_value, erased_shared_ptr_ty.into());
    let erased_address_value = erase_address.get_operation().deref(&ctx).get_result(0);
    erase_address
        .get_operation()
        .insert_at_back(address_block, &ctx);
    let gep = GetElementPtrOp::new(
        &mut ctx,
        erased_address_value,
        vec![GepIndex::Constant(0)],
        i32_ty.to_ptr(),
    )
    .expect("valid GEP");
    let gep_value = gep.get_operation().deref(&ctx).get_result(0);
    gep.get_operation().insert_at_back(address_block, &ctx);
    BrOp::new(&mut ctx, use_block, vec![])
        .get_operation()
        .insert_at_back(address_block, &ctx);

    LoadOp::new(&mut ctx, gep_value, i32_ty.into())
        .get_operation()
        .insert_at_back(use_block, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(use_block, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = NvvmExportConfig::for_target(Some("sm_86"));
    let ir = export_module_with_externs::<dialect_llvm::export::DeviceExternDecl>(
        &ctx,
        &module,
        &[],
        &config,
    )
    .expect("export succeeds");

    assert!(
        ir.contains("load i32, i32 addrspace(3)* %v"),
        "typed NVVM forward uses of GEP results must keep the GEP pointee type:\n{ir}"
    );
    assert!(
        !ir.contains("bitcast i8 addrspace(3)* %v2 to i32 addrspace(3)*"),
        "typed NVVM should not repair a pretyped GEP result from erased i8*:\n{ir}"
    );
}

#[test]
fn export_addressof_uses_symbol_when_definition_block_prints_later() {
    let mut ctx = Context::new();
    dialect_llvm::register(&mut ctx);

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
    let global = GlobalOp::new_in_address_space(
        &mut ctx,
        "__shared_mem_20".try_into().unwrap(),
        i32_ty.to_ptr(),
        3,
    );
    global.get_operation().insert_at_back(module_block, &ctx);

    let void_ty = VoidType::get(&mut ctx);
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
    )
    .expect("valid GEP");
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
