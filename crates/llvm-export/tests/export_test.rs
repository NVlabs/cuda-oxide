/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use combine::stream::position::SourcePosition;
use llvm_export::{
    export::{
        DebugKind, ExportBackendConfig, NvvmExportConfig, PtxExportConfig, export_module_to_string,
        export_module_to_string_with_config,
    },
    ops::{
        AddressOfOp, BrOp, CallOp, FuncOp, GepIndex, GetElementPtrOp, GlobalOp, InlineAsmOp,
        ReturnOp,
    },
    types::{FuncType, VoidType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, StringAttr},
        op_interfaces::CallOpCallable,
        ops::ModuleOp,
        types::{IntegerType, Signedness},
    },
    context::Context,
    identifier::Identifier,
    linked_list::ContainsLinkedList,
    location::{Located, Location, Source},
    op::Op,
    utils::apint::APInt,
};
use std::{num::NonZero, path::PathBuf};

struct DebugConfig<C> {
    inner: C,
    debug_kind: DebugKind,
}

impl<C: ExportBackendConfig> ExportBackendConfig for DebugConfig<C> {
    fn datalayout(&self) -> &str {
        self.inner.datalayout()
    }

    fn emit_llvm_used(&self) -> bool {
        self.inner.emit_llvm_used()
    }

    fn emit_nvvmir_version(&self) -> bool {
        self.inner.emit_nvvmir_version()
    }

    fn nvvmir_version(&self) -> [i32; 4] {
        self.inner.nvvmir_version()
    }

    fn emit_all_kernel_annotations(&self) -> bool {
        self.inner.emit_all_kernel_annotations()
    }

    fn emit_ptx_kernel_keyword(&self) -> bool {
        self.inner.emit_ptx_kernel_keyword()
    }

    fn debug_kind(&self) -> DebugKind {
        self.debug_kind
    }
}

fn src_location(ctx: &mut Context, file: &str, line: i32, column: i32) -> Location {
    Location::SrcPos {
        src: Source::new_from_file(ctx, PathBuf::from(file)),
        pos: SourcePosition { line, column },
    }
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

#[test]
fn export_inline_asm_respects_sideeffect_marker() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&mut ctx, void_ty.to_ptr(), vec![], false);
    let func = FuncOp::new(&mut ctx, "has_inline_asm".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);

    let default_asm = InlineAsmOp::new(&mut ctx, void_ty.into(), vec![], "bar.sync 0;", "", false);
    default_asm.get_operation().insert_at_back(entry, &ctx);

    let register_only_asm = InlineAsmOp::new(&mut ctx, void_ty.into(), vec![], "nop;", "", true);
    llvm_export::ops::set_inline_asm_sideeffect(&mut ctx, register_only_asm.get_operation(), false);
    register_only_asm
        .get_operation()
        .insert_at_back(entry, &ctx);

    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string(&ctx, &module).expect("export succeeds");

    assert!(
        ir.contains("call void asm sideeffect \"bar.sync 0;\", \"\"()"),
        "inline asm without an explicit marker should remain conservative:\n{ir}"
    );
    assert!(
        ir.contains("call void asm \"nop;\", \"\"() #0"),
        "inline asm marked sideeffect=false should omit the keyword while preserving convergent:\n{ir}"
    );
    assert!(
        ir.contains("attributes #0 = { convergent }"),
        "convergent inline asm must emit the convergent attr group:\n{ir}"
    );
}

#[test]
fn export_inline_asm_escapes_llvm_string_literals() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&mut ctx, void_ty.to_ptr(), vec![], false);
    let func = FuncOp::new(
        &mut ctx,
        "has_escaped_inline_asm".try_into().unwrap(),
        func_ty,
    );
    let entry = func.get_or_create_entry_block(&mut ctx);

    let asm = InlineAsmOp::new(
        &mut ctx,
        void_ty.into(),
        vec![],
        "mov.u32 $0, %laneid;\n// \"quoted\" \\22",
        "~{memory}\\raw",
        false,
    );
    asm.get_operation().insert_at_back(entry, &ctx);

    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string(&ctx, &module).expect("export succeeds");

    assert!(
        ir.contains(
            "call void asm sideeffect \"mov.u32 $0, %laneid;\\0A// \\22quoted\\22 \\5C22\", \"~{memory}\\5Craw\"()"
        ),
        "inline asm template and constraints must be escaped as LLVM string literals:\n{ir}"
    );
}

#[test]
fn nvvm_metadata_version_uses_next_allocated_metadata_id() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&mut ctx, void_ty.to_ptr(), vec![], false);
    let func = FuncOp::new(&mut ctx, "bounded_kernel".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);

    let u32_ty = IntegerType::get(&mut ctx, 32, Signedness::Unsigned);
    let width = NonZero::new(32).unwrap();
    let max_threads = IntegerAttr::new(u32_ty, APInt::from_u32(256, width));
    let min_blocks = IntegerAttr::new(u32_ty, APInt::from_u32(2, width));

    {
        let attrs = &mut func.get_operation().deref_mut(&ctx).attributes;
        attrs.set(
            Identifier::try_from("gpu_kernel").unwrap(),
            StringAttr::new("true".into()),
        );
        attrs.set(Identifier::try_from("maxntid").unwrap(), max_threads);
        attrs.set(Identifier::try_from("minctasm").unwrap(), min_blocks);
    }

    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig)
        .expect("NVVM export succeeds");

    assert!(
        ir.contains("!nvvm.annotations = !{!0, !1, !2, !3}"),
        "launch-bounds annotations should occupy !0..!3:\n{ir}"
    );
    assert!(
        ir.contains("!nvvmir.version = !{!4}\n!4 = !{i32 2, i32 0, i32 3, i32 2}"),
        "version metadata should use the next allocated ID:\n{ir}"
    );
}

#[test]
fn line_table_debug_metadata_emits_function_scope_and_instruction_locations() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&mut ctx, void_ty.to_ptr(), vec![], false);
    let func = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 7, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);

    let entry = func.get_or_create_entry_block(&mut ctx);
    let ret = ReturnOp::new(&mut ctx, None);
    let ret_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 8, 5);
    ret.get_operation().deref_mut(&ctx).set_loc(ret_loc);
    ret.get_operation().insert_at_back(entry, &ctx);

    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::LineTables,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    let define_line = ir
        .lines()
        .find(|line| line.starts_with("define "))
        .expect("function definition");
    assert!(
        define_line.contains("!dbg !"),
        "function definition should reference its DISubprogram:\n{ir}"
    );

    let ret_line = ir
        .lines()
        .find(|line| line.trim_start().starts_with("ret void"))
        .expect("return instruction");
    assert!(
        ret_line.contains(", !dbg !"),
        "real instructions should carry DILocation attachments:\n{ir}"
    );

    assert!(
        ir.contains("!llvm.dbg.cu = !{!"),
        "module should reference a compile unit:\n{ir}"
    );
    assert!(
        ir.contains("!llvm.module.flags = !{!"),
        "module should declare debug-info flags:\n{ir}"
    );
    assert!(
        ir.contains("!DIFile(filename: \"kernel.rs\", directory: \"/tmp/cuda-oxide/tests\")"),
        "source path should be split into DIFile filename and directory:\n{ir}"
    );
    assert!(
        ir.contains("distinct !DICompileUnit(language: DW_LANG_Rust"),
        "debug export should describe the Rust compile unit:\n{ir}"
    );
    assert!(
        ir.contains("distinct !DISubprogram(name: \"debug_kernel\""),
        "function definition should get a DISubprogram:\n{ir}"
    );
    assert!(
        ir.contains("!DILocation(line: 8, column: 5, scope: !"),
        "instruction location should preserve the source line and column:\n{ir}"
    );
}

#[test]
fn debug_metadata_shares_allocator_with_nvvm_metadata() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&mut ctx, void_ty.to_ptr(), vec![], false);
    let func = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 10, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);

    {
        let attrs = &mut func.get_operation().deref_mut(&ctx).attributes;
        attrs.set(
            Identifier::try_from("gpu_kernel").unwrap(),
            StringAttr::new("true".into()),
        );
    }

    let entry = func.get_or_create_entry_block(&mut ctx);
    let ret = ReturnOp::new(&mut ctx, None);
    let ret_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 11, 5);
    ret.get_operation().deref_mut(&ctx).set_loc(ret_loc);
    ret.get_operation().insert_at_back(entry, &ctx);

    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: NvvmExportConfig,
        debug_kind: DebugKind::LineTables,
    };
    let ir = export_module_to_string_with_config(&ctx, &module, &config)
        .expect("debug NVVM export succeeds");

    assert!(
        ir.contains("!0 = !DIFile(filename: \"kernel.rs\", directory: \"/tmp/cuda-oxide/tests\")"),
        "debug file node should take the first metadata ID:\n{ir}"
    );
    assert!(
        ir.contains("!4 = !DILocation(line: 11, column: 5, scope: !3)"),
        "instruction location should be allocated before NVVM metadata:\n{ir}"
    );
    assert!(
        ir.contains("!5 = !{ptr @debug_kernel, !\"kernel\", i32 1}"),
        "NVVM annotations should continue after debug metadata:\n{ir}"
    );
    assert!(
        ir.contains("!nvvm.annotations = !{!5}"),
        "named NVVM metadata should reference its allocated node:\n{ir}"
    );
    assert!(
        ir.contains("!nvvmir.version = !{!6}\n!6 = !{i32 2, i32 0, i32 3, i32 2}"),
        "NVVM version should use the next free metadata ID:\n{ir}"
    );
    assert!(
        ir.contains("!llvm.module.flags = !{!7, !8}"),
        "debug module flags should also use the shared allocator:\n{ir}"
    );
}

#[test]
fn line_table_debug_metadata_adds_fallback_locations_to_calls() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let void_ty = VoidType::get(&ctx);
    let helper_ty = FuncType::get(&mut ctx, i32_ty.to_ptr(), vec![], false);
    let helper = FuncOp::new(&mut ctx, "helper".try_into().unwrap(), helper_ty);
    helper.get_operation().insert_at_back(module_block, &ctx);

    let caller_ty = FuncType::get(&mut ctx, void_ty.to_ptr(), vec![], false);
    let caller = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), caller_ty);
    let caller_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 20, 3);
    caller.get_operation().deref_mut(&ctx).set_loc(caller_loc);

    let entry = caller.get_or_create_entry_block(&mut ctx);
    let call = CallOp::new(
        &mut ctx,
        CallOpCallable::Direct("helper".try_into().unwrap()),
        helper_ty,
        vec![],
    );
    call.get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);

    caller.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::LineTables,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    let call_line = ir
        .lines()
        .find(|line| line.contains("call i32 @helper()"))
        .expect("call instruction");
    assert!(
        call_line.contains(", !dbg !"),
        "calls without their own source span should use the function fallback location:\n{ir}"
    );
    assert!(
        ir.contains("!DILocation(line: 20, column: 3, scope: !"),
        "fallback call location should point at the caller's function line:\n{ir}"
    );
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
