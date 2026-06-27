/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use combine::stream::position::SourcePosition;
use llvm_export::{
    export::{
        DebugKind, DeviceExternAttrs, DeviceExternDecl, DeviceExternType, ExportBackendConfig,
        NvvmExportConfig, NvvmIrDialect, PtxExportConfig, export_module_to_string,
        export_module_to_string_with_config, export_module_with_externs,
    },
    op_interfaces::CastOpInterface,
    ops::{
        AddrSpaceCastOp, AddressOfOp, AllocaOp, BitcastOp, BrOp, CallOp, CondBrOp, ConstantOp,
        DebugLocalTypeKind, DebugLocalVariableInfo, DebugSourcePosition, DebugSourceScope,
        DebugSourceScopeLocation, DebugSourceScopeMap, DebugValueOp, FuncOp, GepIndex,
        GetElementPtrOp, GlobalOp, InlineAsmOp, LoadOp, ReturnOp, SelectOp, StoreOp,
    },
    types::{FuncType, PointerType, VoidType},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, StringAttr},
        op_interfaces::CallOpCallable,
        ops::ModuleOp,
        types::{FP32Type, IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
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

    fn nvvm_ir_dialect(&self) -> Option<llvm_export::export::NvvmIrDialect> {
        self.inner.nvvm_ir_dialect()
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

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let ptr_ty = PointerType::get(&ctx, 0);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![ptr_ty.to_handle()], false);
    let func = FuncOp::new(&mut ctx, "volatile_load_test".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let ptr = entry.deref(&ctx).get_argument(0);

    let load = LoadOp::new(&mut ctx, ptr, i32_ty.to_handle());
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

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let ptr_ty = PointerType::get(&ctx, 0);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(
        &ctx,
        void_ty.to_handle(),
        vec![ptr_ty.to_handle(), i32_ty.to_handle()],
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
fn legacy_export_uses_one_canonical_pointer_with_multiple_typed_views() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "legacy_views".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let ptr_ty = PointerType::get(&ctx, 0);
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let f32_ty = FP32Type::get(&ctx);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.into(), vec![ptr_ty.into()], false);
    let func = FuncOp::new(&mut ctx, "multiple_views".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let pointer = entry.deref(&ctx).get_argument(0);

    LoadOp::new(&mut ctx, pointer, i32_ty.into())
        .get_operation()
        .insert_at_back(entry, &ctx);
    LoadOp::new(&mut ctx, pointer, f32_ty.into())
        .get_operation()
        .insert_at_back(entry, &ctx);
    LoadOp::new(&mut ctx, pointer, i8_ty.into())
        .get_operation()
        .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7);
    let ir = export_module_to_string_with_config(&ctx, &module, &config)
        .expect("legacy export succeeds");

    assert!(ir.contains("define void @multiple_views(i8* %v0)"), "{ir}");
    assert!(ir.contains("bitcast i8* %v0 to i32*"), "{ir}");
    assert!(ir.contains("load i32, i32*"), "{ir}");
    assert!(ir.contains("bitcast i8* %v0 to float*"), "{ir}");
    assert!(ir.contains("load float, float*"), "{ir}");
    assert!(ir.contains("load i8, i8* %v0"), "{ir}");
    assert!(!ir.contains("bitcast i8* %v0 to i8*"), "{ir}");
    assert!(
        !ir.split(|c: char| !c.is_ascii_alphanumeric())
            .any(|t| t == "ptr")
    );
}

#[test]
fn legacy_alloca_rejects_a_non_default_result_address_space() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "invalid_alloca_as".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.into(), vec![], false);
    let func = FuncOp::new(&mut ctx, "invalid_alloca".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);

    let one_attr = IntegerAttr::new(i32_ty, APInt::from_u32(1, NonZero::new(32).unwrap()));
    let one = ConstantOp::new(&mut ctx, one_attr.into());
    let one_value = one.get_operation().deref(&ctx).get_result(0);
    one.get_operation().insert_at_back(entry, &ctx);
    let alloca = AllocaOp::new(&mut ctx, i32_ty.into(), one_value);
    let alloca_result = alloca.get_operation().deref(&ctx).get_result(0);
    let shared_pointer = PointerType::get(&ctx, 3);
    alloca_result.set_type(&ctx, shared_pointer.into());
    alloca.get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    module
        .get_operation()
        .deref(&ctx)
        .verify(&ctx)
        .expect("upstream verification currently does not enforce alloca result AS0");
    let error = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("legacy export must reject an alloca address-space mismatch");
    assert!(
        error.contains("alloca result uses address space 3"),
        "{error}"
    );
}

#[test]
fn legacy_gep_rejects_a_result_address_space_different_from_its_base() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "invalid_gep_as".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let global_pointer = PointerType::get(&ctx, 1);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.into(), vec![global_pointer.into()], false);
    let func = FuncOp::new(&mut ctx, "invalid_gep".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let base = entry.deref(&ctx).get_argument(0);
    let gep = GetElementPtrOp::new(&mut ctx, base, vec![GepIndex::Constant(0)], i32_ty.into());
    let gep_result = gep.get_operation().deref(&ctx).get_result(0);
    let shared_pointer = PointerType::get(&ctx, 3);
    gep_result.set_type(&ctx, shared_pointer.into());
    gep.get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    module
        .get_operation()
        .deref(&ctx)
        .verify(&ctx)
        .expect("upstream verification currently does not enforce GEP result/base AS equality");
    let error = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("legacy export must reject a GEP address-space mismatch");
    assert!(
        error.contains("GEP result address-space mismatch: base is 1, result is 3"),
        "{error}"
    );
}

#[test]
fn legacy_pointer_select_keeps_one_canonical_type() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "legacy_pointer_select".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let ptr_ty = PointerType::get(&ctx, 0);
    let func_ty = FuncType::get(
        &ctx,
        ptr_ty.into(),
        vec![i1_ty.into(), ptr_ty.into(), ptr_ty.into()],
        false,
    );
    let func = FuncOp::new(&mut ctx, "choose_pointer".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let condition = entry.deref(&ctx).get_argument(0);
    let if_true = entry.deref(&ctx).get_argument(1);
    let if_false = entry.deref(&ctx).get_argument(2);
    let select = SelectOp::new(&mut ctx, condition, if_true, if_false);
    let selected = select.get_operation().deref(&ctx).get_result(0);
    select.get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, Some(selected))
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect("legacy pointer select export succeeds");
    assert!(
        ir.contains("select i1 %v0, i8* %v1, i8* %v2"),
        "pointer select must use the canonical byte-pointer type:\n{ir}"
    );
    assert!(ir.contains("ret i8*"), "{ir}");
}

#[test]
fn exporter_rejects_extra_predecessor_values_before_emitting_phis() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "invalid_branch_arity".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.into(), vec![], false);
    let func = FuncOp::new(&mut ctx, "invalid_branch".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let region = func.get_operation().deref(&ctx).get_region(0);
    let destination = BasicBlock::new(&mut ctx, None, vec![]);
    destination.insert_at_back(region, &ctx);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let one_attr = IntegerAttr::new(i32_ty, APInt::from_u32(1, NonZero::new(32).unwrap()));
    let one = ConstantOp::new(&mut ctx, one_attr.into());
    let one_value = one.get_operation().deref(&ctx).get_result(0);
    one.get_operation().insert_at_back(entry, &ctx);
    BrOp::new(&mut ctx, destination, vec![one_value])
        .get_operation()
        .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(destination, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let error = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("extra predecessor values must be rejected");
    assert!(
        error.contains("supplies 1 values") && error.contains("expects 0 block arguments"),
        "{error}"
    );
}

#[test]
fn exporter_rejects_distinct_values_on_duplicate_conditional_edges() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "duplicate_conditional_edge".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let func_ty = FuncType::get(
        &ctx,
        i32_ty.into(),
        vec![i1_ty.into(), i32_ty.into(), i32_ty.into()],
        false,
    );
    let func = FuncOp::new(&mut ctx, "duplicate_edge".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let condition = entry.deref(&ctx).get_argument(0);
    let if_true = entry.deref(&ctx).get_argument(1);
    let if_false = entry.deref(&ctx).get_argument(2);
    let region = func.get_operation().deref(&ctx).get_region(0);
    let destination = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    destination.insert_at_back(region, &ctx);
    CondBrOp::new(
        &mut ctx,
        condition,
        destination,
        vec![if_true],
        destination,
        vec![if_false],
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    let result = destination.deref(&ctx).get_argument(0);
    ReturnOp::new(&mut ctx, Some(result))
        .get_operation()
        .insert_at_back(destination, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    module
        .get_operation()
        .deref(&ctx)
        .verify(&ctx)
        .expect("pliron permits same-destination conditional edges with distinct values");
    let error = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("LLVM PHIs cannot distinguish duplicate predecessor edges");
    assert!(
        error.contains("both edges with different forwarded values"),
        "{error}"
    );
}

#[test]
fn exporter_deduplicates_identical_values_on_duplicate_conditional_edges() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "identical_conditional_edge".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let func_ty = FuncType::get(
        &ctx,
        i32_ty.into(),
        vec![i1_ty.into(), i32_ty.into()],
        false,
    );
    let func = FuncOp::new(&mut ctx, "identical_edge".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let condition = entry.deref(&ctx).get_argument(0);
    let value = entry.deref(&ctx).get_argument(1);
    let region = func.get_operation().deref(&ctx).get_region(0);
    let destination = BasicBlock::new(&mut ctx, None, vec![i32_ty.into()]);
    destination.insert_at_back(region, &ctx);
    CondBrOp::new(
        &mut ctx,
        condition,
        destination,
        vec![value],
        destination,
        vec![value],
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    let result = destination.deref(&ctx).get_argument(0);
    ReturnOp::new(&mut ctx, Some(result))
        .get_operation()
        .insert_at_back(destination, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect("identical duplicate-edge values can use one PHI predecessor");
    let phi = ir
        .lines()
        .find(|line| line.contains(" = phi i32 "))
        .expect("destination block must contain a PHI");
    assert_eq!(phi.matches("%entry").count(), 1, "{phi}");
}

#[test]
fn indirect_call_rejects_non_program_address_space() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "invalid_indirect_callee".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let shared_ptr_ty = PointerType::get(&ctx, 3);
    let void_ty = VoidType::get(&ctx);
    let callee_ty = FuncType::get(&ctx, void_ty.into(), vec![], false);
    let caller_ty = FuncType::get(&ctx, void_ty.into(), vec![shared_ptr_ty.into()], false);
    let caller = FuncOp::new(&mut ctx, "caller".try_into().unwrap(), caller_ty);
    let entry = caller.get_or_create_entry_block(&mut ctx);
    let callee = entry.deref(&ctx).get_argument(0);
    CallOp::new(
        &mut ctx,
        CallOpCallable::Indirect(callee),
        callee_ty,
        vec![],
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    caller.get_operation().insert_at_back(module_block, &ctx);

    for dialect in [NvvmIrDialect::LegacyLlvm7, NvvmIrDialect::Modern] {
        let error =
            export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::new(dialect))
                .expect_err("NVPTX must reject a shared-memory function pointer");
        assert!(error.contains("address space 3"), "{error}");
        assert!(error.contains("function pointers"), "{error}");
    }
}

#[test]
fn pointer_bitcast_cannot_cross_address_spaces_in_either_nvvm_dialect() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "invalid_pointer_bitcast".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let global_pointer = PointerType::get(&ctx, 1);
    let shared_pointer = PointerType::get(&ctx, 3);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.into(), vec![global_pointer.into()], false);
    let func = FuncOp::new(&mut ctx, "invalid_cast".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let input = entry.deref(&ctx).get_argument(0);
    BitcastOp::new(&mut ctx, input, shared_pointer.into())
        .get_operation()
        .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    module
        .get_operation()
        .deref(&ctx)
        .verify(&ctx)
        .expect("upstream bitcast verification currently does not enforce pointer AS equality");
    for dialect in [NvvmIrDialect::LegacyLlvm7, NvvmIrDialect::Modern] {
        let error =
            export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::new(dialect))
                .expect_err("a cross-address-space pointer bitcast must be rejected");
        assert!(
            error.contains("pointer bitcast cannot cross address spaces 1 -> 3"),
            "{dialect:?}: {error}"
        );
    }
}

#[test]
fn addrspacecast_must_change_address_spaces_in_either_nvvm_dialect() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "invalid_addrspacecast".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let shared_pointer = PointerType::get(&ctx, 3);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.into(), vec![shared_pointer.into()], false);
    let func = FuncOp::new(&mut ctx, "invalid_cast".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let input = entry.deref(&ctx).get_argument(0);
    AddrSpaceCastOp::new(&mut ctx, input, shared_pointer.into())
        .get_operation()
        .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    module
        .get_operation()
        .deref(&ctx)
        .verify(&ctx)
        .expect("upstream addrspacecast verification currently permits equal address spaces");
    for dialect in [NvvmIrDialect::LegacyLlvm7, NvvmIrDialect::Modern] {
        let error =
            export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::new(dialect))
                .expect_err("addrspacecast must not encode a no-op address-space conversion");
        assert!(
            error.contains(
                "addrspacecast must change address spaces; source and result are both address space 3"
            ),
            "{dialect:?}: {error}"
        );
    }
}

#[test]
fn legacy_function_address_defined_later_round_trips_through_indirect_call() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "function_address".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let void_ty = VoidType::get(&ctx);
    let callee_ty = FuncType::get(&ctx, void_ty.into(), vec![], false);

    // Print the caller first to prove symbol typing is a module pre-pass, not
    // an accidental dependency on textual definition order.
    let caller = FuncOp::new(
        &mut ctx,
        "call_function_pointer".try_into().unwrap(),
        callee_ty,
    );
    let caller_entry = caller.get_or_create_entry_block(&mut ctx);
    let address = AddressOfOp::new(&mut ctx, "target".try_into().unwrap(), 0);
    let address_value = address.get_operation().deref(&ctx).get_result(0);
    address.get_operation().insert_at_back(caller_entry, &ctx);
    CallOp::new(
        &mut ctx,
        CallOpCallable::Indirect(address_value),
        callee_ty,
        vec![],
    )
    .get_operation()
    .insert_at_back(caller_entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(caller_entry, &ctx);
    caller.get_operation().insert_at_back(module_block, &ctx);

    let target = FuncOp::new(&mut ctx, "target".try_into().unwrap(), callee_ty);
    let target_entry = target.get_or_create_entry_block(&mut ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(target_entry, &ctx);
    target.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect("legacy function-address export succeeds");
    assert!(
        ir.contains("bitcast void ()* @target to i8*"),
        "function address must normalize to the canonical byte pointer:\n{ir}"
    );
    assert!(
        ir.contains("bitcast i8*") && ir.contains("to void ()*"),
        "indirect call must restore the exact function pointer type:\n{ir}"
    );
    assert!(ir.contains("call void %"), "{ir}");
}

#[test]
fn modern_function_address_uses_the_normalized_definition_name() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "modern_function_address".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let void_ty = VoidType::get(&ctx);
    let callee_ty = FuncType::get(&ctx, void_ty.into(), vec![], false);
    let prefixed_name = "cuda_oxide_device_246e25db_target";

    let caller = FuncOp::new(&mut ctx, "caller".try_into().unwrap(), callee_ty);
    let caller_entry = caller.get_or_create_entry_block(&mut ctx);
    let address = AddressOfOp::new(&mut ctx, prefixed_name.try_into().unwrap(), 0);
    let address_value = address.get_operation().deref(&ctx).get_result(0);
    address.get_operation().insert_at_back(caller_entry, &ctx);
    CallOp::new(
        &mut ctx,
        CallOpCallable::Indirect(address_value),
        callee_ty,
        vec![],
    )
    .get_operation()
    .insert_at_back(caller_entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(caller_entry, &ctx);
    caller.get_operation().insert_at_back(module_block, &ctx);

    let target = FuncOp::new(&mut ctx, prefixed_name.try_into().unwrap(), callee_ty);
    let target_entry = target.get_or_create_entry_block(&mut ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(target_entry, &ctx);
    target.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::Modern),
    )
    .expect("modern function-address export succeeds");
    assert!(ir.contains("define void @target()"), "{ir}");
    assert!(ir.contains("call void @target()"), "{ir}");
    assert!(!ir.contains(prefixed_name), "{ir}");
}

#[test]
fn modern_addressof_rejects_global_and_function_address_space_mismatches() {
    let mut ctx = Context::new();
    let void_ty = VoidType::get(&ctx);
    let no_args = FuncType::get(&ctx, void_ty.into(), vec![], false);

    let global_module = ModuleOp::new(&mut ctx, "bad_global_address".try_into().unwrap());
    let global_module_block = module_top_block(&mut ctx, &global_module);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let global = GlobalOp::new(&mut ctx, "shared_value".try_into().unwrap(), i32_ty.into());
    global.set_address_space(&mut ctx, 3);
    global
        .get_operation()
        .insert_at_back(global_module_block, &ctx);
    let global_user = FuncOp::new(&mut ctx, "global_user".try_into().unwrap(), no_args);
    let global_entry = global_user.get_or_create_entry_block(&mut ctx);
    AddressOfOp::new(&mut ctx, "shared_value".try_into().unwrap(), 0)
        .get_operation()
        .insert_at_back(global_entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(global_entry, &ctx);
    global_user
        .get_operation()
        .insert_at_back(global_module_block, &ctx);
    let error = export_module_to_string_with_config(
        &ctx,
        &global_module,
        &NvvmExportConfig::new(NvvmIrDialect::Modern),
    )
    .expect_err("modern global addressof must preserve address spaces");
    assert!(
        error.contains("result is 0, global is 3"),
        "unexpected global error: {error}"
    );

    let function_module = ModuleOp::new(&mut ctx, "bad_function_address".try_into().unwrap());
    let function_module_block = module_top_block(&mut ctx, &function_module);
    FuncOp::new(&mut ctx, "target".try_into().unwrap(), no_args)
        .get_operation()
        .insert_at_back(function_module_block, &ctx);
    let function_user = FuncOp::new(&mut ctx, "function_user".try_into().unwrap(), no_args);
    let function_entry = function_user.get_or_create_entry_block(&mut ctx);
    AddressOfOp::new(&mut ctx, "target".try_into().unwrap(), 3)
        .get_operation()
        .insert_at_back(function_entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(function_entry, &ctx);
    function_user
        .get_operation()
        .insert_at_back(function_module_block, &ctx);
    let error = export_module_to_string_with_config(
        &ctx,
        &function_module,
        &NvvmExportConfig::new(NvvmIrDialect::Modern),
    )
    .expect_err("modern function addressof must use program address space");
    assert!(error.contains("program-address-space (0)"), "{error}");
}

#[test]
fn legacy_device_extern_adapts_exact_pointer_arguments_and_results() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "legacy_extern".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let ptr_ty = PointerType::get(&ctx, 0);
    let external_ty = FuncType::get(&ctx, ptr_ty.into(), vec![ptr_ty.into()], false);
    FuncOp::new(&mut ctx, "float_roundtrip".try_into().unwrap(), external_ty)
        .get_operation()
        .insert_at_back(module_block, &ctx);

    let void_ty = VoidType::get(&ctx);
    let caller_ty = FuncType::get(&ctx, void_ty.into(), vec![ptr_ty.into()], false);
    let caller = FuncOp::new(&mut ctx, "caller".try_into().unwrap(), caller_ty);
    let entry = caller.get_or_create_entry_block(&mut ctx);
    let pointer = entry.deref(&ctx).get_argument(0);
    CallOp::new(
        &mut ctx,
        CallOpCallable::Direct("float_roundtrip".try_into().unwrap()),
        external_ty,
        vec![pointer],
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    caller.get_operation().insert_at_back(module_block, &ctx);

    let externs = [DeviceExternDecl {
        export_name: "float_roundtrip".to_string(),
        param_types: vec![DeviceExternType::pointer_to(DeviceExternType::Float32, 0)],
        return_type: DeviceExternType::pointer_to(DeviceExternType::Float32, 0),
        attrs: DeviceExternAttrs::default(),
    }];
    let ir = export_module_with_externs(
        &ctx,
        &module,
        &externs,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect("legacy extern export succeeds");

    assert!(
        ir.contains("declare float* @float_roundtrip(float*)"),
        "{ir}"
    );
    assert!(ir.contains("bitcast i8* %v0 to float*"), "{ir}");
    assert!(ir.contains("call float* @float_roundtrip(float*"), "{ir}");
    assert!(
        ir.contains(" = bitcast float* ") && ir.contains(" to i8*"),
        "{ir}"
    );
    assert!(
        !ir.split(|c: char| !c.is_ascii_alphanumeric())
            .any(|token| token == "ptr"),
        "{ir}"
    );
}

#[test]
fn legacy_device_extern_preserves_pointer_address_spaces() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "legacy_extern_as".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let ptr_ty = PointerType::get(&ctx, 3);
    let void_ty = VoidType::get(&ctx);
    let external_ty = FuncType::get(&ctx, void_ty.into(), vec![ptr_ty.into()], false);
    FuncOp::new(&mut ctx, "shared_float".try_into().unwrap(), external_ty)
        .get_operation()
        .insert_at_back(module_block, &ctx);
    let caller = FuncOp::new(&mut ctx, "caller".try_into().unwrap(), external_ty);
    let entry = caller.get_or_create_entry_block(&mut ctx);
    let pointer = entry.deref(&ctx).get_argument(0);
    CallOp::new(
        &mut ctx,
        CallOpCallable::Direct("shared_float".try_into().unwrap()),
        external_ty,
        vec![pointer],
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    caller.get_operation().insert_at_back(module_block, &ctx);

    let externs = [DeviceExternDecl {
        export_name: "shared_float".to_string(),
        param_types: vec![DeviceExternType::pointer_to(DeviceExternType::Float32, 3)],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];
    let ir = export_module_with_externs(
        &ctx,
        &module,
        &externs,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect("legacy extern export succeeds");
    assert!(
        ir.contains("declare void @shared_float(float addrspace(3)*)"),
        "{ir}"
    );
    assert!(
        ir.contains("bitcast i8 addrspace(3)* %v0 to float addrspace(3)*"),
        "{ir}"
    );
}

#[test]
fn modern_device_extern_erases_pointee_without_boundary_casts() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "modern_extern".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let ptr_ty = PointerType::get(&ctx, 0);
    let void_ty = VoidType::get(&ctx);
    let external_ty = FuncType::get(&ctx, void_ty.into(), vec![ptr_ty.into()], false);
    FuncOp::new(&mut ctx, "takes_float".try_into().unwrap(), external_ty)
        .get_operation()
        .insert_at_back(module_block, &ctx);
    let caller = FuncOp::new(&mut ctx, "caller".try_into().unwrap(), external_ty);
    let entry = caller.get_or_create_entry_block(&mut ctx);
    let pointer = entry.deref(&ctx).get_argument(0);
    CallOp::new(
        &mut ctx,
        CallOpCallable::Direct("takes_float".try_into().unwrap()),
        external_ty,
        vec![pointer],
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    caller.get_operation().insert_at_back(module_block, &ctx);

    let externs = [DeviceExternDecl {
        export_name: "takes_float".to_string(),
        param_types: vec![DeviceExternType::pointer_to(DeviceExternType::Float32, 0)],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];
    let ir = export_module_with_externs(
        &ctx,
        &module,
        &externs,
        &NvvmExportConfig::new(NvvmIrDialect::Modern),
    )
    .expect("modern extern export succeeds");
    assert!(ir.contains("declare void @takes_float(ptr)"), "{ir}");
    assert!(ir.contains("call void @takes_float(ptr %v0)"), "{ir}");
    assert!(!ir.contains("bitcast"), "{ir}");
}

#[test]
fn device_extern_rejects_invalid_symbol_and_address_space_mismatch() {
    let mut ctx = Context::new();
    let empty = ModuleOp::new(&mut ctx, "empty".try_into().unwrap());
    let invalid = [DeviceExternDecl {
        export_name: "bad.name".to_string(),
        param_types: vec![],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];
    let err = export_module_with_externs(
        &ctx,
        &empty,
        &invalid,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("invalid NVVM symbol must fail");
    assert!(err.contains("global-identifier subset"), "{err}");

    let reserved_intrinsic_prefix = [DeviceExternDecl {
        export_name: "llvm_external".to_string(),
        param_types: vec![],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];
    let err = export_module_with_externs(
        &ctx,
        &empty,
        &reserved_intrinsic_prefix,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("the reserved intrinsic namespace must not be ambiguous");
    assert!(err.contains("reserves for LLVM intrinsics"), "{err}");

    let by_value_array = [DeviceExternDecl {
        export_name: "array_by_value".to_string(),
        param_types: vec![DeviceExternType::Array {
            element: Box::new(DeviceExternType::Float32),
            len: 4,
        }],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];
    let err = export_module_with_externs(
        &ctx,
        &empty,
        &by_value_array,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("by-value aggregate externs must be rejected");
    assert!(err.contains("passes an array by value"), "{err}");

    let nested_half = [DeviceExternDecl {
        export_name: "half_buffer".to_string(),
        param_types: vec![DeviceExternType::pointer_to(
            DeviceExternType::Array {
                element: Box::new(DeviceExternType::Float16),
                len: 4,
            },
            0,
        )],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];
    let err = export_module_with_externs(
        &ctx,
        &empty,
        &nested_half,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("legacy half nested in a pointer must fail");
    assert!(
        err.contains("CUDA 12 legacy") && err.contains("half"),
        "{err}"
    );
    let modern = export_module_with_externs(
        &ctx,
        &empty,
        &nested_half,
        &NvvmExportConfig::new(NvvmIrDialect::Modern),
    )
    .expect("modern opaque-pointer extern may use half pointees");
    assert!(
        modern.contains("declare void @half_buffer(ptr)"),
        "{modern}"
    );

    let module = ModuleOp::new(&mut ctx, "mismatch".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let ptr0 = PointerType::get(&ctx, 0);
    let void_ty = VoidType::get(&ctx);
    let external_ty = FuncType::get(&ctx, void_ty.into(), vec![ptr0.into()], false);
    FuncOp::new(&mut ctx, "shared_only".try_into().unwrap(), external_ty)
        .get_operation()
        .insert_at_back(module_block, &ctx);
    let caller = FuncOp::new(&mut ctx, "caller".try_into().unwrap(), external_ty);
    let entry = caller.get_or_create_entry_block(&mut ctx);
    let pointer = entry.deref(&ctx).get_argument(0);
    CallOp::new(
        &mut ctx,
        CallOpCallable::Direct("shared_only".try_into().unwrap()),
        external_ty,
        vec![pointer],
    )
    .get_operation()
    .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    caller.get_operation().insert_at_back(module_block, &ctx);
    let mismatch = [DeviceExternDecl {
        export_name: "shared_only".to_string(),
        param_types: vec![DeviceExternType::pointer_to(DeviceExternType::Float32, 3)],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];
    let err = export_module_with_externs(
        &ctx,
        &module,
        &mismatch,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("address-space mismatch must fail");
    assert!(
        err.contains("parameter, result, or pointer address-space types"),
        "{err}"
    );
}

#[test]
fn device_extern_rejects_same_name_declaration_shape_without_a_call() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "extern_decl_conflict".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let void_ty = VoidType::get(&ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let lowered_type = FuncType::get(&ctx, void_ty.into(), vec![i32_ty.into()], false);
    FuncOp::new(
        &mut ctx,
        "conflicting_decl".try_into().unwrap(),
        lowered_type,
    )
    .get_operation()
    .insert_at_back(module_block, &ctx);

    let externs = [DeviceExternDecl {
        export_name: "conflicting_decl".to_string(),
        param_types: vec![DeviceExternType::Float32],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];
    let error = export_module_with_externs(
        &ctx,
        &module,
        &externs,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("the exporter must independently reject a conflicting declaration");
    assert!(
        error.contains("parameter, result, or pointer address-space types"),
        "{error}"
    );
}

#[test]
fn device_extern_rejects_definition_and_address_taken_shape_conflicts() {
    let mut ctx = Context::new();
    let void_ty = VoidType::get(&ctx);

    let definition_module = ModuleOp::new(&mut ctx, "extern_definition".try_into().unwrap());
    let definition_block = module_top_block(&mut ctx, &definition_module);
    let no_args = FuncType::get(&ctx, void_ty.into(), vec![], false);
    let definition = FuncOp::new(&mut ctx, "defined_external".try_into().unwrap(), no_args);
    let definition_entry = definition.get_or_create_entry_block(&mut ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(definition_entry, &ctx);
    definition
        .get_operation()
        .insert_at_back(definition_block, &ctx);
    let definition_extern = [DeviceExternDecl {
        export_name: "defined_external".to_string(),
        param_types: vec![],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];
    let error = export_module_with_externs(
        &ctx,
        &definition_module,
        &definition_extern,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("a side-table extern must not collide with a definition");
    assert!(error.contains("function definition"), "{error}");

    let address_module = ModuleOp::new(&mut ctx, "extern_address".try_into().unwrap());
    let address_block = module_top_block(&mut ctx, &address_module);
    let generic_pointer = PointerType::get(&ctx, 0);
    let lowered_type = FuncType::get(&ctx, void_ty.into(), vec![generic_pointer.into()], false);
    FuncOp::new(
        &mut ctx,
        "addressed_external".try_into().unwrap(),
        lowered_type,
    )
    .get_operation()
    .insert_at_back(address_block, &ctx);
    let caller = FuncOp::new(&mut ctx, "caller".try_into().unwrap(), lowered_type);
    let caller_entry = caller.get_or_create_entry_block(&mut ctx);
    let argument = caller_entry.deref(&ctx).get_argument(0);
    let address = AddressOfOp::new(&mut ctx, "addressed_external".try_into().unwrap(), 0);
    let address_value = address.get_operation().deref(&ctx).get_result(0);
    address.get_operation().insert_at_back(caller_entry, &ctx);
    CallOp::new(
        &mut ctx,
        CallOpCallable::Indirect(address_value),
        lowered_type,
        vec![argument],
    )
    .get_operation()
    .insert_at_back(caller_entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(caller_entry, &ctx);
    caller.get_operation().insert_at_back(address_block, &ctx);
    let address_extern = [DeviceExternDecl {
        export_name: "addressed_external".to_string(),
        param_types: vec![DeviceExternType::pointer_to(DeviceExternType::Float32, 3)],
        return_type: DeviceExternType::Void,
        attrs: DeviceExternAttrs::default(),
    }];
    let error = export_module_with_externs(
        &ctx,
        &address_module,
        &address_extern,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("address-taking must not bypass exact extern shape validation");
    assert!(
        error.contains("parameter, result, or pointer address-space types"),
        "{error}"
    );
}

#[test]
fn legacy_kernel_metadata_uses_typed_function_references() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "legacy_metadata".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let ptr_ty = PointerType::get(&ctx, 0);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.into(), vec![ptr_ty.into()], false);
    let func = FuncOp::new(&mut ctx, "metadata_kernel".try_into().unwrap(), func_ty);
    func.get_operation().deref_mut(&ctx).attributes.set(
        "gpu_kernel".try_into().unwrap(),
        StringAttr::new("true".into()),
    );
    let entry = func.get_or_create_entry_block(&mut ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7);
    let ir = export_module_to_string_with_config(&ctx, &module, &config)
        .expect("legacy metadata export succeeds");
    assert!(
        ir.contains(
            "@llvm.used = appending global [1 x i8*] [i8* bitcast (void (i8*)* @metadata_kernel to i8*)]"
        ),
        "{ir}"
    );
    assert!(
        ir.contains("!{void (i8*)* @metadata_kernel, !\"kernel\", i32 1}"),
        "{ir}"
    );
}

#[test]
fn legacy_export_rejects_debug_metadata() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "legacy_debug".try_into().unwrap());
    let config = DebugConfig {
        inner: NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
        debug_kind: DebugKind::LineTables,
    };

    let error = export_module_to_string_with_config(&ctx, &module, &config)
        .expect_err("legacy debug output must be rejected");
    assert!(error.contains("legacy LLVM 7"), "{error}");
    assert!(error.contains("debug"), "{error}");
}

#[test]
fn legacy_pointer_slot_is_recursively_canonical() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "legacy_pointer_slot".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let ptr_ty = PointerType::get(&ctx, 0);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.into(), vec![ptr_ty.into()], false);
    let func = FuncOp::new(&mut ctx, "pointer_slot".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let incoming = entry.deref(&ctx).get_argument(0);

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let one_attr = IntegerAttr::new(i32_ty, APInt::from_u32(1, NonZero::new(32).unwrap()));
    let one = ConstantOp::new(&mut ctx, one_attr.into());
    let one_value = one.get_operation().deref(&ctx).get_result(0);
    one.get_operation().insert_at_back(entry, &ctx);

    let slot = AllocaOp::new(&mut ctx, ptr_ty.into(), one_value);
    let slot_value = slot.get_operation().deref(&ctx).get_result(0);
    slot.get_operation().insert_at_back(entry, &ctx);
    StoreOp::new(&mut ctx, incoming, slot_value)
        .get_operation()
        .insert_at_back(entry, &ctx);
    LoadOp::new(&mut ctx, slot_value, ptr_ty.into())
        .get_operation()
        .insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect("legacy pointer-slot export succeeds");
    assert!(ir.contains("alloca i8*"), "{ir}");
    assert!(ir.contains("bitcast i8**"), "{ir}");
    assert!(ir.matches("to i8**").count() >= 2, "{ir}");
    assert!(ir.contains("store i8* %v0, i8**"), "{ir}");
    assert!(ir.contains("load i8*, i8**"), "{ir}");
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

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let global = GlobalOp::new(
        &mut ctx,
        "__shared_mem_20".try_into().unwrap(),
        i32_ty.to_handle(),
    );
    global.set_address_space(&mut ctx, 3);
    global.get_operation().insert_at_back(module_block, &ctx);

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
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
        i32_ty.to_handle(),
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

    let legacy = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect("legacy addressof export succeeds");
    assert!(
        legacy.contains("@__shared_mem_20 = addrspace(3) global i32 undef"),
        "NVVM shared globals must be uninitialized:\n{legacy}"
    );
    assert!(
        legacy.contains("bitcast i32 addrspace(3)* @__shared_mem_20 to i8 addrspace(3)*"),
        "legacy addressof must normalize the global pointer:\n{legacy}"
    );
    assert!(
        legacy.contains("bitcast i8 addrspace(3)*") && legacy.contains("to i32 addrspace(3)*"),
        "legacy GEP must repair its canonical base pointer:\n{legacy}"
    );
    assert_no_undefined_temporaries(&legacy);
}

#[test]
fn nvvm_export_rejects_invalid_global_address_spaces() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let global = GlobalOp::new(
        &mut ctx,
        "thread_local_global".try_into().unwrap(),
        i32_ty.to_handle(),
    );
    global.set_address_space(&mut ctx, 5);
    global.get_operation().insert_at_back(module_block, &ctx);

    let error = export_module_to_string_with_config(
        &ctx,
        &module,
        &NvvmExportConfig::new(NvvmIrDialect::LegacyLlvm7),
    )
    .expect_err("NVVM module-scope local-memory global must be rejected");
    assert!(error.contains("unsupported address space 5"), "{error}");

    // The ordinary LLVM/PTX exporter retains its prior behavior; this
    // restriction is specifically part of the NVVM IR contract.
    assert!(export_module_to_string(&ctx, &module).is_ok());
}

#[test]
fn export_inline_asm_respects_sideeffect_marker() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
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
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
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
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "bounded_kernel".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);

    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
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

    let ir = export_module_to_string_with_config(&ctx, &module, &NvvmExportConfig::default())
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
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
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
fn export_alwaysinline_function_attribute_uses_llvm_define_syntax() {
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "inline_helper".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);

    let key: pliron::identifier::Identifier = "alwaysinline".try_into().unwrap();
    func.get_operation()
        .deref_mut(&ctx)
        .attributes
        .set(key, StringAttr::new("true".to_string()));
    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string(&ctx, &module).expect("export succeeds");
    let define_line = ir
        .lines()
        .find(|line| line.starts_with("define void @inline_helper("))
        .expect("inline helper definition");
    assert_eq!(
        define_line, "define void @inline_helper() alwaysinline #0 {",
        "`alwaysinline` must be emitted after the parameter list, before attr group #0:\n{ir}"
    );
    assert!(
        ir.contains("attributes #0 = { convergent }"),
        "convergent attribute group must still be emitted:\n{ir}"
    );
}

#[test]
fn export_alwaysinline_coexists_with_debug_scope() {
    // alwaysinline and the !dbg scope are emitted on the same define line and
    // must not crowd each other out. This guards the 4-way emission: a future
    // change that drops either one when both are present fails here.
    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_block = module_top_block(&mut ctx, &module);

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "inline_helper".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 7, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let ret = ReturnOp::new(&mut ctx, None);
    let ret_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 8, 5);
    ret.get_operation().deref_mut(&ctx).set_loc(ret_loc);
    ret.get_operation().insert_at_back(entry, &ctx);

    let key: pliron::identifier::Identifier = "alwaysinline".try_into().unwrap();
    func.get_operation()
        .deref_mut(&ctx)
        .attributes
        .set(key, StringAttr::new("true".to_string()));
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::LineTables,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");
    let define_line = ir
        .lines()
        .find(|line| line.starts_with("define void @inline_helper("))
        .expect("inline helper definition");
    assert!(
        define_line.contains("alwaysinline"),
        "alwaysinline must survive when debug info is on:\n{ir}"
    );
    assert!(
        define_line.contains("!dbg !"),
        "!dbg scope must survive when alwaysinline is present:\n{ir}"
    );
}

#[test]
fn line_table_debug_metadata_uses_file_scope_for_cross_file_locations() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 38, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);

    let entry = func.get_or_create_entry_block(&mut ctx);
    let ret = ReturnOp::new(&mut ctx, None);
    let ret_loc = src_location(
        &mut ctx,
        "/tmp/cuda-oxide/crates/cuda-device/src/thread.rs",
        292,
        19,
    );
    ret.get_operation().deref_mut(&ctx).set_loc(ret_loc);
    ret.get_operation().insert_at_back(entry, &ctx);

    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::LineTables,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    let ret_line = ir
        .lines()
        .find(|line| line.trim_start().starts_with("ret void"))
        .expect("return instruction");
    assert!(
        ret_line.contains(", !dbg !"),
        "cross-file instructions should keep source locations:\n{ir}"
    );
    assert!(
        ir.contains(
            "!DIFile(filename: \"thread.rs\", directory: \"/tmp/cuda-oxide/crates/cuda-device/src\")"
        ),
        "cross-file locations should get their own DIFile:\n{ir}"
    );
    assert!(
        ir.contains("!DILexicalBlockFile(scope: !"),
        "cross-file locations should use a file-specific debug scope:\n{ir}"
    );
    assert!(
        ir.contains("!DILocation(line: 292, column: 19, scope: !"),
        "cross-file locations should preserve their real source line:\n{ir}"
    );
}

#[test]
fn line_table_debug_metadata_emits_inlined_at_for_callsite_locations() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 38, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);

    let entry = func.get_or_create_entry_block(&mut ctx);
    let ret = ReturnOp::new(&mut ctx, None);
    let callee = src_location(
        &mut ctx,
        "/tmp/cuda-oxide/crates/cuda-device/src/thread.rs",
        292,
        19,
    );
    let caller = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 39, 13);
    ret.get_operation()
        .deref_mut(&ctx)
        .set_loc(Location::CallSite {
            callee: Box::new(callee),
            caller: Box::new(caller),
        });
    ret.get_operation().insert_at_back(entry, &ctx);

    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::LineTables,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    assert!(
        ir.contains("!DILocation(line: 39, column: 13, scope: !"),
        "callsite metadata should preserve the caller location:\n{ir}"
    );
    assert!(
        ir.contains("!DILocation(line: 292, column: 19, scope: !") && ir.contains(", inlinedAt: !"),
        "callsite metadata should describe the callee location as inlined at the caller:\n{ir}"
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
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
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
        inner: NvvmExportConfig::default(),
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
fn debug_locations_use_rustc_source_scope_positions() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 10, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);
    llvm_export::ops::set_debug_source_scope_map(
        &mut ctx,
        func.get_operation(),
        &DebugSourceScopeMap {
            scopes: vec![
                DebugSourceScope {
                    id: 0,
                    parent: None,
                    span: Some(DebugSourcePosition {
                        file: PathBuf::from("/tmp/cuda-oxide/tests/kernel.rs"),
                        line: 10,
                        column: 1,
                    }),
                    inlined: None,
                },
                DebugSourceScope {
                    id: 1,
                    parent: Some(0),
                    span: Some(DebugSourcePosition {
                        file: PathBuf::from("/tmp/cuda-oxide/tests/kernel.rs"),
                        line: 12,
                        column: 9,
                    }),
                    inlined: None,
                },
            ],
            locations: vec![DebugSourceScopeLocation {
                pos: DebugSourcePosition {
                    file: PathBuf::from("/tmp/cuda-oxide/tests/kernel.rs"),
                    line: 12,
                    column: 9,
                },
                scope: 1,
            }],
        },
    );

    let entry = func.get_or_create_entry_block(&mut ctx);
    let ret = ReturnOp::new(&mut ctx, None);
    let ret_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 12, 9);
    ret.get_operation().deref_mut(&ctx).set_loc(ret_loc);
    ret.get_operation().insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::LineTables,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    let block_id = ir
        .lines()
        .find_map(|line| {
            if line.contains("!DILexicalBlock(scope: !") && line.contains("line: 12, column: 9") {
                line.split_once(" = ")
                    .map(|(id, _)| id.trim_start_matches('!').to_string())
            } else {
                None
            }
        })
        .expect("nested lexical block should be emitted");

    assert!(
        ir.contains(&format!(
            "!DILocation(line: 12, column: 9, scope: !{block_id})"
        )),
        "instruction location should use the exact rustc source scope, not the function scope:\n{ir}"
    );
}

#[test]
fn full_debug_metadata_emits_dbg_declare_for_tagged_allocas() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 30, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);

    let entry = func.get_or_create_entry_block(&mut ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let one_attr = IntegerAttr::new(i32_ty, APInt::from_u32(1, NonZero::new(32).unwrap()));
    let one = ConstantOp::new(&mut ctx, one_attr.into());
    one.get_operation().insert_at_back(entry, &ctx);
    let one_val = one.get_operation().deref(&ctx).get_result(0);

    let tid = AllocaOp::new(&mut ctx, i32_ty.into(), one_val);
    let tid_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 31, 9);
    tid.get_operation().deref_mut(&ctx).set_loc(tid_loc);
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        tid.get_operation(),
        DebugLocalVariableInfo {
            name: "tid".to_string(),
            argument_index: Some(1),
            ty: DebugLocalTypeKind::Basic {
                name: "u32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_unsigned",
            },
        },
    );
    tid.get_operation().insert_at_back(entry, &ctx);

    let ptr_ty = PointerType::get(&ctx, 0);
    let ptr = AllocaOp::new(&mut ctx, ptr_ty.into(), one_val);
    let ptr_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 32, 9);
    ptr.get_operation().deref_mut(&ctx).set_loc(ptr_loc);
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        ptr.get_operation(),
        DebugLocalVariableInfo {
            name: "ptr".to_string(),
            argument_index: None,
            ty: DebugLocalTypeKind::Pointer {
                name: "*mut f32".to_string(),
                size_bits: 64,
            },
        },
    );
    ptr.get_operation().insert_at_back(entry, &ctx);

    let ret = ReturnOp::new(&mut ctx, None);
    let ret_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 33, 1);
    ret.get_operation().deref_mut(&ctx).set_loc(ret_loc);
    ret.get_operation().insert_at_back(entry, &ctx);

    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::Full,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    assert!(
        ir.contains("emissionKind: FullDebug"),
        "full debug should request full DWARF metadata:\n{ir}"
    );
    assert!(
        ir.contains("isOptimized: false"),
        "full debug export should describe the unoptimized debug path:\n{ir}"
    );
    assert!(
        ir.contains("declare void @llvm.dbg.declare(metadata, metadata, metadata)"),
        "full debug should declare the debug intrinsic it calls:\n{ir}"
    );
    assert!(
        ir.contains("call void @llvm.dbg.declare(metadata ptr %"),
        "tagged allocas should be bound to variables with dbg.declare:\n{ir}"
    );
    assert!(
        ir.contains("!DILocalVariable(name: \"tid\", arg: 1, scope: !"),
        "argument debug metadata should preserve the argument number:\n{ir}"
    );
    assert!(
        ir.contains("!DILocalVariable(name: \"ptr\", scope: !"),
        "local debug metadata should omit the arg field:\n{ir}"
    );
    assert!(
        ir.contains("!DIBasicType(name: \"u32\", size: 32, encoding: DW_ATE_unsigned)"),
        "basic integer variables should get DIBasicType metadata:\n{ir}"
    );
    assert!(
        ir.contains(
            "!DIDerivedType(tag: DW_TAG_pointer_type, name: \"*mut f32\", baseType: null, size: 64)"
        ),
        "pointer variables should get a pointer DIType:\n{ir}"
    );
}

#[test]
fn full_debug_metadata_uses_file_scope_for_cross_file_local_variables() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 30, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);

    let entry = func.get_or_create_entry_block(&mut ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let one_attr = IntegerAttr::new(i32_ty, APInt::from_u32(1, NonZero::new(32).unwrap()));
    let one = ConstantOp::new(&mut ctx, one_attr.into());
    one.get_operation().insert_at_back(entry, &ctx);
    let one_val = one.get_operation().deref(&ctx).get_result(0);

    let tid = AllocaOp::new(&mut ctx, i32_ty.into(), one_val);
    let tid_loc = src_location(
        &mut ctx,
        "/tmp/cuda-oxide/crates/cuda-device/src/thread.rs",
        292,
        19,
    );
    tid.get_operation().deref_mut(&ctx).set_loc(tid_loc);
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        tid.get_operation(),
        DebugLocalVariableInfo {
            name: "tid".to_string(),
            argument_index: None,
            ty: DebugLocalTypeKind::Basic {
                name: "u32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_unsigned",
            },
        },
    );
    tid.get_operation().insert_at_back(entry, &ctx);

    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::Full,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    assert!(
        ir.contains("!DILexicalBlockFile(scope: !"),
        "cross-file local variables should get a file-specific debug scope:\n{ir}"
    );
    assert!(
        ir.contains(
            "!DIFile(filename: \"thread.rs\", directory: \"/tmp/cuda-oxide/crates/cuda-device/src\")"
        ),
        "cross-file local variables should reference their source file:\n{ir}"
    );
    assert!(
        ir.contains("!DILocalVariable(name: \"tid\", scope: !") && ir.contains("line: 292"),
        "cross-file local variables should preserve the variable file scope and line:\n{ir}"
    );
    assert!(
        ir.contains("call void @llvm.dbg.declare"),
        "cross-file local variables should still get dbg.declare bindings:\n{ir}"
    );
}

#[test]
fn full_debug_metadata_emits_dbg_value_for_promoted_locals() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![i32_ty.into()], false);
    let func = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 30, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);

    let entry = func.get_or_create_entry_block(&mut ctx);
    let arg = entry.deref(&ctx).get_argument(0);
    let dbg_value = DebugValueOp::new(&mut ctx, arg);
    let dbg_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 31, 13);
    dbg_value.get_operation().deref_mut(&ctx).set_loc(dbg_loc);
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        dbg_value.get_operation(),
        DebugLocalVariableInfo {
            name: "x".to_string(),
            argument_index: Some(1),
            ty: DebugLocalTypeKind::Basic {
                name: "i32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_signed",
            },
        },
    );
    llvm_export::ops::set_debug_local_declaration_location(
        &mut ctx,
        dbg_value.get_operation(),
        PathBuf::from("/tmp/cuda-oxide/tests/declarations.rs"),
        12,
        5,
    );
    dbg_value.get_operation().insert_at_back(entry, &ctx);

    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::Full,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    assert!(
        ir.contains("declare void @llvm.dbg.value(metadata, metadata, metadata)"),
        "full debug should declare dbg.value when it emits one:\n{ir}"
    );
    assert!(
        ir.contains("call void @llvm.dbg.value(metadata i32 %v0, metadata !"),
        "dbg.value should describe the local as the current SSA value:\n{ir}"
    );
    assert!(
        ir.contains("!DILocalVariable(name: \"x\", arg: 1, scope: !"),
        "dbg.value should preserve formal-argument metadata when the source local is an argument:\n{ir}"
    );
    assert!(
        ir.contains("!DILocalVariable(name: \"x\", arg: 1, scope: !")
            && ir.contains("file: !")
            && ir.contains("line: 12"),
        "DILocalVariable should use the source declaration line, not the dbg.value line:\n{ir}"
    );
    assert!(
        ir.contains("!DILocation(line: 31, column: 13, scope: !"),
        "dbg.value should still be located at the value's current source point:\n{ir}"
    );
    assert!(
        !ir.contains("llvm.dbg.declare"),
        "a value-only debug record should not force dbg.declare:\n{ir}"
    );
}

#[test]
fn full_debug_metadata_uses_inlined_callee_scope_for_inlined_arguments() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![i32_ty.into()], false);
    let func = FuncOp::new(&mut ctx, "caller_kernel".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 30, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);
    llvm_export::ops::set_debug_source_scope_map(
        &mut ctx,
        func.get_operation(),
        &DebugSourceScopeMap {
            scopes: vec![
                DebugSourceScope {
                    id: 0,
                    parent: None,
                    span: Some(DebugSourcePosition {
                        file: PathBuf::from("/tmp/cuda-oxide/tests/kernel.rs"),
                        line: 30,
                        column: 1,
                    }),
                    inlined: None,
                },
                DebugSourceScope {
                    id: 1,
                    parent: Some(0),
                    span: Some(DebugSourcePosition {
                        file: PathBuf::from("/tmp/cuda-oxide/tests/helper.rs"),
                        line: 7,
                        column: 1,
                    }),
                    inlined: Some(llvm_export::ops::DebugInlinedScope {
                        callee_name: "helper::next".to_string(),
                        callsite: Some(DebugSourcePosition {
                            file: PathBuf::from("/tmp/cuda-oxide/tests/kernel.rs"),
                            line: 41,
                            column: 13,
                        }),
                    }),
                },
            ],
            locations: vec![],
        },
    );

    let entry = func.get_or_create_entry_block(&mut ctx);
    let arg = entry.deref(&ctx).get_argument(0);

    let caller_value = DebugValueOp::new(&mut ctx, arg);
    let caller_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 31, 9);
    caller_value
        .get_operation()
        .deref_mut(&ctx)
        .set_loc(caller_loc);
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        caller_value.get_operation(),
        DebugLocalVariableInfo {
            name: "data".to_string(),
            argument_index: Some(1),
            ty: DebugLocalTypeKind::Basic {
                name: "i32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_signed",
            },
        },
    );
    llvm_export::ops::set_debug_local_source_scope(&mut ctx, caller_value.get_operation(), 0);
    caller_value.get_operation().insert_at_back(entry, &ctx);

    let inlined_value = DebugValueOp::new(&mut ctx, arg);
    let inlined_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/helper.rs", 8, 17);
    inlined_value
        .get_operation()
        .deref_mut(&ctx)
        .set_loc(inlined_loc);
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        inlined_value.get_operation(),
        DebugLocalVariableInfo {
            name: "self".to_string(),
            argument_index: Some(1),
            ty: DebugLocalTypeKind::Basic {
                name: "i32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_signed",
            },
        },
    );
    llvm_export::ops::set_debug_local_source_scope(&mut ctx, inlined_value.get_operation(), 1);
    inlined_value.get_operation().insert_at_back(entry, &ctx);

    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::Full,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    assert!(
        ir.contains("distinct !DISubprogram(name: \"caller_kernel\""),
        "caller should keep its own function debug scope:\n{ir}"
    );
    assert!(
        ir.contains("distinct !DISubprogram(name: \"helper::next\""),
        "inlined callee should get its own DISubprogram scope:\n{ir}"
    );
    assert!(
        ir.contains("!DILocalVariable(name: \"data\", arg: 1, scope: !"),
        "caller argument should remain arg #1 in the caller scope:\n{ir}"
    );
    assert!(
        ir.contains("!DILocalVariable(name: \"self\", arg: 1, scope: !"),
        "inlined callee argument should remain arg #1 in the callee scope:\n{ir}"
    );
    assert!(
        ir.contains("!DILocation(line: 8, column: 17, scope: !") && ir.contains("inlinedAt: !"),
        "inlined dbg.value location should point at the callee line and caller callsite:\n{ir}"
    );
}

#[test]
fn line_table_debug_metadata_ignores_tagged_alloca_variables() {
    let mut ctx = Context::new();

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = {
        let region = module_region.deref(&ctx);
        region.iter(&ctx).next().unwrap()
    };

    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
    let func = FuncOp::new(&mut ctx, "debug_kernel".try_into().unwrap(), func_ty);
    let func_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 40, 1);
    func.get_operation().deref_mut(&ctx).set_loc(func_loc);

    let entry = func.get_or_create_entry_block(&mut ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let one_attr = IntegerAttr::new(i32_ty, APInt::from_u32(1, NonZero::new(32).unwrap()));
    let one = ConstantOp::new(&mut ctx, one_attr.into());
    one.get_operation().insert_at_back(entry, &ctx);
    let one_val = one.get_operation().deref(&ctx).get_result(0);

    let local = AllocaOp::new(&mut ctx, i32_ty.into(), one_val);
    let local_loc = src_location(&mut ctx, "/tmp/cuda-oxide/tests/kernel.rs", 41, 9);
    local.get_operation().deref_mut(&ctx).set_loc(local_loc);
    llvm_export::ops::set_debug_local_variable(
        &mut ctx,
        local.get_operation(),
        DebugLocalVariableInfo {
            name: "x".to_string(),
            argument_index: None,
            ty: DebugLocalTypeKind::Basic {
                name: "i32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_signed",
            },
        },
    );
    local.get_operation().insert_at_back(entry, &ctx);

    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let config = DebugConfig {
        inner: PtxExportConfig,
        debug_kind: DebugKind::LineTables,
    };
    let ir =
        export_module_to_string_with_config(&ctx, &module, &config).expect("debug export succeeds");

    assert!(
        ir.contains("emissionKind: LineTablesOnly"),
        "line-table mode should stay line-table-only:\n{ir}"
    );
    assert!(
        !ir.contains("llvm.dbg.declare"),
        "line-table mode should not emit variable bindings:\n{ir}"
    );
    assert!(
        !ir.contains("DILocalVariable"),
        "line-table mode should not emit local-variable metadata:\n{ir}"
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

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let void_ty = VoidType::get(&ctx);
    let helper_ty = FuncType::get(&ctx, i32_ty.to_handle(), vec![], false);
    let helper = FuncOp::new(&mut ctx, "helper".try_into().unwrap(), helper_ty);
    helper.get_operation().insert_at_back(module_block, &ctx);

    let caller_ty = FuncType::get(&ctx, void_ty.to_handle(), vec![], false);
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

/// A float binop carrying `FastmathFlags` must export with the matching LLVM
/// fast-math keyword (`fast` for the all-bits set), while a float binop with no
/// flags must export with none. Regression guard: the textual exporter
/// previously dropped fast-math flags entirely, making the `f*_fast` intrinsic
/// lowering inert end to end.
#[test]
fn export_emits_fast_math_flags_only_on_flagged_float_ops() {
    use llvm_export::attributes::FastmathFlags;
    use llvm_export::op_interfaces::{BinArithOp, FloatBinArithOpWithFastMathFlags};
    use llvm_export::ops::{FAddOp, FMulOp};
    use pliron::builtin::types::FP32Type;

    let mut ctx = Context::new();
    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();

    let f32_ty = FP32Type::get(&ctx);
    let void_ty = VoidType::get(&ctx);
    let func_ty = FuncType::get(
        &ctx,
        void_ty.to_handle(),
        vec![f32_ty.into(), f32_ty.into()],
        false,
    );
    let func = FuncOp::new(&mut ctx, "fast_math".try_into().unwrap(), func_ty);
    let entry = func.get_or_create_entry_block(&mut ctx);
    let a = entry.deref(&ctx).get_argument(0);
    let b = entry.deref(&ctx).get_argument(1);

    // fadd with the full fast-math set.
    let fadd = FAddOp::new_with_fast_math_flags(&mut ctx, a, b, FastmathFlags::FAST.into());
    fadd.get_operation().insert_at_back(entry, &ctx);
    // fmul with no flags: must stay flag-free.
    let fmul = FMulOp::new(&mut ctx, a, b);
    fmul.get_operation().insert_at_back(entry, &ctx);
    ReturnOp::new(&mut ctx, None)
        .get_operation()
        .insert_at_back(entry, &ctx);
    func.get_operation().insert_at_back(module_block, &ctx);

    let ir = export_module_to_string(&ctx, &module).expect("export succeeds");

    assert!(
        ir.contains("fadd fast float"),
        "fast-math fadd must export the `fast` keyword:\n{ir}"
    );
    let fmul_line = ir
        .lines()
        .find(|line| line.contains("fmul"))
        .expect("exported fmul line");
    assert!(
        !fmul_line.contains("fast"),
        "a float binop with no fast-math flags must not gain them:\n{ir}"
    );
}
