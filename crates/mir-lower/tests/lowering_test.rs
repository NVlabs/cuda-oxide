/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_mir::ops as mir;
use dialect_nvvm::ops as nvvm;
use llvm_export::ops as llvm;
use pliron::builtin::op_interfaces::{CallOpCallable, CallOpInterface, SymbolOpInterface};
use pliron::builtin::ops::ModuleOp;
use pliron::context::Context;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;

#[test]
fn test_intrinsic_insertion() -> Result<(), anyhow::Error> {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    // Create Module
    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    // Create MirFunc
    let func_name = "kernel_func";
    let func_ty = pliron::builtin::types::FunctionType::get(&ctx, vec![], vec![]);

    // Manual construction of MirFuncOp
    let func_op_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1, // 1 region
    );
    let func_ty_attr = pliron::builtin::attributes::TypeAttr::new(func_ty.into());
    let func = mir::MirFuncOp::new(&mut ctx, func_op_ptr, func_ty_attr);
    func.set_symbol_name(&mut ctx, func_name.try_into().unwrap());

    // Add body - MirFuncOp has 1 region
    let region = func.get_operation().deref(&ctx).get_region(0);

    // Create block if empty (it is empty by default from Operation::new)
    let block = {
        let b = pliron::basic_block::BasicBlock::new(&mut ctx, None, vec![]);
        b.insert_at_back(region, &ctx);
        b
    };

    // Add ReadPtxSregTidXOp
    let int32_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        32,
        pliron::builtin::types::Signedness::Signless,
    );

    let tid_op_ptr = Operation::new(
        &mut ctx,
        nvvm::ReadPtxSregTidXOp::get_concrete_op_info(),
        vec![int32_ty.into()],
        vec![],
        vec![],
        0,
    );
    let tid_op = nvvm::ReadPtxSregTidXOp::new(tid_op_ptr);
    tid_op.get_operation().insert_at_back(block, &ctx);

    // Add Return
    let ret_op_ptr = Operation::new(
        &mut ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    let ret_op = mir::MirReturnOp::new(ret_op_ptr);
    ret_op.get_operation().insert_at_back(block, &ctx);

    // Add Func to Module
    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    func.get_operation().insert_at_back(module_block, &ctx);

    // Run DialectConversion-based lowering
    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    // Verify result
    let mut found_intrinsic = false;
    let mut found_kernel = false;

    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();

    for op in block.deref(&ctx).iter(&ctx) {
        if let Some(func_op) = Operation::get_op::<llvm_export::ops::FuncOp>(op, &ctx) {
            let name = func_op.get_symbol_name(&ctx).to_string();
            if name == "llvm_nvvm_read_ptx_sreg_tid_x" {
                found_intrinsic = true;
                // Intrinsic (declaration) should have 0 regions or empty region
                let num_regions = func_op.get_operation().deref(&ctx).regions().count();
                if num_regions > 0 {
                    assert!(
                        func_op
                            .get_operation()
                            .deref(&ctx)
                            .get_region(0)
                            .deref(&ctx)
                            .iter(&ctx)
                            .next()
                            .is_none()
                    );
                }
            } else if name == "kernel_func" {
                found_kernel = true;
                // Kernel should have body (1 region, not empty)
                assert!(func_op.get_operation().deref(&ctx).regions().count() > 0);
                assert!(
                    func_op
                        .get_operation()
                        .deref(&ctx)
                        .get_region(0)
                        .deref(&ctx)
                        .iter(&ctx)
                        .next()
                        .is_some()
                );
            }
        }
    }

    assert!(found_intrinsic, "Intrinsic function declaration not found");
    assert!(found_kernel, "Kernel function not found");

    Ok(())
}

#[test]
fn test_globaltimer_lowers_to_intrinsic_call() -> Result<(), anyhow::Error> {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    let func_name = "kernel_func";
    let func_ty = pliron::builtin::types::FunctionType::get(&ctx, vec![], vec![]);

    let func_op_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func_ty_attr = pliron::builtin::attributes::TypeAttr::new(func_ty.into());
    let func = mir::MirFuncOp::new(&mut ctx, func_op_ptr, func_ty_attr);
    func.set_symbol_name(&mut ctx, func_name.try_into().unwrap());

    let region = func.get_operation().deref(&ctx).get_region(0);
    let block = {
        let b = pliron::basic_block::BasicBlock::new(&mut ctx, None, vec![]);
        b.insert_at_back(region, &ctx);
        b
    };

    let i64_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        64,
        pliron::builtin::types::Signedness::Signless,
    );
    let timer_op = Operation::new(
        &mut ctx,
        nvvm::ReadPtxSregGlobaltimerOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![],
        vec![],
        0,
    );
    timer_op.insert_at_back(block, &ctx);

    let ret_op_ptr = Operation::new(
        &mut ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    let ret_op = mir::MirReturnOp::new(ret_op_ptr);
    ret_op.get_operation().insert_at_back(block, &ctx);

    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    func.get_operation().insert_at_back(module_block, &ctx);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    const INTRINSIC: &str = "llvm_nvvm_read_ptx_sreg_globaltimer";

    let mut found_decl = false;
    let mut found_call = false;
    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();

    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm_export::ops::FuncOp>(op, &ctx) else {
            continue;
        };
        let name = func_op.get_symbol_name(&ctx).to_string();

        if name == INTRINSIC {
            // Intrinsic declaration: present with empty body.
            found_decl = true;
            let num_regions = func_op.get_operation().deref(&ctx).regions().count();
            if num_regions > 0 {
                assert!(
                    func_op
                        .get_operation()
                        .deref(&ctx)
                        .get_region(0)
                        .deref(&ctx)
                        .iter(&ctx)
                        .next()
                        .is_none(),
                    "intrinsic declaration must have empty body"
                );
            }
        } else if name == func_name {
            let func_region = func_op.get_operation().deref(&ctx).get_region(0);
            for func_block in func_region.deref(&ctx).iter(&ctx) {
                for body_op in func_block.deref(&ctx).iter(&ctx) {
                    if let Some(call) = Operation::get_op::<llvm::CallOp>(body_op, &ctx)
                        && let CallOpCallable::Direct(sym) = call.callee(&ctx)
                        && sym.to_string() == INTRINSIC
                    {
                        found_call = true;
                    }
                    assert!(
                        Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx).is_none(),
                        "globaltimer must not lower to inline asm"
                    );
                }
            }
        }
    }

    assert!(
        found_decl,
        "Expected `{INTRINSIC}` declaration in lowered module"
    );
    assert!(
        found_call,
        "Expected call to `{INTRINSIC}` in lowered kernel body"
    );
    Ok(())
}

/// Lower a single zero-operand, i32-result special-register op and assert it
/// emits a declaration of and direct call to `intrinsic` (and no inline asm).
fn assert_sreg_i32_lowers_to_intrinsic(
    op_info: (
        fn(pliron::context::Ptr<pliron::operation::Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
    intrinsic: &str,
) -> Result<(), anyhow::Error> {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    let func_name = "kernel_func";
    let func_ty = pliron::builtin::types::FunctionType::get(&ctx, vec![], vec![]);

    let func_op_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func_ty_attr = pliron::builtin::attributes::TypeAttr::new(func_ty.into());
    let func = mir::MirFuncOp::new(&mut ctx, func_op_ptr, func_ty_attr);
    func.set_symbol_name(&mut ctx, func_name.try_into().unwrap());

    let region = func.get_operation().deref(&ctx).get_region(0);
    let block = {
        let b = pliron::basic_block::BasicBlock::new(&mut ctx, None, vec![]);
        b.insert_at_back(region, &ctx);
        b
    };

    let i32_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        32,
        pliron::builtin::types::Signedness::Signless,
    );
    let sreg_op = Operation::new(&mut ctx, op_info, vec![i32_ty.into()], vec![], vec![], 0);
    sreg_op.insert_at_back(block, &ctx);

    let ret_op_ptr = Operation::new(
        &mut ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    let ret_op = mir::MirReturnOp::new(ret_op_ptr);
    ret_op.get_operation().insert_at_back(block, &ctx);

    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    func.get_operation().insert_at_back(module_block, &ctx);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut found_decl = false;
    let mut found_call = false;
    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();

    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm_export::ops::FuncOp>(op, &ctx) else {
            continue;
        };
        let name = func_op.get_symbol_name(&ctx).to_string();

        if name == intrinsic {
            found_decl = true;
        } else if name == func_name {
            let func_region = func_op.get_operation().deref(&ctx).get_region(0);
            for func_block in func_region.deref(&ctx).iter(&ctx) {
                for body_op in func_block.deref(&ctx).iter(&ctx) {
                    if let Some(call) = Operation::get_op::<llvm::CallOp>(body_op, &ctx)
                        && let CallOpCallable::Direct(sym) = call.callee(&ctx)
                        && sym.to_string() == intrinsic
                    {
                        found_call = true;
                    }
                    assert!(
                        Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx).is_none(),
                        "{intrinsic} must not lower to inline asm"
                    );
                }
            }
        }
    }

    assert!(
        found_decl,
        "Expected `{intrinsic}` declaration in lowered module"
    );
    assert!(
        found_call,
        "Expected call to `{intrinsic}` in lowered kernel body"
    );
    Ok(())
}

fn assert_sreg_lowers_to_inline_asm(
    op_info: (
        fn(pliron::context::Ptr<Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    ),
    result_width: u32,
    expected_template: &str,
    expected_constraints: &str,
    expected_kind: llvm::AsmKind,
) -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let result_ty = IntegerType::get(&ctx, result_width, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![]);
    let sreg_op = Operation::new(&mut ctx, op_info, vec![result_ty.into()], vec![], vec![], 0);
    sreg_op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let mut matches = 0usize;
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in module_block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }

        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                let template = inline_asm
                    .get_attr_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()));
                if template.as_deref() != Some(expected_template) {
                    continue;
                }

                matches += 1;
                assert_eq!(
                    inline_asm
                        .get_attr_inline_asm_constraints(&ctx)
                        .map(|value| String::from((*value).clone()))
                        .as_deref(),
                    Some(expected_constraints)
                );
                assert_eq!(llvm::asm_kind(&ctx, &inline_asm), expected_kind);
            }
        }
    }

    assert_eq!(matches, 1, "expected one exact `{expected_template}` read");
    Ok(())
}

#[test]
fn test_lanemask_ops_lower_to_sreg_intrinsic_calls() -> Result<(), anyhow::Error> {
    // Each lane-position mask op lowers to its matching read-only sreg intrinsic
    // (underscores become dots on export: `..._lanemask_lt` -> `...lanemask.lt`).
    assert_sreg_i32_lowers_to_intrinsic(
        nvvm::ReadPtxSregLanemaskLtOp::get_concrete_op_info(),
        "llvm_nvvm_read_ptx_sreg_lanemask_lt",
    )?;
    assert_sreg_i32_lowers_to_intrinsic(
        nvvm::ReadPtxSregLanemaskLeOp::get_concrete_op_info(),
        "llvm_nvvm_read_ptx_sreg_lanemask_le",
    )?;
    assert_sreg_i32_lowers_to_intrinsic(
        nvvm::ReadPtxSregLanemaskEqOp::get_concrete_op_info(),
        "llvm_nvvm_read_ptx_sreg_lanemask_eq",
    )?;
    assert_sreg_i32_lowers_to_intrinsic(
        nvvm::ReadPtxSregLanemaskGeOp::get_concrete_op_info(),
        "llvm_nvvm_read_ptx_sreg_lanemask_ge",
    )?;
    assert_sreg_i32_lowers_to_intrinsic(
        nvvm::ReadPtxSregLanemaskGtOp::get_concrete_op_info(),
        "llvm_nvvm_read_ptx_sreg_lanemask_gt",
    )?;
    Ok(())
}

#[test]
fn test_warpid_ops_preserve_snapshot_semantics() -> Result<(), anyhow::Error> {
    assert_sreg_lowers_to_inline_asm(
        nvvm::ReadPtxSregWarpIdOp::get_concrete_op_info(),
        32,
        "mov.u32 $0, %warpid;",
        "=r",
        llvm::AsmKind::SideEffect,
    )?;
    assert_sreg_i32_lowers_to_intrinsic(
        nvvm::ReadPtxSregNwarpIdOp::get_concrete_op_info(),
        "llvm_nvvm_read_ptx_sreg_nwarpid",
    )?;
    Ok(())
}

#[test]
fn test_smid_ops_preserve_snapshot_semantics() -> Result<(), anyhow::Error> {
    assert_sreg_lowers_to_inline_asm(
        nvvm::ReadPtxSregSmIdOp::get_concrete_op_info(),
        32,
        "mov.u32 $0, %smid;",
        "=r",
        llvm::AsmKind::SideEffect,
    )?;
    assert_sreg_i32_lowers_to_intrinsic(
        nvvm::ReadPtxSregNsmIdOp::get_concrete_op_info(),
        "llvm_nvvm_read_ptx_sreg_nsmid",
    )?;
    Ok(())
}

#[test]
fn test_repeated_location_samples_remain_side_effecting_reads() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![]);

    for op_info in [
        nvvm::ReadPtxSregWarpIdOp::get_concrete_op_info(),
        nvvm::ReadPtxSregSmIdOp::get_concrete_op_info(),
        nvvm::ReadPtxSregWarpIdOp::get_concrete_op_info(),
        nvvm::ReadPtxSregSmIdOp::get_concrete_op_info(),
    ] {
        let op = Operation::new(&mut ctx, op_info, vec![i32_ty.into()], vec![], vec![], 0);
        op.insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let mut warpid_reads = 0usize;
    let mut smid_reads = 0usize;
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in module_block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                let template = inline_asm
                    .get_attr_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()));
                match template.as_deref() {
                    Some("mov.u32 $0, %warpid;") => warpid_reads += 1,
                    Some("mov.u32 $0, %smid;") => smid_reads += 1,
                    _ => continue,
                }
                assert_eq!(
                    llvm::asm_kind(&ctx, &inline_asm),
                    llvm::AsmKind::SideEffect,
                    "location snapshots must survive LLVM CSE"
                );
            }
        }
    }

    assert_eq!(warpid_reads, 2);
    assert_eq!(smid_reads, 2);
    Ok(())
}

#[test]
fn test_gridid_op_lowers_to_full_width_inline_ptx() -> Result<(), anyhow::Error> {
    assert_sreg_lowers_to_inline_asm(
        nvvm::ReadPtxSregGridIdOp::get_concrete_op_info(),
        64,
        "mov.u64 $0, %gridid;",
        "=l",
        llvm::AsmKind::Pure,
    )
}

#[test]
fn test_smem_size_ops_lower_to_portable_inline_ptx() -> Result<(), anyhow::Error> {
    assert_sreg_lowers_to_inline_asm(
        nvvm::ReadPtxSregDynamicSmemSizeOp::get_concrete_op_info(),
        32,
        "mov.u32 $0, %dynamic_smem_size;",
        "=r",
        llvm::AsmKind::Pure,
    )?;
    assert_sreg_lowers_to_inline_asm(
        nvvm::ReadPtxSregTotalSmemSizeOp::get_concrete_op_info(),
        32,
        "mov.u32 $0, %total_smem_size;",
        "=r",
        llvm::AsmKind::Pure,
    )?;
    Ok(())
}

#[test]
fn test_threadfence_system_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    let func_name = "kernel_func";
    let func_ty = pliron::builtin::types::FunctionType::get(&ctx, vec![], vec![]);

    let func_op_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func_ty_attr = pliron::builtin::attributes::TypeAttr::new(func_ty.into());
    let func = mir::MirFuncOp::new(&mut ctx, func_op_ptr, func_ty_attr);
    func.set_symbol_name(&mut ctx, func_name.try_into().unwrap());

    let region = func.get_operation().deref(&ctx).get_region(0);
    let block = {
        let b = pliron::basic_block::BasicBlock::new(&mut ctx, None, vec![]);
        b.insert_at_back(region, &ctx);
        b
    };

    let fence_op_ptr = Operation::new(
        &mut ctx,
        nvvm::ThreadfenceSystemOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    let fence_op = nvvm::ThreadfenceSystemOp::new(fence_op_ptr);
    fence_op.get_operation().insert_at_back(block, &ctx);

    let ret_op_ptr = Operation::new(
        &mut ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    let ret_op = mir::MirReturnOp::new(ret_op_ptr);
    ret_op.get_operation().insert_at_back(block, &ctx);

    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    func.get_operation().insert_at_back(module_block, &ctx);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut found_inline_asm = false;

    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();

    for op in block.deref(&ctx).iter(&ctx) {
        if let Some(func_op) = Operation::get_op::<llvm_export::ops::FuncOp>(op, &ctx) {
            let name = func_op.get_symbol_name(&ctx).to_string();
            if name != func_name {
                continue;
            }

            let func_region = func_op.get_operation().deref(&ctx).get_region(0);
            for func_block in func_region.deref(&ctx).iter(&ctx) {
                for body_op in func_block.deref(&ctx).iter(&ctx) {
                    if let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx)
                        && inline_asm
                            .get_attr_inline_asm_template(&ctx)
                            .is_some_and(|s| String::from((*s).clone()) == "membar.sys;")
                    {
                        found_inline_asm = true;
                        assert!(
                            inline_asm
                                .get_attr_inline_asm_convergent(&ctx)
                                .is_some_and(|b| bool::from((*b).clone()))
                        );
                    }
                }
            }
        }
    }

    assert!(
        found_inline_asm,
        "Expected membar.sys inline asm in lowered kernel"
    );
    Ok(())
}

/// `elect.sync` (Hopper sm_90+) lowers to convergent inline PTX, not to an
/// LLVM intrinsic: current LLVM ships no NVPTX selection pattern for
/// `@llvm.nvvm.elect.sync` (llc dies with "Cannot select"). Inline asm is
/// opaque to LLVM, so a wrong template / swapped constraint order / missing
/// `convergent` would compile cleanly and only surface as wrong PTX or a
/// ptxas/runtime failure far downstream. This pins the exact contract:
///   - template `{ .reg .pred p; elect.sync $0|p, $2; selp.b32 $1, 1, 0, p; }`
///   - constraints `=r,=r,r` (leader out, elected out, mask in)
///   - convergent = true
///   - two `extractvalue`s (both struct fields of the `{i32,i32}` asm result)
///   - one `trunc` (predicate i32 -> i1)
#[test]
fn test_elect_sync_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![i32_ty.into()]);
    let mask = entry.deref(&ctx).get_argument(0);

    // ElectSyncOp: 1 i32 operand (mask) -> 2 results [leader i32, is_elected i1].
    let op = Operation::new(
        &mut ctx,
        nvvm::ElectSyncOp::get_concrete_op_info(),
        vec![i32_ty.into(), i1_ty.into()],
        vec![mask],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut found_asm = false;
    let mut extract_count = 0usize;
    let mut trunc_count = 0usize;

    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();

    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                if let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) {
                    assert_eq!(
                        inline_asm
                            .get_attr_inline_asm_template(&ctx)
                            .map(|s| String::from((*s).clone()))
                            .as_deref(),
                        Some("{ .reg .pred p; elect.sync $0|p, $2; selp.b32 $1, 1, 0, p; }"),
                        "elect.sync must use the exact inline PTX template"
                    );
                    assert_eq!(
                        inline_asm
                            .get_attr_inline_asm_constraints(&ctx)
                            .map(|s| String::from((*s).clone()))
                            .as_deref(),
                        Some("=r,=r,r"),
                        "elect.sync constraints must be [leader out, elected out, mask in]"
                    );
                    assert!(
                        inline_asm
                            .get_attr_inline_asm_convergent(&ctx)
                            .is_some_and(|b| bool::from((*b).clone())),
                        "elect.sync inline asm must be convergent"
                    );
                    found_asm = true;
                }
                if Operation::get_op::<llvm::ExtractValueOp>(body_op, &ctx).is_some() {
                    extract_count += 1;
                }
                if Operation::get_op::<llvm::TruncOp>(body_op, &ctx).is_some() {
                    trunc_count += 1;
                }
            }
        }
    }

    assert!(found_asm, "elect.sync must lower to inline asm");
    assert_eq!(
        extract_count, 2,
        "must extract both fields of the {{i32,i32}} elect.sync result struct"
    );
    assert_eq!(
        trunc_count, 1,
        "elect.sync predicate must be truncated from i32 to i1"
    );

    Ok(())
}

/// The exact inline-PTX template `convert_shuffle_i64` must emit for `mode`/`clamp`.
/// Mirrors the production `format!` so a drift in either side fails the test.
fn expected_shfl_i64_template(mode: &str, clamp: i32) -> String {
    format!(
        "{{ .reg .b32 lo; .reg .b32 hi; mov.b64 {{lo, hi}}, $1; \
         shfl.sync.{mode}.b32 lo, lo, $2, {clamp}, $3; \
         shfl.sync.{mode}.b32 hi, hi, $2, {clamp}, $3; \
         mov.b64 $0, {{lo, hi}}; }}"
    )
}

/// 64-bit warp shuffle has no LLVM intrinsic (`shfl.sync` is 32-bit only), so it
/// lowers to convergent inline PTX that splits the value into two halves and runs
/// two `shfl.sync.*.b32`. Inline asm is opaque to LLVM, so a wrong mnemonic,
/// swapped operand order, wrong clamp, or missing `convergent` would only surface
/// as bad PTX downstream. This pins, for every mode, the exact template (incl. the
/// per-mode clamp: 31 for idx/bfly/down, 0 for up), the `=l,l,r,r` constraints,
/// and the convergent flag.
#[test]
fn test_shuffle_i64_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&mut ctx, 64, Signedness::Signless);
    // Kernel args: [mask (i32), value (i64), lane/delta (i32)].
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![i32_ty.into(), i64_ty.into(), i32_ty.into()]);
    let mask = entry.deref(&ctx).get_argument(0);
    let value = entry.deref(&ctx).get_argument(1);
    let lane = entry.deref(&ctx).get_argument(2);

    // One op per mode, all sharing the same [mask, value, lane] operands.
    type OpInfo = (
        fn(pliron::context::Ptr<Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    );
    let modes: [(OpInfo, &str, i32); 4] = [
        (nvvm::ShflSyncIdxI64Op::get_concrete_op_info(), "idx", 31),
        (nvvm::ShflSyncBflyI64Op::get_concrete_op_info(), "bfly", 31),
        (nvvm::ShflSyncDownI64Op::get_concrete_op_info(), "down", 31),
        (nvvm::ShflSyncUpI64Op::get_concrete_op_info(), "up", 0),
    ];
    for (opid, _, _) in modes {
        let op = Operation::new(
            &mut ctx,
            opid,
            vec![i64_ty.into()],
            vec![mask, value, lane],
            vec![],
            0,
        );
        op.insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    // Collect every inline-asm template emitted into the kernel body.
    let mut templates: Vec<String> = Vec::new();
    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                assert_eq!(
                    inline_asm
                        .get_attr_inline_asm_constraints(&ctx)
                        .map(|s| String::from((*s).clone()))
                        .as_deref(),
                    Some("=l,l,r,r"),
                    "shfl.b64 constraints must be [out i64, value i64, lane i32, mask i32]"
                );
                assert!(
                    inline_asm
                        .get_attr_inline_asm_convergent(&ctx)
                        .is_some_and(|b| bool::from((*b).clone())),
                    "shfl.b64 inline asm must be convergent"
                );
                templates.push(
                    inline_asm
                        .get_attr_inline_asm_template(&ctx)
                        .map(|s| String::from((*s).clone()))
                        .unwrap_or_default(),
                );
            }
        }
    }

    assert_eq!(
        templates.len(),
        4,
        "each of the 4 shfl.b64 modes must lower to one inline-asm op"
    );
    for (_, mode, clamp) in modes {
        let want = expected_shfl_i64_template(mode, clamp);
        assert!(
            templates.contains(&want),
            "missing inline PTX for shfl.sync.{mode}.b32 (clamp {clamp}); got {templates:?}"
        );
    }

    Ok(())
}

/// Regression cover for the per-call-site address-space coercion pass.
///
/// When a caller passes a pointer in one address space to a callee whose
/// declared parameter lives in a different address space (the
/// `*mut SharedArray<T, N>` / `addrspace(3)` case that surfaces from
/// `block_reduce` and friends), the lowerer must look up the callee's
/// declared signature and insert an `llvm.addrspacecast` so the LLVM-IR
/// verifier sees matching pointer types at the call site.
///
/// This test builds two MIR functions in one module:
///   - `callee(p: *mut i32 in addrspace(3))`
///   - `caller(p: *mut i32 in addrspace(0)) { callee(p) }`
///
/// and asserts the lowered `caller` body contains an `AddrSpaceCastOp`.
#[test]
fn addrspace_coercion_inserts_addrspacecast_at_call_site() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use llvm_export::ops::AddrSpaceCastOp;
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::{StringAttr, TypeAttr};
    use pliron::builtin::types::{FunctionType, IntegerType, Signedness};

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_addrspace_coercion".try_into().unwrap());
    let module_ptr = module.get_operation();
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();

    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let shared_ptr_ty = MirPtrType::get_shared(&mut ctx, i32_ty.into(), true);
    let generic_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);

    // Callee: takes a *mut i32 in addrspace(3), returns ().
    let callee_func_ty = FunctionType::get(&ctx, vec![shared_ptr_ty.into()], vec![]);
    let callee_func_op = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let callee_func = mir::MirFuncOp::new(
        &mut ctx,
        callee_func_op,
        TypeAttr::new(callee_func_ty.into()),
    );
    callee_func.set_symbol_name(&mut ctx, "callee".try_into().unwrap());
    {
        let region = callee_func.get_operation().deref(&ctx).get_region(0);
        let block = BasicBlock::new(&mut ctx, None, vec![shared_ptr_ty.into()]);
        block.insert_at_back(region, &ctx);

        let ret_op = Operation::new(
            &mut ctx,
            mir::MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        ret_op.insert_at_back(block, &ctx);
    }
    callee_func
        .get_operation()
        .insert_at_back(module_block, &ctx);

    // Caller: takes a *mut i32 in addrspace(0), calls `callee` with that
    // pointer. The lowerer is responsible for inserting an addrspacecast
    // since the callee's declared addrspace differs.
    let caller_func_ty = FunctionType::get(&ctx, vec![generic_ptr_ty.into()], vec![]);
    let caller_func_op = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let caller_func = mir::MirFuncOp::new(
        &mut ctx,
        caller_func_op,
        TypeAttr::new(caller_func_ty.into()),
    );
    caller_func.set_symbol_name(&mut ctx, "caller".try_into().unwrap());
    {
        let region = caller_func.get_operation().deref(&ctx).get_region(0);
        let block = BasicBlock::new(&mut ctx, None, vec![generic_ptr_ty.into()]);
        block.insert_at_back(region, &ctx);
        let arg = block.deref(&ctx).get_argument(0);

        let call_op_ptr = Operation::new(
            &mut ctx,
            mir::MirCallOp::get_concrete_op_info(),
            vec![],
            vec![arg],
            vec![],
            0,
        );
        let call_op = mir::MirCallOp::new(call_op_ptr);
        call_op.set_attr_callee(&ctx, StringAttr::new("callee".to_string()));
        call_op_ptr.insert_at_back(block, &ctx);

        let ret_op = Operation::new(
            &mut ctx,
            mir::MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        ret_op.insert_at_back(block, &ctx);
    }
    caller_func
        .get_operation()
        .insert_at_back(module_block, &ctx);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut found_addrspace_cast = false;
    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != "caller" {
            continue;
        }
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                if Operation::get_op::<AddrSpaceCastOp>(body_op, &ctx).is_some() {
                    found_addrspace_cast = true;
                }
            }
        }
    }

    assert!(
        found_addrspace_cast,
        "caller body must contain llvm.addrspacecast for the addrspace(0) -> (3) coercion at the call site",
    );
    Ok(())
}

/// A zero-sized MIR result is erased from the NVPTX function ABI, but its
/// typed value can remain live inside MIR (for example, when one ZST-returning
/// function returns the result of another). Lowering must keep the void call
/// for side effects and replace only its value result with a typed LLVM undef.
#[test]
fn zst_union_call_result_keeps_void_call_and_replaces_live_uses() -> Result<(), anyhow::Error> {
    use dialect_mir::types::{MirTupleType, MirUnionType};
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::{StringAttr, TypeAttr};
    use pliron::builtin::types::FunctionType;
    use pliron::r#type::{TypeHandle, Typed};

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let unit_ty: TypeHandle = MirTupleType::get(&mut ctx, vec![]).into();
    let union_ty: TypeHandle = MirUnionType::get(
        &mut ctx,
        "AlignedZeroUnion".into(),
        vec!["unit".into()],
        vec![unit_ty],
        0,
        16,
    )
    .into();

    let module = ModuleOp::new(&mut ctx, "test_zst_union_call".try_into().unwrap());
    let module_ptr = module.get_operation();
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();

    let callee_ty = FunctionType::get(&ctx, vec![], vec![union_ty]);
    let callee_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let callee = mir::MirFuncOp::new(&mut ctx, callee_ptr, TypeAttr::new(callee_ty.into()));
    callee.set_symbol_name(&mut ctx, "make_zero".try_into().unwrap());
    {
        let region = callee.get_operation().deref(&ctx).get_region(0);
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        block.insert_at_back(region, &ctx);

        let undef = mir::MirUndefOp::new(&mut ctx, union_ty);
        undef.get_operation().insert_at_back(block, &ctx);
        let value = undef.get_operation().deref(&ctx).get_result(0);

        let ret = Operation::new(
            &mut ctx,
            mir::MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![value],
            vec![],
            0,
        );
        ret.insert_at_back(block, &ctx);
    }
    callee.get_operation().insert_at_back(module_block, &ctx);

    let caller_ty = FunctionType::get(&ctx, vec![], vec![union_ty]);
    let caller_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let caller = mir::MirFuncOp::new(&mut ctx, caller_ptr, TypeAttr::new(caller_ty.into()));
    caller.set_symbol_name(&mut ctx, "return_called_zero".try_into().unwrap());
    {
        let region = caller.get_operation().deref(&ctx).get_region(0);
        let block = BasicBlock::new(&mut ctx, None, vec![]);
        block.insert_at_back(region, &ctx);

        let call_ptr = Operation::new(
            &mut ctx,
            mir::MirCallOp::get_concrete_op_info(),
            vec![union_ty],
            vec![],
            vec![],
            0,
        );
        let call = mir::MirCallOp::new(call_ptr);
        call.set_attr_callee(&ctx, StringAttr::new("make_zero".to_string()));
        call_ptr.insert_at_back(block, &ctx);
        let value = call_ptr.deref(&ctx).get_result(0);

        let ret = Operation::new(
            &mut ctx,
            mir::MirReturnOp::get_concrete_op_info(),
            vec![],
            vec![value],
            vec![],
            0,
        );
        ret.insert_at_back(block, &ctx);
    }
    caller.get_operation().insert_at_back(module_block, &ctx);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut found_void_call = false;
    let mut caller_undefs = 0;
    let mut found_void_return = false;
    for op in module_block.deref(&ctx).iter(&ctx) {
        let Some(func) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func.get_symbol_name(&ctx).to_string() != "return_called_zero" {
            continue;
        }
        let region = func.get_operation().deref(&ctx).get_region(0);
        for block in region.deref(&ctx).iter(&ctx) {
            for body_op in block.deref(&ctx).iter(&ctx) {
                if let Some(call) = Operation::get_op::<llvm::CallOp>(body_op, &ctx)
                    && let CallOpCallable::Direct(callee) = call.callee(&ctx)
                    && callee.to_string() == "make_zero"
                {
                    let call_op = call.get_operation().deref(&ctx);
                    assert_eq!(call_op.get_num_results(), 1);
                    assert!(
                        call_op
                            .get_result(0)
                            .get_type(&ctx)
                            .deref(&ctx)
                            .is::<llvm_export::types::VoidType>(),
                        "the ZST-returning call must use the void ABI"
                    );
                    found_void_call = true;
                }
                if Operation::get_op::<llvm::UndefOp>(body_op, &ctx).is_some() {
                    caller_undefs += 1;
                }
                if let Some(ret) = Operation::get_op::<llvm::ReturnOp>(body_op, &ctx) {
                    assert_eq!(
                        ret.get_operation().deref(&ctx).get_num_operands(),
                        0,
                        "the caller's ZST return must also use the void ABI"
                    );
                    found_void_return = true;
                }
            }
        }
    }

    assert!(
        found_void_call,
        "the LLVM call must be retained because the callee may have side effects"
    );
    assert_eq!(
        caller_undefs, 1,
        "the live MIR result must be replaced by one typed LLVM undef"
    );
    assert!(
        found_void_return,
        "the caller must retain its return terminator"
    );
    Ok(())
}

/// Lock the comparison-predicate lowering table to the rustc_codegen_ssa
/// reference (`bin_op_to_fcmp_predicate` / `bin_op_to_icmp_predicate`):
///
/// | MIR op   | float `fcmp`      | signed `icmp` | unsigned `icmp` |
/// |----------|-------------------|---------------|-----------------|
/// | `mir.eq` | `oeq` (ordered)   | `eq`          | `eq`            |
/// | `mir.ne` | `une` (UNordered) | `ne`          | `ne`            |
/// | `mir.lt` | `olt`             | `slt`         | `ult`           |
/// | `mir.le` | `ole`             | `sle`         | `ule`           |
/// | `mir.gt` | `ogt`             | `sgt`         | `ugt`           |
/// | `mir.ge` | `oge`             | `sge`         | `uge`           |
///
/// `ne` is the one float predicate that must be UNordered: Rust requires
/// `a != b == !(a == b)`, so `x != x` must be true for NaN (issue #123;
/// the ordered `one` folds the canonical NaN check to `false`).
///
/// The test also locks fastmath flags to *empty* on every lowered `fcmp`:
/// a future `nnan` default would make `fcmp nnan une x, x` poison for NaN
/// and silently re-break NaN detection while the predicate assertion above
/// stays green.
#[test]
fn test_cmp_predicate_lowering() -> Result<(), anyhow::Error> {
    use llvm_export::attributes::{FCmpPredicateAttr, FastmathFlagsAttr, ICmpPredicateAttr};
    use llvm_export::op_interfaces::FastMathFlags;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    let f32_ty = pliron::builtin::types::FP32Type::get(&ctx);
    let i32_signed = pliron::builtin::types::IntegerType::get(
        &ctx,
        32,
        pliron::builtin::types::Signedness::Signed,
    );
    let u32_unsigned = pliron::builtin::types::IntegerType::get(
        &ctx,
        32,
        pliron::builtin::types::Signedness::Unsigned,
    );
    let bool_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        1,
        pliron::builtin::types::Signedness::Signless,
    );

    // Args: (f32, f32, i32, u32). The integer args carry pre-conversion
    // signedness, which is what selects signed vs unsigned icmp predicates.
    let arg_tys: Vec<pliron::r#type::TypeHandle> = vec![
        f32_ty.into(),
        f32_ty.into(),
        i32_signed.into(),
        u32_unsigned.into(),
    ];
    let func_name = "cmp_func";
    let func_ty = pliron::builtin::types::FunctionType::get(&ctx, arg_tys.clone(), vec![]);

    let func_op_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func_ty_attr = pliron::builtin::attributes::TypeAttr::new(func_ty.into());
    let func = mir::MirFuncOp::new(&mut ctx, func_op_ptr, func_ty_attr);
    func.set_symbol_name(&mut ctx, func_name.try_into().unwrap());

    let region = func.get_operation().deref(&ctx).get_region(0);
    let block = {
        let b = pliron::basic_block::BasicBlock::new(&mut ctx, None, arg_tys);
        b.insert_at_back(region, &ctx);
        b
    };
    let fa = block.deref(&ctx).get_argument(0);
    let fb = block.deref(&ctx).get_argument(1);
    let si = block.deref(&ctx).get_argument(2);
    let ui = block.deref(&ctx).get_argument(3);

    // One comparison op per table row, in a fixed program order. The raw
    // `Operation::new` construction mirrors how the importer builds these
    // ops (mir-importer translator/rvalue.rs BinaryOp arm).
    let cmp_infos = [
        // Floats: all six predicates.
        (mir::MirEqOp::get_concrete_op_info(), fa, fb),
        (mir::MirNeOp::get_concrete_op_info(), fa, fb),
        (mir::MirLtOp::get_concrete_op_info(), fa, fb),
        (mir::MirLeOp::get_concrete_op_info(), fa, fb),
        (mir::MirGtOp::get_concrete_op_info(), fa, fb),
        (mir::MirGeOp::get_concrete_op_info(), fa, fb),
        // Signed integers: eq/ne are sign-agnostic, the rest must be s*.
        (mir::MirEqOp::get_concrete_op_info(), si, si),
        (mir::MirNeOp::get_concrete_op_info(), si, si),
        (mir::MirLtOp::get_concrete_op_info(), si, si),
        (mir::MirLeOp::get_concrete_op_info(), si, si),
        (mir::MirGtOp::get_concrete_op_info(), si, si),
        (mir::MirGeOp::get_concrete_op_info(), si, si),
        // Unsigned integers: the relational predicates must be u*.
        (mir::MirLtOp::get_concrete_op_info(), ui, ui),
        (mir::MirLeOp::get_concrete_op_info(), ui, ui),
        (mir::MirGtOp::get_concrete_op_info(), ui, ui),
        (mir::MirGeOp::get_concrete_op_info(), ui, ui),
    ];
    for (info, lhs, rhs) in cmp_infos {
        let op = Operation::new(
            &mut ctx,
            info,
            vec![bool_ty.into()],
            vec![lhs, rhs],
            vec![],
            0,
        );
        op.insert_at_back(block, &ctx);
    }

    let ret_op_ptr = Operation::new(
        &mut ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret_op_ptr.insert_at_back(block, &ctx);

    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    func.get_operation().insert_at_back(module_block, &ctx);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    // Collect lowered predicates in program order.
    let mut fcmp_preds = Vec::new();
    let mut icmp_preds = Vec::new();
    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != func_name {
            continue;
        }
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                if let Some(fcmp) = Operation::get_op::<llvm::FCmpOp>(body_op, &ctx) {
                    fcmp_preds.push(fcmp.predicate(&ctx));
                    // fcmp carries `contract` (set by add_fastmath_flags) which is a
                    // no-op for comparisons at the LLVM / PTX level. Critically, nnan
                    // is NOT set, so NaN checks like `x != x` still evaluate correctly.
                    let expected: FastmathFlagsAttr =
                        llvm_export::attributes::FastmathFlags::CONTRACT.into();
                    assert_eq!(
                        fcmp.fast_math_flags(&ctx),
                        expected,
                        "fcmp must carry only the contract flag (nnan would poison NaN checks)"
                    );
                }
                if let Some(icmp) = Operation::get_op::<llvm::ICmpOp>(body_op, &ctx) {
                    icmp_preds.push(icmp.predicate(&ctx));
                }
            }
        }
    }

    assert_eq!(
        fcmp_preds,
        vec![
            FCmpPredicateAttr::OEQ,
            FCmpPredicateAttr::UNE,
            FCmpPredicateAttr::OLT,
            FCmpPredicateAttr::OLE,
            FCmpPredicateAttr::OGT,
            FCmpPredicateAttr::OGE,
        ],
        "float comparison predicates must mirror rustc: ordered except Ne (une)"
    );
    assert_eq!(
        icmp_preds,
        vec![
            ICmpPredicateAttr::EQ,
            ICmpPredicateAttr::NE,
            ICmpPredicateAttr::SLT,
            ICmpPredicateAttr::SLE,
            ICmpPredicateAttr::SGT,
            ICmpPredicateAttr::SGE,
            ICmpPredicateAttr::ULT,
            ICmpPredicateAttr::ULE,
            ICmpPredicateAttr::UGT,
            ICmpPredicateAttr::UGE,
        ],
        "integer comparison predicates must respect pre-conversion signedness"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: build a void-returning kernel with a single NVVM op, lower it, and
// assert the kernel body contains an InlineAsmOp whose template includes the
// given `expected_asm` substring.
// ---------------------------------------------------------------------------

/// Build a kernel whose entry block contains `op` + `mir.return`, lower to LLVM,
/// and verify an `InlineAsmOp` with `expected_asm` in its template exists.
fn assert_inline_asm_lowering(
    ctx: &mut Context,
    module_ptr: pliron::context::Ptr<Operation>,
    expected_asm: &str,
) -> Result<(), anyhow::Error> {
    mir_lower::lower_mir_to_llvm(ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut found = false;
    let module_op = module_ptr.deref(ctx);
    let region = module_op.get_region(0);
    let block = region.deref(ctx).iter(ctx).next().unwrap();

    for op in block.deref(ctx).iter(ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, ctx) else {
            continue;
        };
        if func_op.get_symbol_name(ctx).to_string() != "kernel_func" {
            continue;
        }
        let func_region = func_op.get_operation().deref(ctx).get_region(0);
        for func_block in func_region.deref(ctx).iter(ctx) {
            for body_op in func_block.deref(ctx).iter(ctx) {
                if let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, ctx)
                    && inline_asm
                        .get_attr_inline_asm_template(ctx)
                        .is_some_and(|s| String::from((*s).clone()).contains(expected_asm))
                {
                    found = true;
                }
            }
        }
    }

    assert!(
        found,
        "Expected inline asm containing `{expected_asm}` in lowered kernel"
    );
    Ok(())
}

/// Helper: fresh context with all dialects registered.
fn make_test_ctx() -> Context {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);
    ctx
}

/// Helper: build a module + MirFuncOp("kernel_func") with given arg types,
/// returning the module ptr and entry block.
fn build_test_kernel(
    ctx: &mut Context,
    arg_tys: Vec<pliron::r#type::TypeHandle>,
) -> (
    pliron::context::Ptr<Operation>,
    pliron::context::Ptr<pliron::basic_block::BasicBlock>,
) {
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::TypeAttr;
    use pliron::builtin::types::FunctionType;

    let module = ModuleOp::new(ctx, "test_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    let func_ty = FunctionType::get(ctx, arg_tys.clone(), vec![]);
    let func_op_ptr = Operation::new(
        ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func = mir::MirFuncOp::new(ctx, func_op_ptr, TypeAttr::new(func_ty.into()));
    func.set_symbol_name(ctx, "kernel_func".try_into().unwrap());

    let region = func.get_operation().deref(ctx).get_region(0);
    let entry = BasicBlock::new(ctx, None, arg_tys);
    entry.insert_at_back(region, ctx);

    let module_region = module_ptr.deref(ctx).get_region(0);
    let module_block = module_region.deref(ctx).iter(ctx).next().unwrap();
    func.get_operation().insert_at_back(module_block, ctx);

    (module_ptr, entry)
}

/// Helper: append a mir.return (void) to a block.
fn append_return(ctx: &mut Context, block: pliron::context::Ptr<pliron::basic_block::BasicBlock>) {
    let ret = Operation::new(
        ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret.insert_at_back(block, ctx);
}

#[test]
fn test_cluster_mbarrier_and_fences_lower_to_exact_inline_ptx() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let bar_ptr_ty = MirPtrType::get_shared(&mut ctx, i64_ty.into(), false);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![bar_ptr_ty.into(), i32_ty.into()]);

    let bar_ptr = entry.deref(&ctx).get_argument(0);
    let bytes_or_parity = entry.deref(&ctx).get_argument(1);

    let arrive = Operation::new(
        &mut ctx,
        nvvm::MbarrierArriveExpectTxClusterOp::get_concrete_op_info(),
        vec![i64_ty.into()],
        vec![bar_ptr, bytes_or_parity],
        vec![],
        0,
    );
    arrive.insert_at_back(entry, &ctx);

    let try_wait = Operation::new(
        &mut ctx,
        nvvm::MbarrierTryWaitParityClusterOp::get_concrete_op_info(),
        vec![i1_ty.into()],
        vec![bar_ptr, bytes_or_parity],
        vec![],
        0,
    );
    try_wait.insert_at_back(entry, &ctx);

    let mbarrier_fence = Operation::new(
        &mut ctx,
        nvvm::FenceMbarrierInitReleaseClusterOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    mbarrier_fence.insert_at_back(entry, &ctx);

    let proxy_release_fence = Operation::new(
        &mut ctx,
        nvvm::FenceProxyAsyncGenericReleaseSharedCtaClusterOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    proxy_release_fence.insert_at_back(entry, &ctx);

    let proxy_acquire_fence = Operation::new(
        &mut ctx,
        nvvm::FenceProxyAsyncGenericAcquireSharedClusterClusterOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    proxy_acquire_fence.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let expected = [
        (
            "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64 $0, [$1], $2;",
            "=l,l,r,~{memory}",
        ),
        (
            "{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], $2; selp.b32 $0, 1, 0, p; }",
            "=r,l,r,~{memory}",
        ),
        ("fence.mbarrier_init.release.cluster;", "~{memory}"),
        (
            "fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;",
            "~{memory}",
        ),
        (
            "fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;",
            "~{memory}",
        ),
    ];
    let mut matches = [0usize; 5];

    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in module_block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }

        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                let template = inline_asm
                    .get_attr_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()));
                let Some(index) = expected.iter().position(|(expected_template, _)| {
                    template.as_deref() == Some(*expected_template)
                }) else {
                    continue;
                };

                matches[index] += 1;
                assert_eq!(
                    inline_asm
                        .get_attr_inline_asm_constraints(&ctx)
                        .map(|value| String::from((*value).clone()))
                        .as_deref(),
                    Some(expected[index].1)
                );
                assert_eq!(
                    llvm::asm_kind_opt(&ctx, &inline_asm),
                    Some(llvm::AsmKind::Convergent)
                );
            }
        }
    }

    assert_eq!(
        matches, [1; 5],
        "each cluster barrier/fence must lower to its exact PTX template once"
    );
    Ok(())
}

#[test]
fn test_fast_float_intrinsics_lower_to_explicit_fast_binops() -> Result<(), anyhow::Error> {
    use dialect_mir::rust_intrinsics;
    use llvm_export::attributes::{FastmathFlags, FastmathFlagsAttr};
    use llvm_export::op_interfaces::FastMathFlags;
    use pliron::builtin::attributes::StringAttr;
    use pliron::builtin::op_interfaces::CallOpInterface;
    use pliron::builtin::types::{FP32Type, FP64Type};
    use pliron::r#type::{TypeHandle, Typed};

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let f64_ty = FP64Type::get(&ctx);
    let f32_ty_obj: TypeHandle = f32_ty.into();
    let f64_ty_obj: TypeHandle = f64_ty.into();
    let (module_ptr, entry) = build_test_kernel(
        &mut ctx,
        vec![f32_ty_obj, f32_ty_obj, f64_ty_obj, f64_ty_obj],
    );
    let f32_lhs = entry.deref(&ctx).get_argument(0);
    let f32_rhs = entry.deref(&ctx).get_argument(1);
    let f64_lhs = entry.deref(&ctx).get_argument(2);
    let f64_rhs = entry.deref(&ctx).get_argument(3);

    for (callee, lhs, rhs, result_ty) in [
        (
            rust_intrinsics::CALLEE_FADD_FAST,
            f32_lhs,
            f32_rhs,
            f32_ty_obj,
        ),
        (
            rust_intrinsics::CALLEE_FSUB_FAST,
            f32_lhs,
            f32_rhs,
            f32_ty_obj,
        ),
        (
            rust_intrinsics::CALLEE_FMUL_FAST,
            f32_lhs,
            f32_rhs,
            f32_ty_obj,
        ),
        (
            rust_intrinsics::CALLEE_FDIV_FAST,
            f32_lhs,
            f32_rhs,
            f32_ty_obj,
        ),
        (
            rust_intrinsics::CALLEE_FREM_FAST,
            f32_lhs,
            f32_rhs,
            f32_ty_obj,
        ),
        (
            rust_intrinsics::CALLEE_FADD_FAST,
            f64_lhs,
            f64_rhs,
            f64_ty_obj,
        ),
        (
            rust_intrinsics::CALLEE_FSUB_FAST,
            f64_lhs,
            f64_rhs,
            f64_ty_obj,
        ),
        (
            rust_intrinsics::CALLEE_FMUL_FAST,
            f64_lhs,
            f64_rhs,
            f64_ty_obj,
        ),
        (
            rust_intrinsics::CALLEE_FDIV_FAST,
            f64_lhs,
            f64_rhs,
            f64_ty_obj,
        ),
        (
            rust_intrinsics::CALLEE_FREM_FAST,
            f64_lhs,
            f64_rhs,
            f64_ty_obj,
        ),
    ] {
        let call_ptr = Operation::new(
            &mut ctx,
            mir::MirCallOp::get_concrete_op_info(),
            vec![result_ty],
            vec![lhs, rhs],
            vec![],
            0,
        );
        let call = mir::MirCallOp::new(call_ptr);
        call.set_attr_callee(&ctx, StringAttr::new(callee.to_string()));
        call_ptr.insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let explicit_fast_flags: FastmathFlagsAttr = FastmathFlags::FAST.into();
    assert_ne!(
        explicit_fast_flags,
        FastmathFlagsAttr::default(),
        "FastmathFlagsAttr::default() is empty; f*_fast must use explicit fast flags"
    );

    let mut fadd_counts = [0usize; 2];
    let mut fsub_counts = [0usize; 2];
    let mut fmul_counts = [0usize; 2];
    let mut fdiv_counts = [0usize; 2];
    let mut frem_counts = [0usize; 2];

    macro_rules! count_fast_binop {
        ($body_op:expr, $op_ty:ty, $counts:ident, $name:literal) => {
            if let Some(op) = Operation::get_op::<$op_ty>($body_op, &ctx) {
                assert_eq!(
                    op.fast_math_flags(&ctx),
                    explicit_fast_flags,
                    concat!($name, " must carry explicit LLVM fast-math flags")
                );
                let result_ty = op.get_operation().deref(&ctx).get_result(0).get_type(&ctx);
                if result_ty == f32_ty_obj {
                    $counts[0] += 1;
                } else if result_ty == f64_ty_obj {
                    $counts[1] += 1;
                } else {
                    panic!(concat!($name, " lowered to an unexpected result type"));
                }
            }
        };
    }

    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                assert!(
                    Operation::get_op::<mir::MirCallOp>(body_op, &ctx).is_none(),
                    "f*_fast placeholder mir.call must not survive MIR lowering"
                );
                if let Some(call) = Operation::get_op::<llvm::CallOp>(body_op, &ctx)
                    && let CallOpCallable::Direct(sym) = call.callee(&ctx)
                {
                    let callee = sym.to_string();
                    assert!(
                        !callee.starts_with(rust_intrinsics::PLACEHOLDER_PREFIX),
                        "lowered LLVM must not call unresolved Rust intrinsic placeholder `{callee}`"
                    );
                }
                count_fast_binop!(body_op, llvm::FAddOp, fadd_counts, "fadd_fast");
                count_fast_binop!(body_op, llvm::FSubOp, fsub_counts, "fsub_fast");
                count_fast_binop!(body_op, llvm::FMulOp, fmul_counts, "fmul_fast");
                count_fast_binop!(body_op, llvm::FDivOp, fdiv_counts, "fdiv_fast");
                count_fast_binop!(body_op, llvm::FRemOp, frem_counts, "frem_fast");
            }
        }
    }

    assert_eq!(fadd_counts, [1, 1], "fadd_fast must lower for f32 and f64");
    assert_eq!(fsub_counts, [1, 1], "fsub_fast must lower for f32 and f64");
    assert_eq!(fmul_counts, [1, 1], "fmul_fast must lower for f32 and f64");
    assert_eq!(fdiv_counts, [1, 1], "fdiv_fast must lower for f32 and f64");
    assert_eq!(frem_counts, [1, 1], "frem_fast must lower for f32 and f64");

    Ok(())
}

// ---------------------------------------------------------------------------
// cvt.f16x2 intrinsic lowering test
// ---------------------------------------------------------------------------

#[test]
fn test_cvt_f16x2_f32_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![f32_ty.into(), f32_ty.into()]);

    let lo_val = entry.deref(&ctx).get_argument(0);
    let hi_val = entry.deref(&ctx).get_argument(1);

    // CvtF16x2F32Op: 2 f32 operands, 1 i32 result
    let op = Operation::new(
        &mut ctx,
        nvvm::CvtF16x2F32Op::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![lo_val, hi_val],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    assert_inline_asm_lowering(&mut ctx, module_ptr, "cvt.rn.f16x2.f32")
}

#[test]
fn test_cvt_rz_f16x2_f32_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![f32_ty.into(), f32_ty.into()]);

    let lo_val = entry.deref(&ctx).get_argument(0);
    let hi_val = entry.deref(&ctx).get_argument(1);

    let op = Operation::new(
        &mut ctx,
        nvvm::CvtRzF16x2F32Op::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![lo_val, hi_val],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    assert_inline_asm_lowering(&mut ctx, module_ptr, "cvt.rz.f16x2.f32")
}

#[test]
fn test_cvt_rn_relu_f16x2_f32_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![f32_ty.into(), f32_ty.into()]);

    let lo_val = entry.deref(&ctx).get_argument(0);
    let hi_val = entry.deref(&ctx).get_argument(1);

    let op = Operation::new(
        &mut ctx,
        nvvm::CvtRnReluF16x2F32Op::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![lo_val, hi_val],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    assert_inline_asm_lowering(&mut ctx, module_ptr, "cvt.rn.relu.f16x2.f32")
}

#[test]
fn test_cvt_rn_relu_bf16x2_f32_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![f32_ty.into(), f32_ty.into()]);

    let lo_val = entry.deref(&ctx).get_argument(0);
    let hi_val = entry.deref(&ctx).get_argument(1);

    let op = Operation::new(
        &mut ctx,
        nvvm::CvtRnReluBf16x2F32Op::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![lo_val, hi_val],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    assert_inline_asm_lowering(&mut ctx, module_ptr, "cvt.rn.relu.bf16x2.f32")
}

#[test]
fn test_cvt_rz_bf16x2_f32_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![f32_ty.into(), f32_ty.into()]);

    let lo_val = entry.deref(&ctx).get_argument(0);
    let hi_val = entry.deref(&ctx).get_argument(1);

    let op = Operation::new(
        &mut ctx,
        nvvm::CvtRzBf16x2F32Op::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![lo_val, hi_val],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    assert_inline_asm_lowering(&mut ctx, module_ptr, "cvt.rz.bf16x2.f32")
}

#[test]
fn test_inline_ptx_op_lowers_to_inline_asm_attrs() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![i32_ty.into()]);
    let input = entry.deref(&ctx).get_argument(0);

    let inline_ptx = nvvm::InlinePtxOp::build(
        &mut ctx,
        vec![i32_ty.into()],
        vec![input],
        "add.u32 $0, $1, $1;",
        "=r,r",
        true,
        true,
    );
    inline_ptx.insert_at_back(entry, &ctx);
    let register_only_ptx = nvvm::InlinePtxOp::build(
        &mut ctx,
        vec![i32_ty.into()],
        vec![input],
        "mul.lo.u32 $0, $1, $1;",
        "=r,r",
        false,
        true,
    );
    register_only_ptx.insert_at_back(entry, &ctx);
    let may_diverge_ptx = nvvm::InlinePtxOp::build(
        &mut ctx,
        vec![i32_ty.into()],
        vec![input],
        "cvt.u32.u32 $0, $1;",
        "=r,r",
        false,
        false,
    );
    may_diverge_ptx.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut found_conservative = false;
    let mut found_register_only = false;
    let mut found_may_diverge = false;
    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();

    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                let template = inline_asm
                    .get_attr_inline_asm_template(&ctx)
                    .map(|s| String::from((*s).clone()));
                match template.as_deref() {
                    Some("add.u32 $0, $1, $1;") => {
                        found_conservative = true;
                        assert_eq!(
                            inline_asm
                                .get_attr_inline_asm_constraints(&ctx)
                                .map(|s| String::from((*s).clone()))
                                .as_deref(),
                            Some("=r,r")
                        );
                        assert!(
                            inline_asm
                                .get_attr_inline_asm_convergent(&ctx)
                                .is_some_and(|b| bool::from((*b).clone()))
                        );
                        assert!(llvm::inline_asm_sideeffect(
                            &ctx,
                            inline_asm.get_operation()
                        ));
                    }
                    Some("mul.lo.u32 $0, $1, $1;") => {
                        found_register_only = true;
                        assert_eq!(
                            inline_asm
                                .get_attr_inline_asm_constraints(&ctx)
                                .map(|s| String::from((*s).clone()))
                                .as_deref(),
                            Some("=r,r")
                        );
                        assert!(
                            inline_asm
                                .get_attr_inline_asm_convergent(&ctx)
                                .is_some_and(|b| bool::from((*b).clone()))
                        );
                        assert!(!llvm::inline_asm_sideeffect(
                            &ctx,
                            inline_asm.get_operation()
                        ));
                    }
                    Some("cvt.u32.u32 $0, $1;") => {
                        found_may_diverge = true;
                        assert_eq!(
                            inline_asm
                                .get_attr_inline_asm_constraints(&ctx)
                                .map(|s| String::from((*s).clone()))
                                .as_deref(),
                            Some("=r,r")
                        );
                        assert!(
                            inline_asm
                                .get_attr_inline_asm_convergent(&ctx)
                                .is_some_and(|b| !bool::from((*b).clone()))
                        );
                        assert!(!llvm::inline_asm_sideeffect(
                            &ctx,
                            inline_asm.get_operation()
                        ));
                    }
                    _ => continue,
                }
            }
        }
    }

    assert!(
        found_conservative,
        "Expected conservative inline PTX asm op"
    );
    assert!(
        found_register_only,
        "Expected register-only inline PTX asm op"
    );
    assert!(found_may_diverge, "Expected may-diverge inline PTX asm op");
    Ok(())
}

/// Regression cover for PR #141: comparisons whose operand is a bool phi.
///
/// Bools are signless i1, which `can_convert_type` rejects (signless is
/// already the LLVM form), so DialectConversion records no type history for
/// a bool block argument. `is_signed_int_op` used to error out for such
/// operands ("expected IntegerType or MirPtrType operand in arithmetic op");
/// it must instead fall back to the live operand type and lower the
/// comparison as unsigned.
///
/// The function mirrors the MIR of a short-circuit kernel:
///
/// ```text
/// let p = a || b;            // bool phi: merge block argument
/// out = (p == q, p < q);     // icmp eq i1 / icmp ult i1
/// ```
///
/// ```text
/// bb0(a: i1, b: i1, q: i1):  mir.cond_br a, bb2(a), bb1()
/// bb1():                     mir.goto bb2(b)
/// bb2(p: i1):                mir.eq p, q ; mir.lt p, q ; mir.return
/// ```
#[test]
fn test_bool_phi_cmp_lowers_to_unsigned_i1_icmp() -> Result<(), anyhow::Error> {
    use llvm_export::attributes::ICmpPredicateAttr;
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::op_interfaces::OperandSegmentInterface;
    use pliron::builtin::types::{FunctionType, IntegerType, Signedness};
    use pliron::r#type::Typed;

    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    let bool_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let arg_tys: Vec<pliron::r#type::TypeHandle> =
        vec![bool_ty.into(), bool_ty.into(), bool_ty.into()];
    let func_name = "bool_phi_cmp";
    let func_ty = FunctionType::get(&ctx, arg_tys.clone(), vec![]);

    let func_op_ptr = Operation::new(
        &mut ctx,
        mir::MirFuncOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        1,
    );
    let func_ty_attr = pliron::builtin::attributes::TypeAttr::new(func_ty.into());
    let func = mir::MirFuncOp::new(&mut ctx, func_op_ptr, func_ty_attr);
    func.set_symbol_name(&mut ctx, func_name.try_into().unwrap());

    let region = func.get_operation().deref(&ctx).get_region(0);

    // bb0(a, b, q): the function entry.
    let bb0 = BasicBlock::new(&mut ctx, None, arg_tys);
    bb0.insert_at_back(region, &ctx);
    let a = bb0.deref(&ctx).get_argument(0);
    let b = bb0.deref(&ctx).get_argument(1);
    let q = bb0.deref(&ctx).get_argument(2);

    // bb1(): the short-circuit "evaluate b" block.
    let bb1 = BasicBlock::new(&mut ctx, None, vec![]);
    bb1.insert_at_back(region, &ctx);

    // bb2(p): the merge block; `p` is the bool phi.
    let bb2 = BasicBlock::new(&mut ctx, None, vec![bool_ty.into()]);
    bb2.insert_at_back(region, &ctx);
    let p = bb2.deref(&ctx).get_argument(0);

    // bb0: cond_br a, bb2(a), bb1(). On the true edge `a` is true, so
    // passing `a` itself is `a || b` without needing a constant.
    let (flat_operands, segment_sizes) =
        mir::MirCondBranchOp::compute_segment_sizes(vec![vec![a], vec![a], vec![]]);
    let cond_br = Operation::new(
        &mut ctx,
        mir::MirCondBranchOp::get_concrete_op_info(),
        vec![],
        flat_operands,
        vec![bb2, bb1],
        0,
    );
    Operation::get_op::<mir::MirCondBranchOp>(cond_br, &ctx)
        .expect("MirCondBranchOp")
        .set_operand_segment_sizes(&ctx, segment_sizes);
    cond_br.insert_at_back(bb0, &ctx);

    // bb1: goto bb2(b).
    let goto = Operation::new(
        &mut ctx,
        mir::MirGotoOp::get_concrete_op_info(),
        vec![],
        vec![b],
        vec![bb2],
        0,
    );
    goto.insert_at_back(bb1, &ctx);

    // bb2: p == q, then p < q.
    for info in [
        mir::MirEqOp::get_concrete_op_info(),
        mir::MirLtOp::get_concrete_op_info(),
    ] {
        let cmp = Operation::new(&mut ctx, info, vec![bool_ty.into()], vec![p, q], vec![], 0);
        cmp.insert_at_back(bb2, &ctx);
    }
    let ret_op = Operation::new(
        &mut ctx,
        mir::MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret_op.insert_at_back(bb2, &ctx);

    let module_region = module.get_operation().deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    func.get_operation().insert_at_back(module_block, &ctx);

    // Before the fallback, this failed with "expected IntegerType or
    // MirPtrType operand in arithmetic op".
    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut icmps = Vec::new();
    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != func_name {
            continue;
        }
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                if let Some(icmp) = Operation::get_op::<llvm::ICmpOp>(body_op, &ctx) {
                    let lhs_ty = body_op.deref(&ctx).get_operand(0).get_type(&ctx);
                    icmps.push((icmp.predicate(&ctx), lhs_ty));
                }
            }
        }
    }

    let i1: pliron::r#type::TypeHandle = bool_ty.into();
    assert_eq!(
        icmps,
        vec![(ICmpPredicateAttr::EQ, i1), (ICmpPredicateAttr::ULT, i1),],
        "bool-phi comparisons must lower to `icmp eq i1` and `icmp ult i1`"
    );
    Ok(())
}

// =============================================================================
// Integer dot product (dp4a / dp2a) lowering tests
// =============================================================================

#[test]
fn test_dp4a_s32_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![i32_ty.into(), i32_ty.into(), i32_ty.into()]);

    let a_val = entry.deref(&ctx).get_argument(0);
    let b_val = entry.deref(&ctx).get_argument(1);
    let c_val = entry.deref(&ctx).get_argument(2);

    let op = Operation::new(
        &mut ctx,
        nvvm::Dp4aS32Op::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![a_val, b_val, c_val],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    assert_inline_asm_lowering(&mut ctx, module_ptr, "dp4a.s32.s32")
}

#[test]
fn test_dp4a_u32_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![i32_ty.into(), i32_ty.into(), i32_ty.into()]);

    let a_val = entry.deref(&ctx).get_argument(0);
    let b_val = entry.deref(&ctx).get_argument(1);
    let c_val = entry.deref(&ctx).get_argument(2);

    let op = Operation::new(
        &mut ctx,
        nvvm::Dp4aU32Op::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![a_val, b_val, c_val],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    assert_inline_asm_lowering(&mut ctx, module_ptr, "dp4a.u32.u32")
}

#[test]
fn test_dp2a_s32_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![i32_ty.into(), i32_ty.into(), i32_ty.into()]);

    let a_val = entry.deref(&ctx).get_argument(0);
    let b_val = entry.deref(&ctx).get_argument(1);
    let c_val = entry.deref(&ctx).get_argument(2);

    let op = Operation::new(
        &mut ctx,
        nvvm::Dp2aS32Op::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![a_val, b_val, c_val],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    assert_inline_asm_lowering(&mut ctx, module_ptr, "dp2a.lo.s32.s32")
}

#[test]
fn test_dp2a_u32_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![i32_ty.into(), i32_ty.into(), i32_ty.into()]);

    let a_val = entry.deref(&ctx).get_argument(0);
    let b_val = entry.deref(&ctx).get_argument(1);
    let c_val = entry.deref(&ctx).get_argument(2);

    let op = Operation::new(
        &mut ctx,
        nvvm::Dp2aU32Op::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![a_val, b_val, c_val],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    assert_inline_asm_lowering(&mut ctx, module_ptr, "dp2a.lo.u32.u32")
}

// =============================================================================
// cp.async lowering tests
// =============================================================================

#[test]
fn test_cp_async_ca_4_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let dst_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let src_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![dst_ty.into(), src_ty.into()]);

    let dst = entry.deref(&ctx).get_argument(0);
    let src = entry.deref(&ctx).get_argument(1);

    let op = Operation::new(
        &mut ctx,
        nvvm::CpAsyncCa4Op::get_concrete_op_info(),
        vec![],
        vec![dst, src],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    assert_cp_async_inline_asm_lowering(&mut ctx, module_ptr, 4)
}

#[test]
fn test_cp_async_ca_8_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let dst_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let src_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), false);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![dst_ty.into(), src_ty.into()]);

    let dst = entry.deref(&ctx).get_argument(0);
    let src = entry.deref(&ctx).get_argument(1);

    let op = Operation::new(
        &mut ctx,
        nvvm::CpAsyncCa8Op::get_concrete_op_info(),
        vec![],
        vec![dst, src],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    assert_cp_async_inline_asm_lowering(&mut ctx, module_ptr, 8)
}

fn assert_cp_async_inline_asm_lowering(
    ctx: &mut Context,
    module_ptr: pliron::context::Ptr<Operation>,
    copy_size: u32,
) -> Result<(), anyhow::Error> {
    use pliron::r#type::Typed;

    mir_lower::lower_mir_to_llvm(ctx, module_ptr).map_err(|e| anyhow::anyhow!("{e}"))?;

    let expected_template = format!(
        "{{ .reg .u64 %smem64; .reg .u32 %smem32; .reg .u64 %gmem64; \
         cvta.to.shared.u64 %smem64, $0; cvt.u32.u64 %smem32, %smem64; \
         cvta.to.global.u64 %gmem64, $1; \
         cp.async.ca.shared.global [%smem32], [%gmem64], {copy_size}; }}"
    );
    let mut matches = 0;
    let module_region = module_ptr.deref(ctx).get_region(0);
    let module_block = module_region.deref(ctx).iter(ctx).next().unwrap();

    for op in module_block.deref(ctx).iter(ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, ctx) else {
            continue;
        };
        if func_op.get_symbol_name(ctx).to_string() != "kernel_func" {
            continue;
        }

        let func_region = func_op.get_operation().deref(ctx).get_region(0);
        for func_block in func_region.deref(ctx).iter(ctx) {
            for body_op in func_block.deref(ctx).iter(ctx) {
                let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, ctx) else {
                    continue;
                };
                let template = inline_asm
                    .get_attr_inline_asm_template(ctx)
                    .map(|s| String::from((*s).clone()));
                if template.as_deref() != Some(expected_template.as_str()) {
                    continue;
                }

                matches += 1;
                assert_eq!(
                    inline_asm
                        .get_attr_inline_asm_constraints(ctx)
                        .map(|s| String::from((*s).clone()))
                        .as_deref(),
                    Some("l,l,~{memory}")
                );
                assert_eq!(llvm::asm_kind(ctx, &inline_asm), llvm::AsmKind::SideEffect);
                assert!(
                    inline_asm
                        .get_attr_inline_asm_convergent(ctx)
                        .is_some_and(|value| !bool::from((*value).clone()))
                );

                let operands: Vec<_> = inline_asm.get_operation().deref(ctx).operands().collect();
                assert_eq!(operands.len(), 2);
                for operand in operands {
                    let ty = operand.get_type(ctx);
                    let ty = ty.deref(ctx);
                    let ptr_ty = ty
                        .downcast_ref::<llvm_export::types::PointerType>()
                        .expect("cp.async operands must lower to LLVM pointers");
                    assert_eq!(ptr_ty.address_space(), 0);
                }
            }
        }
    }

    assert_eq!(matches, 1, "missing exact {copy_size}-byte cp.async asm");
    Ok(())
}

// =============================================================================
// cp.async zero-fill lowering tests
// =============================================================================

#[test]
fn test_cp_async_ca_zfill_4_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let dst_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let src_ty = MirPtrType::get_generic(&mut ctx, i8_ty.into(), false);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![dst_ty.into(), src_ty.into(), i32_ty.into()]);

    let dst = entry.deref(&ctx).get_argument(0);
    let src = entry.deref(&ctx).get_argument(1);
    let src_size = entry.deref(&ctx).get_argument(2);

    let op = Operation::new(
        &mut ctx,
        nvvm::CpAsyncCaZfill4Op::get_concrete_op_info(),
        vec![],
        vec![dst, src, src_size],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    assert_cp_async_zfill_inline_asm_lowering(&mut ctx, module_ptr, 4)
}

#[test]
fn test_cp_async_ca_zfill_8_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let dst_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let src_ty = MirPtrType::get_generic(&mut ctx, i8_ty.into(), false);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![dst_ty.into(), src_ty.into(), i32_ty.into()]);

    let dst = entry.deref(&ctx).get_argument(0);
    let src = entry.deref(&ctx).get_argument(1);
    let src_size = entry.deref(&ctx).get_argument(2);

    let op = Operation::new(
        &mut ctx,
        nvvm::CpAsyncCaZfill8Op::get_concrete_op_info(),
        vec![],
        vec![dst, src, src_size],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    assert_cp_async_zfill_inline_asm_lowering(&mut ctx, module_ptr, 8)
}

#[test]
fn test_cp_async_ca_zfill_16_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let dst_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);
    let src_ty = MirPtrType::get_generic(&mut ctx, i8_ty.into(), false);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![dst_ty.into(), src_ty.into(), i32_ty.into()]);

    let dst = entry.deref(&ctx).get_argument(0);
    let src = entry.deref(&ctx).get_argument(1);
    let src_size = entry.deref(&ctx).get_argument(2);

    let op = Operation::new(
        &mut ctx,
        nvvm::CpAsyncCaZfill16Op::get_concrete_op_info(),
        vec![],
        vec![dst, src, src_size],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    assert_cp_async_zfill_inline_asm_lowering(&mut ctx, module_ptr, 16)
}

fn assert_cp_async_zfill_inline_asm_lowering(
    ctx: &mut Context,
    module_ptr: pliron::context::Ptr<Operation>,
    copy_size: u32,
) -> Result<(), anyhow::Error> {
    use pliron::builtin::types::IntegerType;
    use pliron::r#type::Typed;

    mir_lower::lower_mir_to_llvm(ctx, module_ptr).map_err(|e| anyhow::anyhow!("{e}"))?;

    let expected_template = format!(
        "{{ .reg .u64 %smem64; .reg .u32 %smem32; .reg .u64 %gmem64; \
         cvta.to.shared.u64 %smem64, $0; cvt.u32.u64 %smem32, %smem64; \
         cvta.to.global.u64 %gmem64, $1; \
         cp.async.ca.shared.global [%smem32], [%gmem64], {copy_size}, $2; }}"
    );
    let mut matches = 0;
    let module_region = module_ptr.deref(ctx).get_region(0);
    let module_block = module_region.deref(ctx).iter(ctx).next().unwrap();

    for op in module_block.deref(ctx).iter(ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, ctx) else {
            continue;
        };
        if func_op.get_symbol_name(ctx).to_string() != "kernel_func" {
            continue;
        }

        let func_region = func_op.get_operation().deref(ctx).get_region(0);
        for func_block in func_region.deref(ctx).iter(ctx) {
            for body_op in func_block.deref(ctx).iter(ctx) {
                let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, ctx) else {
                    continue;
                };
                let template = inline_asm
                    .get_attr_inline_asm_template(ctx)
                    .map(|s| String::from((*s).clone()));
                if template.as_deref() != Some(expected_template.as_str()) {
                    continue;
                }

                matches += 1;
                assert_eq!(
                    inline_asm
                        .get_attr_inline_asm_constraints(ctx)
                        .map(|s| String::from((*s).clone()))
                        .as_deref(),
                    Some("l,l,r,~{memory}")
                );
                assert_eq!(llvm::asm_kind(ctx, &inline_asm), llvm::AsmKind::SideEffect);
                assert!(
                    inline_asm
                        .get_attr_inline_asm_convergent(ctx)
                        .is_some_and(|value| !bool::from((*value).clone()))
                );

                let operands: Vec<_> = inline_asm.get_operation().deref(ctx).operands().collect();
                assert_eq!(operands.len(), 3);
                for operand in &operands[..2] {
                    let ty = operand.get_type(ctx);
                    let ty = ty.deref(ctx);
                    let ptr_ty = ty
                        .downcast_ref::<llvm_export::types::PointerType>()
                        .expect("cp.async pointer operands must lower to LLVM pointers");
                    assert_eq!(ptr_ty.address_space(), 0);
                }

                let src_size_ty = operands[2].get_type(ctx);
                let src_size_ty = src_size_ty.deref(ctx);
                let src_size_ty = src_size_ty
                    .downcast_ref::<IntegerType>()
                    .expect("cp.async src_size must lower to an integer");
                assert_eq!(src_size_ty.width(), 32);
            }
        }
    }

    assert_eq!(
        matches, 1,
        "missing exact {copy_size}-byte zero-fill cp.async asm"
    );
    Ok(())
}

// =============================================================================
// Packed bf16x2 arithmetic lowering tests
// =============================================================================

#[test]
fn test_bf16x2_arithmetic_lowers_to_exact_pure_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![i32_ty.into(), i32_ty.into(), i32_ty.into()]);

    type OpInfo = (
        fn(pliron::context::Ptr<Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    );
    let cases: [(OpInfo, usize, &str, &str); 9] = [
        (
            nvvm::FmaBf16x2Op::get_concrete_op_info(),
            3,
            "fma.rn.bf16x2 $0, $1, $2, $3;",
            "=r,r,r,r",
        ),
        (
            nvvm::FmaReluBf16x2Op::get_concrete_op_info(),
            3,
            "fma.rn.relu.bf16x2 $0, $1, $2, $3;",
            "=r,r,r,r",
        ),
        (
            nvvm::AddBf16x2Op::get_concrete_op_info(),
            2,
            "add.rn.bf16x2 $0, $1, $2;",
            "=r,r,r",
        ),
        (
            nvvm::SubBf16x2Op::get_concrete_op_info(),
            2,
            "sub.rn.bf16x2 $0, $1, $2;",
            "=r,r,r",
        ),
        (
            nvvm::MulBf16x2Op::get_concrete_op_info(),
            2,
            "mul.rn.bf16x2 $0, $1, $2;",
            "=r,r,r",
        ),
        (
            nvvm::MinBf16x2Op::get_concrete_op_info(),
            2,
            "min.bf16x2 $0, $1, $2;",
            "=r,r,r",
        ),
        (
            nvvm::MaxBf16x2Op::get_concrete_op_info(),
            2,
            "max.bf16x2 $0, $1, $2;",
            "=r,r,r",
        ),
        (
            nvvm::NegBf16x2Op::get_concrete_op_info(),
            1,
            "neg.bf16x2 $0, $1;",
            "=r,r",
        ),
        (
            nvvm::AbsBf16x2Op::get_concrete_op_info(),
            1,
            "abs.bf16x2 $0, $1;",
            "=r,r",
        ),
    ];

    let operands = [
        entry.deref(&ctx).get_argument(0),
        entry.deref(&ctx).get_argument(1),
        entry.deref(&ctx).get_argument(2),
    ];
    for &(op_info, operand_count, _, _) in &cases {
        let op = Operation::new(
            &mut ctx,
            op_info,
            vec![i32_ty.into()],
            operands[..operand_count].to_vec(),
            vec![],
            0,
        );
        op.insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut lowered = Vec::new();
    let module_op = module_ptr.deref(&ctx);
    let region = module_op.get_region(0);
    let block = region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                let inline_asm_op = inline_asm.get_operation();
                let operand_count = inline_asm_op.deref(&ctx).operands().count();
                let result_count = inline_asm_op.deref(&ctx).get_num_results();
                lowered.push((
                    inline_asm
                        .get_attr_inline_asm_template(&ctx)
                        .map(|s| String::from((*s).clone()))
                        .expect("bf16x2 inline asm must have a template"),
                    inline_asm
                        .get_attr_inline_asm_constraints(&ctx)
                        .map(|s| String::from((*s).clone()))
                        .expect("bf16x2 inline asm must have constraints"),
                    llvm::asm_kind_opt(&ctx, &inline_asm),
                    inline_asm
                        .get_attr_inline_asm_convergent(&ctx)
                        .map(|b| bool::from((*b).clone())),
                    operand_count,
                    result_count,
                ));
            }
        }
    }

    assert_eq!(
        lowered.len(),
        cases.len(),
        "each bf16x2 operation must lower to exactly one inline-asm op"
    );
    for &(_, expected_operand_count, expected_template, expected_constraints) in &cases {
        let matches: Vec<_> = lowered
            .iter()
            .filter(|(template, _, _, _, _, _)| template == expected_template)
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "expected one exact `{expected_template}` lowering"
        );
        let (_, constraints, kind, convergent, operand_count, result_count) = matches[0];
        assert_eq!(constraints, expected_constraints, "{expected_template}");
        assert_eq!(*kind, Some(llvm::AsmKind::Pure), "{expected_template}");
        assert_eq!(*convergent, Some(false), "{expected_template}");
        assert_eq!(
            *operand_count, expected_operand_count,
            "{expected_template} input arity"
        );
        assert_eq!(*result_count, 1, "{expected_template} result arity");
    }

    Ok(())
}

#[test]
fn test_stmatrix_forms_lower_to_exact_convergent_memory_asm() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let ptr_ty = MirPtrType::get_generic(&mut ctx, i8_ty.into(), true);
    let (module_ptr, entry) = build_test_kernel(
        &mut ctx,
        vec![
            ptr_ty.into(),
            i32_ty.into(),
            i32_ty.into(),
            i32_ty.into(),
            i32_ty.into(),
        ],
    );
    let args: Vec<_> = (0..5)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect();

    for (op_info, operands) in [
        (nvvm::StmatrixM8n8X4Op::get_concrete_op_info(), args.clone()),
        (
            nvvm::StmatrixM8n8X4TransOp::get_concrete_op_info(),
            args.clone(),
        ),
        (
            nvvm::StmatrixM8n8X2Op::get_concrete_op_info(),
            args[..3].to_vec(),
        ),
        (
            nvvm::StmatrixM8n8X2TransOp::get_concrete_op_info(),
            args[..3].to_vec(),
        ),
    ] {
        Operation::new(&mut ctx, op_info, vec![], operands, vec![], 0).insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let expected = [
        (
            "stmatrix.sync.aligned.m8n8.x4.shared.b16 [%ptr32], {$1, $2, $3, $4};",
            "l,r,r,r,r,~{memory}",
        ),
        (
            "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%ptr32], {$1, $2, $3, $4};",
            "l,r,r,r,r,~{memory}",
        ),
        (
            "stmatrix.sync.aligned.m8n8.x2.shared.b16 [%ptr32], {$1, $2};",
            "l,r,r,~{memory}",
        ),
        (
            "stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%ptr32], {$1, $2};",
            "l,r,r,~{memory}",
        ),
    ];
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    let mut lowered = Vec::new();

    for op in module_block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                let template = inline_asm
                    .get_attr_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .unwrap_or_default();
                if !template.contains("stmatrix.sync.aligned.") {
                    continue;
                }
                lowered.push((
                    template,
                    inline_asm
                        .get_attr_inline_asm_constraints(&ctx)
                        .map(|value| String::from((*value).clone()))
                        .unwrap_or_default(),
                    llvm::asm_kind(&ctx, &inline_asm),
                ));
            }
        }
    }

    assert_eq!(lowered.len(), expected.len());
    for (instruction, constraints) in expected {
        let matches: Vec<_> = lowered
            .iter()
            .filter(|(template, _, _)| template.contains(instruction))
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "expected one exact {instruction} lowering"
        );
        assert_eq!(matches[0].1, constraints);
        assert_eq!(matches[0].2, llvm::AsmKind::Convergent);
    }
    Ok(())
}

// =============================================================================
// Warp-level matrix (`movmatrix`) lowering test
// =============================================================================

#[test]
fn test_movmatrix_trans_b16_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![i32_ty.into()]);

    let a_val = entry.deref(&ctx).get_argument(0);

    let op = Operation::new(
        &mut ctx,
        nvvm::MovmatrixTransB16Op::get_concrete_op_info(),
        vec![i32_ty.into()],
        vec![a_val],
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let mut found = 0;
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    for module_op in module_block.deref(&ctx).iter(&ctx) {
        let Some(function) = Operation::get_op::<llvm::FuncOp>(module_op, &ctx) else {
            continue;
        };
        if function.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let body = function.get_operation().deref(&ctx).get_region(0);
        for block in body.deref(&ctx).iter(&ctx) {
            for body_op in block.deref(&ctx).iter(&ctx) {
                let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                found += 1;
                let template = asm
                    .get_attr_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()));
                let constraints = asm
                    .get_attr_inline_asm_constraints(&ctx)
                    .map(|value| String::from((*value).clone()));
                assert_eq!(
                    template.as_deref(),
                    Some("movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;")
                );
                assert_eq!(constraints.as_deref(), Some("=r,r"));
                assert_eq!(
                    llvm::asm_kind_opt(&ctx, &asm),
                    Some(llvm::AsmKind::Convergent)
                );
                assert!(
                    asm.get_attr_inline_asm_convergent(&ctx)
                        .is_some_and(|value| bool::from((*value).clone()))
                );
                assert!(
                    !constraints.as_deref().unwrap().contains("memory"),
                    "register-only movmatrix must not claim a memory clobber"
                );
            }
        }
    }

    assert_eq!(found, 1, "expected one movmatrix inline-asm operation");
    Ok(())
}

// =============================================================================
// ldmatrix lowering tests
// =============================================================================

#[test]
fn test_ldmatrix_forms_return_registers_in_exact_convergent_memory_asm() -> Result<(), anyhow::Error>
{
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let ptr_ty = MirPtrType::get_generic(&mut ctx, i8_ty.into(), true);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![ptr_ty.into()]);
    let smem_ptr = entry.deref(&ctx).get_argument(0);

    let cases = [
        (
            nvvm::LdmatrixX1Op::get_concrete_op_info(),
            1usize,
            "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {$0}, [%ptr32];",
            "=r,l,~{memory}",
            "$1",
        ),
        (
            nvvm::LdmatrixX1TransOp::get_concrete_op_info(),
            1,
            "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {$0}, [%ptr32];",
            "=r,l,~{memory}",
            "$1",
        ),
        (
            nvvm::LdmatrixX2Op::get_concrete_op_info(),
            2,
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {$0, $1}, [%ptr32];",
            "=r,=r,l,~{memory}",
            "$2",
        ),
        (
            nvvm::LdmatrixX2TransOp::get_concrete_op_info(),
            2,
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {$0, $1}, [%ptr32];",
            "=r,=r,l,~{memory}",
            "$2",
        ),
        (
            nvvm::LdmatrixX4Op::get_concrete_op_info(),
            4,
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [%ptr32];",
            "=r,=r,=r,=r,l,~{memory}",
            "$4",
        ),
        (
            nvvm::LdmatrixX4TransOp::get_concrete_op_info(),
            4,
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [%ptr32];",
            "=r,=r,=r,=r,l,~{memory}",
            "$4",
        ),
    ];

    for &(op_info, result_count, _, _, _) in &cases {
        let results = (0..result_count).map(|_| i32_ty.into()).collect();
        Operation::new(&mut ctx, op_info, results, vec![smem_ptr], vec![], 0)
            .insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    let mut lowered = Vec::new();

    for op in module_block.deref(&ctx).iter(&ctx) {
        let Some(func_op) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func_op.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }

        let func_region = func_op.get_operation().deref(&ctx).get_region(0);
        for func_block in func_region.deref(&ctx).iter(&ctx) {
            for body_op in func_block.deref(&ctx).iter(&ctx) {
                let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                let template = inline_asm
                    .get_attr_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .unwrap_or_default();
                if !template.contains("ldmatrix.sync.aligned.") {
                    continue;
                }

                let constraints = inline_asm
                    .get_attr_inline_asm_constraints(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .unwrap_or_default();
                lowered.push((
                    template,
                    constraints,
                    llvm::asm_kind(&ctx, &inline_asm),
                    body_op.deref(&ctx).get_num_operands(),
                    body_op.deref(&ctx).get_num_results(),
                ));
            }
        }
    }

    assert_eq!(lowered.len(), cases.len());
    for &(_, _, expected_instruction, expected_constraints, pointer_operand) in &cases {
        let matches: Vec<_> = lowered
            .iter()
            .filter(|(template, _, _, _, _)| template.contains(expected_instruction))
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "expected one exact ldmatrix lowering for {expected_instruction}"
        );

        let (template, constraints, kind, operands, results) = matches[0];
        assert!(
            template.contains(&format!("cvta.to.shared.u64 %ptr64, {pointer_operand};")),
            "pointer must follow all output operands: {template}"
        );
        assert!(
            !template.contains("st.b32"),
            "no hidden stack stores: {template}"
        );
        assert_eq!(constraints, expected_constraints);
        assert_eq!(*kind, llvm::AsmKind::Convergent);
        assert_eq!(*operands, 1, "only the shared-memory pointer is an input");
        assert_eq!(*results, 1, "LLVM inline asm returns one scalar or struct");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// mma.sync m16n8k16 bf16 intrinsic lowering test
// ---------------------------------------------------------------------------

#[test]
fn test_mma_m16n8k16_f32_bf16_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let f32_ty = pliron::builtin::types::FP32Type::get(&ctx);
    let i32_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        32,
        pliron::builtin::types::Signedness::Signless,
    );
    let argument_types = (0..4)
        .map(|_| f32_ty.into())
        .chain((0..6).map(|_| i32_ty.into()))
        .collect();
    let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);
    let operands = (0..10)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect();

    let op = Operation::new(
        &mut ctx,
        nvvm::MmaM16N8K16F32Bf16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        operands,
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let mut found = 0;
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    for module_op in module_block.deref(&ctx).iter(&ctx) {
        let Some(function) = Operation::get_op::<llvm::FuncOp>(module_op, &ctx) else {
            continue;
        };
        if function.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let body = function.get_operation().deref(&ctx).get_region(0);
        for block in body.deref(&ctx).iter(&ctx) {
            for body_op in block.deref(&ctx).iter(&ctx) {
                let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                let template = asm
                    .get_attr_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()));
                let constraints = asm
                    .get_attr_inline_asm_constraints(&ctx)
                    .map(|value| String::from((*value).clone()));
                if !template.as_deref().is_some_and(|t| {
                    t.contains("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")
                }) {
                    continue;
                }
                found += 1;
                let template = template.expect("MMA inline asm must have a template");
                assert_eq!(
                    template,
                    concat!(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ",
                        "{$0, $1, $2, $3}, ",
                        "{$8, $9, $10, $11}, ",
                        "{$12, $13}, ",
                        "{$4, $5, $6, $7};"
                    )
                );
                for forbidden in [".reg", "ld.", "st.", "["] {
                    assert!(
                        !template.contains(forbidden),
                        "register-only MMA must not contain {forbidden:?}: {template}"
                    );
                }
                assert_eq!(
                    constraints.as_deref(),
                    Some("=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r")
                );
                assert_eq!(
                    llvm::asm_kind_opt(&ctx, &asm),
                    Some(llvm::AsmKind::Convergent)
                );
                assert_eq!(
                    body_op.deref(&ctx).get_num_operands(),
                    10,
                    "expected C, A, and B scalar register operands"
                );
                assert_eq!(
                    body_op.deref(&ctx).get_num_results(),
                    1,
                    "LLVM inline asm returns the four D registers as one struct"
                );
            }
        }
    }

    assert_eq!(found, 1, "expected one mma.sync inline-asm operation");
    Ok(())
}

// ---------------------------------------------------------------------------
// mma.sync m16n8k16 f16 intrinsic lowering test
// ---------------------------------------------------------------------------

#[test]
fn test_mma_m16n8k16_f32_f16_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let f32_ty = pliron::builtin::types::FP32Type::get(&ctx);
    let i32_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        32,
        pliron::builtin::types::Signedness::Signless,
    );
    let argument_types = (0..4)
        .map(|_| f32_ty.into())
        .chain((0..6).map(|_| i32_ty.into()))
        .collect();
    let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);
    let operands = (0..10)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect();

    let op = Operation::new(
        &mut ctx,
        nvvm::MmaM16N8K16F32F16Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        operands,
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let mut found = 0;
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    for module_op in module_block.deref(&ctx).iter(&ctx) {
        let Some(function) = Operation::get_op::<llvm::FuncOp>(module_op, &ctx) else {
            continue;
        };
        if function.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let body = function.get_operation().deref(&ctx).get_region(0);
        for block in body.deref(&ctx).iter(&ctx) {
            for body_op in block.deref(&ctx).iter(&ctx) {
                let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                let template = asm
                    .get_attr_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()));
                let constraints = asm
                    .get_attr_inline_asm_constraints(&ctx)
                    .map(|value| String::from((*value).clone()));
                if !template.as_deref().is_some_and(|t| {
                    t.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32")
                }) {
                    continue;
                }
                found += 1;
                let template = template.expect("MMA inline asm must have a template");
                assert_eq!(
                    template,
                    concat!(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ",
                        "{$0, $1, $2, $3}, ",
                        "{$8, $9, $10, $11}, ",
                        "{$12, $13}, ",
                        "{$4, $5, $6, $7};"
                    )
                );
                for forbidden in [".reg", "ld.", "st.", "["] {
                    assert!(
                        !template.contains(forbidden),
                        "register-only MMA must not contain {forbidden:?}: {template}"
                    );
                }
                assert_eq!(
                    constraints.as_deref(),
                    Some("=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r")
                );
                assert_eq!(
                    llvm::asm_kind_opt(&ctx, &asm),
                    Some(llvm::AsmKind::Convergent)
                );
                assert_eq!(
                    body_op.deref(&ctx).get_num_operands(),
                    10,
                    "expected C, A, and B scalar register operands"
                );
                assert_eq!(
                    body_op.deref(&ctx).get_num_results(),
                    1,
                    "LLVM inline asm returns the four D registers as one struct"
                );
            }
        }
    }

    assert_eq!(found, 1, "expected one mma.sync inline-asm operation");
    Ok(())
}

#[test]
fn test_mma_m16n8k8_f32_tf32_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let f32_ty = pliron::builtin::types::FP32Type::get(&ctx);
    let i32_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        32,
        pliron::builtin::types::Signedness::Signless,
    );
    let argument_types = (0..4)
        .map(|_| f32_ty.into())
        .chain((0..6).map(|_| i32_ty.into()))
        .collect();
    let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);
    let operands = (0..10)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect();

    let op = Operation::new(
        &mut ctx,
        nvvm::MmaM16N8K8F32Tf32Op::get_concrete_op_info(),
        vec![f32_ty.into(); 4],
        operands,
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let mut found = 0;
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    for module_op in module_block.deref(&ctx).iter(&ctx) {
        let Some(function) = Operation::get_op::<llvm::FuncOp>(module_op, &ctx) else {
            continue;
        };
        if function.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let body = function.get_operation().deref(&ctx).get_region(0);
        for block in body.deref(&ctx).iter(&ctx) {
            for body_op in block.deref(&ctx).iter(&ctx) {
                let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                let template = asm
                    .get_attr_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()));
                let constraints = asm
                    .get_attr_inline_asm_constraints(&ctx)
                    .map(|value| String::from((*value).clone()));
                if !template.as_deref().is_some_and(|t| {
                    t.contains("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32")
                }) {
                    continue;
                }
                found += 1;
                let template = template.expect("MMA inline asm must have a template");
                assert_eq!(
                    template,
                    concat!(
                        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 ",
                        "{$0, $1, $2, $3}, ",
                        "{$8, $9, $10, $11}, ",
                        "{$12, $13}, ",
                        "{$4, $5, $6, $7};"
                    )
                );
                for forbidden in [".reg", "ld.", "st.", "["] {
                    assert!(
                        !template.contains(forbidden),
                        "register-only MMA must not contain {forbidden:?}: {template}"
                    );
                }
                assert_eq!(
                    constraints.as_deref(),
                    Some("=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r")
                );
                assert_eq!(
                    llvm::asm_kind_opt(&ctx, &asm),
                    Some(llvm::AsmKind::Convergent)
                );
                assert_eq!(
                    body_op.deref(&ctx).get_num_operands(),
                    10,
                    "expected C, A, and B scalar register operands"
                );
                assert_eq!(
                    body_op.deref(&ctx).get_num_results(),
                    1,
                    "LLVM inline asm returns the four D registers as one struct"
                );
            }
        }
    }

    assert_eq!(found, 1, "expected one mma.sync inline-asm operation");
    Ok(())
}

#[test]
fn test_packed_atomic_add_lowers_to_exact_side_effecting_ptx() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let ptr_ty = MirPtrType::get_generic(&mut ctx, u32_ty.into(), true);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![ptr_ty.into(), u32_ty.into()]);
    let address = entry.deref(&ctx).get_argument(0);
    let addend = entry.deref(&ctx).get_argument(1);

    for op_info in [
        nvvm::NvvmAtomAddF16x2Op::get_concrete_op_info(),
        nvvm::NvvmAtomAddBf16x2Op::get_concrete_op_info(),
    ] {
        Operation::new(
            &mut ctx,
            op_info,
            vec![u32_ty.into()],
            vec![address, addend],
            vec![],
            0,
        )
        .insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let expected = [
        "atom.global.add.noftz.f16x2 $0, [$1], $2;",
        "atom.global.add.noftz.bf16x2 $0, [$1], $2;",
    ];
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    let mut lowered = Vec::new();

    for op in module_block.deref(&ctx).iter(&ctx) {
        let Some(function) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if function.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let body = function.get_operation().deref(&ctx).get_region(0);
        for block in body.deref(&ctx).iter(&ctx) {
            for body_op in block.deref(&ctx).iter(&ctx) {
                let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                let template = asm
                    .get_attr_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .unwrap_or_default();
                if !template.starts_with("atom.global.add.noftz.") {
                    continue;
                }
                lowered.push((
                    template,
                    asm.get_attr_inline_asm_constraints(&ctx)
                        .map(|value| String::from((*value).clone()))
                        .unwrap_or_default(),
                    llvm::asm_kind(&ctx, &asm),
                    body_op.deref(&ctx).get_num_operands(),
                    body_op.deref(&ctx).get_num_results(),
                ));
            }
        }
    }

    assert_eq!(lowered.len(), expected.len());
    for instruction in expected {
        let matches: Vec<_> = lowered
            .iter()
            .filter(|(template, _, _, _, _)| template == instruction)
            .collect();
        assert_eq!(matches.len(), 1, "missing exact lowering for {instruction}");
        let (_, constraints, kind, operands, results) = matches[0];
        assert_eq!(constraints, "=r,l,r,~{memory}");
        assert_eq!(*kind, llvm::AsmKind::SideEffect);
        assert_eq!(*operands, 2);
        assert_eq!(*results, 1);
    }

    Ok(())
}

#[test]
fn test_mma_m8n8k4_f64_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::FP64Type;

    let mut ctx = make_test_ctx();
    let f64_ty = FP64Type::get(&ctx);
    let (module_ptr, entry) = build_test_kernel(
        &mut ctx,
        vec![f64_ty.into(), f64_ty.into(), f64_ty.into(), f64_ty.into()],
    );
    let operands = (0..4)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect();

    let op = Operation::new(
        &mut ctx,
        nvvm::MmaM8N8K4F64Op::get_concrete_op_info(),
        vec![f64_ty.into(), f64_ty.into()],
        operands,
        vec![],
        0,
    );
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let mut found = 0;
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    for module_op in module_block.deref(&ctx).iter(&ctx) {
        let Some(function) = Operation::get_op::<llvm::FuncOp>(module_op, &ctx) else {
            continue;
        };
        if function.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let body = function.get_operation().deref(&ctx).get_region(0);
        for block in body.deref(&ctx).iter(&ctx) {
            for body_op in block.deref(&ctx).iter(&ctx) {
                let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                let template = asm
                    .get_attr_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()));
                let constraints = asm
                    .get_attr_inline_asm_constraints(&ctx)
                    .map(|value| String::from((*value).clone()));
                if !template
                    .as_deref()
                    .is_some_and(|t| t.contains("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"))
                {
                    continue;
                }
                found += 1;
                let template = template.unwrap();
                assert_eq!(
                    template,
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$4}, {$5}, {$2, $3};"
                );
                assert!(!template.contains(".reg"));
                assert!(!template.contains("ld."));
                assert!(!template.contains("st."));
                assert_eq!(constraints.as_deref(), Some("=d,=d,d,d,d,d"));
                assert_eq!(
                    llvm::asm_kind_opt(&ctx, &asm),
                    Some(llvm::AsmKind::Convergent)
                );
                assert_eq!(
                    body_op.deref(&ctx).get_num_operands(),
                    4,
                    "expected four register inputs (c0, c1, a, b)"
                );
                assert_eq!(
                    body_op.deref(&ctx).get_num_results(),
                    1,
                    "LLVM inline asm returns one aggregate containing d0 and d1"
                );
            }
        }
    }

    assert_eq!(found, 1, "expected one mma.sync inline-asm operation");
    Ok(())
}
