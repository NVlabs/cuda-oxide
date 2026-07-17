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
fn test_generated_vote_sync_family_lowers_to_exact_typed_intrinsics() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::r#type::Typed;

    let mut ctx = make_test_ctx();
    let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![i32_ty.into(), i1_ty.into()]);
    let mask = entry.deref(&ctx).get_argument(0);
    let predicate = entry.deref(&ctx).get_argument(1);

    for vote in [
        nvvm::VoteSyncAllOp::build(&mut ctx, mask, predicate),
        nvvm::VoteSyncAnyOp::build(&mut ctx, mask, predicate),
        nvvm::VoteSyncBallotOp::build(&mut ctx, mask, predicate),
        nvvm::VoteSyncUniOp::build(&mut ctx, mask, predicate),
    ] {
        vote.insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let expected = [
        ("llvm_nvvm_vote_all_sync", 1),
        ("llvm_nvvm_vote_any_sync", 1),
        ("llvm_nvvm_vote_ballot_sync", 32),
        ("llvm_nvvm_vote_uni_sync", 1),
    ];
    let mut found = Vec::new();
    for op in lowered_kernel_body(&ctx, module_ptr) {
        assert!(
            Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
            "generated vote.sync operations must use typed LLVM intrinsics"
        );
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
            continue;
        };
        let callee = callee.to_string();
        let Some((_, result_width)) = expected.iter().find(|(name, _)| *name == callee) else {
            continue;
        };

        let call = call.get_operation().deref(&ctx);
        assert_eq!(call.get_num_operands(), 2);
        assert_eq!(call.get_num_results(), 1);

        let integer_shape = |value: pliron::value::Value| {
            let ty = value.get_type(&ctx);
            let ty = ty.deref(&ctx);
            let integer = ty
                .downcast_ref::<IntegerType>()
                .expect("vote.sync operands and results are integers");
            (integer.width(), integer.signedness())
        };
        assert_eq!(
            [
                integer_shape(call.get_operand(0)),
                integer_shape(call.get_operand(1)),
            ],
            [(32, Signedness::Signless), (1, Signedness::Signless),],
            "{callee} must preserve [mask, predicate] operand order"
        );
        assert_eq!(
            integer_shape(call.get_result(0)),
            (*result_width, Signedness::Signless),
            "{callee} returned the wrong LLVM integer type"
        );
        found.push((callee, *result_width));
    }

    found.sort();
    let mut expected = expected
        .map(|(name, width)| (name.to_owned(), width))
        .to_vec();
    expected.sort();
    assert_eq!(found, expected);
    Ok(())
}

fn lower_generated_active_mask(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![]);
    nvvm::ActiveMaskOp::build(&mut ctx).insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

#[test]
fn test_generated_active_mask_llvm_nvptx_uses_typed_intrinsic() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::IntegerType;
    use pliron::r#type::Typed;

    let (ctx, module_ptr) = lower_generated_active_mask(mir_lower::IntrinsicBackend::LlvmNvptx)?;
    let mut found = 0;
    for op in lowered_kernel_body(&ctx, module_ptr) {
        assert!(
            Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
            "LLVM-NVPTX active_mask must use the typed intrinsic"
        );
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
            continue;
        };
        if callee.to_string() != "llvm_nvvm_activemask" {
            continue;
        }

        found += 1;
        let call = op.deref(&ctx);
        assert_eq!(call.get_num_operands(), 0);
        assert_eq!(call.get_num_results(), 1);
        let result_ty = call.get_result(0).get_type(&ctx);
        let result_ty = result_ty.deref(&ctx);
        let result_ty = result_ty
            .downcast_ref::<IntegerType>()
            .expect("active_mask returns an integer");
        assert_eq!(result_ty.width(), 32);
    }
    assert_eq!(found, 1, "expected one typed active_mask call");

    let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
    let ir = llvm_export::export::export_module_to_string(&ctx, &module)
        .map_err(|error| anyhow::anyhow!(error))?;
    assert!(ir.contains("call i32 @llvm.nvvm.activemask()"), "{ir}");
    Ok(())
}

#[test]
fn test_generated_active_mask_libnvvm_uses_convergent_sideeffect_asm() -> Result<(), anyhow::Error>
{
    use pliron::builtin::types::IntegerType;
    use pliron::r#type::Typed;

    let (ctx, module_ptr) = lower_generated_active_mask(mir_lower::IntrinsicBackend::LibNvvm)?;
    let mut found = 0;
    for op in lowered_kernel_body(&ctx, module_ptr) {
        if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
            && let CallOpCallable::Direct(callee) = call.callee(&ctx)
        {
            assert_ne!(callee.to_string(), "llvm_nvvm_activemask");
        }
        let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            continue;
        };

        found += 1;
        assert_eq!(
            asm.get_attr_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .as_deref(),
            Some("activemask.b32 $0;")
        );
        assert_eq!(
            asm.get_attr_inline_asm_constraints(&ctx)
                .map(|value| String::from((*value).clone()))
                .as_deref(),
            Some("=r,~{memory}")
        );
        assert_eq!(llvm::asm_kind(&ctx, &asm), llvm::AsmKind::Convergent);
        assert!(
            asm.get_attr_inline_asm_convergent(&ctx)
                .is_some_and(|value| bool::from((*value).clone()))
        );
        let asm = op.deref(&ctx);
        assert_eq!(asm.get_num_operands(), 0);
        assert_eq!(asm.get_num_results(), 1);
        let result_ty = asm.get_result(0).get_type(&ctx);
        let result_ty = result_ty.deref(&ctx);
        let result_ty = result_ty
            .downcast_ref::<IntegerType>()
            .expect("active_mask inline asm returns an integer");
        assert_eq!(result_ty.width(), 32);
    }
    assert_eq!(found, 1, "expected one exact active_mask asm block");

    let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
    let ir = llvm_export::export::export_module_to_string(&ctx, &module)
        .map_err(|error| anyhow::anyhow!(error))?;
    assert!(
        ir.contains("call i32 asm sideeffect \"activemask.b32 $0;\", \"=r,~{memory}\"()"),
        "{ir}"
    );
    assert!(ir.contains("attributes #0 = { convergent }"), "{ir}");
    Ok(())
}

#[test]
fn test_generated_warp_match_family_uses_exact_typed_calls_and_mask_projection()
-> Result<(), anyhow::Error> {
    use llvm_export::types::StructType;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::r#type::Typed;

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let i64_ty = IntegerType::get(&ctx, 64, Signedness::Signless);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![i32_ty.into(), i32_ty.into(), i64_ty.into()]);
    let mask = entry.deref(&ctx).get_argument(0);
    let value32 = entry.deref(&ctx).get_argument(1);
    let value64 = entry.deref(&ctx).get_argument(2);
    for warp_match in [
        nvvm::MatchAnySyncI32Op::build(&mut ctx, mask, value32),
        nvvm::MatchAnySyncI64Op::build(&mut ctx, mask, value64),
        nvvm::MatchAllSyncI32Op::build(&mut ctx, mask, value32),
        nvvm::MatchAllSyncI64Op::build(&mut ctx, mask, value64),
    ] {
        warp_match.insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: mir_lower::IntrinsicBackend::LlvmNvptx,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;

    let expected = [
        ("llvm_nvvm_match_any_sync_i32", 32, false),
        ("llvm_nvvm_match_any_sync_i64", 64, false),
        ("llvm_nvvm_match_all_sync_i32p", 32, true),
        ("llvm_nvvm_match_all_sync_i64p", 64, true),
    ];
    let body = lowered_kernel_body(&ctx, module_ptr);
    let integer_width = |value: pliron::value::Value| {
        let ty = value.get_type(&ctx);
        let ty = ty.deref(&ctx);
        ty.downcast_ref::<IntegerType>()
            .expect("warp-match value is an integer")
            .width()
    };
    let mut found = Vec::new();
    let mut aggregate_results = Vec::new();
    for &op in &body {
        assert!(
            Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
            "warp match must use typed LLVM intrinsics"
        );
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
            continue;
        };
        let callee = callee.to_string();
        let Some((_, value_width, aggregate)) =
            expected.iter().find(|(name, _, _)| *name == callee)
        else {
            continue;
        };

        let call = op.deref(&ctx);
        assert_eq!(call.get_num_operands(), 2);
        assert_eq!(
            [
                integer_width(call.get_operand(0)),
                integer_width(call.get_operand(1)),
            ],
            [32, *value_width],
            "{callee} has the wrong typed signature"
        );
        assert_eq!(call.get_num_results(), 1);
        let result = call.get_result(0);
        if *aggregate {
            let result_ty = result.get_type(&ctx);
            let result_ty = result_ty.deref(&ctx);
            let result_ty = result_ty
                .downcast_ref::<StructType>()
                .expect("match.all returns an LLVM aggregate");
            assert_eq!(result_ty.num_fields(), 2);
            let field_widths = (0..2)
                .map(|index| {
                    let field = result_ty.field_type(index);
                    let field = field.deref(&ctx);
                    field
                        .downcast_ref::<IntegerType>()
                        .expect("match.all aggregate fields are integers")
                        .width()
                })
                .collect::<Vec<_>>();
            assert_eq!(field_widths, [32, 1]);
            aggregate_results.push((callee.clone(), result));
        } else {
            assert_eq!(integer_width(result), 32);
        }
        found.push(callee);
    }

    found.sort();
    let mut expected_calls = expected.map(|(name, _, _)| name.to_owned());
    expected_calls.sort();
    assert_eq!(found, expected_calls);

    let mut projected = Vec::new();
    for &op in &body {
        let Some(extract) = Operation::get_op::<llvm::ExtractValueOp>(op, &ctx) else {
            continue;
        };
        assert_eq!(extract.indices(&ctx), vec![0]);
        let extract = op.deref(&ctx);
        assert_eq!(extract.get_num_operands(), 1);
        assert_eq!(extract.get_num_results(), 1);
        assert_eq!(integer_width(extract.get_result(0)), 32);
        let aggregate = extract.get_operand(0);
        let callee = aggregate_results
            .iter()
            .find_map(|(callee, result)| (*result == aggregate).then(|| callee.clone()))
            .expect("match.all must extract from its aggregate call result");
        projected.push(callee);
    }
    projected.sort();
    assert_eq!(
        projected,
        [
            "llvm_nvvm_match_all_sync_i32p".to_owned(),
            "llvm_nvvm_match_all_sync_i64p".to_owned(),
        ]
    );
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
fn generated_threadfences_use_typed_intrinsics_on_both_backends() -> Result<(), anyhow::Error> {
    const EXPECTED: [&str; 3] = [
        "llvm_nvvm_membar_cta",
        "llvm_nvvm_membar_gl",
        "llvm_nvvm_membar_sys",
    ];

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        let mut ctx = make_test_ctx();
        let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![]);
        for op_info in [
            nvvm::ThreadfenceBlockOp::get_concrete_op_info(),
            nvvm::ThreadfenceOp::get_concrete_op_info(),
            nvvm::ThreadfenceSystemOp::get_concrete_op_info(),
        ] {
            Operation::new(&mut ctx, op_info, vec![], vec![], vec![], 0)
                .insert_at_back(entry, &ctx);
        }
        append_return(&mut ctx, entry);

        mir_lower::lower_mir_to_llvm_with_options(
            &mut ctx,
            module_ptr,
            mir_lower::LoweringOptions {
                intrinsic_backend: backend,
                ..Default::default()
            },
        )
        .map_err(|error| anyhow::anyhow!("{error}"))?;

        let mut found = Vec::new();
        for op in lowered_kernel_body(&ctx, module_ptr) {
            assert!(
                Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
                "threadfences must use their reviewed typed route"
            );
            let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
                continue;
            };
            let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
                continue;
            };
            let callee = callee.to_string();
            if EXPECTED.contains(&callee.as_str()) {
                assert_eq!(op.deref(&ctx).get_num_operands(), 0);
                found.push(callee);
            }
        }
        found.sort();
        assert_eq!(found, EXPECTED);

        let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
        let ir = llvm_export::export::export_module_to_string(&ctx, &module)
            .map_err(|error| anyhow::anyhow!(error))?;
        for symbol in [
            "llvm.nvvm.membar.cta",
            "llvm.nvvm.membar.gl",
            "llvm.nvvm.membar.sys",
        ] {
            assert!(ir.contains(&format!("call void @{symbol}()")), "{ir}");
        }
    }
    Ok(())
}

/// LLVM uses its typed intrinsic. libNVVM uses the reviewed inline-PTX fallback.
#[test]
fn generated_elect_sync_uses_the_selected_backend_route() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        let mut ctx = make_test_ctx();
        let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
        let i1_ty = IntegerType::get(&ctx, 1, Signedness::Signless);
        let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![i32_ty.into()]);
        let mask = entry.deref(&ctx).get_argument(0);
        Operation::new(
            &mut ctx,
            nvvm::ElectSyncOp::get_concrete_op_info(),
            vec![i32_ty.into(), i1_ty.into()],
            vec![mask],
            vec![],
            0,
        )
        .insert_at_back(entry, &ctx);
        append_return(&mut ctx, entry);

        mir_lower::lower_mir_to_llvm_with_options(
            &mut ctx,
            module_ptr,
            mir_lower::LoweringOptions {
                intrinsic_backend: backend,
                ..Default::default()
            },
        )
        .map_err(|error| anyhow::anyhow!("{error}"))?;

        let mut inline_asm_count = 0;
        let mut typed_call_count = 0;
        let mut extract_count = 0;
        let mut trunc_count = 0;
        for body_op in lowered_kernel_body(&ctx, module_ptr) {
            if let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) {
                assert_eq!(
                    inline_asm
                        .get_attr_inline_asm_template(&ctx)
                        .map(|value| String::from((*value).clone()))
                        .as_deref(),
                    Some("{ .reg .pred p; elect.sync $0|p, $2; selp.b32 $1, 1, 0, p; }")
                );
                assert_eq!(
                    inline_asm
                        .get_attr_inline_asm_constraints(&ctx)
                        .map(|value| String::from((*value).clone()))
                        .as_deref(),
                    Some("=r,=r,r")
                );
                assert!(
                    inline_asm
                        .get_attr_inline_asm_convergent(&ctx)
                        .is_some_and(|value| bool::from((*value).clone()))
                );
                inline_asm_count += 1;
            }
            if let Some(call) = Operation::get_op::<llvm::CallOp>(body_op, &ctx)
                && let CallOpCallable::Direct(callee) = call.callee(&ctx)
                && callee.to_string() == "llvm_nvvm_elect_sync"
            {
                typed_call_count += 1;
            }
            if Operation::get_op::<llvm::ExtractValueOp>(body_op, &ctx).is_some() {
                extract_count += 1;
            }
            if Operation::get_op::<llvm::TruncOp>(body_op, &ctx).is_some() {
                trunc_count += 1;
            }
        }

        assert_eq!(extract_count, 2);
        match backend {
            mir_lower::IntrinsicBackend::LlvmNvptx => {
                assert_eq!(typed_call_count, 1);
                assert_eq!(inline_asm_count, 0);
                assert_eq!(trunc_count, 0);
            }
            mir_lower::IntrinsicBackend::LibNvvm => {
                assert_eq!(typed_call_count, 0);
                assert_eq!(inline_asm_count, 1);
                assert_eq!(trunc_count, 1);
            }
        }
    }
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

fn lower_basic_mbarrier(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let bar_ptr_ty = MirPtrType::get_shared(&mut ctx, u64_ty.into(), false);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![bar_ptr_ty.into(), u32_ty.into()]);
    let barrier = entry.deref(&ctx).get_argument(0);
    let count = entry.deref(&ctx).get_argument(1);

    nvvm::MbarrierInitSharedOp::build(&mut ctx, barrier, count).insert_at_back(entry, &ctx);
    let arrive = nvvm::MbarrierArriveSharedOp::build(&mut ctx, barrier);
    let token = arrive.deref(&ctx).get_result(0);
    arrive.insert_at_back(entry, &ctx);
    nvvm::MbarrierTestWaitSharedOp::build(&mut ctx, barrier, token).insert_at_back(entry, &ctx);
    nvvm::MbarrierInvalSharedOp::build(&mut ctx, barrier).insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

#[test]
fn test_generated_basic_mbarrier_uses_shared_lowering_on_both_backends() -> Result<(), anyhow::Error>
{
    let expected_calls = [
        ("llvm_nvvm_mbarrier_init_shared", 2),
        ("llvm_nvvm_mbarrier_arrive_shared", 1),
        ("llvm_nvvm_mbarrier_inval_shared", 1),
    ];
    let init_template = "mbarrier.init.shared.b64 [$0], $1;";
    let arrive_template = "mbarrier.arrive.shared.b64 $0, [$1];";
    let test_wait_template =
        "{ .reg .pred %p0; mbarrier.test_wait.shared.b64 %p0, [$1], $2; selp.b32 $0, 1, 0, %p0; }";
    let inval_template = "mbarrier.inval.shared.b64 [$0];";

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        let (ctx, module_ptr) = lower_basic_mbarrier(backend)?;
        let mut call_counts = [0usize; 3];
        let expected_asm = match backend {
            mir_lower::IntrinsicBackend::LlvmNvptx => {
                vec![(test_wait_template, "=r,l,l,~{memory}", 2)]
            }
            mir_lower::IntrinsicBackend::LibNvvm => vec![
                (init_template, "l,r,~{memory}", 2),
                (arrive_template, "=l,l,~{memory}", 1),
                (test_wait_template, "=r,l,l,~{memory}", 2),
                (inval_template, "l,~{memory}", 1),
            ],
        };
        let mut asm_counts = vec![0usize; expected_asm.len()];
        let mut trunc_count = 0;

        for op in lowered_kernel_body(&ctx, module_ptr) {
            if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) {
                let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
                    continue;
                };
                let callee = callee.to_string();
                let Some(index) = expected_calls
                    .iter()
                    .position(|(expected, _)| callee == *expected)
                else {
                    assert_ne!(callee, "llvm_nvvm_mbarrier_test_wait_shared");
                    continue;
                };
                call_counts[index] += 1;
                assert_eq!(op.deref(&ctx).get_num_operands(), expected_calls[index].1);
            }

            if let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) {
                let template = inline_asm
                    .get_attr_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()));
                let index = expected_asm
                    .iter()
                    .position(|(expected, _, _)| template.as_deref() == Some(*expected))
                    .unwrap_or_else(|| panic!("unexpected {backend:?} inline PTX: {template:?}"));
                let (_, constraints, operand_count) = expected_asm[index];
                asm_counts[index] += 1;
                assert_eq!(
                    inline_asm
                        .get_attr_inline_asm_constraints(&ctx)
                        .map(|value| String::from((*value).clone()))
                        .as_deref(),
                    Some(constraints)
                );
                assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Convergent);
                let asm = op.deref(&ctx);
                assert_eq!(asm.get_num_operands(), operand_count);
                assert_eq!(asm.get_num_results(), 1);
            }

            if Operation::get_op::<llvm::TruncOp>(op, &ctx).is_some() {
                trunc_count += 1;
            }
        }

        match backend {
            mir_lower::IntrinsicBackend::LlvmNvptx => assert_eq!(call_counts, [1; 3]),
            mir_lower::IntrinsicBackend::LibNvvm => assert_eq!(call_counts, [0; 3]),
        }
        assert_eq!(asm_counts, vec![1; expected_asm.len()]);
        assert_eq!(
            trunc_count, 1,
            "test-wait must adapt its i32 predicate to i1"
        );

        let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
        let ir = llvm_export::export::export_module_to_string(&ctx, &module)
            .map_err(|error| anyhow::anyhow!(error))?;
        match backend {
            mir_lower::IntrinsicBackend::LlvmNvptx => {
                assert!(
                    ir.contains("call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3)"),
                    "{ir}"
                );
                assert!(
                    ir.contains("call i64 @llvm.nvvm.mbarrier.arrive.shared(ptr addrspace(3)"),
                    "{ir}"
                );
                assert!(
                    ir.contains("call void @llvm.nvvm.mbarrier.inval.shared(ptr addrspace(3)"),
                    "{ir}"
                );
                assert!(!ir.contains(init_template), "{ir}");
                assert!(!ir.contains(arrive_template), "{ir}");
                assert!(!ir.contains(inval_template), "{ir}");
            }
            mir_lower::IntrinsicBackend::LibNvvm => {
                for symbol in [
                    "@llvm.nvvm.mbarrier.init.shared",
                    "@llvm.nvvm.mbarrier.arrive.shared",
                    "@llvm.nvvm.mbarrier.inval.shared",
                ] {
                    assert!(
                        !ir.contains(symbol),
                        "libNVVM route retained {symbol}:\n{ir}"
                    );
                }
                for template in [init_template, arrive_template, inval_template] {
                    assert!(ir.contains(template), "{ir}");
                }
            }
        }
        assert!(ir.contains(test_wait_template), "{ir}");
        assert!(ir.contains("asm sideeffect"), "{ir}");
        assert!(ir.contains("trunc i32") && ir.contains("to i1"), "{ir}");
        assert!(ir.contains("attributes #0 = { convergent }"), "{ir}");
    }
    Ok(())
}

fn lower_cluster_barriers(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![]);
    for mode in [
        nvvm::ClusterBarrierModeAttr::Arrive,
        nvvm::ClusterBarrierModeAttr::ArriveAligned,
        nvvm::ClusterBarrierModeAttr::ArriveRelaxed,
        nvvm::ClusterBarrierModeAttr::ArriveRelaxedAligned,
        nvvm::ClusterBarrierModeAttr::Wait,
        nvvm::ClusterBarrierModeAttr::WaitAligned,
    ] {
        nvvm::ClusterBarrierOp::build(&mut ctx, mode).insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);
    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

#[test]
fn generated_cluster_barriers_lower_exactly_on_both_backends() -> Result<(), anyhow::Error> {
    let recipes = [
        (
            "llvm_nvvm_barrier_cluster_arrive",
            "barrier.cluster.arrive;",
        ),
        (
            "llvm_nvvm_barrier_cluster_arrive_aligned",
            "barrier.cluster.arrive.aligned;",
        ),
        (
            "llvm_nvvm_barrier_cluster_arrive_relaxed",
            "barrier.cluster.arrive.relaxed;",
        ),
        (
            "llvm_nvvm_barrier_cluster_arrive_relaxed_aligned",
            "barrier.cluster.arrive.relaxed.aligned;",
        ),
        ("llvm_nvvm_barrier_cluster_wait", "barrier.cluster.wait;"),
        (
            "llvm_nvvm_barrier_cluster_wait_aligned",
            "barrier.cluster.wait.aligned;",
        ),
    ];

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        let (ctx, module_ptr) = lower_cluster_barriers(backend)?;
        let mut calls = [0usize; 6];
        let mut asm = [0usize; 6];
        for op in lowered_kernel_body(&ctx, module_ptr) {
            if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) {
                let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
                    continue;
                };
                if let Some(index) = recipes
                    .iter()
                    .position(|(symbol, _)| callee.to_string() == *symbol)
                {
                    calls[index] += 1;
                }
            }
            if let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) {
                let template = inline_asm
                    .get_attr_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()));
                if let Some(index) = recipes
                    .iter()
                    .position(|(_, expected)| template.as_deref() == Some(*expected))
                {
                    asm[index] += 1;
                    assert_eq!(
                        inline_asm
                            .get_attr_inline_asm_constraints(&ctx)
                            .map(|value| String::from((*value).clone()))
                            .as_deref(),
                        Some("~{memory}")
                    );
                    assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Convergent);
                }
            }
        }
        match backend {
            mir_lower::IntrinsicBackend::LlvmNvptx => {
                assert_eq!(calls, [1; 6]);
                assert_eq!(asm, [0; 6]);
            }
            mir_lower::IntrinsicBackend::LibNvvm => {
                assert_eq!(calls, [0; 6]);
                assert_eq!(asm, [1; 6]);
            }
        }
    }
    Ok(())
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
            "{ .reg .pred %p0; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 %p0, [$1], $2; selp.b32 $0, 1, 0, %p0; }",
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

#[test]
fn test_cluster_grid_compatibility_ops_keep_original_lowering() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_type = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![]);
    for op_info in [
        nvvm::ReadPtxSregClusterIdxOp::get_concrete_op_info(),
        nvvm::ReadPtxSregNclusterIdOp::get_concrete_op_info(),
    ] {
        Operation::new(&mut ctx, op_info, vec![i32_type.into()], vec![], vec![], 0)
            .insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .map_err(|error| anyhow::anyhow!("{error}"))?;

    let lowered = lowered_kernel_body(&ctx, module_ptr)
        .into_iter()
        .filter_map(|op| Operation::get_op::<llvm::InlineAsmOp>(op, &ctx))
        .filter_map(|asm| {
            let template = asm
                .get_attr_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))?;
            (template.contains("%clusterid") || template.contains("%nclusterid"))
                .then(|| (template, asm))
        })
        .collect::<Vec<_>>();
    assert_eq!(lowered.len(), 2);
    assert!(lowered.iter().any(|(template, _)| {
        template
            == "{ .reg .u32 %cx, %cy, %cz, %nx, %ny, %nxy, %xy; mov.u32 %cx, %clusterid.x; mov.u32 %cy, %clusterid.y; mov.u32 %cz, %clusterid.z; mov.u32 %nx, %nclusterid.x; mov.u32 %ny, %nclusterid.y; mul.lo.u32 %nxy, %nx, %ny; mad.lo.u32 %xy, %cy, %nx, %cx; mad.lo.u32 $0, %cz, %nxy, %xy; }"
    }));
    assert!(lowered.iter().any(|(template, _)| {
        template
            == "{ .reg .u32 %nx, %ny, %nz, %nxy; mov.u32 %nx, %nclusterid.x; mov.u32 %ny, %nclusterid.y; mov.u32 %nz, %nclusterid.z; mul.lo.u32 %nxy, %nx, %ny; mul.lo.u32 $0, %nxy, %nz; }"
    }));
    for (_, asm) in lowered {
        assert_eq!(
            asm.get_attr_inline_asm_constraints(&ctx)
                .map(|value| String::from((*value).clone()))
                .as_deref(),
            Some("=r")
        );
        assert_eq!(llvm::asm_kind(&ctx, &asm), llvm::AsmKind::Convergent);
    }
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

const DOT_PRODUCT_TYPED_INTRINSICS: [&str; 4] = [
    "llvm_nvvm_idp4a_s_s",
    "llvm_nvvm_idp4a_u_u",
    "llvm_nvvm_idp2a_s_s",
    "llvm_nvvm_idp2a_u_u",
];

const DOT_PRODUCT_PTX: [&str; 4] = [
    "dp4a.s32.s32 $0, $1, $2, $3;",
    "dp4a.u32.u32 $0, $1, $2, $3;",
    "dp2a.lo.s32.s32 $0, $1, $2, $3;",
    "dp2a.lo.u32.u32 $0, $1, $2, $3;",
];

fn lower_all_dot_product_forms(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![i32_ty.into(), i32_ty.into(), i32_ty.into()]);
    let operands = (0..3)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect::<Vec<_>>();
    for op_info in [
        nvvm::Dp4aS32Op::get_concrete_op_info(),
        nvvm::Dp4aU32Op::get_concrete_op_info(),
        nvvm::Dp2aS32Op::get_concrete_op_info(),
        nvvm::Dp2aU32Op::get_concrete_op_info(),
    ] {
        Operation::new(
            &mut ctx,
            op_info,
            vec![i32_ty.into()],
            operands.clone(),
            vec![],
            0,
        )
        .insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

#[test]
fn test_dot_product_llvm_nvptx_uses_typed_intrinsics_and_low_selector() -> Result<(), anyhow::Error>
{
    use pliron::builtin::attributes::IntegerAttr;

    let (ctx, module_ptr) = lower_all_dot_product_forms(mir_lower::IntrinsicBackend::LlvmNvptx)?;
    let body = lowered_kernel_body(&ctx, module_ptr);
    let mut calls = Vec::new();
    for op in body {
        assert!(
            Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
            "LLVM-NVPTX dot products must use typed intrinsics"
        );
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
            continue;
        };
        let callee = callee.to_string();
        if !callee.starts_with("llvm_nvvm_idp") {
            continue;
        }
        let expected_arity = if callee.contains("idp2a") { 4 } else { 3 };
        assert_eq!(op.deref(&ctx).get_num_operands(), expected_arity);
        if expected_arity == 4 {
            let selector = op.deref(&ctx).get_operand(2);
            let defining_op = selector.defining_op().expect("selector is a constant");
            let constant = Operation::get_op::<llvm::ConstantOp>(defining_op, &ctx)
                .expect("selector is an LLVM constant");
            let attribute = constant.get_value(&ctx);
            let integer = attribute
                .downcast_ref::<IntegerAttr>()
                .expect("selector constant is an integer");
            assert_eq!(integer.value().bw(), 1);
            assert_eq!(integer.value().to_u64(), 0, "dp2a must select `.lo`");
        }
        calls.push(callee);
    }
    calls.sort();
    let mut expected = DOT_PRODUCT_TYPED_INTRINSICS.map(str::to_owned);
    expected.sort();
    assert_eq!(calls, expected);
    Ok(())
}

#[test]
fn test_dot_product_libnvvm_uses_exact_pure_inline_ptx() -> Result<(), anyhow::Error> {
    let (ctx, module_ptr) = lower_all_dot_product_forms(mir_lower::IntrinsicBackend::LibNvvm)?;
    let body = lowered_kernel_body(&ctx, module_ptr);
    let mut inline_ptx = Vec::new();
    for op in body {
        assert!(
            Operation::get_op::<llvm::CallOp>(op, &ctx).is_none(),
            "libNVVM dot products must not use typed intrinsic calls"
        );
        let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            continue;
        };
        inline_ptx.push(
            inline_asm
                .get_attr_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .unwrap_or_default(),
        );
        assert_eq!(
            inline_asm
                .get_attr_inline_asm_constraints(&ctx)
                .map(|value| String::from((*value).clone()))
                .as_deref(),
            Some("=r,r,r,r")
        );
        assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Pure);
        assert_eq!(op.deref(&ctx).get_num_operands(), 3);
        assert_eq!(op.deref(&ctx).get_num_results(), 1);
    }
    inline_ptx.sort();
    let mut expected = DOT_PRODUCT_PTX.map(str::to_owned);
    expected.sort();
    assert_eq!(inline_ptx, expected);
    Ok(())
}

// =============================================================================
// Byte permutation lowering tests
// =============================================================================

const PRMT_TYPED_INTRINSICS: [(&str, usize); 7] = [
    ("llvm_nvvm_prmt", 3),
    ("llvm_nvvm_prmt_f4e", 3),
    ("llvm_nvvm_prmt_b4e", 3),
    ("llvm_nvvm_prmt_rc8", 2),
    ("llvm_nvvm_prmt_ecl", 2),
    ("llvm_nvvm_prmt_ecr", 2),
    ("llvm_nvvm_prmt_rc16", 2),
];

const PRMT_INLINE_PTX: [(&str, &str, usize); 7] = [
    ("prmt.b32 $0, $1, $2, $3;", "=r,r,r,r", 3),
    ("prmt.b32.f4e $0, $1, $2, $3;", "=r,r,r,r", 3),
    ("prmt.b32.b4e $0, $1, $2, $3;", "=r,r,r,r", 3),
    ("prmt.b32.rc8 $0, $1, 0, $2;", "=r,r,r", 2),
    ("prmt.b32.ecl $0, $1, 0, $2;", "=r,r,r", 2),
    ("prmt.b32.ecr $0, $1, 0, $2;", "=r,r,r", 2),
    ("prmt.b32.rc16 $0, $1, 0, $2;", "=r,r,r", 2),
];

fn lower_all_prmt_modes(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![i32_ty.into(), i32_ty.into(), i32_ty.into()]);
    let operands = (0..3)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect::<Vec<_>>();

    for mode in [
        nvvm::PrmtModeAttr::Generic,
        nvvm::PrmtModeAttr::F4e,
        nvvm::PrmtModeAttr::B4e,
    ] {
        nvvm::PrmtOp::build(&mut ctx, operands.clone(), mode).insert_at_back(entry, &ctx);
    }
    for mode in [
        nvvm::PrmtModeAttr::Rc8,
        nvvm::PrmtModeAttr::Ecl,
        nvvm::PrmtModeAttr::Ecr,
        nvvm::PrmtModeAttr::Rc16,
    ] {
        nvvm::PrmtOp::build(&mut ctx, vec![operands[0], operands[2]], mode)
            .insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

#[test]
fn test_prmt_llvm_nvptx_uses_exact_typed_intrinsics() -> Result<(), anyhow::Error> {
    let (ctx, module_ptr) = lower_all_prmt_modes(mir_lower::IntrinsicBackend::LlvmNvptx)?;
    let body = lowered_kernel_body(&ctx, module_ptr);
    let mut calls = Vec::new();
    for op in body {
        assert!(
            Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
            "LLVM-NVPTX byte permutations must use typed intrinsics"
        );
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
            continue;
        };
        let callee = callee.to_string();
        let Some((_, expected_arity)) = PRMT_TYPED_INTRINSICS
            .iter()
            .find(|(expected, _)| *expected == callee)
        else {
            continue;
        };
        assert_eq!(op.deref(&ctx).get_num_operands(), *expected_arity);
        calls.push((callee, *expected_arity));
    }
    calls.sort();
    let mut expected = PRMT_TYPED_INTRINSICS.map(|(name, arity)| (name.to_owned(), arity));
    expected.sort();
    assert_eq!(calls, expected);
    Ok(())
}

#[test]
fn test_prmt_libnvvm_uses_exact_pure_inline_ptx() -> Result<(), anyhow::Error> {
    let (ctx, module_ptr) = lower_all_prmt_modes(mir_lower::IntrinsicBackend::LibNvvm)?;
    let body = lowered_kernel_body(&ctx, module_ptr);
    let mut inline_ptx = Vec::new();
    for op in body {
        if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
            && let CallOpCallable::Direct(callee) = call.callee(&ctx)
        {
            assert!(
                !callee.to_string().starts_with("llvm_nvvm_prmt"),
                "libNVVM byte permutations must not use typed intrinsic calls"
            );
        }
        let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            continue;
        };
        let template = inline_asm
            .get_attr_inline_asm_template(&ctx)
            .map(|value| String::from((*value).clone()))
            .unwrap_or_default();
        let Some((_, expected_constraints, expected_arity)) = PRMT_INLINE_PTX
            .iter()
            .find(|(expected, _, _)| *expected == template)
        else {
            continue;
        };
        assert_eq!(
            inline_asm
                .get_attr_inline_asm_constraints(&ctx)
                .map(|value| String::from((*value).clone()))
                .as_deref(),
            Some(*expected_constraints)
        );
        assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Pure);
        assert_eq!(op.deref(&ctx).get_num_operands(), *expected_arity);
        assert_eq!(op.deref(&ctx).get_num_results(), 1);
        inline_ptx.push((template, *expected_constraints, *expected_arity));
    }
    inline_ptx.sort();
    let mut expected = PRMT_INLINE_PTX
        .map(|(template, constraints, arity)| (template.to_owned(), constraints, arity));
    expected.sort();
    assert_eq!(inline_ptx, expected);
    Ok(())
}

fn lower_sync_threads(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![]);
    Operation::new(
        &mut ctx,
        nvvm::Barrier0Op::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    )
    .insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

#[test]
fn test_sync_threads_llvm_nvptx_uses_typed_intrinsic_with_fixed_zero() -> Result<(), anyhow::Error>
{
    use pliron::builtin::attributes::IntegerAttr;

    let (ctx, module_ptr) = lower_sync_threads(mir_lower::IntrinsicBackend::LlvmNvptx)?;
    let body = lowered_kernel_body(&ctx, module_ptr);
    let mut found = false;
    for op in body {
        assert!(
            Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
            "LLVM-NVPTX sync_threads must use the typed intrinsic"
        );
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
            continue;
        };
        if callee.to_string() != "llvm_nvvm_barrier_cta_sync_aligned_all" {
            continue;
        }
        let call = op.deref(&ctx);
        assert_eq!(call.get_num_operands(), 1);
        let barrier_id = call.get_operand(0);
        let defining_op = barrier_id.defining_op().expect("barrier ID is constant");
        let constant = Operation::get_op::<llvm::ConstantOp>(defining_op, &ctx)
            .expect("barrier ID is an LLVM constant");
        let value = constant.get_value(&ctx);
        let integer = value
            .downcast_ref::<IntegerAttr>()
            .expect("barrier ID is an integer");
        assert_eq!(integer.value().bw(), 32);
        assert_eq!(integer.value().to_u64(), 0);
        found = true;
    }
    assert!(found, "modern typed CTA barrier call was not emitted");

    let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
    let ir = llvm_export::export::export_module_to_string(&ctx, &module)
        .map_err(|error| anyhow::anyhow!(error))?;
    assert!(ir.contains("@llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)"));
    assert!(!ir.contains("@llvm.nvvm.barrier0"));
    Ok(())
}

#[test]
fn test_sync_threads_libnvvm_uses_exact_convergent_inline_ptx() -> Result<(), anyhow::Error> {
    let (ctx, module_ptr) = lower_sync_threads(mir_lower::IntrinsicBackend::LibNvvm)?;
    let body = lowered_kernel_body(&ctx, module_ptr);
    let mut found = false;
    for op in body {
        let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
                && let CallOpCallable::Direct(callee) = call.callee(&ctx)
            {
                assert_ne!(callee.to_string(), "llvm_nvvm_barrier_cta_sync_aligned_all");
            }
            continue;
        };
        assert_eq!(
            inline_asm
                .get_attr_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .as_deref(),
            Some("bar.sync 0;")
        );
        assert_eq!(
            inline_asm
                .get_attr_inline_asm_constraints(&ctx)
                .map(|value| String::from((*value).clone()))
                .as_deref(),
            Some("~{memory}")
        );
        assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Convergent);
        assert_eq!(op.deref(&ctx).get_num_operands(), 0);
        found = true;
    }
    assert!(found, "exact libNVVM barrier inline PTX was not emitted");

    let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
    let ir = llvm_export::export::export_module_to_string(&ctx, &module)
        .map_err(|error| anyhow::anyhow!(error))?;
    assert!(
        ir.contains("call void asm sideeffect \"bar.sync 0;\", \"~{memory}\"() #0"),
        "{ir}"
    );
    assert!(ir.contains("attributes #0 = { convergent }"), "{ir}");
    assert!(!ir.contains("@llvm.nvvm.barrier.cta.sync.aligned.all"));
    Ok(())
}

fn lower_warp_barrier(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![i32_ty.into()]);
    let member_mask = entry.deref(&ctx).get_argument(0);
    nvvm::BarWarpSyncOp::build(&mut ctx, member_mask).insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

#[test]
fn test_warp_barrier_uses_typed_intrinsic_on_both_backends() -> Result<(), anyhow::Error> {
    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        let (ctx, module_ptr) = lower_warp_barrier(backend)?;
        let body = lowered_kernel_body(&ctx, module_ptr);
        let mut calls = 0;
        for op in body {
            assert!(
                Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
                "warp barrier must use its typed intrinsic"
            );
            let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
                continue;
            };
            let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
                continue;
            };
            if callee.to_string() == "llvm_nvvm_bar_warp_sync" {
                assert_eq!(op.deref(&ctx).get_num_operands(), 1);
                assert_eq!(op.deref(&ctx).get_num_results(), 1);
                calls += 1;
            }
        }
        assert_eq!(calls, 1, "expected one typed warp-barrier call");

        let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
        let ir = llvm_export::export::export_module_to_string(&ctx, &module)
            .map_err(|error| anyhow::anyhow!(error))?;
        assert!(ir.contains("@llvm.nvvm.bar.warp.sync(i32"), "{ir}");
    }
    Ok(())
}

// =============================================================================
// cp.async lowering tests
// =============================================================================

fn lower_all_classic_cp_async(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let mut ctx = make_test_ctx();
    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let dst_ty = MirPtrType::get_generic(&mut ctx, u32_ty.into(), true);
    let src32_ty = MirPtrType::get_generic(&mut ctx, u32_ty.into(), false);
    let src8_ty = MirPtrType::get_generic(&mut ctx, u8_ty.into(), false);
    let (module_ptr, entry) = build_test_kernel(
        &mut ctx,
        vec![
            dst_ty.into(),
            src32_ty.into(),
            src8_ty.into(),
            u32_ty.into(),
        ],
    );
    let dst = entry.deref(&ctx).get_argument(0);
    let src32 = entry.deref(&ctx).get_argument(1);
    let src8 = entry.deref(&ctx).get_argument(2);
    let source_size = entry.deref(&ctx).get_argument(3);

    let zero_op = Operation::new(
        &mut ctx,
        mir::MirConstantOp::get_concrete_op_info(),
        vec![u32_ty.into()],
        vec![],
        vec![],
        0,
    );
    mir::MirConstantOp::new(zero_op).set_attr_value(
        &ctx,
        IntegerAttr::new(u32_ty, APInt::from_u32(0, NonZeroUsize::new(32).unwrap())),
    );
    zero_op.insert_at_back(entry, &ctx);
    let zero = zero_op.deref(&ctx).get_result(0);

    for copy in [
        nvvm::CpAsyncCa4Op::build(&mut ctx, dst, src32),
        nvvm::CpAsyncCa8Op::build(&mut ctx, dst, src32),
        nvvm::CpAsyncCa16Op::build(&mut ctx, dst, src32),
        nvvm::CpAsyncCaZfill4Op::build(&mut ctx, dst, src8, source_size),
        nvvm::CpAsyncCaZfill8Op::build(&mut ctx, dst, src8, source_size),
        nvvm::CpAsyncCaZfill16Op::build(&mut ctx, dst, src8, source_size),
        nvvm::CpAsyncCg16Op::build(&mut ctx, dst, src32),
        nvvm::CpAsyncCgZfill16Op::build(&mut ctx, dst, src8, source_size),
    ] {
        copy.insert_at_back(entry, &ctx);
    }
    for control in [
        nvvm::CpAsyncCommitGroupOp::build(&mut ctx),
        nvvm::CpAsyncWaitGroupOp::build(&mut ctx, zero),
        nvvm::CpAsyncWaitAllOp::build(&mut ctx),
    ] {
        control.insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

fn lower_all_cp_async_mbarrier(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let u64_ty = IntegerType::get(&ctx, 64, Signedness::Unsigned);
    let generic_ty = MirPtrType::get_generic(&mut ctx, u64_ty.into(), true);
    let shared_ty = MirPtrType::get_shared(&mut ctx, u64_ty.into(), true);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![generic_ty.into(), shared_ty.into()]);
    let generic = entry.deref(&ctx).get_argument(0);
    let shared = entry.deref(&ctx).get_argument(1);

    for bridge in [
        nvvm::CpAsyncMbarrierArriveOp::build(&mut ctx, shared),
        nvvm::CpAsyncMbarrierArriveSharedOp::build(&mut ctx, generic),
        nvvm::CpAsyncMbarrierArriveNoIncOp::build(&mut ctx, shared),
        nvvm::CpAsyncMbarrierArriveNoIncSharedOp::build(&mut ctx, generic),
    ] {
        bridge.insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

#[test]
fn test_generated_cp_async_mbarrier_preserves_backend_and_address_routes()
-> Result<(), anyhow::Error> {
    use llvm_export::types::PointerType;
    use pliron::r#type::Typed;

    let typed = [
        ("llvm_nvvm_cp_async_mbarrier_arrive", 0),
        ("llvm_nvvm_cp_async_mbarrier_arrive_shared", 3),
        ("llvm_nvvm_cp_async_mbarrier_arrive_noinc", 0),
        ("llvm_nvvm_cp_async_mbarrier_arrive_noinc_shared", 3),
    ];
    let templates = [
        ("cp.async.mbarrier.arrive.b64 [$0];", 0),
        ("cp.async.mbarrier.arrive.shared.b64 [$0];", 3),
        ("cp.async.mbarrier.arrive.noinc.b64 [$0];", 0),
        ("cp.async.mbarrier.arrive.noinc.shared.b64 [$0];", 3),
    ];

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        let (ctx, module_ptr) = lower_all_cp_async_mbarrier(backend)?;
        let mut call_counts = [0usize; 4];
        let mut asm_counts = [0usize; 4];

        for op in lowered_kernel_body(&ctx, module_ptr) {
            if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) {
                let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
                    continue;
                };
                let callee = callee.to_string();
                let Some(index) = typed.iter().position(|(name, _)| callee == *name) else {
                    continue;
                };
                call_counts[index] += 1;
                let pointer = op.deref(&ctx).get_operand(0).get_type(&ctx);
                assert_eq!(
                    pointer
                        .deref(&ctx)
                        .downcast_ref::<PointerType>()
                        .unwrap()
                        .address_space(),
                    typed[index].1
                );
            }
            if let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) {
                let template = inline_asm
                    .get_attr_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .unwrap();
                let Some(index) = templates
                    .iter()
                    .position(|(expected, _)| template == *expected)
                else {
                    continue;
                };
                asm_counts[index] += 1;
                assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Convergent);
                assert!(
                    inline_asm
                        .get_attr_inline_asm_convergent(&ctx)
                        .is_some_and(|value| bool::from((*value).clone()))
                );
                assert_eq!(
                    inline_asm
                        .get_attr_inline_asm_constraints(&ctx)
                        .map(|value| String::from((*value).clone()))
                        .as_deref(),
                    Some("l,~{memory}")
                );
                let pointer = op.deref(&ctx).get_operand(0).get_type(&ctx);
                assert_eq!(
                    pointer
                        .deref(&ctx)
                        .downcast_ref::<PointerType>()
                        .unwrap()
                        .address_space(),
                    templates[index].1
                );
            }
        }

        let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
        let ir = llvm_export::export::export_module_to_string(&ctx, &module)
            .map_err(|error| anyhow::anyhow!(error))?;
        match backend {
            mir_lower::IntrinsicBackend::LlvmNvptx => {
                assert_eq!(call_counts, [1; 4]);
                assert_eq!(asm_counts, [0; 4]);
                assert!(
                    ir.contains("@llvm.nvvm.cp.async.mbarrier.arrive(ptr"),
                    "{ir}"
                );
                assert!(
                    ir.contains("@llvm.nvvm.cp.async.mbarrier.arrive.shared(ptr addrspace(3)"),
                    "{ir}"
                );
                assert!(
                    ir.contains("@llvm.nvvm.cp.async.mbarrier.arrive.noinc(ptr"),
                    "{ir}"
                );
                assert!(
                    ir.contains(
                        "@llvm.nvvm.cp.async.mbarrier.arrive.noinc.shared(ptr addrspace(3)"
                    ),
                    "{ir}"
                );
                assert!(!ir.contains("cp.async.mbarrier.arrive.b64 [$0]"), "{ir}");
            }
            mir_lower::IntrinsicBackend::LibNvvm => {
                assert_eq!(call_counts, [0; 4]);
                assert_eq!(asm_counts, [1; 4]);
                assert!(!ir.contains("@llvm.nvvm.cp.async.mbarrier"), "{ir}");
                for (template, _) in templates {
                    assert!(ir.contains(template), "{ir}");
                }
                assert!(ir.contains("asm sideeffect"), "{ir}");
                assert!(ir.contains("convergent"), "{ir}");
            }
        }
    }
    Ok(())
}

#[test]
fn test_generated_cp_async_llvm_nvptx_uses_all_typed_intrinsics() -> Result<(), anyhow::Error> {
    use llvm_export::types::PointerType;
    use pliron::r#type::Typed;

    let (ctx, module_ptr) = lower_all_classic_cp_async(mir_lower::IntrinsicBackend::LlvmNvptx)?;
    let mut found = Vec::new();
    for op in lowered_kernel_body(&ctx, module_ptr) {
        assert!(
            Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
            "LLVM-NVPTX cp.async must use typed intrinsics"
        );
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
            continue;
        };
        let callee = callee.to_string();
        if !callee.starts_with("llvm_nvvm_cp_async") {
            continue;
        }
        if callee.contains("shared_global") {
            let call = op.deref(&ctx);
            let destination_ty = call.get_operand(0).get_type(&ctx);
            let source_ty = call.get_operand(1).get_type(&ctx);
            assert_eq!(
                destination_ty
                    .deref(&ctx)
                    .downcast_ref::<PointerType>()
                    .unwrap()
                    .address_space(),
                3
            );
            assert_eq!(
                source_ty
                    .deref(&ctx)
                    .downcast_ref::<PointerType>()
                    .unwrap()
                    .address_space(),
                1
            );
        }
        found.push(callee);
    }
    found.sort();
    let mut expected = [
        "llvm_nvvm_cp_async_ca_shared_global_4",
        "llvm_nvvm_cp_async_ca_shared_global_4_s",
        "llvm_nvvm_cp_async_ca_shared_global_8",
        "llvm_nvvm_cp_async_ca_shared_global_8_s",
        "llvm_nvvm_cp_async_ca_shared_global_16",
        "llvm_nvvm_cp_async_ca_shared_global_16_s",
        "llvm_nvvm_cp_async_cg_shared_global_16",
        "llvm_nvvm_cp_async_cg_shared_global_16_s",
        "llvm_nvvm_cp_async_commit_group",
        "llvm_nvvm_cp_async_wait_all",
        "llvm_nvvm_cp_async_wait_group",
    ]
    .map(str::to_owned);
    expected.sort();
    assert_eq!(found, expected);
    Ok(())
}

#[test]
fn test_generated_cp_async_libnvvm_uses_all_exact_inline_ptx() -> Result<(), anyhow::Error> {
    let (ctx, module_ptr) = lower_all_classic_cp_async(mir_lower::IntrinsicBackend::LibNvvm)?;
    let mut lowered = Vec::new();
    for op in lowered_kernel_body(&ctx, module_ptr) {
        let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            continue;
        };
        assert_eq!(llvm::asm_kind(&ctx, &asm), llvm::AsmKind::SideEffect);
        assert!(
            asm.get_attr_inline_asm_convergent(&ctx)
                .is_some_and(|value| !bool::from((*value).clone()))
        );
        lowered.push((
            asm.get_attr_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .unwrap(),
            asm.get_attr_inline_asm_constraints(&ctx)
                .map(|value| String::from((*value).clone()))
                .unwrap(),
        ));
    }

    let expected = [
        (
            "cp.async.ca.shared.global [%smem32], [%gmem64], 4;",
            "l,l,~{memory}",
        ),
        (
            "cp.async.ca.shared.global [%smem32], [%gmem64], 8;",
            "l,l,~{memory}",
        ),
        (
            "cp.async.ca.shared.global [%smem32], [%gmem64], 16;",
            "l,l,~{memory}",
        ),
        (
            "cp.async.ca.shared.global [%smem32], [%gmem64], 4, $2;",
            "l,l,r,~{memory}",
        ),
        (
            "cp.async.ca.shared.global [%smem32], [%gmem64], 8, $2;",
            "l,l,r,~{memory}",
        ),
        (
            "cp.async.ca.shared.global [%smem32], [%gmem64], 16, $2;",
            "l,l,r,~{memory}",
        ),
        (
            "cp.async.cg.shared.global [%smem32], [%gmem64], 16;",
            "l,l,~{memory}",
        ),
        (
            "cp.async.cg.shared.global [%smem32], [%gmem64], 16, $2;",
            "l,l,r,~{memory}",
        ),
        ("cp.async.commit_group;", "~{memory}"),
        ("cp.async.wait_group $0;", "n,~{memory}"),
        ("cp.async.wait_all;", "~{memory}"),
    ];
    for (instruction, constraints) in expected {
        assert_eq!(
            lowered
                .iter()
                .filter(|(template, actual_constraints)| {
                    template.contains(instruction) && actual_constraints == constraints
                })
                .count(),
            1,
            "missing exact `{instruction}`"
        );
    }
    assert_eq!(lowered.len(), expected.len());
    Ok(())
}

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

    mir_lower::lower_mir_to_llvm_with_options(
        ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: mir_lower::IntrinsicBackend::LibNvvm,
            ..Default::default()
        },
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

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

    mir_lower::lower_mir_to_llvm_with_options(
        ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: mir_lower::IntrinsicBackend::LibNvvm,
            ..Default::default()
        },
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

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
// Generated packed arithmetic lowering tests
// =============================================================================

#[test]
fn test_generated_packed_arithmetic_lowers_to_exact_pure_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![i32_ty.into(), i32_ty.into(), i32_ty.into()]);

    type OpInfo = (
        fn(pliron::context::Ptr<Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    );
    let cases: [(OpInfo, usize, &str, &str); 18] = [
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
        (
            nvvm::FmaF16x2Op::get_concrete_op_info(),
            3,
            "fma.rn.f16x2 $0, $1, $2, $3;",
            "=r,r,r,r",
        ),
        (
            nvvm::FmaReluF16x2Op::get_concrete_op_info(),
            3,
            "fma.rn.relu.f16x2 $0, $1, $2, $3;",
            "=r,r,r,r",
        ),
        (
            nvvm::AddF16x2Op::get_concrete_op_info(),
            2,
            "add.rn.f16x2 $0, $1, $2;",
            "=r,r,r",
        ),
        (
            nvvm::SubF16x2Op::get_concrete_op_info(),
            2,
            "sub.rn.f16x2 $0, $1, $2;",
            "=r,r,r",
        ),
        (
            nvvm::MulF16x2Op::get_concrete_op_info(),
            2,
            "mul.rn.f16x2 $0, $1, $2;",
            "=r,r,r",
        ),
        (
            nvvm::MinF16x2Op::get_concrete_op_info(),
            2,
            "min.f16x2 $0, $1, $2;",
            "=r,r,r",
        ),
        (
            nvvm::MaxF16x2Op::get_concrete_op_info(),
            2,
            "max.f16x2 $0, $1, $2;",
            "=r,r,r",
        ),
        (
            nvvm::NegF16x2Op::get_concrete_op_info(),
            1,
            "neg.f16x2 $0, $1;",
            "=r,r",
        ),
        (
            nvvm::AbsF16x2Op::get_concrete_op_info(),
            1,
            "abs.f16x2 $0, $1;",
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
                        .expect("packed inline asm must have a template"),
                    inline_asm
                        .get_attr_inline_asm_constraints(&ctx)
                        .map(|s| String::from((*s).clone()))
                        .expect("packed inline asm must have constraints"),
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
        "each packed operation must lower to exactly one inline-asm op"
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
fn test_generated_packed_conversions_lower_to_exact_pure_inline_asm() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::{FP32Type, IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![f32_ty.into(), f32_ty.into()]);
    let low = entry.deref(&ctx).get_argument(0);
    let high = entry.deref(&ctx).get_argument(1);

    type OpInfo = (
        fn(pliron::context::Ptr<Operation>) -> pliron::op::OpObj,
        std::any::TypeId,
    );
    let cases: [(OpInfo, &str); 6] = [
        (
            nvvm::CvtF32x2Bf16x2Op::get_concrete_op_info(),
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
        ),
        (
            nvvm::CvtF16x2F32Op::get_concrete_op_info(),
            "cvt.rn.f16x2.f32 $0, $2, $1;",
        ),
        (
            nvvm::CvtRzF16x2F32Op::get_concrete_op_info(),
            "cvt.rz.f16x2.f32 $0, $2, $1;",
        ),
        (
            nvvm::CvtRnReluF16x2F32Op::get_concrete_op_info(),
            "cvt.rn.relu.f16x2.f32 $0, $2, $1;",
        ),
        (
            nvvm::CvtRnReluBf16x2F32Op::get_concrete_op_info(),
            "cvt.rn.relu.bf16x2.f32 $0, $2, $1;",
        ),
        (
            nvvm::CvtRzBf16x2F32Op::get_concrete_op_info(),
            "cvt.rz.bf16x2.f32 $0, $2, $1;",
        ),
    ];
    for &(op_info, _) in &cases {
        let op = Operation::new(
            &mut ctx,
            op_info,
            vec![i32_ty.into()],
            vec![low, high],
            vec![],
            0,
        );
        op.insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr).map_err(|e| anyhow::anyhow!("{}", e))?;

    let mut lowered = Vec::new();
    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    for op in module_block.deref(&ctx).iter(&ctx) {
        let Some(func) = Operation::get_op::<llvm::FuncOp>(op, &ctx) else {
            continue;
        };
        if func.get_symbol_name(&ctx).to_string() != "kernel_func" {
            continue;
        }
        let region = func.get_operation().deref(&ctx).get_region(0);
        for block in region.deref(&ctx).iter(&ctx) {
            for body_op in block.deref(&ctx).iter(&ctx) {
                let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(body_op, &ctx) else {
                    continue;
                };
                let template = asm
                    .get_attr_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .expect("packed conversion must have an asm template");
                if !template.starts_with("cvt.") {
                    continue;
                }
                lowered.push((
                    template,
                    asm.get_attr_inline_asm_constraints(&ctx)
                        .map(|value| String::from((*value).clone())),
                    asm.get_attr_inline_asm_convergent(&ctx)
                        .map(|value| bool::from((*value).clone())),
                    llvm::asm_kind_opt(&ctx, &asm),
                    asm.get_operation().deref(&ctx).operands().count(),
                    asm.get_operation().deref(&ctx).get_num_results(),
                ));
            }
        }
    }

    assert_eq!(
        lowered.len(),
        cases.len(),
        "each packed conversion must lower to one inline-asm op"
    );
    for &(_, expected_template) in &cases {
        let matches: Vec<_> = lowered
            .iter()
            .filter(|(template, _, _, _, _, _)| template == expected_template)
            .collect();
        assert_eq!(
            matches.len(),
            1,
            "expected one exact `{expected_template}` lowering"
        );
        let (_, constraints, convergent, kind, operands, results) = matches[0];
        assert_eq!(
            constraints.as_deref(),
            Some("=r,f,f"),
            "{expected_template}"
        );
        assert_eq!(*convergent, Some(false), "{expected_template}");
        assert_eq!(*kind, Some(llvm::AsmKind::Pure), "{expected_template}");
        assert_eq!(*operands, 2, "{expected_template} input arity");
        assert_eq!(*results, 1, "{expected_template} result arity");
    }

    Ok(())
}

const FP8_CONVERSION_INTRINSICS: [&str; 4] = [
    "llvm_nvvm_ff_to_e4m3x2_rn",
    "llvm_nvvm_ff_to_e4m3x2_rn_relu",
    "llvm_nvvm_ff_to_e5m2x2_rn",
    "llvm_nvvm_ff_to_e5m2x2_rn_relu",
];

const FP8_CONVERSION_PTX: [&str; 4] = [
    "cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;",
    "cvt.rn.satfinite.relu.e4m3x2.f32 $0, $2, $1;",
    "cvt.rn.satfinite.e5m2x2.f32 $0, $2, $1;",
    "cvt.rn.satfinite.relu.e5m2x2.f32 $0, $2, $1;",
];

fn lower_all_fp8_conversions(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use pliron::builtin::types::FP32Type;

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![f32_ty.into(), f32_ty.into()]);
    let low = entry.deref(&ctx).get_argument(0);
    let high = entry.deref(&ctx).get_argument(1);

    nvvm::CvtRnSatfiniteE4m3x2F32Op::build(&mut ctx, low, high).insert_at_back(entry, &ctx);
    nvvm::CvtRnSatfiniteReluE4m3x2F32Op::build(&mut ctx, low, high).insert_at_back(entry, &ctx);
    nvvm::CvtRnSatfiniteE5m2x2F32Op::build(&mut ctx, low, high).insert_at_back(entry, &ctx);
    nvvm::CvtRnSatfiniteReluE5m2x2F32Op::build(&mut ctx, low, high).insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

#[test]
fn test_fp8_conversions_llvm_nvptx_use_exact_typed_calls() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::IntegerType;
    use pliron::r#type::Typed;

    let (ctx, module_ptr) = lower_all_fp8_conversions(mir_lower::IntrinsicBackend::LlvmNvptx)?;
    let mut calls = Vec::new();
    for op in lowered_kernel_body(&ctx, module_ptr) {
        assert!(
            Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
            "LLVM-NVPTX FP8 conversions must use typed intrinsics"
        );
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
            continue;
        };
        let callee = callee.to_string();
        if !FP8_CONVERSION_INTRINSICS.contains(&callee.as_str()) {
            continue;
        }
        let call_op = call.get_operation();
        assert_eq!(call_op.deref(&ctx).get_num_operands(), 2);
        assert_eq!(call_op.deref(&ctx).get_num_results(), 1);
        let block = call_op.deref(&ctx).get_parent_block().unwrap();
        assert_eq!(
            call_op.deref(&ctx).get_operand(0),
            block.deref(&ctx).get_argument(1)
        );
        assert_eq!(
            call_op.deref(&ctx).get_operand(1),
            block.deref(&ctx).get_argument(0)
        );
        let result_ty = call_op.deref(&ctx).get_result(0).get_type(&ctx);
        assert_eq!(
            result_ty
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .expect("FP8 conversion result is an integer")
                .width(),
            16
        );
        calls.push(callee);
    }
    calls.sort();
    let mut expected = FP8_CONVERSION_INTRINSICS.map(str::to_owned);
    expected.sort();
    assert_eq!(calls, expected);
    Ok(())
}

#[test]
fn test_fp8_conversions_libnvvm_use_exact_pure_inline_ptx() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::IntegerType;
    use pliron::r#type::Typed;

    let (ctx, module_ptr) = lower_all_fp8_conversions(mir_lower::IntrinsicBackend::LibNvvm)?;
    let mut templates = Vec::new();
    for op in lowered_kernel_body(&ctx, module_ptr) {
        if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
            && let CallOpCallable::Direct(callee) = call.callee(&ctx)
        {
            assert!(
                !FP8_CONVERSION_INTRINSICS.contains(&callee.to_string().as_str()),
                "libNVVM FP8 conversions must not use typed intrinsics"
            );
        }
        let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            continue;
        };
        let template = inline_asm
            .get_attr_inline_asm_template(&ctx)
            .map(|value| String::from((*value).clone()))
            .unwrap_or_default();
        if !FP8_CONVERSION_PTX.contains(&template.as_str()) {
            continue;
        }
        assert_eq!(
            inline_asm
                .get_attr_inline_asm_constraints(&ctx)
                .map(|value| String::from((*value).clone()))
                .as_deref(),
            Some("=h,f,f")
        );
        assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Pure);
        assert_eq!(
            inline_asm
                .get_attr_inline_asm_convergent(&ctx)
                .map(|value| bool::from((*value).clone())),
            Some(false)
        );
        let asm_op = inline_asm.get_operation();
        assert_eq!(asm_op.deref(&ctx).get_num_operands(), 2);
        assert_eq!(asm_op.deref(&ctx).get_num_results(), 1);
        let result_ty = asm_op.deref(&ctx).get_result(0).get_type(&ctx);
        assert_eq!(
            result_ty
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .expect("FP8 conversion result is an integer")
                .width(),
            16
        );
        templates.push(template);
    }
    templates.sort();
    let mut expected = FP8_CONVERSION_PTX.map(str::to_owned);
    expected.sort();
    assert_eq!(templates, expected);
    Ok(())
}

const SCALAR_CONVERSION_INTRINSICS: [&str; 10] = [
    "llvm_nvvm_f2tf32_rna",
    "llvm_nvvm_f2tf32_rna_satfinite",
    "llvm_nvvm_f2tf32_rn",
    "llvm_nvvm_f2tf32_rn_relu",
    "llvm_nvvm_f2tf32_rn_satfinite",
    "llvm_nvvm_f2tf32_rn_relu_satfinite",
    "llvm_nvvm_f2tf32_rz",
    "llvm_nvvm_f2tf32_rz_relu",
    "llvm_nvvm_f2tf32_rz_satfinite",
    "llvm_nvvm_f2tf32_rz_relu_satfinite",
];

const SCALAR_CONVERSION_PTX: [&str; 10] = [
    "cvt.rna.tf32.f32 $0, $1;",
    "cvt.rna.satfinite.tf32.f32 $0, $1;",
    "cvt.rn.tf32.f32 $0, $1;",
    "cvt.rn.relu.tf32.f32 $0, $1;",
    "cvt.rn.satfinite.tf32.f32 $0, $1;",
    "cvt.rn.relu.satfinite.tf32.f32 $0, $1;",
    "cvt.rz.tf32.f32 $0, $1;",
    "cvt.rz.relu.tf32.f32 $0, $1;",
    "cvt.rz.satfinite.tf32.f32 $0, $1;",
    "cvt.rz.relu.satfinite.tf32.f32 $0, $1;",
];

fn lower_all_scalar_conversions(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use pliron::builtin::types::FP32Type;

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![f32_ty.into()]);
    let value = entry.deref(&ctx).get_argument(0);

    let variants = [
        (
            nvvm::ScalarConversionRoundingAttr::NearestAway,
            nvvm::ScalarConversionSaturationAttr::None,
        ),
        (
            nvvm::ScalarConversionRoundingAttr::NearestAway,
            nvvm::ScalarConversionSaturationAttr::Satfinite,
        ),
        (
            nvvm::ScalarConversionRoundingAttr::NearestEven,
            nvvm::ScalarConversionSaturationAttr::None,
        ),
        (
            nvvm::ScalarConversionRoundingAttr::NearestEven,
            nvvm::ScalarConversionSaturationAttr::Relu,
        ),
        (
            nvvm::ScalarConversionRoundingAttr::NearestEven,
            nvvm::ScalarConversionSaturationAttr::Satfinite,
        ),
        (
            nvvm::ScalarConversionRoundingAttr::NearestEven,
            nvvm::ScalarConversionSaturationAttr::ReluSatfinite,
        ),
        (
            nvvm::ScalarConversionRoundingAttr::TowardZero,
            nvvm::ScalarConversionSaturationAttr::None,
        ),
        (
            nvvm::ScalarConversionRoundingAttr::TowardZero,
            nvvm::ScalarConversionSaturationAttr::Relu,
        ),
        (
            nvvm::ScalarConversionRoundingAttr::TowardZero,
            nvvm::ScalarConversionSaturationAttr::Satfinite,
        ),
        (
            nvvm::ScalarConversionRoundingAttr::TowardZero,
            nvvm::ScalarConversionSaturationAttr::ReluSatfinite,
        ),
    ];
    for (rounding, saturation) in variants {
        nvvm::ScalarConversionOp::build(&mut ctx, value, rounding, saturation)
            .insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

#[test]
fn test_scalar_conversions_llvm_nvptx_use_exact_typed_calls() -> Result<(), anyhow::Error> {
    use llvm_export::types as llvm_types;
    use pliron::builtin::type_interfaces::FunctionTypeInterface;
    use pliron::builtin::types::{FP32Type, IntegerType};

    let (ctx, module_ptr) = lower_all_scalar_conversions(mir_lower::IntrinsicBackend::LlvmNvptx)?;
    let mut calls = Vec::new();
    for op in lowered_kernel_body(&ctx, module_ptr) {
        assert!(
            Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
            "LLVM-NVPTX scalar conversions must use typed intrinsics"
        );
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
            continue;
        };
        let callee = callee.to_string();
        if !SCALAR_CONVERSION_INTRINSICS.contains(&callee.as_str()) {
            continue;
        }

        let call_op = call.get_operation();
        assert_eq!(call_op.deref(&ctx).get_num_operands(), 1, "{callee}");
        assert_eq!(call_op.deref(&ctx).get_num_results(), 1, "{callee}");
        let block = call_op.deref(&ctx).get_parent_block().unwrap();
        assert_eq!(
            call_op.deref(&ctx).get_operand(0),
            block.deref(&ctx).get_argument(0),
            "{callee} source operand"
        );

        let function_ty = call.callee_type(&ctx);
        let function_ty = function_ty.deref(&ctx);
        let function_ty = function_ty
            .downcast_ref::<llvm_types::FuncType>()
            .expect("scalar conversion callee has an LLVM function type");
        assert_eq!(function_ty.arg_types().len(), 1, "{callee}");
        assert!(
            function_ty.arg_types()[0]
                .deref(&ctx)
                .downcast_ref::<FP32Type>()
                .is_some(),
            "{callee} source type"
        );
        assert_eq!(
            function_ty
                .result_type()
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .expect("scalar conversion result is an integer")
                .width(),
            32,
            "{callee} result type"
        );
        calls.push(callee);
    }
    calls.sort();
    let mut expected = SCALAR_CONVERSION_INTRINSICS.map(str::to_owned);
    expected.sort();
    assert_eq!(calls, expected);

    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    let mut declarations: Vec<_> = module_block
        .deref(&ctx)
        .iter(&ctx)
        .filter_map(|op| Operation::get_op::<llvm::FuncOp>(op, &ctx))
        .map(|func| func.get_symbol_name(&ctx).to_string())
        .filter(|name| SCALAR_CONVERSION_INTRINSICS.contains(&name.as_str()))
        .collect();
    declarations.sort();
    assert_eq!(declarations, expected);
    Ok(())
}

#[test]
fn test_scalar_conversions_libnvvm_use_exact_pure_inline_ptx() -> Result<(), anyhow::Error> {
    use pliron::builtin::types::IntegerType;
    use pliron::r#type::Typed;

    let (ctx, module_ptr) = lower_all_scalar_conversions(mir_lower::IntrinsicBackend::LibNvvm)?;
    let mut templates = Vec::new();
    for op in lowered_kernel_body(&ctx, module_ptr) {
        if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
            && let CallOpCallable::Direct(callee) = call.callee(&ctx)
        {
            assert!(
                !SCALAR_CONVERSION_INTRINSICS.contains(&callee.to_string().as_str()),
                "libNVVM scalar conversions must not use typed intrinsics"
            );
        }
        let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            continue;
        };
        let template = inline_asm
            .get_attr_inline_asm_template(&ctx)
            .map(|value| String::from((*value).clone()))
            .unwrap_or_default();
        assert!(
            SCALAR_CONVERSION_PTX.contains(&template.as_str()),
            "unexpected scalar conversion template `{template}`"
        );
        assert_eq!(
            inline_asm
                .get_attr_inline_asm_constraints(&ctx)
                .map(|value| String::from((*value).clone()))
                .as_deref(),
            Some("=r,f"),
            "{template}"
        );
        assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Pure);
        assert_eq!(
            inline_asm
                .get_attr_inline_asm_convergent(&ctx)
                .map(|value| bool::from((*value).clone())),
            Some(false),
            "{template}"
        );

        let asm_op = inline_asm.get_operation();
        assert_eq!(asm_op.deref(&ctx).get_num_operands(), 1, "{template}");
        assert_eq!(asm_op.deref(&ctx).get_num_results(), 1, "{template}");
        let block = asm_op.deref(&ctx).get_parent_block().unwrap();
        assert_eq!(
            asm_op.deref(&ctx).get_operand(0),
            block.deref(&ctx).get_argument(0),
            "{template} source operand"
        );
        assert_eq!(
            asm_op
                .deref(&ctx)
                .get_result(0)
                .get_type(&ctx)
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .expect("scalar conversion result is an integer")
                .width(),
            32,
            "{template} result type"
        );
        templates.push(template);
    }
    templates.sort();
    let mut expected = SCALAR_CONVERSION_PTX.map(str::to_owned);
    expected.sort();
    assert_eq!(templates, expected);
    Ok(())
}

#[test]
fn test_scalar_conversion_invalid_variant_fails_closed() {
    use pliron::builtin::types::FP32Type;

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![f32_ty.into()]);
    let value = entry.deref(&ctx).get_argument(0);
    nvvm::ScalarConversionOp::build(
        &mut ctx,
        value,
        nvvm::ScalarConversionRoundingAttr::NearestAway,
        nvvm::ScalarConversionSaturationAttr::Relu,
    )
    .insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    let result = mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: mir_lower::IntrinsicBackend::LlvmNvptx,
            ..Default::default()
        },
    );
    let error = result.expect_err("unadmitted scalar conversion must not lower");
    assert!(
        error.to_string().contains("scalar_conversion"),
        "unexpected error: {error}"
    );
}

fn lower_representative_scalar_arithmetic(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use pliron::builtin::types::FP32Type;

    let mut ctx = make_test_ctx();
    let f32_ty = FP32Type::get(&ctx);
    let (module_ptr, entry) =
        build_test_kernel(&mut ctx, vec![f32_ty.into(), f32_ty.into(), f32_ty.into()]);
    let args: Vec<_> = (0..3)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect();

    for (operation, saturation, operands) in [
        (
            nvvm::ScalarArithmeticOperationAttr::Add,
            nvvm::ScalarArithmeticSaturationAttr::None,
            vec![args[0], args[1]],
        ),
        (
            nvvm::ScalarArithmeticOperationAttr::Add,
            nvvm::ScalarArithmeticSaturationAttr::Sat,
            vec![args[0], args[1]],
        ),
        (
            nvvm::ScalarArithmeticOperationAttr::Fma,
            nvvm::ScalarArithmeticSaturationAttr::None,
            args.clone(),
        ),
        (
            nvvm::ScalarArithmeticOperationAttr::Fma,
            nvvm::ScalarArithmeticSaturationAttr::Sat,
            args,
        ),
    ] {
        nvvm::ScalarArithmeticOp::build(
            &mut ctx,
            operands,
            nvvm::ScalarArithmeticFormatAttr::F32,
            operation,
            nvvm::ScalarArithmeticRoundingAttr::Rn,
            nvvm::ScalarArithmeticSubnormalAttr::Preserve,
            saturation,
        )
        .insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;
    Ok((ctx, module_ptr))
}

fn is_representative_scalar_arithmetic_intrinsic(name: &str) -> bool {
    name.starts_with("llvm_nvvm_add_rn") || name.starts_with("llvm_nvvm_fma_rn")
}

#[test]
fn test_scalar_arithmetic_llvm_uses_inline_ptx_only_for_saturation() -> Result<(), anyhow::Error> {
    let (ctx, module_ptr) =
        lower_representative_scalar_arithmetic(mir_lower::IntrinsicBackend::LlvmNvptx)?;
    let mut calls = Vec::new();
    let mut inline_ptx = Vec::new();
    for op in lowered_kernel_body(&ctx, module_ptr) {
        if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
            && let CallOpCallable::Direct(callee) = call.callee(&ctx)
            && is_representative_scalar_arithmetic_intrinsic(&callee.to_string())
        {
            calls.push(callee.to_string());
        }
        let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            continue;
        };
        let template = inline_asm
            .get_attr_inline_asm_template(&ctx)
            .map(|value| String::from((*value).clone()))
            .unwrap_or_default();
        let constraints = inline_asm
            .get_attr_inline_asm_constraints(&ctx)
            .map(|value| String::from((*value).clone()))
            .unwrap_or_default();
        assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Pure);
        assert_eq!(
            inline_asm
                .get_attr_inline_asm_convergent(&ctx)
                .map(|value| bool::from((*value).clone())),
            Some(false)
        );
        inline_ptx.push((template, constraints));
    }

    calls.sort();
    assert_eq!(calls, ["llvm_nvvm_add_rn_f", "llvm_nvvm_fma_rn_f"]);
    inline_ptx.sort();
    assert_eq!(
        inline_ptx,
        [
            ("add.rn.sat.f32 $0, $1, $2;".into(), "=f,f,f".into()),
            ("fma.rn.sat.f32 $0, $1, $2, $3;".into(), "=f,f,f,f".into(),),
        ]
    );

    let module_region = module_ptr.deref(&ctx).get_region(0);
    let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
    let mut declarations: Vec<_> = module_block
        .deref(&ctx)
        .iter(&ctx)
        .filter_map(|op| Operation::get_op::<llvm::FuncOp>(op, &ctx))
        .map(|func| func.get_symbol_name(&ctx).to_string())
        .filter(|name| is_representative_scalar_arithmetic_intrinsic(name))
        .collect();
    declarations.sort();
    assert_eq!(declarations, calls);
    Ok(())
}

#[test]
fn test_scalar_arithmetic_libnvvm_uses_exact_inline_ptx() -> Result<(), anyhow::Error> {
    let (ctx, module_ptr) =
        lower_representative_scalar_arithmetic(mir_lower::IntrinsicBackend::LibNvvm)?;
    let mut calls = Vec::new();
    let mut inline_ptx = Vec::new();
    for op in lowered_kernel_body(&ctx, module_ptr) {
        if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
            && let CallOpCallable::Direct(callee) = call.callee(&ctx)
            && is_representative_scalar_arithmetic_intrinsic(&callee.to_string())
        {
            calls.push(callee.to_string());
        }
        let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            continue;
        };
        let template = inline_asm
            .get_attr_inline_asm_template(&ctx)
            .map(|value| String::from((*value).clone()))
            .unwrap_or_default();
        let constraints = inline_asm
            .get_attr_inline_asm_constraints(&ctx)
            .map(|value| String::from((*value).clone()))
            .unwrap_or_default();
        assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Pure);
        inline_ptx.push((template, constraints));
    }

    assert!(calls.is_empty());
    inline_ptx.sort();
    assert_eq!(
        inline_ptx,
        [
            ("add.rn.f32 $0, $1, $2;".into(), "=f,f,f".into()),
            ("add.rn.sat.f32 $0, $1, $2;".into(), "=f,f,f".into()),
            ("fma.rn.f32 $0, $1, $2, $3;".into(), "=f,f,f,f".into(),),
            ("fma.rn.sat.f32 $0, $1, $2, $3;".into(), "=f,f,f,f".into(),),
        ]
    );
    Ok(())
}

const STMATRIX_TYPED_INTRINSICS: [&str; 4] = [
    "llvm_nvvm_stmatrix_sync_aligned_m8n8_x2_b16_p3",
    "llvm_nvvm_stmatrix_sync_aligned_m8n8_x2_trans_b16_p3",
    "llvm_nvvm_stmatrix_sync_aligned_m8n8_x4_b16_p3",
    "llvm_nvvm_stmatrix_sync_aligned_m8n8_x4_trans_b16_p3",
];

const STMATRIX_PTX: [(&str, &str); 4] = [
    (
        "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x2.shared.b16 [%ptr32], {$1, $2}; }",
        "l,r,r,~{memory}",
    ),
    (
        "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%ptr32], {$1, $2}; }",
        "l,r,r,~{memory}",
    ),
    (
        "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x4.shared.b16 [%ptr32], {$1, $2, $3, $4}; }",
        "l,r,r,r,r,~{memory}",
    ),
    (
        "{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%ptr32], {$1, $2, $3, $4}; }",
        "l,r,r,r,r,~{memory}",
    ),
];

fn lower_all_stmatrix_forms(
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
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
        (
            nvvm::StmatrixM8n8X2Op::get_concrete_op_info(),
            args[..3].to_vec(),
        ),
        (
            nvvm::StmatrixM8n8X2TransOp::get_concrete_op_info(),
            args[..3].to_vec(),
        ),
        (nvvm::StmatrixM8n8X4Op::get_concrete_op_info(), args.clone()),
        (nvvm::StmatrixM8n8X4TransOp::get_concrete_op_info(), args),
    ] {
        Operation::new(&mut ctx, op_info, vec![], operands, vec![], 0).insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;

    Ok((ctx, module_ptr))
}

#[test]
fn test_stmatrix_llvm_nvptx_uses_exact_typed_p3_intrinsics() -> Result<(), anyhow::Error> {
    use llvm_export::types as llvm_types;
    use pliron::r#type::Typed;

    let (ctx, module_ptr) = lower_all_stmatrix_forms(mir_lower::IntrinsicBackend::LlvmNvptx)?;
    let mut callees = Vec::new();

    for op in lowered_kernel_body(&ctx, module_ptr) {
        assert!(
            Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
            "LLVM-NVPTX stmatrix lowering must not emit inline PTX"
        );
        let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
            continue;
        };
        let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
            continue;
        };
        let callee = callee.to_string();
        if !STMATRIX_TYPED_INTRINSICS.contains(&callee.as_str()) {
            continue;
        }

        let call_op = call.get_operation().deref(&ctx);
        assert!(matches!(call_op.get_num_operands(), 3 | 5));
        assert_eq!(call_op.get_num_results(), 1);
        let pointer_ty = call_op.get_operand(0).get_type(&ctx);
        let pointer_ty = pointer_ty.deref(&ctx);
        let pointer_ty = pointer_ty
            .downcast_ref::<llvm_types::PointerType>()
            .expect("stmatrix first argument is a pointer");
        assert_eq!(pointer_ty.address_space(), 3);
        callees.push(callee);
    }

    callees.sort();
    let mut expected = STMATRIX_TYPED_INTRINSICS.map(str::to_owned);
    expected.sort();
    assert_eq!(callees, expected);

    let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
    let ir = llvm_export::export::export_module_to_string(&ctx, &module)
        .expect("typed stmatrix module exports to LLVM IR");
    for intrinsic in STMATRIX_TYPED_INTRINSICS {
        let dotted = intrinsic.replace('_', ".");
        assert!(
            ir.contains(&format!("@{dotted}(ptr addrspace(3)")),
            "missing exact typed stmatrix declaration:\n{ir}"
        );
    }
    assert!(!ir.contains("asm sideeffect"), "{ir}");
    Ok(())
}

#[test]
fn test_stmatrix_libnvvm_uses_exact_convergent_memory_asm() -> Result<(), anyhow::Error> {
    let (ctx, module_ptr) = lower_all_stmatrix_forms(mir_lower::IntrinsicBackend::LibNvvm)?;
    let mut lowered = Vec::new();

    for op in lowered_kernel_body(&ctx, module_ptr) {
        if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
            && let CallOpCallable::Direct(callee) = call.callee(&ctx)
        {
            assert!(
                !STMATRIX_TYPED_INTRINSICS.contains(&callee.to_string().as_str()),
                "libNVVM stmatrix lowering must not emit typed intrinsic calls"
            );
        }
        let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
            continue;
        };
        lowered.push((
            inline_asm
                .get_attr_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .unwrap_or_default(),
            inline_asm
                .get_attr_inline_asm_constraints(&ctx)
                .map(|value| String::from((*value).clone()))
                .unwrap_or_default(),
            llvm::asm_kind(&ctx, &inline_asm),
            op.deref(&ctx).get_num_operands(),
            op.deref(&ctx).get_num_results(),
        ));
    }

    assert_eq!(lowered.len(), STMATRIX_PTX.len());
    for (template, constraints) in STMATRIX_PTX {
        let matches: Vec<_> = lowered
            .iter()
            .filter(|(actual, _, _, _, _)| actual == template)
            .collect();
        assert_eq!(matches.len(), 1, "missing exact PTX {template}");
        let (_, actual_constraints, kind, operands, results) = matches[0];
        assert_eq!(actual_constraints, constraints);
        assert_eq!(*kind, llvm::AsmKind::Convergent);
        assert!(matches!(*operands, 3 | 5));
        assert_eq!(*results, 1);
    }

    let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
    let ir = llvm_export::export::export_module_to_string(&ctx, &module)
        .expect("inline stmatrix module exports to LLVM IR");
    assert_eq!(ir.matches("asm sideeffect").count(), 4, "{ir}");
    assert_eq!(ir.matches("~{memory}").count(), 4, "{ir}");
    assert!(ir.contains("attributes #0 = { convergent }"), "{ir}");
    assert!(!ir.contains("@llvm.nvvm.stmatrix"), "{ir}");
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

const LDMATRIX_TYPED_INTRINSICS: [&str; 6] = [
    "llvm_nvvm_ldmatrix_sync_aligned_m8n8_x1_b16_p3",
    "llvm_nvvm_ldmatrix_sync_aligned_m8n8_x1_trans_b16_p3",
    "llvm_nvvm_ldmatrix_sync_aligned_m8n8_x2_b16_p3",
    "llvm_nvvm_ldmatrix_sync_aligned_m8n8_x2_trans_b16_p3",
    "llvm_nvvm_ldmatrix_sync_aligned_m8n8_x4_b16_p3",
    "llvm_nvvm_ldmatrix_sync_aligned_m8n8_x4_trans_b16_p3",
];

const LDMATRIX_PTX_TEMPLATES: [&str; 6] = [
    "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {$0}, [$1];",
    "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {$0}, [$1];",
    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {$0, $1}, [$2];",
    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {$0, $1}, [$2];",
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [$4];",
];

const LDMATRIX_PTX_CONSTRAINTS: [&str; 6] = [
    "=r,r,~{memory}",
    "=r,r,~{memory}",
    "=r,=r,r,~{memory}",
    "=r,=r,r,~{memory}",
    "=r,=r,=r,=r,r,~{memory}",
    "=r,=r,=r,=r,r,~{memory}",
];

const BLACKWELL_LDMATRIX_CASES: [(&str, &str, &str, usize); 12] = [
    (
        "llvm_nvvm_ldmatrix_sync_aligned_m16n16_x1_trans_b8_p3",
        "llvm.nvvm.ldmatrix.sync.aligned.m16n16.x1.trans.b8.p3",
        "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {$0, $1}, [$2];",
        2,
    ),
    (
        "llvm__nvvm_dldmatrix_dsync_daligned_dm16n16_dx1_dtrans_db8x16_db4x16_up64_dp3",
        "llvm.nvvm.ldmatrix.sync.aligned.m16n16.x1.trans.b8x16.b4x16_p64.p3",
        "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b4x16_p64 {$0, $1}, [$2];",
        2,
    ),
    (
        "llvm__nvvm_dldmatrix_dsync_daligned_dm16n16_dx1_dtrans_db8x16_db6x16_up32_dp3",
        "llvm.nvvm.ldmatrix.sync.aligned.m16n16.x1.trans.b8x16.b6x16_p32.p3",
        "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b6x16_p32 {$0, $1}, [$2];",
        2,
    ),
    (
        "llvm_nvvm_ldmatrix_sync_aligned_m16n16_x2_trans_b8_p3",
        "llvm.nvvm.ldmatrix.sync.aligned.m16n16.x2.trans.b8.p3",
        "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {$0, $1, $2, $3}, [$4];",
        4,
    ),
    (
        "llvm__nvvm_dldmatrix_dsync_daligned_dm16n16_dx2_dtrans_db8x16_db4x16_up64_dp3",
        "llvm.nvvm.ldmatrix.sync.aligned.m16n16.x2.trans.b8x16.b4x16_p64.p3",
        "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b4x16_p64 {$0, $1, $2, $3}, [$4];",
        4,
    ),
    (
        "llvm__nvvm_dldmatrix_dsync_daligned_dm16n16_dx2_dtrans_db8x16_db6x16_up32_dp3",
        "llvm.nvvm.ldmatrix.sync.aligned.m16n16.x2.trans.b8x16.b6x16_p32.p3",
        "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b6x16_p32 {$0, $1, $2, $3}, [$4];",
        4,
    ),
    (
        "llvm__nvvm_dldmatrix_dsync_daligned_dm8n16_dx1_db8x16_db4x16_up64_dp3",
        "llvm.nvvm.ldmatrix.sync.aligned.m8n16.x1.b8x16.b4x16_p64.p3",
        "ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b4x16_p64 {$0}, [$1];",
        1,
    ),
    (
        "llvm__nvvm_dldmatrix_dsync_daligned_dm8n16_dx1_db8x16_db6x16_up32_dp3",
        "llvm.nvvm.ldmatrix.sync.aligned.m8n16.x1.b8x16.b6x16_p32.p3",
        "ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b6x16_p32 {$0}, [$1];",
        1,
    ),
    (
        "llvm__nvvm_dldmatrix_dsync_daligned_dm8n16_dx2_db8x16_db4x16_up64_dp3",
        "llvm.nvvm.ldmatrix.sync.aligned.m8n16.x2.b8x16.b4x16_p64.p3",
        "ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 {$0, $1}, [$2];",
        2,
    ),
    (
        "llvm__nvvm_dldmatrix_dsync_daligned_dm8n16_dx2_db8x16_db6x16_up32_dp3",
        "llvm.nvvm.ldmatrix.sync.aligned.m8n16.x2.b8x16.b6x16_p32.p3",
        "ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b6x16_p32 {$0, $1}, [$2];",
        2,
    ),
    (
        "llvm__nvvm_dldmatrix_dsync_daligned_dm8n16_dx4_db8x16_db4x16_up64_dp3",
        "llvm.nvvm.ldmatrix.sync.aligned.m8n16.x4.b8x16.b4x16_p64.p3",
        "ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {$0, $1, $2, $3}, [$4];",
        4,
    ),
    (
        "llvm__nvvm_dldmatrix_dsync_daligned_dm8n16_dx4_db8x16_db6x16_up32_dp3",
        "llvm.nvvm.ldmatrix.sync.aligned.m8n16.x4.b8x16.b6x16_p32.p3",
        "ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b6x16_p32 {$0, $1, $2, $3}, [$4];",
        4,
    ),
];

fn ldmatrix_constraints(register_count: usize) -> &'static str {
    match register_count {
        1 => "=r,r,~{memory}",
        2 => "=r,=r,r,~{memory}",
        4 => "=r,=r,=r,=r,r,~{memory}",
        _ => unreachable!("closed ldmatrix register count"),
    }
}

fn lower_all_ldmatrix_forms(
    address_space: u32,
    backend: mir_lower::IntrinsicBackend,
    compatibility: bool,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let ptr_ty = MirPtrType::get(&mut ctx, i8_ty.into(), true, address_space);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![ptr_ty.into()]);
    let pointer = entry.deref(&ctx).get_argument(0);
    if compatibility {
        let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
        for (op_info, result_count) in [
            (nvvm::LdmatrixX1Op::get_concrete_op_info(), 1),
            (nvvm::LdmatrixX1TransOp::get_concrete_op_info(), 1),
            (nvvm::LdmatrixX2Op::get_concrete_op_info(), 2),
            (nvvm::LdmatrixX2TransOp::get_concrete_op_info(), 2),
            (nvvm::LdmatrixX4Op::get_concrete_op_info(), 4),
            (nvvm::LdmatrixX4TransOp::get_concrete_op_info(), 4),
        ] {
            Operation::new(
                &mut ctx,
                op_info,
                vec![u32_ty.into(); result_count],
                vec![pointer],
                vec![],
                0,
            )
            .insert_at_back(entry, &ctx);
        }
    } else {
        for (multiplicity, layout) in [
            (
                nvvm::LdmatrixMultiplicityAttr::X1,
                nvvm::LdmatrixLayoutAttr::Normal,
            ),
            (
                nvvm::LdmatrixMultiplicityAttr::X1,
                nvvm::LdmatrixLayoutAttr::Transposed,
            ),
            (
                nvvm::LdmatrixMultiplicityAttr::X2,
                nvvm::LdmatrixLayoutAttr::Normal,
            ),
            (
                nvvm::LdmatrixMultiplicityAttr::X2,
                nvvm::LdmatrixLayoutAttr::Transposed,
            ),
            (
                nvvm::LdmatrixMultiplicityAttr::X4,
                nvvm::LdmatrixLayoutAttr::Normal,
            ),
            (
                nvvm::LdmatrixMultiplicityAttr::X4,
                nvvm::LdmatrixLayoutAttr::Transposed,
            ),
        ] {
            nvvm::LdmatrixOp::build(
                &mut ctx,
                pointer,
                nvvm::LdmatrixShapeAttr::M8n8,
                multiplicity,
                layout,
                nvvm::LdmatrixElementAttr::B16,
                nvvm::LdmatrixStateSpaceAttr::Shared,
            )
            .insert_at_back(entry, &ctx);
        }
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;

    Ok((ctx, module_ptr))
}

fn lower_all_blackwell_ldmatrix_forms(
    address_space: u32,
    backend: mir_lower::IntrinsicBackend,
) -> Result<(Context, pliron::context::Ptr<Operation>), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
    let ptr_ty = MirPtrType::get(&mut ctx, u8_ty.into(), true, address_space);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![ptr_ty.into()]);
    let pointer = entry.deref(&ctx).get_argument(0);

    for (shape, multiplicity, layout, element) in [
        (
            nvvm::LdmatrixShapeAttr::M16n16,
            nvvm::LdmatrixMultiplicityAttr::X1,
            nvvm::LdmatrixLayoutAttr::Transposed,
            nvvm::LdmatrixElementAttr::B8,
        ),
        (
            nvvm::LdmatrixShapeAttr::M16n16,
            nvvm::LdmatrixMultiplicityAttr::X1,
            nvvm::LdmatrixLayoutAttr::Transposed,
            nvvm::LdmatrixElementAttr::B8x16B4x16P64,
        ),
        (
            nvvm::LdmatrixShapeAttr::M16n16,
            nvvm::LdmatrixMultiplicityAttr::X1,
            nvvm::LdmatrixLayoutAttr::Transposed,
            nvvm::LdmatrixElementAttr::B8x16B6x16P32,
        ),
        (
            nvvm::LdmatrixShapeAttr::M16n16,
            nvvm::LdmatrixMultiplicityAttr::X2,
            nvvm::LdmatrixLayoutAttr::Transposed,
            nvvm::LdmatrixElementAttr::B8,
        ),
        (
            nvvm::LdmatrixShapeAttr::M16n16,
            nvvm::LdmatrixMultiplicityAttr::X2,
            nvvm::LdmatrixLayoutAttr::Transposed,
            nvvm::LdmatrixElementAttr::B8x16B4x16P64,
        ),
        (
            nvvm::LdmatrixShapeAttr::M16n16,
            nvvm::LdmatrixMultiplicityAttr::X2,
            nvvm::LdmatrixLayoutAttr::Transposed,
            nvvm::LdmatrixElementAttr::B8x16B6x16P32,
        ),
        (
            nvvm::LdmatrixShapeAttr::M8n16,
            nvvm::LdmatrixMultiplicityAttr::X1,
            nvvm::LdmatrixLayoutAttr::Normal,
            nvvm::LdmatrixElementAttr::B8x16B4x16P64,
        ),
        (
            nvvm::LdmatrixShapeAttr::M8n16,
            nvvm::LdmatrixMultiplicityAttr::X1,
            nvvm::LdmatrixLayoutAttr::Normal,
            nvvm::LdmatrixElementAttr::B8x16B6x16P32,
        ),
        (
            nvvm::LdmatrixShapeAttr::M8n16,
            nvvm::LdmatrixMultiplicityAttr::X2,
            nvvm::LdmatrixLayoutAttr::Normal,
            nvvm::LdmatrixElementAttr::B8x16B4x16P64,
        ),
        (
            nvvm::LdmatrixShapeAttr::M8n16,
            nvvm::LdmatrixMultiplicityAttr::X2,
            nvvm::LdmatrixLayoutAttr::Normal,
            nvvm::LdmatrixElementAttr::B8x16B6x16P32,
        ),
        (
            nvvm::LdmatrixShapeAttr::M8n16,
            nvvm::LdmatrixMultiplicityAttr::X4,
            nvvm::LdmatrixLayoutAttr::Normal,
            nvvm::LdmatrixElementAttr::B8x16B4x16P64,
        ),
        (
            nvvm::LdmatrixShapeAttr::M8n16,
            nvvm::LdmatrixMultiplicityAttr::X4,
            nvvm::LdmatrixLayoutAttr::Normal,
            nvvm::LdmatrixElementAttr::B8x16B6x16P32,
        ),
    ] {
        nvvm::LdmatrixOp::build(
            &mut ctx,
            pointer,
            shape,
            multiplicity,
            layout,
            element,
            nvvm::LdmatrixStateSpaceAttr::Shared,
        )
        .insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: backend,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;

    Ok((ctx, module_ptr))
}

fn lowered_kernel_body(
    ctx: &Context,
    module_ptr: pliron::context::Ptr<Operation>,
) -> Vec<pliron::context::Ptr<Operation>> {
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
        return func_region
            .deref(ctx)
            .iter(ctx)
            .flat_map(|block| block.deref(ctx).iter(ctx))
            .collect();
    }
    panic!("lowered kernel function not found")
}

fn assert_ldmatrix_producer_result_shape(
    ctx: &Context,
    op: pliron::context::Ptr<Operation>,
    register_count: usize,
) {
    use llvm_export::types as llvm_types;
    use pliron::builtin::types::IntegerType;
    use pliron::r#type::Typed;

    let ty = op.deref(ctx).get_result(0).get_type(ctx);
    let ty = ty.deref(ctx);
    if register_count == 1 {
        assert_eq!(
            ty.downcast_ref::<IntegerType>()
                .expect("single-register ldmatrix returns i32")
                .width(),
            32
        );
    } else {
        let ty = ty
            .downcast_ref::<llvm_types::StructType>()
            .expect("multi-register ldmatrix returns an LLVM struct");
        assert_eq!(ty.num_fields(), register_count);
        for index in 0..register_count {
            assert_eq!(
                ty.field_type(index)
                    .deref(ctx)
                    .downcast_ref::<IntegerType>()
                    .expect("ldmatrix fragment field is i32")
                    .width(),
                32
            );
        }
    }
}

#[test]
fn test_ldmatrix_llvm_nvptx_uses_exact_typed_p3_intrinsics() -> Result<(), anyhow::Error> {
    use llvm_export::types as llvm_types;
    use pliron::builtin::type_interfaces::FunctionTypeInterface;
    use pliron::builtin::types::IntegerType;
    use pliron::r#type::Typed;

    for address_space in [0, 3] {
        let (ctx, module_ptr) =
            lower_all_ldmatrix_forms(address_space, mir_lower::IntrinsicBackend::LlvmNvptx, false)?;
        let body = lowered_kernel_body(&ctx, module_ptr);
        let mut callees = Vec::new();
        let mut cast_count = 0;
        let mut extract_count = 0;

        for op in body {
            if let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx)
                && let CallOpCallable::Direct(callee) = call.callee(&ctx)
            {
                let callee = callee.to_string();
                let register_count = if callee.contains("_x1_") {
                    1
                } else if callee.contains("_x2_") {
                    2
                } else {
                    4
                };
                let function_ty = call.callee_type(&ctx);
                let function_ty = function_ty.deref(&ctx);
                let function_ty = function_ty
                    .downcast_ref::<llvm_types::FuncType>()
                    .expect("ldmatrix callee has an LLVM function type");
                assert_eq!(function_ty.arg_types().len(), 1);
                let argument_ty = function_ty.arg_types()[0].deref(&ctx);
                let argument_ty = argument_ty
                    .downcast_ref::<llvm_types::PointerType>()
                    .expect("ldmatrix argument is a pointer");
                assert_eq!(argument_ty.address_space(), 3);

                let result_ty = function_ty.result_type();
                let result_ty = result_ty.deref(&ctx);
                if register_count == 1 {
                    let result_ty = result_ty
                        .downcast_ref::<IntegerType>()
                        .expect("x1 returns i32");
                    assert_eq!(result_ty.width(), 32);
                } else {
                    let result_ty = result_ty
                        .downcast_ref::<llvm_types::StructType>()
                        .expect("x2/x4 return an LLVM struct");
                    assert_eq!(result_ty.num_fields(), register_count);
                    for index in 0..result_ty.num_fields() {
                        let field = result_ty.field_type(index);
                        let field = field.deref(&ctx);
                        let field = field
                            .downcast_ref::<IntegerType>()
                            .expect("fragment register is i32");
                        assert_eq!(field.width(), 32);
                    }
                }

                callees.push(callee);
                assert_eq!(op.deref(&ctx).get_num_operands(), 1);
                assert_eq!(op.deref(&ctx).get_num_results(), 1);
            }
            if Operation::get_op::<llvm::AddrSpaceCastOp>(op, &ctx).is_some() {
                cast_count += 1;
                let cast = op.deref(&ctx);
                let source_ty = cast.get_operand(0).get_type(&ctx);
                let source_ty = source_ty.deref(&ctx);
                let source_ty = source_ty
                    .downcast_ref::<llvm_types::PointerType>()
                    .expect("addrspacecast source is a pointer");
                let result_ty = cast.get_result(0).get_type(&ctx);
                let result_ty = result_ty.deref(&ctx);
                let result_ty = result_ty
                    .downcast_ref::<llvm_types::PointerType>()
                    .expect("addrspacecast result is a pointer");
                assert_eq!(
                    (source_ty.address_space(), result_ty.address_space()),
                    (0, 3)
                );
            }
            extract_count +=
                usize::from(Operation::get_op::<llvm::ExtractValueOp>(op, &ctx).is_some());
            assert!(
                Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none(),
                "LLVM-NVPTX ldmatrix lowering must not emit inline PTX"
            );
            assert!(
                Operation::get_op::<llvm::PtrToIntOp>(op, &ctx).is_none(),
                "the typed intrinsic consumes the shared pointer directly"
            );
        }

        callees.sort();
        let mut expected = LDMATRIX_TYPED_INTRINSICS.map(str::to_owned);
        expected.sort();
        assert_eq!(callees, expected);
        assert_eq!(cast_count, if address_space == 0 { 6 } else { 0 });
        assert_eq!(
            extract_count, 12,
            "x2/x4 structs must preserve result order"
        );

        let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
        let ir = llvm_export::export::export_module_to_string(&ctx, &module)
            .expect("typed ldmatrix module exports to LLVM IR");
        for intrinsic in LDMATRIX_TYPED_INTRINSICS {
            let dotted = intrinsic.replace('_', ".");
            assert!(
                ir.contains(&format!("@{dotted}(ptr addrspace(3)")),
                "underscore symbol must export as exact dotted p3 intrinsic:\n{ir}"
            );
        }
        assert!(!ir.contains("@llvm_nvvm_ldmatrix"));
    }
    Ok(())
}

#[test]
fn test_ldmatrix_libnvvm_uses_exact_convergent_shared_ptx() -> Result<(), anyhow::Error> {
    use llvm_export::types as llvm_types;
    use pliron::r#type::Typed;

    for address_space in [0, 3] {
        let (ctx, module_ptr) =
            lower_all_ldmatrix_forms(address_space, mir_lower::IntrinsicBackend::LibNvvm, false)?;
        let body = lowered_kernel_body(&ctx, module_ptr);
        let mut lowered = Vec::new();
        let mut cast_count = 0;
        let mut ptrtoint_count = 0;

        for op in body {
            assert!(
                Operation::get_op::<llvm::CallOp>(op, &ctx).is_none(),
                "libNVVM ldmatrix lowering must not emit typed intrinsic calls"
            );
            cast_count +=
                usize::from(Operation::get_op::<llvm::AddrSpaceCastOp>(op, &ctx).is_some());
            if Operation::get_op::<llvm::PtrToIntOp>(op, &ctx).is_some() {
                ptrtoint_count += 1;
                let cast = op.deref(&ctx);
                let source_ty = cast.get_operand(0).get_type(&ctx);
                let source_ty = source_ty.deref(&ctx);
                let source_ty = source_ty
                    .downcast_ref::<llvm_types::PointerType>()
                    .expect("ptrtoint source is a pointer");
                assert_eq!(source_ty.address_space(), 3);
            }

            let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
                continue;
            };
            lowered.push((
                inline_asm
                    .get_attr_inline_asm_template(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .unwrap_or_default(),
                inline_asm
                    .get_attr_inline_asm_constraints(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .unwrap_or_default(),
                llvm::asm_kind(&ctx, &inline_asm),
                op.deref(&ctx).get_num_operands(),
                op.deref(&ctx).get_num_results(),
            ));
        }

        assert_eq!(lowered.len(), 6);
        for (index, expected_template) in LDMATRIX_PTX_TEMPLATES.iter().enumerate() {
            let matching: Vec<_> = lowered
                .iter()
                .filter(|(template, _, _, _, _)| template == expected_template)
                .collect();
            assert_eq!(matching.len(), 1, "missing exact PTX `{expected_template}`");
            let (_, constraints, kind, operands, results) = matching[0];
            assert_eq!(constraints, LDMATRIX_PTX_CONSTRAINTS[index]);
            assert_eq!(*kind, llvm::AsmKind::Convergent);
            assert_eq!(*operands, 1, "inline PTX consumes one i32 shared address");
            assert_eq!(*results, 1, "inline PTX returns one scalar or struct");
            assert!(!expected_template.contains("cvta.to.shared"));
        }
        assert_eq!(cast_count, if address_space == 0 { 6 } else { 0 });
        assert_eq!(ptrtoint_count, 6);

        let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
        let ir = llvm_export::export::export_module_to_string(&ctx, &module)
            .expect("inline ldmatrix module exports to LLVM IR");
        assert_eq!(ir.matches("asm sideeffect").count(), 6, "{ir}");
        assert_eq!(ir.matches("~{memory}").count(), 6, "{ir}");
        assert!(ir.contains("attributes #0 = { convergent }"), "{ir}");
        assert!(!ir.contains("@llvm.nvvm.ldmatrix"), "{ir}");
    }
    Ok(())
}

#[test]
fn test_blackwell_ldmatrix_llvm_uses_all_exact_lossless_p3_intrinsics() -> Result<(), anyhow::Error>
{
    use llvm_export::types as llvm_types;
    use pliron::builtin::type_interfaces::FunctionTypeInterface;

    for address_space in [0, 3] {
        let (ctx, module_ptr) = lower_all_blackwell_ldmatrix_forms(
            address_space,
            mir_lower::IntrinsicBackend::LlvmNvptx,
        )?;
        let mut seen = [0; 12];
        let mut cast_count = 0;
        let mut extract_count = 0;

        for op in lowered_kernel_body(&ctx, module_ptr) {
            assert!(Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none());
            assert!(Operation::get_op::<llvm::PtrToIntOp>(op, &ctx).is_none());
            cast_count +=
                usize::from(Operation::get_op::<llvm::AddrSpaceCastOp>(op, &ctx).is_some());
            extract_count +=
                usize::from(Operation::get_op::<llvm::ExtractValueOp>(op, &ctx).is_some());

            let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
                continue;
            };
            let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
                panic!("Blackwell ldmatrix intrinsic call must be direct");
            };
            let callee = callee.to_string();
            let index = BLACKWELL_LDMATRIX_CASES
                .iter()
                .position(|(identifier, _, _, _)| *identifier == callee)
                .expect("exact lossless Blackwell ldmatrix identifier");
            seen[index] += 1;
            let register_count = BLACKWELL_LDMATRIX_CASES[index].3;
            assert_ldmatrix_producer_result_shape(&ctx, op, register_count);

            let function_ty = call.callee_type(&ctx);
            let function_ty = function_ty.deref(&ctx);
            let function_ty = function_ty
                .downcast_ref::<llvm_types::FuncType>()
                .expect("Blackwell ldmatrix has an LLVM function type");
            assert_eq!(function_ty.arg_types().len(), 1);
            assert_eq!(
                function_ty.arg_types()[0]
                    .deref(&ctx)
                    .downcast_ref::<llvm_types::PointerType>()
                    .expect("Blackwell ldmatrix argument is a pointer")
                    .address_space(),
                3
            );
        }

        assert_eq!(seen, [1; 12]);
        assert_eq!(cast_count, if address_space == 0 { 12 } else { 0 });
        assert_eq!(extract_count, 30, "all multi-register results are unpacked");

        let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
        let ir = llvm_export::export::export_module_to_string(&ctx, &module)
            .expect("typed Blackwell ldmatrix module exports to LLVM IR");
        for (identifier, symbol, _, _) in BLACKWELL_LDMATRIX_CASES {
            assert!(
                ir.contains(&format!("@{symbol}(ptr addrspace(3)")),
                "missing exact intrinsic symbol {symbol}:\n{ir}"
            );
            assert!(
                !ir.contains(&format!("@{identifier}(")),
                "encoded Rust identifier leaked into LLVM IR: {identifier}"
            );
        }
        assert!(ir.contains("b4x16_p64.p3"), "literal _p64 was lost: {ir}");
        assert!(ir.contains("b6x16_p32.p3"), "literal _p32 was lost: {ir}");
    }
    Ok(())
}

#[test]
fn test_blackwell_ldmatrix_libnvvm_uses_all_exact_convergent_templates_without_externs()
-> Result<(), anyhow::Error> {
    for address_space in [0, 3] {
        let (ctx, module_ptr) = lower_all_blackwell_ldmatrix_forms(
            address_space,
            mir_lower::IntrinsicBackend::LibNvvm,
        )?;
        let mut seen = [0; 12];
        let mut cast_count = 0;
        let mut ptrtoint_count = 0;
        let mut extract_count = 0;

        for op in lowered_kernel_body(&ctx, module_ptr) {
            assert!(
                Operation::get_op::<llvm::CallOp>(op, &ctx).is_none(),
                "libNVVM Blackwell ldmatrix must not emit an extern call"
            );
            cast_count +=
                usize::from(Operation::get_op::<llvm::AddrSpaceCastOp>(op, &ctx).is_some());
            ptrtoint_count +=
                usize::from(Operation::get_op::<llvm::PtrToIntOp>(op, &ctx).is_some());
            extract_count +=
                usize::from(Operation::get_op::<llvm::ExtractValueOp>(op, &ctx).is_some());

            let Some(asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
                continue;
            };
            let template = asm
                .get_attr_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))
                .unwrap_or_default();
            let index = BLACKWELL_LDMATRIX_CASES
                .iter()
                .position(|(_, _, expected, _)| *expected == template)
                .expect("exact Blackwell ldmatrix inline-PTX template");
            seen[index] += 1;
            let register_count = BLACKWELL_LDMATRIX_CASES[index].3;
            assert_eq!(
                asm.get_attr_inline_asm_constraints(&ctx)
                    .map(|value| String::from((*value).clone()))
                    .as_deref(),
                Some(ldmatrix_constraints(register_count))
            );
            assert_eq!(llvm::asm_kind(&ctx, &asm), llvm::AsmKind::Convergent);
            assert_ldmatrix_producer_result_shape(&ctx, op, register_count);
        }

        assert_eq!(seen, [1; 12]);
        assert_eq!(cast_count, if address_space == 0 { 12 } else { 0 });
        assert_eq!(ptrtoint_count, 12);
        assert_eq!(extract_count, 30, "all multi-register results are unpacked");

        let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
        let ir = llvm_export::export::export_module_to_string(&ctx, &module)
            .expect("inline Blackwell ldmatrix module exports to LLVM IR");
        assert_eq!(ir.matches("asm sideeffect").count(), 12, "{ir}");
        assert_eq!(ir.matches("~{memory}").count(), 12, "{ir}");
        assert!(!ir.contains("@llvm.nvvm.ldmatrix"), "{ir}");
        assert!(!ir.contains("llvm__nvvm_dldmatrix"), "{ir}");
    }
    Ok(())
}

#[test]
fn test_blackwell_ldmatrix_rejects_unadmitted_m16n16_x4() {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        let mut ctx = make_test_ctx();
        let u8_ty = IntegerType::get(&ctx, 8, Signedness::Unsigned);
        let ptr_ty = MirPtrType::get_shared(&mut ctx, u8_ty.into(), false);
        let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![ptr_ty.into()]);
        let pointer = entry.deref(&ctx).get_argument(0);
        nvvm::LdmatrixOp::build(
            &mut ctx,
            pointer,
            nvvm::LdmatrixShapeAttr::M16n16,
            nvvm::LdmatrixMultiplicityAttr::X4,
            nvvm::LdmatrixLayoutAttr::Transposed,
            nvvm::LdmatrixElementAttr::B8,
            nvvm::LdmatrixStateSpaceAttr::Shared,
        )
        .insert_at_back(entry, &ctx);
        append_return(&mut ctx, entry);

        let error = mir_lower::lower_mir_to_llvm_with_options(
            &mut ctx,
            module_ptr,
            mir_lower::LoweringOptions {
                intrinsic_backend: backend,
                ..Default::default()
            },
        )
        .expect_err("m16n16.x4 must fail closed")
        .to_string();
        assert!(
            error.contains("variant has no generated lowering recipe"),
            "{error}"
        );
    }
}

#[test]
fn test_classic_ldmatrix_compatibility_ops_keep_exact_lowering() -> Result<(), anyhow::Error> {
    use llvm_export::types as llvm_types;
    use pliron::builtin::types::IntegerType;
    use pliron::r#type::Typed;

    fn assert_result_shape(
        ctx: &Context,
        op: pliron::context::Ptr<Operation>,
        register_count: usize,
    ) {
        let operation = op.deref(ctx);
        assert_eq!(operation.get_num_operands(), 1);
        assert_eq!(operation.get_num_results(), 1);
        let result_ty = operation.get_result(0).get_type(ctx);
        let result_ty = result_ty.deref(ctx);
        if register_count == 1 {
            let result_ty = result_ty
                .downcast_ref::<IntegerType>()
                .expect("x1 returns i32");
            assert_eq!(result_ty.width(), 32);
        } else {
            let result_ty = result_ty
                .downcast_ref::<llvm_types::StructType>()
                .expect("x2/x4 return an LLVM struct");
            assert_eq!(result_ty.num_fields(), register_count);
            for index in 0..result_ty.num_fields() {
                let field = result_ty.field_type(index);
                let field = field.deref(ctx);
                let field = field
                    .downcast_ref::<IntegerType>()
                    .expect("fragment register is i32");
                assert_eq!(field.width(), 32);
            }
        }
    }

    let register_counts = [1, 1, 2, 2, 4, 4];
    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        let (ctx, module_ptr) = lower_all_ldmatrix_forms(3, backend, true)?;
        let mut seen = [0; 6];
        let mut extract_count = 0;

        for op in lowered_kernel_body(&ctx, module_ptr) {
            extract_count +=
                usize::from(Operation::get_op::<llvm::ExtractValueOp>(op, &ctx).is_some());
            match backend {
                mir_lower::IntrinsicBackend::LlvmNvptx => {
                    assert!(Operation::get_op::<llvm::InlineAsmOp>(op, &ctx).is_none());
                    let Some(call) = Operation::get_op::<llvm::CallOp>(op, &ctx) else {
                        continue;
                    };
                    let CallOpCallable::Direct(callee) = call.callee(&ctx) else {
                        panic!("ldmatrix intrinsic call must be direct");
                    };
                    let callee = callee.to_string();
                    let index = LDMATRIX_TYPED_INTRINSICS
                        .iter()
                        .position(|expected| *expected == callee)
                        .expect("exact typed ldmatrix intrinsic");
                    seen[index] += 1;
                    assert_result_shape(&ctx, op, register_counts[index]);
                }
                mir_lower::IntrinsicBackend::LibNvvm => {
                    assert!(Operation::get_op::<llvm::CallOp>(op, &ctx).is_none());
                    let Some(inline_asm) = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx) else {
                        continue;
                    };
                    let template = inline_asm
                        .get_attr_inline_asm_template(&ctx)
                        .map(|value| String::from((*value).clone()))
                        .unwrap_or_default();
                    let index = LDMATRIX_PTX_TEMPLATES
                        .iter()
                        .position(|expected| *expected == template)
                        .expect("exact ldmatrix PTX template");
                    seen[index] += 1;
                    assert_eq!(
                        inline_asm
                            .get_attr_inline_asm_constraints(&ctx)
                            .map(|value| String::from((*value).clone()))
                            .as_deref(),
                        Some(LDMATRIX_PTX_CONSTRAINTS[index])
                    );
                    assert_eq!(llvm::asm_kind(&ctx, &inline_asm), llvm::AsmKind::Convergent);
                    assert_result_shape(&ctx, op, register_counts[index]);
                }
            }
        }

        assert_eq!(seen, [1; 6]);
        assert_eq!(extract_count, 12, "x2/x4 results keep their order");
    }
    Ok(())
}

#[test]
fn test_ldmatrix_rejects_non_shared_pointer_spaces() {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for address_space in [1, 4, 5] {
            let mut ctx = make_test_ctx();
            let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
            let ptr_ty = MirPtrType::get(&mut ctx, i8_ty.into(), false, address_space);
            let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![ptr_ty.into()]);
            let pointer = entry.deref(&ctx).get_argument(0);
            nvvm::LdmatrixOp::build(
                &mut ctx,
                pointer,
                nvvm::LdmatrixShapeAttr::M8n8,
                nvvm::LdmatrixMultiplicityAttr::X1,
                nvvm::LdmatrixLayoutAttr::Normal,
                nvvm::LdmatrixElementAttr::B16,
                nvvm::LdmatrixStateSpaceAttr::Shared,
            )
            .insert_at_back(entry, &ctx);
            append_return(&mut ctx, entry);

            let error = mir_lower::lower_mir_to_llvm_with_options(
                &mut ctx,
                module_ptr,
                mir_lower::LoweringOptions {
                    intrinsic_backend: backend,
                    ..Default::default()
                },
            )
            .expect_err("global/constant pointers must fail closed")
            .to_string();
            assert!(
                error.contains(&format!("got address space {address_space}")),
                "{error}"
            );
        }
    }
}

#[test]
fn test_ldmatrix_rejects_non_pointer_operand() {
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![i32_ty.into()]);
    let not_a_pointer = entry.deref(&ctx).get_argument(0);
    nvvm::LdmatrixOp::build(
        &mut ctx,
        not_a_pointer,
        nvvm::LdmatrixShapeAttr::M8n8,
        nvvm::LdmatrixMultiplicityAttr::X1,
        nvvm::LdmatrixLayoutAttr::Normal,
        nvvm::LdmatrixElementAttr::B16,
        nvvm::LdmatrixStateSpaceAttr::Shared,
    )
    .insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    let error = mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .expect_err("non-pointer ldmatrix input must fail closed")
        .to_string();
    assert!(
        error.contains("requires an LLVM pointer operand"),
        "{error}"
    );
}

#[test]
fn test_ldmatrix_rejects_wrong_result_arity() {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let i8_ty = IntegerType::get(&ctx, 8, Signedness::Signless);
    let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
    let ptr_ty = MirPtrType::get_shared(&mut ctx, i8_ty.into(), false);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![ptr_ty.into()]);
    let pointer = entry.deref(&ctx).get_argument(0);
    let op = Operation::new(
        &mut ctx,
        nvvm::LdmatrixOp::get_concrete_op_info(),
        vec![i32_ty.into(), i32_ty.into()],
        vec![pointer],
        vec![],
        0,
    );
    let ldmatrix = nvvm::LdmatrixOp::new(op);
    ldmatrix.set_attr_nvvm_ldmatrix_shape(&mut ctx, nvvm::LdmatrixShapeAttr::M8n8);
    ldmatrix.set_attr_nvvm_ldmatrix_multiplicity(&mut ctx, nvvm::LdmatrixMultiplicityAttr::X1);
    ldmatrix.set_attr_nvvm_ldmatrix_layout(&mut ctx, nvvm::LdmatrixLayoutAttr::Normal);
    ldmatrix.set_attr_nvvm_ldmatrix_element(&mut ctx, nvvm::LdmatrixElementAttr::B16);
    ldmatrix.set_attr_nvvm_ldmatrix_state_space(&mut ctx, nvvm::LdmatrixStateSpaceAttr::Shared);
    op.insert_at_back(entry, &ctx);
    append_return(&mut ctx, entry);

    let error = mir_lower::lower_mir_to_llvm(&mut ctx, module_ptr)
        .expect_err("x1 must return exactly one register")
        .to_string();
    assert!(
        error.contains("requires 1 i32 result register(s), got 2"),
        "{error}"
    );
}

#[test]
fn test_generated_register_mma_variants_lower_to_exact_convergent_inline_ptx()
-> Result<(), anyhow::Error> {
    use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
    use pliron::r#type::TypeHandle;

    #[derive(Clone, Copy)]
    enum Carrier {
        F32,
        F64,
        I32,
        U32,
    }

    struct Case {
        shape: nvvm::RegisterMmaShapeAttr,
        operation: nvvm::RegisterMmaOperationAttr,
        accumulator: nvvm::RegisterMmaAccumulatorAttr,
        a_element: nvvm::RegisterMmaElementAttr,
        b_element: nvvm::RegisterMmaElementAttr,
        overflow: nvvm::RegisterMmaOverflowAttr,
        operands: &'static [Carrier],
        results: &'static [Carrier],
        template: String,
        constraints: &'static str,
    }

    let mut cases = vec![
        Case {
            shape: nvvm::RegisterMmaShapeAttr::M16n8k16,
            operation: nvvm::RegisterMmaOperationAttr::Multiply,
            accumulator: nvvm::RegisterMmaAccumulatorAttr::F32,
            a_element: nvvm::RegisterMmaElementAttr::Bf16,
            b_element: nvvm::RegisterMmaElementAttr::Bf16,
            overflow: nvvm::RegisterMmaOverflowAttr::NotApplicable,
            operands: &[
                Carrier::F32,
                Carrier::F32,
                Carrier::F32,
                Carrier::F32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
            ],
            results: &[Carrier::F32; 4],
            template: concat!(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ",
                "{$0, $1, $2, $3}, {$8, $9, $10, $11}, ",
                "{$12, $13}, {$4, $5, $6, $7};"
            )
            .into(),
            constraints: "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r",
        },
        Case {
            shape: nvvm::RegisterMmaShapeAttr::M16n8k16,
            operation: nvvm::RegisterMmaOperationAttr::Multiply,
            accumulator: nvvm::RegisterMmaAccumulatorAttr::F32,
            a_element: nvvm::RegisterMmaElementAttr::F16,
            b_element: nvvm::RegisterMmaElementAttr::F16,
            overflow: nvvm::RegisterMmaOverflowAttr::NotApplicable,
            operands: &[
                Carrier::F32,
                Carrier::F32,
                Carrier::F32,
                Carrier::F32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
            ],
            results: &[Carrier::F32; 4],
            template: concat!(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ",
                "{$0, $1, $2, $3}, {$8, $9, $10, $11}, ",
                "{$12, $13}, {$4, $5, $6, $7};"
            )
            .into(),
            constraints: "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r",
        },
        Case {
            shape: nvvm::RegisterMmaShapeAttr::M16n8k8,
            operation: nvvm::RegisterMmaOperationAttr::Multiply,
            accumulator: nvvm::RegisterMmaAccumulatorAttr::F32,
            a_element: nvvm::RegisterMmaElementAttr::Tf32,
            b_element: nvvm::RegisterMmaElementAttr::Tf32,
            overflow: nvvm::RegisterMmaOverflowAttr::NotApplicable,
            operands: &[
                Carrier::F32,
                Carrier::F32,
                Carrier::F32,
                Carrier::F32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
                Carrier::U32,
            ],
            results: &[Carrier::F32; 4],
            template: concat!(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 ",
                "{$0, $1, $2, $3}, {$8, $9, $10, $11}, ",
                "{$12, $13}, {$4, $5, $6, $7};"
            )
            .into(),
            constraints: "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r",
        },
        Case {
            shape: nvvm::RegisterMmaShapeAttr::M8n8k4,
            operation: nvvm::RegisterMmaOperationAttr::Multiply,
            accumulator: nvvm::RegisterMmaAccumulatorAttr::F64,
            a_element: nvvm::RegisterMmaElementAttr::F64,
            b_element: nvvm::RegisterMmaElementAttr::F64,
            overflow: nvvm::RegisterMmaOverflowAttr::NotApplicable,
            operands: &[Carrier::F64; 4],
            results: &[Carrier::F64; 2],
            template: concat!(
                "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 ",
                "{$0, $1}, {$4}, {$5}, {$2, $3};"
            )
            .into(),
            constraints: "=d,=d,d,d,d,d",
        },
    ];
    let c2_a1_b1: &'static [Carrier] = &[Carrier::I32, Carrier::I32, Carrier::U32, Carrier::U32];
    let c4_a2_b1: &'static [Carrier] = &[
        Carrier::I32,
        Carrier::I32,
        Carrier::I32,
        Carrier::I32,
        Carrier::U32,
        Carrier::U32,
        Carrier::U32,
    ];
    let c4_a4_b2: &'static [Carrier] = &[
        Carrier::I32,
        Carrier::I32,
        Carrier::I32,
        Carrier::I32,
        Carrier::U32,
        Carrier::U32,
        Carrier::U32,
        Carrier::U32,
        Carrier::U32,
        Carrier::U32,
    ];
    let d2_i32: &'static [Carrier] = &[Carrier::I32; 2];
    let d4_i32: &'static [Carrier] = &[Carrier::I32; 4];
    let register_list = |first, count| {
        format!(
            "{{{}}}",
            (first..first + count)
                .map(|index| format!("${index}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };

    for (shape, shape_attr, operands, results, accumulator_count, a_count, b_count, constraints) in [
        (
            "m8n8k16",
            nvvm::RegisterMmaShapeAttr::M8n8k16,
            c2_a1_b1,
            d2_i32,
            2,
            1,
            1,
            "=r,=r,r,r,r,r",
        ),
        (
            "m16n8k16",
            nvvm::RegisterMmaShapeAttr::M16n8k16,
            c4_a2_b1,
            d4_i32,
            4,
            2,
            1,
            "=r,=r,=r,=r,r,r,r,r,r,r,r",
        ),
        (
            "m16n8k32",
            nvvm::RegisterMmaShapeAttr::M16n8k32,
            c4_a4_b2,
            d4_i32,
            4,
            4,
            2,
            "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r",
        ),
    ] {
        for (a_name, a_element) in [
            ("s8", nvvm::RegisterMmaElementAttr::S8),
            ("u8", nvvm::RegisterMmaElementAttr::U8),
        ] {
            for (b_name, b_element) in [
                ("s8", nvvm::RegisterMmaElementAttr::S8),
                ("u8", nvvm::RegisterMmaElementAttr::U8),
            ] {
                for (overflow_name, overflow) in [
                    ("", nvvm::RegisterMmaOverflowAttr::Wrapping),
                    (".satfinite", nvvm::RegisterMmaOverflowAttr::Satfinite),
                ] {
                    let result_count = results.len();
                    let d = register_list(0, result_count);
                    let c = register_list(result_count, accumulator_count);
                    let a = register_list(result_count + accumulator_count, a_count);
                    let b = register_list(result_count + accumulator_count + a_count, b_count);
                    cases.push(Case {
                        shape: shape_attr.clone(),
                        operation: nvvm::RegisterMmaOperationAttr::Multiply,
                        accumulator: nvvm::RegisterMmaAccumulatorAttr::S32,
                        a_element: a_element.clone(),
                        b_element: b_element.clone(),
                        overflow,
                        operands,
                        results,
                        template: format!(
                            "mma.sync.aligned.{shape}.row.col{overflow_name}.s32.{a_name}.{b_name}.s32 {d}, {a}, {b}, {c};"
                        ),
                        constraints,
                    });
                }
            }
        }
    }

    for (shape, shape_attr, operands, results, accumulator_count, a_count, b_count, constraints) in [
        (
            "m8n8k32",
            nvvm::RegisterMmaShapeAttr::M8n8k32,
            c2_a1_b1,
            d2_i32,
            2,
            1,
            1,
            "=r,=r,r,r,r,r",
        ),
        (
            "m16n8k32",
            nvvm::RegisterMmaShapeAttr::M16n8k32,
            c4_a2_b1,
            d4_i32,
            4,
            2,
            1,
            "=r,=r,=r,=r,r,r,r,r,r,r,r",
        ),
        (
            "m16n8k64",
            nvvm::RegisterMmaShapeAttr::M16n8k64,
            c4_a4_b2,
            d4_i32,
            4,
            4,
            2,
            "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r",
        ),
    ] {
        for (a_name, a_element) in [
            ("s4", nvvm::RegisterMmaElementAttr::S4),
            ("u4", nvvm::RegisterMmaElementAttr::U4),
        ] {
            for (b_name, b_element) in [
                ("s4", nvvm::RegisterMmaElementAttr::S4),
                ("u4", nvvm::RegisterMmaElementAttr::U4),
            ] {
                for (overflow_name, overflow) in [
                    ("", nvvm::RegisterMmaOverflowAttr::Wrapping),
                    (".satfinite", nvvm::RegisterMmaOverflowAttr::Satfinite),
                ] {
                    let result_count = results.len();
                    let d = register_list(0, result_count);
                    let c = register_list(result_count, accumulator_count);
                    let a = register_list(result_count + accumulator_count, a_count);
                    let b = register_list(result_count + accumulator_count + a_count, b_count);
                    cases.push(Case {
                        shape: shape_attr.clone(),
                        operation: nvvm::RegisterMmaOperationAttr::Multiply,
                        accumulator: nvvm::RegisterMmaAccumulatorAttr::S32,
                        a_element: a_element.clone(),
                        b_element: b_element.clone(),
                        overflow,
                        operands,
                        results,
                        template: format!(
                            "mma.sync.aligned.{shape}.row.col{overflow_name}.s32.{a_name}.{b_name}.s32 {d}, {a}, {b}, {c};"
                        ),
                        constraints,
                    });
                }
            }
        }
    }

    for (shape, shape_attr, operands, results, accumulator_count, a_count, b_count, constraints) in [
        (
            "m8n8k128",
            nvvm::RegisterMmaShapeAttr::M8n8k128,
            c2_a1_b1,
            d2_i32,
            2,
            1,
            1,
            "=r,=r,r,r,r,r",
        ),
        (
            "m16n8k128",
            nvvm::RegisterMmaShapeAttr::M16n8k128,
            c4_a2_b1,
            d4_i32,
            4,
            2,
            1,
            "=r,=r,=r,=r,r,r,r,r,r,r,r",
        ),
        (
            "m16n8k256",
            nvvm::RegisterMmaShapeAttr::M16n8k256,
            c4_a4_b2,
            d4_i32,
            4,
            4,
            2,
            "=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r",
        ),
    ] {
        let result_count = results.len();
        let d = register_list(0, result_count);
        let c = register_list(result_count, accumulator_count);
        let a = register_list(result_count + accumulator_count, a_count);
        let b = register_list(result_count + accumulator_count + a_count, b_count);
        for (operation_name, operation) in [
            ("xor", nvvm::RegisterMmaOperationAttr::XorPopc),
            ("and", nvvm::RegisterMmaOperationAttr::AndPopc),
        ] {
            cases.push(Case {
                shape: shape_attr.clone(),
                operation,
                accumulator: nvvm::RegisterMmaAccumulatorAttr::S32,
                a_element: nvvm::RegisterMmaElementAttr::B1,
                b_element: nvvm::RegisterMmaElementAttr::B1,
                overflow: nvvm::RegisterMmaOverflowAttr::Wrapping,
                operands,
                results,
                template: format!(
                    "mma.sync.aligned.{shape}.row.col.s32.b1.b1.s32.{operation_name}.popc {d}, {a}, {b}, {c};"
                ),
                constraints,
            });
        }
    }
    assert_eq!(cases.len(), 58);

    let carrier_type = |ctx: &Context, carrier: Carrier| -> TypeHandle {
        match carrier {
            Carrier::F32 => FP32Type::get(ctx).into(),
            Carrier::F64 => FP64Type::get(ctx).into(),
            Carrier::I32 => IntegerType::get(ctx, 32, Signedness::Signed).into(),
            Carrier::U32 => IntegerType::get(ctx, 32, Signedness::Unsigned).into(),
        }
    };

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for case in &cases {
            let mut ctx = make_test_ctx();
            let argument_types = case
                .operands
                .iter()
                .map(|carrier| carrier_type(&ctx, *carrier))
                .collect();
            let result_types = case
                .results
                .iter()
                .map(|carrier| carrier_type(&ctx, *carrier))
                .collect();
            let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);
            let operands = (0..case.operands.len())
                .map(|index| entry.deref(&ctx).get_argument(index))
                .collect();
            let operation = Operation::new(
                &mut ctx,
                nvvm::RegisterMmaOp::get_concrete_op_info(),
                result_types,
                operands,
                vec![],
                0,
            );
            let mma = nvvm::RegisterMmaOp::new(operation);
            mma.set_attr_nvvm_register_mma_shape(&ctx, case.shape.clone());
            let uses_legacy_default = matches!(
                (&case.operation, &case.a_element),
                (
                    nvvm::RegisterMmaOperationAttr::Multiply,
                    nvvm::RegisterMmaElementAttr::Bf16
                )
            );
            if !uses_legacy_default {
                mma.set_attr_nvvm_register_mma_operation(&ctx, case.operation.clone());
            }
            mma.set_attr_nvvm_register_mma_accumulator(&ctx, case.accumulator.clone());
            mma.set_attr_nvvm_register_mma_a_element(&ctx, case.a_element.clone());
            mma.set_attr_nvvm_register_mma_b_element(&ctx, case.b_element.clone());
            mma.set_attr_nvvm_register_mma_a_layout(&ctx, nvvm::RegisterMmaLayoutAttr::Row);
            mma.set_attr_nvvm_register_mma_b_layout(&ctx, nvvm::RegisterMmaLayoutAttr::Col);
            mma.set_attr_nvvm_register_mma_overflow(&ctx, case.overflow.clone());
            operation.insert_at_back(entry, &ctx);
            append_return(&mut ctx, entry);

            mir_lower::lower_mir_to_llvm_with_options(
                &mut ctx,
                module_ptr,
                mir_lower::LoweringOptions {
                    intrinsic_backend: backend,
                    ..Default::default()
                },
            )
            .map_err(|error| anyhow::anyhow!("{error}"))?;

            let body = lowered_kernel_body(&ctx, module_ptr);
            let lowered = body
                .iter()
                .filter_map(|op| Operation::get_op::<llvm::InlineAsmOp>(*op, &ctx))
                .collect::<Vec<_>>();
            assert_eq!(lowered.len(), 1, "{:?}", backend);
            let asm = &lowered[0];
            assert_eq!(
                asm.get_attr_inline_asm_template(&ctx)
                    .as_deref()
                    .map(|value| String::from(value.clone())),
                Some(case.template.clone())
            );
            assert_eq!(
                asm.get_attr_inline_asm_constraints(&ctx)
                    .as_deref()
                    .map(|value| String::from(value.clone())),
                Some(case.constraints.to_string())
            );
            assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);
            assert_eq!(
                asm.get_operation().deref(&ctx).get_num_operands(),
                case.operands.len()
            );
            assert_eq!(asm.get_operation().deref(&ctx).get_num_results(), 1);
            assert_eq!(
                body.iter()
                    .filter(|op| Operation::get_op::<llvm::ExtractValueOp>(**op, &ctx).is_some())
                    .count(),
                case.results.len()
            );

            let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
            let ir = llvm_export::export::export_module_to_string(&ctx, &module)
                .expect("generated register MMA exports to LLVM IR");
            assert!(ir.contains("asm sideeffect"), "{ir}");
            assert!(ir.contains("{ convergent }"), "{ir}");
        }
    }

    Ok(())
}

#[test]
fn test_generated_sparse_mma_variants_lower_to_exact_convergent_inline_ptx()
-> Result<(), anyhow::Error> {
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let cases = [
        (
            "s8",
            "s8",
            "",
            nvvm::SparseMmaElementAttr::S8,
            nvvm::SparseMmaElementAttr::S8,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "s8",
            "u8",
            "",
            nvvm::SparseMmaElementAttr::S8,
            nvvm::SparseMmaElementAttr::U8,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "u8",
            "u8",
            "",
            nvvm::SparseMmaElementAttr::U8,
            nvvm::SparseMmaElementAttr::U8,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "u8",
            "s8",
            "",
            nvvm::SparseMmaElementAttr::U8,
            nvvm::SparseMmaElementAttr::S8,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "s8",
            "s8",
            ".satfinite",
            nvvm::SparseMmaElementAttr::S8,
            nvvm::SparseMmaElementAttr::S8,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
        (
            "s8",
            "u8",
            ".satfinite",
            nvvm::SparseMmaElementAttr::S8,
            nvvm::SparseMmaElementAttr::U8,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
        (
            "u8",
            "u8",
            ".satfinite",
            nvvm::SparseMmaElementAttr::U8,
            nvvm::SparseMmaElementAttr::U8,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
        (
            "u8",
            "s8",
            ".satfinite",
            nvvm::SparseMmaElementAttr::U8,
            nvvm::SparseMmaElementAttr::S8,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
    ];
    let metadata_modes = [
        ("sp", nvvm::SparseMmaMetadataAttr::Standard),
        ("sp::ordered_metadata", nvvm::SparseMmaMetadataAttr::Ordered),
    ];

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for (metadata_name, metadata) in &metadata_modes {
            for (index, (a_name, b_name, overflow_name, a_element, b_element, overflow)) in
                cases.iter().enumerate()
            {
                let mut ctx = make_test_ctx();
                let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
                let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
                let argument_types = (0..4)
                    .map(|_| i32_ty.into())
                    .chain((0..5).map(|_| u32_ty.into()))
                    .collect();
                let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);

                let selector_value = (index % 2) as u32;
                let selector_op = Operation::new(
                    &mut ctx,
                    mir::MirConstantOp::get_concrete_op_info(),
                    vec![u32_ty.into()],
                    vec![],
                    vec![],
                    0,
                );
                mir::MirConstantOp::new(selector_op).set_attr_value(
                    &ctx,
                    IntegerAttr::new(
                        u32_ty,
                        APInt::from_u32(selector_value, NonZeroUsize::new(32).unwrap()),
                    ),
                );
                selector_op.insert_at_back(entry, &ctx);
                let selector = selector_op.deref(&ctx).get_result(0);

                let operands = (0..9)
                    .map(|operand| entry.deref(&ctx).get_argument(operand))
                    .chain(std::iter::once(selector))
                    .collect();
                let operation = Operation::new(
                    &mut ctx,
                    nvvm::SparseMmaOp::get_concrete_op_info(),
                    vec![i32_ty.into(); 4],
                    operands,
                    vec![],
                    0,
                );
                let mma = nvvm::SparseMmaOp::new(operation);
                mma.set_attr_nvvm_sparse_mma_shape(&ctx, nvvm::SparseMmaShapeAttr::M16n8k32);
                mma.set_attr_nvvm_sparse_mma_accumulator(&ctx, nvvm::SparseMmaAccumulatorAttr::S32);
                mma.set_attr_nvvm_sparse_mma_a_element(&ctx, a_element.clone());
                mma.set_attr_nvvm_sparse_mma_b_element(&ctx, b_element.clone());
                mma.set_attr_nvvm_sparse_mma_a_layout(&ctx, nvvm::SparseMmaLayoutAttr::Row);
                mma.set_attr_nvvm_sparse_mma_b_layout(&ctx, nvvm::SparseMmaLayoutAttr::Col);
                mma.set_attr_nvvm_sparse_mma_overflow(&ctx, overflow.clone());
                mma.set_attr_nvvm_sparse_mma_metadata(&ctx, metadata.clone());
                mma.set_attr_nvvm_sparse_mma_selector(
                    &ctx,
                    nvvm::SparseMmaSelectorAttr::ImmediateZeroOrOne,
                );
                operation.insert_at_back(entry, &ctx);
                append_return(&mut ctx, entry);

                mir_lower::lower_mir_to_llvm_with_options(
                    &mut ctx,
                    module_ptr,
                    mir_lower::LoweringOptions {
                        intrinsic_backend: backend,
                        ..Default::default()
                    },
                )
                .map_err(|error| anyhow::anyhow!("{error}"))?;

                let body = lowered_kernel_body(&ctx, module_ptr);
                let lowered = body
                    .iter()
                    .filter_map(|op| Operation::get_op::<llvm::InlineAsmOp>(*op, &ctx))
                    .collect::<Vec<_>>();
                assert_eq!(lowered.len(), 1, "{backend:?}");
                let asm = &lowered[0];
                let expected_template = format!(
                    "mma.{metadata_name}.sync.aligned.m16n8k32.row.col{overflow_name}.s32.{a_name}.{b_name}.s32 {{$0, $1, $2, $3}}, {{$8, $9}}, {{$10, $11}}, {{$4, $5, $6, $7}}, $12, $13;"
                );
                assert_eq!(
                    asm.get_attr_inline_asm_template(&ctx)
                        .as_deref()
                        .map(|value| String::from(value.clone())),
                    Some(expected_template)
                );
                assert_eq!(
                    asm.get_attr_inline_asm_constraints(&ctx)
                        .as_deref()
                        .map(|value| String::from(value.clone())),
                    Some("=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,n".to_string())
                );
                assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);
                let asm_operation = asm.get_operation().deref(&ctx);
                assert_eq!(asm_operation.get_num_operands(), 10);
                assert_eq!(asm_operation.get_num_results(), 1);
                let lowered_selector = asm_operation.get_operand(9);
                let defining_op = lowered_selector
                    .defining_op()
                    .expect("sparse MMA selector remains an LLVM constant");
                let constant = Operation::get_op::<llvm::ConstantOp>(defining_op, &ctx)
                    .expect("sparse MMA selector remains an LLVM integer constant");
                let attribute = constant.get_value(&ctx);
                let integer = attribute
                    .downcast_ref::<IntegerAttr>()
                    .expect("sparse MMA selector is an integer");
                assert_eq!(integer.value().bw(), 32);
                assert_eq!(integer.value().to_u64(), selector_value as u64);
                assert_eq!(
                    body.iter()
                        .filter(|op| {
                            Operation::get_op::<llvm::ExtractValueOp>(**op, &ctx).is_some()
                        })
                        .count(),
                    4
                );

                let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
                let ir = llvm_export::export::export_module_to_string(&ctx, &module)
                    .expect("generated sparse MMA exports to LLVM IR");
                assert!(ir.contains("asm sideeffect"), "{ir}");
                assert!(ir.contains("{ convergent }"), "{ir}");
            }
        }
    }

    Ok(())
}

#[test]
fn test_generated_sparse_mma_m16n8k64_lowers_to_exact_convergent_inline_ptx()
-> Result<(), anyhow::Error> {
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let cases = [
        (
            "s8",
            "u8",
            "",
            nvvm::SparseMmaElementAttr::S8,
            nvvm::SparseMmaElementAttr::U8,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "u8",
            "s8",
            ".satfinite",
            nvvm::SparseMmaElementAttr::U8,
            nvvm::SparseMmaElementAttr::S8,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
    ];
    let metadata_cases = [
        ("sp", nvvm::SparseMmaMetadataAttr::Standard),
        ("sp::ordered_metadata", nvvm::SparseMmaMetadataAttr::Ordered),
    ];

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for (metadata_name, metadata) in &metadata_cases {
            for (a_name, b_name, overflow_name, a_element, b_element, overflow) in &cases {
                let mut ctx = make_test_ctx();
                let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
                let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
                let argument_types = (0..4)
                    .map(|_| i32_ty.into())
                    .chain((0..9).map(|_| u32_ty.into()))
                    .collect();
                let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);

                let selector_op = Operation::new(
                    &mut ctx,
                    mir::MirConstantOp::get_concrete_op_info(),
                    vec![u32_ty.into()],
                    vec![],
                    vec![],
                    0,
                );
                mir::MirConstantOp::new(selector_op).set_attr_value(
                    &ctx,
                    IntegerAttr::new(u32_ty, APInt::from_u32(0, NonZeroUsize::new(32).unwrap())),
                );
                selector_op.insert_at_back(entry, &ctx);
                let selector = selector_op.deref(&ctx).get_result(0);

                let operands = (0..13)
                    .map(|index| entry.deref(&ctx).get_argument(index))
                    .chain(std::iter::once(selector))
                    .collect();
                let operation = Operation::new(
                    &mut ctx,
                    nvvm::SparseMmaOp::get_concrete_op_info(),
                    vec![i32_ty.into(); 4],
                    operands,
                    vec![],
                    0,
                );
                let mma = nvvm::SparseMmaOp::new(operation);
                mma.set_attr_nvvm_sparse_mma_shape(&ctx, nvvm::SparseMmaShapeAttr::M16n8k64);
                mma.set_attr_nvvm_sparse_mma_accumulator(&ctx, nvvm::SparseMmaAccumulatorAttr::S32);
                mma.set_attr_nvvm_sparse_mma_a_element(&ctx, a_element.clone());
                mma.set_attr_nvvm_sparse_mma_b_element(&ctx, b_element.clone());
                mma.set_attr_nvvm_sparse_mma_a_layout(&ctx, nvvm::SparseMmaLayoutAttr::Row);
                mma.set_attr_nvvm_sparse_mma_b_layout(&ctx, nvvm::SparseMmaLayoutAttr::Col);
                mma.set_attr_nvvm_sparse_mma_overflow(&ctx, overflow.clone());
                mma.set_attr_nvvm_sparse_mma_metadata(&ctx, metadata.clone());
                mma.set_attr_nvvm_sparse_mma_selector(
                    &ctx,
                    nvvm::SparseMmaSelectorAttr::ImmediateZero,
                );
                operation.insert_at_back(entry, &ctx);
                append_return(&mut ctx, entry);

                mir_lower::lower_mir_to_llvm_with_options(
                    &mut ctx,
                    module_ptr,
                    mir_lower::LoweringOptions {
                        intrinsic_backend: backend,
                        ..Default::default()
                    },
                )
                .map_err(|error| anyhow::anyhow!("{error}"))?;

                let body = lowered_kernel_body(&ctx, module_ptr);
                let lowered = body
                    .iter()
                    .filter_map(|op| Operation::get_op::<llvm::InlineAsmOp>(*op, &ctx))
                    .collect::<Vec<_>>();
                assert_eq!(lowered.len(), 1, "{backend:?}");
                let asm = &lowered[0];
                let expected_template = format!(
                    "mma.{metadata_name}.sync.aligned.m16n8k64.row.col{overflow_name}.s32.{a_name}.{b_name}.s32 {{$0, $1, $2, $3}}, {{$8, $9, $10, $11}}, {{$12, $13, $14, $15}}, {{$4, $5, $6, $7}}, $16, $17;"
                );
                assert_eq!(
                    asm.get_attr_inline_asm_template(&ctx)
                        .as_deref()
                        .map(|value| String::from(value.clone())),
                    Some(expected_template)
                );
                assert_eq!(
                    asm.get_attr_inline_asm_constraints(&ctx)
                        .as_deref()
                        .map(|value| String::from(value.clone())),
                    Some("=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r,r,n".to_string())
                );
                assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);
                let asm_operation = asm.get_operation().deref(&ctx);
                assert_eq!(asm_operation.get_num_operands(), 14);
                assert_eq!(asm_operation.get_num_results(), 1);
                let lowered_selector = asm_operation.get_operand(13);
                let defining_op = lowered_selector
                    .defining_op()
                    .expect("sparse MMA selector remains an LLVM constant");
                let constant = Operation::get_op::<llvm::ConstantOp>(defining_op, &ctx)
                    .expect("sparse MMA selector remains an LLVM integer constant");
                let attribute = constant.get_value(&ctx);
                let integer = attribute
                    .downcast_ref::<IntegerAttr>()
                    .expect("sparse MMA selector is an integer");
                assert_eq!(integer.value().bw(), 32);
                assert_eq!(integer.value().to_u64(), 0);
                assert_eq!(
                    body.iter()
                        .filter(|op| {
                            Operation::get_op::<llvm::ExtractValueOp>(**op, &ctx).is_some()
                        })
                        .count(),
                    4
                );

                let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
                let ir = llvm_export::export::export_module_to_string(&ctx, &module)
                    .expect("generated sparse MMA exports to LLVM IR");
                assert!(ir.contains("asm sideeffect"), "{ir}");
                assert!(ir.contains("{ convergent }"), "{ir}");
            }
        }
    }

    Ok(())
}

#[test]
fn test_generated_sparse_mma_m16n8k64_int4_lowers_both_metadata_modes() -> Result<(), anyhow::Error>
{
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let cases = [
        (
            "s4",
            "s4",
            "",
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "s4",
            "u4",
            "",
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "u4",
            "u4",
            "",
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "u4",
            "s4",
            "",
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "s4",
            "s4",
            ".satfinite",
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
        (
            "s4",
            "u4",
            ".satfinite",
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
        (
            "u4",
            "u4",
            ".satfinite",
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
        (
            "u4",
            "s4",
            ".satfinite",
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
    ];

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for (metadata_name, metadata) in [
            ("sp", nvvm::SparseMmaMetadataAttr::Standard),
            ("sp::ordered_metadata", nvvm::SparseMmaMetadataAttr::Ordered),
        ] {
            for (index, (a_name, b_name, overflow_name, a_element, b_element, overflow)) in
                cases.iter().enumerate()
            {
                let mut ctx = make_test_ctx();
                let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
                let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
                let argument_types = (0..4)
                    .map(|_| i32_ty.into())
                    .chain((0..5).map(|_| u32_ty.into()))
                    .collect();
                let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);

                let selector_value = (index % 2) as u32;
                let selector_op = Operation::new(
                    &mut ctx,
                    mir::MirConstantOp::get_concrete_op_info(),
                    vec![u32_ty.into()],
                    vec![],
                    vec![],
                    0,
                );
                mir::MirConstantOp::new(selector_op).set_attr_value(
                    &ctx,
                    IntegerAttr::new(
                        u32_ty,
                        APInt::from_u32(selector_value, NonZeroUsize::new(32).unwrap()),
                    ),
                );
                selector_op.insert_at_back(entry, &ctx);
                let selector = selector_op.deref(&ctx).get_result(0);

                let operands = (0..9)
                    .map(|operand| entry.deref(&ctx).get_argument(operand))
                    .chain(std::iter::once(selector))
                    .collect();
                let operation = Operation::new(
                    &mut ctx,
                    nvvm::SparseMmaOp::get_concrete_op_info(),
                    vec![i32_ty.into(); 4],
                    operands,
                    vec![],
                    0,
                );
                let mma = nvvm::SparseMmaOp::new(operation);
                mma.set_attr_nvvm_sparse_mma_shape(&ctx, nvvm::SparseMmaShapeAttr::M16n8k64);
                mma.set_attr_nvvm_sparse_mma_accumulator(&ctx, nvvm::SparseMmaAccumulatorAttr::S32);
                mma.set_attr_nvvm_sparse_mma_a_element(&ctx, a_element.clone());
                mma.set_attr_nvvm_sparse_mma_b_element(&ctx, b_element.clone());
                mma.set_attr_nvvm_sparse_mma_a_layout(&ctx, nvvm::SparseMmaLayoutAttr::Row);
                mma.set_attr_nvvm_sparse_mma_b_layout(&ctx, nvvm::SparseMmaLayoutAttr::Col);
                mma.set_attr_nvvm_sparse_mma_overflow(&ctx, overflow.clone());
                mma.set_attr_nvvm_sparse_mma_metadata(&ctx, metadata.clone());
                mma.set_attr_nvvm_sparse_mma_selector(
                    &ctx,
                    nvvm::SparseMmaSelectorAttr::ImmediateZeroOrOne,
                );
                operation.insert_at_back(entry, &ctx);
                append_return(&mut ctx, entry);

                mir_lower::lower_mir_to_llvm_with_options(
                    &mut ctx,
                    module_ptr,
                    mir_lower::LoweringOptions {
                        intrinsic_backend: backend,
                        ..Default::default()
                    },
                )
                .map_err(|error| anyhow::anyhow!("{error}"))?;

                let body = lowered_kernel_body(&ctx, module_ptr);
                let lowered = body
                    .iter()
                    .filter_map(|op| Operation::get_op::<llvm::InlineAsmOp>(*op, &ctx))
                    .collect::<Vec<_>>();
                assert_eq!(lowered.len(), 1, "{backend:?}");
                let asm = &lowered[0];
                let expected_template = format!(
                    "mma.{metadata_name}.sync.aligned.m16n8k64.row.col{overflow_name}.s32.{a_name}.{b_name}.s32 {{$0, $1, $2, $3}}, {{$8, $9}}, {{$10, $11}}, {{$4, $5, $6, $7}}, $12, $13;"
                );
                assert_eq!(
                    asm.get_attr_inline_asm_template(&ctx)
                        .as_deref()
                        .map(|value| String::from(value.clone())),
                    Some(expected_template)
                );
                assert_eq!(
                    asm.get_attr_inline_asm_constraints(&ctx)
                        .as_deref()
                        .map(|value| String::from(value.clone())),
                    Some("=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,n".to_string())
                );
                assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);
                let asm_operation = asm.get_operation().deref(&ctx);
                assert_eq!(asm_operation.get_num_operands(), 10);
                assert_eq!(asm_operation.get_num_results(), 1);
                let lowered_selector = asm_operation.get_operand(9);
                let defining_op = lowered_selector
                    .defining_op()
                    .expect("sparse MMA selector remains an LLVM constant");
                let constant = Operation::get_op::<llvm::ConstantOp>(defining_op, &ctx)
                    .expect("sparse MMA selector remains an LLVM integer constant");
                let attribute = constant.get_value(&ctx);
                let integer = attribute
                    .downcast_ref::<IntegerAttr>()
                    .expect("sparse MMA selector is an integer");
                assert_eq!(integer.value().to_u64(), selector_value as u64);

                let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
                let ir = llvm_export::export::export_module_to_string(&ctx, &module)
                    .expect("generated sparse MMA exports to LLVM IR");
                assert!(ir.contains("asm sideeffect"), "{ir}");
                assert!(ir.contains("{ convergent }"), "{ir}");
            }
        }
    }

    Ok(())
}

#[test]
fn test_generated_sparse_mma_m16n8k128_int4_lowers_both_metadata_modes() -> Result<(), anyhow::Error>
{
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::utils::apint::APInt;
    use std::num::NonZeroUsize;

    let cases = [
        (
            "s4",
            "s4",
            "",
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "s4",
            "u4",
            "",
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "u4",
            "u4",
            "",
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "u4",
            "s4",
            "",
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaOverflowAttr::Wrapping,
        ),
        (
            "s4",
            "s4",
            ".satfinite",
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
        (
            "s4",
            "u4",
            ".satfinite",
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
        (
            "u4",
            "u4",
            ".satfinite",
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
        (
            "u4",
            "s4",
            ".satfinite",
            nvvm::SparseMmaElementAttr::U4,
            nvvm::SparseMmaElementAttr::S4,
            nvvm::SparseMmaOverflowAttr::Satfinite,
        ),
    ];

    for backend in [
        mir_lower::IntrinsicBackend::LlvmNvptx,
        mir_lower::IntrinsicBackend::LibNvvm,
    ] {
        for (metadata_name, metadata) in [
            ("sp", nvvm::SparseMmaMetadataAttr::Standard),
            ("sp::ordered_metadata", nvvm::SparseMmaMetadataAttr::Ordered),
        ] {
            for (a_name, b_name, overflow_name, a_element, b_element, overflow) in &cases {
                let mut ctx = make_test_ctx();
                let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signed);
                let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
                let argument_types = (0..4)
                    .map(|_| i32_ty.into())
                    .chain((0..9).map(|_| u32_ty.into()))
                    .collect();
                let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);

                let selector_op = Operation::new(
                    &mut ctx,
                    mir::MirConstantOp::get_concrete_op_info(),
                    vec![u32_ty.into()],
                    vec![],
                    vec![],
                    0,
                );
                mir::MirConstantOp::new(selector_op).set_attr_value(
                    &ctx,
                    IntegerAttr::new(u32_ty, APInt::from_u32(0, NonZeroUsize::new(32).unwrap())),
                );
                selector_op.insert_at_back(entry, &ctx);
                let selector = selector_op.deref(&ctx).get_result(0);

                let operands = (0..13)
                    .map(|operand| entry.deref(&ctx).get_argument(operand))
                    .chain(std::iter::once(selector))
                    .collect();
                let operation = Operation::new(
                    &mut ctx,
                    nvvm::SparseMmaOp::get_concrete_op_info(),
                    vec![i32_ty.into(); 4],
                    operands,
                    vec![],
                    0,
                );
                let mma = nvvm::SparseMmaOp::new(operation);
                mma.set_attr_nvvm_sparse_mma_shape(&ctx, nvvm::SparseMmaShapeAttr::M16n8k128);
                mma.set_attr_nvvm_sparse_mma_accumulator(&ctx, nvvm::SparseMmaAccumulatorAttr::S32);
                mma.set_attr_nvvm_sparse_mma_a_element(&ctx, a_element.clone());
                mma.set_attr_nvvm_sparse_mma_b_element(&ctx, b_element.clone());
                mma.set_attr_nvvm_sparse_mma_a_layout(&ctx, nvvm::SparseMmaLayoutAttr::Row);
                mma.set_attr_nvvm_sparse_mma_b_layout(&ctx, nvvm::SparseMmaLayoutAttr::Col);
                mma.set_attr_nvvm_sparse_mma_overflow(&ctx, overflow.clone());
                mma.set_attr_nvvm_sparse_mma_metadata(&ctx, metadata.clone());
                mma.set_attr_nvvm_sparse_mma_selector(
                    &ctx,
                    nvvm::SparseMmaSelectorAttr::ImmediateZero,
                );
                operation.insert_at_back(entry, &ctx);
                append_return(&mut ctx, entry);

                mir_lower::lower_mir_to_llvm_with_options(
                    &mut ctx,
                    module_ptr,
                    mir_lower::LoweringOptions {
                        intrinsic_backend: backend,
                        ..Default::default()
                    },
                )
                .map_err(|error| anyhow::anyhow!("{error}"))?;

                let body = lowered_kernel_body(&ctx, module_ptr);
                let lowered = body
                    .iter()
                    .filter_map(|op| Operation::get_op::<llvm::InlineAsmOp>(*op, &ctx))
                    .collect::<Vec<_>>();
                assert_eq!(lowered.len(), 1, "{backend:?}");
                let asm = &lowered[0];
                let expected_template = format!(
                    "mma.{metadata_name}.sync.aligned.m16n8k128.row.col{overflow_name}.s32.{a_name}.{b_name}.s32 {{$0, $1, $2, $3}}, {{$8, $9, $10, $11}}, {{$12, $13, $14, $15}}, {{$4, $5, $6, $7}}, $16, $17;"
                );
                assert_eq!(
                    asm.get_attr_inline_asm_template(&ctx)
                        .as_deref()
                        .map(|value| String::from(value.clone())),
                    Some(expected_template)
                );
                assert_eq!(
                    asm.get_attr_inline_asm_constraints(&ctx)
                        .as_deref()
                        .map(|value| String::from(value.clone())),
                    Some("=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r,r,r,r,n".to_string())
                );
                assert_eq!(llvm::asm_kind(&ctx, asm), llvm::AsmKind::Convergent);
                let asm_operation = asm.get_operation().deref(&ctx);
                assert_eq!(asm_operation.get_num_operands(), 14);
                assert_eq!(asm_operation.get_num_results(), 1);
                let lowered_selector = asm_operation.get_operand(13);
                let defining_op = lowered_selector
                    .defining_op()
                    .expect("sparse MMA selector remains an LLVM constant");
                let constant = Operation::get_op::<llvm::ConstantOp>(defining_op, &ctx)
                    .expect("sparse MMA selector remains an LLVM integer constant");
                let attribute = constant.get_value(&ctx);
                let integer = attribute
                    .downcast_ref::<IntegerAttr>()
                    .expect("sparse MMA selector is an integer");
                assert_eq!(integer.value().to_u64(), 0);

                let module = Operation::get_op::<ModuleOp>(module_ptr, &ctx).unwrap();
                let ir = llvm_export::export::export_module_to_string(&ctx, &module)
                    .expect("generated sparse MMA exports to LLVM IR");
                assert!(ir.contains("asm sideeffect"), "{ir}");
                assert!(ir.contains("{ convergent }"), "{ir}");
            }
        }
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
    for format in [
        nvvm::PackedAtomicFormatAttr::F16x2,
        nvvm::PackedAtomicFormatAttr::Bf16x2,
    ] {
        nvvm::PackedAtomicAddOp::build(&mut ctx, address, addend, format)
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
                assert!(
                    !template.contains("atom.cas"),
                    "packed atomic exact-native lowering must not use a CAS loop: {template}"
                );
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

    assert_eq!(lowered.len(), expected.len() * 2);
    for instruction in expected {
        let matches: Vec<_> = lowered
            .iter()
            .filter(|(template, _, _, _, _)| template == instruction)
            .collect();
        assert_eq!(
            matches.len(),
            2,
            "legacy and generated paths must have exact PTX parity for {instruction}"
        );
        for (_, constraints, kind, operands, results) in matches {
            assert_eq!(constraints, "=r,l,r,~{memory}");
            assert_eq!(*kind, llvm::AsmKind::SideEffect);
            assert_eq!(*operands, 2);
            assert_eq!(*results, 1);
        }
    }

    Ok(())
}

#[test]
fn test_generated_packed_atomic_add_libnvvm_route_is_exact() -> Result<(), anyhow::Error> {
    use dialect_mir::types::MirPtrType;
    use pliron::builtin::types::{IntegerType, Signedness};

    let mut ctx = make_test_ctx();
    let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
    let ptr_ty = MirPtrType::get_generic(&mut ctx, u32_ty.into(), true);
    let (module_ptr, entry) = build_test_kernel(&mut ctx, vec![ptr_ty.into(), u32_ty.into()]);
    let address = entry.deref(&ctx).get_argument(0);
    let addend = entry.deref(&ctx).get_argument(1);
    for format in [
        nvvm::PackedAtomicFormatAttr::F16x2,
        nvvm::PackedAtomicFormatAttr::Bf16x2,
    ] {
        nvvm::PackedAtomicAddOp::build(&mut ctx, address, addend, format)
            .insert_at_back(entry, &ctx);
    }
    append_return(&mut ctx, entry);

    mir_lower::lower_mir_to_llvm_with_options(
        &mut ctx,
        module_ptr,
        mir_lower::LoweringOptions {
            intrinsic_backend: mir_lower::IntrinsicBackend::LibNvvm,
            ..Default::default()
        },
    )
    .map_err(|error| anyhow::anyhow!("{error}"))?;

    let lowered = lowered_kernel_body(&ctx, module_ptr)
        .into_iter()
        .filter_map(|op| {
            let asm = Operation::get_op::<llvm::InlineAsmOp>(op, &ctx)?;
            let template = asm
                .get_attr_inline_asm_template(&ctx)
                .map(|value| String::from((*value).clone()))?;
            template.starts_with("atom.global.add.noftz.").then(|| {
                (
                    template,
                    asm.get_attr_inline_asm_constraints(&ctx)
                        .map(|value| String::from((*value).clone())),
                    llvm::asm_kind(&ctx, &asm),
                )
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(lowered.len(), 2);
    assert!(
        lowered
            .iter()
            .any(|(template, _, _)| { template == "atom.global.add.noftz.f16x2 $0, [$1], $2;" })
    );
    assert!(
        lowered
            .iter()
            .any(|(template, _, _)| { template == "atom.global.add.noftz.bf16x2 $0, [$1], $2;" })
    );
    for (_, constraints, kind) in lowered {
        assert_eq!(constraints.as_deref(), Some("=r,l,r,~{memory}"));
        assert_eq!(kind, llvm::AsmKind::SideEffect);
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

#[test]
fn test_mma_m16n8k32_s32_s8_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    let mut ctx = make_test_ctx();
    let i32_ty = pliron::builtin::types::IntegerType::get(
        &ctx,
        32,
        pliron::builtin::types::Signedness::Signless,
    );
    let argument_types = (0..10).map(|_| i32_ty.into()).collect();
    let (module_ptr, entry) = build_test_kernel(&mut ctx, argument_types);
    let operands = (0..10)
        .map(|index| entry.deref(&ctx).get_argument(index))
        .collect();

    let op = Operation::new(
        &mut ctx,
        nvvm::MmaM16N8K32S32S8Op::get_concrete_op_info(),
        vec![i32_ty.into(); 4],
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
                    .is_some_and(|t| t.contains("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"))
                {
                    continue;
                }
                found += 1;
                let template = template.expect("MMA inline asm must have a template");
                assert_eq!(
                    template,
                    concat!(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ",
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
                    Some("=r,=r,=r,=r,r,r,r,r,r,r,r,r,r,r")
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
