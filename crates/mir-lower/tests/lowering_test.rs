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
    let func_ty = pliron::builtin::types::FunctionType::get(&mut ctx, vec![], vec![]);

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
        &mut ctx,
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
    let func_ty = pliron::builtin::types::FunctionType::get(&mut ctx, vec![], vec![]);

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
        &mut ctx,
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

#[test]
fn test_threadfence_system_lowers_to_inline_asm() -> Result<(), anyhow::Error> {
    let mut ctx = Context::new();
    dialect_mir::register(&mut ctx);
    dialect_nvvm::register(&mut ctx);
    mir_lower::register(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_ptr = module.get_operation();

    let func_name = "kernel_func";
    let func_ty = pliron::builtin::types::FunctionType::get(&mut ctx, vec![], vec![]);

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

    let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless);
    let shared_ptr_ty = MirPtrType::get_shared(&mut ctx, i32_ty.into(), true);
    let generic_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty.into(), true);

    // Callee: takes a *mut i32 in addrspace(3), returns ().
    let callee_func_ty = FunctionType::get(&mut ctx, vec![shared_ptr_ty.into()], vec![]);
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
    let caller_func_ty = FunctionType::get(&mut ctx, vec![generic_ptr_ty.into()], vec![]);
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
        &mut ctx,
        32,
        pliron::builtin::types::Signedness::Signed,
    );
    let u32_unsigned = pliron::builtin::types::IntegerType::get(
        &mut ctx,
        32,
        pliron::builtin::types::Signedness::Unsigned,
    );
    let bool_ty = pliron::builtin::types::IntegerType::get(
        &mut ctx,
        1,
        pliron::builtin::types::Signedness::Signless,
    );

    // Args: (f32, f32, i32, u32). The integer args carry pre-conversion
    // signedness, which is what selects signed vs unsigned icmp predicates.
    let arg_tys: Vec<pliron::context::Ptr<pliron::r#type::TypeObj>> = vec![
        f32_ty.into(),
        f32_ty.into(),
        i32_signed.into(),
        u32_unsigned.into(),
    ];
    let func_name = "cmp_func";
    let func_ty = pliron::builtin::types::FunctionType::get(&mut ctx, arg_tys.clone(), vec![]);

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
                    assert_eq!(
                        fcmp.fast_math_flags(&ctx),
                        FastmathFlagsAttr::default(),
                        "fcmp must carry empty fastmath flags: nnan would poison NaN checks"
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
