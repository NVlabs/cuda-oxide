/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Proof that the standalone API can compile a module whose float intrinsics
//! lower to libdevice, and that opting in changes nothing else.
//!
//! Two routes reach libdevice and both are covered here:
//!
//!   1. A Rust intrinsic placeholder (`sqrtf32`), which `mir-lower` maps to
//!      `__nv_sqrtf` through `libdevice_name`. This is what a frontend gets
//!      for free by emitting the placeholder callee.
//!   2. A `__nv_*` symbol the frontend names itself. Several libdevice
//!      functions have no Rust intrinsic and so no placeholder to reach them
//!      through; `__nv_erff` is one. Lowering self-declares the symbol at
//!      `mir-lower/src/convert/ops/call.rs:697`, and the pre-lowering scan in
//!      `typed_mir_uses_libdevice` sees the same callee, so the two libdevice
//!      detectors agree and the consistency refusal does not fire.

use cuda_oxide_codegen::experimental::{
    CodegenModule, CompileError, CompileOptions, Compiler, Linking, Target,
};

use dialect_mir::ops::{MirCallOp, MirFuncOp, MirLoadOp, MirPtrOffsetOp, MirReturnOp, MirStoreOp};
use dialect_mir::types::MirPtrType;
use dialect_nvvm::ops::ReadPtxSregTidXOp;
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{StringAttr, TypeAttr},
        op_interfaces::SymbolOpInterface,
        types::{FP32Type, FunctionType, IntegerType, Signedness},
    },
    context::Context,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
};

/// Build `kernel(a: *const f32, out: *mut f32) { out[tid] = callee(a[tid]) }`.
///
/// `callee` is a `mir.call` target, either a Rust intrinsic placeholder such
/// as `dialect_mir::rust_intrinsics::CALLEE_SQRT_F32` or a bare `__nv_*`
/// symbol. Indexing is by `tid.x` alone, which keeps the kernel to the
/// smallest shape that still produces a real load, a real call and a real
/// store.
fn build_unary_call_kernel(module: &mut CodegenModule, callee: &str) {
    module.edit(|ctx, module| {
        let module_op = module.get_operation();
        let module_region = module_op.deref(ctx).get_region(0);
        let module_block = {
            let existing = {
                let region = module_region.deref(ctx);
                region.iter(ctx).next()
            };
            if let Some(block) = existing {
                block
            } else {
                let block = BasicBlock::new(ctx, None, vec![]);
                block.insert_at_back(module_region, ctx);
                block
            }
        };

        let f32_ty = FP32Type::get(ctx);
        let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
        let in_ptr_ty = MirPtrType::get_global(ctx, f32_ty.into(), false);
        let out_ptr_ty = MirPtrType::get_global(ctx, f32_ty.into(), true);

        let func_type = FunctionType::get(ctx, vec![in_ptr_ty.into(), out_ptr_ty.into()], vec![]);
        let func = {
            let op = Operation::new(
                ctx,
                MirFuncOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                1,
            );
            let func = MirFuncOp::new(ctx, op, TypeAttr::new(func_type.into()));
            func.set_symbol_name(ctx, "unary_kernel".try_into().unwrap());
            func
        };

        let entry = BasicBlock::new(ctx, None, vec![in_ptr_ty.into(), out_ptr_ty.into()]);
        let func_region = func.get_operation().deref(ctx).get_region(0);
        entry.insert_at_front(func_region, ctx);

        let a = entry.deref(ctx).get_argument(0);
        let out = entry.deref(ctx).get_argument(1);

        let emit = |ctx: &mut Context,
                    info: (
            fn(pliron::context::Ptr<Operation>) -> pliron::op::OpObj,
            std::any::TypeId,
        ),
                    results: Vec<pliron::r#type::TypeHandle>,
                    operands: Vec<pliron::value::Value>|
         -> Option<pliron::value::Value> {
            let op = Operation::new(ctx, info, results.clone(), operands, vec![], 0);
            let res = if results.is_empty() {
                None
            } else {
                Some(op.deref(ctx).get_result(0))
            };
            op.insert_at_back(entry, ctx);
            res
        };

        let i = emit(
            ctx,
            ReadPtxSregTidXOp::get_concrete_op_info(),
            vec![i32_ty.into()],
            vec![],
        )
        .unwrap();

        let a_ptr = emit(
            ctx,
            MirPtrOffsetOp::get_concrete_op_info(),
            vec![in_ptr_ty.into()],
            vec![a, i],
        )
        .unwrap();
        let a_val = emit(
            ctx,
            MirLoadOp::get_concrete_op_info(),
            vec![f32_ty.into()],
            vec![a_ptr],
        )
        .unwrap();

        // The call carries its signature through the operation's own operand
        // and result types; `MirCallOp` verifies only that a callee attribute
        // is present.
        let call_op = Operation::new(
            ctx,
            MirCallOp::get_concrete_op_info(),
            vec![f32_ty.into()],
            vec![a_val],
            vec![],
            0,
        );
        let call = MirCallOp::new(call_op);
        call.set_attr_callee(ctx, StringAttr::new(callee.to_string()));
        call_op.insert_at_back(entry, ctx);
        let result = call_op.deref(ctx).get_result(0);

        let out_ptr = emit(
            ctx,
            MirPtrOffsetOp::get_concrete_op_info(),
            vec![out_ptr_ty.into()],
            vec![out, i],
        )
        .unwrap();
        emit(
            ctx,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![out_ptr, result],
        );
        emit(ctx, MirReturnOp::get_concrete_op_info(), vec![], vec![]);

        func.get_operation().insert_at_back(module_block, ctx);
    });
    module.mark_kernel_entry("unary_kernel").unwrap();
}

/// Locate `ptxas`, matching `tests/spine_kernel_ptx.rs`.
fn find_ptxas() -> std::path::PathBuf {
    ["CUDA_TOOLKIT_PATH", "CUDA_HOME"]
        .iter()
        .filter_map(|var| std::env::var(var).ok())
        .filter(|root| !root.trim().is_empty())
        .map(|root| std::path::PathBuf::from(root).join("bin/ptxas"))
        .chain(std::iter::once(std::path::PathBuf::from(
            "/usr/local/cuda/bin/ptxas",
        )))
        .find(|candidate| candidate.exists())
        .unwrap_or_else(|| std::path::PathBuf::from("ptxas"))
}

/// Write `ptx` to a scratch file and require `ptxas -arch=sm_120` to accept it.
fn assert_ptxas_accepts(ptx: &[u8], label: &str) {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!(
        "libdevice_linking_{label}_{}_{}",
        std::process::id(),
        unique
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let ptx_path = dir.join("kernel.ptx");
    let cubin_path = dir.join("kernel.cubin");
    std::fs::write(&ptx_path, ptx).unwrap();

    let ptxas = find_ptxas();
    let result = std::process::Command::new(&ptxas)
        .arg("-arch=sm_120")
        .arg("--compile-only")
        .arg(&ptx_path)
        .arg("-o")
        .arg(&cubin_path)
        .output();

    let _ = std::fs::remove_dir_all(&dir);

    let out = result.unwrap_or_else(|error| {
        panic!(
            "could not run {}: {error}\nThis test needs `ptxas` from a CUDA \
             toolkit (no GPU required). Point CUDA_TOOLKIT_PATH or CUDA_HOME \
             at the install root, or put `ptxas` on PATH.",
            ptxas.display()
        )
    });
    assert!(
        out.status.success(),
        "ptxas rejected the {label} PTX:\nstderr:\n{}\n\nPTX:\n{}",
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(ptx)
    );
}

#[test]
fn self_contained_linking_still_rejects_a_libdevice_kernel() {
    let mut module = CodegenModule::new("sqrt_module").unwrap();
    build_unary_call_kernel(&mut module, dialect_mir::rust_intrinsics::CALLEE_SQRT_F32);
    let compiler = Compiler::discover().expect("LLVM 21+ llc/opt are installed");
    let options = CompileOptions::new(Target::parse("sm_120").unwrap());

    let error = compiler
        .compile(&mut module, &options)
        .expect_err("the default policy has no link step");
    match error {
        CompileError::UnsupportedLinking { symbols } => assert!(
            symbols.iter().any(|symbol| symbol == "__nv_sqrtf"),
            "the rejection names the unresolved libdevice symbol: {symbols:?}"
        ),
        other => panic!("expected UnsupportedLinking, got {other}"),
    }
}

#[test]
fn libdevice_linking_resolves_a_rust_intrinsic_kernel() {
    let mut module = CodegenModule::new("sqrt_module").unwrap();
    build_unary_call_kernel(&mut module, dialect_mir::rust_intrinsics::CALLEE_SQRT_F32);
    let compiler = Compiler::discover().expect("LLVM 21+ llc/opt are installed");
    let options =
        CompileOptions::new(Target::parse("sm_120").unwrap()).with_linking(Linking::Libdevice);

    let ptx = compiler
        .compile(&mut module, &options)
        .expect("libdevice linking resolves __nv_sqrtf")
        .into_ptx();
    let text = String::from_utf8(ptx.clone()).expect("PTX is utf-8");

    assert!(
        text.contains(".visible .entry"),
        "kernel entry present:\n{text}"
    );
    assert!(
        text.contains("sqrt.rn.f32"),
        "libdevice sqrt was inlined to a native PTX instruction:\n{text}"
    );
    assert!(
        !text.contains(".extern .func __nv_"),
        "no unresolved libdevice declaration survives into the artifact:\n{text}"
    );

    assert_ptxas_accepts(&ptx, "sqrt");
}

#[test]
fn libdevice_linking_resolves_a_frontend_declared_symbol() {
    // `__nv_erff` has no Rust intrinsic and so no placeholder callee. A
    // frontend reaches it by naming the symbol, which is the route the
    // `__nv_` prefix filter admits.
    let mut module = CodegenModule::new("erf_module").unwrap();
    build_unary_call_kernel(&mut module, "__nv_erff");
    let compiler = Compiler::discover().expect("LLVM 21+ llc/opt are installed");
    let options =
        CompileOptions::new(Target::parse("sm_120").unwrap()).with_linking(Linking::Libdevice);

    let ptx = compiler
        .compile(&mut module, &options)
        .expect("libdevice linking resolves a frontend-declared __nv_ symbol")
        .into_ptx();
    let text = String::from_utf8(ptx.clone()).expect("PTX is utf-8");

    assert!(
        text.contains(".visible .entry"),
        "kernel entry present:\n{text}"
    );
    assert!(
        !text.contains(".extern .func __nv_"),
        "the declared symbol was resolved, not left extern:\n{text}"
    );
    // erff is a polynomial evaluation in libdevice, so the linked body
    // arrives as real arithmetic rather than a single instruction.
    assert!(
        text.contains("fma.rn.f32") || text.contains("mul.f32"),
        "the linked erff body is present:\n{text}"
    );

    assert_ptxas_accepts(&ptx, "erf");
}

/// Return the sole top-level block of `module`, creating it if the module is
/// still empty. Mirrors `module_block` in `tests/compile_to_ptx.rs`; kept
/// separate here rather than lifted out of `build_unary_call_kernel`, which
/// stays as Task 6 left it.
fn module_block(
    ctx: &mut Context,
    module: &pliron::builtin::ops::ModuleOp,
) -> pliron::context::Ptr<BasicBlock> {
    let module_region = module.get_operation().deref(ctx).get_region(0);
    let existing = {
        let region = module_region.deref(ctx);
        region.iter(ctx).next()
    };
    existing.unwrap_or_else(|| {
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(module_region, ctx);
        block
    })
}

/// Insert a body-less LLVM-dialect function declaration named `name` into
/// `module`. Mirrors `add_llvm_declaration` in `tests/compile_to_ptx.rs`.
///
/// This is the only way to place a genuinely unresolved external symbol into
/// a standalone-API module: MIR lowering self-declares a callee only when it
/// resolves to a `__nv_*` name, so a `mir.call` to any other bare symbol has
/// no matching declaration and fails LLVM-dialect verification before the
/// unresolved-symbol scan that `UnsupportedLinking` comes from ever runs.
/// Declaring the symbol directly, with no call site at all, is what
/// `standalone_v1_rejects_libdevice_and_other_externs` already relies on to
/// reach that scan for `user_device_external`.
fn add_llvm_declaration(module: &mut CodegenModule, name: &str) {
    module.edit(|ctx, module| {
        use llvm_export::{ops::FuncOp, types::FuncType};

        let block = module_block(ctx, module);
        let i32_type = IntegerType::get(ctx, 32, Signedness::Signless);
        let function_type = FuncType::get(ctx, i32_type.into(), vec![i32_type.into()], false);
        let function = FuncOp::new(ctx, name.try_into().unwrap(), function_type);
        function.get_operation().insert_at_back(block, ctx);
    });
}

#[test]
fn libdevice_linking_still_rejects_a_device_extern() {
    // The opt-in narrows the rejection to symbols that are not libdevice. A
    // module carrying both kinds must lose only the libdevice one, or
    // `UnsupportedLinking` would stop meaning anything for device externs.
    let mut module = CodegenModule::new("extern_module").unwrap();
    add_llvm_declaration(&mut module, "__nv_erff");
    add_llvm_declaration(&mut module, "my_device_extern");
    let compiler = Compiler::discover().expect("LLVM 21+ llc/opt are installed");
    let options =
        CompileOptions::new(Target::parse("sm_120").unwrap()).with_linking(Linking::Libdevice);

    let error = compiler
        .compile(&mut module, &options)
        .expect_err("a non-libdevice extern is still unresolvable");
    match error {
        CompileError::UnsupportedLinking { symbols } => {
            assert_eq!(symbols, ["my_device_extern"]);
        }
        other => panic!("expected UnsupportedLinking, got {other}"),
    }
}
