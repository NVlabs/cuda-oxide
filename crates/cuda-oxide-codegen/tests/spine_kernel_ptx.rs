/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Deliverable A proof: the hand-written elementwise-add "spine" kernel
//! compiles to a launchable PTX `.entry` that `ptxas -arch=sm_120` accepts.
//!
//! This builds the irreducible `out[i] = a[i] + b[i]` kernel directly in
//! `dialect-mir` + `dialect-nvvm` (no rustc, no CubeCL), drives it through
//! the experimental `Compiler`, and asserts the emitted PTX carries a
//! `.visible .entry` for `sm_120` that `ptxas` compiles to a cubin. The kernel
//! constructed here is the recipe a later CubeCL-walk task mirrors.
//!
//! ## Kernel-entry marking mechanism (Step 1 discovery)
//!
//! A `MirFuncOp` becomes an LLVM `ptx_kernel` (PTX `.visible .entry`) purely by
//! carrying a `gpu_kernel` *attribute* on the func op. There is no calling
//! convention to set by hand and no naming convention. The chain is:
//!
//!   1. `mir-lower`'s `is_kernel_func` (crates/mir-lower/src/convert/types.rs)
//!      returns `true` iff the func op's `attributes` contain a `StringAttr`
//!      under the identifier `gpu_kernel`. The *value* is not inspected, only
//!      presence; the rest of the pipeline writes `"true"`.
//!   2. During lowering (lowering.rs:132) `propagate_kernel_attrs` copies a
//!      `gpu_kernel="true"` `StringAttr` onto the produced `llvm::FuncOp` (plus
//!      any optional `cluster_dim_*`/`maxntid`/`minctasm` ints).
//!   3. `llvm-export`'s `PtxExportConfig::emit_ptx_kernel_keyword()` is `true`,
//!      so a func carrying `gpu_kernel` is emitted with the `ptx_kernel`
//!      calling convention, which `llc` renders as `.visible .entry`.
//!
//! The owned module's `mark_kernel_entry` method owns this internal marker spelling.

use cuda_oxide_codegen::experimental::{CodegenModule, CompileOptions, Compiler, Target};

use dialect_mir::ops::{
    MirAddOp, MirFuncOp, MirLoadOp, MirMulOp, MirPtrOffsetOp, MirReturnOp, MirStoreOp,
};
use dialect_mir::types::MirPtrType;
use dialect_nvvm::ops::{ReadPtxSregCtaidXOp, ReadPtxSregNtidXOp, ReadPtxSregTidXOp};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::TypeAttr,
        op_interfaces::SymbolOpInterface,
        types::{FP32Type, FunctionType, IntegerType, Signedness},
    },
    context::Context,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
};

/// Build the elementwise-add spine kernel in an owned codegen module.
///
/// Signature (all pointers in the GLOBAL address space):
///   `add_kernel(a: *const f32, b: *const f32, out: *mut f32)`
/// Body: `let i = ctaid.x * ntid.x + tid.x; out[i] = a[i] + b[i];`
fn build_add_kernel(module: &mut CodegenModule) {
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

        // Scalar + pointer types. Index math is 32-bit to match the i32 special
        // registers; `mir.ptr_offset` accepts any integer index (lowered to a GEP).
        let f32_ty = FP32Type::get(ctx);
        let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
        // a, b are read-only (`is_mutable = false`); out is writable (`true`).
        let in_ptr_ty = MirPtrType::get_global(ctx, f32_ty.into(), false);
        let out_ptr_ty = MirPtrType::get_global(ctx, f32_ty.into(), true);

        // Function type: (a, b, out) -> void.
        let func_type = FunctionType::get(
            ctx,
            vec![in_ptr_ty.into(), in_ptr_ty.into(), out_ptr_ty.into()],
            vec![],
        );
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
            func.set_symbol_name(ctx, "add_kernel".try_into().unwrap());
            func
        };

        // Entry block: three pointer arguments matching the function signature.
        let entry = BasicBlock::new(
            ctx,
            None,
            vec![in_ptr_ty.into(), in_ptr_ty.into(), out_ptr_ty.into()],
        );
        let func_region = func.get_operation().deref(ctx).get_region(0);
        entry.insert_at_front(func_region, ctx);

        let a = entry.deref(ctx).get_argument(0);
        let b = entry.deref(ctx).get_argument(1);
        let out = entry.deref(ctx).get_argument(2);

        // Helper to append a 0-region op and hand back its single result value.
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

        // i = ctaid.x * ntid.x + tid.x
        let tid = emit(
            ctx,
            ReadPtxSregTidXOp::get_concrete_op_info(),
            vec![i32_ty.into()],
            vec![],
        )
        .unwrap();
        let ctaid = emit(
            ctx,
            ReadPtxSregCtaidXOp::get_concrete_op_info(),
            vec![i32_ty.into()],
            vec![],
        )
        .unwrap();
        let ntid = emit(
            ctx,
            ReadPtxSregNtidXOp::get_concrete_op_info(),
            vec![i32_ty.into()],
            vec![],
        )
        .unwrap();
        let block_base = emit(
            ctx,
            MirMulOp::get_concrete_op_info(),
            vec![i32_ty.into()],
            vec![ctaid, ntid],
        )
        .unwrap();
        let i = emit(
            ctx,
            MirAddOp::get_concrete_op_info(),
            vec![i32_ty.into()],
            vec![block_base, tid],
        )
        .unwrap();

        // a_i = a[i]
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

        // b_i = b[i]
        let b_ptr = emit(
            ctx,
            MirPtrOffsetOp::get_concrete_op_info(),
            vec![in_ptr_ty.into()],
            vec![b, i],
        )
        .unwrap();
        let b_val = emit(
            ctx,
            MirLoadOp::get_concrete_op_info(),
            vec![f32_ty.into()],
            vec![b_ptr],
        )
        .unwrap();

        // sum = a_i + b_i
        let sum = emit(
            ctx,
            MirAddOp::get_concrete_op_info(),
            vec![f32_ty.into()],
            vec![a_val, b_val],
        )
        .unwrap();

        // out[i] = sum  (MirStoreOp operands are [dest_ptr, value], no result)
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
            vec![out_ptr, sum],
        );

        // return (void)
        emit(ctx, MirReturnOp::get_concrete_op_info(), vec![], vec![]);

        func.get_operation().insert_at_back(module_block, ctx);
    });
    module.mark_kernel_entry("add_kernel").unwrap();
}

#[test]
fn spine_add_kernel_emits_entry_and_validates() {
    let mut module = CodegenModule::new("spine_module").unwrap();
    build_add_kernel(&mut module);
    let compiler = Compiler::discover().expect("LLVM 21+ llc/opt are installed");
    let options = CompileOptions::new(Target::parse("sm_120").unwrap());
    let ptx = compiler
        .compile(&mut module, &options)
        .expect("compiles to PTX")
        .into_ptx();
    let text = String::from_utf8(ptx.clone()).expect("PTX is utf-8");

    assert!(
        text.contains(".visible .entry"),
        "kernel entry present:\n{text}"
    );
    assert!(
        text.contains(".target sm_120"),
        "PTX targets sm_120:\n{text}"
    );

    // ptxas must accept it for the real target.
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!(
        "spine_kernel_ptx_{}_{}",
        std::process::id(),
        unique
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let ptx_path = dir.join("spine.ptx");
    let cubin_path = dir.join("spine.cubin");
    std::fs::write(&ptx_path, &ptx).unwrap();

    let ptxas = if std::path::Path::new("/usr/local/cuda/bin/ptxas").exists() {
        "/usr/local/cuda/bin/ptxas"
    } else {
        "ptxas"
    };
    // Capture the Result before cleanup so the scratch dir is always removed,
    // even when ptxas is absent and `.output()` would otherwise panic.
    let ptxas_result = std::process::Command::new(ptxas)
        .arg("-arch=sm_120")
        .arg("--compile-only")
        .arg(&ptx_path)
        .arg("-o")
        .arg(&cubin_path)
        .output();

    // Cleanup before any assert or expect so the dir is reclaimed on all paths.
    let _ = std::fs::remove_dir_all(&dir);

    let out = ptxas_result.expect("ptxas runs");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert!(
        out.status.success(),
        "ptxas rejected PTX:\nstderr:\n{stderr}\n\nPTX:\n{text}"
    );
}
