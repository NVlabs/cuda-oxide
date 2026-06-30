/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integration test for the non-rustc `compile_module_to_ptx` entry point.
//!
//! Builds a trivial empty `MirFuncOp` module by hand (no rustc, no CubeCL,
//! mirroring the construction proven by `translator::body`'s
//! `inline_always_flag_reaches_llvm_func_attr_before_export` unit test) and
//! drives it through verify -> lower -> export -> opt -> llc to PTX.

// `mir-importer` links the compiler's `rustc_driver` dylib, so any crate that
// links it (including this external test binary) must enable the same feature
// and force the single dylib copy of the rustc deps, otherwise linking fails
// with "cannot satisfy dependencies so X only shows up once".
#![feature(rustc_private)]

extern crate rustc_driver;

use mir_importer::{PtxConfig, compile_module_to_ptx};

use dialect_mir::ops::{MirFuncOp, MirReturnOp};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::TypeAttr, op_interfaces::SymbolOpInterface, ops::ModuleOp, types::FunctionType,
    },
    context::Context,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
};

#[test]
fn empty_module_compiles_to_sm120_ptx() {
    let mut ctx = Context::new();
    mir_importer::translator::register_dialects(&mut ctx);

    let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
    let module_op = module.get_operation();
    let module_region = module_op.deref(&ctx).get_region(0);
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

    let func_type = FunctionType::get(&ctx, vec![], vec![]);
    let func_type_attr = TypeAttr::new(func_type.into());
    let func = {
        let op = Operation::new(
            &mut ctx,
            MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let func = MirFuncOp::new(&mut ctx, op, func_type_attr);
        func.set_symbol_name(&mut ctx, "empty".try_into().unwrap());
        func
    };

    // A `void` definition needs an entry block ending in `mir.return`;
    // an empty body would lower to invalid `define void @empty() { }`.
    let entry = BasicBlock::new(&mut ctx, None, vec![]);
    let func_region = func.get_operation().deref(&ctx).get_region(0);
    entry.insert_at_back(func_region, &ctx);
    let ret = Operation::new(
        &mut ctx,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
        vec![],
        0,
    );
    ret.insert_at_back(entry, &ctx);

    func.get_operation().insert_at_back(module_block, &ctx);

    let cfg = PtxConfig::new("sm_120");
    let ptx = compile_module_to_ptx(&mut ctx, module_op, &cfg).expect("compiles to PTX");

    let text = String::from_utf8(ptx).expect("PTX is utf-8");
    assert!(!text.is_empty(), "PTX is non-empty");
    assert!(
        text.contains(".target sm_120"),
        "PTX targets sm_120:\n{text}"
    );
}
