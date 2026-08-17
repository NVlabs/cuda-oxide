/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end tour of the structured PTX dialect: build a kernel with a
//! counting loop, show the typed IR (including the typed guard predicate on
//! the loop terminator), emit canonical PTX, and prove the emitted text
//! re-parses losslessly with ptx-parse.

use dialect_ptx::attributes::PredicateAttr;
use dialect_ptx::ops::PtxInstructionOp;
use dialect_ptx::{PtxBuilder, emit_module};
use pliron::context::Context;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let mut ctx = Context::new();
    dialect_ptx::register(&mut ctx);

    let mut builder = PtxBuilder::new(&mut ctx);
    builder.version("8.9").target("sm_120a").address_size(64);
    let kernel = builder.visible_entry("demo_kernel", "()", |body| {
        body.directive(".reg", ".pred %p<2>;");
        body.directive(".reg", ".b32 %r<3>;");
        body.instruction("mov.u32", ["%r1", "0"]);
        body.label("$L_loop");
        body.instruction("add.u32", ["%r1", "%r1", "1"]);
        body.instruction("setp.lt.u32", ["%p1", "%r1", "16"]);
        body.predicated_instruction(PredicateAttr::new("%p1", false), "bra", ["$L_loop"]);
        body.instruction("ret", std::iter::empty::<&str>());
    });
    let module = builder.finish();

    println!("== typed IR ==");
    println!("{}", module.get_operation().disp(&ctx));

    let terminator = kernel
        .entry_block(&ctx)
        .expect("definition has a body")
        .deref(&ctx)
        .iter(&ctx)
        .filter(|operation| Operation::is_op::<PtxInstructionOp>(*operation, &ctx))
        .nth(3)
        .expect("loop back-edge instruction");
    println!("== loop terminator ==");
    println!("{}", terminator.disp(&ctx));

    let emitted = emit_module(&ctx, &module)?;
    println!("== canonical PTX ==");
    print!("{emitted}");

    let reparsed = ptx_parse::Document::parse(&emitted)?;
    if !reparsed.coverage().is_complete() {
        return Err(format!(
            "canonical PTX is structurally incomplete: {:?}",
            reparsed.coverage()
        )
        .into());
    }
    let back_edge = reparsed
        .instructions()
        .iter()
        .find(|instruction| instruction.head() == "bra")
        .expect("emitted kernel keeps its loop");
    let predicate = back_edge.predicate().expect("back-edge stays guarded");
    assert_eq!(predicate.register(), "%p1");
    assert!(!predicate.is_negated());
    println!("== round trip ==");
    println!(
        "back-edge guard survives: @{}{}",
        if predicate.is_negated() { "!" } else { "" },
        predicate.register()
    );
    Ok(())
}
