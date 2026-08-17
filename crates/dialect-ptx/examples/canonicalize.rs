/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use dialect_ptx::{Projection, emit_module};
use pliron::context::Context;
use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let input = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: canonicalize <input.ptx> [output.ptx]")?;
    let output = std::env::args_os().nth(2).map(PathBuf::from);
    let source = std::fs::read_to_string(&input)?;
    let mut ctx = Context::new();
    dialect_ptx::register(&mut ctx);
    let projection = Projection::parse(&mut ctx, &source)?;
    let emitted = emit_module(&ctx, &projection.module())?;
    let reparsed = ptx_parse::Document::parse(&emitted)?;
    if !reparsed.coverage().is_complete() {
        return Err(format!(
            "canonical PTX is structurally incomplete: {:?}",
            reparsed.coverage()
        )
        .into());
    }
    if let Some(output) = output {
        std::fs::write(output, emitted)?;
    } else {
        print!("{emitted}");
    }
    Ok(())
}
