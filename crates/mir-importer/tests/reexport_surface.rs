/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-time proof that the pre-extraction mir-importer API still resolves.
#![feature(rustc_private)]
extern crate rustc_driver;

type CompileModuleToPtxFn = fn(
    &mut pliron::context::Context,
    pliron::context::Ptr<pliron::operation::Operation>,
    &mir_importer::PtxConfig,
) -> Result<Vec<u8>, mir_importer::PtxError>;

#[test]
fn reexports_resolve() {
    let _: CompileModuleToPtxFn = mir_importer::compile_to_ptx;
    let _ = mir_importer::PipelineError::Lowering(String::new());
}
