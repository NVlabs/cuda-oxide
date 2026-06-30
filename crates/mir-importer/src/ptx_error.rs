/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Public error type for [`compile_module_to_ptx`](crate::compile_module_to_ptx).

use crate::pipeline::PipelineError;

/// Errors from the non-rustc PTX compilation entry point.
#[derive(Debug)]
pub enum PtxError {
    /// The Pliron module failed verification before lowering.
    Verification(String),
    /// MIR to LLVM dialect lowering failed.
    Lowering(String),
    /// LLVM IR export (`.ll` rendering) failed.
    Export(String),
    /// `opt` or `llc` failed, or PTX could not be read back.
    Codegen(String),
    /// The requested target arch string was empty or malformed.
    InvalidConfig(String),
}

impl std::fmt::Display for PtxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PtxError::Verification(m) => write!(f, "module verification failed: {m}"),
            PtxError::Lowering(m) => write!(f, "MIR to LLVM lowering failed: {m}"),
            PtxError::Export(m) => write!(f, "LLVM IR export failed: {m}"),
            PtxError::Codegen(m) => write!(f, "PTX codegen failed: {m}"),
            PtxError::InvalidConfig(m) => write!(f, "invalid PtxConfig: {m}"),
        }
    }
}

impl std::error::Error for PtxError {}

impl From<PipelineError> for PtxError {
    fn from(e: PipelineError) -> Self {
        match e {
            PipelineError::Verification { message, .. } => PtxError::Verification(message),
            PipelineError::Lowering(m) => PtxError::Lowering(m),
            PipelineError::Export(m) => PtxError::Export(m),
            PipelineError::PtxGeneration(m) => PtxError::Codegen(m),
            other => PtxError::Codegen(format!("{other:?}")),
        }
    }
}
