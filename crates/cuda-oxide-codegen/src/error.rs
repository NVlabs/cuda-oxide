/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Public error type for [`compile_to_ptx`](crate::compile_to_ptx).

/// Errors from the non-rustc PTX compilation entry point.
#[derive(Debug)]
#[non_exhaustive]
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

/// Errors from pipeline execution, categorized by stage.
#[derive(Debug)]
pub enum PipelineError {
    /// Function has no MIR body (shouldn't happen for collected functions).
    NoBody(String),
    /// MIR→Pliron IR translation failed.
    Translation(String),
    /// Pliron IR verification failed (includes failing operation if found).
    Verification {
        name: String,
        message: String,
        operation: Option<String>,
    },
    /// MIR→LLVM lowering failed.
    Lowering(String),
    /// LLVM IR export failed.
    Export(String),
    /// PTX generation via `llc` failed.
    PtxGeneration(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoBody(name) => write!(f, "Function '{}' has no MIR body", name),
            Self::Translation(msg) => write!(f, "Translation failed: {}", msg),
            Self::Verification {
                name,
                message,
                operation,
            } => {
                writeln!(f, "Verification failed for '{}':", name)?;
                writeln!(f, "  {}", message)?;
                if let Some(op) = operation {
                    writeln!(f, "  Failed operation:\n{}", op)?;
                }
                Ok(())
            }
            Self::Lowering(msg) => write!(f, "Lowering failed: {}", msg),
            Self::Export(msg) => write!(f, "Export failed: {}", msg),
            Self::PtxGeneration(msg) => write!(f, "PTX generation failed: {}", msg),
        }
    }
}

impl std::error::Error for PipelineError {}
