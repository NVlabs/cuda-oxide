/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Backend configuration traits and built-in implementations.

/// Minimal data layout for PTX mode (default behavior).
pub(super) const NVPTX_DATALAYOUT_PTX: &str = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64";

/// Full NVPTX data layout for libNVVM/LTOIR mode (Blackwell+, modern dialect).
///
/// This matches nvcc's output for sm_100+ and is required for full NVVM compatibility.
pub(super) const NVPTX_DATALAYOUT_FULL: &str = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-\
    i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-\
    v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8";

/// NVPTX data layout for the legacy typed-pointer NVVM dialect.
pub(super) const NVPTX_DATALAYOUT_LEGACY_NVVM: &str = "e-p:64:64:64-i1:8:8-i8:8:8-\
    i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-\
    v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64";

/// Configuration trait for export backends (PTX, LTOIR, etc.).
///
/// This trait allows different backends to customize IR generation without
/// exposing backend-specific details in the public API.
pub trait ExportBackendConfig {
    /// Data layout string for the target.
    fn datalayout(&self) -> &str;

    /// Whether to emit `@llvm.used` for kernel functions.
    /// This prevents the optimizer from removing "unused" kernels.
    fn emit_llvm_used(&self) -> bool;

    /// Whether to emit `!nvvmir.version` metadata.
    fn emit_nvvmir_version(&self) -> bool;

    /// The version tuple for `!nvvmir.version` metadata.
    /// Format: [major, minor, debug_major, debug_minor]
    fn nvvmir_version(&self) -> [i32; 4];

    /// Whether to emit `!nvvm.annotations` for ALL kernels.
    /// When false, only kernels with special attributes get annotations.
    fn emit_all_kernel_annotations(&self) -> bool;

    /// Whether kernel definitions should use the `ptx_kernel` calling convention.
    fn emit_ptx_kernel_keyword(&self) -> bool;

    /// NVVM IR dialect to use, when this is an NVVM IR backend.
    fn nvvm_ir_dialect(&self) -> Option<NvvmIrDialect> {
        None
    }
}

/// NVVM IR syntax family selected for libNVVM input.
///
/// NVIDIA's modern NVVM IR dialect uses opaque pointers and is valid for
/// Blackwell (`compute_100`) and newer targets. Pre-Blackwell libNVVM input
/// needs the older typed-pointer dialect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NvvmIrDialect {
    /// LLVM 7-style NVVM IR with typed pointers.
    TypedPointers,
    /// Modern NVVM IR with opaque pointers.
    OpaquePointers,
}

impl NvvmIrDialect {
    /// Select the NVVM IR dialect for a CUDA target such as `sm_86`,
    /// `sm_100a`, or `compute_120`.
    ///
    /// Unknown targets default to the modern dialect, preserving existing
    /// behavior for callers that do their own target validation.
    pub fn for_target(target: Option<&str>) -> Self {
        match target.and_then(cuda_arch_major) {
            Some(major) if major < 10 => Self::TypedPointers,
            _ => Self::OpaquePointers,
        }
    }
}

/// Default PTX export configuration.
///
/// Uses minimal settings appropriate for standard PTX generation via llc.
#[derive(Clone, Debug, Default)]
pub struct PtxExportConfig;

impl ExportBackendConfig for PtxExportConfig {
    fn datalayout(&self) -> &str {
        NVPTX_DATALAYOUT_PTX
    }

    fn emit_llvm_used(&self) -> bool {
        false
    }

    fn emit_nvvmir_version(&self) -> bool {
        false
    }

    fn nvvmir_version(&self) -> [i32; 4] {
        [0, 0, 0, 0] // Not used in PTX mode
    }

    fn emit_all_kernel_annotations(&self) -> bool {
        false
    }

    fn emit_ptx_kernel_keyword(&self) -> bool {
        true
    }
}

/// Export configuration for NVVM IR output.
///
/// Emits LLVM IR with full NVVM compatibility:
/// - Full NVPTX datalayout string
/// - `@llvm.used` to prevent kernel optimization
/// - `!nvvm.annotations` for all kernels
/// - `!nvvmir.version` metadata
///
/// This produces IR suitable for consumption by libNVVM (e.g., `nvvmCompileProgram -gen-lto`)
/// or other NVVM-compatible tools.
///
#[derive(Clone, Debug)]
pub struct NvvmExportConfig {
    dialect: NvvmIrDialect,
}

impl NvvmExportConfig {
    /// Create an NVVM export configuration for `target`.
    pub fn for_target(target: Option<&str>) -> Self {
        Self {
            dialect: NvvmIrDialect::for_target(target),
        }
    }

    /// Return the selected NVVM IR dialect.
    pub fn dialect(&self) -> NvvmIrDialect {
        self.dialect
    }
}

impl Default for NvvmExportConfig {
    fn default() -> Self {
        Self {
            dialect: NvvmIrDialect::OpaquePointers,
        }
    }
}

impl ExportBackendConfig for NvvmExportConfig {
    fn datalayout(&self) -> &str {
        match self.dialect {
            NvvmIrDialect::TypedPointers => NVPTX_DATALAYOUT_LEGACY_NVVM,
            NvvmIrDialect::OpaquePointers => NVPTX_DATALAYOUT_FULL,
        }
    }

    fn emit_llvm_used(&self) -> bool {
        true
    }

    fn emit_nvvmir_version(&self) -> bool {
        true
    }

    fn nvvmir_version(&self) -> [i32; 4] {
        match self.dialect {
            // CUDA 12.x pre-Blackwell libNVVM uses LLVM 7 IR with debug
            // metadata 3.1. Emitting the modern debug version makes
            // nvvmCompileProgram reject otherwise typed-pointer-compatible IR.
            NvvmIrDialect::TypedPointers => [2, 0, 3, 1],
            NvvmIrDialect::OpaquePointers => [2, 0, 3, 2],
        }
    }

    fn emit_all_kernel_annotations(&self) -> bool {
        true // Emit annotations for all kernels
    }

    fn emit_ptx_kernel_keyword(&self) -> bool {
        false
    }

    fn nvvm_ir_dialect(&self) -> Option<NvvmIrDialect> {
        Some(self.dialect)
    }
}

fn cuda_arch_major(target: &str) -> Option<u32> {
    let rest = target
        .strip_prefix("sm_")
        .or_else(|| target.strip_prefix("compute_"))?;
    let digits: String = rest.chars().take_while(|ch| ch.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let value = digits.parse::<u32>().ok()?;
    Some(value / 10)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvvm_ir_dialect_selects_typed_pointers_for_pre_blackwell() {
        assert_eq!(
            NvvmIrDialect::for_target(Some("sm_75")),
            NvvmIrDialect::TypedPointers
        );
        assert_eq!(
            NvvmIrDialect::for_target(Some("sm_86")),
            NvvmIrDialect::TypedPointers
        );
        assert_eq!(
            NvvmIrDialect::for_target(Some("compute_90")),
            NvvmIrDialect::TypedPointers
        );
    }

    #[test]
    fn nvvm_ir_dialect_selects_opaque_pointers_for_blackwell_and_unknown() {
        assert_eq!(
            NvvmIrDialect::for_target(Some("sm_100")),
            NvvmIrDialect::OpaquePointers
        );
        assert_eq!(
            NvvmIrDialect::for_target(Some("sm_100a")),
            NvvmIrDialect::OpaquePointers
        );
        assert_eq!(
            NvvmIrDialect::for_target(Some("compute_120")),
            NvvmIrDialect::OpaquePointers
        );
        assert_eq!(
            NvvmIrDialect::for_target(None),
            NvvmIrDialect::OpaquePointers
        );
        assert_eq!(
            NvvmIrDialect::for_target(Some("nvvm-ir")),
            NvvmIrDialect::OpaquePointers
        );
    }

    #[test]
    fn typed_nvvm_ir_uses_cuda_12_debug_metadata_version() {
        let config = NvvmExportConfig::for_target(Some("sm_86"));
        assert_eq!(config.nvvmir_version(), [2, 0, 3, 1]);
    }

    #[test]
    fn typed_nvvm_ir_uses_legacy_data_layout() {
        let config = NvvmExportConfig::for_target(Some("sm_86"));
        assert_eq!(config.datalayout(), NVPTX_DATALAYOUT_LEGACY_NVVM);
    }
}
