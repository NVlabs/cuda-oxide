/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device extern declaration types for FFI with external LTOIR.

/// An external device function declaration (for linking with external LTOIR).
///
/// These declarations are emitted as LLVM `declare` statements and resolved
/// at link time by nvJitLink when linking with external LTOIR (e.g., CCCL).
#[derive(Debug, Clone)]
pub struct DeviceExternDecl {
    /// The export name (e.g., "cub_block_reduce_sum").
    pub export_name: String,

    /// Function parameter types (LLVM type strings like "float", "ptr", "i32").
    pub param_types: Vec<String>,

    /// Return type (LLVM type string like "float", "void", "i32").
    pub return_type: String,

    /// NVVM attributes for this function.
    pub attrs: DeviceExternAttrs,
}

/// NVVM attributes for device extern declarations.
///
/// NOTE: These attributes are currently **not emitted** to the LLVM IR output.
/// When linking LTOIR via nvJitLink, the external library's LTOIR already contains
/// proper attributes (convergent, nounwind, memory, etc.) on the function DEFINITIONS.
/// nvJitLink uses the definition's attributes during LTO, making attributes on
/// declarations redundant.
///
/// This struct is retained for potential future use or for debugging/inspection.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct DeviceExternAttrs {
    /// Function is convergent (all threads must execute together).
    pub is_convergent: bool,

    /// Function is pure (no side effects). Maps to LLVM `readnone`.
    pub is_pure: bool,

    /// Function is read-only (only reads memory). Maps to LLVM `readonly`.
    pub is_readonly: bool,
}

/// Trait for types that can be converted to [`DeviceExternDecl`].
///
/// This allows mir-importer to pass its own DeviceExternDecl type
/// without dialect-llvm depending on mir-importer.
pub trait AsDeviceExtern {
    fn as_device_extern(&self) -> DeviceExternDecl;
}

impl AsDeviceExtern for DeviceExternDecl {
    fn as_device_extern(&self) -> DeviceExternDecl {
        self.clone()
    }
}
