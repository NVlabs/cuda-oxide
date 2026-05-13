/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Export LLVM dialect to textual LLVM IR.
//!
//! Two pieces worth knowing about live in this module: the pre-pass that
//! assigns deterministic anonymous-value names so the textual IR is stable
//! across runs, and the block-argument → PHI-node translation that bridges
//! pliron's basic-block argument convention to LLVM's PHI-node convention.
//!
//! # Backend Configuration
//!
//! The export process can be customized via the [`ExportBackendConfig`] trait.
//! Different backends (PTX, etc.) can provide their own configuration for:
//!
//! - Data layout string
//! - Whether to emit `@llvm.used` for kernel preservation
//! - Whether to emit `!nvvmir.version` metadata
//! - Whether to emit `!nvvm.annotations` for all kernels
//!
//! The default [`PtxExportConfig`] uses minimal settings appropriate for standard
//! PTX generation via llc.

use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{FPDoubleAttr, FPSingleAttr, IntegerAttr},
        op_interfaces::{CallOpCallable, CallOpInterface, OneRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::{FP32Type, FP64Type, IntegerType},
    },
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    r#type::{TypeObj, Typed},
    value::Value,
};
use std::collections::HashMap;
use std::fmt::Write;

use crate::{
    attributes::{FPHalfAttr, GepIndexAttr, ICmpPredicateAttr},
    ops::{self, FuncOp, atomic::LlvmAtomicOpInterface},
    types::{FuncType, HalfType, PointerType, StructType, VoidType},
};
use pliron::builtin::type_interfaces::FunctionTypeInterface;

/// Minimal data layout for PTX mode (default behavior).
const NVPTX_DATALAYOUT_PTX: &str = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64";

// ============================================================================
// Device Extern Declaration Types (for FFI with external LTOIR)
// ============================================================================

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

/// Full NVPTX data layout for libNVVM/LTOIR mode (Blackwell+, LLVM 20 dialect).
///
/// This matches nvcc's output for sm_100+ and is required for full NVVM compatibility.
const NVPTX_DATALAYOUT_FULL: &str = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-\
    i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-\
    v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8";

// ============================================================================
// Export Backend Configuration Trait
// ============================================================================

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
/// Currently supports NVVM 20 dialect (Blackwell+, opaque pointers).
/// NVVM 7 dialect (pre-Blackwell, typed pointers) is not yet supported.
#[derive(Clone, Debug, Default)]
pub struct NvvmExportConfig;

impl ExportBackendConfig for NvvmExportConfig {
    fn datalayout(&self) -> &str {
        NVPTX_DATALAYOUT_FULL
    }

    fn emit_llvm_used(&self) -> bool {
        true
    }

    fn emit_nvvmir_version(&self) -> bool {
        true
    }

    fn nvvmir_version(&self) -> [i32; 4] {
        [2, 0, 3, 2] // NVVM IR 2.0, debug 3.2
    }

    fn emit_all_kernel_annotations(&self) -> bool {
        true // Emit annotations for all kernels
    }

    fn emit_ptx_kernel_keyword(&self) -> bool {
        false
    }
}

/// Export a module op to a String containing LLVM IR.
///
/// Uses default PTX export mode. For alternate backends, use [`export_module_to_string_with_config`].
pub fn export_module_to_string(ctx: &Context, module: &ModuleOp) -> Result<String, String> {
    export_module_to_string_with_config(ctx, module, &PtxExportConfig)
}

/// Export a module op with device extern declarations to a String containing LLVM IR.
///
/// This is the primary export function for Device FFI support. It emits:
/// 1. Header (datalayout, target triple)
/// 2. Device extern declarations (`declare` statements)
/// 3. Module functions (from pliron operations)
/// 4. Attribute groups
/// 5. Metadata (nvvm.annotations, etc.)
pub fn export_module_with_externs<T: AsDeviceExtern>(
    ctx: &Context,
    module: &ModuleOp,
    device_externs: &[T],
    config: &dyn ExportBackendConfig,
) -> Result<String, String> {
    // Convert device externs to our internal format
    let externs: Vec<DeviceExternDecl> = device_externs
        .iter()
        .map(|e| e.as_device_extern())
        .collect();

    export_module_with_externs_impl(ctx, module, &externs, config)
}

/// Trait for types that can be converted to DeviceExternDecl.
///
/// This allows mir-importer to pass its own DeviceExternDecl type
/// without dialect-llvm depending on mir-importer.
pub trait AsDeviceExtern {
    fn as_device_extern(&self) -> DeviceExternDecl;
}

// Self-implementation for DeviceExternDecl
impl AsDeviceExtern for DeviceExternDecl {
    fn as_device_extern(&self) -> DeviceExternDecl {
        self.clone()
    }
}

/// Internal implementation of export with device externs.
fn export_module_with_externs_impl(
    ctx: &Context,
    module: &ModuleOp,
    device_externs: &[DeviceExternDecl],
    config: &dyn ExportBackendConfig,
) -> Result<String, String> {
    let mut output = String::new();
    let emit_all_annotations = config.emit_all_kernel_annotations();
    let emit_ptx_kernel_keyword = config.emit_ptx_kernel_keyword();
    let mut state = ModuleExportState::new(ctx, emit_all_annotations, emit_ptx_kernel_keyword);
    state.build_source_key_map(module);

    // 1. Header
    writeln!(
        &mut output,
        "; ModuleID = '{}'",
        Operation::get_opid(module.get_operation(), ctx)
    )
    .unwrap();
    writeln!(
        &mut output,
        "source_filename = \"{}\"",
        module.get_symbol_name(ctx)
    )
    .unwrap();
    writeln!(
        &mut output,
        "target datalayout = \"{}\"",
        config.datalayout()
    )
    .unwrap();
    writeln!(&mut output, "target triple = \"nvptx64-nvidia-cuda\"").unwrap();
    writeln!(&mut output).unwrap();

    // 2. Device extern declarations (before function definitions)
    //
    // NOTE: We intentionally do NOT emit LLVM attributes on these declarations.
    // The external LTOIR (from nvcc -dc -dlto) already contains proper attributes
    // (convergent, nounwind, memory, etc.) on the function DEFINITIONS.
    // When nvJitLink performs LTO linking, it uses the definition's attributes.
    // Attributes on declarations are redundant and were causing issues where
    // all externs incorrectly got the same attribute group.
    if !device_externs.is_empty() {
        writeln!(
            &mut output,
            "; External device function declarations (resolved by nvJitLink)"
        )
        .unwrap();
        for decl in device_externs {
            let params = decl.param_types.join(", ");
            writeln!(
                &mut output,
                "declare {} @{}({})",
                decl.return_type, decl.export_name, params
            )
            .unwrap();
        }
        writeln!(&mut output).unwrap();
    }

    // 3. Process Globals and Functions (including intrinsic declarations)
    // Skip device extern declarations - they were already emitted in section 2 with proper attributes
    let device_extern_names: std::collections::HashSet<&str> = device_externs
        .iter()
        .map(|d| d.export_name.as_str())
        .collect();

    let region = module.get_region(ctx).deref(ctx);
    if let Some(block) = region.iter(ctx).next() {
        let mut last_was_decl = false;
        for op in block.deref(ctx).iter(ctx) {
            if let Some(func) = Operation::get_op::<FuncOp>(op, ctx) {
                let is_decl = func.get_operation().deref(ctx).regions().count() == 0;
                let func_name = func.get_symbol_name(ctx);

                // Skip device extern declarations - already emitted in section 2
                if is_decl && device_extern_names.contains(func_name.as_str()) {
                    continue;
                }

                if !is_decl && last_was_decl {
                    writeln!(&mut output).unwrap();
                }

                state.export_function(&func, &mut output)?;
                last_was_decl = is_decl;
            } else if let Some(global) = Operation::get_op::<ops::GlobalOp>(op, ctx) {
                state.export_global(&global, &mut output)?;
                last_was_decl = false;
            } else {
                writeln!(
                    &mut output,
                    "; Unsupported top-level op: {}",
                    Operation::get_opid(op, ctx)
                )
                .unwrap();
                last_was_decl = false;
            }
        }
    }

    // 4. @llvm.used — preserve kernels and/or standalone device functions from DCE
    //
    // Kernels have no callers in the device module (invoked from host), and standalone
    // device functions have no callers when compiled without a kernel (consumed by
    // external C++ via LTOIR). Both need @llvm.used to survive optimization.
    if config.emit_llvm_used() {
        let mut used_refs: Vec<String> = Vec::new();

        // Include all kernels
        for k in &state.all_kernels {
            used_refs.push(format!("ptr @{}", k.name));
        }

        // Include standalone device functions (when no kernels present)
        if state.all_kernels.is_empty() {
            for name in &state.device_functions {
                used_refs.push(format!("ptr @{}", name));
            }
        }

        if !used_refs.is_empty() {
            writeln!(&mut output).unwrap();
            writeln!(
                &mut output,
                "@llvm.used = appending global [{} x ptr] [{}], section \"llvm.metadata\"",
                used_refs.len(),
                used_refs.join(", ")
            )
            .unwrap();
        }
    }

    // 5. Emit attribute groups for convergent intrinsics used by module functions
    // Note: Device extern declarations no longer get attribute groups - see section 2 comment.
    if state.convergent_used {
        writeln!(&mut output).unwrap();
        writeln!(&mut output, "attributes #0 = {{ convergent }}").unwrap();
    }

    // 6. nvvm.annotations metadata (same as original)
    let has_special_kernels =
        !state.cluster_kernels.is_empty() || !state.launch_bounds_kernels.is_empty();
    let needs_annotations =
        has_special_kernels || (emit_all_annotations && !state.all_kernels.is_empty());

    if needs_annotations {
        writeln!(&mut output).unwrap();
        emit_nvvm_annotations(&mut output, &state, emit_all_annotations);
    }

    // 7. nvvmir.version metadata (if backend requires)
    // Must use a unique metadata ID that doesn't conflict with nvvm.annotations
    if config.emit_nvvmir_version() {
        writeln!(&mut output).unwrap();
        let ver = config.nvvmir_version();
        let md_id = md_id_after_annotations(&state);
        writeln!(
            &mut output,
            "!nvvmir.version = !{{!{}}}\n!{} = !{{i32 {}, i32 {}, i32 {}, i32 {}}}",
            md_id, md_id, ver[0], ver[1], ver[2], ver[3]
        )
        .unwrap();
    }

    Ok(output)
}

/// Helper to emit nvvm.annotations metadata.
fn emit_nvvm_annotations(
    output: &mut String,
    state: &ModuleExportState,
    emit_all_annotations: bool,
) {
    let mut metadata_refs = Vec::new();
    let mut md_id = 0;

    // Collect names of kernels that have special configs
    let special_kernel_names: std::collections::HashSet<&str> = state
        .cluster_kernels
        .iter()
        .map(|k| k.name.as_str())
        .chain(state.launch_bounds_kernels.iter().map(|k| k.name.as_str()))
        .collect();

    // Emit basic annotation for kernels WITHOUT special configs
    if emit_all_annotations {
        for kernel in state.all_kernels.iter() {
            if !special_kernel_names.contains(kernel.name.as_str()) {
                writeln!(
                    output,
                    "!{} = !{{ptr @{}, !\"kernel\", i32 1}}",
                    md_id, kernel.name
                )
                .unwrap();
                metadata_refs.push(format!("!{}", md_id));
                md_id += 1;
            }
        }
    }

    // Emit cluster config annotations
    for cfg in state.cluster_kernels.iter() {
        writeln!(
            output,
            "!{} = !{{ptr @{}, !\"kernel\", i32 1, !\"cluster_dim_x\", i32 {}, !\"cluster_dim_y\", i32 {}, !\"cluster_dim_z\", i32 {}}}",
            md_id, cfg.name, cfg.dim_x, cfg.dim_y, cfg.dim_z
        )
        .unwrap();
        metadata_refs.push(format!("!{}", md_id));
        md_id += 1;
    }

    // Emit launch bounds annotations
    for bounds in state.launch_bounds_kernels.iter() {
        if let Some(min_blocks) = bounds.min_blocks {
            writeln!(
                output,
                "!{} = !{{ptr @{}, !\"kernel\", i32 1, !\"maxntidx\", i32 {}, !\"minctasm\", i32 {}}}",
                md_id, bounds.name, bounds.max_threads, min_blocks
            )
            .unwrap();
        } else {
            writeln!(
                output,
                "!{} = !{{ptr @{}, !\"kernel\", i32 1, !\"maxntidx\", i32 {}}}",
                md_id, bounds.name, bounds.max_threads
            )
            .unwrap();
        }
        metadata_refs.push(format!("!{}", md_id));
        md_id += 1;
    }

    // Emit named metadata referencing all annotation nodes
    if !metadata_refs.is_empty() {
        writeln!(
            output,
            "!nvvm.annotations = !{{{}}}",
            metadata_refs.join(", ")
        )
        .unwrap();
    }
}

/// Export a module op to a String containing LLVM IR with custom backend configuration.
///
/// The `config` parameter controls backend-specific IR generation options like
/// data layout, metadata emission, and symbol preservation.
pub fn export_module_to_string_with_config(
    ctx: &Context,
    module: &ModuleOp,
    config: &dyn ExportBackendConfig,
) -> Result<String, String> {
    let mut output = String::new();
    let emit_all_annotations = config.emit_all_kernel_annotations();
    let emit_ptx_kernel_keyword = config.emit_ptx_kernel_keyword();
    let mut state = ModuleExportState::new(ctx, emit_all_annotations, emit_ptx_kernel_keyword);
    state.build_source_key_map(module);

    // 1. Header
    writeln!(
        &mut output,
        "; ModuleID = '{}'",
        Operation::get_opid(module.get_operation(), ctx)
    )
    .unwrap();
    writeln!(
        &mut output,
        "source_filename = \"{}\"",
        module.get_symbol_name(ctx)
    )
    .unwrap();

    // Use backend-specific data layout
    writeln!(
        &mut output,
        "target datalayout = \"{}\"",
        config.datalayout()
    )
    .unwrap();
    writeln!(&mut output, "target triple = \"nvptx64-nvidia-cuda\"").unwrap();
    writeln!(&mut output).unwrap(); // Separate header from body

    // 2. Process Globals and Functions (including intrinsic declarations)
    let region = module.get_region(ctx).deref(ctx);
    if let Some(block) = region.iter(ctx).next() {
        let mut last_was_decl = false;
        for op in block.deref(ctx).iter(ctx) {
            if let Some(func) = Operation::get_op::<FuncOp>(op, ctx) {
                let is_decl = func.get_operation().deref(ctx).regions().count() == 0;

                // If we are transitioning from a declaration to a definition (or anything else)
                // insert a newline to separate the declaration block from the definitions.
                if !is_decl && last_was_decl {
                    writeln!(&mut output).unwrap();
                }

                state.export_function(&func, &mut output)?;
                last_was_decl = is_decl;
            } else if let Some(global) = Operation::get_op::<ops::GlobalOp>(op, ctx) {
                // Export global variable (typically shared memory)
                state.export_global(&global, &mut output)?;
                last_was_decl = false;
            } else {
                writeln!(
                    &mut output,
                    "; Unsupported top-level op: {}",
                    Operation::get_opid(op, ctx)
                )
                .unwrap();
                last_was_decl = false;
            }
        }
    }

    // Emit @llvm.used if backend requests it (prevents symbols from being optimized away).
    //
    // WHY THIS IS NEEDED:
    // Kernels have no callers within the device module - they're invoked by host code.
    // Standalone device functions have no callers when compiled without a kernel - they're
    // consumed by external C++ via LTOIR linking.
    // Without explicit marking, LLVM's optimizer sees them as "dead code" and removes them.
    // The @llvm.used global tells LLVM: "preserve these symbols, they're used externally."
    if config.emit_llvm_used() {
        let mut used_refs: Vec<String> = Vec::new();

        for k in &state.all_kernels {
            used_refs.push(format!("ptr @{}", k.name));
        }

        // Include standalone device functions when no kernels are present
        if state.all_kernels.is_empty() {
            for name in &state.device_functions {
                used_refs.push(format!("ptr @{}", name));
            }
        }

        if !used_refs.is_empty() {
            writeln!(&mut output).unwrap();
            writeln!(
                &mut output,
                "@llvm.used = appending global [{} x ptr] [{}], section \"llvm.metadata\"",
                used_refs.len(),
                used_refs.join(", ")
            )
            .unwrap();
        }
    }

    // Emit attributes section if convergent operations were used
    if state.convergent_used {
        writeln!(&mut output).unwrap();
        writeln!(&mut output, "attributes #0 = {{ convergent }}").unwrap();
    }

    // Emit nvvm.annotations metadata
    // - Default: Only for kernels with cluster configuration or launch bounds
    // - Alternate backends: May require annotations for ALL kernels
    let has_special_kernels =
        !state.cluster_kernels.is_empty() || !state.launch_bounds_kernels.is_empty();
    let needs_annotations =
        has_special_kernels || (emit_all_annotations && !state.all_kernels.is_empty());

    if needs_annotations {
        writeln!(&mut output).unwrap();

        let mut metadata_refs = Vec::new();
        let mut md_id = 0;

        // If backend requires annotations for all kernels, emit basic annotations first
        // (unless they have cluster/launch_bounds which will be emitted below with more detail)
        if emit_all_annotations {
            // Collect names of kernels that have special configs (they'll get detailed annotations)
            let special_kernel_names: std::collections::HashSet<&str> = state
                .cluster_kernels
                .iter()
                .map(|k| k.name.as_str())
                .chain(state.launch_bounds_kernels.iter().map(|k| k.name.as_str()))
                .collect();

            // Emit basic annotation for kernels WITHOUT special configs
            for kernel in state.all_kernels.iter() {
                if !special_kernel_names.contains(kernel.name.as_str()) {
                    // Basic kernel annotation: !{ptr @kernel_name, !"kernel", i32 1}
                    writeln!(
                        &mut output,
                        "!{} = !{{ptr @{}, !\"kernel\", i32 1}}",
                        md_id, kernel.name
                    )
                    .unwrap();
                    metadata_refs.push(format!("!{}", md_id));
                    md_id += 1;
                }
            }
        }

        // Each kernel with cluster config gets its own metadata node
        // Format: !{ptr @kernel_name, !"kernel", i32 1, !"cluster_dim_x", i32 X, ...}
        for cfg in state.cluster_kernels.iter() {
            writeln!(
                &mut output,
                "!{} = !{{ptr @{}, !\"kernel\", i32 1, !\"cluster_dim_x\", i32 {}, !\"cluster_dim_y\", i32 {}, !\"cluster_dim_z\", i32 {}}}",
                md_id, cfg.name, cfg.dim_x, cfg.dim_y, cfg.dim_z
            )
            .unwrap();
            metadata_refs.push(format!("!{}", md_id));
            md_id += 1;
        }

        // Each kernel with launch bounds gets its own metadata node
        // LLVM NVPTX expects separate annotations: !"maxntidx", !"maxntidy", !"maxntidz", !"minctapersm"
        // See: https://llvm.org/docs/NVPTXUsage.html
        for cfg in state.launch_bounds_kernels.iter() {
            // Emit maxntidx (we use the single max_threads value for 1D block size)
            writeln!(
                &mut output,
                "!{} = !{{ptr @{}, !\"maxntidx\", i32 {}}}",
                md_id, cfg.name, cfg.max_threads
            )
            .unwrap();
            metadata_refs.push(format!("!{}", md_id));
            md_id += 1;

            // Emit maxntidy = 1 (for complete 3D specification)
            writeln!(
                &mut output,
                "!{} = !{{ptr @{}, !\"maxntidy\", i32 1}}",
                md_id, cfg.name
            )
            .unwrap();
            metadata_refs.push(format!("!{}", md_id));
            md_id += 1;

            // Emit maxntidz = 1 (for complete 3D specification)
            writeln!(
                &mut output,
                "!{} = !{{ptr @{}, !\"maxntidz\", i32 1}}",
                md_id, cfg.name
            )
            .unwrap();
            metadata_refs.push(format!("!{}", md_id));
            md_id += 1;

            // Emit minctasm as separate metadata node if specified (generates .minnctapersm in PTX)
            if let Some(min_blocks) = cfg.min_blocks {
                writeln!(
                    &mut output,
                    "!{} = !{{ptr @{}, !\"minctasm\", i32 {}}}",
                    md_id, cfg.name, min_blocks
                )
                .unwrap();
                metadata_refs.push(format!("!{}", md_id));
                md_id += 1;
            }
        }

        // The nvvm.annotations named metadata references all kernel metadata
        writeln!(
            &mut output,
            "!nvvm.annotations = !{{{}}}",
            metadata_refs.join(", ")
        )
        .unwrap();
    }

    // Emit !nvvmir.version metadata if backend requests it
    if config.emit_nvvmir_version() {
        writeln!(&mut output).unwrap();
        let version = config.nvvmir_version();
        writeln!(
            &mut output,
            "!nvvmir.version = !{{!{}}}",
            md_id_after_annotations(&state)
        )
        .unwrap();
        writeln!(
            &mut output,
            "!{} = !{{i32 {}, i32 {}, i32 {}, i32 {}}}",
            md_id_after_annotations(&state),
            version[0],
            version[1],
            version[2],
            version[3]
        )
        .unwrap();
    }

    Ok(output)
}

/// Calculate the next metadata ID after annotations (for !nvvmir.version).
fn md_id_after_annotations(state: &ModuleExportState) -> usize {
    let mut count = state.all_kernels.len();

    // Subtract kernels that have special configs (they're not double-counted)
    let special_kernel_names: std::collections::HashSet<&str> = state
        .cluster_kernels
        .iter()
        .map(|k| k.name.as_str())
        .chain(state.launch_bounds_kernels.iter().map(|k| k.name.as_str()))
        .collect();

    for kernel in &state.all_kernels {
        if special_kernel_names.contains(kernel.name.as_str()) {
            count -= 1;
        }
    }

    // Add cluster kernels
    count += state.cluster_kernels.len();

    // Add launch bounds kernels (each has multiple metadata entries)
    for cfg in &state.launch_bounds_kernels {
        count += 3; // maxntidx, maxntidy, maxntidz
        if cfg.min_blocks.is_some() {
            count += 1; // minctasm
        }
    }

    count
}

/// Map from block to its predecessors, with the values passed to each predecessor.
/// Used for PHI node generation when exporting to LLVM IR.
type PredecessorMap = HashMap<Ptr<BasicBlock>, Vec<(Ptr<BasicBlock>, Vec<Value>)>>;

/// Cluster dimensions for a kernel (from `#[cluster(x,y,z)]` attribute).
struct KernelClusterConfig {
    name: String,
    dim_x: u32,
    dim_y: u32,
    dim_z: u32,
}

/// Launch bounds for a kernel (from `#[launch_bounds(max, min)]` attribute).
struct KernelLaunchBounds {
    name: String,
    max_threads: u32,
    min_blocks: Option<u32>, // None if not specified (0 in attribute)
}

/// Basic kernel info (for backends that need annotations for all kernels).
struct KernelInfo {
    name: String,
}

// Device-symbol detection and base-name extraction route through
// `reserved-oxide-symbols`, the workspace-internal source of truth for the
// `cuda_oxide_*` namespace.
//
// Note on FQDN forms: MIR import converts `::` to `__`, so a fully-qualified
// device symbol can appear as `mycrate__cuda_oxide_device_<hash>_foo`. Because
// the helpers in `reserved-oxide-symbols` use substring matching (not
// `starts_with`), they handle both bare and FQDN forms uniformly — no separate
// `FQDN_DEVICE_PREFIX` constant is needed.
use reserved_oxide_symbols::{device_base_name, is_device_extern_symbol, is_device_symbol};

/// Returns true if `name` is a device function (definition, not extern).
fn has_device_prefix(name: &str) -> bool {
    is_device_symbol(name)
}

/// Strip the device-function prefix from `name` if present.
///
/// The reserved prefix is needed internally for MIR-level detection but
/// should not leak into the final LLVM IR / PTX / LTOIR output. Returns
/// `name` unchanged for non-device symbols and for device-extern declarations
/// (those keep their original-name `link_name` attribute).
fn strip_device_prefix(name: &str) -> String {
    if is_device_extern_symbol(name) {
        return name.to_string();
    }
    device_base_name(name)
        .map(str::to_string)
        .unwrap_or_else(|| name.to_string())
}

struct ModuleExportState<'a> {
    ctx: &'a Context,
    /// Track if any convergent operations were used (for emitting attributes section)
    convergent_used: bool,
    /// Track kernels with cluster configurations for nvvm.annotations metadata
    cluster_kernels: Vec<KernelClusterConfig>,
    /// Track kernels with launch bounds for nvvm.annotations metadata
    launch_bounds_kernels: Vec<KernelLaunchBounds>,
    /// Track ALL kernels (for backends that require annotations for every kernel)
    all_kernels: Vec<KernelInfo>,
    /// Whether to track all kernels (set by backend config)
    track_all_kernels: bool,
    /// Whether to print `ptx_kernel` on kernel definitions.
    emit_ptx_kernel_keyword: bool,
    /// Track device function names for @llvm.used (standalone device fn compilation)
    device_functions: Vec<String>,
    /// Map from source-level static key (Rust mangled name) to the LLVM
    /// symbol name we emit for that global. Populated by a pre-pass over
    /// the module before any function or global is exported, so that
    /// initializer relocations referencing other statics by source key
    /// can resolve to the synthetic `__device_global_N` LLVM name.
    source_key_to_llvm_name: HashMap<String, String>,
}

impl<'a> ModuleExportState<'a> {
    fn new(ctx: &'a Context, track_all_kernels: bool, emit_ptx_kernel_keyword: bool) -> Self {
        Self {
            ctx,
            convergent_used: false,
            cluster_kernels: Vec::new(),
            launch_bounds_kernels: Vec::new(),
            all_kernels: Vec::new(),
            track_all_kernels,
            emit_ptx_kernel_keyword,
            device_functions: Vec::new(),
            source_key_to_llvm_name: HashMap::new(),
        }
    }

    /// Walk all GlobalOps in the module and populate
    /// `source_key_to_llvm_name`. Must run before any global is exported
    /// so that initializer relocations can resolve target symbols.
    fn build_source_key_map(&mut self, module: &ModuleOp) {
        use pliron::builtin::attributes::StringAttr;

        let region = module.get_region(self.ctx).deref(self.ctx);
        let Some(block) = region.iter(self.ctx).next() else {
            return;
        };
        for op in block.deref(self.ctx).iter(self.ctx) {
            let Some(global) = Operation::get_op::<ops::GlobalOp>(op, self.ctx) else {
                continue;
            };
            let attrs = &global.get_operation().deref(self.ctx).attributes.0;
            let key = attrs
                .get(&"llvm_global_source_key".try_into().unwrap())
                .and_then(|a| a.downcast_ref::<StringAttr>())
                .map(|a| String::from((*a).clone()));
            if let Some(key) = key {
                let name = global.get_symbol_name(self.ctx).to_string();
                self.source_key_to_llvm_name.insert(key, name);
            }
        }
    }

    /// Check if a function name is a known convergent intrinsic.
    ///
    /// These intrinsics require warp-synchronous execution semantics and must
    /// be marked convergent to prevent LLVM from applying optimizations that
    /// would break GPU synchronization (like duplicating them into divergent branches).
    fn is_convergent_intrinsic(name: &str) -> bool {
        // Block-level barriers
        name == "llvm.nvvm.barrier0"
            || name.starts_with("llvm.nvvm.barrier")
            // mbarrier operations
            || name.starts_with("llvm.nvvm.mbarrier")
            // Warp shuffles (though LLVM usually handles these)
            || name.starts_with("llvm.nvvm.shfl")
            // Warp votes
            || name.starts_with("llvm.nvvm.vote")
            // Async bulk operations (TMA)
            || name.starts_with("llvm.nvvm.cp.async.bulk")
    }

    /// Export a global variable (typically shared memory for GPU kernels)
    fn export_global(&mut self, global: &ops::GlobalOp, output: &mut String) -> Result<(), String> {
        use crate::attributes::LinkageAttr;
        use pliron::builtin::attributes::StringAttr;
        use pliron::r#type::Typed;

        let name = global.get_symbol_name(self.ctx);
        let ty = global.get_type(self.ctx);
        let address_space = global.get_address_space(self.ctx);

        // Check for external linkage (dynamic shared memory)
        let is_external = global
            .get_attr_llvm_global_linkage(self.ctx)
            .map(|linkage| matches!(*linkage, LinkageAttr::ExternalLinkage))
            .unwrap_or(false);

        // Get alignment from attribute, or compute natural alignment from type
        let alignment = global.get_alignment(self.ctx).unwrap_or_else(|| {
            // Compute natural alignment from array element type
            // For [N x T], alignment is size_of(T) (common case: f32 = 4, i64 = 8)
            let ty_ref = ty.deref(self.ctx);
            if let Some(array_ty) = ty_ref.downcast_ref::<crate::types::ArrayType>() {
                let elem_ty = array_ty.elem_type();
                let elem_ref = elem_ty.deref(self.ctx);
                if elem_ref.is::<pliron::builtin::types::IntegerType>() {
                    let int_ty = elem_ref
                        .downcast_ref::<pliron::builtin::types::IntegerType>()
                        .unwrap();
                    u64::from(int_ty.width() / 8)
                } else if elem_ref.is::<pliron::builtin::types::FP32Type>() {
                    4
                } else {
                    8 // Default alignment (FP64Type and unknown types)
                }
            } else {
                8 // Default alignment
            }
        });

        if is_external {
            // External linkage: declaration with size determined elsewhere.
            write!(
                output,
                "@{name} = external addrspace({address_space}) global "
            )
            .unwrap();
            self.export_type(ty, output)?;
            writeln!(output, ", align {alignment}").unwrap();
            return Ok(());
        }

        // Read initializer attrs (set by mir-lower on cross-static refs).
        let attrs = &global.get_operation().deref(self.ctx).attributes.0;
        let init_bytes_hex = attrs
            .get(&"llvm_initializer_bytes".try_into().unwrap())
            .and_then(|a| a.downcast_ref::<StringAttr>())
            .map(|a| String::from((*a).clone()))
            .unwrap_or_default();
        let init_relocs = attrs
            .get(&"llvm_initializer_relocations".try_into().unwrap())
            .and_then(|a| a.downcast_ref::<StringAttr>())
            .map(|a| String::from((*a).clone()))
            .unwrap_or_default();

        if init_bytes_hex.is_empty() && init_relocs.is_empty() {
            // No initializer info — fall back to historical behaviour.
            write!(output, "@{name} = addrspace({address_space}) global ").unwrap();
            self.export_type(ty, output)?;
            writeln!(output, " zeroinitializer, align {alignment}").unwrap();
            return Ok(());
        }

        // We have a real initializer. Decode bytes + relocations.
        let bytes = decode_hex_bytes(&init_bytes_hex).map_err(|e| {
            format!("Failed to decode initializer bytes for @{name}: {e}")
        })?;
        let mut relocations = decode_relocations(&init_relocs)
            .map_err(|e| format!("Failed to decode relocations for @{name}: {e}"))?;
        // Sort by offset so the packed-struct walk is monotonic.
        relocations.sort_by_key(|(off, _)| *off);

        // Resolve relocation target names through the same dedup table the
        // exporter uses for direct global references. We don't have direct
        // access here, but since mir-lower keyed `device_globals` by the
        // static's source name and the resulting `__device_global_N`
        // symbols are what live in this module, the relocation target
        // strings in `init_relocs` are the mir-importer-side keys.
        //
        // We'll resolve them by name at the IR level using a `@<key>`
        // reference — that requires the GlobalOp for each target to have
        // been emitted with that exact name. mir-lower assigns synthetic
        // `__device_global_N` names, so to make the linkage work we need
        // a key→name map at this layer. The simplest path is to look up
        // each target by traversing the module's globals and matching on
        // a sidecar attribute. We sidestep this by emitting the original
        // global_key as the GlobalOp symbol name when it differs from a
        // synthetic counter — see the mir-lower change that names globals
        // after their key when an initializer is present.

        if relocations.is_empty() {
            // Pure-byte initializer (e.g. `static INNER: [u64; 4] = [...]`).
            // Override the type to `[N x i8]` to match the byte payload —
            // LLVM accepts the underlying global as opaque storage either
            // way, and this lets us avoid reconstructing the typed initializer.
            let n = bytes.len();
            write!(
                output,
                "@{name} = addrspace({address_space}) global [{n} x i8] c\""
            )
            .unwrap();
            for b in &bytes {
                write!(output, "\\{:02X}", b).unwrap();
            }
            writeln!(output, "\", align {alignment}").unwrap();
        } else {
            // Mixed bytes + ptr relocations: emit a packed struct that
            // interleaves byte runs with the relocated pointer slots.
            // Each relocation occupies 8 bytes (64-bit pointer width).
            let mut cursor: u64 = 0;
            let mut type_parts: Vec<String> = Vec::new();
            let mut value_parts: Vec<String> = Vec::new();
            for (off, target_key) in &relocations {
                if *off > cursor {
                    let run_len = (*off - cursor) as usize;
                    type_parts.push(format!("[{run_len} x i8]"));
                    let mut s = format!("[{run_len} x i8] c\"");
                    for b in &bytes[cursor as usize..*off as usize] {
                        s.push_str(&format!("\\{:02X}", b));
                    }
                    s.push('"');
                    value_parts.push(s);
                }
                type_parts.push("ptr".to_string());
                // Resolve the source-level static key to the LLVM symbol
                // name through the pre-built map. If the target wasn't
                // emitted in this module (which shouldn't happen, since
                // mir-importer emits the full reachable closure), fall
                // back to a `null` pointer rather than a dangling ref.
                //
                // The target lives in addrspace(1) (CUDA global memory).
                // The struct field is a generic `ptr` (addrspace 0), since
                // that's how Rust references lower in our pipeline. We
                // bridge with a constant `addrspacecast` so the value
                // typechecks under opaque-pointer LLVM, and the runtime
                // address still resolves correctly because addrspace 0
                // (generic) covers addrspace 1 (global) for loads.
                let llvm_name = self
                    .source_key_to_llvm_name
                    .get(target_key)
                    .cloned()
                    .unwrap_or_default();
                if llvm_name.is_empty() {
                    value_parts.push("ptr null".to_string());
                } else {
                    value_parts.push(format!(
                        "ptr addrspacecast (ptr addrspace(1) @{llvm_name} to ptr)"
                    ));
                }
                cursor = *off + 8;
            }
            if (cursor as usize) < bytes.len() {
                let run_len = bytes.len() - cursor as usize;
                type_parts.push(format!("[{run_len} x i8]"));
                let mut s = format!("[{run_len} x i8] c\"");
                for b in &bytes[cursor as usize..] {
                    s.push_str(&format!("\\{:02X}", b));
                }
                s.push('"');
                value_parts.push(s);
            }
            let struct_ty = format!("<{{ {} }}>", type_parts.join(", "));
            let struct_val = format!("<{{ {} }}>", value_parts.join(", "));
            writeln!(
                output,
                "@{name} = addrspace({address_space}) global {struct_ty} {struct_val}, align {alignment}"
            )
            .unwrap();
        }

        Ok(())
    }

    fn export_function(&mut self, func: &FuncOp, output: &mut String) -> Result<(), String> {
        let func_name = func.get_symbol_name(self.ctx);
        // LLVM intrinsics (NVVM and standard, e.g. llvm.fptosi.sat) use dots in IR
        // but Pliron IR identifiers use underscores; convert for export.
        let fixed_func_name = if func_name.starts_with("llvm_") {
            func_name.replace('_', ".")
        } else {
            // Strip cuda_oxide_device_ prefix for clean export names.
            // Internal MIR translation uses prefixed names; we strip at the final
            // export layer so definitions and call targets are renamed consistently.
            strip_device_prefix(&func_name)
        };

        // Check for kernel attribute
        let kernel_key: pliron::identifier::Identifier = "gpu_kernel".try_into().unwrap();
        let attrs = &func.get_operation().deref(self.ctx).attributes.0;
        let is_kernel = attrs.contains_key(&kernel_key);

        // Track ALL kernels if backend requires annotations for every kernel
        if is_kernel && self.track_all_kernels {
            self.all_kernels.push(KernelInfo {
                name: fixed_func_name.clone(),
            });
        }

        // Track device function definitions (not declarations) for @llvm.used preservation
        // in standalone device function compilation. Extern declarations are excluded
        // because they're resolved at link time — only definitions need DCE protection.
        if !is_kernel && has_device_prefix(&func_name) {
            self.device_functions.push(fixed_func_name.clone());
        }

        // Check for cluster dimension attributes (from #[cluster(x,y,z)])
        // These will be emitted as nvvm.annotations metadata
        if is_kernel {
            let x_key: pliron::identifier::Identifier = "cluster_dim_x".try_into().unwrap();
            let y_key: pliron::identifier::Identifier = "cluster_dim_y".try_into().unwrap();
            let z_key: pliron::identifier::Identifier = "cluster_dim_z".try_into().unwrap();

            if let (Some(x_attr), Some(y_attr), Some(z_attr)) =
                (attrs.get(&x_key), attrs.get(&y_key), attrs.get(&z_key))
            {
                // Extract integer values from attributes
                use pliron::attribute::AttrObj;
                let get_int = |attr: &AttrObj| -> Option<u32> {
                    attr.downcast_ref::<pliron::builtin::attributes::IntegerAttr>()
                        .map(|int_attr| int_attr.value().to_u32())
                };

                if let (Some(dim_x), Some(dim_y), Some(dim_z)) =
                    (get_int(x_attr), get_int(y_attr), get_int(z_attr))
                {
                    self.cluster_kernels.push(KernelClusterConfig {
                        name: fixed_func_name.clone(),
                        dim_x,
                        dim_y,
                        dim_z,
                    });
                }
            }

            // Check for launch bounds attributes (from #[launch_bounds(max, min)])
            // These will be emitted as nvvm.annotations metadata for maxntid and minctasm
            let maxntid_key: pliron::identifier::Identifier = "maxntid".try_into().unwrap();
            let minctasm_key: pliron::identifier::Identifier = "minctasm".try_into().unwrap();

            if let Some(max_attr) = attrs.get(&maxntid_key) {
                use pliron::attribute::AttrObj;
                let get_int = |attr: &AttrObj| -> Option<u32> {
                    attr.downcast_ref::<pliron::builtin::attributes::IntegerAttr>()
                        .map(|int_attr| int_attr.value().to_u32())
                };

                if let Some(max_threads) = get_int(max_attr) {
                    let min_blocks = attrs.get(&minctasm_key).and_then(get_int);
                    self.launch_bounds_kernels.push(KernelLaunchBounds {
                        name: fixed_func_name.clone(),
                        max_threads,
                        min_blocks: if min_blocks == Some(0) {
                            None
                        } else {
                            min_blocks
                        },
                    });
                }
            }
        }

        let func_type = func.get_type(self.ctx);
        let ft = Ptr::<TypeObj>::from(func_type);
        let ft_ref = ft.deref(self.ctx);
        let func_ty = ft_ref
            .downcast_ref::<FuncType>()
            .ok_or("Not a function type")?;

        let ret_ty = func_ty.result_type();

        // Check if function has a body
        if func.get_operation().deref(self.ctx).regions().count() == 0 {
            // Function Declaration
            write!(output, "declare ").unwrap();
            self.export_type(ret_ty, output)?;
            write!(output, " @{fixed_func_name}(").unwrap();

            let args = func_ty.arg_types();
            for (i, arg_ty) in args.iter().enumerate() {
                if i > 0 {
                    write!(output, ", ").unwrap();
                }
                self.export_type(*arg_ty, output)?;
            }
            write!(output, ")").unwrap();

            // Check if this is a known convergent intrinsic
            let is_convergent_intrinsic = Self::is_convergent_intrinsic(&fixed_func_name);
            if is_convergent_intrinsic {
                writeln!(output, " #0").unwrap();
                self.convergent_used = true;
            } else {
                writeln!(output).unwrap();
            }
            // No extra newline after declarations to keep them grouped
            return Ok(());
        }

        // Function Body
        let entry_block_opt = func
            .get_operation()
            .deref(self.ctx)
            .get_region(0)
            .deref(self.ctx)
            .iter(self.ctx)
            .next();

        if let Some(entry_block) = entry_block_opt {
            write!(output, "define ").unwrap();
            if is_kernel && self.emit_ptx_kernel_keyword {
                write!(output, "ptx_kernel ").unwrap();
            }
            self.export_type(ret_ty, output)?;
            write!(output, " @{fixed_func_name}(").unwrap();

            let mut value_names = HashMap::new();
            let mut next_value_id = 0;

            let block = entry_block.deref(self.ctx);
            let args = block.arguments();
            // Parameters are emitted bare: `<type> %vN` with no LLVM parameter
            // attributes (no `noalias`, `nocapture`, `dereferenceable`, etc.).
            // This is deliberate and load-bearing for `DisjointSlice`.
            //
            // `DisjointSlice::from_raw_parts` is `unsafe fn` whose contract
            // says callers must not construct two slices over the same range.
            // Violating that contract creates two `&mut T` to the same byte —
            // which is simply UB. Today, because we don't tag pointer
            // parameters with `noalias`, LLVM treats them conservatively and
            // the violation doesn't *miscompile*; it just runs as written.
            //
            // If a future change here adds `noalias` (e.g. for a perf win on
            // read-only `&[T]` inputs), that property goes away and any code
            // that double-constructed a `DisjointSlice` starts seeing folded
            // writes / reordered reads on PTX. Don't add parameter attributes
            // here without re-auditing the `from_raw_parts` callers.
            for (i, arg) in args.enumerate() {
                if i > 0 {
                    write!(output, ", ").unwrap();
                }
                let arg_ty = arg.get_type(self.ctx);
                self.export_type(arg_ty, output)?;
                let name = format!("%v{next_value_id}");
                value_names.insert(arg, name.clone());
                write!(output, " {name}").unwrap();
                next_value_id += 1;
            }
            writeln!(output, ") {{").unwrap();

            // Assign labels to all blocks
            let mut block_labels = HashMap::new();
            let mut next_label_id = 0;
            for (i, block_node) in func
                .get_operation()
                .deref(self.ctx)
                .get_region(0)
                .deref(self.ctx)
                .iter(self.ctx)
                .enumerate()
            {
                if i == 0 {
                    // Entry block usually doesn't need label in LLVM if it's first
                    block_labels.insert(block_node, "entry".to_string());
                } else {
                    let label = format!("bb{next_label_id}");
                    next_label_id += 1;
                    block_labels.insert(block_node, label);
                }
            }

            // PRE-PASS: Assign names to ALL values before exporting
            // This is needed because PHI nodes may reference values from blocks that
            // come later in the block list (e.g., back-edges in loops).
            for block_node in func
                .get_operation()
                .deref(self.ctx)
                .get_region(0)
                .deref(self.ctx)
                .iter(self.ctx)
            {
                // Name block arguments (skip entry block which was already done)
                if block_node != entry_block {
                    for arg in block_node.deref(self.ctx).arguments() {
                        let name = format!("%v{next_value_id}");
                        next_value_id += 1;
                        value_names.insert(arg, name);
                    }
                }

                // Name operation results
                for op in block_node.deref(self.ctx).iter(self.ctx) {
                    let op_ref = op.deref(self.ctx);
                    let op_id = Operation::get_opid(op, self.ctx);

                    // Skip ops that don't produce named results (UndefOp is handled specially)
                    if op_id == ops::UndefOp::get_opid_static() {
                        // UndefOp result will be named "undef"
                        continue;
                    }

                    // CRITICAL: ConstantOp MUST be registered in pre-pass, not during export!
                    // PHI nodes may reference constants from blocks that appear later in the
                    // iteration order. If we delay constant naming until export, the PHI
                    // export will fail to find the constant in value_names and emit "undef".
                    //
                    // Example: bb6 has PHI receiving constant 0 from bb14, but bb6 is
                    // exported before bb14. Without pre-pass registration, the constant's
                    // Value is not in value_names when bb6's PHI is emitted.
                    if op_id == ops::ConstantOp::get_opid_static() {
                        let const_op = Operation::get_op::<ops::ConstantOp>(op, self.ctx).unwrap();
                        let val_attr = const_op.get_value(self.ctx);

                        let const_str = if let Some(int_attr) =
                            val_attr.downcast_ref::<IntegerAttr>()
                        {
                            int_attr.value().to_string_unsigned_decimal()
                        } else if let Some(fp16_attr) = val_attr.downcast_ref::<FPHalfAttr>() {
                            format_half_literal(fp16_attr.to_bits())
                        } else if let Some(fp32_attr) = val_attr.downcast_ref::<FPSingleAttr>() {
                            let float_val: f32 = fp32_attr.clone().into();
                            format_float_literal(f64::from(float_val))
                        } else if let Some(fp64_attr) = val_attr.downcast_ref::<FPDoubleAttr>() {
                            let float_val: f64 = fp64_attr.clone().into();
                            format_float_literal(float_val)
                        } else {
                            "0".to_string() // Fallback
                        };

                        let res = op_ref.get_result(0);
                        value_names.insert(res, const_str);
                        continue;
                    }

                    // CRITICAL: AddressOfOp is also "virtual" — its export emits no
                    // instruction text and just remaps its result to `@global_name`.
                    // If the AddressOfOp lives in a block exported AFTER its uses
                    // (e.g. mir-importer's static-translation path inserts it deep
                    // in a conditional chain whose block comes later in the
                    // iteration), the uses get the stale `%vN` fallback before the
                    // export pass overwrites with the global name. Pre-pass the
                    // mapping the same way ConstantOp does.
                    if op_id == ops::AddressOfOp::get_opid_static() {
                        let address_of = Operation::get_op::<ops::AddressOfOp>(op, self.ctx).unwrap();
                        let global_name = address_of.get_global_name(self.ctx);
                        let res = op_ref.get_result(0);
                        value_names.insert(res, format!("@{global_name}"));
                        continue;
                    }

                    for res in op_ref.results() {
                        let name = format!("%v{next_value_id}");
                        next_value_id += 1;
                        value_names.insert(res, name);
                    }
                }
            }

            // Build predecessor map for PHI generation
            let mut pred_map: PredecessorMap = HashMap::new();
            for block in func
                .get_operation()
                .deref(self.ctx)
                .get_region(0)
                .deref(self.ctx)
                .iter(self.ctx)
            {
                let block_ref = block.deref(self.ctx);
                if let Some(term) = block_ref.iter(self.ctx).last() {
                    let op_id = Operation::get_opid(term, self.ctx);

                    if op_id == ops::BrOp::get_opid_static() {
                        // BrOp has 1 successor and all operands are passed to it
                        let dest = term.deref(self.ctx).successors().next().unwrap();
                        let args: Vec<_> = term.deref(self.ctx).operands().collect();
                        pred_map.entry(dest).or_default().push((block, args));
                    } else if op_id == ops::CondBrOp::get_opid_static() {
                        let succs: Vec<_> = term.deref(self.ctx).successors().collect();
                        let true_dest = succs[0];
                        let false_dest = succs[1];

                        // Calculate split point for operands
                        // [cond, true_args..., false_args...]
                        let num_true = true_dest.deref(self.ctx).arguments().count();
                        let num_false = false_dest.deref(self.ctx).arguments().count();

                        let all_ops: Vec<_> = term.deref(self.ctx).operands().collect();
                        if all_ops.len() >= 1 + num_true + num_false {
                            let true_args = all_ops[1..=num_true].to_vec();
                            let false_args =
                                all_ops[1 + num_true..1 + num_true + num_false].to_vec();

                            pred_map
                                .entry(true_dest)
                                .or_default()
                                .push((block, true_args));
                            pred_map
                                .entry(false_dest)
                                .or_default()
                                .push((block, false_args));
                        }
                    }
                }
            }

            // Export blocks
            for (i, block_node) in func
                .get_operation()
                .deref(self.ctx)
                .get_region(0)
                .deref(self.ctx)
                .iter(self.ctx)
                .enumerate()
            {
                self.export_block(
                    block_node,
                    &mut value_names,
                    &mut next_value_id,
                    &block_labels,
                    &pred_map,
                    i == 0,
                    output,
                )?;
            }

            writeln!(output, "}}").unwrap();
        } else {
            // This block handled the declaration case, but we now check get_num_regions() above.
            // If we are here, get_num_regions() >= 1 but entry_block_opt is None (empty region).
            // This is also a declaration in some contexts, or an empty function.
            // Let's treat it as a declaration if it's empty.

            write!(output, "declare ").unwrap();
            self.export_type(ret_ty, output)?;
            write!(output, " @{fixed_func_name}(").unwrap();

            let args = func_ty.arg_types();
            for (i, arg_ty) in args.iter().enumerate() {
                if i > 0 {
                    write!(output, ", ").unwrap();
                }
                self.export_type(*arg_ty, output)?;
            }
            writeln!(output, ")").unwrap();
        }

        writeln!(output).unwrap();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn export_block(
        &mut self,
        block: Ptr<BasicBlock>,
        value_names: &mut HashMap<Value, String>,
        next_value_id: &mut usize,
        block_labels: &HashMap<Ptr<BasicBlock>, String>,
        pred_map: &PredecessorMap,
        is_entry: bool,
        output: &mut String,
    ) -> Result<(), String> {
        // Always print label to ensure it can be referenced by PHI nodes
        let label = block_labels.get(&block).unwrap();
        writeln!(output, "{label}:").unwrap();

        // Generate PHI nodes for block arguments (except entry block which uses function args)
        let args: Vec<_> = block.deref(self.ctx).arguments().collect();
        if !args.is_empty() && !is_entry {
            let preds = pred_map
                .get(&block)
                .ok_or_else(|| "Block with args has no predecessors".to_string())?;

            for (arg_idx, arg) in args.iter().enumerate() {
                // Use pre-assigned name or generate new one
                let arg_name = if let Some(name) = value_names.get(arg) {
                    name.clone()
                } else {
                    let name = format!("%v{next_value_id}");
                    *next_value_id += 1;
                    value_names.insert(*arg, name.clone());
                    name
                };

                write!(output, "  {arg_name} = phi ").unwrap();
                self.export_type(arg.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();

                for (i, (pred_block, pred_args)) in preds.iter().enumerate() {
                    if i > 0 {
                        write!(output, ", ").unwrap();
                    }

                    if arg_idx < pred_args.len() {
                        let val = pred_args[arg_idx];
                        write!(output, "[ ").unwrap();
                        self.export_value(val, value_names, output)?;
                        let label = block_labels.get(pred_block).unwrap();
                        write!(output, ", %{label} ]").unwrap();
                    } else {
                        write!(
                            output,
                            "[ undef, %{} ]",
                            block_labels.get(pred_block).unwrap()
                        )
                        .unwrap();
                    }
                }
                writeln!(output).unwrap();
            }
        }

        for op in block.deref(self.ctx).iter(self.ctx) {
            self.export_op(op, value_names, next_value_id, block_labels, output)?;
        }
        Ok(())
    }

    fn export_op(
        &mut self,
        op: Ptr<Operation>,
        value_names: &mut HashMap<Value, String>,
        next_value_id: &mut usize,
        block_labels: &HashMap<Ptr<BasicBlock>, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.deref(self.ctx);
        let op_id = Operation::get_opid(op, self.ctx);
        let op_obj = Operation::get_op_dyn(op, self.ctx);

        // Register result names (skip if already named in pre-pass)
        for res in op_ref.results() {
            value_names.entry(res).or_insert_with(|| {
                let name = format!("%v{next_value_id}");
                *next_value_id += 1;
                name.clone()
            });
        }

        // Match on operation type using guards (op_id is runtime, not enum)
        match op_id {
            // --- Terminators ---
            id if id == ops::ReturnOp::get_opid_static() => {
                write!(output, "  ret ").unwrap();
                if op_ref.operands().count() == 0 {
                    write!(output, "void").unwrap();
                } else {
                    let val = op_ref.operands().next().unwrap();
                    self.export_type(val.get_type(self.ctx), output)?;
                    write!(output, " ").unwrap();
                    self.export_value(val, value_names, output)?;
                }
                writeln!(output).unwrap();
            }
            id if id == ops::UnreachableOp::get_opid_static() => {
                writeln!(output, "  unreachable").unwrap();
            }
            id if id == ops::BrOp::get_opid_static() => {
                let dest = op_ref.successors().next().unwrap();
                let label = block_labels.get(&dest).ok_or("Missing block label")?;
                writeln!(output, "  br label %{label}").unwrap();
            }
            id if id == ops::CondBrOp::get_opid_static() => {
                let mut succs = op_ref.successors();
                let true_dest = succs.next().unwrap();
                let false_dest = succs.next().unwrap();
                let true_label = block_labels.get(&true_dest).ok_or("Missing true label")?;
                let false_label = block_labels.get(&false_dest).ok_or("Missing false label")?;
                let cond = op_ref.get_operand(0);

                write!(output, "  br i1 ").unwrap();
                self.export_value(cond, value_names, output)?;
                writeln!(output, ", label %{true_label}, label %{false_label}").unwrap();
            }

            // --- Memory Ops ---
            id if id == ops::LoadOp::get_opid_static() => {
                let res = op_ref.get_result(0);
                let ptr = op_ref.get_operand(0);
                let res_name = value_names.get(&res).unwrap();
                let ty = res.get_type(self.ctx);

                // Check pointer address space
                let ptr_ty = ptr.get_type(self.ctx);
                let addrspace = ptr_ty
                    .deref(self.ctx)
                    .downcast_ref::<PointerType>()
                    .map_or(0, super::types::PointerType::address_space);

                write!(output, "  {res_name} = load ").unwrap();
                self.export_type(ty, output)?;
                if addrspace != 0 {
                    write!(output, ", ptr addrspace({addrspace}) ").unwrap();
                } else {
                    write!(output, ", ptr ").unwrap();
                }
                self.export_value(ptr, value_names, output)?;
                writeln!(output).unwrap();
            }
            id if id == ops::StoreOp::get_opid_static() => {
                let val = op_ref.get_operand(0);
                let ptr = op_ref.get_operand(1);

                // Check pointer address space
                let ptr_ty = ptr.get_type(self.ctx);
                let addrspace = ptr_ty
                    .deref(self.ctx)
                    .downcast_ref::<PointerType>()
                    .map_or(0, super::types::PointerType::address_space);

                write!(output, "  store ").unwrap();
                self.export_type(val.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();
                self.export_value(val, value_names, output)?;
                if addrspace != 0 {
                    write!(output, ", ptr addrspace({addrspace}) ").unwrap();
                } else {
                    write!(output, ", ptr ").unwrap();
                }
                self.export_value(ptr, value_names, output)?;
                writeln!(output).unwrap();
            }
            // --- Atomic Ops ---
            id if id == ops::AtomicLoadOp::get_opid_static() => {
                // %val = load atomic i32, ptr [addrspace(N)] %p syncscope("device") acquire
                let atomic_load = op_obj.as_ref().downcast_ref::<ops::AtomicLoadOp>().unwrap();
                let res = op_ref.get_result(0);
                let ptr = op_ref.get_operand(0);
                let res_name = value_names.get(&res).unwrap();
                let ty = res.get_type(self.ctx);
                let syncscope_str = ops::atomic::format_syncscope(&atomic_load.syncscope(self.ctx));
                let ordering_str = ops::atomic::format_ordering(&atomic_load.ordering(self.ctx));

                let ptr_ty = ptr.get_type(self.ctx);
                let addrspace = ptr_ty
                    .deref(self.ctx)
                    .downcast_ref::<PointerType>()
                    .map_or(0, super::types::PointerType::address_space);

                write!(output, "  {res_name} = load atomic ").unwrap();
                self.export_type(ty, output)?;
                if addrspace != 0 {
                    write!(output, ", ptr addrspace({addrspace}) ").unwrap();
                } else {
                    write!(output, ", ptr ").unwrap();
                }
                self.export_value(ptr, value_names, output)?;
                let align = self.natural_alignment(ty);
                writeln!(output, "{syncscope_str} {ordering_str}, align {align}").unwrap();
            }
            id if id == ops::AtomicStoreOp::get_opid_static() => {
                // store atomic i32 %v, ptr [addrspace(N)] %p syncscope("device") release
                let atomic_store = op_obj
                    .as_ref()
                    .downcast_ref::<ops::AtomicStoreOp>()
                    .unwrap();
                let val = op_ref.get_operand(0);
                let ptr = op_ref.get_operand(1);
                let syncscope_str =
                    ops::atomic::format_syncscope(&atomic_store.syncscope(self.ctx));
                let ordering_str = ops::atomic::format_ordering(&atomic_store.ordering(self.ctx));

                let ptr_ty = ptr.get_type(self.ctx);
                let addrspace = ptr_ty
                    .deref(self.ctx)
                    .downcast_ref::<PointerType>()
                    .map_or(0, super::types::PointerType::address_space);

                write!(output, "  store atomic ").unwrap();
                self.export_type(val.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();
                self.export_value(val, value_names, output)?;
                if addrspace != 0 {
                    write!(output, ", ptr addrspace({addrspace}) ").unwrap();
                } else {
                    write!(output, ", ptr ").unwrap();
                }
                self.export_value(ptr, value_names, output)?;
                let align = self.natural_alignment(val.get_type(self.ctx));
                writeln!(output, "{syncscope_str} {ordering_str}, align {align}").unwrap();
            }
            id if id == ops::AtomicRmwOp::get_opid_static() => {
                // %old = atomicrmw add ptr [addrspace(N)] %p, i32 %v syncscope("device") monotonic
                let atomic_rmw = op_obj.as_ref().downcast_ref::<ops::AtomicRmwOp>().unwrap();
                let res = op_ref.get_result(0);
                let ptr = op_ref.get_operand(0);
                let val = op_ref.get_operand(1);
                let res_name = value_names.get(&res).unwrap();
                let rmw_kind_str = ops::atomic::format_rmw_kind(&atomic_rmw.rmw_kind(self.ctx));
                let syncscope_str = ops::atomic::format_syncscope(&atomic_rmw.syncscope(self.ctx));
                let ordering_str = ops::atomic::format_ordering(&atomic_rmw.ordering(self.ctx));

                let ptr_ty = ptr.get_type(self.ctx);
                let addrspace = ptr_ty
                    .deref(self.ctx)
                    .downcast_ref::<PointerType>()
                    .map_or(0, super::types::PointerType::address_space);

                write!(output, "  {res_name} = atomicrmw {rmw_kind_str} ").unwrap();
                if addrspace != 0 {
                    write!(output, "ptr addrspace({addrspace}) ").unwrap();
                } else {
                    write!(output, "ptr ").unwrap();
                }
                self.export_value(ptr, value_names, output)?;
                write!(output, ", ").unwrap();
                self.export_type(val.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();
                self.export_value(val, value_names, output)?;
                writeln!(output, "{syncscope_str} {ordering_str}").unwrap();
            }
            id if id == ops::AtomicCmpxchgOp::get_opid_static() => {
                // %result_struct = cmpxchg ptr %p, i32 %cmp, i32 %new syncscope("device") acq_rel acquire
                // %old = extractvalue { i32, i1 } %result_struct, 0
                // %success = extractvalue { i32, i1 } %result_struct, 1
                let atomic_cmpxchg = op_obj
                    .as_ref()
                    .downcast_ref::<ops::AtomicCmpxchgOp>()
                    .unwrap();
                let res = op_ref.get_result(0);
                let ptr = op_ref.get_operand(0);
                let cmp = op_ref.get_operand(1);
                let new_val = op_ref.get_operand(2);
                let res_name = value_names.get(&res).unwrap();
                let success_ord_str =
                    ops::atomic::format_ordering(&atomic_cmpxchg.success_ordering(self.ctx));
                let failure_ord_str =
                    ops::atomic::format_ordering(&atomic_cmpxchg.failure_ordering(self.ctx));
                let syncscope_str =
                    ops::atomic::format_syncscope(&atomic_cmpxchg.syncscope(self.ctx));
                let val_ty = cmp.get_type(self.ctx);

                let ptr_ty = ptr.get_type(self.ctx);
                let addrspace = ptr_ty
                    .deref(self.ctx)
                    .downcast_ref::<PointerType>()
                    .map_or(0, super::types::PointerType::address_space);

                // Emit the cmpxchg instruction -- returns { T, i1 }
                let struct_name = format!("{res_name}.cx");
                write!(output, "  {struct_name} = cmpxchg ").unwrap();
                if addrspace != 0 {
                    write!(output, "ptr addrspace({addrspace}) ").unwrap();
                } else {
                    write!(output, "ptr ").unwrap();
                }
                self.export_value(ptr, value_names, output)?;
                write!(output, ", ").unwrap();
                self.export_type(val_ty, output)?;
                write!(output, " ").unwrap();
                self.export_value(cmp, value_names, output)?;
                write!(output, ", ").unwrap();
                self.export_type(val_ty, output)?;
                write!(output, " ").unwrap();
                self.export_value(new_val, value_names, output)?;
                writeln!(
                    output,
                    "{syncscope_str} {success_ord_str} {failure_ord_str}"
                )
                .unwrap();

                // Extract the old value (element 0 of the { T, i1 } struct)
                write!(output, "  {res_name} = extractvalue {{ ").unwrap();
                self.export_type(val_ty, output)?;
                writeln!(output, ", i1 }} {struct_name}, 0").unwrap();
            }
            id if id == ops::FenceOp::get_opid_static() => {
                // fence syncscope("device") release
                let fence = op_obj.as_ref().downcast_ref::<ops::FenceOp>().unwrap();
                let syncscope_str = ops::atomic::format_syncscope(&fence.syncscope(self.ctx));
                let ordering_str = ops::atomic::format_ordering(&fence.ordering(self.ctx));
                writeln!(output, "  fence{syncscope_str} {ordering_str}").unwrap();
            }

            id if id == ops::AllocaOp::get_opid_static() => {
                // %res = alloca <type>, align <N>
                let res = op_ref.get_result(0);
                let res_name = value_names.get(&res).unwrap();

                // Get the element type from the attribute
                let alloca_op = op_obj.as_ref().downcast_ref::<ops::AllocaOp>().unwrap();
                let elem_ty = alloca_op
                    .get_attr_alloca_element_type(self.ctx)
                    .expect("Missing alloca_element_type");
                let elem_ty_ptr = elem_ty.get_type(self.ctx);

                // Conservatively over-align the alloca so wider-than-natural
                // loads/stores against this storage stay aligned. See doc
                // on `conservative_alloca_align` for the tradeoff + TODO.
                let align = self.conservative_alloca_align(elem_ty_ptr);

                write!(output, "  {res_name} = alloca ").unwrap();
                self.export_type(elem_ty_ptr, output)?;
                writeln!(output, ", align {align}").unwrap();
            }
            id if id == ops::GetElementPtrOp::get_opid_static() => {
                // %res = getelementptr inbounds TYPE, ptr addrspace(N) %ptr, i32 %idx...
                let res = op_ref.get_result(0);
                let res_name = value_names.get(&res).unwrap();
                let ptr = op_ref.get_operand(0);

                let gep_op = op_obj
                    .as_ref()
                    .downcast_ref::<ops::GetElementPtrOp>()
                    .unwrap();
                let elem_ty = gep_op
                    .get_attr_gep_src_elem_type(self.ctx)
                    .expect("Missing gep_src_elem_type")
                    .get_type(self.ctx); // Ptr<TypeObj>

                write!(output, "  {res_name} = getelementptr inbounds ").unwrap();
                self.export_type(elem_ty, output)?;

                // Check if pointer has a non-default address space
                let ptr_ty = ptr.get_type(self.ctx);
                let addrspace = ptr_ty
                    .deref(self.ctx)
                    .downcast_ref::<PointerType>()
                    .map_or(0, super::types::PointerType::address_space);

                if addrspace != 0 {
                    write!(output, ", ptr addrspace({addrspace}) ").unwrap();
                } else {
                    write!(output, ", ptr ").unwrap();
                }
                self.export_value(ptr, value_names, output)?;

                // Indices
                let indices = &gep_op.get_attr_gep_indices(self.ctx).unwrap().0;
                for idx_attr in indices {
                    write!(output, ", ").unwrap();
                    match idx_attr {
                        GepIndexAttr::Constant(val) => {
                            write!(output, "i32 {val}").unwrap();
                        }
                        GepIndexAttr::OperandIdx(operand_idx) => {
                            let val = op_ref.get_operand(*operand_idx);
                            self.export_type(val.get_type(self.ctx), output)?;
                            write!(output, " ").unwrap();
                            self.export_value(val, value_names, output)?;
                        }
                    }
                }
                writeln!(output).unwrap();
            }

            // --- Arithmetic ---
            id if id == ops::AddOp::get_opid_static() => {
                self.export_binop("add", op, value_names, output)?;
            }
            id if id == ops::SubOp::get_opid_static() => {
                self.export_binop("sub", op, value_names, output)?;
            }
            id if id == ops::MulOp::get_opid_static() => {
                self.export_binop("mul", op, value_names, output)?;
            }
            id if id == ops::FAddOp::get_opid_static() => {
                self.export_binop("fadd", op, value_names, output)?;
            }
            id if id == ops::FSubOp::get_opid_static() => {
                self.export_binop("fsub", op, value_names, output)?;
            }
            id if id == ops::FMulOp::get_opid_static() => {
                self.export_binop("fmul", op, value_names, output)?;
            }
            id if id == ops::FDivOp::get_opid_static() => {
                self.export_binop("fdiv", op, value_names, output)?;
            }
            id if id == ops::FRemOp::get_opid_static() => {
                self.export_binop("frem", op, value_names, output)?;
            }
            id if id == ops::FNegOp::get_opid_static() => {
                let res = op_ref.get_result(0);
                let res_name = value_names.get(&res).unwrap();
                let arg = op_ref.get_operand(0);

                write!(output, "  {res_name} = fneg ").unwrap();
                self.export_type(arg.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();
                self.export_value(arg, value_names, output)?;
                writeln!(output).unwrap();
            }
            id if id == ops::SDivOp::get_opid_static() => {
                self.export_binop("sdiv", op, value_names, output)?;
            }
            id if id == ops::UDivOp::get_opid_static() => {
                self.export_binop("udiv", op, value_names, output)?;
            }
            id if id == ops::SRemOp::get_opid_static() => {
                self.export_binop("srem", op, value_names, output)?;
            }
            id if id == ops::URemOp::get_opid_static() => {
                self.export_binop("urem", op, value_names, output)?;
            }
            id if id == ops::XorOp::get_opid_static() => {
                self.export_binop("xor", op, value_names, output)?;
            }
            id if id == ops::ShlOp::get_opid_static() => {
                self.export_binop("shl", op, value_names, output)?;
            }
            id if id == ops::LShrOp::get_opid_static() => {
                self.export_binop("lshr", op, value_names, output)?;
            }
            id if id == ops::AShrOp::get_opid_static() => {
                self.export_binop("ashr", op, value_names, output)?;
            }
            id if id == ops::AndOp::get_opid_static() => {
                self.export_binop("and", op, value_names, output)?;
            }
            id if id == ops::OrOp::get_opid_static() => {
                self.export_binop("or", op, value_names, output)?;
            }
            id if id == ops::ICmpOp::get_opid_static() => {
                let res = op_ref.get_result(0);
                let res_name = value_names.get(&res).unwrap();
                let lhs = op_ref.get_operand(0);
                let rhs = op_ref.get_operand(1);

                let icmp = op_obj.as_ref().downcast_ref::<ops::ICmpOp>().unwrap();
                let pred_attr = icmp.predicate(self.ctx);
                let pred_str = match pred_attr {
                    ICmpPredicateAttr::EQ => "eq",
                    ICmpPredicateAttr::NE => "ne",
                    ICmpPredicateAttr::SLT => "slt",
                    ICmpPredicateAttr::SLE => "sle",
                    ICmpPredicateAttr::SGT => "sgt",
                    ICmpPredicateAttr::SGE => "sge",
                    ICmpPredicateAttr::ULT => "ult",
                    ICmpPredicateAttr::ULE => "ule",
                    ICmpPredicateAttr::UGT => "ugt",
                    ICmpPredicateAttr::UGE => "uge",
                };

                write!(output, "  {res_name} = icmp {pred_str} ").unwrap();
                self.export_type(lhs.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();
                self.export_value(lhs, value_names, output)?;
                write!(output, ", ").unwrap();
                self.export_value(rhs, value_names, output)?;
                writeln!(output).unwrap();
            }
            id if id == ops::FCmpOp::get_opid_static() => {
                let res = op_ref.get_result(0);
                let res_name = value_names.get(&res).unwrap();
                let lhs = op_ref.get_operand(0);
                let rhs = op_ref.get_operand(1);

                let fcmp = op_obj.as_ref().downcast_ref::<ops::FCmpOp>().unwrap();
                let pred_attr = fcmp.predicate(self.ctx);
                let pred_str = match pred_attr {
                    crate::attributes::FCmpPredicateAttr::False => "false",
                    crate::attributes::FCmpPredicateAttr::OEQ => "oeq",
                    crate::attributes::FCmpPredicateAttr::OGT => "ogt",
                    crate::attributes::FCmpPredicateAttr::OGE => "oge",
                    crate::attributes::FCmpPredicateAttr::OLT => "olt",
                    crate::attributes::FCmpPredicateAttr::OLE => "ole",
                    crate::attributes::FCmpPredicateAttr::ONE => "one",
                    crate::attributes::FCmpPredicateAttr::ORD => "ord",
                    crate::attributes::FCmpPredicateAttr::UEQ => "ueq",
                    crate::attributes::FCmpPredicateAttr::UGT => "ugt",
                    crate::attributes::FCmpPredicateAttr::UGE => "uge",
                    crate::attributes::FCmpPredicateAttr::ULT => "ult",
                    crate::attributes::FCmpPredicateAttr::ULE => "ule",
                    crate::attributes::FCmpPredicateAttr::UNE => "une",
                    crate::attributes::FCmpPredicateAttr::UNO => "uno",
                    crate::attributes::FCmpPredicateAttr::True => "true",
                };

                write!(output, "  {res_name} = fcmp {pred_str} ").unwrap();
                self.export_type(lhs.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();
                self.export_value(lhs, value_names, output)?;
                write!(output, ", ").unwrap();
                self.export_value(rhs, value_names, output)?;
                writeln!(output).unwrap();
            }

            // --- Calls ---
            // LLVM call instruction format:
            //   - Non-void: %result = call <ret_type> @func(<args>)
            //   - Void:     call void @func(<args>)
            //
            // IMPORTANT: Void-returning calls must NOT have a result assignment.
            // Invalid: "%v1 = call void @foo()" - llc will reject this!
            // Valid:   "call void @foo()"
            id if id == ops::CallOp::get_opid_static() => {
                let call = op_obj.as_ref().downcast_ref::<ops::CallOp>().unwrap();
                let callee = call.callee(self.ctx);

                // Extract return type from the call's function type to determine
                // if this is a void call (no result assignment) or value call
                let func_ty = call.callee_type(self.ctx);
                let func_ty_ref = func_ty.deref(self.ctx);
                let llvm_func_ty = func_ty_ref.downcast_ref::<FuncType>().unwrap();
                let ret_ty = llvm_func_ty.result_type();
                let is_void = ret_ty.deref(self.ctx).is::<VoidType>();

                // Void calls: "call void @func(...)"
                // Non-void:   "%vN = call <type> @func(...)"
                if is_void {
                    write!(output, "  call void").unwrap();
                } else {
                    let res = op_ref.get_result(0);
                    let res_name = value_names.get(&res).unwrap();
                    write!(output, "  {res_name} = call ").unwrap();
                    self.export_type(ret_ty, output)?;
                }

                // Track if callee is a convergent intrinsic
                let mut is_convergent_call = false;

                // Callee can be direct (@function_name) or indirect (function pointer)
                match callee {
                    CallOpCallable::Direct(identifier) => {
                        let name = identifier.to_string();
                        // LLVM intrinsics use dots in IR; Pliron IR identifiers use underscores.
                        let fixed_name = if name.starts_with("llvm_") {
                            name.replace('_', ".")
                        } else {
                            // Strip cuda_oxide_device_ prefix from call targets to match
                            // the stripped function definitions (clean export names).
                            strip_device_prefix(&name)
                        };
                        is_convergent_call = Self::is_convergent_intrinsic(&fixed_name);
                        write!(output, " @{fixed_name}(").unwrap();
                    }
                    CallOpCallable::Indirect(val) => {
                        write!(output, " ").unwrap();
                        self.export_value(val, value_names, output).unwrap();
                        write!(output, "(").unwrap();
                    }
                }

                // Export call arguments with their types
                for (i, arg) in op_ref.operands().enumerate() {
                    if i > 0 {
                        write!(output, ", ").unwrap();
                    }
                    self.export_type(arg.get_type(self.ctx), output)?;
                    write!(output, " ").unwrap();
                    self.export_value(arg, value_names, output)?;
                }

                // Add convergent attribute reference for sync intrinsics
                if is_convergent_call {
                    writeln!(output, ") #0").unwrap();
                    self.convergent_used = true;
                } else {
                    writeln!(output, ")").unwrap();
                }
            }

            // --- Inline Assembly ---
            id if id == ops::InlineAsmOp::get_opid_static() => {
                let inline_asm = op_obj.as_ref().downcast_ref::<ops::InlineAsmOp>().unwrap();
                let asm_template = inline_asm.asm_template(self.ctx);
                let constraints = inline_asm.constraints(self.ctx);
                let is_convergent = inline_asm.is_convergent(self.ctx);

                // Check if there's a result
                let has_result = op_ref.get_num_results() > 0;

                if has_result {
                    let res = op_ref.get_result(0);
                    let res_name = value_names.get(&res).unwrap();
                    let res_ty = res.get_type(self.ctx);
                    write!(output, "  {res_name} = call ").unwrap();
                    self.export_type(res_ty, output)?;
                } else {
                    write!(output, "  call void").unwrap();
                }

                // Format: call <type> asm sideeffect "<template>", "<constraints>"(<args>...)
                write!(
                    output,
                    " asm sideeffect \"{asm_template}\", \"{constraints}\"("
                )
                .unwrap();

                // Export input operands with types
                for (i, arg) in op_ref.operands().enumerate() {
                    if i > 0 {
                        write!(output, ", ").unwrap();
                    }
                    self.export_type(arg.get_type(self.ctx), output)?;
                    write!(output, " ").unwrap();
                    self.export_value(arg, value_names, output)?;
                }

                // Add convergent attribute reference if needed
                if is_convergent {
                    writeln!(output, ") #0").unwrap();
                    self.convergent_used = true;
                } else {
                    writeln!(output, ")").unwrap();
                }
            }

            // --- Multi-Output Inline Assembly ---
            id if id == ops::InlineAsmMultiOp::get_opid_static() => {
                let inline_asm = op_obj
                    .as_ref()
                    .downcast_ref::<ops::InlineAsmMultiOp>()
                    .unwrap();
                let asm_template = inline_asm.asm_template(self.ctx);
                let constraints = inline_asm.constraints(self.ctx);
                let is_convergent = inline_asm.is_convergent(self.ctx);
                let num_results = op_ref.get_num_results();

                if num_results == 0 {
                    // Void return - simple case
                    write!(output, "  call void").unwrap();
                    write!(
                        output,
                        " asm sideeffect \"{asm_template}\", \"{constraints}\"("
                    )
                    .unwrap();

                    for (i, arg) in op_ref.operands().enumerate() {
                        if i > 0 {
                            write!(output, ", ").unwrap();
                        }
                        self.export_type(arg.get_type(self.ctx), output)?;
                        write!(output, " ").unwrap();
                        self.export_value(arg, value_names, output)?;
                    }

                    if is_convergent {
                        writeln!(output, ") #0").unwrap();
                        self.convergent_used = true;
                    } else {
                        writeln!(output, ")").unwrap();
                    }
                } else {
                    // Multi-output: returns a struct, need extractvalue for each
                    // Step 1: Build the struct type string
                    let mut struct_type = String::from("{");
                    for i in 0..num_results {
                        if i > 0 {
                            struct_type.push_str(", ");
                        }
                        let res_ty = op_ref.get_result(i).get_type(self.ctx);
                        let mut ty_str = String::new();
                        self.export_type(res_ty, &mut ty_str)?;
                        struct_type.push_str(&ty_str);
                    }
                    struct_type.push('}');

                    // Step 2: Generate the asm call returning the struct
                    // We need a temporary name for the struct result
                    // Use the first result's name with "_struct" suffix
                    let first_res = op_ref.get_result(0);
                    let first_res_name = value_names.get(&first_res).unwrap();
                    let struct_result_name = format!("{first_res_name}_struct");

                    write!(output, "  {struct_result_name} = call {struct_type}").unwrap();
                    write!(
                        output,
                        " asm sideeffect \"{asm_template}\", \"{constraints}\"("
                    )
                    .unwrap();

                    for (i, arg) in op_ref.operands().enumerate() {
                        if i > 0 {
                            write!(output, ", ").unwrap();
                        }
                        self.export_type(arg.get_type(self.ctx), output)?;
                        write!(output, " ").unwrap();
                        self.export_value(arg, value_names, output)?;
                    }

                    if is_convergent {
                        writeln!(output, ") #0").unwrap();
                        self.convergent_used = true;
                    } else {
                        writeln!(output, ")").unwrap();
                    }

                    // Step 3: Generate extractvalue for each result
                    for i in 0..num_results {
                        let res = op_ref.get_result(i);
                        let res_name = value_names.get(&res).unwrap();

                        writeln!(
                            output,
                            "  {res_name} = extractvalue {struct_type} {struct_result_name}, {i}"
                        )
                        .unwrap();
                    }
                }
            }

            // --- Casts ---
            id if id == ops::BitcastOp::get_opid_static() => {
                self.export_cast("bitcast", op, value_names, output)?;
            }
            id if id == ops::AddrSpaceCastOp::get_opid_static() => {
                self.export_cast("addrspacecast", op, value_names, output)?;
            }
            id if id == ops::ZExtOp::get_opid_static() => {
                let res = op_ref.get_result(0);
                let res_name = value_names.get(&res).unwrap();
                let val = op_ref.get_operand(0);

                let zext = op_obj.as_ref().downcast_ref::<ops::ZExtOp>().unwrap();
                // Manual attribute access since helper is missing
                let nneg_key: pliron::identifier::Identifier = "llvm_nneg_flag".try_into().unwrap();
                let nneg = zext
                    .get_operation()
                    .deref(self.ctx)
                    .attributes
                    .0
                    .get(&nneg_key)
                    .and_then(|attr| {
                        attr.downcast_ref::<pliron::builtin::attributes::BoolAttr>()
                            .map(|b| bool::from(b.clone()))
                    })
                    .unwrap_or(false);

                write!(output, "  {res_name} = zext ").unwrap();
                if nneg {
                    write!(output, "nneg ").unwrap();
                }
                self.export_type(val.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();
                self.export_value(val, value_names, output)?;
                write!(output, " to ").unwrap();
                self.export_type(res.get_type(self.ctx), output)?;
                writeln!(output).unwrap();
            }
            id if id == ops::SExtOp::get_opid_static() => {
                self.export_cast("sext", op, value_names, output)?;
            }
            id if id == ops::TruncOp::get_opid_static() => {
                self.export_cast("trunc", op, value_names, output)?;
            }
            id if id == ops::PtrToIntOp::get_opid_static() => {
                self.export_cast("ptrtoint", op, value_names, output)?;
            }
            id if id == ops::IntToPtrOp::get_opid_static() => {
                self.export_cast("inttoptr", op, value_names, output)?;
            }
            id if id == ops::UIToFPOp::get_opid_static() => {
                self.export_cast("uitofp", op, value_names, output)?;
            }
            id if id == ops::SIToFPOp::get_opid_static() => {
                self.export_cast("sitofp", op, value_names, output)?;
            }
            id if id == ops::FPToUIOp::get_opid_static() => {
                self.export_cast("fptoui", op, value_names, output)?;
            }
            id if id == ops::FPToSIOp::get_opid_static() => {
                self.export_cast("fptosi", op, value_names, output)?;
            }
            id if id == ops::FPExtOp::get_opid_static() => {
                self.export_cast("fpext", op, value_names, output)?;
            }
            id if id == ops::FPTruncOp::get_opid_static() => {
                self.export_cast("fptrunc", op, value_names, output)?;
            }
            id if id == ops::UndefOp::get_opid_static() => {
                let res = op_ref.get_result(0);
                value_names.insert(res, "undef".to_string());
            }

            // --- Aggregate Ops ---
            id if id == ops::ExtractValueOp::get_opid_static() => {
                let res = op_ref.get_result(0);
                let res_name = value_names.get(&res).unwrap();
                let agg = op_ref.get_operand(0);

                let extract_op = op_obj
                    .as_ref()
                    .downcast_ref::<ops::ExtractValueOp>()
                    .unwrap();
                let indices = extract_op.indices(self.ctx);

                write!(output, "  {res_name} = extractvalue ").unwrap();
                self.export_type(agg.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();
                self.export_value(agg, value_names, output)?;
                for idx in indices {
                    write!(output, ", {idx}").unwrap();
                }
                writeln!(output).unwrap();
            }
            id if id == ops::InsertValueOp::get_opid_static() => {
                let res = op_ref.get_result(0);
                let res_name = value_names.get(&res).unwrap();
                let agg = op_ref.get_operand(0);
                let val = op_ref.get_operand(1);

                let insert_op = op_obj
                    .as_ref()
                    .downcast_ref::<ops::InsertValueOp>()
                    .unwrap();
                let indices = insert_op.indices(self.ctx);

                write!(output, "  {res_name} = insertvalue ").unwrap();
                self.export_type(agg.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();
                self.export_value(agg, value_names, output)?;
                write!(output, ", ").unwrap();
                self.export_type(val.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();
                self.export_value(val, value_names, output)?;

                for idx in indices {
                    write!(output, ", {idx}").unwrap();
                }
                writeln!(output).unwrap();
            }

            // --- Address Operations ---
            id if id == ops::AddressOfOp::get_opid_static() => {
                // AddressOfOp gets the address of a global variable.
                // In LLVM IR, this is represented by using the global name directly.
                let address_of = op_obj.as_ref().downcast_ref::<ops::AddressOfOp>().unwrap();
                let global_name = address_of.get_global_name(self.ctx);

                // The result is the global address - just use @global_name
                let res = op_ref.get_result(0);
                value_names.insert(res, format!("@{global_name}"));
            }

            // --- Constants (Virtual) ---
            id if id == ops::ConstantOp::get_opid_static() => {
                let const_op = op_obj.as_ref().downcast_ref::<ops::ConstantOp>().unwrap();
                let val_attr = const_op.get_value(self.ctx);

                let const_str = if let Some(int_attr) = val_attr.downcast_ref::<IntegerAttr>() {
                    // Use APInt's proper decimal string conversion instead of parsing debug format.
                    // The old code parsed debug strings like "APInt { value: 0x4000_0000_0000_u64 }"
                    // by splitting on '_', which broke for values with underscore grouping
                    // (e.g., 1u64 << 46 = 0x4000_0000_0000 would become 0x4000 = 16384).
                    int_attr.value().to_string_unsigned_decimal()
                } else if let Some(fp16_attr) = val_attr.downcast_ref::<FPHalfAttr>() {
                    format_half_literal(fp16_attr.to_bits())
                } else if let Some(fp32_attr) = val_attr.downcast_ref::<FPSingleAttr>() {
                    // Float constant (f32) - format as LLVM float literal
                    let float_val: f32 = fp32_attr.clone().into();
                    format_float_literal(f64::from(float_val))
                } else if let Some(fp64_attr) = val_attr.downcast_ref::<FPDoubleAttr>() {
                    // Double constant (f64) - format as LLVM float literal
                    let float_val: f64 = fp64_attr.clone().into();
                    format_float_literal(float_val)
                } else {
                    "0".to_string() // Fallback
                };

                // Overwrite register name with constant literal
                let res = op_ref.get_result(0);
                value_names.insert(res, const_str);
            }

            // --- Unknown op fallback ---
            _ => {
                writeln!(output, "  ; Unknown op: {op_id}").unwrap();
            }
        }

        Ok(())
    }

    fn export_binop(
        &self,
        op_name: &str,
        op: Ptr<Operation>,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.deref(self.ctx);
        let res = op_ref.get_result(0);
        let lhs = op_ref.get_operand(0);
        let rhs = op_ref.get_operand(1);
        let res_name = value_names.get(&res).unwrap();

        write!(output, "  {res_name} = {op_name} ").unwrap();
        self.export_type(lhs.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(lhs, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_value(rhs, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    /// Export a cast operation: `%res = <op_name> <src_type> <val> to <dst_type>`
    fn export_cast(
        &self,
        op_name: &str,
        op: Ptr<Operation>,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.deref(self.ctx);
        let res = op_ref.get_result(0);
        let val = op_ref.get_operand(0);
        let res_name = value_names.get(&res).unwrap();

        write!(output, "  {res_name} = {op_name} ").unwrap();
        self.export_type(val.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        write!(output, " to ").unwrap();
        self.export_type(res.get_type(self.ctx), output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    fn export_value(
        &self,
        val: Value,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        if let Some(name) = value_names.get(&val) {
            write!(output, "{name}").unwrap();
            Ok(())
        } else {
            write!(output, "undef").unwrap();
            Ok(())
        }
    }

    /// Compute natural alignment (in bytes) for a type.
    /// Used for atomic load/store which require explicit alignment in LLVM IR.
    fn natural_alignment(&self, ty: Ptr<TypeObj>) -> u32 {
        let ty_ref = ty.deref(self.ctx);
        if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
            let width = int_ty.width();
            // Alignment = ceil(width / 8), minimum 1
            std::cmp::max(1, width / 8)
        } else if ty_ref.is::<pliron::builtin::types::FP32Type>() {
            4
        } else if ty_ref.is::<pliron::builtin::types::FP64Type>() {
            8
        } else {
            // Default: 8 bytes (conservative for pointers, etc.)
            8
        }
    }

    /// Compute byte size of an LLVM type. Used for picking an alloca
    /// alignment that won't fault when the storage is later accessed
    /// via a wider load/store than its natural type suggests.
    fn byte_size_of_type(&self, ty: Ptr<TypeObj>) -> u64 {
        let ty_ref = ty.deref(self.ctx);
        if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
            return (int_ty.width() as u64).div_ceil(8);
        }
        if ty_ref.is::<HalfType>() {
            return 2;
        }
        if ty_ref.is::<FP32Type>() {
            return 4;
        }
        if ty_ref.is::<FP64Type>() {
            return 8;
        }
        if ty_ref.is::<PointerType>() {
            return 8;
        }
        if let Some(arr_ty) = ty_ref.downcast_ref::<crate::types::ArrayType>() {
            let elem_ty = arr_ty.elem_type();
            let n = arr_ty.size();
            drop(ty_ref);
            return self.byte_size_of_type(elem_ty) * n;
        }
        if let Some(vec_ty) = ty_ref.downcast_ref::<crate::types::VectorType>() {
            let elem_ty = vec_ty.elem_type();
            let n = vec_ty.size();
            drop(ty_ref);
            return self.byte_size_of_type(elem_ty) * n;
        }
        if let Some(struct_ty) = ty_ref.downcast_ref::<StructType>() {
            let fields: Vec<Ptr<TypeObj>> = struct_ty.fields().collect();
            drop(ty_ref);
            // Sum of field sizes — ignores padding. Safe for our use
            // (over-sizing only over-aligns, which is conservative).
            return fields.iter().map(|f| self.byte_size_of_type(*f)).sum();
        }
        // Unknown: claim 1 so the caller falls back to natural-by-element.
        1
    }

    /// Pick a conservative alignment for a stack alloca of `elem_ty`.
    ///
    /// Rule: `min(16, next_pow2(byte_size))`. NVPTX wide ops cap at
    /// 16-byte natural alignment (`b128`, `v2.b64`, `v4.b32`), so 16
    /// is sufficient. `next_pow2(size)` caps the cost — a 4-byte
    /// alloca only gets aligned to 4, not 16.
    ///
    /// TODO: this is a blunt instrument. The "right" fix is to thread
    /// pointer alignment through mir-lower's load/store emission so we
    /// can attach `align N` metadata to each load/store, letting NVPTX
    /// backend pick the right lowering instead of forcing the storage
    /// to satisfy the worst-case consumer. That's a bigger refactor;
    /// this rule eliminates the bug class as a one-locality fix and
    /// wastes at most ~7 bytes of stack per under-aligned alloca.
    /// See `examples/xoshiro_seed_misalign/` for the surfacing context.
    fn conservative_alloca_align(&self, elem_ty: Ptr<TypeObj>) -> u64 {
        let size = self.byte_size_of_type(elem_ty);
        if size == 0 {
            return 1;
        }
        size.next_power_of_two().min(16)
    }

    fn export_type(&self, ty: Ptr<TypeObj>, output: &mut String) -> Result<(), String> {
        let ty_ref = ty.deref(self.ctx);
        if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
            write!(output, "i{}", int_ty.width()).unwrap();
        } else if let Some(ptr_ty) = ty_ref.downcast_ref::<PointerType>() {
            let addrspace = ptr_ty.address_space();
            if addrspace != 0 {
                write!(output, "ptr addrspace({addrspace})").unwrap();
            } else {
                write!(output, "ptr").unwrap();
            }
        } else if ty_ref.is::<VoidType>() {
            write!(output, "void").unwrap();
        } else if ty_ref.is::<HalfType>() {
            write!(output, "half").unwrap();
        } else if ty_ref.is::<FP32Type>() {
            write!(output, "float").unwrap();
        } else if ty_ref.is::<FP64Type>() {
            write!(output, "double").unwrap();
        } else if let Some(struct_ty) = ty_ref.downcast_ref::<StructType>() {
            write!(output, "{{ ").unwrap();
            for (i, elem_ty) in struct_ty.fields().enumerate() {
                if i > 0 {
                    write!(output, ", ").unwrap();
                }
                self.export_type(elem_ty, output)?;
            }
            write!(output, " }}").unwrap();
        } else if let Some(array_ty) = ty_ref.downcast_ref::<crate::types::ArrayType>() {
            // Format: [N x element_type]
            write!(output, "[{} x ", array_ty.size()).unwrap();
            self.export_type(array_ty.elem_type(), output)?;
            write!(output, "]").unwrap();
        } else if let Some(vec_ty) = ty_ref.downcast_ref::<crate::types::VectorType>() {
            // Format: <N x element_type>
            write!(output, "<{} x ", vec_ty.size()).unwrap();
            self.export_type(vec_ty.elem_type(), output)?;
            write!(output, ">").unwrap();
        } else {
            write!(output, "void /* unknown: {} */", ty_ref.disp(self.ctx)).unwrap();
        }
        Ok(())
    }
}

fn format_half_literal(bits: u16) -> String {
    format!("0xH{bits:04X}")
}

/// Format a float value as an LLVM IR literal.
/// LLVM requires float literals to have a decimal point (e.g., "0.0" not "0").
fn format_float_literal(value: f64) -> String {
    if value.is_nan() {
        "nan".to_string()
    } else if value.is_infinite() {
        if value.is_sign_positive() {
            "0x7FF0000000000000".to_string() // +inf
        } else {
            "0xFFF0000000000000".to_string() // -inf
        }
    } else {
        // Format the float, ensuring it has a decimal point
        let s = format!("{value}");
        if s.contains('.') || s.contains('e') || s.contains('E') {
            s
        } else {
            format!("{s}.0")
        }
    }
}

/// Decode a `[0-9a-fA-F]*` hex string into bytes. Lower-case is what
/// mir-importer emits; this accepts both cases for robustness. Returns
/// an error on odd-length input or non-hex characters.
fn decode_hex_bytes(hex: &str) -> Result<Vec<u8>, String> {
    if hex.len() % 2 != 0 {
        return Err(format!("hex string has odd length {}", hex.len()));
    }
    let bytes = hex.as_bytes();
    let mut out = Vec::with_capacity(hex.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let hi = hex_digit(pair[0])?;
        let lo = hex_digit(pair[1])?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn hex_digit(b: u8) -> Result<u8, String> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(format!("invalid hex digit: {:?}", b as char)),
    }
}

/// Decode `"OFF:KEY,OFF:KEY,..."` into `Vec<(offset, target_key)>`.
/// Returns an error on malformed input. Empty string yields empty Vec.
fn decode_relocations(s: &str) -> Result<Vec<(u64, String)>, String> {
    if s.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in s.split(',') {
        let (off_str, name) = entry
            .split_once(':')
            .ok_or_else(|| format!("relocation entry missing colon: {entry:?}"))?;
        let off: u64 = off_str
            .parse()
            .map_err(|e| format!("invalid offset {off_str:?}: {e}"))?;
        out.push((off, name.to_string()));
    }
    Ok(out)
}
