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
        op_interfaces::{OneRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
    },
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    value::Value,
};
use std::collections::HashMap;
use std::fmt::Write;

use crate::ops::{self, FuncOp};

mod block;
mod config;
mod externs;
mod function;
mod literals;
mod metadata;
mod names;
mod op_emission;
mod type_printing;
mod values;

pub use config::{ExportBackendConfig, NvvmExportConfig, PtxExportConfig};
pub use externs::{AsDeviceExtern, DeviceExternAttrs, DeviceExternDecl};
use literals::{format_float_literal, format_half_literal};
use metadata::{emit_nvvm_annotations, md_id_after_annotations};
use names::{has_device_prefix, strip_device_prefix};

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
        } else {
            // Internal linkage: static storage in the global's address space.
            write!(output, "@{name} = addrspace({address_space}) global ").unwrap();
            self.export_type(ty, output)?;
            writeln!(output, " zeroinitializer, align {alignment}").unwrap();
        }

        Ok(())
    }
}
