/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Module-level export flow.

use std::fmt::Write;

use pliron::{
    builtin::{
        op_interfaces::{OneRegionInterface, SymbolOpInterface},
        ops::ModuleOp,
    },
    context::Context,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
};

use crate::ops;

use super::{
    config::ExportBackendConfig,
    externs::DeviceExternDecl,
    metadata::{emit_nvvm_annotations, md_id_after_annotations},
    state::ModuleExportState,
};

/// Internal implementation of module export with device externs.
pub(super) fn export_module_with_externs_impl(
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
            if let Some(func) = Operation::get_op::<ops::FuncOp>(op, ctx) {
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

    // 6. nvvm.annotations metadata
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
pub(super) fn export_module_to_string_with_config(
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
            if let Some(func) = Operation::get_op::<ops::FuncOp>(op, ctx) {
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
