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
    metadata::{emit_nvvm_annotations, emit_nvvmir_version, needs_nvvm_annotations},
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
    let mut state = ModuleExportState::new(
        ctx,
        emit_all_annotations,
        emit_ptx_kernel_keyword,
        config.debug_kind(),
    );

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

    // 5. Debug intrinsic declarations used by full-debug local variables.
    if state.debug_declare_used {
        writeln!(&mut output).unwrap();
        state.emit_debug_intrinsic_declarations(&mut output);
    }

    // 6. Emit attribute groups for convergent intrinsics used by module functions
    // Note: Device extern declarations no longer get attribute groups - see section 2 comment.
    if state.convergent_used {
        writeln!(&mut output).unwrap();
        writeln!(&mut output, "attributes #0 = {{ convergent }}").unwrap();
    }

    // 7. nvvm.annotations metadata
    if needs_nvvm_annotations(&state, emit_all_annotations) {
        writeln!(&mut output).unwrap();
        emit_nvvm_annotations(&mut output, &mut state, emit_all_annotations);
    }

    // 8. nvvmir.version metadata (if backend requires)
    if config.emit_nvvmir_version() {
        writeln!(&mut output).unwrap();
        emit_nvvmir_version(&mut output, &mut state, config.nvvmir_version());
    }

    // 9. DWARF metadata (if requested and source locations exist)
    if state.has_debug_metadata() {
        writeln!(&mut output).unwrap();
        state.emit_debug_metadata(&mut output);
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
    let mut state = ModuleExportState::new(
        ctx,
        emit_all_annotations,
        emit_ptx_kernel_keyword,
        config.debug_kind(),
    );

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

    // Emit debug intrinsic declarations used by full-debug local variables.
    if state.debug_declare_used {
        writeln!(&mut output).unwrap();
        state.emit_debug_intrinsic_declarations(&mut output);
    }

    // Emit attributes section if convergent operations were used
    if state.convergent_used {
        writeln!(&mut output).unwrap();
        writeln!(&mut output, "attributes #0 = {{ convergent }}").unwrap();
    }

    // Emit nvvm.annotations metadata
    // - Default: Only for kernels with cluster configuration or launch bounds
    // - Alternate backends: May require annotations for ALL kernels
    if needs_nvvm_annotations(&state, emit_all_annotations) {
        writeln!(&mut output).unwrap();
        emit_nvvm_annotations(&mut output, &mut state, emit_all_annotations);
    }

    // Emit !nvvmir.version metadata if backend requests it
    if config.emit_nvvmir_version() {
        writeln!(&mut output).unwrap();
        emit_nvvmir_version(&mut output, &mut state, config.nvvmir_version());
    }

    // Emit DWARF line-table metadata if debug export requested it and at least
    // one function had a real source location.
    if state.has_debug_metadata() {
        writeln!(&mut output).unwrap();
        state.emit_debug_metadata(&mut output);
    }

    Ok(output)
}
