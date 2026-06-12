/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! NVVM annotations, version metadata, and llvm.used emission.

use std::fmt::Write;

use super::state::ModuleExportState;

/// Emit `!nvvm.annotations` metadata nodes for kernels.
///
/// Used by the `export_module_with_externs` path.
pub(super) fn emit_nvvm_annotations(
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

/// Calculate the next metadata ID after annotations (for `!nvvmir.version`).
pub(super) fn md_id_after_annotations(state: &ModuleExportState) -> usize {
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
