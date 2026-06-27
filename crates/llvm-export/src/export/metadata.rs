/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! NVVM annotations and version metadata.

use std::collections::HashSet;
use std::fmt::Write;

use super::state::ModuleExportState;

pub(super) fn needs_nvvm_annotations(
    state: &ModuleExportState,
    emit_all_annotations: bool,
) -> bool {
    let has_special_kernels =
        !state.cluster_kernels.is_empty() || !state.launch_bounds_kernels.is_empty();
    has_special_kernels || (emit_all_annotations && !state.all_kernels.is_empty())
}

/// Emit `!nvvm.annotations` metadata nodes for kernels.
///
/// Metadata IDs come from `ModuleExportState` because `!nvvm.annotations`,
/// `!nvvmir.version`, and future debug-info nodes all share LLVM's flat
/// module metadata namespace.
pub(super) fn emit_nvvm_annotations(
    output: &mut String,
    state: &mut ModuleExportState,
    emit_all_annotations: bool,
) -> Result<(), String> {
    let mut metadata_refs = Vec::new();

    // Collect names of kernels that have special configs
    let special_kernel_names: HashSet<String> = state
        .cluster_kernels
        .iter()
        .map(|k| k.name.clone())
        .chain(state.launch_bounds_kernels.iter().map(|k| k.name.clone()))
        .collect();

    // Emit basic annotation for kernels WITHOUT special configs
    if emit_all_annotations {
        let basic_kernels: Vec<String> = state
            .all_kernels
            .iter()
            .filter(|kernel| !special_kernel_names.contains(&kernel.name))
            .map(|kernel| kernel.name.clone())
            .collect();

        for kernel_name in basic_kernels {
            let md_id = state.alloc_metadata_id();
            write!(output, "!{md_id} = !{{").unwrap();
            emit_function_reference(output, state, &kernel_name)?;
            writeln!(output, ", !\"kernel\", i32 1}}").unwrap();
            metadata_refs.push(format!("!{}", md_id));
        }
    }

    // Emit cluster config annotations
    let cluster_kernels: Vec<_> = state
        .cluster_kernels
        .iter()
        .map(|cfg| (cfg.name.clone(), cfg.dim_x, cfg.dim_y, cfg.dim_z))
        .collect();

    for (name, dim_x, dim_y, dim_z) in cluster_kernels {
        let md_id = state.alloc_metadata_id();
        write!(output, "!{md_id} = !{{").unwrap();
        emit_function_reference(output, state, &name)?;
        writeln!(
            output,
            ", !\"kernel\", i32 1, !\"cluster_dim_x\", i32 {dim_x}, !\"cluster_dim_y\", i32 {dim_y}, !\"cluster_dim_z\", i32 {dim_z}}}"
        )
        .unwrap();
        metadata_refs.push(format!("!{}", md_id));
    }

    // Emit launch bounds annotations
    let launch_bounds_kernels: Vec<_> = state
        .launch_bounds_kernels
        .iter()
        .map(|bounds| (bounds.name.clone(), bounds.max_threads, bounds.min_blocks))
        .collect();

    for (name, max_threads, min_blocks) in launch_bounds_kernels {
        for (key, value) in [("maxntidx", max_threads), ("maxntidy", 1), ("maxntidz", 1)] {
            let md_id = state.alloc_metadata_id();
            write!(output, "!{md_id} = !{{").unwrap();
            emit_function_reference(output, state, &name)?;
            writeln!(output, ", !\"{key}\", i32 {value}}}").unwrap();
            metadata_refs.push(format!("!{}", md_id));
        }

        if let Some(min_blocks) = min_blocks {
            let md_id = state.alloc_metadata_id();
            write!(output, "!{md_id} = !{{").unwrap();
            emit_function_reference(output, state, &name)?;
            writeln!(output, ", !\"minctasm\", i32 {min_blocks}}}").unwrap();
            metadata_refs.push(format!("!{}", md_id));
        }
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
    Ok(())
}

fn emit_function_reference(
    output: &mut String,
    state: &ModuleExportState<'_>,
    name: &str,
) -> Result<(), String> {
    if state.legacy_typed_pointers() {
        state.export_function_pointer_type(state.function_type(name)?, output)?;
    } else {
        write!(output, "ptr").unwrap();
    }
    write!(output, " @{name}").unwrap();
    Ok(())
}

pub(super) fn emit_nvvmir_version(
    output: &mut String,
    state: &mut ModuleExportState,
    version: [i32; 4],
) {
    let md_id = state.alloc_metadata_id();
    writeln!(output, "!nvvmir.version = !{{!{}}}", md_id).unwrap();
    writeln!(
        output,
        "!{} = !{{i32 {}, i32 {}, i32 {}, i32 {}}}",
        md_id, version[0], version[1], version[2], version[3]
    )
    .unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::config::DebugKind;
    use crate::export::state::{
        KernelClusterConfig, KernelInfo, KernelLaunchBounds, ModuleExportState,
    };
    use pliron::context::Context;

    fn test_state<'a>(ctx: &'a Context) -> ModuleExportState<'a> {
        ModuleExportState::new(ctx, true, false, DebugKind::Off, None)
    }

    #[test]
    fn allocator_returns_contiguous_module_metadata_ids() {
        let ctx = Context::new();
        let mut state = test_state(&ctx);

        assert_eq!(state.alloc_metadata_id(), 0);
        assert_eq!(state.alloc_metadata_id(), 1);
        assert_eq!(state.alloc_metadata_id(), 2);
        assert_eq!(state.next_metadata_id(), 3);
    }

    #[test]
    fn nvvm_metadata_uses_one_shared_allocator() {
        let ctx = Context::new();
        let mut state = test_state(&ctx);
        state.all_kernels.push(KernelInfo {
            name: "plain".into(),
        });
        state.all_kernels.push(KernelInfo {
            name: "clustered".into(),
        });
        state.all_kernels.push(KernelInfo {
            name: "bounded".into(),
        });
        state.cluster_kernels.push(KernelClusterConfig {
            name: "clustered".into(),
            dim_x: 2,
            dim_y: 3,
            dim_z: 4,
        });
        state.launch_bounds_kernels.push(KernelLaunchBounds {
            name: "bounded".into(),
            max_threads: 256,
            min_blocks: Some(2),
        });

        let mut output = String::new();
        emit_nvvm_annotations(&mut output, &mut state, true).unwrap();
        emit_nvvmir_version(&mut output, &mut state, [2, 0, 3, 2]);

        assert_eq!(
            output,
            concat!(
                "!0 = !{ptr @plain, !\"kernel\", i32 1}\n",
                "!1 = !{ptr @clustered, !\"kernel\", i32 1, !\"cluster_dim_x\", i32 2, !\"cluster_dim_y\", i32 3, !\"cluster_dim_z\", i32 4}\n",
                "!2 = !{ptr @bounded, !\"maxntidx\", i32 256}\n",
                "!3 = !{ptr @bounded, !\"maxntidy\", i32 1}\n",
                "!4 = !{ptr @bounded, !\"maxntidz\", i32 1}\n",
                "!5 = !{ptr @bounded, !\"minctasm\", i32 2}\n",
                "!nvvm.annotations = !{!0, !1, !2, !3, !4, !5}\n",
                "!nvvmir.version = !{!6}\n",
                "!6 = !{i32 2, i32 0, i32 3, i32 2}\n",
            )
        );
        assert_eq!(state.next_metadata_id(), 7);
    }

    #[test]
    fn basic_kernel_annotations_are_skipped_when_backend_does_not_need_them() {
        let ctx = Context::new();
        let mut state = test_state(&ctx);
        state.all_kernels.push(KernelInfo {
            name: "plain".into(),
        });

        let mut output = String::new();
        emit_nvvm_annotations(&mut output, &mut state, false).unwrap();

        assert!(output.is_empty());
        assert_eq!(state.next_metadata_id(), 0);
    }
}
