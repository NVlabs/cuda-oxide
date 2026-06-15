/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Exporter state and kernel bookkeeping.

use pliron::{basic_block::BasicBlock, context::Ptr, value::Value};
use std::{collections::HashMap, path::PathBuf};

use super::config::DebugKind;

/// Map from block to its predecessors with the values passed to each predecessor.
/// Used for PHI node generation when exporting to LLVM IR.
pub(super) type PredecessorMap = HashMap<Ptr<BasicBlock>, Vec<(Ptr<BasicBlock>, Vec<Value>)>>;

/// Cluster dimensions for a kernel (from `#[cluster(x,y,z)]` attribute).
pub(super) struct KernelClusterConfig {
    pub(super) name: String,
    pub(super) dim_x: u32,
    pub(super) dim_y: u32,
    pub(super) dim_z: u32,
}

/// Launch bounds for a kernel (from `#[launch_bounds(max, min)]` attribute).
pub(super) struct KernelLaunchBounds {
    pub(super) name: String,
    pub(super) max_threads: u32,
    pub(super) min_blocks: Option<u32>, // None if not specified (0 in attribute)
}

/// Basic kernel info (for backends that need annotations for all kernels).
pub(super) struct KernelInfo {
    pub(super) name: String,
}

pub(super) struct ModuleExportState<'a> {
    pub(super) ctx: &'a pliron::context::Context,
    /// Track if any convergent operations were used (for emitting attributes section)
    pub(super) convergent_used: bool,
    /// Track kernels with cluster configurations for nvvm.annotations metadata
    pub(super) cluster_kernels: Vec<KernelClusterConfig>,
    /// Track kernels with launch bounds for nvvm.annotations metadata
    pub(super) launch_bounds_kernels: Vec<KernelLaunchBounds>,
    /// Track ALL kernels (for backends that require annotations for every kernel)
    pub(super) all_kernels: Vec<KernelInfo>,
    /// Whether to track all kernels (set by backend config)
    pub(super) track_all_kernels: bool,
    /// Whether to print `ptx_kernel` on kernel definitions.
    pub(super) emit_ptx_kernel_keyword: bool,
    /// Track device function names for @llvm.used (standalone device fn compilation)
    pub(super) device_functions: Vec<String>,
    /// Next `!N` metadata ID in this module.
    ///
    /// LLVM has one flat numbered metadata namespace per module. Today this is
    /// used for NVVM annotations/version nodes; debug-info nodes will use the
    /// same counter so the exporter never has to guess which IDs are free.
    next_metadata_id: usize,
    /// Which debug metadata tier this export should emit.
    pub(super) debug_kind: DebugKind,
    /// The single compile unit used for Stage 2 line-table debug info.
    pub(super) debug_compile_unit: Option<usize>,
    /// `DIFile` nodes keyed by the source path they describe.
    pub(super) debug_files: HashMap<PathBuf, usize>,
    /// Shared empty function type used by all line-table-only subprograms.
    pub(super) debug_subroutine_type: Option<usize>,
    /// `DISubprogram` file paths, used to avoid attaching locations to the wrong scope.
    pub(super) debug_subprogram_files: HashMap<usize, PathBuf>,
    /// Fallback line/column for calls that LLVM requires to have a location.
    pub(super) debug_subprogram_fallbacks: HashMap<usize, (i32, i32)>,
    /// `DILocation` nodes keyed by `(scope, line, column)`.
    pub(super) debug_locations: HashMap<(usize, i32, i32), usize>,
    /// Numbered debug metadata definitions, in allocation order.
    pub(super) debug_nodes: Vec<(usize, String)>,
}

impl<'a> ModuleExportState<'a> {
    pub(super) fn new(
        ctx: &'a pliron::context::Context,
        track_all_kernels: bool,
        emit_ptx_kernel_keyword: bool,
        debug_kind: DebugKind,
    ) -> Self {
        Self {
            ctx,
            convergent_used: false,
            cluster_kernels: Vec::new(),
            launch_bounds_kernels: Vec::new(),
            all_kernels: Vec::new(),
            track_all_kernels,
            emit_ptx_kernel_keyword,
            device_functions: Vec::new(),
            next_metadata_id: 0,
            debug_kind,
            debug_compile_unit: None,
            debug_files: HashMap::new(),
            debug_subroutine_type: None,
            debug_subprogram_files: HashMap::new(),
            debug_subprogram_fallbacks: HashMap::new(),
            debug_locations: HashMap::new(),
            debug_nodes: Vec::new(),
        }
    }

    pub(super) fn alloc_metadata_id(&mut self) -> usize {
        let id = self.next_metadata_id;
        self.next_metadata_id += 1;
        id
    }

    #[cfg(test)]
    pub(super) fn next_metadata_id(&self) -> usize {
        self.next_metadata_id
    }

    /// Check if a function name is a known convergent intrinsic.
    ///
    /// These intrinsics require warp-synchronous execution semantics and must
    /// be marked convergent to prevent LLVM from applying optimizations that
    /// would break GPU synchronization (like duplicating them into divergent branches).
    pub(super) fn is_convergent_intrinsic(name: &str) -> bool {
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
}
