/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::PathBuf;

/// Explicit backend knobs; replaces every `CUDA_OXIDE_*` env read inside the
/// backend. `run_pipeline` (mir-importer) builds one from the environment at
/// its own boundary. The experimental API builds one from typed compile
/// options without reading the environment.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BackendOptions {
    /// Hard target override (`llc -mcpu=`), e.g. `"sm_120"`.
    pub target_arch: Option<String>,
    /// Human-readable name for whatever set `target_arch`, used only to
    /// describe target provenance in diagnostics and errors (e.g.
    /// `"CUDA_OXIDE_TARGET"` for the env-driven rustc pipeline, or a
    /// caller-facing description for the standalone experimental API).
    ///
    /// Keep this in step with `target_arch`: whoever writes one writes the
    /// other, or a target error names a source the caller never used.
    pub target_arch_source: &'static str,
    /// Advisory local-GPU arch; used only when it satisfies detected features.
    pub device_arch_hint: Option<String>,
    /// Skip the `opt -O2` middle-end.
    pub no_opt: bool,
    /// Suppress `llc -fp-contract=fast` (fmul+fadd fusion to fma).
    pub no_fma: bool,
    /// Print progress and tool-selection notes to stderr.
    pub verbose: bool,
    /// Explicit `llc` binary (was `CUDA_OXIDE_LLC`).
    pub llc_override: Option<PathBuf>,
    /// Explicit `opt` binary (was `CUDA_OXIDE_OPT`).
    pub opt_override: Option<PathBuf>,
}

impl Default for BackendOptions {
    fn default() -> Self {
        Self {
            target_arch: None,
            target_arch_source: "CUDA_OXIDE_TARGET",
            device_arch_hint: None,
            no_opt: false,
            no_fma: false,
            verbose: false,
            llc_override: None,
            opt_override: None,
        }
    }
}

impl BackendOptions {
    /// Reads the historical `CUDA_OXIDE_*` variables. The ONLY env access in
    /// this crate outside this crate's own tests; called by rustc-pipeline
    /// hosts, never by the backend itself.
    pub fn from_env() -> Self {
        Self {
            target_arch: std::env::var("CUDA_OXIDE_TARGET").ok(),
            target_arch_source: "CUDA_OXIDE_TARGET",
            device_arch_hint: std::env::var("CUDA_OXIDE_DEVICE_ARCH").ok(),
            no_opt: std::env::var("CUDA_OXIDE_NO_OPT").is_ok(),
            no_fma: std::env::var("CUDA_OXIDE_NO_FMA").is_ok(),
            verbose: std::env::var("CUDA_OXIDE_VERBOSE").is_ok(),
            llc_override: std::env::var("CUDA_OXIDE_LLC").ok().map(PathBuf::from),
            opt_override: std::env::var("CUDA_OXIDE_OPT").ok().map(PathBuf::from),
        }
    }
}
