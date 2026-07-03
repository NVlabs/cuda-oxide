/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::path::PathBuf;

/// Configuration for [`compile_to_ptx`](crate::compile_to_ptx).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PtxConfig {
    /// Target arch string passed to `llc -mcpu=`, e.g. `"sm_120"`.
    pub target_arch: String,
    /// Run `opt -O2` on the exported `.ll` before `llc`.
    pub optimize: bool,
    /// Allow `fmul`+`fadd` contraction to `fma` (`-fp-contract=fast`).
    pub fma: bool,
    /// Emit debug info.
    pub debug: bool,
}

impl PtxConfig {
    /// Default config for `arch`: optimised, fma on, no debug.
    pub fn new(arch: impl Into<String>) -> Self {
        Self {
            target_arch: arch.into(),
            optimize: true,
            fma: true,
            debug: false,
        }
    }
}

/// Explicit backend knobs; replaces every `CUDA_OXIDE_*` env read inside the
/// backend. `run_pipeline` (mir-importer) builds one from the environment at
/// its own boundary; [`compile_to_ptx`](crate::compile_to_ptx)
/// builds one from [`PtxConfig`].
///
/// This type is host-facing and is deliberately not re-exported at the crate
/// root; frontends use [`PtxConfig`] instead.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct BackendOptions {
    /// Hard target override (`llc -mcpu=`), e.g. `"sm_120"`.
    pub target_arch: Option<String>,
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

impl BackendOptions {
    /// Reads the historical `CUDA_OXIDE_*` variables. The ONLY env access in
    /// this crate outside this crate's own tests; called by rustc-pipeline
    /// hosts, never by the backend itself.
    pub fn from_env() -> Self {
        Self {
            target_arch: std::env::var("CUDA_OXIDE_TARGET").ok(),
            device_arch_hint: std::env::var("CUDA_OXIDE_DEVICE_ARCH").ok(),
            no_opt: std::env::var("CUDA_OXIDE_NO_OPT").is_ok(),
            no_fma: std::env::var("CUDA_OXIDE_NO_FMA").is_ok(),
            verbose: std::env::var("CUDA_OXIDE_VERBOSE").is_ok(),
            llc_override: std::env::var("CUDA_OXIDE_LLC").ok().map(PathBuf::from),
            opt_override: std::env::var("CUDA_OXIDE_OPT").ok().map(PathBuf::from),
        }
    }

    /// Options for a single [`compile_to_ptx`](crate::compile_to_ptx)
    /// call; no env access.
    ///
    /// This is the frontend entry point, and it is deliberately env-knob-free:
    /// no verbose/llc/opt override is reachable through [`PtxConfig`].
    pub fn from_ptx_config(cfg: &PtxConfig) -> Self {
        Self {
            target_arch: Some(cfg.target_arch.trim().to_string()),
            device_arch_hint: None,
            no_opt: !cfg.optimize,
            no_fma: !cfg.fma,
            verbose: false,
            llc_override: None,
            opt_override: None,
        }
    }
}
