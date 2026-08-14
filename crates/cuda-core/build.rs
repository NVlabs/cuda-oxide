/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Probes the CUDA headers for the multicast driver API (CUDA 12.1+).
//!
//! The `cuMulticast*` entry points first appeared in CUDA 12.1, and
//! `cuda-bindings` binds whatever the host `cuda.h` declares, so building
//! against a CUDA 12.0 toolkit would otherwise fail to compile all of
//! `cuda-core`. The `cuda_has_multicast` cfg gates the multicast surface of
//! `vmm` to toolkits that declare the API, mirroring the
//! `cuda_has_cuEventElapsedTime_v2` probe in `cuda-bindings`.
//!
//! Toolkit discovery matches `cuda-bindings/build.rs`: the first set
//! variable among `CUDA_TOOLKIT_PATH` and `CUDA_HOME`, else
//! `/usr/local/cuda`, with both the standard `include/` and the
//! redistributable `targets/<dir>/include/` layouts probed. A missing or
//! unreadable `cuda.h` leaves the cfg unset (multicast unavailable) rather
//! than erroring; `cuda-bindings` reports the authoritative failure for a
//! genuinely broken toolkit.

use std::env;
use std::path::{Path, PathBuf};

const TOOLKIT_ENV_VARS: &[&str] = &["CUDA_TOOLKIT_PATH", "CUDA_HOME"];
const DEFAULT_TOOLKIT_DIR: &str = "/usr/local/cuda";

/// Overrides the `targets/<dir>` selection with a single directory name,
/// like nvcc's `-target-dir`; matches `cuda-bindings`.
const TOOLKIT_TARGET_DIR_ENV: &str = "CUDA_TOOLKIT_TARGET_DIR";

fn main() {
    println!("cargo::rustc-check-cfg=cfg(cuda_has_multicast)");
    for var in TOOLKIT_ENV_VARS {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rerun-if-env-changed={TOOLKIT_TARGET_DIR_ENV}");

    let Some(cuda_h) = find_cuda_header() else {
        return;
    };
    println!("cargo:rerun-if-changed={}", cuda_h.display());
    if std::fs::read_to_string(&cuda_h).is_ok_and(|header| header.contains("cuMulticastCreate")) {
        println!("cargo:rustc-cfg=cuda_has_multicast");
    }
}

/// CUDA toolkit `targets/` directory names to probe for cargo's build
/// target, most specific first.
///
/// Kept in lockstep BY HAND with the selection table in
/// `crates/cuda-bindings/toolkit_target.rs` (`resolve_toolkit_target_dirs`):
/// build scripts cannot import each other's sources across crates. If the
/// selection there changes, mirror it here. CUDA names these layouts after
/// the GPU platform, not the Rust triple, and an aarch64 Linux triple is
/// ambiguous between servers (`sbsa-linux`) and Tegra (`aarch64-linux`), so
/// both are probed in that order. A non-blank [`TOOLKIT_TARGET_DIR_ENV`]
/// replaces the table with that single directory.
fn toolkit_target_dirs() -> Vec<String> {
    if let Some(dir) = env::var(TOOLKIT_TARGET_DIR_ENV)
        .ok()
        .filter(|dir| !dir.trim().is_empty())
    {
        return vec![dir];
    }
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if os != "linux" {
        return vec![];
    }
    match arch.as_str() {
        "x86_64" => vec!["x86_64-linux".to_string()],
        "aarch64" => vec!["sbsa-linux".to_string(), "aarch64-linux".to_string()],
        _ => vec![],
    }
}

/// Returns the path of `cuda.h`: `{toolkit}/include` for standard installs,
/// or `{toolkit}/targets/<dir>/include` for redistributable layouts.
fn find_cuda_header() -> Option<PathBuf> {
    let toolkit = TOOLKIT_ENV_VARS
        .iter()
        .find_map(|var| env::var(var).ok())
        .unwrap_or_else(|| DEFAULT_TOOLKIT_DIR.to_string());
    let base = Path::new(&toolkit);
    let mut candidates = vec![base.join("include")];
    for target_dir in toolkit_target_dirs() {
        candidates.push(base.join("targets").join(target_dir).join("include"));
    }
    candidates
        .into_iter()
        .map(|dir| dir.join("cuda.h"))
        .find(|header| header.is_file())
}
