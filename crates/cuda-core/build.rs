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

fn main() {
    println!("cargo::rustc-check-cfg=cfg(cuda_has_multicast)");
    for var in TOOLKIT_ENV_VARS {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let Some(cuda_h) = find_cuda_header() else {
        return;
    };
    println!("cargo:rerun-if-changed={}", cuda_h.display());
    if std::fs::read_to_string(&cuda_h).is_ok_and(|header| header.contains("cuMulticastCreate")) {
        println!("cargo:rustc-cfg=cuda_has_multicast");
    }
}

/// CUDA toolkit `targets/` directory name for cargo's build `TARGET`,
/// matching `cuda-bindings`: CUDA names these layouts after the GPU
/// platform, not the Rust triple.
fn toolkit_target_dir() -> Option<&'static str> {
    let target = env::var("TARGET").ok()?;
    match target.split('-').next()? {
        "x86_64" => Some("x86_64-linux"),
        "aarch64" => Some("sbsa-linux"),
        _ => None,
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
    if let Some(target_dir) = toolkit_target_dir() {
        candidates.push(base.join("targets").join(target_dir).join("include"));
    }
    candidates
        .into_iter()
        .map(|dir| dir.join("cuda.h"))
        .find(|header| header.is_file())
}
