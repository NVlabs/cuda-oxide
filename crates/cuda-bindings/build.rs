/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::{env, error::Error, path::Path, path::PathBuf, process::exit};

/// Returns the CUDA toolkit install root.
///
/// Resolution order:
///   1. `CUDA_TOOLKIT_PATH` (primary)
///   2. `CUDA_HOME`
///   3. `CUDA_PATH`
///   4. `CUDA_ROOT`
///   5. `/usr/local/cuda`
///
/// Most environment-management tools (nix, conda, modules) set
/// `CUDA_HOME` / `CUDA_PATH` rather than `CUDA_TOOLKIT_PATH`; falling
/// back through the common aliases keeps detection working without
/// asking the user to export an extra variable.
fn cuda_toolkit_dir() -> String {
    const CANDIDATES: &[&str] = &["CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"];
    for name in CANDIDATES {
        if let Ok(v) = env::var(name) {
            if !v.is_empty() {
                return v;
            }
        }
    }
    "/usr/local/cuda".to_string()
}

/// Runs [`run`]; on error, prints the message and exits with status 1.
fn main() {
    if let Err(error) = run() {
        eprintln!("{}", error);
        exit(1);
    }
}

/// Configures the crate build: declares rerun triggers, adds native link search paths for `libcuda`,
/// links `cuda`, and invokes bindgen on `wrapper.h` with `-I{toolkit}/include`, writing
/// `bindings.rs` into `OUT_DIR`.
fn run() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo::rustc-check-cfg=cfg(cuda_has_cuEventElapsedTime_v2)");

    let toolkit = cuda_toolkit_dir();
    let cuda_h = Path::new(&toolkit).join("include/cuda.h");
    println!("cargo:rerun-if-changed={}", cuda_h.display());

    match std::fs::read_to_string(&cuda_h) {
        Ok(contents) => {
            if contents.contains("cuEventElapsedTime_v2") {
                println!("cargo:rustc-cfg=cuda_has_cuEventElapsedTime_v2");
            }
        }
        Err(err) => {
            println!(
                "cargo:warning=cuda-bindings: Could not read cuda.h at {}: {}",
                cuda_h.display(),
                err
            );
        }
    }

    for path in collect_lib_paths(&toolkit) {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
    println!("cargo:rustc-link-lib=dylib=cuda");

    bindgen::builder()
        .header("wrapper.h")
        .clang_arg(format!("-I{}/include", toolkit))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // CUDA 13.2+ adds types to CUlaunchAttributeValue that bindgen/libclang
        // cannot translate, collapsing the struct to a 1-byte opaque blob while the
        // size assertion still expects the real C size. Making both the struct and its
        // inner union opaque produces correctly-sized byte blobs across CUDA versions.
        // launch_kernel_ex in cuda-core constructs this struct via raw pointer writes.
        .opaque_type("CUlaunchAttribute_st")
        .opaque_type("CUlaunchAttributeValue_union")
        .generate()
        .expect("Unable to generate CUDA bindings")
        .write_to_file(Path::new(&env::var("OUT_DIR")?).join("bindings.rs"))?;

    Ok(())
}

/// Candidate directories for `rustc-link-search=native` when linking against the driver library.
///
/// Probes (in order):
/// - `{toolkit}/lib64{,/stubs}` — classic x86_64 CUDA toolkit layout
/// - `{toolkit}/lib{,/stubs}` — aarch64 / nix-style CUDA toolkit layout where
///   the driver library lives in plain `lib/`
/// - `{toolkit}/targets/<arch>-linux/lib{,/stubs}` — redistributable / cross
///   layout. `<arch>` follows `CARGO_CFG_TARGET_ARCH` (x86_64 / aarch64) and
///   defaults to the host's arch when unset.
///
/// Only directories that actually exist are emitted. Order is preserved;
/// duplicates are not filtered.
fn collect_lib_paths(toolkit: &str) -> Vec<PathBuf> {
    let base = PathBuf::from(toolkit);
    let mut paths = vec![];

    let lib64 = base.join("lib64");
    if lib64.is_dir() {
        paths.push(lib64.clone());
        let stubs = lib64.join("stubs");
        if stubs.is_dir() {
            paths.push(stubs);
        }
    }

    let lib = base.join("lib");
    if lib.is_dir() {
        paths.push(lib.clone());
        let stubs = lib.join("stubs");
        if stubs.is_dir() {
            paths.push(stubs);
        }
    }

    for target_dir in target_layout_dirs(&base) {
        if target_dir.join("include/cuda.h").is_file() || target_dir.join("lib").is_dir() {
            let lib = target_dir.join("lib");
            if lib.is_dir() {
                paths.push(lib.clone());
            }
            let stubs = lib.join("stubs");
            if stubs.is_dir() {
                paths.push(stubs);
            }
        }
    }

    paths
}

/// Returns the `{toolkit}/targets/<arch>-linux/` directories to probe.
///
/// Picks the arch from `CARGO_CFG_TARGET_ARCH` (set by cargo for build
/// scripts); falls back to probing both common host arches when the env
/// var is unset. `aarch64` also probes `sbsa-linux` because the SBSA
/// redist archives use that name.
fn target_layout_dirs(base: &Path) -> Vec<PathBuf> {
    let arches: &[&str] = match env::var("CARGO_CFG_TARGET_ARCH").ok().as_deref() {
        Some("x86_64") => &["x86_64"],
        Some("aarch64") => &["aarch64", "sbsa"],
        _ => &["x86_64", "aarch64", "sbsa"],
    };
    arches
        .iter()
        .map(|a| base.join(format!("targets/{}-linux", a)))
        .collect()
}
