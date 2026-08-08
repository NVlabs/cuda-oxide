/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::{env, error::Error, path::Path, path::PathBuf, process::exit};

/// Environment variables consulted (in order) to locate the CUDA toolkit root.
/// `CUDA_HOME` is the conventional name used by nvcc wrappers and CI images.
const TOOLKIT_ENV_VARS: [&str; 2] = ["CUDA_TOOLKIT_PATH", "CUDA_HOME"];

/// Toolkit root fallback when none of [`TOOLKIT_ENV_VARS`] is set.
const DEFAULT_TOOLKIT_DIR: &str = "/usr/local/cuda";

/// Environment variable naming the CUDA `targets/<dir>` tree to use,
/// overriding the arch+OS table in `toolkit_target.rs`. The value is a
/// directory name under `{toolkit}/targets/` (e.g. `aarch64-linux`), the
/// same shape nvcc's `-target-dir` flag takes; CMake exposes the equivalent
/// selection as `CUDAToolkit_TARGET_DIR`. The named directory is still
/// probed for `cuda.h`, so a wrong value fails with the clear discovery
/// error instead of silently falling back to the table.
const TOOLKIT_TARGET_DIR_ENV: &str = "CUDA_TOOLKIT_TARGET_DIR";

/// Returns the CUDA toolkit install root: the first set variable among
/// [`TOOLKIT_ENV_VARS`], otherwise [`DEFAULT_TOOLKIT_DIR`]. Used for include
/// paths, library search paths, and bindgen’s Clang configuration.
fn cuda_toolkit_dir() -> String {
    TOOLKIT_ENV_VARS
        .iter()
        .find_map(|var| env::var(var).ok())
        .unwrap_or_else(|| DEFAULT_TOOLKIT_DIR.to_string())
}

/// Runs [`run`]; on error, prints the message and exits with status 1.
fn main() {
    if let Err(error) = run() {
        eprintln!("{}", error);
        exit(1);
    }
}

/// Configures the crate build: declares rerun triggers, discovers the CUDA
/// include directory, adds native link search paths for `libcuda`, links
/// `cuda`, and invokes bindgen on `wrapper.h` with the discovered include
/// directory, writing `bindings.rs` into `OUT_DIR`.
fn run() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-changed=wrapper.h");
    // Emitting any rerun-if-changed disables cargo's default "rerun on any
    // package change", so the `include!`d selection table needs naming.
    println!("cargo:rerun-if-changed=toolkit_target.rs");
    for var in TOOLKIT_ENV_VARS {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rerun-if-env-changed={TOOLKIT_TARGET_DIR_ENV}");
    println!("cargo::rustc-check-cfg=cfg(cuda_has_cuEventElapsedTime_v2)");

    let toolkit = cuda_toolkit_dir();
    let include_dir = find_cuda_include_dir(&toolkit)?;
    probe_event_elapsed_time_v2(&include_dir.join("cuda.h"));

    for path in collect_lib_paths(&toolkit) {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
    println!("cargo:rustc-link-lib=dylib=cuda");

    bindgen::builder()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", include_dir.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // CUDA 13.2+ adds types to CUlaunchAttributeValue that bindgen/libclang
        // cannot translate, collapsing the struct to a 1-byte opaque blob while the
        // size assertion still expects the real C size. Making both the struct and its
        // inner union opaque produces correctly-sized byte blobs across CUDA versions.
        // launch_kernel_ex in cuda-core constructs this struct via raw pointer writes.
        .opaque_type("CUlaunchAttribute_st")
        .opaque_type("CUlaunchAttributeValue_union")
        .generate()
        .map_err(|error| format!("cuda-bindings: failed to generate CUDA bindings: {error}"))?
        .write_to_file(Path::new(&env::var("OUT_DIR")?).join("bindings.rs"))?;

    Ok(())
}

// The `targets/<dir>` selection table, shared verbatim with
// `tests/toolkit_target.rs` so it can be unit tested.
include!("toolkit_target.rs");

/// The `targets/<dir>` candidates for the target cargo is building for: the
/// [`TOOLKIT_TARGET_DIR_ENV`] override when set, otherwise
/// [`toolkit_target_dirs`] for `CARGO_CFG_TARGET_ARCH` /
/// `CARGO_CFG_TARGET_OS`. Empty when the override is absent and either cfg
/// is unset, which keeps discovery on the plain `{toolkit}/include` layout
/// rather than guessing.
fn build_target_dirs() -> Vec<String> {
    let override_dir = env::var(TOOLKIT_TARGET_DIR_ENV).ok();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    resolve_toolkit_target_dirs(override_dir.as_deref(), &arch, &os)
}

/// Returns the include directory containing `cuda.h`: `{toolkit}/include`
/// for standard installs, or `{toolkit}/targets/<dir>/include` for
/// redistributable layouts that have no top-level `include/`.
///
/// Only the `targets/` directories [`build_target_dirs`] yields (the
/// [`TOOLKIT_TARGET_DIR_ENV`] override, or the [`toolkit_target_dirs`] table
/// for the build target's own architecture) are probed, in order. Globbing
/// all of `targets/*` would let another architecture's headers and stubs
/// shadow the right ones on multi-target installs (`sbsa-linux` sorts before
/// `x86_64-linux`).
///
/// A missing `cuda.h` is a hard error here: bindgen cannot run without it,
/// and failing now produces one clean message instead of raw clang
/// diagnostics.
fn find_cuda_include_dir(toolkit: &str) -> Result<PathBuf, String> {
    let base = Path::new(toolkit);
    let candidates = toolkit_include_candidates(base, &build_target_dirs());

    if let Some(dir) = select_include_dir(&candidates) {
        return Ok(dir.clone());
    }

    let probed: Vec<String> = candidates
        .iter()
        .map(|dir| format!("  {}", dir.join("cuda.h").display()))
        .collect();
    Err(format!(
        "cuda-bindings: could not find cuda.h in the CUDA toolkit at `{toolkit}`.\n\
         Probed:\n\
         {}\n\
         Set CUDA_TOOLKIT_PATH or CUDA_HOME to a CUDA Toolkit install root; \
         when neither is set, {DEFAULT_TOOLKIT_DIR} is used.",
        probed.join("\n")
    ))
}

/// Probes the discovered `cuda.h` for `cuEventElapsedTime_v2` and emits the
/// `cuda_has_cuEventElapsedTime_v2` cfg when present.
///
/// CUDA 12.8 renamed the event elapsed-time driver entry point to
/// `cuEventElapsedTime_v2`; earlier toolkits only declare
/// `cuEventElapsedTime`. The cfg lets `src/lib.rs` dispatch to whichever
/// symbol the headers used for this build actually declare.
///
/// A missing `cuda.h` is already a hard error in [`find_cuda_include_dir`];
/// a present but unreadable `cuda.h` stays a warning here (treated as the
/// pre-12.8 spelling) because bindgen reports the authoritative failure
/// right after.
fn probe_event_elapsed_time_v2(cuda_h: &Path) {
    println!("cargo:rerun-if-changed={}", cuda_h.display());
    match std::fs::read_to_string(cuda_h) {
        Ok(header) => {
            if header.contains("cuEventElapsedTime_v2") {
                println!("cargo:rustc-cfg=cuda_has_cuEventElapsedTime_v2");
            }
        }
        Err(error) => {
            println!(
                "cargo:warning=cuda-bindings: failed to probe {}: {error}",
                cuda_h.display()
            );
        }
    }
}

/// Candidate directories for `rustc-link-search=native` when linking against the driver library.
///
/// Adds `{toolkit}/lib64` and `{toolkit}/lib64/stubs` when `lib64` exists. For each
/// `targets/<dir>` candidate whose `include/cuda.h` exists (redistributable / cross-layout
/// install), also adds that target's `lib` and `lib/stubs`. Only the candidates
/// [`build_target_dirs`] yields for the build target (override or table) are considered,
/// never all of `targets/*`. Order is preserved; duplicates are not filtered.
fn collect_lib_paths(toolkit: &str) -> Vec<PathBuf> {
    let base = PathBuf::from(toolkit);
    let mut paths = vec![];

    let lib64 = base.join("lib64");
    if lib64.is_dir() {
        paths.push(lib64.clone());
        paths.push(lib64.join("stubs"));
    }

    for target_dir in build_target_dirs() {
        let target_root = base.join("targets").join(target_dir);
        if target_root.join("include/cuda.h").is_file() {
            paths.push(target_root.join("lib"));
            paths.push(target_root.join("lib/stubs"));
        }
    }

    paths
}
