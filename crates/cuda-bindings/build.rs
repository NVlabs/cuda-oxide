/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::{env, error::Error, path::Path, path::PathBuf, process::exit};

/// Returns the CUDA toolkit install root: `CUDA_TOOLKIT_PATH` if set, otherwise `/usr/local/cuda`.
/// Used for include paths, library search paths, and bindgen’s Clang configuration.
fn cuda_toolkit_dir() -> String {
    env::var("CUDA_TOOLKIT_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string())
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
    println!("cargo::rustc-check-cfg=cfg(cuda_has_cuEventElapsedTime_v2)");

    let toolkit = cuda_toolkit_dir();
    let cuda_h = Path::new(&toolkit).join("include/cuda.h");
    println!("cargo:rerun-if-changed={}", cuda_h.display());
    if std::fs::read_to_string(&cuda_h)
        .map(|contents| contents.contains("cuEventElapsedTime_v2"))
        .unwrap_or(false)
    {
        println!("cargo:rustc-cfg=cuda_has_cuEventElapsedTime_v2");
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
/// Adds `{toolkit}/lib64` and `{toolkit}/lib64/stubs` when `lib64` exists. If
/// `{toolkit}/targets/x86_64-linux/include/cuda.h` exists (redistributable / cross-layout install),
/// also adds `targets/x86_64-linux/lib` and `.../lib/stubs`. Order is preserved; duplicates are not
/// filtered.
fn collect_lib_paths(toolkit: &str) -> Vec<PathBuf> {
    let base = PathBuf::from(toolkit);
    let mut paths = vec![];

    let lib64 = base.join("lib64");
    if lib64.is_dir() {
        paths.push(lib64.clone());
        paths.push(lib64.join("stubs"));
    }

    let targets = base.join("targets/x86_64-linux");
    if targets.join("include/cuda.h").is_file() {
        paths.push(targets.join("lib"));
        paths.push(targets.join("lib/stubs"));
    }

    paths
}
