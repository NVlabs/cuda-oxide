/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Backend discovery and building.
//!
//! Finds or builds `librustc_codegen_cuda.so` using this priority:
//!
//! 1. `CUDA_OXIDE_BACKEND` env var (explicit override)
//! 2. Local repo (detected by presence of `crates/rustc-codegen-cuda`)
//! 3. Cached `.so` at `~/.cargo/cuda-oxide/librustc_codegen_cuda.so`
//! 4. Auto-fetch from git and build (one-time)

use std::path::{Path, PathBuf};
use std::process::Command;

/// Finds the workspace root by walking up from CWD looking for Cargo.toml
/// with a `crates/rustc-codegen-cuda` directory.
pub fn find_workspace_root() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        if dir.join("crates/rustc-codegen-cuda").is_dir() && dir.join("Cargo.toml").is_file() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Returns the path to the codegen backend `.so`, building it if necessary.
///
/// Discovery order:
/// 1. `CUDA_OXIDE_BACKEND` env var
/// 2. Local repo build (crates/rustc-codegen-cuda)
/// 3. Cached build at ~/.cargo/cuda-oxide/
/// 4. Auto-fetch + build from git
pub fn find_or_build_backend(workspace_root: &Path) -> PathBuf {
    // 1. Explicit override
    if let Ok(path) = std::env::var("CUDA_OXIDE_BACKEND") {
        let p = PathBuf::from(&path);
        if p.exists() {
            return p;
        }
        eprintln!(
            "Warning: CUDA_OXIDE_BACKEND={} does not exist, falling back to auto-detection",
            path
        );
    }

    // 2. Local repo
    let codegen_crate = workspace_root.join("crates/rustc-codegen-cuda");
    if codegen_crate.is_dir() {
        let so_path = codegen_target_dir(&codegen_crate).join("debug/librustc_codegen_cuda.so");
        build_backend_from_source(&codegen_crate);
        return so_path;
    }

    // 3. Cached .so
    if let Some(cache_dir) = cache_directory() {
        let cached_so = cache_dir.join("librustc_codegen_cuda.so");
        if cached_so.exists() {
            return cached_so;
        }
    }

    // 4. Auto-fetch from git
    auto_fetch_and_build()
}

/// Builds the backend from a local source tree.
pub fn build_backend_from_source(codegen_crate: &Path) {
    println!("Building rustc-codegen-cuda backend...");

    let rustc_sysroot = get_rustc_sysroot();
    let lib_path = rustc_sysroot.as_ref().map(|s| format!("{}/lib", s));

    let mut cmd = Command::new("cargo");
    cmd.args(["build"]).current_dir(codegen_crate);

    if let Some(ref path) = lib_path {
        cmd.env("LIBRARY_PATH", path);
        cmd.env("LD_LIBRARY_PATH", build_ld_library_path(path));
    }

    let status = cmd.status().expect("Failed to run cargo build");

    if !status.success() {
        eprintln!("Failed to build rustc-codegen-cuda");
        std::process::exit(status.code().unwrap_or(1));
    }

    let so_path = codegen_target_dir(codegen_crate).join("debug/librustc_codegen_cuda.so");
    if so_path.exists() {
        println!("✓ Backend built: {}", so_path.display());
    } else {
        eprintln!("Warning: Expected .so not found at {}", so_path.display());
    }
}

/// Resolves the cargo target directory for the codegen-cuda crate.
///
/// Honors `CARGO_TARGET_DIR` (and `CARGO_BUILD_TARGET_DIR`) — when set, the
/// child `cargo build` writes artifacts there instead of `<crate>/target`,
/// so the `.so` discovery has to follow.
fn codegen_target_dir(codegen_crate: &Path) -> PathBuf {
    if let Ok(dir) = std::env::var("CARGO_TARGET_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    if let Ok(dir) = std::env::var("CARGO_BUILD_TARGET_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir);
        }
    }
    codegen_crate.join("target")
}

/// Returns the cache directory for cuda-oxide artifacts: `~/.cargo/cuda-oxide/`.
fn cache_directory() -> Option<PathBuf> {
    dirs_path().map(|d| d.join("cuda-oxide"))
}

/// Resolves the Cargo home directory (`$CARGO_HOME` or `$HOME/.cargo`).
fn dirs_path() -> Option<PathBuf> {
    std::env::var("CARGO_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".cargo"))
        })
}

/// Clones the cuda-oxide repo into the cache directory and builds the backend.
///
/// This is the last-resort discovery path for external users who don't have
/// the repo checked out locally. The clone is shallow (`--depth 1`) to keep
/// the download small.
fn auto_fetch_and_build() -> PathBuf {
    let cache_dir = cache_directory().unwrap_or_else(|| {
        eprintln!("Error: Cannot determine cache directory.");
        eprintln!("Set CARGO_HOME or HOME environment variable.");
        std::process::exit(1);
    });

    let src_dir = cache_dir.join("src");
    let so_path = cache_dir.join("librustc_codegen_cuda.so");

    std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");

    if !src_dir.join("Cargo.toml").exists() {
        eprintln!("Backend not found. Fetching cuda-oxide source (one-time setup)...");
        eprintln!();
        let status = Command::new("git")
            .args([
                "clone",
                "--depth",
                "1",
                "https://github.com/NVlabs/cuda-oxide.git",
                src_dir.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to run git clone. Is git installed?");

        if !status.success() {
            eprintln!("Failed to clone cuda-oxide repository.");
            eprintln!("You can manually set CUDA_OXIDE_BACKEND=/path/to/librustc_codegen_cuda.so");
            std::process::exit(1);
        }
    }

    let codegen_crate = src_dir.join("crates/rustc-codegen-cuda");
    build_backend_from_source(&codegen_crate);

    let built_so = codegen_target_dir(&codegen_crate).join("debug/librustc_codegen_cuda.so");
    if built_so.exists() {
        std::fs::copy(&built_so, &so_path).expect("Failed to copy backend to cache");
        eprintln!("✓ Backend cached at {}", so_path.display());
    }

    so_path
}

/// Returns the active rustc sysroot path (e.g., `~/.rustup/toolchains/nightly-...`).
///
/// Used to locate `libstd`, `librustc_driver`, and other compiler libraries that
/// must be on `LD_LIBRARY_PATH` when loading the codegen backend `.so`.
pub fn get_rustc_sysroot() -> Option<String> {
    let output = Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .ok()?;

    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        None
    }
}

/// Build LD_LIBRARY_PATH preserving existing paths (important for NixOS, etc.).
pub fn build_ld_library_path(sysroot_lib: &str) -> String {
    if let Ok(existing) = std::env::var("LD_LIBRARY_PATH") {
        format!("{}:{}", existing, sysroot_lib)
    } else {
        sysroot_lib.to_string()
    }
}
