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
//! 3. Cached `.so` at `~/.cargo/cuda-oxide/librustc_codegen_cuda.so`,
//!    but only when it isn't older than the running `cargo-oxide` binary
//! 4. Auto-fetch from git and build (one-time, or after a stale-cache miss)
//!
//! ## Cache staleness (issue #49)
//!
//! `cargo install` always rewrites `~/.cargo/bin/cargo-oxide` on every
//! upgrade, bumping its mtime. The cached `.so` is only ever written by
//! step 4 below, so a binary newer than the cache is the canonical signal
//! that the user has just upgraded `cargo-oxide` and the cached backend
//! no longer matches the binary loading it. When step 3 detects that, we
//! drop both the cached `.so` *and* the cached source tree so that step 4
//! re-clones fresh and rebuilds, rather than rebuilding from a clone that
//! was taken whenever the user first installed.

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
        let so_path = codegen_crate.join("target/debug/librustc_codegen_cuda.so");
        build_backend_from_source(&codegen_crate);
        return so_path;
    }

    // 3. Cached .so. Only honored when it isn't older than the running
    //    cargo-oxide binary; see the module-level comment about issue #49.
    if let Some(cache_dir) = cache_directory() {
        let cached_so = cache_dir.join("librustc_codegen_cuda.so");
        if cached_so.exists() {
            if !cached_backend_is_stale(&cached_so) {
                return cached_so;
            }
            invalidate_cache(&cache_dir);
        }
    }

    // 4. Auto-fetch from git
    auto_fetch_and_build()
}

/// Returns where the backend `.so` lives (or would live), with NO side
/// effects: never builds, never clones, never touches the network.
///
/// Mirrors the discovery order of [`find_or_build_backend`] minus its
/// build/clone steps:
///
/// 1. `CUDA_OXIDE_BACKEND` env var, returned even when the file is missing
///    so the caller can report the configured-but-absent path.
/// 2. Local repo build path (`crates/rustc-codegen-cuda/target/debug/...`).
/// 3. Cache path at `~/.cargo/cuda-oxide/librustc_codegen_cuda.so`.
///
/// `cargo oxide doctor` uses this so that a diagnostic run never triggers a
/// multi-minute backend build or a git clone before it can print anything.
pub fn backend_so_candidate(workspace_root: &Path) -> PathBuf {
    if let Ok(path) = std::env::var("CUDA_OXIDE_BACKEND") {
        return PathBuf::from(path);
    }

    let codegen_crate = workspace_root.join("crates/rustc-codegen-cuda");
    if codegen_crate.is_dir() {
        return codegen_crate.join("target/debug/librustc_codegen_cuda.so");
    }

    cache_directory()
        .map(|dir| dir.join("librustc_codegen_cuda.so"))
        .unwrap_or_else(|| PathBuf::from("librustc_codegen_cuda.so"))
}

/// Returns true when the cached backend `.so` is older than the running
/// `cargo-oxide` binary, which means the user has upgraded the binary
/// since the cache was last built.
///
/// Conservative on errors: if we can't resolve our own executable path or
/// stat either file, we report "not stale" so a working cache is never
/// invalidated on a failed metadata read.
fn cached_backend_is_stale(cached_so: &Path) -> bool {
    let Ok(self_path) = std::env::current_exe() else {
        return false;
    };
    let Ok(self_meta) = std::fs::metadata(&self_path) else {
        return false;
    };
    let Ok(so_meta) = std::fs::metadata(cached_so) else {
        return false;
    };
    let (Ok(self_mtime), Ok(so_mtime)) = (self_meta.modified(), so_meta.modified()) else {
        return false;
    };
    self_mtime > so_mtime
}

/// Drop both the cached `.so` and the cached source tree at `cache_dir`.
///
/// Removing `src/` is what forces the auto-fetch step to re-clone instead
/// of rebuilding from a checkout that was taken at first-install time.
/// Both removals are best-effort; if either fails (e.g. permissions), we
/// fall through to step 4, which will fail loudly with a clear error.
fn invalidate_cache(cache_dir: &Path) {
    eprintln!(
        "Detected upgraded cargo-oxide; refreshing cached backend at {} (issue #49).",
        cache_dir.display()
    );
    let _ = std::fs::remove_file(cache_dir.join("librustc_codegen_cuda.so"));
    let _ = std::fs::remove_dir_all(cache_dir.join("src"));
}

/// Builds the backend from a local source tree.
pub fn build_backend_from_source(codegen_crate: &Path) {
    println!("Building rustc-codegen-cuda backend...");

    let rustc_sysroot = get_rustc_sysroot();
    let lib_path = rustc_sysroot.as_ref().map(|s| format!("{}/lib", s));

    let mut cmd = Command::new("cargo");
    cmd.args(["build"]).current_dir(codegen_crate);

    if let Some(ref path) = lib_path {
        cmd.env("LIBRARY_PATH", build_library_path(path));
        cmd.env("LD_LIBRARY_PATH", build_ld_library_path(path));
    }

    let status = cmd.status().expect("Failed to run cargo build");

    if !status.success() {
        eprintln!("Failed to build rustc-codegen-cuda");
        std::process::exit(status.code().unwrap_or(1));
    }

    let so_path = codegen_crate.join("target/debug/librustc_codegen_cuda.so");
    if so_path.exists() {
        println!("✓ Backend built: {}", so_path.display());
    } else {
        eprintln!("Warning: Expected .so not found at {}", so_path.display());
    }
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

    let built_so = codegen_crate.join("target/debug/librustc_codegen_cuda.so");
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

/// Build LIBRARY_PATH preserving existing paths.
pub fn build_library_path(sysroot_lib: &str) -> String {
    if let Ok(existing) = std::env::var("LIBRARY_PATH") {
        format!("{}:{}", existing, sysroot_lib)
    } else {
        sysroot_lib.to_string()
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::time::{Duration, SystemTime};

    /// A cached `.so` whose mtime predates the running test binary should
    /// be reported stale. The test binary is `current_exe()`, which was
    /// just rebuilt by `cargo test`, so its mtime is necessarily newer
    /// than a file we explicitly backdate.
    #[test]
    fn stale_when_cache_predates_running_binary() {
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        write_with_mtime(
            &so,
            b"stale",
            SystemTime::now() - Duration::from_secs(365 * 24 * 60 * 60),
        );

        assert!(
            cached_backend_is_stale(&so),
            "cache backdated by 1y must be reported stale"
        );
    }

    /// A cached `.so` written *after* the running binary is fresh and
    /// must not be reported stale, otherwise we'd thrash the cache on
    /// every invocation.
    #[test]
    fn fresh_when_cache_postdates_running_binary() {
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        write_with_mtime(
            &so,
            b"fresh",
            SystemTime::now() + Duration::from_secs(365 * 24 * 60 * 60),
        );

        assert!(
            !cached_backend_is_stale(&so),
            "cache postdating the test binary must be reported fresh"
        );
    }

    /// Missing cache file: we report not-stale and the caller's
    /// `cached_so.exists()` guard is what skips it. This keeps the
    /// helper conservative on stat failures.
    #[test]
    fn not_stale_when_cache_file_missing() {
        let dir = tempdir();
        let so = dir.join("does_not_exist.so");
        assert!(!cached_backend_is_stale(&so));
    }

    fn tempdir() -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "cargo-oxide-backend-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn write_with_mtime(path: &Path, contents: &[u8], mtime: SystemTime) {
        let mut f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)
            .unwrap();
        f.write_all(contents).unwrap();
        f.set_modified(mtime).unwrap();
    }
}
