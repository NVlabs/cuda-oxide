/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Backend discovery and building.
//!
//! Finds or builds `librustc_codegen_cuda.so` using this priority:
//!
//! 1. `CUDA_OXIDE_BACKEND` env var (explicit override)
//! 2. Project config (`.cargo/cuda-oxide.toml`)
//! 3. Local repo (detected by presence of `crates/rustc-codegen-cuda`)
//! 4. Cached `.so` at `~/.cargo/cuda-oxide/librustc_codegen_cuda.so`,
//!    but only when it isn't older than the running `cargo-oxide` binary
//! 5. Auto-fetch from git and build (one-time, or after a stale-cache miss)
//!
//! ## Cache staleness (issue #49)
//!
//! `cargo install` always rewrites `~/.cargo/bin/cargo-oxide` on every
//! upgrade, bumping its mtime. The cached `.so` is only ever written by
//! step 5 below, so a binary newer than the cache is the canonical signal
//! that the user has just upgraded `cargo-oxide` and the cached backend
//! no longer matches the binary loading it. When step 4 detects that, we
//! drop both the cached `.so` *and* the cached source tree so that step 5
//! re-clones fresh and rebuilds, rather than rebuilding from a clone that
//! was taken whenever the user first installed.
//!
//! ## Cache staleness vs. source (backend source advances)
//!
//! The binary-mtime check above does not fire when the developer updates
//! the backend SOURCE (the `rustc-codegen-cuda` crate) but leaves the
//! `cargo-oxide` binary unchanged. In that case the cached `.so` is older
//! than the source it was built from, yet the binary check sees no upgrade
//! and the stale backend is silently reused. To catch this we also compare
//! the cached `.so` against the newest mtime of the backend source inputs
//! (the crate's `src/**` and `Cargo.toml`) found in the cached source tree.
//! When the source tree cannot be located we degrade gracefully to the
//! binary-only check rather than erroring.
//!
//! The two stale signals call for different recovery. A binary upgrade means
//! the cached source may no longer match the new binary, so we drop the
//! source tree and re-clone fresh (above). A source advance means the cached
//! source IS the newer truth, so we rebuild the `.so` from that existing
//! source in place; re-cloning would throw away the very source that
//! triggered the rebuild. Binary staleness takes precedence when both fire.
//!
//! ## Cache staleness vs. toolchain (the active rustc changes)
//!
//! The mtime checks above miss a toolchain swap: the cached `.so` is
//! dynamically linked against one specific `librustc_driver-<hash>.so`, but a
//! repo `rust-toolchain.toml` or a changed default nightly leaves the
//! `cargo-oxide` binary and the cached source untouched. The stale `.so` then
//! loads against the wrong driver and fails with a cryptic
//! `librustc_driver-<hash>.so: cannot open shared object file`. To catch this
//! we record the active toolchain fingerprint (`rustc -vV`) next to the cached
//! `.so` at build time and compare it on every lookup; a recorded fingerprint
//! that differs from the active toolchain forces a fresh re-clone and rebuild.
//! This check has the highest precedence, since a toolchain mismatch makes the
//! cached `.so` unloadable regardless of mtimes. A cache predating the
//! fingerprint file defers to the mtime checks (a `cargo-oxide` reinstall or
//! `rm -rf ~/.cargo/cuda-oxide` heals those).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

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
/// 2. Project config (`.cargo/cuda-oxide.toml`)
/// 3. Local repo build (crates/rustc-codegen-cuda)
/// 4. Cached build at ~/.cargo/cuda-oxide/
/// 5. Auto-fetch + build from git
pub fn find_or_build_backend(workspace_root: &Path, configured_backend: Option<&Path>) -> PathBuf {
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

    // 2. Project config
    if let Some(path) = configured_backend {
        if path.exists() {
            return path.to_path_buf();
        }
        eprintln!(
            "Error: configured cuda-oxide backend does not exist: {}",
            path.display()
        );
        eprintln!("Build it or update `.cargo/cuda-oxide.toml`.");
        std::process::exit(1);
    }

    // 3. Local repo
    let codegen_crate = workspace_root.join("crates/rustc-codegen-cuda");
    if codegen_crate.is_dir() {
        return build_backend_from_source(&codegen_crate);
    }

    // 4. Cached .so. Only honored when it isn't older than the running
    //    cargo-oxide binary; see the module-level comment about issue #49.
    if let Some(cache_dir) = cache_directory() {
        let cached_so = cache_dir.join("librustc_codegen_cuda.so");
        if cached_so.exists() {
            let source_dir = cache_dir.join("src/crates/rustc-codegen-cuda");
            match cached_backend_status(&cached_so, Some(&source_dir)) {
                CacheStatus::Fresh => return cached_so,
                CacheStatus::StaleVsBinary => invalidate_cache(&cache_dir),
                CacheStatus::StaleVsToolchain => {
                    eprintln!(
                        "Cached backend was built against a different Rust \
                         toolchain; re-cloning and rebuilding at {}.",
                        cache_dir.display()
                    );
                    invalidate_cache(&cache_dir);
                }
                CacheStatus::StaleVsSource => {
                    // The cached source advanced; rebuild the `.so` from it in
                    // place. We do NOT invalidate the cache here, so the
                    // auto-fetch step below skips the clone (the source tree is
                    // still present) and rebuilds from the existing source.
                    eprintln!(
                        "Cached backend source at {} is newer than the cached \
                         library; rebuilding from it in place.",
                        source_dir.display()
                    );
                }
            }
        }
    }

    // 5. Auto-fetch from git
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
/// 2. Project config (`.cargo/cuda-oxide.toml`), returned even when missing
///    so the caller can report the configured-but-absent path.
/// 3. Local repo host build path
///    (`crates/rustc-codegen-cuda/target/<host>/debug/...`).
/// 4. Cache path at `~/.cargo/cuda-oxide/librustc_codegen_cuda.so`.
///
/// `cargo oxide doctor` uses this so that a diagnostic run never triggers a
/// multi-minute backend build or a git clone before it can print anything.
pub fn backend_so_candidate(workspace_root: &Path, configured_backend: Option<&Path>) -> PathBuf {
    if let Ok(path) = std::env::var("CUDA_OXIDE_BACKEND") {
        return PathBuf::from(path);
    }

    if let Some(path) = configured_backend {
        return path.to_path_buf();
    }

    let codegen_crate = workspace_root.join("crates/rustc-codegen-cuda");
    if codegen_crate.is_dir() {
        return backend_so_path_candidate(&codegen_crate);
    }

    cache_directory()
        .map(|dir| dir.join("librustc_codegen_cuda.so"))
        .unwrap_or_else(|| PathBuf::from("librustc_codegen_cuda.so"))
}

/// Why the cached backend is out of date, or that it is current. The two
/// stale variants drive different recovery (re-clone vs. rebuild in place);
/// see the module-level comment.
#[derive(Debug, PartialEq, Eq)]
enum CacheStatus {
    /// Cache is up to date; reuse the cached `.so`.
    Fresh,
    /// The running `cargo-oxide` binary is newer than the cache: the user
    /// upgraded the binary, so the cached source may no longer match it.
    StaleVsBinary,
    /// The cached backend source is newer than the cached `.so`: the source
    /// was advanced in place and the `.so` should be rebuilt from it.
    StaleVsSource,
    /// The cached `.so` was built against a different Rust toolchain than the
    /// active one: it links a `librustc_driver` hash that no longer resolves,
    /// so it must be re-cloned and rebuilt. Highest precedence: an unloadable
    /// `.so` is stale regardless of mtimes.
    StaleVsToolchain,
}

/// Classifies the cached backend `.so` against the running `cargo-oxide`
/// binary (the user upgraded the binary) and the newest backend source input
/// (the developer advanced the source). When `source_dir` is `None`, or no
/// source inputs can be found under it, only the binary check applies.
/// Binary staleness takes precedence when both fire, since a binary upgrade
/// wants a fresh clone (which also picks up the newest source).
///
/// Conservative on errors: if we can't stat the cached `.so`, we report
/// [`CacheStatus::Fresh`] so a working cache is never invalidated on a failed
/// metadata read.
fn cached_backend_status(cached_so: &Path, source_dir: Option<&Path>) -> CacheStatus {
    let Ok(so_meta) = std::fs::metadata(cached_so) else {
        return CacheStatus::Fresh;
    };
    let Ok(so_mtime) = so_meta.modified() else {
        return CacheStatus::Fresh;
    };

    // Toolchain check (highest precedence): a toolchain swap makes the cached
    // `.so` unloadable no matter what the mtimes say, so it wins over the
    // binary/source mtime signals below.
    if let Some(cache_dir) = cached_so.parent()
        && toolchain_fingerprint_mismatch(cache_dir)
    {
        return CacheStatus::StaleVsToolchain;
    }

    let self_mtime = std::env::current_exe()
        .ok()
        .and_then(|p| std::fs::metadata(p).ok())
        .and_then(|m| m.modified().ok());

    // Binary check: if we can't stat our own executable, fall through to the
    // source check rather than declaring the cache fresh, so the source
    // signal is still honoured.
    if matches!(self_mtime, Some(self_mtime) if self_mtime > so_mtime) {
        return CacheStatus::StaleVsBinary;
    }

    let stale_vs_source = source_dir
        .and_then(newest_backend_source_mtime)
        .map(|src_mtime| src_mtime > so_mtime)
        .unwrap_or(false);
    if stale_vs_source {
        return CacheStatus::StaleVsSource;
    }

    CacheStatus::Fresh
}

/// File next to the cached `.so` recording the toolchain it was built against.
const TOOLCHAIN_FINGERPRINT_FILE: &str = "toolchain-fingerprint.txt";

/// A stable fingerprint of the active Rust toolchain: the full `rustc -vV`
/// output (release, commit-hash, host, LLVM version). The cached backend `.so`
/// links against this toolchain's `librustc_driver`, so any change here means
/// the cache can no longer be loaded.
fn current_toolchain_fingerprint() -> Option<String> {
    let output = Command::new("rustc").args(["-vV"]).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Host target triple of the active rustc, as reported by `rustc -vV`.
fn active_host_triple() -> Option<String> {
    current_toolchain_fingerprint()?
        .lines()
        .find_map(|line| line.strip_prefix("host: ").map(str::to_owned))
}

/// True when the cached backend records a toolchain fingerprint that differs
/// from the active toolchain. Conservative: if the active fingerprint cannot be
/// read, or no fingerprint was recorded (a cache predating this check), returns
/// `false` and defers to the mtime checks rather than thrashing a working
/// cache. Pre-fingerprint caches are healed by the binary-mtime check on the
/// next `cargo-oxide` reinstall, or by `rm -rf ~/.cargo/cuda-oxide`.
fn toolchain_fingerprint_mismatch(cache_dir: &Path) -> bool {
    let Some(current) = current_toolchain_fingerprint() else {
        return false;
    };
    match std::fs::read_to_string(cache_dir.join(TOOLCHAIN_FINGERPRINT_FILE)) {
        Ok(stored) => stored.trim() != current,
        Err(_) => false,
    }
}

/// Records the active toolchain fingerprint next to the cached `.so`. Best
/// effort: a write failure just means the next run re-detects a mismatch and
/// rebuilds again.
fn write_toolchain_fingerprint(cache_dir: &Path) {
    if let Some(fp) = current_toolchain_fingerprint() {
        let _ = std::fs::write(cache_dir.join(TOOLCHAIN_FINGERPRINT_FILE), fp);
    }
}

/// Returns the newest mtime among the backend source inputs under
/// `source_dir`: every file in `src/**` plus the crate `Cargo.toml`.
///
/// Returns `None` when the directory cannot be located or yields no
/// readable inputs, which lets [`cached_backend_status`] degrade to the
/// binary-only check. The walk is best-effort: unreadable entries are
/// skipped rather than treated as failures.
fn newest_backend_source_mtime(source_dir: &Path) -> Option<SystemTime> {
    let mut newest: Option<SystemTime> = None;

    let mut consider = |path: &Path| {
        if let Ok(mtime) = std::fs::metadata(path).and_then(|m| m.modified()) {
            newest = Some(match newest {
                Some(cur) if cur >= mtime => cur,
                _ => mtime,
            });
        }
    };

    consider(&source_dir.join("Cargo.toml"));
    visit_files(&source_dir.join("src"), &mut consider);

    newest
}

/// Recursively visits every regular file under `dir`, calling `f` on each.
/// Best-effort: directories that cannot be read are skipped silently.
fn visit_files(dir: &Path, f: &mut dyn FnMut(&Path)) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        match entry.file_type() {
            Ok(ft) if ft.is_dir() => visit_files(&path, f),
            Ok(ft) if ft.is_file() => f(&path),
            _ => {}
        }
    }
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
pub fn build_backend_from_source(codegen_crate: &Path) -> PathBuf {
    println!("Building rustc-codegen-cuda backend...");

    let rustc_sysroot = get_rustc_sysroot();
    let lib_path = rustc_sysroot.as_ref().map(|s| format!("{}/lib", s));

    let mut cmd = backend_build_command(codegen_crate, lib_path.as_deref());
    let output = cmd.output().unwrap_or_else(|error| {
        eprintln!("Failed to run cargo build for rustc-codegen-cuda: {error}");
        std::process::exit(1);
    });

    render_cargo_diagnostics(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        eprint!("{stderr}");
    }

    if !output.status.success() {
        eprintln!("Failed to build rustc-codegen-cuda");
        std::process::exit(output.status.code().unwrap_or(1));
    }

    let so_path =
        backend_artifact_from_cargo_output(codegen_crate, &output.stdout).unwrap_or_else(|error| {
            eprintln!("Backend build succeeded, but {error}");
            std::process::exit(1);
        });
    if !so_path.is_file() {
        eprintln!(
            "Backend build reported {}, but that file does not exist",
            so_path.display()
        );
        std::process::exit(1);
    }
    println!("✓ Backend built: {}", so_path.display());
    so_path
}

fn backend_build_command(codegen_crate: &Path, rustc_lib_path: Option<&str>) -> Command {
    let codegen_crate = absolute_path(codegen_crate);
    let target_dir = codegen_crate.join("target");
    let mut cmd = Command::new("cargo");
    cmd.args([
        "build",
        "--lib",
        "--target",
        "host-tuple",
        "--message-format=json-render-diagnostics",
        "--target-dir",
    ])
    .arg(&target_dir)
    .current_dir(&codegen_crate);

    // The backend is a host rustc plugin, not an application artifact. Keep it
    // out of an application's target directory and override both
    // CARGO_BUILD_TARGET and `[build] target`: an explicit `--target
    // host-tuple` makes Cargo compile the dylib for the running toolchain.
    cmd.env("CARGO_TARGET_DIR", &target_dir);
    cmd.env_remove("CARGO_BUILD_TARGET");

    if let Some(path) = rustc_lib_path {
        cmd.env("LIBRARY_PATH", build_library_path(path));
        cmd.env("LD_LIBRARY_PATH", build_ld_library_path(path));
    }

    cmd
}

fn absolute_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|current| current.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    }
}

fn backend_dylib_filename() -> String {
    format!(
        "{}rustc_codegen_cuda{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    )
}

fn backend_so_path_candidate(codegen_crate: &Path) -> PathBuf {
    let target_dir = codegen_crate.join("target");
    let profile_dir = active_host_triple()
        .map(|host| target_dir.join(host).join("debug"))
        .unwrap_or_else(|| target_dir.join("debug"));
    profile_dir.join(backend_dylib_filename())
}

fn render_cargo_diagnostics(stdout: &[u8]) {
    for line in String::from_utf8_lossy(stdout).lines() {
        let Ok(message) = serde_json::from_str::<serde_json::Value>(line) else {
            if !line.is_empty() {
                println!("{line}");
            }
            continue;
        };
        if let Some(rendered) = message
            .get("message")
            .and_then(|message| message.get("rendered"))
            .and_then(|rendered| rendered.as_str())
        {
            eprint!("{rendered}");
        }
    }
}

/// Select the backend path Cargo reported for this exact manifest and dylib
/// target. There is deliberately no guessed-path fallback: a successful Cargo
/// exit without this artifact must fail instead of loading an older file left
/// in `target/debug` by a previous host build.
fn backend_artifact_from_cargo_output(
    codegen_crate: &Path,
    stdout: &[u8],
) -> Result<PathBuf, String> {
    let expected_manifest = codegen_crate
        .join("Cargo.toml")
        .canonicalize()
        .map_err(|error| format!("could not resolve backend manifest: {error}"))?;
    let expected_filename = backend_dylib_filename();
    let mut artifacts = Vec::new();

    for line in String::from_utf8_lossy(stdout).lines() {
        let Ok(message) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if message.get("reason").and_then(|reason| reason.as_str()) != Some("compiler-artifact") {
            continue;
        }
        let manifest_matches = message
            .get("manifest_path")
            .and_then(|path| path.as_str())
            .and_then(|path| Path::new(path).canonicalize().ok())
            .is_some_and(|path| path == expected_manifest);
        let target_matches = message
            .get("target")
            .and_then(|target| target.get("name"))
            .and_then(|name| name.as_str())
            == Some("rustc_codegen_cuda")
            && message
                .get("target")
                .and_then(|target| target.get("kind"))
                .and_then(|kind| kind.as_array())
                .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_str() == Some("dylib")));
        if !manifest_matches || !target_matches {
            continue;
        }

        let Some(filenames) = message.get("filenames").and_then(|files| files.as_array()) else {
            continue;
        };
        for filename in filenames.iter().filter_map(|file| file.as_str()) {
            let path = PathBuf::from(filename);
            if path.file_name() == Some(std::ffi::OsStr::new(&expected_filename))
                && !artifacts.contains(&path)
            {
                artifacts.push(path);
            }
        }
    }

    match artifacts.as_slice() {
        [artifact] => Ok(artifact.clone()),
        [] => Err(format!(
            "Cargo reported no `{expected_filename}` artifact for rustc_codegen_cuda"
        )),
        _ => Err(format!(
            "Cargo reported multiple `{expected_filename}` artifacts for rustc_codegen_cuda"
        )),
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
    let built_so = build_backend_from_source(&codegen_crate);
    if built_so.exists() {
        std::fs::copy(&built_so, &so_path).expect("Failed to copy backend to cache");
        write_toolchain_fingerprint(&cache_dir);
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
    use std::ffi::OsStr;
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::time::{Duration, SystemTime};

    /// The codegen backend is part of the cuda-oxide toolchain, not the
    /// application being debugged/sanitized. A user-supplied
    /// `CARGO_TARGET_DIR` for the application must therefore not change where
    /// cargo-oxide builds or looks for `librustc_codegen_cuda.so`.
    #[test]
    fn backend_build_command_isolates_target_dir_and_forces_the_host() {
        let command = backend_build_command(Path::new("/tmp/codegen"), Some("/tmp/rustc/lib"));

        let cargo_target_dir = command
            .get_envs()
            .find_map(|(key, value)| (key == OsStr::new("CARGO_TARGET_DIR")).then_some(value));
        let cargo_build_target = command
            .get_envs()
            .find_map(|(key, value)| (key == OsStr::new("CARGO_BUILD_TARGET")).then_some(value));
        let args = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>();

        assert_eq!(
            cargo_target_dir.flatten(),
            Some(OsStr::new("/tmp/codegen/target"))
        );
        assert_eq!(cargo_build_target, Some(None));
        assert_eq!(command.get_current_dir(), Some(Path::new("/tmp/codegen")));
        assert!(
            args.windows(2)
                .any(|args| args == ["--target", "host-tuple"])
        );
        assert!(
            args.windows(2)
                .any(|args| args == ["--target-dir", "/tmp/codegen/target"])
        );
        assert!(
            args.iter()
                .any(|arg| arg == "--message-format=json-render-diagnostics")
        );
    }

    #[test]
    fn backend_artifact_uses_cargos_host_path_not_a_stale_legacy_path() {
        let root = tempdir();
        let codegen = root.join("crates/rustc-codegen-cuda");
        std::fs::create_dir_all(&codegen).unwrap();
        std::fs::write(
            codegen.join("Cargo.toml"),
            "[package]\nname='rustc_codegen_cuda'\n",
        )
        .unwrap();

        let stale = codegen.join("target/debug").join(backend_dylib_filename());
        let fresh = codegen
            .join("target/x86_64-unknown-linux-gnu/debug")
            .join(backend_dylib_filename());
        std::fs::create_dir_all(stale.parent().unwrap()).unwrap();
        std::fs::create_dir_all(fresh.parent().unwrap()).unwrap();
        std::fs::write(&stale, b"stale backend").unwrap();
        std::fs::write(&fresh, b"fresh host backend").unwrap();

        let message = serde_json::json!({
            "reason": "compiler-artifact",
            "manifest_path": codegen.join("Cargo.toml"),
            "target": {
                "kind": ["dylib"],
                "name": "rustc_codegen_cuda"
            },
            "filenames": [fresh]
        });
        let output = format!("{message}\n");

        assert_eq!(
            backend_artifact_from_cargo_output(&codegen, output.as_bytes()).unwrap(),
            fresh
        );

        let no_artifact = b"{\"reason\":\"build-finished\",\"success\":true}\n";
        assert!(
            backend_artifact_from_cargo_output(&codegen, no_artifact).is_err(),
            "a stale target/debug dylib must never be used when Cargo did not report it"
        );
    }

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

        assert_eq!(
            cached_backend_status(&so, None),
            CacheStatus::StaleVsBinary,
            "cache backdated by 1y must be stale vs the running binary"
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

        assert_eq!(
            cached_backend_status(&so, None),
            CacheStatus::Fresh,
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
        assert_eq!(cached_backend_status(&so, None), CacheStatus::Fresh);
    }

    /// A backend source input newer than the cached `.so` must report
    /// `StaleVsSource` (the "developer advanced the source" case that issue
    /// #49's binary check alone misses). To isolate the source signal from
    /// the binary signal, the `.so` is future-dated past the running test
    /// binary, and the source file is dated later still.
    #[test]
    fn stale_when_source_postdates_cache() {
        let year = Duration::from_secs(365 * 24 * 60 * 60);
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        // Cache newer than the running binary so binary-staleness does NOT fire.
        let cache_mtime = SystemTime::now() + year;
        write_with_mtime(&so, b"built", cache_mtime);

        let src = dir.join("src");
        std::fs::create_dir_all(&src).unwrap();
        // Source newer than the cache: this is what trips source-staleness.
        write_with_mtime(
            &src.join("lib.rs"),
            b"// updated source",
            cache_mtime + year,
        );
        // Cargo.toml older than the .so; the src file is what trips staleness.
        write_with_mtime(&dir.join("Cargo.toml"), b"[package]", SystemTime::now());

        assert_eq!(
            cached_backend_status(&so, Some(&dir)),
            CacheStatus::StaleVsSource,
            "source newer than cached .so must be stale vs source (rebuild in place)"
        );
    }

    /// When every source input predates the cached `.so` (and the running
    /// binary too), the cache is fresh and must not be invalidated.
    #[test]
    fn fresh_when_source_predates_cache() {
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        // Cache far in the future so the running test binary can't make it stale.
        let cache_mtime = SystemTime::now() + Duration::from_secs(365 * 24 * 60 * 60);
        write_with_mtime(&so, b"built", cache_mtime);

        let src = dir.join("src");
        std::fs::create_dir_all(&src).unwrap();
        write_with_mtime(
            &src.join("lib.rs"),
            b"// old source",
            SystemTime::now() - Duration::from_secs(60),
        );
        write_with_mtime(
            &dir.join("Cargo.toml"),
            b"[package]",
            SystemTime::now() - Duration::from_secs(60),
        );

        assert_eq!(
            cached_backend_status(&so, Some(&dir)),
            CacheStatus::Fresh,
            "source older than cached .so must be reported fresh"
        );
    }

    /// A missing source tree must degrade to the binary-only check rather
    /// than erroring or spuriously invalidating a future-dated cache.
    #[test]
    fn fresh_when_source_dir_absent() {
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        write_with_mtime(
            &so,
            b"fresh",
            SystemTime::now() + Duration::from_secs(365 * 24 * 60 * 60),
        );
        let absent = dir.join("no-such-src-tree");
        assert_eq!(
            cached_backend_status(&so, Some(&absent)),
            CacheStatus::Fresh,
            "absent source tree must fall back to binary-only (fresh here)"
        );
    }

    /// When BOTH the running binary and the cached source postdate the `.so`,
    /// the binary signal wins so recovery re-clones fresh rather than
    /// rebuilding from a source tree that a binary upgrade may have outdated.
    #[test]
    fn binary_staleness_takes_precedence_over_source() {
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        // Backdate the `.so` so the freshly built test binary is newer than it.
        let base = SystemTime::now() - Duration::from_secs(365 * 24 * 60 * 60);
        write_with_mtime(&so, b"built", base);

        // Make the cached source newer than the `.so` too, so both signals fire.
        let src = dir.join("src");
        std::fs::create_dir_all(&src).unwrap();
        write_with_mtime(
            &src.join("lib.rs"),
            b"// updated source",
            base + Duration::from_secs(30),
        );

        assert_eq!(
            cached_backend_status(&so, Some(&dir)),
            CacheStatus::StaleVsBinary,
            "binary staleness must win over source staleness"
        );
    }

    /// A cached `.so` whose recorded toolchain fingerprint differs from the
    /// active toolchain must be `StaleVsToolchain`, even when the mtimes alone
    /// would call it fresh. This is the case the mtime checks miss: the active
    /// rustc changed (e.g. a repo `rust-toolchain.toml`) while the binary and
    /// source are untouched, leaving the cached `.so` linked against a
    /// `librustc_driver` hash that no longer resolves.
    #[test]
    fn stale_when_toolchain_fingerprint_differs() {
        let year = Duration::from_secs(365 * 24 * 60 * 60);
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        // Future-date the `.so` so the binary/source mtime checks cannot fire.
        write_with_mtime(&so, b"built", SystemTime::now() + year);
        std::fs::write(
            dir.join(TOOLCHAIN_FINGERPRINT_FILE),
            "rustc 0.0.0 (deadbeef 1970-01-01)",
        )
        .unwrap();

        assert_eq!(
            cached_backend_status(&so, None),
            CacheStatus::StaleVsToolchain,
            "a recorded fingerprint differing from the active toolchain must be stale"
        );
    }

    /// A cached `.so` whose recorded fingerprint matches the active toolchain
    /// (with fresh mtimes) must be `Fresh`.
    #[test]
    fn fresh_when_toolchain_fingerprint_matches() {
        let Some(fp) = current_toolchain_fingerprint() else {
            return; // no rustc here; nothing to assert
        };
        let year = Duration::from_secs(365 * 24 * 60 * 60);
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        write_with_mtime(&so, b"built", SystemTime::now() + year);
        std::fs::write(dir.join(TOOLCHAIN_FINGERPRINT_FILE), fp).unwrap();

        assert_eq!(
            cached_backend_status(&so, None),
            CacheStatus::Fresh,
            "a matching fingerprint with fresh mtimes must be fresh"
        );
    }

    /// A missing fingerprint file (a cache predating this check) must defer to
    /// the mtime checks rather than forcing a rebuild, so existing caches are
    /// not thrashed. Here the future-dated `.so` is therefore `Fresh`.
    #[test]
    fn missing_toolchain_fingerprint_defers_to_mtime() {
        let year = Duration::from_secs(365 * 24 * 60 * 60);
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        write_with_mtime(&so, b"built", SystemTime::now() + year);
        // No fingerprint file written.
        assert_eq!(
            cached_backend_status(&so, None),
            CacheStatus::Fresh,
            "absent fingerprint must defer to mtime checks (fresh here)"
        );
    }

    /// The toolchain check has the highest precedence: a differing fingerprint
    /// wins even when the cache is also stale-vs-binary, because an unloadable
    /// `.so` must be re-cloned regardless of why else it is stale.
    #[test]
    fn toolchain_staleness_takes_precedence_over_binary() {
        let year = Duration::from_secs(365 * 24 * 60 * 60);
        let dir = tempdir();
        let so = dir.join("librustc_codegen_cuda.so");
        // Backdate the `.so` so binary-staleness would otherwise fire.
        write_with_mtime(&so, b"built", SystemTime::now() - year);
        std::fs::write(
            dir.join(TOOLCHAIN_FINGERPRINT_FILE),
            "rustc 0.0.0 (deadbeef 1970-01-01)",
        )
        .unwrap();

        assert_eq!(
            cached_backend_status(&so, None),
            CacheStatus::StaleVsToolchain,
            "toolchain mismatch must win over binary staleness"
        );
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
