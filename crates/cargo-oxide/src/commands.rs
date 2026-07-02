/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Command implementations for cargo-oxide.
//!
//! These port the xtask commands with improvements:
//! - Backend path resolved via discovery chain instead of hardcoded relative path
//! - Workspace root resolved by walking up from CWD instead of assuming CWD

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::backend;

/// Project-local cuda-oxide defaults loaded from `.cargo/cuda-oxide.toml`.
#[derive(Debug, Clone, Default)]
pub struct OxideConfig {
    /// Explicit backend shared object path.
    pub backend: Option<PathBuf>,
    /// Default CUDA architecture for codegen commands.
    pub default_arch: Option<String>,
    /// Additional rustflags appended after cuda-oxide's required flags.
    pub extra_rustflags: Vec<String>,
    /// Environment variables applied to child Cargo invocations.
    pub env: Vec<(String, String)>,
}

/// Pre-resolved context shared across all commands.
///
/// Built once at startup by [`resolve_context`] and passed by reference to
/// every command handler. Avoids repeated filesystem walks and backend builds.
pub struct Context {
    /// Absolute path to the workspace root (contains top-level `Cargo.toml`).
    pub workspace_root: PathBuf,
    /// Path to `crates/rustc-codegen-cuda` (backend source tree).
    pub codegen_crate: PathBuf,
    /// Path to `crates/rustc-codegen-cuda/examples/`.
    pub examples_dir: PathBuf,
    /// Path to the built `librustc_codegen_cuda.so` shared object.
    pub backend_so: PathBuf,
    /// True when running from inside the cuda-oxide workspace; false for
    /// standalone projects scaffolded by `cargo oxide new`.
    pub is_workspace: bool,
    /// Project-local cuda-oxide defaults.
    pub config: OxideConfig,
}

/// Resolve the workspace root and backend, or exit with a helpful error.
///
/// Supports two modes:
/// - **Workspace mode**: CWD is inside the cuda-oxide repo (detected by
///   `crates/rustc-codegen-cuda` directory). Examples are resolved from the
///   workspace examples directory.
/// - **Standalone mode**: CWD has a `Cargo.toml` but is not inside the
///   workspace. The backend is located via cache or auto-fetch. Commands
///   like `run` operate on the current directory directly.
pub fn resolve_context() -> Context {
    if let Some(workspace_root) = backend::find_workspace_root() {
        let codegen_crate = workspace_root.join("crates/rustc-codegen-cuda");
        let examples_dir = codegen_crate.join("examples");
        let config = load_oxide_config(&workspace_root);
        let backend_so = backend::find_or_build_backend(&workspace_root, config.backend.as_deref());
        return Context {
            workspace_root,
            codegen_crate,
            examples_dir,
            backend_so,
            is_workspace: true,
            config,
        };
    }

    let cwd = std::env::current_dir().unwrap_or_else(|e| {
        eprintln!("Error: cannot determine current directory: {}", e);
        std::process::exit(1);
    });

    if cwd.join("Cargo.toml").is_file() {
        let config = load_oxide_config(&cwd);
        let backend_so = backend::find_or_build_backend(&cwd, config.backend.as_deref());
        return Context {
            workspace_root: cwd.clone(),
            codegen_crate: cwd.clone(),
            examples_dir: cwd.clone(),
            backend_so,
            is_workspace: false,
            config,
        };
    }

    eprintln!("Error: Could not find cuda-oxide workspace or a standalone Cargo.toml.");
    eprintln!();
    eprintln!("Run from inside the cuda-oxide repository, or from a project created");
    eprintln!("with `cargo oxide new <name>`.");
    std::process::exit(1);
}

/// Resolve a context for `cargo oxide doctor` with NO side effects.
///
/// Identical discovery to [`resolve_context`], except the backend `.so` is
/// only *located* (via [`backend::backend_so_candidate`]), never built and
/// never cloned. A diagnostic command must be runnable on a machine where
/// nothing is set up yet; gating it behind a multi-minute backend build (or
/// a network clone) would hide the very problems it exists to report.
/// `run`/`build`/`pipeline`/`setup` still build the backend on demand.
pub fn resolve_doctor_context() -> Context {
    if let Some(workspace_root) = backend::find_workspace_root() {
        let codegen_crate = workspace_root.join("crates/rustc-codegen-cuda");
        let examples_dir = codegen_crate.join("examples");
        let config = load_oxide_config(&workspace_root);
        let backend_so = backend::backend_so_candidate(&workspace_root, config.backend.as_deref());
        return Context {
            workspace_root,
            codegen_crate,
            examples_dir,
            backend_so,
            is_workspace: true,
            config,
        };
    }

    let cwd = std::env::current_dir().unwrap_or_else(|e| {
        eprintln!("Error: cannot determine current directory: {}", e);
        std::process::exit(1);
    });

    if cwd.join("Cargo.toml").is_file() {
        let config = load_oxide_config(&cwd);
        let backend_so = backend::backend_so_candidate(&cwd, config.backend.as_deref());
        return Context {
            workspace_root: cwd.clone(),
            codegen_crate: cwd.clone(),
            examples_dir: cwd.clone(),
            backend_so,
            is_workspace: false,
            config,
        };
    }

    eprintln!("Error: Could not find cuda-oxide workspace or a standalone Cargo.toml.");
    eprintln!();
    eprintln!("Run from inside the cuda-oxide repository, or from a project created");
    eprintln!("with `cargo oxide new <name>`.");
    std::process::exit(1);
}

// =============================================================================
// Run command
// =============================================================================

/// Build and run an example with the custom codegen backend.
///
/// Cleans stale artifacts, sets encoded rustc flags to point at the backend `.so`,
/// and invokes `cargo run --release` from the example directory. Environment
/// variables control output format (PTX / NVVM IR) and verbosity.
#[allow(clippy::too_many_arguments)]
pub fn codegen_run(
    ctx: &Context,
    example: &str,
    verbose: bool,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
    features: Option<&str>,
    bin: Option<&str>,
    no_fmad: bool,
) {
    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };

    let interop = load_interop_config(&example_dir);

    let output_format = format_label(emit_nvvm_ir);
    let target_arch = configured_arch(ctx, arch);
    // Target precedence for `cargo oxide run` (highest first):
    //   1. --arch <sm_XX>            explicit user override   -> CUDA_OXIDE_TARGET
    //   2. CUDA_OXIDE_TARGET=<sm_XX> explicit env override (from the parent)
    //   3. detected GPU arch (via nvidia-smi) -> CUDA_OXIDE_DEVICE_ARCH (a hint)
    //   4. backend feature-based default (`select_target` in mir-importer)
    //
    // Slot 3 is a HINT, not an override: the backend builds for the detected
    // GPU only when that GPU can run the kernel. If the kernel needs a newer
    // arch (tcgen05 needs sm_100a even on a consumer sm_120 GPU), the backend
    // builds for the required arch and the module simply skips at load time.
    // We only detect for `run`, not `build`/`pipeline`: `run` loads the cubin
    // on the local GPU, whereas those may legitimately cross-compile for
    // another machine.
    let detected_device_arch = detect_run_target_arch(target_arch, emit_nvvm_ir);

    if let Some(interop) = interop.filter(|config| !config.device_crates.is_empty()) {
        codegen_run_interop(
            ctx,
            example,
            &example_dir,
            &interop,
            verbose,
            emit_nvvm_ir,
            target_arch,
            detected_device_arch.as_deref(),
            features,
            bin,
            no_fmad,
        );
        return;
    }

    clean_generated_files(&example_dir, example);

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA: {}", example);
    println!("=========================================");
    println!();
    if emit_nvvm_ir {
        println!("Output format: {}", output_format);
        println!(
            "Target arch: {}",
            configured_arch_label(ctx, arch)
                .expect("--emit-nvvm-ir requires a configured architecture")
        );
        println!();
    } else if let Some(dev) = detected_device_arch.as_deref() {
        // Surface the detected GPU so it isn't silent magic. It is a hint, not
        // a hard target: the backend builds for it unless a kernel needs a
        // newer arch (e.g. tcgen05 forces sm_100a even on a consumer sm_120
        // GPU), so the final PTX target may differ.
        println!("Detected GPU arch: {dev} (via nvidia-smi)");
        println!();
    }
    println!("This is the proper cargo workflow:");
    println!("  CARGO_ENCODED_RUSTFLAGS=<cuda-oxide flags> cargo run");
    println!();

    touch_main_rs(&example_dir);

    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--release"]).current_dir(&example_dir);

    if let Some(bin) = bin {
        cmd.args(["--bin", bin]);
    }
    if let Some(features) = features {
        cmd.args(["--features", features]);
    }

    apply_common_codegen_env(&mut cmd, ctx, verbose, no_fmad);
    apply_codegen_rustflags(&mut cmd, ctx, false, &[]);
    apply_output_mode(&mut cmd, emit_nvvm_ir, target_arch);
    apply_device_arch_hint(&mut cmd, target_arch, detected_device_arch.as_deref());

    if let Some(bin) = bin {
        println!("Building and running {} (bin: {})...", example, bin);
    } else {
        println!("Building and running {}...", example);
    }
    println!();

    let status = cmd.status().expect("Failed to run cargo");
    if !status.success() {
        eprintln!("\nFailed with exit code: {:?}", status.code());
        std::process::exit(status.code().unwrap_or(1));
    }
}

// =============================================================================
// Sanitize command
// =============================================================================

/// Build an example and run the produced host binary under NVIDIA Compute
/// Sanitizer.
#[allow(clippy::too_many_arguments)]
pub fn codegen_sanitize(
    ctx: &Context,
    example: &str,
    tool: &str,
    sanitizer_args: &[String],
    application_args: &[String],
    verbose: bool,
    arch: Option<&str>,
    features: Option<&str>,
    bin: Option<&str>,
    no_fmad: bool,
) {
    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };

    let interop = load_interop_config(&example_dir);
    let target_arch = configured_arch(ctx, arch);
    let detected_device_arch = detect_run_target_arch(target_arch, false);

    if let Some(interop) = interop.filter(|config| !config.device_crates.is_empty()) {
        println!("=========================================");
        println!("RUSTC-CODEGEN-CUDA SANITIZE INTEROP: {}", example);
        println!("=========================================");
        if let Some(kind) = &interop.kind {
            println!("Interop kind: {}", kind);
        }
        if let Some(dev) = detected_device_arch.as_deref() {
            println!("Detected GPU arch: {dev} (via nvidia-smi)");
        }
        println!("Compute Sanitizer tool: {tool}");
        println!();

        build_interop_device_crates(
            ctx,
            &example_dir,
            &interop,
            verbose,
            target_arch,
            detected_device_arch.as_deref(),
            InteropDeviceBuildOptions {
                no_fmad,
                sanitizer_line_tables: true,
            },
        );
        let binary = build_host_cargo(ctx, example, &example_dir, features, bin, verbose);
        run_compute_sanitizer(
            ctx,
            &example_dir,
            tool,
            sanitizer_args,
            application_args,
            &binary,
        );
        return;
    }

    clean_generated_files(&example_dir, example);

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA SANITIZE: {}", example);
    println!("=========================================");
    if let Some(dev) = detected_device_arch.as_deref() {
        println!("Detected GPU arch: {dev} (via nvidia-smi)");
    }
    println!("Compute Sanitizer tool: {tool}");
    println!();

    touch_main_rs(&example_dir);
    let binary = codegen_build_host_binary(
        ctx,
        example,
        &example_dir,
        verbose,
        target_arch,
        detected_device_arch.as_deref(),
        features,
        bin,
        no_fmad,
    );
    run_compute_sanitizer(
        ctx,
        &example_dir,
        tool,
        sanitizer_args,
        application_args,
        &binary,
    );
}

// =============================================================================
// Interop host/device workflow
// =============================================================================

#[derive(Debug, Clone)]
struct InteropConfig {
    kind: Option<String>,
    device_crates: Vec<DeviceCrateConfig>,
}

#[derive(Debug, Clone)]
struct DeviceCrateConfig {
    manifest_path: PathBuf,
    ptx_dir: PathBuf,
    artifact_name: Option<String>,
}

#[derive(Clone, Copy, Debug, Default)]
struct InteropDeviceBuildOptions {
    no_fmad: bool,
    sanitizer_line_tables: bool,
}

impl InteropDeviceBuildOptions {
    fn standard(no_fmad: bool) -> Self {
        Self {
            no_fmad,
            sanitizer_line_tables: false,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn codegen_run_interop(
    ctx: &Context,
    example: &str,
    example_dir: &Path,
    interop: &InteropConfig,
    verbose: bool,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
    detected_device_arch: Option<&str>,
    features: Option<&str>,
    bin: Option<&str>,
    no_fmad: bool,
) {
    reject_interop_nvvm_ir(emit_nvvm_ir);

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA INTEROP: {}", example);
    println!("=========================================");
    if let Some(kind) = &interop.kind {
        println!("Interop kind: {}", kind);
    }
    if let Some(dev) = detected_device_arch {
        println!("Detected GPU arch: {dev} (via nvidia-smi)");
    }
    println!();

    build_interop_device_crates(
        ctx,
        example_dir,
        interop,
        verbose,
        arch,
        detected_device_arch,
        InteropDeviceBuildOptions::standard(no_fmad),
    );
    run_host_cargo(ctx, example, example_dir, "run", features, bin, verbose);
}

#[allow(clippy::too_many_arguments)]
fn codegen_build_interop(
    ctx: &Context,
    example: &str,
    example_dir: &Path,
    interop: &InteropConfig,
    verbose: bool,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
    features: Option<&str>,
    no_fmad: bool,
) {
    reject_interop_nvvm_ir(emit_nvvm_ir);

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA INTEROP BUILD: {}", example);
    println!("=========================================");
    if let Some(kind) = &interop.kind {
        println!("Interop kind: {}", kind);
    }
    println!();

    // `build` may cross-compile for another machine, so no device-arch hint:
    // only an explicit `--arch` pins the target here.
    build_interop_device_crates(
        ctx,
        example_dir,
        interop,
        verbose,
        arch,
        None,
        InteropDeviceBuildOptions::standard(no_fmad),
    );
    run_host_cargo(ctx, example, example_dir, "build", features, None, verbose);
}

fn reject_interop_nvvm_ir(emit_nvvm_ir: bool) {
    if emit_nvvm_ir {
        eprintln!("Error: --emit-nvvm-ir is not supported for metadata interop examples yet.");
        eprintln!("Interop host crates embed PTX artifacts produced by nested device crates.");
        std::process::exit(2);
    }
}

fn build_interop_device_crates(
    ctx: &Context,
    example_dir: &Path,
    interop: &InteropConfig,
    verbose: bool,
    arch: Option<&str>,
    detected_device_arch: Option<&str>,
    options: InteropDeviceBuildOptions,
) {
    for device_crate in &interop.device_crates {
        build_interop_device_crate(
            ctx,
            example_dir,
            device_crate,
            verbose,
            arch,
            detected_device_arch,
            options,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn build_interop_device_crate(
    ctx: &Context,
    example_dir: &Path,
    device_crate: &DeviceCrateConfig,
    verbose: bool,
    arch: Option<&str>,
    detected_device_arch: Option<&str>,
    options: InteropDeviceBuildOptions,
) {
    let manifest_path = example_dir.join(&device_crate.manifest_path);
    let manifest_path = manifest_path.canonicalize().unwrap_or_else(|e| {
        eprintln!(
            "Error: could not resolve device crate manifest {}: {}",
            manifest_path.display(),
            e
        );
        std::process::exit(1);
    });
    let device_dir = manifest_path.parent().unwrap_or(example_dir);
    let ptx_dir = example_dir.join(&device_crate.ptx_dir);
    std::fs::create_dir_all(&ptx_dir).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not create device artifact directory {}: {}",
            ptx_dir.display(),
            e
        );
        std::process::exit(1);
    });

    let package_name = package_name_from_manifest(&manifest_path);
    let artifact_name = device_crate
        .artifact_name
        .clone()
        .unwrap_or_else(|| normalize_crate_name(&package_name));
    clean_generated_files(&ptx_dir, &artifact_name);
    touch_main_rs(device_dir);

    println!("Building device crate {}...", manifest_path.display());

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release", "--manifest-path"])
        .arg(&manifest_path)
        .current_dir(device_dir);

    apply_interop_device_codegen_options(&mut cmd, ctx, verbose, options);
    let fingerprinted_cfgs = options
        .sanitizer_line_tables
        .then(|| {
            sanitize_codegen_fingerprint_cfg(
                ctx,
                verbose,
                options.no_fmad,
                arch,
                detected_device_arch,
                Some(&ptx_dir),
            )
        })
        .into_iter()
        .collect::<Vec<_>>();
    apply_codegen_rustflags(&mut cmd, ctx, false, &fingerprinted_cfgs);
    // This is an internal artifact contract, so it must override a project
    // `[env]` default for the same variable.
    cmd.env("CUDA_OXIDE_PTX_DIR", &ptx_dir);
    apply_output_mode(&mut cmd, false, arch);
    apply_device_arch_hint(&mut cmd, arch, detected_device_arch);

    let status = cmd.status().expect("Failed to build interop device crate");
    if !status.success() {
        eprintln!(
            "\nDevice crate build failed with exit code: {:?}",
            status.code()
        );
        std::process::exit(status.code().unwrap_or(1));
    }

    let ptx_path = ptx_dir.join(format!("{}.ptx", artifact_stem(&artifact_name)));
    if !ptx_path.exists() {
        eprintln!(
            "Error: device crate build succeeded but did not produce {}",
            ptx_path.display()
        );
        std::process::exit(1);
    }
    println!("PTX written: {}", ptx_path.display());
}

fn run_host_cargo(
    ctx: &Context,
    example: &str,
    example_dir: &Path,
    cargo_subcommand: &str,
    features: Option<&str>,
    bin: Option<&str>,
    verbose: bool,
) {
    let mut cmd = Command::new("cargo");
    cmd.arg(cargo_subcommand)
        .arg("--release")
        .current_dir(example_dir);

    if cargo_subcommand == "run"
        && let Some(bin) = bin
    {
        cmd.args(["--bin", bin]);
    }
    if let Some(features) = features {
        cmd.args(["--features", features]);
    }

    apply_config_env(&mut cmd, ctx);
    apply_ld_library_path(&mut cmd, ctx);

    if cargo_subcommand == "run" {
        if let Some(bin) = bin {
            println!("Building and running {} (bin: {})...", example, bin);
        } else {
            println!("Building and running {}...", example);
        }
    } else {
        println!("Building host crate {}...", example);
    }
    println!();

    if verbose {
        cmd.env("CUDA_OXIDE_VERBOSE", "1");
    }

    let status = cmd.status().expect("Failed to run host cargo command");
    if !status.success() {
        eprintln!(
            "\nHost cargo command failed with exit code: {:?}",
            status.code()
        );
        std::process::exit(status.code().unwrap_or(1));
    }
}

#[allow(clippy::too_many_arguments)]
fn codegen_build_host_binary(
    ctx: &Context,
    example: &str,
    example_dir: &Path,
    verbose: bool,
    arch: Option<&str>,
    detected_device_arch: Option<&str>,
    features: Option<&str>,
    bin: Option<&str>,
    no_fmad: bool,
) -> PathBuf {
    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"]).current_dir(example_dir);

    if let Some(bin) = bin {
        cmd.args(["--bin", bin]);
    }
    if let Some(features) = features {
        cmd.args(["--features", features]);
    }

    apply_common_codegen_env(&mut cmd, ctx, verbose, no_fmad);
    apply_default_sanitizer_line_tables(&mut cmd, ctx);
    let fingerprint =
        sanitize_codegen_fingerprint_cfg(ctx, verbose, no_fmad, arch, detected_device_arch, None);
    apply_codegen_rustflags(&mut cmd, ctx, false, &[fingerprint]);
    apply_output_mode(&mut cmd, false, arch);
    apply_device_arch_hint(&mut cmd, arch, detected_device_arch);

    if let Some(bin) = bin {
        println!("Building {} (bin: {})...", example, bin);
    } else {
        println!("Building {}...", example);
    }
    println!();

    let preferred_bin = bin
        .map(str::to_string)
        .or_else(|| preferred_binary_name(&example_dir.join("Cargo.toml")));
    run_cargo_build_for_executable(&mut cmd, preferred_bin.as_deref()).unwrap_or_else(|message| {
        eprintln!("\nBuild failed: {message}");
        std::process::exit(1);
    })
}

fn build_host_cargo(
    ctx: &Context,
    example: &str,
    example_dir: &Path,
    features: Option<&str>,
    bin: Option<&str>,
    verbose: bool,
) -> PathBuf {
    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"]).current_dir(example_dir);

    if let Some(bin) = bin {
        cmd.args(["--bin", bin]);
    }
    if let Some(features) = features {
        cmd.args(["--features", features]);
    }

    apply_config_env(&mut cmd, ctx);
    apply_ld_library_path(&mut cmd, ctx);

    if let Some(bin) = bin {
        println!("Building host crate {} (bin: {})...", example, bin);
    } else {
        println!("Building host crate {}...", example);
    }
    println!();

    if verbose {
        cmd.env("CUDA_OXIDE_VERBOSE", "1");
    }

    let preferred_bin = bin
        .map(str::to_string)
        .or_else(|| preferred_binary_name(&example_dir.join("Cargo.toml")));
    run_cargo_build_for_executable(&mut cmd, preferred_bin.as_deref()).unwrap_or_else(|message| {
        eprintln!("\nHost cargo build failed: {message}");
        std::process::exit(1);
    })
}

fn run_cargo_build_for_executable(
    cmd: &mut Command,
    preferred_bin: Option<&str>,
) -> Result<PathBuf, String> {
    cmd.arg("--message-format=json-render-diagnostics");
    let output = cmd
        .output()
        .map_err(|error| format!("could not start Cargo: {error}"))?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        eprint!("{stderr}");
    }

    let mut executables = BTreeMap::<PathBuf, String>::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let message: serde_json::Value = match serde_json::from_str(line) {
            Ok(message) => message,
            Err(_) => {
                if !line.is_empty() {
                    println!("{line}");
                }
                continue;
            }
        };

        if let Some(rendered) = message
            .get("message")
            .and_then(|message| message.get("rendered"))
            .and_then(|rendered| rendered.as_str())
        {
            eprint!("{rendered}");
        }

        if message.get("reason").and_then(|reason| reason.as_str()) != Some("compiler-artifact") {
            continue;
        }
        let is_binary = message
            .get("target")
            .and_then(|target| target.get("kind"))
            .and_then(|kind| kind.as_array())
            .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_str() == Some("bin")));
        if !is_binary {
            continue;
        }
        let Some(path) = message.get("executable").and_then(|path| path.as_str()) else {
            continue;
        };
        let Some(name) = message
            .get("target")
            .and_then(|target| target.get("name"))
            .and_then(|name| name.as_str())
        else {
            continue;
        };
        executables.insert(PathBuf::from(path), name.to_string());
    }

    if !output.status.success() {
        return Err(format!("Cargo exited with status {}", output.status));
    }

    if executables.len() == 1 {
        return Ok(executables.into_keys().next().expect("one executable"));
    }

    if let Some(preferred_bin) = preferred_bin {
        let mut matches = executables
            .iter()
            .filter(|(_, name)| name.as_str() == preferred_bin)
            .map(|(path, _)| path.clone());
        if let Some(path) = matches.next()
            && matches.next().is_none()
        {
            return Ok(path);
        }
    }

    if executables.is_empty() {
        return Err("Cargo produced no executable binary artifact".to_string());
    }

    let choices = executables
        .iter()
        .map(|(path, name)| format!("{name} ({})", path.display()))
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!(
        "Cargo produced multiple executable binaries: {choices}; pass --bin <name>"
    ))
}

fn preferred_binary_name(manifest_path: &Path) -> Option<String> {
    let source = std::fs::read_to_string(manifest_path).ok()?;
    let document: toml::Value = toml::from_str(&source).ok()?;
    let package = document.get("package")?;
    package
        .get("default-run")
        .or_else(|| package.get("name"))
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

const DEFAULT_SANITIZER_ERROR_EXITCODE: &str = "86";

#[derive(Debug, PartialEq, Eq)]
struct SanitizerInvocationArgs {
    args: Vec<String>,
    uses_default_error_exitcode: bool,
    status_checks_weakened: bool,
}

fn sanitizer_invocation_args(sanitizer_args: &[String]) -> SanitizerInvocationArgs {
    let has_explicit_error_exitcode = sanitizer_args
        .iter()
        .any(|arg| arg == "--error-exitcode" || arg.starts_with("--error-exitcode="));
    if has_explicit_error_exitcode {
        return SanitizerInvocationArgs {
            args: sanitizer_args.to_vec(),
            uses_default_error_exitcode: false,
            status_checks_weakened: sanitizer_option_is_no(sanitizer_args, "check-exit-code")
                || sanitizer_option_is_no(sanitizer_args, "require-cuda-init"),
        };
    }

    let mut args = Vec::with_capacity(sanitizer_args.len() + 2);
    args.extend([
        "--error-exitcode".to_string(),
        DEFAULT_SANITIZER_ERROR_EXITCODE.to_string(),
    ]);
    args.extend_from_slice(sanitizer_args);
    SanitizerInvocationArgs {
        args,
        uses_default_error_exitcode: true,
        status_checks_weakened: sanitizer_option_is_no(sanitizer_args, "check-exit-code")
            || sanitizer_option_is_no(sanitizer_args, "require-cuda-init"),
    }
}

fn sanitizer_option_is_no(args: &[String], name: &str) -> bool {
    let option = format!("--{name}");
    let equals_prefix = format!("{option}=");
    args.iter().enumerate().any(|(index, arg)| {
        arg.strip_prefix(&equals_prefix)
            .is_some_and(|value| value.eq_ignore_ascii_case("no"))
            || (arg == &option
                && args
                    .get(index + 1)
                    .is_some_and(|value| value.eq_ignore_ascii_case("no")))
    })
}

fn run_compute_sanitizer(
    ctx: &Context,
    example_dir: &Path,
    tool: &str,
    sanitizer_args: &[String],
    application_args: &[String],
    binary: &Path,
) {
    let compute_sanitizer = find_cuda_toolkit_executable(
        ctx,
        "compute-sanitizer",
        &[
            "/usr/local/cuda/bin/compute-sanitizer",
            "/opt/cuda/bin/compute-sanitizer",
            "/usr/bin/compute-sanitizer",
        ],
    )
    .unwrap_or_else(|| {
        eprintln!("Error: compute-sanitizer not found.");
        eprintln!(
            "It is installed with the CUDA Toolkit; run `cargo oxide doctor` to check CUDA setup."
        );
        std::process::exit(1);
    });

    let invocation_args = sanitizer_invocation_args(sanitizer_args);
    let mut cmd = Command::new(compute_sanitizer);
    cmd.args(["--tool", tool])
        .args(&invocation_args.args)
        .arg(binary)
        .args(application_args)
        .current_dir(example_dir);
    apply_config_env(&mut cmd, ctx);
    apply_ld_library_path(&mut cmd, ctx);

    let forwarded_args = if invocation_args.args.is_empty() {
        String::new()
    } else {
        format!(" {}", invocation_args.args.join(" "))
    };
    let displayed_application_args = if application_args.is_empty() {
        String::new()
    } else {
        format!(" {}", application_args.join(" "))
    };
    println!(
        "Running compute-sanitizer --tool {tool}{forwarded_args} {}{displayed_application_args}...",
        binary.display()
    );
    println!();

    let status = cmd.status().expect("Failed to run compute-sanitizer");
    if !status.success() {
        eprintln!(
            "\nCompute Sanitizer failed with exit code: {:?}",
            status.code()
        );
        std::process::exit(status.code().unwrap_or(1));
    }

    println!();
    println!("Compute Sanitizer completed with exit code 0.");
    if !invocation_args.uses_default_error_exitcode {
        println!(
            "An explicit --error-exitcode was supplied, so it controls whether findings fail the command."
        );
    }
    if invocation_args.status_checks_weakened {
        println!(
            "The supplied sanitizer options can allow target or CUDA-initialization failures to exit 0."
        );
    }
    println!(
        "Inspect the sanitizer report above; exit status alone is not a clean-report assertion."
    );
}

// =============================================================================
// Build command (compile only, don't run)
// =============================================================================

/// Compile an example without running it.
///
/// Same as [`codegen_run`] but uses `cargo build --release` instead of
/// `cargo run`. Useful for cross-compilation or when the target hardware
/// (e.g., Blackwell tensor cores) isn't available on the build machine.
pub fn codegen_build(
    ctx: &Context,
    example: &str,
    verbose: bool,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
    features: Option<&str>,
    no_fmad: bool,
) {
    let target_arch = configured_arch(ctx, arch);
    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };

    if let Some(interop) =
        load_interop_config(&example_dir).filter(|config| !config.device_crates.is_empty())
    {
        codegen_build_interop(
            ctx,
            example,
            &example_dir,
            &interop,
            verbose,
            emit_nvvm_ir,
            target_arch,
            features,
            no_fmad,
        );
        return;
    }

    clean_generated_files(&example_dir, example);

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA BUILD: {}", example);
    println!("=========================================");
    println!();

    touch_main_rs(&example_dir);

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"]).current_dir(&example_dir);

    if let Some(features) = features {
        cmd.args(["--features", features]);
    }

    apply_common_codegen_env(&mut cmd, ctx, verbose, no_fmad);
    apply_codegen_rustflags(&mut cmd, ctx, false, &[]);
    apply_output_mode(&mut cmd, emit_nvvm_ir, target_arch);

    println!("Building {}...", example);
    println!();

    let status = cmd.status().expect("Failed to run cargo");
    if !status.success() {
        eprintln!("\nBuild failed with exit code: {:?}", status.code());
        std::process::exit(status.code().unwrap_or(1));
    }
}

// =============================================================================
// emit-ltoir command
// =============================================================================

/// Compile a crate's device code to a binary LTOIR artifact in one step.
///
/// `cargo oxide build --emit-nvvm-ir` produces NVVM IR, which a consumer then
/// has to run through libNVVM separately to get linkable LTOIR. This folds both
/// halves into one command for the Tile-to-SIMT interop workflow (#96): it
/// builds the crate in NVVM IR mode, then compiles the emitted `<crate>.ll`
/// with libNVVM `-gen-lto` and writes `<crate>.ltoir` (or `output`) plus the
/// matching `.target` and `.options` files used for runtime loading and final
/// nvJitLink policy.
///
/// `arch` is required because LTOIR is architecture-specific. It accepts
/// `sm_XX`, `compute_XX`, or a bare `XX`, all mapped to libNVVM's
/// `-arch=compute_XX`.
pub fn emit_ltoir(
    ctx: &Context,
    example: &str,
    arch: &str,
    features: Option<&str>,
    output: Option<&Path>,
    verbose: bool,
    no_fmad: bool,
) {
    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };

    if load_interop_config(&example_dir).is_some_and(|config| !config.device_crates.is_empty()) {
        eprintln!("Error: emit-ltoir does not support metadata interop examples.");
        eprintln!("Point it at a single SIMT device crate instead.");
        std::process::exit(1);
    }

    // Normalize once: libNVVM consumes compute_XX, while the compiler records
    // and nvJitLink consumes the equivalent sm_XX spelling.
    let parsed_arch = parse_nvvm_arch(arch).unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(1);
    });
    let sm_arch = parsed_arch.sm();

    // Step 1: build in NVVM IR mode so the backend writes `<crate>.ll` as
    // libNVVM-ready NVVM IR. codegen_build exits on build failure. Pass
    // quiet=true so the intermediate "✓ Build succeeded" line is suppressed;
    // emit_ltoir prints its own unified summary at the end.
    codegen_build(
        ctx,
        example,
        verbose,
        true,
        Some(&sm_arch),
        features,
        no_fmad,
    );

    // Step 2: compile that NVVM IR to LTOIR via libNVVM -gen-lto.
    let ll_path = example_dir.join(format!("{example}.ll"));
    let ir = std::fs::read(&ll_path).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not read emitted NVVM IR at {}: {e}",
            ll_path.display()
        );
        std::process::exit(1);
    });
    let source_options_path = ll_path.with_extension("options");
    let source_options = std::fs::read_to_string(&source_options_path).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not read emitted compile options at {}: {e}",
            source_options_path.display()
        );
        std::process::exit(1);
    });
    let compile_options = oxide_artifacts::ArtifactCompileOptions::from_sidecar_text(
        &source_options,
    )
    .unwrap_or_else(|e| {
        eprintln!(
            "Error: invalid emitted compile options at {}: {e}",
            source_options_path.display()
        );
        std::process::exit(1);
    });

    let compute_arch = parsed_arch.compute();
    let ltoir = compile_nvvm_to_ltoir(
        &ir,
        example,
        &parsed_arch,
        compile_options.fma_contraction_enabled(),
    );

    // Step 3: write the artifact.
    let out_path = output
        .map(Path::to_path_buf)
        .unwrap_or_else(|| example_dir.join(format!("{example}.ltoir")));
    for metadata_path in [
        out_path.with_extension("target"),
        out_path.with_extension("options"),
    ] {
        match std::fs::remove_file(&metadata_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                eprintln!(
                    "Error: could not clear stale LTOIR metadata {}: {error}",
                    metadata_path.display()
                );
                std::process::exit(1);
            }
        }
    }
    std::fs::write(&out_path, &ltoir).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not write LTOIR to {}: {e}",
            out_path.display()
        );
        std::process::exit(1);
    });
    let options_path = out_path.with_extension("options");
    std::fs::write(&options_path, compile_options.sidecar_text()).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not write LTOIR compile options to {}: {e}",
            options_path.display()
        );
        std::process::exit(1);
    });
    let target_path = out_path.with_extension("target");
    std::fs::write(
        &target_path,
        format!(
            "{sm_arch}\n{}\n",
            oxide_artifacts::COMPILE_OPTIONS_TARGET_MARKER
        ),
    )
    .unwrap_or_else(|e| {
        eprintln!(
            "Error: could not write LTOIR target metadata to {}: {e}",
            target_path.display()
        );
        std::process::exit(1);
    });

    println!();
    println!(
        "✓ LTOIR written to {} ({} bytes, {compute_arch})",
        out_path.display(),
        ltoir.len()
    );
}

/// Normalize a target architecture to libNVVM's `compute_XX` form.
///
/// Accepts `sm_XX` (the form `--arch` and the rest of cargo-oxide use),
/// `compute_XX` (passed through), or a bare `XX`.
fn parse_nvvm_arch(arch: &str) -> Result<libnvvm_sys::CudaArch, libnvvm_sys::CudaArchParseError> {
    let normalized = if arch.starts_with("sm_") || arch.starts_with("compute_") {
        arch.to_string()
    } else {
        format!("compute_{arch}")
    };
    normalized.parse()
}

/// Compile NVVM IR text to binary LTOIR with libNVVM `-gen-lto`. Exits with a
/// diagnostic on any libNVVM failure (the program log is attached to the error).
///
fn compile_nvvm_to_ltoir(
    ir: &[u8],
    name: &str,
    arch: &libnvvm_sys::CudaArch,
    allow_fma_contraction: bool,
) -> Vec<u8> {
    let nvvm = libnvvm_sys::LibNvvm::load().unwrap_or_else(|e| {
        eprintln!("Error: could not load libNVVM: {e}");
        eprintln!("libNVVM ships with the CUDA Toolkit at <CUDA>/nvvm/lib64/libnvvm.so.");
        eprintln!("Run `cargo oxide doctor` to check your toolkit setup.");
        std::process::exit(1);
    });
    let ir_version = nvvm.ir_version().unwrap_or_else(|e| {
        eprintln!("Error: could not query libNVVM's accepted IR version: {e}");
        std::process::exit(1);
    });
    if (ir_version.ir_major, ir_version.ir_minor) != (2, 0) {
        eprintln!(
            "Error: installed libNVVM accepts NVVM IR {}.{}, but cuda-oxide emits NVVM IR 2.0",
            ir_version.ir_major, ir_version.ir_minor
        );
        std::process::exit(1);
    }
    if let Some(llvm_major) = nvvm.llvm_version(arch).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not query libNVVM's LLVM dialect for {}: {e}",
            arch.compute()
        );
        std::process::exit(1);
    }) {
        let mismatch = if arch.uses_legacy_llvm() {
            llvm_major != 7
        } else {
            llvm_major == 7
        };
        if mismatch {
            let expected = if arch.uses_legacy_llvm() {
                "legacy LLVM 7 typed-pointer"
            } else {
                "modern opaque-pointer"
            };
            eprintln!(
                "Error: libNVVM reports LLVM {llvm_major} for {}, but cuda-oxide selected the {expected} dialect",
                arch.compute()
            );
            std::process::exit(1);
        }
    }
    let mut program = libnvvm_sys::Program::new(&nvvm).unwrap_or_else(|e| {
        eprintln!("Error: nvvmCreateProgram failed: {e}");
        std::process::exit(1);
    });

    // Add libdevice before the kernel module so any __nv_* math calls (exp,
    // sin, cos, etc.) are resolved at LTOIR compile time, matching the pattern
    // used by NVCC and cuda-host's own LTOIR path.
    let libdevice_path = libnvvm_sys::find_libdevice().unwrap_or_else(|e| {
        eprintln!("Error: could not locate libdevice.10.bc: {e}");
        eprintln!("Set CUDA_OXIDE_LIBDEVICE, CUDA_TOOLKIT_PATH, or CUDA_HOME.");
        std::process::exit(1);
    });
    let libdevice_bytes = std::fs::read(&libdevice_path).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not read libdevice at {}: {e}",
            libdevice_path.display()
        );
        std::process::exit(1);
    });
    program
        .add_module(&libdevice_bytes, "libdevice.10.bc")
        .unwrap_or_else(|e| {
            eprintln!("Error: libNVVM rejected libdevice module: {e}");
            std::process::exit(1);
        });

    program.add_module(ir, name).unwrap_or_else(|e| {
        eprintln!("Error: libNVVM rejected the NVVM IR module: {e}");
        std::process::exit(1);
    });
    let arch_opt = format!("-arch={}", arch.compute());
    program.verify(&[&arch_opt]).unwrap_or_else(|e| {
        eprintln!("Error: libNVVM verification failed: {e}");
        std::process::exit(1);
    });
    let fma_opt = if allow_fma_contraction {
        "-fma=1"
    } else {
        "-fma=0"
    };
    program
        .compile(&[&arch_opt, "-gen-lto", fma_opt])
        .unwrap_or_else(|e| {
            eprintln!("Error: libNVVM -gen-lto compilation failed: {e}");
            std::process::exit(1);
        })
}

/// Options for `cargo oxide build -- ...` / `cargo oxide test -- ...`.
#[derive(Clone, Copy)]
pub struct CargoPassthroughOptions<'a> {
    pub verbose: bool,
    pub emit_nvvm_ir: bool,
    pub arch: Option<&'a str>,
    pub features: Option<&'a str>,
    pub cargo_target_dir: Option<&'a Path>,
    pub device_codegen_crate: Option<&'a str>,
    pub device_cfgs: &'a [String],
    pub no_fmad: bool,
}

fn normalize_device_codegen_crates(raw: &str) -> Result<String, String> {
    let mut normalized = Vec::new();
    for item in raw.split(',') {
        let name = item.trim().replace('-', "_");
        if name.is_empty() {
            return Err(
                "--device-codegen-crate requires a comma-separated list without empty entries"
                    .to_string(),
            );
        }
        if !name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
        {
            return Err(format!(
                "invalid device-codegen crate name `{item}`; use Cargo crate names separated by commas"
            ));
        }
        if !normalized.contains(&name) {
            normalized.push(name);
        }
    }
    Ok(normalized.join(","))
}

fn project_config_env<'a>(ctx: &'a Context, key: &str) -> Option<&'a str> {
    ctx.config
        .env
        .iter()
        .find(|(configured_key, _)| configured_key == key)
        .map(|(_, value)| value.as_str())
}

fn configured_device_codegen_crates(
    ctx: &Context,
    explicit: Option<&str>,
) -> Result<Option<String>, String> {
    let inherited = std::env::var("CUDA_OXIDE_DEVICE_CODEGEN_CRATE").ok();
    resolve_device_codegen_crates(
        explicit,
        inherited.as_deref(),
        project_config_env(ctx, "CUDA_OXIDE_DEVICE_CODEGEN_CRATE"),
    )
}

fn resolve_device_codegen_crates(
    explicit: Option<&str>,
    inherited: Option<&str>,
    configured: Option<&str>,
) -> Result<Option<String>, String> {
    if let Some(explicit) = explicit {
        return normalize_device_codegen_crates(explicit).map(Some);
    }

    inherited
        .or(configured)
        .filter(|value| !value.trim().is_empty())
        .map(normalize_device_codegen_crates)
        .transpose()
}

fn passthrough_codegen_fingerprint(
    ctx: &Context,
    opts: &CargoPassthroughOptions<'_>,
    owner_filter: Option<&str>,
    target_arch: Option<&str>,
) -> String {
    let inherited_env: BTreeMap<String, Option<String>> = std::env::vars_os()
        .filter_map(|(key, value)| {
            key.into_string()
                .ok()
                .map(|key| (key, value.into_string().ok()))
        })
        .collect();
    passthrough_codegen_fingerprint_with_env(ctx, opts, owner_filter, target_arch, &inherited_env)
}

fn passthrough_codegen_fingerprint_with_env(
    ctx: &Context,
    opts: &CargoPassthroughOptions<'_>,
    owner_filter: Option<&str>,
    target_arch: Option<&str>,
    inherited_env: &BTreeMap<String, Option<String>>,
) -> String {
    let mut effective_env = BTreeMap::new();
    effective_env.insert(
        "__CUDA_OXIDE_BACKEND_ARTIFACT".to_string(),
        backend_artifact_identity(&ctx.backend_so),
    );

    // Project-configured CUDA_OXIDE_* variables are defaults. Mirror the same
    // parent override rule as `apply_config_env` so changes that can affect
    // codegen also change Cargo's rustflags fingerprint.
    for (key, configured_value) in &ctx.config.env {
        if !key.starts_with("CUDA_OXIDE_") {
            continue;
        }
        match inherited_env.get(key) {
            Some(Some(value)) => {
                effective_env.insert(key.clone(), value.clone());
            }
            // `apply_config_env` sees the non-Unicode parent value through
            // var_os and does not replace it; backend readers using `var`
            // ignore it, so there is no effective Unicode value to hash.
            Some(None) => {}
            None => {
                effective_env.insert(key.clone(), configured_value.clone());
            }
        }
    }
    // Capture backend settings inherited outside project config, including
    // current and future CUDA_OXIDE_* switches.
    for (key, value) in inherited_env
        .iter()
        .filter(|(key, value)| key.starts_with("CUDA_OXIDE_") && value.is_some())
    {
        effective_env.insert(
            key.clone(),
            value
                .as_ref()
                .expect("filtered to Unicode environment values")
                .clone(),
        );
    }

    if opts.verbose {
        effective_env.insert("CUDA_OXIDE_VERBOSE".to_string(), "1".to_string());
    }
    if opts.no_fmad {
        effective_env.insert("CUDA_OXIDE_NO_FMA".to_string(), "1".to_string());
    }
    if opts.emit_nvvm_ir {
        effective_env.insert("CUDA_OXIDE_EMIT_NVVM_IR".to_string(), "1".to_string());
    }
    if let Some(target_arch) = target_arch {
        effective_env.insert("CUDA_OXIDE_TARGET".to_string(), target_arch.to_string());
    }
    if let Some(owner_filter) = owner_filter {
        effective_env.insert(
            "CUDA_OXIDE_DEVICE_CODEGEN_CRATE".to_string(),
            owner_filter.to_string(),
        );
    }

    // Stable FNV-1a over length-delimited key/value pairs. The cfg carries
    // only the digest, so backend settings are not included verbatim in rustc
    // command lines or diagnostics.
    let mut hash = 0xcbf29ce484222325_u64;
    for (key, value) in effective_env {
        update_codegen_fingerprint_hash(&mut hash, key.as_bytes());
        update_codegen_fingerprint_hash(&mut hash, value.as_bytes());
    }
    format!("{hash:016x}")
}

fn update_codegen_fingerprint_hash(hash: &mut u64, bytes: &[u8]) {
    for byte in (bytes.len() as u64).to_le_bytes().iter().chain(bytes) {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}

/// Put sanitizer-only output settings into an otherwise-unused cfg so Cargo
/// recompiles every selected Rust target, including `src/bin/*` and virtual
/// workspace members. Cargo does not fingerprint arbitrary backend env vars.
fn sanitize_codegen_fingerprint_cfg(
    ctx: &Context,
    verbose: bool,
    no_fmad: bool,
    target_arch: Option<&str>,
    detected_device_arch: Option<&str>,
    ptx_dir: Option<&Path>,
) -> String {
    let opts = CargoPassthroughOptions {
        verbose,
        emit_nvvm_ir: false,
        arch: target_arch,
        features: None,
        cargo_target_dir: None,
        device_codegen_crate: None,
        device_cfgs: &[],
        no_fmad,
    };
    let base = passthrough_codegen_fingerprint(ctx, &opts, None, target_arch);
    let mut hash = 0xcbf29ce484222325_u64;
    for bytes in [
        "sanitize-line-tables-v1".as_bytes(),
        base.as_bytes(),
        detected_device_arch.unwrap_or("").as_bytes(),
    ] {
        update_codegen_fingerprint_hash(&mut hash, bytes);
    }
    if let Some(ptx_dir) = ptx_dir {
        update_codegen_fingerprint_hash(&mut hash, ptx_dir.as_os_str().as_encoded_bytes());
    }
    format!("cuda_oxide_internal_codegen_env=\"{hash:016x}\"")
}

fn backend_artifact_identity(path: &Path) -> String {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let path = canonical.to_string_lossy();
    let Ok(metadata) = std::fs::metadata(&canonical) else {
        return format!("{path}|missing");
    };
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    format!("{path}|{}|{modified}", metadata.len())
}

fn cargo_passthrough_command(
    ctx: &Context,
    cargo_subcommand: &str,
    opts: &CargoPassthroughOptions<'_>,
    cargo_args: &[String],
) -> Result<Command, String> {
    let target_arch = configured_arch(ctx, opts.arch);
    let owner_filter = configured_device_codegen_crates(ctx, opts.device_codegen_crate)?;
    let mut fingerprinted_device_cfgs = opts.device_cfgs.to_vec();
    // Cargo does not fingerprint arbitrary child environment variables. An
    // otherwise-unused cfg makes every effective codegen setting part of the
    // rustc command line, so changing target/output/FMA/filter settings reruns
    // the backend instead of silently reusing stale PTX or NVVM IR.
    let fingerprint =
        passthrough_codegen_fingerprint(ctx, opts, owner_filter.as_deref(), target_arch);
    fingerprinted_device_cfgs.push(format!("cuda_oxide_internal_codegen_env=\"{fingerprint}\""));
    let mut cmd = Command::new("cargo");
    cmd.arg(cargo_subcommand);
    if let Some(features) = opts.features {
        cmd.args(["--features", features]);
    }
    cmd.args(cargo_args).current_dir(&ctx.workspace_root);

    // Project configuration provides defaults. Explicit wrapper flags and
    // internal compiler requirements are applied afterward and therefore win.
    apply_common_codegen_env(&mut cmd, ctx, opts.verbose, opts.no_fmad);
    apply_codegen_rustflags(&mut cmd, ctx, false, &fingerprinted_device_cfgs);

    if let Some(cargo_target_dir) = opts.cargo_target_dir {
        cmd.env("CARGO_TARGET_DIR", cargo_target_dir);
    }
    if let Some(owner_filter) = owner_filter {
        cmd.env("CUDA_OXIDE_DEVICE_CODEGEN_CRATE", owner_filter);
    }
    apply_output_mode(&mut cmd, opts.emit_nvvm_ir, target_arch);
    Ok(cmd)
}

/// Run an arbitrary Cargo build-like subcommand through the cuda-oxide backend.
///
/// Unlike example mode, this does not touch source files or clean generated
/// artifacts. It is intended for final-target workspace builds where Cargo's
/// incremental behavior should remain intact.
pub fn codegen_cargo_passthrough(
    ctx: &Context,
    cargo_subcommand: &str,
    opts: CargoPassthroughOptions<'_>,
    cargo_args: &[String],
) {
    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA CARGO {}", cargo_subcommand);
    println!("=========================================");
    println!();

    let mut cmd = cargo_passthrough_command(ctx, cargo_subcommand, &opts, cargo_args)
        .unwrap_or_else(|error| {
            eprintln!("Error: {error}");
            std::process::exit(2);
        });

    let displayed_args: Vec<_> = cmd
        .get_args()
        .skip(1)
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    if displayed_args.is_empty() {
        println!("Running cargo {}...", cargo_subcommand);
    } else {
        println!(
            "Running cargo {} {}...",
            cargo_subcommand,
            displayed_args.join(" ")
        );
    }
    println!();

    let status = cmd.status().expect("Failed to run cargo");
    if !status.success() {
        eprintln!(
            "\nCargo {} failed with exit code: {:?}",
            cargo_subcommand,
            status.code()
        );
        std::process::exit(status.code().unwrap_or(1));
    }

    println!();
    println!("✓ Cargo {} succeeded", cargo_subcommand);
}

// =============================================================================
// Pipeline command
// =============================================================================

/// Show verbose pipeline progress and the available intermediate artifacts.
///
/// Enables all diagnostic env vars (`CUDA_OXIDE_VERBOSE`, `SHOW_RUSTC_MIR`,
/// `DUMP_MIR`, `DUMP_LLVM`) so the user can see MIR collection, the
/// `dialect-mir` module (pre- and post-`mem2reg`), the LLVM dialect
/// module, textual LLVM IR, and the final PTX or NVVM IR. After the build,
/// generated artifacts are printed to stdout.
pub fn codegen_show_pipeline(
    ctx: &Context,
    example: &str,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
    no_fmad: bool,
) {
    let target_arch = configured_arch(ctx, arch);
    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };

    clean_generated_files(&example_dir, example);

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA PIPELINE: {}", example);
    println!("=========================================");
    println!();
    let target_arch_label = configured_arch_label(ctx, arch);
    match (emit_nvvm_ir, target_arch_label.as_deref()) {
        (true, Some(target_arch)) => println!("Output format: NVVM IR (arch: {})", target_arch),
        (false, Some(target_arch)) => {
            println!("Output format: PTX (arch override: {})", target_arch)
        }
        (false, None) => println!("Output format: PTX (auto-detected arch)"),
        (true, None) => unreachable!("--emit-nvvm-ir requires a configured architecture"),
    }
    println!();
    println!("Required flags (applied via CARGO_ENCODED_RUSTFLAGS):");
    println!("  -C opt-level=3              MIR optimization");
    println!("  -C debug-assertions=off     Remove debug checks");
    println!("  -Z mir-enable-passes=-JumpThreading");
    println!("                              Prevent barrier duplication");
    println!();
    println!("Note: panic=abort is NOT required - the codegen backend treats");
    println!("      unwind paths as unreachable (CUDA toolchain limitation, not HW).");
    println!();

    touch_main_rs(&example_dir);

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"]).current_dir(&example_dir);

    apply_config_env(&mut cmd, ctx);
    apply_codegen_rustflags(&mut cmd, ctx, false, &[]);
    cmd.env("CUDA_OXIDE_VERBOSE", "1");
    cmd.env("CUDA_OXIDE_SHOW_RUSTC_MIR", "1");
    cmd.env("CUDA_OXIDE_DUMP_MIR", "1");
    cmd.env("CUDA_OXIDE_DUMP_LLVM", "1");
    if no_fmad {
        cmd.env("CUDA_OXIDE_NO_FMA", "1");
    }

    apply_output_mode(&mut cmd, emit_nvvm_ir, target_arch);
    apply_ld_library_path(&mut cmd, ctx);

    println!("Building {}...", example);
    println!();

    let status = cmd.status().expect("Failed to run cargo");

    if !status.success() {
        eprintln!("\nBuild failed with exit code: {:?}", status.code());
        std::process::exit(status.code().unwrap_or(1));
    }

    show_generated_artifacts(&example_dir, example);
}

// =============================================================================
// Debug command
// =============================================================================

/// Build with debug info and launch cuda-gdb (or cgdb).
///
/// Compiles the example with `-C debuginfo=2` on top of the normal release
/// flags, then launches the debugger on the resulting binary. Prints a
/// quick-reference cheat sheet for common cuda-gdb commands before handing
/// control to the debugger.
pub fn codegen_debug(
    ctx: &Context,
    example: &str,
    arch: Option<&str>,
    use_cgdb: bool,
    use_tui: bool,
) {
    let cuda_gdb = find_executable(
        "cuda-gdb",
        &[
            "/usr/local/cuda/bin/cuda-gdb",
            "/opt/cuda/bin/cuda-gdb",
            "/usr/bin/cuda-gdb",
        ],
    )
    .unwrap_or_else(|| {
        eprintln!("Error: cuda-gdb not found!");
        eprintln!();
        eprintln!("Make sure CUDA toolkit is installed and cuda-gdb is in your PATH:");
        eprintln!("  export PATH=\"/usr/local/cuda/bin:$PATH\"");
        std::process::exit(1);
    });

    let cgdb_path = if use_cgdb {
        Some(find_executable("cgdb", &[]).unwrap_or_else(|| {
            eprintln!("Error: cgdb not found!");
            eprintln!("Install with: sudo apt install cgdb");
            std::process::exit(1);
        }))
    } else {
        None
    };

    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };

    let target_arch = configured_arch(ctx, arch);
    let detected_device_arch = detect_run_target_arch(target_arch, false);

    println!("Building {} with debug info...", example);
    if let Some(dev) = detected_device_arch.as_deref() {
        println!("Detected GPU arch: {dev} (via nvidia-smi)");
    }

    clean_generated_files(&example_dir, example);

    touch_main_rs(&example_dir);

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"]).current_dir(&example_dir);

    apply_config_env(&mut cmd, ctx);
    apply_codegen_rustflags(&mut cmd, ctx, true, &[]);
    cmd.env("CARGO_PROFILE_RELEASE_DEBUG", "2");
    apply_output_mode(&mut cmd, false, target_arch);
    apply_device_arch_hint(&mut cmd, target_arch, detected_device_arch.as_deref());
    apply_ld_library_path(&mut cmd, ctx);

    let status = cmd.status().expect("Failed to run cargo build");
    if !status.success() {
        eprintln!("Failed to build {}", example);
        std::process::exit(status.code().unwrap_or(1));
    }

    let binary = example_dir.join("target/release").join(example);
    if !binary.exists() {
        eprintln!("Error: Binary not found at {:?}", binary);
        std::process::exit(1);
    }

    if cgdb_path.is_some() {
        println!("Launching cgdb (cuda-gdb frontend)...");
    } else {
        println!(
            "Launching cuda-gdb{}...",
            if use_tui { " (TUI mode)" } else { "" }
        );
    }
    println!();
    println!("Quick reference:");
    println!("  set cuda break_on_launch application");
    println!("                           - Break at start of any kernel");
    println!("  run                      - Start the program");
    println!("  info cuda kernels        - List active kernels");
    println!("  info cuda threads        - List GPU threads");
    println!("  cuda thread (0,0,0)      - Switch to thread");
    println!("  cuda block (0,0,0)       - Switch to block");
    println!("  print <var>              - Print variable");
    println!("  next / step / continue   - Execution control");
    println!("  quit                     - Exit debugger");
    if cgdb_path.is_some() {
        println!();
        println!("cgdb shortcuts:");
        println!("  Esc                      - Focus source window (vim keys work)");
        println!("  i                        - Focus command window");
        println!("  space                    - Set breakpoint on current line");
        println!("  o                        - Open file dialog");
    } else if use_tui {
        println!();
        println!("TUI shortcuts:");
        println!("  Ctrl+x a                 - Toggle TUI mode");
        println!("  Ctrl+x 2                 - Split view (source + asm)");
        println!("  Ctrl+l                   - Refresh screen");
    }
    println!();

    let status = if let Some(cgdb) = cgdb_path {
        Command::new(cgdb)
            .arg("-d")
            .arg(&cuda_gdb)
            .arg(&binary)
            .current_dir(&example_dir)
            .status()
            .expect("Failed to launch cgdb")
    } else {
        let mut gdb_cmd = Command::new(&cuda_gdb);
        if use_tui {
            gdb_cmd.arg("--tui");
        }
        gdb_cmd.arg(&binary);
        gdb_cmd.current_dir(&example_dir);
        gdb_cmd.status().expect("Failed to launch cuda-gdb")
    };

    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
}

// =============================================================================
// Fmt command
// =============================================================================

/// Format (or check formatting of) all crates in the workspace.
///
/// Runs `cargo fmt --all` in three scopes: root workspace, codegen backend
/// crate, and every example that has a `Cargo.toml`. In `check` mode,
/// reports which files need formatting without modifying them.
pub fn format_all(ctx: &Context, check: bool) {
    let mode = if check { "Checking" } else { "Formatting" };
    let mut failed = false;

    println!("📦 {} root workspace...", mode);
    if !run_cargo_fmt(&ctx.workspace_root, check) {
        failed = true;
    }

    println!("📦 {} rustc-codegen-cuda...", mode);
    if !run_cargo_fmt(&ctx.codegen_crate, check) {
        failed = true;
    }

    if let Ok(entries) = std::fs::read_dir(&ctx.examples_dir) {
        let mut examples: Vec<_> = entries.flatten().filter(|e| e.path().is_dir()).collect();
        examples.sort_by_key(|e| e.file_name());

        for entry in examples {
            let example_name = entry.file_name();
            let example_path = entry.path();

            if !example_path.join("Cargo.toml").exists() {
                continue;
            }

            println!("📦 {} example: {}...", mode, example_name.to_string_lossy());
            if !run_cargo_fmt(&example_path, check) {
                failed = true;
            }
        }
    }

    if failed {
        if check {
            eprintln!();
            eprintln!("❌ Some files need formatting. Run: cargo oxide fmt");
        } else {
            eprintln!();
            eprintln!("⚠️  Some formatting commands failed (see above)");
        }
        std::process::exit(1);
    } else {
        println!();
        if check {
            println!("✅ All files are properly formatted");
        } else {
            println!("✅ All crates formatted");
        }
    }
}

/// Run `cargo fmt --all` in a single directory. Returns `true` on success.
fn run_cargo_fmt(dir: &Path, check: bool) -> bool {
    let mut cmd = Command::new("cargo");
    cmd.arg("fmt").arg("--all").current_dir(dir);

    if check {
        cmd.arg("--check");
    }

    match cmd.status() {
        Ok(status) => status.success(),
        Err(e) => {
            eprintln!("  Failed to run cargo fmt: {}", e);
            false
        }
    }
}

// =============================================================================
// Doctor command
// =============================================================================

/// Validate the development environment.
///
/// Checks for: Rust nightly toolchain, `rust-toolchain.toml`, the codegen
/// backend `.so` (informational), CUDA headers (`cuda.h`), CUDA toolkit
/// (`nvcc`, libNVVM, nvJitLink, libdevice), LLVM (`llc`), clang/libclang,
/// the NVIDIA driver / GPU (informational), and optionally `cuda-gdb`.
/// Exits non-zero if any required check fails.
///
/// Doctor itself needs neither the CUDA toolkit nor a driver: every check
/// is a subprocess, a filesystem probe, or a runtime `dlopen`, and the
/// caller resolves the context via [`resolve_doctor_context`] so nothing is
/// built first. This is what lets it diagnose a bare machine (issue #87).
pub fn doctor(ctx: &Context) {
    println!("cargo-oxide environment check");
    println!("==============================");
    println!();

    let mut ok = true;

    // 1. Rust toolchain
    print!("Rust nightly toolchain... ");
    match Command::new("rustc").args(["--version"]).output() {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout);
            let version = version.trim();
            if version.contains("nightly") {
                println!("✓ {}", version);
            } else {
                println!("✗ expected nightly, got: {}", version);
                ok = false;
            }
        }
        _ => {
            println!("✗ rustc not found");
            ok = false;
        }
    }

    // 2. rust-toolchain.toml
    let toolchain_file = ctx.workspace_root.join("rust-toolchain.toml");
    print!("rust-toolchain.toml... ");
    if toolchain_file.exists() {
        println!("✓ present");
    } else {
        println!("✗ not found at {}", toolchain_file.display());
        ok = false;
    }

    // 3. Backend .so. Informational, not fatal: `run`/`build`/`pipeline`
    // build the backend on demand, so "not built yet" is a healthy state
    // for a fresh clone.
    print!("Codegen backend... ");
    if ctx.backend_so.exists() {
        println!("✓ {}", ctx.backend_so.display());
    } else {
        println!("- not built yet (run `cargo oxide setup`)");
    }

    // 4. CUDA headers (cuda.h). The host `cuda-bindings` crate cannot build
    // without them; cargo-oxide itself deliberately can, which is what makes
    // this check reachable on a toolkit-less machine instead of dying inside
    // cuda-bindings' build script (issue #87).
    print!("CUDA headers (cuda.h)... ");
    let toolkit = cuda_toolkit_root(|var| std::env::var(var).ok());
    let header_candidates = cuda_header_candidates(&toolkit, std::env::consts::ARCH);
    match header_candidates.iter().find(|path| path.is_file()) {
        Some(found) => println!("✓ {}", found.display()),
        None => {
            println!("✗ not found in the CUDA toolkit at `{}`", toolkit);
            eprintln!("  Probed:");
            for candidate in &header_candidates {
                eprintln!("    {}", candidate.display());
            }
            eprintln!("  Host crates (cuda-bindings) cannot build without cuda.h. Set");
            eprintln!("  CUDA_TOOLKIT_PATH or CUDA_HOME to a CUDA Toolkit install root;");
            eprintln!("  when neither is set, /usr/local/cuda is used.");
            ok = false;
        }
    }

    // 5. CUDA toolkit
    print!("CUDA toolkit (nvcc)... ");
    match Command::new("nvcc").arg("--version").output() {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = version.lines().find(|l| l.contains("release")) {
                println!("✓ {}", line.trim());
            } else {
                println!("✓ (version unknown)");
            }
        }
        _ => {
            println!("✗ nvcc not found");
            ok = false;
        }
    }

    // 5b. libNVVM + nvJitLink + libdevice (only required when a kernel uses
    // CUDA libdevice math, e.g. sin/cos/exp/pow). All three ship with the
    // CUDA Toolkit; checking them here surfaces missing or split packagings
    // before a runtime failure inside `cuda_host::ltoir::load_kernel_module`.
    print!("libNVVM (libnvvm.so)... ");
    match libnvvm_sys::LibNvvm::load() {
        Ok(nvvm) => match nvvm.version() {
            Ok((major, minor)) => println!("✓ libNVVM {}.{}", major, minor),
            Err(_) => println!("✓ (version query failed but library loaded)"),
        },
        Err(e) => {
            println!("✗ {}", e);
            eprintln!("  Required only when kernels call CUDA libdevice math");
            eprintln!("  (sin/cos/exp/pow/...). Ships with the CUDA Toolkit at");
            eprintln!("  <CUDA>/nvvm/lib64/libnvvm.so. No separate download.");
            ok = false;
        }
    }

    print!("nvJitLink (libnvJitLink.so)... ");
    match nvjitlink_sys::LibNvJitLink::load() {
        Ok(nvj) => match nvj.version() {
            Some((major, minor)) => println!("✓ nvJitLink {}.{}", major, minor),
            None => println!("✓ (version symbol not exported on this CTK)"),
        },
        Err(e) => {
            println!("✗ {}", e);
            eprintln!("  Required only when kernels call CUDA libdevice math.");
            eprintln!("  Ships with the CUDA Toolkit at <CUDA>/lib64/libnvJitLink.so.");
            ok = false;
        }
    }

    print!("libdevice (libdevice.10.bc)... ");
    match libnvvm_sys::find_libdevice() {
        Ok(path) => println!("✓ {}", path.display()),
        Err(e) => {
            println!("✗ {}", e);
            eprintln!("  Required only when kernels call CUDA libdevice math.");
            eprintln!("  Ships with the CUDA Toolkit at");
            eprintln!("  <CUDA>/nvvm/libdevice/libdevice.10.bc. Override the search");
            eprintln!("  with `CUDA_OXIDE_LIBDEVICE=<path>` if you have it elsewhere.");
            ok = false;
        }
    }

    // 6. llc (LLVM static compiler for PTX)
    //
    // cuda-oxide requires LLVM 21+: earlier releases reject modern TMA /
    // tcgen05 / WGMMA intrinsic signatures. Probe in the same order as the
    // pipeline:
    //   1. `CUDA_OXIDE_LLC` (caller-supplied override)
    //   2. Rust toolchain's `llvm-tools` component (auto-installed via rustup)
    //   3. `llc-22`, `llc-21`, `llc` on `PATH`
    // Whatever we pick, reject if the major version is < 21.
    print!("llc (LLVM)... ");

    // The pipeline's primary entry: the `llc` bundled with the pinned Rust
    // toolchain's `llvm-tools` component. Built with the NVPTX backend
    // enabled, so the typical novice path is `rustup component add llvm-tools`
    // and that's it. Surface the absolute path so doctor's output matches
    // what the pipeline actually invokes.
    let rustup_llc_path: Option<String> = Command::new("rustc")
        .args(["--print", "sysroot", "--print", "host-tuple"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|output| {
            let stdout = String::from_utf8(output.stdout).ok()?;
            let mut lines = stdout.lines();
            let sysroot = lines.next()?;
            let host = lines.next()?;
            let path: std::path::PathBuf = [sysroot, "lib", "rustlib", host, "bin", "llc"]
                .iter()
                .collect();
            path.is_file()
                .then(|| path.to_str().map(str::to_string))
                .flatten()
        });

    let mut candidates: Vec<String> = Vec::new();
    if let Ok(env_llc) = std::env::var("CUDA_OXIDE_LLC") {
        candidates.push(env_llc);
    }
    if let Some(rustup) = rustup_llc_path.clone() {
        candidates.push(rustup);
    }
    for name in ["llc-22", "llc-21", "llc"] {
        candidates.push(name.to_string());
    }

    let llc_pick = candidates.iter().find_map(|candidate| {
        Command::new(candidate)
            .arg("--version")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| {
                (
                    candidate.clone(),
                    String::from_utf8_lossy(&o.stdout).into_owned(),
                )
            })
    });
    match llc_pick {
        Some((binary, stdout)) => {
            let banner = stdout
                .lines()
                .find(|l| l.contains("LLVM version"))
                .unwrap_or("(version unknown)")
                .trim()
                .to_string();
            let major = banner
                .split("LLVM version")
                .nth(1)
                .and_then(|rest| rest.trim().split('.').next())
                .and_then(|s| s.parse::<u32>().ok());
            match major {
                Some(v) if v >= 21 => println!("✓ {} ({})", banner, binary),
                Some(v) => {
                    println!("✗ {} ({}) — need LLVM 21+", banner, binary);
                    eprintln!(
                        "  Your `{}` reports LLVM {}, which rejects the TMA / tcgen05 /",
                        binary, v
                    );
                    eprintln!("  WGMMA intrinsic signatures cuda-oxide emits. Install a newer");
                    eprintln!("  toolchain (`rustup component add llvm-tools` is usually enough,");
                    eprintln!("  or `sudo apt install llvm-21`) and either add it to PATH or set");
                    eprintln!("  `CUDA_OXIDE_LLC=/path/to/llc`.");
                    ok = false;
                }
                None => println!("✓ {} ({}, version could not be parsed)", banner, binary),
            }
        }
        None => {
            println!("✗ llc not found");
            eprintln!("  cuda-oxide probes (in order): $CUDA_OXIDE_LLC, the Rust toolchain's");
            eprintln!("  llvm-tools llc, then llc-22/llc-21/llc on PATH. Easiest fix:");
            eprintln!("    rustup component add llvm-tools");
            eprintln!("  Alternative: `sudo apt install llvm-21` (older versions reject");
            eprintln!("  modern TMA / tcgen05 / WGMMA intrinsics).");
            ok = false;
        }
    }

    // 7. clang / libclang resource dir (host `cuda-bindings` / bindgen)
    //
    // The host `cuda-bindings` crate's build.rs runs bindgen, which loads
    // libclang at runtime to parse `wrapper.h`. That parse pulls in
    // `<stddef.h>`, which must be served from clang's own resource
    // directory — the system/GCC copy is not compatible. Fresh installs of
    // bare `libclang1-*` (without the matching `libclang-common-*-dev`)
    // leave `/usr/lib/clang/*/include` empty and bindgen explodes with a
    // mysterious "'stddef.h' file not found". Catch that up front.
    print!("clang / libclang resource dir... ");
    let clang_resource_dir = Command::new("clang")
        .arg("-print-resource-dir")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    match clang_resource_dir {
        Some(ref dir) if std::path::Path::new(&format!("{}/include/stddef.h", dir)).exists() => {
            println!("✓ {}", dir);
        }
        Some(ref dir) => {
            println!(
                "✗ resource dir present but `include/stddef.h` missing: {}",
                dir
            );
            eprintln!("  Host `cuda-bindings` uses bindgen, which needs clang's own stddef.h.");
            eprintln!("  Install the matching dev headers: sudo apt install clang-21");
            eprintln!("  (or libclang-common-21-dev)");
            ok = false;
        }
        None => {
            println!("✗ clang not found");
            eprintln!(
                "  Host `cuda-bindings` uses bindgen, which needs clang + its resource headers."
            );
            eprintln!("  Install with: sudo apt install clang-21");
            eprintln!("  (or at minimum `libclang-common-21-dev` alongside your libclang)");
            ok = false;
        }
    }

    // 8. NVIDIA driver / GPU. Informational, not fatal: only `cargo oxide
    // run` (kernel execution) needs a driver. Cross-compiling and GPU-less
    // CI boxes are supported workflows (`build`/`pipeline` work fine), and
    // the examples-compile CI job is exactly that.
    print!("NVIDIA driver / GPU... ");
    match query_gpu_name_and_compute_cap() {
        Some((name, (major, minor))) => {
            println!("✓ {} (compute capability {}.{})", name, major, minor);
        }
        None => {
            // Some containers mount the kernel driver without shipping
            // nvidia-smi; /proc distinguishes "driver loaded, tool broken"
            // from "no driver at all".
            if Path::new("/proc/driver/nvidia/version").exists() {
                println!("- driver loaded, but nvidia-smi is missing or not reporting a GPU");
                eprintln!("  A kernel-mode NVIDIA driver is present (/proc/driver/nvidia/");
                eprintln!("  version), but `nvidia-smi` did not report a usable GPU.");
                eprintln!("  `cargo oxide run` may still work; arch auto-detection will fall");
                eprintln!("  back to the backend default (override with --arch=<sm_XX>).");
            } else {
                println!("- no NVIDIA driver detected");
                eprintln!("  Only `cargo oxide run` (kernel execution) needs the driver;");
                eprintln!("  `cargo oxide build` and `pipeline` work without one.");
            }
        }
    }

    // 9. cuda-gdb (optional)
    print!("cuda-gdb (optional)... ");
    match Command::new("cuda-gdb").arg("--version").output() {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = version.lines().next() {
                println!("✓ {}", line.trim());
            } else {
                println!("✓");
            }
        }
        _ => {
            println!("- not found (only needed for `cargo oxide debug`)");
        }
    }

    println!();
    if ok {
        println!("✅ Environment looks good!");
    } else {
        println!("❌ Some checks failed. Fix the issues above and re-run `cargo oxide doctor`.");
        std::process::exit(1);
    }
}

/// CUDA toolkit install root for doctor's `cuda.h` probe: the first set
/// variable among `CUDA_TOOLKIT_PATH`, `CUDA_HOME`, else `/usr/local/cuda`.
///
/// Kept in lockstep BY HAND with `crates/cuda-bindings/build.rs`
/// (`cuda_toolkit_dir` / `find_cuda_include_dir` / `toolkit_target_dir`):
/// doctor cannot import that probe because build.rs logic is not a library,
/// and cuda-bindings is the NVIDIA-proprietary crate cargo-oxide must not
/// depend on. If the build.rs discovery changes, mirror it here.
fn cuda_toolkit_root(mut get_env: impl FnMut(&str) -> Option<String>) -> String {
    ["CUDA_TOOLKIT_PATH", "CUDA_HOME"]
        .iter()
        .find_map(|var| get_env(var).filter(|value| !value.trim().is_empty()))
        .unwrap_or_else(|| "/usr/local/cuda".to_string())
}

/// Candidate `cuda.h` paths under `toolkit`, in probe order: the standard
/// `include/` layout first, then the redistributable `targets/<dir>/include`
/// layout. CUDA names the target dirs after the GPU platform, not the Rust
/// triple: x86_64 hosts use `x86_64-linux`, aarch64 servers use `sbsa-linux`.
///
/// `arch` is the host CPU architecture; the caller passes
/// `std::env::consts::ARCH` (doctor runs at runtime, so there is no cargo
/// `TARGET` to consult). Injected as a parameter for unit tests.
fn cuda_header_candidates(toolkit: &str, arch: &str) -> Vec<PathBuf> {
    let base = Path::new(toolkit);
    let mut candidates = vec![base.join("include/cuda.h")];
    let target_dir = match arch {
        "x86_64" => Some("x86_64-linux"),
        "aarch64" => Some("sbsa-linux"),
        _ => None,
    };
    if let Some(dir) = target_dir {
        candidates.push(base.join("targets").join(dir).join("include/cuda.h"));
    }
    candidates
}

// =============================================================================
// Setup command
// =============================================================================

/// Explicitly build (or rebuild) the codegen backend.
///
/// Normally the backend is built automatically on every `run`/`build`/`pipeline`
/// invocation. `setup` exists for first-time setup, CI, or after pulling new
/// changes when you want to rebuild without running an example.
pub fn setup(ctx: &Context) {
    println!("Building cuda-oxide codegen backend...");
    println!();

    backend::build_backend_from_source(&ctx.codegen_crate);

    println!();
    println!("✓ Backend is ready. You can now use:");
    println!("  cargo oxide run <example>");
    println!("  cargo oxide build <example>");
}

// =============================================================================
// Helpers
// =============================================================================

fn load_oxide_config(workspace_root: &Path) -> OxideConfig {
    let config_path = workspace_root.join(".cargo/cuda-oxide.toml");
    if !config_path.exists() {
        return OxideConfig::default();
    }

    let source = std::fs::read_to_string(&config_path).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not read cuda-oxide config {}: {}",
            config_path.display(),
            e
        );
        std::process::exit(1);
    });
    let document: toml::Value = toml::from_str(&source).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not parse cuda-oxide config {}: {}",
            config_path.display(),
            e
        );
        std::process::exit(1);
    });
    let table = document.as_table().unwrap_or_else(|| {
        eprintln!(
            "Error: cuda-oxide config {} must be a TOML table",
            config_path.display()
        );
        std::process::exit(1);
    });

    let backend = optional_config_string(table, "backend", &config_path)
        .map(PathBuf::from)
        .map(|path| absolutize_config_path(path, &config_path));
    let default_arch = optional_config_string(table, "default-arch", &config_path);
    let extra_rustflags = optional_config_string_array(table, "extra-rustflags", &config_path);
    let env = table
        .get("env")
        .map(|value| parse_config_env(value, &config_path))
        .unwrap_or_default();

    OxideConfig {
        backend,
        default_arch,
        extra_rustflags,
        env,
    }
}

fn absolutize_config_path(path: PathBuf, config_path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path;
    }
    config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(path)
}

fn optional_config_string(table: &toml::Table, key: &str, config_path: &Path) -> Option<String> {
    table.get(key).map(|value| {
        value.as_str().map(str::to_string).unwrap_or_else(|| {
            eprintln!(
                "Error: cuda-oxide config {} field `{}` must be a string",
                config_path.display(),
                key
            );
            std::process::exit(1);
        })
    })
}

fn optional_config_string_array(table: &toml::Table, key: &str, config_path: &Path) -> Vec<String> {
    table
        .get(key)
        .map(|value| {
            value
                .as_array()
                .unwrap_or_else(|| {
                    eprintln!(
                        "Error: cuda-oxide config {} field `{}` must be an array of strings",
                        config_path.display(),
                        key
                    );
                    std::process::exit(1);
                })
                .iter()
                .map(|item| {
                    item.as_str().map(str::to_string).unwrap_or_else(|| {
                        eprintln!(
                            "Error: cuda-oxide config {} field `{}` must be an array of strings",
                            config_path.display(),
                            key
                        );
                        std::process::exit(1);
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_config_env(value: &toml::Value, config_path: &Path) -> Vec<(String, String)> {
    let table = value.as_table().unwrap_or_else(|| {
        eprintln!(
            "Error: cuda-oxide config {} field `env` must be a table of strings",
            config_path.display()
        );
        std::process::exit(1);
    });
    let mut env: Vec<_> = table
        .iter()
        .map(|(key, value)| {
            let value = value.as_str().unwrap_or_else(|| {
                eprintln!(
                    "Error: cuda-oxide config {} env value `{}` must be a string",
                    config_path.display(),
                    key
                );
                std::process::exit(1);
            });
            (key.clone(), value.to_string())
        })
        .collect();
    env.sort_by(|left, right| left.0.cmp(&right.0));
    env
}

fn load_interop_config(example_dir: &Path) -> Option<InteropConfig> {
    let manifest_path = example_dir.join("Cargo.toml");
    let source = std::fs::read_to_string(&manifest_path).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not read manifest {}: {}",
            manifest_path.display(),
            e
        );
        std::process::exit(1);
    });
    let document: toml::Value = toml::from_str(&source).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not parse manifest {}: {}",
            manifest_path.display(),
            e
        );
        std::process::exit(1);
    });

    let oxide = document
        .get("package")
        .and_then(|value| value.get("metadata"))
        .and_then(|value| value.get("cuda-oxide"))?;

    let kind = oxide.get("interop").and_then(|value| {
        value.as_str().map(str::to_string).or_else(|| {
            value
                .get("kind")
                .and_then(|kind| kind.as_str())
                .map(str::to_string)
        })
    });

    let device_crates = oxide
        .get("device-crates")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .map(|item| parse_device_crate_config(item, &manifest_path))
                .collect()
        })
        .unwrap_or_default();

    Some(InteropConfig {
        kind,
        device_crates,
    })
}

fn parse_device_crate_config(value: &toml::Value, manifest_path: &Path) -> DeviceCrateConfig {
    let table = value.as_table().unwrap_or_else(|| {
        eprintln!(
            "Error: each package.metadata.cuda-oxide.device-crates entry in {} must be a table",
            manifest_path.display()
        );
        std::process::exit(1);
    });

    let device_manifest = required_metadata_string(table, "manifest-path", manifest_path);
    let ptx_dir = optional_metadata_string(table, "ptx-dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(&device_manifest)
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        });
    let artifact_name = optional_metadata_string(table, "artifact-name");

    DeviceCrateConfig {
        manifest_path: PathBuf::from(device_manifest),
        ptx_dir,
        artifact_name,
    }
}

fn required_metadata_string(table: &toml::Table, key: &str, manifest_path: &Path) -> String {
    optional_metadata_string(table, key).unwrap_or_else(|| {
        eprintln!(
            "Error: package.metadata.cuda-oxide.device-crates entry in {} is missing string field `{}`",
            manifest_path.display(),
            key
        );
        std::process::exit(1);
    })
}

fn optional_metadata_string(table: &toml::Table, key: &str) -> Option<String> {
    table
        .get(key)
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

fn package_name_from_manifest(manifest_path: &Path) -> String {
    let source = std::fs::read_to_string(manifest_path).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not read device manifest {}: {}",
            manifest_path.display(),
            e
        );
        std::process::exit(1);
    });
    let document: toml::Value = toml::from_str(&source).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not parse device manifest {}: {}",
            manifest_path.display(),
            e
        );
        std::process::exit(1);
    });

    document
        .get("package")
        .and_then(|value| value.get("name"))
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| {
            eprintln!(
                "Error: device manifest {} is missing package.name",
                manifest_path.display()
            );
            std::process::exit(1);
        })
}

fn normalize_crate_name(package_name: &str) -> String {
    package_name.replace('-', "_")
}

/// Resolve an example name to its directory path, or exit with a list of
/// available examples if not found.
fn resolve_example_dir(ctx: &Context, example: &str) -> PathBuf {
    let example_dir = ctx.examples_dir.join(example);
    if !example_dir.exists() {
        eprintln!("Error: Example not found: {}", example_dir.display());
        eprintln!();
        eprintln!("Available examples:");
        if let Ok(entries) = std::fs::read_dir(&ctx.examples_dir) {
            let mut names: Vec<_> = entries
                .flatten()
                .filter(|e| e.path().is_dir())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect();
            names.sort();
            for name in names {
                eprintln!("  - {}", name);
            }
        }
        std::process::exit(1);
    }
    example_dir
}

const ENCODED_RUSTFLAGS_SEPARATOR: char = '\u{1f}';

/// Construct boundary-preserving rustc flags for Cargo.
///
/// `RUSTFLAGS` is whitespace-split by Cargo, which corrupts a single flag
/// containing spaces. `CARGO_ENCODED_RUSTFLAGS` uses unit separators and keeps
/// every configured array element and `--device-cfg` value intact.
fn build_encoded_rustflags(ctx: &Context, debug: bool, device_cfgs: &[String]) -> String {
    let existing_encoded = std::env::var("CARGO_ENCODED_RUSTFLAGS").ok();
    let existing = std::env::var("RUSTFLAGS").ok();
    let mut explicit_rustflags = Vec::new();
    for cfg in device_cfgs {
        explicit_rustflags.push("--cfg".to_string());
        explicit_rustflags.push(cfg.clone());
    }
    build_encoded_rustflags_with_existing(
        &ctx.backend_so,
        debug,
        &ctx.config.extra_rustflags,
        &explicit_rustflags,
        existing_encoded.as_deref(),
        existing.as_deref(),
    )
}

fn build_encoded_rustflags_with_existing(
    backend_so: &Path,
    debug: bool,
    configured_rustflags: &[String],
    explicit_rustflags: &[String],
    existing_encoded_rustflags: Option<&str>,
    existing_rustflags: Option<&str>,
) -> String {
    // Project flags are defaults, inherited flags are user overrides, and
    // explicit wrapper flags are stronger. cuda-oxide's compiler invariants
    // come last because rustc resolves repeated -C/-Z options last-one-wins.
    let mut flags = configured_rustflags.to_vec();

    if let Some(existing) = existing_encoded_rustflags {
        flags.extend(
            existing
                .split(ENCODED_RUSTFLAGS_SEPARATOR)
                .filter(|flag| !flag.is_empty())
                .map(str::to_string),
        );
    } else if let Some(existing) = existing_rustflags {
        // Match Cargo's legacy RUSTFLAGS behavior when converting it to the
        // encoded representation.
        flags.extend(existing.split_whitespace().map(str::to_string));
    }
    flags.extend(explicit_rustflags.iter().cloned());
    flags.extend([
        format!("-Zcodegen-backend={}", backend_so.display()),
        "-Copt-level=3".to_string(),
        "-Cdebug-assertions=off".to_string(),
        "-Zmir-enable-passes=-JumpThreading".to_string(),
        "-Csymbol-mangling-version=v0".to_string(),
    ]);
    if debug {
        flags.push("-Cdebuginfo=2".to_string());
    }
    flags.join(&ENCODED_RUSTFLAGS_SEPARATOR.to_string())
}

fn apply_codegen_rustflags(cmd: &mut Command, ctx: &Context, debug: bool, device_cfgs: &[String]) {
    cmd.env(
        "CARGO_ENCODED_RUSTFLAGS",
        build_encoded_rustflags(ctx, debug, device_cfgs),
    )
    .env_remove("RUSTFLAGS");
}

/// Set environment variables for the codegen backend.
///
/// `arch` is an explicit pin (`--arch`); it becomes `CUDA_OXIDE_TARGET`, the
/// hard override the backend honors as-is. The auto-detected GPU arch is *not*
/// routed here -- see [`apply_device_arch_hint`].
fn apply_output_mode(cmd: &mut Command, emit_nvvm_ir: bool, arch: Option<&str>) {
    if let Some(target_arch) = arch {
        cmd.env("CUDA_OXIDE_TARGET", target_arch);
    }
    if emit_nvvm_ir {
        cmd.env("CUDA_OXIDE_EMIT_NVVM_IR", "1");
    }
}

fn configured_arch<'a>(ctx: &'a Context, cli_arch: Option<&'a str>) -> Option<&'a str> {
    if cli_arch.is_some() || std::env::var("CUDA_OXIDE_TARGET").is_ok() {
        cli_arch
    } else {
        ctx.config
            .default_arch
            .as_deref()
            .or_else(|| project_config_env(ctx, "CUDA_OXIDE_TARGET"))
    }
}

fn configured_arch_label(ctx: &Context, cli_arch: Option<&str>) -> Option<String> {
    cli_arch
        .map(str::to_string)
        .or_else(|| std::env::var("CUDA_OXIDE_TARGET").ok())
        .or_else(|| ctx.config.default_arch.clone())
        .or_else(|| project_config_env(ctx, "CUDA_OXIDE_TARGET").map(str::to_string))
}

pub fn has_configured_arch(ctx: &Context, cli_arch: Option<&str>) -> bool {
    cli_arch.is_some()
        || std::env::var("CUDA_OXIDE_TARGET").is_ok()
        || ctx.config.default_arch.is_some()
        || project_config_env(ctx, "CUDA_OXIDE_TARGET").is_some()
}

fn apply_config_env(cmd: &mut Command, ctx: &Context) {
    for (key, value) in &ctx.config.env {
        if matches!(key.as_str(), "RUSTFLAGS" | "CARGO_ENCODED_RUSTFLAGS") {
            continue;
        }
        // Project values are defaults. An explicitly inherited environment is
        // stronger, and command-specific CLI/internal settings are applied
        // after this helper and are stronger still.
        if std::env::var_os(key).is_none() {
            cmd.env(key, value);
        }
    }
}

fn apply_common_codegen_env(cmd: &mut Command, ctx: &Context, verbose: bool, no_fmad: bool) {
    apply_config_env(cmd, ctx);
    if verbose {
        cmd.env("CUDA_OXIDE_VERBOSE", "1");
    }
    if no_fmad {
        cmd.env("CUDA_OXIDE_NO_FMA", "1");
    }
    apply_ld_library_path(cmd, ctx);
}

/// Give Compute Sanitizer source line attribution without disabling normal
/// device optimization. An explicit process or project setting remains
/// authoritative, including an intentional `CUDA_OXIDE_DEBUG=off`.
fn apply_default_sanitizer_line_tables(cmd: &mut Command, ctx: &Context) {
    if std::env::var_os("CUDA_OXIDE_DEBUG").is_none()
        && project_config_env(ctx, "CUDA_OXIDE_DEBUG").is_none()
    {
        cmd.env("CUDA_OXIDE_DEBUG", "line-tables");
    }
}

fn apply_interop_device_codegen_options(
    cmd: &mut Command,
    ctx: &Context,
    verbose: bool,
    options: InteropDeviceBuildOptions,
) {
    apply_common_codegen_env(cmd, ctx, verbose, options.no_fmad);
    if options.sanitizer_line_tables {
        apply_default_sanitizer_line_tables(cmd, ctx);
    }
}

/// Forward the auto-detected GPU arch as a *hint* via `CUDA_OXIDE_DEVICE_ARCH`.
///
/// Unlike `CUDA_OXIDE_TARGET` (a hard override), this is advisory: the backend
/// builds for the detected GPU only when that GPU can actually run the kernel.
/// If the kernel needs a newer arch (e.g. tcgen05 / cta_group TMA multicast
/// need sm_100a, which a consumer sm_120 GPU lacks), the backend builds for the
/// required arch instead. Skipped when the user pinned `--arch` (that explicit
/// choice already went to `CUDA_OXIDE_TARGET`).
fn apply_device_arch_hint(
    cmd: &mut Command,
    explicit_arch: Option<&str>,
    detected_device_arch: Option<&str>,
) {
    if let (None, Some(dev)) = (explicit_arch, detected_device_arch) {
        cmd.env("CUDA_OXIDE_DEVICE_ARCH", dev);
    }
}

/// Pick a runnable target for `cargo oxide run` when the user has not pinned
/// one explicitly.
///
/// # Precedence
///
/// `cargo oxide run` resolves the target architecture in this order, highest
/// priority first:
///
/// 1. `--arch <sm_XX>`            (explicit user override)
/// 2. `CUDA_OXIDE_TARGET=<sm_XX>` (explicit env override, set in the parent
///    process before invoking `cargo oxide run`)
/// 3. **This function**: the compute capability of the first GPU reported by
///    `nvidia-smi`, forwarded as the `CUDA_OXIDE_DEVICE_ARCH` *hint*. Emits
///    the arch-specific `sm_XYa` form for cc >= 9.0 (so the backend can lower
///    WGMMA / tcgen05 / TMA-multicast when the GPU supports them) and the
///    plain `sm_XY` form for cc < 9.0.
/// 4. Backend feature-based default (`select_target` in
///    `mir-importer::pipeline`), which picks the minimum `sm_XX` required by
///    the IR shape (e.g. `Basic -> sm_80`, `Cluster -> sm_90`, `Tma -> sm_100`).
///
/// Slot 3 is advisory: the backend builds for the detected GPU only when that
/// GPU can run the kernel, otherwise it falls back to slot 4 (the arch the
/// kernel requires). This function returns `Some(sm_XY[a])` to fill slot 3, or
/// `None` (falling through to slot 4) when the machine has no usable GPU.
///
/// # Why only `run`
///
/// `run` immediately loads the generated module on the local GPU and launches
/// the kernel, so a target older than the local GPU's compute capability is
/// the only safe default. `build` and `pipeline` may legitimately
/// cross-compile to a different machine, so they keep the backend's
/// feature-based default untouched.
///
/// # Why this is needed even with the backend default
///
/// The backend's `select_target` picks the minimum `sm_XX` the IR requires.
/// `Basic → sm_80` is a fine *compilation* baseline, but PTX for `sm_80` will
/// not load on a Turing (`sm_75`) GPU because the JIT refuses
/// forward-incompatible PTX. Detecting the device CC in `run` keeps the
/// generated module loadable on the actual hardware that will execute it.
///
/// # When this returns `None`
///
/// - The user passed `--arch` (slot 1 wins).
/// - `CUDA_OXIDE_TARGET` is set in the environment (slot 2 wins).
/// - `--emit-nvvm-ir` is in effect (NVVM IR mode requires explicit `--arch`,
///   enforced by the CLI parser).
/// - No CUDA driver / GPU is available on the machine (CI runners without
///   GPUs, headless build boxes), or `nvidia-smi` is missing or broken. The
///   caller falls through to slot 4 and the backend's feature-based default
///   applies.
fn detect_run_target_arch(arch: Option<&str>, emit_nvvm_ir: bool) -> Option<String> {
    if arch.is_some() || emit_nvvm_ir || std::env::var_os("CUDA_OXIDE_TARGET").is_some() {
        return None;
    }

    query_device_compute_cap().map(format_sm_arch)
}

/// Query the compute capability of the first GPU via `nvidia-smi`.
///
/// Runs `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` and parses
/// the first output line. A subprocess probe (rather than the CUDA driver
/// API) keeps cargo-oxide free of any link-time or dlopen dependency on
/// `libcuda`, so the subcommand builds and runs on machines with no CUDA
/// toolkit and no driver; `scripts/smoketest.sh` derives `sm_XX` from
/// `nvidia-smi` the same way.
///
/// Caveat: `nvidia-smi` enumerates GPUs in PCI bus order, while CUDA's
/// default device order is fastest-first, so on heterogeneous multi-GPU
/// machines this may describe a different GPU than CUDA device 0. That is
/// safe because `CUDA_OXIDE_DEVICE_ARCH` is advisory (the backend only
/// honors a compatible hint) and `--arch` / `CUDA_OXIDE_TARGET` remain hard
/// overrides.
fn query_device_compute_cap() -> Option<(u32, u32)> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    parse_compute_cap(&String::from_utf8_lossy(&output.stdout))
}

/// Parse the first line of `nvidia-smi --query-gpu=compute_cap` output as a
/// `(major, minor)` compute-capability pair. Returns `None` for anything
/// that is not shaped `<digits>.<digits>`.
fn parse_compute_cap(stdout: &str) -> Option<(u32, u32)> {
    parse_compute_cap_field(stdout.lines().next()?)
}

/// Parse a single `compute_cap` CSV field (e.g. `"12.0"`).
///
/// Only the `<digits>.<digits>` shape is accepted: `nvidia-smi` prints its
/// failure banners ("NVIDIA-SMI has failed ...") to *stdout*, sometimes with
/// exit status 0, so this shape check is the real gate, not the exit status.
fn parse_compute_cap_field(field: &str) -> Option<(u32, u32)> {
    let (major, minor) = field.trim().split_once('.')?;
    let all_digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if !all_digits(major) || !all_digits(minor) {
        return None;
    }
    Some((major.parse().ok()?, minor.parse().ok()?))
}

/// Query the name and compute capability of the first GPU via `nvidia-smi`,
/// for doctor's driver / GPU report. Same trust rules as
/// [`query_device_compute_cap`].
fn query_gpu_name_and_compute_cap() -> Option<(String, (u32, u32))> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name,compute_cap", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    parse_gpu_name_and_compute_cap(&String::from_utf8_lossy(&output.stdout))
}

/// Parse the first line of `nvidia-smi --query-gpu=name,compute_cap` output
/// into the GPU name and `(major, minor)` pair. Splits on the LAST comma:
/// GPU names may contain commas in principle, `compute_cap` never does.
fn parse_gpu_name_and_compute_cap(stdout: &str) -> Option<(String, (u32, u32))> {
    let line = stdout.lines().next()?;
    let (name, cap) = line.rsplit_once(',')?;
    Some((name.trim().to_string(), parse_compute_cap_field(cap)?))
}

/// Format a `(major, minor)` compute-capability tuple as the `sm_XX` /
/// `sm_XXX[a]` string the codegen backend expects on `CUDA_OXIDE_TARGET`.
///
/// Concatenates without a separator, matching CUDA conventions:
/// `(7, 5)` → `"sm_75"`, `(12, 0)` → `"sm_120a"`.
///
/// # Arch-specific (`a`) suffix
///
/// Compute capability ≥ 9.0 always has an arch-specific PTX target (`sm_90a`,
/// `sm_100a`, `sm_103a`, `sm_120a`, …) that is a strict superset of the plain
/// target on that chip. The `a` form is what unlocks WGMMA on Hopper and
/// `tcgen05` / TMA multicast / `cta_group::*` on Blackwell datacenter — and
/// every chip that reports cc ≥ 9.0 *is* the `a`-variant chip in NVIDIA's
/// lineup (there is no consumer Hopper, no non-`a` sm_100, and so on).
///
/// This helper is only used by [`detect_run_target_arch`] in `cargo oxide
/// run`, where the local GPU is known exactly and no cross-compile is in
/// flight. Emitting the `a` form there:
///
/// - **No false negatives:** kernels that need `tcgen05` / WGMMA compile and
///   load on that GPU (was: silent fallback to `sm_100` / `sm_90` and a
///   `ptxas: 'tcgen05.alloc' not supported on .target 'sm_100'` failure).
/// - **No false positives:** cc < 9.0 keeps the plain `sm_XY` form, since
///   there is no `sm_80a` / `sm_86a` / `sm_89a` target in the PTX ISA.
/// - **Strict superset:** PTX targeting `sm_XYa` accepts every kernel that
///   would have compiled for plain `sm_XY`; the `a` form only permits
///   *additional* arch-specific intrinsics.
fn format_sm_arch((major, minor): (u32, u32)) -> String {
    if major >= 9 {
        format!("sm_{}{}a", major, minor)
    } else {
        format!("sm_{}{}", major, minor)
    }
}

fn inherited_or_configured_env(ctx: &Context, key: &str) -> Option<String> {
    std::env::var(key).ok().or_else(|| {
        ctx.config
            .env
            .iter()
            .find(|(configured_key, _)| configured_key == key)
            .map(|(_, value)| value.clone())
    })
}

/// Build `LD_LIBRARY_PATH` for the child cargo process.
///
/// Includes the rustc sysroot lib (for `librustc_driver.so` etc.), the
/// libmathdx lib (when `LIBMATHDX_PATH` is set), and any existing
/// `LD_LIBRARY_PATH` from the parent environment.
fn apply_ld_library_path(cmd: &mut Command, ctx: &Context) {
    let mut ld_paths: Vec<String> = Vec::new();
    if let Some(sysroot) = backend::get_rustc_sysroot() {
        ld_paths.push(format!("{}/lib", sysroot));
    }
    if let Some(libmathdx_path) = inherited_or_configured_env(ctx, "LIBMATHDX_PATH") {
        ld_paths.push(format!("{}/lib", libmathdx_path));
    }
    if let Some(existing) = inherited_or_configured_env(ctx, "LD_LIBRARY_PATH") {
        ld_paths.push(existing);
    }
    if !ld_paths.is_empty() {
        cmd.env("LD_LIBRARY_PATH", ld_paths.join(":"));
    }
}

/// Touch main.rs to force recompilation (faster than cargo clean).
fn touch_main_rs(example_dir: &Path) {
    // Force a rebuild so the codegen backend re-runs and emits a fresh
    // .ptx alongside the example. Touch every source file that might
    // host `#[kernel]` items so multi-bin layouts (kernels in `lib.rs`,
    // tests in `main.rs`, perf bench in `bin/<name>.rs`, etc.) all
    // re-codegen on every `cargo oxide run/build` invocation.
    for rel in ["src/main.rs", "src/lib.rs"] {
        let path = example_dir.join(rel);
        if path.exists()
            && let Ok(content) = std::fs::read(&path)
        {
            let _ = std::fs::write(&path, content);
        }
    }
}

/// Artifacts are named after the crate, and cargo normalizes hyphens in
/// package names to underscores (`rustlantis-smoke` emits
/// `rustlantis_smoke.ptx`). Always go through this when deriving an
/// artifact filename from an example name, or hyphenated examples keep
/// stale artifacts forever.
fn artifact_stem(example: &str) -> String {
    example.replace('-', "_")
}

/// Remove stale generated artifacts (`.ptx`, `.ll`, `.ltoir`, `.cubin`) from a
/// previous run so we can verify the build produces fresh output.
fn clean_generated_files(example_dir: &Path, example: &str) {
    let stem = artifact_stem(example);
    for ext in &[
        "ptx",
        "ll",
        "opt.ll",
        "ltoir",
        "cubin",
        "target",
        "options",
        "cubin.target",
    ] {
        let file = example_dir.join(format!("{}.{}", stem, ext));
        if file.exists() {
            let _ = std::fs::remove_file(&file);
        }
    }
}

/// Human-readable label for the selected output format.
fn format_label(emit_nvvm_ir: bool) -> &'static str {
    if emit_nvvm_ir { "NVVM IR" } else { "PTX" }
}

/// Print generated artifacts (LLVM IR or PTX) to stdout after a pipeline build.
fn show_generated_artifacts(example_dir: &Path, example: &str) {
    let stem = artifact_stem(example);
    let ll_file = example_dir.join(format!("{}.ll", stem));
    let ptx_file = example_dir.join(format!("{}.ptx", stem));

    if ll_file.exists() {
        println!();
        println!("=========================================");
        println!("LLVM IR ({}.ll)", stem);
        println!("=========================================");
        if let Ok(content) = std::fs::read_to_string(&ll_file) {
            println!("{}", content);
        }
    }

    if ptx_file.exists() {
        println!();
        println!("=========================================");
        println!("PTX ({}.ptx)", stem);
        println!("=========================================");
        if let Ok(content) = std::fs::read_to_string(&ptx_file) {
            println!("{}", content);
        }
    }
}

// =========================================================================
// cargo oxide new -- standalone project scaffolding
// =========================================================================

const GIT_REPO: &str = "https://github.com/NVlabs/cuda-oxide.git";

const RUST_TOOLCHAIN_TOML: &str = r#"[toolchain]
channel = "nightly-2026-04-03"
components = ["rust-src", "rustc-dev", "rust-analyzer", "clippy", "llvm-tools"]
"#;

/// Scaffold a new standalone cuda-oxide project.
pub fn scaffold_new(name: &str, async_mode: bool) {
    let project_dir = PathBuf::from(name);
    if project_dir.exists() {
        eprintln!("Error: directory '{}' already exists.", name);
        std::process::exit(1);
    }

    let src_dir = project_dir.join("src");
    std::fs::create_dir_all(&src_dir).unwrap_or_else(|e| {
        eprintln!("Error creating directory: {}", e);
        std::process::exit(1);
    });

    let cargo_toml = if async_mode {
        format!(
            r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2024"

[workspace]

[dependencies]
cuda-device = {{ git = "{GIT_REPO}" }}
cuda-host = {{ git = "{GIT_REPO}", features = ["async"] }}
cuda-core = {{ git = "{GIT_REPO}" }}
cuda-async = {{ git = "{GIT_REPO}" }}
cuda-bindings = {{ git = "{GIT_REPO}" }}
tokio = {{ version = "1", features = ["rt", "rt-multi-thread", "macros"] }}
"#
        )
    } else {
        format!(
            r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2024"

[workspace]

[dependencies]
cuda-device = {{ git = "{GIT_REPO}" }}
cuda-host = {{ git = "{GIT_REPO}" }}
cuda-core = {{ git = "{GIT_REPO}" }}
"#
        )
    };

    let main_rs = if async_mode {
        r#"use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::cuda_module;
use cuda_async::device_context::init_device_contexts;
use cuda_async::device_operation::DeviceOperation;
use cuda_core::LaunchConfig;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::device_box::DeviceBox;
    use cuda_core::memory::{malloc_async, memcpy_dtoh_async, memcpy_htod_async};
    use std::mem;

    init_device_contexts(0, 1)?;
    let module = kernels::load_async(0)?;

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let (a_dev, b_dev, mut c_dev) = cuda_async::device_context::with_cuda_context(0, |ctx| {
        let stream = ctx.default_stream();
        let num_bytes = N * mem::size_of::<f32>();
        unsafe {
            let a = malloc_async(stream.cu_stream(), num_bytes).unwrap();
            let b = malloc_async(stream.cu_stream(), num_bytes).unwrap();
            let c = malloc_async(stream.cu_stream(), num_bytes).unwrap();
            memcpy_htod_async(a, a_host.as_ptr(), num_bytes, stream.cu_stream()).unwrap();
            memcpy_htod_async(b, b_host.as_ptr(), num_bytes, stream.cu_stream()).unwrap();
            stream.synchronize().unwrap();
            (
                DeviceBox::<[f32]>::from_raw_parts(a, N, 0),
                DeviceBox::<[f32]>::from_raw_parts(b, N, 0),
                DeviceBox::<[f32]>::from_raw_parts(c, N, 0),
            )
        }
    })?;

    module
        .vecadd_async(
            LaunchConfig::for_num_elems(N as u32),
            &a_dev,
            &b_dev,
            &mut c_dev,
        )?
        .sync()?;

    let mut c_host = vec![0.0f32; N];
    cuda_async::device_context::with_cuda_context(0, |ctx| {
        let stream = ctx.default_stream();
        unsafe {
            memcpy_dtoh_async(
                c_host.as_mut_ptr(),
                c_dev.cu_deviceptr(),
                N * mem::size_of::<f32>(),
                stream.cu_stream(),
            )
            .unwrap();
            stream.synchronize().unwrap();
        }
    })?;

    let errors = (0..N)
        .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
        .count();

    if errors == 0 {
        println!("PASSED: all {} elements correct", N);
    } else {
        eprintln!("FAILED: {} errors", errors);
        std::process::exit(1);
    }

    Ok(())
}
"#
        .to_string()
    } else {
        r#"use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::cuda_module;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }
}
fn main() {
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    module
        .vecadd(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &a_dev,
            &b_dev,
            &mut c_dev,
        )
        .expect("Kernel launch failed");

    let c_host = c_dev.to_host_vec(&stream).unwrap();

    let errors = (0..N)
        .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
        .count();

    if errors == 0 {
        println!("PASSED: all {} elements correct", N);
    } else {
        eprintln!("FAILED: {} errors", errors);
        std::process::exit(1);
    }
}
"#
        .to_string()
    };

    std::fs::write(project_dir.join("Cargo.toml"), cargo_toml).expect("Failed to write Cargo.toml");
    std::fs::write(project_dir.join("rust-toolchain.toml"), RUST_TOOLCHAIN_TOML)
        .expect("Failed to write rust-toolchain.toml");
    std::fs::write(src_dir.join("main.rs"), main_rs).expect("Failed to write src/main.rs");

    let mode = if async_mode { " (async)" } else { "" };
    println!("✓ Created cuda-oxide project '{}'{}", name, mode);
    println!();
    println!("  cd {}", name);
    println!("  cargo oxide run {}", name);
}

/// Locate an executable by name, first via `which` (PATH lookup), then by
/// checking a list of common fallback absolute paths.
fn find_executable(name: &str, fallback_paths: &[&str]) -> Option<PathBuf> {
    if let Ok(output) = Command::new("which").arg(name).output()
        && output.status.success()
    {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    for path in fallback_paths {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Locate a CUDA Toolkit executable using the same configured toolkit roots as
/// `doctor`, after the user's PATH and before generic system fallbacks.
fn find_cuda_toolkit_executable(
    ctx: &Context,
    name: &str,
    fallback_paths: &[&str],
) -> Option<PathBuf> {
    if let Some(path) = find_executable(name, &[]) {
        return Some(path);
    }

    let toolkit = cuda_toolkit_root(|key| {
        std::env::var(key)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| project_config_env(ctx, key).map(str::to_owned))
    });
    let configured = PathBuf::from(toolkit).join("bin").join(name);
    if configured.exists() {
        return Some(configured);
    }

    for path in fallback_paths {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    fn command_env(cmd: &Command, key: &str) -> Option<String> {
        cmd.get_envs()
            .find(|(name, _)| *name == OsStr::new(key))
            .and_then(|(_, value)| value.map(|v| v.to_string_lossy().into_owned()))
    }

    fn decoded_rustflags(encoded: &str) -> Vec<&str> {
        encoded.split(ENCODED_RUSTFLAGS_SEPARATOR).collect()
    }

    fn has_codegen_env_fingerprint(flags: &[&str]) -> bool {
        flags.windows(2).any(|pair| {
            pair[0] == "--cfg"
                && pair[1].starts_with("cuda_oxide_internal_codegen_env=\"")
                && pair[1].ends_with('"')
        })
    }

    fn test_context(config: OxideConfig) -> Context {
        Context {
            workspace_root: PathBuf::from("/tmp/cargo-oxide-test-workspace"),
            codegen_crate: PathBuf::from("/tmp/cargo-oxide-test-codegen"),
            examples_dir: PathBuf::from("/tmp/cargo-oxide-test-examples"),
            backend_so: PathBuf::from("/tmp/backend path/librustc_codegen_cuda.so"),
            is_workspace: false,
            config,
        }
    }

    #[test]
    fn artifact_stem_normalizes_hyphens_like_cargo() {
        assert_eq!(artifact_stem("rustlantis-smoke"), "rustlantis_smoke");
        assert_eq!(artifact_stem("vecadd"), "vecadd");
    }

    #[test]
    fn generated_file_cleanup_preserves_ltoir_cubin_cache() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_clean_cache_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&root).unwrap();
        for extension in ["ptx", "ll", "ltoir", "cubin", "target"] {
            std::fs::write(root.join(format!("my_kernel.{extension}")), b"stale").unwrap();
        }
        let cached_cubin =
            root.join(".oxide-artifacts/ltoir-cubin-cache/v1/entries/key/image.cubin");
        std::fs::create_dir_all(cached_cubin.parent().unwrap()).unwrap();
        std::fs::write(&cached_cubin, b"persistent cache entry").unwrap();

        clean_generated_files(&root, "my-kernel");

        for extension in ["ptx", "ll", "ltoir", "cubin", "target"] {
            assert!(!root.join(format!("my_kernel.{extension}")).exists());
        }
        assert_eq!(
            std::fs::read(&cached_cubin).unwrap(),
            b"persistent cache entry"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn preferred_binary_name_prefers_default_run() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cargo_oxide_default_run_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&root).unwrap();
        let manifest = root.join("Cargo.toml");
        std::fs::write(
            &manifest,
            r#"
[package]
name = "multi-bin-package"
default-run = "main_bin"
version = "0.1.0"
edition = "2024"
"#,
        )
        .unwrap();

        assert_eq!(
            preferred_binary_name(&manifest).as_deref(),
            Some("main_bin")
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cargo_json_selects_custom_bin_in_configured_target_dir() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cargo_oxide_sanitize_binary_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(root.join(".cargo")).unwrap();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            r#"
[package]
name = "package-bin"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "actual-bin"
path = "src/main.rs"
"#,
        )
        .unwrap();
        std::fs::write(root.join("src/main.rs"), "fn main() {}\n").unwrap();
        std::fs::write(
            root.join(".cargo/config.toml"),
            "[build]\ntarget-dir = \"configured-target\"\n",
        )
        .unwrap();

        let mut cmd = Command::new("cargo");
        cmd.args(["build", "--release"]).current_dir(&root);
        let binary = run_cargo_build_for_executable(&mut cmd, Some("package-bin")).unwrap();

        assert!(binary.exists());
        let expected_name = format!("actual-bin{}", std::env::consts::EXE_SUFFIX);
        assert_eq!(
            binary.file_name().and_then(OsStr::to_str),
            Some(expected_name.as_str())
        );
        assert!(
            binary
                .components()
                .any(|part| part.as_os_str() == "configured-target")
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cargo_json_selects_single_binary_from_virtual_workspace() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cargo_oxide_sanitize_workspace_{}_{}",
            std::process::id(),
            unique
        ));
        let member = root.join("member");
        std::fs::create_dir_all(member.join("src")).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"member\"]\nresolver = \"2\"\n",
        )
        .unwrap();
        std::fs::write(
            member.join("Cargo.toml"),
            r#"
[package]
name = "workspace-package"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "workspace-bin"
path = "src/main.rs"
"#,
        )
        .unwrap();
        std::fs::write(member.join("src/main.rs"), "fn main() {}\n").unwrap();

        assert_eq!(preferred_binary_name(&root.join("Cargo.toml")), None);
        let mut cmd = Command::new("cargo");
        cmd.args(["build", "--release"]).current_dir(&root);
        let binary = run_cargo_build_for_executable(&mut cmd, None).unwrap();

        let expected_name = format!("workspace-bin{}", std::env::consts::EXE_SUFFIX);
        assert_eq!(
            binary.file_name().and_then(OsStr::to_str),
            Some(expected_name.as_str())
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sanitizer_adds_nonzero_error_exitcode_by_default() {
        let invocation =
            sanitizer_invocation_args(&["--leak-check".to_string(), "full".to_string()]);

        assert_eq!(
            invocation.args,
            ["--error-exitcode", "86", "--leak-check", "full"]
        );
        assert!(invocation.uses_default_error_exitcode);
        assert!(!invocation.status_checks_weakened);
    }

    #[test]
    fn sanitizer_preserves_explicit_zero_error_exitcode_without_claiming_detection() {
        let separated = sanitizer_invocation_args(&[
            "--error-exitcode".to_string(),
            "0".to_string(),
            "--leak-check".to_string(),
        ]);
        let equals = sanitizer_invocation_args(&["--error-exitcode=0".to_string()]);
        let repeated = sanitizer_invocation_args(&[
            "--error-exitcode=86".to_string(),
            "--error-exitcode=0".to_string(),
        ]);

        assert_eq!(separated.args, ["--error-exitcode", "0", "--leak-check"]);
        assert!(!separated.uses_default_error_exitcode);
        assert!(!separated.status_checks_weakened);
        assert_eq!(equals.args, ["--error-exitcode=0"]);
        assert!(!equals.uses_default_error_exitcode);
        assert_eq!(repeated.args, ["--error-exitcode=86", "--error-exitcode=0"]);
        assert!(!repeated.uses_default_error_exitcode);
    }

    #[test]
    fn sanitizer_detects_options_that_weaken_success_status() {
        for args in [
            vec!["--check-exit-code=no".to_string()],
            vec!["--check-exit-code".to_string(), "no".to_string()],
            vec!["--require-cuda-init=no".to_string()],
            vec!["--require-cuda-init".to_string(), "NO".to_string()],
        ] {
            let invocation = sanitizer_invocation_args(&args);
            assert!(invocation.status_checks_weakened, "{args:?}");
        }
    }

    #[test]
    fn sanitize_interop_codegen_defaults_to_line_tables_and_forwards_no_fmad() {
        let ctx = test_context(OxideConfig::default());
        let mut cmd = Command::new("cargo");

        apply_interop_device_codegen_options(
            &mut cmd,
            &ctx,
            false,
            InteropDeviceBuildOptions {
                no_fmad: true,
                sanitizer_line_tables: true,
            },
        );

        assert_eq!(command_env(&cmd, "CUDA_OXIDE_NO_FMA").as_deref(), Some("1"));
        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_DEBUG").as_deref(),
            Some("line-tables")
        );

        let fingerprint = sanitize_codegen_fingerprint_cfg(
            &ctx,
            false,
            true,
            Some("sm_80"),
            None,
            Some(Path::new("/tmp/generated-ptx")),
        );
        apply_codegen_rustflags(&mut cmd, &ctx, false, &[fingerprint]);
        let encoded = command_env(&cmd, "CARGO_ENCODED_RUSTFLAGS").unwrap();
        assert!(has_codegen_env_fingerprint(&decoded_rustflags(&encoded)));
    }

    #[test]
    fn standard_interop_codegen_forwards_no_fmad_without_debug_override() {
        let ctx = test_context(OxideConfig::default());
        let mut cmd = Command::new("cargo");

        apply_interop_device_codegen_options(
            &mut cmd,
            &ctx,
            false,
            InteropDeviceBuildOptions::standard(true),
        );

        assert_eq!(command_env(&cmd, "CUDA_OXIDE_NO_FMA").as_deref(), Some("1"));
        assert_eq!(command_env(&cmd, "CUDA_OXIDE_DEBUG"), None);
    }

    #[test]
    fn sanitize_fingerprint_tracks_output_affecting_settings() {
        let ctx = test_context(OxideConfig::default());
        let base = sanitize_codegen_fingerprint_cfg(&ctx, false, false, None, Some("sm_80"), None);

        for changed in [
            sanitize_codegen_fingerprint_cfg(&ctx, false, true, None, Some("sm_80"), None),
            sanitize_codegen_fingerprint_cfg(&ctx, false, false, None, Some("sm_90"), None),
            sanitize_codegen_fingerprint_cfg(&ctx, false, false, Some("sm_80"), None, None),
            sanitize_codegen_fingerprint_cfg(
                &ctx,
                false,
                false,
                None,
                Some("sm_80"),
                Some(Path::new("/tmp/generated-ptx")),
            ),
        ] {
            assert_ne!(base, changed);
        }
    }

    #[test]
    fn sanitizer_tool_lookup_uses_project_cuda_toolkit_root() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cargo_oxide_sanitizer_tool_{}_{}",
            std::process::id(),
            unique
        ));
        let tool = root.join("bin/cuda-oxide-test-sanitizer");
        std::fs::create_dir_all(tool.parent().unwrap()).unwrap();
        std::fs::write(&tool, b"fake tool").unwrap();
        let ctx = test_context(OxideConfig {
            env: vec![(
                "CUDA_TOOLKIT_PATH".to_string(),
                root.to_string_lossy().into_owned(),
            )],
            ..OxideConfig::default()
        });

        assert_eq!(
            find_cuda_toolkit_executable(&ctx, "cuda-oxide-test-sanitizer", &[]),
            Some(tool)
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn encoded_rustflags_preserve_inherited_flags_but_required_flags_win() {
        let rustflags = build_encoded_rustflags_with_existing(
            Path::new("/tmp/librustc_codegen_cuda.so"),
            false,
            &[],
            &[],
            Some(
                "-Lnative=/nix/store/cuda-cudart/lib\u{1f}-Copt-level=0\u{1f}-Zcodegen-backend=llvm",
            ),
            Some("-L native=/nix/store/cuda-cudart/lib"),
        );
        let flags = decoded_rustflags(&rustflags);

        assert_eq!(flags[0], "-Lnative=/nix/store/cuda-cudart/lib");
        assert!(flags.contains(&"-Copt-level=0"));
        assert!(flags.contains(&"-Zcodegen-backend=llvm"));
        assert_eq!(
            &flags[flags.len() - 5..],
            [
                "-Zcodegen-backend=/tmp/librustc_codegen_cuda.so",
                "-Copt-level=3",
                "-Cdebug-assertions=off",
                "-Zmir-enable-passes=-JumpThreading",
                "-Csymbol-mangling-version=v0",
            ]
        );
        assert!(!flags.contains(&"native=/nix/store/cuda-cudart/lib"));
    }

    #[test]
    fn encoded_rustflags_preserve_configured_flag_boundaries_and_spaces() {
        let rustflags = build_encoded_rustflags_with_existing(
            Path::new("/tmp/backend path/librustc_codegen_cuda.so"),
            false,
            &["--cfg".to_string(), "model=\"alpha beta\"".to_string()],
            &[],
            None,
            Some("-L native=/nix/store/cuda-cudart/lib"),
        );
        let flags = decoded_rustflags(&rustflags);

        assert!(
            flags
                .windows(2)
                .any(|pair| pair == ["--cfg", "model=\"alpha beta\""])
        );
        assert_eq!(&flags[2..4], ["-L", "native=/nix/store/cuda-cudart/lib"]);
        assert_eq!(
            flags[flags.len() - 5],
            "-Zcodegen-backend=/tmp/backend path/librustc_codegen_cuda.so"
        );
    }

    #[test]
    fn encoded_rustflags_ignore_empty_existing_flags() {
        let rustflags = build_encoded_rustflags_with_existing(
            Path::new("/tmp/librustc_codegen_cuda.so"),
            true,
            &[],
            &[],
            None,
            Some(""),
        );
        let flags = decoded_rustflags(&rustflags);

        assert!(flags.contains(&"-Cdebuginfo=2"));
        assert!(!flags.contains(&""));
    }

    #[test]
    fn project_config_parser_loads_backend_arch_flags_and_env() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cargo_oxide_config_test_{}_{}",
            std::process::id(),
            unique
        ));
        let cargo_dir = root.join(".cargo");
        std::fs::create_dir_all(&cargo_dir).unwrap();
        std::fs::write(
            cargo_dir.join("cuda-oxide.toml"),
            r#"
backend = "../backend/librustc_codegen_cuda.so"
default-arch = "sm_90"
extra-rustflags = ["--cfg", "model=\"alpha beta\""]

[env]
MY_BUILD_FLAG = "configured"
"#,
        )
        .unwrap();

        let config = load_oxide_config(&root);
        assert_eq!(
            config.backend,
            Some(cargo_dir.join("../backend/librustc_codegen_cuda.so"))
        );
        assert_eq!(config.default_arch.as_deref(), Some("sm_90"));
        assert_eq!(config.extra_rustflags, ["--cfg", "model=\"alpha beta\""]);
        assert_eq!(
            config.env,
            vec![("MY_BUILD_FLAG".to_string(), "configured".to_string())]
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn passthrough_command_preserves_argv_and_cli_overrides_config_defaults() {
        let config = OxideConfig {
            extra_rustflags: vec!["--cfg".to_string(), "from_config".to_string()],
            env: vec![
                ("CARGO_TARGET_DIR".to_string(), "config-target".to_string()),
                (
                    "CUDA_OXIDE_DEVICE_CODEGEN_CRATE".to_string(),
                    "config_owner".to_string(),
                ),
                ("CUDA_OXIDE_VERBOSE".to_string(), "configured".to_string()),
            ],
            ..OxideConfig::default()
        };
        let ctx = test_context(config);
        let device_cfgs = vec!["model=\"alpha beta\"".to_string()];
        let opts = CargoPassthroughOptions {
            verbose: true,
            emit_nvvm_ir: false,
            arch: Some("sm_90"),
            features: Some("wrapper_feature"),
            cargo_target_dir: Some(Path::new("cli-target")),
            device_codegen_crate: Some("gpu-kernels, math_gpu"),
            device_cfgs: &device_cfgs,
            no_fmad: false,
        };
        let cargo_args = vec![
            "-p".to_string(),
            "gpu-app".to_string(),
            "--".to_string(),
            "--nocapture".to_string(),
        ];

        let cmd = cargo_passthrough_command(&ctx, "test", &opts, &cargo_args).unwrap();
        assert_eq!(
            cmd.get_args()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect::<Vec<_>>(),
            [
                "test",
                "--features",
                "wrapper_feature",
                "-p",
                "gpu-app",
                "--",
                "--nocapture",
            ]
        );
        assert_eq!(
            command_env(&cmd, "CARGO_TARGET_DIR").as_deref(),
            Some("cli-target")
        );
        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_DEVICE_CODEGEN_CRATE").as_deref(),
            Some("gpu_kernels,math_gpu")
        );
        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_TARGET").as_deref(),
            Some("sm_90")
        );
        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_VERBOSE").as_deref(),
            Some("1")
        );

        let encoded = command_env(&cmd, "CARGO_ENCODED_RUSTFLAGS").unwrap();
        let flags = decoded_rustflags(&encoded);
        assert!(
            flags
                .windows(2)
                .any(|pair| pair == ["--cfg", "from_config"])
        );
        assert!(
            flags
                .windows(2)
                .any(|pair| pair == ["--cfg", "model=\"alpha beta\""])
        );
        assert!(has_codegen_env_fingerprint(&flags));
        assert!(
            cmd.get_envs()
                .any(|(key, value)| key == OsStr::new("RUSTFLAGS") && value.is_none())
        );
    }

    #[test]
    fn passthrough_command_accepts_empty_cargo_args() {
        let ctx = test_context(OxideConfig::default());
        let opts = CargoPassthroughOptions {
            verbose: false,
            emit_nvvm_ir: false,
            arch: None,
            features: None,
            cargo_target_dir: None,
            device_codegen_crate: None,
            device_cfgs: &[],
            no_fmad: false,
        };

        let cmd = cargo_passthrough_command(&ctx, "test", &opts, &[]).unwrap();
        assert_eq!(
            cmd.get_args()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect::<Vec<_>>(),
            ["test"]
        );
    }

    #[test]
    fn owner_filter_resolution_is_normalized_and_has_explicit_precedence() {
        assert_eq!(
            resolve_device_codegen_crates(None, None, Some("gpu-kernels, math_gpu"))
                .unwrap()
                .as_deref(),
            Some("gpu_kernels,math_gpu"),
        );
        assert_eq!(
            resolve_device_codegen_crates(None, Some("parent-owner"), Some("config-owner"))
                .unwrap()
                .as_deref(),
            Some("parent_owner"),
        );
        assert!(
            resolve_device_codegen_crates(Some(""), Some("parent-owner"), Some("config-owner"))
                .is_err()
        );
    }

    #[test]
    fn passthrough_fingerprint_tracks_output_affecting_settings() {
        let ctx = test_context(OxideConfig::default());
        let base = CargoPassthroughOptions {
            verbose: false,
            emit_nvvm_ir: false,
            arch: Some("sm_80"),
            features: None,
            cargo_target_dir: None,
            device_codegen_crate: None,
            device_cfgs: &[],
            no_fmad: false,
        };
        let inherited_env = BTreeMap::new();
        let base_hash = passthrough_codegen_fingerprint_with_env(
            &ctx,
            &base,
            None,
            Some("sm_80"),
            &inherited_env,
        );

        let arch = CargoPassthroughOptions {
            arch: Some("sm_90"),
            ..base
        };
        let emit = CargoPassthroughOptions {
            emit_nvvm_ir: true,
            ..base
        };
        let no_fmad = CargoPassthroughOptions {
            no_fmad: true,
            ..base
        };
        let configured_ptx = test_context(OxideConfig {
            env: vec![(
                "CUDA_OXIDE_PTX_DIR".to_string(),
                "configured-ptx".to_string(),
            )],
            ..OxideConfig::default()
        });

        assert_ne!(
            base_hash,
            passthrough_codegen_fingerprint_with_env(
                &ctx,
                &arch,
                None,
                Some("sm_90"),
                &inherited_env,
            )
        );
        assert_ne!(
            base_hash,
            passthrough_codegen_fingerprint_with_env(
                &ctx,
                &emit,
                None,
                Some("sm_80"),
                &inherited_env,
            )
        );
        assert_ne!(
            base_hash,
            passthrough_codegen_fingerprint_with_env(
                &ctx,
                &no_fmad,
                None,
                Some("sm_80"),
                &inherited_env,
            )
        );
        assert_ne!(
            base_hash,
            passthrough_codegen_fingerprint_with_env(
                &ctx,
                &base,
                Some("gpu_kernel"),
                Some("sm_80"),
                &inherited_env,
            )
        );
        assert_ne!(
            base_hash,
            passthrough_codegen_fingerprint_with_env(
                &configured_ptx,
                &base,
                None,
                Some("sm_80"),
                &inherited_env,
            )
        );
    }

    #[test]
    fn passthrough_fingerprint_tracks_backend_rebuild_at_same_path() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cargo_oxide_backend_fingerprint_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&root).unwrap();
        let backend = root.join("librustc_codegen_cuda.so");
        std::fs::write(&backend, b"first").unwrap();

        let mut ctx = test_context(OxideConfig::default());
        ctx.backend_so = backend.clone();
        let opts = CargoPassthroughOptions {
            verbose: false,
            emit_nvvm_ir: false,
            arch: None,
            features: None,
            cargo_target_dir: None,
            device_codegen_crate: None,
            device_cfgs: &[],
            no_fmad: false,
        };
        let inherited_env = BTreeMap::new();
        let before =
            passthrough_codegen_fingerprint_with_env(&ctx, &opts, None, None, &inherited_env);
        std::fs::write(&backend, b"second-build-is-larger").unwrap();
        let after =
            passthrough_codegen_fingerprint_with_env(&ctx, &opts, None, None, &inherited_env);

        assert_ne!(before, after);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn owner_filter_rejects_empty_or_invalid_entries() {
        assert_eq!(
            normalize_device_codegen_crates("gpu-kernels, math_gpu").unwrap(),
            "gpu_kernels,math_gpu"
        );
        assert!(normalize_device_codegen_crates("").is_err());
        assert!(normalize_device_codegen_crates("   ").is_err());
        assert!(normalize_device_codegen_crates("gpu,").is_err());
        assert!(normalize_device_codegen_crates("gpu,not a crate").is_err());
    }

    #[test]
    fn internal_ptx_directory_overrides_project_env_default() {
        let ctx = test_context(OxideConfig {
            env: vec![(
                "CUDA_OXIDE_PTX_DIR".to_string(),
                "configured-ptx".to_string(),
            )],
            ..OxideConfig::default()
        });
        let mut cmd = Command::new("cargo");
        apply_common_codegen_env(&mut cmd, &ctx, false, false);
        cmd.env("CUDA_OXIDE_PTX_DIR", "internal-ptx");
        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_PTX_DIR").as_deref(),
            Some("internal-ptx")
        );
    }

    #[test]
    fn nvvm_arch_normalizes_all_accepted_forms() {
        // `sm_XX` is the form `--arch` and the rest of cargo-oxide use.
        assert_eq!(parse_nvvm_arch("sm_120").unwrap().compute(), "compute_120");
        assert_eq!(parse_nvvm_arch("sm_90").unwrap().compute(), "compute_90");
        // `compute_XX` passes through unchanged.
        assert_eq!(
            parse_nvvm_arch("compute_100").unwrap().compute(),
            "compute_100"
        );
        // A bare capability is accepted too.
        assert_eq!(parse_nvvm_arch("120").unwrap().compute(), "compute_120");
        assert!(parse_nvvm_arch("sm_90x").is_err());
    }

    #[test]
    fn apply_output_mode_sets_target_for_arch_override() {
        let mut cmd = Command::new("cargo");

        apply_output_mode(&mut cmd, false, Some("sm_120"));

        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_TARGET").as_deref(),
            Some("sm_120")
        );
        assert_eq!(command_env(&cmd, "CUDA_OXIDE_EMIT_NVVM_IR"), None);
    }

    #[test]
    fn apply_output_mode_sets_nvvm_ir_flag_and_target() {
        let mut cmd = Command::new("cargo");

        apply_output_mode(&mut cmd, true, Some("sm_100a"));

        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_TARGET").as_deref(),
            Some("sm_100a")
        );
        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_EMIT_NVVM_IR").as_deref(),
            Some("1")
        );
    }

    #[test]
    fn apply_output_mode_leaves_auto_detect_ptx_unset() {
        let mut cmd = Command::new("cargo");

        apply_output_mode(&mut cmd, false, None);

        assert_eq!(command_env(&cmd, "CUDA_OXIDE_TARGET"), None);
        assert_eq!(command_env(&cmd, "CUDA_OXIDE_EMIT_NVVM_IR"), None);
    }

    #[test]
    fn apply_device_arch_hint_sets_hint_when_no_explicit_arch() {
        let mut cmd = Command::new("cargo");

        apply_device_arch_hint(&mut cmd, None, Some("sm_120a"));

        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_DEVICE_ARCH").as_deref(),
            Some("sm_120a")
        );
        // The hint must never masquerade as the hard override.
        assert_eq!(command_env(&cmd, "CUDA_OXIDE_TARGET"), None);
    }

    #[test]
    fn apply_device_arch_hint_skipped_when_arch_explicit() {
        // An explicit --arch already went to CUDA_OXIDE_TARGET; don't also
        // emit a competing device hint.
        let mut cmd = Command::new("cargo");

        apply_device_arch_hint(&mut cmd, Some("sm_90"), Some("sm_120a"));

        assert_eq!(command_env(&cmd, "CUDA_OXIDE_DEVICE_ARCH"), None);
    }

    #[test]
    fn apply_device_arch_hint_noop_without_detection() {
        let mut cmd = Command::new("cargo");

        apply_device_arch_hint(&mut cmd, None, None);

        assert_eq!(command_env(&cmd, "CUDA_OXIDE_DEVICE_ARCH"), None);
    }

    #[test]
    fn debug_output_mode_forwards_detected_gpu_hint() {
        let mut cmd = Command::new("cargo");

        apply_output_mode(&mut cmd, false, None);
        apply_device_arch_hint(&mut cmd, None, Some("sm_120a"));

        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_DEVICE_ARCH").as_deref(),
            Some("sm_120a")
        );
        assert_eq!(command_env(&cmd, "CUDA_OXIDE_TARGET"), None);
        assert_eq!(command_env(&cmd, "CUDA_OXIDE_EMIT_NVVM_IR"), None);
    }

    #[test]
    fn debug_output_mode_honors_explicit_arch_override() {
        let mut cmd = Command::new("cargo");

        apply_output_mode(&mut cmd, false, Some("sm_90"));
        apply_device_arch_hint(&mut cmd, Some("sm_90"), Some("sm_120a"));

        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_TARGET").as_deref(),
            Some("sm_90")
        );
        assert_eq!(command_env(&cmd, "CUDA_OXIDE_DEVICE_ARCH"), None);
        assert_eq!(command_env(&cmd, "CUDA_OXIDE_EMIT_NVVM_IR"), None);
    }

    #[test]
    fn format_sm_arch_uses_cuda_target_spelling() {
        // cc < 9.0 — no arch-specific target exists in the PTX ISA, so we
        // emit the plain `sm_XY` form. Confirms we do not produce false
        // positives like `sm_75a` / `sm_80a` / `sm_89a`.
        assert_eq!(format_sm_arch((7, 0)), "sm_70");
        assert_eq!(format_sm_arch((7, 5)), "sm_75");
        assert_eq!(format_sm_arch((8, 0)), "sm_80");
        assert_eq!(format_sm_arch((8, 6)), "sm_86");
        assert_eq!(format_sm_arch((8, 9)), "sm_89");

        // cc ≥ 9.0 — every chip that reports this CC is an arch-specific
        // (`a`) variant. Auto-detect emits the `a` form so the codegen
        // backend can lower WGMMA / tcgen05 / TMA-multicast / cta_group
        // intrinsics without falling through to a plain target that ptxas
        // would reject. Confirms we do not produce false negatives.
        assert_eq!(format_sm_arch((9, 0)), "sm_90a"); // Hopper (H100/H200)
        assert_eq!(format_sm_arch((10, 0)), "sm_100a"); // Blackwell DC
        assert_eq!(format_sm_arch((10, 1)), "sm_101a");
        assert_eq!(format_sm_arch((10, 3)), "sm_103a");
        assert_eq!(format_sm_arch((12, 0)), "sm_120a"); // consumer Blackwell
    }

    #[test]
    fn parse_compute_cap_accepts_real_nvidia_smi_output() {
        assert_eq!(parse_compute_cap("12.0\n"), Some((12, 0)));
        assert_eq!(parse_compute_cap("7.5\n"), Some((7, 5)));
        assert_eq!(parse_compute_cap("10.3"), Some((10, 3)));
        // End-to-end with format_sm_arch: the values the backend sees.
        assert_eq!(
            format_sm_arch(parse_compute_cap("12.0\n").unwrap()),
            "sm_120a"
        );
        assert_eq!(format_sm_arch(parse_compute_cap("7.5\n").unwrap()), "sm_75");
    }

    #[test]
    fn parse_compute_cap_takes_first_gpu_on_multi_gpu_machines() {
        assert_eq!(parse_compute_cap("9.0\n12.0\n"), Some((9, 0)));
    }

    #[test]
    fn parse_gpu_name_and_compute_cap_splits_on_last_comma() {
        assert_eq!(
            parse_gpu_name_and_compute_cap("NVIDIA GeForce RTX 5090, 12.0\n"),
            Some(("NVIDIA GeForce RTX 5090".to_string(), (12, 0)))
        );
        // Failure banner: no comma-separated cc field.
        assert_eq!(
            parse_gpu_name_and_compute_cap("NVIDIA-SMI has failed.\n"),
            None
        );
        assert_eq!(parse_gpu_name_and_compute_cap(""), None);
    }

    #[test]
    fn cuda_toolkit_root_prefers_toolkit_path_then_home_then_default() {
        let toolkit_and_home = cuda_toolkit_root(|var| match var {
            "CUDA_TOOLKIT_PATH" => Some("/cuda/toolkit".to_string()),
            "CUDA_HOME" => Some("/cuda/home".to_string()),
            _ => None,
        });
        assert_eq!(toolkit_and_home, "/cuda/toolkit");

        let home_only =
            cuda_toolkit_root(|var| (var == "CUDA_HOME").then(|| "/cuda/home".to_string()));
        assert_eq!(home_only, "/cuda/home");

        let empty_toolkit_path = cuda_toolkit_root(|var| match var {
            "CUDA_TOOLKIT_PATH" => Some("  ".to_string()),
            "CUDA_HOME" => Some("/cuda/home".to_string()),
            _ => None,
        });
        assert_eq!(empty_toolkit_path, "/cuda/home");

        assert_eq!(cuda_toolkit_root(|_| None), "/usr/local/cuda");
    }

    #[test]
    fn cuda_header_candidates_cover_standard_and_redistributable_layouts() {
        // Standard install layout first, then the matching targets/ layout.
        assert_eq!(
            cuda_header_candidates("/usr/local/cuda", "x86_64"),
            vec![
                PathBuf::from("/usr/local/cuda/include/cuda.h"),
                PathBuf::from("/usr/local/cuda/targets/x86_64-linux/include/cuda.h"),
            ]
        );
        // aarch64 servers use the sbsa-linux target dir.
        assert_eq!(
            cuda_header_candidates("/opt/ctk", "aarch64"),
            vec![
                PathBuf::from("/opt/ctk/include/cuda.h"),
                PathBuf::from("/opt/ctk/targets/sbsa-linux/include/cuda.h"),
            ]
        );
        // Unknown host arch: only the standard layout is probed.
        assert_eq!(
            cuda_header_candidates("/opt/ctk", "riscv64"),
            vec![PathBuf::from("/opt/ctk/include/cuda.h")]
        );
    }

    #[test]
    fn parse_compute_cap_rejects_failure_banners_and_garbage() {
        // nvidia-smi prints failure text to STDOUT, not stderr.
        assert_eq!(
            parse_compute_cap(
                "NVIDIA-SMI has failed because it couldn't communicate \
                 with the NVIDIA driver.\n"
            ),
            None
        );
        assert_eq!(parse_compute_cap(""), None);
        assert_eq!(parse_compute_cap("\n"), None);
        assert_eq!(parse_compute_cap("N/A\n"), None);
        assert_eq!(parse_compute_cap("12\n"), None);
        assert_eq!(parse_compute_cap("12.\n"), None);
        assert_eq!(parse_compute_cap(".5\n"), None);
        assert_eq!(parse_compute_cap("12.0.1\n"), None);
    }

    #[test]
    fn detect_run_target_arch_skips_when_arch_explicit() {
        // --arch wins; never query the GPU.
        assert_eq!(detect_run_target_arch(Some("sm_120"), false), None);
    }

    #[test]
    fn detect_run_target_arch_skips_when_emit_nvvm_ir() {
        // NVVM IR mode requires explicit --arch; auto-detect must not run.
        assert_eq!(detect_run_target_arch(None, true), None);
    }

    #[test]
    fn detect_run_target_arch_skips_when_env_target_set() {
        // Test in isolation; the `CUDA_OXIDE_TARGET` env handle is process-wide.
        // SAFETY: single-threaded test serialised by the cargo test harness.
        unsafe {
            std::env::set_var("CUDA_OXIDE_TARGET", "sm_75");
        }
        let result = detect_run_target_arch(None, false);
        unsafe {
            std::env::remove_var("CUDA_OXIDE_TARGET");
        }
        assert_eq!(result, None);
    }
}
