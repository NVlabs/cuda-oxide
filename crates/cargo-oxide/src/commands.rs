/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Command implementations for cargo-oxide.
//!
//! These port the xtask commands with improvements:
//! - Backend path resolved via discovery chain instead of hardcoded relative path
//! - Workspace root resolved by walking up from CWD instead of assuming CWD

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::backend;

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
        let backend_so = backend::find_or_build_backend(&workspace_root);
        return Context {
            workspace_root,
            codegen_crate,
            examples_dir,
            backend_so,
            is_workspace: true,
        };
    }

    let cwd = std::env::current_dir().unwrap_or_else(|e| {
        eprintln!("Error: cannot determine current directory: {}", e);
        std::process::exit(1);
    });

    if cwd.join("Cargo.toml").is_file() {
        let backend_so = backend::find_or_build_backend(&cwd);
        return Context {
            workspace_root: cwd.clone(),
            codegen_crate: cwd.clone(),
            examples_dir: cwd.clone(),
            backend_so,
            is_workspace: false,
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
/// Cleans stale artifacts, sets `RUSTFLAGS` to point at the backend `.so`,
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
) {
    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };

    let interop = load_interop_config(&example_dir);

    let output_format = format_label(emit_nvvm_ir);
    // Target precedence for `cargo oxide run` (highest first):
    //   1. --arch <sm_XX>            explicit user override   -> CUDA_OXIDE_TARGET
    //   2. CUDA_OXIDE_TARGET=<sm_XX> explicit env override (from the parent)
    //   3. detected GPU arch of CUDA device 0 -> CUDA_OXIDE_DEVICE_ARCH (a hint)
    //   4. backend feature-based default (`select_target` in mir-importer)
    //
    // Slot 3 is a HINT, not an override: the backend builds for the detected
    // GPU only when that GPU can run the kernel. If the kernel needs a newer
    // arch (tcgen05 needs sm_100a even on a consumer sm_120 GPU), the backend
    // builds for the required arch and the module simply skips at load time.
    // We only detect for `run`, not `build`/`pipeline`: `run` loads the cubin
    // on device 0, whereas those may legitimately cross-compile for another
    // machine.
    let detected_device_arch = detect_run_target_arch(arch, emit_nvvm_ir);

    if let Some(interop) = interop.filter(|config| !config.device_crates.is_empty()) {
        codegen_run_interop(
            ctx,
            example,
            &example_dir,
            &interop,
            verbose,
            emit_nvvm_ir,
            arch,
            detected_device_arch.as_deref(),
            features,
            bin,
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
            arch.expect("--emit-nvvm-ir requires --arch")
        );
        println!();
    } else if let Some(dev) = detected_device_arch.as_deref() {
        // Surface the detected GPU so it isn't silent magic. It is a hint, not
        // a hard target: the backend builds for it unless a kernel needs a
        // newer arch (e.g. tcgen05 forces sm_100a even on a consumer sm_120
        // GPU), so the final PTX target may differ.
        println!("Detected GPU arch: {dev} (CUDA device 0)");
        println!();
    }
    println!("This is the proper cargo workflow:");
    println!("  RUSTFLAGS=\"-Z codegen-backend=...\" cargo run");
    println!();

    let rustflags = build_rustflags(&ctx.backend_so, false);

    touch_main_rs(&example_dir);

    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--release"])
        .current_dir(&example_dir)
        .env("RUSTFLAGS", &rustflags);

    if let Some(bin) = bin {
        cmd.args(["--bin", bin]);
    }
    if let Some(features) = features {
        cmd.args(["--features", features]);
    }

    if verbose || std::env::var("CUDA_OXIDE_VERBOSE").is_ok() {
        cmd.env("CUDA_OXIDE_VERBOSE", "1");
    } else {
        cmd.env_remove("CUDA_OXIDE_VERBOSE");
    }
    forward_env_var(&mut cmd, "CUDA_OXIDE_SHOW_RUSTC_MIR");
    forward_env_var(&mut cmd, "CUDA_OXIDE_DUMP_MIR");
    forward_env_var(&mut cmd, "CUDA_OXIDE_DUMP_LLVM");

    apply_output_mode(&mut cmd, emit_nvvm_ir, arch);
    apply_device_arch_hint(&mut cmd, arch, detected_device_arch.as_deref());
    apply_ld_library_path(&mut cmd);

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
) {
    reject_interop_nvvm_ir(emit_nvvm_ir);

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA INTEROP: {}", example);
    println!("=========================================");
    if let Some(kind) = &interop.kind {
        println!("Interop kind: {}", kind);
    }
    if let Some(dev) = detected_device_arch {
        println!("Detected GPU arch: {dev} (CUDA device 0)");
    }
    println!();

    build_interop_device_crates(
        ctx,
        example_dir,
        interop,
        verbose,
        arch,
        detected_device_arch,
    );
    run_host_cargo(example, example_dir, "run", features, bin, verbose);
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
    build_interop_device_crates(ctx, example_dir, interop, verbose, arch, None);
    run_host_cargo(example, example_dir, "build", features, None, verbose);

    println!();
    println!("✓ Build succeeded");
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
) {
    for device_crate in &interop.device_crates {
        build_interop_device_crate(
            ctx,
            example_dir,
            device_crate,
            verbose,
            arch,
            detected_device_arch,
        );
    }
}

fn build_interop_device_crate(
    ctx: &Context,
    example_dir: &Path,
    device_crate: &DeviceCrateConfig,
    verbose: bool,
    arch: Option<&str>,
    detected_device_arch: Option<&str>,
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

    let rustflags = build_rustflags(&ctx.backend_so, false);
    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release", "--manifest-path"])
        .arg(&manifest_path)
        .current_dir(device_dir)
        .env("RUSTFLAGS", &rustflags)
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env("CUDA_OXIDE_PTX_DIR", &ptx_dir);

    if verbose || std::env::var("CUDA_OXIDE_VERBOSE").is_ok() {
        cmd.env("CUDA_OXIDE_VERBOSE", "1");
    } else {
        cmd.env_remove("CUDA_OXIDE_VERBOSE");
    }
    forward_env_var(&mut cmd, "CUDA_OXIDE_SHOW_RUSTC_MIR");
    forward_env_var(&mut cmd, "CUDA_OXIDE_DUMP_MIR");
    forward_env_var(&mut cmd, "CUDA_OXIDE_DUMP_LLVM");
    apply_output_mode(&mut cmd, false, arch);
    apply_device_arch_hint(&mut cmd, arch, detected_device_arch);
    apply_ld_library_path(&mut cmd);

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

    apply_ld_library_path(&mut cmd);

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

// =============================================================================
// Build command (compile only, don't run)
// =============================================================================

/// Compile an example without running it.
///
/// Same as [`codegen_run`] but uses `cargo build --release` instead of
/// `cargo run`. Useful for cross-compilation or when the target hardware
/// (e.g., Blackwell tensor cores) isn't available on the build machine.
pub fn codegen_build_example(
    ctx: &Context,
    example: &str,
    verbose: bool,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
    features: Option<&str>,
) {
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
            arch,
            features,
        );
        return;
    }

    clean_generated_files(&example_dir, example);

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA BUILD: {}", example);
    println!("=========================================");
    println!();

    let rustflags = build_rustflags(&ctx.backend_so, false);

    touch_main_rs(&example_dir);

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"])
        .current_dir(&example_dir)
        .env("RUSTFLAGS", &rustflags);

    if let Some(features) = features {
        cmd.args(["--features", features]);
    }

    if verbose || std::env::var("CUDA_OXIDE_VERBOSE").is_ok() {
        cmd.env("CUDA_OXIDE_VERBOSE", "1");
    } else {
        cmd.env_remove("CUDA_OXIDE_VERBOSE");
    }
    forward_env_var(&mut cmd, "CUDA_OXIDE_SHOW_RUSTC_MIR");
    forward_env_var(&mut cmd, "CUDA_OXIDE_DUMP_MIR");
    forward_env_var(&mut cmd, "CUDA_OXIDE_DUMP_LLVM");

    apply_output_mode(&mut cmd, emit_nvvm_ir, arch);
    apply_ld_library_path(&mut cmd);

    println!("Building {}...", example);
    println!();

    let status = cmd.status().expect("Failed to run cargo");
    if !status.success() {
        eprintln!("\nBuild failed with exit code: {:?}", status.code());
        std::process::exit(status.code().unwrap_or(1));
    }

    println!();
    println!("✓ Build succeeded");
}

// =============================================================================
// Pipeline command
// =============================================================================

/// Show the full compilation pipeline with verbose output at every stage.
///
/// Enables all diagnostic env vars (`CUDA_OXIDE_VERBOSE`, `SHOW_RUSTC_MIR`,
/// `DUMP_MIR`, `DUMP_LLVM`) so the user can see MIR collection, the
/// `dialect-mir` module (pre- and post-`mem2reg`), the LLVM dialect
/// module, textual LLVM IR, and the final PTX or NVVM IR. After the build,
/// generated artifacts are printed to stdout.
pub fn codegen_show_pipeline(ctx: &Context, example: &str, emit_nvvm_ir: bool, arch: Option<&str>) {
    let example_dir = resolve_example_dir(ctx, example);

    clean_generated_files(&example_dir, example);

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA PIPELINE: {}", example);
    println!("=========================================");
    println!();
    match (emit_nvvm_ir, arch) {
        (true, Some(target_arch)) => println!("Output format: NVVM IR (arch: {})", target_arch),
        (false, Some(target_arch)) => {
            println!("Output format: PTX (arch override: {})", target_arch)
        }
        (false, None) => println!("Output format: PTX (auto-detected arch)"),
        (true, None) => unreachable!("--emit-nvvm-ir requires --arch"),
    }
    println!();
    println!("Required flags (applied via RUSTFLAGS):");
    println!("  -C opt-level=3              MIR optimization");
    println!("  -C debug-assertions=off     Remove debug checks");
    println!("  -Z mir-enable-passes=-JumpThreading");
    println!("                              Prevent barrier duplication");
    println!();
    println!("Note: panic=abort is NOT required - the codegen backend treats");
    println!("      unwind paths as unreachable (CUDA toolchain limitation, not HW).");
    println!();

    let rustflags = build_rustflags(&ctx.backend_so, false);

    touch_main_rs(&example_dir);

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"])
        .current_dir(&example_dir)
        .env("RUSTFLAGS", &rustflags);

    cmd.env("CUDA_OXIDE_VERBOSE", "1");
    cmd.env("CUDA_OXIDE_SHOW_RUSTC_MIR", "1");
    cmd.env("CUDA_OXIDE_DUMP_MIR", "1");
    cmd.env("CUDA_OXIDE_DUMP_LLVM", "1");

    apply_output_mode(&mut cmd, emit_nvvm_ir, arch);
    apply_ld_library_path(&mut cmd);

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
pub fn codegen_debug(ctx: &Context, example: &str, use_cgdb: bool, use_tui: bool) {
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

    let example_dir = resolve_example_dir(ctx, example);

    println!("Building {} with debug info...", example);

    let rustflags = build_rustflags(&ctx.backend_so, true);

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"])
        .current_dir(&example_dir)
        .env("RUSTFLAGS", &rustflags)
        .env("CARGO_PROFILE_RELEASE_DEBUG", "2");

    apply_ld_library_path(&mut cmd);

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
/// backend `.so`, CUDA toolkit (`nvcc`), LLVM (`llc`), and optionally
/// `cuda-gdb`. Exits non-zero if any required check fails.
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

    // 3. Backend .so
    print!("Codegen backend... ");
    if ctx.backend_so.exists() {
        println!("✓ {}", ctx.backend_so.display());
    } else {
        println!("✗ not found (run `cargo oxide setup`)");
        ok = false;
    }

    // 4. CUDA toolkit
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

    // 4b. libNVVM + nvJitLink + libdevice (only required when a kernel uses
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
    match cuda_host::ltoir::find_libdevice() {
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

    // 5. llc (LLVM static compiler for PTX)
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

    // 6. clang / libclang resource dir (host `cuda-bindings` / bindgen)
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

    // 7. cuda-gdb (optional)
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

/// Construct the `RUSTFLAGS` string that configures rustc to use our backend.
///
/// Always includes `-Z codegen-backend`, `-C opt-level=3`, disabled debug
/// assertions, suppressed JumpThreading (prevents barrier duplication), and
/// v0 symbol mangling. Appends `-C debuginfo=2` when `debug` is true, then
/// appends any existing user-provided `RUSTFLAGS`.
fn build_rustflags(backend_so: &Path, debug: bool) -> String {
    let existing = std::env::var("RUSTFLAGS").ok();
    build_rustflags_with_existing(backend_so, debug, existing.as_deref())
}

fn build_rustflags_with_existing(
    backend_so: &Path,
    debug: bool,
    existing_rustflags: Option<&str>,
) -> String {
    let mut flags = format!(
        "-Z codegen-backend={} -C opt-level=3 -C debug-assertions=off -Z mir-enable-passes=-JumpThreading -Csymbol-mangling-version=v0",
        backend_so.display()
    );
    if debug {
        flags.push_str(" -C debuginfo=2");
    }
    if let Some(existing) = existing_rustflags
        && !existing.is_empty()
    {
        flags.push(' ');
        flags.push_str(existing);
    }
    flags
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
/// 3. **This function**: the compute capability of the GPU in CUDA device 0,
///    forwarded as the `CUDA_OXIDE_DEVICE_ARCH` *hint*. Emits the arch-specific
///    `sm_XYa` form for cc >= 9.0 (so the backend can lower WGMMA / tcgen05 /
///    TMA-multicast when the GPU supports them) and the plain `sm_XY` form for
///    cc < 9.0.
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
/// `run` immediately loads the generated module on device 0 and launches the
/// kernel, so a target older than the local GPU's compute capability is the
/// only safe default. `build` and `pipeline` may legitimately cross-compile
/// to a different machine, so they keep the backend's feature-based default
/// untouched.
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
/// - No CUDA driver / device 0 is available on the machine (CI runners without
///   GPUs, headless build boxes). The caller falls through to slot 4 and
///   the backend's feature-based default applies.
fn detect_run_target_arch(arch: Option<&str>, emit_nvvm_ir: bool) -> Option<String> {
    if arch.is_some() || emit_nvvm_ir || std::env::var_os("CUDA_OXIDE_TARGET").is_some() {
        return None;
    }

    cuda_core::CudaContext::new(0)
        .and_then(|ctx| ctx.compute_capability())
        .ok()
        .map(format_sm_arch)
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
fn format_sm_arch((major, minor): (i32, i32)) -> String {
    if major >= 9 {
        format!("sm_{}{}a", major, minor)
    } else {
        format!("sm_{}{}", major, minor)
    }
}

/// Forward an env var to the child process if it's set in the parent, otherwise remove it.
fn forward_env_var(cmd: &mut Command, var: &str) {
    if let Ok(val) = std::env::var(var) {
        cmd.env(var, val);
    } else {
        cmd.env_remove(var);
    }
}

/// Build `LD_LIBRARY_PATH` for the child cargo process.
///
/// Includes the rustc sysroot lib (for `librustc_driver.so` etc.), the
/// libmathdx lib (when `LIBMATHDX_PATH` is set), and any existing
/// `LD_LIBRARY_PATH` from the parent environment.
fn apply_ld_library_path(cmd: &mut Command) {
    let mut ld_paths: Vec<String> = Vec::new();
    if let Some(sysroot) = backend::get_rustc_sysroot() {
        ld_paths.push(format!("{}/lib", sysroot));
    }
    if let Ok(libmathdx_path) = std::env::var("LIBMATHDX_PATH") {
        ld_paths.push(format!("{}/lib", libmathdx_path));
    }
    if let Ok(existing) = std::env::var("LD_LIBRARY_PATH") {
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
    for ext in &["ptx", "ll", "opt.ll", "ltoir", "cubin"] {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    fn command_env(cmd: &Command, key: &str) -> Option<String> {
        cmd.get_envs()
            .find(|(name, _)| *name == OsStr::new(key))
            .and_then(|(_, value)| value.map(|v| v.to_string_lossy().into_owned()))
    }

    #[test]
    fn artifact_stem_normalizes_hyphens_like_cargo() {
        assert_eq!(artifact_stem("rustlantis-smoke"), "rustlantis_smoke");
        assert_eq!(artifact_stem("vecadd"), "vecadd");
    }

    #[test]
    fn build_rustflags_appends_existing_rustflags_after_required_flags() {
        let rustflags = build_rustflags_with_existing(
            Path::new("/tmp/librustc_codegen_cuda.so"),
            false,
            Some("-L native=/nix/store/cuda-cudart/lib"),
        );

        assert!(
            rustflags
                .starts_with("-Z codegen-backend=/tmp/librustc_codegen_cuda.so -C opt-level=3")
        );
        assert!(rustflags.ends_with(" -L native=/nix/store/cuda-cudart/lib"));
    }

    #[test]
    fn build_rustflags_ignores_empty_existing_rustflags() {
        let rustflags = build_rustflags_with_existing(
            Path::new("/tmp/librustc_codegen_cuda.so"),
            true,
            Some(""),
        );

        assert!(rustflags.contains(" -C debuginfo=2"));
        assert!(!rustflags.ends_with(' '));
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
