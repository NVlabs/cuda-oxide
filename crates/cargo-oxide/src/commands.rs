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
/// variables control output format (PTX / LTOIR / NVVM IR) and verbosity.
#[allow(clippy::too_many_arguments)]
pub fn codegen_run(
    ctx: &Context,
    example: &str,
    verbose: bool,
    dlto: bool,
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

    clean_generated_files(&example_dir, example);

    let output_format = format_label(dlto, emit_nvvm_ir);
    let target_arch = arch.unwrap_or(if dlto { "sm_100" } else { "sm_90" });

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA: {}", example);
    println!("=========================================");
    println!();
    if dlto || emit_nvvm_ir {
        println!("Output format: {}", output_format);
        if dlto {
            println!("Target arch: {}", target_arch);
        }
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

    apply_output_mode(&mut cmd, dlto, emit_nvvm_ir, arch, target_arch);
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
    dlto: bool,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
    features: Option<&str>,
) {
    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };

    clean_generated_files(&example_dir, example);

    let target_arch = arch.unwrap_or(if dlto { "sm_100" } else { "sm_90" });

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

    apply_output_mode(&mut cmd, dlto, emit_nvvm_ir, arch, target_arch);
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
/// `dialect-mir` module (pre- and post-`mem2reg`), the `dialect-llvm`
/// module, textual LLVM IR, and the final PTX or LTOIR. After the build,
/// generated artifacts are printed to stdout.
pub fn codegen_show_pipeline(
    ctx: &Context,
    example: &str,
    dlto: bool,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
) {
    let example_dir = resolve_example_dir(ctx, example);

    clean_generated_files(&example_dir, example);

    let output_format = format_label(dlto, emit_nvvm_ir);
    let target_arch = arch.unwrap_or(if dlto { "sm_100" } else { "sm_90" });

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA PIPELINE: {}", example);
    println!("=========================================");
    println!();
    println!("Output format: {} (arch: {})", output_format, target_arch);
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

    apply_output_mode(&mut cmd, dlto, emit_nvvm_ir, arch, target_arch);
    apply_ld_library_path(&mut cmd);

    println!("Building {}...", example);
    println!();

    let status = cmd.status().expect("Failed to run cargo");

    if !status.success() {
        eprintln!("\nBuild failed with exit code: {:?}", status.code());
        std::process::exit(status.code().unwrap_or(1));
    }

    show_generated_artifacts(&example_dir, example, dlto);
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
    // pipeline (llc-22 → llc-21), falling back to bare `llc` only for
    // reporting purposes. Whatever we pick, reject if the major version
    // is < 21.
    print!("llc (LLVM)... ");
    let llc_pick = ["llc-22", "llc-21", "llc"].iter().find_map(|candidate| {
        Command::new(candidate)
            .arg("--version")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| (*candidate, String::from_utf8_lossy(&o.stdout).into_owned()))
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
                    eprintln!("  toolchain (`sudo apt install llvm-21`) and either add it to");
                    eprintln!("  PATH or set `CUDA_OXIDE_LLC=/usr/bin/llc-21`.");
                    ok = false;
                }
                None => println!("✓ {} ({}, version could not be parsed)", banner, binary),
            }
        }
        None => {
            println!("✗ llc not found");
            eprintln!("  Install LLVM 21+: sudo apt install llvm-21");
            eprintln!("  (cuda-oxide probes llc-22 then llc-21 on PATH;");
            eprintln!("   older versions reject modern TMA/tcgen05 intrinsics)");
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
/// v0 symbol mangling. Appends `-C debuginfo=2` when `debug` is true.
fn build_rustflags(backend_so: &Path, debug: bool) -> String {
    let mut flags = format!(
        "-Z codegen-backend={} -C opt-level=3 -C debug-assertions=off -Z mir-enable-passes=-JumpThreading -Csymbol-mangling-version=v0",
        backend_so.display()
    );
    if debug {
        flags.push_str(" -C debuginfo=2");
    }
    flags
}

/// Set environment variables that tell the codegen backend which output
/// format to produce (LTOIR, NVVM IR) and the target GPU architecture.
fn apply_output_mode(
    cmd: &mut Command,
    dlto: bool,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
    target_arch: &str,
) {
    if dlto {
        cmd.env("CUDA_OXIDE_EMIT_LTOIR", "1");
        cmd.env("CUDA_OXIDE_ARCH", target_arch);
    }
    if emit_nvvm_ir || arch.is_some() {
        cmd.env("CUDA_OXIDE_TARGET", target_arch);
    }
    if emit_nvvm_ir {
        cmd.env("CUDA_OXIDE_EMIT_NVVM_IR", "1");
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

/// Remove stale generated artifacts (`.ptx`, `.ll`, `.ltoir`) from a
/// previous run so we can verify the build produces fresh output.
fn clean_generated_files(example_dir: &Path, example: &str) {
    for ext in &["ptx", "ll", "ltoir"] {
        let file = example_dir.join(format!("{}.{}", example, ext));
        if file.exists() {
            let _ = std::fs::remove_file(&file);
        }
    }
}

/// Human-readable label for the selected output format.
fn format_label(dlto: bool, emit_nvvm_ir: bool) -> &'static str {
    if dlto {
        "LTOIR"
    } else if emit_nvvm_ir {
        "NVVM IR"
    } else {
        "PTX"
    }
}

/// Print generated artifacts (LLVM IR, PTX, or LTOIR) to stdout after a
/// pipeline build. For LTOIR (binary), prints the file path and disassembly
/// instructions instead of raw content.
fn show_generated_artifacts(example_dir: &Path, example: &str, dlto: bool) {
    let ll_file = example_dir.join(format!("{}.ll", example));
    let ptx_file = example_dir.join(format!("{}.ptx", example));
    let ltoir_file = example_dir.join(format!("{}.ltoir", example));

    if ll_file.exists() {
        println!();
        println!("=========================================");
        println!("LLVM IR ({}.ll)", example);
        println!("=========================================");
        if let Ok(content) = std::fs::read_to_string(&ll_file) {
            println!("{}", content);
        }
    }

    if dlto && ltoir_file.exists() {
        println!();
        println!("=========================================");
        println!("LTOIR ({}.ltoir)", example);
        println!("=========================================");
        println!("Binary LTOIR file generated: {}", ltoir_file.display());
        println!();
        println!("To disassemble, use nvvm-dis (NVVM 2.0 dialect for Blackwell+):");
        println!(
            "  export LD_LIBRARY_PATH=/path/to/nvvm-tools-next/Linux_amd64_release:$LD_LIBRARY_PATH"
        );
        println!("  nvvm-dis {}.ltoir", example);
    } else if ptx_file.exists() {
        println!();
        println!("=========================================");
        println!("PTX ({}.ptx)", example);
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
components = ["rust-src", "rustc-dev", "rust-analyzer"]
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
cuda-host = {{ git = "{GIT_REPO}" }}
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
edition = "2021"

[workspace]

[dependencies]
cuda-device = {{ git = "{GIT_REPO}" }}
cuda-host = {{ git = "{GIT_REPO}" }}
cuda-core = {{ git = "{GIT_REPO}" }}
"#
        )
    };

    let main_rs = if async_mode {
        format!(
            r#"use cuda_device::{{kernel, thread, DisjointSlice}};
use cuda_host::cuda_launch_async;
use cuda_async::device_context::{{init_device_contexts, load_kernel_module_async}};
use cuda_async::device_operation::DeviceOperation;
use cuda_core::LaunchConfig;

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {{
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {{
        *c_elem = a[idx.get()] + b[idx.get()];
    }}
}}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {{
    use cuda_async::device_box::DeviceBox;
    use cuda_core::memory::{{malloc_async, memcpy_dtoh_async, memcpy_htod_async}};
    use std::mem;

    init_device_contexts(0, 1)?;
    // Loads `{name}.ptx` directly when cuda-oxide produced PTX, or builds a
    // cubin from `{name}.ll` when cuda-oxide auto-detected libdevice math
    // (`sin`, `pow`, `exp`, ...). Requires CUDA Toolkit on the host.
    let module = load_kernel_module_async("{name}", 0)?;

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let (a_dev, b_dev, mut c_dev) = cuda_async::device_context::with_cuda_context(0, |ctx| {{
        let stream = ctx.default_stream();
        let num_bytes = N * mem::size_of::<f32>();
        unsafe {{
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
        }}
    }})?;

    cuda_launch_async! {{
        kernel: vecadd,
        module: module,
        config: LaunchConfig::for_num_elems(N as u32),
        args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
    }}
    .sync()?;

    let mut c_host = vec![0.0f32; N];
    cuda_async::device_context::with_cuda_context(0, |ctx| {{
        let stream = ctx.default_stream();
        unsafe {{
            memcpy_dtoh_async(
                c_host.as_mut_ptr(),
                c_dev.cu_deviceptr(),
                N * mem::size_of::<f32>(),
                stream.cu_stream(),
            )
            .unwrap();
            stream.synchronize().unwrap();
        }}
    }})?;

    let errors = (0..N)
        .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
        .count();

    if errors == 0 {{
        println!("PASSED: all {{}} elements correct", N);
    }} else {{
        eprintln!("FAILED: {{}} errors", errors);
        std::process::exit(1);
    }}

    Ok(())
}}
"#
        )
    } else {
        format!(
            r#"use cuda_device::{{kernel, thread, DisjointSlice}};
use cuda_core::{{CudaContext, DeviceBuffer, LaunchConfig}};
use cuda_host::{{cuda_launch, load_kernel_module}};

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {{
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {{
        *c_elem = a[idx.get()] + b[idx.get()];
    }}
}}

fn main() {{
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    // Loads `{name}.ptx` directly when cuda-oxide produced PTX, or builds a
    // cubin from `{name}.ll` when cuda-oxide auto-detected libdevice math
    // (`sin`, `pow`, `exp`, ...). Requires CUDA Toolkit on the host.
    let module = load_kernel_module(&ctx, "{name}")
        .expect("Failed to load kernel module");

    cuda_launch! {{
        kernel: vecadd,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(N as u32),
        args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
    }}
    .expect("Kernel launch failed");

    let c_host = c_dev.to_host_vec(&stream).unwrap();

    let errors = (0..N)
        .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
        .count();

    if errors == 0 {{
        println!("PASSED: all {{}} elements correct", N);
    }} else {{
        eprintln!("FAILED: {{}} errors", errors);
        std::process::exit(1);
    }}
}}
"#
        )
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
