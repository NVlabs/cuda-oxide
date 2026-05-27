/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! cargo-oxide: Cargo subcommand for building and running cuda-oxide programs.
//!
//! Replaces the xtask pattern with a proper cargo subcommand that works both
//! inside the cuda-oxide repo (for developers) and externally (for users).
//!
//! # Usage
//!
//! ```bash
//! cargo oxide run vecadd              # build + run an example
//! cargo oxide build vecadd            # build only
//! cargo oxide pipeline vecadd         # verbose pipeline dump
//! cargo oxide debug vecadd --tui      # build + cuda-gdb
//! cargo oxide new my_kernel            # scaffold a standalone project
//! cargo oxide new my_kernel --async   # scaffold with async template
//! cargo oxide fmt                     # format all crates
//! cargo oxide doctor                  # check environment
//! cargo oxide setup                   # explicitly build/install backend
//! ```

use clap::{Parser, Subcommand};

mod backend;
mod commands;

/// Top-level CLI structure parsed by clap.
///
/// The binary is named `cargo-oxide` so that `cargo oxide <subcommand>` works
/// as a cargo subcommand. The workspace alias in `.cargo/config.toml` also
/// routes `cargo oxide` here when run inside the repo.
#[derive(Parser)]
#[command(
    name = "cargo-oxide",
    bin_name = "cargo oxide",
    about = "Build and run Rust GPU programs with cuda-oxide",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Available subcommands for `cargo oxide`.
#[derive(Subcommand)]
enum Commands {
    /// Build and run an example or project
    Run {
        /// Example name (required in workspace, optional for standalone projects)
        example: Option<String>,
        /// Generate NVVM IR (use with libNVVM -gen-lto)
        #[arg(long)]
        emit_nvvm_ir: bool,
        /// Target architecture (e.g., sm_90, sm_100, sm_120). When omitted,
        /// `run` auto-detects the compute capability of CUDA device 0 so the
        /// generated module loads on the local GPU; set `CUDA_OXIDE_TARGET`
        /// in the environment for a non-interactive override.
        #[arg(long)]
        arch: Option<String>,
        /// Comma-separated list of features to enable
        #[arg(long)]
        features: Option<String>,
        /// Pick a specific binary in a multi-bin package (forwarded as
        /// `cargo run --bin <name>`). Defaults to the package's
        /// `default-run`.
        #[arg(long)]
        bin: Option<String>,
        /// Show verbose compilation output
        #[arg(short, long)]
        verbose: bool,
    },
    /// Build an example or project (compile only, don't run)
    Build {
        /// Example name (required in workspace, optional for standalone projects)
        example: Option<String>,
        /// Generate NVVM IR (use with libNVVM -gen-lto)
        #[arg(long)]
        emit_nvvm_ir: bool,
        /// Target architecture (e.g., sm_90, sm_100, sm_120)
        #[arg(long)]
        arch: Option<String>,
        /// Comma-separated list of features to enable
        #[arg(long)]
        features: Option<String>,
        /// Show verbose compilation output
        #[arg(short, long)]
        verbose: bool,
    },
    /// Show the full compilation pipeline (MIR -> PTX/NVVM IR) with verbose output
    Pipeline {
        /// Example name (from crates/rustc-codegen-cuda/examples/)
        example: String,
        /// Generate NVVM IR (use with libNVVM -gen-lto)
        #[arg(long)]
        emit_nvvm_ir: bool,
        /// Target architecture (e.g., sm_90, sm_100, sm_120)
        #[arg(long)]
        arch: Option<String>,
    },
    /// Build with debug info and launch cuda-gdb
    Debug {
        /// Example name (from crates/rustc-codegen-cuda/examples/)
        example: String,
        /// Use cgdb frontend (better source view, vim keys)
        #[arg(long)]
        cgdb: bool,
        /// Use GDB's built-in TUI interface
        #[arg(long)]
        tui: bool,
    },
    /// Format all crates (root workspace, codegen backend, examples)
    Fmt {
        /// Check formatting without modifying files
        #[arg(long)]
        check: bool,
    },
    /// Scaffold a new standalone cuda-oxide project
    New {
        /// Project name (becomes directory name and package name)
        name: String,
        /// Use async template (tokio + cuda-async + DeviceOperation)
        #[arg(long = "async")]
        async_mode: bool,
    },
    /// Check that your environment is set up correctly
    Doctor,
    /// Build and cache the codegen backend
    Setup,
}

fn main() {
    // Handle both invocation methods:
    // 1. Cargo subcommand: `cargo oxide run vecadd` → argv = ["cargo-oxide", "oxide", "run", "vecadd"]
    // 2. Cargo alias:      `cargo oxide run vecadd` → argv = ["target/.../cargo-oxide", "run", "vecadd"]
    let args: Vec<String> = std::env::args().collect();
    let effective_args = if args.get(1).map(|s| s.as_str()) == Some("oxide") {
        let mut filtered = vec![args[0].clone()];
        filtered.extend(args[2..].iter().cloned());
        filtered
    } else {
        args
    };

    let cli = Cli::parse_from(effective_args);

    match cli.command {
        Commands::Run {
            example,
            emit_nvvm_ir,
            arch,
            features,
            bin,
            verbose,
        } => {
            let ctx = commands::resolve_context();
            let example = resolve_example_name(example, &ctx);
            validate_nvvm_ir_arch(&example, emit_nvvm_ir, &arch);
            commands::codegen_run(
                &ctx,
                &example,
                verbose,
                emit_nvvm_ir,
                arch.as_deref(),
                features.as_deref(),
                bin.as_deref(),
            );
        }
        Commands::Build {
            example,
            emit_nvvm_ir,
            arch,
            features,
            verbose,
        } => {
            let ctx = commands::resolve_context();
            let example = resolve_example_name(example, &ctx);
            validate_nvvm_ir_arch(&example, emit_nvvm_ir, &arch);
            commands::codegen_build_example(
                &ctx,
                &example,
                verbose,
                emit_nvvm_ir,
                arch.as_deref(),
                features.as_deref(),
            );
        }
        Commands::Pipeline {
            example,
            emit_nvvm_ir,
            arch,
        } => {
            validate_nvvm_ir_arch(&example, emit_nvvm_ir, &arch);
            let ctx = commands::resolve_context();
            commands::codegen_show_pipeline(&ctx, &example, emit_nvvm_ir, arch.as_deref());
        }
        Commands::Debug { example, cgdb, tui } => {
            let ctx = commands::resolve_context();
            commands::codegen_debug(&ctx, &example, cgdb, tui);
        }
        Commands::Fmt { check } => {
            let ctx = commands::resolve_context();
            commands::format_all(&ctx, check);
        }
        Commands::New { name, async_mode } => {
            commands::scaffold_new(&name, async_mode);
        }
        Commands::Doctor => {
            let ctx = commands::resolve_context();
            commands::doctor(&ctx);
        }
        Commands::Setup => {
            let ctx = commands::resolve_context();
            commands::setup(&ctx);
        }
    }
}

/// Resolves the example/project name from the CLI argument or context.
///
/// In workspace mode the name is required; in standalone mode it defaults
/// to the current directory name (which matches the package name from
/// `cargo oxide new`).
fn resolve_example_name(name: Option<String>, ctx: &commands::Context) -> String {
    if let Some(n) = name {
        return n;
    }
    if !ctx.is_workspace {
        return std::env::current_dir()
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_else(|| {
                eprintln!("Error: could not determine project name from current directory");
                std::process::exit(1);
            });
    }
    eprintln!("Error: <EXAMPLE> is required when running inside the cuda-oxide workspace.");
    eprintln!();
    eprintln!("Usage: cargo oxide run <EXAMPLE>");
    eprintln!();
    eprintln!("Available examples are in crates/rustc-codegen-cuda/examples/");
    std::process::exit(1);
}

/// Ensures `--arch` is provided when `--emit-nvvm-ir` is used.
///
/// NVVM IR output is architecture-specific, so omitting `--arch` would produce
/// an unusable artifact. Exits with a descriptive error and usage example.
fn validate_nvvm_ir_arch(example: &str, emit_nvvm_ir: bool, arch: &Option<String>) {
    if emit_nvvm_ir && arch.is_none() {
        eprintln!("Error: --emit-nvvm-ir requires --arch=sm_XXX");
        eprintln!();
        eprintln!("NVVM IR output is architecture-specific. Please specify the target:");
        eprintln!("  --arch sm_120    Blackwell (RTX 50 series)");
        eprintln!("  --arch sm_100    Blackwell");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  cargo oxide run {} --emit-nvvm-ir --arch sm_120", example);
        std::process::exit(1);
    }
}
