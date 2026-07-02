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
//! cargo oxide sanitize vecadd         # run under NVIDIA Compute Sanitizer
//! cargo oxide debug vecadd --tui      # build + cuda-gdb
//! cargo oxide new my_kernel            # scaffold a standalone project
//! cargo oxide new my_kernel --async   # scaffold with async template
//! cargo oxide fmt                     # format all crates
//! cargo oxide doctor                  # check environment
//! cargo oxide setup                   # explicitly build/install backend
//! ```

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

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
        /// Disable FMA contraction (default: on, matching nvcc --fmad=true).
        /// Also settable via CUDA_OXIDE_NO_FMA=1.
        #[arg(long)]
        no_fmad: bool,
    },
    /// Build and run an example or project under NVIDIA Compute Sanitizer
    Sanitize {
        /// Example name (required in workspace, optional for standalone projects)
        example: Option<String>,
        /// Compute Sanitizer tool to run
        #[arg(long, value_enum, default_value_t = SanitizerTool::Memcheck)]
        tool: SanitizerTool,
        /// Target architecture (e.g., sm_90, sm_100, sm_120). When omitted,
        /// `sanitize` uses the same local-GPU target detection as `run`.
        #[arg(long)]
        arch: Option<String>,
        /// Comma-separated list of features to enable
        #[arg(long)]
        features: Option<String>,
        /// Pick a specific binary in a multi-bin package
        #[arg(long)]
        bin: Option<String>,
        /// Show verbose compilation output
        #[arg(short, long)]
        verbose: bool,
        /// Disable implicit FMA contraction in device codegen.
        /// Also settable via CUDA_OXIDE_NO_FMA=1.
        #[arg(long)]
        no_fmad: bool,
        /// Additional arguments passed to compute-sanitizer before the binary.
        /// Use a second `--` inside this list to pass arguments to the target
        /// program after the binary.
        #[arg(last = true, num_args = 0.., allow_hyphen_values = true)]
        sanitizer_args: Vec<String>,
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
        /// Disable FMA contraction (default: on, matching nvcc --fmad=true).
        /// Also settable via CUDA_OXIDE_NO_FMA=1.
        #[arg(long)]
        no_fmad: bool,
        /// Cargo target directory for passthrough mode
        #[arg(long)]
        cargo_target_dir: Option<PathBuf>,
        /// Comma-separated cuda-oxide owner crate filter for device codegen
        #[arg(long)]
        device_codegen_crate: Option<String>,
        /// Repeatable cfg appended as `--cfg NAME` for passthrough device codegen
        #[arg(long = "device-cfg")]
        device_cfgs: Vec<String>,
        /// Cargo build arguments for passthrough mode. Use after `--`.
        #[arg(last = true, num_args = 0.., allow_hyphen_values = true)]
        cargo_args: Vec<String>,
    },
    /// Run Cargo tests through the cuda-oxide backend
    Test {
        /// Target architecture (e.g., sm_90, sm_100, sm_120)
        #[arg(long)]
        arch: Option<String>,
        /// Cargo target directory
        #[arg(long)]
        cargo_target_dir: Option<PathBuf>,
        /// Comma-separated cuda-oxide owner crate filter for device codegen
        #[arg(long)]
        device_codegen_crate: Option<String>,
        /// Repeatable cfg appended as `--cfg NAME` for device codegen
        #[arg(long = "device-cfg")]
        device_cfgs: Vec<String>,
        /// Show verbose compilation output
        #[arg(short, long)]
        verbose: bool,
        /// Cargo test arguments. Use after `--`; empty runs plain `cargo test`.
        #[arg(last = true, num_args = 0.., allow_hyphen_values = true)]
        cargo_args: Vec<String>,
    },
    /// Compile a crate's device code to a binary LTOIR artifact in one step.
    ///
    /// Produces the SIMT artifact a tile or C++ kernel links against
    /// (NVVM IR emission followed by libNVVM `-gen-lto`), writing
    /// `<crate>.ltoir` plus target/options sidecars. See the Tile-to-SIMT
    /// interop tracker (#96).
    EmitLtoir {
        /// Crate name (required in workspace, optional for standalone projects)
        example: Option<String>,
        /// Target architecture (e.g. sm_90, sm_100, sm_120). Required: LTOIR is
        /// architecture-specific.
        #[arg(long)]
        arch: String,
        /// Comma-separated list of features to enable
        #[arg(long)]
        features: Option<String>,
        /// Output path for the `.ltoir` file (default: `<crate-dir>/<crate>.ltoir`)
        #[arg(long, short)]
        output: Option<PathBuf>,
        /// Show verbose compilation output
        #[arg(short, long)]
        verbose: bool,
        /// Disable implicit FMA contraction in both libNVVM and nvJitLink.
        /// Also settable via CUDA_OXIDE_NO_FMA=1.
        #[arg(long)]
        no_fmad: bool,
    },
    /// Show the full compilation pipeline (MIR -> PTX/NVVM IR) with verbose output
    Pipeline {
        /// Example name (required in workspace, optional for standalone projects)
        example: Option<String>,
        /// Generate NVVM IR (use with libNVVM -gen-lto)
        #[arg(long)]
        emit_nvvm_ir: bool,
        /// Target architecture (e.g., sm_90, sm_100, sm_120)
        #[arg(long)]
        arch: Option<String>,
        /// Disable FMA contraction (default: on, matching nvcc --fmad=true).
        /// Also settable via CUDA_OXIDE_NO_FMA=1.
        #[arg(long)]
        no_fmad: bool,
    },
    /// Build with debug info and launch cuda-gdb
    Debug {
        /// Example name (required in workspace, optional for standalone projects)
        example: Option<String>,
        /// Target architecture (e.g., sm_90, sm_100, sm_120). When omitted,
        /// `debug` auto-detects the compute capability of CUDA device 0 so the
        /// generated module loads on the local GPU; set `CUDA_OXIDE_TARGET`
        /// in the environment for a non-interactive override.
        #[arg(long)]
        arch: Option<String>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum SanitizerTool {
    Memcheck,
    Racecheck,
    Initcheck,
    Synccheck,
}

impl SanitizerTool {
    fn as_str(self) -> &'static str {
        match self {
            Self::Memcheck => "memcheck",
            Self::Racecheck => "racecheck",
            Self::Initcheck => "initcheck",
            Self::Synccheck => "synccheck",
        }
    }
}

fn split_sanitizer_and_application_args(args: &[String]) -> (Vec<String>, Vec<String>) {
    match args.iter().position(|arg| arg == "--") {
        Some(separator) => (args[..separator].to_vec(), args[separator + 1..].to_vec()),
        None => (args.to_vec(), Vec::new()),
    }
}

fn has_passthrough_separator(args: &[String]) -> bool {
    args.iter().skip(2).any(|arg| arg == "--")
}

fn use_build_passthrough(
    explicit_separator: bool,
    cargo_target_dir_is_set: bool,
    owner_filter_is_set: bool,
    has_device_cfgs: bool,
    has_cargo_args: bool,
) -> bool {
    explicit_separator
        || cargo_target_dir_is_set
        || owner_filter_is_set
        || has_device_cfgs
        || has_cargo_args
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

    let explicit_passthrough = has_passthrough_separator(&effective_args);
    let cli = Cli::parse_from(effective_args);

    match cli.command {
        Commands::Run {
            example,
            emit_nvvm_ir,
            arch,
            features,
            bin,
            verbose,
            no_fmad,
        } => {
            let ctx = commands::resolve_context();
            let example = resolve_example_name(example, &ctx, "run");
            validate_nvvm_ir_arch(&ctx, &example, emit_nvvm_ir, arch.as_deref());
            commands::codegen_run(
                &ctx,
                &example,
                verbose,
                emit_nvvm_ir,
                arch.as_deref(),
                features.as_deref(),
                bin.as_deref(),
                no_fmad,
            );
        }
        Commands::Sanitize {
            example,
            tool,
            arch,
            features,
            bin,
            verbose,
            no_fmad,
            sanitizer_args,
        } => {
            let ctx = commands::resolve_context();
            let example = resolve_example_name(example, &ctx, "sanitize");
            let (sanitizer_args, application_args) =
                split_sanitizer_and_application_args(&sanitizer_args);
            commands::codegen_sanitize(
                &ctx,
                &example,
                tool.as_str(),
                &sanitizer_args,
                &application_args,
                verbose,
                arch.as_deref(),
                features.as_deref(),
                bin.as_deref(),
                no_fmad,
            );
        }
        Commands::Build {
            example,
            emit_nvvm_ir,
            arch,
            features,
            verbose,
            no_fmad,
            cargo_target_dir,
            device_codegen_crate,
            device_cfgs,
            cargo_args,
        } => {
            let ctx = commands::resolve_context();
            let passthrough = use_build_passthrough(
                explicit_passthrough,
                cargo_target_dir.is_some(),
                device_codegen_crate.is_some(),
                !device_cfgs.is_empty(),
                !cargo_args.is_empty(),
            );
            if !passthrough {
                let example = resolve_example_name(example, &ctx, "build");
                validate_nvvm_ir_arch(&ctx, &example, emit_nvvm_ir, arch.as_deref());
                commands::codegen_build(
                    &ctx,
                    &example,
                    verbose,
                    emit_nvvm_ir,
                    arch.as_deref(),
                    features.as_deref(),
                    no_fmad,
                );
            } else {
                if example.is_some() {
                    eprintln!(
                        "Error: `cargo oxide build` accepts either an example name or passthrough args after `--`, not both"
                    );
                    std::process::exit(2);
                }
                validate_nvvm_ir_arch(&ctx, "cargo build", emit_nvvm_ir, arch.as_deref());
                commands::codegen_cargo_passthrough(
                    &ctx,
                    "build",
                    commands::CargoPassthroughOptions {
                        verbose,
                        emit_nvvm_ir,
                        arch: arch.as_deref(),
                        features: features.as_deref(),
                        cargo_target_dir: cargo_target_dir.as_deref(),
                        device_codegen_crate: device_codegen_crate.as_deref(),
                        device_cfgs: &device_cfgs,
                        no_fmad,
                    },
                    &cargo_args,
                );
            }
        }
        Commands::Test {
            arch,
            cargo_target_dir,
            device_codegen_crate,
            device_cfgs,
            verbose,
            cargo_args,
        } => {
            let ctx = commands::resolve_context();
            commands::codegen_cargo_passthrough(
                &ctx,
                "test",
                commands::CargoPassthroughOptions {
                    verbose,
                    emit_nvvm_ir: false,
                    arch: arch.as_deref(),
                    features: None,
                    cargo_target_dir: cargo_target_dir.as_deref(),
                    device_codegen_crate: device_codegen_crate.as_deref(),
                    device_cfgs: &device_cfgs,
                    no_fmad: false,
                },
                &cargo_args,
            );
        }
        Commands::EmitLtoir {
            example,
            arch,
            features,
            output,
            verbose,
            no_fmad,
        } => {
            let ctx = commands::resolve_context();
            let example = resolve_example_name(example, &ctx, "emit-ltoir");
            commands::emit_ltoir(
                &ctx,
                &example,
                &arch,
                features.as_deref(),
                output.as_deref(),
                verbose,
                no_fmad,
            );
        }
        Commands::Pipeline {
            example,
            emit_nvvm_ir,
            arch,
            no_fmad,
        } => {
            let ctx = commands::resolve_context();
            let example = resolve_example_name(example, &ctx, "pipeline");
            validate_nvvm_ir_arch(&ctx, &example, emit_nvvm_ir, arch.as_deref());
            commands::codegen_show_pipeline(&ctx, &example, emit_nvvm_ir, arch.as_deref(), no_fmad);
        }
        Commands::Debug {
            example,
            arch,
            cgdb,
            tui,
        } => {
            let ctx = commands::resolve_context();
            let example = resolve_example_name(example, &ctx, "debug");
            commands::codegen_debug(&ctx, &example, arch.as_deref(), cgdb, tui);
        }
        Commands::Fmt { check } => {
            let ctx = commands::resolve_context();
            commands::format_all(&ctx, check);
        }
        Commands::New { name, async_mode } => {
            commands::scaffold_new(&name, async_mode);
        }
        Commands::Doctor => {
            // Side-effect-free resolver: doctor must never build the backend
            // (or clone anything) before it can diagnose the environment.
            let ctx = commands::resolve_doctor_context();
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
fn resolve_example_name(name: Option<String>, ctx: &commands::Context, subcommand: &str) -> String {
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
    eprintln!("Usage: cargo oxide {subcommand} <EXAMPLE>");
    eprintln!();
    eprintln!("Available examples are in crates/rustc-codegen-cuda/examples/");
    std::process::exit(1);
}

/// Ensures an architecture is configured when `--emit-nvvm-ir` is used.
///
/// NVVM IR output is architecture-specific, so omitting every target source
/// would produce an unusable artifact. Exits with a descriptive error.
fn validate_nvvm_ir_arch(
    ctx: &commands::Context,
    example: &str,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
) {
    if emit_nvvm_ir && !commands::has_configured_arch(ctx, arch) {
        eprintln!("Error: --emit-nvvm-ir requires a target architecture");
        eprintln!();
        eprintln!("NVVM IR output is architecture-specific. Pass --arch, set");
        eprintln!("CUDA_OXIDE_TARGET, or configure default-arch. For example:");
        eprintln!("  --arch sm_120    Blackwell (RTX 50 series)");
        eprintln!("  --arch sm_100    Blackwell");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  cargo oxide run {} --emit-nvvm-ir --arch sm_120", example);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| (*arg).to_string()).collect()
    }

    #[test]
    fn build_parser_preserves_nested_cargo_and_test_separators() {
        let args = strings(&[
            "cargo-oxide",
            "build",
            "--cargo-target-dir",
            "target/cuda",
            "--",
            "-p",
            "gpu-app",
            "--test",
            "smoke",
            "--",
            "--nocapture",
        ]);
        assert!(has_passthrough_separator(&args));

        let cli = Cli::try_parse_from(args).expect("passthrough CLI should parse");
        let Commands::Build {
            cargo_target_dir,
            cargo_args,
            ..
        } = cli.command
        else {
            panic!("expected build command");
        };
        assert_eq!(cargo_target_dir, Some(PathBuf::from("target/cuda")));
        assert_eq!(
            cargo_args,
            strings(&["-p", "gpu-app", "--test", "smoke", "--", "--nocapture"])
        );
    }

    #[test]
    fn empty_test_and_explicit_empty_build_passthrough_are_distinct() {
        let test_cli = Cli::try_parse_from(["cargo-oxide", "test"])
            .expect("cargo oxide test should accept no Cargo arguments");
        let Commands::Test { cargo_args, .. } = test_cli.command else {
            panic!("expected test command");
        };
        assert!(cargo_args.is_empty());

        let build_args = strings(&["cargo-oxide", "build", "--"]);
        assert!(has_passthrough_separator(&build_args));
        let build_cli = Cli::try_parse_from(build_args).expect("empty passthrough should parse");
        let Commands::Build { cargo_args, .. } = build_cli.command else {
            panic!("expected build command");
        };
        assert!(cargo_args.is_empty());
    }

    #[test]
    fn sanitize_parser_accepts_tool_and_trailing_sanitizer_args() {
        let cli = Cli::try_parse_from([
            "cargo-oxide",
            "sanitize",
            "vecadd",
            "--tool",
            "racecheck",
            "--",
            "--kernel-name",
            "kns=vecadd",
        ])
        .expect("sanitize command should parse");

        let Commands::Sanitize {
            example,
            tool,
            sanitizer_args,
            ..
        } = cli.command
        else {
            panic!("expected sanitize command");
        };
        assert_eq!(example.as_deref(), Some("vecadd"));
        assert_eq!(tool, SanitizerTool::Racecheck);
        assert_eq!(sanitizer_args, strings(&["--kernel-name", "kns=vecadd"]));
    }

    #[test]
    fn sanitize_args_split_at_second_separator_for_application_args() {
        let raw_args = strings(&[
            "--leak-check",
            "full",
            "--",
            "--case",
            "oob",
            "--verbose-target",
        ]);

        let (sanitizer_args, application_args) = split_sanitizer_and_application_args(&raw_args);

        assert_eq!(sanitizer_args, strings(&["--leak-check", "full"]));
        assert_eq!(
            application_args,
            strings(&["--case", "oob", "--verbose-target"])
        );
    }

    #[test]
    fn sanitize_parser_defaults_to_memcheck() {
        let cli = Cli::try_parse_from(["cargo-oxide", "sanitize", "vecadd"])
            .expect("sanitize command should parse");

        let Commands::Sanitize { tool, .. } = cli.command else {
            panic!("expected sanitize command");
        };
        assert_eq!(tool, SanitizerTool::Memcheck);
    }

    #[test]
    fn build_mode_uses_only_unambiguous_passthrough_signals() {
        assert!(!use_build_passthrough(false, false, false, false, false));
        assert!(use_build_passthrough(true, false, false, false, false));
        assert!(use_build_passthrough(false, true, false, false, false));
        assert!(use_build_passthrough(false, false, true, false, false));
        assert!(use_build_passthrough(false, false, false, true, false));
        assert!(use_build_passthrough(false, false, false, false, true));
    }
}
