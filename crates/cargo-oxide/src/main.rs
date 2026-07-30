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
//! cargo oxide run debug -- --fail-assert  # forward args to the example binary
//! cargo oxide build vecadd            # build only
//! cargo oxide pipeline vecadd         # verbose pipeline dump
//! cargo oxide sanitize vecadd         # run under NVIDIA Compute Sanitizer
//! cargo oxide debug vecadd --tui      # build + cuda-gdb
//! cargo oxide inspect vecadd          # build + print generated PTX
//! cargo oxide new my_kernel           # scaffold a standalone project
//! cargo oxide new my_kernel --async   # scaffold with async template
//! cargo oxide list                    # list bundled examples
//! cargo oxide list --json             # machine-readable output
//! cargo oxide fmt                     # format all crates
//! cargo oxide doctor                  # check environment
//! cargo oxide clean                   # remove local build outputs
//! cargo oxide setup                   # explicitly build/install backend
//! cargo oxide update                  # refresh cached backend (external)
//! ```

use clap::{CommandFactory, Parser, Subcommand, ValueEnum, error::ErrorKind};
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
    /// Compile embedded NVVM IR to a target-specific cubin during the build.
    /// Requires an explicit/configured architecture and exact CUDA-tool
    /// provenance. The final binary then does not need libNVVM or nvJitLink.
    #[arg(long, global = true)]
    materialize_cubin: bool,
    #[command(subcommand)]
    command: Commands,
}

/// Available subcommands for `cargo oxide`.
#[derive(Subcommand)]
enum Commands {
    /// Internal helper: discover exact CUDA compiler provenance in the same
    /// startup environment that will be given to Cargo/rustc.
    #[command(name = "__materializer-provenance", hide = true)]
    MaterializerProvenance,
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
        /// Emit device line-number information for profilers and Compute
        /// Sanitizer, leaving optimization intact (nvcc `-lineinfo`).
        /// Also settable via CUDA_OXIDE_DEBUG=line.
        #[arg(long)]
        lineinfo: bool,
        /// Emit full device debug information; libNVVM finalization runs
        /// unoptimized (nvcc `-G`). Supersedes --lineinfo.
        /// Also settable via CUDA_OXIDE_DEBUG=full.
        #[arg(long)]
        device_debug: bool,
        /// Elide slice/array bounds checks in every device kernel
        /// (out-of-bounds indexing becomes UB, like get_unchecked).
        /// Also settable via CUDA_OXIDE_UNCHECKED_INDEXING=1.
        #[arg(long)]
        unchecked_indexing: bool,
        /// Arguments forwarded to the example binary. Use after `--`,
        /// e.g. `cargo oxide run debug -- --fail-assert`.
        #[arg(last = true, num_args = 0.., allow_hyphen_values = true)]
        app_args: Vec<String>,
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
        /// Emit device line-number information for profilers and Compute
        /// Sanitizer, leaving optimization intact (nvcc `-lineinfo`).
        /// Also settable via CUDA_OXIDE_DEBUG=line.
        #[arg(long)]
        lineinfo: bool,
        /// Emit full device debug information; libNVVM finalization runs
        /// unoptimized (nvcc `-G`). Supersedes --lineinfo.
        /// Also settable via CUDA_OXIDE_DEBUG=full.
        #[arg(long)]
        device_debug: bool,
        /// Elide slice/array bounds checks in every device kernel
        /// (out-of-bounds indexing becomes UB, like get_unchecked).
        /// Also settable via CUDA_OXIDE_UNCHECKED_INDEXING=1.
        #[arg(long)]
        unchecked_indexing: bool,
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
        /// Emit device line-number information for profilers and Compute
        /// Sanitizer, leaving optimization intact (nvcc `-lineinfo`).
        /// Also settable via CUDA_OXIDE_DEBUG=line.
        #[arg(long)]
        lineinfo: bool,
        /// Emit full device debug information; libNVVM finalization runs
        /// unoptimized (nvcc `-G`). Supersedes --lineinfo.
        /// Also settable via CUDA_OXIDE_DEBUG=full.
        #[arg(long)]
        device_debug: bool,
        /// Elide slice/array bounds checks in every device kernel
        /// (out-of-bounds indexing becomes UB, like get_unchecked).
        /// Also settable via CUDA_OXIDE_UNCHECKED_INDEXING=1.
        #[arg(long)]
        unchecked_indexing: bool,
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
        /// Disable FMA contraction (default: on, matching nvcc --fmad=true).
        /// Also settable via CUDA_OXIDE_NO_FMA=1.
        #[arg(long)]
        no_fmad: bool,
        /// Emit device line-number information for profilers and Compute
        /// Sanitizer, leaving optimization intact (nvcc `-lineinfo`).
        /// Also settable via CUDA_OXIDE_DEBUG=line.
        #[arg(long)]
        lineinfo: bool,
        /// Emit full device debug information; libNVVM finalization runs
        /// unoptimized (nvcc `-G`). Supersedes --lineinfo.
        /// Also settable via CUDA_OXIDE_DEBUG=full.
        #[arg(long)]
        device_debug: bool,
        /// Elide slice/array bounds checks in every device kernel
        /// (out-of-bounds indexing becomes UB, like get_unchecked).
        /// Also settable via CUDA_OXIDE_UNCHECKED_INDEXING=1.
        #[arg(long)]
        unchecked_indexing: bool,
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
        /// Emit device line-number information for profilers and Compute
        /// Sanitizer, leaving optimization intact (nvcc `-lineinfo`).
        /// Also settable via CUDA_OXIDE_DEBUG=line.
        #[arg(long)]
        lineinfo: bool,
        /// Emit full device debug information; libNVVM finalization runs
        /// unoptimized (nvcc `-G`). Supersedes --lineinfo.
        /// Also settable via CUDA_OXIDE_DEBUG=full.
        #[arg(long)]
        device_debug: bool,
        /// Elide slice/array bounds checks in every device kernel
        /// (out-of-bounds indexing becomes UB, like get_unchecked).
        /// Also settable via CUDA_OXIDE_UNCHECKED_INDEXING=1.
        #[arg(long)]
        unchecked_indexing: bool,
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
        /// Emit device line-number information for profilers and Compute
        /// Sanitizer, leaving optimization intact (nvcc `-lineinfo`).
        /// Also settable via CUDA_OXIDE_DEBUG=line.
        #[arg(long)]
        lineinfo: bool,
        /// Emit full device debug information; libNVVM finalization runs
        /// unoptimized (nvcc `-G`). Supersedes --lineinfo.
        /// Also settable via CUDA_OXIDE_DEBUG=full.
        #[arg(long)]
        device_debug: bool,
        /// Elide slice/array bounds checks in every device kernel
        /// (out-of-bounds indexing becomes UB, like get_unchecked).
        /// Also settable via CUDA_OXIDE_UNCHECKED_INDEXING=1.
        #[arg(long)]
        unchecked_indexing: bool,
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
        /// Cargo features to enable
        #[arg(long)]
        features: Option<String>,
        /// Specific binary target to build and debug
        #[arg(long)]
        bin: Option<String>,
        /// Use cgdb frontend (better source view, vim keys)
        #[arg(long)]
        cgdb: bool,
        /// Use GDB's built-in TUI interface
        #[arg(long)]
        tui: bool,
    },
    /// List the examples bundled with the cuda-oxide workspace
    List {
        /// Emit stable machine-readable JSON
        #[arg(long)]
        json: bool,
    },
    /// Build an example or project and print the generated PTX
    Inspect {
        /// Example name (required in workspace, optional for standalone projects)
        example: Option<String>,
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
        /// Settable also via CUDA_OXIDE_NO_FMA=1.
        #[arg(long)]
        no_fmad: bool,
        /// Emit device line-number information for profilers and Compute
        /// Sanitizer, leaving optimization intact (nvcc `-lineinfo`).
        /// Also settable via CUDA_OXIDE_DEBUG=line.
        #[arg(long)]
        lineinfo: bool,
        /// Emit full device debug information; libNVVM finalization runs
        /// unoptimized (nvcc `-G`). Supersedes --lineinfo.
        /// Also settable via CUDA_OXIDE_DEBUG=full.
        #[arg(long)]
        device_debug: bool,
        /// Elide slice/array bounds checks in every device kernel.
        /// Settable also via CUDA_OXIDE_UNCHECKED_INDEXING=1.
        #[arg(long)]
        unchecked_indexing: bool,
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
    /// Remove project-local build outputs and generated cuda-oxide artifacts
    Clean,
    /// Check that your environment is set up correctly
    Doctor,
    /// Build and cache the codegen backend
    Setup,
    /// Refresh the cached codegen backend (or run setup inside the workspace)
    Update {
        /// Inside the workspace, run `setup` instead of only advising it.
        /// Outside the workspace, refresh is already the default.
        #[arg(long)]
        force: bool,
    },
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

fn validate_materialization_cli(cli: &Cli) -> Result<(), String> {
    if !cli.materialize_cubin {
        return Ok(());
    }

    match &cli.command {
        Commands::Run {
            emit_nvvm_ir: true,
            ..
        }
        | Commands::Build {
            emit_nvvm_ir: true,
            ..
        }
        | Commands::Pipeline {
            emit_nvvm_ir: true,
            ..
        } => Err(
            "--materialize-cubin cannot be combined with --emit-nvvm-ir; one requests a final cubin and the other requests NVVM IR"
                .to_string(),
        ),
        Commands::Run { .. }
        | Commands::Sanitize { .. }
        | Commands::Build { .. }
        | Commands::Test { .. }
        | Commands::Pipeline { .. }
        | Commands::Debug { .. } => Ok(()),
        Commands::Inspect { .. } => Err(
            "--materialize-cubin cannot be used with inspect because inspect displays PTX"
                .to_string(),
        ),
        Commands::EmitLtoir { .. } => Err(
            "--materialize-cubin cannot be combined with emit-ltoir; one emits a final cubin and the other emits linkable LTOIR"
                .to_string(),
        ),
        Commands::List { .. } => Err(
            "--materialize-cubin cannot be used with list because list does not compile device code"
                .to_string(),
        ),
        Commands::Fmt { .. } => Err(
            "--materialize-cubin cannot be used with fmt because fmt does not compile device code"
                .to_string(),
        ),
        Commands::New { .. } => Err(
            "--materialize-cubin cannot be used with new because new does not compile device code"
                .to_string(),
        ),
        Commands::Clean => Err(
            "--materialize-cubin cannot be used with clean because clean does not compile device code"
                .to_string(),
        ),
        Commands::Doctor => Err(
            "--materialize-cubin cannot be used with doctor because doctor does not compile device code"
                .to_string(),
        ),
        Commands::Setup => Err(
            "--materialize-cubin cannot be used with setup because setup only builds the codegen backend"
                .to_string(),
        ),
        Commands::Update { .. } => Err(
            "--materialize-cubin cannot be used with update because update only refreshes the codegen backend"
                .to_string(),
        ),
        Commands::MaterializerProvenance => Err(
            "--materialize-cubin cannot be passed to the internal materializer discovery helper"
                .to_string(),
        ),
    }
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
    if let Err(error) = validate_materialization_cli(&cli) {
        Cli::command()
            .error(ErrorKind::ArgumentConflict, error)
            .exit();
    }
    let materialize_cubin = cli.materialize_cubin;

    match cli.command {
        Commands::MaterializerProvenance => {
            commands::print_materializer_provenance();
        }
        Commands::Run {
            example,
            emit_nvvm_ir,
            arch,
            features,
            bin,
            verbose,
            no_fmad,
            unchecked_indexing,
            lineinfo,
            device_debug,
            app_args,
        } => {
            let ctx = commands::resolve_context();
            let example = resolve_example_name(example, &ctx, "run");
            validate_output_arch(
                &ctx,
                &example,
                emit_nvvm_ir,
                materialize_cubin,
                arch.as_deref(),
            );
            commands::codegen_run(
                &ctx,
                &example,
                verbose,
                emit_nvvm_ir,
                arch.as_deref(),
                features.as_deref(),
                bin.as_deref(),
                no_fmad,
                unchecked_indexing,
                commands::DeviceDebug::from_flags(lineinfo, device_debug),
                materialize_cubin,
                &app_args,
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
            unchecked_indexing,
            lineinfo,
            device_debug,
            sanitizer_args,
        } => {
            let ctx = commands::resolve_context();
            let example = resolve_example_name(example, &ctx, "sanitize");
            validate_output_arch(&ctx, &example, false, materialize_cubin, arch.as_deref());
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
                unchecked_indexing,
                commands::DeviceDebug::from_flags(lineinfo, device_debug),
                materialize_cubin,
            );
        }
        Commands::Build {
            example,
            emit_nvvm_ir,
            arch,
            features,
            verbose,
            no_fmad,
            unchecked_indexing,
            lineinfo,
            device_debug,
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
                validate_output_arch(
                    &ctx,
                    &example,
                    emit_nvvm_ir,
                    materialize_cubin,
                    arch.as_deref(),
                );
                commands::codegen_build(
                    &ctx,
                    &example,
                    verbose,
                    emit_nvvm_ir,
                    arch.as_deref(),
                    features.as_deref(),
                    no_fmad,
                    unchecked_indexing,
                    commands::DeviceDebug::from_flags(lineinfo, device_debug),
                    materialize_cubin,
                );
            } else {
                if example.is_some() {
                    eprintln!(
                        "Error: `cargo oxide build` accepts either an example name or passthrough args after `--`, not both"
                    );
                    std::process::exit(2);
                }
                validate_output_arch(
                    &ctx,
                    "cargo build",
                    emit_nvvm_ir,
                    materialize_cubin,
                    arch.as_deref(),
                );
                commands::codegen_cargo_passthrough(
                    &ctx,
                    commands::CargoPassthroughSubcommand::Build,
                    commands::CargoPassthroughOptions {
                        verbose,
                        emit_nvvm_ir,
                        arch: arch.as_deref(),
                        features: features.as_deref(),
                        cargo_target_dir: cargo_target_dir.as_deref(),
                        device_codegen_crate: device_codegen_crate.as_deref(),
                        device_cfgs: &device_cfgs,
                        no_fmad,
                        unchecked_indexing,
                        materialize_cubin,
                        device_debug: commands::DeviceDebug::from_flags(lineinfo, device_debug),
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
            no_fmad,
            unchecked_indexing,
            lineinfo,
            device_debug,
            cargo_args,
        } => {
            let ctx = commands::resolve_context();
            validate_output_arch(
                &ctx,
                "cargo test",
                false,
                materialize_cubin,
                arch.as_deref(),
            );
            commands::codegen_cargo_passthrough(
                &ctx,
                commands::CargoPassthroughSubcommand::Test,
                commands::CargoPassthroughOptions {
                    verbose,
                    emit_nvvm_ir: false,
                    arch: arch.as_deref(),
                    features: None,
                    cargo_target_dir: cargo_target_dir.as_deref(),
                    device_codegen_crate: device_codegen_crate.as_deref(),
                    device_cfgs: &device_cfgs,
                    no_fmad,
                    unchecked_indexing,
                    materialize_cubin,
                    device_debug: commands::DeviceDebug::from_flags(lineinfo, device_debug),
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
            unchecked_indexing,
            lineinfo,
            device_debug,
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
                unchecked_indexing,
                commands::DeviceDebug::from_flags(lineinfo, device_debug),
            );
        }
        Commands::Pipeline {
            example,
            emit_nvvm_ir,
            arch,
            no_fmad,
            unchecked_indexing,
            lineinfo,
            device_debug,
        } => {
            let ctx = commands::resolve_context();
            let example = resolve_example_name(example, &ctx, "pipeline");
            validate_output_arch(
                &ctx,
                &example,
                emit_nvvm_ir,
                materialize_cubin,
                arch.as_deref(),
            );
            commands::codegen_show_pipeline(
                &ctx,
                &example,
                emit_nvvm_ir,
                arch.as_deref(),
                no_fmad,
                unchecked_indexing,
                commands::DeviceDebug::from_flags(lineinfo, device_debug),
                materialize_cubin,
            );
        }
        Commands::Inspect {
            example,
            arch,
            features,
            verbose,
            no_fmad,
            unchecked_indexing,
            lineinfo,
            device_debug,
        } => {
            let ctx = commands::resolve_context();
            let example = resolve_example_name(example, &ctx, "inspect");
            commands::codegen_inspect_ptx(
                &ctx,
                &example,
                arch.as_deref(),
                features.as_deref(),
                verbose,
                no_fmad,
                unchecked_indexing,
                commands::DeviceDebug::from_flags(lineinfo, device_debug),
            );
        }
        Commands::Debug {
            example,
            arch,
            features,
            bin,
            cgdb,
            tui,
        } => {
            let ctx = commands::resolve_context();
            let example = resolve_example_name(example, &ctx, "debug");
            validate_output_arch(&ctx, &example, false, materialize_cubin, arch.as_deref());
            commands::codegen_debug(
                &ctx,
                &example,
                arch.as_deref(),
                features.as_deref(),
                bin.as_deref(),
                cgdb,
                tui,
                materialize_cubin,
            );
        }
        Commands::List { json } => {
            let ctx = commands::resolve_passive_context();
            commands::list_examples(&ctx, json);
        }
        Commands::Fmt { check } => {
            let ctx = commands::resolve_context();
            commands::format_all(&ctx, check);
        }
        Commands::New { name, async_mode } => {
            commands::scaffold_new(&name, async_mode);
        }
        Commands::Clean => {
            let ctx = commands::resolve_passive_context();
            commands::clean(&ctx);
        }
        Commands::Doctor => {
            // Side-effect-free resolver: doctor must never build the backend
            // (or clone anything) before it can diagnose the environment.
            let ctx = commands::resolve_passive_context();
            commands::doctor(&ctx);
        }
        Commands::Setup => {
            let ctx = commands::resolve_context();
            commands::setup(&ctx);
        }
        Commands::Update { force } => {
            // All plans resolve passively. `setup` only needs
            // `ctx.codegen_crate`, which the passive resolver provides
            // identically, while eager resolution would auto-fetch and build
            // the backend before `update` runs: a double clone+build ahead of
            // the RefreshCache arm's own clear-and-rebuild, and a wasted
            // build ahead of the pinned-backend refusals.
            let ctx = commands::resolve_passive_context();
            commands::update(&ctx, force);
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
fn validate_output_arch(
    ctx: &commands::Context,
    example: &str,
    emit_nvvm_ir: bool,
    materialize_cubin: bool,
    arch: Option<&str>,
) {
    if (emit_nvvm_ir || materialize_cubin) && !commands::has_configured_arch(ctx, arch) {
        let option = if materialize_cubin {
            "--materialize-cubin"
        } else {
            "--emit-nvvm-ir"
        };
        eprintln!("Error: {option} requires a target architecture");
        eprintln!();
        eprintln!("NVVM IR output is architecture-specific. Pass --arch, set");
        eprintln!("CUDA_OXIDE_TARGET, or configure default-arch. For example:");
        eprintln!("  --arch sm_120    Blackwell (RTX 50 series)");
        eprintln!("  --arch sm_100    Blackwell");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  cargo oxide run {example} {option} --arch sm_120");
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
    fn clean_parser_accepts_command_without_arguments() {
        let cli =
            Cli::try_parse_from(["cargo-oxide", "clean"]).expect("clean command should parse");

        assert!(matches!(cli.command, Commands::Clean));
    }

    #[test]
    fn update_parser_accepts_force_flag() {
        let plain =
            Cli::try_parse_from(["cargo-oxide", "update"]).expect("update command should parse");
        let Commands::Update { force } = plain.command else {
            panic!("expected Update");
        };
        assert!(!force);

        let forced = Cli::try_parse_from(["cargo-oxide", "update", "--force"])
            .expect("update --force should parse");
        let Commands::Update { force } = forced.command else {
            panic!("expected Update");
        };
        assert!(force);
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
    fn materialize_cubin_is_a_global_codegen_option() {
        let after_subcommand = Cli::try_parse_from([
            "cargo-oxide",
            "build",
            "demo",
            "--materialize-cubin",
            "--arch",
            "sm_90",
        ])
        .unwrap();
        assert!(after_subcommand.materialize_cubin);
        assert!(validate_materialization_cli(&after_subcommand).is_ok());

        let before_subcommand = Cli::try_parse_from([
            "cargo-oxide",
            "--materialize-cubin",
            "test",
            "--arch",
            "sm_90",
        ])
        .unwrap();
        assert!(before_subcommand.materialize_cubin);
        assert!(validate_materialization_cli(&before_subcommand).is_ok());
    }

    #[test]
    fn materialize_cubin_rejects_non_codegen_subcommands() {
        for args in [
            &["cargo-oxide", "fmt", "--materialize-cubin"][..],
            &["cargo-oxide", "new", "demo", "--materialize-cubin"],
            &["cargo-oxide", "doctor", "--materialize-cubin"],
            &["cargo-oxide", "setup", "--materialize-cubin"],
            &["cargo-oxide", "clean", "--materialize-cubin"],
            &[
                "cargo-oxide",
                "emit-ltoir",
                "demo",
                "--arch",
                "sm_90",
                "--materialize-cubin",
            ],
            &["cargo-oxide", "inspect", "demo", "--materialize-cubin"],
        ] {
            let cli = Cli::try_parse_from(args).expect("global flag should parse first");
            let error = validate_materialization_cli(&cli)
                .expect_err("materialization must be rejected for this subcommand");
            assert!(error.contains("--materialize-cubin"), "{error}");
        }
    }

    #[test]
    fn materialize_cubin_rejects_explicit_nvvm_ir_output() {
        for subcommand in ["run", "build", "pipeline"] {
            let cli = Cli::try_parse_from([
                "cargo-oxide",
                subcommand,
                "demo",
                "--emit-nvvm-ir",
                "--materialize-cubin",
                "--arch",
                "sm_90",
            ])
            .unwrap();
            let error = validate_materialization_cli(&cli)
                .expect_err("two distinct final output modes must conflict");
            assert!(error.contains("cannot be combined with --emit-nvvm-ir"));
        }
    }

    #[test]
    fn empty_test_and_explicit_empty_build_passthrough_are_distinct() {
        let test_cli = Cli::try_parse_from(["cargo-oxide", "test"])
            .expect("cargo oxide test should accept no Cargo arguments");
        let Commands::Test {
            cargo_args,
            no_fmad,
            unchecked_indexing,
            lineinfo,
            device_debug,
            ..
        } = test_cli.command
        else {
            panic!("expected test command");
        };
        assert!(cargo_args.is_empty());
        assert!(!no_fmad);
        assert!(!unchecked_indexing);
        assert!(!lineinfo, "--lineinfo must default off");
        assert!(!device_debug, "--device-debug must default off");

        let build_args = strings(&["cargo-oxide", "build", "--"]);
        assert!(has_passthrough_separator(&build_args));
        let build_cli = Cli::try_parse_from(build_args).expect("empty passthrough should parse");
        let Commands::Build { cargo_args, .. } = build_cli.command else {
            panic!("expected build command");
        };
        assert!(cargo_args.is_empty());
    }

    #[test]
    fn test_parser_accepts_codegen_flags() {
        let cli = Cli::try_parse_from([
            "cargo-oxide",
            "test",
            "--no-fmad",
            "--unchecked-indexing",
            "--",
            "-p",
            "gpu-app",
        ])
        .expect("test codegen flags should parse");
        let Commands::Test {
            no_fmad,
            unchecked_indexing,
            lineinfo,
            device_debug,
            cargo_args,
            ..
        } = cli.command
        else {
            panic!("expected test command");
        };
        assert!(no_fmad);
        assert!(unchecked_indexing);
        assert!(!lineinfo, "--lineinfo must default off");
        assert!(!device_debug, "--device-debug must default off");
        assert_eq!(cargo_args, strings(&["-p", "gpu-app"]));
    }

    #[test]
    fn run_parser_forwards_trailing_application_args() {
        let cli = Cli::try_parse_from([
            "cargo-oxide",
            "run",
            "debug",
            "--",
            "--fail-assert",
            "positional",
        ])
        .expect("run command with trailing args should parse");

        let Commands::Run {
            example, app_args, ..
        } = cli.command
        else {
            panic!("expected run command");
        };
        assert_eq!(example.as_deref(), Some("debug"));
        assert_eq!(app_args, strings(&["--fail-assert", "positional"]));
    }

    #[test]
    fn run_parser_defaults_to_no_application_args() {
        let cli =
            Cli::try_parse_from(["cargo-oxide", "run", "debug"]).expect("run command should parse");

        let Commands::Run { app_args, .. } = cli.command else {
            panic!("expected run command");
        };
        assert!(app_args.is_empty());
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
    fn debug_parser_accepts_bin_and_features() {
        let cli = Cli::try_parse_from([
            "cargo-oxide",
            "debug",
            "my_app",
            "--bin",
            "debug-target",
            "--features",
            "foo,bar",
            "--tui",
        ])
        .expect("debug command should parse");

        let Commands::Debug {
            example,
            features,
            bin,
            tui,
            ..
        } = cli.command
        else {
            panic!("expected debug command");
        };
        assert_eq!(example.as_deref(), Some("my_app"));
        assert_eq!(bin.as_deref(), Some("debug-target"));
        assert_eq!(features.as_deref(), Some("foo,bar"));
        assert!(tui);
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

    #[test]
    fn list_parser_defaults_to_human_output() {
        let cli = Cli::try_parse_from(["cargo-oxide", "list"]).expect("list command should parse");

        let Commands::List { json } = cli.command else {
            panic!("expected list command");
        };

        assert!(!json);
    }

    #[test]
    fn list_parser_accepts_json_output() {
        let cli = Cli::try_parse_from(["cargo-oxide", "list", "--json"])
            .expect("list --json should parse");

        let Commands::List { json } = cli.command else {
            panic!("expected list command");
        };

        assert!(json);
    }

    #[test]
    fn materialize_cubin_rejects_list() {
        let cli = Cli::try_parse_from(["cargo-oxide", "list", "--materialize-cubin"])
            .expect("global option should parse");

        let error =
            validate_materialization_cli(&cli).expect_err("list must reject materialization");

        assert!(error.contains("--materialize-cubin"));
        assert!(error.contains("list"));
    }

    #[test]
    fn inspect_parser_accepts_codegen_options() {
        let cli = Cli::try_parse_from([
            "cargo-oxide",
            "inspect",
            "vecadd",
            "--arch",
            "sm_90",
            "--features",
            "foo,bar",
            "--verbose",
            "--no-fmad",
            "--unchecked-indexing",
        ])
        .expect("inspect command should parse");

        let Commands::Inspect {
            example,
            arch,
            features,
            verbose,
            no_fmad,
            unchecked_indexing,
            lineinfo,
            device_debug,
        } = cli.command
        else {
            panic!("expected inspect command");
        };

        assert_eq!(example.as_deref(), Some("vecadd"));
        assert_eq!(arch.as_deref(), Some("sm_90"));
        assert_eq!(features.as_deref(), Some("foo,bar"));
        assert!(verbose);
        assert!(no_fmad);
        assert!(unchecked_indexing);
        assert!(!lineinfo, "--lineinfo must default off");
        assert!(!device_debug, "--device-debug must default off");
    }

    #[test]
    fn parser_accepts_device_debug_flags() {
        let cli = Cli::try_parse_from(["cargo-oxide", "build", "vecadd", "--lineinfo"])
            .expect("--lineinfo should parse");
        let Commands::Build {
            lineinfo,
            device_debug,
            ..
        } = cli.command
        else {
            panic!("expected build command");
        };
        assert!(lineinfo);
        assert!(!device_debug);

        // #552 gave `test` the codegen flags, so the debug policy belongs there too.
        let cli = Cli::try_parse_from(["cargo-oxide", "test", "--device-debug"])
            .expect("--device-debug should parse on test");
        let Commands::Test {
            lineinfo,
            device_debug,
            ..
        } = cli.command
        else {
            panic!("expected test command");
        };
        assert!(!lineinfo);
        assert!(device_debug);
    }

    #[test]
    fn device_debug_flags_resolve_in_nvcc_order() {
        use commands::DeviceDebug;
        // Absent flags stay Off so an ambient CUDA_OXIDE_DEBUG survives.
        assert_eq!(DeviceDebug::from_flags(false, false), DeviceDebug::Off);
        assert_eq!(
            DeviceDebug::from_flags(true, false),
            DeviceDebug::LineTables
        );
        assert_eq!(DeviceDebug::from_flags(false, true), DeviceDebug::Full);
        // Full debug already carries line tables, so it wins over --lineinfo.
        assert_eq!(DeviceDebug::from_flags(true, true), DeviceDebug::Full);
    }
}
