/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compilation pipeline: MIR → `dialect-mir` → LLVM dialect → LLVM IR → PTX.
//!
//! Orchestrates the full compilation flow from collected MIR functions to
//! executable PTX code.
//!
//! # Pipeline Steps
//!
//! ```text
//! MIR -> dialect-mir -> verify -> mem2reg -> annotated loop unroll
//!     -> LLVM dialect -> LLVM IR -> PTX
//! ```
//!
//! Builds with variable debug information skip `mem2reg` and loop unrolling so
//! source variables remain in stable stack slots.
//!
//! # GPU Target Selection
//!
//! The pipeline auto-detects GPU features in the generated LLVM IR and selects
//! an appropriate target:
//!
//! | Feature                       | Target  | Architecture         |
//! |-------------------------------|---------|----------------------|
//! | tcgen05/TMEM                  | sm_100a | Blackwell datacenter |
//! | TMA multicast                 | sm_100a | Blackwell datacenter |
//! | WGMMA                         | sm_90a  | Hopper only          |
//! | TMA/mbarrier                  | sm_100  | Hopper+ compatible   |
//! | bf16x2 add/sub/mul            | sm_90   | Hopper+ compatible   |
//! | other bf16x2 ALU              | sm_80   | Ampere+ compatible   |
//! | INT8 `mma.m16n8k32`           | sm_80   | PTX 7.0+             |
//! | `cp.async` (non-bulk)         | sm_80   | Ampere+              |
//! | Basic CUDA                    | sm_80   | Ampere+ (max compat) |
//!
//! Override with `CUDA_OXIDE_TARGET=<target>` environment variable.

use cuda_oxide_codegen::__private::{
    BackendOptions, ModuleArtifactKind, ModulePipelineRequest, OutputFiles, PipelineTrace,
    append_to_module, compile_translated_module, verify_operation,
};
pub use cuda_oxide_codegen::__private::{DeviceExternAttrs, DeviceExternDecl, PipelineError};
use llvm_export::export::DebugKind;
pub use llvm_export::export::DeviceExternType;
use pliron::context::Context;
use pliron::identifier::Legaliser;
use pliron::op::Op;
use pliron::printable::Printable;
use rustc_public::mir::mono::Instance;
use std::path::Path;

fn stderr_pipeline_trace(message: &str) {
    eprintln!("{message}");
}

/// A function collected for GPU compilation.
///
/// Represents a monomorphized function instance that will be translated to PTX.
/// For generic functions like `add::<f32>`, the instance contains the concrete
/// type substitutions.
#[derive(Debug, Clone)]
pub struct CollectedFunction {
    /// The monomorphized stable_mir instance (includes concrete generic args).
    pub instance: Instance,
    /// True if this is a GPU kernel entry point (has `#[kernel]` attribute).
    pub is_kernel: bool,
    /// The name to export in PTX. For kernels, this is the user-visible name.
    pub export_name: String,
    /// rustc MIR source-scope data used to build inlined debug scopes.
    pub debug_source_scopes: Option<llvm_export::ops::DebugSourceScopeMap>,
    /// True if the function is marked `#[inline(always)]` in rustc's
    /// `CodegenFnAttrs`. The stable_mir API does not expose inline hints, so
    /// this is queried via `rustc_middle::TyCtxt::codegen_fn_attrs` in
    /// `rustc-codegen-cuda` and threaded through.
    ///
    /// When true, the LLVM `alwaysinline` attribute is emitted on the
    /// function definition. The existing matched LLVM middle-end (`opt -O2`),
    /// when available, can then honor the attribute before PTX generation;
    /// this flag does not add a separate mandatory inliner pass.
    ///
    /// This preserves Rust's inline intent for device helpers and avoids
    /// making helper boundaries depend entirely on later optimizer heuristics.
    pub is_inline_always: bool,
}

/// Device artifact format produced by a successful pipeline run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilationArtifactKind {
    /// Textual PTX assembly, loadable by the CUDA driver.
    Ptx,
    /// NVVM-compatible LLVM IR, intended for libNVVM/nvJitLink.
    NvvmIr,
    /// Binary LTOIR, intended for nvJitLink.
    Ltoir,
    /// Final cubin image, loadable by the CUDA driver.
    Cubin,
}

/// Output paths, target, and artifact format from successful compilation.
pub struct CompilationResult {
    /// Path to generated LLVM IR (`.ll` file).
    pub ll_path: std::path::PathBuf,
    /// Path to generated PTX assembly (`.ptx` file).
    pub ptx_path: std::path::PathBuf,
    /// Path to the artifact that should be embedded or consumed by the caller.
    pub artifact_path: std::path::PathBuf,
    /// Format of `artifact_path`.
    pub artifact_kind: CompilationArtifactKind,
    /// GPU target architecture used (e.g., `sm_90a`, `sm_80`).
    pub target: String,
    /// Floating-point contraction policy that later compilation stages must
    /// preserve.
    pub allow_fma_contraction: bool,
}

/// Configuration for the compilation pipeline.
pub struct PipelineConfig {
    /// Directory for output files (`.ll`, `.ptx`).
    pub output_dir: std::path::PathBuf,
    /// Base name for output files (e.g., `"kernel"` → `kernel.ll`, `kernel.ptx`).
    pub output_name: String,
    /// Print progress messages to stdout.
    pub verbose: bool,
    /// Dump the `dialect-mir` module after translation (for debugging).
    pub show_mir_dialect: bool,
    /// Dump the LLVM dialect module after lowering (for debugging).
    pub show_llvm_dialect: bool,
    /// Emit NVVM IR suitable for libNVVM or other NVVM-compatible tools.
    ///
    /// When true:
    /// - Uses full NVPTX datalayout
    /// - Adds `@llvm.used` to preserve kernels from optimization
    /// - Adds `!nvvm.annotations` for all kernels
    /// - Adds `!nvvmir.version` metadata
    /// - Outputs `.ll` file in NVVM IR format
    ///
    /// The output can be compiled to LTOIR using `nvvmCompileProgram -gen-lto`.
    ///
    /// Pre-Blackwell targets use the legacy LLVM 7 dialect; Blackwell and
    /// newer targets use the modern opaque-pointer dialect. Architecture is
    /// controlled by `target_arch` or `device_arch_hint` (normally populated
    /// by `cargo oxide`). When an ordinary build switches to NVVM IR after
    /// detecting libdevice, the pipeline may instead select the module's
    /// feature-based target floor.
    pub emit_nvvm_ir: bool,
    /// Explicit CUDA target used to choose NVVM IR syntax.
    ///
    /// Normally set by `cargo oxide --arch` or `CUDA_OXIDE_TARGET`.
    pub target_arch: Option<String>,
    /// Detected architecture of the local GPU (`CUDA_OXIDE_DEVICE_ARCH`).
    ///
    /// Used only when no explicit target is provided.
    pub device_arch_hint: Option<String>,
    /// Device debug metadata tier.
    pub debug_kind: DebugKind,
    /// Whether ordinary floating-point multiply/add or multiply/subtract
    /// expressions may contract into fused operations.
    ///
    /// Explicit fused operations, such as `f32::mul_add`, are unaffected.
    pub allow_fma_contraction: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            output_dir: std::env::current_dir().unwrap_or_else(|_| ".".into()),
            output_name: "kernel".to_string(),
            verbose: true,
            show_mir_dialect: false,
            show_llvm_dialect: false,
            emit_nvvm_ir: false,
            target_arch: None,
            device_arch_hint: None,
            debug_kind: DebugKind::Off,
            allow_fma_contraction: true,
        }
    }
}

/// Runs the full compilation pipeline on collected functions.
///
/// # Pipeline Steps
///
/// 1. Register the `dialect-mir`, `dialect-nvvm`, and LLVM dialects
/// 2. Translate each function's MIR body into `dialect-mir`
/// 3. Verify the `dialect-mir` module
/// 4. Unless full variable-debug mode is enabled, run `mem2reg` to promote slot
///    allocas back into SSA
/// 5. In the same modes, unroll annotated loops and clean up changed functions
/// 6. Lower `dialect-mir` → LLVM dialect (via `mir-lower`)
/// 7. Verify the LLVM dialect module
/// 8. Export the LLVM dialect to a `.ll` file (including device extern declarations)
/// 9. Invoke `llc` to generate PTX (or emit LTOIR/NVVM IR when requested)
///
/// # Target Selection
///
/// Automatically detects GPU features (WGMMA, TMA, tcgen05) and selects
/// an appropriate SM target. Can be overridden via `CUDA_OXIDE_TARGET`.
///
/// # Device Externs
///
/// External device function declarations (from `#[device] extern "C" { ... }`)
/// are emitted as LLVM `declare` statements. These are resolved at link time
/// by nvJitLink when linking with external LTOIR (e.g., CCCL libraries).
///
/// # Errors
///
/// Returns [`PipelineError`] with details on which step failed.
pub fn run_pipeline(
    functions: &[CollectedFunction],
    device_externs: &[DeviceExternDecl],
    config: &PipelineConfig,
) -> Result<CompilationResult, PipelineError> {
    prepare_output_dir(&config.output_dir)?;

    let mut ctx = Context::new();

    // Step 1: Register dialects
    crate::translator::register_dialects(&mut ctx);

    // Step 2: Create module
    let module_name: pliron::identifier::Identifier = config
        .output_name
        .clone()
        .try_into()
        .unwrap_or_else(|_| "kernel".try_into().unwrap());
    let module = pliron::builtin::ops::ModuleOp::new(&mut ctx, module_name);
    let module_op_ptr = module.get_operation();

    let mut legaliser = Legaliser::default();

    // Step 3: Translate all functions
    for func in functions {
        if config.verbose {
            eprintln!(
                "Translating {}: {}",
                if func.is_kernel {
                    "kernel"
                } else {
                    "device fn"
                },
                func.export_name
            );
        }

        let body = func
            .instance
            .body()
            .ok_or_else(|| PipelineError::NoBody(func.export_name.clone()))?;

        let func_op_ptr = crate::translator::body::translate_body(
            &mut ctx,
            &body,
            &func.instance,
            func.is_kernel,
            func.is_inline_always,
            Some(&func.export_name),
            &mut legaliser,
            config.debug_kind,
            func.debug_source_scopes.as_ref(),
        )
        .map_err(|e| {
            // Use .disp(&ctx) for rich error formatting with location and backtrace
            PipelineError::Translation(format!("{}: {}", func.export_name, e.disp(&ctx)))
        })?;

        // Dump the per-function IR BEFORE verification so users can see
        // what the translator produced even when verification fails. If we
        // verified first and bailed, `--show-mir-dialect` / `CUDA_OXIDE_DUMP_MIR`
        // would silently print nothing for the offending function.
        if config.show_mir_dialect {
            eprintln!(
                "\n=== dialect-mir func: {} (pre-verify) ===",
                func.export_name
            );
            eprintln!("{}", func_op_ptr.deref(&ctx).disp(&ctx));
        }

        verify_operation(&ctx, func_op_ptr, &func.export_name)?;

        // Append to module
        append_to_module(&ctx, module_op_ptr, func_op_ptr);
    }

    let ll_path = config.output_dir.join(format!("{}.ll", config.output_name));
    let ptx_path = config
        .output_dir
        .join(format!("{}.ptx", config.output_name));
    let stale_artifacts = stale_compilation_artifact_paths(&config.output_dir, &config.output_name);

    // Environment-derived compatibility options are read once at the rustc
    // frontend boundary. Explicit pipeline configuration retains precedence.
    let mut backend_options = BackendOptions::from_env();
    if config.target_arch.is_some() {
        backend_options.target_arch = config.target_arch.clone();
    }
    if config.device_arch_hint.is_some() {
        backend_options.device_arch_hint = config.device_arch_hint.clone();
    }
    backend_options.verbose = backend_options.verbose || config.verbose;
    backend_options.no_fma = !config.allow_fma_contraction;

    let request = ModulePipelineRequest::for_rust_pipeline(
        device_externs,
        config.emit_nvvm_ir,
        &backend_options,
        config.debug_kind,
        OutputFiles {
            llvm_ir: &ll_path,
            ptx: &ptx_path,
            stale_before_export: &stale_artifacts,
        },
        PipelineTrace {
            verbose: config.verbose,
            dump_mir: config.show_mir_dialect,
            dump_llvm: config.show_llvm_dialect,
            sink: Some(stderr_pipeline_trace),
        },
    );
    let generated = compile_translated_module(&mut ctx, module_op_ptr, &request)?;

    match generated.artifact_kind {
        ModuleArtifactKind::NvvmIr => {
            write_nvvm_compile_options_sidecar(
                &config.output_dir,
                &config.output_name,
                config.allow_fma_contraction,
            )?;
            // Publish the target last: its version marker is the completion record
            // that says the sibling options file is required.
            write_nvvm_target_sidecar(&config.output_dir, &config.output_name, &generated.target)?;
            Ok(CompilationResult {
                artifact_path: ll_path.clone(),
                artifact_kind: CompilationArtifactKind::NvvmIr,
                ll_path,
                ptx_path,
                target: generated.target,
                allow_fma_contraction: config.allow_fma_contraction,
            })
        }
        ModuleArtifactKind::Ptx => Ok(CompilationResult {
            artifact_path: ptx_path.clone(),
            artifact_kind: CompilationArtifactKind::Ptx,
            ll_path,
            ptx_path,
            target: generated.target,
            allow_fma_contraction: config.allow_fma_contraction,
        }),
    }
}

/// Ensures the configured output directory exists before any emission step.
///
/// The pipeline writes every generated artifact under `PipelineConfig::output_dir`.
/// Creating the directory at the pipeline boundary lets callers provide fresh
/// sidecar paths without separately seeding them first.
fn prepare_output_dir(output_dir: &Path) -> Result<(), PipelineError> {
    std::fs::create_dir_all(output_dir).map_err(|e| {
        PipelineError::Export(format!(
            "failed to create output directory {}: {}",
            output_dir.display(),
            e
        ))
    })
}

/// Records the resolved NVVM target alongside the emitted `.ll`.
///
/// The `.target` sidecar carries the completion marker that tells the consumer
/// the sibling `.options` file is present and required. These sidecars are a
/// host artifact concern (`oxide-artifacts`), so they stay in `mir-importer`
/// rather than the rustc-free `cuda-oxide-codegen` backend.
fn write_nvvm_target_sidecar(
    output_dir: &Path,
    output_name: &str,
    target: &str,
) -> Result<(), PipelineError> {
    let path = output_dir.join(format!("{output_name}.target"));
    std::fs::write(
        &path,
        format!(
            "{target}\n{}\n",
            oxide_artifacts::COMPILE_OPTIONS_TARGET_MARKER
        ),
    )
    .map_err(|error| {
        PipelineError::Export(format!(
            "failed to record NVVM target in {}: {error}",
            path.display()
        ))
    })
}

/// Records the compile-wide options (currently the fma-contraction policy) that
/// downstream LTOIR builds must preserve, next to the emitted `.ll`.
fn write_nvvm_compile_options_sidecar(
    output_dir: &Path,
    output_name: &str,
    allow_fma_contraction: bool,
) -> Result<(), PipelineError> {
    let path = output_dir.join(format!("{output_name}.options"));
    let options =
        oxide_artifacts::ArtifactCompileOptions::new().with_fma_contraction(allow_fma_contraction);
    std::fs::write(&path, options.sidecar_text()).map_err(|error| {
        PipelineError::Export(format!(
            "failed to record NVVM compile options in {}: {error}",
            path.display()
        ))
    })
}

fn stale_compilation_artifact_paths(
    output_dir: &Path,
    output_name: &str,
) -> Vec<std::path::PathBuf> {
    [
        "ll",
        "ptx",
        "target",
        "options",
        "ltoir",
        "cubin",
        "cubin.target",
    ]
    .into_iter()
    .map(|suffix| output_dir.join(format!("{output_name}.{suffix}")))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_pipeline_config_default_values() {
        let config = PipelineConfig::default();

        assert_eq!(config.output_name, "kernel");
        assert!(config.verbose);
        assert!(!config.show_mir_dialect);
        assert!(!config.show_llvm_dialect);
        assert!(!config.emit_nvvm_ir);
        assert_eq!(config.target_arch, None);
        assert_eq!(config.device_arch_hint, None);
        assert_eq!(config.debug_kind, DebugKind::Off);
    }

    #[test]
    fn stale_artifact_invalidation_removes_every_competing_output() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_stale_artifacts_{}_{}",
            std::process::id(),
            unique
        ));
        fs::create_dir_all(&root).unwrap();
        for suffix in [
            "ll",
            "ptx",
            "target",
            "options",
            "ltoir",
            "cubin",
            "cubin.target",
        ] {
            fs::write(root.join(format!("kernel.{suffix}")), b"stale").unwrap();
        }
        let cached_cubin =
            root.join(".oxide-artifacts/ltoir-cubin-cache/v1/entries/key/image.cubin");
        fs::create_dir_all(cached_cubin.parent().unwrap()).unwrap();
        fs::write(&cached_cubin, b"persistent cache entry").unwrap();

        let config = PipelineConfig {
            output_dir: root.clone(),
            output_name: "kernel".to_string(),
            verbose: false,
            show_mir_dialect: false,
            show_llvm_dialect: false,
            emit_nvvm_ir: true,
            target_arch: Some("sm_86".to_string()),
            device_arch_hint: None,
            debug_kind: DebugKind::Off,
            allow_fma_contraction: true,
        };
        let result = run_pipeline(&[], &[], &config).expect("pipeline run");

        assert_eq!(result.artifact_kind, CompilationArtifactKind::NvvmIr);
        assert_ne!(fs::read(&result.ll_path).unwrap(), b"stale");
        for suffix in ["ptx", "ltoir", "cubin", "cubin.target"] {
            assert!(!root.join(format!("kernel.{suffix}")).exists(), "{suffix}");
        }
        assert_ne!(fs::read(root.join("kernel.target")).unwrap(), b"stale");
        assert_ne!(fs::read(root.join("kernel.options")).unwrap(), b"stale");
        assert_eq!(
            fs::read(&cached_cubin).unwrap(),
            b"persistent cache entry",
            "content-addressed cache entries must survive compiler cleanup"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn run_pipeline_creates_missing_output_dir_before_export() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_mir_importer_output_dir_{}_{}",
            std::process::id(),
            unique
        ));
        let output_dir = root.join("fresh").join("nested");
        fs::remove_dir_all(&root).ok();
        assert!(!output_dir.exists());

        let config = PipelineConfig {
            output_dir: output_dir.clone(),
            output_name: "empty".to_string(),
            verbose: false,
            show_mir_dialect: false,
            show_llvm_dialect: false,
            emit_nvvm_ir: true,
            target_arch: Some("sm_86".to_string()),
            device_arch_hint: None,
            debug_kind: DebugKind::Off,
            allow_fma_contraction: true,
        };

        let result = run_pipeline(&[], &[], &config).expect("pipeline run");

        assert!(output_dir.is_dir());
        assert!(result.ll_path.is_file());
        assert_eq!(result.artifact_path, result.ll_path);
        assert_eq!(result.artifact_kind, CompilationArtifactKind::NvvmIr);
        assert_eq!(result.target, "sm_86");
        assert_eq!(
            fs::read_to_string(output_dir.join("empty.target")).unwrap(),
            format!(
                "sm_86\n{}\n",
                oxide_artifacts::COMPILE_OPTIONS_TARGET_MARKER
            )
        );
        assert_eq!(
            fs::read_to_string(output_dir.join("empty.options")).unwrap(),
            oxide_artifacts::ArtifactCompileOptions::new()
                .with_fma_contraction(true)
                .sidecar_text()
        );

        fs::remove_dir_all(&root).expect("clean up temp output dir");
    }

    #[test]
    fn structured_device_extern_survives_pre_lowering_insertion() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_mir_importer_extern_{}_{}",
            std::process::id(),
            unique
        ));
        let config = PipelineConfig {
            output_dir: root.clone(),
            output_name: "extern_only".to_string(),
            verbose: false,
            show_mir_dialect: false,
            show_llvm_dialect: false,
            emit_nvvm_ir: true,
            target_arch: Some("sm_86".to_string()),
            device_arch_hint: None,
            debug_kind: DebugKind::Off,
            allow_fma_contraction: true,
        };
        let externs = [DeviceExternDecl {
            export_name: "consume_float".to_string(),
            param_types: vec![DeviceExternType::pointer_to(DeviceExternType::Float32, 0)],
            return_type: DeviceExternType::Void,
            attrs: DeviceExternAttrs::default(),
        }];

        let result = run_pipeline(&[], &externs, &config).expect("pipeline run");
        let ir = fs::read_to_string(result.ll_path).expect("read exported IR");
        assert!(
            ir.contains("declare void @consume_float(float*)"),
            "structured pointee must survive through export:\n{ir}"
        );
        assert!(
            !ir.split(|c: char| !c.is_ascii_alphanumeric())
                .any(|token| token == "ptr"),
            "legacy device-extern output must not contain opaque pointers:\n{ir}"
        );

        fs::remove_dir_all(&root).expect("clean up temp output dir");
    }
}
