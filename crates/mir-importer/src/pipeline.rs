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

pub use cuda_oxide_codegen::{
    DeviceExternAttrs, DeviceExternDecl, PipelineError, PtxConfig, compile_to_ptx,
};
pub use llvm_export::export::DeviceExternType;
use llvm_export::export::{DebugKind, NvvmIrDialect};
use cuda_oxide_codegen::export::{
    export_llvm_ir, module_uses_libdevice, render_llvm_ir, resolve_nvvm_target,
    validate_nvvm_debug_support,
};
use cuda_oxide_codegen::lower::{add_device_extern_declarations, append_to_module, lower_to_llvm};
use cuda_oxide_codegen::ptx::generate_ptx;
use cuda_oxide_codegen::target::detect_features_in_llvm_text;
use cuda_oxide_codegen::verify::verify_operation;
use pliron::context::Context;
use pliron::identifier::Legaliser;
use pliron::op::Op;
use pliron::printable::Printable;
use rustc_public::mir::mono::Instance;
use std::path::Path;

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

    // Step 4: Verify module. Dump BEFORE verify so module-level verification
    // failures still surface the consolidated IR to the user.
    if config.show_mir_dialect {
        eprintln!("\n=== dialect-mir module (pre-verify) ===");
        eprintln!("{}", module_op_ptr.deref(&ctx).disp(&ctx));
    }
    if config.verbose {
        eprintln!("\n=== Verifying dialect-mir module ===");
    }
    verify_operation(&ctx, module_op_ptr, "module")?;
    if config.verbose {
        eprintln!("dialect-mir verification successful ✓");
    }

    // Step 4.5: Run mem2reg (promote `mir.alloca` + `mir.load`/`mir.store`
    // chains back to SSA values).
    //
    // Full-debug is a `-G`-style build: we keep every source local in its stack
    // slot so cuda-gdb can read it from a stable memory location for the whole
    // scope (via `llvm.dbg.declare`). Promoting locals to SSA would narrow each
    // variable's inspectable range to its register's liveness, which is why an
    // optimized `dbg.value` build shows `<optimized out>` for in-scope locals.
    // We therefore skip mem2reg whenever variable info is requested. The
    // promotion-aware `mir.dbg_value` salvage (see `dialect-mir::ops::debug`)
    // remains the mechanism for any future optimized-debug tier that *does*
    // promote.
    if config.debug_kind.variables_enabled() {
        if config.verbose {
            eprintln!("\n=== Skipping mem2reg (full debug keeps locals in memory) ===");
        }
    } else {
        if config.verbose {
            eprintln!("\n=== Running mem2reg ===");
        }
        // pliron's pass infra now threads an AnalysisManager through mem2reg
        // (caches dominator trees etc.); we run it standalone, so a fresh empty
        // manager suffices. The returned IRStatus (Changed/Unchanged) is discarded.
        let mut analyses = pliron::pass_manager::AnalysisManager::default();
        pliron::opts::mem2reg::mem2reg(module_op_ptr, &mut ctx, &mut analyses).map_err(|e| {
            PipelineError::Verification {
                name: "mem2reg".to_string(),
                message: e.disp(&ctx).to_string(),
                operation: None,
            }
        })?;
        if config.verbose {
            eprintln!("mem2reg successful ✓");
        }
        if config.show_mir_dialect {
            eprintln!("\n=== dialect-mir module (after mem2reg) ===");
            eprintln!("{}", module_op_ptr.deref(&ctx).disp(&ctx));
        }
        verify_operation(&ctx, module_op_ptr, "module post-mem2reg")?;

        // Step 4.6: annotation-driven loop unrolling (#[unroll] / #[unroll(N)]).
        // Runs on the SSA form mem2reg just produced; a no-op unless a loop
        // contains a `mir.unroll_hint` operation. The pass receives mem2reg's
        // AnalysisManager for the standard pass shape, but recomputes dominance
        // after each CFG rewrite.
        if config.verbose {
            eprintln!("\n=== Running loop-unroll ===");
        }
        mir_transforms::unroll::unroll_annotated_loops(module_op_ptr, &mut ctx, &mut analyses)
            .map_err(|e| PipelineError::Verification {
                name: "loop-unroll".to_string(),
                message: e.disp(&ctx).to_string(),
                operation: None,
            })?;
        verify_operation(&ctx, module_op_ptr, "module post-unroll")?;
        // Constant folding (sccp -> simplify_cfg -> dce) runs inside the unroll
        // pass, scoped to functions it actually unrolled; see
        // `mir_transforms::unroll`. Non-unrolled kernels are left for `opt`/NVVM.
    }

    // Step 4.9: Add structured device-extern declarations before call
    // lowering. The call converter consults these declarations to preserve
    // pointer address spaces and insert an explicit addrspacecast when the
    // caller and external ABI differ. Adding declarations only after lowering
    // is too late: every unknown pointer argument has already fallen back to
    // generic addrspace(0) by then.
    if !device_externs.is_empty() {
        if config.verbose {
            eprintln!(
                "\n=== Adding {} device extern declarations ===",
                device_externs.len()
            );
        }
        add_device_extern_declarations(&mut ctx, module_op_ptr, device_externs)?;
    }

    // Step 5: Lower dialect-mir → LLVM dialect.
    if config.verbose {
        eprintln!("\n=== Lowering dialect-mir → LLVM dialect ===");
    }
    lower_to_llvm(&mut ctx, module_op_ptr, config.allow_fma_contraction)?;

    // Detect CUDA libdevice usage.
    //
    // Lowering the rustc float-math intrinsics emits `__nv_*` libdevice
    // calls (e.g. `__nv_sinf`, `__nv_pow`). `llc` cannot resolve those — they
    // need libNVVM + nvJitLink + `libdevice.10.bc`, which the example owns
    // (see `examples/device_ffi_test/tools/`). When we see them we:
    //   1. Force NVVM IR mode so the `.ll` is suitable for libNVVM input.
    //   2. Skip the `llc → .ptx` step, because the resulting PTX would have
    //      unresolved `__nv_*` extern calls and `cuModuleLoad` would reject
    //      it.
    // The example is then expected to feed the `.ll` through the LTOIR
    // pipeline (compile_ltoir + link_ltoir) and load the resulting cubin.
    let needs_libdevice = module_uses_libdevice(&ctx, module_op_ptr);
    let emit_nvvm_ir = config.emit_nvvm_ir || needs_libdevice;
    if needs_libdevice && !config.emit_nvvm_ir && config.verbose {
        eprintln!(
            "\n=== Detected CUDA libdevice (`__nv_*`) calls; \
             auto-emitting NVVM IR (skip llc) ==="
        );
    }

    // An ordinary zero-flag build may discover only now that libdevice makes
    // NVVM IR necessary. Preserve the normal target policy in that case:
    // explicit target, then a compatible local-GPU hint, then the compiler's
    // feature-based target. Feature detection uses the
    // same LLVM text that the ordinary PTX path would inspect, but keeps this
    // preview in memory because the final pointer dialect is not known yet.
    let automatic_features =
        if needs_libdevice && !config.emit_nvvm_ir && config.target_arch.is_none() {
            let preview = render_llvm_ir(
                &ctx,
                module_op_ptr,
                device_externs,
                false,
                None,
                config.debug_kind,
            )?;
            Some(detect_features_in_llvm_text(&preview))
        } else {
            None
        };

    // Pre-Blackwell and Blackwell GPUs use different NVVM IR pointer syntax.
    // Resolve one concrete target before export and record it with the
    // artifact.
    let (nvvm_target, nvvm_dialect) = if emit_nvvm_ir {
        let target = resolve_nvvm_target(
            config.target_arch.as_deref(),
            config.device_arch_hint.as_deref(),
            automatic_features,
        )?;
        let dialect = if target.uses_legacy_llvm() {
            NvvmIrDialect::LegacyLlvm7
        } else {
            NvvmIrDialect::Modern
        };
        validate_nvvm_debug_support(&target, dialect, config.debug_kind)?;
        (Some(target), Some(dialect))
    } else {
        (None, None)
    };

    // Step 5.5: Convert LLVM operations to the forms supported by the selected
    // NVVM dialect, then verify the changed module before text export.
    if let Some(dialect) = nvvm_dialect {
        if config.verbose {
            if dialect == NvvmIrDialect::LegacyLlvm7 {
                eprintln!("\n=== Legalizing LLVM dialect for legacy NVVM ===");
            } else {
                eprintln!("\n=== Legalizing NVVM bit-intrinsic widths ===");
            }
        }
        nvvm_transforms::legalize_for_nvvm(&mut ctx, module_op_ptr, dialect)
            .map_err(|error| PipelineError::Lowering(error.disp(&ctx).to_string()))?;
    }

    // Step 6: Verify the final LLVM dialect module. Dump BEFORE verify so
    // verification failures still surface the exact post-legalization IR.
    if config.show_llvm_dialect {
        eprintln!("\n=== LLVM dialect (pre-verify) ===");
        eprintln!("{}", module_op_ptr.deref(&ctx).disp(&ctx));
    }
    if config.verbose {
        eprintln!("=== Verifying LLVM dialect module ===");
    }
    verify_operation(&ctx, module_op_ptr, "llvm module")?;
    if config.verbose {
        eprintln!("LLVM dialect verification successful ✓");
    }

    // Step 7: Export to LLVM IR
    if config.verbose {
        let mode = if emit_nvvm_ir { "NVVM IR" } else { "PTX" };
        eprintln!("\n=== Exporting to LLVM IR ({} mode) ===", mode);
    }
    let ll_path = config.output_dir.join(format!("{}.ll", config.output_name));
    // Remove artifacts from earlier builds so changing output mode cannot
    // leave older PTX, LTOIR, or cubin selected by the loader.
    clear_stale_compilation_artifacts(&config.output_dir, &config.output_name)?;
    let _llvm_ir = export_llvm_ir(
        &ctx,
        module_op_ptr,
        device_externs,
        &ll_path,
        emit_nvvm_ir,
        nvvm_dialect,
        config.debug_kind,
    )?;
    if config.verbose {
        eprintln!("LLVM IR written to {}", ll_path.display());
    }

    // Step 8: Generate PTX or stop at NVVM IR for libNVVM-owned paths.
    if emit_nvvm_ir {
        // Skip llc. Return a would-be ptx_path so callers see a stable shape;
        // the file does not exist and the consumer must build its own cubin
        // from `ll_path` via libNVVM + nvJitLink.
        let ptx_path = config
            .output_dir
            .join(format!("{}.ptx", config.output_name));
        if config.verbose {
            let reason = if needs_libdevice {
                "libdevice present"
            } else {
                "NVVM IR requested"
            };
            eprintln!("\n=== Skipping llc ({reason}); consumer owns libNVVM/nvJitLink build ===");
        }
        let target = nvvm_target
            .as_ref()
            .expect("NVVM target was resolved before export")
            .sm();
        write_nvvm_compile_options_sidecar(
            &config.output_dir,
            &config.output_name,
            config.allow_fma_contraction,
        )?;
        // Publish the target last: its version marker is the completion record
        // that says the sibling options file is required.
        write_nvvm_target_sidecar(&config.output_dir, &config.output_name, &target)?;
        Ok(CompilationResult {
            artifact_path: ll_path.clone(),
            artifact_kind: CompilationArtifactKind::NvvmIr,
            ll_path,
            ptx_path,
            target,
            allow_fma_contraction: config.allow_fma_contraction,
        })
    } else {
        if config.verbose {
            eprintln!("\n=== Generating PTX ===");
        }
        let ptx_path = config
            .output_dir
            .join(format!("{}.ptx", config.output_name));

        // Build the backend options at this process's own boundary: start
        // from the CUDA_OXIDE_* env vars (the same ones rustc-codegen-cuda
        // already read to populate `config.target_arch`/`device_arch_hint`
        // below, so this is not a second, independent env read), then let an
        // explicit `config` value win when present. This reproduces today's
        // effective precedence exactly, because `config.target_arch` and
        // `config.device_arch_hint` were themselves populated from
        // CUDA_OXIDE_TARGET/CUDA_OXIDE_DEVICE_ARCH by the sole caller
        // (`device_codegen.rs`) before `generate_ptx` used to re-read the
        // same variables directly.
        let mut backend_opts = cuda_oxide_codegen::options::BackendOptions::from_env();
        if config.target_arch.is_some() {
            backend_opts.target_arch = config.target_arch.clone();
        }
        if config.device_arch_hint.is_some() {
            backend_opts.device_arch_hint = config.device_arch_hint.clone();
        }
        backend_opts.verbose = backend_opts.verbose || config.verbose;
        // #326: the fma-contraction policy is a compile-wide decision threaded
        // from cargo-oxide's `--no-fmad`. It drives both the IR-level contract
        // flag (via `lower_to_llvm` above) and the llc `-fp-contract` gate
        // inside `generate_ptx`, so the two stages cannot disagree.
        backend_opts.no_fma = !config.allow_fma_contraction;

        let target = generate_ptx(&ll_path, &ptx_path, config.debug_kind, &backend_opts)?;
        if config.verbose {
            eprintln!(
                "✓ PTX written to {} (target: {})",
                ptx_path.display(),
                target
            );
        }

        Ok(CompilationResult {
            artifact_path: ptx_path.clone(),
            artifact_kind: CompilationArtifactKind::Ptx,
            ll_path,
            ptx_path,
            target,
            allow_fma_contraction: config.allow_fma_contraction,
        })
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
        format!("{target}\n{}\n", oxide_artifacts::COMPILE_OPTIONS_TARGET_MARKER),
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

fn clear_stale_compilation_artifacts(
    output_dir: &Path,
    output_name: &str,
) -> Result<(), PipelineError> {
    for suffix in ["ll", "ptx", "target", "ltoir", "cubin", "cubin.target"] {
        let path = output_dir.join(format!("{output_name}.{suffix}"));
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(PipelineError::Export(format!(
                    "failed to invalidate stale CUDA artifact {}: {error}",
                    path.display()
                )));
            }
        }
    }
    Ok(())
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
        for suffix in ["ll", "ptx", "target", "ltoir", "cubin", "cubin.target"] {
            fs::write(root.join(format!("kernel.{suffix}")), b"stale").unwrap();
        }
        let cached_cubin =
            root.join(".oxide-artifacts/ltoir-cubin-cache/v1/entries/key/image.cubin");
        fs::create_dir_all(cached_cubin.parent().unwrap()).unwrap();
        fs::write(&cached_cubin, b"persistent cache entry").unwrap();

        clear_stale_compilation_artifacts(&root, "kernel").unwrap();

        for suffix in ["ll", "ptx", "target", "ltoir", "cubin", "cubin.target"] {
            assert!(!root.join(format!("kernel.{suffix}")).exists(), "{suffix}");
        }
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
            format!("sm_86\n{}\n", oxide_artifacts::COMPILE_OPTIONS_TARGET_MARKER)
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
