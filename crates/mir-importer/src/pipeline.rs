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
//! | tcgen05/TMEM, CTA-group TMA   | sm_100a | Blackwell datacenter |
//! | CLC / newer generic TMA       | sm_100  | Blackwell+           |
//! | CLC multicast / accel. TMA    | sm_100a | Blackwell family     |
//! | PTX 8.6 matrix shapes/types   | sm_100a | Blackwell family     |
//! | TMA multicast                 | sm_100a | sm_90+ (a/f advised) |
//! | WGMMA                         | sm_90a  | Hopper only          |
//! | `stmatrix.m8n8.b16`          | sm_90   | PTX 7.8+             |
//! | TMA/mbarrier                  | sm_100  | Hopper+ compatible   |
//! | bf16x2 add/sub/mul            | sm_90   | Hopper+ compatible   |
//! | other bf16x2 ALU              | sm_80   | Ampere+ compatible   |
//! | BF16 `mma.m16n8k16`           | sm_80   | PTX 7.0+             |
//! | `cp.async` (non-bulk)         | sm_80   | Ampere+              |
//! | `movmatrix.m8n8.b16`          | sm_75   | PTX 7.8+             |
//! | `ldmatrix.m8n8.b16`           | sm_75   | PTX 6.5+             |
//! | Basic CUDA                    | sm_80   | Ampere+ (max compat) |
//!
//! Override with `CUDA_OXIDE_TARGET=<target>` environment variable.

use libnvvm_sys::CudaArch;
pub use llvm_export::export::DeviceExternType;
use llvm_export::export::{DebugKind, ExportBackendConfig, NvvmIrDialect};
use pliron::common_traits::Verify;
use rustc_public::mir::mono::Instance;

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

/// An external device function declaration (for FFI with external LTOIR).
///
/// Unlike `CollectedFunction`, these have no MIR body - they're just declarations
/// that will be emitted as LLVM `declare` statements for nvJitLink to resolve
/// when linking with external LTOIR (e.g., CCCL libraries).
#[derive(Debug, Clone)]
pub struct DeviceExternDecl {
    /// The export name (the original function name, e.g., "cub_block_reduce_sum").
    pub export_name: String,

    /// Structured LLVM ABI parameter types. Pointer pointees are retained even
    /// though the lowered pliron LLVM module itself uses opaque pointers.
    pub param_types: Vec<DeviceExternType>,

    /// Structured LLVM ABI return type.
    pub return_type: DeviceExternType,

    /// NVVM attributes for this function.
    pub attrs: DeviceExternAttrs,
}

/// NVVM attributes for device extern declarations.
///
/// NOTE: These attributes are currently **not emitted** to the LLVM IR output.
/// When linking LTOIR via nvJitLink, the external library's LTOIR already contains
/// proper attributes (convergent, nounwind, memory, etc.) on the function DEFINITIONS.
/// nvJitLink uses the definition's attributes during LTO, making attributes on our
/// declarations redundant.
///
/// This struct is retained for the pipeline API but values are not used in code generation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct DeviceExternAttrs {
    /// Function is convergent (all threads must execute together).
    /// NOTE: Not currently emitted to LLVM IR.
    pub is_convergent: bool,

    /// Function is pure (no side effects, result depends only on inputs).
    /// NOTE: Not currently emitted to LLVM IR.
    pub is_pure: bool,

    /// Function is read-only (only reads memory, doesn't write).
    /// NOTE: Not currently emitted to LLVM IR.
    pub is_readonly: bool,
}

// Implement AsDeviceExtern trait for llvm-export integration
impl llvm_export::export::AsDeviceExtern for DeviceExternDecl {
    fn as_device_extern(&self) -> llvm_export::export::DeviceExternDecl {
        llvm_export::export::DeviceExternDecl {
            export_name: self.export_name.clone(),
            param_types: self.param_types.clone(),
            return_type: self.return_type.clone(),
            attrs: llvm_export::export::DeviceExternAttrs {
                is_convergent: self.attrs.is_convergent,
                is_pure: self.attrs.is_pure,
                is_readonly: self.attrs.is_readonly,
            },
        }
    }
}
use crate::llvm_tools::LlvmToolchain;
use pliron::builtin::op_interfaces::{CallOpCallable, CallOpInterface, SymbolOpInterface};
use pliron::context::{Context, Ptr};
use pliron::identifier::Legaliser;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;
use std::path::{Path, PathBuf};

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
    lower_to_llvm(&mut ctx, module_op_ptr)?;

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

    // NVVM IR export must validate its explicit target against the same
    // module feature floor as ordinary PTX generation. An ordinary zero-flag
    // build can also discover only now that libdevice makes NVVM IR necessary.
    // Render one in-memory preview before choosing the final pointer dialect;
    // the preview is only inspected and is not written as an artifact.
    let automatic_features = if emit_nvvm_ir {
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
        write_nvvm_target_sidecar(&config.output_dir, &config.output_name, &target)?;
        Ok(CompilationResult {
            artifact_path: ll_path.clone(),
            artifact_kind: CompilationArtifactKind::NvvmIr,
            ll_path,
            ptx_path,
            target,
        })
    } else {
        if config.verbose {
            eprintln!("\n=== Generating PTX ===");
        }
        let ptx_path = config
            .output_dir
            .join(format!("{}.ptx", config.output_name));
        let target = generate_ptx(&ll_path, &ptx_path, config.debug_kind)?;
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

/// Returns true when lowering emitted CUDA libdevice calls.
///
/// Float math intrinsics (sin, cos, exp, log, pow, …) lower to `__nv_*`
/// entry points from `libdevice.10.bc`. `llc` cannot resolve these; they
/// need libNVVM + nvJitLink + libdevice. When we see any `__nv_*` symbol
/// the example owns the LTOIR build (see `examples/device_ffi_test/tools/`).
fn module_uses_libdevice(ctx: &Context, module_op_ptr: Ptr<Operation>) -> bool {
    op_uses_libdevice(ctx, module_op_ptr)
}

/// Recursively scan for declared or called CUDA libdevice functions.
fn op_uses_libdevice(ctx: &Context, op_ptr: Ptr<Operation>) -> bool {
    if let Some(func) = Operation::get_op::<llvm_export::ops::FuncOp>(op_ptr, ctx)
        && func.get_symbol_name(ctx).starts_with("__nv_")
    {
        return true;
    }

    if let Some(call) = Operation::get_op::<llvm_export::ops::CallOp>(op_ptr, ctx)
        && let CallOpCallable::Direct(callee) = call.callee(ctx)
        && callee.to_string().starts_with("__nv_")
    {
        return true;
    }

    let op_ref = op_ptr.deref(ctx);
    for region in op_ref.regions() {
        let region_ref = region.deref(ctx);
        for block in region_ref.iter(ctx) {
            let block_ref = block.deref(ctx);
            for child_op in block_ref.iter(ctx) {
                if op_uses_libdevice(ctx, child_op) {
                    return true;
                }
            }
        }
    }

    false
}

/// Recursively verifies an operation and all nested operations.
///
/// On failure, attempts to find the innermost failing operation for better
/// error messages.
fn verify_operation(
    ctx: &Context,
    op_ptr: Ptr<Operation>,
    name: &str,
) -> Result<(), PipelineError> {
    if let Err(e) = op_ptr.deref(ctx).verify(ctx) {
        // Try to find specific failing operation
        if let Some((err_op, err_msg)) = find_inner_verification_error(ctx, op_ptr) {
            return Err(PipelineError::Verification {
                name: name.to_string(),
                message: err_msg,
                operation: Some(err_op.deref(ctx).disp(ctx).to_string()),
            });
        }

        // Use .disp(ctx) to get full error with location and backtrace
        return Err(PipelineError::Verification {
            name: name.to_string(),
            message: e.disp(ctx).to_string(),
            operation: None,
        });
    }
    Ok(())
}

/// Inserts a function operation into the module's block.
fn append_to_module(ctx: &Context, module_op_ptr: Ptr<Operation>, func_op_ptr: Ptr<Operation>) {
    let region = module_op_ptr.deref(ctx).get_region(0).deref(ctx);
    let block = region.iter(ctx).next().expect("Module should have a block");
    func_op_ptr.insert_at_back(block, ctx);
}

/// Lowers `dialect-mir` operations to the LLVM dialect.
///
/// Runs `mir-lower`'s `DialectConversion`-based pass, which converts each
/// `dialect-mir`/`dialect-nvvm` op to its LLVM dialect equivalent. The LLVM
/// dialect auto-registers when the `Context` is created, so no explicit
/// registration is needed here.
fn lower_to_llvm(ctx: &mut Context, module_op_ptr: Ptr<Operation>) -> Result<(), PipelineError> {
    mir_lower::register(ctx);

    match mir_lower::lower_mir_to_llvm(ctx, module_op_ptr) {
        Ok(()) => Ok(()),
        // Format with `ctx` so the failing op's location/span survives.
        Err(e) => Err(PipelineError::Lowering(e.disp(ctx).to_string())),
    }
}

/// Adds device extern function declarations to the LLVM dialect module.
///
/// Creates LLVM dialect `FuncOp` declarations (without bodies) for each
/// device extern function. These declarations ensure that calls to extern
/// functions pass verification; the matching `declare` statements with
/// attributes are emitted during LLVM IR export.
///
/// This runs before MIR-to-LLVM call lowering so the call converter can read
/// exact parameter address spaces. It is still idempotent with respect to any
/// LLVM declaration already present in the mixed module; inserting a second
/// `FuncOp` for the same symbol would fail module verification.
fn add_device_extern_declarations(
    ctx: &mut Context,
    module_op_ptr: Ptr<Operation>,
    device_externs: &[DeviceExternDecl],
) -> Result<(), PipelineError> {
    use llvm_export::ops::FuncOp;
    use llvm_export::types::FuncType;
    use pliron::builtin::type_interfaces::FunctionTypeInterface;
    use pliron::identifier::Identifier;
    use std::collections::HashMap;

    // Get the module's block pointer first (this is a Ptr, not a Ref, so no borrow issues)
    let block = {
        let region = module_op_ptr.deref(ctx).get_region(0).deref(ctx);
        region.iter(ctx).next().expect("Module should have a block")
    };

    let declared_symbols: HashMap<_, _> = block
        .deref(ctx)
        .iter(ctx)
        .filter_map(|op| {
            Operation::get_op::<FuncOp>(op, ctx)
                .map(|f| (f.get_symbol_name(ctx).to_string(), f.get_type(ctx)))
        })
        .collect();

    for decl in device_externs {
        let param_types: Vec<_> = decl
            .param_types
            .iter()
            .map(|ty| device_extern_type_to_pliron(ctx, ty, false))
            .collect::<Result<_, _>>()?;
        let return_type = device_extern_type_to_pliron(ctx, &decl.return_type, true)?;

        // Create function type (result, args, is_variadic)
        let func_type = FuncType::get(ctx, return_type, param_types, false);

        if let Some(existing_type) = declared_symbols.get(&decl.export_name) {
            let existing_ref = existing_type.deref(ctx);
            let existing = &*existing_ref;
            let expected_ref = func_type.deref(ctx);
            let expected = &*expected_ref;
            if existing.result_type() != expected.result_type()
                || existing.arg_types() != expected.arg_types()
                || existing.is_var_arg() != expected.is_var_arg()
            {
                return Err(PipelineError::Export(format!(
                    "device extern `@{}` conflicts with the call-site declaration: expected `{}`, found `{}`",
                    decl.export_name,
                    expected_ref.disp(ctx),
                    existing_ref.disp(ctx),
                )));
            }
            continue;
        }

        // Use the original export name (NOT the prefixed name).
        // The MIR sees calls to `cuda_oxide_device_extern_<hash>_foo`, but
        // mir-lower/convert/ops/call.rs strips the reserved prefix via
        // `reserved_oxide_symbols::device_extern_base_name`, so the LLVM IR
        // emits `call @foo(...)`. For that to resolve, we declare `@foo` here.
        let func_ident: Identifier = decl.export_name.clone().try_into().map_err(|_| {
            PipelineError::Export(format!(
                "device-extern symbol `{}` cannot be represented by the LLVM dialect",
                decl.export_name
            ))
        })?;

        // Create function declaration (no body = declaration)
        let func_op = FuncOp::new(ctx, func_ident, func_type);

        // Insert at the front of the module (declarations come before definitions)
        func_op.get_operation().insert_at_front(block, ctx);
    }

    Ok(())
}

/// Convert the structured device-extern ABI type to the opaque-pointer pliron
/// LLVM type used for verification and call lowering.
fn device_extern_type_to_pliron(
    ctx: &mut Context,
    ty: &DeviceExternType,
    allow_void: bool,
) -> Result<pliron::r#type::TypeHandle, PipelineError> {
    use llvm_export::types::{ArrayType, HalfType, PointerType, VoidType};
    use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};

    Ok(match ty {
        DeviceExternType::Void if allow_void => VoidType::get(ctx).into(),
        DeviceExternType::Void => {
            return Err(PipelineError::Export(
                "device-extern parameters and aggregate elements cannot be `void`".to_string(),
            ));
        }
        DeviceExternType::Integer(bits) if *bits > 0 => {
            IntegerType::get(ctx, *bits, Signedness::Signless).into()
        }
        DeviceExternType::Integer(_) => {
            return Err(PipelineError::Export(
                "device-extern integer width must be non-zero".to_string(),
            ));
        }
        DeviceExternType::Float16 => HalfType::get(ctx).into(),
        DeviceExternType::Float32 => FP32Type::get(ctx).into(),
        DeviceExternType::Float64 => FP64Type::get(ctx).into(),
        DeviceExternType::Pointer {
            pointee,
            address_space,
        } => {
            if matches!(pointee.as_ref(), DeviceExternType::Void) {
                return Err(PipelineError::Export(
                    "device-extern pointer cannot have `void` as its pointee; use i8".to_string(),
                ));
            }
            PointerType::get(ctx, *address_space).into()
        }
        DeviceExternType::Array { element, len } => {
            let element = device_extern_type_to_pliron(ctx, element, false)?;
            ArrayType::get(ctx, element, *len).into()
        }
    })
}

fn resolve_nvvm_target(
    explicit_target: Option<&str>,
    device_arch_hint: Option<&str>,
    automatic_features: Option<DetectedFeatures>,
) -> Result<CudaArch, PipelineError> {
    let parse = |target: &str, source: &str| {
        target.parse::<CudaArch>().map_err(|error| {
            PipelineError::Export(format!(
                "cannot select an NVVM IR dialect from the {source} `{target}`: {error}"
            ))
        })
    };

    if let Some(target) = explicit_target {
        let parsed = parse(target, "explicit CUDA target")?;
        if let Some(features) = automatic_features {
            validate_target_features(&parsed, features).map_err(PipelineError::Export)?;
        }
        return Ok(parsed);
    }

    if let Some(features) = automatic_features {
        if let Some(target) = device_arch_hint {
            let parsed = parse(target, "detected GPU architecture")?;
            if arch_satisfies(&parsed.sm(), features) {
                return Ok(parsed);
            }
        }
        let target = select_target(features).map_err(PipelineError::Export)?;
        return parse(target, "feature-based compiler default");
    }

    if let Some(target) = device_arch_hint {
        return parse(target, "detected GPU architecture");
    }

    Err(PipelineError::Export(
        "NVVM IR requires a concrete CUDA target because pre-Blackwell and Blackwell+ \
         use different LLVM dialects; pass `cargo oxide ... --arch sm_XX` (or set \
         CUDA_OXIDE_TARGET)"
            .to_string(),
    ))
}

fn validate_nvvm_debug_support(
    target: &CudaArch,
    dialect: NvvmIrDialect,
    debug_kind: DebugKind,
) -> Result<(), PipelineError> {
    if dialect == NvvmIrDialect::LegacyLlvm7 && debug_kind != DebugKind::Off {
        return Err(PipelineError::Export(format!(
            "legacy LLVM 7 NVVM IR for {} does not yet support cuda-oxide debug metadata; \
             rebuild without device debug information",
            target.sm()
        )));
    }
    Ok(())
}

fn write_nvvm_target_sidecar(
    output_dir: &Path,
    output_name: &str,
    target: &str,
) -> Result<(), PipelineError> {
    let path = output_dir.join(format!("{output_name}.target"));
    std::fs::write(&path, format!("{target}\n")).map_err(|error| {
        PipelineError::Export(format!(
            "failed to record NVVM target in {}: {error}",
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

/// Exports an LLVM dialect module to textual LLVM IR (`.ll` file).
///
/// Backend configuration is selected based on flags:
/// - `emit_nvvm_ir`: Uses `NvvmExportConfig` for NVVM IR output
/// - Otherwise: Uses default `PtxExportConfig` for standard PTX generation
///
/// Device extern declarations are emitted before the main module content.
fn export_llvm_ir(
    ctx: &Context,
    module_op_ptr: Ptr<Operation>,
    device_externs: &[DeviceExternDecl],
    path: &Path,
    emit_nvvm_ir: bool,
    nvvm_dialect: Option<NvvmIrDialect>,
    debug_kind: DebugKind,
) -> Result<String, PipelineError> {
    let llvm_ir = render_llvm_ir(
        ctx,
        module_op_ptr,
        device_externs,
        emit_nvvm_ir,
        nvvm_dialect,
        debug_kind,
    )?;

    std::fs::write(path, &llvm_ir).map_err(|e| PipelineError::Export(e.to_string()))?;

    Ok(llvm_ir)
}

/// Render LLVM text without publishing an artifact.
///
/// Automatic libdevice mode uses this once before NVVM legalization to detect
/// the same target features as the normal PTX path. The final export still
/// happens exactly once, after the target-specific legalization pass.
fn render_llvm_ir(
    ctx: &Context,
    module_op_ptr: Ptr<Operation>,
    device_externs: &[DeviceExternDecl],
    emit_nvvm_ir: bool,
    nvvm_dialect: Option<NvvmIrDialect>,
    debug_kind: DebugKind,
) -> Result<String, PipelineError> {
    let module_op = Operation::get_op::<pliron::builtin::ops::ModuleOp>(module_op_ptr, ctx)
        .ok_or_else(|| PipelineError::Export("Not a module op".to_string()))?;

    let llvm_ir = if emit_nvvm_ir {
        let dialect = nvvm_dialect.ok_or_else(|| {
            PipelineError::Export("NVVM export reached without a selected IR dialect".to_string())
        })?;
        let config = PipelineExportConfig {
            inner: llvm_export::export::NvvmExportConfig::new(dialect),
            debug_kind,
        };
        llvm_export::export::export_module_with_externs(ctx, &module_op, device_externs, &config)
            .map_err(PipelineError::Export)?
    } else {
        let config = PipelineExportConfig {
            inner: llvm_export::export::PtxExportConfig,
            debug_kind,
        };
        llvm_export::export::export_module_with_externs(ctx, &module_op, device_externs, &config)
            .map_err(PipelineError::Export)?
    };

    Ok(llvm_ir)
}

struct PipelineExportConfig<C> {
    inner: C,
    debug_kind: DebugKind,
}

impl<C: ExportBackendConfig> ExportBackendConfig for PipelineExportConfig<C> {
    fn datalayout(&self) -> &str {
        self.inner.datalayout()
    }

    fn emit_llvm_used(&self) -> bool {
        self.inner.emit_llvm_used()
    }

    fn emit_nvvmir_version(&self) -> bool {
        self.inner.emit_nvvmir_version()
    }

    fn nvvmir_version(&self) -> [i32; 4] {
        self.inner.nvvmir_version()
    }

    fn emit_all_kernel_annotations(&self) -> bool {
        self.inner.emit_all_kernel_annotations()
    }

    fn emit_ptx_kernel_keyword(&self) -> bool {
        self.inner.emit_ptx_kernel_keyword()
    }

    fn nvvm_ir_dialect(&self) -> Option<NvvmIrDialect> {
        self.inner.nvvm_ir_dialect()
    }

    fn debug_kind(&self) -> DebugKind {
        self.debug_kind
    }
}

/// Checks for WGMMA instructions (Hopper sm_90a only, NOT forward-compatible).
///
/// WGMMA (Warpgroup Matrix Multiply-Accumulate) requires sm_90a specifically.
/// These are NOT forward-compatible - only work on H100/H200.
fn contains_wgmma_features(contents: &str) -> bool {
    contents.contains("wgmma.fence")
        || contents.contains("wgmma.commit_group")
        || contents.contains("wgmma.wait_group")
        || contents.contains("wgmma.mma_async")
}

/// Checks for Thread Block Cluster instructions (sm_90+).
///
/// Cluster features require Hopper (sm_90) or newer:
/// - Cluster special registers (%cluster_ctaid, %cluster_nctaid)
/// - Cluster synchronization (cluster.sync)
/// - Distributed shared memory (mapa.shared::cluster)
fn contains_cluster_features(contents: &str) -> bool {
    // Cluster special registers
    contents.contains("cluster_ctaid")
        || contents.contains("cluster_nctaid")
        || contents.contains("cluster_ctarank")
        || contents.contains("cluster_nctarank")
        || contents.contains("%clusterid")
        || contents.contains("%nclusterid")
        || contents.contains("%is_explicit_cluster")
        || contents.contains("!\"cluster_dim_x\"")
        || contents.contains("!\"cluster_dim_y\"")
        || contents.contains("!\"cluster_dim_z\"")
        // Cluster synchronization
        || contents.contains("cluster.sync")
        || contents.contains("barrier.cluster.")
        // Distributed shared memory
        || contents.contains("mapa.shared::cluster")
        || contents.contains(".shared::cluster")
        || contains_cluster_fence_features(contents)
        || contains_cluster_scoped_memory_features(contents)
}

fn contains_cluster_fence_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("fence.sc.cluster")
            || statement.contains("fence.acq_rel.cluster")
            || statement.contains("fence.acquire.cluster")
            || statement.contains("fence.release.cluster")
    })
}

fn contains_cluster_scoped_memory_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        !statement.contains("multimem.")
            && statement.contains(".cluster.")
            && ["ld.", "st.", "atom.", "red."]
                .iter()
                .any(|mnemonic| statement.contains(mnemonic))
    })
}

/// Checks the one-way fence semantics added in PTX 8.6.
///
/// Unlike the older `.sc` / `.acq_rel` forms, `.acquire` and `.release`
/// require sm_90 for every scope, not just `.cluster`.
fn contains_fence_acquire_release_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("fence.acquire.") || statement.contains("fence.release.")
    })
}

/// Checks the multimem instruction family introduced for sm_90.
///
/// Base forms need PTX 8.1. The pipeline currently has no 8.1 feature switch,
/// so PTX 8.6 is the nearest conservative version supported by LLVM.
fn contains_multimem_features(contents: &str) -> bool {
    contents.split(';').any(is_multimem_instruction)
}

fn is_multimem_instruction(statement: &str) -> bool {
    ["multimem.ld_reduce", "multimem.st", "multimem.red"]
        .iter()
        .any(|instruction| statement.contains(instruction))
}

/// Checks PTX 8.6 multimem formats that require a Blackwell family target.
fn contains_multimem_blackwell_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        is_multimem_instruction(statement)
            && [".e4m3", ".e5m2", ".acc::f16"]
                .iter()
                .any(|qualifier| statement.contains(qualifier))
    })
}

/// Checks the PTX 8.6 floating-point extension to `redux.sync`.
fn contains_redux_f32_features(contents: &str) -> bool {
    contents
        .split(';')
        .any(|statement| statement.contains("redux.sync") && statement.contains(".f32"))
}

/// Checks for forward-compatible instructions whose minimum target is sm_90.
///
/// Keep this category architecture-neutral: unlike WGMMA, these instructions
/// are not Hopper-specific and remain available on newer architectures.
fn contains_sm90_features(contents: &str) -> bool {
    ["add.rn.bf16x2", "sub.rn.bf16x2", "mul.rn.bf16x2"]
        .iter()
        .any(|mnemonic| contains_instruction_mnemonic(contents, mnemonic))
        || contains_stmatrix_features(contents)
        || contains_elect_features(contents)
        || contains_fence_acquire_release_features(contents)
        || contains_multimem_features(contents)
}

fn contains_elect_features(contents: &str) -> bool {
    contents.contains("elect.sync")
}

/// Checks for the register-only 8x8 matrix transpose (PTX 7.8, sm_75+).
fn contains_movmatrix_features(contents: &str) -> bool {
    contains_instruction_mnemonic(contents, "movmatrix.sync.aligned.m8n8.trans.b16")
}

/// Checks the dense BF16 MMA form added by the typed device intrinsic.
///
/// MMA shapes and types have different architecture and PTX ISA floors, so
/// this intentionally matches the complete operation-specific mnemonic.
fn contains_mma_m16n8k16_f32_bf16_features(contents: &str) -> bool {
    contains_instruction_mnemonic(
        contents,
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
    )
}

fn contains_instruction_mnemonic(contents: &str, mnemonic: &str) -> bool {
    contents.match_indices(mnemonic).any(|(index, _)| {
        let preceding = &contents[..index];
        let following = &contents[index + mnemonic.len()..];
        let escapes = ["\\09", "\\0A", "\\0B", "\\0C", "\\0D"];
        // Use PTX token delimiters rather than treating arbitrary punctuation
        // as a boundary. In particular, `$` and `%` participate in PTX
        // identifiers, and guarded opcodes have whitespace after `@{!}p`.
        let begins_at_instruction_boundary = preceding.is_empty()
            || preceding
                .chars()
                .next_back()
                .is_some_and(|ch| ch.is_whitespace() || matches!(ch, '"' | ';' | ':' | '{' | '}'))
            || escapes.iter().any(|escape| preceding.ends_with(escape))
            || preceding.ends_with("*/");
        let ends_at_instruction_boundary =
            following.chars().next().is_some_and(char::is_whitespace)
                || escapes.iter().any(|escape| following.starts_with(escape));
        begins_at_instruction_boundary && ends_at_instruction_boundary
    })
}

/// Checks the full PTX instruction families, including inline `ptx_asm!`
/// forms that cuda-oxide does not yet expose as typed wrappers.
///
/// Broad family matching is intentional. Missing a valid spelling can
/// silently select an architecture or PTX ISA that is too old; an invalid
/// spelling still reaches ptxas and fails there after conservative targeting.
fn contains_ldmatrix_features(contents: &str) -> bool {
    contents.contains("ldmatrix.sync.aligned.")
}

fn contains_stmatrix_features(contents: &str) -> bool {
    contents.contains("stmatrix.sync.aligned.")
}

/// PTX 8.6 matrix shapes/types have a Blackwell architecture-family floor.
fn contains_blackwell_matrix_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        let newer_ldmatrix = statement.contains("ldmatrix.sync.aligned.")
            && [".m16n16.", ".m8n16.", ".b8", ".src_fmt", ".dst_fmt"]
                .iter()
                .any(|token| statement.contains(token));
        let newer_stmatrix = statement.contains("stmatrix.sync.aligned.")
            && [".m16n8.", ".b8"]
                .iter()
                .any(|token| statement.contains(token));
        newer_ldmatrix || newer_stmatrix
    })
}

fn contains_ldmatrix_cta_state_space(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("ldmatrix.sync.aligned.") && statement.contains(".shared::cta.")
    })
}

/// Checks for features whose minimum target is sm_80.
///
/// This category includes packed bf16 operations introduced on Ampere and
/// non-bulk asynchronous copies. Match both the PTX spellings used in inline
/// assembly and the dotted LLVM NVVM intrinsic names for `cp.async`. Bulk and
/// tensor-copy forms have stronger requirements and are classified first.
fn contains_sm80_features(contents: &str) -> bool {
    [
        "fma.rn.bf16x2",
        "fma.rn.relu.bf16x2",
        "min.bf16x2",
        "max.bf16x2",
        "neg.bf16x2",
        "abs.bf16x2",
    ]
    .iter()
    .any(|mnemonic| contains_instruction_mnemonic(contents, mnemonic))
        || contents
            .split(';')
            .any(|statement| statement.contains("cvt.") && statement.contains(".bf16x2.f32"))
        || contains_mbarrier_features(contents)
        || contents.contains("redux.sync")
        || contents.contains("cp.async.ca.shared")
        || contents.contains("cp.async.cg.shared")
        || contents.contains("cp.async.commit_group")
        || contents.contains("cp.async.commit.group")
        || contents.contains("cp.async.wait_group")
        || contents.contains("cp.async.wait.group")
        || contents.contains("cp.async.wait_all")
        || contents.contains("cp.async.wait.all")
        || contains_mma_m16n8k16_f32_bf16_features(contents)
}

/// Checks for TMA/mbarrier instructions (Hopper+ compatible with Blackwell).
///
/// These instructions work on BOTH Hopper and Blackwell:
/// - TMA: Tensor Memory Accelerator bulk copies
/// - mbarrier: Async hardware barriers with transaction tracking
///
/// The architecture floor is generic sm_90; automatic cross-compilation keeps
/// the existing sm_100 default for forward-compatible Blackwell PTX.
fn contains_tma_features(contents: &str) -> bool {
    // TMA tensor copies and their commit/wait group controls.
    contains_cp_async_bulk_features(contents)
        || contains_mbarrier_sm90_features(contents)
        || contents.contains("fence.mbarrier_init")
        // Proxy fence for async operations
        || contents.contains("fence.proxy.async")
        || contents.contains(".sync_restrict")
}

fn contains_cp_async_bulk_features(contents: &str) -> bool {
    contents.contains("cp.async.bulk.")
}

fn contains_mbarrier_features(contents: &str) -> bool {
    contents.contains("mbarrier.") || contents.contains("llvm.nvvm.mbarrier")
}

fn contains_mbarrier_sm90_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        (statement.contains("mbarrier.") || statement.contains("llvm.nvvm.mbarrier"))
            && [
                "try_wait",
                "expect_tx",
                "complete_tx",
                "shared::cluster",
                ".acquire.",
                ".release.",
                ".relaxed",
            ]
            .iter()
            .any(|feature| statement.contains(feature))
    })
}

fn contains_mbarrier_ptx71_features(contents: &str) -> bool {
    contents
        .split(';')
        .any(|statement| statement.contains("mbarrier.test_wait") && statement.contains(".parity"))
}

fn contains_mbarrier_ptx78_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("mbarrier.")
            && (statement.contains("try_wait") || statement.contains("shared::cta"))
    })
}

fn contains_mbarrier_ptx80_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("mbarrier.")
            && [
                "expect_tx",
                "complete_tx",
                "shared::cluster",
                ".acquire.",
                ".release.",
            ]
            .iter()
            .any(|feature| statement.contains(feature))
    })
}

/// Checks for Blackwell tcgen05 instructions (sm_100a+).
///
/// These instructions require a datacenter-Blackwell `a`/`f` target; consumer
/// sm_120 does not provide Tensor Memory:
/// - tcgen05: Tensor Core Gen 5 (TMEM allocation, MMA, sync primitives)
///
/// Key differences from Hopper:
/// - tcgen05 MMA is single-thread (vs WGMMA's 128 threads)
/// - Uses Tensor Memory (TMEM) instead of registers
/// - Different synchronization model (mbarrier-based)
fn contains_blackwell_features(contents: &str) -> bool {
    // Keep the instruction-family match broad enough for inline PTX and LLVM
    // intrinsic names, but do not treat debug filenames such as `tcgen05.rs`
    // as an instruction.
    [
        "tcgen05.alloc",
        "tcgen05.dealloc",
        "tcgen05.relinquish_alloc_permit",
        "tcgen05.fence",
        "tcgen05.commit",
        "tcgen05.mma",
        "tcgen05.cp",
        "tcgen05.shift",
        "tcgen05.ld",
        "tcgen05.st",
        "tcgen05.wait",
    ]
    .iter()
    .any(|instruction| contents.contains(instruction))
}

/// Checks for base TMA multicast in LLVM IR or inline PTX.
///
/// TMA multicast (`cp.async.bulk.tensor...multicast::cluster`) is an optional
/// qualifier that broadcasts a tile to all CTAs in a cluster. It is legal on
/// sm_90+, although NVIDIA advises an `a`/`f` target
/// for performance. In the LLVM intrinsic this is controlled by the trailing
/// `use_cta_mask` i1 argument being set to true.
fn contains_tma_multicast(contents: &str) -> bool {
    contents.lines().any(|line| {
        line.contains("g2s.tile") && (line.contains(", i1 1, i1") || line.contains(", i1 true, i1"))
    }) || contents.split(';').any(|statement| {
        statement.contains("cp.async.bulk.tensor") && statement.contains(".multicast::cluster")
    })
}

/// Checks Blackwell-only TMA forms with an explicit CTA-group qualifier.
fn contains_tma_cta_group_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("cp.async.bulk.tensor")
            && (statement.contains(".cta_group::1") || statement.contains(".cta_group::2"))
    }) || contents.lines().any(|line| {
        line.contains("g2s.tile") && (line.contains(", i32 1)") || line.contains(", i32 2)"))
    })
}

/// Checks TMA copies whose destination is CTA-local shared memory.
///
/// `.shared::cta` already existed as a source state space for shared-to-global
/// copies, so the following `.global` source qualifier is part of the match.
/// The destination form was introduced in PTX 8.6 but is valid on sm_90.
fn contains_tma_shared_cta_destination(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("cp.async.bulk.") && statement.contains(".shared::cta.global")
    })
}

/// Checks PTX 8.6 TMA modifiers with a generic sm_100 architecture floor.
fn contains_tma_sm100_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        if !statement.contains("cp.async.bulk.") {
            return false;
        }
        statement.contains(".cp_mask")
            || (contains_tma_gather_or_im2col(statement)
                && statement.contains(".shared::cta.global"))
    })
}

/// Checks PTX 8.6 TMA modes restricted to datacenter Blackwell targets.
fn contains_tma_blackwell_accelerated_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        if !statement.contains("cp.async.bulk.") {
            return false;
        }
        statement.contains(".tile::scatter4")
            || statement.contains(".im2col::w::128")
            || (contains_tma_gather_or_im2col(statement)
                && !statement.contains(".shared::cta.global"))
    })
}

fn contains_tma_gather_or_im2col(statement: &str) -> bool {
    statement.contains(".tile::gather4")
        || (statement.contains(".im2col::w") && !statement.contains(".im2col::w::128"))
}

fn contains_tma_ptx86_features(contents: &str) -> bool {
    contains_tma_sm100_features(contents)
        || contains_tma_blackwell_accelerated_features(contents)
        || contents.contains(".sync_restrict")
        || contents
            .split(';')
            .any(|statement| statement.contains("mbarrier.") && statement.contains(".relaxed"))
}

fn contains_clc_features(contents: &str) -> bool {
    contents.contains("clusterlaunchcontrol.")
}

fn contains_clc_multicast_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("clusterlaunchcontrol.")
            && statement.contains(".multicast::cluster::all")
    })
}

fn contains_cluster_ptx80_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("barrier.cluster.")
            && [".release", ".relaxed", ".acquire"]
                .iter()
                .any(|qualifier| statement.contains(qualifier))
    })
}

/// GPU feature requirements detected in one LLVM module.
///
/// This is a set rather than a single "strongest" feature: architecture
/// families are not totally ordered. For example, WGMMA requires Hopper
/// `sm_90a`, while PTX 8.6 matrix forms require Blackwell. Keeping every bit
/// lets target validation enforce the intersection instead of silently
/// choosing whichever instruction happened to have higher detector priority.
#[derive(Clone, Copy, PartialEq, Eq)]
struct DetectedFeatures(u32);

#[allow(non_upper_case_globals)]
impl DetectedFeatures {
    /// tcgen05/TMEM (Blackwell datacenter, sm_100a).
    const Blackwell: Self = Self(1 << 0);
    /// Base TMA multicast (sm_90+, with architecture/family targets preferred).
    const TmaMulticast: Self = Self(1 << 1);
    /// Explicit CTA-group TMA forms (Blackwell datacenter family).
    const TmaCtaGroup: Self = Self(1 << 2);
    /// PTX 8.6 ldmatrix/stmatrix shapes supported on Blackwell family targets.
    const MatrixBlackwell: Self = Self(1 << 3);
    /// WGMMA (Hopper only, sm_90a - NOT forward-compatible).
    const Wgmma: Self = Self(1 << 4);
    /// TMA/mbarrier (Hopper+ compatible).
    const Tma: Self = Self(1 << 5);
    /// Thread Block Clusters (sm_90+, forward-compatible).
    const Cluster: Self = Self(1 << 6);
    /// Forward-compatible instructions with an sm_90 floor.
    const Sm90: Self = Self(1 << 7);
    /// Forward-compatible instructions with an sm_80 floor.
    const Sm80: Self = Self(1 << 8);
    /// Warp matrix register transpose introduced in PTX 7.8 on sm_75.
    const Movmatrix: Self = Self(1 << 9);
    /// Warp matrix shared-memory load introduced in PTX 6.5 on sm_75.
    const Ldmatrix: Self = Self(1 << 10);
    /// No special features (Volta+, with an sm_80 cross-compile default).
    const Basic: Self = Self(1 << 11);
    /// Generic Blackwell-or-newer operations such as base CLC and TMA cp_mask.
    const Sm100: Self = Self(1 << 12);
    /// Architecture/family-specific Blackwell features also available on consumers.
    const BlackwellFamily: Self = Self(1 << 13);
    /// Architecture/family-specific datacenter Blackwell TMA modes.
    const BlackwellAccelerated: Self = Self(1 << 14);
    /// Floating-point `redux.sync` (the sm_100/sm_103 architecture family).
    const ReduxF32: Self = Self(1 << 15);
    /// FP8 / f16-accumulator multimem forms on supported Blackwell families.
    const MultimemFp8: Self = Self(1 << 16);

    const ALL: [Self; 17] = [
        Self::Blackwell,
        Self::TmaCtaGroup,
        Self::BlackwellAccelerated,
        Self::BlackwellFamily,
        Self::ReduxF32,
        Self::MultimemFp8,
        Self::TmaMulticast,
        Self::MatrixBlackwell,
        Self::Wgmma,
        Self::Tma,
        Self::Cluster,
        Self::Sm90,
        Self::Sm80,
        Self::Movmatrix,
        Self::Ldmatrix,
        Self::Sm100,
        Self::Basic,
    ];

    const fn empty() -> Self {
        Self(0)
    }

    const fn contains(self, feature: Self) -> bool {
        self.0 & feature.0 != 0
    }

    fn insert(&mut self, feature: Self) {
        self.0 |= feature.0;
    }

    fn iter(self) -> impl Iterator<Item = Self> {
        Self::ALL
            .into_iter()
            .filter(move |feature| self.contains(*feature))
    }

    fn name(self) -> &'static str {
        match self {
            Self::Blackwell => "Blackwell",
            Self::TmaMulticast => "TmaMulticast",
            Self::TmaCtaGroup => "TmaCtaGroup",
            Self::MatrixBlackwell => "MatrixBlackwell",
            Self::Wgmma => "Wgmma",
            Self::Tma => "Tma",
            Self::Cluster => "Cluster",
            Self::Sm90 => "Sm90",
            Self::Sm80 => "Sm80",
            Self::Movmatrix => "Movmatrix",
            Self::Ldmatrix => "Ldmatrix",
            Self::Sm100 => "Sm100",
            Self::BlackwellFamily => "BlackwellFamily",
            Self::BlackwellAccelerated => "BlackwellAccelerated",
            Self::ReduxF32 => "ReduxF32",
            Self::MultimemFp8 => "MultimemFp8",
            Self::Basic => "Basic",
            _ => "Unknown",
        }
    }
}

impl std::fmt::Debug for DetectedFeatures {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for feature in self.iter() {
            if !first {
                formatter.write_str(" + ")?;
            }
            formatter.write_str(feature.name())?;
            first = false;
        }
        Ok(())
    }
}

impl std::ops::BitOr for DetectedFeatures {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// PTX ISA requirements are independent of the GPU architecture floor.
///
/// For example, a module may need sm_80 because it uses `cp.async` and still
/// need PTX 7.8 because it also uses `movmatrix`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PtxIsaRequirement {
    Default,
    Ptx65,
    Ptx70,
    Ptx71,
    Ptx78,
    Ptx80,
    Ptx86,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModuleRequirements {
    features: DetectedFeatures,
    ptx_isa: PtxIsaRequirement,
}

/// Detect every architecture requirement in exported LLVM text.
///
/// Both the ordinary PTX path and automatic libdevice mode use this exact
/// detector. The latter renders an in-memory preview before choosing the NVVM
/// pointer dialect.
fn detect_features_in_llvm_text(contents: &str) -> DetectedFeatures {
    let mut features = DetectedFeatures::empty();
    for (present, feature) in [
        (
            contains_blackwell_features(contents),
            DetectedFeatures::Blackwell,
        ),
        (
            contains_tma_cta_group_features(contents),
            DetectedFeatures::TmaCtaGroup,
        ),
        (
            contains_tma_blackwell_accelerated_features(contents),
            DetectedFeatures::BlackwellAccelerated,
        ),
        (
            contains_clc_multicast_features(contents),
            DetectedFeatures::BlackwellFamily,
        ),
        (
            contains_redux_f32_features(contents),
            DetectedFeatures::ReduxF32,
        ),
        (
            contains_multimem_blackwell_features(contents),
            DetectedFeatures::MultimemFp8,
        ),
        (
            contains_tma_multicast(contents),
            DetectedFeatures::TmaMulticast,
        ),
        (
            contains_blackwell_matrix_features(contents),
            DetectedFeatures::MatrixBlackwell,
        ),
        (contains_wgmma_features(contents), DetectedFeatures::Wgmma),
        (contains_tma_features(contents), DetectedFeatures::Tma),
        (
            contains_cluster_features(contents),
            DetectedFeatures::Cluster,
        ),
        (contains_sm90_features(contents), DetectedFeatures::Sm90),
        (contains_sm80_features(contents), DetectedFeatures::Sm80),
        (
            contains_movmatrix_features(contents),
            DetectedFeatures::Movmatrix,
        ),
        (
            contains_ldmatrix_features(contents),
            DetectedFeatures::Ldmatrix,
        ),
        (
            contains_tma_sm100_features(contents) || contains_clc_features(contents),
            DetectedFeatures::Sm100,
        ),
    ] {
        if present {
            features.insert(feature);
        }
    }
    if features == DetectedFeatures::empty() {
        features.insert(DetectedFeatures::Basic);
    }
    features
}

fn detect_module_requirements_in_llvm_text(contents: &str) -> ModuleRequirements {
    let mut ptx_isa = PtxIsaRequirement::Default;
    if contains_ldmatrix_features(contents) {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::Ptx65);
    }
    if contains_mbarrier_features(contents)
        || contents.contains("redux.sync")
        || contains_mma_m16n8k16_f32_bf16_features(contents)
    {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::Ptx70);
    }
    if contains_mbarrier_ptx71_features(contents) {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::Ptx71);
    }
    if contains_movmatrix_features(contents)
        || contains_stmatrix_features(contents)
        || contains_ldmatrix_cta_state_space(contents)
        || contains_cluster_features(contents)
        || contains_mbarrier_ptx78_features(contents)
    {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::Ptx78);
    }
    if contains_cp_async_bulk_features(contents)
        || contains_wgmma_features(contents)
        || contains_cluster_ptx80_features(contents)
        || contains_elect_features(contents)
        || contains_mbarrier_ptx80_features(contents)
        || contents.contains("fence.mbarrier_init")
        || contents.contains("fence.proxy.async")
    {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::Ptx80);
    }
    if contains_blackwell_matrix_features(contents)
        || contains_tma_cta_group_features(contents)
        || contains_tma_shared_cta_destination(contents)
        || contains_tma_ptx86_features(contents)
        || contains_clc_features(contents)
        || contains_blackwell_features(contents)
        || contains_fence_acquire_release_features(contents)
        || contains_multimem_features(contents)
        || contains_redux_f32_features(contents)
    {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::Ptx86);
    }

    ModuleRequirements {
        features: detect_features_in_llvm_text(contents),
        ptx_isa,
    }
}

fn detect_module_requirements_in_llvm_file(
    ll_path: &Path,
) -> Result<ModuleRequirements, PipelineError> {
    let contents = std::fs::read_to_string(ll_path).map_err(|error| {
        PipelineError::PtxGeneration(format!(
            "failed to inspect generated LLVM IR {}: {error}",
            ll_path.display()
        ))
    })?;
    Ok(detect_module_requirements_in_llvm_text(&contents))
}

/// Select a concrete architecture that satisfies every detected feature.
///
/// The first candidate preserves the established default for a module's most
/// restrictive-looking feature. The remaining candidates handle intersections
/// such as WGMMA + TMA multicast, whose only common target is `sm_90a`.
fn select_target(features: DetectedFeatures) -> Result<&'static str, String> {
    let preferred = if features.contains(DetectedFeatures::Blackwell)
        || features.contains(DetectedFeatures::TmaCtaGroup)
        || features.contains(DetectedFeatures::BlackwellAccelerated)
        || features.contains(DetectedFeatures::BlackwellFamily)
        || features.contains(DetectedFeatures::ReduxF32)
        || features.contains(DetectedFeatures::MultimemFp8)
        || features.contains(DetectedFeatures::TmaMulticast)
        || features.contains(DetectedFeatures::MatrixBlackwell)
    {
        "sm_100a"
    } else if features.contains(DetectedFeatures::Wgmma) {
        "sm_90a"
    } else if features.contains(DetectedFeatures::Sm100) {
        "sm_100"
    } else if features.contains(DetectedFeatures::Tma) {
        // Plain TMA is compatible with Hopper, but sm_100 is the existing
        // cross-compilation default because it produces forward-compatible
        // PTX for generic Blackwell devices.
        "sm_100"
    } else if features.contains(DetectedFeatures::Cluster)
        || features.contains(DetectedFeatures::Sm90)
    {
        "sm_90"
    } else if features.contains(DetectedFeatures::Sm80) {
        "sm_80"
    } else if features.contains(DetectedFeatures::Movmatrix)
        || features.contains(DetectedFeatures::Ldmatrix)
    {
        "sm_75"
    } else {
        "sm_80"
    };

    for candidate in [
        preferred, "sm_100a", "sm_90a", "sm_100", "sm_90", "sm_80", "sm_75",
    ] {
        if arch_satisfies(candidate, features) {
            return Ok(candidate);
        }
    }

    Err(format!(
        "detected CUDA features {features:?} do not share a compatible GPU architecture"
    ))
}

/// Does `arch` (e.g. `"sm_120a"`, `"sm_90"`) support the kernel's detected
/// features?
///
/// tcgen05/TMEM and explicit `cta_group` TMA forms exist only in the sm_100
/// datacenter-Blackwell family: consumer Blackwell (sm_120) and Hopper (sm_90)
/// lack them, so an sm_120 GPU cannot run an sm_100 tcgen05 kernel even though
/// 120 > 100. WGMMA is Hopper-only. The remaining features are forward
/// compatible from their floor (TMA / cluster / sm_90 features need sm_90+,
/// sm_80 features need sm_80+, and basic needs sm_70+).
///
/// Used to decide whether the GPU in this machine (the `CUDA_OXIDE_DEVICE_ARCH`
/// hint) can actually run the kernel, or whether we must build for the arch the
/// IR requires instead.
fn arch_satisfies(arch: &str, features: DetectedFeatures) -> bool {
    let Some((capability, suffix)) = arch_compute_capability_and_suffix(arch) else {
        return false;
    };
    if !is_known_cuda_target(capability, suffix) {
        return false;
    }
    features
        .iter()
        .all(|feature| arch_satisfies_feature(capability, suffix, feature))
}

fn arch_satisfies_feature(
    capability: u32,
    suffix: Option<char>,
    feature: DetectedFeatures,
) -> bool {
    let major = capability / 10;
    match feature {
        DetectedFeatures::Blackwell | DetectedFeatures::TmaCtaGroup => {
            supports_tcgen_target(capability, suffix)
        }
        DetectedFeatures::BlackwellAccelerated => {
            supports_blackwell_accelerated_target(capability, suffix)
        }
        DetectedFeatures::BlackwellFamily | DetectedFeatures::MatrixBlackwell => {
            supports_blackwell_family_target(capability, suffix)
        }
        DetectedFeatures::ReduxF32 => supports_redux_f32_target(capability, suffix),
        DetectedFeatures::MultimemFp8 => supports_multimem_fp8_target(capability, suffix),
        // The PTX ISA requires only sm_90+. The suffixed targets are advised
        // for performance, so target selection still prefers sm_100a.
        DetectedFeatures::TmaMulticast => major >= 9,
        DetectedFeatures::Wgmma => capability == 90 && suffix == Some('a'),
        DetectedFeatures::Sm100 => is_known_blackwell_capability(capability),
        DetectedFeatures::Tma | DetectedFeatures::Cluster | DetectedFeatures::Sm90 => major >= 9,
        DetectedFeatures::Sm80 => major >= 8,
        DetectedFeatures::Movmatrix | DetectedFeatures::Ldmatrix => capability >= 75,
        // Basic kernels are supported on the project's Volta+ floor. The
        // cross-compilation default remains sm_80, but a detected sm_70/sm_75
        // GPU is a valid and more useful target for `cargo oxide run`.
        DetectedFeatures::Basic => major >= 7,
        // `iter` only yields the single-bit constants above.
        _ => false,
    }
}

/// tcgen05/TMEM exists only on the datacenter Blackwell architecture or
/// family targets. Consumer sm_120 and generic targets without an `a`/`f`
/// suffix do not provide Tensor Memory.
fn supports_tcgen_target(capability: u32, suffix: Option<char>) -> bool {
    match suffix {
        // Architecture-specific targets are exact, not numerically forward
        // compatible. `sm_101a` is the PTX 8.x spelling later renamed to
        // `sm_110a`; accept both spellings plus the distinct sm_103 target.
        Some('a') => matches!(capability, 100 | 101 | 103 | 110),
        Some('f') => matches!(capability, 100 | 101 | 103 | 110),
        _ => false,
    }
}

fn supports_blackwell_accelerated_target(capability: u32, suffix: Option<char>) -> bool {
    match suffix {
        Some('a') => matches!(capability, 100 | 101 | 103 | 110),
        Some('f') => matches!(capability, 100 | 101 | 103 | 110),
        _ => false,
    }
}

fn supports_blackwell_family_target(capability: u32, suffix: Option<char>) -> bool {
    match suffix {
        Some('a') => matches!(capability, 100 | 101 | 110 | 120),
        Some('f') => matches!(capability, 100 | 101 | 103 | 110 | 120 | 121),
        _ => false,
    }
}

/// Floating-point `redux.sync` is scoped to the sm_100/sm_103 family.
fn supports_redux_f32_target(capability: u32, suffix: Option<char>) -> bool {
    matches!(suffix, Some('a' | 'f')) && matches!(capability, 100 | 103)
}

/// FP8 / f16-accumulator multimem forms span several Blackwell architecture
/// targets, but consumer family (`f`) targets do not support the sm_120 line.
fn supports_multimem_fp8_target(capability: u32, suffix: Option<char>) -> bool {
    match suffix {
        Some('a') => matches!(capability, 100 | 101 | 103 | 110 | 120 | 121),
        Some('f') => matches!(capability, 100 | 101 | 103 | 110),
        _ => false,
    }
}

fn is_known_blackwell_capability(capability: u32) -> bool {
    matches!(capability, 100 | 101 | 103 | 110 | 120 | 121)
}

fn is_known_cuda_target(capability: u32, suffix: Option<char>) -> bool {
    let known_capability = matches!(
        capability,
        70 | 72 | 75 | 80 | 86 | 87 | 88 | 89 | 90 | 100 | 101 | 103 | 110 | 120 | 121
    );
    known_capability
        && match suffix {
            None => true,
            Some('a') => capability == 90 || is_known_blackwell_capability(capability),
            Some('f') => is_known_blackwell_capability(capability),
            _ => false,
        }
}

fn validate_target_features(target: &CudaArch, features: DetectedFeatures) -> Result<(), String> {
    let compatible_default = select_target(features)?;
    if arch_satisfies(&target.sm(), features) {
        return Ok(());
    }

    Err(format!(
        "CUDA target {} cannot lower detected feature {features:?}; \
         cuda-oxide requires a target compatible with {} for this module",
        target.sm(),
        compatible_default
    ))
}

fn resolve_ptx_target(
    explicit_override: Option<&str>,
    device_hint: Option<&str>,
    detected: DetectedFeatures,
) -> Result<(String, &'static str), PipelineError> {
    if let Some(target) = explicit_override {
        let parsed = target.parse::<CudaArch>().map_err(|error| {
            PipelineError::PtxGeneration(format!("invalid CUDA_OXIDE_TARGET `{target}`: {error}"))
        })?;
        validate_target_features(&parsed, detected).map_err(PipelineError::PtxGeneration)?;
        return Ok((parsed.sm(), "CUDA_OXIDE_TARGET"));
    }

    if let Some(device) = device_hint.filter(|target| arch_satisfies(target, detected)) {
        return Ok((device.to_string(), "detected GPU"));
    }

    let target = select_target(detected).map_err(PipelineError::PtxGeneration)?;
    Ok((target.to_string(), "feature requirement"))
}

/// Select the PTX ISA independently from the GPU architecture.
///
/// LLVM GPU CPUs select a default PTX ISA independently from the hardware
/// feature floor. Raise that ISA only when the selected CPU's default is too
/// old; never force a newer target back to an older PTX version.
fn required_ptx_feature(target: &str, requirement: PtxIsaRequirement) -> Option<&'static str> {
    let capability = arch_compute_capability(target)?;
    let minimum = target_minimum_ptx_isa(capability)?;
    let requested = match requirement {
        PtxIsaRequirement::Default => return None,
        PtxIsaRequirement::Ptx65 => 65,
        PtxIsaRequirement::Ptx70 => 70,
        PtxIsaRequirement::Ptx71 => 71,
        PtxIsaRequirement::Ptx78 => 78,
        PtxIsaRequirement::Ptx80 => 80,
        PtxIsaRequirement::Ptx86 => 86,
    };
    if requested <= minimum {
        return None;
    }
    match requirement {
        PtxIsaRequirement::Default => None,
        PtxIsaRequirement::Ptx65 => Some("+ptx65"),
        PtxIsaRequirement::Ptx70 => Some("+ptx70"),
        PtxIsaRequirement::Ptx71 => Some("+ptx71"),
        PtxIsaRequirement::Ptx78 => Some("+ptx78"),
        PtxIsaRequirement::Ptx80 => Some("+ptx80"),
        PtxIsaRequirement::Ptx86 => Some("+ptx86"),
    }
}

/// Minimum PTX ISA accepted by LLVM for each concrete target. Passing an
/// older `+ptxNN` feature does not merely do nothing: LLVM aborts because that
/// ISA cannot name the selected processor.
fn target_minimum_ptx_isa(capability: u32) -> Option<u32> {
    match capability {
        70 => Some(60),
        72 => Some(61),
        75 => Some(63),
        80 => Some(70),
        86 => Some(71),
        87 => Some(74),
        88 => Some(90),
        89 | 90 => Some(78),
        100 | 101 => Some(86),
        103 => Some(88),
        110 => Some(90),
        120 => Some(87),
        121 => Some(88),
        _ => None,
    }
}

/// Reject targets that the supported LLVM 21 backend silently mishandles.
///
/// LLVM 21 accepts `-mcpu=sm_88` / `sm_110*` but only prints a warning and
/// emits PTX 6.0, which ptxas then rejects. LLVM 22 is the first backend in
/// cuda-oxide's supported toolchain set that emits valid PTX for these PTX 9.0
/// target spellings. An unknown version is rejected because it cannot prove
/// that the backend knows the processor.
fn validate_target_for_llvm_major(target: &str, llc_major: Option<u32>) -> Result<(), String> {
    let capability = arch_compute_capability(target);
    if matches!(capability, Some(88 | 110)) && llc_major.is_none_or(|major| major < 22) {
        let backend = llc_major.map_or_else(
            || "an LLVM backend with an unknown version".to_string(),
            |major| format!("LLVM {major}"),
        );
        return Err(format!(
            "CUDA target {target} requires LLVM 22 or newer; {backend} does not reliably emit valid PTX for this PTX 9.0 target"
        ));
    }
    Ok(())
}

/// Extract the compute-capability *major* version from an `sm_…` target string.
///
/// CUDA concatenates major+minor without a separator, so `"sm_120a"` is cc 12.0
/// (major 12), `"sm_90"` is cc 9.0, `"sm_103a"` is cc 10.3. We read the digit
/// run after `sm_` and divide by ten. Returns `None` when there are no digits.
#[cfg(test)]
fn arch_major(arch: &str) -> Option<u32> {
    arch_compute_capability(arch).map(|capability| capability / 10)
}

/// Extract the numeric compute capability from an `sm_…` target.
fn arch_compute_capability(arch: &str) -> Option<u32> {
    arch_compute_capability_and_suffix(arch).map(|(capability, _)| capability)
}

fn arch_compute_capability_and_suffix(arch: &str) -> Option<(u32, Option<char>)> {
    if !arch.starts_with("sm_") {
        return None;
    }
    let target = arch.parse::<CudaArch>().ok()?;
    Some((target.capability(), target.suffix()))
}

/// Runs LLVM's middle-end (`opt -O2`) on the emitted IR before `llc`.
///
/// This is what consumes the per-op ABI alignment we emit: the
/// LoadStoreVectorizer fuses aligned aggregate/element accesses, SROA
/// scalarizes stack aggregates, and InferAddressSpaces promotes generic
/// pointers to `.global` (LDG/STG). Gated on alignment — fusion only fires
/// when loads/stores carry matching `align N` hints.
///
/// The `opt` binary comes from the resolved [`LlvmToolchain`], which
/// guarantees it shares the LLVM major of the `llc` that will consume its
/// output (issue #150: an LLVM 22 `opt` emits sizeless
/// `llvm.lifetime.start/end` intrinsics that an LLVM 21 `llc` rejects).
///
/// Returns the optimised `.ll` path, or `None` when the middle-end is off
/// (`CUDA_OXIDE_NO_OPT=1`), no same-major `opt` exists, or the chosen `opt`
/// fails at runtime; the caller then feeds the unoptimised `ll_path` to
/// `llc`, which is always safe.
fn optimize_ll(ll_path: &Path, toolchain: &LlvmToolchain, verbose: bool) -> Option<PathBuf> {
    let opt = toolchain.opt.as_ref()?;

    let opt_ll = ll_path.with_extension("opt.ll");
    match std::process::Command::new(&opt.path)
        .arg("-O2")
        .arg(ll_path)
        .arg("-S")
        .arg("-o")
        .arg(&opt_ll)
        .output()
    {
        Ok(o) if o.status.success() => {
            if verbose {
                eprintln!("opt -O2 via {}: {}", opt.path, opt_ll.display());
            }
            Some(opt_ll)
        }
        Ok(o) => {
            // The matched opt exists but rejected the input. Warn loudly
            // (there is no second candidate any more) and fall back to
            // unoptimised IR rather than to a different LLVM major.
            eprintln!(
                "warning: opt ({}) failed; continuing with unoptimised IR:\n{}",
                opt.path,
                String::from_utf8_lossy(&o.stderr).trim()
            );
            None
        }
        Err(e) => {
            eprintln!(
                "warning: opt ({}): {e}; continuing with unoptimised IR",
                opt.path
            );
            None
        }
    }
}

/// Generates PTX from LLVM IR using `llc`.
///
/// LLVM 21+ is the minimum supported version:
/// earlier `llc` releases reject the modern TMA / tcgen05 / WGMMA
/// intrinsic signatures that cuda-oxide emits (e.g. the 10-operand
/// `llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d` with `addrspace(7)` + CTA
/// group parameter requires LLVM 21). If `CUDA_OXIDE_LLC` is set, it is used
/// exclusively — power users can point this at an older `llc` at their own
/// risk (most examples will still compile but modern intrinsics will not).
///
/// `opt` and `llc` are resolved together via [`LlvmToolchain`] so the
/// middle-end never runs under a different LLVM major than the backend
/// (issue #150).
///
/// Target arch resolves (highest priority first) to: an explicit
/// `CUDA_OXIDE_TARGET` override, else the detected-GPU hint
/// (`CUDA_OXIDE_DEVICE_ARCH`) when that GPU can run the kernel, else the minimum
/// arch the IR's features require (`select_target`).
fn generate_ptx(
    ll_path: &Path,
    ptx_path: &Path,
    debug_kind: DebugKind,
) -> Result<String, PipelineError> {
    // Explicit, hard override: `--arch` or a parent-set `CUDA_OXIDE_TARGET`.
    let explicit_override = std::env::var("CUDA_OXIDE_TARGET").ok();
    // Advisory hint: the arch of the GPU in this machine, forwarded by
    // `cargo oxide run`. Used only when that GPU can actually run the kernel.
    let device_hint = std::env::var("CUDA_OXIDE_DEVICE_ARCH").ok();

    let requirements = detect_module_requirements_in_llvm_file(ll_path)?;
    let detected = requirements.features;

    // Resolve the final target:
    //   1. explicit override -- accepted only if it can lower the kernel's
    //      features; reject an invalid floor before llc emits unusable PTX.
    //   2. detected-device hint -- used only if that GPU can run the kernel;
    //      otherwise we build for `feature_arch`. The resulting PTX will not
    //      load on this GPU, but feature-gated examples handle that at load time
    //      (cuModuleLoad reports INVALID_PTX and they skip execution).
    //   3. neither set -- the feature floor.
    let (target, target_source) = resolve_ptx_target(
        explicit_override.as_deref(),
        device_hint.as_deref(),
        detected,
    )?;

    // Log target selection
    if std::env::var("CUDA_OXIDE_VERBOSE").is_ok() {
        eprintln!("Target: {target} (from {target_source}; detected {detected:?})");
    }

    let verbose = std::env::var("CUDA_OXIDE_VERBOSE").is_ok();

    // Resolve `opt` and `llc` as a matched pair (issue #150): llc first
    // (CUDA_OXIDE_LLC, then the Rust toolchain's llvm-tools llc, then
    // llc-22 / llc-21 on PATH — newest first for best atomics/scope
    // support), then an opt of the same LLVM major. LLVM 21 is the floor:
    // older releases reject modern TMA / tcgen05 / WGMMA intrinsic
    // signatures that cuda-oxide emits. Users on older distros can opt in
    // to a specific `llc` via `CUDA_OXIDE_LLC`.
    let Some(toolchain) = LlvmToolchain::resolve(verbose) else {
        return Err(PipelineError::PtxGeneration(
            "No working llc found.\n\
             cuda-oxide tries (in order): CUDA_OXIDE_LLC, the Rust toolchain's \
             llvm-tools llc, then llc-22 / llc-21 on PATH. \
             LLVM 21+ is required (earlier versions reject the TMA / tcgen05 / \
             WGMMA intrinsic signatures we emit).\n\
             Easiest fix: `rustup component add llvm-tools` (auto-picked up).\n\
             Alternative: `sudo apt install llvm-21` (or `llvm-22`).\n\
             Or set CUDA_OXIDE_LLC=/path/to/llc to use a specific binary."
                .to_string(),
        ));
    };
    validate_target_for_llvm_major(&target, toolchain.llc_major)
        .map_err(PipelineError::PtxGeneration)?;

    // Run the LLVM middle-end (opt -O2) before llc. Feature detection above
    // intentionally reads the original (pre-opt) IR so the target is
    // determined by what the source actually needs, not what opt elides.
    //
    // Full-debug is a `-G`-style build: it keeps every local in memory and
    // describes it with `llvm.dbg.declare`. Running `opt -O2` would promote
    // those slots to registers and collapse their live ranges, turning most
    // in-scope locals into `<optimized out>` under cuda-gdb. So we feed the
    // unoptimized IR straight to llc when variable info is requested, matching
    // nvcc `-G`. (llc itself is invoked at `-O0` for the same builds below.)
    let optimized = if debug_kind.variables_enabled() {
        if verbose {
            eprintln!("Skipping opt -O2 (full debug keeps locals inspectable)");
        }
        None
    } else {
        optimize_ll(ll_path, &toolchain, verbose)
    };
    let llc_input: &Path = optimized.as_deref().unwrap_or(ll_path);

    // Target reference:
    //   - sm_100a: Blackwell datacenter (tcgen05/TMEM)
    //   - sm_90a:  Hopper only (WGMMA + TMA) - NOT forward-compatible
    //   - sm_120:  Blackwell consumer (TMA with PTX 8.7)
    //   - sm_80:   Ampere+ (maximum compatibility)
    if verbose {
        let source = if toolchain.llc_from_env {
            "from CUDA_OXIDE_LLC"
        } else {
            "auto-detected"
        };
        eprintln!("Using llc: {} ({source})", toolchain.llc_description());
    }
    // How to name the llc in errors: keep the env var visible when it was
    // the source so users connect the failure to their own pin.
    let llc_desc = if toolchain.llc_from_env {
        format!("CUDA_OXIDE_LLC={}", toolchain.llc_path)
    } else {
        format!("llc ({})", toolchain.llc_path)
    };

    let mut llc_cmd = std::process::Command::new(&toolchain.llc_path);
    llc_cmd
        .arg("-march=nvptx64")
        .arg(format!("-mcpu={}", target));
    if let Some(feature) = required_ptx_feature(&target, requirements.ptx_isa) {
        llc_cmd.arg(format!("-mattr={feature}"));
    }
    // Full-debug (`-G`-style): run llc at -O0 so its own mem2reg/SROA does not
    // promote the stack slots we deliberately kept in memory, which would
    // invalidate the `llvm.dbg.declare` locations cuda-gdb reads.
    if debug_kind.variables_enabled() {
        llc_cmd.arg("-O0");
    }
    // Fuse fmul+fadd/fsub into fma.rn.f32, matching nvcc's default --fmad=true.
    // The IR-side `contract` flag (set by add_fastmath_flags in mir-lower) grants
    // permission; this llc flag activates the NVPTX backend's contract mode.
    // Set CUDA_OXIDE_NO_FMA=1 or pass --no-fmad to cargo oxide to opt out.
    if std::env::var("CUDA_OXIDE_NO_FMA").is_err() {
        llc_cmd.arg("-fp-contract=fast");
    }
    let result = llc_cmd.arg(llc_input).arg("-o").arg(ptx_path).output();

    match result {
        Ok(output) if output.status.success() => {
            if matches!(debug_kind, DebugKind::LineTables) {
                strip_target_debug_from_ptx(ptx_path)?;
                if verbose {
                    eprintln!(
                        "line-table debug: stripped PTX target debug flag; source line tables remain"
                    );
                }
            }
            Ok(target.to_string())
        }
        Ok(output) => Err(PipelineError::PtxGeneration(format!(
            "{} failed:\n{}",
            llc_desc,
            String::from_utf8_lossy(&output.stderr).trim()
        ))),
        Err(e) => Err(PipelineError::PtxGeneration(format!("{llc_desc}: {e}"))),
    }
}

fn strip_target_debug_from_ptx(ptx_path: &Path) -> Result<(), PipelineError> {
    let ptx = std::fs::read_to_string(ptx_path).map_err(|e| {
        PipelineError::PtxGeneration(format!(
            "failed to read PTX for line-table debug cleanup ({}): {e}",
            ptx_path.display()
        ))
    })?;
    let stripped = strip_target_debug_from_ptx_text(&ptx);
    if stripped != ptx {
        std::fs::write(ptx_path, stripped).map_err(|e| {
            PipelineError::PtxGeneration(format!(
                "failed to write PTX after line-table debug cleanup ({}): {e}",
                ptx_path.display()
            ))
        })?;
    }
    Ok(())
}

fn strip_target_debug_from_ptx_text(ptx: &str) -> String {
    let mut out = String::with_capacity(ptx.len());
    for line in ptx.split_inclusive('\n') {
        let (line_body, newline) = line
            .strip_suffix('\n')
            .map_or((line, ""), |without_newline| (without_newline, "\n"));
        out.push_str(&strip_target_debug_from_ptx_line(line_body));
        out.push_str(newline);
    }
    out
}

fn strip_target_debug_from_ptx_line(line: &str) -> String {
    let indent_len = line.len() - line.trim_start().len();
    let indent = &line[..indent_len];
    let body = &line[indent_len..];
    let Some(rest) = body.strip_prefix(".target") else {
        return line.to_string();
    };

    let mut parts = rest.split(',');
    let Some(arch) = parts.next() else {
        return line.to_string();
    };

    let options: Vec<&str> = parts
        .map(str::trim)
        .filter(|option| *option != "debug")
        .collect();
    if !rest
        .split(',')
        .skip(1)
        .any(|option| option.trim() == "debug")
    {
        return line.to_string();
    }

    let mut stripped = format!("{indent}.target{arch}");
    for option in options {
        stripped.push_str(", ");
        stripped.push_str(option);
    }
    stripped
}

/// Recursively finds the innermost operation that failed verification.
///
/// Helps produce better error messages by pointing to the specific failing
/// operation rather than just the containing module/function.
fn find_inner_verification_error(
    ctx: &Context,
    op_ptr: Ptr<Operation>,
) -> Option<(Ptr<Operation>, String)> {
    let op = op_ptr.deref(ctx);

    for region in op.regions() {
        let region_ref = region.deref(ctx);
        for block in region_ref.iter(ctx) {
            let block_ref = block.deref(ctx);
            for child_op in block_ref.iter(ctx) {
                if let Some(err) = find_inner_verification_error(ctx, child_op) {
                    return Some(err);
                }
            }
        }
    }

    if let Err(e) = op.verify(ctx) {
        // Use .disp(ctx) to get full error with location and backtrace
        return Some((op_ptr, e.disp(ctx).to_string()));
    }

    None
}

/// Errors from pipeline execution, categorized by stage.
#[derive(Debug)]
pub enum PipelineError {
    /// Function has no MIR body (shouldn't happen for collected functions).
    NoBody(String),
    /// MIR→Pliron IR translation failed.
    Translation(String),
    /// Pliron IR verification failed (includes failing operation if found).
    Verification {
        name: String,
        message: String,
        operation: Option<String>,
    },
    /// MIR→LLVM lowering failed.
    Lowering(String),
    /// LLVM IR export failed.
    Export(String),
    /// PTX generation via `llc` failed.
    PtxGeneration(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoBody(name) => write!(f, "Function '{}' has no MIR body", name),
            Self::Translation(msg) => write!(f, "Translation failed: {}", msg),
            Self::Verification {
                name,
                message,
                operation,
            } => {
                writeln!(f, "Verification failed for '{}':", name)?;
                writeln!(f, "  {}", message)?;
                if let Some(op) = operation {
                    writeln!(f, "  Failed operation:\n{}", op)?;
                }
                Ok(())
            }
            Self::Lowering(msg) => write!(f, "Lowering failed: {}", msg),
            Self::Export(msg) => write!(f, "Export failed: {}", msg),
            Self::PtxGeneration(msg) => write!(f, "PTX generation failed: {}", msg),
        }
    }
}

impl std::error::Error for PipelineError {}

#[cfg(test)]
mod tests {
    use super::*;
    use llvm_export::export::AsDeviceExtern;
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
    fn line_table_ptx_cleanup_strips_only_target_debug_flag() {
        let ptx = "\
.version 8.9
.target sm_120a, debug
.address_size 64

.section .debug_info
\t.b8 1;
";

        let stripped = strip_target_debug_from_ptx_text(ptx);

        assert!(
            stripped.contains(".target sm_120a\n"),
            "line-table mode should not ask the driver for debug compilation:\n{stripped}"
        );
        assert!(
            stripped.contains(".section .debug_info"),
            "line-table mode must keep the emitted DWARF sections:\n{stripped}"
        );
    }

    #[test]
    fn line_table_ptx_cleanup_preserves_other_target_options() {
        let ptx = ".target sm_90a, texmode_independent, debug\n";

        let stripped = strip_target_debug_from_ptx_text(ptx);

        assert_eq!(stripped, ".target sm_90a, texmode_independent\n");
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
        };

        let result = run_pipeline(&[], &[], &config).expect("pipeline run");

        assert!(output_dir.is_dir());
        assert!(result.ll_path.is_file());
        assert_eq!(result.artifact_path, result.ll_path);
        assert_eq!(result.artifact_kind, CompilationArtifactKind::NvvmIr);
        assert_eq!(result.target, "sm_86");
        assert_eq!(
            fs::read_to_string(output_dir.join("empty.target")).unwrap(),
            "sm_86\n"
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

    #[test]
    fn nvvm_target_resolution_is_concrete_and_strict() {
        let legacy = resolve_nvvm_target(Some("compute_90a"), Some("sm_120"), None).unwrap();
        assert_eq!(legacy.sm(), "sm_90a");
        assert!(legacy.uses_legacy_llvm());

        let modern = resolve_nvvm_target(None, Some("sm_120f"), None).unwrap();
        assert_eq!(modern.compute(), "compute_120f");
        assert!(!modern.uses_legacy_llvm());

        for target in [None, Some("nvvm-ir"), Some("sm_90x"), Some("86")] {
            assert!(
                resolve_nvvm_target(target, None, None).is_err(),
                "{target:?}"
            );
        }
    }

    #[test]
    fn automatic_nvvm_target_uses_the_module_feature_floor() {
        for (features, expected, is_legacy) in [
            (DetectedFeatures::Basic, "sm_80", true),
            (DetectedFeatures::Ldmatrix, "sm_75", true),
            (DetectedFeatures::Movmatrix, "sm_75", true),
            (DetectedFeatures::Sm80, "sm_80", true),
            (DetectedFeatures::Sm90, "sm_90", true),
            (DetectedFeatures::Cluster, "sm_90", true),
            (DetectedFeatures::Wgmma, "sm_90a", true),
            (DetectedFeatures::Tma, "sm_100", false),
            (DetectedFeatures::Sm100, "sm_100", false),
            (DetectedFeatures::TmaCtaGroup, "sm_100a", false),
            (DetectedFeatures::BlackwellAccelerated, "sm_100a", false),
            (DetectedFeatures::BlackwellFamily, "sm_100a", false),
            (DetectedFeatures::TmaMulticast, "sm_100a", false),
            (DetectedFeatures::MatrixBlackwell, "sm_100a", false),
            (DetectedFeatures::Blackwell, "sm_100a", false),
        ] {
            let target = resolve_nvvm_target(None, None, Some(features)).unwrap();
            assert_eq!(target.sm(), expected, "{features:?}");
            assert_eq!(target.uses_legacy_llvm(), is_legacy, "{features:?}");
        }
    }

    #[test]
    fn automatic_nvvm_target_uses_only_a_compatible_device_hint() {
        let turing =
            resolve_nvvm_target(None, Some("sm_75"), Some(DetectedFeatures::Basic)).unwrap();
        assert_eq!(turing.sm(), "sm_75");

        let sm80_on_turing =
            resolve_nvvm_target(None, Some("sm_75"), Some(DetectedFeatures::Sm80)).unwrap();
        assert_eq!(sm80_on_turing.sm(), "sm_80");

        let movmatrix_on_volta =
            resolve_nvvm_target(None, Some("sm_70"), Some(DetectedFeatures::Movmatrix)).unwrap();
        assert_eq!(movmatrix_on_volta.sm(), "sm_75");

        let ldmatrix_on_volta =
            resolve_nvvm_target(None, Some("sm_70"), Some(DetectedFeatures::Ldmatrix)).unwrap();
        assert_eq!(ldmatrix_on_volta.sm(), "sm_75");

        let blackwell =
            resolve_nvvm_target(None, Some("sm_120a"), Some(DetectedFeatures::Basic)).unwrap();
        assert_eq!(blackwell.sm(), "sm_120a");

        let sm80_on_blackwell =
            resolve_nvvm_target(None, Some("sm_120a"), Some(DetectedFeatures::Sm80)).unwrap();
        assert_eq!(sm80_on_blackwell.sm(), "sm_120a");

        let ampere =
            resolve_nvvm_target(None, Some("sm_80"), Some(DetectedFeatures::Sm80)).unwrap();
        assert_eq!(ampere.sm(), "sm_80");

        let hopper_floor =
            resolve_nvvm_target(None, Some("sm_80"), Some(DetectedFeatures::Sm90)).unwrap();
        assert_eq!(hopper_floor.sm(), "sm_90");

        let forward_compatible =
            resolve_nvvm_target(None, Some("sm_120"), Some(DetectedFeatures::Sm90)).unwrap();
        assert_eq!(forward_compatible.sm(), "sm_120");

        let hopper =
            resolve_nvvm_target(None, Some("sm_120a"), Some(DetectedFeatures::Wgmma)).unwrap();
        assert_eq!(hopper.sm(), "sm_90a");

        assert!(
            resolve_nvvm_target(None, Some("not-an-arch"), Some(DetectedFeatures::Basic)).is_err()
        );
    }

    #[test]
    fn explicit_nvvm_target_rejects_a_detected_feature_below_its_floor() {
        let error = resolve_nvvm_target(Some("sm_70"), None, Some(DetectedFeatures::Movmatrix))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("cannot lower detected feature Movmatrix"),
            "{error}"
        );

        let target =
            resolve_nvvm_target(Some("sm_75"), None, Some(DetectedFeatures::Movmatrix)).unwrap();
        assert_eq!(target.sm(), "sm_75");
    }

    #[test]
    fn compatible_explicit_nvvm_target_wins_over_automatic_selection() {
        let target =
            resolve_nvvm_target(Some("sm_86"), Some("sm_120a"), Some(DetectedFeatures::Sm80))
                .unwrap();
        assert_eq!(target.sm(), "sm_86");
    }

    #[test]
    fn explicit_ptx_target_cannot_undercut_matrix_feature_floors() {
        let ldmatrix = resolve_ptx_target(Some("sm_70"), None, DetectedFeatures::Ldmatrix)
            .expect_err("sm_70 cannot lower ldmatrix")
            .to_string();
        assert!(ldmatrix.contains("cannot lower detected feature Ldmatrix"));

        let stmatrix = resolve_ptx_target(Some("sm_80"), None, DetectedFeatures::Sm90)
            .expect_err("sm_80 cannot lower stmatrix")
            .to_string();
        assert!(stmatrix.contains("cannot lower detected feature Sm90"));

        let newer = resolve_ptx_target(Some("sm_100"), None, DetectedFeatures::MatrixBlackwell)
            .expect_err("generic sm_100 lacks architecture-family matrix features")
            .to_string();
        assert!(newer.contains("cannot lower detected feature MatrixBlackwell"));
    }

    #[test]
    fn legacy_nvvm_debug_is_rejected() {
        let legacy = resolve_nvvm_target(Some("sm_90"), None, None).unwrap();
        assert!(
            validate_nvvm_debug_support(
                &legacy,
                NvvmIrDialect::LegacyLlvm7,
                DebugKind::LineTables,
            )
            .is_err()
        );
        validate_nvvm_debug_support(&legacy, NvvmIrDialect::LegacyLlvm7, DebugKind::Off).unwrap();

        let modern = resolve_nvvm_target(Some("sm_120"), None, None).unwrap();
        validate_nvvm_debug_support(&modern, NvvmIrDialect::Modern, DebugKind::Full).unwrap();
    }

    #[test]
    fn test_device_extern_decl_converts_to_export_decl() {
        let decl = DeviceExternDecl {
            export_name: "device_add".to_string(),
            param_types: vec![
                DeviceExternType::pointer_to(DeviceExternType::Float32, 0),
                DeviceExternType::Integer(32),
            ],
            return_type: DeviceExternType::Void,
            attrs: DeviceExternAttrs {
                is_convergent: true,
                is_pure: false,
                is_readonly: true,
            },
        };

        let exported = decl.as_device_extern();

        assert_eq!(exported.export_name, "device_add");
        assert_eq!(
            exported.param_types,
            [
                DeviceExternType::pointer_to(DeviceExternType::Float32, 0),
                DeviceExternType::Integer(32),
            ]
        );
        assert_eq!(exported.return_type, DeviceExternType::Void);
        assert!(exported.attrs.is_convergent);
        assert!(!exported.attrs.is_pure);
        assert!(exported.attrs.is_readonly);
    }

    #[test]
    fn test_feature_detection_reads_llvm_ir_snippets() {
        let llvm = r#"
            call void asm sideeffect "wgmma.fence.sync.aligned", ""()
            call void @llvm.nvvm.tcgen05.alloc()
            call void asm sideeffect "cluster.sync.aligned", ""()
            call void asm sideeffect "cp.async.bulk.tensor.2d.shared::cluster.global", ""()
            call void asm sideeffect "cp.async.ca.shared.global", ""()
        "#;

        assert!(contains_wgmma_features(llvm));
        assert!(contains_blackwell_features(llvm));
        assert!(contains_cluster_features(llvm));
        assert!(contains_tma_features(llvm));
        assert!(contains_sm80_features(llvm));
        let detected = detect_features_in_llvm_text(llvm);
        for feature in [
            DetectedFeatures::Blackwell,
            DetectedFeatures::Wgmma,
            DetectedFeatures::Cluster,
            DetectedFeatures::Tma,
            DetectedFeatures::Sm80,
        ] {
            assert!(detected.contains(feature), "missing {feature:?}");
        }
        assert!(
            select_target(detected).is_err(),
            "Hopper-only WGMMA and Blackwell-only tcgen05 are incompatible"
        );
    }

    #[test]
    fn test_sm80_detection_accepts_inline_ptx_and_nvvm_intrinsics() {
        for llvm in [
            r#"call void asm sideeffect "cp.async.ca.shared.global [%0], [%1], 4;", "l,l"()"#,
            "call void @llvm.nvvm.cp.async.ca.shared.global.8(ptr addrspace(3) %dst, ptr addrspace(1) %src)",
            r#"call void asm sideeffect "cp.async.commit_group;", ""()"#,
            "call void @llvm.nvvm.cp.async.wait.all()",
        ] {
            assert!(contains_sm80_features(llvm), "missed cp.async in {llvm}");
            assert_eq!(detect_features_in_llvm_text(llvm), DetectedFeatures::Sm80);
        }
    }

    #[test]
    fn test_bf16x2_detection_matches_exact_architecture_floors() {
        for mnemonic in [
            "add.rn.bf16x2 $0, $1, $2;",
            "sub.rn.bf16x2 $0, $1, $2;",
            "mul.rn.bf16x2 $0, $1, $2;",
        ] {
            assert!(contains_sm90_features(mnemonic));
            assert!(!contains_sm80_features(mnemonic));
            assert_eq!(
                detect_features_in_llvm_text(mnemonic),
                DetectedFeatures::Sm90
            );
        }

        for mnemonic in ["add.rn.bf16x2\t$0, $1, $2;", "sub.rn.bf16x2\\09$0, $1, $2;"] {
            assert_eq!(
                detect_features_in_llvm_text(mnemonic),
                DetectedFeatures::Sm90,
                "{mnemonic:?}"
            );
        }

        for mnemonic in [
            "fma.rn.bf16x2 $0, $1, $2, $3;",
            "fma.rn.relu.bf16x2 $0, $1, $2, $3;",
            "min.bf16x2 $0, $1, $2;",
            "max.bf16x2 $0, $1, $2;",
            "neg.bf16x2 $0, $1;",
            "abs.bf16x2 $0, $1;",
        ] {
            assert!(!contains_sm90_features(mnemonic));
            assert!(contains_sm80_features(mnemonic));
            assert_eq!(
                detect_features_in_llvm_text(mnemonic),
                DetectedFeatures::Sm80
            );
        }

        for near_miss in [
            "add.rn.bf16x2x $0, $1, $2;",
            "fma.rn.bf16x2x $0, $1, $2, $3;",
            "add.rn.bf16x2\\5C09$0, $1, $2;",
        ] {
            assert!(!contains_sm90_features(near_miss));
            assert!(!contains_sm80_features(near_miss));
            assert_eq!(
                detect_features_in_llvm_text(near_miss),
                DetectedFeatures::Basic
            );
        }
    }

    #[test]
    fn dense_bf16_mma_detection_applies_exact_sm80_and_ptx70_floors() {
        let mnemonic =
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};";
        for spelling in [
            mnemonic,
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\t{$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\\09{$0}, {$1}, {$2}, {$3};",
            ";mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "prefix\\0Amma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "\"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "{mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "$L:mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "/* comment */mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "@p mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "@!%p\\09mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        ] {
            assert!(
                contains_mma_m16n8k16_f32_bf16_features(spelling),
                "missed {spelling:?}"
            );
        }

        let requirements = detect_module_requirements_in_llvm_text(mnemonic);
        assert_eq!(
            requirements,
            ModuleRequirements {
                features: DetectedFeatures::Sm80,
                ptx_isa: PtxIsaRequirement::Ptx70,
            }
        );
        assert_eq!(select_target(requirements.features).unwrap(), "sm_80");

        let lower_target =
            resolve_ptx_target(Some("sm_75"), None, requirements.features).unwrap_err();
        assert!(
            lower_target
                .to_string()
                .contains("cannot lower detected feature Sm80"),
            "{lower_target}"
        );
        let (target, _) = resolve_ptx_target(Some("sm_80"), None, requirements.features).unwrap();
        assert_eq!(target, "sm_80");

        for near_miss in [
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "mma.sp.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32x {$0}, {$1}, {$2}, {$3};",
            "not_mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "$mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "%mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "@mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "!mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "@!mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "not$mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "/mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            ")mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        ] {
            assert!(
                !contains_mma_m16n8k16_f32_bf16_features(near_miss),
                "matched {near_miss:?}"
            );
        }

        let combined = format!(
            "{mnemonic}\n{}",
            "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;"
        );
        assert_eq!(
            detect_module_requirements_in_llvm_text(&combined),
            ModuleRequirements {
                features: DetectedFeatures::Sm80 | DetectedFeatures::Movmatrix,
                ptx_isa: PtxIsaRequirement::Ptx78,
            }
        );
    }

    #[test]
    fn test_movmatrix_detection_separates_sm75_from_the_ptx78_floor() {
        let mnemonic = "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;";
        for spelling in [
            mnemonic,
            "movmatrix.sync.aligned.m8n8.trans.b16\t$0, $1;",
            "movmatrix.sync.aligned.m8n8.trans.b16\n$0, $1;",
            "movmatrix.sync.aligned.m8n8.trans.b16\\09$0, $1;",
            "movmatrix.sync.aligned.m8n8.trans.b16\\0A$0, $1;",
            "movmatrix.sync.aligned.m8n8.trans.b16\\0D\\0A$0, $1;",
        ] {
            assert!(contains_movmatrix_features(spelling), "{spelling:?}");
        }
        assert_eq!(
            detect_features_in_llvm_text(mnemonic),
            DetectedFeatures::Movmatrix
        );
        assert_eq!(select_target(DetectedFeatures::Movmatrix).unwrap(), "sm_75");
        assert_eq!(
            detect_module_requirements_in_llvm_text(mnemonic).ptx_isa,
            PtxIsaRequirement::Ptx78
        );

        for near_miss in [
            "movmatrix.sync.aligned.m8n8.b16 $0, $1;",
            "movmatrix.sync.aligned.m16n8.trans.b16 $0, $1;",
            "movmatrix.sync.aligned.m8n8.trans.b32 $0, $1;",
            "movmatrix.sync.aligned.m8n8.trans.b16x2 $0, $1;",
            "movmatrix.sync.aligned.m8n8.trans.b16\\5C09$0, $1;",
        ] {
            assert!(
                !contains_movmatrix_features(near_miss),
                "matched {near_miss}"
            );
            assert_eq!(
                detect_module_requirements_in_llvm_text(near_miss),
                ModuleRequirements {
                    features: DetectedFeatures::Basic,
                    ptx_isa: PtxIsaRequirement::Default,
                }
            );
        }

        let combined = format!("{mnemonic}\ncp.async.ca.shared.global [$0], [$1], 4;");
        assert_eq!(
            detect_module_requirements_in_llvm_text(&combined),
            ModuleRequirements {
                features: DetectedFeatures::Sm80 | DetectedFeatures::Movmatrix,
                ptx_isa: PtxIsaRequirement::Ptx78,
            },
            "the architecture and PTX ISA floors must compose independently"
        );

        let sm_70: CudaArch = "sm_70".parse().unwrap();
        let sm_75: CudaArch = "sm_75".parse().unwrap();
        let sm_80: CudaArch = "sm_80".parse().unwrap();
        assert!(validate_target_features(&sm_70, DetectedFeatures::Movmatrix).is_err());
        assert!(validate_target_features(&sm_75, DetectedFeatures::Movmatrix).is_ok());
        assert!(validate_target_features(&sm_80, DetectedFeatures::Movmatrix).is_ok());

        for target in ["sm_75", "sm_80", "sm_86", "sm_87"] {
            assert_eq!(
                required_ptx_feature(target, PtxIsaRequirement::Ptx78),
                Some("+ptx78"),
                "{target} needs an explicit PTX 7.8 floor"
            );
        }
        assert_eq!(
            required_ptx_feature("sm_90", PtxIsaRequirement::Ptx78),
            None
        );
        for target in ["sm_88", "sm_89"] {
            assert_eq!(
                required_ptx_feature(target, PtxIsaRequirement::Ptx78),
                None,
                "{target} already requires PTX 7.8 or newer"
            );
        }
        assert_eq!(
            required_ptx_feature("sm_75", PtxIsaRequirement::Default),
            None
        );
    }

    #[test]
    fn matrix_memory_detection_composes_architecture_and_ptx_isa_floors() {
        let base_ldmatrix = "ldmatrix.sync.aligned.m8n8.x4.b16 {$0, $1, $2, $3}, [$4];";
        assert_eq!(
            detect_module_requirements_in_llvm_text(base_ldmatrix),
            ModuleRequirements {
                features: DetectedFeatures::Ldmatrix,
                ptx_isa: PtxIsaRequirement::Ptx65,
            }
        );

        let cta_ldmatrix = "ldmatrix.sync.aligned.m8n8.x1.shared::cta.b16 {$0}, [$1];";
        assert_eq!(
            detect_module_requirements_in_llvm_text(cta_ldmatrix),
            ModuleRequirements {
                features: DetectedFeatures::Ldmatrix,
                ptx_isa: PtxIsaRequirement::Ptx78,
            }
        );

        for stmatrix in [
            "stmatrix.sync.aligned.m8n8.x1.b16 [$0], {$1};",
            "stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 [$0], {$1, $2, $3, $4};",
        ] {
            assert_eq!(
                detect_module_requirements_in_llvm_text(stmatrix),
                ModuleRequirements {
                    features: DetectedFeatures::Sm90,
                    ptx_isa: PtxIsaRequirement::Ptx78,
                }
            );
        }

        for newer in [
            "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {$0, $1}, [$2];",
            "ldmatrix.sync.aligned.m8n16.x2.shared::cta.b8x16.b6x16_p32 {$0, $1}, [$2];",
            "stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [$0], {$1};",
        ] {
            assert_eq!(
                detect_module_requirements_in_llvm_text(newer),
                ModuleRequirements {
                    features: DetectedFeatures::MatrixBlackwell
                        | if newer.starts_with("ldmatrix") {
                            DetectedFeatures::Ldmatrix
                        } else {
                            DetectedFeatures::Sm90
                        },
                    ptx_isa: PtxIsaRequirement::Ptx86,
                },
                "{newer}"
            );
        }

        let mixed = format!(
            "{base_ldmatrix}\n{}",
            "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;"
        );
        assert_eq!(
            detect_module_requirements_in_llvm_text(&mixed),
            ModuleRequirements {
                features: DetectedFeatures::Movmatrix | DetectedFeatures::Ldmatrix,
                ptx_isa: PtxIsaRequirement::Ptx78,
            },
            "the strongest PTX ISA floor must survive equal sm_75 feature families"
        );

        assert_eq!(
            required_ptx_feature("sm_75", PtxIsaRequirement::Ptx65),
            Some("+ptx65")
        );
        assert_eq!(
            required_ptx_feature("sm_80", PtxIsaRequirement::Ptx65),
            None
        );
        assert_eq!(
            required_ptx_feature("sm_100a", PtxIsaRequirement::Ptx86),
            None
        );

        let adjacent_unrelated_b8 = concat!(
            "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {$0}, [$1]; ",
            "mov.b8 $2, $3;"
        );
        assert_eq!(
            detect_module_requirements_in_llvm_text(adjacent_unrelated_b8),
            ModuleRequirements {
                features: DetectedFeatures::Ldmatrix,
                ptx_isa: PtxIsaRequirement::Ptx65,
            },
            "an unrelated b8 instruction must not raise the ldmatrix family"
        );
    }

    #[test]
    fn tma_and_wgmma_raise_their_independent_ptx_floors() {
        for tma in [
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes;",
            "cp.async.bulk.commit_group;",
            "cp.async.bulk.wait_group 0;",
            "cp.async.bulk.wait_group.read 0;",
        ] {
            let requirements = detect_module_requirements_in_llvm_text(tma);
            assert!(
                requirements.features.contains(DetectedFeatures::Tma),
                "{tma}"
            );
            assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx80, "{tma}");
        }

        let non_bulk = "cp.async.commit_group;";
        assert_eq!(
            detect_module_requirements_in_llvm_text(non_bulk),
            ModuleRequirements {
                features: DetectedFeatures::Sm80,
                ptx_isa: PtxIsaRequirement::Default,
            }
        );

        let tma_and_movmatrix = concat!(
            "cp.async.bulk.commit_group; ",
            "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;"
        );
        assert_eq!(
            detect_module_requirements_in_llvm_text(tma_and_movmatrix).ptx_isa,
            PtxIsaRequirement::Ptx80
        );

        let wgmma = "wgmma.fence.sync.aligned;";
        assert_eq!(
            detect_module_requirements_in_llvm_text(wgmma),
            ModuleRequirements {
                features: DetectedFeatures::Wgmma,
                ptx_isa: PtxIsaRequirement::Ptx80,
            }
        );

        let shared_cta =
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes;";
        assert!(contains_tma_shared_cta_destination(shared_cta));
        let shared_cta_requirements = detect_module_requirements_in_llvm_text(shared_cta);
        assert_eq!(shared_cta_requirements.features, DetectedFeatures::Tma);
        assert_eq!(shared_cta_requirements.ptx_isa, PtxIsaRequirement::Ptx86);

        let shared_source = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group;";
        assert!(!contains_tma_shared_cta_destination(shared_source));
        assert_eq!(
            detect_module_requirements_in_llvm_text(shared_source).ptx_isa,
            PtxIsaRequirement::Ptx80
        );

        let cta_group = "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1;";
        assert_eq!(
            detect_module_requirements_in_llvm_text(cta_group).ptx_isa,
            PtxIsaRequirement::Ptx86
        );

        assert_eq!(
            required_ptx_feature("sm_90", PtxIsaRequirement::Ptx80),
            Some("+ptx80")
        );
        assert_eq!(
            required_ptx_feature("sm_90a", PtxIsaRequirement::Ptx86),
            Some("+ptx86")
        );
        assert_eq!(
            required_ptx_feature("sm_100a", PtxIsaRequirement::Ptx80),
            None
        );
    }

    #[test]
    fn related_cluster_mbarrier_and_clc_requirements_are_detected() {
        for ptx in [
            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$0];",
            "fence.mbarrier_init.release.cluster;",
        ] {
            let requirements = detect_module_requirements_in_llvm_text(ptx);
            assert!(
                requirements.features.contains(DetectedFeatures::Tma),
                "{ptx}"
            );
            assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx80, "{ptx}");
            assert!(arch_satisfies("sm_90", requirements.features));
        }

        for (ptx, expected_isa) in [
            (
                "mbarrier.init.shared.b64 [$0], 1;",
                PtxIsaRequirement::Ptx70,
            ),
            (
                "mbarrier.test_wait.parity.shared.b64 $0, [$1], $2;",
                PtxIsaRequirement::Ptx71,
            ),
            (
                "mbarrier.try_wait.parity.shared::cta.b64 $0, [$1], $2;",
                PtxIsaRequirement::Ptx78,
            ),
        ] {
            let requirements = detect_module_requirements_in_llvm_text(ptx);
            assert!(
                requirements.features.contains(DetectedFeatures::Sm80),
                "{ptx}"
            );
            assert_eq!(requirements.ptx_isa, expected_isa, "{ptx}");
            if ptx.contains("try_wait") {
                assert!(requirements.features.contains(DetectedFeatures::Tma));
                assert!(!arch_satisfies("sm_80", requirements.features));
            } else {
                assert!(arch_satisfies("sm_80", requirements.features));
                assert!(!arch_satisfies("sm_75", requirements.features));
            }
        }

        for ptx in [
            "redux.sync.add.u32 $0, $1, $2;",
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "cvt.rn.relu.bf16x2.f32 $0, $1, $2;",
            "cvt.rz.bf16x2.f32 $0, $1, $2;",
        ] {
            assert!(
                detect_features_in_llvm_text(ptx).contains(DetectedFeatures::Sm80),
                "{ptx}"
            );
        }
        assert_eq!(
            required_ptx_feature("sm_80", PtxIsaRequirement::Ptx70),
            None
        );
        assert_eq!(
            required_ptx_feature("sm_80", PtxIsaRequirement::Ptx71),
            Some("+ptx71")
        );
        for target in ["sm_86", "sm_87", "sm_88", "sm_89"] {
            assert_eq!(
                required_ptx_feature(target, PtxIsaRequirement::Ptx71),
                None,
                "{target} cannot be downgraded below its minimum PTX ISA"
            );
        }

        for ptx in [
            "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64 $0, [$1], $2;",
            "fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;",
            "fence.acquire.sync_restrict::shared::cluster.cluster;",
        ] {
            let requirements = detect_module_requirements_in_llvm_text(ptx);
            assert!(
                requirements.features.contains(DetectedFeatures::Tma),
                "{ptx}"
            );
            assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx86, "{ptx}");
            assert!(!arch_satisfies("sm_80", requirements.features));
        }

        for ptx in [
            "mbarrier.test_wait.acquire.cta.shared::cta.b64 $0, [$1], $2;",
            "mbarrier.arrive.release.cta.shared::cta.b64 $0, [$1];",
        ] {
            let requirements = detect_module_requirements_in_llvm_text(ptx);
            assert!(requirements.features.contains(DetectedFeatures::Tma));
            assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx80);
            assert!(!arch_satisfies("sm_80", requirements.features));
        }

        let cluster_sync = "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;";
        assert_eq!(
            detect_module_requirements_in_llvm_text(cluster_sync),
            ModuleRequirements {
                features: DetectedFeatures::Cluster,
                ptx_isa: PtxIsaRequirement::Ptx78,
            }
        );
        assert_eq!(select_target(DetectedFeatures::Cluster).unwrap(), "sm_90");

        let cluster_release = "barrier.cluster.arrive.release;";
        assert_eq!(
            detect_module_requirements_in_llvm_text(cluster_release).ptx_isa,
            PtxIsaRequirement::Ptx80
        );

        for ptx in [
            "fence.sc.cluster;",
            "fence.acq_rel.cluster;",
            "ld.shared::cluster.u32 $0, [$1];",
            "ld.acquire.cluster.global.u32 $0, [$1];",
            "getctarank.shared::cluster.u32 $0, $1;",
        ] {
            let requirements = detect_module_requirements_in_llvm_text(ptx);
            assert!(requirements.features.contains(DetectedFeatures::Cluster));
            assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx78);
            assert!(!arch_satisfies("sm_80", requirements.features));
        }

        for ptx in [
            "fence.acquire.cta;",
            "fence.release.gpu;",
            "fence.acquire.cluster;",
            "fence.release.sys;",
        ] {
            let requirements = detect_module_requirements_in_llvm_text(ptx);
            assert!(
                requirements.features.contains(DetectedFeatures::Sm90),
                "{ptx}"
            );
            assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx86, "{ptx}");
            assert_eq!(
                requirements.features.contains(DetectedFeatures::Cluster),
                ptx.contains(".cluster"),
                "{ptx}"
            );
            assert!(!arch_satisfies("sm_80", requirements.features));
        }

        let multimem = "multimem.red.relaxed.cluster.global.add.u32 [$0], $1;";
        let requirements = detect_module_requirements_in_llvm_text(multimem);
        assert_eq!(requirements.features, DetectedFeatures::Sm90);
        assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx86);
        assert_eq!(select_target(requirements.features).unwrap(), "sm_90");
        let multimem_debug_filename = r#"!9 = !DIFile(filename: "multimem.rs", directory: "/tmp")"#;
        assert_eq!(
            detect_module_requirements_in_llvm_text(multimem_debug_filename),
            ModuleRequirements {
                features: DetectedFeatures::Basic,
                ptx_isa: PtxIsaRequirement::Default,
            }
        );

        for multimem in [
            "multimem.ld_reduce.relaxed.cta.add.v4.e4m3 {$0, $1, $2, $3}, [$4];",
            "multimem.st.relaxed.gpu.e5m2 [$0], $1;",
            "multimem.ld_reduce.add.acc::f16.v4.e5m2 {$0, $1, $2, $3}, [$4];",
        ] {
            let requirements = detect_module_requirements_in_llvm_text(multimem);
            assert_eq!(
                requirements.features,
                DetectedFeatures::MultimemFp8 | DetectedFeatures::Sm90,
                "{multimem}"
            );
            assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx86, "{multimem}");
            assert_eq!(select_target(requirements.features).unwrap(), "sm_100a");
            for target in [
                "sm_100a", "sm_103a", "sm_110a", "sm_120a", "sm_121a", "sm_100f", "sm_103f",
                "sm_110f",
            ] {
                assert!(arch_satisfies(target, requirements.features), "{target}");
            }
            for target in ["sm_100", "sm_90a", "sm_120f", "sm_121f"] {
                assert!(!arch_satisfies(target, requirements.features), "{target}");
            }
        }

        let redux_f32 = "redux.sync.min.abs.NaN.f32 $0, $1, $2;";
        let requirements = detect_module_requirements_in_llvm_text(redux_f32);
        assert_eq!(
            requirements.features,
            DetectedFeatures::ReduxF32 | DetectedFeatures::Sm80
        );
        assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx86);
        assert_eq!(select_target(requirements.features).unwrap(), "sm_100a");
        for target in ["sm_100a", "sm_103a", "sm_100f", "sm_103f"] {
            assert!(arch_satisfies(target, requirements.features), "{target}");
        }
        for target in ["sm_100", "sm_110a", "sm_120a", "sm_121f"] {
            assert!(!arch_satisfies(target, requirements.features), "{target}");
        }

        for sreg in [
            "mov.u32 $0, %clusterid.x;",
            "mov.u32 $0, %nclusterid.z;",
            "mov.u32 $0, %cluster_ctarank;",
            "mov.u32 $0, %cluster_nctarank;",
            "mov.pred $0, %is_explicit_cluster;",
        ] {
            assert_eq!(
                detect_module_requirements_in_llvm_text(sreg),
                ModuleRequirements {
                    features: DetectedFeatures::Cluster,
                    ptx_isa: PtxIsaRequirement::Ptx78,
                },
                "{sreg}"
            );
        }

        let cluster_metadata = r#"!0 = !{!"cluster_dim_x", i32 2}
            !1 = !{!"cluster_dim_y", i32 1}
            !2 = !{!"cluster_dim_z", i32 1}"#;
        assert_eq!(
            detect_module_requirements_in_llvm_text(cluster_metadata),
            ModuleRequirements {
                features: DetectedFeatures::Cluster,
                ptx_isa: PtxIsaRequirement::Ptx78,
            }
        );
        let cluster_debug_local =
            r#"!8 = !DILocalVariable(name: "cluster_dim_x", scope: !1, file: !2, line: 3)"#;
        assert_eq!(
            detect_module_requirements_in_llvm_text(cluster_debug_local),
            ModuleRequirements {
                features: DetectedFeatures::Basic,
                ptx_isa: PtxIsaRequirement::Default,
            }
        );

        let elect = "elect.sync $0|p, $1;";
        assert_eq!(
            detect_module_requirements_in_llvm_text(elect),
            ModuleRequirements {
                features: DetectedFeatures::Sm90,
                ptx_isa: PtxIsaRequirement::Ptx80,
            }
        );

        let tcgen_wait = "tcgen05.wait::ld.sync.aligned;";
        assert_eq!(
            detect_module_requirements_in_llvm_text(tcgen_wait),
            ModuleRequirements {
                features: DetectedFeatures::Blackwell,
                ptx_isa: PtxIsaRequirement::Ptx86,
            }
        );

        let tcgen_debug_filename = r#"!7 = !DIFile(filename: "tcgen05.rs", directory: "/tmp")"#;
        assert_eq!(
            detect_module_requirements_in_llvm_text(tcgen_debug_filename),
            ModuleRequirements {
                features: DetectedFeatures::Basic,
                ptx_isa: PtxIsaRequirement::Default,
            }
        );

        let clc = "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 $0, $1;";
        assert_eq!(
            detect_module_requirements_in_llvm_text(clc),
            ModuleRequirements {
                features: DetectedFeatures::Sm100,
                ptx_isa: PtxIsaRequirement::Ptx86,
            }
        );
        assert_eq!(select_target(DetectedFeatures::Sm100).unwrap(), "sm_100");
        assert!(!arch_satisfies("sm_90", DetectedFeatures::Sm100));
        assert!(arch_satisfies("sm_120", DetectedFeatures::Sm100));

        let clc_multicast = "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [$0], [$1];";
        let requirements = detect_module_requirements_in_llvm_text(clc_multicast);
        assert_eq!(
            requirements.features,
            DetectedFeatures::Sm100 | DetectedFeatures::BlackwellFamily
        );
        assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx86);
        assert_eq!(select_target(requirements.features).unwrap(), "sm_100a");
        assert!(!arch_satisfies("sm_100", requirements.features));
        assert!(arch_satisfies("sm_120a", requirements.features));
        for arch in ["sm_100f", "sm_101f", "sm_110f", "sm_121f"] {
            assert!(arch_satisfies(arch, requirements.features), "{arch}");
        }
        for arch in ["sm_103a", "sm_121a"] {
            assert!(!arch_satisfies(arch, requirements.features), "{arch}");
        }
    }

    #[test]
    fn ptx86_tma_modes_enforce_their_architecture_families() {
        for ptx in [
            "cp.async.bulk.global.shared::cta.bulk_group.cp_mask [$0], [$1], 16, $2;",
            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes;",
            "cp.async.bulk.tensor.3d.shared::cta.global.im2col::w.mbarrier::complete_tx::bytes;",
        ] {
            let requirements = detect_module_requirements_in_llvm_text(ptx);
            assert!(
                requirements.features.contains(DetectedFeatures::Tma),
                "{ptx}"
            );
            assert!(
                requirements.features.contains(DetectedFeatures::Sm100),
                "{ptx}"
            );
            assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx86, "{ptx}");
            assert!(!arch_satisfies("sm_90", requirements.features));
            assert!(arch_satisfies("sm_100", requirements.features));
        }

        for ptx in [
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes;",
            "cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4.bulk_group;",
            "cp.async.bulk.tensor.3d.shared::cta.global.im2col::w::128.mbarrier::complete_tx::bytes;",
            "cp.async.bulk.prefetch.tensor.3d.L2.global.im2col::w::128;",
        ] {
            let requirements = detect_module_requirements_in_llvm_text(ptx);
            assert!(
                requirements.features.contains(DetectedFeatures::Tma),
                "{ptx}"
            );
            assert!(
                requirements
                    .features
                    .contains(DetectedFeatures::BlackwellAccelerated),
                "{ptx}"
            );
            assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx86, "{ptx}");
            assert_eq!(select_target(requirements.features).unwrap(), "sm_100a");
            assert!(!arch_satisfies("sm_100", requirements.features));
            assert!(!arch_satisfies("sm_120a", requirements.features));
            assert!(arch_satisfies("sm_103f", requirements.features));
        }

        assert!(!contains_tma_sm100_features("custom.op.cp_mask $0;"));
        assert!(!contains_tma_blackwell_accelerated_features(
            "custom.tile::scatter4 $0;"
        ));
    }

    #[test]
    fn test_sm90_floor_wins_when_sm80_features_are_also_present() {
        let llvm = r#"
            call i32 asm pure "add.rn.bf16x2 $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
            call void asm sideeffect "cp.async.ca.shared.global [%0], [%1], 4;", "l,l"()
        "#;

        assert!(contains_sm90_features(llvm));
        assert!(contains_sm80_features(llvm));
        assert_eq!(
            detect_features_in_llvm_text(llvm),
            DetectedFeatures::Sm90 | DetectedFeatures::Sm80
        );
    }

    #[test]
    fn test_tma_multicast_detection_requires_cta_mask() {
        let multicast = "call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile(i32 0, i1 1, i1 false)";
        let unicast = "call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile(i32 0, i1 0, i1 false)";
        let literal_multicast = "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster";
        let cg1 = "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1";
        let cg2 = "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2";
        let cg1_intrinsic = "call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %dst, i1 0, i1 false, i32 1)";
        let cg2_intrinsic = "call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %dst, i1 1, i1 false, i32 2)";
        let unrelated_i32 = "call void @unrelated(i32 2)";

        assert!(contains_tma_multicast(multicast));
        assert!(contains_tma_multicast(literal_multicast));
        assert!(!contains_tma_multicast(unicast));
        assert_eq!(
            detect_features_in_llvm_text(multicast),
            DetectedFeatures::TmaMulticast | DetectedFeatures::Tma
        );
        assert_eq!(
            detect_features_in_llvm_text(literal_multicast),
            DetectedFeatures::TmaMulticast | DetectedFeatures::Tma | DetectedFeatures::Cluster
        );
        assert_eq!(detect_features_in_llvm_text(unicast), DetectedFeatures::Tma);
        assert_eq!(
            detect_features_in_llvm_text(cg1),
            DetectedFeatures::TmaCtaGroup | DetectedFeatures::Tma
        );
        assert_eq!(
            detect_features_in_llvm_text(cg1_intrinsic),
            DetectedFeatures::TmaCtaGroup | DetectedFeatures::Tma
        );
        assert_eq!(
            detect_features_in_llvm_text(cg2),
            DetectedFeatures::TmaCtaGroup
                | DetectedFeatures::TmaMulticast
                | DetectedFeatures::Tma
                | DetectedFeatures::Cluster
        );
        assert_eq!(
            detect_features_in_llvm_text(cg2_intrinsic),
            DetectedFeatures::TmaCtaGroup | DetectedFeatures::TmaMulticast | DetectedFeatures::Tma
        );
        assert!(!contains_tma_cta_group_features(unrelated_i32));
    }

    #[test]
    fn test_select_target_prefers_required_architecture() {
        for (features, expected) in [
            (DetectedFeatures::Blackwell, "sm_100a"),
            (DetectedFeatures::TmaCtaGroup, "sm_100a"),
            (DetectedFeatures::BlackwellAccelerated, "sm_100a"),
            (DetectedFeatures::BlackwellFamily, "sm_100a"),
            (DetectedFeatures::ReduxF32, "sm_100a"),
            (DetectedFeatures::MultimemFp8, "sm_100a"),
            (DetectedFeatures::TmaMulticast, "sm_100a"),
            (DetectedFeatures::MatrixBlackwell, "sm_100a"),
            (DetectedFeatures::Wgmma, "sm_90a"),
            (DetectedFeatures::Sm100, "sm_100"),
            (DetectedFeatures::Tma, "sm_100"),
            (DetectedFeatures::Cluster, "sm_90"),
            (DetectedFeatures::Sm90, "sm_90"),
            (DetectedFeatures::Sm80, "sm_80"),
            (DetectedFeatures::Movmatrix, "sm_75"),
            (DetectedFeatures::Ldmatrix, "sm_75"),
            (DetectedFeatures::Basic, "sm_80"),
        ] {
            assert_eq!(select_target(features).unwrap(), expected, "{features:?}");
        }
    }

    #[test]
    fn target_selection_enforces_feature_intersections() {
        let multicast = "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster";
        let hopper_pair = format!("{multicast};\nwgmma.fence.sync.aligned;");
        let hopper_requirements = detect_features_in_llvm_text(&hopper_pair);
        assert!(hopper_requirements.contains(DetectedFeatures::TmaMulticast));
        assert!(hopper_requirements.contains(DetectedFeatures::Wgmma));
        assert_eq!(select_target(hopper_requirements).unwrap(), "sm_90a");
        assert!(arch_satisfies("sm_90a", hopper_requirements));
        assert!(!arch_satisfies("sm_100a", hopper_requirements));

        let blackwell_pair = format!(
            "{multicast};\n{}",
            "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {$0, $1}, [$2];"
        );
        let blackwell_requirements = detect_features_in_llvm_text(&blackwell_pair);
        assert!(blackwell_requirements.contains(DetectedFeatures::TmaMulticast));
        assert!(blackwell_requirements.contains(DetectedFeatures::MatrixBlackwell));
        assert_eq!(select_target(blackwell_requirements).unwrap(), "sm_100a");
        assert!(arch_satisfies("sm_100a", blackwell_requirements));
        assert!(!arch_satisfies("sm_90a", blackwell_requirements));

        let impossible = DetectedFeatures::Wgmma | DetectedFeatures::MatrixBlackwell;
        let error = select_target(impossible).expect_err("families have no common target");
        assert!(error.contains("do not share a compatible GPU architecture"));
        assert!(resolve_ptx_target(Some("sm_90a"), None, impossible).is_err());
        assert!(resolve_ptx_target(Some("sm_100a"), None, impossible).is_err());
    }

    #[test]
    fn test_arch_major_parses_cuda_spelling() {
        assert_eq!(arch_compute_capability("sm_75"), Some(75));
        assert_eq!(arch_compute_capability("sm_100a"), Some(100));
        assert_eq!(arch_major("sm_75"), Some(7));
        assert_eq!(arch_major("sm_80"), Some(8));
        assert_eq!(arch_major("sm_90a"), Some(9));
        assert_eq!(arch_major("sm_100a"), Some(10));
        assert_eq!(arch_major("sm_103a"), Some(10));
        assert_eq!(arch_major("sm_120a"), Some(12));
        assert_eq!(arch_major("nvvm-ir"), None);
        assert_eq!(arch_major("sm_"), None);
    }

    #[test]
    fn ptx9_targets_require_an_llvm22_backend() {
        for target in ["sm_88", "sm_110", "sm_110a", "sm_110f"] {
            assert!(
                validate_target_for_llvm_major(target, Some(21)).is_err(),
                "{target}"
            );
            assert!(
                validate_target_for_llvm_major(target, None).is_err(),
                "{target}"
            );
            assert!(
                validate_target_for_llvm_major(target, Some(22)).is_ok(),
                "{target}"
            );
            assert!(
                validate_target_for_llvm_major(target, Some(23)).is_ok(),
                "{target}"
            );
        }
        for target in ["sm_87", "sm_103a", "sm_120a", "sm_121f"] {
            assert!(
                validate_target_for_llvm_major(target, Some(21)).is_ok(),
                "{target}"
            );
        }
        assert_eq!(target_minimum_ptx_isa(121), Some(88));
    }

    #[test]
    fn test_arch_satisfies_sm100_only_features() {
        // tcgen05 and explicit cta_group TMA are datacenter-Blackwell only:
        // consumer Blackwell (sm_120) and Hopper (sm_90) cannot run them, even
        // though 120 > 100. This is the gemm_sol regression guard.
        for f in [DetectedFeatures::Blackwell, DetectedFeatures::TmaCtaGroup] {
            assert!(arch_satisfies("sm_100a", f), "sm_100a must satisfy {f:?}");
            assert!(arch_satisfies("sm_103a", f), "sm_103a must satisfy {f:?}");
            assert!(arch_satisfies("sm_103f", f), "sm_103f must satisfy {f:?}");
            assert!(
                !arch_satisfies("sm_100", f),
                "generic sm_100 must NOT satisfy {f:?}"
            );
            assert!(
                !arch_satisfies("sm_120a", f),
                "sm_120a must NOT satisfy {f:?}"
            );
            assert!(
                !arch_satisfies("sm_90a", f),
                "sm_90a must NOT satisfy {f:?}"
            );
            assert!(
                !arch_satisfies("sm_102a", f),
                "unknown architecture-specific targets must not be accepted"
            );
            assert!(
                !arch_satisfies("sm_102f", f),
                "unknown family-specific targets must not be accepted"
            );
        }
    }

    #[test]
    fn test_arch_satisfies_base_tma_multicast_targets() {
        for arch in [
            "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_103f", "sm_110a", "sm_120", "sm_120a",
        ] {
            assert!(
                arch_satisfies(arch, DetectedFeatures::TmaMulticast),
                "{arch}"
            );
        }
        for arch in ["sm_80", "sm_89", "sm_102a", "sm_102f"] {
            assert!(
                !arch_satisfies(arch, DetectedFeatures::TmaMulticast),
                "{arch}"
            );
        }
    }

    #[test]
    fn test_arch_satisfies_wgmma_is_hopper_only() {
        assert!(arch_satisfies("sm_90a", DetectedFeatures::Wgmma));
        assert!(!arch_satisfies("sm_90", DetectedFeatures::Wgmma));
        assert!(!arch_satisfies("sm_100a", DetectedFeatures::Wgmma));
        assert!(!arch_satisfies("sm_120a", DetectedFeatures::Wgmma));
    }

    #[test]
    fn test_arch_satisfies_blackwell_matrix_family_targets() {
        for arch in [
            "sm_100a", "sm_101a", "sm_110a", "sm_120a", "sm_100f", "sm_103f", "sm_120f", "sm_121f",
        ] {
            assert!(
                arch_satisfies(arch, DetectedFeatures::MatrixBlackwell),
                "{arch}"
            );
            assert!(
                arch_satisfies(arch, DetectedFeatures::BlackwellFamily),
                "{arch}"
            );
        }
        for arch in [
            "sm_100a", "sm_101a", "sm_103a", "sm_110a", "sm_100f", "sm_103f", "sm_110f",
        ] {
            assert!(
                arch_satisfies(arch, DetectedFeatures::BlackwellAccelerated),
                "{arch}"
            );
        }
        for arch in ["sm_100", "sm_120a", "sm_120f", "sm_102f"] {
            assert!(
                !arch_satisfies(arch, DetectedFeatures::BlackwellAccelerated),
                "{arch}"
            );
        }
        for arch in ["sm_100", "sm_103", "sm_110", "sm_120", "sm_121a"] {
            assert!(arch_satisfies(arch, DetectedFeatures::Sm100), "{arch}");
        }
        for arch in ["sm_90a", "sm_102", "sm_102a"] {
            assert!(!arch_satisfies(arch, DetectedFeatures::Sm100), "{arch}");
        }
        for arch in ["sm_90a", "sm_100", "sm_102f", "sm_120"] {
            assert!(
                !arch_satisfies(arch, DetectedFeatures::MatrixBlackwell),
                "{arch}"
            );
            assert!(
                !arch_satisfies(arch, DetectedFeatures::BlackwellFamily),
                "{arch}"
            );
        }
    }

    #[test]
    fn test_arch_satisfies_forward_compatible_features() {
        // Plain TMA / cluster / sm_90-floor instructions lower on any sm_90+
        // device, sm_80-floor instructions on any sm_80+ device, movmatrix and
        // base ldmatrix on sm_75+, and basic kernels on Volta+.
        // So a consumer sm_120 GPU is a valid target for these (it runs locally
        // instead of being downgraded to the feature floor).
        for arch in ["sm_90a", "sm_100a", "sm_120a"] {
            assert!(arch_satisfies(arch, DetectedFeatures::Tma));
            assert!(arch_satisfies(arch, DetectedFeatures::Cluster));
            assert!(arch_satisfies(arch, DetectedFeatures::Sm90));
            assert!(arch_satisfies(arch, DetectedFeatures::Sm80));
            assert!(arch_satisfies(arch, DetectedFeatures::Movmatrix));
            assert!(arch_satisfies(arch, DetectedFeatures::Ldmatrix));
            assert!(arch_satisfies(arch, DetectedFeatures::Basic));
        }
        assert!(arch_satisfies("sm_80", DetectedFeatures::Sm80));
        assert!(!arch_satisfies("sm_75", DetectedFeatures::Sm80));
        assert!(arch_satisfies("sm_75", DetectedFeatures::Movmatrix));
        assert!(arch_satisfies("sm_80", DetectedFeatures::Movmatrix));
        assert!(!arch_satisfies("sm_70", DetectedFeatures::Movmatrix));
        assert!(arch_satisfies("sm_75", DetectedFeatures::Ldmatrix));
        assert!(!arch_satisfies("sm_70", DetectedFeatures::Ldmatrix));
        assert!(arch_satisfies("sm_80", DetectedFeatures::Basic));
        assert!(arch_satisfies("sm_75", DetectedFeatures::Basic));
        assert!(arch_satisfies("sm_70", DetectedFeatures::Basic));
        assert!(!arch_satisfies("sm_80", DetectedFeatures::Tma));
        assert!(!arch_satisfies("sm_80", DetectedFeatures::Sm90));
        assert!(!arch_satisfies("sm_80a", DetectedFeatures::Basic));
        assert!(!arch_satisfies("sm_90f", DetectedFeatures::Tma));
    }

    /// Build a minimal LLVM dialect module containing a single function
    /// declaration named `name`. The module is intentionally empty otherwise;
    /// the auto-detect logic only inspects the symbol name on declarations
    /// and on direct call sites.
    fn build_module_with_func_decl(ctx: &mut Context, name: &str) -> Ptr<Operation> {
        use llvm_export::ops::FuncOp as LlvmFuncOp;
        use llvm_export::types::FuncType as LlvmFuncType;
        use pliron::basic_block::BasicBlock;
        use pliron::builtin::ops::ModuleOp;
        use pliron::builtin::types::{IntegerType, Signedness};

        let module = ModuleOp::new(ctx, "test_module".try_into().unwrap());
        let module_ptr = module.get_operation();
        let module_region = module_ptr.deref(ctx).get_region(0);

        let module_block = {
            let region_ref = module_region.deref(ctx);
            if let Some(first_block) = region_ref.iter(ctx).next() {
                first_block
            } else {
                drop(region_ref);
                let new_block = BasicBlock::new(ctx, None, vec![]);
                new_block.insert_at_back(module_region, ctx);
                new_block
            }
        };

        let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
        let func_ty = LlvmFuncType::get(ctx, i32_ty.into(), vec![i32_ty.into()], false);
        let func = LlvmFuncOp::new(ctx, name.try_into().unwrap(), func_ty);
        func.get_operation().insert_at_back(module_block, ctx);

        module_ptr
    }

    #[test]
    fn test_module_uses_libdevice_detects_nv_func_decl() {
        let mut ctx = Context::new();
        let module_ptr = build_module_with_func_decl(&mut ctx, "__nv_sqrtf");
        assert!(
            module_uses_libdevice(&ctx, module_ptr),
            "module containing `__nv_*` function declaration must be flagged"
        );
    }

    #[test]
    fn in_memory_llvm_preview_uses_the_shared_feature_detector() {
        let mut ctx = Context::new();
        let module_ptr = build_module_with_func_decl(&mut ctx, "llvm_nvvm_tcgen05_alloc");

        let preview = render_llvm_ir(&ctx, module_ptr, &[], false, None, DebugKind::Off).unwrap();

        assert!(preview.contains("@llvm.nvvm.tcgen05.alloc"), "{preview}");
        assert_eq!(
            detect_features_in_llvm_text(&preview),
            DetectedFeatures::Blackwell
        );
    }

    #[test]
    fn test_module_uses_libdevice_ignores_unrelated_funcs() {
        let mut ctx = Context::new();
        let module_ptr = build_module_with_func_decl(&mut ctx, "kernel_main");
        assert!(
            !module_uses_libdevice(&ctx, module_ptr),
            "module without any `__nv_*` symbols must not be flagged"
        );
    }

    #[test]
    fn test_module_uses_libdevice_does_not_match_partial_prefix() {
        // "__nvm_foo" starts with "__nv" but not "__nv_". The detection rule
        // is the full `__nv_` prefix, so this must not trigger auto-detect.
        let mut ctx = Context::new();
        let module_ptr = build_module_with_func_decl(&mut ctx, "__nvm_foo");
        assert!(
            !module_uses_libdevice(&ctx, module_ptr),
            "names starting with `__nv` but not `__nv_` must not be flagged"
        );
    }

    /// `module_uses_libdevice` must also fire when the libdevice symbol
    /// appears as the callee of a direct `CallOp` -- this is the realistic
    /// case where a normal kernel calls `__nv_sqrtf`. The auto-detect
    /// recursion has to walk through the module region and visit the
    /// `CallOp` even when no enclosing `FuncOp` matches the prefix rule.
    #[test]
    fn test_module_uses_libdevice_detects_direct_nv_call() {
        use llvm_export::ops::CallOp as LlvmCallOp;
        use llvm_export::types::FuncType as LlvmFuncType;
        use pliron::basic_block::BasicBlock;
        use pliron::builtin::ops::ModuleOp;
        use pliron::builtin::types::{IntegerType, Signedness};

        let mut ctx = Context::new();

        let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
        let module_ptr = module.get_operation();
        let module_region = module_ptr.deref(&ctx).get_region(0);
        let module_block = BasicBlock::new(&mut ctx, None, vec![]);
        module_block.insert_at_back(module_region, &ctx);

        let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
        let callee_ty = LlvmFuncType::get(&ctx, i32_ty.into(), vec![], false);
        let callee_ident: pliron::identifier::Identifier = "__nv_sqrtf".try_into().unwrap();
        let nv_call = LlvmCallOp::new(
            &mut ctx,
            CallOpCallable::Direct(callee_ident),
            callee_ty,
            vec![],
        );
        nv_call.get_operation().insert_at_back(module_block, &ctx);

        assert!(
            module_uses_libdevice(&ctx, module_ptr),
            "direct call to a `__nv_*` symbol must be detected"
        );
    }
}
