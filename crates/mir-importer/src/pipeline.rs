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
//! | Feature           | Target    | Architecture         |
//! |-------------------|-----------|----------------------|
//! | tcgen05/TMEM      | sm_100a   | Blackwell datacenter |
//! | TMA multicast     | sm_100a   | Blackwell datacenter |
//! | WGMMA             | sm_90a    | Hopper only          |
//! | TMA/mbarrier      | sm_100    | Hopper+ compatible   |
//! | Basic CUDA        | sm_80     | Ampere+ (max compat) |
//!
//! Override with `CUDA_OXIDE_TARGET=<target>` environment variable.

use llvm_export::export::{DebugKind, ExportBackendConfig};
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

    /// Function parameter types (LLVM type strings like "float", "ptr", "i32").
    pub param_types: Vec<String>,

    /// Return type (LLVM type string like "float", "void", "i32").
    pub return_type: String,

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
    /// Currently supports NVVM 20 dialect (Blackwell+). Architecture is
    /// controlled by `--arch` flag.
    pub emit_nvvm_ir: bool,
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

    // Step 5: Lower dialect-mir → LLVM dialect.
    if config.verbose {
        eprintln!("\n=== Lowering dialect-mir → LLVM dialect ===");
    }
    lower_to_llvm(&mut ctx, module_op_ptr)?;

    // Step 5.5: Add device extern declarations to the LLVM dialect module.
    // These are needed before verification so calls to extern functions are valid.
    if !device_externs.is_empty() {
        if config.verbose {
            eprintln!(
                "\n=== Adding {} device extern declarations ===",
                device_externs.len()
            );
        }
        add_device_extern_declarations(&mut ctx, module_op_ptr, device_externs)?;
    }

    // Step 6: Verify the LLVM dialect module. Dump BEFORE verify so
    // verification failures still surface the IR to the user.
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

    // Step 7: Export to LLVM IR
    if config.verbose {
        let mode = if emit_nvvm_ir { "NVVM IR" } else { "PTX" };
        eprintln!("\n=== Exporting to LLVM IR ({} mode) ===", mode);
    }
    let ll_path = config.output_dir.join(format!("{}.ll", config.output_name));
    let _llvm_ir = export_llvm_ir(
        &ctx,
        module_op_ptr,
        device_externs,
        &ll_path,
        emit_nvvm_ir,
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
        // Record the real GPU arch in the bundle target when the caller
        // pinned one via CUDA_OXIDE_TARGET; otherwise leave the legacy
        // "nvvm-ir" sentinel that cuda-host's loader knows to re-resolve.
        let target = std::env::var("CUDA_OXIDE_TARGET").unwrap_or_else(|_| "nvvm-ir".to_string());
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
/// Idempotent with respect to the lowering step that runs before it: a
/// symbol may already have been declared at a call site (mir-lower declares
/// `__nv_*` libdevice callees on demand). Inserting a second `FuncOp` for
/// the same symbol would fail module verification with a multiple-definition
/// error, so symbols that already exist in the module are skipped.
fn add_device_extern_declarations(
    ctx: &mut Context,
    module_op_ptr: Ptr<Operation>,
    device_externs: &[DeviceExternDecl],
) -> Result<(), PipelineError> {
    use llvm_export::ops::FuncOp;
    use llvm_export::types::{FuncType, VoidType};
    use pliron::identifier::Identifier;
    use std::collections::HashSet;

    // Get the module's block pointer first (this is a Ptr, not a Ref, so no borrow issues)
    let block = {
        let region = module_op_ptr.deref(ctx).get_region(0).deref(ctx);
        region.iter(ctx).next().expect("Module should have a block")
    };

    let declared_symbols: HashSet<String> = block
        .deref(ctx)
        .iter(ctx)
        .filter_map(|op| {
            Operation::get_op::<FuncOp>(op, ctx).map(|f| f.get_symbol_name(ctx).to_string())
        })
        .collect();

    for decl in device_externs {
        if declared_symbols.contains(&decl.export_name) {
            continue;
        }

        // Parse parameter types from strings
        let param_types: Vec<_> = decl
            .param_types
            .iter()
            .map(|t| llvm_type_string_to_pliron(ctx, t))
            .collect();

        // Parse return type - for void, use VoidType
        let return_type = if decl.return_type == "void" {
            VoidType::get(ctx).into()
        } else {
            llvm_type_string_to_pliron(ctx, &decl.return_type)
        };

        // Create function type (result, args, is_variadic)
        let func_type = FuncType::get(ctx, return_type, param_types, false);

        // Use the original export name (NOT the prefixed name).
        // The MIR sees calls to `cuda_oxide_device_extern_<hash>_foo`, but
        // mir-lower/convert/ops/call.rs strips the reserved prefix via
        // `reserved_oxide_symbols::device_extern_base_name`, so the LLVM IR
        // emits `call @foo(...)`. For that to resolve, we declare `@foo` here.
        let func_ident: Identifier = decl
            .export_name
            .clone()
            .try_into()
            .unwrap_or_else(|_| "extern_func".try_into().unwrap());

        // Create function declaration (no body = declaration)
        let func_op = FuncOp::new(ctx, func_ident, func_type);

        // Insert at the front of the module (declarations come before definitions)
        func_op.get_operation().insert_at_front(block, ctx);
    }

    Ok(())
}

/// Convert LLVM type string to pliron type.
fn llvm_type_string_to_pliron(ctx: &mut Context, type_str: &str) -> pliron::r#type::TypeHandle {
    use llvm_export::types::PointerType;
    use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};

    match type_str {
        "float" => FP32Type::get(ctx).into(),
        "double" => FP64Type::get(ctx).into(),
        "half" => llvm_export::types::HalfType::get(ctx).into(),
        "i1" => IntegerType::get(ctx, 1, Signedness::Signless).into(),
        "i8" => IntegerType::get(ctx, 8, Signedness::Signless).into(),
        "i16" => IntegerType::get(ctx, 16, Signedness::Signless).into(),
        "i32" => IntegerType::get(ctx, 32, Signedness::Signless).into(),
        "i64" => IntegerType::get(ctx, 64, Signedness::Signless).into(),
        "i128" => IntegerType::get(ctx, 128, Signedness::Signless).into(),
        _ => PointerType::get(ctx, 0).into(), // Default to ptr
    }
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
    debug_kind: DebugKind,
) -> Result<String, PipelineError> {
    let module_op = Operation::get_op::<pliron::builtin::ops::ModuleOp>(module_op_ptr, ctx)
        .ok_or_else(|| PipelineError::Export("Not a module op".to_string()))?;

    let llvm_ir = if emit_nvvm_ir {
        let config = PipelineExportConfig {
            inner: llvm_export::export::NvvmExportConfig,
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

    std::fs::write(path, &llvm_ir).map_err(|e| PipelineError::Export(e.to_string()))?;

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

    fn debug_kind(&self) -> DebugKind {
        self.debug_kind
    }
}

/// Checks for WGMMA instructions (Hopper sm_90a only, NOT forward-compatible).
///
/// WGMMA (Warpgroup Matrix Multiply-Accumulate) requires sm_90a specifically.
/// These are NOT forward-compatible - only work on H100/H200.
fn contains_wgmma_features(ll_path: &Path) -> bool {
    if let Ok(contents) = std::fs::read_to_string(ll_path) {
        contents.contains("wgmma.fence")
            || contents.contains("wgmma.commit_group")
            || contents.contains("wgmma.wait_group")
            || contents.contains("wgmma.mma_async")
    } else {
        false
    }
}

/// Checks for Thread Block Cluster instructions (sm_90+).
///
/// Cluster features require Hopper (sm_90) or newer:
/// - Cluster special registers (%cluster_ctaid, %cluster_nctaid)
/// - Cluster synchronization (cluster.sync)
/// - Distributed shared memory (mapa.shared::cluster)
fn contains_cluster_features(ll_path: &Path) -> bool {
    if let Ok(contents) = std::fs::read_to_string(ll_path) {
        // Cluster special registers
        contents.contains("cluster_ctaid")
            || contents.contains("cluster_nctaid")
            // Cluster synchronization
            || contents.contains("cluster.sync")
            // Distributed shared memory
            || contents.contains("mapa.shared::cluster")
    } else {
        false
    }
}

/// Checks for TMA/mbarrier instructions (Hopper+ compatible with Blackwell).
///
/// These instructions work on BOTH Hopper and Blackwell:
/// - TMA: Tensor Memory Accelerator bulk copies
/// - mbarrier: Async hardware barriers with transaction tracking
///
/// Use sm_90 (not sm_90a) for forward compatibility with sm_120 (Blackwell).
fn contains_tma_features(ll_path: &Path) -> bool {
    if let Ok(contents) = std::fs::read_to_string(ll_path) {
        // TMA instructions
        contents.contains("cp.async.bulk.tensor")
            // mbarrier with transaction tracking (Hopper+)
            || contents.contains("mbarrier.arrive.expect_tx")
            || contents.contains("mbarrier.try_wait")
            // Proxy fence for async operations
            || contents.contains("fence.proxy.async")
    } else {
        false
    }
}
/// Checks for Blackwell tcgen05 instructions (sm_100a+).
///
/// These instructions require sm_100a/sm_120a (Blackwell) or newer:
/// - tcgen05: Tensor Core Gen 5 (TMEM allocation, MMA, sync primitives)
///
/// Key differences from Hopper:
/// - tcgen05 MMA is single-thread (vs WGMMA's 128 threads)
/// - Uses Tensor Memory (TMEM) instead of registers
/// - Different synchronization model (mbarrier-based)
fn contains_blackwell_features(ll_path: &Path) -> bool {
    if let Ok(contents) = std::fs::read_to_string(ll_path) {
        // tcgen05 TMEM allocation/deallocation
        contents.contains("tcgen05.alloc")
            || contents.contains("tcgen05.dealloc")
            || contents.contains("tcgen05.relinquish_alloc_permit")
            // tcgen05 synchronization
            || contents.contains("tcgen05.fence")
            || contents.contains("tcgen05.commit")
            // tcgen05 MMA instructions (ws and non-ws/cta_group forms)
            || contents.contains("tcgen05.mma")
            // tcgen05 data movement
            || contents.contains("tcgen05.cp")
    } else {
        false
    }
}

/// Checks for TMA multicast in LLVM IR (requires sm_100a).
///
/// TMA multicast (`cp.async.bulk.tensor...multicast::cluster`) is an
/// architecture-specific extension that broadcasts a tile to all CTAs in a
/// cluster. In the LLVM intrinsic, this is controlled by the `use_cta_mask`
/// parameter (second-to-last i1 argument) being set to true.
fn contains_tma_multicast(ll_path: &Path) -> bool {
    if let Ok(contents) = std::fs::read_to_string(ll_path) {
        contents
            .lines()
            .any(|line| line.contains("g2s.tile") && line.contains(", i1 1, i1"))
    } else {
        false
    }
}

/// GPU features detected in LLVM IR that determine target selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DetectedFeatures {
    /// tcgen05/TMEM (Blackwell datacenter, sm_100a).
    Blackwell,
    /// TMA multicast (arch-specific extension, sm_100a).
    TmaMulticast,
    /// WGMMA (Hopper only, sm_90a - NOT forward-compatible).
    Wgmma,
    /// TMA/mbarrier (Hopper+ compatible).
    Tma,
    /// Thread Block Clusters (sm_90+, forward-compatible).
    Cluster,
    /// No special features (maximum compatibility, sm_80).
    Basic,
}

/// Maps detected features to GPU target architecture.
fn select_target(features: DetectedFeatures) -> &'static str {
    match features {
        DetectedFeatures::Blackwell => "sm_100a",
        DetectedFeatures::TmaMulticast => "sm_100a",
        DetectedFeatures::Wgmma => "sm_90a",
        // TMA needs PTX 8.0+ which requires sm_90a or sm_100+.
        // sm_90a is NOT forward-compatible to Blackwell, so use sm_100 which:
        // - Generates PTX 8.6 (supports all TMA features)
        // - Works on all Blackwell variants (sm_100, sm_120)
        // - Hopper users can override with CUDA_OXIDE_TARGET=sm_90a
        DetectedFeatures::Tma => "sm_100",
        // Cluster features require sm_90+ but are forward-compatible.
        // Use sm_90 for Hopper compatibility, works on Blackwell too.
        DetectedFeatures::Cluster => "sm_90",
        DetectedFeatures::Basic => "sm_80",
    }
}

/// Does `arch` (e.g. `"sm_120a"`, `"sm_90"`) support the kernel's detected
/// features?
///
/// tcgen05/TMEM and `cta_group` TMA multicast exist only in the sm_100
/// datacenter-Blackwell family: consumer Blackwell (sm_120) and Hopper (sm_90)
/// lack them, so an sm_120 GPU cannot run an sm_100 tcgen05 kernel even though
/// 120 > 100. WGMMA is Hopper-only. The remaining features are forward
/// compatible from their floor (TMA / cluster need sm_90+, basic needs sm_80+).
///
/// Used to decide whether the GPU in this machine (the `CUDA_OXIDE_DEVICE_ARCH`
/// hint) can actually run the kernel, or whether we must build for the arch the
/// IR requires instead.
fn arch_satisfies(arch: &str, features: DetectedFeatures) -> bool {
    let Some(major) = arch_major(arch) else {
        return false;
    };
    match features {
        DetectedFeatures::Blackwell | DetectedFeatures::TmaMulticast => major == 10,
        DetectedFeatures::Wgmma => major == 9,
        DetectedFeatures::Tma | DetectedFeatures::Cluster => major >= 9,
        DetectedFeatures::Basic => major >= 8,
    }
}

/// Extract the compute-capability *major* version from an `sm_…` target string.
///
/// CUDA concatenates major+minor without a separator, so `"sm_120a"` is cc 12.0
/// (major 12), `"sm_90"` is cc 9.0, `"sm_103a"` is cc 10.3. We read the digit
/// run after `sm_` and divide by ten. Returns `None` when there are no digits.
fn arch_major(arch: &str) -> Option<u32> {
    let digits: String = arch
        .trim_start_matches("sm_")
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse::<u32>().ok().map(|n| n / 10)
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

    // Detect features (order matters: most specific first)
    let detected = match (
        contains_blackwell_features(ll_path),
        contains_tma_multicast(ll_path),
        contains_wgmma_features(ll_path),
        contains_tma_features(ll_path),
        contains_cluster_features(ll_path),
    ) {
        (true, _, _, _, _) => DetectedFeatures::Blackwell,
        (_, true, _, _, _) => DetectedFeatures::TmaMulticast,
        (_, _, true, _, _) => DetectedFeatures::Wgmma,
        (_, _, _, true, _) => DetectedFeatures::Tma,
        (_, _, _, _, true) => DetectedFeatures::Cluster,
        _ => DetectedFeatures::Basic,
    };

    // Arch the IR actually requires (the hard floor).
    let feature_arch = select_target(detected);

    // Resolve the final target:
    //   1. explicit override -- honored as-is. If it cannot lower the kernel's
    //      features we warn (otherwise llc aborts with a cryptic backend error).
    //   2. detected-device hint -- used only if that GPU can run the kernel;
    //      otherwise we build for `feature_arch`. The resulting PTX will not
    //      load on this GPU, but feature-gated examples handle that at load time
    //      (cuModuleLoad reports INVALID_PTX and they skip execution).
    //   3. neither set -- the feature floor.
    let (target, target_source): (String, &str) = if let Some(t) = explicit_override {
        if !arch_satisfies(&t, detected) {
            eprintln!(
                "warning: CUDA_OXIDE_TARGET={t} cannot lower the detected feature \
                 {detected:?} (needs {feature_arch}); PTX generation will likely \
                 fail. Unset CUDA_OXIDE_TARGET to let cuda-oxide select \
                 {feature_arch} automatically."
            );
        }
        (t, "CUDA_OXIDE_TARGET")
    } else if let Some(dev) = device_hint.filter(|d| arch_satisfies(d, detected)) {
        (dev, "detected GPU")
    } else {
        (feature_arch.to_string(), "feature requirement")
    };

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
    use std::{fs, path::PathBuf};

    fn write_temp_ll(name: &str, contents: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "cuda_oxide_mir_importer_{}_{}.ll",
            std::process::id(),
            name
        ));
        fs::write(&path, contents).expect("write temp LLVM IR");
        path
    }

    #[test]
    fn test_pipeline_config_default_values() {
        let config = PipelineConfig::default();

        assert_eq!(config.output_name, "kernel");
        assert!(config.verbose);
        assert!(!config.show_mir_dialect);
        assert!(!config.show_llvm_dialect);
        assert!(!config.emit_nvvm_ir);
        assert_eq!(config.debug_kind, DebugKind::Off);
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
            debug_kind: DebugKind::Off,
        };

        let result = run_pipeline(&[], &[], &config).expect("pipeline run");

        assert!(output_dir.is_dir());
        assert!(result.ll_path.is_file());
        assert_eq!(result.artifact_path, result.ll_path);
        assert_eq!(result.artifact_kind, CompilationArtifactKind::NvvmIr);

        fs::remove_dir_all(&root).expect("clean up temp output dir");
    }

    #[test]
    fn test_device_extern_decl_converts_to_export_decl() {
        let decl = DeviceExternDecl {
            export_name: "device_add".to_string(),
            param_types: vec!["ptr".to_string(), "i32".to_string()],
            return_type: "void".to_string(),
            attrs: DeviceExternAttrs {
                is_convergent: true,
                is_pure: false,
                is_readonly: true,
            },
        };

        let exported = decl.as_device_extern();

        assert_eq!(exported.export_name, "device_add");
        assert_eq!(exported.param_types, ["ptr", "i32"]);
        assert_eq!(exported.return_type, "void");
        assert!(exported.attrs.is_convergent);
        assert!(!exported.attrs.is_pure);
        assert!(exported.attrs.is_readonly);
    }

    #[test]
    fn test_feature_detection_reads_llvm_ir_snippets() {
        let path = write_temp_ll(
            "features",
            r#"
                call void asm sideeffect "wgmma.fence.sync.aligned", ""()
                call void @llvm.nvvm.tcgen05.alloc()
                call void asm sideeffect "cluster.sync.aligned", ""()
                call void asm sideeffect "cp.async.bulk.tensor.2d.shared::cluster.global", ""()
            "#,
        );

        assert!(contains_wgmma_features(&path));
        assert!(contains_blackwell_features(&path));
        assert!(contains_cluster_features(&path));
        assert!(contains_tma_features(&path));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn test_tma_multicast_detection_requires_cta_mask() {
        let multicast = write_temp_ll(
            "tma_multicast",
            "call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile(i32 0, i1 1, i1 false)",
        );
        let unicast = write_temp_ll(
            "tma_unicast",
            "call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile(i32 0, i1 0, i1 false)",
        );

        assert!(contains_tma_multicast(&multicast));
        assert!(!contains_tma_multicast(&unicast));

        let _ = fs::remove_file(multicast);
        let _ = fs::remove_file(unicast);
    }

    #[test]
    fn test_select_target_prefers_required_architecture() {
        assert_eq!(select_target(DetectedFeatures::Blackwell), "sm_100a");
        assert_eq!(select_target(DetectedFeatures::TmaMulticast), "sm_100a");
        assert_eq!(select_target(DetectedFeatures::Wgmma), "sm_90a");
        assert_eq!(select_target(DetectedFeatures::Tma), "sm_100");
        assert_eq!(select_target(DetectedFeatures::Cluster), "sm_90");
        assert_eq!(select_target(DetectedFeatures::Basic), "sm_80");
    }

    #[test]
    fn test_arch_major_parses_cuda_spelling() {
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
    fn test_arch_satisfies_sm100_only_features() {
        // tcgen05 and cta_group TMA multicast are sm_100-family only:
        // consumer Blackwell (sm_120) and Hopper (sm_90) cannot run them, even
        // though 120 > 100. This is the gemm_sol regression guard.
        for f in [DetectedFeatures::Blackwell, DetectedFeatures::TmaMulticast] {
            assert!(arch_satisfies("sm_100a", f), "sm_100a must satisfy {f:?}");
            assert!(
                !arch_satisfies("sm_120a", f),
                "sm_120a must NOT satisfy {f:?}"
            );
            assert!(
                !arch_satisfies("sm_90a", f),
                "sm_90a must NOT satisfy {f:?}"
            );
        }
    }

    #[test]
    fn test_arch_satisfies_wgmma_is_hopper_only() {
        assert!(arch_satisfies("sm_90a", DetectedFeatures::Wgmma));
        assert!(!arch_satisfies("sm_100a", DetectedFeatures::Wgmma));
        assert!(!arch_satisfies("sm_120a", DetectedFeatures::Wgmma));
    }

    #[test]
    fn test_arch_satisfies_forward_compatible_features() {
        // Plain TMA / cluster lower on any sm_90+ device; basic on any sm_80+.
        // So a consumer sm_120 GPU is a valid target for these (it runs locally
        // instead of being downgraded to the feature floor).
        for arch in ["sm_90a", "sm_100a", "sm_120a"] {
            assert!(arch_satisfies(arch, DetectedFeatures::Tma));
            assert!(arch_satisfies(arch, DetectedFeatures::Cluster));
            assert!(arch_satisfies(arch, DetectedFeatures::Basic));
        }
        assert!(arch_satisfies("sm_80", DetectedFeatures::Basic));
        assert!(!arch_satisfies("sm_80", DetectedFeatures::Tma));
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
