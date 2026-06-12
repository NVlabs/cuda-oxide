/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! # Device Code Generation via cuda-oxide Pipeline
//!
//! This module bridges rustc's internal MIR representation to cuda-oxide's
//! existing MIR→PTX pipeline using `rustc_public::rustc_internal` to convert
//! between internal and stable_mir types.
//!
//! ## The Bridge Problem
//!
//! We have two different MIR representations:
//!
//! | API                         | Used By                       | Type                                |
//! |-----------------------------|-------------------------------|-------------------------------------|
//! | `rustc_middle` (internal)   | rustc internals, this backend | `rustc_middle::ty::Instance<'tcx>`  |
//! | `rustc_public` (stable MIR) | mir-importer pipeline         | `rustc_public::mir::mono::Instance` |
//!
//! The cuda-oxide pipeline (mir-importer) was built using `rustc_public` APIs because
//! they're more stable. But as a codegen backend, we receive `rustc_middle` types from
//! rustc. This module bridges between them.
//!
//! ## Bridge Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────────┐
//! │                         DEVICE CODE GENERATION                                  │
//! │                                                                                 │
//! │   Input: Vec<CollectedFunction<'tcx>>                                           │
//! │          (using rustc_middle::ty::Instance)                                     │
//! │                                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────────────┐   │
//! │   │  STEP 1: Enter stable_mir Context                                       │   │
//! │   │                                                                         │   │
//! │   │  rustc_internal::run(tcx, || { ... })                                   │   │
//! │   │                                                                         │   │
//! │   │  This sets up the Tables and CompilerCtxt that enable type conversion   │   │
//! │   │  between rustc_middle and rustc_public types.                           │   │
//! │   └─────────────────────────────────────────────────────────────────────────┘   │
//! │                              │                                                  │
//! │                              ▼                                                  │
//! │   ┌─────────────────────────────────────────────────────────────────────────┐   │
//! │   │  STEP 2: Convert Instances                                              │   │
//! │   │                                                                         │   │
//! │   │  for each CollectedFunction<'tcx>:                                      │   │
//! │   │      stable_instance = rustc_internal::stable(func.instance)            │   │
//! │   │                                                                         │   │
//! │   │  This converts:                                                         │   │
//! │   │    rustc_middle::ty::Instance<'tcx>                                     │   │
//! │   │         ▼                                                               │   │
//! │   │    rustc_public::mir::mono::Instance                                    │   │
//! │   └─────────────────────────────────────────────────────────────────────────┘   │
//! │                              │                                                  │
//! │                              ▼                                                  │
//! │   ┌─────────────────────────────────────────────────────────────────────────┐   │
//! │   │  STEP 3: Run cuda-oxide Pipeline                                        │   │
//! │   │                                                                         │   │
//! │   │  mir_importer::run_pipeline(&stable_functions, &config)                 │   │
//! │   │                                                                         │   │
//! │   │  Pipeline stages:                                                       │   │
//! │   │    1. Rust MIR → `dialect-mir` (alloca form)                            │   │
//! │   │    2. `dialect-mir` → `dialect-mir` (mem2reg → SSA)                     │   │
//! │   │    3. `dialect-mir` → LLVM dialect (via `mir-lower`)                    │   │
//! │   │    4. LLVM dialect → textual LLVM IR (.ll)                              │   │
//! │   │    5. LLVM IR → PTX via `llc` (.ptx)                                    │   │
//! │   └─────────────────────────────────────────────────────────────────────────┘   │
//! │                              │                                                  │
//! │                              ▼                                                  │
//! │   ┌─────────────────────────────────────────────────────────────────────────┐   │
//! │   │  Output: DeviceCodegenResult                                            │   │
//! │   │                                                                         │   │
//! │   │    - ptx_path: Path to generated .ptx file                              │   │
//! │   │    - ll_path: Path to generated .ll file                                │   │
//! │   │    - target: GPU target (e.g., "sm_80", "sm_90a")                       │   │
//! │   │    - ptx_content: PTX as string, when PTX was generated                 │   │
//! │   └─────────────────────────────────────────────────────────────────────────┘   │
//! │                                                                                 │
//! └─────────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Why This Design?
//!
//! We chose to bridge to stable_mir rather than rewrite mir-importer because:
//!
//! 1. **Code reuse**: mir-importer already works and is well-tested
//! 2. **Stability**: rustc_public APIs change less than rustc internals
//! 3. **Simplicity**: ~100 lines of bridge code vs rewriting the pipeline
//! 4. **Maintainability**: Changes to mir-importer automatically work here
//!
//! The cost is one extra type conversion step, but this happens once per function
//! and is negligible compared to actual compilation time.

use crate::collector::{CollectedFunction, DeviceExternDecl};
use rustc_middle::ty::{Ty, TyCtxt, TyKind};
use std::path::PathBuf;

/// Convert a rustc type to an LLVM type string for device extern declarations.
///
/// This is a simplified conversion that handles common FFI types.
/// For device code, we primarily deal with:
/// - Primitives: i8, i16, i32, i64, f32, f64
/// - Pointers: all become `ptr` (opaque pointers)
/// - Unit: becomes `void`
fn rustc_ty_to_llvm_type_string(ty: Ty<'_>) -> String {
    match ty.kind() {
        // Integer types
        TyKind::Int(int_ty) => match int_ty {
            rustc_middle::ty::IntTy::I8 => "i8".to_string(),
            rustc_middle::ty::IntTy::I16 => "i16".to_string(),
            rustc_middle::ty::IntTy::I32 => "i32".to_string(),
            rustc_middle::ty::IntTy::I64 => "i64".to_string(),
            rustc_middle::ty::IntTy::I128 => "i128".to_string(),
            rustc_middle::ty::IntTy::Isize => "i64".to_string(), // nvptx64
        },
        TyKind::Uint(uint_ty) => match uint_ty {
            rustc_middle::ty::UintTy::U8 => "i8".to_string(),
            rustc_middle::ty::UintTy::U16 => "i16".to_string(),
            rustc_middle::ty::UintTy::U32 => "i32".to_string(),
            rustc_middle::ty::UintTy::U64 => "i64".to_string(),
            rustc_middle::ty::UintTy::U128 => "i128".to_string(),
            rustc_middle::ty::UintTy::Usize => "i64".to_string(), // nvptx64
        },

        // Float types
        TyKind::Float(float_ty) => match float_ty {
            rustc_middle::ty::FloatTy::F16 => "half".to_string(),
            rustc_middle::ty::FloatTy::F32 => "float".to_string(),
            rustc_middle::ty::FloatTy::F64 => "double".to_string(),
            rustc_middle::ty::FloatTy::F128 => "fp128".to_string(),
        },

        // Bool
        TyKind::Bool => "i1".to_string(),

        // Char (32-bit in Rust)
        TyKind::Char => "i32".to_string(),

        // Unit type (void in LLVM)
        TyKind::Tuple(tys) if tys.is_empty() => "void".to_string(),

        // Pointers and references - all become opaque ptr in LLVM 20+
        TyKind::RawPtr(_, _) | TyKind::Ref(_, _, _) => "ptr".to_string(),

        // Never type - we shouldn't see this in extern signatures
        TyKind::Never => "void".to_string(),

        // For any other type, use ptr as a fallback
        // This handles arrays, slices, structs, etc. that are passed by pointer
        _ => "ptr".to_string(),
    }
}

/// Result of device code generation.
///
/// Contains paths to generated artifacts and the payload selected for
/// embedding in the host binary.
pub struct DeviceCodegenResult {
    /// Path to generated PTX assembly file.
    ///
    /// In NVVM IR modes this is the would-be PTX path and may not exist.
    pub ptx_path: PathBuf,
    /// Path to generated LLVM IR file.
    pub ll_path: PathBuf,
    /// GPU target architecture used (e.g., "sm_80", "sm_90a", "sm_100a").
    ///
    /// Auto-detected based on GPU features used, or overridden via
    /// `CUDA_OXIDE_TARGET` environment variable.
    pub target: String,
    /// PTX content as a string, ready for embedding in the host binary.
    ///
    /// NVVM IR / LTOIR flows intentionally skip PTX generation.
    pub ptx_content: Option<String>,
    /// Device artifact payload selected for embedding.
    pub artifact: Option<DeviceCodegenArtifact>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceCodegenArtifactKind {
    Ptx,
    NvvmIr,
    Ltoir,
    Cubin,
}

pub struct DeviceCodegenArtifact {
    pub kind: DeviceCodegenArtifactKind,
    pub name: String,
    pub bytes: Vec<u8>,
}

/// Configuration for device codegen.
///
/// Controls output paths and diagnostic output during compilation.
pub struct DeviceCodegenConfig {
    /// Output directory for generated files (.ll, .ptx).
    pub output_dir: PathBuf,
    /// Base name for output files (e.g., "kernel" → kernel.ll, kernel.ptx).
    pub output_name: String,
    /// Print verbose progress to stderr.
    pub verbose: bool,
    /// Dump raw rustc MIR before translation.
    pub dump_rustc_mir: bool,
    /// Dump the `dialect-mir` module during compilation.
    pub dump_mir_dialect: bool,
    /// Dump the LLVM dialect module during compilation.
    pub dump_llvm_dialect: bool,
}

impl Default for DeviceCodegenConfig {
    fn default() -> Self {
        Self {
            output_dir: std::env::current_dir().unwrap_or_else(|_| ".".into()),
            output_name: "kernel".to_string(),
            verbose: false,
            dump_rustc_mir: false,
            dump_mir_dialect: false,
            dump_llvm_dialect: false,
        }
    }
}

/// Errors that can occur during device code generation.
#[derive(Debug)]
pub enum DeviceCodegenError {
    /// No kernels were found to compile.
    NoKernels,
    /// Failed to enter or exit stable_mir context.
    StableMirError(String),
    /// MIR to Pliron IR translation failed.
    Translation(String),
    /// PTX generation (llc invocation) failed.
    PtxGeneration(String),
    /// IO error (file read/write).
    Io(std::io::Error),
}

impl std::fmt::Display for DeviceCodegenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoKernels => write!(f, "No kernel functions found"),
            Self::StableMirError(msg) => write!(f, "stable_mir error: {}", msg),
            Self::Translation(msg) => write!(f, "Translation failed: {}", msg),
            Self::PtxGeneration(msg) => write!(f, "PTX generation failed: {}", msg),
            Self::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for DeviceCodegenError {}

impl From<std::io::Error> for DeviceCodegenError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Generates PTX for device functions using the cuda-oxide pipeline.
///
/// This is the main entry point for device codegen. It bridges between
/// rustc's internal types (`rustc_middle`) and mir-importer's stable_mir-based
/// pipeline (`rustc_public`).
///
/// ## Parameters
///
/// - `tcx`: The type context from rustc
/// - `functions`: Collected device functions from the collector module
/// - `config`: Output and diagnostic configuration
///
/// ## Returns
///
/// - `Ok(DeviceCodegenResult)`: Paths to generated .ll and .ptx files
/// - `Err(DeviceCodegenError)`: Description of what went wrong
///
/// ## Pipeline Stages
///
/// ```text
/// CollectedFunction<'tcx>
///         │
///         ├──▶ rustc_internal::stable() ──▶ rustc_public::Instance
///         │
///         └──▶ mir_importer::run_pipeline()
///                     │
///                     ├──▶ `dialect-mir` (alloca form)
///                     │
///                     ├──▶ `dialect-mir` (mem2reg → SSA)
///                     │
///                     ├──▶ LLVM dialect
///                     │
///                     ├──▶ textual LLVM IR (.ll)
///                     │
///                     └──▶ PTX (.ptx) via `llc`
/// ```
pub fn generate_device_code<'tcx>(
    tcx: TyCtxt<'tcx>,
    functions: &[CollectedFunction<'tcx>],
    device_externs: &[DeviceExternDecl],
    config: &DeviceCodegenConfig,
) -> Result<DeviceCodegenResult, DeviceCodegenError> {
    use rustc_public::rustc_internal;

    if functions.is_empty() {
        return Err(DeviceCodegenError::NoKernels);
    }

    if config.verbose {
        eprintln!(
            "[device_codegen] Compiling {} functions, {} device externs to PTX via cuda-oxide pipeline",
            functions.len(),
            device_externs.len()
        );
        for func in functions {
            eprintln!(
                "[device_codegen]   {} {}",
                if func.is_kernel { "kernel" } else { "device" },
                func.export_name
            );
        }
        for decl in device_externs {
            eprintln!(
                "[device_codegen]   extern {} (convergent={}, pure={}, readonly={})",
                decl.export_name,
                decl.attrs.is_convergent,
                decl.attrs.is_pure,
                decl.attrs.is_readonly
            );
        }
    }

    // Prepare data we need to pass into the stable_mir closure
    // (closures can't capture references to local TyCtxt data)
    let export_names: Vec<(String, bool)> = functions
        .iter()
        .map(|f| (f.export_name.clone(), f.is_kernel))
        .collect();

    // Convert device externs to mir-importer format
    // We extract signature info from rustc here since we have access to TyCtxt
    let stable_device_externs: Vec<mir_importer::DeviceExternDecl> = device_externs
        .iter()
        .map(|decl| {
            // Get function signature from rustc
            let fn_sig = tcx.fn_sig(decl.def_id).instantiate_identity();
            let fn_sig = fn_sig.skip_binder();

            // Convert parameter types to LLVM type strings
            let param_types: Vec<String> = fn_sig
                .inputs()
                .iter()
                .map(|ty| rustc_ty_to_llvm_type_string(*ty))
                .collect();

            // Convert return type to LLVM type string
            let return_type = rustc_ty_to_llvm_type_string(fn_sig.output());

            mir_importer::DeviceExternDecl {
                export_name: decl.export_name.clone(),
                param_types,
                return_type,
                attrs: mir_importer::DeviceExternAttrs {
                    is_convergent: decl.attrs.is_convergent,
                    is_pure: decl.attrs.is_pure,
                    is_readonly: decl.attrs.is_readonly,
                },
            }
        })
        .collect();

    let output_dir = config.output_dir.clone();
    let output_name = config.output_name.clone();
    let verbose = config.verbose;
    let show_rustc_mir = config.dump_rustc_mir;
    let show_mir = config.dump_mir_dialect;
    let show_llvm = config.dump_llvm_dialect;

    // Print raw rustc MIR if requested (before conversion to stable_mir)
    if show_rustc_mir {
        use rustc_middle::ty::print::with_no_trimmed_paths;

        eprintln!();
        eprintln!("=== Rustc MIR (before translation) ===");
        for func in functions {
            let mir = tcx.instance_mir(func.instance.def);
            eprintln!();
            eprintln!("fn {} {{", func.export_name);

            // Print locals
            eprintln!(
                "    let mut _0: {:?};",
                mir.local_decls[rustc_middle::mir::Local::from_u32(0)].ty
            );
            for (local, decl) in mir.local_decls.iter_enumerated().skip(1) {
                let mutability = if decl.mutability == rustc_middle::mir::Mutability::Mut {
                    "mut "
                } else {
                    ""
                };
                eprintln!("    let {}_{}:  {:?};", mutability, local.index(), decl.ty);
            }

            // Print debug info
            for debug_info in &mir.var_debug_info {
                with_no_trimmed_paths!(eprintln!(
                    "    debug {:?} => {:?};",
                    debug_info.name, debug_info.value
                ));
            }

            // Print basic blocks
            for (bb_idx, bb_data) in mir.basic_blocks.iter_enumerated() {
                eprintln!("    bb{}: {{", bb_idx.index());
                for stmt in &bb_data.statements {
                    with_no_trimmed_paths!(eprintln!("        {:?}", stmt));
                }
                with_no_trimmed_paths!(eprintln!("        {:?}", bb_data.terminator().kind));
                eprintln!("    }}");
            }
            eprintln!("}}");
        }
        eprintln!();
    }

    // Enter stable_mir context and run the pipeline.
    //
    // rustc_internal::run() does the following:
    // 1. Creates Tables for type/instance interning
    // 2. Sets up thread-local CompilerCtxt
    // 3. Runs our closure with access to stable() conversion
    // 4. Tears down the context and returns our result
    let result = rustc_internal::run(tcx, || {
        // Convert internal Instance<'tcx> to stable_mir Instance
        let stable_functions: Vec<mir_importer::CollectedFunction> = functions
            .iter()
            .zip(export_names.iter())
            .map(|(func, (export_name, is_kernel))| {
                // Use rustc_internal::stable() to convert the Instance.
                // This is the key bridge between rustc_middle and rustc_public types.
                let stable_instance = rustc_internal::stable(func.instance);

                mir_importer::CollectedFunction {
                    instance: stable_instance,
                    is_kernel: *is_kernel,
                    export_name: export_name.clone(),
                }
            })
            .collect();

        // Check for NVVM IR mode (set by cargo oxide --emit-nvvm-ir)
        let emit_nvvm_ir = std::env::var("CUDA_OXIDE_EMIT_NVVM_IR").is_ok();

        if verbose {
            eprintln!(
                "[device_codegen] Converted {} functions to stable_mir format",
                stable_functions.len()
            );
            if emit_nvvm_ir {
                eprintln!("[device_codegen] NVVM IR mode enabled");
            }
        }

        // Create pipeline config
        let pipeline_config = mir_importer::PipelineConfig {
            output_dir: output_dir.clone(),
            output_name: output_name.clone(),
            verbose,
            show_mir_dialect: show_mir,
            show_llvm_dialect: show_llvm,
            emit_nvvm_ir,
        };

        // Run the cuda-oxide pipeline!
        // Rust MIR → `dialect-mir` → mem2reg → LLVM dialect → LLVM IR → PTX.
        // Device externs are emitted as `declare` statements in LLVM IR
        mir_importer::run_pipeline(&stable_functions, &stable_device_externs, &pipeline_config)
    });

    // Handle the result from rustc_internal::run.
    // We have nested Results: outer from run(), inner from run_pipeline().
    match result {
        Ok(pipeline_result) => match pipeline_result {
            Ok(compilation_result) => {
                let artifact = read_compilation_artifact(&compilation_result)?;
                let ptx_content = match artifact.as_ref() {
                    Some(artifact) if artifact.kind == DeviceCodegenArtifactKind::Ptx => {
                        Some(String::from_utf8(artifact.bytes.clone()).map_err(|e| {
                            DeviceCodegenError::PtxGeneration(format!(
                                "generated PTX is not valid UTF-8: {e}"
                            ))
                        })?)
                    }
                    _ => None,
                };

                if config.verbose {
                    if let Some(artifact) = artifact.as_ref() {
                        eprintln!(
                            "[device_codegen] Embeddable artifact generated: {} ({:?}, target: {})",
                            artifact.name, artifact.kind, compilation_result.target
                        );
                    } else {
                        eprintln!(
                            "[device_codegen] No embeddable artifact found for {} (target: {})",
                            compilation_result.ll_path.display(),
                            compilation_result.target
                        );
                    }
                }

                Ok(DeviceCodegenResult {
                    ptx_path: compilation_result.ptx_path,
                    ll_path: compilation_result.ll_path,
                    target: compilation_result.target,
                    ptx_content,
                    artifact,
                })
            }
            Err(pipeline_err) => Err(DeviceCodegenError::PtxGeneration(format!(
                "{}",
                pipeline_err
            ))),
        },
        Err(stable_mir_err) => Err(DeviceCodegenError::StableMirError(format!(
            "{:?}",
            stable_mir_err
        ))),
    }
}

fn read_compilation_artifact(
    result: &mir_importer::CompilationResult,
) -> Result<Option<DeviceCodegenArtifact>, DeviceCodegenError> {
    let kind = match result.artifact_kind {
        mir_importer::CompilationArtifactKind::Ptx => DeviceCodegenArtifactKind::Ptx,
        mir_importer::CompilationArtifactKind::NvvmIr => DeviceCodegenArtifactKind::NvvmIr,
        mir_importer::CompilationArtifactKind::Ltoir => DeviceCodegenArtifactKind::Ltoir,
        mir_importer::CompilationArtifactKind::Cubin => DeviceCodegenArtifactKind::Cubin,
    };

    match std::fs::read(&result.artifact_path) {
        Ok(bytes) => Ok(Some(DeviceCodegenArtifact {
            kind,
            name: result
                .artifact_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("device-artifact")
                .to_string(),
            bytes,
        })),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(DeviceCodegenError::Io(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = DeviceCodegenConfig::default();
        assert!(!config.verbose);
        assert_eq!(config.output_name, "kernel");
    }

    #[test]
    fn read_compilation_artifact_uses_declared_nvvm_ir_path() {
        let temp_dir = unique_temp_dir("cuda-codegen-artifact");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let ll_path = temp_dir.join("demo.ll");
        let ptx_path = temp_dir.join("demo.ptx");
        std::fs::write(&ll_path, b"nvvm ir").unwrap();
        std::fs::write(&ptx_path, b"stale ptx").unwrap();

        let result = mir_importer::CompilationResult {
            ll_path: ll_path.clone(),
            ptx_path,
            artifact_path: ll_path,
            artifact_kind: mir_importer::CompilationArtifactKind::NvvmIr,
            target: "sm_90".to_string(),
        };

        let artifact = read_compilation_artifact(&result).unwrap().unwrap();
        assert_eq!(artifact.kind, DeviceCodegenArtifactKind::NvvmIr);
        assert_eq!(artifact.name, "demo.ll");
        assert_eq!(artifact.bytes, b"nvvm ir");

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn read_compilation_artifact_reads_declared_cubin_path() {
        let temp_dir = unique_temp_dir("cuda-codegen-artifact");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let ll_path = temp_dir.join("demo.ll");
        let cubin_path = temp_dir.join("demo.cubin");
        let ptx_path = temp_dir.join("demo.ptx");
        std::fs::write(&cubin_path, b"cubin").unwrap();

        let result = mir_importer::CompilationResult {
            ll_path,
            ptx_path,
            artifact_path: cubin_path,
            artifact_kind: mir_importer::CompilationArtifactKind::Cubin,
            target: "sm_90".to_string(),
        };

        let artifact = read_compilation_artifact(&result).unwrap().unwrap();
        assert_eq!(artifact.kind, DeviceCodegenArtifactKind::Cubin);
        assert_eq!(artifact.name, "demo.cubin");
        assert_eq!(artifact.bytes, b"cubin");

        let _ = std::fs::remove_dir_all(temp_dir);
    }

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{name}-{}-{nanos}", std::process::id()))
    }
}
