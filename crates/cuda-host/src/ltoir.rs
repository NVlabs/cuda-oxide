/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Build a cubin from cuda-oxide's NVVM IR output via libNVVM + libdevice + nvJitLink.
//!
//! When a kernel uses Rust float math intrinsics (`sin`, `cos`, `exp`, `pow`,
//! ...), cuda-oxide lowers them to CUDA `__nv_*` libdevice calls, auto-detects
//! their presence, emits NVVM IR (`<name>.ll`) instead of `.ptx`, and skips
//! `llc`. The application then has to:
//!
//! 1. Compile the NVVM IR to LTOIR via libNVVM, with libdevice added so the
//!    `__nv_*` symbols are inlined.
//! 2. Link the resulting LTOIR via nvJitLink to produce either a cubin for
//!    the same architecture, or PTX when a pre-Blackwell module is loaded on
//!    a Blackwell GPU.
//! 3. Load that image with the CUDA driver.
//!
//! This module wraps that pipeline behind file and in-memory helpers:
//!
//! - [`build_cubin_from_ll`] -- explicit form, takes a `.ll` path and arch.
//! - [`load_kernel_module`] -- the convenience form. Looks at the example's
//!   directory and selects PTX, cubin, NVVM IR, or LTOIR using the recorded
//!   artifact mode. Use this for normal module loading.
//!
//! All work is done via [`libnvvm_sys`] and [`nvjitlink_sys`] (`dlopen` of
//! `libnvvm.so` and `libnvJitLink.so` from the CUDA Toolkit). No external
//! C tools are required, no symlinked `tools/` directory, no boilerplate.
//!
//! # Discovery
//!
//! - **libNVVM**: `LIBNVVM_PATH`, then `<root>/nvvm/lib64/libnvvm.so` for
//!   `<root>` in `CUDA_TOOLKIT_PATH`, `CUDA_HOME`, `CUDA_PATH`,
//!   `/usr/local/cuda`, `/opt/cuda`, then the system loader.
//! - **nvJitLink**: the same order, using `LIBNVJITLINK_PATH` and
//!   `<root>/lib64/libnvJitLink.so`.
//! - **libdevice**: `CUDA_OXIDE_LIBDEVICE` env var, then
//!   `<root>/nvvm/libdevice/libdevice.10.bc` for the same roots.
//! - **Artifact target**: the emitted `<name>.target` file, then the explicit
//!   `CUDA_OXIDE_TARGET` override (set by `cargo oxide --arch=<sm_XX>`).
//!   The CUDA context is queried separately for the GPU that will execute the
//!   module.
//!
//! # Example
//!
//! ```no_run
//! use cuda_core::CudaContext;
//! use cuda_host::ltoir;
//!
//! let ctx = CudaContext::new(0)?;
//! // Loads my_kernel.cubin (or builds + loads from my_kernel.ll).
//! let module = ltoir::load_kernel_module(&ctx, "my_kernel")?;
//! # Ok::<_, Box<dyn std::error::Error>>(())
//! ```

use cuda_core::{CudaContext, CudaModule, DriverError};
use libnvvm_sys::{CudaArch, CudaArchParseError, LibNvvm, NvvmError, Program};
use nvjitlink_sys::{InputType, LibNvJitLink, Linker, NvJitLinkError};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

// ============================================================================
// Errors
// ============================================================================

/// Failures while building or loading a module via the LTOIR pipeline.
#[derive(Debug, Error)]
pub enum LtoirError {
    /// A target string was not a concrete CUDA architecture.
    #[error(transparent)]
    InvalidTarget(#[from] CudaArchParseError),

    /// NVVM IR/LTOIR was loaded without its original target architecture.
    #[error(
        "NVVM IR/LTOIR does not record its CUDA target. Keep the generated .target file, \
         rebuild the artifact, or explicitly set CUDA_OXIDE_TARGET to its original target."
    )]
    TargetNotFound,

    /// The caller tried to compile target-specific NVVM IR for another target.
    #[error(
        "NVVM IR target mismatch: {ir_path} was emitted for {emitted}, but the loader requested {requested}"
    )]
    TargetMismatch {
        ir_path: PathBuf,
        emitted: String,
        requested: String,
    },

    /// An artifact is not compatible with the context's GPU.
    #[error("cannot execute CUDA artifact emitted for {emitted} on {execution}: {reason}")]
    IncompatibleExecutionTarget {
        /// Architecture for which the NVVM IR or LTOIR was emitted.
        emitted: String,
        /// Numeric architecture reported by the CUDA context.
        execution: String,
        /// Why the artifact is incompatible.
        reason: &'static str,
    },

    /// The installed toolkit does not accept the NVVM IR version cuda-oxide emits.
    #[error("installed libNVVM accepts NVVM IR {major}.{minor}, but cuda-oxide emits NVVM IR 2.0")]
    UnsupportedNvvmIrVersion { major: i32, minor: i32 },

    /// Runtime toolkit dialect discovery disagreed with cuda-oxide's target policy.
    #[error(
        "libNVVM reports LLVM {llvm_major} for {target}, which disagrees with cuda-oxide's expected {expected} dialect"
    )]
    DialectMismatch {
        target: String,
        llvm_major: i32,
        expected: &'static str,
    },

    /// libNVVM failed (load, symbol resolution, or compile call). Forwards
    /// the underlying [`NvvmError`].
    #[error("libnvvm: {0}")]
    Nvvm(#[from] NvvmError),

    /// nvJitLink failed (load, symbol resolution, or link call). Forwards
    /// the underlying [`NvJitLinkError`].
    #[error("nvJitLink: {0}")]
    NvJitLink(#[from] NvJitLinkError),

    /// `libdevice.10.bc` could not be located. `tried` lists every path
    /// that was probed, in order, joined by newlines.
    #[error(
        "Could not locate libdevice.10.bc. Set CUDA_OXIDE_LIBDEVICE, CUDA_TOOLKIT_PATH, or CUDA_HOME, or install the CUDA Toolkit. Tried:\n  {tried}"
    )]
    LibdeviceNotFound {
        /// Newline-joined list of paths that were probed.
        tried: String,
    },

    /// Reading or writing one of the pipeline artifacts (`.ll`,
    /// `libdevice.10.bc`, `.ltoir`, `.cubin`) failed.
    #[error("Failed reading {path}: {source}")]
    Io {
        /// Path of the file that could not be read or written.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// [`load_kernel_module`] could not find a supported artifact in the
    /// binary's manifest directory.
    #[error(
        "Could not find any kernel artifact for {name} in {dir}. \
         Looked for {name}.cubin, {name}.ptx, {name}.ll, {name}.ltoir. \
         Did `cargo oxide run` complete successfully?"
    )]
    NoArtifact {
        /// Kernel artifact stem that was looked up.
        name: String,
        /// Directory that was searched.
        dir: PathBuf,
    },

    /// `cuModuleLoad` (or another driver call) returned an error after the
    /// pipeline produced a cubin.
    #[error("CUDA driver: {0}")]
    Driver(#[from] DriverError),
}

// ============================================================================
// Build (NVVM IR + libdevice -> LTOIR -> cubin)
// ============================================================================

/// Compile NVVM IR at `ll_path` to a cubin and return its path.
///
/// Steps:
/// 1. Read `ll_path` (NVVM IR text) and the libdevice bitcode (located via
///    [`find_libdevice`]).
/// 2. Compile both via libNVVM with `-gen-lto` to produce LTOIR. The LTOIR
///    is written next to `ll_path` as `<stem>.ltoir` for debugging.
/// 3. Link the LTOIR via nvJitLink with `-arch=<arch> -lto` to produce a
///    cubin. The cubin is written next to `ll_path` as `<stem>.cubin`.
///
/// `arch` must be a concrete `sm_XX` or `compute_XX` CUDA target. It is
/// normalized to `compute_XX` for libNVVM and `sm_XX` for nvJitLink.
///
/// This function rebuilds from the source bytes on every call. A correct cache
/// would also need the source contents and the versions of libNVVM, libdevice,
/// and nvJitLink; file timestamps and a target string are not enough.
pub fn build_cubin_from_ll(ll_path: &Path, arch: &str) -> Result<PathBuf, LtoirError> {
    let arch: CudaArch = arch.parse()?;
    // Read the source before writing target metadata or derived artifacts.
    let ll_bytes = std::fs::read(ll_path).map_err(|source| LtoirError::Io {
        path: ll_path.to_path_buf(),
        source,
    })?;
    validate_ir_target_sidecar(ll_path, &arch)?;
    // Record the supplied target for older or manually created `.ll` files.
    // The sibling `.ltoir` can then be loaded after the `.ll` is removed.
    write_artifact_target_sidecar(ll_path, &arch)?;

    let stem = ll_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("kernel");
    let dir = ll_path.parent().unwrap_or_else(|| Path::new("."));
    let ltoir_path = dir.join(format!("{stem}.ltoir"));
    let cubin_path = dir.join(format!("{stem}.cubin"));

    let ltoir = compile_nvvm_ir_to_ltoir_parsed(&ll_bytes, &ll_path.display().to_string(), &arch)?;

    std::fs::write(&ltoir_path, &ltoir).map_err(|source| LtoirError::Io {
        path: ltoir_path.clone(),
        source,
    })?;

    // ---- nvJitLink: LTOIR -> cubin --------------------------------------
    let cubin = link_ltoir_to_cubin_parsed(&ltoir, &ltoir_path.display().to_string(), &arch)?;

    std::fs::write(&cubin_path, &cubin).map_err(|source| LtoirError::Io {
        path: cubin_path.clone(),
        source,
    })?;

    Ok(cubin_path)
}

/// Compile NVVM IR bytes to a loadable cubin image in memory.
///
/// This is the embedded-artifact counterpart of [`build_cubin_from_ll`]. It
/// adds `libdevice.10.bc`, asks libNVVM for LTOIR, links that LTOIR with
/// nvJitLink, and returns the final cubin bytes without creating sidecar files.
pub fn build_cubin_from_nvvm_ir(
    nvvm_ir: &[u8],
    module_name: &str,
    arch: &str,
) -> Result<Vec<u8>, LtoirError> {
    let arch: CudaArch = arch.parse()?;
    let ltoir = compile_nvvm_ir_to_ltoir_parsed(nvvm_ir, module_name, &arch)?;
    let ltoir_name = format!("{module_name}.ltoir");
    link_ltoir_to_cubin_parsed(&ltoir, &ltoir_name, &arch)
}

/// Compile NVVM IR bytes to forward-compatible PTX.
///
/// `arch` is the architecture for which the IR was **emitted**, not the GPU
/// on which it will execute. nvJitLink preserves that virtual architecture;
/// the CUDA driver later JIT-compiles the resulting PTX for the current GPU.
pub fn build_ptx_from_nvvm_ir(
    nvvm_ir: &[u8],
    module_name: &str,
    arch: &str,
) -> Result<Vec<u8>, LtoirError> {
    let arch: CudaArch = arch.parse()?;
    let ltoir = compile_nvvm_ir_to_ltoir_parsed(nvvm_ir, module_name, &arch)?;
    let ltoir_name = format!("{module_name}.ltoir");
    link_ltoir_to_ptx_parsed(&ltoir, &ltoir_name, &arch)
}

/// Link a single LTOIR payload to a loadable cubin image in memory.
pub fn link_ltoir_to_cubin(
    ltoir: &[u8],
    module_name: &str,
    arch: &str,
) -> Result<Vec<u8>, LtoirError> {
    let arch: CudaArch = arch.parse()?;
    link_ltoir_to_cubin_parsed(ltoir, module_name, &arch)
}

/// Link one LTOIR payload to forward-compatible PTX.
///
/// `arch` must be the target recorded with the LTOIR payload. The returned PTX
/// keeps that target and is JIT-compiled by the CUDA driver for the current GPU.
pub fn link_ltoir_to_ptx(
    ltoir: &[u8],
    module_name: &str,
    arch: &str,
) -> Result<Vec<u8>, LtoirError> {
    let arch: CudaArch = arch.parse()?;
    link_ltoir_to_ptx_parsed(ltoir, module_name, &arch)
}

fn link_ltoir_to_cubin_parsed(
    ltoir: &[u8],
    module_name: &str,
    arch: &CudaArch,
) -> Result<Vec<u8>, LtoirError> {
    let nvj = LibNvJitLink::load()?;
    let arch_opt = format!("-arch={}", arch.sm());
    let mut linker = Linker::new(&nvj, &[&arch_opt, "-lto"])?;
    linker.add(InputType::Ltoir, ltoir, module_name)?;
    Ok(linker.finish()?)
}

fn link_ltoir_to_ptx_parsed(
    ltoir: &[u8],
    module_name: &str,
    arch: &CudaArch,
) -> Result<Vec<u8>, LtoirError> {
    let nvj = LibNvJitLink::load()?;
    let arch_opt = format!("-arch={}", arch.sm());
    let mut linker = Linker::new(&nvj, &[&arch_opt, "-lto", "-ptx"])?;
    linker.add(InputType::Ltoir, ltoir, module_name)?;
    Ok(linker.finish_ptx()?)
}

fn compile_nvvm_ir_to_ltoir_parsed(
    nvvm_ir: &[u8],
    module_name: &str,
    arch: &CudaArch,
) -> Result<Vec<u8>, LtoirError> {
    let arch_compute = arch.compute();

    let libdevice_path = find_libdevice()?;
    let libdevice_bytes = std::fs::read(&libdevice_path).map_err(|source| LtoirError::Io {
        path: libdevice_path.clone(),
        source,
    })?;

    // ---- libNVVM: NVVM IR + libdevice -> LTOIR --------------------------
    let nvvm = LibNvvm::load()?;
    let ir_version = nvvm.ir_version()?;
    if (ir_version.ir_major, ir_version.ir_minor) != (2, 0) {
        return Err(LtoirError::UnsupportedNvvmIrVersion {
            major: ir_version.ir_major,
            minor: ir_version.ir_minor,
        });
    }
    if let Some(llvm_major) = nvvm.llvm_version(arch)? {
        let mismatch = if arch.uses_legacy_llvm() {
            llvm_major != 7
        } else {
            llvm_major == 7
        };
        if mismatch {
            return Err(LtoirError::DialectMismatch {
                target: arch.compute(),
                llvm_major,
                expected: if arch.uses_legacy_llvm() {
                    "legacy LLVM 7"
                } else {
                    "modern opaque-pointer"
                },
            });
        }
    }
    let mut prog = Program::new(&nvvm)?;
    // Add libdevice first so the kernel module's __nv_* references are
    // resolved at compile time. Order doesn't strictly matter -- libNVVM
    // does its own symbol resolution -- but this matches the pattern used
    // by NVCC and the device_ffi_test C tools.
    prog.add_module(&libdevice_bytes, "libdevice.10.bc")?;
    prog.add_module(nvvm_ir, module_name)?;

    let arch_opt = format!("-arch={arch_compute}");
    prog.verify(&[&arch_opt])?;
    Ok(prog.compile(&[&arch_opt, "-gen-lto"])?)
}

// ============================================================================
// Convenience: pick the right artifact and load it
// ============================================================================

/// Convenience wrapper: load a kernel module by `name` from the binary's
/// own directory, building the cubin on demand if cuda-oxide emitted NVVM IR.
///
/// Lookup order, inside `CARGO_MANIFEST_DIR` (the directory containing the
/// executable's `Cargo.toml`, where cuda-oxide writes its build artifacts):
///
/// 1. A target-recorded `<name>.ll` or `<name>.ltoir` -- `<name>.target`
///    identifies NVVM output and gives it precedence over older PTX.
/// 2. `<name>.ptx` -- the PTX path when no NVVM target file exists.
/// 3. An unrecorded `<name>.ll` / `<name>.ltoir` -- accepted only when
///    `CUDA_OXIDE_TARGET` explicitly supplies its original source target.
/// 4. `<name>.cubin` -- a standalone cubin when no source artifact exists.
///
/// If none of the four exist, returns [`LtoirError::NoArtifact`].
///
/// Use [`build_cubin_from_ll`] directly if you need explicit control over
/// the path or arch.
pub fn load_kernel_module(
    ctx: &Arc<CudaContext>,
    name: &str,
) -> Result<Arc<CudaModule>, LtoirError> {
    let dir = manifest_dir();
    let cubin = dir.join(format!("{name}.cubin"));
    let ptx = dir.join(format!("{name}.ptx"));
    let ll = dir.join(format!("{name}.ll"));
    let ltoir = dir.join(format!("{name}.ltoir"));

    let has_recorded_nvvm_target =
        emitted_target_path(&ll)
            .try_exists()
            .map_err(|source| LtoirError::Io {
                path: emitted_target_path(&ll),
                source,
            })?;
    let selected = select_file_artifact(
        ptx.exists(),
        ll.exists(),
        ltoir.exists(),
        cubin.exists(),
        has_recorded_nvvm_target,
    )
    .ok_or_else(|| LtoirError::NoArtifact {
        name: name.to_string(),
        dir,
    })?;

    match selected {
        FileArtifact::Ptx => Ok(ctx.load_module_from_file(
            ptx.to_str()
                .expect("kernel artifact path is not valid UTF-8"),
        )?),
        FileArtifact::NvvmIr => {
            let emitted = target_arch_for_artifact(&ll)?;
            let execution = execution_arch_for_context(ctx)?;
            match execution_route(&emitted, &execution)? {
                ExecutionRoute::Cubin => {
                    let cubin = build_cubin_from_ll(&ll, &emitted.sm())?;
                    Ok(ctx.load_module_from_file(
                        cubin
                            .to_str()
                            .expect("kernel artifact path is not valid UTF-8"),
                    )?)
                }
                ExecutionRoute::PtxBridge => {
                    let nvvm_ir = read_artifact(&ll)?;
                    let ptx =
                        build_ptx_from_nvvm_ir(&nvvm_ir, &ll.display().to_string(), &emitted.sm())?;
                    Ok(ctx.load_module_from_image(&ptx)?)
                }
            }
        }
        FileArtifact::Ltoir => {
            let emitted = target_arch_for_artifact(&ltoir)?;
            let execution = execution_arch_for_context(ctx)?;
            let bytes = read_artifact(&ltoir)?;
            let image = match execution_route(&emitted, &execution)? {
                ExecutionRoute::Cubin => {
                    link_ltoir_to_cubin_parsed(&bytes, &ltoir.display().to_string(), &emitted)?
                }
                ExecutionRoute::PtxBridge => {
                    link_ltoir_to_ptx_parsed(&bytes, &ltoir.display().to_string(), &emitted)?
                }
            };
            Ok(ctx.load_module_from_image(&image)?)
        }
        FileArtifact::Cubin => Ok(ctx.load_module_from_file(
            cubin
                .to_str()
                .expect("kernel artifact path is not valid UTF-8"),
        )?),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FileArtifact {
    Ptx,
    NvvmIr,
    Ltoir,
    Cubin,
}

/// Select one file artifact. A `.target` file identifies NVVM IR/LTOIR as the
/// current output; without one, PTX takes precedence.
fn select_file_artifact(
    has_ptx: bool,
    has_nvvm_ir: bool,
    has_ltoir: bool,
    has_cubin: bool,
    has_recorded_nvvm_target: bool,
) -> Option<FileArtifact> {
    if has_recorded_nvvm_target {
        if has_nvvm_ir {
            return Some(FileArtifact::NvvmIr);
        }
        if has_ltoir {
            return Some(FileArtifact::Ltoir);
        }
        // The target file identifies NVVM output, but that output is missing.
        // Do not fall back to an older PTX or cubin.
        return None;
    }

    if has_ptx {
        Some(FileArtifact::Ptx)
    } else if has_nvvm_ir {
        // Older or manual NVVM IR requires an explicit CUDA_OXIDE_TARGET.
        Some(FileArtifact::NvvmIr)
    } else if has_ltoir {
        Some(FileArtifact::Ltoir)
    } else if has_cubin {
        Some(FileArtifact::Cubin)
    } else {
        None
    }
}

// ============================================================================
// Discovery helpers (libdevice, arch, manifest dir)
// ============================================================================

/// Locate `libdevice.10.bc` from the CUDA Toolkit.
///
/// Search order:
/// 1. `CUDA_OXIDE_LIBDEVICE` env var (used as-is if it points to an
///    existing file).
/// 2. `<root>/nvvm/libdevice/libdevice.10.bc` for `<root>` in
///    `CUDA_TOOLKIT_PATH`, `CUDA_HOME`, `CUDA_PATH`, `/usr/local/cuda`,
///    `/opt/cuda`.
///
/// Returns [`LtoirError::LibdeviceNotFound`] with the full list of probed
/// paths if nothing matches.
///
/// Thin wrapper over [`libnvvm_sys::find_libdevice`], which owns the probe
/// (libdevice ships in the toolkit's `nvvm/` component next to `libnvvm.so`).
pub fn find_libdevice() -> Result<PathBuf, LtoirError> {
    libnvvm_sys::find_libdevice()
        .map_err(|libnvvm_sys::LibdeviceNotFound { tried }| LtoirError::LibdeviceNotFound { tried })
}

/// Read and validate an explicit source target from `CUDA_OXIDE_TARGET`.
///
/// Artifact loaders prefer the target recorded with the artifact. Older
/// artifacts without a `.target` file require this explicit value because the
/// current GPU does not reveal which target was used to produce existing IR.
pub fn target_arch() -> Result<String, LtoirError> {
    let explicit = std::env::var("CUDA_OXIDE_TARGET").ok();
    resolve_source_target(None, explicit.as_deref()).map(|target| target.sm())
}

/// Query the physical execution GPU. Target-related environment variables
/// describe the input artifact, not the GPU that will execute it.
pub(crate) fn execution_arch_for_context(ctx: &CudaContext) -> Result<CudaArch, LtoirError> {
    let (major, minor) = ctx.compute_capability()?;
    format!("sm_{major}{minor}")
        .parse::<CudaArch>()
        .map_err(LtoirError::from)
}

/// Directory to search for kernel artifacts (`.cubin` / `.ptx` / `.ll` /
/// `.ltoir`).
///
/// Reads `CARGO_MANIFEST_DIR`, which `cargo run` sets to the directory of
/// the executable's `Cargo.toml` -- the same directory cuda-oxide writes
/// its build artifacts to. Falls back to the current working directory if
/// the env var is unset (e.g. when the binary is launched outside cargo).
///
/// Note: `env!("CARGO_MANIFEST_DIR")` cannot be used here because it
/// resolves to *this* crate's manifest dir at compile time, not the
/// downstream binary's.
fn manifest_dir() -> PathBuf {
    if let Ok(d) = std::env::var("CARGO_MANIFEST_DIR") {
        return PathBuf::from(d);
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

// ============================================================================
// Internal utilities
// ============================================================================

fn target_arch_for_artifact(artifact_path: &Path) -> Result<CudaArch, LtoirError> {
    let explicit = std::env::var("CUDA_OXIDE_TARGET").ok();
    target_arch_for_artifact_with_explicit(artifact_path, explicit.as_deref())
}

fn target_arch_for_artifact_with_explicit(
    artifact_path: &Path,
    explicit_target: Option<&str>,
) -> Result<CudaArch, LtoirError> {
    resolve_source_target(read_emitted_target(artifact_path)?, explicit_target)
}

/// Resolve the original target from artifact metadata or an explicit value.
/// The execution GPU is not used to infer the artifact's target.
pub(crate) fn resolve_source_target(
    recorded_target: Option<CudaArch>,
    explicit_target: Option<&str>,
) -> Result<CudaArch, LtoirError> {
    if let Some(target) = recorded_target {
        return Ok(target);
    }
    explicit_target
        .ok_or(LtoirError::TargetNotFound)?
        .parse()
        .map_err(LtoirError::from)
}

fn emitted_target_path(ll_path: &Path) -> PathBuf {
    ll_path.with_extension("target")
}

fn read_emitted_target(ll_path: &Path) -> Result<Option<CudaArch>, LtoirError> {
    let path = emitted_target_path(ll_path);
    match std::fs::read_to_string(&path) {
        Ok(target) => Ok(Some(target.trim().parse()?)),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(LtoirError::Io { path, source }),
    }
}

fn validate_ir_target_sidecar(ll_path: &Path, requested: &CudaArch) -> Result<(), LtoirError> {
    let Some(emitted) = read_emitted_target(ll_path)? else {
        return Ok(());
    };
    if emitted != *requested {
        return Err(LtoirError::TargetMismatch {
            ir_path: ll_path.to_path_buf(),
            emitted: emitted.sm(),
            requested: requested.sm(),
        });
    }
    Ok(())
}

fn write_artifact_target_sidecar(
    artifact_path: &Path,
    target: &CudaArch,
) -> Result<(), LtoirError> {
    let path = emitted_target_path(artifact_path);
    std::fs::write(&path, format!("{}\n", target.sm()))
        .map_err(|source| LtoirError::Io { path, source })
}

fn read_artifact(path: &Path) -> Result<Vec<u8>, LtoirError> {
    std::fs::read(path).map_err(|source| LtoirError::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExecutionRoute {
    /// Produce native code for the architecture that emitted the IR.
    Cubin,
    /// Keep the original virtual architecture in PTX and let the driver JIT it
    /// for the newer GPU.
    PtxBridge,
}

/// Select an execution route without changing either architecture.
pub(crate) fn execution_route(
    emitted: &CudaArch,
    execution: &CudaArch,
) -> Result<ExecutionRoute, LtoirError> {
    if emitted.capability() == execution.capability() {
        return Ok(ExecutionRoute::Cubin);
    }

    let incompatible = |reason| LtoirError::IncompatibleExecutionTarget {
        emitted: emitted.sm(),
        execution: execution.sm(),
        reason,
    };

    if emitted.suffix().is_some() {
        return Err(incompatible(
            "targets with an architecture suffix, such as sm_90a, cannot be forwarded to a different GPU",
        ));
    }
    if emitted.capability() > execution.capability() {
        return Err(incompatible(
            "an artifact built for a newer GPU cannot run on an older GPU",
        ));
    }
    if emitted.uses_legacy_llvm() && execution.capability() >= 100 {
        return Ok(ExecutionRoute::PtxBridge);
    }

    Err(incompatible(
        "only standard pre-Blackwell targets, such as sm_86, can be converted to PTX for Blackwell",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cuda_oxide_ltoir_{name}_{}_{}",
            std::process::id(),
            unique
        ))
    }

    #[test]
    fn recorded_target_takes_precedence_over_environment() {
        let dir = temp_dir("target_sidecar");
        std::fs::create_dir_all(&dir).unwrap();
        let ll = dir.join("kernel.ll");
        std::fs::write(&ll, "; test\n").unwrap();
        std::fs::write(dir.join("kernel.target"), "compute_86\n").unwrap();

        let emitted = read_emitted_target(&ll).unwrap().unwrap();
        assert_eq!(emitted.sm(), "sm_86");
        assert_eq!(
            target_arch_for_artifact_with_explicit(&ll, Some("sm_120"))
                .unwrap()
                .sm(),
            "sm_86",
            "the recorded target must take precedence"
        );
        validate_ir_target_sidecar(&ll, &"sm_86".parse().unwrap()).unwrap();

        let mismatch = validate_ir_target_sidecar(&ll, &"sm_120".parse().unwrap())
            .expect_err("cross-target reuse must fail");
        assert!(matches!(mismatch, LtoirError::TargetMismatch { .. }));

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn artifact_without_target_requires_explicit_target() {
        let dir = temp_dir("missing_target");
        std::fs::create_dir_all(&dir).unwrap();
        let ll = dir.join("kernel.ll");
        std::fs::write(&ll, "; target-specific NVVM IR\n").unwrap();

        let error = target_arch_for_artifact_with_explicit(&ll, None)
            .expect_err("the original build target is required");
        assert!(matches!(error, LtoirError::TargetNotFound));

        let asserted = target_arch_for_artifact_with_explicit(&ll, Some("compute_86")).unwrap();
        assert_eq!(asserted.sm(), "sm_86");

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn explicit_build_target_is_recorded_for_sibling_ltoir() {
        let dir = temp_dir("record_explicit_target");
        std::fs::create_dir_all(&dir).unwrap();
        let ll = dir.join("kernel.ll");
        std::fs::write(&ll, "; manually supplied NVVM IR\n").unwrap();
        let arch: CudaArch = "compute_86".parse().unwrap();

        validate_ir_target_sidecar(&ll, &arch).unwrap();
        write_artifact_target_sidecar(&ll, &arch).unwrap();

        assert_eq!(
            std::fs::read_to_string(dir.join("kernel.target")).unwrap(),
            "sm_86\n"
        );
        let ltoir = dir.join("kernel.ltoir");
        std::fs::write(&ltoir, b"placeholder").unwrap();
        std::fs::remove_file(&ll).unwrap();
        assert_eq!(
            target_arch_for_artifact_with_explicit(&ltoir, None)
                .unwrap()
                .sm(),
            "sm_86"
        );

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn recorded_nvvm_output_takes_precedence_over_stale_ptx() {
        assert_eq!(
            select_file_artifact(true, true, false, true, true),
            Some(FileArtifact::NvvmIr)
        );
        assert_eq!(
            select_file_artifact(true, false, true, true, true),
            Some(FileArtifact::Ltoir)
        );
        assert_eq!(
            select_file_artifact(true, false, false, true, true),
            None,
            "a missing NVVM artifact must not fall back to older output"
        );

        // In an ordinary PTX build there is also an LLVM `.ll`, but no NVVM
        // target sidecar. PTX is therefore the current loadable artifact.
        assert_eq!(
            select_file_artifact(true, true, false, true, false),
            Some(FileArtifact::Ptx)
        );
    }

    #[test]
    fn file_artifact_selection_accepts_older_unrecorded_nvvm_output() {
        assert_eq!(
            select_file_artifact(false, true, false, true, false),
            Some(FileArtifact::NvvmIr)
        );
        assert_eq!(
            select_file_artifact(false, false, true, true, false),
            Some(FileArtifact::Ltoir)
        );
        assert_eq!(
            select_file_artifact(false, false, false, true, false),
            Some(FileArtifact::Cubin)
        );
        assert_eq!(
            select_file_artifact(false, false, false, false, false),
            None
        );
    }

    #[test]
    fn missing_ll_cannot_reuse_old_cubin_or_publish_target() {
        let dir = temp_dir("missing_ll_old_cubin");
        std::fs::create_dir_all(&dir).unwrap();
        let ll = dir.join("kernel.ll");
        std::fs::write(dir.join("kernel.cubin"), b"old cubin").unwrap();

        let error = build_cubin_from_ll(&ll, "sm_86")
            .expect_err("a derived cubin is invalid without its source NVVM IR");
        assert!(matches!(error, LtoirError::Io { path, .. } if path == ll));
        assert!(
            !dir.join("kernel.target").exists(),
            "a missing source must not gain target metadata"
        );

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn invalid_link_target_fails_before_loading_nvidia_libraries() {
        let error = link_ltoir_to_cubin(&[], "empty", "nvvm-ir").unwrap_err();
        assert!(matches!(error, LtoirError::InvalidTarget(_)));

        let error = link_ltoir_to_ptx(&[], "empty", "nvvm-ir").unwrap_err();
        assert!(matches!(error, LtoirError::InvalidTarget(_)));
    }

    #[test]
    fn same_target_keeps_native_cubin_route() {
        for (emitted, execution) in [
            ("sm_86", "sm_86"),
            ("compute_120", "sm_120"),
            // The suffix is preserved in the nvJitLink target. The CUDA
            // driver remains responsible for accepting it on this same
            // numeric device architecture.
            ("sm_90a", "sm_90"),
        ] {
            assert_eq!(
                execution_route(&emitted.parse().unwrap(), &execution.parse().unwrap()).unwrap(),
                ExecutionRoute::Cubin,
                "{emitted} on {execution}"
            );
        }
    }

    #[test]
    fn standard_legacy_target_bridges_forward_to_blackwell_as_ptx() {
        for (emitted, execution) in [
            ("sm_75", "sm_100"),
            ("compute_86", "sm_120"),
            ("sm_90", "sm_120"),
        ] {
            assert_eq!(
                execution_route(&emitted.parse().unwrap(), &execution.parse().unwrap()).unwrap(),
                ExecutionRoute::PtxBridge,
                "{emitted} on {execution}"
            );
        }
    }

    #[test]
    fn ptx_bridge_rejects_suffixed_and_unsupported_forward_targets() {
        for (emitted, execution) in [
            ("sm_90a", "sm_120"),
            ("sm_90f", "sm_120"),
            ("sm_86", "sm_90"),
            ("sm_100", "sm_120"),
        ] {
            let error = execution_route(&emitted.parse().unwrap(), &execution.parse().unwrap())
                .expect_err("cross-target route must be explicitly supported");
            assert!(
                matches!(error, LtoirError::IncompatibleExecutionTarget { .. }),
                "{emitted} on {execution}: {error}"
            );
        }
    }

    #[test]
    fn execution_route_rejects_backward_targets() {
        let error = execution_route(&"sm_120".parse().unwrap(), &"sm_86".parse().unwrap())
            .expect_err("a newer artifact cannot run on an older GPU");

        let LtoirError::IncompatibleExecutionTarget {
            emitted,
            execution,
            reason,
        } = error
        else {
            panic!("unexpected error: {error}");
        };
        assert_eq!(emitted, "sm_120");
        assert_eq!(execution, "sm_86");
        assert!(reason.contains("newer GPU"));
    }
}
