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
//! 2. Link the resulting LTOIR via nvJitLink to produce a cubin.
//! 3. Load the cubin via [`cuda_core::CudaContext::load_module_from_file`].
//!
//! This module wraps that pipeline behind two helpers:
//!
//! - [`build_cubin_from_ll`] -- explicit form, takes a `.ll` path and arch.
//! - [`load_kernel_module`] -- the convenience form. Looks at the example's
//!   directory and loads `<name>.cubin`, `<name>.ptx`, or builds the cubin
//!   from `<name>.ll` automatically. **This is the one most callers want.**
//!
//! All work is done via [`libnvvm_sys`] and [`nvjitlink_sys`] (`dlopen` of
//! `libnvvm.so` and `libnvJitLink.so` from the CUDA Toolkit). No external
//! C tools are required, no symlinked `tools/` directory, no boilerplate.
//!
//! # Discovery
//!
//! - **libNVVM**: `LIBNVVM_PATH` env var, then system loader, then
//!   `<root>/nvvm/lib64/libnvvm.so` for `<root>` in `CUDA_HOME`,
//!   `CUDA_PATH`, `/usr/local/cuda`, `/opt/cuda`.
//! - **nvJitLink**: same, but at `<root>/lib64/libnvJitLink.so`.
//! - **libdevice**: `CUDA_OXIDE_LIBDEVICE` env var, then
//!   `<root>/nvvm/libdevice/libdevice.10.bc` for the same roots.
//! - **Arch**: `CUDA_OXIDE_TARGET` (set by `cargo oxide`'s `--arch=<sm_XX>`),
//!   then the `CUDA_OXIDE_DEVICE_ARCH` hint (auto-detected GPU arch), then a
//!   `sm_120` default.
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
use libnvvm_sys::{LibNvvm, NvvmError, Program};
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
        "Could not locate libdevice.10.bc. Set CUDA_OXIDE_LIBDEVICE or CUDA_HOME, or install the CUDA Toolkit. Tried:\n  {tried}"
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

    /// [`load_kernel_module`] could not find any of `<name>.cubin`,
    /// `<name>.ptx`, or `<name>.ll` in the binary's manifest directory.
    #[error(
        "Could not find any kernel artifact for {name} in {dir}. \
         Looked for {name}.cubin, {name}.ptx, {name}.ll. \
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
/// `arch` is the GPU SM target (e.g. `"sm_120"`); it is rewritten to
/// `compute_XX` for the libNVVM compile and passed verbatim for the
/// nvJitLink link. If `arch` does not start with `sm_` it is passed
/// through unchanged.
///
/// # Caching
///
/// If `<stem>.cubin` already exists and is newer than `ll_path`, the
/// existing cubin path is returned and no work is done. Touch the `.ll`
/// (or delete the `.cubin`) to force a rebuild.
pub fn build_cubin_from_ll(ll_path: &Path, arch: &str) -> Result<PathBuf, LtoirError> {
    let stem = ll_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("kernel");
    let dir = ll_path.parent().unwrap_or_else(|| Path::new("."));
    let ltoir_path = dir.join(format!("{stem}.ltoir"));
    let cubin_path = dir.join(format!("{stem}.cubin"));

    // Cache: skip work if cubin is fresher than the .ll input.
    if !needs_rebuild(&cubin_path, &[ll_path]) {
        return Ok(cubin_path);
    }

    let ll_bytes = std::fs::read(ll_path).map_err(|source| LtoirError::Io {
        path: ll_path.to_path_buf(),
        source,
    })?;

    let ltoir = compile_nvvm_ir_to_ltoir(&ll_bytes, &ll_path.display().to_string(), arch)?;

    std::fs::write(&ltoir_path, &ltoir).map_err(|source| LtoirError::Io {
        path: ltoir_path.clone(),
        source,
    })?;

    // ---- nvJitLink: LTOIR -> cubin --------------------------------------
    let cubin = link_ltoir_to_cubin(&ltoir, &ltoir_path.display().to_string(), arch)?;

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
    let ltoir = compile_nvvm_ir_to_ltoir(nvvm_ir, module_name, arch)?;
    let ltoir_name = format!("{module_name}.ltoir");
    link_ltoir_to_cubin(&ltoir, &ltoir_name, arch)
}

/// Link a single LTOIR payload to a loadable cubin image in memory.
pub fn link_ltoir_to_cubin(
    ltoir: &[u8],
    module_name: &str,
    arch: &str,
) -> Result<Vec<u8>, LtoirError> {
    let nvj = LibNvJitLink::load()?;
    let arch_opt = format!("-arch={arch}");
    let mut linker = Linker::new(&nvj, &[&arch_opt, "-lto"])?;
    linker.add(InputType::Ltoir, ltoir, module_name)?;
    Ok(linker.finish()?)
}

fn compile_nvvm_ir_to_ltoir(
    nvvm_ir: &[u8],
    module_name: &str,
    arch: &str,
) -> Result<Vec<u8>, LtoirError> {
    let libdevice_path = find_libdevice()?;
    let libdevice_bytes = std::fs::read(&libdevice_path).map_err(|source| LtoirError::Io {
        path: libdevice_path.clone(),
        source,
    })?;

    let arch_compute = sm_to_compute(arch);

    // ---- libNVVM: NVVM IR + libdevice -> LTOIR --------------------------
    let nvvm = LibNvvm::load()?;
    let mut prog = Program::new(&nvvm)?;
    // Add libdevice first so the kernel module's __nv_* references are
    // resolved at compile time. Order doesn't strictly matter -- libNVVM
    // does its own symbol resolution -- but this matches the pattern used
    // by NVCC and the device_ffi_test C tools.
    prog.add_module(&libdevice_bytes, "libdevice.10.bc")?;
    prog.add_module(nvvm_ir, module_name)?;

    let arch_opt = format!("-arch={arch_compute}");
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
/// 1. `<name>.cubin` -- already built; load directly.
/// 2. `<name>.ptx` -- standard PTX path; load directly.
/// 3. `<name>.ll` -- NVVM IR (cuda-oxide auto-detected libdevice). Build a
///    cubin via [`build_cubin_from_ll`] using the arch from
///    [`target_arch`], then load it.
///
/// If none of the three exist, returns [`LtoirError::NoArtifact`].
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

    let to_load = if cubin.exists() {
        cubin
    } else if ptx.exists() {
        ptx
    } else if ll.exists() {
        let arch = target_arch();
        build_cubin_from_ll(&ll, &arch)?
    } else {
        return Err(LtoirError::NoArtifact {
            name: name.to_string(),
            dir,
        });
    };

    Ok(ctx.load_module_from_file(
        to_load
            .to_str()
            .expect("kernel artifact path is not valid UTF-8"),
    )?)
}

// ============================================================================
// Discovery helpers (libdevice, arch, manifest dir)
// ============================================================================

/// Locate `libdevice.10.bc` from the CUDA Toolkit.
///
/// Search order:
/// 1. `CUDA_OXIDE_LIBDEVICE` env var (used as-is if it points to an
///    existing file).
/// 2. `<root>/nvvm/libdevice/libdevice.10.bc` for `<root>` in `CUDA_HOME`,
///    `CUDA_PATH`, `/usr/local/cuda`, `/opt/cuda`.
///
/// Returns [`LtoirError::LibdeviceNotFound`] with the full list of probed
/// paths if nothing matches.
pub fn find_libdevice() -> Result<PathBuf, LtoirError> {
    if let Ok(p) = std::env::var("CUDA_OXIDE_LIBDEVICE") {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
    }
    let mut tried = Vec::new();
    for root in cuda_roots() {
        let candidate = root.join("nvvm/libdevice/libdevice.10.bc");
        tried.push(candidate.display().to_string());
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(LtoirError::LibdeviceNotFound {
        tried: tried.join("\n  "),
    })
}

/// Read the GPU arch (`sm_XX`) for the cubin build, defaulting to `sm_120`
/// (consumer Blackwell, RTX 5090) when nothing else is set.
///
/// Resolution order:
/// - `CUDA_OXIDE_TARGET` -- an explicit pin. `cargo oxide run --arch=<arch>`
///   sets it for the spawned binary, so `--arch=sm_90` yields `"sm_90"`.
/// - `CUDA_OXIDE_DEVICE_ARCH` -- the auto-detected arch of the GPU in this
///   machine, forwarded by `cargo oxide run` when no `--arch` was given.
/// - `sm_120` fallback.
pub fn target_arch() -> String {
    std::env::var("CUDA_OXIDE_TARGET")
        .or_else(|_| std::env::var("CUDA_OXIDE_DEVICE_ARCH"))
        .unwrap_or_else(|_| "sm_120".to_string())
}

/// Directory to search for kernel artifacts (`.cubin` / `.ptx` / `.ll`).
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

fn cuda_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(r) = std::env::var(var) {
            roots.push(PathBuf::from(r));
        }
    }
    roots.push(PathBuf::from("/usr/local/cuda"));
    roots.push(PathBuf::from("/opt/cuda"));
    roots
}

/// Convert `sm_120` to `compute_120`. Returns the input unchanged if it
/// doesn't start with `sm_`.
fn sm_to_compute(arch: &str) -> String {
    if let Some(rest) = arch.strip_prefix("sm_") {
        format!("compute_{rest}")
    } else {
        arch.to_string()
    }
}

/// `true` if `target` is missing or older than any source in `sources`.
fn needs_rebuild(target: &Path, sources: &[&Path]) -> bool {
    let Ok(target_meta) = target.metadata() else {
        return true;
    };
    let Ok(target_time) = target_meta.modified() else {
        return true;
    };
    for src in sources {
        if let Ok(src_meta) = src.metadata()
            && let Ok(src_time) = src_meta.modified()
            && src_time > target_time
        {
            return true;
        }
    }
    false
}
