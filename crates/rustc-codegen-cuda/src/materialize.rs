/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Opt-in build-time compilation of NVVM IR / LTOIR device artifacts to a
//! final cubin, so the *embedded* payload is directly loadable by the CUDA
//! driver.
//!
//! By default the backend embeds NVVM IR and LTOIR artifacts as-is and
//! `cuda-host` compiles them at first load, choosing between a native cubin
//! and a forward-compatible PTX bridge based on the GPU actually present
//! (see `cuda-host`'s `ExecutionRoute`). Setting
//! `CUDA_OXIDE_MATERIALIZE_CUBIN` moves that compilation to host build time
//! instead:
//!
//! - deployment hosts no longer need libNVVM or nvJitLink, and there is no
//!   first-load compile hit;
//! - the build host must have libNVVM and nvJitLink available;
//! - the embedded cubin is pinned to the emitted architecture — the
//!   load-time PTX bridge to newer GPUs no longer applies. Use the default
//!   when one binary must serve multiple GPU architectures.
//!
//! This module talks to `libnvvm-sys` / `nvjitlink-sys` directly (both load
//! their libraries at call time via `dlopen`) rather than going through
//! `cuda-host`, which would link the backend dylib against the CUDA driver
//! (`libcuda.so.1`) and break building or loading the backend on machines
//! without a GPU driver. The compile sequence mirrors `cuda-host`'s
//! `ltoir` module and `cargo-oxide`'s `emit-ltoir`: validate the libNVVM
//! frontend, add `libdevice.10.bc`, compile with `-gen-lto`, then link the
//! LTOIR to a cubin with nvJitLink.

use libnvvm_sys::{CudaArch, CudaArchParseError, LibNvvm, NvvmError, Program};
use nvjitlink_sys::{InputType, LibNvJitLink, Linker, NvJitLinkError};
use thiserror::Error;

/// Whether the user asked for build-time cubin materialization.
///
/// Presence-based, mirroring `CUDA_OXIDE_EMIT_NVVM_IR`.
pub(crate) fn materialize_cubin_enabled() -> bool {
    std::env::var("CUDA_OXIDE_MATERIALIZE_CUBIN").is_ok()
}

/// Failures while materializing an embedded artifact to a cubin.
#[derive(Debug, Error)]
pub(crate) enum MaterializeError {
    /// The recorded artifact target was not a concrete CUDA architecture.
    #[error(transparent)]
    InvalidTarget(#[from] CudaArchParseError),

    /// The installed toolkit does not accept the NVVM IR version cuda-oxide emits.
    #[error("installed libNVVM accepts NVVM IR {major}.{minor}, but cuda-oxide emits NVVM IR 2.0")]
    UnsupportedNvvmIrVersion { major: i32, minor: i32 },

    /// Toolkit dialect discovery disagreed with cuda-oxide's target policy.
    #[error(
        "libNVVM reports LLVM {llvm_major} for {target}, which disagrees with cuda-oxide's expected {expected} dialect"
    )]
    DialectMismatch {
        target: String,
        llvm_major: i32,
        expected: &'static str,
    },

    /// libNVVM failed (load, symbol resolution, or compile call).
    #[error("libnvvm: {0}")]
    Nvvm(#[from] NvvmError),

    /// nvJitLink failed (load, symbol resolution, or link call).
    #[error("nvJitLink: {0}")]
    NvJitLink(#[from] NvJitLinkError),

    /// `libdevice.10.bc` could not be located or read.
    #[error("CUDA_OXIDE_MATERIALIZE_CUBIN requires libdevice.10.bc on the build host. {0}")]
    Libdevice(#[from] libnvvm_sys::LibdeviceNotFound),

    /// Reading `libdevice.10.bc` failed.
    #[error("failed reading {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
}

/// Compile an NVVM IR payload to a cubin for the architecture it was
/// emitted for.
pub(crate) fn nvvm_ir_to_cubin(
    nvvm_ir: &[u8],
    module_name: &str,
    target: &str,
) -> Result<Vec<u8>, MaterializeError> {
    let arch: CudaArch = target.parse()?;
    let ltoir = compile_nvvm_ir_to_ltoir(nvvm_ir, module_name, &arch)?;
    link_ltoir(&ltoir, &format!("{module_name}.ltoir"), &arch)
}

/// Link an LTOIR payload to a cubin for the architecture it was emitted for.
pub(crate) fn ltoir_to_cubin(
    ltoir: &[u8],
    module_name: &str,
    target: &str,
) -> Result<Vec<u8>, MaterializeError> {
    let arch: CudaArch = target.parse()?;
    link_ltoir(ltoir, module_name, &arch)
}

fn compile_nvvm_ir_to_ltoir(
    nvvm_ir: &[u8],
    module_name: &str,
    arch: &CudaArch,
) -> Result<Vec<u8>, MaterializeError> {
    let nvvm = LibNvvm::load()?;
    validate_nvvm_frontend(&nvvm, arch)?;

    let libdevice_path = libnvvm_sys::find_libdevice()?;
    let libdevice = std::fs::read(&libdevice_path).map_err(|source| MaterializeError::Io {
        path: libdevice_path,
        source,
    })?;

    let mut prog = Program::new(&nvvm)?;
    // Add libdevice first so the kernel module's __nv_* references are
    // resolved at compile time, matching cuda-host's ltoir pipeline.
    prog.add_module(&libdevice, "libdevice.10.bc")?;
    prog.add_module(nvvm_ir, module_name)?;

    let arch_opt = format!("-arch={}", arch.compute());
    prog.verify(&[&arch_opt])?;
    Ok(prog.compile(&[&arch_opt, "-gen-lto"])?)
}

/// Frontend validation identical to cuda-host's `ltoir` module: cuda-oxide
/// emits NVVM IR 2.0, and the LLVM dialect libNVVM reports for the target
/// must match the dialect the backend generated for it.
fn validate_nvvm_frontend(nvvm: &LibNvvm, arch: &CudaArch) -> Result<(), MaterializeError> {
    let ir_version = nvvm.ir_version()?;
    if (ir_version.ir_major, ir_version.ir_minor) != (2, 0) {
        return Err(MaterializeError::UnsupportedNvvmIrVersion {
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
            return Err(MaterializeError::DialectMismatch {
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
    Ok(())
}

fn link_ltoir(
    ltoir: &[u8],
    module_name: &str,
    arch: &CudaArch,
) -> Result<Vec<u8>, MaterializeError> {
    let nvj = LibNvJitLink::load()?;
    let arch_opt = format!("-arch={}", arch.sm());
    let mut linker = Linker::new(&nvj, &[&arch_opt, "-lto"])?;
    linker.add(InputType::Ltoir, ltoir, module_name)?;
    Ok(linker.finish()?)
}
