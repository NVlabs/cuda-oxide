/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::options::FinalizationOptions;
use crate::provenance::{
    StableDigest, compiler_provenance_digest, digest_bytes, digest_file_handle, recipe_digest,
    with_revalidated_tool_identity,
};
use crate::{FinalizerError, validate_name};
use libnvvm_sys::{LibNvvm, Program, find_libdevice};
use std::sync::{Arc, Mutex, OnceLock};

struct LoadedNvvmTool {
    library: Arc<LibNvvm>,
    digest: Option<[u8; 32]>,
}

static NVVM_TOOL: OnceLock<Arc<LoadedNvvmTool>> = OnceLock::new();
static NVVM_TOOL_LOAD: OnceLock<Mutex<()>> = OnceLock::new();

/// Driver-independent libNVVM compiler with exact libdevice provenance.
#[derive(Clone)]
pub struct NvvmCompiler {
    tool: Arc<LoadedNvvmTool>,
    libdevice: Arc<[u8]>,
    libdevice_digest: [u8; 32],
}

impl NvvmCompiler {
    /// Discover and pin libNVVM, then read the selected libdevice bytes.
    pub fn discover() -> Result<Self, FinalizerError> {
        let path = find_libdevice().map_err(|libnvvm_sys::LibdeviceNotFound { tried }| {
            FinalizerError::LibdeviceNotFound { tried }
        })?;
        let libdevice = std::fs::read(&path).map_err(|source| FinalizerError::Io {
            path: path.clone(),
            source,
        })?;
        let libdevice_digest = digest_bytes(&libdevice);
        Ok(Self {
            tool: load_nvvm_tool()?,
            libdevice: libdevice.into(),
            libdevice_digest,
        })
    }

    /// Digest of the exact loaded libNVVM file, when its identity is known.
    pub fn libnvvm_digest(&self) -> Option<[u8; 32]> {
        let digest = self.tool.digest?;
        if self.tool.library.loaded_file_if_unchanged().is_some() {
            Some(digest)
        } else {
            report_changed_tool("libNVVM");
            None
        }
    }

    /// Digest of the exact libdevice bytes that will be compiled.
    pub fn libdevice_digest(&self) -> [u8; 32] {
        self.libdevice_digest
    }

    /// Exact route provenance, or `None` when the loaded DSO is unidentifiable.
    pub fn provenance_digest(&self) -> Option<[u8; 32]> {
        self.libnvvm_digest()
            .map(|digest| compiler_provenance_digest(&digest, &self.libdevice_digest))
    }

    /// Compile one NVVM IR module plus libdevice into LTOIR.
    pub fn compile_nvvm_ir_to_ltoir(
        &self,
        module_name: &str,
        nvvm_ir: &[u8],
        options: &FinalizationOptions,
    ) -> Result<Vec<u8>, FinalizerError> {
        validate_name(module_name)?;
        if nvvm_ir.is_empty() {
            return Err(FinalizerError::EmptyInput {
                name: module_name.to_string(),
            });
        }
        with_revalidated_tool_identity(
            "libNVVM",
            self.tool.digest,
            || current_nvvm_tool_digest(&self.tool),
            || {
                validate_nvvm_frontend(&self.tool.library, options)?;

                let mut program = Program::new(&self.tool.library)?;
                // libdevice must precede user IR so the plan, diagnostics, and
                // provenance all use one deterministic module order.
                program.add_module(&self.libdevice, "libdevice.10.bc")?;
                program.add_module(nvvm_ir, module_name)?;

                let verify = options.nvvm_verify_options();
                let verify_refs = verify.iter().map(String::as_str).collect::<Vec<_>>();
                program.verify(&verify_refs)?;
                let compile = options.nvvm_compile_options();
                let compile_refs = compile.iter().map(String::as_str).collect::<Vec<_>>();
                Ok(program.compile(&compile_refs)?)
            },
        )
    }

    /// Digest every semantic input to the NVVM IR to LTOIR stage.
    pub fn artifact_digest(
        &self,
        module_name: &str,
        nvvm_ir: &[u8],
        options: &FinalizationOptions,
    ) -> Option<[u8; 32]> {
        let libnvvm = self.libnvvm_digest()?;
        Some(nvvm_ir_artifact_digest_parts(
            module_name,
            nvvm_ir,
            options,
            &self.libdevice_digest,
            &libnvvm,
        ))
    }
}

fn current_nvvm_tool_digest(tool: &LoadedNvvmTool) -> Option<[u8; 32]> {
    let file = tool.library.loaded_file_if_unchanged()?;
    digest_file_handle(file).ok()
}

fn validate_nvvm_frontend(
    nvvm: &LibNvvm,
    options: &FinalizationOptions,
) -> Result<(), FinalizerError> {
    let ir_version = nvvm.ir_version()?;
    if (ir_version.ir_major, ir_version.ir_minor) != (2, 0) {
        return Err(FinalizerError::UnsupportedNvvmIrVersion {
            major: ir_version.ir_major,
            minor: ir_version.ir_minor,
        });
    }
    let arch = options.target();
    if let Some(llvm_major) = nvvm.llvm_version(arch)? {
        let mismatch = if arch.uses_legacy_llvm() {
            llvm_major != 7
        } else {
            llvm_major == 7
        };
        if mismatch {
            return Err(FinalizerError::DialectMismatch {
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

fn load_nvvm_tool() -> Result<Arc<LoadedNvvmTool>, FinalizerError> {
    if let Some(loaded) = NVVM_TOOL.get() {
        return Ok(Arc::clone(loaded));
    }
    let _guard = NVVM_TOOL_LOAD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(loaded) = NVVM_TOOL.get() {
        return Ok(Arc::clone(loaded));
    }

    let library = LibNvvm::load_for_cache()?;
    let digest = loaded_tool_digest("libNVVM", library.loaded_file_if_unchanged());
    let digest = if digest.is_some() && library.loaded_file_if_unchanged().is_none() {
        report_changed_tool("libNVVM");
        None
    } else {
        digest
    };
    let loaded = Arc::new(LoadedNvvmTool {
        library: Arc::new(library),
        digest,
    });
    let _ = NVVM_TOOL.set(Arc::clone(&loaded));
    Ok(loaded)
}

pub(crate) fn loaded_tool_digest(label: &str, file: Option<&std::fs::File>) -> Option<[u8; 32]> {
    let Some(file) = file else {
        if std::env::var_os("CUDA_OXIDE_VERBOSE").is_some() {
            eprintln!(
                "cuda-oxide: {label} has no exact loaded-file identity; disabling artifact reuse"
            );
        }
        return None;
    };
    match digest_file_handle(file) {
        Ok(digest) => Some(digest),
        Err(error) => {
            if std::env::var_os("CUDA_OXIDE_VERBOSE").is_some() {
                eprintln!(
                    "cuda-oxide: could not fingerprint loaded {label} ({error}); disabling artifact reuse"
                );
            }
            None
        }
    }
}

pub(crate) fn report_changed_tool(label: &str) {
    if std::env::var_os("CUDA_OXIDE_VERBOSE").is_some() {
        eprintln!(
            "cuda-oxide: {label} changed while it was fingerprinted; disabling artifact reuse"
        );
    }
}

pub(crate) fn nvvm_ir_artifact_digest_parts(
    module_name: &str,
    nvvm_ir: &[u8],
    options: &FinalizationOptions,
    libdevice_digest: &[u8; 32],
    libnvvm_digest: &[u8; 32],
) -> [u8; 32] {
    let mut digest = StableDigest::new()
        .field("recipe", recipe_digest())
        .field("route", b"nvvm-ir-to-ltoir")
        .field("module-name", module_name.as_bytes())
        .field("module", nvvm_ir)
        .field("module-order", b"libdevice.10.bc,user-nvvm-ir")
        .field("libdevice-sha256", libdevice_digest);
    for option in options.nvvm_verify_options() {
        digest = digest.field("nvvm-verify-option", option.as_bytes());
    }
    for option in options.nvvm_compile_options() {
        digest = digest.field("nvvm-compile-option", option.as_bytes());
    }
    digest.field("libnvvm-sha256", libnvvm_digest).finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvvm_digest_covers_module_name_bytes_options_and_libdevice() {
        let options = FinalizationOptions::new("sm_90".parse().unwrap());
        let baseline =
            nvvm_ir_artifact_digest_parts("kernel.ll", b"ir", &options, &[1; 32], &[2; 32]);
        assert_ne!(
            baseline,
            nvvm_ir_artifact_digest_parts("other.ll", b"ir", &options, &[1; 32], &[2; 32])
        );
        assert_ne!(
            baseline,
            nvvm_ir_artifact_digest_parts("kernel.ll", b"changed", &options, &[1; 32], &[2; 32])
        );
        assert_ne!(
            baseline,
            nvvm_ir_artifact_digest_parts(
                "kernel.ll",
                b"ir",
                &options.clone().with_fma_contraction(false),
                &[1; 32],
                &[2; 32]
            )
        );
        assert_ne!(
            baseline,
            nvvm_ir_artifact_digest_parts("kernel.ll", b"ir", &options, &[3; 32], &[2; 32])
        );
        assert_ne!(
            baseline,
            nvvm_ir_artifact_digest_parts("kernel.ll", b"ir", &options, &[1; 32], &[4; 32])
        );
        assert_ne!(
            baseline,
            nvvm_ir_artifact_digest_parts(
                "kernel.ll",
                b"ir",
                &FinalizationOptions::new("sm_120".parse().unwrap()),
                &[1; 32],
                &[2; 32]
            )
        );
        assert_ne!(
            baseline,
            nvvm_ir_artifact_digest_parts(
                "kernel.ll",
                b"ir",
                &options.with_debug_policy(crate::DebugPolicy::Full),
                &[1; 32],
                &[2; 32]
            )
        );
    }
}
