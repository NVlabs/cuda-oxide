/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Driver-independent CUDA artifact finalization.
//!
//! This crate is the single owner of cuda-oxide's libNVVM and nvJitLink
//! compilation policy. It deliberately does not link the CUDA Driver. Both
//! build-time materialization and runtime fallback use the same typed target,
//! FMA, debug, input-order, validation, and provenance rules.

mod link;
mod nvvm;
mod options;
mod provenance;
mod validation;

pub use libnvvm_sys::{CudaArch, CudaArchParseError, LibdeviceNotFound, NvvmError, find_libdevice};
pub use link::LtoLinker;
pub use nvjitlink_sys::NvJitLinkError;
pub use nvvm::NvvmCompiler;
pub use options::{DebugPolicy, FinalizationOptions, FinalizerOutput, NamedInput};
pub use provenance::{ToolProvenance, recipe_digest};
pub use validation::is_valid_cubin;

use provenance::common_provenance_digest;
use std::path::PathBuf;
use thiserror::Error;

/// Failures while compiling NVVM IR or linking LTOIR.
#[derive(Debug, Error)]
pub enum FinalizerError {
    /// libNVVM failed to load, validate, or compile.
    #[error("libnvvm: {0}")]
    Nvvm(#[from] libnvvm_sys::NvvmError),

    /// nvJitLink failed to load or link.
    #[error("nvJitLink: {0}")]
    NvJitLink(#[from] nvjitlink_sys::NvJitLinkError),

    /// `libdevice.10.bc` could not be found.
    #[error(
        "Could not locate libdevice.10.bc. Set CUDA_OXIDE_LIBDEVICE, CUDA_TOOLKIT_PATH, or CUDA_HOME, or install the CUDA Toolkit. Tried:\n  {tried}"
    )]
    LibdeviceNotFound {
        /// Newline-separated discovery paths.
        tried: String,
    },

    /// A finalizer input could not be read.
    #[error("Failed reading {path}: {source}")]
    Io {
        /// Path that could not be read.
        path: PathBuf,
        /// Underlying filesystem failure.
        #[source]
        source: std::io::Error,
    },

    /// The installed toolkit does not accept cuda-oxide's NVVM IR version.
    #[error("installed libNVVM accepts NVVM IR {major}.{minor}, but cuda-oxide emits NVVM IR 2.0")]
    UnsupportedNvvmIrVersion { major: i32, minor: i32 },

    /// Runtime toolkit dialect discovery disagreed with the target policy.
    #[error(
        "libNVVM reports LLVM {llvm_major} for {target}, which disagrees with cuda-oxide's expected {expected} dialect"
    )]
    DialectMismatch {
        target: String,
        llvm_major: i32,
        expected: &'static str,
    },

    /// A diagnostic input name cannot be represented by the CUDA C APIs.
    #[error("CUDA artifact input name contains an interior NUL byte: {name:?}")]
    InvalidInputName { name: String },

    /// A supplied compiler or linker input contained no bytes.
    #[error("CUDA artifact input is empty: {name}")]
    EmptyInput { name: String },

    /// nvJitLink was invoked without an input module.
    #[error("at least one ordered LTOIR input is required")]
    NoLinkInputs,

    /// nvJitLink returned bytes that are not a complete CUDA ELF image.
    #[error("nvJitLink returned an invalid or truncated cubin")]
    InvalidCubin,

    /// nvJitLink returned no PTX bytes.
    #[error("nvJitLink returned an empty PTX artifact")]
    EmptyPtx,

    /// A pinned CUDA compiler DSO changed around an operation. Its output can
    /// no longer be attributed to the provenance used by Cargo or a cache key.
    #[error(
        "the pinned {tool} file changed before or during CUDA artifact finalization; refusing the unverified output"
    )]
    ToolIdentityChanged { tool: &'static str },
}

/// Complete NVVM IR to cubin/PTX finalizer.
#[derive(Clone)]
pub struct Finalizer {
    compiler: NvvmCompiler,
    linker: LtoLinker,
}

impl Finalizer {
    /// Discover libNVVM, libdevice, and nvJitLink without loading the Driver.
    pub fn discover() -> Result<Self, FinalizerError> {
        Ok(Self {
            compiler: NvvmCompiler::discover()?,
            linker: LtoLinker::discover()?,
        })
    }

    /// Compile one NVVM IR module and return a validated target-specific cubin.
    pub fn materialize_nvvm_ir(
        &self,
        module_name: &str,
        nvvm_ir: &[u8],
        options: &FinalizationOptions,
    ) -> Result<Vec<u8>, FinalizerError> {
        let ltoir = self
            .compiler
            .compile_nvvm_ir_to_ltoir(module_name, nvvm_ir, options)?;
        let ltoir_name = format!("{module_name}.ltoir");
        self.linker.link_ltoir(
            &[NamedInput::new(&ltoir_name, &ltoir)],
            options,
            FinalizerOutput::Cubin,
        )
    }

    /// Link ordered LTOIR modules to cubin or PTX.
    pub fn link_ltoir(
        &self,
        inputs: &[NamedInput<'_>],
        options: &FinalizationOptions,
        output: FinalizerOutput,
    ) -> Result<Vec<u8>, FinalizerError> {
        self.linker.link_ltoir(inputs, options, output)
    }

    /// Compiler component, including exact libdevice bytes and provenance.
    pub fn compiler(&self) -> &NvvmCompiler {
        &self.compiler
    }

    /// Ordered LTOIR linker component.
    pub fn linker(&self) -> &LtoLinker {
        &self.linker
    }

    /// Exact discovered tool and libdevice digests.
    pub fn provenance(&self) -> ToolProvenance {
        ToolProvenance {
            libnvvm_sha256: self.compiler.libnvvm_digest(),
            nvjitlink_sha256: self.linker.nvjitlink_digest(),
            libdevice_sha256: self.compiler.libdevice_digest(),
        }
    }

    /// Exact full-pipeline provenance, or `None` if a loaded DSO is unknown.
    pub fn provenance_digest(&self) -> Option<[u8; 32]> {
        let provenance = self.provenance();
        Some(common_provenance_digest(
            &provenance.libnvvm_sha256?,
            &provenance.nvjitlink_sha256?,
            &provenance.libdevice_sha256,
        ))
    }

    /// Digest the full NVVM IR to output recipe, including ordered options.
    pub fn nvvm_ir_artifact_digest(
        &self,
        module_name: &str,
        ltoir_module_name: &str,
        nvvm_ir: &[u8],
        options: &FinalizationOptions,
        output: FinalizerOutput,
    ) -> Option<[u8; 32]> {
        nvvm_ir_artifact_digest_with_provenance(
            module_name,
            ltoir_module_name,
            nvvm_ir,
            options,
            output,
            self.provenance(),
        )
    }
}

/// Digest a complete finalization plan from already-established provenance.
///
/// This is useful to fingerprint Cargo work before executing the plan. It
/// returns `None` unless both loaded tool identities are exact.
pub fn nvvm_ir_artifact_digest_with_provenance(
    module_name: &str,
    ltoir_module_name: &str,
    nvvm_ir: &[u8],
    options: &FinalizationOptions,
    output: FinalizerOutput,
    provenance: ToolProvenance,
) -> Option<[u8; 32]> {
    let compiler_digest = nvvm::nvvm_ir_artifact_digest_parts(
        module_name,
        nvvm_ir,
        options,
        &provenance.libdevice_sha256,
        &provenance.libnvvm_sha256?,
    );
    let linker_digest = link::ltoir_artifact_digest_parts(
        &[NamedInput::new(ltoir_module_name, &compiler_digest)],
        options,
        output,
        &provenance.nvjitlink_sha256?,
    );
    Some(
        provenance::StableDigest::new()
            .field("recipe", recipe_digest())
            .field("route", b"nvvm-ir-to-final-output")
            .field("compiler-plan", compiler_digest)
            .field("linker-plan", linker_digest)
            .finish(),
    )
}

/// Digest an ordered LTOIR link from an established exact linker identity.
pub fn ltoir_artifact_digest_with_provenance(
    inputs: &[NamedInput<'_>],
    options: &FinalizationOptions,
    output: FinalizerOutput,
    nvjitlink_sha256: &[u8; 32],
) -> [u8; 32] {
    link::ltoir_artifact_digest_parts(inputs, options, output, nvjitlink_sha256)
}

fn validate_name(name: &str) -> Result<(), FinalizerError> {
    if name.as_bytes().contains(&0) {
        Err(FinalizerError::InvalidInputName {
            name: name.to_string(),
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;

    const LEGACY_NVVM_IR: &[u8] = br#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @kernel() {
entry:
  ret void
}

!nvvm.annotations = !{!0}
!nvvmir.version = !{!1}
!0 = !{void ()* @kernel, !"kernel", i32 1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
"#;

    #[test]
    #[ignore = "requires discoverable CUDA Toolkit libNVVM, nvJitLink, and libdevice"]
    fn live_pipeline_emits_valid_cubin_and_ptx_for_both_fma_policies() {
        let finalizer = Finalizer::discover().unwrap();
        assert!(finalizer.provenance_digest().is_some());
        let target: CudaArch = "sm_86".parse().unwrap();

        for (allow_fma, debug) in [
            (false, DebugPolicy::None),
            (false, DebugPolicy::LineTables),
            (true, DebugPolicy::Full),
        ] {
            let options = FinalizationOptions::new(target.clone())
                .with_fma_contraction(allow_fma)
                .with_debug_policy(debug);
            let ltoir = finalizer
                .compiler()
                .compile_nvvm_ir_to_ltoir("kernel.ll", LEGACY_NVVM_IR, &options)
                .unwrap();
            assert!(!ltoir.is_empty());
            let input = [NamedInput::new("kernel.ltoir", &ltoir)];
            let cubin = finalizer
                .link_ltoir(&input, &options, FinalizerOutput::Cubin)
                .unwrap();
            assert!(is_valid_cubin(&cubin));
            let ptx = finalizer
                .link_ltoir(&input, &options, FinalizerOutput::Ptx)
                .unwrap();
            assert!(
                ptx.windows(b".version".len())
                    .any(|part| part == b".version")
            );
        }
    }
}
