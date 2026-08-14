/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Direct assembly of an already-linked PTX module.
//!
//! This route is deliberately separate from [`crate::Finalizer::discover`],
//! so applications that only finalize NVVM IR do not acquire an
//! nvPTXCompiler run-time dependency.

use crate::diagnostics::parse_ptxas_resource_usage;
use crate::nvvm::{loaded_tool_digest, report_changed_tool};
use crate::provenance::{
    StableDigest, digest_file_handle, recipe_digest, with_revalidated_tool_identity,
};
use crate::{FinalizationOptions, FinalizerError, LinkReport, NamedInput, is_valid_cubin};
use nvptxcompiler_sys::{LibNvPtxCompiler, Program};
use std::sync::{Arc, Mutex, OnceLock};

struct LoadedPtxCompilerTool {
    library: Arc<LibNvPtxCompiler>,
    digest: Option<[u8; 32]>,
}

static PTX_COMPILER_TOOL: OnceLock<Arc<LoadedPtxCompilerTool>> = OnceLock::new();
static PTX_COMPILER_TOOL_LOAD: OnceLock<Mutex<()>> = OnceLock::new();

/// Driver-independent assembler for one already-linked PTX module.
#[derive(Clone)]
pub struct PtxAssembler {
    tool: Arc<LoadedPtxCompilerTool>,
}

impl PtxAssembler {
    /// Discover and pin nvPTXCompiler without loading libNVVM or the Driver.
    pub fn discover() -> Result<Self, FinalizerError> {
        Ok(Self {
            tool: load_ptx_compiler_tool()?,
        })
    }

    /// Digest of the exact loaded nvPTXCompiler file, when it can be proven.
    pub fn nvptxcompiler_digest(&self) -> Option<[u8; 32]> {
        let digest = self.tool.digest?;
        if current_ptx_compiler_digest(&self.tool).is_some() {
            Some(digest)
        } else {
            report_changed_tool("nvPTXCompiler");
            None
        }
    }

    /// Digest every semantic input to the standalone PTX assembly stage.
    pub fn artifact_digest(
        &self,
        input: NamedInput<'_>,
        options: &FinalizationOptions,
    ) -> Result<Option<[u8; 32]>, FinalizerError> {
        let logical = logical_ptx(input)?;
        let Some(nvptxcompiler) = self.nvptxcompiler_digest() else {
            return Ok(None);
        };
        Ok(Some(ptx_assembly_artifact_digest_parts(
            NamedInput::new(input.name, logical),
            options,
            &nvptxcompiler,
        )))
    }

    /// Assemble one PTX module into a validated target-specific cubin.
    pub fn assemble_ptx(
        &self,
        input: NamedInput<'_>,
        options: &FinalizationOptions,
    ) -> Result<Vec<u8>, FinalizerError> {
        Ok(self.assemble_ptx_impl(input, options, false)?.image)
    }

    /// Assemble one PTX module and collect ptxas resource diagnostics.
    pub fn assemble_ptx_with_report(
        &self,
        input: NamedInput<'_>,
        options: &FinalizationOptions,
    ) -> Result<LinkReport, FinalizerError> {
        self.assemble_ptx_impl(input, options, true)
    }

    fn assemble_ptx_impl(
        &self,
        input: NamedInput<'_>,
        options: &FinalizationOptions,
        collect_resource_usage: bool,
    ) -> Result<LinkReport, FinalizerError> {
        let logical = logical_ptx(input)?;
        with_revalidated_tool_identity(
            "nvPTXCompiler",
            self.tool.digest,
            || current_ptx_compiler_digest(&self.tool),
            || assemble(&self.tool.library, logical, options, collect_resource_usage),
        )
    }
}

fn logical_ptx(input: NamedInput<'_>) -> Result<&[u8], FinalizerError> {
    crate::validate_name(input.name)?;
    let logical =
        nvjitlink_sys::logical_ptx(input.bytes, input.name).map_err(|error| match error {
            nvjitlink_sys::NvJitLinkError::InteriorNulPtx { name, .. } => {
                FinalizerError::InteriorNulPtx { name }
            }
            other => FinalizerError::NvJitLink(other),
        })?;
    if logical.is_empty() {
        return Err(FinalizerError::EmptyInput {
            name: input.name.to_owned(),
        });
    }
    Ok(logical)
}

fn assemble(
    library: &LibNvPtxCompiler,
    logical: &[u8],
    options: &FinalizationOptions,
    collect_resource_usage: bool,
) -> Result<LinkReport, FinalizerError> {
    let mut program = Program::new(library, logical)?;
    let option_storage = options.nvptxcompiler_options(collect_resource_usage);
    let option_refs = option_storage
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    program.compile(&option_refs)?;
    let image = program.compiled_program()?;
    if !is_valid_cubin(&image) {
        return Err(FinalizerError::InvalidCubin);
    }

    let info_log = if collect_resource_usage {
        program.info_log()
    } else {
        None
    };
    let resource_usage = info_log
        .as_deref()
        .map(parse_ptxas_resource_usage)
        .unwrap_or_default();
    Ok(LinkReport {
        image,
        info_log,
        resource_usage,
    })
}

fn current_ptx_compiler_digest(tool: &LoadedPtxCompilerTool) -> Option<[u8; 32]> {
    let file = tool.library.loaded_file_if_unchanged()?;
    digest_file_handle(file).ok()
}

pub(crate) fn ptx_assembly_artifact_digest_parts(
    input: NamedInput<'_>,
    options: &FinalizationOptions,
    nvptxcompiler_digest: &[u8; 32],
) -> [u8; 32] {
    let logical = input.bytes.strip_suffix(&[0]).unwrap_or(input.bytes);
    let mut digest = StableDigest::new()
        .field("recipe", recipe_digest())
        .field("route", b"ptx-to-cubin-standalone")
        .field("input-name", input.name.as_bytes())
        .field("input", logical);
    for option in options.nvptxcompiler_options(false) {
        digest = digest.field("nvptxcompiler-option", option.as_bytes());
    }
    digest
        .field("libnvptxcompiler-sha256", nvptxcompiler_digest)
        .finish()
}

fn load_ptx_compiler_tool() -> Result<Arc<LoadedPtxCompilerTool>, FinalizerError> {
    if let Some(loaded) = PTX_COMPILER_TOOL.get() {
        return Ok(Arc::clone(loaded));
    }
    let _guard = PTX_COMPILER_TOOL_LOAD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(loaded) = PTX_COMPILER_TOOL.get() {
        return Ok(Arc::clone(loaded));
    }

    let library = LibNvPtxCompiler::load_for_cache()?;
    let digest = loaded_tool_digest("nvPTXCompiler", library.loaded_file_if_unchanged());
    let loaded = Arc::new(LoadedPtxCompilerTool {
        library: Arc::new(library),
        digest,
    });
    let _ = PTX_COMPILER_TOOL.set(Arc::clone(&loaded));
    Ok(loaded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use libnvvm_sys::CudaArch;

    fn options() -> FinalizationOptions {
        FinalizationOptions::new("sm_103a".parse::<CudaArch>().unwrap())
    }

    #[test]
    fn assembly_digest_covers_input_name_bytes_options_and_tool() {
        let options = options();
        let base = ptx_assembly_artifact_digest_parts(
            NamedInput::new("kernel.ptx", b"ptx"),
            &options,
            &[1; 32],
        );
        assert_ne!(
            base,
            ptx_assembly_artifact_digest_parts(
                NamedInput::new("other.ptx", b"ptx"),
                &options,
                &[1; 32],
            )
        );
        assert_ne!(
            base,
            ptx_assembly_artifact_digest_parts(
                NamedInput::new("kernel.ptx", b"changed"),
                &options,
                &[1; 32],
            )
        );
        assert_ne!(
            base,
            ptx_assembly_artifact_digest_parts(
                NamedInput::new("kernel.ptx", b"ptx"),
                &options.clone().with_fma_contraction(false),
                &[1; 32],
            )
        );
        assert_ne!(
            base,
            ptx_assembly_artifact_digest_parts(
                NamedInput::new("kernel.ptx", b"ptx"),
                &options,
                &[2; 32],
            )
        );
        assert_eq!(
            base,
            ptx_assembly_artifact_digest_parts(
                NamedInput::new("kernel.ptx", b"ptx\0"),
                &options,
                &[1; 32],
            )
        );
    }

    #[test]
    fn standalone_ptx_validation_normalizes_one_terminator_and_rejects_invalid_inputs() {
        assert_eq!(
            logical_ptx(NamedInput::new("kernel.ptx", b"ptx\0")).unwrap(),
            b"ptx"
        );
        assert!(matches!(
            logical_ptx(NamedInput::new("bad.ptx", b"abc\0def")),
            Err(FinalizerError::InteriorNulPtx { ref name }) if name == "bad.ptx"
        ));
        assert!(matches!(
            logical_ptx(NamedInput::new("empty.ptx", b"\0")),
            Err(FinalizerError::EmptyInput { ref name }) if name == "empty.ptx"
        ));
    }

    #[test]
    #[ignore = "requires discoverable CUDA Toolkit nvPTXCompiler"]
    fn live_assembly_emits_a_kernel_cubin() {
        const PTX: &[u8] = br#"
.version 8.0
.target sm_80
.address_size 64

.visible .entry kernel() {
    ret;
}
"#;
        let assembler = PtxAssembler::discover().unwrap();
        assert!(assembler.nvptxcompiler_digest().is_some());
        assert!(
            assembler
                .artifact_digest(NamedInput::new("kernel.ptx", PTX), &options())
                .unwrap()
                .is_some()
        );
        let report = assembler
            .assemble_ptx_with_report(NamedInput::new("kernel.ptx", PTX), &options())
            .unwrap();
        assert!(is_valid_cubin(&report.image));
        assert!(
            report
                .image
                .windows(b"kernel".len())
                .any(|bytes| bytes == b"kernel")
        );
        assert!(report.info_log.is_some());
    }
}
