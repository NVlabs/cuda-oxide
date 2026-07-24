/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::nvvm::{loaded_tool_digest, report_changed_tool};
use crate::options::{FinalizationOptions, FinalizerOutput, NamedInput};
use crate::provenance::{
    StableDigest, digest_file_handle, linker_provenance_digest, recipe_digest,
    with_revalidated_tool_identity,
};
use crate::validation::is_valid_cubin;
use crate::{FinalizerError, validate_name};
use nvjitlink_sys::{InputType, LibNvJitLink, Linker};
use std::sync::{Arc, Mutex, OnceLock};

struct LoadedLinkerTool {
    library: Arc<LibNvJitLink>,
    digest: Option<[u8; 32]>,
}

static LINKER_TOOL: OnceLock<Arc<LoadedLinkerTool>> = OnceLock::new();
static LINKER_TOOL_LOAD: OnceLock<Mutex<()>> = OnceLock::new();

/// Driver-independent ordered LTOIR linker.
#[derive(Clone)]
pub struct LtoLinker {
    tool: Arc<LoadedLinkerTool>,
}

impl LtoLinker {
    /// Discover and pin nvJitLink without loading libNVVM or the CUDA Driver.
    pub fn discover() -> Result<Self, FinalizerError> {
        Ok(Self {
            tool: load_linker_tool()?,
        })
    }

    /// Digest of the exact loaded nvJitLink file, when its identity is known.
    pub fn nvjitlink_digest(&self) -> Option<[u8; 32]> {
        let digest = self.tool.digest?;
        if self.tool.library.loaded_file_if_unchanged().is_some() {
            Some(digest)
        } else {
            report_changed_tool("nvJitLink");
            None
        }
    }

    /// Exact route provenance, or `None` when the loaded DSO is unidentifiable.
    pub fn provenance_digest(&self) -> Option<[u8; 32]> {
        self.nvjitlink_digest()
            .map(|digest| linker_provenance_digest(&digest))
    }

    /// Link one or more LTOIR modules in the exact supplied order.
    pub fn link_ltoir(
        &self,
        inputs: &[NamedInput<'_>],
        options: &FinalizationOptions,
        output: FinalizerOutput,
    ) -> Result<Vec<u8>, FinalizerError> {
        validate_inputs(inputs)?;
        with_revalidated_tool_identity(
            "nvJitLink",
            self.tool.digest,
            || current_linker_tool_digest(&self.tool),
            || {
                let option_storage = options.nvjitlink_options(output);
                let option_refs = option_storage
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>();
                let mut linker = Linker::new(&self.tool.library, &option_refs)?;
                for input in inputs {
                    linker.add(InputType::Ltoir, input.bytes, input.name)?;
                }
                let image = match output {
                    FinalizerOutput::Cubin => linker.finish()?,
                    FinalizerOutput::Ptx => linker.finish_ptx()?,
                };
                if output == FinalizerOutput::Cubin && !is_valid_cubin(&image) {
                    return Err(FinalizerError::InvalidCubin);
                }
                if output == FinalizerOutput::Ptx && image.is_empty() {
                    return Err(FinalizerError::EmptyPtx);
                }
                Ok(image)
            },
        )
    }

    /// Digest every semantic input to an ordered LTOIR link.
    pub fn artifact_digest(
        &self,
        inputs: &[NamedInput<'_>],
        options: &FinalizationOptions,
        output: FinalizerOutput,
    ) -> Option<[u8; 32]> {
        let nvjitlink = self.nvjitlink_digest()?;
        Some(ltoir_artifact_digest_parts(
            inputs, options, output, &nvjitlink,
        ))
    }
}

fn current_linker_tool_digest(tool: &LoadedLinkerTool) -> Option<[u8; 32]> {
    let file = tool.library.loaded_file_if_unchanged()?;
    digest_file_handle(file).ok()
}

fn validate_inputs(inputs: &[NamedInput<'_>]) -> Result<(), FinalizerError> {
    if inputs.is_empty() {
        return Err(FinalizerError::NoLinkInputs);
    }
    for input in inputs {
        validate_name(input.name)?;
        if input.bytes.is_empty() {
            return Err(FinalizerError::EmptyInput {
                name: input.name.to_string(),
            });
        }
    }
    Ok(())
}

fn load_linker_tool() -> Result<Arc<LoadedLinkerTool>, FinalizerError> {
    if let Some(loaded) = LINKER_TOOL.get() {
        return Ok(Arc::clone(loaded));
    }
    let _guard = LINKER_TOOL_LOAD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(loaded) = LINKER_TOOL.get() {
        return Ok(Arc::clone(loaded));
    }

    let library = LibNvJitLink::load_for_cache()?;
    let digest = loaded_tool_digest("nvJitLink", library.loaded_file_if_unchanged());
    let digest = if digest.is_some() && library.loaded_file_if_unchanged().is_none() {
        report_changed_tool("nvJitLink");
        None
    } else {
        digest
    };
    let loaded = Arc::new(LoadedLinkerTool {
        library: Arc::new(library),
        digest,
    });
    let _ = LINKER_TOOL.set(Arc::clone(&loaded));
    Ok(loaded)
}

pub(crate) fn ltoir_artifact_digest_parts(
    inputs: &[NamedInput<'_>],
    options: &FinalizationOptions,
    output: FinalizerOutput,
    nvjitlink_digest: &[u8; 32],
) -> [u8; 32] {
    let output_name = match output {
        FinalizerOutput::Cubin => b"elf-cubin".as_slice(),
        FinalizerOutput::Ptx => b"ptx".as_slice(),
    };
    let mut digest = StableDigest::new()
        .field("recipe", recipe_digest())
        .field("route", b"ltoir-to-output")
        .field("output", output_name);
    for input in inputs {
        digest = digest
            .field("ltoir-name", input.name.as_bytes())
            .field("ltoir", input.bytes);
    }
    for option in options.nvjitlink_options(output) {
        digest = digest.field("nvjitlink-option", option.as_bytes());
    }
    digest
        .field("libnvjitlink-sha256", nvjitlink_digest)
        .finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_digest_preserves_input_order_names_output_and_policy() {
        let options = FinalizationOptions::new("sm_120".parse().unwrap());
        let a = NamedInput::new("a.ltoir", b"a");
        let b = NamedInput::new("b.ltoir", b"b");
        let baseline =
            ltoir_artifact_digest_parts(&[a, b], &options, FinalizerOutput::Cubin, &[7; 32]);
        assert_ne!(
            baseline,
            ltoir_artifact_digest_parts(&[b, a], &options, FinalizerOutput::Cubin, &[7; 32])
        );
        assert_ne!(
            baseline,
            ltoir_artifact_digest_parts(
                &[NamedInput::new("renamed.ltoir", b"a"), b],
                &options,
                FinalizerOutput::Cubin,
                &[7; 32]
            )
        );
        assert_ne!(
            baseline,
            ltoir_artifact_digest_parts(
                &[a, b],
                &FinalizationOptions::new("sm_90".parse().unwrap()),
                FinalizerOutput::Cubin,
                &[7; 32]
            )
        );
        assert_ne!(
            baseline,
            ltoir_artifact_digest_parts(
                &[a, b],
                &options
                    .clone()
                    .with_debug_policy(crate::DebugPolicy::LineTables),
                FinalizerOutput::Cubin,
                &[7; 32]
            )
        );
        assert_ne!(
            baseline,
            ltoir_artifact_digest_parts(&[a, b], &options, FinalizerOutput::Cubin, &[8; 32])
        );
        assert_ne!(
            baseline,
            ltoir_artifact_digest_parts(&[a, b], &options, FinalizerOutput::Ptx, &[7; 32])
        );
        assert_ne!(
            baseline,
            ltoir_artifact_digest_parts(
                &[a, b],
                &options.clone().with_fma_contraction(false),
                FinalizerOutput::Cubin,
                &[7; 32]
            )
        );
    }

    #[test]
    fn input_validation_rejects_zero_inputs_empty_data_and_nul_names() {
        assert!(matches!(
            validate_inputs(&[]),
            Err(FinalizerError::NoLinkInputs)
        ));
        assert!(matches!(
            validate_inputs(&[NamedInput::new("empty", b"")]),
            Err(FinalizerError::EmptyInput { .. })
        ));
        assert!(matches!(
            validate_inputs(&[NamedInput::new("bad\0name", b"x")]),
            Err(FinalizerError::InvalidInputName { .. })
        ));
    }
}
