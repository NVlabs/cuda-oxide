/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host-side loading for embedded device artifact bundles.

use crate::ltoir;
pub use cuda_core::embedded::{
    ArtifactPayloadKind, EmbeddedModule, OwnedArtifactBundle, artifact_bundles_from_binary_path,
    artifact_bundles_from_current_exe, embedded_modules_from_current_exe,
};
use cuda_core::{CudaContext, CudaModule, DriverError};
use std::sync::Arc;
use thiserror::Error;

/// Errors while discovering, building, or loading an embedded CUDA module.
#[derive(Debug, Error)]
pub enum EmbeddedModuleError {
    /// Reading the embedded artifact section failed.
    #[error(transparent)]
    Core(#[from] cuda_core::EmbeddedModuleError),

    /// The named bundle was not present in the current executable.
    #[error("embedded CUDA module '{name}' was not found")]
    ModuleNotFound { name: String },

    /// No embedded bundles with loadable payloads were found.
    #[error("no embedded CUDA modules were found")]
    NoModules,

    /// A bundle existed, but it contained no supported payload.
    #[error("embedded CUDA module '{name}' has no supported payload")]
    UnsupportedPayload { name: String },

    /// NVVM IR or LTOIR payload compilation failed.
    #[error("failed to build embedded CUDA module: {0}")]
    Ltoir(#[from] ltoir::LtoirError),

    /// The CUDA driver rejected the selected module image.
    #[error("failed to load embedded CUDA module: {0}")]
    Driver(#[from] DriverError),
}

/// Load a named embedded artifact bundle from the current executable.
///
/// Cubin and PTX payloads are loaded directly with the CUDA driver. NVVM IR and
/// LTOIR payloads are first linked to an in-memory cubin via the existing
/// libNVVM/nvJitLink path.
pub fn load_embedded_module(
    ctx: &Arc<CudaContext>,
    name: &str,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    let bundle = artifact_bundles_from_current_exe()?
        .into_iter()
        .find(|bundle| bundle.name == name)
        .ok_or_else(|| EmbeddedModuleError::ModuleNotFound {
            name: name.to_string(),
        })?;
    load_bundle(ctx, &bundle)
}

/// Load the first embedded artifact bundle with a supported payload.
pub fn load_first_embedded_module(
    ctx: &Arc<CudaContext>,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    for bundle in artifact_bundles_from_current_exe()? {
        match load_bundle(ctx, &bundle) {
            Ok(module) => return Ok(module),
            Err(EmbeddedModuleError::UnsupportedPayload { .. }) => continue,
            Err(error) => return Err(error),
        }
    }
    Err(EmbeddedModuleError::NoModules)
}

fn load_bundle(
    ctx: &Arc<CudaContext>,
    bundle: &OwnedArtifactBundle,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    if let Some(cubin) = bundle.payload(ArtifactPayloadKind::Cubin) {
        return Ok(ctx.load_module_from_image(cubin)?);
    }

    if let Some(ptx) = bundle.payload(ArtifactPayloadKind::Ptx) {
        return Ok(ctx.load_module_from_image(ptx)?);
    }

    if let Some(nvvm_ir) = bundle.payload(ArtifactPayloadKind::NvvmIr) {
        let arch = target_arch_for_bundle(bundle);
        let cubin = ltoir::build_cubin_from_nvvm_ir(nvvm_ir, &bundle.name, &arch)?;
        return Ok(ctx.load_module_from_image(&cubin)?);
    }

    if let Some(ltoir) = bundle.payload(ArtifactPayloadKind::Ltoir) {
        let arch = target_arch_for_bundle(bundle);
        let cubin = ltoir::link_ltoir_to_cubin(ltoir, &bundle.name, &arch)?;
        return Ok(ctx.load_module_from_image(&cubin)?);
    }

    Err(EmbeddedModuleError::UnsupportedPayload {
        name: bundle.name.clone(),
    })
}

fn target_arch_for_bundle(bundle: &OwnedArtifactBundle) -> String {
    if is_cuda_arch(&bundle.target) {
        bundle.target.clone()
    } else {
        ltoir::target_arch()
    }
}

fn is_cuda_arch(target: &str) -> bool {
    target.starts_with("sm_") || target.starts_with("compute_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_arch_uses_bundle_sm_target() {
        let bundle = bundle_with_target("sm_90");
        assert_eq!(target_arch_for_bundle(&bundle), "sm_90");
    }

    #[test]
    fn target_arch_falls_back_for_non_arch_target() {
        let bundle = bundle_with_target("libdevice");
        assert!(!target_arch_for_bundle(&bundle).is_empty());
    }

    fn bundle_with_target(target: &str) -> OwnedArtifactBundle {
        OwnedArtifactBundle {
            name: "demo".to_string(),
            target: target.to_string(),
            payloads: Vec::new(),
            entries: Vec::new(),
        }
    }
}
