/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Strict, opt-in build-time finalization of embedded device artifacts.
//!
//! `cargo oxide --materialize-cubin` discovers and fingerprints the exact
//! libNVVM, nvJitLink, and libdevice inputs before invoking Cargo. The digest
//! is part of Cargo's rustc fingerprint and is passed here through an internal
//! environment variable. We rediscover the tools and compare their bytes
//! before compiling. Setting the internal opt-in around raw Cargo is
//! unsupported: Cargo can reuse an existing artifact without invoking this
//! backend, and when the backend does run it rejects a missing handshake.

use cuda_artifact_finalizer::{
    CudaArch, CudaArchParseError, DebugPolicy, FinalizationOptions, Finalizer, FinalizerError,
    FinalizerOutput, NamedInput,
};
use thiserror::Error;

pub(crate) const MATERIALIZE_ENV: &str = "CUDA_OXIDE_MATERIALIZE_CUBIN";
pub(crate) const EXPECTED_PROVENANCE_ENV: &str = "CUDA_OXIDE_INTERNAL_MATERIALIZER_PROVENANCE";
pub(crate) const WRAPPER_PROVENANCE_CFG: &str = "cuda_oxide_internal_materializer_provenance";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MaterializationRequest {
    expected_provenance: [u8; 32],
}

/// Failures in the wrapper/backend materialization contract.
#[derive(Debug, Error)]
pub(crate) enum MaterializeError {
    #[error(
        "{MATERIALIZE_ENV} must be a boolean (accepted true values: 1, true, yes, on; false values: 0, false, no, off), got {value:?}"
    )]
    InvalidBoolean { value: String },

    #[error("{MATERIALIZE_ENV} is not valid Unicode")]
    NonUnicodeBoolean,

    #[error(
        "{MATERIALIZE_ENV}=true requires cargo-oxide's provenance handshake; use `cargo oxide build --materialize-cubin` instead of invoking raw Cargo"
    )]
    MissingExpectedProvenance,

    #[error(
        "{MATERIALIZE_ENV}=true is missing cargo-oxide's rustc fingerprint; use `cargo oxide build --materialize-cubin` instead of invoking raw Cargo"
    )]
    MissingCargoFingerprint,

    #[error(
        "cargo-oxide's materializer provenance cfg must be exactly 64 lowercase hexadecimal characters, got {value:?}"
    )]
    InvalidCargoProvenance { value: String },

    #[error(
        "cargo-oxide's materializer provenance cfg does not match {EXPECTED_PROVENANCE_ENV}; refusing an unfingerprinted materialization build"
    )]
    CargoProvenanceMismatch,

    #[error(
        "{EXPECTED_PROVENANCE_ENV} must be exactly 64 lowercase hexadecimal characters, got {value:?}"
    )]
    InvalidExpectedProvenance { value: String },

    #[error(
        "the loaded CUDA tools cannot be tied to exact files, so their provenance cannot be verified; refusing build-time cubin materialization"
    )]
    UnverifiableProvenance,

    #[error(
        "CUDA materializer provenance changed after Cargo fingerprinting (expected {expected}, loaded {actual}); rerun `cargo oxide build --materialize-cubin`"
    )]
    ProvenanceMismatch { expected: String, actual: String },

    #[error(
        "build-time cubin materialization does not yet support generic #[cuda_module] loading because it merges PTX bundles across crates at run time"
    )]
    RequiresPtxBundleMerge,

    #[error(
        "build-time cubin materialization does not yet support #[device] extern declarations because their ordered external link inputs are not available to the backend"
    )]
    HasDeviceExterns,

    #[error(
        "build-time cubin materialization requires an NVVM IR or LTOIR artifact, but codegen produced PTX; use cargo-oxide so materialization can force NVVM IR emission"
    )]
    PtxInput,

    #[error(
        "build-time cubin materialization expected compiler IR, but codegen already produced a cubin; refusing to bypass the provenance-checked finalization recipe"
    )]
    CubinInput,

    #[error(transparent)]
    InvalidTarget(#[from] CudaArchParseError),

    #[error(transparent)]
    Finalizer(#[from] FinalizerError),
}

/// Parse the strict opt-in and its wrapper-generated provenance handshake.
/// No CUDA library is loaded here.
pub(crate) fn request_from_env(
    wrapper_provenance: Option<&str>,
) -> Result<Option<MaterializationRequest>, MaterializeError> {
    let enabled = match std::env::var(MATERIALIZE_ENV) {
        Ok(value) => parse_bool(&value)?,
        Err(std::env::VarError::NotPresent) => false,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(MaterializeError::NonUnicodeBoolean);
        }
    };
    if !enabled {
        return Ok(None);
    }
    let value = std::env::var(EXPECTED_PROVENANCE_ENV)
        .map_err(|_| MaterializeError::MissingExpectedProvenance)?;
    let expected_provenance = parse_digest(&value)?;
    validate_wrapper_provenance(wrapper_provenance, &expected_provenance)?;
    Ok(Some(MaterializationRequest {
        expected_provenance,
    }))
}

/// Reject artifact-loading models the finalizer cannot reproduce, before any
/// CUDA compiler library is discovered or loaded.
pub(crate) fn validate_collection(
    request: Option<MaterializationRequest>,
    has_device_externs: bool,
    requires_ptx_bundle_merge: bool,
) -> Result<(), MaterializeError> {
    if request.is_none() {
        return Ok(());
    }
    if requires_ptx_bundle_merge {
        return Err(MaterializeError::RequiresPtxBundleMerge);
    }
    if has_device_externs {
        return Err(MaterializeError::HasDeviceExterns);
    }
    Ok(())
}

pub(crate) fn nvvm_ir_to_cubin(
    request: MaterializationRequest,
    nvvm_ir: &[u8],
    module_name: &str,
    target: &str,
    allow_fma_contraction: bool,
    debug_policy: DebugPolicy,
) -> Result<Vec<u8>, MaterializeError> {
    let options = options(target, allow_fma_contraction, debug_policy)?;
    let finalizer = checked_finalizer(request)?;
    Ok(finalizer.materialize_nvvm_ir(module_name, nvvm_ir, &options)?)
}

pub(crate) fn ltoir_to_cubin(
    request: MaterializationRequest,
    ltoir: &[u8],
    module_name: &str,
    target: &str,
    allow_fma_contraction: bool,
    debug_policy: DebugPolicy,
) -> Result<Vec<u8>, MaterializeError> {
    let options = options(target, allow_fma_contraction, debug_policy)?;
    let finalizer = checked_finalizer(request)?;
    Ok(finalizer.link_ltoir(
        &[NamedInput::new(module_name, ltoir)],
        &options,
        FinalizerOutput::Cubin,
    )?)
}

fn options(
    target: &str,
    allow_fma_contraction: bool,
    debug_policy: DebugPolicy,
) -> Result<FinalizationOptions, MaterializeError> {
    let target: CudaArch = target.parse()?;
    Ok(FinalizationOptions::new(target)
        .with_fma_contraction(allow_fma_contraction)
        .with_debug_policy(debug_policy))
}

fn checked_finalizer(request: MaterializationRequest) -> Result<Finalizer, MaterializeError> {
    let finalizer = Finalizer::discover()?;
    let actual = finalizer
        .provenance_digest()
        .ok_or(MaterializeError::UnverifiableProvenance)?;
    if actual != request.expected_provenance {
        return Err(MaterializeError::ProvenanceMismatch {
            expected: digest_hex(&request.expected_provenance),
            actual: digest_hex(&actual),
        });
    }
    Ok(finalizer)
}

fn parse_bool(value: &str) -> Result<bool, MaterializeError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(MaterializeError::InvalidBoolean {
            value: value.to_string(),
        }),
    }
}

fn validate_wrapper_provenance(
    value: Option<&str>,
    expected: &[u8; 32],
) -> Result<(), MaterializeError> {
    let value = value.ok_or(MaterializeError::MissingCargoFingerprint)?;
    let wrapper = parse_digest(value).map_err(|_| MaterializeError::InvalidCargoProvenance {
        value: value.to_string(),
    })?;
    if &wrapper == expected {
        Ok(())
    } else {
        Err(MaterializeError::CargoProvenanceMismatch)
    }
}

fn parse_digest(value: &str) -> Result<[u8; 32], MaterializeError> {
    if value.len() != 64
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(MaterializeError::InvalidExpectedProvenance {
            value: value.to_string(),
        });
    }
    let mut digest = [0_u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        let offset = index * 2;
        *byte = u8::from_str_radix(&value[offset..offset + 2], 16).map_err(|_| {
            MaterializeError::InvalidExpectedProvenance {
                value: value.to_string(),
            }
        })?;
    }
    Ok(digest)
}

fn digest_hex(digest: &[u8; 32]) -> String {
    use std::fmt::Write;
    let mut hex = String::with_capacity(64);
    for byte in digest {
        write!(&mut hex, "{byte:02x}").expect("writing to String cannot fail");
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_boolean_parser_accepts_only_documented_values() {
        for value in ["1", " true ", "YES", "on"] {
            assert!(parse_bool(value).unwrap());
        }
        for value in ["0", " false ", "NO", "off"] {
            assert!(!parse_bool(value).unwrap());
        }
        for value in ["", "enabled", "2", "truthy"] {
            assert!(matches!(
                parse_bool(value),
                Err(MaterializeError::InvalidBoolean { .. })
            ));
        }
    }

    #[test]
    fn provenance_digest_requires_canonical_lower_hex() {
        let value = "0123456789abcdef".repeat(4);
        let digest = parse_digest(&value).unwrap();
        assert_eq!(digest_hex(&digest), value);
        assert!(parse_digest(&"A".repeat(64)).is_err());
        assert!(parse_digest(&"0".repeat(63)).is_err());
        assert!(parse_digest(&format!("{}g", "0".repeat(63))).is_err());
    }

    #[test]
    fn unsupported_collection_models_fail_without_tools() {
        let request = Some(MaterializationRequest {
            expected_provenance: [0; 32],
        });
        assert!(matches!(
            validate_collection(request, false, true),
            Err(MaterializeError::RequiresPtxBundleMerge)
        ));
        assert!(matches!(
            validate_collection(request, true, false),
            Err(MaterializeError::HasDeviceExterns)
        ));
        assert!(validate_collection(None, true, true).is_ok());
    }

    #[test]
    fn backend_invocation_rejects_raw_cargo_materialization_without_fingerprint() {
        let expected = [0; 32];
        assert!(matches!(
            validate_wrapper_provenance(None, &expected),
            Err(MaterializeError::MissingCargoFingerprint)
        ));
        assert!(validate_wrapper_provenance(Some(&"00".repeat(32)), &expected).is_ok());
        assert!(matches!(
            validate_wrapper_provenance(Some(&"11".repeat(32)), &expected),
            Err(MaterializeError::CargoProvenanceMismatch)
        ));
    }
}
