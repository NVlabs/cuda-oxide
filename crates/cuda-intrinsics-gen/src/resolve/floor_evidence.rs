/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::OverlayIntrinsic;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

pub(crate) const FLOOR_EVIDENCE_SCHEMA: u32 = 1;
pub(crate) const MINIMUM_PROBEABLE_PTX: &str = "6.3";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FloorEvidenceFile {
    pub schema: u32,
    pub profile: String,
    pub tool_path: String,
    pub tool_version: String,
    pub tool_sha256: String,
    pub llvm_tool_path: String,
    pub llvm_tool_version: String,
    pub llvm_tool_sha256: String,
    pub minimum_supported_target: String,
    pub minimum_probeable_ptx: String,
    pub records: Vec<FloorEvidenceRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum FloorEvidenceRecord {
    Verified {
        id: String,
        minimum_ptx: String,
        target: String,
        accepted_ptx: String,
        rejected_ptx: String,
        rejection_detail: String,
    },
    Unverifiable {
        id: String,
        minimum_ptx: String,
        reason: String,
    },
}

impl FloorEvidenceRecord {
    pub(crate) fn id(&self) -> &str {
        match self {
            Self::Verified { id, .. } | Self::Unverifiable { id, .. } => id,
        }
    }

    pub(crate) fn minimum_ptx(&self) -> &str {
        match self {
            Self::Verified { minimum_ptx, .. } | Self::Unverifiable { minimum_ptx, .. } => {
                minimum_ptx
            }
        }
    }
}

pub(crate) fn evidence_path(repo_root: &Path) -> PathBuf {
    repo_root.join("intrinsics/floor-evidence/cuda-13.2.51-ptxas.json")
}

pub(super) fn validate_floor_evidence(
    repo_root: &Path,
    policies: &[OverlayIntrinsic],
) -> Result<()> {
    let path = evidence_path(repo_root);
    let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let file: FloorEvidenceFile = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse floor evidence {}", path.display()))?;
    ensure!(
        file.schema == FLOOR_EVIDENCE_SCHEMA,
        "unsupported floor evidence schema"
    );
    ensure!(
        file.minimum_supported_target == "sm_75",
        "floor evidence must record sm_75 as the CUDA 13.2 minimum target"
    );
    ensure!(
        file.minimum_probeable_ptx == MINIMUM_PROBEABLE_PTX,
        "floor evidence must record PTX {MINIMUM_PROBEABLE_PTX} as its measurement boundary"
    );
    ensure!(
        !file.tool_version.trim().is_empty() && file.tool_sha256.len() == 64,
        "floor evidence does not pin a concrete ptxas"
    );
    ensure!(
        !file.llvm_tool_version.trim().is_empty() && file.llvm_tool_sha256.len() == 64,
        "floor evidence does not pin the LLVM tool that produced its PTX"
    );

    let mut indexed = BTreeMap::new();
    for record in &file.records {
        ensure!(
            indexed.insert(record.id(), record).is_none(),
            "duplicate floor evidence for {}",
            record.id()
        );
    }
    ensure!(
        indexed.len() == policies.len(),
        "floor evidence has {} records, catalog policy has {}",
        indexed.len(),
        policies.len()
    );
    let policy_ids: BTreeSet<_> = policies.iter().map(|policy| policy.id.as_str()).collect();
    ensure!(
        indexed.keys().copied().collect::<BTreeSet<_>>() == policy_ids,
        "floor evidence IDs do not exactly cover catalog policy IDs"
    );

    for policy in policies {
        let record = indexed[policy.id.as_str()];
        ensure!(
            record.minimum_ptx() == policy.minimum_ptx,
            "{} floor evidence records PTX {}, catalog declares PTX {}",
            policy.id,
            record.minimum_ptx(),
            policy.minimum_ptx
        );
        let encoded = parse_version(&policy.minimum_ptx)?;
        match record {
            FloorEvidenceRecord::Verified {
                accepted_ptx,
                rejected_ptx,
                rejection_detail,
                ..
            } => {
                ensure!(
                    encoded >= 63,
                    "{} claims a measured floor below the CUDA 13.2 measurement boundary",
                    policy.id
                );
                ensure!(
                    accepted_ptx == &policy.minimum_ptx,
                    "{} accepted floor does not match its declaration",
                    policy.id
                );
                ensure!(
                    parse_version(rejected_ptx)? < encoded,
                    "{} rejected spelling is not below its accepted spelling",
                    policy.id
                );
                ensure!(
                    !rejection_detail.trim().is_empty(),
                    "{} verified verdict omits the negative-control diagnostic",
                    policy.id
                );
            }
            FloorEvidenceRecord::Unverifiable { reason, .. } => {
                ensure!(
                    encoded < 63,
                    "{} is marked unverifiable even though CUDA 13.2 can express its floor",
                    policy.id
                );
                ensure!(
                    !reason.trim().is_empty(),
                    "{} unverifiable verdict has no reason",
                    policy.id
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_canonical_ptx_versions() {
        assert_eq!(parse_version("6.3").unwrap(), 63);
        assert_eq!(parse_version("8.7").unwrap(), 87);
        assert!(parse_version("63").is_err());
        assert!(parse_version("6.30").is_err());
    }
}

pub(crate) fn parse_version(value: &str) -> Result<u16> {
    let (major, minor) = value
        .split_once('.')
        .with_context(|| format!("invalid PTX version {value}"))?;
    ensure!(minor.len() == 1, "invalid PTX version {value}");
    Ok(major.parse::<u16>()? * 10 + minor.parse::<u16>()?)
}
