/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::extract::{IMPORTED_SCHEMA, read_upstream_lock};
use crate::model::{
    AbiLedgerEntry, AbiLedgerFile, AbiRawRustSignature, ActiveMaskAdapter, ActiveMaskObservation,
    BackendLoweringMechanism, CatalogBackend, CatalogBackendLowering, CatalogDialect, CatalogFile,
    CatalogHalfOpenRange, CatalogHardwareAlternative, CatalogHardwareTarget, CatalogInputs,
    CatalogIntrinsic, CatalogLdmatrix, CatalogLlvm, CatalogLlvmResultFacts, CatalogRust,
    CatalogSelection, CatalogSemantics, CatalogSource, CatalogTarget, CatalogTargetRequirement,
    DotProductAdapter, DotProductOperation, DotProductSignedness, EvidenceArtifactKind,
    EvidenceFile, EvidenceRecord, EvidenceStageKind, ImportedAddressSpace, ImportedFile,
    ImportedIntrinsic, IntrinsicBackend, IntrinsicSource, LdmatrixAdapter, LdmatrixAddressContract,
    LdmatrixElement, LdmatrixLayout, LdmatrixMemoryOrder, LdmatrixMultiplicity,
    LdmatrixParticipation, LdmatrixShape, LdmatrixStateSpace, MaskEncoding, MatchOperandEncoding,
    OverlayFile, OverlayIntrinsic, OverlayShardFile, PackedAtomicAccessContract,
    PackedAtomicAdapter, PackedAtomicAtomicity, PackedAtomicCodegenContract, PackedAtomicFormat,
    PackedAtomicOperation, PackedAtomicOrdering, PackedAtomicPointerContract,
    PackedAtomicReturnContract, PackedAtomicRounding, PackedAtomicScope, PackedAtomicScopeContract,
    PackedAtomicStateSpace, PackedAtomicSubnormal, PtxVersion, ReduxAdapter, ReduxOperation,
    ReduxParticipation, RuntimeValidation, VoteAdapter, VoteMode, VoteParticipation,
    WarpMatchAdapter, WarpMatchMode, WarpMatchParticipation, WarpMatchValueWidth,
};
#[cfg(test)]
use crate::ptx::InstructionPattern;
use crate::ptx::OperandPattern;
use crate::util::{read_json, sha256_bytes, sha256_file};
use anyhow::{Context, Result, bail, ensure};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

const OVERLAY_SCHEMA: u32 = 8;
const OVERLAY_SHARD_SCHEMA: u32 = 4;
pub(crate) const CATALOG_SCHEMA: u32 = 7;

pub fn resolve(repo_root: &Path) -> Result<CatalogFile> {
    let lock = read_upstream_lock(repo_root)?;
    let imported_path = repo_root.join("intrinsics/imported.json");
    let overlay_path = repo_root.join("intrinsics/overlay.toml");
    let imported: ImportedFile = read_json(&imported_path)?;
    let (mut overlay, overlay_sha256) = read_overlay(repo_root, &overlay_path)?;
    let ledger_path = repo_root.join(format!("intrinsics/abi-v{}.toml", overlay.intrinsic_abi));
    let ledger_text = fs::read_to_string(&ledger_path)
        .with_context(|| format!("read {}", ledger_path.display()))?;
    let ledger: AbiLedgerFile =
        toml::from_str(&ledger_text).with_context(|| format!("parse {}", ledger_path.display()))?;

    ensure!(
        imported.schema == IMPORTED_SCHEMA,
        "unsupported imported.json schema {}",
        imported.schema
    );
    ensure!(
        overlay.schema == OVERLAY_SCHEMA,
        "unsupported overlay.toml schema {}",
        overlay.schema
    );
    ensure!(
        overlay.intrinsic_abi > 0,
        "intrinsic_abi must be a positive integer"
    );
    ensure!(
        imported.source.llvm_revision == lock.llvm.revision,
        "imported facts use LLVM {}, but upstream.lock pins {}",
        imported.source.llvm_revision,
        lock.llvm.revision
    );
    ensure!(
        imported.source.llvm_tblgen_source_revision == lock.llvm.revision,
        "imported facts were not produced by llvm-tblgen built from the pinned source"
    );
    ensure!(
        imported.source.llvm_tblgen_version == lock.llvm_tblgen.version_line,
        "imported facts use llvm-tblgen {:?}, but upstream.lock pins {:?}",
        imported.source.llvm_tblgen_version,
        lock.llvm_tblgen.version_line
    );
    ensure!(
        imported.source.intrinsics_json_sha256 == lock.dumps.intrinsics_sha256,
        "imported intrinsic dump hash does not match upstream.lock"
    );
    ensure!(
        imported.source.nvptx_json_sha256 == lock.dumps.nvptx_sha256,
        "imported NVPTX dump hash does not match upstream.lock"
    );
    let imported_sha256 = sha256_file(&imported_path)?;
    ensure!(
        imported_sha256 == lock.dumps.normalized_imported_sha256,
        "normalized imported.json hash mismatch: upstream.lock records {}, found {}; regenerate from the pinned dumps, and refresh the lock explicitly only for an intentional normalizer change",
        lock.dumps.normalized_imported_sha256,
        imported_sha256
    );

    overlay
        .intrinsics
        .sort_by(|left, right| left.id.cmp(&right.id));
    validate_unique_overlay(&overlay.intrinsics, overlay.intrinsic_abi)?;
    validate_abi_ledger(&overlay, &ledger)?;
    let imported_by_record: BTreeMap<_, _> = imported
        .intrinsics
        .iter()
        .map(|intrinsic| (intrinsic.source_record.as_str(), intrinsic))
        .collect();
    ensure!(
        imported_by_record.len() == imported.intrinsics.len(),
        "imported.json contains duplicate source records"
    );

    let (evidence_files, evidence_hashes) = read_evidence(repo_root)?;
    let evidence_by_profile_id = index_evidence(&evidence_files, &lock.llvm.revision)?;

    let mut intrinsics = Vec::with_capacity(overlay.intrinsics.len());
    for policy in &overlay.intrinsics {
        let source = resolve_policy_source(policy)?;
        let declaration = match &source {
            IntrinsicSource::LlvmImported { source_record } => Some(
                *imported_by_record
                    .get(source_record.as_str())
                    .with_context(|| {
                        format!(
                            "overlay intrinsic {} references missing imported record {}",
                            policy.id, source_record
                        )
                    })?,
            ),
            IntrinsicSource::PtxNative { .. } => None,
        };
        validate_policy(policy, &source, declaration, overlay.intrinsic_abi)?;
        let evidence = evidence_by_profile_id
            .get(&(overlay.backend_profile.as_str(), policy.id.as_str()))
            .with_context(|| {
                format!(
                    "intrinsic {} has no evidence record in selected profile {}",
                    policy.id, overlay.backend_profile
                )
            })?;
        validate_evidence(policy, evidence, None)?;
        let backend_lowerings = resolve_backend_lowerings(policy, &evidence_by_profile_id)?;
        intrinsics.push(resolve_record(
            policy,
            source,
            declaration,
            evidence.record,
            &overlay.backend_profile,
            evidence.backend_version,
            evidence.backend_sha256,
            backend_lowerings,
            overlay.intrinsic_abi,
        )?);
    }
    for (_, evidence_id) in evidence_by_profile_id.keys() {
        ensure!(
            overlay
                .intrinsics
                .iter()
                .any(|record| record.id == *evidence_id),
            "evidence exists for unknown catalog ID {evidence_id}"
        );
    }

    Ok(CatalogFile {
        schema: CATALOG_SCHEMA,
        catalog_version: overlay.catalog_version,
        intrinsic_abi: overlay.intrinsic_abi,
        generator_version: env!("CARGO_PKG_VERSION").to_owned(),
        source: CatalogSource {
            llvm_repository: lock.llvm.repository,
            llvm_revision: lock.llvm.revision,
            llvm_tblgen_version: lock.llvm_tblgen.version_line,
            llvm_tblgen_source_revision: lock
                .llvm_tblgen
                .built_from_llvm_revision
                .context("pinned llvm-tblgen has no source revision")?,
        },
        inputs: CatalogInputs {
            imported_sha256,
            overlay_sha256,
            abi_ledger_sha256: sha256_file(&ledger_path)?,
            evidence_sha256: evidence_hashes,
        },
        intrinsics,
    })
}

fn read_overlay(repo_root: &Path, manifest_path: &Path) -> Result<(OverlayFile, String)> {
    let manifest_bytes =
        fs::read(manifest_path).with_context(|| format!("read {}", manifest_path.display()))?;
    let mut overlay: OverlayFile = toml::from_slice(&manifest_bytes)
        .with_context(|| format!("parse {}", manifest_path.display()))?;

    ensure!(
        overlay.intrinsics.is_empty(),
        "overlay.toml must list family shards instead of inline intrinsic records"
    );
    ensure!(
        !overlay.shards.is_empty(),
        "overlay.toml must list at least one family shard"
    );

    let mut previous = None;
    let mut seen = BTreeSet::new();
    let mut hash_input = Vec::new();
    append_overlay_hash_input(&mut hash_input, "intrinsics/overlay.toml", &manifest_bytes);

    for shard_name in &overlay.shards {
        validate_overlay_shard_path(shard_name)?;
        ensure!(
            seen.insert(shard_name.as_str()),
            "overlay.toml lists duplicate shard {shard_name}"
        );
        if let Some(previous) = previous {
            ensure!(
                previous < shard_name.as_str(),
                "overlay.toml shards must be sorted"
            );
        }
        previous = Some(shard_name.as_str());

        let relative = Path::new("intrinsics").join(shard_name);
        let path = repo_root.join(&relative);
        let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let shard: OverlayShardFile =
            toml::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        ensure!(
            shard.schema == OVERLAY_SHARD_SCHEMA,
            "unsupported overlay shard schema {} in {}",
            shard.schema,
            path.display()
        );
        ensure!(
            !shard.intrinsics.is_empty(),
            "overlay shard {} contains no intrinsic records",
            path.display()
        );
        for record in &shard.intrinsics {
            ensure!(
                record.family == shard.family,
                "overlay shard {} declares family {}, but intrinsic {} uses family {}",
                path.display(),
                shard.family,
                record.id,
                record.family
            );
        }

        append_overlay_hash_input(
            &mut hash_input,
            relative
                .to_str()
                .context("overlay shard path is not valid UTF-8")?,
            &bytes,
        );
        overlay.intrinsics.extend(shard.intrinsics);
    }

    Ok((overlay, sha256_bytes(&hash_input)))
}

fn validate_overlay_shard_path(path: &str) -> Result<()> {
    let path = Path::new(path);
    ensure!(
        path.extension().and_then(|extension| extension.to_str()) == Some("toml"),
        "overlay shard path must name a TOML file: {}",
        path.display()
    );
    let components: Vec<_> = path.components().collect();
    ensure!(
        components.len() >= 2 && components[0] == Component::Normal("overlay".as_ref()),
        "overlay shard path must stay under intrinsics/overlay: {}",
        path.display()
    );
    ensure!(
        components
            .iter()
            .all(|component| matches!(component, Component::Normal(_))),
        "overlay shard path contains a non-normal component: {}",
        path.display()
    );
    Ok(())
}

fn append_overlay_hash_input(output: &mut Vec<u8>, path: &str, contents: &[u8]) {
    output.extend_from_slice(&(path.len() as u64).to_le_bytes());
    output.extend_from_slice(path.as_bytes());
    output.extend_from_slice(&(contents.len() as u64).to_le_bytes());
    output.extend_from_slice(contents);
}

fn validate_unique_overlay(records: &[OverlayIntrinsic], intrinsic_abi: u32) -> Result<()> {
    let mut ids = BTreeSet::new();
    let mut abi_ids = BTreeSet::new();
    let mut operation_keys = BTreeSet::new();
    let mut paths = BTreeSet::new();
    let mut op_variants = BTreeSet::new();
    let mut op_type_names = BTreeMap::new();
    let mut symbols = BTreeSet::new();
    let mut rust_items = BTreeSet::new();
    for record in records {
        insert_unique(&mut ids, &record.id, "catalog ID")?;
        validate_abi_id(&record.abi_id)?;
        insert_unique(&mut abi_ids, &record.abi_id, "intrinsic ABI ID")?;
        validate_operation_key(&record.operation_key)?;
        insert_unique(
            &mut operation_keys,
            &record.operation_key,
            "intrinsic operation key",
        )?;
        if let Some(previous_name) = op_type_names.insert(
            record.dialect_op_type.as_str(),
            record.dialect_op_name.as_str(),
        ) {
            ensure!(
                previous_name == record.dialect_op_name,
                "dialect op type {} maps to both {} and {}",
                record.dialect_op_type,
                previous_name,
                record.dialect_op_name
            );
        }
        let op_variant = format!(
            "{}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}",
            record.dialect_op_name,
            record.ldmatrix_variant,
            record.packed_atomic,
            record.redux,
            record.vote,
            record.active_mask,
            record.warp_match,
            record.dot_product,
        );
        insert_unique(&mut op_variants, &op_variant, "dialect op variant")?;
        if let Some(symbol) = &record.llvm_symbol {
            insert_unique(&mut symbols, symbol, "LLVM symbol")?;
        }
        insert_unique(
            &mut rust_items,
            &format!("{}::{}", record.rust_module, record.rust_name),
            "raw Rust item",
        )?;
        insert_unique(
            &mut paths,
            &canonical_rust_path(intrinsic_abi, &record.abi_id),
            "canonical Rust path",
        )?;
        insert_unique(&mut paths, &record.public_rust_path, "public Rust path")?;
        for path in &record.compatibility_rust_paths {
            insert_unique(&mut paths, path, "compatibility Rust path")?;
        }
    }
    Ok(())
}

fn validate_abi_ledger(overlay: &OverlayFile, ledger: &AbiLedgerFile) -> Result<()> {
    ensure!(
        ledger.schema == 1,
        "unsupported ABI ledger schema {}",
        ledger.schema
    );
    ensure!(
        ledger.intrinsic_abi == overlay.intrinsic_abi,
        "ABI ledger v{} does not match overlay ABI v{}",
        ledger.intrinsic_abi,
        overlay.intrinsic_abi
    );
    ensure!(!ledger.entries.is_empty(), "ABI ledger contains no entries");

    let overlay_by_abi_id: BTreeMap<_, _> = overlay
        .intrinsics
        .iter()
        .map(|record| (record.abi_id.as_str(), record))
        .collect();
    let mut abi_ids = BTreeSet::new();
    let mut catalog_ids = BTreeSet::new();
    let mut operation_keys = BTreeSet::new();
    let mut previous_abi_id: Option<&str> = None;
    for entry in &ledger.entries {
        validate_abi_id(&entry.abi_id)?;
        if let Some(previous) = previous_abi_id {
            ensure!(
                previous < entry.abi_id.as_str(),
                "ABI ledger IDs must be unique and append-only in ascending order: {} follows {}",
                entry.abi_id,
                previous
            );
        }
        previous_abi_id = Some(&entry.abi_id);
        insert_unique(&mut abi_ids, &entry.abi_id, "ABI ledger ID")?;
        insert_unique(&mut catalog_ids, &entry.catalog_id, "ABI ledger catalog ID")?;
        validate_operation_key(&entry.operation_key)?;
        insert_unique(
            &mut operation_keys,
            &entry.operation_key,
            "ABI ledger operation key",
        )?;
        ensure!(
            !entry.catalog_id.is_empty()
                && !entry.raw_rust_signature.result.is_empty()
                && entry
                    .raw_rust_signature
                    .arguments
                    .iter()
                    .all(|argument| !argument.is_empty()),
            "ABI ledger entry {} has incomplete identity data",
            entry.abi_id
        );

        let overlay_record = overlay_by_abi_id.get(entry.abi_id.as_str()).copied();
        match entry.status.as_str() {
            "active" => {
                let record = overlay_record.with_context(|| {
                    format!(
                        "active ABI ledger entry {} ({}) has no overlay record",
                        entry.abi_id, entry.catalog_id
                    )
                })?;
                validate_active_ledger_entry(entry, record)?;
            }
            "tombstone" => ensure!(
                overlay_record.is_none(),
                "tombstoned ABI ID {} cannot reappear in the overlay",
                entry.abi_id
            ),
            status => bail!(
                "ABI ledger entry {} has unsupported status {status:?}; expected active or tombstone",
                entry.abi_id
            ),
        }
    }

    for record in &overlay.intrinsics {
        ensure!(
            abi_ids.contains(&record.abi_id),
            "overlay intrinsic {} uses ABI ID {} with no ledger entry",
            record.id,
            record.abi_id
        );
    }
    Ok(())
}

fn validate_active_ledger_entry(entry: &AbiLedgerEntry, record: &OverlayIntrinsic) -> Result<()> {
    let comparisons = [
        ("catalog ID", entry.catalog_id.as_str(), record.id.as_str()),
        (
            "operation key",
            entry.operation_key.as_str(),
            record.operation_key.as_str(),
        ),
    ];
    for (field, ledger_value, overlay_value) in comparisons {
        ensure!(
            ledger_value == overlay_value,
            "ABI ledger {} {field} mismatch: ledger {ledger_value:?}, overlay {overlay_value:?}",
            entry.abi_id
        );
    }
    let expected_signature = raw_rust_signature(record);
    ensure!(
        entry.raw_rust_signature == expected_signature,
        "ABI ledger {} raw Rust signature mismatch: ledger {:?}, overlay {:?}",
        entry.abi_id,
        entry.raw_rust_signature,
        expected_signature
    );
    Ok(())
}

fn raw_rust_signature(record: &OverlayIntrinsic) -> AbiRawRustSignature {
    AbiRawRustSignature {
        safe: record.safe,
        arguments: record.rust_arguments.clone(),
        result: record.rust_result.clone(),
    }
}

fn validate_abi_id(abi_id: &str) -> Result<()> {
    ensure!(
        abi_id.len() == 5
            && abi_id.starts_with('i')
            && abi_id[1..].bytes().all(|byte| byte.is_ascii_digit()),
        "intrinsic ABI ID `{abi_id}` must use the stable `iNNNN` form"
    );
    Ok(())
}

pub(crate) fn validate_operation_key(operation_key: &str) -> Result<()> {
    ensure!(
        !operation_key.is_empty()
            && operation_key.split('.').all(|segment| {
                !segment.is_empty()
                    && segment.bytes().all(|byte| {
                        byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_'
                    })
            }),
        "intrinsic operation key `{operation_key}` must contain dot-separated lowercase semantic components"
    );
    Ok(())
}

fn canonical_rust_path(intrinsic_abi: u32, abi_id: &str) -> String {
    format!("cuda_intrinsics::__cuda_oxide_intrinsic_abi_v{intrinsic_abi}::{abi_id}")
}

fn insert_unique(set: &mut BTreeSet<String>, value: &str, kind: &str) -> Result<()> {
    ensure!(set.insert(value.to_owned()), "duplicate {kind}: {value}");
    Ok(())
}

fn resolve_policy_source(policy: &OverlayIntrinsic) -> Result<IntrinsicSource> {
    match (&policy.source, &policy.source_record) {
        (None, Some(source_record)) => Ok(IntrinsicSource::LlvmImported {
            source_record: source_record.clone(),
        }),
        (Some(source @ IntrinsicSource::PtxNative { .. }), None) => Ok(source.clone()),
        (Some(IntrinsicSource::LlvmImported { source_record }), None) => {
            ensure!(
                !source_record.trim().is_empty(),
                "{} has an empty imported LLVM source record",
                policy.id
            );
            Ok(IntrinsicSource::LlvmImported {
                source_record: source_record.clone(),
            })
        }
        (Some(_), Some(_)) => bail!(
            "{} mixes tagged source provenance with the legacy source_record field",
            policy.id
        ),
        (None, None) => bail!("{} has no intrinsic source provenance", policy.id),
    }
}

fn validate_policy(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
    declaration: Option<&ImportedIntrinsic>,
    intrinsic_abi: u32,
) -> Result<()> {
    validate_abi_id(&policy.abi_id)?;
    parse_ptx_version(&policy.minimum_ptx, &policy.id)?;
    parse_hardware_target(policy)?;
    policy.expected_ptx.validate().map_err(|reason| {
        anyhow::anyhow!(
            "{} has an invalid expected PTX pattern: {reason}",
            policy.id
        )
    })?;
    let public_path = format!(
        "cuda_intrinsics::{}::{}",
        policy.rust_module, policy.rust_name
    );
    ensure!(
        policy.public_rust_path == public_path,
        "{} public Rust path must be {}",
        policy.id,
        public_path
    );
    let canonical_path = canonical_rust_path(intrinsic_abi, &policy.abi_id);
    ensure!(
        canonical_path != policy.public_rust_path
            && !policy
                .compatibility_rust_paths
                .iter()
                .any(|path| path == &canonical_path || path == &policy.public_rust_path),
        "{} must keep canonical, public, and compatibility Rust paths distinct",
        policy.id
    );
    match (source, declaration) {
        (IntrinsicSource::LlvmImported { .. }, Some(declaration)) => {
            ensure!(
                policy.llvm_symbol.as_deref() == Some(declaration.llvm_name.as_str()),
                "{} LLVM symbol mismatch: imported {}, overlay {:?}",
                policy.id,
                declaration.llvm_name,
                policy.llvm_symbol
            );
            ensure!(
                declaration.arguments == policy.llvm_arguments,
                "{} LLVM argument signature mismatch: imported {:?}, overlay {:?}",
                policy.id,
                declaration.arguments,
                policy.llvm_arguments
            );
            ensure!(
                declaration.results == policy.llvm_results,
                "{} LLVM result signature mismatch: imported {:?}, overlay {:?}",
                policy.id,
                declaration.results,
                policy.llvm_results
            );
        }
        (IntrinsicSource::PtxNative { instruction }, None) => ensure!(
            !instruction.trim().is_empty()
                && policy.llvm_symbol.is_none()
                && policy.resolved_llvm_symbol.is_none()
                && policy.llvm_arguments.is_empty()
                && policy.llvm_results.is_empty(),
            "{} PTX-native source must not invent LLVM source facts",
            policy.id
        ),
        _ => bail!(
            "{} source kind and imported declaration disagree",
            policy.id
        ),
    }
    match policy.family.as_str() {
        "sreg" => validate_sreg_policy(
            policy,
            declaration.context("sreg requires imported LLVM declaration")?,
        )?,
        "ldmatrix" => validate_ldmatrix_policy(
            policy,
            declaration.context("ldmatrix requires imported LLVM declaration")?,
        )?,
        "packed_atomic" => validate_packed_atomic_policy(policy, source)?,
        "redux" => validate_redux_policy(
            policy,
            declaration.context("redux requires imported LLVM declaration")?,
        )?,
        "dotprod" => validate_dot_product_policy(
            policy,
            declaration.context("dotprod requires imported LLVM declaration")?,
        )?,
        "sync" => validate_sync_policy(
            policy,
            declaration.context("sync requires imported LLVM declaration")?,
        )?,
        "vote" => validate_vote_policy(
            policy,
            declaration.context("vote requires imported LLVM declaration")?,
        )?,
        "active_mask" => validate_active_mask_policy(
            policy,
            declaration.context("active_mask requires imported LLVM declaration")?,
        )?,
        "warp_match" => validate_warp_match_policy(
            policy,
            declaration.context("warp_match requires imported LLVM declaration")?,
        )?,
        family => bail!("{} uses unsupported generated family {family:?}", policy.id),
    }
    ensure!(
        !policy.execution_scope.trim().is_empty(),
        "{} has no execution scope",
        policy.id
    );
    ensure!(
        !policy.ptx_isa_version.trim().is_empty()
            && !policy.ptx_isa_section.trim().is_empty()
            && policy.ptx_isa_url.starts_with("https://docs.nvidia.com/"),
        "{} has incomplete or non-authoritative PTX ISA provenance",
        policy.id
    );
    match (policy.safe, policy.safe_allowlist_reason.as_deref()) {
        (true, Some(reason)) if !reason.trim().is_empty() => {}
        (true, _) => bail!(
            "{} is safe but has no nonempty safe_allowlist_reason",
            policy.id
        ),
        (false, Some(reason)) if !reason.trim().is_empty() => bail!(
            "{} is unsafe but has a safe_allowlist_reason; safe exceptions apply only to safe items",
            policy.id
        ),
        (false, _) => {}
    }
    if let Some(declaration) = declaration {
        if policy.pure {
            ensure!(
                declaration
                    .classes
                    .iter()
                    .any(|class| class == "NVVMPureIntrinsic"),
                "{} is marked pure, but its imported declaration is not an NVVMPureIntrinsic",
                policy.id
            );
        }
        if policy.memory == "none" {
            ensure!(
                declaration
                    .properties
                    .iter()
                    .any(|property| property == "IntrNoMem"),
                "{} is marked no-memory, but its imported declaration lacks IntrNoMem",
                policy.id
            );
        }
        let imported_convergent = declaration
            .properties
            .iter()
            .any(|property| property == "IntrConvergent");
        ensure!(
            imported_convergent == policy.convergent,
            "{} convergence mismatch: imported {}, overlay {}",
            policy.id,
            imported_convergent,
            policy.convergent
        );
        ensure!(
            !declaration.selections.is_empty(),
            "{} has a declaration but no NVPTX TableGen selection record",
            policy.id
        );
        let matching_selections: Vec<_> = declaration
            .selections
            .iter()
            .filter(|selection| selection_matches_policy(policy, selection))
            .collect();
        let expected_selection_count = match policy.family.as_str() {
            "vote" => 2,
            "warp_match" => 4,
            _ => 1,
        };
        ensure!(
            matching_selections.len() == expected_selection_count,
            "{} expected PTX {:?} does not agree with its closed imported selection set",
            policy.id,
            policy.expected_ptx
        );
        for selection in matching_selections {
            validate_selected_target_predicates(policy, selection)?;
        }
    }
    Ok(())
}

fn selection_matches_policy(
    policy: &OverlayIntrinsic,
    selection: &crate::model::ImportedSelection,
) -> bool {
    if policy.family == "sync" {
        return policy.id == "sync_threads"
            && selection.source_record == "BARRIER_CTA_SYNC_ALIGNED_ALL_i"
            && selection.asm == "bar.sync \t$i;"
            && selection.predicates.is_empty()
            && selection.constraints.address_space.is_none()
            && selection.constraints.immediate_bindings.is_empty();
    }

    if policy.family == "vote" {
        let Some(vote) = &policy.vote else {
            return false;
        };
        let recipe = vote_recipe(vote.mode);
        return [recipe.immediate_selection, recipe.register_selection]
            .contains(&selection.source_record.as_str())
            && policy.expected_ptx.matches(&selection.asm)
            && selection.constraints.address_space.is_none()
            && selection.constraints.immediate_bindings.is_empty();
    }

    if policy.family == "warp_match" {
        let Some(warp_match) = &policy.warp_match else {
            return false;
        };
        let recipe = warp_match_recipe(warp_match.mode, warp_match.value_width);
        return recipe
            .selections
            .contains(&selection.source_record.as_str())
            && policy.expected_ptx.matches(&selection.asm)
            && selection.constraints.is_empty();
    }

    if !policy.expected_ptx.matches(&selection.asm)
        || policy
            .selected_address_space
            .is_some_and(|address_space| selection.constraints.address_space != Some(address_space))
    {
        return false;
    }

    let Some(dot_product) = &policy.dot_product else {
        return true;
    };
    if selection.constraints.address_space.is_some() {
        return false;
    }
    match dot_product.adapter {
        DotProductAdapter::DirectThreeOperands => {
            selection.constraints.immediate_bindings.is_empty()
        }
        DotProductAdapter::InsertLowHalfFalse => {
            selection.constraints.immediate_bindings.len() == 1
                && selection.constraints.immediate_bindings[0].argument_index == 2
                && selection.constraints.immediate_bindings[0].value == 0
        }
    }
}

fn validate_sync_policy(policy: &OverlayIntrinsic, declaration: &ImportedIntrinsic) -> Result<()> {
    ensure!(
        policy.id == "sync_threads"
            && policy.abi_id == "i0034"
            && policy.operation_key == "synchronization.cta.barrier.aligned.all"
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some("int_nvvm_barrier_cta_sync_aligned_all")
            && policy.llvm_symbol.as_deref() == Some("llvm.nvvm.barrier.cta.sync.aligned.all")
            && policy.resolved_llvm_symbol.is_none(),
        "{} sync identity does not match the closed sync_threads recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "thread"
            && policy.rust_name == "sync_threads"
            && policy.rust_arguments.is_empty()
            && policy.rust_result == "()"
            && !policy.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == "cuda_intrinsics::thread::sync_threads"
            && policy.compatibility_rust_paths
                == [
                    "cuda_device::thread::sync_threads",
                    "cuda_device::sync_threads",
                ],
        "{} must preserve the unsafe sync_threads raw API and both cuda-device compatibility paths",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == "Barrier0Op"
            && policy.dialect_op_name == "nvvm.barrier0"
            && policy.dialect_operands.is_empty()
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments == ["i32"]
            && policy.llvm_results.is_empty()
            && policy.lowering == "generated_sync_threads",
        "{} is outside the fixed-zero sync_threads lowering recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "read_write"
            && policy.convergent
            && policy.execution_scope == "cta"
            && policy.minimum_ptx == "1.0"
            && policy.minimum_sm.is_none()
            && policy.ptx_result == "()"
            && policy.targets == "all",
        "{} sync effects or native target floor disagree with the closed recipe",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "9.7.14.1 Parallel Synchronization and Communication Instructions: bar, barrier"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar-barrier",
        "{} sync PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        declaration.properties == ["IntrConvergent", "IntrNoCallback"],
        "{} sync properties disagree with the imported LLVM declaration",
        policy.id
    );
    ensure!(
        policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.packed_atomic.is_none()
            && policy.redux.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.dot_product.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with sync",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "bar"
            && policy.expected_ptx.modifiers == ["sync"]
            && policy.expected_ptx.operands == [OperandPattern::Exact { value: "0".into() }],
        "{} expected PTX does not match literal bar.sync 0",
        policy.id
    );

    let backend_pairs: BTreeSet<_> = policy
        .backend_lowerings
        .iter()
        .map(|lowering| (lowering.backend, lowering.mechanism))
        .collect();
    ensure!(
        policy.backend_lowerings.len() == 2
            && backend_pairs
                == BTreeSet::from([
                    (
                        IntrinsicBackend::LlvmNvptx,
                        BackendLoweringMechanism::TypedNvvm,
                    ),
                    (
                        IntrinsicBackend::LibNvvm,
                        BackendLoweringMechanism::InlinePtx,
                    ),
                ]),
        "{} must define exactly the reviewed LLVM typed and libNVVM inline-PTX routes",
        policy.id
    );
    for lowering in &policy.backend_lowerings {
        let floor_matches = match lowering.backend {
            IntrinsicBackend::LlvmNvptx => {
                lowering.mechanism == BackendLoweringMechanism::TypedNvvm
                    && lowering.minimum_ptx.as_deref() == Some("3.2")
                    && lowering.minimum_sm.as_deref() == Some("sm_20")
            }
            IntrinsicBackend::LibNvvm => {
                lowering.mechanism == BackendLoweringMechanism::InlinePtx
                    && lowering.minimum_ptx.is_none()
                    && lowering.minimum_sm.as_deref() == Some("sm_75")
            }
        };
        ensure!(
            floor_matches && !lowering.evidence_profile.trim().is_empty(),
            "{} backend {:?} does not carry its reviewed sync profile floor",
            policy.id,
            lowering.backend
        );
    }
    Ok(())
}

fn validate_vote_policy(policy: &OverlayIntrinsic, declaration: &ImportedIntrinsic) -> Result<()> {
    let vote = policy
        .vote
        .as_ref()
        .with_context(|| format!("{} has no closed vote contract", policy.id))?;
    let recipe = vote_recipe(vote.mode);
    ensure!(
        vote.participation
            == VoteParticipation::ExecutingLaneNamedAllNamedLanesSameInstructionAndMask
            && vote.adapter == VoteAdapter::DirectMaskPredicate
            && vote.mask_encoding == MaskEncoding::RegisterOrImmediate,
        "{} requests an unsupported vote participation, adapter, or mask encoding",
        policy.id
    );
    ensure!(
        policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none(),
        "{} vote identity does not match its closed mode recipe",
        policy.id
    );
    let expected_compatibility_paths: Vec<String> = if recipe.has_compatibility_path {
        vec![format!("cuda_device::warp::{}", recipe.rust_name)]
    } else {
        vec![]
    };
    ensure!(
        policy.rust_module == "warp"
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == ["u32", "bool"]
            && policy.rust_result == recipe.rust_result
            && !policy.safe
            && policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::warp::{}", recipe.rust_name)
            && policy.compatibility_rust_paths == expected_compatibility_paths,
        "{} must preserve its unsafe must-use vote raw API and reviewed compatibility path",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == ["i32", "i1"]
            && policy.dialect_results == [recipe.llvm_result]
            && policy.llvm_arguments == ["i32", "i1"]
            && policy.llvm_results == [recipe.llvm_result]
            && policy.lowering == "generated_vote",
        "{} is outside the closed two-operand vote lowering recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "inaccessible_read_write"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == "6.0"
            && policy.minimum_sm.as_deref() == Some("sm_30")
            && policy.ptx_result == recipe.rust_result
            && policy.targets == "all",
        "{} vote effects, carrier, or target floor disagree with its mode recipe",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == "9.7.14.10 Warp Vote Instructions: vote.sync"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync",
        "{} vote PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        declaration.properties
            == [
                "IntrConvergent",
                "IntrInaccessibleMemOnly",
                "IntrNoCallback",
            ],
        "{} vote memory and convergence effects disagree with the imported declaration",
        policy.id
    );
    ensure!(
        policy.backend_lowerings.is_empty()
            && policy.packed_atomic.is_none()
            && policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.redux.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.dot_product.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with vote",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "vote"
            && policy.expected_ptx.modifiers == ["sync", recipe.ptx_mode, recipe.ptx_type]
            && policy.expected_ptx.operands
                == [
                    OperandPattern::Register,
                    OperandPattern::Register,
                    OperandPattern::RegisterOrImmediate,
                ],
        "{} expected PTX does not match its closed vote mode recipe",
        policy.id
    );

    let expected_selection_records =
        BTreeSet::from([recipe.immediate_selection, recipe.register_selection]);
    let actual_selection_records: BTreeSet<_> = declaration
        .selections
        .iter()
        .map(|selection| selection.source_record.as_str())
        .collect();
    ensure!(
        declaration.selections.len() == 2 && actual_selection_records == expected_selection_records,
        "{} vote declaration must contain exactly its immediate/register selection pair",
        policy.id
    );
    let expected_asm = format!(
        "vote.sync.{}.{} \t$dest, $pred, $mask;",
        recipe.ptx_mode, recipe.ptx_type
    );
    for selection in &declaration.selections {
        ensure!(
            selection.asm == expected_asm
                && selection.predicates
                    == [
                        "Subtarget->getPTXVersion() >= 60",
                        "Subtarget->getSmVersion() >= 30",
                    ]
                && selection.constraints.is_empty(),
            "{} vote immediate/register selections disagree on PTX shape, target predicates, or constraints",
            policy.id
        );
    }
    Ok(())
}

struct VoteRecipe {
    id: &'static str,
    abi_id: &'static str,
    operation_key: &'static str,
    source_record: &'static str,
    llvm_symbol: &'static str,
    rust_name: &'static str,
    rust_result: &'static str,
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    llvm_result: &'static str,
    ptx_mode: &'static str,
    ptx_type: &'static str,
    immediate_selection: &'static str,
    register_selection: &'static str,
    has_compatibility_path: bool,
}

fn vote_recipe(mode: VoteMode) -> VoteRecipe {
    match mode {
        VoteMode::All => VoteRecipe {
            id: "all_sync",
            abi_id: "i0040",
            operation_key: "warp.vote.sync.all.pred",
            source_record: "int_nvvm_vote_all_sync",
            llvm_symbol: "llvm.nvvm.vote.all.sync",
            rust_name: "all_sync",
            rust_result: "bool",
            dialect_op_type: "VoteSyncAllOp",
            dialect_op_name: "nvvm.vote_sync_all",
            llvm_result: "i1",
            ptx_mode: "all",
            ptx_type: "pred",
            immediate_selection: "VOTE_SYNC_ALLi",
            register_selection: "VOTE_SYNC_ALLr",
            has_compatibility_path: true,
        },
        VoteMode::Any => VoteRecipe {
            id: "any_sync",
            abi_id: "i0041",
            operation_key: "warp.vote.sync.any.pred",
            source_record: "int_nvvm_vote_any_sync",
            llvm_symbol: "llvm.nvvm.vote.any.sync",
            rust_name: "any_sync",
            rust_result: "bool",
            dialect_op_type: "VoteSyncAnyOp",
            dialect_op_name: "nvvm.vote_sync_any",
            llvm_result: "i1",
            ptx_mode: "any",
            ptx_type: "pred",
            immediate_selection: "VOTE_SYNC_ANYi",
            register_selection: "VOTE_SYNC_ANYr",
            has_compatibility_path: true,
        },
        VoteMode::Ballot => VoteRecipe {
            id: "ballot_sync",
            abi_id: "i0042",
            operation_key: "warp.vote.sync.ballot.b32",
            source_record: "int_nvvm_vote_ballot_sync",
            llvm_symbol: "llvm.nvvm.vote.ballot.sync",
            rust_name: "ballot_sync",
            rust_result: "u32",
            dialect_op_type: "VoteSyncBallotOp",
            dialect_op_name: "nvvm.vote_sync_ballot",
            llvm_result: "i32",
            ptx_mode: "ballot",
            ptx_type: "b32",
            immediate_selection: "VOTE_SYNC_BALLOTi",
            register_selection: "VOTE_SYNC_BALLOTr",
            has_compatibility_path: true,
        },
        VoteMode::Uni => VoteRecipe {
            id: "uni_sync",
            abi_id: "i0043",
            operation_key: "warp.vote.sync.uni.pred",
            source_record: "int_nvvm_vote_uni_sync",
            llvm_symbol: "llvm.nvvm.vote.uni.sync",
            rust_name: "uni_sync",
            rust_result: "bool",
            dialect_op_type: "VoteSyncUniOp",
            dialect_op_name: "nvvm.vote_sync_uni",
            llvm_result: "i1",
            ptx_mode: "uni",
            ptx_type: "pred",
            immediate_selection: "VOTE_SYNC_UNIi",
            register_selection: "VOTE_SYNC_UNIr",
            has_compatibility_path: false,
        },
    }
}

fn validate_active_mask_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let active_mask = policy
        .active_mask
        .as_ref()
        .with_context(|| format!("{} has no closed active-mask contract", policy.id))?;
    ensure!(
        active_mask.observation == ActiveMaskObservation::ExecutingLanesAtInstruction
            && active_mask.adapter == ActiveMaskAdapter::DirectZeroOperandMask,
        "{} requests an unsupported active-mask observation or adapter",
        policy.id
    );
    ensure!(
        policy.id == "active_mask"
            && policy.abi_id == "i0044"
            && policy.operation_key == "warp.active_mask"
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some("int_nvvm_activemask")
            && policy.llvm_symbol.as_deref() == Some("llvm.nvvm.activemask")
            && policy.resolved_llvm_symbol.is_none(),
        "{} active-mask identity does not match the closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "warp"
            && policy.rust_name == "active_mask"
            && policy.rust_arguments.is_empty()
            && policy.rust_result == "u32"
            && policy.safe
            && policy.must_use
            && policy
                .safe_allowlist_reason
                .as_deref()
                .is_some_and(|reason| !reason.is_empty())
            && policy.public_rust_path == "cuda_intrinsics::warp::active_mask"
            && policy.compatibility_rust_paths == ["cuda_device::warp::active_mask"],
        "{} must preserve its safe must-use raw and compatibility APIs",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == "ActiveMaskOp"
            && policy.dialect_op_name == "nvvm.activemask"
            && policy.dialect_operands.is_empty()
            && policy.dialect_results == ["i32"]
            && policy.llvm_arguments.is_empty()
            && policy.llvm_results == ["i32"]
            && policy.lowering == "generated_active_mask",
        "{} is outside the closed zero-operand active-mask lowering recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "inaccessible_read_write"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == "6.2"
            && policy.minimum_sm.as_deref() == Some("sm_30")
            && policy.ptx_result == "u32"
            && policy.targets == "all",
        "{} active-mask effects or target floor disagree with the closed recipe",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "9.7.14.12 Parallel Synchronization and Communication Instructions: activemask"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-activemask",
        "{} active-mask PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        declaration.properties
            == [
                "IntrConvergent",
                "IntrHasSideEffects",
                "IntrInaccessibleMemOnly",
                "IntrNoCallback",
            ]
            && declaration.selections.len() == 1
            && declaration.selections[0].source_record == "ACTIVEMASK"
            && declaration.selections[0].asm == "activemask.b32 \t$dest;"
            && declaration.selections[0].predicates
                == [
                    "Subtarget->getSmVersion() >= 30",
                    "Subtarget->getPTXVersion() >= 62",
                ]
            && declaration.selections[0].constraints.is_empty(),
        "{} active-mask declaration or selection facts changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "activemask"
            && policy.expected_ptx.modifiers == ["b32"]
            && policy.expected_ptx.operands == [OperandPattern::Register],
        "{} expected PTX does not match activemask.b32",
        policy.id
    );
    ensure!(
        policy.packed_atomic.is_none()
            && policy.redux.is_none()
            && policy.vote.is_none()
            && policy.warp_match.is_none()
            && policy.dot_product.is_none()
            && policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with active_mask",
        policy.id
    );

    let backend_pairs: BTreeSet<_> = policy
        .backend_lowerings
        .iter()
        .map(|lowering| (lowering.backend, lowering.mechanism))
        .collect();
    ensure!(
        policy.backend_lowerings.len() == 2
            && backend_pairs
                == BTreeSet::from([
                    (
                        IntrinsicBackend::LlvmNvptx,
                        BackendLoweringMechanism::TypedNvvm,
                    ),
                    (
                        IntrinsicBackend::LibNvvm,
                        BackendLoweringMechanism::InlinePtx,
                    ),
                ]),
        "{} must keep the LLVM typed and libNVVM inline-PTX routes explicit",
        policy.id
    );
    for lowering in &policy.backend_lowerings {
        let floor_matches = match lowering.backend {
            IntrinsicBackend::LlvmNvptx => {
                lowering.mechanism == BackendLoweringMechanism::TypedNvvm
                    && lowering.minimum_ptx.as_deref() == Some("6.2")
                    && lowering.minimum_sm.as_deref() == Some("sm_30")
            }
            IntrinsicBackend::LibNvvm => {
                lowering.mechanism == BackendLoweringMechanism::InlinePtx
                    && lowering.minimum_ptx.is_none()
                    && lowering.minimum_sm.as_deref() == Some("sm_75")
            }
        };
        ensure!(
            floor_matches && !lowering.evidence_profile.trim().is_empty(),
            "{} backend {:?} does not carry its reviewed active-mask floor",
            policy.id,
            lowering.backend
        );
    }
    Ok(())
}

struct WarpMatchRecipe {
    id: &'static str,
    abi_id: &'static str,
    operation_key: &'static str,
    source_record: &'static str,
    llvm_symbol: &'static str,
    rust_name: &'static str,
    rust_value: &'static str,
    llvm_value: &'static str,
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    ptx_mode: &'static str,
    ptx_type: &'static str,
    selections: [&'static str; 4],
    adapter: WarpMatchAdapter,
}

fn warp_match_recipe(mode: WarpMatchMode, width: WarpMatchValueWidth) -> WarpMatchRecipe {
    match (mode, width) {
        (WarpMatchMode::Any, WarpMatchValueWidth::B32) => WarpMatchRecipe {
            id: "match_any_sync",
            abi_id: "i0045",
            operation_key: "warp.match.sync.any.b32",
            source_record: "int_nvvm_match_any_sync_i32",
            llvm_symbol: "llvm.nvvm.match.any.sync.i32",
            rust_name: "match_any_sync",
            rust_value: "u32",
            llvm_value: "i32",
            dialect_op_type: "MatchAnySyncI32Op",
            dialect_op_name: "nvvm.match_any_sync_i32",
            ptx_mode: "any",
            ptx_type: "b32",
            selections: [
                "MATCH_ANY_SYNC_32ii",
                "MATCH_ANY_SYNC_32ir",
                "MATCH_ANY_SYNC_32ri",
                "MATCH_ANY_SYNC_32rr",
            ],
            adapter: WarpMatchAdapter::DirectMask,
        },
        (WarpMatchMode::Any, WarpMatchValueWidth::B64) => WarpMatchRecipe {
            id: "match_any_i64_sync",
            abi_id: "i0046",
            operation_key: "warp.match.sync.any.b64",
            source_record: "int_nvvm_match_any_sync_i64",
            llvm_symbol: "llvm.nvvm.match.any.sync.i64",
            rust_name: "match_any_i64_sync",
            rust_value: "u64",
            llvm_value: "i64",
            dialect_op_type: "MatchAnySyncI64Op",
            dialect_op_name: "nvvm.match_any_sync_i64",
            ptx_mode: "any",
            ptx_type: "b64",
            selections: [
                "MATCH_ANY_SYNC_64ii",
                "MATCH_ANY_SYNC_64ir",
                "MATCH_ANY_SYNC_64ri",
                "MATCH_ANY_SYNC_64rr",
            ],
            adapter: WarpMatchAdapter::DirectMask,
        },
        (WarpMatchMode::All, WarpMatchValueWidth::B32) => WarpMatchRecipe {
            id: "match_all_sync",
            abi_id: "i0047",
            operation_key: "warp.match.sync.all.b32",
            source_record: "int_nvvm_match_all_sync_i32p",
            llvm_symbol: "llvm.nvvm.match.all.sync.i32p",
            rust_name: "match_all_sync",
            rust_value: "u32",
            llvm_value: "i32",
            dialect_op_type: "MatchAllSyncI32Op",
            dialect_op_name: "nvvm.match_all_sync_i32",
            ptx_mode: "all",
            ptx_type: "b32",
            selections: [
                "MATCH_ALLP_SYNC_32ii",
                "MATCH_ALLP_SYNC_32ir",
                "MATCH_ALLP_SYNC_32ri",
                "MATCH_ALLP_SYNC_32rr",
            ],
            adapter: WarpMatchAdapter::ProjectMaskDiscardPredicate,
        },
        (WarpMatchMode::All, WarpMatchValueWidth::B64) => WarpMatchRecipe {
            id: "match_all_i64_sync",
            abi_id: "i0048",
            operation_key: "warp.match.sync.all.b64",
            source_record: "int_nvvm_match_all_sync_i64p",
            llvm_symbol: "llvm.nvvm.match.all.sync.i64p",
            rust_name: "match_all_i64_sync",
            rust_value: "u64",
            llvm_value: "i64",
            dialect_op_type: "MatchAllSyncI64Op",
            dialect_op_name: "nvvm.match_all_sync_i64",
            ptx_mode: "all",
            ptx_type: "b64",
            selections: [
                "MATCH_ALLP_SYNC_64ii",
                "MATCH_ALLP_SYNC_64ir",
                "MATCH_ALLP_SYNC_64ri",
                "MATCH_ALLP_SYNC_64rr",
            ],
            adapter: WarpMatchAdapter::ProjectMaskDiscardPredicate,
        },
    }
}

fn validate_warp_match_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let warp_match = policy
        .warp_match
        .as_ref()
        .with_context(|| format!("{} has no closed warp-match contract", policy.id))?;
    let recipe = warp_match_recipe(warp_match.mode, warp_match.value_width);
    ensure!(
        warp_match.participation
            == WarpMatchParticipation::ExecutingLaneNamedAllNamedLanesSameInstructionAndMask
            && warp_match.adapter == recipe.adapter
            && warp_match.value_encoding == MatchOperandEncoding::RegisterOrImmediate
            && warp_match.mask_encoding == MatchOperandEncoding::RegisterOrImmediate,
        "{} requests an unsupported warp-match participation, adapter, or encoding",
        policy.id
    );
    ensure!(
        policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none(),
        "{} warp-match identity does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "warp"
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == ["u32", recipe.rust_value]
            && policy.rust_result == "u32"
            && !policy.safe
            && policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::warp::{}", recipe.rust_name)
            && policy.compatibility_rust_paths
                == [format!("cuda_device::warp::{}", recipe.rust_name)],
        "{} must preserve its unsafe raw and stable compatibility paths",
        policy.id
    );
    let expected_llvm_results: &[&str] = match warp_match.mode {
        WarpMatchMode::Any => &["i32"],
        WarpMatchMode::All => &["i32", "i1"],
    };
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == ["i32", recipe.llvm_value]
            && policy.dialect_results == ["i32"]
            && policy.llvm_arguments == ["i32", recipe.llvm_value]
            && policy.llvm_results == expected_llvm_results
            && policy.lowering == "generated_warp_match",
        "{} is outside the closed two-operand warp-match lowering recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "inaccessible_read_write"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == "6.0"
            && policy.minimum_sm.as_deref() == Some("sm_70")
            && policy.ptx_result == "u32"
            && policy.targets == "all",
        "{} warp-match effects, carrier, or target floor disagree with its recipe",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "9.7.14.11 Parallel Synchronization and Communication Instructions: match.sync"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-match-sync",
        "{} warp-match PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        declaration.properties
            == [
                "IntrConvergent",
                "IntrInaccessibleMemOnly",
                "IntrNoCallback",
            ],
        "{} warp-match effects disagree with the imported declaration",
        policy.id
    );
    ensure!(
        policy.backend_lowerings.is_empty()
            && policy.packed_atomic.is_none()
            && policy.redux.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.dot_product.is_none()
            && policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with warp_match",
        policy.id
    );
    let destination = match warp_match.mode {
        WarpMatchMode::Any => OperandPattern::Register,
        WarpMatchMode::All => OperandPattern::RegisterPredicatePair,
    };
    ensure!(
        policy.expected_ptx.mnemonic == "match"
            && policy.expected_ptx.modifiers == [recipe.ptx_mode, "sync", recipe.ptx_type]
            && policy.expected_ptx.operands
                == [
                    destination,
                    OperandPattern::RegisterOrImmediate,
                    OperandPattern::RegisterOrImmediate,
                ],
        "{} expected PTX does not match its closed match.sync recipe",
        policy.id
    );
    let actual_selection_records: BTreeSet<_> = declaration
        .selections
        .iter()
        .map(|selection| selection.source_record.as_str())
        .collect();
    ensure!(
        declaration.selections.len() == 4
            && actual_selection_records == BTreeSet::from(recipe.selections),
        "{} warp-match declaration must contain exactly ii/ir/ri/rr selections",
        policy.id
    );
    let destination = if warp_match.mode == WarpMatchMode::All {
        "$dest|$pred"
    } else {
        "$dest"
    };
    let expected_asm = format!(
        "match.{}.sync.{} \t{}, $value, $mask;",
        recipe.ptx_mode, recipe.ptx_type, destination
    );
    for selection in &declaration.selections {
        ensure!(
            selection.asm == expected_asm
                && selection.predicates
                    == [
                        "Subtarget->getPTXVersion() >= 60",
                        "Subtarget->getSmVersion() >= 70",
                    ]
                && selection.constraints.is_empty(),
            "{} warp-match selections disagree on PTX shape, predicates, or constraints",
            policy.id
        );
    }
    Ok(())
}

fn validate_selected_target_predicates(
    policy: &OverlayIntrinsic,
    selection: &crate::model::ImportedSelection,
) -> Result<()> {
    let mut imported_ptx = None;
    let mut imported_sm = None;
    let mut has_dot_instructions = false;
    for predicate in &selection.predicates {
        if let Some(value) = predicate.strip_prefix("Subtarget->getPTXVersion() >= ") {
            ensure!(
                imported_ptx.is_none(),
                "{} has duplicate PTX predicates",
                policy.id
            );
            imported_ptx = Some(value.parse::<u16>().with_context(|| {
                format!("{} has malformed PTX predicate {predicate:?}", policy.id)
            })?);
        } else if let Some(value) = predicate.strip_prefix("Subtarget->getSmVersion() >= ") {
            ensure!(
                imported_sm.is_none(),
                "{} has duplicate SM predicates",
                policy.id
            );
            imported_sm = Some(value.parse::<u16>().with_context(|| {
                format!("{} has malformed SM predicate {predicate:?}", policy.id)
            })?);
        } else if predicate == "hasDotInstructions" {
            ensure!(
                policy.family == "dotprod",
                "{} selected instruction uses dot-product target gating outside the dotprod family",
                policy.id
            );
            ensure!(
                !has_dot_instructions && imported_ptx.is_none() && imported_sm.is_none(),
                "{} has duplicate or conflicting dot-product target predicates",
                policy.id
            );
            has_dot_instructions = true;
            imported_ptx = Some(50);
            imported_sm = Some(61);
        } else {
            bail!(
                "{} selected instruction has unsupported target predicate {predicate:?}; target gates must fail closed",
                policy.id
            );
        }
    }
    let overlay_ptx = parse_ptx_version(&policy.minimum_ptx, &policy.id)?.encoded();
    if let Some(imported_ptx) = imported_ptx {
        ensure!(
            overlay_ptx == imported_ptx,
            "{} minimum PTX {} disagrees with selected instruction predicate PTX {}",
            policy.id,
            policy.minimum_ptx,
            format_args!("{}.{}", imported_ptx / 10, imported_ptx % 10)
        );
    }
    if let Some(imported_sm) = imported_sm {
        let overlay_target = parse_hardware_target(policy)?;
        ensure!(
            overlay_target
                == CatalogHardwareTarget::AnyOf {
                    alternatives: vec![CatalogHardwareAlternative::MinimumSm { sm: imported_sm }]
                },
            "{} minimum SM {:?} disagrees with selected instruction predicate sm_{}",
            policy.id,
            policy.minimum_sm,
            imported_sm
        );
    }
    if policy.family == "ldmatrix" {
        ensure!(
            imported_ptx.is_some() && imported_sm.is_some(),
            "{} ldmatrix selection must carry both PTX and SM predicates",
            policy.id
        );
    } else if policy.family == "dotprod" {
        ensure!(
            has_dot_instructions && selection.predicates.len() == 1,
            "{} dotprod selection must carry only the hasDotInstructions predicate",
            policy.id
        );
    } else if matches!(
        policy.family.as_str(),
        "vote" | "active_mask" | "warp_match"
    ) {
        ensure!(
            imported_ptx.is_some() && imported_sm.is_some() && selection.predicates.len() == 2,
            "{} selection must carry exactly its PTX and SM predicates",
            policy.id
        );
    }
    Ok(())
}

fn validate_sreg_policy(policy: &OverlayIntrinsic, declaration: &ImportedIntrinsic) -> Result<()> {
    ensure!(
        policy.rust_arguments.is_empty() && policy.llvm_arguments.is_empty(),
        "{} is not a zero-operand intrinsic; the sreg recipe cannot lower it",
        policy.id
    );
    ensure!(
        matches!(policy.rust_result.as_str(), "u32" | "u64"),
        "{} has unsupported raw scalar result {}",
        policy.id,
        policy.rust_result
    );
    let expected_llvm_result = match policy.rust_result.as_str() {
        "u32" => "i32",
        "u64" => "i64",
        _ => unreachable!(),
    };
    ensure!(
        policy.llvm_results == [expected_llvm_result]
            && policy.ptx_result == policy.rust_result
            && policy.lowering == "direct_nvvm",
        "{} has a signature or lowering outside the scalar direct-NVVM sreg recipe",
        policy.id
    );
    ensure!(
        policy.resolved_llvm_symbol.is_none()
            && policy.backend_lowerings.is_empty()
            && policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.packed_atomic.is_none()
            && policy.redux.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.dot_product.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with an sreg",
        policy.id
    );
    if policy.id.starts_with("lanemask_") {
        validate_lanemask_policy(policy, declaration)?;
    }
    Ok(())
}

fn validate_lanemask_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let (suffix, abi_id, section, op_type) = match policy.id.as_str() {
        "lanemask_lt" => ("lt", "i0035", "10.13", "ReadPtxSregLanemaskLtOp"),
        "lanemask_le" => ("le", "i0036", "10.12", "ReadPtxSregLanemaskLeOp"),
        "lanemask_eq" => ("eq", "i0037", "10.11", "ReadPtxSregLanemaskEqOp"),
        "lanemask_ge" => ("ge", "i0038", "10.14", "ReadPtxSregLanemaskGeOp"),
        "lanemask_gt" => ("gt", "i0039", "10.15", "ReadPtxSregLanemaskGtOp"),
        _ => bail!("{} is not a reviewed lane-mask special register", policy.id),
    };
    ensure!(
        policy.abi_id == abi_id
            && policy.operation_key == format!("warp.lane_mask.{suffix}")
            && policy.source.is_none()
            && policy.source_record.as_deref()
                == Some(format!("int_nvvm_read_ptx_sreg_lanemask_{suffix}").as_str())
            && policy.llvm_symbol.as_deref()
                == Some(format!("llvm.nvvm.read.ptx.sreg.lanemask.{suffix}").as_str())
            && policy.resolved_llvm_symbol.is_none(),
        "{} lane-mask identity does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "sreg"
            && policy.rust_name == policy.id
            && policy.rust_arguments.is_empty()
            && policy.rust_result == "u32"
            && policy.safe
            && policy.must_use
            && policy
                .safe_allowlist_reason
                .as_deref()
                .is_some_and(|reason| !reason.is_empty())
            && policy.public_rust_path == format!("cuda_intrinsics::sreg::{}", policy.id)
            && policy.compatibility_rust_paths == [format!("cuda_device::warp::{}", policy.id)],
        "{} must preserve its safe must-use raw and compatibility APIs",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == op_type
            && policy.dialect_op_name == format!("nvvm.read_ptx_sreg_lanemask_{suffix}")
            && policy.dialect_operands.is_empty()
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments.is_empty()
            && policy.llvm_results == ["i32"]
            && policy.lowering == "direct_nvvm",
        "{} is outside the closed lane-mask lowering recipe",
        policy.id
    );
    ensure!(
        policy.pure
            && policy.memory == "none"
            && !policy.convergent
            && policy.execution_scope == "thread"
            && policy.minimum_ptx == "2.0"
            && policy.minimum_sm.as_deref() == Some("sm_20")
            && policy.ptx_result == "u32"
            && policy.targets == "all",
        "{} lane-mask effects or target floor disagree with PTX",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == format!("{section} Special Registers: %lanemask_{suffix}")
            && policy.ptx_isa_url
                == format!(
                    "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-lanemask-{suffix}"
                ),
        "{} lane-mask PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        declaration.properties == ["IntrNoMem", "IntrSpeculatable", "NoUndef<ret>"],
        "{} lane-mask properties disagree with the imported LLVM declaration",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "mov"
            && policy.expected_ptx.modifiers == ["u32"]
            && policy.expected_ptx.operands
                == [
                    OperandPattern::Register,
                    OperandPattern::Exact {
                        value: format!("%lanemask_{suffix}"),
                    },
                ],
        "{} expected PTX does not match its lane-mask register",
        policy.id
    );
    Ok(())
}

fn validate_redux_policy(policy: &OverlayIntrinsic, declaration: &ImportedIntrinsic) -> Result<()> {
    let redux = policy
        .redux
        .as_ref()
        .with_context(|| format!("{} has no closed redux contract", policy.id))?;
    let recipe = redux_recipe(redux.operation);
    ensure!(
        redux.participation
            == ReduxParticipation::ExecutingLaneNamedAllNamedLanesSameInstructionAndMask
            && redux.adapter == ReduxAdapter::MaskValueToSourceMemberMask,
        "{} requests an unsupported redux participation contract or operand adapter",
        policy.id
    );
    ensure!(
        policy.id == recipe.id
            && policy.operation_key == recipe.operation_key
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none(),
        "{} redux identity does not match its closed operation recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "warp"
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == ["u32", recipe.rust_value]
            && policy.rust_result == recipe.rust_value
            && !policy.safe
            && policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::warp::{}", recipe.rust_name)
            && policy.compatibility_rust_paths
                == [format!("cuda_device::warp::{}", recipe.rust_name)],
        "{} must preserve the unsafe must-use redux raw API and legacy cuda-device compatibility DefPath",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == ["i32", "i32"]
            && policy.dialect_results == ["i32"]
            && policy.llvm_arguments == ["i32", "i32"]
            && policy.llvm_results == ["i32"]
            && policy.lowering == "generated_redux",
        "{} is outside the generated two-operand redux recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "inaccessible_read_write"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == "7.0"
            && policy.minimum_sm.as_deref() == Some("sm_80")
            && policy.ptx_result == recipe.rust_value
            && policy.targets == "all",
        "{} redux effects, carrier, or target floor disagree with its operation recipe",
        policy.id
    );
    ensure!(
        declaration
            .properties
            .iter()
            .any(|property| property == "IntrConvergent")
            && declaration
                .properties
                .iter()
                .any(|property| property == "IntrInaccessibleMemOnly")
            && declaration
                .properties
                .iter()
                .any(|property| property == "IntrNoCallback")
            && !declaration.properties.iter().any(|property| matches!(
                property.as_str(),
                "IntrNoMem" | "IntrReadMem" | "IntrWriteMem"
            )),
        "{} redux memory and convergence effects disagree with the imported declaration",
        policy.id
    );
    ensure!(
        policy.backend_lowerings.is_empty()
            && policy.packed_atomic.is_none()
            && policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.dot_product.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with redux",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "redux"
            && policy.expected_ptx.modifiers == ["sync", recipe.ptx_operation, recipe.ptx_type]
            && policy.expected_ptx.operands
                == [
                    OperandPattern::Register,
                    OperandPattern::Register,
                    OperandPattern::Register,
                ],
        "{} expected PTX does not match its closed redux operation recipe",
        policy.id
    );
    Ok(())
}

struct ReduxRecipe {
    id: &'static str,
    operation_key: &'static str,
    source_record: &'static str,
    llvm_symbol: &'static str,
    rust_name: &'static str,
    rust_value: &'static str,
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    ptx_operation: &'static str,
    ptx_type: &'static str,
}

fn redux_recipe(operation: ReduxOperation) -> ReduxRecipe {
    match operation {
        ReduxOperation::Add => ReduxRecipe {
            id: "redux_sync_add",
            operation_key: "warp.redux.sync.add.wrap32",
            source_record: "int_nvvm_redux_sync_add",
            llvm_symbol: "llvm.nvvm.redux.sync.add",
            rust_name: "redux_sync_add",
            rust_value: "u32",
            dialect_op_type: "ReduxSyncAddOp",
            dialect_op_name: "nvvm.redux_sync_add",
            ptx_operation: "add",
            ptx_type: "s32",
        },
        ReduxOperation::Umin => ReduxRecipe {
            id: "redux_sync_min_u32",
            operation_key: "warp.redux.sync.min.u32",
            source_record: "int_nvvm_redux_sync_umin",
            llvm_symbol: "llvm.nvvm.redux.sync.umin",
            rust_name: "redux_sync_min_u32",
            rust_value: "u32",
            dialect_op_type: "ReduxSyncUminOp",
            dialect_op_name: "nvvm.redux_sync_umin",
            ptx_operation: "min",
            ptx_type: "u32",
        },
        ReduxOperation::Min => ReduxRecipe {
            id: "redux_sync_min_i32",
            operation_key: "warp.redux.sync.min.s32",
            source_record: "int_nvvm_redux_sync_min",
            llvm_symbol: "llvm.nvvm.redux.sync.min",
            rust_name: "redux_sync_min_i32",
            rust_value: "i32",
            dialect_op_type: "ReduxSyncMinOp",
            dialect_op_name: "nvvm.redux_sync_min",
            ptx_operation: "min",
            ptx_type: "s32",
        },
        ReduxOperation::Umax => ReduxRecipe {
            id: "redux_sync_max_u32",
            operation_key: "warp.redux.sync.max.u32",
            source_record: "int_nvvm_redux_sync_umax",
            llvm_symbol: "llvm.nvvm.redux.sync.umax",
            rust_name: "redux_sync_max_u32",
            rust_value: "u32",
            dialect_op_type: "ReduxSyncUmaxOp",
            dialect_op_name: "nvvm.redux_sync_umax",
            ptx_operation: "max",
            ptx_type: "u32",
        },
        ReduxOperation::Max => ReduxRecipe {
            id: "redux_sync_max_i32",
            operation_key: "warp.redux.sync.max.s32",
            source_record: "int_nvvm_redux_sync_max",
            llvm_symbol: "llvm.nvvm.redux.sync.max",
            rust_name: "redux_sync_max_i32",
            rust_value: "i32",
            dialect_op_type: "ReduxSyncMaxOp",
            dialect_op_name: "nvvm.redux_sync_max",
            ptx_operation: "max",
            ptx_type: "s32",
        },
        ReduxOperation::And => ReduxRecipe {
            id: "redux_sync_and",
            operation_key: "warp.redux.sync.and.b32",
            source_record: "int_nvvm_redux_sync_and",
            llvm_symbol: "llvm.nvvm.redux.sync.and",
            rust_name: "redux_sync_and",
            rust_value: "u32",
            dialect_op_type: "ReduxSyncAndOp",
            dialect_op_name: "nvvm.redux_sync_and",
            ptx_operation: "and",
            ptx_type: "b32",
        },
        ReduxOperation::Or => ReduxRecipe {
            id: "redux_sync_or",
            operation_key: "warp.redux.sync.or.b32",
            source_record: "int_nvvm_redux_sync_or",
            llvm_symbol: "llvm.nvvm.redux.sync.or",
            rust_name: "redux_sync_or",
            rust_value: "u32",
            dialect_op_type: "ReduxSyncOrOp",
            dialect_op_name: "nvvm.redux_sync_or",
            ptx_operation: "or",
            ptx_type: "b32",
        },
        ReduxOperation::Xor => ReduxRecipe {
            id: "redux_sync_xor",
            operation_key: "warp.redux.sync.xor.b32",
            source_record: "int_nvvm_redux_sync_xor",
            llvm_symbol: "llvm.nvvm.redux.sync.xor",
            rust_name: "redux_sync_xor",
            rust_value: "u32",
            dialect_op_type: "ReduxSyncXorOp",
            dialect_op_name: "nvvm.redux_sync_xor",
            ptx_operation: "xor",
            ptx_type: "b32",
        },
    }
}

fn validate_dot_product_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let dot_product = policy
        .dot_product
        .as_ref()
        .with_context(|| format!("{} has no closed dot-product contract", policy.id))?;
    let recipe = dot_product_recipe(dot_product.operation, dot_product.signedness);
    ensure!(
        dot_product.adapter == recipe.adapter,
        "{} dot-product source adapter does not match its operation",
        policy.id
    );
    ensure!(
        policy.id == recipe.id
            && policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none(),
        "{} dot-product identity does not match its closed operation recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "dotprod"
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == ["u32", "u32", recipe.rust_value]
            && policy.rust_result == recipe.rust_value
            && policy.safe
            && !policy.must_use
            && policy.public_rust_path == format!("cuda_intrinsics::dotprod::{}", recipe.rust_name)
            && policy.compatibility_rust_paths
                == [format!("cuda_device::dotprod::{}", recipe.rust_name)],
        "{} must preserve the safe, non-must-use dotprod raw API and legacy cuda-device DefPath",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == ["i32", "i32", "i32"]
            && policy.dialect_results == ["i32"]
            && policy.llvm_arguments == recipe.llvm_arguments
            && policy.llvm_results == ["i32"]
            && policy.lowering == "generated_dotprod",
        "{} is outside the closed three-operand dot-product lowering recipe",
        policy.id
    );
    ensure!(
        policy.pure
            && policy.memory == "none"
            && !policy.convergent
            && policy.execution_scope == "thread"
            && policy.minimum_ptx == "5.0"
            && policy.minimum_sm.as_deref() == Some("sm_61")
            && policy.ptx_result == recipe.rust_value
            && policy.targets == "all",
        "{} dot-product effects, carrier, or target floor disagree with its operation recipe",
        policy.id
    );
    ensure!(
        declaration
            .classes
            .iter()
            .any(|class| class == "NVVMPureIntrinsic")
            && declaration.properties == recipe.llvm_properties,
        "{} dot-product effects or immediate contract disagree with the imported declaration",
        policy.id
    );
    ensure!(
        policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.packed_atomic.is_none()
            && policy.redux.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with dotprod",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == recipe.ptx_mnemonic
            && policy.expected_ptx.modifiers == recipe.ptx_modifiers
            && policy.expected_ptx.operands
                == [
                    OperandPattern::Register,
                    OperandPattern::Register,
                    OperandPattern::Register,
                    OperandPattern::Register,
                ],
        "{} expected PTX does not match its closed dot-product recipe",
        policy.id
    );

    let backend_pairs: BTreeSet<_> = policy
        .backend_lowerings
        .iter()
        .map(|lowering| (lowering.backend, lowering.mechanism))
        .collect();
    ensure!(
        policy.backend_lowerings.len() == 2
            && backend_pairs
                == BTreeSet::from([
                    (
                        IntrinsicBackend::LlvmNvptx,
                        BackendLoweringMechanism::TypedNvvm,
                    ),
                    (
                        IntrinsicBackend::LibNvvm,
                        BackendLoweringMechanism::InlinePtx,
                    ),
                ]),
        "{} must define exactly the reviewed LLVM typed and libNVVM inline-PTX routes",
        policy.id
    );
    for lowering in &policy.backend_lowerings {
        let floor_matches = match lowering.backend {
            IntrinsicBackend::LlvmNvptx => {
                lowering.mechanism == BackendLoweringMechanism::TypedNvvm
                    && lowering.minimum_ptx.is_none()
                    && lowering.minimum_sm.is_none()
            }
            IntrinsicBackend::LibNvvm => {
                lowering.mechanism == BackendLoweringMechanism::InlinePtx
                    && lowering.minimum_ptx.is_none()
                    && lowering.minimum_sm.as_deref() == Some("sm_75")
            }
        };
        ensure!(
            floor_matches && !lowering.evidence_profile.trim().is_empty(),
            "{} backend {:?} does not carry its reviewed dot-product profile floor",
            policy.id,
            lowering.backend
        );
    }
    Ok(())
}

struct DotProductRecipe {
    id: &'static str,
    operation_key: &'static str,
    source_record: &'static str,
    llvm_symbol: &'static str,
    rust_name: &'static str,
    rust_value: &'static str,
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    llvm_arguments: &'static [&'static str],
    llvm_properties: &'static [&'static str],
    adapter: DotProductAdapter,
    ptx_mnemonic: &'static str,
    ptx_modifiers: &'static [&'static str],
}

fn dot_product_recipe(
    operation: DotProductOperation,
    signedness: DotProductSignedness,
) -> DotProductRecipe {
    match (operation, signedness) {
        (DotProductOperation::Dp4a, DotProductSignedness::Signed) => DotProductRecipe {
            id: "dp4a_s32",
            operation_key: "integer.dot_product.dp4a.s32",
            source_record: "int_nvvm_idp4a_s_s",
            llvm_symbol: "llvm.nvvm.idp4a.s.s",
            rust_name: "dp4a_s32",
            rust_value: "i32",
            dialect_op_type: "Dp4aS32Op",
            dialect_op_name: "nvvm.dp4a_s32",
            llvm_arguments: &["i32", "i32", "i32"],
            llvm_properties: &["IntrNoMem", "IntrSpeculatable"],
            adapter: DotProductAdapter::DirectThreeOperands,
            ptx_mnemonic: "dp4a",
            ptx_modifiers: &["s32", "s32"],
        },
        (DotProductOperation::Dp4a, DotProductSignedness::Unsigned) => DotProductRecipe {
            id: "dp4a_u32",
            operation_key: "integer.dot_product.dp4a.u32",
            source_record: "int_nvvm_idp4a_u_u",
            llvm_symbol: "llvm.nvvm.idp4a.u.u",
            rust_name: "dp4a_u32",
            rust_value: "u32",
            dialect_op_type: "Dp4aU32Op",
            dialect_op_name: "nvvm.dp4a_u32",
            llvm_arguments: &["i32", "i32", "i32"],
            llvm_properties: &["IntrNoMem", "IntrSpeculatable"],
            adapter: DotProductAdapter::DirectThreeOperands,
            ptx_mnemonic: "dp4a",
            ptx_modifiers: &["u32", "u32"],
        },
        (DotProductOperation::Dp2a, DotProductSignedness::Signed) => DotProductRecipe {
            id: "dp2a_s32",
            operation_key: "integer.dot_product.dp2a.lo.s32",
            source_record: "int_nvvm_idp2a_s_s",
            llvm_symbol: "llvm.nvvm.idp2a.s.s",
            rust_name: "dp2a_s32",
            rust_value: "i32",
            dialect_op_type: "Dp2aS32Op",
            dialect_op_name: "nvvm.dp2a_s32",
            llvm_arguments: &["i32", "i32", "i1", "i32"],
            llvm_properties: &["ImmArg<arg2>", "IntrNoMem", "IntrSpeculatable"],
            adapter: DotProductAdapter::InsertLowHalfFalse,
            ptx_mnemonic: "dp2a",
            ptx_modifiers: &["lo", "s32", "s32"],
        },
        (DotProductOperation::Dp2a, DotProductSignedness::Unsigned) => DotProductRecipe {
            id: "dp2a_u32",
            operation_key: "integer.dot_product.dp2a.lo.u32",
            source_record: "int_nvvm_idp2a_u_u",
            llvm_symbol: "llvm.nvvm.idp2a.u.u",
            rust_name: "dp2a_u32",
            rust_value: "u32",
            dialect_op_type: "Dp2aU32Op",
            dialect_op_name: "nvvm.dp2a_u32",
            llvm_arguments: &["i32", "i32", "i1", "i32"],
            llvm_properties: &["ImmArg<arg2>", "IntrNoMem", "IntrSpeculatable"],
            adapter: DotProductAdapter::InsertLowHalfFalse,
            ptx_mnemonic: "dp2a",
            ptx_modifiers: &["lo", "u32", "u32"],
        },
    }
}

fn validate_ldmatrix_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let variant = policy
        .ldmatrix_variant
        .as_ref()
        .with_context(|| format!("{} has no closed ldmatrix variant", policy.id))?;
    let safety = policy
        .ldmatrix_safety
        .as_ref()
        .with_context(|| format!("{} has no ldmatrix safety contract", policy.id))?;
    ensure!(
        variant.shape == LdmatrixShape::M8n8
            && variant.element == LdmatrixElement::B16
            && variant.state_space == LdmatrixStateSpace::Shared,
        "{} requests an unsupported ldmatrix shape, element, or state space",
        policy.id
    );
    ensure!(
        safety.participation == LdmatrixParticipation::AllWarpLanesSameInstruction
            && safety.address_contract
                == LdmatrixAddressContract::WarpLaneAddressesMappedByMultiplicitySixteenByteAlignedSixteenBytesReadableWithSm75Replication
            && safety.memory_order == LdmatrixMemoryOrder::Weak,
        "{} has an unsupported ldmatrix safety contract",
        policy.id
    );
    let count = variant.multiplicity.register_count();
    let count_name = match variant.multiplicity {
        LdmatrixMultiplicity::X1 => "x1",
        LdmatrixMultiplicity::X2 => "x2",
        LdmatrixMultiplicity::X4 => "x4",
    };
    let trans_record = match variant.layout {
        LdmatrixLayout::Normal => "",
        LdmatrixLayout::Transposed => "_trans",
    };
    let trans_symbol = match variant.layout {
        LdmatrixLayout::Normal => "",
        LdmatrixLayout::Transposed => ".trans",
    };
    let layout_name = match variant.layout {
        LdmatrixLayout::Normal => "normal",
        LdmatrixLayout::Transposed => "transposed",
    };
    let expected_source =
        format!("int_nvvm_ldmatrix_sync_aligned_m8n8_{count_name}{trans_record}_b16");
    let expected_symbol =
        format!("llvm.nvvm.ldmatrix.sync.aligned.m8n8.{count_name}{trans_symbol}.b16");
    let expected_name = format!("ldmatrix_m8n8_{count_name}{trans_record}_b16");
    let expected_result = if count == 1 {
        "u32".to_owned()
    } else {
        format!("[u32; {count}]")
    };
    let expected_adapter = if count == 1 {
        LdmatrixAdapter::SingleResultDirect
    } else {
        LdmatrixAdapter::MultipleResultsToArray
    };
    ensure!(
        policy.source_record.as_deref() == Some(expected_source.as_str())
            && policy.llvm_symbol.as_deref() == Some(expected_symbol.as_str()),
        "{} ldmatrix variant does not match its imported source record or base LLVM symbol",
        policy.id
    );
    ensure!(
        policy.resolved_llvm_symbol.as_deref() == Some(format!("{expected_symbol}.p3").as_str()),
        "{} must keep the imported base symbol distinct from the resolved `.p3` overload",
        policy.id
    );
    ensure!(
        policy.rust_arguments == ["*const u32"]
            && policy.rust_result == expected_result
            && policy.llvm_arguments == ["anyptr"]
            && policy.llvm_results == vec!["i32"; count]
            && policy.ptx_result == policy.rust_result,
        "{} ldmatrix Rust, imported LLVM, and PTX carrier signatures disagree",
        policy.id
    );
    ensure!(
        policy.id == expected_name
            && policy.operation_key
                == format!("matrix.ldmatrix.m8n8.{count_name}.{layout_name}.b16.shared")
            && policy.rust_module == "matrix"
            && policy.rust_name == expected_name
            && !policy.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.compatibility_rust_paths
                == [format!(
                    "cuda_device::wmma::ldmatrix_{count_name}{trans_record}"
                )]
            && policy.lowering == "generated_ldmatrix"
            && policy.ldmatrix_adapter == Some(expected_adapter)
            && policy.selected_address_space == Some(ImportedAddressSpace::Shared),
        "{} must preserve the closed raw/compatibility ldmatrix API, result adapter, and shared selection",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "read"
            && policy.convergent
            && policy.execution_scope == "warp"
            && declaration
                .properties
                .iter()
                .any(|property| property == "IntrArgMemOnly")
            && declaration
                .properties
                .iter()
                .any(|property| property == "IntrReadMem"),
        "{} ldmatrix effects disagree with the imported declaration",
        policy.id
    );
    let backend_pairs: BTreeSet<_> = policy
        .backend_lowerings
        .iter()
        .map(|lowering| (lowering.backend, lowering.mechanism))
        .collect();
    ensure!(
        backend_pairs
            == BTreeSet::from([
                (
                    IntrinsicBackend::LlvmNvptx,
                    BackendLoweringMechanism::TypedNvvm
                ),
                (
                    IntrinsicBackend::LibNvvm,
                    BackendLoweringMechanism::InlinePtx
                ),
            ]),
        "{} must define exactly the reviewed LLVM typed and libNVVM inline-PTX lowerings",
        policy.id
    );
    ensure!(
        policy
            .backend_lowerings
            .iter()
            .all(|lowering| !lowering.evidence_profile.trim().is_empty()),
        "{} backend lowering omits its evidence profile",
        policy.id
    );
    ensure!(
        policy.packed_atomic.is_none()
            && policy.redux.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.dot_product.is_none(),
        "{} mixes another generated-family contract with ldmatrix",
        policy.id
    );
    Ok(())
}

fn validate_packed_atomic_policy(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
) -> Result<()> {
    let packed = policy
        .packed_atomic
        .as_ref()
        .with_context(|| format!("{} has no closed packed-atomic contract", policy.id))?;
    ensure!(
        packed.operation == PackedAtomicOperation::Add
            && packed.state_space == PackedAtomicStateSpace::Global
            && packed.ordering == PackedAtomicOrdering::Relaxed
            && packed.scope == PackedAtomicScope::Gpu
            && packed.rounding == PackedAtomicRounding::NearestEven
            && packed.subnormal == PackedAtomicSubnormal::Preserve
            && packed.atomicity == PackedAtomicAtomicity::PerElement
            && packed.pointer_contract == PackedAtomicPointerContract::MutableGlobalU32Aligned4
            && packed.access_contract
                == PackedAtomicAccessContract::NoMixedWholeWordOrNonAtomicAccess
            && packed.scope_contract == PackedAtomicScopeContract::RacingAtomicsMutuallyInclusive
            && packed.codegen_contract == PackedAtomicCodegenContract::ExactNativeInstruction
            && packed.return_contract
                == PackedAtomicReturnContract::OldValuesPerElementMayBeNoncoherent
            && packed.adapter == PackedAtomicAdapter::OldPackedU32,
        "{} requests an unsupported packed-atomic semantic or safety contract",
        policy.id
    );
    let (format, native_sm, minimum_ptx, minimum_sm, public_name) = match packed.format {
        PackedAtomicFormat::F16x2 => ("f16x2", 60, "6.2", "sm_70", "atom_add_f16x2"),
        PackedAtomicFormat::Bf16x2 => ("bf16x2", 90, "7.8", "sm_90", "atom_add_bf16x2"),
    };
    ensure!(
        packed.native_minimum_sm == native_sm,
        "{} PTX-native hardware floor does not match the selected packed format",
        policy.id
    );
    ensure!(
        source
            == &IntrinsicSource::PtxNative {
                instruction: format!("atom.global.add.noftz.{format}"),
            },
        "{} PTX-native source does not match its packed format",
        policy.id
    );
    ensure!(
        policy.rust_module == "atomic"
            && policy.rust_name == public_name
            && policy.rust_arguments == ["*mut u32", "u32"]
            && policy.rust_result == "u32"
            && !policy.safe
            && policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::atomic::{public_name}")
            && policy.compatibility_rust_paths == [format!("cuda_device::atomic::{public_name}")],
        "{} must preserve the unsafe must-use packed atomic raw/compatibility API",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == "PackedAtomicAddOp"
            && policy.dialect_op_name == "nvvm.packed_atomic_add"
            && policy.dialect_operands == ["ptr", "i32"]
            && policy.dialect_results == ["i32"]
            && policy.lowering == "generated_packed_atomic_inline_ptx",
        "{} is outside the one closed packed-atomic dialect recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "read_write"
            && !policy.convergent
            && policy.execution_scope == "thread"
            && policy.minimum_ptx == minimum_ptx
            && policy.minimum_sm.as_deref() == Some(minimum_sm)
            && policy.targets == "all"
            && policy.ptx_result == "u32"
            && policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.redux.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.dot_product.is_none()
            && policy.selected_address_space.is_none(),
        "{} packed-atomic effects, carrier, or native target floor disagree",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "atom"
            && policy.expected_ptx.modifiers == ["global", "add", "noftz", format]
            && policy.expected_ptx.operands
                == [
                    OperandPattern::Register,
                    OperandPattern::Address,
                    OperandPattern::Register,
                ],
        "{} expected PTX must match the exact packed global add spelling",
        policy.id
    );
    let backend_pairs: BTreeSet<_> = policy
        .backend_lowerings
        .iter()
        .map(|lowering| (lowering.backend, lowering.mechanism))
        .collect();
    ensure!(
        backend_pairs
            == BTreeSet::from([
                (
                    IntrinsicBackend::LlvmNvptx,
                    BackendLoweringMechanism::InlinePtx,
                ),
                (
                    IntrinsicBackend::LibNvvm,
                    BackendLoweringMechanism::InlinePtx,
                ),
            ]),
        "{} must define exactly the reviewed LLVM-NVPTX and libNVVM inline-PTX routes",
        policy.id
    );
    for lowering in &policy.backend_lowerings {
        let expected_sm = match (packed.format, lowering.backend) {
            (PackedAtomicFormat::F16x2, IntrinsicBackend::LlvmNvptx) => "sm_70",
            (PackedAtomicFormat::F16x2, IntrinsicBackend::LibNvvm) => "sm_75",
            (PackedAtomicFormat::Bf16x2, _) => "sm_90",
        };
        ensure!(
            lowering.minimum_ptx.as_deref() == Some(minimum_ptx)
                && lowering.minimum_sm.as_deref() == Some(expected_sm)
                && !lowering.evidence_profile.trim().is_empty(),
            "{} backend {:?} does not carry its exact reviewed profile floor",
            policy.id,
            lowering.backend
        );
    }
    Ok(())
}

fn parse_ptx_version(value: &str, intrinsic_id: &str) -> Result<PtxVersion> {
    value
        .parse()
        .map_err(|reason: String| anyhow::anyhow!("{intrinsic_id} minimum_ptx {value:?}: {reason}"))
}

fn parse_hardware_target(policy: &OverlayIntrinsic) -> Result<CatalogHardwareTarget> {
    parse_hardware_target_fields(&policy.id, &policy.targets, policy.minimum_sm.as_deref())
}

fn parse_hardware_target_fields(
    intrinsic_id: &str,
    targets: &str,
    minimum_sm: Option<&str>,
) -> Result<CatalogHardwareTarget> {
    ensure!(
        targets == "all",
        "{} targets {:?} is not a reviewed hardware rule; use `targets = \"all\"` with optional monotonic `minimum_sm = \"sm_NN\"`",
        intrinsic_id,
        targets
    );
    let Some(minimum_sm) = minimum_sm else {
        return Ok(CatalogHardwareTarget::All);
    };
    let digits = minimum_sm.strip_prefix("sm_").with_context(|| {
        format!(
            "{} minimum_sm {:?} must use unsuffixed sm_NN form",
            intrinsic_id, minimum_sm
        )
    })?;
    ensure!(
        matches!(digits.len(), 2 | 3) && digits.bytes().all(|byte| byte.is_ascii_digit()),
        "{} minimum_sm {:?} must use unsuffixed sm_NN form",
        intrinsic_id,
        minimum_sm
    );
    let sm: u16 = digits
        .parse()
        .with_context(|| format!("{} minimum_sm is too large", intrinsic_id))?;
    ensure!(
        sm > 0 && format!("sm_{sm}") == minimum_sm,
        "{} minimum_sm {:?} is not canonical",
        intrinsic_id,
        minimum_sm
    );
    Ok(CatalogHardwareTarget::AnyOf {
        alternatives: vec![CatalogHardwareAlternative::MinimumSm { sm }],
    })
}

fn backend_target_requirement(
    policy: &OverlayIntrinsic,
    lowering: &crate::model::OverlayBackendLowering,
) -> Result<CatalogTargetRequirement> {
    let minimum_ptx = lowering
        .minimum_ptx
        .as_deref()
        .unwrap_or(&policy.minimum_ptx);
    let minimum_sm = lowering
        .minimum_sm
        .as_deref()
        .or(policy.minimum_sm.as_deref());
    Ok(CatalogTargetRequirement {
        minimum_ptx: parse_ptx_version(minimum_ptx, &policy.id)?,
        hardware: parse_hardware_target_fields(&policy.id, &policy.targets, minimum_sm)?,
    })
}

fn read_evidence(repo_root: &Path) -> Result<(Vec<EvidenceFile>, Vec<String>)> {
    let directory = repo_root.join("intrinsics/evidence");
    let mut paths: Vec<PathBuf> = fs::read_dir(&directory)
        .with_context(|| format!("read {}", directory.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("json"))
        .collect();
    paths.sort();
    ensure!(
        !paths.is_empty(),
        "no evidence JSON files in {}",
        directory.display()
    );
    let mut files = Vec::with_capacity(paths.len());
    let mut hashes = Vec::with_capacity(paths.len());
    for path in paths {
        let file: EvidenceFile = read_json(&path)?;
        ensure!(
            matches!(file.schema, 2..=5),
            "unsupported evidence schema in {}",
            path.display()
        );
        let name = path.file_name().unwrap().to_string_lossy();
        hashes.push(format!("{name}:{}", sha256_file(&path)?));
        files.push(file);
    }
    Ok((files, hashes))
}

#[derive(Debug, Clone, Copy)]
struct IndexedEvidence<'a> {
    file: &'a EvidenceFile,
    record: &'a EvidenceRecord,
    backend_version: &'a str,
    backend_sha256: &'a str,
}

fn index_evidence<'a>(
    files: &'a [EvidenceFile],
    llvm_revision: &str,
) -> Result<BTreeMap<(&'a str, &'a str), IndexedEvidence<'a>>> {
    let mut result = BTreeMap::new();
    for file in files {
        ensure!(
            !file.backend_profile.trim().is_empty() && !file.llvm_revision.trim().is_empty(),
            "evidence file has no concrete backend profile or LLVM revision"
        );
        ensure!(
            !file.backend_version.trim().is_empty() && !file.backend_sha256.trim().is_empty(),
            "evidence does not identify the backend binary"
        );
        if file.backend_kind != Some(IntrinsicBackend::LibNvvm) {
            ensure!(
                file.llvm_revision == llvm_revision,
                "selected evidence LLVM revision {} does not match pinned {}",
                file.llvm_revision,
                llvm_revision
            );
        }
        for record in &file.records {
            ensure!(
                result
                    .insert(
                        (file.backend_profile.as_str(), record.id.as_str()),
                        IndexedEvidence {
                            file,
                            record,
                            backend_version: &file.backend_version,
                            backend_sha256: &file.backend_sha256,
                        },
                    )
                    .is_none(),
                "duplicate evidence for catalog ID {}",
                record.id
            );
        }
    }
    Ok(result)
}

fn validate_evidence(
    policy: &OverlayIntrinsic,
    evidence: &IndexedEvidence<'_>,
    lowering: Option<&crate::model::OverlayBackendLowering>,
) -> Result<()> {
    let record = evidence.record;
    record.expected_ptx.validate().map_err(|reason| {
        anyhow::anyhow!(
            "{} evidence has an invalid expected PTX pattern: {reason}",
            policy.id
        )
    })?;
    let policy_source = resolve_policy_source(policy)?;
    let evidence_source = match (&record.source, &record.source_record) {
        (None, Some(source_record)) => IntrinsicSource::LlvmImported {
            source_record: source_record.clone(),
        },
        (Some(source), None) => source.clone(),
        (Some(_), Some(_)) => bail!(
            "{} evidence mixes tagged source with legacy source_record",
            policy.id
        ),
        (None, None) => bail!("{} evidence has no source provenance", policy.id),
    };
    ensure!(
        evidence_source == policy_source,
        "{} evidence source provenance mismatch",
        policy.id
    );
    ensure!(
        record.llvm_symbol == policy.llvm_symbol
            && record.llvm_arguments == policy.llvm_arguments
            && record.llvm_results == policy.llvm_results,
        "{} evidence signature mismatch",
        policy.id
    );
    if matches!(policy_source, IntrinsicSource::PtxNative { .. }) {
        ensure!(
            record.llvm_symbol.is_none()
                && record.resolved_llvm_symbol.is_none()
                && record.llvm_arguments.is_empty()
                && record.llvm_results.is_empty()
                && record.concrete_llvm_arguments.is_empty()
                && record.concrete_llvm_results.is_empty()
                && record.declaration_attributes_canonicalized.is_none(),
            "{} PTX-native evidence must not invent LLVM declaration facts",
            policy.id
        );
    }
    ensure!(
        record.expected_ptx == policy.expected_ptx,
        "{} evidence PTX expectation mismatch",
        policy.id
    );
    ensure!(
        matches!(record.status.as_str(), "lowered" | "validated" | "executed"),
        "{} evidence status {} is too weak to generate a lowering",
        policy.id,
        record.status
    );
    ensure!(
        !record.target_triple.is_empty()
            && !record.gpu_target.is_empty()
            && !record.ptx_feature.is_empty(),
        "{} evidence omits its full target profile",
        policy.id
    );
    if let Some(lowering) = lowering {
        ensure!(
            evidence.file.backend_kind == Some(lowering.backend),
            "{} evidence profile {} has the wrong backend kind",
            policy.id,
            evidence.file.backend_profile
        );
        match record.status.as_str() {
            "executed" => ensure!(
                record.runtime_validation == Some(RuntimeValidation::Executed),
                "{} executed evidence must record runtime_validation = executed",
                policy.id
            ),
            _ => ensure!(
                record.runtime_validation == Some(RuntimeValidation::Unexecuted),
                "{} non-executed backend evidence must record runtime_validation = unexecuted",
                policy.id
            ),
        }
        ensure!(
            !record.stages.is_empty(),
            "{} backend evidence omits compilation stages",
            policy.id
        );
        ensure!(
            record.stages.iter().any(|stage| {
                stage.stage == EvidenceStageKind::BackendCodegen
                    && stage.mechanism == Some(lowering.mechanism)
                    && stage.outcome == "succeeded"
            }),
            "{} evidence has no successful backend-codegen stage for {:?}",
            policy.id,
            lowering.mechanism
        );
        validate_selected_stage_targets(policy, record, lowering)?;
        if lowering.mechanism == BackendLoweringMechanism::TypedNvvm {
            validate_typed_llvm_evidence(policy, record)?;
        }
        if lowering.backend == IntrinsicBackend::LlvmNvptx
            && matches!(record.status.as_str(), "validated" | "executed")
        {
            ensure!(
                has_valid_ptx_assembly_stage(record, lowering.mechanism),
                "{} validated LLVM-NVPTX evidence requires a successful PTX-assembly stage with exact tool identity",
                policy.id
            );
        } else if lowering.backend == IntrinsicBackend::LibNvvm
            && matches!(record.status.as_str(), "validated" | "executed")
        {
            ensure!(
                has_valid_cubin_device_link_stage(record, lowering.mechanism),
                "{} validated libNVVM evidence requires a successful cubin-producing device-link stage with exact tool identity",
                policy.id
            );
        }
        if record.status == "executed" {
            ensure!(
                record.stages.iter().any(|stage| {
                    stage.stage == EvidenceStageKind::Runtime
                        && stage.mechanism == Some(lowering.mechanism)
                        && stage.outcome == "succeeded"
                }),
                "{} executed evidence requires a successful runtime stage for the selected mechanism",
                policy.id
            );
        }
    }
    Ok(())
}

fn validate_typed_llvm_evidence(policy: &OverlayIntrinsic, record: &EvidenceRecord) -> Result<()> {
    let concrete_arguments = policy
        .llvm_arguments
        .iter()
        .map(|argument| {
            if argument != "anyptr" {
                return Ok(argument.clone());
            }
            match policy.selected_address_space.with_context(|| {
                format!(
                    "{} has a polymorphic LLVM pointer without a selected address space",
                    policy.id
                )
            })? {
                ImportedAddressSpace::Generic => Ok("ptr".into()),
                ImportedAddressSpace::Shared => Ok("ptr addrspace(3)".into()),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        record.resolved_llvm_symbol == policy.resolved_llvm_symbol
            && record.concrete_llvm_arguments == concrete_arguments
            && record.concrete_llvm_results == policy.llvm_results
            && record.declaration_attributes_canonicalized == Some(true),
        "{} typed LLVM evidence does not prove its resolved signature and canonical declaration attributes",
        policy.id
    );
    Ok(())
}

fn validate_selected_stage_targets(
    policy: &OverlayIntrinsic,
    record: &EvidenceRecord,
    lowering: &crate::model::OverlayBackendLowering,
) -> Result<()> {
    for stage in &record.stages {
        ensure!(
            !stage.targets.is_empty(),
            "{} evidence stage {:?} has no targets",
            policy.id,
            stage.stage
        );
        for target in &stage.targets {
            ensure!(
                is_normalized_stage_target(target),
                "{} evidence stage has unsupported target spelling {target:?}",
                policy.id
            );
        }
    }

    let terminal_kind = match lowering.backend {
        IntrinsicBackend::LlvmNvptx => EvidenceStageKind::PtxAssembly,
        IntrinsicBackend::LibNvvm => EvidenceStageKind::DeviceLink,
    };
    let backend = successful_stage(
        record,
        lowering.mechanism,
        EvidenceStageKind::BackendCodegen,
    )
    .with_context(|| {
        format!(
            "{} has no successful selected backend-codegen stage",
            policy.id
        )
    })?;
    let requirement = backend_target_requirement(policy, lowering)?;
    let expected_ptx = requirement.minimum_ptx.encoded();
    let expected_sm = match requirement.hardware {
        CatalogHardwareTarget::AnyOf { alternatives }
            if alternatives.len() == 1
                && matches!(
                    alternatives[0],
                    CatalogHardwareAlternative::MinimumSm { .. }
                ) =>
        {
            match alternatives[0] {
                CatalogHardwareAlternative::MinimumSm { sm } => sm,
                _ => unreachable!(),
            }
        }
        _ => bail!(
            "{} selected backend stages require one minimum SM floor",
            policy.id
        ),
    };
    let mut required_stages = vec![backend];
    if matches!(record.status.as_str(), "validated" | "executed") {
        required_stages.push(
            successful_stage(record, lowering.mechanism, terminal_kind).with_context(|| {
                format!("{} has no successful selected terminal stage", policy.id)
            })?,
        );
    }
    for stage in required_stages {
        let (sm, ptx) = selected_stage_floor(stage)?;
        let sm_matches = if stage.stage == EvidenceStageKind::BackendCodegen {
            sm == expected_sm
        } else {
            // A versioned terminal tool may have dropped an older architecture
            // while still accepting its forward-compatible PTX for a newer SM.
            // Codegen proves the admitted floor; terminal validation proves the
            // exact instruction is accepted at an architecture satisfying it.
            sm >= expected_sm
        };
        let ptx_matches = match lowering.backend {
            IntrinsicBackend::LlvmNvptx => ptx == expected_ptx,
            IntrinsicBackend::LibNvvm => ptx >= expected_ptx,
        };
        ensure!(
            sm_matches && ptx_matches,
            "{} evidence stage {:?} targets sm_{} / PTX {}.{} instead of a compatible target at catalog floor sm_{} / PTX {}.{}",
            policy.id,
            stage.stage,
            sm,
            ptx / 10,
            ptx % 10,
            expected_sm,
            expected_ptx / 10,
            expected_ptx % 10
        );
    }
    if record.status == "executed" {
        let runtime = successful_stage(record, lowering.mechanism, EvidenceStageKind::Runtime)
            .with_context(|| {
                format!(
                    "{} executed evidence has no successful runtime stage",
                    policy.id
                )
            })?;
        let (sm, ptx) = selected_stage_floor(runtime)?;
        let ptx_matches = match lowering.backend {
            IntrinsicBackend::LlvmNvptx => ptx == expected_ptx,
            IntrinsicBackend::LibNvvm => ptx >= expected_ptx,
        };
        ensure!(
            sm >= expected_sm && ptx_matches,
            "{} runtime stage target does not satisfy its catalog floor",
            policy.id
        );
    }
    Ok(())
}

fn successful_stage(
    record: &EvidenceRecord,
    mechanism: BackendLoweringMechanism,
    kind: EvidenceStageKind,
) -> Option<&crate::model::EvidenceStage> {
    record.stages.iter().find(|stage| {
        stage.stage == kind && stage.mechanism == Some(mechanism) && stage.outcome == "succeeded"
    })
}

fn is_normalized_stage_target(target: &str) -> bool {
    if let Some(value) = target.strip_prefix("ptx") {
        return value.len() == 2 && value.bytes().all(|byte| byte.is_ascii_digit());
    }
    let value = target
        .strip_prefix("sm_")
        .or_else(|| target.strip_prefix("compute_"));
    let Some(value) = value else { return false };
    let digits = value.strip_suffix('a').unwrap_or(value);
    matches!(digits.len(), 2 | 3) && digits.bytes().all(|byte| byte.is_ascii_digit())
}

fn selected_stage_floor(stage: &crate::model::EvidenceStage) -> Result<(u16, u16)> {
    let mut sm = None;
    let mut ptx = None;
    for target in &stage.targets {
        if let Some(value) = target.strip_prefix("ptx") {
            let value = value.parse::<u16>()?;
            ensure!(
                ptx.replace(value).is_none(),
                "stage has duplicate PTX targets"
            );
        } else if let Some(value) = target
            .strip_prefix("sm_")
            .or_else(|| target.strip_prefix("compute_"))
        {
            ensure!(
                !value.ends_with('a'),
                "selected stage cannot use suffixed target {target}"
            );
            let value = value.parse::<u16>()?;
            ensure!(
                sm.replace(value).is_none(),
                "stage has duplicate architecture targets"
            );
        }
    }
    Ok((
        sm.context("selected stage has no architecture target")?,
        ptx.context("selected stage has no PTX target")?,
    ))
}

fn has_valid_ptx_assembly_stage(
    record: &EvidenceRecord,
    mechanism: BackendLoweringMechanism,
) -> bool {
    has_valid_tool_stage(record, mechanism, EvidenceStageKind::PtxAssembly)
}

fn has_valid_cubin_device_link_stage(
    record: &EvidenceRecord,
    mechanism: BackendLoweringMechanism,
) -> bool {
    has_valid_tool_stage(record, mechanism, EvidenceStageKind::DeviceLink)
        && successful_stage(record, mechanism, EvidenceStageKind::DeviceLink)
            .is_some_and(|stage| stage.artifact_kind == Some(EvidenceArtifactKind::Cubin))
}

fn has_valid_tool_stage(
    record: &EvidenceRecord,
    mechanism: BackendLoweringMechanism,
    stage_kind: EvidenceStageKind,
) -> bool {
    record.stages.iter().any(|stage| {
        stage.stage == stage_kind
            && stage.mechanism == Some(mechanism)
            && stage.outcome == "succeeded"
            && stage
                .tool_path
                .as_deref()
                .is_some_and(|value| !value.is_empty())
            && stage
                .tool_version
                .as_deref()
                .is_some_and(|value| !value.is_empty())
            && stage.tool_sha256.as_deref().is_some_and(|value| {
                value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
    })
}

fn resolve_backend_lowerings(
    policy: &OverlayIntrinsic,
    evidence_by_profile_id: &BTreeMap<(&str, &str), IndexedEvidence<'_>>,
) -> Result<Vec<CatalogBackendLowering>> {
    let mut resolved = Vec::with_capacity(policy.backend_lowerings.len());
    let mut runtime_states = Vec::with_capacity(policy.backend_lowerings.len());
    for lowering in &policy.backend_lowerings {
        let evidence = evidence_by_profile_id
            .get(&(lowering.evidence_profile.as_str(), policy.id.as_str()))
            .with_context(|| {
                format!(
                    "{} has no evidence in backend profile {}",
                    policy.id, lowering.evidence_profile
                )
            })?;
        validate_evidence(policy, evidence, Some(lowering))?;
        runtime_states.push(evidence.record.runtime_validation);
        resolved.push(CatalogBackendLowering {
            backend: lowering.backend,
            mechanism: lowering.mechanism,
            evidence_profile: lowering.evidence_profile.clone(),
            target: backend_target_requirement(policy, lowering)?,
            version: evidence.backend_version.to_owned(),
            sha256: evidence.backend_sha256.to_owned(),
            artifact_path: evidence.file.artifact_path.clone(),
            build_id_prefix: evidence.file.build_id_prefix.clone(),
            status: evidence.record.status.clone(),
            stages: evidence.record.stages.clone(),
        });
    }
    if let Some(safety) = &policy.ldmatrix_safety {
        match safety.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} overlay says runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} overlay says runtime is executed but no backend evidence has an executed runtime stage",
                policy.id
            ),
        }
    }
    if let Some(packed) = &policy.packed_atomic {
        match packed.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} packed-atomic runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} packed-atomic runtime is executed but no backend evidence records execution",
                policy.id
            ),
        }
    }
    resolved.sort_by_key(|lowering| lowering.backend);
    Ok(resolved)
}

#[allow(clippy::too_many_arguments)]
fn resolve_record(
    policy: &OverlayIntrinsic,
    source: IntrinsicSource,
    declaration: Option<&ImportedIntrinsic>,
    evidence: &EvidenceRecord,
    backend_profile: &str,
    backend_version: &str,
    backend_sha256: &str,
    backend_lowerings: Vec<CatalogBackendLowering>,
    intrinsic_abi: u32,
) -> Result<CatalogIntrinsic> {
    let llvm = if let Some(declaration) = declaration {
        Some(CatalogLlvm {
            symbol: policy
                .llvm_symbol
                .clone()
                .expect("validated imported LLVM symbol"),
            resolved_symbol: policy.resolved_llvm_symbol.clone(),
            arguments: policy.llvm_arguments.clone(),
            results: policy.llvm_results.clone(),
            properties: declaration.properties.clone(),
            result_facts: imported_result_facts(&declaration.properties)?,
        })
    } else {
        None
    };
    let preserves_empty_dialect_signature = policy.family == "sync" && policy.id == "sync_threads";
    let dialect_operands =
        if policy.dialect_operands.is_empty() && !preserves_empty_dialect_signature {
            policy.llvm_arguments.clone()
        } else {
            policy.dialect_operands.clone()
        };
    let dialect_results = if policy.dialect_results.is_empty() && !preserves_empty_dialect_signature
    {
        policy.llvm_results.clone()
    } else {
        policy.dialect_results.clone()
    };
    Ok(CatalogIntrinsic {
        id: policy.id.clone(),
        operation_key: policy.operation_key.clone(),
        family: policy.family.clone(),
        source,
        selections: declaration
            .into_iter()
            .flat_map(|declaration| declaration.selections.iter())
            .filter(|selection| selection_matches_policy(policy, selection))
            .map(|selection| CatalogSelection {
                source_record: selection.source_record.clone(),
                asm: selection.asm.clone(),
                predicates: selection.predicates.clone(),
                constraints: selection.constraints.clone(),
            })
            .collect(),
        rust: CatalogRust {
            abi_id: policy.abi_id.clone(),
            module: policy.rust_module.clone(),
            name: policy.rust_name.clone(),
            arguments: policy.rust_arguments.clone(),
            result: policy.rust_result.clone(),
            safe: policy.safe,
            must_use: policy.must_use,
            safe_allowlist_reason: policy.safe_allowlist_reason.clone(),
            canonical_path: canonical_rust_path(intrinsic_abi, &policy.abi_id),
            public_path: policy.public_rust_path.clone(),
            compatibility_paths: policy.compatibility_rust_paths.clone(),
        },
        dialect: CatalogDialect {
            op_type: policy.dialect_op_type.clone(),
            op_name: policy.dialect_op_name.clone(),
            operands: dialect_operands,
            results: dialect_results,
        },
        llvm,
        semantics: CatalogSemantics {
            pure: policy.pure,
            memory: policy.memory.clone(),
            convergent: policy.convergent,
            execution_scope: policy.execution_scope.clone(),
        },
        target: CatalogTarget {
            minimum_ptx: parse_ptx_version(&policy.minimum_ptx, &policy.id)?,
            hardware: parse_hardware_target(policy)?,
            ptx_result: policy.ptx_result.clone(),
            targets: policy.targets.clone(),
            ptx_isa_version: policy.ptx_isa_version.clone(),
            ptx_isa_section: policy.ptx_isa_section.clone(),
            ptx_isa_url: policy.ptx_isa_url.clone(),
        },
        backend: CatalogBackend {
            profile: backend_profile.to_owned(),
            version: backend_version.to_owned(),
            sha256: backend_sha256.to_owned(),
            status: evidence.status.clone(),
            target_triple: evidence.target_triple.clone(),
            gpu_target: evidence.gpu_target.clone(),
            ptx_feature: evidence.ptx_feature.clone(),
        },
        backend_lowerings,
        packed_atomic: policy.packed_atomic.clone(),
        redux: policy.redux.clone(),
        vote: policy.vote.clone(),
        active_mask: policy.active_mask.clone(),
        warp_match: policy.warp_match.clone(),
        dot_product: policy.dot_product.clone(),
        ldmatrix: policy
            .ldmatrix_variant
            .clone()
            .map(|variant| CatalogLdmatrix {
                variant,
                safety: policy
                    .ldmatrix_safety
                    .clone()
                    .expect("validated ldmatrix safety"),
                adapter: policy.ldmatrix_adapter.expect("validated ldmatrix adapter"),
                selected_address_space: policy
                    .selected_address_space
                    .expect("validated ldmatrix address space"),
            }),
        lowering: policy.lowering.clone(),
        expected_ptx: policy.expected_ptx.clone(),
        summary: policy.summary.clone(),
    })
}

fn imported_result_facts(properties: &[String]) -> Result<CatalogLlvmResultFacts> {
    let no_undef = properties.iter().any(|property| property == "NoUndef<ret>");
    let mut range = None;
    for property in properties {
        let Some(bounds) = property
            .strip_prefix("Range<ret,")
            .and_then(|value| value.strip_suffix('>'))
        else {
            continue;
        };
        let (lower, upper_exclusive) = bounds
            .split_once(',')
            .with_context(|| format!("malformed return range property {property:?}"))?;
        ensure!(
            !lower.is_empty() && !upper_exclusive.is_empty(),
            "malformed return range property {property:?}"
        );
        ensure!(range.is_none(), "duplicate return range properties");
        range = Some(CatalogHalfOpenRange {
            lower: lower.to_owned(),
            upper_exclusive: upper_exclusive.to_owned(),
        });
    }
    Ok(CatalogLlvmResultFacts { no_undef, range })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ImportedSelection;

    fn sreg_pattern(special_register: &str) -> InstructionPattern {
        InstructionPattern::new(
            "mov",
            &["u32"],
            vec![
                OperandPattern::Register,
                OperandPattern::Exact {
                    value: special_register.into(),
                },
            ],
        )
    }

    fn policy() -> OverlayIntrinsic {
        OverlayIntrinsic {
            id: "thread_idx_x".into(),
            abi_id: "i0001".into(),
            operation_key: "launch.thread_index.x".into(),
            family: "sreg".into(),
            source: None,
            source_record: Some("int_nvvm_read_ptx_sreg_tid_x".into()),
            rust_module: "sreg".into(),
            rust_name: "thread_idx_x".into(),
            rust_arguments: vec![],
            rust_result: "u32".into(),
            safe: true,
            must_use: false,
            safe_allowlist_reason: Some("no caller obligations".into()),
            public_rust_path: "cuda_intrinsics::sreg::thread_idx_x".into(),
            compatibility_rust_paths: vec!["cuda_device::thread::threadIdx_x".into()],
            dialect_op_type: "ReadPtxSregTidXOp".into(),
            dialect_op_name: "nvvm.read_ptx_sreg_tid_x".into(),
            dialect_operands: vec![],
            dialect_results: vec![],
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.tid.x".into()),
            resolved_llvm_symbol: None,
            llvm_arguments: vec![],
            llvm_results: vec!["i32".into()],
            pure: true,
            memory: "none".into(),
            convergent: false,
            execution_scope: "thread".into(),
            minimum_ptx: "2.0".into(),
            minimum_sm: None,
            ptx_result: "u32".into(),
            targets: "all".into(),
            ptx_isa_version: "9.3".into(),
            ptx_isa_section: "10.1 Special Registers: %tid".into(),
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/".into(),
            lowering: "direct_nvvm".into(),
            backend_lowerings: vec![],
            packed_atomic: None,
            redux: None,
            vote: None,
            active_mask: None,
            warp_match: None,
            dot_product: None,
            ldmatrix_variant: None,
            ldmatrix_safety: None,
            ldmatrix_adapter: None,
            selected_address_space: None,
            expected_ptx: sreg_pattern("%tid.x"),
            summary: "thread index".into(),
        }
    }

    fn distinct_policy() -> OverlayIntrinsic {
        let mut record = policy();
        record.id = "thread_idx_y".into();
        record.abi_id = "i0002".into();
        record.operation_key = "launch.thread_index.y".into();
        record.source_record = Some("int_nvvm_read_ptx_sreg_tid_y".into());
        record.rust_name = "thread_idx_y".into();
        record.public_rust_path = "cuda_intrinsics::sreg::thread_idx_y".into();
        record.compatibility_rust_paths = vec!["cuda_device::thread::threadIdx_y".into()];
        record.dialect_op_type = "ReadPtxSregTidYOp".into();
        record.dialect_op_name = "nvvm.read_ptx_sreg_tid_y".into();
        record.llvm_symbol = Some("llvm.nvvm.read.ptx.sreg.tid.y".into());
        record.expected_ptx = sreg_pattern("%tid.y");
        record
    }

    fn declaration() -> ImportedIntrinsic {
        ImportedIntrinsic {
            source_record: "int_nvvm_read_ptx_sreg_tid_x".into(),
            llvm_name: "llvm.nvvm.read.ptx.sreg.tid.x".into(),
            arguments: vec![],
            results: vec!["i32".into()],
            classes: vec!["NVVMPureIntrinsic".into()],
            properties: vec![
                "IntrNoMem".into(),
                "IntrSpeculatable".into(),
                "NoUndef<ret>".into(),
                "Range<ret,0,1024>".into(),
            ],
            selections: vec![ImportedSelection {
                source_record: "INT_PTX_SREG_TID_x".into(),
                asm: "mov.u32 $d, %tid.x;".into(),
                predicates: vec![],
                constraints: Default::default(),
            }],
        }
    }

    fn evidence() -> EvidenceRecord {
        EvidenceRecord {
            id: "thread_idx_x".into(),
            source: None,
            source_record: Some("int_nvvm_read_ptx_sreg_tid_x".into()),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.tid.x".into()),
            resolved_llvm_symbol: None,
            llvm_arguments: vec![],
            llvm_results: vec!["i32".into()],
            concrete_llvm_arguments: vec![],
            concrete_llvm_results: vec![],
            target_triple: "nvptx64-nvidia-cuda".into(),
            gpu_target: "sm_70".into(),
            ptx_feature: "+ptx60".into(),
            status: "lowered".into(),
            stages: vec![],
            declaration_attributes_canonicalized: None,
            runtime_validation: None,
            expected_ptx: sreg_pattern("%tid.x"),
        }
    }

    fn validate_test_evidence(policy: &OverlayIntrinsic, record: EvidenceRecord) -> Result<()> {
        let file = EvidenceFile {
            schema: 3,
            backend_profile: "test".into(),
            backend_kind: None,
            llvm_revision: "test".into(),
            backend_version: "LLVM version test".into(),
            backend_sha256: "0123456789abcdef".into(),
            artifact_path: None,
            build_id_prefix: None,
            nvvm_ir_version: None,
            debug_ir_version: None,
            records: vec![record],
        };
        let indexed = IndexedEvidence {
            file: &file,
            record: &file.records[0],
            backend_version: &file.backend_version,
            backend_sha256: &file.backend_sha256,
        };
        validate_evidence(policy, &indexed, None)
    }

    fn overlay_file(records: Vec<OverlayIntrinsic>) -> OverlayFile {
        OverlayFile {
            schema: OVERLAY_SCHEMA,
            catalog_version: "test".into(),
            intrinsic_abi: 1,
            backend_profile: "test".into(),
            shards: vec![],
            intrinsics: records,
        }
    }

    fn validate_imported_policy(
        policy: &OverlayIntrinsic,
        declaration: &ImportedIntrinsic,
    ) -> Result<()> {
        let source = resolve_policy_source(policy)?;
        validate_policy(policy, &source, Some(declaration), 1)
    }

    fn pinned_active_mask_and_warp_match_records()
    -> BTreeMap<String, (OverlayIntrinsic, ImportedIntrinsic)> {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let (overlay, _) =
            read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml")).unwrap();
        let imported: ImportedFile =
            read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
        let declarations: BTreeMap<_, _> = imported
            .intrinsics
            .into_iter()
            .map(|record| (record.source_record.clone(), record))
            .collect();

        overlay
            .intrinsics
            .into_iter()
            .filter(|record| matches!(record.family.as_str(), "active_mask" | "warp_match"))
            .map(|policy| {
                let declaration = declarations[policy.source_record.as_deref().unwrap()].clone();
                (policy.id.clone(), (policy, declaration))
            })
            .collect()
    }

    fn packed_policy(id: &str) -> OverlayIntrinsic {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml"))
            .unwrap()
            .0
            .intrinsics
            .into_iter()
            .find(|record| record.id == id)
            .unwrap()
    }

    fn redux_policy() -> OverlayIntrinsic {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml"))
            .unwrap()
            .0
            .intrinsics
            .into_iter()
            .find(|record| record.id == "redux_sync_add")
            .unwrap()
    }

    fn redux_declaration() -> ImportedIntrinsic {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let text = std::fs::read_to_string(repo_root.join("intrinsics/imported.json")).unwrap();
        serde_json::from_str::<ImportedFile>(&text)
            .unwrap()
            .intrinsics
            .into_iter()
            .find(|record| record.source_record == "int_nvvm_redux_sync_add")
            .unwrap()
    }

    fn sync_policy() -> OverlayIntrinsic {
        packed_policy("sync_threads")
    }

    fn sync_declaration() -> ImportedIntrinsic {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let text = std::fs::read_to_string(repo_root.join("intrinsics/imported.json")).unwrap();
        serde_json::from_str::<ImportedFile>(&text)
            .unwrap()
            .intrinsics
            .into_iter()
            .find(|record| record.source_record == "int_nvvm_barrier_cta_sync_aligned_all")
            .unwrap()
    }

    fn vote_policy(mode: VoteMode) -> OverlayIntrinsic {
        let recipe = vote_recipe(mode);
        let mut record = policy();
        record.id = recipe.id.into();
        record.abi_id = recipe.abi_id.into();
        record.operation_key = recipe.operation_key.into();
        record.family = "vote".into();
        record.source_record = Some(recipe.source_record.into());
        record.rust_module = "warp".into();
        record.rust_name = recipe.rust_name.into();
        record.rust_arguments = vec!["u32".into(), "bool".into()];
        record.rust_result = recipe.rust_result.into();
        record.safe = false;
        record.must_use = true;
        record.safe_allowlist_reason = None;
        record.public_rust_path = format!("cuda_intrinsics::warp::{}", recipe.rust_name);
        record.compatibility_rust_paths = if recipe.has_compatibility_path {
            vec![format!("cuda_device::warp::{}", recipe.rust_name)]
        } else {
            vec![]
        };
        record.dialect_op_type = recipe.dialect_op_type.into();
        record.dialect_op_name = recipe.dialect_op_name.into();
        record.dialect_operands = vec!["i32".into(), "i1".into()];
        record.dialect_results = vec![recipe.llvm_result.into()];
        record.llvm_symbol = Some(recipe.llvm_symbol.into());
        record.llvm_arguments = vec!["i32".into(), "i1".into()];
        record.llvm_results = vec![recipe.llvm_result.into()];
        record.pure = false;
        record.memory = "inaccessible_read_write".into();
        record.convergent = true;
        record.execution_scope = "warp".into();
        record.minimum_ptx = "6.0".into();
        record.minimum_sm = Some("sm_30".into());
        record.ptx_result = recipe.rust_result.into();
        record.ptx_isa_section = "9.7.14.10 Warp Vote Instructions: vote.sync".into();
        record.ptx_isa_url = "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync".into();
        record.lowering = "generated_vote".into();
        record.vote = Some(crate::model::Vote {
            mode,
            participation: VoteParticipation::ExecutingLaneNamedAllNamedLanesSameInstructionAndMask,
            adapter: VoteAdapter::DirectMaskPredicate,
            mask_encoding: MaskEncoding::RegisterOrImmediate,
        });
        record.expected_ptx = InstructionPattern::new(
            "vote",
            &["sync", recipe.ptx_mode, recipe.ptx_type],
            vec![
                OperandPattern::Register,
                OperandPattern::Register,
                OperandPattern::RegisterOrImmediate,
            ],
        );
        record.summary = "warp vote".into();
        record
    }

    fn vote_declaration(mode: VoteMode) -> ImportedIntrinsic {
        let recipe = vote_recipe(mode);
        let selection = |source_record: &str| ImportedSelection {
            source_record: source_record.into(),
            asm: format!(
                "vote.sync.{}.{} \t$dest, $pred, $mask;",
                recipe.ptx_mode, recipe.ptx_type
            ),
            predicates: vec![
                "Subtarget->getPTXVersion() >= 60".into(),
                "Subtarget->getSmVersion() >= 30".into(),
            ],
            constraints: Default::default(),
        };
        ImportedIntrinsic {
            source_record: recipe.source_record.into(),
            llvm_name: recipe.llvm_symbol.into(),
            arguments: vec!["i32".into(), "i1".into()],
            results: vec![recipe.llvm_result.into()],
            classes: vec![
                "ClangBuiltin".into(),
                "NVVMBuiltin".into(),
                "SDPatternOperator".into(),
                "Intrinsic".into(),
            ],
            properties: vec![
                "IntrConvergent".into(),
                "IntrInaccessibleMemOnly".into(),
                "IntrNoCallback".into(),
            ],
            selections: vec![
                selection(recipe.immediate_selection),
                selection(recipe.register_selection),
            ],
        }
    }

    fn sync_evidence(policy: &OverlayIntrinsic) -> EvidenceRecord {
        let mut record = evidence();
        record.id = policy.id.clone();
        record.source_record = policy.source_record.clone();
        record.llvm_symbol = policy.llvm_symbol.clone();
        record.llvm_arguments = policy.llvm_arguments.clone();
        record.llvm_results = policy.llvm_results.clone();
        record.expected_ptx = policy.expected_ptx.clone();
        record
    }

    fn dot_product_policy(
        operation: DotProductOperation,
        signedness: DotProductSignedness,
    ) -> OverlayIntrinsic {
        let recipe = dot_product_recipe(operation, signedness);
        let mut record = policy();
        record.id = recipe.id.into();
        record.abi_id = match (operation, signedness) {
            (DotProductOperation::Dp4a, DotProductSignedness::Signed) => "i0030",
            (DotProductOperation::Dp4a, DotProductSignedness::Unsigned) => "i0031",
            (DotProductOperation::Dp2a, DotProductSignedness::Signed) => "i0032",
            (DotProductOperation::Dp2a, DotProductSignedness::Unsigned) => "i0033",
        }
        .into();
        record.operation_key = recipe.operation_key.into();
        record.family = "dotprod".into();
        record.source = None;
        record.source_record = Some(recipe.source_record.into());
        record.rust_module = "dotprod".into();
        record.rust_name = recipe.rust_name.into();
        record.rust_arguments = vec!["u32".into(), "u32".into(), recipe.rust_value.into()];
        record.rust_result = recipe.rust_value.into();
        record.safe = true;
        record.must_use = false;
        record.safe_allowlist_reason = Some(
            "per-thread integer arithmetic has no memory, pointer, or participation obligations"
                .into(),
        );
        record.public_rust_path = format!("cuda_intrinsics::dotprod::{}", recipe.rust_name);
        record.compatibility_rust_paths =
            vec![format!("cuda_device::dotprod::{}", recipe.rust_name)];
        record.dialect_op_type = recipe.dialect_op_type.into();
        record.dialect_op_name = recipe.dialect_op_name.into();
        record.dialect_operands = vec!["i32".into(), "i32".into(), "i32".into()];
        record.dialect_results = vec!["i32".into()];
        record.llvm_symbol = Some(recipe.llvm_symbol.into());
        record.resolved_llvm_symbol = None;
        record.llvm_arguments = recipe
            .llvm_arguments
            .iter()
            .map(|argument| (*argument).into())
            .collect();
        record.llvm_results = vec!["i32".into()];
        record.pure = true;
        record.memory = "none".into();
        record.convergent = false;
        record.execution_scope = "thread".into();
        record.minimum_ptx = "5.0".into();
        record.minimum_sm = Some("sm_61".into());
        record.ptx_result = recipe.rust_value.into();
        record.targets = "all".into();
        record.lowering = "generated_dotprod".into();
        record.backend_lowerings = vec![
            crate::model::OverlayBackendLowering {
                backend: IntrinsicBackend::LlvmNvptx,
                mechanism: BackendLoweringMechanism::TypedNvvm,
                evidence_profile: "llvm-test".into(),
                minimum_ptx: None,
                minimum_sm: None,
            },
            crate::model::OverlayBackendLowering {
                backend: IntrinsicBackend::LibNvvm,
                mechanism: BackendLoweringMechanism::InlinePtx,
                evidence_profile: "libnvvm-test".into(),
                minimum_ptx: None,
                minimum_sm: Some("sm_75".into()),
            },
        ];
        record.dot_product = Some(crate::model::DotProduct {
            operation,
            signedness,
            adapter: recipe.adapter,
        });
        record.expected_ptx = InstructionPattern::new(
            recipe.ptx_mnemonic,
            recipe.ptx_modifiers,
            vec![OperandPattern::Register; 4],
        );
        record.summary = "packed integer dot product".into();
        record
    }

    fn dot_product_declaration(
        operation: DotProductOperation,
        signedness: DotProductSignedness,
    ) -> ImportedIntrinsic {
        let recipe = dot_product_recipe(operation, signedness);
        let selection = |source_record: &str, half: Option<(&str, i64)>| ImportedSelection {
            source_record: source_record.into(),
            asm: format!(
                "{}.{} $dst, $a, $b, $c;",
                recipe.ptx_mnemonic,
                match half {
                    Some((name, _)) => {
                        let types = &recipe.ptx_modifiers[1..];
                        format!("{name}.{}", types.join("."))
                    }
                    None => recipe.ptx_modifiers.join("."),
                }
            ),
            predicates: vec!["hasDotInstructions".into()],
            constraints: crate::model::ImportedSelectionConstraints {
                address_space: None,
                immediate_bindings: half
                    .map(|(_, value)| {
                        vec![crate::model::ImportedImmediateBinding {
                            argument_index: 2,
                            value,
                        }]
                    })
                    .unwrap_or_default(),
            },
        };
        let selections = match operation {
            DotProductOperation::Dp4a => vec![selection("DOT4", None)],
            DotProductOperation::Dp2a => vec![
                selection("DOT2_hi", Some(("hi", -1))),
                selection("DOT2_lo", Some(("lo", 0))),
            ],
        };
        ImportedIntrinsic {
            source_record: recipe.source_record.into(),
            llvm_name: recipe.llvm_symbol.into(),
            arguments: recipe
                .llvm_arguments
                .iter()
                .map(|argument| (*argument).into())
                .collect(),
            results: vec!["i32".into()],
            classes: vec!["NVVMPureIntrinsic".into()],
            properties: recipe
                .llvm_properties
                .iter()
                .map(|property| (*property).into())
                .collect(),
            selections,
        }
    }

    fn dot_product_evidence(policy: &OverlayIntrinsic) -> EvidenceRecord {
        let mut record = evidence();
        record.id = policy.id.clone();
        record.source_record = policy.source_record.clone();
        record.llvm_symbol = policy.llvm_symbol.clone();
        record.llvm_arguments = policy.llvm_arguments.clone();
        record.llvm_results = policy.llvm_results.clone();
        record.concrete_llvm_arguments = policy.llvm_arguments.clone();
        record.concrete_llvm_results = policy.llvm_results.clone();
        record.declaration_attributes_canonicalized = Some(true);
        record.gpu_target = "sm_61".into();
        record.ptx_feature = "+ptx50".into();
        record.expected_ptx = policy.expected_ptx.clone();
        record
    }

    fn validate_ptx_native_policy(policy: &OverlayIntrinsic) -> Result<()> {
        let source = resolve_policy_source(policy)?;
        validate_policy(policy, &source, None, 1)
    }

    fn ledger_entry(record: &OverlayIntrinsic) -> AbiLedgerEntry {
        AbiLedgerEntry {
            abi_id: record.abi_id.clone(),
            status: "active".into(),
            catalog_id: record.id.clone(),
            operation_key: record.operation_key.clone(),
            raw_rust_signature: raw_rust_signature(record),
        }
    }

    fn ledger(entries: Vec<AbiLedgerEntry>) -> AbiLedgerFile {
        AbiLedgerFile {
            schema: 1,
            intrinsic_abi: 1,
            entries,
        }
    }

    #[test]
    fn duplicate_values_are_rejected() {
        let mut values = BTreeSet::new();
        insert_unique(&mut values, "thread_idx_x", "catalog ID").unwrap();
        let error = insert_unique(&mut values, "thread_idx_x", "catalog ID").unwrap_err();
        assert!(error.to_string().contains("duplicate catalog ID"));
    }

    #[test]
    fn overlay_manifest_loads_sorted_family_shards() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let (overlay, hash) =
            read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml")).unwrap();
        assert_eq!(overlay.schema, OVERLAY_SCHEMA);
        assert_eq!(overlay.shards.len(), 9);
        assert_eq!(overlay.intrinsics.len(), 48);
        assert_eq!(
            overlay
                .intrinsics
                .iter()
                .filter(|record| record.family == "active_mask")
                .count(),
            1
        );
        assert_eq!(
            overlay
                .intrinsics
                .iter()
                .filter(|record| record.family == "dotprod")
                .count(),
            4
        );
        assert_eq!(
            overlay
                .intrinsics
                .iter()
                .filter(|record| record.family == "ldmatrix")
                .count(),
            6
        );
        assert_eq!(
            overlay
                .intrinsics
                .iter()
                .filter(|record| record.family == "sync")
                .count(),
            1
        );
        assert_eq!(
            overlay
                .intrinsics
                .iter()
                .filter(|record| record.family == "vote")
                .count(),
            4
        );
        assert_eq!(
            overlay
                .intrinsics
                .iter()
                .filter(|record| record.family == "warp_match")
                .count(),
            4
        );
        assert_eq!(hash.len(), 64);

        for invalid in [
            "../outside.toml",
            "/absolute.toml",
            "other/family.toml",
            "overlay/../outside.toml",
            "overlay/not-toml.json",
        ] {
            assert!(validate_overlay_shard_path(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn pinned_active_mask_and_warp_match_recipes_resolve() {
        let records = pinned_active_mask_and_warp_match_records();
        assert_eq!(records.len(), 5);

        for (policy, declaration) in records.values() {
            validate_imported_policy(policy, declaration).unwrap();
        }
    }

    #[test]
    fn active_mask_and_warp_match_recipes_fail_closed() {
        let records = pinned_active_mask_and_warp_match_records();
        let reject =
            |policy: &OverlayIntrinsic, declaration: &ImportedIntrinsic, expected: &str| {
                let error = validate_imported_policy(policy, declaration).unwrap_err();
                let message = error.to_string();
                assert!(message.contains(expected), "unexpected error: {message}");
            };

        let (active_mask, active_mask_declaration) = &records["active_mask"];
        let mut wrong_identity = active_mask.clone();
        wrong_identity.operation_key = "warp.active_mask.changed".into();
        reject(
            &wrong_identity,
            active_mask_declaration,
            "active-mask identity",
        );

        let mut wrong_effects = active_mask.clone();
        wrong_effects.memory = "none".into();
        reject(
            &wrong_effects,
            active_mask_declaration,
            "active-mask effects or target floor",
        );

        let (match_any, match_any_declaration) = &records["match_any_sync"];
        let mut wrong_adapter = match_any.clone();
        wrong_adapter.warp_match.as_mut().unwrap().adapter =
            WarpMatchAdapter::ProjectMaskDiscardPredicate;
        reject(
            &wrong_adapter,
            match_any_declaration,
            "warp-match participation, adapter, or encoding",
        );

        let (match_all, match_all_declaration) = &records["match_all_sync"];
        let mut wrong_projection = match_all.clone();
        wrong_projection.dialect_results.push("i1".into());
        reject(
            &wrong_projection,
            match_all_declaration,
            "two-operand warp-match lowering recipe",
        );

        let (match_any_i64, match_any_i64_declaration) = &records["match_any_i64_sync"];
        let mut incomplete_selections = match_any_i64_declaration.clone();
        incomplete_selections.selections.pop();
        reject(
            match_any_i64,
            &incomplete_selections,
            "exactly ii/ir/ri/rr selections",
        );

        let (match_all_i64, match_all_i64_declaration) = &records["match_all_i64_sync"];
        let mut wrong_predicates = match_all_i64_declaration.clone();
        wrong_predicates.selections[0].predicates[0] = "Subtarget->getPTXVersion() >= 61".into();
        reject(
            match_all_i64,
            &wrong_predicates,
            "PTX shape, predicates, or constraints",
        );
    }

    #[test]
    fn ptx_native_source_provenance_fails_closed() {
        let mut mixed = packed_policy("packed_atomic_add_f16x2");
        mixed.source_record = Some("invented_llvm_record".into());
        assert!(
            resolve_policy_source(&mixed)
                .unwrap_err()
                .to_string()
                .contains("mixes tagged source provenance")
        );

        let mut fake_llvm = packed_policy("packed_atomic_add_f16x2");
        fake_llvm.llvm_symbol = Some("llvm.fake.packed.atomic".into());
        fake_llvm.llvm_arguments = vec!["ptr".into(), "i32".into()];
        fake_llvm.llvm_results = vec!["i32".into()];
        assert!(
            validate_ptx_native_policy(&fake_llvm)
                .unwrap_err()
                .to_string()
                .contains("must not invent LLVM source facts")
        );

        let mut wrong_instruction = packed_policy("packed_atomic_add_f16x2");
        wrong_instruction.source = Some(IntrinsicSource::PtxNative {
            instruction: "atom.global.add.noftz.bf16x2".into(),
        });
        assert!(
            validate_ptx_native_policy(&wrong_instruction)
                .unwrap_err()
                .to_string()
                .contains("does not match its packed format")
        );

        let mut wrong_kind = packed_policy("packed_atomic_add_f16x2");
        wrong_kind.source = Some(IntrinsicSource::LlvmImported {
            source_record: "invented_llvm_record".into(),
        });
        assert!(
            validate_ptx_native_policy(&wrong_kind)
                .unwrap_err()
                .to_string()
                .contains("source kind and imported declaration disagree")
        );
    }

    #[test]
    fn packed_atomic_closed_semantics_reject_every_unreviewed_mutation() {
        let valid = packed_policy("packed_atomic_add_f16x2");
        validate_ptx_native_policy(&valid).unwrap();

        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let overlay =
            std::fs::read_to_string(repo_root.join("intrinsics/overlay/packed_atomic.toml"))
                .unwrap();
        for (field, accepted, rejected) in [
            (
                "state space",
                "state_space = \"global\"",
                "state_space = \"shared\"",
            ),
            (
                "ordering",
                "ordering = \"relaxed\"",
                "ordering = \"acquire\"",
            ),
            ("scope", "scope = \"gpu\"", "scope = \"system\""),
            (
                "rounding",
                "rounding = \"nearest_even\"",
                "rounding = \"toward_zero\"",
            ),
            (
                "subnormal",
                "subnormal = \"preserve\"",
                "subnormal = \"flush\"",
            ),
            (
                "atomicity",
                "atomicity = \"per_element\"",
                "atomicity = \"coherent_pair\"",
            ),
            (
                "pointer safety",
                "pointer_contract = \"mutable_global_u32_aligned4\"",
                "pointer_contract = \"unaligned\"",
            ),
            (
                "mixed access safety",
                "access_contract = \"no_mixed_whole_word_or_non_atomic_access\"",
                "access_contract = \"mixed_access_allowed\"",
            ),
            (
                "scope safety",
                "scope_contract = \"racing_atomics_mutually_inclusive\"",
                "scope_contract = \"scope_unchecked\"",
            ),
            (
                "codegen",
                "codegen_contract = \"exact_native_instruction\"",
                "codegen_contract = \"semantic_equivalence\"",
            ),
        ] {
            let mutated = overlay.replacen(accepted, rejected, 1);
            let error = toml::from_str::<OverlayShardFile>(&mutated).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains(rejected.split(" = ").next().unwrap()),
                "{field} mutation did not fail closed: {error}"
            );
        }

        let mut safe = valid;
        safe.safe = true;
        safe.safe_allowlist_reason = Some("incorrectly claims no caller obligations".into());
        assert!(
            validate_ptx_native_policy(&safe)
                .unwrap_err()
                .to_string()
                .contains("unsafe must-use packed atomic")
        );
    }

    #[test]
    fn redux_contract_validates_effects_participation_and_operand_adapter() {
        let valid = redux_policy();
        let imported = redux_declaration();
        validate_imported_policy(&valid, &imported).unwrap();

        assert_eq!(
            valid.redux.as_ref().unwrap().adapter,
            ReduxAdapter::MaskValueToSourceMemberMask
        );

        let mut missing_contract = valid.clone();
        missing_contract.redux = None;
        assert!(
            validate_imported_policy(&missing_contract, &imported)
                .unwrap_err()
                .to_string()
                .contains("closed redux contract")
        );

        let mut wrong_effect = valid.clone();
        wrong_effect.memory = "none".into();
        assert!(
            validate_imported_policy(&wrong_effect, &imported)
                .unwrap_err()
                .to_string()
                .contains("redux effects")
        );

        let mut missing_imported_effect = imported;
        missing_imported_effect
            .properties
            .retain(|property| property != "IntrInaccessibleMemOnly");
        assert!(
            validate_imported_policy(&valid, &missing_imported_effect)
                .unwrap_err()
                .to_string()
                .contains("memory and convergence effects")
        );
    }

    #[test]
    fn vote_modes_keep_exact_abi_identity_and_both_selection_encodings() {
        for mode in [
            VoteMode::All,
            VoteMode::Any,
            VoteMode::Ballot,
            VoteMode::Uni,
        ] {
            let policy = vote_policy(mode);
            let declaration = vote_declaration(mode);
            validate_imported_policy(&policy, &declaration).unwrap();

            let selected: Vec<_> = declaration
                .selections
                .iter()
                .filter(|selection| selection_matches_policy(&policy, selection))
                .collect();
            assert_eq!(selected.len(), 2);
            assert!(selected.iter().any(|selection| {
                selection.source_record == vote_recipe(mode).immediate_selection
            }));
            assert!(selected.iter().any(|selection| {
                selection.source_record == vote_recipe(mode).register_selection
            }));

            let mut record = evidence();
            record.id = policy.id.clone();
            record.source_record = policy.source_record.clone();
            record.llvm_symbol = policy.llvm_symbol.clone();
            record.llvm_arguments = policy.llvm_arguments.clone();
            record.llvm_results = policy.llvm_results.clone();
            record.expected_ptx = policy.expected_ptx.clone();
            let resolved = resolve_record(
                &policy,
                resolve_policy_source(&policy).unwrap(),
                Some(&declaration),
                &record,
                "test",
                "LLVM version test",
                "0123456789abcdef",
                vec![],
                1,
            )
            .unwrap();
            assert_eq!(resolved.selections.len(), 2);
            assert_eq!(resolved.vote, policy.vote);
        }
    }

    #[test]
    fn vote_contract_rejects_unreviewed_identity_effect_and_selection_changes() {
        let valid = vote_policy(VoteMode::All);
        let declaration = vote_declaration(VoteMode::All);

        let mut wrong_abi = valid.clone();
        wrong_abi.abi_id = "i0041".into();
        assert!(
            validate_imported_policy(&wrong_abi, &declaration)
                .unwrap_err()
                .to_string()
                .contains("vote identity")
        );

        let mut safe = valid.clone();
        safe.safe = true;
        safe.safe_allowlist_reason = Some("incorrectly hides participation obligations".into());
        assert!(
            validate_imported_policy(&safe, &declaration)
                .unwrap_err()
                .to_string()
                .contains("unsafe must-use vote")
        );

        let mut wrong_memory = valid.clone();
        wrong_memory.memory = "none".into();
        assert!(
            validate_imported_policy(&wrong_memory, &declaration)
                .unwrap_err()
                .to_string()
                .contains("vote effects")
        );

        let mut register_only_mask = valid.clone();
        register_only_mask.expected_ptx.operands[2] = OperandPattern::Register;
        assert!(
            validate_imported_policy(&register_only_mask, &declaration)
                .unwrap_err()
                .to_string()
                .contains("expected PTX")
        );

        let mut one_selection = declaration.clone();
        one_selection.selections.pop();
        assert!(
            validate_imported_policy(&valid, &one_selection)
                .unwrap_err()
                .to_string()
                .contains("immediate/register selection pair")
        );

        let mut different_predicates = declaration;
        different_predicates.selections[1].predicates[0] =
            "Subtarget->getPTXVersion() >= 61".into();
        assert!(
            validate_imported_policy(&valid, &different_predicates)
                .unwrap_err()
                .to_string()
                .contains("disagree on PTX shape")
        );
    }

    #[test]
    fn uni_vote_is_raw_only_while_existing_votes_keep_compatibility_paths() {
        for mode in [VoteMode::All, VoteMode::Any, VoteMode::Ballot] {
            assert_eq!(vote_policy(mode).compatibility_rust_paths.len(), 1);
        }
        let uni = vote_policy(VoteMode::Uni);
        assert!(uni.compatibility_rust_paths.is_empty());

        let mut invented_compatibility_path = uni.clone();
        invented_compatibility_path.compatibility_rust_paths =
            vec!["cuda_device::warp::uni_sync".into()];
        assert!(
            validate_imported_policy(
                &invented_compatibility_path,
                &vote_declaration(VoteMode::Uni),
            )
            .unwrap_err()
            .to_string()
            .contains("reviewed compatibility path")
        );
    }

    #[test]
    fn sync_threads_selects_only_the_fixed_immediate_barrier_recipe() {
        let policy = sync_policy();
        let declaration = sync_declaration();
        validate_imported_policy(&policy, &declaration).unwrap();

        let selected: Vec<_> = declaration
            .selections
            .iter()
            .filter(|selection| selection_matches_policy(&policy, selection))
            .collect();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].source_record, "BARRIER_CTA_SYNC_ALIGNED_ALL_i");
        assert_eq!(selected[0].asm, "bar.sync \t$i;");
        assert!(policy.expected_ptx.matches("bar.sync 0;"));
        assert!(!policy.expected_ptx.matches(&selected[0].asm));
        assert_eq!(policy.minimum_ptx, "1.0");
        assert!(policy.minimum_sm.is_none());
        let llvm_route = policy
            .backend_lowerings
            .iter()
            .find(|lowering| lowering.backend == IntrinsicBackend::LlvmNvptx)
            .unwrap();
        assert_eq!(llvm_route.minimum_ptx.as_deref(), Some("3.2"));
        assert_eq!(llvm_route.minimum_sm.as_deref(), Some("sm_20"));

        let resolved = resolve_record(
            &policy,
            resolve_policy_source(&policy).unwrap(),
            Some(&declaration),
            &sync_evidence(&policy),
            "test",
            "LLVM version test",
            "0123456789abcdef",
            vec![],
            1,
        )
        .unwrap();
        assert!(resolved.dialect.operands.is_empty());
        assert!(resolved.dialect.results.is_empty());
        assert_eq!(resolved.selections.len(), 1);
        assert_eq!(
            resolved.selections[0].source_record,
            "BARRIER_CTA_SYNC_ALIGNED_ALL_i"
        );
    }

    #[test]
    fn sync_threads_recipe_rejects_unreviewed_selection_effect_and_floor_changes() {
        let valid = sync_policy();
        let declaration = sync_declaration();

        let mut register_only = declaration.clone();
        register_only
            .selections
            .retain(|selection| selection.source_record.ends_with("_r"));
        assert!(
            validate_imported_policy(&valid, &register_only)
                .unwrap_err()
                .to_string()
                .contains("does not agree")
        );

        let mut wrong_properties = declaration.clone();
        wrong_properties.properties.pop();
        assert!(
            validate_imported_policy(&valid, &wrong_properties)
                .unwrap_err()
                .to_string()
                .contains("sync properties")
        );

        let mut wrong_source = valid.clone();
        wrong_source.source_record = Some("int_nvvm_barrier0".into());
        assert!(
            validate_imported_policy(&wrong_source, &declaration)
                .unwrap_err()
                .to_string()
                .contains("sync identity")
        );

        let mut wrong_signature = valid.clone();
        wrong_signature.llvm_arguments.clear();
        assert!(
            validate_imported_policy(&wrong_signature, &declaration)
                .unwrap_err()
                .to_string()
                .contains("LLVM argument signature mismatch")
        );

        let mut wrong_path = valid.clone();
        wrong_path.compatibility_rust_paths.swap(0, 1);
        assert!(
            validate_imported_policy(&wrong_path, &declaration)
                .unwrap_err()
                .to_string()
                .contains("both cuda-device compatibility paths")
        );

        let mut safe = valid.clone();
        safe.safe = true;
        safe.safe_allowlist_reason = Some("incorrectly hides the participation contract".into());
        assert!(
            validate_imported_policy(&safe, &declaration)
                .unwrap_err()
                .to_string()
                .contains("unsafe sync_threads raw API")
        );

        let mut wrong_effect = valid.clone();
        wrong_effect.memory = "none".into();
        assert!(
            validate_imported_policy(&wrong_effect, &declaration)
                .unwrap_err()
                .to_string()
                .contains("sync effects")
        );

        let mut native_floor = valid.clone();
        native_floor.minimum_sm = Some("sm_75".into());
        assert!(
            validate_imported_policy(&native_floor, &declaration)
                .unwrap_err()
                .to_string()
                .contains("native target floor")
        );

        let mut missing_profile_floor = valid;
        missing_profile_floor
            .backend_lowerings
            .iter_mut()
            .find(|lowering| lowering.backend == IntrinsicBackend::LibNvvm)
            .unwrap()
            .minimum_sm = None;
        assert!(
            validate_imported_policy(&missing_profile_floor, &declaration)
                .unwrap_err()
                .to_string()
                .contains("profile floor")
        );

        let mut wrong_llvm_floor = sync_policy();
        wrong_llvm_floor
            .backend_lowerings
            .iter_mut()
            .find(|lowering| lowering.backend == IntrinsicBackend::LlvmNvptx)
            .unwrap()
            .minimum_ptx = None;
        assert!(
            validate_imported_policy(&wrong_llvm_floor, &declaration)
                .unwrap_err()
                .to_string()
                .contains("profile floor")
        );
    }

    #[test]
    fn every_integer_redux_variant_matches_its_closed_recipe() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let (overlay, _) =
            read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml")).unwrap();
        let imported: ImportedFile =
            read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
        assert_eq!(imported.schema, IMPORTED_SCHEMA);
        let declarations: BTreeMap<_, _> = imported
            .intrinsics
            .iter()
            .map(|record| (record.source_record.as_str(), record))
            .collect();
        let redux: Vec<_> = overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "redux")
            .collect();
        assert_eq!(redux.len(), 8);

        for policy in redux {
            let declaration = declarations
                .get(policy.source_record.as_deref().unwrap())
                .unwrap();
            validate_imported_policy(policy, declaration).unwrap();
        }

        let mut mismatched = packed_policy("redux_sync_min_u32");
        mismatched.redux.as_mut().unwrap().operation = ReduxOperation::Umax;
        let declaration = declarations["int_nvvm_redux_sync_umin"];
        assert!(
            validate_imported_policy(&mismatched, declaration)
                .unwrap_err()
                .to_string()
                .contains("closed operation recipe")
        );
    }

    #[test]
    fn every_dot_product_variant_matches_its_closed_recipe() {
        let variants = [
            (
                DotProductOperation::Dp4a,
                DotProductSignedness::Signed,
                "dp4a_s32",
                "int_nvvm_idp4a_s_s",
                "integer.dot_product.dp4a.s32",
            ),
            (
                DotProductOperation::Dp4a,
                DotProductSignedness::Unsigned,
                "dp4a_u32",
                "int_nvvm_idp4a_u_u",
                "integer.dot_product.dp4a.u32",
            ),
            (
                DotProductOperation::Dp2a,
                DotProductSignedness::Signed,
                "dp2a_s32",
                "int_nvvm_idp2a_s_s",
                "integer.dot_product.dp2a.lo.s32",
            ),
            (
                DotProductOperation::Dp2a,
                DotProductSignedness::Unsigned,
                "dp2a_u32",
                "int_nvvm_idp2a_u_u",
                "integer.dot_product.dp2a.lo.u32",
            ),
        ];

        for (operation, signedness, id, source_record, operation_key) in variants {
            let policy = dot_product_policy(operation, signedness);
            let declaration = dot_product_declaration(operation, signedness);
            assert_eq!(policy.id, id);
            assert_eq!(policy.source_record.as_deref(), Some(source_record));
            assert_eq!(policy.operation_key, operation_key);
            validate_imported_policy(&policy, &declaration).unwrap();
        }
    }

    #[test]
    fn pinned_dot_product_records_match_the_reviewed_overlay() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let (overlay, _) =
            read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml")).unwrap();
        let imported: ImportedFile =
            read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
        let declarations: BTreeMap<_, _> = imported
            .intrinsics
            .iter()
            .map(|record| (record.source_record.as_str(), record))
            .collect();
        let dot_products: Vec<_> = overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "dotprod")
            .collect();
        assert_eq!(dot_products.len(), 4);

        for policy in dot_products {
            let declaration = declarations[policy.source_record.as_deref().unwrap()];
            validate_imported_policy(policy, declaration).unwrap();
            let selected: Vec<_> = declaration
                .selections
                .iter()
                .filter(|selection| selection_matches_policy(policy, selection))
                .collect();
            assert_eq!(selected.len(), 1);
            if policy.id.starts_with("dp2a") {
                assert_eq!(selected[0].constraints.immediate_bindings[0].value, 0);
            }
        }
    }

    #[test]
    fn dp2a_selects_only_the_reviewed_low_half_binding() {
        let policy = dot_product_policy(DotProductOperation::Dp2a, DotProductSignedness::Signed);
        let declaration =
            dot_product_declaration(DotProductOperation::Dp2a, DotProductSignedness::Signed);
        let resolved = resolve_record(
            &policy,
            resolve_policy_source(&policy).unwrap(),
            Some(&declaration),
            &dot_product_evidence(&policy),
            "test",
            "LLVM version test",
            "0123456789abcdef",
            vec![],
            1,
        )
        .unwrap();

        assert_eq!(resolved.selections.len(), 1);
        assert_eq!(resolved.selections[0].source_record, "DOT2_lo");
        assert_eq!(
            resolved.selections[0].constraints.immediate_bindings,
            [crate::model::ImportedImmediateBinding {
                argument_index: 2,
                value: 0,
            }]
        );
        assert_eq!(
            resolved.dot_product.as_ref().unwrap().adapter,
            DotProductAdapter::InsertLowHalfFalse
        );

        let mut wrong_binding = declaration;
        wrong_binding.selections[1].constraints.immediate_bindings[0].value = -1;
        let error = validate_imported_policy(&policy, &wrong_binding).unwrap_err();
        assert!(error.to_string().contains("does not agree"));
    }

    #[test]
    fn dot_product_recipe_rejects_unreviewed_api_and_adapter_changes() {
        let valid = dot_product_policy(DotProductOperation::Dp2a, DotProductSignedness::Unsigned);
        let declaration =
            dot_product_declaration(DotProductOperation::Dp2a, DotProductSignedness::Unsigned);

        let mut wrong_adapter = valid.clone();
        wrong_adapter.dot_product.as_mut().unwrap().adapter =
            DotProductAdapter::DirectThreeOperands;
        assert!(
            validate_imported_policy(&wrong_adapter, &declaration)
                .unwrap_err()
                .to_string()
                .contains("source adapter")
        );

        let mut must_use = valid.clone();
        must_use.must_use = true;
        assert!(
            validate_imported_policy(&must_use, &declaration)
                .unwrap_err()
                .to_string()
                .contains("non-must-use")
        );

        let mut wrong_llvm_signature = valid;
        wrong_llvm_signature.llvm_arguments = vec!["i32".into(); 3];
        assert!(
            validate_imported_policy(&wrong_llvm_signature, &declaration)
                .unwrap_err()
                .to_string()
                .contains("LLVM argument signature mismatch")
        );
    }

    #[test]
    fn dot_product_target_predicate_is_closed_to_ptx50_and_sm61() {
        let policy = dot_product_policy(DotProductOperation::Dp4a, DotProductSignedness::Signed);
        let selection =
            &dot_product_declaration(DotProductOperation::Dp4a, DotProductSignedness::Signed)
                .selections[0];
        validate_selected_target_predicates(&policy, selection).unwrap();

        let mut wrong_ptx = policy.clone();
        wrong_ptx.minimum_ptx = "5.1".into();
        assert!(
            validate_selected_target_predicates(&wrong_ptx, selection)
                .unwrap_err()
                .to_string()
                .contains("minimum PTX")
        );

        let mut wrong_sm = policy;
        wrong_sm.minimum_sm = Some("sm_60".into());
        assert!(
            validate_selected_target_predicates(&wrong_sm, selection)
                .unwrap_err()
                .to_string()
                .contains("minimum SM")
        );
    }

    #[test]
    fn typed_evidence_accepts_direct_scalar_intrinsic_signatures() {
        let policy = dot_product_policy(DotProductOperation::Dp2a, DotProductSignedness::Signed);
        let mut record = dot_product_evidence(&policy);
        validate_typed_llvm_evidence(&policy, &record).unwrap();

        record.concrete_llvm_arguments.remove(2);
        let error = validate_typed_llvm_evidence(&policy, &record).unwrap_err();
        assert!(error.to_string().contains("resolved signature"));
    }

    #[test]
    fn duplicate_identity_surfaces_are_rejected_independently() {
        let first = policy();

        let mut second = distinct_policy();
        second.abi_id = first.abi_id.clone();
        assert!(
            validate_unique_overlay(&[first.clone(), second], 1)
                .unwrap_err()
                .to_string()
                .contains("duplicate intrinsic ABI ID")
        );

        let mut second = distinct_policy();
        second.operation_key = first.operation_key.clone();
        assert!(
            validate_unique_overlay(&[first.clone(), second], 1)
                .unwrap_err()
                .to_string()
                .contains("duplicate intrinsic operation key")
        );

        let mut second = distinct_policy();
        second.public_rust_path = first.public_rust_path.clone();
        assert!(
            validate_unique_overlay(&[first.clone(), second], 1)
                .unwrap_err()
                .to_string()
                .contains("duplicate public Rust path")
        );

        let mut second = distinct_policy();
        second.dialect_op_name = first.dialect_op_name.clone();
        assert!(
            validate_unique_overlay(&[first.clone(), second], 1)
                .unwrap_err()
                .to_string()
                .contains("duplicate dialect op variant")
        );

        let mut second = distinct_policy();
        second.llvm_symbol = first.llvm_symbol.clone();
        assert!(
            validate_unique_overlay(&[first, second], 1)
                .unwrap_err()
                .to_string()
                .contains("duplicate LLVM symbol")
        );
    }

    #[test]
    fn signature_and_evidence_mismatches_are_rejected() {
        let mut imported = declaration();
        imported.results = vec!["i64".into()];
        assert!(
            validate_imported_policy(&policy(), &imported)
                .unwrap_err()
                .to_string()
                .contains("LLVM result signature mismatch")
        );

        let mut backend_evidence = evidence();
        backend_evidence.llvm_results = vec!["i64".into()];
        assert!(
            validate_test_evidence(&policy(), backend_evidence)
                .unwrap_err()
                .to_string()
                .contains("evidence signature mismatch")
        );

        let mut backend_evidence = evidence();
        backend_evidence.expected_ptx = sreg_pattern("%tid.y");
        assert!(
            validate_test_evidence(&policy(), backend_evidence)
                .unwrap_err()
                .to_string()
                .contains("evidence PTX expectation mismatch")
        );
    }

    #[test]
    fn validated_llvm_evidence_requires_exact_ptxas_identity() {
        let mut record = evidence();
        record.status = "validated".into();
        record.stages.push(crate::model::EvidenceStage {
            targets: vec!["sm_75".into()],
            representation: "probe PTX".into(),
            stage: EvidenceStageKind::PtxAssembly,
            mechanism: Some(BackendLoweringMechanism::TypedNvvm),
            outcome: "succeeded".into(),
            detail: "accepted".into(),
            artifact_kind: None,
            tool_path: Some("/usr/local/cuda/bin/ptxas".into()),
            tool_version: Some("CUDA 13.3 V13.3.33".into()),
            tool_sha256: Some(
                "7fdd01a4cf50e30746da98989c9272a907f491e6fd7fecfda14642e4375f88fb".into(),
            ),
        });
        assert!(has_valid_ptx_assembly_stage(
            &record,
            BackendLoweringMechanism::TypedNvvm
        ));

        let stage = record.stages.last_mut().unwrap();
        stage.tool_path = None;
        assert!(!has_valid_ptx_assembly_stage(
            &record,
            BackendLoweringMechanism::TypedNvvm
        ));
        record.stages.clear();
        assert!(!has_valid_ptx_assembly_stage(
            &record,
            BackendLoweringMechanism::TypedNvvm
        ));
    }

    #[test]
    fn validated_libnvvm_evidence_requires_a_real_cubin_terminal() {
        let mut record = evidence();
        record.stages.push(crate::model::EvidenceStage {
            targets: vec!["sm_90".into(), "ptx78".into()],
            representation: "linked output".into(),
            stage: EvidenceStageKind::DeviceLink,
            mechanism: Some(BackendLoweringMechanism::InlinePtx),
            outcome: "succeeded".into(),
            detail: "test".into(),
            artifact_kind: None,
            tool_path: Some("/usr/local/cuda-13.3/lib64/libnvJitLink.so.13.3.33".into()),
            tool_version: Some("V13.3.33".into()),
            tool_sha256: Some(
                "3ba1e744347cd68617b862eccfd98b125482e882b7a6319f42abc9a768513db8".into(),
            ),
        });
        assert!(!has_valid_cubin_device_link_stage(
            &record,
            BackendLoweringMechanism::InlinePtx
        ));
        record.stages[0].artifact_kind = Some(EvidenceArtifactKind::Cubin);
        assert!(has_valid_cubin_device_link_stage(
            &record,
            BackendLoweringMechanism::InlinePtx
        ));
    }

    fn evidence_stage(
        stage: EvidenceStageKind,
        mechanism: BackendLoweringMechanism,
        targets: &[&str],
    ) -> crate::model::EvidenceStage {
        crate::model::EvidenceStage {
            targets: targets.iter().map(|target| (*target).into()).collect(),
            representation: "test".into(),
            stage,
            mechanism: Some(mechanism),
            outcome: "succeeded".into(),
            detail: "test".into(),
            artifact_kind: None,
            tool_path: None,
            tool_version: None,
            tool_sha256: None,
        }
    }

    #[test]
    fn backend_stage_targets_and_executed_status_are_monotonic() {
        let mut target_policy = policy();
        target_policy.minimum_ptx = "6.5".into();
        target_policy.minimum_sm = Some("sm_75".into());
        let lowering = crate::model::OverlayBackendLowering {
            backend: IntrinsicBackend::LlvmNvptx,
            mechanism: BackendLoweringMechanism::TypedNvvm,
            evidence_profile: "test".into(),
            minimum_ptx: None,
            minimum_sm: None,
        };
        let mut record = evidence();
        record.status = "validated".into();
        record.runtime_validation = Some(RuntimeValidation::Unexecuted);
        record.stages = vec![
            evidence_stage(
                EvidenceStageKind::BackendCodegen,
                BackendLoweringMechanism::TypedNvvm,
                &["sm_75", "ptx65"],
            ),
            evidence_stage(
                EvidenceStageKind::PtxAssembly,
                BackendLoweringMechanism::TypedNvvm,
                &["sm_75", "ptx65"],
            ),
        ];
        validate_selected_stage_targets(&target_policy, &record, &lowering).unwrap();

        record.stages[1].targets = vec!["sm_80".into(), "ptx65".into()];
        validate_selected_stage_targets(&target_policy, &record, &lowering).unwrap();

        record.stages[1].targets = vec!["sm_74".into(), "ptx65".into()];
        assert!(
            validate_selected_stage_targets(&target_policy, &record, &lowering)
                .unwrap_err()
                .to_string()
                .contains("catalog floor sm_75")
        );

        record.stages[1].targets = vec!["sm_75".into(), "ptx65".into()];
        record.status = "executed".into();
        record.runtime_validation = Some(RuntimeValidation::Executed);
        assert!(
            validate_selected_stage_targets(&target_policy, &record, &lowering)
                .unwrap_err()
                .to_string()
                .contains("runtime stage")
        );
    }

    #[test]
    fn libnvvm_stage_may_report_newer_ptx_than_the_native_instruction_floor() {
        let mut target_policy = policy();
        target_policy.minimum_ptx = "1.0".into();
        target_policy.minimum_sm = None;
        let lowering = crate::model::OverlayBackendLowering {
            backend: IntrinsicBackend::LibNvvm,
            mechanism: BackendLoweringMechanism::InlinePtx,
            evidence_profile: "test".into(),
            minimum_ptx: None,
            minimum_sm: Some("sm_75".into()),
        };
        let mut record = evidence();
        record.stages = vec![evidence_stage(
            EvidenceStageKind::BackendCodegen,
            BackendLoweringMechanism::InlinePtx,
            &["sm_75", "ptx93"],
        )];
        validate_selected_stage_targets(&target_policy, &record, &lowering).unwrap();

        record.stages[0].targets = vec!["sm_75".into(), "ptx09".into()];
        assert!(
            validate_selected_stage_targets(&target_policy, &record, &lowering)
                .unwrap_err()
                .to_string()
                .contains("catalog floor sm_75 / PTX 1.0")
        );

        let llvm_lowering = crate::model::OverlayBackendLowering {
            backend: IntrinsicBackend::LlvmNvptx,
            mechanism: BackendLoweringMechanism::TypedNvvm,
            evidence_profile: "test".into(),
            minimum_ptx: Some("3.2".into()),
            minimum_sm: Some("sm_20".into()),
        };
        record.stages = vec![evidence_stage(
            EvidenceStageKind::BackendCodegen,
            BackendLoweringMechanism::TypedNvvm,
            &["sm_20", "ptx93"],
        )];
        assert!(
            validate_selected_stage_targets(&target_policy, &record, &llvm_lowering)
                .unwrap_err()
                .to_string()
                .contains("catalog floor sm_20 / PTX 3.2")
        );
    }

    #[test]
    fn imported_selection_must_match_the_full_ptx_shape() {
        let mut imported = declaration();
        imported.selections[0].asm = "mov.u32 $d, %tid.xy;".into();
        let error = validate_imported_policy(&policy(), &imported).unwrap_err();
        assert!(error.to_string().contains("does not agree"));

        imported.selections[0].asm = "mov.u32.relaxed $d, %tid.x;".into();
        let error = validate_imported_policy(&policy(), &imported).unwrap_err();
        assert!(error.to_string().contains("does not agree"));

        imported.selections[0].asm = "mov.u32 $d, %tid.x;".into();
        validate_imported_policy(&policy(), &imported).unwrap();
    }

    #[test]
    fn selected_target_predicates_fail_closed() {
        let selection = ImportedSelection {
            source_record: "selection".into(),
            asm: "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{$r0, $r1, $r2, $r3}}, [$src];".into(),
            predicates: vec![
                "Subtarget->getPTXVersion() >= 65".into(),
                "Subtarget->getSmVersion() >= 75".into(),
            ],
            constraints: Default::default(),
        };

        let mut too_low_ptx = policy();
        too_low_ptx.minimum_ptx = "6.4".into();
        too_low_ptx.minimum_sm = Some("sm_75".into());
        assert!(
            validate_selected_target_predicates(&too_low_ptx, &selection)
                .unwrap_err()
                .to_string()
                .contains("minimum PTX")
        );

        let mut too_low_sm = policy();
        too_low_sm.minimum_ptx = "6.5".into();
        too_low_sm.minimum_sm = Some("sm_74".into());
        assert!(
            validate_selected_target_predicates(&too_low_sm, &selection)
                .unwrap_err()
                .to_string()
                .contains("minimum SM")
        );

        let mut unknown = selection;
        unknown
            .predicates
            .push("Subtarget->hasMysteryFeature()".into());
        assert!(
            validate_selected_target_predicates(&too_low_sm, &unknown)
                .unwrap_err()
                .to_string()
                .contains("fail closed")
        );
    }

    #[test]
    fn safe_record_requires_an_allowlist_reason() {
        let mut record = policy();
        record.safe_allowlist_reason = None;
        assert!(
            validate_imported_policy(&record, &declaration())
                .unwrap_err()
                .to_string()
                .contains("safe_allowlist_reason")
        );
    }

    #[test]
    fn intrinsic_abi_identity_is_stable_and_explicit() {
        let policy = policy();
        let declaration = declaration();
        let resolved = resolve_record(
            &policy,
            resolve_policy_source(&policy).unwrap(),
            Some(&declaration),
            &evidence(),
            "test",
            "LLVM version test",
            "0123456789abcdef",
            vec![],
            1,
        )
        .unwrap();

        assert_eq!(resolved.rust.abi_id, "i0001");
        assert_eq!(
            resolved.rust.canonical_path,
            "cuda_intrinsics::__cuda_oxide_intrinsic_abi_v1::i0001"
        );
        assert_eq!(
            resolved.rust.public_path,
            "cuda_intrinsics::sreg::thread_idx_x"
        );
        assert_eq!(
            resolved.rust.compatibility_paths,
            ["cuda_device::thread::threadIdx_x"]
        );
        assert_eq!(
            resolved.llvm.as_ref().unwrap().properties,
            [
                "IntrNoMem",
                "IntrSpeculatable",
                "NoUndef<ret>",
                "Range<ret,0,1024>"
            ]
        );
        assert!(resolved.llvm.as_ref().unwrap().result_facts.no_undef);
        assert_eq!(
            resolved.llvm.as_ref().unwrap().result_facts.range,
            Some(CatalogHalfOpenRange {
                lower: "0".into(),
                upper_exclusive: "1024".into(),
            })
        );
        assert_eq!(resolved.backend.version, "LLVM version test");
        assert_eq!(resolved.backend.sha256, "0123456789abcdef");
    }

    #[test]
    fn malformed_intrinsic_abi_ids_are_rejected() {
        for abi_id in ["thread_idx_x", "i1", "x0001", "i00a1"] {
            let mut record = policy();
            record.abi_id = abi_id.into();
            let error = validate_unique_overlay(&[record], 1).unwrap_err();
            assert!(error.to_string().contains("stable `iNNNN` form"));
        }
    }

    #[test]
    fn ptx_versions_are_parsed_once_and_serialize_compatibly() {
        for (text, encoded) in [("2.0", 20), ("6.5", 65), ("10.0", 100)] {
            let version = parse_ptx_version(text, "test").unwrap();
            assert_eq!(version.encoded(), encoded);
            assert_eq!(
                serde_json::to_string(&version).unwrap(),
                format!("\"{text}\"")
            );
            assert_eq!(
                serde_json::from_str::<PtxVersion>(&format!("\"{text}\"")).unwrap(),
                version
            );
        }
        for malformed in ["6", "6.05", " 6.5", "06.5", "6.5 "] {
            assert!(parse_ptx_version(malformed, "test").is_err(), "{malformed}");
        }
    }

    #[test]
    fn legacy_all_target_and_monotonic_minimum_sm_are_typed() {
        let all = policy();
        assert_eq!(
            parse_hardware_target(&all).unwrap(),
            CatalogHardwareTarget::All
        );

        let mut minimum = policy();
        minimum.minimum_sm = Some("sm_75".into());
        assert_eq!(
            parse_hardware_target(&minimum).unwrap(),
            CatalogHardwareTarget::AnyOf {
                alternatives: vec![CatalogHardwareAlternative::MinimumSm { sm: 75 }],
            }
        );

        minimum.minimum_sm = Some("sm_90a".into());
        assert!(parse_hardware_target(&minimum).is_err());
        minimum.minimum_sm = None;
        minimum.targets = "sm_75+".into();
        assert!(parse_hardware_target(&minimum).is_err());
    }

    #[test]
    fn abi_ledger_requires_exact_active_identity() {
        let record = policy();
        let frozen_entry = ledger_entry(&record);
        validate_abi_ledger(
            &overlay_file(vec![record.clone()]),
            &ledger(vec![frozen_entry.clone()]),
        )
        .unwrap();

        let mut reassigned = record.clone();
        reassigned.id = "different_catalog_id".into();
        let error = validate_abi_ledger(
            &overlay_file(vec![reassigned]),
            &ledger(vec![frozen_entry.clone()]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("catalog ID mismatch"));

        let mut reassigned = record.clone();
        reassigned.operation_key = "launch.block_index.x".into();
        let error = validate_abi_ledger(
            &overlay_file(vec![reassigned]),
            &ledger(vec![frozen_entry.clone()]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("operation key mismatch"));

        for mutate in [
            |record: &mut OverlayIntrinsic| record.safe = false,
            |record: &mut OverlayIntrinsic| record.rust_arguments.push("u32".into()),
            |record: &mut OverlayIntrinsic| record.rust_result = "u64".into(),
        ] {
            let mut changed_signature = record.clone();
            mutate(&mut changed_signature);
            let error = validate_abi_ledger(
                &overlay_file(vec![changed_signature]),
                &ledger(vec![frozen_entry.clone()]),
            )
            .unwrap_err();
            assert!(error.to_string().contains("raw Rust signature mismatch"));
        }
    }

    #[test]
    fn abi_ledger_does_not_freeze_public_or_backend_implementation_details() {
        let record = policy();
        let frozen_entry = ledger_entry(&record);
        let mut evolved = record.clone();
        evolved.rust_module = "coordinates".into();
        evolved.rust_name = "thread_x".into();
        evolved.public_rust_path = "cuda_intrinsics::coordinates::thread_x".into();
        evolved.llvm_symbol = Some("llvm.nvvm.backend.v2.tid.x".into());
        evolved.llvm_arguments = vec!["i8".into()];
        evolved.llvm_results = vec!["i64".into()];
        evolved.dialect_op_type = "ReadThreadIndexXOpV2".into();
        evolved.dialect_op_name = "nvvm.read_thread_index_x_v2".into();
        evolved.lowering = "backend_v2_adapter".into();

        validate_abi_ledger(&overlay_file(vec![evolved]), &ledger(vec![frozen_entry])).unwrap();
    }

    #[test]
    fn tombstoned_or_unlisted_abi_ids_cannot_reappear() {
        let record = policy();
        let mut tombstone = ledger_entry(&record);
        tombstone.status = "tombstone".into();
        let error = validate_abi_ledger(
            &overlay_file(vec![record.clone()]),
            &ledger(vec![tombstone]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("cannot reappear"));

        let error = validate_abi_ledger(&overlay_file(vec![record]), &ledger(vec![])).unwrap_err();
        assert!(error.to_string().contains("contains no entries"));
    }

    #[test]
    fn every_active_ledger_entry_requires_an_overlay_record() {
        let record = policy();
        let error =
            validate_abi_ledger(&overlay_file(vec![]), &ledger(vec![ledger_entry(&record)]))
                .unwrap_err();
        assert!(error.to_string().contains("has no overlay record"));
    }

    #[test]
    fn return_range_properties_are_half_open_and_unique() {
        let facts =
            imported_result_facts(&["NoUndef<ret>".into(), "Range<ret,1,1025>".into()]).unwrap();
        assert!(facts.no_undef);
        let range = facts.range.unwrap();
        assert_eq!(range.lower, "1");
        assert_eq!(range.upper_exclusive, "1025");

        let duplicate =
            imported_result_facts(&["Range<ret,0,32>".into(), "Range<ret,0,64>".into()])
                .unwrap_err();
        assert!(duplicate.to_string().contains("duplicate return range"));
    }
}
