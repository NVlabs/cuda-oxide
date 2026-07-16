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
    Clc, ClcAdapter, ClcAdmission, ClcOperation, ClusterBarrier, ClusterBarrierAdmission,
    ClusterBarrierMode, ClusterBarrierOrdering, ClusterMemory, ClusterMemoryAdapter,
    ClusterMemoryAdmission, ClusterMemoryOperation, ClusterMemorySourceContract,
    ClusterSregAdmission, CpAsyncAdapter, CpAsyncCachePolicy, CpAsyncControlAdapter,
    CpAsyncControlOperation, CpAsyncCopySize, CpAsyncMbarrierAdapter, CpAsyncMbarrierOperation,
    CpAsyncMbarrierStateSpace, CpAsyncSourceSize, DebugControl, DebugControlAdapter,
    DebugControlAdmission, DebugControlOperation, DotProductAdapter, DotProductOperation,
    DotProductSignedness, EvidenceArtifactKind, EvidenceFile, EvidenceFileV6, EvidenceMatrix,
    EvidenceMatrixTemplate, EvidenceRecord, EvidenceRecordDefaults, EvidenceStage,
    EvidenceStageKind, ImportedAddressSpace, ImportedFile, ImportedIntrinsic, IntrinsicBackend,
    IntrinsicSource, LdmatrixAdapter, LdmatrixAddressContract, LdmatrixElement, LdmatrixLayout,
    LdmatrixMemoryOrder, LdmatrixMultiplicity, LdmatrixParticipation, LdmatrixShape,
    LdmatrixStateSpace, MaskEncoding, MatchOperandEncoding, MbarrierBasicAdapter,
    MbarrierBasicOperation, MbarrierExtended, MbarrierExtendedAdapter, MbarrierExtendedAdmission,
    MbarrierExtendedOperation, MbarrierExtendedSourceContract, MbarrierStateSpace,
    MovmatrixAdapter, MovmatrixParticipation, OverlayBackendLowering, OverlayFile,
    OverlayIntrinsic, OverlayShardFile, PackedAluAdapter, PackedAluFormat, PackedAluOperation,
    PackedAtomicAccessContract, PackedAtomicAdapter, PackedAtomicAtomicity,
    PackedAtomicCodegenContract, PackedAtomicFormat, PackedAtomicOperation, PackedAtomicOrdering,
    PackedAtomicPointerContract, PackedAtomicReturnContract, PackedAtomicRounding,
    PackedAtomicScope, PackedAtomicScopeContract, PackedAtomicStateSpace, PackedAtomicSubnormal,
    PackedConversionAdapter, PackedConversionDestinationFormat, PackedConversionFp8Admission,
    PackedConversionRounding, PackedConversionSaturation, PackedConversionSourceFormat,
    PreSm70MemberMaskRule, Prmt, PrmtAdapter, PrmtAdmission, PrmtMode, PtxVersion, ReduxAdapter,
    ReduxOperation, ReduxParticipation, RegisterMma, RegisterMmaAccumulator, RegisterMmaAdapter,
    RegisterMmaBinaryAdmission, RegisterMmaCompatibilitySource, RegisterMmaElement,
    RegisterMmaIntegerAdmission, RegisterMmaLayout, RegisterMmaOperation, RegisterMmaOverflow,
    RegisterMmaParticipation, RegisterMmaShape, RuntimeValidation, SparseMma, SparseMmaAccumulator,
    SparseMmaAdapter, SparseMmaCompatibilitySource, SparseMmaElement, SparseMmaF8F6F4Admission,
    SparseMmaIntegerAdmission, SparseMmaLayout, SparseMmaLlvmAdapter, SparseMmaMetadata,
    SparseMmaOverflow, SparseMmaParticipation, SparseMmaSelector, SparseMmaShape, SpecialRegister,
    SpecialRegisterAdmission, SpecialRegisterKind, SpecialRegisterLlvmExclusion,
    SpecialRegisterLlvmExclusionReason, SpecialRegisterObservation,
    SpecialRegisterOutputConstraint, SpecialRegisterPtxType, SpecialRegisterWidth,
    StmatrixAdmission, StmatrixLayout, StmatrixMultiplicity, ThreadfenceAdmission,
    ThreadfenceScope, VoteAdapter, VoteMode, VoteParticipation, WarpBarrierAdapter,
    WarpBarrierMaskEncoding, WarpBarrierMemoryOrdering, WarpBarrierParticipation, WarpMatchAdapter,
    WarpMatchMode, WarpMatchParticipation, WarpMatchValueWidth, WarpShuffleAdapter,
    WarpShuffleMode, WarpShuffleOperandEncoding, WarpShuffleParticipation, WarpShuffleSourceLane,
    WarpShuffleValueKind,
};
use crate::ptx::{InstructionPattern, OperandPattern};
use crate::util::{read_json, sha256_bytes, sha256_file};
use anyhow::{Context, Result, bail, ensure};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

const OVERLAY_SCHEMA: u32 = 39;
const MINIMUM_OVERLAY_SHARD_SCHEMA: u32 = 26;
const OVERLAY_SHARD_SCHEMA: u32 = 40;
const SPARSE_MMA_F8F6F4_SHARD_SCHEMA: u32 = 27;
const PRMT_SHARD_SCHEMA: u32 = 28;
const PACKED_CONVERSION_FP8_SHARD_SCHEMA: u32 = 29;
const CLUSTER_SREG_SHARD_SCHEMA: u32 = 30;
const CLUSTER_BARRIER_SHARD_SCHEMA: u32 = 31;
const SPECIAL_REGISTER_SHARD_SCHEMA: u32 = 32;
const DEBUG_CONTROL_SHARD_SCHEMA: u32 = 33;
const THREADFENCE_SHARD_SCHEMA: u32 = 34;
const STMATRIX_SHARD_SCHEMA: u32 = 35;
const CLUSTER_MEMORY_SHARD_SCHEMA: u32 = 39;
const CLC_SHARD_SCHEMA: u32 = 40;
const MBARRIER_EXTENDED_SHARD_SCHEMA: u32 = 40;
pub(crate) const CATALOG_SCHEMA: u32 = 38;

struct ResolutionBase {
    overlay: OverlayFile,
    imported: ImportedFile,
    source: CatalogSource,
    imported_sha256: String,
    overlay_sha256: String,
    abi_ledger_sha256: String,
}

#[derive(Debug)]
pub(crate) struct CandidateResolution {
    pub catalog: CatalogFile,
    pub mechanism: BackendLoweringMechanism,
    pub requirement: CatalogTargetRequirement,
}

pub fn resolve(repo_root: &Path) -> Result<CatalogFile> {
    let base = load_resolution_base(repo_root)?;
    let ResolutionBase {
        overlay,
        imported,
        source,
        imported_sha256,
        overlay_sha256,
        abi_ledger_sha256,
    } = base;
    let imported_by_record = index_imported_intrinsics(&imported)?;
    let llvm_revision = source.llvm_revision.clone();
    let (evidence_files, evidence_hashes) = read_evidence(repo_root)?;
    let evidence_by_profile_id = index_evidence(&evidence_files, &llvm_revision)?;

    let mut intrinsics = Vec::with_capacity(overlay.intrinsics.len());
    for policy in &overlay.intrinsics {
        let source = resolve_policy_source(policy)?;
        let declaration = resolve_imported_declaration(policy, &source, &imported_by_record)?;
        validate_special_register_llvm_exclusion(policy, &imported_by_record)?;
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
        source,
        inputs: CatalogInputs {
            imported_sha256,
            overlay_sha256,
            abi_ledger_sha256,
            evidence_sha256: evidence_hashes,
        },
        intrinsics,
    })
}

#[cfg(test)]
pub(crate) fn test_catalog_with_clc(repo_root: &Path) -> Result<CatalogFile> {
    let mut catalog = resolve(repo_root)?;
    let imported: ImportedFile = read_json(&repo_root.join("intrinsics/imported.json"))?;
    let imported_by_record = index_imported_intrinsics(&imported)?;
    let operations = [
        ClcOperation::TryCancel,
        ClcOperation::TryCancelMulticast,
        ClcOperation::QueryIsCanceled,
        ClcOperation::QueryGetFirstCtaidX,
        ClcOperation::QueryGetFirstCtaidY,
        ClcOperation::QueryGetFirstCtaidZ,
    ];
    let admission = ClcAdmission {
        llvm_evidence_profile: "llvm-clc-test".into(),
        libnvvm_evidence_profile: "libnvvm-clc-test".into(),
        runtime_validation: RuntimeValidation::Unexecuted,
        variants: operations
            .into_iter()
            .map(|operation| crate::model::ClcAdmissionVariant {
                abi_id: clc_recipe(operation).abi_id.into(),
                operation,
            })
            .collect(),
    };
    for policy in expand_clc_admission(&admission)? {
        let source = resolve_policy_source(&policy)?;
        let declaration = resolve_imported_declaration(&policy, &source, &imported_by_record)?;
        validate_policy(&policy, &source, declaration, catalog.intrinsic_abi)?;
        let backend_lowerings = policy
            .backend_lowerings
            .iter()
            .map(|lowering| {
                Ok(CatalogBackendLowering {
                    backend: lowering.backend,
                    mechanism: lowering.mechanism,
                    evidence_profile: lowering.evidence_profile.clone(),
                    target: backend_target_requirement(&policy, lowering)?,
                    version: "test".into(),
                    sha256: "0".repeat(64),
                    artifact_path: None,
                    build_id_prefix: None,
                    status: "validated".into(),
                    stages: vec![],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        catalog.intrinsics.push(materialize_record(
            &policy,
            source,
            declaration,
            CatalogBackend {
                profile: "clc-test".into(),
                version: "test".into(),
                sha256: "0".repeat(64),
                status: "validated".into(),
                target_triple: "nvptx64-nvidia-cuda".into(),
                gpu_target: "sm_100".into(),
                ptx_feature: "+ptx86".into(),
            },
            backend_lowerings,
            catalog.intrinsic_abi,
        )?);
    }
    catalog
        .intrinsics
        .sort_by(|left, right| left.id.cmp(&right.id));
    Ok(catalog)
}

fn load_resolution_base(repo_root: &Path) -> Result<ResolutionBase> {
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

    bind_generated_abi_ids(&mut overlay, &ledger)?;
    overlay
        .intrinsics
        .sort_by(|left, right| left.id.cmp(&right.id));
    validate_unique_overlay(&overlay.intrinsics, overlay.intrinsic_abi)?;
    validate_abi_ledger(&overlay, &ledger)?;
    Ok(ResolutionBase {
        overlay,
        imported,
        source: CatalogSource {
            llvm_repository: lock.llvm.repository,
            llvm_revision: lock.llvm.revision,
            llvm_tblgen_version: lock.llvm_tblgen.version_line,
            llvm_tblgen_source_revision: lock
                .llvm_tblgen
                .built_from_llvm_revision
                .context("pinned llvm-tblgen has no source revision")?,
        },
        imported_sha256,
        overlay_sha256,
        abi_ledger_sha256: sha256_file(&ledger_path)?,
    })
}

fn index_imported_intrinsics(
    imported: &ImportedFile,
) -> Result<BTreeMap<&str, &ImportedIntrinsic>> {
    let imported_by_record: BTreeMap<_, _> = imported
        .intrinsics
        .iter()
        .map(|intrinsic| (intrinsic.source_record.as_str(), intrinsic))
        .collect();
    ensure!(
        imported_by_record.len() == imported.intrinsics.len(),
        "imported.json contains duplicate source records"
    );
    Ok(imported_by_record)
}

fn resolve_imported_declaration<'a>(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
    imported_by_record: &'a BTreeMap<&str, &'a ImportedIntrinsic>,
) -> Result<Option<&'a ImportedIntrinsic>> {
    match source {
        IntrinsicSource::LlvmImported { source_record } => Ok(Some(
            *imported_by_record
                .get(source_record.as_str())
                .with_context(|| {
                    format!(
                        "overlay intrinsic {} references missing imported record {}",
                        policy.id, source_record
                    )
                })?,
        )),
        IntrinsicSource::PtxNative { .. } => Ok(None),
    }
}

pub(crate) fn resolve_candidate(
    repo_root: &Path,
    intrinsic_id: &str,
    backend_version: &str,
    backend_sha256: &str,
    gpu_target: &str,
    ptx_feature: &str,
) -> Result<CandidateResolution> {
    let base = load_resolution_base(repo_root)?;
    let imported_by_record = index_imported_intrinsics(&base.imported)?;
    let policy = base
        .overlay
        .intrinsics
        .iter()
        .find(|policy| policy.id == intrinsic_id)
        .with_context(|| format!("unknown overlay intrinsic {intrinsic_id}"))?;
    let source = resolve_policy_source(policy)?;
    let declaration = resolve_imported_declaration(policy, &source, &imported_by_record)?;
    validate_special_register_llvm_exclusion(policy, &imported_by_record)?;
    validate_policy(policy, &source, declaration, base.overlay.intrinsic_abi)?;
    ensure!(
        !backend_version.trim().is_empty()
            && backend_sha256.len() == 64
            && backend_sha256.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "candidate backend identity is incomplete"
    );
    let (mechanism, target) = candidate_llvm_route(policy)?;
    validate_candidate_target(&target, gpu_target, ptx_feature, intrinsic_id)?;

    let record = materialize_record(
        policy,
        source,
        declaration,
        CatalogBackend {
            profile: "candidate-comparison".into(),
            version: backend_version.into(),
            sha256: backend_sha256.into(),
            status: "candidate".into(),
            target_triple: "nvptx64-nvidia-cuda".into(),
            gpu_target: gpu_target.into(),
            ptx_feature: ptx_feature.into(),
        },
        Vec::new(),
        base.overlay.intrinsic_abi,
    )?;
    Ok(CandidateResolution {
        catalog: CatalogFile {
            schema: CATALOG_SCHEMA,
            catalog_version: base.overlay.catalog_version,
            intrinsic_abi: base.overlay.intrinsic_abi,
            generator_version: env!("CARGO_PKG_VERSION").to_owned(),
            source: base.source,
            inputs: CatalogInputs {
                imported_sha256: base.imported_sha256,
                overlay_sha256: base.overlay_sha256,
                abi_ledger_sha256: base.abi_ledger_sha256,
                evidence_sha256: Vec::new(),
            },
            intrinsics: vec![record],
        },
        mechanism,
        requirement: target,
    })
}

fn candidate_llvm_route(
    policy: &OverlayIntrinsic,
) -> Result<(BackendLoweringMechanism, CatalogTargetRequirement)> {
    let routes = policy
        .backend_lowerings
        .iter()
        .filter(|lowering| lowering.backend == IntrinsicBackend::LlvmNvptx)
        .collect::<Vec<_>>();
    ensure!(
        routes.len() <= 1,
        "{} has more than one LLVM-NVPTX route",
        policy.id
    );
    if let Some(route) = routes.first() {
        return Ok((route.mechanism, backend_target_requirement(policy, route)?));
    }
    ensure!(
        matches!(
            resolve_policy_source(policy)?,
            IntrinsicSource::LlvmImported { .. }
        ),
        "{} has no LLVM-NVPTX candidate route",
        policy.id
    );
    Ok((
        BackendLoweringMechanism::TypedNvvm,
        CatalogTargetRequirement {
            minimum_ptx: parse_ptx_version(&policy.minimum_ptx, &policy.id)?,
            hardware: parse_hardware_target(policy)?,
        },
    ))
}

fn validate_candidate_target(
    requirement: &CatalogTargetRequirement,
    gpu_target: &str,
    ptx_feature: &str,
    intrinsic_id: &str,
) -> Result<()> {
    ensure!(
        gpu_target.starts_with("sm_"),
        "candidate GPU target {gpu_target:?} must use sm_NN, sm_NNa, or sm_NNf"
    );
    let hardware = parse_stage_hardware(gpu_target).with_context(|| {
        format!("candidate GPU target {gpu_target:?} must use sm_NN, sm_NNa, or sm_NNf")
    })?;
    ensure!(
        describe_stage_hardware(hardware) == gpu_target,
        "candidate GPU target {gpu_target:?} is not canonical"
    );
    let ptx = parse_candidate_ptx_feature(ptx_feature)?;
    ensure!(
        ptx >= requirement.minimum_ptx,
        "candidate target {gpu_target} / {ptx_feature} is below {intrinsic_id} PTX floor {}",
        requirement.minimum_ptx
    );
    let hardware_matches = match &requirement.hardware {
        CatalogHardwareTarget::All => true,
        CatalogHardwareTarget::AnyOf { alternatives } => alternatives
            .iter()
            .any(|expected| selected_stage_hardware_matches(hardware, *expected, true)),
    };
    ensure!(
        hardware_matches,
        "candidate GPU target {gpu_target} does not satisfy {intrinsic_id} hardware requirement {:?}",
        requirement.hardware
    );
    Ok(())
}

fn parse_candidate_ptx_feature(value: &str) -> Result<PtxVersion> {
    let digits = value
        .strip_prefix("+ptx")
        .with_context(|| format!("candidate PTX feature {value:?} must use +ptxNN"))?;
    ensure!(
        matches!(digits.len(), 2 | 3) && digits.bytes().all(|byte| byte.is_ascii_digit()),
        "candidate PTX feature {value:?} must use +ptxNN"
    );
    let split = digits.len() - 1;
    let version = format!("{}.{}", &digits[..split], &digits[split..]);
    version
        .parse()
        .map_err(|reason: String| anyhow::anyhow!("candidate PTX feature {value:?}: {reason}"))
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
        let mut shard: OverlayShardFile =
            toml::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        validate_overlay_shard_schema(&shard, &path)?;
        let int4_mma_admission = shard.register_mma_int4.take();
        let int8_mma_admission = shard.register_mma_int8.take();
        let binary_mma_admission = shard.register_mma_b1.take();
        let sparse_mma_admission = shard.sparse_mma_integer.take();
        let sparse_mma_f8f6f4_admission = shard.sparse_mma_f8f6f4_f32.take();
        let prmt_admission = shard.prmt.take();
        let packed_conversion_fp8_admission = shard.packed_conversion_fp8.take();
        let cluster_sreg_admission = shard.cluster_sreg.take();
        let cluster_barrier_admission = shard.cluster_barrier.take();
        let mbarrier_extended_admission = shard.mbarrier_extended.take();
        let special_register_admission = shard.special_registers.take();
        let debug_control_admission = shard.debug_control.take();
        let threadfence_admission = shard.threadfence.take();
        let cluster_memory_admission = shard.cluster_memory.take();
        let stmatrix_admission = shard.stmatrix.take();
        let clc_admission = shard.clc.take();
        let compact_mma_count = usize::from(int4_mma_admission.is_some())
            + usize::from(int8_mma_admission.is_some())
            + usize::from(binary_mma_admission.is_some())
            + usize::from(sparse_mma_admission.is_some())
            + usize::from(sparse_mma_f8f6f4_admission.is_some());
        ensure!(
            compact_mma_count <= 1,
            "overlay shard {} contains more than one compact MMA admission",
            path.display()
        );
        let integer_mma_admission = int4_mma_admission
            .map(|admission| (RegisterMmaIntegerKind::Int4, admission))
            .or_else(|| {
                int8_mma_admission.map(|admission| (RegisterMmaIntegerKind::Int8, admission))
            });
        if let Some((kind, admission)) = integer_mma_admission {
            ensure!(
                shard.family == "register_mma" && shard.intrinsics.is_empty(),
                "compact integer MMA admission must be the only content of a register_mma shard"
            );
            shard.intrinsics = expand_register_mma_integer_admission(kind, &admission)?;
        }
        if let Some(admission) = binary_mma_admission {
            ensure!(
                shard.family == "register_mma" && shard.intrinsics.is_empty(),
                "compact binary MMA admission must be the only content of a register_mma shard"
            );
            shard.intrinsics = expand_register_mma_binary_admission(&admission)?;
        }
        if let Some(admission) = sparse_mma_admission {
            ensure!(
                shard.family == "sparse_mma" && shard.intrinsics.is_empty(),
                "compact sparse MMA admission must be the only content of a sparse_mma shard"
            );
            shard.intrinsics = expand_sparse_mma_integer_admission(&admission)?;
        }
        if let Some(admission) = sparse_mma_f8f6f4_admission {
            ensure!(
                shard.family == "sparse_mma" && shard.intrinsics.is_empty(),
                "compact sparse f8f6f4 MMA admission must be the only content of a sparse_mma shard"
            );
            shard.intrinsics = expand_sparse_mma_f8f6f4_admission(&admission)?;
        }
        if let Some(admission) = prmt_admission {
            ensure!(
                shard.family == "prmt" && shard.intrinsics.is_empty(),
                "compact prmt admission must be the only content of a prmt shard"
            );
            shard.intrinsics = expand_prmt_admission(&admission)?;
        }
        if let Some(admission) = packed_conversion_fp8_admission {
            ensure!(
                shard.family == "packed_conversion" && shard.intrinsics.is_empty(),
                "compact FP8 conversion admission must be the only content of a packed_conversion shard"
            );
            shard.intrinsics = expand_packed_conversion_fp8_admission(&admission)?;
        }
        if let Some(admission) = cluster_sreg_admission {
            ensure!(
                shard.family == "sreg" && shard.intrinsics.is_empty() && compact_mma_count == 0,
                "compact cluster-sreg admission must be the only content of an sreg shard"
            );
            shard.intrinsics = expand_cluster_sreg_admission(&admission)?;
        }
        if let Some(admission) = cluster_barrier_admission {
            ensure!(
                shard.family == "cluster_barrier" && shard.intrinsics.is_empty(),
                "compact cluster-barrier admission must be the only content of its shard"
            );
            shard.intrinsics = expand_cluster_barrier_admission(&admission)?;
        }
        if let Some(admission) = mbarrier_extended_admission {
            ensure!(
                shard.family == "mbarrier_extended" && shard.intrinsics.is_empty(),
                "compact extended-mbarrier admission must be the only content of its shard"
            );
            shard.intrinsics = expand_mbarrier_extended_admission(&admission)?;
        }
        if let Some(admission) = special_register_admission {
            ensure!(
                shard.family == "sreg" && shard.intrinsics.is_empty(),
                "compact special-register admission must be the only content of an sreg shard"
            );
            shard.intrinsics = expand_special_register_admission(&admission)?;
        }
        if let Some(admission) = debug_control_admission {
            ensure!(
                shard.family == "debug_control" && shard.intrinsics.is_empty(),
                "compact debug-control admission must be the only content of a debug_control shard"
            );
            shard.intrinsics = expand_debug_control_admission(&admission)?;
        }
        if let Some(admission) = threadfence_admission {
            ensure!(
                shard.family == "sync" && shard.intrinsics.is_empty(),
                "compact threadfence admission must be the only content of a sync shard"
            );
            shard.intrinsics = expand_threadfence_admission(&admission)?;
        }
        if let Some(admission) = cluster_memory_admission {
            ensure!(
                shard.family == "cluster_memory" && shard.intrinsics.is_empty(),
                "compact cluster-memory admission must be the only content of its shard"
            );
            shard.intrinsics = expand_cluster_memory_admission(&admission)?;
        }
        if let Some(admission) = stmatrix_admission {
            ensure!(
                shard.family == "stmatrix" && shard.intrinsics.is_empty(),
                "compact stmatrix admission must be the only content of its shard"
            );
            shard.intrinsics = expand_stmatrix_admission(&admission)?;
        }
        if let Some(admission) = clc_admission {
            ensure!(
                shard.family == "clc" && shard.intrinsics.is_empty(),
                "compact CLC admission must be the only content of a clc shard"
            );
            shard.intrinsics = expand_clc_admission(&admission)?;
        }
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

fn validate_overlay_shard_schema(shard: &OverlayShardFile, path: &Path) -> Result<()> {
    validate_overlay_shard_schema_with_max(shard, path, OVERLAY_SHARD_SCHEMA)
}

fn validate_overlay_shard_schema_with_max(
    shard: &OverlayShardFile,
    path: &Path,
    maximum_schema: u32,
) -> Result<()> {
    ensure!(
        (MINIMUM_OVERLAY_SHARD_SCHEMA..=maximum_schema).contains(&shard.schema),
        "unsupported overlay shard schema {} in {}",
        shard.schema,
        path.display()
    );
    ensure!(
        shard.sparse_mma_f8f6f4_f32.is_none() || shard.schema >= SPARSE_MMA_F8F6F4_SHARD_SCHEMA,
        "compact sparse f8f6f4 MMA admission requires overlay shard schema {}",
        SPARSE_MMA_F8F6F4_SHARD_SCHEMA
    );
    ensure!(
        shard.prmt.is_none() || shard.schema >= PRMT_SHARD_SCHEMA,
        "compact prmt admission requires overlay shard schema {}",
        PRMT_SHARD_SCHEMA
    );
    ensure!(
        shard.packed_conversion_fp8.is_none() || shard.schema >= PACKED_CONVERSION_FP8_SHARD_SCHEMA,
        "compact FP8 conversion admission requires overlay shard schema {}",
        PACKED_CONVERSION_FP8_SHARD_SCHEMA
    );
    ensure!(
        shard.cluster_sreg.is_none() || shard.schema >= CLUSTER_SREG_SHARD_SCHEMA,
        "compact cluster-sreg admission requires overlay shard schema {}",
        CLUSTER_SREG_SHARD_SCHEMA
    );
    ensure!(
        shard.cluster_barrier.is_none() || shard.schema >= CLUSTER_BARRIER_SHARD_SCHEMA,
        "compact cluster-barrier admission requires overlay shard schema {}",
        CLUSTER_BARRIER_SHARD_SCHEMA
    );
    ensure!(
        shard.special_registers.is_none() || shard.schema >= SPECIAL_REGISTER_SHARD_SCHEMA,
        "compact special-register admission requires overlay shard schema {}",
        SPECIAL_REGISTER_SHARD_SCHEMA
    );
    ensure!(
        shard.debug_control.is_none() || shard.schema >= DEBUG_CONTROL_SHARD_SCHEMA,
        "compact debug-control admission requires overlay shard schema {}",
        DEBUG_CONTROL_SHARD_SCHEMA
    );
    ensure!(
        shard.threadfence.is_none() || shard.schema >= THREADFENCE_SHARD_SCHEMA,
        "compact threadfence admission requires overlay shard schema {}",
        THREADFENCE_SHARD_SCHEMA
    );
    ensure!(
        shard.cluster_memory.is_none() || shard.schema >= CLUSTER_MEMORY_SHARD_SCHEMA,
        "compact cluster-memory admission requires overlay shard schema {}",
        CLUSTER_MEMORY_SHARD_SCHEMA
    );
    ensure!(
        shard.stmatrix.is_none() || shard.schema >= STMATRIX_SHARD_SCHEMA,
        "compact stmatrix admission requires overlay shard schema {}",
        STMATRIX_SHARD_SCHEMA
    );
    ensure!(
        shard.clc.is_none() || shard.schema >= CLC_SHARD_SCHEMA,
        "compact CLC admission requires overlay shard schema {}",
        CLC_SHARD_SCHEMA
    );
    ensure!(
        shard.mbarrier_extended.is_none() || shard.schema >= MBARRIER_EXTENDED_SHARD_SCHEMA,
        "compact extended-mbarrier admission requires overlay shard schema {}",
        MBARRIER_EXTENDED_SHARD_SCHEMA
    );
    Ok(())
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
    let mut symbol_bases = BTreeMap::new();
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
            "{}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}",
            record.dialect_op_name,
            record.ldmatrix_variant,
            record.packed_atomic,
            record.redux,
            record.vote,
            record.active_mask,
            record.warp_match,
            record.warp_barrier,
            record.warp_shuffle,
            record.dot_product,
            record.cp_async_copy,
            record.cp_async_control,
            record.cp_async_mbarrier,
            record.mbarrier_basic,
            record.movmatrix,
            record.mbarrier_extended,
            record.register_mma,
            record.sparse_mma,
            record.prmt,
            record.cluster_barrier,
            record.cluster_memory,
        );
        insert_unique(&mut op_variants, &op_variant, "dialect op variant")?;
        if let Some(symbol) = &record.llvm_symbol {
            let is_resolved = record.resolved_llvm_symbol.is_some();
            if let Some(previous_was_resolved) = symbol_bases.insert(symbol, is_resolved) {
                ensure!(
                    previous_was_resolved && is_resolved,
                    "duplicate LLVM symbol {symbol} is reused without a resolved symbol"
                );
            }
            insert_unique(
                &mut symbols,
                record.resolved_llvm_symbol.as_ref().unwrap_or(symbol),
                "resolved LLVM symbol",
            )?;
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

struct AbiLedgerIndex<'a> {
    by_catalog_id: BTreeMap<&'a str, &'a AbiLedgerEntry>,
}

impl<'a> AbiLedgerIndex<'a> {
    fn new(ledger: &'a AbiLedgerFile) -> Result<Self> {
        let mut by_catalog_id = BTreeMap::new();
        for entry in &ledger.entries {
            ensure!(
                by_catalog_id
                    .insert(entry.catalog_id.as_str(), entry)
                    .is_none(),
                "duplicate ABI ledger catalog ID: {}",
                entry.catalog_id
            );
        }
        Ok(Self { by_catalog_id })
    }

    fn active(&self, catalog_id: &str) -> Result<&'a AbiLedgerEntry> {
        let entry = self
            .by_catalog_id
            .get(catalog_id)
            .copied()
            .with_context(|| format!("generated intrinsic {catalog_id} has no ABI ledger entry"))?;
        ensure!(
            entry.status == "active",
            "generated intrinsic {catalog_id} maps to non-active ABI ledger entry {}",
            entry.abi_id
        );
        Ok(entry)
    }
}

fn bind_generated_abi_ids(overlay: &mut OverlayFile, ledger: &AbiLedgerFile) -> Result<()> {
    let index = AbiLedgerIndex::new(ledger)?;
    for record in overlay
        .intrinsics
        .iter_mut()
        .filter(|record| record.abi_id.is_empty())
    {
        let entry = index.active(&record.id)?;
        ensure!(
            entry.operation_key == record.operation_key,
            "generated intrinsic {} operation key mismatch: ledger {:?}, derived {:?}",
            record.id,
            entry.operation_key,
            record.operation_key
        );
        let derived_signature = raw_rust_signature(record);
        ensure!(
            entry.raw_rust_signature == derived_signature,
            "generated intrinsic {} raw Rust signature mismatch: ledger {:?}, derived {:?}",
            record.id,
            entry.raw_rust_signature,
            derived_signature
        );
        record.abi_id.clone_from(&entry.abi_id);
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
        "sreg" => validate_sreg_policy(policy, source, declaration)?,
        "ldmatrix" => validate_ldmatrix_policy(
            policy,
            declaration.context("ldmatrix requires imported LLVM declaration")?,
        )?,
        "stmatrix" => validate_stmatrix_policy(
            policy,
            declaration.context("stmatrix requires imported LLVM declaration")?,
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
        "warp_barrier" => validate_warp_barrier_policy(
            policy,
            declaration.context("warp_barrier requires imported LLVM declaration")?,
        )?,
        "warp_shuffle" => validate_warp_shuffle_policy(policy, declaration)?,
        "packed_alu" => validate_packed_alu_policy(policy, source, declaration)?,
        "packed_conversion" => validate_packed_conversion_policy(policy, source, declaration)?,
        "cp_async_copy" => validate_cp_async_copy_policy(
            policy,
            declaration.context("cp_async_copy requires imported LLVM declaration")?,
        )?,
        "cp_async_control" => validate_cp_async_control_policy(
            policy,
            declaration.context("cp_async_control requires imported LLVM declaration")?,
        )?,
        "cp_async_mbarrier" => validate_cp_async_mbarrier_policy(
            policy,
            declaration.context("cp_async_mbarrier requires imported LLVM declaration")?,
        )?,
        "mbarrier_basic" => validate_mbarrier_basic_policy(
            policy,
            declaration.context("mbarrier_basic requires imported LLVM declaration")?,
        )?,
        "movmatrix" => validate_movmatrix_policy(policy, source)?,
        "mbarrier_extended" => validate_mbarrier_extended_policy(policy, source, declaration)?,
        "register_mma" => validate_register_mma_policy(
            policy,
            declaration.context("register_mma requires imported LLVM declaration")?,
        )?,
        "sparse_mma" => validate_sparse_mma_policy(
            policy,
            declaration.context("sparse_mma requires imported LLVM declaration")?,
        )?,
        "prmt" => validate_prmt_policy(
            policy,
            declaration.context("prmt requires imported LLVM declaration")?,
        )?,
        "cluster_barrier" => validate_cluster_barrier_policy(
            policy,
            declaration.context("cluster_barrier requires imported LLVM declaration")?,
        )?,
        "debug_control" => validate_debug_control_policy(policy, source)?,
        "cluster_memory" => validate_cluster_memory_policy(policy, source, declaration)?,
        "clc" => validate_clc_policy(
            policy,
            declaration.context("clc requires imported LLVM declaration")?,
        )?,
        family => bail!("{} uses unsupported generated family {family:?}", policy.id),
    }
    ensure!(
        (policy.family == "movmatrix") == policy.movmatrix.is_some(),
        "{} mixes the movmatrix contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "mbarrier_extended") == policy.mbarrier_extended.is_some(),
        "{} mixes the extended-mbarrier contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "register_mma") == policy.register_mma.is_some(),
        "{} mixes the register-MMA contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "sparse_mma") == policy.sparse_mma.is_some(),
        "{} mixes the sparse-MMA contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "prmt") == policy.prmt.is_some(),
        "{} mixes the prmt contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "cluster_barrier") == policy.cluster_barrier.is_some(),
        "{} mixes the cluster-barrier contract with another generated family",
        policy.id
    );
    ensure!(
        policy.special_register.is_none() || policy.family == "sreg",
        "{} mixes the special-register contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "debug_control") == policy.debug_control.is_some(),
        "{} mixes the debug-control contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "cluster_memory") == policy.cluster_memory.is_some(),
        "{} mixes the cluster-memory contract with another generated family",
        policy.id
    );
    ensure!(
        (policy.family == "clc") == policy.clc.is_some(),
        "{} mixes the CLC contract with another generated family",
        policy.id
    );
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
                    .any(|class| class == "NVVMPureIntrinsic")
                    || (policy.family == "packed_alu"
                        && declaration
                            .properties
                            .iter()
                            .any(|property| property == "IntrNoMem")
                        && declaration
                            .properties
                            .iter()
                            .any(|property| property == "IntrSpeculatable")),
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
        let convergence_supplied_by_ptx =
            (matches!(policy.family.as_str(), "register_mma" | "sparse_mma")
                && (policy.register_mma.is_some() || policy.sparse_mma.is_some())
                || (policy.family == "cluster_memory"
                    && policy.cluster_memory.is_some()
                    && policy.backend_lowerings.iter().all(|lowering| {
                        lowering.mechanism == BackendLoweringMechanism::InlinePtx
                    }))
                || (policy.family == "mbarrier_extended"
                    && policy.mbarrier_extended.is_some()
                    && policy.backend_lowerings.iter().all(|lowering| {
                        lowering.mechanism == BackendLoweringMechanism::InlinePtx
                    })))
                && policy.convergent
                && !imported_convergent;
        ensure!(
            imported_convergent == policy.convergent || convergence_supplied_by_ptx,
            "{} convergence mismatch: imported {}, overlay {}",
            policy.id,
            imported_convergent,
            policy.convergent
        );
        let selectionless_closed_family = (policy.family == "packed_conversion"
            && policy.packed_conversion.is_some())
            || (policy.family == "register_mma" && policy.register_mma.is_some())
            || (policy.family == "sparse_mma" && policy.sparse_mma.is_some())
            || (policy.family == "prmt" && policy.prmt.is_some())
            || (policy.family == "clc"
                && policy.clc.as_ref().is_some_and(|clc| {
                    matches!(
                        clc.operation,
                        ClcOperation::QueryIsCanceled
                            | ClcOperation::QueryGetFirstCtaidX
                            | ClcOperation::QueryGetFirstCtaidY
                            | ClcOperation::QueryGetFirstCtaidZ
                    )
                }))
            || (policy.family == "sreg"
                && policy.special_register.as_ref().is_some_and(|special| {
                    matches!(
                        special.register,
                        SpecialRegisterKind::Envreg1 | SpecialRegisterKind::Envreg2
                    )
                }))
            || policy.family == "stmatrix";
        ensure!(
            !declaration.selections.is_empty() || selectionless_closed_family,
            "{} has a declaration but no NVPTX TableGen selection record",
            policy.id
        );
        let matching_selections: Vec<_> = declaration
            .selections
            .iter()
            .filter(|selection| selection_matches_policy(policy, selection))
            .collect();
        let expected_selection_count = match policy.family.as_str() {
            "vote" | "warp_barrier" => 2,
            "warp_match" => 4,
            "warp_shuffle" => 8,
            "packed_conversion" | "register_mma" | "sparse_mma" | "prmt" | "stmatrix" => 0,
            "clc"
                if policy.clc.as_ref().is_some_and(|clc| {
                    matches!(
                        clc.operation,
                        ClcOperation::QueryIsCanceled
                            | ClcOperation::QueryGetFirstCtaidX
                            | ClcOperation::QueryGetFirstCtaidY
                            | ClcOperation::QueryGetFirstCtaidZ
                    )
                }) =>
            {
                0
            }
            "sreg"
                if policy.special_register.as_ref().is_some_and(|special| {
                    matches!(
                        special.register,
                        SpecialRegisterKind::Envreg1 | SpecialRegisterKind::Envreg2
                    )
                }) =>
            {
                0
            }
            "mbarrier_extended"
                if policy.mbarrier_extended.as_ref().is_some_and(|mbarrier| {
                    matches!(
                        mbarrier.operation,
                        MbarrierExtendedOperation::ArriveExpectTxCta
                            | MbarrierExtendedOperation::ArriveExpectTxCluster
                            | MbarrierExtendedOperation::TryWaitParityCta
                            | MbarrierExtendedOperation::TryWaitParityCluster
                    )
                }) =>
            {
                0
            }
            "mbarrier_extended" if policy.id == "nanosleep" => 2,
            "cp_async_copy"
                if policy
                    .cp_async_copy
                    .as_ref()
                    .is_some_and(|copy| copy.source_size == CpAsyncSourceSize::Runtime) =>
            {
                2
            }
            "cluster_memory" if policy.id == "map_shared_rank" => 2,
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
        if policy.id == "sync_threads" {
            return selection.source_record == "BARRIER_CTA_SYNC_ALIGNED_ALL_i"
                && selection.asm == "bar.sync \t$i;"
                && selection.predicates.is_empty()
                && selection.constraints.is_empty();
        }
        let Some(scope) = threadfence_scope_for_id(&policy.id) else {
            return false;
        };
        let recipe = threadfence_recipe(scope);
        return selection.source_record == recipe.selection_record
            && selection.asm == format!("membar.{};", recipe.ptx_level)
            && selection.predicates.is_empty()
            && selection.constraints.is_empty();
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

    if policy.family == "warp_barrier" {
        return policy.warp_barrier.is_some()
            && ["INT_BAR_WARP_SYNC_I", "INT_BAR_WARP_SYNC_R"]
                .contains(&selection.source_record.as_str())
            && policy.expected_ptx.matches(&selection.asm)
            && selection.constraints.is_empty();
    }

    if policy.family == "warp_shuffle" {
        let Some(shuffle) = &policy.warp_shuffle else {
            return false;
        };
        let recipe = warp_shuffle_recipe(shuffle.mode, shuffle.value_kind);
        return selection.asm
            == format!(
                "shfl.sync.{}.b32 \t$dst, $src, $offset, $mask, $threadmask;",
                recipe.ptx_mode
            )
            && selection.constraints.is_empty();
    }

    if policy.family == "cp_async_copy" {
        let Some(copy) = &policy.cp_async_copy else {
            return false;
        };
        let Some(recipe) = cp_async_copy_recipe(copy) else {
            return false;
        };
        return recipe
            .selections
            .contains(&selection.source_record.as_str())
            && policy.expected_ptx.matches(&selection.asm)
            && selection.constraints.is_empty();
    }

    if policy.family == "cp_async_control" {
        let Some(control) = &policy.cp_async_control else {
            return false;
        };
        let recipe = cp_async_control_recipe(control.operation);
        let instruction_matches = if control.operation == CpAsyncControlOperation::WaitGroup {
            selection.asm == "cp.async.wait_group \t$n;"
        } else {
            policy.expected_ptx.matches(&selection.asm)
        };
        return selection.source_record == recipe.selection
            && instruction_matches
            && selection.constraints.is_empty();
    }

    if policy.family == "cp_async_mbarrier" {
        let Some(bridge) = &policy.cp_async_mbarrier else {
            return false;
        };
        let recipe = cp_async_mbarrier_recipe(bridge.operation, bridge.state_space);
        return selection.source_record == recipe.selection
            && selection.asm == recipe.selection_asm
            && selection.constraints.is_empty();
    }

    if policy.family == "mbarrier_basic" {
        let Some(mbarrier) = &policy.mbarrier_basic else {
            return false;
        };
        let recipe = mbarrier_basic_recipe(mbarrier.operation);
        return selection.source_record == recipe.selection
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

#[derive(Clone, Copy)]
struct ThreadfenceRecipe {
    scope: ThreadfenceScope,
    abi_id: &'static str,
    id: &'static str,
    operation_key: &'static str,
    source_record: &'static str,
    selection_record: &'static str,
    llvm_symbol: &'static str,
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    ptx_level: &'static str,
    execution_scope: &'static str,
    minimum_ptx: &'static str,
    minimum_sm: Option<&'static str>,
    summary: &'static str,
}

fn threadfence_recipe(scope: ThreadfenceScope) -> ThreadfenceRecipe {
    match scope {
        ThreadfenceScope::Cta => ThreadfenceRecipe {
            scope,
            abi_id: "i0298",
            id: "threadfence_block",
            operation_key: "memory.fence.cta.sc",
            source_record: "int_nvvm_membar_cta",
            selection_record: "INT_MEMBAR_CTA",
            llvm_symbol: "llvm.nvvm.membar.cta",
            dialect_op_type: "ThreadfenceBlockOp",
            dialect_op_name: "nvvm.threadfence_block",
            ptx_level: "cta",
            execution_scope: "cta",
            minimum_ptx: "1.4",
            minimum_sm: None,
            summary: "Orders this thread's memory operations for observers in its CTA.",
        },
        ThreadfenceScope::Device => ThreadfenceRecipe {
            scope,
            abi_id: "i0299",
            id: "threadfence",
            operation_key: "memory.fence.device.sc",
            source_record: "int_nvvm_membar_gl",
            selection_record: "INT_MEMBAR_GL",
            llvm_symbol: "llvm.nvvm.membar.gl",
            dialect_op_type: "ThreadfenceOp",
            dialect_op_name: "nvvm.threadfence",
            ptx_level: "gl",
            execution_scope: "device",
            minimum_ptx: "1.4",
            minimum_sm: None,
            summary: "Orders this thread's memory operations for observers on its GPU.",
        },
        ThreadfenceScope::System => ThreadfenceRecipe {
            scope,
            abi_id: "i0300",
            id: "threadfence_system",
            operation_key: "memory.fence.system.sc",
            source_record: "int_nvvm_membar_sys",
            selection_record: "INT_MEMBAR_SYS",
            llvm_symbol: "llvm.nvvm.membar.sys",
            dialect_op_type: "ThreadfenceSystemOp",
            dialect_op_name: "nvvm.threadfence_system",
            ptx_level: "sys",
            execution_scope: "system",
            minimum_ptx: "2.0",
            minimum_sm: Some("sm_20"),
            summary: "Orders this thread's memory operations for system-wide observers.",
        },
    }
}

fn threadfence_scope_for_id(id: &str) -> Option<ThreadfenceScope> {
    match id {
        "threadfence_block" => Some(ThreadfenceScope::Cta),
        "threadfence" => Some(ThreadfenceScope::Device),
        "threadfence_system" => Some(ThreadfenceScope::System),
        _ => None,
    }
}

fn expand_threadfence_admission(admission: &ThreadfenceAdmission) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "threadfence runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "compact threadfence admission requires both backend evidence profiles"
    );
    let expected_scopes = [
        ThreadfenceScope::Cta,
        ThreadfenceScope::Device,
        ThreadfenceScope::System,
    ];
    let actual_scopes = admission
        .variants
        .iter()
        .map(|variant| variant.scope)
        .collect::<Vec<_>>();
    ensure!(
        actual_scopes == expected_scopes,
        "compact threadfence admission must contain each reviewed scope exactly once in canonical order"
    );

    admission
        .variants
        .iter()
        .map(|variant| {
            let recipe = threadfence_recipe(variant.scope);
            ensure!(
                variant.abi_id == recipe.abi_id,
                "{} must keep reserved ABI ID {}",
                recipe.id,
                recipe.abi_id
            );
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: variant.abi_id.clone(),
                operation_key: recipe.operation_key.into(),
                family: "sync".into(),
                source: None,
                source_record: Some(recipe.source_record.into()),
                rust_module: "fence".into(),
                rust_name: recipe.id.into(),
                rust_arguments: vec![],
                rust_result: "()".into(),
                safe: true,
                must_use: false,
                safe_allowlist_reason: Some(
                    "a fence only orders the calling thread's memory operations and has no caller preconditions"
                        .into(),
                ),
                public_rust_path: format!("cuda_intrinsics::fence::{}", recipe.id),
                compatibility_rust_paths: vec![
                    format!("cuda_device::fence::{}", recipe.id),
                    format!("cuda_device::{}", recipe.id),
                ],
                dialect_op_type: recipe.dialect_op_type.into(),
                dialect_op_name: recipe.dialect_op_name.into(),
                dialect_operands: vec![],
                dialect_results: vec![],
                llvm_symbol: Some(recipe.llvm_symbol.into()),
                resolved_llvm_symbol: None,
                llvm_arguments: vec![],
                llvm_results: vec![],
                pure: false,
                memory: "read_write".into(),
                convergent: false,
                execution_scope: recipe.execution_scope.into(),
                minimum_ptx: recipe.minimum_ptx.into(),
                minimum_sm: recipe.minimum_sm.map(Into::into),
                ptx_result: "()".into(),
                targets: "all".into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section:
                    "9.7.14.4 Parallel Synchronization and Communication Instructions: membar / fence"
                        .into(),
                ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar-fence".into(),
                lowering: "direct_nvvm".into(),
                backend_lowerings: vec![
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LlvmNvptx,
                        mechanism: BackendLoweringMechanism::TypedNvvm,
                        evidence_profile: admission.llvm_evidence_profile.clone(),
                        minimum_ptx: Some("3.2".into()),
                        minimum_sm: Some("sm_20".into()),
                    },
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LibNvvm,
                        mechanism: BackendLoweringMechanism::TypedNvvm,
                        evidence_profile: admission.libnvvm_evidence_profile.clone(),
                        minimum_ptx: Some("7.0".into()),
                        minimum_sm: Some("sm_80".into()),
                    },
                ],
                packed_atomic: None,
                redux: None,
                vote: None,
                active_mask: None,
                warp_match: None,
                warp_barrier: None,
                warp_shuffle: None,
                dot_product: None,
                packed_alu: None,
                packed_conversion: None,
                cp_async_copy: None,
                cp_async_control: None,
                cp_async_mbarrier: None,
                mbarrier_basic: None,
                movmatrix: None,
                mbarrier_extended: None,
                register_mma: None,
                sparse_mma: None,
                prmt: None,
                cluster_barrier: None,
                special_register: None,
                debug_control: None,
                cluster_memory: None,
                clc: None,
                ldmatrix_variant: None,
                ldmatrix_safety: None,
                ldmatrix_adapter: None,
                selected_address_space: None,
                expected_ptx: InstructionPattern {
                    mnemonic: "membar".into(),
                    modifiers: vec![recipe.ptx_level.into()],
                    operands: vec![],
                },
                summary: recipe.summary.into(),
            })
        })
        .collect()
}

fn validate_threadfence_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
    scope: ThreadfenceScope,
) -> Result<()> {
    let recipe = threadfence_recipe(scope);
    ensure!(
        recipe.scope == scope
            && policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none(),
        "{} threadfence identity does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "fence"
            && policy.rust_name == recipe.id
            && policy.rust_arguments.is_empty()
            && policy.rust_result == "()"
            && policy.safe
            && !policy.must_use
            && policy.public_rust_path == format!("cuda_intrinsics::fence::{}", recipe.id)
            && policy.compatibility_rust_paths
                == [
                    format!("cuda_device::fence::{}", recipe.id),
                    format!("cuda_device::{}", recipe.id),
                ],
        "{} threadfence Rust API does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands.is_empty()
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments.is_empty()
            && policy.llvm_results.is_empty()
            && policy.lowering == "direct_nvvm",
        "{} threadfence carrier or lowering does not match its closed recipe",
        policy.id
    );
    ensure!(
        declaration.properties == ["IntrNoCallback"]
            && !policy.pure
            && policy.memory == "read_write"
            && !policy.convergent
            && policy.execution_scope == recipe.execution_scope,
        "{} threadfence effects disagree with the imported declaration",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == recipe.minimum_ptx
            && policy.minimum_sm.as_deref() == recipe.minimum_sm
            && policy.ptx_result == "()"
            && policy.targets == "all"
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "9.7.14.4 Parallel Synchronization and Communication Instructions: membar / fence"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar-fence",
        "{} threadfence target floor or PTX provenance changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx
            == (InstructionPattern {
                mnemonic: "membar".into(),
                modifiers: vec![recipe.ptx_level.into()],
                operands: vec![],
            }),
        "{} expected PTX does not match its closed threadfence scope",
        policy.id
    );
    ensure!(
        policy.backend_lowerings.len() == 2
            && policy.backend_lowerings.iter().any(|route| {
                route.backend == IntrinsicBackend::LlvmNvptx
                    && route.mechanism == BackendLoweringMechanism::TypedNvvm
                    && route.minimum_ptx.as_deref() == Some("3.2")
                    && route.minimum_sm.as_deref() == Some("sm_20")
            })
            && policy.backend_lowerings.iter().any(|route| {
                route.backend == IntrinsicBackend::LibNvvm
                    && route.mechanism == BackendLoweringMechanism::TypedNvvm
                    && route.minimum_ptx.as_deref() == Some("7.0")
                    && route.minimum_sm.as_deref() == Some("sm_80")
            }),
        "{} must keep both reviewed typed threadfence routes",
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
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.cp_async_mbarrier.is_none()
            && policy.mbarrier_basic.is_none()
            && policy.register_mma.is_none()
            && policy.sparse_mma.is_none()
            && policy.prmt.is_none()
            && policy.cluster_barrier.is_none()
            && policy.special_register.is_none()
            && policy.debug_control.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with threadfence",
        policy.id
    );
    Ok(())
}

#[derive(Clone, Copy)]
struct ClcRecipe {
    operation: ClcOperation,
    abi_id: &'static str,
    id: &'static str,
    operation_key: &'static str,
    source_record: &'static str,
    llvm_symbol: &'static str,
    rust_arguments: &'static [&'static str],
    llvm_arguments: &'static [&'static str],
    llvm_results: &'static [&'static str],
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    dialect_operands: &'static [&'static str],
    dialect_results: &'static [&'static str],
    adapter: ClcAdapter,
    modifiers: &'static [&'static str],
    operands: &'static [OperandPattern],
    targets: &'static str,
    minimum_sm: Option<&'static str>,
    pure: bool,
    memory: &'static str,
    execution_scope: &'static str,
    summary: &'static str,
}

fn clc_recipe(operation: ClcOperation) -> ClcRecipe {
    const TRY_OPERANDS: &[OperandPattern] = &[OperandPattern::Address, OperandPattern::Address];
    const QUERY_OPERANDS: &[OperandPattern] = &[OperandPattern::Register, OperandPattern::Register];
    let (abi_id, id, operation_key, source_record, llvm_symbol, op_type, op_name) = match operation
    {
        ClcOperation::TryCancel => (
            "i0322",
            "clc_try_cancel",
            "cluster.launch_control.try_cancel",
            "int_nvvm_clusterlaunchcontrol_try_cancel_async_shared",
            "llvm.nvvm.clusterlaunchcontrol.try_cancel.async.shared",
            "ClcTryCancelOp",
            "nvvm.clc_try_cancel",
        ),
        ClcOperation::TryCancelMulticast => (
            "i0323",
            "clc_try_cancel_multicast",
            "cluster.launch_control.try_cancel.multicast",
            "int_nvvm_clusterlaunchcontrol_try_cancel_async_multicast_shared",
            "llvm.nvvm.clusterlaunchcontrol.try_cancel.async.multicast.shared",
            "ClcTryCancelMulticastOp",
            "nvvm.clc_try_cancel_multicast",
        ),
        ClcOperation::QueryIsCanceled => (
            "i0324",
            "clc_query_is_canceled",
            "cluster.launch_control.query.is_canceled",
            "int_nvvm_clusterlaunchcontrol_query_cancel_is_canceled",
            "llvm.nvvm.clusterlaunchcontrol.query_cancel.is_canceled",
            "ClcQueryIsCanceledOp",
            "nvvm.clc_query_is_canceled",
        ),
        ClcOperation::QueryGetFirstCtaidX => (
            "i0325",
            "clc_query_get_first_ctaid_x",
            "cluster.launch_control.query.first_ctaid.x",
            "int_nvvm_clusterlaunchcontrol_query_cancel_get_first_ctaid_x",
            "llvm.nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.x",
            "ClcQueryGetFirstCtaidXOp",
            "nvvm.clc_query_get_first_ctaid_x",
        ),
        ClcOperation::QueryGetFirstCtaidY => (
            "i0326",
            "clc_query_get_first_ctaid_y",
            "cluster.launch_control.query.first_ctaid.y",
            "int_nvvm_clusterlaunchcontrol_query_cancel_get_first_ctaid_y",
            "llvm.nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.y",
            "ClcQueryGetFirstCtaidYOp",
            "nvvm.clc_query_get_first_ctaid_y",
        ),
        ClcOperation::QueryGetFirstCtaidZ => (
            "i0327",
            "clc_query_get_first_ctaid_z",
            "cluster.launch_control.query.first_ctaid.z",
            "int_nvvm_clusterlaunchcontrol_query_cancel_get_first_ctaid_z",
            "llvm.nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.z",
            "ClcQueryGetFirstCtaidZOp",
            "nvvm.clc_query_get_first_ctaid_z",
        ),
    };
    match operation {
        ClcOperation::TryCancel | ClcOperation::TryCancelMulticast => ClcRecipe {
            operation,
            abi_id,
            id,
            operation_key,
            source_record,
            llvm_symbol,
            rust_arguments: &["*mut u8", "*mut u64"],
            llvm_arguments: &["shared_ptr", "shared_ptr"],
            llvm_results: &[],
            dialect_op_type: op_type,
            dialect_op_name: op_name,
            dialect_operands: &["ptr", "ptr"],
            dialect_results: &[],
            adapter: ClcAdapter::GenericPointersToShared,
            modifiers: if operation == ClcOperation::TryCancel {
                &[
                    "try_cancel",
                    "async",
                    "shared::cta",
                    "mbarrier::complete_tx::bytes",
                    "b128",
                ]
            } else {
                &[
                    "try_cancel",
                    "async",
                    "shared::cta",
                    "mbarrier::complete_tx::bytes",
                    "multicast::cluster::all",
                    "b128",
                ]
            },
            operands: TRY_OPERANDS,
            targets: if operation == ClcOperation::TryCancel {
                "all"
            } else {
                "sm_100a|sm_101a|sm_110a|sm_120a"
            },
            minimum_sm: if operation == ClcOperation::TryCancel {
                Some("sm_100")
            } else {
                None
            },
            pure: false,
            memory: "read_write",
            execution_scope: "cta",
            summary: if operation == ClcOperation::TryCancel {
                "Requests one pending CTA and writes its response to shared memory."
            } else {
                "Requests one pending CTA and multicasts its response across the cluster."
            },
        },
        ClcOperation::QueryIsCanceled
        | ClcOperation::QueryGetFirstCtaidX
        | ClcOperation::QueryGetFirstCtaidY
        | ClcOperation::QueryGetFirstCtaidZ => {
            let (adapter, modifiers, summary) = match operation {
                ClcOperation::QueryIsCanceled => (
                    ClcAdapter::PairU64ToI128BoolToU32,
                    &["query_cancel", "is_canceled", "pred", "b128"] as &[_],
                    "Returns whether the Cluster Launch Control request was canceled.",
                ),
                ClcOperation::QueryGetFirstCtaidX => (
                    ClcAdapter::PairU64ToI128U32,
                    &["query_cancel", "get_first_ctaid::x", "b32", "b128"] as &[_],
                    "Returns the X coordinate from a successful cancellation response.",
                ),
                ClcOperation::QueryGetFirstCtaidY => (
                    ClcAdapter::PairU64ToI128U32,
                    &["query_cancel", "get_first_ctaid::y", "b32", "b128"] as &[_],
                    "Returns the Y coordinate from a successful cancellation response.",
                ),
                ClcOperation::QueryGetFirstCtaidZ => (
                    ClcAdapter::PairU64ToI128U32,
                    &["query_cancel", "get_first_ctaid::z", "b32", "b128"] as &[_],
                    "Returns the Z coordinate from a successful cancellation response.",
                ),
                _ => unreachable!(),
            };
            ClcRecipe {
                operation,
                abi_id,
                id,
                operation_key,
                source_record,
                llvm_symbol,
                rust_arguments: &["u64", "u64"],
                llvm_arguments: &["i128"],
                llvm_results: if operation == ClcOperation::QueryIsCanceled {
                    &["i1"]
                } else {
                    &["i32"]
                },
                dialect_op_type: op_type,
                dialect_op_name: op_name,
                dialect_operands: &["i64", "i64"],
                dialect_results: &["i32"],
                adapter,
                modifiers,
                operands: QUERY_OPERANDS,
                targets: "all",
                minimum_sm: Some("sm_100"),
                pure: true,
                memory: "none",
                execution_scope: "thread",
                summary,
            }
        }
    }
}

fn expand_clc_admission(admission: &ClcAdmission) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "CLC runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "compact CLC admission requires both backend evidence profiles"
    );
    let expected = [
        ClcOperation::TryCancel,
        ClcOperation::TryCancelMulticast,
        ClcOperation::QueryIsCanceled,
        ClcOperation::QueryGetFirstCtaidX,
        ClcOperation::QueryGetFirstCtaidY,
        ClcOperation::QueryGetFirstCtaidZ,
    ];
    ensure!(
        admission
            .variants
            .iter()
            .map(|variant| variant.operation)
            .eq(expected),
        "compact CLC admission must list all six operations in canonical order"
    );

    admission
        .variants
        .iter()
        .map(|variant| {
            let recipe = clc_recipe(variant.operation);
            ensure!(
                variant.abi_id == recipe.abi_id,
                "{} must keep reserved ABI ID {}",
                recipe.id,
                recipe.abi_id
            );
            let query = recipe.pure;
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: variant.abi_id.clone(),
                operation_key: recipe.operation_key.into(),
                family: "clc".into(),
                source: None,
                source_record: Some(recipe.source_record.into()),
                rust_module: "clc".into(),
                rust_name: recipe.id.into(),
                rust_arguments: recipe.rust_arguments.iter().map(|value| (*value).into()).collect(),
                rust_result: if query { "u32".into() } else { "()".into() },
                safe: false,
                must_use: false,
                safe_allowlist_reason: None,
                public_rust_path: format!("cuda_intrinsics::clc::{}", recipe.id),
                compatibility_rust_paths: vec![format!("cuda_device::clc::{}", recipe.id)],
                dialect_op_type: recipe.dialect_op_type.into(),
                dialect_op_name: recipe.dialect_op_name.into(),
                dialect_operands: recipe.dialect_operands.iter().map(|value| (*value).into()).collect(),
                dialect_results: recipe.dialect_results.iter().map(|value| (*value).into()).collect(),
                llvm_symbol: Some(recipe.llvm_symbol.into()),
                resolved_llvm_symbol: None,
                llvm_arguments: recipe.llvm_arguments.iter().map(|value| (*value).into()).collect(),
                llvm_results: recipe.llvm_results.iter().map(|value| (*value).into()).collect(),
                pure: recipe.pure,
                memory: recipe.memory.into(),
                convergent: false,
                execution_scope: recipe.execution_scope.into(),
                minimum_ptx: "8.6".into(),
                minimum_sm: recipe.minimum_sm.map(Into::into),
                ptx_result: if query { "u32".into() } else { "()".into() },
                targets: recipe.targets.into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section: "9.7.14.18-19 Cluster Launch Control".into(),
                ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel".into(),
                lowering: "generated_clc".into(),
                backend_lowerings: vec![
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LlvmNvptx,
                        mechanism: BackendLoweringMechanism::TypedNvvm,
                        evidence_profile: admission.llvm_evidence_profile.clone(),
                        minimum_ptx: Some("8.6".into()),
                        minimum_sm: recipe.minimum_sm.map(Into::into),
                    },
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LibNvvm,
                        mechanism: BackendLoweringMechanism::TypedNvvm,
                        evidence_profile: admission.libnvvm_evidence_profile.clone(),
                        minimum_ptx: Some("8.6".into()),
                        minimum_sm: recipe.minimum_sm.map(Into::into),
                    },
                ],
                packed_atomic: None,
                redux: None,
                vote: None,
                active_mask: None,
                warp_match: None,
                warp_barrier: None,
                warp_shuffle: None,
                dot_product: None,
                packed_alu: None,
                packed_conversion: None,
                cp_async_copy: None,
                cp_async_control: None,
                cp_async_mbarrier: None,
                mbarrier_basic: None,
                movmatrix: None,
                mbarrier_extended: None,
                register_mma: None,
                sparse_mma: None,
                prmt: None,
                cluster_barrier: None,
                special_register: None,
                debug_control: None,
                cluster_memory: None,
                clc: Some(Clc {
                    operation: recipe.operation,
                    adapter: recipe.adapter,
                    runtime_validation: admission.runtime_validation,
                }),
                ldmatrix_variant: None,
                ldmatrix_safety: None,
                ldmatrix_adapter: None,
                selected_address_space: None,
                expected_ptx: InstructionPattern {
                    mnemonic: "clusterlaunchcontrol".into(),
                    modifiers: recipe.modifiers.iter().map(|value| (*value).into()).collect(),
                    operands: if query {
                        vec![
                            OperandPattern::Register,
                            OperandPattern::Exact { value: "%clc_handle".into() },
                        ]
                    } else {
                        recipe.operands.to_vec()
                    },
                },
                summary: recipe.summary.into(),
            })
        })
        .collect()
}

fn validate_clc_policy(policy: &OverlayIntrinsic, declaration: &ImportedIntrinsic) -> Result<()> {
    let clc = policy
        .clc
        .as_ref()
        .with_context(|| format!("{} has no closed CLC contract", policy.id))?;
    let recipe = clc_recipe(clc.operation);
    let query = recipe.pure;
    ensure!(
        policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none()
            && declaration.source_record == recipe.source_record
            && declaration.llvm_name == recipe.llvm_symbol,
        "{} CLC identity changed",
        policy.id
    );
    ensure!(
        policy.rust_module == "clc"
            && policy.rust_name == recipe.id
            && policy.rust_arguments == recipe.rust_arguments
            && policy.rust_result == if query { "u32" } else { "()" }
            && !policy.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::clc::{}", recipe.id)
            && policy.compatibility_rust_paths == [format!("cuda_device::clc::{}", recipe.id)],
        "{} CLC Rust API changed",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == recipe.dialect_operands
            && policy.dialect_results == recipe.dialect_results
            && policy.llvm_arguments == recipe.llvm_arguments
            && policy.llvm_results == recipe.llvm_results
            && policy.lowering == "generated_clc",
        "{} CLC carrier or LLVM adapter changed",
        policy.id
    );
    ensure!(
        declaration.arguments == recipe.llvm_arguments
            && declaration.results == recipe.llvm_results
            && declaration.properties
                == if query {
                    vec!["IntrNoMem", "IntrSpeculatable"]
                } else {
                    vec!["IntrArgMemOnly", "IntrHasSideEffects"]
                },
        "{} imported CLC declaration changed",
        policy.id
    );
    ensure!(
        policy.pure == recipe.pure
            && policy.memory == recipe.memory
            && !policy.convergent
            && policy.execution_scope == recipe.execution_scope
            && clc.adapter == recipe.adapter
            && clc.runtime_validation == RuntimeValidation::Unexecuted,
        "{} CLC semantics changed",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == "8.6"
            && policy.minimum_sm.as_deref() == recipe.minimum_sm
            && policy.targets == recipe.targets
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_result == if query { "u32" } else { "()" },
        "{} CLC target contract changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "clusterlaunchcontrol"
            && policy.expected_ptx.modifiers == recipe.modifiers
            && policy.expected_ptx.operands
                == if query {
                    vec![
                        OperandPattern::Register,
                        OperandPattern::Exact {
                            value: "%clc_handle".into(),
                        },
                    ]
                } else {
                    recipe.operands.to_vec()
                }
            && policy.backend_lowerings.len() == 2
            && policy.backend_lowerings.iter().all(|route| {
                route.mechanism == BackendLoweringMechanism::TypedNvvm
                    && route.minimum_ptx.as_deref() == Some("8.6")
                    && route.minimum_sm.as_deref() == recipe.minimum_sm
                    && !route.evidence_profile.trim().is_empty()
            }),
        "{} CLC PTX shape or backend route changed",
        policy.id
    );
    ensure_no_other_family_contract(policy, "CLC")?;
    Ok(())
}

fn validate_sync_policy(policy: &OverlayIntrinsic, declaration: &ImportedIntrinsic) -> Result<()> {
    if let Some(scope) = threadfence_scope_for_id(&policy.id) {
        return validate_threadfence_policy(policy, declaration, scope);
    }
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
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.mbarrier_basic.is_none()
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
            && vote.legacy_pre_sm70
                == PreSm70MemberMaskRule::AllNamedLanesConvergedAndOnlyNamedLanesActive
            && vote.adapter == VoteAdapter::DirectMaskPredicate
            && vote.mask_encoding == MaskEncoding::RegisterOrImmediate,
        "{} requests an unsupported vote participation, pre-sm70 rule, adapter, or mask encoding",
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
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.mbarrier_basic.is_none()
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
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.mbarrier_basic.is_none()
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
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.mbarrier_basic.is_none()
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
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

fn validate_warp_barrier_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let barrier = policy
        .warp_barrier
        .as_ref()
        .with_context(|| format!("{} has no closed warp-barrier contract", policy.id))?;
    ensure!(
        barrier.participation
            == WarpBarrierParticipation::ExecutingLaneNamedAllNamedLanesSameInstructionAndMask
            && barrier.legacy_pre_sm70
                == PreSm70MemberMaskRule::AllNamedLanesConvergedAndOnlyNamedLanesActive
            && barrier.adapter == WarpBarrierAdapter::DirectMemberMask
            && barrier.mask_encoding == WarpBarrierMaskEncoding::RegisterOrImmediate
            && barrier.memory_ordering == WarpBarrierMemoryOrdering::ParticipatingLanes,
        "{} requests an unsupported warp-barrier participation, legacy rule, adapter, mask encoding, or memory ordering",
        policy.id
    );
    ensure!(
        policy.id == "sync_mask"
            && policy.abi_id == "i0049"
            && policy.operation_key == "warp.barrier.sync.masked"
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some("int_nvvm_bar_warp_sync")
            && policy.llvm_symbol.as_deref() == Some("llvm.nvvm.bar.warp.sync")
            && policy.resolved_llvm_symbol.is_none(),
        "{} warp-barrier identity does not match the closed sync_mask recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "warp"
            && policy.rust_name == "sync_mask"
            && policy.rust_arguments == ["u32"]
            && policy.rust_result == "()"
            && !policy.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == "cuda_intrinsics::warp::sync_mask"
            && policy.compatibility_rust_paths == ["cuda_device::warp::sync_mask"],
        "{} must keep its unsafe raw API and safe cuda-device compatibility path distinct",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == "BarWarpSyncOp"
            && policy.dialect_op_name == "nvvm.bar_warp_sync"
            && policy.dialect_operands == ["i32"]
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments == ["i32"]
            && policy.llvm_results.is_empty()
            && policy.lowering == "generated_warp_barrier",
        "{} is outside the closed one-mask warp-barrier lowering recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "read_write"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == "6.0"
            && policy.minimum_sm.as_deref() == Some("sm_30")
            && policy.ptx_result == "()"
            && policy.targets == "all",
        "{} warp-barrier effects or target floor disagree with the closed recipe",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "9.7.14.2 Parallel Synchronization and Communication Instructions: bar.warp.sync"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-bar-warp-sync",
        "{} warp-barrier PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        declaration.properties == ["IntrConvergent", "IntrNoCallback"],
        "{} warp-barrier effects disagree with the imported declaration",
        policy.id
    );
    ensure!(
        policy.packed_atomic.is_none()
            && policy.redux.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.warp_shuffle.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.mbarrier_basic.is_none()
            && policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with warp_barrier",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "bar"
            && policy.expected_ptx.modifiers == ["warp", "sync"]
            && policy.expected_ptx.operands == [OperandPattern::RegisterOrImmediate],
        "{} expected PTX does not match bar.warp.sync mask",
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
                        BackendLoweringMechanism::TypedNvvm,
                    ),
                ]),
        "{} must define exactly the reviewed typed LLVM and libNVVM routes",
        policy.id
    );
    for lowering in &policy.backend_lowerings {
        let floor_matches = match lowering.backend {
            IntrinsicBackend::LlvmNvptx => {
                lowering.mechanism == BackendLoweringMechanism::TypedNvvm
                    && lowering.minimum_ptx.as_deref() == Some("6.0")
                    && lowering.minimum_sm.as_deref() == Some("sm_30")
            }
            IntrinsicBackend::LibNvvm => {
                lowering.mechanism == BackendLoweringMechanism::TypedNvvm
                    && lowering.minimum_ptx.as_deref() == Some("6.0")
                    && lowering.minimum_sm.as_deref() == Some("sm_75")
            }
        };
        ensure!(
            floor_matches && !lowering.evidence_profile.trim().is_empty(),
            "{} backend {:?} does not carry its reviewed warp-barrier profile floor",
            policy.id,
            lowering.backend
        );
    }

    let expected_selection_records = BTreeSet::from(["INT_BAR_WARP_SYNC_I", "INT_BAR_WARP_SYNC_R"]);
    let actual_selection_records: BTreeSet<_> = declaration
        .selections
        .iter()
        .map(|selection| selection.source_record.as_str())
        .collect();
    ensure!(
        declaration.selections.len() == 2 && actual_selection_records == expected_selection_records,
        "{} warp-barrier declaration must contain exactly its immediate/register selection pair",
        policy.id
    );
    for selection in &declaration.selections {
        ensure!(
            selection.asm == "bar.warp.sync \t$i;"
                && selection.predicates
                    == [
                        "Subtarget->getPTXVersion() >= 60",
                        "Subtarget->getSmVersion() >= 30",
                    ]
                && selection.constraints.is_empty(),
            "{} warp-barrier selections disagree on PTX shape, target predicates, or constraints",
            policy.id
        );
    }
    Ok(())
}

fn validate_warp_shuffle_policy(
    policy: &OverlayIntrinsic,
    declaration: Option<&ImportedIntrinsic>,
) -> Result<()> {
    let shuffle = policy
        .warp_shuffle
        .as_ref()
        .with_context(|| format!("{} has no closed warp-shuffle contract", policy.id))?;
    let recipe = warp_shuffle_recipe(shuffle.mode, shuffle.value_kind);
    ensure!(
        shuffle.participation
            == WarpShuffleParticipation::ExecutingLaneNamedAllNamedLanesSameInstructionAndMask
            && shuffle.legacy_pre_sm70
                == PreSm70MemberMaskRule::AllNamedLanesConvergedAndOnlyNamedLanesActive
            && shuffle.source_lane
                == WarpShuffleSourceLane::InRangeSourceActiveAndNamedOutOfRangeCopiesSelf
            && shuffle.adapter == recipe.adapter
            && shuffle.clamp == recipe.clamp
            && shuffle.lane_encoding == recipe.operand_encoding
            && shuffle.mask_encoding == recipe.operand_encoding,
        "{} requests an unsupported warp-shuffle semantic or operand contract",
        policy.id
    );

    let source_matches = match recipe.source {
        WarpShuffleRecipeSource::LlvmImported {
            source_record,
            llvm_symbol,
        } => {
            policy.source.is_none()
                && policy.source_record.as_deref() == Some(source_record)
                && policy.llvm_symbol.as_deref() == Some(llvm_symbol)
                && policy.resolved_llvm_symbol.is_none()
        }
        WarpShuffleRecipeSource::PtxNative { instruction } => {
            policy.source
                == Some(IntrinsicSource::PtxNative {
                    instruction: instruction.into(),
                })
                && policy.source_record.is_none()
                && policy.llvm_symbol.is_none()
                && policy.resolved_llvm_symbol.is_none()
                && policy.llvm_arguments.is_empty()
                && policy.llvm_results.is_empty()
        }
    };
    ensure!(
        policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && source_matches,
        "{} warp-shuffle identity does not match its closed mode and value recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "warp"
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == ["u32", recipe.rust_value, "u32"]
            && policy.rust_result == recipe.rust_value
            && !policy.safe
            && policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::warp::{}", recipe.rust_name)
            && policy.compatibility_rust_paths
                == [format!("cuda_device::warp::{}", recipe.rust_name)],
        "{} must preserve its unsafe must-use warp-shuffle raw API and compatibility path",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == ["i32", recipe.dialect_value, "i32"]
            && policy.dialect_results == [recipe.dialect_value]
            && policy.lowering == recipe.lowering
            && match recipe.source {
                WarpShuffleRecipeSource::LlvmImported { .. } => {
                    policy.llvm_arguments == ["i32", recipe.dialect_value, "i32", "i32"]
                        && policy.llvm_results == [recipe.dialect_value]
                }
                WarpShuffleRecipeSource::PtxNative { .. } => {
                    policy.llvm_arguments.is_empty() && policy.llvm_results.is_empty()
                }
            },
        "{} is outside the closed warp-shuffle lowering recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "inaccessible_read_write"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == "6.0"
            && policy.minimum_sm.as_deref() == Some("sm_30")
            && policy.ptx_result == recipe.rust_value
            && policy.targets == "all",
        "{} warp-shuffle effects, carrier, or target floor disagree with its recipe",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "9.7.9.6 Data Movement and Conversion Instructions: shfl.sync"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync",
        "{} warp-shuffle PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    if let Some(declaration) = declaration {
        ensure!(
            matches!(recipe.source, WarpShuffleRecipeSource::LlvmImported { .. })
                && declaration.classes
                    == [
                        "ClangBuiltin",
                        "NVVMBuiltin",
                        "SDPatternOperator",
                        "Intrinsic"
                    ]
                && declaration.properties
                    == [
                        "IntrConvergent",
                        "IntrInaccessibleMemOnly",
                        "IntrNoCallback",
                    ],
            "{} warp-shuffle class or effects disagree with the imported declaration",
            policy.id
        );
    } else {
        ensure!(
            matches!(recipe.source, WarpShuffleRecipeSource::PtxNative { .. }),
            "{} imported warp shuffle is missing its LLVM declaration",
            policy.id
        );
    }
    ensure!(
        policy.packed_atomic.is_none()
            && policy.redux.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.warp_barrier.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.mbarrier_basic.is_none()
            && policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with warp_shuffle",
        policy.id
    );
    let expected_operands = match recipe.adapter {
        WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp => vec![
            OperandPattern::Register,
            OperandPattern::Register,
            OperandPattern::RegisterOrImmediate,
            OperandPattern::Exact {
                value: recipe.clamp.to_string(),
            },
            OperandPattern::RegisterOrImmediate,
        ],
        WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble => vec![
            OperandPattern::Exact { value: "lo".into() },
            OperandPattern::Exact { value: "lo".into() },
            OperandPattern::Register,
            OperandPattern::Exact {
                value: recipe.clamp.to_string(),
            },
            OperandPattern::Register,
        ],
    };
    ensure!(
        policy.expected_ptx.mnemonic == "shfl"
            && policy.expected_ptx.modifiers == ["sync", recipe.ptx_mode, "b32"]
            && policy.expected_ptx.operands == expected_operands,
        "{} expected PTX does not match its closed shfl.sync recipe",
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
                    (IntrinsicBackend::LlvmNvptx, recipe.backend_mechanism),
                    (IntrinsicBackend::LibNvvm, recipe.backend_mechanism),
                ]),
        "{} must define exactly the reviewed LLVM and libNVVM routes",
        policy.id
    );
    for lowering in &policy.backend_lowerings {
        let floor_matches = match lowering.backend {
            IntrinsicBackend::LlvmNvptx => {
                lowering.mechanism == recipe.backend_mechanism
                    && lowering.minimum_ptx.as_deref() == Some("6.0")
                    && lowering.minimum_sm.as_deref() == Some("sm_30")
            }
            IntrinsicBackend::LibNvvm => {
                lowering.mechanism == recipe.backend_mechanism
                    && lowering.minimum_ptx.as_deref() == Some("6.0")
                    && lowering.minimum_sm.as_deref() == Some("sm_75")
            }
        };
        ensure!(
            floor_matches && !lowering.evidence_profile.trim().is_empty(),
            "{} backend {:?} does not carry its reviewed warp-shuffle profile floor",
            policy.id,
            lowering.backend
        );
    }

    if let Some(declaration) = declaration {
        let selection_records: BTreeSet<_> = declaration
            .selections
            .iter()
            .map(|selection| selection.source_record.as_str())
            .collect();
        ensure!(
            declaration.selections.len() == 8
                && selection_records.len() == 8
                && selection_records
                    .iter()
                    .all(|source_record| !source_record.trim().is_empty()),
            "{} warp-shuffle declaration must contain exactly eight distinct operand-encoding selections",
            policy.id
        );
        let expected_asm = format!(
            "shfl.sync.{}.b32 \t$dst, $src, $offset, $mask, $threadmask;",
            recipe.ptx_mode
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
                "{} warp-shuffle selections disagree on PTX shape, target predicates, or constraints",
                policy.id
            );
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum WarpShuffleRecipeSource {
    LlvmImported {
        source_record: &'static str,
        llvm_symbol: &'static str,
    },
    PtxNative {
        instruction: &'static str,
    },
}

struct WarpShuffleRecipe {
    id: &'static str,
    abi_id: &'static str,
    operation_key: &'static str,
    source: WarpShuffleRecipeSource,
    rust_name: &'static str,
    rust_value: &'static str,
    dialect_value: &'static str,
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    ptx_mode: &'static str,
    clamp: u32,
    adapter: WarpShuffleAdapter,
    operand_encoding: WarpShuffleOperandEncoding,
    lowering: &'static str,
    backend_mechanism: BackendLoweringMechanism,
}

fn warp_shuffle_recipe(
    mode: WarpShuffleMode,
    value_kind: WarpShuffleValueKind,
) -> WarpShuffleRecipe {
    match (mode, value_kind) {
        (WarpShuffleMode::Idx, WarpShuffleValueKind::I32) => WarpShuffleRecipe {
            id: "shuffle_sync",
            abi_id: "i0050",
            operation_key: "warp.shuffle.sync.idx.i32",
            source: WarpShuffleRecipeSource::LlvmImported {
                source_record: "int_nvvm_shfl_sync_idx_i32",
                llvm_symbol: "llvm.nvvm.shfl.sync.idx.i32",
            },
            rust_name: "shuffle_sync",
            rust_value: "u32",
            dialect_value: "i32",
            dialect_op_type: "ShflSyncIdxI32Op",
            dialect_op_name: "nvvm.shfl_sync_idx_i32",
            ptx_mode: "idx",
            clamp: 31,
            adapter: WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOrImmediate,
            lowering: "generated_warp_shuffle",
            backend_mechanism: BackendLoweringMechanism::TypedNvvm,
        },
        (WarpShuffleMode::Bfly, WarpShuffleValueKind::I32) => WarpShuffleRecipe {
            id: "shuffle_xor_sync",
            abi_id: "i0051",
            operation_key: "warp.shuffle.sync.bfly.i32",
            source: WarpShuffleRecipeSource::LlvmImported {
                source_record: "int_nvvm_shfl_sync_bfly_i32",
                llvm_symbol: "llvm.nvvm.shfl.sync.bfly.i32",
            },
            rust_name: "shuffle_xor_sync",
            rust_value: "u32",
            dialect_value: "i32",
            dialect_op_type: "ShflSyncBflyI32Op",
            dialect_op_name: "nvvm.shfl_sync_bfly_i32",
            ptx_mode: "bfly",
            clamp: 31,
            adapter: WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOrImmediate,
            lowering: "generated_warp_shuffle",
            backend_mechanism: BackendLoweringMechanism::TypedNvvm,
        },
        (WarpShuffleMode::Down, WarpShuffleValueKind::I32) => WarpShuffleRecipe {
            id: "shuffle_down_sync",
            abi_id: "i0052",
            operation_key: "warp.shuffle.sync.down.i32",
            source: WarpShuffleRecipeSource::LlvmImported {
                source_record: "int_nvvm_shfl_sync_down_i32",
                llvm_symbol: "llvm.nvvm.shfl.sync.down.i32",
            },
            rust_name: "shuffle_down_sync",
            rust_value: "u32",
            dialect_value: "i32",
            dialect_op_type: "ShflSyncDownI32Op",
            dialect_op_name: "nvvm.shfl_sync_down_i32",
            ptx_mode: "down",
            clamp: 31,
            adapter: WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOrImmediate,
            lowering: "generated_warp_shuffle",
            backend_mechanism: BackendLoweringMechanism::TypedNvvm,
        },
        (WarpShuffleMode::Up, WarpShuffleValueKind::I32) => WarpShuffleRecipe {
            id: "shuffle_up_sync",
            abi_id: "i0053",
            operation_key: "warp.shuffle.sync.up.i32",
            source: WarpShuffleRecipeSource::LlvmImported {
                source_record: "int_nvvm_shfl_sync_up_i32",
                llvm_symbol: "llvm.nvvm.shfl.sync.up.i32",
            },
            rust_name: "shuffle_up_sync",
            rust_value: "u32",
            dialect_value: "i32",
            dialect_op_type: "ShflSyncUpI32Op",
            dialect_op_name: "nvvm.shfl_sync_up_i32",
            ptx_mode: "up",
            clamp: 0,
            adapter: WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOrImmediate,
            lowering: "generated_warp_shuffle",
            backend_mechanism: BackendLoweringMechanism::TypedNvvm,
        },
        (WarpShuffleMode::Idx, WarpShuffleValueKind::F32) => WarpShuffleRecipe {
            id: "shuffle_f32_sync",
            abi_id: "i0054",
            operation_key: "warp.shuffle.sync.idx.f32",
            source: WarpShuffleRecipeSource::LlvmImported {
                source_record: "int_nvvm_shfl_sync_idx_f32",
                llvm_symbol: "llvm.nvvm.shfl.sync.idx.f32",
            },
            rust_name: "shuffle_f32_sync",
            rust_value: "f32",
            dialect_value: "f32",
            dialect_op_type: "ShflSyncIdxF32Op",
            dialect_op_name: "nvvm.shfl_sync_idx_f32",
            ptx_mode: "idx",
            clamp: 31,
            adapter: WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOrImmediate,
            lowering: "generated_warp_shuffle",
            backend_mechanism: BackendLoweringMechanism::TypedNvvm,
        },
        (WarpShuffleMode::Bfly, WarpShuffleValueKind::F32) => WarpShuffleRecipe {
            id: "shuffle_xor_f32_sync",
            abi_id: "i0055",
            operation_key: "warp.shuffle.sync.bfly.f32",
            source: WarpShuffleRecipeSource::LlvmImported {
                source_record: "int_nvvm_shfl_sync_bfly_f32",
                llvm_symbol: "llvm.nvvm.shfl.sync.bfly.f32",
            },
            rust_name: "shuffle_xor_f32_sync",
            rust_value: "f32",
            dialect_value: "f32",
            dialect_op_type: "ShflSyncBflyF32Op",
            dialect_op_name: "nvvm.shfl_sync_bfly_f32",
            ptx_mode: "bfly",
            clamp: 31,
            adapter: WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOrImmediate,
            lowering: "generated_warp_shuffle",
            backend_mechanism: BackendLoweringMechanism::TypedNvvm,
        },
        (WarpShuffleMode::Down, WarpShuffleValueKind::F32) => WarpShuffleRecipe {
            id: "shuffle_down_f32_sync",
            abi_id: "i0056",
            operation_key: "warp.shuffle.sync.down.f32",
            source: WarpShuffleRecipeSource::LlvmImported {
                source_record: "int_nvvm_shfl_sync_down_f32",
                llvm_symbol: "llvm.nvvm.shfl.sync.down.f32",
            },
            rust_name: "shuffle_down_f32_sync",
            rust_value: "f32",
            dialect_value: "f32",
            dialect_op_type: "ShflSyncDownF32Op",
            dialect_op_name: "nvvm.shfl_sync_down_f32",
            ptx_mode: "down",
            clamp: 31,
            adapter: WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOrImmediate,
            lowering: "generated_warp_shuffle",
            backend_mechanism: BackendLoweringMechanism::TypedNvvm,
        },
        (WarpShuffleMode::Up, WarpShuffleValueKind::F32) => WarpShuffleRecipe {
            id: "shuffle_up_f32_sync",
            abi_id: "i0057",
            operation_key: "warp.shuffle.sync.up.f32",
            source: WarpShuffleRecipeSource::LlvmImported {
                source_record: "int_nvvm_shfl_sync_up_f32",
                llvm_symbol: "llvm.nvvm.shfl.sync.up.f32",
            },
            rust_name: "shuffle_up_f32_sync",
            rust_value: "f32",
            dialect_value: "f32",
            dialect_op_type: "ShflSyncUpF32Op",
            dialect_op_name: "nvvm.shfl_sync_up_f32",
            ptx_mode: "up",
            clamp: 0,
            adapter: WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOrImmediate,
            lowering: "generated_warp_shuffle",
            backend_mechanism: BackendLoweringMechanism::TypedNvvm,
        },
        (WarpShuffleMode::Idx, WarpShuffleValueKind::I64) => WarpShuffleRecipe {
            id: "shuffle_u64_sync",
            abi_id: "i0058",
            operation_key: "warp.shuffle.sync.idx.i64",
            source: WarpShuffleRecipeSource::PtxNative {
                instruction: "shfl.sync.idx.b32",
            },
            rust_name: "shuffle_u64_sync",
            rust_value: "u64",
            dialect_value: "i64",
            dialect_op_type: "ShflSyncIdxI64Op",
            dialect_op_name: "nvvm.shfl_sync_idx_i64",
            ptx_mode: "idx",
            clamp: 31,
            adapter:
                WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOnly,
            lowering: "generated_warp_shuffle_i64_inline_ptx",
            backend_mechanism: BackendLoweringMechanism::InlinePtx,
        },
        (WarpShuffleMode::Bfly, WarpShuffleValueKind::I64) => WarpShuffleRecipe {
            id: "shuffle_xor_u64_sync",
            abi_id: "i0059",
            operation_key: "warp.shuffle.sync.bfly.i64",
            source: WarpShuffleRecipeSource::PtxNative {
                instruction: "shfl.sync.bfly.b32",
            },
            rust_name: "shuffle_xor_u64_sync",
            rust_value: "u64",
            dialect_value: "i64",
            dialect_op_type: "ShflSyncBflyI64Op",
            dialect_op_name: "nvvm.shfl_sync_bfly_i64",
            ptx_mode: "bfly",
            clamp: 31,
            adapter:
                WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOnly,
            lowering: "generated_warp_shuffle_i64_inline_ptx",
            backend_mechanism: BackendLoweringMechanism::InlinePtx,
        },
        (WarpShuffleMode::Down, WarpShuffleValueKind::I64) => WarpShuffleRecipe {
            id: "shuffle_down_u64_sync",
            abi_id: "i0060",
            operation_key: "warp.shuffle.sync.down.i64",
            source: WarpShuffleRecipeSource::PtxNative {
                instruction: "shfl.sync.down.b32",
            },
            rust_name: "shuffle_down_u64_sync",
            rust_value: "u64",
            dialect_value: "i64",
            dialect_op_type: "ShflSyncDownI64Op",
            dialect_op_name: "nvvm.shfl_sync_down_i64",
            ptx_mode: "down",
            clamp: 31,
            adapter:
                WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOnly,
            lowering: "generated_warp_shuffle_i64_inline_ptx",
            backend_mechanism: BackendLoweringMechanism::InlinePtx,
        },
        (WarpShuffleMode::Up, WarpShuffleValueKind::I64) => WarpShuffleRecipe {
            id: "shuffle_up_u64_sync",
            abi_id: "i0061",
            operation_key: "warp.shuffle.sync.up.i64",
            source: WarpShuffleRecipeSource::PtxNative {
                instruction: "shfl.sync.up.b32",
            },
            rust_name: "shuffle_up_u64_sync",
            rust_value: "u64",
            dialect_value: "i64",
            dialect_op_type: "ShflSyncUpI64Op",
            dialect_op_name: "nvvm.shfl_sync_up_i64",
            ptx_mode: "up",
            clamp: 0,
            adapter:
                WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble,
            operand_encoding: WarpShuffleOperandEncoding::RegisterOnly,
            lowering: "generated_warp_shuffle_i64_inline_ptx",
            backend_mechanism: BackendLoweringMechanism::InlinePtx,
        },
    }
}

fn validate_selected_target_predicates(
    policy: &OverlayIntrinsic,
    selection: &crate::model::ImportedSelection,
) -> Result<()> {
    let mut imported_ptx = None;
    let mut imported_sm = None;
    let mut has_dot_instructions = false;
    let mut has_clc_multicast_support = false;
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
        } else if predicate == "Subtarget->hasClusterLaunchControlTryCancelMulticastSupport()" {
            ensure!(
                policy.family == "clc"
                    && policy
                        .clc
                        .as_ref()
                        .is_some_and(|clc| { clc.operation == ClcOperation::TryCancelMulticast }),
                "{} uses the CLC multicast target predicate outside that operation",
                policy.id
            );
            ensure!(
                !has_clc_multicast_support
                    && imported_ptx.is_none()
                    && imported_sm.is_none()
                    && !has_dot_instructions,
                "{} has duplicate or conflicting CLC multicast target predicates",
                policy.id
            );
            has_clc_multicast_support = true;
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
        if let Some(packed) = &policy.packed_alu {
            ensure!(
                packed.native_minimum_sm == imported_sm,
                "{} native minimum SM {} disagrees with selected instruction predicate sm_{}",
                policy.id,
                packed.native_minimum_sm,
                imported_sm
            );
        } else {
            let overlay_target = parse_hardware_target(policy)?;
            ensure!(
                overlay_target
                    == CatalogHardwareTarget::AnyOf {
                        alternatives: vec![CatalogHardwareAlternative::MinimumSm {
                            sm: imported_sm
                        }]
                    },
                "{} minimum SM {:?} disagrees with selected instruction predicate sm_{}",
                policy.id,
                policy.minimum_sm,
                imported_sm
            );
        }
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
    } else if policy.family == "clc" {
        match policy.clc.as_ref().map(|clc| clc.operation) {
            Some(ClcOperation::TryCancel) => ensure!(
                imported_ptx.is_some() && imported_sm.is_some() && selection.predicates.len() == 2,
                "{} selection must carry exactly its PTX and SM predicates",
                policy.id
            ),
            Some(ClcOperation::TryCancelMulticast) => ensure!(
                has_clc_multicast_support
                    && selection.predicates.len() == 1
                    && parse_hardware_target(policy)?
                        == CatalogHardwareTarget::AnyOf {
                            alternatives: vec![
                                CatalogHardwareAlternative::ExactArchitecture { sm: 100 },
                                CatalogHardwareAlternative::ExactArchitecture { sm: 101 },
                                CatalogHardwareAlternative::ExactArchitecture { sm: 110 },
                                CatalogHardwareAlternative::ExactArchitecture { sm: 120 },
                            ],
                        },
                "{} multicast target predicate must map to the reviewed exact architectures",
                policy.id
            ),
            _ => bail!(
                "{} query operation unexpectedly has an instruction selection",
                policy.id
            ),
        }
    } else if matches!(
        policy.family.as_str(),
        "vote"
            | "active_mask"
            | "warp_match"
            | "warp_barrier"
            | "warp_shuffle"
            | "cp_async_copy"
            | "cp_async_control"
            | "mbarrier_basic"
            | "cluster_barrier"
    ) {
        ensure!(
            imported_ptx.is_some() && imported_sm.is_some() && selection.predicates.len() == 2,
            "{} selection must carry exactly its PTX and SM predicates",
            policy.id
        );
    }
    Ok(())
}

const REVIEWED_SPECIAL_REGISTERS: [SpecialRegisterKind; 12] = [
    SpecialRegisterKind::Clock,
    SpecialRegisterKind::Clock64,
    SpecialRegisterKind::Globaltimer,
    SpecialRegisterKind::Envreg1,
    SpecialRegisterKind::Envreg2,
    SpecialRegisterKind::Smid,
    SpecialRegisterKind::Nsmid,
    SpecialRegisterKind::Gridid,
    SpecialRegisterKind::Warpid,
    SpecialRegisterKind::Nwarpid,
    SpecialRegisterKind::DynamicSmemSize,
    SpecialRegisterKind::TotalSmemSize,
];

#[derive(Clone, Copy)]
struct SpecialRegisterRecipe {
    kind: SpecialRegisterKind,
    id: &'static str,
    operation_key: &'static str,
    source_record: Option<&'static str>,
    llvm_symbol: Option<&'static str>,
    rust_module: &'static str,
    compatibility_paths: &'static [&'static str],
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    register_spelling: &'static str,
    observation: SpecialRegisterObservation,
    result_width: SpecialRegisterWidth,
    ptx_type: SpecialRegisterPtxType,
    output_constraint: SpecialRegisterOutputConstraint,
    llvm_mechanism: BackendLoweringMechanism,
    libnvvm_mechanism: BackendLoweringMechanism,
    minimum_ptx: &'static str,
    minimum_sm: Option<&'static str>,
    execution_scope: &'static str,
    ptx_isa_section: &'static str,
    ptx_isa_url: &'static str,
    selection_record: Option<&'static str>,
    selection_asm: Option<&'static str>,
    summary: &'static str,
}

fn special_register_recipe(kind: SpecialRegisterKind) -> SpecialRegisterRecipe {
    use BackendLoweringMechanism::{InlinePtx, TypedNvvm};
    use SpecialRegisterKind::*;
    use SpecialRegisterObservation::{StablePure, VolatileObservation};
    use SpecialRegisterOutputConstraint::{Register32, Register64};
    use SpecialRegisterPtxType::{B32, U32, U64};
    use SpecialRegisterWidth::{B32 as Width32, B64 as Width64};

    match kind {
        Clock => SpecialRegisterRecipe {
            kind,
            id: "clock",
            operation_key: "debug.clock",
            source_record: Some("int_nvvm_read_ptx_sreg_clock"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.clock"),
            rust_module: "debug",
            compatibility_paths: &["cuda_device::debug::clock"],
            dialect_op_type: "ReadPtxSregClockOp",
            dialect_op_name: "nvvm.read_ptx_sreg_clock",
            register_spelling: "%clock",
            observation: VolatileObservation,
            result_width: Width32,
            ptx_type: U32,
            output_constraint: Register32,
            llvm_mechanism: TypedNvvm,
            libnvvm_mechanism: TypedNvvm,
            minimum_ptx: "1.0",
            minimum_sm: None,
            execution_scope: "sm",
            ptx_isa_section: "10.23 Special Registers: %clock, %clock_hi",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-clock-clock-hi",
            selection_record: Some("SREG_CLOCK"),
            selection_asm: Some("mov.u32 \t$d, %clock;"),
            summary: "Samples the current SM's 32-bit clock counter.",
        },
        Clock64 => SpecialRegisterRecipe {
            kind,
            id: "clock64",
            operation_key: "debug.clock64",
            source_record: Some("int_nvvm_read_ptx_sreg_clock64"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.clock64"),
            rust_module: "debug",
            compatibility_paths: &["cuda_device::debug::clock64"],
            dialect_op_type: "ReadPtxSregClock64Op",
            dialect_op_name: "nvvm.read_ptx_sreg_clock64",
            register_spelling: "%clock64",
            observation: VolatileObservation,
            result_width: Width64,
            ptx_type: U64,
            output_constraint: Register64,
            llvm_mechanism: TypedNvvm,
            libnvvm_mechanism: TypedNvvm,
            minimum_ptx: "2.0",
            minimum_sm: Some("sm_20"),
            execution_scope: "sm",
            ptx_isa_section: "10.24 Special Registers: %clock64",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-clock64",
            selection_record: Some("SREG_CLOCK64"),
            selection_asm: Some("mov.u64 \t$d, %clock64;"),
            summary: "Samples the current SM's 64-bit clock counter.",
        },
        Globaltimer => SpecialRegisterRecipe {
            kind,
            id: "globaltimer",
            operation_key: "debug.global_timer",
            source_record: Some("int_nvvm_read_ptx_sreg_globaltimer"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.globaltimer"),
            rust_module: "debug",
            compatibility_paths: &["cuda_device::debug::globaltimer"],
            dialect_op_type: "ReadPtxSregGlobaltimerOp",
            dialect_op_name: "nvvm.read_ptx_sreg_globaltimer",
            register_spelling: "%globaltimer",
            observation: VolatileObservation,
            result_width: Width64,
            ptx_type: U64,
            output_constraint: Register64,
            llvm_mechanism: TypedNvvm,
            libnvvm_mechanism: InlinePtx,
            minimum_ptx: "3.1",
            minimum_sm: Some("sm_30"),
            execution_scope: "device",
            ptx_isa_section: "10.28 Special Registers: %globaltimer, %globaltimer_lo, %globaltimer_hi",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-globaltimer",
            selection_record: Some("SREG_GLOBALTIMER"),
            selection_asm: Some("mov.u64 \t$d, %globaltimer;"),
            summary: "Samples the device-wide 64-bit global timer.",
        },
        Envreg1 => SpecialRegisterRecipe {
            kind,
            id: "envreg1",
            operation_key: "grid.environment_register.1",
            source_record: Some("int_nvvm_read_ptx_sreg_envreg1"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.envreg1"),
            rust_module: "grid",
            compatibility_paths: &["cuda_device::grid::envreg1"],
            dialect_op_type: "ReadPtxSregEnvReg1Op",
            dialect_op_name: "nvvm.read_ptx_sreg_envreg1",
            register_spelling: "%envreg1",
            observation: StablePure,
            result_width: Width32,
            ptx_type: B32,
            output_constraint: Register32,
            llvm_mechanism: TypedNvvm,
            libnvvm_mechanism: TypedNvvm,
            minimum_ptx: "2.1",
            minimum_sm: None,
            execution_scope: "grid",
            ptx_isa_section: "10.27 Special Registers: %envreg<32>",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-envreg",
            selection_record: None,
            selection_asm: None,
            summary: "Reads PTX environment register 1.",
        },
        Envreg2 => SpecialRegisterRecipe {
            kind,
            id: "envreg2",
            operation_key: "grid.environment_register.2",
            source_record: Some("int_nvvm_read_ptx_sreg_envreg2"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.envreg2"),
            rust_module: "grid",
            compatibility_paths: &["cuda_device::grid::envreg2"],
            dialect_op_type: "ReadPtxSregEnvReg2Op",
            dialect_op_name: "nvvm.read_ptx_sreg_envreg2",
            register_spelling: "%envreg2",
            observation: StablePure,
            result_width: Width32,
            ptx_type: B32,
            output_constraint: Register32,
            llvm_mechanism: TypedNvvm,
            libnvvm_mechanism: TypedNvvm,
            minimum_ptx: "2.1",
            minimum_sm: None,
            execution_scope: "grid",
            ptx_isa_section: "10.27 Special Registers: %envreg<32>",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-envreg",
            selection_record: None,
            selection_asm: None,
            summary: "Reads PTX environment register 2.",
        },
        Smid => SpecialRegisterRecipe {
            kind,
            id: "smid",
            operation_key: "execution.sm_identifier",
            source_record: Some("int_nvvm_read_ptx_sreg_smid"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.smid"),
            rust_module: "thread",
            compatibility_paths: &["cuda_device::thread::smid", "cuda_device::smid"],
            dialect_op_type: "ReadPtxSregSmIdOp",
            dialect_op_name: "nvvm.read_ptx_sreg_smid",
            register_spelling: "%smid",
            observation: VolatileObservation,
            result_width: Width32,
            ptx_type: U32,
            output_constraint: Register32,
            llvm_mechanism: InlinePtx,
            libnvvm_mechanism: InlinePtx,
            minimum_ptx: "1.3",
            minimum_sm: None,
            execution_scope: "thread",
            ptx_isa_section: "10.8 Special Registers: %smid",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-smid",
            selection_record: Some("SREG_SMID"),
            selection_asm: Some("mov.u32 \t$d, %smid;"),
            summary: "Samples the SM currently executing this thread.",
        },
        Nsmid => SpecialRegisterRecipe {
            kind,
            id: "nsmid",
            operation_key: "execution.sm_identifier_bound",
            source_record: Some("int_nvvm_read_ptx_sreg_nsmid"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.nsmid"),
            rust_module: "thread",
            compatibility_paths: &["cuda_device::thread::nsmid", "cuda_device::nsmid"],
            dialect_op_type: "ReadPtxSregNsmIdOp",
            dialect_op_name: "nvvm.read_ptx_sreg_nsmid",
            register_spelling: "%nsmid",
            observation: StablePure,
            result_width: Width32,
            ptx_type: U32,
            output_constraint: Register32,
            llvm_mechanism: TypedNvvm,
            libnvvm_mechanism: TypedNvvm,
            minimum_ptx: "2.0",
            minimum_sm: Some("sm_20"),
            execution_scope: "device",
            ptx_isa_section: "10.9 Special Registers: %nsmid",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-nsmid",
            selection_record: Some("SREG_NSMID"),
            selection_asm: Some("mov.u32 \t$d, %nsmid;"),
            summary: "Returns the upper bound for SM identifiers.",
        },
        Gridid => SpecialRegisterRecipe {
            kind,
            id: "gridid",
            operation_key: "launch.grid_identifier",
            source_record: None,
            llvm_symbol: None,
            rust_module: "thread",
            compatibility_paths: &["cuda_device::thread::gridid", "cuda_device::gridid"],
            dialect_op_type: "ReadPtxSregGridIdOp",
            dialect_op_name: "nvvm.read_ptx_sreg_gridid",
            register_spelling: "%gridid",
            observation: StablePure,
            result_width: Width64,
            ptx_type: U64,
            output_constraint: Register64,
            llvm_mechanism: InlinePtx,
            libnvvm_mechanism: InlinePtx,
            minimum_ptx: "3.0",
            minimum_sm: Some("sm_30"),
            execution_scope: "grid",
            ptx_isa_section: "10.10 Special Registers: %gridid",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-gridid",
            selection_record: None,
            selection_asm: None,
            summary: "Returns the full 64-bit temporal grid identifier.",
        },
        Warpid => SpecialRegisterRecipe {
            kind,
            id: "warpid",
            operation_key: "warp.hardware_identifier",
            source_record: Some("int_nvvm_read_ptx_sreg_warpid"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.warpid"),
            rust_module: "warp",
            compatibility_paths: &["cuda_device::warp::warpid"],
            dialect_op_type: "ReadPtxSregWarpIdOp",
            dialect_op_name: "nvvm.read_ptx_sreg_warpid",
            register_spelling: "%warpid",
            observation: VolatileObservation,
            result_width: Width32,
            ptx_type: U32,
            output_constraint: Register32,
            llvm_mechanism: InlinePtx,
            libnvvm_mechanism: InlinePtx,
            minimum_ptx: "1.3",
            minimum_sm: None,
            execution_scope: "cta",
            ptx_isa_section: "10.4 Special Registers: %warpid",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-warpid",
            selection_record: Some("SREG_WARPID"),
            selection_asm: Some("mov.u32 \t$d, %warpid;"),
            summary: "Samples the hardware warp currently executing this thread.",
        },
        Nwarpid => SpecialRegisterRecipe {
            kind,
            id: "nwarpid",
            operation_key: "warp.hardware_identifier_bound",
            source_record: Some("int_nvvm_read_ptx_sreg_nwarpid"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.nwarpid"),
            rust_module: "warp",
            compatibility_paths: &["cuda_device::warp::nwarpid"],
            dialect_op_type: "ReadPtxSregNwarpIdOp",
            dialect_op_name: "nvvm.read_ptx_sreg_nwarpid",
            register_spelling: "%nwarpid",
            observation: StablePure,
            result_width: Width32,
            ptx_type: U32,
            output_constraint: Register32,
            llvm_mechanism: TypedNvvm,
            libnvvm_mechanism: TypedNvvm,
            minimum_ptx: "2.0",
            minimum_sm: Some("sm_20"),
            execution_scope: "cta",
            ptx_isa_section: "10.5 Special Registers: %nwarpid",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-nwarpid",
            selection_record: Some("SREG_NWARPID"),
            selection_asm: Some("mov.u32 \t$d, %nwarpid;"),
            summary: "Returns the upper bound for hardware warp identifiers.",
        },
        DynamicSmemSize => SpecialRegisterRecipe {
            kind,
            id: "dynamic_smem_size",
            operation_key: "shared.dynamic_size",
            source_record: Some("int_nvvm_read_ptx_sreg_dynamic_smem_size"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.dynamic_smem_size"),
            rust_module: "shared",
            compatibility_paths: &["cuda_device::shared::dynamic_smem_size"],
            dialect_op_type: "ReadPtxSregDynamicSmemSizeOp",
            dialect_op_name: "nvvm.read_ptx_sreg_dynamic_smem_size",
            register_spelling: "%dynamic_smem_size",
            observation: StablePure,
            result_width: Width32,
            ptx_type: U32,
            output_constraint: Register32,
            llvm_mechanism: InlinePtx,
            libnvvm_mechanism: InlinePtx,
            minimum_ptx: "4.1",
            minimum_sm: Some("sm_20"),
            execution_scope: "cta",
            ptx_isa_section: "10.32 Special Registers: %dynamic_smem_size",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-dynamic-smem-size",
            selection_record: Some("INT_PTX_SREG_DYNAMIC_SMEM_SIZE"),
            selection_asm: Some("mov.u32 \t$d, %dynamic_smem_size;"),
            summary: "Returns the launch-time dynamic shared-memory size in bytes.",
        },
        TotalSmemSize => SpecialRegisterRecipe {
            kind,
            id: "total_smem_size",
            operation_key: "shared.total_size",
            source_record: Some("int_nvvm_read_ptx_sreg_total_smem_size"),
            llvm_symbol: Some("llvm.nvvm.read.ptx.sreg.total_smem_size"),
            rust_module: "shared",
            compatibility_paths: &["cuda_device::shared::total_smem_size"],
            dialect_op_type: "ReadPtxSregTotalSmemSizeOp",
            dialect_op_name: "nvvm.read_ptx_sreg_total_smem_size",
            register_spelling: "%total_smem_size",
            observation: StablePure,
            result_width: Width32,
            ptx_type: U32,
            output_constraint: Register32,
            llvm_mechanism: InlinePtx,
            libnvvm_mechanism: InlinePtx,
            minimum_ptx: "4.1",
            minimum_sm: Some("sm_20"),
            execution_scope: "cta",
            ptx_isa_section: "10.30 Special Registers: %total_smem_size",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-total-smem-size",
            selection_record: Some("INT_PTX_SREG_TOTAL_SMEM_SIZE"),
            selection_asm: Some("mov.u32 \t$d, %total_smem_size;"),
            summary: "Returns the total user shared-memory allocation in bytes.",
        },
    }
}

fn special_register_ptx_type(ptx_type: SpecialRegisterPtxType) -> &'static str {
    match ptx_type {
        SpecialRegisterPtxType::B32 => "b32",
        SpecialRegisterPtxType::U32 => "u32",
        SpecialRegisterPtxType::U64 => "u64",
    }
}

fn special_register_backend_floor(
    recipe: SpecialRegisterRecipe,
    backend: IntrinsicBackend,
) -> (Option<&'static str>, Option<&'static str>) {
    match backend {
        IntrinsicBackend::LlvmNvptx => {
            let minimum_ptx = if matches!(recipe.minimum_ptx, "4.1") {
                "4.1"
            } else {
                "3.2"
            };
            let minimum_sm = if recipe.minimum_sm == Some("sm_30") {
                "sm_30"
            } else {
                "sm_20"
            };
            (Some(minimum_ptx), Some(minimum_sm))
        }
        IntrinsicBackend::LibNvvm => (None, Some("sm_75")),
    }
}

fn special_register_contract(recipe: SpecialRegisterRecipe) -> SpecialRegister {
    let llvm_exclusion =
        (recipe.kind == SpecialRegisterKind::Gridid).then(|| SpecialRegisterLlvmExclusion {
            source_record: "int_nvvm_read_ptx_sreg_gridid".into(),
            llvm_symbol: "llvm.nvvm.read.ptx.sreg.gridid".into(),
            imported_result_width: SpecialRegisterWidth::B32,
            reason: SpecialRegisterLlvmExclusionReason::ResultWidthMismatch,
        });
    SpecialRegister {
        register: recipe.kind,
        observation: recipe.observation,
        result_width: recipe.result_width,
        ptx_type: recipe.ptx_type,
        output_constraint: recipe.output_constraint,
        llvm_exclusion,
    }
}

fn expand_special_register_admission(
    admission: &SpecialRegisterAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "special-register runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "compact special-register admission requires both backend evidence profiles"
    );
    ensure!(
        admission.registers == REVIEWED_SPECIAL_REGISTERS
            && admission.product_count == REVIEWED_SPECIAL_REGISTERS.len(),
        "compact special-register admission must list the canonical 12 registers exactly once and in order"
    );

    admission
        .registers
        .iter()
        .copied()
        .map(|kind| {
            let recipe = special_register_recipe(kind);
            let width = recipe.result_width.bits();
            let rust_result = format!("u{width}");
            let dialect_result = format!("i{width}");
            let source = match recipe.source_record {
                Some(_) => None,
                None => Some(IntrinsicSource::PtxNative {
                    instruction: format!(
                        "mov.{} {}",
                        special_register_ptx_type(recipe.ptx_type),
                        recipe.register_spelling
                    ),
                }),
            };
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: String::new(),
                operation_key: recipe.operation_key.into(),
                family: "sreg".into(),
                source,
                source_record: recipe.source_record.map(str::to_owned),
                rust_module: recipe.rust_module.into(),
                rust_name: recipe.id.into(),
                rust_arguments: vec![],
                rust_result: rust_result.clone(),
                safe: true,
                must_use: false,
                safe_allowlist_reason: Some(
                    "reading this special register has no caller obligations.".into(),
                ),
                public_rust_path: format!("cuda_intrinsics::{}::{}", recipe.rust_module, recipe.id),
                compatibility_rust_paths: recipe
                    .compatibility_paths
                    .iter()
                    .map(|path| (*path).into())
                    .collect(),
                dialect_op_type: recipe.dialect_op_type.into(),
                dialect_op_name: recipe.dialect_op_name.into(),
                dialect_operands: vec![],
                dialect_results: vec![dialect_result.clone()],
                llvm_symbol: recipe.llvm_symbol.map(str::to_owned),
                resolved_llvm_symbol: None,
                llvm_arguments: vec![],
                llvm_results: recipe
                    .source_record
                    .map(|_| dialect_result)
                    .into_iter()
                    .collect(),
                pure: recipe.observation == SpecialRegisterObservation::StablePure,
                memory: if matches!(
                    recipe.kind,
                    SpecialRegisterKind::Clock
                        | SpecialRegisterKind::Clock64
                        | SpecialRegisterKind::Globaltimer
                ) {
                    "inaccessible_read_write".into()
                } else {
                    "none".into()
                },
                convergent: false,
                execution_scope: recipe.execution_scope.into(),
                minimum_ptx: recipe.minimum_ptx.into(),
                minimum_sm: recipe.minimum_sm.map(str::to_owned),
                ptx_result: rust_result,
                targets: "all".into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section: recipe.ptx_isa_section.into(),
                ptx_isa_url: recipe.ptx_isa_url.into(),
                lowering: "generated_special_register".into(),
                backend_lowerings: [
                    (
                        IntrinsicBackend::LlvmNvptx,
                        recipe.llvm_mechanism,
                        &admission.llvm_evidence_profile,
                    ),
                    (
                        IntrinsicBackend::LibNvvm,
                        recipe.libnvvm_mechanism,
                        &admission.libnvvm_evidence_profile,
                    ),
                ]
                .into_iter()
                .map(
                    |(backend, mechanism, evidence_profile)| OverlayBackendLowering {
                        minimum_ptx: special_register_backend_floor(recipe, backend)
                            .0
                            .map(str::to_owned),
                        minimum_sm: special_register_backend_floor(recipe, backend)
                            .1
                            .map(str::to_owned),
                        backend,
                        mechanism,
                        evidence_profile: evidence_profile.clone(),
                    },
                )
                .collect(),
                packed_atomic: None,
                redux: None,
                vote: None,
                active_mask: None,
                warp_match: None,
                warp_barrier: None,
                warp_shuffle: None,
                dot_product: None,
                packed_alu: None,
                packed_conversion: None,
                cp_async_copy: None,
                cp_async_control: None,
                cp_async_mbarrier: None,
                mbarrier_basic: None,
                movmatrix: None,
                mbarrier_extended: None,
                register_mma: None,
                sparse_mma: None,
                prmt: None,
                cluster_barrier: None,
                special_register: Some(special_register_contract(recipe)),
                debug_control: None,
                cluster_memory: None,
                clc: None,
                ldmatrix_variant: None,
                ldmatrix_safety: None,
                ldmatrix_adapter: None,
                selected_address_space: None,
                expected_ptx: InstructionPattern {
                    mnemonic: "mov".into(),
                    modifiers: vec![special_register_ptx_type(recipe.ptx_type).into()],
                    operands: vec![
                        OperandPattern::Register,
                        OperandPattern::Exact {
                            value: recipe.register_spelling.into(),
                        },
                    ],
                },
                summary: recipe.summary.into(),
            })
        })
        .collect()
}

fn validate_special_register_policy(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
    declaration: Option<&ImportedIntrinsic>,
) -> Result<()> {
    let special = policy
        .special_register
        .as_ref()
        .with_context(|| format!("{} has no closed special-register contract", policy.id))?;
    let recipe = special_register_recipe(special.register);
    ensure!(
        special == &special_register_contract(recipe),
        "{} special-register width, PTX type, constraint, observation, or LLVM exclusion changed",
        policy.id
    );
    let expected_source = match recipe.source_record {
        Some(source_record) => IntrinsicSource::LlvmImported {
            source_record: source_record.into(),
        },
        None => IntrinsicSource::PtxNative {
            instruction: format!(
                "mov.{} {}",
                special_register_ptx_type(recipe.ptx_type),
                recipe.register_spelling
            ),
        },
    };
    ensure!(
        policy.id == recipe.id
            && policy.operation_key == recipe.operation_key
            && source == &expected_source
            && policy.source_record.as_deref() == recipe.source_record
            && policy.llvm_symbol.as_deref() == recipe.llvm_symbol
            && policy.resolved_llvm_symbol.is_none()
            && policy.llvm_arguments.is_empty(),
        "{} special-register identity or source changed",
        policy.id
    );
    let width = recipe.result_width.bits();
    let rust_result = format!("u{width}");
    let dialect_result = format!("i{width}");
    let expected_llvm_results = recipe
        .source_record
        .map(|_| dialect_result.clone())
        .into_iter()
        .collect::<Vec<_>>();
    ensure!(
        policy.rust_module == recipe.rust_module
            && policy.rust_name == recipe.id
            && policy.rust_arguments.is_empty()
            && policy.rust_result == rust_result
            && policy.safe
            && !policy.must_use
            && policy.public_rust_path
                == format!("cuda_intrinsics::{}::{}", recipe.rust_module, recipe.id)
            && policy.compatibility_rust_paths
                == recipe
                    .compatibility_paths
                    .iter()
                    .map(|path| (*path).to_owned())
                    .collect::<Vec<_>>(),
        "{} special-register Rust API changed",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands.is_empty()
            && policy.dialect_results == [dialect_result.as_str()]
            && policy.llvm_results == expected_llvm_results
            && policy.lowering == "generated_special_register",
        "{} special-register carrier, result width, or lowering changed",
        policy.id
    );
    let expected_pure = recipe.observation == SpecialRegisterObservation::StablePure;
    let expected_memory = if matches!(
        recipe.kind,
        SpecialRegisterKind::Clock
            | SpecialRegisterKind::Clock64
            | SpecialRegisterKind::Globaltimer
    ) {
        "inaccessible_read_write"
    } else {
        "none"
    };
    ensure!(
        policy.pure == expected_pure
            && policy.memory == expected_memory
            && !policy.convergent
            && policy.execution_scope == recipe.execution_scope,
        "{} special-register observation or effects changed",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == recipe.minimum_ptx
            && policy.minimum_sm.as_deref() == recipe.minimum_sm
            && policy.ptx_result == rust_result
            && policy.targets == "all"
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == recipe.ptx_isa_section
            && policy.ptx_isa_url == recipe.ptx_isa_url,
        "{} special-register target floor or PTX provenance changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx
            == InstructionPattern {
                mnemonic: "mov".into(),
                modifiers: vec![special_register_ptx_type(recipe.ptx_type).into()],
                operands: vec![
                    OperandPattern::Register,
                    OperandPattern::Exact {
                        value: recipe.register_spelling.into(),
                    },
                ],
            },
        "{} special-register PTX shape changed",
        policy.id
    );
    let expected_routes = [
        (
            IntrinsicBackend::LlvmNvptx,
            recipe.llvm_mechanism,
            special_register_backend_floor(recipe, IntrinsicBackend::LlvmNvptx),
        ),
        (
            IntrinsicBackend::LibNvvm,
            recipe.libnvvm_mechanism,
            special_register_backend_floor(recipe, IntrinsicBackend::LibNvvm),
        ),
    ];
    ensure!(
        policy.backend_lowerings.len() == expected_routes.len()
            && policy.backend_lowerings.iter().zip(expected_routes).all(
                |(actual, (backend, mechanism, (minimum_ptx, minimum_sm)))| {
                    actual.backend == backend
                        && actual.mechanism == mechanism
                        && !actual.evidence_profile.trim().is_empty()
                        && actual.minimum_ptx.as_deref() == minimum_ptx
                        && actual.minimum_sm.as_deref() == minimum_sm
                }
            ),
        "{} special-register backend routes changed",
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
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.cp_async_mbarrier.is_none()
            && policy.mbarrier_basic.is_none()
            && policy.register_mma.is_none()
            && policy.sparse_mma.is_none()
            && policy.prmt.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes another generated-family contract with a special register",
        policy.id
    );

    match (recipe.source_record, declaration) {
        (None, None) => {}
        (Some(_), Some(declaration)) => {
            let timer = matches!(
                recipe.kind,
                SpecialRegisterKind::Clock
                    | SpecialRegisterKind::Clock64
                    | SpecialRegisterKind::Globaltimer
            );
            let expected_properties: &[&str] = if timer {
                &[
                    "IntrInaccessibleMemOnly",
                    "IntrNoCallback",
                    "IntrNoFree",
                    "IntrWillReturn",
                    "NoUndef<ret>",
                ]
            } else {
                &["IntrNoMem", "IntrSpeculatable", "NoUndef<ret>"]
            };
            ensure!(
                declaration.arguments.is_empty()
                    && declaration.results == [dialect_result.as_str()]
                    && declaration.properties
                        == expected_properties
                            .iter()
                            .map(|property| (*property).to_owned())
                            .collect::<Vec<_>>()
                    && if timer {
                        declaration
                            .classes
                            .iter()
                            .any(|class| class == "PTXReadNCSRegIntrinsic")
                    } else {
                        declaration
                            .classes
                            .iter()
                            .any(|class| class == "NVVMPureIntrinsic")
                    },
                "{} imported special-register signature, class, or properties changed",
                policy.id
            );
            match (recipe.selection_record, recipe.selection_asm) {
                (None, None) => ensure!(
                    declaration.selections.is_empty(),
                    "{} selectionless environment-register contract changed",
                    policy.id
                ),
                (Some(selection_record), Some(selection_asm)) => ensure!(
                    declaration.selections.len() == 1
                        && declaration.selections[0].source_record == selection_record
                        && declaration.selections[0].asm == selection_asm
                        && declaration.selections[0].predicates.is_empty()
                        && declaration.selections[0].constraints.is_empty(),
                    "{} imported special-register selection changed",
                    policy.id
                ),
                _ => unreachable!("closed special-register selection recipe"),
            }
        }
        _ => bail!(
            "{} special-register source and imported declaration disagree",
            policy.id
        ),
    }
    Ok(())
}

fn validate_special_register_llvm_exclusion(
    policy: &OverlayIntrinsic,
    imported_by_record: &BTreeMap<&str, &ImportedIntrinsic>,
) -> Result<()> {
    let Some(exclusion) = policy
        .special_register
        .as_ref()
        .and_then(|special| special.llvm_exclusion.as_ref())
    else {
        return Ok(());
    };
    let declaration = imported_by_record
        .get(exclusion.source_record.as_str())
        .with_context(|| {
            format!(
                "{} excludes missing imported LLVM record {}",
                policy.id, exclusion.source_record
            )
        })?;
    ensure!(
        policy.id == "gridid"
            && exclusion.reason == SpecialRegisterLlvmExclusionReason::ResultWidthMismatch
            && exclusion.imported_result_width == SpecialRegisterWidth::B32
            && declaration.llvm_name == exclusion.llvm_symbol
            && declaration.arguments.is_empty()
            && declaration.results == ["i32"]
            && declaration.properties == ["IntrNoMem", "IntrSpeculatable", "NoUndef<ret>"]
            && declaration.selections.len() == 1
            && declaration.selections[0].source_record == "SREG_GRIDID"
            && declaration.selections[0].asm == "mov.u32 \t$d, %gridid;"
            && declaration.selections[0].predicates.is_empty()
            && declaration.selections[0].constraints.is_empty()
            && policy.rust_result == "u64"
            && policy.dialect_results == ["i64"],
        "{} LLVM exclusion no longer proves the reviewed i32-to-u64 gridid width mismatch",
        policy.id
    );
    Ok(())
}

fn validate_sreg_policy(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
    declaration: Option<&ImportedIntrinsic>,
) -> Result<()> {
    if policy.special_register.is_some() {
        return validate_special_register_policy(policy, source, declaration);
    }
    let declaration = declaration.context("sreg requires imported LLVM declaration")?;
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
            && policy.special_register.is_none(),
        "{} uses a backend contract outside the direct-NVVM sreg recipe",
        policy.id
    );
    ensure_no_other_family_contract(policy, "sreg")?;
    if policy.id.starts_with("lanemask_") {
        validate_lanemask_policy(policy, declaration)?;
    }
    if is_cluster_sreg_source(&declaration.source_record) {
        validate_cluster_sreg_policy(policy, declaration)?;
    }
    Ok(())
}

fn is_cluster_sreg_source(source_record: &str) -> bool {
    source_record == "int_nvvm_read_ptx_sreg_cluster_ctarank"
        || source_record == "int_nvvm_read_ptx_sreg_cluster_nctarank"
        || source_record.starts_with("int_nvvm_read_ptx_sreg_cluster_ctaid_")
        || source_record.starts_with("int_nvvm_read_ptx_sreg_cluster_nctaid_")
        || source_record.starts_with("int_nvvm_read_ptx_sreg_clusterid_")
        || source_record.starts_with("int_nvvm_read_ptx_sreg_nclusterid_")
}

#[derive(Clone)]
struct ClusterSregRecipe {
    id: String,
    abi_id: String,
    operation_key: String,
    source_suffix: String,
    llvm_suffix: String,
    selection_record: String,
    ptx_register: String,
    compatibility_path: Option<String>,
    op_type: String,
    scope: &'static str,
    section: &'static str,
    anchor: &'static str,
    range: Option<&'static str>,
    safe_reason: String,
    summary: String,
}

#[derive(Clone, Copy)]
struct ClusterSregXyzFamilyRecipe {
    id_prefix: &'static str,
    abi_start: u16,
    operation_key_prefix: &'static str,
    source_prefix: &'static str,
    llvm_prefix: &'static str,
    selection_prefix: &'static str,
    ptx_prefix: &'static str,
    compatibility_prefix: Option<&'static str>,
    op_type_prefix: &'static str,
    scope: &'static str,
    section: &'static str,
    anchor: &'static str,
    x_range: &'static str,
    yz_range: &'static str,
    safe_reason: &'static str,
    summary: &'static str,
}

const CLUSTER_SREG_AXES: [&str; 3] = ["x", "y", "z"];

const CLUSTER_SREG_XYZ_FAMILIES: [ClusterSregXyzFamilyRecipe; 4] = [
    ClusterSregXyzFamilyRecipe {
        id_prefix: "cluster_block_idx",
        abi_start: 263,
        operation_key_prefix: "launch.cluster.block_index",
        source_prefix: "cluster_ctaid_",
        llvm_prefix: "cluster.ctaid.",
        selection_prefix: "INT_PTX_SREG_CLUSTER_CTAID_",
        ptx_prefix: "%cluster_ctaid.",
        compatibility_prefix: Some("cuda_device::cluster::cluster_ctaid"),
        op_type_prefix: "ReadPtxSregClusterCtaid",
        scope: "cta",
        section: "10.14 Special Registers: %cluster_ctaid",
        anchor: "cluster-ctaid",
        x_range: "Range<ret,0,2147483647>",
        yz_range: "Range<ret,0,65535>",
        safe_reason: "reading the read-only block index within its cluster has no caller obligations",
        summary: "Returns the block's {axis} index within its thread block cluster.",
    },
    ClusterSregXyzFamilyRecipe {
        id_prefix: "cluster_dim",
        abi_start: 266,
        operation_key_prefix: "launch.cluster.dimension",
        source_prefix: "cluster_nctaid_",
        llvm_prefix: "cluster.nctaid.",
        selection_prefix: "INT_PTX_SREG_CLUSTER_NCTAID_",
        ptx_prefix: "%cluster_nctaid.",
        compatibility_prefix: Some("cuda_device::cluster::cluster_nctaid"),
        op_type_prefix: "ReadPtxSregClusterNctaid",
        scope: "cluster",
        section: "10.15 Special Registers: %cluster_nctaid",
        anchor: "cluster-nctaid",
        x_range: "Range<ret,1,2147483648>",
        yz_range: "Range<ret,1,65536>",
        safe_reason: "reading the read-only cluster dimension has no caller obligations",
        summary: "Returns the number of blocks in the cluster's {axis} dimension.",
    },
    ClusterSregXyzFamilyRecipe {
        id_prefix: "cluster_idx",
        abi_start: 269,
        operation_key_prefix: "launch.cluster.index",
        source_prefix: "clusterid_",
        llvm_prefix: "clusterid.",
        selection_prefix: "INT_PTX_SREG_CLUSTERID_",
        ptx_prefix: "%clusterid.",
        compatibility_prefix: Some("cuda_device::cluster::__cluster_idx"),
        op_type_prefix: "ReadPtxSregClusterId",
        scope: "cluster",
        section: "10.12 Special Registers: %clusterid",
        anchor: "clusterid",
        x_range: "Range<ret,0,2147483647>",
        yz_range: "Range<ret,0,65535>",
        safe_reason: "reading the read-only cluster index has no caller obligations",
        summary: "Returns the cluster's {axis} index within the grid.",
    },
    ClusterSregXyzFamilyRecipe {
        id_prefix: "cluster_grid_dim",
        abi_start: 272,
        operation_key_prefix: "launch.cluster.grid_dimension",
        source_prefix: "nclusterid_",
        llvm_prefix: "nclusterid.",
        selection_prefix: "INT_PTX_SREG_NCLUSTERID_",
        ptx_prefix: "%nclusterid.",
        compatibility_prefix: Some("cuda_device::cluster::__cluster_grid_dim"),
        op_type_prefix: "ReadPtxSregNclusterId",
        scope: "grid",
        section: "10.13 Special Registers: %nclusterid",
        anchor: "nclusterid",
        x_range: "Range<ret,1,2147483648>",
        yz_range: "Range<ret,1,65536>",
        safe_reason: "reading the read-only cluster-grid dimension has no caller obligations",
        summary: "Returns the number of clusters in the grid's {axis} dimension.",
    },
];

fn cluster_sreg_recipes() -> Vec<ClusterSregRecipe> {
    let mut recipes = Vec::with_capacity(14);
    for family in CLUSTER_SREG_XYZ_FAMILIES {
        for (axis_index, axis) in CLUSTER_SREG_AXES.into_iter().enumerate() {
            let axis_upper = axis.to_ascii_uppercase();
            recipes.push(ClusterSregRecipe {
                id: format!("{}_{axis}", family.id_prefix),
                abi_id: format!("i{:04}", family.abi_start + axis_index as u16),
                operation_key: format!("{}.{axis}", family.operation_key_prefix),
                source_suffix: format!("{}{axis}", family.source_prefix),
                llvm_suffix: format!("{}{axis}", family.llvm_prefix),
                selection_record: format!("{}{axis}", family.selection_prefix),
                ptx_register: format!("{}{axis}", family.ptx_prefix),
                compatibility_path: family
                    .compatibility_prefix
                    .map(|prefix| format!("{prefix}{axis_upper}")),
                op_type: format!("{}{axis_upper}Op", family.op_type_prefix),
                scope: family.scope,
                section: family.section,
                anchor: family.anchor,
                range: Some(if axis == "x" {
                    family.x_range
                } else {
                    family.yz_range
                }),
                safe_reason: family.safe_reason.into(),
                summary: family.summary.replace("{axis}", &axis_upper),
            });
        }
    }
    recipes.extend([
        ClusterSregRecipe {
            id: "cluster_block_rank".into(),
            abi_id: "i0275".into(),
            operation_key: "launch.cluster.block_rank".into(),
            source_suffix: "cluster_ctarank".into(),
            llvm_suffix: "cluster.ctarank".into(),
            selection_record: "INT_PTX_SREG_CLUSTER_CTARANK".into(),
            ptx_register: "%cluster_ctarank".into(),
            compatibility_path: None,
            op_type: "ReadPtxSregClusterCtarankOp".into(),
            scope: "cta",
            section: "10.16 Special Registers: %cluster_ctarank",
            anchor: "cluster-ctarank",
            range: None,
            safe_reason:
                "reading the read-only block rank within its cluster has no caller obligations"
                    .into(),
            summary: "Returns the block's linear rank within its thread block cluster.".into(),
        },
        ClusterSregRecipe {
            id: "cluster_block_count".into(),
            abi_id: "i0276".into(),
            operation_key: "launch.cluster.block_count".into(),
            source_suffix: "cluster_nctarank".into(),
            llvm_suffix: "cluster.nctarank".into(),
            selection_record: "INT_PTX_SREG_CLUSTER_NCTARANK".into(),
            ptx_register: "%cluster_nctarank".into(),
            compatibility_path: None,
            op_type: "ReadPtxSregClusterNctarankOp".into(),
            scope: "cluster",
            section: "10.17 Special Registers: %cluster_nctarank",
            anchor: "cluster-nctarank",
            range: None,
            safe_reason: "reading the read-only block count has no caller obligations".into(),
            summary: "Returns the total number of blocks in the thread block cluster.".into(),
        },
    ]);
    recipes
}

fn expand_cluster_sreg_admission(
    admission: &ClusterSregAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.axes == CLUSTER_SREG_AXES,
        "cluster-sreg axes must be exactly x, y, z"
    );
    ensure!(
        admission.xyz_product_count == 12 && admission.record_count == 14,
        "cluster-sreg admission must expand to 12 xyz and 14 total records"
    );
    let recipes = cluster_sreg_recipes();
    ensure!(
        recipes.len() == admission.record_count,
        "cluster-sreg recipe count disagrees with its admission"
    );
    Ok(recipes.into_iter().map(cluster_sreg_policy).collect())
}

fn cluster_sreg_policy(recipe: ClusterSregRecipe) -> OverlayIntrinsic {
    let compatibility_rust_paths = recipe.compatibility_path.iter().cloned().collect();
    OverlayIntrinsic {
        id: recipe.id.clone(),
        abi_id: recipe.abi_id,
        operation_key: recipe.operation_key,
        family: "sreg".into(),
        source: None,
        source_record: Some(format!("int_nvvm_read_ptx_sreg_{}", recipe.source_suffix)),
        rust_module: "sreg".into(),
        rust_name: recipe.id.clone(),
        rust_arguments: vec![],
        rust_result: "u32".into(),
        safe: true,
        must_use: false,
        safe_allowlist_reason: Some(recipe.safe_reason),
        public_rust_path: format!("cuda_intrinsics::sreg::{}", recipe.id),
        compatibility_rust_paths,
        dialect_op_type: recipe.op_type,
        dialect_op_name: format!("nvvm.read_ptx_sreg_{}", recipe.source_suffix),
        dialect_operands: vec![],
        dialect_results: vec![],
        llvm_symbol: Some(format!("llvm.nvvm.read.ptx.sreg.{}", recipe.llvm_suffix)),
        resolved_llvm_symbol: None,
        llvm_arguments: vec![],
        llvm_results: vec!["i32".into()],
        pure: true,
        memory: "none".into(),
        convergent: false,
        execution_scope: recipe.scope.into(),
        minimum_ptx: "7.8".into(),
        minimum_sm: Some("sm_90".into()),
        ptx_result: "u32".into(),
        targets: "all".into(),
        ptx_isa_version: "9.3".into(),
        ptx_isa_section: recipe.section.into(),
        ptx_isa_url: format!(
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-{}",
            recipe.anchor
        ),
        lowering: "direct_nvvm".into(),
        backend_lowerings: vec![],
        packed_atomic: None,
        redux: None,
        vote: None,
        active_mask: None,
        warp_match: None,
        warp_barrier: None,
        warp_shuffle: None,
        dot_product: None,
        packed_alu: None,
        packed_conversion: None,
        cp_async_copy: None,
        cp_async_control: None,
        cp_async_mbarrier: None,
        mbarrier_basic: None,
        movmatrix: None,
        mbarrier_extended: None,
        register_mma: None,
        sparse_mma: None,
        prmt: None,
        cluster_barrier: None,
        special_register: None,
        debug_control: None,
        cluster_memory: None,
        clc: None,
        ldmatrix_variant: None,
        ldmatrix_safety: None,
        ldmatrix_adapter: None,
        selected_address_space: None,
        expected_ptx: InstructionPattern {
            mnemonic: "mov".into(),
            modifiers: vec!["u32".into()],
            operands: vec![
                OperandPattern::Register,
                OperandPattern::Exact {
                    value: recipe.ptx_register,
                },
            ],
        },
        summary: recipe.summary,
    }
}

fn validate_cluster_sreg_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    ensure!(
        !declaration.source_record.ends_with("_w"),
        "{} selects an unused always-zero fourth cluster-register component",
        policy.id
    );
    let recipe = cluster_sreg_recipes()
        .into_iter()
        .find(|recipe| recipe.id == policy.id)
        .with_context(|| format!("{} is not a reviewed cluster special register", policy.id))?;

    let source_record = format!("int_nvvm_read_ptx_sreg_{}", recipe.source_suffix);
    let llvm_symbol = format!("llvm.nvvm.read.ptx.sreg.{}", recipe.llvm_suffix);
    let compatibility_paths = recipe
        .compatibility_path
        .iter()
        .cloned()
        .collect::<Vec<_>>();
    ensure!(
        policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(source_record.as_str())
            && policy.llvm_symbol.as_deref() == Some(llvm_symbol.as_str())
            && policy.resolved_llvm_symbol.is_none(),
        "{} cluster-register identity does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "sreg"
            && policy.rust_name == policy.id
            && policy.rust_arguments.is_empty()
            && policy.rust_result == "u32"
            && policy.safe
            && !policy.must_use
            && policy
                .safe_allowlist_reason
                .as_deref()
                .is_some_and(|reason| !reason.is_empty())
            && policy.public_rust_path == format!("cuda_intrinsics::sreg::{}", policy.id)
            && policy.compatibility_rust_paths == compatibility_paths,
        "{} must preserve its reviewed raw and compatibility APIs",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.op_type
            && policy.dialect_op_name == format!("nvvm.read_ptx_sreg_{}", recipe.source_suffix)
            && policy.dialect_operands.is_empty()
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments.is_empty()
            && policy.llvm_results == ["i32"]
            && policy.lowering == "direct_nvvm",
        "{} is outside the closed cluster-register lowering recipe",
        policy.id
    );
    ensure!(
        policy.pure
            && policy.memory == "none"
            && !policy.convergent
            && policy.execution_scope == recipe.scope
            && policy.minimum_ptx == "7.8"
            && policy.minimum_sm.as_deref() == Some("sm_90")
            && policy.ptx_result == "u32"
            && policy.targets == "all",
        "{} cluster-register effects or target floor disagree with PTX",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == recipe.section
            && policy.ptx_isa_url
                == format!(
                    "https://docs.nvidia.com/cuda/parallel-thread-execution/#special-registers-{}",
                    recipe.anchor
                ),
        "{} cluster-register PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    let mut properties = vec!["IntrNoMem", "IntrSpeculatable", "NoUndef<ret>"];
    properties.extend(recipe.range);
    ensure!(
        declaration.arguments.is_empty()
            && declaration.results == ["i32"]
            && declaration.classes
                == [
                    "SDPatternOperator",
                    "Intrinsic",
                    "DefaultAttrsIntrinsic",
                    "NVVMPureIntrinsic",
                    "PTXReadSRegIntrinsicNB_r32",
                ]
            && declaration.properties == properties,
        "{} declaration shape or properties disagree with LLVM TableGen",
        policy.id
    );
    let [selection] = declaration.selections.as_slice() else {
        bail!("{} must have exactly one LLVM selection", policy.id);
    };
    ensure!(
        selection.source_record == recipe.selection_record
            && selection.asm == format!("mov.u32 \t$d, {};", recipe.ptx_register)
            && selection.predicates
                == [
                    "Subtarget->getSmVersion() >= 90",
                    "Subtarget->getPTXVersion() >= 78",
                ]
            && selection.constraints.address_space.is_none()
            && selection.constraints.immediate_bindings.is_empty(),
        "{} selector disagrees with LLVM TableGen",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "mov"
            && policy.expected_ptx.modifiers == ["u32"]
            && policy.expected_ptx.operands
                == [
                    OperandPattern::Register,
                    OperandPattern::Exact {
                        value: recipe.ptx_register,
                    },
                ],
        "{} expected PTX does not match its cluster register",
        policy.id
    );
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
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.mbarrier_basic.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
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
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.mbarrier_basic.is_none()
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

#[derive(Clone, Copy)]
struct StmatrixRecipe {
    multiplicity: StmatrixMultiplicity,
    layout: StmatrixLayout,
    abi_id: &'static str,
    id: &'static str,
    operation_key: &'static str,
    source_record: &'static str,
    llvm_symbol: &'static str,
    compatibility_name: &'static str,
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    summary: &'static str,
}

fn stmatrix_recipe(multiplicity: StmatrixMultiplicity, layout: StmatrixLayout) -> StmatrixRecipe {
    match (multiplicity, layout) {
        (StmatrixMultiplicity::X2, StmatrixLayout::Normal) => StmatrixRecipe {
            multiplicity,
            layout,
            abi_id: "i0301",
            id: "stmatrix_m8n8_x2_b16",
            operation_key: "matrix.stmatrix.m8n8.x2.normal.b16.shared",
            source_record: "int_nvvm_stmatrix_sync_aligned_m8n8_x2_b16",
            llvm_symbol: "llvm.nvvm.stmatrix.sync.aligned.m8n8.x2.b16",
            compatibility_name: "stmatrix_m8n8_x2",
            dialect_op_type: "StmatrixM8n8X2Op",
            dialect_op_name: "nvvm.stmatrix_m8n8_x2",
            summary: "Stores two 8×8 b16 matrix fragments cooperatively to shared memory.",
        },
        (StmatrixMultiplicity::X2, StmatrixLayout::Transposed) => StmatrixRecipe {
            multiplicity,
            layout,
            abi_id: "i0302",
            id: "stmatrix_m8n8_x2_trans_b16",
            operation_key: "matrix.stmatrix.m8n8.x2.transposed.b16.shared",
            source_record: "int_nvvm_stmatrix_sync_aligned_m8n8_x2_trans_b16",
            llvm_symbol: "llvm.nvvm.stmatrix.sync.aligned.m8n8.x2.trans.b16",
            compatibility_name: "stmatrix_m8n8_x2_trans",
            dialect_op_type: "StmatrixM8n8X2TransOp",
            dialect_op_name: "nvvm.stmatrix_m8n8_x2_trans",
            summary: "Stores two transposed 8×8 b16 matrix fragments cooperatively to shared memory.",
        },
        (StmatrixMultiplicity::X4, StmatrixLayout::Normal) => StmatrixRecipe {
            multiplicity,
            layout,
            abi_id: "i0303",
            id: "stmatrix_m8n8_x4_b16",
            operation_key: "matrix.stmatrix.m8n8.x4.normal.b16.shared",
            source_record: "int_nvvm_stmatrix_sync_aligned_m8n8_x4_b16",
            llvm_symbol: "llvm.nvvm.stmatrix.sync.aligned.m8n8.x4.b16",
            compatibility_name: "stmatrix_m8n8_x4",
            dialect_op_type: "StmatrixM8n8X4Op",
            dialect_op_name: "nvvm.stmatrix_m8n8_x4",
            summary: "Stores four 8×8 b16 matrix fragments cooperatively to shared memory.",
        },
        (StmatrixMultiplicity::X4, StmatrixLayout::Transposed) => StmatrixRecipe {
            multiplicity,
            layout,
            abi_id: "i0304",
            id: "stmatrix_m8n8_x4_trans_b16",
            operation_key: "matrix.stmatrix.m8n8.x4.transposed.b16.shared",
            source_record: "int_nvvm_stmatrix_sync_aligned_m8n8_x4_trans_b16",
            llvm_symbol: "llvm.nvvm.stmatrix.sync.aligned.m8n8.x4.trans.b16",
            compatibility_name: "stmatrix_m8n8_x4_trans",
            dialect_op_type: "StmatrixM8n8X4TransOp",
            dialect_op_name: "nvvm.stmatrix_m8n8_x4_trans",
            summary: "Stores four transposed 8×8 b16 matrix fragments cooperatively to shared memory.",
        },
    }
}

fn stmatrix_variant_for_id(id: &str) -> Option<(StmatrixMultiplicity, StmatrixLayout)> {
    match id {
        "stmatrix_m8n8_x2_b16" => Some((StmatrixMultiplicity::X2, StmatrixLayout::Normal)),
        "stmatrix_m8n8_x2_trans_b16" => {
            Some((StmatrixMultiplicity::X2, StmatrixLayout::Transposed))
        }
        "stmatrix_m8n8_x4_b16" => Some((StmatrixMultiplicity::X4, StmatrixLayout::Normal)),
        "stmatrix_m8n8_x4_trans_b16" => {
            Some((StmatrixMultiplicity::X4, StmatrixLayout::Transposed))
        }
        _ => None,
    }
}

fn expand_stmatrix_admission(admission: &StmatrixAdmission) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "stmatrix runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "compact stmatrix admission requires both backend evidence profiles"
    );
    let expected_variants = [
        (StmatrixMultiplicity::X2, StmatrixLayout::Normal),
        (StmatrixMultiplicity::X2, StmatrixLayout::Transposed),
        (StmatrixMultiplicity::X4, StmatrixLayout::Normal),
        (StmatrixMultiplicity::X4, StmatrixLayout::Transposed),
    ];
    let actual_variants = admission
        .variants
        .iter()
        .map(|variant| (variant.multiplicity, variant.layout))
        .collect::<Vec<_>>();
    ensure!(
        actual_variants == expected_variants,
        "compact stmatrix admission must contain the four reviewed variants in canonical order"
    );

    admission
        .variants
        .iter()
        .map(|variant| {
            let recipe = stmatrix_recipe(variant.multiplicity, variant.layout);
            ensure!(
                variant.abi_id == recipe.abi_id,
                "{} must keep reserved ABI ID {}",
                recipe.id,
                recipe.abi_id
            );
            let count = recipe.multiplicity.register_count();
            let mut rust_arguments = vec!["*mut u8".to_owned()];
            rust_arguments.extend(std::iter::repeat_n("u32".to_owned(), count));
            let mut dialect_operands = vec!["ptr".to_owned()];
            dialect_operands.extend(std::iter::repeat_n("i32".to_owned(), count));
            let mut llvm_arguments = vec!["anyptr".to_owned()];
            llvm_arguments.extend(std::iter::repeat_n("i32".to_owned(), count));
            let multiplicity = match recipe.multiplicity {
                StmatrixMultiplicity::X2 => "x2",
                StmatrixMultiplicity::X4 => "x4",
            };
            let mut modifiers = vec![
                "sync".into(),
                "aligned".into(),
                "m8n8".into(),
                multiplicity.into(),
            ];
            if recipe.layout == StmatrixLayout::Transposed {
                modifiers.push("trans".into());
            }
            modifiers.extend(["shared".into(), "b16".into()]);
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: variant.abi_id.clone(),
                operation_key: recipe.operation_key.into(),
                family: "stmatrix".into(),
                source: None,
                source_record: Some(recipe.source_record.into()),
                rust_module: "matrix".into(),
                rust_name: recipe.id.into(),
                rust_arguments,
                rust_result: "()".into(),
                safe: false,
                must_use: false,
                safe_allowlist_reason: None,
                public_rust_path: format!("cuda_intrinsics::matrix::{}", recipe.id),
                compatibility_rust_paths: vec![format!(
                    "cuda_device::tcgen05::{}",
                    recipe.compatibility_name
                )],
                dialect_op_type: recipe.dialect_op_type.into(),
                dialect_op_name: recipe.dialect_op_name.into(),
                dialect_operands,
                dialect_results: vec![],
                llvm_symbol: Some(recipe.llvm_symbol.into()),
                resolved_llvm_symbol: Some(format!("{}.p3", recipe.llvm_symbol)),
                llvm_arguments,
                llvm_results: vec![],
                pure: false,
                memory: "write".into(),
                convergent: true,
                execution_scope: "warp".into(),
                minimum_ptx: "7.8".into(),
                minimum_sm: Some("sm_90".into()),
                ptx_result: "()".into(),
                targets: "all".into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section: "9.7.14.5.16 Warp-level matrix store instruction: stmatrix"
                    .into(),
                ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-stmatrix".into(),
                lowering: "generated_stmatrix".into(),
                backend_lowerings: vec![
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LlvmNvptx,
                        mechanism: BackendLoweringMechanism::TypedNvvm,
                        evidence_profile: admission.llvm_evidence_profile.clone(),
                        minimum_ptx: Some("7.8".into()),
                        minimum_sm: Some("sm_90".into()),
                    },
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LibNvvm,
                        mechanism: BackendLoweringMechanism::InlinePtx,
                        evidence_profile: admission.libnvvm_evidence_profile.clone(),
                        minimum_ptx: Some("7.8".into()),
                        minimum_sm: Some("sm_90".into()),
                    },
                ],
                packed_atomic: None,
                redux: None,
                vote: None,
                active_mask: None,
                warp_match: None,
                warp_barrier: None,
                warp_shuffle: None,
                dot_product: None,
                packed_alu: None,
                packed_conversion: None,
                cp_async_copy: None,
                cp_async_control: None,
                cp_async_mbarrier: None,
                mbarrier_basic: None,
                movmatrix: None,
                mbarrier_extended: None,
                register_mma: None,
                sparse_mma: None,
                prmt: None,
                cluster_barrier: None,
                special_register: None,
                debug_control: None,
                cluster_memory: None,
                clc: None,
                ldmatrix_variant: None,
                ldmatrix_safety: None,
                ldmatrix_adapter: None,
                selected_address_space: Some(ImportedAddressSpace::Shared),
                expected_ptx: InstructionPattern {
                    mnemonic: "stmatrix".into(),
                    modifiers,
                    operands: vec![
                        OperandPattern::Address,
                        OperandPattern::RegisterList { length: count },
                    ],
                },
                summary: recipe.summary.into(),
            })
        })
        .collect()
}

fn validate_stmatrix_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let (multiplicity, layout) = stmatrix_variant_for_id(&policy.id)
        .with_context(|| format!("{} has no closed stmatrix recipe", policy.id))?;
    let recipe = stmatrix_recipe(multiplicity, layout);
    let count = multiplicity.register_count();
    let mut rust_arguments = vec!["*mut u8".to_owned()];
    rust_arguments.extend(std::iter::repeat_n("u32".to_owned(), count));
    let mut dialect_operands = vec!["ptr".to_owned()];
    dialect_operands.extend(std::iter::repeat_n("i32".to_owned(), count));
    let mut llvm_arguments = vec!["anyptr".to_owned()];
    llvm_arguments.extend(std::iter::repeat_n("i32".to_owned(), count));

    ensure!(
        policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.as_deref()
                == Some(format!("{}.p3", recipe.llvm_symbol).as_str()),
        "{} stmatrix identity does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "matrix"
            && policy.rust_name == recipe.id
            && policy.rust_arguments == rust_arguments
            && policy.rust_result == "()"
            && !policy.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::matrix::{}", recipe.id)
            && policy.compatibility_rust_paths
                == [format!(
                    "cuda_device::tcgen05::{}",
                    recipe.compatibility_name
                )],
        "{} stmatrix Rust API does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == dialect_operands
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments == llvm_arguments
            && policy.llvm_results.is_empty()
            && policy.lowering == "generated_stmatrix"
            && policy.selected_address_space == Some(ImportedAddressSpace::Shared),
        "{} stmatrix carriers or lowering do not match its closed recipe",
        policy.id
    );
    ensure!(
        declaration.properties
            == [
                "IntrArgMemOnly",
                "IntrConvergent",
                "IntrNoCallback",
                "IntrWriteMem",
                "NoCapture<arg0>",
                "WriteOnly<arg0>",
            ]
            && !policy.pure
            && policy.memory == "write"
            && policy.convergent
            && policy.execution_scope == "warp",
        "{} stmatrix effects disagree with the imported declaration",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == "7.8"
            && policy.minimum_sm.as_deref() == Some("sm_90")
            && policy.ptx_result == "()"
            && policy.targets == "all"
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "9.7.14.5.16 Warp-level matrix store instruction: stmatrix"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-stmatrix",
        "{} stmatrix target floor or PTX provenance changed",
        policy.id
    );
    let multiplicity_name = match multiplicity {
        StmatrixMultiplicity::X2 => "x2",
        StmatrixMultiplicity::X4 => "x4",
    };
    let mut modifiers = vec![
        "sync".into(),
        "aligned".into(),
        "m8n8".into(),
        multiplicity_name.into(),
    ];
    if layout == StmatrixLayout::Transposed {
        modifiers.push("trans".into());
    }
    modifiers.extend(["shared".into(), "b16".into()]);
    ensure!(
        policy.expected_ptx
            == (InstructionPattern {
                mnemonic: "stmatrix".into(),
                modifiers,
                operands: vec![
                    OperandPattern::Address,
                    OperandPattern::RegisterList { length: count },
                ],
            }),
        "{} expected PTX does not match its closed stmatrix shape",
        policy.id
    );
    ensure!(
        policy.backend_lowerings.len() == 2
            && policy.backend_lowerings.iter().all(|route| {
                !route.evidence_profile.trim().is_empty()
                    && route.minimum_ptx.as_deref() == Some("7.8")
                    && route.minimum_sm.as_deref() == Some("sm_90")
            })
            && policy.backend_lowerings.iter().any(|route| {
                route.backend == IntrinsicBackend::LlvmNvptx
                    && route.mechanism == BackendLoweringMechanism::TypedNvvm
            })
            && policy.backend_lowerings.iter().any(|route| {
                route.backend == IntrinsicBackend::LibNvvm
                    && route.mechanism == BackendLoweringMechanism::InlinePtx
            }),
        "{} must keep its reviewed typed-NVVM and inline-PTX routes",
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
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.cp_async_mbarrier.is_none()
            && policy.mbarrier_basic.is_none()
            && policy.register_mma.is_none()
            && policy.sparse_mma.is_none()
            && policy.prmt.is_none()
            && policy.cluster_barrier.is_none()
            && policy.special_register.is_none()
            && policy.debug_control.is_none(),
        "{} mixes another generated-family contract with stmatrix",
        policy.id
    );
    Ok(())
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
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.mbarrier_basic.is_none(),
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
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
            && policy.dot_product.is_none()
            && policy.packed_alu.is_none()
            && policy.packed_conversion.is_none()
            && policy.cp_async_copy.is_none()
            && policy.cp_async_control.is_none()
            && policy.mbarrier_basic.is_none()
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

enum PackedAluRecipeSource {
    Imported {
        record: &'static str,
        symbol: &'static str,
        resolved_symbol: Option<&'static str>,
        arguments: &'static [&'static str],
        results: &'static [&'static str],
        properties: &'static [&'static str],
        selection: &'static str,
        selection_asm: &'static str,
    },
    PtxNative,
}

struct PackedAluRecipe {
    id: &'static str,
    abi_id: &'static str,
    operation_key: &'static str,
    rust_name: &'static str,
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    arity: usize,
    must_use: bool,
    ptx_mnemonic: &'static str,
    modifiers: &'static [&'static str],
    native_minimum_sm: u16,
    minimum_ptx: &'static str,
    minimum_sm: &'static str,
    ptx_isa_section: &'static str,
    ptx_isa_url: &'static str,
    source: PackedAluRecipeSource,
}

fn packed_alu_recipe(format: PackedAluFormat, operation: PackedAluOperation) -> PackedAluRecipe {
    match format {
        PackedAluFormat::Bf16x2 => packed_bf16x2_alu_recipe(operation),
        PackedAluFormat::F16x2 => packed_f16x2_alu_recipe(operation),
    }
}

fn packed_bf16x2_alu_recipe(operation: PackedAluOperation) -> PackedAluRecipe {
    const PURE: &[&str] = &["IntrNoMem", "IntrSpeculatable"];
    const COMMUTATIVE_PURE: &[&str] = &["Commutative", "IntrNoMem", "IntrSpeculatable"];
    match operation {
        PackedAluOperation::Fma => PackedAluRecipe {
            id: "fma_bf16x2",
            abi_id: "i0062",
            operation_key: "packed.alu.bf16x2.fma",
            rust_name: "fma_bf16x2",
            dialect_op_type: "FmaBf16x2Op",
            dialect_op_name: "nvvm.fma_bf16x2",
            arity: 3,
            must_use: false,
            ptx_mnemonic: "fma.rn.bf16x2",
            modifiers: &["rn", "bf16x2"],
            native_minimum_sm: 80,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_isa_section: "9.7.4.4 Half Precision Floating Point Instructions: fma",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fma_rn_bf16x2",
                symbol: "llvm.nvvm.fma.rn.bf16x2",
                resolved_symbol: None,
                arguments: &["v2bf16", "v2bf16", "v2bf16"],
                results: &["v2bf16"],
                properties: PURE,
                selection: "INT_NVVM_FMA_rn_bf16x2",
                selection_asm: "fma.rn.bf16x2 \t$dst, $src0, $src1, $src2;",
            },
        },
        PackedAluOperation::FmaRelu => PackedAluRecipe {
            id: "fma_relu_bf16x2",
            abi_id: "i0063",
            operation_key: "packed.alu.bf16x2.fma.relu",
            rust_name: "fma_relu_bf16x2",
            dialect_op_type: "FmaReluBf16x2Op",
            dialect_op_name: "nvvm.fma_relu_bf16x2",
            arity: 3,
            must_use: false,
            ptx_mnemonic: "fma.rn.relu.bf16x2",
            modifiers: &["rn", "relu", "bf16x2"],
            native_minimum_sm: 80,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_isa_section: "9.7.4.4 Half Precision Floating Point Instructions: fma",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fma_rn_relu_bf16x2",
                symbol: "llvm.nvvm.fma.rn.relu.bf16x2",
                resolved_symbol: None,
                arguments: &["v2bf16", "v2bf16", "v2bf16"],
                results: &["v2bf16"],
                properties: PURE,
                selection: "INT_NVVM_FMA_rn_relu_bf16x2",
                selection_asm: "fma.rn.relu.bf16x2 \t$dst, $src0, $src1, $src2;",
            },
        },
        PackedAluOperation::Add => PackedAluRecipe {
            id: "add_bf16x2",
            abi_id: "i0064",
            operation_key: "packed.alu.bf16x2.add",
            rust_name: "add_bf16x2",
            dialect_op_type: "AddBf16x2Op",
            dialect_op_name: "nvvm.add_bf16x2",
            arity: 2,
            must_use: false,
            ptx_mnemonic: "add.rn.bf16x2",
            modifiers: &["rn", "bf16x2"],
            native_minimum_sm: 90,
            minimum_ptx: "7.8",
            minimum_sm: "sm_90",
            ptx_isa_section: "9.7.4.1 Half Precision Floating Point Instructions: add",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-add",
            source: PackedAluRecipeSource::PtxNative,
        },
        PackedAluOperation::Sub => PackedAluRecipe {
            id: "sub_bf16x2",
            abi_id: "i0065",
            operation_key: "packed.alu.bf16x2.sub",
            rust_name: "sub_bf16x2",
            dialect_op_type: "SubBf16x2Op",
            dialect_op_name: "nvvm.sub_bf16x2",
            arity: 2,
            must_use: false,
            ptx_mnemonic: "sub.rn.bf16x2",
            modifiers: &["rn", "bf16x2"],
            native_minimum_sm: 90,
            minimum_ptx: "7.8",
            minimum_sm: "sm_90",
            ptx_isa_section: "9.7.4.2 Half Precision Floating Point Instructions: sub",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-sub",
            source: PackedAluRecipeSource::PtxNative,
        },
        PackedAluOperation::Mul => PackedAluRecipe {
            id: "mul_bf16x2",
            abi_id: "i0066",
            operation_key: "packed.alu.bf16x2.mul",
            rust_name: "mul_bf16x2",
            dialect_op_type: "MulBf16x2Op",
            dialect_op_name: "nvvm.mul_bf16x2",
            arity: 2,
            must_use: false,
            ptx_mnemonic: "mul.rn.bf16x2",
            modifiers: &["rn", "bf16x2"],
            native_minimum_sm: 90,
            minimum_ptx: "7.8",
            minimum_sm: "sm_90",
            ptx_isa_section: "9.7.4.3 Half Precision Floating Point Instructions: mul",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-mul",
            source: PackedAluRecipeSource::PtxNative,
        },
        PackedAluOperation::Min => PackedAluRecipe {
            id: "min_bf16x2",
            abi_id: "i0067",
            operation_key: "packed.alu.bf16x2.min",
            rust_name: "min_bf16x2",
            dialect_op_type: "MinBf16x2Op",
            dialect_op_name: "nvvm.min_bf16x2",
            arity: 2,
            must_use: false,
            ptx_mnemonic: "min.bf16x2",
            modifiers: &["bf16x2"],
            native_minimum_sm: 80,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_isa_section: "9.7.4.7 Half Precision Floating Point Instructions: min",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fmin_bf16x2",
                symbol: "llvm.nvvm.fmin.bf16x2",
                resolved_symbol: None,
                arguments: &["v2bf16", "v2bf16"],
                results: &["v2bf16"],
                properties: COMMUTATIVE_PURE,
                selection: "INT_NVVM_FMIN_bf16x2",
                selection_asm: "min.bf16x2 \t$dst, $src0, $src1;",
            },
        },
        PackedAluOperation::Max => PackedAluRecipe {
            id: "max_bf16x2",
            abi_id: "i0068",
            operation_key: "packed.alu.bf16x2.max",
            rust_name: "max_bf16x2",
            dialect_op_type: "MaxBf16x2Op",
            dialect_op_name: "nvvm.max_bf16x2",
            arity: 2,
            must_use: false,
            ptx_mnemonic: "max.bf16x2",
            modifiers: &["bf16x2"],
            native_minimum_sm: 80,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_isa_section: "9.7.4.8 Half Precision Floating Point Instructions: max",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fmax_bf16x2",
                symbol: "llvm.nvvm.fmax.bf16x2",
                resolved_symbol: None,
                arguments: &["v2bf16", "v2bf16"],
                results: &["v2bf16"],
                properties: COMMUTATIVE_PURE,
                selection: "INT_NVVM_FMAN_bf16x2",
                selection_asm: "max.bf16x2 \t$dst, $src0, $src1;",
            },
        },
        PackedAluOperation::Neg => PackedAluRecipe {
            id: "neg_bf16x2",
            abi_id: "i0069",
            operation_key: "packed.alu.bf16x2.neg",
            rust_name: "neg_bf16x2",
            dialect_op_type: "NegBf16x2Op",
            dialect_op_name: "nvvm.neg_bf16x2",
            arity: 1,
            must_use: false,
            ptx_mnemonic: "neg.bf16x2",
            modifiers: &["bf16x2"],
            native_minimum_sm: 80,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_isa_section: "9.7.4.5 Half Precision Floating Point Instructions: neg",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-neg",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_neg_bf16x2",
                symbol: "llvm.nvvm.neg.bf16x2",
                resolved_symbol: None,
                arguments: &["v2bf16"],
                results: &["v2bf16"],
                properties: PURE,
                selection: "INT_NVVM_NEG_BF16X2",
                selection_asm: "neg.bf16x2 \t$dst, $src0;",
            },
        },
        PackedAluOperation::Abs => PackedAluRecipe {
            id: "abs_bf16x2",
            abi_id: "i0070",
            operation_key: "packed.alu.bf16x2.abs",
            rust_name: "abs_bf16x2",
            dialect_op_type: "AbsBf16x2Op",
            dialect_op_name: "nvvm.abs_bf16x2",
            arity: 1,
            must_use: false,
            ptx_mnemonic: "abs.bf16x2",
            modifiers: &["bf16x2"],
            native_minimum_sm: 80,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_isa_section: "9.7.4.6 Half Precision Floating Point Instructions: abs",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-abs",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fabs",
                symbol: "llvm.nvvm.fabs",
                resolved_symbol: Some("llvm.nvvm.fabs.v2bf16"),
                arguments: &["anonymous_14"],
                results: &["anyfloat"],
                properties: PURE,
                selection: "ABS_BF16X2",
                selection_asm: "abs.bf16x2 \t$dst, $src0;",
            },
        },
    }
}

fn packed_f16x2_alu_recipe(operation: PackedAluOperation) -> PackedAluRecipe {
    const PURE: &[&str] = &["IntrNoMem", "IntrSpeculatable"];
    const COMMUTATIVE_PURE: &[&str] = &["Commutative", "IntrNoMem", "IntrSpeculatable"];
    match operation {
        PackedAluOperation::Fma => PackedAluRecipe {
            id: "fma_f16x2",
            abi_id: "i0072",
            operation_key: "packed.alu.f16x2.fma",
            rust_name: "fma_f16x2",
            dialect_op_type: "FmaF16x2Op",
            dialect_op_name: "nvvm.fma_f16x2",
            arity: 3,
            must_use: true,
            ptx_mnemonic: "fma.rn.f16x2",
            modifiers: &["rn", "f16x2"],
            native_minimum_sm: 53,
            minimum_ptx: "4.2",
            minimum_sm: "sm_70",
            ptx_isa_section: "9.7.4.4 Half Precision Floating Point Instructions: fma",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fma_rn_f16x2",
                symbol: "llvm.nvvm.fma.rn.f16x2",
                resolved_symbol: None,
                arguments: &["v2f16", "v2f16", "v2f16"],
                results: &["v2f16"],
                properties: PURE,
                selection: "INT_NVVM_FMA_rn_f16x2",
                selection_asm: "fma.rn.f16x2 \t$dst, $src0, $src1, $src2;",
            },
        },
        PackedAluOperation::FmaRelu => PackedAluRecipe {
            id: "fma_relu_f16x2",
            abi_id: "i0073",
            operation_key: "packed.alu.f16x2.fma.relu",
            rust_name: "fma_relu_f16x2",
            dialect_op_type: "FmaReluF16x2Op",
            dialect_op_name: "nvvm.fma_relu_f16x2",
            arity: 3,
            must_use: true,
            ptx_mnemonic: "fma.rn.relu.f16x2",
            modifiers: &["rn", "relu", "f16x2"],
            native_minimum_sm: 80,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_isa_section: "9.7.4.4 Half Precision Floating Point Instructions: fma",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fma_rn_relu_f16x2",
                symbol: "llvm.nvvm.fma.rn.relu.f16x2",
                resolved_symbol: None,
                arguments: &["v2f16", "v2f16", "v2f16"],
                results: &["v2f16"],
                properties: PURE,
                selection: "INT_NVVM_FMA_rn_relu_f16x2",
                selection_asm: "fma.rn.relu.f16x2 \t$dst, $src0, $src1, $src2;",
            },
        },
        PackedAluOperation::Add => PackedAluRecipe {
            id: "add_f16x2",
            abi_id: "i0074",
            operation_key: "packed.alu.f16x2.add",
            rust_name: "add_f16x2",
            dialect_op_type: "AddF16x2Op",
            dialect_op_name: "nvvm.add_f16x2",
            arity: 2,
            must_use: true,
            ptx_mnemonic: "add.rn.f16x2",
            modifiers: &["rn", "f16x2"],
            native_minimum_sm: 53,
            minimum_ptx: "4.2",
            minimum_sm: "sm_70",
            ptx_isa_section: "9.7.4.1 Half Precision Floating Point Instructions: add",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-add",
            source: PackedAluRecipeSource::PtxNative,
        },
        PackedAluOperation::Sub => PackedAluRecipe {
            id: "sub_f16x2",
            abi_id: "i0075",
            operation_key: "packed.alu.f16x2.sub",
            rust_name: "sub_f16x2",
            dialect_op_type: "SubF16x2Op",
            dialect_op_name: "nvvm.sub_f16x2",
            arity: 2,
            must_use: true,
            ptx_mnemonic: "sub.rn.f16x2",
            modifiers: &["rn", "f16x2"],
            native_minimum_sm: 53,
            minimum_ptx: "4.2",
            minimum_sm: "sm_70",
            ptx_isa_section: "9.7.4.2 Half Precision Floating Point Instructions: sub",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-sub",
            source: PackedAluRecipeSource::PtxNative,
        },
        PackedAluOperation::Mul => PackedAluRecipe {
            id: "mul_f16x2",
            abi_id: "i0076",
            operation_key: "packed.alu.f16x2.mul",
            rust_name: "mul_f16x2",
            dialect_op_type: "MulF16x2Op",
            dialect_op_name: "nvvm.mul_f16x2",
            arity: 2,
            must_use: true,
            ptx_mnemonic: "mul.rn.f16x2",
            modifiers: &["rn", "f16x2"],
            native_minimum_sm: 53,
            minimum_ptx: "4.2",
            minimum_sm: "sm_70",
            ptx_isa_section: "9.7.4.3 Half Precision Floating Point Instructions: mul",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-mul",
            source: PackedAluRecipeSource::PtxNative,
        },
        PackedAluOperation::Min => PackedAluRecipe {
            id: "min_f16x2",
            abi_id: "i0077",
            operation_key: "packed.alu.f16x2.min",
            rust_name: "min_f16x2",
            dialect_op_type: "MinF16x2Op",
            dialect_op_name: "nvvm.min_f16x2",
            arity: 2,
            must_use: true,
            ptx_mnemonic: "min.f16x2",
            modifiers: &["f16x2"],
            native_minimum_sm: 80,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_isa_section: "9.7.4.7 Half Precision Floating Point Instructions: min",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fmin_f16x2",
                symbol: "llvm.nvvm.fmin.f16x2",
                resolved_symbol: None,
                arguments: &["v2f16", "v2f16"],
                results: &["v2f16"],
                properties: COMMUTATIVE_PURE,
                selection: "INT_NVVM_FMIN_f16x2",
                selection_asm: "min.f16x2 \t$dst, $src0, $src1;",
            },
        },
        PackedAluOperation::Max => PackedAluRecipe {
            id: "max_f16x2",
            abi_id: "i0078",
            operation_key: "packed.alu.f16x2.max",
            rust_name: "max_f16x2",
            dialect_op_type: "MaxF16x2Op",
            dialect_op_name: "nvvm.max_f16x2",
            arity: 2,
            must_use: true,
            ptx_mnemonic: "max.f16x2",
            modifiers: &["f16x2"],
            native_minimum_sm: 80,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_isa_section: "9.7.4.8 Half Precision Floating Point Instructions: max",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fmax_f16x2",
                symbol: "llvm.nvvm.fmax.f16x2",
                resolved_symbol: None,
                arguments: &["v2f16", "v2f16"],
                results: &["v2f16"],
                properties: COMMUTATIVE_PURE,
                selection: "INT_NVVM_FMAN_f16x2",
                selection_asm: "max.f16x2 \t$dst, $src0, $src1;",
            },
        },
        PackedAluOperation::Neg => PackedAluRecipe {
            id: "neg_f16x2",
            abi_id: "i0079",
            operation_key: "packed.alu.f16x2.neg",
            rust_name: "neg_f16x2",
            dialect_op_type: "NegF16x2Op",
            dialect_op_name: "nvvm.neg_f16x2",
            arity: 1,
            must_use: true,
            ptx_mnemonic: "neg.f16x2",
            modifiers: &["f16x2"],
            native_minimum_sm: 53,
            minimum_ptx: "6.0",
            minimum_sm: "sm_70",
            ptx_isa_section: "9.7.4.5 Half Precision Floating Point Instructions: neg",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-neg",
            source: PackedAluRecipeSource::PtxNative,
        },
        PackedAluOperation::Abs => PackedAluRecipe {
            id: "abs_f16x2",
            abi_id: "i0080",
            operation_key: "packed.alu.f16x2.abs",
            rust_name: "abs_f16x2",
            dialect_op_type: "AbsF16x2Op",
            dialect_op_name: "nvvm.abs_f16x2",
            arity: 1,
            must_use: true,
            ptx_mnemonic: "abs.f16x2",
            modifiers: &["f16x2"],
            native_minimum_sm: 53,
            minimum_ptx: "6.5",
            minimum_sm: "sm_70",
            ptx_isa_section: "9.7.4.6 Half Precision Floating Point Instructions: abs",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-abs",
            source: PackedAluRecipeSource::Imported {
                record: "int_nvvm_fabs",
                symbol: "llvm.nvvm.fabs",
                resolved_symbol: Some("llvm.nvvm.fabs.v2f16"),
                arguments: &["anonymous_14"],
                results: &["anyfloat"],
                properties: PURE,
                selection: "ABS_F16X2",
                selection_asm: "abs.f16x2 \t$dst, $src0;",
            },
        },
    }
}

fn packed_alu_backend_floor(
    format: PackedAluFormat,
    operation: PackedAluOperation,
    backend: IntrinsicBackend,
) -> (&'static str, &'static str) {
    let recipe = packed_alu_recipe(format, operation);
    match (format, operation, backend) {
        (
            PackedAluFormat::F16x2,
            PackedAluOperation::Fma
            | PackedAluOperation::Add
            | PackedAluOperation::Sub
            | PackedAluOperation::Mul,
            IntrinsicBackend::LlvmNvptx,
        ) => ("6.0", "sm_70"),
        (
            PackedAluFormat::F16x2,
            PackedAluOperation::Fma
            | PackedAluOperation::Add
            | PackedAluOperation::Sub
            | PackedAluOperation::Mul,
            IntrinsicBackend::LibNvvm,
        ) => ("4.2", "sm_75"),
        (PackedAluFormat::F16x2, PackedAluOperation::Neg, IntrinsicBackend::LlvmNvptx) => {
            ("6.0", "sm_70")
        }
        (PackedAluFormat::F16x2, PackedAluOperation::Neg, IntrinsicBackend::LibNvvm) => {
            ("6.0", "sm_75")
        }
        (PackedAluFormat::F16x2, PackedAluOperation::Abs, IntrinsicBackend::LlvmNvptx) => {
            ("6.5", "sm_70")
        }
        (PackedAluFormat::F16x2, PackedAluOperation::Abs, IntrinsicBackend::LibNvvm) => {
            ("6.5", "sm_75")
        }
        _ => (recipe.minimum_ptx, recipe.minimum_sm),
    }
}

fn validate_packed_alu_policy(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
    declaration: Option<&ImportedIntrinsic>,
) -> Result<()> {
    let packed = policy
        .packed_alu
        .as_ref()
        .with_context(|| format!("{} has no closed packed-ALU contract", policy.id))?;
    ensure!(
        packed.adapter == PackedAluAdapter::DirectPackedU32,
        "{} requests an unsupported packed-ALU adapter",
        policy.id
    );
    let recipe = packed_alu_recipe(packed.format, packed.operation);
    let rust_module = match packed.format {
        PackedAluFormat::Bf16x2 => "bf16x2",
        PackedAluFormat::F16x2 => "f16x2",
    };
    let rust_arguments = vec!["u32"; recipe.arity];
    let dialect_operands = vec!["i32"; recipe.arity];
    ensure!(
        policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key,
        "{} packed-ALU identity does not match its closed operation recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == rust_module
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == rust_arguments
            && policy.rust_result == "u32"
            && policy.safe
            && policy.must_use == recipe.must_use
            && policy
                .safe_allowlist_reason
                .as_deref()
                .is_some_and(|reason| !reason.trim().is_empty())
            && policy.public_rust_path
                == format!("cuda_intrinsics::{rust_module}::{}", recipe.rust_name)
            && policy.compatibility_rust_paths
                == [format!("cuda_device::{rust_module}::{}", recipe.rust_name)],
        "{} must preserve its reviewed safe packed-ALU API",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == dialect_operands
            && policy.dialect_results == ["i32"]
            && policy.lowering == "generated_packed_alu_inline_ptx",
        "{} is outside the closed packed-ALU dialect and lowering recipe",
        policy.id
    );
    ensure!(
        policy.pure
            && policy.memory == "none"
            && !policy.convergent
            && policy.execution_scope == "thread"
            && policy.minimum_ptx == recipe.minimum_ptx
            && policy.minimum_sm.as_deref() == Some(recipe.minimum_sm)
            && policy.ptx_result == "u32"
            && policy.targets == "all"
            && packed.native_minimum_sm == recipe.native_minimum_sm,
        "{} packed-ALU effects, carrier, or target floor disagree",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == recipe.ptx_isa_section
            && policy.ptx_isa_url == recipe.ptx_isa_url,
        "{} packed-ALU PTX provenance does not match its reviewed instruction section",
        policy.id
    );
    let expected_operands = vec![OperandPattern::Register; recipe.arity + 1];
    ensure!(
        policy.expected_ptx.mnemonic
            == recipe.ptx_mnemonic.split('.').next().expect("PTX mnemonic")
            && policy.expected_ptx.modifiers == recipe.modifiers
            && policy.expected_ptx.operands == expected_operands,
        "{} expected PTX does not match its exact packed-ALU instruction",
        policy.id
    );

    match &recipe.source {
        PackedAluRecipeSource::PtxNative => {
            ensure!(
                source
                    == &IntrinsicSource::PtxNative {
                        instruction: recipe.ptx_mnemonic.to_owned(),
                    }
                    && declaration.is_none(),
                "{} packed-ALU source does not match its PTX-native recipe",
                policy.id
            );
        }
        PackedAluRecipeSource::Imported {
            record,
            symbol,
            resolved_symbol,
            arguments,
            results,
            properties,
            selection,
            selection_asm,
        } => {
            let declaration = declaration.context("imported packed ALU has no declaration")?;
            ensure!(
                source
                    == &IntrinsicSource::LlvmImported {
                        source_record: (*record).to_owned(),
                    }
                    && policy.llvm_symbol.as_deref() == Some(*symbol)
                    && policy.resolved_llvm_symbol.as_deref() == *resolved_symbol
                    && policy.llvm_arguments == *arguments
                    && policy.llvm_results == *results,
                "{} packed-ALU LLVM source or signature changed",
                policy.id
            );
            let matching_selections: Vec<_> = declaration
                .selections
                .iter()
                .filter(|candidate| candidate.source_record == *selection)
                .collect();
            let expected_selection_count = if *record == "int_nvvm_fabs" { 6 } else { 1 };
            ensure!(
                declaration.properties == *properties
                    && declaration.selections.len() == expected_selection_count
                    && matching_selections.len() == 1
                    && matching_selections[0].asm == *selection_asm
                    && matching_selections[0].predicates
                        == [
                            format!("Subtarget->getSmVersion() >= {}", recipe.native_minimum_sm),
                            format!(
                                "Subtarget->getPTXVersion() >= {}",
                                recipe.minimum_ptx.replace('.', "")
                            ),
                        ]
                    && matching_selections[0].constraints.is_empty(),
                "{} packed-ALU imported properties or selection changed",
                policy.id
            );
        }
    }
    let llvm_floor =
        packed_alu_backend_floor(packed.format, packed.operation, IntrinsicBackend::LlvmNvptx);
    let libnvvm_floor =
        packed_alu_backend_floor(packed.format, packed.operation, IntrinsicBackend::LibNvvm);
    ensure_exact_inline_ptx_backends(
        policy,
        [
            (
                IntrinsicBackend::LlvmNvptx,
                llvm_floor.0,
                Some(llvm_floor.1),
            ),
            (
                IntrinsicBackend::LibNvvm,
                libnvvm_floor.0,
                Some(libnvvm_floor.1),
            ),
        ],
        "packed-ALU",
    )?;
    ensure_no_other_family_contract(policy, "packed-ALU")?;
    Ok(())
}

struct PackedConversionRecipe {
    id: &'static str,
    abi_id: &'static str,
    operation_key: &'static str,
    rust_name: &'static str,
    compatibility_path: &'static str,
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    source_record: &'static str,
    llvm_symbol: &'static str,
    llvm_result: &'static str,
    summary: &'static str,
}

struct CpAsyncCopyRecipe {
    id: &'static str,
    abi_id: &'static str,
    operation_key: &'static str,
    rust_name: &'static str,
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    source_record: &'static str,
    llvm_symbol: &'static str,
    selections: &'static [&'static str],
    summary: &'static str,
}

fn cp_async_copy_recipe(copy: &crate::model::CpAsyncCopy) -> Option<CpAsyncCopyRecipe> {
    match (copy.cache_policy, copy.copy_size, copy.source_size) {
        (CpAsyncCachePolicy::Ca, CpAsyncCopySize::B4, CpAsyncSourceSize::Full) => {
            Some(CpAsyncCopyRecipe {
                id: "cp_async_ca_4",
                abi_id: "i0086",
                operation_key: "memory.copy.async.global_to_shared.ca.4.full",
                rust_name: "cp_async_ca_4",
                dialect_op_type: "CpAsyncCa4Op",
                dialect_op_name: "nvvm.cp_async_ca_4",
                source_record: "int_nvvm_cp_async_ca_shared_global_4",
                llvm_symbol: "llvm.nvvm.cp.async.ca.shared.global.4",
                selections: &["CP_ASYNC_CA_SHARED_GLOBAL_4"],
                summary: "Starts a four-byte cache-all asynchronous copy from global to shared memory.",
            })
        }
        (CpAsyncCachePolicy::Ca, CpAsyncCopySize::B4, CpAsyncSourceSize::Runtime) => {
            Some(CpAsyncCopyRecipe {
                id: "cp_async_ca_zfill_4",
                abi_id: "i0087",
                operation_key: "memory.copy.async.global_to_shared.ca.4.runtime_source_size",
                rust_name: "cp_async_ca_zfill_4",
                dialect_op_type: "CpAsyncCaZfill4Op",
                dialect_op_name: "nvvm.cp_async_ca_zfill_4",
                source_record: "int_nvvm_cp_async_ca_shared_global_4_s",
                llvm_symbol: "llvm.nvvm.cp.async.ca.shared.global.4.s",
                selections: &[
                    "CP_ASYNC_CA_SHARED_GLOBAL_4_s",
                    "CP_ASYNC_CA_SHARED_GLOBAL_4_si",
                ],
                summary: "Starts a four-byte cache-all asynchronous copy and zero-fills bytes beyond the runtime source size.",
            })
        }
        (CpAsyncCachePolicy::Ca, CpAsyncCopySize::B8, CpAsyncSourceSize::Full) => {
            Some(CpAsyncCopyRecipe {
                id: "cp_async_ca_8",
                abi_id: "i0088",
                operation_key: "memory.copy.async.global_to_shared.ca.8.full",
                rust_name: "cp_async_ca_8",
                dialect_op_type: "CpAsyncCa8Op",
                dialect_op_name: "nvvm.cp_async_ca_8",
                source_record: "int_nvvm_cp_async_ca_shared_global_8",
                llvm_symbol: "llvm.nvvm.cp.async.ca.shared.global.8",
                selections: &["CP_ASYNC_CA_SHARED_GLOBAL_8"],
                summary: "Starts an eight-byte cache-all asynchronous copy from global to shared memory.",
            })
        }
        (CpAsyncCachePolicy::Ca, CpAsyncCopySize::B8, CpAsyncSourceSize::Runtime) => {
            Some(CpAsyncCopyRecipe {
                id: "cp_async_ca_zfill_8",
                abi_id: "i0089",
                operation_key: "memory.copy.async.global_to_shared.ca.8.runtime_source_size",
                rust_name: "cp_async_ca_zfill_8",
                dialect_op_type: "CpAsyncCaZfill8Op",
                dialect_op_name: "nvvm.cp_async_ca_zfill_8",
                source_record: "int_nvvm_cp_async_ca_shared_global_8_s",
                llvm_symbol: "llvm.nvvm.cp.async.ca.shared.global.8.s",
                selections: &[
                    "CP_ASYNC_CA_SHARED_GLOBAL_8_s",
                    "CP_ASYNC_CA_SHARED_GLOBAL_8_si",
                ],
                summary: "Starts an eight-byte cache-all asynchronous copy and zero-fills bytes beyond the runtime source size.",
            })
        }
        (CpAsyncCachePolicy::Ca, CpAsyncCopySize::B16, CpAsyncSourceSize::Full) => {
            Some(CpAsyncCopyRecipe {
                id: "cp_async_ca_16",
                abi_id: "i0090",
                operation_key: "memory.copy.async.global_to_shared.ca.16.full",
                rust_name: "cp_async_ca_16",
                dialect_op_type: "CpAsyncCa16Op",
                dialect_op_name: "nvvm.cp_async_ca_16",
                source_record: "int_nvvm_cp_async_ca_shared_global_16",
                llvm_symbol: "llvm.nvvm.cp.async.ca.shared.global.16",
                selections: &["CP_ASYNC_CA_SHARED_GLOBAL_16"],
                summary: "Starts a sixteen-byte cache-all asynchronous copy from global to shared memory.",
            })
        }
        (CpAsyncCachePolicy::Ca, CpAsyncCopySize::B16, CpAsyncSourceSize::Runtime) => {
            Some(CpAsyncCopyRecipe {
                id: "cp_async_ca_zfill_16",
                abi_id: "i0091",
                operation_key: "memory.copy.async.global_to_shared.ca.16.runtime_source_size",
                rust_name: "cp_async_ca_zfill_16",
                dialect_op_type: "CpAsyncCaZfill16Op",
                dialect_op_name: "nvvm.cp_async_ca_zfill_16",
                source_record: "int_nvvm_cp_async_ca_shared_global_16_s",
                llvm_symbol: "llvm.nvvm.cp.async.ca.shared.global.16.s",
                selections: &[
                    "CP_ASYNC_CA_SHARED_GLOBAL_16_s",
                    "CP_ASYNC_CA_SHARED_GLOBAL_16_si",
                ],
                summary: "Starts a sixteen-byte cache-all asynchronous copy and zero-fills bytes beyond the runtime source size.",
            })
        }
        (CpAsyncCachePolicy::Cg, CpAsyncCopySize::B16, CpAsyncSourceSize::Full) => {
            Some(CpAsyncCopyRecipe {
                id: "cp_async_cg_16",
                abi_id: "i0092",
                operation_key: "memory.copy.async.global_to_shared.cg.16.full",
                rust_name: "cp_async_cg_16",
                dialect_op_type: "CpAsyncCg16Op",
                dialect_op_name: "nvvm.cp_async_cg_16",
                source_record: "int_nvvm_cp_async_cg_shared_global_16",
                llvm_symbol: "llvm.nvvm.cp.async.cg.shared.global.16",
                selections: &["CP_ASYNC_CG_SHARED_GLOBAL_16"],
                summary: "Starts a sixteen-byte cache-global asynchronous copy from global to shared memory.",
            })
        }
        (CpAsyncCachePolicy::Cg, CpAsyncCopySize::B16, CpAsyncSourceSize::Runtime) => {
            Some(CpAsyncCopyRecipe {
                id: "cp_async_cg_zfill_16",
                abi_id: "i0093",
                operation_key: "memory.copy.async.global_to_shared.cg.16.runtime_source_size",
                rust_name: "cp_async_cg_zfill_16",
                dialect_op_type: "CpAsyncCgZfill16Op",
                dialect_op_name: "nvvm.cp_async_cg_zfill_16",
                source_record: "int_nvvm_cp_async_cg_shared_global_16_s",
                llvm_symbol: "llvm.nvvm.cp.async.cg.shared.global.16.s",
                selections: &[
                    "CP_ASYNC_CG_SHARED_GLOBAL_16_s",
                    "CP_ASYNC_CG_SHARED_GLOBAL_16_si",
                ],
                summary: "Starts a sixteen-byte cache-global asynchronous copy and zero-fills bytes beyond the runtime source size.",
            })
        }
        _ => None,
    }
}

fn validate_cp_async_copy_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let copy = policy
        .cp_async_copy
        .as_ref()
        .with_context(|| format!("{} has no closed cp.async copy contract", policy.id))?;
    let recipe = cp_async_copy_recipe(copy).with_context(|| {
        format!(
            "{} requests an unsupported classic cp.async copy form",
            policy.id
        )
    })?;
    let dynamic_source_size = copy.source_size == CpAsyncSourceSize::Runtime;
    let expected_adapter = if dynamic_source_size {
        CpAsyncAdapter::DirectPointersAndSourceSize
    } else {
        CpAsyncAdapter::DirectPointers
    };
    ensure!(
        copy.adapter == expected_adapter,
        "{} cp.async source-size form and adapter disagree",
        policy.id
    );
    ensure!(
        copy.runtime_validation == RuntimeValidation::Unexecuted,
        "{} cannot claim unrecorded cp.async runtime validation",
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
        "{} cp.async identity does not match its closed recipe",
        policy.id
    );
    let rust_arguments = if dynamic_source_size {
        vec!["*mut u32", "*const u8", "u32"]
    } else {
        vec!["*mut u32", "*const u32"]
    };
    ensure!(
        policy.rust_module == "async_copy"
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == rust_arguments
            && policy.rust_result == "()"
            && !policy.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path
                == format!("cuda_intrinsics::async_copy::{}", recipe.rust_name)
            && policy.compatibility_rust_paths
                == [format!("cuda_device::async_copy::{}", recipe.rust_name)],
        "{} must preserve its unsafe cp.async raw and compatibility API",
        policy.id
    );
    let llvm_arguments = if dynamic_source_size {
        vec!["shared_ptr", "global_ptr", "i32"]
    } else {
        vec!["shared_ptr", "global_ptr"]
    };
    let dialect_operands = if dynamic_source_size {
        vec!["ptr", "ptr", "i32"]
    } else {
        vec!["ptr", "ptr"]
    };
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == dialect_operands
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments == llvm_arguments
            && policy.llvm_results.is_empty()
            && policy.lowering == "generated_cp_async_copy",
        "{} is outside the closed cp.async signature and lowering recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "read_write"
            && !policy.convergent
            && policy.execution_scope == "thread"
            && policy.minimum_ptx == "7.0"
            && policy.minimum_sm.as_deref() == Some("sm_80")
            && policy.ptx_result == "()"
            && policy.targets == "all",
        "{} cp.async effects or target floor disagree with the closed recipe",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "9.7.9.26.3.1 Data Movement and Conversion Instructions: cp.async"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async",
        "{} cp.async PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        policy.summary == recipe.summary,
        "{} cp.async summary does not match its closed recipe",
        policy.id
    );
    ensure!(
        declaration.properties
            == [
                "IntrArgMemOnly",
                "IntrNoCallback",
                "NoAlias<arg0>",
                "NoAlias<arg1>",
                "ReadOnly<arg1>",
                "WriteOnly<arg0>",
            ],
        "{} cp.async effects disagree with the imported LLVM declaration",
        policy.id
    );
    let cache = match copy.cache_policy {
        CpAsyncCachePolicy::Ca => "ca",
        CpAsyncCachePolicy::Cg => "cg",
    };
    let mut operands = vec![
        OperandPattern::Address,
        OperandPattern::Address,
        OperandPattern::Exact {
            value: copy.copy_size.bytes().to_string(),
        },
    ];
    if dynamic_source_size {
        operands.push(OperandPattern::RegisterOrImmediate);
    }
    ensure!(
        policy.expected_ptx.mnemonic == "cp"
            && policy.expected_ptx.modifiers == ["async", cache, "shared", "global"]
            && policy.expected_ptx.operands == operands,
        "{} expected PTX does not match its cp.async recipe",
        policy.id
    );
    let actual_selections: BTreeSet<_> = declaration
        .selections
        .iter()
        .map(|selection| selection.source_record.as_str())
        .collect();
    ensure!(
        actual_selections == recipe.selections.iter().copied().collect(),
        "{} imported cp.async selection set changed",
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
                ])
            && policy.backend_lowerings.iter().all(|lowering| {
                lowering.minimum_ptx.is_none()
                    && lowering.minimum_sm.is_none()
                    && !lowering.evidence_profile.trim().is_empty()
            }),
        "{} must define the reviewed typed-LLVM and inline-PTX cp.async routes",
        policy.id
    );
    ensure_no_other_family_contract(policy, "cp_async_copy")?;
    ensure!(
        policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes ldmatrix state with cp_async_copy",
        policy.id
    );
    Ok(())
}

struct CpAsyncControlRecipe {
    id: &'static str,
    abi_id: &'static str,
    operation_key: &'static str,
    rust_name: &'static str,
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    source_record: &'static str,
    llvm_symbol: &'static str,
    selection: &'static str,
    ptx_modifier: &'static str,
    summary: &'static str,
}

fn cp_async_control_recipe(operation: CpAsyncControlOperation) -> CpAsyncControlRecipe {
    match operation {
        CpAsyncControlOperation::CommitGroup => CpAsyncControlRecipe {
            id: "cp_async_commit_group",
            abi_id: "i0094",
            operation_key: "memory.copy.async.group.commit",
            rust_name: "cp_async_commit_group",
            dialect_op_type: "CpAsyncCommitGroupOp",
            dialect_op_name: "nvvm.cp_async_commit_group",
            source_record: "int_nvvm_cp_async_commit_group",
            llvm_symbol: "llvm.nvvm.cp.async.commit.group",
            selection: "CP_ASYNC_COMMIT_GROUP",
            ptx_modifier: "commit_group",
            summary: "Commits this thread's uncommitted asynchronous copies as one group.",
        },
        CpAsyncControlOperation::WaitAll => CpAsyncControlRecipe {
            id: "cp_async_wait_all",
            abi_id: "i0095",
            operation_key: "memory.copy.async.group.wait_all",
            rust_name: "cp_async_wait_all",
            dialect_op_type: "CpAsyncWaitAllOp",
            dialect_op_name: "nvvm.cp_async_wait_all",
            source_record: "int_nvvm_cp_async_wait_all",
            llvm_symbol: "llvm.nvvm.cp.async.wait.all",
            selection: "CP_ASYNC_WAIT_ALL",
            ptx_modifier: "wait_all",
            summary: "Waits for all asynchronous copy groups issued by this thread.",
        },
        CpAsyncControlOperation::WaitGroup => CpAsyncControlRecipe {
            id: "cp_async_wait_group",
            abi_id: "i0096",
            operation_key: "memory.copy.async.group.wait_max_pending",
            rust_name: "cp_async_wait_group",
            dialect_op_type: "CpAsyncWaitGroupOp",
            dialect_op_name: "nvvm.cp_async_wait_group",
            source_record: "int_nvvm_cp_async_wait_group",
            llvm_symbol: "llvm.nvvm.cp.async.wait.group",
            selection: "CP_ASYNC_WAIT_GROUP",
            ptx_modifier: "wait_group",
            summary: "Waits until at most the compile-time constant number of recent asynchronous copy groups remain pending.",
        },
    }
}

fn validate_cp_async_control_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let control = policy
        .cp_async_control
        .as_ref()
        .with_context(|| format!("{} has no closed cp.async control contract", policy.id))?;
    let recipe = cp_async_control_recipe(control.operation);
    let has_operand = control.operation == CpAsyncControlOperation::WaitGroup;
    let expected_adapter = if has_operand {
        CpAsyncControlAdapter::CompileTimeConstantMaxPending
    } else {
        CpAsyncControlAdapter::NoOperands
    };
    ensure!(
        control.adapter == expected_adapter,
        "{} cp.async control and adapter disagree",
        policy.id
    );
    ensure!(
        control.runtime_validation == RuntimeValidation::Unexecuted,
        "{} cannot claim unrecorded cp.async control runtime validation",
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
        "{} cp.async control identity does not match its closed recipe",
        policy.id
    );
    let rust_arguments = if has_operand { vec!["u32"] } else { vec![] };
    let dialect_operands = if has_operand { vec!["i32"] } else { vec![] };
    ensure!(
        policy.rust_module == "async_copy"
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == rust_arguments
            && policy.rust_result == "()"
            && !policy.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path
                == format!("cuda_intrinsics::async_copy::{}", recipe.rust_name)
            && policy.compatibility_rust_paths
                == [format!("cuda_device::async_copy::{}", recipe.rust_name)]
            && policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == dialect_operands
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments == dialect_operands
            && policy.llvm_results.is_empty()
            && policy.lowering == "generated_cp_async_control",
        "{} is outside the closed cp.async control API and lowering recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "read_write"
            && !policy.convergent
            && policy.execution_scope == "thread"
            && policy.minimum_ptx == "7.0"
            && policy.minimum_sm.as_deref() == Some("sm_80")
            && policy.ptx_result == "()"
            && policy.targets == "all",
        "{} cp.async control effects or target floor disagree with the closed recipe",
        policy.id
    );
    let (ptx_isa_section, ptx_isa_url) = match control.operation {
        CpAsyncControlOperation::CommitGroup => (
            "9.7.9.26.3.2 Data Movement and Conversion Instructions: cp.async.commit_group",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-commit-group",
        ),
        CpAsyncControlOperation::WaitAll | CpAsyncControlOperation::WaitGroup => (
            "9.7.9.26.3.3 Data Movement and Conversion Instructions: cp.async.wait_group / cp.async.wait_all",
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all",
        ),
    };
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == ptx_isa_section
            && policy.ptx_isa_url == ptx_isa_url,
        "{} cp.async control PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        policy.summary == recipe.summary,
        "{} cp.async control summary does not match its closed recipe",
        policy.id
    );
    let expected_properties: Vec<String> = if has_operand {
        vec!["ImmArg<arg0>".into()]
    } else {
        vec![]
    };
    ensure!(
        declaration.properties == expected_properties,
        "{} cp.async control properties disagree with the imported declaration",
        policy.id
    );
    let operands = if has_operand {
        vec![OperandPattern::Immediate]
    } else {
        vec![]
    };
    ensure!(
        policy.expected_ptx.mnemonic == "cp"
            && policy.expected_ptx.modifiers == ["async", recipe.ptx_modifier]
            && policy.expected_ptx.operands == operands
            && declaration.selections.len() == 1
            && declaration.selections[0].source_record == recipe.selection,
        "{} expected PTX or imported selection disagrees with its cp.async control recipe",
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
                ])
            && policy.backend_lowerings.iter().all(|lowering| {
                lowering.minimum_ptx.is_none()
                    && lowering.minimum_sm.is_none()
                    && !lowering.evidence_profile.trim().is_empty()
            }),
        "{} must define the reviewed typed-LLVM and inline-PTX cp.async control routes",
        policy.id
    );
    ensure_no_other_family_contract(policy, "cp_async_control")?;
    ensure!(
        policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.selected_address_space.is_none(),
        "{} mixes ldmatrix state with cp_async_control",
        policy.id
    );
    Ok(())
}

struct CpAsyncMbarrierRecipe {
    id: &'static str,
    abi_id: &'static str,
    operation_key: &'static str,
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    source_record: &'static str,
    llvm_symbol: &'static str,
    llvm_argument: &'static str,
    selection: &'static str,
    selection_asm: &'static str,
    modifiers: &'static [&'static str],
    summary: &'static str,
}

fn cp_async_mbarrier_recipe(
    operation: CpAsyncMbarrierOperation,
    state_space: CpAsyncMbarrierStateSpace,
) -> CpAsyncMbarrierRecipe {
    match (operation, state_space) {
        (CpAsyncMbarrierOperation::Arrive, CpAsyncMbarrierStateSpace::Generic) => {
            CpAsyncMbarrierRecipe {
                id: "cp_async_mbarrier_arrive",
                abi_id: "i0101",
                operation_key: "memory.copy.async.mbarrier.arrive.generic.increment",
                dialect_op_type: "CpAsyncMbarrierArriveOp",
                dialect_op_name: "nvvm.cp_async_mbarrier_arrive",
                source_record: "int_nvvm_cp_async_mbarrier_arrive",
                llvm_symbol: "llvm.nvvm.cp.async.mbarrier.arrive",
                llvm_argument: "ptr",
                selection: "CP_ASYNC_MBARRIER_ARRIVE",
                selection_asm: "cp.async.mbarrier.arrive.b64 \t[$addr];",
                modifiers: &["async", "mbarrier", "arrive", "b64"],
                summary: "Associates this thread's prior asynchronous copies with a shared-memory barrier using balanced pending-count accounting.",
            }
        }
        (CpAsyncMbarrierOperation::ArriveNoInc, CpAsyncMbarrierStateSpace::Generic) => {
            CpAsyncMbarrierRecipe {
                id: "cp_async_mbarrier_arrive_noinc",
                abi_id: "i0103",
                operation_key: "memory.copy.async.mbarrier.arrive.generic.no_increment",
                dialect_op_type: "CpAsyncMbarrierArriveNoIncOp",
                dialect_op_name: "nvvm.cp_async_mbarrier_arrive_noinc",
                source_record: "int_nvvm_cp_async_mbarrier_arrive_noinc",
                llvm_symbol: "llvm.nvvm.cp.async.mbarrier.arrive.noinc",
                llvm_argument: "ptr",
                selection: "CP_ASYNC_MBARRIER_ARRIVE_NOINC",
                selection_asm: "cp.async.mbarrier.arrive.noinc.b64 \t[$addr];",
                modifiers: &["async", "mbarrier", "arrive", "noinc", "b64"],
                summary: "Associates this thread's prior asynchronous copies with a shared-memory barrier without incrementing its pending count.",
            }
        }
        (CpAsyncMbarrierOperation::ArriveNoInc, CpAsyncMbarrierStateSpace::Shared) => {
            CpAsyncMbarrierRecipe {
                id: "cp_async_mbarrier_arrive_noinc_shared",
                abi_id: "i0104",
                operation_key: "memory.copy.async.mbarrier.arrive.shared.no_increment",
                dialect_op_type: "CpAsyncMbarrierArriveNoIncSharedOp",
                dialect_op_name: "nvvm.cp_async_mbarrier_arrive_noinc_shared",
                source_record: "int_nvvm_cp_async_mbarrier_arrive_noinc_shared",
                llvm_symbol: "llvm.nvvm.cp.async.mbarrier.arrive.noinc.shared",
                llvm_argument: "shared_ptr",
                selection: "CP_ASYNC_MBARRIER_ARRIVE_NOINC_SHARED",
                selection_asm: "cp.async.mbarrier.arrive.noinc.shared.b64 \t[$addr];",
                modifiers: &["async", "mbarrier", "arrive", "noinc", "shared", "b64"],
                summary: "Associates this thread's prior asynchronous copies with a shared-address barrier without incrementing its pending count.",
            }
        }
        (CpAsyncMbarrierOperation::Arrive, CpAsyncMbarrierStateSpace::Shared) => {
            CpAsyncMbarrierRecipe {
                id: "cp_async_mbarrier_arrive_shared",
                abi_id: "i0102",
                operation_key: "memory.copy.async.mbarrier.arrive.shared.increment",
                dialect_op_type: "CpAsyncMbarrierArriveSharedOp",
                dialect_op_name: "nvvm.cp_async_mbarrier_arrive_shared",
                source_record: "int_nvvm_cp_async_mbarrier_arrive_shared",
                llvm_symbol: "llvm.nvvm.cp.async.mbarrier.arrive.shared",
                llvm_argument: "shared_ptr",
                selection: "CP_ASYNC_MBARRIER_ARRIVE_SHARED",
                selection_asm: "cp.async.mbarrier.arrive.shared.b64 \t[$addr];",
                modifiers: &["async", "mbarrier", "arrive", "shared", "b64"],
                summary: "Associates this thread's prior asynchronous copies with a shared-address barrier using balanced pending-count accounting.",
            }
        }
    }
}

fn validate_cp_async_mbarrier_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let bridge = policy
        .cp_async_mbarrier
        .as_ref()
        .with_context(|| format!("{} has no closed cp.async mbarrier contract", policy.id))?;
    let recipe = cp_async_mbarrier_recipe(bridge.operation, bridge.state_space);
    ensure!(
        bridge.adapter == CpAsyncMbarrierAdapter::PointerToVoid,
        "{} cp.async mbarrier adapter changed",
        policy.id
    );
    ensure!(
        bridge.runtime_validation == RuntimeValidation::Unexecuted,
        "{} cannot claim unrecorded cp.async mbarrier runtime validation",
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
        "{} cp.async mbarrier identity does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "async_copy"
            && policy.rust_name == recipe.id
            && policy.rust_arguments == ["*mut u64"]
            && policy.rust_result == "()"
            && !policy.safe
            && !policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::async_copy::{}", recipe.id)
            && policy.compatibility_rust_paths
                == [format!("cuda_device::async_copy::{}", recipe.id)],
        "{} is outside the closed cp.async mbarrier Rust API",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == ["ptr"]
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments == [recipe.llvm_argument]
            && policy.llvm_results.is_empty()
            && policy.lowering == "generated_cp_async_mbarrier",
        "{} is outside the closed cp.async mbarrier signature and lowering recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "read_write"
            && policy.convergent
            && policy.execution_scope == "thread"
            && policy.minimum_ptx == "7.0"
            && policy.minimum_sm.as_deref() == Some("sm_80")
            && policy.ptx_result == "()"
            && policy.targets == "all",
        "{} cp.async mbarrier effects or target floor disagree with the closed recipe",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "9.7.14.16.18 Parallel Synchronization and Communication Instructions: cp.async.mbarrier.arrive"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive",
        "{} cp.async mbarrier PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        policy.summary == recipe.summary,
        "{} cp.async mbarrier summary does not match its closed recipe",
        policy.id
    );
    ensure!(
        declaration.properties == ["IntrConvergent", "IntrNoCallback"],
        "{} cp.async mbarrier properties disagree with the imported declaration",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "cp"
            && policy
                .expected_ptx
                .modifiers
                .iter()
                .map(String::as_str)
                .eq(recipe.modifiers.iter().copied())
            && policy.expected_ptx.operands == [OperandPattern::Address],
        "{} expected PTX does not match its cp.async mbarrier recipe",
        policy.id
    );
    ensure!(
        declaration.selections.len() == 1
            && declaration.selections[0].source_record == recipe.selection
            && declaration.selections[0].asm == recipe.selection_asm
            && declaration.selections[0].predicates
                == [
                    "Subtarget->getSmVersion() >= 80",
                    "Subtarget->getPTXVersion() >= 70",
                ]
            && declaration.selections[0].constraints.is_empty(),
        "{} imported cp.async mbarrier selection changed",
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
                ])
            && policy.backend_lowerings.iter().all(|lowering| {
                lowering.minimum_ptx.is_none()
                    && lowering.minimum_sm.is_none()
                    && !lowering.evidence_profile.trim().is_empty()
            }),
        "{} must define the reviewed typed-LLVM and inline-PTX cp.async mbarrier routes",
        policy.id
    );
    ensure_no_other_family_contract(policy, "cp_async_mbarrier")?;
    Ok(())
}

struct MbarrierBasicRecipe {
    id: &'static str,
    abi_id: &'static str,
    operation_key: &'static str,
    rust_arguments: &'static [&'static str],
    rust_result: &'static str,
    must_use: bool,
    adapter: MbarrierBasicAdapter,
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    dialect_operands: &'static [&'static str],
    dialect_results: &'static [&'static str],
    source_record: &'static str,
    llvm_symbol: &'static str,
    llvm_arguments: &'static [&'static str],
    llvm_results: &'static [&'static str],
    memory: &'static str,
    ptx_result: &'static str,
    selection: &'static str,
    selection_asm: &'static str,
    ptx_modifier: &'static str,
    ptx_isa_section: &'static str,
    ptx_isa_url: &'static str,
    llvm_nvptx_mechanism: BackendLoweringMechanism,
    lib_nvvm_mechanism: BackendLoweringMechanism,
    properties: &'static [&'static str],
    summary: &'static str,
}

fn mbarrier_basic_recipe(operation: MbarrierBasicOperation) -> MbarrierBasicRecipe {
    match operation {
        MbarrierBasicOperation::Init => MbarrierBasicRecipe {
            id: "mbarrier_init",
            abi_id: "i0097",
            operation_key: "barrier.mbarrier.init.shared.cta",
            rust_arguments: &["*mut u64", "u32"],
            rust_result: "()",
            must_use: false,
            adapter: MbarrierBasicAdapter::PointerCountToVoid,
            dialect_op_type: "MbarrierInitSharedOp",
            dialect_op_name: "nvvm.mbarrier_init_shared",
            dialect_operands: &["ptr", "i32"],
            dialect_results: &[],
            source_record: "int_nvvm_mbarrier_init_shared",
            llvm_symbol: "llvm.nvvm.mbarrier.init.shared",
            llvm_arguments: &["shared_ptr", "i32"],
            llvm_results: &[],
            memory: "read_write",
            ptx_result: "()",
            selection: "MBARRIER_INIT_SHARED",
            selection_asm: "mbarrier.init.shared.b64 \t[$addr], $count;",
            ptx_modifier: "init",
            ptx_isa_section: "9.7.14.16.12 Parallel Synchronization and Communication Instructions: mbarrier.init",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-init",
            llvm_nvptx_mechanism: BackendLoweringMechanism::TypedNvvm,
            lib_nvvm_mechanism: BackendLoweringMechanism::InlinePtx,
            properties: &["IntrConvergent", "IntrNoCallback"],
            summary: "Initializes a CTA shared-memory barrier with the expected arrival count.",
        },
        MbarrierBasicOperation::Arrive => MbarrierBasicRecipe {
            id: "mbarrier_arrive",
            abi_id: "i0098",
            operation_key: "barrier.mbarrier.arrive.shared.cta",
            rust_arguments: &["*const u64"],
            rust_result: "u64",
            must_use: true,
            adapter: MbarrierBasicAdapter::PointerToToken,
            dialect_op_type: "MbarrierArriveSharedOp",
            dialect_op_name: "nvvm.mbarrier_arrive_shared",
            dialect_operands: &["ptr"],
            dialect_results: &["i64"],
            source_record: "int_nvvm_mbarrier_arrive_shared",
            llvm_symbol: "llvm.nvvm.mbarrier.arrive.shared",
            llvm_arguments: &["shared_ptr"],
            llvm_results: &["i64"],
            memory: "read_write",
            ptx_result: "u64",
            selection: "MBARRIER_ARRIVE_SHARED",
            selection_asm: "mbarrier.arrive.shared.b64 \t$state, [$addr];",
            ptx_modifier: "arrive",
            ptx_isa_section: "9.7.14.16.16 Parallel Synchronization and Communication Instructions: mbarrier.arrive",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive",
            llvm_nvptx_mechanism: BackendLoweringMechanism::TypedNvvm,
            lib_nvvm_mechanism: BackendLoweringMechanism::InlinePtx,
            properties: &["IntrConvergent", "IntrNoCallback"],
            summary: "Arrives at a CTA shared-memory barrier and returns its phase token.",
        },
        MbarrierBasicOperation::TestWait => MbarrierBasicRecipe {
            id: "mbarrier_test_wait",
            abi_id: "i0099",
            operation_key: "barrier.mbarrier.test_wait.shared.cta",
            rust_arguments: &["*const u64", "u64"],
            rust_result: "bool",
            must_use: true,
            adapter: MbarrierBasicAdapter::PointerTokenToPredicate,
            dialect_op_type: "MbarrierTestWaitSharedOp",
            dialect_op_name: "nvvm.mbarrier_test_wait_shared",
            dialect_operands: &["ptr", "i64"],
            dialect_results: &["i1"],
            source_record: "int_nvvm_mbarrier_test_wait_shared",
            llvm_symbol: "llvm.nvvm.mbarrier.test.wait.shared",
            llvm_arguments: &["shared_ptr", "i64"],
            llvm_results: &["i1"],
            memory: "read_write",
            ptx_result: "bool",
            selection: "MBARRIER_TEST_WAIT_SHARED",
            selection_asm: "mbarrier.test_wait.shared.b64 \t$res, [$addr], $state;",
            ptx_modifier: "test_wait",
            ptx_isa_section: "9.7.14.16.19 Parallel Synchronization and Communication Instructions: mbarrier.test_wait / mbarrier.try_wait",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait",
            llvm_nvptx_mechanism: BackendLoweringMechanism::InlinePtx,
            lib_nvvm_mechanism: BackendLoweringMechanism::InlinePtx,
            properties: &["IntrConvergent", "IntrNoCallback"],
            summary: "Tests whether the CTA shared-memory barrier phase for a token is complete.",
        },
        MbarrierBasicOperation::Inval => MbarrierBasicRecipe {
            id: "mbarrier_inval",
            abi_id: "i0100",
            operation_key: "barrier.mbarrier.inval.shared.cta",
            rust_arguments: &["*mut u64"],
            rust_result: "()",
            must_use: false,
            adapter: MbarrierBasicAdapter::PointerToVoid,
            dialect_op_type: "MbarrierInvalSharedOp",
            dialect_op_name: "nvvm.mbarrier_inval_shared",
            dialect_operands: &["ptr"],
            dialect_results: &[],
            source_record: "int_nvvm_mbarrier_inval_shared",
            llvm_symbol: "llvm.nvvm.mbarrier.inval.shared",
            llvm_arguments: &["shared_ptr"],
            llvm_results: &[],
            memory: "write",
            ptx_result: "()",
            selection: "MBARRIER_INVAL_SHARED",
            selection_asm: "mbarrier.inval.shared.b64 \t[$addr];",
            ptx_modifier: "inval",
            ptx_isa_section: "9.7.14.16.13 Parallel Synchronization and Communication Instructions: mbarrier.inval",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-inval",
            llvm_nvptx_mechanism: BackendLoweringMechanism::TypedNvvm,
            lib_nvvm_mechanism: BackendLoweringMechanism::InlinePtx,
            properties: &[
                "IntrArgMemOnly",
                "IntrConvergent",
                "IntrNoCallback",
                "IntrWriteMem",
                "NoCapture<arg0>",
                "WriteOnly<arg0>",
            ],
            summary: "Invalidates a CTA shared-memory barrier after its users have finished.",
        },
    }
}

fn mbarrier_expected_operands(operation: MbarrierBasicOperation) -> Vec<OperandPattern> {
    match operation {
        MbarrierBasicOperation::Init => {
            vec![OperandPattern::Address, OperandPattern::RegisterOrImmediate]
        }
        MbarrierBasicOperation::Arrive => {
            vec![OperandPattern::Register, OperandPattern::Address]
        }
        MbarrierBasicOperation::TestWait => vec![
            OperandPattern::Register,
            OperandPattern::Address,
            OperandPattern::Register,
        ],
        MbarrierBasicOperation::Inval => vec![OperandPattern::Address],
    }
}

fn validate_mbarrier_basic_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let mbarrier = policy
        .mbarrier_basic
        .as_ref()
        .with_context(|| format!("{} has no closed basic mbarrier contract", policy.id))?;
    let recipe = mbarrier_basic_recipe(mbarrier.operation);
    ensure!(
        mbarrier.state_space == MbarrierStateSpace::Shared && mbarrier.adapter == recipe.adapter,
        "{} mbarrier operation, state space, and adapter disagree",
        policy.id
    );
    ensure!(
        mbarrier.runtime_validation == RuntimeValidation::Unexecuted,
        "{} cannot claim unrecorded mbarrier runtime validation",
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
        "{} mbarrier identity does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "barrier"
            && policy.rust_name == recipe.id
            && policy
                .rust_arguments
                .iter()
                .map(String::as_str)
                .eq(recipe.rust_arguments.iter().copied())
            && policy.rust_result == recipe.rust_result
            && !policy.safe
            && policy.must_use == recipe.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.public_rust_path == format!("cuda_intrinsics::barrier::{}", recipe.id)
            && policy.compatibility_rust_paths == [format!("cuda_device::barrier::{}", recipe.id)],
        "{} must preserve its unsafe mbarrier raw and compatibility API",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy
                .dialect_operands
                .iter()
                .map(String::as_str)
                .eq(recipe.dialect_operands.iter().copied())
            && policy
                .dialect_results
                .iter()
                .map(String::as_str)
                .eq(recipe.dialect_results.iter().copied())
            && policy
                .llvm_arguments
                .iter()
                .map(String::as_str)
                .eq(recipe.llvm_arguments.iter().copied())
            && policy
                .llvm_results
                .iter()
                .map(String::as_str)
                .eq(recipe.llvm_results.iter().copied())
            && policy.lowering == "generated_mbarrier_basic",
        "{} is outside the closed mbarrier signature and lowering recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == recipe.memory
            && policy.convergent
            && policy.execution_scope == "cta"
            && policy.minimum_ptx == "7.0"
            && policy.minimum_sm.as_deref() == Some("sm_80")
            && policy.ptx_result == recipe.ptx_result
            && policy.targets == "all",
        "{} mbarrier effects or target floor disagree with the closed recipe",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == recipe.ptx_isa_section
            && policy.ptx_isa_url == recipe.ptx_isa_url,
        "{} mbarrier PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        policy.summary == recipe.summary,
        "{} mbarrier summary does not match its closed recipe",
        policy.id
    );
    ensure!(
        declaration
            .properties
            .iter()
            .map(String::as_str)
            .eq(recipe.properties.iter().copied()),
        "{} mbarrier properties disagree with the imported LLVM declaration",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "mbarrier"
            && policy.expected_ptx.modifiers == [recipe.ptx_modifier, "shared", "b64"]
            && policy.expected_ptx.operands == mbarrier_expected_operands(mbarrier.operation),
        "{} expected PTX does not match its mbarrier recipe",
        policy.id
    );
    ensure!(
        declaration.selections.len() == 1
            && declaration.selections[0].source_record == recipe.selection
            && declaration.selections[0].asm == recipe.selection_asm
            && declaration.selections[0].predicates
                == [
                    "Subtarget->getSmVersion() >= 80",
                    "Subtarget->getPTXVersion() >= 70",
                ]
            && declaration.selections[0].constraints.is_empty(),
        "{} imported mbarrier selection changed",
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
                    (IntrinsicBackend::LlvmNvptx, recipe.llvm_nvptx_mechanism),
                    (IntrinsicBackend::LibNvvm, recipe.lib_nvvm_mechanism),
                ])
            && policy.backend_lowerings.iter().all(|lowering| {
                lowering.minimum_ptx.is_none()
                    && lowering.minimum_sm.is_none()
                    && !lowering.evidence_profile.trim().is_empty()
            }),
        "{} must define exactly the reviewed mbarrier backend routes",
        policy.id
    );
    ensure_no_other_family_contract(policy, "mbarrier_basic")?;
    Ok(())
}

fn packed_conversion_recipe(
    conversion: &crate::model::PackedConversion,
) -> Option<PackedConversionRecipe> {
    match (
        conversion.destination_format,
        conversion.rounding,
        conversion.saturation,
    ) {
        (
            PackedConversionDestinationFormat::Bf16x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::None,
        ) => Some(PackedConversionRecipe {
            id: "cvt_f32x2_bf16x2",
            abi_id: "i0071",
            operation_key: "packed.convert.f32x2.bf16x2.nearest_even",
            rust_name: "cvt_f32x2_bf16x2",
            compatibility_path: "cuda_device::tcgen05::cvt_f32x2_bf16x2",
            dialect_op_type: "CvtF32x2Bf16x2Op",
            dialect_op_name: "nvvm.cvt_f32x2_bf16x2",
            source_record: "int_nvvm_ff2bf16x2_rn",
            llvm_symbol: "llvm.nvvm.ff2bf16x2.rn",
            llvm_result: "v2bf16",
            summary: "Converts two f32 values to packed bf16x2 with the first argument in the low half.",
        }),
        (
            PackedConversionDestinationFormat::F16x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::None,
        ) => Some(PackedConversionRecipe {
            id: "cvt_f16x2_f32",
            abi_id: "i0081",
            operation_key: "packed.convert.f32x2.f16x2.nearest_even",
            rust_name: "cvt_f16x2_f32",
            compatibility_path: "cuda_device::convert::cvt_f16x2_f32",
            dialect_op_type: "CvtF16x2F32Op",
            dialect_op_name: "nvvm.cvt_f16x2_f32",
            source_record: "int_nvvm_ff2f16x2_rn",
            llvm_symbol: "llvm.nvvm.ff2f16x2.rn",
            llvm_result: "v2f16",
            summary: "Converts two f32 values to packed f16x2 with nearest-even rounding and the first argument in the low half.",
        }),
        (
            PackedConversionDestinationFormat::F16x2,
            PackedConversionRounding::TowardZero,
            PackedConversionSaturation::None,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rz_f16x2_f32",
            abi_id: "i0082",
            operation_key: "packed.convert.f32x2.f16x2.toward_zero",
            rust_name: "cvt_rz_f16x2_f32",
            compatibility_path: "cuda_device::convert::cvt_rz_f16x2_f32",
            dialect_op_type: "CvtRzF16x2F32Op",
            dialect_op_name: "nvvm.cvt_rz_f16x2_f32",
            source_record: "int_nvvm_ff2f16x2_rz",
            llvm_symbol: "llvm.nvvm.ff2f16x2.rz",
            llvm_result: "v2f16",
            summary: "Converts two f32 values to packed f16x2 with toward-zero rounding and the first argument in the low half.",
        }),
        (
            PackedConversionDestinationFormat::F16x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::Relu,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rn_relu_f16x2_f32",
            abi_id: "i0083",
            operation_key: "packed.convert.f32x2.f16x2.nearest_even.relu",
            rust_name: "cvt_rn_relu_f16x2_f32",
            compatibility_path: "cuda_device::convert::cvt_rn_relu_f16x2_f32",
            dialect_op_type: "CvtRnReluF16x2F32Op",
            dialect_op_name: "nvvm.cvt_rn_relu_f16x2_f32",
            source_record: "int_nvvm_ff2f16x2_rn_relu",
            llvm_symbol: "llvm.nvvm.ff2f16x2.rn.relu",
            llvm_result: "v2f16",
            summary: "Converts two f32 values to packed f16x2 with nearest-even rounding, ReLU, and the first argument in the low half.",
        }),
        (
            PackedConversionDestinationFormat::Bf16x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::Relu,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rn_relu_bf16x2_f32",
            abi_id: "i0084",
            operation_key: "packed.convert.f32x2.bf16x2.nearest_even.relu",
            rust_name: "cvt_rn_relu_bf16x2_f32",
            compatibility_path: "cuda_device::convert::cvt_rn_relu_bf16x2_f32",
            dialect_op_type: "CvtRnReluBf16x2F32Op",
            dialect_op_name: "nvvm.cvt_rn_relu_bf16x2_f32",
            source_record: "int_nvvm_ff2bf16x2_rn_relu",
            llvm_symbol: "llvm.nvvm.ff2bf16x2.rn.relu",
            llvm_result: "v2bf16",
            summary: "Converts two f32 values to packed bf16x2 with nearest-even rounding, ReLU, and the first argument in the low half.",
        }),
        (
            PackedConversionDestinationFormat::Bf16x2,
            PackedConversionRounding::TowardZero,
            PackedConversionSaturation::None,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rz_bf16x2_f32",
            abi_id: "i0085",
            operation_key: "packed.convert.f32x2.bf16x2.toward_zero",
            rust_name: "cvt_rz_bf16x2_f32",
            compatibility_path: "cuda_device::convert::cvt_rz_bf16x2_f32",
            dialect_op_type: "CvtRzBf16x2F32Op",
            dialect_op_name: "nvvm.cvt_rz_bf16x2_f32",
            source_record: "int_nvvm_ff2bf16x2_rz",
            llvm_symbol: "llvm.nvvm.ff2bf16x2.rz",
            llvm_result: "v2bf16",
            summary: "Converts two f32 values to packed bf16x2 with toward-zero rounding and the first argument in the low half.",
        }),
        (
            PackedConversionDestinationFormat::E4m3x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::Satfinite,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rn_satfinite_e4m3x2_f32",
            abi_id: "i0259",
            operation_key: "packed.convert.f32x2.e4m3x2.nearest_even.satfinite",
            rust_name: "cvt_rn_satfinite_e4m3x2_f32",
            compatibility_path: "cuda_device::convert::cvt_rn_satfinite_e4m3x2_f32",
            dialect_op_type: "CvtRnSatfiniteE4m3x2F32Op",
            dialect_op_name: "nvvm.cvt_rn_satfinite_e4m3x2_f32",
            source_record: "int_nvvm_ff_to_e4m3x2_rn",
            llvm_symbol: "llvm.nvvm.ff.to.e4m3x2.rn",
            llvm_result: "i16",
            summary: "Converts two f32 values to packed e4m3x2 with nearest-even finite saturation and the first argument in the low byte.",
        }),
        (
            PackedConversionDestinationFormat::E4m3x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::SatfiniteRelu,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rn_satfinite_relu_e4m3x2_f32",
            abi_id: "i0260",
            operation_key: "packed.convert.f32x2.e4m3x2.nearest_even.satfinite.relu",
            rust_name: "cvt_rn_satfinite_relu_e4m3x2_f32",
            compatibility_path: "cuda_device::convert::cvt_rn_satfinite_relu_e4m3x2_f32",
            dialect_op_type: "CvtRnSatfiniteReluE4m3x2F32Op",
            dialect_op_name: "nvvm.cvt_rn_satfinite_relu_e4m3x2_f32",
            source_record: "int_nvvm_ff_to_e4m3x2_rn_relu",
            llvm_symbol: "llvm.nvvm.ff.to.e4m3x2.rn.relu",
            llvm_result: "i16",
            summary: "Converts two f32 values to packed e4m3x2 with nearest-even finite saturation, ReLU, and the first argument in the low byte.",
        }),
        (
            PackedConversionDestinationFormat::E5m2x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::Satfinite,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rn_satfinite_e5m2x2_f32",
            abi_id: "i0261",
            operation_key: "packed.convert.f32x2.e5m2x2.nearest_even.satfinite",
            rust_name: "cvt_rn_satfinite_e5m2x2_f32",
            compatibility_path: "cuda_device::convert::cvt_rn_satfinite_e5m2x2_f32",
            dialect_op_type: "CvtRnSatfiniteE5m2x2F32Op",
            dialect_op_name: "nvvm.cvt_rn_satfinite_e5m2x2_f32",
            source_record: "int_nvvm_ff_to_e5m2x2_rn",
            llvm_symbol: "llvm.nvvm.ff.to.e5m2x2.rn",
            llvm_result: "i16",
            summary: "Converts two f32 values to packed e5m2x2 with nearest-even finite saturation and the first argument in the low byte.",
        }),
        (
            PackedConversionDestinationFormat::E5m2x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::SatfiniteRelu,
        ) => Some(PackedConversionRecipe {
            id: "cvt_rn_satfinite_relu_e5m2x2_f32",
            abi_id: "i0262",
            operation_key: "packed.convert.f32x2.e5m2x2.nearest_even.satfinite.relu",
            rust_name: "cvt_rn_satfinite_relu_e5m2x2_f32",
            compatibility_path: "cuda_device::convert::cvt_rn_satfinite_relu_e5m2x2_f32",
            dialect_op_type: "CvtRnSatfiniteReluE5m2x2F32Op",
            dialect_op_name: "nvvm.cvt_rn_satfinite_relu_e5m2x2_f32",
            source_record: "int_nvvm_ff_to_e5m2x2_rn_relu",
            llvm_symbol: "llvm.nvvm.ff.to.e5m2x2.rn.relu",
            llvm_result: "i16",
            summary: "Converts two f32 values to packed e5m2x2 with nearest-even finite saturation, ReLU, and the first argument in the low byte.",
        }),
        _ => None,
    }
}

fn packed_conversion_ptx_modifiers(
    conversion: &crate::model::PackedConversion,
) -> Vec<&'static str> {
    let rounding = match conversion.rounding {
        PackedConversionRounding::NearestEven => "rn",
        PackedConversionRounding::TowardZero => "rz",
    };
    let format = match conversion.destination_format {
        PackedConversionDestinationFormat::Bf16x2 => "bf16x2",
        PackedConversionDestinationFormat::E4m3x2 => "e4m3x2",
        PackedConversionDestinationFormat::E5m2x2 => "e5m2x2",
        PackedConversionDestinationFormat::F16x2 => "f16x2",
    };
    let mut modifiers = vec![rounding];
    match conversion.saturation {
        PackedConversionSaturation::None => {}
        PackedConversionSaturation::Relu => modifiers.push("relu"),
        PackedConversionSaturation::Satfinite => modifiers.push("satfinite"),
        PackedConversionSaturation::SatfiniteRelu => modifiers.extend(["satfinite", "relu"]),
    }
    modifiers.extend([format, "f32"]);
    modifiers
}

fn packed_conversion_result_width(conversion: &crate::model::PackedConversion) -> u32 {
    match conversion.destination_format {
        PackedConversionDestinationFormat::Bf16x2 | PackedConversionDestinationFormat::F16x2 => 32,
        PackedConversionDestinationFormat::E4m3x2 | PackedConversionDestinationFormat::E5m2x2 => 16,
    }
}

fn packed_conversion_floor(
    conversion: &crate::model::PackedConversion,
) -> (&'static str, &'static str) {
    match conversion.destination_format {
        PackedConversionDestinationFormat::Bf16x2 | PackedConversionDestinationFormat::F16x2 => {
            ("7.0", "sm_80")
        }
        PackedConversionDestinationFormat::E4m3x2 | PackedConversionDestinationFormat::E5m2x2 => {
            ("8.1", "sm_89")
        }
    }
}

fn packed_conversion_backend_mechanism(
    conversion: &crate::model::PackedConversion,
    backend: IntrinsicBackend,
) -> BackendLoweringMechanism {
    match (conversion.destination_format, backend) {
        (
            PackedConversionDestinationFormat::E4m3x2 | PackedConversionDestinationFormat::E5m2x2,
            IntrinsicBackend::LlvmNvptx,
        ) => BackendLoweringMechanism::TypedNvvm,
        _ => BackendLoweringMechanism::InlinePtx,
    }
}

fn packed_conversion_lowering(conversion: &crate::model::PackedConversion) -> &'static str {
    match conversion.destination_format {
        PackedConversionDestinationFormat::Bf16x2 | PackedConversionDestinationFormat::F16x2 => {
            "generated_packed_conversion_inline_ptx"
        }
        PackedConversionDestinationFormat::E4m3x2 | PackedConversionDestinationFormat::E5m2x2 => {
            "generated_packed_conversion_backend"
        }
    }
}

fn expand_packed_conversion_fp8_admission(
    admission: &PackedConversionFp8Admission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "FP8 conversion runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        admission.destination_formats
            == [
                PackedConversionDestinationFormat::E4m3x2,
                PackedConversionDestinationFormat::E5m2x2,
            ],
        "compact FP8 conversion admission must list the canonical two formats"
    );
    ensure!(
        admission.saturations
            == [
                PackedConversionSaturation::Satfinite,
                PackedConversionSaturation::SatfiniteRelu,
            ],
        "compact FP8 conversion admission must list base and ReLU finite saturation"
    );
    ensure!(
        admission.product_count
            == admission
                .destination_formats
                .len()
                .checked_mul(admission.saturations.len())
                .context("compact FP8 conversion product count overflow")?
            && admission.product_count == 4,
        "compact FP8 conversion product_count must be exactly 4"
    );

    let mut records = Vec::with_capacity(admission.product_count);
    for &destination_format in &admission.destination_formats {
        for &saturation in &admission.saturations {
            let conversion = crate::model::PackedConversion {
                source_format: PackedConversionSourceFormat::F32x2,
                destination_format,
                rounding: PackedConversionRounding::NearestEven,
                saturation,
                adapter: PackedConversionAdapter::ReverseHighLowOperands,
            };
            records.push(packed_conversion_overlay_record(
                conversion,
                &admission.llvm_evidence_profile,
                &admission.libnvvm_evidence_profile,
            )?);
        }
    }
    ensure!(records.len() == admission.product_count);
    Ok(records)
}

fn packed_conversion_overlay_record(
    conversion: crate::model::PackedConversion,
    llvm_evidence_profile: &str,
    libnvvm_evidence_profile: &str,
) -> Result<OverlayIntrinsic> {
    let recipe = packed_conversion_recipe(&conversion)
        .context("compact FP8 conversion is outside the closed recipe set")?;
    let result_width = packed_conversion_result_width(&conversion);
    let rust_result = format!("u{result_width}");
    let dialect_result = format!("i{result_width}");
    let (minimum_ptx, minimum_sm) = packed_conversion_floor(&conversion);
    Ok(OverlayIntrinsic {
        id: recipe.id.into(),
        abi_id: String::new(),
        operation_key: recipe.operation_key.into(),
        family: "packed_conversion".into(),
        source: None,
        source_record: Some(recipe.source_record.into()),
        rust_module: "convert".into(),
        rust_name: recipe.rust_name.into(),
        rust_arguments: vec!["f32".into(), "f32".into()],
        rust_result: rust_result.clone(),
        safe: true,
        must_use: false,
        safe_allowlist_reason: Some("This conversion has no caller obligations.".into()),
        public_rust_path: format!("cuda_intrinsics::convert::{}", recipe.rust_name),
        compatibility_rust_paths: vec![recipe.compatibility_path.into()],
        dialect_op_type: recipe.dialect_op_type.into(),
        dialect_op_name: recipe.dialect_op_name.into(),
        dialect_operands: vec!["f32".into(), "f32".into()],
        dialect_results: vec![dialect_result],
        llvm_symbol: Some(recipe.llvm_symbol.into()),
        resolved_llvm_symbol: None,
        llvm_arguments: vec!["f32".into(), "f32".into()],
        llvm_results: vec![recipe.llvm_result.into()],
        pure: true,
        memory: "none".into(),
        convergent: false,
        execution_scope: "thread".into(),
        minimum_ptx: minimum_ptx.into(),
        minimum_sm: Some(minimum_sm.into()),
        ptx_result: rust_result,
        targets: "all".into(),
        ptx_isa_version: "9.3".into(),
        ptx_isa_section: "9.7.9.22 Data Movement and Conversion Instructions: cvt".into(),
        ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt".into(),
        lowering: packed_conversion_lowering(&conversion).into(),
        backend_lowerings: [
            (IntrinsicBackend::LlvmNvptx, llvm_evidence_profile),
            (IntrinsicBackend::LibNvvm, libnvvm_evidence_profile),
        ]
        .into_iter()
        .map(|(backend, evidence_profile)| OverlayBackendLowering {
            backend,
            mechanism: packed_conversion_backend_mechanism(&conversion, backend),
            evidence_profile: evidence_profile.into(),
            minimum_ptx: Some(minimum_ptx.into()),
            minimum_sm: Some(minimum_sm.into()),
        })
        .collect(),
        packed_atomic: None,
        redux: None,
        vote: None,
        active_mask: None,
        warp_match: None,
        warp_barrier: None,
        warp_shuffle: None,
        dot_product: None,
        packed_alu: None,
        packed_conversion: Some(conversion.clone()),
        cp_async_copy: None,
        cp_async_control: None,
        cp_async_mbarrier: None,
        mbarrier_basic: None,
        movmatrix: None,
        mbarrier_extended: None,
        register_mma: None,
        sparse_mma: None,
        prmt: None,
        cluster_barrier: None,
        special_register: None,
        debug_control: None,
        cluster_memory: None,
        clc: None,
        ldmatrix_variant: None,
        ldmatrix_safety: None,
        ldmatrix_adapter: None,
        selected_address_space: None,
        expected_ptx: InstructionPattern {
            mnemonic: "cvt".into(),
            modifiers: packed_conversion_ptx_modifiers(&conversion)
                .into_iter()
                .map(str::to_owned)
                .collect(),
            operands: vec![OperandPattern::Register; 3],
        },
        summary: recipe.summary.into(),
    })
}

fn validate_packed_conversion_policy(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
    declaration: Option<&ImportedIntrinsic>,
) -> Result<()> {
    let conversion = policy
        .packed_conversion
        .as_ref()
        .with_context(|| format!("{} has no closed packed-conversion contract", policy.id))?;
    ensure!(
        conversion.source_format == PackedConversionSourceFormat::F32x2
            && conversion.adapter == PackedConversionAdapter::ReverseHighLowOperands,
        "{} requests an unsupported packed-conversion source or adapter",
        policy.id
    );
    let recipe = packed_conversion_recipe(conversion).with_context(|| {
        format!(
            "{} requests an unsupported packed-conversion destination, rounding, or saturation combination",
            policy.id
        )
    })?;
    let result_width = packed_conversion_result_width(conversion);
    let rust_result = format!("u{result_width}");
    let dialect_result = format!("i{result_width}");
    let (minimum_ptx, minimum_sm) = packed_conversion_floor(conversion);
    ensure!(
        policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && source
                == &IntrinsicSource::LlvmImported {
                    source_record: recipe.source_record.into(),
                }
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none()
            && policy.llvm_arguments == ["f32", "f32"]
            && policy.llvm_results == [recipe.llvm_result],
        "{} packed-conversion identity or LLVM source changed",
        policy.id
    );
    let declaration = declaration.context("packed conversion has no imported declaration")?;
    ensure!(
        declaration.properties == ["IntrNoMem", "IntrSpeculatable"]
            && declaration.selections.is_empty(),
        "{} selectionless packed-conversion declaration changed",
        policy.id
    );
    ensure!(
        policy.rust_module == "convert"
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == ["f32", "f32"]
            && policy.rust_result == rust_result
            && policy.safe
            && !policy.must_use
            && policy
                .safe_allowlist_reason
                .as_deref()
                .is_some_and(|reason| !reason.trim().is_empty())
            && policy.public_rust_path == format!("cuda_intrinsics::convert::{}", recipe.rust_name)
            && policy.compatibility_rust_paths == [recipe.compatibility_path],
        "{} must preserve its safe non-must-use conversion API",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == ["f32", "f32"]
            && policy.dialect_results == [dialect_result.as_str()]
            && policy.lowering == packed_conversion_lowering(conversion),
        "{} is outside the closed packed-conversion dialect and lowering recipe",
        policy.id
    );
    ensure!(
        policy.pure
            && policy.memory == "none"
            && !policy.convergent
            && policy.execution_scope == "thread"
            && policy.minimum_ptx == minimum_ptx
            && policy.minimum_sm.as_deref() == Some(minimum_sm)
            && policy.ptx_result == rust_result
            && policy.targets == "all"
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == "9.7.9.22 Data Movement and Conversion Instructions: cvt"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt",
        "{} packed-conversion effects, carrier, target floor, or PTX provenance disagree",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "cvt"
            && policy.expected_ptx.modifiers == packed_conversion_ptx_modifiers(conversion)
            && policy.expected_ptx.operands
                == [
                    OperandPattern::Register,
                    OperandPattern::Register,
                    OperandPattern::Register,
                ],
        "{} expected PTX does not match the reversed high/low conversion recipe",
        policy.id
    );
    ensure!(
        policy.summary == recipe.summary,
        "{} packed-conversion summary does not match its closed recipe",
        policy.id
    );
    let backend_pairs = policy
        .backend_lowerings
        .iter()
        .map(|lowering| (lowering.backend, lowering.mechanism))
        .collect::<BTreeSet<_>>();
    let expected_pairs = [IntrinsicBackend::LlvmNvptx, IntrinsicBackend::LibNvvm]
        .map(|backend| {
            (
                backend,
                packed_conversion_backend_mechanism(conversion, backend),
            )
        })
        .into_iter()
        .collect::<BTreeSet<_>>();
    ensure!(
        policy.backend_lowerings.len() == 2 && backend_pairs == expected_pairs,
        "{} must define exactly the reviewed packed-conversion backend routes",
        policy.id
    );
    for lowering in &policy.backend_lowerings {
        ensure!(
            lowering.minimum_ptx.as_deref() == Some(minimum_ptx)
                && lowering.minimum_sm.as_deref() == Some(minimum_sm)
                && !lowering.evidence_profile.trim().is_empty(),
            "{} backend {:?} does not carry its exact packed-conversion floor",
            policy.id,
            lowering.backend
        );
    }
    ensure_no_other_family_contract(policy, "packed conversion")?;
    Ok(())
}

struct RegisterMmaRecipe {
    id: &'static str,
    abi_id: &'static str,
    operation_key: &'static str,
    source_record: &'static str,
    llvm_symbol: &'static str,
    rust_arguments: &'static [&'static str],
    rust_result: &'static str,
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    dialect_operands: &'static [&'static str],
    dialect_results: &'static [&'static str],
    llvm_arguments: &'static [&'static str],
    llvm_results: &'static [&'static str],
    adapter: RegisterMmaAdapter,
    compatibility_source: RegisterMmaCompatibilitySource,
    minimum_ptx: &'static str,
    minimum_sm: &'static str,
    ptx_modifiers: Vec<&'static str>,
    ptx_register_counts: [usize; 4],
}

fn integer_register_mma_recipe(mma: &RegisterMma, common: bool) -> Option<RegisterMmaRecipe> {
    use RegisterMmaAdapter::{
        C2I32A1U32B1U32ToD2I32, C4I32A2U32B1U32ToD4I32, C4I32A4U32B2U32ToD4I32,
    };
    use RegisterMmaCompatibilitySource::{ExistingStub, GeneratedStub};
    use RegisterMmaElement::{S4, S8, U4, U8};
    use RegisterMmaOverflow::{Satfinite, Wrapping};
    use RegisterMmaShape::{M8n8k16, M8n8k32, M16n8k16, M16n8k32, M16n8k64};

    if !common
        || mma.operation != RegisterMmaOperation::Multiply
        || mma.accumulator != RegisterMmaAccumulator::S32
    {
        return None;
    }

    let (id, abi_id, operation_key, source_record, llvm_symbol, compatibility_source) =
        match (mma.shape, mma.a_element, mma.b_element, mma.overflow) {
            (M16n8k32, S8, S8, Wrapping) => (
                "mma_m16n8k32_s32_s8",
                "i0108",
                "matrix.mma.m16n8k32.row.col.s32.s8.s8.s32.wrapping",
                "int_nvvm_mma_m16n8k32_row_col_s8",
                "llvm.nvvm.mma.m16n8k32.row.col.s8",
                ExistingStub,
            ),
            (M16n8k16, S8, S8, Wrapping) => (
                "mma_m16n8k16_s32_s8",
                "i0110",
                "matrix.mma.m16n8k16.row.col.s32.s8.s8.s32.wrapping",
                "int_nvvm_mma_m16n8k16_row_col_s8",
                "llvm.nvvm.mma.m16n8k16.row.col.s8",
                GeneratedStub,
            ),
            (M16n8k16, S8, U8, Wrapping) => (
                "mma_m16n8k16_s32_s8_u8",
                "i0111",
                "matrix.mma.m16n8k16.row.col.s32.s8.u8.s32.wrapping",
                "int_nvvm_mma_m16n8k16_row_col_s8_u8",
                "llvm.nvvm.mma.m16n8k16.row.col.s8.u8",
                GeneratedStub,
            ),
            (M16n8k16, U8, U8, Wrapping) => (
                "mma_m16n8k16_s32_u8",
                "i0112",
                "matrix.mma.m16n8k16.row.col.s32.u8.u8.s32.wrapping",
                "int_nvvm_mma_m16n8k16_row_col_u8",
                "llvm.nvvm.mma.m16n8k16.row.col.u8",
                GeneratedStub,
            ),
            (M16n8k16, U8, S8, Wrapping) => (
                "mma_m16n8k16_s32_u8_s8",
                "i0113",
                "matrix.mma.m16n8k16.row.col.s32.u8.s8.s32.wrapping",
                "int_nvvm_mma_m16n8k16_row_col_u8_s8",
                "llvm.nvvm.mma.m16n8k16.row.col.u8.s8",
                GeneratedStub,
            ),
            (M16n8k32, S8, U8, Wrapping) => (
                "mma_m16n8k32_s32_s8_u8",
                "i0114",
                "matrix.mma.m16n8k32.row.col.s32.s8.u8.s32.wrapping",
                "int_nvvm_mma_m16n8k32_row_col_s8_u8",
                "llvm.nvvm.mma.m16n8k32.row.col.s8.u8",
                GeneratedStub,
            ),
            (M16n8k32, U8, U8, Wrapping) => (
                "mma_m16n8k32_s32_u8",
                "i0115",
                "matrix.mma.m16n8k32.row.col.s32.u8.u8.s32.wrapping",
                "int_nvvm_mma_m16n8k32_row_col_u8",
                "llvm.nvvm.mma.m16n8k32.row.col.u8",
                GeneratedStub,
            ),
            (M16n8k32, U8, S8, Wrapping) => (
                "mma_m16n8k32_s32_u8_s8",
                "i0116",
                "matrix.mma.m16n8k32.row.col.s32.u8.s8.s32.wrapping",
                "int_nvvm_mma_m16n8k32_row_col_u8_s8",
                "llvm.nvvm.mma.m16n8k32.row.col.u8.s8",
                GeneratedStub,
            ),
            (M16n8k16, S8, S8, Satfinite) => (
                "mma_m16n8k16_s32_s8_satfinite",
                "i0117",
                "matrix.mma.m16n8k16.row.col.s32.s8.s8.s32.satfinite",
                "int_nvvm_mma_m16n8k16_row_col_satfinite_s8",
                "llvm.nvvm.mma.m16n8k16.row.col.satfinite.s8",
                GeneratedStub,
            ),
            (M16n8k16, S8, U8, Satfinite) => (
                "mma_m16n8k16_s32_s8_u8_satfinite",
                "i0118",
                "matrix.mma.m16n8k16.row.col.s32.s8.u8.s32.satfinite",
                "int_nvvm_mma_m16n8k16_row_col_satfinite_s8_u8",
                "llvm.nvvm.mma.m16n8k16.row.col.satfinite.s8.u8",
                GeneratedStub,
            ),
            (M16n8k16, U8, U8, Satfinite) => (
                "mma_m16n8k16_s32_u8_satfinite",
                "i0119",
                "matrix.mma.m16n8k16.row.col.s32.u8.u8.s32.satfinite",
                "int_nvvm_mma_m16n8k16_row_col_satfinite_u8",
                "llvm.nvvm.mma.m16n8k16.row.col.satfinite.u8",
                GeneratedStub,
            ),
            (M16n8k16, U8, S8, Satfinite) => (
                "mma_m16n8k16_s32_u8_s8_satfinite",
                "i0120",
                "matrix.mma.m16n8k16.row.col.s32.u8.s8.s32.satfinite",
                "int_nvvm_mma_m16n8k16_row_col_satfinite_u8_s8",
                "llvm.nvvm.mma.m16n8k16.row.col.satfinite.u8.s8",
                GeneratedStub,
            ),
            (M16n8k32, S8, S8, Satfinite) => (
                "mma_m16n8k32_s32_s8_satfinite",
                "i0121",
                "matrix.mma.m16n8k32.row.col.s32.s8.s8.s32.satfinite",
                "int_nvvm_mma_m16n8k32_row_col_satfinite_s8",
                "llvm.nvvm.mma.m16n8k32.row.col.satfinite.s8",
                GeneratedStub,
            ),
            (M16n8k32, S8, U8, Satfinite) => (
                "mma_m16n8k32_s32_s8_u8_satfinite",
                "i0122",
                "matrix.mma.m16n8k32.row.col.s32.s8.u8.s32.satfinite",
                "int_nvvm_mma_m16n8k32_row_col_satfinite_s8_u8",
                "llvm.nvvm.mma.m16n8k32.row.col.satfinite.s8.u8",
                GeneratedStub,
            ),
            (M16n8k32, U8, U8, Satfinite) => (
                "mma_m16n8k32_s32_u8_satfinite",
                "i0123",
                "matrix.mma.m16n8k32.row.col.s32.u8.u8.s32.satfinite",
                "int_nvvm_mma_m16n8k32_row_col_satfinite_u8",
                "llvm.nvvm.mma.m16n8k32.row.col.satfinite.u8",
                GeneratedStub,
            ),
            (M16n8k32, U8, S8, Satfinite) => (
                "mma_m16n8k32_s32_u8_s8_satfinite",
                "i0124",
                "matrix.mma.m16n8k32.row.col.s32.u8.s8.s32.satfinite",
                "int_nvvm_mma_m16n8k32_row_col_satfinite_u8_s8",
                "llvm.nvvm.mma.m16n8k32.row.col.satfinite.u8.s8",
                GeneratedStub,
            ),
            (M8n8k16, S8, S8, Wrapping) => (
                "mma_m8n8k16_s32_s8",
                "i0125",
                "matrix.mma.m8n8k16.row.col.s32.s8.s8.s32.wrapping",
                "int_nvvm_mma_m8n8k16_row_col_s8",
                "llvm.nvvm.mma.m8n8k16.row.col.s8",
                GeneratedStub,
            ),
            (M8n8k16, S8, U8, Wrapping) => (
                "mma_m8n8k16_s32_s8_u8",
                "i0126",
                "matrix.mma.m8n8k16.row.col.s32.s8.u8.s32.wrapping",
                "int_nvvm_mma_m8n8k16_row_col_s8_u8",
                "llvm.nvvm.mma.m8n8k16.row.col.s8.u8",
                GeneratedStub,
            ),
            (M8n8k16, U8, U8, Wrapping) => (
                "mma_m8n8k16_s32_u8",
                "i0127",
                "matrix.mma.m8n8k16.row.col.s32.u8.u8.s32.wrapping",
                "int_nvvm_mma_m8n8k16_row_col_u8",
                "llvm.nvvm.mma.m8n8k16.row.col.u8",
                GeneratedStub,
            ),
            (M8n8k16, U8, S8, Wrapping) => (
                "mma_m8n8k16_s32_u8_s8",
                "i0128",
                "matrix.mma.m8n8k16.row.col.s32.u8.s8.s32.wrapping",
                "int_nvvm_mma_m8n8k16_row_col_u8_s8",
                "llvm.nvvm.mma.m8n8k16.row.col.u8.s8",
                GeneratedStub,
            ),
            (M8n8k16, S8, S8, Satfinite) => (
                "mma_m8n8k16_s32_s8_satfinite",
                "i0129",
                "matrix.mma.m8n8k16.row.col.s32.s8.s8.s32.satfinite",
                "int_nvvm_mma_m8n8k16_row_col_satfinite_s8",
                "llvm.nvvm.mma.m8n8k16.row.col.satfinite.s8",
                GeneratedStub,
            ),
            (M8n8k16, S8, U8, Satfinite) => (
                "mma_m8n8k16_s32_s8_u8_satfinite",
                "i0130",
                "matrix.mma.m8n8k16.row.col.s32.s8.u8.s32.satfinite",
                "int_nvvm_mma_m8n8k16_row_col_satfinite_s8_u8",
                "llvm.nvvm.mma.m8n8k16.row.col.satfinite.s8.u8",
                GeneratedStub,
            ),
            (M8n8k16, U8, U8, Satfinite) => (
                "mma_m8n8k16_s32_u8_satfinite",
                "i0131",
                "matrix.mma.m8n8k16.row.col.s32.u8.u8.s32.satfinite",
                "int_nvvm_mma_m8n8k16_row_col_satfinite_u8",
                "llvm.nvvm.mma.m8n8k16.row.col.satfinite.u8",
                GeneratedStub,
            ),
            (M8n8k16, U8, S8, Satfinite) => (
                "mma_m8n8k16_s32_u8_s8_satfinite",
                "i0132",
                "matrix.mma.m8n8k16.row.col.s32.u8.s8.s32.satfinite",
                "int_nvvm_mma_m8n8k16_row_col_satfinite_u8_s8",
                "llvm.nvvm.mma.m8n8k16.row.col.satfinite.u8.s8",
                GeneratedStub,
            ),
            (M8n8k32, S4, S4, Wrapping) => (
                "mma_m8n8k32_s32_s4",
                "i0133",
                "matrix.mma.m8n8k32.row.col.s32.s4.s4.s32.wrapping",
                "int_nvvm_mma_m8n8k32_row_col_s4",
                "llvm.nvvm.mma.m8n8k32.row.col.s4",
                GeneratedStub,
            ),
            (M8n8k32, S4, U4, Wrapping) => (
                "mma_m8n8k32_s32_s4_u4",
                "i0134",
                "matrix.mma.m8n8k32.row.col.s32.s4.u4.s32.wrapping",
                "int_nvvm_mma_m8n8k32_row_col_s4_u4",
                "llvm.nvvm.mma.m8n8k32.row.col.s4.u4",
                GeneratedStub,
            ),
            (M8n8k32, U4, U4, Wrapping) => (
                "mma_m8n8k32_s32_u4",
                "i0135",
                "matrix.mma.m8n8k32.row.col.s32.u4.u4.s32.wrapping",
                "int_nvvm_mma_m8n8k32_row_col_u4",
                "llvm.nvvm.mma.m8n8k32.row.col.u4",
                GeneratedStub,
            ),
            (M8n8k32, U4, S4, Wrapping) => (
                "mma_m8n8k32_s32_u4_s4",
                "i0136",
                "matrix.mma.m8n8k32.row.col.s32.u4.s4.s32.wrapping",
                "int_nvvm_mma_m8n8k32_row_col_u4_s4",
                "llvm.nvvm.mma.m8n8k32.row.col.u4.s4",
                GeneratedStub,
            ),
            (M8n8k32, S4, S4, Satfinite) => (
                "mma_m8n8k32_s32_s4_satfinite",
                "i0137",
                "matrix.mma.m8n8k32.row.col.s32.s4.s4.s32.satfinite",
                "int_nvvm_mma_m8n8k32_row_col_satfinite_s4",
                "llvm.nvvm.mma.m8n8k32.row.col.satfinite.s4",
                GeneratedStub,
            ),
            (M8n8k32, S4, U4, Satfinite) => (
                "mma_m8n8k32_s32_s4_u4_satfinite",
                "i0138",
                "matrix.mma.m8n8k32.row.col.s32.s4.u4.s32.satfinite",
                "int_nvvm_mma_m8n8k32_row_col_satfinite_s4_u4",
                "llvm.nvvm.mma.m8n8k32.row.col.satfinite.s4.u4",
                GeneratedStub,
            ),
            (M8n8k32, U4, U4, Satfinite) => (
                "mma_m8n8k32_s32_u4_satfinite",
                "i0139",
                "matrix.mma.m8n8k32.row.col.s32.u4.u4.s32.satfinite",
                "int_nvvm_mma_m8n8k32_row_col_satfinite_u4",
                "llvm.nvvm.mma.m8n8k32.row.col.satfinite.u4",
                GeneratedStub,
            ),
            (M8n8k32, U4, S4, Satfinite) => (
                "mma_m8n8k32_s32_u4_s4_satfinite",
                "i0140",
                "matrix.mma.m8n8k32.row.col.s32.u4.s4.s32.satfinite",
                "int_nvvm_mma_m8n8k32_row_col_satfinite_u4_s4",
                "llvm.nvvm.mma.m8n8k32.row.col.satfinite.u4.s4",
                GeneratedStub,
            ),
            (M16n8k32, S4, S4, Wrapping) => (
                "mma_m16n8k32_s32_s4",
                "i0141",
                "matrix.mma.m16n8k32.row.col.s32.s4.s4.s32.wrapping",
                "int_nvvm_mma_m16n8k32_row_col_s4",
                "llvm.nvvm.mma.m16n8k32.row.col.s4",
                GeneratedStub,
            ),
            (M16n8k32, S4, U4, Wrapping) => (
                "mma_m16n8k32_s32_s4_u4",
                "i0142",
                "matrix.mma.m16n8k32.row.col.s32.s4.u4.s32.wrapping",
                "int_nvvm_mma_m16n8k32_row_col_s4_u4",
                "llvm.nvvm.mma.m16n8k32.row.col.s4.u4",
                GeneratedStub,
            ),
            (M16n8k32, U4, U4, Wrapping) => (
                "mma_m16n8k32_s32_u4",
                "i0143",
                "matrix.mma.m16n8k32.row.col.s32.u4.u4.s32.wrapping",
                "int_nvvm_mma_m16n8k32_row_col_u4",
                "llvm.nvvm.mma.m16n8k32.row.col.u4",
                GeneratedStub,
            ),
            (M16n8k32, U4, S4, Wrapping) => (
                "mma_m16n8k32_s32_u4_s4",
                "i0144",
                "matrix.mma.m16n8k32.row.col.s32.u4.s4.s32.wrapping",
                "int_nvvm_mma_m16n8k32_row_col_u4_s4",
                "llvm.nvvm.mma.m16n8k32.row.col.u4.s4",
                GeneratedStub,
            ),
            (M16n8k64, S4, S4, Wrapping) => (
                "mma_m16n8k64_s32_s4",
                "i0145",
                "matrix.mma.m16n8k64.row.col.s32.s4.s4.s32.wrapping",
                "int_nvvm_mma_m16n8k64_row_col_s4",
                "llvm.nvvm.mma.m16n8k64.row.col.s4",
                GeneratedStub,
            ),
            (M16n8k64, S4, U4, Wrapping) => (
                "mma_m16n8k64_s32_s4_u4",
                "i0146",
                "matrix.mma.m16n8k64.row.col.s32.s4.u4.s32.wrapping",
                "int_nvvm_mma_m16n8k64_row_col_s4_u4",
                "llvm.nvvm.mma.m16n8k64.row.col.s4.u4",
                GeneratedStub,
            ),
            (M16n8k64, U4, U4, Wrapping) => (
                "mma_m16n8k64_s32_u4",
                "i0147",
                "matrix.mma.m16n8k64.row.col.s32.u4.u4.s32.wrapping",
                "int_nvvm_mma_m16n8k64_row_col_u4",
                "llvm.nvvm.mma.m16n8k64.row.col.u4",
                GeneratedStub,
            ),
            (M16n8k64, U4, S4, Wrapping) => (
                "mma_m16n8k64_s32_u4_s4",
                "i0148",
                "matrix.mma.m16n8k64.row.col.s32.u4.s4.s32.wrapping",
                "int_nvvm_mma_m16n8k64_row_col_u4_s4",
                "llvm.nvvm.mma.m16n8k64.row.col.u4.s4",
                GeneratedStub,
            ),
            (M16n8k32, S4, S4, Satfinite) => (
                "mma_m16n8k32_s32_s4_satfinite",
                "i0149",
                "matrix.mma.m16n8k32.row.col.s32.s4.s4.s32.satfinite",
                "int_nvvm_mma_m16n8k32_row_col_satfinite_s4",
                "llvm.nvvm.mma.m16n8k32.row.col.satfinite.s4",
                GeneratedStub,
            ),
            (M16n8k32, S4, U4, Satfinite) => (
                "mma_m16n8k32_s32_s4_u4_satfinite",
                "i0150",
                "matrix.mma.m16n8k32.row.col.s32.s4.u4.s32.satfinite",
                "int_nvvm_mma_m16n8k32_row_col_satfinite_s4_u4",
                "llvm.nvvm.mma.m16n8k32.row.col.satfinite.s4.u4",
                GeneratedStub,
            ),
            (M16n8k32, U4, U4, Satfinite) => (
                "mma_m16n8k32_s32_u4_satfinite",
                "i0151",
                "matrix.mma.m16n8k32.row.col.s32.u4.u4.s32.satfinite",
                "int_nvvm_mma_m16n8k32_row_col_satfinite_u4",
                "llvm.nvvm.mma.m16n8k32.row.col.satfinite.u4",
                GeneratedStub,
            ),
            (M16n8k32, U4, S4, Satfinite) => (
                "mma_m16n8k32_s32_u4_s4_satfinite",
                "i0152",
                "matrix.mma.m16n8k32.row.col.s32.u4.s4.s32.satfinite",
                "int_nvvm_mma_m16n8k32_row_col_satfinite_u4_s4",
                "llvm.nvvm.mma.m16n8k32.row.col.satfinite.u4.s4",
                GeneratedStub,
            ),
            (M16n8k64, S4, S4, Satfinite) => (
                "mma_m16n8k64_s32_s4_satfinite",
                "i0153",
                "matrix.mma.m16n8k64.row.col.s32.s4.s4.s32.satfinite",
                "int_nvvm_mma_m16n8k64_row_col_satfinite_s4",
                "llvm.nvvm.mma.m16n8k64.row.col.satfinite.s4",
                GeneratedStub,
            ),
            (M16n8k64, S4, U4, Satfinite) => (
                "mma_m16n8k64_s32_s4_u4_satfinite",
                "i0154",
                "matrix.mma.m16n8k64.row.col.s32.s4.u4.s32.satfinite",
                "int_nvvm_mma_m16n8k64_row_col_satfinite_s4_u4",
                "llvm.nvvm.mma.m16n8k64.row.col.satfinite.s4.u4",
                GeneratedStub,
            ),
            (M16n8k64, U4, U4, Satfinite) => (
                "mma_m16n8k64_s32_u4_satfinite",
                "i0155",
                "matrix.mma.m16n8k64.row.col.s32.u4.u4.s32.satfinite",
                "int_nvvm_mma_m16n8k64_row_col_satfinite_u4",
                "llvm.nvvm.mma.m16n8k64.row.col.satfinite.u4",
                GeneratedStub,
            ),
            (M16n8k64, U4, S4, Satfinite) => (
                "mma_m16n8k64_s32_u4_s4_satfinite",
                "i0156",
                "matrix.mma.m16n8k64.row.col.s32.u4.s4.s32.satfinite",
                "int_nvvm_mma_m16n8k64_row_col_satfinite_u4_s4",
                "llvm.nvvm.mma.m16n8k64.row.col.satfinite.u4.s4",
                GeneratedStub,
            ),
            _ => return None,
        };

    let int4 = matches!(mma.a_element, S4 | U4);
    let (rust_arguments, dialect_operands, llvm_arguments, adapter, shape, register_counts) =
        match (mma.shape, int4) {
            (M8n8k16, false) | (M8n8k32, true) => (
                &["[i32; 2]", "u32", "u32"] as &'static [&'static str],
                &["i32", "i32", "i32", "i32"] as &'static [&'static str],
                &["i32", "i32", "i32", "i32"] as &'static [&'static str],
                C2I32A1U32B1U32ToD2I32,
                match mma.shape {
                    M8n8k16 => "m8n8k16",
                    M8n8k32 => "m8n8k32",
                    _ => unreachable!(),
                },
                [2, 1, 1, 2],
            ),
            (M16n8k16, false) | (M16n8k32, true) => (
                &["[i32; 4]", "[u32; 2]", "u32"] as &'static [&'static str],
                &["i32", "i32", "i32", "i32", "i32", "i32", "i32"] as &'static [&'static str],
                &["i32", "i32", "i32", "i32", "i32", "i32", "i32"] as &'static [&'static str],
                C4I32A2U32B1U32ToD4I32,
                match mma.shape {
                    M16n8k16 => "m16n8k16",
                    M16n8k32 => "m16n8k32",
                    _ => unreachable!(),
                },
                [4, 2, 1, 4],
            ),
            (M16n8k32, false) | (M16n8k64, true) => (
                &["[i32; 4]", "[u32; 4]", "[u32; 2]"] as &'static [&'static str],
                &[
                    "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32",
                ] as &'static [&'static str],
                &[
                    "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32",
                ] as &'static [&'static str],
                C4I32A4U32B2U32ToD4I32,
                match mma.shape {
                    M16n8k32 => "m16n8k32",
                    M16n8k64 => "m16n8k64",
                    _ => unreachable!(),
                },
                [4, 4, 2, 4],
            ),
            _ => return None,
        };
    if mma.adapter != adapter {
        return None;
    }

    let (rust_result, dialect_results, llvm_results, minimum_ptx, minimum_sm) = match mma.shape {
        M8n8k16 | M8n8k32 => (
            "[i32; 2]",
            &["i32", "i32"] as &'static [&'static str],
            &["i32", "i32"] as &'static [&'static str],
            "6.5",
            "sm_75",
        ),
        M16n8k16 | M16n8k32 | M16n8k64 => (
            "[i32; 4]",
            &["i32", "i32", "i32", "i32"] as &'static [&'static str],
            &["i32", "i32", "i32", "i32"] as &'static [&'static str],
            "7.0",
            "sm_80",
        ),
        _ => return None,
    };

    let element = |element| match element {
        S4 => Some("s4"),
        U4 => Some("u4"),
        S8 => Some("s8"),
        U8 => Some("u8"),
        _ => None,
    };
    let mut ptx_modifiers = vec!["sync", "aligned", shape, "row", "col"];
    if mma.overflow == Satfinite {
        ptx_modifiers.push("satfinite");
    }
    ptx_modifiers.extend([
        "s32",
        element(mma.a_element)?,
        element(mma.b_element)?,
        "s32",
    ]);

    Some(RegisterMmaRecipe {
        id,
        abi_id,
        operation_key,
        source_record,
        llvm_symbol,
        rust_arguments,
        rust_result,
        dialect_op_type: "RegisterMmaOp",
        dialect_op_name: "nvvm.register_mma",
        dialect_operands,
        dialect_results,
        llvm_arguments,
        llvm_results,
        adapter,
        compatibility_source,
        minimum_ptx,
        minimum_sm,
        ptx_modifiers,
        ptx_register_counts: register_counts,
    })
}

fn binary_register_mma_recipe(mma: &RegisterMma, common: bool) -> Option<RegisterMmaRecipe> {
    use RegisterMmaAdapter::{
        C2I32A1U32B1U32ToD2I32, C4I32A2U32B1U32ToD4I32, C4I32A4U32B2U32ToD4I32,
    };
    use RegisterMmaOperation::{AndPopc, XorPopc};
    use RegisterMmaShape::{M8n8k128, M16n8k128, M16n8k256};

    if !common
        || mma.accumulator != RegisterMmaAccumulator::S32
        || mma.a_element != RegisterMmaElement::B1
        || mma.b_element != RegisterMmaElement::B1
        || mma.overflow != RegisterMmaOverflow::Wrapping
        || mma.compatibility_source != RegisterMmaCompatibilitySource::GeneratedStub
    {
        return None;
    }

    let (id, abi_id, operation_key, source_record, llvm_symbol, operation_name) =
        match (mma.shape, mma.operation) {
            (M8n8k128, XorPopc) => (
                "mma_m8n8k128_s32_b1_xor_popc",
                "i0157",
                "matrix.mma.m8n8k128.row.col.s32.b1.b1.s32.xor.popc",
                "int_nvvm_mma_xor_popc_m8n8k128_row_col_b1",
                "llvm.nvvm.mma.xor.popc.m8n8k128.row.col.b1",
                "xor",
            ),
            (M16n8k128, XorPopc) => (
                "mma_m16n8k128_s32_b1_xor_popc",
                "i0158",
                "matrix.mma.m16n8k128.row.col.s32.b1.b1.s32.xor.popc",
                "int_nvvm_mma_xor_popc_m16n8k128_row_col_b1",
                "llvm.nvvm.mma.xor.popc.m16n8k128.row.col.b1",
                "xor",
            ),
            (M16n8k256, XorPopc) => (
                "mma_m16n8k256_s32_b1_xor_popc",
                "i0159",
                "matrix.mma.m16n8k256.row.col.s32.b1.b1.s32.xor.popc",
                "int_nvvm_mma_xor_popc_m16n8k256_row_col_b1",
                "llvm.nvvm.mma.xor.popc.m16n8k256.row.col.b1",
                "xor",
            ),
            (M8n8k128, AndPopc) => (
                "mma_m8n8k128_s32_b1_and_popc",
                "i0160",
                "matrix.mma.m8n8k128.row.col.s32.b1.b1.s32.and.popc",
                "int_nvvm_mma_and_popc_m8n8k128_row_col_b1",
                "llvm.nvvm.mma.and.popc.m8n8k128.row.col.b1",
                "and",
            ),
            (M16n8k128, AndPopc) => (
                "mma_m16n8k128_s32_b1_and_popc",
                "i0161",
                "matrix.mma.m16n8k128.row.col.s32.b1.b1.s32.and.popc",
                "int_nvvm_mma_and_popc_m16n8k128_row_col_b1",
                "llvm.nvvm.mma.and.popc.m16n8k128.row.col.b1",
                "and",
            ),
            (M16n8k256, AndPopc) => (
                "mma_m16n8k256_s32_b1_and_popc",
                "i0162",
                "matrix.mma.m16n8k256.row.col.s32.b1.b1.s32.and.popc",
                "int_nvvm_mma_and_popc_m16n8k256_row_col_b1",
                "llvm.nvvm.mma.and.popc.m16n8k256.row.col.b1",
                "and",
            ),
            _ => return None,
        };

    let (
        rust_arguments,
        dialect_operands,
        llvm_arguments,
        rust_result,
        result_types,
        adapter,
        counts,
    ) = match mma.shape {
        M8n8k128 => (
            &["[i32; 2]", "u32", "u32"] as &'static [&'static str],
            &["i32", "i32", "i32", "i32"] as &'static [&'static str],
            &["i32", "i32", "i32", "i32"] as &'static [&'static str],
            "[i32; 2]",
            &["i32", "i32"] as &'static [&'static str],
            C2I32A1U32B1U32ToD2I32,
            [2, 1, 1, 2],
        ),
        M16n8k128 => (
            &["[i32; 4]", "[u32; 2]", "u32"] as &'static [&'static str],
            &["i32", "i32", "i32", "i32", "i32", "i32", "i32"] as &'static [&'static str],
            &["i32", "i32", "i32", "i32", "i32", "i32", "i32"] as &'static [&'static str],
            "[i32; 4]",
            &["i32", "i32", "i32", "i32"] as &'static [&'static str],
            C4I32A2U32B1U32ToD4I32,
            [4, 2, 1, 4],
        ),
        M16n8k256 => (
            &["[i32; 4]", "[u32; 4]", "[u32; 2]"] as &'static [&'static str],
            &[
                "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32",
            ] as &'static [&'static str],
            &[
                "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32",
            ] as &'static [&'static str],
            "[i32; 4]",
            &["i32", "i32", "i32", "i32"] as &'static [&'static str],
            C4I32A4U32B2U32ToD4I32,
            [4, 4, 2, 4],
        ),
        _ => return None,
    };
    if mma.adapter != adapter {
        return None;
    }

    let (minimum_ptx, minimum_sm) = match (mma.operation, mma.shape) {
        (XorPopc, M8n8k128) => ("7.0", "sm_75"),
        (XorPopc, M16n8k128 | M16n8k256) => ("7.0", "sm_80"),
        (AndPopc, M8n8k128 | M16n8k128 | M16n8k256) => ("7.1", "sm_80"),
        _ => return None,
    };
    let shape = match mma.shape {
        M8n8k128 => "m8n8k128",
        M16n8k128 => "m16n8k128",
        M16n8k256 => "m16n8k256",
        _ => return None,
    };

    Some(RegisterMmaRecipe {
        id,
        abi_id,
        operation_key,
        source_record,
        llvm_symbol,
        rust_arguments,
        rust_result,
        dialect_op_type: "RegisterMmaOp",
        dialect_op_name: "nvvm.register_mma",
        dialect_operands,
        dialect_results: result_types,
        llvm_arguments,
        llvm_results: result_types,
        adapter,
        compatibility_source: RegisterMmaCompatibilitySource::GeneratedStub,
        minimum_ptx,
        minimum_sm,
        ptx_modifiers: vec![
            "sync",
            "aligned",
            shape,
            "row",
            "col",
            "s32",
            "b1",
            "b1",
            "s32",
            operation_name,
            "popc",
        ],
        ptx_register_counts: counts,
    })
}

fn register_mma_recipe(mma: &RegisterMma) -> Option<RegisterMmaRecipe> {
    use RegisterMmaAccumulator::{F32, F64};
    use RegisterMmaAdapter::{C2F64A1F64B1F64ToD2F64, C4F32A4U32B2U32ToD4F32};
    use RegisterMmaElement::{Bf16, F16, F64 as F64Element, Tf32};
    use RegisterMmaShape::{M8n8k4, M16n8k8, M16n8k16};

    let common = mma.a_layout == RegisterMmaLayout::Row
        && mma.b_layout == RegisterMmaLayout::Col
        && mma.participation
            == RegisterMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes;
    if let Some(recipe) = integer_register_mma_recipe(mma, common) {
        return Some(recipe);
    }
    if let Some(recipe) = binary_register_mma_recipe(mma, common) {
        return Some(recipe);
    }
    if mma.operation != RegisterMmaOperation::Multiply {
        return None;
    }
    match (
        mma.shape,
        mma.accumulator,
        mma.a_element,
        mma.b_element,
        mma.overflow,
        mma.adapter,
        common,
    ) {
        (
            M16n8k16,
            F32,
            Bf16,
            Bf16,
            RegisterMmaOverflow::NotApplicable,
            C4F32A4U32B2U32ToD4F32,
            true,
        ) => Some(RegisterMmaRecipe {
            id: "mma_m16n8k16_f32_bf16",
            abi_id: "i0105",
            operation_key: "matrix.mma.m16n8k16.row.col.f32.bf16.bf16.f32",
            source_record: "int_nvvm_mma_m16n8k16_row_col_bf16",
            llvm_symbol: "llvm.nvvm.mma.m16n8k16.row.col.bf16",
            rust_arguments: &["[f32; 4]", "[u32; 4]", "[u32; 2]"],
            rust_result: "[f32; 4]",
            dialect_op_type: "RegisterMmaOp",
            dialect_op_name: "nvvm.register_mma",
            dialect_operands: &[
                "f32", "f32", "f32", "f32", "i32", "i32", "i32", "i32", "i32", "i32",
            ],
            dialect_results: &["f32", "f32", "f32", "f32"],
            llvm_arguments: &[
                "i32", "i32", "i32", "i32", "i32", "i32", "f32", "f32", "f32", "f32",
            ],
            llvm_results: &["f32", "f32", "f32", "f32"],
            adapter: C4F32A4U32B2U32ToD4F32,
            compatibility_source: RegisterMmaCompatibilitySource::ExistingStub,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_modifiers: vec![
                "sync", "aligned", "m16n8k16", "row", "col", "f32", "bf16", "bf16", "f32",
            ],
            ptx_register_counts: [4, 4, 2, 4],
        }),
        (
            M16n8k16,
            F32,
            F16,
            F16,
            RegisterMmaOverflow::NotApplicable,
            C4F32A4U32B2U32ToD4F32,
            true,
        ) => Some(RegisterMmaRecipe {
            id: "mma_m16n8k16_f32_f16",
            abi_id: "i0106",
            operation_key: "matrix.mma.m16n8k16.row.col.f32.f16.f16.f32",
            source_record: "int_nvvm_mma_m16n8k16_row_col_f32_f32",
            llvm_symbol: "llvm.nvvm.mma.m16n8k16.row.col.f32.f32",
            rust_arguments: &["[f32; 4]", "[u32; 4]", "[u32; 2]"],
            rust_result: "[f32; 4]",
            dialect_op_type: "RegisterMmaOp",
            dialect_op_name: "nvvm.register_mma",
            dialect_operands: &[
                "f32", "f32", "f32", "f32", "i32", "i32", "i32", "i32", "i32", "i32",
            ],
            dialect_results: &["f32", "f32", "f32", "f32"],
            llvm_arguments: &[
                "v2f16", "v2f16", "v2f16", "v2f16", "v2f16", "v2f16", "f32", "f32", "f32", "f32",
            ],
            llvm_results: &["f32", "f32", "f32", "f32"],
            adapter: C4F32A4U32B2U32ToD4F32,
            compatibility_source: RegisterMmaCompatibilitySource::ExistingStub,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_modifiers: vec![
                "sync", "aligned", "m16n8k16", "row", "col", "f32", "f16", "f16", "f32",
            ],
            ptx_register_counts: [4, 4, 2, 4],
        }),
        (
            M16n8k8,
            F32,
            Tf32,
            Tf32,
            RegisterMmaOverflow::NotApplicable,
            C4F32A4U32B2U32ToD4F32,
            true,
        ) => Some(RegisterMmaRecipe {
            id: "mma_m16n8k8_f32_tf32",
            abi_id: "i0107",
            operation_key: "matrix.mma.m16n8k8.row.col.f32.tf32.tf32.f32",
            source_record: "int_nvvm_mma_m16n8k8_row_col_tf32",
            llvm_symbol: "llvm.nvvm.mma.m16n8k8.row.col.tf32",
            rust_arguments: &["[f32; 4]", "[u32; 4]", "[u32; 2]"],
            rust_result: "[f32; 4]",
            dialect_op_type: "RegisterMmaOp",
            dialect_op_name: "nvvm.register_mma",
            dialect_operands: &[
                "f32", "f32", "f32", "f32", "i32", "i32", "i32", "i32", "i32", "i32",
            ],
            dialect_results: &["f32", "f32", "f32", "f32"],
            llvm_arguments: &[
                "i32", "i32", "i32", "i32", "i32", "i32", "f32", "f32", "f32", "f32",
            ],
            llvm_results: &["f32", "f32", "f32", "f32"],
            adapter: C4F32A4U32B2U32ToD4F32,
            compatibility_source: RegisterMmaCompatibilitySource::ExistingStub,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_modifiers: vec![
                "sync", "aligned", "m16n8k8", "row", "col", "f32", "tf32", "tf32", "f32",
            ],
            ptx_register_counts: [4, 4, 2, 4],
        }),
        (
            M8n8k4,
            F64,
            F64Element,
            F64Element,
            RegisterMmaOverflow::NotApplicable,
            C2F64A1F64B1F64ToD2F64,
            true,
        ) => Some(RegisterMmaRecipe {
            id: "mma_m8n8k4_f64",
            abi_id: "i0109",
            operation_key: "matrix.mma.m8n8k4.row.col.f64.f64.f64.f64",
            source_record: "int_nvvm_mma_m8n8k4_row_col_f64",
            llvm_symbol: "llvm.nvvm.mma.m8n8k4.row.col.f64",
            rust_arguments: &["[f64; 2]", "f64", "f64"],
            rust_result: "[f64; 2]",
            dialect_op_type: "RegisterMmaOp",
            dialect_op_name: "nvvm.register_mma",
            dialect_operands: &["f64", "f64", "f64", "f64"],
            dialect_results: &["f64", "f64"],
            llvm_arguments: &["f64", "f64", "f64", "f64"],
            llvm_results: &["f64", "f64"],
            adapter: C2F64A1F64B1F64ToD2F64,
            compatibility_source: RegisterMmaCompatibilitySource::ExistingStub,
            minimum_ptx: "7.0",
            minimum_sm: "sm_80",
            ptx_modifiers: vec![
                "sync", "aligned", "m8n8k4", "row", "col", "f64", "f64", "f64", "f64",
            ],
            ptx_register_counts: [2, 1, 1, 2],
        }),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
enum RegisterMmaIntegerKind {
    Int4,
    Int8,
}

impl RegisterMmaIntegerKind {
    fn label(self) -> &'static str {
        match self {
            Self::Int4 => "INT4",
            Self::Int8 => "INT8",
        }
    }

    fn supports(self, shape: RegisterMmaShape, element: RegisterMmaElement) -> bool {
        match self {
            Self::Int4 => {
                matches!(
                    shape,
                    RegisterMmaShape::M8n8k32
                        | RegisterMmaShape::M16n8k32
                        | RegisterMmaShape::M16n8k64
                ) && matches!(element, RegisterMmaElement::S4 | RegisterMmaElement::U4)
            }
            Self::Int8 => {
                matches!(
                    shape,
                    RegisterMmaShape::M8n8k16
                        | RegisterMmaShape::M16n8k16
                        | RegisterMmaShape::M16n8k32
                ) && matches!(element, RegisterMmaElement::S8 | RegisterMmaElement::U8)
            }
        }
    }
}

fn expand_register_mma_integer_admission(
    kind: RegisterMmaIntegerKind,
    admission: &RegisterMmaIntegerAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    use RegisterMmaAdapter::{
        C2I32A1U32B1U32ToD2I32, C4I32A2U32B1U32ToD4I32, C4I32A4U32B2U32ToD4I32,
    };
    use RegisterMmaCompatibilitySource::GeneratedStub;
    use RegisterMmaLayout::{Col, Row};
    use RegisterMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes;
    use RegisterMmaShape::{M8n8k16, M8n8k32, M16n8k16, M16n8k32, M16n8k64};

    ensure!(
        !admission.variants.is_empty(),
        "compact {} MMA admission has no variants",
        kind.label()
    );
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "{} MMA runtime validation may be marked executed only with GPU evidence",
        kind.label()
    );

    let mut seen = BTreeSet::new();
    let mut records = Vec::with_capacity(admission.variants.len());
    for variant in &admission.variants {
        ensure!(
            seen.insert((
                variant.shape,
                variant.a_element,
                variant.b_element,
                variant.overflow,
            )),
            "compact {} MMA admission contains a duplicate variant",
            kind.label()
        );
        ensure!(
            kind.supports(variant.shape, variant.a_element)
                && kind.supports(variant.shape, variant.b_element),
            "compact {} MMA admission contains an unsupported shape or element",
            kind.label()
        );
        let adapter =
            match (kind, variant.shape) {
                (RegisterMmaIntegerKind::Int8, M8n8k16)
                | (RegisterMmaIntegerKind::Int4, M8n8k32) => C2I32A1U32B1U32ToD2I32,
                (RegisterMmaIntegerKind::Int8, M16n8k16)
                | (RegisterMmaIntegerKind::Int4, M16n8k32) => C4I32A2U32B1U32ToD4I32,
                (RegisterMmaIntegerKind::Int8, M16n8k32)
                | (RegisterMmaIntegerKind::Int4, M16n8k64) => C4I32A4U32B2U32ToD4I32,
                _ => bail!(
                    "compact {} MMA admission contains an unsupported shape",
                    kind.label()
                ),
            };
        let mma = RegisterMma {
            shape: variant.shape,
            operation: RegisterMmaOperation::Multiply,
            accumulator: RegisterMmaAccumulator::S32,
            a_element: variant.a_element,
            b_element: variant.b_element,
            a_layout: Row,
            b_layout: Col,
            overflow: variant.overflow,
            participation: AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
            adapter,
            compatibility_source: GeneratedStub,
            runtime_validation: admission.runtime_validation,
        };
        let recipe = register_mma_recipe(&mma).with_context(|| {
            format!(
                "compact {} MMA admission requests a variant outside the closed recipe set",
                kind.label()
            )
        })?;
        ensure!(
            recipe.compatibility_source == GeneratedStub,
            "compact {} MMA admission may only add generated compatibility stubs",
            kind.label()
        );

        let element = |element| match (kind, element) {
            (RegisterMmaIntegerKind::Int4, RegisterMmaElement::S4)
            | (RegisterMmaIntegerKind::Int8, RegisterMmaElement::S8) => Ok("signed"),
            (RegisterMmaIntegerKind::Int4, RegisterMmaElement::U4)
            | (RegisterMmaIntegerKind::Int8, RegisterMmaElement::U8) => Ok("unsigned"),
            _ => bail!(
                "compact {} MMA admission contains an unsupported element",
                kind.label()
            ),
        };
        let overflow = match variant.overflow {
            RegisterMmaOverflow::Wrapping => "wrapping",
            RegisterMmaOverflow::Satfinite => "saturating",
            RegisterMmaOverflow::NotApplicable => {
                bail!(
                    "compact {} MMA admission requires an integer overflow mode",
                    kind.label()
                )
            }
        };
        let summary = format!(
            "Multiplies warp-distributed {} A and {} B {} fragments and adds a {overflow} s32 accumulator.",
            element(variant.a_element)?,
            element(variant.b_element)?,
            kind.label(),
        );

        records.push(OverlayIntrinsic {
            id: recipe.id.into(),
            abi_id: recipe.abi_id.into(),
            operation_key: recipe.operation_key.into(),
            family: "register_mma".into(),
            source: None,
            source_record: Some(recipe.source_record.into()),
            rust_module: "matrix".into(),
            rust_name: recipe.id.into(),
            rust_arguments: recipe
                .rust_arguments
                .iter()
                .map(|value| (*value).into())
                .collect(),
            rust_result: recipe.rust_result.into(),
            safe: false,
            must_use: true,
            safe_allowlist_reason: None,
            public_rust_path: format!("cuda_intrinsics::matrix::{}", recipe.id),
            compatibility_rust_paths: vec![format!("cuda_device::wmma::{}", recipe.id)],
            dialect_op_type: recipe.dialect_op_type.into(),
            dialect_op_name: recipe.dialect_op_name.into(),
            dialect_operands: recipe
                .dialect_operands
                .iter()
                .map(|value| (*value).into())
                .collect(),
            dialect_results: recipe
                .dialect_results
                .iter()
                .map(|value| (*value).into())
                .collect(),
            llvm_symbol: Some(recipe.llvm_symbol.into()),
            resolved_llvm_symbol: None,
            llvm_arguments: recipe
                .llvm_arguments
                .iter()
                .map(|value| (*value).into())
                .collect(),
            llvm_results: recipe
                .llvm_results
                .iter()
                .map(|value| (*value).into())
                .collect(),
            pure: false,
            memory: "none".into(),
            convergent: true,
            execution_scope: "warp".into(),
            minimum_ptx: recipe.minimum_ptx.into(),
            minimum_sm: Some(recipe.minimum_sm.into()),
            ptx_result: recipe.rust_result.into(),
            targets: "all".into(),
            ptx_isa_version: "9.3".into(),
            ptx_isa_section: "9.7.15.5.14 Multiply-and-Accumulate Instruction: mma".into(),
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma".into(),
            lowering: "generated_register_mma".into(),
            backend_lowerings: vec![
                OverlayBackendLowering {
                    backend: IntrinsicBackend::LlvmNvptx,
                    mechanism: BackendLoweringMechanism::InlinePtx,
                    evidence_profile: admission.llvm_evidence_profile.clone(),
                    minimum_ptx: Some(recipe.minimum_ptx.into()),
                    minimum_sm: Some(recipe.minimum_sm.into()),
                },
                OverlayBackendLowering {
                    backend: IntrinsicBackend::LibNvvm,
                    mechanism: BackendLoweringMechanism::InlinePtx,
                    evidence_profile: admission.libnvvm_evidence_profile.clone(),
                    minimum_ptx: Some(recipe.minimum_ptx.into()),
                    minimum_sm: Some(recipe.minimum_sm.into()),
                },
            ],
            packed_atomic: None,
            redux: None,
            vote: None,
            active_mask: None,
            warp_match: None,
            warp_barrier: None,
            warp_shuffle: None,
            dot_product: None,
            packed_alu: None,
            packed_conversion: None,
            cp_async_copy: None,
            cp_async_control: None,
            cp_async_mbarrier: None,
            mbarrier_basic: None,
            movmatrix: None,
            mbarrier_extended: None,
            register_mma: Some(mma),
            sparse_mma: None,
            prmt: None,
            cluster_barrier: None,
            special_register: None,
            debug_control: None,
            cluster_memory: None,
            clc: None,
            ldmatrix_variant: None,
            ldmatrix_safety: None,
            ldmatrix_adapter: None,
            selected_address_space: None,
            expected_ptx: InstructionPattern {
                mnemonic: "mma".into(),
                modifiers: recipe
                    .ptx_modifiers
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                operands: recipe
                    .ptx_register_counts
                    .map(|length| OperandPattern::RegisterList { length })
                    .into(),
            },
            summary,
        });
    }
    Ok(records)
}

fn expand_register_mma_binary_admission(
    admission: &RegisterMmaBinaryAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    use RegisterMmaAdapter::{
        C2I32A1U32B1U32ToD2I32, C4I32A2U32B1U32ToD4I32, C4I32A4U32B2U32ToD4I32,
    };
    use RegisterMmaLayout::{Col, Row};
    use RegisterMmaOperation::{AndPopc, XorPopc};
    use RegisterMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes;
    use RegisterMmaShape::{M8n8k128, M16n8k128, M16n8k256};

    ensure!(
        !admission.variants.is_empty(),
        "compact binary MMA admission has no variants"
    );
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "binary MMA runtime validation may be marked executed only with GPU evidence"
    );

    let mut seen = BTreeSet::new();
    let mut records = Vec::with_capacity(admission.variants.len());
    for variant in &admission.variants {
        ensure!(
            seen.insert((variant.shape, variant.operation)),
            "compact binary MMA admission contains a duplicate variant"
        );
        ensure!(
            matches!(variant.operation, AndPopc | XorPopc),
            "compact binary MMA admission contains a non-binary operation"
        );
        let adapter = match variant.shape {
            M8n8k128 => C2I32A1U32B1U32ToD2I32,
            M16n8k128 => C4I32A2U32B1U32ToD4I32,
            M16n8k256 => C4I32A4U32B2U32ToD4I32,
            _ => bail!("compact binary MMA admission contains an unsupported shape"),
        };
        let mma = RegisterMma {
            shape: variant.shape,
            operation: variant.operation,
            accumulator: RegisterMmaAccumulator::S32,
            a_element: RegisterMmaElement::B1,
            b_element: RegisterMmaElement::B1,
            a_layout: Row,
            b_layout: Col,
            overflow: RegisterMmaOverflow::Wrapping,
            participation: AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
            adapter,
            compatibility_source: RegisterMmaCompatibilitySource::GeneratedStub,
            runtime_validation: admission.runtime_validation,
        };
        let recipe = register_mma_recipe(&mma).with_context(
            || "compact binary MMA admission requests a variant outside the closed recipe set",
        )?;
        let operation = match variant.operation {
            AndPopc => "AND and population count",
            XorPopc => "XOR and population count",
            RegisterMmaOperation::Multiply => unreachable!(),
        };
        let summary = format!(
            "Computes warp-distributed binary matrix products with {operation}, then adds a wrapping s32 accumulator."
        );

        records.push(OverlayIntrinsic {
            id: recipe.id.into(),
            abi_id: recipe.abi_id.into(),
            operation_key: recipe.operation_key.into(),
            family: "register_mma".into(),
            source: None,
            source_record: Some(recipe.source_record.into()),
            rust_module: "matrix".into(),
            rust_name: recipe.id.into(),
            rust_arguments: recipe
                .rust_arguments
                .iter()
                .map(|value| (*value).into())
                .collect(),
            rust_result: recipe.rust_result.into(),
            safe: false,
            must_use: true,
            safe_allowlist_reason: None,
            public_rust_path: format!("cuda_intrinsics::matrix::{}", recipe.id),
            compatibility_rust_paths: vec![format!("cuda_device::wmma::{}", recipe.id)],
            dialect_op_type: recipe.dialect_op_type.into(),
            dialect_op_name: recipe.dialect_op_name.into(),
            dialect_operands: recipe
                .dialect_operands
                .iter()
                .map(|value| (*value).into())
                .collect(),
            dialect_results: recipe
                .dialect_results
                .iter()
                .map(|value| (*value).into())
                .collect(),
            llvm_symbol: Some(recipe.llvm_symbol.into()),
            resolved_llvm_symbol: None,
            llvm_arguments: recipe
                .llvm_arguments
                .iter()
                .map(|value| (*value).into())
                .collect(),
            llvm_results: recipe
                .llvm_results
                .iter()
                .map(|value| (*value).into())
                .collect(),
            pure: false,
            memory: "none".into(),
            convergent: true,
            execution_scope: "warp".into(),
            minimum_ptx: recipe.minimum_ptx.into(),
            minimum_sm: Some(recipe.minimum_sm.into()),
            ptx_result: recipe.rust_result.into(),
            targets: "all".into(),
            ptx_isa_version: "9.3".into(),
            ptx_isa_section: "9.7.15.5.14 Multiply-and-Accumulate Instruction: mma".into(),
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma".into(),
            lowering: "generated_register_mma".into(),
            backend_lowerings: vec![
                OverlayBackendLowering {
                    backend: IntrinsicBackend::LlvmNvptx,
                    mechanism: BackendLoweringMechanism::InlinePtx,
                    evidence_profile: admission.llvm_evidence_profile.clone(),
                    minimum_ptx: Some(recipe.minimum_ptx.into()),
                    minimum_sm: Some(recipe.minimum_sm.into()),
                },
                OverlayBackendLowering {
                    backend: IntrinsicBackend::LibNvvm,
                    mechanism: BackendLoweringMechanism::InlinePtx,
                    evidence_profile: admission.libnvvm_evidence_profile.clone(),
                    minimum_ptx: Some(recipe.minimum_ptx.into()),
                    minimum_sm: Some(recipe.minimum_sm.into()),
                },
            ],
            packed_atomic: None,
            redux: None,
            vote: None,
            active_mask: None,
            warp_match: None,
            warp_barrier: None,
            warp_shuffle: None,
            dot_product: None,
            packed_alu: None,
            packed_conversion: None,
            cp_async_copy: None,
            cp_async_control: None,
            cp_async_mbarrier: None,
            mbarrier_basic: None,
            movmatrix: None,
            mbarrier_extended: None,
            register_mma: Some(mma),
            sparse_mma: None,
            prmt: None,
            cluster_barrier: None,
            special_register: None,
            debug_control: None,
            cluster_memory: None,
            clc: None,
            ldmatrix_variant: None,
            ldmatrix_safety: None,
            ldmatrix_adapter: None,
            selected_address_space: None,
            expected_ptx: InstructionPattern {
                mnemonic: "mma".into(),
                modifiers: recipe
                    .ptx_modifiers
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                operands: recipe
                    .ptx_register_counts
                    .map(|length| OperandPattern::RegisterList { length })
                    .into(),
            },
            summary,
        });
    }
    Ok(records)
}

#[derive(Clone, Copy)]
struct SparseMmaCarrierRecipe {
    adapter: SparseMmaAdapter,
    llvm_adapter: SparseMmaLlvmAdapter,
    accumulator: SparseMmaAccumulator,
    selector: SparseMmaSelector,
    a_registers: usize,
    b_registers: usize,
}

impl SparseMmaCarrierRecipe {
    fn operand_count(self) -> usize {
        4 + self.a_registers + self.b_registers + 2
    }

    fn selector_index(self) -> usize {
        self.operand_count() - 1
    }

    fn selector_upper_exclusive(self) -> u8 {
        match self.selector {
            SparseMmaSelector::ImmediateZeroOrOne => 2,
            SparseMmaSelector::ImmediateZero => 1,
        }
    }

    fn rust_arguments(self) -> Vec<String> {
        let accumulator = match self.accumulator {
            SparseMmaAccumulator::F32 => "[f32; 4]",
            SparseMmaAccumulator::S32 => "[i32; 4]",
        };
        vec![
            accumulator.into(),
            format!("[u32; {}]", self.a_registers),
            format!("[u32; {}]", self.b_registers),
            "u32".into(),
            "u32".into(),
        ]
    }

    fn dialect_operands(self) -> Vec<String> {
        let accumulator = match self.accumulator {
            SparseMmaAccumulator::F32 => "f32",
            SparseMmaAccumulator::S32 => "i32",
        };
        std::iter::repeat_n(accumulator.to_owned(), 4)
            .chain(std::iter::repeat_n(
                "u32".to_owned(),
                self.operand_count() - 4,
            ))
            .collect()
    }

    fn llvm_arguments(self) -> Vec<String> {
        let accumulator = match self.accumulator {
            SparseMmaAccumulator::F32 => "f32",
            SparseMmaAccumulator::S32 => "i32",
        };
        std::iter::repeat_n("i32".to_owned(), self.a_registers + self.b_registers)
            .chain(std::iter::repeat_n(accumulator.to_owned(), 4))
            .chain(std::iter::repeat_n("i32".to_owned(), 2))
            .collect()
    }

    fn rust_result(self) -> String {
        match self.accumulator {
            SparseMmaAccumulator::F32 => "[f32; 4]",
            SparseMmaAccumulator::S32 => "[i32; 4]",
        }
        .into()
    }

    fn scalar_result(self) -> &'static str {
        match self.accumulator {
            SparseMmaAccumulator::F32 => "f32",
            SparseMmaAccumulator::S32 => "i32",
        }
    }

    fn results(self) -> Vec<String> {
        vec![self.scalar_result().into(); 4]
    }

    fn expected_ptx_operands(self) -> Vec<OperandPattern> {
        vec![
            OperandPattern::RegisterList { length: 4 },
            OperandPattern::RegisterList {
                length: self.a_registers,
            },
            OperandPattern::RegisterList {
                length: self.b_registers,
            },
            OperandPattern::RegisterList { length: 4 },
            OperandPattern::Register,
            OperandPattern::Immediate,
        ]
    }

    fn imported_properties(self) -> Vec<String> {
        let selector = self.selector_index();
        vec![
            format!("ImmArg<arg{selector}>"),
            "IntrNoCallback".into(),
            "IntrNoMem".into(),
            format!("Range<arg{selector},0,{}>", self.selector_upper_exclusive()),
        ]
    }
}

fn sparse_mma_carrier_recipe(
    shape: SparseMmaShape,
    a_element: SparseMmaElement,
    b_element: SparseMmaElement,
) -> Option<SparseMmaCarrierRecipe> {
    use SparseMmaElement::{E2m1, E2m3, E3m2, E4m3, E5m2, S4, S8, U4, U8};
    use SparseMmaShape::{M16n8k32, M16n8k64, M16n8k128};

    match (shape, a_element, b_element) {
        (M16n8k32, S8 | U8, S8 | U8) | (M16n8k64, S4 | U4, S4 | U4) => {
            Some(SparseMmaCarrierRecipe {
                adapter: SparseMmaAdapter::C4I32A2U32B2U32MetadataU32SelectorU32ToD4I32,
                llvm_adapter: SparseMmaLlvmAdapter::A2I32B2I32C4I32MetadataI32SelectorI32ToD4I32,
                accumulator: SparseMmaAccumulator::S32,
                selector: SparseMmaSelector::ImmediateZeroOrOne,
                a_registers: 2,
                b_registers: 2,
            })
        }
        (M16n8k64, S8 | U8, S8 | U8) => Some(SparseMmaCarrierRecipe {
            adapter: SparseMmaAdapter::C4I32A4U32B4U32MetadataU32SelectorU32ToD4I32,
            llvm_adapter: SparseMmaLlvmAdapter::A4I32B4I32C4I32MetadataI32SelectorI32ToD4I32,
            accumulator: SparseMmaAccumulator::S32,
            selector: SparseMmaSelector::ImmediateZero,
            a_registers: 4,
            b_registers: 4,
        }),
        (M16n8k128, S4 | U4, S4 | U4) => Some(SparseMmaCarrierRecipe {
            adapter: SparseMmaAdapter::C4I32A4U32B4U32MetadataU32SelectorU32ToD4I32,
            llvm_adapter: SparseMmaLlvmAdapter::A4I32B4I32C4I32MetadataI32SelectorI32ToD4I32,
            accumulator: SparseMmaAccumulator::S32,
            selector: SparseMmaSelector::ImmediateZero,
            a_registers: 4,
            b_registers: 4,
        }),
        (M16n8k64, E2m1 | E2m3 | E3m2 | E4m3 | E5m2, E2m1 | E2m3 | E3m2 | E4m3 | E5m2) => {
            Some(SparseMmaCarrierRecipe {
                adapter: SparseMmaAdapter::C4F32A4U32B4U32MetadataU32SelectorU32ToD4F32,
                llvm_adapter: SparseMmaLlvmAdapter::A4I32B4I32C4F32MetadataI32SelectorI32ToD4F32,
                accumulator: SparseMmaAccumulator::F32,
                selector: SparseMmaSelector::ImmediateZero,
                a_registers: 4,
                b_registers: 4,
            })
        }
        _ => None,
    }
}

struct SparseMmaIdentity {
    id: String,
    operation_key: String,
    source_record: String,
    llvm_symbol: String,
    ptx_modifiers: Vec<&'static str>,
}

struct SparseMmaRecipe {
    identity: SparseMmaIdentity,
    carrier: SparseMmaCarrierRecipe,
}

fn sparse_mma_shape_name(shape: SparseMmaShape) -> &'static str {
    match shape {
        SparseMmaShape::M16n8k32 => "m16n8k32",
        SparseMmaShape::M16n8k64 => "m16n8k64",
        SparseMmaShape::M16n8k128 => "m16n8k128",
    }
}

fn sparse_mma_element_name(element: SparseMmaElement) -> &'static str {
    match element {
        SparseMmaElement::E2m1 => "e2m1",
        SparseMmaElement::E2m3 => "e2m3",
        SparseMmaElement::E3m2 => "e3m2",
        SparseMmaElement::E4m3 => "e4m3",
        SparseMmaElement::E5m2 => "e5m2",
        SparseMmaElement::S4 => "s4",
        SparseMmaElement::U4 => "u4",
        SparseMmaElement::S8 => "s8",
        SparseMmaElement::U8 => "u8",
    }
}

fn sparse_mma_identity(mma: &SparseMma) -> SparseMmaIdentity {
    let shape = sparse_mma_shape_name(mma.shape);
    let a_element = sparse_mma_element_name(mma.a_element);
    let b_element = sparse_mma_element_name(mma.b_element);
    if mma.accumulator == SparseMmaAccumulator::F32 {
        return SparseMmaIdentity {
            id: format!(
                "mma_sp_ordered_metadata_{shape}_kind_f8f6f4_f32_{a_element}_{b_element}_f32"
            ),
            operation_key: format!(
                "matrix.mma.sp.{shape}.row.col.kind_f8f6f4.f32.{a_element}.{b_element}.f32.not_applicable.ordered_metadata"
            ),
            source_record: format!(
                "int_nvvm_mma_sp_ordered_metadata_{shape}_row_col_kind_f8f6f4_f32_{a_element}_{b_element}_f32"
            ),
            llvm_symbol: format!(
                "llvm.nvvm.mma.sp.ordered.metadata.{shape}.row.col.kind.f8f6f4.f32.{a_element}.{b_element}.f32"
            ),
            ptx_modifiers: vec![
                "sp::ordered_metadata",
                "sync",
                "aligned",
                shape,
                "row",
                "col",
                "kind::f8f6f4",
                "f32",
                a_element,
                b_element,
                "f32",
            ],
        };
    }
    let compact_elements = if mma.a_element == mma.b_element {
        a_element.to_owned()
    } else {
        format!("{a_element}_{b_element}")
    };
    let dotted_elements = if mma.a_element == mma.b_element {
        a_element.to_owned()
    } else {
        format!("{a_element}.{b_element}")
    };
    let metadata_id_prefix = match mma.metadata {
        SparseMmaMetadata::Standard => "",
        SparseMmaMetadata::Ordered => "ordered_metadata_",
    };
    let metadata_source_prefix = metadata_id_prefix;
    let metadata_symbol_prefix = match mma.metadata {
        SparseMmaMetadata::Standard => "",
        SparseMmaMetadata::Ordered => "ordered.metadata.",
    };
    let metadata_key = match mma.metadata {
        SparseMmaMetadata::Standard => "standard_metadata",
        SparseMmaMetadata::Ordered => "ordered_metadata",
    };
    let overflow_id_suffix = match mma.overflow {
        SparseMmaOverflow::NotApplicable => unreachable!("integer sparse MMA has overflow"),
        SparseMmaOverflow::Wrapping => "",
        SparseMmaOverflow::Satfinite => "_satfinite",
    };
    let overflow_source_prefix = match mma.overflow {
        SparseMmaOverflow::NotApplicable => unreachable!("integer sparse MMA has overflow"),
        SparseMmaOverflow::Wrapping => "",
        SparseMmaOverflow::Satfinite => "satfinite_",
    };
    let overflow_symbol_prefix = match mma.overflow {
        SparseMmaOverflow::NotApplicable => unreachable!("integer sparse MMA has overflow"),
        SparseMmaOverflow::Wrapping => "",
        SparseMmaOverflow::Satfinite => "satfinite.",
    };
    let overflow_key = match mma.overflow {
        SparseMmaOverflow::NotApplicable => unreachable!("integer sparse MMA has overflow"),
        SparseMmaOverflow::Wrapping => "wrapping",
        SparseMmaOverflow::Satfinite => "satfinite",
    };

    let mut ptx_modifiers = vec![
        match mma.metadata {
            SparseMmaMetadata::Standard => "sp",
            SparseMmaMetadata::Ordered => "sp::ordered_metadata",
        },
        "sync",
        "aligned",
        shape,
        "row",
        "col",
    ];
    if mma.overflow == SparseMmaOverflow::Satfinite {
        ptx_modifiers.push("satfinite");
    }
    ptx_modifiers.extend(["s32", a_element, b_element, "s32"]);

    SparseMmaIdentity {
        id: format!(
            "mma_sp_{metadata_id_prefix}{shape}_s32_{compact_elements}{overflow_id_suffix}"
        ),
        operation_key: format!(
            "matrix.mma.sp.{shape}.row.col.s32.{a_element}.{b_element}.s32.{overflow_key}.{metadata_key}"
        ),
        source_record: format!(
            "int_nvvm_mma_sp_{metadata_source_prefix}{shape}_row_col_{overflow_source_prefix}{compact_elements}"
        ),
        llvm_symbol: format!(
            "llvm.nvvm.mma.sp.{metadata_symbol_prefix}{shape}.row.col.{overflow_symbol_prefix}{dotted_elements}"
        ),
        ptx_modifiers,
    }
}

fn sparse_mma_recipe(mma: &SparseMma) -> Option<SparseMmaRecipe> {
    let carrier = sparse_mma_carrier_recipe(mma.shape, mma.a_element, mma.b_element)?;

    let scalar_contract = match carrier.accumulator {
        SparseMmaAccumulator::F32 => {
            mma.accumulator == SparseMmaAccumulator::F32
                && mma.overflow == SparseMmaOverflow::NotApplicable
                && mma.metadata == SparseMmaMetadata::Ordered
        }
        SparseMmaAccumulator::S32 => {
            mma.accumulator == SparseMmaAccumulator::S32
                && matches!(
                    mma.overflow,
                    SparseMmaOverflow::Wrapping | SparseMmaOverflow::Satfinite
                )
        }
    };
    if !scalar_contract
        || mma.a_layout != SparseMmaLayout::Row
        || mma.b_layout != SparseMmaLayout::Col
        || mma.selector != carrier.selector
        || mma.participation
            != SparseMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes
        || mma.adapter != carrier.adapter
        || mma.llvm_adapter != carrier.llvm_adapter
        || mma.compatibility_source != SparseMmaCompatibilitySource::GeneratedStub
    {
        return None;
    }

    Some(SparseMmaRecipe {
        identity: sparse_mma_identity(mma),
        carrier,
    })
}

fn sparse_mma_minimum_ptx(mma: &SparseMma) -> &'static str {
    if mma.accumulator == SparseMmaAccumulator::F32 {
        return "8.7";
    }
    match mma.metadata {
        SparseMmaMetadata::Standard => "7.1",
        SparseMmaMetadata::Ordered => "8.5",
    }
}

fn sparse_mma_hardware(mma: &SparseMma) -> (&'static str, Option<&'static str>) {
    if mma.accumulator == SparseMmaAccumulator::F32 {
        ("sm_120a", None)
    } else {
        ("all", Some("sm_80"))
    }
}

fn sparse_mma_ptx_section(_: SparseMmaMetadata) -> &'static str {
    "9.7.15.6.3 Multiply-and-Accumulate Instruction: mma.sp"
}

fn expand_sparse_mma_integer_admission(
    admission: &SparseMmaIntegerAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        !admission.variants.is_empty(),
        "compact sparse integer MMA admission has no variants"
    );
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "sparse integer MMA runtime validation may be marked executed only with GPU evidence"
    );

    let mut seen = BTreeSet::new();
    let mut records = Vec::with_capacity(admission.variants.len());
    for variant in &admission.variants {
        ensure!(
            seen.insert((
                variant.shape,
                variant.a_element,
                variant.b_element,
                variant.overflow,
            )),
            "compact sparse integer MMA admission contains a duplicate variant"
        );
        let carrier = sparse_mma_carrier_recipe(
            variant.shape,
            variant.a_element,
            variant.b_element,
        )
        .with_context(
            || "compact sparse integer MMA admission uses unsupported or mixed-width elements",
        )?;
        let mma = SparseMma {
            shape: variant.shape,
            accumulator: SparseMmaAccumulator::S32,
            a_element: variant.a_element,
            b_element: variant.b_element,
            a_layout: SparseMmaLayout::Row,
            b_layout: SparseMmaLayout::Col,
            overflow: variant.overflow,
            metadata: admission.metadata,
            selector: carrier.selector,
            participation:
                SparseMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
            adapter: carrier.adapter,
            llvm_adapter: carrier.llvm_adapter,
            compatibility_source: SparseMmaCompatibilitySource::GeneratedStub,
            runtime_validation: admission.runtime_validation,
        };
        let recipe = sparse_mma_recipe(&mma).with_context(
            || "compact sparse integer MMA admission requests a variant outside the closed recipe set",
        )?;
        let signedness = |element| match element {
            SparseMmaElement::S4 => "signed",
            SparseMmaElement::U4 => "unsigned",
            SparseMmaElement::S8 => "signed",
            SparseMmaElement::U8 => "unsigned",
            _ => unreachable!("integer admission rejects floating formats"),
        };
        let width = match (variant.a_element, variant.b_element) {
            (
                SparseMmaElement::S4 | SparseMmaElement::U4,
                SparseMmaElement::S4 | SparseMmaElement::U4,
            ) => "INT4",
            (
                SparseMmaElement::S8 | SparseMmaElement::U8,
                SparseMmaElement::S8 | SparseMmaElement::U8,
            ) => "INT8",
            _ => unreachable!("carrier selection rejects mixed element widths"),
        };
        let overflow = match variant.overflow {
            SparseMmaOverflow::NotApplicable => {
                unreachable!("integer admission rejects inapplicable overflow")
            }
            SparseMmaOverflow::Wrapping => "wrapping",
            SparseMmaOverflow::Satfinite => "saturating",
        };
        let metadata = match admission.metadata {
            SparseMmaMetadata::Standard => "",
            SparseMmaMetadata::Ordered => " with ordered sparsity metadata",
        };
        let summary = format!(
            "Multiplies warp-distributed sparse {} A and {} B {width} fragments{metadata} and adds a {overflow} s32 accumulator.",
            signedness(variant.a_element),
            signedness(variant.b_element),
        );
        records.push(sparse_mma_overlay_record(
            mma,
            recipe,
            &admission.llvm_evidence_profile,
            &admission.libnvvm_evidence_profile,
            summary,
        ));
    }
    Ok(records)
}

fn expand_sparse_mma_f8f6f4_admission(
    admission: &SparseMmaF8F6F4Admission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "sparse f8f6f4 MMA runtime validation may be marked executed only with GPU evidence"
    );

    let formats = vec![
        SparseMmaElement::E2m1,
        SparseMmaElement::E2m3,
        SparseMmaElement::E3m2,
        SparseMmaElement::E4m3,
        SparseMmaElement::E5m2,
    ];
    ensure!(
        admission.a_elements == formats,
        "compact sparse f8f6f4 MMA admission must list the canonical five A formats"
    );
    ensure!(
        admission.b_elements == formats,
        "compact sparse f8f6f4 MMA admission must list the canonical five B formats"
    );
    ensure!(
        admission.product_count
            == admission
                .a_elements
                .len()
                .checked_mul(admission.b_elements.len())
                .context("compact sparse f8f6f4 MMA admission product count overflow")?
            && admission.product_count == 25,
        "compact sparse f8f6f4 MMA admission product_count must be exactly 25"
    );
    let mut records = Vec::with_capacity(admission.product_count);
    for &a_element in &admission.a_elements {
        for &b_element in &admission.b_elements {
            let carrier = sparse_mma_carrier_recipe(SparseMmaShape::M16n8k64, a_element, b_element)
                .with_context(
                    || "compact sparse f8f6f4 MMA admission uses an unsupported format",
                )?;
            ensure!(
                carrier.accumulator == SparseMmaAccumulator::F32,
                "compact sparse f8f6f4 MMA admission contains an integer format"
            );
            let mma = SparseMma {
                shape: SparseMmaShape::M16n8k64,
                accumulator: SparseMmaAccumulator::F32,
                a_element,
                b_element,
                a_layout: SparseMmaLayout::Row,
                b_layout: SparseMmaLayout::Col,
                overflow: SparseMmaOverflow::NotApplicable,
                metadata: SparseMmaMetadata::Ordered,
                selector: carrier.selector,
                participation:
                    SparseMmaParticipation::AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
                adapter: carrier.adapter,
                llvm_adapter: carrier.llvm_adapter,
                compatibility_source: SparseMmaCompatibilitySource::GeneratedStub,
                runtime_validation: admission.runtime_validation,
            };
            let recipe = sparse_mma_recipe(&mma).with_context(|| {
            "compact sparse f8f6f4 MMA admission requests a variant outside the closed recipe set"
        })?;
            let summary = format!(
                "Multiplies warp-distributed sparse {} A and {} B fragments and adds an f32 accumulator.",
                sparse_mma_element_name(a_element),
                sparse_mma_element_name(b_element),
            );
            records.push(sparse_mma_overlay_record(
                mma,
                recipe,
                &admission.llvm_evidence_profile,
                &admission.libnvvm_evidence_profile,
                summary,
            ));
        }
    }
    ensure!(records.len() == admission.product_count);
    Ok(records)
}

fn sparse_mma_overlay_record(
    mma: SparseMma,
    recipe: SparseMmaRecipe,
    llvm_evidence_profile: &str,
    libnvvm_evidence_profile: &str,
    summary: String,
) -> OverlayIntrinsic {
    let identity = &recipe.identity;
    let minimum_ptx = sparse_mma_minimum_ptx(&mma);
    let (targets, minimum_sm) = sparse_mma_hardware(&mma);
    OverlayIntrinsic {
        id: identity.id.clone(),
        abi_id: String::new(),
        operation_key: identity.operation_key.clone(),
        family: "sparse_mma".into(),
        source: None,
        source_record: Some(identity.source_record.clone()),
        rust_module: "matrix".into(),
        rust_name: identity.id.clone(),
        rust_arguments: recipe.carrier.rust_arguments(),
        rust_result: recipe.carrier.rust_result(),
        safe: false,
        must_use: true,
        safe_allowlist_reason: None,
        public_rust_path: format!("cuda_intrinsics::matrix::{}", identity.id),
        compatibility_rust_paths: vec![format!("cuda_device::wmma::{}", identity.id)],
        dialect_op_type: "SparseMmaOp".into(),
        dialect_op_name: "nvvm.sparse_mma".into(),
        dialect_operands: recipe.carrier.dialect_operands(),
        dialect_results: recipe.carrier.results(),
        llvm_symbol: Some(identity.llvm_symbol.clone()),
        resolved_llvm_symbol: None,
        llvm_arguments: recipe.carrier.llvm_arguments(),
        llvm_results: recipe.carrier.results(),
        pure: false,
        memory: "none".into(),
        convergent: true,
        execution_scope: "warp".into(),
        minimum_ptx: minimum_ptx.into(),
        minimum_sm: minimum_sm.map(str::to_owned),
        ptx_result: recipe.carrier.rust_result(),
        targets: targets.into(),
        ptx_isa_version: "9.3".into(),
        ptx_isa_section: sparse_mma_ptx_section(mma.metadata).into(),
        ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma-sp".into(),
        lowering: "generated_sparse_mma".into(),
        backend_lowerings: [
            (IntrinsicBackend::LlvmNvptx, llvm_evidence_profile),
            (IntrinsicBackend::LibNvvm, libnvvm_evidence_profile),
        ]
        .into_iter()
        .map(|(backend, evidence_profile)| OverlayBackendLowering {
            backend,
            mechanism: BackendLoweringMechanism::InlinePtx,
            evidence_profile: evidence_profile.into(),
            minimum_ptx: Some(minimum_ptx.into()),
            minimum_sm: minimum_sm.map(str::to_owned),
        })
        .collect(),
        packed_atomic: None,
        redux: None,
        vote: None,
        active_mask: None,
        warp_match: None,
        warp_barrier: None,
        warp_shuffle: None,
        dot_product: None,
        packed_alu: None,
        packed_conversion: None,
        cp_async_copy: None,
        cp_async_control: None,
        cp_async_mbarrier: None,
        mbarrier_basic: None,
        movmatrix: None,
        mbarrier_extended: None,
        register_mma: None,
        sparse_mma: Some(mma),
        prmt: None,
        cluster_barrier: None,
        special_register: None,
        debug_control: None,
        cluster_memory: None,
        clc: None,
        ldmatrix_variant: None,
        ldmatrix_safety: None,
        ldmatrix_adapter: None,
        selected_address_space: None,
        expected_ptx: InstructionPattern {
            mnemonic: "mma".into(),
            modifiers: identity
                .ptx_modifiers
                .iter()
                .map(|value| (*value).into())
                .collect(),
            operands: recipe.carrier.expected_ptx_operands(),
        },
        summary,
    }
}

fn validate_sparse_mma_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let mma = policy
        .sparse_mma
        .as_ref()
        .with_context(|| format!("{} has no closed sparse-MMA contract", policy.id))?;
    let recipe = sparse_mma_recipe(mma)
        .with_context(|| format!("{} requests an unsupported sparse-MMA variant", policy.id))?;
    let identity = &recipe.identity;
    let minimum_ptx = sparse_mma_minimum_ptx(mma);
    let (targets, minimum_sm) = sparse_mma_hardware(mma);
    ensure!(
        policy.id == identity.id
            && policy.operation_key == identity.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(identity.source_record.as_str())
            && policy.llvm_symbol.as_deref() == Some(identity.llvm_symbol.as_str())
            && policy.resolved_llvm_symbol.is_none(),
        "{} sparse-MMA identity does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "matrix"
            && policy.rust_name == identity.id
            && policy.public_rust_path == format!("cuda_intrinsics::matrix::{}", identity.id)
            && policy.rust_arguments == recipe.carrier.rust_arguments()
            && policy.rust_result == recipe.carrier.rust_result()
            && !policy.safe
            && policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.compatibility_rust_paths == [format!("cuda_device::wmma::{}", identity.id)],
        "{} must preserve its unsafe must-use Rust sparse-MMA API",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == "SparseMmaOp"
            && policy.dialect_op_name == "nvvm.sparse_mma"
            && policy.dialect_operands == recipe.carrier.dialect_operands()
            && policy.dialect_results == recipe.carrier.results()
            && policy.llvm_arguments == recipe.carrier.llvm_arguments()
            && policy.llvm_results == recipe.carrier.results()
            && policy.ptx_result == recipe.carrier.rust_result()
            && policy.lowering == "generated_sparse_mma",
        "{} sparse-MMA carrier or lowering adapter disagrees with its recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "none"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == minimum_ptx
            && policy.minimum_sm.as_deref() == minimum_sm
            && policy.targets == targets,
        "{} sparse-MMA effects or target floor disagree with PTX",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == sparse_mma_ptx_section(mma.metadata)
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma-sp",
        "{} sparse-MMA PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        declaration.classes == ["SDPatternOperator", "Intrinsic", "NVVM_MMA_SP"]
            && declaration.properties == recipe.carrier.imported_properties()
            && declaration.selections.is_empty(),
        "{} imported sparse MMA declaration changed its class, immediate range, properties, or selectionless contract",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "mma"
            && policy.expected_ptx.modifiers == identity.ptx_modifiers
            && policy.expected_ptx.operands == recipe.carrier.expected_ptx_operands(),
        "{} expected PTX does not match its exact sparse-MMA spelling",
        policy.id
    );
    ensure_exact_inline_ptx_backends(
        policy,
        [
            (IntrinsicBackend::LlvmNvptx, minimum_ptx, minimum_sm),
            (IntrinsicBackend::LibNvvm, minimum_ptx, minimum_sm),
        ],
        "sparse MMA",
    )?;
    ensure_no_other_family_contract(policy, "sparse MMA")?;
    Ok(())
}

fn validate_register_mma_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let mma = policy
        .register_mma
        .as_ref()
        .with_context(|| format!("{} has no closed register-MMA contract", policy.id))?;
    let recipe = register_mma_recipe(mma)
        .with_context(|| format!("{} requests an unsupported register-MMA variant", policy.id))?;
    ensure!(
        policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none(),
        "{} register-MMA identity does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "matrix"
            && policy.rust_name == recipe.id
            && policy.rust_arguments == recipe.rust_arguments
            && policy.rust_result == recipe.rust_result
            && !policy.safe
            && policy.must_use
            && policy.safe_allowlist_reason.is_none()
            && policy.compatibility_rust_paths == [format!("cuda_device::wmma::{}", recipe.id)],
        "{} must preserve its unsafe must-use Rust MMA API",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == recipe.dialect_operands
            && policy.dialect_results == recipe.dialect_results
            && policy.llvm_arguments == recipe.llvm_arguments
            && policy.llvm_results == recipe.llvm_results
            && policy.ptx_result == recipe.rust_result
            && mma.adapter == recipe.adapter
            && mma.compatibility_source == recipe.compatibility_source
            && policy.lowering == "generated_register_mma",
        "{} register-MMA carrier or lowering adapter disagrees with its recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "none"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == recipe.minimum_ptx
            && policy.minimum_sm.as_deref() == Some(recipe.minimum_sm)
            && policy.targets == "all",
        "{} register-MMA effects or target floor disagree with PTX",
        policy.id
    );
    ensure!(
        policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == "9.7.15.5.14 Multiply-and-Accumulate Instruction: mma"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma",
        "{} register-MMA PTX provenance disagrees with the reviewed recipe",
        policy.id
    );
    ensure!(
        declaration.classes.iter().any(|class| class == "NVVM_MMA")
            && declaration.properties == ["IntrNoCallback", "IntrNoMem"]
            && declaration.selections.is_empty(),
        "{} imported MMA declaration changed its class, properties, or selectionless contract",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == "mma"
            && policy.expected_ptx.modifiers == recipe.ptx_modifiers
            && policy.expected_ptx.operands
                == recipe
                    .ptx_register_counts
                    .map(|length| OperandPattern::RegisterList { length }),
        "{} expected PTX does not match its exact register-MMA spelling",
        policy.id
    );
    ensure_exact_inline_ptx_backends(
        policy,
        [
            (
                IntrinsicBackend::LlvmNvptx,
                recipe.minimum_ptx,
                Some(recipe.minimum_sm),
            ),
            (
                IntrinsicBackend::LibNvvm,
                recipe.minimum_ptx,
                Some(recipe.minimum_sm),
            ),
        ],
        "register MMA",
    )?;
    ensure_no_other_family_contract(policy, "register MMA")?;
    Ok(())
}

#[derive(Clone, Copy)]
struct PrmtRecipe {
    abi_id: &'static str,
    id: &'static str,
    operation_key: &'static str,
    source_record: &'static str,
    llvm_symbol: &'static str,
    modifier: Option<&'static str>,
    adapter: PrmtAdapter,
    summary: &'static str,
}

fn prmt_recipe(mode: PrmtMode) -> PrmtRecipe {
    use PrmtAdapter::{DirectThreeOperands, InsertZeroSecondSource};
    match mode {
        PrmtMode::Generic => PrmtRecipe {
            abi_id: "i0252",
            id: "prmt",
            operation_key: "integer.prmt.b32",
            source_record: "int_nvvm_prmt",
            llvm_symbol: "llvm.nvvm.prmt",
            modifier: None,
            adapter: DirectThreeOperands,
            summary: "Permutes bytes selected from two 32-bit inputs.",
        },
        PrmtMode::F4e => PrmtRecipe {
            abi_id: "i0253",
            id: "prmt_f4e",
            operation_key: "integer.prmt.b32.f4e",
            source_record: "int_nvvm_prmt_f4e",
            llvm_symbol: "llvm.nvvm.prmt.f4e",
            modifier: Some("f4e"),
            adapter: DirectThreeOperands,
            summary: "Permutes bytes with the forward four-byte extract mode.",
        },
        PrmtMode::B4e => PrmtRecipe {
            abi_id: "i0254",
            id: "prmt_b4e",
            operation_key: "integer.prmt.b32.b4e",
            source_record: "int_nvvm_prmt_b4e",
            llvm_symbol: "llvm.nvvm.prmt.b4e",
            modifier: Some("b4e"),
            adapter: DirectThreeOperands,
            summary: "Permutes bytes with the backward four-byte extract mode.",
        },
        PrmtMode::Rc8 => PrmtRecipe {
            abi_id: "i0255",
            id: "prmt_rc8",
            operation_key: "integer.prmt.b32.rc8",
            source_record: "int_nvvm_prmt_rc8",
            llvm_symbol: "llvm.nvvm.prmt.rc8",
            modifier: Some("rc8"),
            adapter: InsertZeroSecondSource,
            summary: "Replicates a selected byte across the 32-bit result.",
        },
        PrmtMode::Ecl => PrmtRecipe {
            abi_id: "i0256",
            id: "prmt_ecl",
            operation_key: "integer.prmt.b32.ecl",
            source_record: "int_nvvm_prmt_ecl",
            llvm_symbol: "llvm.nvvm.prmt.ecl",
            modifier: Some("ecl"),
            adapter: InsertZeroSecondSource,
            summary: "Clamps a byte extract toward the least-significant byte.",
        },
        PrmtMode::Ecr => PrmtRecipe {
            abi_id: "i0257",
            id: "prmt_ecr",
            operation_key: "integer.prmt.b32.ecr",
            source_record: "int_nvvm_prmt_ecr",
            llvm_symbol: "llvm.nvvm.prmt.ecr",
            modifier: Some("ecr"),
            adapter: InsertZeroSecondSource,
            summary: "Clamps a byte extract toward the most-significant byte.",
        },
        PrmtMode::Rc16 => PrmtRecipe {
            abi_id: "i0258",
            id: "prmt_rc16",
            operation_key: "integer.prmt.b32.rc16",
            source_record: "int_nvvm_prmt_rc16",
            llvm_symbol: "llvm.nvvm.prmt.rc16",
            modifier: Some("rc16"),
            adapter: InsertZeroSecondSource,
            summary: "Replicates a selected 16-bit half across the result.",
        },
    }
}

fn expand_prmt_admission(admission: &PrmtAdmission) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "prmt runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "compact prmt admission requires both backend evidence profiles"
    );
    let expected_modes = BTreeSet::from([
        PrmtMode::Generic,
        PrmtMode::F4e,
        PrmtMode::B4e,
        PrmtMode::Rc8,
        PrmtMode::Ecl,
        PrmtMode::Ecr,
        PrmtMode::Rc16,
    ]);
    let actual_modes: BTreeSet<_> = admission
        .variants
        .iter()
        .map(|variant| variant.mode)
        .collect();
    ensure!(
        admission.variants.len() == expected_modes.len() && actual_modes == expected_modes,
        "compact prmt admission must contain each of the seven reviewed modes exactly once"
    );

    admission
        .variants
        .iter()
        .map(|variant| {
            let recipe = prmt_recipe(variant.mode);
            ensure!(
                variant.abi_id == recipe.abi_id,
                "{} must keep reserved ABI ID {}",
                recipe.id,
                recipe.abi_id
            );
            let three_operands = recipe.adapter == PrmtAdapter::DirectThreeOperands;
            let rust_arguments = vec!["u32".into(); if three_operands { 3 } else { 2 }];
            let llvm_arguments = vec!["i32".into(); if three_operands { 3 } else { 2 }];
            let mut modifiers = vec!["b32".into()];
            if let Some(modifier) = recipe.modifier {
                modifiers.push(modifier.into());
            }
            let operands = if three_operands {
                vec![
                    OperandPattern::Register,
                    OperandPattern::Register,
                    OperandPattern::Register,
                    OperandPattern::Register,
                ]
            } else {
                vec![
                    OperandPattern::Register,
                    OperandPattern::Register,
                    OperandPattern::Exact { value: "0".into() },
                    OperandPattern::Register,
                ]
            };
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: variant.abi_id.clone(),
                operation_key: recipe.operation_key.into(),
                family: "prmt".into(),
                source: None,
                source_record: Some(recipe.source_record.into()),
                rust_module: "prmt".into(),
                rust_name: recipe.id.into(),
                rust_arguments,
                rust_result: "u32".into(),
                safe: true,
                must_use: true,
                safe_allowlist_reason: Some(
                    "it only permutes register bytes and has no caller preconditions.".into(),
                ),
                public_rust_path: format!("cuda_intrinsics::prmt::{}", recipe.id),
                compatibility_rust_paths: vec![format!("cuda_device::prmt::{}", recipe.id)],
                dialect_op_type: "PrmtOp".into(),
                dialect_op_name: "nvvm.prmt".into(),
                dialect_operands: llvm_arguments.clone(),
                dialect_results: vec!["i32".into()],
                llvm_symbol: Some(recipe.llvm_symbol.into()),
                resolved_llvm_symbol: None,
                llvm_arguments,
                llvm_results: vec!["i32".into()],
                pure: true,
                memory: "none".into(),
                convergent: false,
                execution_scope: "thread".into(),
                minimum_ptx: "2.0".into(),
                minimum_sm: Some("sm_20".into()),
                ptx_result: "u32".into(),
                targets: "all".into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section: "9.7.9.7 Data Movement and Conversion Instructions: prmt".into(),
                ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prmt".into(),
                lowering: "generated_prmt".into(),
                backend_lowerings: vec![
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LlvmNvptx,
                        mechanism: BackendLoweringMechanism::TypedNvvm,
                        evidence_profile: admission.llvm_evidence_profile.clone(),
                        minimum_ptx: Some("3.2".into()),
                        minimum_sm: Some("sm_20".into()),
                    },
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LibNvvm,
                        mechanism: BackendLoweringMechanism::InlinePtx,
                        evidence_profile: admission.libnvvm_evidence_profile.clone(),
                        minimum_ptx: None,
                        minimum_sm: Some("sm_75".into()),
                    },
                ],
                packed_atomic: None,
                redux: None,
                vote: None,
                active_mask: None,
                warp_match: None,
                warp_barrier: None,
                warp_shuffle: None,
                dot_product: None,
                packed_alu: None,
                packed_conversion: None,
                cp_async_copy: None,
                cp_async_control: None,
                cp_async_mbarrier: None,
                mbarrier_basic: None,
                movmatrix: None,
                mbarrier_extended: None,
                register_mma: None,
                sparse_mma: None,
                prmt: Some(Prmt {
                    mode: variant.mode,
                    adapter: recipe.adapter,
                }),
                cluster_barrier: None,
                special_register: None,
                debug_control: None,
                cluster_memory: None,
                clc: None,
                ldmatrix_variant: None,
                ldmatrix_safety: None,
                ldmatrix_adapter: None,
                selected_address_space: None,
                expected_ptx: InstructionPattern {
                    mnemonic: "prmt".into(),
                    modifiers,
                    operands,
                },
                summary: recipe.summary.into(),
            })
        })
        .collect()
}

fn validate_prmt_policy(policy: &OverlayIntrinsic, declaration: &ImportedIntrinsic) -> Result<()> {
    let prmt = policy
        .prmt
        .as_ref()
        .with_context(|| format!("{} has no closed prmt contract", policy.id))?;
    let recipe = prmt_recipe(prmt.mode);
    let three_operands = recipe.adapter == PrmtAdapter::DirectThreeOperands;
    ensure!(
        prmt.adapter == recipe.adapter
            && policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(recipe.source_record)
            && policy.llvm_symbol.as_deref() == Some(recipe.llvm_symbol)
            && policy.resolved_llvm_symbol.is_none(),
        "{} prmt identity does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "prmt"
            && policy.rust_name == recipe.id
            && policy.rust_arguments == vec!["u32"; if three_operands { 3 } else { 2 }]
            && policy.rust_result == "u32"
            && policy.safe
            && policy.must_use
            && policy.public_rust_path == format!("cuda_intrinsics::prmt::{}", recipe.id)
            && policy.compatibility_rust_paths == [format!("cuda_device::prmt::{}", recipe.id)],
        "{} prmt Rust API does not match its closed recipe",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == "PrmtOp"
            && policy.dialect_op_name == "nvvm.prmt"
            && policy.dialect_operands == vec!["i32"; if three_operands { 3 } else { 2 }]
            && policy.dialect_results == ["i32"]
            && policy.llvm_arguments == policy.dialect_operands
            && policy.llvm_results == ["i32"]
            && policy.lowering == "generated_prmt",
        "{} prmt carrier or lowering does not match its closed recipe",
        policy.id
    );
    ensure!(
        declaration
            .classes
            .iter()
            .any(|class| class == "NVVMPureIntrinsic")
            && declaration.properties == ["IntrNoMem", "IntrSpeculatable"]
            && policy.pure
            && policy.memory == "none"
            && !policy.convergent
            && policy.execution_scope == "thread",
        "{} prmt effects disagree with the imported declaration",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == "2.0"
            && policy.minimum_sm.as_deref() == Some("sm_20")
            && policy.targets == "all"
            && policy.ptx_result == "u32"
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == "9.7.9.7 Data Movement and Conversion Instructions: prmt"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-prmt",
        "{} prmt target floor or PTX provenance changed",
        policy.id
    );
    let mut modifiers = vec!["b32"];
    if let Some(modifier) = recipe.modifier {
        modifiers.push(modifier);
    }
    let expected_operands = if three_operands {
        vec![
            OperandPattern::Register,
            OperandPattern::Register,
            OperandPattern::Register,
            OperandPattern::Register,
        ]
    } else {
        vec![
            OperandPattern::Register,
            OperandPattern::Register,
            OperandPattern::Exact { value: "0".into() },
            OperandPattern::Register,
        ]
    };
    ensure!(
        policy.expected_ptx.mnemonic == "prmt"
            && policy.expected_ptx.modifiers == modifiers
            && policy.expected_ptx.operands == expected_operands,
        "{} expected PTX does not match its closed prmt mode",
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
        "{} must define exactly the reviewed prmt backend routes",
        policy.id
    );
    for lowering in &policy.backend_lowerings {
        let floor_matches = match lowering.backend {
            IntrinsicBackend::LlvmNvptx => {
                lowering.minimum_ptx.as_deref() == Some("3.2")
                    && lowering.minimum_sm.as_deref() == Some("sm_20")
            }
            IntrinsicBackend::LibNvvm => {
                lowering.minimum_ptx.is_none() && lowering.minimum_sm.as_deref() == Some("sm_75")
            }
        };
        ensure!(
            floor_matches && !lowering.evidence_profile.trim().is_empty(),
            "{} backend {:?} does not carry its reviewed prmt floor",
            policy.id,
            lowering.backend
        );
    }
    ensure_no_other_family_contract(policy, "prmt")?;
    Ok(())
}

#[derive(Clone, Copy)]
struct ClusterBarrierRecipe {
    mode: ClusterBarrierMode,
    abi_id: &'static str,
    id: &'static str,
    suffix: &'static str,
    minimum_ptx: &'static str,
    ordering: ClusterBarrierOrdering,
    aligned: bool,
    summary: &'static str,
}

fn cluster_barrier_recipe(mode: ClusterBarrierMode) -> ClusterBarrierRecipe {
    match mode {
        ClusterBarrierMode::Arrive => ClusterBarrierRecipe {
            mode,
            abi_id: "i0277",
            id: "barrier_cluster_arrive",
            suffix: "arrive",
            minimum_ptx: "7.8",
            ordering: ClusterBarrierOrdering::Release,
            aligned: false,
            summary: "Arrives at the cluster barrier with release ordering.",
        },
        ClusterBarrierMode::ArriveAligned => ClusterBarrierRecipe {
            mode,
            abi_id: "i0278",
            id: "barrier_cluster_arrive_aligned",
            suffix: "arrive.aligned",
            minimum_ptx: "7.8",
            ordering: ClusterBarrierOrdering::Release,
            aligned: true,
            summary: "Arrives at the cluster barrier in aligned mode with release ordering.",
        },
        ClusterBarrierMode::ArriveRelaxed => ClusterBarrierRecipe {
            mode,
            abi_id: "i0279",
            id: "barrier_cluster_arrive_relaxed",
            suffix: "arrive.relaxed",
            minimum_ptx: "8.0",
            ordering: ClusterBarrierOrdering::Relaxed,
            aligned: false,
            summary: "Arrives at the cluster barrier without a release guarantee.",
        },
        ClusterBarrierMode::ArriveRelaxedAligned => ClusterBarrierRecipe {
            mode,
            abi_id: "i0280",
            id: "barrier_cluster_arrive_relaxed_aligned",
            suffix: "arrive.relaxed.aligned",
            minimum_ptx: "8.0",
            ordering: ClusterBarrierOrdering::Relaxed,
            aligned: true,
            summary: "Arrives at the cluster barrier in aligned mode without a release guarantee.",
        },
        ClusterBarrierMode::Wait => ClusterBarrierRecipe {
            mode,
            abi_id: "i0281",
            id: "barrier_cluster_wait",
            suffix: "wait",
            minimum_ptx: "7.8",
            ordering: ClusterBarrierOrdering::Acquire,
            aligned: false,
            summary: "Waits at the cluster barrier with acquire ordering.",
        },
        ClusterBarrierMode::WaitAligned => ClusterBarrierRecipe {
            mode,
            abi_id: "i0282",
            id: "barrier_cluster_wait_aligned",
            suffix: "wait.aligned",
            minimum_ptx: "7.8",
            ordering: ClusterBarrierOrdering::Acquire,
            aligned: true,
            summary: "Waits at the cluster barrier in aligned mode with acquire ordering.",
        },
    }
}

fn expand_cluster_barrier_admission(
    admission: &ClusterBarrierAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "cluster-barrier runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "compact cluster-barrier admission requires both backend evidence profiles"
    );
    let expected_modes = BTreeSet::from([
        ClusterBarrierMode::Arrive,
        ClusterBarrierMode::ArriveAligned,
        ClusterBarrierMode::ArriveRelaxed,
        ClusterBarrierMode::ArriveRelaxedAligned,
        ClusterBarrierMode::Wait,
        ClusterBarrierMode::WaitAligned,
    ]);
    let actual_modes: BTreeSet<_> = admission
        .variants
        .iter()
        .map(|variant| variant.mode)
        .collect();
    ensure!(
        admission.variants.len() == expected_modes.len() && actual_modes == expected_modes,
        "compact cluster-barrier admission must contain each reviewed mode exactly once"
    );

    admission
        .variants
        .iter()
        .map(|variant| {
            let recipe = cluster_barrier_recipe(variant.mode);
            ensure!(
                variant.abi_id == recipe.abi_id,
                "{} must keep reserved ABI ID {}",
                recipe.id,
                recipe.abi_id
            );
            let source_record = format!("int_nvvm_barrier_cluster_{}", recipe.suffix.replace('.', "_"));
            let llvm_symbol = format!("llvm.nvvm.barrier.cluster.{}", recipe.suffix);
            let modifiers: Vec<String> = recipe.suffix.split('.').map(str::to_owned).collect();
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: variant.abi_id.clone(),
                operation_key: format!("cluster.barrier.{}", recipe.suffix),
                family: "cluster_barrier".into(),
                source: None,
                source_record: Some(source_record),
                rust_module: "cluster".into(),
                rust_name: recipe.id.into(),
                rust_arguments: vec![],
                rust_result: "()".into(),
                safe: false,
                must_use: false,
                safe_allowlist_reason: None,
                public_rust_path: format!("cuda_intrinsics::cluster::{}", recipe.id),
                compatibility_rust_paths: vec![format!("cuda_device::cluster::{}", recipe.id)],
                dialect_op_type: "ClusterBarrierOp".into(),
                dialect_op_name: "nvvm.cluster_barrier".into(),
                dialect_operands: vec![],
                dialect_results: vec![],
                llvm_symbol: Some(llvm_symbol),
                resolved_llvm_symbol: None,
                llvm_arguments: vec![],
                llvm_results: vec![],
                pure: false,
                memory: "read_write".into(),
                convergent: true,
                execution_scope: "cluster".into(),
                minimum_ptx: recipe.minimum_ptx.into(),
                minimum_sm: Some("sm_90".into()),
                ptx_result: "()".into(),
                targets: "all".into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section:
                    "Parallel Synchronization and Communication Instructions: barrier.cluster"
                        .into(),
                ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster".into(),
                lowering: "generated_cluster_barrier".into(),
                backend_lowerings: vec![
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LlvmNvptx,
                        mechanism: BackendLoweringMechanism::TypedNvvm,
                        evidence_profile: admission.llvm_evidence_profile.clone(),
                        minimum_ptx: Some(recipe.minimum_ptx.into()),
                        minimum_sm: Some("sm_90".into()),
                    },
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LibNvvm,
                        mechanism: BackendLoweringMechanism::InlinePtx,
                        evidence_profile: admission.libnvvm_evidence_profile.clone(),
                        minimum_ptx: Some(recipe.minimum_ptx.into()),
                        minimum_sm: Some("sm_90".into()),
                    },
                ],
                packed_atomic: None,
                redux: None,
                vote: None,
                active_mask: None,
                warp_match: None,
                warp_barrier: None,
                warp_shuffle: None,
                dot_product: None,
                packed_alu: None,
                packed_conversion: None,
                cp_async_copy: None,
                cp_async_control: None,
                cp_async_mbarrier: None,
                mbarrier_basic: None,
                movmatrix: None,
                mbarrier_extended: None,
                register_mma: None,
                sparse_mma: None,
                prmt: None,
                cluster_barrier: Some(ClusterBarrier {
                    mode: recipe.mode,
                    ordering: recipe.ordering,
                    aligned: recipe.aligned,
                }),
                special_register: None,
                debug_control: None,
                cluster_memory: None,
                clc: None,
                ldmatrix_variant: None,
                ldmatrix_safety: None,
                ldmatrix_adapter: None,
                selected_address_space: None,
                expected_ptx: InstructionPattern {
                    mnemonic: "barrier".into(),
                    modifiers: std::iter::once("cluster".into()).chain(modifiers).collect(),
                    operands: vec![],
                },
                summary: recipe.summary.into(),
            })
        })
        .collect()
}

fn validate_cluster_barrier_policy(
    policy: &OverlayIntrinsic,
    declaration: &ImportedIntrinsic,
) -> Result<()> {
    let barrier = policy
        .cluster_barrier
        .as_ref()
        .with_context(|| format!("{} has no closed cluster-barrier contract", policy.id))?;
    let recipe = cluster_barrier_recipe(barrier.mode);
    let source_record = format!(
        "int_nvvm_barrier_cluster_{}",
        recipe.suffix.replace('.', "_")
    );
    let llvm_symbol = format!("llvm.nvvm.barrier.cluster.{}", recipe.suffix);
    ensure!(
        barrier.ordering == recipe.ordering
            && barrier.aligned == recipe.aligned
            && policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == format!("cluster.barrier.{}", recipe.suffix)
            && policy.source.is_none()
            && policy.source_record.as_deref() == Some(source_record.as_str())
            && policy.llvm_symbol.as_deref() == Some(llvm_symbol.as_str())
            && policy.resolved_llvm_symbol.is_none(),
        "{} identity or semantics do not match its closed cluster-barrier recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "cluster"
            && policy.rust_name == recipe.id
            && policy.rust_arguments.is_empty()
            && policy.rust_result == "()"
            && !policy.safe
            && !policy.must_use
            && policy.public_rust_path == format!("cuda_intrinsics::cluster::{}", recipe.id)
            && policy.compatibility_rust_paths == [format!("cuda_device::cluster::{}", recipe.id)],
        "{} Rust API does not match its closed cluster-barrier recipe",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == "ClusterBarrierOp"
            && policy.dialect_op_name == "nvvm.cluster_barrier"
            && policy.dialect_operands.is_empty()
            && policy.dialect_results.is_empty()
            && policy.llvm_arguments.is_empty()
            && policy.llvm_results.is_empty()
            && policy.lowering == "generated_cluster_barrier",
        "{} carrier or lowering does not match its closed cluster-barrier recipe",
        policy.id
    );
    ensure!(
        declaration.classes == ["SDPatternOperator", "Intrinsic"]
            && declaration.properties == ["IntrConvergent", "IntrNoCallback"]
            && !policy.pure
            && policy.memory == "read_write"
            && policy.convergent
            && policy.execution_scope == "cluster",
        "{} effects disagree with the imported cluster-barrier declaration",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == recipe.minimum_ptx
            && policy.minimum_sm.as_deref() == Some("sm_90")
            && policy.targets == "all"
            && policy.ptx_result == "()"
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section
                == "Parallel Synchronization and Communication Instructions: barrier.cluster"
            && policy.ptx_isa_url
                == "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-barrier-cluster",
        "{} target floor or PTX provenance changed",
        policy.id
    );
    let expected_modifiers: Vec<_> = std::iter::once("cluster")
        .chain(recipe.suffix.split('.'))
        .collect();
    ensure!(
        policy.expected_ptx.mnemonic == "barrier"
            && policy.expected_ptx.modifiers == expected_modifiers
            && policy.expected_ptx.operands.is_empty(),
        "{} expected PTX does not match its exact cluster-barrier spelling",
        policy.id
    );
    ensure!(
        (recipe.ordering == ClusterBarrierOrdering::Relaxed) == recipe.suffix.contains(".relaxed")
            && recipe.aligned == recipe.suffix.ends_with(".aligned")
            && matches!(
                (recipe.mode, recipe.ordering),
                (
                    ClusterBarrierMode::Arrive | ClusterBarrierMode::ArriveAligned,
                    ClusterBarrierOrdering::Release
                ) | (
                    ClusterBarrierMode::ArriveRelaxed | ClusterBarrierMode::ArriveRelaxedAligned,
                    ClusterBarrierOrdering::Relaxed
                ) | (
                    ClusterBarrierMode::Wait | ClusterBarrierMode::WaitAligned,
                    ClusterBarrierOrdering::Acquire
                )
            ),
        "{} cluster-barrier semantic recipe is inconsistent",
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
                ])
            && policy.backend_lowerings.iter().all(|lowering| {
                lowering.minimum_ptx.as_deref() == Some(recipe.minimum_ptx)
                    && lowering.minimum_sm.as_deref() == Some("sm_90")
                    && !lowering.evidence_profile.trim().is_empty()
            }),
        "{} must define exactly the reviewed cluster-barrier backend routes",
        policy.id
    );
    ensure_no_other_family_contract(policy, "cluster barrier")?;
    Ok(())
}

#[derive(Clone, Copy)]
struct DebugControlRecipe {
    id: &'static str,
    operation_key: &'static str,
    rust_name: &'static str,
    rust_arguments: &'static [&'static str],
    rust_result: &'static str,
    compatibility_path: &'static str,
    op_type: &'static str,
    op_name: &'static str,
    instruction: &'static str,
    minimum_ptx: &'static str,
    minimum_sm: Option<&'static str>,
    section: &'static str,
    anchor: &'static str,
    adapter: DebugControlAdapter,
    summary: &'static str,
}

fn debug_control_recipe(operation: DebugControlOperation) -> DebugControlRecipe {
    match operation {
        DebugControlOperation::Trap => DebugControlRecipe {
            id: "trap",
            operation_key: "debug.control.trap",
            rust_name: "trap",
            rust_arguments: &[],
            rust_result: "!",
            compatibility_path: "cuda_device::debug::trap",
            op_type: "TrapOp",
            op_name: "nvvm.trap",
            instruction: "trap",
            minimum_ptx: "1.0",
            minimum_sm: None,
            section: "9.7.20.4 Miscellaneous Instructions: trap",
            anchor: "miscellaneous-instructions-trap",
            adapter: DebugControlAdapter::Direct,
            summary: "Aborts device execution and reports an interrupt to the host.",
        },
        DebugControlOperation::Breakpoint => DebugControlRecipe {
            id: "breakpoint",
            operation_key: "debug.control.breakpoint",
            rust_name: "breakpoint",
            rust_arguments: &[],
            rust_result: "()",
            compatibility_path: "cuda_device::debug::breakpoint",
            op_type: "BreakpointOp",
            op_name: "nvvm.brkpt",
            instruction: "brkpt",
            minimum_ptx: "1.0",
            minimum_sm: Some("sm_11"),
            section: "9.7.20.1 Miscellaneous Instructions: brkpt",
            anchor: "miscellaneous-instructions-brkpt",
            adapter: DebugControlAdapter::Direct,
            summary: "Suspends device execution for a debugger breakpoint.",
        },
        DebugControlOperation::Pmevent => DebugControlRecipe {
            id: "pmevent",
            operation_key: "debug.profiler.event",
            rust_name: "pmevent",
            rust_arguments: &["u32"],
            rust_result: "()",
            compatibility_path: "cuda_device::debug::__prof_trigger",
            op_type: "PmEventOp",
            op_name: "nvvm.pmevent",
            instruction: "pmevent",
            minimum_ptx: "1.4",
            minimum_sm: None,
            section: "9.7.20.3 Miscellaneous Instructions: pmevent",
            anchor: "miscellaneous-instructions-pmevent",
            adapter: DebugControlAdapter::ConstGenericToImmediateU32,
            summary: "Triggers one compile-time-selected performance monitor event.",
        },
    }
}

fn expand_debug_control_admission(
    admission: &DebugControlAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "debug-control runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "compact debug-control admission requires both backend evidence profiles"
    );
    ensure!(
        admission.operations
            == [
                DebugControlOperation::Trap,
                DebugControlOperation::Breakpoint,
                DebugControlOperation::Pmevent,
            ],
        "compact debug-control admission must list trap, breakpoint, and pmevent exactly once in canonical order"
    );
    ensure!(
        admission.abi_ids.len() == admission.operations.len(),
        "pending debug-control admission needs exactly three ABI IDs before aggregation"
    );
    let unique_abi_ids = admission.abi_ids.iter().collect::<BTreeSet<_>>();
    ensure!(
        unique_abi_ids.len() == admission.abi_ids.len(),
        "debug-control ABI IDs must be unique"
    );

    admission
        .operations
        .iter()
        .zip(&admission.abi_ids)
        .map(|(&operation, abi_id)| {
            validate_abi_id(abi_id)?;
            let recipe = debug_control_recipe(operation);
            let immediate = operation == DebugControlOperation::Pmevent;
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: abi_id.clone(),
                operation_key: recipe.operation_key.into(),
                family: "debug_control".into(),
                source: Some(IntrinsicSource::PtxNative {
                    instruction: recipe.instruction.into(),
                }),
                source_record: None,
                rust_module: "debug".into(),
                rust_name: recipe.rust_name.into(),
                rust_arguments: recipe
                    .rust_arguments
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                rust_result: recipe.rust_result.into(),
                safe: true,
                must_use: false,
                safe_allowlist_reason: Some(
                    match operation {
                        DebugControlOperation::Trap => {
                            "aborting the kernel has no memory-safety preconditions"
                        }
                        DebugControlOperation::Breakpoint => {
                            "requesting a debugger breakpoint has no memory-safety preconditions"
                        }
                        DebugControlOperation::Pmevent => {
                            "the importer accepts only the documented immediate event range"
                        }
                    }
                    .into(),
                ),
                public_rust_path: format!("cuda_intrinsics::debug::{}", recipe.rust_name),
                compatibility_rust_paths: vec![recipe.compatibility_path.into()],
                dialect_op_type: recipe.op_type.into(),
                dialect_op_name: recipe.op_name.into(),
                dialect_operands: vec![],
                dialect_results: vec![],
                llvm_symbol: None,
                resolved_llvm_symbol: None,
                llvm_arguments: vec![],
                llvm_results: vec![],
                pure: false,
                memory: "none".into(),
                convergent: false,
                execution_scope: "thread".into(),
                minimum_ptx: recipe.minimum_ptx.into(),
                minimum_sm: recipe.minimum_sm.map(Into::into),
                ptx_result: "()".into(),
                targets: "all".into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section: recipe.section.into(),
                ptx_isa_url: format!(
                    "https://docs.nvidia.com/cuda/parallel-thread-execution/#{}",
                    recipe.anchor
                ),
                lowering: "generated_debug_control".into(),
                backend_lowerings: vec![
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LlvmNvptx,
                        mechanism: BackendLoweringMechanism::InlinePtx,
                        evidence_profile: admission.llvm_evidence_profile.clone(),
                        minimum_ptx: Some("3.2".into()),
                        minimum_sm: Some("sm_20".into()),
                    },
                    OverlayBackendLowering {
                        backend: IntrinsicBackend::LibNvvm,
                        mechanism: BackendLoweringMechanism::InlinePtx,
                        evidence_profile: admission.libnvvm_evidence_profile.clone(),
                        minimum_ptx: Some("9.3".into()),
                        minimum_sm: Some("sm_75".into()),
                    },
                ],
                packed_atomic: None,
                redux: None,
                vote: None,
                active_mask: None,
                warp_match: None,
                warp_barrier: None,
                warp_shuffle: None,
                dot_product: None,
                packed_alu: None,
                packed_conversion: None,
                cp_async_copy: None,
                cp_async_control: None,
                cp_async_mbarrier: None,
                mbarrier_basic: None,
                movmatrix: None,
                mbarrier_extended: None,
                register_mma: None,
                sparse_mma: None,
                prmt: None,
                cluster_barrier: None,
                special_register: None,
                debug_control: Some(DebugControl {
                    operation,
                    adapter: recipe.adapter,
                    runtime_validation: admission.runtime_validation,
                }),
                cluster_memory: None,
                clc: None,
                ldmatrix_variant: None,
                ldmatrix_safety: None,
                ldmatrix_adapter: None,
                selected_address_space: None,
                expected_ptx: InstructionPattern {
                    mnemonic: recipe.instruction.into(),
                    modifiers: vec![],
                    operands: if immediate {
                        vec![OperandPattern::Immediate]
                    } else {
                        vec![]
                    },
                },
                summary: recipe.summary.into(),
            })
        })
        .collect()
}

fn validate_debug_control_policy(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
) -> Result<()> {
    let debug = policy
        .debug_control
        .as_ref()
        .with_context(|| format!("{} has no closed debug-control contract", policy.id))?;
    let recipe = debug_control_recipe(debug.operation);
    let immediate = debug.operation == DebugControlOperation::Pmevent;
    ensure!(
        debug.adapter == recipe.adapter
            && debug.runtime_validation == RuntimeValidation::Unexecuted
            && policy.id == recipe.id
            && policy.operation_key == recipe.operation_key
            && source
                == &IntrinsicSource::PtxNative {
                    instruction: recipe.instruction.into(),
                }
            && policy.source_record.is_none()
            && policy.llvm_symbol.is_none()
            && policy.resolved_llvm_symbol.is_none()
            && policy.llvm_arguments.is_empty()
            && policy.llvm_results.is_empty(),
        "{} identity must remain PTX-native and match its closed debug-control recipe",
        policy.id
    );
    ensure!(
        policy.rust_module == "debug"
            && policy.rust_name == recipe.rust_name
            && policy.rust_arguments == recipe.rust_arguments
            && policy.rust_result == recipe.rust_result
            && policy.safe
            && !policy.must_use
            && policy.public_rust_path == format!("cuda_intrinsics::debug::{}", recipe.rust_name)
            && policy.compatibility_rust_paths == [recipe.compatibility_path],
        "{} Rust API or compatibility adapter changed",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.op_type
            && policy.dialect_op_name == recipe.op_name
            && policy.dialect_operands.is_empty()
            && policy.dialect_results.is_empty()
            && policy.lowering == "generated_debug_control",
        "{} dialect carrier or lowering changed",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "none"
            && !policy.convergent
            && policy.execution_scope == "thread",
        "{} debug-control effects changed",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == recipe.minimum_ptx
            && policy.minimum_sm.as_deref() == recipe.minimum_sm
            && policy.targets == "all"
            && policy.ptx_result == "()"
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == recipe.section
            && policy.ptx_isa_url
                == format!(
                    "https://docs.nvidia.com/cuda/parallel-thread-execution/#{}",
                    recipe.anchor
                ),
        "{} native target floor or PTX provenance changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx.mnemonic == recipe.instruction
            && policy.expected_ptx.modifiers.is_empty()
            && policy.expected_ptx.operands
                == if immediate {
                    vec![OperandPattern::Immediate]
                } else {
                    vec![]
                },
        "{} expected PTX changed",
        policy.id
    );
    ensure_exact_inline_ptx_backends(
        policy,
        [
            (IntrinsicBackend::LlvmNvptx, "3.2", Some("sm_20")),
            (IntrinsicBackend::LibNvvm, "9.3", Some("sm_75")),
        ],
        "debug-control",
    )?;
    ensure_no_other_family_contract(policy, "debug-control")?;
    Ok(())
}

#[derive(Clone)]
struct ClusterMemoryRecipe {
    operation: ClusterMemoryOperation,
    abi_id: &'static str,
    id: &'static str,
    operation_key: &'static str,
    source_record: Option<&'static str>,
    llvm_symbol: Option<&'static str>,
    ptx_native_instruction: Option<&'static str>,
    rust_arguments: &'static [&'static str],
    rust_result: &'static str,
    compatibility_paths: &'static [&'static str],
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    dialect_operands: &'static [&'static str],
    dialect_results: &'static [&'static str],
    llvm_arguments: &'static [&'static str],
    llvm_results: &'static [&'static str],
    adapter: ClusterMemoryAdapter,
    source_contract: ClusterMemorySourceContract,
    memory: &'static str,
    expected_ptx: InstructionPattern,
    inline_ptx: &'static str,
    inline_constraints: &'static str,
    ptx_isa_section: &'static str,
    ptx_isa_anchor: &'static str,
    summary: &'static str,
}

fn cluster_memory_recipe(operation: ClusterMemoryOperation) -> ClusterMemoryRecipe {
    match operation {
        ClusterMemoryOperation::MapSharedRank => ClusterMemoryRecipe {
            operation,
            abi_id: "i0320",
            id: "map_shared_rank",
            operation_key: "cluster.shared_address.map_rank",
            source_record: Some("int_nvvm_mapa_shared_cluster"),
            llvm_symbol: Some("llvm.nvvm.mapa.shared.cluster"),
            ptx_native_instruction: None,
            rust_arguments: &["*const u8", "u32"],
            rust_result: "*const u8",
            compatibility_paths: &[
                "cuda_device::cluster::map_shared_rank",
                "cuda_device::cluster::map_shared_rank_mut",
            ],
            dialect_op_type: "MapaSharedClusterOp",
            dialect_op_name: "nvvm.mapa_shared_cluster",
            dialect_operands: &["ptr", "i32"],
            dialect_results: &["ptr"],
            llvm_arguments: &["shared_ptr", "i32"],
            llvm_results: &["shared_cluster_ptr"],
            adapter: ClusterMemoryAdapter::GenericConstAndMutPointerRankToSamePointer,
            source_contract: ClusterMemorySourceContract::LlvmMapaSharedClusterAs7IdentityInlinePtx,
            memory: "none",
            expected_ptx: InstructionPattern {
                mnemonic: "mapa".into(),
                modifiers: vec!["shared::cluster".into(), "u64".into()],
                operands: vec![
                    OperandPattern::Register,
                    OperandPattern::Register,
                    OperandPattern::Register,
                ],
            },
            inline_ptx: "mapa.shared::cluster.u64 $0, $1, $2;",
            inline_constraints: "=l,l,r",
            ptx_isa_section: "9.7.9.24 Data Movement and Conversion Instructions: mapa",
            ptx_isa_anchor: "data-movement-and-conversion-instructions-mapa",
            summary: "Maps a CTA-shared address to the same offset in another cluster rank.",
        },
        ClusterMemoryOperation::ReadU32 => ClusterMemoryRecipe {
            operation,
            abi_id: "i0321",
            id: "dsmem_read_u32",
            operation_key: "cluster.shared_memory.map_rank_then_read_u32",
            source_record: None,
            llvm_symbol: None,
            ptx_native_instruction: Some("mapa.shared::cluster.u64 + ld.shared::cluster.u32"),
            rust_arguments: &["*const u32", "u32"],
            rust_result: "u32",
            compatibility_paths: &["cuda_device::cluster::dsmem_read_u32"],
            dialect_op_type: "DsmemReadU32Op",
            dialect_op_name: "nvvm.dsmem_read_u32",
            dialect_operands: &["ptr", "i32"],
            dialect_results: &["i32"],
            llvm_arguments: &[],
            llvm_results: &[],
            adapter: ClusterMemoryAdapter::ConstU32PointerRankToU32,
            source_contract: ClusterMemorySourceContract::PtxNativeMapaThenWeakClusterLoad,
            memory: "read",
            expected_ptx: InstructionPattern {
                mnemonic: "ld".into(),
                modifiers: vec!["shared::cluster".into(), "u32".into()],
                operands: vec![OperandPattern::Register, OperandPattern::Address],
            },
            inline_ptx: "{ .reg .u64 %mapped; mapa.shared::cluster.u64 %mapped, $1, $2; ld.shared::cluster.u32 $0, [%mapped]; }",
            inline_constraints: "=r,l,r,~{memory}",
            ptx_isa_section: "9.7.9.8 Data Movement and Conversion Instructions: ld",
            ptx_isa_anchor: "data-movement-and-conversion-instructions-ld",
            summary: "Maps a CTA-shared address to another cluster rank and reads one weak u32 value.",
        },
    }
}

pub(crate) fn cluster_memory_inline_recipe(
    operation: ClusterMemoryOperation,
) -> (&'static str, &'static str) {
    let recipe = cluster_memory_recipe(operation);
    (recipe.inline_ptx, recipe.inline_constraints)
}

#[cfg(test)]
pub(crate) fn test_cluster_memory_records(template: &CatalogIntrinsic) -> Vec<CatalogIntrinsic> {
    [
        ClusterMemoryOperation::MapSharedRank,
        ClusterMemoryOperation::ReadU32,
    ]
    .into_iter()
    .map(|operation| {
        let recipe = cluster_memory_recipe(operation);
        let minimum_ptx = "7.8".parse::<PtxVersion>().unwrap();
        let hardware = CatalogHardwareTarget::AnyOf {
            alternatives: vec![CatalogHardwareAlternative::MinimumSm { sm: 90 }],
        };
        let mut record = template.clone();
        record.id = recipe.id.into();
        record.operation_key = recipe.operation_key.into();
        record.family = "cluster_memory".into();
        record.source = match (recipe.source_record, recipe.ptx_native_instruction) {
            (Some(source_record), None) => IntrinsicSource::LlvmImported {
                source_record: source_record.into(),
            },
            (None, Some(instruction)) => IntrinsicSource::PtxNative {
                instruction: instruction.into(),
            },
            _ => unreachable!("closed cluster-memory recipe has one source"),
        };
        record.selections = if operation == ClusterMemoryOperation::MapSharedRank {
            ["mapa_shared_cluster_64", "mapa_shared_cluster_64i"]
                .into_iter()
                .map(|source_record| CatalogSelection {
                    source_record: source_record.into(),
                    asm: "mapa.shared::cluster.u64 \t$d, $a, $b;".into(),
                    predicates: vec![
                        "Subtarget->getSmVersion() >= 90".into(),
                        "Subtarget->getPTXVersion() >= 78".into(),
                    ],
                    constraints: Default::default(),
                })
                .collect()
        } else {
            vec![]
        };
        record.rust = CatalogRust {
            abi_id: recipe.abi_id.into(),
            module: "cluster".into(),
            name: recipe.id.into(),
            arguments: recipe
                .rust_arguments
                .iter()
                .map(|value| (*value).into())
                .collect(),
            result: recipe.rust_result.into(),
            safe: false,
            must_use: true,
            safe_allowlist_reason: None,
            canonical_path: canonical_rust_path(1, recipe.abi_id),
            public_path: format!("cuda_intrinsics::cluster::{}", recipe.id),
            compatibility_paths: recipe
                .compatibility_paths
                .iter()
                .map(|path| (*path).into())
                .collect(),
        };
        record.dialect = CatalogDialect {
            op_type: recipe.dialect_op_type.into(),
            op_name: recipe.dialect_op_name.into(),
            operands: recipe
                .dialect_operands
                .iter()
                .map(|value| (*value).into())
                .collect(),
            results: recipe
                .dialect_results
                .iter()
                .map(|value| (*value).into())
                .collect(),
        };
        record.llvm = recipe.llvm_symbol.map(|symbol| CatalogLlvm {
            symbol: symbol.into(),
            resolved_symbol: None,
            arguments: recipe
                .llvm_arguments
                .iter()
                .map(|value| (*value).into())
                .collect(),
            results: recipe
                .llvm_results
                .iter()
                .map(|value| (*value).into())
                .collect(),
            properties: vec![
                "IntrNoMem".into(),
                "IntrSpeculatable".into(),
                "NoCapture<arg0>".into(),
            ],
            result_facts: CatalogLlvmResultFacts {
                no_undef: false,
                range: None,
            },
        });
        record.semantics = CatalogSemantics {
            pure: false,
            memory: recipe.memory.into(),
            convergent: true,
            execution_scope: "cluster".into(),
        };
        record.target = CatalogTarget {
            minimum_ptx,
            hardware: hardware.clone(),
            ptx_result: recipe.rust_result.into(),
            targets: "all".into(),
            ptx_isa_version: "9.3".into(),
            ptx_isa_section: recipe.ptx_isa_section.into(),
            ptx_isa_url: format!(
                "https://docs.nvidia.com/cuda/parallel-thread-execution/#{}",
                recipe.ptx_isa_anchor
            ),
        };
        assert_eq!(record.backend_lowerings.len(), 2);
        for lowering in &mut record.backend_lowerings {
            lowering.mechanism = BackendLoweringMechanism::InlinePtx;
            lowering.target = CatalogTargetRequirement {
                minimum_ptx,
                hardware: hardware.clone(),
            };
        }
        record.mbarrier_basic = None;
        record.cluster_memory = Some(ClusterMemory {
            operation,
            adapter: recipe.adapter,
            source_contract: recipe.source_contract,
            runtime_validation: RuntimeValidation::Unexecuted,
        });
        record.lowering = "generated_cluster_memory_inline_ptx".into();
        record.expected_ptx = recipe.expected_ptx;
        record.summary = recipe.summary.into();
        record
    })
    .collect()
}

fn expand_cluster_memory_admission(
    admission: &ClusterMemoryAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "cluster-memory runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "compact cluster-memory admission requires both backend evidence profiles"
    );
    let expected = [
        ClusterMemoryOperation::MapSharedRank,
        ClusterMemoryOperation::ReadU32,
    ];
    ensure!(
        admission.variants.len() == expected.len()
            && admission
                .variants
                .iter()
                .map(|variant| variant.operation)
                .eq(expected),
        "compact cluster-memory admission must list map_shared_rank and read_u32 once in canonical order"
    );

    admission
        .variants
        .iter()
        .map(|variant| {
            let recipe = cluster_memory_recipe(variant.operation);
            ensure!(
                variant.abi_id == recipe.abi_id,
                "{} must keep reserved ABI ID {}",
                recipe.id,
                recipe.abi_id
            );
            let source =
                recipe
                    .ptx_native_instruction
                    .map(|instruction| IntrinsicSource::PtxNative {
                        instruction: instruction.into(),
                    });
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: variant.abi_id.clone(),
                operation_key: recipe.operation_key.into(),
                family: "cluster_memory".into(),
                source,
                source_record: recipe.source_record.map(Into::into),
                rust_module: "cluster".into(),
                rust_name: recipe.id.into(),
                rust_arguments: recipe
                    .rust_arguments
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                rust_result: recipe.rust_result.into(),
                safe: false,
                must_use: true,
                safe_allowlist_reason: None,
                public_rust_path: format!("cuda_intrinsics::cluster::{}", recipe.id),
                compatibility_rust_paths: recipe
                    .compatibility_paths
                    .iter()
                    .map(|path| (*path).into())
                    .collect(),
                dialect_op_type: recipe.dialect_op_type.into(),
                dialect_op_name: recipe.dialect_op_name.into(),
                dialect_operands: recipe
                    .dialect_operands
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                dialect_results: recipe
                    .dialect_results
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                llvm_symbol: recipe.llvm_symbol.map(Into::into),
                resolved_llvm_symbol: None,
                llvm_arguments: recipe
                    .llvm_arguments
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                llvm_results: recipe
                    .llvm_results
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                pure: false,
                memory: recipe.memory.into(),
                convergent: true,
                execution_scope: "cluster".into(),
                minimum_ptx: "7.8".into(),
                minimum_sm: Some("sm_90".into()),
                ptx_result: recipe.rust_result.into(),
                targets: "all".into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section: recipe.ptx_isa_section.into(),
                ptx_isa_url: format!(
                    "https://docs.nvidia.com/cuda/parallel-thread-execution/#{}",
                    recipe.ptx_isa_anchor
                ),
                lowering: "generated_cluster_memory_inline_ptx".into(),
                backend_lowerings: [IntrinsicBackend::LlvmNvptx, IntrinsicBackend::LibNvvm]
                    .into_iter()
                    .map(|backend| OverlayBackendLowering {
                        backend,
                        mechanism: BackendLoweringMechanism::InlinePtx,
                        evidence_profile: match backend {
                            IntrinsicBackend::LlvmNvptx => admission.llvm_evidence_profile.clone(),
                            IntrinsicBackend::LibNvvm => admission.libnvvm_evidence_profile.clone(),
                        },
                        minimum_ptx: Some("7.8".into()),
                        minimum_sm: Some("sm_90".into()),
                    })
                    .collect(),
                packed_atomic: None,
                redux: None,
                vote: None,
                active_mask: None,
                warp_match: None,
                warp_barrier: None,
                warp_shuffle: None,
                dot_product: None,
                packed_alu: None,
                packed_conversion: None,
                cp_async_copy: None,
                cp_async_control: None,
                cp_async_mbarrier: None,
                mbarrier_basic: None,
                movmatrix: None,
                mbarrier_extended: None,
                register_mma: None,
                sparse_mma: None,
                prmt: None,
                cluster_barrier: None,
                special_register: None,
                debug_control: None,
                cluster_memory: Some(ClusterMemory {
                    operation: recipe.operation,
                    adapter: recipe.adapter,
                    source_contract: recipe.source_contract,
                    runtime_validation: admission.runtime_validation,
                }),
                clc: None,
                ldmatrix_variant: None,
                ldmatrix_safety: None,
                ldmatrix_adapter: None,
                selected_address_space: None,
                expected_ptx: recipe.expected_ptx,
                summary: recipe.summary.into(),
            })
        })
        .collect()
}

fn validate_cluster_memory_policy(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
    declaration: Option<&ImportedIntrinsic>,
) -> Result<()> {
    let cluster = policy
        .cluster_memory
        .as_ref()
        .with_context(|| format!("{} has no closed cluster-memory contract", policy.id))?;
    let recipe = cluster_memory_recipe(cluster.operation);
    ensure!(
        cluster.adapter == recipe.adapter
            && cluster.source_contract == recipe.source_contract
            && cluster.runtime_validation == RuntimeValidation::Unexecuted
            && policy.id == recipe.id
            && policy.operation_key == recipe.operation_key,
        "{} does not match its closed cluster-memory identity",
        policy.id
    );
    match cluster.source_contract {
        ClusterMemorySourceContract::LlvmMapaSharedClusterAs7IdentityInlinePtx => {
            let declaration = declaration.context("map_shared_rank requires its LLVM identity")?;
            ensure!(
                matches!(source, IntrinsicSource::LlvmImported { source_record }
                    if source_record == "int_nvvm_mapa_shared_cluster")
                    && policy.source.is_none()
                    && policy.source_record.as_deref() == recipe.source_record
                    && policy.llvm_symbol.as_deref() == recipe.llvm_symbol
                    && declaration.arguments == ["shared_ptr", "i32"]
                    && declaration.results == ["shared_cluster_ptr"]
                    && declaration.properties
                        == ["IntrNoMem", "IntrSpeculatable", "NoCapture<arg0>"],
                "{} must retain the AS7-returning LLVM mapa record as identity only",
                policy.id
            );
            let selections = declaration
                .selections
                .iter()
                .filter(|selection| selection_matches_policy(policy, selection))
                .collect::<Vec<_>>();
            ensure!(
                selections.len() == 2
                    && selections
                        .iter()
                        .map(|selection| selection.source_record.as_str())
                        .collect::<BTreeSet<_>>()
                        == BTreeSet::from(["mapa_shared_cluster_64", "mapa_shared_cluster_64i",])
                    && selections.iter().all(|selection| {
                        selection.asm == "mapa.shared::cluster.u64 \t$d, $a, $b;"
                            && selection.predicates
                                == [
                                    "Subtarget->getSmVersion() >= 90",
                                    "Subtarget->getPTXVersion() >= 78",
                                ]
                    }),
                "{} must retain both exact 64-bit mapa selections",
                policy.id
            );
        }
        ClusterMemorySourceContract::PtxNativeMapaThenWeakClusterLoad => ensure!(
            matches!(source, IntrinsicSource::PtxNative { instruction }
                if Some(instruction.as_str()) == recipe.ptx_native_instruction)
                && policy.source_record.is_none()
                && policy.llvm_symbol.is_none()
                && declaration.is_none()
                && policy.llvm_arguments.is_empty()
                && policy.llvm_results.is_empty(),
            "{} must remain a PTX-native mapa plus weak cluster-load composite",
            policy.id
        ),
    }
    ensure!(
        policy.rust_module == "cluster"
            && policy.rust_name == recipe.id
            && policy.rust_arguments == recipe.rust_arguments
            && policy.rust_result == recipe.rust_result
            && !policy.safe
            && policy.must_use
            && policy.public_rust_path == format!("cuda_intrinsics::cluster::{}", recipe.id)
            && policy.compatibility_rust_paths == recipe.compatibility_paths,
        "{} Rust API or compatibility paths changed",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == recipe.dialect_operands
            && policy.dialect_results == recipe.dialect_results
            && policy.llvm_arguments == recipe.llvm_arguments
            && policy.llvm_results == recipe.llvm_results
            && policy.lowering == "generated_cluster_memory_inline_ptx",
        "{} dialect carrier or AS7 source boundary changed",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == recipe.memory
            && policy.convergent
            && policy.execution_scope == "cluster",
        "{} effects changed",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == "7.8"
            && policy.minimum_sm.as_deref() == Some("sm_90")
            && policy.targets == "all"
            && policy.ptx_result == recipe.rust_result
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == recipe.ptx_isa_section
            && policy.ptx_isa_url
                == format!(
                    "https://docs.nvidia.com/cuda/parallel-thread-execution/#{}",
                    recipe.ptx_isa_anchor
                ),
        "{} PTX provenance or target floor changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx == recipe.expected_ptx,
        "{} expected PTX changed",
        policy.id
    );
    ensure_exact_inline_ptx_backends(
        policy,
        [
            (IntrinsicBackend::LlvmNvptx, "7.8", Some("sm_90")),
            (IntrinsicBackend::LibNvvm, "7.8", Some("sm_90")),
        ],
        "cluster-memory",
    )?;
    ensure_no_other_family_contract(policy, "cluster-memory")?;
    Ok(())
}

#[derive(Clone)]
struct MbarrierExtendedRecipe {
    operation: MbarrierExtendedOperation,
    abi_id: &'static str,
    id: &'static str,
    operation_key: &'static str,
    source_record: Option<&'static str>,
    llvm_symbol: Option<&'static str>,
    ptx_native_instruction: Option<&'static str>,
    rust_arguments: &'static [&'static str],
    rust_result: &'static str,
    must_use: bool,
    dialect_op_type: &'static str,
    dialect_op_name: &'static str,
    dialect_operands: &'static [&'static str],
    dialect_results: &'static [&'static str],
    llvm_arguments: &'static [&'static str],
    llvm_results: &'static [&'static str],
    llvm_properties: &'static [&'static str],
    adapter: MbarrierExtendedAdapter,
    source_contract: MbarrierExtendedSourceContract,
    execution_scope: &'static str,
    minimum_ptx: &'static str,
    minimum_sm: &'static str,
    ptx_result: &'static str,
    expected_ptx: InstructionPattern,
    inline_ptx: &'static str,
    inline_constraints: &'static str,
    ptx_isa_section: &'static str,
    ptx_isa_url: &'static str,
    summary: &'static str,
}

fn mbarrier_extended_recipe(operation: MbarrierExtendedOperation) -> MbarrierExtendedRecipe {
    let instruction = |modifiers: &[&str], operands| InstructionPattern {
        mnemonic: if modifiers.first() == Some(&"nanosleep") {
            "nanosleep".into()
        } else if modifiers.first() == Some(&"fence") {
            "fence".into()
        } else {
            "mbarrier".into()
        },
        modifiers: modifiers[1..].iter().map(|value| (*value).into()).collect(),
        operands,
    };
    match operation {
        MbarrierExtendedOperation::ArriveExpectTxCta => MbarrierExtendedRecipe {
            operation,
            abi_id: "i0306",
            id: "mbarrier_arrive_expect_tx",
            operation_key: "barrier.mbarrier.arrive.expect_tx.shared.cta.release.cta",
            source_record: Some("int_nvvm_mbarrier_arrive_expect_tx_scope_cta_space_cta"),
            llvm_symbol: Some("llvm.nvvm.mbarrier.arrive.expect.tx.scope.cta.space.cta"),
            ptx_native_instruction: None,
            rust_arguments: &["*const u64", "u32", "u32"],
            rust_result: "u64",
            must_use: true,
            dialect_op_type: "MbarrierArriveExpectTxSharedOp",
            dialect_op_name: "nvvm.mbarrier_arrive_expect_tx_shared",
            dialect_operands: &["ptr", "i32"],
            dialect_results: &["i64"],
            llvm_arguments: &["shared_ptr", "i32"],
            llvm_results: &["i64"],
            llvm_properties: &["IntrConvergent", "IntrNoCallback"],
            adapter: MbarrierExtendedAdapter::PointerTxCountBytesToTokenDroppingTxCount,
            source_contract: MbarrierExtendedSourceContract::LlvmImported,
            execution_scope: "cta",
            minimum_ptx: "8.0",
            minimum_sm: "sm_90",
            ptx_result: "u64",
            expected_ptx: instruction(
                &[
                    "mbarrier",
                    "arrive",
                    "expect_tx",
                    "release",
                    "cta",
                    "shared::cta",
                    "b64",
                ],
                vec![
                    OperandPattern::Register,
                    OperandPattern::Address,
                    OperandPattern::Register,
                ],
            ),
            inline_ptx: "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 $0, [$1], $2;",
            inline_constraints: "=l,l,r,~{memory}",
            ptx_isa_section: "Parallel Synchronization and Communication Instructions: mbarrier.arrive",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive",
            summary: "Arrives at a CTA-shared barrier and adds expected transaction bytes.",
        },
        MbarrierExtendedOperation::ArriveExpectTxCluster => MbarrierExtendedRecipe {
            operation,
            abi_id: "i0307",
            id: "mbarrier_arrive_expect_tx_cluster",
            operation_key: "barrier.mbarrier.arrive.expect_tx.shared.cta.relaxed.cluster",
            source_record: Some(
                "int_nvvm_mbarrier_arrive_expect_tx_relaxed_scope_cluster_space_cta",
            ),
            llvm_symbol: Some(
                "llvm.nvvm.mbarrier.arrive.expect.tx.relaxed.scope.cluster.space.cta",
            ),
            ptx_native_instruction: None,
            rust_arguments: &["*const u64", "u32", "u32"],
            rust_result: "u64",
            must_use: true,
            dialect_op_type: "MbarrierArriveExpectTxClusterOp",
            dialect_op_name: "nvvm.mbarrier_arrive_expect_tx_cluster",
            dialect_operands: &["ptr", "i32"],
            dialect_results: &["i64"],
            llvm_arguments: &["shared_ptr", "i32"],
            llvm_results: &["i64"],
            llvm_properties: &["IntrArgMemOnly", "IntrConvergent", "IntrNoCallback"],
            adapter: MbarrierExtendedAdapter::PointerTxCountBytesToTokenDroppingTxCount,
            source_contract: MbarrierExtendedSourceContract::LlvmImported,
            execution_scope: "cluster",
            minimum_ptx: "8.6",
            minimum_sm: "sm_90",
            ptx_result: "u64",
            expected_ptx: instruction(
                &[
                    "mbarrier",
                    "arrive",
                    "expect_tx",
                    "relaxed",
                    "cluster",
                    "shared::cta",
                    "b64",
                ],
                vec![
                    OperandPattern::Register,
                    OperandPattern::Address,
                    OperandPattern::Register,
                ],
            ),
            inline_ptx: "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64 $0, [$1], $2;",
            inline_constraints: "=l,l,r,~{memory}",
            ptx_isa_section: "Parallel Synchronization and Communication Instructions: mbarrier.arrive",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive",
            summary: "Arrives at a CTA-shared barrier with cluster-scope transaction tracking.",
        },
        MbarrierExtendedOperation::ArriveRemoteCluster => MbarrierExtendedRecipe {
            operation,
            abi_id: "i0308",
            id: "mbarrier_arrive_cluster",
            operation_key: "barrier.mbarrier.arrive.shared.cluster.release.cluster.raw_address",
            source_record: None,
            llvm_symbol: None,
            ptx_native_instruction: Some("mbarrier.arrive.release.cluster.shared::cluster.b64"),
            rust_arguments: &["u64"],
            rust_result: "()",
            must_use: false,
            dialect_op_type: "MbarrierArriveClusterOp",
            dialect_op_name: "nvvm.mbarrier_arrive_cluster",
            dialect_operands: &["i64"],
            dialect_results: &[],
            llvm_arguments: &[],
            llvm_results: &[],
            llvm_properties: &[],
            adapter: MbarrierExtendedAdapter::RawClusterAddressToVoid,
            source_contract: MbarrierExtendedSourceContract::PtxNativeRawClusterAddress,
            execution_scope: "cluster",
            minimum_ptx: "8.0",
            minimum_sm: "sm_90",
            ptx_result: "()",
            expected_ptx: instruction(
                &[
                    "mbarrier",
                    "arrive",
                    "release",
                    "cluster",
                    "shared::cluster",
                    "b64",
                ],
                vec![
                    OperandPattern::Exact { value: "_".into() },
                    OperandPattern::Address,
                ],
            ),
            inline_ptx: "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$0];",
            inline_constraints: "l,~{memory}",
            ptx_isa_section: "Parallel Synchronization and Communication Instructions: mbarrier.arrive",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive",
            summary: "Arrives at a remote cluster-shared barrier through its raw address.",
        },
        MbarrierExtendedOperation::TryWaitTokenCta => MbarrierExtendedRecipe {
            operation,
            abi_id: "i0309",
            id: "mbarrier_try_wait",
            operation_key: "barrier.mbarrier.try_wait.shared.cta.token",
            source_record: Some("int_nvvm_mbarrier_try_wait_scope_cta_space_cta"),
            llvm_symbol: Some("llvm.nvvm.mbarrier.try.wait.scope.cta.space.cta"),
            ptx_native_instruction: None,
            rust_arguments: &["*const u64", "u64"],
            rust_result: "bool",
            must_use: true,
            dialect_op_type: "MbarrierTryWaitSharedOp",
            dialect_op_name: "nvvm.mbarrier_try_wait_shared",
            dialect_operands: &["ptr", "i64"],
            dialect_results: &["i1"],
            llvm_arguments: &["shared_ptr", "i64"],
            llvm_results: &["i1"],
            llvm_properties: &["IntrConvergent", "IntrNoCallback", "NoCapture<arg0>"],
            adapter: MbarrierExtendedAdapter::PointerTokenToPredicate,
            source_contract: MbarrierExtendedSourceContract::LlvmImported,
            execution_scope: "cta",
            minimum_ptx: "7.8",
            minimum_sm: "sm_90",
            ptx_result: "bool",
            expected_ptx: instruction(
                &["mbarrier", "try_wait", "shared", "b64"],
                vec![
                    OperandPattern::Register,
                    OperandPattern::Address,
                    OperandPattern::Register,
                ],
            ),
            inline_ptx: "{ .reg .pred %p0; mbarrier.try_wait.shared.b64 %p0, [$1], $2; selp.b32 $0, 1, 0, %p0; }",
            inline_constraints: "=r,l,l,~{memory}",
            ptx_isa_section: "Parallel Synchronization and Communication Instructions: mbarrier.test_wait / mbarrier.try_wait",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait",
            summary: "Tests a CTA-shared barrier token with a scheduling hint.",
        },
        MbarrierExtendedOperation::TryWaitParityCta => MbarrierExtendedRecipe {
            operation,
            abi_id: "i0310",
            id: "mbarrier_try_wait_parity",
            operation_key: "barrier.mbarrier.try_wait.parity.shared.cta",
            source_record: Some("int_nvvm_mbarrier_try_wait_parity_scope_cta_space_cta"),
            llvm_symbol: Some("llvm.nvvm.mbarrier.try.wait.parity.scope.cta.space.cta"),
            ptx_native_instruction: None,
            rust_arguments: &["*const u64", "u32"],
            rust_result: "bool",
            must_use: true,
            dialect_op_type: "MbarrierTryWaitParitySharedOp",
            dialect_op_name: "nvvm.mbarrier_try_wait_parity_shared",
            dialect_operands: &["ptr", "i32"],
            dialect_results: &["i1"],
            llvm_arguments: &["shared_ptr", "i32"],
            llvm_results: &["i1"],
            llvm_properties: &["IntrConvergent", "IntrNoCallback", "NoCapture<arg0>"],
            adapter: MbarrierExtendedAdapter::PointerParityToPredicate,
            source_contract: MbarrierExtendedSourceContract::LlvmImported,
            execution_scope: "cta",
            minimum_ptx: "7.8",
            minimum_sm: "sm_90",
            ptx_result: "bool",
            expected_ptx: instruction(
                &["mbarrier", "try_wait", "parity", "shared::cta", "b64"],
                vec![
                    OperandPattern::Register,
                    OperandPattern::Address,
                    OperandPattern::Register,
                ],
            ),
            inline_ptx: "{ .reg .pred %p0; mbarrier.try_wait.parity.shared::cta.b64 %p0, [$1], $2; selp.b32 $0, 1, 0, %p0; }",
            inline_constraints: "=r,l,r,~{memory}",
            ptx_isa_section: "Parallel Synchronization and Communication Instructions: mbarrier.test_wait / mbarrier.try_wait",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait",
            summary: "Tests a CTA-shared barrier phase by parity.",
        },
        MbarrierExtendedOperation::TryWaitParityCluster => MbarrierExtendedRecipe {
            operation,
            abi_id: "i0311",
            id: "mbarrier_try_wait_parity_cluster",
            operation_key: "barrier.mbarrier.try_wait.parity.shared.cta.acquire.cluster",
            source_record: Some("int_nvvm_mbarrier_try_wait_parity_scope_cluster_space_cta"),
            llvm_symbol: Some("llvm.nvvm.mbarrier.try.wait.parity.scope.cluster.space.cta"),
            ptx_native_instruction: None,
            rust_arguments: &["*const u64", "u32"],
            rust_result: "bool",
            must_use: true,
            dialect_op_type: "MbarrierTryWaitParityClusterOp",
            dialect_op_name: "nvvm.mbarrier_try_wait_parity_cluster",
            dialect_operands: &["ptr", "i32"],
            dialect_results: &["i1"],
            llvm_arguments: &["shared_ptr", "i32"],
            llvm_results: &["i1"],
            llvm_properties: &["IntrConvergent", "IntrNoCallback", "NoCapture<arg0>"],
            adapter: MbarrierExtendedAdapter::PointerParityToPredicate,
            source_contract: MbarrierExtendedSourceContract::LlvmImported,
            execution_scope: "cluster",
            minimum_ptx: "8.0",
            minimum_sm: "sm_90",
            ptx_result: "bool",
            expected_ptx: instruction(
                &[
                    "mbarrier",
                    "try_wait",
                    "parity",
                    "acquire",
                    "cluster",
                    "shared::cta",
                    "b64",
                ],
                vec![
                    OperandPattern::Register,
                    OperandPattern::Address,
                    OperandPattern::Register,
                ],
            ),
            inline_ptx: "{ .reg .pred %p0; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 %p0, [$1], $2; selp.b32 $0, 1, 0, %p0; }",
            inline_constraints: "=r,l,r,~{memory}",
            ptx_isa_section: "Parallel Synchronization and Communication Instructions: mbarrier.test_wait / mbarrier.try_wait",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-mbarrier-try-wait",
            summary: "Tests barrier parity with cluster-scope acquire ordering.",
        },
        MbarrierExtendedOperation::FenceProxyAsyncSharedCta => MbarrierExtendedRecipe {
            operation,
            abi_id: "i0312",
            id: "fence_proxy_async_shared_cta",
            operation_key: "fence.proxy.async.shared.cta",
            source_record: Some("int_nvvm_fence_proxy_async_shared_cta"),
            llvm_symbol: Some("llvm.nvvm.fence.proxy.async.shared_cta"),
            ptx_native_instruction: None,
            rust_arguments: &[],
            rust_result: "()",
            must_use: false,
            dialect_op_type: "FenceProxyAsyncSharedCtaOp",
            dialect_op_name: "nvvm.fence_proxy_async_shared_cta",
            dialect_operands: &[],
            dialect_results: &[],
            llvm_arguments: &[],
            llvm_results: &[],
            llvm_properties: &["IntrNoCallback"],
            adapter: MbarrierExtendedAdapter::ZeroOperandsToVoid,
            source_contract: MbarrierExtendedSourceContract::LlvmImported,
            execution_scope: "cta",
            minimum_ptx: "8.0",
            minimum_sm: "sm_90",
            ptx_result: "()",
            expected_ptx: instruction(&["fence", "proxy", "async", "shared::cta"], vec![]),
            inline_ptx: "fence.proxy.async.shared::cta;",
            inline_constraints: "~{memory}",
            ptx_isa_section: "Parallel Synchronization and Communication Instructions: fence.proxy",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-fence-proxy",
            summary: "Makes CTA-shared generic-proxy writes visible to the async proxy.",
        },
        MbarrierExtendedOperation::FenceMbarrierInitReleaseCluster => MbarrierExtendedRecipe {
            operation,
            abi_id: "i0313",
            id: "fence_mbarrier_init_release_cluster",
            operation_key: "fence.mbarrier_init.release.cluster",
            source_record: Some("int_nvvm_fence_mbarrier_init_release_cluster"),
            llvm_symbol: Some("llvm.nvvm.fence.mbarrier_init.release.cluster"),
            ptx_native_instruction: None,
            rust_arguments: &[],
            rust_result: "()",
            must_use: false,
            dialect_op_type: "FenceMbarrierInitReleaseClusterOp",
            dialect_op_name: "nvvm.fence_mbarrier_init_release_cluster",
            dialect_operands: &[],
            dialect_results: &[],
            llvm_arguments: &[],
            llvm_results: &[],
            llvm_properties: &["IntrNoCallback"],
            adapter: MbarrierExtendedAdapter::ZeroOperandsToVoid,
            source_contract: MbarrierExtendedSourceContract::LlvmImported,
            execution_scope: "cluster",
            minimum_ptx: "8.0",
            minimum_sm: "sm_90",
            ptx_result: "()",
            expected_ptx: instruction(&["fence", "mbarrier_init", "release", "cluster"], vec![]),
            inline_ptx: "fence.mbarrier_init.release.cluster;",
            inline_constraints: "~{memory}",
            ptx_isa_section: "Parallel Synchronization and Communication Instructions: fence.mbarrier_init",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-fence-mbarrier-init",
            summary: "Releases mbarrier initialization at cluster scope.",
        },
        MbarrierExtendedOperation::FenceProxyAsyncGenericReleaseSharedCtaCluster => {
            MbarrierExtendedRecipe {
                operation,
                abi_id: "i0314",
                id: "fence_proxy_async_generic_release_shared_cta_cluster",
                operation_key: "fence.proxy.async_generic.release.sync_restrict.shared_cta.cluster",
                source_record: Some(
                    "int_nvvm_fence_proxy_async_generic_release_sync_restrict_space_cta_scope_cluster",
                ),
                llvm_symbol: Some(
                    "llvm.nvvm.fence.proxy.async_generic.release.sync_restrict.space.cta.scope.cluster",
                ),
                ptx_native_instruction: None,
                rust_arguments: &[],
                rust_result: "()",
                must_use: false,
                dialect_op_type: "FenceProxyAsyncGenericReleaseSharedCtaClusterOp",
                dialect_op_name: "nvvm.fence_proxy_async_generic_release_shared_cta_cluster",
                dialect_operands: &[],
                dialect_results: &[],
                llvm_arguments: &[],
                llvm_results: &[],
                llvm_properties: &["IntrNoCallback"],
                adapter: MbarrierExtendedAdapter::ZeroOperandsToVoid,
                source_contract: MbarrierExtendedSourceContract::LlvmImported,
                execution_scope: "cluster",
                minimum_ptx: "8.6",
                minimum_sm: "sm_90",
                ptx_result: "()",
                expected_ptx: instruction(
                    &[
                        "fence",
                        "proxy",
                        "async::generic",
                        "release",
                        "sync_restrict::shared::cta",
                        "cluster",
                    ],
                    vec![],
                ),
                inline_ptx: "fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;",
                inline_constraints: "~{memory}",
                ptx_isa_section: "Parallel Synchronization and Communication Instructions: fence.proxy",
                ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-fence-proxy",
                summary: "Releases CTA-shared generic-proxy writes to the async proxy at cluster scope.",
            }
        }
        MbarrierExtendedOperation::FenceProxyAsyncGenericAcquireSharedClusterCluster => {
            MbarrierExtendedRecipe {
                operation,
                abi_id: "i0315",
                id: "fence_proxy_async_generic_acquire_shared_cluster_cluster",
                operation_key: "fence.proxy.async_generic.acquire.sync_restrict.shared_cluster.cluster",
                source_record: Some(
                    "int_nvvm_fence_proxy_async_generic_acquire_sync_restrict_space_cluster_scope_cluster",
                ),
                llvm_symbol: Some(
                    "llvm.nvvm.fence.proxy.async_generic.acquire.sync_restrict.space.cluster.scope.cluster",
                ),
                ptx_native_instruction: None,
                rust_arguments: &[],
                rust_result: "()",
                must_use: false,
                dialect_op_type: "FenceProxyAsyncGenericAcquireSharedClusterClusterOp",
                dialect_op_name: "nvvm.fence_proxy_async_generic_acquire_shared_cluster_cluster",
                dialect_operands: &[],
                dialect_results: &[],
                llvm_arguments: &[],
                llvm_results: &[],
                llvm_properties: &["IntrNoCallback"],
                adapter: MbarrierExtendedAdapter::ZeroOperandsToVoid,
                source_contract: MbarrierExtendedSourceContract::LlvmImported,
                execution_scope: "cluster",
                minimum_ptx: "8.6",
                minimum_sm: "sm_90",
                ptx_result: "()",
                expected_ptx: instruction(
                    &[
                        "fence",
                        "proxy",
                        "async::generic",
                        "acquire",
                        "sync_restrict::shared::cluster",
                        "cluster",
                    ],
                    vec![],
                ),
                inline_ptx: "fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;",
                inline_constraints: "~{memory}",
                ptx_isa_section: "Parallel Synchronization and Communication Instructions: fence.proxy",
                ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-fence-proxy",
                summary: "Acquires cluster-shared async-proxy writes through the generic proxy.",
            }
        }
        MbarrierExtendedOperation::Nanosleep => MbarrierExtendedRecipe {
            operation,
            abi_id: "i0316",
            id: "nanosleep",
            operation_key: "thread.nanosleep.u32",
            source_record: Some("int_nvvm_nanosleep"),
            llvm_symbol: Some("llvm.nvvm.nanosleep"),
            ptx_native_instruction: None,
            rust_arguments: &["u32"],
            rust_result: "()",
            must_use: false,
            dialect_op_type: "NanosleepOp",
            dialect_op_name: "nvvm.nanosleep",
            dialect_operands: &["i32"],
            dialect_results: &[],
            llvm_arguments: &["i32"],
            llvm_results: &[],
            llvm_properties: &["IntrConvergent", "IntrHasSideEffects", "IntrNoMem"],
            adapter: MbarrierExtendedAdapter::NanosecondsToVoid,
            source_contract: MbarrierExtendedSourceContract::LlvmImported,
            execution_scope: "thread",
            minimum_ptx: "6.3",
            minimum_sm: "sm_70",
            ptx_result: "()",
            expected_ptx: instruction(
                &["nanosleep", "u32"],
                vec![OperandPattern::RegisterOrImmediate],
            ),
            inline_ptx: "nanosleep.u32 $0;",
            inline_constraints: "r,~{memory}",
            ptx_isa_section: "Parallel Synchronization and Communication Instructions: nanosleep",
            ptx_isa_url: "https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-nanosleep",
            summary: "Suspends the executing thread for approximately the requested nanoseconds.",
        },
    }
}

pub(crate) fn mbarrier_extended_inline_recipe(
    operation: MbarrierExtendedOperation,
) -> (&'static str, &'static str) {
    let recipe = mbarrier_extended_recipe(operation);
    (recipe.inline_ptx, recipe.inline_constraints)
}

fn mbarrier_extended_backend_floor(
    operation: MbarrierExtendedOperation,
    backend: IntrinsicBackend,
) -> (&'static str, &'static str) {
    let recipe = mbarrier_extended_recipe(operation);
    match (operation, backend) {
        (MbarrierExtendedOperation::Nanosleep, IntrinsicBackend::LibNvvm) => ("6.3", "sm_75"),
        _ => (recipe.minimum_ptx, recipe.minimum_sm),
    }
}

fn expand_mbarrier_extended_admission(
    admission: &MbarrierExtendedAdmission,
) -> Result<Vec<OverlayIntrinsic>> {
    ensure!(
        admission.runtime_validation == RuntimeValidation::Unexecuted,
        "extended-mbarrier runtime validation may be marked executed only with GPU evidence"
    );
    ensure!(
        !admission.llvm_evidence_profile.trim().is_empty()
            && !admission.libnvvm_evidence_profile.trim().is_empty(),
        "compact extended-mbarrier admission requires both backend evidence profiles"
    );
    let expected_operations = BTreeSet::from([
        MbarrierExtendedOperation::ArriveExpectTxCta,
        MbarrierExtendedOperation::ArriveExpectTxCluster,
        MbarrierExtendedOperation::ArriveRemoteCluster,
        MbarrierExtendedOperation::TryWaitTokenCta,
        MbarrierExtendedOperation::TryWaitParityCta,
        MbarrierExtendedOperation::TryWaitParityCluster,
        MbarrierExtendedOperation::FenceProxyAsyncSharedCta,
        MbarrierExtendedOperation::FenceMbarrierInitReleaseCluster,
        MbarrierExtendedOperation::FenceProxyAsyncGenericReleaseSharedCtaCluster,
        MbarrierExtendedOperation::FenceProxyAsyncGenericAcquireSharedClusterCluster,
        MbarrierExtendedOperation::Nanosleep,
    ]);
    let actual_operations = admission
        .variants
        .iter()
        .map(|variant| variant.operation)
        .collect::<BTreeSet<_>>();
    ensure!(
        admission.variants.len() == expected_operations.len()
            && actual_operations == expected_operations,
        "compact extended-mbarrier admission must contain each reviewed operation exactly once"
    );

    admission
        .variants
        .iter()
        .map(|variant| {
            let recipe = mbarrier_extended_recipe(variant.operation);
            ensure!(
                variant.abi_id == recipe.abi_id,
                "{} must keep reserved ABI ID {}",
                recipe.id,
                recipe.abi_id
            );
            let source =
                recipe
                    .ptx_native_instruction
                    .map(|instruction| IntrinsicSource::PtxNative {
                        instruction: instruction.into(),
                    });
            Ok(OverlayIntrinsic {
                id: recipe.id.into(),
                abi_id: variant.abi_id.clone(),
                operation_key: recipe.operation_key.into(),
                family: "mbarrier_extended".into(),
                source,
                source_record: recipe.source_record.map(str::to_owned),
                rust_module: "barrier".into(),
                rust_name: recipe.id.into(),
                rust_arguments: recipe
                    .rust_arguments
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                rust_result: recipe.rust_result.into(),
                safe: false,
                must_use: recipe.must_use,
                safe_allowlist_reason: None,
                public_rust_path: format!("cuda_intrinsics::barrier::{}", recipe.id),
                compatibility_rust_paths: vec![format!("cuda_device::barrier::{}", recipe.id)],
                dialect_op_type: recipe.dialect_op_type.into(),
                dialect_op_name: recipe.dialect_op_name.into(),
                dialect_operands: recipe
                    .dialect_operands
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                dialect_results: recipe
                    .dialect_results
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                llvm_symbol: recipe.llvm_symbol.map(str::to_owned),
                resolved_llvm_symbol: None,
                llvm_arguments: recipe
                    .llvm_arguments
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                llvm_results: recipe
                    .llvm_results
                    .iter()
                    .map(|value| (*value).into())
                    .collect(),
                pure: false,
                memory: "read_write".into(),
                convergent: true,
                execution_scope: recipe.execution_scope.into(),
                minimum_ptx: recipe.minimum_ptx.into(),
                minimum_sm: Some(recipe.minimum_sm.into()),
                ptx_result: recipe.ptx_result.into(),
                targets: "all".into(),
                ptx_isa_version: "9.3".into(),
                ptx_isa_section: recipe.ptx_isa_section.into(),
                ptx_isa_url: recipe.ptx_isa_url.into(),
                lowering: "generated_mbarrier_extended_inline_ptx".into(),
                backend_lowerings: [IntrinsicBackend::LlvmNvptx, IntrinsicBackend::LibNvvm]
                    .into_iter()
                    .map(|backend| {
                        let (minimum_ptx, minimum_sm) =
                            mbarrier_extended_backend_floor(recipe.operation, backend);
                        OverlayBackendLowering {
                            backend,
                            mechanism: BackendLoweringMechanism::InlinePtx,
                            evidence_profile: match backend {
                                IntrinsicBackend::LlvmNvptx => {
                                    admission.llvm_evidence_profile.clone()
                                }
                                IntrinsicBackend::LibNvvm => {
                                    admission.libnvvm_evidence_profile.clone()
                                }
                            },
                            minimum_ptx: Some(minimum_ptx.into()),
                            minimum_sm: Some(minimum_sm.into()),
                        }
                    })
                    .collect(),
                packed_atomic: None,
                redux: None,
                vote: None,
                active_mask: None,
                warp_match: None,
                warp_barrier: None,
                warp_shuffle: None,
                dot_product: None,
                packed_alu: None,
                packed_conversion: None,
                cp_async_copy: None,
                cp_async_control: None,
                cp_async_mbarrier: None,
                mbarrier_basic: None,
                movmatrix: None,
                mbarrier_extended: Some(MbarrierExtended {
                    operation: recipe.operation,
                    adapter: recipe.adapter,
                    source_contract: recipe.source_contract,
                    runtime_validation: admission.runtime_validation,
                }),
                register_mma: None,
                sparse_mma: None,
                prmt: None,
                cluster_barrier: None,
                special_register: None,
                debug_control: None,
                cluster_memory: None,
                clc: None,
                ldmatrix_variant: None,
                ldmatrix_safety: None,
                ldmatrix_adapter: None,
                selected_address_space: None,
                expected_ptx: recipe.expected_ptx,
                summary: recipe.summary.into(),
            })
        })
        .collect()
}

fn validate_mbarrier_extended_policy(
    policy: &OverlayIntrinsic,
    source: &IntrinsicSource,
    declaration: Option<&ImportedIntrinsic>,
) -> Result<()> {
    let contract = policy
        .mbarrier_extended
        .as_ref()
        .with_context(|| format!("{} has no closed extended-mbarrier contract", policy.id))?;
    let recipe = mbarrier_extended_recipe(contract.operation);
    ensure!(
        contract.adapter == recipe.adapter
            && contract.source_contract == recipe.source_contract
            && policy.id == recipe.id
            && policy.abi_id == recipe.abi_id
            && policy.operation_key == recipe.operation_key,
        "{} identity or adapter does not match its closed extended-mbarrier recipe",
        policy.id
    );
    match recipe.source_contract {
        MbarrierExtendedSourceContract::LlvmImported => ensure!(
            matches!(
                source,
                IntrinsicSource::LlvmImported { source_record }
                    if Some(source_record.as_str()) == recipe.source_record
            ) && policy.source.is_none()
                && policy.source_record.as_deref() == recipe.source_record
                && policy.llvm_symbol.as_deref() == recipe.llvm_symbol
                && declaration.is_some_and(|record| {
                    record.source_record == recipe.source_record.unwrap()
                        && record.properties == recipe.llvm_properties
                }),
            "{} LLVM source contract changed",
            policy.id
        ),
        MbarrierExtendedSourceContract::PtxNativeRawClusterAddress => ensure!(
            matches!(
                source,
                IntrinsicSource::PtxNative { instruction }
                    if Some(instruction.as_str()) == recipe.ptx_native_instruction
            ) && policy.source_record.is_none()
                && policy.llvm_symbol.is_none()
                && declaration.is_none(),
            "{} must remain the PTX-native raw-cluster-address carrier",
            policy.id
        ),
    }
    ensure!(
        policy.rust_module == "barrier"
            && policy.rust_name == recipe.id
            && policy.rust_arguments == recipe.rust_arguments
            && policy.rust_result == recipe.rust_result
            && !policy.safe
            && policy.must_use == recipe.must_use
            && policy.public_rust_path == format!("cuda_intrinsics::barrier::{}", recipe.id)
            && policy.compatibility_rust_paths == [format!("cuda_device::barrier::{}", recipe.id)],
        "{} Rust API does not match its closed extended-mbarrier recipe",
        policy.id
    );
    ensure!(
        policy.dialect_op_type == recipe.dialect_op_type
            && policy.dialect_op_name == recipe.dialect_op_name
            && policy.dialect_operands == recipe.dialect_operands
            && policy.dialect_results == recipe.dialect_results
            && policy.llvm_arguments == recipe.llvm_arguments
            && policy.llvm_results == recipe.llvm_results
            && policy.lowering == "generated_mbarrier_extended_inline_ptx",
        "{} carrier or lowering does not match its closed extended-mbarrier recipe",
        policy.id
    );
    ensure!(
        !policy.pure
            && policy.memory == "read_write"
            && policy.convergent
            && policy.execution_scope == recipe.execution_scope,
        "{} convergence, memory clobber, or execution scope changed",
        policy.id
    );
    ensure!(
        policy.minimum_ptx == recipe.minimum_ptx
            && policy.minimum_sm.as_deref() == Some(recipe.minimum_sm)
            && policy.ptx_result == recipe.ptx_result
            && policy.targets == "all"
            && policy.ptx_isa_version == "9.3"
            && policy.ptx_isa_section == recipe.ptx_isa_section
            && policy.ptx_isa_url == recipe.ptx_isa_url,
        "{} target floor or PTX provenance changed",
        policy.id
    );
    ensure!(
        policy.expected_ptx == recipe.expected_ptx,
        "{} expected PTX does not match its closed extended-mbarrier recipe",
        policy.id
    );
    ensure_exact_inline_ptx_backends(
        policy,
        [
            (
                IntrinsicBackend::LlvmNvptx,
                mbarrier_extended_backend_floor(recipe.operation, IntrinsicBackend::LlvmNvptx).0,
                Some(
                    mbarrier_extended_backend_floor(recipe.operation, IntrinsicBackend::LlvmNvptx)
                        .1,
                ),
            ),
            (
                IntrinsicBackend::LibNvvm,
                mbarrier_extended_backend_floor(recipe.operation, IntrinsicBackend::LibNvvm).0,
                Some(
                    mbarrier_extended_backend_floor(recipe.operation, IntrinsicBackend::LibNvvm).1,
                ),
            ),
        ],
        "extended-mbarrier",
    )?;
    ensure_no_other_family_contract(policy, "extended mbarrier")?;
    Ok(())
}

fn ensure_exact_inline_ptx_backends(
    policy: &OverlayIntrinsic,
    requirements: [(IntrinsicBackend, &str, Option<&str>); 2],
    family: &str,
) -> Result<()> {
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
                        BackendLoweringMechanism::InlinePtx,
                    ),
                    (
                        IntrinsicBackend::LibNvvm,
                        BackendLoweringMechanism::InlinePtx,
                    ),
                ]),
        "{} must define exactly two reviewed {family} inline-PTX routes",
        policy.id
    );
    let requirements: BTreeMap<_, _> = requirements
        .into_iter()
        .map(|(backend, ptx, minimum_sm)| (backend, (ptx, minimum_sm)))
        .collect();
    for lowering in &policy.backend_lowerings {
        let (minimum_ptx, minimum_sm) = requirements[&lowering.backend];
        ensure!(
            lowering.minimum_ptx.as_deref() == Some(minimum_ptx)
                && lowering.minimum_sm.as_deref() == minimum_sm
                && !lowering.evidence_profile.trim().is_empty(),
            "{} backend {:?} does not carry its exact {family} floor",
            policy.id,
            lowering.backend
        );
    }
    Ok(())
}

fn validate_movmatrix_policy(policy: &OverlayIntrinsic, source: &IntrinsicSource) -> Result<()> {
    let contract = policy
        .movmatrix
        .as_ref()
        .context("movmatrix requires its closed contract")?;
    ensure!(
        policy.id == "movmatrix_trans_b16"
            && policy.operation_key == "movmatrix.m8n8.trans.b16"
            && matches!(
                source,
                IntrinsicSource::PtxNative { instruction }
                    if instruction == "movmatrix.sync.aligned.m8n8.trans.b16"
            )
            && policy.rust_module == "matrix"
            && policy.rust_name == "movmatrix_trans_b16"
            && policy.rust_arguments == ["u32"]
            && policy.rust_result == "u32"
            && !policy.safe
            && policy.must_use
            && policy.public_rust_path == "cuda_intrinsics::matrix::movmatrix_trans_b16"
            && policy.compatibility_rust_paths == ["cuda_device::wmma::movmatrix_trans_b16"]
            && policy.dialect_op_type == "MovmatrixTransB16Op"
            && policy.dialect_op_name == "nvvm.movmatrix_trans_b16"
            && policy.dialect_operands == ["i32"]
            && policy.dialect_results == ["i32"]
            && policy.pure
            && policy.memory == "none"
            && policy.convergent
            && policy.execution_scope == "warp"
            && policy.minimum_ptx == "7.8"
            && policy.minimum_sm.as_deref() == Some("sm_75")
            && policy.ptx_result == "u32"
            && policy.targets == "all"
            && policy.lowering == "generated_movmatrix_inline_ptx"
            && policy.expected_ptx
                == InstructionPattern {
                    mnemonic: "movmatrix".into(),
                    modifiers: ["sync", "aligned", "m8n8", "trans", "b16"]
                        .into_iter()
                        .map(str::to_owned)
                        .collect(),
                    operands: vec![OperandPattern::Register, OperandPattern::Register],
                },
        "{} is outside the closed movmatrix recipe",
        policy.id
    );
    ensure!(
        contract.participation == MovmatrixParticipation::AllWarpLanesSameInstructionNoExitedLanes
            && contract.adapter == MovmatrixAdapter::PackedB16x2U32ToPackedB16x2U32,
        "{} has an unreviewed movmatrix safety or adapter contract",
        policy.id
    );
    ensure_exact_inline_ptx_backends(
        policy,
        [
            (IntrinsicBackend::LlvmNvptx, "7.8", Some("sm_75")),
            (IntrinsicBackend::LibNvvm, "7.8", Some("sm_75")),
        ],
        "movmatrix",
    )?;
    ensure_no_other_family_contract(policy, "movmatrix")?;
    Ok(())
}

fn ensure_no_other_family_contract(policy: &OverlayIntrinsic, family: &str) -> Result<()> {
    ensure!(
        policy.packed_atomic.is_none()
            && policy.redux.is_none()
            && policy.vote.is_none()
            && policy.active_mask.is_none()
            && policy.warp_match.is_none()
            && policy.warp_barrier.is_none()
            && policy.warp_shuffle.is_none()
            && policy.dot_product.is_none()
            && policy.ldmatrix_variant.is_none()
            && policy.ldmatrix_safety.is_none()
            && policy.ldmatrix_adapter.is_none()
            && policy.selected_address_space.is_none()
            && (policy.family == "packed_alu") == policy.packed_alu.is_some()
            && (policy.family == "packed_conversion") == policy.packed_conversion.is_some()
            && (policy.family == "cp_async_copy") == policy.cp_async_copy.is_some()
            && (policy.family == "cp_async_control") == policy.cp_async_control.is_some()
            && (policy.family == "cp_async_mbarrier") == policy.cp_async_mbarrier.is_some()
            && (policy.family == "mbarrier_basic") == policy.mbarrier_basic.is_some()
            && (policy.family == "movmatrix") == policy.movmatrix.is_some()
            && (policy.family == "mbarrier_extended") == policy.mbarrier_extended.is_some()
            && (policy.family == "register_mma") == policy.register_mma.is_some()
            && (policy.family == "sparse_mma") == policy.sparse_mma.is_some()
            && (policy.family == "prmt") == policy.prmt.is_some()
            && (policy.family == "cluster_barrier") == policy.cluster_barrier.is_some()
            && (policy.family == "debug_control") == policy.debug_control.is_some()
            && (policy.family == "cluster_memory") == policy.cluster_memory.is_some()
            && (policy.family == "clc") == policy.clc.is_some(),
        "{} mixes another generated-family contract with {family}",
        policy.id
    );
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
    if targets == "all" {
        let Some(minimum_sm) = minimum_sm else {
            return Ok(CatalogHardwareTarget::All);
        };
        let sm = parse_sm_spelling(intrinsic_id, "minimum_sm", minimum_sm, None)?;
        return Ok(CatalogHardwareTarget::AnyOf {
            alternatives: vec![CatalogHardwareAlternative::MinimumSm { sm }],
        });
    }

    if targets.contains('|') {
        ensure!(
            minimum_sm.is_none(),
            "{} target alternatives {:?} cannot be combined with minimum_sm",
            intrinsic_id,
            targets
        );
        let spellings = targets.split('|').collect::<Vec<_>>();
        ensure!(
            spellings.len() >= 2,
            "{} target alternatives must contain at least two targets",
            intrinsic_id
        );
        ensure!(
            spellings.windows(2).all(|pair| pair[0] < pair[1]),
            "{} target alternatives must be unique and sorted",
            intrinsic_id
        );
        return Ok(CatalogHardwareTarget::AnyOf {
            alternatives: spellings
                .into_iter()
                .map(|spelling| parse_exact_hardware_alternative(intrinsic_id, spelling))
                .collect::<Result<Vec<_>>>()?,
        });
    }

    ensure!(
        minimum_sm.is_none(),
        "{} exact targets {:?} cannot be combined with minimum_sm",
        intrinsic_id,
        targets
    );
    Ok(CatalogHardwareTarget::AnyOf {
        alternatives: vec![parse_exact_hardware_alternative(intrinsic_id, targets)?],
    })
}

fn parse_exact_hardware_alternative(
    intrinsic_id: &str,
    target: &str,
) -> Result<CatalogHardwareAlternative> {
    let suffix = target
        .chars()
        .last()
        .filter(|suffix| matches!(suffix, 'a' | 'f'));
    let Some(suffix) = suffix else {
        bail!(
            "{} targets {:?} must be `all`, exact `sm_Na`, or family `sm_Nf`",
            intrinsic_id,
            target
        );
    };
    let sm = parse_sm_spelling(intrinsic_id, "targets", target, Some(suffix))?;
    Ok(match suffix {
        'a' => CatalogHardwareAlternative::ExactArchitecture { sm },
        'f' => CatalogHardwareAlternative::FamilyTarget { sm },
        _ => unreachable!(),
    })
}

fn parse_sm_spelling(
    intrinsic_id: &str,
    field: &str,
    value: &str,
    suffix: Option<char>,
) -> Result<u16> {
    let body = value.strip_prefix("sm_").with_context(|| {
        format!("{intrinsic_id} {field} {value:?} must use canonical sm_NN spelling")
    })?;
    let digits = match suffix {
        Some(suffix) => body.strip_suffix(suffix).with_context(|| {
            format!("{intrinsic_id} {field} {value:?} has the wrong target suffix")
        })?,
        None => body,
    };
    ensure!(
        matches!(digits.len(), 2 | 3) && digits.bytes().all(|byte| byte.is_ascii_digit()),
        "{} {} {:?} must use canonical sm_NN{} spelling",
        intrinsic_id,
        field,
        value,
        suffix.map_or("", |suffix| if suffix == 'a' { "a" } else { "f" })
    );
    let sm: u16 = digits
        .parse()
        .with_context(|| format!("{intrinsic_id} {field} target is too large"))?;
    let canonical = match suffix {
        Some(suffix) => format!("sm_{sm}{suffix}"),
        None => format!("sm_{sm}"),
    };
    ensure!(
        sm > 0 && canonical == value,
        "{} {} {:?} is not canonical",
        intrinsic_id,
        field,
        value
    );
    Ok(sm)
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

fn read_evidence_file(path: &Path) -> Result<EvidenceFile> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    parse_evidence_bytes(&bytes, &path.display().to_string())
}

fn parse_evidence_bytes(bytes: &[u8], source: &str) -> Result<EvidenceFile> {
    #[derive(serde::Deserialize)]
    struct Schema {
        schema: u32,
    }

    let schema: Schema =
        serde_json::from_slice(bytes).with_context(|| format!("parse JSON {source}"))?;
    match schema.schema {
        2..=5 => serde_json::from_slice(bytes)
            .with_context(|| format!("parse legacy evidence JSON {source}")),
        6 => {
            let file: EvidenceFileV6 = serde_json::from_slice(bytes)
                .with_context(|| format!("parse matrix evidence JSON {source}"))?;
            expand_evidence_file_v6(file)
                .with_context(|| format!("expand matrix evidence {source}"))
        }
        _ => bail!("unsupported evidence schema in {source}"),
    }
}

fn expand_evidence_file_v6(file: EvidenceFileV6) -> Result<EvidenceFile> {
    ensure!(file.schema == 6, "matrix evidence must use schema 6");
    ensure!(
        !file.records.is_empty() || !file.matrices.is_empty(),
        "schema-6 evidence contains no records or matrices"
    );
    reject_default_placeholders(&file.defaults, "evidence defaults", false)?;

    let mut fixture_by_id = BTreeMap::new();
    let mut fixture_coverage = BTreeMap::new();
    for fixture in &file.fixtures {
        ensure!(
            is_safe_matrix_token(&fixture.id),
            "evidence fixture ID {:?} is not a safe token",
            fixture.id
        );
        ensure!(
            fixture.coverage_count > 0 && !fixture.stages.is_empty(),
            "evidence fixture {} has no coverage or stages",
            fixture.id
        );
        reject_stage_placeholders(&fixture.stages, &format!("fixture {}", fixture.id))?;
        ensure!(
            fixture_by_id.insert(fixture.id.as_str(), fixture).is_none(),
            "duplicate evidence fixture ID {}",
            fixture.id
        );
        fixture_coverage.insert(fixture.id.as_str(), 0usize);
    }

    let mut records = file.records;
    let mut record_ids = BTreeSet::new();
    for record in &records {
        ensure!(
            record_ids.insert(record.id.clone()),
            "duplicate expanded evidence ID {}",
            record.id
        );
        validate_stage_pairs(&record.stages, &record.id)?;
    }

    for matrix in &file.matrices {
        let (expanded, referenced_fixtures) =
            expand_evidence_matrix(matrix, &file.defaults, &fixture_by_id)?;
        for fixture_id in referenced_fixtures {
            *fixture_coverage
                .get_mut(fixture_id.as_str())
                .expect("validated fixture reference") += expanded.len();
        }
        for record in expanded {
            ensure!(
                record_ids.insert(record.id.clone()),
                "duplicate expanded evidence ID {}",
                record.id
            );
            records.push(record);
        }
    }

    for fixture in &file.fixtures {
        let actual = fixture_coverage[fixture.id.as_str()];
        ensure!(
            actual > 0,
            "evidence fixture {} is not referenced by any matrix",
            fixture.id
        );
        ensure!(
            actual == fixture.coverage_count,
            "evidence fixture {} covers {actual} expanded records, expected {}",
            fixture.id,
            fixture.coverage_count
        );
    }

    Ok(EvidenceFile {
        schema: file.schema,
        backend_profile: file.backend_profile,
        backend_kind: file.backend_kind,
        llvm_revision: file.llvm_revision,
        backend_version: file.backend_version,
        backend_sha256: file.backend_sha256,
        artifact_path: file.artifact_path,
        build_id_prefix: file.build_id_prefix,
        nvvm_ir_version: file.nvvm_ir_version,
        debug_ir_version: file.debug_ir_version,
        records,
    })
}

fn expand_evidence_matrix(
    matrix: &EvidenceMatrix,
    defaults: &EvidenceRecordDefaults,
    fixtures: &BTreeMap<&str, &crate::model::EvidenceFixture>,
) -> Result<(Vec<EvidenceRecord>, Vec<String>)> {
    ensure!(!matrix.axes.is_empty(), "evidence matrix has no axes");
    let mut previous_axis: Option<&str> = None;
    let mut product_count = 1usize;
    let mut bindings = vec![BTreeMap::<String, String>::new()];
    for axis in &matrix.axes {
        ensure!(
            is_safe_matrix_token(&axis.name),
            "evidence matrix axis {:?} is not a safe token",
            axis.name
        );
        if let Some(previous) = previous_axis {
            ensure!(
                previous < axis.name.as_str(),
                "evidence matrix axes must be unique and sorted: {} follows {}",
                axis.name,
                previous
            );
        }
        previous_axis = Some(&axis.name);
        ensure!(
            !axis.values.is_empty(),
            "evidence matrix axis {} has no values",
            axis.name
        );
        let mut values = BTreeSet::new();
        for value in &axis.values {
            ensure!(
                is_safe_matrix_token(value),
                "evidence matrix axis {} has unsafe value {:?}",
                axis.name,
                value
            );
            ensure!(
                values.insert(value.as_str()),
                "evidence matrix axis {} has duplicate value {:?}",
                axis.name,
                value
            );
        }
        product_count = product_count
            .checked_mul(axis.values.len())
            .context("evidence matrix product count overflow")?;
        let mut next = Vec::with_capacity(product_count);
        for binding in bindings {
            for value in &axis.values {
                let mut expanded = binding.clone();
                expanded.insert(axis.name.clone(), value.clone());
                next.push(expanded);
            }
        }
        bindings = next;
    }
    ensure!(
        product_count == matrix.product_count,
        "evidence matrix expands to {product_count} records, expected {}",
        matrix.product_count
    );
    ensure!(
        !matrix.fixtures.is_empty(),
        "evidence matrix references no shared fixture"
    );

    let mut fixture_ids = BTreeSet::new();
    let mut previous_fixture: Option<&str> = None;
    let mut fixture_stages = Vec::new();
    for fixture_id in &matrix.fixtures {
        if let Some(previous) = previous_fixture {
            ensure!(
                previous < fixture_id.as_str(),
                "evidence matrix fixtures must be unique and sorted: {fixture_id} follows {previous}"
            );
        }
        previous_fixture = Some(fixture_id);
        let fixture = fixtures
            .get(fixture_id.as_str())
            .with_context(|| format!("evidence matrix references unknown fixture {fixture_id}"))?;
        fixture_ids.insert(fixture_id.clone());
        fixture_stages.extend(fixture.stages.iter().cloned());
    }

    reject_default_placeholders(&matrix.template.facts, "matrix template facts", true)?;
    validate_matrix_identity(&matrix.template)?;
    let axis_names = matrix
        .axes
        .iter()
        .map(|axis| axis.name.as_str())
        .collect::<BTreeSet<_>>();
    let mut used_axes = BTreeSet::new();
    let mut records = Vec::with_capacity(product_count);
    for binding in &bindings {
        let record = materialize_evidence_record(
            &matrix.template,
            defaults,
            binding,
            &mut used_axes,
            &fixture_stages,
        )?;
        validate_stage_pairs(&record.stages, &record.id)?;
        records.push(record);
    }
    for axis in axis_names {
        ensure!(
            used_axes.contains(axis),
            "evidence matrix declares unused axis {axis}"
        );
    }
    Ok((records, fixture_ids.into_iter().collect()))
}

fn validate_matrix_identity(template: &EvidenceMatrixTemplate) -> Result<()> {
    ensure!(
        !template.id.is_empty(),
        "evidence matrix template has an empty ID"
    );
    match (&template.source, &template.source_record) {
        (Some(_), None) | (None, Some(_)) => {}
        (Some(_), Some(_)) => bail!("evidence matrix template mixes source forms"),
        (None, None) => bail!("evidence matrix template has no source"),
    }
    reject_disallowed_placeholder(&template.expected_ptx.mnemonic, "PTX mnemonic")?;
    Ok(())
}

fn materialize_evidence_record(
    template: &EvidenceMatrixTemplate,
    defaults: &EvidenceRecordDefaults,
    bindings: &BTreeMap<String, String>,
    used_axes: &mut BTreeSet<String>,
    fixture_stages: &[EvidenceStage],
) -> Result<EvidenceRecord> {
    let id = expand_axis_placeholders(&template.id, bindings, used_axes, "evidence ID")?;
    let source = template
        .source
        .as_ref()
        .map(|source| expand_evidence_source(source, bindings, used_axes))
        .transpose()?;
    let source_record = template
        .source_record
        .as_deref()
        .map(|value| expand_axis_placeholders(value, bindings, used_axes, "source record"))
        .transpose()?;
    let llvm_symbol = template
        .llvm_symbol
        .as_deref()
        .map(|value| expand_axis_placeholders(value, bindings, used_axes, "LLVM symbol"))
        .transpose()?;
    validate_expanded_matrix_identity(
        &id,
        source.as_ref(),
        source_record.as_deref(),
        llvm_symbol.as_deref(),
    )?;
    let resolved_llvm_symbol = select_fact(
        &template.facts.resolved_llvm_symbol,
        &defaults.resolved_llvm_symbol,
    )
    .map(|value| expand_axis_placeholders(&value, bindings, used_axes, "resolved LLVM symbol"))
    .transpose()?;
    let mut expected_ptx = template.expected_ptx.clone();
    for modifier in &mut expected_ptx.modifiers {
        *modifier = expand_axis_placeholders(modifier, bindings, used_axes, "PTX modifier")?;
    }
    for operand in &mut expected_ptx.operands {
        if let OperandPattern::Exact { value } = operand {
            *value = expand_axis_placeholders(value, bindings, used_axes, "exact PTX operand")?;
        }
    }

    let mut stages = defaults.stages.clone();
    stages.extend(template.facts.stages.iter().cloned());
    stages.extend(fixture_stages.iter().cloned());
    let target_triple = required_fact(
        select_fact(&template.facts.target_triple, &defaults.target_triple),
        &id,
        "target triple",
    )?;
    let gpu_target = required_fact(
        select_fact(&template.facts.gpu_target, &defaults.gpu_target),
        &id,
        "GPU target",
    )?;
    let ptx_feature = required_fact(
        select_fact(&template.facts.ptx_feature, &defaults.ptx_feature),
        &id,
        "PTX feature",
    )?;
    let status = required_fact(
        select_fact(&template.facts.status, &defaults.status),
        &id,
        "status",
    )?;
    Ok(EvidenceRecord {
        id,
        source,
        source_record,
        llvm_symbol,
        resolved_llvm_symbol,
        llvm_arguments: select_fact(&template.facts.llvm_arguments, &defaults.llvm_arguments)
            .unwrap_or_default(),
        llvm_results: select_fact(&template.facts.llvm_results, &defaults.llvm_results)
            .unwrap_or_default(),
        concrete_llvm_arguments: select_fact(
            &template.facts.concrete_llvm_arguments,
            &defaults.concrete_llvm_arguments,
        )
        .unwrap_or_default(),
        concrete_llvm_results: select_fact(
            &template.facts.concrete_llvm_results,
            &defaults.concrete_llvm_results,
        )
        .unwrap_or_default(),
        target_triple,
        gpu_target,
        ptx_feature,
        status,
        stages,
        declaration_attributes_canonicalized: template
            .facts
            .declaration_attributes_canonicalized
            .or(defaults.declaration_attributes_canonicalized),
        runtime_validation: template
            .facts
            .runtime_validation
            .or(defaults.runtime_validation),
        expected_ptx,
    })
}

fn validate_expanded_matrix_identity(
    id: &str,
    source: Option<&IntrinsicSource>,
    source_record: Option<&str>,
    llvm_symbol: Option<&str>,
) -> Result<()> {
    ensure!(!id.is_empty(), "expanded evidence has an empty ID");
    let imported_source = match (source, source_record) {
        (Some(IntrinsicSource::LlvmImported { source_record }), None) => {
            ensure!(
                !source_record.is_empty(),
                "expanded evidence {id} has an empty source record"
            );
            true
        }
        (Some(IntrinsicSource::PtxNative { instruction }), None) => {
            ensure!(
                !instruction.is_empty(),
                "expanded evidence {id} has an empty PTX source"
            );
            false
        }
        (None, Some(source_record)) => {
            ensure!(
                !source_record.is_empty(),
                "expanded evidence {id} has an empty source record"
            );
            true
        }
        _ => unreachable!("matrix source shape was validated before expansion"),
    };
    if imported_source {
        ensure!(
            llvm_symbol.is_some_and(|symbol| !symbol.is_empty()),
            "expanded imported evidence {id} has no LLVM symbol"
        );
    } else {
        ensure!(
            llvm_symbol.is_none(),
            "expanded PTX-native evidence {id} invents an LLVM symbol"
        );
    }
    Ok(())
}

fn select_fact<T: Clone>(specific: &Option<T>, default: &Option<T>) -> Option<T> {
    specific.clone().or_else(|| default.clone())
}

fn required_fact(value: Option<String>, id: &str, field: &str) -> Result<String> {
    let value = value.with_context(|| format!("expanded evidence {id} has no {field}"))?;
    ensure!(
        !value.trim().is_empty(),
        "expanded evidence {id} has an empty {field}"
    );
    Ok(value)
}

fn expand_evidence_source(
    source: &IntrinsicSource,
    bindings: &BTreeMap<String, String>,
    used_axes: &mut BTreeSet<String>,
) -> Result<IntrinsicSource> {
    Ok(match source {
        IntrinsicSource::LlvmImported { source_record } => IntrinsicSource::LlvmImported {
            source_record: expand_axis_placeholders(
                source_record,
                bindings,
                used_axes,
                "tagged source record",
            )?,
        },
        IntrinsicSource::PtxNative { instruction } => IntrinsicSource::PtxNative {
            instruction: expand_axis_placeholders(
                instruction,
                bindings,
                used_axes,
                "PTX-native source",
            )?,
        },
    })
}

fn expand_axis_placeholders(
    value: &str,
    bindings: &BTreeMap<String, String>,
    used_axes: &mut BTreeSet<String>,
    field: &str,
) -> Result<String> {
    let mut output = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(position) = rest.find('$') {
        output.push_str(&rest[..position]);
        let placeholder = &rest[position..];
        ensure!(
            placeholder.starts_with("${"),
            "{field} contains malformed matrix placeholder {value:?}"
        );
        let close = placeholder
            .find('}')
            .with_context(|| format!("{field} contains an unterminated matrix placeholder"))?;
        let axis = &placeholder[2..close];
        ensure!(
            is_safe_matrix_token(axis),
            "{field} contains malformed matrix axis {axis:?}"
        );
        let replacement = bindings
            .get(axis)
            .with_context(|| format!("{field} references unknown matrix axis {axis}"))?;
        output.push_str(replacement);
        used_axes.insert(axis.to_owned());
        rest = &placeholder[close + 1..];
    }
    output.push_str(rest);
    Ok(output)
}

fn is_safe_matrix_token(value: &str) -> bool {
    let mut bytes = value.bytes();
    bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
        && bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

fn reject_default_placeholders(
    defaults: &EvidenceRecordDefaults,
    context: &str,
    allow_resolved_symbol: bool,
) -> Result<()> {
    if !allow_resolved_symbol && let Some(value) = &defaults.resolved_llvm_symbol {
        reject_disallowed_placeholder(value, context)?;
    }
    for value in defaults
        .target_triple
        .iter()
        .chain(defaults.gpu_target.iter())
        .chain(defaults.ptx_feature.iter())
        .chain(defaults.status.iter())
        .chain(defaults.llvm_arguments.iter().flatten())
        .chain(defaults.llvm_results.iter().flatten())
        .chain(defaults.concrete_llvm_arguments.iter().flatten())
        .chain(defaults.concrete_llvm_results.iter().flatten())
    {
        reject_disallowed_placeholder(value, context)?;
    }
    reject_stage_placeholders(&defaults.stages, context)
}

fn reject_stage_placeholders(stages: &[EvidenceStage], context: &str) -> Result<()> {
    for stage in stages {
        for value in stage
            .targets
            .iter()
            .chain(std::iter::once(&stage.representation))
            .chain(std::iter::once(&stage.detail))
            .chain(stage.tool_path.iter())
            .chain(stage.tool_version.iter())
            .chain(stage.tool_sha256.iter())
        {
            reject_disallowed_placeholder(value, context)?;
        }
    }
    Ok(())
}

fn reject_disallowed_placeholder(value: &str, field: &str) -> Result<()> {
    ensure!(
        !value.contains("${"),
        "{field} cannot contain matrix placeholders"
    );
    Ok(())
}

fn validate_stage_pairs(stages: &[EvidenceStage], id: &str) -> Result<()> {
    let mut pairs = Vec::new();
    for stage in stages {
        let pair = (stage.stage, stage.mechanism);
        ensure!(
            !pairs.contains(&pair),
            "expanded evidence {id} has conflicting duplicate {:?}/{:?} stages",
            stage.stage,
            stage.mechanism
        );
        pairs.push(pair);
    }
    Ok(())
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
        let file = read_evidence_file(&path)?;
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
        validate_packed_conversion_backend_evidence(policy, record, lowering)?;
        validate_cluster_barrier_backend_evidence(policy, record, lowering)?;
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

fn validate_cluster_barrier_backend_evidence(
    policy: &OverlayIntrinsic,
    record: &EvidenceRecord,
    lowering: &crate::model::OverlayBackendLowering,
) -> Result<()> {
    if policy.family != "cluster_barrier" || lowering.backend != IntrinsicBackend::LibNvvm {
        return Ok(());
    }
    for stage in [
        EvidenceStageKind::BackendCodegen,
        EvidenceStageKind::DeviceLink,
    ] {
        ensure!(
            record.stages.iter().any(|candidate| {
                candidate.stage == stage
                    && candidate.mechanism == Some(BackendLoweringMechanism::TypedNvvm)
                    && candidate.outcome == "failed"
            }),
            "{} libNVVM inline-PTX evidence must record the failed typed-NVVM {:?} comparison",
            policy.id,
            stage
        );
    }
    ensure!(
        !record.stages.iter().any(|candidate| {
            candidate.stage == EvidenceStageKind::DeviceLink
                && candidate.mechanism == Some(BackendLoweringMechanism::TypedNvvm)
                && candidate.outcome == "succeeded"
        }),
        "{} libNVVM evidence cannot select inline PTX after a successful typed-NVVM terminal",
        policy.id
    );
    Ok(())
}

fn validate_packed_conversion_backend_evidence(
    policy: &OverlayIntrinsic,
    record: &EvidenceRecord,
    lowering: &crate::model::OverlayBackendLowering,
) -> Result<()> {
    if policy.family != "packed_conversion" {
        return Ok(());
    }
    match lowering.backend {
        IntrinsicBackend::LlvmNvptx => {
            validate_typed_llvm_evidence(policy, record)?;
            for stage in [
                EvidenceStageKind::DeclarationCanonicalization,
                EvidenceStageKind::BackendCodegen,
            ] {
                ensure!(
                    successful_stage(record, BackendLoweringMechanism::TypedNvvm, stage).is_some(),
                    "{} LLVM packed-conversion evidence requires a successful auxiliary typed-NVVM {:?} stage",
                    policy.id,
                    stage
                );
            }
            ensure!(
                has_valid_ptx_assembly_stage(record, BackendLoweringMechanism::TypedNvvm),
                "{} LLVM packed-conversion evidence requires a successful auxiliary typed-NVVM PTX-assembly stage with exact tool identity",
                policy.id
            );
            ensure!(
                matches!(record.status.as_str(), "validated" | "executed"),
                "{} LLVM packed-conversion evidence requires validated evidence status for its auxiliary typed-NVVM terminal stage",
                policy.id
            );
            let typed_lowering = crate::model::OverlayBackendLowering {
                mechanism: BackendLoweringMechanism::TypedNvvm,
                ..lowering.clone()
            };
            validate_selected_stage_targets(policy, record, &typed_lowering)?;
            Ok(())
        }
        IntrinsicBackend::LibNvvm => {
            ensure!(
                record.resolved_llvm_symbol.is_none()
                    && record.concrete_llvm_arguments.is_empty()
                    && record.concrete_llvm_results.is_empty()
                    && record.declaration_attributes_canonicalized.is_none()
                    && !record.stages.iter().any(|stage| {
                        stage.mechanism == Some(BackendLoweringMechanism::TypedNvvm)
                    }),
                "{} libNVVM inline-PTX evidence must not claim typed LLVM support",
                policy.id
            );
            Ok(())
        }
    }
}

fn validate_typed_llvm_evidence(policy: &OverlayIntrinsic, record: &EvidenceRecord) -> Result<()> {
    let concrete_arguments = policy
        .llvm_arguments
        .iter()
        .map(|argument| {
            match argument.as_str() {
                "shared_ptr" => return Ok("ptr addrspace(3)".into()),
                "global_ptr" => return Ok("ptr addrspace(1)".into()),
                "ptr" => return Ok("ptr".into()),
                "anyptr" => {}
                _ => return Ok(argument.clone()),
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
    let expected_hardware = match requirement.hardware {
        CatalogHardwareTarget::AnyOf { alternatives } if !alternatives.is_empty() => alternatives,
        _ => bail!(
            "{} selected backend stages require a hardware target",
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
        let (hardware, ptx) = selected_stage_floor(stage)?;
        let allow_forward_minimum = stage.stage != EvidenceStageKind::BackendCodegen;
        let hardware_matches = expected_hardware.iter().any(|expected| {
            selected_stage_hardware_matches(hardware, *expected, allow_forward_minimum)
        });
        let ptx_matches = match lowering.backend {
            IntrinsicBackend::LlvmNvptx => ptx == expected_ptx,
            IntrinsicBackend::LibNvvm => ptx >= expected_ptx,
        };
        ensure!(
            hardware_matches && ptx_matches,
            "{} evidence stage {:?} targets {} / PTX {}.{} instead of a compatible target at catalog floor {} / PTX {}.{}",
            policy.id,
            stage.stage,
            describe_stage_hardware(hardware),
            ptx / 10,
            ptx % 10,
            expected_hardware
                .iter()
                .map(|hardware| describe_stage_hardware(*hardware))
                .collect::<Vec<_>>()
                .join(" or "),
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
        let (hardware, ptx) = selected_stage_floor(runtime)?;
        let ptx_matches = match lowering.backend {
            IntrinsicBackend::LlvmNvptx => ptx == expected_ptx,
            IntrinsicBackend::LibNvvm => ptx >= expected_ptx,
        };
        ensure!(
            expected_hardware
                .iter()
                .any(|expected| selected_stage_hardware_matches(hardware, *expected, true))
                && ptx_matches,
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
    parse_stage_hardware(target).is_some()
}

fn parse_stage_hardware(target: &str) -> Option<CatalogHardwareAlternative> {
    let value = target
        .strip_prefix("sm_")
        .or_else(|| target.strip_prefix("compute_"))?;
    let suffix = value
        .chars()
        .last()
        .filter(|suffix| matches!(suffix, 'a' | 'f'));
    let digits = suffix.map_or(value, |suffix| &value[..value.len() - suffix.len_utf8()]);
    if !matches!(digits.len(), 2 | 3) || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let sm: u16 = digits.parse().ok()?;
    if sm == 0 || sm.to_string() != digits {
        return None;
    }
    Some(match suffix {
        None => CatalogHardwareAlternative::MinimumSm { sm },
        Some('a') => CatalogHardwareAlternative::ExactArchitecture { sm },
        Some('f') => CatalogHardwareAlternative::FamilyTarget { sm },
        _ => unreachable!(),
    })
}

fn selected_stage_hardware_matches(
    actual: CatalogHardwareAlternative,
    expected: CatalogHardwareAlternative,
    allow_forward_minimum: bool,
) -> bool {
    match expected {
        CatalogHardwareAlternative::MinimumSm { sm: expected } => {
            if allow_forward_minimum {
                match actual {
                    CatalogHardwareAlternative::MinimumSm { sm }
                    | CatalogHardwareAlternative::ExactArchitecture { sm }
                    | CatalogHardwareAlternative::FamilyTarget { sm } => sm >= expected,
                }
            } else {
                actual == CatalogHardwareAlternative::MinimumSm { sm: expected }
            }
        }
        CatalogHardwareAlternative::ExactArchitecture { .. }
        | CatalogHardwareAlternative::FamilyTarget { .. } => actual == expected,
    }
}

fn describe_stage_hardware(hardware: CatalogHardwareAlternative) -> String {
    match hardware {
        CatalogHardwareAlternative::MinimumSm { sm } => format!("sm_{sm}"),
        CatalogHardwareAlternative::ExactArchitecture { sm } => format!("sm_{sm}a"),
        CatalogHardwareAlternative::FamilyTarget { sm } => format!("sm_{sm}f"),
    }
}

fn selected_stage_floor(
    stage: &crate::model::EvidenceStage,
) -> Result<(CatalogHardwareAlternative, u16)> {
    let mut hardware = None;
    let mut ptx = None;
    for target in &stage.targets {
        if let Some(value) = target.strip_prefix("ptx") {
            let value = value.parse::<u16>()?;
            ensure!(
                ptx.replace(value).is_none(),
                "stage has duplicate PTX targets"
            );
        } else {
            let value = parse_stage_hardware(target)
                .with_context(|| format!("stage has unsupported target spelling {target:?}"))?;
            ensure!(
                hardware.replace(value).is_none(),
                "stage has duplicate architecture targets"
            );
        }
    }
    Ok((
        hardware.context("selected stage has no architecture target")?,
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
    if policy.family == "stmatrix" {
        ensure!(
            runtime_states
                .iter()
                .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
            "{} stmatrix source contract is unexecuted but backend evidence disagrees",
            policy.id
        );
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
    if let Some(mbarrier) = &policy.mbarrier_basic {
        match mbarrier.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} mbarrier runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} mbarrier runtime is executed but no backend evidence records execution",
                policy.id
            ),
        }
    }
    if let Some(movmatrix) = &policy.movmatrix {
        match movmatrix.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} movmatrix runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} movmatrix runtime is executed but no backend evidence records execution",
                policy.id
            ),
        }
    }
    if let Some(mbarrier) = &policy.mbarrier_extended {
        match mbarrier.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} extended-mbarrier runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} extended-mbarrier runtime is executed but no backend evidence records execution",
                policy.id
            ),
        }
    }
    if let Some(bridge) = &policy.cp_async_mbarrier {
        match bridge.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} cp.async mbarrier runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} cp.async mbarrier runtime is executed but no backend evidence records execution",
                policy.id
            ),
        }
    }
    if let Some(mma) = &policy.register_mma {
        match mma.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} register-MMA runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} register-MMA runtime is executed but no backend evidence records execution",
                policy.id
            ),
        }
    }
    if let Some(mma) = &policy.sparse_mma {
        match mma.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} sparse-MMA runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} sparse-MMA runtime is executed but no backend evidence records execution",
                policy.id
            ),
        }
    }
    if let Some(debug) = &policy.debug_control {
        match debug.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} debug-control runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} debug-control runtime is executed but no backend evidence records execution",
                policy.id
            ),
        }
    }
    if let Some(cluster_memory) = &policy.cluster_memory {
        match cluster_memory.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} cluster-memory runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} cluster-memory runtime is executed but no backend evidence records execution",
                policy.id
            ),
        }
    }
    if let Some(clc) = &policy.clc {
        match clc.runtime_validation {
            RuntimeValidation::Unexecuted => ensure!(
                runtime_states
                    .iter()
                    .all(|state| *state == Some(RuntimeValidation::Unexecuted)),
                "{} CLC runtime is unexecuted but backend evidence disagrees",
                policy.id
            ),
            RuntimeValidation::Executed => ensure!(
                runtime_states.contains(&Some(RuntimeValidation::Executed)),
                "{} CLC runtime is executed but no backend evidence records execution",
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
    materialize_record(
        policy,
        source,
        declaration,
        CatalogBackend {
            profile: backend_profile.to_owned(),
            version: backend_version.to_owned(),
            sha256: backend_sha256.to_owned(),
            status: evidence.status.clone(),
            target_triple: evidence.target_triple.clone(),
            gpu_target: evidence.gpu_target.clone(),
            ptx_feature: evidence.ptx_feature.clone(),
        },
        backend_lowerings,
        intrinsic_abi,
    )
}

fn materialize_record(
    policy: &OverlayIntrinsic,
    source: IntrinsicSource,
    declaration: Option<&ImportedIntrinsic>,
    backend: CatalogBackend,
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
        backend,
        backend_lowerings,
        packed_atomic: policy.packed_atomic.clone(),
        redux: policy.redux.clone(),
        vote: policy.vote.clone(),
        active_mask: policy.active_mask.clone(),
        warp_match: policy.warp_match.clone(),
        warp_barrier: policy.warp_barrier.clone(),
        warp_shuffle: policy.warp_shuffle.clone(),
        dot_product: policy.dot_product.clone(),
        packed_alu: policy.packed_alu.clone(),
        packed_conversion: policy.packed_conversion.clone(),
        cp_async_copy: policy.cp_async_copy.clone(),
        cp_async_control: policy.cp_async_control.clone(),
        cp_async_mbarrier: policy.cp_async_mbarrier.clone(),
        mbarrier_basic: policy.mbarrier_basic.clone(),
        movmatrix: policy.movmatrix.clone(),
        mbarrier_extended: policy.mbarrier_extended.clone(),
        register_mma: policy.register_mma.clone(),
        sparse_mma: policy.sparse_mma.clone(),
        prmt: policy.prmt.clone(),
        cluster_barrier: policy.cluster_barrier.clone(),
        special_register: policy.special_register.clone(),
        debug_control: policy.debug_control.clone(),
        cluster_memory: policy.cluster_memory.clone(),
        clc: policy.clc.clone(),
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
            warp_barrier: None,
            warp_shuffle: None,
            dot_product: None,
            packed_alu: None,
            packed_conversion: None,
            cp_async_copy: None,
            cp_async_control: None,
            cp_async_mbarrier: None,
            mbarrier_basic: None,
            movmatrix: None,
            mbarrier_extended: None,
            register_mma: None,
            sparse_mma: None,
            prmt: None,
            cluster_barrier: None,
            special_register: None,
            debug_control: None,
            cluster_memory: None,
            clc: None,
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

    fn movmatrix_policy() -> OverlayIntrinsic {
        let mut record = policy();
        record.id = "movmatrix_trans_b16".into();
        record.abi_id = "i0305".into();
        record.operation_key = "movmatrix.m8n8.trans.b16".into();
        record.family = "movmatrix".into();
        record.source = Some(IntrinsicSource::PtxNative {
            instruction: "movmatrix.sync.aligned.m8n8.trans.b16".into(),
        });
        record.source_record = None;
        record.rust_module = "matrix".into();
        record.rust_name = "movmatrix_trans_b16".into();
        record.rust_arguments = vec!["u32".into()];
        record.rust_result = "u32".into();
        record.safe = false;
        record.must_use = true;
        record.safe_allowlist_reason = None;
        record.public_rust_path = "cuda_intrinsics::matrix::movmatrix_trans_b16".into();
        record.compatibility_rust_paths = vec!["cuda_device::wmma::movmatrix_trans_b16".into()];
        record.dialect_op_type = "MovmatrixTransB16Op".into();
        record.dialect_op_name = "nvvm.movmatrix_trans_b16".into();
        record.dialect_operands = vec!["i32".into()];
        record.dialect_results = vec!["i32".into()];
        record.llvm_symbol = None;
        record.resolved_llvm_symbol = None;
        record.llvm_arguments.clear();
        record.llvm_results.clear();
        record.pure = true;
        record.memory = "none".into();
        record.convergent = true;
        record.execution_scope = "warp".into();
        record.minimum_ptx = "7.8".into();
        record.minimum_sm = Some("sm_75".into());
        record.ptx_result = "u32".into();
        record.targets = "all".into();
        record.ptx_isa_section =
            "9.7.15.5.17 Warp-level matrix transpose instruction: movmatrix".into();
        record.lowering = "generated_movmatrix_inline_ptx".into();
        record.backend_lowerings = [IntrinsicBackend::LlvmNvptx, IntrinsicBackend::LibNvvm]
            .into_iter()
            .map(|backend| OverlayBackendLowering {
                backend,
                mechanism: BackendLoweringMechanism::InlinePtx,
                evidence_profile: match backend {
                    IntrinsicBackend::LlvmNvptx => "llvm-test",
                    IntrinsicBackend::LibNvvm => "libnvvm-test",
                }
                .into(),
                minimum_ptx: Some("7.8".into()),
                minimum_sm: Some("sm_75".into()),
            })
            .collect();
        record.movmatrix = Some(crate::model::Movmatrix {
            participation: MovmatrixParticipation::AllWarpLanesSameInstructionNoExitedLanes,
            adapter: MovmatrixAdapter::PackedB16x2U32ToPackedB16x2U32,
            runtime_validation: RuntimeValidation::Unexecuted,
        });
        record.expected_ptx = InstructionPattern::new(
            "movmatrix",
            &["sync", "aligned", "m8n8", "trans", "b16"],
            vec![OperandPattern::Register, OperandPattern::Register],
        );
        record.summary = "Transposes one packed b16 matrix fragment across a warp.".into();
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

    fn shared_matrix_stage() -> EvidenceStage {
        EvidenceStage {
            targets: vec!["sm_80".into(), "ptx71".into()],
            representation: "shared fixture".into(),
            stage: EvidenceStageKind::BackendCodegen,
            mechanism: Some(BackendLoweringMechanism::InlinePtx),
            outcome: "succeeded".into(),
            detail: "$dst remains fixture text".into(),
            artifact_kind: None,
            tool_path: None,
            tool_version: None,
            tool_sha256: None,
        }
    }

    fn synthetic_matrix_json() -> serde_json::Value {
        serde_json::json!({
            "schema": 6,
            "backend_profile": "matrix-test",
            "backend_kind": "llvm_nvptx",
            "llvm_revision": "test",
            "backend_version": "LLVM matrix test",
            "backend_sha256": "0123456789abcdef",
            "defaults": {
                "llvm_arguments": ["i32"],
                "llvm_results": ["i32"],
                "target_triple": "nvptx64-nvidia-cuda",
                "gpu_target": "sm_80",
                "ptx_feature": "+ptx71",
                "status": "lowered"
            },
            "fixtures": [{
                "id": "shared",
                "coverage_count": 2,
                "stages": [{
                    "targets": ["sm_80", "ptx71"],
                    "representation": "shared fixture",
                    "stage": "backend_codegen",
                    "mechanism": "inline_ptx",
                    "outcome": "succeeded",
                    "detail": "$dst remains fixture text"
                }]
            }],
            "matrices": [{
                "axes": [{
                    "name": "element",
                    "values": ["s8", "u8"]
                }],
                "product_count": 2,
                "fixtures": ["shared"],
                "template": {
                    "id": "synthetic_${element}",
                    "source_record": "int_synthetic_${element}",
                    "llvm_symbol": "llvm.synthetic.${element}",
                    "expected_ptx": {
                        "mnemonic": "mma",
                        "modifiers": ["sync", "${element}"],
                        "operands": [{"kind": "register"}]
                    }
                }
            }],
            "records": [{
                "id": "synthetic_explicit",
                "source_record": "int_synthetic_explicit",
                "llvm_symbol": "llvm.synthetic.explicit",
                "llvm_arguments": ["i32"],
                "llvm_results": ["i32"],
                "target_triple": "nvptx64-nvidia-cuda",
                "gpu_target": "sm_80",
                "ptx_feature": "+ptx71",
                "status": "lowered",
                "expected_ptx": {
                    "mnemonic": "mma",
                    "modifiers": ["sync", "explicit"],
                    "operands": [{"kind": "register"}]
                }
            }]
        })
    }

    fn policy_matrix_json() -> serde_json::Value {
        serde_json::json!({
            "schema": 6,
            "backend_profile": "matrix-test",
            "llvm_revision": "test",
            "backend_version": "LLVM matrix test",
            "backend_sha256": "0123456789abcdef",
            "defaults": {
                "llvm_arguments": [],
                "llvm_results": ["i32"],
                "target_triple": "nvptx64-nvidia-cuda",
                "gpu_target": "sm_70",
                "ptx_feature": "+ptx60",
                "status": "lowered"
            },
            "fixtures": [{
                "id": "policy_fixture",
                "coverage_count": 1,
                "stages": [{
                    "targets": ["sm_70", "ptx60"],
                    "representation": "policy fixture",
                    "stage": "backend_codegen",
                    "mechanism": "typed_nvvm",
                    "outcome": "succeeded",
                    "detail": "shared policy fixture"
                }]
            }],
            "matrices": [{
                "axes": [{
                    "name": "axis",
                    "values": ["x"]
                }],
                "product_count": 1,
                "fixtures": ["policy_fixture"],
                "template": {
                    "id": "thread_idx_${axis}",
                    "source_record": "int_nvvm_read_ptx_sreg_tid_${axis}",
                    "llvm_symbol": "llvm.nvvm.read.ptx.sreg.tid.${axis}",
                    "expected_ptx": {
                        "mnemonic": "mov",
                        "modifiers": ["u32"],
                        "operands": [
                            {"kind": "register"},
                            {"kind": "exact", "value": "%tid.${axis}"}
                        ]
                    }
                }
            }]
        })
    }

    fn parse_synthetic_evidence(value: &serde_json::Value) -> Result<EvidenceFile> {
        parse_evidence_bytes(&serde_json::to_vec(value).unwrap(), "synthetic evidence")
    }

    fn assert_synthetic_evidence_error(value: &serde_json::Value, expected: &str) {
        let error = parse_synthetic_evidence(value).unwrap_err();
        let message = format!("{error:#}");
        assert!(
            message.contains(expected),
            "expected {expected:?} in {message:?}"
        );
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

    fn bind_pinned_abi_ids(repo_root: &Path, overlay: &mut OverlayFile) {
        let ledger_path = repo_root.join(format!("intrinsics/abi-v{}.toml", overlay.intrinsic_abi));
        let ledger: AbiLedgerFile =
            toml::from_str(&std::fs::read_to_string(ledger_path).unwrap()).unwrap();
        bind_generated_abi_ids(overlay, &ledger).unwrap();
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

    fn pinned_mbarrier_basic_records() -> BTreeMap<String, (OverlayIntrinsic, ImportedIntrinsic)> {
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
            .filter(|record| record.family == "mbarrier_basic")
            .map(|policy| {
                let declaration = declarations[policy.source_record.as_deref().unwrap()].clone();
                (policy.id.clone(), (policy, declaration))
            })
            .collect()
    }

    fn pinned_cp_async_mbarrier_records() -> BTreeMap<String, (OverlayIntrinsic, ImportedIntrinsic)>
    {
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
            .filter(|record| record.family == "cp_async_mbarrier")
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

    fn packed_alu_policy(
        format: PackedAluFormat,
        operation: PackedAluOperation,
    ) -> OverlayIntrinsic {
        let recipe = packed_alu_recipe(format, operation);
        let rust_module = match format {
            PackedAluFormat::Bf16x2 => "bf16x2",
            PackedAluFormat::F16x2 => "f16x2",
        };
        let mut record = policy();
        record.id = recipe.id.into();
        record.abi_id = recipe.abi_id.into();
        record.operation_key = recipe.operation_key.into();
        record.family = "packed_alu".into();
        match &recipe.source {
            PackedAluRecipeSource::Imported {
                record: source_record,
                symbol,
                resolved_symbol,
                arguments,
                results,
                ..
            } => {
                record.source = None;
                record.source_record = Some((*source_record).into());
                record.llvm_symbol = Some((*symbol).into());
                record.resolved_llvm_symbol = resolved_symbol.map(str::to_owned);
                record.llvm_arguments = arguments.iter().map(|value| (*value).into()).collect();
                record.llvm_results = results.iter().map(|value| (*value).into()).collect();
            }
            PackedAluRecipeSource::PtxNative => {
                record.source = Some(IntrinsicSource::PtxNative {
                    instruction: recipe.ptx_mnemonic.into(),
                });
                record.source_record = None;
                record.llvm_symbol = None;
                record.resolved_llvm_symbol = None;
                record.llvm_arguments.clear();
                record.llvm_results.clear();
            }
        }
        record.rust_module = rust_module.into();
        record.rust_name = recipe.rust_name.into();
        record.rust_arguments = vec!["u32".into(); recipe.arity];
        record.rust_result = "u32".into();
        record.safe = true;
        record.must_use = recipe.must_use;
        record.safe_allowlist_reason = Some("the operation has no caller obligations".into());
        record.public_rust_path = format!("cuda_intrinsics::{rust_module}::{}", recipe.rust_name);
        record.compatibility_rust_paths =
            vec![format!("cuda_device::{rust_module}::{}", recipe.rust_name)];
        record.dialect_op_type = recipe.dialect_op_type.into();
        record.dialect_op_name = recipe.dialect_op_name.into();
        record.dialect_operands = vec!["i32".into(); recipe.arity];
        record.dialect_results = vec!["i32".into()];
        record.pure = true;
        record.memory = "none".into();
        record.convergent = false;
        record.execution_scope = "thread".into();
        record.minimum_ptx = recipe.minimum_ptx.into();
        record.minimum_sm = Some(recipe.minimum_sm.into());
        record.ptx_result = "u32".into();
        record.ptx_isa_section = recipe.ptx_isa_section.into();
        record.ptx_isa_url = recipe.ptx_isa_url.into();
        record.lowering = "generated_packed_alu_inline_ptx".into();
        record.backend_lowerings = [IntrinsicBackend::LlvmNvptx, IntrinsicBackend::LibNvvm]
            .into_iter()
            .map(|backend| {
                let (minimum_ptx, minimum_sm) =
                    packed_alu_backend_floor(format, operation, backend);
                crate::model::OverlayBackendLowering {
                    backend,
                    mechanism: BackendLoweringMechanism::InlinePtx,
                    evidence_profile: format!("{backend:?}-test"),
                    minimum_ptx: Some(minimum_ptx.into()),
                    minimum_sm: Some(minimum_sm.into()),
                }
            })
            .collect();
        record.packed_alu = Some(crate::model::PackedAlu {
            format,
            native_minimum_sm: recipe.native_minimum_sm,
            operation,
            adapter: PackedAluAdapter::DirectPackedU32,
        });
        record.expected_ptx = InstructionPattern::new(
            recipe.ptx_mnemonic.split('.').next().unwrap(),
            recipe.modifiers,
            vec![OperandPattern::Register; recipe.arity + 1],
        );
        record.summary = format!("packed {rust_module} arithmetic");
        record
    }

    fn packed_alu_declaration(
        format: PackedAluFormat,
        operation: PackedAluOperation,
    ) -> Option<ImportedIntrinsic> {
        let recipe = packed_alu_recipe(format, operation);
        let PackedAluRecipeSource::Imported {
            record,
            symbol,
            arguments,
            results,
            properties,
            selection,
            selection_asm,
            ..
        } = recipe.source
        else {
            return None;
        };
        let classes = if matches!(operation, PackedAluOperation::Min | PackedAluOperation::Max) {
            vec!["Intrinsic".into()]
        } else {
            vec!["Intrinsic".into(), "NVVMPureIntrinsic".into()]
        };
        let mut selections = vec![ImportedSelection {
            source_record: selection.into(),
            asm: selection_asm.into(),
            predicates: vec![
                format!("Subtarget->getSmVersion() >= {}", recipe.native_minimum_sm),
                format!(
                    "Subtarget->getPTXVersion() >= {}",
                    recipe.minimum_ptx.replace('.', "")
                ),
            ],
            constraints: Default::default(),
        }];
        if operation == PackedAluOperation::Abs {
            selections.extend((0..5).map(|index| ImportedSelection {
                source_record: format!("OTHER_ABS_{index}"),
                asm: "abs.f32 $dst, $src0;".into(),
                predicates: vec![],
                constraints: Default::default(),
            }));
        }
        Some(ImportedIntrinsic {
            source_record: record.into(),
            llvm_name: symbol.into(),
            arguments: arguments.iter().map(|value| (*value).into()).collect(),
            results: results.iter().map(|value| (*value).into()).collect(),
            classes,
            properties: properties.iter().map(|value| (*value).into()).collect(),
            selections,
        })
    }

    fn packed_conversion_policy(
        destination_format: PackedConversionDestinationFormat,
        rounding: PackedConversionRounding,
        saturation: PackedConversionSaturation,
    ) -> OverlayIntrinsic {
        let conversion = crate::model::PackedConversion {
            source_format: PackedConversionSourceFormat::F32x2,
            destination_format,
            rounding,
            saturation,
            adapter: PackedConversionAdapter::ReverseHighLowOperands,
        };
        let recipe = packed_conversion_recipe(&conversion).expect("test packed-conversion recipe");
        let mut record = policy();
        record.id = recipe.id.into();
        record.abi_id = recipe.abi_id.into();
        record.operation_key = recipe.operation_key.into();
        record.family = "packed_conversion".into();
        record.source_record = Some(recipe.source_record.into());
        record.rust_module = "convert".into();
        record.rust_name = recipe.rust_name.into();
        record.rust_arguments = vec!["f32".into(), "f32".into()];
        let result_width = packed_conversion_result_width(&conversion);
        record.rust_result = format!("u{result_width}");
        record.safe = true;
        record.must_use = false;
        record.safe_allowlist_reason = Some("the operation has no caller obligations".into());
        record.public_rust_path = format!("cuda_intrinsics::convert::{}", recipe.rust_name);
        record.compatibility_rust_paths = vec![recipe.compatibility_path.into()];
        record.dialect_op_type = recipe.dialect_op_type.into();
        record.dialect_op_name = recipe.dialect_op_name.into();
        record.dialect_operands = vec!["f32".into(), "f32".into()];
        record.dialect_results = vec![format!("i{result_width}")];
        record.llvm_symbol = Some(recipe.llvm_symbol.into());
        record.llvm_arguments = vec!["f32".into(), "f32".into()];
        record.llvm_results = vec![recipe.llvm_result.into()];
        record.pure = true;
        record.memory = "none".into();
        record.convergent = false;
        record.execution_scope = "thread".into();
        let (minimum_ptx, minimum_sm) = packed_conversion_floor(&conversion);
        record.minimum_ptx = minimum_ptx.into();
        record.minimum_sm = Some(minimum_sm.into());
        record.ptx_result = format!("u{result_width}");
        record.ptx_isa_section = "9.7.9.22 Data Movement and Conversion Instructions: cvt".into();
        record.ptx_isa_url = "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt".into();
        record.lowering = packed_conversion_lowering(&conversion).into();
        record.backend_lowerings = [IntrinsicBackend::LlvmNvptx, IntrinsicBackend::LibNvvm]
            .into_iter()
            .map(|backend| OverlayBackendLowering {
                backend,
                mechanism: packed_conversion_backend_mechanism(&conversion, backend),
                evidence_profile: "test".into(),
                minimum_ptx: Some(minimum_ptx.into()),
                minimum_sm: Some(minimum_sm.into()),
            })
            .collect();
        let modifiers = packed_conversion_ptx_modifiers(&conversion);
        record.packed_conversion = Some(conversion);
        record.expected_ptx =
            InstructionPattern::new("cvt", &modifiers, vec![OperandPattern::Register; 3]);
        record.summary = recipe.summary.into();
        record
    }

    fn packed_conversion_declaration(policy: &OverlayIntrinsic) -> ImportedIntrinsic {
        ImportedIntrinsic {
            source_record: policy.source_record.clone().unwrap(),
            llvm_name: policy.llvm_symbol.clone().unwrap(),
            arguments: policy.llvm_arguments.clone(),
            results: policy.llvm_results.clone(),
            classes: vec!["Intrinsic".into(), "NVVMPureIntrinsic".into()],
            properties: vec!["IntrNoMem".into(), "IntrSpeculatable".into()],
            selections: vec![],
        }
    }

    fn packed_conversion_evidence(policy: &OverlayIntrinsic) -> EvidenceRecord {
        let mut record = evidence();
        record.id = policy.id.clone();
        record.source_record = policy.source_record.clone();
        record.llvm_symbol = policy.llvm_symbol.clone();
        record.resolved_llvm_symbol = policy.resolved_llvm_symbol.clone();
        record.llvm_arguments = policy.llvm_arguments.clone();
        record.llvm_results = policy.llvm_results.clone();
        record.concrete_llvm_arguments = policy.llvm_arguments.clone();
        record.concrete_llvm_results = policy.llvm_results.clone();
        record.declaration_attributes_canonicalized = Some(true);
        record.expected_ptx = policy.expected_ptx.clone();
        record
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

    fn warp_barrier_policy() -> OverlayIntrinsic {
        packed_policy("sync_mask")
    }

    fn warp_barrier_declaration() -> ImportedIntrinsic {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let text = std::fs::read_to_string(repo_root.join("intrinsics/imported.json")).unwrap();
        serde_json::from_str::<ImportedFile>(&text)
            .unwrap()
            .intrinsics
            .into_iter()
            .find(|record| record.source_record == "int_nvvm_bar_warp_sync")
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
            legacy_pre_sm70: PreSm70MemberMaskRule::AllNamedLanesConvergedAndOnlyNamedLanesActive,
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

    fn warp_shuffle_policy(
        mode: WarpShuffleMode,
        value_kind: WarpShuffleValueKind,
    ) -> OverlayIntrinsic {
        let recipe = warp_shuffle_recipe(mode, value_kind);
        let mut record = policy();
        record.id = recipe.id.into();
        record.abi_id = recipe.abi_id.into();
        record.operation_key = recipe.operation_key.into();
        record.family = "warp_shuffle".into();
        match recipe.source {
            WarpShuffleRecipeSource::LlvmImported {
                source_record,
                llvm_symbol,
            } => {
                record.source_record = Some(source_record.into());
                record.llvm_symbol = Some(llvm_symbol.into());
                record.llvm_arguments = vec![
                    "i32".into(),
                    recipe.dialect_value.into(),
                    "i32".into(),
                    "i32".into(),
                ];
                record.llvm_results = vec![recipe.dialect_value.into()];
            }
            WarpShuffleRecipeSource::PtxNative { instruction } => {
                record.source = Some(IntrinsicSource::PtxNative {
                    instruction: instruction.into(),
                });
                record.source_record = None;
                record.llvm_symbol = None;
                record.llvm_arguments.clear();
                record.llvm_results.clear();
            }
        }
        record.rust_module = "warp".into();
        record.rust_name = recipe.rust_name.into();
        record.rust_arguments = vec!["u32".into(), recipe.rust_value.into(), "u32".into()];
        record.rust_result = recipe.rust_value.into();
        record.safe = false;
        record.must_use = true;
        record.safe_allowlist_reason = None;
        record.public_rust_path = format!("cuda_intrinsics::warp::{}", recipe.rust_name);
        record.compatibility_rust_paths = vec![format!("cuda_device::warp::{}", recipe.rust_name)];
        record.dialect_op_type = recipe.dialect_op_type.into();
        record.dialect_op_name = recipe.dialect_op_name.into();
        record.dialect_operands = vec!["i32".into(), recipe.dialect_value.into(), "i32".into()];
        record.dialect_results = vec![recipe.dialect_value.into()];
        record.pure = false;
        record.memory = "inaccessible_read_write".into();
        record.convergent = true;
        record.execution_scope = "warp".into();
        record.minimum_ptx = "6.0".into();
        record.minimum_sm = Some("sm_30".into());
        record.ptx_result = recipe.rust_value.into();
        record.targets = "all".into();
        record.ptx_isa_section =
            "9.7.9.6 Data Movement and Conversion Instructions: shfl.sync".into();
        record.ptx_isa_url = "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync".into();
        record.lowering = recipe.lowering.into();
        record.backend_lowerings = vec![
            crate::model::OverlayBackendLowering {
                backend: IntrinsicBackend::LlvmNvptx,
                mechanism: recipe.backend_mechanism,
                evidence_profile: "llvm-test".into(),
                minimum_ptx: Some("6.0".into()),
                minimum_sm: Some("sm_30".into()),
            },
            crate::model::OverlayBackendLowering {
                backend: IntrinsicBackend::LibNvvm,
                mechanism: recipe.backend_mechanism,
                evidence_profile: "libnvvm-test".into(),
                minimum_ptx: Some("6.0".into()),
                minimum_sm: Some("sm_75".into()),
            },
        ];
        record.warp_shuffle = Some(crate::model::WarpShuffle {
            mode,
            value_kind,
            participation:
                WarpShuffleParticipation::ExecutingLaneNamedAllNamedLanesSameInstructionAndMask,
            legacy_pre_sm70: PreSm70MemberMaskRule::AllNamedLanesConvergedAndOnlyNamedLanesActive,
            source_lane: WarpShuffleSourceLane::InRangeSourceActiveAndNamedOutOfRangeCopiesSelf,
            adapter: recipe.adapter,
            clamp: recipe.clamp,
            lane_encoding: recipe.operand_encoding,
            mask_encoding: recipe.operand_encoding,
        });
        let operands = match recipe.adapter {
            WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp => vec![
                OperandPattern::Register,
                OperandPattern::Register,
                OperandPattern::RegisterOrImmediate,
                OperandPattern::Exact {
                    value: recipe.clamp.to_string(),
                },
                OperandPattern::RegisterOrImmediate,
            ],
            WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble => {
                vec![
                    OperandPattern::Exact { value: "lo".into() },
                    OperandPattern::Exact { value: "lo".into() },
                    OperandPattern::Register,
                    OperandPattern::Exact {
                        value: recipe.clamp.to_string(),
                    },
                    OperandPattern::Register,
                ]
            }
        };
        record.expected_ptx =
            InstructionPattern::new("shfl", &["sync", recipe.ptx_mode, "b32"], operands);
        record.summary = "synchronized warp shuffle".into();
        record
    }

    fn warp_shuffle_declaration(
        mode: WarpShuffleMode,
        value_kind: WarpShuffleValueKind,
    ) -> ImportedIntrinsic {
        let recipe = warp_shuffle_recipe(mode, value_kind);
        let WarpShuffleRecipeSource::LlvmImported {
            source_record,
            llvm_symbol,
        } = recipe.source
        else {
            panic!("PTX-native i64 shuffles have no imported declaration");
        };
        let selections = (0..8)
            .map(|index| ImportedSelection {
                source_record: format!("anonymous_test_{index}"),
                asm: format!(
                    "shfl.sync.{}.b32 \t$dst, $src, $offset, $mask, $threadmask;",
                    recipe.ptx_mode
                ),
                predicates: vec![
                    "Subtarget->getPTXVersion() >= 60".into(),
                    "Subtarget->getSmVersion() >= 30".into(),
                ],
                constraints: Default::default(),
            })
            .collect();
        ImportedIntrinsic {
            source_record: source_record.into(),
            llvm_name: llvm_symbol.into(),
            arguments: vec![
                "i32".into(),
                recipe.dialect_value.into(),
                "i32".into(),
                "i32".into(),
            ],
            results: vec![recipe.dialect_value.into()],
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
            selections,
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
    fn legacy_evidence_schema_is_unchanged_and_rejects_matrix_fields() {
        let legacy = EvidenceFile {
            schema: 5,
            backend_profile: "legacy".into(),
            backend_kind: None,
            llvm_revision: "test".into(),
            backend_version: "LLVM legacy test".into(),
            backend_sha256: "0123456789abcdef".into(),
            artifact_path: None,
            build_id_prefix: None,
            nvvm_ir_version: None,
            debug_ir_version: None,
            records: vec![evidence()],
        };
        let bytes = serde_json::to_vec(&legacy).unwrap();
        assert_eq!(parse_evidence_bytes(&bytes, "legacy").unwrap(), legacy);

        let mut with_matrix_field = serde_json::to_value(&legacy).unwrap();
        with_matrix_field["matrices"] = serde_json::json!([]);
        let error = parse_synthetic_evidence(&with_matrix_field).unwrap_err();
        assert!(error.to_string().contains("legacy evidence"));
        assert!(format!("{error:#}").contains("unknown field"));
    }

    #[test]
    fn compact_evidence_matrix_equals_explicit_records() {
        let expanded = parse_synthetic_evidence(&synthetic_matrix_json()).unwrap();
        let mut expected = vec![EvidenceRecord {
            id: "synthetic_explicit".into(),
            source: None,
            source_record: Some("int_synthetic_explicit".into()),
            llvm_symbol: Some("llvm.synthetic.explicit".into()),
            resolved_llvm_symbol: None,
            llvm_arguments: vec!["i32".into()],
            llvm_results: vec!["i32".into()],
            concrete_llvm_arguments: vec![],
            concrete_llvm_results: vec![],
            target_triple: "nvptx64-nvidia-cuda".into(),
            gpu_target: "sm_80".into(),
            ptx_feature: "+ptx71".into(),
            status: "lowered".into(),
            stages: vec![],
            declaration_attributes_canonicalized: None,
            runtime_validation: None,
            expected_ptx: InstructionPattern {
                mnemonic: "mma".into(),
                modifiers: vec!["sync".into(), "explicit".into()],
                operands: vec![OperandPattern::Register],
            },
        }];
        expected.extend(["s8", "u8"].into_iter().map(|element| EvidenceRecord {
            id: format!("synthetic_{element}"),
            source: None,
            source_record: Some(format!("int_synthetic_{element}")),
            llvm_symbol: Some(format!("llvm.synthetic.{element}")),
            resolved_llvm_symbol: None,
            llvm_arguments: vec!["i32".into()],
            llvm_results: vec!["i32".into()],
            concrete_llvm_arguments: vec![],
            concrete_llvm_results: vec![],
            target_triple: "nvptx64-nvidia-cuda".into(),
            gpu_target: "sm_80".into(),
            ptx_feature: "+ptx71".into(),
            status: "lowered".into(),
            stages: vec![shared_matrix_stage()],
            declaration_attributes_canonicalized: None,
            runtime_validation: None,
            expected_ptx: InstructionPattern {
                mnemonic: "mma".into(),
                modifiers: vec!["sync".into(), element.into()],
                operands: vec![OperandPattern::Register],
            },
        }));
        assert_eq!(expanded.schema, 6);
        assert_eq!(expanded.records, expected);
    }

    #[test]
    fn matrix_identity_mutations_reach_existing_evidence_validation() {
        let mut expanded = parse_synthetic_evidence(&policy_matrix_json()).unwrap();
        let record = expanded.records.pop().unwrap();
        validate_test_evidence(&policy(), record.clone()).unwrap();

        let mut wrong_source = record.clone();
        wrong_source.source_record = Some("int_nvvm_read_ptx_sreg_tid_y".into());
        assert!(
            validate_test_evidence(&policy(), wrong_source)
                .unwrap_err()
                .to_string()
                .contains("source provenance mismatch")
        );

        let mut wrong_symbol = record.clone();
        wrong_symbol.llvm_symbol = Some("llvm.nvvm.read.ptx.sreg.tid.y".into());
        assert!(
            validate_test_evidence(&policy(), wrong_symbol)
                .unwrap_err()
                .to_string()
                .contains("signature mismatch")
        );

        let mut wrong_signature = record.clone();
        wrong_signature.llvm_arguments.push("i32".into());
        assert!(
            validate_test_evidence(&policy(), wrong_signature)
                .unwrap_err()
                .to_string()
                .contains("signature mismatch")
        );

        let mut wrong_ptx = record;
        wrong_ptx.expected_ptx.modifiers.push("changed".into());
        assert!(
            validate_test_evidence(&policy(), wrong_ptx)
                .unwrap_err()
                .to_string()
                .contains("PTX expectation mismatch")
        );
    }

    #[test]
    fn evidence_matrix_rejects_bad_counts_fixtures_placeholders_and_collisions() {
        let base = synthetic_matrix_json();

        let mut bad_product = base.clone();
        bad_product["matrices"][0]["product_count"] = 3.into();
        assert_synthetic_evidence_error(&bad_product, "expands to 2 records");

        let mut unknown_fixture = base.clone();
        unknown_fixture["matrices"][0]["fixtures"][0] = "missing".into();
        assert_synthetic_evidence_error(&unknown_fixture, "unknown fixture");

        let mut uncovered_fixture = base.clone();
        let extra = uncovered_fixture["fixtures"][0].clone();
        uncovered_fixture["fixtures"]
            .as_array_mut()
            .unwrap()
            .push(extra);
        uncovered_fixture["fixtures"][1]["id"] = "unused".into();
        assert_synthetic_evidence_error(&uncovered_fixture, "not referenced");

        let mut wrong_coverage = base.clone();
        wrong_coverage["fixtures"][0]["coverage_count"] = 1.into();
        assert_synthetic_evidence_error(&wrong_coverage, "covers 2 expanded records");

        let mut malformed = base.clone();
        malformed["matrices"][0]["template"]["id"] = "synthetic_$element".into();
        assert_synthetic_evidence_error(&malformed, "malformed matrix placeholder");

        let mut unknown_axis = base.clone();
        unknown_axis["matrices"][0]["template"]["id"] = "synthetic_${other}".into();
        assert_synthetic_evidence_error(&unknown_axis, "unknown matrix axis");

        let mut collision = base.clone();
        collision["matrices"][0]["template"]["id"] = "synthetic".into();
        assert_synthetic_evidence_error(&collision, "duplicate expanded evidence ID");
    }

    #[test]
    fn exact_operand_matrix_placeholders_fail_closed() {
        let base = policy_matrix_json();

        let mut unknown = base.clone();
        unknown["matrices"][0]["template"]["expected_ptx"]["operands"][1]["value"] =
            "%tid.${other}".into();
        assert_synthetic_evidence_error(&unknown, "unknown matrix axis other");

        let mut unterminated = base.clone();
        unterminated["matrices"][0]["template"]["expected_ptx"]["operands"][1]["value"] =
            "%tid.${axis".into();
        assert_synthetic_evidence_error(&unterminated, "unterminated matrix placeholder");

        let mut disallowed = base;
        disallowed["matrices"][0]["template"]["expected_ptx"]["mnemonic"] = "mov.${axis}".into();
        assert_synthetic_evidence_error(&disallowed, "PTX mnemonic cannot contain");
    }

    #[test]
    fn evidence_matrix_rejects_bad_axes_fixture_ids_and_stage_conflicts() {
        let base = synthetic_matrix_json();

        let mut no_fixture = base.clone();
        no_fixture["matrices"][0]["fixtures"] = serde_json::json!([]);
        assert_synthetic_evidence_error(&no_fixture, "references no shared fixture");

        let mut empty_axes = base.clone();
        empty_axes["matrices"][0]["axes"] = serde_json::json!([]);
        assert_synthetic_evidence_error(&empty_axes, "has no axes");

        let mut duplicate_axis = base.clone();
        let axis = duplicate_axis["matrices"][0]["axes"][0].clone();
        duplicate_axis["matrices"][0]["axes"]
            .as_array_mut()
            .unwrap()
            .push(axis);
        duplicate_axis["matrices"][0]["product_count"] = 4.into();
        assert_synthetic_evidence_error(&duplicate_axis, "axes must be unique and sorted");

        let mut empty_values = base.clone();
        empty_values["matrices"][0]["axes"][0]["values"] = serde_json::json!([]);
        assert_synthetic_evidence_error(&empty_values, "has no values");

        let mut empty_axis_name = base.clone();
        empty_axis_name["matrices"][0]["axes"][0]["name"] = "".into();
        assert_synthetic_evidence_error(&empty_axis_name, "is not a safe token");

        let mut empty_value = base.clone();
        empty_value["matrices"][0]["axes"][0]["values"][0] = "".into();
        assert_synthetic_evidence_error(&empty_value, "unsafe value");

        let mut duplicate_value = base.clone();
        duplicate_value["matrices"][0]["axes"][0]["values"][1] = "s8".into();
        assert_synthetic_evidence_error(&duplicate_value, "duplicate value");

        let mut unsafe_value = base.clone();
        unsafe_value["matrices"][0]["axes"][0]["values"][0] = "../s8".into();
        assert_synthetic_evidence_error(&unsafe_value, "unsafe value");

        let mut unused_axis = base.clone();
        unused_axis["matrices"][0]["axes"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"name": "other", "values": ["x"]}));
        assert_synthetic_evidence_error(&unused_axis, "unused axis");

        let mut duplicate_fixture = base.clone();
        let fixture = duplicate_fixture["fixtures"][0].clone();
        duplicate_fixture["fixtures"]
            .as_array_mut()
            .unwrap()
            .push(fixture);
        assert_synthetic_evidence_error(&duplicate_fixture, "duplicate evidence fixture ID");

        let mut fixture_placeholder = base.clone();
        fixture_placeholder["fixtures"][0]["stages"][0]["detail"] = "covers ${element}".into();
        assert_synthetic_evidence_error(&fixture_placeholder, "cannot contain matrix placeholders");

        let mut missing_symbol = base.clone();
        missing_symbol["matrices"][0]["template"]
            .as_object_mut()
            .unwrap()
            .remove("llvm_symbol");
        assert_synthetic_evidence_error(&missing_symbol, "missing field `llvm_symbol`");

        let mut conflicting_stage = base;
        conflicting_stage["matrices"][0]["template"]["facts"]["stages"] =
            conflicting_stage["fixtures"][0]["stages"].clone();
        assert_synthetic_evidence_error(&conflicting_stage, "conflicting duplicate");
    }

    #[test]
    fn overloaded_symbols_require_distinct_resolved_identities() {
        let bf16 = packed_alu_policy(PackedAluFormat::Bf16x2, PackedAluOperation::Abs);
        let f16 = packed_alu_policy(PackedAluFormat::F16x2, PackedAluOperation::Abs);
        validate_unique_overlay(&[bf16.clone(), f16.clone()], 1).unwrap();

        let mut unresolved = f16.clone();
        unresolved.resolved_llvm_symbol = None;
        let error = validate_unique_overlay(&[bf16.clone(), unresolved], 1).unwrap_err();
        assert!(error.to_string().contains("without a resolved symbol"));

        let mut duplicate = f16;
        duplicate.resolved_llvm_symbol = bf16.resolved_llvm_symbol.clone();
        let error = validate_unique_overlay(&[bf16, duplicate], 1).unwrap_err();
        assert!(error.to_string().contains("duplicate resolved LLVM symbol"));
    }

    #[test]
    fn overlay_manifest_loads_sorted_family_shards() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let (overlay, hash) =
            read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml")).unwrap();
        assert_eq!(overlay.schema, OVERLAY_SCHEMA);
        assert_eq!(overlay.shards.len(), 42);
        assert_eq!(overlay.intrinsics.len(), 316);
        assert_eq!(
            overlay
                .intrinsics
                .iter()
                .filter(|record| record.family == "prmt")
                .count(),
            7
        );
        assert_eq!(
            overlay
                .intrinsics
                .iter()
                .filter(|record| record.family == "debug_control")
                .count(),
            3
        );
        assert_eq!(
            overlay
                .intrinsics
                .iter()
                .filter(|record| record.family == "stmatrix")
                .count(),
            4
        );
        assert_eq!(
            overlay
                .intrinsics
                .iter()
                .filter(|record| record.family == "packed_alu")
                .count(),
            18
        );
        assert_eq!(
            overlay
                .intrinsics
                .iter()
                .filter(|record| record.family == "packed_conversion")
                .count(),
            10
        );
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
                .filter(|record| record.family == "register_mma")
                .count(),
            58
        );
        assert_eq!(
            overlay
                .intrinsics
                .iter()
                .filter(|record| record.family == "sparse_mma")
                .count(),
            89
        );
        assert_eq!(
            overlay
                .intrinsics
                .iter()
                .filter(|record| record.family == "cp_async_copy")
                .count(),
            8
        );
        assert_eq!(
            overlay
                .intrinsics
                .iter()
                .filter(|record| record.family == "cp_async_control")
                .count(),
            3
        );
        assert_eq!(
            overlay
                .intrinsics
                .iter()
                .filter(|record| record.family == "cp_async_mbarrier")
                .count(),
            4
        );
        assert_eq!(
            overlay
                .intrinsics
                .iter()
                .filter(|record| record.family == "mbarrier_basic")
                .count(),
            4
        );
        assert_eq!(
            overlay
                .intrinsics
                .iter()
                .filter(|record| record.family == "sync")
                .count(),
            4
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
                .filter(|record| record.family == "warp_barrier")
                .count(),
            1
        );
        assert_eq!(
            overlay
                .intrinsics
                .iter()
                .filter(|record| record.family == "warp_match")
                .count(),
            4
        );
        assert_eq!(
            overlay
                .intrinsics
                .iter()
                .filter(|record| record.family == "warp_shuffle")
                .count(),
            12
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

    fn test_f8f6f4_admission() -> SparseMmaF8F6F4Admission {
        let formats = vec![
            SparseMmaElement::E2m1,
            SparseMmaElement::E2m3,
            SparseMmaElement::E3m2,
            SparseMmaElement::E4m3,
            SparseMmaElement::E5m2,
        ];
        SparseMmaF8F6F4Admission {
            llvm_evidence_profile: "llvm-test".into(),
            libnvvm_evidence_profile: "libnvvm-test".into(),
            runtime_validation: RuntimeValidation::Unexecuted,
            a_elements: formats.clone(),
            b_elements: formats,
            product_count: 25,
        }
    }

    fn test_prmt_admission() -> PrmtAdmission {
        let variants = [
            PrmtMode::Generic,
            PrmtMode::F4e,
            PrmtMode::B4e,
            PrmtMode::Rc8,
            PrmtMode::Ecl,
            PrmtMode::Ecr,
            PrmtMode::Rc16,
        ]
        .map(|mode| crate::model::PrmtAdmissionVariant {
            abi_id: prmt_recipe(mode).abi_id.into(),
            mode,
        })
        .into();
        PrmtAdmission {
            llvm_evidence_profile: "llvm-test".into(),
            libnvvm_evidence_profile: "libnvvm-test".into(),
            runtime_validation: RuntimeValidation::Unexecuted,
            variants,
        }
    }

    fn test_fp8_conversion_admission() -> PackedConversionFp8Admission {
        PackedConversionFp8Admission {
            llvm_evidence_profile: "llvm-test".into(),
            libnvvm_evidence_profile: "libnvvm-test".into(),
            runtime_validation: RuntimeValidation::Unexecuted,
            destination_formats: vec![
                PackedConversionDestinationFormat::E4m3x2,
                PackedConversionDestinationFormat::E5m2x2,
            ],
            saturations: vec![
                PackedConversionSaturation::Satfinite,
                PackedConversionSaturation::SatfiniteRelu,
            ],
            product_count: 4,
        }
    }

    fn test_cluster_barrier_admission() -> ClusterBarrierAdmission {
        let variants = [
            ClusterBarrierMode::Arrive,
            ClusterBarrierMode::ArriveAligned,
            ClusterBarrierMode::ArriveRelaxed,
            ClusterBarrierMode::ArriveRelaxedAligned,
            ClusterBarrierMode::Wait,
            ClusterBarrierMode::WaitAligned,
        ]
        .map(|mode| crate::model::ClusterBarrierAdmissionVariant {
            abi_id: cluster_barrier_recipe(mode).abi_id.into(),
            mode,
        })
        .into();
        ClusterBarrierAdmission {
            llvm_evidence_profile: "llvm-test".into(),
            libnvvm_evidence_profile: "libnvvm-test".into(),
            runtime_validation: RuntimeValidation::Unexecuted,
            variants,
        }
    }

    fn test_special_register_admission() -> SpecialRegisterAdmission {
        SpecialRegisterAdmission {
            llvm_evidence_profile: "rust-llvm-22.1.2-1cb4e383".into(),
            libnvvm_evidence_profile: "cuda-13.3-libnvvm-13.3.33".into(),
            runtime_validation: RuntimeValidation::Unexecuted,
            registers: REVIEWED_SPECIAL_REGISTERS.into(),
            product_count: REVIEWED_SPECIAL_REGISTERS.len(),
        }
    }

    fn test_debug_control_admission() -> DebugControlAdmission {
        DebugControlAdmission {
            llvm_evidence_profile: "llvm-debug-test".into(),
            libnvvm_evidence_profile: "libnvvm-debug-test".into(),
            runtime_validation: RuntimeValidation::Unexecuted,
            operations: vec![
                DebugControlOperation::Trap,
                DebugControlOperation::Breakpoint,
                DebugControlOperation::Pmevent,
            ],
            abi_ids: vec!["i9001".into(), "i9002".into(), "i9003".into()],
        }
    }

    fn test_clc_admission() -> ClcAdmission {
        let operations = [
            ClcOperation::TryCancel,
            ClcOperation::TryCancelMulticast,
            ClcOperation::QueryIsCanceled,
            ClcOperation::QueryGetFirstCtaidX,
            ClcOperation::QueryGetFirstCtaidY,
            ClcOperation::QueryGetFirstCtaidZ,
        ];
        ClcAdmission {
            llvm_evidence_profile: "llvm-clc-test".into(),
            libnvvm_evidence_profile: "libnvvm-clc-test".into(),
            runtime_validation: RuntimeValidation::Unexecuted,
            variants: operations
                .into_iter()
                .map(|operation| crate::model::ClcAdmissionVariant {
                    abi_id: clc_recipe(operation).abi_id.into(),
                    operation,
                })
                .collect(),
        }
    }

    fn test_threadfence_admission() -> ThreadfenceAdmission {
        ThreadfenceAdmission {
            llvm_evidence_profile: "llvm-test".into(),
            libnvvm_evidence_profile: "libnvvm-test".into(),
            runtime_validation: RuntimeValidation::Unexecuted,
            variants: vec![
                crate::model::ThreadfenceAdmissionVariant {
                    abi_id: "i0298".into(),
                    scope: ThreadfenceScope::Cta,
                },
                crate::model::ThreadfenceAdmissionVariant {
                    abi_id: "i0299".into(),
                    scope: ThreadfenceScope::Device,
                },
                crate::model::ThreadfenceAdmissionVariant {
                    abi_id: "i0300".into(),
                    scope: ThreadfenceScope::System,
                },
            ],
        }
    }

    fn test_cluster_memory_admission() -> ClusterMemoryAdmission {
        ClusterMemoryAdmission {
            llvm_evidence_profile: "llvm-cluster-memory-test".into(),
            libnvvm_evidence_profile: "libnvvm-cluster-memory-test".into(),
            runtime_validation: RuntimeValidation::Unexecuted,
            variants: vec![
                crate::model::ClusterMemoryAdmissionVariant {
                    abi_id: "i0320".into(),
                    operation: ClusterMemoryOperation::MapSharedRank,
                },
                crate::model::ClusterMemoryAdmissionVariant {
                    abi_id: "i0321".into(),
                    operation: ClusterMemoryOperation::ReadU32,
                },
            ],
        }
    }

    fn test_stmatrix_admission() -> StmatrixAdmission {
        let variants = [
            (StmatrixMultiplicity::X2, StmatrixLayout::Normal, "i0301"),
            (
                StmatrixMultiplicity::X2,
                StmatrixLayout::Transposed,
                "i0302",
            ),
            (StmatrixMultiplicity::X4, StmatrixLayout::Normal, "i0303"),
            (
                StmatrixMultiplicity::X4,
                StmatrixLayout::Transposed,
                "i0304",
            ),
        ]
        .map(
            |(multiplicity, layout, abi_id)| crate::model::StmatrixAdmissionVariant {
                abi_id: abi_id.into(),
                multiplicity,
                layout,
            },
        )
        .into();
        StmatrixAdmission {
            llvm_evidence_profile: "llvm-test".into(),
            libnvvm_evidence_profile: "libnvvm-test".into(),
            runtime_validation: RuntimeValidation::Unexecuted,
            variants,
        }
    }

    fn test_mbarrier_extended_admission() -> MbarrierExtendedAdmission {
        let variants = [
            MbarrierExtendedOperation::ArriveExpectTxCta,
            MbarrierExtendedOperation::ArriveExpectTxCluster,
            MbarrierExtendedOperation::ArriveRemoteCluster,
            MbarrierExtendedOperation::TryWaitTokenCta,
            MbarrierExtendedOperation::TryWaitParityCta,
            MbarrierExtendedOperation::TryWaitParityCluster,
            MbarrierExtendedOperation::FenceProxyAsyncSharedCta,
            MbarrierExtendedOperation::FenceMbarrierInitReleaseCluster,
            MbarrierExtendedOperation::FenceProxyAsyncGenericReleaseSharedCtaCluster,
            MbarrierExtendedOperation::FenceProxyAsyncGenericAcquireSharedClusterCluster,
            MbarrierExtendedOperation::Nanosleep,
        ]
        .map(|operation| crate::model::MbarrierExtendedAdmissionVariant {
            abi_id: mbarrier_extended_recipe(operation).abi_id.into(),
            operation,
        })
        .into();
        MbarrierExtendedAdmission {
            llvm_evidence_profile: "llvm-test".into(),
            libnvvm_evidence_profile: "libnvvm-test".into(),
            runtime_validation: RuntimeValidation::Unexecuted,
            variants,
        }
    }

    #[test]
    fn overlay_shard_schema_range_is_composable_and_new_fields_fail_closed() {
        let shard = |schema, sparse_mma_f8f6f4_f32, prmt| OverlayShardFile {
            schema,
            family: "sparse_mma".into(),
            intrinsics: vec![],
            register_mma_int4: None,
            register_mma_int8: None,
            register_mma_b1: None,
            sparse_mma_integer: None,
            sparse_mma_f8f6f4_f32,
            prmt,
            packed_conversion_fp8: None,
            cluster_sreg: None,
            cluster_barrier: None,
            mbarrier_extended: None,
            special_registers: None,
            debug_control: None,
            threadfence: None,
            cluster_memory: None,
            stmatrix: None,
            clc: None,
        };
        let path = Path::new("intrinsics/overlay/test.toml");
        validate_overlay_shard_schema(&shard(26, None, None), path).unwrap();
        validate_overlay_shard_schema(&shard(27, None, None), path).unwrap();
        validate_overlay_shard_schema(&shard(28, None, None), path).unwrap();
        validate_overlay_shard_schema(&shard(29, None, None), path).unwrap();
        validate_overlay_shard_schema(&shard(30, None, None), path).unwrap();
        validate_overlay_shard_schema(&shard(31, None, None), path).unwrap();
        validate_overlay_shard_schema(&shard(32, None, None), path).unwrap();
        validate_overlay_shard_schema(&shard(33, None, None), path).unwrap();
        validate_overlay_shard_schema(&shard(34, None, None), path).unwrap();
        validate_overlay_shard_schema(&shard(35, None, None), path).unwrap();
        validate_overlay_shard_schema(&shard(27, Some(test_f8f6f4_admission()), None), path)
            .unwrap();
        validate_overlay_shard_schema_with_max(
            &shard(27, Some(test_f8f6f4_admission()), None),
            path,
            30,
        )
        .unwrap();
        validate_overlay_shard_schema(&shard(28, None, Some(test_prmt_admission())), path).unwrap();
        validate_overlay_shard_schema_with_max(
            &shard(28, None, Some(test_prmt_admission())),
            path,
            30,
        )
        .unwrap();

        assert!(validate_overlay_shard_schema(&shard(25, None, None), path).is_err());
        for schema in 35..=OVERLAY_SHARD_SCHEMA {
            validate_overlay_shard_schema(&shard(schema, None, None), path).unwrap();
        }
        assert!(
            validate_overlay_shard_schema(&shard(OVERLAY_SHARD_SCHEMA + 1, None, None), path)
                .is_err()
        );
        let error =
            validate_overlay_shard_schema(&shard(26, Some(test_f8f6f4_admission()), None), path)
                .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires overlay shard schema 27")
        );
        let error =
            validate_overlay_shard_schema(&shard(27, None, Some(test_prmt_admission())), path)
                .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires overlay shard schema 28")
        );

        let fp8_shard = |schema| OverlayShardFile {
            schema,
            family: "packed_conversion".into(),
            intrinsics: vec![],
            register_mma_int4: None,
            register_mma_int8: None,
            register_mma_b1: None,
            sparse_mma_integer: None,
            sparse_mma_f8f6f4_f32: None,
            prmt: None,
            packed_conversion_fp8: Some(test_fp8_conversion_admission()),
            cluster_sreg: None,
            cluster_barrier: None,
            mbarrier_extended: None,
            special_registers: None,
            debug_control: None,
            threadfence: None,
            cluster_memory: None,
            stmatrix: None,
            clc: None,
        };
        validate_overlay_shard_schema(&fp8_shard(29), path).unwrap();
        validate_overlay_shard_schema_with_max(&fp8_shard(29), path, 30).unwrap();
        let error = validate_overlay_shard_schema(&fp8_shard(28), path).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires overlay shard schema 29")
        );

        let cluster_shard = OverlayShardFile {
            schema: 31,
            family: "cluster_barrier".into(),
            intrinsics: vec![],
            register_mma_int4: None,
            register_mma_int8: None,
            register_mma_b1: None,
            sparse_mma_integer: None,
            sparse_mma_f8f6f4_f32: None,
            prmt: None,
            packed_conversion_fp8: None,
            cluster_sreg: None,
            cluster_barrier: Some(test_cluster_barrier_admission()),
            mbarrier_extended: None,
            special_registers: None,
            debug_control: None,
            threadfence: None,
            cluster_memory: None,
            stmatrix: None,
            clc: None,
        };
        validate_overlay_shard_schema_with_max(&cluster_shard, path, 31).unwrap();
        let mut old_cluster_shard = cluster_shard;
        old_cluster_shard.schema = 30;
        assert!(
            validate_overlay_shard_schema_with_max(&old_cluster_shard, path, 31)
                .unwrap_err()
                .to_string()
                .contains("requires overlay shard schema 31")
        );

        let extended_shard = OverlayShardFile {
            schema: MBARRIER_EXTENDED_SHARD_SCHEMA,
            family: "mbarrier_extended".into(),
            intrinsics: vec![],
            register_mma_int4: None,
            register_mma_int8: None,
            register_mma_b1: None,
            sparse_mma_integer: None,
            sparse_mma_f8f6f4_f32: None,
            prmt: None,
            packed_conversion_fp8: None,
            cluster_sreg: None,
            cluster_barrier: None,
            mbarrier_extended: Some(test_mbarrier_extended_admission()),
            special_registers: None,
            debug_control: None,
            threadfence: None,
            cluster_memory: None,
            stmatrix: None,
            clc: None,
        };
        validate_overlay_shard_schema_with_max(
            &extended_shard,
            path,
            MBARRIER_EXTENDED_SHARD_SCHEMA,
        )
        .unwrap();
        let mut old_extended_shard = extended_shard;
        old_extended_shard.schema = MBARRIER_EXTENDED_SHARD_SCHEMA - 1;
        assert!(
            validate_overlay_shard_schema_with_max(
                &old_extended_shard,
                path,
                MBARRIER_EXTENDED_SHARD_SCHEMA,
            )
            .unwrap_err()
            .to_string()
            .contains("requires overlay shard schema 40")
        );
    }

    #[test]
    fn stmatrix_admission_is_closed_and_uses_schema_35() {
        let shard = |schema| {
            toml::from_str::<OverlayShardFile>(&format!(
                r#"
schema = {schema}
family = "stmatrix"

[stmatrix]
llvm_evidence_profile = "llvm-test"
libnvvm_evidence_profile = "libnvvm-test"
runtime_validation = "unexecuted"

[[stmatrix.variant]]
abi_id = "i0301"
multiplicity = "x2"
layout = "normal"

[[stmatrix.variant]]
abi_id = "i0302"
multiplicity = "x2"
layout = "transposed"

[[stmatrix.variant]]
abi_id = "i0303"
multiplicity = "x4"
layout = "normal"

[[stmatrix.variant]]
abi_id = "i0304"
multiplicity = "x4"
layout = "transposed"
"#
            ))
            .unwrap()
        };
        let path = Path::new("intrinsics/overlay/stmatrix.toml");

        let old = shard(STMATRIX_SHARD_SCHEMA - 1);
        assert!(
            validate_overlay_shard_schema(&old, path)
                .unwrap_err()
                .to_string()
                .contains("requires overlay shard schema 35")
        );

        let current = shard(STMATRIX_SHARD_SCHEMA);
        validate_overlay_shard_schema(&current, path).unwrap();
        let admission = current.stmatrix.unwrap();
        let records = expand_stmatrix_admission(&admission).unwrap();
        assert_eq!(
            records
                .iter()
                .map(|record| (record.abi_id.as_str(), record.id.as_str()))
                .collect::<Vec<_>>(),
            [
                ("i0301", "stmatrix_m8n8_x2_b16"),
                ("i0302", "stmatrix_m8n8_x2_trans_b16"),
                ("i0303", "stmatrix_m8n8_x4_b16"),
                ("i0304", "stmatrix_m8n8_x4_trans_b16"),
            ]
        );

        let mut reordered = admission.clone();
        reordered.variants.swap(0, 1);
        assert!(expand_stmatrix_admission(&reordered).is_err());

        let mut wrong_id = admission.clone();
        wrong_id.variants[0].abi_id = "i0302".into();
        assert!(expand_stmatrix_admission(&wrong_id).is_err());

        let mut executed = admission;
        executed.runtime_validation = RuntimeValidation::Executed;
        assert!(expand_stmatrix_admission(&executed).is_err());
    }

    #[test]
    fn pinned_stmatrix_records_match_llvm_and_reject_contract_drift() {
        let records = expand_stmatrix_admission(&test_stmatrix_admission()).unwrap();
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let imported: ImportedFile =
            read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
        let declarations = imported
            .intrinsics
            .iter()
            .map(|record| (record.source_record.as_str(), record))
            .collect::<BTreeMap<_, _>>();

        for record in &records {
            let declaration = declarations[record.source_record.as_deref().unwrap()];
            assert!(declaration.selections.is_empty());
            validate_imported_policy(record, declaration).unwrap();
        }

        let declaration = declarations["int_nvvm_stmatrix_sync_aligned_m8n8_x2_b16"];
        let mut changed = records[0].clone();
        changed.memory = "read_write".into();
        assert!(validate_imported_policy(&changed, declaration).is_err());

        let mut changed = records[0].clone();
        changed.convergent = false;
        assert!(validate_imported_policy(&changed, declaration).is_err());

        let mut changed = records[0].clone();
        changed.backend_lowerings[0].mechanism = BackendLoweringMechanism::InlinePtx;
        assert!(validate_imported_policy(&changed, declaration).is_err());

        let mut changed = records[0].clone();
        changed.minimum_sm = Some("sm_80".into());
        assert!(validate_imported_policy(&changed, declaration).is_err());
    }

    #[test]
    fn cluster_sreg_admission_uses_its_fixed_introduction_schema() {
        let shard = |schema| {
            toml::from_str::<OverlayShardFile>(&format!(
                r#"
schema = {schema}
family = "sreg"

[cluster_sreg]
axes = ["x", "y", "z"]
xyz_product_count = 12
record_count = 14
"#
            ))
            .unwrap()
        };
        let path = Path::new("intrinsics/overlay/sreg_cluster.toml");

        let old = shard(CLUSTER_SREG_SHARD_SCHEMA - 1);
        assert!(
            validate_overlay_shard_schema(&old, path)
                .unwrap_err()
                .to_string()
                .contains("requires overlay shard schema 30")
        );

        let current = shard(CLUSTER_SREG_SHARD_SCHEMA);
        validate_overlay_shard_schema(&current, path).unwrap();
        let admission = current.cluster_sreg.unwrap();
        assert_eq!(expand_cluster_sreg_admission(&admission).unwrap().len(), 14);

        let mut wrong_axes = admission.clone();
        wrong_axes.axes.swap(0, 1);
        assert!(expand_cluster_sreg_admission(&wrong_axes).is_err());

        let mut wrong_count = admission;
        wrong_count.record_count = 13;
        assert!(expand_cluster_sreg_admission(&wrong_count).is_err());
    }

    #[test]
    fn special_register_admission_is_closed_and_schema_gated() {
        let admission = test_special_register_admission();
        let records = expand_special_register_admission(&admission).unwrap();
        assert_eq!(records.len(), 12);
        assert_eq!(
            records
                .iter()
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            [
                "clock",
                "clock64",
                "globaltimer",
                "envreg1",
                "envreg2",
                "smid",
                "nsmid",
                "gridid",
                "warpid",
                "nwarpid",
                "dynamic_smem_size",
                "total_smem_size",
            ]
        );

        let mut reordered = admission.clone();
        reordered.registers.swap(0, 1);
        assert!(expand_special_register_admission(&reordered).is_err());

        let mut wrong_count = admission.clone();
        wrong_count.product_count -= 1;
        assert!(expand_special_register_admission(&wrong_count).is_err());

        let mut executed = admission.clone();
        executed.runtime_validation = RuntimeValidation::Executed;
        assert!(expand_special_register_admission(&executed).is_err());

        let shard = |schema| OverlayShardFile {
            schema,
            family: "sreg".into(),
            intrinsics: vec![],
            register_mma_int4: None,
            register_mma_int8: None,
            register_mma_b1: None,
            sparse_mma_integer: None,
            sparse_mma_f8f6f4_f32: None,
            prmt: None,
            packed_conversion_fp8: None,
            cluster_sreg: None,
            cluster_barrier: None,
            mbarrier_extended: None,
            special_registers: Some(admission.clone()),
            debug_control: None,
            threadfence: None,
            cluster_memory: None,
            stmatrix: None,
            clc: None,
        };
        let path = Path::new("intrinsics/overlay/sreg_special.toml");
        validate_overlay_shard_schema(&shard(SPECIAL_REGISTER_SHARD_SCHEMA), path).unwrap();
        assert!(
            validate_overlay_shard_schema(&shard(SPECIAL_REGISTER_SHARD_SCHEMA - 1), path)
                .unwrap_err()
                .to_string()
                .contains("requires overlay shard schema 32")
        );
    }

    #[test]
    fn pinned_special_registers_preserve_apis_widths_and_backend_routes() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let base = load_resolution_base(&repo_root).unwrap();
        let declarations = index_imported_intrinsics(&base.imported).unwrap();
        let records = base
            .overlay
            .intrinsics
            .iter()
            .filter(|record| record.special_register.is_some())
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 12);
        assert_eq!(
            records
                .iter()
                .map(|record| record.abi_id.clone())
                .collect::<BTreeSet<_>>(),
            (283..=294)
                .map(|id| format!("i{id:04}"))
                .collect::<BTreeSet<_>>()
        );

        for record in &records {
            let source = resolve_policy_source(record).unwrap();
            let declaration = resolve_imported_declaration(record, &source, &declarations).unwrap();
            validate_special_register_policy(record, &source, declaration).unwrap();
            validate_special_register_llvm_exclusion(record, &declarations).unwrap();
        }

        for (id, section, anchor) in [
            ("clock", "10.23", "special-registers-clock-clock-hi"),
            ("clock64", "10.24", "special-registers-clock64"),
            ("globaltimer", "10.28", "special-registers-globaltimer"),
            ("envreg1", "10.27", "special-registers-envreg"),
            ("envreg2", "10.27", "special-registers-envreg"),
            ("smid", "10.8", "special-registers-smid"),
            ("nsmid", "10.9", "special-registers-nsmid"),
            ("gridid", "10.10", "special-registers-gridid"),
            ("warpid", "10.4", "special-registers-warpid"),
            ("nwarpid", "10.5", "special-registers-nwarpid"),
            (
                "dynamic_smem_size",
                "10.32",
                "special-registers-dynamic-smem-size",
            ),
            (
                "total_smem_size",
                "10.30",
                "special-registers-total-smem-size",
            ),
        ] {
            let record = records.iter().find(|record| record.id == id).unwrap();
            assert!(record.ptx_isa_section.starts_with(section));
            assert!(record.ptx_isa_url.ends_with(anchor));
        }

        let gridid = records.iter().find(|record| record.id == "gridid").unwrap();
        assert_eq!(gridid.rust_result, "u64");
        assert_eq!(gridid.dialect_results, ["i64"]);
        assert!(matches!(
            resolve_policy_source(gridid).unwrap(),
            IntrinsicSource::PtxNative { .. }
        ));
        assert!(
            gridid
                .special_register
                .as_ref()
                .unwrap()
                .llvm_exclusion
                .is_some()
        );

        let clock = records.iter().find(|record| record.id == "clock").unwrap();
        let source = resolve_policy_source(clock).unwrap();
        let declaration = resolve_imported_declaration(clock, &source, &declarations)
            .unwrap()
            .unwrap();

        let mut wrong_effects = (*clock).clone();
        wrong_effects.memory = "none".into();
        assert!(
            validate_special_register_policy(&wrong_effects, &source, Some(declaration)).is_err()
        );

        let mut wrong_contract = (*clock).clone();
        wrong_contract
            .special_register
            .as_mut()
            .unwrap()
            .output_constraint = SpecialRegisterOutputConstraint::Register64;
        assert!(
            validate_special_register_policy(&wrong_contract, &source, Some(declaration)).is_err()
        );

        let mut wrong_route = (*clock).clone();
        wrong_route.backend_lowerings[0].mechanism = BackendLoweringMechanism::InlinePtx;
        assert!(
            validate_special_register_policy(&wrong_route, &source, Some(declaration)).is_err()
        );
    }

    #[test]
    fn special_register_evidence_validates_both_backend_routes() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let policies =
            expand_special_register_admission(&test_special_register_admission()).unwrap();
        let mut evidence_files = vec![
            read_evidence_file(
                &repo_root
                    .join("intrinsics/evidence/rust-llvm-22.1.2-1cb4e383-special-registers.json"),
            )
            .unwrap(),
            read_evidence_file(
                &repo_root
                    .join("intrinsics/evidence/cuda-13.3-libnvvm-13.3.33-special-registers.json"),
            )
            .unwrap(),
        ];
        let llvm_revision = "1cb4e3833c1919c2e6fb579a23ac0e2b22587b7e";
        let indexed = index_evidence(&evidence_files, llvm_revision).unwrap();
        for policy in &policies {
            for lowering in &policy.backend_lowerings {
                let evidence = indexed
                    .get(&(lowering.evidence_profile.as_str(), policy.id.as_str()))
                    .unwrap();
                validate_evidence(policy, evidence, Some(lowering)).unwrap();
            }
        }

        let libnvvm = evidence_files
            .iter_mut()
            .find(|file| file.backend_kind == Some(IntrinsicBackend::LibNvvm))
            .unwrap();
        libnvvm
            .records
            .iter_mut()
            .find(|record| record.id == "gridid")
            .unwrap()
            .stages
            .retain(|stage| stage.stage != EvidenceStageKind::DeviceLink);
        let indexed = index_evidence(&evidence_files, llvm_revision).unwrap();
        let policy = policies
            .iter()
            .find(|policy| policy.id == "gridid")
            .unwrap();
        let lowering = policy
            .backend_lowerings
            .iter()
            .find(|lowering| lowering.backend == IntrinsicBackend::LibNvvm)
            .unwrap();
        let evidence = indexed
            .get(&(lowering.evidence_profile.as_str(), policy.id.as_str()))
            .unwrap();
        assert!(validate_evidence(policy, evidence, Some(lowering)).is_err());
    }

    #[test]
    fn threadfence_admission_is_closed_and_uses_schema_34() {
        let shard = |schema| {
            toml::from_str::<OverlayShardFile>(&format!(
                r#"
schema = {schema}
family = "sync"

[threadfence]
llvm_evidence_profile = "llvm-test"
libnvvm_evidence_profile = "libnvvm-test"
runtime_validation = "unexecuted"

[[threadfence.variant]]
abi_id = "i0298"
scope = "cta"

[[threadfence.variant]]
abi_id = "i0299"
scope = "device"

[[threadfence.variant]]
abi_id = "i0300"
scope = "system"
"#
            ))
            .unwrap()
        };
        let path = Path::new("intrinsics/overlay/threadfence.toml");

        let old = shard(THREADFENCE_SHARD_SCHEMA - 1);
        assert!(
            validate_overlay_shard_schema(&old, path)
                .unwrap_err()
                .to_string()
                .contains("requires overlay shard schema 34")
        );

        let current = shard(THREADFENCE_SHARD_SCHEMA);
        validate_overlay_shard_schema(&current, path).unwrap();
        let admission = current.threadfence.unwrap();
        let records = expand_threadfence_admission(&admission).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(
            records
                .iter()
                .map(|record| (record.abi_id.as_str(), record.id.as_str()))
                .collect::<Vec<_>>(),
            [
                ("i0298", "threadfence_block"),
                ("i0299", "threadfence"),
                ("i0300", "threadfence_system"),
            ]
        );

        let mut reordered = admission.clone();
        reordered.variants.swap(0, 1);
        assert!(expand_threadfence_admission(&reordered).is_err());

        let mut wrong_id = admission.clone();
        wrong_id.variants[0].abi_id = "i0300".into();
        assert!(expand_threadfence_admission(&wrong_id).is_err());

        let mut executed = admission.clone();
        executed.runtime_validation = RuntimeValidation::Executed;
        assert!(expand_threadfence_admission(&executed).is_err());
    }

    #[test]
    fn pinned_threadfences_match_llvm_and_reject_contract_drift() {
        let records = expand_threadfence_admission(&test_threadfence_admission()).unwrap();
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let imported: ImportedFile =
            read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
        let declarations = imported
            .intrinsics
            .iter()
            .map(|record| (record.source_record.as_str(), record))
            .collect::<BTreeMap<_, _>>();

        for record in &records {
            let declaration = declarations[record.source_record.as_deref().unwrap()];
            validate_imported_policy(record, declaration).unwrap();
            assert_eq!(
                declaration
                    .selections
                    .iter()
                    .filter(|selection| selection_matches_policy(record, selection))
                    .count(),
                1
            );
        }

        let declaration = declarations["int_nvvm_membar_cta"];
        let mut changed = records[0].clone();
        changed.memory = "none".into();
        assert!(validate_imported_policy(&changed, declaration).is_err());

        let mut changed = records[0].clone();
        changed.convergent = true;
        assert!(validate_imported_policy(&changed, declaration).is_err());

        let mut changed = records[0].clone();
        changed.backend_lowerings[0].mechanism = BackendLoweringMechanism::InlinePtx;
        assert!(validate_imported_policy(&changed, declaration).is_err());

        let mut changed = records[0].clone();
        changed.minimum_ptx = "2.0".into();
        assert!(validate_imported_policy(&changed, declaration).is_err());
    }

    #[test]
    fn pinned_cluster_sregs_preserve_helpers_and_reject_unused_w_components() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let (overlay, _) =
            read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml")).unwrap();
        let imported: ImportedFile =
            read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
        let declarations = imported
            .intrinsics
            .iter()
            .map(|record| (record.source_record.as_str(), record))
            .collect::<BTreeMap<_, _>>();
        let records = overlay
            .intrinsics
            .iter()
            .filter(|record| {
                record
                    .source_record
                    .as_deref()
                    .is_some_and(is_cluster_sreg_source)
            })
            .collect::<Vec<_>>();

        assert_eq!(records.len(), 14);
        let actual_abi_ids = records
            .iter()
            .map(|record| record.abi_id.clone())
            .collect::<BTreeSet<_>>();
        let expected_abi_ids = (263..=276)
            .map(|id| format!("i{id:04}"))
            .collect::<BTreeSet<_>>();
        assert_eq!(actual_abi_ids, expected_abi_ids);
        for record in &records {
            let declaration = declarations[record.source_record.as_deref().unwrap()];
            validate_imported_policy(record, declaration).unwrap();
            assert!(!declaration.source_record.ends_with("_w"));
        }

        let compatibility_paths = records
            .iter()
            .flat_map(|record| record.compatibility_rust_paths.iter().map(String::as_str))
            .collect::<BTreeSet<_>>();
        assert_eq!(
            compatibility_paths,
            [
                "cuda_device::cluster::cluster_ctaidX",
                "cuda_device::cluster::cluster_ctaidY",
                "cuda_device::cluster::cluster_ctaidZ",
                "cuda_device::cluster::cluster_nctaidX",
                "cuda_device::cluster::cluster_nctaidY",
                "cuda_device::cluster::cluster_nctaidZ",
                "cuda_device::cluster::__cluster_grid_dimX",
                "cuda_device::cluster::__cluster_grid_dimY",
                "cuda_device::cluster::__cluster_grid_dimZ",
                "cuda_device::cluster::__cluster_idxX",
                "cuda_device::cluster::__cluster_idxY",
                "cuda_device::cluster::__cluster_idxZ",
            ]
            .into_iter()
            .collect()
        );

        let source = resolve_policy_source(records[0]).unwrap();
        let error = validate_sreg_policy(
            records[0],
            &source,
            Some(declarations["int_nvvm_read_ptx_sreg_cluster_ctaid_w"]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("unused always-zero"));

        let mut mixed = (*records[0]).clone();
        mixed.sparse_mma = overlay
            .intrinsics
            .iter()
            .find_map(|record| record.sparse_mma.clone());
        let source = resolve_policy_source(&mixed).unwrap();
        let error = validate_sreg_policy(
            &mixed,
            &source,
            Some(declarations[mixed.source_record.as_deref().unwrap()]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("mixes another generated-family"));
    }

    #[test]
    fn compact_f8f6f4_axes_require_the_exact_canonical_matrix() {
        assert_eq!(
            expand_sparse_mma_f8f6f4_admission(&test_f8f6f4_admission())
                .unwrap()
                .len(),
            25
        );

        let mut missing = test_f8f6f4_admission();
        missing.a_elements.pop();
        assert!(expand_sparse_mma_f8f6f4_admission(&missing).is_err());

        let mut duplicate = test_f8f6f4_admission();
        duplicate.a_elements[4] = SparseMmaElement::E4m3;
        assert!(expand_sparse_mma_f8f6f4_admission(&duplicate).is_err());

        let mut extra = test_f8f6f4_admission();
        extra.b_elements.push(SparseMmaElement::S4);
        assert!(expand_sparse_mma_f8f6f4_admission(&extra).is_err());

        let mut unsorted = test_f8f6f4_admission();
        unsorted.b_elements.swap(0, 1);
        assert!(expand_sparse_mma_f8f6f4_admission(&unsorted).is_err());

        let mut wrong_count = test_f8f6f4_admission();
        wrong_count.product_count = 24;
        assert!(expand_sparse_mma_f8f6f4_admission(&wrong_count).is_err());
    }

    #[test]
    fn compact_prmt_admission_requires_every_mode_and_reserved_abi_id() {
        assert_eq!(
            expand_prmt_admission(&test_prmt_admission()).unwrap().len(),
            7
        );

        let mut missing = test_prmt_admission();
        missing.variants.pop();
        assert!(expand_prmt_admission(&missing).is_err());

        let mut duplicate = test_prmt_admission();
        duplicate.variants[6].mode = PrmtMode::Rc8;
        assert!(expand_prmt_admission(&duplicate).is_err());

        let mut wrong_abi = test_prmt_admission();
        wrong_abi.variants[0].abi_id = "i9999".into();
        assert!(expand_prmt_admission(&wrong_abi).is_err());
    }

    #[test]
    fn compact_fp8_conversion_axes_require_the_exact_closed_product() {
        let records =
            expand_packed_conversion_fp8_admission(&test_fp8_conversion_admission()).unwrap();
        assert_eq!(records.len(), 4);
        assert_eq!(records[0].id, "cvt_rn_satfinite_e4m3x2_f32");
        assert_eq!(records[1].id, "cvt_rn_satfinite_relu_e4m3x2_f32");
        assert_eq!(records[2].id, "cvt_rn_satfinite_e5m2x2_f32");
        assert_eq!(records[3].id, "cvt_rn_satfinite_relu_e5m2x2_f32");
        assert!(records.iter().all(|record| {
            record.rust_result == "u16"
                && record.dialect_results == ["i16"]
                && record.llvm_results == ["i16"]
                && record.minimum_ptx == "8.1"
                && record.minimum_sm.as_deref() == Some("sm_89")
                && record.pure
                && !record.convergent
        }));

        let mut missing_format = test_fp8_conversion_admission();
        missing_format.destination_formats.pop();
        assert!(expand_packed_conversion_fp8_admission(&missing_format).is_err());

        let mut reversed_formats = test_fp8_conversion_admission();
        reversed_formats.destination_formats.reverse();
        assert!(expand_packed_conversion_fp8_admission(&reversed_formats).is_err());

        let mut missing_saturation = test_fp8_conversion_admission();
        missing_saturation.saturations.pop();
        assert!(expand_packed_conversion_fp8_admission(&missing_saturation).is_err());

        let mut reversed_saturations = test_fp8_conversion_admission();
        reversed_saturations.saturations.reverse();
        assert!(expand_packed_conversion_fp8_admission(&reversed_saturations).is_err());

        let mut wrong_count = test_fp8_conversion_admission();
        wrong_count.product_count = 3;
        assert!(expand_packed_conversion_fp8_admission(&wrong_count).is_err());

        let mut executed = test_fp8_conversion_admission();
        executed.runtime_validation = RuntimeValidation::Executed;
        assert!(expand_packed_conversion_fp8_admission(&executed).is_err());
    }

    #[test]
    fn compact_debug_control_admission_is_closed() {
        let records = expand_debug_control_admission(&test_debug_control_admission()).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].id, "trap");
        assert_eq!(records[1].id, "breakpoint");
        assert_eq!(records[2].id, "pmevent");
        for record in &records {
            validate_ptx_native_policy(record).unwrap();
            assert_eq!(record.backend_lowerings.len(), 2);
            assert!(
                record
                    .debug_control
                    .as_ref()
                    .is_some_and(|debug| debug.runtime_validation == RuntimeValidation::Unexecuted)
            );
        }
        assert_eq!(records[0].minimum_ptx, "1.0");
        assert_eq!(records[0].minimum_sm, None);
        assert_eq!(records[1].minimum_ptx, "1.0");
        assert_eq!(records[1].minimum_sm.as_deref(), Some("sm_11"));
        assert_eq!(records[2].minimum_ptx, "1.4");
        assert_eq!(
            records[2].expected_ptx.operands,
            [OperandPattern::Immediate]
        );

        let mut pending = test_debug_control_admission();
        pending.abi_ids.clear();
        assert!(expand_debug_control_admission(&pending).is_err());

        let mut missing = test_debug_control_admission();
        missing.operations.pop();
        assert!(expand_debug_control_admission(&missing).is_err());

        let mut duplicate_operation = test_debug_control_admission();
        duplicate_operation.operations[2] = DebugControlOperation::Breakpoint;
        assert!(expand_debug_control_admission(&duplicate_operation).is_err());

        let mut duplicate_id = test_debug_control_admission();
        duplicate_id.abi_ids[2] = duplicate_id.abi_ids[1].clone();
        assert!(expand_debug_control_admission(&duplicate_id).is_err());

        let mut malformed_id = test_debug_control_admission();
        malformed_id.abi_ids[0] = "debug1".into();
        assert!(expand_debug_control_admission(&malformed_id).is_err());

        let mut executed = test_debug_control_admission();
        executed.runtime_validation = RuntimeValidation::Executed;
        assert!(expand_debug_control_admission(&executed).is_err());

        let mut wrong_source = records[0].clone();
        wrong_source.source = Some(IntrinsicSource::LlvmImported {
            source_record: "invented".into(),
        });
        assert!(validate_ptx_native_policy(&wrong_source).is_err());

        let mut wrong_adapter = records[2].clone();
        wrong_adapter.debug_control.as_mut().unwrap().adapter = DebugControlAdapter::Direct;
        assert!(validate_ptx_native_policy(&wrong_adapter).is_err());

        let mut wrong_immediate = records[2].clone();
        wrong_immediate.expected_ptx.operands = vec![OperandPattern::Register];
        assert!(validate_ptx_native_policy(&wrong_immediate).is_err());

        let mut wrong_floor = records[1].clone();
        wrong_floor.backend_lowerings[0].minimum_sm = Some("sm_75".into());
        assert!(validate_ptx_native_policy(&wrong_floor).is_err());
    }

    #[test]
    fn compact_clc_admission_matches_llvm_and_fails_closed() {
        let records = expand_clc_admission(&test_clc_admission()).unwrap();
        assert_eq!(records.len(), 6);
        assert_eq!(
            records
                .iter()
                .map(|record| (record.abi_id.as_str(), record.id.as_str()))
                .collect::<Vec<_>>(),
            [
                ("i0322", "clc_try_cancel"),
                ("i0323", "clc_try_cancel_multicast"),
                ("i0324", "clc_query_is_canceled"),
                ("i0325", "clc_query_get_first_ctaid_x"),
                ("i0326", "clc_query_get_first_ctaid_y"),
                ("i0327", "clc_query_get_first_ctaid_z"),
            ]
        );

        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let imported: ImportedFile =
            read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
        let declarations = imported
            .intrinsics
            .iter()
            .map(|record| (record.source_record.as_str(), record))
            .collect::<BTreeMap<_, _>>();
        for record in &records {
            let declaration = declarations[record.source_record.as_deref().unwrap()];
            validate_imported_policy(record, declaration).unwrap();
        }

        assert_eq!(
            parse_hardware_target(&records[1]).unwrap(),
            CatalogHardwareTarget::AnyOf {
                alternatives: vec![
                    CatalogHardwareAlternative::ExactArchitecture { sm: 100 },
                    CatalogHardwareAlternative::ExactArchitecture { sm: 101 },
                    CatalogHardwareAlternative::ExactArchitecture { sm: 110 },
                    CatalogHardwareAlternative::ExactArchitecture { sm: 120 },
                ],
            }
        );

        let mut missing = test_clc_admission();
        missing.variants.pop();
        assert!(expand_clc_admission(&missing).is_err());

        let mut reordered = test_clc_admission();
        reordered.variants.swap(0, 1);
        assert!(expand_clc_admission(&reordered).is_err());

        let mut wrong_abi = test_clc_admission();
        wrong_abi.variants[0].abi_id = "i9999".into();
        assert!(expand_clc_admission(&wrong_abi).is_err());

        let mut executed = test_clc_admission();
        executed.runtime_validation = RuntimeValidation::Executed;
        assert!(expand_clc_admission(&executed).is_err());

        let declaration = declarations[records[2].source_record.as_deref().unwrap()];
        let mut wrong_adapter = records[2].clone();
        wrong_adapter.clc.as_mut().unwrap().adapter = ClcAdapter::PairU64ToI128U32;
        assert!(validate_imported_policy(&wrong_adapter, declaration).is_err());

        let mut unsorted_targets = records[1].clone();
        unsorted_targets.targets = "sm_120a|sm_100a".into();
        assert!(parse_hardware_target(&unsorted_targets).is_err());

        let mut duplicate_targets = records[1].clone();
        duplicate_targets.targets = "sm_100a|sm_100a".into();
        assert!(parse_hardware_target(&duplicate_targets).is_err());
    }

    #[test]
    fn clc_compact_schema_is_reserved_for_aggregation() {
        let shard = |schema| OverlayShardFile {
            schema,
            family: "clc".into(),
            intrinsics: vec![],
            register_mma_int4: None,
            register_mma_int8: None,
            register_mma_b1: None,
            sparse_mma_integer: None,
            sparse_mma_f8f6f4_f32: None,
            prmt: None,
            packed_conversion_fp8: None,
            cluster_sreg: None,
            cluster_barrier: None,
            mbarrier_extended: None,
            special_registers: None,
            debug_control: None,
            threadfence: None,
            cluster_memory: None,
            stmatrix: None,
            clc: Some(test_clc_admission()),
        };
        let path = Path::new("intrinsics/overlay/clc.toml");
        validate_overlay_shard_schema_with_max(&shard(CLC_SHARD_SCHEMA), path, CLC_SHARD_SCHEMA)
            .unwrap();
        assert!(
            validate_overlay_shard_schema_with_max(
                &shard(CLC_SHARD_SCHEMA - 1),
                path,
                CLC_SHARD_SCHEMA,
            )
            .unwrap_err()
            .to_string()
            .contains("requires overlay shard schema 40")
        );
    }

    #[test]
    fn debug_control_compact_schema_is_reserved_for_aggregation() {
        let shard = |schema| OverlayShardFile {
            schema,
            family: "debug_control".into(),
            intrinsics: vec![],
            register_mma_int4: None,
            register_mma_int8: None,
            register_mma_b1: None,
            sparse_mma_integer: None,
            sparse_mma_f8f6f4_f32: None,
            prmt: None,
            packed_conversion_fp8: None,
            cluster_sreg: None,
            cluster_barrier: None,
            mbarrier_extended: None,
            special_registers: None,
            debug_control: Some(test_debug_control_admission()),
            threadfence: None,
            cluster_memory: None,
            stmatrix: None,
            clc: None,
        };
        let path = Path::new("intrinsics/overlay/debug_control.toml");
        validate_overlay_shard_schema_with_max(&shard(33), path, 33).unwrap();
        assert!(validate_overlay_shard_schema_with_max(&shard(33), path, 32).is_err());
        let error = validate_overlay_shard_schema_with_max(&shard(32), path, 33).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires overlay shard schema 33")
        );
    }

    #[test]
    fn cluster_memory_compact_schema_is_reserved_and_fail_closed() {
        let shard = |schema| OverlayShardFile {
            schema,
            family: "cluster_memory".into(),
            intrinsics: vec![],
            register_mma_int4: None,
            register_mma_int8: None,
            register_mma_b1: None,
            sparse_mma_integer: None,
            sparse_mma_f8f6f4_f32: None,
            prmt: None,
            packed_conversion_fp8: None,
            cluster_sreg: None,
            cluster_barrier: None,
            mbarrier_extended: None,
            special_registers: None,
            debug_control: None,
            threadfence: None,
            cluster_memory: Some(test_cluster_memory_admission()),
            stmatrix: None,
            clc: None,
        };
        let path = Path::new("intrinsics/overlay/cluster_memory.toml");
        validate_overlay_shard_schema_with_max(
            &shard(CLUSTER_MEMORY_SHARD_SCHEMA),
            path,
            CLUSTER_MEMORY_SHARD_SCHEMA,
        )
        .unwrap();
        assert!(
            validate_overlay_shard_schema_with_max(
                &shard(CLUSTER_MEMORY_SHARD_SCHEMA),
                path,
                CLUSTER_MEMORY_SHARD_SCHEMA - 1,
            )
            .is_err()
        );
        let error = validate_overlay_shard_schema_with_max(
            &shard(CLUSTER_MEMORY_SHARD_SCHEMA - 1),
            path,
            CLUSTER_MEMORY_SHARD_SCHEMA,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires overlay shard schema 39")
        );
    }

    #[test]
    fn cluster_memory_admission_preserves_mapa_identity_and_ptx_native_read() {
        let admission = test_cluster_memory_admission();
        let records = expand_cluster_memory_admission(&admission).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id, "map_shared_rank");
        assert_eq!(records[1].id, "dsmem_read_u32");
        assert_eq!(records[0].abi_id, "i0320");
        assert_eq!(records[1].abi_id, "i0321");
        assert_eq!(
            cluster_memory_inline_recipe(ClusterMemoryOperation::MapSharedRank),
            ("mapa.shared::cluster.u64 $0, $1, $2;", "=l,l,r")
        );
        assert_eq!(
            cluster_memory_inline_recipe(ClusterMemoryOperation::ReadU32),
            (
                "{ .reg .u64 %mapped; mapa.shared::cluster.u64 %mapped, $1, $2; ld.shared::cluster.u32 $0, [%mapped]; }",
                "=r,l,r,~{memory}"
            )
        );

        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let imported: ImportedFile =
            read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
        let mapa = &records[0];
        let declaration = imported
            .intrinsics
            .iter()
            .find(|declaration| declaration.source_record == "int_nvvm_mapa_shared_cluster")
            .unwrap();
        assert_eq!(declaration.arguments, ["shared_ptr", "i32"]);
        assert_eq!(declaration.results, ["shared_cluster_ptr"]);
        assert_eq!(
            declaration.properties,
            ["IntrNoMem", "IntrSpeculatable", "NoCapture<arg0>"]
        );
        validate_imported_policy(mapa, declaration).unwrap();
        assert_eq!(
            declaration
                .selections
                .iter()
                .filter(|selection| selection.asm.starts_with("mapa.shared::cluster.u64"))
                .map(|selection| selection.source_record.as_str())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["mapa_shared_cluster_64", "mapa_shared_cluster_64i"])
        );

        let read = &records[1];
        validate_ptx_native_policy(read).unwrap();
        assert!(read.source_record.is_none());
        assert!(read.llvm_symbol.is_none());
        assert!(matches!(
            resolve_policy_source(read).unwrap(),
            IntrinsicSource::PtxNative { .. }
        ));
        assert_eq!(read.memory, "read");

        let mut wrong_adapter = mapa.clone();
        wrong_adapter.cluster_memory.as_mut().unwrap().adapter =
            ClusterMemoryAdapter::ConstU32PointerRankToU32;
        assert!(validate_imported_policy(&wrong_adapter, declaration).is_err());

        let mut typed_as3 = mapa.clone();
        typed_as3.llvm_results = vec!["shared_ptr".into()];
        assert!(validate_imported_policy(&typed_as3, declaration).is_err());

        let mut wrong_route = mapa.clone();
        wrong_route.backend_lowerings[0].mechanism = BackendLoweringMechanism::TypedNvvm;
        assert!(validate_imported_policy(&wrong_route, declaration).is_err());

        let mut wrong_floor = read.clone();
        wrong_floor.minimum_sm = Some("sm_80".into());
        assert!(validate_ptx_native_policy(&wrong_floor).is_err());

        let mut missing = admission.clone();
        missing.variants.pop();
        assert!(expand_cluster_memory_admission(&missing).is_err());

        let mut duplicate = admission.clone();
        duplicate.variants[1].operation = ClusterMemoryOperation::MapSharedRank;
        assert!(expand_cluster_memory_admission(&duplicate).is_err());

        let mut wrong_abi = admission.clone();
        wrong_abi.variants[0].abi_id = "i9999".into();
        assert!(expand_cluster_memory_admission(&wrong_abi).is_err());

        let mut executed = admission;
        executed.runtime_validation = RuntimeValidation::Executed;
        assert!(expand_cluster_memory_admission(&executed).is_err());
    }

    #[test]
    fn active_debug_control_sources_parse_and_prove_both_backend_routes() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let shard_path = repo_root.join("intrinsics/overlay/debug_control.toml");
        let shard: OverlayShardFile =
            toml::from_str(&fs::read_to_string(&shard_path).unwrap()).unwrap();
        validate_overlay_shard_schema_with_max(&shard, &shard_path, 33).unwrap();
        let admission = shard.debug_control.unwrap();
        assert_eq!(admission.abi_ids, ["i0295", "i0296", "i0297"]);
        let records = expand_debug_control_admission(&admission).unwrap();

        let evidence = vec![
            read_evidence_file(
                &repo_root.join("intrinsics/evidence/rust-llvm-22.1.2-1cb4e383-debug-control.json"),
            )
            .unwrap(),
            read_evidence_file(
                &repo_root.join("intrinsics/evidence/cuda-13.3-libnvvm-13.3.33-debug-control.json"),
            )
            .unwrap(),
        ];
        let indexed =
            index_evidence(&evidence, "1cb4e3833c1919c2e6fb579a23ac0e2b22587b7e").unwrap();
        for record in &records {
            let routes = resolve_backend_lowerings(record, &indexed).unwrap();
            assert_eq!(routes.len(), 2);
            assert!(routes.iter().all(|route| {
                route.mechanism == BackendLoweringMechanism::InlinePtx
                    && route.status == "validated"
            }));
        }
    }

    #[test]
    fn compact_cluster_barrier_admission_and_semantics_fail_closed() {
        let records = expand_cluster_barrier_admission(&test_cluster_barrier_admission()).unwrap();
        assert_eq!(records.len(), 6);

        let imported: ImportedFile = read_json(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("intrinsics/imported.json"),
        )
        .unwrap();
        for record in &records {
            let declaration = imported
                .intrinsics
                .iter()
                .find(|declaration| {
                    Some(declaration.source_record.as_str()) == record.source_record.as_deref()
                })
                .unwrap();
            validate_imported_policy(record, declaration).unwrap();
        }

        let declaration_for = |record: &OverlayIntrinsic| {
            imported
                .intrinsics
                .iter()
                .find(|declaration| {
                    Some(declaration.source_record.as_str()) == record.source_record.as_deref()
                })
                .unwrap()
        };
        let base = records
            .iter()
            .find(|record| {
                record
                    .cluster_barrier
                    .as_ref()
                    .is_some_and(|barrier| barrier.mode == ClusterBarrierMode::ArriveAligned)
            })
            .unwrap();

        let mut wrong_mode = base.clone();
        wrong_mode.cluster_barrier.as_mut().unwrap().mode = ClusterBarrierMode::WaitAligned;
        assert!(validate_imported_policy(&wrong_mode, declaration_for(base)).is_err());

        let mut wrong_order = base.clone();
        wrong_order.cluster_barrier.as_mut().unwrap().ordering = ClusterBarrierOrdering::Relaxed;
        assert!(validate_imported_policy(&wrong_order, declaration_for(base)).is_err());

        let mut wrong_alignment = base.clone();
        wrong_alignment.cluster_barrier.as_mut().unwrap().aligned = false;
        assert!(validate_imported_policy(&wrong_alignment, declaration_for(base)).is_err());

        let mut missing = test_cluster_barrier_admission();
        missing.variants.pop();
        assert!(expand_cluster_barrier_admission(&missing).is_err());

        let mut duplicate = test_cluster_barrier_admission();
        duplicate.variants[5].mode = ClusterBarrierMode::Arrive;
        assert!(expand_cluster_barrier_admission(&duplicate).is_err());

        let mut wrong_abi = test_cluster_barrier_admission();
        wrong_abi.variants[0].abi_id = "i9999".into();
        assert!(expand_cluster_barrier_admission(&wrong_abi).is_err());
    }

    #[test]
    fn compact_extended_mbarrier_admission_preserves_all_manual_contracts() {
        let records =
            expand_mbarrier_extended_admission(&test_mbarrier_extended_admission()).unwrap();
        assert_eq!(records.len(), 11);
        assert_eq!(
            records
                .iter()
                .map(|record| record.abi_id.as_str())
                .collect::<Vec<_>>(),
            (306..=316)
                .map(|id| format!("i{id:04}"))
                .collect::<Vec<_>>()
        );

        let imported: ImportedFile = read_json(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join("intrinsics/imported.json"),
        )
        .unwrap();
        for record in &records {
            let contract = record.mbarrier_extended.as_ref().unwrap();
            let (template, constraints) = mbarrier_extended_inline_recipe(contract.operation);
            assert!(template.ends_with(';') || template.ends_with("; }"));
            assert!(constraints.contains("~{memory}"));
            assert!(record.convergent && record.memory == "read_write");
            assert!(
                record
                    .backend_lowerings
                    .iter()
                    .all(|lowering| { lowering.mechanism == BackendLoweringMechanism::InlinePtx })
            );
            match contract.source_contract {
                MbarrierExtendedSourceContract::LlvmImported => {
                    let declaration = imported
                        .intrinsics
                        .iter()
                        .find(|declaration| {
                            Some(declaration.source_record.as_str())
                                == record.source_record.as_deref()
                        })
                        .unwrap();
                    validate_imported_policy(record, declaration).unwrap();
                }
                MbarrierExtendedSourceContract::PtxNativeRawClusterAddress => {
                    validate_ptx_native_policy(record).unwrap();
                    assert_eq!(record.rust_arguments, ["u64"]);
                    assert_eq!(record.dialect_operands, ["i64"]);
                }
            }
        }

        let base = records
            .iter()
            .find(|record| record.id == "mbarrier_arrive_expect_tx")
            .unwrap();
        let declaration = imported
            .intrinsics
            .iter()
            .find(|declaration| {
                Some(declaration.source_record.as_str()) == base.source_record.as_deref()
            })
            .unwrap();
        let mut wrong_adapter = base.clone();
        wrong_adapter.mbarrier_extended.as_mut().unwrap().adapter =
            MbarrierExtendedAdapter::PointerTokenToPredicate;
        assert!(validate_imported_policy(&wrong_adapter, declaration).is_err());
        let mut wrong_floor = base.clone();
        wrong_floor.minimum_ptx = "8.6".into();
        assert!(validate_imported_policy(&wrong_floor, declaration).is_err());
        let mut lost_clobber = base.clone();
        lost_clobber.memory = "none".into();
        assert!(validate_imported_policy(&lost_clobber, declaration).is_err());

        let remote = records
            .iter()
            .find(|record| record.id == "mbarrier_arrive_cluster")
            .unwrap();
        let incompatible = imported
            .intrinsics
            .iter()
            .find(|declaration| {
                declaration.source_record == "int_nvvm_mbarrier_arrive_scope_cluster_space_cluster"
            })
            .unwrap();
        assert_eq!(incompatible.arguments, ["shared_cluster_ptr", "i32"]);
        assert!(
            validate_mbarrier_extended_policy(
                remote,
                &IntrinsicSource::LlvmImported {
                    source_record: incompatible.source_record.clone(),
                },
                Some(incompatible),
            )
            .is_err()
        );

        let mut missing = test_mbarrier_extended_admission();
        missing.variants.pop();
        assert!(expand_mbarrier_extended_admission(&missing).is_err());
        let mut duplicate = test_mbarrier_extended_admission();
        duplicate.variants[10].operation = MbarrierExtendedOperation::ArriveExpectTxCta;
        assert!(expand_mbarrier_extended_admission(&duplicate).is_err());
        let mut wrong_abi = test_mbarrier_extended_admission();
        wrong_abi.variants[0].abi_id = "i9999".into();
        assert!(expand_mbarrier_extended_admission(&wrong_abi).is_err());
    }

    #[test]
    fn cluster_barrier_evidence_validates_both_backend_routes() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut admission = test_cluster_barrier_admission();
        admission.llvm_evidence_profile = "rust-llvm-22.1.2-1cb4e383".into();
        admission.libnvvm_evidence_profile = "cuda-13.3-libnvvm-13.3.33-cluster-barrier".into();
        let policies = expand_cluster_barrier_admission(&admission).unwrap();
        let evidence_files = vec![
            read_evidence_file(
                &repo_root
                    .join("intrinsics/evidence/rust-llvm-22.1.2-1cb4e383-cluster-barrier.json"),
            )
            .unwrap(),
            read_evidence_file(
                &repo_root
                    .join("intrinsics/evidence/cuda-13.3-libnvvm-13.3.33-cluster-barrier.json"),
            )
            .unwrap(),
        ];
        let indexed =
            index_evidence(&evidence_files, "1cb4e3833c1919c2e6fb579a23ac0e2b22587b7e").unwrap();

        for policy in &policies {
            for lowering in &policy.backend_lowerings {
                let evidence = indexed
                    .get(&(lowering.evidence_profile.as_str(), policy.id.as_str()))
                    .unwrap();
                validate_evidence(policy, evidence, Some(lowering)).unwrap();
            }
        }

        let mut missing_typed_failure = evidence_files.clone();
        let libnvvm = missing_typed_failure
            .iter_mut()
            .find(|file| file.backend_kind == Some(IntrinsicBackend::LibNvvm))
            .unwrap();
        for record in &mut libnvvm.records {
            record.stages.retain(|stage| {
                stage.mechanism != Some(BackendLoweringMechanism::TypedNvvm)
                    || stage.stage != EvidenceStageKind::DeviceLink
            });
        }
        let indexed = index_evidence(
            &missing_typed_failure,
            "1cb4e3833c1919c2e6fb579a23ac0e2b22587b7e",
        )
        .unwrap();
        let policy = &policies[0];
        let lowering = policy
            .backend_lowerings
            .iter()
            .find(|lowering| lowering.backend == IntrinsicBackend::LibNvvm)
            .unwrap();
        let evidence = indexed
            .get(&(lowering.evidence_profile.as_str(), policy.id.as_str()))
            .unwrap();
        assert!(validate_evidence(policy, evidence, Some(lowering)).is_err());
    }

    #[test]
    fn pinned_register_mma_records_match_the_closed_recipes_and_fail_closed() {
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
        let records: Vec<_> = overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "register_mma")
            .collect();
        assert_eq!(records.len(), 58);

        let integer_records: Vec<_> = records
            .iter()
            .copied()
            .filter(|record| {
                record.register_mma.as_ref().is_some_and(|mma| {
                    mma.operation == RegisterMmaOperation::Multiply
                        && mma.accumulator == RegisterMmaAccumulator::S32
                })
            })
            .collect();
        assert_eq!(integer_records.len(), 48);
        let binary_records = records
            .iter()
            .copied()
            .filter(|record| {
                record
                    .register_mma
                    .as_ref()
                    .is_some_and(|mma| mma.operation != RegisterMmaOperation::Multiply)
            })
            .collect::<Vec<_>>();
        assert_eq!(binary_records.len(), 6);
        let int8_records = integer_records
            .iter()
            .copied()
            .filter(|record| {
                let mma = record.register_mma.as_ref().unwrap();
                matches!(
                    mma.a_element,
                    RegisterMmaElement::S8 | RegisterMmaElement::U8
                ) && matches!(
                    mma.b_element,
                    RegisterMmaElement::S8 | RegisterMmaElement::U8
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(int8_records.len(), 24);
        let int4_records = integer_records
            .iter()
            .copied()
            .filter(|record| {
                let mma = record.register_mma.as_ref().unwrap();
                matches!(
                    mma.a_element,
                    RegisterMmaElement::S4 | RegisterMmaElement::U4
                ) && matches!(
                    mma.b_element,
                    RegisterMmaElement::S4 | RegisterMmaElement::U4
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(int4_records.len(), 24);
        let actual_variants = integer_records
            .iter()
            .map(|record| {
                let mma = record.register_mma.as_ref().unwrap();
                (mma.shape, mma.a_element, mma.b_element, mma.overflow)
            })
            .collect::<BTreeSet<_>>();
        let expected_int8_variants = [
            RegisterMmaShape::M8n8k16,
            RegisterMmaShape::M16n8k16,
            RegisterMmaShape::M16n8k32,
        ]
        .into_iter()
        .flat_map(|shape| {
            [RegisterMmaElement::S8, RegisterMmaElement::U8]
                .into_iter()
                .flat_map(move |a_element| {
                    [RegisterMmaElement::S8, RegisterMmaElement::U8]
                        .into_iter()
                        .flat_map(move |b_element| {
                            [
                                RegisterMmaOverflow::Wrapping,
                                RegisterMmaOverflow::Satfinite,
                            ]
                            .into_iter()
                            .map(move |overflow| (shape, a_element, b_element, overflow))
                        })
                })
        })
        .collect::<BTreeSet<_>>();
        let expected_int4_variants = [
            RegisterMmaShape::M8n8k32,
            RegisterMmaShape::M16n8k32,
            RegisterMmaShape::M16n8k64,
        ]
        .into_iter()
        .flat_map(|shape| {
            [RegisterMmaElement::S4, RegisterMmaElement::U4]
                .into_iter()
                .flat_map(move |a_element| {
                    [RegisterMmaElement::S4, RegisterMmaElement::U4]
                        .into_iter()
                        .flat_map(move |b_element| {
                            [
                                RegisterMmaOverflow::Wrapping,
                                RegisterMmaOverflow::Satfinite,
                            ]
                            .into_iter()
                            .map(move |overflow| (shape, a_element, b_element, overflow))
                        })
                })
        })
        .collect::<BTreeSet<_>>();
        let expected_variants = expected_int8_variants
            .union(&expected_int4_variants)
            .copied()
            .collect::<BTreeSet<_>>();
        assert_eq!(actual_variants, expected_variants);
        assert_eq!(
            integer_records
                .iter()
                .filter(|record| {
                    record.register_mma.as_ref().unwrap().compatibility_source
                        == RegisterMmaCompatibilitySource::GeneratedStub
                })
                .count(),
            47
        );

        let actual_binary_variants = binary_records
            .iter()
            .map(|record| {
                let mma = record.register_mma.as_ref().unwrap();
                (mma.shape, mma.operation)
            })
            .collect::<BTreeSet<_>>();
        let expected_binary_variants = [
            RegisterMmaShape::M8n8k128,
            RegisterMmaShape::M16n8k128,
            RegisterMmaShape::M16n8k256,
        ]
        .into_iter()
        .flat_map(|shape| {
            [RegisterMmaOperation::XorPopc, RegisterMmaOperation::AndPopc]
                .into_iter()
                .map(move |operation| (shape, operation))
        })
        .collect::<BTreeSet<_>>();
        assert_eq!(actual_binary_variants, expected_binary_variants);
        assert!(binary_records.iter().all(|record| {
            let mma = record.register_mma.as_ref().unwrap();
            mma.accumulator == RegisterMmaAccumulator::S32
                && mma.a_element == RegisterMmaElement::B1
                && mma.b_element == RegisterMmaElement::B1
                && mma.overflow == RegisterMmaOverflow::Wrapping
                && mma.compatibility_source == RegisterMmaCompatibilitySource::GeneratedStub
                && record.expected_ptx.modifiers.ends_with(&[
                    match mma.operation {
                        RegisterMmaOperation::XorPopc => "xor".into(),
                        RegisterMmaOperation::AndPopc => "and".into(),
                        RegisterMmaOperation::Multiply => unreachable!(),
                    },
                    "popc".into(),
                ])
        }));

        for record in &binary_records {
            let mma = record.register_mma.as_ref().unwrap();
            let (arguments, result, adapter) = match mma.shape {
                RegisterMmaShape::M8n8k128 => (
                    &["[i32; 2]", "u32", "u32"] as &[_],
                    "[i32; 2]",
                    RegisterMmaAdapter::C2I32A1U32B1U32ToD2I32,
                ),
                RegisterMmaShape::M16n8k128 => (
                    &["[i32; 4]", "[u32; 2]", "u32"] as &[_],
                    "[i32; 4]",
                    RegisterMmaAdapter::C4I32A2U32B1U32ToD4I32,
                ),
                RegisterMmaShape::M16n8k256 => (
                    &["[i32; 4]", "[u32; 4]", "[u32; 2]"] as &[_],
                    "[i32; 4]",
                    RegisterMmaAdapter::C4I32A4U32B2U32ToD4I32,
                ),
                _ => unreachable!(),
            };
            assert_eq!(record.rust_arguments, arguments);
            assert_eq!(record.rust_result, result);
            assert_eq!(mma.adapter, adapter);
            let expected_floor = match (mma.shape, mma.operation) {
                (RegisterMmaShape::M8n8k128, RegisterMmaOperation::XorPopc) => ("7.0", "sm_75"),
                (_, RegisterMmaOperation::XorPopc) => ("7.0", "sm_80"),
                (_, RegisterMmaOperation::AndPopc) => ("7.1", "sm_80"),
                _ => unreachable!(),
            };
            assert_eq!(record.minimum_ptx, expected_floor.0);
            assert_eq!(record.minimum_sm.as_deref(), Some(expected_floor.1));
        }

        for record in integer_records.iter().filter(|record| {
            matches!(
                record.register_mma.as_ref().unwrap().shape,
                RegisterMmaShape::M8n8k16 | RegisterMmaShape::M8n8k32
            )
        }) {
            assert_eq!(record.rust_arguments, ["[i32; 2]", "u32", "u32"]);
            assert_eq!(record.rust_result, "[i32; 2]");
            assert_eq!(record.minimum_ptx, "6.5");
            assert_eq!(record.minimum_sm.as_deref(), Some("sm_75"));
            assert_eq!(
                record.register_mma.as_ref().unwrap().adapter,
                RegisterMmaAdapter::C2I32A1U32B1U32ToD2I32
            );
        }

        for record in int4_records.iter().filter(|record| {
            record.register_mma.as_ref().unwrap().shape == RegisterMmaShape::M16n8k32
        }) {
            assert_eq!(record.rust_arguments, ["[i32; 4]", "[u32; 2]", "u32"]);
            assert_eq!(record.rust_result, "[i32; 4]");
            assert_eq!(record.minimum_ptx, "7.0");
            assert_eq!(record.minimum_sm.as_deref(), Some("sm_80"));
            assert_eq!(
                record.register_mma.as_ref().unwrap().adapter,
                RegisterMmaAdapter::C4I32A2U32B1U32ToD4I32
            );
        }

        for record in int4_records.iter().filter(|record| {
            record.register_mma.as_ref().unwrap().shape == RegisterMmaShape::M16n8k64
        }) {
            assert_eq!(record.rust_arguments, ["[i32; 4]", "[u32; 4]", "[u32; 2]"]);
            assert_eq!(record.rust_result, "[i32; 4]");
            assert_eq!(record.minimum_ptx, "7.0");
            assert_eq!(record.minimum_sm.as_deref(), Some("sm_80"));
            assert_eq!(
                record.register_mma.as_ref().unwrap().adapter,
                RegisterMmaAdapter::C4I32A4U32B2U32ToD4I32
            );
        }

        let actual_int4_abi_ids = int4_records
            .iter()
            .map(|record| record.abi_id.as_str())
            .collect::<BTreeSet<_>>();
        let expected_int4_abi_ids = (133..=156)
            .map(|id| format!("i{id:04}"))
            .collect::<BTreeSet<_>>();
        assert_eq!(
            actual_int4_abi_ids,
            expected_int4_abi_ids
                .iter()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
        );

        let int8_k32 = int8_records
            .iter()
            .find(|record| {
                record.register_mma.as_ref().unwrap().shape == RegisterMmaShape::M16n8k32
            })
            .unwrap();
        let int4_k32 = int4_records
            .iter()
            .find(|record| {
                record.register_mma.as_ref().unwrap().shape == RegisterMmaShape::M16n8k32
            })
            .unwrap();
        assert_eq!(
            int8_k32.register_mma.as_ref().unwrap().adapter,
            RegisterMmaAdapter::C4I32A4U32B2U32ToD4I32
        );
        assert_eq!(
            int4_k32.register_mma.as_ref().unwrap().adapter,
            RegisterMmaAdapter::C4I32A2U32B1U32ToD4I32
        );

        for policy in &records {
            let declaration = declarations[policy.source_record.as_deref().unwrap()];
            assert!(declaration.selections.is_empty());
            validate_imported_policy(policy, declaration).unwrap();
        }

        let valid = records[0];
        let declaration = declarations[valid.source_record.as_deref().unwrap()];

        let mut non_convergent = valid.clone();
        non_convergent.convergent = false;
        assert!(
            validate_imported_policy(&non_convergent, declaration)
                .unwrap_err()
                .to_string()
                .contains("effects")
        );

        let mut typed_route = valid.clone();
        typed_route.backend_lowerings[0].mechanism = BackendLoweringMechanism::TypedNvvm;
        assert!(validate_imported_policy(&typed_route, declaration).is_err());

        let mut crossed_variant = valid.clone();
        crossed_variant.register_mma.as_mut().unwrap().a_element = RegisterMmaElement::F16;
        assert!(validate_imported_policy(&crossed_variant, declaration).is_err());

        let generated = int8_records
            .iter()
            .copied()
            .find(|record| record.id == "mma_m16n8k16_s32_s8_u8_satfinite")
            .unwrap();
        let generated_declaration = declarations[generated.source_record.as_deref().unwrap()];

        let mut wrong_stub_owner = generated.clone();
        wrong_stub_owner
            .register_mma
            .as_mut()
            .unwrap()
            .compatibility_source = RegisterMmaCompatibilitySource::ExistingStub;
        assert!(validate_imported_policy(&wrong_stub_owner, generated_declaration).is_err());

        let mut wrong_b_element = generated.clone();
        wrong_b_element.register_mma.as_mut().unwrap().b_element = RegisterMmaElement::S8;
        assert!(validate_imported_policy(&wrong_b_element, generated_declaration).is_err());

        let mut wrong_overflow = generated.clone();
        wrong_overflow.register_mma.as_mut().unwrap().overflow = RegisterMmaOverflow::Wrapping;
        assert!(validate_imported_policy(&wrong_overflow, generated_declaration).is_err());

        let mut wrong_shape = generated.clone();
        wrong_shape.register_mma.as_mut().unwrap().shape = RegisterMmaShape::M16n8k32;
        assert!(validate_imported_policy(&wrong_shape, generated_declaration).is_err());

        let mut wrong_adapter = generated.clone();
        wrong_adapter.register_mma.as_mut().unwrap().adapter =
            RegisterMmaAdapter::C4I32A4U32B2U32ToD4I32;
        assert!(validate_imported_policy(&wrong_adapter, generated_declaration).is_err());

        let binary = binary_records
            .iter()
            .copied()
            .find(|record| record.id == "mma_m8n8k128_s32_b1_xor_popc")
            .unwrap();
        let binary_declaration = declarations[binary.source_record.as_deref().unwrap()];

        let mut wrong_binary_operation = binary.clone();
        wrong_binary_operation
            .register_mma
            .as_mut()
            .unwrap()
            .operation = RegisterMmaOperation::AndPopc;
        assert!(validate_imported_policy(&wrong_binary_operation, binary_declaration).is_err());

        let mut wrong_binary_floor = binary.clone();
        wrong_binary_floor.minimum_sm = Some("sm_80".into());
        assert!(validate_imported_policy(&wrong_binary_floor, binary_declaration).is_err());

        let mut wrong_binary_element = binary.clone();
        wrong_binary_element
            .register_mma
            .as_mut()
            .unwrap()
            .a_element = RegisterMmaElement::U4;
        assert!(validate_imported_policy(&wrong_binary_element, binary_declaration).is_err());
    }

    #[test]
    fn pinned_sparse_mma_records_close_shape_specific_selectors_and_ranges() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let (mut overlay, _) =
            read_overlay(&repo_root, &repo_root.join("intrinsics/overlay.toml")).unwrap();
        bind_pinned_abi_ids(&repo_root, &mut overlay);
        let imported: ImportedFile =
            read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
        let declarations: BTreeMap<_, _> = imported
            .intrinsics
            .iter()
            .map(|record| (record.source_record.as_str(), record))
            .collect();
        let records = overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "sparse_mma")
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 89);
        assert_eq!(
            records
                .iter()
                .map(|record| record.abi_id.as_str())
                .collect::<BTreeSet<_>>(),
            (163..=251)
                .map(|id| format!("i{id:04}"))
                .collect::<BTreeSet<_>>()
                .iter()
                .map(String::as_str)
                .collect()
        );

        let mut derived_ids = BTreeSet::new();
        let mut derived_operation_keys = BTreeSet::new();
        let mut derived_source_records = BTreeSet::new();
        let mut derived_llvm_symbols = BTreeSet::new();
        for record in &records {
            let identity = &sparse_mma_recipe(record.sparse_mma.as_ref().unwrap())
                .unwrap()
                .identity;
            assert_eq!(record.id, identity.id);
            assert_eq!(record.operation_key, identity.operation_key);
            assert_eq!(
                record.source_record.as_deref(),
                Some(identity.source_record.as_str())
            );
            assert_eq!(
                record.llvm_symbol.as_deref(),
                Some(identity.llvm_symbol.as_str())
            );
            assert_eq!(record.expected_ptx.modifiers, identity.ptx_modifiers);
            assert!(derived_ids.insert(identity.id.clone()));
            assert!(derived_operation_keys.insert(identity.operation_key.clone()));
            assert!(derived_source_records.insert(identity.source_record.clone()));
            assert!(derived_llvm_symbols.insert(identity.llvm_symbol.clone()));
        }
        assert_eq!(derived_ids.len(), 89);
        assert_eq!(derived_operation_keys.len(), 89);
        assert_eq!(derived_source_records.len(), 89);
        assert_eq!(derived_llvm_symbols.len(), 89);

        let integer_records = records
            .iter()
            .copied()
            .filter(|record| {
                record.sparse_mma.as_ref().unwrap().accumulator == SparseMmaAccumulator::S32
            })
            .collect::<Vec<_>>();
        assert_eq!(integer_records.len(), 64);
        let f32_records = records
            .iter()
            .copied()
            .filter(|record| {
                record.sparse_mma.as_ref().unwrap().accumulator == SparseMmaAccumulator::F32
            })
            .collect::<Vec<_>>();
        assert_eq!(f32_records.len(), 25);

        let variants = integer_records
            .iter()
            .map(|record| {
                let mma = record.sparse_mma.as_ref().unwrap();
                let carrier =
                    sparse_mma_carrier_recipe(mma.shape, mma.a_element, mma.b_element).unwrap();
                assert_eq!(mma.accumulator, SparseMmaAccumulator::S32);
                assert_eq!(mma.selector, carrier.selector);
                assert_eq!(mma.adapter, carrier.adapter);
                assert_eq!(mma.llvm_adapter, carrier.llvm_adapter);
                assert_eq!(record.rust_arguments, carrier.rust_arguments());
                assert_eq!(record.dialect_operands, carrier.dialect_operands());
                assert_eq!(record.llvm_arguments, carrier.llvm_arguments());
                assert_eq!(
                    record.expected_ptx.operands,
                    carrier.expected_ptx_operands()
                );
                assert_eq!(record.minimum_ptx, sparse_mma_minimum_ptx(mma));
                assert_eq!(record.minimum_sm.as_deref(), Some("sm_80"));
                assert_eq!(
                    record.expected_ptx.operands.last(),
                    Some(&OperandPattern::Immediate)
                );
                assert_eq!(
                    record.expected_ptx.modifiers.first().map(String::as_str),
                    Some(match mma.metadata {
                        SparseMmaMetadata::Standard => "sp",
                        SparseMmaMetadata::Ordered => "sp::ordered_metadata",
                    })
                );
                (
                    mma.shape,
                    mma.a_element,
                    mma.b_element,
                    mma.overflow,
                    mma.metadata,
                )
            })
            .collect::<BTreeSet<_>>();
        let mut expected_variants = BTreeSet::new();
        for shape in [SparseMmaShape::M16n8k32, SparseMmaShape::M16n8k64] {
            let metadata = match shape {
                SparseMmaShape::M16n8k32 => [
                    Some(SparseMmaMetadata::Standard),
                    Some(SparseMmaMetadata::Ordered),
                ],
                SparseMmaShape::M16n8k64 => [
                    Some(SparseMmaMetadata::Standard),
                    Some(SparseMmaMetadata::Ordered),
                ],
                SparseMmaShape::M16n8k128 => [None, None],
            };
            for a_element in [SparseMmaElement::S8, SparseMmaElement::U8] {
                for b_element in [SparseMmaElement::S8, SparseMmaElement::U8] {
                    for overflow in [SparseMmaOverflow::Wrapping, SparseMmaOverflow::Satfinite] {
                        for metadata in metadata.into_iter().flatten() {
                            expected_variants
                                .insert((shape, a_element, b_element, overflow, metadata));
                        }
                    }
                }
            }
        }
        for shape in [SparseMmaShape::M16n8k64, SparseMmaShape::M16n8k128] {
            for a_element in [SparseMmaElement::S4, SparseMmaElement::U4] {
                for b_element in [SparseMmaElement::S4, SparseMmaElement::U4] {
                    for overflow in [SparseMmaOverflow::Wrapping, SparseMmaOverflow::Satfinite] {
                        for metadata in [SparseMmaMetadata::Standard, SparseMmaMetadata::Ordered] {
                            expected_variants
                                .insert((shape, a_element, b_element, overflow, metadata));
                        }
                    }
                }
            }
        }
        assert_eq!(variants, expected_variants);

        let f8f6f4_formats = [
            SparseMmaElement::E2m1,
            SparseMmaElement::E2m3,
            SparseMmaElement::E3m2,
            SparseMmaElement::E4m3,
            SparseMmaElement::E5m2,
        ];
        assert_eq!(
            f32_records
                .iter()
                .map(|record| {
                    let mma = record.sparse_mma.as_ref().unwrap();
                    (mma.a_element, mma.b_element)
                })
                .collect::<BTreeSet<_>>(),
            f8f6f4_formats
                .into_iter()
                .flat_map(|a| f8f6f4_formats.into_iter().map(move |b| (a, b)))
                .collect()
        );
        assert_eq!(
            f32_records
                .iter()
                .map(|record| record.abi_id.as_str())
                .collect::<BTreeSet<_>>(),
            (227..=251)
                .map(|id| format!("i{id:04}"))
                .collect::<BTreeSet<_>>()
                .iter()
                .map(String::as_str)
                .collect()
        );
        for record in &f32_records {
            let mma = record.sparse_mma.as_ref().unwrap();
            assert_eq!(mma.shape, SparseMmaShape::M16n8k64);
            assert_eq!(mma.accumulator, SparseMmaAccumulator::F32);
            assert_eq!(mma.overflow, SparseMmaOverflow::NotApplicable);
            assert_eq!(mma.metadata, SparseMmaMetadata::Ordered);
            assert_eq!(mma.selector, SparseMmaSelector::ImmediateZero);
            assert_eq!(
                mma.adapter,
                SparseMmaAdapter::C4F32A4U32B4U32MetadataU32SelectorU32ToD4F32
            );
            assert_eq!(
                mma.llvm_adapter,
                SparseMmaLlvmAdapter::A4I32B4I32C4F32MetadataI32SelectorI32ToD4F32
            );
            assert_eq!(
                record.rust_arguments,
                ["[f32; 4]", "[u32; 4]", "[u32; 4]", "u32", "u32"]
            );
            assert_eq!(record.rust_result, "[f32; 4]");
            assert_eq!(
                record.dialect_operands,
                [
                    "f32", "f32", "f32", "f32", "u32", "u32", "u32", "u32", "u32", "u32", "u32",
                    "u32", "u32", "u32"
                ]
            );
            assert_eq!(record.dialect_results, ["f32", "f32", "f32", "f32"]);
            assert_eq!(
                record.llvm_arguments,
                [
                    "i32", "i32", "i32", "i32", "i32", "i32", "i32", "i32", "f32", "f32", "f32",
                    "f32", "i32", "i32"
                ]
            );
            assert_eq!(record.llvm_results, ["f32", "f32", "f32", "f32"]);
            assert_eq!(record.minimum_ptx, "8.7");
            assert_eq!(record.minimum_sm, None);
            assert_eq!(record.targets, "sm_120a");
            assert_eq!(record.backend_lowerings.len(), 2);
            assert!(record.backend_lowerings.iter().all(|lowering| {
                lowering.mechanism == BackendLoweringMechanism::InlinePtx
                    && lowering.minimum_ptx.as_deref() == Some("8.7")
                    && lowering.minimum_sm.is_none()
            }));
            assert_eq!(
                record.expected_ptx.operands,
                [
                    OperandPattern::RegisterList { length: 4 },
                    OperandPattern::RegisterList { length: 4 },
                    OperandPattern::RegisterList { length: 4 },
                    OperandPattern::RegisterList { length: 4 },
                    OperandPattern::Register,
                    OperandPattern::Immediate,
                ]
            );
        }

        for policy in &records {
            let declaration = declarations[policy.source_record.as_deref().unwrap()];
            validate_imported_policy(policy, declaration).unwrap();
        }

        for (id, range_prefix, wrong_range) in [
            ("mma_sp_m16n8k32_s32_s8", "Range<arg9", "Range<arg9,0,3>"),
            ("mma_sp_m16n8k64_s32_s8", "Range<arg13", "Range<arg13,0,2>"),
            (
                "mma_sp_ordered_metadata_m16n8k64_s32_s4",
                "Range<arg9",
                "Range<arg9,0,1>",
            ),
            ("mma_sp_m16n8k64_s32_s4", "Range<arg9", "Range<arg9,0,1>"),
            (
                "mma_sp_ordered_metadata_m16n8k128_s32_s4",
                "Range<arg13",
                "Range<arg13,0,2>",
            ),
            ("mma_sp_m16n8k128_s32_s4", "Range<arg13", "Range<arg13,0,2>"),
            (
                "mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m1_e2m1_f32",
                "Range<arg13",
                "Range<arg13,0,2>",
            ),
        ] {
            let valid = records
                .iter()
                .copied()
                .find(|record| record.id == id)
                .unwrap();
            let declaration = declarations[valid.source_record.as_deref().unwrap()];

            let mut runtime_selector = valid.clone();
            *runtime_selector.expected_ptx.operands.last_mut().unwrap() = OperandPattern::Register;
            assert!(
                validate_imported_policy(&runtime_selector, declaration)
                    .unwrap_err()
                    .to_string()
                    .contains("expected PTX")
            );

            let mut wrong_declaration = declaration.clone();
            *wrong_declaration
                .properties
                .iter_mut()
                .find(|property| property.starts_with(range_prefix))
                .unwrap() = wrong_range.into();
            assert!(
                validate_imported_policy(valid, &wrong_declaration)
                    .unwrap_err()
                    .to_string()
                    .contains("immediate range")
            );
        }

        let k64 = records
            .iter()
            .copied()
            .find(|record| record.id == "mma_sp_m16n8k64_s32_s8")
            .unwrap();
        let k64_declaration = declarations[k64.source_record.as_deref().unwrap()];
        assert_eq!(k64.minimum_ptx, "7.1");
        let ordered_k64 = records
            .iter()
            .copied()
            .find(|record| record.id == "mma_sp_ordered_metadata_m16n8k64_s32_s8")
            .unwrap();
        assert_eq!(ordered_k64.minimum_ptx, "8.5");

        let f8f6f4 = f32_records[0];
        let f8f6f4_declaration = declarations[f8f6f4.source_record.as_deref().unwrap()];
        let mut widened_family = f8f6f4.clone();
        widened_family.targets = "sm_120f".into();
        assert!(validate_imported_policy(&widened_family, f8f6f4_declaration).is_err());
        let mut widened_architecture = f8f6f4.clone();
        widened_architecture.targets = "sm_121a".into();
        assert!(validate_imported_policy(&widened_architecture, f8f6f4_declaration).is_err());

        let ordered_k64_int4 = records
            .iter()
            .copied()
            .find(|record| record.id == "mma_sp_ordered_metadata_m16n8k64_s32_s4")
            .unwrap();
        let ordered_k64_int4_declaration =
            declarations[ordered_k64_int4.source_record.as_deref().unwrap()];
        assert_eq!(ordered_k64_int4.minimum_ptx, "8.5");
        assert_eq!(ordered_k64_int4.rust_arguments[1], "[u32; 2]");
        assert_eq!(ordered_k64_int4.rust_arguments[2], "[u32; 2]");
        assert_eq!(ordered_k64_int4.llvm_arguments.len(), 10);
        assert_eq!(
            ordered_k64_int4.sparse_mma.as_ref().unwrap().selector,
            SparseMmaSelector::ImmediateZeroOrOne
        );
        assert_eq!(
            ordered_k64_int4.sparse_mma.as_ref().unwrap().adapter,
            SparseMmaAdapter::C4I32A2U32B2U32MetadataU32SelectorU32ToD4I32
        );
        assert_eq!(
            ordered_k64_int4.sparse_mma.as_ref().unwrap().llvm_adapter,
            SparseMmaLlvmAdapter::A2I32B2I32C4I32MetadataI32SelectorI32ToD4I32
        );
        let standard_k64_int4 = records
            .iter()
            .copied()
            .find(|record| record.id == "mma_sp_m16n8k64_s32_s4")
            .unwrap();
        assert_eq!(standard_k64_int4.minimum_ptx, "7.1");
        assert_eq!(
            standard_k64_int4.rust_arguments,
            ordered_k64_int4.rust_arguments
        );
        assert_eq!(
            standard_k64_int4.llvm_arguments,
            ordered_k64_int4.llvm_arguments
        );

        let ordered_k128_int4 = records
            .iter()
            .copied()
            .find(|record| record.id == "mma_sp_ordered_metadata_m16n8k128_s32_s4")
            .unwrap();
        let ordered_k128_int4_declaration =
            declarations[ordered_k128_int4.source_record.as_deref().unwrap()];
        assert_eq!(ordered_k128_int4.minimum_ptx, "8.5");
        assert_eq!(ordered_k128_int4.rust_arguments[1], "[u32; 4]");
        assert_eq!(ordered_k128_int4.rust_arguments[2], "[u32; 4]");
        assert_eq!(ordered_k128_int4.llvm_arguments.len(), 14);
        assert_eq!(
            ordered_k128_int4.sparse_mma.as_ref().unwrap().selector,
            SparseMmaSelector::ImmediateZero
        );
        assert_eq!(
            ordered_k128_int4.sparse_mma.as_ref().unwrap().adapter,
            SparseMmaAdapter::C4I32A4U32B4U32MetadataU32SelectorU32ToD4I32
        );
        assert_eq!(
            ordered_k128_int4.sparse_mma.as_ref().unwrap().llvm_adapter,
            SparseMmaLlvmAdapter::A4I32B4I32C4I32MetadataI32SelectorI32ToD4I32
        );
        let standard_k128_int4 = records
            .iter()
            .copied()
            .find(|record| record.id == "mma_sp_m16n8k128_s32_s4")
            .unwrap();
        assert_eq!(standard_k128_int4.minimum_ptx, "7.1");
        assert_eq!(
            standard_k128_int4.rust_arguments,
            ordered_k128_int4.rust_arguments
        );
        assert_eq!(
            standard_k128_int4.llvm_arguments,
            ordered_k128_int4.llvm_arguments
        );

        let mut wrong_k128_selector = ordered_k128_int4.clone();
        wrong_k128_selector.sparse_mma.as_mut().unwrap().selector =
            SparseMmaSelector::ImmediateZeroOrOne;
        assert!(
            validate_imported_policy(&wrong_k128_selector, ordered_k128_int4_declaration).is_err()
        );

        let mut mixed_k128_width = ordered_k128_int4.clone();
        mixed_k128_width.sparse_mma.as_mut().unwrap().b_element = SparseMmaElement::U8;
        assert!(
            validate_imported_policy(&mixed_k128_width, ordered_k128_int4_declaration)
                .unwrap_err()
                .to_string()
                .contains("unsupported sparse-MMA variant")
        );

        let mut mixed_width = ordered_k64_int4.clone();
        mixed_width.sparse_mma.as_mut().unwrap().b_element = SparseMmaElement::U8;
        assert!(
            validate_imported_policy(&mixed_width, ordered_k64_int4_declaration)
                .unwrap_err()
                .to_string()
                .contains("unsupported sparse-MMA variant")
        );

        let mut wrong_k64_selector = k64.clone();
        wrong_k64_selector.sparse_mma.as_mut().unwrap().selector =
            SparseMmaSelector::ImmediateZeroOrOne;
        assert!(validate_imported_policy(&wrong_k64_selector, k64_declaration).is_err());

        let mut wrong_k64_adapter = k64.clone();
        wrong_k64_adapter.sparse_mma.as_mut().unwrap().adapter =
            SparseMmaAdapter::C4I32A2U32B2U32MetadataU32SelectorU32ToD4I32;
        assert!(validate_imported_policy(&wrong_k64_adapter, k64_declaration).is_err());

        let mut wrong_k64_llvm_adapter = k64.clone();
        wrong_k64_llvm_adapter
            .sparse_mma
            .as_mut()
            .unwrap()
            .llvm_adapter = SparseMmaLlvmAdapter::A2I32B2I32C4I32MetadataI32SelectorI32ToD4I32;
        assert!(validate_imported_policy(&wrong_k64_llvm_adapter, k64_declaration).is_err());

        let mut wrong_k64_shape = k64.clone();
        wrong_k64_shape.sparse_mma.as_mut().unwrap().shape = SparseMmaShape::M16n8k32;
        assert!(validate_imported_policy(&wrong_k64_shape, k64_declaration).is_err());

        let mut wrong_k64_carriers = k64.clone();
        wrong_k64_carriers.dialect_operands.pop();
        assert!(validate_imported_policy(&wrong_k64_carriers, k64_declaration).is_err());

        let mut wrong_k64_lowering = k64.clone();
        wrong_k64_lowering.lowering = "generated_register_mma".into();
        assert!(validate_imported_policy(&wrong_k64_lowering, k64_declaration).is_err());

        let mut mismatched_metadata_identity = k64.clone();
        mismatched_metadata_identity
            .sparse_mma
            .as_mut()
            .unwrap()
            .metadata = SparseMmaMetadata::Ordered;
        assert!(validate_imported_policy(&mismatched_metadata_identity, k64_declaration).is_err());
    }

    #[test]
    fn cp_async_copy_recipe_admits_only_classic_llvm_forms() {
        let cases = [
            (
                CpAsyncCachePolicy::Ca,
                CpAsyncCopySize::B4,
                CpAsyncSourceSize::Full,
                Some("cp_async_ca_4"),
            ),
            (
                CpAsyncCachePolicy::Ca,
                CpAsyncCopySize::B4,
                CpAsyncSourceSize::Runtime,
                Some("cp_async_ca_zfill_4"),
            ),
            (
                CpAsyncCachePolicy::Ca,
                CpAsyncCopySize::B8,
                CpAsyncSourceSize::Full,
                Some("cp_async_ca_8"),
            ),
            (
                CpAsyncCachePolicy::Ca,
                CpAsyncCopySize::B8,
                CpAsyncSourceSize::Runtime,
                Some("cp_async_ca_zfill_8"),
            ),
            (
                CpAsyncCachePolicy::Ca,
                CpAsyncCopySize::B16,
                CpAsyncSourceSize::Full,
                Some("cp_async_ca_16"),
            ),
            (
                CpAsyncCachePolicy::Ca,
                CpAsyncCopySize::B16,
                CpAsyncSourceSize::Runtime,
                Some("cp_async_ca_zfill_16"),
            ),
            (
                CpAsyncCachePolicy::Cg,
                CpAsyncCopySize::B4,
                CpAsyncSourceSize::Full,
                None,
            ),
            (
                CpAsyncCachePolicy::Cg,
                CpAsyncCopySize::B4,
                CpAsyncSourceSize::Runtime,
                None,
            ),
            (
                CpAsyncCachePolicy::Cg,
                CpAsyncCopySize::B8,
                CpAsyncSourceSize::Full,
                None,
            ),
            (
                CpAsyncCachePolicy::Cg,
                CpAsyncCopySize::B8,
                CpAsyncSourceSize::Runtime,
                None,
            ),
            (
                CpAsyncCachePolicy::Cg,
                CpAsyncCopySize::B16,
                CpAsyncSourceSize::Full,
                Some("cp_async_cg_16"),
            ),
            (
                CpAsyncCachePolicy::Cg,
                CpAsyncCopySize::B16,
                CpAsyncSourceSize::Runtime,
                Some("cp_async_cg_zfill_16"),
            ),
        ];

        for (cache_policy, copy_size, source_size, expected) in cases {
            let copy = crate::model::CpAsyncCopy {
                cache_policy,
                copy_size,
                source_size,
                adapter: if source_size == CpAsyncSourceSize::Runtime {
                    CpAsyncAdapter::DirectPointersAndSourceSize
                } else {
                    CpAsyncAdapter::DirectPointers
                },
                runtime_validation: RuntimeValidation::Unexecuted,
            };
            assert_eq!(
                cp_async_copy_recipe(&copy).map(|recipe| recipe.id),
                expected
            );
        }
    }

    #[test]
    fn pinned_cp_async_records_match_the_closed_recipes() {
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
        let policies: Vec<_> = overlay
            .intrinsics
            .iter()
            .filter(|record| matches!(record.family.as_str(), "cp_async_copy" | "cp_async_control"))
            .collect();

        assert_eq!(policies.len(), 11);
        for policy in policies {
            let declaration = declarations[policy.source_record.as_deref().unwrap()];
            validate_imported_policy(policy, declaration).unwrap();
        }
    }

    #[test]
    fn pinned_cp_async_mbarrier_records_match_the_closed_recipes() {
        let records = pinned_cp_async_mbarrier_records();
        assert_eq!(records.len(), 4);

        for (policy, declaration) in records.values() {
            validate_imported_policy(policy, declaration).unwrap();
        }
    }

    #[test]
    fn cp_async_mbarrier_recipes_fail_closed() {
        let records = pinned_cp_async_mbarrier_records();
        let (arrive, declaration) = &records["cp_async_mbarrier_arrive"];
        let reject =
            |policy: &OverlayIntrinsic, declaration: &ImportedIntrinsic, expected: &str| {
                let error = validate_imported_policy(policy, declaration).unwrap_err();
                let message = error.to_string();
                assert!(message.contains(expected), "unexpected error: {message}");
            };

        let mut wrong_symbol = arrive.clone();
        wrong_symbol.llvm_symbol = Some("llvm.nvvm.cp.async.mbarrier.changed".into());
        reject(&wrong_symbol, declaration, "LLVM symbol mismatch");

        let mut wrong_signature = arrive.clone();
        wrong_signature.rust_arguments = vec!["*const u64".into()];
        reject(
            &wrong_signature,
            declaration,
            "closed cp.async mbarrier Rust API",
        );

        let mut wrong_operation = arrive.clone();
        wrong_operation
            .cp_async_mbarrier
            .as_mut()
            .unwrap()
            .operation = CpAsyncMbarrierOperation::ArriveNoInc;
        reject(&wrong_operation, declaration, "identity does not match");

        let mut wrong_state_space = arrive.clone();
        wrong_state_space
            .cp_async_mbarrier
            .as_mut()
            .unwrap()
            .state_space = CpAsyncMbarrierStateSpace::Shared;
        reject(&wrong_state_space, declaration, "identity does not match");

        let mut wrong_adapter = arrive.clone();
        wrong_adapter.cp_async_mbarrier.as_mut().unwrap().adapter =
            CpAsyncMbarrierAdapter::PointerToVoid;
        wrong_adapter.rust_result = "u64".into();
        reject(
            &wrong_adapter,
            declaration,
            "closed cp.async mbarrier Rust API",
        );

        let mut executed_without_evidence = arrive.clone();
        executed_without_evidence
            .cp_async_mbarrier
            .as_mut()
            .unwrap()
            .runtime_validation = RuntimeValidation::Executed;
        reject(
            &executed_without_evidence,
            declaration,
            "unrecorded cp.async mbarrier runtime validation",
        );

        let mut wrong_properties = declaration.clone();
        wrong_properties.properties.pop();
        reject(arrive, &wrong_properties, "cp.async mbarrier properties");

        let mut wrong_selection = declaration.clone();
        wrong_selection.selections[0].source_record = "CP_ASYNC_MBARRIER_CHANGED".into();
        reject(
            arrive,
            &wrong_selection,
            "imported cp.async mbarrier selection changed",
        );

        let mut wrong_floor = arrive.clone();
        wrong_floor.minimum_sm = Some("sm_90".into());
        reject(&wrong_floor, declaration, "effects or target floor");

        let mut wrong_llvm_route = arrive.clone();
        wrong_llvm_route
            .backend_lowerings
            .iter_mut()
            .find(|lowering| lowering.backend == IntrinsicBackend::LlvmNvptx)
            .unwrap()
            .mechanism = BackendLoweringMechanism::InlinePtx;
        reject(&wrong_llvm_route, declaration, "reviewed typed-LLVM");

        let mut wrong_lib_route = arrive.clone();
        wrong_lib_route
            .backend_lowerings
            .iter_mut()
            .find(|lowering| lowering.backend == IntrinsicBackend::LibNvvm)
            .unwrap()
            .mechanism = BackendLoweringMechanism::TypedNvvm;
        reject(&wrong_lib_route, declaration, "reviewed typed-LLVM");

        let mut mixed_family = arrive.clone();
        mixed_family.cp_async_control = Some(crate::model::CpAsyncControl {
            operation: CpAsyncControlOperation::CommitGroup,
            adapter: CpAsyncControlAdapter::NoOperands,
            runtime_validation: RuntimeValidation::Unexecuted,
        });
        reject(
            &mixed_family,
            declaration,
            "mixes another generated-family contract",
        );
    }

    #[test]
    fn pinned_mbarrier_basic_records_match_the_closed_recipes() {
        let records = pinned_mbarrier_basic_records();
        assert_eq!(records.len(), 4);

        for (policy, declaration) in records.values() {
            validate_imported_policy(policy, declaration).unwrap();
        }
    }

    #[test]
    fn mbarrier_basic_recipes_fail_closed() {
        let records = pinned_mbarrier_basic_records();
        let (init, init_declaration) = &records["mbarrier_init"];
        let reject =
            |policy: &OverlayIntrinsic, declaration: &ImportedIntrinsic, expected: &str| {
                let error = validate_imported_policy(policy, declaration).unwrap_err();
                let message = error.to_string();
                assert!(message.contains(expected), "unexpected error: {message}");
            };

        let mut wrong_symbol = init.clone();
        wrong_symbol.llvm_symbol = Some("llvm.nvvm.mbarrier.init.changed".into());
        reject(&wrong_symbol, init_declaration, "LLVM symbol mismatch");

        let mut wrong_signature = init.clone();
        wrong_signature.rust_arguments = vec!["*mut u32".into(), "u32".into()];
        reject(
            &wrong_signature,
            init_declaration,
            "unsafe mbarrier raw and compatibility API",
        );

        let mut wrong_operation = init.clone();
        wrong_operation.mbarrier_basic.as_mut().unwrap().operation = MbarrierBasicOperation::Arrive;
        reject(
            &wrong_operation,
            init_declaration,
            "operation, state space, and adapter disagree",
        );

        let mut wrong_adapter = init.clone();
        wrong_adapter.mbarrier_basic.as_mut().unwrap().adapter =
            MbarrierBasicAdapter::PointerToVoid;
        reject(
            &wrong_adapter,
            init_declaration,
            "operation, state space, and adapter disagree",
        );

        let mut executed_without_evidence = init.clone();
        executed_without_evidence
            .mbarrier_basic
            .as_mut()
            .unwrap()
            .runtime_validation = RuntimeValidation::Executed;
        reject(
            &executed_without_evidence,
            init_declaration,
            "unrecorded mbarrier runtime validation",
        );

        let mut wrong_properties = init_declaration.clone();
        wrong_properties.properties.pop();
        reject(init, &wrong_properties, "mbarrier properties");

        let mut wrong_selection = init_declaration.clone();
        wrong_selection.selections[0].source_record = "MBARRIER_INIT_CHANGED".into();
        reject(
            init,
            &wrong_selection,
            "imported mbarrier selection changed",
        );

        let mut wrong_ptx_floor = init.clone();
        wrong_ptx_floor.minimum_ptx = "7.1".into();
        reject(
            &wrong_ptx_floor,
            init_declaration,
            "effects or target floor",
        );

        let mut wrong_sm_floor = init.clone();
        wrong_sm_floor.minimum_sm = Some("sm_90".into());
        reject(&wrong_sm_floor, init_declaration, "effects or target floor");

        let mut wrong_llvm_route = init.clone();
        wrong_llvm_route
            .backend_lowerings
            .iter_mut()
            .find(|lowering| lowering.backend == IntrinsicBackend::LlvmNvptx)
            .unwrap()
            .mechanism = BackendLoweringMechanism::InlinePtx;
        reject(
            &wrong_llvm_route,
            init_declaration,
            "reviewed mbarrier backend routes",
        );

        let mut wrong_lib_nvvm_route = init.clone();
        wrong_lib_nvvm_route
            .backend_lowerings
            .iter_mut()
            .find(|lowering| lowering.backend == IntrinsicBackend::LibNvvm)
            .unwrap()
            .mechanism = BackendLoweringMechanism::TypedNvvm;
        reject(
            &wrong_lib_nvvm_route,
            init_declaration,
            "reviewed mbarrier backend routes",
        );

        let mut route_with_unreviewed_floor = init.clone();
        route_with_unreviewed_floor.backend_lowerings[0].minimum_sm = Some("sm_90".into());
        reject(
            &route_with_unreviewed_floor,
            init_declaration,
            "reviewed mbarrier backend routes",
        );

        let mut mixed_family = init.clone();
        mixed_family.cp_async_control = Some(crate::model::CpAsyncControl {
            operation: CpAsyncControlOperation::CommitGroup,
            adapter: CpAsyncControlAdapter::NoOperands,
            runtime_validation: RuntimeValidation::Unexecuted,
        });
        reject(
            &mixed_family,
            init_declaration,
            "mixes another generated-family contract",
        );
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
    fn movmatrix_recipe_is_exact_and_fails_closed() {
        let valid = movmatrix_policy();
        validate_ptx_native_policy(&valid).unwrap();

        let reject = |policy: &OverlayIntrinsic, expected: &str| {
            let message = validate_ptx_native_policy(policy).unwrap_err().to_string();
            assert!(message.contains(expected), "unexpected error: {message}");
        };

        let mut wrong_shape = valid.clone();
        wrong_shape.expected_ptx.modifiers[2] = "m16n8".into();
        reject(&wrong_shape, "closed movmatrix recipe");

        let mut wrong_participation = valid.clone();
        wrong_participation.convergent = false;
        reject(&wrong_participation, "closed movmatrix recipe");

        let mut wrong_floor = valid.clone();
        wrong_floor.backend_lowerings[0].minimum_ptx = Some("8.0".into());
        reject(&wrong_floor, "exact movmatrix floor");

        let mut mixed = valid;
        mixed.warp_barrier = Some(crate::model::WarpBarrier {
            participation:
                WarpBarrierParticipation::ExecutingLaneNamedAllNamedLanesSameInstructionAndMask,
            legacy_pre_sm70: PreSm70MemberMaskRule::AllNamedLanesConvergedAndOnlyNamedLanesActive,
            adapter: WarpBarrierAdapter::DirectMemberMask,
            mask_encoding: WarpBarrierMaskEncoding::RegisterOrImmediate,
            memory_ordering: WarpBarrierMemoryOrdering::ParticipatingLanes,
        });
        reject(&mixed, "mixes another generated-family contract");
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
            assert_eq!(
                policy.vote.as_ref().unwrap().legacy_pre_sm70,
                PreSm70MemberMaskRule::AllNamedLanesConvergedAndOnlyNamedLanesActive
            );

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
    fn warp_shuffle_variants_keep_exact_identity_clamp_and_eight_selections() {
        for (mode, value_kind, clamp) in [
            (WarpShuffleMode::Idx, WarpShuffleValueKind::I32, 31),
            (WarpShuffleMode::Bfly, WarpShuffleValueKind::I32, 31),
            (WarpShuffleMode::Down, WarpShuffleValueKind::I32, 31),
            (WarpShuffleMode::Up, WarpShuffleValueKind::I32, 0),
            (WarpShuffleMode::Idx, WarpShuffleValueKind::F32, 31),
            (WarpShuffleMode::Bfly, WarpShuffleValueKind::F32, 31),
            (WarpShuffleMode::Down, WarpShuffleValueKind::F32, 31),
            (WarpShuffleMode::Up, WarpShuffleValueKind::F32, 0),
        ] {
            let policy = warp_shuffle_policy(mode, value_kind);
            let declaration = warp_shuffle_declaration(mode, value_kind);
            validate_imported_policy(&policy, &declaration).unwrap();

            assert_eq!(policy.warp_shuffle.as_ref().unwrap().clamp, clamp);
            assert_eq!(
                declaration
                    .selections
                    .iter()
                    .filter(|selection| selection_matches_policy(&policy, selection))
                    .count(),
                8
            );

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
            assert_eq!(resolved.selections.len(), 8);
            assert_eq!(resolved.warp_shuffle, policy.warp_shuffle);
        }
    }

    #[test]
    fn pinned_warp_shuffle_records_match_the_closed_recipes() {
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
        let all_policies: Vec<_> = overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "warp_shuffle")
            .collect();
        assert_eq!(all_policies.len(), 12);
        let native_policies: Vec<_> = all_policies
            .iter()
            .copied()
            .filter(|record| record.source.is_some())
            .collect();
        assert_eq!(native_policies.len(), 4);
        for policy in native_policies {
            validate_ptx_native_policy(policy).unwrap();
        }
        let policies: Vec<_> = overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "warp_shuffle" && record.source_record.is_some())
            .collect();
        assert_eq!(policies.len(), 8);
        for policy in policies {
            let declaration = declarations[policy.source_record.as_deref().unwrap()];
            validate_imported_policy(policy, declaration).unwrap();
        }
    }

    #[test]
    fn pinned_llvm_has_no_direct_i64_or_f64_shuffle_record() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let imported: ImportedFile =
            read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
        let direct_64: Vec<_> = imported
            .intrinsics
            .iter()
            .filter(|record| {
                record.llvm_name.starts_with("llvm.nvvm.shfl")
                    && (record.llvm_name.contains(".i64") || record.llvm_name.contains(".f64"))
            })
            .map(|record| record.llvm_name.as_str())
            .collect();
        assert!(
            direct_64.is_empty(),
            "unexpected LLVM records: {direct_64:?}"
        );
    }

    #[test]
    fn i64_warp_shuffle_recipes_are_exact_ptx_native_pairs() {
        let cases = [
            (
                WarpShuffleMode::Idx,
                "shuffle_u64_sync",
                "i0058",
                "warp.shuffle.sync.idx.i64",
                "shfl.sync.idx.b32",
                "idx",
                31,
                "ShflSyncIdxI64Op",
                "nvvm.shfl_sync_idx_i64",
            ),
            (
                WarpShuffleMode::Bfly,
                "shuffle_xor_u64_sync",
                "i0059",
                "warp.shuffle.sync.bfly.i64",
                "shfl.sync.bfly.b32",
                "bfly",
                31,
                "ShflSyncBflyI64Op",
                "nvvm.shfl_sync_bfly_i64",
            ),
            (
                WarpShuffleMode::Down,
                "shuffle_down_u64_sync",
                "i0060",
                "warp.shuffle.sync.down.i64",
                "shfl.sync.down.b32",
                "down",
                31,
                "ShflSyncDownI64Op",
                "nvvm.shfl_sync_down_i64",
            ),
            (
                WarpShuffleMode::Up,
                "shuffle_up_u64_sync",
                "i0061",
                "warp.shuffle.sync.up.i64",
                "shfl.sync.up.b32",
                "up",
                0,
                "ShflSyncUpI64Op",
                "nvvm.shfl_sync_up_i64",
            ),
        ];

        for (mode, id, abi_id, operation_key, instruction, ptx_mode, clamp, op_type, op_name) in
            cases
        {
            let policy = warp_shuffle_policy(mode, WarpShuffleValueKind::I64);
            validate_ptx_native_policy(&policy).unwrap();

            assert_eq!(policy.id, id);
            assert_eq!(policy.abi_id, abi_id);
            assert_eq!(policy.operation_key, operation_key);
            assert_eq!(
                policy.source,
                Some(IntrinsicSource::PtxNative {
                    instruction: instruction.into(),
                })
            );
            assert!(policy.source_record.is_none());
            assert!(policy.llvm_symbol.is_none());
            assert!(policy.resolved_llvm_symbol.is_none());
            assert!(policy.llvm_arguments.is_empty());
            assert!(policy.llvm_results.is_empty());
            assert_eq!(policy.rust_arguments, ["u32", "u64", "u32"]);
            assert_eq!(policy.rust_result, "u64");
            assert!(!policy.safe);
            assert!(policy.must_use);
            assert_eq!(policy.dialect_op_type, op_type);
            assert_eq!(policy.dialect_op_name, op_name);
            assert_eq!(policy.dialect_operands, ["i32", "i64", "i32"]);
            assert_eq!(policy.dialect_results, ["i64"]);
            assert_eq!(policy.lowering, "generated_warp_shuffle_i64_inline_ptx");

            let shuffle = policy.warp_shuffle.as_ref().unwrap();
            assert_eq!(shuffle.clamp, clamp);
            assert_eq!(
                shuffle.adapter,
                WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble
            );
            assert_eq!(
                shuffle.lane_encoding,
                WarpShuffleOperandEncoding::RegisterOnly
            );
            assert_eq!(
                shuffle.mask_encoding,
                WarpShuffleOperandEncoding::RegisterOnly
            );
            assert_eq!(
                policy.expected_ptx,
                InstructionPattern::new(
                    "shfl",
                    &["sync", ptx_mode, "b32"],
                    vec![
                        OperandPattern::Exact { value: "lo".into() },
                        OperandPattern::Exact { value: "lo".into() },
                        OperandPattern::Register,
                        OperandPattern::Exact {
                            value: clamp.to_string(),
                        },
                        OperandPattern::Register,
                    ],
                )
            );

            let routes: BTreeMap<_, _> = policy
                .backend_lowerings
                .iter()
                .map(|route| (route.backend, route))
                .collect();
            assert_eq!(routes.len(), 2);
            for backend in [IntrinsicBackend::LlvmNvptx, IntrinsicBackend::LibNvvm] {
                assert_eq!(
                    routes[&backend].mechanism,
                    BackendLoweringMechanism::InlinePtx
                );
                assert_eq!(routes[&backend].minimum_ptx.as_deref(), Some("6.0"));
            }
            assert_eq!(
                routes[&IntrinsicBackend::LlvmNvptx].minimum_sm.as_deref(),
                Some("sm_30")
            );
            assert_eq!(
                routes[&IntrinsicBackend::LibNvvm].minimum_sm.as_deref(),
                Some("sm_75")
            );

            let mut record = evidence();
            record.id = policy.id.clone();
            record.source = policy.source.clone();
            record.source_record = None;
            record.llvm_symbol = None;
            record.llvm_arguments.clear();
            record.llvm_results.clear();
            record.expected_ptx = policy.expected_ptx.clone();
            let resolved = resolve_record(
                &policy,
                resolve_policy_source(&policy).unwrap(),
                None,
                &record,
                "test",
                "LLVM version test",
                "0123456789abcdef",
                vec![],
                1,
            )
            .unwrap();
            assert!(resolved.llvm.is_none());
            assert!(resolved.selections.is_empty());
            assert_eq!(resolved.warp_shuffle, policy.warp_shuffle);
        }
    }

    #[test]
    fn i64_warp_shuffle_contract_rejects_unreviewed_changes() {
        let valid = warp_shuffle_policy(WarpShuffleMode::Idx, WarpShuffleValueKind::I64);
        validate_ptx_native_policy(&valid).unwrap();

        let reject = |policy: &OverlayIntrinsic, expected: &str| {
            let error = validate_ptx_native_policy(policy).unwrap_err();
            let message = error.to_string();
            assert!(message.contains(expected), "unexpected error: {message}");
        };

        let mut fabricated_llvm = valid.clone();
        fabricated_llvm.source = None;
        fabricated_llvm.source_record = Some("int_nvvm_shfl_sync_idx_i64".into());
        fabricated_llvm.llvm_symbol = Some("llvm.nvvm.shfl.sync.idx.i64".into());
        fabricated_llvm.llvm_arguments =
            vec!["i32".into(), "i64".into(), "i32".into(), "i32".into()];
        fabricated_llvm.llvm_results = vec!["i64".into()];
        reject(
            &fabricated_llvm,
            "source kind and imported declaration disagree",
        );

        let mut wrong_source = valid.clone();
        wrong_source.source = Some(IntrinsicSource::PtxNative {
            instruction: "shfl.sync.down.b32".into(),
        });
        reject(&wrong_source, "warp-shuffle identity");

        let mut wrong_adapter = valid.clone();
        wrong_adapter.warp_shuffle.as_mut().unwrap().adapter =
            WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp;
        reject(&wrong_adapter, "semantic or operand contract");

        let mut wrong_mode = valid.clone();
        wrong_mode.warp_shuffle.as_mut().unwrap().mode = WarpShuffleMode::Up;
        wrong_mode.warp_shuffle.as_mut().unwrap().clamp = 0;
        wrong_mode.expected_ptx.modifiers[1] = "up".into();
        wrong_mode.expected_ptx.operands[3] = OperandPattern::Exact { value: "0".into() };
        reject(&wrong_mode, "warp-shuffle identity");

        let mut wrong_clamp = valid.clone();
        wrong_clamp.warp_shuffle.as_mut().unwrap().clamp = 0;
        reject(&wrong_clamp, "semantic or operand contract");

        let mut broad_encoding = valid.clone();
        broad_encoding.warp_shuffle.as_mut().unwrap().lane_encoding =
            WarpShuffleOperandEncoding::RegisterOrImmediate;
        reject(&broad_encoding, "semantic or operand contract");

        let mut typed_backend = valid.clone();
        typed_backend.backend_lowerings[0].mechanism = BackendLoweringMechanism::TypedNvvm;
        reject(&typed_backend, "reviewed LLVM and libNVVM routes");

        let mut wrong_native_floor = valid.clone();
        wrong_native_floor.minimum_sm = Some("sm_70".into());
        reject(&wrong_native_floor, "target floor");

        let mut wrong_profile_floor = valid.clone();
        wrong_profile_floor
            .backend_lowerings
            .iter_mut()
            .find(|route| route.backend == IntrinsicBackend::LibNvvm)
            .unwrap()
            .minimum_sm = Some("sm_80".into());
        reject(&wrong_profile_floor, "profile floor");

        let mut safe = valid.clone();
        safe.safe = true;
        safe.safe_allowlist_reason = Some("incorrectly hides participation obligations".into());
        reject(&safe, "unsafe must-use warp-shuffle");

        let mut wrong_ptx = valid;
        wrong_ptx.expected_ptx.operands[0] = OperandPattern::Register;
        reject(&wrong_ptx, "closed shfl.sync recipe");
    }

    #[test]
    fn warp_shuffle_contract_rejects_unreviewed_policy_changes() {
        let valid = warp_shuffle_policy(WarpShuffleMode::Idx, WarpShuffleValueKind::I32);
        let declaration = warp_shuffle_declaration(WarpShuffleMode::Idx, WarpShuffleValueKind::I32);

        let reject_policy = |policy: &OverlayIntrinsic, expected: &str| {
            let error = match validate_imported_policy(policy, &declaration) {
                Ok(()) => panic!("{expected} mutation was accepted"),
                Err(error) => error,
            };
            let message = error.to_string();
            assert!(message.contains(expected), "unexpected error: {message}");
        };

        let mut wrong_identity = valid.clone();
        wrong_identity.operation_key = "warp.shuffle.sync.idx.changed".into();
        reject_policy(&wrong_identity, "warp-shuffle identity");

        let mut safe = valid.clone();
        safe.safe = true;
        safe.safe_allowlist_reason = Some("incorrectly hides participation obligations".into());
        reject_policy(&safe, "unsafe must-use warp-shuffle");

        let mut wrong_signature = valid.clone();
        wrong_signature.dialect_operands.pop();
        reject_policy(&wrong_signature, "closed warp-shuffle lowering recipe");

        let mut wrong_clamp = valid.clone();
        wrong_clamp.warp_shuffle.as_mut().unwrap().clamp = 0;
        reject_policy(&wrong_clamp, "semantic or operand contract");

        let mut missing_contract = valid.clone();
        missing_contract.warp_shuffle = None;
        reject_policy(&missing_contract, "closed warp-shuffle contract");

        let mut mixed_contract = valid.clone();
        mixed_contract.vote = vote_policy(VoteMode::All).vote;
        reject_policy(&mixed_contract, "mixes another generated-family contract");

        let mut wrong_backend_floor = valid.clone();
        wrong_backend_floor
            .backend_lowerings
            .iter_mut()
            .find(|lowering| lowering.backend == IntrinsicBackend::LibNvvm)
            .unwrap()
            .minimum_sm = Some("sm_80".into());
        reject_policy(&wrong_backend_floor, "profile floor");
    }

    #[test]
    fn warp_shuffle_contract_rejects_selection_drift() {
        let valid = warp_shuffle_policy(WarpShuffleMode::Down, WarpShuffleValueKind::F32);
        let declaration =
            warp_shuffle_declaration(WarpShuffleMode::Down, WarpShuffleValueKind::F32);
        let reject = |declaration: &ImportedIntrinsic, expected: &str| {
            let error = validate_imported_policy(&valid, declaration).unwrap_err();
            let message = error.to_string();
            assert!(message.contains(expected), "unexpected error: {message}");
        };

        let mut missing_selection = declaration.clone();
        missing_selection.selections.pop();
        reject(
            &missing_selection,
            "eight distinct operand-encoding selections",
        );

        let mut duplicate_selection = declaration.clone();
        duplicate_selection.selections[7].source_record =
            duplicate_selection.selections[0].source_record.clone();
        reject(
            &duplicate_selection,
            "eight distinct operand-encoding selections",
        );

        let mut empty_selection_name = declaration.clone();
        empty_selection_name.selections[7].source_record.clear();
        reject(
            &empty_selection_name,
            "eight distinct operand-encoding selections",
        );

        let mut wrong_asm = declaration.clone();
        wrong_asm.selections[0].asm =
            "shfl.sync.up.b32 \t$dst, $src, $offset, $mask, $threadmask;".into();
        reject(&wrong_asm, "selections disagree on PTX shape");

        let mut wrong_predicate = declaration.clone();
        wrong_predicate.selections[0].predicates[0] = "Subtarget->getPTXVersion() >= 61".into();
        reject(&wrong_predicate, "selections disagree on PTX shape");

        let mut constrained = declaration;
        constrained.selections[0]
            .constraints
            .immediate_bindings
            .push(crate::model::ImportedImmediateBinding {
                argument_index: 2,
                value: 1,
            });
        reject(&constrained, "selections disagree on PTX shape");

        let mut wrong_classes =
            warp_shuffle_declaration(WarpShuffleMode::Down, WarpShuffleValueKind::F32);
        wrong_classes.classes.pop();
        reject(&wrong_classes, "class or effects");
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
    fn sync_mask_matches_the_closed_warp_barrier_recipe() {
        let policy = warp_barrier_policy();
        let declaration = warp_barrier_declaration();
        validate_imported_policy(&policy, &declaration).unwrap();

        let selected: Vec<_> = declaration
            .selections
            .iter()
            .filter(|selection| selection_matches_policy(&policy, selection))
            .collect();
        assert_eq!(selected.len(), 2);
        assert_eq!(
            selected
                .iter()
                .map(|selection| selection.source_record.as_str())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["INT_BAR_WARP_SYNC_I", "INT_BAR_WARP_SYNC_R"])
        );

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
        assert_eq!(resolved.warp_barrier, policy.warp_barrier);
    }

    #[test]
    fn sync_mask_recipe_rejects_unreviewed_contract_and_selection_changes() {
        let valid = warp_barrier_policy();
        let declaration = warp_barrier_declaration();

        let mut wrong_identity = valid.clone();
        wrong_identity.id = "bar_warp_sync".into();
        assert!(
            validate_imported_policy(&wrong_identity, &declaration)
                .unwrap_err()
                .to_string()
                .contains("warp-barrier identity")
        );

        let mut missing_contract = valid.clone();
        missing_contract.warp_barrier = None;
        assert!(
            validate_imported_policy(&missing_contract, &declaration)
                .unwrap_err()
                .to_string()
                .contains("closed warp-barrier contract")
        );

        let mut safe_raw_api = valid.clone();
        safe_raw_api.safe = true;
        safe_raw_api.safe_allowlist_reason = Some("incorrectly hides participation rules".into());
        assert!(
            validate_imported_policy(&safe_raw_api, &declaration)
                .unwrap_err()
                .to_string()
                .contains("unsafe raw API")
        );

        let mut wrong_memory = valid.clone();
        wrong_memory.memory = "none".into();
        assert!(
            validate_imported_policy(&wrong_memory, &declaration)
                .unwrap_err()
                .to_string()
                .contains("effects or target floor")
        );

        let mut register_only = valid.clone();
        register_only.expected_ptx.operands[0] = OperandPattern::Register;
        assert!(
            validate_imported_policy(&register_only, &declaration)
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

        let mut wrong_predicate = declaration.clone();
        wrong_predicate.selections[1].predicates[0] = "Subtarget->getPTXVersion() >= 61".into();
        assert!(
            validate_imported_policy(&valid, &wrong_predicate)
                .unwrap_err()
                .to_string()
                .contains("selections disagree")
        );

        let mut missing_libnvvm_floor = valid;
        missing_libnvvm_floor
            .backend_lowerings
            .iter_mut()
            .find(|lowering| lowering.backend == IntrinsicBackend::LibNvvm)
            .unwrap()
            .minimum_sm = None;
        assert!(
            validate_imported_policy(&missing_libnvvm_floor, &declaration)
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
    fn packed_conversion_evidence_separates_llvm_declaration_facts_from_libnvvm() {
        for (destination, result) in [
            (PackedConversionDestinationFormat::Bf16x2, "v2bf16"),
            (PackedConversionDestinationFormat::F16x2, "v2f16"),
        ] {
            let policy = packed_conversion_policy(
                destination,
                PackedConversionRounding::NearestEven,
                PackedConversionSaturation::None,
            );
            let llvm = policy
                .backend_lowerings
                .iter()
                .find(|lowering| lowering.backend == IntrinsicBackend::LlvmNvptx)
                .unwrap();
            let mut record = packed_conversion_evidence(&policy);
            record.status = "validated".into();
            record.stages = [
                EvidenceStageKind::DeclarationCanonicalization,
                EvidenceStageKind::BackendCodegen,
                EvidenceStageKind::PtxAssembly,
            ]
            .into_iter()
            .map(|stage| {
                evidence_stage(
                    stage,
                    BackendLoweringMechanism::TypedNvvm,
                    &["sm_80", "ptx70"],
                )
            })
            .collect();
            let assembly = record
                .stages
                .iter_mut()
                .find(|stage| stage.stage == EvidenceStageKind::PtxAssembly)
                .unwrap();
            assembly.tool_path = Some("/usr/local/cuda/bin/ptxas".into());
            assembly.tool_version = Some("CUDA 13.3 V13.3.33".into());
            assembly.tool_sha256 =
                Some("7fdd01a4cf50e30746da98989c9272a907f491e6fd7fecfda14642e4375f88fb".into());
            assert_eq!(record.concrete_llvm_results, [result]);
            validate_packed_conversion_backend_evidence(&policy, &record, llvm).unwrap();

            let mut lowered = record.clone();
            lowered.status = "lowered".into();
            let error =
                validate_packed_conversion_backend_evidence(&policy, &lowered, llvm).unwrap_err();
            assert!(
                error.to_string().contains("validated evidence status"),
                "{error:#}"
            );

            for required in [
                EvidenceStageKind::DeclarationCanonicalization,
                EvidenceStageKind::BackendCodegen,
                EvidenceStageKind::PtxAssembly,
            ] {
                let mut missing = record.clone();
                missing.stages.retain(|stage| stage.stage != required);
                let error = validate_packed_conversion_backend_evidence(&policy, &missing, llvm)
                    .unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains("successful auxiliary typed-NVVM"),
                    "{error:#}"
                );

                let mut failed = record.clone();
                failed
                    .stages
                    .iter_mut()
                    .find(|stage| stage.stage == required)
                    .unwrap()
                    .outcome = "failed".into();
                let error = validate_packed_conversion_backend_evidence(&policy, &failed, llvm)
                    .unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains("successful auxiliary typed-NVVM"),
                    "{error:#}"
                );

                let mut wrong_mechanism = record.clone();
                wrong_mechanism
                    .stages
                    .iter_mut()
                    .find(|stage| stage.stage == required)
                    .unwrap()
                    .mechanism = Some(BackendLoweringMechanism::InlinePtx);
                let error =
                    validate_packed_conversion_backend_evidence(&policy, &wrong_mechanism, llvm)
                        .unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains("successful auxiliary typed-NVVM"),
                    "{error:#}"
                );
            }

            let mut missing_tool_identity = record.clone();
            missing_tool_identity
                .stages
                .iter_mut()
                .find(|stage| stage.stage == EvidenceStageKind::PtxAssembly)
                .unwrap()
                .tool_sha256 = None;
            let error =
                validate_packed_conversion_backend_evidence(&policy, &missing_tool_identity, llvm)
                    .unwrap_err();
            assert!(
                error.to_string().contains("exact tool identity"),
                "{error:#}"
            );

            for stage_kind in [
                EvidenceStageKind::BackendCodegen,
                EvidenceStageKind::PtxAssembly,
            ] {
                let mut wrong_floor = record.clone();
                wrong_floor
                    .stages
                    .iter_mut()
                    .find(|stage| stage.stage == stage_kind)
                    .unwrap()
                    .targets = vec!["sm_75".into(), "ptx70".into()];
                let error =
                    validate_packed_conversion_backend_evidence(&policy, &wrong_floor, llvm)
                        .unwrap_err();
                assert!(
                    error.to_string().contains("catalog floor sm_80"),
                    "{error:#}"
                );
            }

            record.declaration_attributes_canonicalized = None;
            let error =
                validate_packed_conversion_backend_evidence(&policy, &record, llvm).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("canonical declaration attributes")
            );
        }

        let policy = packed_conversion_policy(
            PackedConversionDestinationFormat::Bf16x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::None,
        );
        let libnvvm = policy
            .backend_lowerings
            .iter()
            .find(|lowering| lowering.backend == IntrinsicBackend::LibNvvm)
            .unwrap();
        let mut record = packed_conversion_evidence(&policy);
        record.concrete_llvm_arguments.clear();
        record.concrete_llvm_results.clear();
        record.declaration_attributes_canonicalized = None;
        validate_packed_conversion_backend_evidence(&policy, &record, libnvvm).unwrap();

        record.concrete_llvm_arguments = policy.llvm_arguments.clone();
        let error =
            validate_packed_conversion_backend_evidence(&policy, &record, libnvvm).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("must not claim typed LLVM support")
        );

        record.concrete_llvm_arguments.clear();
        record.stages.push(evidence_stage(
            EvidenceStageKind::BackendCodegen,
            BackendLoweringMechanism::TypedNvvm,
            &["sm_80", "ptx70"],
        ));
        let error =
            validate_packed_conversion_backend_evidence(&policy, &record, libnvvm).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("must not claim typed LLVM support")
        );
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

        record.stages[0].targets = vec!["sm_75a".into(), "ptx65".into()];
        assert!(validate_selected_stage_targets(&target_policy, &record, &lowering).is_err());
        record.stages[0].targets = vec!["sm_75".into(), "ptx65".into()];

        record.stages[1].targets = vec!["sm_80".into(), "ptx65".into()];
        validate_selected_stage_targets(&target_policy, &record, &lowering).unwrap();

        record.stages[1].targets = vec!["sm_90a".into(), "ptx65".into()];
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
    fn exact_and_family_evidence_targets_match_at_every_stage() {
        let lowering = crate::model::OverlayBackendLowering {
            backend: IntrinsicBackend::LlvmNvptx,
            mechanism: BackendLoweringMechanism::InlinePtx,
            evidence_profile: "test".into(),
            minimum_ptx: None,
            minimum_sm: None,
        };

        for (target, wrong_targets) in [
            ("sm_120a", ["sm_120", "sm_120f", "sm_121a"]),
            ("sm_120f", ["sm_120", "sm_120a", "sm_121f"]),
        ] {
            let mut target_policy = policy();
            target_policy.minimum_ptx = "8.7".into();
            target_policy.targets = target.into();
            let mut record = evidence();
            record.status = "validated".into();
            record.runtime_validation = Some(RuntimeValidation::Unexecuted);
            record.stages = vec![
                evidence_stage(
                    EvidenceStageKind::BackendCodegen,
                    BackendLoweringMechanism::InlinePtx,
                    &[target, "ptx87"],
                ),
                evidence_stage(
                    EvidenceStageKind::PtxAssembly,
                    BackendLoweringMechanism::InlinePtx,
                    &[target, "ptx87"],
                ),
            ];
            validate_selected_stage_targets(&target_policy, &record, &lowering).unwrap();

            for wrong in wrong_targets {
                record.stages[0].targets = vec![wrong.into(), "ptx87".into()];
                let error = validate_selected_stage_targets(&target_policy, &record, &lowering)
                    .unwrap_err()
                    .to_string();
                assert!(error.contains(target), "{error}");
            }
            record.stages[0].targets = vec![target.into(), "ptx87".into()];
            for wrong in wrong_targets {
                record.stages[1].targets = vec![wrong.into(), "ptx87".into()];
                let error = validate_selected_stage_targets(&target_policy, &record, &lowering)
                    .unwrap_err()
                    .to_string();
                assert!(error.contains(target), "{error}");
            }
        }
    }

    #[test]
    fn suffixed_evidence_target_spellings_are_normalized() {
        for target in ["sm_120a", "compute_120a", "sm_120f", "compute_120f"] {
            assert!(is_normalized_stage_target(target), "{target}");
        }
        for target in ["sm_120", "compute_120", "ptx87"] {
            assert!(is_normalized_stage_target(target), "{target}");
        }
        for target in ["sm_0120a", "sm_120af", "sm_120x", "compute_120A"] {
            assert!(!is_normalized_stage_target(target), "{target}");
        }
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
    fn hardware_targets_are_parsed_without_losing_suffix_semantics() {
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

        let mut exact = policy();
        exact.targets = "sm_120a".into();
        assert_eq!(
            parse_hardware_target(&exact).unwrap(),
            CatalogHardwareTarget::AnyOf {
                alternatives: vec![CatalogHardwareAlternative::ExactArchitecture { sm: 120 }],
            }
        );

        let mut family = policy();
        family.targets = "sm_120f".into();
        assert_eq!(
            parse_hardware_target(&family).unwrap(),
            CatalogHardwareTarget::AnyOf {
                alternatives: vec![CatalogHardwareAlternative::FamilyTarget { sm: 120 }],
            }
        );
    }

    #[test]
    fn malformed_or_conflicting_hardware_targets_are_rejected() {
        for malformed in [
            "sm_120",
            "sm_120af",
            "sm_120A",
            "sm_0120a",
            "sm_0a",
            "sm_120+",
            "compute_120a",
            "all ",
        ] {
            let mut record = policy();
            record.targets = malformed.into();
            assert!(parse_hardware_target(&record).is_err(), "{malformed}");
        }

        let mut suffixed_minimum = policy();
        suffixed_minimum.minimum_sm = Some("sm_90a".into());
        assert!(parse_hardware_target(&suffixed_minimum).is_err());

        for target in ["sm_120a", "sm_120f"] {
            let mut conflicting = policy();
            conflicting.targets = target.into();
            conflicting.minimum_sm = Some("sm_120".into());
            let error = parse_hardware_target(&conflicting).unwrap_err().to_string();
            assert!(error.contains("cannot be combined"), "{error}");
        }
    }

    #[test]
    fn exact_inline_ptx_routes_can_inherit_exact_or_family_targets() {
        for target in ["sm_120a", "sm_120f"] {
            let mut record = policy();
            record.minimum_ptx = "8.7".into();
            record.targets = target.into();
            record.backend_lowerings = [IntrinsicBackend::LlvmNvptx, IntrinsicBackend::LibNvvm]
                .into_iter()
                .map(|backend| crate::model::OverlayBackendLowering {
                    backend,
                    mechanism: BackendLoweringMechanism::InlinePtx,
                    evidence_profile: "test".into(),
                    minimum_ptx: Some("8.7".into()),
                    minimum_sm: None,
                })
                .collect();

            ensure_exact_inline_ptx_backends(
                &record,
                [
                    (IntrinsicBackend::LlvmNvptx, "8.7", None),
                    (IntrinsicBackend::LibNvvm, "8.7", None),
                ],
                "test",
            )
            .unwrap();
            for lowering in &record.backend_lowerings {
                assert_eq!(
                    backend_target_requirement(&record, lowering)
                        .unwrap()
                        .hardware,
                    parse_hardware_target(&record).unwrap()
                );
            }

            record.backend_lowerings[0].minimum_sm = Some("sm_120".into());
            assert!(
                ensure_exact_inline_ptx_backends(
                    &record,
                    [
                        (IntrinsicBackend::LlvmNvptx, "8.7", None),
                        (IntrinsicBackend::LibNvvm, "8.7", None),
                    ],
                    "test",
                )
                .is_err()
            );
        }
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
    fn generated_abi_binding_uses_catalog_identity_not_axis_position() {
        let mut record = policy();
        let mut frozen = ledger_entry(&record);
        frozen.abi_id = "i9001".into();
        record.abi_id.clear();
        let mut overlay = overlay_file(vec![record]);

        bind_generated_abi_ids(&mut overlay, &ledger(vec![frozen])).unwrap();

        assert_eq!(overlay.intrinsics[0].abi_id, "i9001");
    }

    #[test]
    fn generated_abi_binding_rejects_missing_tombstoned_or_ambiguous_identity() {
        let record = policy();
        let mut unbound = record.clone();
        unbound.abi_id.clear();

        let error = bind_generated_abi_ids(
            &mut overlay_file(vec![unbound.clone()]),
            &ledger(vec![ledger_entry(&distinct_policy())]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("has no ABI ledger entry"));

        let mut tombstone = ledger_entry(&record);
        tombstone.status = "tombstone".into();
        let error = bind_generated_abi_ids(
            &mut overlay_file(vec![unbound.clone()]),
            &ledger(vec![tombstone]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("non-active ABI ledger entry"));

        let first = ledger_entry(&record);
        let mut duplicate = first.clone();
        duplicate.abi_id = "i9002".into();
        let error = bind_generated_abi_ids(
            &mut overlay_file(vec![unbound]),
            &ledger(vec![first, duplicate]),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("duplicate ABI ledger catalog ID")
        );
    }

    #[test]
    fn generated_abi_binding_checks_derived_operation_and_raw_signature() {
        let record = policy();
        let mut unbound = record.clone();
        unbound.abi_id.clear();

        let mut wrong_operation = ledger_entry(&record);
        wrong_operation.operation_key = "launch.block_index.x".into();
        let error = bind_generated_abi_ids(
            &mut overlay_file(vec![unbound.clone()]),
            &ledger(vec![wrong_operation]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("operation key mismatch"));

        let mut wrong_signature = ledger_entry(&record);
        wrong_signature
            .raw_rust_signature
            .arguments
            .push("u32".into());
        let error = bind_generated_abi_ids(
            &mut overlay_file(vec![unbound]),
            &ledger(vec![wrong_signature]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("raw Rust signature mismatch"));
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

    #[test]
    fn packed_alu_recipes_accept_only_the_reviewed_source_shape_and_floor() {
        let operations = [
            PackedAluOperation::Add,
            PackedAluOperation::Sub,
            PackedAluOperation::Mul,
            PackedAluOperation::Fma,
            PackedAluOperation::FmaRelu,
            PackedAluOperation::Min,
            PackedAluOperation::Max,
            PackedAluOperation::Neg,
            PackedAluOperation::Abs,
        ];
        for format in [PackedAluFormat::Bf16x2, PackedAluFormat::F16x2] {
            for operation in operations {
                let policy = packed_alu_policy(format, operation);
                match packed_alu_declaration(format, operation) {
                    Some(declaration) => validate_imported_policy(&policy, &declaration).unwrap(),
                    None => validate_ptx_native_policy(&policy).unwrap(),
                }
            }
        }

        let declaration =
            packed_alu_declaration(PackedAluFormat::Bf16x2, PackedAluOperation::Fma).unwrap();
        let reject_imported = |policy: &OverlayIntrinsic, message: &str| {
            let error = validate_imported_policy(policy, &declaration).unwrap_err();
            assert!(error.to_string().contains(message), "{error:#}");
        };

        let valid = packed_alu_policy(PackedAluFormat::Bf16x2, PackedAluOperation::Fma);
        let mut wrong_source = valid.clone();
        wrong_source.source_record = Some("int_nvvm_fma_rn_bf16".into());
        reject_imported(&wrong_source, "source");

        let mut wrong_format = valid.clone();
        wrong_format.packed_alu.as_mut().unwrap().format = PackedAluFormat::F16x2;
        reject_imported(&wrong_format, "identity");

        let mut wrong_operation = valid.clone();
        wrong_operation.packed_alu.as_mut().unwrap().operation = PackedAluOperation::Max;
        reject_imported(&wrong_operation, "identity");

        let mut wrong_floor = valid.clone();
        wrong_floor.minimum_sm = Some("sm_90".into());
        reject_imported(&wrong_floor, "target floor");

        let mut wrong_effects = valid.clone();
        wrong_effects.memory = "read".into();
        reject_imported(&wrong_effects, "effects");

        let mut wrong_section = valid.clone();
        wrong_section.ptx_isa_section = "9.7.4 Floating Point Instructions".into();
        reject_imported(&wrong_section, "PTX provenance");

        let mut wrong_url = valid.clone();
        wrong_url.ptx_isa_url =
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#floating-point-instructions"
                .into();
        reject_imported(&wrong_url, "PTX provenance");

        let mut wrong_adapter = valid.clone();
        wrong_adapter.lowering = "direct_nvvm".into();
        reject_imported(&wrong_adapter, "lowering recipe");

        let mut wrong_backend = valid;
        wrong_backend.backend_lowerings[0].mechanism = BackendLoweringMechanism::TypedNvvm;
        reject_imported(&wrong_backend, "inline-PTX routes");

        let mut wrong_native = packed_alu_policy(PackedAluFormat::Bf16x2, PackedAluOperation::Add);
        wrong_native.source = Some(IntrinsicSource::PtxNative {
            instruction: "add.bf16x2".into(),
        });
        let error = validate_ptx_native_policy(&wrong_native).unwrap_err();
        assert!(error.to_string().contains("PTX-native recipe"));

        let mut invented_llvm = packed_alu_policy(PackedAluFormat::F16x2, PackedAluOperation::Add);
        invented_llvm.llvm_symbol = Some("llvm.nvvm.add.rn.f16x2".into());
        let error = validate_ptx_native_policy(&invented_llvm).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("must not invent LLVM source facts")
        );

        let mut unreviewed_modifier =
            packed_alu_policy(PackedAluFormat::F16x2, PackedAluOperation::Add);
        unreviewed_modifier.expected_ptx.modifiers =
            vec!["rn".into(), "ftz".into(), "f16x2".into()];
        let error = validate_ptx_native_policy(&unreviewed_modifier).unwrap_err();
        assert!(error.to_string().contains("exact packed-ALU instruction"));

        let mut wrong_arity = packed_alu_policy(PackedAluFormat::F16x2, PackedAluOperation::Add);
        wrong_arity.expected_ptx.operands.pop();
        let error = validate_ptx_native_policy(&wrong_arity).unwrap_err();
        assert!(error.to_string().contains("exact packed-ALU instruction"));

        let f16_declaration =
            packed_alu_declaration(PackedAluFormat::F16x2, PackedAluOperation::Fma).unwrap();
        let reject_f16 = |policy: &OverlayIntrinsic, message: &str| {
            let error = validate_imported_policy(policy, &f16_declaration).unwrap_err();
            assert!(error.to_string().contains(message), "{error:#}");
        };
        let f16 = packed_alu_policy(PackedAluFormat::F16x2, PackedAluOperation::Fma);

        let mut wrong_signature = f16.clone();
        wrong_signature.llvm_arguments = vec!["v2bf16".into(); 3];
        reject_f16(&wrong_signature, "LLVM argument signature mismatch");

        let mut missing_must_use = f16.clone();
        missing_must_use.must_use = false;
        reject_f16(&missing_must_use, "reviewed safe packed-ALU API");

        let mut wrong_native_floor = f16.clone();
        wrong_native_floor
            .packed_alu
            .as_mut()
            .unwrap()
            .native_minimum_sm = 70;
        reject_f16(&wrong_native_floor, "target floor");

        let mut wrong_backend_floor = f16;
        wrong_backend_floor.backend_lowerings[0].minimum_ptx = Some("4.2".into());
        reject_f16(&wrong_backend_floor, "exact packed-ALU floor");

        let abs_declaration =
            packed_alu_declaration(PackedAluFormat::F16x2, PackedAluOperation::Abs).unwrap();
        let mut wrong_abs = packed_alu_policy(PackedAluFormat::F16x2, PackedAluOperation::Abs);
        wrong_abs.resolved_llvm_symbol = Some("llvm.nvvm.fabs.v2bf16".into());
        let error = validate_imported_policy(&wrong_abs, &abs_declaration).unwrap_err();
        assert!(error.to_string().contains("LLVM source or signature"));
    }

    #[test]
    fn pinned_packed_alu_records_match_the_closed_recipes() {
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
        let packed: Vec<_> = overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "packed_alu")
            .collect();
        assert_eq!(packed.len(), 18);
        for policy in packed {
            let source = resolve_policy_source(policy).unwrap();
            let declaration = policy
                .source_record
                .as_deref()
                .map(|record| declarations[record]);
            validate_policy(policy, &source, declaration, 1).unwrap();
        }
    }

    #[test]
    fn pinned_packed_conversion_records_match_the_closed_recipes() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let overlay = load_resolution_base(&repo_root).unwrap().overlay;
        let imported: ImportedFile =
            read_json(&repo_root.join("intrinsics/imported.json")).unwrap();
        let declarations: BTreeMap<_, _> = imported
            .intrinsics
            .iter()
            .map(|record| (record.source_record.as_str(), record))
            .collect();
        let packed: Vec<_> = overlay
            .intrinsics
            .iter()
            .filter(|record| record.family == "packed_conversion")
            .collect();
        assert_eq!(packed.len(), 10);
        for policy in packed {
            let source = resolve_policy_source(policy).unwrap();
            let declaration = policy
                .source_record
                .as_deref()
                .map(|record| declarations[record]);
            validate_policy(policy, &source, declaration, 1).unwrap();
        }
    }

    #[test]
    fn selectionless_packed_conversion_is_admitted_only_by_its_closed_recipe() {
        let cases = [
            (
                PackedConversionDestinationFormat::Bf16x2,
                PackedConversionRounding::NearestEven,
                PackedConversionSaturation::None,
            ),
            (
                PackedConversionDestinationFormat::F16x2,
                PackedConversionRounding::NearestEven,
                PackedConversionSaturation::None,
            ),
            (
                PackedConversionDestinationFormat::F16x2,
                PackedConversionRounding::TowardZero,
                PackedConversionSaturation::None,
            ),
            (
                PackedConversionDestinationFormat::F16x2,
                PackedConversionRounding::NearestEven,
                PackedConversionSaturation::Relu,
            ),
            (
                PackedConversionDestinationFormat::Bf16x2,
                PackedConversionRounding::NearestEven,
                PackedConversionSaturation::Relu,
            ),
            (
                PackedConversionDestinationFormat::Bf16x2,
                PackedConversionRounding::TowardZero,
                PackedConversionSaturation::None,
            ),
            (
                PackedConversionDestinationFormat::E4m3x2,
                PackedConversionRounding::NearestEven,
                PackedConversionSaturation::Satfinite,
            ),
            (
                PackedConversionDestinationFormat::E4m3x2,
                PackedConversionRounding::NearestEven,
                PackedConversionSaturation::SatfiniteRelu,
            ),
            (
                PackedConversionDestinationFormat::E5m2x2,
                PackedConversionRounding::NearestEven,
                PackedConversionSaturation::Satfinite,
            ),
            (
                PackedConversionDestinationFormat::E5m2x2,
                PackedConversionRounding::NearestEven,
                PackedConversionSaturation::SatfiniteRelu,
            ),
        ];
        for (destination, rounding, saturation) in cases {
            let policy = packed_conversion_policy(destination, rounding, saturation);
            let declaration = packed_conversion_declaration(&policy);
            validate_imported_policy(&policy, &declaration).unwrap();
        }

        let valid = packed_conversion_policy(
            PackedConversionDestinationFormat::Bf16x2,
            PackedConversionRounding::NearestEven,
            PackedConversionSaturation::None,
        );
        let declaration = packed_conversion_declaration(&valid);

        let reject = |policy: &OverlayIntrinsic, declaration: &ImportedIntrinsic, message: &str| {
            let error = validate_imported_policy(policy, declaration).unwrap_err();
            assert!(error.to_string().contains(message), "{error:#}");
        };

        let mut wrong_source = valid.clone();
        wrong_source.source_record = Some("int_nvvm_ff2bf16x2_rz".into());
        reject(&wrong_source, &declaration, "identity or LLVM source");

        let mut wrong_floor = valid.clone();
        wrong_floor.minimum_ptx = "7.8".into();
        reject(&wrong_floor, &declaration, "target floor");

        let mut wrong_effect = valid.clone();
        wrong_effect.pure = false;
        reject(&wrong_effect, &declaration, "effects");

        let mut wrong_section = valid.clone();
        wrong_section.ptx_isa_section = "9.7.9 Data Movement and Conversion Instructions".into();
        reject(&wrong_section, &declaration, "PTX provenance");

        let mut wrong_url = valid.clone();
        wrong_url.ptx_isa_url =
            "https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions"
                .into();
        reject(&wrong_url, &declaration, "PTX provenance");

        let mut wrong_shape = valid.clone();
        wrong_shape.expected_ptx.modifiers.swap(1, 2);
        reject(&wrong_shape, &declaration, "reversed high/low");

        let mut wrong_identity = valid.clone();
        wrong_identity.id = "cvt_f16x2_f32".into();
        reject(&wrong_identity, &declaration, "identity or LLVM source");

        let mut unsupported = valid.clone();
        let conversion = unsupported.packed_conversion.as_mut().unwrap();
        conversion.rounding = PackedConversionRounding::TowardZero;
        conversion.saturation = PackedConversionSaturation::Relu;
        reject(
            &unsupported,
            &declaration,
            "unsupported packed-conversion destination",
        );

        let mut wrong_compatibility = valid.clone();
        wrong_compatibility.compatibility_rust_paths =
            vec!["cuda_device::convert::cvt_f32x2_bf16x2".into()];
        reject(&wrong_compatibility, &declaration, "conversion API");

        let mut wrong_result = valid.clone();
        wrong_result.llvm_results = vec!["v2f16".into()];
        reject(&wrong_result, &declaration, "result signature mismatch");

        let mut selected = declaration.clone();
        selected.selections.push(ImportedSelection {
            source_record: "UNREVIEWED".into(),
            asm: "cvt.rn.bf16x2.f32 $d, $a, $b;".into(),
            predicates: vec![],
            constraints: Default::default(),
        });
        reject(&valid, &selected, "selectionless");
    }

    #[test]
    fn candidate_targets_are_canonical_and_satisfy_every_floor() {
        let requirement = CatalogTargetRequirement {
            minimum_ptx: "7.0".parse().unwrap(),
            hardware: CatalogHardwareTarget::AnyOf {
                alternatives: vec![CatalogHardwareAlternative::MinimumSm { sm: 80 }],
            },
        };
        validate_candidate_target(&requirement, "sm_80", "+ptx70", "test").unwrap();
        validate_candidate_target(&requirement, "sm_90a", "+ptx86", "test").unwrap();
        assert!(
            validate_candidate_target(&requirement, "sm_75", "+ptx70", "test")
                .unwrap_err()
                .to_string()
                .contains("hardware requirement")
        );
        assert!(
            validate_candidate_target(&requirement, "sm_80", "+ptx69", "test")
                .unwrap_err()
                .to_string()
                .contains("PTX floor")
        );
        for malformed in ["compute_80", "sm_080", "sm_80x"] {
            assert!(
                validate_candidate_target(&requirement, malformed, "+ptx70", "test").is_err(),
                "{malformed}"
            );
        }
        for malformed in ["ptx70", "+ptx7", "+ptx070"] {
            assert!(
                validate_candidate_target(&requirement, "sm_80", malformed, "test").is_err(),
                "{malformed}"
            );
        }

        let exact = CatalogTargetRequirement {
            minimum_ptx: "8.7".parse().unwrap(),
            hardware: CatalogHardwareTarget::AnyOf {
                alternatives: vec![CatalogHardwareAlternative::ExactArchitecture { sm: 120 }],
            },
        };
        validate_candidate_target(&exact, "sm_120a", "+ptx87", "test").unwrap();
        assert!(validate_candidate_target(&exact, "sm_120a", "+ptx86", "test").is_err());
        assert!(validate_candidate_target(&exact, "sm_120", "+ptx87", "test").is_err());
        assert!(validate_candidate_target(&exact, "sm_120f", "+ptx87", "test").is_err());

        let family = CatalogTargetRequirement {
            minimum_ptx: "8.7".parse().unwrap(),
            hardware: CatalogHardwareTarget::AnyOf {
                alternatives: vec![CatalogHardwareAlternative::FamilyTarget { sm: 120 }],
            },
        };
        validate_candidate_target(&family, "sm_120f", "+ptx87", "test").unwrap();
        assert!(validate_candidate_target(&family, "sm_120a", "+ptx87", "test").is_err());
    }

    struct CandidateTestRepo(PathBuf);

    impl Drop for CandidateTestRepo {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn repo_without_evidence() -> CandidateTestRepo {
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT: AtomicU64 = AtomicU64::new(0);
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let root = std::env::temp_dir().join(format!(
            "cuda-intrinsics-candidate-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let input = root.join("intrinsics");
        fs::create_dir_all(input.join("overlay")).unwrap();
        for name in [
            "upstream.lock",
            "imported.json",
            "overlay.toml",
            "abi-v1.toml",
        ] {
            fs::copy(source.join("intrinsics").join(name), input.join(name)).unwrap();
        }
        for entry in fs::read_dir(source.join("intrinsics/overlay")).unwrap() {
            let entry = entry.unwrap();
            if entry.path().extension().and_then(|value| value.to_str()) == Some("toml") {
                fs::copy(entry.path(), input.join("overlay").join(entry.file_name())).unwrap();
            }
        }
        CandidateTestRepo(root)
    }

    #[test]
    fn candidate_resolution_is_the_only_path_that_can_omit_evidence() {
        let repo = repo_without_evidence();
        let candidate = resolve_candidate(
            &repo.0,
            "thread_idx_x",
            "LLVM version candidate",
            &"a".repeat(64),
            "sm_80",
            "+ptx70",
        )
        .unwrap();
        assert_eq!(candidate.catalog.intrinsics.len(), 1);
        assert_eq!(candidate.catalog.intrinsics[0].id, "thread_idx_x");
        assert_eq!(candidate.catalog.intrinsics[0].backend.status, "candidate");
        assert!(candidate.catalog.inputs.evidence_sha256.is_empty());

        let error = resolve(&repo.0).unwrap_err();
        assert!(
            error.to_string().contains("intrinsics/evidence"),
            "{error:#}"
        );
        let error = resolve_candidate(
            &repo.0,
            "not_an_intrinsic",
            "LLVM version candidate",
            &"a".repeat(64),
            "sm_80",
            "+ptx70",
        )
        .unwrap_err();
        assert!(error.to_string().contains("unknown overlay intrinsic"));
    }

    #[test]
    fn candidate_resolution_cannot_change_normal_catalog_bytes() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let before = crate::util::pretty_json(&resolve(&repo_root).unwrap()).unwrap();
        resolve_candidate(
            &repo_root,
            "thread_idx_x",
            "LLVM version candidate",
            &"a".repeat(64),
            "sm_80",
            "+ptx70",
        )
        .unwrap();
        let after = crate::util::pretty_json(&resolve(&repo_root).unwrap()).unwrap();
        assert_eq!(before.as_bytes(), after.as_bytes());
    }
}
