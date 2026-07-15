/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::ptx::InstructionPattern;
use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpstreamLock {
    pub schema: u32,
    pub llvm: LockedLlvm,
    pub llvm_tblgen: LockedTool,
    #[serde(default)]
    pub comparison_tools: Vec<LockedTool>,
    pub dumps: LockedDumps,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LockedLlvm {
    pub repository: String,
    pub revision: String,
    pub provenance: String,
    pub public_output_allowed: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LockedTool {
    pub name: String,
    pub version_line: String,
    pub sha256: String,
    #[serde(default)]
    pub enforce_sha256: bool,
    pub provenance: String,
    pub built_from_llvm_revision: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LockedDumps {
    pub intrinsics_sha256: String,
    pub nvptx_sha256: String,
    pub normalized_imported_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedFile {
    pub schema: u32,
    pub source: ImportedSource,
    pub intrinsics: Vec<ImportedIntrinsic>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedSource {
    pub llvm_repository: String,
    pub llvm_revision: String,
    pub llvm_tblgen_version: String,
    pub llvm_tblgen_source_revision: String,
    pub intrinsics_json_sha256: String,
    pub nvptx_json_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedIntrinsic {
    pub source_record: String,
    pub llvm_name: String,
    pub arguments: Vec<String>,
    pub results: Vec<String>,
    pub classes: Vec<String>,
    pub properties: Vec<String>,
    pub selections: Vec<ImportedSelection>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedSelection {
    pub source_record: String,
    pub asm: String,
    pub predicates: Vec<String>,
    #[serde(
        default,
        skip_serializing_if = "ImportedSelectionConstraints::is_empty"
    )]
    pub constraints: ImportedSelectionConstraints,
}

/// Normalized constraints attached to an NVPTX instruction-selection record.
///
/// TableGen represents address-space-specific patterns through anonymous
/// `PatFrag` records and can bind intrinsic arguments to integer literals.
/// Keeping those facts separate from the assembly spelling lets policy select
/// an exact lowering without parsing PTX text.
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedSelectionConstraints {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub address_space: Option<ImportedAddressSpace>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub immediate_bindings: Vec<ImportedImmediateBinding>,
}

/// One integer literal fixed by an NVPTX instruction-selection pattern.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportedImmediateBinding {
    pub argument_index: usize,
    pub value: i64,
}

impl ImportedSelectionConstraints {
    pub fn is_empty(&self) -> bool {
        self.address_space.is_none() && self.immediate_bindings.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImportedAddressSpace {
    Generic,
    Shared,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlayFile {
    pub schema: u32,
    pub catalog_version: String,
    pub intrinsic_abi: u32,
    pub backend_profile: String,
    #[serde(default)]
    pub shards: Vec<String>,
    #[serde(rename = "intrinsic")]
    #[serde(default)]
    pub intrinsics: Vec<OverlayIntrinsic>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlayShardFile {
    pub schema: u32,
    pub family: String,
    #[serde(rename = "intrinsic")]
    pub intrinsics: Vec<OverlayIntrinsic>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlayIntrinsic {
    pub id: String,
    pub abi_id: String,
    pub operation_key: String,
    pub family: String,
    /// Imported LLVM records use the legacy `source_record` field below.
    /// PTX-native records must instead carry an explicit tagged source.
    #[serde(default)]
    pub source: Option<IntrinsicSource>,
    #[serde(default)]
    pub source_record: Option<String>,
    pub rust_module: String,
    pub rust_name: String,
    #[serde(default)]
    pub rust_arguments: Vec<String>,
    pub rust_result: String,
    pub safe: bool,
    #[serde(default)]
    pub must_use: bool,
    pub safe_allowlist_reason: Option<String>,
    pub public_rust_path: String,
    #[serde(default)]
    pub compatibility_rust_paths: Vec<String>,
    pub dialect_op_type: String,
    pub dialect_op_name: String,
    #[serde(default)]
    pub dialect_operands: Vec<String>,
    #[serde(default)]
    pub dialect_results: Vec<String>,
    #[serde(default)]
    pub llvm_symbol: Option<String>,
    #[serde(default)]
    pub resolved_llvm_symbol: Option<String>,
    #[serde(default)]
    pub llvm_arguments: Vec<String>,
    #[serde(default)]
    pub llvm_results: Vec<String>,
    pub pure: bool,
    pub memory: String,
    pub convergent: bool,
    pub execution_scope: String,
    pub minimum_ptx: String,
    #[serde(default)]
    pub minimum_sm: Option<String>,
    pub ptx_result: String,
    pub targets: String,
    pub ptx_isa_version: String,
    pub ptx_isa_section: String,
    pub ptx_isa_url: String,
    pub lowering: String,
    #[serde(default)]
    pub backend_lowerings: Vec<OverlayBackendLowering>,
    #[serde(default)]
    pub packed_atomic: Option<PackedAtomic>,
    #[serde(default)]
    pub redux: Option<Redux>,
    #[serde(default)]
    pub vote: Option<Vote>,
    #[serde(default)]
    pub active_mask: Option<ActiveMask>,
    #[serde(default)]
    pub warp_match: Option<WarpMatch>,
    #[serde(default)]
    pub warp_barrier: Option<WarpBarrier>,
    #[serde(default)]
    pub warp_shuffle: Option<WarpShuffle>,
    #[serde(default)]
    pub dot_product: Option<DotProduct>,
    #[serde(default)]
    pub packed_alu: Option<PackedAlu>,
    #[serde(default)]
    pub packed_conversion: Option<PackedConversion>,
    #[serde(default)]
    pub cp_async_copy: Option<CpAsyncCopy>,
    #[serde(default)]
    pub cp_async_control: Option<CpAsyncControl>,
    #[serde(default)]
    pub cp_async_mbarrier: Option<CpAsyncMbarrier>,
    #[serde(default)]
    pub mbarrier_basic: Option<MbarrierBasic>,
    #[serde(default)]
    pub ldmatrix_variant: Option<LdmatrixVariant>,
    #[serde(default)]
    pub ldmatrix_safety: Option<LdmatrixSafety>,
    #[serde(default)]
    pub ldmatrix_adapter: Option<LdmatrixAdapter>,
    #[serde(default)]
    pub selected_address_space: Option<ImportedAddressSpace>,
    pub expected_ptx: InstructionPattern,
    pub summary: String,
}

/// Backend-specific lowering selected by reviewed evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlayBackendLowering {
    pub backend: IntrinsicBackend,
    pub mechanism: BackendLoweringMechanism,
    pub evidence_profile: String,
    /// Optional backend-profile floor. When absent, the intrinsic's native
    /// target requirement is used.
    #[serde(default)]
    pub minimum_ptx: Option<String>,
    #[serde(default)]
    pub minimum_sm: Option<String>,
}

/// Provenance for a generated intrinsic. PTX-native operations deliberately
/// have no invented LLVM TableGen record or LLVM intrinsic symbol.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum IntrinsicSource {
    LlvmImported { source_record: String },
    PtxNative { instruction: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntrinsicBackend {
    LlvmNvptx,
    LibNvvm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendLoweringMechanism {
    TypedNvvm,
    InlinePtx,
}

/// Closed semantic identity for the generated `ldmatrix` family.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LdmatrixVariant {
    pub shape: LdmatrixShape,
    pub multiplicity: LdmatrixMultiplicity,
    pub layout: LdmatrixLayout,
    pub element: LdmatrixElement,
    pub state_space: LdmatrixStateSpace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LdmatrixShape {
    M8n8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LdmatrixMultiplicity {
    X1,
    X2,
    X4,
}

impl LdmatrixMultiplicity {
    pub const fn register_count(self) -> usize {
        match self {
            Self::X1 => 1,
            Self::X2 => 2,
            Self::X4 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LdmatrixLayout {
    Normal,
    Transposed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LdmatrixElement {
    B16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LdmatrixStateSpace {
    Shared,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LdmatrixSafety {
    pub participation: LdmatrixParticipation,
    pub address_contract: LdmatrixAddressContract,
    pub memory_order: LdmatrixMemoryOrder,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LdmatrixParticipation {
    AllWarpLanesSameInstruction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LdmatrixAddressContract {
    WarpLaneAddressesMappedByMultiplicitySixteenByteAlignedSixteenBytesReadableWithSm75Replication,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LdmatrixMemoryOrder {
    Weak,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeValidation {
    Unexecuted,
    Executed,
}

/// Closed semantic contract for the generated packed global atomic-add
/// family. These fields are intentionally enums rather than free-form strings:
/// accepting an unreviewed state space, scope, or floating-point mode must
/// require a generator change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedAtomic {
    pub format: PackedAtomicFormat,
    /// PTX ISA hardware floor, kept separate from cuda-oxide's admitted floor
    /// and from backend-profile floors.
    pub native_minimum_sm: u16,
    pub operation: PackedAtomicOperation,
    pub state_space: PackedAtomicStateSpace,
    pub ordering: PackedAtomicOrdering,
    pub scope: PackedAtomicScope,
    pub rounding: PackedAtomicRounding,
    pub subnormal: PackedAtomicSubnormal,
    pub atomicity: PackedAtomicAtomicity,
    pub pointer_contract: PackedAtomicPointerContract,
    pub access_contract: PackedAtomicAccessContract,
    pub scope_contract: PackedAtomicScopeContract,
    pub codegen_contract: PackedAtomicCodegenContract,
    pub return_contract: PackedAtomicReturnContract,
    pub adapter: PackedAtomicAdapter,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicFormat {
    F16x2,
    Bf16x2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicOperation {
    Add,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicStateSpace {
    Global,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicOrdering {
    Relaxed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicScope {
    Gpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicRounding {
    NearestEven,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicSubnormal {
    Preserve,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicAtomicity {
    PerElement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicPointerContract {
    MutableGlobalU32Aligned4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicAccessContract {
    NoMixedWholeWordOrNonAtomicAccess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicScopeContract {
    RacingAtomicsMutuallyInclusive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicCodegenContract {
    ExactNativeInstruction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicReturnContract {
    OldValuesPerElementMayBeNoncoherent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAtomicAdapter {
    OldPackedU32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LdmatrixAdapter {
    SingleResultDirect,
    MultipleResultsToArray,
}

/// Closed semantic and lowering contract for the generated integer
/// `redux.sync` family.
///
/// The Rust and NVVM dialect APIs intentionally put the participation mask
/// first, while LLVM's NVVM intrinsic puts the lane value first. Keeping that
/// adapter typed prevents a generic direct-call renderer from silently
/// swapping the collective's source and member mask.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Redux {
    pub operation: ReduxOperation,
    pub participation: ReduxParticipation,
    pub adapter: ReduxAdapter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReduxOperation {
    Add,
    Umin,
    Min,
    Umax,
    Max,
    And,
    Or,
    Xor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReduxParticipation {
    ExecutingLaneNamedAllNamedLanesSameInstructionAndMask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReduxAdapter {
    MaskValueToSourceMemberMask,
}

/// Closed semantic and lowering contract for the generated `vote.sync`
/// family.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vote {
    pub mode: VoteMode,
    pub participation: VoteParticipation,
    pub legacy_pre_sm70: PreSm70MemberMaskRule,
    pub adapter: VoteAdapter,
    pub mask_encoding: MaskEncoding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoteMode {
    All,
    Any,
    Ballot,
    Uni,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoteParticipation {
    ExecutingLaneNamedAllNamedLanesSameInstructionAndMask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoteAdapter {
    DirectMaskPredicate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskEncoding {
    RegisterOrImmediate,
}

/// Closed semantic and lowering contract for `activemask`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActiveMask {
    pub observation: ActiveMaskObservation,
    pub adapter: ActiveMaskAdapter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActiveMaskObservation {
    ExecutingLanesAtInstruction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActiveMaskAdapter {
    DirectZeroOperandMask,
}

/// Closed semantic and lowering contract for `match.sync`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WarpMatch {
    pub mode: WarpMatchMode,
    pub value_width: WarpMatchValueWidth,
    pub participation: WarpMatchParticipation,
    pub adapter: WarpMatchAdapter,
    pub value_encoding: MatchOperandEncoding,
    pub mask_encoding: MatchOperandEncoding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpMatchMode {
    Any,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpMatchValueWidth {
    B32,
    B64,
}

impl WarpMatchValueWidth {
    pub const fn bits(self) -> u32 {
        match self {
            Self::B32 => 32,
            Self::B64 => 64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpMatchParticipation {
    ExecutingLaneNamedAllNamedLanesSameInstructionAndMask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpMatchAdapter {
    DirectMask,
    ProjectMaskDiscardPredicate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchOperandEncoding {
    RegisterOrImmediate,
}

/// Closed semantic and lowering contract for `bar.warp.sync`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WarpBarrier {
    pub participation: WarpBarrierParticipation,
    pub legacy_pre_sm70: PreSm70MemberMaskRule,
    pub adapter: WarpBarrierAdapter,
    pub mask_encoding: WarpBarrierMaskEncoding,
    pub memory_ordering: WarpBarrierMemoryOrdering,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpBarrierParticipation {
    ExecutingLaneNamedAllNamedLanesSameInstructionAndMask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PreSm70MemberMaskRule {
    AllNamedLanesConvergedAndOnlyNamedLanesActive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpBarrierAdapter {
    DirectMemberMask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpBarrierMaskEncoding {
    RegisterOrImmediate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpBarrierMemoryOrdering {
    ParticipatingLanes,
}

/// Closed semantic and lowering contract for `shfl.sync`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WarpShuffle {
    pub mode: WarpShuffleMode,
    pub value_kind: WarpShuffleValueKind,
    pub participation: WarpShuffleParticipation,
    pub legacy_pre_sm70: PreSm70MemberMaskRule,
    pub source_lane: WarpShuffleSourceLane,
    pub adapter: WarpShuffleAdapter,
    pub clamp: u32,
    pub lane_encoding: WarpShuffleOperandEncoding,
    pub mask_encoding: WarpShuffleOperandEncoding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpShuffleMode {
    Idx,
    Bfly,
    Down,
    Up,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpShuffleValueKind {
    I32,
    F32,
    I64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpShuffleParticipation {
    ExecutingLaneNamedAllNamedLanesSameInstructionAndMask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpShuffleSourceLane {
    InRangeSourceActiveAndNamedOutOfRangeCopiesSelf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpShuffleAdapter {
    MaskValueLaneOrDeltaInsertClamp,
    /// Split i64 into low/high b32 halves, shuffle both in one convergent
    /// side-effecting block, then reassemble the original bit layout.
    MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarpShuffleOperandEncoding {
    RegisterOrImmediate,
    RegisterOnly,
}

/// Closed identity and source adapter for generated packed integer dot products.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DotProduct {
    pub operation: DotProductOperation,
    pub signedness: DotProductSignedness,
    pub adapter: DotProductAdapter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DotProductOperation {
    Dp2a,
    Dp4a,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DotProductSignedness {
    Signed,
    Unsigned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DotProductAdapter {
    DirectThreeOperands,
    InsertLowHalfFalse,
}

/// Closed identity and carrier contract for packed floating-point ALU ops.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedAlu {
    pub format: PackedAluFormat,
    /// Hardware floor of the native PTX instruction, independent of the
    /// target floor admitted by cuda-oxide.
    pub native_minimum_sm: u16,
    pub operation: PackedAluOperation,
    pub adapter: PackedAluAdapter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAluFormat {
    Bf16x2,
    F16x2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAluOperation {
    Add,
    Sub,
    Mul,
    Fma,
    FmaRelu,
    Min,
    Max,
    Neg,
    Abs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedAluAdapter {
    DirectPackedU32,
}

/// Closed contract for converting two scalar values into one packed value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedConversion {
    pub source_format: PackedConversionSourceFormat,
    pub destination_format: PackedConversionDestinationFormat,
    pub rounding: PackedConversionRounding,
    pub saturation: PackedConversionSaturation,
    pub adapter: PackedConversionAdapter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedConversionSourceFormat {
    F32x2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedConversionDestinationFormat {
    Bf16x2,
    F16x2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedConversionRounding {
    NearestEven,
    TowardZero,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedConversionSaturation {
    None,
    Relu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedConversionAdapter {
    ReverseHighLowOperands,
}

/// Closed contract for classic global-to-shared `cp.async` copies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpAsyncCopy {
    pub cache_policy: CpAsyncCachePolicy,
    pub copy_size: CpAsyncCopySize,
    pub source_size: CpAsyncSourceSize,
    pub adapter: CpAsyncAdapter,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpAsyncCachePolicy {
    Ca,
    Cg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpAsyncCopySize {
    B4,
    B8,
    B16,
}

impl CpAsyncCopySize {
    pub const fn bytes(self) -> u32 {
        match self {
            Self::B4 => 4,
            Self::B8 => 8,
            Self::B16 => 16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpAsyncSourceSize {
    Full,
    Runtime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpAsyncAdapter {
    DirectPointers,
    DirectPointersAndSourceSize,
}

/// Closed contract for classic `cp.async` group controls.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpAsyncControl {
    pub operation: CpAsyncControlOperation,
    pub adapter: CpAsyncControlAdapter,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpAsyncControlOperation {
    CommitGroup,
    WaitAll,
    WaitGroup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpAsyncControlAdapter {
    NoOperands,
    CompileTimeConstantMaxPending,
}

/// Closed contract for associating classic `cp.async` completion with an mbarrier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpAsyncMbarrier {
    pub operation: CpAsyncMbarrierOperation,
    pub state_space: CpAsyncMbarrierStateSpace,
    pub adapter: CpAsyncMbarrierAdapter,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpAsyncMbarrierOperation {
    Arrive,
    ArriveNoInc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpAsyncMbarrierStateSpace {
    Generic,
    Shared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpAsyncMbarrierAdapter {
    PointerToVoid,
}

/// Closed contract for the basic shared-memory mbarrier lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MbarrierBasic {
    pub operation: MbarrierBasicOperation,
    pub state_space: MbarrierStateSpace,
    pub adapter: MbarrierBasicAdapter,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MbarrierBasicOperation {
    Init,
    Arrive,
    TestWait,
    Inval,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MbarrierStateSpace {
    Shared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MbarrierBasicAdapter {
    PointerCountToVoid,
    PointerToToken,
    PointerTokenToPredicate,
    PointerToVoid,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbiLedgerFile {
    pub schema: u32,
    pub intrinsic_abi: u32,
    #[serde(rename = "entry")]
    pub entries: Vec<AbiLedgerEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbiLedgerEntry {
    pub abi_id: String,
    pub status: String,
    pub catalog_id: String,
    pub operation_key: String,
    pub raw_rust_signature: AbiRawRustSignature,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbiRawRustSignature {
    pub safe: bool,
    pub arguments: Vec<String>,
    pub result: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceFile {
    pub schema: u32,
    pub backend_profile: String,
    #[serde(default)]
    pub backend_kind: Option<IntrinsicBackend>,
    pub llvm_revision: String,
    pub backend_version: String,
    pub backend_sha256: String,
    #[serde(default)]
    pub artifact_path: Option<String>,
    #[serde(default)]
    pub build_id_prefix: Option<String>,
    #[serde(default)]
    pub nvvm_ir_version: Option<String>,
    #[serde(default)]
    pub debug_ir_version: Option<String>,
    pub records: Vec<EvidenceRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRecord {
    pub id: String,
    #[serde(default)]
    pub source: Option<IntrinsicSource>,
    #[serde(default)]
    pub source_record: Option<String>,
    #[serde(default)]
    pub llvm_symbol: Option<String>,
    #[serde(default)]
    pub resolved_llvm_symbol: Option<String>,
    #[serde(default)]
    pub llvm_arguments: Vec<String>,
    #[serde(default)]
    pub llvm_results: Vec<String>,
    #[serde(default)]
    pub concrete_llvm_arguments: Vec<String>,
    #[serde(default)]
    pub concrete_llvm_results: Vec<String>,
    pub target_triple: String,
    pub gpu_target: String,
    pub ptx_feature: String,
    pub status: String,
    #[serde(default)]
    pub stages: Vec<EvidenceStage>,
    #[serde(default)]
    pub declaration_attributes_canonicalized: Option<bool>,
    #[serde(default)]
    pub runtime_validation: Option<RuntimeValidation>,
    pub expected_ptx: InstructionPattern,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceStage {
    pub targets: Vec<String>,
    pub representation: String,
    pub stage: EvidenceStageKind,
    #[serde(default)]
    pub mechanism: Option<BackendLoweringMechanism>,
    pub outcome: String,
    pub detail: String,
    #[serde(default)]
    pub artifact_kind: Option<EvidenceArtifactKind>,
    #[serde(default)]
    pub tool_path: Option<String>,
    #[serde(default)]
    pub tool_version: Option<String>,
    #[serde(default)]
    pub tool_sha256: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceArtifactKind {
    Cubin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStageKind {
    DeclarationCanonicalization,
    BackendCodegen,
    DeviceLink,
    PtxAssembly,
    Runtime,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogFile {
    pub schema: u32,
    pub catalog_version: String,
    pub intrinsic_abi: u32,
    pub generator_version: String,
    pub source: CatalogSource,
    pub inputs: CatalogInputs,
    pub intrinsics: Vec<CatalogIntrinsic>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogSource {
    pub llvm_repository: String,
    pub llvm_revision: String,
    pub llvm_tblgen_version: String,
    pub llvm_tblgen_source_revision: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogInputs {
    pub imported_sha256: String,
    pub overlay_sha256: String,
    pub abi_ledger_sha256: String,
    pub evidence_sha256: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogIntrinsic {
    pub id: String,
    pub operation_key: String,
    pub family: String,
    pub source: IntrinsicSource,
    pub selections: Vec<CatalogSelection>,
    pub rust: CatalogRust,
    pub dialect: CatalogDialect,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llvm: Option<CatalogLlvm>,
    pub semantics: CatalogSemantics,
    pub target: CatalogTarget,
    pub backend: CatalogBackend,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub backend_lowerings: Vec<CatalogBackendLowering>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed_atomic: Option<PackedAtomic>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub redux: Option<Redux>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vote: Option<Vote>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_mask: Option<ActiveMask>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warp_match: Option<WarpMatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warp_barrier: Option<WarpBarrier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warp_shuffle: Option<WarpShuffle>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dot_product: Option<DotProduct>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed_alu: Option<PackedAlu>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed_conversion: Option<PackedConversion>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cp_async_copy: Option<CpAsyncCopy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cp_async_control: Option<CpAsyncControl>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cp_async_mbarrier: Option<CpAsyncMbarrier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mbarrier_basic: Option<MbarrierBasic>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ldmatrix: Option<CatalogLdmatrix>,
    pub lowering: String,
    pub expected_ptx: InstructionPattern,
    pub summary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogSelection {
    pub source_record: String,
    pub asm: String,
    pub predicates: Vec<String>,
    #[serde(
        default,
        skip_serializing_if = "ImportedSelectionConstraints::is_empty"
    )]
    pub constraints: ImportedSelectionConstraints,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogRust {
    pub abi_id: String,
    pub module: String,
    pub name: String,
    pub arguments: Vec<String>,
    pub result: String,
    pub safe: bool,
    pub must_use: bool,
    pub safe_allowlist_reason: Option<String>,
    pub canonical_path: String,
    pub public_path: String,
    pub compatibility_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogDialect {
    pub op_type: String,
    pub op_name: String,
    pub operands: Vec<String>,
    pub results: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogLlvm {
    pub symbol: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_symbol: Option<String>,
    pub arguments: Vec<String>,
    pub results: Vec<String>,
    pub properties: Vec<String>,
    pub result_facts: CatalogLlvmResultFacts,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogLlvmResultFacts {
    pub no_undef: bool,
    pub range: Option<CatalogHalfOpenRange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogHalfOpenRange {
    pub lower: String,
    pub upper_exclusive: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogSemantics {
    pub pure: bool,
    pub memory: String,
    pub convergent: bool,
    pub execution_scope: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogTarget {
    pub minimum_ptx: PtxVersion,
    pub hardware: CatalogHardwareTarget,
    pub ptx_result: String,
    pub targets: String,
    pub ptx_isa_version: String,
    pub ptx_isa_section: String,
    pub ptx_isa_url: String,
}

/// A PTX ISA version encoded as `major * 10 + minor`.
///
/// PTX currently uses one decimal minor digit. The resolver validates that
/// shape before constructing this value, so generated consumers compare a
/// number rather than reparsing policy text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PtxVersion(u16);

impl PtxVersion {
    pub const fn encoded(self) -> u16 {
        self.0
    }

    pub const fn major(self) -> u16 {
        self.0 / 10
    }

    pub const fn minor(self) -> u16 {
        self.0 % 10
    }
}

impl FromStr for PtxVersion {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (major, minor) = value
            .split_once('.')
            .ok_or_else(|| "expected major.minor".to_owned())?;
        if major.is_empty()
            || !major.bytes().all(|byte| byte.is_ascii_digit())
            || minor.len() != 1
            || !minor.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err("expected numeric major.minor with one minor digit".to_owned());
        }
        let major: u16 = major.parse().map_err(|_| "major version is too large")?;
        let minor: u16 = minor.parse().unwrap();
        if format!("{major}.{minor}") != value {
            return Err("version is not in canonical major.minor form".to_owned());
        }
        let encoded = major
            .checked_mul(10)
            .and_then(|value| value.checked_add(minor))
            .ok_or_else(|| "version is too large".to_owned())?;
        Ok(Self(encoded))
    }
}

impl Serialize for PtxVersion {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for PtxVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for PtxVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}.{}", self.major(), self.minor())
    }
}

/// Reviewed hardware availability for an intrinsic.
///
/// The current overlay accepts `All` and monotonic `MinimumSm` requirements.
/// The explicit suffix variants reserve a typed representation for PTX `a`
/// architecture sets and `f` family sets without incorrectly reducing either
/// to a monotonic minimum.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CatalogHardwareTarget {
    All,
    AnyOf {
        alternatives: Vec<CatalogHardwareAlternative>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CatalogHardwareAlternative {
    MinimumSm { sm: u16 },
    ExactArchitecture { sm: u16 },
    FamilyTarget { sm: u16 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogBackend {
    pub profile: String,
    pub version: String,
    pub sha256: String,
    pub status: String,
    pub target_triple: String,
    pub gpu_target: String,
    pub ptx_feature: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogBackendLowering {
    pub backend: IntrinsicBackend,
    pub mechanism: BackendLoweringMechanism,
    pub evidence_profile: String,
    pub target: CatalogTargetRequirement,
    pub version: String,
    pub sha256: String,
    pub artifact_path: Option<String>,
    pub build_id_prefix: Option<String>,
    pub status: String,
    pub stages: Vec<EvidenceStage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogTargetRequirement {
    pub minimum_ptx: PtxVersion,
    pub hardware: CatalogHardwareTarget,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogLdmatrix {
    pub variant: LdmatrixVariant,
    pub safety: LdmatrixSafety,
    pub adapter: LdmatrixAdapter,
    pub selected_address_space: ImportedAddressSpace,
}

impl CatalogIntrinsic {
    pub fn scalar_width(&self) -> Option<u32> {
        match self.rust.result.as_str() {
            "u32" => Some(32),
            "u64" => Some(64),
            _ => None,
        }
    }

    pub fn llvm_identifier(&self) -> String {
        self.llvm
            .as_ref()
            .expect("LLVM-backed intrinsic")
            .symbol
            .replace('.', "_")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locked_tool_rejects_misspelled_security_field() {
        let input = r#"
name = "llvm-tblgen"
version_line = "LLVM version test"
sha256 = "abc"
enforce_sha25 = true
provenance = "test"
"#;
        let error = toml::from_str::<LockedTool>(input).unwrap_err();
        assert!(error.to_string().contains("enforce_sha25"));
    }

    #[test]
    fn imported_selection_rejects_misspelled_constraint() {
        let input = r#"{
            "source_record": "selection",
            "asm": "op;",
            "predicates": [],
            "constraints": { "adress_space": "shared" }
        }"#;
        let error = serde_json::from_str::<ImportedSelection>(input).unwrap_err();
        assert!(error.to_string().contains("adress_space"));
    }

    #[test]
    fn imported_selection_preserves_immediate_binding() {
        let input = r#"{
            "source_record": "DOT2_lo_ss",
            "asm": "dp2a.lo.s32.s32 $dst, $a, $b, $c;",
            "predicates": ["hasDotInstructions"],
            "constraints": {
                "immediate_bindings": [
                    { "argument_index": 2, "value": 0 }
                ]
            }
        }"#;
        let selection = serde_json::from_str::<ImportedSelection>(input).unwrap();
        assert_eq!(
            selection.constraints.immediate_bindings,
            [ImportedImmediateBinding {
                argument_index: 2,
                value: 0,
            }]
        );
        assert!(!selection.constraints.is_empty());
    }

    #[test]
    fn imported_immediate_binding_rejects_misspelled_index() {
        let input = r#"{
            "source_record": "DOT2_lo_ss",
            "asm": "dp2a.lo.s32.s32 $dst, $a, $b, $c;",
            "predicates": [],
            "constraints": {
                "immediate_bindings": [
                    { "argument_indx": 2, "value": 0 }
                ]
            }
        }"#;
        let error = serde_json::from_str::<ImportedSelection>(input).unwrap_err();
        assert!(error.to_string().contains("argument_indx"));
    }

    #[test]
    fn redux_contract_rejects_unknown_operand_adapter() {
        let input = r#"
operation = "add"
participation = "executing_lane_named_all_named_lanes_same_instruction_and_mask"
adapter = "mask_value_direct"
"#;
        let error = toml::from_str::<Redux>(input).unwrap_err();
        assert!(error.to_string().contains("mask_value_direct"));
    }

    #[test]
    fn packed_alu_contract_rejects_unknown_format_operation_and_adapter() {
        let valid = r#"
format = "bf16x2"
native_minimum_sm = 80
operation = "fma"
adapter = "direct_packed_u32"
"#;
        toml::from_str::<PackedAlu>(valid).unwrap();
        for invalid in [
            valid.replace("format = \"bf16x2\"", "format = \"bf16\""),
            valid.replace("native_minimum_sm = 80\n", ""),
            valid.replace("native_minimum_sm = 80", "native_minimum_sm = \"80\""),
            valid.replace("operation = \"fma\"", "operation = \"mad\""),
            valid.replace(
                "adapter = \"direct_packed_u32\"",
                "adapter = \"bitcast_any\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(toml::from_str::<PackedAlu>(&invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn packed_conversion_contract_rejects_open_ended_policy() {
        let valid = r#"
source_format = "f32x2"
destination_format = "bf16x2"
rounding = "nearest_even"
saturation = "none"
adapter = "reverse_high_low_operands"
"#;
        toml::from_str::<PackedConversion>(valid).unwrap();
        for invalid in [
            valid.replace("source_format = \"f32x2\"", "source_format = \"f16x2\""),
            valid.replace(
                "destination_format = \"bf16x2\"",
                "destination_format = \"f8x2\"",
            ),
            valid.replace("rounding = \"nearest_even\"", "rounding = \"zero\""),
            valid.replace("saturation = \"none\"", "saturation = \"finite\""),
            valid.replace(
                "adapter = \"reverse_high_low_operands\"",
                "adapter = \"direct\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(
                toml::from_str::<PackedConversion>(&invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn vote_contract_rejects_unknown_modes_and_mask_encodings() {
        let valid = r#"
mode = "all"
participation = "executing_lane_named_all_named_lanes_same_instruction_and_mask"
legacy_pre_sm70 = "all_named_lanes_converged_and_only_named_lanes_active"
adapter = "direct_mask_predicate"
mask_encoding = "register_or_immediate"
"#;
        toml::from_str::<Vote>(valid).unwrap();

        for invalid in [
            valid.replace("mode = \"all\"", "mode = \"match\""),
            valid.replace(
                "mask_encoding = \"register_or_immediate\"",
                "mask_encoding = \"any_operand\"",
            ),
            valid.replace(
                "legacy_pre_sm70 = \"all_named_lanes_converged_and_only_named_lanes_active\"",
                "legacy_pre_sm70 = \"independent_threads\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(toml::from_str::<Vote>(&invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn warp_shuffle_contract_rejects_open_ended_policy() {
        let valid = r#"
mode = "idx"
value_kind = "i32"
participation = "executing_lane_named_all_named_lanes_same_instruction_and_mask"
legacy_pre_sm70 = "all_named_lanes_converged_and_only_named_lanes_active"
source_lane = "in_range_source_active_and_named_out_of_range_copies_self"
adapter = "mask_value_lane_or_delta_insert_clamp"
clamp = 31
lane_encoding = "register_or_immediate"
mask_encoding = "register_or_immediate"
"#;
        toml::from_str::<WarpShuffle>(valid).unwrap();

        for invalid in [
            valid.replace("mode = \"idx\"", "mode = \"rotate\""),
            valid.replace("value_kind = \"i32\"", "value_kind = \"b32\""),
            valid.replace(
                "source_lane = \"in_range_source_active_and_named_out_of_range_copies_self\"",
                "source_lane = \"unchecked\"",
            ),
            valid.replace(
                "lane_encoding = \"register_or_immediate\"",
                "lane_encoding = \"anything\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(
                toml::from_str::<WarpShuffle>(&invalid).is_err(),
                "{invalid}"
            );
        }

        let i64 = r#"
mode = "down"
value_kind = "i64"
participation = "executing_lane_named_all_named_lanes_same_instruction_and_mask"
legacy_pre_sm70 = "all_named_lanes_converged_and_only_named_lanes_active"
source_lane = "in_range_source_active_and_named_out_of_range_copies_self"
adapter = "mask_value_lane_or_delta_split_i64_low_high_b32_insert_clamp_reassemble"
clamp = 31
lane_encoding = "register_only"
mask_encoding = "register_only"
"#;
        let parsed = toml::from_str::<WarpShuffle>(i64).unwrap();
        assert_eq!(parsed.value_kind, WarpShuffleValueKind::I64);
        assert_eq!(
            parsed.adapter,
            WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble
        );
        assert_eq!(
            parsed.lane_encoding,
            WarpShuffleOperandEncoding::RegisterOnly
        );

        for invalid in [
            i64.replace("value_kind = \"i64\"", "value_kind = \"u64\""),
            i64.replace(
                "adapter = \"mask_value_lane_or_delta_split_i64_low_high_b32_insert_clamp_reassemble\"",
                "adapter = \"split_any_width\"",
            ),
            i64.replace(
                "mask_encoding = \"register_only\"",
                "mask_encoding = \"any_operand\"",
            ),
        ] {
            assert!(
                toml::from_str::<WarpShuffle>(&invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn warp_match_contract_rejects_open_ended_adapters_and_encodings() {
        let valid = r#"
mode = "all"
value_width = "b64"
participation = "executing_lane_named_all_named_lanes_same_instruction_and_mask"
adapter = "project_mask_discard_predicate"
value_encoding = "register_or_immediate"
mask_encoding = "register_or_immediate"
"#;
        toml::from_str::<WarpMatch>(valid).unwrap();

        for invalid in [
            valid.replace("mode = \"all\"", "mode = \"equal\""),
            valid.replace(
                "adapter = \"project_mask_discard_predicate\"",
                "adapter = \"first_result\"",
            ),
            valid.replace(
                "value_encoding = \"register_or_immediate\"",
                "value_encoding = \"anything\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(toml::from_str::<WarpMatch>(&invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn warp_barrier_contract_rejects_open_ended_policy() {
        let valid = r#"
participation = "executing_lane_named_all_named_lanes_same_instruction_and_mask"
legacy_pre_sm70 = "all_named_lanes_converged_and_only_named_lanes_active"
adapter = "direct_member_mask"
mask_encoding = "register_or_immediate"
memory_ordering = "participating_lanes"
"#;
        toml::from_str::<WarpBarrier>(valid).unwrap();

        for invalid in [
            valid.replace("adapter = \"direct_member_mask\"", "adapter = \"direct\""),
            valid.replace(
                "legacy_pre_sm70 = \"all_named_lanes_converged_and_only_named_lanes_active\"",
                "legacy_pre_sm70 = \"independent_threads\"",
            ),
            valid.replace(
                "mask_encoding = \"register_or_immediate\"",
                "mask_encoding = \"any_operand\"",
            ),
            valid.replace(
                "memory_ordering = \"participating_lanes\"",
                "memory_ordering = \"none\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(
                toml::from_str::<WarpBarrier>(&invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn mbarrier_basic_contract_rejects_open_ended_policy() {
        let valid = r#"
operation = "test_wait"
state_space = "shared"
adapter = "pointer_token_to_predicate"
runtime_validation = "unexecuted"
"#;
        let parsed = toml::from_str::<MbarrierBasic>(valid).unwrap();
        assert_eq!(parsed.operation, MbarrierBasicOperation::TestWait);
        assert_eq!(
            parsed.adapter,
            MbarrierBasicAdapter::PointerTokenToPredicate
        );

        for invalid in [
            valid.replace("operation = \"test_wait\"", "operation = \"wait\""),
            valid.replace("state_space = \"shared\"", "state_space = \"global\""),
            valid.replace(
                "adapter = \"pointer_token_to_predicate\"",
                "adapter = \"direct\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(
                toml::from_str::<MbarrierBasic>(&invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn cp_async_mbarrier_contract_rejects_open_ended_policy() {
        let valid = r#"
operation = "arrive_no_inc"
state_space = "shared"
adapter = "pointer_to_void"
runtime_validation = "unexecuted"
"#;
        let parsed = toml::from_str::<CpAsyncMbarrier>(valid).unwrap();
        assert_eq!(parsed.operation, CpAsyncMbarrierOperation::ArriveNoInc);
        assert_eq!(parsed.state_space, CpAsyncMbarrierStateSpace::Shared);

        for invalid in [
            valid.replace("operation = \"arrive_no_inc\"", "operation = \"wait\""),
            valid.replace("state_space = \"shared\"", "state_space = \"global\""),
            valid.replace("adapter = \"pointer_to_void\"", "adapter = \"direct\""),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(
                toml::from_str::<CpAsyncMbarrier>(&invalid).is_err(),
                "{invalid}"
            );
        }
    }
}
