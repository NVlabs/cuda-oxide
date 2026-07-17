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
    #[serde(default)]
    pub intrinsics: Vec<OverlayIntrinsic>,
    #[serde(default)]
    pub register_mma_int4: Option<RegisterMmaIntegerAdmission>,
    #[serde(default)]
    pub register_mma_int8: Option<RegisterMmaIntegerAdmission>,
    #[serde(default)]
    pub register_mma_b1: Option<RegisterMmaBinaryAdmission>,
    #[serde(default)]
    pub register_mma_f8f6f4_f32: Option<RegisterMmaF8F6F4Admission>,
    #[serde(default)]
    pub register_mma_f8f6f4_f16: Option<RegisterMmaF8F6F4Admission>,
    #[serde(default)]
    pub register_mma_fp8: Option<RegisterMmaFp8Admission>,
    #[serde(default)]
    pub register_mma_ampere_float: Option<RegisterMmaAmpereFloatAdmission>,
    #[serde(default, alias = "sparse_mma_int8")]
    pub sparse_mma_integer: Option<SparseMmaIntegerAdmission>,
    #[serde(default)]
    pub sparse_mma_f8f6f4_f32: Option<SparseMmaF8F6F4Admission>,
    #[serde(default)]
    pub sparse_mma_f8f6f4_f16: Option<SparseMmaF8F6F4F16Admission>,
    #[serde(default)]
    pub prmt: Option<PrmtAdmission>,
    #[serde(default)]
    pub packed_conversion_fp8: Option<PackedConversionFp8Admission>,
    #[serde(default)]
    pub scalar_conversion: Option<ScalarConversionAdmission>,
    #[serde(default)]
    pub scalar_arithmetic: Option<ScalarArithmeticAdmission>,
    #[serde(default)]
    pub extended_minmax: Option<ExtendedMinMaxAdmission>,
    #[serde(default)]
    pub cluster_sreg: Option<ClusterSregAdmission>,
    #[serde(default)]
    pub cluster_barrier: Option<ClusterBarrierAdmission>,
    #[serde(default)]
    pub mbarrier_extended: Option<MbarrierExtendedAdmission>,
    #[serde(default)]
    pub special_registers: Option<SpecialRegisterAdmission>,
    #[serde(default)]
    pub debug_control: Option<DebugControlAdmission>,
    #[serde(default)]
    pub threadfence: Option<ThreadfenceAdmission>,
    #[serde(default)]
    pub cluster_memory: Option<ClusterMemoryAdmission>,
    #[serde(default)]
    pub stmatrix: Option<StmatrixAdmission>,
    #[serde(default)]
    pub clc: Option<ClcAdmission>,
    #[serde(default)]
    pub wgmma_controls: Option<WgmmaControlAdmission>,
    #[serde(default)]
    pub tma: Option<TmaAdmission>,
    #[serde(default)]
    pub tcgen05: Option<Tcgen05Admission>,
}

/// Compact admission for the four existing `stmatrix` stores.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StmatrixAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<StmatrixAdmissionVariant>,
}

/// One reviewed `stmatrix` multiplicity and layout.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StmatrixAdmissionVariant {
    pub abi_id: String,
    pub multiplicity: StmatrixMultiplicity,
    pub layout: StmatrixLayout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StmatrixMultiplicity {
    X2,
    X4,
}

impl StmatrixMultiplicity {
    pub const fn register_count(self) -> usize {
        match self {
            Self::X2 => 2,
            Self::X4 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StmatrixLayout {
    Normal,
    Transposed,
}

/// Compact admission for the remaining handwritten mbarrier operations.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MbarrierExtendedAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<MbarrierExtendedAdmissionVariant>,
}

/// One reviewed extended-mbarrier operation and its reserved ABI ID.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MbarrierExtendedAdmissionVariant {
    pub abi_id: String,
    pub operation: MbarrierExtendedOperation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MbarrierExtendedOperation {
    ArriveExpectTxCta,
    ArriveExpectTxCluster,
    ArriveRemoteCluster,
    TryWaitTokenCta,
    TryWaitParityCta,
    TryWaitParityCluster,
    FenceProxyAsyncSharedCta,
    FenceMbarrierInitReleaseCluster,
    FenceProxyAsyncGenericReleaseSharedCtaCluster,
    FenceProxyAsyncGenericAcquireSharedClusterCluster,
    Nanosleep,
}

/// Compact admission for Hopper cluster special registers.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterSregAdmission {
    pub axes: Vec<String>,
    pub xyz_product_count: usize,
    pub record_count: usize,
}

/// Compact admission for the six cluster-barrier instructions.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterBarrierAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<ClusterBarrierAdmissionVariant>,
}

/// One reviewed cluster-barrier spelling.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterBarrierAdmissionVariant {
    pub abi_id: String,
    pub mode: ClusterBarrierMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusterBarrierMode {
    Arrive,
    ArriveAligned,
    ArriveRelaxed,
    ArriveRelaxedAligned,
    Wait,
    WaitAligned,
}

/// Closed semantics for one cluster-barrier spelling.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterBarrier {
    pub mode: ClusterBarrierMode,
    pub ordering: ClusterBarrierOrdering,
    pub aligned: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusterBarrierOrdering {
    Release,
    Relaxed,
    Acquire,
}

/// Compact admission for cluster address mapping and remote shared reads.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterMemoryAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<ClusterMemoryAdmissionVariant>,
}

/// One reviewed cluster-memory operation and its reserved ABI ID.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterMemoryAdmissionVariant {
    pub abi_id: String,
    pub operation: ClusterMemoryOperation,
}

/// Compact admission for the three WGMMA control instructions.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WgmmaControlAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<WgmmaControlAdmissionVariant>,
}

/// One reviewed WGMMA control operation.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WgmmaControlAdmissionVariant {
    pub abi_id: String,
    pub mode: WgmmaControlMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusterMemoryOperation {
    MapSharedRank,
    ReadU32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WgmmaControlMode {
    Fence,
    CommitGroup,
    WaitGroup,
}

/// Closed semantics for one WGMMA control operation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WgmmaControl {
    pub mode: WgmmaControlMode,
    pub adapter: WgmmaControlAdapter,
    pub participation: WgmmaControlParticipation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WgmmaControlAdapter {
    NoArguments,
    ConstGenericU32ToI64Immediate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WgmmaControlParticipation {
    WarpgroupAllThreadsSameInstruction,
}

/// Compact admission for the reviewed non-launch special-register reads.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialRegisterAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    pub registers: Vec<SpecialRegisterKind>,
    pub product_count: usize,
}

/// Compact admission for PTX debug controls.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DebugControlAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    pub operations: Vec<DebugControlOperation>,
    /// Filled only when this pending shard is aggregated.
    #[serde(default)]
    pub abi_ids: Vec<String>,
}

/// Compact admission for the three CUDA thread fences.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThreadfenceAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<ThreadfenceAdmissionVariant>,
}

/// One reviewed thread-fence scope.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThreadfenceAdmissionVariant {
    pub abi_id: String,
    pub scope: ThreadfenceScope,
}

/// Scope encoded by a PTX `membar` instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThreadfenceScope {
    Cta,
    Device,
    System,
}

/// Compact admission for Cluster Launch Control.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClcAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<ClcAdmissionVariant>,
}

/// One reviewed Cluster Launch Control operation.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClcAdmissionVariant {
    pub abi_id: String,
    pub operation: ClcOperation,
}

/// Compact admission for the existing TMA copy and group operations.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TmaAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<TmaAdmissionVariant>,
}

/// One reviewed TMA operation and its reserved ABI ID.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TmaAdmissionVariant {
    pub abi_id: String,
    pub operation: TmaOperation,
}

/// Compact admission for the existing Tensor Core Generation 5 operations.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05Admission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    #[serde(default)]
    pub cp_llvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub cp_libnvvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub ld_llvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub ld_libnvvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub st_llvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub st_libnvvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub offset_llvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub offset_libnvvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub control_llvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub control_libnvvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub mma_llvm_evidence_profile: Option<String>,
    #[serde(default)]
    pub mma_libnvvm_evidence_profile: Option<String>,
    #[serde(rename = "mma_llvm_target_contract", default)]
    pub mma_llvm_target_contracts: Vec<TargetContract>,
    #[serde(rename = "mma_libnvvm_target_contract", default)]
    pub mma_libnvvm_target_contracts: Vec<TargetContract>,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<Tcgen05AdmissionVariant>,
    #[serde(rename = "cp_variant", default)]
    pub cp_variants: Vec<Tcgen05CpAdmissionVariant>,
    #[serde(rename = "ld_variant", default)]
    pub ld_variants: Vec<Tcgen05LdAdmissionVariant>,
    #[serde(rename = "st_variant", default)]
    pub st_variants: Vec<Tcgen05StAdmissionVariant>,
    #[serde(rename = "ld_offset_variant", default)]
    pub ld_offset_variants: Vec<Tcgen05LdAdmissionVariant>,
    #[serde(rename = "st_offset_variant", default)]
    pub st_offset_variants: Vec<Tcgen05StAdmissionVariant>,
    #[serde(rename = "mma_variant", default)]
    pub mma_variants: Vec<Tcgen05MmaAdmissionVariant>,
}

/// One reviewed tcgen05 operation and its reserved ABI ID.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05AdmissionVariant {
    pub abi_id: String,
    pub operation: Tcgen05Operation,
}

/// One reviewed tcgen05 copy member and CTA group.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05CpAdmissionVariant {
    pub abi_id: String,
    pub member: Tcgen05CpMember,
    pub group: Tcgen05CpGroup,
}

/// One reviewed tcgen05 load shape, repetition, and packing mode.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05LdAdmissionVariant {
    pub abi_id: String,
    pub shape: Tcgen05LdShape,
    pub multiplicity: Tcgen05LdMultiplicity,
    pub pack16: bool,
}

/// One reviewed tcgen05 store shape, repetition, and unpacking mode.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05StAdmissionVariant {
    pub abi_id: String,
    pub shape: Tcgen05LdShape,
    pub multiplicity: Tcgen05LdMultiplicity,
    pub unpack16: bool,
}

/// One reviewed tcgen05 MMA source form or compatibility alias.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05MmaAdmissionVariant {
    pub abi_id: String,
    pub form: Tcgen05MmaForm,
    #[serde(default)]
    pub alias: Option<Tcgen05MmaAlias>,
}

/// Compact admission for the closed `prmt` family.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrmtAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<PrmtAdmissionVariant>,
}

/// One reviewed member of the `prmt` family.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrmtAdmissionVariant {
    pub abi_id: String,
    pub mode: PrmtMode,
}

/// Compact admission for the closed scalar-f32 to packed-FP8 family.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedConversionFp8Admission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    pub destination_formats: Vec<PackedConversionDestinationFormat>,
    pub saturations: Vec<PackedConversionSaturation>,
    pub product_count: usize,
}

/// Compact admission for scalar F32-to-TF32 conversions.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarConversionAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<ScalarConversionAdmissionVariant>,
}

/// One reviewed scalar conversion variant.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarConversionAdmissionVariant {
    pub abi_id: String,
    pub rounding: ScalarConversionRounding,
    pub saturation: ScalarConversionSaturation,
}

/// Compact admission for scalar floating-point arithmetic.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarArithmeticAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<ScalarArithmeticAdmissionVariant>,
}

/// One reviewed scalar arithmetic variant.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarArithmeticAdmissionVariant {
    pub abi_id: String,
    pub format: ScalarArithmeticFormat,
    pub operation: ScalarArithmeticOperation,
    pub rounding: ScalarArithmeticRounding,
    pub subnormal: ScalarArithmeticSubnormal,
    pub saturation: ScalarArithmeticSaturation,
}

/// Compact admission for the exact extended floating-point min/max family.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtendedMinMaxAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<ExtendedMinMaxAdmissionVariant>,
}

/// One reviewed extended min/max variant.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtendedMinMaxAdmissionVariant {
    pub abi_id: String,
    pub format: ExtendedMinMaxFormat,
    pub operation: ExtendedMinMaxOperation,
    pub subnormal: ExtendedMinMaxSubnormal,
    pub nan: ExtendedMinMaxNan,
    pub xorsign_abs: bool,
}

/// Compact admission for a closed dense integer register-MMA family.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterMmaIntegerAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<RegisterMmaIntegerVariant>,
}

/// One reviewed member of a dense integer register-MMA family.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterMmaIntegerVariant {
    pub shape: RegisterMmaShape,
    pub a_element: RegisterMmaElement,
    pub b_element: RegisterMmaElement,
    pub overflow: RegisterMmaOverflow,
}

/// Compact admission for the closed dense binary register-MMA family.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterMmaBinaryAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    #[serde(rename = "variant")]
    pub variants: Vec<RegisterMmaBinaryVariant>,
}

/// One reviewed member of the dense binary register-MMA family.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterMmaBinaryVariant {
    pub shape: RegisterMmaShape,
    pub operation: RegisterMmaOperation,
}

/// Compact admission for one dense Blackwell `kind::f8f6f4` matrix.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterMmaF8F6F4Admission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    pub first_abi_id: String,
    pub a_elements: Vec<RegisterMmaElement>,
    pub b_elements: Vec<RegisterMmaElement>,
    pub product_count: usize,
    pub targets: Vec<String>,
}

/// Compact admission for the standard FP8 register-MMA family.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterMmaFp8Admission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    pub first_abi_id: String,
    pub shapes: Vec<RegisterMmaShape>,
    pub accumulators: Vec<RegisterMmaAccumulator>,
    pub a_elements: Vec<RegisterMmaElement>,
    pub b_elements: Vec<RegisterMmaElement>,
    pub product_count: usize,
}

/// Compact admission for the reviewed Ampere floating-point MMA forms.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterMmaAmpereFloatAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    pub first_abi_id: String,
    pub product_count: usize,
    #[serde(rename = "variant")]
    pub variants: Vec<RegisterMmaAmpereFloatVariant>,
}

/// One reviewed Ampere floating-point MMA form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterMmaAmpereFloatVariant {
    pub shape: RegisterMmaShape,
    pub accumulator: RegisterMmaAccumulator,
    pub element: RegisterMmaElement,
}

/// Compact admission for a sparse integer register-MMA family.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseMmaIntegerAdmission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    pub metadata: SparseMmaMetadata,
    #[serde(rename = "variant")]
    pub variants: Vec<SparseMmaIntegerVariant>,
}

/// One reviewed member of a sparse integer register-MMA family.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseMmaIntegerVariant {
    pub shape: SparseMmaShape,
    pub a_element: SparseMmaElement,
    pub b_element: SparseMmaElement,
    pub overflow: SparseMmaOverflow,
}

/// Compact admission for ordered sparse `kind::f8f6f4` F32 MMA.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseMmaF8F6F4Admission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    pub a_elements: Vec<SparseMmaElement>,
    pub b_elements: Vec<SparseMmaElement>,
    pub product_count: usize,
}

/// Compact admission for ordered sparse `kind::f8f6f4` packed-F16 MMA.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseMmaF8F6F4F16Admission {
    pub llvm_evidence_profile: String,
    pub libnvvm_evidence_profile: String,
    pub runtime_validation: RuntimeValidation,
    pub first_abi_id: String,
    pub a_elements: Vec<SparseMmaElement>,
    pub b_elements: Vec<SparseMmaElement>,
    pub product_count: usize,
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
    pub scalar_conversion: Option<ScalarConversion>,
    #[serde(default)]
    pub scalar_arithmetic: Option<ScalarArithmetic>,
    #[serde(default)]
    pub extended_minmax: Option<ExtendedMinMax>,
    #[serde(default)]
    pub cp_async_copy: Option<CpAsyncCopy>,
    #[serde(default)]
    pub cp_async_control: Option<CpAsyncControl>,
    #[serde(default)]
    pub cp_async_mbarrier: Option<CpAsyncMbarrier>,
    #[serde(default)]
    pub mbarrier_basic: Option<MbarrierBasic>,
    #[serde(default)]
    pub movmatrix: Option<Movmatrix>,
    #[serde(default)]
    pub mbarrier_extended: Option<MbarrierExtended>,
    #[serde(default)]
    pub register_mma: Option<RegisterMma>,
    #[serde(default)]
    pub sparse_mma: Option<SparseMma>,
    #[serde(default)]
    pub prmt: Option<Prmt>,
    #[serde(default)]
    pub cluster_barrier: Option<ClusterBarrier>,
    #[serde(default)]
    pub wgmma_control: Option<WgmmaControl>,
    #[serde(default)]
    pub special_register: Option<SpecialRegister>,
    #[serde(default)]
    pub debug_control: Option<DebugControl>,
    #[serde(default)]
    pub cluster_memory: Option<ClusterMemory>,
    #[serde(default)]
    pub clc: Option<Clc>,
    #[serde(default)]
    pub tma: Option<Tma>,
    #[serde(default)]
    pub tcgen05: Option<Tcgen05>,
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

/// Closed semantic contract for byte permutation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prmt {
    pub mode: PrmtMode,
    pub adapter: PrmtAdapter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrmtMode {
    Generic,
    F4e,
    B4e,
    Rc8,
    Ecl,
    Ecr,
    Rc16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrmtAdapter {
    DirectThreeOperands,
    InsertZeroSecondSource,
}

/// Closed semantic and lowering contract for one special-register read.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialRegister {
    pub register: SpecialRegisterKind,
    pub observation: SpecialRegisterObservation,
    pub result_width: SpecialRegisterWidth,
    pub ptx_type: SpecialRegisterPtxType,
    pub output_constraint: SpecialRegisterOutputConstraint,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llvm_exclusion: Option<SpecialRegisterLlvmExclusion>,
}

/// Closed semantic contract for PTX debug controls.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DebugControl {
    pub operation: DebugControlOperation,
    pub adapter: DebugControlAdapter,
    pub runtime_validation: RuntimeValidation,
}

/// Closed contract for cluster address mapping and remote shared reads.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterMemory {
    pub operation: ClusterMemoryOperation,
    pub adapter: ClusterMemoryAdapter,
    pub source_contract: ClusterMemorySourceContract,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusterMemoryAdapter {
    GenericConstAndMutPointerRankToSamePointer,
    ConstU32PointerRankToU32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClusterMemorySourceContract {
    LlvmMapaSharedClusterAs7IdentityInlinePtx,
    PtxNativeMapaThenWeakClusterLoad,
}

/// Closed semantic contract for Cluster Launch Control.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Clc {
    pub operation: ClcOperation,
    pub adapter: ClcAdapter,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClcOperation {
    TryCancel,
    TryCancelMulticast,
    QueryIsCanceled,
    QueryGetFirstCtaidX,
    QueryGetFirstCtaidY,
    QueryGetFirstCtaidZ,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClcAdapter {
    GenericPointersToShared,
    PairU64ToI128BoolToU32,
    PairU64ToI128U32,
}

/// Closed semantic contract for a TMA operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tma {
    pub operation: TmaOperation,
    pub adapter: TmaAdapter,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TmaOperation {
    G2sTile1d,
    G2sTile2d,
    G2sTile2dMulticast,
    G2sTile2dMulticastCg2,
    G2sTile3d,
    G2sTile4d,
    G2sTile5d,
    S2gTile1d,
    S2gTile2d,
    S2gTile3d,
    S2gTile4d,
    S2gTile5d,
    CommitGroup,
    WaitGroup,
    WaitGroupRead,
}

impl TmaOperation {
    pub const fn dimensions(self) -> Option<usize> {
        match self {
            Self::G2sTile1d | Self::S2gTile1d => Some(1),
            Self::G2sTile2d
            | Self::G2sTile2dMulticast
            | Self::G2sTile2dMulticastCg2
            | Self::S2gTile2d => Some(2),
            Self::G2sTile3d | Self::S2gTile3d => Some(3),
            Self::G2sTile4d | Self::S2gTile4d => Some(4),
            Self::G2sTile5d | Self::S2gTile5d => Some(5),
            Self::CommitGroup | Self::WaitGroup | Self::WaitGroupRead => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TmaAdapter {
    G2sPointersCoordinatesBarrierInjectDefaults,
    G2sPointersCoordinatesBarrierMaskInjectDefaults,
    S2gPointersCoordinatesInjectDefaults,
    NoOperands,
    CompileTimeConstantMaxPending,
}

/// Closed semantic contract for one tcgen05 operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05 {
    pub operation: Tcgen05Operation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cp: Option<Tcgen05Cp>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ld: Option<Tcgen05Ld>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub st: Option<Tcgen05St>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mma: Option<Tcgen05Mma>,
    pub adapter: Tcgen05Adapter,
    pub source_contract: Tcgen05SourceContract,
    pub runtime_validation: RuntimeValidation,
}

/// Closed identity and selector contract for one tcgen05 MMA API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05Mma {
    pub form: Tcgen05MmaForm,
    pub selector_layout: Tcgen05MmaSelectorLayout,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixed_selectors: Option<Tcgen05MmaFixedSelectors>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alias: Option<Tcgen05MmaAlias>,
    pub llvm_target: CatalogTargetRequirement,
    pub libnvvm_target: CatalogTargetRequirement,
}

/// The 14 LLVM source forms covered by the first tcgen05 MMA batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tcgen05MmaForm {
    Shared,
    Tensor,
    TensorAshift,
    SpShared,
    SpTensor,
    SpTensorAshift,
    WsShared,
    WsSharedZeroColMask,
    WsSpShared,
    WsSpSharedZeroColMask,
    WsSpTensor,
    WsSpTensorZeroColMask,
    WsTensor,
    WsTensorZeroColMask,
}

/// Immediate arguments that select one imported tcgen05 MMA spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Tcgen05MmaSelectorLayout {
    Base {
        kind_argument: u8,
        cta_group_argument: u8,
        collector_a_argument: u8,
        collector_a_upper_exclusive: u8,
    },
    WarpSpecialized {
        kind_argument: u8,
        b_buffer_argument: u8,
        b_usage_argument: u8,
    },
}

/// A fixed warp-specialized selector tuple used by compatibility aliases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05MmaFixedSelectors {
    pub kind: Tcgen05MmaKind,
    pub b_buffer: u8,
    pub b_usage: Tcgen05MmaBUsage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tcgen05MmaKind {
    F16,
    Tf32,
    F8f6f4,
    I8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tcgen05MmaBUsage {
    Discard,
    LastUse,
    Fill,
    Use,
}

impl Tcgen05MmaBUsage {
    pub const fn selector_value(self) -> u8 {
        match self {
            Self::Discard => 0,
            Self::LastUse => 1,
            Self::Fill => 2,
            Self::Use => 3,
        }
    }
}

/// Public names proposed by PR #346 for the generic f8f6f4 carrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tcgen05MmaAlias {
    E4m3,
    E5m2,
    E2m3,
    E3m2,
    E2m1,
}

/// Closed identity for one tcgen05 shared-to-tensor-memory copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05Cp {
    pub member: Tcgen05CpMember,
    pub group: Tcgen05CpGroup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Tcgen05CpMember {
    #[serde(rename = "128x128b_b4x16_p64")]
    M128x128bB4x16P64,
    #[serde(rename = "128x128b_b6x16_p32")]
    M128x128bB6x16P32,
    #[serde(rename = "128x128b")]
    M128x128b,
    #[serde(rename = "128x256b_b4x16_p64")]
    M128x256bB4x16P64,
    #[serde(rename = "128x256b_b6x16_p32")]
    M128x256bB6x16P32,
    #[serde(rename = "32x128b_warpx4_b4x16_p64")]
    M32x128bWarpx4B4x16P64,
    #[serde(rename = "32x128b_warpx4_b6x16_p32")]
    M32x128bWarpx4B6x16P32,
    #[serde(rename = "32x128b_warpx4")]
    M32x128bWarpx4,
    #[serde(rename = "4x256b_b4x16_p64")]
    M4x256bB4x16P64,
    #[serde(rename = "4x256b_b6x16_p32")]
    M4x256bB6x16P32,
    #[serde(rename = "4x256b")]
    M4x256b,
    #[serde(rename = "64x128b_warpx2_01_23_b4x16_p64")]
    M64x128bWarpx2Pair0123B4x16P64,
    #[serde(rename = "64x128b_warpx2_01_23_b6x16_p32")]
    M64x128bWarpx2Pair0123B6x16P32,
    #[serde(rename = "64x128b_warpx2_01_23")]
    M64x128bWarpx2Pair0123,
    #[serde(rename = "64x128b_warpx2_02_13_b4x16_p64")]
    M64x128bWarpx2Pair0213B4x16P64,
    #[serde(rename = "64x128b_warpx2_02_13_b6x16_p32")]
    M64x128bWarpx2Pair0213B6x16P32,
    #[serde(rename = "64x128b_warpx2_02_13")]
    M64x128bWarpx2Pair0213,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tcgen05CpGroup {
    Cg1,
    Cg2,
}

/// Closed identity for one tcgen05 tensor-memory load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05Ld {
    pub shape: Tcgen05LdShape,
    pub multiplicity: Tcgen05LdMultiplicity,
    pub pack16: bool,
}

/// Closed identity for one tcgen05 tensor-memory store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tcgen05St {
    pub shape: Tcgen05LdShape,
    pub multiplicity: Tcgen05LdMultiplicity,
    pub unpack16: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Tcgen05LdShape {
    #[serde(rename = "16x32bx2")]
    M16x32bx2,
    #[serde(rename = "16x64b")]
    M16x64b,
    #[serde(rename = "16x128b")]
    M16x128b,
    #[serde(rename = "16x256b")]
    M16x256b,
    #[serde(rename = "32x32b")]
    M32x32b,
}

impl Tcgen05LdShape {
    pub const fn register_multiplier(self) -> usize {
        match self {
            Self::M16x32bx2 | Self::M16x64b | Self::M32x32b => 1,
            Self::M16x128b => 2,
            Self::M16x256b => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tcgen05LdMultiplicity {
    X1,
    X2,
    X4,
    X8,
    X16,
    X32,
    X64,
    X128,
}

impl Tcgen05LdMultiplicity {
    pub const fn count(self) -> usize {
        match self {
            Self::X1 => 1,
            Self::X2 => 2,
            Self::X4 => 4,
            Self::X8 => 8,
            Self::X16 => 16,
            Self::X32 => 32,
            Self::X64 => 64,
            Self::X128 => 128,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tcgen05Operation {
    Alloc,
    Dealloc,
    RelinquishAllocPermit,
    FenceBeforeThreadSync,
    FenceAfterThreadSync,
    Commit,
    CommitSharedCluster,
    MmaWsF16,
    MmaF16,
    MmaWsBf16,
    MmaWsTf32,
    CpSmemToTmem,
    Ld16x256bX8Pure,
    Ld16x256bPure,
    LoadWait,
    StoreWait,
    AllocCg2,
    DeallocCg2,
    RelinquishAllocPermitCg2,
    MmaF16Cg2,
    CommitCg2,
    CommitSharedClusterCg2,
    CommitMulticastCg2,
    CpSmemToTmemCg2,
    Ld,
    St,
    CommitMulticast,
    ShiftDown,
    ShiftDownCg2,
    Mma,
}

impl Tcgen05Operation {
    pub const fn execution_scope(self) -> &'static str {
        match self {
            Self::Alloc
            | Self::Dealloc
            | Self::RelinquishAllocPermit
            | Self::Ld16x256bX8Pure
            | Self::Ld16x256bPure
            | Self::LoadWait
            | Self::StoreWait
            | Self::AllocCg2
            | Self::DeallocCg2
            | Self::RelinquishAllocPermitCg2
            | Self::Ld
            | Self::St => "warp",
            Self::FenceBeforeThreadSync
            | Self::FenceAfterThreadSync
            | Self::Commit
            | Self::CommitSharedCluster
            | Self::MmaWsF16
            | Self::MmaF16
            | Self::MmaWsBf16
            | Self::MmaWsTf32
            | Self::CpSmemToTmem
            | Self::MmaF16Cg2
            | Self::CommitCg2
            | Self::CommitSharedClusterCg2
            | Self::CommitMulticastCg2
            | Self::CpSmemToTmemCg2
            | Self::CommitMulticast
            | Self::ShiftDown
            | Self::ShiftDownCg2
            | Self::Mma => "thread",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tcgen05Adapter {
    SharedPointerColumnsToVoid,
    TmemAddressColumnsToVoid,
    NoOperands,
    BarrierPointerToVoid,
    MmaWsDropLegacyADescriptor,
    MmaInjectZeroDisableLanes,
    TmemDescriptorToVoid,
    TmemToF32x32,
    TmemToF32x4,
    BarrierPointerMaskToVoid,
    TmemInjectPack16ToU32Registers,
    TmemU32RegistersInjectUnpack16ToVoid,
    TmemHalfSplitOffsetInjectPack16ToU32Registers,
    TmemHalfSplitOffsetU32RegistersInjectUnpack16ToVoid,
    TmemAddressToVoid,
    MmaDirectSelectors,
    MmaWsFixedSelectorsDropLegacyADescriptor,
}

/// Relationship between the public operation and LLVM's NVPTX selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tcgen05SourceContract {
    ExactTablegenSelection,
    TablegenSelectionChangesPtx,
    LlvmCustomLoweringWithoutSelection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecialRegisterKind {
    Clock,
    Clock64,
    Globaltimer,
    Envreg1,
    Envreg2,
    Smid,
    Nsmid,
    Gridid,
    Warpid,
    Nwarpid,
    DynamicSmemSize,
    TotalSmemSize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DebugControlOperation {
    Trap,
    Breakpoint,
    Pmevent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecialRegisterObservation {
    StablePure,
    VolatileObservation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecialRegisterWidth {
    B32,
    B64,
}

impl SpecialRegisterWidth {
    pub const fn bits(self) -> u32 {
        match self {
            Self::B32 => 32,
            Self::B64 => 64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecialRegisterPtxType {
    B32,
    U32,
    U64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecialRegisterOutputConstraint {
    Register32,
    Register64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecialRegisterLlvmExclusion {
    pub source_record: String,
    pub llvm_symbol: String,
    pub imported_result_width: SpecialRegisterWidth,
    pub reason: SpecialRegisterLlvmExclusionReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecialRegisterLlvmExclusionReason {
    ResultWidthMismatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DebugControlAdapter {
    Direct,
    ConstGenericToImmediateU32,
}

/// Backend-specific lowering selected by reviewed evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlayBackendLowering {
    pub backend: IntrinsicBackend,
    pub mechanism: BackendLoweringMechanism,
    pub evidence_profile: String,
    /// Optional exact target alternatives for this backend route.
    #[serde(default)]
    pub targets: Option<String>,
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

impl LdmatrixVariant {
    pub const fn register_count(&self) -> usize {
        let matrices = self.multiplicity.register_count();
        match self.shape {
            LdmatrixShape::M8n8 | LdmatrixShape::M8n16 => matrices,
            LdmatrixShape::M16n16 => matrices * 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LdmatrixShape {
    M8n8,
    M8n16,
    M16n16,
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
    B8,
    B8x16B4x16P64,
    B8x16B6x16P32,
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
    AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LdmatrixAddressContract {
    WarpLaneAddressesMappedByMultiplicitySixteenByteAlignedSixteenBytesReadableWithSm75Replication,
    WarpLaneAddressesMappedByMultiplicitySixteenByteAlignedSixteenBytesReadable,
    WarpLaneAddressesMappedByMultiplicitySixteenByteAlignedThirtyTwoBytesReadable,
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

/// Closed contract for the in-register warp matrix transpose.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Movmatrix {
    pub participation: MovmatrixParticipation,
    pub adapter: MovmatrixAdapter,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MovmatrixParticipation {
    AllWarpLanesSameInstructionNoExitedLanes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MovmatrixAdapter {
    PackedB16x2U32ToPackedB16x2U32,
}

/// Closed contract for register-only warp-level `mma.sync` operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisterMma {
    pub shape: RegisterMmaShape,
    pub operation: RegisterMmaOperation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<RegisterMmaKind>,
    pub accumulator: RegisterMmaAccumulator,
    pub a_element: RegisterMmaElement,
    pub b_element: RegisterMmaElement,
    pub a_layout: RegisterMmaLayout,
    pub b_layout: RegisterMmaLayout,
    pub overflow: RegisterMmaOverflow,
    pub participation: RegisterMmaParticipation,
    pub adapter: RegisterMmaAdapter,
    pub compatibility_source: RegisterMmaCompatibilitySource,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterMmaKind {
    Standard,
    F8f6f4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterMmaShape {
    M8n8k4,
    M8n8k16,
    M8n8k32,
    M16n8k4,
    M16n8k8,
    M16n8k16,
    M16n8k32,
    M16n8k64,
    M8n8k128,
    M16n8k128,
    M16n8k256,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterMmaOperation {
    Multiply,
    AndPopc,
    XorPopc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterMmaAccumulator {
    F16,
    F32,
    F64,
    S32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterMmaElement {
    B1,
    Bf16,
    E2m1,
    E2m3,
    E3m2,
    E4m3,
    E5m2,
    F16,
    Tf32,
    F64,
    S4,
    U4,
    S8,
    U8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterMmaLayout {
    Row,
    Col,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterMmaOverflow {
    NotApplicable,
    Wrapping,
    Satfinite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterMmaParticipation {
    AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
}

/// Rust `C, A, B` fragment shape used by the importer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterMmaAdapter {
    C2U32A2U32B1U32ToD2U32,
    C2U32A4U32B2U32ToD2U32,
    C4F32A2U32B1U32ToD4F32,
    C4F32A4U32B2U32ToD4F32,
    C2F64A1F64B1F64ToD2F64,
    C2I32A1U32B1U32ToD2I32,
    C4I32A4U32B2U32ToD4I32,
    C4I32A2U32B1U32ToD4I32,
}

/// Where the stable `cuda_device::wmma` callable is defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegisterMmaCompatibilitySource {
    ExistingStub,
    GeneratedStub,
}

/// Closed semantic contract for register-only sparse `mma.sp` operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SparseMma {
    pub shape: SparseMmaShape,
    pub accumulator: SparseMmaAccumulator,
    pub a_element: SparseMmaElement,
    pub b_element: SparseMmaElement,
    pub a_layout: SparseMmaLayout,
    pub b_layout: SparseMmaLayout,
    pub overflow: SparseMmaOverflow,
    pub metadata: SparseMmaMetadata,
    pub selector: SparseMmaSelector,
    pub participation: SparseMmaParticipation,
    pub adapter: SparseMmaAdapter,
    pub llvm_adapter: SparseMmaLlvmAdapter,
    pub compatibility_source: SparseMmaCompatibilitySource,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMmaShape {
    M16n8k32,
    M16n8k64,
    M16n8k128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMmaAccumulator {
    F16,
    F32,
    S32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMmaElement {
    E2m1,
    E2m3,
    E3m2,
    E4m3,
    E5m2,
    S4,
    U4,
    S8,
    U8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMmaLayout {
    Row,
    Col,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMmaOverflow {
    NotApplicable,
    Wrapping,
    Satfinite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMmaMetadata {
    Standard,
    Ordered,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMmaSelector {
    ImmediateZeroOrOne,
    ImmediateZero,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMmaParticipation {
    AllWarpLanesSameInstructionAndQualifiersNoExitedLanes,
}

/// Rust `C, A, B, metadata, selector` shape used by the importer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMmaAdapter {
    C2U32A4U32B4U32MetadataU32SelectorU32ToD2U32,
    C4F32A4U32B4U32MetadataU32SelectorU32ToD4F32,
    C4I32A2U32B2U32MetadataU32SelectorU32ToD4I32,
    C4I32A4U32B4U32MetadataU32SelectorU32ToD4I32,
}

/// LLVM `A, B, C, metadata, selector` shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMmaLlvmAdapter {
    A4I32B4I32C2V2F16MetadataI32SelectorI32ToD2V2F16,
    A4I32B4I32C4F32MetadataI32SelectorI32ToD4F32,
    A2I32B2I32C4I32MetadataI32SelectorI32ToD4I32,
    A4I32B4I32C4I32MetadataI32SelectorI32ToD4I32,
}

/// Where the stable `cuda_device::wmma` callable is defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SparseMmaCompatibilitySource {
    GeneratedStub,
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

/// Closed contract for scalar floating-point arithmetic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarArithmetic {
    pub format: ScalarArithmeticFormat,
    pub operation: ScalarArithmeticOperation,
    pub rounding: ScalarArithmeticRounding,
    pub subnormal: ScalarArithmeticSubnormal,
    pub saturation: ScalarArithmeticSaturation,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarArithmeticFormat {
    F32,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarArithmeticOperation {
    Mul,
    Div,
    Fma,
    Add,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarArithmeticRounding {
    Rn,
    Rz,
    Rm,
    Rp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarArithmeticSubnormal {
    Preserve,
    Ftz,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarArithmeticSaturation {
    None,
    Sat,
}

/// Closed identity and carrier contract for extended floating-point min/max.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtendedMinMax {
    pub format: ExtendedMinMaxFormat,
    pub operation: ExtendedMinMaxOperation,
    pub subnormal: ExtendedMinMaxSubnormal,
    pub nan: ExtendedMinMaxNan,
    pub xorsign_abs: bool,
    pub adapter: ExtendedMinMaxAdapter,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtendedMinMaxFormat {
    F32,
    F16x2,
    Bf16x2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtendedMinMaxOperation {
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtendedMinMaxSubnormal {
    Preserve,
    Ftz,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtendedMinMaxNan {
    Number,
    Nan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtendedMinMaxAdapter {
    DirectF32,
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
    E4m3x2,
    E5m2x2,
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
    Satfinite,
    SatfiniteRelu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedConversionAdapter {
    ReverseHighLowOperands,
}

/// Closed contract for one scalar floating-point conversion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarConversion {
    pub source_format: ScalarConversionSourceFormat,
    pub destination_format: ScalarConversionDestinationFormat,
    pub rounding: ScalarConversionRounding,
    pub saturation: ScalarConversionSaturation,
    pub result_representation: ScalarConversionResultRepresentation,
    pub adapter: ScalarConversionAdapter,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarConversionSourceFormat {
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarConversionDestinationFormat {
    Tf32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarConversionRounding {
    NearestAway,
    NearestEven,
    TowardZero,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarConversionSaturation {
    None,
    Relu,
    Satfinite,
    ReluSatfinite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarConversionResultRepresentation {
    RawU32Bits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarConversionAdapter {
    DirectF32ToRawU32Bits,
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
    #[serde(rename = "pointer_count_to_void")]
    InitPointerCountToVoid,
    #[serde(rename = "pointer_to_token")]
    ArrivePointerToToken,
    #[serde(rename = "pointer_token_to_predicate")]
    TestWaitPointerTokenToPredicate,
    #[serde(rename = "pointer_to_void")]
    InvalPointerToVoid,
}

/// Closed contract for the remaining handwritten mbarrier operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MbarrierExtended {
    pub operation: MbarrierExtendedOperation,
    pub adapter: MbarrierExtendedAdapter,
    pub source_contract: MbarrierExtendedSourceContract,
    pub runtime_validation: RuntimeValidation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MbarrierExtendedAdapter {
    PointerTxCountBytesToTokenDroppingTxCount,
    RawClusterAddressToVoid,
    PointerTokenToPredicate,
    PointerParityToPredicate,
    ZeroOperandsToVoid,
    NanosecondsToVoid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MbarrierExtendedSourceContract {
    LlvmImported,
    PtxNativeRawClusterAddress,
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

/// Schema-6 evidence before matrix expansion.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceFileV6 {
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
    #[serde(default)]
    pub defaults: EvidenceRecordDefaults,
    #[serde(default)]
    pub fixtures: Vec<EvidenceFixture>,
    #[serde(default)]
    pub matrices: Vec<EvidenceMatrix>,
    #[serde(default)]
    pub records: Vec<EvidenceRecord>,
}

/// Facts shared by every record in one evidence matrix.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRecordDefaults {
    #[serde(default)]
    pub resolved_llvm_symbol: Option<String>,
    #[serde(default)]
    pub llvm_arguments: Option<Vec<String>>,
    #[serde(default)]
    pub llvm_results: Option<Vec<String>>,
    #[serde(default)]
    pub concrete_llvm_arguments: Option<Vec<String>>,
    #[serde(default)]
    pub concrete_llvm_results: Option<Vec<String>>,
    #[serde(default)]
    pub target_triple: Option<String>,
    #[serde(default)]
    pub gpu_target: Option<String>,
    #[serde(default)]
    pub ptx_feature: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub stages: Vec<EvidenceStage>,
    #[serde(default)]
    pub declaration_attributes_canonicalized: Option<bool>,
    #[serde(default)]
    pub runtime_validation: Option<RuntimeValidation>,
}

/// One shared fixture and the number of records it covers.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceFixture {
    pub id: String,
    pub coverage_count: usize,
    pub stages: Vec<EvidenceStage>,
}

/// One Cartesian evidence matrix.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceMatrix {
    pub axes: Vec<EvidenceMatrixAxis>,
    pub product_count: usize,
    #[serde(default)]
    pub fixtures: Vec<String>,
    pub template: EvidenceMatrixTemplate,
}

/// One named matrix axis.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceMatrixAxis {
    pub name: String,
    pub values: Vec<String>,
}

/// Identity and matrix-specific facts for one record template.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceMatrixTemplate {
    pub id: String,
    #[serde(default)]
    pub source: Option<IntrinsicSource>,
    #[serde(default)]
    pub source_record: Option<String>,
    #[serde(deserialize_with = "deserialize_required_optional_string")]
    pub llvm_symbol: Option<String>,
    pub expected_ptx: InstructionPattern,
    #[serde(default)]
    pub facts: EvidenceRecordDefaults,
}

fn deserialize_required_optional_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer)
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
    pub scalar_conversion: Option<ScalarConversion>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scalar_arithmetic: Option<ScalarArithmetic>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extended_minmax: Option<ExtendedMinMax>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cp_async_copy: Option<CpAsyncCopy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cp_async_control: Option<CpAsyncControl>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cp_async_mbarrier: Option<CpAsyncMbarrier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mbarrier_basic: Option<MbarrierBasic>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub movmatrix: Option<Movmatrix>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mbarrier_extended: Option<MbarrierExtended>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub register_mma: Option<RegisterMma>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparse_mma: Option<SparseMma>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prmt: Option<Prmt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_barrier: Option<ClusterBarrier>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wgmma_control: Option<WgmmaControl>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub special_register: Option<SpecialRegister>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub debug_control: Option<DebugControl>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_memory: Option<ClusterMemory>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clc: Option<Clc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tma: Option<Tma>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tcgen05: Option<Tcgen05>,
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
/// Exact `a` and `f` targets stay distinct from monotonic minimums. A target
/// matrix also keeps each hardware target paired with its PTX floor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CatalogHardwareTarget {
    All,
    AnyOf {
        alternatives: Vec<CatalogHardwareAlternative>,
    },
    /// Closed selector contracts with their exact PTX and hardware pairs.
    TargetMatrix {
        contracts: Vec<CatalogTargetContract>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CatalogHardwareAlternative {
    MinimumSm { sm: u16 },
    ExactArchitecture { sm: u16 },
    FamilyTarget { sm: u16 },
}

/// One PTX floor paired with one hardware alternative.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogTargetAlternative {
    pub minimum_ptx: PtxVersion,
    pub hardware: CatalogHardwareAlternative,
}

/// One selector tuple and its exact PTX and hardware pairs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogTargetContract {
    pub selectors: Vec<TargetSelectorBinding>,
    pub alternatives: Vec<CatalogTargetAlternative>,
}

/// One field/value pair that selects a target contract.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetSelectorBinding {
    pub name: String,
    pub value: String,
}

/// One selector-specific target contract from admission policy.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetContract {
    #[serde(default)]
    pub selectors: Vec<TargetSelectorBinding>,
    pub alternatives: Vec<TargetContractAlternative>,
}

/// One target spelling and PTX floor.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetContractAlternative {
    pub target: String,
    pub minimum_ptx: String,
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
        llvm_symbol_to_identifier(&self.llvm.as_ref().expect("LLVM-backed intrinsic").symbol)
    }

    pub fn resolved_llvm_identifier(&self) -> String {
        let llvm = self.llvm.as_ref().expect("LLVM-backed intrinsic");
        llvm_symbol_to_identifier(llvm.resolved_symbol.as_deref().unwrap_or(&llvm.symbol))
    }
}

fn llvm_symbol_to_identifier(symbol: &str) -> String {
    if !symbol.contains('_') {
        return symbol.replace('.', "_");
    }

    let suffix = symbol
        .strip_prefix("llvm.")
        .expect("LLVM intrinsic symbol must start with llvm.");
    let mut output = String::from("llvm__");
    for ch in suffix.chars() {
        match ch {
            '.' => output.push_str("_d"),
            '_' => output.push_str("_u"),
            ch if ch.is_ascii_alphanumeric() => output.push(ch),
            _ => panic!("LLVM intrinsic symbol contains an unsupported character"),
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llvm_identifier_encoding_preserves_literal_underscores() {
        assert_eq!(
            llvm_symbol_to_identifier("llvm.nvvm.read.ptx.sreg.tid.x"),
            "llvm_nvvm_read_ptx_sreg_tid_x"
        );
        assert_eq!(
            llvm_symbol_to_identifier("llvm.nvvm.wgmma.wait_group.sync.aligned"),
            "llvm__nvvm_dwgmma_dwait_ugroup_dsync_daligned"
        );
        assert_eq!(
            llvm_symbol_to_identifier(
                "llvm.nvvm.ldmatrix.sync.aligned.m16n16.x1.trans.b8x16.b4x16_p64.p3"
            ),
            "llvm__nvvm_dldmatrix_dsync_daligned_dm16n16_dx1_dtrans_db8x16_db4x16_up64_dp3"
        );
        assert_eq!(
            llvm_symbol_to_identifier(
                "llvm.nvvm.ldmatrix.sync.aligned.m8n16.x4.b8x16.b6x16_p32.p3"
            ),
            "llvm__nvvm_dldmatrix_dsync_daligned_dm8n16_dx4_db8x16_db6x16_up32_dp3"
        );
    }

    #[test]
    fn ldmatrix_register_count_includes_the_shape_width() {
        let variant = |shape, multiplicity| LdmatrixVariant {
            shape,
            multiplicity,
            layout: LdmatrixLayout::Normal,
            element: LdmatrixElement::B8,
            state_space: LdmatrixStateSpace::Shared,
        };

        assert_eq!(
            variant(LdmatrixShape::M8n16, LdmatrixMultiplicity::X4).register_count(),
            4
        );
        assert_eq!(
            variant(LdmatrixShape::M16n16, LdmatrixMultiplicity::X1).register_count(),
            2
        );
        assert_eq!(
            variant(LdmatrixShape::M16n16, LdmatrixMultiplicity::X2).register_count(),
            4
        );
    }

    #[test]
    fn blackwell_ldmatrix_address_contracts_keep_readable_widths_distinct() {
        assert_eq!(
            serde_json::from_str::<LdmatrixAddressContract>(
                r#""warp_lane_addresses_mapped_by_multiplicity_sixteen_byte_aligned_sixteen_bytes_readable""#
            )
            .unwrap(),
            LdmatrixAddressContract::WarpLaneAddressesMappedByMultiplicitySixteenByteAlignedSixteenBytesReadable
        );
        assert_eq!(
            serde_json::from_str::<LdmatrixAddressContract>(
                r#""warp_lane_addresses_mapped_by_multiplicity_sixteen_byte_aligned_thirty_two_bytes_readable""#
            )
            .unwrap(),
            LdmatrixAddressContract::WarpLaneAddressesMappedByMultiplicitySixteenByteAlignedThirtyTwoBytesReadable
        );
    }

    #[test]
    fn wgmma_control_contract_is_closed() {
        let parsed: WgmmaControl = serde_json::from_str(
            r#"{
                "mode": "wait_group",
                "adapter": "const_generic_u32_to_i64_immediate",
                "participation": "warpgroup_all_threads_same_instruction"
            }"#,
        )
        .unwrap();
        assert_eq!(parsed.mode, WgmmaControlMode::WaitGroup);
        assert_eq!(
            parsed.adapter,
            WgmmaControlAdapter::ConstGenericU32ToI64Immediate
        );
        assert_eq!(
            parsed.participation,
            WgmmaControlParticipation::WarpgroupAllThreadsSameInstruction
        );

        let open_ended = r#"{
            "mode": "wait_group",
            "adapter": "const_generic_u32_to_i64_immediate",
            "participation": "warpgroup_all_threads_same_instruction",
            "custom_ptx": "wgmma.wait_group.sync.aligned 0;"
        }"#;
        assert!(serde_json::from_str::<WgmmaControl>(open_ended).is_err());
    }

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
            MbarrierBasicAdapter::TestWaitPointerTokenToPredicate
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
    fn movmatrix_contract_rejects_open_ended_policy() {
        let valid = r#"
participation = "all_warp_lanes_same_instruction_no_exited_lanes"
adapter = "packed_b16x2_u32_to_packed_b16x2_u32"
runtime_validation = "unexecuted"
"#;
        let parsed = toml::from_str::<Movmatrix>(valid).unwrap();
        assert_eq!(
            parsed.participation,
            MovmatrixParticipation::AllWarpLanesSameInstructionNoExitedLanes
        );

        for invalid in [
            valid.replace(
                "all_warp_lanes_same_instruction_no_exited_lanes",
                "participating_lanes",
            ),
            valid.replace("packed_b16x2_u32_to_packed_b16x2_u32", "direct_u32"),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(toml::from_str::<Movmatrix>(&invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn mbarrier_extended_contract_rejects_open_ended_policy() {
        let valid = r#"
operation = "arrive_expect_tx_cta"
adapter = "pointer_tx_count_bytes_to_token_dropping_tx_count"
source_contract = "llvm_imported"
runtime_validation = "unexecuted"
"#;
        let parsed = toml::from_str::<MbarrierExtended>(valid).unwrap();
        assert_eq!(
            parsed.adapter,
            MbarrierExtendedAdapter::PointerTxCountBytesToTokenDroppingTxCount
        );
        assert_eq!(
            parsed.source_contract,
            MbarrierExtendedSourceContract::LlvmImported
        );

        for invalid in [
            valid.replace("llvm_imported", "auto"),
            valid.replace(
                "pointer_tx_count_bytes_to_token_dropping_tx_count",
                "direct",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(
                toml::from_str::<MbarrierExtended>(&invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn register_mma_contract_parses_unsigned_k16_generated_stub() {
        let valid = r#"
shape = "m16n8k16"
operation = "multiply"
accumulator = "s32"
a_element = "s8"
b_element = "u8"
a_layout = "row"
b_layout = "col"
overflow = "satfinite"
participation = "all_warp_lanes_same_instruction_and_qualifiers_no_exited_lanes"
adapter = "c4_i32_a2_u32_b1_u32_to_d4_i32"
compatibility_source = "generated_stub"
runtime_validation = "unexecuted"
"#;
        let parsed = toml::from_str::<RegisterMma>(valid).unwrap();
        assert_eq!(parsed.b_element, RegisterMmaElement::U8);
        assert_eq!(parsed.adapter, RegisterMmaAdapter::C4I32A2U32B1U32ToD4I32);
        assert_eq!(
            parsed.compatibility_source,
            RegisterMmaCompatibilitySource::GeneratedStub
        );

        for invalid in [
            valid.replace("b_element = \"u8\"", "b_element = \"i8\""),
            valid.replace(
                "adapter = \"c4_i32_a2_u32_b1_u32_to_d4_i32\"",
                "adapter = \"direct\"",
            ),
            valid.replace(
                "compatibility_source = \"generated_stub\"",
                "compatibility_source = \"automatic\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(
                toml::from_str::<RegisterMma>(&invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn sparse_mma_contract_closes_the_selector_and_metadata_modes() {
        let valid = r#"
shape = "m16n8k32"
accumulator = "s32"
a_element = "s8"
b_element = "u8"
a_layout = "row"
b_layout = "col"
overflow = "satfinite"
metadata = "standard"
selector = "immediate_zero_or_one"
participation = "all_warp_lanes_same_instruction_and_qualifiers_no_exited_lanes"
adapter = "c4_i32_a2_u32_b2_u32_metadata_u32_selector_u32_to_d4_i32"
llvm_adapter = "a2_i32_b2_i32_c4_i32_metadata_i32_selector_i32_to_d4_i32"
compatibility_source = "generated_stub"
runtime_validation = "unexecuted"
"#;
        let parsed = toml::from_str::<SparseMma>(valid).unwrap();
        assert_eq!(parsed.metadata, SparseMmaMetadata::Standard);
        assert_eq!(parsed.selector, SparseMmaSelector::ImmediateZeroOrOne);

        let ordered = valid.replace("metadata = \"standard\"", "metadata = \"ordered\"");
        assert_eq!(
            toml::from_str::<SparseMma>(&ordered).unwrap().metadata,
            SparseMmaMetadata::Ordered
        );

        let k64 = ordered
            .replace("shape = \"m16n8k32\"", "shape = \"m16n8k64\"")
            .replace(
                "selector = \"immediate_zero_or_one\"",
                "selector = \"immediate_zero\"",
            )
            .replace(
                "adapter = \"c4_i32_a2_u32_b2_u32_metadata_u32_selector_u32_to_d4_i32\"",
                "adapter = \"c4_i32_a4_u32_b4_u32_metadata_u32_selector_u32_to_d4_i32\"",
            )
            .replace(
                "llvm_adapter = \"a2_i32_b2_i32_c4_i32_metadata_i32_selector_i32_to_d4_i32\"",
                "llvm_adapter = \"a4_i32_b4_i32_c4_i32_metadata_i32_selector_i32_to_d4_i32\"",
            );
        let parsed_k64 = toml::from_str::<SparseMma>(&k64).unwrap();
        assert_eq!(parsed_k64.shape, SparseMmaShape::M16n8k64);
        assert_eq!(parsed_k64.selector, SparseMmaSelector::ImmediateZero);
        assert_eq!(
            parsed_k64.adapter,
            SparseMmaAdapter::C4I32A4U32B4U32MetadataU32SelectorU32ToD4I32
        );
        assert_eq!(
            parsed_k64.llvm_adapter,
            SparseMmaLlvmAdapter::A4I32B4I32C4I32MetadataI32SelectorI32ToD4I32
        );

        let int4 = ordered
            .replace("shape = \"m16n8k32\"", "shape = \"m16n8k64\"")
            .replace("a_element = \"s8\"", "a_element = \"s4\"")
            .replace("b_element = \"u8\"", "b_element = \"u4\"");
        let parsed_int4 = toml::from_str::<SparseMma>(&int4).unwrap();
        assert_eq!(parsed_int4.a_element, SparseMmaElement::S4);
        assert_eq!(parsed_int4.b_element, SparseMmaElement::U4);

        let k128_int4 = int4
            .replace("shape = \"m16n8k64\"", "shape = \"m16n8k128\"")
            .replace(
                "selector = \"immediate_zero_or_one\"",
                "selector = \"immediate_zero\"",
            )
            .replace(
                "adapter = \"c4_i32_a2_u32_b2_u32_metadata_u32_selector_u32_to_d4_i32\"",
                "adapter = \"c4_i32_a4_u32_b4_u32_metadata_u32_selector_u32_to_d4_i32\"",
            )
            .replace(
                "llvm_adapter = \"a2_i32_b2_i32_c4_i32_metadata_i32_selector_i32_to_d4_i32\"",
                "llvm_adapter = \"a4_i32_b4_i32_c4_i32_metadata_i32_selector_i32_to_d4_i32\"",
            );
        let parsed_k128_int4 = toml::from_str::<SparseMma>(&k128_int4).unwrap();
        assert_eq!(parsed_k128_int4.shape, SparseMmaShape::M16n8k128);
        assert_eq!(parsed_k128_int4.selector, SparseMmaSelector::ImmediateZero);

        for invalid in [
            valid.replace(
                "selector = \"immediate_zero_or_one\"",
                "selector = \"runtime\"",
            ),
            valid.replace("metadata = \"standard\"", "metadata = \"unreviewed\""),
            valid.replace(
                "llvm_adapter = \"a2_i32_b2_i32_c4_i32_metadata_i32_selector_i32_to_d4_i32\"",
                "llvm_adapter = \"c_then_a_then_b\"",
            ),
            valid.replace("shape = \"m16n8k32\"", "shape = \"m16n8k256\""),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(toml::from_str::<SparseMma>(&invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn sparse_mma_admission_accepts_the_canonical_name_and_legacy_alias() {
        let canonical = r#"
schema = 25
family = "sparse_mma"

[sparse_mma_integer]
llvm_evidence_profile = "llvm"
libnvvm_evidence_profile = "libnvvm"
runtime_validation = "unexecuted"
metadata = "ordered"

[[sparse_mma_integer.variant]]
shape = "m16n8k64"
a_element = "s4"
b_element = "u4"
overflow = "wrapping"
"#;
        let parsed = toml::from_str::<OverlayShardFile>(canonical).unwrap();
        assert_eq!(
            parsed.sparse_mma_integer.unwrap().variants[0].b_element,
            SparseMmaElement::U4
        );

        let legacy = canonical.replace("sparse_mma_integer", "sparse_mma_int8");
        assert!(
            toml::from_str::<OverlayShardFile>(&legacy)
                .unwrap()
                .sparse_mma_integer
                .is_some()
        );
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

    #[test]
    fn cluster_barrier_contract_rejects_open_ended_semantics() {
        let valid = r#"
mode = "arrive_relaxed_aligned"
ordering = "relaxed"
aligned = true
"#;
        let parsed = toml::from_str::<ClusterBarrier>(valid).unwrap();
        assert_eq!(parsed.mode, ClusterBarrierMode::ArriveRelaxedAligned);
        assert_eq!(parsed.ordering, ClusterBarrierOrdering::Relaxed);
        assert!(parsed.aligned);

        for invalid in [
            valid.replace("ordering = \"relaxed\"", "ordering = \"unordered\""),
            valid.replace("aligned = true", "aligned = \"sometimes\""),
            valid.replace(
                "mode = \"arrive_relaxed_aligned\"",
                "mode = \"arrive_release_aligned\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(
                toml::from_str::<ClusterBarrier>(&invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn debug_control_contract_rejects_open_ended_policy() {
        let valid = r#"
operation = "pmevent"
adapter = "const_generic_to_immediate_u32"
runtime_validation = "unexecuted"
"#;
        let parsed = toml::from_str::<DebugControl>(valid).unwrap();
        assert_eq!(parsed.operation, DebugControlOperation::Pmevent);
        assert_eq!(
            parsed.adapter,
            DebugControlAdapter::ConstGenericToImmediateU32
        );

        for invalid in [
            valid.replace("operation = \"pmevent\"", "operation = \"profiler\""),
            valid.replace(
                "adapter = \"const_generic_to_immediate_u32\"",
                "adapter = \"runtime_u32\"",
            ),
            valid.replace(
                "runtime_validation = \"unexecuted\"",
                "runtime_validation = \"assumed\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(
                toml::from_str::<DebugControl>(&invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn cluster_memory_contract_rejects_open_ended_policy() {
        let valid = r#"
operation = "map_shared_rank"
adapter = "generic_const_and_mut_pointer_rank_to_same_pointer"
source_contract = "llvm_mapa_shared_cluster_as7_identity_inline_ptx"
runtime_validation = "unexecuted"
"#;
        let parsed = toml::from_str::<ClusterMemory>(valid).unwrap();
        assert_eq!(parsed.operation, ClusterMemoryOperation::MapSharedRank);
        assert_eq!(
            parsed.source_contract,
            ClusterMemorySourceContract::LlvmMapaSharedClusterAs7IdentityInlinePtx
        );

        for invalid in [
            valid.replace("map_shared_rank", "map_generic_rank"),
            valid.replace(
                "generic_const_and_mut_pointer_rank_to_same_pointer",
                "direct",
            ),
            valid.replace(
                "llvm_mapa_shared_cluster_as7_identity_inline_ptx",
                "llvm_typed_as3",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(
                toml::from_str::<ClusterMemory>(&invalid).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn clc_contract_rejects_open_ended_policy() {
        let valid = r#"
operation = "query_is_canceled"
adapter = "pair_u64_to_i128_bool_to_u32"
runtime_validation = "unexecuted"
"#;
        let parsed = toml::from_str::<Clc>(valid).unwrap();
        assert_eq!(parsed.operation, ClcOperation::QueryIsCanceled);
        assert_eq!(parsed.adapter, ClcAdapter::PairU64ToI128BoolToU32);

        for invalid in [
            valid.replace("operation = \"query_is_canceled\"", "operation = \"query\""),
            valid.replace(
                "adapter = \"pair_u64_to_i128_bool_to_u32\"",
                "adapter = \"pair_u64_to_i128\"",
            ),
            valid.replace(
                "runtime_validation = \"unexecuted\"",
                "runtime_validation = \"assumed\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(toml::from_str::<Clc>(&invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn tma_contract_rejects_open_ended_policy() {
        let valid = r#"
operation = "g2s_tile2d_multicast"
adapter = "g2s_pointers_coordinates_barrier_mask_inject_defaults"
runtime_validation = "unexecuted"
"#;
        let parsed = toml::from_str::<Tma>(valid).unwrap();
        assert_eq!(parsed.operation, TmaOperation::G2sTile2dMulticast);
        assert_eq!(
            parsed.adapter,
            TmaAdapter::G2sPointersCoordinatesBarrierMaskInjectDefaults
        );

        for invalid in [
            valid.replace("g2s_tile2d_multicast", "g2s_multicast"),
            valid.replace(
                "g2s_pointers_coordinates_barrier_mask_inject_defaults",
                "direct",
            ),
            valid.replace(
                "runtime_validation = \"unexecuted\"",
                "runtime_validation = \"assumed\"",
            ),
            format!("{valid}unreviewed = true\n"),
        ] {
            assert!(toml::from_str::<Tma>(&invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn tcgen05_execution_scope_is_closed() {
        for operation in [
            Tcgen05Operation::Alloc,
            Tcgen05Operation::Dealloc,
            Tcgen05Operation::RelinquishAllocPermit,
            Tcgen05Operation::AllocCg2,
            Tcgen05Operation::DeallocCg2,
            Tcgen05Operation::RelinquishAllocPermitCg2,
            Tcgen05Operation::Ld16x256bX8Pure,
            Tcgen05Operation::Ld16x256bPure,
            Tcgen05Operation::LoadWait,
            Tcgen05Operation::StoreWait,
            Tcgen05Operation::Ld,
            Tcgen05Operation::St,
        ] {
            assert_eq!(operation.execution_scope(), "warp");
        }
        for operation in [
            Tcgen05Operation::FenceBeforeThreadSync,
            Tcgen05Operation::FenceAfterThreadSync,
            Tcgen05Operation::Commit,
            Tcgen05Operation::CommitSharedCluster,
            Tcgen05Operation::MmaWsF16,
            Tcgen05Operation::MmaWsBf16,
            Tcgen05Operation::MmaWsTf32,
            Tcgen05Operation::MmaF16,
            Tcgen05Operation::CpSmemToTmem,
            Tcgen05Operation::MmaF16Cg2,
            Tcgen05Operation::CommitCg2,
            Tcgen05Operation::CommitSharedClusterCg2,
            Tcgen05Operation::CommitMulticastCg2,
            Tcgen05Operation::CpSmemToTmemCg2,
            Tcgen05Operation::CommitMulticast,
            Tcgen05Operation::ShiftDown,
            Tcgen05Operation::ShiftDownCg2,
        ] {
            assert_eq!(operation.execution_scope(), "thread");
        }
    }
}
