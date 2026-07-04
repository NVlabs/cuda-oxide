/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Architecture feature detection and PTX target selection.
//!
//! Detects the architecture and PTX-ISA requirements of exported LLVM IR and
//! selects the minimum `sm_XX` that can lower them. The backend owns this so an
//! experimental frontend gets the same target selection as the Rust MIR path
//! in `mir-importer`.

use crate::error::PipelineError;
use libnvvm_sys::CudaArch;
use std::path::Path;

fn contains_wgmma_features(contents: &str) -> bool {
    contents.contains("wgmma.fence")
        || contents.contains("wgmma.commit_group")
        || contents.contains("wgmma.wait_group")
        || contents.contains("wgmma.mma_async")
}

/// Checks for Thread Block Cluster instructions (sm_90+).
///
/// Cluster features require Hopper (sm_90) or newer:
/// - Cluster special registers (%cluster_ctaid, %cluster_nctaid)
/// - Cluster synchronization (cluster.sync)
/// - Distributed shared memory (mapa.shared::cluster)
fn contains_cluster_features(contents: &str) -> bool {
    // Cluster special registers
    contents.contains("cluster_ctaid")
        || contents.contains("cluster_nctaid")
        || contents.contains("cluster_ctarank")
        || contents.contains("cluster_nctarank")
        || contents.contains("%clusterid")
        || contents.contains("%nclusterid")
        || contents.contains("%is_explicit_cluster")
        || contents.contains("!\"cluster_dim_x\"")
        || contents.contains("!\"cluster_dim_y\"")
        || contents.contains("!\"cluster_dim_z\"")
        // Cluster synchronization
        || contents.contains("cluster.sync")
        || contents.contains("barrier.cluster.")
        // Distributed shared memory
        || contents.contains("mapa.shared::cluster")
        || contents.contains(".shared::cluster")
        || contains_cluster_fence_features(contents)
        || contains_cluster_scoped_memory_features(contents)
}

fn contains_cluster_fence_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("fence.sc.cluster")
            || statement.contains("fence.acq_rel.cluster")
            || statement.contains("fence.acquire.cluster")
            || statement.contains("fence.release.cluster")
    })
}

fn contains_cluster_scoped_memory_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        !statement.contains("multimem.")
            && statement.contains(".cluster.")
            && ["ld.", "st.", "atom.", "red."]
                .iter()
                .any(|mnemonic| statement.contains(mnemonic))
    })
}

/// Checks the one-way fence semantics added in PTX 8.6.
///
/// Unlike the older `.sc` / `.acq_rel` forms, `.acquire` and `.release`
/// require sm_90 for every scope, not just `.cluster`.
fn contains_fence_acquire_release_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("fence.acquire.") || statement.contains("fence.release.")
    })
}

/// Checks the multimem instruction family introduced for sm_90.
///
/// Base forms need PTX 8.1. The pipeline currently has no 8.1 feature switch,
/// so PTX 8.6 is the nearest conservative version supported by LLVM.
fn contains_multimem_features(contents: &str) -> bool {
    contents.split(';').any(is_multimem_instruction)
}

fn is_multimem_instruction(statement: &str) -> bool {
    ["multimem.ld_reduce", "multimem.st", "multimem.red"]
        .iter()
        .any(|instruction| statement.contains(instruction))
}

/// Checks PTX 8.6 multimem formats that require a Blackwell family target.
fn contains_multimem_blackwell_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        is_multimem_instruction(statement)
            && [".e4m3", ".e5m2", ".acc::f16"]
                .iter()
                .any(|qualifier| statement.contains(qualifier))
    })
}

/// Checks the PTX 8.6 floating-point extension to `redux.sync`.
fn contains_redux_f32_features(contents: &str) -> bool {
    contents
        .split(';')
        .any(|statement| statement.contains("redux.sync") && statement.contains(".f32"))
}

/// Checks for forward-compatible instructions whose minimum target is sm_90.
///
/// Keep this category architecture-neutral: unlike WGMMA, these instructions
/// are not Hopper-specific and remain available on newer architectures.
fn contains_sm90_features(contents: &str) -> bool {
    ["add.rn.bf16x2", "sub.rn.bf16x2", "mul.rn.bf16x2"]
        .iter()
        .any(|mnemonic| contains_instruction_mnemonic(contents, mnemonic))
        || contains_packed_bf16_atomic_features(contents)
        || contains_stmatrix_features(contents)
        || contains_elect_features(contents)
        || contains_fence_acquire_release_features(contents)
        || contains_multimem_features(contents)
}

/// Native packed bf16 atomic add was added in PTX 7.8 for sm_90.
fn contains_packed_bf16_atomic_features(contents: &str) -> bool {
    contains_instruction_mnemonic(contents, "atom.global.add.noftz.bf16x2")
}

/// Packed f16 atomic add needs PTX 6.2. Its hardware floor predates
/// cuda-oxide's Volta floor, so only the independent PTX ISA requirement must
/// be raised.
fn contains_packed_f16_atomic_features(contents: &str) -> bool {
    contains_instruction_mnemonic(contents, "atom.global.add.noftz.f16x2")
}

fn contains_elect_features(contents: &str) -> bool {
    contents.contains("elect.sync")
}

/// Checks for the register-only 8x8 matrix transpose (PTX 7.8, sm_75+).
fn contains_movmatrix_features(contents: &str) -> bool {
    contains_instruction_mnemonic(contents, "movmatrix.sync.aligned.m8n8.trans.b16")
}

/// Checks the dense BF16 MMA form added by the typed device intrinsic.
///
/// MMA shapes and types have different architecture and PTX ISA floors, so
/// this intentionally matches the complete operation-specific mnemonic.
fn contains_mma_m16n8k16_f32_bf16_features(contents: &str) -> bool {
    contains_instruction_mnemonic(
        contents,
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
    )
}

/// Checks for the Ampere TF32 MMA operation (PTX 7.0, sm_80+).
fn contains_mma_m16n8k8_f32_tf32_features(contents: &str) -> bool {
    contains_instruction_mnemonic(
        contents,
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32",
    )
}

/// Checks for the Ampere INT8 MMA operation (PTX 7.0, sm_80+).
///
/// PTX permits both wrapping and `.satfinite` accumulator-overflow behavior.
/// Match each complete legal mnemonic so a qualifier near-miss cannot raise
/// the module target accidentally.
fn contains_mma_m16n8k32_s32_s8_features(contents: &str) -> bool {
    [
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32",
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32",
    ]
    .into_iter()
    .any(|mnemonic| contains_instruction_mnemonic(contents, mnemonic))
}

/// Checks for the Ampere FP64 tensor-core MMA operation (PTX 7.0, sm_80+).
fn contains_mma_m8n8k4_f64_features(contents: &str) -> bool {
    contains_instruction_mnemonic(contents, "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64")
}

/// Checks the dense F16 MMA form added by the typed device intrinsic.
///
/// MMA shapes and types have different architecture and PTX ISA floors, so
/// this intentionally matches the complete operation-specific mnemonic.
fn contains_mma_m16n8k16_f32_f16_features(contents: &str) -> bool {
    contains_instruction_mnemonic(
        contents,
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
    )
}

fn contains_instruction_mnemonic(contents: &str, mnemonic: &str) -> bool {
    contents.match_indices(mnemonic).any(|(index, _)| {
        let preceding = &contents[..index];
        let following = &contents[index + mnemonic.len()..];
        let escapes = ["\\09", "\\0A", "\\0B", "\\0C", "\\0D"];
        // Use PTX token delimiters rather than treating arbitrary punctuation
        // as a boundary. In particular, `$` and `%` participate in PTX
        // identifiers, and guarded opcodes have whitespace after `@{!}p`.
        let begins_at_instruction_boundary = preceding.is_empty()
            || preceding
                .chars()
                .next_back()
                .is_some_and(|ch| ch.is_whitespace() || matches!(ch, '"' | ';' | ':' | '{' | '}'))
            || escapes.iter().any(|escape| preceding.ends_with(escape))
            || preceding.ends_with("*/");
        let ends_at_instruction_boundary =
            following.chars().next().is_some_and(char::is_whitespace)
                || escapes.iter().any(|escape| following.starts_with(escape));
        begins_at_instruction_boundary && ends_at_instruction_boundary
    })
}

/// Checks the full PTX instruction families, including inline `ptx_asm!`
/// forms that cuda-oxide does not yet expose as typed wrappers.
///
/// Broad family matching is intentional. Missing a valid spelling can
/// silently select an architecture or PTX ISA that is too old; an invalid
/// spelling still reaches ptxas and fails there after conservative targeting.
fn contains_ldmatrix_features(contents: &str) -> bool {
    contents.contains("ldmatrix.sync.aligned.")
}

fn contains_stmatrix_features(contents: &str) -> bool {
    contents.contains("stmatrix.sync.aligned.")
}

/// PTX 8.6 matrix shapes/types have a Blackwell architecture-family floor.
fn contains_blackwell_matrix_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        let newer_ldmatrix = statement.contains("ldmatrix.sync.aligned.")
            && [".m16n16.", ".m8n16.", ".b8", ".src_fmt", ".dst_fmt"]
                .iter()
                .any(|token| statement.contains(token));
        let newer_stmatrix = statement.contains("stmatrix.sync.aligned.")
            && [".m16n8.", ".b8"]
                .iter()
                .any(|token| statement.contains(token));
        newer_ldmatrix || newer_stmatrix
    })
}

fn contains_ldmatrix_cta_state_space(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("ldmatrix.sync.aligned.") && statement.contains(".shared::cta.")
    })
}

/// Checks for features whose minimum target is sm_80.
///
/// This category includes packed bf16 operations introduced on Ampere and
/// non-bulk asynchronous copies. Match both the PTX spellings used in inline
/// assembly and the dotted LLVM NVVM intrinsic names for `cp.async`. Bulk and
/// tensor-copy forms have stronger requirements and are classified first.
fn contains_sm80_features(contents: &str) -> bool {
    [
        "fma.rn.bf16x2",
        "fma.rn.relu.bf16x2",
        "min.bf16x2",
        "max.bf16x2",
        "neg.bf16x2",
        "abs.bf16x2",
    ]
    .iter()
    .any(|mnemonic| contains_instruction_mnemonic(contents, mnemonic))
        || contains_mma_m8n8k4_f64_features(contents)
        || contents
            .split(';')
            .any(|statement| statement.contains("cvt.") && statement.contains(".bf16x2.f32"))
        || contains_mbarrier_features(contents)
        || contents.contains("redux.sync")
        || contents.contains("cp.async.ca.shared")
        || contents.contains("cp.async.cg.shared")
        || contents.contains("cp.async.commit_group")
        || contents.contains("cp.async.commit.group")
        || contents.contains("cp.async.wait_group")
        || contents.contains("cp.async.wait.group")
        || contents.contains("cp.async.wait_all")
        || contents.contains("cp.async.wait.all")
        || contains_mma_m16n8k16_f32_bf16_features(contents)
        || contains_mma_m16n8k16_f32_f16_features(contents)
        || contains_mma_m16n8k8_f32_tf32_features(contents)
        || contains_mma_m16n8k32_s32_s8_features(contents)
}

/// Checks for TMA/mbarrier instructions (Hopper+ compatible with Blackwell).
///
/// These instructions work on BOTH Hopper and Blackwell:
/// - TMA: Tensor Memory Accelerator bulk copies
/// - mbarrier: Async hardware barriers with transaction tracking
///
/// The architecture floor is generic sm_90; automatic cross-compilation keeps
/// the existing sm_100 default for forward-compatible Blackwell PTX.
fn contains_tma_features(contents: &str) -> bool {
    // TMA tensor copies and their commit/wait group controls.
    contains_cp_async_bulk_features(contents)
        || contains_mbarrier_sm90_features(contents)
        || contents.contains("fence.mbarrier_init")
        // Proxy fence for async operations
        || contents.contains("fence.proxy.async")
        || contents.contains(".sync_restrict")
}

fn contains_cp_async_bulk_features(contents: &str) -> bool {
    contents.contains("cp.async.bulk.")
}

fn contains_mbarrier_features(contents: &str) -> bool {
    contents.contains("mbarrier.") || contents.contains("llvm.nvvm.mbarrier")
}

fn contains_mbarrier_sm90_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        (statement.contains("mbarrier.") || statement.contains("llvm.nvvm.mbarrier"))
            && [
                "try_wait",
                "expect_tx",
                "complete_tx",
                "shared::cluster",
                ".acquire.",
                ".release.",
                ".relaxed",
            ]
            .iter()
            .any(|feature| statement.contains(feature))
    })
}

fn contains_mbarrier_ptx71_features(contents: &str) -> bool {
    contents
        .split(';')
        .any(|statement| statement.contains("mbarrier.test_wait") && statement.contains(".parity"))
}

fn contains_mbarrier_ptx78_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("mbarrier.")
            && (statement.contains("try_wait") || statement.contains("shared::cta"))
    })
}

fn contains_mbarrier_ptx80_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("mbarrier.")
            && [
                "expect_tx",
                "complete_tx",
                "shared::cluster",
                ".acquire.",
                ".release.",
            ]
            .iter()
            .any(|feature| statement.contains(feature))
    })
}

/// Checks for Blackwell tcgen05 instructions (sm_100a+).
///
/// These instructions require a datacenter-Blackwell `a`/`f` target; consumer
/// sm_120 does not provide Tensor Memory:
/// - tcgen05: Tensor Core Gen 5 (TMEM allocation, MMA, sync primitives)
///
/// Key differences from Hopper:
/// - tcgen05 MMA is single-thread (vs WGMMA's 128 threads)
/// - Uses Tensor Memory (TMEM) instead of registers
/// - Different synchronization model (mbarrier-based)
fn contains_blackwell_features(contents: &str) -> bool {
    // Keep the instruction-family match broad enough for inline PTX and LLVM
    // intrinsic names, but do not treat debug filenames such as `tcgen05.rs`
    // as an instruction.
    [
        "tcgen05.alloc",
        "tcgen05.dealloc",
        "tcgen05.relinquish_alloc_permit",
        "tcgen05.fence",
        "tcgen05.commit",
        "tcgen05.mma",
        "tcgen05.cp",
        "tcgen05.shift",
        "tcgen05.ld",
        "tcgen05.st",
        "tcgen05.wait",
    ]
    .iter()
    .any(|instruction| contents.contains(instruction))
}

/// Checks for base TMA multicast in LLVM IR or inline PTX.
///
/// TMA multicast (`cp.async.bulk.tensor...multicast::cluster`) is an optional
/// qualifier that broadcasts a tile to all CTAs in a cluster. It is legal on
/// sm_90+, although NVIDIA advises an `a`/`f` target
/// for performance. In the LLVM intrinsic this is controlled by the trailing
/// `use_cta_mask` i1 argument being set to true.
fn contains_tma_multicast(contents: &str) -> bool {
    contents.lines().any(|line| {
        line.contains("g2s.tile") && (line.contains(", i1 1, i1") || line.contains(", i1 true, i1"))
    }) || contents.split(';').any(|statement| {
        statement.contains("cp.async.bulk.tensor") && statement.contains(".multicast::cluster")
    })
}

/// Checks Blackwell-only TMA forms with an explicit CTA-group qualifier.
fn contains_tma_cta_group_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("cp.async.bulk.tensor")
            && (statement.contains(".cta_group::1") || statement.contains(".cta_group::2"))
    }) || contents.lines().any(|line| {
        line.contains("g2s.tile") && (line.contains(", i32 1)") || line.contains(", i32 2)"))
    })
}

/// Checks TMA copies whose destination is CTA-local shared memory.
///
/// `.shared::cta` already existed as a source state space for shared-to-global
/// copies, so the following `.global` source qualifier is part of the match.
/// The destination form was introduced in PTX 8.6 but is valid on sm_90.
fn contains_tma_shared_cta_destination(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("cp.async.bulk.") && statement.contains(".shared::cta.global")
    })
}

/// Checks PTX 8.6 TMA modifiers with a generic sm_100 architecture floor.
fn contains_tma_sm100_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        if !statement.contains("cp.async.bulk.") {
            return false;
        }
        statement.contains(".cp_mask")
            || (contains_tma_gather_or_im2col(statement)
                && statement.contains(".shared::cta.global"))
    })
}

/// Checks PTX 8.6 TMA modes restricted to datacenter Blackwell targets.
fn contains_tma_blackwell_accelerated_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        if !statement.contains("cp.async.bulk.") {
            return false;
        }
        statement.contains(".tile::scatter4")
            || statement.contains(".im2col::w::128")
            || (contains_tma_gather_or_im2col(statement)
                && !statement.contains(".shared::cta.global"))
    })
}

fn contains_tma_gather_or_im2col(statement: &str) -> bool {
    statement.contains(".tile::gather4")
        || (statement.contains(".im2col::w") && !statement.contains(".im2col::w::128"))
}

fn contains_tma_ptx86_features(contents: &str) -> bool {
    contains_tma_sm100_features(contents)
        || contains_tma_blackwell_accelerated_features(contents)
        || contents.contains(".sync_restrict")
        || contents
            .split(';')
            .any(|statement| statement.contains("mbarrier.") && statement.contains(".relaxed"))
}

fn contains_clc_features(contents: &str) -> bool {
    contents.contains("clusterlaunchcontrol.")
}

fn contains_clc_multicast_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("clusterlaunchcontrol.")
            && statement.contains(".multicast::cluster::all")
    })
}

fn contains_cluster_ptx80_features(contents: &str) -> bool {
    contents.split(';').any(|statement| {
        statement.contains("barrier.cluster.")
            && [".release", ".relaxed", ".acquire"]
                .iter()
                .any(|qualifier| statement.contains(qualifier))
    })
}

/// GPU feature requirements detected in one LLVM module.
///
/// This is a set rather than a single "strongest" feature: architecture
/// families are not totally ordered. For example, WGMMA requires Hopper
/// `sm_90a`, while PTX 8.6 matrix forms require Blackwell. Keeping every bit
/// lets target validation enforce the intersection instead of silently
/// choosing whichever instruction happened to have higher detector priority.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DetectedFeatures(u32);

#[allow(non_upper_case_globals)]
impl DetectedFeatures {
    /// tcgen05/TMEM (Blackwell datacenter, sm_100a).
    pub(crate) const Blackwell: Self = Self(1 << 0);
    /// Base TMA multicast (sm_90+, with architecture/family targets preferred).
    pub(crate) const TmaMulticast: Self = Self(1 << 1);
    /// Explicit CTA-group TMA forms (Blackwell datacenter family).
    pub(crate) const TmaCtaGroup: Self = Self(1 << 2);
    /// PTX 8.6 ldmatrix/stmatrix shapes supported on Blackwell family targets.
    pub(crate) const MatrixBlackwell: Self = Self(1 << 3);
    /// WGMMA (Hopper only, sm_90a - NOT forward-compatible).
    pub(crate) const Wgmma: Self = Self(1 << 4);
    /// TMA/mbarrier (Hopper+ compatible).
    pub(crate) const Tma: Self = Self(1 << 5);
    /// Thread Block Clusters (sm_90+, forward-compatible).
    pub(crate) const Cluster: Self = Self(1 << 6);
    /// Forward-compatible instructions with an sm_90 floor.
    pub(crate) const Sm90: Self = Self(1 << 7);
    /// Forward-compatible instructions with an sm_80 floor.
    pub(crate) const Sm80: Self = Self(1 << 8);
    /// Warp matrix register transpose introduced in PTX 7.8 on sm_75.
    pub(crate) const Movmatrix: Self = Self(1 << 9);
    /// Warp matrix shared-memory load introduced in PTX 6.5 on sm_75.
    pub(crate) const Ldmatrix: Self = Self(1 << 10);
    /// No special features (Volta+, with an sm_80 cross-compile default).
    pub(crate) const Basic: Self = Self(1 << 11);
    /// Generic Blackwell-or-newer operations such as base CLC and TMA cp_mask.
    pub(crate) const Sm100: Self = Self(1 << 12);
    /// Architecture/family-specific Blackwell features also available on consumers.
    pub(crate) const BlackwellFamily: Self = Self(1 << 13);
    /// Architecture/family-specific datacenter Blackwell TMA modes.
    pub(crate) const BlackwellAccelerated: Self = Self(1 << 14);
    /// Floating-point `redux.sync` (the sm_100/sm_103 architecture family).
    pub(crate) const ReduxF32: Self = Self(1 << 15);
    /// FP8 / f16-accumulator multimem forms on supported Blackwell families.
    pub(crate) const MultimemFp8: Self = Self(1 << 16);

    const ALL: [Self; 17] = [
        Self::Blackwell,
        Self::TmaCtaGroup,
        Self::BlackwellAccelerated,
        Self::BlackwellFamily,
        Self::ReduxF32,
        Self::MultimemFp8,
        Self::TmaMulticast,
        Self::MatrixBlackwell,
        Self::Wgmma,
        Self::Tma,
        Self::Cluster,
        Self::Sm90,
        Self::Sm80,
        Self::Movmatrix,
        Self::Ldmatrix,
        Self::Sm100,
        Self::Basic,
    ];

    const fn empty() -> Self {
        Self(0)
    }

    const fn contains(self, feature: Self) -> bool {
        self.0 & feature.0 != 0
    }

    fn insert(&mut self, feature: Self) {
        self.0 |= feature.0;
    }

    fn iter(self) -> impl Iterator<Item = Self> {
        Self::ALL
            .into_iter()
            .filter(move |feature| self.contains(*feature))
    }

    fn name(self) -> &'static str {
        match self {
            Self::Blackwell => "Blackwell",
            Self::TmaMulticast => "TmaMulticast",
            Self::TmaCtaGroup => "TmaCtaGroup",
            Self::MatrixBlackwell => "MatrixBlackwell",
            Self::Wgmma => "Wgmma",
            Self::Tma => "Tma",
            Self::Cluster => "Cluster",
            Self::Sm90 => "Sm90",
            Self::Sm80 => "Sm80",
            Self::Movmatrix => "Movmatrix",
            Self::Ldmatrix => "Ldmatrix",
            Self::Sm100 => "Sm100",
            Self::BlackwellFamily => "BlackwellFamily",
            Self::BlackwellAccelerated => "BlackwellAccelerated",
            Self::ReduxF32 => "ReduxF32",
            Self::MultimemFp8 => "MultimemFp8",
            Self::Basic => "Basic",
            _ => "Unknown",
        }
    }
}

impl std::fmt::Debug for DetectedFeatures {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for feature in self.iter() {
            if !first {
                formatter.write_str(" + ")?;
            }
            formatter.write_str(feature.name())?;
            first = false;
        }
        Ok(())
    }
}

impl std::ops::BitOr for DetectedFeatures {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

/// PTX ISA requirements are independent of the GPU architecture floor.
///
/// For example, a module may need sm_80 because it uses `cp.async` and still
/// need PTX 7.8 because it also uses `movmatrix`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PtxIsaRequirement {
    Default,
    Ptx62,
    Ptx65,
    Ptx70,
    Ptx71,
    Ptx78,
    Ptx80,
    Ptx86,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModuleRequirements {
    pub features: DetectedFeatures,
    pub ptx_isa: PtxIsaRequirement,
}

/// Detect every architecture requirement in exported LLVM text.
///
/// Both the ordinary PTX path and automatic libdevice mode use this exact
/// detector. The latter renders an in-memory preview before choosing the NVVM
/// pointer dialect.
pub fn detect_features_in_llvm_text(contents: &str) -> DetectedFeatures {
    let mut features = DetectedFeatures::empty();
    for (present, feature) in [
        (
            contains_blackwell_features(contents),
            DetectedFeatures::Blackwell,
        ),
        (
            contains_tma_cta_group_features(contents),
            DetectedFeatures::TmaCtaGroup,
        ),
        (
            contains_tma_blackwell_accelerated_features(contents),
            DetectedFeatures::BlackwellAccelerated,
        ),
        (
            contains_clc_multicast_features(contents),
            DetectedFeatures::BlackwellFamily,
        ),
        (
            contains_redux_f32_features(contents),
            DetectedFeatures::ReduxF32,
        ),
        (
            contains_multimem_blackwell_features(contents),
            DetectedFeatures::MultimemFp8,
        ),
        (
            contains_tma_multicast(contents),
            DetectedFeatures::TmaMulticast,
        ),
        (
            contains_blackwell_matrix_features(contents),
            DetectedFeatures::MatrixBlackwell,
        ),
        (contains_wgmma_features(contents), DetectedFeatures::Wgmma),
        (contains_tma_features(contents), DetectedFeatures::Tma),
        (
            contains_cluster_features(contents),
            DetectedFeatures::Cluster,
        ),
        (contains_sm90_features(contents), DetectedFeatures::Sm90),
        (contains_sm80_features(contents), DetectedFeatures::Sm80),
        (
            contains_movmatrix_features(contents),
            DetectedFeatures::Movmatrix,
        ),
        (
            contains_ldmatrix_features(contents),
            DetectedFeatures::Ldmatrix,
        ),
        (
            contains_tma_sm100_features(contents) || contains_clc_features(contents),
            DetectedFeatures::Sm100,
        ),
    ] {
        if present {
            features.insert(feature);
        }
    }
    if features == DetectedFeatures::empty() {
        features.insert(DetectedFeatures::Basic);
    }
    features
}

fn detect_module_requirements_in_llvm_text(contents: &str) -> ModuleRequirements {
    let mut ptx_isa = PtxIsaRequirement::Default;
    if contains_packed_f16_atomic_features(contents) {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::Ptx62);
    }
    if contains_ldmatrix_features(contents) {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::Ptx65);
    }
    if contains_mbarrier_features(contents)
        || contents.contains("redux.sync")
        || contains_mma_m16n8k16_f32_bf16_features(contents)
        || contains_mma_m16n8k16_f32_f16_features(contents)
        || contains_mma_m16n8k8_f32_tf32_features(contents)
        || contains_mma_m16n8k32_s32_s8_features(contents)
        || contains_mma_m8n8k4_f64_features(contents)
    {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::Ptx70);
    }
    if contains_mbarrier_ptx71_features(contents) {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::Ptx71);
    }
    if contains_movmatrix_features(contents)
        || contains_stmatrix_features(contents)
        || contains_ldmatrix_cta_state_space(contents)
        || contains_cluster_features(contents)
        || contains_mbarrier_ptx78_features(contents)
        || contains_packed_bf16_atomic_features(contents)
    {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::Ptx78);
    }
    if contains_cp_async_bulk_features(contents)
        || contains_wgmma_features(contents)
        || contains_cluster_ptx80_features(contents)
        || contains_elect_features(contents)
        || contains_mbarrier_ptx80_features(contents)
        || contents.contains("fence.mbarrier_init")
        || contents.contains("fence.proxy.async")
    {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::Ptx80);
    }
    if contains_blackwell_matrix_features(contents)
        || contains_tma_cta_group_features(contents)
        || contains_tma_shared_cta_destination(contents)
        || contains_tma_ptx86_features(contents)
        || contains_clc_features(contents)
        || contains_blackwell_features(contents)
        || contains_fence_acquire_release_features(contents)
        || contains_multimem_features(contents)
        || contains_redux_f32_features(contents)
    {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::Ptx86);
    }

    ModuleRequirements {
        features: detect_features_in_llvm_text(contents),
        ptx_isa,
    }
}

pub fn detect_module_requirements_in_llvm_file(
    ll_path: &Path,
) -> Result<ModuleRequirements, PipelineError> {
    let contents = std::fs::read_to_string(ll_path).map_err(|error| {
        PipelineError::PtxGeneration(format!(
            "failed to inspect generated LLVM IR {}: {error}",
            ll_path.display()
        ))
    })?;
    Ok(detect_module_requirements_in_llvm_text(&contents))
}

/// Select a concrete architecture that satisfies every detected feature.
///
/// The first candidate preserves the established default for a module's most
/// restrictive-looking feature. The remaining candidates handle intersections
/// such as WGMMA + TMA multicast, whose only common target is `sm_90a`.
pub fn select_target(features: DetectedFeatures) -> Result<&'static str, String> {
    let preferred = if features.contains(DetectedFeatures::Blackwell)
        || features.contains(DetectedFeatures::TmaCtaGroup)
        || features.contains(DetectedFeatures::BlackwellAccelerated)
        || features.contains(DetectedFeatures::BlackwellFamily)
        || features.contains(DetectedFeatures::ReduxF32)
        || features.contains(DetectedFeatures::MultimemFp8)
        || features.contains(DetectedFeatures::TmaMulticast)
        || features.contains(DetectedFeatures::MatrixBlackwell)
    {
        "sm_100a"
    } else if features.contains(DetectedFeatures::Wgmma) {
        "sm_90a"
    } else if features.contains(DetectedFeatures::Sm100) {
        "sm_100"
    } else if features.contains(DetectedFeatures::Tma) {
        // Plain TMA is compatible with Hopper, but sm_100 is the existing
        // cross-compilation default because it produces forward-compatible
        // PTX for generic Blackwell devices.
        "sm_100"
    } else if features.contains(DetectedFeatures::Cluster)
        || features.contains(DetectedFeatures::Sm90)
    {
        "sm_90"
    } else if features.contains(DetectedFeatures::Sm80) {
        "sm_80"
    } else if features.contains(DetectedFeatures::Movmatrix)
        || features.contains(DetectedFeatures::Ldmatrix)
    {
        "sm_75"
    } else {
        "sm_80"
    };

    for candidate in [
        preferred, "sm_100a", "sm_90a", "sm_100", "sm_90", "sm_80", "sm_75",
    ] {
        if arch_satisfies(candidate, features) {
            return Ok(candidate);
        }
    }

    Err(format!(
        "detected CUDA features {features:?} do not share a compatible GPU architecture"
    ))
}

/// Does `arch` (e.g. `"sm_120a"`, `"sm_90"`) support the kernel's detected
/// features?
///
/// tcgen05/TMEM and explicit `cta_group` TMA forms exist only in the sm_100
/// datacenter-Blackwell family: consumer Blackwell (sm_120) and Hopper (sm_90)
/// lack them, so an sm_120 GPU cannot run an sm_100 tcgen05 kernel even though
/// 120 > 100. WGMMA is Hopper-only. The remaining features are forward
/// compatible from their floor (TMA / cluster / sm_90 features need sm_90+,
/// sm_80 features need sm_80+, and basic needs sm_70+).
///
/// Used to decide whether the GPU in this machine (the `CUDA_OXIDE_DEVICE_ARCH`
/// hint) can actually run the kernel, or whether we must build for the arch the
/// IR requires instead.
pub fn arch_satisfies(arch: &str, features: DetectedFeatures) -> bool {
    let Some((capability, suffix)) = arch_compute_capability_and_suffix(arch) else {
        return false;
    };
    if !is_known_cuda_target(capability, suffix) {
        return false;
    }
    features
        .iter()
        .all(|feature| arch_satisfies_feature(capability, suffix, feature))
}

fn arch_satisfies_feature(
    capability: u32,
    suffix: Option<char>,
    feature: DetectedFeatures,
) -> bool {
    let major = capability / 10;
    match feature {
        DetectedFeatures::Blackwell | DetectedFeatures::TmaCtaGroup => {
            supports_tcgen_target(capability, suffix)
        }
        DetectedFeatures::BlackwellAccelerated => {
            supports_blackwell_accelerated_target(capability, suffix)
        }
        DetectedFeatures::BlackwellFamily | DetectedFeatures::MatrixBlackwell => {
            supports_blackwell_family_target(capability, suffix)
        }
        DetectedFeatures::ReduxF32 => supports_redux_f32_target(capability, suffix),
        DetectedFeatures::MultimemFp8 => supports_multimem_fp8_target(capability, suffix),
        // The PTX ISA requires only sm_90+. The suffixed targets are advised
        // for performance, so target selection still prefers sm_100a.
        DetectedFeatures::TmaMulticast => major >= 9,
        DetectedFeatures::Wgmma => capability == 90 && suffix == Some('a'),
        DetectedFeatures::Sm100 => is_known_blackwell_capability(capability),
        DetectedFeatures::Tma | DetectedFeatures::Cluster | DetectedFeatures::Sm90 => major >= 9,
        DetectedFeatures::Sm80 => major >= 8,
        DetectedFeatures::Movmatrix | DetectedFeatures::Ldmatrix => capability >= 75,
        // Basic kernels are supported on the project's Volta+ floor. The
        // cross-compilation default remains sm_80, but a detected sm_70/sm_75
        // GPU is a valid and more useful target for `cargo oxide run`.
        DetectedFeatures::Basic => major >= 7,
        // `iter` only yields the single-bit constants above.
        _ => false,
    }
}

/// tcgen05/TMEM exists only on the datacenter Blackwell architecture or
/// family targets. Consumer sm_120 and generic targets without an `a`/`f`
/// suffix do not provide Tensor Memory.
fn supports_tcgen_target(capability: u32, suffix: Option<char>) -> bool {
    match suffix {
        // Architecture-specific targets are exact, not numerically forward
        // compatible. `sm_101a` is the PTX 8.x spelling later renamed to
        // `sm_110a`; accept both spellings plus the distinct sm_103 target.
        Some('a') => matches!(capability, 100 | 101 | 103 | 110),
        Some('f') => matches!(capability, 100 | 101 | 103 | 110),
        _ => false,
    }
}

fn supports_blackwell_accelerated_target(capability: u32, suffix: Option<char>) -> bool {
    match suffix {
        Some('a') => matches!(capability, 100 | 101 | 103 | 110),
        Some('f') => matches!(capability, 100 | 101 | 103 | 110),
        _ => false,
    }
}

fn supports_blackwell_family_target(capability: u32, suffix: Option<char>) -> bool {
    match suffix {
        Some('a') => matches!(capability, 100 | 101 | 110 | 120),
        Some('f') => matches!(capability, 100 | 101 | 103 | 110 | 120 | 121),
        _ => false,
    }
}

/// Floating-point `redux.sync` is scoped to the sm_100/sm_103 family.
fn supports_redux_f32_target(capability: u32, suffix: Option<char>) -> bool {
    matches!(suffix, Some('a' | 'f')) && matches!(capability, 100 | 103)
}

/// FP8 / f16-accumulator multimem forms span several Blackwell architecture
/// targets, but consumer family (`f`) targets do not support the sm_120 line.
fn supports_multimem_fp8_target(capability: u32, suffix: Option<char>) -> bool {
    match suffix {
        Some('a') => matches!(capability, 100 | 101 | 103 | 110 | 120 | 121),
        Some('f') => matches!(capability, 100 | 101 | 103 | 110),
        _ => false,
    }
}

fn is_known_blackwell_capability(capability: u32) -> bool {
    matches!(capability, 100 | 101 | 103 | 110 | 120 | 121)
}

fn is_known_cuda_target(capability: u32, suffix: Option<char>) -> bool {
    let known_capability = matches!(
        capability,
        70 | 72 | 75 | 80 | 86 | 87 | 88 | 89 | 90 | 100 | 101 | 103 | 110 | 120 | 121
    );
    known_capability
        && match suffix {
            None => true,
            Some('a') => capability == 90 || is_known_blackwell_capability(capability),
            Some('f') => is_known_blackwell_capability(capability),
            _ => false,
        }
}

pub fn validate_target_features(
    target: &CudaArch,
    features: DetectedFeatures,
) -> Result<(), String> {
    let compatible_default = select_target(features)?;
    if arch_satisfies(&target.sm(), features) {
        return Ok(());
    }

    Err(format!(
        "CUDA target {} cannot lower detected feature {features:?}; \
         cuda-oxide requires a target compatible with {} for this module",
        target.sm(),
        compatible_default
    ))
}

pub fn resolve_ptx_target(
    explicit_override: Option<&str>,
    device_hint: Option<&str>,
    detected: DetectedFeatures,
) -> Result<(String, &'static str), PipelineError> {
    if let Some(target) = explicit_override {
        let parsed = target.parse::<CudaArch>().map_err(|error| {
            PipelineError::PtxGeneration(format!("invalid CUDA_OXIDE_TARGET `{target}`: {error}"))
        })?;
        validate_target_features(&parsed, detected).map_err(PipelineError::PtxGeneration)?;
        return Ok((parsed.sm(), "CUDA_OXIDE_TARGET"));
    }

    if let Some(device) = device_hint.filter(|target| arch_satisfies(target, detected)) {
        return Ok((device.to_string(), "detected GPU"));
    }

    let target = select_target(detected).map_err(PipelineError::PtxGeneration)?;
    Ok((target.to_string(), "feature requirement"))
}

/// Select the PTX ISA independently from the GPU architecture.
///
/// LLVM GPU CPUs select a default PTX ISA independently from the hardware
/// feature floor. Raise that ISA only when the selected CPU's default is too
/// old; never force a newer target back to an older PTX version.
pub fn required_ptx_feature(target: &str, requirement: PtxIsaRequirement) -> Option<&'static str> {
    let capability = arch_compute_capability(target)?;
    let minimum = target_minimum_ptx_isa(capability)?;
    let requested = match requirement {
        PtxIsaRequirement::Default => return None,
        PtxIsaRequirement::Ptx62 => 62,
        PtxIsaRequirement::Ptx65 => 65,
        PtxIsaRequirement::Ptx70 => 70,
        PtxIsaRequirement::Ptx71 => 71,
        PtxIsaRequirement::Ptx78 => 78,
        PtxIsaRequirement::Ptx80 => 80,
        PtxIsaRequirement::Ptx86 => 86,
    };
    if requested <= minimum {
        return None;
    }
    match requirement {
        PtxIsaRequirement::Default => None,
        PtxIsaRequirement::Ptx62 => Some("+ptx62"),
        PtxIsaRequirement::Ptx65 => Some("+ptx65"),
        PtxIsaRequirement::Ptx70 => Some("+ptx70"),
        PtxIsaRequirement::Ptx71 => Some("+ptx71"),
        PtxIsaRequirement::Ptx78 => Some("+ptx78"),
        PtxIsaRequirement::Ptx80 => Some("+ptx80"),
        PtxIsaRequirement::Ptx86 => Some("+ptx86"),
    }
}

/// Minimum PTX ISA accepted by LLVM for each concrete target. Passing an
/// older `+ptxNN` feature does not merely do nothing: LLVM aborts because that
/// ISA cannot name the selected processor.
fn target_minimum_ptx_isa(capability: u32) -> Option<u32> {
    match capability {
        70 => Some(60),
        72 => Some(61),
        75 => Some(63),
        80 => Some(70),
        86 => Some(71),
        87 => Some(74),
        88 => Some(90),
        89 | 90 => Some(78),
        100 | 101 => Some(86),
        103 => Some(88),
        110 => Some(90),
        120 => Some(87),
        121 => Some(88),
        _ => None,
    }
}

/// Reject targets that the supported LLVM 21 backend silently mishandles.
///
/// LLVM 21 accepts `-mcpu=sm_88` / `sm_110*` but only prints a warning and
/// emits PTX 6.0, which ptxas then rejects. LLVM 22 is the first backend in
/// cuda-oxide's supported toolchain set that emits valid PTX for these PTX 9.0
/// target spellings. An unknown version is rejected because it cannot prove
/// that the backend knows the processor.
pub fn validate_target_for_llvm_major(target: &str, llc_major: Option<u32>) -> Result<(), String> {
    let capability = arch_compute_capability(target);
    if matches!(capability, Some(88 | 110)) && llc_major.is_none_or(|major| major < 22) {
        let backend = llc_major.map_or_else(
            || "an LLVM backend with an unknown version".to_string(),
            |major| format!("LLVM {major}"),
        );
        return Err(format!(
            "CUDA target {target} requires LLVM 22 or newer; {backend} does not reliably emit valid PTX for this PTX 9.0 target"
        ));
    }
    Ok(())
}

/// Extract the compute-capability *major* version from an `sm_…` target string.
///
/// CUDA concatenates major+minor without a separator, so `"sm_120a"` is cc 12.0
/// (major 12), `"sm_90"` is cc 9.0, `"sm_103a"` is cc 10.3. We read the digit
/// run after `sm_` and divide by ten. Returns `None` when there are no digits.
#[cfg(test)]
fn arch_major(arch: &str) -> Option<u32> {
    arch_compute_capability(arch).map(|capability| capability / 10)
}

/// Extract the numeric compute capability from an `sm_…` target.
fn arch_compute_capability(arch: &str) -> Option<u32> {
    arch_compute_capability_and_suffix(arch).map(|(capability, _)| capability)
}

fn arch_compute_capability_and_suffix(arch: &str) -> Option<(u32, Option<char>)> {
    if !arch.starts_with("sm_") {
        return None;
    }
    let target = arch.parse::<CudaArch>().ok()?;
    Some((target.capability(), target.suffix()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_detection_reads_llvm_ir_snippets() {
        let llvm = r#"
            call void asm sideeffect "wgmma.fence.sync.aligned", ""()
            call void @llvm.nvvm.tcgen05.alloc()
            call void asm sideeffect "cluster.sync.aligned", ""()
            call void asm sideeffect "cp.async.bulk.tensor.2d.shared::cluster.global", ""()
            call void asm sideeffect "cp.async.ca.shared.global", ""()
        "#;

        assert!(contains_wgmma_features(llvm));
        assert!(contains_blackwell_features(llvm));
        assert!(contains_cluster_features(llvm));
        assert!(contains_tma_features(llvm));
        assert!(contains_sm80_features(llvm));
        let detected = detect_features_in_llvm_text(llvm);
        for feature in [
            DetectedFeatures::Blackwell,
            DetectedFeatures::Wgmma,
            DetectedFeatures::Cluster,
            DetectedFeatures::Tma,
            DetectedFeatures::Sm80,
        ] {
            assert!(detected.contains(feature), "missing {feature:?}");
        }
        assert!(
            select_target(detected).is_err(),
            "Hopper-only WGMMA and Blackwell-only tcgen05 are incompatible"
        );
    }

    #[test]
    fn test_sm80_detection_accepts_inline_ptx_and_nvvm_intrinsics() {
        for llvm in [
            r#"call void asm sideeffect "cp.async.ca.shared.global [%0], [%1], 4;", "l,l"()"#,
            "call void @llvm.nvvm.cp.async.ca.shared.global.8(ptr addrspace(3) %dst, ptr addrspace(1) %src)",
            r#"call void asm sideeffect "cp.async.commit_group;", ""()"#,
            "call void @llvm.nvvm.cp.async.wait.all()",
        ] {
            assert!(contains_sm80_features(llvm), "missed cp.async in {llvm}");
            assert_eq!(detect_features_in_llvm_text(llvm), DetectedFeatures::Sm80);
        }
    }

    #[test]
    fn test_bf16x2_detection_matches_exact_architecture_floors() {
        for mnemonic in [
            "add.rn.bf16x2 $0, $1, $2;",
            "sub.rn.bf16x2 $0, $1, $2;",
            "mul.rn.bf16x2 $0, $1, $2;",
        ] {
            assert!(contains_sm90_features(mnemonic));
            assert!(!contains_sm80_features(mnemonic));
            assert_eq!(
                detect_features_in_llvm_text(mnemonic),
                DetectedFeatures::Sm90
            );
        }

        for mnemonic in ["add.rn.bf16x2\t$0, $1, $2;", "sub.rn.bf16x2\\09$0, $1, $2;"] {
            assert_eq!(
                detect_features_in_llvm_text(mnemonic),
                DetectedFeatures::Sm90,
                "{mnemonic:?}"
            );
        }

        for mnemonic in [
            "fma.rn.bf16x2 $0, $1, $2, $3;",
            "fma.rn.relu.bf16x2 $0, $1, $2, $3;",
            "min.bf16x2 $0, $1, $2;",
            "max.bf16x2 $0, $1, $2;",
            "neg.bf16x2 $0, $1;",
            "abs.bf16x2 $0, $1;",
        ] {
            assert!(!contains_sm90_features(mnemonic));
            assert!(contains_sm80_features(mnemonic));
            assert_eq!(
                detect_features_in_llvm_text(mnemonic),
                DetectedFeatures::Sm80
            );
        }

        for near_miss in [
            "add.rn.bf16x2x $0, $1, $2;",
            "fma.rn.bf16x2x $0, $1, $2, $3;",
            "add.rn.bf16x2\\5C09$0, $1, $2;",
        ] {
            assert!(!contains_sm90_features(near_miss));
            assert!(!contains_sm80_features(near_miss));
            assert_eq!(
                detect_features_in_llvm_text(near_miss),
                DetectedFeatures::Basic
            );
        }
    }

    #[test]
    fn dense_bf16_mma_detection_applies_exact_sm80_and_ptx70_floors() {
        let mnemonic =
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};";
        for spelling in [
            mnemonic,
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\t{$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\\09{$0}, {$1}, {$2}, {$3};",
            ";mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "prefix\\0Amma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "\"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "{mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "$L:mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "/* comment */mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "@p mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "@!%p\\09mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        ] {
            assert!(
                contains_mma_m16n8k16_f32_bf16_features(spelling),
                "missed {spelling:?}"
            );
        }

        let requirements = detect_module_requirements_in_llvm_text(mnemonic);
        assert_eq!(
            requirements,
            ModuleRequirements {
                features: DetectedFeatures::Sm80,
                ptx_isa: PtxIsaRequirement::Ptx70,
            }
        );
        assert_eq!(select_target(requirements.features).unwrap(), "sm_80");

        let lower_target =
            resolve_ptx_target(Some("sm_75"), None, requirements.features).unwrap_err();
        assert!(
            lower_target
                .to_string()
                .contains("cannot lower detected feature Sm80"),
            "{lower_target}"
        );
        let (target, _) = resolve_ptx_target(Some("sm_80"), None, requirements.features).unwrap();
        assert_eq!(target, "sm_80");

        for near_miss in [
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "mma.sp.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32x {$0}, {$1}, {$2}, {$3};",
            "not_mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "$mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "%mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "@mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "!mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "@!mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "not$mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "/mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            ")mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
        ] {
            assert!(
                !contains_mma_m16n8k16_f32_bf16_features(near_miss),
                "matched {near_miss:?}"
            );
        }

        let combined = format!(
            "{mnemonic}\n{}",
            "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;"
        );
        assert_eq!(
            detect_module_requirements_in_llvm_text(&combined),
            ModuleRequirements {
                features: DetectedFeatures::Sm80 | DetectedFeatures::Movmatrix,
                ptx_isa: PtxIsaRequirement::Ptx78,
            }
        );
    }

    #[test]
    fn packed_atomic_detection_enforces_native_architecture_and_ptx_floors() {
        for f16 in [
            "atom.global.add.noftz.f16x2 $0, [$1], $2;",
            "atom.global.add.noftz.f16x2\t$0, [$1], $2;",
            "atom.global.add.noftz.f16x2\\09$0, [$1], $2;",
            ";atom.global.add.noftz.f16x2 $0, [$1], $2;",
            "prefix\\0Aatom.global.add.noftz.f16x2 $0, [$1], $2;",
            "\"atom.global.add.noftz.f16x2 $0, [$1], $2;",
            "{atom.global.add.noftz.f16x2 $0, [$1], $2;",
            "$L:atom.global.add.noftz.f16x2 $0, [$1], $2;",
            "/* comment */atom.global.add.noftz.f16x2 $0, [$1], $2;",
            "@p atom.global.add.noftz.f16x2 $0, [$1], $2;",
            "@!%p\\09atom.global.add.noftz.f16x2 $0, [$1], $2;",
        ] {
            assert!(contains_packed_f16_atomic_features(f16), "{f16:?}");
            assert!(!contains_packed_bf16_atomic_features(f16), "{f16:?}");
            assert_eq!(detect_features_in_llvm_text(f16), DetectedFeatures::Basic);
            assert_eq!(
                detect_module_requirements_in_llvm_text(f16).ptx_isa,
                PtxIsaRequirement::Ptx62
            );
        }
        assert_eq!(
            required_ptx_feature("sm_70", PtxIsaRequirement::Ptx62),
            Some("+ptx62")
        );
        assert_eq!(
            resolve_ptx_target(Some("sm_70"), None, DetectedFeatures::Basic)
                .unwrap()
                .0,
            "sm_70"
        );

        for bf16 in [
            "atom.global.add.noftz.bf16x2 $0, [$1], $2;",
            "atom.global.add.noftz.bf16x2\t$0, [$1], $2;",
            "atom.global.add.noftz.bf16x2\\0A$0, [$1], $2;",
        ] {
            assert!(contains_packed_bf16_atomic_features(bf16), "{bf16:?}");
            assert!(!contains_packed_f16_atomic_features(bf16), "{bf16:?}");
            assert_eq!(detect_features_in_llvm_text(bf16), DetectedFeatures::Sm90);
            assert_eq!(
                detect_module_requirements_in_llvm_text(bf16).ptx_isa,
                PtxIsaRequirement::Ptx78
            );
        }
        assert_eq!(select_target(DetectedFeatures::Sm90).unwrap(), "sm_90");
        let rejected = resolve_ptx_target(Some("sm_80"), None, DetectedFeatures::Sm90)
            .expect_err("native bf16x2 atomic add must reject sm_80")
            .to_string();
        assert!(rejected.contains("cannot lower detected feature Sm90"));
        let near_miss = resolve_ptx_target(Some("sm_89"), None, DetectedFeatures::Sm90)
            .expect_err("the architecture immediately below sm_90 must be rejected")
            .to_string();
        assert!(near_miss.contains("cannot lower detected feature Sm90"));

        let both = "atom.global.add.noftz.f16x2 $0, [$1], $2; \
                    atom.global.add.noftz.bf16x2 $0, [$1], $2;";
        let requirements = detect_module_requirements_in_llvm_text(both);
        assert_eq!(requirements.features, DetectedFeatures::Sm90);
        assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx78);

        let dense_bf16_mma =
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};";
        let mma_f16_requirements = detect_module_requirements_in_llvm_text(&format!(
            "{dense_bf16_mma}\natom.global.add.noftz.f16x2 $0, [$1], $2;"
        ));
        assert_eq!(
            mma_f16_requirements,
            ModuleRequirements {
                features: DetectedFeatures::Sm80,
                ptx_isa: PtxIsaRequirement::Ptx70,
            }
        );
        assert_eq!(
            select_target(mma_f16_requirements.features).unwrap(),
            "sm_80"
        );

        let mma_bf16_requirements = detect_module_requirements_in_llvm_text(&format!(
            "{dense_bf16_mma}\natom.global.add.noftz.bf16x2 $0, [$1], $2;"
        ));
        assert_eq!(
            mma_bf16_requirements,
            ModuleRequirements {
                features: DetectedFeatures::Sm90 | DetectedFeatures::Sm80,
                ptx_isa: PtxIsaRequirement::Ptx78,
            }
        );
        assert_eq!(
            select_target(mma_bf16_requirements.features).unwrap(),
            "sm_90"
        );

        for near_miss in [
            "atom.global.add.noftz.f16x2x $0, [$1], $2;",
            "atom.global.add.noftz.bf16x2x $0, [$1], $2;",
            "not_atom.global.add.noftz.f16x2 $0, [$1], $2;",
            "not_atom.global.add.noftz.bf16x2 $0, [$1], $2;",
            "not.atom.global.add.noftz.f16x2 $0, [$1], $2;",
            "not.atom.global.add.noftz.bf16x2 $0, [$1], $2;",
            "$atom.global.add.noftz.f16x2 $0, [$1], $2;",
            "%atom.global.add.noftz.bf16x2 $0, [$1], $2;",
            "@atom.global.add.noftz.f16x2 $0, [$1], $2;",
            "!atom.global.add.noftz.bf16x2 $0, [$1], $2;",
            "@!atom.global.add.noftz.f16x2 $0, [$1], $2;",
            "not$atom.global.add.noftz.f16x2 $0, [$1], $2;",
            "/atom.global.add.noftz.bf16x2 $0, [$1], $2;",
            ")atom.global.add.noftz.f16x2 $0, [$1], $2;",
            "atom.shared.add.noftz.f16x2 $0, [$1], $2;",
            "atom.global.add.bf16x2 $0, [$1], $2;",
            "red.global.add.noftz.bf16x2 [$0], $1;",
            "atom.global.add.noftz.f16x2\\5C09$0, [$1], $2;",
        ] {
            assert!(!contains_packed_f16_atomic_features(near_miss));
            assert!(!contains_packed_bf16_atomic_features(near_miss));
            assert_eq!(
                detect_module_requirements_in_llvm_text(near_miss),
                ModuleRequirements {
                    features: DetectedFeatures::Basic,
                    ptx_isa: PtxIsaRequirement::Default,
                },
                "{near_miss:?}"
            );
        }
    }

    #[test]
    fn fp64_mma_and_packed_atomics_take_the_strongest_target_floor() {
        let dense_bf16_mma =
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};";
        let dense_fp64_mma =
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};";

        let fp64_f16_requirements = detect_module_requirements_in_llvm_text(&format!(
            "{dense_fp64_mma}\natom.global.add.noftz.f16x2 $0, [$1], $2;"
        ));
        assert_eq!(
            fp64_f16_requirements,
            ModuleRequirements {
                features: DetectedFeatures::Sm80,
                ptx_isa: PtxIsaRequirement::Ptx70,
            }
        );
        assert_eq!(
            select_target(fp64_f16_requirements.features).unwrap(),
            "sm_80"
        );

        let fp64_bf16_requirements = detect_module_requirements_in_llvm_text(&format!(
            "{dense_fp64_mma}\natom.global.add.noftz.bf16x2 $0, [$1], $2;"
        ));
        assert_eq!(
            fp64_bf16_requirements,
            ModuleRequirements {
                features: DetectedFeatures::Sm90 | DetectedFeatures::Sm80,
                ptx_isa: PtxIsaRequirement::Ptx78,
            }
        );
        assert_eq!(
            select_target(fp64_bf16_requirements.features).unwrap(),
            "sm_90"
        );

        let all_four = format!(
            "{dense_bf16_mma}\n{dense_fp64_mma}\n\
             atom.global.add.noftz.f16x2 $0, [$1], $2;\n\
             atom.global.add.noftz.bf16x2 $0, [$1], $2;"
        );
        let all_four_requirements = detect_module_requirements_in_llvm_text(&all_four);
        assert_eq!(
            all_four_requirements,
            ModuleRequirements {
                features: DetectedFeatures::Sm90 | DetectedFeatures::Sm80,
                ptx_isa: PtxIsaRequirement::Ptx78,
            }
        );
        assert_eq!(
            select_target(all_four_requirements.features).unwrap(),
            "sm_90"
        );
    }

    #[test]
    fn dense_f16_mma_detection_applies_exact_sm80_and_ptx70_floors() {
        let mnemonic = "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};";
        for spelling in [
            mnemonic,
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\t{$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\\09{$0}, {$1}, {$2}, {$3};",
            ";mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
            "prefix\\0Amma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
            "\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
            "{mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
            "$L:mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
            "/* comment */mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
            "@p mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
            "@!%p\\09mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
        ] {
            assert!(
                contains_mma_m16n8k16_f32_f16_features(spelling),
                "missed {spelling:?}"
            );
        }

        let requirements = detect_module_requirements_in_llvm_text(mnemonic);
        assert_eq!(
            requirements,
            ModuleRequirements {
                features: DetectedFeatures::Sm80,
                ptx_isa: PtxIsaRequirement::Ptx70,
            }
        );
        assert_eq!(select_target(requirements.features).unwrap(), "sm_80");

        let lower_target =
            resolve_ptx_target(Some("sm_75"), None, requirements.features).unwrap_err();
        assert!(
            lower_target
                .to_string()
                .contains("cannot lower detected feature Sm80"),
            "{lower_target}"
        );
        let (target, _) = resolve_ptx_target(Some("sm_80"), None, requirements.features).unwrap();
        assert_eq!(target, "sm_80");

        for near_miss in [
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
            "mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32x {$0}, {$1}, {$2}, {$3};",
            "not_mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
            "$mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
            "%mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
            "@mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
            "!mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
            "@!mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
            "not$mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
            "/mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
            ")mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
        ] {
            assert!(
                !contains_mma_m16n8k16_f32_f16_features(near_miss),
                "matched {near_miss:?}"
            );
        }

        let combined = format!(
            "{mnemonic}\n{}",
            "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;"
        );
        assert_eq!(
            detect_module_requirements_in_llvm_text(&combined),
            ModuleRequirements {
                features: DetectedFeatures::Sm80 | DetectedFeatures::Movmatrix,
                ptx_isa: PtxIsaRequirement::Ptx78,
            }
        );
    }

    #[test]
    fn tf32_mma_detection_applies_exact_sm80_and_ptx70_floors() {
        let mnemonic = concat!(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 ",
            "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
        );
        for spelling in [
            mnemonic,
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32\t{$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32\\09{$0}, {$1}, {$2}, {$3};",
            ";mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
            "prefix\\0Amma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
            "\"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
            "{mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
            "$L:mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
            "/* comment */mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
            "@p mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
            "@!%p\\09mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
        ] {
            assert!(
                contains_mma_m16n8k8_f32_tf32_features(spelling),
                "missed {spelling:?}"
            );
        }

        let requirements = detect_module_requirements_in_llvm_text(mnemonic);
        assert_eq!(requirements.features, DetectedFeatures::Sm80);
        assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx70);
        let (target, _) =
            resolve_ptx_target(None, None, requirements.features).expect("auto-resolve");
        assert_eq!(target, "sm_80");

        for near_miss in [
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
            "mma.sp.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32x {$0}, {$1}, {$2}, {$3};",
            "not_mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
            "$mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
            "%mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
            "@mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
            "!mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
            "@!mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
            "not$mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
            "/mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
            ")mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {$0}, {$1}, {$2}, {$3};",
        ] {
            assert!(
                !contains_mma_m16n8k8_f32_tf32_features(near_miss),
                "matched {near_miss:?}"
            );
        }

        let sm_75: CudaArch = "sm_75".parse().unwrap();
        let sm_80: CudaArch = "sm_80".parse().unwrap();
        assert!(validate_target_features(&sm_75, requirements.features).is_err());
        assert!(validate_target_features(&sm_80, requirements.features).is_ok());
        let error = resolve_ptx_target(Some("sm_75"), None, requirements.features)
            .expect_err("sm_75 must not accept TF32 tensor-core MMA")
            .to_string();
        assert!(
            error.contains("cannot lower detected feature Sm80"),
            "{error}"
        );

        let combined = format!("{mnemonic}\nmovmatrix.sync.aligned.m8n8.trans.b16 $0, $1;");
        assert_eq!(
            detect_module_requirements_in_llvm_text(&combined),
            ModuleRequirements {
                features: DetectedFeatures::Sm80 | DetectedFeatures::Movmatrix,
                ptx_isa: PtxIsaRequirement::Ptx78,
            }
        );
    }

    #[test]
    fn int8_mma_detection_applies_exact_sm80_and_ptx70_floors() {
        let mnemonic = concat!(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ",
            "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
        );
        let satfinite_mnemonic = concat!(
            "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 ",
            "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
        );
        for spelling in [
            mnemonic,
            satfinite_mnemonic,
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\t{$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32\\09{$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32\t{$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32\\09{$0}, {$1}, {$2}, {$3};",
            ";mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            ";mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "prefix\\0Amma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "{mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "$L:mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "/* comment */mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "@p mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "@!%p\\09mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "@p mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        ] {
            assert!(
                contains_mma_m16n8k32_s32_s8_features(spelling),
                "missed {spelling:?}"
            );
        }

        for spelling in [mnemonic, satfinite_mnemonic] {
            let requirements = detect_module_requirements_in_llvm_text(spelling);
            assert_eq!(requirements.features, DetectedFeatures::Sm80, "{spelling}");
            assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx70, "{spelling}");
        }
        let requirements = detect_module_requirements_in_llvm_text(mnemonic);
        let (target, _) =
            resolve_ptx_target(None, None, requirements.features).expect("auto-resolve");
        assert_eq!(target, "sm_80");

        for near_miss in [
            "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.col.row.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32x {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.row.col.satfiniteX.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32x {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.u32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "not_mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "$mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "$mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "%mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "@mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "!mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "@!mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "not$mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "/mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            ")mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        ] {
            assert!(
                !contains_mma_m16n8k32_s32_s8_features(near_miss),
                "matched {near_miss:?}"
            );
        }

        let sm_75: CudaArch = "sm_75".parse().unwrap();
        let sm_80: CudaArch = "sm_80".parse().unwrap();
        assert!(validate_target_features(&sm_75, requirements.features).is_err());
        assert!(validate_target_features(&sm_80, requirements.features).is_ok());
        let error = resolve_ptx_target(Some("sm_75"), None, requirements.features)
            .expect_err("sm_75 must not accept INT8 tensor-core MMA")
            .to_string();
        assert!(
            error.contains("cannot lower detected feature Sm80"),
            "{error}"
        );

        let combined = format!("{mnemonic}\nmovmatrix.sync.aligned.m8n8.trans.b16 $0, $1;");
        assert_eq!(
            detect_module_requirements_in_llvm_text(&combined),
            ModuleRequirements {
                features: DetectedFeatures::Sm80 | DetectedFeatures::Movmatrix,
                ptx_isa: PtxIsaRequirement::Ptx78,
            }
        );
    }

    #[test]
    fn mma_m8n8k4_f64_detection_enforces_sm80_and_ptx70() {
        let mnemonic = concat!(
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 ",
            "{$0, $1}, {$2}, {$3}, {$4, $5};"
        );
        for spelling in [
            mnemonic,
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64\t{$0, $1}, {$2}, {$3}, {$4, $5};",
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64\n{$0, $1}, {$2}, {$3}, {$4, $5};",
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64\\09{$0, $1}, {$2}, {$3}, {$4, $5};",
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64\\0A{$0, $1}, {$2}, {$3}, {$4, $5};",
            ";mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
            "prefix\\0Amma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
            "\"mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
            "{mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
            "$L:mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
            "/* comment */mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
            "@p mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
            "@!%p\\09mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
        ] {
            assert!(contains_mma_m8n8k4_f64_features(spelling), "{spelling:?}");
        }

        let requirements = detect_module_requirements_in_llvm_text(mnemonic);
        assert_eq!(
            requirements,
            ModuleRequirements {
                features: DetectedFeatures::Sm80,
                ptx_isa: PtxIsaRequirement::Ptx70,
            }
        );
        assert_eq!(select_target(requirements.features).unwrap(), "sm_80");

        for near_miss in [
            "mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
            "mma.sync.aligned.m8n8k4.col.row.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
            "mma.sync.aligned.m8n8k4.row.col.f32.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64x2 {$0, $1}, {$2}, {$3}, {$4, $5};",
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64\\5C09{$0, $1}, {$2}, {$3}, {$4, $5};",
            "not_mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
            "$mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
            "%mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
            "@mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
            "!mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
            "@!mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
            "not$mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
            "/mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
            ")mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {$0, $1}, {$2}, {$3}, {$4, $5};",
        ] {
            assert!(!contains_mma_m8n8k4_f64_features(near_miss));
            assert_eq!(
                detect_module_requirements_in_llvm_text(near_miss),
                ModuleRequirements {
                    features: DetectedFeatures::Basic,
                    ptx_isa: PtxIsaRequirement::Default,
                },
                "matched near-miss {near_miss}"
            );
        }

        let sm_75: CudaArch = "sm_75".parse().unwrap();
        let sm_80: CudaArch = "sm_80".parse().unwrap();
        assert!(validate_target_features(&sm_75, requirements.features).is_err());
        assert!(validate_target_features(&sm_80, requirements.features).is_ok());
        let error = resolve_ptx_target(Some("sm_75"), None, requirements.features)
            .expect_err("sm_75 must not accept FP64 tensor-core MMA")
            .to_string();
        assert!(
            error.contains("cannot lower detected feature Sm80"),
            "{error}"
        );

        let combined = format!("{mnemonic}\nmovmatrix.sync.aligned.m8n8.trans.b16 $0, $1;");
        assert_eq!(
            detect_module_requirements_in_llvm_text(&combined),
            ModuleRequirements {
                features: DetectedFeatures::Sm80 | DetectedFeatures::Movmatrix,
                ptx_isa: PtxIsaRequirement::Ptx78,
            }
        );
    }

    #[test]
    fn test_movmatrix_detection_separates_sm75_from_the_ptx78_floor() {
        let mnemonic = "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;";
        for spelling in [
            mnemonic,
            "movmatrix.sync.aligned.m8n8.trans.b16\t$0, $1;",
            "movmatrix.sync.aligned.m8n8.trans.b16\n$0, $1;",
            "movmatrix.sync.aligned.m8n8.trans.b16\\09$0, $1;",
            "movmatrix.sync.aligned.m8n8.trans.b16\\0A$0, $1;",
            "movmatrix.sync.aligned.m8n8.trans.b16\\0D\\0A$0, $1;",
        ] {
            assert!(contains_movmatrix_features(spelling), "{spelling:?}");
        }
        assert_eq!(
            detect_features_in_llvm_text(mnemonic),
            DetectedFeatures::Movmatrix
        );
        assert_eq!(select_target(DetectedFeatures::Movmatrix).unwrap(), "sm_75");
        assert_eq!(
            detect_module_requirements_in_llvm_text(mnemonic).ptx_isa,
            PtxIsaRequirement::Ptx78
        );

        for near_miss in [
            "movmatrix.sync.aligned.m8n8.b16 $0, $1;",
            "movmatrix.sync.aligned.m16n8.trans.b16 $0, $1;",
            "movmatrix.sync.aligned.m8n8.trans.b32 $0, $1;",
            "movmatrix.sync.aligned.m8n8.trans.b16x2 $0, $1;",
            "movmatrix.sync.aligned.m8n8.trans.b16\\5C09$0, $1;",
        ] {
            assert!(
                !contains_movmatrix_features(near_miss),
                "matched {near_miss}"
            );
            assert_eq!(
                detect_module_requirements_in_llvm_text(near_miss),
                ModuleRequirements {
                    features: DetectedFeatures::Basic,
                    ptx_isa: PtxIsaRequirement::Default,
                }
            );
        }

        let combined = format!("{mnemonic}\ncp.async.ca.shared.global [$0], [$1], 4;");
        assert_eq!(
            detect_module_requirements_in_llvm_text(&combined),
            ModuleRequirements {
                features: DetectedFeatures::Sm80 | DetectedFeatures::Movmatrix,
                ptx_isa: PtxIsaRequirement::Ptx78,
            },
            "the architecture and PTX ISA floors must compose independently"
        );

        let sm_70: CudaArch = "sm_70".parse().unwrap();
        let sm_75: CudaArch = "sm_75".parse().unwrap();
        let sm_80: CudaArch = "sm_80".parse().unwrap();
        assert!(validate_target_features(&sm_70, DetectedFeatures::Movmatrix).is_err());
        assert!(validate_target_features(&sm_75, DetectedFeatures::Movmatrix).is_ok());
        assert!(validate_target_features(&sm_80, DetectedFeatures::Movmatrix).is_ok());

        for target in ["sm_75", "sm_80", "sm_86", "sm_87"] {
            assert_eq!(
                required_ptx_feature(target, PtxIsaRequirement::Ptx78),
                Some("+ptx78"),
                "{target} needs an explicit PTX 7.8 floor"
            );
        }
        assert_eq!(
            required_ptx_feature("sm_90", PtxIsaRequirement::Ptx78),
            None
        );
        for target in ["sm_88", "sm_89"] {
            assert_eq!(
                required_ptx_feature(target, PtxIsaRequirement::Ptx78),
                None,
                "{target} already requires PTX 7.8 or newer"
            );
        }
        assert_eq!(
            required_ptx_feature("sm_75", PtxIsaRequirement::Default),
            None
        );
    }

    #[test]
    fn matrix_memory_detection_composes_architecture_and_ptx_isa_floors() {
        let base_ldmatrix = "ldmatrix.sync.aligned.m8n8.x4.b16 {$0, $1, $2, $3}, [$4];";
        assert_eq!(
            detect_module_requirements_in_llvm_text(base_ldmatrix),
            ModuleRequirements {
                features: DetectedFeatures::Ldmatrix,
                ptx_isa: PtxIsaRequirement::Ptx65,
            }
        );

        let cta_ldmatrix = "ldmatrix.sync.aligned.m8n8.x1.shared::cta.b16 {$0}, [$1];";
        assert_eq!(
            detect_module_requirements_in_llvm_text(cta_ldmatrix),
            ModuleRequirements {
                features: DetectedFeatures::Ldmatrix,
                ptx_isa: PtxIsaRequirement::Ptx78,
            }
        );

        for stmatrix in [
            "stmatrix.sync.aligned.m8n8.x1.b16 [$0], {$1};",
            "stmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16 [$0], {$1, $2, $3, $4};",
        ] {
            assert_eq!(
                detect_module_requirements_in_llvm_text(stmatrix),
                ModuleRequirements {
                    features: DetectedFeatures::Sm90,
                    ptx_isa: PtxIsaRequirement::Ptx78,
                }
            );
        }

        for newer in [
            "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {$0, $1}, [$2];",
            "ldmatrix.sync.aligned.m8n16.x2.shared::cta.b8x16.b6x16_p32 {$0, $1}, [$2];",
            "stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [$0], {$1};",
        ] {
            assert_eq!(
                detect_module_requirements_in_llvm_text(newer),
                ModuleRequirements {
                    features: DetectedFeatures::MatrixBlackwell
                        | if newer.starts_with("ldmatrix") {
                            DetectedFeatures::Ldmatrix
                        } else {
                            DetectedFeatures::Sm90
                        },
                    ptx_isa: PtxIsaRequirement::Ptx86,
                },
                "{newer}"
            );
        }

        let mixed = format!(
            "{base_ldmatrix}\n{}",
            "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;"
        );
        assert_eq!(
            detect_module_requirements_in_llvm_text(&mixed),
            ModuleRequirements {
                features: DetectedFeatures::Movmatrix | DetectedFeatures::Ldmatrix,
                ptx_isa: PtxIsaRequirement::Ptx78,
            },
            "the strongest PTX ISA floor must survive equal sm_75 feature families"
        );

        assert_eq!(
            required_ptx_feature("sm_75", PtxIsaRequirement::Ptx65),
            Some("+ptx65")
        );
        assert_eq!(
            required_ptx_feature("sm_80", PtxIsaRequirement::Ptx65),
            None
        );
        assert_eq!(
            required_ptx_feature("sm_100a", PtxIsaRequirement::Ptx86),
            None
        );

        let adjacent_unrelated_b8 = concat!(
            "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {$0}, [$1]; ",
            "mov.b8 $2, $3;"
        );
        assert_eq!(
            detect_module_requirements_in_llvm_text(adjacent_unrelated_b8),
            ModuleRequirements {
                features: DetectedFeatures::Ldmatrix,
                ptx_isa: PtxIsaRequirement::Ptx65,
            },
            "an unrelated b8 instruction must not raise the ldmatrix family"
        );
    }

    #[test]
    fn tma_and_wgmma_raise_their_independent_ptx_floors() {
        for tma in [
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes;",
            "cp.async.bulk.commit_group;",
            "cp.async.bulk.wait_group 0;",
            "cp.async.bulk.wait_group.read 0;",
        ] {
            let requirements = detect_module_requirements_in_llvm_text(tma);
            assert!(
                requirements.features.contains(DetectedFeatures::Tma),
                "{tma}"
            );
            assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx80, "{tma}");
        }

        let non_bulk = "cp.async.commit_group;";
        assert_eq!(
            detect_module_requirements_in_llvm_text(non_bulk),
            ModuleRequirements {
                features: DetectedFeatures::Sm80,
                ptx_isa: PtxIsaRequirement::Default,
            }
        );

        let tma_and_movmatrix = concat!(
            "cp.async.bulk.commit_group; ",
            "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;"
        );
        assert_eq!(
            detect_module_requirements_in_llvm_text(tma_and_movmatrix).ptx_isa,
            PtxIsaRequirement::Ptx80
        );

        let wgmma = "wgmma.fence.sync.aligned;";
        assert_eq!(
            detect_module_requirements_in_llvm_text(wgmma),
            ModuleRequirements {
                features: DetectedFeatures::Wgmma,
                ptx_isa: PtxIsaRequirement::Ptx80,
            }
        );

        let shared_cta =
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes;";
        assert!(contains_tma_shared_cta_destination(shared_cta));
        let shared_cta_requirements = detect_module_requirements_in_llvm_text(shared_cta);
        assert_eq!(shared_cta_requirements.features, DetectedFeatures::Tma);
        assert_eq!(shared_cta_requirements.ptx_isa, PtxIsaRequirement::Ptx86);

        let shared_source = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group;";
        assert!(!contains_tma_shared_cta_destination(shared_source));
        assert_eq!(
            detect_module_requirements_in_llvm_text(shared_source).ptx_isa,
            PtxIsaRequirement::Ptx80
        );

        let cta_group = "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1;";
        assert_eq!(
            detect_module_requirements_in_llvm_text(cta_group).ptx_isa,
            PtxIsaRequirement::Ptx86
        );

        assert_eq!(
            required_ptx_feature("sm_90", PtxIsaRequirement::Ptx80),
            Some("+ptx80")
        );
        assert_eq!(
            required_ptx_feature("sm_90a", PtxIsaRequirement::Ptx86),
            Some("+ptx86")
        );
        assert_eq!(
            required_ptx_feature("sm_100a", PtxIsaRequirement::Ptx80),
            None
        );
    }

    #[test]
    fn related_cluster_mbarrier_and_clc_requirements_are_detected() {
        for ptx in [
            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$0];",
            "fence.mbarrier_init.release.cluster;",
        ] {
            let requirements = detect_module_requirements_in_llvm_text(ptx);
            assert!(
                requirements.features.contains(DetectedFeatures::Tma),
                "{ptx}"
            );
            assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx80, "{ptx}");
            assert!(arch_satisfies("sm_90", requirements.features));
        }

        for (ptx, expected_isa) in [
            (
                "mbarrier.init.shared.b64 [$0], 1;",
                PtxIsaRequirement::Ptx70,
            ),
            (
                "mbarrier.test_wait.parity.shared.b64 $0, [$1], $2;",
                PtxIsaRequirement::Ptx71,
            ),
            (
                "mbarrier.try_wait.parity.shared::cta.b64 $0, [$1], $2;",
                PtxIsaRequirement::Ptx78,
            ),
        ] {
            let requirements = detect_module_requirements_in_llvm_text(ptx);
            assert!(
                requirements.features.contains(DetectedFeatures::Sm80),
                "{ptx}"
            );
            assert_eq!(requirements.ptx_isa, expected_isa, "{ptx}");
            if ptx.contains("try_wait") {
                assert!(requirements.features.contains(DetectedFeatures::Tma));
                assert!(!arch_satisfies("sm_80", requirements.features));
            } else {
                assert!(arch_satisfies("sm_80", requirements.features));
                assert!(!arch_satisfies("sm_75", requirements.features));
            }
        }

        for ptx in [
            "redux.sync.add.u32 $0, $1, $2;",
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "cvt.rn.relu.bf16x2.f32 $0, $1, $2;",
            "cvt.rz.bf16x2.f32 $0, $1, $2;",
        ] {
            assert!(
                detect_features_in_llvm_text(ptx).contains(DetectedFeatures::Sm80),
                "{ptx}"
            );
        }
        assert_eq!(
            required_ptx_feature("sm_80", PtxIsaRequirement::Ptx70),
            None
        );
        assert_eq!(
            required_ptx_feature("sm_80", PtxIsaRequirement::Ptx71),
            Some("+ptx71")
        );
        for target in ["sm_86", "sm_87", "sm_88", "sm_89"] {
            assert_eq!(
                required_ptx_feature(target, PtxIsaRequirement::Ptx71),
                None,
                "{target} cannot be downgraded below its minimum PTX ISA"
            );
        }

        for ptx in [
            "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64 $0, [$1], $2;",
            "fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;",
            "fence.acquire.sync_restrict::shared::cluster.cluster;",
        ] {
            let requirements = detect_module_requirements_in_llvm_text(ptx);
            assert!(
                requirements.features.contains(DetectedFeatures::Tma),
                "{ptx}"
            );
            assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx86, "{ptx}");
            assert!(!arch_satisfies("sm_80", requirements.features));
        }

        for ptx in [
            "mbarrier.test_wait.acquire.cta.shared::cta.b64 $0, [$1], $2;",
            "mbarrier.arrive.release.cta.shared::cta.b64 $0, [$1];",
        ] {
            let requirements = detect_module_requirements_in_llvm_text(ptx);
            assert!(requirements.features.contains(DetectedFeatures::Tma));
            assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx80);
            assert!(!arch_satisfies("sm_80", requirements.features));
        }

        let cluster_sync = "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;";
        assert_eq!(
            detect_module_requirements_in_llvm_text(cluster_sync),
            ModuleRequirements {
                features: DetectedFeatures::Cluster,
                ptx_isa: PtxIsaRequirement::Ptx78,
            }
        );
        assert_eq!(select_target(DetectedFeatures::Cluster).unwrap(), "sm_90");

        let cluster_release = "barrier.cluster.arrive.release;";
        assert_eq!(
            detect_module_requirements_in_llvm_text(cluster_release).ptx_isa,
            PtxIsaRequirement::Ptx80
        );

        for ptx in [
            "fence.sc.cluster;",
            "fence.acq_rel.cluster;",
            "ld.shared::cluster.u32 $0, [$1];",
            "ld.acquire.cluster.global.u32 $0, [$1];",
            "getctarank.shared::cluster.u32 $0, $1;",
        ] {
            let requirements = detect_module_requirements_in_llvm_text(ptx);
            assert!(requirements.features.contains(DetectedFeatures::Cluster));
            assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx78);
            assert!(!arch_satisfies("sm_80", requirements.features));
        }

        for ptx in [
            "fence.acquire.cta;",
            "fence.release.gpu;",
            "fence.acquire.cluster;",
            "fence.release.sys;",
        ] {
            let requirements = detect_module_requirements_in_llvm_text(ptx);
            assert!(
                requirements.features.contains(DetectedFeatures::Sm90),
                "{ptx}"
            );
            assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx86, "{ptx}");
            assert_eq!(
                requirements.features.contains(DetectedFeatures::Cluster),
                ptx.contains(".cluster"),
                "{ptx}"
            );
            assert!(!arch_satisfies("sm_80", requirements.features));
        }

        let multimem = "multimem.red.relaxed.cluster.global.add.u32 [$0], $1;";
        let requirements = detect_module_requirements_in_llvm_text(multimem);
        assert_eq!(requirements.features, DetectedFeatures::Sm90);
        assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx86);
        assert_eq!(select_target(requirements.features).unwrap(), "sm_90");
        let multimem_debug_filename = r#"!9 = !DIFile(filename: "multimem.rs", directory: "/tmp")"#;
        assert_eq!(
            detect_module_requirements_in_llvm_text(multimem_debug_filename),
            ModuleRequirements {
                features: DetectedFeatures::Basic,
                ptx_isa: PtxIsaRequirement::Default,
            }
        );

        for multimem in [
            "multimem.ld_reduce.relaxed.cta.add.v4.e4m3 {$0, $1, $2, $3}, [$4];",
            "multimem.st.relaxed.gpu.e5m2 [$0], $1;",
            "multimem.ld_reduce.add.acc::f16.v4.e5m2 {$0, $1, $2, $3}, [$4];",
        ] {
            let requirements = detect_module_requirements_in_llvm_text(multimem);
            assert_eq!(
                requirements.features,
                DetectedFeatures::MultimemFp8 | DetectedFeatures::Sm90,
                "{multimem}"
            );
            assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx86, "{multimem}");
            assert_eq!(select_target(requirements.features).unwrap(), "sm_100a");
            for target in [
                "sm_100a", "sm_103a", "sm_110a", "sm_120a", "sm_121a", "sm_100f", "sm_103f",
                "sm_110f",
            ] {
                assert!(arch_satisfies(target, requirements.features), "{target}");
            }
            for target in ["sm_100", "sm_90a", "sm_120f", "sm_121f"] {
                assert!(!arch_satisfies(target, requirements.features), "{target}");
            }
        }

        let redux_f32 = "redux.sync.min.abs.NaN.f32 $0, $1, $2;";
        let requirements = detect_module_requirements_in_llvm_text(redux_f32);
        assert_eq!(
            requirements.features,
            DetectedFeatures::ReduxF32 | DetectedFeatures::Sm80
        );
        assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx86);
        assert_eq!(select_target(requirements.features).unwrap(), "sm_100a");
        for target in ["sm_100a", "sm_103a", "sm_100f", "sm_103f"] {
            assert!(arch_satisfies(target, requirements.features), "{target}");
        }
        for target in ["sm_100", "sm_110a", "sm_120a", "sm_121f"] {
            assert!(!arch_satisfies(target, requirements.features), "{target}");
        }

        for sreg in [
            "mov.u32 $0, %clusterid.x;",
            "mov.u32 $0, %nclusterid.z;",
            "mov.u32 $0, %cluster_ctarank;",
            "mov.u32 $0, %cluster_nctarank;",
            "mov.pred $0, %is_explicit_cluster;",
        ] {
            assert_eq!(
                detect_module_requirements_in_llvm_text(sreg),
                ModuleRequirements {
                    features: DetectedFeatures::Cluster,
                    ptx_isa: PtxIsaRequirement::Ptx78,
                },
                "{sreg}"
            );
        }

        let cluster_metadata = r#"!0 = !{!"cluster_dim_x", i32 2}
            !1 = !{!"cluster_dim_y", i32 1}
            !2 = !{!"cluster_dim_z", i32 1}"#;
        assert_eq!(
            detect_module_requirements_in_llvm_text(cluster_metadata),
            ModuleRequirements {
                features: DetectedFeatures::Cluster,
                ptx_isa: PtxIsaRequirement::Ptx78,
            }
        );
        let cluster_debug_local =
            r#"!8 = !DILocalVariable(name: "cluster_dim_x", scope: !1, file: !2, line: 3)"#;
        assert_eq!(
            detect_module_requirements_in_llvm_text(cluster_debug_local),
            ModuleRequirements {
                features: DetectedFeatures::Basic,
                ptx_isa: PtxIsaRequirement::Default,
            }
        );

        let elect = "elect.sync $0|p, $1;";
        assert_eq!(
            detect_module_requirements_in_llvm_text(elect),
            ModuleRequirements {
                features: DetectedFeatures::Sm90,
                ptx_isa: PtxIsaRequirement::Ptx80,
            }
        );

        let tcgen_wait = "tcgen05.wait::ld.sync.aligned;";
        assert_eq!(
            detect_module_requirements_in_llvm_text(tcgen_wait),
            ModuleRequirements {
                features: DetectedFeatures::Blackwell,
                ptx_isa: PtxIsaRequirement::Ptx86,
            }
        );

        let tcgen_debug_filename = r#"!7 = !DIFile(filename: "tcgen05.rs", directory: "/tmp")"#;
        assert_eq!(
            detect_module_requirements_in_llvm_text(tcgen_debug_filename),
            ModuleRequirements {
                features: DetectedFeatures::Basic,
                ptx_isa: PtxIsaRequirement::Default,
            }
        );

        let clc = "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 $0, $1;";
        assert_eq!(
            detect_module_requirements_in_llvm_text(clc),
            ModuleRequirements {
                features: DetectedFeatures::Sm100,
                ptx_isa: PtxIsaRequirement::Ptx86,
            }
        );
        assert_eq!(select_target(DetectedFeatures::Sm100).unwrap(), "sm_100");
        assert!(!arch_satisfies("sm_90", DetectedFeatures::Sm100));
        assert!(arch_satisfies("sm_120", DetectedFeatures::Sm100));

        let clc_multicast = "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [$0], [$1];";
        let requirements = detect_module_requirements_in_llvm_text(clc_multicast);
        assert_eq!(
            requirements.features,
            DetectedFeatures::Sm100 | DetectedFeatures::BlackwellFamily
        );
        assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx86);
        assert_eq!(select_target(requirements.features).unwrap(), "sm_100a");
        assert!(!arch_satisfies("sm_100", requirements.features));
        assert!(arch_satisfies("sm_120a", requirements.features));
        for arch in ["sm_100f", "sm_101f", "sm_110f", "sm_121f"] {
            assert!(arch_satisfies(arch, requirements.features), "{arch}");
        }
        for arch in ["sm_103a", "sm_121a"] {
            assert!(!arch_satisfies(arch, requirements.features), "{arch}");
        }
    }

    #[test]
    fn ptx86_tma_modes_enforce_their_architecture_families() {
        for ptx in [
            "cp.async.bulk.global.shared::cta.bulk_group.cp_mask [$0], [$1], 16, $2;",
            "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes;",
            "cp.async.bulk.tensor.3d.shared::cta.global.im2col::w.mbarrier::complete_tx::bytes;",
        ] {
            let requirements = detect_module_requirements_in_llvm_text(ptx);
            assert!(
                requirements.features.contains(DetectedFeatures::Tma),
                "{ptx}"
            );
            assert!(
                requirements.features.contains(DetectedFeatures::Sm100),
                "{ptx}"
            );
            assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx86, "{ptx}");
            assert!(!arch_satisfies("sm_90", requirements.features));
            assert!(arch_satisfies("sm_100", requirements.features));
        }

        for ptx in [
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes;",
            "cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4.bulk_group;",
            "cp.async.bulk.tensor.3d.shared::cta.global.im2col::w::128.mbarrier::complete_tx::bytes;",
            "cp.async.bulk.prefetch.tensor.3d.L2.global.im2col::w::128;",
        ] {
            let requirements = detect_module_requirements_in_llvm_text(ptx);
            assert!(
                requirements.features.contains(DetectedFeatures::Tma),
                "{ptx}"
            );
            assert!(
                requirements
                    .features
                    .contains(DetectedFeatures::BlackwellAccelerated),
                "{ptx}"
            );
            assert_eq!(requirements.ptx_isa, PtxIsaRequirement::Ptx86, "{ptx}");
            assert_eq!(select_target(requirements.features).unwrap(), "sm_100a");
            assert!(!arch_satisfies("sm_100", requirements.features));
            assert!(!arch_satisfies("sm_120a", requirements.features));
            assert!(arch_satisfies("sm_103f", requirements.features));
        }

        assert!(!contains_tma_sm100_features("custom.op.cp_mask $0;"));
        assert!(!contains_tma_blackwell_accelerated_features(
            "custom.tile::scatter4 $0;"
        ));
    }

    #[test]
    fn test_sm90_floor_wins_when_sm80_features_are_also_present() {
        let llvm = r#"
            call i32 asm pure "add.rn.bf16x2 $0, $1, $2;", "=r,r,r"(i32 %a, i32 %b)
            call void asm sideeffect "cp.async.ca.shared.global [%0], [%1], 4;", "l,l"()
        "#;

        assert!(contains_sm90_features(llvm));
        assert!(contains_sm80_features(llvm));
        assert_eq!(
            detect_features_in_llvm_text(llvm),
            DetectedFeatures::Sm90 | DetectedFeatures::Sm80
        );
    }

    #[test]
    fn test_tma_multicast_detection_requires_cta_mask() {
        let multicast = "call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile(i32 0, i1 1, i1 false)";
        let unicast = "call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile(i32 0, i1 0, i1 false)";
        let literal_multicast = "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster";
        let cg1 = "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1";
        let cg2 = "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2";
        let cg1_intrinsic = "call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %dst, i1 0, i1 false, i32 1)";
        let cg2_intrinsic = "call void @llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d(ptr addrspace(7) %dst, i1 1, i1 false, i32 2)";
        let unrelated_i32 = "call void @unrelated(i32 2)";

        assert!(contains_tma_multicast(multicast));
        assert!(contains_tma_multicast(literal_multicast));
        assert!(!contains_tma_multicast(unicast));
        assert_eq!(
            detect_features_in_llvm_text(multicast),
            DetectedFeatures::TmaMulticast | DetectedFeatures::Tma
        );
        assert_eq!(
            detect_features_in_llvm_text(literal_multicast),
            DetectedFeatures::TmaMulticast | DetectedFeatures::Tma | DetectedFeatures::Cluster
        );
        assert_eq!(detect_features_in_llvm_text(unicast), DetectedFeatures::Tma);
        assert_eq!(
            detect_features_in_llvm_text(cg1),
            DetectedFeatures::TmaCtaGroup | DetectedFeatures::Tma
        );
        assert_eq!(
            detect_features_in_llvm_text(cg1_intrinsic),
            DetectedFeatures::TmaCtaGroup | DetectedFeatures::Tma
        );
        assert_eq!(
            detect_features_in_llvm_text(cg2),
            DetectedFeatures::TmaCtaGroup
                | DetectedFeatures::TmaMulticast
                | DetectedFeatures::Tma
                | DetectedFeatures::Cluster
        );
        assert_eq!(
            detect_features_in_llvm_text(cg2_intrinsic),
            DetectedFeatures::TmaCtaGroup | DetectedFeatures::TmaMulticast | DetectedFeatures::Tma
        );
        assert!(!contains_tma_cta_group_features(unrelated_i32));
    }

    #[test]
    fn test_select_target_prefers_required_architecture() {
        for (features, expected) in [
            (DetectedFeatures::Blackwell, "sm_100a"),
            (DetectedFeatures::TmaCtaGroup, "sm_100a"),
            (DetectedFeatures::BlackwellAccelerated, "sm_100a"),
            (DetectedFeatures::BlackwellFamily, "sm_100a"),
            (DetectedFeatures::ReduxF32, "sm_100a"),
            (DetectedFeatures::MultimemFp8, "sm_100a"),
            (DetectedFeatures::TmaMulticast, "sm_100a"),
            (DetectedFeatures::MatrixBlackwell, "sm_100a"),
            (DetectedFeatures::Wgmma, "sm_90a"),
            (DetectedFeatures::Sm100, "sm_100"),
            (DetectedFeatures::Tma, "sm_100"),
            (DetectedFeatures::Cluster, "sm_90"),
            (DetectedFeatures::Sm90, "sm_90"),
            (DetectedFeatures::Sm80, "sm_80"),
            (DetectedFeatures::Movmatrix, "sm_75"),
            (DetectedFeatures::Ldmatrix, "sm_75"),
            (DetectedFeatures::Basic, "sm_80"),
        ] {
            assert_eq!(select_target(features).unwrap(), expected, "{features:?}");
        }
    }

    #[test]
    fn target_selection_enforces_feature_intersections() {
        let multicast = "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster";
        let hopper_pair = format!("{multicast};\nwgmma.fence.sync.aligned;");
        let hopper_requirements = detect_features_in_llvm_text(&hopper_pair);
        assert!(hopper_requirements.contains(DetectedFeatures::TmaMulticast));
        assert!(hopper_requirements.contains(DetectedFeatures::Wgmma));
        assert_eq!(select_target(hopper_requirements).unwrap(), "sm_90a");
        assert!(arch_satisfies("sm_90a", hopper_requirements));
        assert!(!arch_satisfies("sm_100a", hopper_requirements));

        let blackwell_pair = format!(
            "{multicast};\n{}",
            "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {$0, $1}, [$2];"
        );
        let blackwell_requirements = detect_features_in_llvm_text(&blackwell_pair);
        assert!(blackwell_requirements.contains(DetectedFeatures::TmaMulticast));
        assert!(blackwell_requirements.contains(DetectedFeatures::MatrixBlackwell));
        assert_eq!(select_target(blackwell_requirements).unwrap(), "sm_100a");
        assert!(arch_satisfies("sm_100a", blackwell_requirements));
        assert!(!arch_satisfies("sm_90a", blackwell_requirements));

        let impossible = DetectedFeatures::Wgmma | DetectedFeatures::MatrixBlackwell;
        let error = select_target(impossible).expect_err("families have no common target");
        assert!(error.contains("do not share a compatible GPU architecture"));
        assert!(resolve_ptx_target(Some("sm_90a"), None, impossible).is_err());
        assert!(resolve_ptx_target(Some("sm_100a"), None, impossible).is_err());
    }

    #[test]
    fn test_arch_major_parses_cuda_spelling() {
        assert_eq!(arch_compute_capability("sm_75"), Some(75));
        assert_eq!(arch_compute_capability("sm_100a"), Some(100));
        assert_eq!(arch_major("sm_75"), Some(7));
        assert_eq!(arch_major("sm_80"), Some(8));
        assert_eq!(arch_major("sm_90a"), Some(9));
        assert_eq!(arch_major("sm_100a"), Some(10));
        assert_eq!(arch_major("sm_103a"), Some(10));
        assert_eq!(arch_major("sm_120a"), Some(12));
        assert_eq!(arch_major("nvvm-ir"), None);
        assert_eq!(arch_major("sm_"), None);
    }

    #[test]
    fn ptx9_targets_require_an_llvm22_backend() {
        for target in ["sm_88", "sm_110", "sm_110a", "sm_110f"] {
            assert!(
                validate_target_for_llvm_major(target, Some(21)).is_err(),
                "{target}"
            );
            assert!(
                validate_target_for_llvm_major(target, None).is_err(),
                "{target}"
            );
            assert!(
                validate_target_for_llvm_major(target, Some(22)).is_ok(),
                "{target}"
            );
            assert!(
                validate_target_for_llvm_major(target, Some(23)).is_ok(),
                "{target}"
            );
        }
        for target in ["sm_87", "sm_103a", "sm_120a", "sm_121f"] {
            assert!(
                validate_target_for_llvm_major(target, Some(21)).is_ok(),
                "{target}"
            );
        }
        assert_eq!(target_minimum_ptx_isa(121), Some(88));
    }

    #[test]
    fn test_arch_satisfies_sm100_only_features() {
        // tcgen05 and explicit cta_group TMA are datacenter-Blackwell only:
        // consumer Blackwell (sm_120) and Hopper (sm_90) cannot run them, even
        // though 120 > 100. This is the gemm_sol regression guard.
        for f in [DetectedFeatures::Blackwell, DetectedFeatures::TmaCtaGroup] {
            assert!(arch_satisfies("sm_100a", f), "sm_100a must satisfy {f:?}");
            assert!(arch_satisfies("sm_103a", f), "sm_103a must satisfy {f:?}");
            assert!(arch_satisfies("sm_103f", f), "sm_103f must satisfy {f:?}");
            assert!(
                !arch_satisfies("sm_100", f),
                "generic sm_100 must NOT satisfy {f:?}"
            );
            assert!(
                !arch_satisfies("sm_120a", f),
                "sm_120a must NOT satisfy {f:?}"
            );
            assert!(
                !arch_satisfies("sm_90a", f),
                "sm_90a must NOT satisfy {f:?}"
            );
            assert!(
                !arch_satisfies("sm_102a", f),
                "unknown architecture-specific targets must not be accepted"
            );
            assert!(
                !arch_satisfies("sm_102f", f),
                "unknown family-specific targets must not be accepted"
            );
        }
    }

    #[test]
    fn test_arch_satisfies_base_tma_multicast_targets() {
        for arch in [
            "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_103f", "sm_110a", "sm_120", "sm_120a",
        ] {
            assert!(
                arch_satisfies(arch, DetectedFeatures::TmaMulticast),
                "{arch}"
            );
        }
        for arch in ["sm_80", "sm_89", "sm_102a", "sm_102f"] {
            assert!(
                !arch_satisfies(arch, DetectedFeatures::TmaMulticast),
                "{arch}"
            );
        }
    }

    #[test]
    fn test_arch_satisfies_wgmma_is_hopper_only() {
        assert!(arch_satisfies("sm_90a", DetectedFeatures::Wgmma));
        assert!(!arch_satisfies("sm_90", DetectedFeatures::Wgmma));
        assert!(!arch_satisfies("sm_100a", DetectedFeatures::Wgmma));
        assert!(!arch_satisfies("sm_120a", DetectedFeatures::Wgmma));
    }

    #[test]
    fn test_arch_satisfies_blackwell_matrix_family_targets() {
        for arch in [
            "sm_100a", "sm_101a", "sm_110a", "sm_120a", "sm_100f", "sm_103f", "sm_120f", "sm_121f",
        ] {
            assert!(
                arch_satisfies(arch, DetectedFeatures::MatrixBlackwell),
                "{arch}"
            );
            assert!(
                arch_satisfies(arch, DetectedFeatures::BlackwellFamily),
                "{arch}"
            );
        }
        for arch in [
            "sm_100a", "sm_101a", "sm_103a", "sm_110a", "sm_100f", "sm_103f", "sm_110f",
        ] {
            assert!(
                arch_satisfies(arch, DetectedFeatures::BlackwellAccelerated),
                "{arch}"
            );
        }
        for arch in ["sm_100", "sm_120a", "sm_120f", "sm_102f"] {
            assert!(
                !arch_satisfies(arch, DetectedFeatures::BlackwellAccelerated),
                "{arch}"
            );
        }
        for arch in ["sm_100", "sm_103", "sm_110", "sm_120", "sm_121a"] {
            assert!(arch_satisfies(arch, DetectedFeatures::Sm100), "{arch}");
        }
        for arch in ["sm_90a", "sm_102", "sm_102a"] {
            assert!(!arch_satisfies(arch, DetectedFeatures::Sm100), "{arch}");
        }
        for arch in ["sm_90a", "sm_100", "sm_102f", "sm_120"] {
            assert!(
                !arch_satisfies(arch, DetectedFeatures::MatrixBlackwell),
                "{arch}"
            );
            assert!(
                !arch_satisfies(arch, DetectedFeatures::BlackwellFamily),
                "{arch}"
            );
        }
    }

    #[test]
    fn test_arch_satisfies_forward_compatible_features() {
        // Plain TMA / cluster / sm_90-floor instructions lower on any sm_90+
        // device, sm_80-floor instructions on any sm_80+ device, movmatrix and
        // base ldmatrix on sm_75+, and basic kernels on Volta+.
        // So a consumer sm_120 GPU is a valid target for these (it runs locally
        // instead of being downgraded to the feature floor).
        for arch in ["sm_90a", "sm_100a", "sm_120a"] {
            assert!(arch_satisfies(arch, DetectedFeatures::Tma));
            assert!(arch_satisfies(arch, DetectedFeatures::Cluster));
            assert!(arch_satisfies(arch, DetectedFeatures::Sm90));
            assert!(arch_satisfies(arch, DetectedFeatures::Sm80));
            assert!(arch_satisfies(arch, DetectedFeatures::Movmatrix));
            assert!(arch_satisfies(arch, DetectedFeatures::Ldmatrix));
            assert!(arch_satisfies(arch, DetectedFeatures::Basic));
        }
        assert!(arch_satisfies("sm_80", DetectedFeatures::Sm80));
        assert!(!arch_satisfies("sm_75", DetectedFeatures::Sm80));
        assert!(arch_satisfies("sm_75", DetectedFeatures::Movmatrix));
        assert!(arch_satisfies("sm_80", DetectedFeatures::Movmatrix));
        assert!(!arch_satisfies("sm_70", DetectedFeatures::Movmatrix));
        assert!(arch_satisfies("sm_75", DetectedFeatures::Ldmatrix));
        assert!(!arch_satisfies("sm_70", DetectedFeatures::Ldmatrix));
        assert!(arch_satisfies("sm_80", DetectedFeatures::Basic));
        assert!(arch_satisfies("sm_75", DetectedFeatures::Basic));
        assert!(arch_satisfies("sm_70", DetectedFeatures::Basic));
        assert!(!arch_satisfies("sm_80", DetectedFeatures::Tma));
        assert!(!arch_satisfies("sm_80", DetectedFeatures::Sm90));
        assert!(!arch_satisfies("sm_80a", DetectedFeatures::Basic));
        assert!(!arch_satisfies("sm_90f", DetectedFeatures::Tma));
    }
}
