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
use crate::generated::{
    GeneratedModuleRequirements, GeneratedResolvedRequirement, GeneratedResolvedTarget,
};
use crate::generated_intrinsic_targets::{
    GeneratedHardwareAlternative, GeneratedHardwareTarget, GeneratedTargetContract,
    GeneratedTargetRequirement,
};
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

/// Checks the dense Ampere INT8 MMA forms (PTX 7.0, sm_80+).
fn contains_dense_int8_mma_features(contents: &str) -> bool {
    const MNEMONICS: &[&str] = &[
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32",
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32",
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32",
        "mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32",
        "mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.s8.s32",
        "mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.u8.s32",
        "mma.sync.aligned.m16n8k16.row.col.satfinite.s32.u8.s8.s32",
        "mma.sync.aligned.m16n8k16.row.col.satfinite.s32.u8.u8.s32",
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32",
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32",
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32",
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32",
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32",
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.u8.s32",
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.s8.s32",
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.u8.s32",
    ];

    MNEMONICS
        .iter()
        .any(|mnemonic| contains_instruction_mnemonic(contents, mnemonic))
}

/// Checks the m8n8k16 INT8 MMA forms (PTX 6.5, sm_75+).
fn contains_mma_m8n8k16_int8_features(contents: &str) -> bool {
    const MNEMONICS: &[&str] = &[
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32",
        "mma.sync.aligned.m8n8k16.row.col.s32.s8.u8.s32",
        "mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32",
        "mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32",
        "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.s32",
        "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.u8.s32",
        "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.s8.s32",
        "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32",
    ];

    MNEMONICS
        .iter()
        .any(|mnemonic| contains_instruction_mnemonic(contents, mnemonic))
}

/// Checks the m8n8k32 INT4 MMA forms (PTX 6.5, sm_75+).
fn contains_mma_m8n8k32_int4_features(contents: &str) -> bool {
    const MNEMONICS: &[&str] = &[
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32",
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.u4.s32",
        "mma.sync.aligned.m8n8k32.row.col.s32.u4.s4.s32",
        "mma.sync.aligned.m8n8k32.row.col.s32.u4.u4.s32",
        "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.s4.s32",
        "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.u4.s32",
        "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.s4.s32",
        "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32",
    ];

    MNEMONICS
        .iter()
        .any(|mnemonic| contains_instruction_mnemonic(contents, mnemonic))
}

/// Checks the dense Ampere INT4 MMA forms (PTX 7.0, sm_80+).
fn contains_dense_int4_mma_features(contents: &str) -> bool {
    const MNEMONICS: &[&str] = &[
        "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32",
        "mma.sync.aligned.m16n8k32.row.col.s32.s4.u4.s32",
        "mma.sync.aligned.m16n8k32.row.col.s32.u4.s4.s32",
        "mma.sync.aligned.m16n8k32.row.col.s32.u4.u4.s32",
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s4.s4.s32",
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s4.u4.s32",
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u4.s4.s32",
        "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u4.u4.s32",
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32",
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.u4.s32",
        "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32",
        "mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32",
        "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32",
        "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.u4.s32",
        "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.u4.s4.s32",
        "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.u4.u4.s32",
    ];

    MNEMONICS
        .iter()
        .any(|mnemonic| contains_instruction_mnemonic(contents, mnemonic))
}

const B1_XOR_MMA_MNEMONICS: &[&str] = &[
    "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc",
    "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc",
    "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc",
];

const B1_AND_MMA_MNEMONICS: &[&str] = &[
    "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc",
    "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc",
    "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc",
];

/// Checks the three dense binary XOR/POPC MMA forms (PTX 7.0).
fn contains_b1_xor_mma_features(contents: &str) -> bool {
    B1_XOR_MMA_MNEMONICS
        .iter()
        .any(|mnemonic| contains_instruction_mnemonic(contents, mnemonic))
}

/// Checks the three dense binary AND/POPC MMA forms (PTX 7.1, sm_80+).
fn contains_b1_and_mma_features(contents: &str) -> bool {
    B1_AND_MMA_MNEMONICS
        .iter()
        .any(|mnemonic| contains_instruction_mnemonic(contents, mnemonic))
}

/// Checks the only dense binary MMA form that can run below sm_80.
fn contains_mma_m8n8k128_b1_xor_features(contents: &str) -> bool {
    contains_instruction_mnemonic(contents, B1_XOR_MMA_MNEMONICS[0])
}

fn contains_sm80_b1_mma_features(contents: &str) -> bool {
    contains_b1_and_mma_features(contents)
        || B1_XOR_MMA_MNEMONICS[1..]
            .iter()
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
        || contains_dense_int8_mma_features(contents)
        || contains_dense_int4_mma_features(contents)
        || contains_sm80_b1_mma_features(contents)
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
    /// Forward-compatible instructions with an sm_75 floor.
    pub(crate) const Sm75: Self = Self(1 << 17);
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

    const ALL: [Self; 18] = [
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
        Self::Sm75,
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
            Self::Sm75 => "Sm75",
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
    Ptx87,
    Ptx88,
    Ptx90,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModuleRequirements {
    pub features: DetectedFeatures,
    pub ptx_isa: PtxIsaRequirement,
}

/// Convert catalog PTX floors to the discrete `llc` feature spellings this
/// compiler supports. A floor between two spellings rounds upward; a future
/// floor beyond the newest supported spelling is rejected instead of ignored.
pub(crate) fn generated_ptx_isa_requirement(
    generated: &GeneratedModuleRequirements,
) -> Result<PtxIsaRequirement, String> {
    let mut requirement = PtxIsaRequirement::Default;
    for resolved in generated.resolved_targets() {
        let target_requirement = resolved_requirement(generated, resolved)?;
        let minimum_ptx =
            resolved_requirement_minimum_ptx(target_requirement).ok_or_else(|| {
                format!(
                    "generated intrinsic `{}` (`{}`) has an empty resolved target contract",
                    resolved.target.id, resolved.target.marker
                )
            })?;
        requirement = requirement.max(ptx_isa_requirement_for_floor(
            minimum_ptx,
            resolved.target.id,
            resolved.target.marker,
        )?);
    }
    Ok(requirement)
}

/// Resolve the PTX floor for the selected hardware alternative.
pub(crate) fn generated_ptx_isa_requirement_for_target(
    generated: &GeneratedModuleRequirements,
    arch: &str,
) -> Result<PtxIsaRequirement, String> {
    let mut requirement = PtxIsaRequirement::Default;
    for resolved in generated.resolved_targets() {
        let target_requirement = resolved_requirement(generated, resolved)?;
        let floor = resolved_requirement_ptx_floor(arch, target_requirement).ok_or_else(|| {
            format!(
                "CUDA target {arch} cannot lower generated intrinsic `{}` (`{}`); requires {}",
                resolved.target.id,
                resolved.target.marker,
                describe_resolved_requirement(target_requirement)
            )
        })?;
        requirement = requirement.max(ptx_isa_requirement_for_floor(
            floor,
            resolved.target.id,
            resolved.target.marker,
        )?);
    }
    validate_sm101_ptx_pair(arch, requirement)?;
    Ok(requirement)
}

fn resolved_requirement(
    generated: &GeneratedModuleRequirements,
    resolved: &GeneratedResolvedTarget,
) -> Result<GeneratedResolvedRequirement, String> {
    generated.resolved_requirement(resolved).ok_or_else(|| {
        let selector = resolved
            .selector
            .map(|selector| format!("{}={}", selector.name, selector.value))
            .unwrap_or_else(|| "<none>".to_string());
        format!(
            "generated intrinsic `{}` (`{}`) has no unique target contract for {selector}",
            resolved.target.id, resolved.target.marker
        )
    })
}

fn resolved_requirement_minimum_ptx(requirement: GeneratedResolvedRequirement) -> Option<u16> {
    match requirement {
        GeneratedResolvedRequirement::Target(requirement) => {
            Some(requirement.minimum_ptx.encoded())
        }
        GeneratedResolvedRequirement::Contract(contract) => contract
            .alternatives
            .iter()
            .map(|alternative| alternative.minimum_ptx.encoded())
            .min(),
    }
}

fn ptx_isa_requirement_for_floor(
    encoded: u16,
    id: &str,
    marker: &str,
) -> Result<PtxIsaRequirement, String> {
    match encoded {
        0..=60 => Ok(PtxIsaRequirement::Default),
        61..=62 => Ok(PtxIsaRequirement::Ptx62),
        63..=65 => Ok(PtxIsaRequirement::Ptx65),
        66..=70 => Ok(PtxIsaRequirement::Ptx70),
        71 => Ok(PtxIsaRequirement::Ptx71),
        72..=78 => Ok(PtxIsaRequirement::Ptx78),
        79..=80 => Ok(PtxIsaRequirement::Ptx80),
        81..=86 => Ok(PtxIsaRequirement::Ptx86),
        87 => Ok(PtxIsaRequirement::Ptx87),
        88 => Ok(PtxIsaRequirement::Ptx88),
        89..=90 => Ok(PtxIsaRequirement::Ptx90),
        _ => Err(format!(
            "generated intrinsic `{id}` (`{marker}`) requires PTX {}.{}, newer than cuda-oxide can request",
            encoded / 10,
            encoded % 10
        )),
    }
}

pub(crate) fn merge_generated_module_requirements(
    mut text: ModuleRequirements,
    generated: &GeneratedModuleRequirements,
) -> Result<ModuleRequirements, String> {
    text.ptx_isa = text.ptx_isa.max(generated_ptx_isa_requirement(generated)?);
    Ok(text)
}

pub(crate) fn merge_generated_module_requirements_for_target(
    mut text: ModuleRequirements,
    generated: &GeneratedModuleRequirements,
    arch: &str,
) -> Result<ModuleRequirements, String> {
    text.ptx_isa = text
        .ptx_isa
        .max(generated_ptx_isa_requirement_for_target(generated, arch)?);
    validate_sm101_ptx_pair(arch, text.ptx_isa)?;
    Ok(text)
}

/// Reject LLVM target names whose meaning changes at a newer PTX ISA.
fn validate_sm101_ptx_pair(arch: &str, requirement: PtxIsaRequirement) -> Result<(), String> {
    let Some((capability, suffix)) = arch_compute_capability_and_suffix(arch) else {
        return Ok(());
    };
    if capability == 101
        && matches!(suffix, Some('a' | 'f'))
        && requirement >= PtxIsaRequirement::Ptx90
    {
        return Err(format!(
            "CUDA target {arch} cannot be combined with PTX 9.0 or newer; LLVM renamed the sm_101 target to sm_110 at that PTX level"
        ));
    }
    Ok(())
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
            contains_mma_m8n8k16_int8_features(contents)
                || contains_mma_m8n8k32_int4_features(contents)
                || contains_mma_m8n8k128_b1_xor_features(contents),
            DetectedFeatures::Sm75,
        ),
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
    if contains_ldmatrix_features(contents)
        || contains_mma_m8n8k16_int8_features(contents)
        || contains_mma_m8n8k32_int4_features(contents)
    {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::Ptx65);
    }
    if contains_mbarrier_features(contents)
        || contents.contains("redux.sync")
        || contains_mma_m16n8k16_f32_bf16_features(contents)
        || contains_mma_m16n8k16_f32_f16_features(contents)
        || contains_mma_m16n8k8_f32_tf32_features(contents)
        || contains_dense_int8_mma_features(contents)
        || contains_dense_int4_mma_features(contents)
        || contains_b1_xor_mma_features(contents)
        || contains_mma_m8n8k4_f64_features(contents)
    {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::Ptx70);
    }
    if contains_mbarrier_ptx71_features(contents) {
        ptx_isa = ptx_isa.max(PtxIsaRequirement::Ptx71);
    }
    if contains_b1_and_mma_features(contents) {
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
    } else if features.contains(DetectedFeatures::Sm75)
        || features.contains(DetectedFeatures::Movmatrix)
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

/// Select one concrete target satisfying both text-detected features and every
/// generated intrinsic used by the module.
///
/// Generated hardware requirements are a module-wide AND. Each intrinsic's
/// `AnyOf` list remains an OR, so the search finds one architecture in the
/// intersection rather than selecting a separate target per call.
pub(crate) fn select_target_with_generated(
    features: DetectedFeatures,
    generated: &GeneratedModuleRequirements,
) -> Result<String, String> {
    let preferred = select_target(features)?;
    if generated.is_empty() {
        return Ok(preferred.to_string());
    }

    let mut candidates = vec![preferred.to_string()];
    let mut push_candidate = |candidate: String| {
        if !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    };

    // Try each catalog spelling before the exhaustive known-target list. This
    // preserves catalog alternative order while still finding intersections
    // such as `minimum sm_80` AND `sm_90a exactly`.
    for resolved in generated.resolved_targets() {
        let requirement = resolved_requirement(generated, resolved)?;
        for_each_resolved_hardware_alternative(requirement, |alternative| {
            push_candidate(generated_hardware_candidate(alternative));
        });
    }

    // This is the reviewed set accepted by `is_known_cuda_target`. Family and
    // architecture spellings are included because a generated requirement may
    // need to intersect with an existing text-detected feature.
    for candidate in [
        "sm_70", "sm_72", "sm_75", "sm_80", "sm_86", "sm_87", "sm_88", "sm_89", "sm_90", "sm_100",
        "sm_101", "sm_103", "sm_110", "sm_120", "sm_121", "sm_90a", "sm_100a", "sm_101a",
        "sm_103a", "sm_110a", "sm_120a", "sm_121a", "sm_100f", "sm_101f", "sm_103f", "sm_110f",
        "sm_120f", "sm_121f",
    ] {
        push_candidate(candidate.to_string());
    }

    if let Some(candidate) = candidates.into_iter().find(|candidate| {
        arch_satisfies(candidate, features) && generated_target_satisfied(candidate, generated)
    }) {
        return Ok(candidate);
    }

    let generated_ids = generated
        .resolved_targets()
        .iter()
        .map(|resolved| {
            let selector = resolved
                .selector
                .map(|selector| format!(" for {}={}", selector.name, selector.value))
                .unwrap_or_default();
            format!(
                "{} ({}){selector}",
                resolved.target.id, resolved.target.marker
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!(
        "detected CUDA features {features:?} and generated intrinsics [{generated_ids}] do not share a compatible GPU architecture"
    ))
}

pub(crate) fn generated_target_satisfied(
    arch: &str,
    generated: &GeneratedModuleRequirements,
) -> bool {
    generated.resolved_targets().iter().all(|resolved| {
        let Ok(requirement) = resolved_requirement(generated, resolved) else {
            return false;
        };
        resolved_hardware_satisfied(arch, requirement)
            && resolved_requirement_ptx_floor(arch, requirement).is_some_and(|floor| {
                ptx_isa_requirement_for_floor(floor, resolved.target.id, resolved.target.marker)
                    .is_ok()
            })
    })
}

fn resolved_hardware_satisfied(arch: &str, requirement: GeneratedResolvedRequirement) -> bool {
    match requirement {
        GeneratedResolvedRequirement::Target(requirement) => {
            generated_hardware_satisfied(arch, requirement.hardware)
        }
        GeneratedResolvedRequirement::Contract(contract) => {
            generated_contract_satisfied(arch, contract)
        }
    }
}

fn generated_hardware_satisfied(arch: &str, hardware: GeneratedHardwareTarget) -> bool {
    let Some((capability, suffix)) = arch_compute_capability_and_suffix(arch) else {
        return false;
    };
    if !is_known_cuda_target(capability, suffix) {
        return false;
    }

    match hardware {
        GeneratedHardwareTarget::All => true,
        GeneratedHardwareTarget::AnyOf(alternatives) => alternatives.iter().any(|alternative| {
            generated_hardware_alternative_satisfied(capability, suffix, *alternative)
        }),
        GeneratedHardwareTarget::TargetMatrix { contracts } => contracts.iter().any(|contract| {
            contract.alternatives.iter().any(|alternative| {
                generated_hardware_alternative_satisfied(capability, suffix, alternative.hardware)
            })
        }),
    }
}

fn generated_contract_satisfied(arch: &str, contract: &GeneratedTargetContract) -> bool {
    let Some((capability, suffix)) = arch_compute_capability_and_suffix(arch) else {
        return false;
    };
    is_known_cuda_target(capability, suffix)
        && contract.alternatives.iter().any(|alternative| {
            generated_hardware_alternative_satisfied(capability, suffix, alternative.hardware)
        })
}

fn for_each_resolved_hardware_alternative(
    requirement: GeneratedResolvedRequirement,
    mut visit: impl FnMut(GeneratedHardwareAlternative),
) {
    match requirement {
        GeneratedResolvedRequirement::Target(requirement) => match requirement.hardware {
            GeneratedHardwareTarget::All => {}
            GeneratedHardwareTarget::AnyOf(alternatives) => {
                for alternative in alternatives {
                    visit(*alternative);
                }
            }
            GeneratedHardwareTarget::TargetMatrix { contracts } => {
                for contract in contracts {
                    for alternative in contract.alternatives {
                        visit(alternative.hardware);
                    }
                }
            }
        },
        GeneratedResolvedRequirement::Contract(contract) => {
            for alternative in contract.alternatives {
                visit(alternative.hardware);
            }
        }
    }
}

fn generated_hardware_candidate(alternative: GeneratedHardwareAlternative) -> String {
    match alternative {
        GeneratedHardwareAlternative::MinimumSm(capability) => format!("sm_{capability}"),
        GeneratedHardwareAlternative::ExactArchitecture(capability) => {
            format!("sm_{capability}a")
        }
        GeneratedHardwareAlternative::FamilyTarget(capability) => format!("sm_{capability}f"),
    }
}

fn generated_hardware_requirement_label(alternative: GeneratedHardwareAlternative) -> String {
    match alternative {
        GeneratedHardwareAlternative::MinimumSm(capability) => {
            format!("sm_{capability} or newer")
        }
        GeneratedHardwareAlternative::ExactArchitecture(capability) => {
            format!("sm_{capability}a exactly")
        }
        GeneratedHardwareAlternative::FamilyTarget(capability) => {
            format!("sm_{capability}f exactly")
        }
    }
}

fn generated_hardware_alternative_satisfied(
    capability: u32,
    suffix: Option<char>,
    alternative: GeneratedHardwareAlternative,
) -> bool {
    match alternative {
        GeneratedHardwareAlternative::MinimumSm(minimum) => capability >= u32::from(minimum),
        GeneratedHardwareAlternative::ExactArchitecture(exact) => {
            capability == u32::from(exact) && suffix == Some('a')
        }
        // Family targets match only the named `sm_Nf` spelling.
        GeneratedHardwareAlternative::FamilyTarget(family) => {
            capability == u32::from(family) && suffix == Some('f')
        }
    }
}

fn generated_requirement_ptx_floor(
    arch: &str,
    requirement: GeneratedTargetRequirement,
) -> Option<u16> {
    let (capability, suffix) = arch_compute_capability_and_suffix(arch)?;
    if !is_known_cuda_target(capability, suffix) {
        return None;
    }
    match requirement.hardware {
        GeneratedHardwareTarget::All => Some(requirement.minimum_ptx.encoded()),
        GeneratedHardwareTarget::AnyOf(alternatives) => alternatives
            .iter()
            .any(|alternative| {
                generated_hardware_alternative_satisfied(capability, suffix, *alternative)
            })
            .then(|| requirement.minimum_ptx.encoded()),
        GeneratedHardwareTarget::TargetMatrix { contracts } => contracts
            .iter()
            .flat_map(|contract| contract.alternatives.iter())
            .filter(|alternative| {
                generated_hardware_alternative_satisfied(capability, suffix, alternative.hardware)
            })
            .map(|alternative| alternative.minimum_ptx.encoded())
            .min(),
    }
}

fn resolved_requirement_ptx_floor(
    arch: &str,
    requirement: GeneratedResolvedRequirement,
) -> Option<u16> {
    match requirement {
        GeneratedResolvedRequirement::Target(requirement) => {
            generated_requirement_ptx_floor(arch, requirement)
        }
        GeneratedResolvedRequirement::Contract(contract) => {
            let (capability, suffix) = arch_compute_capability_and_suffix(arch)?;
            is_known_cuda_target(capability, suffix).then_some(())?;
            contract
                .alternatives
                .iter()
                .filter(|alternative| {
                    generated_hardware_alternative_satisfied(
                        capability,
                        suffix,
                        alternative.hardware,
                    )
                })
                .map(|alternative| alternative.minimum_ptx.encoded())
                .min()
        }
    }
}

pub(crate) fn validate_generated_target(
    arch: &str,
    generated: &GeneratedModuleRequirements,
) -> Result<(), String> {
    for resolved in generated.resolved_targets() {
        let requirement = resolved_requirement(generated, resolved)?;
        if !resolved_hardware_satisfied(arch, requirement) {
            return Err(format!(
                "CUDA target {arch} cannot lower generated intrinsic `{}` (`{}`); requires {}",
                resolved.target.id,
                resolved.target.marker,
                describe_resolved_requirement(requirement)
            ));
        }
        ptx_isa_requirement_for_floor(
            resolved_requirement_ptx_floor(arch, requirement).unwrap(),
            resolved.target.id,
            resolved.target.marker,
        )?;
    }
    Ok(())
}

fn describe_generated_hardware(hardware: GeneratedHardwareTarget) -> String {
    match hardware {
        GeneratedHardwareTarget::All => "any supported CUDA target".to_string(),
        GeneratedHardwareTarget::AnyOf(alternatives) => alternatives
            .iter()
            .map(|alternative| generated_hardware_requirement_label(*alternative))
            .collect::<Vec<_>>()
            .join(" or "),
        GeneratedHardwareTarget::TargetMatrix { contracts } => contracts
            .iter()
            .map(describe_generated_contract)
            .collect::<Vec<_>>()
            .join(" or "),
    }
}

fn describe_resolved_requirement(requirement: GeneratedResolvedRequirement) -> String {
    match requirement {
        GeneratedResolvedRequirement::Target(requirement) => {
            describe_generated_hardware(requirement.hardware)
        }
        GeneratedResolvedRequirement::Contract(contract) => describe_generated_contract(contract),
    }
}

fn describe_generated_contract(contract: &GeneratedTargetContract) -> String {
    let alternatives = contract
        .alternatives
        .iter()
        .map(|alternative| {
            format!(
                "{} at PTX {}.{}",
                generated_hardware_requirement_label(alternative.hardware),
                alternative.minimum_ptx.major(),
                alternative.minimum_ptx.minor()
            )
        })
        .collect::<Vec<_>>()
        .join(" or ");
    if contract.selectors.is_empty() {
        alternatives
    } else {
        let selectors = contract
            .selectors
            .iter()
            .map(|selector| format!("{}={}", selector.name, selector.value))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{alternatives} for {selectors}")
    }
}

/// Does `arch` (e.g. `"sm_120a"`, `"sm_90"`) support the kernel's detected
/// features?
///
/// tcgen05/TMEM and explicit `cta_group` TMA forms exist only in the sm_100
/// datacenter-Blackwell family: consumer Blackwell (sm_120) and Hopper (sm_90)
/// lack them, so an sm_120 GPU cannot run an sm_100 tcgen05 kernel even though
/// 120 > 100. WGMMA is Hopper-only. The remaining features are forward
/// compatible from their floor (TMA / cluster / sm_90 features need sm_90+,
/// sm_80 features need sm_80+, sm_75 features need sm_75+, and basic needs
/// sm_70+).
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
        DetectedFeatures::BlackwellFamily => supports_blackwell_family_target(capability, suffix),
        DetectedFeatures::MatrixBlackwell => supports_blackwell_matrix_target(capability, suffix),
        DetectedFeatures::ReduxF32 => supports_redux_f32_target(capability, suffix),
        DetectedFeatures::MultimemFp8 => supports_multimem_fp8_target(capability, suffix),
        // The PTX ISA requires only sm_90+. The suffixed targets are advised
        // for performance, so target selection still prefers sm_100a.
        DetectedFeatures::TmaMulticast => major >= 9,
        DetectedFeatures::Wgmma => capability == 90 && suffix == Some('a'),
        DetectedFeatures::Sm100 => is_known_blackwell_capability(capability),
        DetectedFeatures::Tma | DetectedFeatures::Cluster | DetectedFeatures::Sm90 => major >= 9,
        DetectedFeatures::Sm80 => major >= 8,
        DetectedFeatures::Sm75 | DetectedFeatures::Movmatrix | DetectedFeatures::Ldmatrix => {
            capability >= 75
        }
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

fn supports_blackwell_matrix_target(capability: u32, suffix: Option<char>) -> bool {
    // LLVM's sm_101 aliases stop selecting these instructions at PTX 9.0.
    match suffix {
        Some('a' | 'f') => matches!(capability, 100 | 103 | 110 | 120 | 121),
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

#[cfg(test)]
pub fn resolve_ptx_target(
    explicit_override: Option<&str>,
    device_hint: Option<&str>,
    detected: DetectedFeatures,
) -> Result<(String, &'static str), PipelineError> {
    resolve_ptx_target_with_generated(
        explicit_override,
        device_hint,
        detected,
        &GeneratedModuleRequirements::default(),
    )
}

pub(crate) fn resolve_ptx_target_with_generated(
    explicit_override: Option<&str>,
    device_hint: Option<&str>,
    detected: DetectedFeatures,
    generated: &GeneratedModuleRequirements,
) -> Result<(String, &'static str), PipelineError> {
    if let Some(target) = explicit_override {
        let parsed = target.parse::<CudaArch>().map_err(|error| {
            PipelineError::PtxGeneration(format!("invalid CUDA_OXIDE_TARGET `{target}`: {error}"))
        })?;
        validate_target_features(&parsed, detected).map_err(PipelineError::PtxGeneration)?;
        validate_generated_target(&parsed.sm(), generated).map_err(PipelineError::PtxGeneration)?;
        return Ok((parsed.sm(), "CUDA_OXIDE_TARGET"));
    }

    if let Some(device) = device_hint.filter(|target| {
        arch_satisfies(target, detected) && generated_target_satisfied(target, generated)
    }) {
        return Ok((device.to_string(), "detected GPU"));
    }

    let target =
        select_target_with_generated(detected, generated).map_err(PipelineError::PtxGeneration)?;
    Ok((target, "feature requirement"))
}

/// Select the PTX ISA independently from the GPU architecture.
///
/// LLVM GPU CPUs select a default PTX ISA independently from the hardware
/// feature floor. Raise that ISA only when the selected CPU's default is too
/// old; never force a newer target back to an older PTX version.
pub fn required_ptx_feature(target: &str, requirement: PtxIsaRequirement) -> Option<&'static str> {
    let (capability, suffix) = arch_compute_capability_and_suffix(target)?;
    let minimum = target_minimum_ptx_isa(capability, suffix)?;
    let requested = match requirement {
        PtxIsaRequirement::Default => return None,
        PtxIsaRequirement::Ptx62 => 62,
        PtxIsaRequirement::Ptx65 => 65,
        PtxIsaRequirement::Ptx70 => 70,
        PtxIsaRequirement::Ptx71 => 71,
        PtxIsaRequirement::Ptx78 => 78,
        PtxIsaRequirement::Ptx80 => 80,
        PtxIsaRequirement::Ptx86 => 86,
        PtxIsaRequirement::Ptx87 => 87,
        PtxIsaRequirement::Ptx88 => 88,
        PtxIsaRequirement::Ptx90 => 90,
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
        PtxIsaRequirement::Ptx87 => Some("+ptx87"),
        PtxIsaRequirement::Ptx88 => Some("+ptx88"),
        PtxIsaRequirement::Ptx90 => Some("+ptx90"),
    }
}

/// Minimum PTX ISA accepted by LLVM for each concrete target. Passing an
/// older `+ptxNN` feature does not merely do nothing: LLVM aborts because that
/// ISA cannot name the selected processor.
fn target_minimum_ptx_isa(capability: u32, suffix: Option<char>) -> Option<u32> {
    match (capability, suffix) {
        (100 | 101 | 120, Some('f')) => Some(88),
        (capability, _) => match capability {
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
        },
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

pub(crate) fn validate_ptx_isa_for_llvm_major(
    requirement: PtxIsaRequirement,
    llc_major: Option<u32>,
) -> Result<(), String> {
    if requirement >= PtxIsaRequirement::Ptx90 && llc_major.is_none_or(|major| major < 22) {
        let backend = llc_major.map_or_else(
            || "an LLVM backend with an unknown version".to_string(),
            |major| format!("LLVM {major}"),
        );
        return Err(format!(
            "PTX 9.0 or newer requires LLVM 22 or newer; {backend} does not support the required PTX feature"
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
    use crate::generated_intrinsic_targets::{
        GeneratedBackendRequirement, GeneratedIntrinsicBackend, GeneratedIntrinsicTarget,
        GeneratedIntrinsicVariant, GeneratedPtxVersion, GeneratedTargetAlternative,
        GeneratedTargetRequirement, GeneratedTargetSelectorBinding,
    };

    static EXACT_SM120A: &[GeneratedHardwareAlternative] =
        &[GeneratedHardwareAlternative::ExactArchitecture(120)];
    static PTX87_EXACT_SM120A: GeneratedIntrinsicTarget = GeneratedIntrinsicTarget {
        marker: "test:ptx87",
        id: "ptx87_exact_sm120a",
        abi_id: "test",
        dialect_op: "test.ptx87",
        variant: GeneratedIntrinsicVariant::Scalar,
        requirement: GeneratedTargetRequirement {
            minimum_ptx: GeneratedPtxVersion::from_encoded(87),
            hardware: GeneratedHardwareTarget::AnyOf(EXACT_SM120A),
        },
        backend_requirements: &[],
        selections: &[],
        llvm: None,
    };
    static PTX88: GeneratedIntrinsicTarget = GeneratedIntrinsicTarget {
        marker: "test:ptx88",
        id: "ptx88",
        abi_id: "test",
        dialect_op: "test.ptx88",
        variant: GeneratedIntrinsicVariant::Scalar,
        requirement: GeneratedTargetRequirement {
            minimum_ptx: GeneratedPtxVersion::from_encoded(88),
            hardware: GeneratedHardwareTarget::All,
        },
        backend_requirements: &[],
        selections: &[],
        llvm: None,
    };
    static PTX90: GeneratedIntrinsicTarget = GeneratedIntrinsicTarget {
        marker: "test:ptx90",
        id: "ptx90",
        abi_id: "test",
        dialect_op: "test.ptx90",
        variant: GeneratedIntrinsicVariant::Scalar,
        requirement: GeneratedTargetRequirement {
            minimum_ptx: GeneratedPtxVersion::from_encoded(90),
            hardware: GeneratedHardwareTarget::All,
        },
        backend_requirements: &[],
        selections: &[],
        llvm: None,
    };
    static PTX91_FUTURE: GeneratedIntrinsicTarget = GeneratedIntrinsicTarget {
        marker: "test:ptx91",
        id: "ptx91_future",
        abi_id: "test",
        dialect_op: "test.ptx91",
        variant: GeneratedIntrinsicVariant::Scalar,
        requirement: GeneratedTargetRequirement {
            minimum_ptx: GeneratedPtxVersion::from_encoded(91),
            hardware: GeneratedHardwareTarget::All,
        },
        backend_requirements: &[],
        selections: &[],
        llvm: None,
    };
    static TCGEN_F16_SELECTORS: &[GeneratedTargetSelectorBinding] =
        &[GeneratedTargetSelectorBinding {
            name: "kind",
            value: "f16",
        }];
    static TCGEN_F16_TARGETS: &[GeneratedTargetAlternative] = &[
        GeneratedTargetAlternative {
            minimum_ptx: GeneratedPtxVersion::from_encoded(86),
            hardware: GeneratedHardwareAlternative::ExactArchitecture(100),
        },
        GeneratedTargetAlternative {
            minimum_ptx: GeneratedPtxVersion::from_encoded(86),
            hardware: GeneratedHardwareAlternative::ExactArchitecture(101),
        },
        GeneratedTargetAlternative {
            minimum_ptx: GeneratedPtxVersion::from_encoded(88),
            hardware: GeneratedHardwareAlternative::FamilyTarget(100),
        },
        GeneratedTargetAlternative {
            minimum_ptx: GeneratedPtxVersion::from_encoded(88),
            hardware: GeneratedHardwareAlternative::FamilyTarget(101),
        },
        GeneratedTargetAlternative {
            minimum_ptx: GeneratedPtxVersion::from_encoded(88),
            hardware: GeneratedHardwareAlternative::ExactArchitecture(103),
        },
        GeneratedTargetAlternative {
            minimum_ptx: GeneratedPtxVersion::from_encoded(88),
            hardware: GeneratedHardwareAlternative::FamilyTarget(103),
        },
        GeneratedTargetAlternative {
            minimum_ptx: GeneratedPtxVersion::from_encoded(90),
            hardware: GeneratedHardwareAlternative::ExactArchitecture(110),
        },
        GeneratedTargetAlternative {
            minimum_ptx: GeneratedPtxVersion::from_encoded(90),
            hardware: GeneratedHardwareAlternative::FamilyTarget(110),
        },
    ];
    static TCGEN_F16_CONTRACTS: &[GeneratedTargetContract] = &[GeneratedTargetContract {
        selectors: TCGEN_F16_SELECTORS,
        alternatives: TCGEN_F16_TARGETS,
    }];
    static TCGEN_F16: GeneratedIntrinsicTarget = GeneratedIntrinsicTarget {
        marker: "test:tcgen_f16",
        id: "tcgen_f16",
        abi_id: "test",
        dialect_op: "test.tcgen_f16",
        variant: GeneratedIntrinsicVariant::Scalar,
        requirement: GeneratedTargetRequirement {
            minimum_ptx: GeneratedPtxVersion::from_encoded(86),
            hardware: GeneratedHardwareTarget::TargetMatrix {
                contracts: TCGEN_F16_CONTRACTS,
            },
        },
        backend_requirements: &[],
        selections: &[],
        llvm: None,
    };
    static TCGEN_I8_SELECTORS: &[GeneratedTargetSelectorBinding] =
        &[GeneratedTargetSelectorBinding {
            name: "kind",
            value: "i8",
        }];
    static TCGEN_I8_TARGETS: &[GeneratedTargetAlternative] = &[
        GeneratedTargetAlternative {
            minimum_ptx: GeneratedPtxVersion::from_encoded(86),
            hardware: GeneratedHardwareAlternative::ExactArchitecture(100),
        },
        GeneratedTargetAlternative {
            minimum_ptx: GeneratedPtxVersion::from_encoded(86),
            hardware: GeneratedHardwareAlternative::ExactArchitecture(101),
        },
        GeneratedTargetAlternative {
            minimum_ptx: GeneratedPtxVersion::from_encoded(90),
            hardware: GeneratedHardwareAlternative::ExactArchitecture(110),
        },
    ];
    static TCGEN_I8_LIBNVVM_TARGETS: &[GeneratedTargetAlternative] = &[
        GeneratedTargetAlternative {
            minimum_ptx: GeneratedPtxVersion::from_encoded(86),
            hardware: GeneratedHardwareAlternative::ExactArchitecture(100),
        },
        GeneratedTargetAlternative {
            minimum_ptx: GeneratedPtxVersion::from_encoded(90),
            hardware: GeneratedHardwareAlternative::ExactArchitecture(110),
        },
    ];
    static TCGEN_I8_CONTRACTS: &[GeneratedTargetContract] = &[GeneratedTargetContract {
        selectors: TCGEN_I8_SELECTORS,
        alternatives: TCGEN_I8_TARGETS,
    }];
    static TCGEN_I8_LIBNVVM_CONTRACTS: &[GeneratedTargetContract] = &[GeneratedTargetContract {
        selectors: TCGEN_I8_SELECTORS,
        alternatives: TCGEN_I8_LIBNVVM_TARGETS,
    }];
    static TCGEN_I8_BACKENDS: &[GeneratedBackendRequirement] = &[GeneratedBackendRequirement {
        backend: GeneratedIntrinsicBackend::LibNvvm,
        requirement: GeneratedTargetRequirement {
            minimum_ptx: GeneratedPtxVersion::from_encoded(86),
            hardware: GeneratedHardwareTarget::TargetMatrix {
                contracts: TCGEN_I8_LIBNVVM_CONTRACTS,
            },
        },
    }];
    static TCGEN_I8: GeneratedIntrinsicTarget = GeneratedIntrinsicTarget {
        marker: "test:tcgen_i8",
        id: "tcgen_i8",
        abi_id: "test",
        dialect_op: "test.tcgen_i8",
        variant: GeneratedIntrinsicVariant::Scalar,
        requirement: GeneratedTargetRequirement {
            minimum_ptx: GeneratedPtxVersion::from_encoded(86),
            hardware: GeneratedHardwareTarget::TargetMatrix {
                contracts: TCGEN_I8_CONTRACTS,
            },
        },
        backend_requirements: TCGEN_I8_BACKENDS,
        selections: &[],
        llvm: None,
    };

    #[test]
    fn paired_target_floors_compose_with_target_cpu_minima() {
        let generated = GeneratedModuleRequirements::from_targets(vec![&TCGEN_F16]);

        assert_eq!(
            generated_ptx_isa_requirement(&generated).unwrap(),
            PtxIsaRequirement::Ptx86
        );
        for target in ["sm_100a", "sm_101a", "sm_103a", "sm_110a"] {
            assert!(generated_target_satisfied(target, &generated), "{target}");
        }
        for (target, requirement) in [
            ("sm_100a", PtxIsaRequirement::Ptx86),
            ("sm_101a", PtxIsaRequirement::Ptx86),
            ("sm_103a", PtxIsaRequirement::Ptx88),
            ("sm_110a", PtxIsaRequirement::Ptx90),
        ] {
            assert_eq!(
                generated_ptx_isa_requirement_for_target(&generated, target).unwrap(),
                requirement
            );
            assert_eq!(required_ptx_feature(target, requirement), None);
        }
        assert_eq!(
            generated_requirement_ptx_floor("sm_103a", TCGEN_F16.requirement),
            Some(88)
        );
        assert_eq!(
            generated_requirement_ptx_floor("sm_110a", TCGEN_F16.requirement),
            Some(90)
        );
        for target in ["sm_120a", "sm_121a"] {
            assert!(!generated_target_satisfied(target, &generated), "{target}");
        }
        for target in ["sm_100f", "sm_101f", "sm_103f", "sm_110f"] {
            assert!(generated_target_satisfied(target, &generated), "{target}");
        }
        assert_eq!(
            select_target_with_generated(DetectedFeatures::Basic, &generated).unwrap(),
            "sm_100a"
        );

        let generated = GeneratedModuleRequirements::from_targets(vec![&TCGEN_I8]);
        assert_eq!(
            generated_ptx_isa_requirement(&generated).unwrap(),
            PtxIsaRequirement::Ptx86
        );
        assert_eq!(
            generated_ptx_isa_requirement_for_target(&generated, "sm_100a").unwrap(),
            PtxIsaRequirement::Ptx86
        );
        assert_eq!(
            required_ptx_feature("sm_100a", PtxIsaRequirement::Ptx86),
            None
        );
        for (target, requirement) in [
            ("sm_101a", PtxIsaRequirement::Ptx86),
            ("sm_110a", PtxIsaRequirement::Ptx90),
        ] {
            assert_eq!(
                generated_ptx_isa_requirement_for_target(&generated, target).unwrap(),
                requirement
            );
            assert_eq!(required_ptx_feature(target, requirement), None);
        }
        assert!(!generated_target_satisfied("sm_103a", &generated));
        assert!(!generated_target_satisfied("sm_100f", &generated));
    }

    #[test]
    fn sm101_aliases_reject_an_aggregate_ptx90_requirement() {
        let generated = GeneratedModuleRequirements::from_targets(vec![&TCGEN_F16, &PTX90]);

        for target in ["sm_101a", "sm_101f"] {
            let error = generated_ptx_isa_requirement_for_target(&generated, target).unwrap_err();
            assert!(
                error.contains("renamed the sm_101 target to sm_110"),
                "{error}"
            );

            let f16 = GeneratedModuleRequirements::from_targets(vec![&TCGEN_F16]);
            let text = ModuleRequirements {
                features: DetectedFeatures::Basic,
                ptx_isa: PtxIsaRequirement::Ptx90,
            };
            let error =
                merge_generated_module_requirements_for_target(text, &f16, target).unwrap_err();
            assert!(
                error.contains("renamed the sm_101 target to sm_110"),
                "{error}"
            );
        }
    }

    #[test]
    fn paired_minimum_target_diagnostic_preserves_range_meaning() {
        static TARGETS: &[GeneratedTargetAlternative] = &[GeneratedTargetAlternative {
            minimum_ptx: GeneratedPtxVersion::from_encoded(88),
            hardware: GeneratedHardwareAlternative::MinimumSm(100),
        }];
        static CONTRACTS: &[GeneratedTargetContract] = &[GeneratedTargetContract {
            selectors: &[],
            alternatives: TARGETS,
        }];

        assert_eq!(
            describe_generated_hardware(GeneratedHardwareTarget::TargetMatrix {
                contracts: CONTRACTS,
            }),
            "sm_100 or newer at PTX 8.8"
        );
    }

    #[test]
    fn paired_target_matrix_flows_through_backend_target_resolution() {
        let llvm = GeneratedModuleRequirements::from_targets(vec![&TCGEN_I8])
            .for_backend(GeneratedIntrinsicBackend::LlvmNvptx);
        let libnvvm = GeneratedModuleRequirements::from_targets(vec![&TCGEN_I8])
            .for_backend(GeneratedIntrinsicBackend::LibNvvm);

        assert!(generated_target_satisfied("sm_101a", &llvm));
        assert!(!generated_target_satisfied("sm_101a", &libnvvm));
        assert!(!generated_target_satisfied("sm_103a", &llvm));
        assert!(generated_target_satisfied("sm_110a", &libnvvm));

        assert_eq!(
            resolve_ptx_target_with_generated(
                Some("sm_101a"),
                None,
                DetectedFeatures::Basic,
                &llvm,
            )
            .unwrap(),
            ("sm_101a".into(), "CUDA_OXIDE_TARGET")
        );
        assert!(
            resolve_ptx_target_with_generated(
                Some("sm_103a"),
                None,
                DetectedFeatures::Basic,
                &llvm,
            )
            .is_err()
        );
        assert_eq!(
            resolve_ptx_target_with_generated(
                None,
                Some("sm_101a"),
                DetectedFeatures::Basic,
                &llvm,
            )
            .unwrap(),
            ("sm_101a".into(), "detected GPU")
        );
        assert_eq!(
            resolve_ptx_target_with_generated(
                None,
                Some("sm_101a"),
                DetectedFeatures::Basic,
                &libnvvm,
            )
            .unwrap(),
            ("sm_100a".into(), "feature requirement")
        );
        assert_eq!(
            resolve_ptx_target_with_generated(None, None, DetectedFeatures::Basic, &llvm).unwrap(),
            ("sm_100a".into(), "feature requirement")
        );

        let base = ModuleRequirements {
            features: DetectedFeatures::Basic,
            ptx_isa: PtxIsaRequirement::Default,
        };
        assert_eq!(
            merge_generated_module_requirements_for_target(base, &llvm, "sm_101a")
                .unwrap()
                .ptx_isa,
            PtxIsaRequirement::Ptx86
        );
        assert_eq!(
            merge_generated_module_requirements_for_target(base, &llvm, "sm_110a")
                .unwrap()
                .ptx_isa,
            PtxIsaRequirement::Ptx90
        );

        let error = crate::export::resolve_nvvm_target_with_generated(
            Some("sm_103a"),
            None,
            None,
            &libnvvm,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("tcgen_i8"), "{error}");
        assert_eq!(
            crate::export::resolve_nvvm_target_with_generated(
                Some("sm_110a"),
                None,
                None,
                &libnvvm,
            )
            .unwrap()
            .sm(),
            "sm_110a"
        );
        assert_eq!(
            crate::export::resolve_nvvm_target_with_generated(
                None,
                Some("sm_103a"),
                None,
                &libnvvm,
            )
            .unwrap()
            .sm(),
            "sm_100a"
        );
    }

    #[test]
    fn generated_ptx87_exact_sm120a_requirement_is_preserved() {
        let generated = GeneratedModuleRequirements::from_targets(vec![&PTX87_EXACT_SM120A]);

        assert_eq!(
            generated_ptx_isa_requirement(&generated).unwrap(),
            PtxIsaRequirement::Ptx87
        );
        assert!(PtxIsaRequirement::Ptx86 < PtxIsaRequirement::Ptx87);
        assert_eq!(
            required_ptx_feature("sm_100a", PtxIsaRequirement::Ptx87),
            Some("+ptx87")
        );
        assert_eq!(
            required_ptx_feature("sm_120a", PtxIsaRequirement::Ptx87),
            None
        );
        assert_eq!(
            required_ptx_feature("sm_100f", PtxIsaRequirement::Ptx87),
            None
        );
        assert_eq!(
            required_ptx_feature("sm_120f", PtxIsaRequirement::Ptx87),
            None
        );
        assert_eq!(
            select_target_with_generated(DetectedFeatures::Basic, &generated).unwrap(),
            "sm_120a"
        );
        assert!(generated_target_satisfied("sm_120a", &generated));
        for incompatible in ["sm_120", "sm_120f", "sm_121a"] {
            assert!(
                !generated_target_satisfied(incompatible, &generated),
                "{incompatible}"
            );
        }
    }

    #[test]
    fn generated_ptx88_and_ptx90_floors_are_preserved() {
        let generated = GeneratedModuleRequirements::from_targets(vec![&PTX88]);
        assert_eq!(
            generated_ptx_isa_requirement(&generated).unwrap(),
            PtxIsaRequirement::Ptx88
        );
        assert_eq!(
            required_ptx_feature("sm_100a", PtxIsaRequirement::Ptx88),
            Some("+ptx88")
        );
        assert_eq!(
            required_ptx_feature("sm_103a", PtxIsaRequirement::Ptx88),
            None
        );
        assert_eq!(
            ptx_isa_requirement_for_floor(89, "test", "test").unwrap(),
            PtxIsaRequirement::Ptx90
        );
        assert_eq!(
            required_ptx_feature("sm_100a", PtxIsaRequirement::Ptx90),
            Some("+ptx90")
        );
        assert_eq!(
            required_ptx_feature("sm_110a", PtxIsaRequirement::Ptx90),
            None
        );
        validate_ptx_isa_for_llvm_major(PtxIsaRequirement::Ptx87, Some(21)).unwrap();
        validate_ptx_isa_for_llvm_major(PtxIsaRequirement::Ptx88, None).unwrap();
        validate_ptx_isa_for_llvm_major(PtxIsaRequirement::Ptx88, Some(21)).unwrap();
        validate_ptx_isa_for_llvm_major(PtxIsaRequirement::Ptx88, Some(22)).unwrap();
        assert!(validate_ptx_isa_for_llvm_major(PtxIsaRequirement::Ptx90, None).is_err());
        assert!(validate_ptx_isa_for_llvm_major(PtxIsaRequirement::Ptx90, Some(21)).is_err());
        validate_ptx_isa_for_llvm_major(PtxIsaRequirement::Ptx90, Some(22)).unwrap();

        let generated = GeneratedModuleRequirements::from_targets(vec![&PTX91_FUTURE]);
        let error = generated_ptx_isa_requirement(&generated).unwrap_err();

        assert!(error.contains("requires PTX 9.1"), "{error}");
        assert!(
            error.contains("newer than cuda-oxide can request"),
            "{error}"
        );
    }

    #[test]
    fn generated_redux_floor_matches_the_lowered_ptx_detector() {
        use crate::generated_intrinsic_targets::generated_intrinsic_target_by_marker;

        let target = generated_intrinsic_target_by_marker("v1:i0017").unwrap();
        let generated = GeneratedModuleRequirements::from_targets(vec![target]);
        let detected = detect_module_requirements_in_llvm_text("redux.sync.add.s32 $0, $1, $2;");

        assert_eq!(
            generated_ptx_isa_requirement(&generated).unwrap(),
            detected.ptx_isa
        );
        assert_eq!(detected.ptx_isa, PtxIsaRequirement::Ptx70);
        assert!(detected.features.contains(DetectedFeatures::Sm80));
        for arch in ["sm_75", "sm_80", "sm_90"] {
            assert_eq!(
                generated_target_satisfied(arch, &generated),
                arch_satisfies(arch, detected.features),
                "{arch}"
            );
        }
    }

    #[test]
    fn generated_packed_atomic_floors_are_backend_specific() {
        use crate::generated_intrinsic_targets::{
            GeneratedIntrinsicBackend, generated_intrinsic_target_by_marker,
        };

        let f16 = generated_intrinsic_target_by_marker("v1:i0014").unwrap();
        let llvm = GeneratedModuleRequirements::from_targets(vec![f16])
            .for_backend(GeneratedIntrinsicBackend::LlvmNvptx);
        assert!(generated_target_satisfied("sm_70", &llvm));
        assert_eq!(
            generated_ptx_isa_requirement(&llvm).unwrap(),
            PtxIsaRequirement::Ptx62
        );

        let libnvvm = GeneratedModuleRequirements::from_targets(vec![f16])
            .for_backend(GeneratedIntrinsicBackend::LibNvvm);
        assert!(!generated_target_satisfied("sm_70", &libnvvm));
        assert!(generated_target_satisfied("sm_75", &libnvvm));

        let bf16 = generated_intrinsic_target_by_marker("v1:i0015").unwrap();
        let bf16 = GeneratedModuleRequirements::from_targets(vec![bf16]);
        assert!(!generated_target_satisfied("sm_89", &bf16));
        assert!(generated_target_satisfied("sm_90", &bf16));
        let error =
            resolve_ptx_target_with_generated(Some("sm_89"), None, DetectedFeatures::Basic, &bf16)
                .unwrap_err()
                .to_string();
        assert!(error.contains("packed_atomic_add_bf16x2"), "{error}");
        assert!(error.contains("sm_90 or newer"), "{error}");
    }

    #[test]
    fn generated_non_mma_tcgen05_targets_preserve_the_backend_split() {
        use crate::generated_intrinsic_targets::{
            GENERATED_INTRINSIC_TARGETS, GeneratedIntrinsicBackend, GeneratedIntrinsicVariant,
        };

        let targets = GENERATED_INTRINSIC_TARGETS
            .iter()
            .filter(|target| {
                target.id.starts_with("tcgen05_")
                    && !matches!(target.variant, GeneratedIntrinsicVariant::Tcgen05Mma { .. })
            })
            .collect::<Vec<_>>();
        assert_eq!(targets.len(), 209);

        for target in targets {
            let llvm = GeneratedModuleRequirements::from_targets(vec![target])
                .for_backend(GeneratedIntrinsicBackend::LlvmNvptx);
            let libnvvm = GeneratedModuleRequirements::from_targets(vec![target])
                .for_backend(GeneratedIntrinsicBackend::LibNvvm);

            assert_eq!(
                generated_ptx_isa_requirement(&llvm).unwrap(),
                PtxIsaRequirement::Ptx86,
                "{}",
                target.id
            );
            assert_eq!(
                generated_ptx_isa_requirement(&libnvvm).unwrap(),
                PtxIsaRequirement::Ptx86,
                "{}",
                target.id
            );
            for arch in ["sm_100a", "sm_103a", "sm_110a"] {
                assert!(
                    generated_target_satisfied(arch, &llvm),
                    "{} {arch}",
                    target.id
                );
                assert!(
                    generated_target_satisfied(arch, &libnvvm),
                    "{} {arch}",
                    target.id
                );
            }
            assert!(
                generated_target_satisfied("sm_101a", &llvm),
                "{}",
                target.id
            );
            assert!(
                !generated_target_satisfied("sm_101a", &libnvvm),
                "{}",
                target.id
            );
            assert!(
                !generated_target_satisfied("sm_120a", &llvm),
                "{}",
                target.id
            );
            assert!(
                !generated_target_satisfied("sm_120a", &libnvvm),
                "{}",
                target.id
            );
        }
    }

    #[test]
    fn generated_packed_conversion_floors_require_ampere() {
        use crate::generated_intrinsic_targets::{
            GeneratedIntrinsicBackend, generated_intrinsic_target_by_marker,
        };

        for marker in [
            "v1:i0071", "v1:i0081", "v1:i0082", "v1:i0083", "v1:i0084", "v1:i0085",
        ] {
            let target = generated_intrinsic_target_by_marker(marker).unwrap();
            for backend in [
                GeneratedIntrinsicBackend::LlvmNvptx,
                GeneratedIntrinsicBackend::LibNvvm,
            ] {
                let generated =
                    GeneratedModuleRequirements::from_targets(vec![target]).for_backend(backend);
                assert_eq!(
                    generated_ptx_isa_requirement(&generated).unwrap(),
                    PtxIsaRequirement::Ptx70,
                    "{marker} {backend:?}"
                );
                assert!(!generated_target_satisfied("sm_75", &generated), "{marker}");
                assert!(generated_target_satisfied("sm_80", &generated), "{marker}");
                let error = validate_generated_target("sm_75", &generated).unwrap_err();
                assert!(error.contains(target.id), "{error}");
                assert!(error.contains("sm_80 or newer"), "{error}");
            }
        }
    }

    #[test]
    fn generated_cp_async_floors_require_ampere() {
        use crate::generated_intrinsic_targets::{
            GeneratedIntrinsicBackend, generated_intrinsic_target_by_marker,
        };

        for marker in [
            "v1:i0086", "v1:i0087", "v1:i0088", "v1:i0089", "v1:i0090", "v1:i0091", "v1:i0092",
            "v1:i0093", "v1:i0094", "v1:i0095", "v1:i0096", "v1:i0101", "v1:i0102", "v1:i0103",
            "v1:i0104",
        ] {
            let target = generated_intrinsic_target_by_marker(marker).unwrap();
            for backend in [
                GeneratedIntrinsicBackend::LlvmNvptx,
                GeneratedIntrinsicBackend::LibNvvm,
            ] {
                let generated =
                    GeneratedModuleRequirements::from_targets(vec![target]).for_backend(backend);
                assert_eq!(
                    generated_ptx_isa_requirement(&generated).unwrap(),
                    PtxIsaRequirement::Ptx70,
                    "{marker} {backend:?}"
                );
                assert!(!generated_target_satisfied("sm_75", &generated), "{marker}");
                assert!(generated_target_satisfied("sm_80", &generated), "{marker}");
                let error = validate_generated_target("sm_75", &generated).unwrap_err();
                assert!(error.contains(target.id), "{error}");
                assert!(error.contains("sm_80 or newer"), "{error}");
            }
        }
    }

    #[test]
    fn generated_dot_product_floors_record_sm61_and_split_backend_support() {
        use crate::generated_intrinsic_targets::{
            GeneratedIntrinsicBackend, generated_intrinsic_target_by_marker,
        };

        for marker in ["v1:i0030", "v1:i0031", "v1:i0032", "v1:i0033"] {
            let target = generated_intrinsic_target_by_marker(marker).unwrap();
            let llvm = GeneratedModuleRequirements::from_targets(vec![target])
                .for_backend(GeneratedIntrinsicBackend::LlvmNvptx);
            assert!(matches!(
                llvm.requirement(target).hardware,
                GeneratedHardwareTarget::AnyOf(alternatives)
                    if alternatives == [GeneratedHardwareAlternative::MinimumSm(61)]
            ));
            assert!(!generated_target_satisfied("sm_60", &llvm), "{marker}");
            assert!(generated_target_satisfied("sm_70", &llvm), "{marker}");
            assert_eq!(
                generated_ptx_isa_requirement(&llvm).unwrap(),
                PtxIsaRequirement::Default
            );

            let error = validate_generated_target("sm_60", &llvm)
                .unwrap_err()
                .to_string();
            assert!(error.contains(target.id), "{error}");
            assert!(error.contains("sm_61 or newer"), "{error}");

            let libnvvm = GeneratedModuleRequirements::from_targets(vec![target])
                .for_backend(GeneratedIntrinsicBackend::LibNvvm);
            assert!(!generated_target_satisfied("sm_74", &libnvvm));
            assert!(generated_target_satisfied("sm_75", &libnvvm));
        }
    }

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
        let mut forms = 0;
        for shape in ["m16n8k16", "m16n8k32"] {
            for a_type in ["s8", "u8"] {
                for b_type in ["s8", "u8"] {
                    for satfinite in [false, true] {
                        let overflow = if satfinite { ".satfinite" } else { "" };
                        let spelling = format!(
                            "mma.sync.aligned.{shape}.row.col{overflow}.s32.{a_type}.{b_type}.s32 {{$0}}, {{$1}}, {{$2}}, {{$3}};"
                        );
                        assert!(
                            contains_dense_int8_mma_features(&spelling),
                            "missed {spelling:?}"
                        );
                        assert_eq!(
                            detect_module_requirements_in_llvm_text(&spelling),
                            ModuleRequirements {
                                features: DetectedFeatures::Sm80,
                                ptx_isa: PtxIsaRequirement::Ptx70,
                            },
                            "{spelling}"
                        );
                        forms += 1;
                    }
                }
            }
        }
        assert_eq!(forms, 16);

        for spelling in [
            "mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.u8.s32\t{$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32\\09{$0}, {$1}, {$2}, {$3};",
            ";mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32 {$0}, {$1}, {$2}, {$3};",
            "prefix\\0Amma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.u8.s32 {$0}, {$1}, {$2}, {$3};",
            "@p mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32 {$0}, {$1}, {$2}, {$3};",
            "@!%p\\09mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        ] {
            assert!(
                contains_dense_int8_mma_features(spelling),
                "missed {spelling:?}"
            );
        }

        let representative = concat!(
            "mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.u8.s32 ",
            "{$0, $1, $2, $3}, {$4, $5}, {$6}, {$7, $8, $9, $10};"
        );
        let requirements = detect_module_requirements_in_llvm_text(representative);
        let (target, _) =
            resolve_ptx_target(None, None, requirements.features).expect("auto-resolve");
        assert_eq!(target, "sm_80");

        for near_miss in [
            "mma.sync.aligned.m16n8k8.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.col.row.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32x {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.row.col.satfiniteX.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32x {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.u32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k16.row.col.s32.s4.u8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k16.row.col.s32.u8.f16.s32 {$0}, {$1}, {$2}, {$3};",
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
                !contains_dense_int8_mma_features(near_miss),
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

        let combined = format!("{representative}\nmovmatrix.sync.aligned.m8n8.trans.b16 $0, $1;");
        assert_eq!(
            detect_module_requirements_in_llvm_text(&combined),
            ModuleRequirements {
                features: DetectedFeatures::Sm80 | DetectedFeatures::Movmatrix,
                ptx_isa: PtxIsaRequirement::Ptx78,
            }
        );
    }

    #[test]
    fn dense_int4_mma_detection_applies_exact_sm80_and_ptx70_floors() {
        let mut forms = 0;
        for shape in ["m16n8k32", "m16n8k64"] {
            for a_type in ["s4", "u4"] {
                for b_type in ["s4", "u4"] {
                    for satfinite in [false, true] {
                        let overflow = if satfinite { ".satfinite" } else { "" };
                        let spelling = format!(
                            "mma.sync.aligned.{shape}.row.col{overflow}.s32.{a_type}.{b_type}.s32 {{$0}}, {{$1}}, {{$2}}, {{$3}};"
                        );
                        assert!(
                            contains_dense_int4_mma_features(&spelling),
                            "missed {spelling:?}"
                        );
                        assert!(
                            !contains_mma_m8n8k32_int4_features(&spelling),
                            "m16 form entered the m8 INT4 detector: {spelling:?}"
                        );
                        assert!(
                            !contains_dense_int8_mma_features(&spelling),
                            "INT4 form entered the dense INT8 detector: {spelling:?}"
                        );
                        assert_eq!(
                            detect_module_requirements_in_llvm_text(&spelling),
                            ModuleRequirements {
                                features: DetectedFeatures::Sm80,
                                ptx_isa: PtxIsaRequirement::Ptx70,
                            },
                            "{spelling}"
                        );
                        forms += 1;
                    }
                }
            }
        }
        assert_eq!(forms, 16);

        for spelling in [
            "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s4.u4.s32\t{$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32\\09{$0}, {$1}, {$2}, {$3};",
            ";mma.sync.aligned.m16n8k32.row.col.s32.u4.u4.s32 {$0}, {$1}, {$2}, {$3};",
            "prefix\\0Amma.sync.aligned.m16n8k64.row.col.satfinite.s32.u4.u4.s32 {$0}, {$1}, {$2}, {$3};",
            "@p mma.sync.aligned.m16n8k32.row.col.s32.s4.u4.s32 {$0}, {$1}, {$2}, {$3};",
            "@!%p\\09mma.sync.aligned.m16n8k64.row.col.satfinite.s32.u4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        ] {
            assert!(
                contains_dense_int4_mma_features(spelling),
                "missed {spelling:?}"
            );
        }

        let representative = concat!(
            "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.u4.s32 ",
            "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
        );
        let requirements = detect_module_requirements_in_llvm_text(representative);
        assert_eq!(
            requirements,
            ModuleRequirements {
                features: DetectedFeatures::Sm80,
                ptx_isa: PtxIsaRequirement::Ptx70,
            }
        );
        assert_eq!(select_target(requirements.features).unwrap(), "sm_80");
        assert_eq!(
            required_ptx_feature("sm_75", requirements.ptx_isa),
            Some("+ptx70")
        );
        assert_eq!(required_ptx_feature("sm_80", requirements.ptx_isa), None);

        let sm_75: CudaArch = "sm_75".parse().unwrap();
        let sm_80: CudaArch = "sm_80".parse().unwrap();
        assert!(validate_target_features(&sm_75, requirements.features).is_err());
        assert!(validate_target_features(&sm_80, requirements.features).is_ok());
        let error = resolve_ptx_target(Some("sm_75"), None, requirements.features)
            .expect_err("sm_75 must not accept dense m16 INT4 MMA")
            .to_string();
        assert!(
            error.contains("cannot lower detected feature Sm80"),
            "{error}"
        );

        for target in [
            "sm_80", "sm_86", "sm_89", "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_120", "sm_120a",
        ] {
            assert!(
                arch_satisfies(target, requirements.features),
                "rejected {target}"
            );
        }

        let m8_int4 = concat!(
            "mma.sync.aligned.m8n8k32.row.col.s32.s4.u4.s32 ",
            "{$0, $1}, {$4}, {$5}, {$2, $3};"
        );
        let mixed = format!("{representative}\n{m8_int4}");
        assert_eq!(
            detect_module_requirements_in_llvm_text(&mixed),
            ModuleRequirements {
                features: DetectedFeatures::Sm80 | DetectedFeatures::Sm75,
                ptx_isa: PtxIsaRequirement::Ptx70,
            }
        );

        let newer_ptx = format!("{representative}\nmovmatrix.sync.aligned.m8n8.trans.b16 $0, $1;");
        assert_eq!(
            detect_module_requirements_in_llvm_text(&newer_ptx),
            ModuleRequirements {
                features: DetectedFeatures::Sm80 | DetectedFeatures::Movmatrix,
                ptx_isa: PtxIsaRequirement::Ptx78,
            }
        );
    }

    #[test]
    fn dense_int4_mma_detection_rejects_other_mma_families_and_near_misses() {
        for near_miss in [
            "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k16.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k64.row.col.s32.b1.b1.s32.xor.popc {$0}, {$1}, {$2}, {$3};",
            "mma.sp.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "wmma.mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.col.row.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k64.row.row.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32.satfinite {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k64.row.col.s32.satfinite.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.row.col.satfiniteX.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.u32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.row.col.s32.s4.u8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32x {$0}, {$1}, {$2}, {$3};",
            "not_mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "$mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "%mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "@mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "!mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            ")mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        ] {
            assert!(
                !contains_dense_int4_mma_features(near_miss),
                "matched {near_miss:?}"
            );
        }
    }

    #[test]
    fn dense_b1_mma_detection_applies_exact_operation_floors() {
        let cases = [
            (
                "m8n8k128",
                "xor",
                DetectedFeatures::Sm75,
                PtxIsaRequirement::Ptx70,
            ),
            (
                "m16n8k128",
                "xor",
                DetectedFeatures::Sm80,
                PtxIsaRequirement::Ptx70,
            ),
            (
                "m16n8k256",
                "xor",
                DetectedFeatures::Sm80,
                PtxIsaRequirement::Ptx70,
            ),
            (
                "m8n8k128",
                "and",
                DetectedFeatures::Sm80,
                PtxIsaRequirement::Ptx71,
            ),
            (
                "m16n8k128",
                "and",
                DetectedFeatures::Sm80,
                PtxIsaRequirement::Ptx71,
            ),
            (
                "m16n8k256",
                "and",
                DetectedFeatures::Sm80,
                PtxIsaRequirement::Ptx71,
            ),
        ];

        for (shape, operation, features, ptx_isa) in cases {
            let spelling = format!(
                "mma.sync.aligned.{shape}.row.col.s32.b1.b1.s32.{operation}.popc {{$0}}, {{$1}}, {{$2}}, {{$3}};"
            );
            assert_eq!(contains_b1_xor_mma_features(&spelling), operation == "xor");
            assert_eq!(contains_b1_and_mma_features(&spelling), operation == "and");
            assert_eq!(
                contains_mma_m8n8k128_b1_xor_features(&spelling),
                shape == "m8n8k128" && operation == "xor"
            );
            assert_eq!(
                detect_module_requirements_in_llvm_text(&spelling),
                ModuleRequirements { features, ptx_isa },
                "{spelling}"
            );
        }

        let m8_xor = concat!(
            "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc ",
            "{$0, $1}, {$4}, {$5}, {$2, $3};"
        );
        let m8_requirements = detect_module_requirements_in_llvm_text(m8_xor);
        assert_eq!(select_target(m8_requirements.features).unwrap(), "sm_75");
        assert_eq!(
            required_ptx_feature("sm_75", m8_requirements.ptx_isa),
            Some("+ptx70")
        );

        let m16_and = concat!(
            "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc ",
            "{$0, $1, $2, $3}, {$8, $9, $10, $11}, {$12, $13}, {$4, $5, $6, $7};"
        );
        let and_requirements = detect_module_requirements_in_llvm_text(m16_and);
        assert_eq!(select_target(and_requirements.features).unwrap(), "sm_80");
        assert_eq!(
            required_ptx_feature("sm_80", and_requirements.ptx_isa),
            Some("+ptx71")
        );
        let sm_75: CudaArch = "sm_75".parse().unwrap();
        let sm_80: CudaArch = "sm_80".parse().unwrap();
        assert!(validate_target_features(&sm_75, and_requirements.features).is_err());
        assert!(validate_target_features(&sm_80, and_requirements.features).is_ok());

        let combined = format!("{m8_xor}\n{m16_and}");
        assert_eq!(
            detect_module_requirements_in_llvm_text(&combined),
            ModuleRequirements {
                features: DetectedFeatures::Sm80 | DetectedFeatures::Sm75,
                ptx_isa: PtxIsaRequirement::Ptx71,
            }
        );
    }

    #[test]
    fn dense_b1_mma_detection_rejects_other_families_and_near_misses() {
        for near_miss in [
            "wmma.mma.xor.popc.sync.aligned.row.col.m8n8k128.s32.b1.b1.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k64.row.col.s32.b1.b1.s32.xor.popc {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k128.col.row.s32.b1.b1.s32.xor.popc {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.or.popc {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.popc.xor {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popcx {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc.satfinite {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.u32.xor.popc {$0}, {$1}, {$2}, {$3};",
            "not_mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc {$0}, {$1}, {$2}, {$3};",
            "$mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc {$0}, {$1}, {$2}, {$3};",
            "%mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc {$0}, {$1}, {$2}, {$3};",
        ] {
            assert!(
                !contains_b1_xor_mma_features(near_miss),
                "matched {near_miss:?}"
            );
            assert!(
                !contains_b1_and_mma_features(near_miss),
                "matched {near_miss:?}"
            );
        }
    }

    #[test]
    fn generated_b1_floors_match_text_detection_on_both_backends() {
        use crate::generated_intrinsic_targets::{
            GeneratedIntrinsicBackend, generated_intrinsic_target_by_marker,
        };

        for (marker, mnemonic) in [
            ("v1:i0157", B1_XOR_MMA_MNEMONICS[0]),
            ("v1:i0158", B1_XOR_MMA_MNEMONICS[1]),
            ("v1:i0159", B1_XOR_MMA_MNEMONICS[2]),
            ("v1:i0160", B1_AND_MMA_MNEMONICS[0]),
            ("v1:i0161", B1_AND_MMA_MNEMONICS[1]),
            ("v1:i0162", B1_AND_MMA_MNEMONICS[2]),
        ] {
            let instruction = format!("{mnemonic} {{$0}}, {{$1}}, {{$2}}, {{$3}};");
            let detected = detect_module_requirements_in_llvm_text(&instruction);
            let target = generated_intrinsic_target_by_marker(marker).unwrap();
            for backend in [
                GeneratedIntrinsicBackend::LlvmNvptx,
                GeneratedIntrinsicBackend::LibNvvm,
            ] {
                let generated =
                    GeneratedModuleRequirements::from_targets(vec![target]).for_backend(backend);
                assert_eq!(
                    generated_ptx_isa_requirement(&generated).unwrap(),
                    detected.ptx_isa,
                    "{marker} {backend:?}"
                );
                for arch in ["sm_70", "sm_75", "sm_80", "sm_90"] {
                    assert_eq!(
                        generated_target_satisfied(arch, &generated),
                        arch_satisfies(arch, detected.features),
                        "{marker} {backend:?} {arch}"
                    );
                }
            }
        }
    }

    #[test]
    fn m8n8k16_int8_mma_detection_applies_exact_sm75_and_ptx65_floors() {
        let mut forms = 0;
        for a_type in ["s8", "u8"] {
            for b_type in ["s8", "u8"] {
                for satfinite in [false, true] {
                    let overflow = if satfinite { ".satfinite" } else { "" };
                    let spelling = format!(
                        "mma.sync.aligned.m8n8k16.row.col{overflow}.s32.{a_type}.{b_type}.s32 {{$0, $1}}, {{$2}}, {{$3}}, {{$4, $5}};"
                    );
                    assert!(
                        contains_mma_m8n8k16_int8_features(&spelling),
                        "missed {spelling:?}"
                    );
                    assert!(
                        !contains_dense_int8_mma_features(&spelling),
                        "m8 form entered the m16 detector: {spelling:?}"
                    );
                    assert_eq!(
                        detect_module_requirements_in_llvm_text(&spelling),
                        ModuleRequirements {
                            features: DetectedFeatures::Sm75,
                            ptx_isa: PtxIsaRequirement::Ptx65,
                        },
                        "{spelling}"
                    );
                    forms += 1;
                }
            }
        }
        assert_eq!(forms, 8);

        for spelling in [
            "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.u8.s32\t{$0, $1}, {$2}, {$3}, {$4, $5};",
            "mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32\\09{$0, $1}, {$2}, {$3}, {$4, $5};",
            ";mma.sync.aligned.m8n8k16.row.col.s32.u8.u8.s32 {$0, $1}, {$2}, {$3}, {$4, $5};",
            "prefix\\0Amma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.u8.s32 {$0, $1}, {$2}, {$3}, {$4, $5};",
            "@p mma.sync.aligned.m8n8k16.row.col.s32.s8.u8.s32 {$0, $1}, {$2}, {$3}, {$4, $5};",
            "@!%p\\09mma.sync.aligned.m8n8k16.row.col.satfinite.s32.u8.s8.s32 {$0, $1}, {$2}, {$3}, {$4, $5};",
        ] {
            assert!(
                contains_mma_m8n8k16_int8_features(spelling),
                "missed {spelling:?}"
            );
        }

        let representative = concat!(
            "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.u8.s32 ",
            "{$0, $1}, {$4}, {$5}, {$2, $3};"
        );
        let requirements = detect_module_requirements_in_llvm_text(representative);
        let (target, _) =
            resolve_ptx_target(None, None, requirements.features).expect("auto-resolve");
        assert_eq!(target, "sm_75");
        assert_eq!(
            required_ptx_feature("sm_75", requirements.ptx_isa),
            Some("+ptx65")
        );
        assert_eq!(required_ptx_feature("sm_80", requirements.ptx_isa), None);

        let sm_70: CudaArch = "sm_70".parse().unwrap();
        let sm_75: CudaArch = "sm_75".parse().unwrap();
        let sm_80: CudaArch = "sm_80".parse().unwrap();
        assert!(validate_target_features(&sm_70, requirements.features).is_err());
        assert!(validate_target_features(&sm_75, requirements.features).is_ok());
        assert!(validate_target_features(&sm_80, requirements.features).is_ok());
        let error = resolve_ptx_target(Some("sm_70"), None, requirements.features)
            .expect_err("sm_70 must not accept m8n8k16 INT8 MMA")
            .to_string();
        assert!(
            error.contains("cannot lower detected feature Sm75"),
            "{error}"
        );

        for near_miss in [
            "mma.sync.aligned.m8n8k8.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m8n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m8n8k16.col.row.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sp.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32x {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m8n8k16.row.col.satfiniteX.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s8.u32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m8n8k16.row.col.s32.s4.u8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m8n8k16.row.col.s32.u8.f16.s32 {$0}, {$1}, {$2}, {$3};",
            "not_mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "$mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "%mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "@mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "!mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            ")mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
        ] {
            assert!(
                !contains_mma_m8n8k16_int8_features(near_miss),
                "matched {near_miss:?}"
            );
        }

        let m16 = concat!(
            "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 ",
            "{$0, $1, $2, $3}, {$4, $5}, {$6}, {$7, $8, $9, $10};"
        );
        assert!(!contains_mma_m8n8k16_int8_features(m16));
        assert!(contains_dense_int8_mma_features(m16));
        assert_eq!(
            detect_module_requirements_in_llvm_text(m16),
            ModuleRequirements {
                features: DetectedFeatures::Sm80,
                ptx_isa: PtxIsaRequirement::Ptx70,
            }
        );

        let combined = format!("{representative}\n{m16}");
        let combined_requirements = detect_module_requirements_in_llvm_text(&combined);
        assert_eq!(
            combined_requirements,
            ModuleRequirements {
                features: DetectedFeatures::Sm80 | DetectedFeatures::Sm75,
                ptx_isa: PtxIsaRequirement::Ptx70,
            }
        );
        let (target, _) = resolve_ptx_target(None, None, combined_requirements.features)
            .expect("combined m8 and m16 MMA should auto-resolve");
        assert_eq!(target, "sm_80");
    }

    #[test]
    fn m8n8k32_int4_mma_detection_applies_exact_sm75_and_ptx65_floors() {
        let mut forms = 0;
        for a_type in ["s4", "u4"] {
            for b_type in ["s4", "u4"] {
                for satfinite in [false, true] {
                    let overflow = if satfinite { ".satfinite" } else { "" };
                    let spelling = format!(
                        "mma.sync.aligned.m8n8k32.row.col{overflow}.s32.{a_type}.{b_type}.s32 {{$0, $1}}, {{$4}}, {{$5}}, {{$2, $3}};"
                    );
                    assert!(
                        contains_mma_m8n8k32_int4_features(&spelling),
                        "missed {spelling:?}"
                    );
                    assert!(
                        !contains_mma_m8n8k16_int8_features(&spelling),
                        "INT4 form entered the m8n8k16 detector: {spelling:?}"
                    );
                    assert!(
                        !contains_dense_int8_mma_features(&spelling),
                        "INT4 form entered the dense INT8 detector: {spelling:?}"
                    );
                    assert_eq!(
                        detect_module_requirements_in_llvm_text(&spelling),
                        ModuleRequirements {
                            features: DetectedFeatures::Sm75,
                            ptx_isa: PtxIsaRequirement::Ptx65,
                        },
                        "{spelling}"
                    );
                    forms += 1;
                }
            }
        }
        assert_eq!(forms, 8);

        for spelling in [
            "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.u4.s32\t{$0, $1}, {$4}, {$5}, {$2, $3};",
            "mma.sync.aligned.m8n8k32.row.col.s32.u4.s4.s32\\09{$0, $1}, {$4}, {$5}, {$2, $3};",
            ";mma.sync.aligned.m8n8k32.row.col.s32.u4.u4.s32 {$0, $1}, {$4}, {$5}, {$2, $3};",
            "prefix\\0Amma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32 {$0, $1}, {$4}, {$5}, {$2, $3};",
            "@p mma.sync.aligned.m8n8k32.row.col.s32.s4.u4.s32 {$0, $1}, {$4}, {$5}, {$2, $3};",
            "@!%p\\09mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.s4.s32 {$0, $1}, {$4}, {$5}, {$2, $3};",
        ] {
            assert!(
                contains_mma_m8n8k32_int4_features(spelling),
                "missed {spelling:?}"
            );
        }

        let representative = concat!(
            "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.u4.s32 ",
            "{$0, $1}, {$4}, {$5}, {$2, $3};"
        );
        let requirements = detect_module_requirements_in_llvm_text(representative);
        assert_eq!(
            requirements,
            ModuleRequirements {
                features: DetectedFeatures::Sm75,
                ptx_isa: PtxIsaRequirement::Ptx65,
            }
        );
        assert_eq!(select_target(requirements.features).unwrap(), "sm_75");
        assert_eq!(
            required_ptx_feature("sm_75", requirements.ptx_isa),
            Some("+ptx65")
        );
        assert_eq!(required_ptx_feature("sm_80", requirements.ptx_isa), None);

        let sm_72: CudaArch = "sm_72".parse().unwrap();
        let sm_75: CudaArch = "sm_75".parse().unwrap();
        let sm_80: CudaArch = "sm_80".parse().unwrap();
        assert!(validate_target_features(&sm_72, requirements.features).is_err());
        assert!(validate_target_features(&sm_75, requirements.features).is_ok());
        assert!(validate_target_features(&sm_80, requirements.features).is_ok());
        let error = resolve_ptx_target(Some("sm_72"), None, requirements.features)
            .expect_err("sm_72 must not accept m8n8k32 INT4 MMA")
            .to_string();
        assert!(
            error.contains("cannot lower detected feature Sm75"),
            "{error}"
        );
    }

    #[test]
    fn m8n8k32_int4_mma_detection_rejects_near_misses() {
        for near_miss in [
            "mma.sync.aligned.m8n8k32.row.col.s32.s8.s8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m8n8k32.row.col.s32.b1.b1.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sp.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m8n8k32.col.row.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m8n8k32.row.row.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32.satfinite {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m8n8k32.row.col.s32.satfinite.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m8n8k32.row.col.satfiniteX.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.s4.u32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m8n8k32.row.col.s32.s4.u8.s32 {$0}, {$1}, {$2}, {$3};",
            "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32x {$0}, {$1}, {$2}, {$3};",
            "not_mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "$mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "%mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "@mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            "!mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
            ")mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {$0}, {$1}, {$2}, {$3};",
        ] {
            assert!(
                !contains_mma_m8n8k32_int4_features(near_miss),
                "matched {near_miss:?}"
            );
        }
    }

    #[test]
    fn m8n8k32_int4_mma_requirements_compose_and_are_forward_compatible() {
        let int4 = concat!(
            "mma.sync.aligned.m8n8k32.row.col.s32.s4.u4.s32 ",
            "{$0, $1}, {$4}, {$5}, {$2, $3};"
        );
        let features = detect_features_in_llvm_text(int4);
        for target in [
            "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_120",
            "sm_120a",
        ] {
            assert!(arch_satisfies(target, features), "rejected {target}");
        }
        assert!(!arch_satisfies("sm_72", features));
        assert_eq!(
            resolve_ptx_target(None, Some("sm_120"), features).unwrap(),
            ("sm_120".to_string(), "detected GPU")
        );

        let m16_int8 = concat!(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 ",
            "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
        );
        let mixed = format!("{int4}\n{m16_int8}");
        let mixed_requirements = detect_module_requirements_in_llvm_text(&mixed);
        assert_eq!(
            mixed_requirements,
            ModuleRequirements {
                features: DetectedFeatures::Sm80 | DetectedFeatures::Sm75,
                ptx_isa: PtxIsaRequirement::Ptx70,
            }
        );
        assert_eq!(select_target(mixed_requirements.features).unwrap(), "sm_80");

        let newer_ptx = format!("{int4}\nmovmatrix.sync.aligned.m8n8.trans.b16 $0, $1;");
        assert_eq!(
            detect_module_requirements_in_llvm_text(&newer_ptx),
            ModuleRequirements {
                features: DetectedFeatures::Sm75 | DetectedFeatures::Movmatrix,
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
        assert_eq!(target_minimum_ptx_isa(100, Some('a')), Some(86));
        assert_eq!(target_minimum_ptx_isa(100, Some('f')), Some(88));
        assert_eq!(target_minimum_ptx_isa(120, Some('a')), Some(87));
        assert_eq!(target_minimum_ptx_isa(120, Some('f')), Some(88));
        assert_eq!(target_minimum_ptx_isa(121, Some('a')), Some(88));
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
            "sm_100a", "sm_103a", "sm_110a", "sm_120a", "sm_121a", "sm_100f", "sm_103f", "sm_110f",
            "sm_120f", "sm_121f",
        ] {
            assert!(
                arch_satisfies(arch, DetectedFeatures::MatrixBlackwell),
                "{arch}"
            );
        }
        for arch in [
            "sm_100a", "sm_101a", "sm_110a", "sm_120a", "sm_100f", "sm_101f", "sm_103f", "sm_110f",
            "sm_120f", "sm_121f",
        ] {
            assert!(
                arch_satisfies(arch, DetectedFeatures::BlackwellFamily),
                "{arch}"
            );
        }
        for arch in ["sm_101a", "sm_101f"] {
            assert!(!arch_satisfies(arch, DetectedFeatures::MatrixBlackwell));
        }
        for arch in ["sm_103a", "sm_121a"] {
            assert!(!arch_satisfies(arch, DetectedFeatures::BlackwellFamily));
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
