/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    ActiveMaskAdapter, BackendLoweringMechanism, CatalogFile, CatalogHardwareAlternative,
    CatalogHardwareTarget, CatalogIntrinsic, CatalogLlvm, CatalogSelection, CpAsyncCachePolicy,
    CpAsyncControlOperation, CpAsyncSourceSize, DotProductAdapter, DotProductOperation,
    DotProductSignedness, EvidenceArtifactKind, EvidenceStageKind, ImportedAddressSpace,
    IntrinsicBackend, IntrinsicSource, LdmatrixElement, LdmatrixLayout, LdmatrixMultiplicity,
    LdmatrixShape, LdmatrixStateSpace, PackedAluAdapter, PackedAluFormat, PackedAluOperation,
    PackedAtomicFormat, PackedConversionAdapter, PackedConversionDestinationFormat,
    PackedConversionRounding, PackedConversionSaturation, PackedConversionSourceFormat,
    ReduxAdapter, VoteAdapter, VoteMode, WarpBarrierAdapter, WarpMatchAdapter, WarpMatchMode,
    WarpShuffleAdapter, WarpShuffleMode, WarpShuffleOperandEncoding, WarpShuffleValueKind,
};
use anyhow::{Result, ensure};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::PathBuf;

pub fn all_outputs(
    catalog: &CatalogFile,
    catalog_json: String,
    catalog_sha256: &str,
) -> Result<BTreeMap<PathBuf, String>> {
    validate_renderable(catalog)?;
    let mut outputs = BTreeMap::new();
    outputs.insert("intrinsics/catalog.json".into(), catalog_json);
    outputs.insert(
        "crates/cuda-intrinsics/src/generated/mod.rs".into(),
        render_raw_mod(catalog, catalog_sha256),
    );
    outputs.insert(
        format!(
            "crates/cuda-intrinsics/src/generated/abi_v{}.rs",
            catalog.intrinsic_abi
        )
        .into(),
        render_raw_abi(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/cuda-device/src/generated/sreg.rs".into(),
        render_compat_sreg(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/cuda-device/src/generated/dotprod.rs".into(),
        render_compat_dotprod(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/cuda-device/src/generated/atomic.rs".into(),
        render_compat_packed_atomic(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/cuda-device/src/generated/async_copy.rs".into(),
        render_compat_cp_async_copy(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/cuda-device/src/generated/bf16x2.rs".into(),
        render_compat_packed_alu(catalog, catalog_sha256, PackedAluFormat::Bf16x2),
    );
    outputs.insert(
        "crates/cuda-device/src/generated/f16x2.rs".into(),
        render_compat_packed_alu(catalog, catalog_sha256, PackedAluFormat::F16x2),
    );
    outputs.insert(
        "crates/cuda-device/src/generated/tcgen05_conversion.rs".into(),
        render_compat_packed_conversion(
            catalog,
            catalog_sha256,
            "cuda_device::tcgen05::",
            "tcgen05",
            ("a", "b"),
        ),
    );
    outputs.insert(
        "crates/cuda-device/src/generated/convert.rs".into(),
        render_compat_packed_conversion(
            catalog,
            catalog_sha256,
            "cuda_device::convert::",
            "convert",
            ("lo", "hi"),
        ),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/mod.rs".into(),
        render_dialect_mod(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/sreg.rs".into(),
        render_dialect_sreg(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/dotprod.rs".into(),
        render_dialect_dotprod(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/ldmatrix.rs".into(),
        render_dialect_ldmatrix(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/packed_atomic.rs".into(),
        render_dialect_packed_atomic(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/packed_alu.rs".into(),
        render_dialect_packed_alu(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/packed_conversion.rs".into(),
        render_dialect_packed_conversion(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/redux.rs".into(),
        render_dialect_redux(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/sync.rs".into(),
        render_dialect_sync(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/vote.rs".into(),
        render_dialect_vote(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/active_mask.rs".into(),
        render_dialect_active_mask(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/cp_async.rs".into(),
        render_dialect_cp_async_copy(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/warp_match.rs".into(),
        render_dialect_warp_match(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/warp_barrier.rs".into(),
        render_dialect_warp_barrier(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/warp_shuffle.rs".into(),
        render_dialect_warp_shuffle(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/mir-importer/src/translator/terminator/intrinsics/generated.rs".into(),
        render_importer(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/mir-lower/src/convert/generated_intrinsics.rs".into(),
        render_lowering(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/rustc-codegen-cuda/src/generated_intrinsics.rs".into(),
        render_collector(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/cuda-oxide-codegen/src/generated_intrinsic_targets.rs".into(),
        render_targets(catalog, catalog_sha256),
    );
    for record in &catalog.intrinsics {
        outputs.insert(
            format!("intrinsics/probes/{}.ll", record.id).into(),
            render_probe(catalog, record, catalog_sha256),
        );
    }
    outputs.insert(
        "intrinsics/generated-reference.md".into(),
        render_reference(catalog, catalog_sha256),
    );
    Ok(outputs)
}

fn validate_renderable(catalog: &CatalogFile) -> Result<()> {
    ensure!(
        !catalog.intrinsics.is_empty(),
        "catalog contains no intrinsics"
    );
    for record in &catalog.intrinsics {
        if !record.rust.safe {
            ensure!(
                (record.family == "ldmatrix" && record.ldmatrix.is_some())
                    || (record.family == "packed_atomic" && record.packed_atomic.is_some())
                    || (record.family == "redux" && record.redux.is_some())
                    || (record.family == "vote" && record.vote.is_some())
                    || (record.family == "warp_match" && record.warp_match.is_some())
                    || (record.family == "warp_barrier" && record.warp_barrier.is_some())
                    || (record.family == "warp_shuffle" && record.warp_shuffle.is_some())
                    || (record.family == "cp_async_copy" && record.cp_async_copy.is_some())
                    || (record.family == "cp_async_control" && record.cp_async_control.is_some())
                    || record.family == "sync",
                "{} is unsafe but has no dedicated family safety renderer",
                record.id
            );
        }
        match record.family.as_str() {
            "sreg" => ensure!(
                record.rust.module == "sreg"
                    && record.rust.arguments.is_empty()
                    && llvm(record).arguments.is_empty()
                    && record.lowering == "direct_nvvm"
                    && record.scalar_width().is_some(),
                "{} is outside the zero-operand scalar direct-NVVM sreg recipe",
                record.id
            ),
            "ldmatrix" => ensure!(
                record.rust.module == "matrix"
                    && record.rust.arguments == ["*const u32"]
                    && record.lowering == "generated_ldmatrix"
                    && record.ldmatrix.is_some(),
                "{} is outside the generated ldmatrix recipe",
                record.id
            ),
            "packed_atomic" => ensure!(
                record.rust.module == "atomic"
                    && record.rust.arguments == ["*mut u32", "u32"]
                    && record.rust.result == "u32"
                    && record.rust.must_use
                    && record.llvm.is_none()
                    && record.lowering == "generated_packed_atomic_inline_ptx"
                    && record.packed_atomic.is_some(),
                "{} is outside the closed generated packed-atomic recipe",
                record.id
            ),
            "redux" => ensure!(
                record.rust.module == "warp"
                    && matches!(record.rust.arguments.as_slice(), [mask, value]
                        if mask == "u32" && value == &record.rust.result)
                    && matches!(record.rust.result.as_str(), "u32" | "i32")
                    && !record.rust.safe
                    && record.llvm.as_ref().is_some_and(|llvm| {
                        llvm.arguments == ["i32", "i32"] && llvm.results == ["i32"]
                    })
                    && record.dialect.operands == ["i32", "i32"]
                    && record.dialect.results == ["i32"]
                    && record.lowering == "generated_redux"
                    && record.redux.is_some(),
                "{} is outside the closed generated redux recipe",
                record.id
            ),
            "dotprod" => ensure!(
                record.rust.module == "dotprod"
                    && matches!(record.rust.arguments.as_slice(), [a, b, c]
                        if a == "u32" && b == "u32" && matches!(c.as_str(), "u32" | "i32"))
                    && record.rust.result == *record.rust.arguments.last().unwrap()
                    && record.rust.safe
                    && !record.rust.must_use
                    && record.llvm.as_ref().is_some_and(|llvm| {
                        llvm.results == ["i32"]
                            && (matches!(llvm.arguments.as_slice(), [a, b, c]
                                if a == "i32" && b == "i32" && c == "i32")
                                || matches!(llvm.arguments.as_slice(), [a, b, selector, c]
                                    if a == "i32" && b == "i32" && selector == "i1" && c == "i32"))
                    })
                    && record.dialect.operands == ["i32", "i32", "i32"]
                    && record.dialect.results == ["i32"]
                    && record.lowering == "generated_dotprod"
                    && record.dot_product.is_some(),
                "{} is outside the closed generated dot-product recipe",
                record.id
            ),
            "packed_alu" => ensure!(
                record.packed_alu.as_ref().is_some_and(|packed| {
                    let (module, must_use) = match packed.format {
                        PackedAluFormat::Bf16x2 => ("bf16x2", false),
                        PackedAluFormat::F16x2 => ("f16x2", true),
                    };
                    record.rust.module == module
                        && record.rust.must_use == must_use
                        && packed.adapter == PackedAluAdapter::DirectPackedU32
                })
                    && (1..=3).contains(&record.rust.arguments.len())
                    && record.rust.arguments.iter().all(|argument| argument == "u32")
                    && record.rust.result == "u32"
                    && record.rust.safe
                    && record.dialect.operands.len() == record.rust.arguments.len()
                    && record.dialect.operands.iter().all(|operand| operand == "i32")
                    && record.dialect.results == ["i32"]
                    && record.lowering == "generated_packed_alu_inline_ptx",
                "{} is outside the closed generated packed-ALU recipe",
                record.id
            ),
            "packed_conversion" => ensure!(
                record.rust.module == "convert"
                    && record.rust.arguments == ["f32", "f32"]
                    && record.rust.result == "u32"
                    && record.rust.safe
                    && !record.rust.must_use
                    && record.dialect.operands == ["f32", "f32"]
                    && record.dialect.results == ["i32"]
                    && record.lowering == "generated_packed_conversion_inline_ptx"
                    && record.packed_conversion.as_ref().is_some_and(|conversion| {
                        conversion.source_format == PackedConversionSourceFormat::F32x2
                            && conversion.adapter
                                == PackedConversionAdapter::ReverseHighLowOperands
                            && matches!(
                                (
                                    conversion.destination_format,
                                    conversion.rounding,
                                    conversion.saturation,
                                ),
                                (
                                    PackedConversionDestinationFormat::Bf16x2,
                                    PackedConversionRounding::NearestEven,
                                    PackedConversionSaturation::None,
                                ) | (
                                    PackedConversionDestinationFormat::F16x2,
                                    PackedConversionRounding::NearestEven,
                                    PackedConversionSaturation::None,
                                ) | (
                                    PackedConversionDestinationFormat::F16x2,
                                    PackedConversionRounding::TowardZero,
                                    PackedConversionSaturation::None,
                                ) | (
                                    PackedConversionDestinationFormat::F16x2,
                                    PackedConversionRounding::NearestEven,
                                    PackedConversionSaturation::Relu,
                                ) | (
                                    PackedConversionDestinationFormat::Bf16x2,
                                    PackedConversionRounding::NearestEven,
                                    PackedConversionSaturation::Relu,
                                ) | (
                                    PackedConversionDestinationFormat::Bf16x2,
                                    PackedConversionRounding::TowardZero,
                                    PackedConversionSaturation::None,
                                )
                            )
                    }),
                "{} is outside the closed generated packed-conversion recipe",
                record.id
            ),
            "cp_async_copy" => ensure!(
                record.rust.module == "async_copy"
                    && record.rust.result == "()"
                    && !record.rust.safe
                    && !record.rust.must_use
                    && record.llvm.as_ref().is_some_and(|llvm| {
                        matches!(llvm.arguments.as_slice(), [dst, src]
                            if dst == "shared_ptr" && src == "global_ptr")
                            || matches!(llvm.arguments.as_slice(), [dst, src, size]
                                if dst == "shared_ptr" && src == "global_ptr" && size == "i32")
                    })
                    && record.llvm.as_ref().is_some_and(|llvm| llvm.results.is_empty())
                    && record.dialect.results.is_empty()
                    && record.lowering == "generated_cp_async_copy"
                    && record.cp_async_copy.is_some(),
                "{} is outside the closed generated cp.async copy recipe",
                record.id
            ),
            "cp_async_control" => ensure!(
                record.rust.module == "async_copy"
                    && record.rust.result == "()"
                    && !record.rust.safe
                    && !record.rust.must_use
                    && record.llvm.as_ref().is_some_and(|llvm| {
                        llvm.results.is_empty()
                            && (llvm.arguments.is_empty() || llvm.arguments == ["i32"])
                    })
                    && record.dialect.results.is_empty()
                    && record.lowering == "generated_cp_async_control"
                    && record.cp_async_control.is_some(),
                "{} is outside the closed generated cp.async control recipe",
                record.id
            ),
            "sync" => ensure!(
                record.id == "sync_threads"
                    && record.rust.module == "thread"
                    && record.rust.name == "sync_threads"
                    && record.rust.arguments.is_empty()
                    && record.rust.result == "()"
                    && !record.rust.safe
                    && !record.rust.must_use
                    && record.llvm.as_ref().is_some_and(|llvm| {
                        llvm.symbol == "llvm.nvvm.barrier.cta.sync.aligned.all"
                            && llvm.arguments == ["i32"]
                            && llvm.results.is_empty()
                    })
                    && record.dialect.op_type == "Barrier0Op"
                    && record.dialect.op_name == "nvvm.barrier0"
                    && record.dialect.operands.is_empty()
                    && record.dialect.results.is_empty()
                    && record.lowering == "generated_sync_threads",
                "{} is outside the fixed-zero generated sync_threads recipe",
                record.id
            ),
            "vote" => ensure!(
                record.rust.module == "warp"
                    && record.rust.arguments == ["u32", "bool"]
                    && matches!(record.rust.result.as_str(), "bool" | "u32")
                    && !record.rust.safe
                    && record.rust.must_use
                    && record.llvm.as_ref().is_some_and(|llvm| {
                        llvm.arguments == ["i32", "i1"]
                            && matches!(llvm.results.as_slice(), [result]
                                if result == "i1" || result == "i32")
                    })
                    && record.dialect.operands == ["i32", "i1"]
                    && matches!(record.dialect.results.as_slice(), [result]
                        if result == "i1" || result == "i32")
                    && record.lowering == "generated_vote"
                    && record.vote.is_some(),
                "{} is outside the closed generated vote.sync recipe",
                record.id
            ),
            "active_mask" => ensure!(
                record.id == "active_mask"
                    && record.rust.module == "warp"
                    && record.rust.arguments.is_empty()
                    && record.rust.result == "u32"
                    && record.rust.safe
                    && record.rust.must_use
                    && record.llvm.as_ref().is_some_and(|llvm| {
                        llvm.arguments.is_empty() && llvm.results == ["i32"]
                    })
                    && record.dialect.operands.is_empty()
                    && record.dialect.results == ["i32"]
                    && record.lowering == "generated_active_mask"
                    && record.active_mask.is_some(),
                "{} is outside the closed generated active-mask recipe",
                record.id
            ),
            "warp_match" => ensure!(
                record.rust.module == "warp"
                    && matches!(record.rust.arguments.as_slice(), [mask, value]
                        if mask == "u32" && matches!(value.as_str(), "u32" | "u64"))
                    && record.rust.result == "u32"
                    && !record.rust.safe
                    && record.rust.must_use
                    && record.llvm.as_ref().is_some_and(|llvm| {
                        matches!(llvm.arguments.as_slice(), [mask, value]
                            if mask == "i32" && matches!(value.as_str(), "i32" | "i64"))
                            && (matches!(llvm.results.as_slice(), [mask] if mask == "i32")
                                || matches!(llvm.results.as_slice(), [mask, predicate]
                                    if mask == "i32" && predicate == "i1"))
                    })
                    && matches!(record.dialect.operands.as_slice(), [mask, value]
                        if mask == "i32" && matches!(value.as_str(), "i32" | "i64"))
                    && record.dialect.results == ["i32"]
                    && record.lowering == "generated_warp_match"
                    && record.warp_match.is_some(),
                "{} is outside the closed generated match.sync recipe",
                record.id
            ),
            "warp_barrier" => ensure!(
                record.id == "sync_mask"
                    && record.rust.module == "warp"
                    && record.rust.name == "sync_mask"
                    && record.rust.arguments == ["u32"]
                    && record.rust.result == "()"
                    && !record.rust.safe
                    && !record.rust.must_use
                    && record.llvm.as_ref().is_some_and(|llvm| {
                        llvm.symbol == "llvm.nvvm.bar.warp.sync"
                            && llvm.arguments == ["i32"]
                            && llvm.results.is_empty()
                    })
                    && record.dialect.op_type == "BarWarpSyncOp"
                    && record.dialect.op_name == "nvvm.bar_warp_sync"
                    && record.dialect.operands == ["i32"]
                    && record.dialect.results.is_empty()
                    && record.lowering == "generated_warp_barrier"
                    && record.warp_barrier.as_ref().is_some_and(|barrier| {
                        barrier.adapter == WarpBarrierAdapter::DirectMemberMask
                    }),
                "{} is outside the closed generated bar.warp.sync recipe",
                record.id
            ),
            "warp_shuffle" => ensure!(
                record.rust.module == "warp"
                    && matches!(record.rust.arguments.as_slice(), [mask, value, lane]
                        if mask == "u32"
                            && lane == "u32"
                            && matches!(value.as_str(), "u32" | "f32" | "u64")
                            && value == &record.rust.result)
                    && !record.rust.safe
                    && record.rust.must_use
                    && matches!(record.dialect.operands.as_slice(), [mask, value, lane]
                        if mask == "i32"
                            && lane == "i32"
                            && matches!(value.as_str(), "i32" | "f32" | "i64"))
                    && matches!(record.dialect.results.as_slice(), [result]
                        if result == &record.dialect.operands[1])
                    && record.warp_shuffle.as_ref().is_some_and(|shuffle| {
                        match shuffle.value_kind {
                            WarpShuffleValueKind::I32 | WarpShuffleValueKind::F32 => {
                                let (value_ty, rust_ty) = match shuffle.value_kind {
                                    WarpShuffleValueKind::I32 => ("i32", "u32"),
                                    WarpShuffleValueKind::F32 => ("f32", "f32"),
                                    WarpShuffleValueKind::I64 => unreachable!(),
                                };
                                shuffle.adapter
                                    == WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp
                                    && shuffle.lane_encoding
                                        == WarpShuffleOperandEncoding::RegisterOrImmediate
                                    && shuffle.mask_encoding
                                        == WarpShuffleOperandEncoding::RegisterOrImmediate
                                    && record.lowering == "generated_warp_shuffle"
                                    && record.rust.result == rust_ty
                                    && record.dialect.operands[1] == value_ty
                                    && record.llvm.as_ref().is_some_and(|llvm| {
                                        matches!(llvm.arguments.as_slice(), [mask, value, lane, clamp]
                                            if mask == "i32"
                                                && value == value_ty
                                                && lane == "i32"
                                                && clamp == "i32")
                                            && llvm.results == [value_ty]
                                    })
                            }
                            WarpShuffleValueKind::I64 => {
                                shuffle.adapter
                                    == WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble
                                    && shuffle.lane_encoding
                                        == WarpShuffleOperandEncoding::RegisterOnly
                                    && shuffle.mask_encoding
                                        == WarpShuffleOperandEncoding::RegisterOnly
                                    && record.rust.result == "u64"
                                    && record.dialect.operands[1] == "i64"
                                    && record.lowering
                                        == "generated_warp_shuffle_i64_inline_ptx"
                                    && record.llvm.is_none()
                                    && matches!(record.source, IntrinsicSource::PtxNative { .. })
                            }
                        }
                    }),
                "{} is outside the closed generated shfl.sync recipe",
                record.id
            ),
            family => ensure!(false, "{} has unrenderable family {family}", record.id),
        };
    }
    Ok(())
}

fn rust_header(catalog: &CatalogFile, catalog_sha256: &str) -> String {
    format!(
        "/*\n * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n * SPDX-License-Identifier: Apache-2.0\n */\n\n// @generated by cuda-intrinsics-gen {}. DO NOT EDIT.\n// Catalog schema/version: {}/{}; intrinsic ABI: v{}; catalog SHA-256: {}.\n// LLVM source: {}.\n\n",
        catalog.generator_version,
        catalog.schema,
        catalog.catalog_version,
        catalog.intrinsic_abi,
        catalog_sha256,
        catalog.source.llvm_revision
    )
}

fn llvm_header(catalog: &CatalogFile, catalog_sha256: &str) -> String {
    format!(
        "; SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n; SPDX-License-Identifier: Apache-2.0\n; @generated by cuda-intrinsics-gen {}. DO NOT EDIT.\n; Catalog schema/version: {}/{}; intrinsic ABI: v{}; catalog SHA-256: {}.\n; LLVM source: {}.\n\n",
        catalog.generator_version,
        catalog.schema,
        catalog.catalog_version,
        catalog.intrinsic_abi,
        catalog_sha256,
        catalog.source.llvm_revision
    )
}

fn markdown_header(catalog: &CatalogFile, catalog_sha256: &str) -> String {
    format!(
        "<!--\nSPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\nSPDX-License-Identifier: Apache-2.0\n@generated by cuda-intrinsics-gen {}. DO NOT EDIT.\nCatalog schema/version: {}/{}; intrinsic ABI: v{}; catalog SHA-256: {}.\nLLVM source: {}.\n-->\n\n",
        catalog.generator_version,
        catalog.schema,
        catalog.catalog_version,
        catalog.intrinsic_abi,
        catalog_sha256,
        catalog.source.llvm_revision
    )
}

fn sregs(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "sreg")
}

fn ldmatrix(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "ldmatrix")
}

fn packed_atomics(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "packed_atomic")
}

fn redux(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "redux")
}

fn dot_products(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "dotprod")
}

fn packed_alus(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "packed_alu")
}

fn packed_conversions(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "packed_conversion")
}

fn cp_async_copies(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "cp_async_copy")
}

fn cp_async_controls(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "cp_async_control")
}

fn sync_intrinsics(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "sync")
}

fn vote_intrinsics(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "vote")
}

fn active_masks(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "active_mask")
}

fn warp_matches(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "warp_match")
}

fn warp_barriers(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "warp_barrier")
}

fn warp_shuffles(catalog: &CatalogFile) -> impl Iterator<Item = &CatalogIntrinsic> {
    catalog
        .intrinsics
        .iter()
        .filter(|record| record.family == "warp_shuffle")
}

fn dot_product_ptx(record: &CatalogIntrinsic) -> &'static str {
    let dot = record.dot_product.as_ref().expect("dot-product record");
    match (dot.operation, dot.signedness, dot.adapter) {
        (
            DotProductOperation::Dp4a,
            DotProductSignedness::Signed,
            DotProductAdapter::DirectThreeOperands,
        ) => "dp4a.s32.s32 $0, $1, $2, $3;",
        (
            DotProductOperation::Dp4a,
            DotProductSignedness::Unsigned,
            DotProductAdapter::DirectThreeOperands,
        ) => "dp4a.u32.u32 $0, $1, $2, $3;",
        (
            DotProductOperation::Dp2a,
            DotProductSignedness::Signed,
            DotProductAdapter::InsertLowHalfFalse,
        ) => "dp2a.lo.s32.s32 $0, $1, $2, $3;",
        (
            DotProductOperation::Dp2a,
            DotProductSignedness::Unsigned,
            DotProductAdapter::InsertLowHalfFalse,
        ) => "dp2a.lo.u32.u32 $0, $1, $2, $3;",
        combination => panic!("unsupported generated dot-product recipe {combination:?}"),
    }
}

fn packed_alu_ptx_mnemonic(record: &CatalogIntrinsic) -> &'static str {
    let packed = record.packed_alu.as_ref().expect("packed-ALU record");
    match (packed.format, packed.operation, packed.adapter) {
        (PackedAluFormat::Bf16x2, PackedAluOperation::Add, PackedAluAdapter::DirectPackedU32) => {
            "add.rn.bf16x2"
        }
        (PackedAluFormat::Bf16x2, PackedAluOperation::Sub, PackedAluAdapter::DirectPackedU32) => {
            "sub.rn.bf16x2"
        }
        (PackedAluFormat::Bf16x2, PackedAluOperation::Mul, PackedAluAdapter::DirectPackedU32) => {
            "mul.rn.bf16x2"
        }
        (PackedAluFormat::Bf16x2, PackedAluOperation::Fma, PackedAluAdapter::DirectPackedU32) => {
            "fma.rn.bf16x2"
        }
        (
            PackedAluFormat::Bf16x2,
            PackedAluOperation::FmaRelu,
            PackedAluAdapter::DirectPackedU32,
        ) => "fma.rn.relu.bf16x2",
        (PackedAluFormat::Bf16x2, PackedAluOperation::Min, PackedAluAdapter::DirectPackedU32) => {
            "min.bf16x2"
        }
        (PackedAluFormat::Bf16x2, PackedAluOperation::Max, PackedAluAdapter::DirectPackedU32) => {
            "max.bf16x2"
        }
        (PackedAluFormat::Bf16x2, PackedAluOperation::Neg, PackedAluAdapter::DirectPackedU32) => {
            "neg.bf16x2"
        }
        (PackedAluFormat::Bf16x2, PackedAluOperation::Abs, PackedAluAdapter::DirectPackedU32) => {
            "abs.bf16x2"
        }
        (PackedAluFormat::F16x2, PackedAluOperation::Add, PackedAluAdapter::DirectPackedU32) => {
            "add.rn.f16x2"
        }
        (PackedAluFormat::F16x2, PackedAluOperation::Sub, PackedAluAdapter::DirectPackedU32) => {
            "sub.rn.f16x2"
        }
        (PackedAluFormat::F16x2, PackedAluOperation::Mul, PackedAluAdapter::DirectPackedU32) => {
            "mul.rn.f16x2"
        }
        (PackedAluFormat::F16x2, PackedAluOperation::Fma, PackedAluAdapter::DirectPackedU32) => {
            "fma.rn.f16x2"
        }
        (
            PackedAluFormat::F16x2,
            PackedAluOperation::FmaRelu,
            PackedAluAdapter::DirectPackedU32,
        ) => "fma.rn.relu.f16x2",
        (PackedAluFormat::F16x2, PackedAluOperation::Min, PackedAluAdapter::DirectPackedU32) => {
            "min.f16x2"
        }
        (PackedAluFormat::F16x2, PackedAluOperation::Max, PackedAluAdapter::DirectPackedU32) => {
            "max.f16x2"
        }
        (PackedAluFormat::F16x2, PackedAluOperation::Neg, PackedAluAdapter::DirectPackedU32) => {
            "neg.f16x2"
        }
        (PackedAluFormat::F16x2, PackedAluOperation::Abs, PackedAluAdapter::DirectPackedU32) => {
            "abs.f16x2"
        }
    }
}

fn packed_conversion_destination(record: &CatalogIntrinsic) -> &'static str {
    match record
        .packed_conversion
        .as_ref()
        .expect("packed-conversion record")
        .destination_format
    {
        PackedConversionDestinationFormat::Bf16x2 => "bf16x2",
        PackedConversionDestinationFormat::F16x2 => "f16x2",
    }
}

fn packed_conversion_element(record: &CatalogIntrinsic) -> &'static str {
    match record
        .packed_conversion
        .as_ref()
        .expect("packed-conversion record")
        .destination_format
    {
        PackedConversionDestinationFormat::Bf16x2 => "bf16",
        PackedConversionDestinationFormat::F16x2 => "f16",
    }
}

fn packed_conversion_ptx_mnemonic(record: &CatalogIntrinsic) -> String {
    let conversion = record
        .packed_conversion
        .as_ref()
        .expect("packed-conversion record");
    debug_assert_eq!(
        conversion.source_format,
        PackedConversionSourceFormat::F32x2
    );
    debug_assert_eq!(
        conversion.adapter,
        PackedConversionAdapter::ReverseHighLowOperands
    );
    let rounding = match conversion.rounding {
        PackedConversionRounding::NearestEven => "rn",
        PackedConversionRounding::TowardZero => "rz",
    };
    let saturation = match conversion.saturation {
        PackedConversionSaturation::None => "",
        PackedConversionSaturation::Relu => ".relu",
    };
    format!(
        "cvt.{rounding}{saturation}.{}.f32",
        packed_conversion_destination(record)
    )
}

fn llvm(record: &CatalogIntrinsic) -> &CatalogLlvm {
    record.llvm.as_ref().expect("LLVM-imported intrinsic")
}

fn source_label(record: &CatalogIntrinsic) -> String {
    match &record.source {
        IntrinsicSource::LlvmImported { source_record } => {
            format!("LLVM `{}` from `{source_record}`", llvm(record).symbol)
        }
        IntrinsicSource::PtxNative { instruction } => {
            format!("PTX-native `{instruction}`")
        }
    }
}

fn ldmatrix_attr_variants(
    record: &CatalogIntrinsic,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    let variant = &record.ldmatrix.as_ref().expect("ldmatrix record").variant;
    let shape = match variant.shape {
        LdmatrixShape::M8n8 => "LdmatrixShapeAttr::M8n8",
    };
    let multiplicity = match variant.multiplicity {
        LdmatrixMultiplicity::X1 => "LdmatrixMultiplicityAttr::X1",
        LdmatrixMultiplicity::X2 => "LdmatrixMultiplicityAttr::X2",
        LdmatrixMultiplicity::X4 => "LdmatrixMultiplicityAttr::X4",
    };
    let layout = match variant.layout {
        LdmatrixLayout::Normal => "LdmatrixLayoutAttr::Normal",
        LdmatrixLayout::Transposed => "LdmatrixLayoutAttr::Transposed",
    };
    let element = match variant.element {
        LdmatrixElement::B16 => "LdmatrixElementAttr::B16",
    };
    let state_space = match variant.state_space {
        LdmatrixStateSpace::Shared => "LdmatrixStateSpaceAttr::Shared",
    };
    (shape, multiplicity, layout, element, state_space)
}

fn intrinsic_marker(catalog: &CatalogFile, record: &CatalogIntrinsic) -> String {
    format!("v{}:{}", catalog.intrinsic_abi, record.rust.abi_id)
}

fn hardware_target_label(target: &CatalogHardwareTarget) -> String {
    match target {
        CatalogHardwareTarget::All => "all".to_owned(),
        CatalogHardwareTarget::AnyOf { alternatives } => alternatives
            .iter()
            .map(|alternative| match alternative {
                CatalogHardwareAlternative::MinimumSm { sm } => format!("sm_{sm}+"),
                CatalogHardwareAlternative::ExactArchitecture { sm } => format!("sm_{sm}a"),
                CatalogHardwareAlternative::FamilyTarget { sm } => format!("sm_{sm}f"),
            })
            .collect::<Vec<_>>()
            .join(" or "),
    }
}

fn generated_hardware_target(target: &CatalogHardwareTarget) -> String {
    match target {
        CatalogHardwareTarget::All => "GeneratedHardwareTarget::All".to_owned(),
        CatalogHardwareTarget::AnyOf { alternatives } => {
            let alternatives = alternatives
                .iter()
                .map(|alternative| match alternative {
                    CatalogHardwareAlternative::MinimumSm { sm } => {
                        format!("GeneratedHardwareAlternative::MinimumSm({sm})")
                    }
                    CatalogHardwareAlternative::ExactArchitecture { sm } => {
                        format!("GeneratedHardwareAlternative::ExactArchitecture({sm})")
                    }
                    CatalogHardwareAlternative::FamilyTarget { sm } => {
                        format!("GeneratedHardwareAlternative::FamilyTarget({sm})")
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("GeneratedHardwareTarget::AnyOf(&[{alternatives}])")
        }
    }
}

fn backend_label(backend: IntrinsicBackend) -> &'static str {
    match backend {
        IntrinsicBackend::LlvmNvptx => "llvm_nvptx",
        IntrinsicBackend::LibNvvm => "lib_nvvm",
    }
}

fn lowering_mechanism_label(mechanism: BackendLoweringMechanism) -> &'static str {
    match mechanism {
        BackendLoweringMechanism::TypedNvvm => "typed_nvvm",
        BackendLoweringMechanism::InlinePtx => "inline_ptx",
    }
}

fn evidence_stage_label(stage: EvidenceStageKind) -> &'static str {
    match stage {
        EvidenceStageKind::DeclarationCanonicalization => "declaration canonicalization",
        EvidenceStageKind::BackendCodegen => "backend codegen",
        EvidenceStageKind::DeviceLink => "device link",
        EvidenceStageKind::PtxAssembly => "PTX assembly",
        EvidenceStageKind::Runtime => "runtime",
    }
}

fn render_raw_mod(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    let abi_module = format!("__cuda_oxide_intrinsic_abi_v{}", catalog.intrinsic_abi);
    writeln!(
        output,
        "#[doc(hidden)]\n#[path = \"abi_v{}.rs\"]\npub mod {abi_module};\n",
        catalog.intrinsic_abi
    )
    .unwrap();
    for module in modules(catalog) {
        writeln!(
            output,
            "/// Generated `{module}` intrinsic source API.\npub mod {module} {{"
        )
        .unwrap();
        for record in catalog
            .intrinsics
            .iter()
            .filter(|record| record.rust.module == module)
        {
            writeln!(
                output,
                "    pub use crate::{abi_module}::{} as {};",
                record.rust.abi_id, record.rust.name
            )
            .unwrap();
        }
        output.push_str("}\n\n");
    }
    output.push_str("#[cfg(test)]\nmod tests {\n");
    for record in &catalog.intrinsics {
        let arguments = record.rust.arguments.join(", ");
        writeln!(
            output,
            "    #[test]\n    fn public_{}_reexports_abi_{}() {{",
            record.rust.name, record.rust.abi_id
        )
        .unwrap();
        writeln!(
            output,
            "        let public: {}fn({}) -> {} = super::{}::{};",
            if record.rust.safe { "" } else { "unsafe " },
            arguments,
            record.rust.result,
            record.rust.module,
            record.rust.name
        )
        .unwrap();
        writeln!(
            output,
            "        let canonical: {}fn({}) -> {} = super::{abi_module}::{};",
            if record.rust.safe { "" } else { "unsafe " },
            arguments,
            record.rust.result,
            record.rust.abi_id
        )
        .unwrap();
        output.push_str("        assert_eq!(public as usize, canonical as usize);\n    }\n");
    }
    output.push_str("}\n");
    output
}

fn render_raw_abi(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str("//! Raw ABI functions recognized by cuda-oxide.\n\n");
    for record in &catalog.intrinsics {
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(output, "///").unwrap();
        writeln!(
            output,
            "/// Catalog ID: `{}`. Source: {}; expects PTX `{}`.",
            record.id,
            source_label(record),
            record.expected_ptx
        )
        .unwrap();
        writeln!(
            output,
            "/// Available on `{}` targets from PTX {}.",
            hardware_target_label(&record.target.hardware),
            record.target.minimum_ptx
        )
        .unwrap();
        if let Some(reason) = &record.rust.safe_allowlist_reason {
            writeln!(output, "/// Safe because {reason}").unwrap();
        }
        if !record.rust.safe {
            output.push_str("///\n/// # Safety\n");
            if let Some(ldmatrix) = &record.ldmatrix {
                let variant = &ldmatrix.variant;
                let contributing_lanes = variant.multiplicity.register_count() * 8;
                writeln!(
                    output,
                    "/// All 32 warp lanes must execute the same instruction. PTX maps {contributing_lanes} lane-provided row addresses for this {:?} variant; addresses may alias, but each used address must be 16-byte aligned and have 16 readable shared-memory bytes.",
                    variant.multiplicity
                )
                .unwrap();
                if contributing_lanes < 32 {
                    output.push_str("/// For portable sm_75 behavior, otherwise-unused lanes must also carry valid addresses replicated from the contributing lanes.\n");
                }
                output.push_str(
                    "/// This weak memory operation does not replace a required barrier or fence.\n",
                );
            } else if record.redux.is_some() {
                output.push_str(
                    "/// The executing lane must be named in `mask`. Every non-exited lane named in `mask` must execute the same `redux.sync` operation with the same qualifiers and mask.\n\
                     /// The instruction waits for those lanes; violating this participation contract makes the PTX operation undefined.\n",
                );
            } else if record.warp_barrier.is_some() {
                output.push_str(
                    "/// The executing lane must be named in `mask`. Every non-exited lane named in `mask` must execute the same `bar.warp.sync` operation with the same mask.\n\
                     /// On `sm_6x` and earlier, all lanes named in `mask` must execute the barrier in convergence, and no lane outside `mask` may be active when it executes.\n\
                     /// The barrier orders memory accesses among participating lanes; violating the participation contract makes the PTX operation undefined.\n",
                );
            } else if record.warp_shuffle.is_some() {
                output.push_str(
                    "/// The executing lane must be named in `mask`. Every non-exited lane named in `mask` must execute the same `shfl.sync` operation with the same qualifiers and mask.\n\
                     /// On `sm_6x` and earlier, all lanes named in `mask` must execute in convergence, and no lane outside `mask` may be active.\n\
                     /// If the computed source lane is in range, it must be active and named in `mask`; otherwise the result is undefined. If PTX marks the computed source out of range, the calling lane's input is copied.\n",
                );
                if record
                    .warp_shuffle
                    .as_ref()
                    .is_some_and(|shuffle| shuffle.value_kind == WarpShuffleValueKind::I64)
                {
                    output.push_str(
                        "/// The 64-bit value is moved by two `b32` shuffles in one convergent block.\n",
                    );
                }
            } else if record.warp_match.is_some() {
                output.push_str(
                    "/// The executing lane must be named in `mask`. Every non-exited lane named in `mask` must execute the same `match.sync` operation with the same qualifiers and mask.\n\
                     /// Violating this participation contract makes the PTX operation undefined.\n",
                );
            } else if record.vote.is_some() {
                output.push_str(
                    "/// The executing lane must be named in `mask`. Every non-exited lane named in `mask` must execute the same `vote.sync` operation with the same qualifiers and mask.\n\
                     /// On `sm_6x` and earlier, all lanes named in `mask` must execute in convergence, and no lane outside `mask` may be active.\n\
                     /// Violating this participation contract makes the PTX operation undefined.\n",
                );
            } else if record.family == "sync" {
                output.push_str(
                    "/// Every active thread in the CTA must reach the same barrier. Calling it from divergent control flow can deadlock the CTA.\n",
                );
            } else if let Some(copy) = &record.cp_async_copy {
                let bytes = copy.copy_size.bytes();
                writeln!(
                    output,
                    "/// `_arg0` must point to {bytes} writable bytes in shared memory and be aligned to {bytes} bytes."
                )
                .unwrap();
                if copy.source_size == CpAsyncSourceSize::Runtime {
                    writeln!(
                        output,
                        "/// `_arg2` must be at most {bytes}; `_arg1` must point to that many readable bytes in global memory and be aligned to {bytes} bytes."
                    )
                    .unwrap();
                } else {
                    writeln!(
                        output,
                        "/// `_arg1` must point to {bytes} readable bytes in global memory and be aligned to {bytes} bytes."
                    )
                    .unwrap();
                }
                output.push_str(
                    "/// Both ranges must remain valid, the source must remain unchanged, and the destination must not be accessed until this copy completes.\n\
                     /// The issuing thread must use a matching `cp.async` completion operation. Synchronize threads after completion before another thread accesses the destination.\n\
                     /// User-authored completion assembly must include a compiler memory clobber.\n",
                );
            } else if let Some(control) = &record.cp_async_control {
                if control.operation == CpAsyncControlOperation::WaitGroup {
                    output.push_str(
                        "/// `_arg0` must be a compile-time constant. Access only destinations whose copy groups this wait completes.\n",
                    );
                } else if control.operation == CpAsyncControlOperation::WaitAll {
                    output.push_str(
                        "/// This waits only for copies issued by the executing thread. Synchronize threads before another thread accesses a completed destination.\n",
                    );
                } else {
                    output.push_str(
                        "/// This commits only copies issued by the executing thread and does not wait for completion.\n",
                    );
                }
            } else {
                output.push_str(
                    "/// `addr` must designate four writable bytes in global memory and be naturally aligned to four bytes.\n\
                     /// Do not overlap this operation with a whole-word atomic or any non-atomic access to either 16-bit lane.\n\
                     /// Racing atomics must use scopes that include each other; this relaxed GPU-scope operation is not atomic with host/system access.\n",
                );
            }
        }
        if record.rust.must_use {
            output.push_str("#[must_use]\n");
        }
        output.push_str("#[inline(never)]\n");
        let safety = if record.rust.safe { "" } else { "unsafe " };
        let arguments = record
            .rust
            .arguments
            .iter()
            .enumerate()
            .map(|(index, ty)| format!("_arg{index}: {ty}"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            output,
            "pub {safety}fn {}({arguments}) -> {} {{",
            record.rust.abi_id, record.rust.result,
        )
        .unwrap();
        writeln!(
            output,
            "    unreachable!(\"generated CUDA intrinsic `{}` executed outside device compilation\")",
            record.rust.canonical_path
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

fn render_compat_sreg(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "// This file is included lexically inside `cuda_device::thread` so existing\n// DefPaths remain stable during the generated-intrinsics migration.\n\n",
    );
    for record in sregs(catalog) {
        let Some(path) = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with("cuda_device::thread::"))
        else {
            continue;
        };
        let name = path.rsplit("::").next().unwrap();
        writeln!(
            output,
            "/// Compatibility spelling for `{}`.",
            record.rust.public_path
        )
        .unwrap();
        writeln!(
            output,
            "/// The compiler replaces this call with `{}`.",
            llvm(record).symbol
        )
        .unwrap();
        output.push_str("#[allow(non_snake_case)]\n#[inline(never)]\n");
        let safety = if record.rust.safe { "" } else { "unsafe " };
        writeln!(
            output,
            "pub {safety}fn {name}() -> {} {{",
            record.rust.result
        )
        .unwrap();
        writeln!(
            output,
            "    unreachable!(\"generated CUDA intrinsic `{path}` executed outside device compilation\")"
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

fn render_compat_dotprod(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::dotprod` to keep existing paths stable.\n\n");
    for record in dot_products(catalog) {
        let path = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with("cuda_device::dotprod::"))
            .expect("dot-product compatibility path");
        let arguments = record
            .rust
            .arguments
            .iter()
            .enumerate()
            .map(|(index, ty)| format!("arg{index}: {ty}"))
            .collect::<Vec<_>>()
            .join(", ");
        let values = (0..record.rust.arguments.len())
            .map(|index| format!("arg{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str("#[inline(never)]\n");
        writeln!(
            output,
            "pub fn {}({arguments}) -> {} {{",
            record.rust.name, record.rust.result
        )
        .unwrap();
        if record.rust.arguments.len() == 1 {
            writeln!(output, "    let _ = {values};").unwrap();
        } else {
            writeln!(output, "    let _ = ({values});").unwrap();
        }
        writeln!(
            output,
            "    unreachable!(\"generated CUDA intrinsic `{path}` executed outside device compilation\")"
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

fn render_compat_packed_atomic(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str("// Included inside `cuda_device::atomic` to keep existing paths stable.\n\n");
    for record in packed_atomics(catalog) {
        let path = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with("cuda_device::atomic::"))
            .expect("packed-atomic compatibility path");
        assert_eq!(path, &format!("cuda_device::atomic::{}", record.rust.name));
        let packed = record
            .packed_atomic
            .as_ref()
            .expect("packed-atomic semantics");
        let lane_type = match packed.format {
            PackedAtomicFormat::F16x2 => "f16",
            PackedAtomicFormat::Bf16x2 => "bf16",
        };
        let minimum_sm = match &record.target.hardware {
            CatalogHardwareTarget::AnyOf { alternatives } => match alternatives.as_slice() {
                [CatalogHardwareAlternative::MinimumSm { sm }] => *sm,
                _ => panic!("packed-atomic compatibility API requires one minimum SM"),
            },
            _ => panic!("packed-atomic compatibility API requires one minimum SM"),
        };
        assert!(!record.rust.safe);
        assert!(record.rust.must_use);
        assert_eq!(record.rust.arguments, ["*mut u32", "u32"]);
        assert_eq!(record.rust.result, "u32");
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "/// `val` and the result pack two {lane_type} lanes into `u32`, low lane first."
        )
        .unwrap();
        output.push_str(
            "/// The lanes are atomic independently and may not form one old 32-bit snapshot.\n",
        );
        output.push_str("/// This is a relaxed GPU-scope operation. Each lane rounds to nearest-even and preserves subnormals.\n");
        writeln!(
            output,
            "/// Requires PTX {} and `sm_{minimum_sm}+`.",
            record.target.minimum_ptx
        )
        .unwrap();
        output.push_str("///\n/// # Safety\n");
        output.push_str(
            "/// `addr` must point to four writable, four-byte-aligned bytes in global memory.\n",
        );
        output.push_str("/// Do not overlap this operation with a whole-word atomic or non-atomic lane access.\n");
        output.push_str("/// Racing atomics must use mutually inclusive scopes; host/system access is not included.\n");
        output.push_str("#[must_use]\n#[inline(never)]\n");
        writeln!(
            output,
            "pub unsafe fn {}(addr: *mut u32, val: u32) -> u32 {{",
            record.rust.name
        )
        .unwrap();
        output.push_str("    let _ = (addr, val);\n");
        writeln!(
            output,
            "    unreachable!(\"{} called outside CUDA kernel context\")",
            record.rust.name
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

fn render_compat_cp_async_copy(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "// Included inside `cuda_device::async_copy` to keep existing paths stable.\n\n",
    );
    for record in cp_async_copies(catalog) {
        let copy = record.cp_async_copy.as_ref().expect("cp.async semantics");
        let bytes = copy.copy_size.bytes();
        let cache = match copy.cache_policy {
            CpAsyncCachePolicy::Ca => "cache-all",
            CpAsyncCachePolicy::Cg => "cache-global",
        };
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "/// Uses the {cache} policy and copies {bytes} bytes."
        )
        .unwrap();
        if copy.source_size == CpAsyncSourceSize::Runtime {
            writeln!(
                output,
                "/// Bytes after `src_size` are filled with zero; `src_size` must be at most {bytes}."
            )
            .unwrap();
        }
        output.push_str("///\n/// # Safety\n");
        writeln!(
            output,
            "/// `shared_dst` must point to {bytes} writable bytes in shared memory and be aligned to {bytes} bytes."
        )
        .unwrap();
        if copy.source_size == CpAsyncSourceSize::Runtime {
            writeln!(
                output,
                "/// `global_src` must point to `src_size` readable bytes in global memory and be aligned to {bytes} bytes."
            )
            .unwrap();
        } else {
            writeln!(
                output,
                "/// `global_src` must point to {bytes} readable bytes in global memory and be aligned to {bytes} bytes."
            )
            .unwrap();
        }
        output.push_str(
            "/// Both ranges must remain valid, the source must remain unchanged, and the destination must not be accessed until this copy completes.\n\
             /// The issuing thread must complete the copy. Synchronize threads afterward before another thread accesses the destination.\n\
             /// User-authored completion assembly must include a compiler memory clobber.\n",
        );
        output.push_str("#[inline(never)]\n");
        if copy.source_size == CpAsyncSourceSize::Runtime {
            writeln!(
                output,
                "pub unsafe fn {}(_shared_dst: *mut u32, _global_src: *const u8, _src_size: u32) {{",
                record.rust.name
            )
            .unwrap();
        } else {
            writeln!(
                output,
                "pub unsafe fn {}(_shared_dst: *mut u32, _global_src: *const u32) {{",
                record.rust.name
            )
            .unwrap();
        }
        writeln!(
            output,
            "    unreachable!(\"{} called outside CUDA kernel context\")",
            record.rust.name
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    for record in cp_async_controls(catalog) {
        let control = record
            .cp_async_control
            .as_ref()
            .expect("cp.async control semantics");
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str("///\n/// # Safety\n");
        match control.operation {
            CpAsyncControlOperation::CommitGroup => output.push_str(
                "/// This commits only copies issued by the executing thread and does not wait for completion.\n",
            ),
            CpAsyncControlOperation::WaitAll => output.push_str(
                "/// This waits only for copies issued by this thread. Synchronize threads before another thread accesses a completed destination.\n",
            ),
            CpAsyncControlOperation::WaitGroup => output.push_str(
                "/// `max_pending` must be a compile-time constant. Access only destinations whose copy groups this wait completes.\n",
            ),
        }
        output.push_str("#[inline(never)]\n");
        if control.operation == CpAsyncControlOperation::WaitGroup {
            writeln!(
                output,
                "pub unsafe fn {}(_max_pending: u32) {{",
                record.rust.name
            )
            .unwrap();
        } else {
            writeln!(output, "pub unsafe fn {}() {{", record.rust.name).unwrap();
        }
        writeln!(
            output,
            "    unreachable!(\"{} called outside CUDA kernel context\")",
            record.rust.name
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

fn render_compat_packed_alu(catalog: &CatalogFile, hash: &str, format: PackedAluFormat) -> String {
    let mut output = rust_header(catalog, hash);
    let module = match format {
        PackedAluFormat::Bf16x2 => "bf16x2",
        PackedAluFormat::F16x2 => "f16x2",
    };
    writeln!(
        output,
        "// Included inside `cuda_device::{module}` to keep existing paths stable.\n"
    )
    .unwrap();
    for record in packed_alus(catalog).filter(|record| {
        record
            .packed_alu
            .as_ref()
            .is_some_and(|packed| packed.format == format)
    }) {
        let path = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with(&format!("cuda_device::{module}::")))
            .expect("packed-ALU compatibility path");
        let arguments = record
            .rust
            .arguments
            .iter()
            .enumerate()
            .map(|(index, ty)| format!("arg{index}: {ty}"))
            .collect::<Vec<_>>()
            .join(", ");
        let values = (0..record.rust.arguments.len())
            .map(|index| format!("arg{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(output, "/// {}", record.summary).unwrap();
        if record.rust.must_use {
            output.push_str("#[must_use]\n");
        }
        output.push_str("#[inline(never)]\n");
        writeln!(output, "pub fn {}({arguments}) -> u32 {{", record.rust.name).unwrap();
        if record.rust.arguments.len() == 1 {
            writeln!(output, "    let _ = {values};").unwrap();
        } else {
            writeln!(output, "    let _ = ({values});").unwrap();
        }
        writeln!(
            output,
            "    unreachable!(\"generated CUDA intrinsic `{path}` executed outside device compilation\")"
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

fn render_compat_packed_conversion(
    catalog: &CatalogFile,
    hash: &str,
    path_prefix: &str,
    containing_module: &str,
    argument_names: (&str, &str),
) -> String {
    let mut output = rust_header(catalog, hash);
    writeln!(
        output,
        "// Included inside `cuda_device::{containing_module}` to keep the existing path stable.\n"
    )
    .unwrap();
    for record in packed_conversions(catalog).filter(|record| {
        record
            .rust
            .compatibility_paths
            .iter()
            .any(|path| path.starts_with(path_prefix))
    }) {
        let path = record
            .rust
            .compatibility_paths
            .iter()
            .find(|path| path.starts_with(path_prefix))
            .expect("packed-conversion compatibility path");
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str("#[inline(never)]\n");
        writeln!(
            output,
            "pub fn {}({}: f32, {}: f32) -> u32 {{",
            record.rust.name, argument_names.0, argument_names.1
        )
        .unwrap();
        writeln!(
            output,
            "    let _ = ({}, {});",
            argument_names.0, argument_names.1
        )
        .unwrap();
        writeln!(
            output,
            "    unreachable!(\"generated CUDA intrinsic `{path}` executed outside device compilation\")"
        )
        .unwrap();
        output.push_str("}\n\n");
    }
    output
}

fn render_dialect_mod(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "mod active_mask;\nmod cp_async;\nmod dotprod;\nmod ldmatrix;\nmod packed_alu;\nmod packed_atomic;\nmod packed_conversion;\nmod redux;\nmod sreg;\nmod sync;\nmod vote;\nmod warp_barrier;\nmod warp_match;\nmod warp_shuffle;\n\npub use active_mask::*;\npub use cp_async::*;\npub use dotprod::*;\npub use ldmatrix::*;\npub use packed_alu::*;\npub use packed_atomic::*;\npub use packed_conversion::*;\npub use redux::*;\npub use sreg::*;\npub use sync::*;\npub use vote::*;\npub use warp_barrier::*;\npub use warp_match::*;\npub use warp_shuffle::*;\n\nuse pliron::context::Context;\n\npub(super) fn register(ctx: &mut Context) {\n    active_mask::register(ctx);\n    cp_async::register(ctx);\n    dotprod::register(ctx);\n    ldmatrix::register(ctx);\n    packed_alu::register(ctx);\n    packed_atomic::register(ctx);\n    packed_conversion::register(ctx);\n    redux::register(ctx);\n    sreg::register(ctx);\n    sync::register(ctx);\n    vote::register(ctx);\n    warp_barrier::register(ctx);\n    warp_match::register(ctx);\n    warp_shuffle::register(ctx);\n}\n",
    );
    output
}

fn render_dialect_sreg(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural NVVM operations for generated special-register reads.\n\nuse pliron::{\n    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},\n    builtin::types::IntegerType,\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\n",
    );
    for record in sregs(catalog) {
        let width = record.scalar_width().unwrap();
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "/// Catalog ID `{}`; `{}` returns one `i{width}` result.",
            record.id,
            llvm(record).symbol
        )
        .unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        output.push_str("    pub fn new(op: Ptr<Operation>) -> Self {\n");
        writeln!(output, "        Self {{ op }}").unwrap();
        output.push_str("    }\n}\n");
        writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
        output.push_str("    fn verify(&self, ctx: &Context) -> Result<(), Error> {\n");
        writeln!(
            output,
            "        verify_scalar_result(ctx, self.get_operation(), {:?}, {width})",
            record.dialect.op_name
        )
        .unwrap();
        output.push_str("    }\n}\n\n");
    }
    output.push_str(
        "fn verify_scalar_result(\n    ctx: &Context,\n    op: Ptr<Operation>,\n    name: &str,\n    width: u32,\n) -> Result<(), Error> {\n    let op = op.deref(ctx);\n    let ty = op.get_result(0).get_type(ctx);\n    let ty_object = ty.deref(ctx);\n    let Some(integer) = ty_object.downcast_ref::<IntegerType>() else {\n        return verify_err!(op.loc(), \"{} result must be an integer\", name);\n    };\n    if integer.width() != width {\n        return verify_err!(\n            op.loc(),\n            \"{} result must be a {}-bit integer\",\n            name,\n            width\n        );\n    }\n    Ok(())\n}\n\npub(super) fn register(ctx: &mut Context) {\n",
    );
    for record in sregs(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

fn render_dialect_sync(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural operation for generated CTA synchronization.\n\nuse pliron::{\n    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},\n    context::{Context, Ptr},\n    op::Op,\n    operation::Operation,\n};\nuse pliron_derive::pliron_op;\n\n",
    );
    for record in sync_intrinsics(catalog) {
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    verifier = \"succ\",\n    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        output.push_str(
            "    pub fn new(op: Ptr<Operation>) -> Self {\n        Self { op }\n    }\n}\n\n",
        );
    }
    output.push_str("pub(super) fn register(ctx: &mut Context) {\n");
    for record in sync_intrinsics(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

fn render_dialect_vote(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural operations for the generated `vote.sync` family.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::{IntegerType, Signedness},\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    value::Value,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\nfn is_integer_width(ctx: &Context, ty: pliron::r#type::TypeHandle, width: u32) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == width)\n}\n\n",
    );
    for record in vote_intrinsics(catalog) {
        let result_width = match record.vote.as_ref().unwrap().mode {
            VoteMode::All | VoteMode::Any | VoteMode::Uni => 1,
            VoteMode::Ballot => 32,
        };
        let result_signedness = if result_width == 32 {
            "Unsigned"
        } else {
            "Signless"
        };
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str(
            "///\n/// Operands are `[member_mask, predicate]`. The generated verifier keeps\n/// the mask, predicate, and result types exact.\n",
        );
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    pub fn new(op: Ptr<Operation>) -> Self {{\n        Self {{ op }}\n    }}\n\n    pub fn build(ctx: &mut Context, member_mask: Value, predicate: Value) -> Ptr<Operation> {{\n        let result_ty = IntegerType::get(ctx, {result_width}, Signedness::{result_signedness});\n        Operation::new(\n            ctx,\n            Self::get_concrete_op_info(),\n            vec![result_ty.into()],\n            vec![member_mask, predicate],\n            vec![],\n            0,\n        )\n    }}\n}}"
        )
        .unwrap();
        writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        let op = self.get_operation().deref(ctx);\n        if op.get_num_operands() != 2 || op.get_num_results() != 1 {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        if !is_integer_width(ctx, op.get_operand(0).get_type(ctx), 32)\n            || !is_integer_width(ctx, op.get_operand(1).get_type(ctx), 1)\n            || !is_integer_width(ctx, op.get_result(0).get_type(ctx), {result_width})\n        {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        Ok(())\n    }}\n}}\n",
            format!(
                "{} requires exactly two operands [member_mask, predicate] and one result",
                record.dialect.op_name
            ),
            format!(
                "{} requires i32 member mask, i1 predicate, and i{result_width} result",
                record.dialect.op_name
            ),
        )
        .unwrap();
    }
    output.push_str("\npub(super) fn register(ctx: &mut Context) {\n");
    for record in vote_intrinsics(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

fn render_dialect_active_mask(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural operation for the generated active warp mask.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::{IntegerType, Signedness},\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\n",
    );
    for record in active_masks(catalog) {
        debug_assert_eq!(
            record.active_mask.as_ref().unwrap().adapter,
            ActiveMaskAdapter::DirectZeroOperandMask
        );
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        output.push_str(
            "    pub fn new(op: Ptr<Operation>) -> Self {\n        Self { op }\n    }\n\n    pub fn build(ctx: &mut Context) -> Ptr<Operation> {\n        let result_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);\n        Operation::new(\n            ctx,\n            Self::get_concrete_op_info(),\n            vec![result_ty.into()],\n            vec![],\n            vec![],\n            0,\n        )\n    }\n}\n",
        );
        writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        let op = self.get_operation().deref(ctx);\n        let valid = op.get_num_operands() == 0\n            && op.get_num_results() == 1\n            && op.get_result(0).get_type(ctx).deref(ctx)\n                .downcast_ref::<IntegerType>()\n                .is_some_and(|integer| integer.width() == 32);\n        if !valid {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        Ok(())\n    }}\n}}\n",
            format!("{} requires no operands and one i32 result", record.dialect.op_name)
        )
        .unwrap();
    }
    output.push_str("\npub(super) fn register(ctx: &mut Context) {\n");
    for record in active_masks(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

fn render_dialect_cp_async_copy(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        r#"//! Structural operations for classic global-to-shared `cp.async` copies.

use dialect_mir::{ops::MirConstantOp, types::{MirPtrType, address_space}};
use pliron::{
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface},
        ops::ConstantOp,
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    value::Value,
    verify_err,
};
use pliron_derive::pliron_op;

fn verify_pointer(
    ctx: &Context,
    op: &Operation,
    value: Value,
    role: &str,
    allowed_address_spaces: &[u32],
) -> Result<(), Error> {
    let ty = value.get_type(ctx);
    let ty = ty.deref(ctx);
    let Some(pointer) = ty.downcast_ref::<MirPtrType>() else {
        return verify_err!(op.loc(), "{role} must be a MIR pointer");
    };
    if !allowed_address_spaces.contains(&pointer.address_space) {
        return verify_err!(op.loc(), "{role} has the wrong address space");
    }
    Ok(())
}

fn verify_cp_async_copy(
    ctx: &Context,
    operation: Ptr<Operation>,
    name: &str,
    has_source_size: bool,
) -> Result<(), Error> {
    let op = operation.deref(ctx);
    let expected_operands = if has_source_size { 3 } else { 2 };
    if op.get_num_operands() != expected_operands || op.get_num_results() != 0 {
        return verify_err!(op.loc(), "{name} has the wrong operand or result count");
    }
    verify_pointer(
        ctx,
        &op,
        op.get_operand(0),
        "shared destination",
        &[address_space::GENERIC, address_space::SHARED],
    )?;
    verify_pointer(
        ctx,
        &op,
        op.get_operand(1),
        "global source",
        &[address_space::GENERIC, address_space::GLOBAL],
    )?;
    if has_source_size {
        let ty = op.get_operand(2).get_type(ctx);
        let ty = ty.deref(ctx);
        let Some(integer) = ty.downcast_ref::<IntegerType>() else {
            return verify_err!(op.loc(), "source size must be u32");
        };
        if integer.width() != 32 || integer.signedness() != Signedness::Unsigned {
            return verify_err!(op.loc(), "source size must be u32");
        }
    }
    Ok(())
}

fn verify_cp_async_control(
    ctx: &Context,
    operation: Ptr<Operation>,
    name: &str,
    has_immediate: bool,
) -> Result<(), Error> {
    let op = operation.deref(ctx);
    let expected_operands = usize::from(has_immediate);
    if op.get_num_operands() != expected_operands || op.get_num_results() != 0 {
        return verify_err!(op.loc(), "{name} has the wrong operand or result count");
    }
    if has_immediate {
        let value = op.get_operand(0);
        let ty = value.get_type(ctx);
        let ty = ty.deref(ctx);
        let Some(integer) = ty.downcast_ref::<IntegerType>() else {
            return verify_err!(op.loc(), "maximum pending group count must be u32");
        };
        if integer.width() != 32 || integer.signedness() != Signedness::Unsigned {
            return verify_err!(op.loc(), "maximum pending group count must be u32");
        }
        let Some(defining_op) = value.defining_op() else {
            return verify_err!(op.loc(), "maximum pending group count must be a compile-time constant");
        };
        if Operation::get_op::<MirConstantOp>(defining_op, ctx).is_none()
            && Operation::get_op::<ConstantOp>(defining_op, ctx).is_none()
        {
            return verify_err!(op.loc(), "maximum pending group count must be a compile-time constant");
        }
    }
    Ok(())
}

"#,
    );
    for record in cp_async_copies(catalog) {
        let copy = record.cp_async_copy.as_ref().unwrap();
        let dynamic = copy.source_size == CpAsyncSourceSize::Runtime;
        let operand_count = if dynamic { 3 } else { 2 };
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(output, "/// Lowers to `{}`.", record.expected_ptx).unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<{operand_count}>, NResultsInterface<0>],\n)]\npub struct {};",
            record.dialect.op_name, record.dialect.op_type
        )
        .unwrap();
        writeln!(output, "impl {} {{", record.dialect.op_type).unwrap();
        output.push_str("    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }\n\n");
        if dynamic {
            output.push_str(
                "    pub fn build(ctx: &mut Context, shared_dst: Value, global_src: Value, source_size: Value) -> Ptr<Operation> {\n        Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![shared_dst, global_src, source_size], vec![], 0)\n    }\n",
            );
        } else {
            output.push_str(
                "    pub fn build(ctx: &mut Context, shared_dst: Value, global_src: Value) -> Ptr<Operation> {\n        Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![shared_dst, global_src], vec![], 0)\n    }\n",
            );
        }
        output.push_str("}\n\n");
        writeln!(output, "impl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        verify_cp_async_copy(ctx, self.get_operation(), {:?}, {dynamic})\n    }}\n}}\n",
            record.dialect.op_name
        )
        .unwrap();
    }
    for record in cp_async_controls(catalog) {
        let control = record.cp_async_control.as_ref().unwrap();
        let has_immediate = control.operation == CpAsyncControlOperation::WaitGroup;
        let operand_count = usize::from(has_immediate);
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(output, "/// Lowers to `{}`.", record.expected_ptx).unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<{operand_count}>, NResultsInterface<0>],\n)]\npub struct {};",
            record.dialect.op_name, record.dialect.op_type
        )
        .unwrap();
        writeln!(output, "impl {} {{", record.dialect.op_type).unwrap();
        output.push_str("    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }\n\n");
        if has_immediate {
            output.push_str(
                "    pub fn build(ctx: &mut Context, max_pending: Value) -> Ptr<Operation> {\n        Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![max_pending], vec![], 0)\n    }\n",
            );
        } else {
            output.push_str(
                "    pub fn build(ctx: &mut Context) -> Ptr<Operation> {\n        Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0)\n    }\n",
            );
        }
        output.push_str("}\n\n");
        writeln!(output, "impl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        verify_cp_async_control(ctx, self.get_operation(), {:?}, {has_immediate})\n    }}\n}}\n",
            record.dialect.op_name
        )
        .unwrap();
    }
    output.push_str("pub(super) fn register(ctx: &mut Context) {\n");
    for record in cp_async_copies(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    for record in cp_async_controls(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

fn render_dialect_warp_match(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural operations for the generated `match.sync` family.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::{IntegerType, Signedness},\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    value::Value,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\nfn is_integer_width(ctx: &Context, ty: pliron::r#type::TypeHandle, width: u32) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == width)\n}\n\n",
    );
    for record in warp_matches(catalog) {
        let warp_match = record.warp_match.as_ref().unwrap();
        let value_width = warp_match.value_width.bits();
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str(
            "///\n/// Operands are `[member_mask, value]`; the result is a 32-bit lane mask.\n",
        );
        if warp_match.mode == WarpMatchMode::All {
            output.push_str(
                "/// LLVM also returns a predicate, which the established API discards.\n",
            );
        }
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        output.push_str(
            "    pub fn new(op: Ptr<Operation>) -> Self {\n        Self { op }\n    }\n\n    pub fn build(ctx: &mut Context, member_mask: Value, value: Value) -> Ptr<Operation> {\n        let result_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);\n        Operation::new(\n            ctx,\n            Self::get_concrete_op_info(),\n            vec![result_ty.into()],\n            vec![member_mask, value],\n            vec![],\n            0,\n        )\n    }\n}\n",
        );
        writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        let op = self.get_operation().deref(ctx);\n        if op.get_num_operands() != 2 || op.get_num_results() != 1 {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        if !is_integer_width(ctx, op.get_operand(0).get_type(ctx), 32)\n            || !is_integer_width(ctx, op.get_operand(1).get_type(ctx), {value_width})\n            || !is_integer_width(ctx, op.get_result(0).get_type(ctx), 32)\n        {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        Ok(())\n    }}\n}}\n",
            format!(
                "{} requires exactly [member_mask, value] and one mask result",
                record.dialect.op_name
            ),
            format!(
                "{} requires i32 member mask, i{value_width} value, and i32 result",
                record.dialect.op_name
            ),
        )
        .unwrap();
    }
    output.push_str("\npub(super) fn register(ctx: &mut Context) {\n");
    for record in warp_matches(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

fn render_dialect_warp_barrier(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural operation for generated warp synchronization.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::IntegerType,\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    value::Value,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\nfn is_i32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == 32)\n}\n\n",
    );
    for record in warp_barriers(catalog) {
        debug_assert_eq!(
            record.warp_barrier.as_ref().unwrap().adapter,
            WarpBarrierAdapter::DirectMemberMask
        );
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str("///\n/// The operand is the 32-bit warp participation mask.\n");
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<1>, NResultsInterface<0>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        output.push_str(
            "    pub fn new(op: Ptr<Operation>) -> Self {\n        Self { op }\n    }\n\n    pub fn build(ctx: &mut Context, member_mask: Value) -> Ptr<Operation> {\n        Operation::new(\n            ctx,\n            Self::get_concrete_op_info(),\n            vec![],\n            vec![member_mask],\n            vec![],\n            0,\n        )\n    }\n}\n",
        );
        writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        let op = self.get_operation().deref(ctx);\n        if op.get_num_operands() != 1 || op.get_num_results() != 0 {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        if !is_i32(ctx, op.get_operand(0).get_type(ctx)) {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        Ok(())\n    }}\n}}\n",
            format!(
                "{} requires exactly one member-mask operand and no results",
                record.dialect.op_name
            ),
            format!("{} member mask must be i32", record.dialect.op_name),
        )
        .unwrap();
    }
    output.push_str("\npub(super) fn register(ctx: &mut Context) {\n");
    for record in warp_barriers(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

fn render_dialect_warp_shuffle(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural operations for the generated `shfl.sync` family.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::{FP32Type, IntegerType, Signedness},\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    value::Value,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\nfn is_i32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == 32)\n}\n\nfn is_i64(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == 64)\n}\n\nfn is_f32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {\n    ty.deref(ctx).downcast_ref::<FP32Type>().is_some()\n}\n\n",
    );
    for record in warp_shuffles(catalog) {
        let shuffle = record.warp_shuffle.as_ref().unwrap();
        debug_assert!(matches!(
            (shuffle.value_kind, shuffle.adapter),
            (
                WarpShuffleValueKind::I32 | WarpShuffleValueKind::F32,
                WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp
            ) | (
                WarpShuffleValueKind::I64,
                WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble
            )
        ));
        let (result_builder, value_check, value_label) = match shuffle.value_kind {
            WarpShuffleValueKind::I32 => (
                "IntegerType::get(ctx, 32, Signedness::Unsigned).into()",
                "is_i32",
                "i32",
            ),
            WarpShuffleValueKind::F32 => ("FP32Type::get(ctx).into()", "is_f32", "f32"),
            WarpShuffleValueKind::I64 => (
                "IntegerType::get(ctx, 64, Signedness::Unsigned).into()",
                "is_i64",
                "i64",
            ),
        };
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str("///\n/// Operands are `[member_mask, value, lane_or_delta]`.\n");
        if shuffle.value_kind == WarpShuffleValueKind::I64 {
            output.push_str(
                "/// Lowering splits the value into two `b32` halves, shuffles both, and reassembles it.\n",
            );
        } else {
            output.push_str(
                "/// Generated lowering inserts the fixed clamp required by the selected shuffle mode.\n",
            );
        }
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    pub fn new(op: Ptr<Operation>) -> Self {{\n        Self {{ op }}\n    }}\n\n    pub fn build(\n        ctx: &mut Context,\n        member_mask: Value,\n        value: Value,\n        lane_or_delta: Value,\n    ) -> Ptr<Operation> {{\n        let result_ty: pliron::r#type::TypeHandle = {result_builder};\n        Operation::new(\n            ctx,\n            Self::get_concrete_op_info(),\n            vec![result_ty],\n            vec![member_mask, value, lane_or_delta],\n            vec![],\n            0,\n        )\n    }}\n}}"
        )
        .unwrap();
        writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        let op = self.get_operation().deref(ctx);\n        if op.get_num_operands() != 3 || op.get_num_results() != 1 {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        if !is_i32(ctx, op.get_operand(0).get_type(ctx))\n            || !{value_check}(ctx, op.get_operand(1).get_type(ctx))\n            || !is_i32(ctx, op.get_operand(2).get_type(ctx))\n            || !{value_check}(ctx, op.get_result(0).get_type(ctx))\n        {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        Ok(())\n    }}\n}}\n",
            format!(
                "{} requires exactly [member_mask, value, lane_or_delta] and one result",
                record.dialect.op_name
            ),
            format!(
                "{} requires i32 mask/lane and {value_label} value/result",
                record.dialect.op_name
            ),
        )
        .unwrap();
    }
    output.push_str("\npub(super) fn register(ctx: &mut Context) {\n");
    for record in warp_shuffles(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

fn render_dialect_dotprod(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural operations for generated packed integer dot products.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::{IntegerType, Signedness},\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    value::Value,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\nfn is_i32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == 32)\n}\n\n",
    );
    for record in dot_products(catalog) {
        let signedness = match record.rust.result.as_str() {
            "i32" => "Signed",
            "u32" => "Unsigned",
            result => panic!("unsupported dot-product result {result}"),
        };
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "/// Lowers to `{}`.",
            dot_product_ptx(record).replace("$0, $1, $2, $3", "%d, %a, %b, %c")
        )
        .unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    pub fn new(op: Ptr<Operation>) -> Self {{\n        Self {{ op }}\n    }}\n\n    pub fn build(ctx: &mut Context, a: Value, b: Value, c: Value) -> Ptr<Operation> {{\n        let result_ty = IntegerType::get(ctx, 32, Signedness::{signedness});\n        Operation::new(\n            ctx,\n            Self::get_concrete_op_info(),\n            vec![result_ty.into()],\n            vec![a, b, c],\n            vec![],\n            0,\n        )\n    }}\n}}"
        )
        .unwrap();
        writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        let op = self.get_operation().deref(ctx);\n        if op.get_num_operands() != 3 || op.get_num_results() != 1 {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        if !(0..3).all(|index| is_i32(ctx, op.get_operand(index).get_type(ctx)))\n            || !is_i32(ctx, op.get_result(0).get_type(ctx))\n        {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        Ok(())\n    }}\n}}\n",
            format!(
                "{} requires exactly three operands and one result",
                record.dialect.op_name
            ),
            format!(
                "{} operands and result must be 32-bit integers",
                record.dialect.op_name
            ),
        )
        .unwrap();
    }
    output.push_str("\npub(super) fn register(ctx: &mut Context) {\n");
    for record in dot_products(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

fn render_dialect_redux(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural NVVM operations for the closed generated `redux.sync` family.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::{IntegerType, Signedness},\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    value::Value,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\nfn is_i32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == 32)\n}\n\n",
    );
    for record in redux(catalog) {
        writeln!(output, "/// {}", record.summary).unwrap();
        output.push_str(
            "///\n/// Dialect operands are `[member_mask, value]`; generated lowering adapts\n/// them to LLVM's `(value, member_mask)` signature.\n",
        );
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        let result_signedness = match record.rust.result.as_str() {
            "u32" => "Unsigned",
            "i32" => "Signed",
            result => panic!("unsupported redux result {result}"),
        };
        writeln!(
            output,
            "    pub fn new(op: Ptr<Operation>) -> Self {{\n        Self {{ op }}\n    }}\n\n    pub fn build(ctx: &mut Context, member_mask: Value, value: Value) -> Ptr<Operation> {{\n        let result_ty = IntegerType::get(ctx, 32, Signedness::{result_signedness});\n        Operation::new(\n            ctx,\n            Self::get_concrete_op_info(),\n            vec![result_ty.into()],\n            vec![member_mask, value],\n            vec![],\n            0,\n        )\n    }}\n}}"
        )
        .unwrap();
        writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        let op = self.get_operation().deref(ctx);\n        if op.get_num_operands() != 2 || op.get_num_results() != 1 {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        if !is_i32(ctx, op.get_operand(0).get_type(ctx))\n            || !is_i32(ctx, op.get_operand(1).get_type(ctx))\n            || !is_i32(ctx, op.get_result(0).get_type(ctx))\n        {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        Ok(())\n    }}\n}}\n",
            format!(
                "{} requires exactly two operands [member_mask, value] and one result",
                record.dialect.op_name
            ),
            format!(
                "{} member mask, value, and result must be 32-bit integers",
                record.dialect.op_name
            ),
        )
        .unwrap();
    }
    output.push_str("\npub(super) fn register(ctx: &mut Context) {\n");
    for record in redux(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

fn render_dialect_ldmatrix(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        r##"//! One structural operation for the closed generated `ldmatrix` family.

use dialect_mir::types::{address_space, MirPtrType};
use pliron::{
    attribute::Attribute,
    builtin::{
        op_interfaces::NOpdsInterface,
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    value::Value,
    verify_err,
};
use pliron_derive::{pliron_attr, pliron_op};

#[pliron_attr(name = "nvvm.ldmatrix_shape", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum LdmatrixShapeAttr {
    M8n8,
}

#[pliron_attr(name = "nvvm.ldmatrix_multiplicity", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum LdmatrixMultiplicityAttr {
    X1,
    X2,
    X4,
}

impl LdmatrixMultiplicityAttr {
    pub const fn register_count(&self) -> usize {
        match self {
            Self::X1 => 1,
            Self::X2 => 2,
            Self::X4 => 4,
        }
    }
}

#[pliron_attr(name = "nvvm.ldmatrix_layout", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum LdmatrixLayoutAttr {
    Normal,
    Transposed,
}

#[pliron_attr(name = "nvvm.ldmatrix_element", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum LdmatrixElementAttr {
    B16,
}

#[pliron_attr(name = "nvvm.ldmatrix_state_space", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum LdmatrixStateSpaceAttr {
    Shared,
}

/// A warp-cooperative matrix load whose exact variant is carried by attributes.
#[pliron_op(
    name = "nvvm.ldmatrix",
    format,
    interfaces = [NOpdsInterface<1>],
    attributes = (
        nvvm_ldmatrix_shape: LdmatrixShapeAttr,
        nvvm_ldmatrix_multiplicity: LdmatrixMultiplicityAttr,
        nvvm_ldmatrix_layout: LdmatrixLayoutAttr,
        nvvm_ldmatrix_element: LdmatrixElementAttr,
        nvvm_ldmatrix_state_space: LdmatrixStateSpaceAttr
    )
)]
pub struct LdmatrixOp;

impl LdmatrixOp {
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }

    pub fn build(
        ctx: &mut Context,
        address: Value,
        shape: LdmatrixShapeAttr,
        multiplicity: LdmatrixMultiplicityAttr,
        layout: LdmatrixLayoutAttr,
        element: LdmatrixElementAttr,
        state_space: LdmatrixStateSpaceAttr,
    ) -> Ptr<Operation> {
        let register_count = multiplicity.register_count();
        let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![u32_ty.into(); register_count],
            vec![address],
            vec![],
            0,
        );
        let this = Self { op };
        this.set_attr_nvvm_ldmatrix_shape(ctx, shape);
        this.set_attr_nvvm_ldmatrix_multiplicity(ctx, multiplicity);
        this.set_attr_nvvm_ldmatrix_layout(ctx, layout);
        this.set_attr_nvvm_ldmatrix_element(ctx, element);
        this.set_attr_nvvm_ldmatrix_state_space(ctx, state_space);
        this.get_operation()
    }
}

impl Verify for LdmatrixOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        let operands: Vec<_> = op.operands().collect();
        if operands.len() != 1 {
            return verify_err!(op.loc(), "nvvm.ldmatrix requires exactly one pointer operand");
        }
        let pointer = operands[0].get_type(ctx);
        let pointer_object = pointer.deref(ctx);
        let Some(pointer) = pointer_object.downcast_ref::<MirPtrType>() else {
            return verify_err!(op.loc(), "nvvm.ldmatrix operand must be a MIR pointer");
        };
        if !matches!(pointer.address_space, address_space::GENERIC | address_space::SHARED) {
            return verify_err!(
                op.loc(),
                "nvvm.ldmatrix pointer must be generic (p0) or shared (p3), not address space {}",
                pointer.address_space
            );
        }
        let pointee = pointer.pointee.deref(ctx);
        let Some(pointee) = pointee.downcast_ref::<IntegerType>() else {
            return verify_err!(op.loc(), "nvvm.ldmatrix pointer must point to u32");
        };
        if pointee.width() != 32 || pointee.signedness() != Signedness::Unsigned {
            return verify_err!(op.loc(), "nvvm.ldmatrix pointer must point to u32");
        }

        let Some(multiplicity) = self.get_attr_nvvm_ldmatrix_multiplicity(ctx) else {
            return verify_err!(op.loc(), "nvvm.ldmatrix requires a multiplicity attribute");
        };
        if self.get_attr_nvvm_ldmatrix_shape(ctx).as_deref() != Some(&LdmatrixShapeAttr::M8n8)
            || self.get_attr_nvvm_ldmatrix_layout(ctx).is_none()
            || self.get_attr_nvvm_ldmatrix_element(ctx).as_deref() != Some(&LdmatrixElementAttr::B16)
            || self.get_attr_nvvm_ldmatrix_state_space(ctx).as_deref()
                != Some(&LdmatrixStateSpaceAttr::Shared)
        {
            return verify_err!(op.loc(), "nvvm.ldmatrix has a missing or unsupported variant attribute");
        }

        let register_count = multiplicity.register_count();
        if op.get_num_results() != register_count {
            return verify_err!(
                op.loc(),
                "nvvm.ldmatrix {:?} requires {} u32 results",
                multiplicity,
                register_count
            );
        }
        for index in 0..register_count {
            let ty = op.get_result(index).get_type(ctx);
            let ty_object = ty.deref(ctx);
            let Some(integer) = ty_object.downcast_ref::<IntegerType>() else {
                return verify_err!(op.loc(), "nvvm.ldmatrix result {} must be u32", index);
            };
            if integer.width() != 32 || integer.signedness() != Signedness::Unsigned {
                return verify_err!(op.loc(), "nvvm.ldmatrix result {} must be u32", index);
            }
        }
        Ok(())
    }
}

pub(super) fn register(ctx: &mut Context) {
    LdmatrixShapeAttr::register(ctx);
    LdmatrixMultiplicityAttr::register(ctx);
    LdmatrixLayoutAttr::register(ctx);
    LdmatrixElementAttr::register(ctx);
    LdmatrixStateSpaceAttr::register(ctx);
    LdmatrixOp::register(ctx);
}
"##,
    );
    output
}

fn render_dialect_packed_atomic(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        r##"//! One structural operation for the closed generated packed-atomic family.

use dialect_mir::types::{address_space, MirPtrType};
use pliron::{
    attribute::Attribute,
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface},
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    value::Value,
    verify_err,
};
use pliron_derive::{pliron_attr, pliron_op};

#[pliron_attr(name = "nvvm.packed_atomic_format", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum PackedAtomicFormatAttr { F16x2, Bf16x2 }

#[pliron_attr(name = "nvvm.packed_atomic_state_space", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum PackedAtomicStateSpaceAttr { Global }

#[pliron_attr(name = "nvvm.packed_atomic_ordering", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum PackedAtomicOrderingAttr { Relaxed }

#[pliron_attr(name = "nvvm.packed_atomic_scope", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum PackedAtomicScopeAttr { Gpu }

#[pliron_attr(name = "nvvm.packed_atomic_rounding", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum PackedAtomicRoundingAttr { Rn }

#[pliron_attr(name = "nvvm.packed_atomic_subnormal", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum PackedAtomicSubnormalAttr { NoFtz }

#[pliron_attr(name = "nvvm.packed_atomic_atomicity", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum PackedAtomicAtomicityAttr { PerElement }

/// Packed global atomic add with exact format and semantic attributes.
#[pliron_op(
    name = "nvvm.packed_atomic_add",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
    attributes = (
        nvvm_packed_atomic_format: PackedAtomicFormatAttr,
        nvvm_packed_atomic_state_space: PackedAtomicStateSpaceAttr,
        nvvm_packed_atomic_ordering: PackedAtomicOrderingAttr,
        nvvm_packed_atomic_scope: PackedAtomicScopeAttr,
        nvvm_packed_atomic_rounding: PackedAtomicRoundingAttr,
        nvvm_packed_atomic_subnormal: PackedAtomicSubnormalAttr,
        nvvm_packed_atomic_atomicity: PackedAtomicAtomicityAttr
    )
)]
pub struct PackedAtomicAddOp;

impl PackedAtomicAddOp {
    pub fn new(op: Ptr<Operation>) -> Self { Self { op } }

    pub fn build(
        ctx: &mut Context,
        address: Value,
        addend: Value,
        format: PackedAtomicFormatAttr,
    ) -> Ptr<Operation> {
        let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![u32_ty.into()],
            vec![address, addend],
            vec![],
            0,
        );
        let this = Self { op };
        this.set_attr_nvvm_packed_atomic_format(ctx, format);
        this.set_attr_nvvm_packed_atomic_state_space(ctx, PackedAtomicStateSpaceAttr::Global);
        this.set_attr_nvvm_packed_atomic_ordering(ctx, PackedAtomicOrderingAttr::Relaxed);
        this.set_attr_nvvm_packed_atomic_scope(ctx, PackedAtomicScopeAttr::Gpu);
        this.set_attr_nvvm_packed_atomic_rounding(ctx, PackedAtomicRoundingAttr::Rn);
        this.set_attr_nvvm_packed_atomic_subnormal(ctx, PackedAtomicSubnormalAttr::NoFtz);
        this.set_attr_nvvm_packed_atomic_atomicity(ctx, PackedAtomicAtomicityAttr::PerElement);
        this.get_operation()
    }
}

fn is_u32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {
    ty.deref(ctx).downcast_ref::<IntegerType>().is_some_and(|integer| {
        integer.width() == 32 && integer.signedness() == Signedness::Unsigned
    })
}

impl Verify for PackedAtomicAddOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = self.get_operation().deref(ctx);
        if op.get_num_operands() != 2 || op.get_num_results() != 1 {
            return verify_err!(op.loc(), "nvvm.packed_atomic_add requires exactly two operands and one result");
        }
        let pointer_ty = op.get_operand(0).get_type(ctx);
        let pointer_object = pointer_ty.deref(ctx);
        let Some(pointer) = pointer_object.downcast_ref::<MirPtrType>() else {
            return verify_err!(op.loc(), "nvvm.packed_atomic_add address must be a MIR pointer");
        };
        if !pointer.is_mutable()
            || !matches!(pointer.address_space(), address_space::GENERIC | address_space::GLOBAL)
            || !is_u32(ctx, pointer.pointee)
        {
            return verify_err!(op.loc(), "nvvm.packed_atomic_add address must be a mutable generic/global pointer to u32");
        }
        if !is_u32(ctx, op.get_operand(1).get_type(ctx)) || !is_u32(ctx, op.get_result(0).get_type(ctx)) {
            return verify_err!(op.loc(), "nvvm.packed_atomic_add addend and result must be u32");
        }
        if self.get_attr_nvvm_packed_atomic_format(ctx).is_none()
            || self.get_attr_nvvm_packed_atomic_state_space(ctx).as_deref() != Some(&PackedAtomicStateSpaceAttr::Global)
            || self.get_attr_nvvm_packed_atomic_ordering(ctx).as_deref() != Some(&PackedAtomicOrderingAttr::Relaxed)
            || self.get_attr_nvvm_packed_atomic_scope(ctx).as_deref() != Some(&PackedAtomicScopeAttr::Gpu)
            || self.get_attr_nvvm_packed_atomic_rounding(ctx).as_deref() != Some(&PackedAtomicRoundingAttr::Rn)
            || self.get_attr_nvvm_packed_atomic_subnormal(ctx).as_deref() != Some(&PackedAtomicSubnormalAttr::NoFtz)
            || self.get_attr_nvvm_packed_atomic_atomicity(ctx).as_deref() != Some(&PackedAtomicAtomicityAttr::PerElement)
        {
            return verify_err!(op.loc(), "nvvm.packed_atomic_add has a missing or unsupported semantic attribute");
        }
        Ok(())
    }
}

pub(super) fn register(ctx: &mut Context) {
    PackedAtomicFormatAttr::register(ctx);
    PackedAtomicStateSpaceAttr::register(ctx);
    PackedAtomicOrderingAttr::register(ctx);
    PackedAtomicScopeAttr::register(ctx);
    PackedAtomicRoundingAttr::register(ctx);
    PackedAtomicSubnormalAttr::register(ctx);
    PackedAtomicAtomicityAttr::register(ctx);
    PackedAtomicAddOp::register(ctx);
}
"##,
    );
    output
}

fn render_dialect_packed_alu(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural operations for generated packed floating-point arithmetic.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::{IntegerType, Signedness},\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    value::Value,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\nfn is_i32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == 32)\n}\n\n",
    );
    for record in packed_alus(catalog) {
        let arity = record.rust.arguments.len();
        let parameters = (0..arity)
            .map(|index| format!("arg{index}: Value"))
            .collect::<Vec<_>>()
            .join(", ");
        let operands = (0..arity)
            .map(|index| format!("arg{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "///\n/// Lowers to `{}`.",
            packed_alu_ptx_mnemonic(record)
        )
        .unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<{arity}>, NResultsInterface<1>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    pub fn new(op: Ptr<Operation>) -> Self {{\n        Self {{ op }}\n    }}\n\n    pub fn build(ctx: &mut Context, {parameters}) -> Ptr<Operation> {{\n        let result_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);\n        Operation::new(\n            ctx,\n            Self::get_concrete_op_info(),\n            vec![result_ty.into()],\n            vec![{operands}],\n            vec![],\n            0,\n        )\n    }}\n}}"
        )
        .unwrap();
        writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        let op = self.get_operation().deref(ctx);\n        if op.get_num_operands() != {arity} || op.get_num_results() != 1 {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        if !(0..{arity}).all(|index| is_i32(ctx, op.get_operand(index).get_type(ctx)))\n            || !is_i32(ctx, op.get_result(0).get_type(ctx))\n        {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        Ok(())\n    }}\n}}\n",
            format!(
                "{} requires exactly {arity} operands and one result",
                record.dialect.op_name
            ),
            format!(
                "{} operands and result must be 32-bit integers",
                record.dialect.op_name
            ),
        )
        .unwrap();
    }
    output.push_str("\npub(super) fn register(ctx: &mut Context) {\n");
    for record in packed_alus(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

fn render_dialect_packed_conversion(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Structural operation for generated packed conversion.\n\nuse pliron::{\n    builtin::{\n        op_interfaces::{NOpdsInterface, NResultsInterface},\n        types::{FP32Type, IntegerType, Signedness},\n    },\n    common_traits::Verify,\n    context::{Context, Ptr},\n    location::Located,\n    op::Op,\n    operation::Operation,\n    result::Error,\n    r#type::Typed,\n    value::Value,\n    verify_err,\n};\nuse pliron_derive::pliron_op;\n\nfn is_f32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {\n    ty.deref(ctx).downcast_ref::<FP32Type>().is_some()\n}\n\nfn is_i32(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {\n    ty.deref(ctx)\n        .downcast_ref::<IntegerType>()\n        .is_some_and(|integer| integer.width() == 32)\n}\n\n",
    );
    for record in packed_conversions(catalog) {
        writeln!(output, "/// {}", record.summary).unwrap();
        writeln!(
            output,
            "///\n/// The first input becomes the low {} lane; the second becomes the high lane.",
            packed_conversion_element(record)
        )
        .unwrap();
        writeln!(
            output,
            "#[pliron_op(\n    name = {:?},\n    format,\n    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],\n)]",
            record.dialect.op_name
        )
        .unwrap();
        writeln!(output, "pub struct {};", record.dialect.op_type).unwrap();
        writeln!(output, "\nimpl {} {{", record.dialect.op_type).unwrap();
        output.push_str(
            "    pub fn new(op: Ptr<Operation>) -> Self {\n        Self { op }\n    }\n\n    pub fn build(ctx: &mut Context, low: Value, high: Value) -> Ptr<Operation> {\n        let result_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);\n        Operation::new(\n            ctx,\n            Self::get_concrete_op_info(),\n            vec![result_ty.into()],\n            vec![low, high],\n            vec![],\n            0,\n        )\n    }\n}\n",
        );
        writeln!(output, "\nimpl Verify for {} {{", record.dialect.op_type).unwrap();
        writeln!(
            output,
            "    fn verify(&self, ctx: &Context) -> Result<(), Error> {{\n        let op = self.get_operation().deref(ctx);\n        if op.get_num_operands() != 2 || op.get_num_results() != 1 {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        if !is_f32(ctx, op.get_operand(0).get_type(ctx))\n            || !is_f32(ctx, op.get_operand(1).get_type(ctx))\n            || !is_i32(ctx, op.get_result(0).get_type(ctx))\n        {{\n            return verify_err!(op.loc(), {:?});\n        }}\n        Ok(())\n    }}\n}}\n",
            format!("{} requires two operands and one result", record.dialect.op_name),
            format!(
                "{} requires f32 operands and one 32-bit integer result",
                record.dialect.op_name
            ),
        )
        .unwrap();
    }
    output.push_str("\npub(super) fn register(ctx: &mut Context) {\n");
    for record in packed_conversions(catalog) {
        writeln!(output, "    {}::register(ctx);", record.dialect.op_type).unwrap();
    }
    output.push_str("}\n");
    output
}

fn render_importer_pure_value_dispatch(
    output: &mut String,
    catalog: &CatalogFile,
    record: &CatalogIntrinsic,
) {
    let mut path_refs = vec![record.rust.canonical_path.as_str()];
    path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
    output.push_str("        ");
    render_inline_patterns(output, &path_refs);
    output.push_str(" => {\n");
    writeln!(
        output,
        "            require_arity(name, args.len(), {}, &loc)?;",
        record.rust.arguments.len()
    )
    .unwrap();
    for index in 0..record.rust.arguments.len() {
        let previous = if index == 0 { "prev_op" } else { "last_op" };
        writeln!(
            output,
            "            let (arg{index}, last_op) = rvalue::translate_operand(\n                ctx, body, &args[{index}], value_map, block_ptr, {previous}, loc.clone(),\n            )?;"
        )
        .unwrap();
    }
    let arguments = (0..record.rust.arguments.len())
        .map(|index| format!("arg{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    writeln!(
        output,
        "            let intrinsic = {}::build(ctx, {arguments});",
        record.dialect.op_type
    )
    .unwrap();
    output.push_str("            intrinsic.deref_mut(ctx).set_loc(loc.clone());\n");
    writeln!(
        output,
        "            helpers::set_generated_intrinsic_marker(ctx, intrinsic, {:?});",
        intrinsic_marker(catalog, record)
    )
    .unwrap();
    output.push_str(
        "            helpers::insert_op(ctx, intrinsic, block_ptr, last_op);\n            let result = intrinsic.deref(ctx).get_result(0);\n",
    );
    writeln!(
        output,
        "            Ok(Some(helpers::emit_store_result_and_goto(\n                ctx, destination, result, target, block_ptr, intrinsic, value_map, block_map, loc,\n                {:?},\n            )?))",
        format!("{} call without target block", record.rust.name)
    )
    .unwrap();
    output.push_str("        }\n");
}

fn render_importer(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Generated raw/compatibility path dispatch for CUDA intrinsics.\n\nuse crate::error::{TranslationErr, TranslationResult};\nuse crate::translator::{rvalue, terminator::helpers};\nuse crate::translator::values::ValueMap;\nuse dialect_nvvm::ops::{",
    );
    for (index, record) in sregs(catalog).enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        output.push_str(&record.dialect.op_type);
    }
    if ldmatrix(catalog).next().is_some() {
        if sregs(catalog).next().is_some() {
            output.push_str(", ");
        }
        output.push_str("LdmatrixElementAttr, LdmatrixLayoutAttr, LdmatrixMultiplicityAttr, LdmatrixOp, LdmatrixShapeAttr, LdmatrixStateSpaceAttr");
    }
    if packed_atomics(catalog).next().is_some() {
        if sregs(catalog).next().is_some() || ldmatrix(catalog).next().is_some() {
            output.push_str(", ");
        }
        output.push_str("PackedAtomicAddOp, PackedAtomicFormatAttr");
    }
    if redux(catalog).next().is_some() {
        if sregs(catalog).next().is_some()
            || ldmatrix(catalog).next().is_some()
            || packed_atomics(catalog).next().is_some()
        {
            output.push_str(", ");
        }
        for (index, record) in redux(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    if vote_intrinsics(catalog).next().is_some() {
        if sregs(catalog).next().is_some()
            || ldmatrix(catalog).next().is_some()
            || packed_atomics(catalog).next().is_some()
            || redux(catalog).next().is_some()
        {
            output.push_str(", ");
        }
        for (index, record) in vote_intrinsics(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    if active_masks(catalog).next().is_some() {
        output.push_str(", ");
        for (index, record) in active_masks(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    if warp_matches(catalog).next().is_some() {
        output.push_str(", ");
        for (index, record) in warp_matches(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    if warp_barriers(catalog).next().is_some() {
        output.push_str(", ");
        for (index, record) in warp_barriers(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    if warp_shuffles(catalog).next().is_some() {
        output.push_str(", ");
        for (index, record) in warp_shuffles(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    if packed_alus(catalog).next().is_some() {
        if sregs(catalog).next().is_some()
            || ldmatrix(catalog).next().is_some()
            || packed_atomics(catalog).next().is_some()
            || redux(catalog).next().is_some()
            || vote_intrinsics(catalog).next().is_some()
            || active_masks(catalog).next().is_some()
            || warp_matches(catalog).next().is_some()
            || warp_barriers(catalog).next().is_some()
            || warp_shuffles(catalog).next().is_some()
        {
            output.push_str(", ");
        }
        for (index, record) in packed_alus(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    if packed_conversions(catalog).next().is_some() {
        if sregs(catalog).next().is_some()
            || ldmatrix(catalog).next().is_some()
            || packed_atomics(catalog).next().is_some()
            || redux(catalog).next().is_some()
            || vote_intrinsics(catalog).next().is_some()
            || active_masks(catalog).next().is_some()
            || warp_matches(catalog).next().is_some()
            || warp_barriers(catalog).next().is_some()
            || warp_shuffles(catalog).next().is_some()
            || packed_alus(catalog).next().is_some()
        {
            output.push_str(", ");
        }
        for (index, record) in packed_conversions(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    if cp_async_copies(catalog).next().is_some() {
        output.push_str(", ");
        for (index, record) in cp_async_copies(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    if cp_async_controls(catalog).next().is_some() {
        output.push_str(", ");
        for (index, record) in cp_async_controls(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    if dot_products(catalog).next().is_some() {
        if sregs(catalog).next().is_some()
            || ldmatrix(catalog).next().is_some()
            || packed_atomics(catalog).next().is_some()
            || redux(catalog).next().is_some()
            || vote_intrinsics(catalog).next().is_some()
            || packed_alus(catalog).next().is_some()
            || packed_conversions(catalog).next().is_some()
        {
            output.push_str(", ");
        }
        for (index, record) in dot_products(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    if sync_intrinsics(catalog).next().is_some() {
        if sregs(catalog).next().is_some()
            || ldmatrix(catalog).next().is_some()
            || packed_atomics(catalog).next().is_some()
            || redux(catalog).next().is_some()
            || vote_intrinsics(catalog).next().is_some()
            || dot_products(catalog).next().is_some()
            || packed_alus(catalog).next().is_some()
            || packed_conversions(catalog).next().is_some()
        {
            output.push_str(", ");
        }
        for (index, record) in sync_intrinsics(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    output.push_str(
        "};\nuse pliron::basic_block::BasicBlock;\nuse pliron::context::{Context, Ptr};\nuse pliron::input_err;\nuse pliron::location::{Located, Location};\nuse pliron::op::Op;\nuse pliron::operation::Operation;\nuse rustc_public::{CrateDef, mir, ty::FnDef};\n\n",
    );
    writeln!(
        output,
        "pub const GENERATED_INTRINSIC_ABI: u32 = {};",
        catalog.intrinsic_abi
    )
    .unwrap();
    writeln!(
        output,
        "pub const GENERATED_INTRINSIC_ABI_NAMESPACE: &str = {:?};\n",
        format!("__cuda_oxide_intrinsic_abi_v{}", catalog.intrinsic_abi)
    )
    .unwrap();
    output.push_str(
        "#[derive(Debug, Clone, PartialEq, Eq)]\npub enum RawIntrinsicIdentity {\n    NotRawCrate,\n    Known(String),\n    UnsupportedAbi(String),\n    UnknownId(String),\n}\n\n/// Classify the unrewritten raw intrinsic DefPath. rustc's ordinary name\n/// printer may prefer a public re-export path, which is source API rather than ABI.\npub fn classify_raw_intrinsic(fn_def: FnDef) -> RawIntrinsicIdentity {\n    let crate_name = fn_def.krate().name.to_string();\n    if !matches!(crate_name.as_str(), \"cuda_intrinsics\" | \"cuda-intrinsics\") {\n        return RawIntrinsicIdentity::NotRawCrate;\n    }\n\n    let mut segments = Vec::new();\n    let mut current = Some(fn_def.def_id());\n    while let Some(def_id) = current {\n        let printed = def_id.name();\n        let segment = printed.as_str().rsplit(\"::\").next().unwrap_or_default();\n        if segment != crate_name {\n            segments.push(segment.to_owned());\n        }\n        current = def_id.parent();\n    }\n    segments.reverse();\n    classify_raw_intrinsic_path(&crate_name, format!(\"{crate_name}::{}\", segments.join(\"::\")))\n}\n\nfn classify_raw_intrinsic_path(crate_name: &str, path: String) -> RawIntrinsicIdentity {\n    if !matches!(crate_name, \"cuda_intrinsics\" | \"cuda-intrinsics\") {\n        return RawIntrinsicIdentity::NotRawCrate;\n    }\n    if is_raw_generated_intrinsic_path(&path) {\n        return RawIntrinsicIdentity::Known(path);\n    }\n    let namespace = path.split(\"::\").nth(1).unwrap_or_default();\n    if namespace.starts_with(\"__cuda_oxide_intrinsic_abi_v\")\n        && namespace != GENERATED_INTRINSIC_ABI_NAMESPACE\n    {\n        RawIntrinsicIdentity::UnsupportedAbi(path)\n    } else {\n        RawIntrinsicIdentity::UnknownId(path)\n    }\n}\n\npub fn require_supported_raw_intrinsic(\n    fn_def: FnDef,\n    loc: &Location,\n) -> TranslationResult<Option<String>> {\n    match classify_raw_intrinsic(fn_def) {\n        RawIntrinsicIdentity::NotRawCrate => Ok(None),\n        RawIntrinsicIdentity::Known(path) => Ok(Some(path)),\n        RawIntrinsicIdentity::UnsupportedAbi(path) => input_err!(\n            loc.clone(),\n            TranslationErr::unsupported(format!(\n                \"cuda-intrinsics ABI mismatch: `{path}` uses an unsupported intrinsic ABI; this compiler supports ABI v{GENERATED_INTRINSIC_ABI}\"\n            ))\n        ),\n        RawIntrinsicIdentity::UnknownId(path) => input_err!(\n            loc.clone(),\n            TranslationErr::unsupported(format!(\n                \"cuda-intrinsics ABI mismatch: `{path}` is not a known intrinsic ID in ABI v{GENERATED_INTRINSIC_ABI}\"\n            ))\n        ),\n    }\n}\n\n",
    );
    output
        .push_str("pub fn is_generated_intrinsic_path(name: &str) -> bool {\n    matches!(name,\n");
    render_compiler_path_patterns(&mut output, catalog, "        ");
    output.push_str("    )\n}\n\npub fn is_raw_generated_intrinsic_path(name: &str) -> bool {\n    matches!(name,\n");
    let raw_paths: Vec<_> = catalog
        .intrinsics
        .iter()
        .map(|record| record.rust.canonical_path.as_str())
        .collect();
    render_string_patterns(&mut output, &raw_paths, "        ");
    output.push_str("    )\n}\n\npub fn generated_intrinsic_marker(name: &str) -> Option<&'static str> {\n    match name {\n");
    for record in &catalog.intrinsics {
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        writeln!(output, " => Some({:?}),", intrinsic_marker(catalog, record)).unwrap();
    }
    output.push_str("        _ => None,\n    }\n}\n\n");
    output.push_str(
        "#[allow(clippy::too_many_arguments)]\npub fn try_dispatch_generated_intrinsic(\n    ctx: &mut Context,\n    body: &mir::Body,\n    name: &str,\n    args: &[mir::Operand],\n    destination: &mir::Place,\n    target: &Option<usize>,\n    block_ptr: Ptr<BasicBlock>,\n    prev_op: Option<Ptr<Operation>>,\n    value_map: &mut ValueMap,\n    block_map: &[Ptr<BasicBlock>],\n    loc: Location,\n) -> TranslationResult<Option<Ptr<Operation>>> {\n    match name {\n",
    );
    for record in sregs(catalog) {
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        writeln!(
            output,
            "            require_arity(name, args.len(), 0, &loc)?;"
        )
        .unwrap();
        let helper = if record.scalar_width() == Some(64) {
            "emit_generated_nvvm_intrinsic_u64"
        } else {
            "emit_generated_nvvm_intrinsic"
        };
        writeln!(output, "            Ok(Some(helpers::{helper}(").unwrap();
        writeln!(
            output,
            "                ctx, {}::get_concrete_op_info(), {:?}, destination, target, block_ptr,",
            record.dialect.op_type,
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
            "                prev_op, value_map, block_map, loc,\n            )?))\n        }\n",
        );
    }
    for record in active_masks(catalog) {
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 0, &loc)?;\n");
        writeln!(
            output,
            "            let active_mask = {}::build(ctx);",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str("            active_mask.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, active_mask, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
            "            helpers::insert_op(ctx, active_mask, block_ptr, prev_op);\n            let result = active_mask.deref(ctx).get_result(0);\n",
        );
        writeln!(
            output,
            "            Ok(Some(helpers::emit_store_result_and_goto(\n                ctx, destination, result, target, block_ptr, active_mask, value_map, block_map, loc,\n                {:?},\n            )?))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("        }\n");
    }
    for record in ldmatrix(catalog) {
        let (shape, multiplicity, layout, element, state_space) = ldmatrix_attr_variants(record);
        let register_count = record
            .ldmatrix
            .as_ref()
            .unwrap()
            .variant
            .multiplicity
            .register_count();
        output.push_str("        ");
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 1, &loc)?;\n");
        output.push_str(
            "            let (address, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n",
        );
        writeln!(
            output,
            "            let load = LdmatrixOp::build(ctx, address, {shape}, {multiplicity}, {layout}, {element}, {state_space});"
        )
        .unwrap();
        output.push_str("            load.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, load, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str("            helpers::insert_op(ctx, load, block_ptr, last_op);\n");
        if register_count == 1 {
            output.push_str("            let value = load.deref(ctx).get_result(0);\n");
            output.push_str("            let last_op = load;\n");
        } else {
            writeln!(
                output,
                "            let (value, last_op) = helpers::bundle_generated_u32_results_as_array(ctx, load, {register_count}, loc.clone());"
            )
            .unwrap();
        }
        writeln!(
            output,
            "            Ok(Some(helpers::emit_store_result_and_goto(\n                ctx, destination, value, target, block_ptr, last_op, value_map, block_map, loc,\n                {:?},\n            )?))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("        }\n");
    }
    for record in packed_atomics(catalog) {
        let format = match record.packed_atomic.as_ref().unwrap().format {
            PackedAtomicFormat::F16x2 => "PackedAtomicFormatAttr::F16x2",
            PackedAtomicFormat::Bf16x2 => "PackedAtomicFormatAttr::Bf16x2",
        };
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 2, &loc)?;\n");
        output.push_str(
            "            let (address, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n",
        );
        output.push_str(
            "            let (addend, last_op) = rvalue::translate_operand(\n                ctx, body, &args[1], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n",
        );
        writeln!(
            output,
            "            let atom = PackedAtomicAddOp::build(ctx, address, addend, {format});"
        )
        .unwrap();
        output.push_str("            atom.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, atom, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str("            helpers::insert_op(ctx, atom, block_ptr, last_op);\n");
        output.push_str("            let value = atom.deref(ctx).get_result(0);\n");
        writeln!(
            output,
            "            Ok(Some(helpers::emit_store_result_and_goto(\n                ctx, destination, value, target, block_ptr, atom, value_map, block_map, loc,\n                {:?},\n            )?))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("        }\n");
    }
    for record in redux(catalog) {
        debug_assert_eq!(
            record.redux.as_ref().unwrap().adapter,
            ReduxAdapter::MaskValueToSourceMemberMask
        );
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 2, &loc)?;\n");
        output.push_str(
            "            let (member_mask, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n",
        );
        output.push_str(
            "            let (value, last_op) = rvalue::translate_operand(\n                ctx, body, &args[1], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n",
        );
        writeln!(
            output,
            "            let reduction = {}::build(ctx, member_mask, value);",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str("            reduction.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, reduction, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
            "            helpers::insert_op(ctx, reduction, block_ptr, last_op);\n            let result = reduction.deref(ctx).get_result(0);\n",
        );
        writeln!(
            output,
            "            Ok(Some(helpers::emit_store_result_and_goto(\n                ctx, destination, result, target, block_ptr, reduction, value_map, block_map, loc,\n                {:?},\n            )?))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("        }\n");
    }
    for record in vote_intrinsics(catalog) {
        debug_assert_eq!(
            record.vote.as_ref().unwrap().adapter,
            VoteAdapter::DirectMaskPredicate
        );
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 2, &loc)?;\n");
        output.push_str(
            "            let (member_mask, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n",
        );
        output.push_str(
            "            let (predicate, last_op) = rvalue::translate_operand(\n                ctx, body, &args[1], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n",
        );
        writeln!(
            output,
            "            let vote = {}::build(ctx, member_mask, predicate);",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str("            vote.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, vote, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
            "            helpers::insert_op(ctx, vote, block_ptr, last_op);\n            let result = vote.deref(ctx).get_result(0);\n",
        );
        writeln!(
            output,
            "            Ok(Some(helpers::emit_store_result_and_goto(\n                ctx, destination, result, target, block_ptr, vote, value_map, block_map, loc,\n                {:?},\n            )?))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("        }\n");
    }
    for record in warp_matches(catalog) {
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 2, &loc)?;\n");
        output.push_str(
            "            let (member_mask, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n",
        );
        output.push_str(
            "            let (value, last_op) = rvalue::translate_operand(\n                ctx, body, &args[1], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n",
        );
        writeln!(
            output,
            "            let warp_match = {}::build(ctx, member_mask, value);",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str("            warp_match.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, warp_match, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
            "            helpers::insert_op(ctx, warp_match, block_ptr, last_op);\n            let result = warp_match.deref(ctx).get_result(0);\n",
        );
        writeln!(
            output,
            "            Ok(Some(helpers::emit_store_result_and_goto(\n                ctx, destination, result, target, block_ptr, warp_match, value_map, block_map, loc,\n                {:?},\n            )?))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("        }\n");
    }
    for record in warp_barriers(catalog) {
        debug_assert_eq!(
            record.warp_barrier.as_ref().unwrap().adapter,
            WarpBarrierAdapter::DirectMemberMask
        );
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 1, &loc)?;\n");
        output.push_str(
            "            let (member_mask, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n",
        );
        writeln!(
            output,
            "            let barrier = {}::build(ctx, member_mask);",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str("            barrier.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, barrier, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
            "            helpers::insert_op(ctx, barrier, block_ptr, last_op);\n            if let Some(target_idx) = target {\n                Ok(Some(helpers::emit_goto(ctx, *target_idx, barrier, block_map, loc)))\n            } else {\n",
        );
        writeln!(
            output,
            "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("            }\n        }\n");
    }
    for record in warp_shuffles(catalog) {
        debug_assert!(matches!(
            record.warp_shuffle.as_ref().unwrap().adapter,
            WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp
                | WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble
        ));
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 3, &loc)?;\n");
        output.push_str(
            "            let (member_mask, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n",
        );
        output.push_str(
            "            let (value, last_op) = rvalue::translate_operand(\n                ctx, body, &args[1], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n",
        );
        output.push_str(
            "            let (lane_or_delta, last_op) = rvalue::translate_operand(\n                ctx, body, &args[2], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n",
        );
        writeln!(
            output,
            "            let shuffle = {}::build(ctx, member_mask, value, lane_or_delta);",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str("            shuffle.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, shuffle, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
            "            helpers::insert_op(ctx, shuffle, block_ptr, last_op);\n            let result = shuffle.deref(ctx).get_result(0);\n",
        );
        writeln!(
            output,
            "            Ok(Some(helpers::emit_store_result_and_goto(\n                ctx, destination, result, target, block_ptr, shuffle, value_map, block_map, loc,\n                {:?},\n            )?))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("        }\n");
    }
    for record in dot_products(catalog) {
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 3, &loc)?;\n");
        output.push_str(
            "            let (a, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n",
        );
        output.push_str(
            "            let (b, last_op) = rvalue::translate_operand(\n                ctx, body, &args[1], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n",
        );
        output.push_str(
            "            let (c, last_op) = rvalue::translate_operand(\n                ctx, body, &args[2], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n",
        );
        writeln!(
            output,
            "            let dot = {}::build(ctx, a, b, c);",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str("            dot.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, dot, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
            "            helpers::insert_op(ctx, dot, block_ptr, last_op);\n            let result = dot.deref(ctx).get_result(0);\n",
        );
        writeln!(
            output,
            "            Ok(Some(helpers::emit_store_result_and_goto(\n                ctx, destination, result, target, block_ptr, dot, value_map, block_map, loc,\n                {:?},\n            )?))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("        }\n");
    }
    for record in packed_alus(catalog) {
        render_importer_pure_value_dispatch(&mut output, catalog, record);
    }
    for record in packed_conversions(catalog) {
        render_importer_pure_value_dispatch(&mut output, catalog, record);
    }
    for record in cp_async_copies(catalog) {
        let copy = record.cp_async_copy.as_ref().unwrap();
        let dynamic = copy.source_size == CpAsyncSourceSize::Runtime;
        let arity = if dynamic { 3 } else { 2 };
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        writeln!(
            output,
            "            require_arity(name, args.len(), {arity}, &loc)?;"
        )
        .unwrap();
        output.push_str(
            "            let (shared_dst, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n",
        );
        output.push_str(
            "            let (global_src, last_op) = rvalue::translate_operand(\n                ctx, body, &args[1], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n",
        );
        if dynamic {
            output.push_str(
                "            let (source_size, last_op) = rvalue::translate_operand(\n                ctx, body, &args[2], value_map, block_ptr, last_op, loc.clone(),\n            )?;\n",
            );
            writeln!(
                output,
                "            let copy = {}::build(ctx, shared_dst, global_src, source_size);",
                record.dialect.op_type
            )
            .unwrap();
        } else {
            writeln!(
                output,
                "            let copy = {}::build(ctx, shared_dst, global_src);",
                record.dialect.op_type
            )
            .unwrap();
        }
        output.push_str("            copy.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, copy, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
            "            helpers::insert_op(ctx, copy, block_ptr, last_op);\n            if let Some(target_idx) = target {\n                Ok(Some(helpers::emit_goto(ctx, *target_idx, copy, block_map, loc)))\n            } else {\n",
        );
        writeln!(
            output,
            "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("            }\n        }\n");
    }
    for record in cp_async_controls(catalog) {
        let control = record.cp_async_control.as_ref().unwrap();
        let has_immediate = control.operation == CpAsyncControlOperation::WaitGroup;
        let arity = usize::from(has_immediate);
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        writeln!(
            output,
            "            require_arity(name, args.len(), {arity}, &loc)?;"
        )
        .unwrap();
        if has_immediate {
            output.push_str(
                "            if !matches!(&args[0], mir::Operand::Constant(_)) {\n                return input_err!(\n                    loc,\n                    TranslationErr::unsupported(\n                        \"cp_async_wait_group requires a compile-time constant max_pending value\".to_owned()\n                    )\n                );\n            }\n",
            );
            output.push_str(
                "            let (max_pending, last_op) = rvalue::translate_operand(\n                ctx, body, &args[0], value_map, block_ptr, prev_op, loc.clone(),\n            )?;\n",
            );
            writeln!(
                output,
                "            let control = {}::build(ctx, max_pending);",
                record.dialect.op_type
            )
            .unwrap();
        } else {
            output.push_str("            let last_op = prev_op;\n");
            writeln!(
                output,
                "            let control = {}::build(ctx);",
                record.dialect.op_type
            )
            .unwrap();
        }
        output.push_str("            control.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, control, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
            "            helpers::insert_op(ctx, control, block_ptr, last_op);\n            if let Some(target_idx) = target {\n                Ok(Some(helpers::emit_goto(ctx, *target_idx, control, block_map, loc)))\n            } else {\n",
        );
        writeln!(
            output,
            "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("            }\n        }\n");
    }
    for record in sync_intrinsics(catalog) {
        let mut path_refs = vec![record.rust.canonical_path.as_str()];
        path_refs.extend(record.rust.compatibility_paths.iter().map(String::as_str));
        output.push_str("        ");
        render_inline_patterns(&mut output, &path_refs);
        output.push_str(" => {\n");
        output.push_str("            require_arity(name, args.len(), 0, &loc)?;\n");
        writeln!(
            output,
            "            let barrier = Operation::new(ctx, {}::get_concrete_op_info(), vec![], vec![], vec![], 0);",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str("            barrier.deref_mut(ctx).set_loc(loc.clone());\n");
        writeln!(
            output,
            "            helpers::set_generated_intrinsic_marker(ctx, barrier, {:?});",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        output.push_str(
            "            helpers::insert_op(ctx, barrier, block_ptr, prev_op);\n            if let Some(target_idx) = target {\n                Ok(Some(helpers::emit_goto(ctx, *target_idx, barrier, block_map, loc)))\n            } else {\n",
        );
        writeln!(
            output,
            "                input_err!(loc, TranslationErr::unsupported({:?}.to_owned()))",
            format!("{} call without target block", record.rust.name)
        )
        .unwrap();
        output.push_str("            }\n        }\n");
    }
    output.push_str("        _ => Ok(None),\n    }\n}\n\n");
    output.push_str(
        "fn require_arity(\n    name: &str,\n    actual: usize,\n    expected: usize,\n    loc: &Location,\n) -> TranslationResult<()> {\n    if actual != expected {\n        return input_err!(\n            loc.clone(),\n            TranslationErr::unsupported(format!(\n                \"generated intrinsic `{name}` expects {expected} arguments, got {actual}\"\n            ))\n        );\n    }\n    Ok(())\n}\n",
    );
    output.push_str("\n#[cfg(test)]\nmod tests {\n    use super::*;\n\n");
    for record in &catalog.intrinsics {
        writeln!(
            output,
            "    #[test]\n    fn {}_uses_only_canonical_or_compatibility_defpaths() {{",
            record.id
        )
        .unwrap();
        writeln!(
            output,
            "        assert!(is_generated_intrinsic_path({:?}));",
            record.rust.canonical_path
        )
        .unwrap();
        writeln!(
            output,
            "        assert!(is_raw_generated_intrinsic_path({:?}));",
            record.rust.canonical_path
        )
        .unwrap();
        writeln!(
            output,
            "        assert_eq!(generated_intrinsic_marker({:?}), Some({:?}));",
            record.rust.canonical_path,
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        writeln!(
            output,
            "        assert!(matches!(classify_raw_intrinsic_path(\"cuda_intrinsics\", {:?}.into()), RawIntrinsicIdentity::Known(_)));",
            record.rust.canonical_path
        )
        .unwrap();
        writeln!(
            output,
            "        assert!(!is_generated_intrinsic_path({:?}));",
            record.rust.public_path
        )
        .unwrap();
        for compatibility_path in &record.rust.compatibility_paths {
            writeln!(
                output,
                "        assert!(is_generated_intrinsic_path({compatibility_path:?}));\n        assert!(!is_raw_generated_intrinsic_path({compatibility_path:?}));\n        assert_eq!(generated_intrinsic_marker({compatibility_path:?}), Some({:?}));",
                intrinsic_marker(catalog, record)
            )
            .unwrap();
        }
        writeln!(
            output,
            "        assert!(!is_generated_intrinsic_path(\"cuda_intrinsics::__cuda_oxide_intrinsic_abi_v{}::{}\"));",
            catalog.intrinsic_abi + 1,
            record.rust.abi_id
        )
        .unwrap();
        output.push_str("    }\n");
    }
    output.push_str(
        "\n    #[test]\n    fn raw_intrinsic_identity_classification_fails_closed() {\n        assert_eq!(\n            classify_raw_intrinsic_path(\"serde\", \"serde::helper\".into()),\n            RawIntrinsicIdentity::NotRawCrate\n        );\n        assert!(matches!(\n            classify_raw_intrinsic_path(\"cuda_intrinsics\", \"cuda_intrinsics::__cuda_oxide_intrinsic_abi_v2::i0001\".into()),\n            RawIntrinsicIdentity::UnsupportedAbi(_)\n        ));\n        assert!(matches!(\n            classify_raw_intrinsic_path(\"cuda_intrinsics\", \"cuda_intrinsics::__cuda_oxide_intrinsic_abi_v1::i9999\".into()),\n            RawIntrinsicIdentity::UnknownId(_)\n        ));\n        assert!(matches!(\n            classify_raw_intrinsic_path(\"cuda_intrinsics\", \"cuda_intrinsics::helper\".into()),\n            RawIntrinsicIdentity::UnknownId(_)\n        ));\n    }\n",
    );
    output.push_str("}\n");
    output
}

fn render_lowering(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Generated conversion interfaces for admitted CUDA intrinsic families.\n\nuse crate::conversion_interface::MirToLlvmConversion;\nuse crate::convert::intrinsics::{atomic::convert_packed_atom_add, common::{call_intrinsic, create_i32_const, inline_asm_convergent}, cp_async::{convert_generated_cp_async_control, convert_generated_cp_async_copy}, dotprod::convert_generated_dot_product, ldmatrix::convert_generated_ldmatrix, packed::{convert_generated_packed_alu, convert_generated_packed_f32x2}, warp::{convert_active_mask, convert_bar_warp_sync, convert_match_all, convert_match_any, convert_redux, convert_shuffle_f32, convert_shuffle_i32, convert_shuffle_i64, convert_vote}};\nuse crate::{context, IntrinsicBackend};\nuse dialect_nvvm::ops::{",
    );
    for (index, record) in sregs(catalog).enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        output.push_str(&record.dialect.op_type);
    }
    if ldmatrix(catalog).next().is_some() {
        if sregs(catalog).next().is_some() {
            output.push_str(", ");
        }
        output.push_str("LdmatrixElementAttr, LdmatrixLayoutAttr, LdmatrixMultiplicityAttr, LdmatrixOp, LdmatrixShapeAttr, LdmatrixStateSpaceAttr");
    }
    if packed_atomics(catalog).next().is_some() {
        if sregs(catalog).next().is_some() || ldmatrix(catalog).next().is_some() {
            output.push_str(", ");
        }
        output.push_str("PackedAtomicAddOp, PackedAtomicAtomicityAttr, PackedAtomicFormatAttr, PackedAtomicOrderingAttr, PackedAtomicRoundingAttr, PackedAtomicScopeAttr, PackedAtomicStateSpaceAttr, PackedAtomicSubnormalAttr");
    }
    if redux(catalog).next().is_some() {
        if sregs(catalog).next().is_some()
            || ldmatrix(catalog).next().is_some()
            || packed_atomics(catalog).next().is_some()
        {
            output.push_str(", ");
        }
        for (index, record) in redux(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    if vote_intrinsics(catalog).next().is_some() {
        if sregs(catalog).next().is_some()
            || ldmatrix(catalog).next().is_some()
            || packed_atomics(catalog).next().is_some()
            || redux(catalog).next().is_some()
        {
            output.push_str(", ");
        }
        for (index, record) in vote_intrinsics(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    if active_masks(catalog).next().is_some() {
        output.push_str(", ");
        for (index, record) in active_masks(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    if warp_matches(catalog).next().is_some() {
        output.push_str(", ");
        for (index, record) in warp_matches(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    if warp_barriers(catalog).next().is_some() {
        output.push_str(", ");
        for (index, record) in warp_barriers(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    if dot_products(catalog).next().is_some() {
        if sregs(catalog).next().is_some()
            || ldmatrix(catalog).next().is_some()
            || packed_atomics(catalog).next().is_some()
            || redux(catalog).next().is_some()
            || vote_intrinsics(catalog).next().is_some()
        {
            output.push_str(", ");
        }
        for (index, record) in dot_products(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    if sync_intrinsics(catalog).next().is_some() {
        if sregs(catalog).next().is_some()
            || ldmatrix(catalog).next().is_some()
            || packed_atomics(catalog).next().is_some()
            || redux(catalog).next().is_some()
            || vote_intrinsics(catalog).next().is_some()
            || dot_products(catalog).next().is_some()
        {
            output.push_str(", ");
        }
        for (index, record) in sync_intrinsics(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    if warp_shuffles(catalog).next().is_some() {
        output.push_str(", ");
        for (index, record) in warp_shuffles(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    if packed_alus(catalog).next().is_some() {
        if catalog
            .intrinsics
            .iter()
            .any(|record| record.family != "packed_alu")
        {
            output.push_str(", ");
        }
        for (index, record) in packed_alus(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    if packed_conversions(catalog).next().is_some() {
        if catalog
            .intrinsics
            .iter()
            .any(|record| record.family != "packed_conversion")
        {
            output.push_str(", ");
        }
        for (index, record) in packed_conversions(catalog).enumerate() {
            if index != 0 {
                output.push_str(", ");
            }
            output.push_str(&record.dialect.op_type);
        }
    }
    for record in cp_async_copies(catalog).chain(cp_async_controls(catalog)) {
        output.push_str(", ");
        output.push_str(&record.dialect.op_type);
    }
    output.push_str(
        "};\nuse llvm_export::types as llvm_types;\nuse pliron::{\n    builtin::types::{IntegerType, Signedness},\n    context::{Context, Ptr},\n    derive::op_interface_impl,\n    irbuild::{\n        dialect_conversion::{DialectConversionRewriter, OperandsInfo},\n        rewriter::Rewriter,\n    },\n    op::Op,\n    operation::Operation,\n    result::Result,\n};\n\n",
    );
    output.push_str("use pliron::location::Located;\n\n");
    output.push_str(
        "fn convert_zero_operand_scalar_direct(\n    ctx: &mut Context,\n    rewriter: &mut DialectConversionRewriter,\n    op: Ptr<Operation>,\n    width: u32,\n    intrinsic_name: &str,\n) -> Result<()> {\n    let result_ty = IntegerType::get(ctx, width, Signedness::Signless);\n    let function_ty = llvm_types::FuncType::get(ctx, result_ty.into(), vec![], false);\n    let call = call_intrinsic(ctx, rewriter, op, intrinsic_name, function_ty, vec![])?;\n    rewriter.replace_operation(ctx, op, call);\n    Ok(())\n}\n\n",
    );
    for record in sregs(catalog) {
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n",
        );
        writeln!(
            output,
            "        convert_zero_operand_scalar_direct(ctx, rewriter, self.get_operation(), {}, {:?})",
            record.scalar_width().unwrap(),
            record.llvm_identifier()
        )
        .unwrap();
        output.push_str("    }\n}\n\n");
    }
    for record in active_masks(catalog) {
        debug_assert_eq!(
            record.active_mask.as_ref().unwrap().adapter,
            ActiveMaskAdapter::DirectZeroOperandMask
        );
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        let op = self.get_operation();\n        match context::lowering_options(ctx).intrinsic_backend {\n            IntrinsicBackend::LlvmNvptx => {\n                convert_active_mask(ctx, rewriter, op, operands_info)\n            }\n            IntrinsicBackend::LibNvvm => {\n                let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);\n                let inline_asm = inline_asm_convergent(\n                    ctx,\n                    rewriter,\n                    i32_ty.into(),\n                    vec![],\n                    \"activemask.b32 $0;\",\n                    \"=r,~{memory}\",\n                );\n                rewriter.replace_operation(ctx, op, inline_asm);\n                Ok(())\n            }\n        }\n    }\n}\n\n",
        );
    }
    if ldmatrix(catalog).next().is_some() {
        output.push_str(
            "#[op_interface_impl]\nimpl MirToLlvmConversion for LdmatrixOp {\n    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        let recipe = {\n            let shape = self.get_attr_nvvm_ldmatrix_shape(ctx);\n            let multiplicity = self.get_attr_nvvm_ldmatrix_multiplicity(ctx);\n            let layout = self.get_attr_nvvm_ldmatrix_layout(ctx);\n            let element = self.get_attr_nvvm_ldmatrix_element(ctx);\n            let state_space = self.get_attr_nvvm_ldmatrix_state_space(ctx);\n            match (shape.as_deref(), multiplicity.as_deref(), layout.as_deref(), element.as_deref(), state_space.as_deref()) {\n",
        );
        for record in ldmatrix(catalog) {
            let (shape, multiplicity, layout, element, state_space) =
                ldmatrix_attr_variants(record);
            let variant = &record.ldmatrix.as_ref().unwrap().variant;
            let register_count = variant.multiplicity.register_count();
            let transposed = variant.layout == LdmatrixLayout::Transposed;
            let intrinsic_name = record
                .llvm
                .as_ref()
                .expect("ldmatrix LLVM facts")
                .resolved_symbol
                .as_ref()
                .expect("ldmatrix resolved symbol")
                .replace('.', "_");
            writeln!(
                output,
                "                (Some(&{shape}), Some(&{multiplicity}), Some(&{layout}), Some(&{element}), Some(&{state_space})) => ({register_count}, {transposed}, {intrinsic_name:?}),"
            )
            .unwrap();
        }
        output.push_str(
            "                _ => return pliron::input_err!(\n                    self.get_operation().deref(ctx).loc(),\n                    \"nvvm.ldmatrix variant has no generated lowering recipe\",\n                ),\n            }\n        };\n        convert_generated_ldmatrix(ctx, rewriter, self.get_operation(), recipe.0, recipe.1, recipe.2)\n    }\n}\n\n",
        );
    }
    if packed_atomics(catalog).next().is_some() {
        output.push_str(
            "#[op_interface_impl]\nimpl MirToLlvmConversion for PackedAtomicAddOp {\n    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        let format = self.get_attr_nvvm_packed_atomic_format(ctx);\n        let state_space = self.get_attr_nvvm_packed_atomic_state_space(ctx);\n        let ordering = self.get_attr_nvvm_packed_atomic_ordering(ctx);\n        let scope = self.get_attr_nvvm_packed_atomic_scope(ctx);\n        let rounding = self.get_attr_nvvm_packed_atomic_rounding(ctx);\n        let subnormal = self.get_attr_nvvm_packed_atomic_subnormal(ctx);\n        let atomicity = self.get_attr_nvvm_packed_atomic_atomicity(ctx);\n        let ptx_type = match (format.as_deref(), state_space.as_deref(), ordering.as_deref(), scope.as_deref(), rounding.as_deref(), subnormal.as_deref(), atomicity.as_deref()) {\n            (Some(&PackedAtomicFormatAttr::F16x2), Some(&PackedAtomicStateSpaceAttr::Global), Some(&PackedAtomicOrderingAttr::Relaxed), Some(&PackedAtomicScopeAttr::Gpu), Some(&PackedAtomicRoundingAttr::Rn), Some(&PackedAtomicSubnormalAttr::NoFtz), Some(&PackedAtomicAtomicityAttr::PerElement)) => \"f16x2\",\n            (Some(&PackedAtomicFormatAttr::Bf16x2), Some(&PackedAtomicStateSpaceAttr::Global), Some(&PackedAtomicOrderingAttr::Relaxed), Some(&PackedAtomicScopeAttr::Gpu), Some(&PackedAtomicRoundingAttr::Rn), Some(&PackedAtomicSubnormalAttr::NoFtz), Some(&PackedAtomicAtomicityAttr::PerElement)) => \"bf16x2\",\n            _ => return pliron::input_err!(\n                self.get_operation().deref(ctx).loc(),\n                \"nvvm.packed_atomic_add attributes have no generated lowering recipe\",\n            ),\n        };\n        convert_packed_atom_add(ctx, rewriter, self.get_operation(), ptx_type)\n    }\n}\n\n",
        );
        output = output.replace(
            "        convert_packed_atom_add(ctx, rewriter, self.get_operation(), ptx_type)",
            "        drop((format, state_space, ordering, scope, rounding, subnormal, atomicity));\n        convert_packed_atom_add(ctx, rewriter, self.get_operation(), ptx_type)",
        );
    }
    for record in redux(catalog) {
        debug_assert_eq!(
            record.redux.as_ref().unwrap().adapter,
            ReduxAdapter::MaskValueToSourceMemberMask
        );
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        operands_info: &OperandsInfo,\n    ) -> Result<()> {\n",
        );
        writeln!(
            output,
            "        convert_redux(ctx, rewriter, self.get_operation(), operands_info, {:?})",
            record.llvm_identifier()
        )
        .unwrap();
        output.push_str("    }\n}\n\n");
    }
    for record in vote_intrinsics(catalog) {
        debug_assert_eq!(
            record.vote.as_ref().unwrap().adapter,
            VoteAdapter::DirectMaskPredicate
        );
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        operands_info: &OperandsInfo,\n    ) -> Result<()> {\n",
        );
        writeln!(
            output,
            "        convert_vote(ctx, rewriter, self.get_operation(), operands_info, {:?})",
            record.llvm_identifier()
        )
        .unwrap();
        output.push_str("    }\n}\n\n");
    }
    for record in warp_matches(catalog) {
        let warp_match = record.warp_match.as_ref().unwrap();
        let helper = match warp_match.adapter {
            WarpMatchAdapter::DirectMask => "convert_match_any",
            WarpMatchAdapter::ProjectMaskDiscardPredicate => "convert_match_all",
        };
        let value_width = warp_match.value_width.bits();
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        operands_info: &OperandsInfo,\n    ) -> Result<()> {\n",
        );
        writeln!(
            output,
            "        let value_ty = IntegerType::get(ctx, {value_width}, Signedness::Signless);"
        )
        .unwrap();
        writeln!(
            output,
            "        {helper}(ctx, rewriter, self.get_operation(), operands_info, {:?}, value_ty.into())",
            record.llvm_identifier()
        )
        .unwrap();
        output.push_str("    }\n}\n\n");
    }
    for record in warp_barriers(catalog) {
        debug_assert_eq!(
            record.warp_barrier.as_ref().unwrap().adapter,
            WarpBarrierAdapter::DirectMemberMask
        );
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        convert_bar_warp_sync(ctx, rewriter, self.get_operation(), operands_info)\n    }\n}\n\n",
        );
    }
    for record in warp_shuffles(catalog) {
        let shuffle = record.warp_shuffle.as_ref().unwrap();
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        operands_info: &OperandsInfo,\n    ) -> Result<()> {\n",
        );
        match shuffle.value_kind {
            WarpShuffleValueKind::I32 | WarpShuffleValueKind::F32 => {
                debug_assert_eq!(
                    shuffle.adapter,
                    WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp
                );
                let helper = match shuffle.value_kind {
                    WarpShuffleValueKind::I32 => "convert_shuffle_i32",
                    WarpShuffleValueKind::F32 => "convert_shuffle_f32",
                    WarpShuffleValueKind::I64 => unreachable!(),
                };
                writeln!(
                    output,
                    "        {helper}(ctx, rewriter, self.get_operation(), operands_info, {:?}, {})",
                    record.llvm_identifier(),
                    shuffle.clamp,
                )
                .unwrap();
            }
            WarpShuffleValueKind::I64 => {
                debug_assert_eq!(
                    shuffle.adapter,
                    WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble
                );
                let mode = match shuffle.mode {
                    WarpShuffleMode::Idx => "idx",
                    WarpShuffleMode::Bfly => "bfly",
                    WarpShuffleMode::Down => "down",
                    WarpShuffleMode::Up => "up",
                };
                writeln!(
                    output,
                    "        convert_shuffle_i64(ctx, rewriter, self.get_operation(), operands_info, {mode:?}, {})",
                    shuffle.clamp,
                )
                .unwrap();
            }
        }
        output.push_str("    }\n}\n\n");
    }
    for record in packed_alus(catalog) {
        debug_assert_eq!(
            record.packed_alu.as_ref().unwrap().adapter,
            PackedAluAdapter::DirectPackedU32
        );
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n",
        );
        writeln!(
            output,
            "        convert_generated_packed_alu(ctx, rewriter, self.get_operation(), {:?})",
            packed_alu_ptx_mnemonic(record)
        )
        .unwrap();
        output.push_str("    }\n}\n\n");
    }
    for record in packed_conversions(catalog) {
        debug_assert_eq!(
            record.packed_conversion.as_ref().unwrap().adapter,
            PackedConversionAdapter::ReverseHighLowOperands
        );
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n",
        );
        writeln!(
            output,
            "        convert_generated_packed_f32x2(ctx, rewriter, self.get_operation(), {:?})",
            packed_conversion_ptx_mnemonic(record)
        )
        .unwrap();
        output.push_str("    }\n}\n\n");
    }
    for record in cp_async_copies(catalog) {
        let copy = record.cp_async_copy.as_ref().unwrap();
        let cache_policy = match copy.cache_policy {
            CpAsyncCachePolicy::Ca => "ca",
            CpAsyncCachePolicy::Cg => "cg",
        };
        let has_source_size = copy.source_size == CpAsyncSourceSize::Runtime;
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n",
        );
        writeln!(
            output,
            "        convert_generated_cp_async_copy(ctx, rewriter, self.get_operation(), {cache_policy:?}, {}, {has_source_size}, {:?})",
            copy.copy_size.bytes(),
            record.llvm_identifier(),
        )
        .unwrap();
        output.push_str("    }\n}\n\n");
    }
    for record in cp_async_controls(catalog) {
        let operation = match record.cp_async_control.as_ref().unwrap().operation {
            CpAsyncControlOperation::CommitGroup => "commit_group",
            CpAsyncControlOperation::WaitAll => "wait_all",
            CpAsyncControlOperation::WaitGroup => "wait_group",
        };
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n",
        );
        writeln!(
            output,
            "        convert_generated_cp_async_control(ctx, rewriter, self.get_operation(), {operation:?}, {:?})",
            record.llvm_identifier(),
        )
        .unwrap();
        output.push_str("    }\n}\n\n");
    }
    for record in dot_products(catalog) {
        let insert_low_half_selector = matches!(
            record.dot_product.as_ref().unwrap().adapter,
            DotProductAdapter::InsertLowHalfFalse
        );
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n",
        );
        writeln!(
            output,
            "        convert_generated_dot_product(ctx, rewriter, self.get_operation(), {:?}, {:?}, {insert_low_half_selector})",
            record.llvm_identifier(),
            dot_product_ptx(record),
        )
        .unwrap();
        output.push_str("    }\n}\n\n");
    }
    for record in sync_intrinsics(catalog) {
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        let op = self.get_operation();\n        let void_ty = llvm_types::VoidType::get(ctx);\n        match context::lowering_options(ctx).intrinsic_backend {\n            IntrinsicBackend::LlvmNvptx => {\n                let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);\n                let barrier_id = create_i32_const(ctx, rewriter, 0);\n                let function_ty = llvm_types::FuncType::get(ctx, void_ty.into(), vec![i32_ty.into()], false);\n",
        );
        writeln!(
            output,
            "                call_intrinsic(ctx, rewriter, op, {:?}, function_ty, vec![barrier_id])?;",
            record.llvm_identifier()
        )
        .unwrap();
        output.push_str(
            "            }\n            IntrinsicBackend::LibNvvm => {\n                inline_asm_convergent(ctx, rewriter, void_ty.into(), vec![], \"bar.sync 0;\", \"~{memory}\");\n            }\n        }\n        rewriter.erase_operation(ctx, op);\n        Ok(())\n    }\n}\n\n",
        );
    }
    output
}

fn render_collector(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    let abi_namespace = format!("__cuda_oxide_intrinsic_abi_v{}", catalog.intrinsic_abi);
    output.push_str("//! Generated collector predicates for intrinsic placeholder functions.\n\n");
    writeln!(
        output,
        "pub(crate) const GENERATED_INTRINSIC_ABI: u32 = {};",
        catalog.intrinsic_abi
    )
    .unwrap();
    writeln!(
        output,
        "pub(crate) const GENERATED_INTRINSIC_ABI_NAMESPACE: &str = {abi_namespace:?};"
    )
    .unwrap();
    output.push_str(
        "pub(crate) const GENERATED_INTRINSIC_CRATES: &[&str] = &[\n    \"cuda_intrinsics\",\n    \"cuda-intrinsics\",\n];\n\npub(crate) const GENERATED_INTRINSIC_CANONICAL_PATHS: &[&str] = &[\n",
    );
    for record in &catalog.intrinsics {
        writeln!(output, "    {:?},", record.rust.canonical_path).unwrap();
    }
    output.push_str("];\n\npub(crate) const GENERATED_INTRINSIC_PUBLIC_PATHS: &[&str] = &[\n");
    for record in &catalog.intrinsics {
        writeln!(output, "    {:?},", record.rust.public_path).unwrap();
    }
    output.push_str("];\n\npub(crate) fn is_generated_intrinsic_crate(crate_name: &str) -> bool {\n    GENERATED_INTRINSIC_CRATES.contains(&crate_name)\n}\n\npub(crate) fn is_generated_intrinsic_canonical_path(path: &str) -> bool {\n    matches!(path,\n");
    let canonical_paths: Vec<_> = catalog
        .intrinsics
        .iter()
        .map(|record| record.rust.canonical_path.as_str())
        .collect();
    render_string_patterns(&mut output, &canonical_paths, "        ");
    output.push_str("    )\n}\n\npub(crate) fn is_generated_intrinsic_compatibility_path(path: &str) -> bool {\n");
    let compatibility_paths: Vec<_> = catalog
        .intrinsics
        .iter()
        .flat_map(|record| record.rust.compatibility_paths.iter().map(String::as_str))
        .collect();
    if compatibility_paths.is_empty() {
        output.push_str("    let _ = path;\n    false\n");
    } else {
        output.push_str("    matches!(path,\n");
        render_string_patterns(&mut output, &compatibility_paths, "        ");
        output.push_str("    )\n");
    }
    output.push_str(
        "}\n\npub(crate) fn is_generated_intrinsic_placeholder(crate_name: &str, path: &str) -> bool {\n    if is_generated_intrinsic_crate(crate_name) {\n        is_generated_intrinsic_canonical_path(path)\n    } else if matches!(crate_name, \"cuda_device\" | \"cuda-device\") {\n        is_generated_intrinsic_compatibility_path(path)\n    } else {\n        false\n    }\n}\n",
    );
    output
}

fn generated_selection_constraints(selection: &CatalogSelection) -> String {
    let address_space = match selection.constraints.address_space {
        None => "None",
        Some(ImportedAddressSpace::Generic) => "Some(GeneratedSelectionAddressSpace::Generic)",
        Some(ImportedAddressSpace::Shared) => "Some(GeneratedSelectionAddressSpace::Shared)",
    };
    let immediate_bindings = selection
        .constraints
        .immediate_bindings
        .iter()
        .map(|binding| {
            format!(
                "GeneratedImmediateBinding {{ argument_index: {}, value: {} }}",
                binding.argument_index, binding.value
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "GeneratedSelectionConstraints {{ address_space: {address_space}, immediate_bindings: &[{immediate_bindings}] }}"
    )
}

fn generated_selection_alternatives(selections: &[CatalogSelection]) -> String {
    let mut output = String::from("&[");
    for selection in selections {
        write!(
            output,
            "GeneratedSelectionAlternative {{ source_record: {:?}, asm: {:?}, predicates: &{:?}, constraints: {} }},",
            selection.source_record,
            selection.asm,
            selection.predicates,
            generated_selection_constraints(selection),
        )
        .unwrap();
    }
    output.push(']');
    output
}

fn generated_intrinsic_variant(record: &CatalogIntrinsic) -> String {
    if let Some(packed) = &record.packed_atomic {
        let format = match packed.format {
            PackedAtomicFormat::F16x2 => "GeneratedPackedAtomicFormat::F16x2",
            PackedAtomicFormat::Bf16x2 => "GeneratedPackedAtomicFormat::Bf16x2",
        };
        return format!("GeneratedIntrinsicVariant::PackedAtomic {{ format: {format} }}");
    }
    let Some(ldmatrix) = &record.ldmatrix else {
        return "GeneratedIntrinsicVariant::Scalar".to_owned();
    };
    let variant = &ldmatrix.variant;
    let shape = match variant.shape {
        LdmatrixShape::M8n8 => "GeneratedLdmatrixShape::M8n8",
    };
    let multiplicity = match variant.multiplicity {
        LdmatrixMultiplicity::X1 => "GeneratedLdmatrixMultiplicity::X1",
        LdmatrixMultiplicity::X2 => "GeneratedLdmatrixMultiplicity::X2",
        LdmatrixMultiplicity::X4 => "GeneratedLdmatrixMultiplicity::X4",
    };
    let layout = match variant.layout {
        LdmatrixLayout::Normal => "GeneratedLdmatrixLayout::Normal",
        LdmatrixLayout::Transposed => "GeneratedLdmatrixLayout::Transposed",
    };
    format!(
        "GeneratedIntrinsicVariant::Ldmatrix {{ shape: {shape}, multiplicity: {multiplicity}, layout: {layout} }}"
    )
}

fn generated_backend_requirements(record: &CatalogIntrinsic) -> String {
    let requirements = record
        .backend_lowerings
        .iter()
        .map(|lowering| {
            let backend = match lowering.backend {
                IntrinsicBackend::LlvmNvptx => "GeneratedIntrinsicBackend::LlvmNvptx",
                IntrinsicBackend::LibNvvm => "GeneratedIntrinsicBackend::LibNvvm",
            };
            format!(
                "GeneratedBackendRequirement {{ backend: {backend}, requirement: GeneratedTargetRequirement {{ minimum_ptx: GeneratedPtxVersion::from_encoded({}), hardware: {} }} }}",
                lowering.target.minimum_ptx.encoded(),
                generated_hardware_target(&lowering.target.hardware),
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("&[{requirements}]")
}

fn render_targets(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Generated target requirements and separately imported LLVM/selection facts.\n\npub const GENERATED_INTRINSIC_MARKER_ATTR: &str = \"cuda_oxide_intrinsic_marker\";\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]\npub struct GeneratedPtxVersion(u16);\nimpl GeneratedPtxVersion {\n    pub const fn from_encoded(encoded: u16) -> Self { Self(encoded) }\n    pub const fn encoded(self) -> u16 { self.0 }\n    pub const fn major(self) -> u16 { self.0 / 10 }\n    pub const fn minor(self) -> u16 { self.0 % 10 }\n}\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedHardwareAlternative { MinimumSm(u16), ExactArchitecture(u16), FamilyTarget(u16) }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedHardwareTarget { All, AnyOf(&'static [GeneratedHardwareAlternative]) }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedTargetRequirement { pub minimum_ptx: GeneratedPtxVersion, pub hardware: GeneratedHardwareTarget }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedIntrinsicBackend { LlvmNvptx, LibNvvm }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedBackendRequirement { pub backend: GeneratedIntrinsicBackend, pub requirement: GeneratedTargetRequirement }\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedSelectionAddressSpace { Generic, Shared }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedImmediateBinding { pub argument_index: usize, pub value: i64 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedSelectionConstraints { pub address_space: Option<GeneratedSelectionAddressSpace>, pub immediate_bindings: &'static [GeneratedImmediateBinding] }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedSelectionAlternative { pub source_record: &'static str, pub asm: &'static str, pub predicates: &'static [&'static str], pub constraints: GeneratedSelectionConstraints }\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedIntrinsicRange { pub lower: &'static str, pub upper_exclusive: &'static str }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedLlvmFacts { pub properties: &'static [&'static str], pub result_no_undef: bool, pub result_range: Option<GeneratedIntrinsicRange> }\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedLdmatrixShape { M8n8 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedLdmatrixMultiplicity { X1, X2, X4 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedLdmatrixLayout { Normal, Transposed }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedPackedAtomicFormat { F16x2, Bf16x2 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedIntrinsicVariant {\n    Scalar,\n    Ldmatrix { shape: GeneratedLdmatrixShape, multiplicity: GeneratedLdmatrixMultiplicity, layout: GeneratedLdmatrixLayout },\n    PackedAtomic { format: GeneratedPackedAtomicFormat },\n}\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedIntrinsicTarget {\n    pub marker: &'static str,\n    pub id: &'static str,\n    pub abi_id: &'static str,\n    pub dialect_op: &'static str,\n    pub variant: GeneratedIntrinsicVariant,\n    pub requirement: GeneratedTargetRequirement,\n    pub backend_requirements: &'static [GeneratedBackendRequirement],\n    pub selections: &'static [GeneratedSelectionAlternative],\n    pub llvm: Option<GeneratedLlvmFacts>,\n}\n\nimpl GeneratedIntrinsicTarget {\n    pub fn requirement_for_backend(&self, backend: GeneratedIntrinsicBackend) -> GeneratedTargetRequirement {\n        self.backend_requirements.iter().find(|entry| entry.backend == backend).map(|entry| entry.requirement).unwrap_or(self.requirement)\n    }\n}\n\npub const GENERATED_INTRINSIC_TARGETS: &[GeneratedIntrinsicTarget] = &[\n",
    );
    for record in &catalog.intrinsics {
        let llvm_facts = match &record.llvm {
            Some(llvm) => {
                let result_range = match &llvm.result_facts.range {
                    Some(range) => format!(
                        "Some(GeneratedIntrinsicRange {{ lower: {:?}, upper_exclusive: {:?} }})",
                        range.lower, range.upper_exclusive
                    ),
                    None => "None".to_owned(),
                };
                format!(
                    "Some(GeneratedLlvmFacts {{ properties: &{:?}, result_no_undef: {}, result_range: {} }})",
                    llvm.properties, llvm.result_facts.no_undef, result_range
                )
            }
            None => "None".to_owned(),
        };
        writeln!(
            output,
            "    GeneratedIntrinsicTarget {{ marker: {:?}, id: {:?}, abi_id: {:?}, dialect_op: {:?}, variant: {}, requirement: GeneratedTargetRequirement {{ minimum_ptx: GeneratedPtxVersion::from_encoded({}), hardware: {} }}, backend_requirements: {}, selections: {}, llvm: {} }},",
            intrinsic_marker(catalog, record),
            record.id,
            record.rust.abi_id,
            record.dialect.op_name,
            generated_intrinsic_variant(record),
            record.target.minimum_ptx.encoded(),
            generated_hardware_target(&record.target.hardware),
            generated_backend_requirements(record),
            generated_selection_alternatives(&record.selections),
            llvm_facts,
        )
        .unwrap();
    }
    output.push_str(
        "];\n\npub fn generated_intrinsic_target_by_marker(marker: &str) -> Option<&'static GeneratedIntrinsicTarget> {\n    GENERATED_INTRINSIC_TARGETS.iter().find(|target| target.marker == marker)\n}\n\npub fn generated_intrinsic_targets_by_op_name(op_name: &str) -> impl Iterator<Item = &'static GeneratedIntrinsicTarget> + '_ {\n    GENERATED_INTRINSIC_TARGETS.iter().filter(move |target| target.dialect_op == op_name)\n}\n\npub fn generated_intrinsic_target_by_op_name(op_name: &str) -> Option<&'static GeneratedIntrinsicTarget> {\n    generated_intrinsic_targets_by_op_name(op_name).next()\n}\n\npub fn generated_intrinsic_target(op_name: &str) -> Option<&'static GeneratedIntrinsicTarget> {\n    generated_intrinsic_target_by_op_name(op_name)\n}\n\npub fn generated_intrinsic_operation_matches(ctx: &Context, target: &GeneratedIntrinsicTarget, operation: Ptr<Operation>) -> bool {\n    match target.variant {\n        GeneratedIntrinsicVariant::Scalar => true,\n        GeneratedIntrinsicVariant::Ldmatrix { shape, multiplicity, layout } => {\n            let Some(op) = Operation::get_op::<LdmatrixOp>(operation, ctx) else { return false; };\n            let shape_matches = matches!(shape, GeneratedLdmatrixShape::M8n8) && op.get_attr_nvvm_ldmatrix_shape(ctx).as_deref() == Some(&LdmatrixShapeAttr::M8n8);\n            let multiplicity_matches = match multiplicity {\n                GeneratedLdmatrixMultiplicity::X1 => op.get_attr_nvvm_ldmatrix_multiplicity(ctx).as_deref() == Some(&LdmatrixMultiplicityAttr::X1),\n                GeneratedLdmatrixMultiplicity::X2 => op.get_attr_nvvm_ldmatrix_multiplicity(ctx).as_deref() == Some(&LdmatrixMultiplicityAttr::X2),\n                GeneratedLdmatrixMultiplicity::X4 => op.get_attr_nvvm_ldmatrix_multiplicity(ctx).as_deref() == Some(&LdmatrixMultiplicityAttr::X4),\n            };\n            let layout_matches = match layout {\n                GeneratedLdmatrixLayout::Normal => op.get_attr_nvvm_ldmatrix_layout(ctx).as_deref() == Some(&LdmatrixLayoutAttr::Normal),\n                GeneratedLdmatrixLayout::Transposed => op.get_attr_nvvm_ldmatrix_layout(ctx).as_deref() == Some(&LdmatrixLayoutAttr::Transposed),\n            };\n            shape_matches && multiplicity_matches && layout_matches\n                && op.get_attr_nvvm_ldmatrix_element(ctx).as_deref() == Some(&LdmatrixElementAttr::B16)\n                && op.get_attr_nvvm_ldmatrix_state_space(ctx).as_deref() == Some(&LdmatrixStateSpaceAttr::Shared)\n        }\n        GeneratedIntrinsicVariant::PackedAtomic { format } => {\n            let Some(op) = Operation::get_op::<PackedAtomicAddOp>(operation, ctx) else { return false; };\n            let format_matches = match format {\n                GeneratedPackedAtomicFormat::F16x2 => op.get_attr_nvvm_packed_atomic_format(ctx).as_deref() == Some(&PackedAtomicFormatAttr::F16x2),\n                GeneratedPackedAtomicFormat::Bf16x2 => op.get_attr_nvvm_packed_atomic_format(ctx).as_deref() == Some(&PackedAtomicFormatAttr::Bf16x2),\n            };\n            format_matches\n                && op.get_attr_nvvm_packed_atomic_state_space(ctx).as_deref() == Some(&PackedAtomicStateSpaceAttr::Global)\n                && op.get_attr_nvvm_packed_atomic_ordering(ctx).as_deref() == Some(&PackedAtomicOrderingAttr::Relaxed)\n                && op.get_attr_nvvm_packed_atomic_scope(ctx).as_deref() == Some(&PackedAtomicScopeAttr::Gpu)\n                && op.get_attr_nvvm_packed_atomic_rounding(ctx).as_deref() == Some(&PackedAtomicRoundingAttr::Rn)\n                && op.get_attr_nvvm_packed_atomic_subnormal(ctx).as_deref() == Some(&PackedAtomicSubnormalAttr::NoFtz)\n                && op.get_attr_nvvm_packed_atomic_atomicity(ctx).as_deref() == Some(&PackedAtomicAtomicityAttr::PerElement)\n        }\n    }\n}\n",
    );
    output.push_str(
        "\nuse dialect_nvvm::ops::{LdmatrixElementAttr, LdmatrixLayoutAttr, LdmatrixMultiplicityAttr, LdmatrixOp, LdmatrixShapeAttr, LdmatrixStateSpaceAttr, PackedAtomicAddOp, PackedAtomicAtomicityAttr, PackedAtomicFormatAttr, PackedAtomicOrderingAttr, PackedAtomicRoundingAttr, PackedAtomicScopeAttr, PackedAtomicStateSpaceAttr, PackedAtomicSubnormalAttr};\nuse pliron::{context::{Context, Ptr}, operation::Operation};\n",
    );
    output.push_str(
        "\n#[cfg(test)]\nmod tests {\n    use super::*;\n    use std::collections::BTreeSet;\n\n    #[test]\n    fn generated_target_table_is_unique_and_lookup_is_complete() {\n        let mut ids = BTreeSet::new();\n        let mut markers = BTreeSet::new();\n        for target in GENERATED_INTRINSIC_TARGETS {\n            assert!(ids.insert(target.id), \"duplicate generated intrinsic ID {}\", target.id);\n            assert!(markers.insert(target.marker), \"duplicate generated marker {}\", target.marker);\n            assert_eq!(generated_intrinsic_target_by_marker(target.marker), Some(target));\n            assert!(generated_intrinsic_targets_by_op_name(target.dialect_op).any(|candidate| candidate == target));\n        }\n",
    );
    for record in &catalog.intrinsics {
        writeln!(
            output,
            "        let target = generated_intrinsic_target_by_marker({:?}).unwrap();",
            intrinsic_marker(catalog, record)
        )
        .unwrap();
        writeln!(
            output,
            "        assert_eq!(target.id, {:?});\n        assert_eq!(target.abi_id, {:?});\n        assert_eq!(target.dialect_op, {:?});\n        assert_eq!(target.variant, {});\n        assert_eq!(target.requirement.minimum_ptx.encoded(), {});\n        assert_eq!(target.requirement.hardware, {});\n        assert_eq!(target.backend_requirements, {});\n        assert_eq!(target.selections, {});",
            record.id,
            record.rust.abi_id,
            record.dialect.op_name,
            generated_intrinsic_variant(record),
            record.target.minimum_ptx.encoded(),
            generated_hardware_target(&record.target.hardware),
            generated_backend_requirements(record),
            generated_selection_alternatives(&record.selections),
        )
        .unwrap();
        match &record.llvm {
            Some(llvm) => {
                writeln!(
                    output,
                    "        assert_eq!(target.llvm.unwrap().properties, &{:?} as &[&str]);",
                    llvm.properties
                )
                .unwrap();
            }
            None => output.push_str("        assert!(target.llvm.is_none());\n"),
        }
    }
    output.push_str("        assert!(generated_intrinsic_target_by_marker(\"v1:i9999\").is_none());\n    }\n}\n");
    output
}

pub(crate) fn render_probe(catalog: &CatalogFile, record: &CatalogIntrinsic, hash: &str) -> String {
    let mut output = llvm_header(catalog, hash);
    output.push_str("target triple = \"nvptx64-nvidia-cuda\"\n\n");
    if record.packed_alu.is_some() {
        let arity = record.rust.arguments.len();
        let parameters = (0..arity)
            .map(|index| format!("i32 %arg{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let arguments = (0..arity)
            .map(|index| format!("i32 %arg{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let operands = (0..=arity)
            .map(|index| format!("${index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let constraints = std::iter::once("=r")
            .chain(std::iter::repeat_n("r", arity))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(output, "define i32 @probe_{}({parameters}) {{", record.id).unwrap();
        writeln!(
            output,
            "  %result = call i32 asm \"{} {operands};\", \"{constraints}\"({arguments})",
            packed_alu_ptx_mnemonic(record)
        )
        .unwrap();
        output.push_str("  ret i32 %result\n}\n");
    } else if record.packed_conversion.is_some() {
        writeln!(
            output,
            "define i32 @probe_{}(float %low, float %high) {{",
            record.id
        )
        .unwrap();
        writeln!(
            output,
            "  %result = call i32 asm \"{} $0, $2, $1;\", \"=r,f,f\"(float %low, float %high)",
            packed_conversion_ptx_mnemonic(record)
        )
        .unwrap();
        output.push_str("  ret i32 %result\n}\n");
    } else if let Some(packed) = &record.packed_atomic {
        let format = match packed.format {
            PackedAtomicFormat::F16x2 => "f16x2",
            PackedAtomicFormat::Bf16x2 => "bf16x2",
        };
        writeln!(
            output,
            "define i32 @probe_{}(ptr %address, i32 %addend) {{",
            record.id
        )
        .unwrap();
        writeln!(
            output,
            "  %old = call i32 asm sideeffect \"atom.global.add.noftz.{format} $0, [$1], $2;\", \"=r,l,r,~{{memory}}\"(ptr %address, i32 %addend)"
        )
        .unwrap();
        output.push_str("  ret i32 %old\n}\n");
    } else if let Some(copy) = &record.cp_async_copy {
        let dynamic_source_size = copy.source_size == CpAsyncSourceSize::Runtime;
        let declaration_arguments = if dynamic_source_size {
            "ptr addrspace(3), ptr addrspace(1), i32"
        } else {
            "ptr addrspace(3), ptr addrspace(1)"
        };
        writeln!(
            output,
            "declare void @{}({declaration_arguments})",
            llvm(record).symbol
        )
        .unwrap();
        output.push('\n');
        if dynamic_source_size {
            writeln!(
                output,
                "define void @probe_{}_register(ptr %shared_generic, ptr %global_generic, i32 %source_size) {{",
                record.id
            )
            .unwrap();
            output.push_str(
                "  %shared = addrspacecast ptr %shared_generic to ptr addrspace(3)\n  %global = addrspacecast ptr %global_generic to ptr addrspace(1)\n",
            );
            writeln!(
                output,
                "  call void @{}(ptr addrspace(3) %shared, ptr addrspace(1) %global, i32 %source_size)",
                llvm(record).symbol
            )
            .unwrap();
            output.push_str("  ret void\n}\n");
            writeln!(
                output,
                "define void @probe_{}_immediate(ptr %shared_generic, ptr %global_generic) {{",
                record.id
            )
            .unwrap();
            output.push_str(
                "  %shared = addrspacecast ptr %shared_generic to ptr addrspace(3)\n  %global = addrspacecast ptr %global_generic to ptr addrspace(1)\n",
            );
            writeln!(
                output,
                "  call void @{}(ptr addrspace(3) %shared, ptr addrspace(1) %global, i32 3)",
                llvm(record).symbol
            )
            .unwrap();
            output.push_str("  ret void\n}\n");
        } else {
            writeln!(
                output,
                "define void @probe_{}(ptr %shared_generic, ptr %global_generic) {{",
                record.id
            )
            .unwrap();
            output.push_str(
                "  %shared = addrspacecast ptr %shared_generic to ptr addrspace(3)\n  %global = addrspacecast ptr %global_generic to ptr addrspace(1)\n",
            );
            writeln!(
                output,
                "  call void @{}(ptr addrspace(3) %shared, ptr addrspace(1) %global)",
                llvm(record).symbol
            )
            .unwrap();
            output.push_str("  ret void\n}\n");
        }
    } else if let Some(control) = &record.cp_async_control {
        let has_immediate = control.operation == CpAsyncControlOperation::WaitGroup;
        let declaration_arguments = if has_immediate { "i32" } else { "" };
        writeln!(
            output,
            "declare void @{}({declaration_arguments})",
            llvm(record).symbol
        )
        .unwrap();
        output.push('\n');
        writeln!(output, "define void @probe_{}() {{", record.id).unwrap();
        if has_immediate {
            writeln!(output, "  call void @{}(i32 3)", llvm(record).symbol).unwrap();
        } else {
            writeln!(output, "  call void @{}()", llvm(record).symbol).unwrap();
        }
        output.push_str("  ret void\n}\n");
    } else if record.family == "sync" {
        writeln!(output, "declare void @{}(i32)", llvm(record).symbol).unwrap();
        output.push('\n');
        writeln!(output, "define void @probe_{}() {{", record.id).unwrap();
        writeln!(output, "  call void @{}(i32 0)", llvm(record).symbol).unwrap();
        output.push_str("  ret void\n}\n");
    } else if let Some(warp_barrier) = &record.warp_barrier {
        debug_assert_eq!(warp_barrier.adapter, WarpBarrierAdapter::DirectMemberMask);
        writeln!(output, "declare void @{}(i32)", llvm(record).symbol).unwrap();
        output.push('\n');
        writeln!(
            output,
            "define void @probe_{}(i32 %member_mask) {{",
            record.id
        )
        .unwrap();
        writeln!(
            output,
            "  call void @{}(i32 %member_mask)",
            llvm(record).symbol
        )
        .unwrap();
        output.push_str("  ret void\n}\n");
        writeln!(output, "define void @probe_{}_immediate() {{", record.id).unwrap();
        writeln!(output, "  call void @{}(i32 -1)", llvm(record).symbol).unwrap();
        output.push_str("  ret void\n}\n");
    } else if let Some(dot) = &record.dot_product {
        match dot.adapter {
            DotProductAdapter::DirectThreeOperands => {
                writeln!(
                    output,
                    "declare i32 @{}(i32, i32, i32)",
                    llvm(record).symbol
                )
                .unwrap();
                output.push('\n');
                writeln!(
                    output,
                    "define i32 @probe_{}(i32 %a, i32 %b, i32 %c) {{",
                    record.id
                )
                .unwrap();
                writeln!(
                    output,
                    "  %result = call i32 @{}(i32 %a, i32 %b, i32 %c)",
                    llvm(record).symbol
                )
                .unwrap();
            }
            DotProductAdapter::InsertLowHalfFalse => {
                writeln!(
                    output,
                    "declare i32 @{}(i32, i32, i1, i32)",
                    llvm(record).symbol
                )
                .unwrap();
                output.push('\n');
                writeln!(
                    output,
                    "define i32 @probe_{}(i32 %a, i32 %b, i32 %c) {{",
                    record.id
                )
                .unwrap();
                writeln!(
                    output,
                    "  %result = call i32 @{}(i32 %a, i32 %b, i1 false, i32 %c)",
                    llvm(record).symbol
                )
                .unwrap();
            }
        }
        output.push_str("  ret i32 %result\n}\n");
    } else if let Some(vote) = &record.vote {
        debug_assert_eq!(vote.adapter, VoteAdapter::DirectMaskPredicate);
        let result_ty = match vote.mode {
            VoteMode::All | VoteMode::Any | VoteMode::Uni => "i1",
            VoteMode::Ballot => "i32",
        };
        writeln!(
            output,
            "declare {result_ty} @{}(i32, i1)",
            llvm(record).symbol
        )
        .unwrap();
        output.push('\n');
        writeln!(
            output,
            "define {result_ty} @probe_{}(i32 %member_mask, i1 %predicate) {{",
            record.id
        )
        .unwrap();
        writeln!(
            output,
            "  %result = call {result_ty} @{}(i32 %member_mask, i1 %predicate)",
            llvm(record).symbol
        )
        .unwrap();
        writeln!(output, "  ret {result_ty} %result\n}}").unwrap();
        writeln!(
            output,
            "define {result_ty} @probe_{}_immediate(i1 %predicate) {{",
            record.id
        )
        .unwrap();
        writeln!(
            output,
            "  %result = call {result_ty} @{}(i32 -1, i1 %predicate)",
            llvm(record).symbol
        )
        .unwrap();
        writeln!(output, "  ret {result_ty} %result\n}}").unwrap();
    } else if let Some(warp_match) = &record.warp_match {
        let value_ty = format!("i{}", warp_match.value_width.bits());
        let result_ty = match warp_match.mode {
            WarpMatchMode::Any => "i32".to_owned(),
            WarpMatchMode::All => "{ i32, i1 }".to_owned(),
        };
        writeln!(
            output,
            "declare {result_ty} @{}(i32, {value_ty})",
            llvm(record).symbol
        )
        .unwrap();
        output.push('\n');
        let forms = [
            (
                "rr",
                "i32 %member_mask, ",
                "i32 %member_mask",
                format!("{value_ty} %value"),
            ),
            ("ri", "", "i32 -1", format!("{value_ty} %value")),
            (
                "ir",
                "i32 %member_mask",
                "i32 %member_mask",
                format!("{value_ty} 7"),
            ),
            ("ii", "", "i32 -1", format!("{value_ty} 7")),
        ];
        for (suffix, first_parameter, mask, value) in forms {
            let parameters = match suffix {
                "rr" => format!("{first_parameter}{value_ty} %value"),
                "ri" => format!("{value_ty} %value"),
                "ir" => first_parameter.to_owned(),
                "ii" => String::new(),
                _ => unreachable!(),
            };
            writeln!(
                output,
                "define {result_ty} @probe_{}_{suffix}({parameters}) {{",
                record.id
            )
            .unwrap();
            writeln!(
                output,
                "  %result = call {result_ty} @{}({mask}, {value})",
                llvm(record).symbol
            )
            .unwrap();
            writeln!(output, "  ret {result_ty} %result\n}}").unwrap();
        }
    } else if let Some(warp_shuffle) = &record.warp_shuffle {
        if warp_shuffle.value_kind == WarpShuffleValueKind::I64 {
            debug_assert_eq!(
                warp_shuffle.adapter,
                WarpShuffleAdapter::MaskValueLaneOrDeltaSplitI64LowHighB32InsertClampReassemble
            );
            let mode = match warp_shuffle.mode {
                WarpShuffleMode::Idx => "idx",
                WarpShuffleMode::Bfly => "bfly",
                WarpShuffleMode::Down => "down",
                WarpShuffleMode::Up => "up",
            };
            let asm = format!(
                "{{ .reg .b32 lo; .reg .b32 hi; mov.b64 {{lo, hi}}, $1; shfl.sync.{mode}.b32 lo, lo, $2, {}, $3; shfl.sync.{mode}.b32 hi, hi, $2, {}, $3; mov.b64 $0, {{lo, hi}}; }}",
                warp_shuffle.clamp, warp_shuffle.clamp
            );
            writeln!(
                output,
                "define i64 @probe_{}(i32 %member_mask, i64 %value, i32 %lane) #0 {{",
                record.id
            )
            .unwrap();
            writeln!(
                output,
                "  %result = call i64 asm sideeffect {asm:?}, \"=l,l,r,r\"(i64 %value, i32 %lane, i32 %member_mask) #0"
            )
            .unwrap();
            output.push_str("  ret i64 %result\n}\n\nattributes #0 = { convergent }\n");
        } else {
            debug_assert_eq!(
                warp_shuffle.adapter,
                WarpShuffleAdapter::MaskValueLaneOrDeltaInsertClamp
            );
            let value_ty = match warp_shuffle.value_kind {
                WarpShuffleValueKind::I32 => "i32",
                WarpShuffleValueKind::F32 => "float",
                WarpShuffleValueKind::I64 => unreachable!(),
            };
            writeln!(
                output,
                "declare {value_ty} @{}(i32, {value_ty}, i32, i32)",
                llvm(record).symbol
            )
            .unwrap();
            output.push('\n');
            let forms = [
                (
                    "rr",
                    format!("i32 %member_mask, {value_ty} %value, i32 %lane"),
                    "i32 %member_mask",
                    "i32 %lane",
                ),
                (
                    "ri",
                    format!("{value_ty} %value, i32 %lane"),
                    "i32 -1",
                    "i32 %lane",
                ),
                (
                    "ir",
                    format!("i32 %member_mask, {value_ty} %value"),
                    "i32 %member_mask",
                    "i32 1",
                ),
                ("ii", format!("{value_ty} %value"), "i32 -1", "i32 1"),
            ];
            for (suffix, parameters, member_mask, lane) in forms {
                writeln!(
                    output,
                    "define {value_ty} @probe_{}_{suffix}({parameters}) {{",
                    record.id
                )
                .unwrap();
                writeln!(
                    output,
                    "  %result = call {value_ty} @{}({member_mask}, {value_ty} %value, {lane}, i32 {})",
                    llvm(record).symbol,
                    warp_shuffle.clamp,
                )
                .unwrap();
                writeln!(output, "  ret {value_ty} %result\n}}").unwrap();
            }
        }
    } else if let Some(redux) = &record.redux {
        debug_assert_eq!(redux.adapter, ReduxAdapter::MaskValueToSourceMemberMask);
        writeln!(output, "declare i32 @{}(i32, i32)", llvm(record).symbol).unwrap();
        output.push('\n');
        writeln!(
            output,
            "define i32 @probe_{}(i32 %member_mask, i32 %value) {{",
            record.id
        )
        .unwrap();
        writeln!(
            output,
            "  %result = call i32 @{}(i32 %value, i32 %member_mask)",
            llvm(record).symbol
        )
        .unwrap();
        output.push_str("  ret i32 %result\n}\n");
    } else if let Some(ldmatrix) = &record.ldmatrix {
        let register_count = ldmatrix.variant.multiplicity.register_count();
        let result_ty = if register_count == 1 {
            "i32".to_owned()
        } else {
            format!(
                "{{ {} }}",
                std::iter::repeat_n("i32", register_count)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let symbol = record
            .llvm
            .as_ref()
            .expect("ldmatrix LLVM facts")
            .resolved_symbol
            .as_ref()
            .expect("ldmatrix resolved symbol");
        writeln!(output, "declare {result_ty} @{symbol}(ptr addrspace(3))").unwrap();
        output.push('\n');
        writeln!(
            output,
            "define {result_ty} @probe_{}(ptr %generic) {{",
            record.id
        )
        .unwrap();
        output.push_str("  %shared = addrspacecast ptr %generic to ptr addrspace(3)\n");
        writeln!(
            output,
            "  %result = call {result_ty} @{symbol}(ptr addrspace(3) %shared)"
        )
        .unwrap();
        writeln!(output, "  ret {result_ty} %result\n}}\n").unwrap();
    } else if let Some(width) = record.scalar_width() {
        writeln!(output, "declare i{width} @{}()", llvm(record).symbol).unwrap();
        output.push('\n');
        writeln!(output, "define i{width} @probe_{}() {{", record.id).unwrap();
        writeln!(
            output,
            "  %result = call i{width} @{}()",
            llvm(record).symbol
        )
        .unwrap();
        writeln!(output, "  ret i{width} %result\n}}\n").unwrap();
    } else {
        unreachable!("generated intrinsic has no probe renderer: {}", record.id);
    }
    output
}

fn render_reference(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = markdown_header(catalog, hash);
    output.push_str(
        "# Generated Intrinsic Reference\n\nThis table is generated from the resolved catalog. Evidence stages below distinguish backend code generation, terminal validation, and GPU runtime execution; no runtime claim is made unless an executed stage is recorded.\n\n| Rust function | CUDA operation | Source | Effects | Availability | Backend evidence |\n|:--|:--|:--|:--|:--|:--|\n",
    );
    for record in &catalog.intrinsics {
        let safety = if record.rust.safe { "safe" } else { "unsafe" };
        writeln!(
            output,
            "| `{}` | `{}` | {} | {safety}; scope {}; {}; memory {}; convergent {} | PTX {} on {} ([PTX ISA {}, {}]({})) | {} on `{}`; expects `{}` (`{}`, SHA-256 `{}`) |",
            record.rust.public_path,
            record.dialect.op_name,
            source_label(record),
            record.semantics.execution_scope,
            if record.semantics.pure { "pure" } else { "impure" },
            record.semantics.memory,
            record.semantics.convergent,
            record.target.minimum_ptx,
            hardware_target_label(&record.target.hardware),
            record.target.ptx_isa_version,
            record.target.ptx_isa_section,
            record.target.ptx_isa_url,
            record.backend.status,
            record.backend.profile,
            record.expected_ptx,
            record.backend.version,
            record.backend.sha256,
        )
        .unwrap();
    }
    output.push_str("\n## Compiler identity and compatibility paths\n\n");
    for record in &catalog.intrinsics {
        writeln!(output, "- `{}` (`{}`)", record.id, record.rust.abi_id).unwrap();
        writeln!(
            output,
            "  - canonical compiler path: `{}`",
            record.rust.canonical_path
        )
        .unwrap();
        writeln!(
            output,
            "  - public source path: `{}`",
            record.rust.public_path
        )
        .unwrap();
        for path in &record.rust.compatibility_paths {
            writeln!(output, "  - compatibility compiler path: `{path}`").unwrap();
        }
    }
    output.push_str("\n## Packed-atomic contracts\n\n");
    for record in packed_atomics(catalog) {
        let packed = record.packed_atomic.as_ref().unwrap();
        let format = match packed.format {
            PackedAtomicFormat::F16x2 => "f16x2",
            PackedAtomicFormat::Bf16x2 => "bf16x2",
        };
        writeln!(
            output,
            "- `{}` (`{format}`): the native PTX instruction starts at `sm_{}`; cuda-oxide admits it from {}. Omitted `.sem` / `.scope` mean relaxed GPU scope. Each 16-bit element rounds to nearest-even and `.noftz` preserves subnormal inputs and results. The elements are atomic independently, so the returned `u32` contains old per-element values that may not form one coherent pair. The pointer must address four writable, four-byte-aligned global bytes; do not mix whole-word or non-atomic overlapping access, and every racing atomic must use a mutually inclusive scope.",
            record.id,
            packed.native_minimum_sm,
            hardware_target_label(&record.target.hardware),
        )
        .unwrap();
    }
    output.push_str("\n## Redux contracts\n\n");
    for record in redux(catalog) {
        writeln!(
            output,
            "- `{}`: raw and dialect operands are `[member_mask, value]`, adapted to LLVM `(value, membermask)`. The executing lane must be named in the mask, and every non-exited named lane must execute the same instruction with the same qualifiers and mask.",
            record.id,
        )
        .unwrap();
    }
    output.push_str("\n## Packed-ALU contracts\n\n");
    for record in packed_alus(catalog) {
        let packed = record.packed_alu.as_ref().unwrap();
        let format = match packed.format {
            PackedAluFormat::Bf16x2 => "bf16x2",
            PackedAluFormat::F16x2 => "f16x2",
        };
        let backend_floors = record
            .backend_lowerings
            .iter()
            .map(|lowering| {
                let backend = match lowering.backend {
                    IntrinsicBackend::LlvmNvptx => "LLVM-NVPTX",
                    IntrinsicBackend::LibNvvm => "libNVVM",
                };
                format!(
                    "{backend} PTX {} on {}",
                    lowering.target.minimum_ptx,
                    hardware_target_label(&lowering.target.hardware),
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        writeln!(
            output,
            "- `{}` carries one packed `{format}` value in a `u32` and lowers to one pure `{}` instruction. The native instruction starts at PTX {} / `sm_{}`; cuda-oxide admits it from {}. Backend profile floors: {backend_floors}.",
            record.id,
            packed_alu_ptx_mnemonic(record),
            record.target.minimum_ptx,
            packed.native_minimum_sm,
            hardware_target_label(&record.target.hardware),
        )
        .unwrap();
    }
    output.push_str("\n## Packed-conversion contract\n\n");
    for record in packed_conversions(catalog) {
        let conversion = record.packed_conversion.as_ref().unwrap();
        let rounding = match conversion.rounding {
            PackedConversionRounding::NearestEven => "nearest-even",
            PackedConversionRounding::TowardZero => "toward-zero",
        };
        let saturation = match conversion.saturation {
            PackedConversionSaturation::None => "without saturation",
            PackedConversionSaturation::Relu => "with ReLU",
        };
        writeln!(
            output,
            "- `{}` converts two `f32` inputs to packed `{}` using {rounding} rounding {saturation}. It lowers to pure `{}` inline PTX. The first input becomes the low lane and the second becomes the high lane, so PTX prints the inputs in reverse order.",
            record.id,
            packed_conversion_destination(record),
            packed_conversion_ptx_mnemonic(record),
        )
        .unwrap();
    }
    output.push_str("\n## Warp vote contracts\n\n");
    for record in vote_intrinsics(catalog) {
        writeln!(
            output,
            "- `{}` keeps raw and dialect operands in `[member_mask, predicate]` order. The executing lane must be named in the mask, and every non-exited named lane must execute the same `vote.sync` instruction with the same qualifiers and mask. On `sm_6x` and earlier, all named lanes must execute in convergence and no unnamed lane may be active. Both immediate and register member masks are admitted.",
            record.id,
        )
        .unwrap();
    }
    output.push_str("\n## Active-mask contract\n\n");
    for record in active_masks(catalog) {
        writeln!(
            output,
            "- `{}` observes the lanes active at the instruction. LLVM uses the typed intrinsic; libNVVM uses reviewed convergent, side-effecting inline PTX because that backend does not select the intrinsic.",
            record.id,
        )
        .unwrap();
    }
    output.push_str("\n## Warp-match contracts\n\n");
    for record in warp_matches(catalog) {
        let adapter = match record.warp_match.as_ref().unwrap().adapter {
            WarpMatchAdapter::DirectMask => "returns LLVM's mask directly",
            WarpMatchAdapter::ProjectMaskDiscardPredicate => {
                "projects field 0 from LLVM's `{i32, i1}` result"
            }
        };
        writeln!(
            output,
            "- `{}` keeps operands in `[member_mask, value]` order and {adapter}. The executing lane must be named in the mask, and every non-exited named lane must execute the same `match.sync` operation with the same qualifiers and mask. All register/immediate value and mask forms are admitted.",
            record.id,
        )
        .unwrap();
    }
    output.push_str("\n## Warp-barrier contract\n\n");
    for record in warp_barriers(catalog) {
        writeln!(
            output,
            "- `{}` passes the 32-bit member mask directly to the typed LLVM intrinsic on both backends. The executing lane must be named in the mask, and every non-exited named lane must execute the same `bar.warp.sync` operation with the same mask. On `sm_6x` and earlier, all named lanes must execute the barrier in convergence, and no unnamed lane may be active when it executes. The barrier orders memory accesses among participating lanes. Both immediate and register masks are admitted.",
            record.id,
        )
        .unwrap();
    }
    output.push_str("\n## Warp-shuffle contracts\n\n");
    for record in warp_shuffles(catalog) {
        let shuffle = record.warp_shuffle.as_ref().unwrap();
        let lowering = match shuffle.value_kind {
            WarpShuffleValueKind::I32 | WarpShuffleValueKind::F32 => {
                "Register and immediate lane/mask forms are admitted."
            }
            WarpShuffleValueKind::I64 => {
                "One convergent, side-effecting inline-PTX block splits `i64`, performs low then high `b32` shuffles with the same register lane/mask, and reassembles it."
            }
        };
        writeln!(
            output,
            "- `{}` keeps raw and dialect operands in `[member_mask, value, lane_or_delta]` order and inserts clamp `{}` during lowering. The executing lane must be named in the mask, and every non-exited named lane must execute the same `shfl.sync` operation with the same qualifiers and mask. On `sm_6x` and earlier, all named lanes must execute in convergence and no unnamed lane may be active. A computed in-range source must be active and named; if PTX marks it out of range, the calling lane's input is copied. {lowering}",
            record.id,
            shuffle.clamp,
        )
        .unwrap();
    }
    output.push_str("\n## CTA synchronization contracts\n\n");
    for record in sync_intrinsics(catalog) {
        writeln!(
            output,
            "- `{}` inserts the fixed barrier ID `0`. Every active CTA thread must reach the same barrier; divergent use can deadlock the CTA.",
            record.id,
        )
        .unwrap();
    }
    output.push_str("\n## Imported LLVM properties and result facts\n\n");
    for record in &catalog.intrinsics {
        let Some(llvm) = &record.llvm else {
            writeln!(
                output,
                "- `{}`: PTX-native source; no LLVM record, symbol, properties, or selection facts.",
                record.id
            )
            .unwrap();
            continue;
        };
        let properties = record
            .llvm
            .as_ref()
            .unwrap()
            .properties
            .iter()
            .map(|property| format!("`{property}`"))
            .collect::<Vec<_>>()
            .join(", ");
        let range = record
            .llvm
            .as_ref()
            .unwrap()
            .result_facts
            .range
            .as_ref()
            .map(|range| format!("[{}, {})", range.lower, range.upper_exclusive))
            .unwrap_or_else(|| "none".to_owned());
        let selection_records = record
            .selections
            .iter()
            .map(|selection| format!("`{}`", selection.source_record))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .join(", ");
        let selection_predicates = record
            .selections
            .iter()
            .flat_map(|selection| selection.predicates.iter())
            .map(|predicate| format!("`{predicate}`"))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            output,
            "- `{}`: properties {}; result `NoUndef` {}; half-open result range `{}`; selection records {}; selection predicates {}",
            record.id,
            properties,
            llvm.result_facts.no_undef,
            range,
            selection_records,
            if selection_predicates.is_empty() {
                "none"
            } else {
                &selection_predicates
            }
        )
        .unwrap();
    }
    output.push_str("\n## Backend-specific lowering evidence\n\n");
    for record in &catalog.intrinsics {
        if record.backend_lowerings.is_empty() {
            continue;
        }
        let runtime = record
            .ldmatrix
            .as_ref()
            .map(|record| format!("{:?}", record.safety.runtime_validation).to_lowercase())
            .or_else(|| {
                record
                    .packed_atomic
                    .as_ref()
                    .map(|record| format!("{:?}", record.runtime_validation).to_lowercase())
            })
            .unwrap_or_else(|| "not recorded".to_owned());
        writeln!(output, "- `{}`: runtime `{runtime}`", record.id).unwrap();
        for lowering in &record.backend_lowerings {
            writeln!(
                output,
                "  - `{}` uses `{}` from profile `{}` at PTX {} / {}: status `{}` (`{}`, SHA-256 `{}`)",
                backend_label(lowering.backend),
                lowering_mechanism_label(lowering.mechanism),
                lowering.evidence_profile,
                lowering.target.minimum_ptx,
                hardware_target_label(&lowering.target.hardware),
                lowering.status,
                lowering.version,
                lowering.sha256,
            )
            .unwrap();
            for stage in &lowering.stages {
                let tool = match (
                    stage.tool_path.as_deref(),
                    stage.tool_version.as_deref(),
                    stage.tool_sha256.as_deref(),
                ) {
                    (Some(path), Some(version), Some(sha256)) => {
                        format!(" Tool `{path}` reports `{version}` (SHA-256 `{sha256}`).")
                    }
                    _ => String::new(),
                };
                let artifact = match stage.artifact_kind {
                    Some(EvidenceArtifactKind::Cubin) => " Artifact `cubin`.",
                    None => "",
                };
                writeln!(
                    output,
                    "    - {} on `{}`: `{}` — {}{}{}",
                    evidence_stage_label(stage.stage),
                    stage.targets.join(", "),
                    stage.outcome,
                    stage.detail,
                    tool,
                    artifact,
                )
                .unwrap();
            }
        }
    }
    output
}

fn render_compiler_path_patterns(output: &mut String, catalog: &CatalogFile, indent: &str) {
    let paths: Vec<_> = catalog
        .intrinsics
        .iter()
        .flat_map(|record| {
            std::iter::once(record.rust.canonical_path.as_str())
                .chain(record.rust.compatibility_paths.iter().map(String::as_str))
        })
        .collect();
    render_string_patterns(output, &paths, indent);
}

fn render_string_patterns(output: &mut String, values: &[&str], indent: &str) {
    for (index, value) in values.iter().enumerate() {
        if index == 0 {
            writeln!(output, "{indent}{value:?}").unwrap();
        } else {
            writeln!(output, "{indent}| {value:?}").unwrap();
        }
    }
}

fn render_inline_patterns(output: &mut String, values: &[&str]) {
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            output.push_str(" | ");
        }
        write!(output, "{value:?}").unwrap();
    }
}

#[allow(dead_code)]
fn modules(catalog: &CatalogFile) -> BTreeSet<&str> {
    catalog
        .intrinsics
        .iter()
        .map(|record| record.rust.module.as_str())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ImportedSelectionConstraints;
    use std::path::Path;

    #[test]
    fn selection_alternatives_keep_predicates_and_constraints_grouped() {
        let selections = vec![
            CatalogSelection {
                source_record: "SELECT_A".into(),
                asm: "op.a $dst;".into(),
                predicates: vec!["HasA".into(), "HasCommon".into()],
                constraints: ImportedSelectionConstraints {
                    address_space: Some(ImportedAddressSpace::Generic),
                    immediate_bindings: vec![],
                },
            },
            CatalogSelection {
                source_record: "SELECT_B".into(),
                asm: "op.b $dst;".into(),
                predicates: vec!["HasB".into()],
                constraints: ImportedSelectionConstraints {
                    address_space: Some(ImportedAddressSpace::Shared),
                    immediate_bindings: vec![],
                },
            },
        ];

        let rendered = generated_selection_alternatives(&selections);
        assert_eq!(rendered.matches("GeneratedSelectionAlternative").count(), 2);
        assert!(rendered.contains(
            "source_record: \"SELECT_A\", asm: \"op.a $dst;\", predicates: &[\"HasA\", \"HasCommon\"], constraints: GeneratedSelectionConstraints { address_space: Some(GeneratedSelectionAddressSpace::Generic), immediate_bindings: &[] }"
        ));
        assert!(rendered.contains(
            "source_record: \"SELECT_B\", asm: \"op.b $dst;\", predicates: &[\"HasB\"], constraints: GeneratedSelectionConstraints { address_space: Some(GeneratedSelectionAddressSpace::Shared), immediate_bindings: &[] }"
        ));
    }

    #[test]
    fn packed_alu_and_conversion_render_exact_pure_inline_ptx_adapters() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let catalog = crate::resolve::resolve(&repo_root).unwrap();
        validate_renderable(&catalog).unwrap();
        assert_eq!(packed_alus(&catalog).count(), 18);
        assert_eq!(packed_conversions(&catalog).count(), 6);

        let dialect = render_dialect_packed_alu(&catalog, "test-hash");
        for op in [
            "FmaBf16x2Op",
            "FmaReluBf16x2Op",
            "AddBf16x2Op",
            "SubBf16x2Op",
            "MulBf16x2Op",
            "MinBf16x2Op",
            "MaxBf16x2Op",
            "NegBf16x2Op",
            "AbsBf16x2Op",
            "FmaF16x2Op",
            "FmaReluF16x2Op",
            "AddF16x2Op",
            "SubF16x2Op",
            "MulF16x2Op",
            "MinF16x2Op",
            "MaxF16x2Op",
            "NegF16x2Op",
            "AbsF16x2Op",
        ] {
            assert!(dialect.contains(&format!("pub struct {op}")));
            assert!(dialect.contains(&format!("{op}::register(ctx)")));
        }
        let conversion_dialect = render_dialect_packed_conversion(&catalog, "test-hash");
        for op in [
            "CvtF32x2Bf16x2Op",
            "CvtF16x2F32Op",
            "CvtRzF16x2F32Op",
            "CvtRnReluF16x2F32Op",
            "CvtRnReluBf16x2F32Op",
            "CvtRzBf16x2F32Op",
        ] {
            assert!(conversion_dialect.contains(&format!("pub struct {op}")));
            assert!(conversion_dialect.contains(&format!("{op}::register(ctx)")));
        }
        assert!(conversion_dialect.contains("low f16 lane"));
        assert!(conversion_dialect.contains("low bf16 lane"));
        assert!(conversion_dialect.contains("vec![low, high]"));

        let importer = render_importer(&catalog, "test-hash");
        assert!(importer.contains("cuda_device::bf16x2::fma_bf16x2"));
        assert!(importer.contains("cuda_device::f16x2::fma_f16x2"));
        assert!(importer.contains("cuda_device::tcgen05::cvt_f32x2_bf16x2"));
        assert!(importer.contains("cuda_device::convert::cvt_f16x2_f32"));
        assert!(importer.contains("cuda_device::convert::cvt_rz_bf16x2_f32"));
        assert!(importer.contains("FmaBf16x2Op::build(ctx, arg0, arg1, arg2)"));
        assert!(importer.contains("CvtF32x2Bf16x2Op::build(ctx, arg0, arg1)"));

        let lowering = render_lowering(&catalog, "test-hash");
        for mnemonic in [
            "fma.rn.bf16x2",
            "fma.rn.relu.bf16x2",
            "add.rn.bf16x2",
            "sub.rn.bf16x2",
            "mul.rn.bf16x2",
            "min.bf16x2",
            "max.bf16x2",
            "neg.bf16x2",
            "abs.bf16x2",
            "fma.rn.f16x2",
            "fma.rn.relu.f16x2",
            "add.rn.f16x2",
            "sub.rn.f16x2",
            "mul.rn.f16x2",
            "min.f16x2",
            "max.f16x2",
            "neg.f16x2",
            "abs.f16x2",
        ] {
            assert!(lowering.contains(&format!(
                "convert_generated_packed_alu(ctx, rewriter, self.get_operation(), \"{mnemonic}\")"
            )));
        }
        for mnemonic in [
            "cvt.rn.bf16x2.f32",
            "cvt.rn.f16x2.f32",
            "cvt.rz.f16x2.f32",
            "cvt.rn.relu.f16x2.f32",
            "cvt.rn.relu.bf16x2.f32",
            "cvt.rz.bf16x2.f32",
        ] {
            assert!(lowering.contains(&format!(
                "convert_generated_packed_f32x2(ctx, rewriter, self.get_operation(), \"{mnemonic}\")"
            )));
        }

        for record in packed_alus(&catalog) {
            let probe = render_probe(&catalog, record, "test-hash");
            let constraints = std::iter::once("=r")
                .chain(std::iter::repeat_n("r", record.rust.arguments.len()))
                .collect::<Vec<_>>()
                .join(",");
            assert!(probe.contains(&format!("\", \"{constraints}\"")));
            assert!(!probe.contains("asm sideeffect"));
            assert!(!probe.contains("~{memory}"));
        }
        for conversion in packed_conversions(&catalog) {
            let probe = render_probe(&catalog, conversion, "test-hash");
            assert!(probe.contains(&format!(
                "asm \"{} $0, $2, $1;\", \"=r,f,f\"(float %low, float %high)",
                packed_conversion_ptx_mnemonic(conversion)
            )));
            assert!(!probe.contains("asm sideeffect"));
        }

        let raw = render_raw_abi(&catalog, "test-hash");
        assert!(raw.contains("pub fn i0062(_arg0: u32, _arg1: u32, _arg2: u32) -> u32"));
        assert!(raw.contains("pub fn i0071(_arg0: f32, _arg1: f32) -> u32"));
        assert!(raw.contains("pub fn i0072(_arg0: u32, _arg1: u32, _arg2: u32) -> u32"));
        assert!(raw.contains("pub fn i0085(_arg0: f32, _arg1: f32) -> u32"));
        assert!(!raw.contains("#[must_use]\n#[inline(never)]\npub fn i0062"));
        let f16_raw = raw.find("pub fn i0072").unwrap();
        assert!(raw[..f16_raw].ends_with("#[must_use]\n#[inline(never)]\n"));

        let compatibility =
            render_compat_packed_alu(&catalog, "test-hash", PackedAluFormat::Bf16x2);
        assert!(compatibility.contains("pub fn fma_bf16x2(arg0: u32, arg1: u32, arg2: u32)"));
        assert!(!compatibility.contains("fma_f16x2"));
        assert!(compatibility.contains("let _ = arg0;"));
        assert!(!compatibility.contains("let _ = (arg0);"));
        let compatibility = render_compat_packed_alu(&catalog, "test-hash", PackedAluFormat::F16x2);
        let f16_compat = compatibility.find("pub fn fma_f16x2").unwrap();
        assert!(compatibility[..f16_compat].ends_with("#[must_use]\n#[inline(never)]\n"));
        assert!(!compatibility.contains("fma_bf16x2"));
        let reference = render_reference(&catalog, "test-hash");
        assert!(reference.contains("`fma_f16x2` carries one packed `f16x2` value in a `u32`"));
        assert!(reference.contains(
            "native instruction starts at PTX 4.2 / `sm_53`; cuda-oxide admits it from sm_70+"
        ));
        assert!(reference.contains("LLVM-NVPTX PTX 6.0 on sm_70+"));
        assert!(reference.contains("libNVVM PTX 4.2 on sm_75+"));
        assert!(reference.contains(
            "`cvt_rn_relu_f16x2_f32` converts two `f32` inputs to packed `f16x2` using nearest-even rounding with ReLU"
        ));
        assert!(reference.contains("pure `cvt.rz.bf16x2.f32` inline PTX"));
        let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
        assert!(outputs.contains_key(&PathBuf::from("crates/cuda-device/src/generated/bf16x2.rs")));
        assert!(outputs.contains_key(&PathBuf::from("crates/cuda-device/src/generated/f16x2.rs")));
        assert!(outputs.contains_key(&PathBuf::from(
            "crates/cuda-device/src/generated/convert.rs"
        )));
        let compatibility = render_compat_packed_conversion(
            &catalog,
            "test-hash",
            "cuda_device::tcgen05::",
            "tcgen05",
            ("a", "b"),
        );
        assert!(compatibility.contains("pub fn cvt_f32x2_bf16x2(a: f32, b: f32) -> u32"));
        assert!(!compatibility.contains("cvt_f16x2_f32"));
        let compatibility = render_compat_packed_conversion(
            &catalog,
            "test-hash",
            "cuda_device::convert::",
            "convert",
            ("lo", "hi"),
        );
        assert!(compatibility.contains("pub fn cvt_f16x2_f32(lo: f32, hi: f32) -> u32"));
        assert!(compatibility.contains("pub fn cvt_rz_bf16x2_f32(lo: f32, hi: f32) -> u32"));
        assert!(!compatibility.contains("cvt_f32x2_bf16x2"));
    }

    #[test]
    fn ldmatrix_family_lowering_uses_one_attribute_dispatch_impl() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let catalog = crate::resolve::resolve(&repo_root).unwrap();

        let rendered = render_lowering(&catalog, "test-hash");
        assert_eq!(
            rendered
                .matches("impl MirToLlvmConversion for LdmatrixOp")
                .count(),
            1
        );
        assert!(rendered.contains("LdmatrixMultiplicityAttr::X4"));
        assert!(rendered.contains("LdmatrixMultiplicityAttr::X2"));
        assert!(rendered.contains("LdmatrixMultiplicityAttr::X1"));
        assert!(rendered.contains("LdmatrixLayoutAttr::Transposed"));
        assert!(rendered.contains("llvm_nvvm_ldmatrix_sync_aligned_m8n8_x2_trans_b16_p3"));
    }

    #[test]
    fn ldmatrix_x1_probe_keeps_its_pointer_operand() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let catalog = crate::resolve::resolve(&repo_root).unwrap();
        let record = catalog
            .intrinsics
            .iter()
            .find(|record| record.id == "ldmatrix_m8n8_x1_b16")
            .unwrap();

        let rendered = render_probe(&catalog, record, "test-hash");
        assert!(rendered.contains(
            "declare i32 @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.b16.p3(ptr addrspace(3))"
        ));
        assert!(rendered.contains("define i32 @probe_ldmatrix_m8n8_x1_b16(ptr %generic)"));
        assert!(!rendered.contains("@llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.b16.p3()"));
    }

    #[test]
    fn packed_atomic_family_uses_one_attribute_dispatch_impl() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let catalog = crate::resolve::resolve(&repo_root).unwrap();
        let rendered = render_lowering(&catalog, "test-hash");
        assert_eq!(
            rendered
                .matches("impl MirToLlvmConversion for PackedAtomicAddOp")
                .count(),
            1
        );
        assert_eq!(
            rendered.matches("PackedAtomicFormatAttr::F16x2)").count(),
            1
        );
        assert_eq!(
            rendered.matches("PackedAtomicFormatAttr::Bf16x2)").count(),
            1
        );
    }

    #[test]
    fn packed_atomic_raw_abi_is_unsafe_and_must_use() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let catalog = crate::resolve::resolve(&repo_root).unwrap();
        let rendered = render_raw_abi(&catalog, "test-hash");
        for abi_id in ["i0014", "i0015"] {
            let signature = format!("pub unsafe fn {abi_id}(_arg0: *mut u32, _arg1: u32) -> u32");
            let index = rendered.find(&signature).unwrap();
            assert!(rendered[..index].ends_with("#[must_use]\n#[inline(never)]\n"));
        }
    }

    #[test]
    fn packed_atomic_compatibility_preserves_paths_signature_and_safety() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let catalog = crate::resolve::resolve(&repo_root).unwrap();
        let rendered = render_compat_packed_atomic(&catalog, "test-hash");

        for name in ["atom_add_f16x2", "atom_add_bf16x2"] {
            let signature = format!("pub unsafe fn {name}(addr: *mut u32, val: u32) -> u32");
            let index = rendered.find(&signature).unwrap();
            assert!(rendered[..index].ends_with("#[must_use]\n#[inline(never)]\n"));
            assert!(rendered.contains(&format!(
                "unreachable!(\"{name} called outside CUDA kernel context\")"
            )));
        }
        assert!(rendered.contains("relaxed GPU-scope operation"));
        assert!(rendered.contains("low lane first"));
        assert!(rendered.contains("may not form one old 32-bit snapshot"));
        assert!(rendered.contains("Requires PTX 6.2 and `sm_70+`"));
        assert!(rendered.contains("Requires PTX 7.8 and `sm_90+`"));
        assert!(rendered.contains("four writable, four-byte-aligned bytes in global memory"));
        assert!(rendered.contains("whole-word atomic or non-atomic lane access"));
        assert!(rendered.contains("Racing atomics must use mutually inclusive scopes"));

        let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
        assert_eq!(
            outputs.get(&PathBuf::from("crates/cuda-device/src/generated/atomic.rs")),
            Some(&rendered)
        );
    }

    #[test]
    fn cp_async_rendering_preserves_compatibility_dispatch_and_backend_routes() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let catalog = crate::resolve::resolve(&repo_root).unwrap();
        validate_renderable(&catalog).unwrap();
        assert_eq!(cp_async_copies(&catalog).count(), 8);
        assert_eq!(cp_async_controls(&catalog).count(), 3);

        let compatibility = render_compat_cp_async_copy(&catalog, "test-hash");
        for signature in [
            "pub unsafe fn cp_async_ca_4(_shared_dst: *mut u32, _global_src: *const u32)",
            "pub unsafe fn cp_async_ca_8(_shared_dst: *mut u32, _global_src: *const u32)",
            "pub unsafe fn cp_async_ca_16(_shared_dst: *mut u32, _global_src: *const u32)",
            "pub unsafe fn cp_async_ca_zfill_4(_shared_dst: *mut u32, _global_src: *const u8, _src_size: u32)",
            "pub unsafe fn cp_async_ca_zfill_8(_shared_dst: *mut u32, _global_src: *const u8, _src_size: u32)",
            "pub unsafe fn cp_async_ca_zfill_16(_shared_dst: *mut u32, _global_src: *const u8, _src_size: u32)",
            "pub unsafe fn cp_async_cg_16(_shared_dst: *mut u32, _global_src: *const u32)",
            "pub unsafe fn cp_async_cg_zfill_16(_shared_dst: *mut u32, _global_src: *const u8, _src_size: u32)",
            "pub unsafe fn cp_async_commit_group()",
            "pub unsafe fn cp_async_wait_all()",
            "pub unsafe fn cp_async_wait_group(_max_pending: u32)",
        ] {
            assert!(compatibility.contains(signature));
        }

        let dialect = render_dialect_cp_async_copy(&catalog, "test-hash");
        let importer = render_importer(&catalog, "test-hash");
        let lowering = render_lowering(&catalog, "test-hash");
        let targets = render_targets(&catalog, "test-hash");
        for record in cp_async_copies(&catalog).chain(cp_async_controls(&catalog)) {
            assert!(dialect.contains(&format!("pub struct {}", record.dialect.op_type)));
            assert!(dialect.contains(&format!("{}::register(ctx)", record.dialect.op_type)));
            assert!(importer.contains(&record.rust.canonical_path));
            for path in &record.rust.compatibility_paths {
                assert!(importer.contains(path));
            }
            assert!(importer.contains(&intrinsic_marker(&catalog, record)));
            assert!(lowering.contains(&format!(
                "impl MirToLlvmConversion for {}",
                record.dialect.op_type
            )));
            assert!(lowering.contains(&record.llvm_identifier()));
            assert!(
                record
                    .backend_lowerings
                    .iter()
                    .any(|entry| entry.backend == IntrinsicBackend::LlvmNvptx)
            );
            assert!(
                record
                    .backend_lowerings
                    .iter()
                    .any(|entry| entry.backend == IntrinsicBackend::LibNvvm)
            );
            assert!(targets.contains(&format!("id: {:?}", record.id)));
        }
        assert_eq!(dialect.matches("::register(ctx);").count(), 11);
        assert!(lowering.contains("convert_generated_cp_async_copy"));
        assert!(lowering.contains("convert_generated_cp_async_control"));
    }

    #[test]
    fn redux_rendering_preserves_mask_first_api_and_source_first_llvm_order() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let catalog = crate::resolve::resolve(&repo_root).unwrap();
        assert_eq!(catalog.schema, crate::resolve::CATALOG_SCHEMA);
        validate_renderable(&catalog).unwrap();

        assert_eq!(redux(&catalog).count(), 8);
        let record = redux(&catalog).next().unwrap();
        assert_eq!(
            record.redux.as_ref().unwrap().adapter,
            ReduxAdapter::MaskValueToSourceMemberMask
        );

        let dialect = render_dialect_redux(&catalog, "test-hash");
        assert!(dialect.contains("name = \"nvvm.redux_sync_add\""));
        assert!(dialect.contains("name = \"nvvm.redux_sync_min\""));
        assert!(dialect.contains("vec![member_mask, value]"));
        assert!(dialect.contains("ReduxSyncAddOp::register(ctx)"));
        assert!(dialect.contains("Signedness::Signed"));

        let importer = render_importer(&catalog, "test-hash");
        assert!(importer.contains("cuda_device::warp::redux_sync_add"));
        assert!(importer.contains("let (member_mask, last_op)"));
        assert!(
            importer.contains("let reduction = ReduxSyncAddOp::build(ctx, member_mask, value)")
        );
        assert!(importer.contains("set_generated_intrinsic_marker(ctx, reduction, \"v1:i0017\")"));

        let lowering = render_lowering(&catalog, "test-hash");
        assert!(lowering.contains("impl MirToLlvmConversion for ReduxSyncAddOp"));
        assert!(
            lowering.contains("convert_redux(ctx, rewriter, self.get_operation(), operands_info")
        );
        assert!(lowering.contains("\"llvm_nvvm_redux_sync_add\""));

        let probe = render_probe(&catalog, record, "test-hash");
        assert!(probe.contains("define i32 @probe_redux_sync_add(i32 %member_mask, i32 %value)"));
        assert!(probe.contains("call i32 @llvm.nvvm.redux.sync.add(i32 %value, i32 %member_mask)"));

        let raw = render_raw_abi(&catalog, "test-hash");
        let signature = "pub unsafe fn i0017(_arg0: u32, _arg1: u32) -> u32";
        let index = raw.find(signature).unwrap();
        assert!(raw[..index].ends_with("#[must_use]\n#[inline(never)]\n"));
        assert!(raw.contains("pub unsafe fn i0019(_arg0: u32, _arg1: i32) -> i32"));
        assert!(raw.contains("pub unsafe fn i0024(_arg0: u32, _arg1: u32) -> u32"));
        assert!(raw.contains("The executing lane must be named in `mask`"));
    }

    #[test]
    fn vote_rendering_keeps_types_selection_pairs_and_raw_only_uni() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let catalog = crate::resolve::resolve(&repo_root).unwrap();
        validate_renderable(&catalog).unwrap();
        assert_eq!(vote_intrinsics(&catalog).count(), 4);

        let dialect = render_dialect_vote(&catalog, "test-hash");
        for op in [
            "VoteSyncAllOp",
            "VoteSyncAnyOp",
            "VoteSyncBallotOp",
            "VoteSyncUniOp",
        ] {
            assert!(dialect.contains(&format!("pub struct {op}")));
            assert!(dialect.contains(&format!("{op}::register(ctx)")));
        }
        assert!(dialect.contains("requires i32 member mask, i1 predicate, and i1 result"));
        assert!(dialect.contains("requires i32 member mask, i1 predicate, and i32 result"));

        let importer = render_importer(&catalog, "test-hash");
        assert!(importer.contains("cuda_device::warp::all_sync"));
        assert!(!importer.contains("cuda_device::warp::uni_sync"));
        assert!(importer.contains("let vote = VoteSyncUniOp::build(ctx, member_mask, predicate)"));
        assert!(importer.contains("set_generated_intrinsic_marker(ctx, vote, \"v1:i0043\")"));

        let lowering = render_lowering(&catalog, "test-hash");
        assert!(lowering.contains("impl MirToLlvmConversion for VoteSyncUniOp"));
        assert!(lowering.contains("\"llvm_nvvm_vote_uni_sync\""));
        assert!(
            lowering.contains("convert_vote(ctx, rewriter, self.get_operation(), operands_info")
        );

        let record = vote_intrinsics(&catalog)
            .find(|record| record.id == "ballot_sync")
            .unwrap();
        assert_eq!(record.selections.len(), 2);
        let probe = render_probe(&catalog, record, "test-hash");
        assert!(probe.contains("define i32 @probe_ballot_sync(i32 %member_mask, i1 %predicate)"));
        assert!(probe.contains("define i32 @probe_ballot_sync_immediate(i1 %predicate)"));
        assert!(probe.contains("i32 -1, i1 %predicate"));

        let raw = render_raw_abi(&catalog, "test-hash");
        for abi_id in ["i0040", "i0041", "i0042", "i0043"] {
            assert!(raw.contains(&format!("pub unsafe fn {abi_id}")));
        }
        assert!(raw.contains("Every non-exited lane named in `mask`"));
    }

    #[test]
    fn active_mask_and_warp_match_rendering_preserves_backend_and_abi_contracts() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let catalog = crate::resolve::resolve(&repo_root).unwrap();
        validate_renderable(&catalog).unwrap();
        assert_eq!(active_masks(&catalog).count(), 1);
        assert_eq!(warp_matches(&catalog).count(), 4);

        let dialect = render_dialect_active_mask(&catalog, "test-hash");
        assert!(dialect.contains("pub struct ActiveMaskOp"));
        assert!(dialect.contains("NOpdsInterface<0>, NResultsInterface<1>"));

        let match_dialect = render_dialect_warp_match(&catalog, "test-hash");
        for op in [
            "MatchAnySyncI32Op",
            "MatchAnySyncI64Op",
            "MatchAllSyncI32Op",
            "MatchAllSyncI64Op",
        ] {
            assert!(match_dialect.contains(&format!("pub struct {op}")));
            assert!(match_dialect.contains(&format!("{op}::register(ctx)")));
        }

        let lowering = render_lowering(&catalog, "test-hash");
        assert!(lowering.contains("IntrinsicBackend::LlvmNvptx => {"));
        assert!(lowering.contains("convert_active_mask(ctx, rewriter, op, operands_info)"));
        assert!(lowering.contains("IntrinsicBackend::LibNvvm =>"));
        assert!(lowering.contains("\"activemask.b32 $0;\""));
        assert!(lowering.contains("\"=r,~{memory}\""));
        assert!(lowering.contains("convert_match_any("));
        assert!(lowering.contains("convert_match_all("));
        assert!(lowering.contains("\"llvm_nvvm_match_all_sync_i64p\""));

        let match_all = warp_matches(&catalog)
            .find(|record| record.id == "match_all_sync")
            .unwrap();
        let probe = render_probe(&catalog, match_all, "test-hash");
        for suffix in ["rr", "ri", "ir", "ii"] {
            assert!(probe.contains(&format!("@probe_match_all_sync_{suffix}")));
        }
        assert!(probe.contains("declare { i32, i1 } @llvm.nvvm.match.all.sync.i32p"));

        let raw = render_raw_abi(&catalog, "test-hash");
        assert!(raw.contains("pub fn i0044() -> u32"));
        assert!(!raw.contains("pub unsafe fn i0044() -> u32"));
        assert!(raw.contains("pub unsafe fn i0045(_arg0: u32, _arg1: u32) -> u32"));
        assert!(raw.contains("pub unsafe fn i0046(_arg0: u32, _arg1: u64) -> u32"));
        assert!(raw.contains("pub unsafe fn i0047(_arg0: u32, _arg1: u32) -> u32"));
        assert!(raw.contains("pub unsafe fn i0048(_arg0: u32, _arg1: u64) -> u32"));
    }

    #[test]
    fn warp_barrier_rendering_preserves_mask_and_void_contracts() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let catalog = crate::resolve::resolve(&repo_root).unwrap();
        validate_renderable(&catalog).unwrap();
        assert_eq!(warp_barriers(&catalog).count(), 1);

        let record = warp_barriers(&catalog).next().unwrap();
        assert_eq!(record.id, "sync_mask");
        assert_eq!(
            record.warp_barrier.as_ref().unwrap().adapter,
            WarpBarrierAdapter::DirectMemberMask
        );

        let dialect_mod = render_dialect_mod(&catalog, "test-hash");
        assert!(dialect_mod.contains("mod warp_barrier;"));
        assert!(dialect_mod.contains("warp_barrier::register(ctx)"));

        let dialect = render_dialect_warp_barrier(&catalog, "test-hash");
        assert!(dialect.contains("pub struct BarWarpSyncOp"));
        assert!(dialect.contains("NOpdsInterface<1>, NResultsInterface<0>"));
        assert!(dialect.contains("vec![member_mask]"));
        assert!(dialect.contains("op.get_num_operands() != 1 || op.get_num_results() != 0"));
        assert!(dialect.contains("if !is_i32(ctx, op.get_operand(0).get_type(ctx))"));
        assert!(dialect.contains("requires exactly one member-mask operand and no results"));
        assert!(dialect.contains("member mask must be i32"));
        assert!(dialect.contains("BarWarpSyncOp::register(ctx)"));

        let importer = render_importer(&catalog, "test-hash");
        assert!(importer.contains("cuda_device::warp::sync_mask"));
        assert!(importer.contains("let barrier = BarWarpSyncOp::build(ctx, member_mask)"));
        assert!(importer.contains("set_generated_intrinsic_marker(ctx, barrier, \"v1:i0049\")"));
        assert!(importer.contains("helpers::emit_goto(ctx, *target_idx, barrier"));

        let lowering = render_lowering(&catalog, "test-hash");
        assert!(lowering.contains("impl MirToLlvmConversion for BarWarpSyncOp"));
        assert!(
            lowering.contains(
                "convert_bar_warp_sync(ctx, rewriter, self.get_operation(), operands_info)"
            )
        );

        let probe = render_probe(&catalog, record, "test-hash");
        assert!(probe.contains("declare void @llvm.nvvm.bar.warp.sync(i32)"));
        assert!(probe.contains("define void @probe_sync_mask(i32 %member_mask)"));
        assert!(probe.contains("call void @llvm.nvvm.bar.warp.sync(i32 %member_mask)"));
        assert!(probe.contains("define void @probe_sync_mask_immediate()"));
        assert!(probe.contains("call void @llvm.nvvm.bar.warp.sync(i32 -1)"));

        let raw = render_raw_abi(&catalog, "test-hash");
        assert!(raw.contains("pub unsafe fn i0049(_arg0: u32) -> ()"));
        assert!(raw.contains("On `sm_6x` and earlier"));
        assert!(raw.contains("no lane outside `mask` may be active"));
        assert!(raw.contains("The barrier orders memory accesses among participating lanes"));

        let reference = render_reference(&catalog, "test-hash");
        assert!(reference.contains("## Warp-barrier contract"));
        assert!(reference.contains("no unnamed lane may be active"));
        assert!(reference.contains("Both immediate and register masks are admitted"));
    }

    #[test]
    fn warp_shuffle_rendering_owns_all_i32_f32_and_i64_modes() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let catalog = crate::resolve::resolve(&repo_root).unwrap();
        validate_renderable(&catalog).unwrap();
        assert_eq!(warp_shuffles(&catalog).count(), 12);

        let dialect_mod = render_dialect_mod(&catalog, "test-hash");
        assert!(dialect_mod.contains("mod warp_shuffle;"));
        assert!(dialect_mod.contains("warp_shuffle::register(ctx)"));

        let dialect = render_dialect_warp_shuffle(&catalog, "test-hash");
        for op in [
            "ShflSyncIdxI32Op",
            "ShflSyncBflyI32Op",
            "ShflSyncDownI32Op",
            "ShflSyncUpI32Op",
            "ShflSyncIdxF32Op",
            "ShflSyncBflyF32Op",
            "ShflSyncDownF32Op",
            "ShflSyncUpF32Op",
            "ShflSyncIdxI64Op",
            "ShflSyncBflyI64Op",
            "ShflSyncDownI64Op",
            "ShflSyncUpI64Op",
        ] {
            assert!(dialect.contains(&format!("pub struct {op}")));
            assert!(dialect.contains(&format!("{op}::register(ctx)")));
        }
        assert!(dialect.contains("vec![member_mask, value, lane_or_delta]"));
        assert!(dialect.contains("requires i32 mask/lane and i32 value/result"));
        assert!(dialect.contains("requires i32 mask/lane and f32 value/result"));
        assert!(dialect.contains("requires i32 mask/lane and i64 value/result"));
        assert!(dialect.contains("fn is_i64"));
        assert!(dialect.contains("integer.width() == 64"));
        assert!(dialect.contains("IntegerType::get(ctx, 64, Signedness::Unsigned)"));

        let importer = render_importer(&catalog, "test-hash");
        assert!(importer.contains("cuda_device::warp::shuffle_sync"));
        assert!(importer.contains("cuda_device::warp::shuffle_up_f32_sync"));
        assert!(importer.contains(
            "let shuffle = ShflSyncIdxI32Op::build(ctx, member_mask, value, lane_or_delta)"
        ));
        assert!(importer.contains("set_generated_intrinsic_marker(ctx, shuffle, \"v1:i0050\")"));
        for (name, op, marker) in [
            ("shuffle_u64_sync", "ShflSyncIdxI64Op", "v1:i0058"),
            ("shuffle_xor_u64_sync", "ShflSyncBflyI64Op", "v1:i0059"),
            ("shuffle_down_u64_sync", "ShflSyncDownI64Op", "v1:i0060"),
            ("shuffle_up_u64_sync", "ShflSyncUpI64Op", "v1:i0061"),
        ] {
            assert!(importer.contains(&format!(
                "cuda_intrinsics::__cuda_oxide_intrinsic_abi_v1::{}",
                marker.strip_prefix("v1:").unwrap()
            )));
            assert!(importer.contains(&format!("cuda_device::warp::{name}")));
            assert!(importer.contains(&format!(
                "let shuffle = {op}::build(ctx, member_mask, value, lane_or_delta)"
            )));
            assert!(importer.contains(&format!(
                "set_generated_intrinsic_marker(ctx, shuffle, {marker:?})"
            )));
        }

        let lowering = render_lowering(&catalog, "test-hash");
        assert!(lowering.contains("impl MirToLlvmConversion for ShflSyncIdxI32Op"));
        assert!(lowering.contains("convert_shuffle_i32(ctx, rewriter"));
        assert!(lowering.contains("\"llvm_nvvm_shfl_sync_idx_i32\", 31)"));
        assert!(lowering.contains("impl MirToLlvmConversion for ShflSyncUpF32Op"));
        assert!(lowering.contains("convert_shuffle_f32(ctx, rewriter"));
        assert!(lowering.contains("\"llvm_nvvm_shfl_sync_up_f32\", 0)"));

        for (op, mode, clamp) in [
            ("ShflSyncIdxI64Op", "idx", 31),
            ("ShflSyncBflyI64Op", "bfly", 31),
            ("ShflSyncDownI64Op", "down", 31),
            ("ShflSyncUpI64Op", "up", 0),
        ] {
            assert!(lowering.contains(&format!("impl MirToLlvmConversion for {op}")));
            assert!(lowering.contains(&format!(
                "convert_shuffle_i64(ctx, rewriter, self.get_operation(), operands_info, {mode:?}, {clamp})"
            )));
        }
        assert!(!lowering.contains("llvm_nvvm_shfl_sync_idx_i64"));
        assert!(!lowering.contains("llvm_nvvm_shfl_sync_bfly_i64"));
        assert!(!lowering.contains("llvm_nvvm_shfl_sync_down_i64"));
        assert!(!lowering.contains("llvm_nvvm_shfl_sync_up_i64"));

        for record in warp_shuffles(&catalog) {
            let probe = render_probe(&catalog, record, "test-hash");
            let shuffle = record.warp_shuffle.as_ref().unwrap();
            if shuffle.value_kind == WarpShuffleValueKind::I64 {
                let mode = match shuffle.mode {
                    WarpShuffleMode::Idx => "idx",
                    WarpShuffleMode::Bfly => "bfly",
                    WarpShuffleMode::Down => "down",
                    WarpShuffleMode::Up => "up",
                };
                let asm = format!(
                    "{{ .reg .b32 lo; .reg .b32 hi; mov.b64 {{lo, hi}}, $1; shfl.sync.{mode}.b32 lo, lo, $2, {}, $3; shfl.sync.{mode}.b32 hi, hi, $2, {}, $3; mov.b64 $0, {{lo, hi}}; }}",
                    shuffle.clamp, shuffle.clamp
                );
                assert!(probe.contains(&format!(
                    "define i64 @probe_{}(i32 %member_mask, i64 %value, i32 %lane) #0",
                    record.id
                )));
                assert!(probe.contains(&format!(
                    "call i64 asm sideeffect {asm:?}, \"=l,l,r,r\"(i64 %value, i32 %lane, i32 %member_mask) #0"
                )));
                assert_eq!(probe.matches("asm sideeffect").count(), 1);
                assert_eq!(probe.matches("attributes #0 = { convergent }").count(), 1);
                assert!(!probe.contains("declare i64 @llvm.nvvm.shfl"));
                for suffix in ["rr", "ri", "ir", "ii"] {
                    assert!(!probe.contains(&format!("@probe_{}_{suffix}", record.id)));
                }
            } else {
                for suffix in ["rr", "ri", "ir", "ii"] {
                    assert!(probe.contains(&format!("@probe_{}_{suffix}", record.id)));
                }
                assert!(probe.contains(&format!(", i32 {})", shuffle.clamp)));
            }
        }

        let raw = render_raw_abi(&catalog, "test-hash");
        assert!(raw.contains("pub unsafe fn i0050(_arg0: u32, _arg1: u32, _arg2: u32) -> u32"));
        assert!(raw.contains("pub unsafe fn i0057(_arg0: u32, _arg1: f32, _arg2: u32) -> f32"));
        for abi_id in ["i0058", "i0059", "i0060", "i0061"] {
            assert!(raw.contains(&format!(
                "pub unsafe fn {abi_id}(_arg0: u32, _arg1: u64, _arg2: u32) -> u64"
            )));
        }
        assert!(raw.contains("If the computed source lane is in range"));
        assert!(raw.contains(
            "If PTX marks the computed source out of range, the calling lane's input is copied"
        ));
        assert!(raw.contains("two `b32` shuffles in one convergent block"));

        let reference = render_reference(&catalog, "test-hash");
        assert!(reference.contains("## Warp-shuffle contracts"));
        assert!(reference.contains("inserts clamp `31` during lowering"));
        assert!(reference.contains("inserts clamp `0` during lowering"));
        assert!(reference.contains("One convergent, side-effecting inline-PTX block splits `i64`"));
        assert!(reference.contains("PTX-native source; no LLVM record"));

        let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
        assert!(outputs.contains_key(&PathBuf::from(
            "crates/dialect-nvvm/src/ops/generated/warp_shuffle.rs"
        )));
    }

    #[test]
    fn dot_product_rendering_preserves_stable_paths_and_low_selector() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let catalog = crate::resolve::resolve(&repo_root).unwrap();
        validate_renderable(&catalog).unwrap();
        assert_eq!(dot_products(&catalog).count(), 4);

        let compatibility = render_compat_dotprod(&catalog, "test-hash");
        for name in ["dp4a_s32", "dp4a_u32", "dp2a_s32", "dp2a_u32"] {
            assert!(compatibility.contains(&format!("pub fn {name}(")));
        }

        let dialect = render_dialect_dotprod(&catalog, "test-hash");
        assert!(dialect.contains("pub struct Dp4aS32Op"));
        assert!(dialect.contains("pub struct Dp2aU32Op"));
        assert!(dialect.contains("NOpdsInterface<3>"));

        let importer = render_importer(&catalog, "test-hash");
        assert!(importer.contains("cuda_device::dotprod::dp2a_s32"));
        assert!(importer.contains("let dot = Dp2aS32Op::build(ctx, a, b, c)"));
        assert!(importer.contains("set_generated_intrinsic_marker(ctx, dot, \"v1:i0032\")"));

        let lowering = render_lowering(&catalog, "test-hash");
        assert!(lowering.contains("impl MirToLlvmConversion for Dp4aS32Op"));
        assert!(lowering.contains("\"llvm_nvvm_idp2a_s_s\""));
        assert!(lowering.contains("\"dp2a.lo.s32.s32 $0, $1, $2, $3;\""));
        assert!(lowering.contains("\"dp2a.lo.s32.s32 $0, $1, $2, $3;\", true)"));

        let low = dot_products(&catalog)
            .find(|record| record.id == "dp2a_s32")
            .unwrap();
        let probe = render_probe(&catalog, low, "test-hash");
        assert!(probe.contains("call i32 @llvm.nvvm.idp2a.s.s(i32 %a, i32 %b, i1 false, i32 %c)"));
        assert!(!probe.contains("i1 true"));

        let target = render_targets(&catalog, "test-hash");
        assert!(target.contains("GeneratedImmediateBinding { argument_index: 2, value: 0 }"));
        assert!(!target.contains("GeneratedImmediateBinding { argument_index: 2, value: -1 }"));
    }

    #[test]
    fn sync_threads_rendering_keeps_fixed_zero_and_backend_routes_explicit() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let catalog = crate::resolve::resolve(&repo_root).unwrap();
        validate_renderable(&catalog).unwrap();
        let record = sync_intrinsics(&catalog).next().unwrap();
        assert_eq!(sync_intrinsics(&catalog).count(), 1);

        let raw = render_raw_abi(&catalog, "test-hash");
        assert!(raw.contains("pub unsafe fn i0034() -> ()"));
        assert!(raw.contains("Every active thread in the CTA must reach the same barrier"));

        let dialect = render_dialect_sync(&catalog, "test-hash");
        assert!(dialect.contains("pub struct Barrier0Op"));
        assert!(dialect.contains("NOpdsInterface<0>, NResultsInterface<0>"));
        assert!(dialect.contains("Barrier0Op::register(ctx)"));

        let importer = render_importer(&catalog, "test-hash");
        assert!(importer.contains("cuda_device::thread::sync_threads"));
        assert!(importer.contains("cuda_device::sync_threads"));
        assert!(importer.contains("Barrier0Op::get_concrete_op_info()"));
        assert!(importer.contains("set_generated_intrinsic_marker(ctx, barrier, \"v1:i0034\")"));
        assert!(importer.contains("helpers::emit_goto(ctx, *target_idx, barrier"));

        let lowering = render_lowering(&catalog, "test-hash");
        assert!(lowering.contains("impl MirToLlvmConversion for Barrier0Op"));
        assert!(lowering.contains("create_i32_const(ctx, rewriter, 0)"));
        assert!(lowering.contains("\"llvm_nvvm_barrier_cta_sync_aligned_all\""));
        assert!(lowering.contains("IntrinsicBackend::LlvmNvptx"));
        assert!(lowering.contains("IntrinsicBackend::LibNvvm"));
        assert!(lowering.contains("\"bar.sync 0;\", \"~{memory}\""));

        let probe = render_probe(&catalog, record, "test-hash");
        assert!(probe.contains("declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32)"));
        assert!(probe.contains("call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)"));

        let targets = render_targets(&catalog, "test-hash");
        assert!(targets.contains("id: \"sync_threads\", abi_id: \"i0034\""));
        assert!(targets.contains("source_record: \"BARRIER_CTA_SYNC_ALIGNED_ALL_i\""));
        assert!(targets.contains(
            "backend: GeneratedIntrinsicBackend::LlvmNvptx, requirement: GeneratedTargetRequirement { minimum_ptx: GeneratedPtxVersion::from_encoded(32), hardware: GeneratedHardwareTarget::AnyOf(&[GeneratedHardwareAlternative::MinimumSm(20)]) }"
        ));
        assert!(targets.contains(
            "backend: GeneratedIntrinsicBackend::LibNvvm, requirement: GeneratedTargetRequirement { minimum_ptx: GeneratedPtxVersion::from_encoded(10), hardware: GeneratedHardwareTarget::AnyOf(&[GeneratedHardwareAlternative::MinimumSm(75)]) }"
        ));
        assert!(
            !record
                .selections
                .iter()
                .any(|selection| selection.source_record.ends_with("_r"))
        );

        let outputs = all_outputs(&catalog, "{}\n".into(), "test-hash").unwrap();
        assert!(outputs.contains_key(&PathBuf::from(
            "crates/dialect-nvvm/src/ops/generated/sync.rs"
        )));
    }
}
