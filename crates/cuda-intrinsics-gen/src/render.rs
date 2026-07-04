/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    BackendLoweringMechanism, CatalogFile, CatalogHardwareAlternative, CatalogHardwareTarget,
    CatalogIntrinsic, CatalogLlvm, CatalogSelection, EvidenceArtifactKind, EvidenceStageKind,
    ImportedAddressSpace, IntrinsicBackend, IntrinsicSource, LdmatrixElement, LdmatrixLayout,
    LdmatrixMultiplicity, LdmatrixShape, LdmatrixStateSpace, PackedAtomicFormat,
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
        "crates/dialect-nvvm/src/ops/generated/mod.rs".into(),
        render_dialect_mod(catalog, catalog_sha256),
    );
    outputs.insert(
        "crates/dialect-nvvm/src/ops/generated/sreg.rs".into(),
        render_dialect_sreg(catalog, catalog_sha256),
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
                    || (record.family == "packed_atomic" && record.packed_atomic.is_some()),
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

fn render_dialect_mod(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "mod ldmatrix;\nmod packed_atomic;\nmod sreg;\n\npub use ldmatrix::*;\npub use packed_atomic::*;\npub use sreg::*;\n\nuse pliron::context::Context;\n\npub(super) fn register(ctx: &mut Context) {\n    ldmatrix::register(ctx);\n    packed_atomic::register(ctx);\n    sreg::register(ctx);\n}\n",
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
        render_inline_patterns(&mut output, &[record.rust.canonical_path.as_str()]);
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
        writeln!(
            output,
            "            let (value, array) = helpers::bundle_generated_u32_results_as_array(ctx, load, {register_count}, loc.clone());"
        )
        .unwrap();
        writeln!(
            output,
            "            Ok(Some(helpers::emit_store_result_and_goto(\n                ctx, destination, value, target, block_ptr, array, value_map, block_map, loc,\n                {:?},\n            )?))",
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
        "//! Generated direct-NVVM, ldmatrix, and packed-atomic conversion interfaces.\n\nuse crate::conversion_interface::MirToLlvmConversion;\nuse crate::convert::intrinsics::{atomic::convert_packed_atom_add, common::call_intrinsic, ldmatrix::convert_generated_ldmatrix};\nuse dialect_nvvm::ops::{",
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
    format!("GeneratedSelectionConstraints {{ address_space: {address_space} }}")
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
        "//! Generated target requirements and separately imported LLVM/selection facts.\n\npub const GENERATED_INTRINSIC_MARKER_ATTR: &str = \"cuda_oxide_intrinsic_marker\";\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]\npub struct GeneratedPtxVersion(u16);\nimpl GeneratedPtxVersion {\n    pub const fn from_encoded(encoded: u16) -> Self { Self(encoded) }\n    pub const fn encoded(self) -> u16 { self.0 }\n    pub const fn major(self) -> u16 { self.0 / 10 }\n    pub const fn minor(self) -> u16 { self.0 % 10 }\n}\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedHardwareAlternative { MinimumSm(u16), ExactArchitecture(u16), FamilyTarget(u16) }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedHardwareTarget { All, AnyOf(&'static [GeneratedHardwareAlternative]) }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedTargetRequirement { pub minimum_ptx: GeneratedPtxVersion, pub hardware: GeneratedHardwareTarget }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedIntrinsicBackend { LlvmNvptx, LibNvvm }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedBackendRequirement { pub backend: GeneratedIntrinsicBackend, pub requirement: GeneratedTargetRequirement }\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedSelectionAddressSpace { Generic, Shared }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedSelectionConstraints { pub address_space: Option<GeneratedSelectionAddressSpace> }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedSelectionAlternative { pub source_record: &'static str, pub asm: &'static str, pub predicates: &'static [&'static str], pub constraints: GeneratedSelectionConstraints }\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedIntrinsicRange { pub lower: &'static str, pub upper_exclusive: &'static str }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedLlvmFacts { pub properties: &'static [&'static str], pub result_no_undef: bool, pub result_range: Option<GeneratedIntrinsicRange> }\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedLdmatrixShape { M8n8 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedLdmatrixMultiplicity { X1, X2, X4 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedLdmatrixLayout { Normal, Transposed }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedPackedAtomicFormat { F16x2, Bf16x2 }\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum GeneratedIntrinsicVariant {\n    Scalar,\n    Ldmatrix { shape: GeneratedLdmatrixShape, multiplicity: GeneratedLdmatrixMultiplicity, layout: GeneratedLdmatrixLayout },\n    PackedAtomic { format: GeneratedPackedAtomicFormat },\n}\n\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct GeneratedIntrinsicTarget {\n    pub marker: &'static str,\n    pub id: &'static str,\n    pub abi_id: &'static str,\n    pub dialect_op: &'static str,\n    pub variant: GeneratedIntrinsicVariant,\n    pub requirement: GeneratedTargetRequirement,\n    pub backend_requirements: &'static [GeneratedBackendRequirement],\n    pub selections: &'static [GeneratedSelectionAlternative],\n    pub llvm: Option<GeneratedLlvmFacts>,\n}\n\nimpl GeneratedIntrinsicTarget {\n    pub fn requirement_for_backend(&self, backend: GeneratedIntrinsicBackend) -> GeneratedTargetRequirement {\n        self.backend_requirements.iter().find(|entry| entry.backend == backend).map(|entry| entry.requirement).unwrap_or(self.requirement)\n    }\n}\n\npub const GENERATED_INTRINSIC_TARGETS: &[GeneratedIntrinsicTarget] = &[\n",
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
                    "        assert_eq!(target.llvm.unwrap().properties, &{:?});",
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
    if let Some(packed) = &record.packed_atomic {
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
        let register_count = record
            .ldmatrix
            .as_ref()
            .expect("non-scalar generated recipe")
            .variant
            .multiplicity
            .register_count();
        let result_ty = format!(
            "{{ {} }}",
            std::iter::repeat_n("i32", register_count)
                .collect::<Vec<_>>()
                .join(", ")
        );
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
                },
            },
            CatalogSelection {
                source_record: "SELECT_B".into(),
                asm: "op.b $dst;".into(),
                predicates: vec!["HasB".into()],
                constraints: ImportedSelectionConstraints {
                    address_space: Some(ImportedAddressSpace::Shared),
                },
            },
        ];

        let rendered = generated_selection_alternatives(&selections);
        assert_eq!(rendered.matches("GeneratedSelectionAlternative").count(), 2);
        assert!(rendered.contains(
            "source_record: \"SELECT_A\", asm: \"op.a $dst;\", predicates: &[\"HasA\", \"HasCommon\"], constraints: GeneratedSelectionConstraints { address_space: Some(GeneratedSelectionAddressSpace::Generic) }"
        ));
        assert!(rendered.contains(
            "source_record: \"SELECT_B\", asm: \"op.b $dst;\", predicates: &[\"HasB\"], constraints: GeneratedSelectionConstraints { address_space: Some(GeneratedSelectionAddressSpace::Shared) }"
        ));
    }

    #[test]
    fn ldmatrix_family_lowering_uses_one_attribute_dispatch_impl() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut catalog = crate::resolve::resolve(&repo_root).unwrap();
        let mut sibling = catalog
            .intrinsics
            .iter()
            .find(|record| record.family == "ldmatrix")
            .unwrap()
            .clone();
        sibling.id = "ldmatrix_m8n8_x2_trans_b16_test".into();
        sibling.rust.abi_id = "i9998".into();
        sibling.ldmatrix.as_mut().unwrap().variant.multiplicity = LdmatrixMultiplicity::X2;
        sibling.ldmatrix.as_mut().unwrap().variant.layout = LdmatrixLayout::Transposed;
        sibling.llvm.as_mut().unwrap().resolved_symbol =
            Some("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.trans.b16.p3".into());
        catalog.intrinsics.push(sibling);

        let rendered = render_lowering(&catalog, "test-hash");
        assert_eq!(
            rendered
                .matches("impl MirToLlvmConversion for LdmatrixOp")
                .count(),
            1
        );
        assert!(rendered.contains("LdmatrixMultiplicityAttr::X4"));
        assert!(rendered.contains("LdmatrixMultiplicityAttr::X2"));
        assert!(rendered.contains("LdmatrixLayoutAttr::Transposed"));
        assert!(rendered.contains("llvm_nvvm_ldmatrix_sync_aligned_m8n8_x2_trans_b16_p3"));
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
        assert_eq!(rendered.matches("#[must_use]").count(), 2);
        for abi_id in ["i0014", "i0015"] {
            let signature = format!("pub unsafe fn {abi_id}(_arg0: *mut u32, _arg1: u32) -> u32");
            let index = rendered.find(&signature).unwrap();
            assert!(rendered[..index].ends_with("#[must_use]\n#[inline(never)]\n"));
        }
    }
}
