/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::error::PipelineError;
use crate::generated::GeneratedModuleRequirements;
use crate::target::{
    DetectedFeatures, arch_satisfies, generated_ptx_isa_requirement, generated_target_satisfied,
    select_target_with_generated, validate_generated_target, validate_target_features,
};
use libnvvm_sys::CudaArch;
use llvm_export::export::{DebugKind, DeviceExternType, ExportBackendConfig, NvvmIrDialect};
use pliron::builtin::op_interfaces::{CallOpCallable, CallOpInterface, SymbolOpInterface};
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use std::path::Path;

/// An external device function declaration (for FFI with external LTOIR).
///
/// Unlike `CollectedFunction`, these have no MIR body - they're just declarations
/// that will be emitted as LLVM `declare` statements for nvJitLink to resolve
/// when linking with external LTOIR (e.g., CCCL libraries).
#[derive(Debug, Clone)]
pub struct DeviceExternDecl {
    /// The export name (the original function name, e.g., "cub_block_reduce_sum").
    pub export_name: String,

    /// Structured LLVM ABI parameter types. Pointer pointees are retained even
    /// though the lowered pliron LLVM module itself uses opaque pointers.
    pub param_types: Vec<DeviceExternType>,

    /// Structured LLVM ABI return type.
    pub return_type: DeviceExternType,

    /// NVVM attributes for this function.
    pub attrs: DeviceExternAttrs,
}

/// NVVM attributes for device extern declarations.
///
/// NOTE: These attributes are currently **not emitted** to the LLVM IR output.
/// When linking LTOIR via nvJitLink, the external library's LTOIR already contains
/// proper attributes (convergent, nounwind, memory, etc.) on the function DEFINITIONS.
/// nvJitLink uses the definition's attributes during LTO, making attributes on our
/// declarations redundant.
///
/// This struct is retained for the pipeline API but values are not used in code generation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct DeviceExternAttrs {
    /// Function is convergent (all threads must execute together).
    /// NOTE: Not currently emitted to LLVM IR.
    pub is_convergent: bool,

    /// Function is pure (no side effects, result depends only on inputs).
    /// NOTE: Not currently emitted to LLVM IR.
    pub is_pure: bool,

    /// Function is read-only (only reads memory, doesn't write).
    /// NOTE: Not currently emitted to LLVM IR.
    pub is_readonly: bool,
}

// Implement AsDeviceExtern trait for llvm-export integration
impl llvm_export::export::AsDeviceExtern for DeviceExternDecl {
    fn as_device_extern(&self) -> llvm_export::export::DeviceExternDecl {
        llvm_export::export::DeviceExternDecl {
            export_name: self.export_name.clone(),
            param_types: self.param_types.clone(),
            return_type: self.return_type.clone(),
            attrs: llvm_export::export::DeviceExternAttrs {
                is_convergent: self.attrs.is_convergent,
                is_pure: self.attrs.is_pure,
                is_readonly: self.attrs.is_readonly,
            },
        }
    }
}

// mir-importer pipeline plumbing; not part of the frontend contract.
#[doc(hidden)]
#[cfg(test)]
pub fn resolve_nvvm_target(
    explicit_target: Option<&str>,
    device_arch_hint: Option<&str>,
    automatic_features: Option<DetectedFeatures>,
) -> Result<CudaArch, PipelineError> {
    resolve_nvvm_target_with_generated(
        explicit_target,
        device_arch_hint,
        automatic_features,
        &GeneratedModuleRequirements::default(),
    )
}

pub(crate) fn resolve_nvvm_target_with_generated(
    explicit_target: Option<&str>,
    device_arch_hint: Option<&str>,
    automatic_features: Option<DetectedFeatures>,
    generated: &GeneratedModuleRequirements,
) -> Result<CudaArch, PipelineError> {
    let parse = |target: &str, source: &str| {
        target.parse::<CudaArch>().map_err(|error| {
            PipelineError::Export(format!(
                "cannot select an NVVM IR dialect from the {source} `{target}`: {error}"
            ))
        })
    };

    // libNVVM chooses the final PTX version itself, but still reject a future
    // catalog floor that this compiler does not know how to represent. This
    // keeps NVVM IR and ordinary PTX builds equally fail closed.
    generated_ptx_isa_requirement(generated).map_err(PipelineError::Export)?;

    if let Some(target) = explicit_target {
        let parsed = parse(target, "explicit CUDA target")?;
        if let Some(features) = automatic_features {
            validate_target_features(&parsed, features).map_err(PipelineError::Export)?;
        }
        validate_generated_target(&parsed.sm(), generated).map_err(PipelineError::Export)?;
        return Ok(parsed);
    }

    if let Some(features) = automatic_features {
        if let Some(target) = device_arch_hint {
            let parsed = parse(target, "detected GPU architecture")?;
            if arch_satisfies(&parsed.sm(), features)
                && generated_target_satisfied(&parsed.sm(), generated)
            {
                return Ok(parsed);
            }
        }
        let target =
            select_target_with_generated(features, generated).map_err(PipelineError::Export)?;
        return parse(&target, "feature-based compiler default");
    }

    if let Some(target) = device_arch_hint {
        let parsed = parse(target, "detected GPU architecture")?;
        if generated_target_satisfied(&parsed.sm(), generated) {
            return Ok(parsed);
        }
    }

    if !generated.is_empty() {
        let target = select_target_with_generated(DetectedFeatures::Basic, generated)
            .map_err(PipelineError::Export)?;
        return parse(&target, "generated-intrinsic requirement");
    }

    Err(PipelineError::Export(
        "NVVM IR requires a concrete CUDA target because pre-Blackwell and Blackwell+ \
         use different LLVM dialects; pass `cargo oxide ... --arch sm_XX` (or set \
         CUDA_OXIDE_TARGET)"
            .to_string(),
    ))
}

// mir-importer pipeline plumbing; not part of the frontend contract.
#[doc(hidden)]
pub fn validate_nvvm_debug_support(
    target: &CudaArch,
    dialect: NvvmIrDialect,
    debug_kind: DebugKind,
) -> Result<(), PipelineError> {
    if dialect == NvvmIrDialect::LegacyLlvm7 && debug_kind != DebugKind::Off {
        return Err(PipelineError::Export(format!(
            "legacy LLVM 7 NVVM IR for {} does not yet support cuda-oxide debug metadata; \
             rebuild without device debug information",
            target.sm()
        )));
    }
    Ok(())
}

/// Exports an LLVM dialect module to textual LLVM IR (`.ll` file).
///
/// Backend configuration is selected based on flags:
/// - `emit_nvvm_ir`: Uses `NvvmExportConfig` for NVVM IR output
/// - Otherwise: Uses default `PtxExportConfig` for standard PTX generation
///
/// Device extern declarations are emitted before the main module content.
// mir-importer pipeline plumbing; not part of the frontend contract.
#[doc(hidden)]
pub fn export_llvm_ir(
    ctx: &Context,
    module_op_ptr: Ptr<Operation>,
    device_externs: &[DeviceExternDecl],
    path: &Path,
    emit_nvvm_ir: bool,
    nvvm_dialect: Option<NvvmIrDialect>,
    debug_kind: DebugKind,
) -> Result<String, PipelineError> {
    let llvm_ir = render_llvm_ir(
        ctx,
        module_op_ptr,
        device_externs,
        emit_nvvm_ir,
        nvvm_dialect,
        debug_kind,
    )?;

    std::fs::write(path, &llvm_ir).map_err(|e| PipelineError::Export(e.to_string()))?;

    Ok(llvm_ir)
}

/// Render LLVM text without publishing an artifact.
///
/// Automatic libdevice mode uses this once before NVVM legalization to detect
/// the same target features as the normal PTX path. The final export still
/// happens exactly once, after the target-specific legalization pass.
// mir-importer pipeline plumbing; not part of the frontend contract.
#[doc(hidden)]
pub fn render_llvm_ir(
    ctx: &Context,
    module_op_ptr: Ptr<Operation>,
    device_externs: &[DeviceExternDecl],
    emit_nvvm_ir: bool,
    nvvm_dialect: Option<NvvmIrDialect>,
    debug_kind: DebugKind,
) -> Result<String, PipelineError> {
    let module_op = Operation::get_op::<pliron::builtin::ops::ModuleOp>(module_op_ptr, ctx)
        .ok_or_else(|| PipelineError::Export("Not a module op".to_string()))?;

    let llvm_ir = if emit_nvvm_ir {
        let dialect = nvvm_dialect.ok_or_else(|| {
            PipelineError::Export("NVVM export reached without a selected IR dialect".to_string())
        })?;
        let config = PipelineExportConfig {
            inner: llvm_export::export::NvvmExportConfig::new(dialect),
            debug_kind,
        };
        llvm_export::export::export_module_with_externs(ctx, &module_op, device_externs, &config)
            .map_err(PipelineError::Export)?
    } else {
        let config = PipelineExportConfig {
            inner: llvm_export::export::PtxExportConfig,
            debug_kind,
        };
        llvm_export::export::export_module_with_externs(ctx, &module_op, device_externs, &config)
            .map_err(PipelineError::Export)?
    };

    Ok(llvm_ir)
}

struct PipelineExportConfig<C> {
    inner: C,
    debug_kind: DebugKind,
}

impl<C: ExportBackendConfig> ExportBackendConfig for PipelineExportConfig<C> {
    fn datalayout(&self) -> &str {
        self.inner.datalayout()
    }

    fn emit_llvm_used(&self) -> bool {
        self.inner.emit_llvm_used()
    }

    fn emit_nvvmir_version(&self) -> bool {
        self.inner.emit_nvvmir_version()
    }

    fn nvvmir_version(&self) -> [i32; 4] {
        self.inner.nvvmir_version()
    }

    fn emit_all_kernel_annotations(&self) -> bool {
        self.inner.emit_all_kernel_annotations()
    }

    fn emit_ptx_kernel_keyword(&self) -> bool {
        self.inner.emit_ptx_kernel_keyword()
    }

    fn nvvm_ir_dialect(&self) -> Option<NvvmIrDialect> {
        self.inner.nvvm_ir_dialect()
    }

    fn debug_kind(&self) -> DebugKind {
        self.debug_kind
    }
}

/// Returns true when lowering emitted CUDA libdevice calls.
///
/// Float math intrinsics (sin, cos, exp, log, pow, …) lower to `__nv_*`
/// entry points from `libdevice.10.bc`. `llc` cannot resolve these; they
/// need libNVVM + nvJitLink + libdevice. When we see any `__nv_*` symbol
/// the example owns the LTOIR build (see `examples/device_ffi_test/tools/`).
// mir-importer pipeline plumbing; not part of the frontend contract.
#[doc(hidden)]
pub fn module_uses_libdevice(ctx: &Context, module_op_ptr: Ptr<Operation>) -> bool {
    op_uses_libdevice(ctx, module_op_ptr)
}

/// Return unresolved non-intrinsic LLVM function declarations.
///
/// Standalone PTX has no link step. LLVM intrinsics are resolved by `llc`, but
/// declarations such as `__nv_sinf`, `vprintf`, or user device externs would
/// leave an artifact that the CUDA driver cannot load by itself.
pub(crate) fn unresolved_external_symbols(
    ctx: &Context,
    module_op_ptr: Ptr<Operation>,
) -> Vec<String> {
    let mut symbols = Vec::new();
    collect_unresolved_external_symbols(ctx, module_op_ptr, &mut symbols);
    symbols.sort();
    symbols.dedup();
    symbols
}

fn collect_unresolved_external_symbols(
    ctx: &Context,
    op_ptr: Ptr<Operation>,
    symbols: &mut Vec<String>,
) {
    if let Some(func) = Operation::get_op::<llvm_export::ops::FuncOp>(op_ptr, ctx) {
        let op = func.get_operation().deref(ctx);
        let name = func.get_symbol_name(ctx).to_string();
        if op.regions().count() == 0 && !name.starts_with("llvm_") {
            symbols.push(name);
        }
    }

    let op_ref = op_ptr.deref(ctx);
    for region in op_ref.regions() {
        let region_ref = region.deref(ctx);
        for block in region_ref.iter(ctx) {
            let block_ref = block.deref(ctx);
            for child_op in block_ref.iter(ctx) {
                collect_unresolved_external_symbols(ctx, child_op, symbols);
            }
        }
    }
}

/// Recursively scan for declared or called CUDA libdevice functions.
fn op_uses_libdevice(ctx: &Context, op_ptr: Ptr<Operation>) -> bool {
    if let Some(func) = Operation::get_op::<llvm_export::ops::FuncOp>(op_ptr, ctx)
        && func.get_symbol_name(ctx).starts_with("__nv_")
    {
        return true;
    }

    if let Some(call) = Operation::get_op::<llvm_export::ops::CallOp>(op_ptr, ctx)
        && let CallOpCallable::Direct(callee) = call.callee(ctx)
        && callee.to_string().starts_with("__nv_")
    {
        return true;
    }

    let op_ref = op_ptr.deref(ctx);
    for region in op_ref.regions() {
        let region_ref = region.deref(ctx);
        for block in region_ref.iter(ctx) {
            let block_ref = block.deref(ctx);
            for child_op in block_ref.iter(ctx) {
                if op_uses_libdevice(ctx, child_op) {
                    return true;
                }
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generated_intrinsic_targets::{
        GeneratedIntrinsicBackend, generated_intrinsic_target_by_marker,
    };
    use llvm_export::export::AsDeviceExtern;
    use pliron::op::Op;

    #[test]
    fn nvvm_target_resolution_applies_generated_backend_floors() {
        let f16 = generated_intrinsic_target_by_marker("v1:i0014").unwrap();
        let requirements = GeneratedModuleRequirements::from_targets(vec![f16])
            .for_backend(GeneratedIntrinsicBackend::LibNvvm);

        let error = resolve_nvvm_target_with_generated(Some("sm_70"), None, None, &requirements)
            .unwrap_err()
            .to_string();
        assert!(error.contains("packed_atomic_add_f16x2"), "{error}");
        assert!(error.contains("sm_75 or newer"), "{error}");
        assert_eq!(
            resolve_nvvm_target_with_generated(Some("sm_75"), None, None, &requirements)
                .unwrap()
                .sm(),
            "sm_75"
        );
    }

    #[test]
    fn nvvm_target_resolution_is_concrete_and_strict() {
        let legacy = resolve_nvvm_target(Some("compute_90a"), Some("sm_120"), None).unwrap();
        assert_eq!(legacy.sm(), "sm_90a");
        assert!(legacy.uses_legacy_llvm());

        let modern = resolve_nvvm_target(None, Some("sm_120f"), None).unwrap();
        assert_eq!(modern.compute(), "compute_120f");
        assert!(!modern.uses_legacy_llvm());

        for target in [None, Some("nvvm-ir"), Some("sm_90x"), Some("86")] {
            assert!(
                resolve_nvvm_target(target, None, None).is_err(),
                "{target:?}"
            );
        }
    }

    #[test]
    fn automatic_nvvm_target_uses_the_module_feature_floor() {
        for (features, expected, is_legacy) in [
            (DetectedFeatures::Basic, "sm_80", true),
            (DetectedFeatures::Sm80, "sm_80", true),
            (DetectedFeatures::Sm90, "sm_90", true),
            (DetectedFeatures::Cluster, "sm_90", true),
            (DetectedFeatures::Wgmma, "sm_90a", true),
            (DetectedFeatures::Tma, "sm_100", false),
            (DetectedFeatures::TmaMulticast, "sm_100a", false),
            (DetectedFeatures::Blackwell, "sm_100a", false),
        ] {
            let target = resolve_nvvm_target(None, None, Some(features)).unwrap();
            assert_eq!(target.sm(), expected, "{features:?}");
            assert_eq!(target.uses_legacy_llvm(), is_legacy, "{features:?}");
        }
    }

    #[test]
    fn automatic_nvvm_target_uses_only_a_compatible_device_hint() {
        let turing =
            resolve_nvvm_target(None, Some("sm_75"), Some(DetectedFeatures::Basic)).unwrap();
        assert_eq!(turing.sm(), "sm_75");

        let sm80_on_turing =
            resolve_nvvm_target(None, Some("sm_75"), Some(DetectedFeatures::Sm80)).unwrap();
        assert_eq!(sm80_on_turing.sm(), "sm_80");

        let blackwell =
            resolve_nvvm_target(None, Some("sm_120a"), Some(DetectedFeatures::Basic)).unwrap();
        assert_eq!(blackwell.sm(), "sm_120a");

        let sm80_on_blackwell =
            resolve_nvvm_target(None, Some("sm_120a"), Some(DetectedFeatures::Sm80)).unwrap();
        assert_eq!(sm80_on_blackwell.sm(), "sm_120a");

        let ampere =
            resolve_nvvm_target(None, Some("sm_80"), Some(DetectedFeatures::Sm80)).unwrap();
        assert_eq!(ampere.sm(), "sm_80");

        let hopper_floor =
            resolve_nvvm_target(None, Some("sm_80"), Some(DetectedFeatures::Sm90)).unwrap();
        assert_eq!(hopper_floor.sm(), "sm_90");

        let forward_compatible =
            resolve_nvvm_target(None, Some("sm_120"), Some(DetectedFeatures::Sm90)).unwrap();
        assert_eq!(forward_compatible.sm(), "sm_120");

        let hopper =
            resolve_nvvm_target(None, Some("sm_120a"), Some(DetectedFeatures::Wgmma)).unwrap();
        assert_eq!(hopper.sm(), "sm_90a");

        assert!(
            resolve_nvvm_target(None, Some("not-an-arch"), Some(DetectedFeatures::Basic)).is_err()
        );
    }

    #[test]
    fn explicit_nvvm_target_rejects_a_detected_feature_below_its_floor() {
        let error = resolve_nvvm_target(Some("sm_70"), None, Some(DetectedFeatures::Movmatrix))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("cannot lower detected feature Movmatrix"),
            "{error}"
        );

        let target =
            resolve_nvvm_target(Some("sm_75"), None, Some(DetectedFeatures::Movmatrix)).unwrap();
        assert_eq!(target.sm(), "sm_75");
    }

    #[test]
    fn compatible_explicit_nvvm_target_wins_over_automatic_selection() {
        let target =
            resolve_nvvm_target(Some("sm_86"), Some("sm_120a"), Some(DetectedFeatures::Sm80))
                .unwrap();
        assert_eq!(target.sm(), "sm_86");
    }

    #[test]
    fn legacy_nvvm_debug_is_rejected() {
        let legacy = resolve_nvvm_target(Some("sm_90"), None, None).unwrap();
        assert!(
            validate_nvvm_debug_support(
                &legacy,
                NvvmIrDialect::LegacyLlvm7,
                DebugKind::LineTables,
            )
            .is_err()
        );
        validate_nvvm_debug_support(&legacy, NvvmIrDialect::LegacyLlvm7, DebugKind::Off).unwrap();

        let modern = resolve_nvvm_target(Some("sm_120"), None, None).unwrap();
        validate_nvvm_debug_support(&modern, NvvmIrDialect::Modern, DebugKind::Full).unwrap();
    }

    #[test]
    fn test_device_extern_decl_converts_to_export_decl() {
        let decl = DeviceExternDecl {
            export_name: "device_add".to_string(),
            param_types: vec![
                DeviceExternType::pointer_to(DeviceExternType::Float32, 0),
                DeviceExternType::Integer(32),
            ],
            return_type: DeviceExternType::Void,
            attrs: DeviceExternAttrs {
                is_convergent: true,
                is_pure: false,
                is_readonly: true,
            },
        };

        let exported = decl.as_device_extern();

        assert_eq!(exported.export_name, "device_add");
        assert_eq!(
            exported.param_types,
            [
                DeviceExternType::pointer_to(DeviceExternType::Float32, 0),
                DeviceExternType::Integer(32),
            ]
        );
        assert_eq!(exported.return_type, DeviceExternType::Void);
        assert!(exported.attrs.is_convergent);
        assert!(!exported.attrs.is_pure);
        assert!(exported.attrs.is_readonly);
    }

    /// Build a minimal LLVM dialect module containing a single function
    /// declaration named `name`. The module is intentionally empty otherwise;
    /// the auto-detect logic only inspects the symbol name on declarations
    /// and on direct call sites.
    fn build_module_with_func_decl(ctx: &mut Context, name: &str) -> Ptr<Operation> {
        use llvm_export::ops::FuncOp as LlvmFuncOp;
        use llvm_export::types::FuncType as LlvmFuncType;
        use pliron::basic_block::BasicBlock;
        use pliron::builtin::ops::ModuleOp;
        use pliron::builtin::types::{IntegerType, Signedness};

        let module = ModuleOp::new(ctx, "test_module".try_into().unwrap());
        let module_ptr = module.get_operation();
        let module_region = module_ptr.deref(ctx).get_region(0);

        let module_block = {
            let region_ref = module_region.deref(ctx);
            if let Some(first_block) = region_ref.iter(ctx).next() {
                first_block
            } else {
                drop(region_ref);
                let new_block = BasicBlock::new(ctx, None, vec![]);
                new_block.insert_at_back(module_region, ctx);
                new_block
            }
        };

        let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
        let func_ty = LlvmFuncType::get(ctx, i32_ty.into(), vec![i32_ty.into()], false);
        let func = LlvmFuncOp::new(ctx, name.try_into().unwrap(), func_ty);
        func.get_operation().insert_at_back(module_block, ctx);

        module_ptr
    }

    #[test]
    fn test_module_uses_libdevice_detects_nv_func_decl() {
        let mut ctx = Context::new();
        let module_ptr = build_module_with_func_decl(&mut ctx, "__nv_sqrtf");
        assert!(
            module_uses_libdevice(&ctx, module_ptr),
            "module containing `__nv_*` function declaration must be flagged"
        );
    }

    #[test]
    fn in_memory_llvm_preview_uses_the_shared_feature_detector() {
        let mut ctx = Context::new();
        let module_ptr = build_module_with_func_decl(&mut ctx, "llvm_nvvm_tcgen05_alloc");

        let preview = render_llvm_ir(&ctx, module_ptr, &[], false, None, DebugKind::Off).unwrap();

        assert!(preview.contains("@llvm.nvvm.tcgen05.alloc"), "{preview}");
        assert_eq!(
            crate::target::detect_features_in_llvm_text(&preview),
            DetectedFeatures::Blackwell
        );
    }

    #[test]
    fn test_module_uses_libdevice_ignores_unrelated_funcs() {
        let mut ctx = Context::new();
        let module_ptr = build_module_with_func_decl(&mut ctx, "kernel_main");
        assert!(
            !module_uses_libdevice(&ctx, module_ptr),
            "module without any `__nv_*` symbols must not be flagged"
        );
    }

    #[test]
    fn test_module_uses_libdevice_does_not_match_partial_prefix() {
        // "__nvm_foo" starts with "__nv" but not "__nv_". The detection rule
        // is the full `__nv_` prefix, so this must not trigger auto-detect.
        let mut ctx = Context::new();
        let module_ptr = build_module_with_func_decl(&mut ctx, "__nvm_foo");
        assert!(
            !module_uses_libdevice(&ctx, module_ptr),
            "names starting with `__nv` but not `__nv_` must not be flagged"
        );
    }

    /// `module_uses_libdevice` must also fire when the libdevice symbol
    /// appears as the callee of a direct `CallOp` -- this is the realistic
    /// case where a normal kernel calls `__nv_sqrtf`. The auto-detect
    /// recursion has to walk through the module region and visit the
    /// `CallOp` even when no enclosing `FuncOp` matches the prefix rule.
    #[test]
    fn test_module_uses_libdevice_detects_direct_nv_call() {
        use llvm_export::ops::CallOp as LlvmCallOp;
        use llvm_export::types::FuncType as LlvmFuncType;
        use pliron::basic_block::BasicBlock;
        use pliron::builtin::ops::ModuleOp;
        use pliron::builtin::types::{IntegerType, Signedness};

        let mut ctx = Context::new();

        let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
        let module_ptr = module.get_operation();
        let module_region = module_ptr.deref(&ctx).get_region(0);
        let module_block = BasicBlock::new(&mut ctx, None, vec![]);
        module_block.insert_at_back(module_region, &ctx);

        let i32_ty = IntegerType::get(&ctx, 32, Signedness::Signless);
        let callee_ty = LlvmFuncType::get(&ctx, i32_ty.into(), vec![], false);
        let callee_ident: pliron::identifier::Identifier = "__nv_sqrtf".try_into().unwrap();
        let nv_call = LlvmCallOp::new(
            &mut ctx,
            CallOpCallable::Direct(callee_ident),
            callee_ty,
            vec![],
        );
        nv_call.get_operation().insert_at_back(module_block, &ctx);

        assert!(
            module_uses_libdevice(&ctx, module_ptr),
            "direct call to a `__nv_*` symbol must be detected"
        );
    }
}
