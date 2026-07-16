/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Generated-intrinsic metadata collected before MIR lowering erases it.

use crate::error::PipelineError;
use crate::generated_intrinsic_targets::{
    GENERATED_INTRINSIC_MARKER_ATTR, GeneratedIntrinsicBackend, GeneratedIntrinsicTarget,
    GeneratedTargetRequirement, generated_intrinsic_operation_matches,
    generated_intrinsic_target_by_marker, generated_intrinsic_targets_by_op_name,
};
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::operation::Operation;
use pliron::printable::Printable;

/// Whether generated dialect operations must carry their Rust-source ABI marker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GeneratedMarkerPolicy {
    /// The rustc frontend must preserve the exact source ABI marker.
    Required,
    /// Direct dialect frontends may select the unique catalog variant structurally.
    Optional,
}

/// Exact generated-intrinsic requirements found in typed, pre-lowering IR.
///
/// The vector is deterministic and contains at most one entry per ABI marker.
/// Multiple calls to one intrinsic therefore do not make target checking
/// depend on call count or traversal order.
#[derive(Debug, Clone)]
pub(crate) struct GeneratedModuleRequirements {
    pub(crate) targets: Vec<&'static GeneratedIntrinsicTarget>,
    backend: GeneratedIntrinsicBackend,
}

impl Default for GeneratedModuleRequirements {
    fn default() -> Self {
        Self {
            targets: Vec::new(),
            backend: GeneratedIntrinsicBackend::LlvmNvptx,
        }
    }
}

impl GeneratedModuleRequirements {
    pub(crate) fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    pub(crate) fn for_backend(mut self, backend: GeneratedIntrinsicBackend) -> Self {
        self.backend = backend;
        self
    }

    pub(crate) fn requirement(
        &self,
        target: &GeneratedIntrinsicTarget,
    ) -> GeneratedTargetRequirement {
        target.requirement_for_backend(self.backend)
    }

    #[cfg(test)]
    pub(crate) fn from_targets(targets: Vec<&'static GeneratedIntrinsicTarget>) -> Self {
        Self {
            targets,
            ..Self::default()
        }
    }
}

/// Collect generated-intrinsic requirements before lowering erases typed
/// operations and their compiler-only source ABI markers.
pub(crate) fn collect_generated_intrinsic_requirements(
    ctx: &Context,
    root: Ptr<Operation>,
    marker_policy: GeneratedMarkerPolicy,
) -> Result<GeneratedModuleRequirements, PipelineError> {
    use pliron::identifier::Identifier;
    use std::collections::BTreeMap;

    let marker_key = Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).map_err(|error| {
        PipelineError::Verification {
            name: "generated intrinsic target requirements".to_string(),
            message: format!("invalid generated-intrinsic marker key: {error}"),
            operation: None,
        }
    })?;
    let mut targets = BTreeMap::new();

    fn visit(
        ctx: &Context,
        op_ptr: Ptr<Operation>,
        marker_key: &pliron::identifier::Identifier,
        marker_policy: GeneratedMarkerPolicy,
        targets: &mut BTreeMap<&'static str, &'static GeneratedIntrinsicTarget>,
    ) -> Result<(), PipelineError> {
        use pliron::builtin::attributes::StringAttr;

        let op_name = Operation::get_opid(op_ptr, ctx).to_string();
        let op_ref = op_ptr.deref(ctx);
        let marker_attribute_exists = op_ref.attributes.0.contains_key(marker_key);
        let marker = op_ref
            .attributes
            .get::<StringAttr>(marker_key)
            .cloned()
            .map(String::from);
        let candidates = generated_intrinsic_targets_by_op_name(&op_name).collect::<Vec<_>>();

        match marker {
            Some(marker) => {
                let target = generated_intrinsic_target_by_marker(&marker).ok_or_else(|| {
                    generated_requirement_error(
                        ctx,
                        op_ptr,
                        format!(
                            "operation `{op_name}` carries unknown generated-intrinsic marker `{marker}`"
                        ),
                    )
                })?;
                if target.dialect_op != op_name {
                    return Err(generated_requirement_error(
                        ctx,
                        op_ptr,
                        format!(
                            "generated-intrinsic marker `{marker}` belongs to `{}`, not `{op_name}`",
                            target.dialect_op
                        ),
                    ));
                }
                if !generated_intrinsic_operation_matches(ctx, target, op_ptr) {
                    return Err(generated_requirement_error(
                        ctx,
                        op_ptr,
                        format!(
                            "generated-intrinsic marker `{marker}` does not match the exact variant attributes on `{op_name}`"
                        ),
                    ));
                }
                targets.entry(target.marker).or_insert(target);
            }
            None if marker_attribute_exists => {
                return Err(generated_requirement_error(
                    ctx,
                    op_ptr,
                    format!(
                        "operation `{op_name}` has a non-string `{GENERATED_INTRINSIC_MARKER_ATTR}` attribute"
                    ),
                ));
            }
            None if !candidates.is_empty() && marker_policy == GeneratedMarkerPolicy::Required => {
                return Err(generated_requirement_error(
                    ctx,
                    op_ptr,
                    format!(
                        "generated intrinsic operation `{op_name}` is missing its exact ABI marker"
                    ),
                ));
            }
            None if !candidates.is_empty() => {
                let matching = candidates
                    .into_iter()
                    .filter(|target| generated_intrinsic_operation_matches(ctx, target, op_ptr))
                    .collect::<Vec<_>>();
                let [target] = matching.as_slice() else {
                    return Err(generated_requirement_error(
                        ctx,
                        op_ptr,
                        format!(
                            "direct dialect operation `{op_name}` matches {} generated catalog variants; expected exactly one",
                            matching.len()
                        ),
                    ));
                };
                targets.entry(target.marker).or_insert(*target);
            }
            None => {}
        }

        for region in op_ref.regions() {
            let region_ref = region.deref(ctx);
            for block in region_ref.iter(ctx) {
                let block_ref = block.deref(ctx);
                for child_op in block_ref.iter(ctx) {
                    visit(ctx, child_op, marker_key, marker_policy, targets)?;
                }
            }
        }
        Ok(())
    }

    visit(ctx, root, &marker_key, marker_policy, &mut targets)?;
    Ok(GeneratedModuleRequirements {
        targets: targets.into_values().collect(),
        ..Default::default()
    })
}

fn generated_requirement_error(
    ctx: &Context,
    op: Ptr<Operation>,
    message: String,
) -> PipelineError {
    PipelineError::Verification {
        name: "generated intrinsic target requirements".to_string(),
        message,
        operation: Some(op.deref(ctx).disp(ctx).to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generated_intrinsic_targets::GENERATED_INTRINSIC_MARKER_ATTR;
    use pliron::builtin::attributes::{StringAttr, TypeAttr};
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::identifier::Identifier;
    use pliron::op::Op;

    fn register_dialects(ctx: &mut Context) {
        dialect_mir::register(ctx);
        dialect_nvvm::register(ctx);
    }

    fn generated_tid_x_op(ctx: &mut Context, marker: Option<&str>) -> Ptr<Operation> {
        let result_type = IntegerType::get(ctx, 32, Signedness::Unsigned).to_handle();
        let op = Operation::new(
            ctx,
            dialect_nvvm::ops::ReadPtxSregTidXOp::get_concrete_op_info(),
            vec![result_type],
            vec![],
            vec![],
            0,
        );
        if let Some(marker) = marker {
            op.deref_mut(ctx).attributes.set(
                Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
                StringAttr::new(marker.to_string()),
            );
        }
        op
    }

    fn generated_test_module(ctx: &mut Context, ops: &[Ptr<Operation>]) -> Ptr<Operation> {
        let module = pliron::builtin::ops::ModuleOp::new(ctx, "test".try_into().unwrap());
        let module_op = module.get_operation();
        for op in ops {
            crate::lower::append_to_module(ctx, module_op, *op);
        }
        module_op
    }

    #[test]
    fn required_markers_are_recursive_and_deduplicated() {
        let mut ctx = Context::new();
        register_dialects(&mut ctx);
        let first = generated_tid_x_op(&mut ctx, Some("v1:i0001"));
        let second = generated_tid_x_op(&mut ctx, Some("v1:i0001"));
        let module = generated_test_module(&mut ctx, &[first, second]);

        let requirements =
            collect_generated_intrinsic_requirements(&ctx, module, GeneratedMarkerPolicy::Required)
                .unwrap();
        assert_eq!(requirements.targets.len(), 1);
        assert_eq!(requirements.targets[0].marker, "v1:i0001");
        assert_eq!(requirements.targets[0].id, "thread_idx_x");
    }

    #[test]
    fn rust_source_requires_a_marker_but_direct_dialect_input_can_derive_it() {
        let mut ctx = Context::new();
        register_dialects(&mut ctx);
        let op = generated_tid_x_op(&mut ctx, None);

        let error =
            collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Required)
                .unwrap_err()
                .to_string();
        assert!(error.contains("missing its exact ABI marker"), "{error}");

        let requirements =
            collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Optional)
                .unwrap();
        assert_eq!(requirements.targets[0].marker, "v1:i0001");
    }

    #[test]
    fn marker_validation_rejects_non_string_unknown_and_wrong_operation_ids() {
        let mut ctx = Context::new();
        register_dialects(&mut ctx);

        let non_string = generated_tid_x_op(&mut ctx, None);
        let i32_type = IntegerType::get(&ctx, 32, Signedness::Unsigned).to_handle();
        non_string.deref_mut(&ctx).attributes.set(
            Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
            TypeAttr::new(i32_type),
        );
        let error = collect_generated_intrinsic_requirements(
            &ctx,
            non_string,
            GeneratedMarkerPolicy::Required,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("has a non-string"), "{error}");

        let unknown = generated_tid_x_op(&mut ctx, Some("v1:i9999"));
        let error = collect_generated_intrinsic_requirements(
            &ctx,
            unknown,
            GeneratedMarkerPolicy::Required,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("unknown generated-intrinsic marker `v1:i9999`"));

        let mismatch = generated_tid_x_op(&mut ctx, Some("v1:i0002"));
        let error = collect_generated_intrinsic_requirements(
            &ctx,
            mismatch,
            GeneratedMarkerPolicy::Required,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("marker `v1:i0002` belongs to `nvvm.read_ptx_sreg_ctaid_x`"));
        assert!(error.contains("not `nvvm.read_ptx_sreg_tid_x`"));
    }

    #[test]
    fn direct_dialect_input_selects_exact_ldmatrix_variant() {
        use dialect_mir::types::MirPtrType;
        use dialect_nvvm::ops::{
            LdmatrixElementAttr, LdmatrixLayoutAttr, LdmatrixMultiplicityAttr, LdmatrixOp,
            LdmatrixShapeAttr, LdmatrixStateSpaceAttr,
        };
        use pliron::basic_block::BasicBlock;

        let mut ctx = Context::new();
        register_dialects(&mut ctx);
        let u32_ty = IntegerType::get(&ctx, 32, Signedness::Unsigned);
        let pointer_ty = MirPtrType::get_shared(&mut ctx, u32_ty.into(), false);
        let block = BasicBlock::new(&mut ctx, None, vec![pointer_ty.into()]);
        let pointer = block.deref(&ctx).get_argument(0);
        let op = LdmatrixOp::build(
            &mut ctx,
            pointer,
            LdmatrixShapeAttr::M8n8,
            LdmatrixMultiplicityAttr::X2,
            LdmatrixLayoutAttr::Normal,
            LdmatrixElementAttr::B16,
            LdmatrixStateSpaceAttr::Shared,
        );

        let requirements =
            collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Optional)
                .unwrap();
        assert_eq!(requirements.targets.len(), 1);
        assert_eq!(requirements.targets[0].id, "ldmatrix_m8n8_x2_b16");

        op.deref_mut(&ctx).attributes.set(
            Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
            StringAttr::new("v1:i0013".to_string()),
        );
        let error =
            collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Required)
                .unwrap_err()
                .to_string();
        assert!(
            error.contains("does not match the exact variant attributes"),
            "{error}"
        );
    }

    #[test]
    fn register_mma_markers_and_attributes_select_one_exact_variant() {
        use dialect_nvvm::ops::{
            RegisterMmaAccumulatorAttr, RegisterMmaElementAttr, RegisterMmaLayoutAttr,
            RegisterMmaOp, RegisterMmaOperationAttr, RegisterMmaOverflowAttr, RegisterMmaShapeAttr,
        };
        use pliron::basic_block::BasicBlock;
        use pliron::builtin::types::FP32Type;

        fn register_mma(
            ctx: &mut Context,
            element: RegisterMmaElementAttr,
            set_operation: bool,
            marker: Option<&str>,
        ) -> Ptr<Operation> {
            let f32_ty = FP32Type::get(ctx);
            let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
            let argument_types = (0..4)
                .map(|_| f32_ty.into())
                .chain((0..6).map(|_| u32_ty.into()))
                .collect();
            let block = BasicBlock::new(ctx, None, argument_types);
            let operands = (0..10)
                .map(|index| block.deref(ctx).get_argument(index))
                .collect();
            let operation = Operation::new(
                ctx,
                RegisterMmaOp::get_concrete_op_info(),
                vec![f32_ty.into(); 4],
                operands,
                vec![],
                0,
            );
            let mma = RegisterMmaOp::new(operation);
            mma.set_attr_nvvm_register_mma_shape(ctx, RegisterMmaShapeAttr::M16n8k16);
            if set_operation {
                mma.set_attr_nvvm_register_mma_operation(ctx, RegisterMmaOperationAttr::Multiply);
            }
            mma.set_attr_nvvm_register_mma_accumulator(ctx, RegisterMmaAccumulatorAttr::F32);
            mma.set_attr_nvvm_register_mma_a_element(ctx, element.clone());
            mma.set_attr_nvvm_register_mma_b_element(ctx, element);
            mma.set_attr_nvvm_register_mma_a_layout(ctx, RegisterMmaLayoutAttr::Row);
            mma.set_attr_nvvm_register_mma_b_layout(ctx, RegisterMmaLayoutAttr::Col);
            mma.set_attr_nvvm_register_mma_overflow(ctx, RegisterMmaOverflowAttr::NotApplicable);
            if let Some(marker) = marker {
                operation.deref_mut(ctx).attributes.set(
                    Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
                    StringAttr::new(marker.to_string()),
                );
            }
            operation
        }

        let mut ctx = Context::new();
        register_dialects(&mut ctx);

        let bf16 = register_mma(&mut ctx, RegisterMmaElementAttr::Bf16, false, None);
        let requirements =
            collect_generated_intrinsic_requirements(&ctx, bf16, GeneratedMarkerPolicy::Optional)
                .unwrap();
        assert_eq!(requirements.targets.len(), 1);
        assert_eq!(requirements.targets[0].marker, "v1:i0105");

        bf16.deref_mut(&ctx).attributes.set(
            Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
            StringAttr::new("v1:i0106".to_string()),
        );
        let error =
            collect_generated_intrinsic_requirements(&ctx, bf16, GeneratedMarkerPolicy::Required)
                .unwrap_err()
                .to_string();
        assert!(
            error.contains("does not match the exact variant attributes"),
            "{error}"
        );

        let f16 = register_mma(
            &mut ctx,
            RegisterMmaElementAttr::F16,
            true,
            Some("v1:i0106"),
        );
        let requirements =
            collect_generated_intrinsic_requirements(&ctx, f16, GeneratedMarkerPolicy::Required)
                .unwrap();
        assert_eq!(requirements.targets.len(), 1);
        assert_eq!(requirements.targets[0].marker, "v1:i0106");

        let crossed = register_mma(&mut ctx, RegisterMmaElementAttr::F16, true, None);
        RegisterMmaOp::new(crossed)
            .set_attr_nvvm_register_mma_b_element(&ctx, RegisterMmaElementAttr::Bf16);
        let error = collect_generated_intrinsic_requirements(
            &ctx,
            crossed,
            GeneratedMarkerPolicy::Optional,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("matches 0 generated catalog variants"),
            "{error}"
        );
    }

    #[test]
    fn integer_register_mma_markers_require_exact_variant_attributes() {
        use dialect_nvvm::ops::{
            RegisterMmaAccumulatorAttr, RegisterMmaElementAttr, RegisterMmaLayoutAttr,
            RegisterMmaOp, RegisterMmaOperationAttr, RegisterMmaOverflowAttr, RegisterMmaShapeAttr,
        };
        use pliron::basic_block::BasicBlock;

        fn register_mma(
            ctx: &mut Context,
            shape: RegisterMmaShapeAttr,
            a_element: RegisterMmaElementAttr,
            b_element: RegisterMmaElementAttr,
            overflow: RegisterMmaOverflowAttr,
            marker: &str,
        ) -> Ptr<Operation> {
            let (a_count, b_count) = match &shape {
                RegisterMmaShapeAttr::M16n8k16 => (2, 1),
                RegisterMmaShapeAttr::M16n8k32 => (4, 2),
                _ => panic!("unsupported integer MMA shape"),
            };
            let i32_ty = IntegerType::get(ctx, 32, Signedness::Signed);
            let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
            let argument_types = (0..4)
                .map(|_| i32_ty.into())
                .chain((0..a_count + b_count).map(|_| u32_ty.into()))
                .collect();
            let block = BasicBlock::new(ctx, None, argument_types);
            let operands = (0..4 + a_count + b_count)
                .map(|index| block.deref(ctx).get_argument(index))
                .collect();
            let operation = Operation::new(
                ctx,
                RegisterMmaOp::get_concrete_op_info(),
                vec![i32_ty.into(); 4],
                operands,
                vec![],
                0,
            );
            let mma = RegisterMmaOp::new(operation);
            mma.set_attr_nvvm_register_mma_shape(ctx, shape);
            mma.set_attr_nvvm_register_mma_operation(ctx, RegisterMmaOperationAttr::Multiply);
            mma.set_attr_nvvm_register_mma_accumulator(ctx, RegisterMmaAccumulatorAttr::S32);
            mma.set_attr_nvvm_register_mma_a_element(ctx, a_element);
            mma.set_attr_nvvm_register_mma_b_element(ctx, b_element);
            mma.set_attr_nvvm_register_mma_a_layout(ctx, RegisterMmaLayoutAttr::Row);
            mma.set_attr_nvvm_register_mma_b_layout(ctx, RegisterMmaLayoutAttr::Col);
            mma.set_attr_nvvm_register_mma_overflow(ctx, overflow);
            operation.deref_mut(ctx).attributes.set(
                Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
                StringAttr::new(marker.to_string()),
            );
            operation
        }

        fn require_marker(ctx: &Context, op: Ptr<Operation>, marker: &str, id: &str) {
            let requirements =
                collect_generated_intrinsic_requirements(ctx, op, GeneratedMarkerPolicy::Required)
                    .unwrap();
            assert_eq!(requirements.targets.len(), 1);
            assert_eq!(requirements.targets[0].marker, marker);
            assert_eq!(requirements.targets[0].id, id);
        }

        fn reject_marker(ctx: &Context, op: Ptr<Operation>) {
            let error =
                collect_generated_intrinsic_requirements(ctx, op, GeneratedMarkerPolicy::Required)
                    .unwrap_err()
                    .to_string();
            assert!(
                error.contains("does not match the exact variant attributes"),
                "{error}"
            );
        }

        let mut ctx = Context::new();
        register_dialects(&mut ctx);

        let k16_satfinite = register_mma(
            &mut ctx,
            RegisterMmaShapeAttr::M16n8k16,
            RegisterMmaElementAttr::S8,
            RegisterMmaElementAttr::U8,
            RegisterMmaOverflowAttr::Satfinite,
            "v1:i0118",
        );
        require_marker(
            &ctx,
            k16_satfinite,
            "v1:i0118",
            "mma_m16n8k16_s32_s8_u8_satfinite",
        );

        let k32_wrapping = register_mma(
            &mut ctx,
            RegisterMmaShapeAttr::M16n8k32,
            RegisterMmaElementAttr::U8,
            RegisterMmaElementAttr::S8,
            RegisterMmaOverflowAttr::Wrapping,
            "v1:i0116",
        );
        require_marker(&ctx, k32_wrapping, "v1:i0116", "mma_m16n8k32_s32_u8_s8");

        let wrong_signedness = register_mma(
            &mut ctx,
            RegisterMmaShapeAttr::M16n8k16,
            RegisterMmaElementAttr::U8,
            RegisterMmaElementAttr::U8,
            RegisterMmaOverflowAttr::Satfinite,
            "v1:i0118",
        );
        reject_marker(&ctx, wrong_signedness);

        let wrong_overflow = register_mma(
            &mut ctx,
            RegisterMmaShapeAttr::M16n8k16,
            RegisterMmaElementAttr::S8,
            RegisterMmaElementAttr::U8,
            RegisterMmaOverflowAttr::Wrapping,
            "v1:i0118",
        );
        reject_marker(&ctx, wrong_overflow);

        let wrong_shape = register_mma(
            &mut ctx,
            RegisterMmaShapeAttr::M16n8k32,
            RegisterMmaElementAttr::S8,
            RegisterMmaElementAttr::U8,
            RegisterMmaOverflowAttr::Satfinite,
            "v1:i0118",
        );
        reject_marker(&ctx, wrong_shape);
    }

    #[test]
    fn b1_register_mma_markers_require_exact_operation_and_shape() {
        use dialect_nvvm::ops::{
            RegisterMmaAccumulatorAttr, RegisterMmaElementAttr, RegisterMmaLayoutAttr,
            RegisterMmaOp, RegisterMmaOperationAttr, RegisterMmaOverflowAttr, RegisterMmaShapeAttr,
        };
        use pliron::basic_block::BasicBlock;

        fn register_mma(
            ctx: &mut Context,
            shape: RegisterMmaShapeAttr,
            operation: Option<RegisterMmaOperationAttr>,
            marker: &str,
        ) -> Ptr<Operation> {
            let (accumulator_count, a_count, b_count, result_count) = match shape {
                RegisterMmaShapeAttr::M8n8k128 => (2, 1, 1, 2),
                RegisterMmaShapeAttr::M16n8k128 => (4, 2, 1, 4),
                RegisterMmaShapeAttr::M16n8k256 => (4, 4, 2, 4),
                _ => panic!("unsupported B1 MMA shape"),
            };
            let i32_ty = IntegerType::get(ctx, 32, Signedness::Signed);
            let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
            let argument_types = (0..accumulator_count)
                .map(|_| i32_ty.into())
                .chain((0..a_count + b_count).map(|_| u32_ty.into()))
                .collect();
            let block = BasicBlock::new(ctx, None, argument_types);
            let operands = (0..accumulator_count + a_count + b_count)
                .map(|index| block.deref(ctx).get_argument(index))
                .collect();
            let op = Operation::new(
                ctx,
                RegisterMmaOp::get_concrete_op_info(),
                vec![i32_ty.into(); result_count],
                operands,
                vec![],
                0,
            );
            let mma = RegisterMmaOp::new(op);
            mma.set_attr_nvvm_register_mma_shape(ctx, shape);
            if let Some(operation) = operation {
                mma.set_attr_nvvm_register_mma_operation(ctx, operation);
            }
            mma.set_attr_nvvm_register_mma_accumulator(ctx, RegisterMmaAccumulatorAttr::S32);
            mma.set_attr_nvvm_register_mma_a_element(ctx, RegisterMmaElementAttr::B1);
            mma.set_attr_nvvm_register_mma_b_element(ctx, RegisterMmaElementAttr::B1);
            mma.set_attr_nvvm_register_mma_a_layout(ctx, RegisterMmaLayoutAttr::Row);
            mma.set_attr_nvvm_register_mma_b_layout(ctx, RegisterMmaLayoutAttr::Col);
            mma.set_attr_nvvm_register_mma_overflow(ctx, RegisterMmaOverflowAttr::Wrapping);
            op.deref_mut(ctx).attributes.set(
                Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
                StringAttr::new(marker.to_string()),
            );
            op
        }

        let mut ctx = Context::new();
        register_dialects(&mut ctx);

        for (shape, operation, marker, id) in [
            (
                RegisterMmaShapeAttr::M8n8k128,
                RegisterMmaOperationAttr::XorPopc,
                "v1:i0157",
                "mma_m8n8k128_s32_b1_xor_popc",
            ),
            (
                RegisterMmaShapeAttr::M16n8k128,
                RegisterMmaOperationAttr::XorPopc,
                "v1:i0158",
                "mma_m16n8k128_s32_b1_xor_popc",
            ),
            (
                RegisterMmaShapeAttr::M16n8k256,
                RegisterMmaOperationAttr::XorPopc,
                "v1:i0159",
                "mma_m16n8k256_s32_b1_xor_popc",
            ),
            (
                RegisterMmaShapeAttr::M8n8k128,
                RegisterMmaOperationAttr::AndPopc,
                "v1:i0160",
                "mma_m8n8k128_s32_b1_and_popc",
            ),
            (
                RegisterMmaShapeAttr::M16n8k128,
                RegisterMmaOperationAttr::AndPopc,
                "v1:i0161",
                "mma_m16n8k128_s32_b1_and_popc",
            ),
            (
                RegisterMmaShapeAttr::M16n8k256,
                RegisterMmaOperationAttr::AndPopc,
                "v1:i0162",
                "mma_m16n8k256_s32_b1_and_popc",
            ),
        ] {
            let op = register_mma(&mut ctx, shape, Some(operation), marker);
            let requirements =
                collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Required)
                    .unwrap();
            assert_eq!(requirements.targets.len(), 1);
            assert_eq!(requirements.targets[0].marker, marker);
            assert_eq!(requirements.targets[0].id, id);
        }

        for op in [
            register_mma(
                &mut ctx,
                RegisterMmaShapeAttr::M8n8k128,
                Some(RegisterMmaOperationAttr::AndPopc),
                "v1:i0157",
            ),
            register_mma(&mut ctx, RegisterMmaShapeAttr::M8n8k128, None, "v1:i0157"),
        ] {
            let error =
                collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Required)
                    .unwrap_err()
                    .to_string();
            assert!(
                error.contains("does not match the exact variant attributes"),
                "{error}"
            );
        }
    }

    #[test]
    fn sparse_mma_markers_require_exact_variant_attributes() {
        use dialect_nvvm::ops::{
            RegisterMmaAccumulatorAttr, RegisterMmaElementAttr, RegisterMmaLayoutAttr,
            RegisterMmaOp, RegisterMmaOperationAttr, RegisterMmaOverflowAttr, RegisterMmaShapeAttr,
            SparseMmaAccumulatorAttr, SparseMmaElementAttr, SparseMmaLayoutAttr,
            SparseMmaMetadataAttr, SparseMmaOp, SparseMmaOverflowAttr, SparseMmaSelectorAttr,
            SparseMmaShapeAttr,
        };
        use pliron::basic_block::BasicBlock;

        fn sparse_mma(
            ctx: &mut Context,
            a_element: SparseMmaElementAttr,
            b_element: SparseMmaElementAttr,
            overflow: SparseMmaOverflowAttr,
            metadata: SparseMmaMetadataAttr,
            marker: Option<&str>,
        ) -> Ptr<Operation> {
            let i32_ty = IntegerType::get(ctx, 32, Signedness::Signed);
            let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
            let argument_types = (0..4)
                .map(|_| i32_ty.into())
                .chain((0..6).map(|_| u32_ty.into()))
                .collect();
            let block = BasicBlock::new(ctx, None, argument_types);
            let operands = (0..10)
                .map(|index| block.deref(ctx).get_argument(index))
                .collect();
            let operation = Operation::new(
                ctx,
                SparseMmaOp::get_concrete_op_info(),
                vec![i32_ty.into(); 4],
                operands,
                vec![],
                0,
            );
            let mma = SparseMmaOp::new(operation);
            mma.set_attr_nvvm_sparse_mma_shape(ctx, SparseMmaShapeAttr::M16n8k32);
            mma.set_attr_nvvm_sparse_mma_accumulator(ctx, SparseMmaAccumulatorAttr::S32);
            mma.set_attr_nvvm_sparse_mma_a_element(ctx, a_element);
            mma.set_attr_nvvm_sparse_mma_b_element(ctx, b_element);
            mma.set_attr_nvvm_sparse_mma_a_layout(ctx, SparseMmaLayoutAttr::Row);
            mma.set_attr_nvvm_sparse_mma_b_layout(ctx, SparseMmaLayoutAttr::Col);
            mma.set_attr_nvvm_sparse_mma_overflow(ctx, overflow);
            mma.set_attr_nvvm_sparse_mma_metadata(ctx, metadata);
            mma.set_attr_nvvm_sparse_mma_selector(ctx, SparseMmaSelectorAttr::ImmediateZeroOrOne);
            if let Some(marker) = marker {
                operation.deref_mut(ctx).attributes.set(
                    Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
                    StringAttr::new(marker.to_string()),
                );
            }
            operation
        }

        fn dense_mma_with_sparse_marker(ctx: &mut Context, marker: &str) -> Ptr<Operation> {
            let i32_ty = IntegerType::get(ctx, 32, Signedness::Signed);
            let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
            let argument_types = (0..4)
                .map(|_| i32_ty.into())
                .chain((0..6).map(|_| u32_ty.into()))
                .collect();
            let block = BasicBlock::new(ctx, None, argument_types);
            let operands = (0..10)
                .map(|index| block.deref(ctx).get_argument(index))
                .collect();
            let operation = Operation::new(
                ctx,
                RegisterMmaOp::get_concrete_op_info(),
                vec![i32_ty.into(); 4],
                operands,
                vec![],
                0,
            );
            let mma = RegisterMmaOp::new(operation);
            mma.set_attr_nvvm_register_mma_shape(ctx, RegisterMmaShapeAttr::M16n8k32);
            mma.set_attr_nvvm_register_mma_operation(ctx, RegisterMmaOperationAttr::Multiply);
            mma.set_attr_nvvm_register_mma_accumulator(ctx, RegisterMmaAccumulatorAttr::S32);
            mma.set_attr_nvvm_register_mma_a_element(ctx, RegisterMmaElementAttr::S8);
            mma.set_attr_nvvm_register_mma_b_element(ctx, RegisterMmaElementAttr::S8);
            mma.set_attr_nvvm_register_mma_a_layout(ctx, RegisterMmaLayoutAttr::Row);
            mma.set_attr_nvvm_register_mma_b_layout(ctx, RegisterMmaLayoutAttr::Col);
            mma.set_attr_nvvm_register_mma_overflow(ctx, RegisterMmaOverflowAttr::Wrapping);
            operation.deref_mut(ctx).attributes.set(
                Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
                StringAttr::new(marker.to_string()),
            );
            operation
        }

        fn require_marker(ctx: &Context, op: Ptr<Operation>, marker: &str, id: &str) {
            let requirements =
                collect_generated_intrinsic_requirements(ctx, op, GeneratedMarkerPolicy::Required)
                    .unwrap();
            assert_eq!(requirements.targets.len(), 1);
            assert_eq!(requirements.targets[0].marker, marker);
            assert_eq!(requirements.targets[0].id, id);
        }

        fn reject_marker(ctx: &Context, op: Ptr<Operation>) {
            let error =
                collect_generated_intrinsic_requirements(ctx, op, GeneratedMarkerPolicy::Required)
                    .unwrap_err()
                    .to_string();
            assert!(
                error.contains("does not match the exact variant attributes"),
                "{error}"
            );
        }

        let mut ctx = Context::new();
        register_dialects(&mut ctx);
        for (a_element, b_element, overflow, metadata, marker, id) in [
            (
                SparseMmaElementAttr::S8,
                SparseMmaElementAttr::S8,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Standard,
                "v1:i0163",
                "mma_sp_m16n8k32_s32_s8",
            ),
            (
                SparseMmaElementAttr::S8,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Standard,
                "v1:i0164",
                "mma_sp_m16n8k32_s32_s8_u8",
            ),
            (
                SparseMmaElementAttr::U8,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Standard,
                "v1:i0165",
                "mma_sp_m16n8k32_s32_u8",
            ),
            (
                SparseMmaElementAttr::U8,
                SparseMmaElementAttr::S8,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Standard,
                "v1:i0166",
                "mma_sp_m16n8k32_s32_u8_s8",
            ),
            (
                SparseMmaElementAttr::S8,
                SparseMmaElementAttr::S8,
                SparseMmaOverflowAttr::Satfinite,
                SparseMmaMetadataAttr::Standard,
                "v1:i0167",
                "mma_sp_m16n8k32_s32_s8_satfinite",
            ),
            (
                SparseMmaElementAttr::S8,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Satfinite,
                SparseMmaMetadataAttr::Standard,
                "v1:i0168",
                "mma_sp_m16n8k32_s32_s8_u8_satfinite",
            ),
            (
                SparseMmaElementAttr::U8,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Satfinite,
                SparseMmaMetadataAttr::Standard,
                "v1:i0169",
                "mma_sp_m16n8k32_s32_u8_satfinite",
            ),
            (
                SparseMmaElementAttr::U8,
                SparseMmaElementAttr::S8,
                SparseMmaOverflowAttr::Satfinite,
                SparseMmaMetadataAttr::Standard,
                "v1:i0170",
                "mma_sp_m16n8k32_s32_u8_s8_satfinite",
            ),
            (
                SparseMmaElementAttr::S8,
                SparseMmaElementAttr::S8,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Ordered,
                "v1:i0171",
                "mma_sp_ordered_metadata_m16n8k32_s32_s8",
            ),
            (
                SparseMmaElementAttr::S8,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Ordered,
                "v1:i0172",
                "mma_sp_ordered_metadata_m16n8k32_s32_s8_u8",
            ),
            (
                SparseMmaElementAttr::U8,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Ordered,
                "v1:i0173",
                "mma_sp_ordered_metadata_m16n8k32_s32_u8",
            ),
            (
                SparseMmaElementAttr::U8,
                SparseMmaElementAttr::S8,
                SparseMmaOverflowAttr::Wrapping,
                SparseMmaMetadataAttr::Ordered,
                "v1:i0174",
                "mma_sp_ordered_metadata_m16n8k32_s32_u8_s8",
            ),
            (
                SparseMmaElementAttr::S8,
                SparseMmaElementAttr::S8,
                SparseMmaOverflowAttr::Satfinite,
                SparseMmaMetadataAttr::Ordered,
                "v1:i0175",
                "mma_sp_ordered_metadata_m16n8k32_s32_s8_satfinite",
            ),
            (
                SparseMmaElementAttr::S8,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Satfinite,
                SparseMmaMetadataAttr::Ordered,
                "v1:i0176",
                "mma_sp_ordered_metadata_m16n8k32_s32_s8_u8_satfinite",
            ),
            (
                SparseMmaElementAttr::U8,
                SparseMmaElementAttr::U8,
                SparseMmaOverflowAttr::Satfinite,
                SparseMmaMetadataAttr::Ordered,
                "v1:i0177",
                "mma_sp_ordered_metadata_m16n8k32_s32_u8_satfinite",
            ),
            (
                SparseMmaElementAttr::U8,
                SparseMmaElementAttr::S8,
                SparseMmaOverflowAttr::Satfinite,
                SparseMmaMetadataAttr::Ordered,
                "v1:i0178",
                "mma_sp_ordered_metadata_m16n8k32_s32_u8_s8_satfinite",
            ),
        ] {
            let op = sparse_mma(
                &mut ctx,
                a_element,
                b_element,
                overflow,
                metadata,
                Some(marker),
            );
            require_marker(&ctx, op, marker, id);
        }

        let structural = sparse_mma(
            &mut ctx,
            SparseMmaElementAttr::U8,
            SparseMmaElementAttr::S8,
            SparseMmaOverflowAttr::Satfinite,
            SparseMmaMetadataAttr::Standard,
            None,
        );
        let requirements = collect_generated_intrinsic_requirements(
            &ctx,
            structural,
            GeneratedMarkerPolicy::Optional,
        )
        .unwrap();
        assert_eq!(requirements.targets.len(), 1);
        assert_eq!(requirements.targets[0].marker, "v1:i0170");

        let wrong_element = sparse_mma(
            &mut ctx,
            SparseMmaElementAttr::S8,
            SparseMmaElementAttr::U8,
            SparseMmaOverflowAttr::Wrapping,
            SparseMmaMetadataAttr::Standard,
            Some("v1:i0163"),
        );
        reject_marker(&ctx, wrong_element);

        let wrong_overflow = sparse_mma(
            &mut ctx,
            SparseMmaElementAttr::S8,
            SparseMmaElementAttr::S8,
            SparseMmaOverflowAttr::Satfinite,
            SparseMmaMetadataAttr::Standard,
            Some("v1:i0163"),
        );
        reject_marker(&ctx, wrong_overflow);

        let standard_with_ordered_marker = sparse_mma(
            &mut ctx,
            SparseMmaElementAttr::S8,
            SparseMmaElementAttr::S8,
            SparseMmaOverflowAttr::Wrapping,
            SparseMmaMetadataAttr::Standard,
            Some("v1:i0171"),
        );
        reject_marker(&ctx, standard_with_ordered_marker);

        let ordered_with_standard_marker = sparse_mma(
            &mut ctx,
            SparseMmaElementAttr::S8,
            SparseMmaElementAttr::S8,
            SparseMmaOverflowAttr::Wrapping,
            SparseMmaMetadataAttr::Ordered,
            Some("v1:i0163"),
        );
        reject_marker(&ctx, ordered_with_standard_marker);

        let wrong_layout = sparse_mma(
            &mut ctx,
            SparseMmaElementAttr::S8,
            SparseMmaElementAttr::S8,
            SparseMmaOverflowAttr::Wrapping,
            SparseMmaMetadataAttr::Standard,
            Some("v1:i0163"),
        );
        SparseMmaOp::new(wrong_layout)
            .set_attr_nvvm_sparse_mma_b_layout(&ctx, SparseMmaLayoutAttr::Row);
        reject_marker(&ctx, wrong_layout);

        let sparse_with_dense_marker = sparse_mma(
            &mut ctx,
            SparseMmaElementAttr::U8,
            SparseMmaElementAttr::S8,
            SparseMmaOverflowAttr::Wrapping,
            SparseMmaMetadataAttr::Standard,
            Some("v1:i0116"),
        );
        let error = collect_generated_intrinsic_requirements(
            &ctx,
            sparse_with_dense_marker,
            GeneratedMarkerPolicy::Required,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("belongs to `nvvm.register_mma`"), "{error}");
        let dense_with_sparse_marker = dense_mma_with_sparse_marker(&mut ctx, "v1:i0163");
        let error = collect_generated_intrinsic_requirements(
            &ctx,
            dense_with_sparse_marker,
            GeneratedMarkerPolicy::Required,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("belongs to `nvvm.sparse_mma`"), "{error}");
    }

    #[test]
    fn sparse_mma_targets_require_metadata_specific_ptx_and_ampere() {
        use crate::generated_intrinsic_targets::{
            GeneratedHardwareAlternative, GeneratedHardwareTarget,
        };

        for (marker, minimum_ptx) in [
            ("v1:i0163", 71),
            ("v1:i0164", 71),
            ("v1:i0165", 71),
            ("v1:i0166", 71),
            ("v1:i0167", 71),
            ("v1:i0168", 71),
            ("v1:i0169", 71),
            ("v1:i0170", 71),
            ("v1:i0171", 85),
            ("v1:i0172", 85),
            ("v1:i0173", 85),
            ("v1:i0174", 85),
            ("v1:i0175", 85),
            ("v1:i0176", 85),
            ("v1:i0177", 85),
            ("v1:i0178", 85),
        ] {
            let target = generated_intrinsic_target_by_marker(marker).unwrap();
            for backend in [
                GeneratedIntrinsicBackend::LlvmNvptx,
                GeneratedIntrinsicBackend::LibNvvm,
            ] {
                let requirement = target.requirement_for_backend(backend);
                assert_eq!(requirement.minimum_ptx.encoded(), minimum_ptx, "{marker}");
                assert_eq!(
                    requirement.hardware,
                    GeneratedHardwareTarget::AnyOf(&[GeneratedHardwareAlternative::MinimumSm(80)]),
                    "{marker}"
                );
            }
        }
    }

    #[test]
    fn integer_register_mma_targets_require_ampere_on_both_backends() {
        use crate::generated_intrinsic_targets::{
            GeneratedHardwareAlternative, GeneratedHardwareTarget,
        };

        for marker in [
            "v1:i0108", "v1:i0110", "v1:i0111", "v1:i0112", "v1:i0113", "v1:i0114", "v1:i0115",
            "v1:i0116", "v1:i0117", "v1:i0118", "v1:i0119", "v1:i0120", "v1:i0121", "v1:i0122",
            "v1:i0123", "v1:i0124",
        ] {
            let target = generated_intrinsic_target_by_marker(marker).unwrap();
            for backend in [
                GeneratedIntrinsicBackend::LlvmNvptx,
                GeneratedIntrinsicBackend::LibNvvm,
            ] {
                let requirement = target.requirement_for_backend(backend);
                assert_eq!(requirement.minimum_ptx.encoded(), 70, "{marker}");
                assert_eq!(
                    requirement.hardware,
                    GeneratedHardwareTarget::AnyOf(&[GeneratedHardwareAlternative::MinimumSm(80)]),
                    "{marker}"
                );
            }
        }
    }

    #[test]
    fn direct_cluster_barrier_arrive_selects_only_i0277() {
        use dialect_nvvm::ops::{ClusterBarrierModeAttr, ClusterBarrierOp};

        let mut ctx = Context::new();
        register_dialects(&mut ctx);
        let op = ClusterBarrierOp::build(&mut ctx, ClusterBarrierModeAttr::Arrive);
        let requirements =
            collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Optional)
                .unwrap();

        assert_eq!(requirements.targets.len(), 1);
        assert_eq!(requirements.targets[0].marker, "v1:i0277");
        assert_eq!(requirements.targets[0].id, "barrier_cluster_arrive");
        assert_eq!(
            requirements
                .requirement(requirements.targets[0])
                .minimum_ptx
                .encoded(),
            78
        );
    }

    #[test]
    fn direct_cluster_barrier_relaxed_selects_only_i0279() {
        use dialect_nvvm::ops::{ClusterBarrierModeAttr, ClusterBarrierOp};

        let mut ctx = Context::new();
        register_dialects(&mut ctx);
        let op = ClusterBarrierOp::build(&mut ctx, ClusterBarrierModeAttr::ArriveRelaxed);
        let requirements =
            collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Optional)
                .unwrap();

        assert_eq!(requirements.targets.len(), 1);
        assert_eq!(requirements.targets[0].marker, "v1:i0279");
        assert_eq!(requirements.targets[0].id, "barrier_cluster_arrive_relaxed");
        assert_eq!(
            requirements
                .requirement(requirements.targets[0])
                .minimum_ptx
                .encoded(),
            80
        );
    }

    #[test]
    fn cluster_barrier_marker_rejects_a_different_mode() {
        use dialect_nvvm::ops::{ClusterBarrierModeAttr, ClusterBarrierOp};

        let mut ctx = Context::new();
        register_dialects(&mut ctx);
        let op = ClusterBarrierOp::build(&mut ctx, ClusterBarrierModeAttr::Arrive);
        op.deref_mut(&ctx).attributes.set(
            Identifier::try_from(GENERATED_INTRINSIC_MARKER_ATTR).unwrap(),
            StringAttr::new("v1:i0279".to_string()),
        );

        let error =
            collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Required)
                .unwrap_err()
                .to_string();
        assert!(
            error.contains("does not match the exact variant attributes"),
            "{error}"
        );
    }
}
