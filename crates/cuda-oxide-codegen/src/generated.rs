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
    fn direct_dialect_input_rejects_an_unlisted_ldmatrix_variant() {
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

        let error =
            collect_generated_intrinsic_requirements(&ctx, op, GeneratedMarkerPolicy::Optional)
                .unwrap_err()
                .to_string();
        assert!(
            error.contains("matches 0 generated catalog variants"),
            "{error}"
        );
    }
}
