/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{
    ActiveMaskAdapter, BackendLoweringMechanism, CatalogFile, ClcOperation, ClusterBarrierMode,
    ClusterMemoryOperation, CpAsyncCachePolicy, CpAsyncControlOperation, CpAsyncMbarrierOperation,
    CpAsyncMbarrierStateSpace, CpAsyncSourceSize, DebugControlOperation, DotProductAdapter,
    ExecutionControlOperation, IntrinsicBackend, MbarrierBasicAdapter, MbarrierBasicOperation,
    MbarrierExtendedAdapter, PackedConversionAdapter, PackedConversionSourceFormat, PrmtAdapter,
    PrmtMode, ReduxAdapter, ScalarArithmeticFormat, ScalarMathFormat, StmatrixLayout,
    Tcgen05Operation, TmaOperation, TmaReductionLoadMode, TmaReductionOperation, VoteAdapter,
    WarpBarrierAdapter, WarpMatchAdapter, WarpShuffleAdapter, WarpShuffleMode,
    WarpShuffleValueKind, WgmmaControlMode,
};
use crate::render::common::rust_header;
use crate::render::families::{
    active_masks, clc_intrinsics, cluster_barrier_attr, cluster_barrier_template, cluster_barriers,
    cluster_memory, cp_async_controls, cp_async_copies, cp_async_mbarriers, debug_controls,
    dot_product_ptx, dot_products, elect_intrinsics, execution_controls, expected_ptx_head,
    extended_minmax, extended_minmax_carrier, extended_minmax_format_attr,
    extended_minmax_nan_attr, extended_minmax_operation_attr, extended_minmax_ptx_mnemonic,
    extended_minmax_subnormal_attr, extended_minmax_xorsign_abs_attr, integer_minmax_ptx_mnemonic,
    integer_minmaxes, ldmatrix, ldmatrix_attr_variants, ldmatrix_compat_op, mbarrier_basics,
    mbarrier_extended, movmatrix, movmatrix_template, packed_alu_ptx_mnemonic, packed_alu_width,
    packed_alus, packed_atomics, packed_conversion_ptx_mnemonic, packed_conversion_result_width,
    packed_conversion_source_width, packed_conversion_typed_llvm_name, packed_conversions, prmts,
    redux, register_mma_attr_variants, register_mma_compat_op_type, register_mma_constraints,
    register_mma_extra_operand_count, register_mma_fragment_counts, register_mma_result_variant,
    register_mma_template, register_mmas, scalar_arithmetic_contract,
    scalar_arithmetic_format_attr, scalar_arithmetic_llvm_mechanism,
    scalar_arithmetic_operation_attr, scalar_arithmetic_ptx_mnemonic,
    scalar_arithmetic_rounding_attr, scalar_arithmetic_saturation_attr,
    scalar_arithmetic_subnormal_attr, scalar_arithmetics, scalar_conversion_ptx_mnemonic,
    scalar_conversion_rounding_attr, scalar_conversion_saturation_attr, scalar_conversions,
    scalar_math_contract, scalar_math_format_attr, scalar_math_llvm_mechanism,
    scalar_math_operation_attr, scalar_math_precision_attr, scalar_math_ptx_mnemonic,
    scalar_math_subnormal_attr, scalar_maths, sparse_mma_attr_variants, sparse_mma_constraints,
    sparse_mma_fragment_counts, sparse_mma_result_variant, sparse_mma_template, sparse_mmas,
    special_register_asm_kind, special_register_backend_mechanism,
    special_register_inline_template, special_register_output_constraint, sregs, stmatrices,
    stmatrix_variant, sync_intrinsics, tcgen05_inline_asm, tcgen05_intrinsics,
    tcgen05_mma_intrinsics, tcgen05_non_mma_intrinsics, threadfence_ptx_level, tma_intrinsics,
    vote_intrinsics, warp_barriers, warp_matches, warp_shuffles, wgmma_control_template,
    wgmma_controls,
};
use std::fmt::Write as _;

pub(super) fn render_lowering(catalog: &CatalogFile, hash: &str) -> String {
    let mut output = rust_header(catalog, hash);
    output.push_str(
        "//! Generated conversion interfaces for admitted CUDA intrinsic families.\n\nuse crate::conversion_interface::MirToLlvmConversion;\nuse crate::convert::intrinsics::{atomic::convert_packed_atom_add, basic::convert_sreg_read_inline, common::{call_intrinsic, create_i32_const, inline_asm_convergent}, cp_async::{convert_generated_cp_async_control, convert_generated_cp_async_copy, convert_generated_cp_async_mbarrier}, dotprod::convert_generated_dot_product, ldmatrix::convert_generated_ldmatrix, mbarrier::{convert_arrive, convert_init, convert_inval, convert_test_wait}, packed::{convert_generated_packed_alu, convert_generated_packed_f32x2, convert_generated_packed_unary}, prmt::convert_generated_prmt, warp::{convert_active_mask, convert_bar_warp_sync, convert_match_all, convert_match_any, convert_redux, convert_shuffle_f32, convert_shuffle_i32, convert_shuffle_i64, convert_vote}, wmma::{convert_generated_register_mma, convert_generated_sparse_mma, GeneratedMmaResultType}};\nuse crate::{context, IntrinsicBackend};\nuse dialect_nvvm::ops::{",
    );
    if debug_controls(catalog).next().is_some() {
        output = output.replace(
            "inline_asm_convergent}",
            "inline_asm_convergent, inline_asm_sideeffect}",
        );
    }
    if integer_minmaxes(catalog).next().is_some() {
        output = output.replace(
            "ldmatrix::convert_generated_ldmatrix, ",
            "integer_minmax::convert_generated_integer_minmax, ldmatrix::convert_generated_ldmatrix, ",
        );
    }
    if stmatrices(catalog).next().is_some() {
        output = output.replace(
            "common::{call_intrinsic,",
            "common::{call_intrinsic, cast_to_shared_addrspace,",
        );
    }
    if clc_intrinsics(catalog).next().is_some() {
        output = output.replace(
            "cp_async::{",
            "clc::{convert_generated_clc_query, convert_generated_clc_try_cancel}, cp_async::{",
        );
    }
    if tma_intrinsics(catalog).next().is_some() {
        output = output.replace(
            "prmt::convert_generated_prmt, warp::{",
            "prmt::convert_generated_prmt, tma::{convert_control, convert_g2s, convert_g2s_multicast_cg2, convert_prefetch_tensormap, convert_prefetch_tile, convert_reduce_s2g, convert_s2g, convert_tensormap_fence, convert_tensormap_replace, PrefetchTileConfig, ReduceConfig}, warp::{",
        );
    }
    if execution_controls(catalog).next().is_some() {
        output = output.replace(
            "dotprod::convert_generated_dot_product, ",
            "dotprod::convert_generated_dot_product, execution_control::{convert_counted_barrier, convert_grid_dependency, convert_setmaxnreg}, ",
        );
    }
    if scalar_conversions(catalog).next().is_some() {
        output = output.replace(
            "prmt::convert_generated_prmt, ",
            "prmt::convert_generated_prmt, scalar_conversion::convert_generated_scalar_conversion, ",
        );
    }
    if scalar_arithmetics(catalog).next().is_some() {
        output = output.replace(
            "prmt::convert_generated_prmt, ",
            "prmt::convert_generated_prmt, scalar_arithmetic::convert_generated_scalar_arithmetic, ",
        );
    }
    if scalar_maths(catalog).next().is_some() {
        output = output.replace(
            "prmt::convert_generated_prmt, ",
            "prmt::convert_generated_prmt, scalar_math::convert_generated_scalar_math, ",
        );
    }
    if extended_minmax(catalog).next().is_some() {
        output = output.replace(
            "dotprod::convert_generated_dot_product, ",
            "dotprod::convert_generated_dot_product, extended_minmax::{MinMaxCarrier, convert_generated_extended_minmax}, ",
        );
    }
    if elect_intrinsics(catalog).next().is_some() {
        output = output.replace(
            "warp::{convert_active_mask,",
            "warp::{convert_active_mask, convert_elect_sync_inline, convert_elect_sync_typed,",
        );
    }
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
        for record in ldmatrix(catalog) {
            if let Some((op_type, _)) = ldmatrix_compat_op(record) {
                output.push_str(", ");
                output.push_str(op_type);
            }
        }
    }
    if let Some(record) = movmatrix(catalog).next() {
        output.push_str(", ");
        output.push_str(&record.dialect.op_type);
    }
    if register_mmas(catalog).next().is_some() {
        for record in
            register_mmas(catalog).filter(|record| register_mma_compat_op_type(record).is_some())
        {
            output.push_str(", ");
            output.push_str(register_mma_compat_op_type(record).unwrap());
        }
        output.push_str(", RegisterMmaAccumulatorAttr, RegisterMmaElementAttr, RegisterMmaKindAttr, RegisterMmaLayoutAttr, RegisterMmaOp, RegisterMmaOperationAttr, RegisterMmaOverflowAttr, RegisterMmaShapeAttr");
    }
    if sparse_mmas(catalog).next().is_some() {
        output.push_str(", SparseMmaAccumulatorAttr, SparseMmaElementAttr, SparseMmaLayoutAttr, SparseMmaMetadataAttr, SparseMmaOp, SparseMmaOverflowAttr, SparseMmaSelectorAttr, SparseMmaShapeAttr");
    }
    if prmts(catalog).next().is_some() {
        output.push_str(", PrmtModeAttr, PrmtOp");
    }
    if scalar_conversions(catalog).next().is_some() {
        output.push_str(
            ", ScalarConversionOp, ScalarConversionRoundingAttr, ScalarConversionSaturationAttr",
        );
    }
    if scalar_arithmetics(catalog).next().is_some() {
        output.push_str(
            ", ScalarArithmeticFormatAttr, ScalarArithmeticOp, ScalarArithmeticOperationAttr, ScalarArithmeticRoundingAttr, ScalarArithmeticSaturationAttr, ScalarArithmeticSubnormalAttr",
        );
    }
    if scalar_maths(catalog).next().is_some() {
        output.push_str(
            ", ScalarMathFormatAttr, ScalarMathOp, ScalarMathOperationAttr, ScalarMathPrecisionAttr, ScalarMathSubnormalAttr",
        );
    }
    if extended_minmax(catalog).next().is_some() {
        output.push_str(
            ", ExtendedMinMaxFormatAttr, ExtendedMinMaxNanAttr, ExtendedMinMaxOp, ExtendedMinMaxOperationAttr, ExtendedMinMaxSubnormalAttr, ExtendedMinMaxXorSignAbsAttr",
        );
    }
    if cluster_barriers(catalog).next().is_some() {
        output.push_str(", ClusterBarrierModeAttr, ClusterBarrierOp, ClusterSyncOp");
    }
    if debug_controls(catalog).next().is_some() {
        output.push_str(", BreakpointOp, PmEventOp, TrapOp");
    }
    for record in clc_intrinsics(catalog) {
        output.push_str(", ");
        output.push_str(&record.dialect.op_type);
    }
    if wgmma_controls(catalog).next().is_some() {
        output.push_str(
            ", WgmmaCommitGroupSyncAlignedOp, WgmmaFenceSyncAlignedOp, WgmmaWaitGroupSyncAlignedOp",
        );
    }
    for record in tcgen05_non_mma_intrinsics(catalog) {
        output.push_str(", ");
        output.push_str(&record.dialect.op_type);
    }
    if tcgen05_mma_intrinsics(catalog).next().is_some() {
        output.push_str(", Tcgen05MmaBBufferAttr, Tcgen05MmaBUsageAttr, Tcgen05MmaCollectorAAttr, Tcgen05MmaCtaGroupAttr, Tcgen05MmaFormAttr, Tcgen05MmaKindAttr, Tcgen05MmaOp");
    }
    if packed_atomics(catalog).next().is_some() {
        if sregs(catalog).next().is_some() || ldmatrix(catalog).next().is_some() {
            output.push_str(", ");
        }
        output.push_str("NvvmAtomAddBf16x2Op, NvvmAtomAddF16x2Op, PackedAtomicAddOp, PackedAtomicAtomicityAttr, PackedAtomicFormatAttr, PackedAtomicOrderingAttr, PackedAtomicRoundingAttr, PackedAtomicScopeAttr, PackedAtomicStateSpaceAttr, PackedAtomicSubnormalAttr");
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
    for record in elect_intrinsics(catalog) {
        output.push_str(", ");
        output.push_str(&record.dialect.op_type);
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
    if integer_minmaxes(catalog).next().is_some() {
        // packed_alu records always exist, so a separator is always needed.
        output.push_str(", ");
        for (index, record) in integer_minmaxes(catalog).enumerate() {
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
    for record in cp_async_copies(catalog)
        .chain(cp_async_controls(catalog))
        .chain(cp_async_mbarriers(catalog))
    {
        output.push_str(", ");
        output.push_str(&record.dialect.op_type);
    }
    for record in mbarrier_basics(catalog) {
        output.push_str(", ");
        output.push_str(&record.dialect.op_type);
    }
    for record in cluster_memory(catalog) {
        output.push_str(", ");
        output.push_str(&record.dialect.op_type);
    }
    for record in stmatrices(catalog) {
        output.push_str(", ");
        output.push_str(&record.dialect.op_type);
    }
    for record in mbarrier_extended(catalog) {
        output.push_str(", ");
        output.push_str(&record.dialect.op_type);
    }
    for record in tma_intrinsics(catalog) {
        output.push_str(", ");
        output.push_str(&record.dialect.op_type);
    }
    for record in execution_controls(catalog) {
        output.push_str(", ");
        output.push_str(&record.dialect.op_type);
    }
    output.push_str(
        "};\nuse llvm_export::{ops::AsmKind, types as llvm_types};\nuse pliron::{\n    builtin::types::{IntegerType, Signedness},\n    context::{Context, Ptr},\n    derive::op_interface_impl,\n    irbuild::{\n        dialect_conversion::{DialectConversionRewriter, OperandsInfo},\n        rewriter::Rewriter,\n    },\n    op::Op,\n    operation::Operation,\n    result::Result,\n};\n\n",
    );
    if cluster_memory(catalog).next().is_some() {
        if !output.contains("cast_to_shared_addrspace") {
            output = output.replace(
                "common::{call_intrinsic, ",
                "common::{call_intrinsic, cast_to_shared_addrspace, ",
            );
        }
        output = output
            .replace(
                "use llvm_export::{ops::AsmKind, types as llvm_types};",
                "use llvm_export::{op_interfaces::CastOpInterface, ops as llvm_ops, ops::AsmKind, types as llvm_types};",
            )
            .replace(
                "dialect_conversion::{DialectConversionRewriter, OperandsInfo},\n        rewriter::Rewriter,",
                "dialect_conversion::{DialectConversionRewriter, OperandsInfo},\n        inserter::Inserter,\n        rewriter::Rewriter,",
            );
    }
    if tcgen05_intrinsics(catalog).next().is_some() {
        output = output.replace(
            "use llvm_export::{ops::AsmKind, types as llvm_types};",
            "use llvm_export::{ops as llvm_ops, ops::AsmKind, types as llvm_types};",
        );
        output = output.replace(
            "builtin::types::{IntegerType, Signedness}",
            "builtin::types::{FP32Type, IntegerType, Signedness}",
        );
        output = output.replace(
            "dialect_conversion::{DialectConversionRewriter, OperandsInfo},\n        rewriter::Rewriter,",
            "dialect_conversion::{DialectConversionRewriter, OperandsInfo},\n        inserter::Inserter,\n        rewriter::Rewriter,",
        );
    }
    output.push_str("use pliron::location::Located;\n\n");
    if mbarrier_extended(catalog).next().is_some() {
        if !output.contains("cast_to_shared_addrspace") {
            output = output.replace(
                "common::{call_intrinsic,",
                "common::{call_intrinsic, cast_to_shared_addrspace,",
            );
        }
        output = if output.contains("inline_asm_sideeffect}") {
            output.replace(
                "inline_asm_sideeffect}",
                "inline_asm_sideeffect, trunc_to_i1}",
            )
        } else {
            output.replace(
                "inline_asm_convergent}",
                "inline_asm_convergent, trunc_to_i1}",
            )
        };
        output.push_str("use pliron::value::DefiningEntity;\n\n");
    }
    output.push_str(
        "fn convert_zero_operand_scalar_direct(\n    ctx: &mut Context,\n    rewriter: &mut DialectConversionRewriter,\n    op: Ptr<Operation>,\n    width: u32,\n    intrinsic_name: &str,\n) -> Result<()> {\n    let result_ty = IntegerType::get(ctx, width, Signedness::Signless);\n    let function_ty = llvm_types::FuncType::get(ctx, result_ty.into(), vec![], false);\n    let call = call_intrinsic(ctx, rewriter, op, intrinsic_name, function_ty, vec![])?;\n    rewriter.replace_operation(ctx, op, call);\n    Ok(())\n}\n\n",
    );
    if tcgen05_intrinsics(catalog).next().is_some() {
        output.push_str(
            r#"fn convert_generated_tcgen05_void(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    arity: usize,
    template: &str,
    constraints: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != arity {
        return pliron::input_err_noloc!(
            "generated tcgen05 operation expects {arity} operands, got {}",
            operands.len()
        );
    }
    let void_ty = llvm_types::VoidType::get(ctx);
    inline_asm_convergent(
        ctx,
        rewriter,
        op,
        void_ty.into(),
        operands,
        template,
        constraints,
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn convert_generated_tcgen05_load(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    arity: usize,
    count: usize,
    integer_results: bool,
    template: &str,
    constraints: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != arity || !(1..=2).contains(&arity) || !(1..=128).contains(&count) || op.deref(ctx).get_num_results() != count {
        return pliron::input_err_noloc!(
            "generated tcgen05 load requires 1..=2 operands and 1..=128 results"
        );
    }
    let scalar_ty: pliron::r#type::TypeHandle = if integer_results {
        IntegerType::get(ctx, 32, Signedness::Signless).into()
    } else {
        FP32Type::get(ctx).into()
    };
    if count == 1 {
        let inline_asm = inline_asm_convergent(
            ctx,
            rewriter,
            op,
            scalar_ty,
            operands,
            template,
            constraints,
        );
        let result = inline_asm.deref(ctx).get_result(0);
        rewriter.replace_operation_with_values(ctx, op, vec![result]);
        return Ok(());
    }
    let result_ty = llvm_types::StructType::get_unnamed(
        ctx,
        (
            (0..count).map(|_| scalar_ty).collect(),
            llvm_types::StructLayout::Unpacked,
        ),
    );
    let inline_asm = inline_asm_convergent(
        ctx,
        rewriter,
        op,
        result_ty.into(),
        operands,
        template,
        constraints,
    );
    let result = inline_asm.deref(ctx).get_result(0);
    let mut values = Vec::with_capacity(count);
    for index in 0..count as u32 {
        let extract = llvm_ops::ExtractValueOp::new(ctx, result, vec![index])
            .map_err(|error| pliron::input_error_noloc!("{}", error))?;
        rewriter.insert_operation(ctx, extract.get_operation());
        values.push(extract.get_operation().deref(ctx).get_result(0));
    }
    rewriter.replace_operation_with_values(ctx, op, values);
    Ok(())
}

"#,
        );
    }
    if stmatrices(catalog).next().is_some() {
        output.push_str(
            r#"fn convert_generated_stmatrix(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    register_count: usize,
    transposed: bool,
    typed_intrinsic_name: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != register_count + 1 || !matches!(register_count, 2 | 4) {
        return pliron::input_err_noloc!(
            "stmatrix requires one pointer and two or four registers"
        );
    }
    let void_ty = llvm_types::VoidType::get(ctx);
    match context::lowering_options(ctx).intrinsic_backend {
        IntrinsicBackend::LlvmNvptx => {
            let shared = cast_to_shared_addrspace(ctx, rewriter, operands[0]);
            let shared_ty = llvm_types::PointerType::get(
                ctx,
                llvm_types::address_space::SHARED,
            );
            let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
            let argument_types = std::iter::once(shared_ty.into())
                .chain(std::iter::repeat_n(i32_ty.into(), register_count))
                .collect();
            let function_ty = llvm_types::FuncType::get(
                ctx,
                void_ty.into(),
                argument_types,
                false,
            );
            let arguments = std::iter::once(shared)
                .chain(operands.iter().skip(1).copied())
                .collect();
            call_intrinsic(
                ctx,
                rewriter,
                op,
                typed_intrinsic_name,
                function_ty,
                arguments,
            )?;
        }
        IntrinsicBackend::LibNvvm => {
            let registers = (1..=register_count)
                .map(|index| format!("${index}"))
                .collect::<Vec<_>>()
                .join(", ");
            let trans = if transposed { ".trans" } else { "" };
            let template = format!(
                "{{ .reg .u64 %ptr64; .reg .u32 %ptr32; cvta.to.shared.u64 %ptr64, $0; cvt.u32.u64 %ptr32, %ptr64; stmatrix.sync.aligned.m8n8.x{register_count}{trans}.shared.b16 [%ptr32], {{{registers}}}; }}"
            );
            let constraints = std::iter::once("l")
                .chain(std::iter::repeat_n("r", register_count))
                .chain(std::iter::once("~{memory}"))
                .collect::<Vec<_>>()
                .join(",");
            inline_asm_convergent(
                ctx,
                rewriter,
                op,
                void_ty.into(),
                operands,
                &template,
                &constraints,
            );
        }
    }
    rewriter.erase_operation(ctx, op);
    Ok(())
}

"#,
        );
    }
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
        if record.special_register.is_none() {
            writeln!(
                output,
                "        convert_zero_operand_scalar_direct(ctx, rewriter, self.get_operation(), {}, {:?})",
                record.scalar_width().unwrap(),
                record.llvm_identifier()
            )
            .unwrap();
        } else {
            output.push_str("        match context::lowering_options(ctx).intrinsic_backend {\n");
            for (backend, backend_variant) in [
                (IntrinsicBackend::LlvmNvptx, "LlvmNvptx"),
                (IntrinsicBackend::LibNvvm, "LibNvvm"),
            ] {
                writeln!(
                    output,
                    "            IntrinsicBackend::{backend_variant} => {{"
                )
                .unwrap();
                match special_register_backend_mechanism(record, backend) {
                    BackendLoweringMechanism::TypedNvvm => writeln!(
                        output,
                        "                convert_zero_operand_scalar_direct(ctx, rewriter, self.get_operation(), {}, {:?})",
                        record.scalar_width().unwrap(),
                        record.llvm_identifier()
                    )
                    .unwrap(),
                    BackendLoweringMechanism::InlinePtx => writeln!(
                        output,
                        "                convert_sreg_read_inline(ctx, rewriter, self.get_operation(), {}, {:?}, {:?}, {})",
                        record.scalar_width().unwrap(),
                        special_register_inline_template(record),
                        special_register_output_constraint(record),
                        special_register_asm_kind(record),
                    )
                    .unwrap(),
                }
                output.push_str("            }\n");
            }
            output.push_str("        }\n");
        }
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
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        let op = self.get_operation();\n        match context::lowering_options(ctx).intrinsic_backend {\n            IntrinsicBackend::LlvmNvptx => {\n                convert_active_mask(ctx, rewriter, op, operands_info)\n            }\n            IntrinsicBackend::LibNvvm => {\n                let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);\n                let inline_asm = inline_asm_convergent(\n                    ctx,\n                    rewriter,\n                    op,\n                    i32_ty.into(),\n                    vec![],\n                    \"activemask.b32 $0;\",\n                    \"=r,~{memory}\",\n                );\n                rewriter.replace_operation(ctx, op, inline_asm);\n                Ok(())\n            }\n        }\n    }\n}\n\n",
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
            let register_count = variant.register_count();
            let instruction_head = expected_ptx_head(record);
            let intrinsic_name = record.resolved_llvm_identifier();
            writeln!(
                output,
                "                (Some(&{shape}), Some(&{multiplicity}), Some(&{layout}), Some(&{element}), Some(&{state_space})) => ({register_count}, {instruction_head:?}, {intrinsic_name:?}),"
            )
            .unwrap();
        }
        output.push_str(
            "                _ => return pliron::input_err!(\n                    self.get_operation().deref(ctx).loc(),\n                    \"nvvm.ldmatrix variant has no generated lowering recipe\",\n                ),\n            }\n        };\n        convert_generated_ldmatrix(ctx, rewriter, self.get_operation(), recipe.0, recipe.1, recipe.2)\n    }\n}\n\n",
        );
        for record in ldmatrix(catalog) {
            let Some((op_type, _)) = ldmatrix_compat_op(record) else {
                continue;
            };
            let variant = &record
                .ldmatrix
                .as_ref()
                .expect("ldmatrix compatibility record")
                .variant;
            let register_count = variant.register_count();
            let instruction_head = expected_ptx_head(record);
            let intrinsic_name = record.resolved_llvm_identifier();
            writeln!(
                output,
                "#[op_interface_impl]\nimpl MirToLlvmConversion for {op_type} {{\n    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {{\n        convert_generated_ldmatrix(\n            ctx, rewriter, self.get_operation(), {register_count}, {instruction_head:?}, {intrinsic_name:?},\n        )\n    }}\n}}\n"
            )
            .unwrap();
        }
    }
    for record in stmatrices(catalog) {
        let (multiplicity, layout) = stmatrix_variant(record).expect("stmatrix variant");
        let count = multiplicity.register_count();
        let transposed = layout == StmatrixLayout::Transposed;
        let intrinsic_name = record.resolved_llvm_identifier();
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        writeln!(
            output,
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {{\n        convert_generated_stmatrix(\n            ctx, rewriter, self.get_operation(), {count}, {transposed}, {intrinsic_name:?},\n        )\n    }}\n}}\n"
        )
        .unwrap();
    }
    if let Some(record) = movmatrix(catalog).next() {
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{\n    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {{\n        let op = self.get_operation();\n        let operands: Vec<_> = op.deref(ctx).operands().collect();\n        if operands.len() != 1 {{\n            return pliron::input_err_noloc!(\n                \"movmatrix_trans_b16 requires 1 operand, got {{}}\",\n                operands.len()\n            );\n        }}\n        let result_ty = IntegerType::get(ctx, 32, Signedness::Signless);\n        let asm = inline_asm_convergent(\n            ctx, rewriter, op, result_ty.into(), operands, {:?}, \"=r,r\",\n        );\n        rewriter.replace_operation(ctx, op, asm);\n        Ok(())\n    }}\n}}\n",
            record.dialect.op_type,
            movmatrix_template(record),
        )
        .unwrap();
    }
    if register_mmas(catalog).next().is_some() {
        output.push_str(
            "#[op_interface_impl]\nimpl MirToLlvmConversion for RegisterMmaOp {\n    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        let operation = self.operation_or_multiply(ctx);\n        let kind = self.kind_or_inferred(ctx);\n        let recipe = match (\n            self.get_attr_nvvm_register_mma_shape(ctx).as_deref(),\n            operation,\n            kind,\n            self.get_attr_nvvm_register_mma_accumulator(ctx).as_deref(),\n            self.get_attr_nvvm_register_mma_a_element(ctx).as_deref(),\n            self.get_attr_nvvm_register_mma_b_element(ctx).as_deref(),\n            self.get_attr_nvvm_register_mma_a_layout(ctx).as_deref(),\n            self.get_attr_nvvm_register_mma_b_layout(ctx).as_deref(),\n            self.get_attr_nvvm_register_mma_overflow(ctx).as_deref(),\n        ) {\n",
        );
        for record in register_mmas(catalog) {
            let (
                shape,
                operation,
                kind,
                accumulator,
                a_element,
                b_element,
                a_layout,
                b_layout,
                overflow,
            ) = register_mma_attr_variants(record);
            let (c_count, a_count, b_count, result_count) = register_mma_fragment_counts(record);
            let expected_operands =
                c_count + a_count + b_count + register_mma_extra_operand_count(record);
            writeln!(
                output,
                "            (Some(&{shape}), {operation}, {kind}, Some(&{accumulator}), Some(&{a_element}), Some(&{b_element}), Some(&{a_layout}), Some(&{b_layout}), Some(&{overflow})) => ({}, {result_count}, {expected_operands}, {:?}, {:?}),",
                register_mma_result_variant(record),
                register_mma_template(record),
                register_mma_constraints(record),
            )
            .unwrap();
        }
        output.push_str(
            "            _ => return pliron::input_err!(\n                self.get_operation().deref(ctx).loc(),\n                \"nvvm.register_mma variant has no generated lowering recipe\",\n            ),\n        };\n        convert_generated_register_mma(\n            ctx, rewriter, self.get_operation(), recipe.0, recipe.1, recipe.2, recipe.3, recipe.4,\n        )\n    }\n}\n\n",
        );
        for record in
            register_mmas(catalog).filter(|record| register_mma_compat_op_type(record).is_some())
        {
            let op_type = register_mma_compat_op_type(record).unwrap();
            let (c_count, a_count, b_count, result_count) = register_mma_fragment_counts(record);
            let expected_operands = c_count + a_count + b_count;
            writeln!(
                output,
                "#[op_interface_impl]\nimpl MirToLlvmConversion for {op_type} {{\n    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {{\n        convert_generated_register_mma(\n            ctx, rewriter, self.get_operation(), {}, {result_count}, {expected_operands}, {:?}, {:?},\n        )\n    }}\n}}\n",
                register_mma_result_variant(record),
                register_mma_template(record),
                register_mma_constraints(record),
            )
            .unwrap();
        }
    }
    if sparse_mmas(catalog).next().is_some() {
        output.push_str(
            "#[op_interface_impl]\nimpl MirToLlvmConversion for SparseMmaOp {\n    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        let recipe = match (\n            self.get_attr_nvvm_sparse_mma_shape(ctx).as_deref(),\n            self.get_attr_nvvm_sparse_mma_accumulator(ctx).as_deref(),\n            self.get_attr_nvvm_sparse_mma_a_element(ctx).as_deref(),\n            self.get_attr_nvvm_sparse_mma_b_element(ctx).as_deref(),\n            self.get_attr_nvvm_sparse_mma_a_layout(ctx).as_deref(),\n            self.get_attr_nvvm_sparse_mma_b_layout(ctx).as_deref(),\n            self.get_attr_nvvm_sparse_mma_overflow(ctx).as_deref(),\n            self.get_attr_nvvm_sparse_mma_metadata(ctx).as_deref(),\n            self.get_attr_nvvm_sparse_mma_selector(ctx).as_deref(),\n        ) {\n",
        );
        for record in sparse_mmas(catalog) {
            let (
                shape,
                accumulator,
                a_element,
                b_element,
                a_layout,
                b_layout,
                overflow,
                metadata,
                selector,
            ) = sparse_mma_attr_variants(record);
            let (c_count, a_count, b_count, result_count) = sparse_mma_fragment_counts(record);
            let expected_operands = c_count + a_count + b_count + 2;
            writeln!(
                output,
                "            (Some(&{shape}), Some(&{accumulator}), Some(&{a_element}), Some(&{b_element}), Some(&{a_layout}), Some(&{b_layout}), Some(&{overflow}), Some(&{metadata}), Some(&{selector})) => ({}, {result_count}, {expected_operands}, {:?}, {:?}),",
                sparse_mma_result_variant(record),
                sparse_mma_template(record),
                sparse_mma_constraints(record),
            )
            .unwrap();
        }
        output.push_str(
            "            _ => return pliron::input_err!(\n                self.get_operation().deref(ctx).loc(),\n                \"nvvm.sparse_mma variant has no generated lowering recipe\",\n            ),\n        };\n        convert_generated_sparse_mma(\n            ctx, rewriter, self.get_operation(), recipe.0, recipe.1, recipe.2, recipe.3, recipe.4,\n        )\n    }\n}\n\n",
        );
    }
    if prmts(catalog).next().is_some() {
        output.push_str(
            "#[op_interface_impl]\nimpl MirToLlvmConversion for PrmtOp {\n    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        let recipe = match self.get_attr_nvvm_prmt_mode(ctx).as_deref() {\n",
        );
        for record in prmts(catalog) {
            let prmt = record.prmt.as_ref().unwrap();
            let mode = match prmt.mode {
                PrmtMode::Generic => "PrmtModeAttr::Generic",
                PrmtMode::F4e => "PrmtModeAttr::F4e",
                PrmtMode::B4e => "PrmtModeAttr::B4e",
                PrmtMode::Rc8 => "PrmtModeAttr::Rc8",
                PrmtMode::Ecl => "PrmtModeAttr::Ecl",
                PrmtMode::Ecr => "PrmtModeAttr::Ecr",
                PrmtMode::Rc16 => "PrmtModeAttr::Rc16",
            };
            let modifier = match prmt.mode {
                PrmtMode::Generic => "",
                PrmtMode::F4e => ".f4e",
                PrmtMode::B4e => ".b4e",
                PrmtMode::Rc8 => ".rc8",
                PrmtMode::Ecl => ".ecl",
                PrmtMode::Ecr => ".ecr",
                PrmtMode::Rc16 => ".rc16",
            };
            let template = match prmt.adapter {
                PrmtAdapter::DirectThreeOperands => {
                    format!("prmt.b32{modifier} $0, $1, $2, $3;")
                }
                PrmtAdapter::InsertZeroSecondSource => {
                    format!("prmt.b32{modifier} $0, $1, 0, $2;")
                }
            };
            writeln!(
                output,
                "            Some(&{mode}) => ({:?}, {:?}),",
                record.llvm_identifier(),
                template
            )
            .unwrap();
        }
        output.push_str(
            "            _ => return pliron::input_err!(\n                self.get_operation().deref(ctx).loc(),\n                \"nvvm.prmt mode has no generated lowering recipe\",\n            ),\n        };\n        convert_generated_prmt(ctx, rewriter, self.get_operation(), recipe.0, recipe.1)\n    }\n}\n\n",
        );
    }
    if scalar_conversions(catalog).next().is_some() {
        output.push_str(
            "#[op_interface_impl]\nimpl MirToLlvmConversion for ScalarConversionOp {\n    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        let recipe = match (\n            self.get_attr_nvvm_scalar_conversion_rounding(ctx).as_deref(),\n            self.get_attr_nvvm_scalar_conversion_saturation(ctx).as_deref(),\n        ) {\n",
        );
        for record in scalar_conversions(catalog) {
            writeln!(
                output,
                "            (Some(&{}), Some(&{})) => ({:?}, {:?}),",
                scalar_conversion_rounding_attr(record),
                scalar_conversion_saturation_attr(record),
                record.llvm_identifier(),
                scalar_conversion_ptx_mnemonic(record),
            )
            .unwrap();
        }
        output.push_str(
            "            _ => return pliron::input_err!(\n                self.get_operation().deref(ctx).loc(),\n                \"nvvm.scalar_conversion variant has no generated lowering recipe\",\n            ),\n        };\n        convert_generated_scalar_conversion(\n            ctx, rewriter, self.get_operation(), recipe.0, recipe.1,\n        )\n    }\n}\n\n",
        );
    }
    if scalar_arithmetics(catalog).next().is_some() {
        output.push_str(
            "#[op_interface_impl]\nimpl MirToLlvmConversion for ScalarArithmeticOp {\n    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        let recipe = match (\n            self.get_attr_nvvm_scalar_arithmetic_format(ctx).as_deref(),\n            self.get_attr_nvvm_scalar_arithmetic_operation(ctx).as_deref(),\n            self.get_attr_nvvm_scalar_arithmetic_rounding(ctx).as_deref(),\n            self.get_attr_nvvm_scalar_arithmetic_subnormal(ctx).as_deref(),\n            self.get_attr_nvvm_scalar_arithmetic_saturation(ctx).as_deref(),\n        ) {\n",
        );
        for record in scalar_arithmetics(catalog) {
            writeln!(
                output,
                "            (Some(&{}), Some(&{}), Some(&{}), Some(&{}), Some(&{})) => ({:?}, {:?}, {}, {}),",
                scalar_arithmetic_format_attr(record),
                scalar_arithmetic_operation_attr(record),
                scalar_arithmetic_rounding_attr(record),
                scalar_arithmetic_subnormal_attr(record),
                scalar_arithmetic_saturation_attr(record),
                record.llvm_identifier(),
                scalar_arithmetic_ptx_mnemonic(record),
                scalar_arithmetic_contract(record).format == ScalarArithmeticFormat::F64,
                scalar_arithmetic_llvm_mechanism(record)
                    == BackendLoweringMechanism::InlinePtx,
            )
            .unwrap();
        }
        output.push_str(
            "            _ => return pliron::input_err!(\n                self.get_operation().deref(ctx).loc(),\n                \"nvvm.scalar_arithmetic variant has no generated lowering recipe\",\n            ),\n        };\n        convert_generated_scalar_arithmetic(\n            ctx, rewriter, self.get_operation(), recipe.0, recipe.1, recipe.2, recipe.3,\n        )\n    }\n}\n\n",
        );
    }
    if scalar_maths(catalog).next().is_some() {
        output.push_str(
            "#[op_interface_impl]\nimpl MirToLlvmConversion for ScalarMathOp {\n    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        let recipe = match (\n            self.get_attr_nvvm_scalar_math_format(ctx).as_deref(),\n            self.get_attr_nvvm_scalar_math_operation(ctx).as_deref(),\n            self.get_attr_nvvm_scalar_math_precision(ctx).as_deref(),\n            self.get_attr_nvvm_scalar_math_subnormal(ctx).as_deref(),\n        ) {\n",
        );
        for record in scalar_maths(catalog) {
            // PTX-native records carry no LLVM symbol. The identifier is
            // only consumed by the typed route, which such records never
            // take (their mechanism is always inline PTX), so render an
            // empty name; the lowering helper rejects it defensively.
            let intrinsic_name = if record.llvm.is_some() {
                record.llvm_identifier()
            } else {
                String::new()
            };
            writeln!(
                output,
                "            (Some(&{}), Some(&{}), Some(&{}), Some(&{})) => ({:?}, {:?}, {}, {}, {}),",
                scalar_math_format_attr(record),
                scalar_math_operation_attr(record),
                scalar_math_precision_attr(record),
                scalar_math_subnormal_attr(record),
                intrinsic_name,
                scalar_math_ptx_mnemonic(record),
                scalar_math_contract(record).format == ScalarMathFormat::F16,
                scalar_math_contract(record).format == ScalarMathFormat::F64,
                scalar_math_llvm_mechanism(record) == BackendLoweringMechanism::InlinePtx,
            )
            .unwrap();
        }
        output.push_str(
            "            _ => return pliron::input_err!(\n                self.get_operation().deref(ctx).loc(),\n                \"nvvm.scalar_math variant has no generated lowering recipe\",\n            ),\n        };\n        convert_generated_scalar_math(\n            ctx, rewriter, self.get_operation(), recipe.0, recipe.1, recipe.2, recipe.3, recipe.4,\n        )\n    }\n}\n\n",
        );
    }
    if extended_minmax(catalog).next().is_some() {
        output.push_str(
            "#[op_interface_impl]\nimpl MirToLlvmConversion for ExtendedMinMaxOp {\n    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        let recipe = match (\n            self.get_attr_nvvm_extended_minmax_format(ctx).as_deref(),\n            self.get_attr_nvvm_extended_minmax_operation(ctx).as_deref(),\n            self.get_attr_nvvm_extended_minmax_subnormal(ctx).as_deref(),\n            self.get_attr_nvvm_extended_minmax_nan(ctx).as_deref(),\n            self.get_attr_nvvm_extended_minmax_xorsign_abs(ctx).as_deref(),\n        ) {\n",
        );
        for record in extended_minmax(catalog) {
            writeln!(
                output,
                "            (Some(&{}), Some(&{}), Some(&{}), Some(&{}), Some(&{})) => ({:?}, {}),",
                extended_minmax_format_attr(record),
                extended_minmax_operation_attr(record),
                extended_minmax_subnormal_attr(record),
                extended_minmax_nan_attr(record),
                extended_minmax_xorsign_abs_attr(record),
                extended_minmax_ptx_mnemonic(record),
                extended_minmax_carrier(record),
            )
            .unwrap();
        }
        output.push_str(
            "            _ => return pliron::input_err!(\n                self.get_operation().deref(ctx).loc(),\n                \"nvvm.extended_minmax variant has no generated lowering recipe\",\n            ),\n        };\n        convert_generated_extended_minmax(\n            ctx, rewriter, self.get_operation(), recipe.0, recipe.1,\n        )\n    }\n}\n\n",
        );
    }
    if cluster_barriers(catalog).next().is_some() {
        output.push_str(
            "#[op_interface_impl]\nimpl MirToLlvmConversion for ClusterBarrierOp {\n    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        let recipe = match self.get_attr_nvvm_cluster_barrier_mode(ctx).as_deref() {\n",
        );
        for record in cluster_barriers(catalog) {
            writeln!(
                output,
                "            Some(&{}) => ({:?}, {:?}),",
                cluster_barrier_attr(record),
                record.llvm_identifier(),
                cluster_barrier_template(record)
            )
            .unwrap();
        }
        output.push_str(
            "            _ => return pliron::input_err!(\n                self.get_operation().deref(ctx).loc(),\n                \"nvvm.cluster_barrier mode has no generated lowering recipe\",\n            ),\n        };\n        let op = self.get_operation();\n        let void_ty = llvm_types::VoidType::get(ctx);\n        match context::lowering_options(ctx).intrinsic_backend {\n            IntrinsicBackend::LlvmNvptx => {\n                let function_ty = llvm_types::FuncType::get(ctx, void_ty.into(), vec![], false);\n                call_intrinsic(ctx, rewriter, op, recipe.0, function_ty, vec![])?;\n            }\n            IntrinsicBackend::LibNvvm => {\n                inline_asm_convergent(ctx, rewriter, op, void_ty.into(), vec![], recipe.1, \"~{memory}\");\n            }\n        }\n        rewriter.erase_operation(ctx, op);\n        Ok(())\n    }\n}\n\n",
        );

        let arrive = cluster_barriers(catalog)
            .find(|record| {
                record
                    .cluster_barrier
                    .as_ref()
                    .is_some_and(|barrier| barrier.mode == ClusterBarrierMode::ArriveAligned)
            })
            .expect("aligned cluster arrival");
        let wait = cluster_barriers(catalog)
            .find(|record| {
                record
                    .cluster_barrier
                    .as_ref()
                    .is_some_and(|barrier| barrier.mode == ClusterBarrierMode::WaitAligned)
            })
            .expect("aligned cluster wait");
        output.push_str(
            "#[op_interface_impl]\nimpl MirToLlvmConversion for ClusterSyncOp {\n    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        let op = self.get_operation();\n        let void_ty = llvm_types::VoidType::get(ctx);\n        match context::lowering_options(ctx).intrinsic_backend {\n            IntrinsicBackend::LlvmNvptx => {\n                let function_ty = llvm_types::FuncType::get(ctx, void_ty.into(), vec![], false);\n",
        );
        writeln!(
            output,
            "                call_intrinsic(ctx, rewriter, op, {:?}, function_ty, vec![])?;",
            arrive.llvm_identifier()
        )
        .unwrap();
        writeln!(
            output,
            "                call_intrinsic(ctx, rewriter, op, {:?}, function_ty, vec![])?;",
            wait.llvm_identifier()
        )
        .unwrap();
        output.push_str("            }\n            IntrinsicBackend::LibNvvm => {\n");
        writeln!(
            output,
            "                inline_asm_convergent(ctx, rewriter, op, void_ty.into(), vec![], {:?}, \"~{{memory}}\");",
            format!(
                "{} {}",
                cluster_barrier_template(arrive),
                cluster_barrier_template(wait)
            )
        )
        .unwrap();
        output.push_str(
            "            }\n        }\n        rewriter.erase_operation(ctx, op);\n        Ok(())\n    }\n}\n\n",
        );
    }
    if wgmma_controls(catalog).next().is_some() {
        output.push_str(
            "fn convert_generated_wgmma_control(\n\
                 ctx: &mut Context,\n\
                 rewriter: &mut DialectConversionRewriter,\n\
                 op: Ptr<Operation>,\n\
                 intrinsic_name: &str,\n\
                 ptx: &str,\n\
                 has_immediate: bool,\n\
             ) -> Result<()> {\n\
                 let operands: Vec<_> = op.deref(ctx).operands().collect();\n\
                 if operands.len() != usize::from(has_immediate) {\n\
                     return pliron::input_err!(\n\
                         op.deref(ctx).loc(),\n\
                         \"generated WGMMA control has the wrong operand count\",\n\
                     );\n\
                 }\n\
                 let void_ty = llvm_types::VoidType::get(ctx);\n\
                 match context::lowering_options(ctx).intrinsic_backend {\n\
                     IntrinsicBackend::LlvmNvptx => {\n\
                         let arguments = if has_immediate {\n\
                             vec![IntegerType::get(ctx, 64, Signedness::Signless).into()]\n\
                         } else {\n\
                             vec![]\n\
                         };\n\
                         let function_ty = llvm_types::FuncType::get(\n\
                             ctx, void_ty.into(), arguments, false,\n\
                         );\n\
                         call_intrinsic(\n\
                             ctx, rewriter, op, intrinsic_name, function_ty, operands,\n\
                         )?;\n\
                     }\n\
                     IntrinsicBackend::LibNvvm => {\n\
                         let constraints = if has_immediate {\n\
                             \"n,~{memory}\"\n\
                         } else {\n\
                             \"~{memory}\"\n\
                         };\n\
                         inline_asm_convergent(\n\
                             ctx, rewriter, op, void_ty.into(), operands, ptx, constraints,\n\
                         );\n\
                     }\n\
                 }\n\
                 rewriter.erase_operation(ctx, op);\n\
                 Ok(())\n\
             }\n\n",
        );
        for record in wgmma_controls(catalog) {
            let has_immediate = record
                .wgmma_control
                .as_ref()
                .is_some_and(|control| control.mode == WgmmaControlMode::WaitGroup);
            writeln!(
                output,
                "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{\n    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {{\n        convert_generated_wgmma_control(\n            ctx, rewriter, self.get_operation(), {:?}, {:?}, {has_immediate},\n        )\n    }}\n}}\n",
                record.dialect.op_type,
                record.llvm_identifier(),
                wgmma_control_template(record),
            )
            .unwrap();
        }
    }
    if packed_atomics(catalog).next().is_some() {
        output.push_str(
            "#[op_interface_impl]\nimpl MirToLlvmConversion for PackedAtomicAddOp {\n    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        let format = self.get_attr_nvvm_packed_atomic_format(ctx);\n        let state_space = self.get_attr_nvvm_packed_atomic_state_space(ctx);\n        let ordering = self.get_attr_nvvm_packed_atomic_ordering(ctx);\n        let scope = self.get_attr_nvvm_packed_atomic_scope(ctx);\n        let rounding = self.get_attr_nvvm_packed_atomic_rounding(ctx);\n        let subnormal = self.get_attr_nvvm_packed_atomic_subnormal(ctx);\n        let atomicity = self.get_attr_nvvm_packed_atomic_atomicity(ctx);\n        let ptx_type = match (format.as_deref(), state_space.as_deref(), ordering.as_deref(), scope.as_deref(), rounding.as_deref(), subnormal.as_deref(), atomicity.as_deref()) {\n            (Some(&PackedAtomicFormatAttr::F16x2), Some(&PackedAtomicStateSpaceAttr::Global), Some(&PackedAtomicOrderingAttr::Relaxed), Some(&PackedAtomicScopeAttr::Gpu), Some(&PackedAtomicRoundingAttr::Rn), Some(&PackedAtomicSubnormalAttr::NoFtz), Some(&PackedAtomicAtomicityAttr::PerElement)) => \"f16x2\",\n            (Some(&PackedAtomicFormatAttr::Bf16x2), Some(&PackedAtomicStateSpaceAttr::Global), Some(&PackedAtomicOrderingAttr::Relaxed), Some(&PackedAtomicScopeAttr::Gpu), Some(&PackedAtomicRoundingAttr::Rn), Some(&PackedAtomicSubnormalAttr::NoFtz), Some(&PackedAtomicAtomicityAttr::PerElement)) => \"bf16x2\",\n            _ => return pliron::input_err!(\n                self.get_operation().deref(ctx).loc(),\n                \"nvvm.packed_atomic_add attributes have no generated lowering recipe\",\n            ),\n        };\n        convert_packed_atom_add(ctx, rewriter, self.get_operation(), ptx_type)\n    }\n}\n\n",
        );
        output = output.replace(
            "        convert_packed_atom_add(ctx, rewriter, self.get_operation(), ptx_type)",
            "        drop((format, state_space, ordering, scope, rounding, subnormal, atomicity));\n        convert_packed_atom_add(ctx, rewriter, self.get_operation(), ptx_type)",
        );
        output.push_str(
            "#[op_interface_impl]\nimpl MirToLlvmConversion for NvvmAtomAddF16x2Op {\n    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        convert_packed_atom_add(ctx, rewriter, self.get_operation(), \"f16x2\")\n    }\n}\n\n#[op_interface_impl]\nimpl MirToLlvmConversion for NvvmAtomAddBf16x2Op {\n    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        convert_packed_atom_add(ctx, rewriter, self.get_operation(), \"bf16x2\")\n    }\n}\n\n",
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
    for record in elect_intrinsics(catalog) {
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        match context::lowering_options(ctx).intrinsic_backend {\n",
        );
        for (backend, variant) in [
            (IntrinsicBackend::LlvmNvptx, "LlvmNvptx"),
            (IntrinsicBackend::LibNvvm, "LibNvvm"),
        ] {
            let mechanism = record
                .backend_lowerings
                .iter()
                .find(|route| route.backend == backend)
                .expect("elect backend route")
                .mechanism;
            writeln!(output, "            IntrinsicBackend::{variant} => {{").unwrap();
            match mechanism {
                BackendLoweringMechanism::TypedNvvm => writeln!(
                    output,
                    "                convert_elect_sync_typed(ctx, rewriter, self.get_operation(), operands_info, {:?})",
                    record.llvm_identifier()
                )
                .unwrap(),
                BackendLoweringMechanism::InlinePtx => output.push_str(
                    "                convert_elect_sync_inline(ctx, rewriter, self.get_operation(), operands_info)\n",
                ),
            }
            output.push_str("            }\n");
        }
        output.push_str("        }\n    }\n}\n\n");
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
        let width = packed_alu_width(record);
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
            "        convert_generated_packed_alu(ctx, rewriter, self.get_operation(), {:?}, {width})",
            packed_alu_ptx_mnemonic(record),
        )
        .unwrap();
        output.push_str("    }\n}\n\n");
    }
    for record in integer_minmaxes(catalog) {
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
            "        convert_generated_integer_minmax(ctx, rewriter, self.get_operation(), {:?})",
            integer_minmax_ptx_mnemonic(record)
        )
        .unwrap();
        output.push_str("    }\n}\n\n");
    }
    for record in packed_conversions(catalog) {
        let conversion = record
            .packed_conversion
            .as_ref()
            .expect("packed-conversion record");
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n",
        );
        match conversion.source_format {
            PackedConversionSourceFormat::F32x2 => {
                debug_assert_eq!(
                    conversion.adapter,
                    PackedConversionAdapter::ReverseHighLowOperands
                );
                let typed_intrinsic = packed_conversion_typed_llvm_name(record)
                    .map(|name| format!("Some({name:?})"))
                    .unwrap_or_else(|| "None".into());
                writeln!(
                    output,
                    "        convert_generated_packed_f32x2(ctx, rewriter, self.get_operation(), {typed_intrinsic}, {:?}, {})",
                    packed_conversion_ptx_mnemonic(record),
                    packed_conversion_result_width(record),
                )
                .unwrap();
            }
            PackedConversionSourceFormat::E4m3x2
            | PackedConversionSourceFormat::E5m2x2
            | PackedConversionSourceFormat::F16x2 => {
                debug_assert_eq!(conversion.adapter, PackedConversionAdapter::Identity);
                writeln!(
                    output,
                    "        convert_generated_packed_unary(ctx, rewriter, self.get_operation(), {:?}, {}, {})",
                    packed_conversion_ptx_mnemonic(record),
                    packed_conversion_result_width(record),
                    packed_conversion_source_width(record),
                )
                .unwrap();
            }
        }
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
    for record in cp_async_mbarriers(catalog) {
        let bridge = record.cp_async_mbarrier.as_ref().unwrap();
        let operation = match bridge.operation {
            CpAsyncMbarrierOperation::Arrive => "arrive",
            CpAsyncMbarrierOperation::ArriveNoInc => "arrive_no_inc",
        };
        let state_space = match bridge.state_space {
            CpAsyncMbarrierStateSpace::Generic => "generic",
            CpAsyncMbarrierStateSpace::Shared => "shared",
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
            "        convert_generated_cp_async_mbarrier(ctx, rewriter, self.get_operation(), {operation:?}, {state_space:?}, {:?})",
            record.llvm_identifier(),
        )
        .unwrap();
        output.push_str("    }\n}\n\n");
    }
    for record in mbarrier_basics(catalog) {
        let mbarrier = record.mbarrier_basic.as_ref().unwrap();
        let helper = match (mbarrier.operation, mbarrier.adapter) {
            (MbarrierBasicOperation::Init, MbarrierBasicAdapter::InitPointerCountToVoid) => {
                "convert_init"
            }
            (MbarrierBasicOperation::Arrive, MbarrierBasicAdapter::ArrivePointerToToken) => {
                "convert_arrive"
            }
            (
                MbarrierBasicOperation::TestWait,
                MbarrierBasicAdapter::TestWaitPointerTokenToPredicate,
            ) => "convert_test_wait",
            (MbarrierBasicOperation::Inval, MbarrierBasicAdapter::InvalPointerToVoid) => {
                "convert_inval"
            }
            _ => unreachable!("resolver admitted an invalid basic mbarrier adapter"),
        };
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
            "        {helper}(ctx, rewriter, self.get_operation(), operands_info)"
        )
        .unwrap();
        output.push_str("    }\n}\n\n");
    }
    for record in cluster_memory(catalog) {
        let cluster = record.cluster_memory.as_ref().unwrap();
        let (template, constraints) =
            crate::resolve::cluster_memory_inline_recipe(cluster.operation);
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        let op = self.get_operation();\n        let operands: Vec<_> = op.deref(ctx).operands().collect();\n",
        );
        writeln!(
            output,
            "        if operands.len() != 2 {{\n            return pliron::input_err_noloc!({:?}, operands.len());\n        }}",
            format!("{} requires 2 operands, got {{}}", record.rust.name)
        )
        .unwrap();
        output.push_str(
            "        let shared_pointer = cast_to_shared_addrspace(ctx, rewriter, operands[0]);\n        let rank = operands[1];\n",
        );
        match cluster.operation {
            ClusterMemoryOperation::MapSharedRank => {
                writeln!(
                    output,
                    "        let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);\n        let asm = inline_asm_convergent(\n            ctx, rewriter, op, i64_ty.into(), vec![shared_pointer, rank], {template:?}, {constraints:?},\n        );\n        let mapped = asm.deref(ctx).get_result(0);\n        let cluster_shared_pointer_ty = llvm_types::PointerType::get(ctx, 7);\n        let int_to_ptr = llvm_ops::IntToPtrOp::new(ctx, mapped, cluster_shared_pointer_ty.into());\n        rewriter.insert_operation(ctx, int_to_ptr.get_operation());\n        rewriter.replace_operation(ctx, op, int_to_ptr.get_operation());\n        Ok(())"
                )
                .unwrap();
            }
            ClusterMemoryOperation::ReadU32 => {
                writeln!(
                    output,
                    "        let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);\n        let asm = inline_asm_convergent(\n            ctx, rewriter, op, i32_ty.into(), vec![shared_pointer, rank], {template:?}, {constraints:?},\n        );\n        rewriter.replace_operation(ctx, op, asm);\n        Ok(())"
                )
                .unwrap();
            }
        }
        output.push_str("    }\n}\n\n");
    }
    for record in mbarrier_extended(catalog) {
        let contract = record.mbarrier_extended.as_ref().unwrap();
        let (template, constraints) =
            crate::resolve::mbarrier_extended_inline_recipe(contract.operation);
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        let mutability = if matches!(
            contract.adapter,
            MbarrierExtendedAdapter::PointerTxCountBytesToTokenDroppingTxCount
                | MbarrierExtendedAdapter::PointerTokenToPredicate
                | MbarrierExtendedAdapter::PointerParityToPredicate
        ) {
            "mut "
        } else {
            ""
        };
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        let op = self.get_operation();\n",
        );
        writeln!(
            output,
            "        let {mutability}operands: Vec<_> = op.deref(ctx).operands().collect();"
        )
        .unwrap();
        let operand_count = record.dialect.operands.len();
        let operand_count_check = if operand_count == 0 {
            "!operands.is_empty()".to_owned()
        } else {
            format!("operands.len() != {operand_count}")
        };
        writeln!(
            output,
            "        if {operand_count_check} {{\n            return pliron::input_err_noloc!({:?}, operands.len());\n        }}",
            format!(
                "{} requires {} operands, got {{}}",
                record.rust.name,
                operand_count
            )
        )
        .unwrap();
        if matches!(
            contract.adapter,
            MbarrierExtendedAdapter::PointerTxCountBytesToTokenDroppingTxCount
                | MbarrierExtendedAdapter::PointerTokenToPredicate
                | MbarrierExtendedAdapter::PointerParityToPredicate
        ) {
            output.push_str(
                "        operands[0] = cast_to_shared_addrspace(ctx, rewriter, operands[0]);\n",
            );
        }
        match contract.adapter {
            MbarrierExtendedAdapter::PointerTxCountBytesToTokenDroppingTxCount => {
                writeln!(
                    output,
                    "        let result_ty = IntegerType::get(ctx, 64, Signedness::Signless);\n        let asm = inline_asm_convergent(ctx, rewriter, op, result_ty.into(), operands, {template:?}, {constraints:?});\n        rewriter.replace_operation(ctx, op, asm);\n        Ok(())"
                )
                .unwrap();
            }
            MbarrierExtendedAdapter::PointerTokenToPredicate
            | MbarrierExtendedAdapter::PointerParityToPredicate => {
                writeln!(
                    output,
                    "        let result_ty = IntegerType::get(ctx, 32, Signedness::Signless);\n        let asm = inline_asm_convergent(ctx, rewriter, op, result_ty.into(), operands, {template:?}, {constraints:?});\n        let asm_result = asm.deref(ctx).get_result(0);\n        let result = trunc_to_i1(ctx, rewriter, asm_result);\n        let DefiningEntity::Op(result_op) = result.defining_entity() else {{ unreachable!() }};\n        rewriter.replace_operation(ctx, op, result_op);\n        Ok(())"
                )
                .unwrap();
            }
            MbarrierExtendedAdapter::RawClusterAddressToVoid
            | MbarrierExtendedAdapter::ZeroOperandsToVoid
            | MbarrierExtendedAdapter::NanosecondsToVoid => {
                writeln!(
                    output,
                    "        let void_ty = llvm_types::VoidType::get(ctx);\n        inline_asm_convergent(ctx, rewriter, op, void_ty.into(), operands, {template:?}, {constraints:?});\n        rewriter.erase_operation(ctx, op);\n        Ok(())"
                )
                .unwrap();
            }
        }
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
    for record in clc_intrinsics(catalog) {
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        operands_info: &OperandsInfo,\n    ) -> Result<()> {\n",
        );
        match record.clc.as_ref().unwrap().operation {
            ClcOperation::TryCancel | ClcOperation::TryCancelMulticast => {
                writeln!(
                    output,
                    "        convert_generated_clc_try_cancel(ctx, rewriter, self.get_operation(), operands_info, {:?})",
                    record.llvm_identifier()
                )
                .unwrap();
            }
            ClcOperation::QueryIsCanceled
            | ClcOperation::QueryGetFirstCtaidX
            | ClcOperation::QueryGetFirstCtaidY
            | ClcOperation::QueryGetFirstCtaidZ => {
                writeln!(
                    output,
                    "        convert_generated_clc_query(ctx, rewriter, self.get_operation(), operands_info, {:?}, {})",
                    record.llvm_identifier(),
                    record.clc.as_ref().unwrap().operation == ClcOperation::QueryIsCanceled
                )
                .unwrap();
            }
        }
        output.push_str("    }\n}\n\n");
    }
    for record in execution_controls(catalog) {
        let operation = ExecutionControlOperation::from_catalog_id(&record.id)
            .expect("closed execution-control record");
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        operands_info: &OperandsInfo,\n    ) -> Result<()> {\n",
        );
        match operation {
            ExecutionControlOperation::BarrierCtaSync
            | ExecutionControlOperation::BarrierCtaSyncAligned
            | ExecutionControlOperation::BarrierCtaArrive
            | ExecutionControlOperation::BarrierCtaArriveAligned => {
                let template = format!(
                    "{}.{} $0, $1;",
                    record.expected_ptx.mnemonic,
                    record.expected_ptx.modifiers.join(".")
                );
                writeln!(
                    output,
                    "        convert_counted_barrier(ctx, rewriter, self.get_operation(), operands_info, {:?}, {template:?})",
                    record.llvm_identifier()
                )
                .unwrap();
            }
            ExecutionControlOperation::GridDependencyLaunchDependents
            | ExecutionControlOperation::GridDependencyWait => {
                let template = format!(
                    "{}.{};",
                    record.expected_ptx.mnemonic,
                    record.expected_ptx.modifiers.join(".")
                );
                writeln!(
                    output,
                    "        convert_grid_dependency(ctx, rewriter, self.get_operation(), operands_info, {:?}, {template:?})",
                    record.llvm_identifier()
                )
                .unwrap();
            }
            ExecutionControlOperation::SetMaxNRegInc | ExecutionControlOperation::SetMaxNRegDec => {
                let direction = match operation {
                    ExecutionControlOperation::SetMaxNRegInc => "inc",
                    ExecutionControlOperation::SetMaxNRegDec => "dec",
                    _ => unreachable!("setmaxnreg operation was matched"),
                };
                output.push_str(
                    "        let register_count = self.register_count(ctx).ok_or_else(|| pliron::input_error_noloc!(\"setmaxnreg register-count attribute is invalid\"))?;\n",
                );
                writeln!(
                    output,
                    "        convert_setmaxnreg(ctx, rewriter, self.get_operation(), operands_info, register_count, {:?}, {direction:?})",
                    record.llvm_identifier()
                )
                .unwrap();
            }
        }
        output.push_str("    }\n}\n\n");
    }
    for record in tma_intrinsics(catalog) {
        let operation = record.tma.as_ref().unwrap().operation;
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        operands_info: &OperandsInfo,\n    ) -> Result<()> {\n",
        );
        match operation {
            TmaOperation::G2sTile1d
            | TmaOperation::G2sTile2d
            | TmaOperation::G2sTile3d
            | TmaOperation::G2sTile4d
            | TmaOperation::G2sTile5d => {
                writeln!(
                    output,
                    "        convert_g2s(ctx, rewriter, self.get_operation(), operands_info, {}, false)",
                    operation.dimensions().unwrap()
                )
                .unwrap();
            }
            TmaOperation::G2sTile2dMulticast => output.push_str(
                "        convert_g2s(ctx, rewriter, self.get_operation(), operands_info, 2, true)\n",
            ),
            TmaOperation::G2sTile2dMulticastCg2 => output.push_str(
                "        convert_g2s_multicast_cg2(ctx, rewriter, self.get_operation(), operands_info)\n",
            ),
            TmaOperation::S2gTile1d
            | TmaOperation::S2gTile2d
            | TmaOperation::S2gTile3d
            | TmaOperation::S2gTile4d
            | TmaOperation::S2gTile5d => {
                writeln!(
                    output,
                    "        convert_s2g(ctx, rewriter, self.get_operation(), operands_info, {})",
                    operation.dimensions().unwrap()
                )
                .unwrap();
            }
            TmaOperation::Reduce => {
                let reduction = record
                    .tma
                    .as_ref()
                    .and_then(|tma| tma.reduction.as_ref())
                    .expect("TMA reduction contract");
                let reduction_name = match reduction.operation {
                    TmaReductionOperation::Add => "add",
                    TmaReductionOperation::And => "and",
                    TmaReductionOperation::Dec => "dec",
                    TmaReductionOperation::Inc => "inc",
                    TmaReductionOperation::Max => "max",
                    TmaReductionOperation::Min => "min",
                    TmaReductionOperation::Or => "or",
                    TmaReductionOperation::Xor => "xor",
                };
                let load_mode = match reduction.load_mode {
                    TmaReductionLoadMode::Tile => "tile",
                    TmaReductionLoadMode::Im2col => "im2col",
                };
                writeln!(
                    output,
                    "        convert_reduce_s2g(ctx, rewriter, self.get_operation(), operands_info, ReduceConfig::new({}, {reduction_name:?}, {load_mode:?}, {:?}))",
                    reduction.dimensions,
                    record.resolved_llvm_identifier()
                )
                    .unwrap();
            }
            TmaOperation::CommitGroup | TmaOperation::WaitGroup | TmaOperation::WaitGroupRead => {
                let operation_name = match operation {
                    TmaOperation::CommitGroup => "commit_group",
                    TmaOperation::WaitGroup => "wait_group",
                    TmaOperation::WaitGroupRead => "wait_group_read",
                    _ => unreachable!("TMA control operation was matched"),
                };
                writeln!(
                    output,
                    "        convert_control(ctx, rewriter, self.get_operation(), operands_info, {operation_name:?}, {:?})",
                    record.resolved_llvm_identifier()
                )
                .unwrap();
            }
            TmaOperation::PrefetchTensorMap => {
                writeln!(
                    output,
                    "        convert_prefetch_tensormap(ctx, rewriter, self.get_operation(), operands_info, {:?})",
                    record.resolved_llvm_identifier()
                )
                .unwrap();
            }
            TmaOperation::PrefetchTile1d
            | TmaOperation::PrefetchTile2d
            | TmaOperation::PrefetchTile3d
            | TmaOperation::PrefetchTile4d
            | TmaOperation::PrefetchTile5d
            | TmaOperation::PrefetchTileGather4TwoDimensional
            | TmaOperation::PrefetchTile1dCacheHint
            | TmaOperation::PrefetchTile2dCacheHint
            | TmaOperation::PrefetchTile3dCacheHint
            | TmaOperation::PrefetchTile4dCacheHint
            | TmaOperation::PrefetchTile5dCacheHint
            | TmaOperation::PrefetchTileGather4TwoDimensionalCacheHint => {
                writeln!(
                    output,
                    "        convert_prefetch_tile(ctx, rewriter, self.get_operation(), operands_info, PrefetchTileConfig::new({}, {}, {}, {:?}))",
                    operation.prefetch_coordinate_count().unwrap(),
                    matches!(
                        operation,
                        TmaOperation::PrefetchTileGather4TwoDimensional
                            | TmaOperation::PrefetchTileGather4TwoDimensionalCacheHint
                    ),
                    operation.uses_prefetch_cache_hint(),
                    record.resolved_llvm_identifier()
                )
                .unwrap();
            }
            TmaOperation::ReplaceBoxDim
            | TmaOperation::ReplaceElementStride
            | TmaOperation::ReplaceElementType
            | TmaOperation::ReplaceFillMode
            | TmaOperation::ReplaceGlobalAddress
            | TmaOperation::ReplaceGlobalDim
            | TmaOperation::ReplaceGlobalStride
            | TmaOperation::ReplaceInterleaveLayout
            | TmaOperation::ReplaceRank
            | TmaOperation::ReplaceSwizzleAtomicity
            | TmaOperation::ReplaceSwizzleMode => {
                let (field, value_kind, ordinal, immediate) = match operation {
                    TmaOperation::ReplaceBoxDim => ("box_dim", "u32", true, false),
                    TmaOperation::ReplaceElementStride => {
                        ("element_stride", "u32", true, false)
                    }
                    TmaOperation::ReplaceElementType => ("elemtype", "u32", false, true),
                    TmaOperation::ReplaceFillMode => ("fill_mode", "u32", false, true),
                    TmaOperation::ReplaceGlobalAddress => {
                        ("global_address", "address", false, false)
                    }
                    TmaOperation::ReplaceGlobalDim => ("global_dim", "u32", true, false),
                    TmaOperation::ReplaceGlobalStride => {
                        ("global_stride", "u64", true, false)
                    }
                    TmaOperation::ReplaceInterleaveLayout => {
                        ("interleave_layout", "u32", false, true)
                    }
                    TmaOperation::ReplaceRank => ("rank", "u32", false, false),
                    TmaOperation::ReplaceSwizzleAtomicity => {
                        ("swizzle_atomicity", "u32", false, true)
                    }
                    TmaOperation::ReplaceSwizzleMode => {
                        ("swizzle_mode", "u32", false, true)
                    }
                    _ => unreachable!("TMA tensor-map replace operation was matched"),
                };
                writeln!(
                    output,
                    "        convert_tensormap_replace(ctx, rewriter, self.get_operation(), operands_info, {:?}, {field:?}, {value_kind:?}, {ordinal}, {immediate})",
                    record.resolved_llvm_identifier()
                )
                .unwrap();
            }
            TmaOperation::FenceProxyTensorMapAcquireCluster
            | TmaOperation::FenceProxyTensorMapAcquireCta
            | TmaOperation::FenceProxyTensorMapAcquireGpu
            | TmaOperation::FenceProxyTensorMapAcquireSystem
            | TmaOperation::FenceProxyTensorMapReleaseCluster
            | TmaOperation::FenceProxyTensorMapReleaseCta
            | TmaOperation::FenceProxyTensorMapReleaseGpu
            | TmaOperation::FenceProxyTensorMapReleaseSystem => {
                let (acquire, scope) = match operation {
                    TmaOperation::FenceProxyTensorMapAcquireCluster => (true, "cluster"),
                    TmaOperation::FenceProxyTensorMapAcquireCta => (true, "cta"),
                    TmaOperation::FenceProxyTensorMapAcquireGpu => (true, "gpu"),
                    TmaOperation::FenceProxyTensorMapAcquireSystem => (true, "sys"),
                    TmaOperation::FenceProxyTensorMapReleaseCluster => (false, "cluster"),
                    TmaOperation::FenceProxyTensorMapReleaseCta => (false, "cta"),
                    TmaOperation::FenceProxyTensorMapReleaseGpu => (false, "gpu"),
                    TmaOperation::FenceProxyTensorMapReleaseSystem => (false, "sys"),
                    _ => unreachable!("TMA tensor-map fence operation was matched"),
                };
                writeln!(
                    output,
                    "        convert_tensormap_fence(ctx, rewriter, self.get_operation(), operands_info, {:?}, {acquire}, {scope:?})",
                    record.resolved_llvm_identifier()
                )
                .unwrap();
            }
        }
        output.push_str("    }\n}\n\n");
    }
    for record in tcgen05_non_mma_intrinsics(catalog) {
        let operation = record.tcgen05.as_ref().unwrap().operation;
        let (template, constraints, result_count) = tcgen05_inline_asm(record);
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n",
        );
        if let Some(count) = result_count {
            let integer_results = operation == Tcgen05Operation::Ld;
            writeln!(
                output,
                "        convert_generated_tcgen05_load(ctx, rewriter, self.get_operation(), {}, {count}, {integer_results}, {template:?}, {constraints:?})",
                record.dialect.operands.len()
            )
            .unwrap();
        } else {
            writeln!(
                output,
                "        convert_generated_tcgen05_void(ctx, rewriter, self.get_operation(), {}, {template:?}, {constraints:?})",
                record.dialect.operands.len()
            )
            .unwrap();
        }
        output.push_str("    }\n}\n\n");
    }
    if tcgen05_mma_intrinsics(catalog).next().is_some() {
        output.push_str(
            r#"#[op_interface_impl]
impl MirToLlvmConversion for Tcgen05MmaOp {
    fn convert(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let form = self
            .get_attr_nvvm_tcgen05_mma_form(ctx)
            .as_deref()
            .cloned();
        let kind = self
            .get_attr_nvvm_tcgen05_mma_kind(ctx)
            .as_deref()
            .cloned();
        let cta_group = self
            .get_attr_nvvm_tcgen05_mma_cta_group(ctx)
            .as_deref()
            .cloned();
        let collector_a = self
            .get_attr_nvvm_tcgen05_mma_collector_a(ctx)
            .as_deref()
            .cloned();
        let b_buffer = self
            .get_attr_nvvm_tcgen05_mma_b_buffer(ctx)
            .as_deref()
            .cloned();
        let b_usage = self
            .get_attr_nvvm_tcgen05_mma_b_usage(ctx)
            .as_deref()
            .cloned();
        let kind = match kind {
            Some(Tcgen05MmaKindAttr::F16) => "f16",
            Some(Tcgen05MmaKindAttr::Tf32) => "tf32",
            Some(Tcgen05MmaKindAttr::F8f6f4) => "f8f6f4",
            Some(Tcgen05MmaKindAttr::I8) => "i8",
            None => return pliron::input_err!(
                op.deref(ctx).loc(),
                "nvvm.tcgen05_mma requires a kind",
            ),
        };
        let (prefix, a, metadata, zero_mask, constraints, arity, base, ashift) =
            match form {
                Some(Tcgen05MmaFormAttr::Shared) =>
                    ("tcgen05.mma", "$1", None, None, "r,l,l,r,r,~{memory}", 5, true, false),
                Some(Tcgen05MmaFormAttr::Tensor) =>
                    ("tcgen05.mma", "[$1]", None, None, "r,r,l,r,r,~{memory}", 5, true, false),
                Some(Tcgen05MmaFormAttr::TensorAshift) =>
                    ("tcgen05.mma", "[$1]", None, None, "r,r,l,r,r,~{memory}", 5, true, true),
                Some(Tcgen05MmaFormAttr::SpShared) =>
                    ("tcgen05.mma.sp", "$1", Some("$5"), None, "r,l,l,r,r,r,~{memory}", 6, true, false),
                Some(Tcgen05MmaFormAttr::SpTensor) =>
                    ("tcgen05.mma.sp", "[$1]", Some("$5"), None, "r,r,l,r,r,r,~{memory}", 6, true, false),
                Some(Tcgen05MmaFormAttr::SpTensorAshift) =>
                    ("tcgen05.mma.sp", "[$1]", Some("$5"), None, "r,r,l,r,r,r,~{memory}", 6, true, true),
                Some(Tcgen05MmaFormAttr::WsShared) =>
                    ("tcgen05.mma.ws", "$1", None, None, "r,l,l,r,r,~{memory}", 5, false, false),
                Some(Tcgen05MmaFormAttr::WsSharedZeroColMask) =>
                    ("tcgen05.mma.ws", "$1", None, Some("$5"), "r,l,l,r,r,l,~{memory}", 6, false, false),
                Some(Tcgen05MmaFormAttr::WsSpShared) =>
                    ("tcgen05.mma.ws.sp", "$1", Some("$5"), None, "r,l,l,r,r,r,~{memory}", 6, false, false),
                Some(Tcgen05MmaFormAttr::WsSpSharedZeroColMask) =>
                    ("tcgen05.mma.ws.sp", "$1", Some("$5"), Some("$6"), "r,l,l,r,r,r,l,~{memory}", 7, false, false),
                Some(Tcgen05MmaFormAttr::WsSpTensor) =>
                    ("tcgen05.mma.ws.sp", "[$1]", Some("$5"), None, "r,r,l,r,r,r,~{memory}", 6, false, false),
                Some(Tcgen05MmaFormAttr::WsSpTensorZeroColMask) =>
                    ("tcgen05.mma.ws.sp", "[$1]", Some("$5"), Some("$6"), "r,r,l,r,r,r,l,~{memory}", 7, false, false),
                Some(Tcgen05MmaFormAttr::WsTensor) =>
                    ("tcgen05.mma.ws", "[$1]", None, None, "r,r,l,r,r,~{memory}", 5, false, false),
                Some(Tcgen05MmaFormAttr::WsTensorZeroColMask) =>
                    ("tcgen05.mma.ws", "[$1]", None, Some("$5"), "r,r,l,r,r,l,~{memory}", 6, false, false),
                None => return pliron::input_err!(
                    op.deref(ctx).loc(),
                    "nvvm.tcgen05_mma requires a form",
                ),
            };

        let selector = if base {
            let group = match cta_group {
                Some(Tcgen05MmaCtaGroupAttr::Cg1) => 1,
                Some(Tcgen05MmaCtaGroupAttr::Cg2) => 2,
                None => return pliron::input_err!(
                    op.deref(ctx).loc(),
                    "base tcgen05 MMA requires a CTA-group selector",
                ),
            };
            let usage = match collector_a {
                Some(Tcgen05MmaCollectorAAttr::Discard) => "discard",
                Some(Tcgen05MmaCollectorAAttr::LastUse) => "lastuse",
                Some(Tcgen05MmaCollectorAAttr::Fill) if !ashift => "fill",
                Some(Tcgen05MmaCollectorAAttr::Use) if !ashift => "use",
                _ => return pliron::input_err!(
                    op.deref(ctx).loc(),
                    "tcgen05 MMA collector-A selector has no generated recipe",
                ),
            };
            format!(
                ".cta_group::{group}.kind::{kind}.collector::a::{usage}{}",
                if ashift { ".ashift" } else { "" },
            )
        } else {
            let buffer = match b_buffer {
                Some(Tcgen05MmaBBufferAttr::B0) => 0,
                Some(Tcgen05MmaBBufferAttr::B1) => 1,
                Some(Tcgen05MmaBBufferAttr::B2) => 2,
                Some(Tcgen05MmaBBufferAttr::B3) => 3,
                None => return pliron::input_err!(
                    op.deref(ctx).loc(),
                    "warp-specialized tcgen05 MMA requires a B-buffer selector",
                ),
            };
            let usage = match b_usage {
                Some(Tcgen05MmaBUsageAttr::Discard) => "discard",
                Some(Tcgen05MmaBUsageAttr::LastUse) => "lastuse",
                Some(Tcgen05MmaBUsageAttr::Fill) => "fill",
                Some(Tcgen05MmaBUsageAttr::Use) => "use",
                None => return pliron::input_err!(
                    op.deref(ctx).loc(),
                    "warp-specialized tcgen05 MMA requires a B-usage selector",
                ),
            };
            format!(
                ".cta_group::1.kind::{kind}.collector::b{buffer}::{usage}"
            )
        };

        let mut operands = format!("[$0], {a}, $2");
        if let Some(metadata) = metadata {
            operands.push_str(&format!(", [{metadata}]"));
        }
        operands.push_str(", $3, %enable_pred");
        if let Some(zero_mask) = zero_mask {
            operands.push_str(&format!(", {zero_mask}"));
        }
        let template = format!(
            "{{ .reg .pred %enable_pred; setp.ne.s32 %enable_pred, $4, 0; {prefix}{selector} {operands}; }}"
        );
        convert_generated_tcgen05_void(
            ctx, rewriter, op, arity, &template, constraints,
        )
    }
}

"#,
        );
    }
    for record in debug_controls(catalog) {
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        output.push_str(
            "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        let op = self.get_operation();\n        let void_ty = llvm_types::VoidType::get(ctx);\n",
        );
        match record.debug_control.as_ref().unwrap().operation {
            DebugControlOperation::Trap => output.push_str(
                "        inline_asm_sideeffect(ctx, rewriter, op, void_ty.into(), vec![], \"trap;\", \"\");\n",
            ),
            DebugControlOperation::Breakpoint => output.push_str(
                "        inline_asm_sideeffect(ctx, rewriter, op, void_ty.into(), vec![], \"brkpt;\", \"\");\n",
            ),
            DebugControlOperation::Pmevent => output.push_str(
                "        let Some(event_id) = self.event_id(ctx) else {\n\
                     return pliron::input_err!(\n\
                         op.deref(ctx).loc(),\n\
                         \"nvvm.pmevent requires a u32 event ID in 0..=15\",\n\
                     );\n\
                 };\n\
                 let template = format!(\"pmevent {event_id};\");\n\
                 inline_asm_sideeffect(ctx, rewriter, op, void_ty.into(), vec![], &template, \"\");\n",
            ),
        }
        output.push_str("        rewriter.erase_operation(ctx, op);\n        Ok(())\n    }\n}\n\n");
    }
    for record in sync_intrinsics(catalog) {
        writeln!(
            output,
            "#[op_interface_impl]\nimpl MirToLlvmConversion for {} {{",
            record.dialect.op_type
        )
        .unwrap();
        if record.id == "sync_threads" {
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
                "            }\n            IntrinsicBackend::LibNvvm => {\n                inline_asm_convergent(ctx, rewriter, op, void_ty.into(), vec![], \"bar.sync 0;\", \"~{memory}\");\n            }\n        }\n        rewriter.erase_operation(ctx, op);\n        Ok(())\n    }\n}\n\n",
            );
        } else {
            debug_assert!(threadfence_ptx_level(record).is_some());
            output.push_str(
                "    fn convert(\n        &self,\n        ctx: &mut Context,\n        rewriter: &mut DialectConversionRewriter,\n        _operands_info: &OperandsInfo,\n    ) -> Result<()> {\n        let op = self.get_operation();\n        let void_ty = llvm_types::VoidType::get(ctx);\n        let function_ty = llvm_types::FuncType::get(ctx, void_ty.into(), vec![], false);\n",
            );
            writeln!(
                output,
                "        call_intrinsic(ctx, rewriter, op, {:?}, function_ty, vec![])?;",
                record.llvm_identifier()
            )
            .unwrap();
            output.push_str(
                "        rewriter.erase_operation(ctx, op);\n        Ok(())\n    }\n}\n\n",
            );
        }
    }
    output
}
