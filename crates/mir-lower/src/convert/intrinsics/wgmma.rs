/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! WGMMA conversion for Hopper `sm_90a`.

use crate::convert::intrinsics::common::*;
use llvm_export::types::VoidType;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::rewriter::Rewriter;
use pliron::operation::Operation;
use pliron::result::Result;

/// Convert WGMMA make_smem_desc to inline PTX.
pub(crate) fn convert_make_smem_desc(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.is_empty() {
        return pliron::input_err_noloc!("wgmma_make_smem_desc requires operand");
    }
    let ptr = operands[0];
    let ptr_casted = cast_to_shared_addrspace(ctx, rewriter, ptr);

    let asm_template = r#"{
    .reg .u64 addr;
    cvta.to.shared.u64 addr, $1;
    shr.u64 addr, addr, 4;
    and.b64 addr, addr, 0x3FFF;
    or.b64 $0, addr, 0xC000000800080000;
}"#;

    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        i64_ty.into(),
        vec![ptr_casted],
        asm_template,
        "=l,l",
    );
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

fn accumulator_register_list() -> String {
    (0..32)
        .map(|index| format!("%acc{index}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn deferred_group_template(mma_count: usize) -> String {
    let mut template = String::from("{\n    .reg .f32 %acc<32>;\n");

    for index in 0..32 {
        let offset = index * 4;
        template.push_str(&format!("    ld.f32 %acc{index}, [$0 + {offset}];\n"));
    }

    template.push_str("    wgmma.fence.sync.aligned;\n");
    let registers = accumulator_register_list();
    for mma_index in 0..mma_count {
        let desc_a = 1 + mma_index * 2;
        let desc_b = desc_a + 1;
        template.push_str(&format!(
            "    wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 \
             {{{registers}}}, ${desc_a}, ${desc_b}, 1, 1, 1, 0, 0;\n"
        ));
    }
    template.push_str("    wgmma.commit_group.sync.aligned;\n");
    template.push_str("    wgmma.wait_group.sync.aligned 0;\n");

    for index in 0..32 {
        let offset = index * 4;
        template.push_str(&format!("    st.f32 [$0 + {offset}], %acc{index};\n"));
    }
    template.push('}');
    template
}

/// Lower a complete deferred BF16 WGMMA group.
///
/// The inline-PTX scope owns 32 explicit accumulator registers. It loads them
/// before the fence, issues every MMA, commits, waits for zero pending groups,
/// and writes them back only after the wait. This avoids exposing pending
/// accumulator values to LLVM or to memory.
pub(crate) fn convert_mma_group(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 3 || operands.len() % 2 == 0 {
        return pliron::input_err_noloc!(
            "deferred WGMMA group requires one accumulator pointer and one or more descriptor pairs"
        );
    }

    let mma_count = (operands.len() - 1) / 2;
    let template = deferred_group_template(mma_count);
    let mut constraints = vec!["l"; operands.len()];
    constraints.push("~{memory}");
    let constraints = constraints.join(",");

    inline_asm_convergent(
        ctx,
        rewriter,
        VoidType::get(ctx).into(),
        operands,
        &template,
        &constraints,
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Reject an unfused pointer-form MMA operation.
///
/// Reaching this converter means the pre-lowering adapter could not prove a
/// complete and sound fence/MMA/commit/wait sequence.
pub(crate) fn convert_mma(
    _ctx: &mut Context,
    _rewriter: &mut DialectConversionRewriter,
    _op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    pliron::input_err_noloc!(
        "WGMMA MMA reached lowering without deferred accumulator fusion; expected a linear fence -> BF16 MMA+ -> commit_group -> wait_group<0> sequence"
    )
}

#[cfg(test)]
mod tests {
    use super::deferred_group_template;

    #[test]
    fn deferred_template_keeps_loads_before_wait_and_stores_after_wait() {
        let template = deferred_group_template(2);
        assert_eq!(template.matches("ld.f32 %acc").count(), 32);
        assert_eq!(template.matches("st.f32 [$0").count(), 32);
        assert_eq!(template.matches("wgmma.mma_async").count(), 2);

        let first_mma = template.find("wgmma.mma_async").unwrap();
        let wait = template.find("wgmma.wait_group.sync.aligned 0").unwrap();
        let first_store = template.find("st.f32 [$0").unwrap();
        assert!(first_mma < wait);
        assert!(wait < first_store);
    }
}
