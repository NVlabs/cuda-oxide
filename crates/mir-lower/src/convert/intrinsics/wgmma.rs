/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! WGMMA (Warpgroup Matrix Multiply-Accumulate) intrinsic conversion for Hopper+ GPUs.
//!
//! # Operations
//!
//! | Operation             | PTX                             | Description                    |
//! |-----------------------|---------------------------------|--------------------------------|
//! | `Fence`               | `wgmma.fence.sync.aligned`      | Memory fence before WGMMA      |
//! | `CommitGroup`         | `wgmma.commit_group.sync.aligned`| Commit pending operations     |
//! | `WaitGroup`           | `wgmma.wait_group.sync.aligned N`| Wait for N groups             |
//! | `MakeSmemDesc`        | cvta + bit manipulation         | Create shared memory descriptor|
//! | `MmaM64N64K16F32Bf16` | `wgmma.mma_async`               | Matrix multiply                |

use crate::convert::intrinsics::common::*;
use llvm_export::types as llvm_types;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::rewriter::Rewriter;
use pliron::operation::Operation;
use pliron::result::Result;

pub(crate) fn convert_fence(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![],
        "wgmma.fence.sync.aligned;",
        "",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

pub(crate) fn convert_commit_group(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);
    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![],
        "wgmma.commit_group.sync.aligned;",
        "",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert WGMMA wait_group to inline PTX.
pub(crate) fn convert_wait_group(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let void_ty = llvm_types::VoidType::get(ctx);

    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.is_empty() {
        return pliron::input_err_noloc!("wgmma_wait_group requires 1 operand");
    }
    let n = operands[0];

    inline_asm_convergent(
        ctx,
        rewriter,
        void_ty.into(),
        vec![n],
        "wgmma.wait_group.sync.aligned $0;",
        "n",
    );
    rewriter.erase_operation(ctx, op);
    Ok(())
}

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

/// Convert WGMMA MMA operation to inline PTX.
///
/// The full lowering requires register allocation for 16+ output registers
/// and is not yet implemented. Until it lands, calls to
/// `cuda_device::wgmma::wgmma_mma_*` from a `#[kernel]` are rejected at
/// codegen time with a clear diagnostic.
///
/// The previous behaviour silently emitted `// wgmma.mma placeholder` as an
/// inline-asm comment and erased the op, producing PTX that loaded and ran
/// but multiplied-accumulated to zero — a silent miscompile with no warning.
pub(crate) fn convert_mma(
    _ctx: &mut Context,
    _rewriter: &mut DialectConversionRewriter,
    _op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    pliron::input_err_noloc!(
        "wgmma.mma_async lowering is not yet implemented; \
         calls to `cuda_device::wgmma::wgmma_mma_*` from a kernel are \
         currently unsupported. Tracking issue: full lowering requires \
         register allocation for 16+ output registers."
    )
}
