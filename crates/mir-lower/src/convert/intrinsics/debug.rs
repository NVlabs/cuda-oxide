/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Debug and profiling intrinsic conversion.
//!
//! | Operation      | Lowering                                | PTX Output              |
//! |----------------|-----------------------------------------|-------------------------|
//! | `Clock`        | `llvm_nvvm_read_ptx_sreg_clock`         | `mov %r, %clock`        |
//! | `Clock64`      | `llvm_nvvm_read_ptx_sreg_clock64`       | `mov %rd, %clock64`     |
//! | `Globaltimer`  | `llvm_nvvm_read_ptx_sreg_globaltimer`   | `mov %rd, %globaltimer` |
//! | `Vprintf`      | `call @vprintf`                         | `call vprintf`          |

use crate::helpers;
use llvm_export::ops as llvm;
use llvm_export::types as llvm_types;
use pliron::builtin::op_interfaces::CallOpCallable;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

pub(crate) fn convert_vprintf(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 2 {
        return pliron::input_err_noloc!("vprintf requires 2 operands, got {}", operands.len());
    }

    let format_ptr = operands[0];
    let args_ptr = operands[1];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let i8_ptr_ty = llvm_types::PointerType::get(ctx, 0);

    let func_ty = llvm_types::FuncType::get(
        ctx,
        i32_ty.into(),
        vec![i8_ptr_ty.into(), i8_ptr_ty.into()],
        false,
    );

    let parent_block = op.deref(ctx).get_parent_block().unwrap();
    helpers::ensure_intrinsic_declared(ctx, parent_block, "vprintf", func_ty)
        .map_err(|e| pliron::input_error_noloc!("{}", e))?;

    let sym_name: pliron::identifier::Identifier = "vprintf".try_into().unwrap();
    let callee = CallOpCallable::Direct(sym_name);
    let call_op = llvm::CallOp::new(ctx, callee, func_ty, vec![format_ptr, args_ptr]);
    rewriter.insert_operation(ctx, call_op.get_operation());
    rewriter.replace_operation(ctx, op, call_op.get_operation());

    Ok(())
}
