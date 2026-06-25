/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Control flow operation conversion: `dialect-mir` → LLVM dialect.
//!
//! Converts `dialect-mir` terminators and control flow operations to their
//! LLVM dialect equivalents.
//!
//! # Operations
//!
//! | MIR Operation      | LLVM Operation              | Description             |
//! |--------------------|-----------------------------|-------------------------|
//! | `mir.return`       | `llvm.return`               | Function return         |
//! | `mir.goto`         | `llvm.br`                   | Unconditional branch    |
//! | `mir.cond_branch`  | `llvm.cond_br`              | Conditional branch      |
//! | `mir.assert`       | `llvm.cond_br` + abort blk  | Runtime assertion       |
//! | `mir.unreachable`  | `llvm.unreachable`          | Unreachable marker      |
//!
//! # Block Handling
//!
//! With `DialectConversion` + `inline_region`, blocks are the ORIGINALS (moved,
//! not copied). Successor pointers are already valid — no block map lookup needed.

use llvm_export::ops as llvm;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::Typed;

/// Convert `mir.return` to `llvm.return`.
///
/// Handles:
/// - Void returns (no operands)
/// - Single value returns
/// - Empty struct returns (treated as void)
pub(crate) fn convert_return(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    let ret_val = match operands.as_slice() {
        [] => None,
        [val] => {
            let ty = val.get_type(ctx);

            let is_empty_struct = ty
                .deref(ctx)
                .downcast_ref::<llvm_export::types::StructType>()
                .is_some_and(|st| st.num_fields() == 0);

            if ty.deref(ctx).is::<llvm_export::types::VoidType>() || is_empty_struct {
                None
            } else {
                Some(*val)
            }
        }
        _ => {
            return pliron::input_err_noloc!("Return with multiple operands not supported");
        }
    };

    let llvm_ret = llvm::ReturnOp::new(ctx, ret_val);
    rewriter.insert_operation(ctx, llvm_ret.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert `mir.unreachable` to `llvm.unreachable`.
pub(crate) fn convert_unreachable(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let unreachable_op = llvm::UnreachableOp::new(ctx);
    rewriter.insert_operation(ctx, unreachable_op.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert `mir.cond_branch` to `llvm.cond_br`.
///
/// MIR conditional branches have:
/// - Operand 0: condition (i1)
/// - Operands 1..N: arguments for true block
/// - Operands N+1..M: arguments for false block
/// - Successor 0: true block
/// - Successor 1: false block
///
/// With `inline_region`, successors are the original blocks — no block map needed.
pub(crate) fn convert_cond_branch(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    let cond = match operands.first() {
        Some(v) => *v,
        None => return pliron::input_err_noloc!("CondBranch requires at least 1 operand"),
    };

    let successors: Vec<_> = op.deref(ctx).successors().collect();
    let (true_block, false_block) = match successors.as_slice() {
        [t, f] => (*t, *f),
        _ => return pliron::input_err_noloc!("CondBranch requires exactly 2 successors"),
    };

    let num_true_args = true_block.deref(ctx).arguments().count();
    let num_false_args = false_block.deref(ctx).arguments().count();

    if operands.len() != 1 + num_true_args + num_false_args {
        return pliron::input_err_noloc!(
            "CondBranch operand count mismatch. Expected {}, got {}",
            1 + num_true_args + num_false_args,
            operands.len()
        );
    }

    let true_args = operands[1..1 + num_true_args].to_vec();
    let false_args = operands[1 + num_true_args..].to_vec();

    let llvm_br = llvm::CondBrOp::new(ctx, cond, true_block, true_args, false_block, false_args);
    rewriter.insert_operation(ctx, llvm_br.get_operation());
    rewriter.erase_operation(ctx, op);

    Ok(())
}

/// Convert `mir.assert` to conditional branch with abort block.
///
/// MIR assert is converted to:
/// 1. Create an abort block with `llvm.unreachable`
/// 2. `llvm.cond_br` to success block (if true) or abort block (if false)
///
/// The abort block is inserted directly (not through the rewriter), since it's
/// a new block, not a replacement for anything.
pub(crate) fn convert_assert(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    let (cond, args) = match operands.as_slice() {
        [cond, args @ ..] => (*cond, args),
        _ => return pliron::input_err_noloc!("Assert requires at least 1 operand"),
    };

    let successors: Vec<_> = op.deref(ctx).successors().collect();
    let success_block = match successors.as_slice() {
        [blk] => *blk,
        _ => return pliron::input_err_noloc!("Assert requires exactly 1 successor"),
    };

    let region = match op
        .deref(ctx)
        .get_parent_block()
        .and_then(|b| b.deref(ctx).get_parent_region())
    {
        Some(r) => r,
        None => return pliron::input_err_noloc!("Block has no parent region"),
    };

    let abort_block = BasicBlock::new(ctx, None, vec![]);
    abort_block.insert_at_back(region, ctx);

    llvm::UnreachableOp::new(ctx)
        .get_operation()
        .insert_at_back(abort_block, ctx);

    let llvm_br = llvm::CondBrOp::new(ctx, cond, success_block, args.to_vec(), abort_block, vec![]);
    rewriter.insert_operation(ctx, llvm_br.get_operation());
    rewriter.erase_operation(ctx, op);

    Ok(())
}

/// Convert `mir.goto` to `llvm.br`.
///
/// Handles ZST (Zero-Sized Type) padding: if the destination block expects
/// more arguments than provided, missing arguments for empty struct types
/// are filled with `undef` values.
///
/// With `inline_region`, the dest block arg types are already converted by
/// the framework, so the ZST check works on LLVM types directly.
pub(crate) fn convert_goto(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let successors: Vec<_> = op.deref(ctx).successors().collect();
    let dest = match successors.as_slice() {
        [dest] => *dest,
        _ => return pliron::input_err_noloc!("Goto requires exactly 1 successor"),
    };

    let mut final_args: Vec<_> = op.deref(ctx).operands().collect();

    let num_dest_args = dest.deref(ctx).arguments().count();

    if final_args.len() < num_dest_args {
        let dest_args: Vec<_> = dest.deref(ctx).arguments().skip(final_args.len()).collect();

        for dest_arg in dest_args {
            let arg_ty = dest_arg.get_type(ctx);

            let is_empty_struct = arg_ty
                .deref(ctx)
                .downcast_ref::<llvm_export::types::StructType>()
                .is_some_and(|st| st.num_fields() == 0);

            if is_empty_struct {
                let undef = llvm::UndefOp::new(ctx, arg_ty);
                rewriter.insert_operation(ctx, undef.get_operation());
                final_args.push(undef.get_operation().deref(ctx).get_result(0));
            } else {
                return pliron::input_err_noloc!(
                    "Goto operand count mismatch. Expected {}, got {}. \
                     Missing argument is not a ZST.",
                    num_dest_args,
                    final_args.len()
                );
            }
        }
    } else if final_args.len() > num_dest_args {
        return pliron::input_err_noloc!(
            "Goto operand count mismatch. Expected {}, got {}",
            num_dest_args,
            final_args.len()
        );
    }

    let llvm_br = llvm::BrOp::new(ctx, dest, final_args);
    rewriter.insert_operation(ctx, llvm_br.get_operation());
    rewriter.erase_operation(ctx, op);

    Ok(())
}

#[cfg(test)]
mod tests {
    //! End-to-end lowering tests for `dialect-mir` terminator ops.
    //!
    //! The `convert_*` functions take a live `DialectConversionRewriter` owned
    //! by the driver, so we can't call them directly — each test builds a
    //! minimal MIR module, runs `lower_mir_to_llvm`, and inspects the result.

    use crate::convert::ops::test_util::*;
    use dialect_mir::ops as mir;
    use dialect_mir::types::MirTupleType;
    use llvm_export::ops as llvm;
    use pliron::builtin::op_interfaces::{BranchOpInterface, OperandSegmentInterface};
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::linked_list::ContainsLinkedList;
    use pliron::op::Op;
    use pliron::operation::Operation;
    use pliron::r#type::TypeHandle;

    #[test]
    fn convert_return_void_lowers_to_llvm_return_without_value() {
        let mut ctx = make_ctx();
        let (module_ptr, entry) = build_kernel(&mut ctx, vec![], vec![]);
        append_mir_return(&mut ctx, entry, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let ret = find_first::<llvm::ReturnOp>(&ctx, &body).expect("expected llvm.return");
        assert_eq!(
            ret.get_operation().deref(&ctx).get_num_operands(),
            0,
            "void return must have no value operand"
        );
        assert_eq!(count_ops::<mir::MirReturnOp>(&ctx, &body), 0);
    }

    #[test]
    fn convert_return_with_scalar_value_lowers_to_llvm_return_with_value() {
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let (module_ptr, entry) = build_kernel(&mut ctx, vec![i32_ty], vec![i32_ty]);
        let arg = entry.deref(&ctx).get_argument(0);
        append_mir_return(&mut ctx, entry, vec![arg]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let ret = find_first::<llvm::ReturnOp>(&ctx, &body).expect("expected llvm.return");
        assert_eq!(
            ret.get_operation().deref(&ctx).get_num_operands(),
            1,
            "scalar return must carry one value operand"
        );
    }

    #[test]
    fn convert_return_empty_struct_treated_as_void() {
        // `mir.return %x` with `%x: ()` must drop the operand to match the
        // converted `-> void` signature. The unit value comes from `mir.undef`
        // to sidestep the function arg ABI, which strips ZSTs.
        //
        // NOTE: this test relies on the MIR type converter lowering `MirTupleType`
        // to an empty `llvm.struct`; `convert_return` checks for the latter,
        // not the former.
        let mut ctx = make_ctx();
        let unit_ty: TypeHandle = MirTupleType::get(&mut ctx, vec![]).into();
        let (module_ptr, entry) = build_kernel(&mut ctx, vec![], vec![unit_ty]);

        let undef = mir::MirUndefOp::new(&mut ctx, unit_ty);
        undef.get_operation().insert_at_back(entry, &ctx);
        let undef_val = undef.get_operation().deref(&ctx).get_result(0);
        append_mir_return(&mut ctx, entry, vec![undef_val]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let ret = find_first::<llvm::ReturnOp>(&ctx, &body).expect("expected llvm.return");
        assert_eq!(
            ret.get_operation().deref(&ctx).get_num_operands(),
            0,
            "empty-struct return value must collapse to void"
        );
    }

    #[test]
    fn convert_unreachable_lowers_to_llvm_unreachable() {
        let mut ctx = make_ctx();
        let (module_ptr, entry) = build_kernel(&mut ctx, vec![], vec![]);

        let unreach = Operation::new(
            &mut ctx,
            mir::MirUnreachableOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
        );
        unreach.insert_at_back(entry, &ctx);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        assert_eq!(count_ops::<llvm::UnreachableOp>(&ctx, &body), 1);
        assert_eq!(count_ops::<mir::MirUnreachableOp>(&ctx, &body), 0);
    }

    #[test]
    fn convert_cond_branch_splits_operands_into_per_block_args() {
        // The [cond | true_args | false_args] split: %val goes to true_block
        // (expects i32), false_block takes none — so the lowered cond_br must
        // expose one true-side operand and zero false-side.
        let mut ctx = make_ctx();
        let i1_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let (module_ptr, entry) = build_kernel(&mut ctx, vec![i1_ty, i32_ty], vec![]);
        let cond = entry.deref(&ctx).get_argument(0);
        let val = entry.deref(&ctx).get_argument(1);

        let true_block = append_block(&mut ctx, entry, vec![i32_ty]);
        let false_block = append_block(&mut ctx, entry, vec![]);
        append_mir_return(&mut ctx, true_block, vec![]);
        append_mir_return(&mut ctx, false_block, vec![]);

        let (operands, segment_sizes) =
            mir::MirCondBranchOp::compute_segment_sizes(vec![vec![cond], vec![val], vec![]]);
        let cond_br = Operation::new(
            &mut ctx,
            mir::MirCondBranchOp::get_concrete_op_info(),
            vec![],
            operands,
            vec![true_block, false_block],
            0,
        );
        mir::MirCondBranchOp::new(cond_br).set_operand_segment_sizes(&ctx, segment_sizes);
        cond_br.insert_at_back(entry, &ctx);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let llvm_br = find_first::<llvm::CondBrOp>(&ctx, &body).expect("expected llvm.cond_br");
        assert_eq!(llvm_br.successor_operands(&ctx, 0).len(), 1);
        assert_eq!(llvm_br.successor_operands(&ctx, 1).len(), 0);
        assert_eq!(count_ops::<mir::MirCondBranchOp>(&ctx, &body), 0);
    }

    #[test]
    fn convert_assert_creates_abort_block_with_unreachable() {
        // mir.assert lowers to a llvm.cond_br whose false side is a fresh
        // block ending in llvm.unreachable.
        let mut ctx = make_ctx();
        let i1_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let (module_ptr, entry) = build_kernel(&mut ctx, vec![i1_ty], vec![]);
        let cond = entry.deref(&ctx).get_argument(0);
        let success = append_block(&mut ctx, entry, vec![]);
        append_mir_return(&mut ctx, success, vec![]);

        let (operands, segment_sizes) =
            mir::MirAssertOp::compute_segment_sizes(vec![vec![cond], vec![]]);
        let assert_op = Operation::new(
            &mut ctx,
            mir::MirAssertOp::get_concrete_op_info(),
            vec![],
            operands,
            vec![success],
            0,
        );
        mir::MirAssertOp::new(assert_op).set_operand_segment_sizes(&ctx, segment_sizes);
        assert_op.insert_at_back(entry, &ctx);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        assert_eq!(count_ops::<llvm::CondBrOp>(&ctx, &body), 1);
        assert_eq!(
            count_ops::<llvm::UnreachableOp>(&ctx, &body),
            1,
            "abort block must terminate with llvm.unreachable"
        );
        assert_eq!(count_ops::<mir::MirAssertOp>(&ctx, &body), 0);

        let abort_block = body
            .iter()
            .find(|&&b| {
                b.deref(&ctx)
                    .iter(&ctx)
                    .any(|op| Operation::get_op::<llvm::UnreachableOp>(op, &ctx).is_some())
            })
            .copied()
            .expect("abort block must exist");
        let llvm_br = find_first::<llvm::CondBrOp>(&ctx, &body).expect("expected llvm.cond_br");
        let false_succ = llvm_br.get_operation().deref(&ctx).get_successor(1);
        assert_eq!(
            false_succ, abort_block,
            "cond_br false side must target the abort block"
        );
    }

    #[test]
    fn convert_goto_lowers_to_llvm_br() {
        // mir.goto next(%arg) -> llvm.br targeting `next`, forwarding %arg.
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let (module_ptr, entry) = build_kernel(&mut ctx, vec![i32_ty], vec![]);
        let arg = entry.deref(&ctx).get_argument(0);

        let next = append_block(&mut ctx, entry, vec![i32_ty]);
        append_mir_return(&mut ctx, next, vec![]);

        let goto = Operation::new(
            &mut ctx,
            mir::MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![arg],
            vec![next],
            0,
        );
        goto.insert_at_back(entry, &ctx);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        assert_eq!(count_ops::<mir::MirGotoOp>(&ctx, &body), 0);
        // The prologue emits its own br, so find the one targeting `next`
        // (Ptr preserved by inline_region) rather than counting all brs.
        let br = find_all::<llvm::BrOp>(&ctx, &body)
            .into_iter()
            .find(|br| {
                br.get_operation()
                    .deref(&ctx)
                    .successors()
                    .any(|s| s == next)
            })
            .expect("expected an llvm.br into `next`");
        assert_eq!(br.successor_operands(&ctx, 0).len(), 1);
    }

    #[test]
    fn convert_goto_pads_missing_zst_arg_with_undef() {
        // `next` expects (i32, ()) but the goto forwards only the i32. The
        // omitted ZST arg must be filled with a synthesised `llvm.undef` so
        // the lowered br still carries both operands.
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let unit_ty: TypeHandle = MirTupleType::get(&mut ctx, vec![]).into();
        let (module_ptr, entry) = build_kernel(&mut ctx, vec![i32_ty], vec![]);
        let arg = entry.deref(&ctx).get_argument(0);

        let next = append_block(&mut ctx, entry, vec![i32_ty, unit_ty]);
        append_mir_return(&mut ctx, next, vec![]);

        let goto = Operation::new(
            &mut ctx,
            mir::MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![arg],
            vec![next],
            0,
        );
        goto.insert_at_back(entry, &ctx);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        assert_eq!(count_ops::<mir::MirGotoOp>(&ctx, &body), 0);
        assert_eq!(
            count_ops::<llvm::UndefOp>(&ctx, &body),
            1,
            "missing ZST block arg must be filled with exactly one llvm.undef"
        );
        let br = find_all::<llvm::BrOp>(&ctx, &body)
            .into_iter()
            .find(|br| {
                br.get_operation()
                    .deref(&ctx)
                    .successors()
                    .any(|s| s == next)
            })
            .expect("expected an llvm.br into `next`");
        assert_eq!(
            br.successor_operands(&ctx, 0).len(),
            2,
            "br must forward the i32 plus the padded undef"
        );
    }

    #[test]
    fn convert_goto_errors_when_missing_arg_is_not_zst() {
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signless).into();
        let (module_ptr, entry) = build_kernel(&mut ctx, vec![i32_ty], vec![]);
        let arg = entry.deref(&ctx).get_argument(0);

        let next = append_block(&mut ctx, entry, vec![i32_ty, i64_ty]);
        append_mir_return(&mut ctx, next, vec![]);

        let goto = Operation::new(
            &mut ctx,
            mir::MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![arg],
            vec![next],
            0,
        );
        goto.insert_at_back(entry, &ctx);

        let err = crate::lower_mir_to_llvm(&mut ctx, module_ptr)
            .expect_err("goto with a non-ZST missing argument must fail to lower");
        assert!(
            err.err.to_string().contains("not a ZST"),
            "unexpected error: {}",
            err.err
        );
    }

    #[test]
    fn convert_return_multiple_operands_errors() {
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let (module_ptr, entry) =
            build_kernel(&mut ctx, vec![i32_ty, i32_ty], vec![i32_ty, i32_ty]);
        let arg0 = entry.deref(&ctx).get_argument(0);
        let arg1 = entry.deref(&ctx).get_argument(1);
        append_mir_return(&mut ctx, entry, vec![arg0, arg1]);

        let err = crate::lower_mir_to_llvm(&mut ctx, module_ptr)
            .expect_err("return with multiple operands must fail");
        assert!(
            err.err
                .to_string()
                .contains("multiple operands not supported"),
            "unexpected error: {}",
            err.err
        );
    }

    #[test]
    fn convert_cond_branch_operand_count_mismatch_errors() {
        let mut ctx = make_ctx();
        let i1_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let (module_ptr, entry) = build_kernel(&mut ctx, vec![i1_ty, i32_ty], vec![]);
        let cond = entry.deref(&ctx).get_argument(0);

        let true_block = append_block(&mut ctx, entry, vec![i32_ty]);
        let false_block = append_block(&mut ctx, entry, vec![]);
        append_mir_return(&mut ctx, true_block, vec![]);
        append_mir_return(&mut ctx, false_block, vec![]);

        let (operands, segment_sizes) =
            mir::MirCondBranchOp::compute_segment_sizes(vec![vec![cond], vec![], vec![]]);
        let cond_br = Operation::new(
            &mut ctx,
            mir::MirCondBranchOp::get_concrete_op_info(),
            vec![],
            operands,
            vec![true_block, false_block],
            0,
        );
        mir::MirCondBranchOp::new(cond_br).set_operand_segment_sizes(&ctx, segment_sizes);
        cond_br.insert_at_back(entry, &ctx);

        let err = crate::lower_mir_to_llvm(&mut ctx, module_ptr)
            .expect_err("cond_branch with operand count mismatch must fail");
        assert!(
            err.err.to_string().contains("operand count mismatch"),
            "unexpected error: {}",
            err.err
        );
    }

    #[test]
    fn convert_assert_missing_successor_errors() {
        let mut ctx = make_ctx();
        let i1_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let (module_ptr, entry) = build_kernel(&mut ctx, vec![i1_ty], vec![]);
        let cond = entry.deref(&ctx).get_argument(0);

        let (operands, segment_sizes) =
            mir::MirAssertOp::compute_segment_sizes(vec![vec![cond], vec![]]);
        let assert_op = Operation::new(
            &mut ctx,
            mir::MirAssertOp::get_concrete_op_info(),
            vec![],
            operands,
            vec![],
            0,
        );
        mir::MirAssertOp::new(assert_op).set_operand_segment_sizes(&ctx, segment_sizes);
        assert_op.insert_at_back(entry, &ctx);

        let err = crate::lower_mir_to_llvm(&mut ctx, module_ptr)
            .expect_err("assert without successor must fail");
        assert!(
            err.err.to_string().contains("exactly 1 successor"),
            "unexpected error: {}",
            err.err
        );
    }

    #[test]
    fn convert_goto_too_many_operands_errors() {
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let (module_ptr, entry) = build_kernel(&mut ctx, vec![i32_ty], vec![]);
        let arg = entry.deref(&ctx).get_argument(0);

        let next = append_block(&mut ctx, entry, vec![]);
        append_mir_return(&mut ctx, next, vec![]);

        let goto = Operation::new(
            &mut ctx,
            mir::MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![arg],
            vec![next],
            0,
        );
        goto.insert_at_back(entry, &ctx);

        let err = crate::lower_mir_to_llvm(&mut ctx, module_ptr)
            .expect_err("goto with too many operands must fail");
        assert!(
            err.err.to_string().contains("operand count mismatch"),
            "unexpected error: {}",
            err.err
        );
    }

    #[test]
    fn convert_cond_branch_forwards_distinct_args_to_each_side() {
        // Both sides take an arg, exercising the split boundary: the i32 must
        // land on the true side and the i64 on the false side, checked by
        // value identity not just counts.
        let mut ctx = make_ctx();
        let i1_ty: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signless).into();
        let (module_ptr, entry) = build_kernel(&mut ctx, vec![i1_ty, i32_ty, i64_ty], vec![]);
        let cond = entry.deref(&ctx).get_argument(0);
        let v_true = entry.deref(&ctx).get_argument(1);
        let v_false = entry.deref(&ctx).get_argument(2);

        let true_block = append_block(&mut ctx, entry, vec![i32_ty]);
        let false_block = append_block(&mut ctx, entry, vec![i64_ty]);
        append_mir_return(&mut ctx, true_block, vec![]);
        append_mir_return(&mut ctx, false_block, vec![]);

        let (operands, segment_sizes) = mir::MirCondBranchOp::compute_segment_sizes(vec![
            vec![cond],
            vec![v_true],
            vec![v_false],
        ]);
        let cond_br = Operation::new(
            &mut ctx,
            mir::MirCondBranchOp::get_concrete_op_info(),
            vec![],
            operands,
            vec![true_block, false_block],
            0,
        );
        mir::MirCondBranchOp::new(cond_br).set_operand_segment_sizes(&ctx, segment_sizes);
        cond_br.insert_at_back(entry, &ctx);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let llvm_br = find_first::<llvm::CondBrOp>(&ctx, &body).expect("expected llvm.cond_br");
        assert_eq!(llvm_br.successor_operands(&ctx, 0), vec![v_true]);
        assert_eq!(llvm_br.successor_operands(&ctx, 1), vec![v_false]);
        assert_eq!(count_ops::<mir::MirCondBranchOp>(&ctx, &body), 0);
    }
}
