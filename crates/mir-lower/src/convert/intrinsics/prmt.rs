// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte permute intrinsic conversion (`prmt.b32`).
//!
//! Lowered to inline PTX assembly (non-convergent, pure).

use llvm_export::ops::{self as llvm, AsmKind, InlineAsmOpExt};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

/// Convert `nvvm.prmt` to inline PTX.
///
/// `prmt.b32 %d, %a, %b, %c;` (per-thread data movement, non-convergent, pure).
pub(crate) fn convert_prmt(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() < 3 {
        return pliron::input_err_noloc!("prmt requires 3 operands");
    }

    let a_val = operands[0];
    let b_val = operands[1];
    let control_val = operands[2];

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);

    let inline_asm = llvm::InlineAsmOp::build(
        ctx,
        i32_ty.into(),
        vec![a_val, b_val, control_val],
        "prmt.b32 $0, $1, $2, $3;",
        "=r,r,r,r",
        AsmKind::Pure,
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

#[cfg(test)]
mod tests {
    use dialect_mir::ops as mir;
    use dialect_nvvm::ops::PrmtOp;
    use llvm_export::ops::{self as llvm, AsmKind};
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::TypeAttr;
    use pliron::builtin::op_interfaces::SymbolOpInterface;
    use pliron::builtin::ops::ModuleOp;
    use pliron::builtin::types::{FunctionType, IntegerType, Signedness};
    use pliron::context::{Context, Ptr};
    use pliron::irbuild::dialect_conversion::{
        DialectConversion, DialectConversionRewriter, OperandsInfo, apply_dialect_conversion,
    };
    use pliron::linked_list::ContainsLinkedList;
    use pliron::op::Op;
    use pliron::operation::Operation;
    use pliron::result::Result;
    use pliron::r#type::TypeHandle;

    use crate::conversion_interface::MirToLlvmConversion;

    /// A minimal conversion that converts `PrmtOp` via its `MirToLlvmConversion` impl.
    struct PrmtConversion;

    impl DialectConversion for PrmtConversion {
        fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
            Operation::get_opid(op, ctx) == PrmtOp::get_opid_static()
        }

        fn can_convert_type(&self, _ctx: &Context, _ty: TypeHandle) -> bool {
            false
        }

        fn convert_type(&mut self, _ctx: &mut Context, ty: TypeHandle) -> Result<TypeHandle> {
            Ok(ty)
        }

        fn rewrite(
            &mut self,
            ctx: &mut Context,
            rewriter: &mut DialectConversionRewriter,
            op: Ptr<Operation>,
            operands_info: &OperandsInfo,
        ) -> Result<()> {
            let prmt = Operation::get_op::<PrmtOp>(op, ctx).expect("expected PrmtOp");
            prmt.convert(ctx, rewriter, operands_info)
        }
    }

    fn make_ctx() -> Context {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        dialect_nvvm::register(&mut ctx);
        crate::register(&mut ctx);
        ctx
    }

    fn build_test_module_with_prmt(ctx: &mut Context) -> Ptr<Operation> {
        let module = ModuleOp::new(ctx, "test_module".try_into().unwrap());
        let module_ptr = module.get_operation();

        let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
        let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
        let func_ty = FunctionType::get(ctx, vec![i32_ty.into(); 3], vec![]);

        let func_op_ptr = Operation::new(
            ctx,
            mir::MirFuncOp::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let func = mir::MirFuncOp::new(ctx, func_op_ptr, TypeAttr::new(func_ty.into()));
        func.set_symbol_name(ctx, "test_fn".try_into().unwrap());

        let region = func.get_operation().deref(ctx).get_region(0);
        let entry = BasicBlock::new(ctx, None, vec![i32_ty.into(); 3]);
        entry.insert_at_back(region, ctx);

        let a = entry.deref(ctx).get_argument(0);
        let b = entry.deref(ctx).get_argument(1);
        let control = entry.deref(ctx).get_argument(2);

        // Build PrmtOp.
        let prmt_op = Operation::new(
            ctx,
            PrmtOp::get_concrete_op_info(),
            vec![u32_ty.into()],
            vec![a, b, control],
            vec![],
            0,
        );
        prmt_op.insert_at_front(entry, ctx);

        let module_region = module_ptr.deref(ctx).get_region(0);
        let module_block = module_region.deref(ctx).iter(ctx).next().unwrap();
        func.get_operation().insert_at_back(module_block, ctx);

        module_ptr
    }

    #[test]
    fn test_prmt_lowers_to_inline_asm() {
        let mut ctx = make_ctx();
        let module_ptr = build_test_module_with_prmt(&mut ctx);

        // Run the conversion.
        let mut conversion = PrmtConversion;
        apply_dialect_conversion(&mut ctx, &mut conversion, module_ptr)
            .expect("prmt conversion should succeed");

        // Navigate to the function body.
        let module_region = module_ptr.deref(&ctx).get_region(0);
        let module_block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
        let func_op = module_block
            .deref(&ctx)
            .iter(&ctx)
            .next()
            .expect("function should exist");
        let func =
            Operation::get_op::<mir::MirFuncOp>(func_op, &ctx).expect("should be a MirFuncOp");
        let func_region = func.get_operation().deref(&ctx).get_region(0);
        let entry = func_region.deref(&ctx).iter(&ctx).next().unwrap();

        // Find InlineAsmOps in the entry block.
        let asm_ops: Vec<llvm::InlineAsmOp> = entry
            .deref(&ctx)
            .iter(&ctx)
            .filter_map(|op: Ptr<Operation>| Operation::get_op::<llvm::InlineAsmOp>(op, &ctx))
            .collect();
        assert_eq!(asm_ops.len(), 1, "expected exactly one InlineAsmOp");

        let asm = &asm_ops[0];

        // Verify the PTX template.
        assert_eq!(
            asm.get_attr_inline_asm_template(&ctx)
                .map(|s| String::from((*s).clone()))
                .as_deref(),
            Some("prmt.b32 $0, $1, $2, $3;")
        );

        // Verify the constraints.
        assert_eq!(
            asm.get_attr_inline_asm_constraints(&ctx)
                .map(|s| String::from((*s).clone()))
                .as_deref(),
            Some("=r,r,r,r")
        );

        // Verify it uses pure (no side effects, not convergent).
        assert_eq!(
            llvm::asm_kind_opt(&ctx, asm),
            Some(AsmKind::Pure),
            "prmt should use Pure asm kind"
        );
    }
}
