/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Operation emission for LLVM IR.

use std::cell::Ref;
use std::collections::HashMap;
use std::fmt::Write;

use pliron::r#type::Typed;
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{BoolAttr, FPDoubleAttr, FPSingleAttr, IntegerAttr, StringAttr},
        op_interfaces::{CallOpCallable, CallOpInterface},
    },
    context::Ptr,
    op::Op,
    operation::Operation,
    value::Value,
};

use crate::{
    attributes::{
        AtomicOrderingAttr, AtomicRmwKindAttr, FCmpPredicateAttr, FPHalfAttr, GepIndexAttr,
        ICmpPredicateAttr,
    },
    ops,
    types::{FuncType, VoidType},
};

use super::{
    literals::{format_float_literal, format_half_literal},
    state::ModuleExportState,
};

/// Typed view of a dispatched LLVM dialect operation.
///
/// Each variant wraps a reference to the typed op so that handler methods can
/// call op-specific accessors (predicates, ordering, asm template, etc.)
/// without repeating the downcast inside every arm.
///
/// # Maintenance contract
///
/// Wiring an LLVM dialect op (defined upstream in `pliron-llvm`) into
/// textual export touches four places in this file:
///
/// 1. Add a variant to `LlvmOp` below.
/// 2. Add a matching entry in the [`classify_op!`] invocation.
/// 3. Add an `emit_*` helper method on [`ModuleExportState`].
/// 4. Add a `Some(LlvmOp::X(op)) => self.emit_x(...)` arm in `export_op`.
///
/// Selective dispatch sites elsewhere in the export pipeline (e.g. the
/// value-name pre-pass in `function.rs`) inspect specific op types via
/// `downcast_ref` directly and do not need to be updated.
enum LlvmOp<'op> {
    // Terminators
    Return(&'op ops::ReturnOp),
    /// `UnreachableOp` carries no operands or attributes, so the typed reference
    /// is intentionally unread. The variant is kept tuple-shaped for uniformity
    /// with the other terminator variants.
    #[allow(dead_code)]
    Unreachable(&'op ops::UnreachableOp),
    Br(&'op ops::BrOp),
    CondBr(&'op ops::CondBrOp),
    // Memory
    Load(&'op ops::LoadOp),
    Store(&'op ops::StoreOp),
    Alloca(&'op ops::AllocaOp),
    GetElementPtr(&'op ops::GetElementPtrOp),
    // Atomics
    AtomicLoad(&'op ops::AtomicLoadOp),
    AtomicStore(&'op ops::AtomicStoreOp),
    AtomicRmw(&'op ops::AtomicRmwOp),
    AtomicCmpxchg(&'op ops::AtomicCmpxchgOp),
    Fence(&'op ops::FenceOp),
    // Integer arithmetic
    Add(&'op ops::AddOp),
    Sub(&'op ops::SubOp),
    Mul(&'op ops::MulOp),
    SDiv(&'op ops::SDivOp),
    UDiv(&'op ops::UDivOp),
    SRem(&'op ops::SRemOp),
    URem(&'op ops::URemOp),
    Shl(&'op ops::ShlOp),
    LShr(&'op ops::LShrOp),
    AShr(&'op ops::AShrOp),
    And(&'op ops::AndOp),
    Or(&'op ops::OrOp),
    Xor(&'op ops::XorOp),
    // Float arithmetic
    FAdd(&'op ops::FAddOp),
    FSub(&'op ops::FSubOp),
    FMul(&'op ops::FMulOp),
    FDiv(&'op ops::FDivOp),
    FRem(&'op ops::FRemOp),
    FNeg(&'op ops::FNegOp),
    // Comparison / select
    ICmp(&'op ops::ICmpOp),
    FCmp(&'op ops::FCmpOp),
    Select(&'op ops::SelectOp),
    // Calls and inline assembly
    Call(&'op ops::CallOp),
    InlineAsm(&'op ops::InlineAsmOp),
    // Casts
    Bitcast(&'op ops::BitcastOp),
    AddrSpaceCast(&'op ops::AddrSpaceCastOp),
    ZExt(&'op ops::ZExtOp),
    SExt(&'op ops::SExtOp),
    Trunc(&'op ops::TruncOp),
    PtrToInt(&'op ops::PtrToIntOp),
    IntToPtr(&'op ops::IntToPtrOp),
    UIToFP(&'op ops::UIToFPOp),
    SIToFP(&'op ops::SIToFPOp),
    FPToUI(&'op ops::FPToUIOp),
    FPToSI(&'op ops::FPToSIOp),
    FPExt(&'op ops::FPExtOp),
    FPTrunc(&'op ops::FPTruncOp),
    // Aggregates
    ExtractValue(&'op ops::ExtractValueOp),
    InsertValue(&'op ops::InsertValueOp),
    // Virtual / constant ops
    Undef(&'op ops::UndefOp),
    Constant(&'op ops::ConstantOp),
    AddressOf(&'op ops::AddressOfOp),
}

/// Try each `(Variant, OpType)` pair in order; return the first match.
///
/// Uses `return` to short-circuit out of the enclosing `try_from` body.
macro_rules! classify_op {
    ($op_obj:expr, { $($Variant:ident => $OpTy:ty),* $(,)? }) => {{
        let op = $op_obj;
        $(
            if let Some(inner) = op.downcast_ref::<$OpTy>() {
                return Ok(Self::$Variant(inner));
            }
        )*
        Err(())
    }};
}

impl<'op> TryFrom<&'op dyn Op> for LlvmOp<'op> {
    type Error = ();

    fn try_from(op_obj: &'op dyn Op) -> Result<Self, ()> {
        classify_op!(op_obj, {
            // Terminators
            Return       => ops::ReturnOp,
            Unreachable  => ops::UnreachableOp,
            Br           => ops::BrOp,
            CondBr       => ops::CondBrOp,
            // Memory
            Load         => ops::LoadOp,
            Store        => ops::StoreOp,
            Alloca       => ops::AllocaOp,
            GetElementPtr=> ops::GetElementPtrOp,
            // Atomics
            AtomicLoad   => ops::AtomicLoadOp,
            AtomicStore  => ops::AtomicStoreOp,
            AtomicRmw    => ops::AtomicRmwOp,
            AtomicCmpxchg=> ops::AtomicCmpxchgOp,
            Fence        => ops::FenceOp,
            // Integer arithmetic
            Add          => ops::AddOp,
            Sub          => ops::SubOp,
            Mul          => ops::MulOp,
            SDiv         => ops::SDivOp,
            UDiv         => ops::UDivOp,
            SRem         => ops::SRemOp,
            URem         => ops::URemOp,
            Shl          => ops::ShlOp,
            LShr         => ops::LShrOp,
            AShr         => ops::AShrOp,
            And          => ops::AndOp,
            Or           => ops::OrOp,
            Xor          => ops::XorOp,
            // Float arithmetic
            FAdd         => ops::FAddOp,
            FSub         => ops::FSubOp,
            FMul         => ops::FMulOp,
            FDiv         => ops::FDivOp,
            FRem         => ops::FRemOp,
            FNeg         => ops::FNegOp,
            // Comparison / select
            ICmp         => ops::ICmpOp,
            FCmp         => ops::FCmpOp,
            Select       => ops::SelectOp,
            // Calls and inline assembly
            Call         => ops::CallOp,
            InlineAsm    => ops::InlineAsmOp,
            // Casts
            Bitcast      => ops::BitcastOp,
            AddrSpaceCast=> ops::AddrSpaceCastOp,
            ZExt         => ops::ZExtOp,
            SExt         => ops::SExtOp,
            Trunc        => ops::TruncOp,
            PtrToInt     => ops::PtrToIntOp,
            IntToPtr     => ops::IntToPtrOp,
            UIToFP       => ops::UIToFPOp,
            SIToFP       => ops::SIToFPOp,
            FPToUI       => ops::FPToUIOp,
            FPToSI       => ops::FPToSIOp,
            FPExt        => ops::FPExtOp,
            FPTrunc      => ops::FPTruncOp,
            // Aggregates
            ExtractValue => ops::ExtractValueOp,
            InsertValue  => ops::InsertValueOp,
            // Virtual / constant ops
            Undef        => ops::UndefOp,
            Constant     => ops::ConstantOp,
            AddressOf    => ops::AddressOfOp,
        })
    }
}

impl<'a> ModuleExportState<'a> {
    pub(super) fn export_op(
        &mut self,
        op: Ptr<Operation>,
        value_names: &mut HashMap<Value, String>,
        next_value_id: &mut usize,
        block_labels: &HashMap<Ptr<BasicBlock>, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.deref(self.ctx);
        let op_obj = Operation::get_op_dyn(op, self.ctx);

        // Register result names (skip if already named in pre-pass)
        for res in op_ref.results() {
            value_names.entry(res).or_insert_with(|| {
                let name = format!("%v{next_value_id}");
                *next_value_id += 1;
                name.clone()
            });
        }

        match LlvmOp::try_from(op_obj.as_ref()).ok() {
            // Terminators
            Some(LlvmOp::Return(op)) => self.emit_return(op, value_names, output)?,
            Some(LlvmOp::Unreachable(_)) => writeln!(output, "  unreachable").unwrap(),
            Some(LlvmOp::Br(op)) => self.emit_br(op, block_labels, output)?,
            Some(LlvmOp::CondBr(op)) => self.emit_cond_br(op, value_names, block_labels, output)?,
            // Memory
            Some(LlvmOp::Load(op)) => self.emit_load(op, value_names, output)?,
            Some(LlvmOp::Store(op)) => self.emit_store(op, value_names, output)?,
            Some(LlvmOp::Alloca(op)) => self.emit_alloca(op, value_names, output)?,
            Some(LlvmOp::GetElementPtr(op)) => self.emit_gep(op, value_names, output)?,
            // Atomics
            Some(LlvmOp::AtomicLoad(op)) => self.emit_atomic_load(op, value_names, output)?,
            Some(LlvmOp::AtomicStore(op)) => self.emit_atomic_store(op, value_names, output)?,
            Some(LlvmOp::AtomicRmw(op)) => self.emit_atomic_rmw(op, value_names, output)?,
            Some(LlvmOp::AtomicCmpxchg(op)) => self.emit_atomic_cmpxchg(op, value_names, output)?,
            Some(LlvmOp::Fence(op)) => self.emit_fence(op, output)?,
            // Integer arithmetic (all map to export_binop)
            Some(LlvmOp::Add(op)) => {
                self.export_binop("add", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::Sub(op)) => {
                self.export_binop("sub", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::Mul(op)) => {
                self.export_binop("mul", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::SDiv(op)) => {
                self.export_binop("sdiv", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::UDiv(op)) => {
                self.export_binop("udiv", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::SRem(op)) => {
                self.export_binop("srem", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::URem(op)) => {
                self.export_binop("urem", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::Shl(op)) => {
                self.export_binop("shl", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::LShr(op)) => {
                self.export_binop("lshr", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::AShr(op)) => {
                self.export_binop("ashr", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::And(op)) => {
                self.export_binop("and", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::Or(op)) => {
                self.export_binop("or", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::Xor(op)) => {
                self.export_binop("xor", op.get_operation(), value_names, output)?
            }
            // Float arithmetic
            Some(LlvmOp::FAdd(op)) => {
                self.export_binop("fadd", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FSub(op)) => {
                self.export_binop("fsub", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FMul(op)) => {
                self.export_binop("fmul", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FDiv(op)) => {
                self.export_binop("fdiv", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FRem(op)) => {
                self.export_binop("frem", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FNeg(op)) => self.emit_fneg(op, value_names, output)?,
            // Comparison / select
            Some(LlvmOp::ICmp(op)) => self.emit_icmp(op, value_names, output)?,
            Some(LlvmOp::FCmp(op)) => self.emit_fcmp(op, value_names, output)?,
            Some(LlvmOp::Select(op)) => self.emit_select(op, value_names, output)?,
            // Calls and inline assembly
            Some(LlvmOp::Call(op)) => self.emit_call(op, value_names, output)?,
            Some(LlvmOp::InlineAsm(op)) => self.emit_inline_asm(op, value_names, output)?,
            // Casts
            Some(LlvmOp::Bitcast(op)) => {
                self.export_cast("bitcast", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::AddrSpaceCast(op)) => {
                self.export_cast("addrspacecast", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::ZExt(op)) => self.emit_zext(op, value_names, output)?,
            Some(LlvmOp::SExt(op)) => {
                self.export_cast("sext", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::Trunc(op)) => {
                self.export_cast("trunc", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::PtrToInt(op)) => {
                self.export_cast("ptrtoint", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::IntToPtr(op)) => {
                self.export_cast("inttoptr", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::UIToFP(op)) => {
                self.export_cast("uitofp", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::SIToFP(op)) => {
                self.export_cast("sitofp", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FPToUI(op)) => {
                self.export_cast("fptoui", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FPToSI(op)) => {
                self.export_cast("fptosi", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FPExt(op)) => {
                self.export_cast("fpext", op.get_operation(), value_names, output)?
            }
            Some(LlvmOp::FPTrunc(op)) => {
                self.export_cast("fptrunc", op.get_operation(), value_names, output)?
            }
            // Aggregates
            Some(LlvmOp::ExtractValue(op)) => self.emit_extract_value(op, value_names, output)?,
            Some(LlvmOp::InsertValue(op)) => self.emit_insert_value(op, value_names, output)?,
            // Virtual ops
            Some(LlvmOp::Undef(op)) => self.emit_undef(op, value_names),
            Some(LlvmOp::Constant(op)) => self.emit_constant(op, value_names),
            Some(LlvmOp::AddressOf(op)) => self.emit_address_of(op, value_names),
            // Unknown
            None => writeln!(
                output,
                "  ; Unknown op: {}",
                Operation::get_opid(op, self.ctx)
            )
            .unwrap(),
        }

        Ok(())
    }

    fn emit_return(
        &mut self,
        op: &ops::ReturnOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        write!(output, "  ret ").unwrap();
        if op_ref.operands().count() == 0 {
            write!(output, "void").unwrap();
        } else {
            let val = op_ref.operands().next().unwrap();
            self.export_type(val.get_type(self.ctx), output)?;
            write!(output, " ").unwrap();
            self.export_value(val, value_names, output)?;
        }
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_br(
        &self,
        op: &ops::BrOp,
        block_labels: &HashMap<Ptr<BasicBlock>, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let dest = op_ref.successors().next().unwrap();
        let label = block_labels.get(&dest).ok_or("Missing block label")?;
        writeln!(output, "  br label %{label}").unwrap();
        Ok(())
    }

    fn emit_cond_br(
        &mut self,
        op: &ops::CondBrOp,
        value_names: &HashMap<Value, String>,
        block_labels: &HashMap<Ptr<BasicBlock>, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let mut succs = op_ref.successors();
        let true_dest = succs.next().unwrap();
        let false_dest = succs.next().unwrap();
        let true_label = block_labels.get(&true_dest).ok_or("Missing true label")?;
        let false_label = block_labels.get(&false_dest).ok_or("Missing false label")?;
        let cond = op_ref.get_operand(0);

        write!(output, "  br i1 ").unwrap();
        self.export_value(cond, value_names, output)?;
        writeln!(output, ", label %{true_label}, label %{false_label}").unwrap();
        Ok(())
    }

    fn emit_load(
        &mut self,
        op: &ops::LoadOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let ptr = op_ref.get_operand(0);
        let res_name = value_names.get(&res).unwrap();
        let ty = res.get_type(self.ctx);
        let addrspace = addrspace_of(ptr.get_type(self.ctx), self.ctx);

        write!(output, "  {res_name} = load ").unwrap();
        self.export_type(ty, output)?;
        write!(output, ", {}", ptr_qualifier(addrspace)).unwrap();
        self.export_value(ptr, value_names, output)?;
        let align = crate::ops::op_alignment(self.ctx, op.get_operation())
            .unwrap_or_else(|| self.natural_alignment(ty));
        writeln!(output, ", align {align}").unwrap();
        Ok(())
    }

    fn emit_store(
        &mut self,
        op: &ops::StoreOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let val = op_ref.get_operand(0);
        let ptr = op_ref.get_operand(1);
        let val_ty = val.get_type(self.ctx);
        let addrspace = addrspace_of(ptr.get_type(self.ctx), self.ctx);

        write!(output, "  store ").unwrap();
        self.export_type(val_ty, output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        write!(output, ", {}", ptr_qualifier(addrspace)).unwrap();
        self.export_value(ptr, value_names, output)?;
        let align = crate::ops::op_alignment(self.ctx, op.get_operation())
            .unwrap_or_else(|| self.natural_alignment(val_ty));
        writeln!(output, ", align {align}").unwrap();
        Ok(())
    }

    fn emit_alloca(
        &mut self,
        op: &ops::AllocaOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let elem_ty = op
            .get_attr_alloca_element_type(self.ctx)
            .expect("Missing alloca_element_type");

        let elem_llvm_ty = elem_ty.get_type(self.ctx);
        write!(output, "  {res_name} = alloca ").unwrap();
        self.export_type(elem_llvm_ty, output)?;
        let align = crate::ops::op_alignment(self.ctx, op.get_operation())
            .unwrap_or_else(|| self.natural_alignment(elem_llvm_ty));
        writeln!(output, ", align {align}").unwrap();
        Ok(())
    }

    fn emit_gep(
        &mut self,
        op: &ops::GetElementPtrOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let ptr = op_ref.get_operand(0);
        let elem_ty = op
            .get_attr_gep_src_elem_type(self.ctx)
            .expect("Missing gep_src_elem_type")
            .get_type(self.ctx);
        let addrspace = addrspace_of(ptr.get_type(self.ctx), self.ctx);

        write!(output, "  {res_name} = getelementptr inbounds ").unwrap();
        self.export_type(elem_ty, output)?;
        write!(output, ", {}", ptr_qualifier(addrspace)).unwrap();
        self.export_value(ptr, value_names, output)?;

        for idx_attr in &op.get_attr_gep_indices(self.ctx).unwrap().0 {
            write!(output, ", ").unwrap();
            match idx_attr {
                GepIndexAttr::Constant(val) => {
                    write!(output, "i32 {val}").unwrap();
                }
                GepIndexAttr::OperandIdx(operand_idx) => {
                    let val = op_ref.get_operand(*operand_idx);
                    self.export_type(val.get_type(self.ctx), output)?;
                    write!(output, " ").unwrap();
                    self.export_value(val, value_names, output)?;
                }
            }
        }
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_atomic_load(
        &mut self,
        op: &ops::AtomicLoadOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let ptr = op_ref.get_operand(0);
        let res_name = value_names.get(&res).unwrap();
        let ty = res.get_type(self.ctx);
        let syncscope = fmt_syncscope(op.get_attr_llvm_ld_syncscope(self.ctx));
        let ordering = fmt_ordering(op.get_attr_llvm_ld_ordering(self.ctx));
        let addrspace = addrspace_of(ptr.get_type(self.ctx), self.ctx);

        write!(output, "  {res_name} = load atomic ").unwrap();
        self.export_type(ty, output)?;
        write!(output, ", {}", ptr_qualifier(addrspace)).unwrap();
        self.export_value(ptr, value_names, output)?;
        let align = self.natural_alignment(ty);
        writeln!(output, "{syncscope} {ordering}, align {align}").unwrap();
        Ok(())
    }

    fn emit_atomic_store(
        &mut self,
        op: &ops::AtomicStoreOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let val = op_ref.get_operand(0);
        let ptr = op_ref.get_operand(1);
        let syncscope = fmt_syncscope(op.get_attr_llvm_st_syncscope(self.ctx));
        let ordering = fmt_ordering(op.get_attr_llvm_st_ordering(self.ctx));
        let addrspace = addrspace_of(ptr.get_type(self.ctx), self.ctx);

        write!(output, "  store atomic ").unwrap();
        self.export_type(val.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        write!(output, ", {}", ptr_qualifier(addrspace)).unwrap();
        self.export_value(ptr, value_names, output)?;
        let align = self.natural_alignment(val.get_type(self.ctx));
        writeln!(output, "{syncscope} {ordering}, align {align}").unwrap();
        Ok(())
    }

    fn emit_atomic_rmw(
        &mut self,
        op: &ops::AtomicRmwOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let ptr = op_ref.get_operand(0);
        let val = op_ref.get_operand(1);
        let res_name = value_names.get(&res).unwrap();
        let rmw_kind = fmt_rmw_kind(op.get_attr_llvm_rmw_kind(self.ctx));
        let syncscope = fmt_syncscope(op.get_attr_llvm_rmw_syncscope(self.ctx));
        let ordering = fmt_ordering(op.get_attr_llvm_rmw_ordering(self.ctx));
        let addrspace = addrspace_of(ptr.get_type(self.ctx), self.ctx);

        write!(output, "  {res_name} = atomicrmw {rmw_kind} ").unwrap();
        write!(output, "{}", ptr_qualifier(addrspace)).unwrap();
        self.export_value(ptr, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_type(val.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        writeln!(output, "{syncscope} {ordering}").unwrap();
        Ok(())
    }

    fn emit_atomic_cmpxchg(
        &mut self,
        op: &ops::AtomicCmpxchgOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let ptr = op_ref.get_operand(0);
        let cmp = op_ref.get_operand(1);
        let new_val = op_ref.get_operand(2);
        let res_name = value_names.get(&res).unwrap();
        let success_ord = fmt_ordering(op.get_attr_llvm_cas_success_ordering(self.ctx));
        let failure_ord = fmt_ordering(op.get_attr_llvm_cas_failure_ordering(self.ctx));
        let syncscope = fmt_syncscope(op.get_attr_llvm_cas_syncscope(self.ctx));
        let val_ty = cmp.get_type(self.ctx);
        let addrspace = addrspace_of(ptr.get_type(self.ctx), self.ctx);

        // pliron-llvm's cmpxchg result is the full `{ T, i1 }` struct; a
        // separate `extractvalue` op (emitted on its own) pulls out the loaded
        // value, so here we emit only the cmpxchg into the struct-typed result.
        write!(output, "  {res_name} = cmpxchg ").unwrap();
        write!(output, "{}", ptr_qualifier(addrspace)).unwrap();
        self.export_value(ptr, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_type(val_ty, output)?;
        write!(output, " ").unwrap();
        self.export_value(cmp, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_type(val_ty, output)?;
        write!(output, " ").unwrap();
        self.export_value(new_val, value_names, output)?;
        writeln!(output, "{syncscope} {success_ord} {failure_ord}").unwrap();
        Ok(())
    }

    fn emit_fence(&self, op: &ops::FenceOp, output: &mut String) -> Result<(), String> {
        let syncscope = fmt_syncscope(op.get_attr_llvm_fence_syncscope(self.ctx));
        let ordering = fmt_ordering(op.get_attr_llvm_fence_ordering(self.ctx));
        writeln!(output, "  fence{syncscope} {ordering}").unwrap();
        Ok(())
    }

    fn emit_fneg(
        &mut self,
        op: &ops::FNegOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let arg = op_ref.get_operand(0);

        write!(output, "  {res_name} = fneg ").unwrap();
        self.export_type(arg.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(arg, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_icmp(
        &mut self,
        op: &ops::ICmpOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let lhs = op_ref.get_operand(0);
        let rhs = op_ref.get_operand(1);
        let pred = match op.predicate(self.ctx) {
            ICmpPredicateAttr::EQ => "eq",
            ICmpPredicateAttr::NE => "ne",
            ICmpPredicateAttr::SLT => "slt",
            ICmpPredicateAttr::SLE => "sle",
            ICmpPredicateAttr::SGT => "sgt",
            ICmpPredicateAttr::SGE => "sge",
            ICmpPredicateAttr::ULT => "ult",
            ICmpPredicateAttr::ULE => "ule",
            ICmpPredicateAttr::UGT => "ugt",
            ICmpPredicateAttr::UGE => "uge",
        };

        write!(output, "  {res_name} = icmp {pred} ").unwrap();
        self.export_type(lhs.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(lhs, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_value(rhs, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_fcmp(
        &mut self,
        op: &ops::FCmpOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let lhs = op_ref.get_operand(0);
        let rhs = op_ref.get_operand(1);
        let pred = match op.predicate(self.ctx) {
            FCmpPredicateAttr::False => "false",
            FCmpPredicateAttr::OEQ => "oeq",
            FCmpPredicateAttr::OGT => "ogt",
            FCmpPredicateAttr::OGE => "oge",
            FCmpPredicateAttr::OLT => "olt",
            FCmpPredicateAttr::OLE => "ole",
            FCmpPredicateAttr::ONE => "one",
            FCmpPredicateAttr::ORD => "ord",
            FCmpPredicateAttr::UEQ => "ueq",
            FCmpPredicateAttr::UGT => "ugt",
            FCmpPredicateAttr::UGE => "uge",
            FCmpPredicateAttr::ULT => "ult",
            FCmpPredicateAttr::ULE => "ule",
            FCmpPredicateAttr::UNE => "une",
            FCmpPredicateAttr::UNO => "uno",
            FCmpPredicateAttr::True => "true",
        };

        write!(output, "  {res_name} = fcmp {pred} ").unwrap();
        self.export_type(lhs.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(lhs, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_value(rhs, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_select(
        &mut self,
        op: &ops::SelectOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let cond = op_ref.get_operand(0);
        let true_val = op_ref.get_operand(1);
        let false_val = op_ref.get_operand(2);
        let val_ty = true_val.get_type(self.ctx);

        write!(output, "  {res_name} = select i1 ").unwrap();
        self.export_value(cond, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_type(val_ty, output)?;
        write!(output, " ").unwrap();
        self.export_value(true_val, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_type(val_ty, output)?;
        write!(output, " ").unwrap();
        self.export_value(false_val, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_call(
        &mut self,
        op: &ops::CallOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let callee = op.callee(self.ctx);
        let func_ty = op.callee_type(self.ctx);
        let func_ty_ref = func_ty.deref(self.ctx);
        let llvm_func_ty = func_ty_ref.downcast_ref::<FuncType>().unwrap();
        let ret_ty = llvm_func_ty.result_type();
        let is_void = ret_ty.deref(self.ctx).is::<VoidType>();

        // Void calls: "call void @func(...)"
        // Non-void:   "%vN = call <type> @func(...)"
        if is_void {
            write!(output, "  call void").unwrap();
        } else {
            let res = op_ref.get_result(0);
            let res_name = value_names.get(&res).unwrap();
            write!(output, "  {res_name} = call ").unwrap();
            self.export_type(ret_ty, output)?;
        }

        match callee {
            CallOpCallable::Direct(identifier) => {
                let name = identifier.to_string();
                // LLVM intrinsics use dots in IR; Pliron IR identifiers use underscores.
                let fixed = if name.starts_with("llvm_") {
                    name.replace('_', ".")
                } else {
                    super::names::strip_device_prefix(&name)
                };
                write!(output, " @{fixed}(").unwrap();
            }
            CallOpCallable::Indirect(val) => {
                write!(output, " ").unwrap();
                self.export_value(val, value_names, output).unwrap();
                write!(output, "(").unwrap();
            }
        }

        for (i, arg) in op_ref.operands().enumerate() {
            if i > 0 {
                write!(output, ", ").unwrap();
            }
            self.export_type(arg.get_type(self.ctx), output)?;
            write!(output, " ").unwrap();
            self.export_value(arg, value_names, output)?;
        }

        // Every device call is emitted `convergent` (attr group #0). GPU code is
        // convergent-by-default (as in Clang/nvcc): if the callee transitively
        // performs a barrier / shuffle / vote, `opt -O2` must not sink or
        // duplicate the call across divergent control flow. opt strips the
        // attribute from calls it proves never reach a convergent op.
        writeln!(output, ") #0").unwrap();
        self.convergent_used = true;
        Ok(())
    }

    fn emit_inline_asm(
        &mut self,
        op: &ops::InlineAsmOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let asm_template = read_string_attr(op.get_attr_inline_asm_template(self.ctx));
        let constraints = read_string_attr(op.get_attr_inline_asm_constraints(self.ctx));
        let is_convergent = read_bool_attr(op.get_attr_inline_asm_convergent(self.ctx));

        // pliron-llvm always stores a single result slot (a void result for
        // no-value asm), so decide void vs valued by the result *type*, not the
        // result count.
        let res = op_ref.get_result(0);
        let res_ty = res.get_type(self.ctx);
        if res_ty.deref(self.ctx).is::<VoidType>() {
            write!(output, "  call void").unwrap();
        } else {
            let res_name = value_names.get(&res).unwrap();
            write!(output, "  {res_name} = call ").unwrap();
            self.export_type(res_ty, output)?;
        }

        write!(
            output,
            " asm sideeffect \"{asm_template}\", \"{constraints}\"("
        )
        .unwrap();
        for (i, arg) in op_ref.operands().enumerate() {
            if i > 0 {
                write!(output, ", ").unwrap();
            }
            self.export_type(arg.get_type(self.ctx), output)?;
            write!(output, " ").unwrap();
            self.export_value(arg, value_names, output)?;
        }

        if is_convergent {
            writeln!(output, ") #0").unwrap();
            self.convergent_used = true;
        } else {
            writeln!(output, ")").unwrap();
        }
        Ok(())
    }

    fn emit_zext(
        &mut self,
        op: &ops::ZExtOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let val = op_ref.get_operand(0);

        // Manual attribute access since helper is missing
        let nneg_key: pliron::identifier::Identifier = "llvm_nneg_flag".try_into().unwrap();
        let nneg = op_ref
            .attributes
            .0
            .get(&nneg_key)
            .and_then(|attr| {
                attr.downcast_ref::<pliron::builtin::attributes::BoolAttr>()
                    .map(|b| bool::from(b.clone()))
            })
            .unwrap_or(false);

        write!(output, "  {res_name} = zext ").unwrap();
        if nneg {
            write!(output, "nneg ").unwrap();
        }
        self.export_type(val.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        write!(output, " to ").unwrap();
        self.export_type(res.get_type(self.ctx), output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_extract_value(
        &mut self,
        op: &ops::ExtractValueOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let agg = op_ref.get_operand(0);

        write!(output, "  {res_name} = extractvalue ").unwrap();
        self.export_type(agg.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(agg, value_names, output)?;
        for idx in op.indices(self.ctx) {
            write!(output, ", {idx}").unwrap();
        }
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_insert_value(
        &mut self,
        op: &ops::InsertValueOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.get_operation().deref(self.ctx);
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let agg = op_ref.get_operand(0);
        let val = op_ref.get_operand(1);

        write!(output, "  {res_name} = insertvalue ").unwrap();
        self.export_type(agg.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(agg, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_type(val.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        for idx in op.indices(self.ctx) {
            write!(output, ", {idx}").unwrap();
        }
        writeln!(output).unwrap();
        Ok(())
    }

    fn emit_undef(&self, op: &ops::UndefOp, value_names: &mut HashMap<Value, String>) {
        let res = op.get_operation().deref(self.ctx).get_result(0);
        value_names.insert(res, "undef".to_string());
    }

    fn emit_constant(&self, op: &ops::ConstantOp, value_names: &mut HashMap<Value, String>) {
        let val_attr = op.get_value(self.ctx);
        let const_str = if let Some(int_attr) = val_attr.downcast_ref::<IntegerAttr>() {
            // Use APInt's proper decimal string conversion instead of parsing debug format.
            // The old code parsed debug strings like "APInt { value: 0x4000_0000_0000_u64 }"
            // by splitting on '_', which broke for values with underscore grouping
            // (e.g., 1u64 << 46 = 0x4000_0000_0000 would become 0x4000 = 16384).
            int_attr.value().to_string_unsigned_decimal()
        } else if let Some(fp16_attr) = val_attr.downcast_ref::<FPHalfAttr>() {
            format_half_literal(crate::fp16_attr_to_bits(fp16_attr))
        } else if let Some(fp32_attr) = val_attr.downcast_ref::<FPSingleAttr>() {
            let float_val: f32 = fp32_attr.clone().into();
            format_float_literal(f64::from(float_val))
        } else if let Some(fp64_attr) = val_attr.downcast_ref::<FPDoubleAttr>() {
            let float_val: f64 = fp64_attr.clone().into();
            format_float_literal(float_val)
        } else {
            "0".to_string() // Fallback
        };

        let res = op.get_operation().deref(self.ctx).get_result(0);
        value_names.insert(res, const_str);
    }

    fn emit_address_of(&self, op: &ops::AddressOfOp, value_names: &HashMap<Value, String>) {
        // AddressOfOp is virtual in textual LLVM IR: every use site prints the
        // global symbol directly. The naming pre-pass in export_function
        // registers the result as `@<global_name>` before any block is emitted,
        // so there is nothing to emit here. The assertion keeps the contract
        // honest if the pre-pass is ever refactored.
        let res = op.get_operation().deref(self.ctx).get_result(0);
        debug_assert!(
            value_names
                .get(&res)
                .is_some_and(|name| name.starts_with('@')),
            "AddressOfOp result must be pre-registered as a global symbol by \
             the naming pre-pass; got {:?}",
            value_names.get(&res),
        );
    }

    pub(super) fn export_binop(
        &self,
        op_name: &str,
        op: Ptr<Operation>,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.deref(self.ctx);
        let res = op_ref.get_result(0);
        let lhs = op_ref.get_operand(0);
        let rhs = op_ref.get_operand(1);
        let res_name = value_names.get(&res).unwrap();

        write!(output, "  {res_name} = {op_name} ").unwrap();
        self.export_type(lhs.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(lhs, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_value(rhs, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    /// Export a cast: `%res = <op_name> <src_type> <val> to <dst_type>`
    pub(super) fn export_cast(
        &self,
        op_name: &str,
        op: Ptr<Operation>,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.deref(self.ctx);
        let res = op_ref.get_result(0);
        let val = op_ref.get_operand(0);
        let res_name = value_names.get(&res).unwrap();

        write!(output, "  {res_name} = {op_name} ").unwrap();
        self.export_type(val.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        write!(output, " to ").unwrap();
        self.export_type(res.get_type(self.ctx), output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    pub(super) fn export_value(
        &self,
        val: Value,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        if let Some(name) = value_names.get(&val) {
            write!(output, "{name}").unwrap();
        } else {
            write!(output, "undef").unwrap();
        }
        Ok(())
    }
}

/// Return the address space of a pointer type, or 0 for non-pointer types.
fn addrspace_of(ty: Ptr<pliron::r#type::TypeObj>, ctx: &pliron::context::Context) -> u32 {
    ty.deref(ctx)
        .downcast_ref::<crate::types::PointerType>()
        .map_or(0, crate::types::PointerType::address_space)
}

/// Format the pointer operand prefix for memory instructions.
///
/// Returns `"ptr addrspace(N) "` for non-default address spaces, `"ptr "` otherwise.
fn ptr_qualifier(addrspace: u32) -> String {
    if addrspace != 0 {
        format!("ptr addrspace({addrspace}) ")
    } else {
        "ptr ".to_string()
    }
}

/// Read an optional `StringAttr` to an owned `String` (absent → empty).
fn read_string_attr(attr: Option<Ref<StringAttr>>) -> String {
    attr.map(|s| String::from((*s).clone())).unwrap_or_default()
}

/// Read an optional `BoolAttr` to a `bool` (absent → false).
fn read_bool_attr(attr: Option<Ref<BoolAttr>>) -> bool {
    attr.map(|b| bool::from((*b).clone())).unwrap_or(false)
}

/// LLVM mnemonic for an atomic ordering. Ordering is always present on atomic
/// ops; `monotonic` is LLVM's weakest default if somehow absent.
fn fmt_ordering(ord: Option<Ref<AtomicOrderingAttr>>) -> &'static str {
    match ord.as_deref() {
        Some(AtomicOrderingAttr::Acquire) => "acquire",
        Some(AtomicOrderingAttr::Release) => "release",
        Some(AtomicOrderingAttr::AcqRel) => "acq_rel",
        Some(AtomicOrderingAttr::SeqCst) => "seq_cst",
        Some(AtomicOrderingAttr::Monotonic) | None => "monotonic",
    }
}

/// LLVM mnemonic for an `atomicrmw` operation.
fn fmt_rmw_kind(kind: Option<Ref<AtomicRmwKindAttr>>) -> &'static str {
    match kind.as_deref() {
        Some(AtomicRmwKindAttr::Xchg) | None => "xchg",
        Some(AtomicRmwKindAttr::Add) => "add",
        Some(AtomicRmwKindAttr::Sub) => "sub",
        Some(AtomicRmwKindAttr::And) => "and",
        Some(AtomicRmwKindAttr::Nand) => "nand",
        Some(AtomicRmwKindAttr::Or) => "or",
        Some(AtomicRmwKindAttr::Xor) => "xor",
        Some(AtomicRmwKindAttr::Max) => "max",
        Some(AtomicRmwKindAttr::Min) => "min",
        Some(AtomicRmwKindAttr::UMax) => "umax",
        Some(AtomicRmwKindAttr::UMin) => "umin",
        Some(AtomicRmwKindAttr::FAdd) => "fadd",
        Some(AtomicRmwKindAttr::FSub) => "fsub",
        Some(AtomicRmwKindAttr::FMax) => "fmax",
        Some(AtomicRmwKindAttr::FMin) => "fmin",
    }
}

/// Format a syncscope suffix. pliron stores syncscope as a free-form string
/// (absent = system scope); any value passes through verbatim.
fn fmt_syncscope(scope: Option<Ref<StringAttr>>) -> String {
    match scope.map(|s| String::from((*s).clone())) {
        Some(s) if !s.is_empty() => format!(" syncscope(\"{s}\")"),
        _ => String::new(),
    }
}
