/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashMap;
use std::fmt::Write;

use pliron::{basic_block::BasicBlock, context::Ptr, op::Op, operation::Operation, value::Value};

use crate::ops;

use super::ModuleExportState;

mod aggregate;
mod arithmetic;
mod asm;
mod atomic;
mod calls;
mod casts;
mod memory;
mod terminator;
mod virtual_ops;

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
        let op_id = Operation::get_opid(op, self.ctx);
        let op_obj = Operation::get_op_dyn(op, self.ctx);

        // Register result names (skip if already named in pre-pass)
        for res in op_ref.results() {
            value_names.entry(res).or_insert_with(|| {
                let name = format!("%v{next_value_id}");
                *next_value_id += 1;
                name.clone()
            });
        }

        // Match on operation type using guards (op_id is runtime, not enum)
        match op_id {
            // --- Terminators ---
            id if id == ops::ReturnOp::get_opid_static() => {
                self.export_return_op(&op_ref, value_names, output)?;
            }
            id if id == ops::UnreachableOp::get_opid_static() => {
                self.export_unreachable_op(output);
            }
            id if id == ops::BrOp::get_opid_static() => {
                self.export_br_op(&op_ref, block_labels, output)?;
            }
            id if id == ops::CondBrOp::get_opid_static() => {
                self.export_cond_br_op(&op_ref, value_names, block_labels, output)?;
            }

            // --- Memory Ops ---
            id if id == ops::LoadOp::get_opid_static() => {
                self.export_load_op(&op_ref, value_names, output)?;
            }
            id if id == ops::StoreOp::get_opid_static() => {
                self.export_store_op(&op_ref, value_names, output)?;
            }
            // --- Atomic Ops ---
            id if id == ops::AtomicLoadOp::get_opid_static() => {
                let atomic_load = op_obj.as_ref().downcast_ref::<ops::AtomicLoadOp>().unwrap();
                self.export_atomic_load_op(&op_ref, atomic_load, value_names, output)?;
            }
            id if id == ops::AtomicStoreOp::get_opid_static() => {
                let atomic_store = op_obj
                    .as_ref()
                    .downcast_ref::<ops::AtomicStoreOp>()
                    .unwrap();
                self.export_atomic_store_op(&op_ref, atomic_store, value_names, output)?;
            }
            id if id == ops::AtomicRmwOp::get_opid_static() => {
                let atomic_rmw = op_obj.as_ref().downcast_ref::<ops::AtomicRmwOp>().unwrap();
                self.export_atomic_rmw_op(&op_ref, atomic_rmw, value_names, output)?;
            }
            id if id == ops::AtomicCmpxchgOp::get_opid_static() => {
                let atomic_cmpxchg = op_obj
                    .as_ref()
                    .downcast_ref::<ops::AtomicCmpxchgOp>()
                    .unwrap();
                self.export_atomic_cmpxchg_op(&op_ref, atomic_cmpxchg, value_names, output)?;
            }
            id if id == ops::FenceOp::get_opid_static() => {
                let fence = op_obj.as_ref().downcast_ref::<ops::FenceOp>().unwrap();
                self.export_fence_op(fence, output);
            }

            id if id == ops::AllocaOp::get_opid_static() => {
                let alloca_op = op_obj.as_ref().downcast_ref::<ops::AllocaOp>().unwrap();
                self.export_alloca_op(&op_ref, alloca_op, value_names, output)?;
            }
            id if id == ops::GetElementPtrOp::get_opid_static() => {
                let gep_op = op_obj
                    .as_ref()
                    .downcast_ref::<ops::GetElementPtrOp>()
                    .unwrap();
                self.export_gep_op(&op_ref, gep_op, value_names, output)?;
            }

            // --- Arithmetic ---
            id if id == ops::AddOp::get_opid_static() => {
                self.export_binop("add", op, value_names, output)?;
            }
            id if id == ops::SubOp::get_opid_static() => {
                self.export_binop("sub", op, value_names, output)?;
            }
            id if id == ops::MulOp::get_opid_static() => {
                self.export_binop("mul", op, value_names, output)?;
            }
            id if id == ops::FAddOp::get_opid_static() => {
                self.export_binop("fadd", op, value_names, output)?;
            }
            id if id == ops::FSubOp::get_opid_static() => {
                self.export_binop("fsub", op, value_names, output)?;
            }
            id if id == ops::FMulOp::get_opid_static() => {
                self.export_binop("fmul", op, value_names, output)?;
            }
            id if id == ops::FDivOp::get_opid_static() => {
                self.export_binop("fdiv", op, value_names, output)?;
            }
            id if id == ops::FRemOp::get_opid_static() => {
                self.export_binop("frem", op, value_names, output)?;
            }
            id if id == ops::FNegOp::get_opid_static() => {
                self.export_fneg_op(&op_ref, value_names, output)?;
            }
            id if id == ops::SDivOp::get_opid_static() => {
                self.export_binop("sdiv", op, value_names, output)?;
            }
            id if id == ops::UDivOp::get_opid_static() => {
                self.export_binop("udiv", op, value_names, output)?;
            }
            id if id == ops::SRemOp::get_opid_static() => {
                self.export_binop("srem", op, value_names, output)?;
            }
            id if id == ops::URemOp::get_opid_static() => {
                self.export_binop("urem", op, value_names, output)?;
            }
            id if id == ops::XorOp::get_opid_static() => {
                self.export_binop("xor", op, value_names, output)?;
            }
            id if id == ops::ShlOp::get_opid_static() => {
                self.export_binop("shl", op, value_names, output)?;
            }
            id if id == ops::LShrOp::get_opid_static() => {
                self.export_binop("lshr", op, value_names, output)?;
            }
            id if id == ops::AShrOp::get_opid_static() => {
                self.export_binop("ashr", op, value_names, output)?;
            }
            id if id == ops::AndOp::get_opid_static() => {
                self.export_binop("and", op, value_names, output)?;
            }
            id if id == ops::OrOp::get_opid_static() => {
                self.export_binop("or", op, value_names, output)?;
            }
            id if id == ops::ICmpOp::get_opid_static() => {
                let icmp = op_obj.as_ref().downcast_ref::<ops::ICmpOp>().unwrap();
                self.export_icmp_op(&op_ref, icmp, value_names, output)?;
            }
            id if id == ops::FCmpOp::get_opid_static() => {
                let fcmp = op_obj.as_ref().downcast_ref::<ops::FCmpOp>().unwrap();
                self.export_fcmp_op(&op_ref, fcmp, value_names, output)?;
            }

            id if id == ops::CallOp::get_opid_static() => {
                let call = op_obj.as_ref().downcast_ref::<ops::CallOp>().unwrap();
                self.export_call_op(&op_ref, call, value_names, output)?;
            }

            // --- Inline Assembly ---
            id if id == ops::InlineAsmOp::get_opid_static() => {
                let inline_asm = op_obj.as_ref().downcast_ref::<ops::InlineAsmOp>().unwrap();
                self.export_inline_asm_op(&op_ref, inline_asm, value_names, output)?;
            }

            // --- Multi-Output Inline Assembly ---
            id if id == ops::InlineAsmMultiOp::get_opid_static() => {
                let inline_asm = op_obj
                    .as_ref()
                    .downcast_ref::<ops::InlineAsmMultiOp>()
                    .unwrap();
                self.export_inline_asm_multi_op(&op_ref, inline_asm, value_names, output)?;
            }

            // --- Casts ---
            id if id == ops::BitcastOp::get_opid_static() => {
                self.export_cast("bitcast", op, value_names, output)?;
            }
            id if id == ops::AddrSpaceCastOp::get_opid_static() => {
                self.export_cast("addrspacecast", op, value_names, output)?;
            }
            id if id == ops::ZExtOp::get_opid_static() => {
                let zext = op_obj.as_ref().downcast_ref::<ops::ZExtOp>().unwrap();
                self.export_zext_op(&op_ref, zext, value_names, output)?;
            }
            id if id == ops::SExtOp::get_opid_static() => {
                self.export_cast("sext", op, value_names, output)?;
            }
            id if id == ops::TruncOp::get_opid_static() => {
                self.export_cast("trunc", op, value_names, output)?;
            }
            id if id == ops::PtrToIntOp::get_opid_static() => {
                self.export_cast("ptrtoint", op, value_names, output)?;
            }
            id if id == ops::IntToPtrOp::get_opid_static() => {
                self.export_cast("inttoptr", op, value_names, output)?;
            }
            id if id == ops::UIToFPOp::get_opid_static() => {
                self.export_cast("uitofp", op, value_names, output)?;
            }
            id if id == ops::SIToFPOp::get_opid_static() => {
                self.export_cast("sitofp", op, value_names, output)?;
            }
            id if id == ops::FPToUIOp::get_opid_static() => {
                self.export_cast("fptoui", op, value_names, output)?;
            }
            id if id == ops::FPToSIOp::get_opid_static() => {
                self.export_cast("fptosi", op, value_names, output)?;
            }
            id if id == ops::FPExtOp::get_opid_static() => {
                self.export_cast("fpext", op, value_names, output)?;
            }
            id if id == ops::FPTruncOp::get_opid_static() => {
                self.export_cast("fptrunc", op, value_names, output)?;
            }
            id if id == ops::UndefOp::get_opid_static() => {
                self.export_undef_op(&op_ref, value_names);
            }

            // --- Aggregate Ops ---
            id if id == ops::ExtractValueOp::get_opid_static() => {
                let extract_op = op_obj
                    .as_ref()
                    .downcast_ref::<ops::ExtractValueOp>()
                    .unwrap();
                self.export_extract_value_op(&op_ref, extract_op, value_names, output)?;
            }
            id if id == ops::InsertValueOp::get_opid_static() => {
                let insert_op = op_obj
                    .as_ref()
                    .downcast_ref::<ops::InsertValueOp>()
                    .unwrap();
                self.export_insert_value_op(&op_ref, insert_op, value_names, output)?;
            }

            // --- Address Operations ---
            id if id == ops::AddressOfOp::get_opid_static() => {
                self.check_address_of_op(&op_ref, value_names);
            }
            id if id == ops::ConstantOp::get_opid_static() => {
                let const_op = op_obj.as_ref().downcast_ref::<ops::ConstantOp>().unwrap();
                self.export_constant_op(&op_ref, const_op, value_names);
            }

            // --- Unknown op fallback ---
            _ => {
                writeln!(output, "  ; Unknown op: {op_id}").unwrap();
            }
        }

        Ok(())
    }
}
