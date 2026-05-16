/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashMap;
use std::fmt::Write;

use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{FPDoubleAttr, FPSingleAttr, IntegerAttr},
        op_interfaces::{CallOpCallable, CallOpInterface},
    },
    context::Ptr,
    op::Op,
    operation::Operation,
    r#type::Typed,
    value::Value,
};

use crate::{
    attributes::{FPHalfAttr, GepIndexAttr, ICmpPredicateAttr},
    ops::{self, atomic::LlvmAtomicOpInterface},
    types::{FuncType, PointerType, VoidType},
};

use super::{ModuleExportState, format_float_literal, format_half_literal, strip_device_prefix};

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
            }
            id if id == ops::UnreachableOp::get_opid_static() => {
                writeln!(output, "  unreachable").unwrap();
            }
            id if id == ops::BrOp::get_opid_static() => {
                let dest = op_ref.successors().next().unwrap();
                let label = block_labels.get(&dest).ok_or("Missing block label")?;
                writeln!(output, "  br label %{label}").unwrap();
            }
            id if id == ops::CondBrOp::get_opid_static() => {
                let mut succs = op_ref.successors();
                let true_dest = succs.next().unwrap();
                let false_dest = succs.next().unwrap();
                let true_label = block_labels.get(&true_dest).ok_or("Missing true label")?;
                let false_label = block_labels.get(&false_dest).ok_or("Missing false label")?;
                let cond = op_ref.get_operand(0);

                write!(output, "  br i1 ").unwrap();
                self.export_value(cond, value_names, output)?;
                writeln!(output, ", label %{true_label}, label %{false_label}").unwrap();
            }

            // --- Memory Ops ---
            id if id == ops::LoadOp::get_opid_static() => {
                let res = op_ref.get_result(0);
                let ptr = op_ref.get_operand(0);
                let res_name = value_names.get(&res).unwrap();
                let ty = res.get_type(self.ctx);

                // Check pointer address space
                let ptr_ty = ptr.get_type(self.ctx);
                let addrspace = ptr_ty
                    .deref(self.ctx)
                    .downcast_ref::<PointerType>()
                    .map_or(0, PointerType::address_space);

                write!(output, "  {res_name} = load ").unwrap();
                self.export_type(ty, output)?;
                if addrspace != 0 {
                    write!(output, ", ptr addrspace({addrspace}) ").unwrap();
                } else {
                    write!(output, ", ptr ").unwrap();
                }
                self.export_value(ptr, value_names, output)?;
                writeln!(output).unwrap();
            }
            id if id == ops::StoreOp::get_opid_static() => {
                let val = op_ref.get_operand(0);
                let ptr = op_ref.get_operand(1);

                // Check pointer address space
                let ptr_ty = ptr.get_type(self.ctx);
                let addrspace = ptr_ty
                    .deref(self.ctx)
                    .downcast_ref::<PointerType>()
                    .map_or(0, PointerType::address_space);

                write!(output, "  store ").unwrap();
                self.export_type(val.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();
                self.export_value(val, value_names, output)?;
                if addrspace != 0 {
                    write!(output, ", ptr addrspace({addrspace}) ").unwrap();
                } else {
                    write!(output, ", ptr ").unwrap();
                }
                self.export_value(ptr, value_names, output)?;
                writeln!(output).unwrap();
            }
            // --- Atomic Ops ---
            id if id == ops::AtomicLoadOp::get_opid_static() => {
                // %val = load atomic i32, ptr [addrspace(N)] %p syncscope("device") acquire
                let atomic_load = op_obj.as_ref().downcast_ref::<ops::AtomicLoadOp>().unwrap();
                let res = op_ref.get_result(0);
                let ptr = op_ref.get_operand(0);
                let res_name = value_names.get(&res).unwrap();
                let ty = res.get_type(self.ctx);
                let syncscope_str = ops::atomic::format_syncscope(&atomic_load.syncscope(self.ctx));
                let ordering_str = ops::atomic::format_ordering(&atomic_load.ordering(self.ctx));

                let ptr_ty = ptr.get_type(self.ctx);
                let addrspace = ptr_ty
                    .deref(self.ctx)
                    .downcast_ref::<PointerType>()
                    .map_or(0, PointerType::address_space);

                write!(output, "  {res_name} = load atomic ").unwrap();
                self.export_type(ty, output)?;
                if addrspace != 0 {
                    write!(output, ", ptr addrspace({addrspace}) ").unwrap();
                } else {
                    write!(output, ", ptr ").unwrap();
                }
                self.export_value(ptr, value_names, output)?;
                let align = self.natural_alignment(ty);
                writeln!(output, "{syncscope_str} {ordering_str}, align {align}").unwrap();
            }
            id if id == ops::AtomicStoreOp::get_opid_static() => {
                // store atomic i32 %v, ptr [addrspace(N)] %p syncscope("device") release
                let atomic_store = op_obj
                    .as_ref()
                    .downcast_ref::<ops::AtomicStoreOp>()
                    .unwrap();
                let val = op_ref.get_operand(0);
                let ptr = op_ref.get_operand(1);
                let syncscope_str =
                    ops::atomic::format_syncscope(&atomic_store.syncscope(self.ctx));
                let ordering_str = ops::atomic::format_ordering(&atomic_store.ordering(self.ctx));

                let ptr_ty = ptr.get_type(self.ctx);
                let addrspace = ptr_ty
                    .deref(self.ctx)
                    .downcast_ref::<PointerType>()
                    .map_or(0, PointerType::address_space);

                write!(output, "  store atomic ").unwrap();
                self.export_type(val.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();
                self.export_value(val, value_names, output)?;
                if addrspace != 0 {
                    write!(output, ", ptr addrspace({addrspace}) ").unwrap();
                } else {
                    write!(output, ", ptr ").unwrap();
                }
                self.export_value(ptr, value_names, output)?;
                let align = self.natural_alignment(val.get_type(self.ctx));
                writeln!(output, "{syncscope_str} {ordering_str}, align {align}").unwrap();
            }
            id if id == ops::AtomicRmwOp::get_opid_static() => {
                // %old = atomicrmw add ptr [addrspace(N)] %p, i32 %v syncscope("device") monotonic
                let atomic_rmw = op_obj.as_ref().downcast_ref::<ops::AtomicRmwOp>().unwrap();
                let res = op_ref.get_result(0);
                let ptr = op_ref.get_operand(0);
                let val = op_ref.get_operand(1);
                let res_name = value_names.get(&res).unwrap();
                let rmw_kind_str = ops::atomic::format_rmw_kind(&atomic_rmw.rmw_kind(self.ctx));
                let syncscope_str = ops::atomic::format_syncscope(&atomic_rmw.syncscope(self.ctx));
                let ordering_str = ops::atomic::format_ordering(&atomic_rmw.ordering(self.ctx));

                let ptr_ty = ptr.get_type(self.ctx);
                let addrspace = ptr_ty
                    .deref(self.ctx)
                    .downcast_ref::<PointerType>()
                    .map_or(0, PointerType::address_space);

                write!(output, "  {res_name} = atomicrmw {rmw_kind_str} ").unwrap();
                if addrspace != 0 {
                    write!(output, "ptr addrspace({addrspace}) ").unwrap();
                } else {
                    write!(output, "ptr ").unwrap();
                }
                self.export_value(ptr, value_names, output)?;
                write!(output, ", ").unwrap();
                self.export_type(val.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();
                self.export_value(val, value_names, output)?;
                writeln!(output, "{syncscope_str} {ordering_str}").unwrap();
            }
            id if id == ops::AtomicCmpxchgOp::get_opid_static() => {
                // %result_struct = cmpxchg ptr %p, i32 %cmp, i32 %new syncscope("device") acq_rel acquire
                // %old = extractvalue { i32, i1 } %result_struct, 0
                // %success = extractvalue { i32, i1 } %result_struct, 1
                let atomic_cmpxchg = op_obj
                    .as_ref()
                    .downcast_ref::<ops::AtomicCmpxchgOp>()
                    .unwrap();
                let res = op_ref.get_result(0);
                let ptr = op_ref.get_operand(0);
                let cmp = op_ref.get_operand(1);
                let new_val = op_ref.get_operand(2);
                let res_name = value_names.get(&res).unwrap();
                let success_ord_str =
                    ops::atomic::format_ordering(&atomic_cmpxchg.success_ordering(self.ctx));
                let failure_ord_str =
                    ops::atomic::format_ordering(&atomic_cmpxchg.failure_ordering(self.ctx));
                let syncscope_str =
                    ops::atomic::format_syncscope(&atomic_cmpxchg.syncscope(self.ctx));
                let val_ty = cmp.get_type(self.ctx);

                let ptr_ty = ptr.get_type(self.ctx);
                let addrspace = ptr_ty
                    .deref(self.ctx)
                    .downcast_ref::<PointerType>()
                    .map_or(0, PointerType::address_space);

                // Emit the cmpxchg instruction -- returns { T, i1 }
                let struct_name = format!("{res_name}.cx");
                write!(output, "  {struct_name} = cmpxchg ").unwrap();
                if addrspace != 0 {
                    write!(output, "ptr addrspace({addrspace}) ").unwrap();
                } else {
                    write!(output, "ptr ").unwrap();
                }
                self.export_value(ptr, value_names, output)?;
                write!(output, ", ").unwrap();
                self.export_type(val_ty, output)?;
                write!(output, " ").unwrap();
                self.export_value(cmp, value_names, output)?;
                write!(output, ", ").unwrap();
                self.export_type(val_ty, output)?;
                write!(output, " ").unwrap();
                self.export_value(new_val, value_names, output)?;
                writeln!(
                    output,
                    "{syncscope_str} {success_ord_str} {failure_ord_str}"
                )
                .unwrap();

                // Extract the old value (element 0 of the { T, i1 } struct)
                write!(output, "  {res_name} = extractvalue {{ ").unwrap();
                self.export_type(val_ty, output)?;
                writeln!(output, ", i1 }} {struct_name}, 0").unwrap();
            }
            id if id == ops::FenceOp::get_opid_static() => {
                // fence syncscope("device") release
                let fence = op_obj.as_ref().downcast_ref::<ops::FenceOp>().unwrap();
                let syncscope_str = ops::atomic::format_syncscope(&fence.syncscope(self.ctx));
                let ordering_str = ops::atomic::format_ordering(&fence.ordering(self.ctx));
                writeln!(output, "  fence{syncscope_str} {ordering_str}").unwrap();
            }

            id if id == ops::AllocaOp::get_opid_static() => {
                // %res = alloca <type>
                let res = op_ref.get_result(0);
                let res_name = value_names.get(&res).unwrap();

                // Get the element type from the attribute
                let alloca_op = op_obj.as_ref().downcast_ref::<ops::AllocaOp>().unwrap();
                let elem_ty = alloca_op
                    .get_attr_alloca_element_type(self.ctx)
                    .expect("Missing alloca_element_type");
                let elem_ty_ptr = elem_ty.get_type(self.ctx);

                write!(output, "  {res_name} = alloca ").unwrap();
                self.export_type(elem_ty_ptr, output)?;
                writeln!(output).unwrap();
            }
            id if id == ops::GetElementPtrOp::get_opid_static() => {
                // %res = getelementptr inbounds TYPE, ptr addrspace(N) %ptr, i32 %idx...
                let res = op_ref.get_result(0);
                let res_name = value_names.get(&res).unwrap();
                let ptr = op_ref.get_operand(0);

                let gep_op = op_obj
                    .as_ref()
                    .downcast_ref::<ops::GetElementPtrOp>()
                    .unwrap();
                let elem_ty = gep_op
                    .get_attr_gep_src_elem_type(self.ctx)
                    .expect("Missing gep_src_elem_type")
                    .get_type(self.ctx); // Ptr<TypeObj>

                write!(output, "  {res_name} = getelementptr inbounds ").unwrap();
                self.export_type(elem_ty, output)?;

                // Check if pointer has a non-default address space
                let ptr_ty = ptr.get_type(self.ctx);
                let addrspace = ptr_ty
                    .deref(self.ctx)
                    .downcast_ref::<PointerType>()
                    .map_or(0, PointerType::address_space);

                if addrspace != 0 {
                    write!(output, ", ptr addrspace({addrspace}) ").unwrap();
                } else {
                    write!(output, ", ptr ").unwrap();
                }
                self.export_value(ptr, value_names, output)?;

                // Indices
                let indices = &gep_op.get_attr_gep_indices(self.ctx).unwrap().0;
                for idx_attr in indices {
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
                let res = op_ref.get_result(0);
                let res_name = value_names.get(&res).unwrap();
                let arg = op_ref.get_operand(0);

                write!(output, "  {res_name} = fneg ").unwrap();
                self.export_type(arg.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();
                self.export_value(arg, value_names, output)?;
                writeln!(output).unwrap();
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
                let res = op_ref.get_result(0);
                let res_name = value_names.get(&res).unwrap();
                let lhs = op_ref.get_operand(0);
                let rhs = op_ref.get_operand(1);

                let icmp = op_obj.as_ref().downcast_ref::<ops::ICmpOp>().unwrap();
                let pred_attr = icmp.predicate(self.ctx);
                let pred_str = match pred_attr {
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

                write!(output, "  {res_name} = icmp {pred_str} ").unwrap();
                self.export_type(lhs.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();
                self.export_value(lhs, value_names, output)?;
                write!(output, ", ").unwrap();
                self.export_value(rhs, value_names, output)?;
                writeln!(output).unwrap();
            }
            id if id == ops::FCmpOp::get_opid_static() => {
                let res = op_ref.get_result(0);
                let res_name = value_names.get(&res).unwrap();
                let lhs = op_ref.get_operand(0);
                let rhs = op_ref.get_operand(1);

                let fcmp = op_obj.as_ref().downcast_ref::<ops::FCmpOp>().unwrap();
                let pred_attr = fcmp.predicate(self.ctx);
                let pred_str = match pred_attr {
                    crate::attributes::FCmpPredicateAttr::False => "false",
                    crate::attributes::FCmpPredicateAttr::OEQ => "oeq",
                    crate::attributes::FCmpPredicateAttr::OGT => "ogt",
                    crate::attributes::FCmpPredicateAttr::OGE => "oge",
                    crate::attributes::FCmpPredicateAttr::OLT => "olt",
                    crate::attributes::FCmpPredicateAttr::OLE => "ole",
                    crate::attributes::FCmpPredicateAttr::ONE => "one",
                    crate::attributes::FCmpPredicateAttr::ORD => "ord",
                    crate::attributes::FCmpPredicateAttr::UEQ => "ueq",
                    crate::attributes::FCmpPredicateAttr::UGT => "ugt",
                    crate::attributes::FCmpPredicateAttr::UGE => "uge",
                    crate::attributes::FCmpPredicateAttr::ULT => "ult",
                    crate::attributes::FCmpPredicateAttr::ULE => "ule",
                    crate::attributes::FCmpPredicateAttr::UNE => "une",
                    crate::attributes::FCmpPredicateAttr::UNO => "uno",
                    crate::attributes::FCmpPredicateAttr::True => "true",
                };

                write!(output, "  {res_name} = fcmp {pred_str} ").unwrap();
                self.export_type(lhs.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();
                self.export_value(lhs, value_names, output)?;
                write!(output, ", ").unwrap();
                self.export_value(rhs, value_names, output)?;
                writeln!(output).unwrap();
            }

            // --- Calls ---
            // LLVM call instruction format:
            //   - Non-void: %result = call <ret_type> @func(<args>)
            //   - Void:     call void @func(<args>)
            //
            // IMPORTANT: Void-returning calls must NOT have a result assignment.
            // Invalid: "%v1 = call void @foo()" - llc will reject this!
            // Valid:   "call void @foo()"
            id if id == ops::CallOp::get_opid_static() => {
                let call = op_obj.as_ref().downcast_ref::<ops::CallOp>().unwrap();
                let callee = call.callee(self.ctx);

                // Extract return type from the call's function type to determine
                // if this is a void call (no result assignment) or value call
                let func_ty = call.callee_type(self.ctx);
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

                // Track if callee is a convergent intrinsic
                let mut is_convergent_call = false;

                // Callee can be direct (@function_name) or indirect (function pointer)
                match callee {
                    CallOpCallable::Direct(identifier) => {
                        let name = identifier.to_string();
                        // LLVM intrinsics use dots in IR; Pliron IR identifiers use underscores.
                        let fixed_name = if name.starts_with("llvm_") {
                            name.replace('_', ".")
                        } else {
                            // Strip cuda_oxide_device_ prefix from call targets to match
                            // the stripped function definitions (clean export names).
                            strip_device_prefix(&name)
                        };
                        is_convergent_call = Self::is_convergent_intrinsic(&fixed_name);
                        write!(output, " @{fixed_name}(").unwrap();
                    }
                    CallOpCallable::Indirect(val) => {
                        write!(output, " ").unwrap();
                        self.export_value(val, value_names, output).unwrap();
                        write!(output, "(").unwrap();
                    }
                }

                // Export call arguments with their types
                for (i, arg) in op_ref.operands().enumerate() {
                    if i > 0 {
                        write!(output, ", ").unwrap();
                    }
                    self.export_type(arg.get_type(self.ctx), output)?;
                    write!(output, " ").unwrap();
                    self.export_value(arg, value_names, output)?;
                }

                // Add convergent attribute reference for sync intrinsics
                if is_convergent_call {
                    writeln!(output, ") #0").unwrap();
                    self.convergent_used = true;
                } else {
                    writeln!(output, ")").unwrap();
                }
            }

            // --- Inline Assembly ---
            id if id == ops::InlineAsmOp::get_opid_static() => {
                let inline_asm = op_obj.as_ref().downcast_ref::<ops::InlineAsmOp>().unwrap();
                let asm_template = inline_asm.asm_template(self.ctx);
                let constraints = inline_asm.constraints(self.ctx);
                let is_convergent = inline_asm.is_convergent(self.ctx);

                // Check if there's a result
                let has_result = op_ref.get_num_results() > 0;

                if has_result {
                    let res = op_ref.get_result(0);
                    let res_name = value_names.get(&res).unwrap();
                    let res_ty = res.get_type(self.ctx);
                    write!(output, "  {res_name} = call ").unwrap();
                    self.export_type(res_ty, output)?;
                } else {
                    write!(output, "  call void").unwrap();
                }

                // Format: call <type> asm sideeffect "<template>", "<constraints>"(<args>...)
                write!(
                    output,
                    " asm sideeffect \"{asm_template}\", \"{constraints}\"("
                )
                .unwrap();

                // Export input operands with types
                for (i, arg) in op_ref.operands().enumerate() {
                    if i > 0 {
                        write!(output, ", ").unwrap();
                    }
                    self.export_type(arg.get_type(self.ctx), output)?;
                    write!(output, " ").unwrap();
                    self.export_value(arg, value_names, output)?;
                }

                // Add convergent attribute reference if needed
                if is_convergent {
                    writeln!(output, ") #0").unwrap();
                    self.convergent_used = true;
                } else {
                    writeln!(output, ")").unwrap();
                }
            }

            // --- Multi-Output Inline Assembly ---
            id if id == ops::InlineAsmMultiOp::get_opid_static() => {
                let inline_asm = op_obj
                    .as_ref()
                    .downcast_ref::<ops::InlineAsmMultiOp>()
                    .unwrap();
                let asm_template = inline_asm.asm_template(self.ctx);
                let constraints = inline_asm.constraints(self.ctx);
                let is_convergent = inline_asm.is_convergent(self.ctx);
                let num_results = op_ref.get_num_results();

                if num_results == 0 {
                    // Void return - simple case
                    write!(output, "  call void").unwrap();
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
                } else {
                    // Multi-output: returns a struct, need extractvalue for each
                    // Step 1: Build the struct type string
                    let mut struct_type = String::from("{");
                    for i in 0..num_results {
                        if i > 0 {
                            struct_type.push_str(", ");
                        }
                        let res_ty = op_ref.get_result(i).get_type(self.ctx);
                        let mut ty_str = String::new();
                        self.export_type(res_ty, &mut ty_str)?;
                        struct_type.push_str(&ty_str);
                    }
                    struct_type.push('}');

                    // Step 2: Generate the asm call returning the struct
                    // We need a temporary name for the struct result
                    // Use the first result's name with "_struct" suffix
                    let first_res = op_ref.get_result(0);
                    let first_res_name = value_names.get(&first_res).unwrap();
                    let struct_result_name = format!("{first_res_name}_struct");

                    write!(output, "  {struct_result_name} = call {struct_type}").unwrap();
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

                    // Step 3: Generate extractvalue for each result
                    for i in 0..num_results {
                        let res = op_ref.get_result(i);
                        let res_name = value_names.get(&res).unwrap();

                        writeln!(
                            output,
                            "  {res_name} = extractvalue {struct_type} {struct_result_name}, {i}"
                        )
                        .unwrap();
                    }
                }
            }

            // --- Casts ---
            id if id == ops::BitcastOp::get_opid_static() => {
                self.export_cast("bitcast", op, value_names, output)?;
            }
            id if id == ops::AddrSpaceCastOp::get_opid_static() => {
                self.export_cast("addrspacecast", op, value_names, output)?;
            }
            id if id == ops::ZExtOp::get_opid_static() => {
                let res = op_ref.get_result(0);
                let res_name = value_names.get(&res).unwrap();
                let val = op_ref.get_operand(0);

                let zext = op_obj.as_ref().downcast_ref::<ops::ZExtOp>().unwrap();
                // Manual attribute access since helper is missing
                let nneg_key: pliron::identifier::Identifier = "llvm_nneg_flag".try_into().unwrap();
                let nneg = zext
                    .get_operation()
                    .deref(self.ctx)
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
                let res = op_ref.get_result(0);
                value_names.insert(res, "undef".to_string());
            }

            // --- Aggregate Ops ---
            id if id == ops::ExtractValueOp::get_opid_static() => {
                let res = op_ref.get_result(0);
                let res_name = value_names.get(&res).unwrap();
                let agg = op_ref.get_operand(0);

                let extract_op = op_obj
                    .as_ref()
                    .downcast_ref::<ops::ExtractValueOp>()
                    .unwrap();
                let indices = extract_op.indices(self.ctx);

                write!(output, "  {res_name} = extractvalue ").unwrap();
                self.export_type(agg.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();
                self.export_value(agg, value_names, output)?;
                for idx in indices {
                    write!(output, ", {idx}").unwrap();
                }
                writeln!(output).unwrap();
            }
            id if id == ops::InsertValueOp::get_opid_static() => {
                let res = op_ref.get_result(0);
                let res_name = value_names.get(&res).unwrap();
                let agg = op_ref.get_operand(0);
                let val = op_ref.get_operand(1);

                let insert_op = op_obj
                    .as_ref()
                    .downcast_ref::<ops::InsertValueOp>()
                    .unwrap();
                let indices = insert_op.indices(self.ctx);

                write!(output, "  {res_name} = insertvalue ").unwrap();
                self.export_type(agg.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();
                self.export_value(agg, value_names, output)?;
                write!(output, ", ").unwrap();
                self.export_type(val.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();
                self.export_value(val, value_names, output)?;

                for idx in indices {
                    write!(output, ", {idx}").unwrap();
                }
                writeln!(output).unwrap();
            }

            // --- Address Operations ---
            id if id == ops::AddressOfOp::get_opid_static() => {
                // AddressOfOp is virtual in textual LLVM IR: every use site
                // prints the global symbol directly. The naming pre-pass in
                // export_func registers the result as `@<global_name>` before
                // any block is emitted, so there is nothing to write here.
                // The debug-only assertion keeps the contract honest if the
                // pre-pass is ever refactored.
                let res = op_ref.get_result(0);
                debug_assert!(
                    value_names
                        .get(&res)
                        .is_some_and(|name| name.starts_with('@')),
                    "AddressOfOp result must be pre-registered as a global \
                     symbol by the naming pre-pass; got {:?}",
                    value_names.get(&res),
                );
            }

            // --- Constants (Virtual) ---
            id if id == ops::ConstantOp::get_opid_static() => {
                let const_op = op_obj.as_ref().downcast_ref::<ops::ConstantOp>().unwrap();
                let val_attr = const_op.get_value(self.ctx);

                let const_str = if let Some(int_attr) = val_attr.downcast_ref::<IntegerAttr>() {
                    // Use APInt's proper decimal string conversion instead of parsing debug format.
                    // The old code parsed debug strings like "APInt { value: 0x4000_0000_0000_u64 }"
                    // by splitting on '_', which broke for values with underscore grouping
                    // (e.g., 1u64 << 46 = 0x4000_0000_0000 would become 0x4000 = 16384).
                    int_attr.value().to_string_unsigned_decimal()
                } else if let Some(fp16_attr) = val_attr.downcast_ref::<FPHalfAttr>() {
                    format_half_literal(fp16_attr.to_bits())
                } else if let Some(fp32_attr) = val_attr.downcast_ref::<FPSingleAttr>() {
                    // Float constant (f32) - format as LLVM float literal
                    let float_val: f32 = fp32_attr.clone().into();
                    format_float_literal(f64::from(float_val))
                } else if let Some(fp64_attr) = val_attr.downcast_ref::<FPDoubleAttr>() {
                    // Double constant (f64) - format as LLVM float literal
                    let float_val: f64 = fp64_attr.clone().into();
                    format_float_literal(float_val)
                } else {
                    "0".to_string() // Fallback
                };

                // Overwrite register name with constant literal
                let res = op_ref.get_result(0);
                value_names.insert(res, const_str);
            }

            // --- Unknown op fallback ---
            _ => {
                writeln!(output, "  ; Unknown op: {op_id}").unwrap();
            }
        }

        Ok(())
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
}
