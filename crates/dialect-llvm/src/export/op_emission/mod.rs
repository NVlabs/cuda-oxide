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
    attributes::{FPHalfAttr, ICmpPredicateAttr},
    ops,
    types::{FuncType, VoidType},
};

use super::{ModuleExportState, format_float_literal, format_half_literal, strip_device_prefix};

mod atomic;
mod memory;
mod terminator;

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
