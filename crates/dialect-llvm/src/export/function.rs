/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashMap;
use std::fmt::Write;

use pliron::{
    builtin::{
        attributes::{FPDoubleAttr, FPSingleAttr, IntegerAttr},
        op_interfaces::SymbolOpInterface,
        type_interfaces::FunctionTypeInterface,
    },
    context::Ptr,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    r#type::{TypeObj, Typed},
};

use crate::{
    attributes::FPHalfAttr,
    ops::{self, FuncOp},
    types::FuncType,
};

use super::{
    KernelClusterConfig, KernelInfo, KernelLaunchBounds, ModuleExportState, PredecessorMap,
    format_float_literal, format_half_literal, has_device_prefix, strip_device_prefix,
};

impl<'a> ModuleExportState<'a> {
    pub(super) fn export_function(
        &mut self,
        func: &FuncOp,
        output: &mut String,
    ) -> Result<(), String> {
        let func_name = func.get_symbol_name(self.ctx);
        // LLVM intrinsics (NVVM and standard, e.g. llvm.fptosi.sat) use dots in IR
        // but Pliron IR identifiers use underscores; convert for export.
        let fixed_func_name = if func_name.starts_with("llvm_") {
            func_name.replace('_', ".")
        } else {
            // Strip cuda_oxide_device_ prefix for clean export names.
            // Internal MIR translation uses prefixed names; we strip at the final
            // export layer so definitions and call targets are renamed consistently.
            strip_device_prefix(&func_name)
        };

        // Check for kernel attribute
        let kernel_key: pliron::identifier::Identifier = "gpu_kernel".try_into().unwrap();
        let attrs = &func.get_operation().deref(self.ctx).attributes.0;
        let is_kernel = attrs.contains_key(&kernel_key);

        // Track ALL kernels if backend requires annotations for every kernel
        if is_kernel && self.track_all_kernels {
            self.all_kernels.push(KernelInfo {
                name: fixed_func_name.clone(),
            });
        }

        // Track device function definitions (not declarations) for @llvm.used preservation
        // in standalone device function compilation. Extern declarations are excluded
        // because they're resolved at link time — only definitions need DCE protection.
        if !is_kernel && has_device_prefix(&func_name) {
            self.device_functions.push(fixed_func_name.clone());
        }

        // Check for cluster dimension attributes (from #[cluster(x,y,z)])
        // These will be emitted as nvvm.annotations metadata
        if is_kernel {
            let x_key: pliron::identifier::Identifier = "cluster_dim_x".try_into().unwrap();
            let y_key: pliron::identifier::Identifier = "cluster_dim_y".try_into().unwrap();
            let z_key: pliron::identifier::Identifier = "cluster_dim_z".try_into().unwrap();

            if let (Some(x_attr), Some(y_attr), Some(z_attr)) =
                (attrs.get(&x_key), attrs.get(&y_key), attrs.get(&z_key))
            {
                // Extract integer values from attributes
                use pliron::attribute::AttrObj;
                let get_int = |attr: &AttrObj| -> Option<u32> {
                    attr.downcast_ref::<pliron::builtin::attributes::IntegerAttr>()
                        .map(|int_attr| int_attr.value().to_u32())
                };

                if let (Some(dim_x), Some(dim_y), Some(dim_z)) =
                    (get_int(x_attr), get_int(y_attr), get_int(z_attr))
                {
                    self.cluster_kernels.push(KernelClusterConfig {
                        name: fixed_func_name.clone(),
                        dim_x,
                        dim_y,
                        dim_z,
                    });
                }
            }

            // Check for launch bounds attributes (from #[launch_bounds(max, min)])
            // These will be emitted as nvvm.annotations metadata for maxntid and minctasm
            let maxntid_key: pliron::identifier::Identifier = "maxntid".try_into().unwrap();
            let minctasm_key: pliron::identifier::Identifier = "minctasm".try_into().unwrap();

            if let Some(max_attr) = attrs.get(&maxntid_key) {
                use pliron::attribute::AttrObj;
                let get_int = |attr: &AttrObj| -> Option<u32> {
                    attr.downcast_ref::<pliron::builtin::attributes::IntegerAttr>()
                        .map(|int_attr| int_attr.value().to_u32())
                };

                if let Some(max_threads) = get_int(max_attr) {
                    let min_blocks = attrs.get(&minctasm_key).and_then(get_int);
                    self.launch_bounds_kernels.push(KernelLaunchBounds {
                        name: fixed_func_name.clone(),
                        max_threads,
                        min_blocks: if min_blocks == Some(0) {
                            None
                        } else {
                            min_blocks
                        },
                    });
                }
            }
        }

        let func_type = func.get_type(self.ctx);
        let ft = Ptr::<TypeObj>::from(func_type);
        let ft_ref = ft.deref(self.ctx);
        let func_ty = ft_ref
            .downcast_ref::<FuncType>()
            .ok_or("Not a function type")?;

        let ret_ty = func_ty.result_type();

        // Check if function has a body
        if func.get_operation().deref(self.ctx).regions().count() == 0 {
            // Function Declaration
            write!(output, "declare ").unwrap();
            self.export_type(ret_ty, output)?;
            write!(output, " @{fixed_func_name}(").unwrap();

            let args = func_ty.arg_types();
            for (i, arg_ty) in args.iter().enumerate() {
                if i > 0 {
                    write!(output, ", ").unwrap();
                }
                self.export_type(*arg_ty, output)?;
            }
            write!(output, ")").unwrap();

            // Check if this is a known convergent intrinsic
            let is_convergent_intrinsic = Self::is_convergent_intrinsic(&fixed_func_name);
            if is_convergent_intrinsic {
                writeln!(output, " #0").unwrap();
                self.convergent_used = true;
            } else {
                writeln!(output).unwrap();
            }
            // No extra newline after declarations to keep them grouped
            return Ok(());
        }

        // Function Body
        let entry_block_opt = func
            .get_operation()
            .deref(self.ctx)
            .get_region(0)
            .deref(self.ctx)
            .iter(self.ctx)
            .next();

        if let Some(entry_block) = entry_block_opt {
            write!(output, "define ").unwrap();
            if is_kernel && self.emit_ptx_kernel_keyword {
                write!(output, "ptx_kernel ").unwrap();
            }
            self.export_type(ret_ty, output)?;
            write!(output, " @{fixed_func_name}(").unwrap();

            let mut value_names = HashMap::new();
            let mut next_value_id = 0;

            let block = entry_block.deref(self.ctx);
            let args = block.arguments();
            // Parameters are emitted bare: `<type> %vN` with no LLVM parameter
            // attributes (no `noalias`, `nocapture`, `dereferenceable`, etc.).
            // This is deliberate and load-bearing for `DisjointSlice`.
            //
            // `DisjointSlice::from_raw_parts` is `unsafe fn` whose contract
            // says callers must not construct two slices over the same range.
            // Violating that contract creates two `&mut T` to the same byte —
            // which is simply UB. Today, because we don't tag pointer
            // parameters with `noalias`, LLVM treats them conservatively and
            // the violation doesn't *miscompile*; it just runs as written.
            //
            // If a future change here adds `noalias` (e.g. for a perf win on
            // read-only `&[T]` inputs), that property goes away and any code
            // that double-constructed a `DisjointSlice` starts seeing folded
            // writes / reordered reads on PTX. Don't add parameter attributes
            // here without re-auditing the `from_raw_parts` callers.
            for (i, arg) in args.enumerate() {
                if i > 0 {
                    write!(output, ", ").unwrap();
                }
                let arg_ty = arg.get_type(self.ctx);
                self.export_type(arg_ty, output)?;
                let name = format!("%v{next_value_id}");
                value_names.insert(arg, name.clone());
                write!(output, " {name}").unwrap();
                next_value_id += 1;
            }
            writeln!(output, ") {{").unwrap();

            // Assign labels to all blocks
            let mut block_labels = HashMap::new();
            let mut next_label_id = 0;
            for (i, block_node) in func
                .get_operation()
                .deref(self.ctx)
                .get_region(0)
                .deref(self.ctx)
                .iter(self.ctx)
                .enumerate()
            {
                if i == 0 {
                    // Entry block usually doesn't need label in LLVM if it's first
                    block_labels.insert(block_node, "entry".to_string());
                } else {
                    let label = format!("bb{next_label_id}");
                    next_label_id += 1;
                    block_labels.insert(block_node, label);
                }
            }

            // PRE-PASS: Assign names to ALL values before exporting
            // This is needed because PHI nodes may reference values from blocks that
            // come later in the block list (e.g., back-edges in loops).
            for block_node in func
                .get_operation()
                .deref(self.ctx)
                .get_region(0)
                .deref(self.ctx)
                .iter(self.ctx)
            {
                // Name block arguments (skip entry block which was already done)
                if block_node != entry_block {
                    for arg in block_node.deref(self.ctx).arguments() {
                        let name = format!("%v{next_value_id}");
                        next_value_id += 1;
                        value_names.insert(arg, name);
                    }
                }

                // Name operation results
                for op in block_node.deref(self.ctx).iter(self.ctx) {
                    let op_ref = op.deref(self.ctx);
                    let op_id = Operation::get_opid(op, self.ctx);

                    // Skip ops that don't produce named results (UndefOp is handled specially)
                    if op_id == ops::UndefOp::get_opid_static() {
                        // UndefOp result will be named "undef"
                        continue;
                    }

                    // CRITICAL: ConstantOp MUST be registered in pre-pass, not during export!
                    // PHI nodes may reference constants from blocks that appear later in the
                    // iteration order. If we delay constant naming until export, the PHI
                    // export will fail to find the constant in value_names and emit "undef".
                    //
                    // Example: bb6 has PHI receiving constant 0 from bb14, but bb6 is
                    // exported before bb14. Without pre-pass registration, the constant's
                    // Value is not in value_names when bb6's PHI is emitted.
                    if op_id == ops::ConstantOp::get_opid_static() {
                        let const_op = Operation::get_op::<ops::ConstantOp>(op, self.ctx).unwrap();
                        let val_attr = const_op.get_value(self.ctx);

                        let const_str = if let Some(int_attr) =
                            val_attr.downcast_ref::<IntegerAttr>()
                        {
                            int_attr.value().to_string_unsigned_decimal()
                        } else if let Some(fp16_attr) = val_attr.downcast_ref::<FPHalfAttr>() {
                            format_half_literal(fp16_attr.to_bits())
                        } else if let Some(fp32_attr) = val_attr.downcast_ref::<FPSingleAttr>() {
                            let float_val: f32 = fp32_attr.clone().into();
                            format_float_literal(f64::from(float_val))
                        } else if let Some(fp64_attr) = val_attr.downcast_ref::<FPDoubleAttr>() {
                            let float_val: f64 = fp64_attr.clone().into();
                            format_float_literal(float_val)
                        } else {
                            "0".to_string() // Fallback
                        };

                        let res = op_ref.get_result(0);
                        value_names.insert(res, const_str);
                        continue;
                    }

                    // AddressOfOp is also virtual in textual LLVM IR: uses
                    // must print the global symbol directly. Pre-register
                    // the result as `@<global_name>` here so CFG order
                    // cannot expose a stale temporary name when a
                    // later-printed block defines the address used by an
                    // earlier-printed block. The op-emit arm in `export_op`
                    // for AddressOfOp asserts this invariant.
                    if op_id == ops::AddressOfOp::get_opid_static() {
                        let address_of =
                            Operation::get_op::<ops::AddressOfOp>(op, self.ctx).unwrap();
                        let global_name = address_of.get_global_name(self.ctx);
                        let res = op_ref.get_result(0);
                        value_names.insert(res, format!("@{global_name}"));
                        continue;
                    }

                    for res in op_ref.results() {
                        let name = format!("%v{next_value_id}");
                        next_value_id += 1;
                        value_names.insert(res, name);
                    }
                }
            }

            // Build predecessor map for PHI generation
            let mut pred_map: PredecessorMap = HashMap::new();
            for block in func
                .get_operation()
                .deref(self.ctx)
                .get_region(0)
                .deref(self.ctx)
                .iter(self.ctx)
            {
                let block_ref = block.deref(self.ctx);
                if let Some(term) = block_ref.iter(self.ctx).last() {
                    let op_id = Operation::get_opid(term, self.ctx);

                    if op_id == ops::BrOp::get_opid_static() {
                        // BrOp has 1 successor and all operands are passed to it
                        let dest = term.deref(self.ctx).successors().next().unwrap();
                        let args: Vec<_> = term.deref(self.ctx).operands().collect();
                        pred_map.entry(dest).or_default().push((block, args));
                    } else if op_id == ops::CondBrOp::get_opid_static() {
                        let succs: Vec<_> = term.deref(self.ctx).successors().collect();
                        let true_dest = succs[0];
                        let false_dest = succs[1];

                        // Calculate split point for operands
                        // [cond, true_args..., false_args...]
                        let num_true = true_dest.deref(self.ctx).arguments().count();
                        let num_false = false_dest.deref(self.ctx).arguments().count();

                        let all_ops: Vec<_> = term.deref(self.ctx).operands().collect();
                        if all_ops.len() >= 1 + num_true + num_false {
                            let true_args = all_ops[1..=num_true].to_vec();
                            let false_args =
                                all_ops[1 + num_true..1 + num_true + num_false].to_vec();

                            pred_map
                                .entry(true_dest)
                                .or_default()
                                .push((block, true_args));
                            pred_map
                                .entry(false_dest)
                                .or_default()
                                .push((block, false_args));
                        }
                    }
                }
            }

            // Export blocks
            for (i, block_node) in func
                .get_operation()
                .deref(self.ctx)
                .get_region(0)
                .deref(self.ctx)
                .iter(self.ctx)
                .enumerate()
            {
                self.export_block(
                    block_node,
                    &mut value_names,
                    &mut next_value_id,
                    &block_labels,
                    &pred_map,
                    i == 0,
                    output,
                )?;
            }

            writeln!(output, "}}").unwrap();
        } else {
            // This block handled the declaration case, but we now check get_num_regions() above.
            // If we are here, get_num_regions() >= 1 but entry_block_opt is None (empty region).
            // This is also a declaration in some contexts, or an empty function.
            // Let's treat it as a declaration if it's empty.

            write!(output, "declare ").unwrap();
            self.export_type(ret_ty, output)?;
            write!(output, " @{fixed_func_name}(").unwrap();

            let args = func_ty.arg_types();
            for (i, arg_ty) in args.iter().enumerate() {
                if i > 0 {
                    write!(output, ", ").unwrap();
                }
                self.export_type(*arg_ty, output)?;
            }
            writeln!(output, ")").unwrap();
        }

        writeln!(output).unwrap();
        Ok(())
    }
}
