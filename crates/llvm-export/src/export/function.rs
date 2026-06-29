/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Function and basic block emission.
//!
//! Contains the pre-pass that assigns deterministic anonymous-value names so the
//! textual IR is stable across runs, and the block-argument → PHI-node translation
//! that bridges pliron's basic-block argument convention to LLVM's PHI-node convention.

use rustc_hash::FxHashMap;
use std::fmt::Write;

use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{FPDoubleAttr, FPSingleAttr, IntegerAttr},
        op_interfaces::{BranchOpInterface, SymbolOpInterface},
        type_interfaces::FunctionTypeInterface,
        types::IntegerType,
    },
    context::Ptr,
    linked_list::ContainsLinkedList,
    location::Located,
    op::Op,
    operation::Operation,
    printable::Printable,
    r#type::Typed,
    value::Value,
};

use crate::{
    attributes::FPHalfAttr,
    ops::{self, FuncOp, GlobalOpExt},
    types::{FuncType, PointerType},
};

use super::{
    literals::{format_float_literal, format_half_literal},
    names::{has_device_prefix, strip_device_prefix},
    state::{
        KernelClusterConfig, KernelInfo, KernelLaunchBounds, ModuleExportState, PredecessorMap,
    },
};

impl<'a> ModuleExportState<'a> {
    /// Export a global variable (typically shared memory for GPU kernels).
    pub(super) fn export_global(
        &mut self,
        global: &ops::GlobalOp,
        output: &mut String,
    ) -> Result<(), String> {
        use crate::attributes::LinkageAttr;

        let name = global.get_symbol_name(self.ctx);
        let ty = global.get_type(self.ctx);
        let address_space = global.address_space(self.ctx);

        // NVVM only permits module-scope variables in generic/global,
        // shared, or constant memory. Local memory (address space 5) is for
        // per-thread allocations, not LLVM globals.
        if self.nvvm_ir_dialect.is_some() && !matches!(address_space, 0 | 1 | 3 | 4) {
            return Err(format!(
                "NVVM global `@{name}` uses unsupported address space {address_space}; expected generic (0), global (1), shared (3), or constant (4)"
            ));
        }

        // Check for external linkage (dynamic shared memory)
        let is_external = global
            .get_attr_llvm_global_linkage(self.ctx)
            .map(|linkage| matches!(*linkage, LinkageAttr::ExternalLinkage))
            .unwrap_or(false);

        // Get alignment from attribute, or compute natural alignment from type
        let alignment = global.get_alignment(self.ctx).unwrap_or_else(|| {
            // Compute natural alignment from array element type
            // For [N x T], alignment is size_of(T) (common case: f32 = 4, i64 = 8)
            let ty_ref = ty.deref(self.ctx);
            if let Some(array_ty) = ty_ref.downcast_ref::<crate::types::ArrayType>() {
                let elem_ty = array_ty.elem_type();
                let elem_ref = elem_ty.deref(self.ctx);
                if elem_ref.is::<pliron::builtin::types::IntegerType>() {
                    let int_ty = elem_ref
                        .downcast_ref::<pliron::builtin::types::IntegerType>()
                        .unwrap();
                    u64::from(int_ty.width() / 8)
                } else if elem_ref.is::<pliron::builtin::types::FP32Type>() {
                    4
                } else {
                    8 // Default alignment (FP64Type and unknown types)
                }
            } else {
                8 // Default alignment
            }
        });

        if is_external {
            // External linkage: declaration with size determined elsewhere.
            write!(
                output,
                "@{name} = external addrspace({address_space}) global "
            )
            .unwrap();
            self.export_type(ty, output)?;
            writeln!(output, ", align {alignment}").unwrap();
        } else {
            // Internal linkage: static storage in the global's address space.
            write!(output, "@{name} = addrspace({address_space}) global ").unwrap();
            self.export_type(ty, output)?;
            if let Some(hex) = global.initializer_hex(self.ctx) {
                let bytes = decode_hex_initializer(&hex)?;
                write!(output, " ").unwrap();
                self.export_byte_initializer(ty, &bytes, output)?;
                writeln!(output, ", align {alignment}").unwrap();
            } else {
                // NVVM forbids initialized shared variables. `undef` denotes
                // uninitialized shared storage and is required by both legacy and
                // modern NVVM IR. Keep the ordinary llc/PTX path's historical zero
                // initializer for compatibility.
                let initializer = if self.nvvm_ir_dialect.is_some() && address_space == 3 {
                    "undef"
                } else {
                    "zeroinitializer"
                };
                writeln!(output, " {initializer}, align {alignment}").unwrap();
            }
        }

        Ok(())
    }

    /// Render an evaluated Rust allocation without interpreting its contents.
    ///
    /// The lowering boundary deliberately represents initialized globals as
    /// `[N x i8]`. Escaping every byte keeps all bit patterns intact, including
    /// NaN payloads, and makes Rust padding explicit without teaching this
    /// exporter Rust's layout rules.
    fn export_byte_initializer(
        &self,
        ty: pliron::r#type::TypeHandle,
        bytes: &[u8],
        output: &mut String,
    ) -> Result<(), String> {
        let ty_ref = ty.deref(self.ctx);
        let array_ty = ty_ref
            .downcast_ref::<crate::types::ArrayType>()
            .ok_or_else(|| {
                format!(
                    "explicit global initializer requires `[N x i8]` storage, found `{}`",
                    ty_ref.disp(self.ctx)
                )
            })?;
        let elem_ty = array_ty.elem_type();
        let elem_ref = elem_ty.deref(self.ctx);
        let is_i8 = elem_ref
            .downcast_ref::<IntegerType>()
            .is_some_and(|int_ty| int_ty.width() == 8);
        if !is_i8 || array_ty.size() != bytes.len() as u64 {
            return Err(format!(
                "explicit global initializer has {} bytes but storage type is `{}`; expected `[{} x i8]`",
                bytes.len(),
                ty_ref.disp(self.ctx),
                bytes.len()
            ));
        }

        write!(output, "c\"").unwrap();
        for byte in bytes {
            write!(output, "\\{byte:02X}").unwrap();
        }
        write!(output, "\"").unwrap();
        Ok(())
    }

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
        let attrs = &func.get_operation().deref(self.ctx).attributes;
        let is_kernel = attrs
            .get::<pliron::builtin::attributes::StringAttr>(&kernel_key)
            .is_some();

        // Check for cluster dimension attributes (from #[cluster(x,y,z)])
        // These will be emitted as nvvm.annotations metadata
        if is_kernel {
            let get_int = |key: &str| -> Option<u32> {
                let key: pliron::identifier::Identifier = key.try_into().unwrap();
                attrs
                    .get::<pliron::builtin::attributes::IntegerAttr>(&key)
                    .map(|int_attr| int_attr.value().to_u32())
            };

            if let (Some(dim_x), Some(dim_y), Some(dim_z)) = (
                get_int("cluster_dim_x"),
                get_int("cluster_dim_y"),
                get_int("cluster_dim_z"),
            ) {
                self.cluster_kernels.push(KernelClusterConfig {
                    name: fixed_func_name.clone(),
                    dim_x,
                    dim_y,
                    dim_z,
                });
            }

            // Check for launch bounds attributes (from #[launch_bounds(max, min)])
            // These will be emitted as nvvm.annotations metadata for maxntid and minctasm
            if let Some(max_threads) = get_int("maxntid") {
                let min_blocks = get_int("minctasm");
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

        let ft: pliron::r#type::TypeHandle = func.get_type(self.ctx).into();
        let ft_ref = ft.deref(self.ctx);
        let func_ty = ft_ref
            .downcast_ref::<FuncType>()
            .ok_or("Not a function type")?;

        self.function_types.insert(fixed_func_name.clone(), ft);

        // Track ALL kernels if backend requires annotations for every kernel.
        if is_kernel && self.track_all_kernels {
            self.all_kernels.push(KernelInfo {
                name: fixed_func_name.clone(),
            });
        }

        // Track device function definitions (not declarations) for @llvm.used
        // preservation in standalone device-function compilation.
        let is_declaration = func.get_operation().deref(self.ctx).regions().count() == 0;
        if !is_declaration && !is_kernel && has_device_prefix(&func_name) {
            self.device_functions.push(fixed_func_name.clone());
        }

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

        // Check for alwaysinline attribute (from #[inline(always)]).
        // Emitted as a function attribute keyword between the parameter
        // list and the body open brace.
        let alwaysinline_key: pliron::identifier::Identifier = "alwaysinline".try_into().unwrap();
        let is_alwaysinline = attrs
            .get::<pliron::builtin::attributes::StringAttr>(&alwaysinline_key)
            .is_some();

        if let Some(entry_block) = entry_block_opt {
            let func_loc = func.get_operation().deref(self.ctx).loc();
            let debug_scope = self.debug_subprogram_for_function(&fixed_func_name, &func_loc);
            if let Some(scope_id) = debug_scope {
                self.register_debug_source_scopes_for_function(scope_id, func.get_operation());
            }

            write!(output, "define ").unwrap();
            if is_kernel && self.emit_ptx_kernel_keyword {
                write!(output, "ptx_kernel ").unwrap();
            }
            self.export_type(ret_ty, output)?;
            write!(output, " @{fixed_func_name}(").unwrap();

            let mut value_names = FxHashMap::default();
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
            // Mark every emitted device function `convergent` (attr group #0).
            // GPU code is convergent-by-default, as in Clang/nvcc: a function
            // that (transitively) performs a barrier / shuffle / vote must not
            // have those ops sunk or duplicated into divergent control flow by
            // `opt -O2`. Without this, an inlined `grid::sync()` / warp collective
            // gets its `bar.sync.aligned` pushed into a `tid`-dependent branch
            // and deadlocks. opt's FunctionAttrs strips `convergent` from
            // functions it proves never reach a convergent op.
            // alwaysinline (from #[inline(always)]) and !dbg are independent:
            // either, both, or neither can be present. Emit the inline keyword
            // before the convergent attr group #0, then the debug scope.
            let inline_attr = if is_alwaysinline { "alwaysinline " } else { "" };
            if let Some(scope_id) = debug_scope {
                writeln!(output, ") {inline_attr}#0 !dbg !{scope_id} {{").unwrap();
            } else {
                writeln!(output, ") {inline_attr}#0 {{").unwrap();
            }
            self.convergent_used = true;

            // Assign labels to all blocks
            let mut block_labels = FxHashMap::default();
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

            // PRE-PASS: Assign names to ALL values before exporting.
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
                    let op_obj = Operation::get_op_dyn(op, self.ctx);
                    let op_dyn = op_obj.as_ref();

                    // Skip ops that don't produce named results (UndefOp is handled specially)
                    if op_dyn.downcast_ref::<ops::UndefOp>().is_some() {
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
                    if let Some(const_op) = op_dyn.downcast_ref::<ops::ConstantOp>() {
                        let val_attr = const_op.get_value(self.ctx);

                        let const_str = if let Some(int_attr) =
                            val_attr.downcast_ref::<IntegerAttr>()
                        {
                            int_attr.value().to_string_unsigned_decimal()
                        } else if let Some(fp16_attr) = val_attr.downcast_ref::<FPHalfAttr>() {
                            format_half_literal(crate::fp16_attr_to_bits(fp16_attr))
                        } else if let Some(fp32_attr) = val_attr.downcast_ref::<FPSingleAttr>() {
                            let float_val: f32 = fp32_attr.clone().into();
                            format_float_literal(f64::from(float_val))
                        } else if let Some(fp64_attr) = val_attr.downcast_ref::<FPDoubleAttr>() {
                            let float_val: f64 = fp64_attr.clone().into();
                            format_float_literal(float_val)
                        } else if self.legacy_typed_pointers() {
                            return Err(format!(
                                "legacy LLVM 7 export cannot render constant attribute `{}`",
                                val_attr.disp(self.ctx)
                            ));
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
                    if !self.legacy_typed_pointers()
                        && let Some(address_of) = op_dyn.downcast_ref::<ops::AddressOfOp>()
                    {
                        let symbol_name = address_of.get_global_name(self.ctx).to_string();
                        let res = op_ref.get_result(0);
                        let result_type = res.get_type(self.ctx);
                        let result_ref = result_type.deref(self.ctx);
                        let result_pointer =
                            result_ref.downcast_ref::<PointerType>().ok_or_else(|| {
                                format!(
                                    "addressof result is not a pointer: `{}`",
                                    result_ref.disp(self.ctx)
                                )
                            })?;

                        let exported_name = if let Some(info) =
                            self.global_symbols.get(&symbol_name)
                        {
                            if result_pointer.address_space() != info.address_space {
                                return Err(format!(
                                    "addressof `@{symbol_name}` address-space mismatch: result is {}, global is {}",
                                    result_pointer.address_space(),
                                    info.address_space
                                ));
                            }
                            symbol_name
                        } else {
                            let function_name = if symbol_name.starts_with("llvm_") {
                                symbol_name.replace('_', ".")
                            } else {
                                strip_device_prefix(&symbol_name)
                            };
                            if !self.function_types.contains_key(&function_name) {
                                return Err(format!(
                                    "addressof references unknown symbol `@{symbol_name}`"
                                ));
                            }
                            if result_pointer.address_space() != 0 {
                                return Err(format!(
                                    "function addressof `@{function_name}` must produce a program-address-space (0) pointer, got address space {}",
                                    result_pointer.address_space()
                                ));
                            }
                            function_name
                        };
                        value_names.insert(res, format!("@{exported_name}"));
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
            let mut pred_map: PredecessorMap = FxHashMap::default();
            let validate_edge = |source: Ptr<BasicBlock>, dest: Ptr<BasicBlock>, args: &[Value]| {
                let expected: Vec<_> = dest.deref(self.ctx).arguments().collect();
                let source_label = block_labels
                    .get(&source)
                    .map(String::as_str)
                    .unwrap_or("<unknown>");
                let dest_label = block_labels
                    .get(&dest)
                    .map(String::as_str)
                    .unwrap_or("<unknown>");
                if args.len() != expected.len() {
                    return Err(format!(
                        "predecessor `%{source_label}` supplies {} values to `%{dest_label}`, which expects {} block arguments",
                        args.len(),
                        expected.len()
                    ));
                }
                for (index, (value, argument)) in args.iter().copied().zip(expected).enumerate() {
                    if value.get_type(self.ctx) != argument.get_type(self.ctx) {
                        return Err(format!(
                            "predecessor `%{source_label}` value {index} has type `{}`, but `%{dest_label}` block argument has type `{}`",
                            value.get_type(self.ctx).disp(self.ctx),
                            argument.get_type(self.ctx).disp(self.ctx)
                        ));
                    }
                }
                Ok(())
            };
            for block in func
                .get_operation()
                .deref(self.ctx)
                .get_region(0)
                .deref(self.ctx)
                .iter(self.ctx)
            {
                let block_ref = block.deref(self.ctx);
                if let Some(term) = block_ref.iter(self.ctx).last() {
                    let term_obj = Operation::get_op_dyn(term, self.ctx);
                    let term_dyn = term_obj.as_ref();

                    if let Some(branch) = term_dyn.downcast_ref::<ops::BrOp>() {
                        // BrOp has 1 successor and all operands are passed to it
                        let dest = term.deref(self.ctx).successors().next().unwrap();
                        let args = branch.successor_operands(self.ctx, 0);
                        validate_edge(block, dest, &args)?;
                        pred_map.entry(dest).or_default().push((block, args));
                    } else if let Some(branch) = term_dyn.downcast_ref::<ops::CondBrOp>() {
                        let succs: Vec<_> = term.deref(self.ctx).successors().collect();
                        let true_dest = succs[0];
                        let false_dest = succs[1];
                        let true_args = branch.successor_operands(self.ctx, 0);
                        let false_args = branch.successor_operands(self.ctx, 1);
                        validate_edge(block, true_dest, &true_args)?;
                        validate_edge(block, false_dest, &false_args)?;
                        if true_dest == false_dest {
                            if true_args != false_args {
                                return Err(format!(
                                    "conditional branch `%{}` reaches `%{}` on both edges with different forwarded values; LLVM PHIs cannot distinguish duplicate edges from one predecessor",
                                    block_labels.get(&block).unwrap(),
                                    block_labels.get(&true_dest).unwrap()
                                ));
                            }
                            pred_map
                                .entry(true_dest)
                                .or_default()
                                .push((block, true_args));
                        } else {
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
                    debug_scope,
                    output,
                )?;
            }

            writeln!(output, "}}").unwrap();
        } else {
            // get_num_regions() >= 1 but the first region has no entry block (empty function).
            // Treat it as a declaration.
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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn export_block(
        &mut self,
        block: Ptr<BasicBlock>,
        value_names: &mut FxHashMap<Value, String>,
        next_value_id: &mut usize,
        block_labels: &FxHashMap<Ptr<BasicBlock>, String>,
        pred_map: &PredecessorMap,
        is_entry: bool,
        debug_scope: Option<usize>,
        output: &mut String,
    ) -> Result<(), String> {
        // Always print label to ensure it can be referenced by PHI nodes
        let label = block_labels.get(&block).unwrap();
        writeln!(output, "{label}:").unwrap();

        // Generate PHI nodes for block arguments (except entry block which uses function args)
        let args: Vec<_> = block.deref(self.ctx).arguments().collect();
        if !args.is_empty() && !is_entry {
            let preds = pred_map
                .get(&block)
                .ok_or_else(|| "Block with args has no predecessors".to_string())?;

            for (arg_idx, arg) in args.iter().enumerate() {
                // Use pre-assigned name or generate new one
                let arg_name = if let Some(name) = value_names.get(arg) {
                    name.clone()
                } else {
                    let name = format!("%v{next_value_id}");
                    *next_value_id += 1;
                    value_names.insert(*arg, name.clone());
                    name
                };

                write!(output, "  {arg_name} = phi ").unwrap();
                self.export_type(arg.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();

                for (i, (pred_block, pred_args)) in preds.iter().enumerate() {
                    if i > 0 {
                        write!(output, ", ").unwrap();
                    }

                    if arg_idx < pred_args.len() {
                        let val = pred_args[arg_idx];
                        write!(output, "[ ").unwrap();
                        self.export_value(val, value_names, output)?;
                        let label = block_labels.get(pred_block).unwrap();
                        write!(output, ", %{label} ]").unwrap();
                    } else {
                        return Err(format!(
                            "predecessor `%{}` does not supply block argument {arg_idx} for `%{label}`",
                            block_labels.get(pred_block).unwrap()
                        ));
                    }
                }
                writeln!(output).unwrap();
            }
        }

        for op in block.deref(self.ctx).iter(self.ctx) {
            self.export_op(
                op,
                value_names,
                next_value_id,
                block_labels,
                debug_scope,
                output,
            )?;
        }
        Ok(())
    }
}

fn decode_hex_initializer(hex: &str) -> Result<Vec<u8>, String> {
    if !hex.len().is_multiple_of(2) {
        return Err("global initializer hex string has odd length".to_string());
    }
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    for chunk in hex.as_bytes().chunks_exact(2) {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        bytes.push((hi << 4) | lo);
    }
    Ok(bytes)
}

fn hex_nibble(byte: u8) -> Result<u8, String> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(format!("invalid hex digit {:?}", byte as char)),
    }
}
