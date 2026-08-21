/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `dialect-mir` → LLVM dialect function lowering via `inline_region`.
//!
//! This module implements [`convert_func`] — the entry point for lowering
//! `MirFuncOp` → `llvm.func` using pliron's `DialectConversion` framework.
//!
//! # Conversion Strategy
//!
//! 1. Creates an LLVM function with a converted (flattened) type signature
//! 2. Propagates GPU kernel attributes (`gpu_kernel`, cluster dims, launch bounds)
//! 3. Pre-scans for maximum dynamic shared memory alignment
//! 4. Uses `inline_region` to move MIR blocks into the LLVM function
//! 5. Reconstructs aggregate types (slices, structs) in an entry prologue
//! 6. Branches to the original MIR entry block with reconstructed values
//!
//! # Entry Block Prologue
//!
//! ```text
//! LLVM entry block (flattened args: ptr, len, field0, field1, ...):
//!   %undef_slice = llvm.mlir.undef : {ptr, i64}
//!   %with_ptr    = llvm.insertvalue %ptr into %undef_slice[0]
//!   %slice       = llvm.insertvalue %len into %with_ptr[1]
//!   llvm.br ^mir_entry(%slice, %field0, %field1, ...)
//! ```

use crate::context::{DynamicSmemAlignmentMap, SharedGlobalsMap};
use crate::convert::types::{
    StructLayoutInfo, build_struct_slot_map, convert_function_type, convert_type, is_kernel_func,
    is_zero_sized_type, llvm_type_size_align, mir_type_abi_align, transparent_scalar_field,
};

use dialect_mir::ops::MirFuncOp;
use dialect_mir::types::{MirDisjointSliceType, MirSliceType, MirStructType};
use llvm_export::ops as llvm;
use pliron::{
    basic_block::BasicBlock,
    builtin::op_interfaces::SymbolOpInterface,
    context::{Context, Ptr},
    irbuild::{
        dialect_conversion::{DialectConversionRewriter, OperandsInfo},
        inserter::{BlockInsertionPoint, Inserter, OpInsertionPoint},
        rewriter::Rewriter,
    },
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::TypeHandle,
    value::Value,
};
use rustc_hash::{FxHashMap, FxHashSet};

const DYNAMIC_SHARED_ALIGNMENT_ATTR: &str = "dynamic_shared_alignment";
// Pliron op-attribute key, not a reserved link symbol; the name stays
// outside the reserved-oxide-symbols kernel prefix family on purpose.
const KERNEL_PARAM_ABI_ALIGN_ATTR_PREFIX: &str = "cuda_oxide_param_abi_align_";
const RETURN_ABI_ALIGN_ATTR: &str = "cuda_oxide_return_abi_align";

// ============================================================================
// Dynamic shared-memory contract propagation
// ============================================================================

/// Propagate dynamic shared-memory alignment markers through the local MIR
/// call graph before any function is lowered.
///
/// The marker normally belongs to a kernel entry. Attribute expansion may put
/// it in a generic helper when `#[launch_contract]` appears above `#[kernel]`,
/// so every marked local function is a propagation root. A dynamic
/// shared-memory access can also live in a deeper ordinary helper. Shared
/// helpers receive the maximum requirement from every marked root that can
/// reach them.
pub(crate) fn propagate_kernel_dynamic_shared_alignments(
    ctx: &mut Context,
    module_op: Ptr<Operation>,
) {
    let mut functions = FxHashMap::default();
    for region in module_op.deref(ctx).regions() {
        for block in region.deref(ctx).iter(ctx) {
            for op in block.deref(ctx).iter(ctx) {
                if let Some(function) = MirFuncOp::wrap(ctx, op) {
                    functions.insert(function.get_symbol_name(ctx).to_string(), op);
                }
            }
        }
    }

    let call_graph: FxHashMap<String, Vec<String>> = functions
        .iter()
        .map(|(name, op)| (name.clone(), collect_mir_callees(ctx, *op)))
        .collect();
    let alignment_roots: Vec<(String, u64)> = functions
        .iter()
        .filter_map(|(name, op)| {
            dynamic_shared_alignment_attr(ctx, *op).map(|value| (name.clone(), value))
        })
        .collect();

    for (name, propagated_alignment) in
        propagate_alignments_through_call_graph(&call_graph, &alignment_roots)
    {
        let Some(function) = functions.get(&name).copied() else {
            continue;
        };
        let alignment = dynamic_shared_alignment_attr(ctx, function)
            .map_or(propagated_alignment, |local| {
                local.max(propagated_alignment)
            });
        set_dynamic_shared_alignment_attr(ctx, function, alignment);
    }
}

fn collect_mir_callees(ctx: &Context, root: Ptr<Operation>) -> Vec<String> {
    fn visit(ctx: &Context, op: Ptr<Operation>, callees: &mut Vec<String>) {
        if let Some(call) = Operation::get_op::<dialect_mir::ops::MirCallOp>(op, ctx)
            && let Some(callee) = call.get_attr_callee(ctx)
        {
            callees.push(String::from((*callee).clone()));
        }

        let children: Vec<_> = op
            .deref(ctx)
            .regions()
            .flat_map(|region| region.deref(ctx).iter(ctx))
            .flat_map(|block| block.deref(ctx).iter(ctx))
            .collect();
        for child in children {
            visit(ctx, child, callees);
        }
    }

    let mut callees = Vec::new();
    visit(ctx, root, &mut callees);
    callees.sort_unstable();
    callees.dedup();
    callees
}

fn propagate_alignments_through_call_graph(
    call_graph: &FxHashMap<String, Vec<String>>,
    alignment_roots: &[(String, u64)],
) -> FxHashMap<String, u64> {
    let mut propagated = FxHashMap::default();

    for (root, alignment) in alignment_roots {
        let mut worklist = vec![root.clone()];
        let mut visited = FxHashSet::default();

        while let Some(function) = worklist.pop() {
            if !visited.insert(function.clone()) {
                continue;
            }
            if !call_graph.contains_key(&function) {
                continue;
            }

            propagated
                .entry(function.clone())
                .and_modify(|current: &mut u64| *current = (*current).max(*alignment))
                .or_insert(*alignment);
            if let Some(callees) = call_graph.get(&function) {
                worklist.extend(callees.iter().cloned());
            }
        }
    }

    propagated
}

fn dynamic_shared_alignment_attr(ctx: &Context, op: Ptr<Operation>) -> Option<u64> {
    let key: pliron::identifier::Identifier = DYNAMIC_SHARED_ALIGNMENT_ATTR
        .try_into()
        .expect("static identifier");
    op.deref(ctx)
        .attributes
        .get::<pliron::builtin::attributes::IntegerAttr>(&key)
        .map(|attribute| attribute.value().to_u64())
}

fn set_dynamic_shared_alignment_attr(ctx: &mut Context, op: Ptr<Operation>, alignment: u64) {
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::builtin::types::Signedness;
    use pliron::utils::apint::APInt;
    use std::num::NonZero;

    let key: pliron::identifier::Identifier = DYNAMIC_SHARED_ALIGNMENT_ATTR
        .try_into()
        .expect("static identifier");
    let u64_ty = pliron::builtin::types::IntegerType::get(ctx, 64, Signedness::Unsigned);
    let value = APInt::from_u64(alignment, NonZero::new(64).unwrap());
    op.deref_mut(ctx)
        .attributes
        .set(key, IntegerAttr::new(u64_ty, value));
}

// ============================================================================
// Function Conversion
// ============================================================================

/// Convert a `MirFuncOp` to `llvm.func` using pliron's `inline_region`.
///
/// Called from `crate::MirToLlvmConversionDriver::rewrite` when the
/// framework encounters a `MirFuncOp`. Creates a new LLVM function,
/// propagates kernel attributes, moves the MIR body via `inline_region`,
/// and builds an entry prologue to reconstruct aggregate arguments.
#[allow(clippy::too_many_arguments)]
pub fn convert_func(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    _shared_globals: &mut SharedGlobalsMap,
    dynamic_smem_alignments: &mut DynamicSmemAlignmentMap,
) -> Result<()> {
    let mir_func = MirFuncOp::wrap(ctx, op).expect("expected MirFuncOp");
    let name = mir_func.get_symbol_name(ctx);
    let func_name_str = name.to_string();

    let kernel_key: pliron::identifier::Identifier = "gpu_kernel".try_into().unwrap();
    let is_kernel = is_kernel_func(ctx, op);

    let func_type = mir_func.get_type(ctx);

    // Kernel parameters are host data: the host writes them (by value at
    // launch, or into DeviceBuffer memory behind a pointer or slice) and
    // the kernel reads the same bytes, so both sides must agree on what
    // every byte means. Importer-produced Direct, Niche, Single, and Empty
    // enum layouts are byte-identical to rustc. A legacy/hand-built Unknown
    // layout has no such proof, so reject it here with a focused ABI error;
    // physical lowering independently rejects Unknown layouts everywhere.
    if is_kernel {
        let mir_arg_types = {
            use pliron::builtin::type_interfaces::FunctionTypeInterface;
            let ft_ref = func_type.deref(ctx);
            ft_ref.arg_types().to_vec()
        };
        for (i, arg_ty) in mir_arg_types.iter().enumerate() {
            let mut visited = Vec::new();
            if let Some(enum_name) =
                crate::convert::types::find_unmodeled_enum_in_abi(ctx, *arg_ty, &mut visited)
                    .map_err(anyhow_to_pliron)?
            {
                return pliron::input_err_noloc!(
                    "kernel `{}` parameter {} contains enum `{}` with unknown physical \
                     rustc layout at the kernel boundary; refusing to guess its bytes. \
                     This indicates legacy or malformed dialect-mir input rather than an \
                     unsupported Rust niche layout.",
                    func_name_str,
                    i,
                    enum_name
                );
            }
        }
    }
    let llvm_func_type =
        convert_function_type(ctx, func_type, is_kernel).map_err(anyhow_to_pliron)?;
    let kernel_param_alignments = if is_kernel {
        kernel_param_abi_alignments(ctx, func_type, llvm_func_type).map_err(anyhow_to_pliron)?
    } else {
        Vec::new()
    };
    let return_abi_alignment =
        function_return_abi_alignment(ctx, func_type, llvm_func_type).map_err(anyhow_to_pliron)?;

    let llvm_func = llvm::FuncOp::new(ctx, name, llvm_func_type);
    llvm::copy_debug_function_name(ctx, op, llvm_func.get_operation());
    llvm::copy_debug_source_scope_map(ctx, op, llvm_func.get_operation());

    if is_kernel {
        propagate_kernel_attrs(ctx, op, &llvm_func, &kernel_key);
        propagate_kernel_param_abi_alignments(ctx, &llvm_func, &kernel_param_alignments);
    }
    propagate_return_abi_alignment(ctx, &llvm_func, return_abi_alignment);

    propagate_alwaysinline_attr(ctx, op, &llvm_func);

    let llvm_entry = llvm_func.get_or_create_entry_block(ctx);

    let mir_region = op.deref(ctx).get_region(0);
    let mir_entry = mir_region.deref(ctx).get_head();

    if let Some(mir_entry) = mir_entry {
        // Pre-scan MIR blocks for max dynamic shared memory alignment.
        // Must happen BEFORE inline_region empties the MIR region.
        let mir_blocks: Vec<_> = mir_region.deref(ctx).iter(ctx).collect();
        let body_max_align = compute_max_dynamic_smem_alignment(ctx, &mir_blocks);
        let contract_min_align = dynamic_shared_alignment_attr(ctx, op);
        let max_align = match (body_max_align, contract_min_align) {
            (Some(body), Some(contract)) => Some(body.max(contract)),
            (body, contract) => body.or(contract),
        };

        if let Some(align) = max_align {
            let symbol_name: pliron::identifier::Identifier =
                format!("__dynamic_smem_{}", func_name_str)
                    .as_str()
                    .try_into()
                    .expect("Invalid function name for symbol");
            dynamic_smem_alignments.insert(func_name_str, (symbol_name, align));
        }

        // Extract MIR arg types for entry prologue reconstruction
        let mir_arg_types = {
            use pliron::builtin::type_interfaces::FunctionTypeInterface;
            let ft_ref = func_type.deref(ctx);
            ft_ref.arg_types().to_vec()
        };

        let reconstructed_args = build_entry_prologue(ctx, &mir_arg_types, llvm_entry, is_kernel)
            .map_err(anyhow_to_pliron)?;

        rewriter.inline_region(ctx, mir_region, BlockInsertionPoint::AfterBlock(llvm_entry));

        // Insert BrOp through the rewriter so the framework sees it as a
        // terminator and converts the MIR entry block's argument types.
        let saved_ip = rewriter.get_insertion_point();
        rewriter.set_insertion_point(OpInsertionPoint::AtBlockEnd(llvm_entry));
        let br = llvm::BrOp::new(ctx, mir_entry, reconstructed_args);
        rewriter.insert_operation(ctx, br.get_operation());
        rewriter.set_insertion_point(saved_ip);
    }

    rewriter.insert_operation(ctx, llvm_func.get_operation());
    rewriter.replace_operation(ctx, op, llvm_func.get_operation());
    Ok(())
}

// ============================================================================
// Kernel Attribute Propagation
// ============================================================================

/// Propagate GPU kernel attributes from MIR func to LLVM func.
fn propagate_kernel_attrs(
    ctx: &mut Context,
    mir_op: Ptr<Operation>,
    llvm_func: &llvm::FuncOp,
    kernel_key: &pliron::identifier::Identifier,
) {
    llvm_func.get_operation().deref_mut(ctx).attributes.set(
        kernel_key.clone(),
        pliron::builtin::attributes::StringAttr::new("true".to_string()),
    );

    // Extract MIR attrs first to avoid borrow overlap with deref_mut below
    let attrs_to_copy: Vec<_> = {
        let mir_attrs = &mir_op.deref(ctx).attributes;
        [
            "cluster_dim_x",
            "cluster_dim_y",
            "cluster_dim_z",
            "maxntid",
            "minctasm",
            "reqntid_x",
            "reqntid_y",
            "reqntid_z",
        ]
        .iter()
        .filter_map(|key_str| {
            let key: pliron::identifier::Identifier = (*key_str).try_into().unwrap();
            mir_attrs
                .get::<pliron::builtin::attributes::IntegerAttr>(&key)
                .map(|attr| (key, attr.clone()))
        })
        .collect()
    };

    for (key, attr) in attrs_to_copy {
        llvm_func
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(key, attr);
    }
}

/// Compute language ABI alignments that LLVM's structural parameter types lose.
///
/// Kernel aggregates are passed directly in `.param` space. A packed LLVM
/// struct has natural alignment one even when Rust's `repr(packed(N))` ABI
/// requires a larger power-of-two alignment. Preserve only the cases where
/// rustc's ABI alignment is stricter than LLVM's natural alignment; the LLVM
/// exporter renders these markers as NVVM `!nvvm.annotations` `"align"`
/// properties, which preserve the contract in both modern NVPTX and libNVVM.
fn kernel_param_abi_alignments(
    ctx: &mut Context,
    mir_func_type: pliron::r#type::TypedHandle<pliron::builtin::types::FunctionType>,
    llvm_func_type: pliron::r#type::TypedHandle<llvm_export::types::FuncType>,
) -> std::result::Result<Vec<(usize, u64)>, anyhow::Error> {
    use pliron::builtin::type_interfaces::FunctionTypeInterface;

    let mir_args = {
        let func_ref = mir_func_type.deref(ctx);
        func_ref.arg_types().to_vec()
    };
    let llvm_args = {
        let func_ref = llvm_func_type.deref(ctx);
        func_ref.arg_types().to_vec()
    };

    let mut result = Vec::new();
    let mut llvm_arg_index = 0usize;

    for mir_ty in mir_args {
        match classify_argument_type(ctx, mir_ty, true)? {
            ReconstructKind::Slice { space_fields } => {
                llvm_arg_index = llvm_arg_index
                    .checked_add(2 + space_fields)
                    .ok_or_else(|| anyhow::anyhow!("kernel parameter index overflow"))?;
            }
            ReconstructKind::TransparentScalar | ReconstructKind::None => {
                let llvm_ty = *llvm_args.get(llvm_arg_index).ok_or_else(|| {
                    anyhow::anyhow!(
                        "kernel parameter alignment mapping ran past LLVM argument {}",
                        llvm_arg_index
                    )
                })?;

                if let (Some(rust_align), Some((_, llvm_align))) = (
                    mir_type_abi_align(ctx, mir_ty),
                    llvm_type_size_align(ctx, llvm_ty),
                ) && rust_align > llvm_align
                {
                    if !rust_align.is_power_of_two() {
                        return Err(anyhow::anyhow!(
                            "kernel parameter {} has non-power-of-two Rust ABI alignment {}",
                            llvm_arg_index,
                            rust_align
                        ));
                    }
                    result.push((llvm_arg_index, rust_align));
                }

                llvm_arg_index += 1;
            }
            ReconstructKind::Zst => {}
            ReconstructKind::Struct(_) => {
                return Err(anyhow::anyhow!(
                    "kernel parameter unexpectedly used the internal flattened struct ABI"
                ));
            }
        }
    }

    if llvm_arg_index != llvm_args.len() {
        return Err(anyhow::anyhow!(
            "kernel parameter alignment mapping consumed {} LLVM arguments, expected {}",
            llvm_arg_index,
            llvm_args.len()
        ));
    }

    Ok(result)
}

/// Compute a non-natural ABI alignment for a direct aggregate return value.
///
/// NVVM encodes return alignment with argument position zero in the same
/// `"align"` global property used for direct by-value aggregate parameters.
/// Internal device functions can return packed aggregates even though their
/// struct parameters use the private flattened ABI, so return alignment is
/// tracked for every lowered function rather than kernels alone.
fn function_return_abi_alignment(
    ctx: &mut Context,
    mir_func_type: pliron::r#type::TypedHandle<pliron::builtin::types::FunctionType>,
    llvm_func_type: pliron::r#type::TypedHandle<llvm_export::types::FuncType>,
) -> std::result::Result<Option<u64>, anyhow::Error> {
    use pliron::builtin::type_interfaces::FunctionTypeInterface;

    let mir_result = {
        let func_ref = mir_func_type.deref(ctx);
        func_ref.res_types().first().copied()
    };
    let Some(mir_result) = mir_result else {
        return Ok(None);
    };

    let llvm_result = {
        let func_ref = llvm_func_type.deref(ctx);
        func_ref.result_type()
    };
    let Some(rust_align) = mir_type_abi_align(ctx, mir_result) else {
        return Ok(None);
    };
    let Some((_, llvm_align)) = llvm_type_size_align(ctx, llvm_result) else {
        return Ok(None);
    };
    if rust_align <= llvm_align {
        return Ok(None);
    }
    if !rust_align.is_power_of_two() {
        return Err(anyhow::anyhow!(
            "function return has non-power-of-two Rust ABI alignment {}",
            rust_align
        ));
    }
    Ok(Some(rust_align))
}

fn propagate_kernel_param_abi_alignments(
    ctx: &mut Context,
    llvm_func: &llvm::FuncOp,
    alignments: &[(usize, u64)],
) {
    use pliron::builtin::attributes::IntegerAttr;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::utils::apint::APInt;
    use std::num::NonZero;

    let u64_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
    let width = NonZero::new(64).expect("64 is non-zero");

    for &(index, alignment) in alignments {
        let key: pliron::identifier::Identifier =
            format!("{KERNEL_PARAM_ABI_ALIGN_ATTR_PREFIX}{index}")
                .as_str()
                .try_into()
                .expect("kernel parameter alignment attribute name is valid");
        let value = APInt::from_u64(alignment, width);
        llvm_func
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(key, IntegerAttr::new(u64_ty, value));
    }
}

fn propagate_return_abi_alignment(
    ctx: &mut Context,
    llvm_func: &llvm::FuncOp,
    alignment: Option<u64>,
) {
    let Some(alignment) = alignment else {
        return;
    };

    use pliron::builtin::attributes::IntegerAttr;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::utils::apint::APInt;
    use std::num::NonZero;

    let key: pliron::identifier::Identifier = RETURN_ABI_ALIGN_ATTR
        .try_into()
        .expect("return ABI alignment attribute name is valid");
    let u64_ty = IntegerType::get(ctx, 64, Signedness::Unsigned);
    let value = APInt::from_u64(alignment, NonZero::new(64).expect("64 is non-zero"));
    llvm_func
        .get_operation()
        .deref_mut(ctx)
        .attributes
        .set(key, IntegerAttr::new(u64_ty, value));
}

/// Propagate the `alwaysinline` attribute from MIR func to LLVM func.
///
/// Set on the MIR func op by `mir-importer` when the source Rust function
/// carries `#[inline(always)]`. The LLVM exporter then emits the
/// `alwaysinline` keyword on the `define` line. Existing `opt -O2` runs can
/// honor that attribute before `llc`, but this propagation is not a mandatory
/// always-inline pass. The goal is to preserve Rust's inline intent for device
/// helpers rather than leaving helper boundaries solely to optimizer
/// heuristics.
fn propagate_alwaysinline_attr(
    ctx: &mut Context,
    mir_op: Ptr<Operation>,
    llvm_func: &llvm::FuncOp,
) {
    let key: pliron::identifier::Identifier = "alwaysinline".try_into().unwrap();
    let attr_opt = mir_op
        .deref(ctx)
        .attributes
        .get::<pliron::builtin::attributes::StringAttr>(&key)
        .cloned();
    if let Some(attr) = attr_opt {
        llvm_func
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(key, attr);
    }
}

// ============================================================================
// Entry Block Prologue
// ============================================================================

/// Build LLVM entry block prologue: reconstruct aggregate args from flattened
/// LLVM block arguments and return the values to pass to the MIR entry block.
///
/// The LLVM entry block args reflect the post-flatten function signature.
/// Slices always arrive as `(ptr, len)` pairs and get re-assembled via
/// `insertvalue`. Ordinary structs only arrive flattened on the internal
/// device-fn ABI; at kernel boundaries they arrive as a single byval value.
/// A rustc-proven `repr(transparent)` scalar wrapper is the exception: its
/// kernel parameter is the underlying scalar, so the entry prologue rebuilds
/// the MIR struct from that one non-ZST field.
fn build_entry_prologue(
    ctx: &mut Context,
    mir_arg_types: &[TypeHandle],
    llvm_entry: Ptr<BasicBlock>,
    is_kernel_entry: bool,
) -> std::result::Result<Vec<Value>, anyhow::Error> {
    let llvm_args: Vec<_> = llvm_entry.deref(ctx).arguments().collect();
    let mut llvm_arg_idx = 0;
    let mut last_op: Option<Ptr<Operation>> = None;
    let mut result_args = Vec::new();

    for &mir_ty in mir_arg_types {
        let kind = classify_argument_type(ctx, mir_ty, is_kernel_entry)?;

        match kind {
            ReconstructKind::Slice { space_fields } => {
                let needed = 2 + space_fields;
                if llvm_arg_idx + needed > llvm_args.len() {
                    return Err(anyhow::anyhow!(
                        "Entry block arg mismatch: need {} more LLVM args for slice",
                        needed
                    ));
                }
                let ptr_val = llvm_args[llvm_arg_idx];
                let len_val = llvm_args[llvm_arg_idx + 1];
                let space_vals: Vec<Value> = (0..space_fields)
                    .map(|i| llvm_args[llvm_arg_idx + 2 + i])
                    .collect();
                llvm_arg_idx += needed;

                let (val, new_last) = reconstruct_slice(
                    ctx,
                    llvm_entry,
                    last_op,
                    mir_ty,
                    ptr_val,
                    len_val,
                    &space_vals,
                )?;
                last_op = Some(new_last);
                result_args.push(val);
            }
            ReconstructKind::Struct(num_fields) => {
                if llvm_arg_idx + num_fields > llvm_args.len() {
                    return Err(anyhow::anyhow!(
                        "Entry block arg mismatch: need {} more LLVM args for struct",
                        num_fields
                    ));
                }
                let field_vals: Vec<Value> = (0..num_fields)
                    .map(|i| llvm_args[llvm_arg_idx + i])
                    .collect();
                llvm_arg_idx += num_fields;

                let (val, new_last) =
                    reconstruct_struct(ctx, llvm_entry, last_op, mir_ty, &field_vals)?;
                last_op = Some(new_last);
                result_args.push(val);
            }
            ReconstructKind::TransparentScalar => {
                if llvm_arg_idx >= llvm_args.len() {
                    return Err(anyhow::anyhow!(
                        "Entry block arg mismatch: no scalar argument available for transparent struct"
                    ));
                }
                let scalar_val = llvm_args[llvm_arg_idx];
                llvm_arg_idx += 1;

                let (val, new_last) =
                    reconstruct_transparent_scalar(ctx, llvm_entry, last_op, mir_ty, scalar_val)?;
                last_op = Some(new_last);
                result_args.push(val);
            }
            ReconstructKind::Zst => {
                let llvm_ty = convert_type(ctx, mir_ty)?;
                let undef = llvm::UndefOp::new(ctx, llvm_ty).get_operation();
                insert_op_sequentially(undef, llvm_entry, last_op, ctx);
                last_op = Some(undef);
                result_args.push(undef.deref(ctx).get_result(0));
            }
            ReconstructKind::None => {
                if llvm_arg_idx >= llvm_args.len() {
                    return Err(anyhow::anyhow!(
                        "Entry block arg mismatch: no more LLVM args available"
                    ));
                }
                result_args.push(llvm_args[llvm_arg_idx]);
                llvm_arg_idx += 1;
            }
        }
    }

    Ok(result_args)
}

// ============================================================================
// Argument Classification
// ============================================================================

/// Classification of argument types for reconstruction strategy.
enum ReconstructKind {
    /// A slice type (`&[T]` or `DisjointSlice<T>`), flattened to `(ptr, len)`
    /// followed by `space_fields` index-space layout arguments.
    Slice { space_fields: usize },
    /// A struct type with N non-ZST fields, flattened to N separate arguments.
    Struct(usize),
    /// A rustc-proven transparent scalar wrapper, passed as one scalar.
    TransparentScalar,
    /// A zero-sized argument omitted from the LLVM signature.
    Zst,
    /// A simple type that passes through without reconstruction.
    None,
}

/// Classify an argument type to determine how to reconstruct it from
/// flattened LLVM entry block arguments.
///
/// At kernel-entry boundaries (`is_kernel_entry = true`) ordinary structs
/// arrive intact and are classified as `None`. A rustc-proven transparent
/// scalar wrapper arrives as one scalar field and is classified as
/// `TransparentScalar` so the MIR aggregate is reconstructed. Slices keep their
/// `(ptr, len)` reconstruction on both ABIs.
fn classify_argument_type(
    ctx: &mut Context,
    arg_ty: TypeHandle,
    is_kernel_entry: bool,
) -> std::result::Result<ReconstructKind, anyhow::Error> {
    if convert_type(ctx, arg_ty).is_ok_and(|llvm_ty| is_zero_sized_type(ctx, llvm_ty)) {
        return Ok(ReconstructKind::Zst);
    }

    let (slice_space_tys, struct_info) = {
        let arg_ty_ref = arg_ty.deref(ctx);
        let slice_space_tys = if arg_ty_ref.is::<MirSliceType>() {
            Some(Vec::new())
        } else {
            arg_ty_ref
                .downcast_ref::<MirDisjointSliceType>()
                .map(|s| s.space_tys.clone())
        };
        let struct_info = arg_ty_ref
            .downcast_ref::<MirStructType>()
            .map(|s| (s.field_types.clone(), s.is_transparent_scalar()));
        (slice_space_tys, struct_info)
    };

    if let Some(space_tys) = slice_space_tys {
        // A zero-sized index-space field contributes no argument, matching
        // `convert_function_type`.
        let space_fields = space_tys
            .iter()
            .filter(|f| {
                convert_type(ctx, **f)
                    .map(|llvm_ty| !is_zero_sized_type(ctx, llvm_ty))
                    .unwrap_or(true)
            })
            .count();
        Ok(ReconstructKind::Slice { space_fields })
    } else if let Some((fields, is_transparent_scalar)) = struct_info {
        // Count non-ZST fields the same way `convert_function_type` does
        // — empty structs and structs of all-ZSTs are themselves ZST and
        // get dropped from the LLVM signature on both ABIs.
        let non_zst_count = fields
            .iter()
            .filter(|f| {
                convert_type(ctx, **f)
                    .map(|llvm_ty| !is_zero_sized_type(ctx, llvm_ty))
                    .unwrap_or(true)
            })
            .count();

        if non_zst_count == 0 {
            Ok(ReconstructKind::Zst)
        } else if is_kernel_entry && is_transparent_scalar {
            // Validate the one-non-ZST-field invariant here as well as during
            // signature conversion, so hand-written dialect input fails closed.
            let _ = transparent_scalar_field(ctx, arg_ty)?;
            Ok(ReconstructKind::TransparentScalar)
        } else if is_kernel_entry {
            // Ordinary kernel-boundary struct arrived as a single byval value,
            // so the MIR entry block can consume it directly.
            Ok(ReconstructKind::None)
        } else {
            Ok(ReconstructKind::Struct(non_zst_count))
        }
    } else {
        Ok(ReconstructKind::None)
    }
}

// ============================================================================
// Aggregate Reconstruction
// ============================================================================

/// Reconstruct a slice value from its flattened fields.
///
/// Generates: `undef → insertvalue ptr[0] → insertvalue len[1]`, then one
/// `insertvalue` per index-space layout field at slot `2 + i`. Leaving those
/// slots undef would give every thread a garbage row width, so the count here
/// has to match what `convert_function_type` put in the signature.
///
/// Returns the final reconstructed value and the last inserted operation.
fn reconstruct_slice(
    ctx: &mut Context,
    llvm_block: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    mir_ty: TypeHandle,
    ptr_val: Value,
    len_val: Value,
    space_vals: &[Value],
) -> std::result::Result<(Value, Ptr<Operation>), anyhow::Error> {
    let struct_ty = convert_type(ctx, mir_ty)?;

    let undef = llvm::UndefOp::new(ctx, struct_ty);
    let undef_op = undef.get_operation();
    insert_op_sequentially(undef_op, llvm_block, prev_op, ctx);
    let undef_val = undef_op.deref(ctx).get_result(0);

    let insert_ptr = llvm::InsertValueOp::new(ctx, undef_val, ptr_val, vec![0]);
    let insert_ptr_op = insert_ptr.get_operation();
    insert_ptr_op.insert_after(ctx, undef_op);
    let val_with_ptr = insert_ptr_op.deref(ctx).get_result(0);

    let insert_len = llvm::InsertValueOp::new(ctx, val_with_ptr, len_val, vec![1]);
    let insert_len_op = insert_len.get_operation();
    insert_len_op.insert_after(ctx, insert_ptr_op);

    let mut last_op = insert_len_op;
    let mut current_val = insert_len_op.deref(ctx).get_result(0);
    for (i, &space_val) in space_vals.iter().enumerate() {
        let insert_space =
            llvm::InsertValueOp::new(ctx, current_val, space_val, vec![2 + i as u32]);
        let insert_space_op = insert_space.get_operation();
        insert_space_op.insert_after(ctx, last_op);
        current_val = insert_space_op.deref(ctx).get_result(0);
        last_op = insert_space_op;
    }

    Ok((current_val, last_op))
}

/// Reconstruct a nested `repr(transparent)` scalar wrapper from one LLVM scalar.
///
/// Each wrapper layer has exactly one non-ZST field. If that field is another
/// transparent scalar struct, rebuild it first, then insert the resulting value
/// into the outer layer. ZST marker fields remain omitted exactly as in ordinary
/// struct reconstruction.
fn reconstruct_transparent_scalar(
    ctx: &mut Context,
    llvm_block: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    mir_ty: TypeHandle,
    scalar_val: Value,
) -> std::result::Result<(Value, Ptr<Operation>), anyhow::Error> {
    let field_ty = transparent_scalar_field(ctx, mir_ty)?;
    let nested_transparent = {
        let field_ref = field_ty.deref(ctx);
        field_ref
            .downcast_ref::<MirStructType>()
            .is_some_and(MirStructType::is_transparent_scalar)
    };

    if nested_transparent {
        let (field_val, nested_last) =
            reconstruct_transparent_scalar(ctx, llvm_block, prev_op, field_ty, scalar_val)?;
        reconstruct_struct(ctx, llvm_block, Some(nested_last), mir_ty, &[field_val])
    } else {
        reconstruct_struct(ctx, llvm_block, prev_op, mir_ty, &[scalar_val])
    }
}

/// Reconstruct a struct value from flattened field values.
///
/// `field_vals` carries the flattened args in memory order with ZST fields
/// skipped (the same walk `convert_function_type` for the callee signature
/// and `flatten_arguments` at call sites use). Each value is inserted at the LLVM
/// slot [`build_struct_slot_map`] assigned to its field, so reconstruction
/// skips `[N x i8]` padding slots instead of inserting into them
/// (issue #128).
///
/// Generates: `undef → insertvalue field[slot] → ...`.
/// Returns the final reconstructed value and the last inserted operation.
fn reconstruct_struct(
    ctx: &mut Context,
    llvm_block: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    mir_ty: TypeHandle,
    field_vals: &[Value],
) -> std::result::Result<(Value, Ptr<Operation>), anyhow::Error> {
    let layout = {
        let ty_ref = mir_ty.deref(ctx);
        match ty_ref.downcast_ref::<MirStructType>() {
            Some(s) => StructLayoutInfo::of_struct(s),
            None => {
                return Err(anyhow::anyhow!(
                    "reconstruct_struct: expected a MirStructType argument"
                ));
            }
        }
    };
    let map = build_struct_slot_map(ctx, &layout)?;

    let undef = llvm::UndefOp::new(ctx, map.llvm_struct_ty);
    let undef_op = undef.get_operation();
    insert_op_sequentially(undef_op, llvm_block, prev_op, ctx);
    let mut current_struct = undef_op.deref(ctx).get_result(0);
    let mut last_op = undef_op;

    let mut vals = field_vals.iter();
    for &decl_idx in &layout.mem_to_decl {
        let Some(slot) = map.decl_to_llvm[decl_idx] else {
            continue; // ZST field: never flattened into an arg.
        };
        let Some(field_val) = vals.next() else {
            return Err(anyhow::anyhow!(
                "reconstruct_struct: fewer flattened args than non-ZST struct fields"
            ));
        };
        let insert_field = llvm::InsertValueOp::new(ctx, current_struct, *field_val, vec![slot]);
        let insert_op = insert_field.get_operation();
        insert_op.insert_after(ctx, last_op);
        current_struct = insert_op.deref(ctx).get_result(0);
        last_op = insert_op;
    }
    if vals.next().is_some() {
        return Err(anyhow::anyhow!(
            "reconstruct_struct: more flattened args than non-ZST struct fields"
        ));
    }

    Ok((current_struct, last_op))
}

/// Insert an op sequentially: after `prev` if given, otherwise at block front.
fn insert_op_sequentially(
    op: Ptr<Operation>,
    block: Ptr<BasicBlock>,
    prev: Option<Ptr<Operation>>,
    ctx: &Context,
) {
    if let Some(prev_op) = prev {
        op.insert_after(ctx, prev_op);
    } else {
        op.insert_at_front(block, ctx);
    }
}

// ============================================================================
// Dynamic Shared Memory Pre-scan
// ============================================================================

/// Compute the maximum dynamic shared memory alignment across all
/// `MirExternSharedOp` operations in a function.
///
/// This pre-pass must run BEFORE `inline_region` moves the blocks, since
/// it iterates the MIR blocks directly. The result is stored in
/// [`DynamicSmemAlignmentMap`] so that later per-op converters can
/// create the global with the correct alignment.
fn compute_max_dynamic_smem_alignment(
    ctx: &Context,
    mir_blocks: &[Ptr<BasicBlock>],
) -> Option<u64> {
    let mut max_alignment: Option<u64> = None;

    for mir_block in mir_blocks {
        for op in mir_block.deref(ctx).iter(ctx) {
            let op_id = Operation::get_opid(op, ctx);
            if op_id == dialect_mir::ops::MirExternSharedOp::get_opid_static() {
                let extern_shared = dialect_mir::ops::MirExternSharedOp::new(op);
                let alignment = extern_shared.get_alignment_value(ctx);

                max_alignment = Some(match max_alignment {
                    Some(current_max) => current_max.max(alignment),
                    None => alignment,
                });
            }
        }
    }

    max_alignment
}

// ============================================================================
// Error Conversion
// ============================================================================

/// Convert an `anyhow::Error` into a `pliron::result::Error`.
fn anyhow_to_pliron(e: anyhow::Error) -> pliron::result::Error {
    pliron::create_error!(
        pliron::location::Location::Unknown,
        pliron::result::ErrorKind::VerificationFailed,
        pliron::result::StringError(e.to_string())
    )
}

// ============================================================================
// Pass Registration
// ============================================================================

/// Register the MIR → LLVM lowering pass (placeholder for pass infrastructure).
pub fn register(_ctx: &mut Context) {}

#[cfg(test)]
mod dynamic_shared_contract_tests {
    use super::*;

    fn graph(entries: &[(&str, &[&str])]) -> FxHashMap<String, Vec<String>> {
        entries
            .iter()
            .map(|(caller, callees)| {
                (
                    (*caller).to_string(),
                    callees.iter().map(|callee| (*callee).to_string()).collect(),
                )
            })
            .collect()
    }

    #[test]
    fn propagation_is_transitive_cycle_safe_and_takes_shared_helper_maximum() {
        let call_graph = graph(&[
            ("kernel_32", &["forward", "external"]),
            ("kernel_256", &["shared"]),
            ("kernel_64", &["cycle_a"]),
            ("forward", &["shared"]),
            ("shared", &[]),
            ("cycle_a", &["cycle_b"]),
            ("cycle_b", &["cycle_a"]),
            ("marked_helper", &["marked_owner"]),
            ("marked_owner", &[]),
            ("uncontracted", &["unreached_helper"]),
            ("unreached_helper", &[]),
        ]);
        let contracts = [
            ("kernel_32".to_string(), 32),
            ("kernel_256".to_string(), 256),
            ("kernel_64".to_string(), 64),
            ("marked_helper".to_string(), 128),
        ];

        let propagated = propagate_alignments_through_call_graph(&call_graph, &contracts);

        assert_eq!(propagated.get("forward"), Some(&32));
        assert_eq!(propagated.get("shared"), Some(&256));
        assert_eq!(propagated.get("cycle_a"), Some(&64));
        assert_eq!(propagated.get("cycle_b"), Some(&64));
        assert_eq!(propagated.get("marked_owner"), Some(&128));
        assert!(!propagated.contains_key("external"));
        assert!(!propagated.contains_key("uncontracted"));
        assert!(!propagated.contains_key("unreached_helper"));
    }
}

#[cfg(test)]
mod transparent_scalar_abi_tests {
    use super::*;
    use dialect_mir::types::StructAbiKind;
    use pliron::builtin::types::{FunctionType, IntegerType, Signedness};

    fn make_ctx() -> Context {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        crate::register(&mut ctx);
        ctx
    }

    fn u32_ty(ctx: &mut Context) -> TypeHandle {
        IntegerType::get(ctx, 32, Signedness::Unsigned).into()
    }

    fn transparent_struct(
        ctx: &mut Context,
        name: &str,
        fields: Vec<TypeHandle>,
        offsets: Vec<u64>,
        total_size: u64,
        abi_align: u64,
    ) -> TypeHandle {
        let names = (0..fields.len()).map(|index| index.to_string()).collect();
        let memory_order = (0..fields.len()).collect();
        MirStructType::get_with_full_layout_and_abi(
            ctx,
            name.into(),
            names,
            fields,
            memory_order,
            offsets,
            total_size,
            abi_align,
            StructAbiKind::TransparentScalar,
        )
        .into()
    }

    #[test]
    fn kernel_transparent_scalar_reconstructs_from_one_field() {
        let mut ctx = make_ctx();
        let value = u32_ty(&mut ctx);
        let wrapper = transparent_struct(&mut ctx, "Scalar", vec![value], vec![0], 4, 4);

        assert!(matches!(
            classify_argument_type(&mut ctx, wrapper, true).unwrap(),
            ReconstructKind::TransparentScalar
        ));
    }

    #[test]
    fn kernel_transparent_scalar_ignores_zst_markers() {
        let mut ctx = make_ctx();
        let value = u32_ty(&mut ctx);
        let marker: TypeHandle =
            MirStructType::get(&mut ctx, "Marker".into(), vec![], vec![]).into();
        let wrapper = transparent_struct(&mut ctx, "Marked", vec![value, marker], vec![0, 4], 4, 4);

        assert!(matches!(
            classify_argument_type(&mut ctx, wrapper, true).unwrap(),
            ReconstructKind::TransparentScalar
        ));
    }

    #[test]
    fn ordinary_one_field_kernel_struct_stays_aggregate() {
        let mut ctx = make_ctx();
        let value = u32_ty(&mut ctx);
        let wrapper: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Ordinary".into(),
            vec!["0".into()],
            vec![value],
            vec![0],
            vec![0],
            4,
            4,
        )
        .into();

        assert!(matches!(
            classify_argument_type(&mut ctx, wrapper, true).unwrap(),
            ReconstructKind::None
        ));
    }

    #[test]
    fn packed_kernel_struct_stays_by_value_and_internal_path_reconstructs_fields() {
        let mut ctx = make_ctx();
        let tag: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let value = u32_ty(&mut ctx);
        let packed: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Packed".into(),
            vec!["tag".into(), "value".into()],
            vec![tag, value],
            vec![0, 1],
            vec![0, 1],
            5,
            1,
        )
        .into();

        assert!(matches!(
            classify_argument_type(&mut ctx, packed, true).unwrap(),
            ReconstructKind::None
        ));
        assert!(matches!(
            classify_argument_type(&mut ctx, packed, false).unwrap(),
            ReconstructKind::Struct(2)
        ));

        let llvm_ty = convert_type(&mut ctx, packed).expect("packed struct must lower");
        let llvm_ty_ref = llvm_ty.deref(&ctx);
        let llvm_struct = llvm_ty_ref
            .downcast_ref::<llvm_export::types::StructType>()
            .expect("packed MIR struct must lower to an LLVM struct");
        assert_eq!(
            llvm_struct.layout(),
            llvm_export::types::StructLayout::Packed
        );
    }

    #[test]
    fn packed_two_kernel_param_carries_rust_abi_alignment_override() {
        let mut ctx = make_ctx();
        let tag: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let value = u32_ty(&mut ctx);
        let packed1: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Packed1".into(),
            vec!["tag".into(), "value".into()],
            vec![tag, value],
            vec![0, 1],
            vec![0, 1],
            5,
            1,
        )
        .into();
        let packed2: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Packed2".into(),
            vec!["tag".into(), "value".into()],
            vec![tag, value],
            vec![0, 1],
            vec![0, 2],
            6,
            2,
        )
        .into();
        let mir_func_type = FunctionType::get(&ctx, vec![packed1, packed2], vec![]);
        let llvm_func_type = convert_function_type(&mut ctx, mir_func_type, true)
            .expect("packed kernel parameters must lower");

        let alignments = kernel_param_abi_alignments(&mut ctx, mir_func_type, llvm_func_type)
            .expect("kernel parameter alignments must map");

        assert_eq!(alignments, vec![(1, 2)]);
    }

    #[test]
    fn packed_two_return_carries_rust_abi_alignment_override() {
        let mut ctx = make_ctx();
        let tag: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let value = u32_ty(&mut ctx);
        let packed2: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Packed2".into(),
            vec!["tag".into(), "value".into()],
            vec![tag, value],
            vec![0, 1],
            vec![0, 2],
            6,
            2,
        )
        .into();
        let mir_func_type = FunctionType::get(&ctx, vec![], vec![packed2]);
        let llvm_func_type = convert_function_type(&mut ctx, mir_func_type, false)
            .expect("packed return must lower");

        assert_eq!(
            function_return_abi_alignment(&mut ctx, mir_func_type, llvm_func_type)
                .expect("return alignment must map"),
            Some(2)
        );
    }

    #[test]
    fn nested_transparent_scalar_reaches_underlying_integer() {
        let mut ctx = make_ctx();
        let value = u32_ty(&mut ctx);
        let inner = transparent_struct(&mut ctx, "Inner", vec![value], vec![0], 4, 4);
        let outer = transparent_struct(&mut ctx, "Outer", vec![inner], vec![0], 4, 4);

        let llvm_ty = crate::convert::types::transparent_scalar_llvm_type(&mut ctx, outer)
            .expect("nested transparent scalar must lower");
        let width = llvm_ty
            .deref(&ctx)
            .downcast_ref::<IntegerType>()
            .expect("underlying type must be an integer")
            .width();
        assert_eq!(width, 32);
    }

    #[test]
    fn transparent_pointer_reaches_underlying_pointer() {
        let mut ctx = make_ctx();
        let value = u32_ty(&mut ctx);
        let pointer: TypeHandle =
            dialect_mir::types::MirPtrType::get_generic(&mut ctx, value, false).into();
        let wrapper = transparent_struct(&mut ctx, "Pointer", vec![pointer], vec![0], 8, 8);

        let llvm_ty = crate::convert::types::transparent_scalar_llvm_type(&mut ctx, wrapper)
            .expect("transparent pointer must lower");
        assert!(llvm_ty.deref(&ctx).is::<llvm_export::types::PointerType>());
    }

    #[test]
    fn malformed_transparent_scalar_fails_closed() {
        let mut ctx = make_ctx();
        let a = u32_ty(&mut ctx);
        let b = u32_ty(&mut ctx);
        let wrapper = transparent_struct(&mut ctx, "Bad", vec![a, b], vec![0, 4], 8, 4);

        let error = classify_argument_type(&mut ctx, wrapper, true)
            .err()
            .expect("malformed transparent scalar must fail");
        assert!(error.to_string().contains("more than one non-ZST field"));
    }
}
