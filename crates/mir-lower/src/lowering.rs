/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `dialect-mir` → `dialect-llvm` function lowering via `inline_region`.
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
    convert_function_type, convert_type, is_kernel_func, is_zero_sized_type,
};

use dialect_llvm::ops as llvm;
use dialect_mir::ops::MirFuncOp;
use dialect_mir::types::{MirDisjointSliceType, MirSliceType, MirStructType};
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
    r#type::TypeObj,
    value::Value,
};

// ============================================================================
// Function Conversion
// ============================================================================

/// Convert a `MirFuncOp` to `llvm.func` using pliron's `inline_region`.
///
/// Called from [`crate::MirToLlvmConversionDriver::rewrite`] when the
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
    let llvm_func_type =
        convert_function_type(ctx, func_type, is_kernel).map_err(anyhow_to_pliron)?;

    let llvm_func = llvm::FuncOp::new(ctx, name, llvm_func_type);

    if is_kernel {
        propagate_kernel_attrs(ctx, op, &llvm_func, &kernel_key);
    }

    let llvm_entry = llvm_func.get_or_create_entry_block(ctx);

    let mir_region = op.deref(ctx).get_region(0);
    let mir_entry = mir_region.deref(ctx).get_head();

    if let Some(mir_entry) = mir_entry {
        // Pre-scan MIR blocks for max dynamic shared memory alignment.
        // Must happen BEFORE inline_region empties the MIR region.
        let mir_blocks: Vec<_> = mir_region.deref(ctx).iter(ctx).collect();
        let max_align = compute_max_dynamic_smem_alignment(ctx, &mir_blocks);

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
    llvm_func
        .get_operation()
        .deref_mut(ctx)
        .attributes
        .0
        .insert(
            kernel_key.clone(),
            pliron::builtin::attributes::StringAttr::new("true".to_string()).into(),
        );

    // Extract MIR attrs first to avoid borrow overlap with deref_mut below
    let attrs_to_copy: Vec<_> = {
        let mir_attrs = &mir_op.deref(ctx).attributes.0;
        [
            "cluster_dim_x",
            "cluster_dim_y",
            "cluster_dim_z",
            "maxntid",
            "minctasm",
        ]
        .iter()
        .filter_map(|key_str| {
            let key: pliron::identifier::Identifier = (*key_str).try_into().unwrap();
            mir_attrs.get(&key).map(|attr| (key, attr.clone()))
        })
        .collect()
    };

    for (key, attr) in attrs_to_copy {
        llvm_func
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .0
            .insert(key, attr);
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
/// `insertvalue`. Structs only arrive flattened on the internal device-fn
/// ABI; at kernel boundaries (`is_kernel_entry = true`) they arrive as a
/// single byval value and pass through. This function decides the shape
/// per-argument and emits the matching reconstruction sequence.
fn build_entry_prologue(
    ctx: &mut Context,
    mir_arg_types: &[Ptr<TypeObj>],
    llvm_entry: Ptr<BasicBlock>,
    is_kernel_entry: bool,
) -> std::result::Result<Vec<Value>, anyhow::Error> {
    let llvm_args: Vec<_> = llvm_entry.deref(ctx).arguments().collect();
    let mut llvm_arg_idx = 0;
    let mut last_op: Option<Ptr<Operation>> = None;
    let mut result_args = Vec::new();

    for &mir_ty in mir_arg_types {
        let kind = classify_argument_type(ctx, mir_ty, is_kernel_entry);

        match kind {
            ReconstructKind::Slice => {
                if llvm_arg_idx + 1 >= llvm_args.len() {
                    return Err(anyhow::anyhow!(
                        "Entry block arg mismatch: need 2 more LLVM args for slice"
                    ));
                }
                let ptr_val = llvm_args[llvm_arg_idx];
                let len_val = llvm_args[llvm_arg_idx + 1];
                llvm_arg_idx += 2;

                let (val, new_last) =
                    reconstruct_slice(ctx, llvm_entry, last_op, mir_ty, ptr_val, len_val)?;
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
    /// A slice type (`&[T]` or `DisjointSlice<T>`), flattened to `(ptr, len)`.
    Slice,
    /// A struct type with N non-ZST fields, flattened to N separate arguments.
    Struct(usize),
    /// A simple type that passes through without reconstruction.
    None,
}

/// Classify an argument type to determine how to reconstruct it from
/// flattened LLVM entry block arguments.
///
/// At kernel-entry boundaries (`is_kernel_entry = true`) structs arrive
/// intact, so they're classified as `None` even though the MIR type is
/// `MirStructType`. Slices keep their `(ptr, len)` reconstruction on
/// both ABIs.
fn classify_argument_type(
    ctx: &mut Context,
    arg_ty: Ptr<TypeObj>,
    is_kernel_entry: bool,
) -> ReconstructKind {
    let (is_slice, struct_fields) = {
        let arg_ty_ref = arg_ty.deref(ctx);
        let is_slice = arg_ty_ref.is::<MirSliceType>() || arg_ty_ref.is::<MirDisjointSliceType>();
        let struct_fields = arg_ty_ref
            .downcast_ref::<MirStructType>()
            .map(|s| s.field_types.clone());
        (is_slice, struct_fields)
    };

    if is_slice {
        ReconstructKind::Slice
    } else if let Some(fields) = struct_fields {
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
            // Whole struct is ZST: `convert_function_type` skipped it,
            // so no LLVM args were emitted. We still need to produce an
            // undef value for the MIR entry block's slot — `Struct(0)`
            // builds that via the existing reconstruct_struct path.
            ReconstructKind::Struct(0)
        } else if is_kernel_entry {
            // Kernel boundary: struct arrived as a single byval value,
            // so the MIR entry block can consume it directly without
            // any insertvalue prologue.
            ReconstructKind::None
        } else {
            ReconstructKind::Struct(non_zst_count)
        }
    } else {
        ReconstructKind::None
    }
}

// ============================================================================
// Aggregate Reconstruction
// ============================================================================

/// Reconstruct a slice value from flattened pointer and length.
///
/// Generates: `undef → insertvalue ptr[0] → insertvalue len[1]`.
/// Returns the final reconstructed value and the last inserted operation.
fn reconstruct_slice(
    ctx: &mut Context,
    llvm_block: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    mir_ty: Ptr<TypeObj>,
    ptr_val: Value,
    len_val: Value,
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
    let final_val = insert_len_op.deref(ctx).get_result(0);

    Ok((final_val, insert_len_op))
}

/// Reconstruct a struct value from flattened field values.
///
/// Generates: `undef → insertvalue field0[0] → insertvalue field1[1] → ...`.
/// Returns the final reconstructed value and the last inserted operation.
fn reconstruct_struct(
    ctx: &mut Context,
    llvm_block: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    mir_ty: Ptr<TypeObj>,
    field_vals: &[Value],
) -> std::result::Result<(Value, Ptr<Operation>), anyhow::Error> {
    let struct_ty = convert_type(ctx, mir_ty)?;

    let undef = llvm::UndefOp::new(ctx, struct_ty);
    let undef_op = undef.get_operation();
    insert_op_sequentially(undef_op, llvm_block, prev_op, ctx);
    let mut current_struct = undef_op.deref(ctx).get_result(0);
    let mut last_op = undef_op;

    for (field_idx, field_val) in field_vals.iter().enumerate() {
        let insert_field =
            llvm::InsertValueOp::new(ctx, current_struct, *field_val, vec![field_idx as u32]);
        let insert_op = insert_field.get_operation();
        insert_op.insert_after(ctx, last_op);
        current_struct = insert_op.deref(ctx).get_result(0);
        last_op = insert_op;
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
