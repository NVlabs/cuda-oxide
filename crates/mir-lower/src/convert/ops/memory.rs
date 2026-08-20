/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Memory operation conversion: `dialect-mir` → LLVM dialect.
//!
//! Converts `dialect-mir` memory operations to their LLVM dialect equivalents.
//!
//! # Operations
//!
//! | MIR Operation        | LLVM Operation(s)                 | Description                  |
//! |----------------------|-----------------------------------|------------------------------|
//! | `mir.load`           | `llvm.load`                       | Load from pointer            |
//! | `mir.store`          | `llvm.store`                      | Store to pointer             |
//! | `mir.ref`            | `llvm.alloca` + `llvm.store`      | Materialize aggregate in mem |
//! | `mir.ptr_offset`     | `llvm.getelementptr`              | Pointer arithmetic           |
//! | `mir.shared_alloc`   | `llvm.global` + `llvm.addressof`  | Static shared memory         |
//! | `mir.extern_shared`  | `llvm.global` + `llvm.addressof`  | Dynamic shared memory        |
//!
//! # Shared Memory
//!
//! ## Static Shared Memory (`SharedArray<T, N, ALIGN>`)
//!
//! Each static shared memory allocation gets a unique global symbol (`__shared_mem_N`).
//! Multiple allocations in the same or different kernels each have their own symbol
//! with their own size and alignment.
//!
//! ## Dynamic Shared Memory (`DynamicSharedArray<T, ALIGN>`)
//!
//! Dynamic shared memory uses a symbol for each function that owns an access
//! (`__dynamic_smem_{function_name}`).
//! Key characteristics:
//!
//! - **Per-owner symbols**: Each function containing an access gets an extern symbol
//! - **Pre-computed alignment**: A pre-pass combines the owner's body alignment with
//!   the strongest launch-contract marker that can reach it
//! - **Single runtime pool per launch**: The symbols refer to dynamic shared memory
//!   sized by `shared_mem_bytes` at launch
//!
//! ### PTX Output Example
//!
//! ```ptx
//! ; Kernel with 128-byte aligned dynamic shared memory
//! .extern .shared .align 128 .b8 __dynamic_smem_my_kernel[];
//!
//! ; Another kernel with 16-byte aligned (default)
//! .extern .shared .align 16 .b8 __dynamic_smem_other_kernel[];
//! ```

use crate::context::{DeviceGlobalsMap, DynamicSmemAlignmentMap, SharedGlobalsMap};
use crate::convert::types::{
    StructLayoutInfo, build_struct_slot_map, convert_type, get_type_size,
    llvm_packed_struct_contains_pointer_in_address_space, mir_type_abi_align,
    validate_initialized_global_layout, validate_relocated_initialized_global_layout,
};
use crate::helpers;
use dialect_mir::types::{MirPtrType, MirStructType};
use llvm_export::attributes::IntegerOverflowFlagsAttr;
use llvm_export::op_interfaces::IntBinArithOpWithOverflowFlag;
use llvm_export::ops as llvm;
use llvm_export::ops::GlobalOpExt;
use llvm_export::types::{ArrayType, FuncType, StructLayout, StructType, VoidType};
use pliron::attribute::AttrObj;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::op_interfaces::CallOpCallable;
use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::identifier::Identifier;
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::linked_list::ContainsLinkedList;
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};
use pliron::utils::apint::APInt;
use pliron::value::Value;

fn anyhow_to_pliron(e: anyhow::Error) -> pliron::result::Error {
    pliron::create_error!(
        pliron::location::Location::Unknown,
        pliron::result::ErrorKind::VerificationFailed,
        pliron::result::StringError(e.to_string())
    )
}

/// Convert `mir.store` to `llvm.store`.
///
/// Operand order: `[ptr, value]` - stores `value` to address `ptr`.
/// No result is produced (store is a side effect).
pub(crate) fn convert_store(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    let (ptr, val) = match operands.as_slice() {
        [ptr, val] => (*ptr, *val),
        _ => {
            return pliron::input_err_noloc!("Store operation requires exactly 2 operands");
        }
    };

    // Packed whole-value stores are byte-faithful now that divergent rustc
    // layouts lower to LLVM packed structs. Keep the target-dependent AS3 case
    // fail-closed because its physical pointer width is selected only later.
    fail_on_target_dependent_packed_aggregate(
        ctx,
        value_mir_type(ctx, operands_info, val),
        "storing",
    )?;

    let llvm_store = llvm::StoreOp::new(ctx, val, ptr);
    if dialect_mir::ops::MirStoreOp::new(op).is_volatile(ctx) {
        llvm_export::ops::set_op_volatile(ctx, llvm_store.get_operation(), true);
    }
    // The stored value's own type answers first, as it did before. A scalar
    // records none, though, so fall back to whatever the address itself proved
    // when it was computed -- for a field projection that is the aggregate's
    // `abi_align` narrowed to the field's offset, which is otherwise lost here
    // and costs the pair its vectorization. This mirrors `convert_load`, which
    // consults the same record for the same reason. When both answer, the
    // weaker wins: a field of a packed aggregate can place an abi-aligned type
    // at a byte-aligned address, and the address's proved alignment is the
    // ceiling of what the store may claim.
    let abi = value_abi_align(ctx, operands_info, val);
    let proved = pointer_proved_alignment(ctx, ptr);
    let align = match (abi, proved) {
        (Some(abi), Some(proved)) => Some(abi.min(proved)),
        (abi, proved) => abi.or(proved),
    };
    if let Some(align) = align {
        llvm_export::ops::set_op_alignment(ctx, llvm_store.get_operation(), align as u32);
    }
    crate::convert::preserve_location(ctx, op, llvm_store.get_operation());
    rewriter.insert_operation(ctx, llvm_store.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Recover the `repr(align(N))` ABI alignment of `value` at conversion time.
///
/// The alignment lives on MIR aggregate types (`abi_align`); the converted
/// LLVM struct types cannot express over-alignment. The conversion driver may
/// already have converted the value's type (block arguments are converted
/// before any rewrite runs; replaced op results carry the new type), but it
/// records every such change. So check the current type first, then walk the
/// value's conversion history, newest first, for a MIR type that records an
/// alignment.
fn value_abi_align(ctx: &Context, operands_info: &OperandsInfo, value: Value) -> Option<u64> {
    mir_type_abi_align(ctx, value.get_type(ctx)).or_else(|| {
        operands_info
            .lookup_operand_history(value)
            .iter()
            .rev()
            .find_map(|ty| mir_type_abi_align(ctx, *ty))
    })
}

/// The MIR type of `value`: its current type when that is still a MIR type,
/// else the newest MIR type in its conversion history (same walk as
/// [`value_abi_align`], for the same reason).
fn value_mir_type(ctx: &Context, operands_info: &OperandsInfo, value: Value) -> TypeHandle {
    let current = value.get_type(ctx);
    if current.deref(ctx).is::<MirStructType>() {
        return current;
    }
    operands_info
        .lookup_operand_history(value)
        .iter()
        .rev()
        .copied()
        .find(|ty| ty.deref(ctx).is::<MirStructType>())
        .unwrap_or(current)
}

/// Refuse only packed by-value images whose physical bytes depend on the
/// selected NVPTX mode.
///
/// Shared-memory pointers are 32-bit in modern NVVM p3:32 but 64-bit in the
/// PTX/legacy layouts. Lowering runs before that target mode is selected, so a
/// packed aggregate containing AS3 cannot safely be loaded or stored as one
/// physical value. Pointer-free packed aggregates and packed aggregates whose
/// pointers use target-stable address spaces are allowed.
fn fail_on_target_dependent_packed_aggregate(
    ctx: &mut Context,
    ty: TypeHandle,
    verb: &str,
) -> Result<()> {
    let llvm_ty = {
        let layout = {
            let ty_ref = ty.deref(ctx);
            ty_ref
                .downcast_ref::<MirStructType>()
                .map(StructLayoutInfo::of_struct)
        };
        if let Some(layout) = layout {
            let map = build_struct_slot_map(ctx, &layout).map_err(anyhow_to_pliron)?;
            if !map.by_value_layout_faithful {
                return pliron::input_err_noloc!(
                    "{} a struct whose rustc layout cannot be represented by an LLVM \
                     struct value is not supported",
                    verb
                );
            }
            map.llvm_struct_ty
        } else {
            convert_type(ctx, ty).map_err(anyhow_to_pliron)?
        }
    };
    if llvm_packed_struct_contains_pointer_in_address_space(
        ctx,
        llvm_ty,
        llvm_export::types::address_space::SHARED,
    ) {
        return pliron::input_err_noloc!(
            "{} a packed aggregate containing a shared-memory pointer by value is \
             target-mode dependent and is not yet supported",
            verb
        );
    }
    Ok(())
}

fn copy_debug_local_variable(ctx: &mut Context, mir_op: Ptr<Operation>, llvm_op: Ptr<Operation>) {
    if let Some(info) = llvm_export::ops::debug_local_variable(ctx, mir_op) {
        llvm_export::ops::set_debug_local_variable(ctx, llvm_op, info);
    }
    let projected = llvm_export::ops::debug_projected_variables(ctx, mir_op);
    if !projected.is_empty() {
        llvm_export::ops::set_debug_projected_variables(ctx, llvm_op, &projected);
    }
    let fragments = llvm_export::ops::debug_fragment_variables(ctx, mir_op);
    if !fragments.is_empty() {
        llvm_export::ops::set_debug_fragment_variables(ctx, llvm_op, &fragments);
    }
    if let Some(scope) = llvm_export::ops::debug_local_source_scope(ctx, mir_op) {
        llvm_export::ops::set_debug_local_source_scope(ctx, llvm_op, scope);
    }
    if let Some((file, pos)) = llvm_export::ops::debug_local_declaration_location(ctx, mir_op) {
        llvm_export::ops::set_debug_local_declaration_location(
            ctx, llvm_op, file, pos.line, pos.column,
        );
    }
    if let Some(expression) = llvm_export::ops::debug_value_expression(ctx, mir_op) {
        llvm_export::ops::set_debug_value_expression(ctx, llvm_op, &expression);
    }
}

/// Carry Rust-local provenance from the MIR alloca to the LLVM alloca.
///
/// This is compiler-only metadata stored in the in-memory dialect operation;
/// it is not emitted as LLVM metadata because an unrecognized metadata node is
/// not a preservation contract for middle-end passes. The textual exporter
/// instead consumes the attribute to choose a stable SSA name before `opt`.
fn copy_local_memory_provenance(
    ctx: &mut Context,
    mir_op: Ptr<Operation>,
    llvm_op: Ptr<Operation>,
) {
    if let Some(provenance) = llvm_export::ops::local_memory_provenance(ctx, mir_op) {
        llvm_export::ops::set_local_memory_provenance(ctx, llvm_op, provenance);
    }
}

/// Convert `mir.memcpy` to the matching `llvm.memcpy.p<dst>.p<src>.i<bits>`.
///
/// MIR's count is measured in pointee elements, while LLVM's memcpy intrinsic
/// expects bytes. The pre-conversion destination pointer type still carries the
/// MIR pointee, so use it to scale the count before emitting the call.
///
/// The intrinsic name is an overload: LLVM encodes each pointer's address
/// space and the length width into it, and its verifier rejects a call whose
/// argument types disagree with the name. Today every pointer reaching a
/// `copy_nonoverlapping` is a Rust raw pointer, which cuda-oxide normalizes to
/// the generic address space (an `addrspacecast` is inserted when the raw
/// pointer is formed), so the operands are always `p0` and `i64`. We still
/// derive the suffix from the real operand types rather than hardcoding
/// `p0.p0.i64`: it matches how every other overloaded intrinsic is named here
/// (`ctpop`, `fptosi.sat`, ...), and it keeps this lowering correct if raw
/// pointers ever start carrying a non-generic address space.
pub(crate) fn convert_memcpy(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    convert_mem_transfer(ctx, rewriter, op, operands_info, "memcpy")
}

/// Convert `mir.memmove` to the matching `llvm.memmove.p<dst>.p<src>.i<bits>`.
///
/// Identical to [`convert_memcpy`] except it emits the overlap-safe
/// `llvm.memmove` intrinsic. `mir.memmove` backs `core::intrinsics::copy`
/// (`ptr::copy`); `mir.memcpy` backs the non-overlapping variant.
pub(crate) fn convert_memmove(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    convert_mem_transfer(ctx, rewriter, op, operands_info, "memmove")
}

/// Shared lowering for `mir.memcpy` / `mir.memmove`. `intrinsic_base` selects
/// the LLVM intrinsic family ("memcpy" or "memmove"); both share the same
/// `(dst, src, len_bytes, isvolatile)` signature and element->byte count scaling.
fn convert_mem_transfer(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
    intrinsic_base: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let (dst, src, count) = match operands.as_slice() {
        [dst, src, count] => (*dst, *src, *count),
        _ => {
            return pliron::input_err_noloc!(
                "{intrinsic_base} operation requires exactly 3 operands"
            );
        }
    };

    let pointee = {
        let dst_ptr_ty = operands_info
            .lookup_most_recent_of_type::<MirPtrType>(ctx, dst)
            .ok_or_else(|| {
                pliron::create_error!(
                    op.deref(ctx).loc(),
                    pliron::result::ErrorKind::VerificationFailed,
                    pliron::result::StringError(format!(
                        "{intrinsic_base} destination must be a MIR pointer before lowering"
                    ))
                )
            })?;
        dst_ptr_ty.pointee
    };
    let elem_ty = convert_type(ctx, pointee).map_err(anyhow_to_pliron)?;
    let elem_size = get_type_size(ctx, elem_ty);

    let bytes = if elem_size == 1 {
        count
    } else {
        let count_ty = count.get_type(ctx);
        let bits = count_ty
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .map(|ty| ty.width())
            .unwrap_or(64);
        let count_int_ty = IntegerType::get(ctx, bits, Signedness::Signless);
        let size_attr: AttrObj = IntegerAttr::new(
            count_int_ty,
            APInt::from_u64(
                elem_size,
                std::num::NonZeroUsize::new(bits as usize).unwrap(),
            ),
        )
        .into();
        let size_const = llvm::ConstantOp::new(ctx, size_attr);
        let size_val = size_const.get_operation().deref(ctx).get_result(0);
        rewriter.insert_operation(ctx, size_const.get_operation());

        let flags = IntegerOverflowFlagsAttr::default();
        let mul = llvm::MulOp::new_with_overflow_flag(ctx, count, size_val, flags);
        rewriter.insert_operation(ctx, mul.get_operation());
        mul.get_operation().deref(ctx).get_result(0)
    };

    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let false_attr: AttrObj = IntegerAttr::new(
        i1_ty,
        APInt::from_u64(0, std::num::NonZeroUsize::new(1).unwrap()),
    )
    .into();
    let volatile = llvm::ConstantOp::new(ctx, false_attr);
    rewriter.insert_operation(ctx, volatile.get_operation());
    let volatile_val = volatile.get_operation().deref(ctx).get_result(0);

    let void_ty = VoidType::get(ctx);
    let func_ty = FuncType::get(
        ctx,
        void_ty.into(),
        vec![
            dst.get_type(ctx),
            src.get_type(ctx),
            bytes.get_type(ctx),
            volatile_val.get_type(ctx),
        ],
        false,
    );
    let parent_block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::create_error!(
            op.deref(ctx).loc(),
            pliron::result::ErrorKind::VerificationFailed,
            pliron::result::StringError(format!("{intrinsic_base} operation has no parent block"))
        )
    })?;
    // Derive the overload suffix from the real (already type-converted)
    // operands so the name can never disagree with the argument types.
    let dst_ty = dst.get_type(ctx);
    let dst_as = dst_ty
        .deref(ctx)
        .downcast_ref::<llvm_export::types::PointerType>()
        .map(|pt| pt.address_space())
        .unwrap_or(0);
    let src_ty = src.get_type(ctx);
    let src_as = src_ty
        .deref(ctx)
        .downcast_ref::<llvm_export::types::PointerType>()
        .map(|pt| pt.address_space())
        .unwrap_or(0);
    let len_bits = bytes
        .get_type(ctx)
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|t| t.width())
        .unwrap_or(64);
    let intrinsic_name = format!("llvm_{intrinsic_base}_p{dst_as}_p{src_as}_i{len_bits}");
    helpers::ensure_intrinsic_declared(ctx, parent_block, &intrinsic_name, func_ty)
        .map_err(anyhow_to_pliron)?;

    let callee: Identifier = intrinsic_name.as_str().try_into().map_err(|e| {
        pliron::create_error!(
            op.deref(ctx).loc(),
            pliron::result::ErrorKind::VerificationFailed,
            pliron::result::StringError(format!("Invalid memcpy intrinsic name: {e:?}"))
        )
    })?;
    let call = llvm::CallOp::new(
        ctx,
        CallOpCallable::Direct(callee),
        func_ty,
        vec![dst, src, bytes, volatile_val],
    );
    crate::convert::preserve_location(ctx, op, call.get_operation());
    rewriter.insert_operation(ctx, call.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Alignment the op that computed `ptr` recorded about the address it produced.
///
/// Field addresses stamp this during their own conversion, where the aggregate's
/// `abi_align` is still in hand; by the time a load that consumes the address is
/// converted, `mir.field_addr` is already a GEP and the MIR aggregate type is no
/// longer reachable from the load's operands.
///
/// Known fidelity limit: the claim rides the address value, not the place
/// access that justified it. `&raw const (*p).x` through a merely 4-aligned
/// `p: *const Aligned8` is legal Rust so long as the place itself is never
/// accessed, yet the later read through the raw pointer consumes this same
/// stamped GEP and inherits the aggregate's alignment, an over-claim for that
/// pointer. rustc keeps the two apart by threading alignment through place
/// evaluation and dropping it at the `&raw` boundary; dialect-mir erases that
/// distinction before lowering runs, so full fidelity would require stamping
/// loads during MIR translation instead. Accepted for now: the pattern needs
/// an actually under-aligned pointer to an over-aligned aggregate, projected
/// via `&raw` inside a kernel, and any direct access through such a place is
/// UB to begin with.
fn pointer_proved_alignment(ctx: &Context, ptr: Value) -> Option<u64> {
    let defining_op = ptr.defining_op()?;
    llvm_export::ops::address_alignment(ctx, defining_op).map(u64::from)
}

/// Convert `mir.load` to `llvm.load`.
///
/// Takes a single pointer operand and returns the loaded value.
/// The result type is derived from the MIR operation's result type.
pub(crate) fn convert_load(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let ptr = op.deref(ctx).get_operand(0);
    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);

    // Packed whole-value loads are byte-faithful now that divergent rustc
    // layouts lower to LLVM packed structs. Keep only the target-dependent AS3
    // physical-image case fail-closed.
    fail_on_target_dependent_packed_aggregate(ctx, result_ty, "loading")?;

    let llvm_ty = convert_type(ctx, result_ty).map_err(anyhow_to_pliron)?;

    let llvm_load = llvm::LoadOp::new(ctx, ptr, llvm_ty);
    if dialect_mir::ops::MirLoadOp::new(op).is_volatile(ctx) {
        llvm_export::ops::set_op_volatile(ctx, llvm_load.get_operation(), true);
    }
    // The loaded value's ABI alignment comes from this op's own result type,
    // which is still the MIR type: result types are only converted by the
    // op's own rewrite. A scalar records none, so fall back to whatever the
    // address itself proved when it was computed -- for a field projection
    // that is the aggregate's `abi_align` narrowed to the field's offset,
    // which is otherwise lost here and costs the pair its vectorization.
    // When both answer, the weaker wins: a field of a packed aggregate can
    // place an abi-aligned type at a byte-aligned address, and the address's
    // proved alignment is the ceiling of what the load may claim.
    let abi = mir_type_abi_align(ctx, result_ty);
    let proved = pointer_proved_alignment(ctx, ptr);
    let align = match (abi, proved) {
        (Some(abi), Some(proved)) => Some(abi.min(proved)),
        (abi, proved) => abi.or(proved),
    };
    if let Some(align) = align {
        llvm_export::ops::set_op_alignment(ctx, llvm_load.get_operation(), align as u32);
    }
    rewriter.insert_operation(ctx, llvm_load.get_operation());
    rewriter.replace_operation(ctx, op, llvm_load.get_operation());

    Ok(())
}

/// Convert `mir.dbg_value` to the LLVM-export debug marker.
///
/// The op is still debug-only after lowering. The textual LLVM exporter later
/// prints it as an `llvm.dbg.value` intrinsic call.
pub(crate) fn convert_dbg_value(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let value = op.deref(ctx).get_operand(0);
    let loc = op.deref(ctx).loc().clone();
    let llvm_dbg_value = llvm::DebugValueOp::new(ctx, value);
    llvm_dbg_value.get_operation().deref_mut(ctx).set_loc(loc);
    copy_debug_local_variable(ctx, op, llvm_dbg_value.get_operation());
    rewriter.insert_operation(ctx, llvm_dbg_value.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert `mir.dbg_value_list` to the LLVM-export multi-value debug marker.
///
/// The ordered operands become a `DIArgList` during textual export. The typed
/// location recipe is carried as generic metadata and copied unchanged here.
pub(crate) fn convert_dbg_value_list(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let values: Vec<_> = op.deref(ctx).operands().collect();
    if values.len() < 2 {
        return pliron::input_err_noloc!("mir.dbg_value_list requires at least two operands");
    }
    if llvm_export::ops::debug_value_expression(ctx, op).is_none() {
        return pliron::input_err_noloc!("mir.dbg_value_list is missing its debug expression");
    }

    let loc = op.deref(ctx).loc().clone();
    let llvm_dbg_value = llvm::DebugValueListOp::new(ctx, values);
    llvm_dbg_value.get_operation().deref_mut(ctx).set_loc(loc);
    copy_debug_local_variable(ctx, op, llvm_dbg_value.get_operation());
    rewriter.insert_operation(ctx, llvm_dbg_value.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Convert `mir.alloca` to `llvm.alloca`.
///
/// `mir.alloca` carries its element type on the result pointer's pointee, and
/// emits a single-element stack slot of that type. We therefore convert the
/// pointee to an LLVM type and emit `llvm.alloca <pointee_ty>, i32 1`.
///
/// No value is stored into the slot; that is the caller's job via subsequent
/// `mir.store` / `llvm.store` ops. This matches the mem2reg-ready translator
/// model where every local is backed by one alloca in the entry block and
/// defs/uses go through `store`/`load` rather than SSA values.
pub(crate) fn convert_alloca(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let mir_pointee = {
        let ty_ref = result_ty.deref(ctx);
        let mir_ptr = ty_ref.downcast_ref::<MirPtrType>().ok_or_else(|| {
            anyhow_to_pliron(anyhow::anyhow!(
                "MirAllocaOp result must be MirPtrType (enforced by verifier)"
            ))
        })?;
        mir_ptr.pointee
    };
    let llvm_pointee = convert_type(ctx, mir_pointee).map_err(anyhow_to_pliron)?;

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let one_apint =
        pliron::utils::apint::APInt::from_i64(1, std::num::NonZeroUsize::new(32).unwrap());
    let one_attr = pliron::builtin::attributes::IntegerAttr::new(i32_ty, one_apint);
    let one_const = llvm::ConstantOp::new(ctx, one_attr.into());
    rewriter.insert_operation(ctx, one_const.get_operation());
    let one_val = one_const.get_operation().deref(ctx).get_result(0);

    let alloca = llvm::AllocaOp::new(ctx, llvm_pointee, one_val);
    // The allocated type's ABI alignment comes from this op's own result
    // pointee, which is still the MIR type at rewrite time.
    if let Some(align) = mir_type_abi_align(ctx, mir_pointee) {
        llvm_export::ops::set_op_alignment(ctx, alloca.get_operation(), align as u32);
    }
    copy_debug_local_variable(ctx, op, alloca.get_operation());
    copy_local_memory_provenance(ctx, op, alloca.get_operation());
    rewriter.insert_operation(ctx, alloca.get_operation());
    rewriter.replace_operation(ctx, op, alloca.get_operation());

    Ok(())
}

/// Convert `mir.ref` — materialize the operand in stack memory via alloca+store.
///
/// `mir.ref` creates a pointer to an SSA value. In SSA form, values don't have
/// addresses, so we must place the value in memory to obtain a pointer.
/// This applies to all types: scalars (e.g. `&factor` where factor is `u32`),
/// aggregates (e.g. `&closure_env`), and pointers (e.g. `&&T`).
pub(crate) fn convert_ref(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let operand = op.deref(ctx).get_operand(0);
    let operand_ty = operand.get_type(ctx);
    let abi_align = value_abi_align(ctx, operands_info, operand);

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let one_apint =
        pliron::utils::apint::APInt::from_i64(1, std::num::NonZeroUsize::new(32).unwrap());
    let one_attr = pliron::builtin::attributes::IntegerAttr::new(i32_ty, one_apint);
    let one_const = llvm::ConstantOp::new(ctx, one_attr.into());
    rewriter.insert_operation(ctx, one_const.get_operation());
    let one_val = one_const.get_operation().deref(ctx).get_result(0);

    let alloca = llvm::AllocaOp::new(ctx, operand_ty, one_val);
    // Honour the referent's repr(align(N)) ABI alignment. Without this, the
    // synthesised alloca would be under-aligned relative to any loads/stores
    // that claim the struct's true alignment.
    if let Some(align) = abi_align {
        llvm_export::ops::set_op_alignment(ctx, alloca.get_operation(), align as u32);
    }
    rewriter.insert_operation(ctx, alloca.get_operation());
    let alloca_ptr = alloca.get_operation().deref(ctx).get_result(0);

    let store = llvm::StoreOp::new(ctx, operand, alloca_ptr);
    if let Some(align) = abi_align {
        llvm_export::ops::set_op_alignment(ctx, store.get_operation(), align as u32);
    }
    rewriter.insert_operation(ctx, store.get_operation());

    rewriter.replace_operation_with_values(ctx, op, vec![alloca_ptr]);

    Ok(())
}

/// Convert `mir.ptr_offset` to `llvm.getelementptr`.
///
/// Operands: `[ptr, offset]` where offset is an integer index.
/// Uses the pointee type from the MIR pointer type for element sizing.
/// Falls back to i8 element type if pointee type cannot be determined.
pub(crate) fn convert_ptr_offset(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    let (ptr, offset) = match operands.as_slice() {
        [ptr, offset] => (*ptr, *offset),
        _ => return pliron::input_err_noloc!("PtrOffset requires exactly 2 operands"),
    };

    let pointee_ty_opt = operands_info
        .lookup_most_recent_of_type::<MirPtrType>(ctx, ptr)
        .map(|mir_ptr| mir_ptr.pointee);

    let elem_ty = if let Some(pointee) = pointee_ty_opt {
        convert_type(ctx, pointee).map_err(anyhow_to_pliron)?
    } else {
        IntegerType::get(ctx, 8, Signedness::Signless).into()
    };

    let llvm_gep = llvm::GetElementPtrOp::new(
        ctx,
        ptr,
        vec![llvm_export::ops::GepIndex::Value(offset)],
        elem_ty,
    );
    let inbounds = dialect_mir::ops::MirPtrOffsetOp::new(op).is_inbounds(ctx);
    llvm::set_gep_inbounds(ctx, llvm_gep.get_operation(), inbounds);
    rewriter.insert_operation(ctx, llvm_gep.get_operation());
    rewriter.replace_operation(ctx, op, llvm_gep.get_operation());

    Ok(())
}

/// Convert `mir.shared_alloc` to LLVM global variable in shared address space.
///
/// GPU shared memory is represented as a global variable with address space 3.
/// Uses `shared_globals` to deduplicate: multiple allocations with the same
/// `alloc_key` share the same global.
///
/// Called directly from `MirToLlvmConversionDriver::rewrite` (not through
/// op_cast dispatch) because it needs the cross-function `SharedGlobalsMap`.
pub fn convert_shared_alloc_dc(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    shared_globals: &mut SharedGlobalsMap,
    next_shared_mem_index: &mut usize,
) -> Result<()> {
    use pliron::builtin::attributes::{IntegerAttr, TypeAttr};

    let (alloc_key, source_name, mir_elem_type, size, alignment) = {
        let shared_alloc_op = dialect_mir::ops::MirSharedAllocOp::new(op);
        let op_ref = op.deref(ctx);

        let alloc_key: Option<String> = shared_alloc_op
            .get_attr_alloc_key(ctx)
            .map(|s| String::from((*s).clone()));

        // Optional and diagnostic: the Rust path of the originating `static`,
        // carried through so the emitted global can name its source.
        let source_name: Option<String> = shared_alloc_op
            .get_attr_source_name(ctx)
            .map(|s| String::from((*s).clone()));

        let elem_type_attr = op_ref
            .attributes
            .get::<TypeAttr>(&"elem_type".try_into().unwrap())
            .ok_or_else(|| {
                anyhow_to_pliron(anyhow::anyhow!(
                    "MirSharedAllocOp missing elem_type TypeAttr attribute"
                ))
            })?;
        let mir_elem_type = elem_type_attr.get_type(ctx);

        let size_attr = op_ref
            .attributes
            .get::<IntegerAttr>(&"size".try_into().unwrap())
            .ok_or_else(|| {
                anyhow_to_pliron(anyhow::anyhow!(
                    "MirSharedAllocOp missing size IntegerAttr attribute"
                ))
            })?;
        let size = size_attr.value().to_u64();

        let alignment = shared_alloc_op.get_alignment_value(ctx).unwrap_or(0);

        (alloc_key, source_name, mir_elem_type, size, alignment)
    };

    // Cache hit only when the op carries a key AND that key is already in
    // `shared_globals`. `as_ref()` borrows for the if-let scope so the else
    // branch can still move `alloc_key` into `create_shared_global` (which
    // takes ownership and inserts it into the cache).
    let global_name = if let Some(key) = alloc_key.as_ref()
        && let Some(existing_name) = shared_globals.get(key)
    {
        existing_name.clone()
    } else {
        create_shared_global(
            ctx,
            op,
            shared_globals,
            next_shared_mem_index,
            SharedAllocSpec {
                mir_elem_type,
                size,
                alignment,
                alloc_key,
                source_name: source_name.as_deref(),
            },
        )?
    };

    let address_of_op = llvm::AddressOfOp::new(ctx, global_name, 3);
    rewriter.insert_operation(ctx, address_of_op.get_operation());
    rewriter.replace_operation(ctx, op, address_of_op.get_operation());

    Ok(())
}

/// Everything `create_shared_global` needs about one `mir.shared_alloc`.
///
/// Mirrors [`DeviceGlobalSpec`] for the shared-memory path.
struct SharedAllocSpec<'a> {
    mir_elem_type: TypeHandle,
    size: u64,
    alignment: u64,
    alloc_key: Option<String>,
    source_name: Option<&'a str>,
}

/// Create a shared memory global variable in the module.
///
/// Creates an LLVM global variable with:
/// - Array type: `[size x element_type]`
/// - Address space 3 (shared memory)
/// - Optional alignment
/// - Unique generated name (`__shared_mem_N`)
///
/// The global is inserted at the front of the module block. When
/// `spec.alloc_key` is `Some`, the key is moved into `shared_globals` so that
/// later allocations with the same key reuse this global (caller is
/// expected to have already checked the cache for a hit).
///
/// `spec.source_name`, when present, is the Rust path of the `static` this
/// allocation came from. The generated symbol stays anonymous; the name is
/// recorded as an attribute on the global so the exporter can render it
/// beside the definition. Only the allocation that *creates* the global
/// contributes a name — a later allocation with the same `alloc_key` hits
/// the cache and never reaches this function — which is consistent because
/// the key and the name are both derived from the same constant.
///
/// `next_shared_mem_index` is scoped to one `MirToLlvmConversionDriver`
/// instance (one module), not a process-global counter: `N` is a function of
/// this module's own MIR walk order, not of how many other modules have
/// lowered a shared allocation earlier in the process (#706).
fn create_shared_global(
    ctx: &mut Context,
    op: Ptr<Operation>,
    shared_globals: &mut SharedGlobalsMap,
    next_shared_mem_index: &mut usize,
    spec: SharedAllocSpec<'_>,
) -> Result<pliron::identifier::Identifier> {
    let llvm_elem_type = convert_type(ctx, spec.mir_elem_type).map_err(anyhow_to_pliron)?;
    let array_type = ArrayType::get(ctx, llvm_elem_type, spec.size);

    let counter = *next_shared_mem_index;
    *next_shared_mem_index += 1;
    let name: pliron::identifier::Identifier =
        format!("__shared_mem_{counter}").try_into().unwrap();

    let global_op = if spec.alignment > 0 {
        llvm::GlobalOp::new_with_alignment(ctx, name.clone(), array_type.into(), spec.alignment)
    } else {
        llvm::GlobalOp::new(ctx, name.clone(), array_type.into())
    };
    global_op.set_address_space(ctx, llvm_export::types::address_space::SHARED);
    if let Some(source_name) = spec.source_name {
        use llvm_export::ops::GlobalOpExt;
        global_op.set_shared_source_name(ctx, source_name);
    }

    let parent_block = op
        .deref(ctx)
        .get_parent_block()
        .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Op has no parent block")))?;
    let module_op = helpers::get_module_from_block(ctx, parent_block).map_err(anyhow_to_pliron)?;
    let region = module_op.deref(ctx).get_region(0);
    let module_block = region
        .deref(ctx)
        .iter(ctx)
        .next()
        .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Module is empty")))?;

    global_op.get_operation().insert_at_front(module_block, ctx);

    if let Some(key) = spec.alloc_key {
        shared_globals.insert(key, name.clone());
    }

    Ok(name)
}

/// Convert `mir.global_alloc` to an LLVM global in CUDA global memory.
///
/// Ordinary Rust `static` / `static mut` values have grid scope and
/// application lifetime, so they live in address space 1 rather than the
/// per-block shared-memory address space.
pub fn convert_global_alloc_dc(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    device_globals: &mut DeviceGlobalsMap,
    next_device_global_index: &mut usize,
) -> Result<()> {
    use pliron::builtin::attributes::{StringAttr, TypeAttr};

    let (
        global_key,
        mir_global_type,
        alignment,
        addr_space,
        initializer_hex,
        initializer_relocations,
        immutable,
    ) = {
        let global_op = dialect_mir::ops::MirGlobalAllocOp::new(op);
        let op_ref = op.deref(ctx);

        let global_key_attr = op_ref
            .attributes
            .get::<StringAttr>(&"global_key".try_into().unwrap())
            .ok_or_else(|| {
                anyhow_to_pliron(anyhow::anyhow!(
                    "MirGlobalAllocOp missing global_key StringAttr attribute"
                ))
            })?;
        let global_key = String::from((*global_key_attr).clone());

        let global_type_attr = op_ref
            .attributes
            .get::<TypeAttr>(&"global_type".try_into().unwrap())
            .ok_or_else(|| {
                anyhow_to_pliron(anyhow::anyhow!(
                    "MirGlobalAllocOp missing global_type TypeAttr attribute"
                ))
            })?;
        let mir_global_type = global_type_attr.get_type(ctx);

        let alignment = global_op.get_alignment_value(ctx).unwrap_or(0);
        let initializer_hex = op_ref
            .attributes
            .get::<StringAttr>(&"global_initializer_hex".try_into().unwrap())
            .map(|attr| String::from((*attr).clone()));
        let initializer_relocations = op_ref
            .attributes
            .get::<StringAttr>(&"global_initializer_relocations".try_into().unwrap())
            .map(|attr| String::from((*attr).clone()));

        // Read the address space the op's result already carries — set by
        // mir-importer based on the static's type (`ConstantMemory<T>` → 4,
        // ordinary → 1). The dialect verifier accepts both.
        let res_ty = op_ref.get_result(0).get_type(ctx);
        let addr_space = res_ty
            .deref(ctx)
            .downcast_ref::<dialect_mir::types::MirPtrType>()
            .map(|p| {
                if p.address_space == dialect_mir::types::address_space::CONSTANT {
                    llvm_export::types::address_space::CONSTANT
                } else {
                    llvm_export::types::address_space::GLOBAL
                }
            })
            .ok_or_else(|| {
                anyhow_to_pliron(anyhow::anyhow!(
                    "MirGlobalAllocOp result is not a MirPtrType"
                ))
            })?;

        (
            global_key,
            mir_global_type,
            alignment,
            addr_space,
            initializer_hex,
            initializer_relocations,
            global_op.is_immutable(ctx),
        )
    };

    let global_name = if let Some(existing_name) = device_globals.get(&global_key) {
        existing_name.clone()
    } else {
        create_device_global(
            ctx,
            op,
            device_globals,
            next_device_global_index,
            DeviceGlobalSpec {
                key: &global_key,
                mir_type: mir_global_type,
                alignment,
                addr_space,
                initializer_hex: initializer_hex.as_deref(),
                initializer_relocations: initializer_relocations.as_deref(),
                immutable,
            },
        )?
    };

    let address_of_op = llvm::AddressOfOp::new(ctx, global_name, addr_space);
    rewriter.insert_operation(ctx, address_of_op.get_operation());
    rewriter.replace_operation(ctx, op, address_of_op.get_operation());

    Ok(())
}

struct DeviceGlobalSpec<'a> {
    key: &'a str,
    mir_type: TypeHandle,
    alignment: u64,
    addr_space: u32,
    initializer_hex: Option<&'a str>,
    initializer_relocations: Option<&'a str>,
    /// Nothing writes this storage, so it exports as LLVM `constant`. Set only
    /// for the compiler's own promoted constants; see `MirGlobalAllocOp`.
    immutable: bool,
}

/// `next_device_global_index` is scoped to one `MirToLlvmConversionDriver`
/// instance (one module), not a process-global counter: `N` is a function of
/// this module's own MIR walk order, not of how many other modules have
/// lowered a device global earlier in the process (#706).
fn create_device_global(
    ctx: &mut Context,
    op: Ptr<Operation>,
    device_globals: &mut DeviceGlobalsMap,
    next_device_global_index: &mut usize,
    spec: DeviceGlobalSpec<'_>,
) -> Result<pliron::identifier::Identifier> {
    // An explicit initializer is already the evaluated Rust allocation image.
    // Pointer-free data stays `[N x i8]`. Initializers with relocations use a
    // segmented LLVM struct whose literal spans remain byte arrays and whose
    // pointer slots become pointer-width integers. This preserves both exact
    // bytes and linker-visible pointer provenance.
    let semantic_llvm_type = convert_type(ctx, spec.mir_type).map_err(anyhow_to_pliron)?;
    let (llvm_global_type, alignment) = if let Some(initializer_hex) = spec.initializer_hex {
        let byte_count = initializer_hex_byte_count(initializer_hex).map_err(anyhow_to_pliron)?;
        if spec.alignment == 0 {
            return Err(anyhow_to_pliron(anyhow::anyhow!(
                "device global initializer is missing its evaluated Rust allocation alignment"
            )));
        }
        let storage_type = if let Some(encoded) = spec.initializer_relocations {
            validate_relocated_initialized_global_layout(
                ctx,
                spec.mir_type,
                byte_count,
                spec.alignment,
            )
            .map_err(anyhow_to_pliron)?;
            relocated_initializer_storage_type(ctx, byte_count, spec.alignment, encoded)
                .map_err(anyhow_to_pliron)?
        } else {
            validate_initialized_global_layout(ctx, spec.mir_type, byte_count, spec.alignment)
                .map_err(anyhow_to_pliron)?;
            let i8_ty = IntegerType::get(ctx, 8, Signedness::Signless);
            ArrayType::get(ctx, i8_ty.into(), byte_count).into()
        };
        (storage_type, spec.alignment)
    } else {
        if spec.initializer_relocations.is_some() {
            return Err(anyhow_to_pliron(anyhow::anyhow!(
                "device global carries relocation metadata without initializer bytes"
            )));
        }
        (semantic_llvm_type, spec.alignment)
    };

    // Constant-memory globals reuse the Rust-side mangled name so host code can
    // resolve them by name via `cuModuleGetGlobal`. Ordinary device globals
    // are private to the kernel and get a counter-based unique name.
    let name: pliron::identifier::Identifier =
        if spec.addr_space == llvm_export::types::address_space::CONSTANT {
            spec.key.try_into().map_err(|e| {
                anyhow_to_pliron(anyhow::anyhow!(
                    "constant global_key {:?} is not a valid identifier: {e:?}",
                    spec.key
                ))
            })?
        } else {
            let counter = *next_device_global_index;
            *next_device_global_index += 1;
            format!("__device_global_{counter}").try_into().unwrap()
        };

    let global_op = if alignment > 0 {
        llvm::GlobalOp::new_with_alignment(ctx, name.clone(), llvm_global_type, alignment)
    } else {
        llvm::GlobalOp::new(ctx, name.clone(), llvm_global_type)
    };
    global_op.set_address_space(ctx, spec.addr_space);
    global_op.set_source_global_key(ctx, spec.key);
    if let Some(initializer_hex) = spec.initializer_hex {
        global_op.set_initializer_hex(ctx, initializer_hex);
    }
    if let Some(initializer_relocations) = spec.initializer_relocations {
        global_op.set_initializer_relocations(ctx, initializer_relocations);
    }
    if spec.immutable {
        global_op.mark_immutable(ctx);
    }

    let parent_block = op
        .deref(ctx)
        .get_parent_block()
        .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Op has no parent block")))?;
    let module_op = helpers::get_module_from_block(ctx, parent_block).map_err(anyhow_to_pliron)?;
    let region = module_op.deref(ctx).get_region(0);
    let module_block = region
        .deref(ctx)
        .iter(ctx)
        .next()
        .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Module is empty")))?;

    global_op.get_operation().insert_at_front(module_block, ctx);
    device_globals.insert(spec.key.to_string(), name.clone());

    Ok(name)
}

fn relocated_initializer_storage_type(
    ctx: &mut Context,
    byte_count: u64,
    allocation_alignment: u64,
    encoded: &str,
) -> std::result::Result<TypeHandle, anyhow::Error> {
    let mut relocations =
        llvm::decode_global_initializer_relocations(encoded).map_err(anyhow::Error::msg)?;
    if relocations.is_empty() {
        anyhow::bail!("device global relocation metadata contains no relocations");
    }
    relocations.sort_by_key(|relocation| relocation.source_offset);

    let mut cursor = 0u64;
    let mut fields = Vec::with_capacity(relocations.len() * 2 + 1);
    let mut requires_packed_storage = false;
    let i8_ty = IntegerType::get(ctx, 8, Signedness::Signless);

    for (index, relocation) in relocations.iter().enumerate() {
        if relocation.width_bytes != 8 {
            anyhow::bail!(
                "device global relocation {index} uses unsupported {}-byte pointer storage; CUDA global/constant pointers require 8 bytes",
                relocation.width_bytes
            );
        }
        if !matches!(relocation.target_address_space, 1 | 4) {
            anyhow::bail!(
                "device global relocation {index} targets unsupported CUDA address space {}",
                relocation.target_address_space
            );
        }
        if relocation.target_key.is_empty() {
            anyhow::bail!("device global relocation {index} has an empty target key");
        }

        let width = u64::from(relocation.width_bytes);
        requires_packed_storage |=
            allocation_alignment < width || !relocation.source_offset.is_multiple_of(width);
        if relocation.source_offset < cursor {
            anyhow::bail!(
                "device global relocation {index} overlaps the previous relocation or literal span"
            );
        }
        let end = relocation
            .source_offset
            .checked_add(width)
            .ok_or_else(|| anyhow::anyhow!("device global relocation {index} offset overflows"))?;
        if end > byte_count {
            anyhow::bail!(
                "device global relocation {index} occupies bytes {}..{} but the initializer is only {} bytes",
                relocation.source_offset,
                end,
                byte_count
            );
        }

        if relocation.source_offset > cursor {
            fields
                .push(ArrayType::get(ctx, i8_ty.into(), relocation.source_offset - cursor).into());
        }
        fields.push(IntegerType::get(ctx, relocation.width_bytes * 8, Signedness::Signless).into());
        cursor = end;
    }

    if cursor < byte_count {
        fields.push(ArrayType::get(ctx, i8_ty.into(), byte_count - cursor).into());
    }

    let layout = if requires_packed_storage {
        StructLayout::Packed
    } else {
        StructLayout::Unpacked
    };
    let storage: TypeHandle = StructType::get_unnamed(ctx, (fields, layout)).into();
    let lowered_size = get_type_size(ctx, storage);
    if lowered_size != byte_count {
        anyhow::bail!(
            "relocated device global storage lowers to {} bytes, but rustc evaluated {} bytes",
            lowered_size,
            byte_count
        );
    }
    Ok(storage)
}

fn initializer_hex_byte_count(hex: &str) -> std::result::Result<u64, anyhow::Error> {
    if !hex.len().is_multiple_of(2) {
        anyhow::bail!("device global initializer has an odd-length hex byte string");
    }
    if let Some(invalid) = hex.bytes().find(|byte| !byte.is_ascii_hexdigit()) {
        anyhow::bail!(
            "device global initializer contains invalid hex digit {:?}",
            invalid as char
        );
    }
    u64::try_from(hex.len() / 2)
        .map_err(|_| anyhow::anyhow!("device global initializer is too large for LLVM"))
}

/// Convert `mir.extern_shared` to LLVM extern global variable in shared address space.
///
/// Dynamic (extern) shared memory is represented as an external global variable
/// with address space 3 and zero-length array type `[0 x i8]`. The actual size
/// is determined at kernel launch via `LaunchConfig::shared_mem_bytes`.
///
/// # Per-Owner Symbols
///
/// Each function that owns an access gets a dynamic shared-memory symbol
/// (`__dynamic_smem_{function_name}`).
///
/// # Alignment
///
/// The alignment is pre-computed during the lowering pre-pass. It is the
/// maximum of the owner's body requirements and every launch-contract marker
/// that can reach it.
///
/// # Byte Offset
///
/// - `DynamicSharedArray::get()` / `get_raw()`: offset = 0, returns base pointer
/// - `DynamicSharedArray::offset(N)`: offset = N bytes, returns base + N
///
/// Called directly from `MirToLlvmConversionDriver::rewrite` (not through
/// op_cast dispatch) because it needs cross-function state maps.
pub fn convert_extern_shared_dc(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
    shared_globals: &mut SharedGlobalsMap,
    dynamic_smem_alignments: &mut DynamicSmemAlignmentMap,
) -> Result<()> {
    let (byte_offset, alignment) = {
        let extern_shared_op = dialect_mir::ops::MirExternSharedOp::new(op);
        let byte_offset = extern_shared_op.get_byte_offset_value(ctx);
        let alignment = extern_shared_op.get_alignment_value(ctx);
        (byte_offset, alignment)
    };

    let func_name: String = {
        let parent_block = op
            .deref(ctx)
            .get_parent_block()
            .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Op has no parent block")))?;
        let func_op_ptr = helpers::get_parent_func(ctx, parent_block).map_err(anyhow_to_pliron)?;
        let llvm_func = Operation::get_op::<llvm::FuncOp>(func_op_ptr, ctx)
            .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Parent op is not an llvm.func")))?;
        llvm_func.get_symbol_name(ctx).to_string()
    };

    let global_name = get_or_create_extern_shared_global(
        ctx,
        op,
        &func_name,
        shared_globals,
        dynamic_smem_alignments,
        alignment,
    )?;

    let address_of_op = llvm::AddressOfOp::new(ctx, global_name, 3);
    rewriter.insert_operation(ctx, address_of_op.get_operation());

    let base_ptr = address_of_op.get_operation().deref(ctx).get_result(0);

    if byte_offset > 0 {
        let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
        let offset_attr = pliron::builtin::attributes::IntegerAttr::new(
            i64_ty,
            pliron::utils::apint::APInt::from_u64(
                byte_offset,
                std::num::NonZeroUsize::new(64).unwrap(),
            ),
        );
        let offset_const = llvm::ConstantOp::new(ctx, offset_attr.into());
        rewriter.insert_operation(ctx, offset_const.get_operation());
        let offset_value = offset_const.get_operation().deref(ctx).get_result(0);

        let i8_ty = IntegerType::get(ctx, 8, Signedness::Signless);
        let gep_op = llvm::GetElementPtrOp::new(
            ctx,
            base_ptr,
            vec![llvm_export::ops::GepIndex::Value(offset_value)],
            i8_ty.into(),
        );
        rewriter.insert_operation(ctx, gep_op.get_operation());
        rewriter.replace_operation(ctx, op, gep_op.get_operation());
    } else {
        rewriter.replace_operation(ctx, op, address_of_op.get_operation());
    }

    Ok(())
}

/// Get or create the extern shared memory global for an owning function.
///
/// Creates an LLVM global variable with:
/// - Zero-length array type: `[0 x i8]`
/// - External linkage (no initializer)
/// - Address space 3 (shared memory)
/// - Pre-computed body and calling-kernel contract alignment
///
/// Each owning function gets its own dynamic shared memory symbol. Uses
/// `shared_globals` for deduplication (only one global per function).
fn get_or_create_extern_shared_global(
    ctx: &mut Context,
    op: Ptr<Operation>,
    func_name: &str,
    shared_globals: &mut SharedGlobalsMap,
    dynamic_smem_alignments: &mut DynamicSmemAlignmentMap,
    _requested_alignment: u64,
) -> Result<pliron::identifier::Identifier> {
    let (symbol_name, max_alignment) = dynamic_smem_alignments.get(func_name).cloned().ok_or_else(
        || {
            anyhow_to_pliron(anyhow::anyhow!(
                "Internal error: dynamic shared memory alignment not pre-computed for function '{}'. \
                 This should have been done in compute_max_dynamic_smem_alignment.",
                func_name
            ))
        },
    )?;

    let global_created_key = format!("__dynamic_smem_global_created_{}", func_name);
    if shared_globals.contains_key(&global_created_key) {
        return Ok(symbol_name);
    }

    let i8_ty = IntegerType::get(ctx, 8, Signedness::Signless);
    let array_type = ArrayType::get(ctx, i8_ty.into(), 0);

    let global_op = llvm::GlobalOp::new_with_alignment(
        ctx,
        symbol_name.clone(),
        array_type.into(),
        max_alignment,
    );
    global_op.set_address_space(ctx, llvm_export::types::address_space::SHARED);

    {
        use llvm_export::attributes::LinkageAttr;
        global_op.set_attr_llvm_global_linkage(ctx, LinkageAttr::ExternalLinkage);
    }

    let parent_block = op
        .deref(ctx)
        .get_parent_block()
        .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Op has no parent block")))?;
    let module_op = helpers::get_module_from_block(ctx, parent_block).map_err(anyhow_to_pliron)?;
    let region = module_op.deref(ctx).get_region(0);
    let module_block = region
        .deref(ctx)
        .iter(ctx)
        .next()
        .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("Module is empty")))?;

    global_op.get_operation().insert_at_front(module_block, ctx);

    shared_globals.insert(global_created_key, symbol_name.clone());

    Ok(symbol_name)
}

#[cfg(test)]
mod tests {
    //! End-to-end lowering tests for `dialect-mir` memory ops.
    //!
    //! The `convert_*` functions in this file take a live
    //! `DialectConversionRewriter`, which is owned by pliron's conversion
    //! driver and not constructible standalone. So each test builds a small
    //! MIR module, runs the full `lower_mir_to_llvm` pass on it, and asserts
    //! the lowered module contains the expected `dialect-llvm` shape — same
    //! pattern as `tests/lowering_test.rs`.

    use super::*;
    use crate::convert::ops::test_util::*;
    use dialect_mir::ops as mir;
    use dialect_mir::types::{MirArrayType, MirPtrType, MirStructType, MirTupleType, MirUnionType};
    use llvm_export::op_interfaces::PointerTypeResult;
    use llvm_export::ops as llvm;
    use llvm_export::types::{PointerType, StructType, address_space as llvm_addr};
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::{StringAttr, TypeAttr};
    use pliron::builtin::op_interfaces::SymbolOpInterface;
    use pliron::builtin::types::{IntegerType, Signedness};
    use pliron::context::Context;
    use pliron::linked_list::ContainsLinkedList;
    use pliron::location::{Location, Source};
    use pliron::op::Op;
    use pliron::operation::Operation;
    use std::path::PathBuf;

    fn ptr_addrspace(ctx: &Context, ty: TypeHandle) -> u32 {
        ty.deref(ctx)
            .downcast_ref::<PointerType>()
            .expect("expected llvm.PointerType")
            .address_space()
    }

    fn src_location(ctx: &mut Context, file: &str, line: i32, column: i32) -> Location {
        Location::SrcPos {
            src: Source::new_from_file(ctx, PathBuf::from(file)),
            pos: combine::stream::position::SourcePosition { line, column },
        }
    }

    fn over_aligned_tuple_ty(ctx: &mut Context) -> TypeHandle {
        let byte: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        let marker: TypeHandle = MirStructType::get_with_full_layout(
            ctx,
            "Align32".into(),
            vec![],
            vec![],
            vec![],
            vec![],
            0,
            32,
        )
        .into();
        MirTupleType::get_with_layout(ctx, vec![marker, byte], vec![0, 1], vec![0, 0], 32, 32)
            .into()
    }

    #[test]
    fn convert_alloca_lowers_to_llvm_alloca() {
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, true);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);

        let alloca_op = Operation::new(
            &mut ctx,
            mir::MirAllocaOp::get_concrete_op_info(),
            vec![mir_ptr_ty.into()],
            vec![],
            vec![],
            0,
        );
        alloca_op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        assert_eq!(
            count_ops::<llvm::AllocaOp>(&ctx, &body),
            1,
            "expected exactly one llvm.alloca"
        );
        let alloca = find_first::<llvm::AllocaOp>(&ctx, &body).unwrap();
        // Element type should round-trip through convert_type as i32.
        let elem_ty = alloca.result_pointee_type(&ctx);
        assert!(elem_ty.deref(&ctx).is::<IntegerType>());
    }

    #[test]
    fn convert_alloca_preserves_nested_array_element_alignment() {
        let mut ctx = make_ctx();
        let tuple_ty = over_aligned_tuple_ty(&mut ctx);
        let inner: TypeHandle = MirArrayType::get(&mut ctx, tuple_ty, 2).into();
        let outer: TypeHandle = MirArrayType::get(&mut ctx, inner, 3).into();
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, outer, true);
        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);

        let alloca_op = Operation::new(
            &mut ctx,
            mir::MirAllocaOp::get_concrete_op_info(),
            vec![mir_ptr_ty.into()],
            vec![],
            vec![],
            0,
        );
        alloca_op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let alloca = find_first::<llvm::AllocaOp>(&ctx, &body).expect("expected llvm.alloca");
        assert_eq!(
            llvm_export::ops::op_alignment(&ctx, alloca.get_operation()),
            Some(32)
        );
    }

    #[test]
    fn convert_alloca_preserves_debug_local_metadata() {
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, true);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);

        let alloca_op = Operation::new(
            &mut ctx,
            mir::MirAllocaOp::get_concrete_op_info(),
            vec![mir_ptr_ty.into()],
            vec![],
            vec![],
            0,
        );
        llvm::set_debug_local_variable(
            &mut ctx,
            alloca_op,
            llvm::DebugLocalVariableInfo {
                name: "x".to_string(),
                argument_index: Some(1),
                ty: llvm::DebugLocalTypeKind::Basic {
                    name: "i32".to_string(),
                    size_bits: 32,
                    encoding: "DW_ATE_signed",
                },
            },
        );
        alloca_op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let alloca = find_first::<llvm::AllocaOp>(&ctx, &body).unwrap();
        let info = llvm::debug_local_variable(&ctx, alloca.get_operation())
            .expect("debug local metadata should survive lowering");

        assert_eq!(info.name, "x");
        assert_eq!(info.argument_index, Some(1));
        assert_eq!(
            info.ty,
            llvm::DebugLocalTypeKind::Basic {
                name: "i32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_signed",
            }
        );
    }

    #[test]
    fn convert_store_lowers_to_llvm_store() {
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, true);

        // Kernel takes (ptr, val) so we can store one into the other.
        let (module_ptr, block) = build_kernel(&mut ctx, vec![mir_ptr_ty.into(), i32_ty], vec![]);
        let ptr_val = block.deref(&ctx).get_argument(0);
        let val = block.deref(&ctx).get_argument(1);

        let store_op = Operation::new(
            &mut ctx,
            mir::MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![ptr_val, val],
            vec![],
            0,
        );
        store_op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        assert_eq!(
            count_ops::<llvm::StoreOp>(&ctx, &body),
            1,
            "expected one llvm.store"
        );
        // The original mir.store must be gone.
        assert_eq!(count_ops::<mir::MirStoreOp>(&ctx, &body), 0);

        // convert_store swaps operand order: mir.store is [ptr, value] but
        // llvm.store takes (value, ptr). Verify that mapping survived.
        let store = find_first::<llvm::StoreOp>(&ctx, &body).unwrap();
        let addr_ty = store.get_operand_address(&ctx).get_type(&ctx);
        assert!(addr_ty.deref(&ctx).is::<PointerType>(), "operand 1 is ptr");
    }

    #[test]
    fn convert_store_preserves_volatile() {
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, true);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![mir_ptr_ty.into(), i32_ty], vec![]);
        let ptr_val = block.deref(&ctx).get_argument(0);
        let val = block.deref(&ctx).get_argument(1);

        let store_op = Operation::new(
            &mut ctx,
            mir::MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![ptr_val, val],
            vec![],
            0,
        );
        mir::MirStoreOp::new(store_op).set_volatile(&mut ctx, true);
        store_op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let store = find_first::<llvm::StoreOp>(&ctx, &body).unwrap();
        assert!(
            llvm_export::ops::op_volatile(&ctx, store.get_operation()),
            "volatile mir.store must lower to a volatile llvm.store"
        );
    }

    #[test]
    fn convert_load_lowers_to_llvm_load() {
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, false);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![mir_ptr_ty.into()], vec![]);
        let ptr_val = block.deref(&ctx).get_argument(0);

        let load_op = Operation::new(
            &mut ctx,
            mir::MirLoadOp::get_concrete_op_info(),
            vec![i32_ty],
            vec![ptr_val],
            vec![],
            0,
        );
        load_op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        assert_eq!(count_ops::<llvm::LoadOp>(&ctx, &body), 1);
        assert_eq!(count_ops::<mir::MirLoadOp>(&ctx, &body), 0);
    }

    /// Lower `mir.load (mir.field_addr %p, field_index)` for a struct of
    /// signless integer fields with the given layout and report the alignment
    /// stamped on the resulting `llvm.load`. `None` means no stamp survived
    /// and the exporter's natural-alignment default applies.
    fn lowered_field_load_alignment(
        field_bit_widths: Vec<u32>,
        field_offsets: Vec<u64>,
        total_size: u64,
        abi_align: u64,
        field_index: u32,
    ) -> Option<u32> {
        use dialect_mir::attributes::FieldIndexAttr;

        let mut ctx = make_ctx();
        let field_types: Vec<TypeHandle> = field_bit_widths
            .iter()
            .map(|w| IntegerType::get(&ctx, *w, Signedness::Signless).into())
            .collect();
        let field_names = (0..field_types.len()).map(|i| format!("f{i}")).collect();
        let struct_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "FieldLoadAlign".into(),
            field_names,
            field_types.clone(),
            vec![],
            field_offsets,
            total_size,
            abi_align,
        )
        .into();
        let struct_ptr_ty = MirPtrType::get_generic(&mut ctx, struct_ty, false);
        let field_ty = field_types[field_index as usize];
        let field_ptr_ty = MirPtrType::get_generic(&mut ctx, field_ty, false);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![struct_ptr_ty.into()], vec![]);
        let struct_ptr_val = block.deref(&ctx).get_argument(0);

        let field_addr_op = Operation::new(
            &mut ctx,
            mir::MirFieldAddrOp::get_concrete_op_info(),
            vec![field_ptr_ty.into()],
            vec![struct_ptr_val],
            vec![],
            0,
        );
        mir::MirFieldAddrOp::new(field_addr_op)
            .set_attr_field_index(&ctx, FieldIndexAttr(field_index));
        field_addr_op.insert_at_back(block, &ctx);
        let field_ptr_val = field_addr_op.deref(&ctx).get_result(0);

        let load_op = Operation::new(
            &mut ctx,
            mir::MirLoadOp::get_concrete_op_info(),
            vec![field_ty],
            vec![field_ptr_val],
            vec![],
            0,
        );
        load_op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let load = find_first::<llvm::LoadOp>(&ctx, &body).expect("expected one llvm.load");
        llvm_export::ops::op_alignment(&ctx, load.get_operation())
    }

    /// Field 0 of an over-aligned struct sits at the aggregate's own
    /// alignment, which the field's scalar result type cannot state on its
    /// own. This is what lets LoadStoreVectorizer fuse the adjacent pair.
    #[test]
    fn convert_load_inherits_overaligned_field_alignment_at_offset_zero() {
        // #[repr(C, align(8))] struct { a: i32, b: i32 }
        assert_eq!(
            lowered_field_load_alignment(vec![32, 32], vec![0, 4], 8, 8, 0),
            Some(8)
        );
    }

    /// A field at a nonzero offset proves `gcd(abi_align, offset)`: an i32 at
    /// offset 8 of an align-16 struct proves 8, beating its natural 4.
    #[test]
    fn convert_load_narrows_field_alignment_to_gcd_of_align_and_offset() {
        // #[repr(C, align(16))] struct { a: i64, b: i32 }
        assert_eq!(
            lowered_field_load_alignment(vec![64, 32], vec![0, 8], 16, 16, 1),
            Some(8)
        );
    }

    /// Whole-value loads of pointer-free packed structs use the packed LLVM
    /// representation, preserving rustc's byte size and field offsets.
    #[test]
    fn packed_struct_whole_value_load_uses_packed_layout() {
        let mut ctx = make_ctx();
        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let packed_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Packed".into(),
            vec!["tag".into(), "value".into()],
            vec![u8_ty, u32_ty],
            vec![0, 1],
            vec![0, 1],
            5,
            1,
        )
        .into();
        let ptr_ty = MirPtrType::get_generic(&mut ctx, packed_ty, false);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![ptr_ty.into()], vec![]);
        let ptr_val = block.deref(&ctx).get_argument(0);

        let load_op = Operation::new(
            &mut ctx,
            mir::MirLoadOp::get_concrete_op_info(),
            vec![packed_ty],
            vec![ptr_val],
            vec![],
            0,
        );
        load_op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr)
            .expect("whole-value load of a pointer-free packed struct must lower");

        let body = kernel_blocks(&ctx, module_ptr);
        let load = find_first::<llvm::LoadOp>(&ctx, &body).expect("expected packed llvm.load");
        let result_ty = load
            .get_operation()
            .deref(&ctx)
            .get_result(0)
            .get_type(&ctx);
        let result_ty_ref = result_ty.deref(&ctx);
        let struct_ty = result_ty_ref
            .downcast_ref::<StructType>()
            .expect("packed load result must be an LLVM struct");
        assert_eq!(struct_ty.layout(), StructLayout::Packed);
        assert_eq!(
            crate::convert::types::llvm_type_size_align(&ctx, result_ty),
            Some((5, 1))
        );
        assert_eq!(
            llvm_export::ops::op_alignment(&ctx, load.get_operation()),
            Some(1)
        );
    }

    #[test]
    fn packed_struct_whole_value_load_with_shared_pointer_fails_closed() {
        let mut ctx = make_ctx();
        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let pointee: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let shared_ty: TypeHandle = MirPtrType::get_shared(&mut ctx, pointee, false).into();
        let packed_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedShared".into(),
            vec!["tag".into(), "ptr".into()],
            vec![u8_ty, shared_ty],
            vec![0, 1],
            vec![0, 1],
            9,
            1,
        )
        .into();
        let ptr_ty = MirPtrType::get_generic(&mut ctx, packed_ty, false);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![ptr_ty.into()], vec![]);
        let ptr_val = block.deref(&ctx).get_argument(0);
        let load_op = Operation::new(
            &mut ctx,
            mir::MirLoadOp::get_concrete_op_info(),
            vec![packed_ty],
            vec![ptr_val],
            vec![],
            0,
        );
        load_op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        let err = crate::lower_mir_to_llvm(&mut ctx, module_ptr)
            .expect_err("packed whole-value load containing AS3 must remain fail-closed");
        assert!(
            format!("{err:?}").contains("target-mode dependent"),
            "the refusal must identify the target-dependent packed AS3 image: {err:?}"
        );
    }

    /// Whole-value stores use the same packed representation as construction
    /// and loads, while preserving the MIR aggregate's proved ABI alignment.
    #[test]
    fn packed_struct_whole_value_store_uses_packed_layout() {
        let mut ctx = make_ctx();
        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let packed_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Packed".into(),
            vec!["tag".into(), "value".into()],
            vec![u8_ty, u32_ty],
            vec![0, 1],
            vec![0, 1],
            5,
            1,
        )
        .into();
        let ptr_ty = MirPtrType::get_generic(&mut ctx, packed_ty, false);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![ptr_ty.into(), packed_ty], vec![]);
        let ptr_val = block.deref(&ctx).get_argument(0);
        let val = block.deref(&ctx).get_argument(1);

        let store_op = Operation::new(
            &mut ctx,
            mir::MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![ptr_val, val],
            vec![],
            0,
        );
        store_op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr)
            .expect("whole-value store of a pointer-free packed struct must lower");

        let body = kernel_blocks(&ctx, module_ptr);
        let store = find_first::<llvm::StoreOp>(&ctx, &body).expect("expected packed llvm.store");
        let value_ty = store
            .get_operation()
            .deref(&ctx)
            .get_operand(0)
            .get_type(&ctx);
        let value_ty_ref = value_ty.deref(&ctx);
        let struct_ty = value_ty_ref
            .downcast_ref::<StructType>()
            .expect("packed store value must be an LLVM struct");
        assert_eq!(struct_ty.layout(), StructLayout::Packed);
        assert_eq!(
            crate::convert::types::llvm_type_size_align(&ctx, value_ty),
            Some((5, 1))
        );
        assert_eq!(
            llvm_export::ops::op_alignment(&ctx, store.get_operation()),
            Some(1)
        );
    }

    /// A naturally aligned inner struct sitting at a packed byte offset: the
    /// field address only proves align 1, and the load must claim that over
    /// the inner type's recorded abi alignment. Claiming the abi would stamp
    /// `align 4` on a 1-aligned address, which llc may honor with a wider
    /// access than the bytes allow.
    #[test]
    fn convert_load_claims_address_alignment_over_abi_at_packed_offsets() {
        use dialect_mir::attributes::FieldIndexAttr;

        let mut ctx = make_ctx();
        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
        let inner_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Inner".into(),
            vec!["v".into()],
            vec![u32_ty],
            vec![0],
            vec![0],
            4,
            4,
        )
        .into();
        let outer_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PackedOuter".into(),
            vec!["tag".into(), "inner".into()],
            vec![u8_ty, inner_ty],
            vec![0, 1],
            vec![0, 1],
            5,
            1,
        )
        .into();
        let outer_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, outer_ty, false).into();
        let inner_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, inner_ty, false).into();

        let (module_ptr, block) = build_kernel(&mut ctx, vec![outer_ptr_ty], vec![]);
        let base = block.deref(&ctx).get_argument(0);

        let field_addr_op = Operation::new(
            &mut ctx,
            mir::MirFieldAddrOp::get_concrete_op_info(),
            vec![inner_ptr_ty],
            vec![base],
            vec![],
            0,
        );
        mir::MirFieldAddrOp::new(field_addr_op).set_attr_field_index(&ctx, FieldIndexAttr(1));
        field_addr_op.insert_at_back(block, &ctx);
        let field_ptr_val = field_addr_op.deref(&ctx).get_result(0);

        let load_op = Operation::new(
            &mut ctx,
            mir::MirLoadOp::get_concrete_op_info(),
            vec![inner_ty],
            vec![field_ptr_val],
            vec![],
            0,
        );
        load_op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let load = find_first::<llvm::LoadOp>(&ctx, &body).expect("expected one llvm.load");
        assert_eq!(
            llvm_export::ops::op_alignment(&ctx, load.get_operation()),
            Some(1),
            "the load must claim the address's proved alignment, not the inner abi"
        );
    }

    /// A struct with no extra alignment proves nothing beyond the scalar's
    /// natural alignment: the stamp equals the exporter's default 4, so the
    /// emitted access is unchanged.
    #[test]
    fn convert_load_keeps_natural_alignment_without_overalignment() {
        // struct { a: i32, b: i32 } with rustc's natural abi_align 4
        assert_eq!(
            lowered_field_load_alignment(vec![32, 32], vec![0, 4], 8, 4, 1),
            Some(4)
        );
    }

    /// dialect-mir only verifier-enforces power-of-two alignment for unions
    /// and enums. A malformed hand-built struct layout must decline the stamp
    /// rather than emit a non-power-of-two `align N` that llc rejects.
    #[test]
    fn convert_load_declines_non_power_of_two_field_alignment() {
        assert_eq!(
            lowered_field_load_alignment(vec![32], vec![0], 12, 12, 0),
            None
        );
    }

    /// Lower `mir.store %v, (mir.field_addr %p, field_index)` for a struct of
    /// signless integer fields with the given layout and report the alignment
    /// stamped on the resulting `llvm.store`. `None` means no stamp survived
    /// and the exporter's natural-alignment default applies.
    fn lowered_field_store_alignment(
        field_bit_widths: Vec<u32>,
        field_offsets: Vec<u64>,
        total_size: u64,
        abi_align: u64,
        field_index: u32,
    ) -> Option<u32> {
        use dialect_mir::attributes::FieldIndexAttr;

        let mut ctx = make_ctx();
        let field_types: Vec<TypeHandle> = field_bit_widths
            .iter()
            .map(|w| IntegerType::get(&ctx, *w, Signedness::Signless).into())
            .collect();
        let field_names = (0..field_types.len()).map(|i| format!("f{i}")).collect();
        let struct_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "FieldStoreAlign".into(),
            field_names,
            field_types.clone(),
            vec![],
            field_offsets,
            total_size,
            abi_align,
        )
        .into();
        let struct_ptr_ty = MirPtrType::get_generic(&mut ctx, struct_ty, true);
        let field_ty = field_types[field_index as usize];
        let field_ptr_ty = MirPtrType::get_generic(&mut ctx, field_ty, true);

        // The stored value arrives as a kernel argument of the field's own
        // scalar type, so `value_abi_align` reports nothing about it and the
        // address's stamp is the only alignment left -- the case this covers.
        let (module_ptr, block) =
            build_kernel(&mut ctx, vec![struct_ptr_ty.into(), field_ty], vec![]);
        let struct_ptr_val = block.deref(&ctx).get_argument(0);
        let val = block.deref(&ctx).get_argument(1);

        let field_addr_op = Operation::new(
            &mut ctx,
            mir::MirFieldAddrOp::get_concrete_op_info(),
            vec![field_ptr_ty.into()],
            vec![struct_ptr_val],
            vec![],
            0,
        );
        mir::MirFieldAddrOp::new(field_addr_op)
            .set_attr_field_index(&ctx, FieldIndexAttr(field_index));
        field_addr_op.insert_at_back(block, &ctx);
        let field_ptr_val = field_addr_op.deref(&ctx).get_result(0);

        let store_op = Operation::new(
            &mut ctx,
            mir::MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![field_ptr_val, val],
            vec![],
            0,
        );
        store_op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let store = find_first::<llvm::StoreOp>(&ctx, &body).expect("expected one llvm.store");
        llvm_export::ops::op_alignment(&ctx, store.get_operation())
    }

    /// Field 0 of an over-aligned struct sits at the aggregate's own alignment,
    /// which the stored scalar's type cannot state. This is what lets
    /// LoadStoreVectorizer fuse the adjacent pair into one wide store.
    #[test]
    fn convert_store_inherits_overaligned_field_alignment_at_offset_zero() {
        // #[repr(C, align(8))] struct { a: i32, b: i32 }
        assert_eq!(
            lowered_field_store_alignment(vec![32, 32], vec![0, 4], 8, 8, 0),
            Some(8)
        );
    }

    /// A field at a nonzero offset proves `gcd(abi_align, offset)`: an i32 at
    /// offset 8 of an align-16 struct proves 8, beating its natural 4.
    #[test]
    fn convert_store_narrows_field_alignment_to_gcd_of_align_and_offset() {
        // #[repr(C, align(16))] struct { a: i64, b: i32 }
        assert_eq!(
            lowered_field_store_alignment(vec![64, 32], vec![0, 8], 16, 16, 1),
            Some(8)
        );
    }

    /// A struct with no extra alignment proves nothing beyond the scalar's
    /// natural alignment, so the emitted store is unchanged. Widening here
    /// would claim an alignment the source never guaranteed.
    #[test]
    fn convert_store_keeps_natural_alignment_without_overalignment() {
        // struct { a: i32, b: i32 } with rustc's natural abi_align 4
        assert_eq!(
            lowered_field_store_alignment(vec![32, 32], vec![0, 4], 8, 4, 1),
            Some(4)
        );
    }

    /// A malformed hand-built layout must decline the stamp rather than emit a
    /// non-power-of-two `align N` that llc rejects. Same guard the load path
    /// has, and it matters more here: an over-aligned store instruction on an
    /// under-aligned address is undefined, not merely slow.
    #[test]
    fn convert_store_declines_non_power_of_two_field_alignment() {
        assert_eq!(
            lowered_field_store_alignment(vec![32], vec![0], 12, 12, 0),
            None
        );
    }

    /// Lower `load (&arr[index])` where `arr` is the array field of an
    /// over-aligned struct, and report the alignment the load ends up with.
    ///
    /// `index` of `Some(i)` builds a constant index, `None` a runtime one.
    fn lowered_element_load_alignment(
        element_bits: u32,
        element_count: u64,
        struct_abi_align: u64,
        index: Option<u64>,
    ) -> Option<u32> {
        use dialect_mir::attributes::FieldIndexAttr;
        use pliron::builtin::attributes::IntegerAttr;
        use std::num::NonZeroUsize;

        let mut ctx = make_ctx();
        let element_ty: TypeHandle =
            IntegerType::get(&ctx, element_bits, Signedness::Signless).into();
        let array_ty: TypeHandle =
            dialect_mir::types::MirArrayType::get(&mut ctx, element_ty, element_count).into();
        let elem_bytes = u64::from(element_bits) / 8;
        let struct_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "ElementLoadAlign".into(),
            vec!["lanes".into()],
            vec![array_ty],
            vec![],
            vec![0],
            elem_bytes * element_count,
            struct_abi_align,
        )
        .into();
        let struct_ptr_ty = MirPtrType::get_generic(&mut ctx, struct_ty, false);
        let array_ptr_ty = MirPtrType::get_generic(&mut ctx, array_ty, false);
        let element_ptr_ty = MirPtrType::get_generic(&mut ctx, element_ty, false);
        let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signed).into();

        let (module_ptr, block) =
            build_kernel(&mut ctx, vec![struct_ptr_ty.into(), i64_ty], vec![]);
        let struct_ptr_val = block.deref(&ctx).get_argument(0);

        // &s.lanes -- carries the struct's alignment onto the array address.
        let field_addr_op = Operation::new(
            &mut ctx,
            mir::MirFieldAddrOp::get_concrete_op_info(),
            vec![array_ptr_ty.into()],
            vec![struct_ptr_val],
            vec![],
            0,
        );
        mir::MirFieldAddrOp::new(field_addr_op).set_attr_field_index(&ctx, FieldIndexAttr(0));
        field_addr_op.insert_at_back(block, &ctx);
        let array_ptr_val = field_addr_op.deref(&ctx).get_result(0);

        let index_val = match index {
            Some(i) => {
                let constant = Operation::new(
                    &mut ctx,
                    mir::MirConstantOp::get_concrete_op_info(),
                    vec![i64_ty],
                    vec![],
                    vec![],
                    0,
                );
                mir::MirConstantOp::new(constant).set_attr_value(
                    &ctx,
                    IntegerAttr::new(
                        IntegerType::get(&ctx, 64, Signedness::Signed),
                        APInt::from_u64(i, NonZeroUsize::new(64).unwrap()),
                    ),
                );
                constant.insert_at_back(block, &ctx);
                constant.deref(&ctx).get_result(0)
            }
            None => block.deref(&ctx).get_argument(1),
        };

        let elem_addr_op = Operation::new(
            &mut ctx,
            mir::MirArrayElementAddrOp::get_concrete_op_info(),
            vec![element_ptr_ty.into()],
            vec![array_ptr_val, index_val],
            vec![],
            0,
        );
        elem_addr_op.insert_at_back(block, &ctx);
        let elem_ptr_val = elem_addr_op.deref(&ctx).get_result(0);

        let load_op = Operation::new(
            &mut ctx,
            mir::MirLoadOp::get_concrete_op_info(),
            vec![element_ty],
            vec![elem_ptr_val],
            vec![],
            0,
        );
        load_op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let load = find_first::<llvm::LoadOp>(&ctx, &body).expect("expected one llvm.load");
        llvm_export::ops::op_alignment(&ctx, load.get_operation())
    }

    /// Element 0 inherits the whole alignment the base address proved, which is
    /// what lets the adjacent pair fuse into one wide load.
    #[test]
    fn convert_load_inherits_base_alignment_at_element_zero() {
        // &(#[repr(C, align(8))] struct { lanes: [i32; 2] }).lanes[0]
        assert_eq!(lowered_element_load_alignment(32, 2, 8, Some(0)), Some(8));
    }

    /// A nonzero constant index proves `gcd(base, i * stride)`: element 1 of an
    /// align-8 `[i32; 2]` sits at byte 4, so it proves 4, not 8.
    #[test]
    fn convert_load_narrows_element_alignment_to_gcd_with_offset() {
        assert_eq!(lowered_element_load_alignment(32, 2, 8, Some(1)), Some(4));
    }

    /// A runtime index can land on any element, so only what every stride
    /// preserves may be claimed -- `gcd(base, stride)`, never the base itself.
    #[test]
    fn convert_load_claims_only_stride_alignment_for_a_runtime_index() {
        assert_eq!(lowered_element_load_alignment(32, 2, 8, None), Some(4));
    }

    /// A base with no extra alignment proves nothing beyond the element's own
    /// natural alignment, so the emitted access is unchanged.
    #[test]
    fn convert_load_keeps_natural_element_alignment_without_overalignment() {
        assert_eq!(lowered_element_load_alignment(32, 2, 4, Some(0)), Some(4));
    }

    /// Like [`lowered_element_load_alignment`], but for an array whose element
    /// is an aggregate built by `element_ty_of` (which also reports the
    /// element's stored size in bytes). The claim on the element address must
    /// then come from the element's *exact* stride — rustc's stored size,
    /// padding included — not from any LLVM-level approximation.
    ///
    /// With `load_first_scalar` the element must itself be an array and the
    /// access becomes `s.lanes[index][0]`, mirroring the nested-read chain
    /// where the outer stamp is inherited by the inner index-0 address and
    /// ends up on a scalar load that LoadStoreVectorizer trusts. Without it,
    /// the element itself is loaded.
    ///
    /// Reports `(element address stamp, final load alignment)`.
    fn lowered_aggregate_element_alignments(
        element_ty_of: impl FnOnce(&mut Context) -> (TypeHandle, u64),
        element_count: u64,
        struct_abi_align: u64,
        index: Option<u64>,
        load_first_scalar: bool,
    ) -> (Option<u32>, Option<u32>) {
        use dialect_mir::attributes::FieldIndexAttr;
        use pliron::builtin::attributes::IntegerAttr;
        use std::num::NonZeroUsize;

        let mut ctx = make_ctx();
        let (element_ty, elem_bytes) = element_ty_of(&mut ctx);
        let inner_scalar_ty = load_first_scalar.then(|| {
            let element_ref = element_ty.deref(&ctx);
            element_ref
                .downcast_ref::<MirArrayType>()
                .expect("load_first_scalar needs an array element")
                .element_type()
        });
        let array_ty: TypeHandle = MirArrayType::get(&mut ctx, element_ty, element_count).into();
        let struct_ty: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "AggregateElementAlign".into(),
            vec!["lanes".into()],
            vec![array_ty],
            vec![],
            vec![0],
            elem_bytes * element_count,
            struct_abi_align,
        )
        .into();
        let struct_ptr_ty = MirPtrType::get_generic(&mut ctx, struct_ty, false);
        let array_ptr_ty = MirPtrType::get_generic(&mut ctx, array_ty, false);
        let element_ptr_ty = MirPtrType::get_generic(&mut ctx, element_ty, false);
        let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signed).into();

        let (module_ptr, block) =
            build_kernel(&mut ctx, vec![struct_ptr_ty.into(), i64_ty], vec![]);
        let struct_ptr_val = block.deref(&ctx).get_argument(0);

        // &s.lanes -- carries the struct's alignment onto the array address.
        let field_addr_op = Operation::new(
            &mut ctx,
            mir::MirFieldAddrOp::get_concrete_op_info(),
            vec![array_ptr_ty.into()],
            vec![struct_ptr_val],
            vec![],
            0,
        );
        mir::MirFieldAddrOp::new(field_addr_op).set_attr_field_index(&ctx, FieldIndexAttr(0));
        field_addr_op.insert_at_back(block, &ctx);
        let array_ptr_val = field_addr_op.deref(&ctx).get_result(0);

        let constant_index = |ctx: &mut Context, i: u64| {
            let constant = Operation::new(
                ctx,
                mir::MirConstantOp::get_concrete_op_info(),
                vec![i64_ty],
                vec![],
                vec![],
                0,
            );
            mir::MirConstantOp::new(constant).set_attr_value(
                ctx,
                IntegerAttr::new(
                    IntegerType::get(ctx, 64, Signedness::Signed),
                    APInt::from_u64(i, NonZeroUsize::new(64).unwrap()),
                ),
            );
            constant.insert_at_back(block, ctx);
            constant.deref(ctx).get_result(0)
        };

        let index_val = match index {
            Some(i) => constant_index(&mut ctx, i),
            None => block.deref(&ctx).get_argument(1),
        };

        let elem_addr_op = Operation::new(
            &mut ctx,
            mir::MirArrayElementAddrOp::get_concrete_op_info(),
            vec![element_ptr_ty.into()],
            vec![array_ptr_val, index_val],
            vec![],
            0,
        );
        elem_addr_op.insert_at_back(block, &ctx);
        let elem_ptr_val = elem_addr_op.deref(&ctx).get_result(0);

        let (loaded_ty, loaded_ptr_val) = match inner_scalar_ty {
            Some(scalar_ty) => {
                let zero_val = constant_index(&mut ctx, 0);
                let scalar_ptr_ty = MirPtrType::get_generic(&mut ctx, scalar_ty, false);
                let inner_addr_op = Operation::new(
                    &mut ctx,
                    mir::MirArrayElementAddrOp::get_concrete_op_info(),
                    vec![scalar_ptr_ty.into()],
                    vec![elem_ptr_val, zero_val],
                    vec![],
                    0,
                );
                inner_addr_op.insert_at_back(block, &ctx);
                (scalar_ty, inner_addr_op.deref(&ctx).get_result(0))
            }
            None => (element_ty, elem_ptr_val),
        };

        let load_op = Operation::new(
            &mut ctx,
            mir::MirLoadOp::get_concrete_op_info(),
            vec![loaded_ty],
            vec![loaded_ptr_val],
            vec![],
            0,
        );
        load_op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        // GEP order follows source order: field address, then the element
        // address under test (then the inner index-0 address when nested).
        let geps = find_all::<llvm::GetElementPtrOp>(&ctx, &body);
        assert_eq!(geps.len(), if load_first_scalar { 3 } else { 2 });
        let element_gep_align = llvm_export::ops::address_alignment(&ctx, geps[1].get_operation());
        let load = find_first::<llvm::LoadOp>(&ctx, &body).expect("expected one llvm.load");
        let load_align = llvm_export::ops::op_alignment(&ctx, load.get_operation());
        (element_gep_align, load_align)
    }

    /// `[f32; 3]` element under an align-8 base: stride is 12, so a runtime
    /// index proves `gcd(8, 12) = 4` — and the inner index-0 scalar read
    /// inherits exactly that. Guards against sizing the element through an
    /// LLVM-level approximation, whose guessed stride of 8 would stamp
    /// align 8 onto addresses that are only 4-aligned (a miscompile once
    /// LoadStoreVectorizer trusts it).
    #[test]
    fn convert_load_claims_exact_aggregate_stride_for_nested_array_elements() {
        use pliron::builtin::types::FP32Type;
        // &(#[repr(C, align(8))] struct { lanes: [[f32; 3]; 4] }).lanes[i][0]
        let nested_f32x3 = |ctx: &mut Context| {
            let f32_ty: TypeHandle = FP32Type::get(ctx).into();
            (MirArrayType::get(ctx, f32_ty, 3).into(), 12)
        };
        assert_eq!(
            lowered_aggregate_element_alignments(nested_f32x3, 4, 8, None, true),
            (Some(4), Some(4))
        );
    }

    /// A constant index into the same nested array uses the exact byte
    /// offset: element 1 sits at byte 12 (`gcd(8, 12) = 4`), element 2 at
    /// byte 24 (`gcd(8, 24) = 8`, the full base alignment again).
    #[test]
    fn convert_load_narrows_nested_element_alignment_by_exact_byte_offset() {
        use pliron::builtin::types::FP32Type;
        let nested_f32x3 = |ctx: &mut Context| {
            let f32_ty: TypeHandle = FP32Type::get(ctx).into();
            (MirArrayType::get(ctx, f32_ty, 3).into(), 12)
        };
        assert_eq!(
            lowered_aggregate_element_alignments(nested_f32x3, 4, 8, Some(1), true),
            (Some(4), Some(4))
        );
        assert_eq!(
            lowered_aggregate_element_alignments(nested_f32x3, 4, 8, Some(2), true),
            (Some(8), Some(8))
        );
    }

    /// A tuple element's stride comes from rustc's recorded `total_size`
    /// (trailing padding included): `(f32, f32, f32)` stores 12 bytes, so an
    /// align-8 base proves only 4 on a runtime element address.
    #[test]
    fn convert_array_element_addr_takes_tuple_stride_from_recorded_layout() {
        use pliron::builtin::types::FP32Type;
        let f32x3_tuple = |ctx: &mut Context| {
            let f32_ty: TypeHandle = FP32Type::get(ctx).into();
            let tuple_ty: TypeHandle =
                MirTupleType::get_with_layout(ctx, vec![f32_ty; 3], vec![], vec![0, 4, 8], 12, 4)
                    .into();
            (tuple_ty, 12)
        };
        let (element_gep_align, _load_align) =
            lowered_aggregate_element_alignments(f32x3_tuple, 4, 8, None, false);
        assert_eq!(element_gep_align, Some(4));
    }

    /// `f16` arrives from the importer as `MirFP16Type`, not the converted
    /// LLVM `half`, and its stride is exactly 2: an align-8 base proves 2 on
    /// a runtime element address and `gcd(8, 4) = 4` at element 2. Guards
    /// the arm the importer actually exercises — the old sizing guessed 8
    /// for this type too, the same over-claim as the aggregate cases.
    #[test]
    fn convert_load_claims_exact_f16_element_stride() {
        let f16_scalar = |ctx: &mut Context| {
            let f16_ty: TypeHandle = dialect_mir::types::MirFP16Type::get(ctx).into();
            (f16_ty, 2)
        };
        assert_eq!(
            lowered_aggregate_element_alignments(f16_scalar, 4, 8, None, false),
            (Some(2), Some(2))
        );
        assert_eq!(
            lowered_aggregate_element_alignments(f16_scalar, 4, 8, Some(2), false),
            (Some(4), Some(4))
        );
    }

    /// An element whose stored size is unknown (a struct built without rustc
    /// layout) must not have its stride guessed: the element address claims
    /// nothing and the load keeps the previous, weaker-but-sound behaviour.
    #[test]
    fn convert_array_element_addr_declines_unknown_element_stride() {
        use pliron::builtin::types::FP32Type;
        let opaque_struct = |ctx: &mut Context| {
            let f32_ty: TypeHandle = FP32Type::get(ctx).into();
            let struct_ty: TypeHandle =
                MirStructType::get(ctx, "OpaqueElement".into(), vec!["x".into()], vec![f32_ty])
                    .into();
            (struct_ty, 4)
        };
        assert_eq!(
            lowered_aggregate_element_alignments(opaque_struct, 4, 8, None, false),
            (None, None)
        );
    }

    #[test]
    fn convert_dbg_value_lowers_to_llvm_dbg_value() {
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();

        let (module_ptr, block) = build_kernel(&mut ctx, vec![i32_ty], vec![]);
        let value = block.deref(&ctx).get_argument(0);

        let dbg_op = mir::MirDbgValueOp::new(&mut ctx, value);
        let dbg_loc = pliron::location::Location::Named {
            name: "current value location".to_string(),
            child_loc: Box::new(pliron::location::Location::Unknown),
        };
        dbg_op
            .get_operation()
            .deref_mut(&ctx)
            .set_loc(dbg_loc.clone());
        llvm::set_debug_local_variable(
            &mut ctx,
            dbg_op.get_operation(),
            llvm::DebugLocalVariableInfo {
                name: "x".to_string(),
                argument_index: None,
                ty: llvm::DebugLocalTypeKind::Basic {
                    name: "i32".to_string(),
                    size_bits: 32,
                    encoding: "DW_ATE_signed",
                },
            },
        );
        llvm::set_debug_local_source_scope(&mut ctx, dbg_op.get_operation(), 42);
        llvm::set_debug_fragment_variables(
            &mut ctx,
            dbg_op.get_operation(),
            &[llvm::DebugFragmentVariableInfo {
                variable: llvm::DebugLocalVariableInfo {
                    name: "pair".to_string(),
                    argument_index: None,
                    ty: llvm::DebugLocalTypeKind::Array {
                        name: "[u32; 2]".to_string(),
                        size_bits: 64,
                        element: Box::new(llvm::DebugLocalTypeKind::Basic {
                            name: "u32".to_string(),
                            size_bits: 32,
                            encoding: "DW_ATE_unsigned",
                        }),
                        count: 2,
                    },
                },
                fragment: llvm::DebugFragment {
                    offset_bits: 32,
                    size_bits: 32,
                },
                source_scope: Some(42),
                declaration: Some(llvm::DebugSourcePosition {
                    file: PathBuf::from("decl.rs"),
                    line: 7,
                    column: 3,
                }),
            }],
        );
        llvm::set_debug_local_declaration_location(
            &mut ctx,
            dbg_op.get_operation(),
            PathBuf::from("decl.rs"),
            7,
            3,
        );
        dbg_op.get_operation().insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        assert_eq!(count_ops::<mir::MirDbgValueOp>(&ctx, &body), 0);
        let dbg_value = find_first::<llvm::DebugValueOp>(&ctx, &body)
            .expect("expected lowered llvm.dbg_value marker");
        assert_eq!(
            dbg_value.get_operation().deref(&ctx).loc(),
            dbg_loc,
            "dbg.value lowering should keep the current-value source location"
        );
        let info = llvm::debug_local_variable(&ctx, dbg_value.get_operation())
            .expect("debug local metadata should survive dbg_value lowering");

        assert_eq!(info.name, "x");
        assert_eq!(
            llvm::debug_local_source_scope(&ctx, dbg_value.get_operation()),
            Some(42),
            "dbg.value lowering should keep the MIR source-scope owner"
        );
        let (decl_file, decl_pos) =
            llvm::debug_local_declaration_location(&ctx, dbg_value.get_operation())
                .expect("declaration location should survive dbg_value lowering");
        assert_eq!(decl_file, PathBuf::from("decl.rs"));
        assert_eq!(decl_pos.line, 7);
        assert_eq!(decl_pos.column, 3);
        assert_eq!(
            info.ty,
            llvm::DebugLocalTypeKind::Basic {
                name: "i32".to_string(),
                size_bits: 32,
                encoding: "DW_ATE_signed",
            }
        );
        let fragments = llvm::debug_fragment_variables(&ctx, dbg_value.get_operation());
        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].fragment.offset_bits, 32);
        assert_eq!(fragments[0].fragment.size_bits, 32);
        assert_eq!(fragments[0].variable.name, "pair");
    }

    #[test]
    fn convert_dbg_value_list_preserves_operands_and_expression() {
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, false);
        let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signless).into();

        let (module_ptr, block) = build_kernel(&mut ctx, vec![ptr_ty.into(), i64_ty], vec![]);
        let base = block.deref(&ctx).get_argument(0);
        let index = block.deref(&ctx).get_argument(1);

        let dbg_op = mir::MirDbgValueListOp::new(&mut ctx, vec![base, index]);
        llvm::set_debug_local_variable(
            &mut ctx,
            dbg_op.get_operation(),
            llvm::DebugLocalVariableInfo {
                name: "item".to_string(),
                argument_index: None,
                ty: llvm::DebugLocalTypeKind::Basic {
                    name: "u32".to_string(),
                    size_bits: 32,
                    encoding: "DW_ATE_unsigned",
                },
            },
        );
        let expression = llvm::DebugValueExpression::new(vec![
            llvm::DebugValueExpressionOp::Arg(0),
            llvm::DebugValueExpressionOp::Arg(1),
            llvm::DebugValueExpressionOp::ConstU(4),
            llvm::DebugValueExpressionOp::Mul,
            llvm::DebugValueExpressionOp::Plus,
            llvm::DebugValueExpressionOp::Deref,
        ]);
        llvm::set_debug_value_expression(&mut ctx, dbg_op.get_operation(), &expression);
        dbg_op.get_operation().insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        assert_eq!(count_ops::<mir::MirDbgValueListOp>(&ctx, &body), 0);
        let dbg_value = find_first::<llvm::DebugValueListOp>(&ctx, &body)
            .expect("expected lowered llvm.dbg_value_list marker");
        assert_eq!(dbg_value.values(&ctx), vec![base, index]);
        assert_eq!(
            llvm::debug_value_expression(&ctx, dbg_value.get_operation()),
            Some(expression)
        );
    }

    #[test]
    fn convert_alloca_preserves_local_memory_provenance() {
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, true);
        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);

        let alloca_op = Operation::new(
            &mut ctx,
            mir::MirAllocaOp::get_concrete_op_info(),
            vec![mir_ptr_ty.into()],
            vec![],
            vec![],
            0,
        );
        let provenance = llvm_export::ops::LocalMemoryProvenanceAttr {
            local_index: 3,
            size_bytes: 16,
            binding_name: "scratch".into(),
            type_name: "[u32; 4]".into(),
        };
        llvm_export::ops::set_local_memory_provenance(&mut ctx, alloca_op, provenance.clone());
        alloca_op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let alloca = find_first::<llvm::AllocaOp>(&ctx, &body).unwrap();
        let copied =
            llvm_export::ops::local_memory_provenance(&ctx, alloca.get_operation()).unwrap();
        assert_eq!(copied, provenance);
    }

    #[test]
    fn mem2reg_salvages_tagged_alloca_into_mir_dbg_value() {
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, true);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![i32_ty], vec![i32_ty]);
        let arg = block.deref(&ctx).get_argument(0);

        let alloca_op = Operation::new(
            &mut ctx,
            mir::MirAllocaOp::get_concrete_op_info(),
            vec![mir_ptr_ty.into()],
            vec![],
            vec![],
            0,
        );
        let decl_loc = src_location(&mut ctx, "kernel.rs", 12, 9);
        alloca_op.deref_mut(&ctx).set_loc(decl_loc.clone());
        llvm::set_debug_local_variable(
            &mut ctx,
            alloca_op,
            llvm::DebugLocalVariableInfo {
                name: "x".to_string(),
                argument_index: Some(1),
                ty: llvm::DebugLocalTypeKind::Basic {
                    name: "i32".to_string(),
                    size_bits: 32,
                    encoding: "DW_ATE_signed",
                },
            },
        );
        llvm::set_debug_local_source_scope(&mut ctx, alloca_op, 9);
        alloca_op.insert_at_back(block, &ctx);
        let slot = alloca_op.deref(&ctx).get_result(0);

        let store_op = Operation::new(
            &mut ctx,
            mir::MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![slot, arg],
            vec![],
            0,
        );
        store_op.insert_at_back(block, &ctx);

        let load_op = Operation::new(
            &mut ctx,
            mir::MirLoadOp::get_concrete_op_info(),
            vec![i32_ty],
            vec![slot],
            vec![],
            0,
        );
        load_op.insert_at_back(block, &ctx);
        let loaded = load_op.deref(&ctx).get_result(0);
        append_mir_return(&mut ctx, block, vec![loaded]);

        let mut analyses = pliron::pass::AnalysisManager::default();
        pliron::opts::mem2reg::mem2reg(module_ptr, &mut ctx, &mut analyses)
            .expect("mem2reg should promote the local slot");

        let blocks = vec![block];
        assert_eq!(count_ops::<mir::MirAllocaOp>(&ctx, &blocks), 0);
        assert_eq!(count_ops::<mir::MirStoreOp>(&ctx, &blocks), 0);
        assert_eq!(count_ops::<mir::MirLoadOp>(&ctx, &blocks), 0);

        let dbg_values = find_all::<mir::MirDbgValueOp>(&ctx, &blocks);
        assert!(
            !dbg_values.is_empty(),
            "mem2reg should leave value-based debug records for promoted locals"
        );
        let info = llvm::debug_local_variable(&ctx, dbg_values[0].get_operation())
            .expect("mir.dbg_value should carry the promoted local metadata");
        assert_eq!(info.name, "x");
        assert_eq!(info.argument_index, Some(1));
        assert_eq!(
            llvm::debug_local_source_scope(&ctx, dbg_values[0].get_operation()),
            Some(9),
            "mem2reg salvage should keep the local's MIR source-scope owner"
        );
        assert_eq!(
            dbg_values[0].get_operation().deref(&ctx).loc(),
            decl_loc,
            "debug records for source-less promoted ops should fall back to the local declaration"
        );
    }

    #[test]
    fn mem2reg_salvages_fragment_only_alloca_into_mir_dbg_value() {
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, true);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![i32_ty], vec![i32_ty]);
        let arg = block.deref(&ctx).get_argument(0);

        let alloca_op = Operation::new(
            &mut ctx,
            mir::MirAllocaOp::get_concrete_op_info(),
            vec![mir_ptr_ty.into()],
            vec![],
            vec![],
            0,
        );
        let alloca_loc = src_location(&mut ctx, "kernel.rs", 20, 9);
        alloca_op.deref_mut(&ctx).set_loc(alloca_loc);
        llvm::set_debug_fragment_variables(
            &mut ctx,
            alloca_op,
            &[llvm::DebugFragmentVariableInfo {
                variable: llvm::DebugLocalVariableInfo {
                    name: "pair".to_string(),
                    argument_index: None,
                    ty: llvm::DebugLocalTypeKind::Array {
                        name: "[u32; 2]".to_string(),
                        size_bits: 64,
                        element: Box::new(llvm::DebugLocalTypeKind::Basic {
                            name: "u32".to_string(),
                            size_bits: 32,
                            encoding: "DW_ATE_unsigned",
                        }),
                        count: 2,
                    },
                },
                fragment: llvm::DebugFragment {
                    offset_bits: 32,
                    size_bits: 32,
                },
                source_scope: Some(9),
                declaration: Some(llvm::DebugSourcePosition {
                    file: PathBuf::from("kernel.rs"),
                    line: 20,
                    column: 9,
                }),
            }],
        );
        alloca_op.insert_at_back(block, &ctx);
        let slot = alloca_op.deref(&ctx).get_result(0);

        let store_op = Operation::new(
            &mut ctx,
            mir::MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![slot, arg],
            vec![],
            0,
        );
        store_op.insert_at_back(block, &ctx);

        let load_op = Operation::new(
            &mut ctx,
            mir::MirLoadOp::get_concrete_op_info(),
            vec![i32_ty],
            vec![slot],
            vec![],
            0,
        );
        load_op.insert_at_back(block, &ctx);
        let loaded = load_op.deref(&ctx).get_result(0);
        append_mir_return(&mut ctx, block, vec![loaded]);

        let mut analyses = pliron::pass::AnalysisManager::default();
        pliron::opts::mem2reg::mem2reg(module_ptr, &mut ctx, &mut analyses)
            .expect("mem2reg should promote fragment storage");

        let dbg_values = find_all::<mir::MirDbgValueOp>(&ctx, &[block]);
        assert!(
            !dbg_values.is_empty(),
            "fragment-only storage should still produce mir.dbg_value salvage"
        );
        let fragments = llvm::debug_fragment_variables(&ctx, dbg_values[0].get_operation());
        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].variable.name, "pair");
        assert_eq!(fragments[0].fragment.offset_bits, 32);
        assert_eq!(fragments[0].fragment.size_bits, 32);
    }

    #[test]
    fn convert_load_preserves_volatile() {
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, false);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![mir_ptr_ty.into()], vec![]);
        let ptr_val = block.deref(&ctx).get_argument(0);

        let load_op = Operation::new(
            &mut ctx,
            mir::MirLoadOp::get_concrete_op_info(),
            vec![i32_ty],
            vec![ptr_val],
            vec![],
            0,
        );
        mir::MirLoadOp::new(load_op).set_volatile(&mut ctx, true);
        load_op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let load = find_first::<llvm::LoadOp>(&ctx, &body).unwrap();
        assert!(
            llvm_export::ops::op_volatile(&ctx, load.get_operation()),
            "volatile mir.load must lower to a volatile llvm.load"
        );
    }

    #[test]
    fn convert_ref_lowers_to_alloca_then_store() {
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, false);

        // Take a u32 by value, build `&x`.
        let (module_ptr, block) = build_kernel(&mut ctx, vec![i32_ty], vec![]);
        let arg = block.deref(&ctx).get_argument(0);

        let ref_op_ptr = Operation::new(
            &mut ctx,
            mir::MirRefOp::get_concrete_op_info(),
            vec![mir_ptr_ty.into()],
            vec![arg],
            vec![],
            0,
        );
        mir::MirRefOp::new(ref_op_ptr).set_mutable(&mut ctx, false);
        ref_op_ptr.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        assert_eq!(
            count_ops::<llvm::AllocaOp>(&ctx, &body),
            1,
            "ref must materialize via alloca"
        );
        assert_eq!(
            count_ops::<llvm::StoreOp>(&ctx, &body),
            1,
            "ref must store the value into the alloca"
        );
        assert_eq!(count_ops::<mir::MirRefOp>(&ctx, &body), 0);
    }

    #[test]
    fn convert_ref_preserves_tuple_alignment_on_alloca_and_store() {
        let mut ctx = make_ctx();
        let tuple_ty = over_aligned_tuple_ty(&mut ctx);
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, tuple_ty, false);
        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);

        let undef = mir::MirUndefOp::new(&mut ctx, tuple_ty);
        undef.get_operation().insert_at_back(block, &ctx);
        let value = undef.get_operation().deref(&ctx).get_result(0);
        let ref_op = Operation::new(
            &mut ctx,
            mir::MirRefOp::get_concrete_op_info(),
            vec![mir_ptr_ty.into()],
            vec![value],
            vec![],
            0,
        );
        mir::MirRefOp::new(ref_op).set_mutable(&mut ctx, false);
        ref_op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let alloca = find_first::<llvm::AllocaOp>(&ctx, &body).expect("expected llvm.alloca");
        let store = find_first::<llvm::StoreOp>(&ctx, &body).expect("expected llvm.store");
        assert_eq!(
            llvm_export::ops::op_alignment(&ctx, alloca.get_operation()),
            Some(32)
        );
        assert_eq!(
            llvm_export::ops::op_alignment(&ctx, store.get_operation()),
            Some(32)
        );
    }

    #[test]
    fn convert_ref_preserves_over_aligned_union_array_layout_and_alignment() {
        for abi_align in [32, 64] {
            let mut ctx = make_ctx();
            let u32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Unsigned).into();
            let union_ty: TypeHandle = MirUnionType::get(
                &mut ctx,
                format!("Align{abi_align}Union"),
                vec!["word".into()],
                vec![u32_ty],
                abi_align,
                abi_align,
            )
            .into();
            let array_ty: TypeHandle = MirArrayType::get(&mut ctx, union_ty, 3).into();
            let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, array_ty, false);
            let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);

            let undef = mir::MirUndefOp::new(&mut ctx, array_ty);
            undef.get_operation().insert_at_back(block, &ctx);
            let value = undef.get_operation().deref(&ctx).get_result(0);
            let ref_op = Operation::new(
                &mut ctx,
                mir::MirRefOp::get_concrete_op_info(),
                vec![mir_ptr_ty.into()],
                vec![value],
                vec![],
                0,
            );
            mir::MirRefOp::new(ref_op).set_mutable(&mut ctx, false);
            ref_op.insert_at_back(block, &ctx);
            append_mir_return(&mut ctx, block, vec![]);

            crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

            let body = kernel_blocks(&ctx, module_ptr);
            let alloca = find_first::<llvm::AllocaOp>(&ctx, &body).expect("expected llvm.alloca");
            let store = find_first::<llvm::StoreOp>(&ctx, &body).expect("expected llvm.store");
            let llvm_array_ty = alloca.result_pointee_type(&ctx);
            let llvm_array_data = llvm_array_ty.deref(&ctx);
            let llvm_array = llvm_array_data
                .downcast_ref::<ArrayType>()
                .expect("over-aligned union array must remain an LLVM array");

            assert_eq!(llvm_array.size(), 3);
            assert_eq!(
                crate::convert::types::llvm_type_size_align(&ctx, llvm_array.elem_type()),
                Some((abi_align, 16))
            );
            assert_eq!(
                crate::convert::types::llvm_type_size_align(&ctx, llvm_array_ty),
                Some((abi_align * 3, 16))
            );
            assert_eq!(
                llvm_export::ops::op_alignment(&ctx, alloca.get_operation()),
                Some(abi_align as u32)
            );
            assert_eq!(
                llvm_export::ops::op_alignment(&ctx, store.get_operation()),
                Some(abi_align as u32)
            );
        }
    }

    #[test]
    fn convert_ptr_offset_lowers_to_gep_with_pointee_elem_type() {
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signless).into();
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, true);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![mir_ptr_ty.into(), i64_ty], vec![]);
        let ptr_val = block.deref(&ctx).get_argument(0);
        let off_val = block.deref(&ctx).get_argument(1);

        let off_op = Operation::new(
            &mut ctx,
            mir::MirPtrOffsetOp::get_concrete_op_info(),
            vec![mir_ptr_ty.into()],
            vec![ptr_val, off_val],
            vec![],
            0,
        );
        off_op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let gep = find_first::<llvm::GetElementPtrOp>(&ctx, &body).expect("expected one llvm.gep");
        // Element type must come from the MirPtrType pointee (i32), not the
        // i8 fallback used when no operand-type info is available.
        let elem_ty = gep.src_elem_type(&ctx);
        let elem_ty_ref = elem_ty.deref(&ctx);
        let int_ty = elem_ty_ref
            .downcast_ref::<IntegerType>()
            .expect("gep src_elem_type should be IntegerType");
        assert_eq!(int_ty.width(), 32, "gep elem type must be i32 (pointee)");
        assert!(
            llvm::gep_inbounds(&ctx, gep.get_operation()),
            "ordinary pointer offsets retain the in-bounds contract"
        );
    }

    #[test]
    fn convert_wrapping_ptr_offset_lowers_to_non_inbounds_gep() {
        let mut ctx = make_ctx();
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signed).into();
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, i32_ty, false);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![mir_ptr_ty.into(), i64_ty], vec![]);
        let ptr_val = block.deref(&ctx).get_argument(0);
        let off_val = block.deref(&ctx).get_argument(1);

        let off_op = Operation::new(
            &mut ctx,
            mir::MirPtrOffsetOp::get_concrete_op_info(),
            vec![mir_ptr_ty.into()],
            vec![ptr_val, off_val],
            vec![],
            0,
        );
        mir::MirPtrOffsetOp::new(off_op).set_inbounds(&mut ctx, false);
        off_op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let body = kernel_blocks(&ctx, module_ptr);
        let gep = find_first::<llvm::GetElementPtrOp>(&ctx, &body).expect("expected one llvm.gep");
        assert!(
            !llvm::gep_inbounds(&ctx, gep.get_operation()),
            "wrapping pointer offsets must not promise in-bounds arithmetic"
        );
    }

    // =========================================================================
    // Enum layout: converted width per shape + divergent-enum rejection
    // =========================================================================

    use dialect_mir::types::{EnumVariant, MirEnumType};

    /// Build a Direct-tag `MirEnumType` the way the importer does:
    /// unsigned tag of `tag_bits`, plus rustc's `total_size`/`abi_align`.
    fn make_enum_ty(
        ctx: &mut Context,
        name: &str,
        tag_bits: u32,
        variants: Vec<EnumVariant>,
        total_size: u64,
        abi_align: u64,
    ) -> TypeHandle {
        let tag_ty: TypeHandle = IntegerType::get(ctx, tag_bits, Signedness::Unsigned).into();
        // Sequential 0..n discriminants: these layout tests only exercise
        // size/width, not value mapping.
        let discriminants: Vec<u64> = (0..variants.len() as u64).collect();
        MirEnumType::get_with_layout(
            ctx,
            name.to_string(),
            tag_ty,
            discriminants,
            variants,
            0, // tag at byte 0, like every shape these tests exercise
            total_size,
            abi_align,
        )
        .into()
    }

    fn unit_variants(n: usize) -> Vec<EnumVariant> {
        (0..n).map(|i| EnumVariant::unit(format!("V{i}"))).collect()
    }

    /// Converted enum allocation size must equal rustc's `total_size` for
    /// every memory-faithful tag shape: that size is what GEP strides by.
    #[test]
    fn enum_conversion_strides_by_rustc_size() {
        use crate::convert::types::llvm_type_size_align;

        let mut ctx = make_ctx();

        // #[repr(u32)] fieldless (issue #118 shape): {i32}, 4 bytes.
        let repr_u32 = make_enum_ty(&mut ctx, "ReprU32", 32, unit_variants(4), 4, 4);
        let conv = convert_type(&mut ctx, repr_u32).unwrap();
        assert_eq!(
            llvm_type_size_align(&ctx, conv),
            Some((4, 4)),
            "repr(u32) tag"
        );

        // #[repr(usize)] fieldless: {i64}, 8 bytes.
        let repr_usize = make_enum_ty(&mut ctx, "ReprUsize", 64, unit_variants(4), 8, 8);
        let conv = convert_type(&mut ctx, repr_usize).unwrap();
        assert_eq!(
            llvm_type_size_align(&ctx, conv),
            Some((8, 8)),
            "repr(usize) tag"
        );

        // repr(align(8)) with an i8 tag: the byte claims alone would give
        // alignment 1, so the slot map raises the storage alignment with a
        // zero-length anchor field. Size and alignment must both match
        // rustc, or arrays and by-value ABI uses would be unsound.
        let padded = make_enum_ty(&mut ctx, "Padded", 8, unit_variants(2), 8, 8);
        let conv = convert_type(&mut ctx, padded).unwrap();
        assert_eq!(
            llvm_type_size_align(&ctx, conv),
            Some((8, 8)),
            "repr(align(8)) i8 tag"
        );

        // u8 tag + i64 payload, rustc size 16: the slot map places the
        // payload at its rustc byte offset 8 behind an explicit
        // [7 x i8] filler, making the layout datalayout-independent.
        let i64_payload: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let payload = make_enum_ty(
            &mut ctx,
            "OnePayload",
            8,
            vec![
                EnumVariant::new_with_layout("A".to_string(), vec![i64_payload], vec![8], vec![8]),
                EnumVariant::unit("B".to_string()),
            ],
            16,
            8,
        );
        let conv = convert_type(&mut ctx, payload).unwrap();
        let (size, _align) = llvm_type_size_align(&ctx, conv).unwrap();
        assert_eq!(size, 16, "natural layout matches rustc size, no pad");
        let conv_ref = conv.deref(&ctx);
        let struct_ty = conv_ref
            .downcast_ref::<llvm_export::types::StructType>()
            .expect("converted enum is a struct");
        assert_eq!(
            struct_ty.fields().count(),
            3,
            "{{tag, [7 x i8] filler, payload}}: explicit filler to byte 8"
        );
    }

    /// Multi-payload enum: variants overlap in Rust, and identical
    /// (offset, converted type) payloads share one typed slot, so the
    /// converted struct is byte-identical to rustc's layout AND every
    /// access stays pure SSA (no spill).
    #[test]
    fn multi_payload_enum_shares_payload_slot() {
        use crate::convert::types::{build_enum_slot_map, llvm_type_size_align};

        let mut ctx = make_ctx();
        let e = make_multi_payload_enum_ty(&mut ctx);
        let map = build_enum_slot_map(&mut ctx, e).unwrap();
        assert_eq!(map.carrier_slot, Some(0));
        assert_eq!(
            map.field_slots,
            vec![Some(1), Some(1)],
            "A.0 and B.0 overlap at byte 4 with the same type: one shared slot"
        );
        assert_eq!(
            llvm_type_size_align(&ctx, map.llvm_struct_ty),
            Some((8, 4)),
            "byte-identical to rustc's 8-byte layout, not the 12-byte concat"
        );
    }

    /// rustc may place the tag AFTER payload bytes; the slot map must
    /// follow the recorded tag_offset, never assume slot 0.
    #[test]
    fn enum_slot_map_tag_not_at_zero() {
        use crate::convert::types::{build_enum_slot_map, llvm_type_size_align};

        let mut ctx = make_ctx();
        let u64_a: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let u64_b: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let tag_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        // enum F { A(u64), B(u64) }: payloads share byte 0, tag at byte 8.
        let ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "TagAtEight".to_string(),
            tag_ty,
            vec![0, 1],
            vec![
                EnumVariant::new_with_layout("A".to_string(), vec![u64_a], vec![0], vec![8]),
                EnumVariant::new_with_layout("B".to_string(), vec![u64_b], vec![0], vec![8]),
            ],
            8,
            16,
            8,
        )
        .into();
        let map = build_enum_slot_map(&mut ctx, ty).unwrap();
        assert_eq!(
            map.field_slots,
            vec![Some(0), Some(0)],
            "payloads share the first slot"
        );
        assert_eq!(
            map.carrier_slot,
            Some(1),
            "tag claims its own slot at byte 8"
        );
        let (size, _align) = llvm_type_size_align(&ctx, map.llvm_struct_ty).unwrap();
        assert_eq!(size, 16, "{{ i64, i8, [7 x i8] }}");
    }

    /// Multi-payload enum whose variants overlap in Rust: both payloads use
    /// bytes 4..8 after the direct `u32` tag, so the complete value is eight
    /// bytes. Mimics `#[repr(u32)] enum E { A(u32), B(u32) }`.
    fn make_multi_payload_enum_ty(ctx: &mut Context) -> TypeHandle {
        let i32_a: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let i32_b: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        make_enum_ty(
            ctx,
            "MultiPayload",
            32,
            vec![
                EnumVariant::new_with_layout("A".to_string(), vec![i32_a], vec![4], vec![4]),
                EnumVariant::new_with_layout("B".to_string(), vec![i32_b], vec![4], vec![4]),
            ],
            8,
            4,
        )
    }

    /// Device-local GEP + load must use the same exact eight-byte rustc layout
    /// as every other enum path. This protects pointer stride in issue #131's
    /// in-kernel `[E; 4]` arrays; the sibling kernel-parameter test proves the
    /// same representation is also valid at the host/device boundary.
    #[test]
    fn device_local_multi_payload_enum_gep_and_load_lower() {
        let mut ctx = make_ctx();
        let enum_ty = make_multi_payload_enum_ty(&mut ctx);
        let i64_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Signless).into();
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, enum_ty, true);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![mir_ptr_ty.into(), i64_ty], vec![]);
        let ptr_val = block.deref(&ctx).get_argument(0);
        let off_val = block.deref(&ctx).get_argument(1);

        let off_op = Operation::new(
            &mut ctx,
            mir::MirPtrOffsetOp::get_concrete_op_info(),
            vec![mir_ptr_ty.into()],
            vec![ptr_val, off_val],
            vec![],
            0,
        );
        off_op.insert_at_back(block, &ctx);
        let elem_ptr = off_op.deref(&ctx).get_result(0);

        let load_op = Operation::new(
            &mut ctx,
            mir::MirLoadOp::get_concrete_op_info(),
            vec![enum_ty],
            vec![elem_ptr],
            vec![],
            0,
        );
        load_op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr)
            .expect("device-local rustc-layout enum GEP + load must lower");
    }

    /// A kernel parameter may carry this enum because the device slot map is
    /// byte-identical to rustc's overlapped host layout.
    #[test]
    fn kernel_param_accepts_multi_payload_enum() {
        use pliron::builtin::attributes::StringAttr;

        let mut ctx = make_ctx();
        let enum_ty = make_multi_payload_enum_ty(&mut ctx);
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, enum_ty, false);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![mir_ptr_ty.into()], vec![]);
        append_mir_return(&mut ctx, block, vec![]);

        // Mark the function as a GPU kernel the way the importer does.
        {
            let module_block = module_ptr
                .deref(&ctx)
                .get_region(0)
                .deref(&ctx)
                .iter(&ctx)
                .next()
                .unwrap();
            let func_op = module_block.deref(&ctx).iter(&ctx).next().unwrap();
            let kernel_attr = StringAttr::new("true".to_string());
            let key: pliron::identifier::Identifier = "gpu_kernel".try_into().unwrap();
            func_op.deref_mut(&ctx).attributes.set(key, kernel_attr);
        }

        // The slot map lowers MultiPayload byte-identically to rustc's
        // layout ({ i32, i32 }, 8 bytes), so the kernel ABI accepts it.
        crate::lower_mir_to_llvm(&mut ctx, module_ptr)
            .expect("multi-payload enum kernel param must lower");
    }

    /// A legacy enum without physical rustc metadata must be rejected at the
    /// kernel boundary. Importer-produced niche layouts are now known and are
    /// accepted; this fixture deliberately constructs `Unknown` metadata.
    #[test]
    fn kernel_param_rejects_enum_with_unknown_layout() {
        use pliron::builtin::attributes::StringAttr;

        let mut ctx = make_ctx();
        // MirEnumType::get with no layout is the legacy Unknown form.
        let i32_ty: TypeHandle = IntegerType::get(&ctx, 32, Signedness::Signless).into();
        let pointee = MirPtrType::get_generic(&mut ctx, i32_ty, false);
        let tag_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let niched: TypeHandle = MirEnumType::get(
            &mut ctx,
            "Option".to_string(),
            tag_ty,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".to_string()),
                EnumVariant::new("Some".to_string(), vec![pointee.into()]),
            ],
        )
        .into();
        let mir_ptr_ty = MirPtrType::get_generic(&mut ctx, niched, false);

        let (module_ptr, block) = build_kernel(&mut ctx, vec![mir_ptr_ty.into()], vec![]);
        append_mir_return(&mut ctx, block, vec![]);

        {
            let module_block = module_ptr
                .deref(&ctx)
                .get_region(0)
                .deref(&ctx)
                .iter(&ctx)
                .next()
                .unwrap();
            let func_op = module_block.deref(&ctx).iter(&ctx).next().unwrap();
            let kernel_attr = StringAttr::new("true".to_string());
            let key: pliron::identifier::Identifier = "gpu_kernel".try_into().unwrap();
            func_op.deref_mut(&ctx).attributes.set(key, kernel_attr);
        }

        let err = crate::lower_mir_to_llvm(&mut ctx, module_ptr)
            .expect_err("unknown-layout enum kernel param must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("Option") && msg.contains("kernel boundary"),
            "error must name the enum and the kernel boundary, got: {msg}"
        );
        assert!(msg.contains("unknown physical rustc layout"));
    }

    /// Build a `mir.shared_alloc` returning `MirPtrType<i32, addrspace=3>` of
    /// length `size`, with the given alloc_key, and append it to `block`.
    fn append_shared_alloc(ctx: &mut Context, block: Ptr<BasicBlock>, alloc_key: &str, size: u64) {
        append_shared_alloc_named(ctx, block, alloc_key, size, None);
    }

    /// As [`append_shared_alloc`], additionally carrying the Rust path of the
    /// `static` the allocation came from.
    fn append_shared_alloc_named(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        alloc_key: &str,
        size: u64,
        source_name: Option<&str>,
    ) {
        use pliron::builtin::attributes::IntegerAttr;
        use pliron::utils::apint::APInt;

        let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();
        let result_ty = MirPtrType::get_shared(ctx, i32_ty, true);
        let op = Operation::new(
            ctx,
            mir::MirSharedAllocOp::get_concrete_op_info(),
            vec![result_ty.into()],
            vec![],
            vec![],
            0,
        );
        let alloc = mir::MirSharedAllocOp::new(op);
        alloc.set_attr_elem_type(ctx, TypeAttr::new(i32_ty));
        let size_attr = IntegerAttr::new(
            IntegerType::get(ctx, 64, Signedness::Signless),
            APInt::from_u64(size, std::num::NonZeroUsize::new(64).unwrap()),
        );
        alloc.set_attr_size(ctx, size_attr);
        alloc.set_attr_alloc_key(ctx, StringAttr::new(alloc_key.to_string()));
        if let Some(source_name) = source_name {
            alloc.set_attr_source_name(ctx, StringAttr::new(source_name.to_string()));
        }
        op.insert_at_back(block, ctx);
    }

    #[test]
    fn convert_shared_alloc_creates_global_in_addrspace_3() {
        let mut ctx = make_ctx();
        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
        append_shared_alloc(&mut ctx, block, "k1", 64);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        // Global lives at module level; addressof lives in the function body.
        let top = module_top_block(&ctx, module_ptr);
        let global = top
            .deref(&ctx)
            .iter(&ctx)
            .find_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
            .expect("expected an llvm.global for the shared allocation");
        assert_eq!(
            global.address_space(&ctx),
            llvm_addr::SHARED,
            "shared_alloc global must live in addrspace 3"
        );
        assert!(
            global
                .get_symbol_name(&ctx)
                .to_string()
                .starts_with("__shared_mem_"),
            "shared global should have __shared_mem_ prefix"
        );

        let body = kernel_blocks(&ctx, module_ptr);
        let addrof =
            find_first::<llvm::AddressOfOp>(&ctx, &body).expect("expected an llvm.addressof");
        assert_eq!(
            ptr_addrspace(
                &ctx,
                addrof
                    .get_operation()
                    .deref(&ctx)
                    .get_result(0)
                    .get_type(&ctx)
            ),
            llvm_addr::SHARED,
        );
    }

    #[test]
    fn convert_shared_alloc_deduplicates_by_alloc_key() {
        let mut ctx = make_ctx();
        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
        // Two allocations sharing the same alloc_key — they must collapse to
        // a single underlying global (this is what enables a single `static`
        // referenced from multiple sites to land in one shared region).
        append_shared_alloc(&mut ctx, block, "same-key", 64);
        append_shared_alloc(&mut ctx, block, "same-key", 64);
        // A third with a different key must NOT dedupe with them.
        append_shared_alloc(&mut ctx, block, "other-key", 32);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let top = module_top_block(&ctx, module_ptr);
        let shared_globals = top
            .deref(&ctx)
            .iter(&ctx)
            .filter_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
            .filter(|g| g.address_space(&ctx) == llvm_addr::SHARED)
            .count();
        assert_eq!(
            shared_globals, 2,
            "two distinct alloc_keys must produce two globals"
        );

        // Each of the three mir.shared_alloc ops becomes one addressof.
        let body = kernel_blocks(&ctx, module_ptr);
        assert_eq!(count_ops::<llvm::AddressOfOp>(&ctx, &body), 3);
    }

    /// Collect `(symbol, source_name)` for every shared global in the module.
    fn shared_global_source_names(
        ctx: &Context,
        module_ptr: Ptr<Operation>,
    ) -> Vec<(String, Option<String>)> {
        use llvm_export::ops::GlobalOpExt;

        let top = module_top_block(ctx, module_ptr);
        let mut named: Vec<_> = top
            .deref(ctx)
            .iter(ctx)
            .filter_map(|op| Operation::get_op::<llvm::GlobalOp>(op, ctx))
            .filter(|g| g.address_space(ctx) == llvm_addr::SHARED)
            .map(|g| {
                (
                    g.get_symbol_name(ctx).to_string(),
                    g.shared_source_name(ctx),
                )
            })
            .collect();
        // Globals are inserted at the front of the module block, so iteration
        // order is the reverse of creation order. Sort for a stable assertion.
        named.sort();
        named
    }

    #[test]
    fn shared_alloc_source_name_reaches_the_generated_global() {
        let mut ctx = make_ctx();
        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
        append_shared_alloc_named(&mut ctx, block, "k1", 64, Some("my_kernel::TILE"));
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let named = shared_global_source_names(&ctx, module_ptr);
        assert_eq!(named.len(), 1, "expected exactly one shared global");
        let (symbol, source_name) = &named[0];
        // The symbol itself must stay anonymous: the whole point of the
        // sidecar attribute is that it does not perturb the emitted name.
        assert!(
            symbol.starts_with("__shared_mem_"),
            "the generated symbol must not be renamed, got `{symbol}`"
        );
        assert_eq!(source_name.as_deref(), Some("my_kernel::TILE"));
    }

    #[test]
    fn shared_alloc_without_source_name_leaves_the_global_unlabelled() {
        let mut ctx = make_ctx();
        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
        append_shared_alloc(&mut ctx, block, "k1", 64);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let named = shared_global_source_names(&ctx, module_ptr);
        assert_eq!(named.len(), 1);
        assert_eq!(named[0].1, None, "an unnamed allocation must stay unnamed");
    }

    #[test]
    fn shared_alloc_source_names_are_per_global_not_shared_across_them() {
        let mut ctx = make_ctx();
        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
        // Two references to one static dedupe onto a single global, and a
        // second static gets its own. Each global must carry its own name —
        // the failure this guards is one name leaking onto every allocation.
        append_shared_alloc_named(&mut ctx, block, "tile", 64, Some("my_kernel::TILE"));
        append_shared_alloc_named(&mut ctx, block, "tile", 64, Some("my_kernel::TILE"));
        append_shared_alloc_named(&mut ctx, block, "scratch", 32, Some("my_kernel::SCRATCH"));
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let names: Vec<_> = shared_global_source_names(&ctx, module_ptr)
            .into_iter()
            .map(|(_, source_name)| source_name)
            .collect();
        assert_eq!(names.len(), 2, "the shared alloc_key must still dedupe");
        let mut names: Vec<_> = names.into_iter().map(|n| n.expect("named")).collect();
        names.sort();
        assert_eq!(names, vec!["my_kernel::SCRATCH", "my_kernel::TILE"]);
    }

    #[test]
    fn shared_alloc_source_name_reaches_the_exported_llvm_ir() {
        // The end the feature exists for: a consumer holding only the emitted
        // artifact can tell which Rust `static` a `__shared_mem_N` block is.
        let mut ctx = make_ctx();
        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
        append_shared_alloc_named(&mut ctx, block, "tile", 64, Some("my_kernel::TILE"));
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let module = Operation::get_op::<pliron::builtin::ops::ModuleOp>(module_ptr, &ctx)
            .expect("lowered top-level op is a module");
        let ir = llvm_export::export::export_module_to_string(&ctx, &module).expect("export");

        let comment_index = ir
            .find("; shared source: my_kernel::TILE")
            .unwrap_or_else(|| panic!("exported IR must name the shared source:\n{ir}"));
        let definition_index = ir
            .find("__shared_mem_")
            .expect("exported IR must declare the shared global");
        assert!(
            comment_index < definition_index,
            "the source comment must precede the global it describes:\n{ir}"
        );
    }

    /// A `__shared_mem_N` or `__device_global_N` index must depend only on
    /// the module being lowered, not on how many allocations any OTHER
    /// module has already lowered in this process (#706). Before the fix,
    /// each `N` came from a `static AtomicUsize` shared across every call in
    /// the process, so lowering the second of these two modules would have
    /// produced `__shared_mem_1` and `__device_global_1`, not the `_0` names.
    #[test]
    fn shared_and_device_global_indices_are_per_module_not_process_global() {
        for _ in 0..2 {
            let mut ctx = make_ctx();
            let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
            append_shared_alloc(&mut ctx, block, "k", 64);
            append_global_alloc(&mut ctx, block, "ordinary_static", false);
            append_mir_return(&mut ctx, block, vec![]);

            crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

            let top = module_top_block(&ctx, module_ptr);
            let names: Vec<String> = top
                .deref(&ctx)
                .iter(&ctx)
                .filter_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
                .map(|g| g.get_symbol_name(&ctx).to_string())
                .collect();
            assert!(
                names.iter().any(|n| n == "__shared_mem_0"),
                "a module with exactly one shared allocation must always name it \
                 __shared_mem_0, regardless of how many other modules already lowered \
                 one in this process (got {names:?})"
            );
            assert!(
                names.iter().any(|n| n == "__device_global_0"),
                "a module with exactly one ordinary device global must always name it \
                 __device_global_0, regardless of how many other modules already \
                 lowered one in this process (got {names:?})"
            );
        }
    }

    fn append_global_alloc(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        global_key: &str,
        constant: bool,
    ) -> Ptr<Operation> {
        let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();
        let result_ty = if constant {
            MirPtrType::get_constant(ctx, i32_ty, false)
        } else {
            MirPtrType::get_global(ctx, i32_ty, true)
        };
        let op = Operation::new(
            ctx,
            mir::MirGlobalAllocOp::get_concrete_op_info(),
            vec![result_ty.into()],
            vec![],
            vec![],
            0,
        );
        let alloc = mir::MirGlobalAllocOp::new(op);
        alloc.set_attr_global_type(ctx, TypeAttr::new(i32_ty));
        alloc.set_attr_global_key(ctx, StringAttr::new(global_key.to_string()));
        op.insert_at_back(block, ctx);
        op
    }

    #[test]
    fn convert_global_alloc_places_in_global_or_constant_addrspace() {
        let mut ctx = make_ctx();
        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
        append_global_alloc(&mut ctx, block, "ordinary_static", false);
        append_global_alloc(&mut ctx, block, "_ZN7my_mod3KEYE", true);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let top = module_top_block(&ctx, module_ptr);
        let globals: Vec<_> = top
            .deref(&ctx)
            .iter(&ctx)
            .filter_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
            .collect();
        let global_addr_global = globals
            .iter()
            .find(|g| g.address_space(&ctx) == llvm_addr::GLOBAL)
            .expect("expected one global in addrspace(1)");
        let global_addr_const = globals
            .iter()
            .find(|g| g.address_space(&ctx) == llvm_addr::CONSTANT)
            .expect("expected one global in addrspace(4)");

        // Constant-memory globals reuse the Rust mangled name so host code can
        // resolve them by name via `cuModuleGetGlobal`; ordinary globals get
        // a counter-suffixed `__device_global_N`.
        assert_eq!(
            global_addr_const.get_symbol_name(&ctx).to_string(),
            "_ZN7my_mod3KEYE",
            "constant globals must keep the mangled global_key as symbol name"
        );
        assert!(
            global_addr_global
                .get_symbol_name(&ctx)
                .to_string()
                .starts_with("__device_global_"),
            "ordinary device globals get the __device_global_ prefix"
        );
    }

    #[test]
    fn immutable_marking_survives_lowering_and_is_not_assumed() {
        let mut ctx = make_ctx();
        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);

        // Two ordinary addrspace(1) globals, distinguished by their source key.
        // Only the promoted one claims immutability; the plain static must not
        // acquire it, or the exporter would write `constant` for storage the
        // host can still overwrite by symbol.
        let promoted = append_global_alloc(&mut ctx, block, "promoted_table", false);
        mir::MirGlobalAllocOp::new(promoted).mark_immutable(&mut ctx);
        append_global_alloc(&mut ctx, block, "plain_static", false);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let top = module_top_block(&ctx, module_ptr);
        let globals: Vec<llvm::GlobalOp> = top
            .deref(&ctx)
            .iter(&ctx)
            .filter_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
            .collect();
        let by_key = |key: &str| -> llvm::GlobalOp {
            *globals
                .iter()
                .find(|g| g.source_global_key(&ctx).as_deref() == Some(key))
                .unwrap_or_else(|| panic!("no lowered global carries source key {key}"))
        };

        assert!(
            by_key("promoted_table").is_immutable(&ctx),
            "a global marked immutable in MIR must stay immutable through lowering"
        );
        assert!(
            !by_key("plain_static").is_immutable(&ctx),
            "lowering must not infer immutability; only the promoted-constant \
             sites may claim it"
        );
    }

    #[test]
    fn initialized_global_lowers_to_byte_storage() {
        let mut ctx = make_ctx();
        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
        let op = append_global_alloc(&mut ctx, block, "nan_payload", false);
        let alloc = mir::MirGlobalAllocOp::new(op);
        alloc.set_alignment_value(&mut ctx, 4);
        let initializer_key: Identifier = "global_initializer_hex".try_into().unwrap();
        op.deref_mut(&ctx)
            .attributes
            .set(initializer_key, StringAttr::new("3412c07f".to_string()));
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let top = module_top_block(&ctx, module_ptr);
        let global = top
            .deref(&ctx)
            .iter(&ctx)
            .find_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
            .expect("expected lowered device global");
        let global_ty = global.get_type(&ctx);
        let global_ty_ref = global_ty.deref(&ctx);
        let array_ty = global_ty_ref
            .downcast_ref::<ArrayType>()
            .expect("initialized global must use byte-array storage");
        assert_eq!(array_ty.size(), 4);
        let elem_ty = array_ty.elem_type();
        let elem_ty_ref = elem_ty.deref(&ctx);
        let elem = elem_ty_ref
            .downcast_ref::<IntegerType>()
            .expect("byte-array element must be an integer");
        assert_eq!(elem.width(), 8);
        assert_eq!(global.get_alignment(&ctx), Some(4));
        assert_eq!(global.initializer_hex(&ctx).as_deref(), Some("3412c07f"));
    }

    #[test]
    fn relocated_global_lowers_to_segmented_storage() {
        let mut ctx = make_ctx();
        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
        let word_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let mir_global_ty: TypeHandle = MirArrayType::get(&mut ctx, word_ty, 3).into();
        let result_ty = MirPtrType::get_global(&mut ctx, mir_global_ty, false);
        let op = Operation::new(
            &mut ctx,
            mir::MirGlobalAllocOp::get_concrete_op_info(),
            vec![result_ty.into()],
            vec![],
            vec![],
            0,
        );
        let alloc = mir::MirGlobalAllocOp::new(op);
        alloc.set_attr_global_type(&ctx, TypeAttr::new(mir_global_ty));
        alloc.set_attr_global_key(&ctx, StringAttr::new("reference_table".to_string()));
        alloc.set_alignment_value(&mut ctx, 8);
        op.deref_mut(&ctx).attributes.set(
            "global_initializer_hex".try_into().unwrap(),
            StringAttr::new("000000000000000000000000000000000000000000000000".to_string()),
        );
        let encoded =
            llvm::encode_global_initializer_relocations(&[llvm::GlobalInitializerRelocation {
                source_offset: 8,
                width_bytes: 8,
                target_address_space: llvm_addr::GLOBAL,
                target_addend: 4,
                target_key: "target_static".to_string(),
            }]);
        op.deref_mut(&ctx).attributes.set(
            "global_initializer_relocations".try_into().unwrap(),
            StringAttr::new(encoded.clone()),
        );
        op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module_ptr).expect("lowering failed");

        let top = module_top_block(&ctx, module_ptr);
        let global = top
            .deref(&ctx)
            .iter(&ctx)
            .find_map(|op| Operation::get_op::<llvm::GlobalOp>(op, &ctx))
            .expect("expected lowered device global");
        let global_ty = global.get_type(&ctx);
        let global_ty_ref = global_ty.deref(&ctx);
        let storage = global_ty_ref
            .downcast_ref::<StructType>()
            .expect("relocated initializer must use segmented struct storage");
        assert_eq!(storage.num_fields(), 3);
        assert_eq!(
            global.source_global_key(&ctx).as_deref(),
            Some("reference_table")
        );
        assert_eq!(
            global.initializer_relocations(&ctx).as_deref(),
            Some(encoded.as_str())
        );
    }

    #[test]
    fn relocated_global_uses_packed_storage_for_unaligned_pointer_slot() {
        let mut ctx = make_ctx();
        let encoded =
            llvm::encode_global_initializer_relocations(&[llvm::GlobalInitializerRelocation {
                source_offset: 1,
                width_bytes: 8,
                target_address_space: llvm_addr::GLOBAL,
                target_addend: 0,
                target_key: "target_static".to_string(),
            }]);

        let storage = relocated_initializer_storage_type(&mut ctx, 9, 1, &encoded)
            .expect("unaligned relocation should use packed storage");
        let storage_ref = storage.deref(&ctx);
        let struct_ty = storage_ref
            .downcast_ref::<StructType>()
            .expect("relocated initializer must use struct storage");
        assert_eq!(struct_ty.layout(), StructLayout::Packed);
        assert_eq!(struct_ty.num_fields(), 2);
        assert_eq!(get_type_size(&ctx, storage), 9);

        let fields: Vec<_> = struct_ty.fields().collect();
        let literal_ref = fields[0].deref(&ctx);
        let literal = literal_ref
            .downcast_ref::<ArrayType>()
            .expect("leading literal span must be a byte array");
        assert_eq!(literal.size(), 1);
        let pointer_ref = fields[1].deref(&ctx);
        let pointer = pointer_ref
            .downcast_ref::<IntegerType>()
            .expect("relocation slot must be an integer carrier");
        assert_eq!(pointer.width(), 64);
    }

    #[test]
    fn relocated_global_uses_packed_storage_for_underaligned_allocation() {
        let mut ctx = make_ctx();
        let encoded =
            llvm::encode_global_initializer_relocations(&[llvm::GlobalInitializerRelocation {
                source_offset: 0,
                width_bytes: 8,
                target_address_space: llvm_addr::GLOBAL,
                target_addend: 0,
                target_key: "target_static".to_string(),
            }]);

        let storage = relocated_initializer_storage_type(&mut ctx, 8, 1, &encoded)
            .expect("underaligned allocation should use packed storage");
        let storage_ref = storage.deref(&ctx);
        let struct_ty = storage_ref
            .downcast_ref::<StructType>()
            .expect("relocated initializer must use struct storage");
        assert_eq!(struct_ty.layout(), StructLayout::Packed);
        assert_eq!(get_type_size(&ctx, storage), 8);
    }

    #[test]
    fn relocated_global_keeps_naturally_aligned_storage_unpacked() {
        let mut ctx = make_ctx();
        let encoded =
            llvm::encode_global_initializer_relocations(&[llvm::GlobalInitializerRelocation {
                source_offset: 8,
                width_bytes: 8,
                target_address_space: llvm_addr::GLOBAL,
                target_addend: 0,
                target_key: "target_static".to_string(),
            }]);

        let storage = relocated_initializer_storage_type(&mut ctx, 16, 8, &encoded)
            .expect("aligned relocation should keep ordinary storage");
        let storage_ref = storage.deref(&ctx);
        let struct_ty = storage_ref
            .downcast_ref::<StructType>()
            .expect("relocated initializer must use struct storage");
        assert_eq!(struct_ty.layout(), StructLayout::Unpacked);
    }

    #[test]
    fn relocated_global_rejects_overlapping_slots() {
        let mut ctx = make_ctx();
        let (module_ptr, block) = build_kernel(&mut ctx, vec![], vec![]);
        let word_ty: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let mir_global_ty: TypeHandle = MirArrayType::get(&mut ctx, word_ty, 2).into();
        let result_ty = MirPtrType::get_global(&mut ctx, mir_global_ty, false);
        let op = Operation::new(
            &mut ctx,
            mir::MirGlobalAllocOp::get_concrete_op_info(),
            vec![result_ty.into()],
            vec![],
            vec![],
            0,
        );
        let alloc = mir::MirGlobalAllocOp::new(op);
        alloc.set_attr_global_type(&ctx, TypeAttr::new(mir_global_ty));
        alloc.set_attr_global_key(&ctx, StringAttr::new("overlap".to_string()));
        alloc.set_alignment_value(&mut ctx, 8);
        op.deref_mut(&ctx).attributes.set(
            "global_initializer_hex".try_into().unwrap(),
            StringAttr::new("00000000000000000000000000000000".to_string()),
        );
        let encoded = llvm::encode_global_initializer_relocations(&[
            llvm::GlobalInitializerRelocation {
                source_offset: 0,
                width_bytes: 8,
                target_address_space: llvm_addr::GLOBAL,
                target_addend: 0,
                target_key: "a".to_string(),
            },
            llvm::GlobalInitializerRelocation {
                source_offset: 0,
                width_bytes: 8,
                target_address_space: llvm_addr::GLOBAL,
                target_addend: 0,
                target_key: "b".to_string(),
            },
        ]);
        op.deref_mut(&ctx).attributes.set(
            "global_initializer_relocations".try_into().unwrap(),
            StringAttr::new(encoded),
        );
        op.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        let error = crate::lower_mir_to_llvm(&mut ctx, module_ptr)
            .expect_err("overlapping relocations must fail closed");
        assert!(error.to_string().contains("overlaps"), "{error}");
    }
}
