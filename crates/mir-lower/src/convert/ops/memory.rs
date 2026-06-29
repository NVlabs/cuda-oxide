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
//! Dynamic shared memory uses a per-kernel symbol (`__dynamic_smem_{kernel_name}`).
//! Key characteristics:
//!
//! - **Per-kernel symbols**: Each kernel gets its own extern shared symbol
//! - **Pre-computed alignment**: A pre-pass scans all `DynamicSharedArray` calls in a kernel
//!   to determine the maximum alignment before creating the global
//! - **Single pool per kernel**: All `DynamicSharedArray` calls within a kernel share the
//!   same runtime pool (sized by `shared_mem_bytes` at launch)
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
use crate::convert::types::{convert_type, get_type_size, validate_initialized_global_layout};
use crate::helpers;
use dialect_mir::types::MirPtrType;
use llvm_export::attributes::IntegerOverflowFlagsAttr;
use llvm_export::op_interfaces::IntBinArithOpWithOverflowFlag;
use llvm_export::ops as llvm;
use llvm_export::ops::GlobalOpExt;
use llvm_export::types::{ArrayType, FuncType, VoidType};
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
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    let (ptr, val) = match operands.as_slice() {
        [ptr, val] => (*ptr, *val),
        _ => {
            return pliron::input_err_noloc!("Store operation requires exactly 2 operands");
        }
    };

    let llvm_store = llvm::StoreOp::new(ctx, val, ptr);
    if dialect_mir::ops::MirStoreOp::new(op).is_volatile(ctx) {
        llvm_export::ops::set_op_volatile(ctx, llvm_store.get_operation(), true);
    }
    copy_alignment(ctx, op, llvm_store.get_operation());
    rewriter.insert_operation(ctx, llvm_store.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
}

/// Copy the ABI alignment stamped on a MIR memory op onto its lowered LLVM op.
///
/// The alignment is stamped by the pre-pass in `lowering.rs` while types are
/// still MIR; this helper transfers it to the newly created LLVM op so the
/// exporter can emit `align N`.
fn copy_alignment(ctx: &mut Context, mir_op: Ptr<Operation>, llvm_op: Ptr<Operation>) {
    if let Some(align) = llvm_export::ops::op_alignment(ctx, mir_op) {
        llvm_export::ops::set_op_alignment(ctx, llvm_op, align);
    }
}

fn copy_debug_local_variable(ctx: &mut Context, mir_op: Ptr<Operation>, llvm_op: Ptr<Operation>) {
    if let Some(info) = llvm_export::ops::debug_local_variable(ctx, mir_op) {
        llvm_export::ops::set_debug_local_variable(ctx, llvm_op, info);
    }
    if let Some(scope) = llvm_export::ops::debug_local_source_scope(ctx, mir_op) {
        llvm_export::ops::set_debug_local_source_scope(ctx, llvm_op, scope);
    }
    if let Some((file, pos)) = llvm_export::ops::debug_local_declaration_location(ctx, mir_op) {
        llvm_export::ops::set_debug_local_declaration_location(
            ctx, llvm_op, file, pos.line, pos.column,
        );
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
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    let (dst, src, count) = match operands.as_slice() {
        [dst, src, count] => (*dst, *src, *count),
        _ => {
            return pliron::input_err_noloc!("Memcpy operation requires exactly 3 operands");
        }
    };

    let pointee = {
        let dst_ptr_ty = operands_info
            .lookup_most_recent_of_type::<MirPtrType>(ctx, dst)
            .ok_or_else(|| {
                pliron::create_error!(
                    op.deref(ctx).loc(),
                    pliron::result::ErrorKind::VerificationFailed,
                    pliron::result::StringError(
                        "Memcpy destination must be a MIR pointer before lowering".to_string()
                    )
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
            pliron::result::StringError("Memcpy operation has no parent block".to_string())
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
    let intrinsic_name = format!("llvm_memcpy_p{dst_as}_p{src_as}_i{len_bits}");
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
    rewriter.insert_operation(ctx, call.get_operation());
    rewriter.erase_operation(ctx, op);
    Ok(())
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
    let llvm_ty = convert_type(ctx, result_ty).map_err(anyhow_to_pliron)?;

    let llvm_load = llvm::LoadOp::new(ctx, ptr, llvm_ty);
    if dialect_mir::ops::MirLoadOp::new(op).is_volatile(ctx) {
        llvm_export::ops::set_op_volatile(ctx, llvm_load.get_operation(), true);
    }
    copy_alignment(ctx, op, llvm_load.get_operation());
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
    copy_alignment(ctx, op, alloca.get_operation());
    copy_debug_local_variable(ctx, op, alloca.get_operation());
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
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operand = op.deref(ctx).get_operand(0);
    let operand_ty = operand.get_type(ctx);

    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let one_apint =
        pliron::utils::apint::APInt::from_i64(1, std::num::NonZeroUsize::new(32).unwrap());
    let one_attr = pliron::builtin::attributes::IntegerAttr::new(i32_ty, one_apint);
    let one_const = llvm::ConstantOp::new(ctx, one_attr.into());
    rewriter.insert_operation(ctx, one_const.get_operation());
    let one_val = one_const.get_operation().deref(ctx).get_result(0);

    let alloca = llvm::AllocaOp::new(ctx, operand_ty, one_val);
    // Propagate alignment stamped by the pre-pass (covers repr(align(N))
    // structs). Without this, the synthesised alloca would be under-aligned
    // relative to any loads/stores that claim the struct's true alignment.
    copy_alignment(ctx, op, alloca.get_operation());
    rewriter.insert_operation(ctx, alloca.get_operation());
    let alloca_ptr = alloca.get_operation().deref(ctx).get_result(0);

    let store = llvm::StoreOp::new(ctx, operand, alloca_ptr);
    copy_alignment(ctx, op, store.get_operation());
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
) -> Result<()> {
    use pliron::builtin::attributes::{IntegerAttr, TypeAttr};

    let (alloc_key, mir_elem_type, size, alignment) = {
        let shared_alloc_op = dialect_mir::ops::MirSharedAllocOp::new(op);
        let op_ref = op.deref(ctx);

        let alloc_key: Option<String> = shared_alloc_op
            .get_attr_alloc_key(ctx)
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

        (alloc_key, mir_elem_type, size, alignment)
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
            mir_elem_type,
            size,
            alignment,
            alloc_key,
        )?
    };

    let address_of_op = llvm::AddressOfOp::new(ctx, global_name, 3);
    rewriter.insert_operation(ctx, address_of_op.get_operation());
    rewriter.replace_operation(ctx, op, address_of_op.get_operation());

    Ok(())
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
/// `alloc_key` is `Some`, the key is moved into `shared_globals` so that
/// later allocations with the same key reuse this global (caller is
/// expected to have already checked the cache for a hit).
fn create_shared_global(
    ctx: &mut Context,
    op: Ptr<Operation>,
    shared_globals: &mut SharedGlobalsMap,
    mir_elem_type: TypeHandle,
    size: u64,
    alignment: u64,
    alloc_key: Option<String>,
) -> Result<pliron::identifier::Identifier> {
    let llvm_elem_type = convert_type(ctx, mir_elem_type).map_err(anyhow_to_pliron)?;
    let array_type = ArrayType::get(ctx, llvm_elem_type, size);

    static SHARED_COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let counter = SHARED_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let name: pliron::identifier::Identifier =
        format!("__shared_mem_{counter}").try_into().unwrap();

    let global_op = if alignment > 0 {
        llvm::GlobalOp::new_with_alignment(ctx, name.clone(), array_type.into(), alignment)
    } else {
        llvm::GlobalOp::new(ctx, name.clone(), array_type.into())
    };
    global_op.set_address_space(ctx, llvm_export::types::address_space::SHARED);

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

    if let Some(key) = alloc_key {
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
) -> Result<()> {
    use pliron::builtin::attributes::{StringAttr, TypeAttr};

    let (global_key, mir_global_type, alignment, addr_space, initializer_hex) = {
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
        )
    };

    let global_name = if let Some(existing_name) = device_globals.get(&global_key) {
        existing_name.clone()
    } else {
        create_device_global(
            ctx,
            op,
            device_globals,
            DeviceGlobalSpec {
                key: &global_key,
                mir_type: mir_global_type,
                alignment,
                addr_space,
                initializer_hex: initializer_hex.as_deref(),
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
}

fn create_device_global(
    ctx: &mut Context,
    op: Ptr<Operation>,
    device_globals: &mut DeviceGlobalsMap,
    spec: DeviceGlobalSpec<'_>,
) -> Result<pliron::identifier::Identifier> {
    // An explicit initializer is already the evaluated Rust allocation image.
    // Keep it as `[N x i8]` all the way through LLVM instead of rebuilding a
    // typed constant. Typed reconstruction can lose NaN payload bits and needs
    // a second, easily-divergent implementation of Rust struct padding.
    let semantic_llvm_type = convert_type(ctx, spec.mir_type).map_err(anyhow_to_pliron)?;
    let (llvm_global_type, alignment) = if let Some(initializer_hex) = spec.initializer_hex {
        let byte_count = initializer_hex_byte_count(initializer_hex).map_err(anyhow_to_pliron)?;
        if spec.alignment == 0 {
            return Err(anyhow_to_pliron(anyhow::anyhow!(
                "device global initializer is missing its evaluated Rust allocation alignment"
            )));
        }
        validate_initialized_global_layout(ctx, spec.mir_type, byte_count, spec.alignment)
            .map_err(anyhow_to_pliron)?;
        let i8_ty = IntegerType::get(ctx, 8, Signedness::Signless);
        (
            ArrayType::get(ctx, i8_ty.into(), byte_count).into(),
            spec.alignment,
        )
    } else {
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
            static DEVICE_GLOBAL_COUNTER: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);
            let counter = DEVICE_GLOBAL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            format!("__device_global_{counter}").try_into().unwrap()
        };

    let global_op = if alignment > 0 {
        llvm::GlobalOp::new_with_alignment(ctx, name.clone(), llvm_global_type, alignment)
    } else {
        llvm::GlobalOp::new(ctx, name.clone(), llvm_global_type)
    };
    global_op.set_address_space(ctx, spec.addr_space);
    if let Some(initializer_hex) = spec.initializer_hex {
        global_op.set_initializer_hex(ctx, initializer_hex);
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
/// # Per-Kernel Symbols
///
/// Each kernel gets its own dynamic shared memory symbol (`__dynamic_smem_{kernel_name}`).
/// This ensures explicit separation in the generated PTX.
///
/// # Alignment
///
/// The alignment is pre-computed during the lowering pre-pass. All
/// `DynamicSharedArray<T, ALIGN>` calls in a kernel share the same global, which
/// uses the maximum alignment requested by any call.
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

/// Get or create the extern shared memory global for a kernel.
///
/// Creates an LLVM global variable with:
/// - Zero-length array type: `[0 x i8]`
/// - External linkage (no initializer)
/// - Address space 3 (shared memory)
/// - Pre-computed maximum alignment from all DynamicSharedArray calls in the kernel
///
/// Each kernel gets its own dynamic shared memory symbol
/// (`__dynamic_smem_kernel_name`). Uses `shared_globals` for deduplication
/// (only one global per kernel).
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
                "Internal error: dynamic shared memory alignment not pre-computed for kernel '{}'. \
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
    use dialect_mir::types::MirPtrType;
    use llvm_export::op_interfaces::PointerTypeResult;
    use llvm_export::ops as llvm;
    use llvm_export::types::{PointerType, address_space as llvm_addr};
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

        let mut analyses = pliron::pass_manager::AnalysisManager::default();
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
        assert_eq!(llvm_type_size_align(&ctx, conv), (4, 4), "repr(u32) tag");

        // #[repr(usize)] fieldless: {i64}, 8 bytes.
        let repr_usize = make_enum_ty(&mut ctx, "ReprUsize", 64, unit_variants(4), 8, 8);
        let conv = convert_type(&mut ctx, repr_usize).unwrap();
        assert_eq!(llvm_type_size_align(&ctx, conv), (8, 8), "repr(usize) tag");

        // u8 tag but 8-byte rustc size (repr(align(8)) raise): the converted
        // struct must gain a trailing [7 x i8] pad to reach 8 bytes.
        let padded = make_enum_ty(&mut ctx, "Padded", 8, unit_variants(2), 8, 8);
        let conv = convert_type(&mut ctx, padded).unwrap();
        let (size, _align) = llvm_type_size_align(&ctx, conv);
        assert_eq!(size, 8, "trailing pad must raise the size to rustc's 8");
        {
            let conv_ref = conv.deref(&ctx);
            let struct_ty = conv_ref
                .downcast_ref::<llvm_export::types::StructType>()
                .expect("converted enum is a struct");
            assert_eq!(
                struct_ty.fields().count(),
                2,
                "tag + one trailing pad field; pad appended at the END"
            );
        }

        // u8 tag + i64 payload, rustc size 16: the slot map places the
        // payload at its rustc byte offset 8 behind an explicit
        // [7 x i8] filler, making the layout datalayout-independent.
        let i64_payload: TypeHandle = IntegerType::get(&ctx, 64, Signedness::Unsigned).into();
        let payload = make_enum_ty(
            &mut ctx,
            "OnePayload",
            8,
            vec![
                EnumVariant::new_with_offsets("A".to_string(), vec![i64_payload], vec![8]),
                EnumVariant::unit("B".to_string()),
            ],
            16,
            8,
        );
        let conv = convert_type(&mut ctx, payload).unwrap();
        let (size, _align) = llvm_type_size_align(&ctx, conv);
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
        assert_eq!(map.tag_slot, 0);
        assert_eq!(
            map.field_slots,
            vec![Some(1), Some(1)],
            "A.0 and B.0 overlap at byte 4 with the same type: one shared slot"
        );
        assert_eq!(
            llvm_type_size_align(&ctx, map.llvm_struct_ty),
            (8, 4),
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
                EnumVariant::new_with_offsets("A".to_string(), vec![u64_a], vec![0]),
                EnumVariant::new_with_offsets("B".to_string(), vec![u64_b], vec![0]),
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
        assert_eq!(map.tag_slot, 1, "tag claims its own slot at byte 8");
        let (size, _align) = llvm_type_size_align(&ctx, map.llvm_struct_ty);
        assert_eq!(size, 16, "{{ i64, i8, [7 x i8] }}");
    }

    /// Multi-payload enum whose variants overlap in Rust (8 bytes) but
    /// concatenate in our model (12 bytes structural): mimics
    /// `#[repr(u32)] enum E { A(u32), B(u32) }`.
    fn make_multi_payload_enum_ty(ctx: &mut Context) -> TypeHandle {
        let i32_a: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let i32_b: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        make_enum_ty(
            ctx,
            "MultiPayload",
            32,
            vec![
                EnumVariant::new_with_offsets("A".to_string(), vec![i32_a], vec![4]),
                EnumVariant::new_with_offsets("B".to_string(), vec![i32_b], vec![4]),
            ],
            8,
            4,
        )
    }

    /// Device-local GEP + load of a layout-divergent enum must LOWER: a
    /// non-kernel pointer is device-laid-out, so the structural
    /// `{tag, fields...}` model sizes both the writes and the reads
    /// consistently (issue #131's in-kernel `[E; 4]` arrays). Only kernel
    /// parameters (host-laid-out memory) reject divergent enums; see
    /// `kernel_param_accepts_multi_payload_enum`.
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
            .expect("device-local divergent enum GEP + load must lower");
    }

    /// A KERNEL parameter that carries a layout-divergent enum across the
    /// host/device ABI boundary must fail loudly: the host lays the data
    /// out with rustc's real (overlapped) layout while the device model
    /// concatenates payloads, so stride and field offsets disagree.
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

    /// `Option<&T>`-style enums store no tag on the host (Rust hides the
    /// variant inside the payload: null means None), but the device
    /// models them WITH an explicit tag, so their bytes disagree and
    /// they must be rejected at the kernel boundary. This pins the hole
    /// the narrowed guard closes: the old size-comparison guard skipped
    /// these enums entirely and let them through.
    #[test]
    fn kernel_param_rejects_niched_enum() {
        use pliron::builtin::attributes::StringAttr;

        let mut ctx = make_ctx();
        // The device's model of Option<&T>: an explicit u8 tag plus the
        // pointer payload, with total_size 0 ("layout not recorded"),
        // exactly what the importer builds for niche-encoded enums.
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
            .expect_err("niched enum kernel param must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("Option") && msg.contains("kernel boundary"),
            "error must name the enum and the kernel boundary, got: {msg}"
        );
        assert!(
            msg.contains("niche"),
            "error must explain the niche layout mismatch, got: {msg}"
        );
    }

    /// Build a `mir.shared_alloc` returning `MirPtrType<i32, addrspace=3>` of
    /// length `size`, with the given alloc_key, and append it to `block`.
    fn append_shared_alloc(ctx: &mut Context, block: Ptr<BasicBlock>, alloc_key: &str, size: u64) {
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
}
