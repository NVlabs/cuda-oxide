/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Memory operation conversion: `dialect-mir` → `dialect-llvm`.
//!
//! Converts `dialect-mir` memory operations to their `dialect-llvm` equivalents.
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
use crate::convert::types::convert_type;
use crate::helpers;
use dialect_llvm::ops as llvm;
use dialect_llvm::types::ArrayType;
use dialect_mir::types::MirPtrType;
use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeObj, Typed};

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
    rewriter.insert_operation(ctx, llvm_store.get_operation());
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
    rewriter.insert_operation(ctx, llvm_load.get_operation());
    rewriter.replace_operation(ctx, op, llvm_load.get_operation());

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
    rewriter.insert_operation(ctx, alloca.get_operation());
    let alloca_ptr = alloca.get_operation().deref(ctx).get_result(0);

    let store = llvm::StoreOp::new(ctx, operand, alloca_ptr);
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
        vec![dialect_llvm::ops::GepIndex::Value(offset)],
        elem_ty,
    )?;
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
            .0
            .get(&"elem_type".try_into().unwrap())
            .ok_or_else(|| {
                anyhow_to_pliron(anyhow::anyhow!(
                    "MirSharedAllocOp missing elem_type attribute"
                ))
            })?;
        let elem_type_attr = elem_type_attr
            .downcast_ref::<TypeAttr>()
            .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("elem_type is not a TypeAttr")))?;
        let mir_elem_type = elem_type_attr.get_type(ctx);

        let size_attr = op_ref
            .attributes
            .0
            .get(&"size".try_into().unwrap())
            .ok_or_else(|| {
                anyhow_to_pliron(anyhow::anyhow!("MirSharedAllocOp missing size attribute"))
            })?;
        let size_attr = size_attr
            .downcast_ref::<IntegerAttr>()
            .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("size is not an IntegerAttr")))?;
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
    mir_elem_type: Ptr<TypeObj>,
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
    global_op.set_address_space(ctx, dialect_llvm::types::address_space::SHARED);

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

    let (global_key, mir_global_type, alignment) = {
        let global_op = dialect_mir::ops::MirGlobalAllocOp::new(op);
        let op_ref = op.deref(ctx);

        let global_key_attr = op_ref
            .attributes
            .0
            .get(&"global_key".try_into().unwrap())
            .ok_or_else(|| {
                anyhow_to_pliron(anyhow::anyhow!(
                    "MirGlobalAllocOp missing global_key attribute"
                ))
            })?;
        let global_key_attr = global_key_attr
            .downcast_ref::<StringAttr>()
            .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("global_key is not a StringAttr")))?;
        let global_key = String::from((*global_key_attr).clone());

        let global_type_attr = op_ref
            .attributes
            .0
            .get(&"global_type".try_into().unwrap())
            .ok_or_else(|| {
                anyhow_to_pliron(anyhow::anyhow!(
                    "MirGlobalAllocOp missing global_type attribute"
                ))
            })?;
        let global_type_attr = global_type_attr
            .downcast_ref::<TypeAttr>()
            .ok_or_else(|| anyhow_to_pliron(anyhow::anyhow!("global_type is not a TypeAttr")))?;
        let mir_global_type = global_type_attr.get_type(ctx);

        let alignment = global_op.get_alignment_value(ctx).unwrap_or(0);

        (global_key, mir_global_type, alignment)
    };

    let global_name = if let Some(existing_name) = device_globals.get(&global_key) {
        existing_name.clone()
    } else {
        create_device_global(
            ctx,
            op,
            device_globals,
            &global_key,
            mir_global_type,
            alignment,
        )?
    };

    let address_of_op =
        llvm::AddressOfOp::new(ctx, global_name, dialect_llvm::types::address_space::GLOBAL);
    rewriter.insert_operation(ctx, address_of_op.get_operation());
    rewriter.replace_operation(ctx, op, address_of_op.get_operation());

    Ok(())
}

fn create_device_global(
    ctx: &mut Context,
    op: Ptr<Operation>,
    device_globals: &mut DeviceGlobalsMap,
    global_key: &str,
    mir_global_type: Ptr<TypeObj>,
    alignment: u64,
) -> Result<pliron::identifier::Identifier> {
    let llvm_global_type = convert_type(ctx, mir_global_type).map_err(anyhow_to_pliron)?;

    static DEVICE_GLOBAL_COUNTER: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);
    let counter = DEVICE_GLOBAL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let name: pliron::identifier::Identifier =
        format!("__device_global_{counter}").try_into().unwrap();

    let global_op = if alignment > 0 {
        llvm::GlobalOp::new_with_alignment(ctx, name.clone(), llvm_global_type, alignment)
    } else {
        llvm::GlobalOp::new(ctx, name.clone(), llvm_global_type)
    };
    global_op.set_address_space(ctx, dialect_llvm::types::address_space::GLOBAL);

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
    device_globals.insert(global_key.to_string(), name.clone());

    Ok(name)
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
            vec![dialect_llvm::ops::GepIndex::Value(offset_value)],
            i8_ty.into(),
        )?;
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
    global_op.set_address_space(ctx, dialect_llvm::types::address_space::SHARED);

    {
        use dialect_llvm::attributes::LinkageAttr;
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
    // TODO (npasham): Add unit tests for memory conversion
}
