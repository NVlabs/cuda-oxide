/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Type conversion from `dialect-mir` types to `dialect-llvm` types.
//!
//! This module handles the translation of `dialect-mir` type representations
//! to their `dialect-llvm` equivalents. Type conversion is foundational to
//! the lowering pass—most operation converters depend on it.
//!
//! # Overview
//!
//! `dialect-mir` types are high-level, Rust-like types that preserve semantic
//! information (signedness, slice semantics, etc.). `dialect-llvm` types are
//! lower-level and match LLVM IR types directly.
//!
//! # Type Mapping Table
//!
//! | `dialect-mir` Type              | `dialect-llvm` Type               | Notes                       |
//! |---------------------------------|-----------------------------------|-----------------------------|
//! | `IntegerType` (signed/unsigned) | `IntegerType` (signless)          | Width preserved             |
//! | `MirFP16Type`                   | `HalfType`                        | Rust `f16` → LLVM `half`    |
//! | `FP32Type`, `FP64Type`          | Same (builtin)                    | Pass-through                |
//! | `MirPtrType`                    | `PointerType`                     | Address space preserved     |
//! | `MirSliceType`                  | `StructType { ptr, i64 }`         | Fat pointer                 |
//! | `MirDisjointSliceType`          | `StructType { ptr, i64 }`         | Same as slice               |
//! | `MirTupleType`                  | `StructType`                      | Empty tuple → empty struct  |
//! | `MirStructType`                 | `StructType`                      | Fields recursively converted|
//! | `MirEnumType`                   | `StructType { discr, fields... }` | Discriminant + all fields   |
//! | `ArrayType`                     | `ArrayType`                       | Element type converted      |
//! | `VectorType`                    | `VectorType`                      | Element type converted      |
//!
//! # Signedness Handling
//!
//! LLVM IR integers are signless—the signedness is encoded in the operations
//! that use them (e.g., `sdiv` vs `udiv`). During type conversion:
//!
//! - Signed/unsigned MIR integers → signless LLVM integers
//! - The original signedness is preserved in operations (see `arithmetic.rs`)
//!
//! # Address Space Handling
//!
//! GPU memory uses address spaces to distinguish memory types:
//!
//! | Address Space | Memory Type | Usage                     |
//! |---------------|-------------|---------------------------|
//! | 0             | Generic     | Can point to any memory   |
//! | 1             | Global      | Device memory (VRAM)      |
//! | 3             | Shared      | Per-block shared memory   |
//! | 4             | Constant    | Read-only device memory   |
//! | 5             | Local       | Per-thread stack/spill    |
//!
//! Pointer address spaces are preserved through conversion. Slice types use
//! generic address space (0) because they can point to any memory type.
//!
//! # Slice Type Representation
//!
//! Rust slices (`&[T]`) are represented as fat pointers in LLVM:
//!
//! ```text
//! MIR: MirSliceType<f32>
//! LLVM: struct { ptr, i64 }  ; pointer + length
//! ```
//!
//! This matches the Rust ABI for slices passed by value.
//!
//! # Enum Type Representation
//!
//! Rust enums are represented as structs with discriminant + payload:
//!
//! ```text
//! MIR: MirEnumType { discriminant: i8, variants: [A(), B(i32)] }
//! LLVM: struct { i8, i32 }  ; discriminant + max payload size
//! ```
//!
//! All variant payloads are included in the struct, sized for the largest.
//!
//! # Function Type Conversion
//!
//! Function types undergo ABI transformations:
//!
//! - Slice arguments are flattened to `(ptr, len)` pairs
//! - Struct arguments are flattened to individual fields
//! - Empty tuple return type becomes void
//!
//! This matches the C ABI for GPU kernels.

use dialect_llvm::types as llvm_types;
use dialect_mir::types::{MirDisjointSliceType, MirSliceType, MirStructType};
use pliron::builtin::type_interfaces::FunctionTypeInterface;
use pliron::builtin::types::{FP32Type, FP64Type, FunctionType, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::operation::Operation;
use pliron::r#type::{TypeObj, type_cast};

use crate::type_conversion_interface::MirTypeConversion;

// =============================================================================
// Kernel-Boundary Detection
// =============================================================================

/// Identifier of the attribute that marks a `MirFuncOp` / `llvm.func` as a
/// GPU kernel entry point.
///
/// Kept as a function (rather than a `const`) because pliron `Identifier`
/// construction needs the `try_into()` fallible path.
fn gpu_kernel_attr() -> pliron::identifier::Identifier {
    "gpu_kernel".try_into().expect("static identifier")
}

/// Returns `true` when `op` carries the `gpu_kernel` attribute.
///
/// The kernel-entry ABI differs from internal device-function ABI: at
/// kernel boundaries, aggregate parameters (structs, closures) are passed
/// as a single byval value to match what the host pushes via
/// `cuLaunchKernel`. Internal call sites still flatten aggregates the
/// same way they always did. This helper is the single source of truth
/// for that branch and is consumed by both [`convert_function_type`] and
/// the entry-block prologue in `lowering.rs`.
pub fn is_kernel_func(ctx: &Context, op: Ptr<Operation>) -> bool {
    op.deref(ctx).attributes.0.contains_key(&gpu_kernel_attr())
}

// =============================================================================
// Zero-Sized Type (ZST) Detection
// =============================================================================

/// Check if a type is zero-sized (empty struct).
///
/// Zero-sized types include:
/// - Empty structs `struct {}`
/// - PhantomData markers (which become empty structs in MIR)
/// - Structs where all fields are themselves zero-sized
///
/// # Why This Matters
///
/// LLVM's NVPTX backend doesn't support empty struct types in function
/// signatures. We strip these during type conversion to avoid:
/// `LLVM ERROR: Empty parameter types are not supported`
///
/// # Background
///
/// Rust's `#[inline(always)]` attribute is stored in `codegen_fn_attrs`, which
/// is not exposed through the stable_mir API. Since we intercept MIR and generate
/// our own LLVM IR, we don't propagate inline hints. When LLVM decides not to
/// inline a function, the empty struct parameters/returns cause NVPTX to crash.
///
/// By stripping ZSTs at the LLVM type level, we avoid this issue regardless of
/// inlining decisions.
pub fn is_zero_sized_type(ctx: &Context, ty: Ptr<TypeObj>) -> bool {
    // Check if LLVM StructType with zero fields
    if let Some(struct_ty) = ty.deref(ctx).downcast_ref::<llvm_types::StructType>() {
        let num_fields = struct_ty.num_fields();
        if num_fields == 0 {
            return true;
        }
        // Also check if ALL fields are zero-sized (nested PhantomData)
        return struct_ty.fields().all(|f| is_zero_sized_type(ctx, f));
    }
    false
}

// =============================================================================
// Type Conversion
// =============================================================================

/// Convert a `dialect-mir` type to its `dialect-llvm` equivalent.
///
/// Dispatches via `MirTypeConversion` type interface — each supported type
/// registers a converter function pointer through `#[type_interface_impl]`
/// in [`super::type_interface_impls`].
///
/// The function-pointer indirection avoids a borrow-checker conflict:
/// `type_cast` borrows `ctx` immutably, but conversion needs `&mut ctx`.
/// We extract the `Copy` function pointer, drop the borrow, then call it.
pub fn convert_type(ctx: &mut Context, ty: Ptr<TypeObj>) -> Result<Ptr<TypeObj>, anyhow::Error> {
    // Phase 1: extract a Copy function pointer while ctx is immutably borrowed.
    let converter_fn = {
        let ty_ref = ty.deref(ctx);
        type_cast::<dyn MirTypeConversion>(&**ty_ref).map(|conv| conv.converter())
    };
    // Phase 2: borrow dropped — ctx is free for &mut.
    if let Some(conv_fn) = converter_fn {
        return conv_fn(ty, ctx);
    }

    let type_display = ty.deref(ctx).disp(ctx).to_string();
    Err(anyhow::anyhow!(
        "Unsupported type conversion: {}\n\
         Supported: integers, fp32, fp64, pointers, slices, tuples, structs, enums, arrays, vectors.",
        type_display
    ))
}

/// Convert a MIR function type to an LLVM function type.
///
/// This handles the ABI-level transformations required for GPU kernels.
/// The transformations ensure that the generated LLVM IR matches the
/// C ABI expected by the CUDA runtime.
///
/// # ABI Transformations
///
/// ## Argument Flattening
///
/// Aggregate types are flattened to primitive types:
///
/// ```text
/// MIR:  fn kernel(slice: &[f32], point: Point)
/// LLVM: fn internal_fn(ptr: !ptr, len: i64, x: f32, y: f32)
/// ```
///
/// | MIR Argument            | Internal call ABI       | Kernel-entry ABI       |
/// |-------------------------|-------------------------|------------------------|
/// | `&[T]`                  | `(ptr, i64)`            | `(ptr, i64)`           |
/// | `DisjointSlice<T>`      | `(ptr, i64)`            | `(ptr, i64)`           |
/// | `struct { a: A, b: B }` | `(a: A', b: B')`        | one byval `{A', B'}`   |
/// | closure with N captures | N separate field args   | one byval struct       |
/// | Other                   | Converted type          | Converted type         |
///
/// Slices keep their `(ptr, len)` flattening on both sides because the
/// host-side launch helpers push the pointer and length as two driver
/// args. Structs and closures are unflattened only at kernel boundaries
/// because the host pushes them as a single scalar — see
/// `cuda_host::push_kernel_scalar`. Internal device-side call sites stay
/// flattened: caller and callee are both inside this backend, so the ABI
/// is private and there is no host to disagree with.
///
/// ## Return Type Handling
///
/// - Empty tuple `()` becomes `void`
/// - Empty struct `struct {}` becomes `void`
/// - Other types are converted normally
///
/// # Arguments
///
/// * `ctx` - The pliron context
/// * `func_type` - The MIR function type to convert
/// * `is_kernel_entry` - When `true`, treat aggregate (non-slice) params
///   as single byval values to match the host-side push ABI. When `false`,
///   keep the existing internal device-fn ABI that flattens struct fields
///   into individual scalars.
///
/// # Returns
///
/// The equivalent LLVM function type with ABI transformations applied.
///
/// # Example
///
/// ```text
/// MIR:  fn foo(a: &[f32], b: i32) -> f32
/// LLVM: fn foo(ptr, i64, i32) -> f32
///
/// MIR:  fn bar() -> ()
/// LLVM: fn bar() -> void
/// ```
///
/// # Note
///
/// At internal device-function boundaries the struct flattening must be
/// reversed in the entry block. At kernel-entry boundaries the param
/// arrives as a single byval struct, so the entry block can pass it
/// through unchanged. See `lowering.rs::build_entry_prologue` for both
/// reconstruction paths.
pub fn convert_function_type(
    ctx: &mut Context,
    func_type: pliron::r#type::TypePtr<FunctionType>,
    is_kernel_entry: bool,
) -> Result<pliron::r#type::TypePtr<llvm_types::FuncType>, anyhow::Error> {
    // Extract input/output types before mutating context
    let (inputs_ptr, results_ptr) = {
        let func_ty_ref = func_type.deref(ctx);
        let interface = type_cast::<dyn FunctionTypeInterface>(&*func_ty_ref)
            .ok_or_else(|| anyhow::anyhow!("Type does not implement FunctionTypeInterface"))?;
        (interface.arg_types(), interface.res_types())
    };

    // Convert inputs, flattening slice/struct types for ABI compatibility.
    // Slices flatten on both ABIs; structs flatten only on the internal
    // device-fn ABI.
    let mut inputs = Vec::new();
    let inputs_vec: Vec<_> = inputs_ptr.to_vec();

    for t in inputs_vec {
        // Determine what kind of flattening this type needs
        // Extract all info first, then drop the borrow
        enum FlattenKind {
            Slice,
            Struct {
                field_types: Vec<Ptr<TypeObj>>,
                mem_to_decl: Vec<usize>,
            },
            None,
        }

        let flatten_kind = {
            let ty_ref = t.deref(ctx);
            if ty_ref.is::<MirSliceType>() || ty_ref.is::<MirDisjointSliceType>() {
                FlattenKind::Slice
            } else if let Some(struct_ty) = ty_ref.downcast_ref::<MirStructType>() {
                if is_kernel_entry {
                    // Kernel-boundary ABI: keep the struct intact so the
                    // host's single `push_kernel_scalar(&closure)` push
                    // matches a single .param entry on the device side.
                    FlattenKind::None
                } else {
                    FlattenKind::Struct {
                        field_types: struct_ty.field_types.clone(),
                        mem_to_decl: struct_ty.memory_order(),
                    }
                }
            } else {
                FlattenKind::None
            }
        };

        match flatten_kind {
            FlattenKind::Slice => {
                let ptr_ty = llvm_types::PointerType::get_generic(ctx);
                let len_ty = IntegerType::get(ctx, 64, Signedness::Signless);
                inputs.push(ptr_ty.into());
                inputs.push(len_ty.into());
            }
            FlattenKind::Struct {
                field_types,
                mem_to_decl,
            } => {
                // Flatten in MEMORY ORDER to match struct layout
                for mem_idx in 0..field_types.len() {
                    let decl_idx = mem_to_decl[mem_idx];
                    let converted = convert_type(ctx, field_types[decl_idx])?;
                    // Skip ZST fields - NVPTX can't handle empty params
                    if !is_zero_sized_type(ctx, converted) {
                        inputs.push(converted);
                    }
                }
            }
            FlattenKind::None => {
                let converted = convert_type(ctx, t)?;
                // Skip ZST args - NVPTX can't handle empty params
                if !is_zero_sized_type(ctx, converted) {
                    inputs.push(converted);
                }
            }
        }
    }

    // Convert return type, treating empty tuple/struct as void
    let ret_ty = if results_ptr.is_empty() {
        llvm_types::VoidType::get(ctx).into()
    } else {
        let ty = convert_type(ctx, results_ptr[0])?;
        // Check if zero-sized (empty struct or struct with only ZST fields)
        // Note: convert_type already strips ZST fields, so we just check for empty
        if is_zero_sized_type(ctx, ty) {
            llvm_types::VoidType::get(ctx).into()
        } else {
            ty
        }
    };

    Ok(llvm_types::FuncType::get(ctx, ret_ty, inputs, false))
}

/// Build an LLVM struct with explicit padding to match rustc's exact layout.
///
/// This ensures perfect ABI compatibility between host and device by using
/// padding arrays `[N x i8]` to place fields at their exact offsets.
///
/// # Why This Matters
///
/// LLVM's datalayout string controls how it computes struct field offsets.
/// If the host uses different alignment rules than our datalayout, the struct
/// layout would differ. By using explicit padding, we match rustc's layout
/// exactly, regardless of LLVM's datalayout.
///
/// # Example
///
/// For `struct Extreme { a: u8, b: i128 }` where rustc computes:
/// - field `b` at offset 0 (16 bytes)
/// - field `a` at offset 16 (1 byte)
/// - total size: 32 bytes (aligned to 16)
///
/// We generate:
/// ```text
/// { i128, i8, [15 x i8] }  ; b at 0, a at 16, padding to 32
/// ```
///
/// # Arguments
///
/// * `ctx` - The pliron context
/// * `field_types` - Field types in declaration order
/// * `mem_to_decl` - Memory order: mem_to_decl[mem_idx] = decl_idx
/// * `field_offsets` - Byte offset of each field in declaration order
/// * `total_size` - Total struct size in bytes
pub(crate) fn build_struct_with_explicit_padding(
    ctx: &mut Context,
    field_types: &[Ptr<TypeObj>],
    mem_to_decl: &[usize],
    field_offsets: &[u64],
    total_size: u64,
) -> Result<Ptr<TypeObj>, anyhow::Error> {
    let mut llvm_fields: Vec<Ptr<TypeObj>> = Vec::new();
    let mut current_offset: u64 = 0;

    // Process fields in memory order
    for mem_idx in 0..field_types.len() {
        let decl_idx = mem_to_decl[mem_idx];
        let field_ty = field_types[decl_idx];
        let target_offset = field_offsets[decl_idx];

        // Insert padding if needed to reach the target offset
        if current_offset < target_offset {
            let padding_size = target_offset - current_offset;
            let padding_ty = make_padding_type(ctx, padding_size);
            llvm_fields.push(padding_ty);
            current_offset = target_offset;
        }

        // Convert and add the field
        let llvm_ty = convert_type(ctx, field_ty)?;

        // Skip ZST fields (PhantomData) - they have no size
        if is_zero_sized_type(ctx, llvm_ty) {
            continue;
        }

        llvm_fields.push(llvm_ty);

        // Advance offset by field size
        let field_size = get_type_size(ctx, llvm_ty);
        current_offset += field_size;
    }

    // Add trailing padding to reach total_size
    if current_offset < total_size {
        let trailing_padding = total_size - current_offset;
        let padding_ty = make_padding_type(ctx, trailing_padding);
        llvm_fields.push(padding_ty);
    }

    Ok(llvm_types::StructType::get_unnamed(ctx, llvm_fields).into())
}

/// Create a padding type: `[N x i8]` for N bytes of padding.
fn make_padding_type(ctx: &mut Context, size: u64) -> Ptr<TypeObj> {
    let i8_ty = IntegerType::get(ctx, 8, Signedness::Signless);
    llvm_types::ArrayType::get(ctx, i8_ty.into(), size).into()
}

/// Get the size of an LLVM type in bytes (approximate).
///
/// This is used for computing padding. For most types we know the exact size;
/// for complex types we make reasonable assumptions.
fn get_type_size(ctx: &Context, ty: Ptr<TypeObj>) -> u64 {
    let ty_ref = ty.deref(ctx);

    // Integer types
    if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        return (int_ty.width() as u64).div_ceil(8); // Round up to bytes
    }

    // Float types
    if ty_ref.is::<llvm_types::HalfType>() {
        return 2;
    }
    if ty_ref.is::<FP32Type>() {
        return 4;
    }
    if ty_ref.is::<FP64Type>() {
        return 8;
    }

    // Pointer types (64-bit)
    if ty_ref.is::<llvm_types::PointerType>() {
        return 8;
    }

    // Array types
    if let Some(arr_ty) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
        let elem_size = get_type_size(ctx, arr_ty.elem_type());
        return elem_size * arr_ty.size();
    }

    // Struct types (sum of field sizes - approximation)
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
        return struct_ty.fields().map(|f| get_type_size(ctx, f)).sum();
    }

    // Default fallback - shouldn't happen for well-formed types
    8
}

/// Create the LLVM struct type used for slice representations.
///
/// Slices are represented as fat pointers: `{ ptr, i64 }` where:
/// - `ptr` is a generic address space (0) pointer to the data
/// - `i64` is the number of elements (not bytes)
///
/// # Layout
///
/// ```text
/// struct {
///     ptr: !llvm.ptr,     ; offset 0, size 8
///     len: i64,           ; offset 8, size 8
/// }                       ; total size: 16 bytes
/// ```
///
/// # Address Space
///
/// The pointer uses generic address space (0) because:
/// - Slices passed to kernels may point to global memory
/// - The kernel doesn't know at compile time which memory space
/// - Generic pointers can be used with any memory type
///
/// # Usage
///
/// This type is used for:
/// - `&[T]` slice arguments
/// - `DisjointSlice<T>` (unique-ownership slice) arguments
/// - Any other fat pointer representation
pub(crate) fn make_slice_struct(ctx: &mut Context) -> Ptr<TypeObj> {
    let ptr_ty = llvm_types::PointerType::get_generic(ctx);
    let len_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    llvm_types::StructType::get_unnamed(ctx, vec![ptr_ty.into(), len_ty.into()]).into()
}

#[cfg(test)]
mod tests {
    // TODO (npasham): Add unit tests for type conversion
}
