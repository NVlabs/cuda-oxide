/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Type conversion from `dialect-mir` types to LLVM dialect types.
//!
//! This module handles the translation of `dialect-mir` type representations
//! to their LLVM dialect equivalents. Type conversion is foundational to
//! the lowering pass—most operation converters depend on it.
//!
//! # Overview
//!
//! `dialect-mir` types are high-level, Rust-like types that preserve semantic
//! information (signedness, slice semantics, etc.). LLVM dialect types are
//! lower-level and match LLVM IR types directly.
//!
//! # Type Mapping Table
//!
//! | `dialect-mir` Type              | LLVM dialect Type                 | Notes                       |
//! |---------------------------------|-----------------------------------|-----------------------------|
//! | `IntegerType` (signed/unsigned) | `IntegerType` (signless)          | Width preserved             |
//! | `MirFP16Type`                   | `HalfType`                        | Rust `f16` → LLVM `half`    |
//! | `FP32Type`, `FP64Type`          | Same (builtin)                    | Pass-through                |
//! | `MirPtrType`                    | `PointerType`                     | Address space preserved     |
//! | `MirSliceType`                  | `StructType { ptr, i64 }`         | Fat pointer                 |
//! | `MirDisjointSliceType`          | `StructType { ptr, i64 }`         | Same as slice               |
//! | `MirTupleType`                  | `StructType`                      | Empty tuple → empty struct  |
//! | `MirStructType`                 | `StructType`                      | Fields recursively converted|
//! | `MirEnumType`                   | `StructType` (rustc byte layout)  | See "Enum Type Representation" |
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
//! A Rust enum is one tag plus the payload of whichever variant is
//! alive; all variants share the same bytes. We build an LLVM struct
//! that puts the tag and every payload field at the exact byte position
//! rustc chose, inserting `[N x i8]` filler for the gaps:
//!
//! ```text
//! #[repr(u32)] enum E { A(u32), B(f32), C }   // rustc: 8 bytes,
//!                                             // tag at 0, payloads at 4
//! LLVM: { i32, i32 }   ; slot 0 = tag, slot 1 = A's payload
//!                      ; B's f32 also lives at byte 4 but has a
//!                      ; different type, so it is read/written through
//!                      ; memory instead of owning a slot
//! ```
//!
//! Because the bytes match rustc exactly, enum data can cross the
//! host/device boundary safely. The tag slot stores the variant's
//! DECLARED discriminant value (`enum E { A = 7 }` stores 7), not its
//! position. See `build_enum_slot_map` in this module for the full
//! story.
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

use dialect_mir::types::{
    MirDisjointSliceType, MirEnumType, MirSliceType, MirStructType, MirTupleType,
};
use llvm_export::types as llvm_types;
use llvm_export::types::PointerTypeExt;
use pliron::builtin::type_interfaces::FunctionTypeInterface;
use pliron::builtin::types::{FP32Type, FP64Type, FunctionType, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, type_cast};

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
    op.deref(ctx)
        .attributes
        .get::<pliron::builtin::attributes::StringAttr>(&gpu_kernel_attr())
        .is_some()
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
pub fn is_zero_sized_type(ctx: &Context, ty: TypeHandle) -> bool {
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

/// Convert a `dialect-mir` type to its LLVM dialect equivalent.
///
/// Dispatches via `MirTypeConversion` type interface — each supported type
/// registers a converter function pointer through `#[type_interface_impl]`
/// in [`super::type_interface_impls`].
///
/// The function-pointer indirection avoids a borrow-checker conflict:
/// `type_cast` borrows `ctx` immutably, but conversion needs `&mut ctx`.
/// We extract the `Copy` function pointer, drop the borrow, then call it.
pub fn convert_type(ctx: &mut Context, ty: TypeHandle) -> Result<TypeHandle, anyhow::Error> {
    // Phase 1: extract a Copy function pointer while ctx is immutably borrowed.
    let converter_fn = {
        let ty_ref = ty.deref(ctx);
        type_cast::<dyn MirTypeConversion>(&*ty_ref).map(|conv| conv.converter())
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
    func_type: pliron::r#type::TypedHandle<FunctionType>,
    is_kernel_entry: bool,
) -> Result<pliron::r#type::TypedHandle<llvm_types::FuncType>, anyhow::Error> {
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
                field_types: Vec<TypeHandle>,
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

// =============================================================================
// Struct Slot Mapping (single source of truth, issue #128)
// =============================================================================

/// Declaration-order layout facts for one MIR aggregate, in the exact form
/// [`build_struct_slot_map`] consumes.
///
/// Extracting this owned carrier first (and dropping the `Ref` returned by
/// `Ptr::deref`) keeps the borrow checker happy: the slot-map build needs
/// `&mut Context` for type interning.
pub(crate) struct StructLayoutInfo {
    /// Field types in declaration order.
    pub field_types: Vec<TypeHandle>,
    /// Memory order: `mem_to_decl[mem_idx] = decl_idx`. Always full length
    /// (identity when rustc did not reorder).
    pub mem_to_decl: Vec<usize>,
    /// Byte offset of each field in declaration order; empty when rustc
    /// layout is unknown.
    pub field_offsets: Vec<u64>,
    /// Total size in bytes including trailing padding; 0 when unknown.
    pub total_size: u64,
}

impl StructLayoutInfo {
    /// Layout facts of a `MirStructType`.
    pub(crate) fn of_struct(s: &MirStructType) -> Self {
        StructLayoutInfo {
            field_types: s.field_types.clone(),
            mem_to_decl: s.memory_order(),
            field_offsets: s.field_offsets().to_vec(),
            total_size: s.total_size(),
        }
    }

    /// Layout facts of a `MirTupleType`: identity order, no rustc layout.
    pub(crate) fn of_tuple(t: &MirTupleType) -> Self {
        let field_types = t.get_types().to_vec();
        let mem_to_decl = (0..field_types.len()).collect();
        StructLayoutInfo {
            field_types,
            mem_to_decl,
            field_offsets: vec![],
            total_size: 0,
        }
    }
}

/// One lowered LLVM struct plus the value-level slot mapping into it.
///
/// [`build_struct_slot_map`] produces the struct type and the index map in
/// the same walk, so every op that indexes into the struct (`insertvalue`,
/// `extractvalue`, GEP, call-boundary flatten/reconstruct) shares the type
/// converter's view of where each field landed. Computing the indices
/// separately is how the issue #128 class of bug (indices that ignore the
/// `[N x i8]` padding slots) happened.
pub(crate) struct StructSlotMap {
    /// The final LLVM struct type, including any `[N x i8]` padding slots.
    pub llvm_struct_ty: TypeHandle,
    /// `decl_to_llvm[decl_idx]` = LLVM slot of that declaration-order field;
    /// `None` when the field is zero-sized and was stripped.
    pub decl_to_llvm: Vec<Option<u32>>,
    /// Converted LLVM type of each declaration-order field (ZSTs included).
    pub field_llvm_types: Vec<TypeHandle>,
}

/// Lower a struct/tuple layout to its LLVM struct type and slot map.
///
/// When rustc layout is present (`field_offsets` non-empty and
/// `total_size > 0`), fields are placed at their exact byte offsets with
/// explicit `[N x i8]` padding slots in between, plus a trailing pad up to
/// `total_size`. This makes the layout independent of LLVM's datalayout
/// and so ABI-identical to what rustc computed on the host. For
/// `struct Extreme { a: u8, b: i128 }` where rustc puts `b` at offset 0
/// and `a` at offset 16 with total size 32, we build:
///
/// ```text
/// { i128, i8, [15 x i8] }   ; slots:  b = 0, a = 1, pad = 2
/// ```
///
/// Without rustc layout, fields are emitted in memory order with no
/// padding. On both paths zero-sized fields (e.g. `PhantomData`) are
/// stripped, because NVPTX rejects empty types; stripped fields get
/// `None` in `decl_to_llvm`.
///
/// Malformed layout metadata (a `mem_to_decl` that is not a permutation,
/// or an offsets vector of the wrong length) is rejected loudly: guessing
/// here would scramble every downstream field access.
pub(crate) fn build_struct_slot_map(
    ctx: &mut Context,
    layout: &StructLayoutInfo,
) -> Result<StructSlotMap, anyhow::Error> {
    let num_fields = layout.field_types.len();

    if layout.mem_to_decl.len() != num_fields {
        return Err(anyhow::anyhow!(
            "struct slot map: memory order has {} entries but the struct has {} fields",
            layout.mem_to_decl.len(),
            num_fields
        ));
    }
    let mut seen = vec![false; num_fields];
    for &decl_idx in &layout.mem_to_decl {
        if decl_idx >= num_fields || seen[decl_idx] {
            return Err(anyhow::anyhow!(
                "struct slot map: memory order {:?} is not a permutation of 0..{}",
                layout.mem_to_decl,
                num_fields
            ));
        }
        seen[decl_idx] = true;
    }
    let has_explicit_layout = !layout.field_offsets.is_empty() && layout.total_size > 0;
    if has_explicit_layout && layout.field_offsets.len() != num_fields {
        return Err(anyhow::anyhow!(
            "struct slot map: {} field offsets for {} fields",
            layout.field_offsets.len(),
            num_fields
        ));
    }

    // Convert every field up front, in declaration order.
    let mut field_llvm_types = Vec::with_capacity(num_fields);
    for &field_ty in &layout.field_types {
        field_llvm_types.push(convert_type(ctx, field_ty)?);
    }

    let mut llvm_fields: Vec<TypeHandle> = Vec::new();
    let mut decl_to_llvm: Vec<Option<u32>> = vec![None; num_fields];
    let mut current_offset: u64 = 0;

    // Place fields in memory order.
    for &decl_idx in &layout.mem_to_decl {
        let llvm_ty = field_llvm_types[decl_idx];

        // ZST fields are stripped: no slot, no offset advance (rustc gives
        // them size 0).
        if is_zero_sized_type(ctx, llvm_ty) {
            continue;
        }

        if has_explicit_layout {
            // Insert padding if needed to reach the rustc field offset.
            let target_offset = layout.field_offsets[decl_idx];
            if current_offset < target_offset {
                let padding_ty = make_padding_type(ctx, target_offset - current_offset);
                llvm_fields.push(padding_ty);
                current_offset = target_offset;
            }
        }

        decl_to_llvm[decl_idx] = Some(llvm_fields.len() as u32);
        llvm_fields.push(llvm_ty);

        if has_explicit_layout {
            // Prefer rustc's stored size for the field over the LLVM-level
            // approximation: nested aggregates carry interior/trailing
            // padding the converted type cannot always reproduce, and a
            // wrong advance here either forces interior padding where
            // rustc has none or overshoots the next field's offset.
            current_offset += mir_stored_size(ctx, layout.field_types[decl_idx])
                .unwrap_or_else(|| get_type_size(ctx, llvm_ty));
        }
    }

    // Add trailing padding to reach total_size.
    if has_explicit_layout && current_offset < layout.total_size {
        let padding_ty = make_padding_type(ctx, layout.total_size - current_offset);
        llvm_fields.push(padding_ty);
    }

    Ok(StructSlotMap {
        llvm_struct_ty: llvm_types::StructType::get_unnamed(ctx, llvm_fields).into(),
        decl_to_llvm,
        field_llvm_types,
    })
}

/// Create a padding type: `[N x i8]` for N bytes of padding.
fn make_padding_type(ctx: &mut Context, size: u64) -> TypeHandle {
    let i8_ty = IntegerType::get(ctx, 8, Signedness::Signless);
    llvm_types::ArrayType::get(ctx, i8_ty.into(), size).into()
}

/// Size of a MIR-level type from rustc layout truth, when stored.
///
/// `MirStructType` and `MirEnumType` carry `total_size` (interior and
/// trailing padding included) straight from rustc's layout query; arrays
/// of such aggregates multiply it out. Returns `None` when no stored size
/// is available (e.g. niched/single-variant enums store 0) and the caller
/// must fall back to the LLVM-level approximation.
fn mir_stored_size(ctx: &Context, mir_ty: TypeHandle) -> Option<u64> {
    let ty_ref = mir_ty.deref(ctx);
    if let Some(s) = ty_ref.downcast_ref::<MirStructType>() {
        if s.total_size() > 0 {
            return Some(s.total_size());
        }
        return None;
    }
    if let Some(e) = ty_ref.downcast_ref::<MirEnumType>() {
        if e.total_size() > 0 {
            return Some(e.total_size());
        }
        return None;
    }
    if let Some(a) = ty_ref.downcast_ref::<dialect_mir::types::MirArrayType>() {
        let elem_ty = a.element_ty;
        let size = a.size;
        return mir_stored_size(ctx, elem_ty).map(|elem_size| elem_size * size);
    }
    None
}

/// LLVM natural-layout `(size, align)` of an exported LLVM type, in bytes.
///
/// Mirrors LLVM's default data layout for nvptx64 (scalars align to their
/// size, arrays to their element, non-packed structs to their widest field).
/// Unlike [`get_type_size`], which sums struct fields without alignment,
/// this computes the real allocation size, which is what GEP striding and
/// the enum size check below need.
pub(crate) fn llvm_type_size_align(ctx: &Context, ty: TypeHandle) -> (u64, u64) {
    let ty_ref = ty.deref(ctx);

    if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        let size = (int_ty.width() as u64).div_ceil(8);
        // i8 → 1, i16 → 2, i32 → 4, i64 → 8, i128 → 16.
        return (size, size.next_power_of_two().min(16));
    }
    if ty_ref.is::<llvm_types::HalfType>() {
        return (2, 2);
    }
    if ty_ref.is::<FP32Type>() {
        return (4, 4);
    }
    if ty_ref.is::<FP64Type>() {
        return (8, 8);
    }
    if ty_ref.is::<llvm_types::PointerType>() {
        return (8, 8);
    }
    if let Some(arr_ty) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
        let (elem_size, elem_align) = llvm_type_size_align(ctx, arr_ty.elem_type());
        return (elem_size * arr_ty.size(), elem_align.max(1));
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
        let fields: Vec<_> = struct_ty.fields().collect();
        let (_end, size, align) = natural_struct_layout(ctx, &fields);
        return (size, align);
    }

    // Vector types and anything unrecognised: conservative 8-byte fallback,
    // matching get_type_size.
    (8, 8)
}

/// Natural (non-packed) LLVM struct layout over `fields`.
///
/// Returns `(end, size, align)` where `end` is the unrounded offset just past
/// the last field, `size` is `end` rounded up to the struct alignment (the
/// allocation size LLVM uses for GEP striding), and `align` is the widest
/// field alignment.
pub(crate) fn natural_struct_layout(ctx: &Context, fields: &[TypeHandle]) -> (u64, u64, u64) {
    let mut end = 0u64;
    let mut align = 1u64;
    for field in fields {
        let (field_size, field_align) = llvm_type_size_align(ctx, *field);
        let field_align = field_align.max(1);
        end = end.div_ceil(field_align) * field_align;
        end += field_size;
        align = align.max(field_align);
    }
    let size = end.div_ceil(align) * align;
    (end, size, align)
}

/// The LLVM struct for an enum, plus a map saying where the tag and each
/// payload field ended up.
///
/// The struct type and the indices into it are produced by one walk in
/// [`build_enum_slot_map`], so they can never disagree. (Computing them
/// separately is how the issue #128 class of bug happened for structs.)
pub(crate) struct EnumSlotMap {
    /// The final LLVM struct type, including any `[N x i8]` filler slots.
    pub llvm_struct_ty: TypeHandle,
    /// Which struct slot holds the tag.
    pub tag_slot: u32,
    /// Which struct slot holds each payload field, in the flattened
    /// order of `MirEnumType::all_field_types`. `None` means the field
    /// has no slot of its own: it is zero-sized, or its bytes are shared
    /// with a different-typed field of another variant. Such fields are
    /// read and written through memory at `field_offsets` instead.
    pub field_slots: Vec<Option<u32>>,
    /// Byte position of each payload field inside the enum (copied from
    /// the type; empty when the layout was not recorded).
    pub field_offsets: Vec<u64>,
    /// Converted LLVM type of each payload field.
    pub field_llvm_types: Vec<TypeHandle>,
}

/// Build the LLVM struct for an enum, placing everything at the byte
/// positions rustc chose.
///
/// Why this matters: the host (CPU) lays out enum values with rustc's
/// layout. If the device used different byte positions, every enum
/// passed to a kernel would be read wrong. So the device struct is built
/// to have the same bytes, position for position.
///
/// The wrinkle is that enum variants SHARE bytes (only one variant is
/// alive at a time), and an LLVM struct cannot say "these two fields
/// overlap". The slot map resolves each field one of three ways:
///
/// ```text
/// #[repr(u32)] enum E { A(u32), B(f32), C }
/// rustc: 8 bytes, tag at byte 0, A's u32 and B's f32 both at byte 4
///
/// LLVM struct: { i32, i32 }
///                 |     |
///        tag_slot=0     A's payload: own slot (nothing else typed i32
///                       wanted byte 4 first... B did, see below)
///
/// - own slot:   the field's bytes collide with nothing already placed.
/// - shared slot: another variant already placed the SAME type at the
///                SAME position; both map to that slot. (If B were
///                B(u32), A and B would simply share slot 1.)
/// - no slot:    the bytes are taken by a different type (B's f32 vs
///                A's u32 here). The field is still at byte 4, just not
///                nameable as a struct field; reads and writes go
///                through memory: spill the value to a stack slot, then
///                use a byte-precise pointer. No slot, but no lie.
/// ```
///
/// Gaps between placed fields, and the tail, are covered with explicit
/// `[N x i8]` filler so the struct's size is exactly rustc's no matter
/// what LLVM's own layout rules would have done.
///
/// Niche-encoded enums (`total_size == 0`, layout not recorded) keep the
/// old simple model instead: `{tag, all fields in order}`. That model is
/// only used inside kernels and never crosses the host boundary.
///
/// If the finished struct's size does not come out equal to rustc's,
/// something is deeply wrong and lowering would miscompile, so that is a
/// hard error rather than a debug assertion.
pub(crate) fn build_enum_slot_map(
    ctx: &mut Context,
    ty: TypeHandle,
) -> Result<EnumSlotMap, anyhow::Error> {
    let (
        name,
        discriminant_ty,
        all_field_types,
        all_field_offsets,
        tag_offset,
        total_size,
        abi_align,
    ) = {
        let ty_ref = ty.deref(ctx);
        let enum_ty = ty_ref
            .downcast_ref::<MirEnumType>()
            .ok_or_else(|| anyhow::anyhow!("build_enum_slot_map: expected MirEnumType"))?;
        (
            enum_ty.name().to_string(),
            enum_ty.discriminant_ty,
            enum_ty.all_field_types.clone(),
            enum_ty.all_field_offsets.clone(),
            enum_ty.tag_offset(),
            enum_ty.total_size(),
            enum_ty.abi_align(),
        )
    };

    let llvm_discr_ty = convert_type(ctx, discriminant_ty)?;
    let mut field_llvm_types = Vec::with_capacity(all_field_types.len());
    for &field_ty in &all_field_types {
        field_llvm_types.push(convert_type(ctx, field_ty)?);
    }

    if total_size == 0 {
        // Layout not recorded (niche-encoded shapes): keep the simple
        // {tag, all fields in order} struct. Fine inside a kernel, never
        // allowed across the host boundary.
        let mut llvm_fields = vec![llvm_discr_ty];
        llvm_fields.extend(field_llvm_types.iter().copied());
        let field_slots = (0..field_llvm_types.len())
            .map(|i| Some(1 + i as u32))
            .collect();
        return Ok(EnumSlotMap {
            llvm_struct_ty: llvm_types::StructType::get_unnamed(ctx, llvm_fields).into(),
            tag_slot: 0,
            field_slots,
            field_offsets: vec![],
            field_llvm_types,
        });
    }

    if all_field_offsets.len() != all_field_types.len() {
        return Err(anyhow::anyhow!(
            "enum slot map: `{}` has {} field offsets for {} fields",
            name,
            all_field_offsets.len(),
            all_field_types.len()
        ));
    }

    // Phase 1: decide who gets a struct slot. The tag goes first so a
    // payload field can never take its bytes.
    // claims: (byte position, byte size, converted type), no two overlap.
    let mut claims: Vec<(u64, u64, TypeHandle)> = Vec::new();
    let (tag_size, tag_align) = llvm_type_size_align(ctx, llvm_discr_ty);
    if tag_offset % tag_align.max(1) != 0 || tag_offset + tag_size > total_size {
        return Err(anyhow::anyhow!(
            "enum slot map: `{}` tag (size {}, align {}) cannot sit at byte {} of {}",
            name,
            tag_size,
            tag_align,
            tag_offset,
            total_size
        ));
    }
    claims.push((tag_offset, tag_size, llvm_discr_ty));
    let tag_claim: usize = 0;

    let mut claim_of_field: Vec<Option<usize>> = vec![None; field_llvm_types.len()];
    let mut order: Vec<usize> = (0..field_llvm_types.len()).collect();
    order.sort_by_key(|&i| (all_field_offsets[i], i));
    for flat in order {
        let llvm_ty = field_llvm_types[flat];
        let (size, align) = llvm_type_size_align(ctx, llvm_ty);
        if size == 0 || is_zero_sized_type(ctx, llvm_ty) {
            // ZSTs own no bytes and no slot.
            continue;
        }
        let offset = all_field_offsets[flat];
        if offset + size > total_size {
            return Err(anyhow::anyhow!(
                "enum slot map: `{}` field {} (size {}) at byte {} exceeds total size {}",
                name,
                flat,
                size,
                offset,
                total_size
            ));
        }
        // Another variant already placed the same type at the same
        // position? Then both fields can simply use that slot: variants
        // share bytes, and here they even agree on the type.
        if let Some(ci) = claims
            .iter()
            .position(|&(o, _, t)| o == offset && t == llvm_ty)
        {
            claim_of_field[flat] = Some(ci);
            continue;
        }
        // The bytes are taken by a different type, or the position is
        // not aligned for this type: no slot. The field keeps its byte
        // position and is accessed through memory instead.
        let collides = claims
            .iter()
            .any(|&(o, s, _)| offset < o + s && o < offset + size);
        if collides || offset % align.max(1) != 0 {
            continue;
        }
        claims.push((offset, size, llvm_ty));
        claim_of_field[flat] = Some(claims.len() - 1);
    }

    // Phase 2: lay the slots down in byte order, filling every gap (and
    // the tail) with [N x i8] so the struct's size is exactly rustc's.
    let mut emit_order: Vec<usize> = (0..claims.len()).collect();
    emit_order.sort_by_key(|&ci| claims[ci].0);
    let mut llvm_fields: Vec<TypeHandle> = Vec::new();
    let mut slot_of_claim: Vec<u32> = vec![0; claims.len()];
    let mut current_offset: u64 = 0;
    for &ci in &emit_order {
        let (offset, size, llvm_ty) = claims[ci];
        if current_offset < offset {
            llvm_fields.push(make_padding_type(ctx, offset - current_offset));
            current_offset = offset;
        }
        slot_of_claim[ci] = llvm_fields.len() as u32;
        llvm_fields.push(llvm_ty);
        current_offset += size;
    }
    if current_offset < total_size {
        llvm_fields.push(make_padding_type(ctx, total_size - current_offset));
    }

    // Sanity: the struct we just built must be exactly rustc's size.
    // Arrays of enums step by this size, so a mismatch means every
    // element after the first is read from the wrong place. That is a
    // guaranteed miscompile, hence a hard error, not a debug check.
    let (_end, natural_size, natural_align) = natural_struct_layout(ctx, &llvm_fields);
    if natural_size != total_size {
        return Err(anyhow::anyhow!(
            "enum slot map: `{}` lowered to {} bytes but rustc says {}",
            name,
            natural_size,
            total_size
        ));
    }
    debug_assert!(
        natural_align <= abi_align.max(1),
        "enum slot map: `{name}` natural align {natural_align} exceeds rustc's {abi_align}"
    );

    let field_slots = claim_of_field
        .into_iter()
        .map(|c| c.map(|ci| slot_of_claim[ci]))
        .collect();
    Ok(EnumSlotMap {
        llvm_struct_ty: llvm_types::StructType::get_unnamed(ctx, llvm_fields).into(),
        tag_slot: slot_of_claim[tag_claim],
        field_slots,
        field_offsets: all_field_offsets,
        field_llvm_types,
    })
}

/// Convert a `MirEnumType` to its LLVM struct representation.
///
/// Thin wrapper over [`build_enum_slot_map`], which explains the layout.
/// Any op that needs an index into the converted enum must take it from
/// the slot map, never compute it by hand.
pub(crate) fn convert_enum_to_llvm(
    ctx: &mut Context,
    ty: TypeHandle,
) -> Result<TypeHandle, anyhow::Error> {
    Ok(build_enum_slot_map(ctx, ty)?.llvm_struct_ty)
}

/// Is this an enum whose device bytes do NOT match the host's?
///
/// Most enums now lower byte-identically to rustc's layout and pass any
/// boundary freely. The exception is enums whose layout we deliberately
/// did not record (`total_size == 0`):
///
/// - Niche-encoded enums like `Option<&T>`. On the host, Rust stores no
///   tag at all; it reuses an impossible payload value (null, for a
///   never-null `&T`) to mean `None`. On the device we give such enums
///   an explicit tag instead, which the host bytes simply do not have.
/// - Multi-variant enums rustc reports as having a single live variant
///   (e.g. `Result<T, Infallible>`): same story, the device tag has no
///   host counterpart.
///
/// WHY the device differs at all: nothing in the hardware demands it.
/// With a real tag, "which variant?" is a one-field load and
/// "construct" is a one-field store, which is all our discriminant and
/// construct ops know how to be. With a niche there is no tag to load:
/// the discriminant must be COMPUTED from the payload (null check,
/// byte-range check, in general rustc's get_discr range arithmetic),
/// and constructing `None` means writing a magic payload value. That
/// per-enum decode/encode logic has not been ported yet, so the device
/// keeps a synthetic tag, which is fine while the bytes stay on the
/// device and a lie the moment they meet host memory. Porting the
/// niche logic is the follow-up that would erase this difference and
/// retire this check.
///
/// One-variant enums with `total_size == 0` are fine: there is nothing
/// for the two sides to disagree about.
///
/// Returns the enum's name when its bytes are unmodeled, else `None`.
pub(crate) fn enum_unmodeled_in_memory(ctx: &Context, ty: TypeHandle) -> Option<String> {
    let ty_ref = ty.deref(ctx);
    let enum_ty = ty_ref.downcast_ref::<MirEnumType>()?;
    (enum_ty.total_size() == 0 && enum_ty.variant_count() > 1).then(|| enum_ty.name().to_string())
}

/// Search a kernel parameter's type for an enum the host and device
/// would disagree about (see [`enum_unmodeled_in_memory`]).
///
/// The search looks everywhere host data can hide: behind pointers,
/// inside slices and arrays, in struct/tuple fields, and in other enums'
/// payloads. It returns the first offending enum's name.
///
/// Only kernel signatures are checked. A kernel parameter is host data
/// (passed by value at launch, or reachable through a `DeviceBuffer`
/// pointer), so its bytes must mean the same thing on both sides. The
/// same enum used purely INSIDE a kernel (locals, construct, match) is
/// fine and is deliberately not rejected: there, both reader and writer
/// are the device, using one consistent layout.
///
/// `visited` breaks cycles through recursive types (`TypeHandle` is
/// interned, so equality is identity).
pub(crate) fn find_unmodeled_enum_in_abi(
    ctx: &mut Context,
    ty: TypeHandle,
    visited: &mut Vec<TypeHandle>,
) -> Result<Option<String>, anyhow::Error> {
    if visited.contains(&ty) {
        return Ok(None);
    }
    visited.push(ty);

    if let Some(name) = enum_unmodeled_in_memory(ctx, ty) {
        return Ok(Some(name));
    }

    let children: Vec<TypeHandle> = {
        let ty_ref = ty.deref(ctx);
        if let Some(p) = ty_ref.downcast_ref::<dialect_mir::types::MirPtrType>() {
            vec![p.pointee]
        } else if let Some(s) = ty_ref.downcast_ref::<MirSliceType>() {
            vec![s.element_ty]
        } else if let Some(s) = ty_ref.downcast_ref::<MirDisjointSliceType>() {
            vec![s.element_ty]
        } else if let Some(a) = ty_ref.downcast_ref::<dialect_mir::types::MirArrayType>() {
            vec![a.element_ty]
        } else if let Some(s) = ty_ref.downcast_ref::<MirStructType>() {
            s.field_types.clone()
        } else if let Some(t) = ty_ref.downcast_ref::<MirTupleType>() {
            t.get_types().to_vec()
        } else if let Some(e) = ty_ref.downcast_ref::<MirEnumType>() {
            e.all_field_types.clone()
        } else {
            vec![]
        }
    };

    for child in children {
        if let Some(name) = find_unmodeled_enum_in_abi(ctx, child, visited)? {
            return Ok(Some(name));
        }
    }
    Ok(None)
}

/// Get the size of an LLVM type in bytes (approximate).
///
/// This is used for computing padding. For most types we know the exact
/// size. For structs the sum of field sizes is exact when the struct was
/// built with explicit padding (the pads are real fields) but an
/// approximation otherwise; prefer [`mir_stored_size`] whenever the MIR
/// type is at hand.
pub(crate) fn get_type_size(ctx: &Context, ty: TypeHandle) -> u64 {
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

    // Struct types: sum of field sizes. Exact for explicitly-padded
    // structs (pads are real [N x i8] fields); an approximation otherwise.
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
pub(crate) fn make_slice_struct(ctx: &mut Context) -> TypeHandle {
    let ptr_ty = llvm_types::PointerType::get_generic(ctx);
    let len_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    llvm_types::StructType::get_unnamed(ctx, vec![ptr_ty.into(), len_ty.into()]).into()
}

#[cfg(test)]
mod tests {
    //! Hardware-free unit tests for [`build_struct_slot_map`]: the slot map
    //! and the LLVM struct type are produced by the same walk, so these
    //! tests pin down both for the layout shapes from issue #128.

    use super::*;
    use dialect_mir::types::{EnumVariant, MirEnumType};

    fn make_ctx() -> Context {
        let mut ctx = Context::new();
        dialect_mir::register(&mut ctx);
        crate::register(&mut ctx);
        ctx
    }

    /// A MIR-level unsigned integer type (what the importer produces).
    fn mir_uint(ctx: &mut Context, width: u32) -> TypeHandle {
        IntegerType::get(ctx, width, Signedness::Unsigned).into()
    }

    /// A converted (signless) LLVM integer type.
    fn llvm_int(ctx: &mut Context, width: u32) -> TypeHandle {
        IntegerType::get(ctx, width, Signedness::Signless).into()
    }

    /// `[n x i8]` padding type, as `make_padding_type` builds it.
    fn pad(ctx: &mut Context, n: u64) -> TypeHandle {
        make_padding_type(ctx, n)
    }

    /// A zero-sized MIR struct (PhantomData shape).
    fn mir_zst(ctx: &mut Context) -> TypeHandle {
        MirStructType::get(ctx, "Phantom".into(), vec![], vec![]).into()
    }

    fn struct_fields(ctx: &Context, ty: TypeHandle) -> Vec<TypeHandle> {
        ty.deref(ctx)
            .downcast_ref::<llvm_types::StructType>()
            .expect("expected an LLVM struct type")
            .fields()
            .collect()
    }

    #[test]
    fn slot_map_reorder_only() {
        let mut ctx = make_ctx();
        // struct { a: u8, b: u64 }, memory order [b, a], no rustc offsets.
        let a = mir_uint(&mut ctx, 8);
        let b = mir_uint(&mut ctx, 64);
        let layout = StructLayoutInfo {
            field_types: vec![a, b],
            mem_to_decl: vec![1, 0],
            field_offsets: vec![],
            total_size: 0,
        };
        let map = build_struct_slot_map(&mut ctx, &layout).unwrap();

        assert_eq!(map.decl_to_llvm, vec![Some(1), Some(0)]);
        let i8s = llvm_int(&mut ctx, 8);
        let i64s = llvm_int(&mut ctx, 64);
        assert_eq!(struct_fields(&ctx, map.llvm_struct_ty), vec![i64s, i8s]);
    }

    #[test]
    fn slot_map_padding_only() {
        let mut ctx = make_ctx();
        // struct { a: u8 @ 0, b: u64 @ 8 }, declaration order == memory
        // order, size 16: lowers to { i8, [7 x i8], i64 }. The pad consumes
        // slot 1, so b lands at slot 2 (the issue #128 sites used 1).
        let a = mir_uint(&mut ctx, 8);
        let b = mir_uint(&mut ctx, 64);
        let layout = StructLayoutInfo {
            field_types: vec![a, b],
            mem_to_decl: vec![0, 1],
            field_offsets: vec![0, 8],
            total_size: 16,
        };
        let map = build_struct_slot_map(&mut ctx, &layout).unwrap();

        assert_eq!(map.decl_to_llvm, vec![Some(0), Some(2)]);
        let i8s = llvm_int(&mut ctx, 8);
        let i64s = llvm_int(&mut ctx, 64);
        let pad7 = pad(&mut ctx, 7);
        assert_eq!(
            struct_fields(&ctx, map.llvm_struct_ty),
            vec![i8s, pad7, i64s]
        );
    }

    #[test]
    fn slot_map_reorder_plus_padding() {
        let mut ctx = make_ctx();
        // struct { a: u8 @ 8, b: u64 @ 0 }, memory order [b, a], size 16:
        // lowers to { i64, i8, [7 x i8] } with a trailing pad.
        let a = mir_uint(&mut ctx, 8);
        let b = mir_uint(&mut ctx, 64);
        let layout = StructLayoutInfo {
            field_types: vec![a, b],
            mem_to_decl: vec![1, 0],
            field_offsets: vec![8, 0],
            total_size: 16,
        };
        let map = build_struct_slot_map(&mut ctx, &layout).unwrap();

        assert_eq!(map.decl_to_llvm, vec![Some(1), Some(0)]);
        let i8s = llvm_int(&mut ctx, 8);
        let i64s = llvm_int(&mut ctx, 64);
        let pad7 = pad(&mut ctx, 7);
        assert_eq!(
            struct_fields(&ctx, map.llvm_struct_ty),
            vec![i64s, i8s, pad7]
        );
    }

    #[test]
    fn slot_map_zst_interleaving() {
        let mut ctx = make_ctx();
        // struct { a: u32 @ 0, z: PhantomData @ 4, b: u32 @ 4 }, size 8.
        // The ZST is stripped (no slot, no pad split): { i32, i32 }.
        let a = mir_uint(&mut ctx, 32);
        let z = mir_zst(&mut ctx);
        let b = mir_uint(&mut ctx, 32);
        let layout = StructLayoutInfo {
            field_types: vec![a, z, b],
            mem_to_decl: vec![0, 1, 2],
            field_offsets: vec![0, 4, 4],
            total_size: 8,
        };
        let map = build_struct_slot_map(&mut ctx, &layout).unwrap();

        assert_eq!(map.decl_to_llvm, vec![Some(0), None, Some(1)]);
        let i32s = llvm_int(&mut ctx, 32);
        assert_eq!(struct_fields(&ctx, map.llvm_struct_ty), vec![i32s, i32s]);
    }

    #[test]
    fn slot_map_issue128_arena_shape() {
        let mut ctx = make_ctx();
        // The exact shape from issue #128 (examples/struct_field_layout):
        //
        //   enum Layout { Aos, Soa, AoSoA(u32) }          // -> { i8, i32 }
        //   struct Arena { layout: Layout, cap: u32, stride: u32, big: u64 }
        //
        // rustc layout: layout @ 0 (8 bytes), big @ 8, cap @ 16,
        // stride @ 20, size 24. The enum's lowered form { i8, i32 } only
        // covers 5 of its 8 bytes, so a [3 x i8] pad takes slot 1:
        //
        //   { { i8, i32 }, [3 x i8], i64, i32, i32 }
        //     layout=0     pad=1     big=2 cap=3 stride=4
        let discr = mir_uint(&mut ctx, 8);
        let payload = mir_uint(&mut ctx, 32);
        let layout_enum: TypeHandle = MirEnumType::get(
            &mut ctx,
            "Layout".into(),
            discr,
            vec![0, 1, 2],
            vec![
                EnumVariant::unit("Aos".into()),
                EnumVariant::unit("Soa".into()),
                EnumVariant::new("AoSoA".into(), vec![payload]),
            ],
        )
        .into();
        let cap = mir_uint(&mut ctx, 32);
        let stride = mir_uint(&mut ctx, 32);
        let big = mir_uint(&mut ctx, 64);

        let layout = StructLayoutInfo {
            field_types: vec![layout_enum, cap, stride, big],
            mem_to_decl: vec![0, 3, 1, 2],
            field_offsets: vec![0, 16, 20, 8],
            total_size: 24,
        };
        let map = build_struct_slot_map(&mut ctx, &layout).unwrap();

        assert_eq!(
            map.decl_to_llvm,
            vec![Some(0), Some(3), Some(4), Some(2)],
            "cap/stride/big must skip the [3 x i8] pad at slot 1"
        );

        let i8s = llvm_int(&mut ctx, 8);
        let i32s = llvm_int(&mut ctx, 32);
        let i64s = llvm_int(&mut ctx, 64);
        let enum_llvm: TypeHandle =
            llvm_types::StructType::get_unnamed(&ctx, vec![i8s, i32s]).into();
        let pad3 = pad(&mut ctx, 3);
        assert_eq!(
            struct_fields(&ctx, map.llvm_struct_ty),
            vec![enum_llvm, pad3, i64s, i32s, i32s]
        );
    }

    #[test]
    fn slot_map_nested_struct_uses_stored_size() {
        let mut ctx = make_ctx();
        // Inner struct whose stored rustc size (16) exceeds the sum of its
        // converted LLVM field sizes (i8 + i64 = 9, no offsets stored).
        // The outer walk must advance by the stored 16, reaching the next
        // field's offset exactly: NO interior pad before it.
        let x = mir_uint(&mut ctx, 8);
        let y = mir_uint(&mut ctx, 64);
        let inner: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Inner".into(),
            vec!["x".into(), "y".into()],
            vec![x, y],
            vec![],
            vec![],
            16,
            0,
        )
        .into();
        let c = mir_uint(&mut ctx, 8);

        let layout = StructLayoutInfo {
            field_types: vec![inner, c],
            mem_to_decl: vec![0, 1],
            field_offsets: vec![0, 16],
            total_size: 24,
        };
        let map = build_struct_slot_map(&mut ctx, &layout).unwrap();

        // inner = slot 0, c = slot 1 (adjacent), trailing [7 x i8] pad.
        assert_eq!(map.decl_to_llvm, vec![Some(0), Some(1)]);
        let fields = struct_fields(&ctx, map.llvm_struct_ty);
        assert_eq!(fields.len(), 3, "exactly one (trailing) pad slot");
        let pad7 = pad(&mut ctx, 7);
        assert_eq!(fields[2], pad7);
    }

    #[test]
    fn slot_map_rejects_malformed_memory_order() {
        let mut ctx = make_ctx();
        let a = mir_uint(&mut ctx, 8);
        let b = mir_uint(&mut ctx, 64);

        // Not a permutation: decl index 0 appears twice.
        let dup = StructLayoutInfo {
            field_types: vec![a, b],
            mem_to_decl: vec![0, 0],
            field_offsets: vec![],
            total_size: 0,
        };
        assert!(build_struct_slot_map(&mut ctx, &dup).is_err());

        // Wrong length.
        let short = StructLayoutInfo {
            field_types: vec![a, b],
            mem_to_decl: vec![0],
            field_offsets: vec![],
            total_size: 0,
        };
        assert!(build_struct_slot_map(&mut ctx, &short).is_err());

        // Offsets vector length mismatch (with explicit layout engaged).
        let bad_offsets = StructLayoutInfo {
            field_types: vec![a, b],
            mem_to_decl: vec![0, 1],
            field_offsets: vec![0],
            total_size: 16,
        };
        assert!(build_struct_slot_map(&mut ctx, &bad_offsets).is_err());
    }
}
