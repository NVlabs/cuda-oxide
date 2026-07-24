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
//! | `MirUnionType`                  | Aligned shared-storage struct    | All fields start at byte zero|
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
    EnumCarrierKind, EnumLayoutKind, MirArrayType, MirDisjointSliceType, MirEnumType, MirSliceType,
    MirStructType, MirTupleType, MirUnionType,
};
use llvm_export::types as llvm_types;
use llvm_export::types::PointerTypeExt;
use pliron::builtin::type_interfaces::{FloatTypeInterface, FunctionTypeInterface};
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
    if let Some(array_ty) = ty.deref(ctx).downcast_ref::<llvm_types::ArrayType>() {
        return array_ty.size() == 0 || is_zero_sized_type(ctx, array_ty.elem_type());
    }

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

    /// Layout facts of a `MirTupleType`.
    ///
    /// Tuples translated from a rustc type carry rustc's exact layout
    /// (offsets, memory order, size), which is consumed here identically to
    /// structs, so reordered tuples like `(u32, &T)` lower byte-correctly.
    /// Only synthetic layout-less tuples (the unit tuple, hand-built test
    /// types) fall back to LLVM natural layout.
    pub(crate) fn of_tuple(t: &MirTupleType) -> Self {
        StructLayoutInfo {
            field_types: t.get_types().to_vec(),
            mem_to_decl: t.memory_order(),
            field_offsets: t.field_offsets().to_vec(),
            total_size: t.total_size(),
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

/// Build byte-exact LLVM storage for a Rust union.
///
/// A union cannot be represented as an LLVM struct containing every declared
/// field: struct fields are consecutive, while union fields all start at byte
/// zero. We choose one byte-faithful field as the storage view and add explicit
/// tail bytes. A zero-length integer array raises the LLVM type's natural
/// alignment without consuming storage. Pointer-bearing fields are preferred
/// so an ordinary union copy keeps LLVM pointer provenance.
///
/// NVPTX gives scalar integers natural alignments up to 16 bytes. Reject a
/// more strongly aligned union instead of silently emitting a by-value type
/// with a weaker ABI alignment.
pub(crate) fn build_union_storage_type(
    ctx: &mut Context,
    union_ty: &MirUnionType,
) -> Result<TypeHandle, anyhow::Error> {
    let size = union_ty.total_size();
    let align = union_ty.abi_align();
    if align == 0 || !align.is_power_of_two() {
        return Err(anyhow::anyhow!(
            "union `{}` has invalid ABI alignment {}",
            union_ty.name(),
            align
        ));
    }
    if align > 16 {
        return Err(anyhow::anyhow!(
            "union `{}` requires {}-byte alignment; cuda-oxide currently supports union alignments up to 16 bytes",
            union_ty.name(),
            align
        ));
    }
    if size > 0 && !size.is_multiple_of(align) {
        return Err(anyhow::anyhow!(
            "union `{}` size {} is not a multiple of its {}-byte ABI alignment",
            union_ty.name(),
            size,
            align
        ));
    }

    let mut fields = Vec::with_capacity(union_ty.field_count());
    let mut pointer_carrier: Option<TypeHandle> = None;
    for (index, &field_ty) in union_ty.field_types().iter().enumerate() {
        let llvm_field_ty = convert_type(ctx, field_ty)?;
        let (field_size, field_align) =
            llvm_type_size_align(ctx, llvm_field_ty).ok_or_else(|| {
                anyhow::anyhow!(
                    "union `{}` field {} has unsupported LLVM size/alignment",
                    union_ty.name(),
                    index
                )
            })?;
        if field_size > size {
            return Err(anyhow::anyhow!(
                "union `{}` field {} lowers to {} bytes but the union is only {} bytes",
                union_ty.name(),
                index,
                field_size,
                size
            ));
        }
        if field_align > align {
            return Err(anyhow::anyhow!(
                "union `{}` field {} lowers with alignment {} but rustc reports union alignment {}",
                union_ty.name(),
                index,
                field_align,
                align
            ));
        }
        let contains_pointer = llvm_type_contains_pointer(ctx, llvm_field_ty);
        if contains_pointer {
            if let Some(first) = pointer_carrier
                && first != llvm_field_ty
            {
                return Err(anyhow::anyhow!(
                    "union `{}` has pointer-bearing fields with different LLVM representations; preserving provenance for that shape is not yet supported",
                    union_ty.name()
                ));
            }
            pointer_carrier = Some(llvm_field_ty);
        }
        fields.push((llvm_field_ty, field_size, field_align, contains_pointer));
    }

    let anchor_int = IntegerType::get(ctx, (align * 8) as u32, Signedness::Signless);
    let anchor: TypeHandle = llvm_types::ArrayType::get(ctx, anchor_int.into(), 0).into();
    let mut storage_fields = vec![anchor];
    if size > 0 {
        let representative = fields
            .iter()
            .filter(|(ty, field_size, _, _)| {
                *field_size > 0
                    && llvm_type_is_byte_faithful(ctx, *ty)
                    && (pointer_carrier.is_none() || llvm_type_contains_pointer(ctx, *ty))
            })
            .max_by_key(|(_, field_size, field_align, contains_pointer)| {
                (*contains_pointer, *field_align, *field_size)
            });
        if let Some(representative) = representative {
            storage_fields.push(representative.0);
            if representative.1 < size {
                storage_fields.push(make_padding_type(ctx, size - representative.1));
            }
        } else if pointer_carrier.is_some() {
            return Err(anyhow::anyhow!(
                "union `{}` has pointer-bearing fields but no byte-faithful pointer carrier; lowering it as raw bytes would discard pointer provenance",
                union_ty.name()
            ));
        } else {
            // Pointer-free unions may safely use raw bytes as their SSA
            // carrier. Field loads/stores still use their declared types.
            storage_fields.push(make_padding_type(ctx, size));
        }
    }
    let storage: TypeHandle = llvm_types::StructType::get_unnamed(ctx, storage_fields).into();
    let (llvm_size, llvm_align) = llvm_type_size_align(ctx, storage).ok_or_else(|| {
        anyhow::anyhow!(
            "union `{}` storage has unsupported LLVM layout",
            union_ty.name()
        )
    })?;
    if llvm_size != size || llvm_align != align {
        return Err(anyhow::anyhow!(
            "union `{}` storage lowered to size/alignment {}/{} but rustc requires {}/{}",
            union_ty.name(),
            llvm_size,
            llvm_align,
            size,
            align
        ));
    }
    Ok(storage)
}

fn llvm_type_contains_pointer(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    if ty_ref.is::<llvm_types::PointerType>() {
        return true;
    }
    if let Some(array) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
        return llvm_type_contains_pointer(ctx, array.elem_type());
    }
    if let Some(vector) = ty_ref.downcast_ref::<llvm_types::VectorType>() {
        return llvm_type_contains_pointer(ctx, vector.elem_type());
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
        return struct_ty
            .fields()
            .any(|field| llvm_type_contains_pointer(ctx, field));
    }
    false
}

/// One pointer-valued leaf in an LLVM aggregate's physical storage.
///
/// Offsets are absolute within the enclosing enum. Keeping the address space
/// in the identity prevents an AS0 pointer carrier from being treated as the
/// same storage as an AS1 payload pointer merely because both are eight bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct LlvmPointerStorage {
    offset: u64,
    size: u64,
    address_space: u32,
}

/// Record every pointer-valued leaf in `ty` at its natural LLVM byte offset.
///
/// This is deliberately a physical-layout walk rather than a simple
/// `contains_pointer` predicate. A niche carrier may be one field inside an
/// aggregate payload, for example the pointer at byte 8 in
/// `Option<(usize, &T)>`. In that case the aggregate and the carrier overlap,
/// but they agree exactly about which bytes hold the pointer.
fn collect_llvm_pointer_storage(
    ctx: &Context,
    ty: TypeHandle,
    base_offset: u64,
    out: &mut Vec<LlvmPointerStorage>,
) -> Option<()> {
    let ty_ref = ty.deref(ctx);
    if let Some(pointer) = ty_ref.downcast_ref::<llvm_types::PointerType>() {
        let (size, _) = llvm_type_size_align(ctx, ty)?;
        out.push(LlvmPointerStorage {
            offset: base_offset,
            size,
            address_space: pointer.address_space(),
        });
        return Some(());
    }
    if let Some(array) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
        // Expanding an arbitrary array into one record per pointer would let
        // a valid but enormous type consume unbounded verifier memory. This
        // repair needs only fixed structs; keep pointer arrays fail-closed.
        return (!llvm_type_contains_pointer(ctx, array.elem_type())).then_some(());
    }
    if let Some(vector) = ty_ref.downcast_ref::<llvm_types::VectorType>() {
        // Pointer vectors have the same unbounded-expansion problem as arrays
        // and also carry vector-specific ABI alignment. Reject rather than
        // approximating either property.
        return (!llvm_type_contains_pointer(ctx, vector.elem_type())).then_some(());
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
        let fields: Vec<_> = struct_ty.fields().collect();
        let mut end = 0u64;
        for field in fields {
            let (field_size, field_align) = llvm_type_size_align(ctx, field)?;
            let field_align = field_align.max(1);
            let remainder = end % field_align;
            let field_offset = if remainder == 0 {
                end
            } else {
                end.checked_add(field_align - remainder)?
            };
            collect_llvm_pointer_storage(ctx, field, base_offset.checked_add(field_offset)?, out)?;
            end = field_offset.checked_add(field_size)?;
        }
        return Some(());
    }

    // All pointer-bearing LLVM types understood by this lowering are handled
    // above. Unknown pointer containers must fail closed.
    (!llvm_type_contains_pointer(ctx, ty)).then_some(())
}

/// Whether a slotless incoming field and the already selected enum storage
/// agree exactly about every pointer byte covered by that field.
///
/// Equality here is intentionally strict: the same absolute offset, extent,
/// and address space must appear on both sides. This admits the real rustc
/// layout of `Option<(usize, &T)>`, while continuing to reject a pointer
/// variant sharing bytes with integer bits or an aggregate with an additional
/// pointer for which the enum has only raw-byte storage.
fn pointer_storage_matches_claims(
    ctx: &Context,
    incoming_offset: u64,
    incoming_size: u64,
    incoming_ty: TypeHandle,
    colliding_claims: &[&(u64, u64, TypeHandle)],
) -> bool {
    let Some(incoming_end) = incoming_offset.checked_add(incoming_size) else {
        return false;
    };
    let mut incoming = Vec::new();
    if collect_llvm_pointer_storage(ctx, incoming_ty, incoming_offset, &mut incoming).is_none() {
        return false;
    }

    let mut existing = Vec::new();
    for &&(offset, _size, claim_ty) in colliding_claims {
        let mut regions = Vec::new();
        if collect_llvm_pointer_storage(ctx, claim_ty, offset, &mut regions).is_none() {
            return false;
        }
        existing.extend(regions.into_iter().filter(|region| {
            let Some(region_end) = region.offset.checked_add(region.size) else {
                return true;
            };
            region.offset < incoming_end && incoming_offset < region_end
        }));
    }

    incoming.sort_unstable();
    existing.sort_unstable();
    incoming == existing
}

pub(crate) fn llvm_type_contains_pointer_in_address_space(
    ctx: &Context,
    ty: TypeHandle,
    address_space: u32,
) -> bool {
    let ty_ref = ty.deref(ctx);
    if let Some(pointer) = ty_ref.downcast_ref::<llvm_types::PointerType>() {
        return pointer.address_space() == address_space;
    }
    if let Some(array) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
        return llvm_type_contains_pointer_in_address_space(ctx, array.elem_type(), address_space);
    }
    if let Some(vector) = ty_ref.downcast_ref::<llvm_types::VectorType>() {
        return llvm_type_contains_pointer_in_address_space(ctx, vector.elem_type(), address_space);
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
        return struct_ty
            .fields()
            .any(|field| llvm_type_contains_pointer_in_address_space(ctx, field, address_space));
    }
    false
}

/// Whether this is an aggregate/vector that contains LLVM `i1` storage.
///
/// Rust `bool` is an SSA `i1`, but its memory representation is one complete
/// byte whose only valid values are 0 and 1. Enum storage never uses `i1` as
/// a physical type: scalar bools claim an explicit i8 byte, and aggregates
/// containing bools claim their byte-faithful twin (see
/// [`llvm_byte_faithful_twin`]), with construction canonicalizing the value
/// before the store.
pub(crate) fn llvm_type_contains_i1(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    if let Some(integer) = ty_ref.downcast_ref::<IntegerType>() {
        return integer.width() == 1;
    }
    if let Some(array) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
        return llvm_type_contains_i1(ctx, array.elem_type());
    }
    if let Some(vector) = ty_ref.downcast_ref::<llvm_types::VectorType>() {
        return llvm_type_contains_i1(ctx, vector.elem_type());
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
        return struct_ty
            .fields()
            .any(|field| llvm_type_contains_i1(ctx, field));
    }
    false
}

/// The byte-faithful storage twin of an LLVM type: every `i1` leaf becomes
/// its canonical `i8` memory byte, recursively through structs and arrays.
///
/// Rust guarantees a bool occupies one full byte holding exactly 0 or 1, so
/// storing the twin (with each bool zero-extended) writes the same bytes the
/// host writes, while re-loading the original type from those canonical
/// bytes remains well-defined. Sizes and alignments are unchanged because an
/// `i1` already occupies one byte of storage.
///
/// Returns `None` for shapes with a different memory story (`i1` vectors are
/// bit-packed masks) or unknown containers; callers must fail closed.
pub(crate) fn llvm_byte_faithful_twin(ctx: &mut Context, ty: TypeHandle) -> Option<TypeHandle> {
    if !llvm_type_contains_i1(ctx, ty) {
        return Some(ty);
    }
    enum Shape {
        Bool,
        Array(TypeHandle, u64),
        Struct(Vec<TypeHandle>),
        Other,
    }
    let shape = {
        let ty_ref = ty.deref(ctx);
        if ty_ref
            .downcast_ref::<IntegerType>()
            .is_some_and(|integer| integer.width() == 1)
        {
            Shape::Bool
        } else if let Some(array) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
            Shape::Array(array.elem_type(), array.size())
        } else if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
            Shape::Struct(struct_ty.fields().collect())
        } else {
            Shape::Other
        }
    };
    match shape {
        Shape::Bool => Some(IntegerType::get(ctx, 8, Signedness::Signless).into()),
        Shape::Array(elem, count) => {
            let twin = llvm_byte_faithful_twin(ctx, elem)?;
            Some(llvm_types::ArrayType::get(ctx, twin, count).into())
        }
        Shape::Struct(fields) => {
            let twins = fields
                .into_iter()
                .map(|field| llvm_byte_faithful_twin(ctx, field))
                .collect::<Option<Vec<_>>>()?;
            Some(llvm_types::StructType::get_unnamed(ctx, twins).into())
        }
        Shape::Other => None,
    }
}

/// Whether a MIR aggregate contains a semantic Rust bool value.
///
/// This deliberately stops at pointers/slices: a pointee bool does not occupy
/// bytes in the aggregate itself. It also stops at nested enums, whose own
/// slot-map construction is responsible for canonicalizing a top-level bool
/// or rejecting a deeper one. Inspecting the MIR type is necessary because a
/// union may lower to raw i8 storage and thereby hide a bool from the LLVM
/// type-level check above.
fn mir_type_contains_i1(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    if let Some(integer) = ty_ref.downcast_ref::<IntegerType>() {
        return integer.width() == 1;
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<MirStructType>() {
        return struct_ty
            .field_types()
            .iter()
            .copied()
            .any(|field| mir_type_contains_i1(ctx, field));
    }
    if let Some(tuple_ty) = ty_ref.downcast_ref::<MirTupleType>() {
        return tuple_ty
            .get_types()
            .iter()
            .copied()
            .any(|field| mir_type_contains_i1(ctx, field));
    }
    if let Some(array_ty) = ty_ref.downcast_ref::<dialect_mir::types::MirArrayType>() {
        return mir_type_contains_i1(ctx, array_ty.element_ty);
    }
    if let Some(union_ty) = ty_ref.downcast_ref::<MirUnionType>() {
        return union_ty
            .field_types()
            .iter()
            .copied()
            .any(|field| mir_type_contains_i1(ctx, field));
    }
    false
}

/// Whether loading and storing this LLVM value preserves every byte in its
/// allocation. In particular, `i1` is not byte-faithful: it occupies one
/// addressable byte, but LLVM does not define the upper seven stored bits.
pub(crate) fn llvm_type_is_byte_faithful(ctx: &Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);
    if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        return int_ty.width().is_multiple_of(8);
    }
    if ty_ref.is::<llvm_types::HalfType>()
        || ty_ref.is::<FP32Type>()
        || ty_ref.is::<FP64Type>()
        || ty_ref.is::<llvm_types::PointerType>()
    {
        return true;
    }
    if let Some(array) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
        return llvm_type_is_byte_faithful(ctx, array.elem_type());
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
        let fields: Vec<_> = struct_ty.fields().collect();
        let mut end = 0u64;
        let mut max_align = 1u64;
        for field in fields {
            if !llvm_type_is_byte_faithful(ctx, field) {
                return false;
            }
            let Some((field_size, field_align)) = llvm_type_size_align(ctx, field) else {
                return false;
            };
            let field_align = field_align.max(1);
            let aligned_end = end.div_ceil(field_align) * field_align;
            if aligned_end != end {
                // LLVM would insert bytes that are not represented by an SSA
                // field. Loading and re-storing the aggregate would lose them.
                return false;
            }
            end += field_size;
            max_align = max_align.max(field_align);
        }
        // Reject implicit trailing padding for the same reason. Explicit
        // `[N x i8]` padding fields keep `end` equal to the allocation size.
        return end.div_ceil(max_align) * max_align == end;
    }
    false
}

/// Size of a MIR-level type from rustc layout truth, when stored.
///
/// `MirStructType`, `MirUnionType`, and `MirEnumType` carry `total_size` (interior and
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
    if let Some(u) = ty_ref.downcast_ref::<MirUnionType>() {
        return Some(u.total_size());
    }
    if let Some(a) = ty_ref.downcast_ref::<dialect_mir::types::MirArrayType>() {
        let elem_ty = a.element_ty;
        let size = a.size;
        return mir_stored_size(ctx, elem_ty).map(|elem_size| elem_size * size);
    }
    None
}

/// Exact ABI alignment carried by a MIR aggregate type, when rustc layout is
/// available.
///
/// LLVM aggregate types cannot encode a Rust `repr(align(N))` raise. Tuples,
/// structs, enums, and unions therefore carry rustc's alignment explicitly in
/// the MIR dialect. Arrays have the same ABI alignment as their element, so
/// recurse through any number of array layers instead of relying on the
/// converted LLVM element's structural alignment.
pub(crate) fn mir_type_abi_align(ctx: &Context, mir_ty: TypeHandle) -> Option<u64> {
    let ty_ref = mir_ty.deref(ctx);
    if let Some(tuple_ty) = ty_ref.downcast_ref::<MirTupleType>() {
        return Some(tuple_ty.abi_align()).filter(|align| *align > 0);
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<MirStructType>() {
        return Some(struct_ty.abi_align).filter(|align| *align > 0);
    }
    if let Some(enum_ty) = ty_ref.downcast_ref::<MirEnumType>() {
        return Some(enum_ty.abi_align()).filter(|align| *align > 0);
    }
    if let Some(union_ty) = ty_ref.downcast_ref::<MirUnionType>() {
        return Some(union_ty.abi_align()).filter(|align| *align > 0);
    }
    if let Some(array_ty) = ty_ref.downcast_ref::<MirArrayType>() {
        let element_ty = array_ty.element_type();
        return mir_type_abi_align(ctx, element_ty);
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
pub(crate) fn llvm_type_size_align(ctx: &Context, ty: TypeHandle) -> Option<(u64, u64)> {
    let ty_ref = ty.deref(ctx);

    if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
        let size = (int_ty.width() as u64).div_ceil(8);
        // i8 → 1, i16 → 2, i32 → 4, i64 → 8, i128 → 16.
        return Some((size, size.next_power_of_two().min(16)));
    }
    if ty_ref.is::<llvm_types::HalfType>() {
        return Some((2, 2));
    }
    if ty_ref.is::<FP32Type>() {
        return Some((4, 4));
    }
    if ty_ref.is::<FP64Type>() {
        return Some((8, 8));
    }
    if ty_ref.is::<llvm_types::PointerType>() {
        // Lowering runs before the exporter chooses the minimal, legacy, or
        // modern NVPTX data layout. The first two use 64-bit pointers in all
        // address spaces; modern NVVM alone uses p3:32. Enum storage rejects
        // shared pointers below because no one target-agnostic size is sound.
        return Some((8, 8));
    }
    if let Some(arr_ty) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
        let (elem_size, elem_align) = llvm_type_size_align(ctx, arr_ty.elem_type())?;
        return Some((elem_size.checked_mul(arr_ty.size())?, elem_align.max(1)));
    }
    if let Some(vector) = ty_ref.downcast_ref::<llvm_types::VectorType>() {
        if vector.is_scalable() {
            return None;
        }
        let element_bits = {
            let element = vector.elem_type();
            let element_ref = element.deref(ctx);
            if let Some(integer) = element_ref.downcast_ref::<IntegerType>() {
                u64::from(integer.width())
            } else if let Some(float) = type_cast::<dyn FloatTypeInterface>(&*element_ref) {
                u64::try_from(float.get_semantics().bits).ok()?
            } else if element_ref.is::<llvm_types::PointerType>() {
                64
            } else {
                return None;
            }
        };
        let total_bits = element_bits.checked_mul(u64::from(vector.num_elements()))?;
        let size = total_bits.div_ceil(8);
        // Both cuda-oxide NVPTX data layouts explicitly define fixed vector
        // ABI alignment only for these widths. Refuse to guess LLVM defaults
        // for any other width in physical Rust layout code.
        let align = match total_bits {
            16 => 2,
            32 => 4,
            64 => 8,
            128 => 16,
            _ => return None,
        };
        return Some((size, align));
    }
    if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
        let fields: Vec<_> = struct_ty.fields().collect();
        let (_end, size, align) = natural_struct_layout(ctx, &fields)?;
        return Some((size, align));
    }

    None
}

/// Natural (non-packed) LLVM struct layout over `fields`.
///
/// Returns `(end, size, align)` where `end` is the unrounded offset just past
/// the last field, `size` is `end` rounded up to the struct alignment (the
/// allocation size LLVM uses for GEP striding), and `align` is the widest
/// field alignment.
pub(crate) fn natural_struct_layout(
    ctx: &Context,
    fields: &[TypeHandle],
) -> Option<(u64, u64, u64)> {
    let mut end = 0u64;
    let mut align = 1u64;
    for field in fields {
        let (field_size, field_align) = llvm_type_size_align(ctx, *field)?;
        let field_align = field_align.max(1);
        end = end.div_ceil(field_align) * field_align;
        end += field_size;
        align = align.max(field_align);
    }
    let size = end.div_ceil(align) * align;
    Some((end, size, align))
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
    /// Which struct slot holds rustc's physical tag/niche carrier. `None`
    /// for `Single` and `Empty` layouts, which have no tag in memory.
    pub carrier_slot: Option<u32>,
    /// Converted physical carrier type (integer or pointer), when present.
    pub carrier_llvm_ty: Option<TypeHandle>,
    /// Which struct slot holds each payload field, in the flattened
    /// order of `MirEnumType::all_field_types`. `None` means the field
    /// has no slot of its own: it is zero-sized, or its bytes are shared
    /// with a different-typed field of another variant. Such fields are
    /// read and written through memory at `field_offsets` instead.
    pub field_slots: Vec<Option<u32>>,
    /// Byte position of each payload field inside the enum.
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
/// Direct and Niche carriers are claimed first, so a semantic field with a
/// different SSA type cannot redefine the same physical bytes. Single and
/// Empty layouts simply have no carrier. Unknown layouts are rejected.
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
        all_field_sizes,
        variant_field_counts,
        variant_inhabited,
        tag_offset,
        total_size,
        abi_align,
        layout_kind,
        carrier_kind,
        carrier_width,
        carrier_address_space,
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
            enum_ty.all_field_sizes.clone(),
            enum_ty.variant_field_counts.clone(),
            enum_ty.variant_inhabited.clone(),
            enum_ty.tag_offset(),
            enum_ty.total_size(),
            enum_ty.abi_align(),
            enum_ty.layout_kind,
            enum_ty.carrier_kind,
            enum_ty.carrier_width,
            enum_ty.carrier_address_space,
        )
    };

    if layout_kind == EnumLayoutKind::Unknown {
        return Err(anyhow::anyhow!(
            "enum slot map: `{}` has unknown physical layout; refusing to guess its bytes",
            name
        ));
    }
    if carrier_kind == EnumCarrierKind::Pointer
        && carrier_address_space == llvm_types::address_space::SHARED
    {
        return Err(anyhow::anyhow!(
            "enum slot map: `{}` uses a shared-memory pointer carrier whose size is target-mode dependent (64-bit PTX/legacy, 32-bit modern NVVM); refusing target-agnostic enum lowering",
            name
        ));
    }
    if carrier_kind == EnumCarrierKind::Integer && !carrier_width.is_multiple_of(8) {
        return Err(anyhow::anyhow!(
            "enum slot map: `{}` integer carrier width {} is not a whole number of bytes; refusing physical storage with unspecified upper byte bits",
            name,
            carrier_width
        ));
    }

    let carrier_ty: Option<TypeHandle> = match carrier_kind {
        EnumCarrierKind::None => None,
        EnumCarrierKind::Integer if layout_kind == EnumLayoutKind::Direct => {
            let converted = convert_type(ctx, discriminant_ty)?;
            let width = converted
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .map(IntegerType::width);
            if width != Some(carrier_width) {
                return Err(anyhow::anyhow!(
                    "enum slot map: `{}` direct carrier does not match its declared discriminant type",
                    name
                ));
            }
            Some(converted)
        }
        EnumCarrierKind::Integer => {
            Some(IntegerType::get(ctx, carrier_width, Signedness::Signless).into())
        }
        EnumCarrierKind::Pointer => {
            Some(llvm_types::PointerType::get(ctx, carrier_address_space).into())
        }
    };
    let mut field_llvm_types = Vec::with_capacity(all_field_types.len());
    for &field_ty in &all_field_types {
        field_llvm_types.push(convert_type(ctx, field_ty)?);
    }

    if all_field_offsets.len() != all_field_types.len()
        || all_field_sizes.len() != all_field_types.len()
    {
        return Err(anyhow::anyhow!(
            "enum slot map: `{}` has {} field offsets for {} fields",
            name,
            all_field_offsets.len().min(all_field_sizes.len()),
            all_field_types.len()
        ));
    }

    // Phase 1: decide who gets a struct slot. The physical carrier goes
    // first so a semantic field can never claim its bytes using a different
    // type (e.g. `bool` is i1 semantically but i8 in Option<bool> storage).
    // claims: (byte position, byte size, converted type), no two overlap.
    let mut claims: Vec<(u64, u64, TypeHandle)> = Vec::new();
    let carrier_claim = if let Some(carrier_ty) = carrier_ty {
        let (carrier_size, carrier_align) =
            llvm_type_size_align(ctx, carrier_ty).ok_or_else(|| {
                anyhow::anyhow!(
                    "enum slot map: `{}` carrier has unsupported LLVM layout",
                    name
                )
            })?;
        let expected_size = u64::from(carrier_width).div_ceil(8);
        if carrier_size != expected_size
            || tag_offset % carrier_align.max(1) != 0
            || tag_offset
                .checked_add(carrier_size)
                .is_none_or(|end| end > total_size)
        {
            return Err(anyhow::anyhow!(
                "enum slot map: `{}` carrier (size {}, align {}) cannot sit at byte {} of {}",
                name,
                carrier_size,
                carrier_align,
                tag_offset,
                total_size
            ));
        }
        claims.push((tag_offset, carrier_size, carrier_ty));
        Some(0usize)
    } else {
        None
    };

    let mut claim_of_field: Vec<Option<usize>> = vec![None; field_llvm_types.len()];
    let mut field_is_inhabited = Vec::with_capacity(field_llvm_types.len());
    for (variant, count) in variant_field_counts.iter().enumerate() {
        field_is_inhabited.extend(std::iter::repeat_n(
            variant_inhabited.get(variant).copied().unwrap_or(0) != 0,
            *count as usize,
        ));
    }
    let mut order: Vec<usize> = (0..field_llvm_types.len()).collect();
    // At one byte position, prefer a representation that preserves all of
    // its stored bits. This makes the result independent of source-variant
    // order and prevents an i1/bool view from claiming storage that a later
    // i8 view needs to preserve. A scalar i1 always uses an i8 storage claim:
    // construction explicitly zero-extends that scalar below.
    order.sort_by_key(|&i| {
        (
            all_field_offsets[i],
            !llvm_type_is_byte_faithful(ctx, field_llvm_types[i]),
            i,
        )
    });
    for flat in order {
        if !field_is_inhabited.get(flat).copied().unwrap_or(false) {
            continue;
        }
        let llvm_ty = field_llvm_types[flat];
        if llvm_type_contains_pointer_in_address_space(
            ctx,
            llvm_ty,
            llvm_types::address_space::SHARED,
        ) {
            return Err(anyhow::anyhow!(
                "enum slot map: `{}` field {} contains a shared-memory pointer whose size is target-mode dependent (64-bit PTX/legacy, 32-bit modern NVVM); refusing target-agnostic enum lowering",
                name,
                flat
            ));
        }
        let (size, align) = llvm_type_size_align(ctx, llvm_ty).ok_or_else(|| {
            anyhow::anyhow!(
                "enum slot map: `{}` field {} has unsupported LLVM size/alignment",
                name,
                flat
            )
        })?;
        if size != all_field_sizes[flat] {
            return Err(anyhow::anyhow!(
                "enum slot map: `{}` field {} lowers to {} bytes but rustc says {}",
                name,
                flat,
                size,
                all_field_sizes[flat]
            ));
        }
        if size == 0 || is_zero_sized_type(ctx, llvm_ty) {
            // ZSTs own no bytes and no slot.
            continue;
        }
        let offset = all_field_offsets[flat];
        if offset.checked_add(size).is_none_or(|end| end > total_size) {
            return Err(anyhow::anyhow!(
                "enum slot map: `{}` field {} (size {}) at byte {} exceeds total size {}",
                name,
                flat,
                size,
                offset,
                total_size
            ));
        }
        if !offset.is_multiple_of(align.max(1)) {
            return Err(anyhow::anyhow!(
                "enum slot map: `{}` field {} requires alignment {} but rustc offset {} is not aligned; packed enum payload access is not yet supported",
                name,
                flat,
                align,
                offset
            ));
        }

        let is_scalar_i1 = llvm_ty
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .is_some_and(|integer| integer.width() == 1);
        // A bool the MIR type promises but the LLVM lowering hides (a union
        // whose storage is a raw byte blob) cannot be canonicalized here:
        // its bytes were written by the union's own stores, not by this
        // enum's construction.
        if !is_scalar_i1
            && mir_type_contains_i1(ctx, all_field_types[flat])
            && !llvm_type_contains_i1(ctx, llvm_ty)
        {
            return Err(anyhow::anyhow!(
                "enum slot map: `{}` field {} contains bool storage hidden behind a union; cuda-oxide cannot prove those bytes are canonical",
                name,
                flat
            ));
        }

        // Rust bool is a semantic i1 but occupies a complete byte in memory.
        // Never make i1 the struct's physical storage type: give a standalone
        // bool an i8 claim, or reuse the exact i8 carrier/field claim already
        // covering that byte. Construction leaves the bool slotless and the
        // deferred store below explicitly zero-extends i1 -> i8; extraction
        // loads i1 from that byte.
        if is_scalar_i1 {
            let colliding_claims = claims
                .iter()
                .filter(|&&(o, s, _)| offset < o + s && o < offset + size)
                .collect::<Vec<_>>();
            if colliding_claims.is_empty() {
                let byte_ty: TypeHandle = IntegerType::get(ctx, 8, Signedness::Signless).into();
                claims.push((offset, 1, byte_ty));
                continue;
            }
            let has_exact_i8_storage = colliding_claims.iter().all(|&&(o, s, claim_ty)| {
                o == offset
                    && s == 1
                    && claim_ty
                        .deref(ctx)
                        .downcast_ref::<IntegerType>()
                        .is_some_and(|integer| integer.width() == 8)
            });
            if has_exact_i8_storage {
                continue;
            }
            return Err(anyhow::anyhow!(
                "enum slot map: `{}` scalar bool field {} overlaps storage other than one exact i8 byte; refusing to expose non-canonical bool bits",
                name,
                flat
            ));
        }

        // Aggregates containing bool claim their byte-faithful twin (every
        // i1 leaf becomes its canonical i8 memory byte) and stay slotless:
        // construction canonicalizes the value and writes it at its byte
        // offset; extraction re-loads the original type from the canonical
        // bytes, exactly like the scalar-bool path above.
        let storage_ty = if llvm_type_contains_i1(ctx, llvm_ty) {
            llvm_byte_faithful_twin(ctx, llvm_ty).ok_or_else(|| {
                anyhow::anyhow!(
                    "enum slot map: `{}` field {} contains bool storage in a shape (e.g. an i1 vector) whose memory bytes cannot be canonicalized",
                    name,
                    flat
                )
            })?
        } else {
            llvm_ty
        };
        let field_gets_slot = storage_ty == llvm_ty;

        // Another variant already placed the same storage type at the same
        // position? Then both fields can simply use that claim: variants
        // share bytes, and here they even agree on the type.
        if let Some(ci) = claims
            .iter()
            .position(|&(o, _, t)| o == offset && t == storage_ty)
        {
            if field_gets_slot {
                claim_of_field[flat] = Some(ci);
            }
            continue;
        }
        // The bytes are taken by a different type. Pointer-free values can
        // use the memory fallback below. Pointer-bearing values may do so only
        // when a physical-layout walk proves that every pointer leaf exactly
        // matches an existing pointer slot. This is the common
        // `Option<(usize, &T)>` shape: the niche carrier is the tuple's pointer
        // field. A pointer/non-pointer overlap, address-space mismatch, or
        // additional pointer without a pointer slot still fails closed.
        let colliding_claims = claims
            .iter()
            .filter(|&&(o, s, _)| offset < o + s && o < offset + size)
            .collect::<Vec<_>>();
        if !colliding_claims.is_empty() {
            let has_pointer_overlap = llvm_type_contains_pointer(ctx, storage_ty)
                || colliding_claims
                    .iter()
                    .any(|&&(_, _, claim_ty)| llvm_type_contains_pointer(ctx, claim_ty));
            if has_pointer_overlap
                && !pointer_storage_matches_claims(ctx, offset, size, storage_ty, &colliding_claims)
            {
                return Err(anyhow::anyhow!(
                    "enum slot map: `{}` has overlapping pointer and non-identical storage at byte {}; refusing to erase LLVM pointer provenance",
                    name,
                    offset
                ));
            }
            let incoming_is_byte_faithful = llvm_type_is_byte_faithful(ctx, storage_ty);
            let claims_are_byte_faithful = colliding_claims
                .iter()
                .all(|&&(_, _, claim_ty)| llvm_type_is_byte_faithful(ctx, claim_ty));
            if !incoming_is_byte_faithful || !claims_are_byte_faithful {
                return Err(anyhow::anyhow!(
                    "enum slot map: `{}` field {} overlaps non-identical storage but its lowered type is not byte-faithful (for example, it may contain implicit padding); refusing a type-punned store",
                    name,
                    flat
                ));
            }
            continue;
        }
        claims.push((offset, size, storage_ty));
        if field_gets_slot {
            claim_of_field[flat] = Some(claims.len() - 1);
        }
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
    let (_end, natural_size, natural_align) = natural_struct_layout(ctx, &llvm_fields)
        .ok_or_else(|| anyhow::anyhow!("enum slot map: `{}` has unsupported LLVM layout", name))?;
    if natural_size != total_size {
        return Err(anyhow::anyhow!(
            "enum slot map: `{}` lowered to {} bytes but rustc says {}",
            name,
            natural_size,
            total_size
        ));
    }
    let required_align = abi_align.max(1);
    if natural_align > required_align {
        return Err(anyhow::anyhow!(
            "enum slot map: `{}` lowered with alignment {} but rustc requires {}; explicit over-aligned enum storage is not yet supported",
            name,
            natural_align,
            abi_align
        ));
    }
    if natural_align < required_align {
        // The byte claims alone can under-align the storage, e.g. when the
        // only claim is an i8 niche carrier inside a 4-aligned enum. Raise
        // the struct's alignment with a zero-length anchor field, the same
        // mechanism union storage uses; it occupies no bytes, so every slot
        // index simply shifts by one.
        let anchor_int = IntegerType::get(ctx, (required_align * 8) as u32, Signedness::Signless);
        let anchor: TypeHandle = llvm_types::ArrayType::get(ctx, anchor_int.into(), 0).into();
        llvm_fields.insert(0, anchor);
        for slot in &mut slot_of_claim {
            *slot += 1;
        }
    }

    let field_slots = claim_of_field
        .into_iter()
        .map(|c| c.map(|ci| slot_of_claim[ci]))
        .collect();
    Ok(EnumSlotMap {
        llvm_struct_ty: llvm_types::StructType::get_unnamed(ctx, llvm_fields).into(),
        carrier_slot: carrier_claim.map(|claim| slot_of_claim[claim]),
        carrier_llvm_ty: carrier_ty,
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

/// Return an enum name only when the dialect lacks rustc's physical layout.
/// All importer-produced Direct, Niche, Single, and Empty layouts are
/// byte-faithful; legacy `Unknown` values are rejected everywhere rather than
/// receiving a guessed internal representation.
pub(crate) fn enum_unmodeled_in_memory(ctx: &Context, ty: TypeHandle) -> Option<String> {
    let ty_ref = ty.deref(ctx);
    let enum_ty = ty_ref.downcast_ref::<MirEnumType>()?;
    (enum_ty.layout_kind == EnumLayoutKind::Unknown).then(|| enum_ty.name().to_string())
}

/// Search a kernel parameter's type for an enum the host and device
/// would disagree about (see [`enum_unmodeled_in_memory`]).
///
/// The search looks everywhere host data can hide: behind pointers,
/// inside slices and arrays, in struct/tuple fields, and in other enums'
/// payloads. It returns the first offending enum's name.
///
/// Kernel signatures are checked early for a focused diagnostic. Lowering also
/// rejects Unknown layouts for locals, globals, and physical operations, so no
/// guessed representation can escape through an internal-only path.
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
        } else if let Some(u) = ty_ref.downcast_ref::<MirUnionType>() {
            u.field_types.clone()
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

/// Prove that an initialized Rust global can be accessed through the LLVM
/// semantic type produced by this lowering pipeline.
///
/// The initializer itself is emitted as exact bytes. That is only half of the
/// contract: later field GEPs and typed loads still use `mir_ty`. If that type
/// places a field at a different byte offset, an exact initializer can still be
/// read incorrectly (or even past the end of the object). Reject every shape
/// for which we cannot prove that the two views agree.
pub(crate) fn validate_initialized_global_layout(
    ctx: &mut Context,
    mir_ty: TypeHandle,
    initializer_size: u64,
    initializer_align: u64,
) -> Result<(), anyhow::Error> {
    if initializer_align == 0 || !initializer_align.is_power_of_two() {
        return Err(anyhow::anyhow!(
            "initialized global has invalid rustc allocation alignment {}",
            initializer_align
        ));
    }

    validate_initialized_global_type(ctx, mir_ty, &mut Vec::new())?;

    let llvm_ty = convert_type(ctx, mir_ty)?;
    let (llvm_size, llvm_align) = llvm_type_size_align(ctx, llvm_ty)
        .ok_or_else(|| anyhow::anyhow!("initialized global has unsupported LLVM size/alignment"))?;
    if llvm_size != initializer_size || llvm_align > initializer_align {
        return Err(anyhow::anyhow!(
            "initialized global type is not byte-compatible with rustc's allocation: the lowered LLVM value has size/alignment {}/{}, but the initializer has size/alignment {}/{}",
            llvm_size,
            llvm_align,
            initializer_size,
            initializer_align
        ));
    }

    Ok(())
}

fn validate_initialized_global_type(
    ctx: &mut Context,
    mir_ty: TypeHandle,
    visited: &mut Vec<TypeHandle>,
) -> Result<(), anyhow::Error> {
    if visited.contains(&mir_ty) {
        return Ok(());
    }
    visited.push(mir_ty);

    if let Some(name) = enum_unmodeled_in_memory(ctx, mir_ty) {
        return Err(anyhow::anyhow!(
            "initialized global contains enum `{}` with unknown physical layout; refusing to guess a byte representation",
            name
        ));
    }

    enum Kind {
        Struct(MirStructType),
        Tuple(MirTupleType),
        Enum(MirEnumType),
        Array(TypeHandle),
        Leaf,
    }

    let kind = {
        let ty_ref = mir_ty.deref(ctx);
        if let Some(struct_ty) = ty_ref.downcast_ref::<MirStructType>() {
            Kind::Struct(struct_ty.clone())
        } else if let Some(tuple_ty) = ty_ref.downcast_ref::<MirTupleType>() {
            Kind::Tuple(tuple_ty.clone())
        } else if let Some(enum_ty) = ty_ref.downcast_ref::<MirEnumType>() {
            Kind::Enum(enum_ty.clone())
        } else if let Some(array_ty) = ty_ref.downcast_ref::<dialect_mir::types::MirArrayType>() {
            Kind::Array(array_ty.element_ty)
        } else {
            Kind::Leaf
        }
    };

    match kind {
        Kind::Struct(struct_ty) => {
            validate_initialized_struct_layout(ctx, mir_ty, &struct_ty)?;
            for field_ty in struct_ty.field_types {
                validate_initialized_global_type(ctx, field_ty, visited)?;
            }
        }
        Kind::Tuple(tuple_ty) => {
            if !tuple_ty.get_types().is_empty() {
                // Tuples carry rustc's field offsets exactly like structs;
                // prove the lowered aggregate reproduces them byte-for-byte.
                let layout = StructLayoutInfo::of_tuple(&tuple_ty);
                validate_initialized_aggregate_layout(
                    ctx,
                    mir_ty,
                    "tuple",
                    "tuple",
                    &layout,
                    tuple_ty.abi_align(),
                    tuple_ty.has_explicit_layout(),
                )?;
            }
            for field_ty in tuple_ty.get_types().iter().copied() {
                validate_initialized_global_type(ctx, field_ty, visited)?;
            }
        }
        Kind::Enum(enum_ty) => {
            if enum_ty.total_size() > 0 {
                let llvm_ty = convert_type(ctx, mir_ty)?;
                let (llvm_size, llvm_align) =
                    llvm_type_size_align(ctx, llvm_ty).ok_or_else(|| {
                        anyhow::anyhow!(
                            "initialized enum `{}` has unsupported LLVM size/alignment",
                            enum_ty.name()
                        )
                    })?;
                if llvm_size != enum_ty.total_size() || llvm_align > enum_ty.abi_align() {
                    return Err(anyhow::anyhow!(
                        "initialized enum `{}` is not byte-compatible with rustc's layout: the lowered LLVM value has size/alignment {}/{}, but rustc requires {}/{}",
                        enum_ty.name(),
                        llvm_size,
                        llvm_align,
                        enum_ty.total_size(),
                        enum_ty.abi_align()
                    ));
                }
            }
            for field_ty in enum_ty.all_field_types {
                validate_initialized_global_type(ctx, field_ty, visited)?;
            }
        }
        Kind::Array(element_ty) => {
            validate_initialized_global_type(ctx, element_ty, visited)?;
        }
        Kind::Leaf => {}
    }

    Ok(())
}

fn validate_initialized_struct_layout(
    ctx: &mut Context,
    mir_ty: TypeHandle,
    struct_ty: &MirStructType,
) -> Result<(), anyhow::Error> {
    let has_layout = struct_ty.has_explicit_layout();
    let layout = StructLayoutInfo::of_struct(struct_ty);
    let name = struct_ty.name().to_string();
    validate_initialized_aggregate_layout(
        ctx,
        mir_ty,
        "struct",
        &name,
        &layout,
        struct_ty.abi_align,
        has_layout,
    )
}

/// Shared byte-layout validation for initialized-global structs and tuples.
///
/// Proves that the lowered LLVM aggregate places every field at exactly the
/// byte offset rustc chose and matches rustc's total size, so a constant
/// initializer written slot-by-slot reproduces the host bytes.
#[allow(clippy::too_many_arguments)]
fn validate_initialized_aggregate_layout(
    ctx: &mut Context,
    mir_ty: TypeHandle,
    kind_noun: &str,
    name: &str,
    layout: &StructLayoutInfo,
    abi_align: u64,
    has_explicit_layout: bool,
) -> Result<(), anyhow::Error> {
    if layout.total_size == 0 {
        let llvm_ty = convert_type(ctx, mir_ty)?;
        let (llvm_size, _) = llvm_type_size_align(ctx, llvm_ty).ok_or_else(|| {
            anyhow::anyhow!(
                "initialized {} `{}` has unsupported LLVM size/alignment",
                kind_noun,
                name
            )
        })?;
        if llvm_size == 0 {
            return Ok(());
        }
        return Err(anyhow::anyhow!(
            "initialized {} `{}` has no stored size but lowers to {} bytes",
            kind_noun,
            name,
            llvm_size
        ));
    }
    if !has_explicit_layout {
        return Err(anyhow::anyhow!(
            "initialized {} `{}` has no rustc field-offset metadata",
            kind_noun,
            name
        ));
    }

    let slots = build_struct_slot_map(ctx, layout)?;
    let llvm_fields: Vec<_> = slots
        .llvm_struct_ty
        .deref(ctx)
        .downcast_ref::<llvm_types::StructType>()
        .expect("aggregate slot map must produce an LLVM struct")
        .fields()
        .collect();

    let mut slot_offsets = Vec::with_capacity(llvm_fields.len());
    let mut current_offset = 0u64;
    for llvm_field in &llvm_fields {
        let (field_size, field_align) =
            llvm_type_size_align(ctx, *llvm_field).ok_or_else(|| {
                anyhow::anyhow!(
                    "initialized {} `{}` field has unsupported LLVM size/alignment",
                    kind_noun,
                    name
                )
            })?;
        current_offset = current_offset.div_ceil(field_align.max(1)) * field_align.max(1);
        slot_offsets.push(current_offset);
        current_offset += field_size;
    }

    for (decl_index, slot) in slots.decl_to_llvm.iter().enumerate() {
        let Some(slot) = slot else {
            continue;
        };
        let actual_offset = slot_offsets[*slot as usize];
        let expected_offset = layout.field_offsets[decl_index];
        if actual_offset != expected_offset {
            return Err(anyhow::anyhow!(
                "initialized {} `{}` field {} lowers at byte {}, but rustc placed it at byte {}; packed and overlapping field layouts are not yet supported",
                kind_noun,
                name,
                decl_index,
                actual_offset,
                expected_offset
            ));
        }
    }

    let (llvm_size, llvm_align) =
        llvm_type_size_align(ctx, slots.llvm_struct_ty).ok_or_else(|| {
            anyhow::anyhow!(
                "initialized {} `{}` has unsupported LLVM size/alignment",
                kind_noun,
                name
            )
        })?;
    if llvm_size != layout.total_size || llvm_align > abi_align {
        return Err(anyhow::anyhow!(
            "initialized {} `{}` lowers to size/alignment {}/{}, but rustc requires {}/{}",
            kind_noun,
            name,
            llvm_size,
            llvm_align,
            layout.total_size,
            abi_align
        ));
    }

    Ok(())
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
    use dialect_mir::types::{
        EnumEncoding, EnumVariant, MirArrayType, MirEnumType, MirPtrType, MirStructType,
        MirTupleType, MirUnionType,
    };

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
    fn mir_abi_alignment_recurses_through_nested_arrays() {
        let mut ctx = make_ctx();
        let byte = mir_uint(&mut ctx, 8);
        let marker: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Align32".into(),
            vec![],
            vec![],
            vec![],
            vec![],
            0,
            32,
        )
        .into();
        let tuple: TypeHandle = MirTupleType::get_with_layout(
            &mut ctx,
            vec![marker, byte],
            vec![0, 1],
            vec![0, 0],
            32,
            32,
        )
        .into();
        let inner: TypeHandle = MirArrayType::get(&mut ctx, tuple, 2).into();
        let outer: TypeHandle = MirArrayType::get(&mut ctx, inner, 3).into();
        let plain: TypeHandle = MirArrayType::get(&mut ctx, byte, 4).into();

        assert_eq!(mir_type_abi_align(&ctx, tuple), Some(32));
        assert_eq!(mir_type_abi_align(&ctx, inner), Some(32));
        assert_eq!(mir_type_abi_align(&ctx, outer), Some(32));
        assert_eq!(mir_type_abi_align(&ctx, plain), None);
    }

    #[test]
    fn union_storage_has_exact_size_alignment_and_stride() {
        let mut ctx = make_ctx();
        let u8_ty = mir_uint(&mut ctx, 8);
        let u32_ty = mir_uint(&mut ctx, 32);
        let bytes_ty: TypeHandle = MirArrayType::get(&mut ctx, u8_ty, 4).into();
        let union_ty = MirUnionType::get(
            &mut ctx,
            "Bits".into(),
            vec!["word".into(), "bytes".into()],
            vec![u32_ty, bytes_ty],
            4,
            4,
        );
        let union_data = union_ty.deref(&ctx).clone();
        let storage = build_union_storage_type(&mut ctx, &union_data).unwrap();
        assert_eq!(llvm_type_size_align(&ctx, storage), Some((4, 4)));

        let union_handle: TypeHandle = union_ty.into();
        let array: TypeHandle = MirArrayType::get(&mut ctx, union_handle, 3).into();
        let llvm_array = convert_type(&mut ctx, array).unwrap();
        assert_eq!(llvm_type_size_align(&ctx, llvm_array), Some((12, 4)));
    }

    #[test]
    fn fixed_vector_layout_uses_packed_bit_width_and_rejects_unknown_widths() {
        let mut ctx = make_ctx();
        let i1 = llvm_int(&mut ctx, 1);
        let v16i1: TypeHandle =
            llvm_types::VectorType::get(&ctx, i1, 16, llvm_types::VectorTypeKind::Fixed).into();
        assert_eq!(
            llvm_type_size_align(&ctx, v16i1),
            Some((2, 2)),
            "<16 x i1> is 16 packed bits, not sixteen bytes"
        );

        let i8 = llvm_int(&mut ctx, 8);
        let v3i8: TypeHandle =
            llvm_types::VectorType::get(&ctx, i8, 3, llvm_types::VectorTypeKind::Fixed).into();
        assert_eq!(
            llvm_type_size_align(&ctx, v3i8),
            None,
            "the NVPTX data layout does not define a 24-bit vector alignment"
        );
    }

    #[test]
    fn enum_array_payload_keeps_exact_size_alignment_and_stride() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 8);
        let element = mir_uint(&mut ctx, 16);
        let payload: TypeHandle = MirArrayType::get(&mut ctx, element, 3).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "ArrayPayload".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::unit("Empty".into()),
                EnumVariant::new_with_layout("Data".into(), vec![payload], vec![2], vec![6]),
            ],
            0,
            8,
            2,
        )
        .into();
        let map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
        assert_eq!(llvm_type_size_align(&ctx, map.llvm_struct_ty), Some((8, 2)));

        let array: TypeHandle = MirArrayType::get(&mut ctx, enum_ty, 5).into();
        let lowered = convert_type(&mut ctx, array).unwrap();
        assert_eq!(llvm_type_size_align(&ctx, lowered), Some((40, 2)));
    }

    #[test]
    fn enum_slot_map_rejects_partial_byte_integer_carriers() {
        let mut ctx = make_ctx();
        let partial_discriminant = mir_uint(&mut ctx, 7);
        let direct: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "PartialDirect".into(),
            partial_discriminant,
            vec![0, 1],
            vec![EnumVariant::unit("A".into()), EnumVariant::unit("B".into())],
            EnumEncoding {
                tag_offset: 0,
                total_size: 1,
                abi_align: 1,
                layout_kind: EnumLayoutKind::Direct,
                carrier_kind: EnumCarrierKind::Integer,
                carrier_width: 7,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();
        let error = build_enum_slot_map(&mut ctx, direct)
            .err()
            .expect("partial-byte Direct carrier must fail closed");
        assert!(
            error.to_string().contains("whole number of bytes"),
            "{error}"
        );

        let logical = mir_uint(&mut ctx, 8);
        let payload = mir_uint(&mut ctx, 8);
        let niche: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "PartialNiche".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![payload], vec![0], vec![1]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 1,
                abi_align: 1,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Integer,
                carrier_width: 7,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();
        let error = build_enum_slot_map(&mut ctx, niche)
            .err()
            .expect("partial-byte Niche carrier must fail closed");
        assert!(
            error.to_string().contains("whole number of bytes"),
            "{error}"
        );
    }

    #[test]
    fn enum_slot_map_rejects_misaligned_inhabited_payload() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 8);
        let word = mir_uint(&mut ctx, 32);
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "PackedPayload".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::unit("Empty".into()),
                EnumVariant::new_with_layout("Data".into(), vec![word], vec![1], vec![4]),
            ],
            0,
            8,
            4,
        )
        .into();
        let error = build_enum_slot_map(&mut ctx, enum_ty)
            .err()
            .expect("misaligned payload must be rejected");
        assert!(error.to_string().contains("offset 1 is not aligned"));
    }

    #[test]
    fn enum_slot_map_uses_i8_storage_for_nonoverlapping_scalar_bool() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 32);
        let boolean: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "DirectBool".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::new_with_layout("A".into(), vec![boolean], vec![4], vec![1]),
                EnumVariant::unit("B".into()),
            ],
            0,
            8,
            4,
        )
        .into();

        let map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
        assert_eq!(map.field_slots, vec![None]);
        let fields = struct_fields(&ctx, map.llvm_struct_ty);
        assert_eq!(fields.len(), 3, "tag, canonical bool byte, tail pad");
        assert_eq!(
            fields[1]
                .deref(&ctx)
                .downcast_ref::<IntegerType>()
                .map(IntegerType::width),
            Some(8),
            "Rust bool storage must be an i8 byte, never an LLVM i1 slot"
        );
    }

    #[test]
    fn enum_slot_map_allows_nonoverlapping_aggregate_padding_without_bool() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 32);
        let byte = mir_uint(&mut ctx, 8);
        let word = mir_uint(&mut ctx, 32);
        let padded: TypeHandle = MirTupleType::get(&mut ctx, vec![byte, word]).into();
        let lowered_padded = convert_type(&mut ctx, padded).unwrap();
        assert!(
            !llvm_type_is_byte_faithful(&ctx, lowered_padded),
            "the LLVM i8/i32 tuple has harmless implicit padding"
        );
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "PaddedPayload".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::new_with_layout("Data".into(), vec![padded], vec![4], vec![8]),
                EnumVariant::unit("Empty".into()),
            ],
            0,
            12,
            4,
        )
        .into();

        let map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
        assert_eq!(map.field_slots, vec![Some(1)]);
        assert_eq!(
            llvm_type_size_align(&ctx, map.llvm_struct_ty),
            Some((12, 4))
        );
    }

    #[test]
    fn enum_slot_map_rejects_nonoverlapping_nested_bool_storage() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 32);
        let boolean: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let wrapper: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "BoolWrapper".into(),
            vec!["value".into()],
            vec![boolean],
            vec![0],
            vec![0],
            1,
            1,
        )
        .into();
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "DirectBoolWrapper".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::new_with_layout("A".into(), vec![wrapper], vec![4], vec![1]),
                EnumVariant::unit("B".into()),
            ],
            0,
            8,
            4,
        )
        .into();

        // The wrapper claims its byte-faithful twin ({i8}) and stays
        // slotless: construction canonicalizes the bool byte, extraction
        // re-loads the original {i1} shape from the canonical byte.
        let map = build_enum_slot_map(&mut ctx, enum_ty)
            .expect("nested bool storage canonicalizes through its byte-faithful twin");
        assert_eq!(map.field_slots, vec![None]);
        assert_eq!(llvm_type_size_align(&ctx, map.llvm_struct_ty), Some((8, 4)));
        let struct_fields: Vec<_> = map
            .llvm_struct_ty
            .deref(&ctx)
            .downcast_ref::<llvm_types::StructType>()
            .unwrap()
            .fields()
            .collect();
        assert!(
            struct_fields
                .iter()
                .all(|field| !llvm_type_contains_i1(&ctx, *field)),
            "enum storage must never contain physical i1"
        );
    }

    #[test]
    fn enum_slot_map_rejects_bool_hidden_by_union_byte_storage() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 32);
        let byte = mir_uint(&mut ctx, 8);
        let boolean: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let union: TypeHandle = MirUnionType::get(
            &mut ctx,
            "BoolOrByte".into(),
            vec!["flag".into(), "byte".into()],
            vec![boolean, byte],
            1,
            1,
        )
        .into();
        let lowered_union = convert_type(&mut ctx, union).unwrap();
        assert!(
            !llvm_type_contains_i1(&ctx, lowered_union),
            "the union's raw-byte carrier intentionally hides its semantic bool"
        );
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "DirectUnionBool".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::new_with_layout("A".into(), vec![union], vec![4], vec![1]),
                EnumVariant::unit("B".into()),
            ],
            0,
            8,
            4,
        )
        .into();

        let error = build_enum_slot_map(&mut ctx, enum_ty)
            .err()
            .expect("a bool hidden by union byte storage must still fail closed");
        assert!(
            error.to_string().contains("hidden behind a union"),
            "{error}"
        );
    }

    #[test]
    fn enum_slot_map_does_not_descend_through_pointer_to_bool() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 32);
        let boolean: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, boolean, false).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "PointerToBool".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::new_with_layout("A".into(), vec![pointer], vec![8], vec![8]),
                EnumVariant::unit("B".into()),
            ],
            0,
            16,
            8,
        )
        .into();

        let map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
        assert!(map.field_slots[0].is_some());
    }

    #[test]
    fn enum_slot_map_rejects_pointer_nonpointer_overlap_in_either_variant_order() {
        for pointer_first in [false, true] {
            let mut ctx = make_ctx();
            let discr = mir_uint(&mut ctx, 32);
            let u8_ty = mir_uint(&mut ctx, 8);
            let u64_ty = mir_uint(&mut ctx, 64);
            let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, u8_ty, false).into();
            let (first_name, first_ty, second_name, second_ty) = if pointer_first {
                ("Ptr", pointer, "Bits", u64_ty)
            } else {
                ("Bits", u64_ty, "Ptr", pointer)
            };
            let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
                &mut ctx,
                "PointerOrBits".into(),
                discr,
                vec![0, 1],
                vec![
                    EnumVariant::new_with_layout(
                        first_name.into(),
                        vec![first_ty],
                        vec![8],
                        vec![8],
                    ),
                    EnumVariant::new_with_layout(
                        second_name.into(),
                        vec![second_ty],
                        vec![8],
                        vec![8],
                    ),
                ],
                EnumEncoding {
                    tag_offset: 0,
                    total_size: 16,
                    abi_align: 8,
                    layout_kind: EnumLayoutKind::Direct,
                    carrier_kind: EnumCarrierKind::Integer,
                    carrier_width: 32,
                    variant_inhabited: vec![1, 1],
                    ..EnumEncoding::default()
                },
            )
            .into();
            let error = build_enum_slot_map(&mut ctx, enum_ty)
                .err()
                .expect("pointer overlap must reject");
            assert!(
                error.to_string().contains("pointer provenance"),
                "pointer_first={pointer_first}: {error}"
            );
        }
    }

    #[test]
    fn enum_slot_map_canonicalizes_overlapping_bool_wrapper_storage() {
        let mut ctx = make_ctx();
        let logical = mir_uint(&mut ctx, 8);
        let boolean: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
        let wrapper: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "BoolWrapper".into(),
            vec!["value".into()],
            vec![boolean],
            vec![0],
            vec![0],
            1,
            1,
        )
        .into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "MaybeBoolWrapper".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![wrapper], vec![0], vec![1]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 1,
                abi_align: 1,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Integer,
                carrier_width: 8,
                niche_start: 2,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();

        // Option<Wrapper(bool)>: the wrapper's byte-faithful twin ({i8})
        // shares the single canonical byte with the i8 niche carrier. The
        // wrapper stays slotless; construction zero-extends its bool.
        let map = build_enum_slot_map(&mut ctx, enum_ty)
            .expect("a bool wrapper shares its canonical byte with the i8 carrier");
        assert_eq!(map.field_slots, vec![None]);
        assert_eq!(llvm_type_size_align(&ctx, map.llvm_struct_ty), Some((1, 1)));
        assert!(
            !llvm_type_contains_i1(&ctx, map.llvm_struct_ty),
            "enum storage must never contain physical i1"
        );
    }

    #[test]
    fn enum_slot_map_canonicalizes_bool_wrapper_overlap_in_either_variant_order() {
        for wrapper_first in [false, true] {
            let mut ctx = make_ctx();
            let tag = mir_uint(&mut ctx, 32);
            let byte = mir_uint(&mut ctx, 8);
            let boolean: TypeHandle = IntegerType::get(&ctx, 1, Signedness::Signless).into();
            let wrapper: TypeHandle = MirStructType::get_with_full_layout(
                &mut ctx,
                "BoolWrapper".into(),
                vec!["value".into()],
                vec![boolean],
                vec![0],
                vec![0],
                1,
                1,
            )
            .into();
            let (first, second) = if wrapper_first {
                (wrapper, byte)
            } else {
                (byte, wrapper)
            };
            let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
                &mut ctx,
                "BoolWrapperOrByte".into(),
                tag,
                vec![0, 1],
                vec![
                    EnumVariant::new_with_layout("First".into(), vec![first], vec![4], vec![1]),
                    EnumVariant::new_with_layout("Second".into(), vec![second], vec![4], vec![1]),
                ],
                EnumEncoding {
                    tag_offset: 0,
                    total_size: 8,
                    abi_align: 4,
                    layout_kind: EnumLayoutKind::Direct,
                    carrier_kind: EnumCarrierKind::Integer,
                    carrier_width: 32,
                    variant_inhabited: vec![1, 1],
                    ..EnumEncoding::default()
                },
            )
            .into();

            // The wrapper's canonical twin ({i8}) and the plain u8 variant
            // share one byte-faithful byte, independent of declaration
            // order; the wrapper stays slotless either way.
            let map = build_enum_slot_map(&mut ctx, enum_ty)
                .expect("canonical bool bytes may share storage with a u8 variant");
            assert_eq!(llvm_type_size_align(&ctx, map.llvm_struct_ty), Some((8, 4)));
            assert!(
                !llvm_type_contains_i1(&ctx, map.llvm_struct_ty),
                "wrapper_first={wrapper_first}: enum storage must never contain physical i1"
            );
            let wrapper_flat = if wrapper_first { 0 } else { 1 };
            assert_eq!(map.field_slots[wrapper_flat], None);
        }
    }

    #[test]
    fn enum_slot_map_rejects_pointer_struct_overlapping_later_niche() {
        let mut ctx = make_ctx();
        let logical = mir_uint(&mut ctx, 8);
        let u32_ty = mir_uint(&mut ctx, 32);
        let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, false).into();
        let payload: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PointerThenNiche".into(),
            vec!["pointer".into(), "niche".into()],
            vec![pointer, u32_ty],
            vec![0, 1],
            vec![0, 8],
            16,
            8,
        )
        .into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "MaybePointerThenNiche".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![payload], vec![0], vec![16]),
            ],
            EnumEncoding {
                tag_offset: 8,
                total_size: 16,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Integer,
                carrier_width: 32,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();
        let error = build_enum_slot_map(&mut ctx, enum_ty)
            .err()
            .expect("nested pointer overlap must reject");
        assert!(error.to_string().contains("pointer provenance"));
    }

    #[test]
    fn enum_slot_map_keeps_plain_pointer_niche_in_one_pointer_slot() {
        let mut ctx = make_ctx();
        let logical = mir_uint(&mut ctx, 8);
        let u32_ty = mir_uint(&mut ctx, 32);
        let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, false).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "MaybePointer".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![pointer], vec![0], vec![8]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 8,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Pointer,
                carrier_width: 64,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();
        let map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
        assert_eq!(map.carrier_slot, Some(0));
        assert_eq!(map.field_slots, vec![Some(0)]);
        assert!(
            struct_fields(&ctx, map.llvm_struct_ty)[0]
                .deref(&ctx)
                .is::<llvm_types::PointerType>()
        );
    }

    #[test]
    fn enum_slot_map_accepts_pointer_first_and_pointer_second_aggregate_niches() {
        for pointer_first in [true, false] {
            let mut ctx = make_ctx();
            let logical = mir_uint(&mut ctx, 64);
            let index = mir_uint(&mut ctx, 64);
            let pointee = mir_uint(&mut ctx, 32);
            let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, pointee, false).into();
            let payload_types = if pointer_first {
                vec![pointer, index]
            } else {
                vec![index, pointer]
            };
            let payload: TypeHandle = MirTupleType::get(&mut ctx, payload_types).into();
            let tag_offset = if pointer_first { 0 } else { 8 };
            let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
                &mut ctx,
                "MaybeIndexedRef".into(),
                logical,
                vec![0, 1],
                vec![
                    EnumVariant::unit("None".into()),
                    EnumVariant::new_with_layout("Some".into(), vec![payload], vec![0], vec![16]),
                ],
                EnumEncoding {
                    tag_offset,
                    total_size: 16,
                    abi_align: 8,
                    layout_kind: EnumLayoutKind::Niche,
                    carrier_kind: EnumCarrierKind::Pointer,
                    carrier_width: 64,
                    untagged_variant: 1,
                    variant_inhabited: vec![1, 1],
                    ..EnumEncoding::default()
                },
            )
            .into();

            let map = build_enum_slot_map(&mut ctx, enum_ty).unwrap();
            assert_eq!(map.field_slots, vec![None]);
            assert_eq!(map.carrier_slot, Some(if pointer_first { 0 } else { 1 }));
            assert_eq!(
                llvm_type_size_align(&ctx, map.llvm_struct_ty),
                Some((16, 8))
            );
            let lowered_pointer = convert_type(&mut ctx, pointer).unwrap();
            let padding = pad(&mut ctx, 8);
            let expected = if pointer_first {
                vec![lowered_pointer, padding]
            } else {
                vec![padding, lowered_pointer]
            };
            assert_eq!(struct_fields(&ctx, map.llvm_struct_ty), expected);
        }
    }

    #[test]
    fn enum_slot_map_rejects_aggregate_with_unrepresented_pointer_leaf() {
        let mut ctx = make_ctx();
        let logical = mir_uint(&mut ctx, 64);
        let pointee = mir_uint(&mut ctx, 32);
        let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, pointee, false).into();
        let payload: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "TwoRefs".into(),
            vec!["first".into(), "second".into()],
            vec![pointer, pointer],
            vec![0, 1],
            vec![0, 8],
            16,
            8,
        )
        .into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "MaybeTwoRefs".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![payload], vec![0], vec![16]),
            ],
            EnumEncoding {
                tag_offset: 8,
                total_size: 16,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Pointer,
                carrier_width: 64,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();

        let error = build_enum_slot_map(&mut ctx, enum_ty)
            .err()
            .expect("the first pointer has no provenance-preserving storage slot");
        assert!(error.to_string().contains("pointer provenance"), "{error}");
    }

    #[test]
    fn enum_slot_map_three_field_tuple_follows_recorded_offsets() {
        // rustc lays out `(u32, f32, &T)` with the pointer first in memory:
        // ptr @ 0, u32 @ 8, f32 @ 12, size 16. With those offsets recorded on
        // the tuple, the pointer leaf coincides exactly with the niche
        // carrier at byte 0 and the payload is accepted.
        let mut ctx = make_ctx();
        let logical = mir_uint(&mut ctx, 64);
        let word = mir_uint(&mut ctx, 32);
        let float: TypeHandle = FP32Type::get(&ctx).into();
        let pointee = mir_uint(&mut ctx, 32);
        let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, pointee, false).into();
        let payload: TypeHandle = MirTupleType::get_with_layout(
            &mut ctx,
            vec![word, float, pointer],
            vec![2, 0, 1],
            vec![8, 12, 0],
            16,
            8,
        )
        .into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "MaybeMixedTuple".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![payload], vec![0], vec![16]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 16,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Pointer,
                carrier_width: 64,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();

        let map = build_enum_slot_map(&mut ctx, enum_ty)
            .expect("recorded tuple offsets prove the pointer word matches the carrier");
        assert_eq!(map.carrier_slot, Some(0));
        assert_eq!(
            llvm_type_size_align(&ctx, map.llvm_struct_ty),
            Some((16, 8))
        );

        // The same tuple with offsets that put the u32 over the pointer
        // carrier (ptr @ 8 instead) must still fail closed: integer bytes
        // may not alias pointer storage.
        let mismatched_payload: TypeHandle = MirTupleType::get_with_layout(
            &mut ctx,
            vec![word, float, pointer],
            vec![0, 1, 2],
            vec![0, 4, 8],
            16,
            8,
        )
        .into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "MaybeMixedTupleShifted".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout(
                    "Some".into(),
                    vec![mismatched_payload],
                    vec![0],
                    vec![16],
                ),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 16,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Pointer,
                carrier_width: 64,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();
        let error = build_enum_slot_map(&mut ctx, enum_ty)
            .err()
            .expect("a non-pointer word over the pointer carrier must fail closed");
        assert!(error.to_string().contains("pointer provenance"), "{error}");
    }

    #[test]
    fn enum_slot_map_rejects_shifted_nested_pointer_carrier() {
        let mut ctx = make_ctx();
        let logical = mir_uint(&mut ctx, 64);
        let index = mir_uint(&mut ctx, 64);
        let pointee = mir_uint(&mut ctx, 32);
        let pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, pointee, false).into();
        let payload: TypeHandle = MirTupleType::get(&mut ctx, vec![index, pointer]).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "ShiftedPointerCarrier".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![payload], vec![0], vec![16]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 16,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Pointer,
                carrier_width: 64,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();

        let error = build_enum_slot_map(&mut ctx, enum_ty)
            .err()
            .expect("pointer leaves at different offsets must not be conflated");
        assert!(error.to_string().contains("pointer provenance"), "{error}");
    }

    #[test]
    fn enum_slot_map_rejects_nested_carrier_address_space_mismatch() {
        let mut ctx = make_ctx();
        let logical = mir_uint(&mut ctx, 64);
        let index = mir_uint(&mut ctx, 64);
        let pointee = mir_uint(&mut ctx, 32);
        let generic_pointer: TypeHandle = MirPtrType::get_generic(&mut ctx, pointee, false).into();
        let payload: TypeHandle = MirTupleType::get(&mut ctx, vec![generic_pointer, index]).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "MismatchedPointerSpace".into(),
            logical,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new_with_layout("Some".into(), vec![payload], vec![0], vec![16]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 16,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Niche,
                carrier_kind: EnumCarrierKind::Pointer,
                carrier_width: 64,
                carrier_address_space: llvm_types::address_space::GLOBAL,
                untagged_variant: 1,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();

        let error = build_enum_slot_map(&mut ctx, enum_ty)
            .err()
            .expect("equal byte ranges in different address spaces must not alias");
        assert!(error.to_string().contains("pointer provenance"), "{error}");
    }

    #[test]
    fn enum_slot_map_rejects_nonoverlapping_shared_pointer_payload() {
        let mut ctx = make_ctx();
        let discr = mir_uint(&mut ctx, 32);
        let u32_ty = mir_uint(&mut ctx, 32);
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, u32_ty, false).into();
        let enum_ty: TypeHandle = MirEnumType::get_with_encoding(
            &mut ctx,
            "SharedPointerPayload".into(),
            discr,
            vec![0, 1],
            vec![
                EnumVariant::unit("Unit".into()),
                EnumVariant::new_with_layout("Ptr".into(), vec![shared], vec![8], vec![8]),
            ],
            EnumEncoding {
                tag_offset: 0,
                total_size: 16,
                abi_align: 8,
                layout_kind: EnumLayoutKind::Direct,
                carrier_kind: EnumCarrierKind::Integer,
                carrier_width: 32,
                variant_inhabited: vec![1, 1],
                ..EnumEncoding::default()
            },
        )
        .into();
        let error = build_enum_slot_map(&mut ctx, enum_ty)
            .err()
            .expect("shared pointer enum payload must reject");
        assert!(error.to_string().contains("target-mode dependent"));
    }

    #[test]
    fn enum_slot_map_rejects_shared_pointer_vector_payload() {
        let mut ctx = make_ctx();
        let tag = mir_uint(&mut ctx, 32);
        let shared_pointer: TypeHandle =
            llvm_types::PointerType::get(&ctx, llvm_types::address_space::SHARED).into();
        let shared_vector: TypeHandle =
            llvm_types::VectorType::get(&ctx, shared_pointer, 2, llvm_types::VectorTypeKind::Fixed)
                .into();
        let enum_ty: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "SharedPointerVector".into(),
            tag,
            vec![0, 1],
            vec![
                EnumVariant::new_with_layout(
                    "Data".into(),
                    vec![shared_vector],
                    vec![16],
                    vec![16],
                ),
                EnumVariant::unit("Empty".into()),
            ],
            0,
            32,
            16,
        )
        .into();

        let error = build_enum_slot_map(&mut ctx, enum_ty)
            .err()
            .expect("a vector of shared pointers must fail closed");
        assert!(
            error.to_string().contains("shared-memory pointer"),
            "{error}"
        );
    }

    #[test]
    fn union_storage_prefers_pointer_carrier() {
        let mut ctx = make_ctx();
        let u32_ty = mir_uint(&mut ctx, 32);
        let u64_ty = mir_uint(&mut ctx, 64);
        let ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, false).into();
        let union_ty = MirUnionType::get(
            &mut ctx,
            "PointerBits".into(),
            vec!["ptr".into(), "bits".into()],
            vec![ptr_ty, u64_ty],
            8,
            8,
        );
        let union_data = union_ty.deref(&ctx).clone();
        let storage = build_union_storage_type(&mut ctx, &union_data).unwrap();
        let fields = struct_fields(&ctx, storage);
        assert!(fields[1].deref(&ctx).is::<llvm_types::PointerType>());
        assert_eq!(llvm_type_size_align(&ctx, storage), Some((8, 8)));
    }

    #[test]
    fn union_storage_rejects_incompatible_pointer_address_spaces() {
        let mut ctx = make_ctx();
        let u32_ty = mir_uint(&mut ctx, 32);
        let generic: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, false).into();
        let shared: TypeHandle = MirPtrType::get_shared(&mut ctx, u32_ty, false).into();
        let union_ty = MirUnionType::get(
            &mut ctx,
            "MixedPointers".into(),
            vec!["generic".into(), "shared".into()],
            vec![generic, shared],
            8,
            8,
        );
        let union_data = union_ty.deref(&ctx).clone();
        let err = build_union_storage_type(&mut ctx, &union_data).unwrap_err();
        assert!(err.to_string().contains("different LLVM representations"));
    }

    #[test]
    fn union_storage_rejects_non_byte_faithful_pointer_carrier() {
        let mut ctx = make_ctx();
        let u8_ty = mir_uint(&mut ctx, 8);
        let u32_ty = mir_uint(&mut ctx, 32);
        let bool_ty = mir_uint(&mut ctx, 1);
        let ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, u32_ty, false).into();
        let ptr_bool: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "PtrBool".into(),
            vec!["ptr".into(), "flag".into()],
            vec![ptr_ty, bool_ty],
            vec![0, 1],
            vec![0, 8],
            16,
            8,
        )
        .into();
        let bytes_ty: TypeHandle = MirArrayType::get(&mut ctx, u8_ty, 16).into();
        let union_ty = MirUnionType::get(
            &mut ctx,
            "PointerBoolBytes".into(),
            vec!["view".into(), "bytes".into()],
            vec![ptr_bool, bytes_ty],
            16,
            8,
        );
        let union_data = union_ty.deref(&ctx).clone();
        let err = build_union_storage_type(&mut ctx, &union_data).unwrap_err();
        assert!(err.to_string().contains("no byte-faithful pointer carrier"));
    }

    #[test]
    fn union_storage_rejects_unrepresentable_over_alignment() {
        let mut ctx = make_ctx();
        let u32_ty = mir_uint(&mut ctx, 32);
        let union_ty = MirUnionType::get(
            &mut ctx,
            "OverAligned".into(),
            vec!["word".into()],
            vec![u32_ty],
            32,
            32,
        );
        let union_data = union_ty.deref(&ctx).clone();
        let err = build_union_storage_type(&mut ctx, &union_data).unwrap_err();
        assert!(err.to_string().contains("up to 16 bytes"));
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
        // stride @ 20, size 24. The enum now carries its own explicit
        // internal padding, so the OUTER struct needs no extra padding slot:
        //
        //   { { i8, [3 x i8], i32 }, i64, i32, i32 }
        //     layout=0                  big=1 cap=2 stride=3
        let discr = mir_uint(&mut ctx, 8);
        let payload = mir_uint(&mut ctx, 32);
        let layout_enum: TypeHandle = MirEnumType::get_with_layout(
            &mut ctx,
            "Layout".into(),
            discr,
            vec![0, 1, 2],
            vec![
                EnumVariant::unit("Aos".into()),
                EnumVariant::unit("Soa".into()),
                EnumVariant::new_with_layout("AoSoA".into(), vec![payload], vec![4], vec![4]),
            ],
            0,
            8,
            4,
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
            vec![Some(0), Some(2), Some(3), Some(1)],
            "the enum's internal pad must not create an outer struct slot"
        );

        let i8s = llvm_int(&mut ctx, 8);
        let i32s = llvm_int(&mut ctx, 32);
        let i64s = llvm_int(&mut ctx, 64);
        let enum_pad3 = pad(&mut ctx, 3);
        let enum_llvm: TypeHandle =
            llvm_types::StructType::get_unnamed(&ctx, vec![i8s, enum_pad3, i32s]).into();
        assert_eq!(
            struct_fields(&ctx, map.llvm_struct_ty),
            vec![enum_llvm, i64s, i32s, i32s]
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

    #[test]
    fn initialized_global_layout_accepts_explicit_overalignment() {
        let mut ctx = make_ctx();
        let zst = mir_zst(&mut ctx);
        validate_initialized_global_layout(&mut ctx, zst, 0, 1).unwrap();

        let byte = mir_uint(&mut ctx, 8);
        let over_aligned: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "OverAligned".into(),
            vec!["byte".into()],
            vec![byte],
            vec![0],
            vec![0],
            16,
            16,
        )
        .into();

        validate_initialized_global_layout(&mut ctx, over_aligned, 16, 16).unwrap();
    }

    #[test]
    fn initialized_global_layout_rejects_packed_and_nested_packed_structs() {
        let mut ctx = make_ctx();
        let byte = mir_uint(&mut ctx, 8);
        let word = mir_uint(&mut ctx, 32);
        let packed: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Packed".into(),
            vec!["tag".into(), "word".into()],
            vec![byte, word],
            vec![0, 1],
            vec![0, 1],
            5,
            1,
        )
        .into();

        let err = validate_initialized_global_layout(&mut ctx, packed, 5, 1).unwrap_err();
        assert!(err.to_string().contains("field 1 lowers at byte 4"));

        // Nesting must not hide the incompatible packed representation.
        let wide = mir_uint(&mut ctx, 64);
        let outer: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "Outer".into(),
            vec!["packed".into(), "wide".into()],
            vec![packed, wide],
            vec![0, 1],
            vec![0, 8],
            16,
            8,
        )
        .into();
        let err = validate_initialized_global_layout(&mut ctx, outer, 16, 8).unwrap_err();
        assert!(err.to_string().contains("lowers at byte"));
    }

    #[test]
    fn initialized_global_layout_rejects_old_union_and_tuple_models() {
        let mut ctx = make_ctx();
        let word = mir_uint(&mut ctx, 32);
        let union_as_struct: TypeHandle = MirStructType::get_with_full_layout(
            &mut ctx,
            "UnionBeforeSharedStorageLowering".into(),
            vec!["left".into(), "right".into()],
            vec![word, word],
            vec![0, 1],
            vec![0, 0],
            4,
            4,
        )
        .into();
        let err = validate_initialized_global_layout(&mut ctx, union_as_struct, 4, 4).unwrap_err();
        assert!(err.to_string().contains("field 1 lowers at byte 4"));

        // A tuple without recorded rustc layout cannot prove its bytes.
        let byte = mir_uint(&mut ctx, 8);
        let wide = mir_uint(&mut ctx, 64);
        let tuple: TypeHandle = MirTupleType::get(&mut ctx, vec![byte, wide]).into();
        let err = validate_initialized_global_layout(&mut ctx, tuple, 16, 8).unwrap_err();
        assert!(
            err.to_string()
                .contains("no stored size but lowers to 16 bytes"),
            "{err}"
        );

        // The same tuple carrying rustc's real (reordered) layout validates:
        // memory order is (wide @ 0, byte @ 8), total size 16.
        let laid_out_tuple: TypeHandle = MirTupleType::get_with_layout(
            &mut ctx,
            vec![byte, wide],
            vec![1, 0],
            vec![8, 0],
            16,
            8,
        )
        .into();
        validate_initialized_global_layout(&mut ctx, laid_out_tuple, 16, 8)
            .expect("tuples with recorded rustc offsets are provable initialized-global storage");
    }

    #[test]
    fn initialized_global_layout_rejects_enum_with_unknown_physical_layout() {
        let mut ctx = make_ctx();
        let discr = mir_uint(&mut ctx, 8);
        let payload = mir_uint(&mut ctx, 32);
        let niche: TypeHandle = MirEnumType::get(
            &mut ctx,
            "OptionNonZero".into(),
            discr,
            vec![0, 1],
            vec![
                EnumVariant::unit("None".into()),
                EnumVariant::new("Some".into(), vec![payload]),
            ],
        )
        .into();

        let err = validate_initialized_global_layout(&mut ctx, niche, 4, 4).unwrap_err();
        assert!(err.to_string().contains("unknown physical layout"));
    }
}
