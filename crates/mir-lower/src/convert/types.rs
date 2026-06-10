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
//! Rust enums are represented as structs with the discriminant tag first,
//! then every variant's payload fields concatenated in declaration order:
//!
//! ```text
//! MIR: MirEnumType { discriminant: i8, discriminants: [0, 1], variants: [A(), B(i32)] }
//! LLVM: struct { i8, i32 }  ; tag + concatenated variant fields
//! ```
//!
//! The discriminant type is rustc's layout-truth tag (width and
//! signedness), and the tag slot stores the variant's DECLARED
//! discriminant value from `discriminants`, not its variant index.
//! When rustc's total size is known (Direct-tag enums), the struct is padded
//! with a trailing `[N x i8]` to match it; multi-payload enums whose
//! concatenation exceeds rustc's size are rejected at the kernel ABI boundary.
//! See `convert_enum_to_llvm` in this module.
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

/// Convert a `dialect-mir` type to its LLVM dialect equivalent.
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
    pub field_types: Vec<Ptr<TypeObj>>,
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
    pub llvm_struct_ty: Ptr<TypeObj>,
    /// `decl_to_llvm[decl_idx]` = LLVM slot of that declaration-order field;
    /// `None` when the field is zero-sized and was stripped.
    pub decl_to_llvm: Vec<Option<u32>>,
    /// Converted LLVM type of each declaration-order field (ZSTs included).
    pub field_llvm_types: Vec<Ptr<TypeObj>>,
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

    let mut llvm_fields: Vec<Ptr<TypeObj>> = Vec::new();
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
fn make_padding_type(ctx: &mut Context, size: u64) -> Ptr<TypeObj> {
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
fn mir_stored_size(ctx: &Context, mir_ty: Ptr<TypeObj>) -> Option<u64> {
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
pub(crate) fn llvm_type_size_align(ctx: &Context, ty: Ptr<TypeObj>) -> (u64, u64) {
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
pub(crate) fn natural_struct_layout(ctx: &Context, fields: &[Ptr<TypeObj>]) -> (u64, u64, u64) {
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

/// Convert a `MirEnumType` to its LLVM struct representation.
///
/// The base model is the concatenated struct `{tag, variant fields...}`
/// (field 0 is the discriminant; every variant's payload fields follow in
/// declaration order). When rustc's total size is known (`total_size > 0`,
/// i.e. Direct-tag enums) the structural size is compared against it:
///
/// - equal: the concatenated struct already matches; return it unchanged.
/// - structural < total (trailing shortfall: `repr(align(N))` raises, unit
///   variants alongside sized payloads elsewhere): append a trailing
///   `[N x i8]` pad so the LLVM allocation size equals rustc's. Appending at
///   the END keeps every existing insertvalue/extractvalue index valid; the
///   pad is simply never written.
/// - structural > total (multi-payload enums whose variants overlap in Rust
///   but concatenate in this model, e.g. `#[repr(u32)] enum E { A(u32),
///   B(u32) }` is 8 bytes in Rust vs 12 structural): padding is impossible,
///   so the concatenated struct is returned as-is. It stays self-consistent
///   for ALL device-local use (construct, match, allocas, loads/stores and
///   GEPs over device-written memory), but payload field OFFSETS remain the
///   concatenated model, NOT Rust's overlapped layout. Such enums are
///   therefore rejected at the kernel ABI boundary, where the host side is
///   laid out with rustc's real layout; see [`enum_memory_divergence`] and
///   [`find_divergent_enum_in_abi`].
pub(crate) fn convert_enum_to_llvm(
    ctx: &mut Context,
    ty: Ptr<TypeObj>,
) -> Result<Ptr<TypeObj>, anyhow::Error> {
    let (discriminant_ty, all_field_types, total_size) = {
        let ty_ref = ty.deref(ctx);
        let enum_ty = ty_ref
            .downcast_ref::<MirEnumType>()
            .ok_or_else(|| anyhow::anyhow!("convert_enum_to_llvm: expected MirEnumType"))?;
        (
            enum_ty.discriminant_ty,
            enum_ty.all_field_types.clone(),
            enum_ty.total_size(),
        )
    };

    let llvm_discr_ty = convert_type(ctx, discriminant_ty)?;
    let mut llvm_fields = vec![llvm_discr_ty];
    for field_ty in all_field_types {
        llvm_fields.push(convert_type(ctx, field_ty)?);
    }

    if total_size > 0 {
        let (end, size, _align) = natural_struct_layout(ctx, &llvm_fields);
        if size < total_size {
            let padding_ty = make_padding_type(ctx, total_size - end);
            llvm_fields.push(padding_ty);
        }
    }

    Ok(llvm_types::StructType::get_unnamed(ctx, llvm_fields).into())
}

/// Detect Direct-tag enums whose concatenated `{tag, fields...}` model cannot
/// match rustc's memory layout: the structural size exceeds rustc's total
/// size because several variants carry payloads that overlap in Rust but are
/// concatenated in our model. Such an enum cannot be padded into shape, and
/// traversing memory with it (GEP stride, load/store width) would silently
/// read or write the wrong bytes, so callers reject it loudly instead.
///
/// Returns `Some(enum_name)` for a divergent enum, `None` when `ty` is not a
/// `MirEnumType` or its layout is memory-faithful (possibly after padding).
pub(crate) fn enum_memory_divergence(
    ctx: &mut Context,
    ty: Ptr<TypeObj>,
) -> Result<Option<String>, anyhow::Error> {
    let info = {
        let ty_ref = ty.deref(ctx);
        ty_ref.downcast_ref::<MirEnumType>().map(|enum_ty| {
            (
                enum_ty.name().to_string(),
                enum_ty.discriminant_ty,
                enum_ty.all_field_types.clone(),
                enum_ty.total_size(),
            )
        })
    };
    let Some((name, discriminant_ty, all_field_types, total_size)) = info else {
        return Ok(None);
    };
    if total_size == 0 {
        // Size unknown (niched / single-variant model): the un-niched model
        // is deliberately self-consistent; nothing to check against.
        return Ok(None);
    }

    let llvm_discr_ty = convert_type(ctx, discriminant_ty)?;
    let mut llvm_fields = vec![llvm_discr_ty];
    for field_ty in all_field_types {
        llvm_fields.push(convert_type(ctx, field_ty)?);
    }
    let (_end, size, _align) = natural_struct_layout(ctx, &llvm_fields);

    Ok((size > total_size).then_some(name))
}

/// Find a layout-divergent enum reachable from `ty` through the kernel ABI.
///
/// Walks the pre-conversion MIR type tree: pointer pointees, slice and
/// array elements, struct/tuple fields, and enum payload fields. Returns
/// the first divergent enum's name (see [`enum_memory_divergence`]), or
/// `None` when the whole tree is memory-faithful.
///
/// Used only for KERNEL signatures: a kernel parameter is laid out by the
/// host with rustc's real layout (by value via `cuLaunchKernel`, or behind
/// pointers/slices into `DeviceBuffer` memory), so a divergent enum there
/// means host and device disagree on stride and field offsets. Device-local
/// use of the same enum (locals, construct, match, loads/stores of allocas)
/// is self-consistent because every access uses the same lowered struct
/// type, and is deliberately NOT rejected.
///
/// `visited` breaks cycles through recursive types (`Ptr<TypeObj>` is
/// interned, so equality is identity).
pub(crate) fn find_divergent_enum_in_abi(
    ctx: &mut Context,
    ty: Ptr<TypeObj>,
    visited: &mut Vec<Ptr<TypeObj>>,
) -> Result<Option<String>, anyhow::Error> {
    if visited.contains(&ty) {
        return Ok(None);
    }
    visited.push(ty);

    if let Some(name) = enum_memory_divergence(ctx, ty)? {
        return Ok(Some(name));
    }

    let children: Vec<Ptr<TypeObj>> = {
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
        if let Some(name) = find_divergent_enum_in_abi(ctx, child, visited)? {
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
pub(crate) fn make_slice_struct(ctx: &mut Context) -> Ptr<TypeObj> {
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
    fn mir_uint(ctx: &mut Context, width: u32) -> Ptr<TypeObj> {
        IntegerType::get(ctx, width, Signedness::Unsigned).into()
    }

    /// A converted (signless) LLVM integer type.
    fn llvm_int(ctx: &mut Context, width: u32) -> Ptr<TypeObj> {
        IntegerType::get(ctx, width, Signedness::Signless).into()
    }

    /// `[n x i8]` padding type, as `make_padding_type` builds it.
    fn pad(ctx: &mut Context, n: u64) -> Ptr<TypeObj> {
        make_padding_type(ctx, n)
    }

    /// A zero-sized MIR struct (PhantomData shape).
    fn mir_zst(ctx: &mut Context) -> Ptr<TypeObj> {
        MirStructType::get(ctx, "Phantom".into(), vec![], vec![]).into()
    }

    fn struct_fields(ctx: &Context, ty: Ptr<TypeObj>) -> Vec<Ptr<TypeObj>> {
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
        // The exact shape from issue #128 (examples/struct_layout_repro):
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
        let layout_enum: Ptr<TypeObj> = MirEnumType::get(
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
        let enum_llvm: Ptr<TypeObj> =
            llvm_types::StructType::get_unnamed(&mut ctx, vec![i8s, i32s]).into();
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
        let inner: Ptr<TypeObj> = MirStructType::get_with_full_layout(
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
