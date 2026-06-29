/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR dialect types.

use pliron::builtin::type_interfaces::FloatTypeInterface;
use pliron::context::Context;
use pliron::derive::{pliron_type, type_interface_impl};
use pliron::location::Location;
use pliron::result::Error;
use pliron::r#type::{Type, TypeHandle, TypedHandle};
use pliron::utils::apfloat::{self, GetSemantics, Semantics};
use pliron::{common_traits::Verify, verify_err};

/// IEEE 754 binary16 type as it appears in Rust MIR (`f16`).
#[pliron_type(name = "mir.fp16", format, generate_get = true, verifier = "succ")]
#[derive(Hash, PartialEq, Eq, Debug)]
pub struct MirFP16Type;

#[type_interface_impl]
impl FloatTypeInterface for MirFP16Type {
    fn get_semantics(&self) -> Semantics {
        <apfloat::Half as GetSemantics>::get_semantics()
    }
}

/// A tuple type.
///
/// Represents a fixed-size collection of heterogeneous types.
/// Syntax: `mir.tuple <type1, type2, ...>`
///
/// # Verification
/// * Structural validity is ensured by `def_type` macro and parser.
/// * Inner types must be valid.
#[pliron_type(name = "mir.tuple", format = "`<` vec($types, CharSpace(`,`)) `>`")]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct MirTupleType {
    pub types: Vec<TypeHandle>,
}

impl MirTupleType {
    pub fn get(ctx: &mut Context, types: Vec<TypeHandle>) -> TypedHandle<Self> {
        Type::register_instance(MirTupleType { types }, ctx)
    }

    pub fn get_existing(ctx: &Context, types: Vec<TypeHandle>) -> Option<TypedHandle<Self>> {
        Type::get_instance(MirTupleType { types }, ctx)
    }

    pub fn get_types(&self) -> &[TypeHandle] {
        &self.types
    }
}

impl Verify for MirTupleType {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        // Tuple types are valid if their contained types are valid.
        // Structural validity is ensured by the parser/builder.
        Ok(())
    }
}

/// CUDA/PTX address space constants (matches NVPTX backend).
pub mod address_space {
    /// Generic address space (can alias any memory)
    pub const GENERIC: u32 = 0;
    /// Global device memory (VRAM)
    pub const GLOBAL: u32 = 1;
    /// Per-block shared memory (fast scratchpad)
    pub const SHARED: u32 = 3;
    /// Read-only constant memory (cached)
    pub const CONSTANT: u32 = 4;
    /// Per-thread local memory (stack/spill)
    pub const LOCAL: u32 = 5;
    /// Tensor Memory - Blackwell+ (sm_100+) tcgen05 operands
    pub const TMEM: u32 = 6;
}

/// A pointer type with mutability and address space tracking.
///
/// Represents a pointer to a value of a specific type in a specific memory space.
/// Syntax: `mir.ptr <type, mutable: bool, addrspace: u32>`
///
/// Address spaces are critical for GPU memory:
/// - 0 (generic): Can point to any memory, resolved at runtime
/// - 1 (global): Device memory (VRAM)
/// - 3 (shared): Per-block shared memory (fast scratchpad)
/// - 4 (constant): Read-only constant memory
/// - 5 (local): Per-thread local memory
/// - 6 (tmem): Tensor Memory - Blackwell+ tcgen05 operands
///
/// # Verification
/// * Pointee type must be valid.
#[pliron_type(
    name = "mir.ptr",
    format = "`<` $pointee `,` `mutable:` $is_mutable `,` `addrspace:` $address_space `>`"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct MirPtrType {
    pub pointee: TypeHandle,
    pub is_mutable: bool,
    pub address_space: u32,
}

impl MirPtrType {
    /// Create a pointer type with explicit address space.
    pub fn get(
        ctx: &mut Context,
        pointee: TypeHandle,
        is_mutable: bool,
        address_space: u32,
    ) -> TypedHandle<Self> {
        Type::register_instance(
            MirPtrType {
                pointee,
                is_mutable,
                address_space,
            },
            ctx,
        )
    }

    /// Create a pointer in generic address space (0).
    pub fn get_generic(
        ctx: &mut Context,
        pointee: TypeHandle,
        is_mutable: bool,
    ) -> TypedHandle<Self> {
        Self::get(ctx, pointee, is_mutable, address_space::GENERIC)
    }

    /// Create a pointer in shared memory address space (3).
    pub fn get_shared(
        ctx: &mut Context,
        pointee: TypeHandle,
        is_mutable: bool,
    ) -> TypedHandle<Self> {
        Self::get(ctx, pointee, is_mutable, address_space::SHARED)
    }

    /// Create a pointer in global memory address space (1).
    pub fn get_global(
        ctx: &mut Context,
        pointee: TypeHandle,
        is_mutable: bool,
    ) -> TypedHandle<Self> {
        Self::get(ctx, pointee, is_mutable, address_space::GLOBAL)
    }

    /// Create a pointer in constant memory address space (4).
    pub fn get_constant(
        ctx: &mut Context,
        pointee: TypeHandle,
        is_mutable: bool,
    ) -> TypedHandle<Self> {
        Self::get(ctx, pointee, is_mutable, address_space::CONSTANT)
    }

    /// Create a pointer in tensor memory address space (6) - Blackwell+ tcgen05.
    pub fn get_tmem(ctx: &mut Context, pointee: TypeHandle, is_mutable: bool) -> TypedHandle<Self> {
        Self::get(ctx, pointee, is_mutable, address_space::TMEM)
    }

    pub fn get_existing(
        ctx: &Context,
        pointee: TypeHandle,
        is_mutable: bool,
        address_space: u32,
    ) -> Option<TypedHandle<Self>> {
        Type::get_instance(
            MirPtrType {
                pointee,
                is_mutable,
                address_space,
            },
            ctx,
        )
    }

    pub fn is_mutable(&self) -> bool {
        self.is_mutable
    }

    pub fn address_space(&self) -> u32 {
        self.address_space
    }

    /// Check if this pointer is in shared memory (addrspace 3).
    pub fn is_shared(&self) -> bool {
        self.address_space == address_space::SHARED
    }

    /// Check if this pointer is in tensor memory (addrspace 6) - Blackwell+ tcgen05.
    pub fn is_tmem(&self) -> bool {
        self.address_space == address_space::TMEM
    }
}

impl Verify for MirPtrType {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        // Pointer types are valid if their pointee type is valid.
        Ok(())
    }
}

/// A slice type: { ptr: *T, len: usize }
///
/// Represents a view into a contiguous sequence of elements.
/// Syntax: `mir.slice <type>`
///
/// # Verification
/// * Element type must be valid.
#[pliron_type(name = "mir.slice", format = "`<` $element_ty `>`")]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct MirSliceType {
    pub element_ty: TypeHandle,
}

impl MirSliceType {
    pub fn get(ctx: &mut Context, element_ty: TypeHandle) -> TypedHandle<Self> {
        Type::register_instance(MirSliceType { element_ty }, ctx)
    }

    pub fn get_existing(ctx: &Context, element_ty: TypeHandle) -> Option<TypedHandle<Self>> {
        Type::get_instance(MirSliceType { element_ty }, ctx)
    }

    pub fn element_type(&self) -> TypeHandle {
        self.element_ty
    }
}

impl Verify for MirSliceType {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        Ok(())
    }
}

/// A disjoint slice type.
///
/// Same layout as slice, but enforces thread-local access semantics in the compiler.
/// Syntax: `mir.disjoint_slice <type>`
///
/// # Verification
/// * Element type must be valid.
#[pliron_type(name = "mir.disjoint_slice", format = "`<` $element_ty `>`")]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct MirDisjointSliceType {
    pub element_ty: TypeHandle,
}

impl MirDisjointSliceType {
    pub fn get(ctx: &mut Context, element_ty: TypeHandle) -> TypedHandle<Self> {
        Type::register_instance(MirDisjointSliceType { element_ty }, ctx)
    }

    pub fn get_existing(ctx: &Context, element_ty: TypeHandle) -> Option<TypedHandle<Self>> {
        Type::get_instance(MirDisjointSliceType { element_ty }, ctx)
    }

    pub fn element_type(&self) -> TypeHandle {
        self.element_ty
    }
}

impl Verify for MirDisjointSliceType {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        Ok(())
    }
}

/// A struct type with named fields.
///
/// Represents a product type with named, typed fields.
/// Syntax: `mir.struct <"Name", ["f0", "f1", ...], [type0, type1, ...]>`
///
/// Unlike tuples, structs have:
/// - A name (for debugging and identification)
/// - Named fields (stored as strings)
/// - Can represent any Rust struct
///
/// # Memory Layout (ABI Compatibility)
///
/// For `#[repr(Rust)]` structs, rustc may reorder fields in memory for better
/// packing. We store the exact layout from rustc to ensure host/device ABI match:
///
/// - `mem_to_decl[mem_idx]` = declaration index of field at memory position `mem_idx`
/// - `field_offsets[decl_idx]` = byte offset of field in declaration order
/// - `total_size` = total struct size including trailing padding
///
/// When lowering to LLVM, we use explicit padding arrays `[N x i8]` to match
/// the exact offsets, making the struct layout independent of LLVM's datalayout.
///
/// # Verification
/// * Field names and types must have same length.
/// * Field types must be valid.
#[pliron_type(
    name = "mir.struct",
    format = "`<` $name `,` `[` vec($field_names, CharSpace(`,`)) `]` `,` `[` vec($field_types, CharSpace(`,`)) `]` `,` `[` vec($mem_to_decl, CharSpace(`,`)) `]` `,` `[` vec($field_offsets, CharSpace(`,`)) `]` `,` $total_size `,` $abi_align `>`"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct MirStructType {
    /// The struct name (e.g., "TmemF32x32")
    pub name: String,
    /// Field names in declaration order
    pub field_names: Vec<String>,
    /// Field types in declaration order (parallel to field_names)
    pub field_types: Vec<TypeHandle>,
    /// Memory order mapping: `mem_to_decl[mem_idx] = decl_idx`.
    /// Empty means identity (no reordering).
    pub mem_to_decl: Vec<usize>,
    /// Byte offset of each field in declaration order (bytes).
    /// Empty means offsets are not known (fallback to LLVM layout).
    pub field_offsets: Vec<u64>,
    /// Total struct size in bytes (including trailing padding).
    /// 0 means size is not known (fallback to LLVM layout).
    pub total_size: u64,
    /// ABI alignment in bytes, from rustc layout. 0 means unknown.
    ///
    /// Captures `repr(align(N))` raises: over-alignment is an operation
    /// property in LLVM, so this is carried here and stamped as `align N`
    /// on loads/stores/allocas during lowering.
    pub abi_align: u64,
}

impl MirStructType {
    /// Create a new struct type with identity memory order (no reordering).
    pub fn get(
        ctx: &mut Context,
        name: String,
        field_names: Vec<String>,
        field_types: Vec<TypeHandle>,
    ) -> TypedHandle<Self> {
        Self::get_with_layout(ctx, name, field_names, field_types, vec![])
    }

    /// Create a new struct type with explicit memory order.
    ///
    /// `mem_to_decl[mem_idx]` = declaration index of field at memory position `mem_idx`.
    /// Pass empty vec for identity (declaration order = memory order).
    pub fn get_with_layout(
        ctx: &mut Context,
        name: String,
        field_names: Vec<String>,
        field_types: Vec<TypeHandle>,
        mem_to_decl: Vec<usize>,
    ) -> TypedHandle<Self> {
        Self::get_with_full_layout(
            ctx,
            name,
            field_names,
            field_types,
            mem_to_decl,
            vec![],
            0,
            0,
        )
    }

    /// Create a new struct type with complete layout information from rustc.
    ///
    /// This is the most accurate way to represent a struct - it includes exact
    /// field offsets and total size, ensuring perfect ABI compatibility.
    ///
    /// # Arguments
    /// * `mem_to_decl` - Memory order mapping (empty = identity)
    /// * `field_offsets` - Byte offset of each field in declaration order (empty = unknown)
    /// * `total_size` - Total struct size in bytes (0 = unknown)
    /// * `abi_align` - ABI alignment in bytes (0 = unknown)
    #[allow(clippy::too_many_arguments)]
    pub fn get_with_full_layout(
        ctx: &mut Context,
        name: String,
        field_names: Vec<String>,
        field_types: Vec<TypeHandle>,
        mem_to_decl: Vec<usize>,
        field_offsets: Vec<u64>,
        total_size: u64,
        abi_align: u64,
    ) -> TypedHandle<Self> {
        Type::register_instance(
            MirStructType {
                name,
                field_names,
                field_types,
                mem_to_decl,
                field_offsets,
                total_size,
                abi_align,
            },
            ctx,
        )
    }

    /// Get an existing struct type if it exists.
    pub fn get_existing(
        ctx: &Context,
        name: String,
        field_names: Vec<String>,
        field_types: Vec<TypeHandle>,
    ) -> Option<TypedHandle<Self>> {
        Type::get_instance(
            MirStructType {
                name,
                field_names,
                field_types,
                mem_to_decl: vec![],
                field_offsets: vec![],
                total_size: 0,
                abi_align: 0,
            },
            ctx,
        )
    }

    /// Get the struct name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the number of fields.
    pub fn field_count(&self) -> usize {
        self.field_types.len()
    }

    /// Get field names.
    pub fn field_names(&self) -> &[String] {
        &self.field_names
    }

    /// Get field types.
    pub fn field_types(&self) -> &[TypeHandle] {
        &self.field_types
    }

    /// Get the index of a field by name.
    pub fn get_field_index(&self, name: &str) -> Option<usize> {
        self.field_names.iter().position(|n| n == name)
    }

    /// Get the memory order mapping.
    /// Returns identity order if no explicit mapping is stored.
    pub fn memory_order(&self) -> Vec<usize> {
        if self.mem_to_decl.is_empty() {
            (0..self.field_types.len()).collect()
        } else {
            self.mem_to_decl.clone()
        }
    }

    /// Check if fields are reordered in memory.
    pub fn is_reordered(&self) -> bool {
        !self.mem_to_decl.is_empty()
    }

    /// Get field offsets in declaration order (bytes).
    /// Returns empty if offsets are not known.
    pub fn field_offsets(&self) -> &[u64] {
        &self.field_offsets
    }

    /// Get total struct size in bytes.
    /// Returns 0 if size is not known.
    pub fn total_size(&self) -> u64 {
        self.total_size
    }

    /// Check if we have explicit layout information from rustc.
    pub fn has_explicit_layout(&self) -> bool {
        !self.field_offsets.is_empty() && self.total_size > 0
    }

    /// Get the type of a field by index.
    pub fn get_field_type(&self, index: usize) -> Option<TypeHandle> {
        self.field_types.get(index).copied()
    }

    /// Get the type of a field by name.
    pub fn get_field_type_by_name(&self, name: &str) -> Option<TypeHandle> {
        self.get_field_index(name)
            .and_then(|idx| self.get_field_type(idx))
    }
}

impl Verify for MirStructType {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        // Struct types are valid if field names and types have same length.
        // This is ensured by the constructor - no runtime check needed.
        // Field types validity is checked separately.
        Ok(())
    }
}

/// A Rust union type.
///
/// Every declared field is a different typed view of the same bytes. Unlike a
/// struct, the fields are never laid out one after another. `total_size` and
/// `abi_align` come directly from rustc and describe that shared storage.
#[pliron_type(
    name = "mir.union",
    format = "`<` $name `,` `[` vec($field_names, CharSpace(`,`)) `]` `,` `[` vec($field_types, CharSpace(`,`)) `]` `,` $total_size `,` $abi_align `>`"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct MirUnionType {
    /// The source-level union name.
    pub name: String,
    /// Field names in declaration order.
    pub field_names: Vec<String>,
    /// Field types in declaration order. All fields begin at byte zero.
    pub field_types: Vec<TypeHandle>,
    /// Exact stored size in bytes, including any tail padding.
    pub total_size: u64,
    /// Exact ABI alignment in bytes.
    pub abi_align: u64,
}

impl MirUnionType {
    pub fn get(
        ctx: &mut Context,
        name: String,
        field_names: Vec<String>,
        field_types: Vec<TypeHandle>,
        total_size: u64,
        abi_align: u64,
    ) -> TypedHandle<Self> {
        Type::register_instance(
            MirUnionType {
                name,
                field_names,
                field_types,
                total_size,
                abi_align,
            },
            ctx,
        )
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn field_count(&self) -> usize {
        self.field_types.len()
    }

    pub fn field_names(&self) -> &[String] {
        &self.field_names
    }

    pub fn field_types(&self) -> &[TypeHandle] {
        &self.field_types
    }

    pub fn get_field_type(&self, index: usize) -> Option<TypeHandle> {
        self.field_types.get(index).copied()
    }

    pub fn total_size(&self) -> u64 {
        self.total_size
    }

    pub fn abi_align(&self) -> u64 {
        self.abi_align
    }
}

impl Verify for MirUnionType {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.field_names.len() != self.field_types.len() {
            return verify_err!(
                Location::Unknown,
                "MirUnionType field name/type counts must match"
            );
        }
        if self.field_types.is_empty() {
            return verify_err!(
                Location::Unknown,
                "MirUnionType must have at least one field"
            );
        }
        if self.abi_align == 0 || !self.abi_align.is_power_of_two() {
            return verify_err!(
                Location::Unknown,
                "MirUnionType ABI alignment must be a non-zero power of two"
            );
        }
        if self.total_size > 0 && !self.total_size.is_multiple_of(self.abi_align) {
            return verify_err!(
                Location::Unknown,
                "MirUnionType size must be a multiple of its ABI alignment"
            );
        }
        Ok(())
    }
}

/// A fixed-size array type.
///
/// Represents a contiguous sequence of N elements of the same type.
/// Syntax: `mir.array <type, size>`
///
/// # Verification
/// * Element type must be valid.
/// * Size must be non-zero.
#[pliron_type(name = "mir.array", format = "`<` $element_ty `,` $size `>`")]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct MirArrayType {
    pub element_ty: TypeHandle,
    pub size: u64,
}

impl MirArrayType {
    /// Create a new array type.
    pub fn get(ctx: &mut Context, element_ty: TypeHandle, size: u64) -> TypedHandle<Self> {
        Type::register_instance(MirArrayType { element_ty, size }, ctx)
    }

    /// Get an existing array type if it exists.
    pub fn get_existing(
        ctx: &Context,
        element_ty: TypeHandle,
        size: u64,
    ) -> Option<TypedHandle<Self>> {
        Type::get_instance(MirArrayType { element_ty, size }, ctx)
    }

    /// Get the element type.
    pub fn element_type(&self) -> TypeHandle {
        self.element_ty
    }

    /// Get the array size.
    pub fn size(&self) -> u64 {
        self.size
    }
}

impl Verify for MirArrayType {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        // Array types are valid if element type is valid.
        // Zero-sized arrays are technically valid in Rust ([T; 0]).
        Ok(())
    }
}

/// An enum variant definition for MirEnumType.
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct EnumVariant {
    /// Variant name (e.g., "Some", "None", "Ok", "Err")
    pub name: String,
    /// Field types for this variant (empty for unit variants like None)
    pub field_types: Vec<TypeHandle>,
    /// Where each field lives, as a byte position inside the ENUM (not
    /// inside the variant), from rustc's layout. Same order as
    /// `field_types`. Different variants reuse the same positions because
    /// they share bytes. Empty when the layout was not recorded.
    pub field_offsets: Vec<u64>,
}

impl EnumVariant {
    /// Create a new enum variant with unknown field offsets.
    pub fn new(name: String, field_types: Vec<TypeHandle>) -> Self {
        EnumVariant {
            name,
            field_types,
            field_offsets: vec![],
        }
    }

    /// Create a new enum variant carrying rustc-layout byte offsets for
    /// each field (parallel to `field_types`).
    pub fn new_with_offsets(
        name: String,
        field_types: Vec<TypeHandle>,
        field_offsets: Vec<u64>,
    ) -> Self {
        EnumVariant {
            name,
            field_types,
            field_offsets,
        }
    }

    /// Create a unit variant (no fields).
    pub fn unit(name: String) -> Self {
        EnumVariant {
            name,
            field_types: vec![],
            field_offsets: vec![],
        }
    }
}

/// An enum type (algebraic data type with multiple variants).
///
/// Represents Rust enums like `Option<T>`, `Result<T,E>`, and custom enums.
///
/// # How Rust lays out an enum, and what this type records
///
/// An enum value in memory is one tag (the "discriminant", saying which
/// variant is alive) plus that variant's payload. All variants share the
/// same bytes, because only one of them exists at a time:
///
/// ```text
/// #[repr(u32)] enum E { A(u32), B(f32), C }     8 bytes total
///
/// byte:  0         4
///        [ tag     | A's u32 ]   when the value is A
///        [ tag     | B's f32 ]   when the value is B   (same bytes!)
///        [ tag     | unused  ]   when the value is C
/// ```
///
/// This type records that layout straight from rustc: the tag's type and
/// byte position, every payload field's byte position, and the total
/// size. The lowering uses these numbers to give the enum exactly the
/// same bytes on the device as on the host, so enum data can cross the
/// kernel boundary safely.
///
/// Two things are easy to get wrong, so they are spelled out here:
///
/// - The tag stores the variant's DECLARED discriminant value, never its
///   position in the enum. For `enum E { A = 7 }`, the tag holds 7.
/// - `Option<&T>` and friends are "niche-encoded": Rust hides the tag
///   inside the payload itself (a `&T` is never null, so null can mean
///   `None`). We do not model that on the device. Such enums get a
///   separate synthetic tag instead, and `total_size` stays 0 to mean
///   "layout not recorded". That model works fine inside a kernel but
///   its bytes do NOT match the host's, so these enums are rejected at
///   the kernel boundary.
///
/// Note: variant info lives in flattened parallel vectors (the
/// `#[format_type]` macro has trouble with nested structs). Use
/// `variant_field_counts` to split the `all_*` vectors per variant.
///
/// # Verification
/// * Must have at least one variant.
/// * Discriminant type must be an integer type.
#[pliron_type(
    name = "mir.enum",
    format = "`<` $name `,` $discriminant_ty `,` `[` vec($variant_names, CharSpace(`,`)) `]` `,` `[` vec($variant_discriminants, CharSpace(`,`)) `]` `,` `[` vec($variant_field_counts, CharSpace(`,`)) `]` `,` `[` vec($all_field_types, CharSpace(`,`)) `]` `,` `[` vec($all_field_offsets, CharSpace(`,`)) `]` `,` $tag_offset `,` $total_size `,` $abi_align `>`"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct MirEnumType {
    /// The enum name (e.g., "Option", "Result")
    pub name: String,
    /// The discriminant type, sourced from rustc's layout: the tag scalar's
    /// width and signedness for Direct-tag enums (so `#[repr(uN/iN)]`,
    /// `#[repr(C)]`, sparse and negative discriminants are all honoured); a
    /// variant-count fallback for the niched / single-variant models.
    pub discriminant_ty: TypeHandle,
    /// Variant names in order
    pub variant_names: Vec<String>,
    /// Declared discriminant VALUES in variant order, as the unsigned bit
    /// pattern at tag width (e.g. `Ordering::Less` = -1 is stored as 255
    /// for an i8 tag). These are values, not variant indices.
    pub variant_discriminants: Vec<u64>,
    /// Number of fields for each variant (parallel to variant_names)
    pub variant_field_counts: Vec<u32>,
    /// All field types concatenated (use variant_field_counts to split)
    pub all_field_types: Vec<TypeHandle>,
    /// Where each field lives, as a byte position inside the enum, from
    /// rustc's layout (same order as `all_field_types`). Positions repeat
    /// across variants because variants share bytes. Empty when the
    /// layout was not recorded (`total_size == 0`).
    pub all_field_offsets: Vec<u64>,
    /// Where the tag lives, as a byte position inside the enum. Usually
    /// 0, but rustc is free to put the tag after a payload, so never
    /// assume it. Meaningful only when `total_size > 0`.
    pub tag_offset: u64,
    /// Total enum size in bytes from rustc layout (including padding).
    /// 0 means unknown / not memory-faithful; mir-lower then keeps the
    /// plain concatenated `{tag, fields...}` struct as-is.
    ///
    /// Populated only for `TagEncoding::Direct` enums: niched and
    /// single-variant shapes use an un-niched model whose size has
    /// nothing to do with rustc's layout, so they stay 0.
    pub total_size: u64,
    /// ABI alignment in bytes, from rustc layout. 0 means unknown.
    pub abi_align: u64,
}

impl MirEnumType {
    /// Create a new enum type from EnumVariant definitions.
    ///
    /// Size and alignment are left 0 ("unknown"); use
    /// [`Self::get_with_layout`] when rustc layout information is available.
    pub fn get(
        ctx: &mut Context,
        name: String,
        discriminant_ty: TypeHandle,
        variant_discriminants: Vec<u64>,
        variants: Vec<EnumVariant>,
    ) -> TypedHandle<Self> {
        Self::get_with_layout(
            ctx,
            name,
            discriminant_ty,
            variant_discriminants,
            variants,
            0,
            0,
            0,
        )
    }

    /// Create a new enum type carrying rustc's layout: where the tag
    /// lives, how big the whole enum is, and how it must be aligned (all
    /// in bytes; size/align 0 means "layout not recorded"). When a size
    /// is given, every variant must also say where its fields live
    /// (build them with [`EnumVariant::new_with_offsets`]); the verifier
    /// checks this.
    #[allow(clippy::too_many_arguments)]
    pub fn get_with_layout(
        ctx: &mut Context,
        name: String,
        discriminant_ty: TypeHandle,
        variant_discriminants: Vec<u64>,
        variants: Vec<EnumVariant>,
        tag_offset: u64,
        total_size: u64,
        abi_align: u64,
    ) -> TypedHandle<Self> {
        // Flatten variants into parallel vectors
        let mut variant_names = Vec::with_capacity(variants.len());
        let mut variant_field_counts = Vec::with_capacity(variants.len());
        let mut all_field_types = Vec::new();
        let mut all_field_offsets = Vec::new();

        for v in variants {
            variant_names.push(v.name);
            variant_field_counts.push(v.field_types.len() as u32);
            all_field_types.extend(v.field_types);
            all_field_offsets.extend(v.field_offsets);
        }

        Type::register_instance(
            MirEnumType {
                name,
                discriminant_ty,
                variant_names,
                variant_discriminants,
                variant_field_counts,
                all_field_types,
                all_field_offsets,
                tag_offset,
                total_size,
                abi_align,
            },
            ctx,
        )
    }

    /// Get the enum name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the discriminant type.
    pub fn discriminant_type(&self) -> TypeHandle {
        self.discriminant_ty
    }

    /// Get total enum size in bytes from rustc layout.
    /// Returns 0 if size is not known.
    pub fn total_size(&self) -> u64 {
        self.total_size
    }

    /// Get the ABI alignment in bytes from rustc layout.
    /// Returns 0 if alignment is not known.
    pub fn abi_align(&self) -> u64 {
        self.abi_align
    }

    /// Get the number of variants.
    pub fn variant_count(&self) -> usize {
        self.variant_names.len()
    }

    /// Get a variant by index, reconstructing EnumVariant.
    pub fn get_variant(&self, index: usize) -> Option<EnumVariant> {
        if index >= self.variant_names.len() {
            return None;
        }

        // Calculate offset into all_field_types
        let field_offset: usize = self.variant_field_counts[..index]
            .iter()
            .map(|&x| x as usize)
            .sum();
        let field_count = self.variant_field_counts[index] as usize;
        let field_types = self.all_field_types[field_offset..field_offset + field_count].to_vec();
        let field_offsets = if self.all_field_offsets.is_empty() {
            vec![]
        } else {
            self.all_field_offsets[field_offset..field_offset + field_count].to_vec()
        };

        Some(EnumVariant {
            name: self.variant_names[index].clone(),
            field_types,
            field_offsets,
        })
    }

    /// Get the rustc-layout byte offsets of a variant's fields (parallel to
    /// that variant's field types). `None` when the index is out of range
    /// or when layout is unknown (`all_field_offsets` empty).
    pub fn variant_field_offsets(&self, index: usize) -> Option<Vec<u64>> {
        if index >= self.variant_names.len() || self.all_field_offsets.is_empty() {
            return None;
        }
        let field_offset: usize = self.variant_field_counts[..index]
            .iter()
            .map(|&x| x as usize)
            .sum();
        let field_count = self.variant_field_counts[index] as usize;
        Some(self.all_field_offsets[field_offset..field_offset + field_count].to_vec())
    }

    /// Get the byte offset of the discriminant tag within the enum.
    /// Meaningful only when `total_size() > 0`.
    pub fn tag_offset(&self) -> u64 {
        self.tag_offset
    }

    /// Get the index of a variant by name.
    pub fn get_variant_index(&self, name: &str) -> Option<usize> {
        self.variant_names.iter().position(|n| n == name)
    }

    /// Get a variant by name.
    pub fn get_variant_by_name(&self, name: &str) -> Option<EnumVariant> {
        self.get_variant_index(name)
            .and_then(|idx| self.get_variant(idx))
    }

    /// Check if this is `Option<T>` type.
    pub fn is_option(&self) -> bool {
        self.name == "Option" && self.variant_names.len() == 2
    }

    /// Check if this is Result<T, E> type.
    pub fn is_result(&self) -> bool {
        self.name == "Result" && self.variant_names.len() == 2
    }
}

impl Verify for MirEnumType {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        // Enum types must have at least one variant
        if self.variant_names.is_empty() {
            return verify_err!(
                Location::Unknown,
                "MirEnumType must have at least one variant"
            );
        }
        if self.variant_names.len() != self.variant_discriminants.len() {
            return verify_err!(
                Location::Unknown,
                "MirEnumType variant discriminant count must match variant count"
            );
        }
        if self.variant_names.len() != self.variant_field_counts.len() {
            return verify_err!(
                Location::Unknown,
                "MirEnumType variant field count must match variant count"
            );
        }
        if self.total_size > 0 {
            // A recorded layout must be complete and self-consistent: one
            // byte position per field, and every position inside the
            // object. (Whether a field also FITS at its position needs
            // type sizes, which this crate does not compute; mir-lower's
            // slot map checks that part.)
            if self.all_field_offsets.len() != self.all_field_types.len() {
                return verify_err!(
                    Location::Unknown,
                    "MirEnumType with known layout must have one field offset per field"
                );
            }
            if self.tag_offset >= self.total_size {
                return verify_err!(
                    Location::Unknown,
                    "MirEnumType tag offset must lie within total_size"
                );
            }
            // `o == total_size` is legal for zero-sized fields, which rustc
            // may place at the very end of the object.
            if self.all_field_offsets.iter().any(|&o| o > self.total_size) {
                return verify_err!(
                    Location::Unknown,
                    "MirEnumType field offsets must lie within total_size"
                );
            }
        }
        Ok(())
    }
}

/// Register dialect types.
pub fn register(ctx: &mut Context) {
    MirFP16Type::register(ctx);
    MirTupleType::register(ctx);
    MirPtrType::register(ctx);
    MirSliceType::register(ctx);
    MirDisjointSliceType::register(ctx);
    MirStructType::register(ctx);
    MirUnionType::register(ctx);
    MirEnumType::register(ctx);
    MirArrayType::register(ctx);
}
