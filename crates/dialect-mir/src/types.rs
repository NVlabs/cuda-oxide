/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR dialect types.

use pliron::builtin::{type_interfaces::FloatTypeInterface, types::IntegerType};
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
    /// Exact rustc storage size of each field, in bytes. This is kept next
    /// to the offsets so the enum verifier can prove every physical field
    /// access stays inside the object, including zero-sized fields placed at
    /// `total_size`.
    pub field_sizes: Vec<u64>,
}

impl EnumVariant {
    /// Create a new enum variant with unknown field offsets.
    pub fn new(name: String, field_types: Vec<TypeHandle>) -> Self {
        EnumVariant {
            name,
            field_types,
            field_offsets: vec![],
            field_sizes: vec![],
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
            field_sizes: vec![],
        }
    }

    /// Create a variant carrying complete rustc field layout information.
    pub fn new_with_layout(
        name: String,
        field_types: Vec<TypeHandle>,
        field_offsets: Vec<u64>,
        field_sizes: Vec<u64>,
    ) -> Self {
        EnumVariant {
            name,
            field_types,
            field_offsets,
            field_sizes,
        }
    }

    /// Create a unit variant (no fields).
    pub fn unit(name: String) -> Self {
        EnumVariant {
            name,
            field_types: vec![],
            field_offsets: vec![],
            field_sizes: vec![],
        }
    }
}

/// An enum type (algebraic data type with multiple variants).
///
/// Represents Rust enums like `Option<T>`, `Result<T,E>`, and custom enums.
///
/// # How Rust lays out an enum, and what this type records
///
/// A direct-tag enum stores a tag plus the active payload. All variants share
/// the same bytes, because only one of them exists at a time:
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
/// - `Option<&T>` and friends are "niche-encoded": Rust stores the variant in
///   an otherwise-invalid payload value (null means `None`). We record that
///   physical carrier, its absolute byte offset, and rustc's wrapping range
///   arithmetic. No extra device-only tag is introduced.
/// - `Single` and `Empty` layouts have no carrier. Their logical
///   discriminants are computed as constants, and impossible variants remain
///   explicitly uninhabited.
///
/// Note: variant info lives in flattened parallel vectors (the
/// `#[format_type]` macro has trouble with nested structs). Use
/// `variant_field_counts` to split the `all_*` vectors per variant.
///
/// # Verification
/// * Only an `Empty` layout may have zero source variants.
/// * Discriminant type must be an integer type.
///
/// Values used by [`MirEnumType::layout_kind`].
///
/// These are integers rather than a Rust enum because `#[pliron_type]` needs
/// every field to have the standard textual parser/formatter.
pub mod enum_layout_kind {
    pub const UNKNOWN: u8 = 0;
    pub const DIRECT: u8 = 1;
    pub const NICHE: u8 = 2;
    pub const SINGLE: u8 = 3;
    pub const EMPTY: u8 = 4;
}

/// Values used by [`MirEnumType::carrier_kind`].
pub mod enum_carrier_kind {
    pub const NONE: u8 = 0;
    pub const INTEGER: u8 = 1;
    pub const POINTER: u8 = 2;
}

#[pliron_type(
    name = "mir.enum",
    format = "`<` $name `,` $discriminant_ty `,` `[` vec($variant_names, CharSpace(`,`)) `]` `,` `[` vec($variant_discriminants, CharSpace(`,`)) `]` `,` `[` vec($variant_field_counts, CharSpace(`,`)) `]` `,` `[` vec($all_field_types, CharSpace(`,`)) `]` `,` `[` vec($all_field_offsets, CharSpace(`,`)) `]` `,` `[` vec($all_field_sizes, CharSpace(`,`)) `]` `,` $tag_offset `,` $total_size `,` $abi_align `,` $layout_kind `,` $carrier_kind `,` $carrier_width `,` $carrier_address_space `,` $niche_start_lo `,` $niche_start_hi `,` $niche_variant_start `,` $niche_variant_end `,` $untagged_variant `,` $single_variant `,` `[` vec($variant_inhabited, CharSpace(`,`)) `]` `>`"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct MirEnumType {
    /// The enum name (e.g., "Option", "Result")
    pub name: String,
    /// The discriminant type, sourced from rustc's layout: the tag scalar's
    /// width and signedness for Direct-tag enums (so `#[repr(uN/iN)]`,
    /// `#[repr(C)]`, sparse and negative discriminants are all honoured); a
    /// logical declared discriminant type for Niche/Single/Empty layouts.
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
    /// across variants because variants share bytes. Empty only for a legacy
    /// `Unknown` layout or an enum with no fields.
    pub all_field_offsets: Vec<u64>,
    /// Exact rustc storage sizes for fields, parallel to
    /// [`Self::all_field_types`] and [`Self::all_field_offsets`].
    pub all_field_sizes: Vec<u64>,
    /// Where the tag lives, as a byte position inside the enum. Usually
    /// 0, but rustc is free to put the tag after a payload, so never
    /// assume it. Meaningful only when `total_size > 0`.
    pub tag_offset: u64,
    /// Total enum size in bytes from rustc layout (including padding). Zero is
    /// a valid known size for ZST Single/Empty layouts; `layout_kind`, not this
    /// number, distinguishes known layout from legacy `Unknown`.
    pub total_size: u64,
    /// ABI alignment in bytes, from rustc layout. 0 means unknown.
    pub abi_align: u64,
    /// Physical rustc layout classification. See [`enum_layout_kind`].
    pub layout_kind: u8,
    /// Physical tag/niche carrier classification. See [`enum_carrier_kind`].
    pub carrier_kind: u8,
    /// Physical carrier width in bits. Zero when there is no carrier.
    pub carrier_width: u32,
    /// Address space of a pointer carrier value. Zero for integer/no carrier.
    pub carrier_address_space: u32,
    /// Low and high halves of rustc's lossless `u128` niche start.
    pub niche_start_lo: u64,
    pub niche_start_hi: u64,
    /// Inclusive variant-index interval described by rustc's niche encoding.
    pub niche_variant_start: u32,
    pub niche_variant_end: u32,
    /// Variant represented by an ordinary valid carrier value.
    pub untagged_variant: u32,
    /// rustc-selected source variant for a `Single` layout. Its fields can
    /// still make that sole variant uninhabited.
    pub single_variant: u32,
    /// One byte per source variant: 1 means proven inhabited, 0 uninhabited.
    pub variant_inhabited: Vec<u8>,
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

    /// Legacy convenience constructor for a Direct layout when `total_size`
    /// is nonzero, or an explicitly Unknown layout otherwise. New importer
    /// code should use [`Self::get_with_encoding`] so zero-sized known layouts
    /// and niche carriers remain unambiguous. Known variants must use
    /// [`EnumVariant::new_with_layout`].
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
        let variant_inhabited = vec![1; variants.len()];
        let layout_kind = if total_size > 0 {
            enum_layout_kind::DIRECT
        } else {
            enum_layout_kind::UNKNOWN
        };
        let carrier_width = if layout_kind == enum_layout_kind::DIRECT {
            discriminant_ty
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .map(IntegerType::width)
                .unwrap_or(0)
        } else {
            0
        };
        Self::get_with_encoding(
            ctx,
            name,
            discriminant_ty,
            variant_discriminants,
            variants,
            tag_offset,
            total_size,
            abi_align,
            layout_kind,
            if layout_kind == enum_layout_kind::DIRECT {
                enum_carrier_kind::INTEGER
            } else {
                enum_carrier_kind::NONE
            },
            carrier_width,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            variant_inhabited,
        )
    }

    /// Create an enum type with its complete physical rustc encoding.
    #[allow(clippy::too_many_arguments)]
    pub fn get_with_encoding(
        ctx: &mut Context,
        name: String,
        discriminant_ty: TypeHandle,
        variant_discriminants: Vec<u64>,
        variants: Vec<EnumVariant>,
        tag_offset: u64,
        total_size: u64,
        abi_align: u64,
        layout_kind: u8,
        carrier_kind: u8,
        carrier_width: u32,
        carrier_address_space: u32,
        niche_start_lo: u64,
        niche_start_hi: u64,
        niche_variant_start: u32,
        niche_variant_end: u32,
        untagged_variant: u32,
        single_variant: u32,
        variant_inhabited: Vec<u8>,
    ) -> TypedHandle<Self> {
        // Flatten variants into parallel vectors
        let mut variant_names = Vec::with_capacity(variants.len());
        let mut variant_field_counts = Vec::with_capacity(variants.len());
        let mut all_field_types = Vec::new();
        let mut all_field_offsets = Vec::new();
        let mut all_field_sizes = Vec::new();

        for v in variants {
            variant_names.push(v.name);
            variant_field_counts.push(v.field_types.len() as u32);
            all_field_types.extend(v.field_types);
            all_field_offsets.extend(v.field_offsets);
            all_field_sizes.extend(v.field_sizes);
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
                all_field_sizes,
                tag_offset,
                total_size,
                abi_align,
                layout_kind,
                carrier_kind,
                carrier_width,
                carrier_address_space,
                niche_start_lo,
                niche_start_hi,
                niche_variant_start,
                niche_variant_end,
                untagged_variant,
                single_variant,
                variant_inhabited,
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
        let field_offset: usize = self
            .variant_field_counts
            .get(..index)?
            .iter()
            .try_fold(0usize, |sum, &count| sum.checked_add(count as usize))?;
        let field_count = *self.variant_field_counts.get(index)? as usize;
        let field_end = field_offset.checked_add(field_count)?;
        let field_types = self.all_field_types.get(field_offset..field_end)?.to_vec();
        let field_offsets = if self.all_field_offsets.is_empty() {
            vec![]
        } else {
            self.all_field_offsets
                .get(field_offset..field_end)?
                .to_vec()
        };
        let field_sizes = if self.all_field_sizes.is_empty() {
            vec![]
        } else {
            self.all_field_sizes.get(field_offset..field_end)?.to_vec()
        };

        Some(EnumVariant {
            name: self.variant_names[index].clone(),
            field_types,
            field_offsets,
            field_sizes,
        })
    }

    /// Get the rustc-layout byte offsets of a variant's fields (parallel to
    /// that variant's field types). `None` when the index is out of range
    /// or when layout is unknown (`all_field_offsets` empty).
    pub fn variant_field_offsets(&self, index: usize) -> Option<Vec<u64>> {
        if index >= self.variant_names.len() || self.all_field_offsets.is_empty() {
            return None;
        }
        let field_offset: usize = self
            .variant_field_counts
            .get(..index)?
            .iter()
            .try_fold(0usize, |sum, &count| sum.checked_add(count as usize))?;
        let field_count = *self.variant_field_counts.get(index)? as usize;
        let field_end = field_offset.checked_add(field_count)?;
        Some(
            self.all_field_offsets
                .get(field_offset..field_end)?
                .to_vec(),
        )
    }

    /// Get the byte offset of the discriminant tag within the enum.
    /// Meaningful only when `total_size() > 0`.
    pub fn tag_offset(&self) -> u64 {
        self.tag_offset
    }

    pub fn niche_start(&self) -> u128 {
        (u128::from(self.niche_start_hi) << 64) | u128::from(self.niche_start_lo)
    }

    pub fn variant_is_inhabited(&self, index: usize) -> Option<bool> {
        self.variant_inhabited.get(index).map(|value| *value != 0)
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
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        // Rust zero-variant enums are real values at the type level. They are
        // valid only with rustc's Empty physical layout; every other layout
        // needs at least one source variant.
        if self.variant_names.is_empty() && self.layout_kind != enum_layout_kind::EMPTY {
            return verify_err!(
                Location::Unknown,
                "Only an Empty MirEnumType may have zero variants"
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
        if self.variant_names.len() != self.variant_inhabited.len() {
            return verify_err!(
                Location::Unknown,
                "MirEnumType variant inhabitedness count must match variant count"
            );
        }
        if self.variant_inhabited.iter().any(|value| *value > 1) {
            return verify_err!(
                Location::Unknown,
                "MirEnumType inhabitedness values must be 0 or 1"
            );
        }
        let Some(field_count) = self
            .variant_field_counts
            .iter()
            .try_fold(0usize, |sum, count| sum.checked_add(*count as usize))
        else {
            return verify_err!(Location::Unknown, "MirEnumType field count overflows usize");
        };
        if field_count != self.all_field_types.len() {
            return verify_err!(
                Location::Unknown,
                "MirEnumType flattened field count must match its field types"
            );
        }
        if !matches!(
            self.layout_kind,
            enum_layout_kind::UNKNOWN
                | enum_layout_kind::DIRECT
                | enum_layout_kind::NICHE
                | enum_layout_kind::SINGLE
                | enum_layout_kind::EMPTY
        ) {
            return verify_err!(Location::Unknown, "MirEnumType has invalid layout kind");
        }
        if !matches!(
            self.carrier_kind,
            enum_carrier_kind::NONE | enum_carrier_kind::INTEGER | enum_carrier_kind::POINTER
        ) {
            return verify_err!(Location::Unknown, "MirEnumType has invalid carrier kind");
        }
        if self.carrier_kind == enum_carrier_kind::INTEGER && !self.carrier_width.is_multiple_of(8)
        {
            return verify_err!(
                Location::Unknown,
                "MirEnumType integer carrier width must be a whole number of bytes"
            );
        }
        let Some(logical_width) = self
            .discriminant_ty
            .deref(ctx)
            .downcast_ref::<IntegerType>()
            .map(IntegerType::width)
        else {
            return verify_err!(
                Location::Unknown,
                "MirEnumType discriminant must be an integer"
            );
        };
        if logical_width == 0 || logical_width > 128 {
            return verify_err!(
                Location::Unknown,
                "MirEnumType discriminant must be 1..=128 bits"
            );
        }
        if logical_width < 64
            && self
                .variant_discriminants
                .iter()
                .any(|value| *value >= (1u64 << logical_width))
        {
            return verify_err!(
                Location::Unknown,
                "MirEnumType declared discriminant does not fit its logical type"
            );
        }

        let variant_count = self.variant_names.len() as u32;
        match self.layout_kind {
            enum_layout_kind::DIRECT => {
                if self.carrier_kind != enum_carrier_kind::INTEGER
                    || self.carrier_width == 0
                    || self.carrier_width > 64
                {
                    return verify_err!(
                        Location::Unknown,
                        "MirEnumType direct layout requires a 1..=64-bit integer carrier because declared discriminants are stored as u64"
                    );
                }
                if logical_width != self.carrier_width {
                    return verify_err!(
                        Location::Unknown,
                        "MirEnumType direct discriminant width must match its carrier"
                    );
                }
                if self.niche_start() != 0
                    || self.niche_variant_start != 0
                    || self.niche_variant_end != 0
                    || self.untagged_variant != 0
                    || self.single_variant != 0
                {
                    return verify_err!(
                        Location::Unknown,
                        "MirEnumType direct layout cannot contain niche/single metadata"
                    );
                }
            }
            enum_layout_kind::NICHE => {
                if !matches!(
                    self.carrier_kind,
                    enum_carrier_kind::INTEGER | enum_carrier_kind::POINTER
                ) || self.carrier_width == 0
                    || self.carrier_width > 128
                {
                    return verify_err!(
                        Location::Unknown,
                        "MirEnumType niche layout requires a 1..=128-bit integer or pointer carrier"
                    );
                }
                if self.niche_variant_start > self.niche_variant_end
                    || self.niche_variant_end >= variant_count
                    || self.untagged_variant >= variant_count
                {
                    return verify_err!(
                        Location::Unknown,
                        "MirEnumType niche variant metadata is out of bounds"
                    );
                }
                if self.variant_inhabited[self.untagged_variant as usize] == 0 {
                    return verify_err!(
                        Location::Unknown,
                        "MirEnumType niche untagged variant must be inhabited"
                    );
                }
                if self
                    .variant_discriminants
                    .iter()
                    .enumerate()
                    .any(|(index, value)| *value != index as u64)
                {
                    return verify_err!(
                        Location::Unknown,
                        "MirEnumType niche layouts require variant indices to equal declared discriminants"
                    );
                }
                if self.carrier_width < 128 && self.niche_start() >= (1u128 << self.carrier_width) {
                    return verify_err!(
                        Location::Unknown,
                        "MirEnumType niche start does not fit the physical carrier"
                    );
                }
                let niche_span = u128::from(self.niche_variant_end - self.niche_variant_start);
                if self.carrier_width < 128 && niche_span >= (1u128 << self.carrier_width) {
                    return verify_err!(
                        Location::Unknown,
                        "MirEnumType niche range is wider than the carrier and would repeat encodings"
                    );
                }
                for (index, inhabited) in self.variant_inhabited.iter().enumerate() {
                    if *inhabited != 0
                        && index as u32 != self.untagged_variant
                        && !(self.niche_variant_start..=self.niche_variant_end)
                            .contains(&(index as u32))
                    {
                        return verify_err!(
                            Location::Unknown,
                            "MirEnumType niche layout has an inhabited unrepresentable variant"
                        );
                    }
                }
                if self.single_variant != 0 {
                    return verify_err!(
                        Location::Unknown,
                        "MirEnumType niche layout cannot contain single-variant metadata"
                    );
                }
            }
            enum_layout_kind::SINGLE => {
                if self.carrier_kind != enum_carrier_kind::NONE
                    || self.single_variant >= variant_count
                    || self
                        .variant_inhabited
                        .iter()
                        .enumerate()
                        .any(|(index, value)| index != self.single_variant as usize && *value != 0)
                {
                    return verify_err!(
                        Location::Unknown,
                        "MirEnumType single layout inhabitedness is inconsistent"
                    );
                }
                if self.tag_offset != 0
                    || self.niche_start() != 0
                    || self.niche_variant_start != 0
                    || self.niche_variant_end != 0
                    || self.untagged_variant != 0
                {
                    return verify_err!(
                        Location::Unknown,
                        "MirEnumType single layout cannot contain carrier/niche metadata"
                    );
                }
            }
            enum_layout_kind::EMPTY => {
                if self.carrier_kind != enum_carrier_kind::NONE
                    || self.variant_inhabited.iter().any(|value| *value != 0)
                {
                    return verify_err!(
                        Location::Unknown,
                        "MirEnumType empty layout cannot contain an inhabited variant or carrier"
                    );
                }
                if self.tag_offset != 0
                    || self.niche_start() != 0
                    || self.niche_variant_start != 0
                    || self.niche_variant_end != 0
                    || self.untagged_variant != 0
                    || self.single_variant != 0
                {
                    return verify_err!(
                        Location::Unknown,
                        "MirEnumType empty layout cannot contain encoding metadata"
                    );
                }
            }
            enum_layout_kind::UNKNOWN => {}
            _ => unreachable!(),
        }
        if self.carrier_kind != enum_carrier_kind::POINTER && self.carrier_address_space != 0 {
            return verify_err!(
                Location::Unknown,
                "Only pointer enum carriers may have a nonzero address space"
            );
        }
        if self.carrier_kind == enum_carrier_kind::NONE && self.carrier_width != 0 {
            return verify_err!(
                Location::Unknown,
                "An enum without a carrier must have zero carrier width"
            );
        }
        if self.carrier_kind == enum_carrier_kind::POINTER {
            // Lowering is target-mode agnostic. Shared pointers are 64-bit
            // under PTX/legacy data layouts but 32-bit under modern NVVM.
            if self.carrier_address_space == 3 {
                return verify_err!(
                    Location::Unknown,
                    "Shared-memory pointer enum carriers are target-mode dependent and unsupported"
                );
            }
            if self.carrier_width != 64 {
                return verify_err!(
                    Location::Unknown,
                    "Non-shared pointer enum carriers must be 64 bits"
                );
            }
        }

        if self.layout_kind == enum_layout_kind::UNKNOWN {
            if self.total_size != 0
                || self.abi_align != 0
                || self.carrier_kind != enum_carrier_kind::NONE
                || self.carrier_width != 0
                || self.carrier_address_space != 0
                || self.tag_offset != 0
                || self.niche_start() != 0
                || self.niche_variant_start != 0
                || self.niche_variant_end != 0
                || self.untagged_variant != 0
                || self.single_variant != 0
            {
                return verify_err!(
                    Location::Unknown,
                    "MirEnumType unknown layout cannot contain physical ABI metadata"
                );
            }
        } else {
            // A recorded layout must be complete and self-consistent: one
            // byte position and rustc storage size per field.
            if self.all_field_offsets.len() != self.all_field_types.len() {
                return verify_err!(
                    Location::Unknown,
                    "MirEnumType with known layout must have one field offset per field"
                );
            }
            if self.all_field_sizes.len() != self.all_field_types.len() {
                return verify_err!(
                    Location::Unknown,
                    "MirEnumType known layout must record one rustc storage size per field"
                );
            }
            if self.abi_align == 0
                || !self.abi_align.is_power_of_two()
                || (self.total_size > 0 && !self.total_size.is_multiple_of(self.abi_align))
            {
                return verify_err!(
                    Location::Unknown,
                    "MirEnumType known layout requires a non-zero power-of-two ABI alignment"
                );
            }
            if matches!(
                self.layout_kind,
                enum_layout_kind::DIRECT | enum_layout_kind::NICHE
            ) && self
                .tag_offset
                .checked_add(u64::from(self.carrier_width).div_ceil(8))
                .is_none_or(|end| end > self.total_size)
            {
                return verify_err!(
                    Location::Unknown,
                    "MirEnumType carrier must fit within total_size"
                );
            }
            let carrier_end = self.tag_offset + u64::from(self.carrier_width).div_ceil(8);
            if self.layout_kind == enum_layout_kind::NICHE {
                let untagged = self.untagged_variant as usize;
                let Some(untagged_base) =
                    self.variant_field_counts
                        .get(..untagged)
                        .and_then(|counts| {
                            counts
                                .iter()
                                .try_fold(0usize, |sum, count| sum.checked_add(*count as usize))
                        })
                else {
                    return verify_err!(
                        Location::Unknown,
                        "MirEnumType niche untagged field range is malformed"
                    );
                };
                let untagged_end = untagged_base + self.variant_field_counts[untagged] as usize;
                let carrier_is_in_untagged_payload = self.all_field_offsets
                    [untagged_base..untagged_end]
                    .iter()
                    .zip(&self.all_field_sizes[untagged_base..untagged_end])
                    .any(|(&offset, &size)| {
                        size != 0
                            && offset <= self.tag_offset
                            && offset
                                .checked_add(size)
                                .is_some_and(|end| end >= carrier_end)
                    });
                if !carrier_is_in_untagged_payload {
                    return verify_err!(
                        Location::Unknown,
                        "MirEnumType niche carrier must be contained in the untagged variant's payload"
                    );
                }
            }
            let mut flat_field = 0usize;
            for (variant, field_count) in self.variant_field_counts.iter().enumerate() {
                let end = flat_field + *field_count as usize;
                if self.variant_inhabited[variant] != 0 {
                    let mut occupied = Vec::<(u64, u64)>::new();
                    for (&offset, &size) in self.all_field_offsets[flat_field..end]
                        .iter()
                        .zip(&self.all_field_sizes[flat_field..end])
                    {
                        let Some(field_end) = offset.checked_add(size) else {
                            return verify_err!(
                                Location::Unknown,
                                "MirEnumType inhabited field storage overflows"
                            );
                        };
                        if field_end > self.total_size {
                            return verify_err!(
                                Location::Unknown,
                                "MirEnumType inhabited field storage must fit within total_size"
                            );
                        }
                        if size == 0 {
                            continue;
                        }
                        let overlaps_carrier = offset < carrier_end && self.tag_offset < field_end;
                        if self.layout_kind == enum_layout_kind::DIRECT && overlaps_carrier {
                            return verify_err!(
                                Location::Unknown,
                                "MirEnumType direct payload field cannot overlap its tag carrier"
                            );
                        }
                        if self.layout_kind == enum_layout_kind::NICHE
                            && variant != self.untagged_variant as usize
                            && overlaps_carrier
                        {
                            return verify_err!(
                                Location::Unknown,
                                "MirEnumType tagged niche-variant payload cannot overlap the carrier"
                            );
                        }
                        if occupied.iter().any(|&(start, previous_end)| {
                            offset < previous_end && start < field_end
                        }) {
                            return verify_err!(
                                Location::Unknown,
                                "MirEnumType fields of one inhabited variant cannot overlap"
                            );
                        }
                        occupied.push((offset, field_end));
                    }
                }
                flat_field = end;
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

#[cfg(test)]
mod enum_layout_tests {
    use super::*;
    use pliron::builtin::types::Signedness;

    fn direct(ctx: &Context) -> MirEnumType {
        MirEnumType {
            name: "Direct".into(),
            discriminant_ty: IntegerType::get(ctx, 8, Signedness::Unsigned).into(),
            variant_names: vec!["A".into(), "B".into()],
            variant_discriminants: vec![3, 7],
            variant_field_counts: vec![0, 0],
            all_field_types: vec![],
            all_field_offsets: vec![],
            all_field_sizes: vec![],
            tag_offset: 0,
            total_size: 1,
            abi_align: 1,
            layout_kind: enum_layout_kind::DIRECT,
            carrier_kind: enum_carrier_kind::INTEGER,
            carrier_width: 8,
            carrier_address_space: 0,
            niche_start_lo: 0,
            niche_start_hi: 0,
            niche_variant_start: 0,
            niche_variant_end: 0,
            untagged_variant: 0,
            single_variant: 0,
            variant_inhabited: vec![1, 1],
        }
    }

    fn niche(ctx: &Context) -> MirEnumType {
        let payload: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        MirEnumType {
            name: "Niche".into(),
            discriminant_ty: IntegerType::get(ctx, 8, Signedness::Unsigned).into(),
            variant_names: vec!["None".into(), "Some".into()],
            variant_discriminants: vec![0, 1],
            variant_field_counts: vec![0, 1],
            all_field_types: vec![payload],
            all_field_offsets: vec![0],
            all_field_sizes: vec![1],
            tag_offset: 0,
            total_size: 1,
            abi_align: 1,
            layout_kind: enum_layout_kind::NICHE,
            carrier_kind: enum_carrier_kind::INTEGER,
            carrier_width: 8,
            carrier_address_space: 0,
            niche_start_lo: 0,
            niche_start_hi: 0,
            niche_variant_start: 0,
            niche_variant_end: 0,
            untagged_variant: 1,
            single_variant: 0,
            variant_inhabited: vec![1, 1],
        }
    }

    #[test]
    fn zero_variant_empty_and_uninhabited_single_are_valid_layouts() {
        let ctx = Context::new();
        let mut empty = direct(&ctx);
        empty.name = "Never".into();
        empty.variant_names.clear();
        empty.variant_discriminants.clear();
        empty.variant_field_counts.clear();
        empty.variant_inhabited.clear();
        empty.layout_kind = enum_layout_kind::EMPTY;
        empty.carrier_kind = enum_carrier_kind::NONE;
        empty.carrier_width = 0;
        empty.total_size = 0;
        empty.abi_align = 1;
        assert!(empty.verify(&ctx).is_ok());

        let mut impossible = direct(&ctx);
        impossible.name = "Impossible".into();
        impossible.discriminant_ty = IntegerType::get(&ctx, 16, Signedness::Unsigned).into();
        impossible.variant_names = vec!["V".into()];
        impossible.variant_discriminants = vec![1_000];
        impossible.variant_field_counts = vec![0];
        impossible.variant_inhabited = vec![0];
        impossible.layout_kind = enum_layout_kind::SINGLE;
        impossible.carrier_kind = enum_carrier_kind::NONE;
        impossible.carrier_width = 0;
        impossible.total_size = 0;
        impossible.abi_align = 1;
        impossible.single_variant = 0;
        assert!(impossible.verify(&ctx).is_ok());
    }

    #[test]
    fn niche_allows_untagged_and_uninhabited_indices_inside_range() {
        let ctx = Context::new();
        let mut value = niche(&ctx);
        value.niche_variant_end = 1;
        value.untagged_variant = 1;
        assert!(value.verify(&ctx).is_ok(), "untagged may lie in range");

        value.variant_names.push("Data".into());
        value.variant_discriminants.push(2);
        value.variant_field_counts[1] = 0;
        value.variant_field_counts.push(1);
        value.variant_inhabited = vec![1, 0, 1];
        value.untagged_variant = 2;
        assert!(
            value.verify(&ctx).is_ok(),
            "an impossible source variant may occupy a niche-range index"
        );
    }

    #[test]
    fn malformed_physical_metadata_is_rejected() {
        let ctx = Context::new();

        let mut carrier_out_of_bounds = direct(&ctx);
        carrier_out_of_bounds.tag_offset = 1;
        assert!(carrier_out_of_bounds.verify(&ctx).is_err());

        let mut invalid_alignment = direct(&ctx);
        invalid_alignment.total_size = 3;
        invalid_alignment.abi_align = 3;
        assert!(invalid_alignment.verify(&ctx).is_err());

        let mut zst_without_alignment = direct(&ctx);
        zst_without_alignment.variant_names.truncate(1);
        zst_without_alignment.variant_discriminants.truncate(1);
        zst_without_alignment.variant_field_counts.truncate(1);
        zst_without_alignment.variant_inhabited.truncate(1);
        zst_without_alignment.layout_kind = enum_layout_kind::SINGLE;
        zst_without_alignment.carrier_kind = enum_carrier_kind::NONE;
        zst_without_alignment.carrier_width = 0;
        zst_without_alignment.total_size = 0;
        zst_without_alignment.abi_align = 0;
        assert!(zst_without_alignment.verify(&ctx).is_err());

        let mut wide_direct = direct(&ctx);
        wide_direct.discriminant_ty = IntegerType::get(&ctx, 128, Signedness::Unsigned).into();
        wide_direct.carrier_width = 128;
        wide_direct.total_size = 16;
        wide_direct.abi_align = 16;
        assert!(wide_direct.verify(&ctx).is_err());

        let mut partial_byte_direct = direct(&ctx);
        partial_byte_direct.discriminant_ty =
            IntegerType::get(&ctx, 7, Signedness::Unsigned).into();
        partial_byte_direct.carrier_width = 7;
        assert!(
            partial_byte_direct.verify(&ctx).is_err(),
            "a Direct physical carrier must not expose unspecified upper byte bits"
        );

        let mut partial_byte_niche = niche(&ctx);
        partial_byte_niche.carrier_width = 7;
        assert!(
            partial_byte_niche.verify(&ctx).is_err(),
            "a Niche physical carrier must not expose unspecified upper byte bits"
        );

        let mut bad_field_count = direct(&ctx);
        bad_field_count.variant_field_counts[0] = 1;
        assert!(bad_field_count.verify(&ctx).is_err());

        let mut unknown_with_encoding = direct(&ctx);
        unknown_with_encoding.layout_kind = enum_layout_kind::UNKNOWN;
        unknown_with_encoding.carrier_kind = enum_carrier_kind::NONE;
        unknown_with_encoding.carrier_width = 0;
        unknown_with_encoding.total_size = 0;
        unknown_with_encoding.abi_align = 0;
        unknown_with_encoding.niche_variant_end = 1;
        assert!(unknown_with_encoding.verify(&ctx).is_err());

        let u8_ty: TypeHandle = IntegerType::get(&ctx, 8, Signedness::Unsigned).into();
        let mut payload_over_tag = direct(&ctx);
        payload_over_tag.variant_field_counts = vec![1, 0];
        payload_over_tag.all_field_types = vec![u8_ty];
        payload_over_tag.all_field_offsets = vec![0];
        payload_over_tag.all_field_sizes = vec![1];
        assert!(
            payload_over_tag.verify(&ctx).is_err(),
            "an inhabited Direct payload cannot alias the tag"
        );

        let mut overlapping_fields = direct(&ctx);
        overlapping_fields.total_size = 2;
        overlapping_fields.variant_field_counts = vec![2, 0];
        overlapping_fields.all_field_types = vec![u8_ty, u8_ty];
        overlapping_fields.all_field_offsets = vec![1, 1];
        overlapping_fields.all_field_sizes = vec![1, 1];
        assert!(
            overlapping_fields.verify(&ctx).is_err(),
            "two non-ZST fields of one inhabited variant cannot overlap"
        );

        let mut niche_without_payload_carrier = niche(&ctx);
        niche_without_payload_carrier.total_size = 2;
        niche_without_payload_carrier.all_field_offsets[0] = 1;
        assert!(
            niche_without_payload_carrier.verify(&ctx).is_err(),
            "the untagged niche payload must physically contain the carrier"
        );

        let mut tagged_payload_over_carrier = niche(&ctx);
        tagged_payload_over_carrier.variant_field_counts = vec![1, 1];
        tagged_payload_over_carrier.all_field_types.insert(0, u8_ty);
        tagged_payload_over_carrier.all_field_offsets.insert(0, 0);
        tagged_payload_over_carrier.all_field_sizes.insert(0, 1);
        assert!(
            tagged_payload_over_carrier.verify(&ctx).is_err(),
            "a tagged niche variant cannot overwrite the carrier"
        );
    }

    #[test]
    fn malformed_flattened_variant_metadata_is_bounds_safe() {
        let ctx = Context::new();
        let mut value = direct(&ctx);
        value.variant_field_counts.clear();
        value.all_field_offsets.push(0);
        assert!(value.get_variant(0).is_none());
        assert!(value.variant_field_offsets(0).is_none());
    }

    #[test]
    fn niche_range_must_fit_carrier_without_repeated_encodings() {
        let ctx = Context::new();
        let mut value = niche(&ctx);
        value.carrier_width = 1;
        value.variant_names.push("Third".into());
        value.variant_discriminants.push(2);
        value.variant_field_counts.push(0);
        value.variant_inhabited.push(1);
        value.niche_variant_end = 2;
        value.untagged_variant = 0;
        assert!(
            value.verify(&ctx).is_err(),
            "three variant encodings cannot fit in an i1 carrier"
        );
    }

    #[test]
    fn pointer_carrier_rejects_target_dependent_shared_address_space() {
        let ctx = Context::new();
        let mut value = niche(&ctx);
        value.carrier_kind = enum_carrier_kind::POINTER;
        value.carrier_width = 32;
        value.total_size = 4;
        value.abi_align = 4;
        assert!(value.verify(&ctx).is_err());

        value.carrier_address_space = 3;
        value.carrier_width = 64;
        value.total_size = 8;
        value.abi_align = 8;
        assert!(value.verify(&ctx).is_err());
    }
}
