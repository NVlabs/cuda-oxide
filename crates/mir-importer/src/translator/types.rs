/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Type translation: Rust types → `dialect-mir` types.
//!
//! Converts Rust's type system representation to `dialect-mir` types.
//!
//! # Type Mapping
//!
//! | Rust Type           | `dialect-mir` Type                  |
//! |---------------------|-------------------------------------|
//! | `i32`, `u64`, etc.  | `IntegerType` (with signedness)     |
//! | `f32`, `f64`        | `FP32Type`, `FP64Type`              |
//! | `bool`              | `i1` (signless)                     |
//! | `char`              | `ui32`                              |
//! | `(A, B, C)`         | `MirTupleType`                      |
//! | `[T; N]`            | `ArrayType`                         |
//! | `*const T`, `*mut T`| `MirPtrType` (generic addrspace)    |
//! | `[T]`, `&[T]`       | `MirSliceType`                      |
//! | `struct S { .. }`   | `MirStructType`                     |
//! | `enum E { .. }`     | `MirEnumType`                       |
//! | Closures            | `MirStructType` (captures as fields)|
//!
//! # Special cuda_device Types
//!
//! | Type              | Translation                           |
//! |-------------------|---------------------------------------|
//! | `DisjointSlice<T>`| `MirDisjointSliceType`                |
//! | `ThreadIndex`     | `u64` (type safety at Rust level)     |
//! | `SharedArray<T,N>`| Empty tuple (ZST marker)              |
//! | `Barrier`         | `u64` (mbarrier state)                |
//! | `TmaDescriptor`   | `[u64; 16]` (128-byte opaque blob)    |

use crate::error::{TranslationErr, TranslationResult};
use pliron::context::{Context, Ptr};
use pliron::r#type::TypeObj;
use pliron::{input_err_noloc, input_error_noloc};
use rustc_public::CrateDef;

// Re-export types from dialect_mir for convenience
pub use dialect_mir::types::{
    EnumVariant, MirDisjointSliceType, MirEnumType, MirPtrType, MirSliceType, MirTupleType,
};
use rustc_public::mir::Mutability;

/// Returns the signed 32-bit integer type.
pub fn get_i32_type(
    ctx: &mut Context,
) -> pliron::r#type::TypePtr<pliron::builtin::types::IntegerType> {
    pliron::builtin::types::IntegerType::get(ctx, 32, pliron::builtin::types::Signedness::Signed)
}

/// Returns the boolean type (i1, signless).
pub fn get_bool_type(
    ctx: &mut Context,
) -> pliron::r#type::TypePtr<pliron::builtin::types::IntegerType> {
    pliron::builtin::types::IntegerType::get(ctx, 1, pliron::builtin::types::Signedness::Signless)
}

/// Returns the `usize` type (u64 on 64-bit targets).
pub fn get_usize_type(
    ctx: &mut Context,
) -> pliron::r#type::TypePtr<pliron::builtin::types::IntegerType> {
    pliron::builtin::types::IntegerType::get(ctx, 64, pliron::builtin::types::Signedness::Unsigned)
}

/// Returns the 32-bit floating point type.
pub fn get_f32_type(
    ctx: &mut Context,
) -> pliron::r#type::TypePtr<pliron::builtin::types::FP32Type> {
    pliron::builtin::types::FP32Type::get(ctx)
}

/// Checks if a `dialect-mir` type is zero-sized (ZST).
///
/// ZSTs are types that occupy no memory at runtime but carry semantic meaning
/// at the type level. Common ZSTs include:
/// - Empty tuples `()`
/// - Empty structs (structs with no fields, like `PhantomData<T>`)
/// - Unit structs (`struct Marker;`)
///
/// ZSTs are important for:
/// - Lifetime/variance tracking (`PhantomData<&'a T>`)
/// - Typestate patterns (`struct Allocated;`, `struct Deallocated;`)
/// - Type-level markers for layout/configuration
pub fn is_zst_type(ctx: &pliron::context::Context, ty: Ptr<TypeObj>) -> bool {
    let ty_ref = ty.deref(ctx);

    // Empty tuple - e.g., () or MirTupleType with no fields
    if let Some(tuple_ty) = ty_ref.downcast_ref::<MirTupleType>() {
        return tuple_ty.get_types().is_empty();
    }

    // Empty struct - structs with no fields (like PhantomData<T>)
    if let Some(struct_ty) = ty_ref.downcast_ref::<dialect_mir::types::MirStructType>() {
        return struct_ty.field_types().is_empty();
    }

    false
}

/// Checks if a Rust type is zero-sized (before translation).
///
/// This checks the Rust type directly before translation. It handles:
/// - ADTs with no fields (like `PhantomData<T>`, unit structs)
/// - Empty tuples
/// - Closures with no captures
///
/// This is useful for early detection before type translation.
pub fn is_rust_type_zst(rust_ty: &rustc_public::ty::Ty) -> bool {
    match rust_ty.kind() {
        // Empty tuple
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Tuple(subtypes)) => {
            subtypes.is_empty()
        }
        // ADT - check if it has no fields (for structs)
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Adt(adt_def, _substs)) => {
            let variants = adt_def.variants();
            // For structs (single variant), check if it has no fields
            if variants.len() == 1 {
                let variant = &variants[0];
                variant.fields().is_empty()
            } else {
                // Enums with multiple variants are not ZSTs (they have discriminants)
                false
            }
        }
        // Closures with no captures are ZST, closures with captures are not
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Closure(_, substs)) => {
            // Check substs[2] which is the tuple of upvar types
            if substs.0.len() >= 3
                && let rustc_public::ty::GenericArgKind::Type(upvar_tuple_ty) = &substs.0[2]
                && let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Tuple(
                    upvar_tys,
                )) = upvar_tuple_ty.kind()
            {
                // ZST if no captures
                return upvar_tys.is_empty();
            }
            // Default to ZST if we can't determine
            true
        }
        _ => false,
    }
}

/// Translates a raw-pointer or reference type to its `dialect-mir` equivalent.
///
/// Most pointers become generic-addrspace `MirPtrType`, but a few Rust-level
/// types are stand-ins for shared-memory objects in a CUDA kernel. We detect
/// those here and produce the correct `addrspace(3)` pointer so that the
/// alloca slot for such a local matches the pointer value produced by
/// shared-memory intrinsics (e.g. `MirSharedAllocOp`). See module docs.
fn translate_pointer_like(
    ctx: &mut Context,
    pointee: &rustc_public::ty::Ty,
    is_mutable: bool,
) -> TranslationResult<Ptr<TypeObj>> {
    match pointee.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Slice(elem_ty)) => {
            // `*const [T]` / `*mut [T]` have the same runtime layout as `&[T]`
            // (a 16-byte fat pointer = data ptr + length), so we use the same
            // `dialect-mir` type. Otherwise a bare `_x = _y` where `_y: &[T]`
            // and `_x: *const [T]` would be a semantic-mismatch store into
            // the alloca slot even though Rust considers these freely
            // interconvertible.
            let elem = translate_type(ctx, &elem_ty)?;
            Ok(MirSliceType::get(ctx, elem).into())
        }
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Adt(adt_def, substs))
            if adt_def.trimmed_name() == "SharedArray" =>
        {
            // `*mut SharedArray<T, N>` / `&mut SharedArray<T, N>` is, at
            // runtime, the base pointer of a shared-memory region holding
            // `[T; N]`. Match the intrinsic-emitted shared-alloc pointer so
            // the alloca slot and the rvalue agree on type.
            let elem = shared_array_element_type(ctx, &substs, "SharedArray")?;
            Ok(dialect_mir::types::MirPtrType::get_shared(ctx, elem, is_mutable).into())
        }
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Adt(adt_def, _substs))
            if adt_def.trimmed_name() == "Barrier" =>
        {
            // `*mut Barrier` / `&mut Barrier` is a pointer into shared memory
            // carrying mbarrier state (a 64-bit opaque value).
            let u64_ty = pliron::builtin::types::IntegerType::get(
                ctx,
                64,
                pliron::builtin::types::Signedness::Unsigned,
            )
            .into();
            Ok(dialect_mir::types::MirPtrType::get_shared(ctx, u64_ty, is_mutable).into())
        }
        _ => {
            let pointee_ty = translate_type(ctx, pointee)?;
            Ok(MirPtrType::get_generic(ctx, pointee_ty, is_mutable).into())
        }
    }
}

/// Extract the element type `T` from a `SharedArray<T, N, ALIGN>` /
/// `DisjointSlice<'_, T>` GenericArgs list. The first type-kind generic arg
/// is the element type.
fn shared_array_element_type(
    ctx: &mut Context,
    substs: &rustc_public::ty::GenericArgs,
    label: &'static str,
) -> TranslationResult<Ptr<TypeObj>> {
    for arg in substs.0.iter() {
        if let rustc_public::ty::GenericArgKind::Type(t) = arg {
            return translate_type(ctx, t);
        }
    }
    input_err_noloc!(TranslationErr::unsupported(format!(
        "{} has no element type parameter",
        label
    )))
}

/// Translates a Rust type to its `dialect-mir` equivalent.
///
/// See module documentation for the type mapping table.
pub fn translate_type(
    ctx: &mut Context,
    rust_ty: &rustc_public::ty::Ty,
) -> TranslationResult<Ptr<TypeObj>> {
    let ty_kind = rust_ty.kind();

    match ty_kind {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Int(int_ty)) => match int_ty {
            rustc_public::ty::IntTy::I32 => Ok(get_i32_type(ctx).into()),
            rustc_public::ty::IntTy::I64 => Ok(pliron::builtin::types::IntegerType::get(
                ctx,
                64,
                pliron::builtin::types::Signedness::Signed,
            )
            .into()),
            rustc_public::ty::IntTy::I8 => Ok(pliron::builtin::types::IntegerType::get(
                ctx,
                8,
                pliron::builtin::types::Signedness::Signed,
            )
            .into()),
            rustc_public::ty::IntTy::I16 => Ok(pliron::builtin::types::IntegerType::get(
                ctx,
                16,
                pliron::builtin::types::Signedness::Signed,
            )
            .into()),
            rustc_public::ty::IntTy::I128 => Ok(pliron::builtin::types::IntegerType::get(
                ctx,
                128,
                pliron::builtin::types::Signedness::Signed,
            )
            .into()),
            rustc_public::ty::IntTy::Isize => Ok(pliron::builtin::types::IntegerType::get(
                ctx,
                64,
                pliron::builtin::types::Signedness::Signed,
            )
            .into()),
        },
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Uint(uint_ty)) => {
            match uint_ty {
                rustc_public::ty::UintTy::U32 => Ok(pliron::builtin::types::IntegerType::get(
                    ctx,
                    32,
                    pliron::builtin::types::Signedness::Unsigned,
                )
                .into()),
                rustc_public::ty::UintTy::U64 => Ok(pliron::builtin::types::IntegerType::get(
                    ctx,
                    64,
                    pliron::builtin::types::Signedness::Unsigned,
                )
                .into()),
                rustc_public::ty::UintTy::U8 => Ok(pliron::builtin::types::IntegerType::get(
                    ctx,
                    8,
                    pliron::builtin::types::Signedness::Unsigned,
                )
                .into()),
                rustc_public::ty::UintTy::U16 => Ok(pliron::builtin::types::IntegerType::get(
                    ctx,
                    16,
                    pliron::builtin::types::Signedness::Unsigned,
                )
                .into()),
                rustc_public::ty::UintTy::U128 => Ok(pliron::builtin::types::IntegerType::get(
                    ctx,
                    128,
                    pliron::builtin::types::Signedness::Unsigned,
                )
                .into()),
                rustc_public::ty::UintTy::Usize => Ok(pliron::builtin::types::IntegerType::get(
                    ctx,
                    64,
                    pliron::builtin::types::Signedness::Unsigned,
                )
                .into()),
            }
        }
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Bool) => {
            Ok(pliron::builtin::types::IntegerType::get(
                ctx,
                1,
                pliron::builtin::types::Signedness::Signless,
            )
            .into())
        }
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Char) => {
            Ok(pliron::builtin::types::IntegerType::get(
                ctx,
                32,
                pliron::builtin::types::Signedness::Unsigned,
            )
            .into())
        }
        // The never type `!` represents computations that never complete (e.g., panic, infinite loop).
        // We translate it to an empty tuple (unit) since the code path is unreachable anyway.
        // This is used by things like Option::unwrap_failed() which returns `!`.
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Never) => {
            Ok(dialect_mir::types::MirTupleType::get(ctx, vec![]).into())
        }
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Float(float_ty)) => {
            match float_ty {
                rustc_public::ty::FloatTy::F32 => {
                    Ok(pliron::builtin::types::FP32Type::get(ctx).into())
                }
                rustc_public::ty::FloatTy::F64 => {
                    Ok(pliron::builtin::types::FP64Type::get(ctx).into())
                }
                rustc_public::ty::FloatTy::F16 => {
                    Ok(dialect_mir::types::MirFP16Type::get(ctx).into())
                }
                rustc_public::ty::FloatTy::F128 => {
                    input_err_noloc!(TranslationErr::unsupported(
                        "f128 (quad precision) not yet supported"
                    ))
                }
            }
        }
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Tuple(subtypes)) => {
            let mut translated_subtypes = Vec::new();
            for subtype in subtypes.iter() {
                translated_subtypes.push(translate_type(ctx, subtype)?);
            }
            Ok(MirTupleType::get(ctx, translated_subtypes).into())
        }
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Array(elem_ty, len_const)) => {
            // Translate the element type
            let elem = translate_type(ctx, &elem_ty)?;

            // Extract the array length from the const
            let len = match &len_const.kind() {
                rustc_public::ty::TyConstKind::Value(_, alloc) => {
                    // The allocation contains the length as bytes
                    // For usize, it's 8 bytes on 64-bit systems
                    let bytes = &alloc.bytes;
                    if bytes.len() >= 8 {
                        let mut arr = [0u8; 8];
                        for (i, b) in bytes.iter().take(8).enumerate() {
                            arr[i] = b.unwrap_or(0);
                        }
                        u64::from_le_bytes(arr)
                    } else {
                        return input_err_noloc!(TranslationErr::unsupported(
                            "Array length constant has unexpected size"
                        ));
                    }
                }
                _ => {
                    return input_err_noloc!(TranslationErr::unsupported(format!(
                        "Array length must be a value constant, got: {:?}",
                        len_const.kind()
                    )));
                }
            };

            Ok(dialect_mir::types::MirArrayType::get(ctx, elem, len).into())
        }
        // Bare slice [T] -> MirSliceType (fat pointer: data ptr + length)
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Slice(elem_ty)) => {
            let elem = translate_type(ctx, &elem_ty)?;
            Ok(MirSliceType::get(ctx, elem).into())
        }
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::RawPtr(ty, mutability)) => {
            let is_mutable = mutability == Mutability::Mut;
            translate_pointer_like(ctx, &ty, is_mutable)
        }
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Ref(
            _region,
            ty,
            mutability,
        )) => {
            let is_mutable = mutability == Mutability::Mut;
            translate_pointer_like(ctx, &ty, is_mutable)
        }
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Adt(adt_def, substs)) => {
            // Get the trimmed name (just the type name without path)
            let trimmed_name = adt_def.trimmed_name();

            // Check if this is DisjointSlice from cuda_device
            if trimmed_name == "DisjointSlice" {
                // Extract the element type from the generic parameter
                // DisjointSlice<'a, T> has T as the second parameter (first is lifetime)
                let generic_args = substs.0;

                // Find the first type argument (skip lifetimes)
                let elem_ty = generic_args
                    .iter()
                    .find_map(|arg| match arg {
                        rustc_public::ty::GenericArgKind::Type(ty) => Some(ty),
                        _ => None,
                    })
                    .ok_or_else(|| {
                        input_error_noloc!(TranslationErr::unsupported(
                            "DisjointSlice requires a type parameter"
                        ))
                    })?;

                let elem = translate_type(ctx, elem_ty)?;
                Ok(MirDisjointSliceType::get(ctx, elem).into())
            } else if trimmed_name == "ThreadIndex" {
                // ThreadIndex is a newtype around usize - translate to usize
                // The type safety is enforced at the Rust level, not the IR level
                Ok(get_usize_type(ctx).into())
            } else if trimmed_name == "SharedArray" {
                // SharedArray<T, N> is a zero-sized marker type.
                // The actual shared memory is allocated when we see the static declaration.
                // For the type itself, we use a unit/empty tuple type.
                //
                // When SharedArray appears as a static, the MIR importer handles it specially
                // to allocate shared memory and generate correct load/store operations.
                Ok(dialect_mir::types::MirTupleType::get(ctx, vec![]).into())
            } else if trimmed_name == "Barrier" {
                // Barrier is a 64-bit hardware barrier state stored in shared memory.
                // It's an opaque type that represents mbarrier state.
                // We represent it as i64 since that's its underlying storage.
                Ok(pliron::builtin::types::IntegerType::get(
                    ctx,
                    64,
                    pliron::builtin::types::Signedness::Unsigned,
                )
                .into())
            } else if trimmed_name == "TmaDescriptor" {
                // TmaDescriptor is a 128-byte opaque TMA descriptor created on host.
                // It's passed to kernels as a pointer. When we need the pointee type,
                // we represent it as an array of 16 i64s (128 bytes total).
                // This matches CUtensorMap which is { opaque: [u64; 16] }.
                let i64_ty = pliron::builtin::types::IntegerType::get(
                    ctx,
                    64,
                    pliron::builtin::types::Signedness::Unsigned,
                );
                Ok(dialect_llvm::types::ArrayType::get(ctx, i64_ty.into(), 16).into())
            } else {
                // Generic ADT handling for user-defined structs and enums
                let variants = adt_def.variants();

                if variants.len() == 1 {
                    // Structs have exactly one variant
                    let variant = &variants[0];
                    let fields = variant.fields();

                    // Extract field names and types (in declaration order)
                    let mut field_names = Vec::with_capacity(fields.len());
                    let mut field_types = Vec::with_capacity(fields.len());

                    for field in fields {
                        // Get field name
                        field_names.push(field.name.to_string());

                        // Get field type, instantiated with the ADT's generic args
                        let field_ty = field.ty_with_args(&substs);
                        let translated_ty = translate_type(ctx, &field_ty)?;
                        field_types.push(translated_ty);
                    }

                    // Query rustc for complete memory layout info
                    let (mem_to_decl, field_offsets, total_size) =
                        if let Ok(layout) = rust_ty.layout() {
                            let shape = layout.shape();

                            // Field order: mem_to_decl[mem_idx] = decl_idx
                            let mem_order = shape.fields.fields_by_offset_order();

                            // Field offsets in declaration order (bytes)
                            let offsets: Vec<u64> = match &shape.fields {
                                rustc_public::abi::FieldsShape::Arbitrary { offsets } => {
                                    offsets.iter().map(|s| s.bytes() as u64).collect()
                                }
                                _ => vec![],
                            };

                            // Total struct size (bytes)
                            let size: u64 = shape.size.bytes() as u64;

                            (mem_order, offsets, size)
                        } else {
                            (vec![], vec![], 0u64)
                        };

                    // Create the struct type with full layout info
                    Ok(dialect_mir::types::MirStructType::get_with_full_layout(
                        ctx,
                        trimmed_name.to_string(),
                        field_names,
                        field_types,
                        mem_to_decl,
                        field_offsets,
                        total_size,
                    )
                    .into())
                } else {
                    // Enums have multiple variants
                    // Determine discriminant type based on number of variants
                    let discriminant_bits = if variants.len() <= 256 {
                        8
                    } else if variants.len() <= 65536 {
                        16
                    } else {
                        32
                    };
                    let discriminant_ty = pliron::builtin::types::IntegerType::get(
                        ctx,
                        discriminant_bits,
                        pliron::builtin::types::Signedness::Unsigned,
                    );

                    // Translate each variant
                    let mut enum_variants = Vec::with_capacity(variants.len());
                    for variant in variants.iter() {
                        let fields = variant.fields();
                        let mut field_types = Vec::with_capacity(fields.len());
                        for field in fields {
                            let field_ty = field.ty_with_args(&substs);
                            let translated_ty = translate_type(ctx, &field_ty)?;
                            field_types.push(translated_ty);
                        }
                        enum_variants
                            .push(EnumVariant::new(variant.name().to_string(), field_types));
                    }

                    // Create the enum type
                    Ok(MirEnumType::get(
                        ctx,
                        trimmed_name.to_string(),
                        discriminant_ty.into(),
                        enum_variants,
                    )
                    .into())
                }
            }
        }
        // Handle Closure types
        // Closures are represented as structs with fields for each captured variable (upvar).
        // The substs for a closure contain:
        //   [0] Internal marker type (usually i8)
        //   [1] Function signature
        //   [2] Tuple of upvar types (the captured variables)
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Closure(
            closure_def,
            substs,
        )) => {
            let closure_name = format!("{:?}", closure_def.def_id());

            // Extract upvar types from substs[2] (the tuple of captured types)
            let mut field_names = Vec::new();
            let mut field_types = Vec::new();

            if substs.0.len() >= 3
                && let rustc_public::ty::GenericArgKind::Type(upvar_tuple_ty) = &substs.0[2]
                && let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Tuple(
                    upvar_tys,
                )) = upvar_tuple_ty.kind()
            {
                for (i, upvar_ty) in upvar_tys.iter().enumerate() {
                    field_names.push(format!("capture_{}", i));
                    field_types.push(translate_type(ctx, upvar_ty)?);
                }
            }

            let (mem_to_decl, field_offsets, total_size) = if let Ok(layout) = rust_ty.layout() {
                let shape = layout.shape();
                let mem_to_decl = shape.fields.fields_by_offset_order();
                let field_offsets = match &shape.fields {
                    rustc_public::abi::FieldsShape::Arbitrary { offsets } => {
                        offsets.iter().map(|offset| offset.bytes() as u64).collect()
                    }
                    _ => vec![],
                };
                (mem_to_decl, field_offsets, shape.size.bytes() as u64)
            } else {
                (vec![], vec![], 0)
            };

            Ok(dialect_mir::types::MirStructType::get_with_full_layout(
                ctx,
                closure_name,
                field_names,
                field_types,
                mem_to_decl,
                field_offsets,
                total_size,
            )
            .into())
        }
        // Handle associated types like <SharedArray<f32, 256> as Index<usize>>::Output
        // or <Closure as FnOnce<(Args,)>>::Output
        rustc_public::ty::TyKind::Alias(rustc_public::ty::AliasKind::Projection, alias_ty) => {
            let def_name = format!("{:?}", alias_ty.def_id);

            // For FnOnce::Output, FnMut::Output, Fn::Output on closures
            // The self type is the closure, and we need its return type
            if (def_name.contains("FnOnce")
                || def_name.contains("FnMut")
                || def_name.contains("Fn"))
                && def_name.contains("Output")
            {
                // The self type (closure) is the first generic argument
                let args = &alias_ty.args.0;
                if let Some(rustc_public::ty::GenericArgKind::Type(self_ty)) = args.first() {
                    // Get the function signature from the type (works for closures, fn ptrs, etc.)
                    // fn_sig() is a method on TyKind that handles Closure, FnDef, and FnPtr
                    if let Some(poly_fn_sig) = self_ty.kind().fn_sig() {
                        let sig = poly_fn_sig.skip_binder();
                        let output = sig.output();
                        return translate_type(ctx, &output);
                    }
                }
                // For non-closure Fn types (like function pointers), fall through to error
            }

            // For Index::Output or IndexMut::Output on SharedArray<T, N>, the output is T
            if def_name.contains("Index") && def_name.contains("Output") {
                // Extract the self type from args
                let args = &alias_ty.args.0;
                if let Some(rustc_public::ty::GenericArgKind::Type(self_ty)) = args.first() {
                    // Check if self type is SharedArray
                    if let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Adt(
                        adt_def,
                        substs,
                    )) = self_ty.kind()
                    {
                        use rustc_public::CrateDef;
                        if adt_def.trimmed_name() == "SharedArray" {
                            // Extract T from SharedArray<T, N>
                            let elem_ty = substs
                                .0
                                .iter()
                                .find_map(|arg| match arg {
                                    rustc_public::ty::GenericArgKind::Type(t) => Some(t),
                                    _ => None,
                                })
                                .ok_or_else(|| {
                                    input_error_noloc!(TranslationErr::unsupported(
                                        "SharedArray missing element type"
                                    ))
                                })?;
                            return translate_type(ctx, elem_ty);
                        }
                    }
                }
            }

            // For arithmetic trait outputs (Mul::Output, Add::Output, Sub::Output, etc.)
            // When monomorphized with primitive types, these resolve to the self type.
            // E.g., <f32 as Mul>::Output = f32, <i32 as Add>::Output = i32
            //
            // This handles: Mul, Add, Sub, Div, Rem, BitAnd, BitOr, BitXor, Shl, Shr, Neg, Not
            // The def_name format can be either:
            //   - "std::ops::Mul::Output" (from std re-export)
            //   - "core::ops::Mul::Output" (from core directly)
            let is_arith_output = (def_name.contains("std::ops::")
                || def_name.contains("core::ops::"))
                && (def_name.contains("Mul::Output")
                    || def_name.contains("Add::Output")
                    || def_name.contains("Sub::Output")
                    || def_name.contains("Div::Output")
                    || def_name.contains("Rem::Output")
                    || def_name.contains("BitAnd::Output")
                    || def_name.contains("BitOr::Output")
                    || def_name.contains("BitXor::Output")
                    || def_name.contains("Shl::Output")
                    || def_name.contains("Shr::Output")
                    || def_name.contains("Neg::Output")
                    || def_name.contains("Not::Output"));

            if is_arith_output {
                // The self type is the first generic argument
                let args = &alias_ty.args.0;
                if let Some(rustc_public::ty::GenericArgKind::Type(self_ty)) = args.first() {
                    // For primitive types implementing arithmetic traits, Output = Self
                    if let rustc_public::ty::TyKind::RigidTy(
                        rustc_public::ty::RigidTy::Int(_)
                        | rustc_public::ty::RigidTy::Uint(_)
                        | rustc_public::ty::RigidTy::Float(_)
                        | rustc_public::ty::RigidTy::Bool
                        | rustc_public::ty::RigidTy::Char,
                    ) = self_ty.kind()
                    {
                        return translate_type(ctx, self_ty);
                    }
                }
            }

            input_err_noloc!(TranslationErr::unsupported(format!(
                "Alias type not yet supported: {:?}",
                alias_ty.def_id
            )))
        }
        // Pattern types (e.g. the storage of `NonZeroUsize` is `Pat<usize, 1..=usize::MAX>`).
        //
        // Layout assumption: a `Pat<T, P>` has the same size and alignment as
        // its base `T`; the pattern only restricts the set of valid values
        // (used by rustc for niche optimisation in enclosing enums). For
        // memory layout, lowering it as the base type is sound, and the
        // niche metadata that rustc relies on is consumed when we query
        // `ty.layout()` on the enclosing ADT, not here.
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Pat(base_ty, _pat)) => {
            translate_type(ctx, &base_ty)
        }
        _ => input_err_noloc!(TranslationErr::unsupported(format!(
            "Type translation not yet implemented for: {:?}",
            ty_kind
        ))),
    }
}
