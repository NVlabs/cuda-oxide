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
//! | `union U { .. }`    | `MirUnionType`                      |
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
use pliron::context::Context;
use pliron::r#type::TypeHandle;
use pliron::{input_err_noloc, input_error_noloc};
use rustc_public::CrateDef;
use rustc_public_bridge::IndexedVal;

// Re-export types from dialect_mir for convenience
pub use dialect_mir::types::{
    EnumVariant, MirDisjointSliceType, MirEnumType, MirPtrType, MirSliceType, MirTupleType,
    MirUnionType,
};
use rustc_public::mir::Mutability;

/// Returns the signed 32-bit integer type.
pub fn get_i32_type(
    ctx: &mut Context,
) -> pliron::r#type::TypedHandle<pliron::builtin::types::IntegerType> {
    pliron::builtin::types::IntegerType::get(ctx, 32, pliron::builtin::types::Signedness::Signed)
}

/// Returns the boolean type (i1, signless).
pub fn get_bool_type(
    ctx: &mut Context,
) -> pliron::r#type::TypedHandle<pliron::builtin::types::IntegerType> {
    pliron::builtin::types::IntegerType::get(ctx, 1, pliron::builtin::types::Signedness::Signless)
}

/// Returns the `usize` type (u64 on 64-bit targets).
pub fn get_usize_type(
    ctx: &mut Context,
) -> pliron::r#type::TypedHandle<pliron::builtin::types::IntegerType> {
    pliron::builtin::types::IntegerType::get(ctx, 64, pliron::builtin::types::Signedness::Unsigned)
}

/// Returns the tupled-upvars types for a `RigidTy::Closure`.
///
/// rustc suffix-encodes closure substitutions as
/// `[parent_args..., closure_kind, closure_sig, tupled_upvars]`, so the upvars
/// tuple is the last generic arg, not a fixed index.
fn closure_upvar_tys(substs: &rustc_public::ty::GenericArgs) -> Option<Vec<rustc_public::ty::Ty>> {
    let rustc_public::ty::GenericArgKind::Type(upvar_tuple_ty) = substs.0.last()? else {
        return None;
    };
    let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Tuple(upvar_tys)) =
        upvar_tuple_ty.kind()
    else {
        return None;
    };
    Some(upvar_tys)
}

/// Returns the `isize` type (i64 on 64-bit targets).
pub fn get_isize_type(
    ctx: &mut Context,
) -> pliron::r#type::TypedHandle<pliron::builtin::types::IntegerType> {
    pliron::builtin::types::IntegerType::get(ctx, 64, pliron::builtin::types::Signedness::Signed)
}

/// Returns the 32-bit floating point type.
pub fn get_f32_type(
    ctx: &mut Context,
) -> pliron::r#type::TypedHandle<pliron::builtin::types::FP32Type> {
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
pub fn is_zst_type(ctx: &pliron::context::Context, ty: TypeHandle) -> bool {
    let ty_ref = ty.deref(ctx);

    // Empty tuple - e.g., () or MirTupleType with no fields
    if let Some(tuple_ty) = ty_ref.downcast_ref::<MirTupleType>() {
        return tuple_ty.get_types().is_empty();
    }

    // Empty struct - structs with no fields (like PhantomData<T>)
    if let Some(struct_ty) = ty_ref.downcast_ref::<dialect_mir::types::MirStructType>() {
        return struct_ty.field_types().is_empty();
    }

    if let Some(union_ty) = ty_ref.downcast_ref::<MirUnionType>() {
        return union_ty.total_size() == 0;
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
            if matches!(adt_def.kind(), rustc_public::ty::AdtKind::Union) {
                // A union can have declared fields and still own no bytes when
                // every field is zero-sized. Source-level field count cannot
                // answer this; use rustc's target layout.
                return rust_ty
                    .layout()
                    .is_ok_and(|layout| layout.shape().size.bytes() == 0);
            }
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
        // Closures with no captures are ZST, closures with captures are not.
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Closure(_, substs)) => {
            if let Some(upvar_tys) = closure_upvar_tys(&substs) {
                // ZST if no captures
                return upvar_tys.is_empty();
            }
            // Default to ZST if we can't determine
            true
        }
        _ => false,
    }
}

/// If `ty` is a struct made unsized by a trailing slice field, return that
/// slice's ELEMENT type. Returns `None` for every other type.
///
/// Rust allows the LAST field of a struct to be an unsized type such as
/// `[T]`; the struct itself then becomes unsized and a reference to it is
/// a fat pointer: (pointer to the struct's first byte, number of trailing
/// elements). The motivating case is `core::array::iter::iter_inner::
/// PolymorphicIter<[MaybeUninit<T>]>`, the type that backs
/// `core::array::IntoIter` and therefore every `for x in arr` loop over a
/// by-value array (issue #138).
///
/// The check recurses through nested structs because the unsized tail may
/// itself sit at the end of an inner struct (`struct A { b: B }` with
/// `struct B { t: [u32] }` makes `A` slice-tailed too).
pub(super) fn slice_tail_element_ty(ty: &rustc_public::ty::Ty) -> Option<rustc_public::ty::Ty> {
    match ty.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Adt(adt_def, substs)) => {
            let variants = adt_def.variants();
            // Only structs (exactly one variant) can have an unsized tail.
            if variants.len() != 1 {
                return None;
            }
            let fields = variants[0].fields();
            let last_field = fields.last()?;
            let last_ty = last_field.ty_with_args(&substs);
            match last_ty.kind() {
                rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Slice(elem)) => {
                    Some(elem)
                }
                rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Adt(..)) => {
                    slice_tail_element_ty(&last_ty)
                }
                _ => None,
            }
        }
        _ => None,
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
) -> TranslationResult<TypeHandle> {
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
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Str) => {
            // `&str` / `*const str` is a fat pointer (data ptr + length),
            // exactly like `&[u8]`. Without this arm it would fall through
            // to the generic case below and become a THIN pointer to the
            // slice struct: 8 bytes where Rust has 16, silently corrupting
            // any local that holds one.
            let u8_ty = pliron::builtin::types::IntegerType::get(
                ctx,
                8,
                pliron::builtin::types::Signedness::Unsigned,
            )
            .into();
            Ok(MirSliceType::get(ctx, u8_ty).into())
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
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Adt(..))
            if slice_tail_element_ty(pointee).is_some() =>
        {
            // A reference to a struct whose last field is a slice (an
            // "unsized tail"), e.g. the `PolymorphicIter<[MaybeUninit<T>]>`
            // backing every `for x in arr` loop. Such a reference is a FAT
            // pointer at runtime: (pointer to the struct, number of tail
            // elements). Modelling it as a thin pointer would silently drop
            // the element count, which feeds slice reborrows of the tail.
            //
            // We reuse `MirSliceType` as the fat-pair carrier with the
            // translated STRUCT as its element type, so the existing
            // fat-pointer machinery (function-boundary flattening into
            // (ptr, len), `PtrMetadata` extraction, fat-value copies) all
            // applies unchanged. Field access through the fat pointer
            // extracts the data pointer (the struct's address) first; see
            // the place-address walker in `rvalue.rs`.
            let struct_model = translate_type(ctx, pointee)?;
            Ok(MirSliceType::get(ctx, struct_model).into())
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
) -> TranslationResult<TypeHandle> {
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

/// Translates the type of a call's destination place to its `dialect-mir`
/// equivalent.
///
/// Call results are typed from the destination in the caller's monomorphized
/// MIR, not from the callee's declared signature. A trait method's declared
/// signature types its result against the trait, so it can contain an
/// associated-type projection that is not yet resolved to a concrete type,
/// for example `<&Foo as Mul>::Output` (issue #133). The destination local
/// already carries the concrete type rustc resolved during monomorphization,
/// and it is by construction the slot the call result is stored into, so the
/// call result type and the destination slot always agree.
pub fn translate_destination_type(
    ctx: &mut Context,
    body: &rustc_public::mir::Body,
    destination: &rustc_public::mir::Place,
    loc: &pliron::location::Location,
) -> TranslationResult<TypeHandle> {
    let dest_rust_ty = match destination.ty(body.locals()) {
        Ok(t) => t,
        Err(e) => {
            return pliron::input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "failed to resolve destination type for call result: {e:?}"
                ))
            );
        }
    };
    translate_type(ctx, &dest_rust_ty)
}

/// Translates a Rust type to its `dialect-mir` equivalent.
///
/// See module documentation for the type mapping table.
pub fn translate_type(
    ctx: &mut Context,
    rust_ty: &rustc_public::ty::Ty,
) -> TranslationResult<TypeHandle> {
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
                Ok(llvm_export::types::ArrayType::get(ctx, i64_ty.into(), 16).into())
            } else {
                // Generic ADT handling for user-defined structs and enums
                let variants = adt_def.variants();

                if matches!(adt_def.kind(), rustc_public::ty::AdtKind::Union) {
                    let variant = variants.first().ok_or_else(|| {
                        input_error_noloc!(TranslationErr::unsupported(format!(
                            "Union {} has no field variant",
                            trimmed_name
                        )))
                    })?;
                    let fields = variant.fields();
                    let mut field_names = Vec::with_capacity(fields.len());
                    let mut field_types = Vec::with_capacity(fields.len());
                    for field in fields {
                        field_names.push(field.name.to_string());
                        field_types.push(translate_type(ctx, &field.ty_with_args(&substs))?);
                    }

                    let layout = rust_ty.layout().map_err(|e| {
                        input_error_noloc!(TranslationErr::unsupported(format!(
                            "Failed to query union layout for {}: {:?}",
                            trimmed_name, e
                        )))
                    })?;
                    let shape = layout.shape();
                    if let rustc_public::abi::FieldsShape::Arbitrary { offsets } = &shape.fields
                        && offsets.iter().any(|offset| offset.bytes() != 0)
                    {
                        return input_err_noloc!(TranslationErr::unsupported(format!(
                            "Union {} has a non-zero field offset in rustc's layout",
                            trimmed_name
                        )));
                    }

                    Ok(MirUnionType::get(
                        ctx,
                        trimmed_name.to_string(),
                        field_names,
                        field_types,
                        shape.size.bytes() as u64,
                        shape.abi_align,
                    )
                    .into())
                } else if variants.len() == 1 {
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
                        let translated_ty = if let rustc_public::ty::TyKind::RigidTy(
                            rustc_public::ty::RigidTy::Slice(elem_ty),
                        ) = field_ty.kind()
                        {
                            // A slice-typed field can only be the struct's
                            // unsized tail (Rust allows `[T]` only as the
                            // last field). The tail's elements live INLINE
                            // after the sized prefix, so we record the
                            // ELEMENT type here: the field's address (from
                            // rustc's layout offset) is then a pointer to
                            // the first element, which is exactly what a
                            // reborrow of the tail needs. Recording the
                            // generic `[T]` fat-pair type instead would make
                            // field addressing produce a pointer to a
                            // (ptr, len) pair that does not exist in memory.
                            translate_type(ctx, &elem_ty)?
                        } else {
                            translate_type(ctx, &field_ty)?
                        };
                        field_types.push(translated_ty);
                    }

                    // Query rustc for complete memory layout info
                    let (mem_to_decl, field_offsets, total_size, abi_align) =
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
                            (mem_order, offsets, size, shape.abi_align)
                        } else {
                            (vec![], vec![], 0u64, 0u64)
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
                        abi_align,
                    )
                    .into())
                } else {
                    // Enums have multiple variants.
                    //
                    // The discriminant ("tag") type comes from rustc's layout,
                    // never from a guess: `#[repr(uN/iN)]` (width AND
                    // signedness), `#[repr(usize/isize)]`, `#[repr(C)]`,
                    // sparse discriminants (`enum E { A = 0, B = 1_000_000 }`
                    // gets a u32 tag) and negative discriminants
                    // (`enum E { N = -1, Z = 0 }` gets a SIGNED i8 tag, so a
                    // later `e as i32` sign-extends instead of zero-extending)
                    // all fall out of the single `TagEncoding::Direct` arm
                    // below.
                    let enum_name = trimmed_name.to_string();
                    let layout_shape = rust_ty
                        .layout()
                        .map_err(|e| {
                            input_error_noloc!(TranslationErr::unsupported(format!(
                                "Failed to query enum layout for {}: {:?}",
                                enum_name, e
                            )))
                        })?
                        .shape();

                    // Fallback tag used where the un-niched `MirEnumType`
                    // model is deliberately self-consistent rather than
                    // memory-faithful (the niched and single-variant arms
                    // below): smallest unsigned width that fits the variant
                    // count.
                    let variant_count_bits: u32 = if variants.len() <= 256 {
                        8
                    } else if variants.len() <= 65536 {
                        16
                    } else {
                        32
                    };

                    // (discriminant type, tag byte offset, total size in
                    // bytes, ABI alignment). Size/align are 0 ("unknown")
                    // except for Direct-tag enums, where mir-lower uses
                    // them to build the memory-faithful representation.
                    let (discriminant_ty, tag_offset, total_size, abi_align): (
                        TypeHandle,
                        u64,
                        u64,
                        u64,
                    ) = match &layout_shape.variants {
                        rustc_public::abi::VariantsShape::Multiple {
                            tag,
                            tag_encoding: rustc_public::abi::TagEncoding::Direct,
                            tag_field,
                            ..
                        } => {
                            let primitive = match tag {
                                rustc_public::abi::Scalar::Initialized { value, .. }
                                | rustc_public::abi::Scalar::Union { value } => *value,
                            };
                            let rustc_public::abi::Primitive::Int { length, signed } = primitive
                            else {
                                return input_err_noloc!(TranslationErr::unsupported(format!(
                                    "Direct enum tag for {} is not an integer: {:?}",
                                    enum_name, primitive
                                )));
                            };
                            let tag_ty = pliron::builtin::types::IntegerType::get(
                                ctx,
                                length.bits() as u32,
                                if signed {
                                    pliron::builtin::types::Signedness::Signed
                                } else {
                                    pliron::builtin::types::Signedness::Unsigned
                                },
                            );
                            // The tag is usually at byte 0, but rustc may
                            // place it after payload bytes; read its real
                            // offset via the same lookup constant decoding
                            // uses (shared `translator::layout` helper).
                            let tag_offset = crate::translator::layout::enum_tag_offset(
                                &layout_shape.fields,
                                *tag_field,
                                pliron::location::Location::Unknown,
                            )? as u64;
                            (
                                tag_ty.into(),
                                tag_offset,
                                layout_shape.size.bytes() as u64,
                                layout_shape.abi_align,
                            )
                        }
                        rustc_public::abi::VariantsShape::Multiple {
                            tag_encoding: rustc_public::abi::TagEncoding::Niche { .. },
                            ..
                        } => {
                            // Niche-encoded enums (e.g. Option<&T>) store
                            // no tag in rustc's layout; the variant is
                            // COMPUTED from the payload (null means None).
                            // Our discriminant/construct ops only know the
                            // load-a-tag / store-a-tag shape, so until
                            // that decode logic is ported, the device
                            // models these enums with an explicit
                            // variant-count tag of its own, and mir-lower
                            // rebuilds the aggregate from
                            // `NicheEncodingAttr`
                            // (`emit_scalar_to_niched_enum`). Reading
                            // rustc's niche tag (the payload scalar
                            // itself) here would break that contract.
                            // Size/align stay 0 ("layout not recorded"):
                            // this model is device-private and must never
                            // meet host bytes; the kernel-boundary check
                            // in mir-lower enforces exactly that.
                            let tag_ty = pliron::builtin::types::IntegerType::get(
                                ctx,
                                variant_count_bits,
                                pliron::builtin::types::Signedness::Unsigned,
                            );
                            (tag_ty.into(), 0u64, 0u64, 0u64)
                        }
                        rustc_public::abi::VariantsShape::Single { .. } => {
                            // NOT an error: rustc reports `Single` for
                            // multi-syntactic-variant enums where all but
                            // one variant is uninhabited (e.g.
                            // `Result<T, Infallible>` from `TryFrom`).
                            // There is no tag in memory; keep the
                            // variant-count tag so in-kernel construct +
                            // discriminant reads stay self-consistent.
                            let tag_ty = pliron::builtin::types::IntegerType::get(
                                ctx,
                                variant_count_bits,
                                pliron::builtin::types::Signedness::Unsigned,
                            );
                            (tag_ty.into(), 0u64, 0u64, 0u64)
                        }
                        rustc_public::abi::VariantsShape::Empty => {
                            // Fully uninhabited enums (e.g. `Infallible`)
                            // appear in statically-dead paths of core
                            // library code (`Result<T, Infallible>` match
                            // arms, iterator adapters). The TYPE must
                            // translate so those dead arms lower; rustc
                            // gives it a zero-sized layout with no tag.
                            // Keep the variant-count tag for shape
                            // consistency. Materializing a VALUE of an
                            // uninhabited enum still fails loudly
                            // ("Cannot materialize a constant for an
                            // uninhabited enum", rvalue.rs).
                            let tag_ty = pliron::builtin::types::IntegerType::get(
                                ctx,
                                variant_count_bits,
                                pliron::builtin::types::Signedness::Unsigned,
                            );
                            (tag_ty.into(), 0u64, 0u64, 0u64)
                        }
                    };

                    // Declared discriminant VALUES (not variant indices),
                    // truncated by rustc to the tag width's unsigned bit
                    // pattern. They must fit in the u64 the dialect stores;
                    // 128-bit discriminants would silently alias otherwise.
                    let mut variant_discriminants = Vec::with_capacity(variants.len());
                    for idx in 0..variants.len() {
                        let variant_idx = rustc_public::ty::VariantIdx::to_val(idx);
                        let discr = adt_def.discriminant_for_variant(variant_idx);
                        let discr_val = u64::try_from(discr.val).map_err(|_| {
                            input_error_noloc!(TranslationErr::unsupported(format!(
                                "Enum discriminant {} for {} variant {} does not fit in 64 bits",
                                discr.val, enum_name, idx
                            )))
                        })?;
                        variant_discriminants.push(discr_val);
                    }

                    // Translate each variant. When the layout is recorded
                    // (total_size > 0), also note where each field lives,
                    // using the same shared helper constant decoding uses.
                    // Positions repeat across variants: variants share
                    // bytes, since only one is alive at a time.
                    let mut enum_variants = Vec::with_capacity(variants.len());
                    for (variant_idx, variant) in variants.iter().enumerate() {
                        let fields = variant.fields();
                        let mut field_types = Vec::with_capacity(fields.len());
                        for field in fields {
                            let field_ty = field.ty_with_args(&substs);
                            let translated_ty = translate_type(ctx, &field_ty)?;
                            field_types.push(translated_ty);
                        }
                        if total_size > 0 {
                            let field_offsets: Vec<u64> =
                                crate::translator::layout::enum_variant_field_offsets(
                                    &layout_shape,
                                    variant_idx,
                                    pliron::location::Location::Unknown,
                                )?
                                .into_iter()
                                .map(|o| o as u64)
                                .collect();
                            enum_variants.push(EnumVariant::new_with_offsets(
                                variant.name().to_string(),
                                field_types,
                                field_offsets,
                            ));
                        } else {
                            enum_variants
                                .push(EnumVariant::new(variant.name().to_string(), field_types));
                        }
                    }

                    // Create the enum type
                    Ok(MirEnumType::get_with_layout(
                        ctx,
                        enum_name,
                        discriminant_ty,
                        variant_discriminants,
                        enum_variants,
                        tag_offset,
                        total_size,
                        abi_align,
                    )
                    .into())
                }
            }
        }
        // Handle Closure types
        // Closures are represented as structs with fields for each captured variable (upvar).
        // The substs for a closure contain:
        //   [parent_args..] Captured parent generics, when the closure appears in a generic item
        //   [n - 3] Closure kind
        //   [n - 2] Function signature
        //   [n - 1] Tuple of upvar types (the captured variables)
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Closure(
            closure_def,
            substs,
        )) => {
            let closure_name = format!("{:?}", closure_def.def_id());

            // Extract upvar types from the tupled-upvars generic arg.
            let mut field_names = Vec::new();
            let mut field_types = Vec::new();

            if let Some(upvar_tys) = closure_upvar_tys(&substs) {
                for (i, upvar_ty) in upvar_tys.iter().enumerate() {
                    field_names.push(format!("capture_{}", i));
                    field_types.push(translate_type(ctx, upvar_ty)?);
                }
            }

            let (mem_to_decl, field_offsets, total_size, abi_align) =
                if let Ok(layout) = rust_ty.layout() {
                    let shape = layout.shape();
                    let mem_to_decl = shape.fields.fields_by_offset_order();
                    let field_offsets = match &shape.fields {
                        rustc_public::abi::FieldsShape::Arbitrary { offsets } => {
                            offsets.iter().map(|offset| offset.bytes() as u64).collect()
                        }
                        _ => vec![],
                    };
                    (
                        mem_to_decl,
                        field_offsets,
                        shape.size.bytes() as u64,
                        shape.abi_align,
                    )
                } else {
                    (vec![], vec![], 0, 0)
                };

            Ok(dialect_mir::types::MirStructType::get_with_full_layout(
                ctx,
                closure_name,
                field_names,
                field_types,
                mem_to_decl,
                field_offsets,
                total_size,
                abi_align,
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

            // No guessing for other associated-type projections. An earlier
            // version of this code assumed that arithmetic-trait outputs
            // (`Mul::Output`, `Add::Output`, ...) always equal the self type.
            // That assumption is wrong in general: `impl Mul for &Foo` with
            // `type Output = Foo` (issue #133) has Output != Self, and so
            // does any `impl Mul for Meters { type Output = SquareMeters }`.
            // Guessing the self type there silently mistypes the value (a
            // miscompile), so we fail loudly instead.
            //
            // Projections should not normally reach this point at all: call
            // results are typed from the caller's destination place (see
            // `translate_destination_type`), which rustc has already
            // normalized to a concrete type. Hitting this error means some
            // code path handed the type translator an unnormalized type
            // taken from a declared trait signature. Fix that path to use
            // the normalized type (destination place, or the signature of
            // the resolved `Instance`) rather than teaching this function
            // to guess what the projection resolves to.
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
        // `str` is an unsized byte sequence (appears in dead panic-message
        // branches). Translate as a `[u8]`-style slice.
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Str) => {
            let u8_ty = pliron::builtin::types::IntegerType::get(
                ctx,
                8,
                pliron::builtin::types::Signedness::Unsigned,
            )
            .into();
            Ok(MirSliceType::get(ctx, u8_ty).into())
        }
        // Function pointer type (e.g. `fmt` fn ptrs in dead panic-formatting
        // branches): a thin opaque pointer.
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnPtr(_)) => {
            let target = dialect_mir::types::MirStructType::get_with_full_layout(
                ctx,
                "FnPtrTarget".to_string(),
                vec![],
                vec![],
                vec![],
                vec![],
                0,
                0,
            )
            .into();
            Ok(dialect_mir::types::MirPtrType::get_generic(ctx, target, false).into())
        }
        // Zero-sized function-item type. Appears only type-level (e.g. dead
        // panic/formatting branches pulled in by `assert!` inside core fns like
        // `f32::clamp`); never materialised as a value.
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnDef(fn_def, _)) => {
            let name = format!("FnDef_{:?}", fn_def.def_id());
            Ok(dialect_mir::types::MirStructType::get_with_full_layout(
                ctx,
                name,
                vec![],
                vec![],
                vec![],
                vec![],
                0,
                0,
            )
            .into())
        }
        _ => input_err_noloc!(TranslationErr::unsupported(format!(
            "Type translation not yet implemented for: {:?}",
            ty_kind
        ))),
    }
}
