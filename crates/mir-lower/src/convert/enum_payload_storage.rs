/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Target-stable physical storage for enum payload values.
//!
//! Rust treats CUDA pointers as one logical pointer-sized value, while modern
//! NVVM uses a 32-bit physical representation for address-space-3 pointers.
//! Enum storage therefore cannot retain a semantic shared pointer directly.
//! Direct and struct/tuple-nested shared pointers are represented as CUDA
//! generic pointers in the enum and converted at construction/extraction
//! boundaries. Rust `bool` leaves are represented by canonical `i8` bytes at
//! the same boundary.
//!
//! Arrays containing shared pointers are rebuilt recursively only when the
//! payload's total array-expanded shared-pointer leaves stay within an
//! explicit code-shape bound. The bound is enforced once at the payload root,
//! so one array of 17 leaves and a struct of two 9-leaf arrays are rejected by
//! the same contract. Pointer vectors remain fail-closed because they require
//! separate ABI and address-space-cast semantics.

use llvm_export::op_interfaces::{CastOpInterface, CastOpWithNNegInterface};
use llvm_export::ops as llvm;
use llvm_export::types as llvm_types;
use llvm_export::types::PointerTypeExt;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::Context;
use pliron::irbuild::dialect_conversion::DialectConversionRewriter;
use pliron::irbuild::inserter::Inserter;
use pliron::op::Op;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;

/// Maximum number of array-expanded shared-pointer leaves one payload rewrite
/// may produce, totalled across the whole payload type.
///
/// Construction and extraction rebuild arrays in SSA, so every shared-pointer
/// leaf produces a pair of aggregate operations around one address-space cast.
/// Struct nesting stays unbounded because its leaf count is proportional to
/// the source text, while `[&shared; N]` expands from three tokens into `N`
/// rebuild sequences. The same constant bounds the pointer-overlap walk in
/// `build_enum_slot_map`, keeping one contract for every payload shape.
pub(crate) const MAX_ENUM_PAYLOAD_ARRAY_REWRITE_LEAVES: u64 = 16;

#[derive(Clone)]
struct StorageRewrite {
    ty: TypeHandle,
    /// Shared-pointer leaves anywhere in the rewritten type.
    shared_pointer_leaves: u64,
    /// Shared-pointer leaves reached through at least one array. Only these
    /// count against [`MAX_ENUM_PAYLOAD_ARRAY_REWRITE_LEAVES`], checked once
    /// at the payload root by [`enum_payload_storage_type`].
    array_shared_pointer_leaves: u64,
}

enum TypeShape {
    Bool,
    Pointer(u32),
    Array(TypeHandle, u64),
    Vector(TypeHandle),
    Struct(Vec<TypeHandle>),
    Identity,
}

/// Return the physical LLVM type used to store one semantic enum payload.
///
/// The transformation is recursive through LLVM structs, which cover MIR
/// structs and tuples after ordinary type conversion:
///
/// - `ptr addrspace(3)` becomes a CUDA generic pointer;
/// - `i1` becomes the canonical one-byte `i8` memory representation;
/// - structs are rebuilt field by field;
/// - arrays are rebuilt recursively while the payload's total array-expanded
///   shared-pointer leaves stay within
///   [`MAX_ENUM_PAYLOAD_ARRAY_REWRITE_LEAVES`];
/// - vectors containing shared pointers are rejected.
pub(crate) fn enum_payload_storage_type(
    ctx: &mut Context,
    semantic_ty: TypeHandle,
) -> std::result::Result<TypeHandle, anyhow::Error> {
    let rewrite = rewrite_storage_type(ctx, semantic_ty)?;
    // One bound for every payload shape, enforced at the root: a single
    // oversized array and a struct of several smaller arrays are both counted
    // by their total array-expanded shared-pointer leaves.
    if rewrite.array_shared_pointer_leaves > MAX_ENUM_PAYLOAD_ARRAY_REWRITE_LEAVES {
        return Err(anyhow::anyhow!(
            "enum payload storage: arrays containing shared-memory pointers are not supported above the bounded rewrite limit; rewrite requires {} pointer conversions, supported bound is {MAX_ENUM_PAYLOAD_ARRAY_REWRITE_LEAVES}",
            rewrite.array_shared_pointer_leaves
        ));
    }
    Ok(rewrite.ty)
}

fn rewrite_storage_type(
    ctx: &mut Context,
    semantic_ty: TypeHandle,
) -> std::result::Result<StorageRewrite, anyhow::Error> {
    let shape = {
        let ty_ref = semantic_ty.deref(ctx);
        if ty_ref
            .downcast_ref::<IntegerType>()
            .is_some_and(|integer| integer.width() == 1)
        {
            TypeShape::Bool
        } else if let Some(pointer) = ty_ref.downcast_ref::<llvm_types::PointerType>() {
            TypeShape::Pointer(pointer.address_space())
        } else if let Some(array) = ty_ref.downcast_ref::<llvm_types::ArrayType>() {
            TypeShape::Array(array.elem_type(), array.size())
        } else if let Some(vector) = ty_ref.downcast_ref::<llvm_types::VectorType>() {
            TypeShape::Vector(vector.elem_type())
        } else if let Some(struct_ty) = ty_ref.downcast_ref::<llvm_types::StructType>() {
            TypeShape::Struct(struct_ty.fields().collect())
        } else {
            TypeShape::Identity
        }
    };

    match shape {
        TypeShape::Bool => Ok(StorageRewrite {
            ty: IntegerType::get(ctx, 8, Signedness::Signless).into(),
            shared_pointer_leaves: 0,
            array_shared_pointer_leaves: 0,
        }),
        TypeShape::Pointer(address_space) if address_space == llvm_types::address_space::SHARED => {
            Ok(StorageRewrite {
                ty: llvm_types::PointerType::get_generic(ctx).into(),
                shared_pointer_leaves: 1,
                array_shared_pointer_leaves: 0,
            })
        }
        TypeShape::Pointer(_) | TypeShape::Identity => Ok(StorageRewrite {
            ty: semantic_ty,
            shared_pointer_leaves: 0,
            array_shared_pointer_leaves: 0,
        }),
        TypeShape::Array(element_ty, count) => {
            let element = rewrite_storage_type(ctx, element_ty)?;
            let shared_pointer_leaves = element
                .shared_pointer_leaves
                .checked_mul(count)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "enum payload storage: shared-pointer array rewrite size overflow"
                    )
                })?;
            let ty = if element.ty == element_ty {
                semantic_ty
            } else {
                llvm_types::ArrayType::get(ctx, element.ty, count).into()
            };
            Ok(StorageRewrite {
                ty,
                shared_pointer_leaves,
                // The array multiplies everything below it, so every shared
                // leaf it contains is array-expanded, including leaves nested
                // through structs inside the element.
                array_shared_pointer_leaves: shared_pointer_leaves,
            })
        }
        TypeShape::Vector(element_ty) => {
            let element = rewrite_storage_type(ctx, element_ty)?;
            if element.ty != element_ty || element.shared_pointer_leaves != 0 {
                return Err(anyhow::anyhow!(
                    "enum payload storage: vectors containing bool or shared-memory pointer elements are not supported"
                ));
            }
            Ok(StorageRewrite {
                ty: semantic_ty,
                shared_pointer_leaves: 0,
                array_shared_pointer_leaves: 0,
            })
        }
        TypeShape::Struct(fields) => {
            let mut storage_fields = Vec::with_capacity(fields.len());
            let mut changed = false;
            let mut shared_pointer_leaves = 0_u64;
            let mut array_shared_pointer_leaves = 0_u64;
            for field in fields {
                let storage = rewrite_storage_type(ctx, field)?;
                changed |= storage.ty != field;
                shared_pointer_leaves = shared_pointer_leaves
                    .checked_add(storage.shared_pointer_leaves)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "enum payload storage: shared-pointer leaf count overflow while rewriting a struct payload"
                        )
                    })?;
                array_shared_pointer_leaves = array_shared_pointer_leaves
                    .checked_add(storage.array_shared_pointer_leaves)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "enum payload storage: shared-pointer leaf count overflow while rewriting a struct payload"
                        )
                    })?;
                storage_fields.push(storage.ty);
            }
            let ty = if changed {
                llvm_types::StructType::get_unnamed(ctx, storage_fields).into()
            } else {
                semantic_ty
            };
            Ok(StorageRewrite {
                ty,
                shared_pointer_leaves,
                array_shared_pointer_leaves,
            })
        }
    }
}

enum ValueCoercion {
    BoolToByte,
    ByteToBool,
    SharedToGeneric,
    GenericToShared,
    Array {
        target_element: TypeHandle,
        count: u64,
    },
    Struct(Vec<TypeHandle>),
    Unsupported,
}

/// Convert an enum payload value between its semantic and physical types.
///
/// This is the value-level counterpart of [`enum_payload_storage_type`]. It
/// recursively rebuilds structs/tuples, inserts the required address-space
/// casts for shared pointer leaves, and canonicalizes/decanonicalizes bool
/// bytes. The conversion is symmetric and is used both when constructing an
/// enum and when extracting its payload.
pub(crate) fn coerce_enum_payload_value(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: Value,
    target_ty: TypeHandle,
) -> Result<Value> {
    let source_ty = value.get_type(ctx);
    if source_ty == target_ty {
        return Ok(value);
    }

    let coercion = {
        let source_ref = source_ty.deref(ctx);
        let target_ref = target_ty.deref(ctx);

        let source_integer = source_ref
            .downcast_ref::<IntegerType>()
            .map(IntegerType::width);
        let target_integer = target_ref
            .downcast_ref::<IntegerType>()
            .map(IntegerType::width);
        if source_integer == Some(1) && target_integer == Some(8) {
            ValueCoercion::BoolToByte
        } else if source_integer == Some(8) && target_integer == Some(1) {
            ValueCoercion::ByteToBool
        } else if let (Some(source_pointer), Some(target_pointer)) = (
            source_ref.downcast_ref::<llvm_types::PointerType>(),
            target_ref.downcast_ref::<llvm_types::PointerType>(),
        ) {
            match (
                source_pointer.address_space(),
                target_pointer.address_space(),
            ) {
                (llvm_types::address_space::SHARED, llvm_types::address_space::GENERIC) => {
                    ValueCoercion::SharedToGeneric
                }
                (llvm_types::address_space::GENERIC, llvm_types::address_space::SHARED) => {
                    ValueCoercion::GenericToShared
                }
                _ => ValueCoercion::Unsupported,
            }
        } else if let (Some(source_array), Some(target_array)) = (
            source_ref.downcast_ref::<llvm_types::ArrayType>(),
            target_ref.downcast_ref::<llvm_types::ArrayType>(),
        ) {
            if source_array.size() == target_array.size() {
                ValueCoercion::Array {
                    target_element: target_array.elem_type(),
                    count: source_array.size(),
                }
            } else {
                ValueCoercion::Unsupported
            }
        } else if let (Some(source_struct), Some(target_struct)) = (
            source_ref.downcast_ref::<llvm_types::StructType>(),
            target_ref.downcast_ref::<llvm_types::StructType>(),
        ) {
            if source_struct.num_fields() == target_struct.num_fields() {
                ValueCoercion::Struct(target_struct.fields().collect())
            } else {
                ValueCoercion::Unsupported
            }
        } else {
            ValueCoercion::Unsupported
        }
    };

    match coercion {
        ValueCoercion::BoolToByte => {
            let zext = llvm::ZExtOp::new_with_nneg(ctx, value, target_ty, false);
            rewriter.insert_operation(ctx, zext.get_operation());
            Ok(zext.get_operation().deref(ctx).get_result(0))
        }
        ValueCoercion::ByteToBool => {
            let trunc = llvm::TruncOp::new(ctx, value, target_ty);
            rewriter.insert_operation(ctx, trunc.get_operation());
            Ok(trunc.get_operation().deref(ctx).get_result(0))
        }
        ValueCoercion::SharedToGeneric | ValueCoercion::GenericToShared => {
            let cast = llvm::AddrSpaceCastOp::new(ctx, value, target_ty);
            rewriter.insert_operation(ctx, cast.get_operation());
            Ok(cast.get_operation().deref(ctx).get_result(0))
        }
        ValueCoercion::Array {
            target_element,
            count,
        } => {
            let undef = llvm::UndefOp::new(ctx, target_ty);
            rewriter.insert_operation(ctx, undef.get_operation());
            let mut current = undef.get_operation().deref(ctx).get_result(0);
            for index in 0..count {
                let index = u32::try_from(index).map_err(|_| {
                    pliron::input_error_noloc!(
                        "enum payload storage array has more than u32::MAX elements"
                    )
                })?;
                let extract = llvm::ExtractValueOp::new(ctx, value, vec![index])?;
                rewriter.insert_operation(ctx, extract.get_operation());
                let element = extract.get_operation().deref(ctx).get_result(0);
                let converted = coerce_enum_payload_value(ctx, rewriter, element, target_element)?;
                let insert = llvm::InsertValueOp::new(ctx, current, converted, vec![index]);
                rewriter.insert_operation(ctx, insert.get_operation());
                current = insert.get_operation().deref(ctx).get_result(0);
            }
            Ok(current)
        }
        ValueCoercion::Struct(fields) => {
            let undef = llvm::UndefOp::new(ctx, target_ty);
            rewriter.insert_operation(ctx, undef.get_operation());
            let mut current = undef.get_operation().deref(ctx).get_result(0);
            for (index, target_field) in fields.into_iter().enumerate() {
                let index = u32::try_from(index).map_err(|_| {
                    pliron::input_error_noloc!(
                        "enum payload storage struct has more than u32::MAX fields"
                    )
                })?;
                let extract = llvm::ExtractValueOp::new(ctx, value, vec![index])?;
                rewriter.insert_operation(ctx, extract.get_operation());
                let field = extract.get_operation().deref(ctx).get_result(0);
                let converted = coerce_enum_payload_value(ctx, rewriter, field, target_field)?;
                let insert = llvm::InsertValueOp::new(ctx, current, converted, vec![index]);
                rewriter.insert_operation(ctx, insert.get_operation());
                current = insert.get_operation().deref(ctx).get_result(0);
            }
            Ok(current)
        }
        ValueCoercion::Unsupported => pliron::input_err_noloc!(
            "enum payload storage type mismatch: {} cannot be adapted to {}",
            source_ty.deref(ctx).disp(ctx),
            target_ty.deref(ctx).disp(ctx)
        ),
    }
}
