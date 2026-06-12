/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cast operation conversion: `dialect-mir` → LLVM dialect.
//!
//! Dispatches on `MirCastKindAttr` (preserved from Rust MIR) to select the
//! correct LLVM instruction. This avoids guessing cast semantics from types.
//!
//! # Cast Dispatch
//!
//! | MirCastKindAttr                | LLVM Operation                                         |
//! |--------------------------------|--------------------------------------------------------|
//! | Transmute                      | `emit_pointer_cast` (see below)                        |
//! | IntToInt (wider, signed)       | `sext`                                                 |
//! | IntToInt (wider, unsigned)     | `zext`                                                 |
//! | IntToInt (narrower)            | `trunc`                                                |
//! | IntToInt (same width)          | `bitcast`                                              |
//! | IntToFloat                     | `sitofp` or `uitofp`                                   |
//! | FloatToInt                     | `llvm.fptosi.sat` / `llvm.fptoui.sat` (Rust semantics) |
//! | FloatToFloat                   | `fpext` or `fptrunc`                                   |
//! | PtrToPtr / FnPtrToPtr          | `emit_pointer_cast` (see below)                        |
//! | PointerCoercionUnsize          | `emit_unsize_cast` → `emit_pointer_cast` (see below)   |
//! | PointerCoercion* (other)       | `emit_pointer_cast` (see below)                        |
//! | PointerExposeAddress           | `ptrtoint`                                             |
//! | PointerWithExposedProvenance   | `inttoptr`                                             |
//!
//! ## `emit_unsize_cast` handles array→slice unsizing:
//! | Source → Dest                  | LLVM Operation                                  |
//! |--------------------------------|-------------------------------------------------|
//! | ptr-to-array → struct (slice)  | `insertvalue` ptr + `insertvalue` len into undef |
//! | other                          | falls through to `emit_pointer_cast`             |
//!
//! ## `emit_pointer_cast` handles struct↔scalar conversions, including
//! niche-encoded enums:
//! | Source → Dest                                  | LLVM Operation                              |
//! |------------------------------------------------|---------------------------------------------|
//! | Transmute with `niche_encoding`, dst struct    | `icmp` + `select` + nested `insertvalue`    |
//! | struct → ptr (fat→thin)                        | `extractvalue` field 0                      |
//! | ptr → struct (thin→fat, no niche)              | `insertvalue` into undef                    |
//! | ptr → integer                                  | `ptrtoint`                                  |
//! | integer → ptr                                  | `inttoptr`                                  |
//! | struct → struct (transmute)                    | `alloca` + `store` + `load`                 |
//! | ptr → ptr (diff addrspace)                     | `addrspacecast`                             |
//! | struct → integer, equal size                   | `alloca` + `store` + `load`                 |
//! | struct → integer, mismatched size              | cuda-oxide error (see issue #21)            |
//! | array ↔ anything, equal size                   | `alloca` + `store` + `load`                 |
//! | array ↔ anything, mismatched size              | cuda-oxide error (see issue #125)           |
//! | otherwise                                      | `bitcast`                                   |
//!
//! The niche path runs first because it is the only correct lowering when
//! the importer has classified the destination as a niche-optimised enum,
//! regardless of whether the scalar source happens to be a pointer (e.g.
//! `Option<&T>`) or an integer (e.g. `Option<NonZeroUsize>`).

use crate::convert::types::convert_type;
use crate::helpers;
use dialect_mir::attributes::MirCastKindAttr;
use dialect_mir::ops::MirCastOp;
use dialect_mir::types::{MirArrayType, MirPtrType};
use llvm_export::op_interfaces::CastOpInterface;
use llvm_export::ops as llvm;
use llvm_export::types::FuncType;
use pliron::builtin::op_interfaces::CallOpCallable;
use pliron::builtin::type_interfaces::FloatTypeInterface;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;
use pliron::result::Result;
use pliron::r#type::{Typed, type_cast};

/// Convert a MIR cast operation to the appropriate LLVM cast instruction.
///
/// Dispatches on the `cast_kind` attribute to determine semantics, then uses
/// source/destination types for the specific instruction selection within each kind.
pub fn convert(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let loc = op.deref(ctx).loc();
    let operands: Vec<_> = op.deref(ctx).operands().collect();

    let val = match operands.as_slice() {
        [val] => *val,
        _ => return pliron::input_err!(loc, "Cast requires exactly 1 operand"),
    };

    let cast_op = MirCastOp::new(op);
    let cast_kind_ref = cast_op.get_attr_cast_kind(ctx).ok_or_else(|| {
        pliron::input_error!(loc.clone(), "MirCastOp missing cast_kind attribute")
    })?;
    let cast_kind = cast_kind_ref.clone();
    drop(cast_kind_ref);

    // Pre-conversion MIR operand type — preserves signedness info from Rust's type system
    let mir_opd = op.deref(ctx).get_operand(0);
    let mir_opd_ty = operands_info
        .lookup_most_recent_type(mir_opd)
        .unwrap_or_else(|| mir_opd.get_type(ctx));
    // Pre-conversion MIR result type — preserves signedness (LLVM types are signless)
    let mir_result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let llvm_ty = convert_type(ctx, mir_result_ty).map_err(|e| pliron::input_error!(loc, "{e}"))?;
    let val_ty = val.get_type(ctx);

    let llvm_op = match &cast_kind {
        MirCastKindAttr::Transmute => emit_pointer_cast(ctx, rewriter, op, val, val_ty, llvm_ty)?,

        MirCastKindAttr::IntToInt => {
            let src_w = val_ty
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .map(|t| t.width())
                .ok_or_else(|| pliron::input_error_noloc!("IntToInt: source is not an integer"))?;
            let dst_w = llvm_ty
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .map(|t| t.width())
                .ok_or_else(|| {
                    pliron::input_error_noloc!("IntToInt: destination is not an integer")
                })?;
            convert_int_to_int(ctx, rewriter, val, llvm_ty, src_w, dst_w, mir_opd_ty)?
        }

        MirCastKindAttr::IntToFloat => {
            convert_int_to_float(ctx, rewriter, val, llvm_ty, mir_opd_ty)?
        }

        MirCastKindAttr::FloatToInt => {
            convert_float_to_int(ctx, rewriter, op, val, llvm_ty, mir_result_ty)?
        }

        MirCastKindAttr::FloatToFloat => {
            convert_float_to_float(ctx, rewriter, val, llvm_ty, val_ty)?
        }

        MirCastKindAttr::PointerCoercionUnsize => {
            emit_unsize_cast(ctx, rewriter, op, val, val_ty, llvm_ty, mir_opd_ty)?
        }

        MirCastKindAttr::PtrToPtr
        | MirCastKindAttr::FnPtrToPtr
        | MirCastKindAttr::PointerCoercionMutToConst
        | MirCastKindAttr::PointerCoercionReifyFnPointer
        | MirCastKindAttr::PointerCoercionUnsafeFnPointer
        | MirCastKindAttr::PointerCoercionClosureFnPointer
        | MirCastKindAttr::PointerCoercionArrayToPointer
        | MirCastKindAttr::Subtype => emit_pointer_cast(ctx, rewriter, op, val, val_ty, llvm_ty)?,

        MirCastKindAttr::PointerExposeAddress => {
            llvm::PtrToIntOp::new(ctx, val, llvm_ty).get_operation()
        }

        MirCastKindAttr::PointerWithExposedProvenance => {
            llvm::IntToPtrOp::new(ctx, val, llvm_ty).get_operation()
        }
    };

    rewriter.insert_operation(ctx, llvm_op);
    rewriter.replace_operation(ctx, op, llvm_op);

    Ok(())
}

/// Integer → integer: extension, truncation, or same-width bitcast.
fn convert_int_to_int(
    ctx: &mut Context,
    _rewriter: &mut DialectConversionRewriter,
    val: pliron::value::Value,
    llvm_ty: Ptr<pliron::r#type::TypeObj>,
    src_w: u32,
    dst_w: u32,
    mir_opd_ty: Ptr<pliron::r#type::TypeObj>,
) -> Result<Ptr<Operation>> {
    if dst_w > src_w {
        let is_signed = {
            let ty_obj = mir_opd_ty.deref(ctx);
            ty_obj
                .downcast_ref::<IntegerType>()
                .ok_or_else(|| {
                    pliron::input_error_noloc!("IntToInt: MIR operand type is not an integer")
                })?
                .signedness()
                == Signedness::Signed
        };

        if is_signed {
            Ok(llvm::SExtOp::new(ctx, val, llvm_ty).get_operation())
        } else {
            let zext = llvm::ZExtOp::new(ctx, val, llvm_ty);
            let nneg_key: pliron::identifier::Identifier = "llvm_nneg_flag".try_into().unwrap();
            zext.get_operation().deref_mut(ctx).attributes.0.insert(
                nneg_key,
                pliron::builtin::attributes::BoolAttr::new(false).into(),
            );
            Ok(zext.get_operation())
        }
    } else if dst_w < src_w {
        Ok(llvm::TruncOp::new(ctx, val, llvm_ty).get_operation())
    } else {
        Ok(llvm::BitcastOp::new(ctx, val, llvm_ty).get_operation())
    }
}

/// Integer → float: signed or unsigned conversion.
fn convert_int_to_float(
    ctx: &mut Context,
    _rewriter: &mut DialectConversionRewriter,
    val: pliron::value::Value,
    llvm_ty: Ptr<pliron::r#type::TypeObj>,
    mir_opd_ty: Ptr<pliron::r#type::TypeObj>,
) -> Result<Ptr<Operation>> {
    let is_signed = {
        let ty_obj = mir_opd_ty.deref(ctx);
        ty_obj
            .downcast_ref::<IntegerType>()
            .ok_or_else(|| {
                pliron::input_error_noloc!("IntToFloat: MIR operand type is not an integer")
            })?
            .signedness()
            == Signedness::Signed
    };

    if is_signed {
        Ok(llvm::SIToFPOp::new(ctx, val, llvm_ty).get_operation())
    } else {
        let uitofp = llvm::UIToFPOp::new(ctx, val, llvm_ty);
        let nneg_key: pliron::identifier::Identifier = "llvm_nneg_flag".try_into().unwrap();
        uitofp.get_operation().deref_mut(ctx).attributes.0.insert(
            nneg_key,
            pliron::builtin::attributes::BoolAttr::new(false).into(),
        );
        Ok(uitofp.get_operation())
    }
}

/// Float → integer: signed or unsigned conversion (saturating, Rust semantics).
///
/// Uses LLVM's `llvm.fptosi.sat` / `llvm.fptoui.sat` intrinsics so that
/// out-of-range values saturate to T::MIN/T::MAX and NaN → 0, matching Rust.
/// Uses the **MIR** result type for signedness — the LLVM integer type is signless.
fn convert_float_to_int(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    val: pliron::value::Value,
    llvm_ty: Ptr<pliron::r#type::TypeObj>,
    mir_result_ty: Ptr<pliron::r#type::TypeObj>,
) -> Result<Ptr<Operation>> {
    let val_ty = val.get_type(ctx);
    let is_signed = {
        let ty_obj = mir_result_ty.deref(ctx);
        ty_obj
            .downcast_ref::<IntegerType>()
            .ok_or_else(|| {
                pliron::input_error_noloc!("FloatToInt: MIR result type is not an integer")
            })?
            .signedness()
            == Signedness::Signed
    };

    let int_width = llvm_ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|t| t.width())
        .ok_or_else(|| {
            pliron::input_error!(
                op.deref(ctx).loc(),
                "FloatToInt: result type is not an integer"
            )
        })?;
    let int_suffix = format!("i{}", int_width);

    let float_suffix = match float_bit_width(ctx, val_ty) {
        Ok(16) => "f16",
        Ok(32) => "f32",
        Ok(64) => "f64",
        Ok(bits) => {
            return pliron::input_err!(
                op.deref(ctx).loc(),
                "FloatToInt: unsupported source float width {bits}"
            );
        }
        Err(err) => return Err(err),
    };

    let intrinsic_name = if is_signed {
        format!("llvm_fptosi_sat_{}_{}", int_suffix, float_suffix)
    } else {
        format!("llvm_fptoui_sat_{}_{}", int_suffix, float_suffix)
    };

    let func_ty = FuncType::get(ctx, llvm_ty, vec![val_ty], false);

    // Navigate from op to its containing block for intrinsic declaration
    let llvm_block = op
        .deref(ctx)
        .get_parent_block()
        .ok_or_else(|| pliron::input_error!(op.deref(ctx).loc(), "Cast op has no parent block"))?;
    helpers::ensure_intrinsic_declared(ctx, llvm_block, &intrinsic_name, func_ty).map_err(|e| {
        pliron::input_error!(op.deref(ctx).loc(), "Failed to declare intrinsic: {e}")
    })?;

    let sym_name: pliron::identifier::Identifier =
        intrinsic_name.as_str().try_into().map_err(|e| {
            pliron::input_error!(op.deref(ctx).loc(), "Invalid intrinsic name: {:?}", e)
        })?;
    let callee = CallOpCallable::Direct(sym_name);
    let llvm_call = llvm::CallOp::new(ctx, callee, func_ty, vec![val]);

    // The call op is the final replacement, but we need intermediate ops inserted by rewriter.
    // Don't insert here — the caller handles insert + replace.
    let _ = &rewriter;
    Ok(llvm_call.get_operation())
}

/// Emit an Unsize coercion: `&[T; N]` → `&[T]` (or `*[T; N]` → `[T]`).
///
/// When the MIR source is a pointer to an array and the LLVM destination is a
/// fat-pointer struct `{ ptr, i64 }`, we construct the full slice by inserting
/// both the data pointer (field 0) and the array length (field 1).
///
/// For other Unsize coercions (e.g., trait objects), falls through to
/// `emit_pointer_cast`.
fn emit_unsize_cast(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    val: pliron::value::Value,
    val_ty: Ptr<pliron::r#type::TypeObj>,
    llvm_ty: Ptr<pliron::r#type::TypeObj>,
    mir_opd_ty: Ptr<pliron::r#type::TypeObj>,
) -> Result<Ptr<Operation>> {
    let array_len = {
        let mir_ref = mir_opd_ty.deref(ctx);
        mir_ref.downcast_ref::<MirPtrType>().and_then(|ptr_ty| {
            let pointee_ref = ptr_ty.pointee.deref(ctx);
            if let Some(arr) = pointee_ref.downcast_ref::<MirArrayType>() {
                // `&[T; N] -> &[T]`: the classic array unsize.
                Some(arr.size())
            } else if let Some(struct_ty) =
                pointee_ref.downcast_ref::<dialect_mir::types::MirStructType>()
            {
                // `&S<[T; N]> -> &S<[T]>` where the struct's LAST field is
                // the array that becomes the unsized tail (e.g. the
                // `PolymorphicIter` inside `core::array::IntoIter`, which
                // every `for x in arr` loop unsizes; issue #138). The fat
                // pointer's metadata is that array's element count.
                let field_types = struct_ty.field_types();
                let last_decl_idx = match struct_ty.memory_order().last().copied() {
                    Some(idx) => idx,
                    None => field_types.len().checked_sub(1)?,
                };
                field_types.get(last_decl_idx).and_then(|t| {
                    t.deref(ctx)
                        .downcast_ref::<MirArrayType>()
                        .map(|a| a.size())
                })
            } else {
                None
            }
        })
    };

    if let Some(len) = array_len {
        let dst_is_struct = llvm_ty.deref(ctx).is::<llvm_export::types::StructType>();

        if dst_is_struct {
            let undef = llvm::UndefOp::new(ctx, llvm_ty);
            rewriter.insert_operation(ctx, undef.get_operation());
            let undef_val = undef.get_operation().deref(ctx).get_result(0);

            let insert_ptr = llvm::InsertValueOp::new(ctx, undef_val, val, vec![0]);
            rewriter.insert_operation(ctx, insert_ptr.get_operation());
            let with_ptr = insert_ptr.get_operation().deref(ctx).get_result(0);

            let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
            let len_apint = pliron::utils::apint::APInt::from_i64(
                len as i64,
                std::num::NonZeroUsize::new(64).unwrap(),
            );
            let len_attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, len_apint);
            let len_const = llvm::ConstantOp::new(ctx, len_attr.into());
            rewriter.insert_operation(ctx, len_const.get_operation());
            let len_val = len_const.get_operation().deref(ctx).get_result(0);

            return Ok(llvm::InsertValueOp::new(ctx, with_ptr, len_val, vec![1]).get_operation());
        }
    }

    emit_pointer_cast(ctx, rewriter, op, val, val_ty, llvm_ty)
}

/// Emit a pointer-compatible cast, handling the struct↔ptr patterns that arise
/// because our type system represents fat pointers (slices) as `{ ptr, i64 }` structs.
///
/// LLVM does not allow `bitcast` between structs and scalars/pointers, so:
/// - struct → ptr: `extractvalue` field 0 (extract data pointer from fat pointer)
/// - ptr → struct: `insertvalue` into undef at field 0 (wrap thin ptr in fat pointer)
/// - ptr → ptr (different address space): `addrspacecast`
/// - array ↔ anything: memory round-trip (`alloca` + `store` + `load`),
///   because `bitcast` is only defined between non-aggregate first-class
///   types (e.g. `u32::from_ne_bytes` transmutes `[u8; 4]` → `u32`)
/// - otherwise: `bitcast`
fn emit_pointer_cast(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    val: pliron::value::Value,
    val_ty: Ptr<pliron::r#type::TypeObj>,
    llvm_ty: Ptr<pliron::r#type::TypeObj>,
) -> Result<Ptr<Operation>> {
    let src_is_struct = val_ty.deref(ctx).is::<llvm_export::types::StructType>();
    let dst_is_struct = llvm_ty.deref(ctx).is::<llvm_export::types::StructType>();
    let src_as = val_ty
        .deref(ctx)
        .downcast_ref::<llvm_export::types::PointerType>()
        .map(|pt| pt.address_space());
    let dst_as = llvm_ty
        .deref(ctx)
        .downcast_ref::<llvm_export::types::PointerType>()
        .map(|pt| pt.address_space());
    let dst_is_ptr = dst_as.is_some();
    let src_is_ptr = src_as.is_some();
    let src_is_int = val_ty.deref(ctx).is::<IntegerType>();
    let src_is_array = val_ty.deref(ctx).is::<llvm_export::types::ArrayType>();
    let dst_is_array = llvm_ty.deref(ctx).is::<llvm_export::types::ArrayType>();

    // Niched-enum Transmute first. Rustc stores `Option<NonZeroT>`,
    // `Option<&T>`, `Option<Box<T>>`, `Option<NonNull<T>>`,
    // `Option<bool>`, `Option<char>`, ... as a single scalar where one
    // forbidden bit pattern of the inner type stands in for `None`. When
    // that scalar form has to be materialised as our un-niched
    // `{ discriminant, payload }` aggregate, rustc emits a Transmute and
    // the importer attaches a `niche_encoding` attribute. We rebuild the
    // aggregate explicitly here. This branch runs **before** the legacy
    // ptr→struct fat-pointer arm because a niched-enum destination has an
    // `i8` discriminant in field 0, not a pointer slot. See issue #21.
    if dst_is_struct
        && let Some(niche) = read_niche_info(ctx, op)
        && (src_is_int || src_is_ptr)
    {
        return emit_scalar_to_niched_enum(ctx, rewriter, op, val, val_ty, llvm_ty, niche);
    }

    if src_is_struct && dst_is_ptr {
        Ok(llvm::ExtractValueOp::new(ctx, val, vec![0])
            .map_err(|e| pliron::input_error_noloc!("pointer cast ExtractValueOp: {e}"))?
            .get_operation())
    } else if src_is_ptr && dst_is_struct {
        let undef = llvm::UndefOp::new(ctx, llvm_ty);
        rewriter.insert_operation(ctx, undef.get_operation());
        let undef_val = undef.get_operation().deref(ctx).get_result(0);
        Ok(llvm::InsertValueOp::new(ctx, undef_val, val, vec![0]).get_operation())
    } else if src_is_ptr && llvm_ty.deref(ctx).is::<IntegerType>() {
        Ok(llvm::PtrToIntOp::new(ctx, val, llvm_ty).get_operation())
    } else if src_is_int && dst_is_ptr {
        Ok(llvm::IntToPtrOp::new(ctx, val, llvm_ty).get_operation())
    } else if src_is_struct && dst_is_struct {
        // struct → struct: LLVM forbids bitcast between aggregates with
        // different field layouts. Go through memory: alloca + store + load.
        let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
        let one = {
            let apint =
                pliron::utils::apint::APInt::from_i64(1, std::num::NonZeroUsize::new(64).unwrap());
            let attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, apint);
            let c = llvm::ConstantOp::new(ctx, attr.into());
            rewriter.insert_operation(ctx, c.get_operation());
            c.get_operation().deref(ctx).get_result(0)
        };
        let alloca = llvm::AllocaOp::new(ctx, val_ty, one);
        rewriter.insert_operation(ctx, alloca.get_operation());
        let ptr = alloca.get_operation().deref(ctx).get_result(0);

        let store = llvm::StoreOp::new(ctx, val, ptr);
        rewriter.insert_operation(ctx, store.get_operation());

        Ok(llvm::LoadOp::new(ctx, ptr, llvm_ty).get_operation())
    } else if let (Some(s), Some(d)) = (src_as, dst_as) {
        if s != d {
            let cast_ty = llvm_export::types::PointerType::get(ctx, d).into();
            Ok(llvm::AddrSpaceCastOp::new(ctx, val, cast_ty).get_operation())
        } else {
            Ok(llvm::BitcastOp::new(ctx, val, llvm_ty).get_operation())
        }
    } else if src_is_int && dst_is_struct {
        // Scalar -> aggregate Transmute with no niche encoding: we cannot
        // safely guess the layout. Refuse loudly rather than fall through
        // to an invalid bitcast.
        pliron::input_err_noloc!(
            "scalar -> aggregate Transmute without niche encoding; the importer did not \
             classify this destination as a niche-optimised enum. Refusing to fall \
             through to an invalid bitcast (see issue #21)."
        )
    } else if src_is_struct && llvm_ty.deref(ctx).is::<IntegerType>() {
        emit_struct_to_scalar(ctx, rewriter, val, val_ty, llvm_ty)
    } else if src_is_array || dst_is_array {
        // Array on either side (e.g. `u32::from_ne_bytes` is a
        // `[u8; 4]` → `u32` Transmute, `u32::to_ne_bytes` the reverse).
        // LLVM's `bitcast` is only defined between non-aggregate
        // first-class types, so an aggregate must go through memory.
        emit_transmute_via_memory(ctx, rewriter, val, val_ty, llvm_ty)
    } else {
        Ok(llvm::BitcastOp::new(ctx, val, llvm_ty).get_operation())
    }
}

fn const_i64(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    n: i64,
) -> pliron::value::Value {
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let apint = pliron::utils::apint::APInt::from_i64(n, std::num::NonZeroUsize::new(64).unwrap());
    let attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, apint);
    let c = llvm::ConstantOp::new(ctx, attr.into());
    rewriter.insert_operation(ctx, c.get_operation());
    c.get_operation().deref(ctx).get_result(0)
}

fn const_int_of(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    ty: Ptr<pliron::r#type::TypeObj>,
    value: i64,
) -> Result<pliron::value::Value> {
    let int_ty = ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .ok_or_else(|| pliron::input_error_noloc!("const_int_of: expected IntegerType"))?
        .clone();
    let width = std::num::NonZeroUsize::new(int_ty.width() as usize)
        .ok_or_else(|| pliron::input_error_noloc!("const_int_of: zero-width integer"))?;
    let apint = pliron::utils::apint::APInt::from_i64(value, width);
    let attr = pliron::builtin::attributes::IntegerAttr::new(
        IntegerType::get(ctx, int_ty.width(), int_ty.signedness()),
        apint,
    );
    let c = llvm::ConstantOp::new(ctx, attr.into());
    rewriter.insert_operation(ctx, c.get_operation());
    Ok(c.get_operation().deref(ctx).get_result(0))
}

/// Maximum newtype-wrapper depth we will descend through when locating a
/// scalar slot inside an aggregate (`NonZero<T>` -> `Pat<T, _>` -> `T`
/// is three layers; eight gives generous headroom for hand-rolled chains
/// without risking pathological infinite loops on cyclic-looking types).
const MAX_NEWTYPE_DEPTH: usize = 8;

/// Find the `insertvalue` index path that lands `scalar_ty` at the deepest
/// scalar slot of `aggregate_ty`, descending through single-field struct
/// wrappers (the `NonZero<T>` -> `Pat<T, _>` -> `T` chain, or
/// `{ ptr }` for `Option<&T>`). Returns `None` when no compatible scalar
/// slot exists within `MAX_NEWTYPE_DEPTH` layers.
fn deep_scalar_index_path(
    ctx: &Context,
    aggregate_ty: Ptr<pliron::r#type::TypeObj>,
    scalar_ty: Ptr<pliron::r#type::TypeObj>,
) -> Option<Vec<u32>> {
    let mut path = Vec::new();
    let mut current = aggregate_ty;
    for _ in 0..MAX_NEWTYPE_DEPTH {
        if current == scalar_ty {
            return Some(path);
        }
        // Same-width integers are interchangeable even if not pointer-equal
        // because LLVM integer types are signless while MIR carries
        // signedness.
        if let (Some(c), Some(t)) = (
            current.deref(ctx).downcast_ref::<IntegerType>(),
            scalar_ty.deref(ctx).downcast_ref::<IntegerType>(),
        ) && c.width() == t.width()
        {
            return Some(path);
        }
        let next = {
            let r = current.deref(ctx);
            let s = r.downcast_ref::<llvm_export::types::StructType>()?;
            if s.num_fields() != 1 {
                return None;
            }
            s.field_type(0)
        };
        path.push(0);
        current = next;
    }
    None
}

/// Read the niche encoding off a `MirCastOp` via its typed accessor.
/// Returns `None` when the importer did not attach one (i.e. the cast is
/// not into a niched enum).
fn read_niche_info(
    ctx: &Context,
    op: Ptr<Operation>,
) -> Option<dialect_mir::attributes::NicheEncodingAttr> {
    dialect_mir::ops::MirCastOp::new(op)
        .get_attr_niche_encoding(ctx)
        .map(|r| r.clone())
}

/// Rebuild a `MirEnumType` aggregate from the scalar `val` rustc passed
/// through a niche-encoded Transmute, using the niche info the importer
/// attached. Accepts both integer-scalar sources (e.g. `i64` for
/// `Option<NonZeroUsize>`) and pointer-scalar sources (e.g. `ptr` for
/// `Option<&T>`); the comparison is done in the source's own type so we
/// never need a `ptrtoint` round-trip.
fn emit_scalar_to_niched_enum(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    val: pliron::value::Value,
    val_ty: Ptr<pliron::r#type::TypeObj>,
    llvm_ty: Ptr<pliron::r#type::TypeObj>,
    niche: dialect_mir::attributes::NicheEncodingAttr,
) -> Result<Ptr<Operation>> {
    let (disc_ty, payload_ty) = {
        let r = llvm_ty.deref(ctx);
        let s = r
            .downcast_ref::<llvm_export::types::StructType>()
            .ok_or_else(|| {
                pliron::input_error_noloc!("emit_scalar_to_niched_enum: dst is not a struct")
            })?;
        if s.num_fields() != 2 {
            return pliron::input_err_noloc!(
                "niched-enum aggregate must have exactly 2 fields (discriminant, payload), got {}",
                s.num_fields()
            );
        }
        (s.field_type(0), s.field_type(1))
    };

    // Field 0 carries ONE tag semantic everywhere: the variant's declared
    // discriminant VALUE. The niche attribute records variant INDICES, so
    // map them through the result enum's variant_discriminants before
    // storing. For Option-likes (discriminants [0, 1]) this is the
    // identity, but enums with explicit discriminants must not see a raw
    // index in the tag slot (issue #132 groundwork).
    let (niche_disc_value, untagged_disc_value) = {
        let mir_result_ty = op.deref(ctx).get_result(0).get_type(ctx);
        let mir_result_ty_obj = mir_result_ty.deref(ctx);
        let enum_ty = mir_result_ty_obj
            .downcast_ref::<dialect_mir::types::MirEnumType>()
            .ok_or_else(|| {
                pliron::input_error_noloc!(
                    "emit_scalar_to_niched_enum: niche-encoded cast result is not a MirEnumType"
                )
            })?;
        let discr_of = |idx: u32| -> Result<u64> {
            enum_ty
                .variant_discriminants
                .get(idx as usize)
                .copied()
                .ok_or_else(|| {
                    pliron::input_error_noloc!(
                        "emit_scalar_to_niched_enum: variant index {} has no discriminant ({} discriminants recorded)",
                        idx,
                        enum_ty.variant_discriminants.len()
                    )
                })
        };
        (
            discr_of(niche.niche_variant_idx)?,
            discr_of(niche.untagged_variant_idx)?,
        )
    };

    // Build a comparison constant in the source's own type. For integer
    // sources that's just the niche bit pattern; for pointer sources we
    // construct it as `inttoptr i64 <niche_start>` (which folds to `null`
    // when niche_start is 0, the case rustc actually emits).
    let src_is_ptr = val_ty.deref(ctx).is::<llvm_export::types::PointerType>();
    let cmp_const = if src_is_ptr {
        let i64_ty: Ptr<pliron::r#type::TypeObj> =
            IntegerType::get(ctx, 64, Signedness::Signless).into();
        let i64_const = const_int_of(ctx, rewriter, i64_ty, niche.niche_start as i64)?;
        let i2p = llvm::IntToPtrOp::new(ctx, i64_const, val_ty);
        rewriter.insert_operation(ctx, i2p.get_operation());
        i2p.get_operation().deref(ctx).get_result(0)
    } else {
        const_int_of(ctx, rewriter, val_ty, niche.niche_start as i64)?
    };

    let icmp = llvm::ICmpOp::new(
        ctx,
        llvm_export::attributes::ICmpPredicateAttr::EQ,
        val,
        cmp_const,
    );
    rewriter.insert_operation(ctx, icmp.get_operation());
    let is_niche = icmp.get_operation().deref(ctx).get_result(0);

    let niche_disc = const_int_of(ctx, rewriter, disc_ty, niche_disc_value as i64)?;
    let untagged_disc = const_int_of(ctx, rewriter, disc_ty, untagged_disc_value as i64)?;
    let disc_select = llvm::SelectOp::new(ctx, is_niche, niche_disc, untagged_disc);
    rewriter.insert_operation(ctx, disc_select.get_operation());
    let disc = disc_select.get_operation().deref(ctx).get_result(0);

    let undef = llvm::UndefOp::new(ctx, llvm_ty);
    rewriter.insert_operation(ctx, undef.get_operation());
    let undef_val = undef.get_operation().deref(ctx).get_result(0);

    let with_disc = llvm::InsertValueOp::new(ctx, undef_val, disc, vec![0]);
    rewriter.insert_operation(ctx, with_disc.get_operation());
    let with_disc_val = with_disc.get_operation().deref(ctx).get_result(0);

    let mut deep_path = vec![1u32];
    let rest = deep_scalar_index_path(ctx, payload_ty, val_ty).ok_or_else(|| {
        pliron::input_error_noloc!(
            "niched-enum payload field has no scalar slot matching the source type"
        )
    })?;
    deep_path.extend(rest);

    let final_insert = llvm::InsertValueOp::new(ctx, with_disc_val, val, deep_path);
    Ok(final_insert.get_operation())
}

/// Aggregate -> scalar memory round-trip (e.g. `{ { i64 } }` -> `i64`).
///
/// The store-then-load only produces the right value when the destination
/// scalar's bit width is at least as wide as the deepest scalar reachable
/// inside the aggregate. If it is narrower we would silently truncate; if
/// the alloca holds padding bytes the load would pull them in. We bail
/// out loudly in those cases rather than emit a quiet miscompile.
fn emit_struct_to_scalar(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    val: pliron::value::Value,
    val_ty: Ptr<pliron::r#type::TypeObj>,
    llvm_ty: Ptr<pliron::r#type::TypeObj>,
) -> Result<Ptr<Operation>> {
    let dst_width = llvm_ty
        .deref(ctx)
        .downcast_ref::<IntegerType>()
        .map(|t| t.width())
        .ok_or_else(|| {
            pliron::input_error_noloc!("emit_struct_to_scalar: destination is not an integer type")
        })?;

    if let Some(src_width) = single_scalar_struct_width(ctx, val_ty) {
        if src_width != dst_width {
            return pliron::input_err_noloc!(
                "struct -> scalar Transmute size mismatch: source {} bits, destination {} bits. \
                 Refusing to fall through to a memory round-trip that would silently miscompile \
                 (see issue #21).",
                src_width,
                dst_width
            );
        }
    } else {
        return pliron::input_err_noloc!(
            "struct -> scalar Transmute over an aggregate that is not a single-scalar newtype \
             wrapper. The memory round-trip is not safe here without an explicit niche \
             reconstruction; refusing to emit it."
        );
    }

    let one = const_i64(ctx, rewriter, 1);
    let alloca = llvm::AllocaOp::new(ctx, val_ty, one);
    rewriter.insert_operation(ctx, alloca.get_operation());
    let ptr = alloca.get_operation().deref(ctx).get_result(0);
    let store = llvm::StoreOp::new(ctx, val, ptr);
    rewriter.insert_operation(ctx, store.get_operation());
    Ok(llvm::LoadOp::new(ctx, ptr, llvm_ty).get_operation())
}

/// Walks single-field struct wrappers (`{ { i64 } }`, `{ ptr }`, etc.)
/// and returns the bit width of the innermost scalar, or `None` if the
/// aggregate has any layer that is not a single-field wrapper.
fn single_scalar_struct_width(ctx: &Context, ty: Ptr<pliron::r#type::TypeObj>) -> Option<u32> {
    let mut current = ty;
    for _ in 0..MAX_NEWTYPE_DEPTH {
        let r = current.deref(ctx);
        if let Some(i) = r.downcast_ref::<IntegerType>() {
            return Some(i.width());
        }
        if let Some(p) = r.downcast_ref::<llvm_export::types::PointerType>() {
            // Pointers are addressed as opaque integer-width bit patterns
            // for the purpose of memory-round-trip sizing. CUDA targets
            // 64-bit pointers across address spaces.
            let _ = p;
            return Some(64);
        }
        let s = r.downcast_ref::<llvm_export::types::StructType>()?;
        if s.num_fields() != 1 {
            return None;
        }
        current = s.field_type(0);
    }
    None
}

/// Equal-size Transmute through memory: `alloca` a stack slot, `store` the
/// source value into it, then `load` it back as the destination type.
///
/// This is the only valid lowering when either side is an aggregate,
/// because LLVM's `bitcast` is restricted to non-aggregate first-class
/// types (an aggregate bitcast such as `bitcast [4 x i8] %v to i32` is
/// rejected by `llc` with "invalid cast opcode"). The `opt -O2` middle
/// end folds the round-trip away, so no real stack traffic survives.
///
/// Guarded by a total-byte-size equality check so a size-mismatched
/// transmute fails loudly at compile time instead of silently truncating
/// the source or loading bytes that were never stored.
///
/// The stack slot is aligned to the larger of the two types' ABI
/// alignments. For `[u8; 4]` → `u32` the byte array alone would give the
/// slot align 1, making the 4-byte integer load under-aligned; raising
/// the slot to align 4 keeps both accesses natural. The chosen alignment
/// is stamped explicitly on all three ops so the textual exporter does
/// not fall back to each type's own (possibly smaller) natural alignment.
fn emit_transmute_via_memory(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    val: pliron::value::Value,
    val_ty: Ptr<pliron::r#type::TypeObj>,
    llvm_ty: Ptr<pliron::r#type::TypeObj>,
) -> Result<Ptr<Operation>> {
    let Some(src_bytes) = type_byte_size(ctx, val_ty) else {
        return pliron::input_err_noloc!(
            "Transmute via memory round-trip: cannot compute the total size of source type {}. \
             Refusing to lower (see issue #125).",
            val_ty.disp(ctx)
        );
    };
    let Some(dst_bytes) = type_byte_size(ctx, llvm_ty) else {
        return pliron::input_err_noloc!(
            "Transmute via memory round-trip: cannot compute the total size of destination \
             type {}. Refusing to lower (see issue #125).",
            llvm_ty.disp(ctx)
        );
    };
    if src_bytes != dst_bytes {
        return pliron::input_err_noloc!(
            "aggregate Transmute size mismatch: source {} is {} bytes, destination {} is {} \
             bytes. Refusing the memory round-trip that would silently miscompile \
             (see issue #125).",
            val_ty.disp(ctx),
            src_bytes,
            llvm_ty.disp(ctx),
            dst_bytes
        );
    }

    // The slot must satisfy whichever side needs the stricter alignment.
    let align = abi_alignment_bytes(ctx, val_ty).max(abi_alignment_bytes(ctx, llvm_ty));

    let one = const_i64(ctx, rewriter, 1);
    let alloca = llvm::AllocaOp::new(ctx, val_ty, one);
    llvm_export::ops::set_op_alignment(ctx, alloca.get_operation(), align);
    rewriter.insert_operation(ctx, alloca.get_operation());
    let ptr = alloca.get_operation().deref(ctx).get_result(0);

    let store = llvm::StoreOp::new(ctx, val, ptr);
    llvm_export::ops::set_op_alignment(ctx, store.get_operation(), align);
    rewriter.insert_operation(ctx, store.get_operation());

    let load = llvm::LoadOp::new(ctx, ptr, llvm_ty);
    llvm_export::ops::set_op_alignment(ctx, load.get_operation(), align);
    Ok(load.get_operation())
}

/// Total size in bytes of an LLVM-dialect type, for transmute size
/// checking. Integer widths are rounded up to whole bytes because the
/// memory round-trip operates at byte granularity (an `i1` occupies one
/// byte in a stack slot).
///
/// Covers integers, floats, pointers (64-bit on our CUDA targets),
/// arrays (element size times length), and structs whose fields tile the
/// layout with no padding. Returns `None` when the size cannot be
/// computed confidently (a struct that needs padding, an opaque struct,
/// or an unknown type), so callers refuse the transmute loudly instead
/// of guessing.
fn type_byte_size(ctx: &Context, ty: Ptr<pliron::r#type::TypeObj>) -> Option<u64> {
    let r = ty.deref(ctx);
    if let Some(i) = r.downcast_ref::<IntegerType>() {
        return Some((i.width() as u64).div_ceil(8));
    }
    if let Some(f) = type_cast::<dyn FloatTypeInterface>(&**r) {
        return Some((f.get_semantics().bits as u64).div_ceil(8));
    }
    if r.is::<llvm_export::types::PointerType>() {
        // CUDA targets use 64-bit pointers in every address space we emit.
        return Some(8);
    }
    if let Some(a) = r.downcast_ref::<llvm_export::types::ArrayType>() {
        let elem_bytes = type_byte_size(ctx, a.elem_type())?;
        return elem_bytes.checked_mul(a.size());
    }
    if let Some(s) = r.downcast_ref::<llvm_export::types::StructType>() {
        if s.is_opaque() {
            return None;
        }
        // A struct's size is only trustworthy when the fields tile the
        // layout exactly: each field starts where the previous one ended
        // (no inter-field padding) and the total is a multiple of the
        // struct's own alignment (no tail padding). Anything else would
        // make the store/load round-trip move padding bytes around.
        let mut offset: u64 = 0;
        let mut max_align: u64 = 1;
        for field in s.fields() {
            let field_align = abi_alignment_bytes(ctx, field) as u64;
            if !offset.is_multiple_of(field_align) {
                return None; // inter-field padding required
            }
            offset += type_byte_size(ctx, field)?;
            max_align = max_align.max(field_align);
        }
        if !offset.is_multiple_of(max_align) {
            return None; // tail padding required
        }
        return Some(offset);
    }
    None
}

/// Conservative ABI alignment (bytes) of an LLVM-dialect type. Mirrors
/// the natural-alignment fallback in the textual `.ll` exporter so the
/// alignment stamped on a transmute stack slot agrees with what the
/// exporter assumes for direct loads/stores of the same type.
fn abi_alignment_bytes(ctx: &Context, ty: Ptr<pliron::r#type::TypeObj>) -> u32 {
    let r = ty.deref(ctx);
    if let Some(i) = r.downcast_ref::<IntegerType>() {
        return std::cmp::max(1, i.width() / 8);
    }
    if let Some(f) = type_cast::<dyn FloatTypeInterface>(&**r) {
        return std::cmp::max(1, (f.get_semantics().bits / 8) as u32);
    }
    if r.is::<llvm_export::types::PointerType>() {
        return 8;
    }
    if let Some(a) = r.downcast_ref::<llvm_export::types::ArrayType>() {
        // An array aligns like its element type.
        return abi_alignment_bytes(ctx, a.elem_type());
    }
    if let Some(s) = r.downcast_ref::<llvm_export::types::StructType>() {
        if s.is_opaque() {
            return 8;
        }
        // A struct aligns like its most-aligned field (1 if empty).
        return s
            .fields()
            .map(|f| abi_alignment_bytes(ctx, f))
            .max()
            .unwrap_or(1);
    }
    // Conservative fallback for unknown types.
    8
}

/// Float → float: extend or truncate precision.
fn convert_float_to_float(
    ctx: &mut Context,
    _rewriter: &mut DialectConversionRewriter,
    val: pliron::value::Value,
    llvm_ty: Ptr<pliron::r#type::TypeObj>,
    val_ty: Ptr<pliron::r#type::TypeObj>,
) -> Result<Ptr<Operation>> {
    let src_width = float_bit_width(ctx, val_ty)?;
    let dst_width = float_bit_width(ctx, llvm_ty)?;

    let flags_key: pliron::identifier::Identifier = "llvm_fast_math_flags".try_into().unwrap();
    let flags = llvm_export::attributes::FastmathFlagsAttr::default();

    if src_width < dst_width {
        let op = llvm::FPExtOp::new(ctx, val, llvm_ty);
        op.get_operation()
            .deref_mut(ctx)
            .attributes
            .0
            .insert(flags_key, flags.into());
        Ok(op.get_operation())
    } else if src_width > dst_width {
        let op = llvm::FPTruncOp::new(ctx, val, llvm_ty);
        op.get_operation()
            .deref_mut(ctx)
            .attributes
            .0
            .insert(flags_key, flags.into());
        Ok(op.get_operation())
    } else {
        Ok(llvm::BitcastOp::new(ctx, val, llvm_ty).get_operation())
    }
}

fn float_bit_width(ctx: &Context, ty: Ptr<pliron::r#type::TypeObj>) -> Result<usize> {
    let ty_ref = ty.deref(ctx);
    let Some(float_ty) = type_cast::<dyn FloatTypeInterface>(&**ty_ref) else {
        return pliron::input_err_noloc!("expected floating-point type");
    };
    Ok(float_ty.get_semantics().bits)
}

#[cfg(test)]
mod tests {
    // TODO (npasham): Add unit tests for cast conversion
}
