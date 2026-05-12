/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cast operation conversion: `dialect-mir` → `dialect-llvm`.
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
//! ## `emit_pointer_cast` handles struct↔ptr (fat/thin pointer) conversions:
//! | Source → Dest                  | LLVM Operation                    |
//! |--------------------------------|-----------------------------------|
//! | struct → ptr (fat→thin)        | `extractvalue` field 0            |
//! | ptr → struct (thin→fat)        | `insertvalue` into undef          |
//! | ptr → integer                  | `ptrtoint`                        |
//! | integer → ptr                  | `inttoptr`                        |
//! | struct → struct (transmute)    | `alloca` + `store` + `load`       |
//! | array ↔ scalar / array (xmute) | `alloca` + `store` + `load`       |
//! | ptr → ptr (diff addrspace)     | `addrspacecast`                   |
//! | otherwise                      | `bitcast`                         |

use crate::convert::types::convert_type;
use crate::helpers;
use dialect_llvm::op_interfaces::CastOpInterface;
use dialect_llvm::ops as llvm;
use dialect_llvm::types::FuncType;
use dialect_mir::attributes::MirCastKindAttr;
use dialect_mir::ops::MirCastOp;
use dialect_mir::types::{MirArrayType, MirPtrType};
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
        MirCastKindAttr::Transmute => emit_pointer_cast(ctx, rewriter, val, val_ty, llvm_ty)?,

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
            emit_unsize_cast(ctx, rewriter, val, val_ty, llvm_ty, mir_opd_ty)?
        }

        MirCastKindAttr::PtrToPtr
        | MirCastKindAttr::FnPtrToPtr
        | MirCastKindAttr::PointerCoercionMutToConst
        | MirCastKindAttr::PointerCoercionReifyFnPointer
        | MirCastKindAttr::PointerCoercionUnsafeFnPointer
        | MirCastKindAttr::PointerCoercionClosureFnPointer
        | MirCastKindAttr::PointerCoercionArrayToPointer
        | MirCastKindAttr::Subtype => emit_pointer_cast(ctx, rewriter, val, val_ty, llvm_ty)?,

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
    val: pliron::value::Value,
    val_ty: Ptr<pliron::r#type::TypeObj>,
    llvm_ty: Ptr<pliron::r#type::TypeObj>,
    mir_opd_ty: Ptr<pliron::r#type::TypeObj>,
) -> Result<Ptr<Operation>> {
    let array_len = {
        let mir_ref = mir_opd_ty.deref(ctx);
        mir_ref.downcast_ref::<MirPtrType>().and_then(|ptr_ty| {
            ptr_ty
                .pointee
                .deref(ctx)
                .downcast_ref::<MirArrayType>()
                .map(|arr| arr.size())
        })
    };

    if let Some(len) = array_len {
        let dst_is_struct = llvm_ty.deref(ctx).is::<dialect_llvm::types::StructType>();

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

    emit_pointer_cast(ctx, rewriter, val, val_ty, llvm_ty)
}

/// Emit a pointer-compatible cast, handling the struct↔ptr patterns that arise
/// because our type system represents fat pointers (slices) as `{ ptr, i64 }` structs.
///
/// LLVM does not allow `bitcast` between structs and scalars/pointers, so:
/// - struct → ptr: `extractvalue` field 0 (extract data pointer from fat pointer)
/// - ptr → struct: `insertvalue` into undef at field 0 (wrap thin ptr in fat pointer)
/// - ptr → ptr (different address space): `addrspacecast`
/// - otherwise: `bitcast`
fn emit_pointer_cast(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    val: pliron::value::Value,
    val_ty: Ptr<pliron::r#type::TypeObj>,
    llvm_ty: Ptr<pliron::r#type::TypeObj>,
) -> Result<Ptr<Operation>> {
    let src_is_struct = val_ty.deref(ctx).is::<dialect_llvm::types::StructType>();
    let dst_is_struct = llvm_ty.deref(ctx).is::<dialect_llvm::types::StructType>();
    let src_is_array = val_ty.deref(ctx).is::<dialect_llvm::types::ArrayType>();
    let dst_is_array = llvm_ty.deref(ctx).is::<dialect_llvm::types::ArrayType>();
    let src_as = val_ty
        .deref(ctx)
        .downcast_ref::<dialect_llvm::types::PointerType>()
        .map(|pt| pt.address_space());
    let dst_as = llvm_ty
        .deref(ctx)
        .downcast_ref::<dialect_llvm::types::PointerType>()
        .map(|pt| pt.address_space());
    let dst_is_ptr = dst_as.is_some();
    let src_is_ptr = src_as.is_some();

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
    } else if val_ty.deref(ctx).is::<IntegerType>() && dst_is_ptr {
        Ok(llvm::IntToPtrOp::new(ctx, val, llvm_ty).get_operation())
    } else if val_ty.deref(ctx).is::<IntegerType>() && dst_is_struct {
        // Integer -> struct: LLVM forbids a direct `bitcast`. This shape
        // arises on panic-edge `core::fmt::Arguments` transmute paths
        // where rustc reinterprets a tag-encoded `usize` as a 1-field
        // `{ ptr }` struct (the Argument value slot). The panic block
        // terminates in `unreachable` so the bytes are never observed.
        //
        // When the struct's first field is a pointer, lower to
        // `inttoptr` + `insertvalue undef, ptr, 0`, which matches the
        // existing `ptr → struct` arm above. Otherwise (non-pointer
        // first field, multi-field structs we can't model directly),
        // fall back to a memory round-trip: alloca {val} typed, store
        // the int, load as the struct type.
        let first_field_is_ptr = {
            let dst_ref = llvm_ty.deref(ctx);
            let st = dst_ref
                .downcast_ref::<dialect_llvm::types::StructType>()
                .unwrap();
            st.fields()
                .next()
                .map(|f| f.deref(ctx).is::<dialect_llvm::types::PointerType>())
                .unwrap_or(false)
        };

        if first_field_is_ptr {
            let first_field_ty = {
                let dst_ref = llvm_ty.deref(ctx);
                let st = dst_ref
                    .downcast_ref::<dialect_llvm::types::StructType>()
                    .unwrap();
                st.fields().next().unwrap()
            };
            let i2p = llvm::IntToPtrOp::new(ctx, val, first_field_ty);
            rewriter.insert_operation(ctx, i2p.get_operation());
            let ptr_val = i2p.get_operation().deref(ctx).get_result(0);

            let undef = llvm::UndefOp::new(ctx, llvm_ty);
            rewriter.insert_operation(ctx, undef.get_operation());
            let undef_val = undef.get_operation().deref(ctx).get_result(0);
            Ok(llvm::InsertValueOp::new(ctx, undef_val, ptr_val, vec![0]).get_operation())
        } else {
            // Memory round-trip: alloca with the dst struct type, store
            // an integer-typed value into it (LLVM accepts mixed-type
            // store-then-load for same-size types), reload as the struct.
            let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
            let one = {
                let apint = pliron::utils::apint::APInt::from_i64(
                    1,
                    std::num::NonZeroUsize::new(64).unwrap(),
                );
                let attr = pliron::builtin::attributes::IntegerAttr::new(i64_ty, apint);
                let c = llvm::ConstantOp::new(ctx, attr.into());
                rewriter.insert_operation(ctx, c.get_operation());
                c.get_operation().deref(ctx).get_result(0)
            };
            let alloca = llvm::AllocaOp::new(ctx, llvm_ty, one);
            rewriter.insert_operation(ctx, alloca.get_operation());
            let ptr = alloca.get_operation().deref(ctx).get_result(0);

            let store = llvm::StoreOp::new(ctx, val, ptr);
            rewriter.insert_operation(ctx, store.get_operation());

            Ok(llvm::LoadOp::new(ctx, ptr, llvm_ty).get_operation())
        }
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
            Ok(llvm::AddrSpaceCastOp::new(ctx, val, d).get_operation())
        } else {
            Ok(llvm::BitcastOp::new(ctx, val, llvm_ty).get_operation())
        }
    } else if src_is_array || dst_is_array {
        // LLVM forbids `bitcast` involving aggregate `[N x T]` types — same
        // constraint that ruled out struct↔scalar bitcasts above. This shape
        // arises on `u32::from_be_bytes` / `from_le_bytes` and the SHA-256
        // message-schedule byte-shuffle, where MIR emits
        // `Cast(Transmute) : [i8; 4] → u32`. Go through memory: alloca the
        // source type, store the source value, load typed as the destination.
        // Sizes are guaranteed to match by transmute's invariants.
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
    } else {
        Ok(llvm::BitcastOp::new(ctx, val, llvm_ty).get_operation())
    }
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
    let flags = dialect_llvm::attributes::FastmathFlagsAttr::default();

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
