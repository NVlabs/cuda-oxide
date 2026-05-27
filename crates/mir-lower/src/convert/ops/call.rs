/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Call operation conversion: `dialect-mir` → `dialect-llvm`.
//!
//! Handles function call lowering with ABI-level transformations:
//! - Slice arguments flattened to (ptr, len) pairs
//! - Struct arguments flattened to individual fields
//! - Unit return type becomes void
//! - Pointer arguments cast to generic address space (for ABI compatibility)
//!
//! These transformations match the function signature flattening done in
//! `convert_function_type`.
//!
//! # Device Extern Symbol Resolution
//!
//! When calling device extern functions (declared with `#[device] extern "C"`),
//! the MIR contains calls to prefixed symbols like `cuda_oxide_device_extern_foo`.
//! This prefix is added by the proc-macro for internal detection. However, the
//! external LTOIR (e.g., CCCL libraries) exports the original symbol name `foo`.
//!
//! We strip the prefix during lowering so the LLVM IR emits:
//! ```llvm
//! call @foo(...)  ; NOT @cuda_oxide_device_extern_foo
//! ```
//!
//! This allows nvJitLink to resolve the symbol against the external LTOIR.
//!
//! # Address Space Handling
//!
//! The call op's `func_type` must match the callee's declared signature
//! exactly, including the address space of every pointer parameter. We
//! achieve this by:
//!
//!   1. Looking up the callee's `llvm::FuncOp` declaration in the parent
//!      module, before flattening the arguments.
//!   2. Coercing each (post-flatten) argument to the corresponding declared
//!      parameter type — when the argument is a pointer in a different
//!      address space than the parameter, we insert `llvm.addrspacecast`
//!      to bridge them.
//!   3. Building the call op's `func_type` directly from the looked-up
//!      declaration, so a tightening of the declaration automatically
//!      flows to every call site.
//!
//! This single mechanism handles both directions of the addrspace dance:
//!
//!   * Path 1 (e.g. `block_reduce(*mut SharedArray<T,N>)`): callee param
//!     is `ptr addrspace(3)`, caller has `ptr addrspace(3)` → no cast.
//!     A caller that legitimately had a generic pointer would be cast UP
//!     to addrspace(3).
//!   * Path 3 (e.g. `extern "C" fn cublasdx_gemm(_: *mut i8)`): callee
//!     param is `ptr addrspace(0)`, caller has `ptr addrspace(3)` (from
//!     `DynamicSharedArray::get`) → cast DOWN to addrspace(0).
//!
//! The previous implementation always cast pointer arguments to addrspace(0)
//! regardless of the callee's declared signature. That worked for Path 3
//! but actively broke Path 1, because the resulting call op carried a
//! `func_type` whose pointer parameter types were `addrspace(0)` while the
//! callee declaration carried `addrspace(3)`. The verifier rejected the
//! mismatch.

use crate::convert::types::{
    convert_function_type, convert_type, is_kernel_func, is_zero_sized_type,
};
use crate::helpers;
use dialect_llvm::op_interfaces::CastOpInterface;
use dialect_llvm::ops as llvm;
use dialect_llvm::types as llvm_types;
use dialect_mir::ops::{MirCallOp, MirFuncOp};
use dialect_mir::rust_intrinsics;
use dialect_mir::types::{MirDisjointSliceType, MirSliceType, MirStructType, MirTupleType};
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::op_interfaces::{CallOpCallable, SymbolOpInterface};
use pliron::builtin::type_interfaces::FunctionTypeInterface;
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::linked_list::ContainsLinkedList;
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeObj, Typed};
use pliron::utils::apint::APInt;
use pliron::value::Value;
use std::num::NonZeroUsize;

/// Generic address space (can alias any memory).
const ADDRSPACE_GENERIC: u32 = 0;

// The `#[device] extern "C"` macro renames a foreign function `foo` to
// `cuda_oxide_device_extern_<hash>_foo` so the collector can find it. During
// MIR lowering we strip that prefix back off so the LLVM IR / LTOIR refers to
// the original symbol the user wrote. The prefix string itself, plus the
// `device_extern_base_name` extractor, lives in `reserved-oxide-symbols`.
use reserved_oxide_symbols::device_extern_base_name;

/// Internal placeholder for rustc bit intrinsics that need LLVM intrinsic calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RustBitIntrinsic {
    RotateLeft,
    RotateRight,
    Ctpop,
    Ctlz { zero_undef: bool },
    Cttz { zero_undef: bool },
    Bswap,
    Bitreverse,
}

impl RustBitIntrinsic {
    /// Convert an importer placeholder name back into the intrinsic it represents.
    fn from_placeholder_callee(callee: &str) -> Option<Self> {
        match callee {
            rust_intrinsics::CALLEE_ROTATE_LEFT => Some(Self::RotateLeft),
            rust_intrinsics::CALLEE_ROTATE_RIGHT => Some(Self::RotateRight),
            rust_intrinsics::CALLEE_CTPOP => Some(Self::Ctpop),
            rust_intrinsics::CALLEE_CTLZ => Some(Self::Ctlz { zero_undef: false }),
            rust_intrinsics::CALLEE_CTLZ_NONZERO => Some(Self::Ctlz { zero_undef: true }),
            rust_intrinsics::CALLEE_CTTZ => Some(Self::Cttz { zero_undef: false }),
            rust_intrinsics::CALLEE_CTTZ_NONZERO => Some(Self::Cttz { zero_undef: true }),
            rust_intrinsics::CALLEE_BSWAP => Some(Self::Bswap),
            rust_intrinsics::CALLEE_BITREVERSE => Some(Self::Bitreverse),
            _ => None,
        }
    }
}

/// Internal placeholder for rustc saturating arithmetic intrinsics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RustSaturatingIntrinsic {
    Add,
    Sub,
}

impl RustSaturatingIntrinsic {
    /// Convert an importer placeholder name back into the intrinsic it represents.
    fn from_placeholder_callee(callee: &str) -> Option<Self> {
        match callee {
            rust_intrinsics::CALLEE_SATURATING_ADD => Some(Self::Add),
            rust_intrinsics::CALLEE_SATURATING_SUB => Some(Self::Sub),
            _ => None,
        }
    }
}

/// Internal placeholder for rustc float math intrinsics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RustFloatMathIntrinsic {
    SqrtF32,
    SqrtF64,
    PowiF32,
    PowiF64,
    SinF32,
    SinF64,
    CosF32,
    CosF64,
    TanF32,
    TanF64,
    PowfF32,
    PowfF64,
    ExpF32,
    ExpF64,
    Exp2F32,
    Exp2F64,
    LogF32,
    LogF64,
    Log2F32,
    Log2F64,
    Log10F32,
    Log10F64,
    FmaF32,
    FmaF64,
    FmuladdF32,
    FmuladdF64,
    FloorF32,
    FloorF64,
    CeilF32,
    CeilF64,
    TruncF32,
    TruncF64,
    RoundF32,
    RoundF64,
    RoundevenF32,
    RoundevenF64,
    Fabs,
    CopysignF32,
    CopysignF64,
}

impl RustFloatMathIntrinsic {
    /// Convert an importer placeholder name back into the intrinsic it represents.
    fn from_placeholder_callee(callee: &str) -> Option<Self> {
        match callee {
            rust_intrinsics::CALLEE_SQRT_F32 => Some(Self::SqrtF32),
            rust_intrinsics::CALLEE_SQRT_F64 => Some(Self::SqrtF64),
            rust_intrinsics::CALLEE_POWI_F32 => Some(Self::PowiF32),
            rust_intrinsics::CALLEE_POWI_F64 => Some(Self::PowiF64),
            rust_intrinsics::CALLEE_SIN_F32 => Some(Self::SinF32),
            rust_intrinsics::CALLEE_SIN_F64 => Some(Self::SinF64),
            rust_intrinsics::CALLEE_COS_F32 => Some(Self::CosF32),
            rust_intrinsics::CALLEE_COS_F64 => Some(Self::CosF64),
            rust_intrinsics::CALLEE_TAN_F32 => Some(Self::TanF32),
            rust_intrinsics::CALLEE_TAN_F64 => Some(Self::TanF64),
            rust_intrinsics::CALLEE_POWF_F32 => Some(Self::PowfF32),
            rust_intrinsics::CALLEE_POWF_F64 => Some(Self::PowfF64),
            rust_intrinsics::CALLEE_EXP_F32 => Some(Self::ExpF32),
            rust_intrinsics::CALLEE_EXP_F64 => Some(Self::ExpF64),
            rust_intrinsics::CALLEE_EXP2_F32 => Some(Self::Exp2F32),
            rust_intrinsics::CALLEE_EXP2_F64 => Some(Self::Exp2F64),
            rust_intrinsics::CALLEE_LOG_F32 => Some(Self::LogF32),
            rust_intrinsics::CALLEE_LOG_F64 => Some(Self::LogF64),
            rust_intrinsics::CALLEE_LOG2_F32 => Some(Self::Log2F32),
            rust_intrinsics::CALLEE_LOG2_F64 => Some(Self::Log2F64),
            rust_intrinsics::CALLEE_LOG10_F32 => Some(Self::Log10F32),
            rust_intrinsics::CALLEE_LOG10_F64 => Some(Self::Log10F64),
            rust_intrinsics::CALLEE_FMA_F32 => Some(Self::FmaF32),
            rust_intrinsics::CALLEE_FMA_F64 => Some(Self::FmaF64),
            rust_intrinsics::CALLEE_FMULADD_F32 => Some(Self::FmuladdF32),
            rust_intrinsics::CALLEE_FMULADD_F64 => Some(Self::FmuladdF64),
            rust_intrinsics::CALLEE_FLOOR_F32 => Some(Self::FloorF32),
            rust_intrinsics::CALLEE_FLOOR_F64 => Some(Self::FloorF64),
            rust_intrinsics::CALLEE_CEIL_F32 => Some(Self::CeilF32),
            rust_intrinsics::CALLEE_CEIL_F64 => Some(Self::CeilF64),
            rust_intrinsics::CALLEE_TRUNC_F32 => Some(Self::TruncF32),
            rust_intrinsics::CALLEE_TRUNC_F64 => Some(Self::TruncF64),
            rust_intrinsics::CALLEE_ROUND_F32 => Some(Self::RoundF32),
            rust_intrinsics::CALLEE_ROUND_F64 => Some(Self::RoundF64),
            rust_intrinsics::CALLEE_ROUNDEVEN_F32 => Some(Self::RoundevenF32),
            rust_intrinsics::CALLEE_ROUNDEVEN_F64 => Some(Self::RoundevenF64),
            rust_intrinsics::CALLEE_FABS => Some(Self::Fabs),
            rust_intrinsics::CALLEE_COPYSIGN_F32 => Some(Self::CopysignF32),
            rust_intrinsics::CALLEE_COPYSIGN_F64 => Some(Self::CopysignF64),
            _ => None,
        }
    }

    /// CUDA libdevice function name for this Rust math intrinsic.
    fn libdevice_name(
        self,
        ctx: &Context,
        result_ty: Ptr<TypeObj>,
        loc: pliron::location::Location,
    ) -> Result<&'static str> {
        match self {
            Self::SqrtF32 => Ok("__nv_sqrtf"),
            Self::SqrtF64 => Ok("__nv_sqrt"),
            Self::PowiF32 => Ok("__nv_powif"),
            Self::PowiF64 => Ok("__nv_powi"),
            Self::SinF32 => Ok("__nv_sinf"),
            Self::SinF64 => Ok("__nv_sin"),
            Self::CosF32 => Ok("__nv_cosf"),
            Self::CosF64 => Ok("__nv_cos"),
            Self::TanF32 => Ok("__nv_tanf"),
            Self::TanF64 => Ok("__nv_tan"),
            Self::PowfF32 => Ok("__nv_powf"),
            Self::PowfF64 => Ok("__nv_pow"),
            Self::ExpF32 => Ok("__nv_expf"),
            Self::ExpF64 => Ok("__nv_exp"),
            Self::Exp2F32 => Ok("__nv_exp2f"),
            Self::Exp2F64 => Ok("__nv_exp2"),
            Self::LogF32 => Ok("__nv_logf"),
            Self::LogF64 => Ok("__nv_log"),
            Self::Log2F32 => Ok("__nv_log2f"),
            Self::Log2F64 => Ok("__nv_log2"),
            Self::Log10F32 => Ok("__nv_log10f"),
            Self::Log10F64 => Ok("__nv_log10"),
            Self::FmaF32 | Self::FmuladdF32 => Ok("__nv_fmaf"),
            Self::FmaF64 | Self::FmuladdF64 => Ok("__nv_fma"),
            Self::FloorF32 => Ok("__nv_floorf"),
            Self::FloorF64 => Ok("__nv_floor"),
            Self::CeilF32 => Ok("__nv_ceilf"),
            Self::CeilF64 => Ok("__nv_ceil"),
            Self::TruncF32 => Ok("__nv_truncf"),
            Self::TruncF64 => Ok("__nv_trunc"),
            Self::RoundF32 => Ok("__nv_roundf"),
            Self::RoundF64 => Ok("__nv_round"),
            Self::RoundevenF32 => Ok("__nv_rintf"),
            Self::RoundevenF64 => Ok("__nv_rint"),
            Self::Fabs => fabs_libdevice_name(ctx, result_ty, loc),
            Self::CopysignF32 => Ok("__nv_copysignf"),
            Self::CopysignF64 => Ok("__nv_copysign"),
        }
    }

    /// Number of operands expected by the libdevice function.
    fn arg_count(self) -> usize {
        match self {
            Self::PowiF32
            | Self::PowiF64
            | Self::PowfF32
            | Self::PowfF64
            | Self::CopysignF32
            | Self::CopysignF64 => 2,
            Self::FmaF32 | Self::FmaF64 | Self::FmuladdF32 | Self::FmuladdF64 => 3,
            _ => 1,
        }
    }
}

fn anyhow_to_pliron(e: anyhow::Error) -> pliron::result::Error {
    pliron::create_error!(
        pliron::location::Location::Unknown,
        pliron::result::ErrorKind::VerificationFailed,
        pliron::result::StringError(e.to_string())
    )
}

/// Convert `mir.call` to `llvm.call` with argument flattening.
///
/// Performs ABI-level transformations to match CUDA calling conventions:
/// - Slice arguments: flattened to `(ptr, len)` pairs
/// - Struct arguments: flattened to individual fields
/// - Unit return type: converted to void
/// - Callee name: `::` mangled to `__` for LLVM identifier validity
/// - Device extern calls: prefix stripped to use original symbol name
pub fn convert(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
) -> Result<()> {
    let callee_name: String = {
        let mir_call = MirCallOp::new(op);
        let callee_attr = match mir_call.get_attr_callee(ctx) {
            Some(a) => a,
            None => {
                return pliron::input_err!(
                    op.deref(ctx).loc(),
                    "MirCallOp missing callee attribute"
                );
            }
        };
        (*callee_attr).clone().into()
    };

    if let Some(intrinsic) = RustBitIntrinsic::from_placeholder_callee(&callee_name) {
        return convert_rust_bit_intrinsic(ctx, rewriter, op, intrinsic);
    }

    if let Some(intrinsic) = RustSaturatingIntrinsic::from_placeholder_callee(&callee_name) {
        return convert_rust_saturating_intrinsic(ctx, rewriter, op, operands_info, intrinsic);
    }

    if let Some(intrinsic) = RustFloatMathIntrinsic::from_placeholder_callee(&callee_name) {
        return convert_rust_float_math_intrinsic(ctx, rewriter, op, intrinsic);
    }

    let callee_ident: pliron::identifier::Identifier = {
        let resolved_name = resolve_device_extern_symbol(&callee_name);

        resolved_name
            .try_into()
            .expect("callee name should have been legalized during MIR import")
    };

    let args: Vec<Value> = op.deref(ctx).operands().collect();

    let has_result = op.deref(ctx).get_num_results() > 0;
    let mir_result_ty_ptr = if has_result {
        Some(op.deref(ctx).get_result(0).get_type(ctx))
    } else {
        None
    };

    let result_type = if let Some(mir_ty) = mir_result_ty_ptr {
        let is_unit = mir_ty.deref(ctx).is::<MirTupleType>();
        if is_unit {
            llvm_types::VoidType::get(ctx).into()
        } else {
            convert_type(ctx, mir_ty).map_err(anyhow_to_pliron)?
        }
    } else {
        llvm_types::VoidType::get(ctx).into()
    };

    // Look up the callee's declared signature in the parent module. The
    // declaration may already have been lowered to `llvm::FuncOp` (typical
    // for device-extern decls inserted by the importer ahead of time, and
    // for callees whose conversion ran first), or it may still be a
    // `MirFuncOp` whose body hasn't been touched yet. In the second case
    // we run the same MIR-to-LLVM signature flattening that the function
    // converter will eventually apply, so the call site sees the exact
    // same parameter types the callee will be lowered to. This makes
    // intra-Rust calls into shared-memory-typed params and device-extern
    // calls with the generic ABI work uniformly.
    let callee_decl_arg_types = find_callee_arg_types(ctx, op, &callee_ident).unwrap_or_default();
    let expected_param_tys = if callee_decl_arg_types.is_empty() {
        None
    } else {
        Some(callee_decl_arg_types.as_slice())
    };

    let (flattened_args, flattened_arg_types) =
        flatten_arguments(ctx, rewriter, &args, operands_info, expected_param_tys)?;

    let func_type = llvm_types::FuncType::get(ctx, result_type, flattened_arg_types, false);
    let llvm_call = llvm::CallOp::new(
        ctx,
        CallOpCallable::Direct(callee_ident),
        func_type,
        flattened_args,
    );
    rewriter.insert_operation(ctx, llvm_call.get_operation());

    let is_void = result_type.deref(ctx).is::<llvm_types::VoidType>();
    if has_result && !is_void && llvm_call.get_operation().deref(ctx).get_num_results() > 0 {
        rewriter.replace_operation(ctx, op, llvm_call.get_operation());
    } else {
        rewriter.erase_operation(ctx, op);
    }

    Ok(())
}

/// Lower placeholder calls for rustc's integer bit intrinsics to LLVM intrinsics.
///
/// Rust methods like `u128::rotate_left` call `core::intrinsics::rotate_left`
/// in libcore. The importer preserves that as a placeholder `mir.call`; here we
/// recover the concrete integer width and emit the corresponding overloaded
/// LLVM intrinsic.
fn convert_rust_bit_intrinsic(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    intrinsic: RustBitIntrinsic,
) -> Result<()> {
    let loc = op.deref(ctx).loc();
    if op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(loc, "Rust bit intrinsic call must have one result");
    }

    let args: Vec<Value> = op.deref(ctx).operands().collect();
    let Some(&value) = args.first() else {
        return pliron::input_err!(loc, "Rust bit intrinsic call missing integer operand");
    };

    let value_ty = value.get_type(ctx);
    let value_width = integer_bit_width(ctx, value_ty, loc.clone())?;
    let mir_result_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let result_type = convert_type(ctx, mir_result_ty).map_err(anyhow_to_pliron)?;

    if matches!(intrinsic, RustBitIntrinsic::Bswap) && value_width == 8 {
        // LLVM has no useful byte swap for a single byte; Rust's semantics are identity.
        let bitcast = llvm::BitcastOp::new(ctx, value, result_type);
        rewriter.insert_operation(ctx, bitcast.get_operation());
        rewriter.replace_operation(ctx, op, bitcast.get_operation());
        return Ok(());
    }

    let (intrinsic_name, intrinsic_args, intrinsic_result_ty) = match intrinsic {
        RustBitIntrinsic::RotateLeft | RustBitIntrinsic::RotateRight => {
            if args.len() != 2 {
                return pliron::input_err!(
                    loc,
                    "rotate intrinsic requires value and shift operands"
                );
            }
            let (shift, _) =
                cast_integer_value_to_type(ctx, rewriter, args[1], value_ty, loc.clone())?;
            let suffix = match intrinsic {
                RustBitIntrinsic::RotateLeft => "fshl",
                RustBitIntrinsic::RotateRight => "fshr",
                _ => unreachable!(),
            };
            (
                format!("llvm_{suffix}_i{value_width}"),
                vec![value, value, shift],
                value_ty,
            )
        }
        RustBitIntrinsic::Ctpop => (format!("llvm_ctpop_i{value_width}"), vec![value], value_ty),
        RustBitIntrinsic::Ctlz { zero_undef } => {
            let zero_undef = create_i1_constant(ctx, rewriter, zero_undef);
            (
                format!("llvm_ctlz_i{value_width}"),
                vec![value, zero_undef],
                value_ty,
            )
        }
        RustBitIntrinsic::Cttz { zero_undef } => {
            let zero_undef = create_i1_constant(ctx, rewriter, zero_undef);
            (
                format!("llvm_cttz_i{value_width}"),
                vec![value, zero_undef],
                value_ty,
            )
        }
        RustBitIntrinsic::Bswap => (format!("llvm_bswap_i{value_width}"), vec![value], value_ty),
        RustBitIntrinsic::Bitreverse => (
            format!("llvm_bitreverse_i{value_width}"),
            vec![value],
            value_ty,
        ),
    };

    let arg_types = intrinsic_args
        .iter()
        .map(|arg| arg.get_type(ctx))
        .collect::<Vec<_>>();
    let func_ty = llvm_types::FuncType::get(ctx, intrinsic_result_ty, arg_types, false);
    let parent_block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(loc.clone(), "Rust bit intrinsic call has no parent block")
    })?;
    helpers::ensure_intrinsic_declared(ctx, parent_block, &intrinsic_name, func_ty)
        .map_err(|e| pliron::input_error!(loc.clone(), "Failed to declare intrinsic: {e}"))?;

    let sym_name: pliron::identifier::Identifier = intrinsic_name
        .as_str()
        .try_into()
        .map_err(|e| pliron::input_error!(loc.clone(), "Invalid intrinsic name: {:?}", e))?;
    let llvm_call = llvm::CallOp::new(
        ctx,
        CallOpCallable::Direct(sym_name),
        func_ty,
        intrinsic_args,
    );
    rewriter.insert_operation(ctx, llvm_call.get_operation());

    let call_result = llvm_call.get_operation().deref(ctx).get_result(0);
    let (_, final_op) =
        cast_integer_value_to_type(ctx, rewriter, call_result, result_type, loc.clone())?;
    let replacement = final_op.unwrap_or_else(|| llvm_call.get_operation());
    rewriter.replace_operation(ctx, op, replacement);

    Ok(())
}

/// Lower placeholder calls for rustc's saturating integer intrinsics.
///
/// Rust preserves signedness in the original MIR type. The converted LLVM value
/// is signless, so this uses `operands_info` to choose `sadd/ssub` versus
/// `uadd/usub`.
fn convert_rust_saturating_intrinsic(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    operands_info: &OperandsInfo,
    intrinsic: RustSaturatingIntrinsic,
) -> Result<()> {
    let loc = op.deref(ctx).loc();
    if op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(loc, "Rust saturating intrinsic call must have one result");
    }

    let args: Vec<Value> = op.deref(ctx).operands().collect();
    if args.len() != 2 {
        return pliron::input_err!(
            loc,
            "Rust saturating intrinsic requires left and right operands"
        );
    }

    let lhs = args[0];
    let rhs = args[1];
    let lhs_ty = lhs.get_type(ctx);
    let width = integer_bit_width(ctx, lhs_ty, loc.clone())?;
    let is_signed =
        if let Some(int_ty) = operands_info.lookup_most_recent_of_type::<IntegerType>(ctx, lhs) {
            int_ty.signedness() == Signedness::Signed
        } else {
            return pliron::input_err!(loc, "expected integer type for Rust saturating intrinsic");
        };

    let (rhs, _) = cast_integer_value_to_type(ctx, rewriter, rhs, lhs_ty, loc.clone())?;
    let op_stem = match (is_signed, intrinsic) {
        (true, RustSaturatingIntrinsic::Add) => "sadd",
        (false, RustSaturatingIntrinsic::Add) => "uadd",
        (true, RustSaturatingIntrinsic::Sub) => "ssub",
        (false, RustSaturatingIntrinsic::Sub) => "usub",
    };
    let intrinsic_name = format!("llvm_{op_stem}_sat_i{width}");
    let func_ty = llvm_types::FuncType::get(ctx, lhs_ty, vec![lhs_ty, lhs_ty], false);
    let parent_block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(
            loc.clone(),
            "Rust saturating intrinsic call has no parent block"
        )
    })?;
    helpers::ensure_intrinsic_declared(ctx, parent_block, &intrinsic_name, func_ty)
        .map_err(|e| pliron::input_error!(loc.clone(), "Failed to declare intrinsic: {e}"))?;

    let sym_name: pliron::identifier::Identifier = intrinsic_name
        .as_str()
        .try_into()
        .map_err(|e| pliron::input_error!(loc.clone(), "Invalid intrinsic name: {:?}", e))?;
    let llvm_call = llvm::CallOp::new(
        ctx,
        CallOpCallable::Direct(sym_name),
        func_ty,
        vec![lhs, rhs],
    );
    rewriter.insert_operation(ctx, llvm_call.get_operation());

    let result_mir_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let result_ty = convert_type(ctx, result_mir_ty).map_err(anyhow_to_pliron)?;
    let call_result = llvm_call.get_operation().deref(ctx).get_result(0);
    let (_, final_op) =
        cast_integer_value_to_type(ctx, rewriter, call_result, result_ty, loc.clone())?;
    let replacement = final_op.unwrap_or_else(|| llvm_call.get_operation());
    rewriter.replace_operation(ctx, op, replacement);

    Ok(())
}

/// Lower placeholder calls for rustc's `f32` / `f64` math intrinsics to libdevice.
fn convert_rust_float_math_intrinsic(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    intrinsic: RustFloatMathIntrinsic,
) -> Result<()> {
    let loc = op.deref(ctx).loc();
    if op.deref(ctx).get_num_results() != 1 {
        return pliron::input_err!(loc, "Rust float math intrinsic call must have one result");
    }

    let args: Vec<Value> = op.deref(ctx).operands().collect();
    let expected_args = intrinsic.arg_count();
    if args.len() != expected_args {
        return pliron::input_err!(
            loc,
            "Rust float math intrinsic requires {expected_args} operand(s)"
        );
    }

    let result_mir_ty = op.deref(ctx).get_result(0).get_type(ctx);
    let result_ty = convert_type(ctx, result_mir_ty).map_err(anyhow_to_pliron)?;
    let intrinsic_name = intrinsic.libdevice_name(ctx, result_ty, loc.clone())?;
    let arg_types = args.iter().map(|arg| arg.get_type(ctx)).collect::<Vec<_>>();
    let func_ty = llvm_types::FuncType::get(ctx, result_ty, arg_types, false);
    let parent_block = op.deref(ctx).get_parent_block().ok_or_else(|| {
        pliron::input_error!(
            loc.clone(),
            "Rust float math intrinsic call has no parent block"
        )
    })?;
    helpers::ensure_intrinsic_declared(ctx, parent_block, intrinsic_name, func_ty)
        .map_err(|e| pliron::input_error!(loc.clone(), "Failed to declare intrinsic: {e}"))?;

    let sym_name: pliron::identifier::Identifier = intrinsic_name
        .try_into()
        .map_err(|e| pliron::input_error!(loc.clone(), "Invalid intrinsic name: {:?}", e))?;
    let llvm_call = llvm::CallOp::new(ctx, CallOpCallable::Direct(sym_name), func_ty, args);
    rewriter.insert_operation(ctx, llvm_call.get_operation());
    rewriter.replace_operation(ctx, op, llvm_call.get_operation());

    Ok(())
}

/// Read the width from an integer type, or report a useful lowering error.
fn integer_bit_width(
    ctx: &Context,
    ty: Ptr<TypeObj>,
    loc: pliron::location::Location,
) -> Result<u32> {
    let ty_ref = ty.deref(ctx);
    let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() else {
        return pliron::input_err!(loc, "expected integer type for Rust bit intrinsic");
    };
    Ok(int_ty.width())
}

/// Return the libdevice `fabs` entry point for the concrete float type.
fn fabs_libdevice_name(
    ctx: &Context,
    ty: Ptr<TypeObj>,
    loc: pliron::location::Location,
) -> Result<&'static str> {
    let ty_ref = ty.deref(ctx);
    if ty_ref.is::<FP32Type>() {
        Ok("__nv_fabsf")
    } else if ty_ref.is::<FP64Type>() {
        Ok("__nv_fabs")
    } else {
        pliron::input_err!(
            loc,
            "expected f32 or f64 type for Rust float math intrinsic"
        )
    }
}

/// Insert the `i1` flag operand used by `llvm.ctlz` and `llvm.cttz`.
fn create_i1_constant(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: bool,
) -> Value {
    let i1_ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let width = NonZeroUsize::new(1).expect("1 is non-zero");
    let apint = APInt::from_u64(u64::from(value), width);
    let attr = IntegerAttr::new(i1_ty, apint);
    let const_op = llvm::ConstantOp::new(ctx, attr.into());
    rewriter.insert_operation(ctx, const_op.get_operation());
    const_op.get_operation().deref(ctx).get_result(0)
}

/// Cast an integer value to the target width when Rust and LLVM disagree.
///
/// This is needed for count/zero intrinsics: LLVM returns `iN`, while Rust's
/// public methods return `u32`.
fn cast_integer_value_to_type(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    value: Value,
    target_ty: Ptr<TypeObj>,
    loc: pliron::location::Location,
) -> Result<(Value, Option<Ptr<Operation>>)> {
    let source_width = integer_bit_width(ctx, value.get_type(ctx), loc.clone())?;
    let target_width = integer_bit_width(ctx, target_ty, loc)?;

    if source_width == target_width {
        return Ok((value, None));
    }

    let cast_op = if source_width < target_width {
        let zext = llvm::ZExtOp::new(ctx, value, target_ty);
        let nneg_key: pliron::identifier::Identifier = "llvm_nneg_flag".try_into().unwrap();
        zext.get_operation().deref_mut(ctx).attributes.0.insert(
            nneg_key,
            pliron::builtin::attributes::BoolAttr::new(false).into(),
        );
        zext.get_operation()
    } else {
        llvm::TruncOp::new(ctx, value, target_ty).get_operation()
    };
    rewriter.insert_operation(ctx, cast_op);
    Ok((cast_op.deref(ctx).get_result(0), Some(cast_op)))
}

/// Flatten arguments according to ABI rules and coerce each one to the
/// callee's expected parameter type.
///
/// - Slice types → (ptr, len) pair
/// - Struct types → individual field values (in MEMORY ORDER)
/// - Other types → pass through
///
/// `expected_param_tys`, when present, is the LLVM-level (post-flatten)
/// parameter signature of the callee. Each emitted (post-flatten) argument
/// is coerced to its corresponding entry — in practice this means inserting
/// an `addrspacecast` whenever a pointer arg's address space differs from
/// the declared parameter's. When `expected_param_tys` is `None` (the
/// callee declaration could not be located), each pointer falls back to
/// being cast down to the generic address space, preserving the legacy
/// behavior so device-extern calls (which always declare generic params)
/// keep working.
fn flatten_arguments(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    args: &[Value],
    operands_info: &OperandsInfo,
    expected_param_tys: Option<&[Ptr<TypeObj>]>,
) -> Result<(Vec<Value>, Vec<Ptr<TypeObj>>)> {
    let mut flattened_args = Vec::new();
    let mut flattened_arg_types = Vec::new();

    // Helper: pull the next expected param type, if any.
    let take_expected = |flattened_arg_types: &Vec<Ptr<TypeObj>>| -> Option<Ptr<TypeObj>> {
        let idx = flattened_arg_types.len();
        expected_param_tys.and_then(|tys| tys.get(idx).copied())
    };

    for arg in args.iter() {
        let arg_ty = arg.get_type(ctx);

        enum FlattenKind {
            Slice,
            Struct {
                field_types: Vec<Ptr<TypeObj>>,
                mem_to_decl: Vec<usize>,
            },
            None,
        }

        let flatten_kind = if let Some(mir_ty) = operands_info.lookup_most_recent_type(*arg) {
            let ty_ref = mir_ty.deref(ctx);
            if ty_ref.is::<MirSliceType>() || ty_ref.is::<MirDisjointSliceType>() {
                FlattenKind::Slice
            } else if let Some(struct_ty) = ty_ref.downcast_ref::<MirStructType>() {
                FlattenKind::Struct {
                    field_types: struct_ty.field_types.clone(),
                    mem_to_decl: struct_ty.memory_order(),
                }
            } else {
                FlattenKind::None
            }
        } else {
            FlattenKind::None
        };

        match flatten_kind {
            FlattenKind::Slice => {
                let ptr_ty = llvm_types::PointerType::get_generic(ctx);
                let len_ty = IntegerType::get(ctx, 64, Signedness::Signless);

                let extract_ptr = llvm::ExtractValueOp::new(ctx, *arg, vec![0])?;
                rewriter.insert_operation(ctx, extract_ptr.get_operation());
                let ptr_val = extract_ptr.get_operation().deref(ctx).get_result(0);

                let extract_len = llvm::ExtractValueOp::new(ctx, *arg, vec![1])?;
                rewriter.insert_operation(ctx, extract_len.get_operation());
                let len_val = extract_len.get_operation().deref(ctx).get_result(0);

                let (ptr_val, ptr_ty) = coerce_arg_to_param_ty(
                    ctx,
                    rewriter,
                    ptr_val,
                    ptr_ty.into(),
                    take_expected(&flattened_arg_types),
                )?;
                flattened_args.push(ptr_val);
                flattened_arg_types.push(ptr_ty);

                let (len_val, len_ty) = coerce_arg_to_param_ty(
                    ctx,
                    rewriter,
                    len_val,
                    len_ty.into(),
                    take_expected(&flattened_arg_types),
                )?;
                flattened_args.push(len_val);
                flattened_arg_types.push(len_ty);
            }
            FlattenKind::Struct {
                field_types,
                mem_to_decl,
            } => {
                let mut llvm_idx = 0u32;
                for mem_idx in 0..field_types.len() {
                    let decl_idx = mem_to_decl[mem_idx];
                    let llvm_field_ty =
                        convert_type(ctx, field_types[decl_idx]).map_err(anyhow_to_pliron)?;
                    if is_zero_sized_type(ctx, llvm_field_ty) {
                        continue;
                    }
                    let extract_op = llvm::ExtractValueOp::new(ctx, *arg, vec![llvm_idx])?;
                    rewriter.insert_operation(ctx, extract_op.get_operation());
                    let field_val = extract_op.get_operation().deref(ctx).get_result(0);

                    let (field_val, field_ty) = coerce_arg_to_param_ty(
                        ctx,
                        rewriter,
                        field_val,
                        llvm_field_ty,
                        take_expected(&flattened_arg_types),
                    )?;
                    flattened_args.push(field_val);
                    flattened_arg_types.push(field_ty);
                    llvm_idx += 1;
                }
            }
            FlattenKind::None => {
                let (final_arg, final_ty) = coerce_arg_to_param_ty(
                    ctx,
                    rewriter,
                    *arg,
                    arg_ty,
                    take_expected(&flattened_arg_types),
                )?;
                flattened_args.push(final_arg);
                flattened_arg_types.push(final_ty);
            }
        }
    }

    Ok((flattened_args, flattened_arg_types))
}

/// Coerce a single (post-flatten) argument so that it satisfies the callee's
/// declared parameter type.
///
/// When `expected_ty` is supplied, this is the principled coercion used for
/// every kind of call: pointer arguments whose address space differs from
/// the declared parameter's are bridged with an `llvm.addrspacecast` to
/// the declared address space (in either direction — generic ↔ shared,
/// shared ↔ global, etc.). Non-pointer mismatches are left for the verifier
/// to surface as a real type error.
///
/// When `expected_ty` is `None` (callee declaration not located), we fall
/// back to the legacy "always generic" policy. That fallback exists because
/// historically every call site cast pointer args down to addrspace(0) on
/// the assumption that all callees declared their parameters as generic.
/// Device-extern functions (`#[device] extern "C"`) still rely on that
/// assumption and benefit from the fallback when their declaration hasn't
/// landed in the module yet during conversion.
fn coerce_arg_to_param_ty(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    arg: Value,
    arg_ty: Ptr<TypeObj>,
    expected_ty: Option<Ptr<TypeObj>>,
) -> Result<(Value, Ptr<TypeObj>)> {
    if let Some(expected_ty) = expected_ty {
        if arg_ty == expected_ty {
            return Ok((arg, arg_ty));
        }
        let arg_addrspace = pointer_addrspace(ctx, arg_ty);
        let expected_addrspace = pointer_addrspace(ctx, expected_ty);
        if let (Some(src_as), Some(dst_as)) = (arg_addrspace, expected_addrspace)
            && src_as != dst_as
        {
            let cast_op = llvm::AddrSpaceCastOp::new(ctx, arg, dst_as);
            rewriter.insert_operation(ctx, cast_op.get_operation());
            let casted_val = cast_op.get_operation().deref(ctx).get_result(0);
            return Ok((casted_val, expected_ty));
        }
        return Ok((arg, arg_ty));
    }

    // Legacy fallback: cast any non-generic pointer down to addrspace(0).
    let arg_addrspace = pointer_addrspace(ctx, arg_ty);
    if let Some(addrspace) = arg_addrspace
        && addrspace != ADDRSPACE_GENERIC
    {
        let cast_op = llvm::AddrSpaceCastOp::new(ctx, arg, ADDRSPACE_GENERIC);
        rewriter.insert_operation(ctx, cast_op.get_operation());
        let casted_val = cast_op.get_operation().deref(ctx).get_result(0);
        let generic_ptr_ty = llvm_types::PointerType::get_generic(ctx);
        return Ok((casted_val, generic_ptr_ty.into()));
    }

    Ok((arg, arg_ty))
}

/// Return the address space of `ty` if it is an LLVM pointer type.
fn pointer_addrspace(ctx: &Context, ty: Ptr<TypeObj>) -> Option<u32> {
    ty.deref(ctx)
        .downcast_ref::<llvm_types::PointerType>()
        .map(|ptr_ty| ptr_ty.address_space())
}

/// Look up the LLVM-level parameter signature of `callee_ident` by walking
/// from `op` up to the parent module and scanning its top-level function
/// declarations.
///
/// Two callee shapes are recognised:
///
///   * `llvm::FuncOp` — already lowered. We return its `arg_types()` directly.
///     This covers device-extern declarations (inserted by the importer
///     ahead of conversion) and any callee whose conversion happened to run
///     before the current call.
///
///   * `MirFuncOp` — still in MIR form. We run `convert_function_type` to
///     project its declared MIR signature into the same LLVM-level
///     (slice/struct-flattened) parameter list that the function lowerer
///     will eventually emit. This guarantees the call site sees the exact
///     same parameter types the callee will end up with — including the
///     address space of every pointer parameter.
///
/// Returns `None` when neither form is present (e.g. the callee lives in a
/// different module). The caller then falls back to its legacy coercion.
fn find_callee_arg_types(
    ctx: &mut Context,
    op: Ptr<Operation>,
    callee_ident: &pliron::identifier::Identifier,
) -> Option<Vec<Ptr<TypeObj>>> {
    let block = op.deref(ctx).get_parent_block()?;
    let func_op = block.deref(ctx).get_parent_op(ctx)?;
    let module_op = func_op.deref(ctx).get_parent_op(ctx)?;

    let region = module_op.deref(ctx).get_region(0);
    let module_block = region.deref(ctx).iter(ctx).next()?;

    // First pass: walk the module's top-level ops once and extract whichever
    // declaration shape (LLVM or MIR) carries the matching symbol name.
    // `Ptr<Operation>` is `Copy`, so we can collect candidates without
    // borrowing the context past this loop.
    let mut llvm_decl_op: Option<Ptr<Operation>> = None;
    let mut mir_decl_op: Option<Ptr<Operation>> = None;
    for existing_op in module_block.deref(ctx).iter(ctx) {
        if let Some(existing_func) = Operation::get_op::<llvm::FuncOp>(existing_op, ctx)
            && &existing_func.get_symbol_name(ctx) == callee_ident
        {
            llvm_decl_op = Some(existing_op);
            break;
        }
        if let Some(existing_func) = MirFuncOp::wrap(ctx, existing_op)
            && &existing_func.get_symbol_name(ctx) == callee_ident
        {
            mir_decl_op = Some(existing_op);
            // Keep scanning — an `llvm::FuncOp` would still take precedence.
        }
    }

    if let Some(existing_op) = llvm_decl_op {
        let func = Operation::get_op::<llvm::FuncOp>(existing_op, ctx)?;
        let func_ty = func.get_type(ctx);
        return Some(func_ty.deref(ctx).arg_types().clone());
    }

    if let Some(existing_op) = mir_decl_op {
        let func = MirFuncOp::wrap(ctx, existing_op)?;
        let mir_func_ty = func.get_type(ctx);
        // Match the callee's own boundary kind: a kernel callee (rare; not
        // produced by Rust code today, but cheap to keep correct) keeps its
        // params as byval, internal callees stay flattened. Reading
        // `is_kernel_func` here keeps `find_callee_arg_types` consistent
        // with what `lowering.rs::convert_func` would produce for the
        // same callee.
        let callee_is_kernel = is_kernel_func(ctx, existing_op);
        let llvm_func_ty = convert_function_type(ctx, mir_func_ty, callee_is_kernel).ok()?;
        return Some(llvm_func_ty.deref(ctx).arg_types().clone());
    }

    None
}

/// Resolve a device-extern symbol name by stripping the internal prefix.
///
/// When calling `#[device] extern "C"` functions, the macro-expanded Rust code
/// calls symbols like `cuda_oxide_device_extern_<hash>_foo`. We strip the
/// reserved prefix here so the emitted LLVM IR refers to the original symbol
/// `foo` that the linked LTOIR provides.
///
/// # Limitations
///
/// This approach derives the original name by stripping the prefix, which means:
/// - Custom `#[link_name = "..."]` attributes are NOT honored (edge case).
/// - If the original function is `bar` but the user wrote `#[link_name = "custom"]`,
///   we'd emit `bar`, not `custom`. Acceptable for current call sites.
fn resolve_device_extern_symbol(callee_name: &str) -> String {
    device_extern_base_name(callee_name)
        .map(str::to_string)
        .unwrap_or_else(|| callee_name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_device_extern_symbol() {
        use reserved_oxide_symbols::{device_extern_symbol, kernel_symbol};

        // Plain form: the hash-suffixed prefix is stripped to leave the
        // user-facing base name.
        assert_eq!(
            resolve_device_extern_symbol(&device_extern_symbol("dot_product")),
            "dot_product"
        );

        // FQDN form (cross-crate): the extractor skips the crate qualifier
        // and the hash-suffixed prefix.
        let fqdn = format!("device_ffi_test::{}", device_extern_symbol("foo"));
        assert_eq!(resolve_device_extern_symbol(&fqdn), "foo");

        // Non-extern symbols pass through unchanged — this is what lets
        // ordinary device-function calls reach the LLVM backend without
        // the device-extern detour.
        assert_eq!(
            resolve_device_extern_symbol("my_module::regular_function"),
            "my_module::regular_function"
        );

        // A kernel symbol must NOT be confused for a device-extern symbol
        // (mutual-exclusion guarantee from reserved-oxide-symbols).
        let kernel_name = kernel_symbol("my_kernel");
        assert_eq!(resolve_device_extern_symbol(&kernel_name), kernel_name);
    }

    /// Sample of the Rust float-math placeholder → libdevice symbol mapping.
    /// This locks the table down so a typo in either the intrinsic enum or
    /// the libdevice symbol surfaces as a unit-test failure rather than a
    /// runtime "symbol not found" error after a long compile cycle.
    #[test]
    fn test_float_math_placeholder_round_trip() {
        let cases: &[(&str, RustFloatMathIntrinsic)] = &[
            (
                rust_intrinsics::CALLEE_SQRT_F32,
                RustFloatMathIntrinsic::SqrtF32,
            ),
            (
                rust_intrinsics::CALLEE_SQRT_F64,
                RustFloatMathIntrinsic::SqrtF64,
            ),
            (
                rust_intrinsics::CALLEE_POWI_F32,
                RustFloatMathIntrinsic::PowiF32,
            ),
            (
                rust_intrinsics::CALLEE_POWI_F64,
                RustFloatMathIntrinsic::PowiF64,
            ),
            (
                rust_intrinsics::CALLEE_FMA_F32,
                RustFloatMathIntrinsic::FmaF32,
            ),
            (
                rust_intrinsics::CALLEE_FMULADD_F64,
                RustFloatMathIntrinsic::FmuladdF64,
            ),
            (rust_intrinsics::CALLEE_FABS, RustFloatMathIntrinsic::Fabs),
            (
                rust_intrinsics::CALLEE_COPYSIGN_F32,
                RustFloatMathIntrinsic::CopysignF32,
            ),
            (
                rust_intrinsics::CALLEE_LOG2_F64,
                RustFloatMathIntrinsic::Log2F64,
            ),
        ];

        for (name, expected) in cases {
            assert_eq!(
                RustFloatMathIntrinsic::from_placeholder_callee(name),
                Some(*expected),
                "placeholder `{name}` did not round-trip"
            );
        }

        assert_eq!(
            RustFloatMathIntrinsic::from_placeholder_callee("not_a_placeholder"),
            None,
        );
    }

    #[test]
    fn test_float_math_arg_count() {
        assert_eq!(RustFloatMathIntrinsic::SqrtF32.arg_count(), 1);
        assert_eq!(RustFloatMathIntrinsic::Fabs.arg_count(), 1);
        assert_eq!(RustFloatMathIntrinsic::FloorF64.arg_count(), 1);
        assert_eq!(RustFloatMathIntrinsic::PowiF32.arg_count(), 2);
        assert_eq!(RustFloatMathIntrinsic::PowfF64.arg_count(), 2);
        assert_eq!(RustFloatMathIntrinsic::CopysignF32.arg_count(), 2);
        assert_eq!(RustFloatMathIntrinsic::FmaF32.arg_count(), 3);
        assert_eq!(RustFloatMathIntrinsic::FmuladdF64.arg_count(), 3);
    }

    /// `Fabs` is the only float-math intrinsic whose libdevice name depends on
    /// the result type (the others are width-suffixed in the enum itself).
    /// Ensure both float widths are dispatched correctly and that anything
    /// else is rejected.
    #[test]
    fn test_fabs_libdevice_name_dispatches_on_float_width() {
        let mut ctx = Context::new();
        let f32_ty = FP32Type::get(&ctx).into();
        let f64_ty = FP64Type::get(&ctx).into();
        let i32_ty = IntegerType::get(&mut ctx, 32, Signedness::Signless).into();
        let loc = pliron::location::Location::Unknown;

        assert_eq!(
            fabs_libdevice_name(&ctx, f32_ty, loc.clone()).unwrap(),
            "__nv_fabsf"
        );
        assert_eq!(
            fabs_libdevice_name(&ctx, f64_ty, loc.clone()).unwrap(),
            "__nv_fabs"
        );
        assert!(fabs_libdevice_name(&ctx, i32_ty, loc).is_err());
    }
}
