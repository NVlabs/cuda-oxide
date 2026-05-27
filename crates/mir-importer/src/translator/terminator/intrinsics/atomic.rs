/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Atomic operation intrinsic handlers.
//!
//! Translates atomic method calls into NVVM atomic dialect operations.
//! Supports two front-ends:
//!
//! 1. **`cuda_device::atomic::*`** — custom GPU atomic types with explicit scope
//! 2. **`core::sync::atomic::*`** — standard library atomics (via `std::intrinsics::atomic_*`)
//!
//! Both front-ends emit the same NVVM ops and share the entire lowering pipeline
//! (mir-lower fence splitting → `dialect-llvm` → export → llc → PTX).
//!
//! # cuda_device Path — Type Resolution
//!
//! The atomic type name encodes scope and element type:
//!
//! ```text
//! BlockAtomicI64::fetch_add
//! ─────┬────────  ────┬────
//!   scope prefix    method
//!       └── AtomicI64 = 64-bit signed integer
//! ```
//!
//! | Prefix            | Scope   | PTX    |
//! |-------------------|---------|--------|
//! | `DeviceAtomic*`   | Device  | `.gpu` |
//! | `BlockAtomic*`    | Block   | `.cta` |
//! | `SystemAtomic*`   | System  | `.sys` |
//!
//! # cuda_device Path — Method → RMW Kind Mapping
//!
//! | Method       | Integer RMW Kind   | Float RMW Kind |
//! |--------------|--------------------|----------------|
//! | `fetch_add`  | `Add`              | `FAdd`         |
//! | `fetch_sub`  | `Sub`              | —              |
//! | `fetch_and`  | `And`              | —              |
//! | `fetch_or`   | `Or`               | —              |
//! | `fetch_xor`  | `Xor`              | —              |
//! | `fetch_min`  | `Min` / `UMin` [*] | —              |
//! | `fetch_max`  | `Max` / `UMax` [*] | —              |
//! | `swap`       | `Xchg`             | `Xchg`         |
//!
//! [*] `fetch_min`/`fetch_max` use signed (`Min`/`Max`) for `I32`/`I64`,
//!     unsigned (`UMin`/`UMax`) for `U32`/`U64`.
//!
//! # core::sync::atomic Path
//!
//! Standard library atomics compile down to `std::intrinsics::atomic_*` (or
//! `core::intrinsics::atomic_*` in `#![no_std]`).  These are generic intrinsics
//! whose ordering is a **const generic**, not a runtime argument:
//!
//! ```text
//! std::intrinsics::atomic_xadd::<u32, u32, AtomicOrdering::Relaxed>(ptr, val)
//! ─────────────────────┬─────    ──┬──      ────────┬───────────── ──┬──  ─┬─
//!                 intrinsic name   type          ordering           ptr   val
//! ```
//!
//! All `core::sync::atomic` operations are lowered with **system scope** (`.sys`)
//! for safe host-device coherence, matching CUDA C++ `cuda::atomic<T>` defaults.

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;

use dialect_nvvm::ops::atomic::{
    AtomicOrdering, AtomicRmwKind, AtomicScope, NvvmAtomicCmpxchgOp, NvvmAtomicLoadOp,
    NvvmAtomicRmwOp, NvvmAtomicStoreOp,
};

use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;
use rustc_public::ty::{GenericArgKind, RigidTy, TyConstKind, TyKind};
// =============================================================================
// Type info — extracted from the atomic type name in the call path
// =============================================================================

/// Describes an atomic type parsed from a `cuda_device::atomic::*` path.
///
/// Example: `BlockAtomicI64` → `{ bit_width: 64, is_float: false, is_signed: true, scope: Block }`
pub struct AtomicTypeInfo {
    pub bit_width: u32,
    pub is_float: bool,
    pub is_signed: bool,
    pub scope: AtomicScope,
}

impl AtomicTypeInfo {
    /// Get the pliron result type for this atomic's element.
    fn element_type(&self, ctx: &mut Context) -> Ptr<pliron::r#type::TypeObj> {
        if self.is_float {
            match self.bit_width {
                32 => FP32Type::get(ctx).into(),
                64 => FP64Type::get(ctx).into(),
                _ => unreachable!("unsupported float atomic width: {}", self.bit_width),
            }
        } else {
            let signedness = if self.is_signed {
                Signedness::Signed
            } else {
                Signedness::Unsigned
            };
            IntegerType::get(ctx, self.bit_width, signedness).to_ptr()
        }
    }
}

/// Parse an atomic type name (e.g., `"DeviceAtomicU32"`, `"BlockAtomicI64"`) into type info.
///
/// Device scope uses the `DeviceAtomic*` prefix to avoid name collision with
/// `core::sync::atomic::Atomic*`. Returns `None` if the name doesn't match.
fn parse_atomic_type_name(type_name: &str) -> Option<AtomicTypeInfo> {
    // Extract scope prefix and base type suffix. Try longer prefixes first.
    let (scope, base) = if let Some(rest) = type_name.strip_prefix("BlockAtomic") {
        (AtomicScope::Block, rest)
    } else if let Some(rest) = type_name.strip_prefix("SystemAtomic") {
        (AtomicScope::System, rest)
    } else if let Some(rest) = type_name.strip_prefix("DeviceAtomic") {
        (AtomicScope::Device, rest)
    } else {
        return None;
    };

    let (bit_width, is_float, is_signed) = match base {
        "U32" => (32, false, false),
        "I32" => (32, false, true),
        "U64" => (64, false, false),
        "I64" => (64, false, true),
        "F32" => (32, true, false),
        "F64" => (64, true, false),
        _ => return None,
    };

    Some(AtomicTypeInfo {
        bit_width,
        is_float,
        is_signed,
        scope,
    })
}

/// Check whether a call path refers to a known atomic type.
///
/// Used as a guard in the `try_dispatch_intrinsic` match arm.
pub fn is_atomic_path(path: &str) -> bool {
    parse_atomic_path(path).is_some()
}

/// Parse a full call path into (type_info, method_name).
///
/// Example: `"cuda_device::atomic::AtomicU32::fetch_add"` → `(AtomicTypeInfo{..}, "fetch_add")`
fn parse_atomic_path(path: &str) -> Option<(AtomicTypeInfo, &str)> {
    let mut parts = path.rsplit("::");
    let method = parts.next()?;
    let type_name = parts.next()?;
    let info = parse_atomic_type_name(type_name)?;
    Some((info, method))
}

// =============================================================================
// RMW kind resolution
// =============================================================================

/// Map a method name to the appropriate `AtomicRmwKind`.
///
/// For `fetch_min`/`fetch_max`, signedness matters:
/// - Unsigned types (U32, U64) → `UMin`/`UMax`
/// - Signed types (I32, I64) → `Min`/`Max`
///
/// For `fetch_add` on float types → `FAdd` (hardware `atom.add.f32/f64`).
fn method_to_rmw_kind(method: &str, info: &AtomicTypeInfo) -> Option<AtomicRmwKind> {
    match method {
        "fetch_add" => {
            if info.is_float {
                Some(AtomicRmwKind::FAdd)
            } else {
                Some(AtomicRmwKind::Add)
            }
        }
        "fetch_sub" => Some(AtomicRmwKind::Sub),
        "fetch_and" => Some(AtomicRmwKind::And),
        "fetch_or" => Some(AtomicRmwKind::Or),
        "fetch_xor" => Some(AtomicRmwKind::Xor),
        "fetch_min" => {
            if info.is_signed {
                Some(AtomicRmwKind::Min)
            } else {
                Some(AtomicRmwKind::UMin)
            }
        }
        "fetch_max" => {
            if info.is_signed {
                Some(AtomicRmwKind::Max)
            } else {
                Some(AtomicRmwKind::UMax)
            }
        }
        "swap" => Some(AtomicRmwKind::Xchg),
        _ => None,
    }
}

// =============================================================================
// Ordering extraction from MIR constants
// =============================================================================

/// Extract an `AtomicOrdering` from a MIR operand that represents
/// a `cuda_device::atomic::AtomicOrdering` enum value.
///
/// The enum has `#[repr(u8)]` with discriminants:
///   Relaxed=0, Acquire=1, Release=2, AcqRel=3, SeqCst=4
fn extract_ordering(operand: &mir::Operand) -> AtomicOrdering {
    if let mir::Operand::Constant(constant) = operand {
        let const_str = format!("{:?}", constant.const_);
        let discr = rvalue::extract_enum_discriminant(&constant.const_, &const_str);
        match discr {
            0 => AtomicOrdering::Relaxed,
            1 => AtomicOrdering::Acquire,
            2 => AtomicOrdering::Release,
            3 => AtomicOrdering::AcqRel,
            4 => AtomicOrdering::SeqCst,
            _ => AtomicOrdering::SeqCst, // Conservative fallback
        }
    } else {
        // Non-constant ordering (dynamic) -- use SeqCst as conservative default.
        AtomicOrdering::SeqCst
    }
}

// =============================================================================
// Top-level dispatch — called from terminator/mod.rs
// =============================================================================

/// Dispatch an atomic intrinsic call to the appropriate emit function.
///
/// Returns `Ok(Some(op))` if the method was handled, `Ok(None)` if the
/// method is not an intrinsic (e.g., `new()`), or `Err` on failure.
#[allow(clippy::too_many_arguments)]
pub fn dispatch(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    path: &str,
) -> TranslationResult<Option<Ptr<Operation>>> {
    let (type_info, method) = match parse_atomic_path(path) {
        Some(parsed) => parsed,
        None => return Ok(None),
    };

    match method {
        "load" => Ok(Some(emit_atomic_load(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            &type_info,
        )?)),

        "store" => Ok(Some(emit_atomic_store(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            &type_info,
        )?)),

        "fetch_add" | "fetch_sub" | "fetch_and" | "fetch_or" | "fetch_xor" | "fetch_min"
        | "fetch_max" | "swap" => {
            let rmw_kind = method_to_rmw_kind(method, &type_info).unwrap();
            Ok(Some(emit_atomic_rmw(
                ctx,
                body,
                args,
                destination,
                target,
                block_ptr,
                prev_op,
                value_map,
                block_map,
                loc,
                &type_info,
                rmw_kind,
            )?))
        }

        "compare_exchange_raw" => Ok(Some(emit_atomic_compare_exchange(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            &type_info,
        )?)),

        // new() is a const fn compiled normally; compare_exchange() is an
        // #[inline(always)] wrapper around compare_exchange_raw — both are
        // handled by regular MIR translation, not intrinsic dispatch.
        _ => Ok(None),
    }
}

// =============================================================================
// Emit functions
// =============================================================================

/// Emit an atomic load.
///
/// MIR args: `[self_ptr, ordering]`
#[allow(clippy::too_many_arguments)]
fn emit_atomic_load(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    type_info: &AtomicTypeInfo,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 2 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "atomic load expects 2 arguments (self, ordering), got {}",
                args.len()
            ))
        );
    }

    let ordering = extract_ordering(&args[1]);
    let result_ty = type_info.element_type(ctx);

    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let nvvm_op =
        NvvmAtomicLoadOp::build(ctx, ptr_val, result_ty, ordering, type_info.scope.clone());
    let op_ptr = nvvm_op.get_operation();
    op_ptr.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op_ptr.insert_after(ctx, prev);
    } else {
        op_ptr.insert_at_front(block_ptr, ctx);
    }

    let result_value = op_ptr.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        op_ptr,
        value_map,
        block_map,
        loc,
        "atomic load call without target block",
    )
}

/// Emit an atomic store.
///
/// MIR args: `[self_ptr, val, ordering]`
#[allow(clippy::too_many_arguments)]
fn emit_atomic_store(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    type_info: &AtomicTypeInfo,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "atomic store expects 3 arguments (self, val, ordering), got {}",
                args.len()
            ))
        );
    }

    let ordering = extract_ordering(&args[2]);

    // Get the value to store (arg 1)
    let (val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Get the pointer (arg 0) -- self
    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let nvvm_op = NvvmAtomicStoreOp::build(ctx, val, ptr_val, ordering, type_info.scope.clone());
    let op_ptr = nvvm_op.get_operation();
    op_ptr.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op_ptr.insert_after(ctx, prev);
    } else {
        op_ptr.insert_at_front(block_ptr, ctx);
    }

    // Store returns unit -- set destination to a unit value
    let unit_ty = dialect_mir::types::MirTupleType::get(ctx, vec![]);
    let unit_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructTupleOp::get_concrete_op_info(),
        vec![unit_ty.into()],
        vec![],
        vec![],
        0,
    );
    unit_op.deref_mut(ctx).set_loc(loc.clone());
    unit_op.insert_after(ctx, op_ptr);
    let unit_val = unit_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        unit_val,
        target,
        block_ptr,
        unit_op,
        value_map,
        block_map,
        loc,
        "atomic store call without target block",
    )
}

/// Emit an atomic read-modify-write operation.
///
/// Handles all RMW methods: `fetch_add`, `fetch_sub`, `fetch_and`, `fetch_or`,
/// `fetch_xor`, `fetch_min`, `fetch_max`, `swap`.
///
/// MIR args: `[self_ptr, val, ordering]`
#[allow(clippy::too_many_arguments)]
fn emit_atomic_rmw(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    type_info: &AtomicTypeInfo,
    rmw_kind: AtomicRmwKind,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "atomic RMW expects 3 arguments (self, val, ordering), got {}",
                args.len()
            ))
        );
    }

    let ordering = extract_ordering(&args[2]);
    let result_ty = type_info.element_type(ctx);

    // Get the pointer (arg 0)
    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Get the value operand (arg 1)
    let (val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let nvvm_op = NvvmAtomicRmwOp::build(
        ctx,
        ptr_val,
        val,
        result_ty,
        rmw_kind,
        ordering,
        type_info.scope.clone(),
    );
    let op_ptr = nvvm_op.get_operation();
    op_ptr.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op_ptr.insert_after(ctx, prev);
    } else {
        op_ptr.insert_at_front(block_ptr, ctx);
    }

    let result_value = op_ptr.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        op_ptr,
        value_map,
        block_map,
        loc,
        "atomic RMW call without target block",
    )
}

/// Emit an atomic compare-and-exchange.
///
/// Only valid for integer types. Float types do not support CAS in PTX.
///
/// MIR args: `[self_ptr, current, new, success_ordering, failure_ordering]`
#[allow(clippy::too_many_arguments)]
fn emit_atomic_compare_exchange(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    type_info: &AtomicTypeInfo,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 5 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "atomic compare_exchange expects 5 arguments (self, current, new, success, failure), got {}",
                args.len()
            ))
        );
    }

    let success_ordering = extract_ordering(&args[3]);
    let failure_ordering = extract_ordering(&args[4]);
    let result_ty = type_info.element_type(ctx);

    // Get the pointer (arg 0)
    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Get the expected (current) value (arg 1)
    let (cmp_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    // Get the new value (arg 2)
    let (new_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let nvvm_op = NvvmAtomicCmpxchgOp::build(
        ctx,
        ptr_val,
        cmp_val,
        new_val,
        result_ty,
        success_ordering,
        failure_ordering,
        type_info.scope.clone(),
    );
    let op_ptr = nvvm_op.get_operation();
    op_ptr.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op_ptr.insert_after(ctx, prev);
    } else {
        op_ptr.insert_at_front(block_ptr, ctx);
    }

    let result_value = op_ptr.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        op_ptr,
        value_map,
        block_map,
        loc,
        "atomic compare_exchange call without target block",
    )
}

// =============================================================================
// core::sync::atomic support — std::intrinsics::atomic_* dispatch
// =============================================================================

/// Check whether a call path is a core/std atomic intrinsic.
///
/// Matches `std::intrinsics::atomic_*` and `core::intrinsics::atomic_*`.
pub fn is_core_atomic_intrinsic(path: &str) -> bool {
    path.starts_with("std::intrinsics::atomic_") || path.starts_with("core::intrinsics::atomic_")
}

/// Extract the operation name from a core atomic intrinsic path.
///
/// Example: `"std::intrinsics::atomic_xadd"` → `Some("xadd")`
fn parse_core_intrinsic_op(path: &str) -> Option<&str> {
    path.strip_prefix("std::intrinsics::atomic_")
        .or_else(|| path.strip_prefix("core::intrinsics::atomic_"))
}

/// Map `std::intrinsics::AtomicOrdering` discriminant to our `AtomicOrdering`.
///
/// **Important**: The discriminant layout differs from `cuda_device::AtomicOrdering`:
///
/// | Discriminant | `std::intrinsics::AtomicOrdering` | `cuda_device::AtomicOrdering` |
/// |--------------|-----------------------------------|-----------------------|
/// |            0 | Relaxed                             | Relaxed               |
/// |            1 | **Release**                         | **Acquire**           |
/// |            2 | **Acquire**                         | **Release**           |
/// |            3 | AcqRel                              | AcqRel                |
/// |            4 | SeqCst                              | SeqCst                |
fn intrinsic_ordering_from_discriminant(discr: u64) -> AtomicOrdering {
    match discr {
        0 => AtomicOrdering::Relaxed,
        1 => AtomicOrdering::Release, // std has Release=1, unlike cuda_device Acquire=1
        2 => AtomicOrdering::Acquire, // std has Acquire=2, unlike cuda_device Release=2
        3 => AtomicOrdering::AcqRel,
        4 => AtomicOrdering::SeqCst,
        _ => AtomicOrdering::SeqCst, // Conservative fallback
    }
}

/// Build `AtomicTypeInfo` from a rustc type, with system scope.
///
/// Core atomics always use system scope for safe host-device coherence.
fn type_info_from_mir_ty(ty: &rustc_public::ty::Ty) -> Option<AtomicTypeInfo> {
    let (bit_width, is_float, is_signed) = match ty.kind() {
        TyKind::RigidTy(RigidTy::Uint(uint_ty)) => {
            use rustc_public::ty::UintTy;
            let width = match uint_ty {
                UintTy::U8 => 8,
                UintTy::U16 => 16,
                UintTy::U32 => 32,
                UintTy::U64 => 64,
                // 128-bit: PTX .b128 requires sm_90+ (Hopper); accepted here and
                // gated downstream by the architecture check.
                UintTy::U128 => 128,
                // usize is target-dependent (32-bit on nvptx, 64-bit on nvptx64).
                // We only target nvptx64 today; making this configurable via
                // `PipelineConfig::target_pointer_width` is a future change.
                UintTy::Usize => 64,
            };
            (width, false, false)
        }
        TyKind::RigidTy(RigidTy::Int(int_ty)) => {
            use rustc_public::ty::IntTy;
            let width = match int_ty {
                IntTy::I8 => 8,
                IntTy::I16 => 16,
                IntTy::I32 => 32,
                IntTy::I64 => 64,
                // 128-bit: PTX .b128 requires sm_90+ (Hopper); accepted here and
                // gated downstream by the architecture check.
                IntTy::I128 => 128,
                // isize is target-dependent (32-bit on nvptx, 64-bit on nvptx64).
                // We only target nvptx64 today; making this configurable via
                // `PipelineConfig::target_pointer_width` is a future change.
                IntTy::Isize => 64,
            };
            (width, false, true)
        }
        TyKind::RigidTy(RigidTy::Float(float_ty)) => {
            use rustc_public::ty::FloatTy;
            let width = match float_ty {
                FloatTy::F16 => 16,
                FloatTy::F32 => 32,
                FloatTy::F64 => 64,
                FloatTy::F128 => 128,
            };
            (width, true, false)
        }
        _ => return None,
    };

    Some(AtomicTypeInfo {
        bit_width,
        is_float,
        is_signed,
        scope: AtomicScope::System, // core atomics always use system scope
    })
}

/// Map a core intrinsic operation name to an `AtomicRmwKind`.
///
/// | Intrinsic op | RMW Kind              |
/// |--------------|-----------------------|
/// | `xadd`       | `Add` / `FAdd`        |
/// | `xsub`       | `Sub`                 |
/// | `and`        | `And`                 |
/// | `or`         | `Or`                  |
/// | `xor`        | `Xor`                 |
/// | `min`        | `Min` (signed)        |
/// | `umin`       | `UMin` (unsigned)     |
/// | `max`        | `Max` (signed)        |
/// | `umax`       | `UMax` (unsigned)     |
/// | `xchg`       | `Xchg`                |
fn intrinsic_op_to_rmw_kind(op: &str, info: &AtomicTypeInfo) -> Option<AtomicRmwKind> {
    match op {
        "xadd" => {
            if info.is_float {
                Some(AtomicRmwKind::FAdd)
            } else {
                Some(AtomicRmwKind::Add)
            }
        }
        "xsub" => Some(AtomicRmwKind::Sub),
        "and" => Some(AtomicRmwKind::And),
        "or" => Some(AtomicRmwKind::Or),
        "xor" => Some(AtomicRmwKind::Xor),
        "min" => Some(AtomicRmwKind::Min),
        "umin" => Some(AtomicRmwKind::UMin),
        "max" => Some(AtomicRmwKind::Max),
        "umax" => Some(AtomicRmwKind::UMax),
        "xchg" => Some(AtomicRmwKind::Xchg),
        _ => None,
    }
}

/// Extract the ordering from the const generic argument of a core atomic intrinsic.
///
/// The ordering is the 3rd generic arg (index 2) and is a const of type
/// `std::intrinsics::AtomicOrdering`.
fn extract_ordering_from_generics(substs: &rustc_public::ty::GenericArgs) -> AtomicOrdering {
    if let Some(GenericArgKind::Const(c)) = substs.0.get(2) {
        let discr = match c.kind() {
            TyConstKind::Value(_, alloc) => alloc.read_uint().unwrap_or(4) as u64,
            _ => c.eval_target_usize().unwrap_or(4),
        };
        intrinsic_ordering_from_discriminant(discr)
    } else {
        // Fallback: SeqCst (conservative)
        AtomicOrdering::SeqCst
    }
}

/// Extract the element type from the first generic type arg.
fn extract_type_info_from_generics(
    substs: &rustc_public::ty::GenericArgs,
) -> Option<AtomicTypeInfo> {
    substs.0.iter().find_map(|arg| match arg {
        GenericArgKind::Type(ty) => type_info_from_mir_ty(ty),
        _ => None,
    })
}

/// Dispatch a `std::intrinsics::atomic_*` / `core::intrinsics::atomic_*` call.
///
/// Extracts the generic args (type, ordering) from the `func` operand and
/// routes to the appropriate emit function.  All operations use **system scope**.
///
/// Returns `Ok(Some(op))` if handled, `Err` on failure.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_core_intrinsic(
    ctx: &mut Context,
    body: &mir::Body,
    func: &mir::Operand,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    path: &str,
) -> TranslationResult<Ptr<Operation>> {
    let op_name = parse_core_intrinsic_op(path).unwrap_or("");

    // Extract generic args from the func operand
    let (type_info, ordering) = extract_core_intrinsic_generics(func, &loc)?;

    // Route by operation name
    if op_name == "load" {
        emit_core_atomic_load(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            &type_info,
            ordering,
        )
    } else if op_name == "store" {
        emit_core_atomic_store(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            &type_info,
            ordering,
        )
    } else if op_name == "cxchg" || op_name == "cxchgweak" {
        emit_core_atomic_cmpxchg(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            &type_info,
            ordering,
        )
    } else if let Some(rmw_kind) = intrinsic_op_to_rmw_kind(op_name, &type_info) {
        emit_core_atomic_rmw(
            ctx,
            body,
            args,
            destination,
            target,
            block_ptr,
            prev_op,
            value_map,
            block_map,
            loc,
            &type_info,
            ordering,
            rmw_kind,
        )
    } else {
        input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!("unsupported core atomic intrinsic: {path}"))
        )
    }
}

/// Extract type info and ordering from a core atomic intrinsic's generic args.
fn extract_core_intrinsic_generics(
    func: &mir::Operand,
    loc: &Location,
) -> TranslationResult<(AtomicTypeInfo, AtomicOrdering)> {
    if let mir::Operand::Constant(const_op) = func
        && let TyKind::RigidTy(RigidTy::FnDef(_, substs)) = const_op.const_.ty().kind()
    {
        if let Some(type_info) = extract_type_info_from_generics(&substs) {
            // PTX has no 8-bit atomics; 16-bit is partial (sm_70+). Reject both for now.
            if type_info.bit_width == 8 {
                return input_err!(
                    loc.clone(),
                    TranslationErr::unsupported(
                        "8-bit atomics are not supported by PTX; use 32-bit or 64-bit"
                    )
                );
            }
            if type_info.bit_width == 16 {
                return input_err!(
                    loc.clone(),
                    TranslationErr::unsupported(
                        "16-bit atomics are not yet supported; use 32-bit or 64-bit"
                    )
                );
            }
            let ordering = extract_ordering_from_generics(&substs);
            return Ok((type_info, ordering));
        }
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(
                "could not extract element type from core atomic intrinsic generics"
            )
        );
    }
    input_err!(
        loc.clone(),
        TranslationErr::unsupported(
            "core atomic intrinsic: could not extract generics from func operand"
        )
    )
}

// =============================================================================
// Core intrinsic emit functions
//
// These handle the MIR arg layout for std::intrinsics::atomic_* which differs
// from cuda_device (no ordering arg, different arg count).  They build the same
// NVVM ops as the cuda_device emit functions.
// =============================================================================

/// Emit a core atomic load.
///
/// MIR args: `[ptr]` -- 1 arg, ordering from const generic.
#[allow(clippy::too_many_arguments)]
fn emit_core_atomic_load(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    type_info: &AtomicTypeInfo,
    ordering: AtomicOrdering,
) -> TranslationResult<Ptr<Operation>> {
    let result_ty = type_info.element_type(ctx);

    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    let nvvm_op =
        NvvmAtomicLoadOp::build(ctx, ptr_val, result_ty, ordering, type_info.scope.clone());
    let op_ptr = nvvm_op.get_operation();
    op_ptr.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op_ptr.insert_after(ctx, prev);
    } else {
        op_ptr.insert_at_front(block_ptr, ctx);
    }

    let result_value = op_ptr.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        op_ptr,
        value_map,
        block_map,
        loc,
        "core atomic load call without target block",
    )
}

/// Emit a core atomic store.
///
/// MIR args: `[ptr, val]` -- 2 args, ordering from const generic.
#[allow(clippy::too_many_arguments)]
fn emit_core_atomic_store(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    type_info: &AtomicTypeInfo,
    ordering: AtomicOrdering,
) -> TranslationResult<Ptr<Operation>> {
    // Get the value to store (arg 1)
    let (val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Get the pointer (arg 0)
    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let nvvm_op = NvvmAtomicStoreOp::build(ctx, val, ptr_val, ordering, type_info.scope.clone());
    let op_ptr = nvvm_op.get_operation();
    op_ptr.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op_ptr.insert_after(ctx, prev);
    } else {
        op_ptr.insert_at_front(block_ptr, ctx);
    }

    // Store returns unit
    let unit_ty = dialect_mir::types::MirTupleType::get(ctx, vec![]);
    let unit_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructTupleOp::get_concrete_op_info(),
        vec![unit_ty.into()],
        vec![],
        vec![],
        0,
    );
    unit_op.deref_mut(ctx).set_loc(loc.clone());
    unit_op.insert_after(ctx, op_ptr);
    let unit_val = unit_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        unit_val,
        target,
        block_ptr,
        unit_op,
        value_map,
        block_map,
        loc,
        "core atomic store call without target block",
    )
}

/// Emit a core atomic read-modify-write operation.
///
/// MIR args: `[ptr, val]` -- 2 args, ordering from const generic.
#[allow(clippy::too_many_arguments)]
fn emit_core_atomic_rmw(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    type_info: &AtomicTypeInfo,
    ordering: AtomicOrdering,
    rmw_kind: AtomicRmwKind,
) -> TranslationResult<Ptr<Operation>> {
    let result_ty = type_info.element_type(ctx);

    // Get the pointer (arg 0)
    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Get the value operand (arg 1)
    let (val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let nvvm_op = NvvmAtomicRmwOp::build(
        ctx,
        ptr_val,
        val,
        result_ty,
        rmw_kind,
        ordering,
        type_info.scope.clone(),
    );
    let op_ptr = nvvm_op.get_operation();
    op_ptr.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op_ptr.insert_after(ctx, prev);
    } else {
        op_ptr.insert_at_front(block_ptr, ctx);
    }

    let result_value = op_ptr.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        op_ptr,
        value_map,
        block_map,
        loc,
        "core atomic RMW call without target block",
    )
}

/// Emit a core atomic compare-and-exchange.
///
/// MIR args: `[ptr, old, new]` -- 3 args, ordering from const generic.
/// Returns `(old_val, bool)` tuple (LLVM cmpxchg semantics).
#[allow(clippy::too_many_arguments)]
fn emit_core_atomic_cmpxchg(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    type_info: &AtomicTypeInfo,
    success_ordering: AtomicOrdering,
) -> TranslationResult<Ptr<Operation>> {
    // For cmpxchg, use Monotonic as failure ordering (conservative but correct;
    // the actual failure ordering would need a 4th const generic which core
    // intrinsics encode separately -- for now Monotonic is safe).
    let failure_ordering = AtomicOrdering::Relaxed;
    let result_ty = type_info.element_type(ctx);

    // Get the pointer (arg 0)
    let (ptr_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;

    // Get the expected (current) value (arg 1)
    let (cmp_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    // Get the new value (arg 2)
    let (new_val, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;

    let nvvm_op = NvvmAtomicCmpxchgOp::build(
        ctx,
        ptr_val,
        cmp_val,
        new_val,
        result_ty,
        success_ordering,
        failure_ordering,
        type_info.scope.clone(),
    );
    let op_ptr = nvvm_op.get_operation();
    op_ptr.deref_mut(ctx).set_loc(loc.clone());

    if let Some(prev) = last_op {
        op_ptr.insert_after(ctx, prev);
    } else {
        op_ptr.insert_at_front(block_ptr, ctx);
    }

    let result_value = op_ptr.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result_value,
        target,
        block_ptr,
        op_ptr,
        value_map,
        block_map,
        loc,
        "core atomic cmpxchg call without target block",
    )
}
