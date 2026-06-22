// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integer dot product operations (`dp4a`, `dp2a`).
//!
//! These are single-thread, non-convergent packed integer dot product
//! instructions lowered to inline PTX. Available from `sm_61+`.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Signed 4-element byte dot product with accumulation: `d = c + dot(a, b)`.
///
/// `a` and `b` are each 4 packed signed bytes; `c` and `d` are signed 32-bit.
///
/// PTX: `dp4a.s32.s32 $0, $1, $2, $3;`  (requires `sm_61+`)
///
/// # Operands
///
/// - `a` (u32): packed 4×i8
/// - `b` (u32): packed 4×i8
/// - `c` (i32): accumulator
///
/// # Results
///
/// - `d` (i32): accumulated dot product
#[pliron_op(
    name = "nvvm.dp4a_s32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct Dp4aS32Op;

impl Dp4aS32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Dp4aS32Op { op }
    }
}

/// Unsigned 4-element byte dot product with accumulation: `d = c + dot(a, b)`.
///
/// `a` and `b` are each 4 packed unsigned bytes; `c` and `d` are unsigned 32-bit.
///
/// PTX: `dp4a.u32.u32 $0, $1, $2, $3;`  (requires `sm_61+`)
///
/// # Operands
///
/// - `a` (u32): packed 4×u8
/// - `b` (u32): packed 4×u8
/// - `c` (u32): accumulator
///
/// # Results
///
/// - `d` (u32): accumulated dot product
#[pliron_op(
    name = "nvvm.dp4a_u32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct Dp4aU32Op;

impl Dp4aU32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Dp4aU32Op { op }
    }
}

/// Signed 2-element half-word × byte dot product (lower half): `d = c + dot(a, b)`.
///
/// `a` is 2 packed signed 16-bit values; `b`'s lower 2 bytes are used.
///
/// PTX: `dp2a.lo.s32.s32 $0, $1, $2, $3;`  (requires `sm_61+`)
///
/// # Operands
///
/// - `a` (u32): packed 2×i16
/// - `b` (u32): packed bytes (lower 2 used)
/// - `c` (i32): accumulator
///
/// # Results
///
/// - `d` (i32): accumulated dot product
#[pliron_op(
    name = "nvvm.dp2a_s32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct Dp2aS32Op;

impl Dp2aS32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Dp2aS32Op { op }
    }
}

/// Unsigned 2-element half-word × byte dot product (lower half): `d = c + dot(a, b)`.
///
/// `a` is 2 packed unsigned 16-bit values; `b`'s lower 2 bytes are used.
///
/// PTX: `dp2a.lo.u32.u32 $0, $1, $2, $3;`  (requires `sm_61+`)
///
/// # Operands
///
/// - `a` (u32): packed 2×u16
/// - `b` (u32): packed bytes (lower 2 used)
/// - `c` (u32): accumulator
///
/// # Results
///
/// - `d` (u32): accumulated dot product
#[pliron_op(
    name = "nvvm.dp2a_u32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct Dp2aU32Op;

impl Dp2aU32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Dp2aU32Op { op }
    }
}

/// Register dot product operations with the context.
pub(super) fn register(ctx: &mut Context) {
    Dp4aS32Op::register(ctx);
    Dp4aU32Op::register(ctx);
    Dp2aS32Op::register(ctx);
    Dp2aU32Op::register(ctx);
}
