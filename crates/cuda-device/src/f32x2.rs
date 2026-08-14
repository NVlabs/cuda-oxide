/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Native SM100 two-lane `f32` arithmetic.
//!
//! Unlike [`crate::vector::F32x2`], which is an over-aligned memory element,
//! these operations consume and produce [`crate::Float2`] register groups and
//! lower to one `fma.rn.f32x2` or `add.rn.f32x2` PTX instruction.

use crate::{Float2, ptx_asm};

/// Pack two scalar f32 values into the PTX f32x2 register representation.
#[inline(always)]
pub fn pack_f32x2(lo: f32, hi: f32) -> u64 {
    let result: u64;
    unsafe {
        ptx_asm!(
            "mov.b64 %0, {%1, %2};",
            out("=l") result,
            in("f") lo,
            in("f") hi,
            options(register_only),
        );
    }
    result
}

/// Return the low scalar lane from a packed PTX f32x2 register.
#[inline(always)]
pub fn unpack_f32x2_lo(value: u64) -> f32 {
    let result: f32;
    unsafe {
        ptx_asm!(
            "mov.b64 {%0, _}, %1;",
            out("=f") result,
            in("l") value,
            options(register_only),
        );
    }
    result
}

/// Return the high scalar lane from a packed PTX f32x2 register.
#[inline(always)]
pub fn unpack_f32x2_hi(value: u64) -> f32 {
    let result: f32;
    unsafe {
        ptx_asm!(
            "mov.b64 {_, %0}, %1;",
            out("=f") result,
            in("l") value,
            options(register_only),
        );
    }
    result
}

/// Add the two scalar lanes of a packed PTX f32x2 register.
#[inline(always)]
pub fn horizontal_add_f32x2_packed(value: u64) -> f32 {
    let result: f32;
    unsafe {
        ptx_asm!(
            "{ .reg .f32 lo, hi; mov.b64 {lo, hi}, %1; add.f32 %0, lo, hi; }",
            out("=f") result,
            in("l") value,
            options(register_only),
        );
    }
    result
}

/// Packed-register form of [`fma_f32x2`].
#[inline(always)]
pub fn fma_f32x2_packed(a: u64, b: u64, c: u64) -> u64 {
    let result: u64;
    unsafe {
        ptx_asm!(
            "fma.rn.f32x2 %0, %1, %2, %3;",
            out("=l") result,
            in("l") a,
            in("l") b,
            in("l") c,
            options(register_only),
        );
    }
    result
}

/// Packed-register form of [`add_f32x2`].
#[inline(always)]
pub fn add_f32x2_packed(a: u64, b: u64) -> u64 {
    let result: u64;
    unsafe {
        ptx_asm!(
            "add.rn.f32x2 %0, %1, %2;",
            out("=l") result,
            in("l") a,
            in("l") b,
            options(register_only),
        );
    }
    result
}

/// Two-lane round-to-nearest-even fused multiply-add (`a * b + c`).
#[inline(always)]
pub fn fma_f32x2(a: Float2, b: Float2, c: Float2) -> Float2 {
    // PTX models f32x2 as one 64-bit register pair. Keep that representation
    // at the asm boundary so ptxas can reuse the input/output pairs directly.
    let result = fma_f32x2_packed(
        pack_f32x2(a.x(), a.y()),
        pack_f32x2(b.x(), b.y()),
        pack_f32x2(c.x(), c.y()),
    );
    Float2::new([unpack_f32x2_lo(result), unpack_f32x2_hi(result)])
}

/// Two-lane round-to-nearest-even addition.
#[inline(always)]
pub fn add_f32x2(a: Float2, b: Float2) -> Float2 {
    let result = add_f32x2_packed(pack_f32x2(a.x(), a.y()), pack_f32x2(b.x(), b.y()));
    Float2::new([unpack_f32x2_lo(result), unpack_f32x2_hi(result)])
}
