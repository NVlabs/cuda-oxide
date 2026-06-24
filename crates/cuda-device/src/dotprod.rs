// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integer dot product intrinsics (`dp4a`, `dp2a`).
//!
//! These instructions perform packed-byte or packed-half dot products with
//! accumulation, useful for integer quantised inference on Ampere+ GPUs.
//!
//! # `dp4a`, 4-element byte dot product
//!
//! Treats `a` and `b` as vectors of 4 packed bytes, multiplies corresponding
//! elements, sums the products, and adds the scalar accumulator `c`:
//!
//! ```text
//! d = c + a.byte0*b.byte0 + a.byte1*b.byte1 + a.byte2*b.byte2 + a.byte3*b.byte3
//! ```
//!
//! # `dp2a`, 2-element half-word × byte dot product
//!
//! Treats `a` as two packed 16-bit values and `b` as packed bytes (lower 2
//! bytes selected by the `.lo` qualifier):
//!
//! ```text
//! d = c + a.half0*b.byte0 + a.half1*b.byte1
//! ```
//!
//! # Supported on
//!
//! - `sm_61+` (`dp4a`, `dp2a`)

/// 4-element signed byte dot product with accumulation.
///
/// Interprets `a` and `b` as 4 packed **signed** bytes each, computes the
/// dot product, and adds the signed 32-bit accumulator `c`.
///
/// # PTX
///
/// ```ptx
/// dp4a.s32.s32 %d, %a, %b, %c;
/// ```
#[inline(never)]
pub fn dp4a_s32(a: u32, b: u32, c: i32) -> i32 {
    let _ = (a, b, c);
    unreachable!("dp4a_s32 called outside CUDA kernel context")
}

/// 4-element unsigned byte dot product with accumulation.
///
/// Interprets `a` and `b` as 4 packed **unsigned** bytes each, computes the
/// dot product, and adds the unsigned 32-bit accumulator `c`.
///
/// # PTX
///
/// ```ptx
/// dp4a.u32.u32 %d, %a, %b, %c;
/// ```
#[inline(never)]
pub fn dp4a_u32(a: u32, b: u32, c: u32) -> u32 {
    let _ = (a, b, c);
    unreachable!("dp4a_u32 called outside CUDA kernel context")
}

/// 2-element signed half-word × byte dot product with accumulation (lower half).
///
/// Interprets `a` as 2 packed **signed** 16-bit values and `b`'s lower 2
/// bytes as signed values. Computes `d = c + a.half0*b.byte0 + a.half1*b.byte1`.
///
/// # PTX
///
/// ```ptx
/// dp2a.lo.s32.s32 %d, %a, %b, %c;
/// ```
#[inline(never)]
pub fn dp2a_s32(a: u32, b: u32, c: i32) -> i32 {
    let _ = (a, b, c);
    unreachable!("dp2a_s32 called outside CUDA kernel context")
}

/// 2-element unsigned half-word × byte dot product with accumulation (lower half).
///
/// Interprets `a` as 2 packed **unsigned** 16-bit values and `b`'s lower 2
/// bytes as unsigned values. Computes `d = c + a.half0*b.byte0 + a.half1*b.byte1`.
///
/// # PTX
///
/// ```ptx
/// dp2a.lo.u32.u32 %d, %a, %b, %c;
/// ```
#[inline(never)]
pub fn dp2a_u32(a: u32, b: u32, c: u32) -> u32 {
    let _ = (a, b, c);
    unreachable!("dp2a_u32 called outside CUDA kernel context")
}
