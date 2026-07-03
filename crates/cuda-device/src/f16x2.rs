// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Packed `f16x2` arithmetic intrinsics.
//!
//! Maxwell/Pascal (`sm_53+`) supports packed add, subtract, multiply, FMA,
//! negation, and absolute value on `f16x2` data. Ampere (`sm_80+`) added
//! packed min, max, and fused multiply-add with ReLU.
//!
//! Each `u32` carries two f16 values: low 16 bits = first lane, high 16 bits
//! = second lane.

/// Packed f16x2 addition: `d = a + b`.
///
/// Both operands and the result are packed `f16x2` carried as `u32`.
///
/// # PTX
///
/// ```ptx
/// add.rn.f16x2 %d, %a, %b;
/// ```
///
/// # Supported on
///
/// - `sm_53+` (Maxwell onwards).
#[inline(never)]
#[must_use]
pub fn add_f16x2(a: u32, b: u32) -> u32 {
    let _ = (a, b);
    unreachable!("add_f16x2 called outside CUDA kernel context")
}

/// Packed f16x2 subtraction: `d = a - b`.
///
/// Both operands and the result are packed `f16x2` carried as `u32`.
///
/// # PTX
///
/// ```ptx
/// sub.rn.f16x2 %d, %a, %b;
/// ```
///
/// # Supported on
///
/// - `sm_53+` (Maxwell onwards).
#[inline(never)]
#[must_use]
pub fn sub_f16x2(a: u32, b: u32) -> u32 {
    let _ = (a, b);
    unreachable!("sub_f16x2 called outside CUDA kernel context")
}

/// Packed f16x2 multiplication: `d = a * b`.
///
/// Both operands and the result are packed `f16x2` carried as `u32`.
///
/// # PTX
///
/// ```ptx
/// mul.rn.f16x2 %d, %a, %b;
/// ```
///
/// # Supported on
///
/// - `sm_53+` (Maxwell onwards).
#[inline(never)]
#[must_use]
pub fn mul_f16x2(a: u32, b: u32) -> u32 {
    let _ = (a, b);
    unreachable!("mul_f16x2 called outside CUDA kernel context")
}

/// Packed f16x2 fused multiply-add: `d = a * b + c`.
///
/// All three operands and the result are packed `f16x2` carried as `u32`.
///
/// # PTX
///
/// ```ptx
/// fma.rn.f16x2 %d, %a, %b, %c;
/// ```
///
/// # Supported on
///
/// - `sm_53+` (Maxwell onwards).
#[inline(never)]
#[must_use]
pub fn fma_f16x2(a: u32, b: u32, c: u32) -> u32 {
    let _ = (a, b, c);
    unreachable!("fma_f16x2 called outside CUDA kernel context")
}

/// Packed f16x2 negation: `d = -a`.
///
/// The operand and result are packed `f16x2` carried as `u32`.
///
/// # PTX
///
/// ```ptx
/// neg.f16x2 %d, %a;
/// ```
///
/// # Supported on
///
/// - `sm_53+` (Maxwell onwards).
#[inline(never)]
#[must_use]
pub fn neg_f16x2(a: u32) -> u32 {
    let _ = a;
    unreachable!("neg_f16x2 called outside CUDA kernel context")
}

/// Packed f16x2 absolute value: `d = |a|`.
///
/// The operand and result are packed `f16x2` carried as `u32`.
///
/// # PTX
///
/// ```ptx
/// abs.f16x2 %d, %a;
/// ```
///
/// # Supported on
///
/// - `sm_53+` (Maxwell onwards).
#[inline(never)]
#[must_use]
pub fn abs_f16x2(a: u32) -> u32 {
    let _ = a;
    unreachable!("abs_f16x2 called outside CUDA kernel context")
}

/// Packed f16x2 minimum: `d = min(a, b)`.
///
/// Both operands and the result are packed `f16x2` carried as `u32`.
///
/// # PTX
///
/// ```ptx
/// min.f16x2 %d, %a, %b;
/// ```
///
/// # Supported on
///
/// - `sm_80+` (Ampere onwards).
#[inline(never)]
#[must_use]
pub fn min_f16x2(a: u32, b: u32) -> u32 {
    let _ = (a, b);
    unreachable!("min_f16x2 called outside CUDA kernel context")
}

/// Packed f16x2 maximum: `d = max(a, b)`.
///
/// Both operands and the result are packed `f16x2` carried as `u32`.
///
/// # PTX
///
/// ```ptx
/// max.f16x2 %d, %a, %b;
/// ```
///
/// # Supported on
///
/// - `sm_80+` (Ampere onwards).
#[inline(never)]
#[must_use]
pub fn max_f16x2(a: u32, b: u32) -> u32 {
    let _ = (a, b);
    unreachable!("max_f16x2 called outside CUDA kernel context")
}

/// Fused multiply-add with ReLU: `max(0, a * b + c)` on packed f16x2 values.
///
/// Each `u32` carries two packed f16 values. The operation computes
/// `fma.rn.relu.f16x2`, applying ReLU (clamp-to-zero) after the FMA.
///
/// # PTX
///
/// ```ptx
/// fma.rn.relu.f16x2 %d, %a, %b, %c;
/// ```
///
/// # Supported on
///
/// - `sm_80+` (Ampere onwards).
#[inline(never)]
#[must_use]
pub fn fma_relu_f16x2(a: u32, b: u32, c: u32) -> u32 {
    let _ = (a, b, c);
    unreachable!("fma_relu_f16x2 called outside CUDA kernel context")
}
