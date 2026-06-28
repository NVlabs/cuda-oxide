// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Packed `bf16x2` arithmetic intrinsics.
//!
//! Ampere (`sm_80+`) supports packed FMA, min, max, negation, and absolute
//! value. Packed add, subtract, and multiply were added for Hopper (`sm_90+`).
//! On Ampere, callers can still express add and multiply through FMA with a
//! packed `1.0` or `0.0` operand.
//!
//! Each `u32` carries two bf16 values: low 16 bits = first lane, high 16 bits
//! = second lane. This matches the layout produced by
//! [`crate::tcgen05::cvt_f32x2_bf16x2`].

/// Packed bf16x2 fused multiply-add: `d = a * b + c`.
///
/// All three operands and the result are packed `bf16x2` carried as `u32`,
/// matching `cvt.rn.bf16x2.f32`'s output layout (low 16 = first lane, high 16
/// = second lane).
///
/// # PTX
///
/// ```ptx
/// fma.rn.bf16x2 %d, %a, %b, %c;
/// ```
///
/// # Supported on
///
/// - `sm_80+` (Ampere onwards). On `sm_70`/`sm_75` this lowering will be
///   rejected by `ptxas`.
///
/// # Notes
///
/// `add.bf16x2` and `mul.bf16x2` require `sm_90+`. To get a hardware packed
/// add on Ampere, build the operation as `fma(a, ONE_BF16X2, b)` where
/// `ONE_BF16X2 = 0x3F803F80u32` encodes packed (1.0, 1.0).
#[inline(never)]
pub fn fma_bf16x2(a: u32, b: u32, c: u32) -> u32 {
    let _ = (a, b, c);
    unreachable!("fma_bf16x2 called outside CUDA kernel context")
}

/// Packed bf16x2 addition: `d = a + b`.
///
/// Both operands and the result are packed `bf16x2` carried as `u32`.
///
/// # PTX
///
/// ```ptx
/// add.rn.bf16x2 %d, %a, %b;
/// ```
///
/// # Supported on
///
/// - `sm_90+` (Hopper onwards).
#[inline(never)]
pub fn add_bf16x2(a: u32, b: u32) -> u32 {
    let _ = (a, b);
    unreachable!("add_bf16x2 called outside CUDA kernel context")
}

/// Packed bf16x2 subtraction: `d = a - b`.
///
/// Both operands and the result are packed `bf16x2` carried as `u32`.
///
/// # PTX
///
/// ```ptx
/// sub.rn.bf16x2 %d, %a, %b;
/// ```
///
/// # Supported on
///
/// - `sm_90+` (Hopper onwards).
#[inline(never)]
pub fn sub_bf16x2(a: u32, b: u32) -> u32 {
    let _ = (a, b);
    unreachable!("sub_bf16x2 called outside CUDA kernel context")
}

/// Packed bf16x2 multiplication: `d = a * b`.
///
/// Both operands and the result are packed `bf16x2` carried as `u32`.
///
/// # PTX
///
/// ```ptx
/// mul.rn.bf16x2 %d, %a, %b;
/// ```
///
/// # Supported on
///
/// - `sm_90+` (Hopper onwards).
#[inline(never)]
pub fn mul_bf16x2(a: u32, b: u32) -> u32 {
    let _ = (a, b);
    unreachable!("mul_bf16x2 called outside CUDA kernel context")
}

/// Packed bf16x2 minimum: `d = min(a, b)`.
///
/// Both operands and the result are packed `bf16x2` carried as `u32`.
///
/// # PTX
///
/// ```ptx
/// min.bf16x2 %d, %a, %b;
/// ```
///
/// # Supported on
///
/// - `sm_80+` (Ampere onwards).
#[inline(never)]
pub fn min_bf16x2(a: u32, b: u32) -> u32 {
    let _ = (a, b);
    unreachable!("min_bf16x2 called outside CUDA kernel context")
}

/// Packed bf16x2 maximum: `d = max(a, b)`.
///
/// Both operands and the result are packed `bf16x2` carried as `u32`.
///
/// # PTX
///
/// ```ptx
/// max.bf16x2 %d, %a, %b;
/// ```
///
/// # Supported on
///
/// - `sm_80+` (Ampere onwards).
#[inline(never)]
pub fn max_bf16x2(a: u32, b: u32) -> u32 {
    let _ = (a, b);
    unreachable!("max_bf16x2 called outside CUDA kernel context")
}

/// Packed bf16x2 negation: `d = -a`.
///
/// The operand and result are packed `bf16x2` carried as `u32`.
///
/// # PTX
///
/// ```ptx
/// neg.bf16x2 %d, %a;
/// ```
///
/// # Supported on
///
/// - `sm_80+` (Ampere onwards).
#[inline(never)]
pub fn neg_bf16x2(a: u32) -> u32 {
    let _ = a;
    unreachable!("neg_bf16x2 called outside CUDA kernel context")
}

/// Packed bf16x2 absolute value: `d = |a|`.
///
/// The operand and result are packed `bf16x2` carried as `u32`.
///
/// # PTX
///
/// ```ptx
/// abs.bf16x2 %d, %a;
/// ```
///
/// # Supported on
///
/// - `sm_80+` (Ampere onwards).
#[inline(never)]
pub fn abs_bf16x2(a: u32) -> u32 {
    let _ = a;
    unreachable!("abs_bf16x2 called outside CUDA kernel context")
}

/// Fused multiply-add with ReLU: `max(0, a * b + c)` on packed bf16x2 values.
///
/// Each `u32` carries two packed bf16 values. The operation computes
/// `fma.rn.relu.bf16x2`, applying ReLU (clamp-to-zero) after the FMA.
///
/// # PTX
///
/// ```ptx
/// fma.rn.relu.bf16x2 %d, %a, %b, %c;
/// ```
///
/// # Supported on
///
/// - `sm_80+` (Ampere onwards).
#[inline(never)]
pub fn fma_relu_bf16x2(a: u32, b: u32, c: u32) -> u32 {
    let _ = (a, b, c);
    unreachable!("fma_relu_bf16x2 called outside CUDA kernel context")
}
