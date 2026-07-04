/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Type conversion intrinsics.
//!
//! These intrinsics provide access to PTX type conversion instructions that
//! are more efficient than scalar Rust casts.

/// Convert two f32 values to a packed f16x2 (u32) in a single instruction.
///
/// This is equivalent to:
/// ```ignore
/// ((lo as f16).to_bits() as u32) | (((hi as f16).to_bits() as u32) << 16)
/// ```
/// but compiles to a single `cvt.rn.f16x2.f32` PTX instruction instead of
/// two separate f32→f16 conversions plus bit manipulation.
///
/// Maps to PTX: `cvt.rn.f16x2.f32 d, hi, lo;`
///
/// Lane placement: the first argument (`lo`) fills bits `[15:0]` and the
/// second argument (`hi`) fills bits `[31:16]`, even though the PTX
/// operand list prints `hi` first. This is the same first-arg-low
/// convention as [`cvt_f32x2_bf16x2`](crate::tcgen05::cvt_f32x2_bf16x2),
/// which differs only in its destination element type (bf16, not f16)
/// and its `a`/`b` argument naming.
///
/// # Arguments
/// - `lo`: f32 value for the low 16 bits (bits `[15:0]`)
/// - `hi`: f32 value for the high 16 bits (bits `[31:16]`)
///
/// # Returns
/// A u32 containing two packed f16 values.
#[inline(never)]
pub fn cvt_f16x2_f32(lo: f32, hi: f32) -> u32 {
    let _ = (lo, hi);
    unreachable!("cvt_f16x2_f32 called outside CUDA kernel context")
}

/// Convert two f32 values to a packed f16x2 (u32) with truncation rounding.
///
/// Uses round-toward-zero (truncation) instead of the default
/// round-to-nearest-even mode.
///
/// Maps to PTX: `cvt.rz.f16x2.f32 d, hi, lo;`
///
/// # Arguments
/// - `lo`: f32 value for the low 16 bits (bits `[15:0]`)
/// - `hi`: f32 value for the high 16 bits (bits `[31:16]`)
///
/// # Returns
/// A u32 containing two packed f16 values (truncation rounded).
#[inline(never)]
pub fn cvt_rz_f16x2_f32(lo: f32, hi: f32) -> u32 {
    let _ = (lo, hi);
    unreachable!("cvt_rz_f16x2_f32 called outside CUDA kernel context")
}

/// Convert two f32 values to a packed f16x2 (u32) with fused ReLU.
///
/// Negative values are clamped to zero before packing.
///
/// Maps to PTX: `cvt.rn.relu.f16x2.f32 d, hi, lo;`
///
/// # Arguments
/// - `lo`: f32 value for the low 16 bits (bits `[15:0]`)
/// - `hi`: f32 value for the high 16 bits (bits `[31:16]`)
///
/// # Returns
/// A u32 containing two packed f16 values (ReLU applied).
#[inline(never)]
pub fn cvt_rn_relu_f16x2_f32(lo: f32, hi: f32) -> u32 {
    let _ = (lo, hi);
    unreachable!("cvt_rn_relu_f16x2_f32 called outside CUDA kernel context")
}

/// Convert two f32 values to a packed bf16x2 (u32) with fused ReLU.
///
/// Negative values are clamped to zero before packing.
///
/// Maps to PTX: `cvt.rn.relu.bf16x2.f32 d, hi, lo;`
///
/// # Arguments
/// - `lo`: f32 value for the low 16 bits (bits `[15:0]`)
/// - `hi`: f32 value for the high 16 bits (bits `[31:16]`)
///
/// # Returns
/// A u32 containing two packed bf16 values (ReLU applied).
#[inline(never)]
pub fn cvt_rn_relu_bf16x2_f32(lo: f32, hi: f32) -> u32 {
    let _ = (lo, hi);
    unreachable!("cvt_rn_relu_bf16x2_f32 called outside CUDA kernel context")
}

/// Convert two f32 values to a packed bf16x2 (u32) with truncation rounding.
///
/// Uses round-toward-zero (truncation) instead of the default
/// round-to-nearest-even mode.
///
/// Maps to PTX: `cvt.rz.bf16x2.f32 d, hi, lo;`
///
/// # Arguments
/// - `lo`: f32 value for the low 16 bits (bits `[15:0]`)
/// - `hi`: f32 value for the high 16 bits (bits `[31:16]`)
///
/// # Returns
/// A u32 containing two packed bf16 values (truncation rounded).
#[inline(never)]
pub fn cvt_rz_bf16x2_f32(lo: f32, hi: f32) -> u32 {
    let _ = (lo, hi);
    unreachable!("cvt_rz_bf16x2_f32 called outside CUDA kernel context")
}

/// Convert two f32 values to a packed e4m3x2 (u16) FP8 pair.
///
/// Each f32 is converted to an `e4m3` FP8 value (1 sign / 4 exponent /
/// 3 mantissa bits) using round-to-nearest-even, with `satfinite`
/// saturation: inputs whose magnitude exceeds the largest finite e4m3
/// value (±448) are clamped to that maximum instead of overflowing to
/// infinity or NaN. The two 8-bit results are packed into a 16-bit
/// register.
///
/// Requires `sm_89`+ (Ada Lovelace) and PTX ISA 8.1+.
///
/// Maps to PTX: `cvt.rn.satfinite.e4m3x2.f32 d, hi, lo;`
///
/// Lane placement: the first argument (`lo`) fills bits `[7:0]` and the
/// second argument (`hi`) fills bits `[15:8]`, even though the PTX
/// operand list prints `hi` first. This is the same first-arg-low
/// convention as [`cvt_f16x2_f32`], differing only in the destination
/// element type (8-bit e4m3 instead of 16-bit f16).
///
/// # Arguments
/// - `lo`: f32 value for the low 8 bits (bits `[7:0]`)
/// - `hi`: f32 value for the high 8 bits (bits `[15:8]`)
///
/// # Returns
/// A u16 containing two packed e4m3 FP8 values.
#[inline(never)]
pub fn cvt_rn_satfinite_e4m3x2_f32(lo: f32, hi: f32) -> u16 {
    let _ = (lo, hi);
    unreachable!("cvt_rn_satfinite_e4m3x2_f32 called outside CUDA kernel context")
}
