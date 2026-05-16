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
/// # Arguments
/// - `lo`: f32 value for the low 16 bits (bits [15:0])
/// - `hi`: f32 value for the high 16 bits (bits [31:16])
///
/// # Returns
/// A u32 containing two packed f16 values.
#[inline(never)]
pub fn cvt_f16x2_f32(lo: f32, hi: f32) -> u32 {
    let _ = (lo, hi);
    unreachable!("cvt_f16x2_f32 called outside CUDA kernel context")
}
