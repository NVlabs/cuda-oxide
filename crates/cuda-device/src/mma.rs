/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level Matrix Multiply-Accumulate (mma.sync) for SM80+.
//!
//! Issues a single PTX `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
//! instruction. Per-lane fragment ABI (warp = 32 lanes, 16x8 = 128 outputs):
//!
//! - A: 4 x b32 per lane (each holds 2 packed bf16) -- 16x16 tile total
//! - B: 2 x b32 per lane (each holds 2 packed bf16) -- 16x8 tile total
//! - C: 4 x f32 per lane                              -- 16x8 accumulator
//! - D: 4 x f32 per lane (output)                     -- 16x8 result
//!
//! All 32 lanes in the warp must execute together with consistent inputs.

use crate::cusimd::CuSimd;

/// 4 packed-bf16 register pairs for matrix A (per lane).
pub type MmaABf16 = CuSimd<u32, 4>;

/// 2 packed-bf16 register pairs for matrix B (per lane).
pub type MmaBBf16 = CuSimd<u32, 2>;

/// 4 f32 accumulator registers (per lane).
pub type MmaAccF32 = CuSimd<f32, 4>;

/// Warp MMA: D = A * B + C using `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
///
/// All 32 lanes in the warp must call this together. A is stored in row-major,
/// B in col-major (per the PTX spec for this instruction).
///
/// # Safety
///
/// - Must be called by all 32 lanes of the warp simultaneously
/// - Must be invoked from a CUDA kernel context targeting sm_80+
#[inline(never)]
pub unsafe fn mma_m16n8k16_bf16_f32(
    a0: u32,
    a1: u32,
    a2: u32,
    a3: u32,
    b0: u32,
    b1: u32,
    c0: f32,
    c1: f32,
    c2: f32,
    c3: f32,
) -> MmaAccF32 {
    let _ = (a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
    unreachable!("mma_m16n8k16_bf16_f32 called outside CUDA kernel context")
}
