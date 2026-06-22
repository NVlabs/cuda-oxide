/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level Matrix Multiply-Accumulate (mma.sync) for Ampere+ architectures.
//!
//! Binary (1-bit) MMA variants using XOR-POPC accumulation.

/// Warp MMA: D = A xor_popc B + C (m16n8k128, s32 output, b1 inputs).
///
/// Performs a 16x8x128 binary matrix multiplication using tensor cores with
/// 1-bit inputs and 32-bit integer accumulator. The operation computes
/// XOR followed by population count (POPC) as the accumulation primitive.
/// All 32 threads in the warp participate.
///
/// # Matrix Dimensions
///
/// - **A**: 16x128 (row-major, b1), distributed as 2 x u32 per thread (each u32 = 32 bits)
/// - **B**: 128x8 (col-major, b1), distributed as 1 x u32 per thread (each u32 = 32 bits)
/// - **D/C**: 16x8 (s32 accumulator), distributed as 4 x i32 per thread
///
/// # Parameters
///
/// - `acc`: Mutable accumulator (4 x i32 per thread, read-modify-write: D = A xor_popc B + acc)
/// - `a`: A fragment (2 x u32, each u32 contains 32 packed b1 values)
/// - `b`: B fragment (1 x u32, contains 32 packed b1 values)
///
/// # PTX
///
/// ```ptx
/// mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc
///     {%d0, %d1, %d2, %d3},
///     {%a0, %a1},
///     {%b0},
///     {%c0, %c1, %c2, %c3};
/// ```
///
/// # Safety
///
/// - Must be called by all threads in a warp
/// - Must be called from within a CUDA kernel context on sm_80+
/// - Fragment values must be correctly distributed across warp lanes
#[inline(never)]
pub unsafe fn mma_m16n8k128_s32_b1(acc: &mut [i32; 4], a: &[u32; 2], b: &u32) {
    let _ = (acc, a, b);
    unreachable!("mma_m16n8k128_s32_b1 called outside CUDA kernel context")
}

/// Warp MMA: D = A xor_popc B + C (m16n8k256, s32 output, b1 inputs).
///
/// Performs a 16x8x256 binary matrix multiplication using tensor cores with
/// 1-bit inputs and 32-bit integer accumulator. The operation computes
/// XOR followed by population count (POPC) as the accumulation primitive.
/// All 32 threads in the warp participate.
///
/// # Matrix Dimensions
///
/// - **A**: 16x256 (row-major, b1), distributed as 4 x u32 per thread (each u32 = 32 bits)
/// - **B**: 256x8 (col-major, b1), distributed as 2 x u32 per thread (each u32 = 32 bits)
/// - **D/C**: 16x8 (s32 accumulator), distributed as 4 x i32 per thread
///
/// # Parameters
///
/// - `acc`: Mutable accumulator (4 x i32 per thread, read-modify-write: D = A xor_popc B + acc)
/// - `a`: A fragment (4 x u32, each u32 contains 32 packed b1 values)
/// - `b`: B fragment (2 x u32, each u32 contains 32 packed b1 values)
///
/// # PTX
///
/// ```ptx
/// mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc
///     {%d0, %d1, %d2, %d3},
///     {%a0, %a1, %a2, %a3},
///     {%b0, %b1},
///     {%c0, %c1, %c2, %c3};
/// ```
///
/// # Safety
///
/// - Must be called by all threads in a warp
/// - Must be called from within a CUDA kernel context on sm_80+
/// - Fragment values must be correctly distributed across warp lanes
#[inline(never)]
pub unsafe fn mma_m16n8k256_s32_b1(acc: &mut [i32; 4], a: &[u32; 4], b: &[u32; 2]) {
    let _ = (acc, a, b);
    unreachable!("mma_m16n8k256_s32_b1 called outside CUDA kernel context")
}
