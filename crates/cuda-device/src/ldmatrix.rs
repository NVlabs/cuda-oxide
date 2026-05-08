/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-cooperative `ldmatrix.sync.aligned.m8n8.x4` (SM75+).
//!
//! Loads four 8x8 b16 matrices from shared memory into the per-lane register
//! layout consumed by `mma.sync.aligned.m16n8k16` (and friends).
//!
//! Per-lane fragment: 4 x b32 outputs (each holding 2 packed b16 elements).
//! All 32 lanes in the warp must participate; lanes 0-7 each provide a row
//! address, fanning out to 8 rows x 4 matrices = 32 8-byte rows total.

use crate::cusimd::CuSimd;

/// Per-lane fragment returned by `ldmatrix.x4`: 4 x b32 register values.
pub type LdmatrixFragx4 = CuSimd<u32, 4>;

/// Load four 8x8 b16 matrices from shared memory into per-lane register fragments.
///
/// Issues a single PTX `ldmatrix.sync.aligned.m8n8.x4.shared.b16`.
///
/// # Safety
///
/// - `smem_ptr` must point to valid shared memory.
/// - All 32 lanes in the warp must call this together.
/// - Lanes 0-7 must each pass a per-row shared address; lanes 8-31 contribute
///   their addresses for the additional 8x8 tiles per the PTX spec.
#[inline(never)]
pub unsafe fn ldmatrix_x4_b16(smem_ptr: *const u8) -> LdmatrixFragx4 {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x4_b16 called outside CUDA kernel context")
}

/// Load four 8x8 b16 matrices, transposed during the load.
///
/// Issues a single PTX `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16`.
///
/// # Safety
///
/// Same warp-cooperative requirements as [`ldmatrix_x4_b16`].
#[inline(never)]
pub unsafe fn ldmatrix_x4_trans_b16(smem_ptr: *const u8) -> LdmatrixFragx4 {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x4_trans_b16 called outside CUDA kernel context")
}
