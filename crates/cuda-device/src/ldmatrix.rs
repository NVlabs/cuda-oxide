/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-cooperative `ldmatrix.sync.aligned.m8n8.x4` (sm_75+).
//!
//! Loads four 8x8 b16 matrices from shared memory into the per-lane register
//! layout consumed by `mma.sync.aligned.m16n8k16` (and friends).
//!
//! Per-lane fragment: 4 x b32 outputs (each holding 2 packed b16 elements).
//! All 32 lanes in the warp must participate; lanes 0-7 each provide a row
//! address, fanning out to 8 rows x 4 matrices = 32 8-byte rows total.
//!
//! # Hardware Support
//!
//! - **sm_75 (Turing)**: T4, RTX 2080
//! - **sm_80 (Ampere)**: A100, A30, A40
//! - **sm_86 (Ampere)**: GA10x consumer
//! - **sm_89 (Ada)**: AD10x (RTX 4090, L40, etc.)
//! - **sm_90 (Hopper)**: H100, H200
//! - **sm_100 (Blackwell datacenter)**: B100, B200
//! - **sm_120 (Blackwell consumer/workstation)**: RTX 5090, RTX PRO 6000 Blackwell
//!
//! Verified on sm_120 (RTX PRO 6000 Blackwell).

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
