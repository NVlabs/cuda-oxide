/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix operations (WMMA).
//!
//! In-register matrix transpose using `movmatrix.sync.aligned` instructions.

/// Transpose an 8×8 matrix of b16 elements in-register across the warp.
///
/// Each lane provides one `u32` that packs two b16 elements of the source
/// matrix. The instruction collectively transposes the 8×8 tile and writes
/// the transposed pair back into each lane's destination register.
///
/// # PTX
///
/// `movmatrix.sync.aligned.m8n8.trans.b16 %d, %a;`
///
/// # Safety
///
/// - Warp-synchronous: all 32 lanes must execute this call together
/// - Calling from divergent control flow is undefined behaviour
/// - Requires sm_90+ (Hopper architecture)
#[inline(never)]
pub unsafe fn movmatrix_trans_b16(a: u32) -> u32 {
    let _ = a;
    unreachable!("movmatrix_trans_b16 called outside CUDA kernel context")
}
