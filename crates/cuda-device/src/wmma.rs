/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix operations.
//!
//! This module provides in-register matrix transpose and related warp-cooperative
//! matrix operations that operate on data distributed across a warp's register file.
//!
//! # `movmatrix` — In-Register 8×8 Transpose
//!
//! The `movmatrix.sync.aligned.m8n8.trans.b16` instruction transposes an 8×8 matrix
//! of 16-bit elements held collectively in the registers of a warp. Each thread
//! contributes one `u32` register (packing 2 × b16 elements); after the transpose,
//! each thread's register holds the transposed pair.
//!
//! This avoids a shared-memory round-trip when the only goal is to change the
//! layout of a warp-distributed matrix fragment (e.g., switching between row-major
//! and column-major for chained MMA operations).
//!
//! # Requirements
//!
//! - **sm_90+** (Hopper and later).
//! - Warp-synchronous: all 32 threads must participate.

/// Transpose an 8×8 matrix of b16 elements in-register across the warp.
///
/// PTX: `movmatrix.sync.aligned.m8n8.trans.b16 %d, %a;`
///
/// Each lane provides one `u32` that packs two b16 elements of the source
/// matrix. The instruction collectively transposes the 8×8 tile and writes
/// the transposed pair back into each lane's destination register.
///
/// # Safety
///
/// This is a warp-synchronous operation. All 32 lanes of the warp must
/// execute this call together. Calling from divergent control flow is
/// undefined behaviour.
///
/// # Example
///
/// ```rust,ignore
/// use cuda_device::wmma;
///
/// // Each thread holds two packed b16 values from an 8×8 tile.
/// let transposed = unsafe { wmma::movmatrix_trans_b16(my_packed_b16x2) };
/// ```
#[inline(never)]
pub unsafe fn movmatrix_trans_b16(a: u32) -> u32 {
    let _ = a;
    unreachable!("movmatrix_trans_b16 called outside CUDA kernel context")
}
