/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level matrix operations.
//!
//! This module provides a register-only 8×8 transpose (`movmatrix`) and
//! warp-cooperative shared-memory loads (`ldmatrix`).
//!
//! For ldmatrix, each group of four lanes loads one naturally aligned
//! 16-byte row:
//!
//! ```text
//! x1: lanes  0..7  provide addresses
//! x2: lanes  0..15 provide addresses
//! x4: lanes  0..31 provide addresses
//! ```
//!
//! On sm_75, x1 and x2 still require valid addresses in all 32 lanes. A
//! common choice is to copy the lower-lane addresses into the upper lanes.
//! The .trans forms use column-major rather than row-major layout.
//!
//! Ldmatrix is a weak memory operation: .sync converges the warp but does not
//! order memory. Callers need an appropriate barrier or fence around dependent
//! memory accesses. Movmatrix is register-only and has no memory effect.

/// Transpose an 8×8 matrix of b16 elements in-register across the warp.
///
/// Each lane provides one `u32` that packs two b16 elements of the source
/// matrix. The instruction collectively transposes the 8×8 tile and writes
/// the transposed pair back into each lane's destination register.
///
/// ```text
/// input  lane 4*r + k: [matrix[r][2*k], matrix[r][2*k + 1]]
/// output lane 4*c + k: [matrix[2*k][c], matrix[2*k + 1][c]]
/// ```
///
/// This operation only exchanges register fragments between lanes. It does
/// not access memory and is not a memory fence.
///
/// # PTX
///
/// `movmatrix.sync.aligned.m8n8.trans.b16 %d, %a;`
///
/// # Safety
///
/// - All 32 lanes must execute the same call together.
/// - Calling from divergent control flow is undefined behavior.
/// - Requires `sm_75+` and PTX ISA 7.8+. cuda-oxide selects both floors
///   automatically, including when targeting Turing or Ampere.
#[inline(never)]
#[must_use]
pub unsafe fn movmatrix_trans_b16(a: u32) -> u32 {
    let _ = a;
    unreachable!("movmatrix_trans_b16 called outside CUDA kernel context")
}

// =============================================================================
// Shared-memory matrix loads
// =============================================================================

/// Load one 8×8 matrix tile from shared memory.
///
/// # PTX
///
/// `ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%r0}, [addr];`
///
/// # Safety
///
/// - Lanes 0-7 must each provide a valid, naturally aligned 16-byte shared-memory row
/// - On sm_75, all 32 lanes must provide a valid address
/// - Must be called by all threads in a warp (warp-synchronous)
/// - Callers must use a suitable barrier or fence to order other memory accesses
/// - Requires sm_75+ (Turing and later)
#[inline(never)]
pub unsafe fn ldmatrix_x1(smem_ptr: *const u32) -> u32 {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x1 called outside CUDA kernel context")
}

/// Load one 8×8 matrix tile from shared memory in column-major layout.
///
/// # PTX
///
/// `ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%r0}, [addr];`
///
/// # Safety
///
/// Same address-lane, synchronization, and target requirements as [`ldmatrix_x1`].
#[inline(never)]
pub unsafe fn ldmatrix_x1_trans(smem_ptr: *const u32) -> u32 {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x1_trans called outside CUDA kernel context")
}

/// Load 2 packed 8×8 matrices from shared memory.
///
/// Returns `[u32; 2]` (each u32 = 2 packed b16 values).
///
/// # PTX
///
/// `ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%r0, %r1}, [addr];`
///
/// # Safety
///
/// - Lanes 0-15 provide the 16 row addresses
/// - On sm_75, all 32 lanes must provide a valid address
/// - All lanes must participate, and callers must order other memory accesses
/// - Requires sm_75+ (Turing and later)
#[inline(never)]
pub unsafe fn ldmatrix_x2(smem_ptr: *const u32) -> [u32; 2] {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x2 called outside CUDA kernel context")
}

/// Load 2 packed 8×8 matrices from shared memory in column-major layout.
///
/// # PTX
///
/// `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r0, %r1}, [addr];`
///
/// # Safety
///
/// Same address-lane, synchronization, and target requirements as [`ldmatrix_x2`].
#[inline(never)]
pub unsafe fn ldmatrix_x2_trans(smem_ptr: *const u32) -> [u32; 2] {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x2_trans called outside CUDA kernel context")
}

/// Load 4 packed 8×8 matrices from shared memory.
///
/// Returns `[u32; 4]` (each u32 = 2 packed b16 values).
///
/// # PTX
///
/// `ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r0, %r1, %r2, %r3}, [addr];`
///
/// # Safety
///
/// - All 32 lanes provide valid, naturally aligned 16-byte row addresses
/// - All lanes must participate, and callers must order other memory accesses
/// - Requires sm_75+ (Turing and later)
#[inline(never)]
pub unsafe fn ldmatrix_x4(smem_ptr: *const u32) -> [u32; 4] {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x4 called outside CUDA kernel context")
}

/// Load 4 packed 8×8 matrices from shared memory in column-major layout.
///
/// # PTX
///
/// `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r0, %r1, %r2, %r3}, [addr];`
///
/// # Safety
///
/// Same address-lane, synchronization, and target requirements as [`ldmatrix_x4`].
#[inline(never)]
pub unsafe fn ldmatrix_x4_trans(smem_ptr: *const u32) -> [u32; 4] {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x4_trans called outside CUDA kernel context")
}
