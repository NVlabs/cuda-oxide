/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level shared memory matrix load (`ldmatrix`) for Ampere+ architectures.
//!
//! These intrinsics use `ldmatrix.sync` PTX instructions to cooperatively load
//! packed 8×8 matrices from shared memory into registers across a 32-thread warp.
//!
//! # Operations
//!
//! | Function          | PTX                                              | Returns   |
//! |-------------------|--------------------------------------------------|-----------|
//! | `ldmatrix_x4`     | `ldmatrix.sync.aligned.m8n8.x4.shared.b16`      | `[u32;4]` |
//! | `ldmatrix_x2`     | `ldmatrix.sync.aligned.m8n8.x2.shared.b16`      | `[u32;2]` |
//! | `ldmatrix_x4_trans`| `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16`| `[u32;4]` |
//! | `ldmatrix_x2_trans`| `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16`| `[u32;2]` |
//!
//! # Hardware Support
//!
//! - **sm_75+** (Turing and later)

// =============================================================================
// ldmatrix: Warp-synchronous shared memory → register matrix load
// =============================================================================

/// Load 4 packed 8×8 matrices from shared memory into registers.
///
/// Each thread provides a shared memory address. The instruction cooperatively
/// loads data across all 32 threads, distributing matrix elements to the
/// appropriate registers for subsequent `mma.sync` operations.
///
/// Returns 4 × u32 values (each u32 = 2 packed f16 values).
///
/// # PTX
///
/// ```ptx
/// ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r0,%r1,%r2,%r3}, [%addr];
/// ```
///
/// # Safety
///
/// - `smem_ptr` must point to valid shared memory, 16-byte aligned
/// - Must be called by all threads in a warp
/// - Must be called from within a CUDA kernel context on sm_75+
///
/// See also: [`ldmatrix_x2`], [`ldmatrix_x4_trans`]
#[inline(never)]
pub unsafe fn ldmatrix_x4(smem_ptr: *const u32) -> [u32; 4] {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x4 called outside CUDA kernel context")
}

/// Load 2 packed 8×8 matrices from shared memory into registers.
///
/// Similar to `ldmatrix_x4` but loads only 2 matrices (for B fragments
/// in m16n8k16 layout).
///
/// Returns 2 × u32 values (each u32 = 2 packed f16 values).
///
/// # PTX
///
/// ```ptx
/// ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%r0,%r1}, [%addr];
/// ```
///
/// # Safety
///
/// - `smem_ptr` must point to valid shared memory, 16-byte aligned
/// - Must be called by all threads in a warp
/// - Must be called from within a CUDA kernel context on sm_75+
///
/// See also: [`ldmatrix_x4`], [`ldmatrix_x2_trans`]
#[inline(never)]
pub unsafe fn ldmatrix_x2(smem_ptr: *const u32) -> [u32; 2] {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x2 called outside CUDA kernel context")
}

/// Load 4 packed 8×8 matrices from shared memory with transpose.
///
/// Same as `ldmatrix_x4` but transposes the loaded matrices. This is
/// useful for loading B matrices that are stored in row-major order
/// but need to be consumed in column-major order by `mma.sync`.
///
/// # PTX
///
/// ```ptx
/// ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r0,%r1,%r2,%r3}, [%addr];
/// ```
///
/// # Safety
///
/// - `smem_ptr` must point to valid shared memory, 16-byte aligned
/// - Must be called by all threads in a warp
/// - Must be called from within a CUDA kernel context on sm_75+
///
/// See also: [`ldmatrix_x4`], [`ldmatrix_x2_trans`]
#[inline(never)]
pub unsafe fn ldmatrix_x4_trans(smem_ptr: *const u32) -> [u32; 4] {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x4_trans called outside CUDA kernel context")
}

/// Load 2 packed 8×8 matrices from shared memory with transpose.
///
/// # PTX
///
/// ```ptx
/// ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r0,%r1}, [%addr];
/// ```
///
/// # Safety
///
/// - `smem_ptr` must point to valid shared memory, 16-byte aligned
/// - Must be called by all threads in a warp
/// - Must be called from within a CUDA kernel context on sm_75+
///
/// See also: [`ldmatrix_x2`], [`ldmatrix_x4_trans`]
#[inline(never)]
pub unsafe fn ldmatrix_x2_trans(smem_ptr: *const u32) -> [u32; 2] {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x2_trans called outside CUDA kernel context")
}
