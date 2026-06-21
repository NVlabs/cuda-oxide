/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level Matrix Multiply-Accumulate (WMMA/mma.sync) for Ampere+ architectures.
//!
//! WMMA operates at the warp level (32 threads) using `mma.sync` PTX instructions
//! to perform tensor core matrix multiplication. Unlike WGMMA (Hopper, 128 threads),
//! WMMA uses a single warp and is available on a much broader range of GPUs.
//!
//! # Architecture
//!
//! ```text
//! mma.sync.aligned.m16n8k16 Operation:
//!
//!     A (16×16)          B (16×8)            D (16×8)
//!   ┌──────────┐      ┌─────────┐        ┌─────────┐
//!   │          │      │         │        │         │
//!   │ 16 rows  │  ×   │ 16 rows │   =    │  16×8   │
//!   │ 16 cols  │      │  8 cols │        │  accum  │
//!   │ (f16)    │      │ (f16)   │        │  (f32)  │
//!   └──────────┘      └─────────┘        └─────────┘
//!   row-major         col-major          distributed across
//!                                        32 threads
//! ```
//!
//! # Per-Thread Register Layout
//!
//! Each thread in the 32-thread warp holds:
//! - **A fragment**: 4 × u32 (each u32 = 2 packed f16 values → 8 f16 total)
//! - **B fragment**: 2 × u32 (each u32 = 2 packed f16 values → 4 f16 total)
//! - **D/C accumulator**: 4 × f32
//!
//! Total per warp: 32 threads × 4 f32 = 128 f32 → 16×8 tile
//!
//! # Usage Pattern
//!
//! ```rust,ignore
//! use cuda_device::wmma::*;
//! use cuda_device::SharedArray;
//!
//! // Load A tile from shared memory (each thread provides its own smem address)
//! let a_frag = unsafe { ldmatrix_x4(a_smem_ptr) };
//!
//! // Load B tile from shared memory
//! let b_frag = unsafe { ldmatrix_x2_trans(b_smem_ptr) };
//!
//! // Accumulate: D += A × B
//! let mut acc = [0.0f32; 4];
//! unsafe { mma_m16n8k16_f32_f16(&mut acc, &a_frag, &b_frag); }
//! ```
//!
//! # Hardware Support
//!
//! - **sm_80 (Ampere)**: A100
//! - **sm_86 (Ampere)**: RTX 3090, RTX A2000, RTX A5000
//! - **sm_87 (Ampere)**: Jetson AGX Orin
//! - **sm_89 (Ada Lovelace)**: RTX 4090, L40
//! - **sm_90+ (Hopper+)**: H100, H200 (also supports WGMMA)
//! - **sm_120 (Blackwell consumer)**: RTX 5090 (uses mma.sync, not WGMMA)

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
#[inline(never)]
pub unsafe fn ldmatrix_x2_trans(smem_ptr: *const u32) -> [u32; 2] {
    let _ = smem_ptr;
    unreachable!("ldmatrix_x2_trans called outside CUDA kernel context")
}

// =============================================================================
// mma.sync: Warp-synchronous matrix multiply-accumulate
// =============================================================================

/// Warp matrix multiply-accumulate: D = A × B + C (m16n8k16, f32 output, f16 inputs).
///
/// Performs a 16×8×16 matrix multiplication using tensor cores. All 32 threads
/// in the warp participate synchronously.
///
/// # Matrix Dimensions
///
/// - **A**: 16×16 (row-major, f16), distributed as 4 × u32 per thread
/// - **B**: 16×8 (col-major, f16), distributed as 2 × u32 per thread
/// - **D/C**: 16×8 (f32 accumulator), distributed as 4 × f32 per thread
///
/// # Parameters
///
/// - `acc`: Mutable accumulator (4 × f32 per thread, read-modify-write: D = A*B + acc)
/// - `a`: A fragment (4 × u32, each u32 contains 2 packed f16 values)
/// - `b`: B fragment (2 × u32, each u32 contains 2 packed f16 values)
///
/// # PTX
///
/// ```ptx
/// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
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
/// - Fragment values must come from `ldmatrix` or be correctly distributed
///
/// # Example
///
/// ```rust,ignore
/// // Process a 16×8 tile with K=64 (4 mma calls, each K=16)
/// let mut acc = [0.0f32; 4];
/// for k_step in 0..4 {
///     let a = unsafe { ldmatrix_x4(a_smem_ptr(k_step)) };
///     let b = unsafe { ldmatrix_x2_trans(b_smem_ptr(k_step)) };
///     unsafe { mma_m16n8k16_f32_f16(&mut acc, &a, &b); }
/// }
/// ```
#[inline(never)]
pub unsafe fn mma_m16n8k16_f32_f16(acc: &mut [f32; 4], a: &[u32; 4], b: &[u32; 2]) {
    let _ = (acc, a, b);
    unreachable!("mma_m16n8k16_f32_f16 called outside CUDA kernel context")
}

/// Type alias for the WMMA accumulator (m16n8 tile, 4 floats per thread).
pub type Acc16x8 = [f32; 4];

/// Type alias for A fragment (m16k16, 4 packed u32s per thread).
pub type FragA = [u32; 4];

/// Type alias for B fragment (k16n8, 2 packed u32s per thread).
pub type FragB = [u32; 2];

/// Initialize an accumulator to zero.
#[inline(always)]
pub const fn zero_acc() -> Acc16x8 {
    [0.0f32; 4]
}
