/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warpgroup Matrix Multiply-Accumulate (WGMMA) for Hopper `sm_90a`.
//!
//! WGMMA operates at the warpgroup level (128 threads = 4 warps) to perform
//! efficient tensor core matrix multiplication. Unlike WMMA which operates
//! per-warp (32 threads), WGMMA leverages the full warpgroup for larger tiles.
//!
//! The control and descriptor helpers are available. MMA calls remain
//! unsupported until pending accumulator registers can stay live through
//! `commit_group` and `wait_group`.
//!
//! # Architecture
//!
//! ```text
//! WGMMA m64n64k16 Operation:
//!
//!     A (64×16)         B (16×64)           D (64×64)
//!   ┌──────────┐      ┌───────────────┐    ┌───────────────┐
//!   │          │      │               │    │               │
//!   │  64 rows │  ×   │   16 rows     │ =  │   64×64       │
//!   │  16 cols │      │   64 cols     │    │ accumulator   │
//!   │          │      │               │    │               │
//!   └──────────┘      └───────────────┘    └───────────────┘
//!   row-major         col-major            distributed across
//!   in SMEM           in SMEM              128 threads
//! ```
//!
//! # Per-Thread Accumulator
//!
//! Each thread in the 128-thread warpgroup holds 32 floats:
//! ```rust,ignore
//! let mut acc: [[f32; 8]; 4] = [[0.0; 8]; 4];  // 32 floats per thread
//! // Total: 128 threads × 32 = 4096 floats = 64×64 tile
//! ```
//!
//! # Usage Pattern
//!
//! ```rust,ignore
//! use cuda_device::wgmma::*;
//!
//! // Initialize accumulator to zero
//! let mut acc: [[f32; 8]; 4] = [[0.0; 8]; 4];
//!
//! // Create descriptors for shared memory tiles
//! let desc_a = make_smem_desc(a_smem_ptr);
//! let desc_b = make_smem_desc(b_smem_ptr);
//!
//! // Fence before WGMMA
//! wgmma_fence();
//!
//! // Issue WGMMA (K=16 per instruction, so 4 calls for K=64)
//! wgmma_mma_m64n64k16_f32_bf16(&mut acc, desc_a, desc_b);
//!
//! // Commit the group
//! wgmma_commit_group();
//!
//! // Wait for completion
//! wgmma_wait_group::<0>();
//! ```
//!
//! # Hardware Support
//!
//! - **sm_90a (Hopper)**: H100, H200

// =============================================================================
// WGMMA Synchronization Primitives
// =============================================================================

include!("generated/wgmma_control.rs");

// =============================================================================
// SMEM Descriptor Creation
// =============================================================================

/// Create a 64-bit shared memory descriptor for WGMMA input matrices.
///
/// This helper creates the fixed-layout descriptor used by the current
/// lowering. It combines the shared-memory address with fixed stride and
/// swizzle fields.
///
/// # Parameters
///
/// - `ptr`: Pointer to matrix data in shared memory
///
/// # Returns
///
/// A 64-bit descriptor suitable for WGMMA instructions.
///
/// # Encoding
///
/// ```rust,ignore
/// ((shared_address >> 4) & 0x3fff) | 0xC000000800080000
/// ```
///
/// # Safety
///
/// - `ptr` must point to valid shared memory
/// - The memory layout must match WGMMA requirements (proper alignment, swizzling)
///
/// # PTX
///
/// Uses `cvta.to.shared.u64` to convert the generic pointer.
#[inline(never)]
pub unsafe fn make_smem_desc(ptr: *const u8) -> u64 {
    let _ = ptr;
    // Lowered to inline PTX:
    // {
    //   .reg .u64 addr;
    //   cvta.to.shared.u64 addr, %ptr;
    //   shr.u64 addr, addr, 4;
    //   and.b64 addr, addr, 0x3fff;
    //   or.b64 %result, addr, 0xC000000800080000;
    // }
    unreachable!("make_smem_desc called outside CUDA kernel context")
}

/// Compatibility entry point for a custom SMEM descriptor.
///
/// This function does not have an importer or lowering path yet. It remains
/// public to avoid breaking existing source code.
///
/// # Parameters
///
/// - `ptr`: Pointer to matrix data in shared memory
/// - `leading_dim`: Leading dimension in bytes (divided by 16 internally)
/// - `stride`: Stride in bytes (divided by 16 internally)
/// - `swizzle_128b`: Enable 128-byte swizzling
///
/// # Safety
///
/// - `ptr` must be a valid pointer to matrix data in shared memory
/// - Must be called from within a CUDA kernel context
#[inline(never)]
pub unsafe fn make_smem_desc_custom(
    ptr: *const u8,
    leading_dim: u32,
    stride: u32,
    swizzle_128b: bool,
) -> u64 {
    let _ = (ptr, leading_dim, stride, swizzle_128b);
    unreachable!("make_smem_desc_custom called outside CUDA kernel context")
}

// =============================================================================
// WGMMA Matrix Multiply-Accumulate Instructions
// =============================================================================

/// Warpgroup matrix multiply-accumulate: D += A × B
///
/// Performs a 64×64×16 matrix multiplication using tensor cores at the
/// warpgroup level. All 128 threads in the warpgroup participate.
///
/// # Matrix Dimensions
///
/// - **A**: 64×16 (M=64 rows, K=16 cols), row-major in shared memory
/// - **B**: 16×64 (K=16 rows, N=64 cols), column-major in shared memory
/// - **D**: 64×64 output, accumulated in registers
///
/// # Accumulator Layout
///
/// The 64×64 output is distributed across 128 threads:
/// - Each thread holds 32 floats in `[[f32; 8]; 4]`
/// - 128 threads × 32 = 4096 floats = 64×64
///
/// # Parameters
///
/// - `acc`: Mutable reference to the accumulator (32 floats per thread)
/// - `desc_a`: SMEM descriptor for matrix A (from `make_smem_desc`)
/// - `desc_b`: SMEM descriptor for matrix B (from `make_smem_desc`)
///
/// # PTX
///
/// ```ptx
/// wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16
///     {%f0, %f1, ..., %f31}, %rd_desc_a, %rd_desc_b,
///     1, 1, 1, 0, 0;
/// ```
///
/// # Safety
///
/// - Descriptors must be valid SMEM descriptors
/// - Must be called by all threads in a warpgroup
/// - Must be called from within a CUDA kernel context on sm_90a
///
/// # Example
///
/// ```rust,ignore
/// // Process a 64×64 K-tile (requires 4 WGMMA calls since K=16 per call)
/// for k in 0..4 {
///     let offset = k * 16 * elem_size;
///     wgmma_mma_m64n64k16_f32_bf16(
///         &mut acc,
///         desc_a + offset as u64,
///         desc_b + offset as u64,
///     );
/// }
/// wgmma_commit_group();
/// wgmma_wait_group::<0>();
/// ```
#[inline(never)]
pub unsafe fn wgmma_mma_m64n64k16_f32_bf16(acc: &mut [[f32; 8]; 4], desc_a: u64, desc_b: u64) {
    let _ = (acc, desc_a, desc_b);
    // Lowered to inline PTX with 32 accumulator registers + 2 descriptors + 5 immediates
    unreachable!("wgmma_mma_m64n64k16_f32_bf16 called outside CUDA kernel context")
}

/// WGMMA with f32 accumulator and f16 (IEEE half) inputs.
///
/// Same as `wgmma_mma_m64n64k16_f32_bf16` but uses f16 input format.
///
/// # Safety
///
/// - Descriptors must be valid SMEM descriptors
/// - Must be called by all threads in a warpgroup
/// - Must be called from within a CUDA kernel context on sm_90a
#[inline(never)]
pub unsafe fn wgmma_mma_m64n64k16_f32_f16(acc: &mut [[f32; 8]; 4], desc_a: u64, desc_b: u64) {
    let _ = (acc, desc_a, desc_b);
    unreachable!("wgmma_mma_m64n64k16_f32_f16 called outside CUDA kernel context")
}

/// WGMMA with f32 accumulator and tf32 (TensorFloat-32) inputs.
///
/// TF32 uses 19 bits (1 sign + 8 exponent + 10 mantissa), providing
/// better precision than bf16 while maintaining tensor core throughput.
///
/// # Safety
///
/// - Descriptors must be valid SMEM descriptors
/// - Must be called by all threads in a warpgroup
/// - Must be called from within a CUDA kernel context on sm_90a
#[inline(never)]
pub unsafe fn wgmma_mma_m64n64k16_f32_tf32(acc: &mut [[f32; 8]; 4], desc_a: u64, desc_b: u64) {
    let _ = (acc, desc_a, desc_b);
    unreachable!("wgmma_mma_m64n64k16_f32_tf32 called outside CUDA kernel context")
}

// =============================================================================
// Accumulator Utilities
// =============================================================================

/// Type alias for the WGMMA accumulator (m64n64 tile, 32 floats per thread).
pub type Acc64x64 = [[f32; 8]; 4];

/// Initialize an accumulator to zero.
///
/// # Returns
///
/// A zeroed accumulator suitable for WGMMA operations.
#[inline(always)]
pub const fn zero_accumulator() -> Acc64x64 {
    [[0.0f32; 8]; 4]
}
