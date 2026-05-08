/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Fast math intrinsics for device code that aren't exposed by Rust's `core`.
//!
//! Rust's float types and `core::intrinsics` cover most math functions
//! (`sqrt`, `exp`, `log`, etc.), and cuda-oxide already lowers those to
//! libdevice. This module provides additional intrinsics that have a single-PTX
//! libdevice equivalent but no Rust analogue, so users don't have to write the
//! slower `1.0 / sqrt(x)` form by hand.

/// Reciprocal square root: `1.0 / sqrt(x)`.
///
/// Lowers to libdevice `__nv_rsqrtf`, which compiles to a single PTX
/// `rsqrt.approx.ftz.f32` instruction on sm_53+. Faster and more accurate at
/// the bit level than `1.0 / x.sqrt()` because the dedicated hardware op
/// avoids the round-trip through square root + division.
///
/// # Examples
///
/// ```rust,ignore
/// use cuda_device::math::rsqrt_f32;
///
/// // RMSNorm style: y = x * rsqrt(mean_sq + eps)
/// let scale = rsqrt_f32(mean_sq + eps);
/// let y = x * scale;
/// ```
#[inline(never)]
pub fn rsqrt_f32(x: f32) -> f32 {
    let _ = x;
    unreachable!("rsqrt_f32 called outside CUDA kernel context")
}

/// Reciprocal square root for `f64`. Lowers to libdevice `__nv_rsqrt`.
#[inline(never)]
pub fn rsqrt_f64(x: f64) -> f64 {
    let _ = x;
    unreachable!("rsqrt_f64 called outside CUDA kernel context")
}
