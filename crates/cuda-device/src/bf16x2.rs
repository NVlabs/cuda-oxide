// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Packed `bf16x2` arithmetic intrinsics.
//!
//! Packed bf16 ALU instructions on Ampere are limited: PTX `add.bf16x2` and
//! `mul.bf16x2` require `sm_90` or higher, while `fma.rn.bf16x2` is supported
//! from `sm_80`. So on `sm_80`/`sm_86` the only hardware-packed bf16 path is
//! FMA; plain add/mul can be expressed via FMA (with a packed `1.0` operand)
//! at no extra cost.
//!
//! Each `u32` carries two bf16 values: low 16 bits = first lane, high 16 bits
//! = second lane. This matches the layout produced by
//! [`crate::tcgen05::cvt_f32x2_bf16x2`].

/// Packed bf16x2 fused multiply-add: `d = a * b + c`.
///
/// All three operands and the result are packed `bf16x2` carried as `u32`,
/// matching `cvt.rn.bf16x2.f32`'s output layout (low 16 = first lane, high 16
/// = second lane).
///
/// # PTX
///
/// ```ptx
/// fma.rn.bf16x2 %d, %a, %b, %c;
/// ```
///
/// # Supported on
///
/// - `sm_80+` (Ampere onwards). On `sm_70`/`sm_75` this lowering will be
///   rejected by `ptxas`.
///
/// # Notes
///
/// This intrinsic is the only packed-bf16 ALU op available on `sm_80`/`sm_86`:
/// `add.bf16x2` / `mul.bf16x2` exist but require `sm_90+`. To get a hardware
/// packed add on Ampere, build the operation as `fma(a, ONE_BF16X2, b)` where
/// `ONE_BF16X2 = 0x3F803F80u32` encodes packed (1.0, 1.0).
#[inline(never)]
pub fn fma_bf16x2(a: u32, b: u32, c: u32) -> u32 {
    let _ = (a, b, c);
    unreachable!("fma_bf16x2 called outside CUDA kernel context")
}
