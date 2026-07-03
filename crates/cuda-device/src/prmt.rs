// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte permute intrinsic (`prmt.b32`).
//!
//! Selects 4 individual bytes from the concatenated 8-byte value `{b, a}`
//! based on the control word, useful for arbitrary byte-level data
//! rearrangement on all GPU architectures.
//!
//! # `prmt.b32`
//!
//! Each nibble of `control` (bits 3:0, 7:4, 11:8, 15:12) selects one byte
//! from the 8-byte value `{b, a}` (byte 0 = LSB of `a`, byte 7 = MSB of `b`).
//! Bit 3 of each nibble selects the sign: 0 = byte value, 1 = sign-extend
//! from bit 7.
//!
//! # Supported on
//!
//! - All architectures (SM 1.0+, PTX ISA 1.0+)

/// Byte permute: select 4 bytes from the concatenation of `a` and `b`.
///
/// Each nibble of `control` (bits 3:0, 7:4, 11:8, 15:12) selects one byte
/// from the 8-byte value `{b, a}` (byte 0 = LSB of `a`, byte 7 = MSB of `b`).
/// Bit 3 of each nibble selects the sign: 0 = byte value, 1 = sign-extend
/// from bit 7.
///
/// # PTX
///
/// ```ptx
/// prmt.b32 %d, %a, %b, %c;
/// ```
#[inline(never)]
#[must_use]
pub fn prmt(a: u32, b: u32, control: u32) -> u32 {
    let _ = (a, b, control);
    unreachable!("prmt called outside CUDA kernel context")
}
