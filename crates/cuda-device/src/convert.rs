/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Type conversion intrinsics.
//!
//! These intrinsics provide access to PTX type conversion instructions that
//! are more efficient than scalar Rust casts.

include!("generated/convert.rs");

// =============================================================================
// Packed f16x2 unpacking
// =============================================================================
//
// The generated conversions above all run f32 -> packed narrow type. The
// inverse is hand-written because there is nothing to generate it from: the
// pinned LLVM metadata has no f16x2-to-f32x2 intrinsic (every `ff2f16x2_*`
// record is the packing direction), and PTX has no single unpacking
// instruction either. Splitting a packed pair is a `mov.b32` into two 16-bit
// registers followed by two `cvt.f32.f16`, which is what the casts below
// lower to.
//
// These matter because packed f16 is the layout that gives the only reliably
// wide global loads: reading weights as `u32`/`u64` and unpacking in registers
// beats four scalar `f16` loads. Doing that inline at every call site is where
// the manual `from_bits`/shift/mask arithmetic came from.

/// Unpacks a packed `f16x2` into its two `f32` values, low half first.
///
/// This is the inverse of [`cvt_f16x2_f32`].
///
/// # Example
///
/// ```rust,ignore
/// // One 32-bit load carrying two f16 weights.
/// let packed = unsafe { *(ptr as *const u32) };
/// let (w0, w1) = convert::cvt_f32x2_f16x2(packed);
/// ```
#[must_use]
#[inline(always)]
pub fn cvt_f32x2_f16x2(packed: u32) -> (f32, f32) {
    let lo = f16::from_bits(packed as u16);
    let hi = f16::from_bits((packed >> 16) as u16);
    (lo as f32, hi as f32)
}

/// Unpacks the low half of a packed `f16x2` to `f32`.
///
/// Prefer [`cvt_f32x2_f16x2`] when both halves are needed; this exists for the
/// case where only one is, so the other conversion is not emitted at all.
#[must_use]
#[inline(always)]
pub fn cvt_f32_f16x2_lo(packed: u32) -> f32 {
    f16::from_bits(packed as u16) as f32
}

/// Unpacks the high half of a packed `f16x2` to `f32`.
///
/// See [`cvt_f32_f16x2_lo`].
#[must_use]
#[inline(always)]
pub fn cvt_f32_f16x2_hi(packed: u32) -> f32 {
    f16::from_bits((packed >> 16) as u16) as f32
}

/// Unpacks a packed `bf16x2` into its two `f32` values, low half first.
///
/// This is the inverse of [`cvt_rz_bf16x2_f32`]. `bf16` shares `f32`'s exponent
/// range, so widening is an exact shift into the high half of the mantissa
/// rather than a conversion.
#[must_use]
#[inline(always)]
pub fn cvt_f32x2_bf16x2(packed: u32) -> (f32, f32) {
    (
        f32::from_bits(packed << 16),
        f32::from_bits(packed & 0xFFFF_0000),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trips through the generated packer so the halves cannot silently
    /// swap: `cvt_f16x2_f32` documents the first argument as the low half.
    ///
    /// The packer is a device intrinsic and panics on the host, so the packed
    /// words here are written out by hand instead.
    #[test]
    fn unpacks_halves_in_the_documented_order() {
        // 1.0 = 0x3C00, 2.0 = 0x4000 in f16.
        let packed = 0x4000_3C00;
        let (lo, hi) = cvt_f32x2_f16x2(packed);
        assert_eq!(lo, 1.0, "low half must come from the low 16 bits");
        assert_eq!(hi, 2.0, "high half must come from the high 16 bits");
        assert_eq!(cvt_f32_f16x2_lo(packed), 1.0);
        assert_eq!(cvt_f32_f16x2_hi(packed), 2.0);
    }

    /// A negative value in one half must not bleed into the other, which is
    /// what an arithmetic shift instead of a logical one would do.
    #[test]
    fn sign_bits_stay_in_their_own_half() {
        // -1.0 = 0xBC00 in the high half, 1.0 = 0x3C00 in the low half.
        let (lo, hi) = cvt_f32x2_f16x2(0xBC00_3C00);
        assert_eq!(lo, 1.0);
        assert_eq!(hi, -1.0);
    }

    #[test]
    fn unpacks_bf16_pairs() {
        // bf16 is the top 16 bits of the f32 bit pattern.
        let one = (1.0f32.to_bits() >> 16) as u32; // 0x3F80
        let two = (2.0f32.to_bits() >> 16) as u32; // 0x4000
        let packed = (two << 16) | one;
        let (lo, hi) = cvt_f32x2_bf16x2(packed);
        assert_eq!(lo, 1.0);
        assert_eq!(hi, 2.0);
    }

    /// Every f16 bit pattern that is not a NaN must survive the trip, including
    /// subnormals and both zeroes.
    #[test]
    fn every_finite_f16_pattern_round_trips() {
        for bits in 0u16..=u16::MAX {
            let expected = f16::from_bits(bits);
            if expected.is_nan() {
                continue;
            }
            let (lo, _) = cvt_f32x2_f16x2(bits as u32);
            assert_eq!(
                lo, expected as f32,
                "low-half mismatch for f16 bits {bits:#06x}"
            );
            let (_, hi) = cvt_f32x2_f16x2((bits as u32) << 16);
            assert_eq!(
                hi, expected as f32,
                "high-half mismatch for f16 bits {bits:#06x}"
            );
        }
    }
}
