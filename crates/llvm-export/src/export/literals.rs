/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Constant and literal formatting for LLVM IR output.

pub(super) fn format_half_literal(bits: u16) -> String {
    format!("0xH{bits:04X}")
}

/// Format a float value as an LLVM IR literal.
///
/// LLVM IR accepts decimal notation with a decimal point (e.g. `1.0`) or a
/// 16-hex-digit double bit pattern (e.g. `0x7FF0000000000000`). The hex form
/// is always interpreted as f64 and narrowed to the instruction's destination
/// type (`float`, `double`, or `half`). NaN and Inf must use the hex form
/// because the bare tokens `nan`/`inf` are not part of the LLVM IR grammar
/// and are rejected by both `llc` and libNVVM.
pub(super) fn format_float_literal(value: f64) -> String {
    if value.is_nan() {
        // Canonical quiet NaN. Sign bit is preserved for symmetry with the
        // Inf arm; payload bits are canonicalized to the qNaN marker only.
        if value.is_sign_negative() {
            "0xFFF8000000000000".to_string()
        } else {
            "0x7FF8000000000000".to_string()
        }
    } else if value.is_infinite() {
        if value.is_sign_positive() {
            "0x7FF0000000000000".to_string() // +inf
        } else {
            "0xFFF0000000000000".to_string() // -inf
        }
    } else {
        // Finite values: ensure a decimal point so LLVM does not mistake the
        // literal for an integer.
        let s = format!("{value}");
        if s.contains('.') || s.contains('e') || s.contains('E') {
            s
        } else {
            format!("{s}.0")
        }
    }
}

#[cfg(test)]
mod float_literal_tests {
    use super::format_float_literal;

    #[test]
    fn nan_is_emitted_as_hex_qnan_not_bare_token() {
        // The bare token `nan` is not valid LLVM IR and is rejected by
        // libNVVM with "parse expected value token".
        assert_eq!(format_float_literal(f64::NAN), "0x7FF8000000000000");
        assert_eq!(
            format_float_literal(f64::from(f32::NAN)),
            "0x7FF8000000000000"
        );
    }

    #[test]
    fn negative_nan_preserves_sign() {
        assert_eq!(format_float_literal(-f64::NAN), "0xFFF8000000000000");
    }

    #[test]
    fn positive_infinity_is_emitted_as_hex() {
        assert_eq!(format_float_literal(f64::INFINITY), "0x7FF0000000000000");
    }

    #[test]
    fn negative_infinity_is_emitted_as_hex() {
        assert_eq!(
            format_float_literal(f64::NEG_INFINITY),
            "0xFFF0000000000000"
        );
    }

    #[test]
    fn finite_values_get_a_decimal_point() {
        let formatted = format_float_literal(42.0);
        assert!(
            formatted.contains('.') || formatted.contains('e') || formatted.contains('E'),
            "expected decimal point or exponent in `{formatted}`"
        );
        assert_eq!(format_float_literal(-0.0), "-0.0");
    }
}
