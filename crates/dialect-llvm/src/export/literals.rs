/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Constant and literal formatting for LLVM IR output.

pub(super) fn format_half_literal(bits: u16) -> String {
    format!("0xH{bits:04X}")
}

/// Format a float value as an LLVM IR literal.
/// LLVM requires float literals to have a decimal point (e.g., "0.0" not "0").
pub(super) fn format_float_literal(value: f64) -> String {
    if value.is_nan() {
        "nan".to_string()
    } else if value.is_infinite() {
        if value.is_sign_positive() {
            "0x7FF0000000000000".to_string() // +inf
        } else {
            "0xFFF0000000000000".to_string() // -inf
        }
    } else {
        // Format the float, ensuring it has a decimal point
        let s = format!("{value}");
        if s.contains('.') || s.contains('e') || s.contains('E') {
            s
        } else {
            format!("{s}.0")
        }
    }
}
