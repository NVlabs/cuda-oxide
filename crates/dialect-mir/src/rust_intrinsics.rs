/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared names for internal Rust compiler intrinsic placeholder calls.
//!
//! The importer emits these names as ordinary `mir.call` callees when it sees a
//! rustc intrinsic that needs target-specific lowering. The MIR-to-LLVM pass
//! recognizes the same names and replaces them with LLVM or CUDA libdevice calls.
//! Keep the prefix centralized here so the planned magic-hash prefix change only
//! needs one edit.

/// Build an internal Rust intrinsic placeholder name from its stable suffix.
macro_rules! placeholder {
    ($suffix:literal) => {
        concat!("__cuda_oxide_rust_intrinsic_", $suffix)
    };
}

/// Prefix used for cuda-oxide internal Rust intrinsic placeholder calls.
pub const PLACEHOLDER_PREFIX: &str = placeholder!("");

/// Placeholder call used for `core::intrinsics::rotate_left`.
pub const CALLEE_ROTATE_LEFT: &str = placeholder!("rotate_left");
/// Placeholder call used for `core::intrinsics::rotate_right`.
pub const CALLEE_ROTATE_RIGHT: &str = placeholder!("rotate_right");
/// Placeholder call used for `core::intrinsics::ctpop`.
pub const CALLEE_CTPOP: &str = placeholder!("ctpop");
/// Placeholder call used for `core::intrinsics::ctlz`.
pub const CALLEE_CTLZ: &str = placeholder!("ctlz");
/// Placeholder call used for `core::intrinsics::ctlz_nonzero`.
pub const CALLEE_CTLZ_NONZERO: &str = placeholder!("ctlz_nonzero");
/// Placeholder call used for `core::intrinsics::cttz`.
pub const CALLEE_CTTZ: &str = placeholder!("cttz");
/// Placeholder call used for `core::intrinsics::cttz_nonzero`.
pub const CALLEE_CTTZ_NONZERO: &str = placeholder!("cttz_nonzero");
/// Placeholder call used for `core::intrinsics::bswap`.
pub const CALLEE_BSWAP: &str = placeholder!("bswap");
/// Placeholder call used for `core::intrinsics::bitreverse`.
pub const CALLEE_BITREVERSE: &str = placeholder!("bitreverse");

/// Placeholder call used for `core::intrinsics::saturating_add`.
pub const CALLEE_SATURATING_ADD: &str = placeholder!("saturating_add");
/// Placeholder call used for `core::intrinsics::saturating_sub`.
pub const CALLEE_SATURATING_SUB: &str = placeholder!("saturating_sub");

/// Placeholder call used for `core::intrinsics::carrying_mul_add`.
/// Backs the bigint helper methods `carrying_mul_add`, `carrying_mul`,
/// and `widening_mul` on integer types.
pub const CALLEE_CARRYING_MUL_ADD: &str = placeholder!("carrying_mul_add");

/// Placeholder call used for `core::intrinsics::sqrtf32`.
pub const CALLEE_SQRT_F32: &str = placeholder!("sqrtf32");
/// Placeholder call used for `core::intrinsics::sqrtf64`.
pub const CALLEE_SQRT_F64: &str = placeholder!("sqrtf64");
/// Placeholder call used for `core::intrinsics::powif32`.
pub const CALLEE_POWI_F32: &str = placeholder!("powif32");
/// Placeholder call used for `core::intrinsics::powif64`.
pub const CALLEE_POWI_F64: &str = placeholder!("powif64");
/// Placeholder call used for `core::intrinsics::sinf32`.
pub const CALLEE_SIN_F32: &str = placeholder!("sinf32");
/// Placeholder call used for `core::intrinsics::sinf64`.
pub const CALLEE_SIN_F64: &str = placeholder!("sinf64");
/// Placeholder call used for `core::intrinsics::cosf32`.
pub const CALLEE_COS_F32: &str = placeholder!("cosf32");
/// Placeholder call used for `core::intrinsics::cosf64`.
pub const CALLEE_COS_F64: &str = placeholder!("cosf64");
/// Placeholder call used for `core::intrinsics::tanf32`.
pub const CALLEE_TAN_F32: &str = placeholder!("tanf32");
/// Placeholder call used for `core::intrinsics::tanf64`.
pub const CALLEE_TAN_F64: &str = placeholder!("tanf64");
/// Placeholder call used for `core::intrinsics::powf32`.
pub const CALLEE_POWF_F32: &str = placeholder!("powf32");
/// Placeholder call used for `core::intrinsics::powf64`.
pub const CALLEE_POWF_F64: &str = placeholder!("powf64");
/// Placeholder call used for `core::intrinsics::expf32`.
pub const CALLEE_EXP_F32: &str = placeholder!("expf32");
/// Placeholder call used for `core::intrinsics::expf64`.
pub const CALLEE_EXP_F64: &str = placeholder!("expf64");
/// Placeholder call used for `core::intrinsics::exp2f32`.
pub const CALLEE_EXP2_F32: &str = placeholder!("exp2f32");
/// Placeholder call used for `core::intrinsics::exp2f64`.
pub const CALLEE_EXP2_F64: &str = placeholder!("exp2f64");
/// Placeholder call used for `core::intrinsics::logf32`.
pub const CALLEE_LOG_F32: &str = placeholder!("logf32");
/// Placeholder call used for `core::intrinsics::logf64`.
pub const CALLEE_LOG_F64: &str = placeholder!("logf64");
/// Placeholder call used for `core::intrinsics::log2f32`.
pub const CALLEE_LOG2_F32: &str = placeholder!("log2f32");
/// Placeholder call used for `core::intrinsics::log2f64`.
pub const CALLEE_LOG2_F64: &str = placeholder!("log2f64");
/// Placeholder call used for `core::intrinsics::log10f32`.
pub const CALLEE_LOG10_F32: &str = placeholder!("log10f32");
/// Placeholder call used for `core::intrinsics::log10f64`.
pub const CALLEE_LOG10_F64: &str = placeholder!("log10f64");
/// Placeholder call used for `core::intrinsics::fmaf32`.
pub const CALLEE_FMA_F32: &str = placeholder!("fmaf32");
/// Placeholder call used for `core::intrinsics::fmaf64`.
pub const CALLEE_FMA_F64: &str = placeholder!("fmaf64");
/// Placeholder call used for `core::intrinsics::fmuladdf32`.
pub const CALLEE_FMULADD_F32: &str = placeholder!("fmuladdf32");
/// Placeholder call used for `core::intrinsics::fmuladdf64`.
pub const CALLEE_FMULADD_F64: &str = placeholder!("fmuladdf64");
/// Placeholder call used for `core::intrinsics::floorf32`.
pub const CALLEE_FLOOR_F32: &str = placeholder!("floorf32");
/// Placeholder call used for `core::intrinsics::floorf64`.
pub const CALLEE_FLOOR_F64: &str = placeholder!("floorf64");
/// Placeholder call used for `core::intrinsics::ceilf32`.
pub const CALLEE_CEIL_F32: &str = placeholder!("ceilf32");
/// Placeholder call used for `core::intrinsics::ceilf64`.
pub const CALLEE_CEIL_F64: &str = placeholder!("ceilf64");
/// Placeholder call used for `core::intrinsics::truncf32`.
pub const CALLEE_TRUNC_F32: &str = placeholder!("truncf32");
/// Placeholder call used for `core::intrinsics::truncf64`.
pub const CALLEE_TRUNC_F64: &str = placeholder!("truncf64");
/// Placeholder call used for `core::intrinsics::roundf32`.
pub const CALLEE_ROUND_F32: &str = placeholder!("roundf32");
/// Placeholder call used for `core::intrinsics::roundf64`.
pub const CALLEE_ROUND_F64: &str = placeholder!("roundf64");
/// Placeholder call used for `core::intrinsics::round_ties_even_f32`.
pub const CALLEE_ROUNDEVEN_F32: &str = placeholder!("round_ties_even_f32");
/// Placeholder call used for `core::intrinsics::round_ties_even_f64`.
pub const CALLEE_ROUNDEVEN_F64: &str = placeholder!("round_ties_even_f64");
/// Placeholder call used for generic `core::intrinsics::fabs`.
pub const CALLEE_FABS: &str = placeholder!("fabs");
/// Placeholder call used for `core::intrinsics::copysignf32`.
pub const CALLEE_COPYSIGN_F32: &str = placeholder!("copysignf32");
/// Placeholder call used for `core::intrinsics::copysignf64`.
pub const CALLEE_COPYSIGN_F64: &str = placeholder!("copysignf64");
/// Placeholder call used for `core::intrinsics::maximum_number_nsz_f32`
/// (the intrinsic backing `f32::max`).
pub const CALLEE_MAXNUM_NSZ_F32: &str = placeholder!("maximum_number_nsz_f32");
/// Placeholder call used for `core::intrinsics::maximum_number_nsz_f64`
/// (the intrinsic backing `f64::max`).
pub const CALLEE_MAXNUM_NSZ_F64: &str = placeholder!("maximum_number_nsz_f64");
/// Placeholder call used for `core::intrinsics::minimum_number_nsz_f32`
/// (the intrinsic backing `f32::min`).
pub const CALLEE_MINNUM_NSZ_F32: &str = placeholder!("minimum_number_nsz_f32");
/// Placeholder call used for `core::intrinsics::minimum_number_nsz_f64`
/// (the intrinsic backing `f64::min`).
pub const CALLEE_MINNUM_NSZ_F64: &str = placeholder!("minimum_number_nsz_f64");
/// Placeholder call used for `f32::asin` / `std::sys::cmath::asinf`.
pub const CALLEE_ASIN_F32: &str = placeholder!("asinf32");
/// Placeholder call used for `f64::asin` / `std::sys::cmath::asin`.
pub const CALLEE_ASIN_F64: &str = placeholder!("asinf64");
/// Placeholder call used for `f32::acos` / `std::sys::cmath::acosf`.
pub const CALLEE_ACOS_F32: &str = placeholder!("acosf32");
/// Placeholder call used for `f64::acos` / `std::sys::cmath::acos`.
pub const CALLEE_ACOS_F64: &str = placeholder!("acosf64");
/// Placeholder call used for `f32::atan2` / `std::sys::cmath::atan2f`.
pub const CALLEE_ATAN2_F32: &str = placeholder!("atan2f32");
/// Placeholder call used for `f64::atan2` / `std::sys::cmath::atan2`.
pub const CALLEE_ATAN2_F64: &str = placeholder!("atan2f64");
/// Placeholder call used for `f32::atan` / `std::sys::cmath::atanf`.
pub const CALLEE_ATAN_F32: &str = placeholder!("atanf32");
/// Placeholder call used for `f64::atan` / `std::sys::cmath::atan`.
pub const CALLEE_ATAN_F64: &str = placeholder!("atanf64");
/// Placeholder call used for `f32::cbrt` / `std::sys::cmath::cbrtf`.
pub const CALLEE_CBRT_F32: &str = placeholder!("cbrtf32");
/// Placeholder call used for `f64::cbrt` / `std::sys::cmath::cbrt`.
pub const CALLEE_CBRT_F64: &str = placeholder!("cbrtf64");

/// Placeholder call used for `f32::sinh` / `std::sys::cmath::sinhf`.
pub const CALLEE_SINH_F32: &str = placeholder!("sinhf32");
/// Placeholder call used for `f64::sinh` / `std::sys::cmath::sinh`.
pub const CALLEE_SINH_F64: &str = placeholder!("sinhf64");
/// Placeholder call used for `f32::cosh` / `std::sys::cmath::coshf`.
pub const CALLEE_COSH_F32: &str = placeholder!("coshf32");
/// Placeholder call used for `f64::cosh` / `std::sys::cmath::cosh`.
pub const CALLEE_COSH_F64: &str = placeholder!("coshf64");
/// Placeholder call used for `f32::tanh` / `std::sys::cmath::tanhf`.
pub const CALLEE_TANH_F32: &str = placeholder!("tanhf32");
/// Placeholder call used for `f64::tanh` / `std::sys::cmath::tanh`.
pub const CALLEE_TANH_F64: &str = placeholder!("tanhf64");
// Note: `f{32,64}::{asinh,acosh,atanh}` are pure-Rust formulas in `std`
// (compositions of `ln`/`sqrt`/`ln_1p`), not `std::sys::cmath` calls, so
// they need no placeholder of their own. `atanh` works once `ln_1p` (below)
// is intercepted.
/// Placeholder call used for `f32::exp_m1` / `std::sys::cmath::expm1f`.
pub const CALLEE_EXPM1_F32: &str = placeholder!("expm1f32");
/// Placeholder call used for `f64::exp_m1` / `std::sys::cmath::expm1`.
pub const CALLEE_EXPM1_F64: &str = placeholder!("expm1f64");
/// Placeholder call used for `f32::ln_1p` / `std::sys::cmath::log1pf`.
pub const CALLEE_LOG1P_F32: &str = placeholder!("log1pf32");
/// Placeholder call used for `f64::ln_1p` / `std::sys::cmath::log1p`.
pub const CALLEE_LOG1P_F64: &str = placeholder!("log1pf64");
/// Placeholder call used for `f32::hypot` / `std::sys::cmath::hypotf` (binary).
pub const CALLEE_HYPOT_F32: &str = placeholder!("hypotf32");
/// Placeholder call used for `f64::hypot` / `std::sys::cmath::hypot` (binary).
pub const CALLEE_HYPOT_F64: &str = placeholder!("hypotf64");

/// Placeholder call used for `core::intrinsics::fadd_fast` (generic over float type).
///
/// Lowered to `llvm.fadd` with explicit `fast` fast-math flags. The `f*_fast` intrinsics
/// assume finite, non-NaN inputs; LLVM's fast-math flags express the same
/// preconditions, so the binop replaces the call directly.
pub const CALLEE_FADD_FAST: &str = placeholder!("fadd_fast");
/// Placeholder call used for `core::intrinsics::fsub_fast` (generic over float type).
pub const CALLEE_FSUB_FAST: &str = placeholder!("fsub_fast");
/// Placeholder call used for `core::intrinsics::fmul_fast` (generic over float type).
pub const CALLEE_FMUL_FAST: &str = placeholder!("fmul_fast");
/// Placeholder call used for `core::intrinsics::fdiv_fast` (generic over float type).
pub const CALLEE_FDIV_FAST: &str = placeholder!("fdiv_fast");
/// Placeholder call used for `core::intrinsics::frem_fast` (generic over float type).
pub const CALLEE_FREM_FAST: &str = placeholder!("frem_fast");

/// Placeholder call used for `core::intrinsics::select_unpredictable`.
///
/// Backs `core::hint::select_unpredictable`, which libcore reaches
/// pervasively from branchless helpers (slice sorting, `Ord` combinators).
/// Lowers to an LLVM `select`; the "unpredictable" branch-weight hint has
/// no device semantics and is dropped.
pub const CALLEE_SELECT_UNPREDICTABLE: &str = placeholder!("select_unpredictable");

/// Return whether an internal placeholder lowers to a CUDA libdevice call.
///
/// This is deliberately an exact allow-list. Other placeholder families lower
/// to LLVM operations directly, including integer, saturating, bigint, and
/// `f*_fast` intrinsics, so matching the common placeholder prefix would select
/// the libNVVM backend for modules that do not need it.
pub fn is_libdevice_backed_placeholder(callee: &str) -> bool {
    matches!(
        callee,
        CALLEE_SQRT_F32
            | CALLEE_SQRT_F64
            | CALLEE_POWI_F32
            | CALLEE_POWI_F64
            | CALLEE_SIN_F32
            | CALLEE_SIN_F64
            | CALLEE_COS_F32
            | CALLEE_COS_F64
            | CALLEE_TAN_F32
            | CALLEE_TAN_F64
            | CALLEE_POWF_F32
            | CALLEE_POWF_F64
            | CALLEE_EXP_F32
            | CALLEE_EXP_F64
            | CALLEE_EXP2_F32
            | CALLEE_EXP2_F64
            | CALLEE_LOG_F32
            | CALLEE_LOG_F64
            | CALLEE_LOG2_F32
            | CALLEE_LOG2_F64
            | CALLEE_LOG10_F32
            | CALLEE_LOG10_F64
            | CALLEE_FMA_F32
            | CALLEE_FMA_F64
            | CALLEE_FMULADD_F32
            | CALLEE_FMULADD_F64
            | CALLEE_FLOOR_F32
            | CALLEE_FLOOR_F64
            | CALLEE_CEIL_F32
            | CALLEE_CEIL_F64
            | CALLEE_TRUNC_F32
            | CALLEE_TRUNC_F64
            | CALLEE_ROUND_F32
            | CALLEE_ROUND_F64
            | CALLEE_ROUNDEVEN_F32
            | CALLEE_ROUNDEVEN_F64
            | CALLEE_FABS
            | CALLEE_COPYSIGN_F32
            | CALLEE_COPYSIGN_F64
            | CALLEE_MAXNUM_NSZ_F32
            | CALLEE_MAXNUM_NSZ_F64
            | CALLEE_MINNUM_NSZ_F32
            | CALLEE_MINNUM_NSZ_F64
            | CALLEE_ASIN_F32
            | CALLEE_ASIN_F64
            | CALLEE_ACOS_F32
            | CALLEE_ACOS_F64
            | CALLEE_ATAN2_F32
            | CALLEE_ATAN2_F64
            | CALLEE_ATAN_F32
            | CALLEE_ATAN_F64
            | CALLEE_CBRT_F32
            | CALLEE_CBRT_F64
            | CALLEE_SINH_F32
            | CALLEE_SINH_F64
            | CALLEE_COSH_F32
            | CALLEE_COSH_F64
            | CALLEE_TANH_F32
            | CALLEE_TANH_F64
            | CALLEE_EXPM1_F32
            | CALLEE_EXPM1_F64
            | CALLEE_LOG1P_F32
            | CALLEE_LOG1P_F64
            | CALLEE_HYPOT_F32
            | CALLEE_HYPOT_F64
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn libdevice_placeholder_classification_is_exact() {
        for callee in [
            CALLEE_SQRT_F32,
            CALLEE_POWI_F64,
            CALLEE_SIN_F32,
            CALLEE_FMA_F64,
            CALLEE_FABS,
            CALLEE_MAXNUM_NSZ_F32,
            CALLEE_ASIN_F64,
            CALLEE_HYPOT_F32,
        ] {
            assert!(
                is_libdevice_backed_placeholder(callee),
                "expected `{callee}` to require libdevice"
            );
        }

        for callee in [
            CALLEE_ROTATE_LEFT,
            CALLEE_CTPOP,
            CALLEE_SATURATING_ADD,
            CALLEE_CARRYING_MUL_ADD,
            CALLEE_FADD_FAST,
            CALLEE_FSUB_FAST,
            CALLEE_FMUL_FAST,
            CALLEE_FDIV_FAST,
            CALLEE_FREM_FAST,
            "__cuda_oxide_rust_intrinsic_unknown",
            "__nv_sinf",
        ] {
            assert!(
                !is_libdevice_backed_placeholder(callee),
                "expected `{callee}` not to require libdevice"
            );
        }
    }
}
