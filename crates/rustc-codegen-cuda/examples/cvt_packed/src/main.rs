/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Packed cvt variants end-to-end example (sm_80+).
//!
//! Tests the four new packed conversion intrinsics added in PR #276 alongside
//! the baseline `cvt_f16x2_f32`, all on the same (lo=3.14, hi=-2.5) inputs:
//!
//!   - `cvt_f16x2_f32`          round-to-nearest f16 pack (baseline)
//!   - `cvt_rz_f16x2_f32`      round-toward-zero f16 pack
//!   - `cvt_rn_relu_f16x2_f32` round-to-nearest + ReLU f16 pack
//!   - `cvt_rn_relu_bf16x2_f32` round-to-nearest + ReLU bf16 pack
//!   - `cvt_rz_bf16x2_f32`     round-toward-zero bf16 pack
//!
//! The host unpacks each u32 result and verifies:
//!   - rn variants: lo ~ 3.14, hi ~ -2.5
//!   - rz variants: lo ~ 3.14 (truncated), hi ~ -2.5 (truncated)
//!   - relu variants: lo ~ 3.14, hi = 0.0 (negative clamped)
//!
//! Run: cargo oxide run cvt_packed

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::convert::{
    cvt_f16x2_f32, cvt_rn_relu_bf16x2_f32, cvt_rn_relu_f16x2_f32, cvt_rz_bf16x2_f32,
    cvt_rz_f16x2_f32,
};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

/// Number of conversion results produced by the kernel.
const NUM_VARIANTS: usize = 5;

// =============================================================================
// KERNEL
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Thread 0 calls all five conversion functions with the same (lo, hi) pair
    /// and writes the five packed u32 results into `out[0..5]`.
    #[kernel]
    pub fn cvt_packed_variants(lo: f32, hi: f32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if idx.get() == 0 {
            unsafe {
                *out.get_unchecked_mut(0) = cvt_f16x2_f32(lo, hi);
                *out.get_unchecked_mut(1) = cvt_rz_f16x2_f32(lo, hi);
                *out.get_unchecked_mut(2) = cvt_rn_relu_f16x2_f32(lo, hi);
                *out.get_unchecked_mut(3) = cvt_rn_relu_bf16x2_f32(lo, hi);
                *out.get_unchecked_mut(4) = cvt_rz_bf16x2_f32(lo, hi);
            }
        }
    }
}

// =============================================================================
// HOST HELPERS - half/bfloat16 unpacking
// =============================================================================

/// Unpack a u32 holding two IEEE 754 half-precision (f16) values.
fn unpack_f16x2(packed: u32) -> (f32, f32) {
    let lo_bits = (packed & 0xFFFF) as u16;
    let hi_bits = (packed >> 16) as u16;
    (f16_to_f32(lo_bits), f16_to_f32(hi_bits))
}

/// Convert an IEEE 754 half-precision bit pattern to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Denormalized: shift mantissa until the implicit 1 appears.
        let mut m = mant;
        let mut e = 0i32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        m &= 0x3FF;
        f32::from_bits((sign << 31) | (((127 - 15 + 1 - e as u32) & 0xFF) << 23) | (m << 13))
    } else if exp == 31 {
        // Inf / NaN
        f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13))
    } else {
        // Normalized
        f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
    }
}

/// Unpack a u32 holding two bfloat16 values.
fn unpack_bf16x2(packed: u32) -> (f32, f32) {
    let lo = f32::from_bits((packed & 0xFFFF) as u32 << 16);
    let hi = f32::from_bits((packed >> 16) << 16);
    (lo, hi)
}

// =============================================================================
// HOST VERIFICATION
// =============================================================================

fn main() {
    println!("=== Packed cvt Variants (sm_80+) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let (major, minor) = ctx.compute_capability().expect("compute capability");
    println!("GPU Compute Capability: sm_{}{}", major, minor);

    // The packed cvt.rz and cvt.rn.relu variants require sm_80+ (Ampere).
    if major < 8 {
        println!("\nskipping: packed cvt variants require sm_80+ (Ampere)");
        println!("  this GPU is sm_{}{}", major, minor);
        return;
    }

    let module = ctx
        .load_module_from_file("cvt_packed.ptx")
        .expect("Failed to load PTX module");
    let module = kernels::from_module(module).expect("Failed to initialize typed CUDA module");

    let lo: f32 = 3.14;
    let hi: f32 = -2.5;

    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, NUM_VARIANTS).unwrap();
    let cfg = LaunchConfig::for_num_elems(1);

    module
        .cvt_packed_variants(&stream, cfg, lo, hi, &mut out_dev)
        .expect("Kernel launch failed");

    let results = out_dev.to_host_vec(&stream).unwrap();
    assert_eq!(results.len(), NUM_VARIANTS);

    let mut failures = 0;

    // f16 tolerance: ~3 decimal digits, use 0.002 for rn and 0.004 for rz.
    let f16_tol = 0.002;
    let f16_rz_tol = 0.004;
    // bf16 tolerance: ~2 decimal digits, use 0.02 for rn and 0.04 for rz.
    let bf16_tol = 0.02;
    let bf16_rz_tol = 0.04;

    // --- 0: cvt_f16x2_f32 (round-to-nearest) ---
    {
        let (got_lo, got_hi) = unpack_f16x2(results[0]);
        println!("[0] cvt_f16x2_f32:           lo={got_lo:.6}, hi={got_hi:.6}  (packed={:#010x})", results[0]);
        if (got_lo - lo).abs() > f16_tol {
            eprintln!("    FAIL: lo expected ~{lo}, got {got_lo}");
            failures += 1;
        }
        if (got_hi - hi).abs() > f16_tol {
            eprintln!("    FAIL: hi expected ~{hi}, got {got_hi}");
            failures += 1;
        }
    }

    // --- 1: cvt_rz_f16x2_f32 (round-toward-zero) ---
    {
        let (got_lo, got_hi) = unpack_f16x2(results[1]);
        println!("[1] cvt_rz_f16x2_f32:        lo={got_lo:.6}, hi={got_hi:.6}  (packed={:#010x})", results[1]);
        if (got_lo - lo).abs() > f16_rz_tol {
            eprintln!("    FAIL: lo expected ~{lo}, got {got_lo}");
            failures += 1;
        }
        if (got_hi - hi).abs() > f16_rz_tol {
            eprintln!("    FAIL: hi expected ~{hi}, got {got_hi}");
            failures += 1;
        }
        // rz truncates toward zero, so |got| <= |exact|
        if got_lo.abs() > lo.abs() + f16_rz_tol {
            eprintln!("    FAIL: rz lo magnitude should be <= input magnitude");
            failures += 1;
        }
        if got_hi.abs() > hi.abs() + f16_rz_tol {
            eprintln!("    FAIL: rz hi magnitude should be <= input magnitude");
            failures += 1;
        }
    }

    // --- 2: cvt_rn_relu_f16x2_f32 (round-to-nearest + ReLU) ---
    {
        let (got_lo, got_hi) = unpack_f16x2(results[2]);
        println!("[2] cvt_rn_relu_f16x2_f32:   lo={got_lo:.6}, hi={got_hi:.6}  (packed={:#010x})", results[2]);
        if (got_lo - lo).abs() > f16_tol {
            eprintln!("    FAIL: lo expected ~{lo}, got {got_lo}");
            failures += 1;
        }
        // hi was -2.5, ReLU should clamp to 0.0
        if got_hi != 0.0 {
            eprintln!("    FAIL: hi expected 0.0 (ReLU clamp), got {got_hi}");
            failures += 1;
        }
    }

    // --- 3: cvt_rn_relu_bf16x2_f32 (round-to-nearest + ReLU, bf16) ---
    {
        let (got_lo, got_hi) = unpack_bf16x2(results[3]);
        println!("[3] cvt_rn_relu_bf16x2_f32:  lo={got_lo:.6}, hi={got_hi:.6}  (packed={:#010x})", results[3]);
        if (got_lo - lo).abs() > bf16_tol {
            eprintln!("    FAIL: lo expected ~{lo}, got {got_lo}");
            failures += 1;
        }
        // hi was -2.5, ReLU should clamp to 0.0
        if got_hi != 0.0 {
            eprintln!("    FAIL: hi expected 0.0 (ReLU clamp), got {got_hi}");
            failures += 1;
        }
    }

    // --- 4: cvt_rz_bf16x2_f32 (round-toward-zero, bf16) ---
    {
        let (got_lo, got_hi) = unpack_bf16x2(results[4]);
        println!("[4] cvt_rz_bf16x2_f32:       lo={got_lo:.6}, hi={got_hi:.6}  (packed={:#010x})", results[4]);
        if (got_lo - lo).abs() > bf16_rz_tol {
            eprintln!("    FAIL: lo expected ~{lo}, got {got_lo}");
            failures += 1;
        }
        if (got_hi - hi).abs() > bf16_rz_tol {
            eprintln!("    FAIL: hi expected ~{hi}, got {got_hi}");
            failures += 1;
        }
        // rz truncates toward zero, so |got| <= |exact|
        if got_lo.abs() > lo.abs() + bf16_rz_tol {
            eprintln!("    FAIL: rz lo magnitude should be <= input magnitude");
            failures += 1;
        }
        if got_hi.abs() > hi.abs() + bf16_rz_tol {
            eprintln!("    FAIL: rz hi magnitude should be <= input magnitude");
            failures += 1;
        }
    }

    // --- Summary ---
    println!();
    if failures == 0 {
        println!(
            "SUCCESS: all {} packed cvt variants produced correct results",
            NUM_VARIANTS
        );
    } else {
        eprintln!("{failures} check(s) failed");
        std::process::exit(1);
    }
}
