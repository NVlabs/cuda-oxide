/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Standalone Device Function Example
//!
//! Tests that #[device] functions can be compiled to PTX/LTOIR without requiring
//! a #[kernel] entry point. This enables building Rust device libraries that can
//! be consumed by CUDA C++ via LTOIR linking.
//!
//! Build with:
//!   cargo oxide run standalone_device_fn
//!
//! ## What This Tests
//!
//! 1. Simple standalone device functions (no kernel, no GPU intrinsics)
//! 2. Device function calling another device function (transitive collection)
//! 3. Generic logic via concrete device function wrappers
//! 4. Device function using GPU intrinsics (thread indexing)
//! 5. Multiple monomorphizations of the same generic
//! 6. Typed device stub through MIR import and lowering to exact PTX

use cuda_device::device;
use cuda_device::thread;
use cuda_device::wmma;

// =============================================================================
// TEST 1: Simple standalone device functions
//
// Pure math functions with no GPU intrinsics and no inter-function calls.
// These are the simplest case — each is an independent compilation root.
// =============================================================================

/// Newton's method square root — a pure math function.
#[device]
pub fn fast_sqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    // Newton's method: 5 iterations
    let mut guess = x * 0.5;
    guess = 0.5 * (guess + x / guess);
    guess = 0.5 * (guess + x / guess);
    guess = 0.5 * (guess + x / guess);
    guess = 0.5 * (guess + x / guess);
    guess = 0.5 * (guess + x / guess);
    guess
}

/// Simple clamp function.
#[device]
pub fn clamp_f32(val: f32, min_val: f32, max_val: f32) -> f32 {
    if val < min_val {
        min_val
    } else if val > max_val {
        max_val
    } else {
        val
    }
}

// =============================================================================
// TEST 2: Device function calling another device function
//
// Tests transitive collection: `safe_sqrt` calls `fast_sqrt` and `clamp_f32`.
// The BFS walk from `safe_sqrt` as a root must discover both callees.
// =============================================================================

/// Clamped square root — calls fast_sqrt and clamp_f32.
#[device]
pub fn safe_sqrt(x: f32) -> f32 {
    let clamped = clamp_f32(x, 0.0, 1e10);
    fast_sqrt(clamped)
}

// =============================================================================
// TEST 3: Generic device functions
//
// #[device] now supports generics, mirroring how #[kernel] handles them:
//   - No #[no_mangle] (generics use mangled symbol names)
//   - #[inline(never)] on the prefixed function (so monomorphizations appear in CGUs)
//   - Wrapper forwards type params via turbofish
//
// Monomorphization is triggered by concrete call sites (other #[device] fns
// or host code). For C++ LTOIR consumption, concrete wrappers provide stable
// extern "C" symbols — same pattern as CCCL/MathDx C++ template wrappers.
// =============================================================================

/// Generic fused multiply-add — directly annotated with #[device].
/// Monomorphized when called with concrete types below.
#[device]
pub fn fma<T: core::ops::Mul<Output = T> + core::ops::Add<Output = T>>(a: T, b: T, c: T) -> T {
    a * b + c
}

/// Concrete f32 wrapper — calls the generic fma::<f32>.
/// Provides a stable #[no_mangle] symbol for C++ LTOIR consumption.
#[device]
pub fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    fma(a, b, c)
}

/// Concrete i32 wrapper — calls the generic fma::<i32>.
/// Tests that multiple monomorphizations of the same generic are collected.
#[device]
pub fn fma_i32(a: i32, b: i32, c: i32) -> i32 {
    fma(a, b, c)
}

// =============================================================================
// TEST 3b: Uninstantiated generic device function
//
// This generic #[device] function has NO concrete callers. Without a call site
// that provides concrete types, rustc won't monomorphize it, so it should NOT
// appear in the PTX. The collector's is_fully_monomorphized check skips it.
// =============================================================================

/// Generic lerp — never called with concrete types.
/// Should NOT appear in PTX (no monomorphization = no CGU entry).
#[device]
pub fn lerp<T: core::ops::Mul<Output = T> + core::ops::Add<Output = T> + Copy>(
    a: T,
    b: T,
    t: T,
) -> T {
    // a + t * (b - a) — but we'd need Sub, so just do a * t + b for simplicity
    a * t + b
}

// =============================================================================
// TEST 4: Device function using GPU intrinsics
//
// Tests that GPU-specific intrinsics (thread indexing) work in standalone
// device functions. The thread::index_1d() intrinsic maps to NVVM's
// tid.x / ctaid.x / ntid.x under the hood.
// =============================================================================

/// Returns the raw 1D global thread index.
/// A thin wrapper around the GPU intrinsic — usable from CUDA C++ via LTOIR.
#[device]
pub fn get_global_thread_id() -> usize {
    thread::index_1d().get()
}

// =============================================================================
// TEST 5: Device stub through MIR import and PTX lowering
//
// The raw INT8 MMA intrinsic is never executed by this host-only example. Its
// device wrapper gives the compiler a checked-in end-to-end path from the
// cuda-device stub through mir-importer and mir-lower to generated PTX.
// =============================================================================

/// Apply one lane's register fragments for warp-wide INT8 MMA.
///
/// # Safety
///
/// The caller must uphold [`wmma::mma_m16n8k32_s32_s8`]'s warp participation
/// and per-lane fragment-layout contract.
#[device]
pub unsafe fn int8_mma_registers(
    c: [i32; 4],
    a: [u32; 4],
    b: [u32; 2],
) -> [i32; 4] {
    unsafe { wmma::mma_m16n8k32_s32_s8(c, a, b) }
}

// =============================================================================
// HOST CODE - Verifies PTX was generated correctly
// =============================================================================

fn main() {
    println!("=== Standalone Device Function Example ===\n");

    let ptx_path = "standalone_device_fn.ptx";

    if !std::path::Path::new(ptx_path).exists() {
        eprintln!("PTX file not found: {}", ptx_path);
        eprintln!(
            "This is expected if standalone device function compilation is not yet supported."
        );
        std::process::exit(1);
    }

    let ptx_content = std::fs::read_to_string(ptx_path).expect("Failed to read PTX file");
    println!("PTX file: {} ({} bytes)\n", ptx_path, ptx_content.len());

    // Define all expected functions and their test categories
    let tests: Vec<(&str, &str)> = vec![
        // (function_name, test_description)
        ("fast_sqrt", "Test 1: simple standalone fn"),
        ("clamp_f32", "Test 1: simple standalone fn"),
        ("safe_sqrt", "Test 2: device fn calling device fn"),
        ("fma_f32", "Test 3: generic instantiation (f32)"),
        ("fma_i32", "Test 3: generic instantiation (i32)"),
        (
            "get_global_thread_id",
            "Test 4: device fn with GPU intrinsics",
        ),
        (
            "int8_mma_registers",
            "Test 5: cuda-device stub imported and lowered",
        ),
    ];

    let mut passed = 0;
    let mut failed = 0;

    for (func_name, description) in &tests {
        if ptx_content.contains(func_name) {
            println!("  PASS  {} — {}", func_name, description);
            passed += 1;
        } else {
            println!("  FAIL  {} — {}", func_name, description);
            failed += 1;
        }
    }

    // Test 3b: Verify uninstantiated generic does NOT appear in PTX
    // "lerp" is generic with no concrete callers, so rustc won't monomorphize it.
    if !ptx_content.contains("lerp") {
        println!("  PASS  lerp absent — Test 3b: uninstantiated generic not compiled");
        passed += 1;
    } else {
        println!("  FAIL  lerp present — Test 3b: uninstantiated generic should not be in PTX");
        failed += 1;
    }

    // Verify no .entry directives (these are device functions, not kernels)
    println!();
    let has_entry = ptx_content.contains(".entry");
    if !has_entry {
        println!("  PASS  No .entry directives (all are .func)");
        passed += 1;
    } else {
        println!("  FAIL  Found .entry directive (unexpected for standalone device functions)");
        failed += 1;
    }

    let int8_mma = "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32";
    if ptx_content.contains(int8_mma) {
        println!("  PASS  INT8 MMA lowered to the exact PTX instruction");
        passed += 1;
    } else {
        println!("  FAIL  Missing exact INT8 MMA PTX instruction");
        failed += 1;
    }

    let ptx_version = ptx_content
        .lines()
        .find_map(|line| line.trim().strip_prefix(".version "))
        .and_then(|version| version.split_once('.'))
        .and_then(|(major, minor)| Some((major.parse::<u32>().ok()?, minor.parse::<u32>().ok()?)));
    if ptx_version.is_some_and(|version| version >= (7, 0)) {
        println!("  PASS  INT8 MMA selected PTX 7.0 or newer");
        passed += 1;
    } else {
        println!("  FAIL  Generated PTX did not meet the PTX 7.0 floor");
        failed += 1;
    }

    let sm_target = ptx_content
        .lines()
        .find_map(|line| line.trim().strip_prefix(".target sm_"))
        .map(|target| target.chars().take_while(char::is_ascii_digit).collect::<String>())
        .and_then(|target| target.parse::<u32>().ok());
    if sm_target.is_some_and(|target| target >= 80) {
        println!("  PASS  INT8 MMA selected sm_80 or newer");
        passed += 1;
    } else {
        println!("  FAIL  Generated PTX did not meet the sm_80 floor");
        failed += 1;
    }

    println!();
    if failed == 0 {
        let total = tests.len() + 5;
        println!(
            "SUCCESS: {}/{} tests passed — all device functions compiled to PTX!",
            passed, total
        );
    } else {
        println!("FAILED: {} test(s) failed, {} passed", failed, passed);
        std::process::exit(1);
    }
}
