/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke test for floating-point special-value literals (`NAN`, `INFINITY`,
//! `NEG_INFINITY`) inside kernels.
//!
//! Before the matching fix to `format_float_literal` in
//! `crates/dialect-llvm/src/export.rs`, embedding `f32::NAN` (or the f64
//! form) in a kernel as a float SSA value (e.g. by storing it through a
//! `*mut f32` write) produced an LLVM IR fragment like
//!
//! ```text
//!   store float nan, ptr %v9
//! ```
//!
//! which libNVVM rejects with
//!
//! ```text
//! Nvvm(Call { operation: "nvvmCompileProgram", code: 9,
//!   log: Some("...ll (..): parse expected value token") })
//! ```
//!
//! The bare token `nan` is not part of the LLVM IR float-constant grammar.
//! After the fix, NaN renders as `0x7FF8000000000000` (a canonical quiet
//! NaN, 16-hex-digit double bit pattern) and the same kernel reaches a
//! cubin through the standard `llc → PTX` path.
//!
//! The kernels intentionally store the float values directly through a
//! `DisjointSlice<f32>` / `DisjointSlice<f64>` write so the constants
//! survive into LLVM IR as `float` / `double` SSA values rather than
//! getting folded into integer `to_bits()` constants by rustc.
//!
//! Build and run with:
//!   cargo oxide run fp_special_literal_smoke

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::{cuda_launch, ltoir};

// =============================================================================
// KERNELS
// =============================================================================

/// Stores `f32::NAN`, `f32::INFINITY`, `f32::NEG_INFINITY`, and a finite
/// control value through a `DisjointSlice<f32>` so the constants reach
/// `format_float_literal` as `float` SSA values.
#[kernel]
pub fn fp_special_literal_f32_kernel(mut out: DisjointSlice<f32>) {
    if thread::index_1d().get() == 0 {
        unsafe {
            *out.get_unchecked_mut(0) = f32::NAN;
            *out.get_unchecked_mut(1) = f32::INFINITY;
            *out.get_unchecked_mut(2) = f32::NEG_INFINITY;
            *out.get_unchecked_mut(3) = 1.5_f32;
        }
    }
}

/// Same as `fp_special_literal_f32_kernel` for `f64`.
#[kernel]
pub fn fp_special_literal_f64_kernel(mut out: DisjointSlice<f64>) {
    if thread::index_1d().get() == 0 {
        unsafe {
            *out.get_unchecked_mut(0) = f64::NAN;
            *out.get_unchecked_mut(1) = f64::INFINITY;
            *out.get_unchecked_mut(2) = f64::NEG_INFINITY;
            *out.get_unchecked_mut(3) = 1.5_f64;
        }
    }
}

// =============================================================================
// HOST
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== fp special literal smoke ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // Even though this kernel does not use libdevice math, going through
    // `ltoir::load_kernel_module` is the most portable loader for examples.
    // It falls back to direct `.ptx` loading when no `__nv_*` call is
    // present, so this example exercises the standard `llc` path rather
    // than libNVVM. The fix being smoked is in
    // `dialect-llvm::format_float_literal`, which runs upstream of both
    // backends, so either loader path validates it.
    let module = ltoir::load_kernel_module(&ctx, "fp_special_literal_smoke")?;
    let cfg = LaunchConfig::for_num_elems(1);

    let mut passed = 0u32;
    let mut failed = 0u32;

    // ---------- f32 ----------
    {
        let mut out = DeviceBuffer::<f32>::zeroed(&stream, 4)?;
        cuda_launch! {
            kernel: fp_special_literal_f32_kernel,
            stream: stream, module: module, config: cfg,
            args: [slice_mut(out)]
        }?;
        let got = out.to_host_vec(&stream)?;
        check_f32_specials(&got, &mut passed, &mut failed);
    }

    // ---------- f64 ----------
    {
        let mut out = DeviceBuffer::<f64>::zeroed(&stream, 4)?;
        cuda_launch! {
            kernel: fp_special_literal_f64_kernel,
            stream: stream, module: module, config: cfg,
            args: [slice_mut(out)]
        }?;
        let got = out.to_host_vec(&stream)?;
        check_f64_specials(&got, &mut passed, &mut failed);
    }

    println!("\n--- summary ---");
    println!("passed: {passed}");
    println!("failed: {failed}");
    if failed != 0 {
        std::process::exit(1);
    }
    println!("\n✓ SUCCESS");
    Ok(())
}

fn check_f32_specials(got: &[f32], passed: &mut u32, failed: &mut u32) {
    let mut ok = true;
    if !got[0].is_nan() {
        println!(
            "FAIL f32::NAN: got {} (bits 0x{:08x})",
            got[0],
            got[0].to_bits()
        );
        ok = false;
    }
    if got[1] != f32::INFINITY {
        println!(
            "FAIL f32::INFINITY: got {} (bits 0x{:08x})",
            got[1],
            got[1].to_bits()
        );
        ok = false;
    }
    if got[2] != f32::NEG_INFINITY {
        println!(
            "FAIL f32::NEG_INFINITY: got {} (bits 0x{:08x})",
            got[2],
            got[2].to_bits()
        );
        ok = false;
    }
    if got[3] != 1.5_f32 {
        println!("FAIL f32 control 1.5: got {}", got[3]);
        ok = false;
    }
    if ok {
        println!(
            "PASS f32 specials: NaN={}, +Inf={}, -Inf={}, control={}",
            got[0].is_nan(),
            got[1],
            got[2],
            got[3]
        );
        *passed += 1;
    } else {
        *failed += 1;
    }
}

fn check_f64_specials(got: &[f64], passed: &mut u32, failed: &mut u32) {
    let mut ok = true;
    if !got[0].is_nan() {
        println!(
            "FAIL f64::NAN: got {} (bits 0x{:016x})",
            got[0],
            got[0].to_bits()
        );
        ok = false;
    }
    if got[1] != f64::INFINITY {
        println!(
            "FAIL f64::INFINITY: got {} (bits 0x{:016x})",
            got[1],
            got[1].to_bits()
        );
        ok = false;
    }
    if got[2] != f64::NEG_INFINITY {
        println!(
            "FAIL f64::NEG_INFINITY: got {} (bits 0x{:016x})",
            got[2],
            got[2].to_bits()
        );
        ok = false;
    }
    if got[3] != 1.5_f64 {
        println!("FAIL f64 control 1.5: got {}", got[3]);
        ok = false;
    }
    if ok {
        println!(
            "PASS f64 specials: NaN={}, +Inf={}, -Inf={}, control={}",
            got[0].is_nan(),
            got[1],
            got[2],
            got[3]
        );
        *passed += 1;
    } else {
        *failed += 1;
    }
}
