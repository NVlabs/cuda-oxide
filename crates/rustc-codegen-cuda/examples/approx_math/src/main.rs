/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Approximate Math via `ptx_asm!`
//!
//! Demonstrates how to use single-instruction PTX approximate math operations
//! through the `ptx_asm!` macro. These bypass libdevice entirely and map to
//! dedicated hardware units, trading precision for throughput.
//!
//! ## Instructions demonstrated
//!
//! | Wrapper              | PTX instruction            | Precision     | SM   |
//! |----------------------|----------------------------|---------------|------|
//! | `tanh_approx`        | `tanh.approx.f32`          | ~2^-8 ULP     | 75+  |
//! | `ex2_approx`         | `ex2.approx.ftz.f32`       | ~2^-8 ULP     | all  |
//! | `rcp_approx`         | `rcp.approx.ftz.f32`       | ~2^-23.1 ULP  | all  |
//! | `lg2_approx`         | `lg2.approx.ftz.f32`       | ~2^-8 ULP     | all  |
//!
//! The example also shows how to compose these into a fast sigmoid:
//! `sigmoid(x) = 0.5 * tanh(x * 0.5) + 0.5`
//!
//! Build and run with:
//!   cargo oxide run approx_math

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, ptx_asm, thread};
use cuda_host::cuda_module;

// =============================================================================
// KERNELS
// =============================================================================

#[cuda_module]
mod kernels {
    use super::*;

    /// Applies `tanh.approx.f32` element-wise.
    ///
    /// Requires sm_75+ (Turing and later).
    #[kernel]
    pub fn tanh_approx_kernel(input: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = output.get_mut(idx) {
            let x = input[i];
            let result: f32;
            unsafe {
                ptx_asm!(
                    "tanh.approx.f32 %0, %1;",
                    out("=f") result,
                    in("f") x,
                    options(register_only),
                );
            }
            *slot = result;
        }
    }

    /// Applies `ex2.approx.ftz.f32` element-wise (computes 2^x).
    ///
    /// Available on all SM architectures. Subnormal inputs flush to zero.
    #[kernel]
    pub fn ex2_approx_kernel(input: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = output.get_mut(idx) {
            let x = input[i];
            let result: f32;
            unsafe {
                ptx_asm!(
                    "ex2.approx.ftz.f32 %0, %1;",
                    out("=f") result,
                    in("f") x,
                    options(register_only),
                );
            }
            *slot = result;
        }
    }

    /// Computes fast sigmoid via `tanh.approx.f32`:
    ///   sigmoid(x) = 0.5 * tanh(x * 0.5) + 0.5
    ///
    /// This is a common ML pattern that avoids the `exp` + `rcp` sequence
    /// used by libdevice's sigmoid, reducing to a single `tanh.approx.f32`
    /// plus two FMAs.
    ///
    /// Requires sm_75+ (Turing and later).
    #[kernel]
    pub fn fast_sigmoid_kernel(input: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = output.get_mut(idx) {
            let x = input[i];
            let half_x = x * 0.5;
            let tanh_val: f32;
            unsafe {
                ptx_asm!(
                    "tanh.approx.f32 %0, %1;",
                    out("=f") tanh_val,
                    in("f") half_x,
                    options(register_only),
                );
            }
            *slot = 0.5 * tanh_val + 0.5;
        }
    }

    /// Applies `rcp.approx.ftz.f32` element-wise (computes 1/x).
    ///
    /// Available on all SM architectures.
    #[kernel]
    pub fn rcp_approx_kernel(input: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = output.get_mut(idx) {
            let x = input[i];
            let result: f32;
            unsafe {
                ptx_asm!(
                    "rcp.approx.ftz.f32 %0, %1;",
                    out("=f") result,
                    in("f") x,
                    options(register_only),
                );
            }
            *slot = result;
        }
    }

    /// Applies `lg2.approx.ftz.f32` element-wise (computes log2(x)).
    ///
    /// Available on all SM architectures.
    #[kernel]
    pub fn lg2_approx_kernel(input: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = output.get_mut(idx) {
            let x = input[i];
            let result: f32;
            unsafe {
                ptx_asm!(
                    "lg2.approx.ftz.f32 %0, %1;",
                    out("=f") result,
                    in("f") x,
                    options(register_only),
                );
            }
            *slot = result;
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    println!("=== Approximate Math via ptx_asm! ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let (major, minor) = ctx.compute_capability().expect("compute capability");

    println!("GPU Compute Capability: sm_{major}{minor}");

    const N: usize = 256;
    let config = LaunchConfig::for_num_elems(N as u32);

    // Test inputs: values in [-3.0, 3.0] to exercise the interesting range.
    let input: Vec<f32> = (0..N)
        .map(|i| -3.0 + 6.0 * (i as f32) / (N as f32 - 1.0))
        .collect();

    let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();

    let module = ctx
        .load_module_from_file("approx_math.ptx")
        .expect("Failed to load PTX module");
    let module = kernels::from_module(module).expect("Failed to initialize typed CUDA module");

    // ---- tanh.approx.f32 ----
    // tanh.approx requires sm_75+.
    let sm_num = major * 10 + minor;
    if sm_num >= 75 {
        print!("--- tanh.approx.f32 --- ");
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
        // SAFETY: launch shape covers the buffer; input/output buffers are correctly sized.
        unsafe { module.tanh_approx_kernel(&stream, config, &input_dev, &mut out_dev) }
            .expect("tanh_approx_kernel failed");
        let out = out_dev.to_host_vec(&stream).unwrap();
        let max_err = check_results(&input, &out, |x| x.tanh());
        println!("max |err| = {max_err:.2e}");
        assert!(max_err < 0.01, "tanh.approx error too large: {max_err}");

        // ---- fast sigmoid ----
        print!("--- fast sigmoid     --- ");
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
        // SAFETY: launch shape covers the buffer; input/output buffers are correctly sized.
        unsafe { module.fast_sigmoid_kernel(&stream, config, &input_dev, &mut out_dev) }
            .expect("fast_sigmoid_kernel failed");
        let out = out_dev.to_host_vec(&stream).unwrap();
        let max_err = check_results(&input, &out, |x| 1.0 / (1.0 + (-x).exp()));
        println!("max |err| = {max_err:.2e}");
        assert!(max_err < 0.01, "fast sigmoid error too large: {max_err}");
    } else {
        println!(
            "Skipping tanh.approx.f32 and fast sigmoid (requires sm_75+, have sm_{})",
            sm_num
        );
    }

    // ---- ex2.approx.ftz.f32 ----
    print!("--- ex2.approx.f32   --- ");
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape covers the buffer; input/output buffers are correctly sized.
    unsafe { module.ex2_approx_kernel(&stream, config, &input_dev, &mut out_dev) }
        .expect("ex2_approx_kernel failed");
    let out = out_dev.to_host_vec(&stream).unwrap();
    let max_err = check_results(&input, &out, |x| (2.0_f32).powf(x));
    println!("max |err| = {max_err:.2e}");
    assert!(max_err < 0.01, "ex2.approx error too large: {max_err}");

    // ---- rcp.approx.ftz.f32 ----
    // Use positive inputs for reciprocal to avoid division edge cases.
    let rcp_input: Vec<f32> = (0..N)
        .map(|i| 0.5 + 3.0 * (i as f32) / (N as f32 - 1.0))
        .collect();
    let rcp_input_dev = DeviceBuffer::from_host(&stream, &rcp_input).unwrap();
    print!("--- rcp.approx.f32   --- ");
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape covers the buffer; input/output buffers are correctly sized.
    unsafe { module.rcp_approx_kernel(&stream, config, &rcp_input_dev, &mut out_dev) }
        .expect("rcp_approx_kernel failed");
    let out = out_dev.to_host_vec(&stream).unwrap();
    let max_err = check_results(&rcp_input, &out, |x| 1.0 / x);
    println!("max |err| = {max_err:.2e}");
    assert!(max_err < 0.001, "rcp.approx error too large: {max_err}");

    // ---- lg2.approx.ftz.f32 ----
    // Use positive inputs for log2.
    let lg2_input: Vec<f32> = (0..N)
        .map(|i| 0.1 + 10.0 * (i as f32) / (N as f32 - 1.0))
        .collect();
    let lg2_input_dev = DeviceBuffer::from_host(&stream, &lg2_input).unwrap();
    print!("--- lg2.approx.f32   --- ");
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape covers the buffer; input/output buffers are correctly sized.
    unsafe { module.lg2_approx_kernel(&stream, config, &lg2_input_dev, &mut out_dev) }
        .expect("lg2_approx_kernel failed");
    let out = out_dev.to_host_vec(&stream).unwrap();
    let max_err = check_results(&lg2_input, &out, |x| x.log2());
    println!("max |err| = {max_err:.2e}");
    assert!(max_err < 0.01, "lg2.approx error too large: {max_err}");

    println!("\nSUCCESS: all approximate math results within tolerance");
}

/// Returns the maximum absolute error between GPU output and reference.
fn check_results(input: &[f32], output: &[f32], reference: impl Fn(f32) -> f32) -> f32 {
    input
        .iter()
        .zip(output)
        .map(|(&x, &got)| (got - reference(x)).abs())
        .fold(0.0_f32, f32::max)
}
