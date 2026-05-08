/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke test for `cuda_device::math::rsqrt_f32`.
//!
//! Issues one libdevice `__nv_rsqrtf` call per thread (compiles to a single
//! `rsqrt.approx.f32` PTX instruction) and verifies the result matches a CPU
//! reference within rsqrt's documented relative error.
//!
//! Build and run with:
//!   cargo oxide run rsqrt_smoke

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::math::rsqrt_f32;
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::{cuda_launch, load_kernel_module};

#[kernel]
pub fn rsqrt_kernel(x: &[f32], mut y: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(y_elem) = y.get_mut(idx) {
        *y_elem = rsqrt_f32(x[idx.get()]);
    }
}

fn main() {
    println!("=== rsqrt smoke test ===");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    // Inputs spanning a few orders of magnitude — none equal to zero.
    const N: u32 = 1024;
    let x_host: Vec<f32> = (1..=N).map(|i| i as f32 * 0.5).collect();

    let x_dev = DeviceBuffer::from_host(&stream, &x_host).unwrap();
    let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, N as usize).unwrap();

    // libdevice-aware loader: kernels that call __nv_* (e.g. rsqrt_f32) need
    // libdevice linked, which load_kernel_module handles transparently.
    let module = load_kernel_module(&ctx, "rsqrt_smoke").expect("Failed to load module");

    cuda_launch! {
        kernel: rsqrt_kernel,
        stream: stream,
        module: module,
        config: LaunchConfig::for_num_elems(N),
        args: [slice(x_dev), slice_mut(y_dev)]
    }
    .unwrap();

    let y = y_dev.to_host_vec(&stream).unwrap();

    // libdevice rsqrtf is documented to produce results within a few ULPs of
    // 1.0 / sqrtf(x). Use a relative tolerance generous enough for the
    // approximate variant on Blackwell.
    const RTOL: f32 = 1e-5;
    let mut max_rel_err: f32 = 0.0;
    for (i, &xi) in x_host.iter().enumerate() {
        let expected = 1.0 / xi.sqrt();
        let got = y[i];
        let rel_err = ((got - expected) / expected).abs();
        if rel_err > max_rel_err {
            max_rel_err = rel_err;
        }
    }

    println!("first 4 inputs:  {:?}", &x_host[0..4]);
    println!("first 4 outputs: {:?}", &y[0..4]);
    println!("max relative error vs 1.0 / sqrt(x): {max_rel_err:.2e}");

    if max_rel_err < RTOL {
        println!("\n✓ SUCCESS");
    } else {
        panic!(
            "rsqrt output exceeded relative tolerance {RTOL} (got {max_rel_err:.2e})"
        );
    }
}
